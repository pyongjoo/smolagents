#!/usr/bin/env python
# coding=utf-8

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Scheduler for parallel task execution.

Keeps a process (or thread) pool, submits ready tasks, and collects
their results. This is the only module that knows about concurrency —
the planner and task graph are both purely sequential.

Design doc §9. Failure handling is explicitly deferred (§9.4): any
worker exception is surfaced directly via :class:`SchedulerFailure`
and the outer agent stops submitting new work.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from smolagents.parallel.events import (
    TaskCompletedEvent,
    TaskScheduledEvent,
    TaskStartedEvent,
)
from smolagents.parallel.task_graph import Task, TaskGraph


if TYPE_CHECKING:
    from smolagents.parallel.events import ParallelEvent


__all__ = [
    "SchedulerFailure",
    "TaskResult",
    "ParallelScheduler",
    "WorkerCallable",
]


# Workers must be picklable when using a process pool; they are
# expected to be module-level functions that accept a single
# picklable spec object and return a :class:`TaskResult`.
WorkerCallable = Callable[[Any], "TaskResult"]


@dataclass
class TaskResult:
    """Return value produced by a worker on successful task completion."""

    task_id: str
    output: Any
    action_steps: list[Any]  # list of dicts, one per inner ActionStep
    logs: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


class SchedulerFailure(RuntimeError):
    """Raised when a task's worker raised or crashed.

    Carries the originating task id and the wrapped exception so the
    outer agent can produce a useful :class:`AgentExecutionError`.
    """

    def __init__(self, task_id: str, cause: BaseException) -> None:
        super().__init__(f"Task {task_id!r} failed: {cause!r}")
        self.task_id = task_id
        self.cause = cause


class ParallelScheduler:
    """Dispatches ready tasks to a pool and yields lifecycle events.

    The scheduler is stateful: it owns both the underlying executor and
    the mapping from in-flight futures to tasks. Callers are expected
    to use it as a context manager so the executor is properly shut
    down::

        with ParallelScheduler(build_spec=spec_fn, max_workers=4) as sched:
            ...
    """

    def __init__(
        self,
        build_spec: Callable[[Task, TaskGraph], Any],
        worker: WorkerCallable,
        max_workers: int = 4,
        executor_kind: Literal["process", "thread"] = "process",
    ) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive.")
        self._build_spec = build_spec
        self._worker = worker
        self._max_workers = max_workers
        self._executor_kind = executor_kind
        self._executor: Executor | None = None
        self._futures: dict[Future, Task] = {}
        self._newly_started: list[Task] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def __enter__(self) -> "ParallelScheduler":
        if self._executor_kind == "process":
            self._executor = ProcessPoolExecutor(max_workers=self._max_workers)
        else:
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown(wait=exc_type is None)

    def shutdown(self, wait: bool = True) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=wait, cancel_futures=not wait)
            self._executor = None

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------
    @property
    def running_count(self) -> int:
        return len(self._futures)

    @property
    def has_capacity(self) -> bool:
        return self.running_count < self._max_workers

    def submit_ready(self, graph: TaskGraph) -> list[TaskScheduledEvent]:
        """Submit as many ready tasks as the pool has capacity for.

        Returns the list of :class:`TaskScheduledEvent` the outer agent
        should yield. The graph is mutated: submitted tasks transition
        to ``RUNNING``.
        """
        if self._executor is None:
            raise RuntimeError("ParallelScheduler must be used as a context manager.")

        events: list[TaskScheduledEvent] = []
        for task in graph.ready_tasks():
            if not self.has_capacity:
                break
            spec = self._build_spec(task, graph)
            future = self._executor.submit(self._worker, spec)
            graph.mark_running(task.id)
            self._futures[future] = task
            self._newly_started.append(task)
            events.append(TaskScheduledEvent(task=task))
        return events

    # ------------------------------------------------------------------
    # Waiting
    # ------------------------------------------------------------------
    def drain_started_events(self) -> Iterator[TaskStartedEvent]:
        """Yield ``TaskStartedEvent`` for tasks submitted since the last drain.

        In this iteration we emit scheduled+started back-to-back; the
        progress-reporting slice will move the started event to a
        genuine worker-side signal.
        """
        while self._newly_started:
            yield TaskStartedEvent(task=self._newly_started.pop(0))

    def wait_for_any(self, timeout: float | None = None) -> list["ParallelEvent"]:
        """Block until at least one future finishes, then drain completions.

        Returns the full batch of events produced by the drain so the
        outer loop can yield them in order.
        """
        if not self._futures:
            return []

        done, _ = wait(list(self._futures.keys()), timeout=timeout, return_when=FIRST_COMPLETED)

        events: list[ParallelEvent] = []
        for future in done:
            task = self._futures.pop(future)
            try:
                result: TaskResult = future.result()
            except BaseException as exc:  # noqa: BLE001 — rethrown via SchedulerFailure below
                raise SchedulerFailure(task_id=task.id, cause=exc) from exc
            if not isinstance(result, TaskResult):
                raise SchedulerFailure(
                    task_id=task.id,
                    cause=TypeError(
                        f"Worker returned {type(result).__name__}, expected TaskResult."
                    ),
                )
            task.result = result.output
            events.append(TaskCompletedEvent(task=task))
        return events
