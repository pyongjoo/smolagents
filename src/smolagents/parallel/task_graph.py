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
"""Task graph for parallel agent execution.

This module owns the DAG of tasks that are scheduled, running, completed,
or failed during a parallel agent run. It is deliberately a plain data
module with no dependencies on the scheduler, planner, or the model
layer — those interact with the graph only through its narrow API.
"""
from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import yaml


if TYPE_CHECKING:
    from smolagents.memory import ActionStep


__all__ = ["Task", "TaskStatus", "TaskGraph"]


class TaskStatus(str, Enum):
    """Lifecycle states a :class:`Task` can be in.

    The state machine is strictly:

        PENDING -> READY -> RUNNING -> {COMPLETED | FAILED}

    ``CANCELLED`` is intentionally not present in this iteration — it
    will be added together with failure handling (see the design doc
    §14).
    """

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# Valid state transitions. Kept as a module-level constant so tests can
# assert against it and so the state machine is easy to audit.
_VALID_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    # In practice the READY state is optional: the scheduler can submit
    # a pending task whose dependencies are satisfied directly to the
    # pool, skipping READY. We keep the state for observability but
    # accept PENDING -> RUNNING too.
    TaskStatus.PENDING: {TaskStatus.READY, TaskStatus.RUNNING, TaskStatus.FAILED},
    TaskStatus.READY: {TaskStatus.RUNNING, TaskStatus.PENDING, TaskStatus.FAILED},
    TaskStatus.RUNNING: {TaskStatus.COMPLETED, TaskStatus.FAILED},
    TaskStatus.COMPLETED: set(),
    TaskStatus.FAILED: set(),
}


@dataclass
class Task:
    """A single planner-produced unit of work.

    A :class:`Task` only describes *what* needs to happen (via ``goal``)
    plus metadata about when it can run and how expensive it is. The
    *how* is decided at execution time by the worker's own inner ReAct
    loop; the task graph itself never contains code or tool call
    sequences.
    """

    id: str
    goal: str
    dependencies: list[str] = field(default_factory=list)
    expected_runtime_s: float | None = None
    resources: list[str] = field(default_factory=list)

    status: TaskStatus = TaskStatus.PENDING
    result: Any | None = None
    error: str | None = None

    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    # Populated on completion so the outer agent can merge the task's
    # inner trace into ``AgentMemory``. Kept loosely typed here to avoid
    # circular imports with ``smolagents.memory``.
    memory_step: "ActionStep | None" = None

    @property
    def actual_runtime_s(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.started_at


class TaskGraph:
    """In-memory DAG of :class:`Task` objects.

    The graph is a plain container: it knows about dependency edges,
    enforces the status state machine, and can serialize itself to the
    small YAML view that gets fed back to the planner. It does not run
    tasks or make LLM calls.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        # Preserve insertion order so snapshots are stable/readable.
        self._order: list[str] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def add_tasks(self, tasks: Iterable[Task]) -> None:
        """Add a batch of newly-planned tasks to the graph.

        The batch is validated as a whole so that dependencies can refer
        either to already-present tasks or to other tasks in the same
        batch. If validation fails, the graph is left unchanged.
        """
        new_tasks = list(tasks)
        incoming_ids = {t.id for t in new_tasks}
        if len(incoming_ids) != len(new_tasks):
            raise ValueError("Duplicate task ids in the same planning batch.")

        for task in new_tasks:
            if task.id in self._tasks:
                raise ValueError(f"Task id {task.id!r} already exists in the graph.")
            for dep in task.dependencies:
                if dep not in self._tasks and dep not in incoming_ids:
                    raise ValueError(
                        f"Task {task.id!r} depends on unknown task id {dep!r}."
                    )
                if dep == task.id:
                    raise ValueError(f"Task {task.id!r} cannot depend on itself.")

        for task in new_tasks:
            self._tasks[task.id] = task
            self._order.append(task.id)

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------
    def get(self, task_id: str) -> Task:
        return self._tasks[task_id]

    def __contains__(self, task_id: str) -> bool:
        return task_id in self._tasks

    def __len__(self) -> int:
        return len(self._tasks)

    def all_tasks(self) -> list[Task]:
        return [self._tasks[tid] for tid in self._order]

    def tasks_with_status(self, status: TaskStatus) -> list[Task]:
        return [self._tasks[tid] for tid in self._order if self._tasks[tid].status == status]

    def completed_tasks(self) -> list[Task]:
        return self.tasks_with_status(TaskStatus.COMPLETED)

    def running_tasks(self) -> list[Task]:
        return self.tasks_with_status(TaskStatus.RUNNING)

    def failed_tasks(self) -> list[Task]:
        return self.tasks_with_status(TaskStatus.FAILED)

    def pending_tasks(self) -> list[Task]:
        """Tasks whose dependencies are not yet satisfied."""
        result = []
        for tid in self._order:
            task = self._tasks[tid]
            if task.status != TaskStatus.PENDING:
                continue
            if not self._deps_satisfied(task):
                result.append(task)
        return result

    def ready_tasks(self) -> list[Task]:
        """Tasks that can be submitted right now.

        A task is *ready* when it is still in ``PENDING`` or ``READY``
        status and every declared dependency is ``COMPLETED``.
        """
        result = []
        for tid in self._order:
            task = self._tasks[tid]
            if task.status not in (TaskStatus.PENDING, TaskStatus.READY):
                continue
            if self._deps_satisfied(task):
                result.append(task)
        return result

    def is_done(self) -> bool:
        """True when the graph cannot make any further progress.

        This is a looser condition than "every task is terminal". If a
        parent task failed, every descendant that depends (transitively)
        on it will stay ``PENDING`` forever; those don't count as active
        work. Concretely: ``is_done()`` is True iff there are no
        running tasks AND no pending/ready task has all of its
        dependencies satisfied.
        """
        for task in self._tasks.values():
            if task.status == TaskStatus.RUNNING:
                return False
            if task.status in (TaskStatus.PENDING, TaskStatus.READY) and self._deps_satisfied(task):
                return False
        return True

    def _deps_satisfied(self, task: Task) -> bool:
        for dep_id in task.dependencies:
            dep = self._tasks.get(dep_id)
            if dep is None or dep.status != TaskStatus.COMPLETED:
                return False
        return True

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------
    def _transition(self, task_id: str, new_status: TaskStatus) -> Task:
        task = self._tasks[task_id]
        allowed = _VALID_TRANSITIONS[task.status]
        if new_status not in allowed:
            raise ValueError(
                f"Illegal status transition for {task_id!r}: "
                f"{task.status.value} -> {new_status.value}"
            )
        task.status = new_status
        return task

    def mark_ready(self, task_id: str) -> None:
        self._transition(task_id, TaskStatus.READY)

    def mark_running(self, task_id: str) -> None:
        task = self._transition(task_id, TaskStatus.RUNNING)
        task.started_at = time.time()

    def mark_completed(
        self,
        task_id: str,
        result: Any,
        memory_step: "ActionStep | None" = None,
    ) -> None:
        task = self._transition(task_id, TaskStatus.COMPLETED)
        task.result = result
        task.memory_step = memory_step
        task.finished_at = time.time()

    def mark_failed(self, task_id: str, error: str) -> None:
        task = self._transition(task_id, TaskStatus.FAILED)
        task.error = error
        task.finished_at = time.time()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_snapshot(self, include_result_summary: bool = True) -> dict[str, Any]:
        """Return a plain-dict snapshot suitable for YAML dumping.

        Only the small, plan-relevant fields are included; raw tool
        outputs stay in memory and are not shipped to the planner.
        """
        tasks: list[dict[str, Any]] = []
        for task in self.all_tasks():
            entry: dict[str, Any] = {
                "id": task.id,
                "goal": task.goal,
                "status": task.status.value,
                "dependencies": list(task.dependencies),
                "expected_runtime_s": task.expected_runtime_s,
                "resources": list(task.resources),
            }
            if task.actual_runtime_s is not None:
                entry["actual_runtime_s"] = round(task.actual_runtime_s, 3)
            if include_result_summary and task.status == TaskStatus.COMPLETED:
                entry["result_summary"] = _summarize_result(task.result)
            if task.status == TaskStatus.FAILED and task.error:
                entry["error"] = task.error
            tasks.append(entry)
        return {"tasks": tasks}

    def to_yaml(self, include_result_summary: bool = True) -> str:
        return yaml.safe_dump(
            self.to_snapshot(include_result_summary=include_result_summary),
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )


def _summarize_result(result: Any, max_len: int = 300) -> str:
    """Produce a short string summary of a task result for prompts."""
    if result is None:
        return ""
    text = str(result)
    text = text.replace("\n", " ").strip()
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text
