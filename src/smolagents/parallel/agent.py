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
"""Parallel CodeAgent that orchestrates a planner + scheduler + graph.

This is the user-facing entry point for parallel execution. When the
``parallel_execution`` flag is off, users should keep using
:class:`~smolagents.CodeAgent`; ``ParallelCodeAgent`` only kicks in
when they opt in explicitly (see design doc §11).
"""
from __future__ import annotations

import time
from collections.abc import Generator
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING, Any, Literal

from smolagents.agents import CodeAgent
from smolagents.memory import (
    ActionStep,
    ParallelTaskStep,
)
from smolagents.models import ChatMessage
from smolagents.monitoring import LogLevel, Timing
from smolagents.parallel.events import (
    PlanningTriggeredEvent,
    TaskCompletedEvent,
)
from smolagents.parallel.planner import NextPlanningTrigger, ParallelPlanner
from smolagents.parallel.scheduler import (
    ParallelScheduler,
    SchedulerFailure,
    TaskResult,
)
from smolagents.parallel.task_graph import Task, TaskGraph, TaskStatus
from smolagents.utils import AgentError, AgentExecutionError


if TYPE_CHECKING:
    import PIL.Image

    from smolagents.models import Model
    from smolagents.tools import Tool


__all__ = ["ParallelCodeAgent", "TaskWorkerSpec", "run_task_worker"]


logger = getLogger(__name__)


# ----------------------------------------------------------------------
# Worker-side: executes a single task in its own process (or thread).
# ----------------------------------------------------------------------


@dataclass
class TaskWorkerSpec:
    """Picklable description of one task execution.

    Anything the worker needs to rebuild a mini :class:`CodeAgent`
    lives here. The :class:`~smolagents.models.Model`, tools, and
    authorized imports must themselves be picklable — for the common
    case of API-backed models (``LiteLLMModel``, ``InferenceClientModel``,
    ``OpenAIModel``, ``AzureOpenAIModel``) this is already true.
    """

    task_id: str
    task_goal: str
    task_dependencies: list[str]
    dependency_results: dict[str, Any]
    model: "Model"
    tools: list["Tool"]
    additional_authorized_imports: list[str]
    max_steps: int
    code_block_tags: tuple[str, str] | Literal["markdown"] | None
    use_structured_outputs_internally: bool
    instructions: str | None


def run_task_worker(spec: TaskWorkerSpec) -> TaskResult:
    """Top-level worker entry point executed inside each child process.

    It rebuilds a fresh :class:`CodeAgent`, seeds the sandbox with the
    completed dependency results, and runs the inner ReAct loop against
    the planner's goal. The loop is exactly the standard Thought →
    Code → Observation loop today's :class:`CodeAgent` uses — this is
    where task-level LLM calls happen.
    """
    agent = CodeAgent(
        tools=list(spec.tools),
        model=spec.model,
        additional_authorized_imports=list(spec.additional_authorized_imports),
        max_steps=max(1, spec.max_steps),
        code_block_tags=spec.code_block_tags,
        use_structured_outputs_internally=spec.use_structured_outputs_internally,
        instructions=spec.instructions,
        return_full_result=True,
    )

    # Seed the worker's sandbox with completed dependency results,
    # bound to variables named after the dependency task ids.
    if spec.dependency_results:
        agent.state.update(spec.dependency_results)

    # The planner's high-level goal becomes the outer task for the
    # mini-agent. No concrete steps are injected here — the worker's
    # own LLM is expected to decide how to achieve the goal.
    inner_task = _build_inner_task_text(spec)

    run_result = agent.run(inner_task)
    output = run_result.output
    steps = run_result.steps or []
    token_usage = run_result.token_usage

    logs = ""
    if isinstance(agent.python_executor.state, dict):
        logs = str(agent.python_executor.state.get("_print_outputs", "")) or ""

    return TaskResult(
        task_id=spec.task_id,
        output=output,
        action_steps=list(steps),
        logs=logs,
        input_tokens=token_usage.input_tokens if token_usage is not None else 0,
        output_tokens=token_usage.output_tokens if token_usage is not None else 0,
    )


def _build_inner_task_text(spec: TaskWorkerSpec) -> str:
    lines = [
        f"You are executing sub-task `{spec.task_id}` of a larger parallel plan.",
        "",
        "Goal:",
        spec.task_goal,
    ]
    if spec.task_dependencies:
        lines.append("")
        lines.append(
            "The results of the following upstream tasks have been bound to "
            "variables with the same names in your Python sandbox:"
        )
        for dep_id in spec.task_dependencies:
            lines.append(f"- {dep_id}")
    lines += [
        "",
        "When you are done, call `final_answer(<result>)` with the value "
        "your downstream consumers (or the user) will need.",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Outer agent
# ----------------------------------------------------------------------


@dataclass
class _PlanningState:
    """Bookkeeping for replanning triggers."""

    trigger: NextPlanningTrigger = field(
        default_factory=lambda: NextPlanningTrigger(kind="never")
    )
    completions_since_last_plan: int = 0
    planning_calls: int = 0


class ParallelCodeAgent(CodeAgent):
    """A :class:`CodeAgent` that runs tasks concurrently.

    See the design doc for the full architecture. At a high level:

    1. The planner is asked for an initial batch of tasks (goals only).
    2. The scheduler submits every ready task to a process pool. Each
       worker runs its own bounded ReAct loop.
    3. As tasks complete, their traces are merged back into the outer
       :class:`~smolagents.memory.AgentMemory` as
       :class:`ParallelTaskStep` entries.
    4. The planner is invoked again whenever the previous call's
       ``next_planning`` trigger fires.

    Args:
        max_parallel_tasks: Pool size; defaults to 4.
        max_task_steps: ``max_steps`` for each worker's inner
            :class:`CodeAgent`. Kept low by default since tasks should
            be focused.
        executor_kind: ``"process"`` (default) or ``"thread"``. Thread
            mode is useful for tests and for environments where
            models/tools are not trivially picklable.
        max_planning_calls: Hard upper bound on planner invocations per
            run, as a safety net against pathological replanning.
    """

    def __init__(
        self,
        *args: Any,
        max_parallel_tasks: int = 4,
        max_task_steps: int = 6,
        executor_kind: Literal["process", "thread"] = "process",
        max_planning_calls: int = 20,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if max_parallel_tasks <= 0:
            raise ValueError("max_parallel_tasks must be positive.")
        if max_task_steps <= 0:
            raise ValueError("max_task_steps must be positive.")
        if max_planning_calls <= 0:
            raise ValueError("max_planning_calls must be positive.")
        self.max_parallel_tasks = max_parallel_tasks
        self.max_task_steps = max_task_steps
        self.executor_kind = executor_kind
        self.max_planning_calls = max_planning_calls
        self.planner = ParallelPlanner(model=self.model, logger=self.logger)

    # ------------------------------------------------------------------
    # Override the ReAct loop with the parallel orchestration loop.
    # ------------------------------------------------------------------
    def _run_stream(
        self,
        task: str,
        max_steps: int,
        images: list["PIL.Image.Image"] | None = None,
    ) -> Generator[Any, None, None]:
        # The outer ``max_steps`` and ``images`` knobs don't apply to
        # the parallel loop: max_steps is replaced by
        # ``max_planning_calls`` + per-task ``max_task_steps``, and
        # images are ignored for now (image-augmented parallel
        # planning can be added later if needed).
        del max_steps, images

        graph = TaskGraph()
        state = _PlanningState()

        with ParallelScheduler(
            build_spec=lambda t, g: self._build_worker_spec(t, g),
            worker=run_task_worker,
            max_workers=self.max_parallel_tasks,
            executor_kind=self.executor_kind,
        ) as scheduler:
            # Initial planning call against an empty graph.
            yield from self._do_planning(task, graph, state)

            while not graph.is_done():
                if self.interrupt_switch:
                    raise AgentError("Agent interrupted.", self.logger)

                # Submit whatever is ready.
                for event in scheduler.submit_ready(graph):
                    yield event
                for event in scheduler.drain_started_events():
                    yield event

                if scheduler.running_count == 0:
                    # Sanity guard: no running work and nothing ready.
                    # Re-plan to avoid hanging.
                    if not self._should_replan(state, graph, force=True):
                        break
                    yield from self._do_planning(task, graph, state)
                    continue

                # Block for at least one completion.
                try:
                    completion_events = scheduler.wait_for_any(timeout=None)
                except SchedulerFailure as failure:
                    graph.mark_failed(failure.task_id, str(failure.cause))
                    raise AgentExecutionError(
                        f"Parallel task {failure.task_id!r} failed: {failure.cause}",
                        self.logger,
                    ) from failure

                for event in completion_events:
                    if isinstance(event, TaskCompletedEvent):
                        self._integrate_completion(event.task, graph, state)
                    yield event

                if self._should_replan(state, graph):
                    yield from self._do_planning(task, graph, state)

            yield from self._finalize(task, graph)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def _do_planning(
        self,
        task: str,
        graph: TaskGraph,
        state: _PlanningState,
    ) -> Generator[Any, None, None]:
        if state.planning_calls >= self.max_planning_calls:
            raise AgentExecutionError(
                f"Exceeded max_planning_calls={self.max_planning_calls}.",
                self.logger,
            )
        state.planning_calls += 1

        self.logger.log_rule("Parallel planning", level=LogLevel.INFO)
        yield PlanningTriggeredEvent(trigger=state.trigger)

        result = self.planner.plan(task=task, memory=self.memory, graph=graph)
        graph.add_tasks(result.new_tasks)

        self.memory.steps.append(result.planning_step)
        self._finalize_step(result.planning_step)

        state.trigger = result.next_trigger
        state.completions_since_last_plan = 0

        self.logger.log(
            f"Planner added {len(result.new_tasks)} new task(s); "
            f"next trigger: {result.next_trigger.to_dict()}",
            level=LogLevel.INFO,
        )
        self.logger.log(graph.to_yaml(), level=LogLevel.DEBUG)

        yield result.planning_step

    def _should_replan(
        self,
        state: _PlanningState,
        graph: TaskGraph,
        force: bool = False,
    ) -> bool:
        if force:
            return not graph.is_done()
        trig = state.trigger
        if trig.kind == "never":
            return False
        if trig.kind == "after_task":
            if trig.task_id and trig.task_id in graph:
                target = graph.get(trig.task_id)
                return target.status == TaskStatus.COMPLETED
            return False
        if trig.kind == "after_n_completions":
            if trig.n and state.completions_since_last_plan >= trig.n:
                return True
            return False
        return False

    # ------------------------------------------------------------------
    # Completion integration
    # ------------------------------------------------------------------
    def _integrate_completion(
        self,
        task: Task,
        graph: TaskGraph,
        state: _PlanningState,
    ) -> None:
        # The scheduler already populated ``task.result`` on the Task
        # object it returned with the completion event. We now update
        # the graph's state machine and wrap the inner steps in a
        # ParallelTaskStep so the planner can see them next round.
        memory_step = self._make_parallel_task_step(task)
        graph.mark_completed(task.id, task.result, memory_step=memory_step)
        self.memory.steps.append(memory_step)
        self._finalize_step(memory_step)
        state.completions_since_last_plan += 1

    def _make_parallel_task_step(self, task: Task) -> ParallelTaskStep:
        duration = task.actual_runtime_s or 0.0
        start = task.started_at or time.time() - duration
        end = task.finished_at or time.time()
        step_number = 1 + sum(
            1 for s in self.memory.steps if isinstance(s, (ActionStep, ParallelTaskStep))
        )
        observations = _short_result(task.result)
        return ParallelTaskStep(
            step_number=step_number,
            timing=Timing(start_time=start, end_time=end),
            action_output=task.result,
            observations=observations,
            is_final_answer=False,
            task_id=task.id,
            task_goal=task.goal,
            dependencies=list(task.dependencies),
            expected_runtime_s=task.expected_runtime_s,
            resources=list(task.resources),
        )

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------
    def _finalize(
        self,
        task: str,
        graph: TaskGraph,
    ) -> Generator[Any, None, None]:
        from smolagents.agent_types import handle_agent_output_types
        from smolagents.memory import FinalAnswerStep

        final_answer: Any = None

        # Prefer the last completed task's result: by convention the
        # planner creates a terminal aggregator that calls
        # ``final_answer``.
        completed = graph.completed_tasks()
        if completed:
            final_answer = completed[-1].result

        if final_answer is None:
            # Fallback: ask the model directly, using the existing
            # provide_final_answer path over the enriched memory.
            try:
                chat: ChatMessage = self.provide_final_answer(task)
                final_answer = chat.content
            except Exception as exc:  # noqa: BLE001
                self.logger.log(f"provide_final_answer fallback failed: {exc}", level=LogLevel.INFO)

        self.logger.log(graph.to_yaml(), level=LogLevel.INFO)

        final_answer_step = FinalAnswerStep(output=handle_agent_output_types(final_answer))
        self._finalize_step(final_answer_step)
        yield final_answer_step

    # ------------------------------------------------------------------
    # Worker spec
    # ------------------------------------------------------------------
    def _build_worker_spec(self, task: Task, graph: TaskGraph) -> TaskWorkerSpec:
        dependency_results: dict[str, Any] = {}
        for dep_id in task.dependencies:
            dep = graph.get(dep_id)
            dependency_results[dep_id] = dep.result
        return TaskWorkerSpec(
            task_id=task.id,
            task_goal=task.goal,
            task_dependencies=list(task.dependencies),
            dependency_results=dependency_results,
            model=self.model,
            tools=[t for t in self.tools.values() if t.name != "final_answer"] + [self.tools["final_answer"]],
            additional_authorized_imports=list(self.additional_authorized_imports),
            max_steps=self.max_task_steps,
            code_block_tags=self.code_block_tags,
            use_structured_outputs_internally=self._use_structured_outputs_internally,
            instructions=self.instructions,
        )


def _short_result(result: Any, max_len: int = 500) -> str:
    if result is None:
        return ""
    text = str(result)
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text
