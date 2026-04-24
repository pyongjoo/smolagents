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

import threading
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from datetime import datetime
from logging import getLogger
from typing import TYPE_CHECKING, Any, Literal

from rich import box
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from smolagents.agents import CodeAgent
from smolagents.memory import (
    ActionStep,
    ParallelTaskStep,
)
from smolagents.models import ChatMessage
from smolagents.monitoring import YELLOW_HEX, LogLevel, Timing
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
    # The original end-user task. Included verbatim in the worker
    # prompt as read-only context so sub-agents can resolve
    # back-references in their goal (e.g. "the requested dictionary")
    # to the literal schema/values the user asked for. Workers must
    # still treat ``task_goal`` as the scope of their own work.
    user_task: str
    model: "Model"
    tools: list["Tool"]
    additional_authorized_imports: list[str]
    max_steps: int
    code_block_tags: tuple[str, str] | Literal["markdown"] | None
    use_structured_outputs_internally: bool
    instructions: str | None
    # Optional real-time step callback. When provided, the worker
    # registers it on the inner agent so each inner ActionStep is
    # reported as soon as it finishes — letting the outer agent print
    # panels live instead of waiting until the whole task completes.
    # Must be None for process-pool execution (closures aren't
    # picklable); the outer agent attaches it only in thread mode.
    step_callback: Callable[[Any], None] | None = None


def run_task_worker(spec: TaskWorkerSpec) -> TaskResult:
    """Top-level worker entry point executed inside each child process.

    It rebuilds a fresh :class:`CodeAgent`, seeds the sandbox with the
    completed dependency results, and runs the inner ReAct loop against
    the planner's goal. The loop is exactly the standard Thought →
    Code → Observation loop today's :class:`CodeAgent` uses — this is
    where task-level LLM calls happen.

    The inner agent is silenced (``verbosity_level=LogLevel.OFF``) so
    its logs don't interleave with other concurrent workers on stdout.
    In thread mode the outer agent passes a ``step_callback`` that
    gets invoked after each inner ActionStep; panels are printed live
    under a shared lock so they still appear grouped per step. In
    process mode the callback is omitted (closures aren't picklable)
    and the outer agent iterates ``action_steps`` on completion
    instead.
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
        verbosity_level=LogLevel.OFF,
    )

    if spec.step_callback is not None:
        agent.step_callbacks.register(ActionStep, spec.step_callback)

    if spec.dependency_results:
        agent.state.update(spec.dependency_results)

    inner_task = _build_inner_task_text(spec)

    started_at = time.time()
    run_result = agent.run(inner_task)
    finished_at = time.time()
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
        started_at=started_at,
        finished_at=finished_at,
    )


def _build_inner_task_text(spec: TaskWorkerSpec) -> str:
    lines = [
        f"You are executing sub-task `{spec.task_id}` of a larger parallel plan.",
    ]
    # Pass the original user task as read-only context so the worker
    # can resolve any back-references in its goal (e.g. "the requested
    # dictionary") to the literal schema or values the user asked
    # for. The worker must still only do the work scoped by ``Goal``.
    if spec.user_task:
        lines += [
            "",
            "Overall user task (for context only — do NOT try to solve it in full):",
            spec.user_task.strip(),
        ]
    lines += [
        "",
        "Goal (this is what YOU must accomplish):",
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
        "your downstream consumers (or the user) will need. If your goal "
        "is the terminal aggregator, make sure the value you pass to "
        "`final_answer` matches the exact shape/schema the overall user "
        "task requires.",
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
        # Shared print lock + per-task visible-step counter. In thread
        # mode worker callbacks fire from the pool threads and grab
        # this lock to print a step's three panels atomically.
        self._print_lock = threading.Lock()
        self._visible_steps_by_task: dict[str, int] = {}
        self._current_user_task: str = ""

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
        self._visible_steps_by_task = {}
        # Stash the user task so ``_build_worker_spec`` can include
        # it verbatim in every sub-agent's prompt as context.
        self._current_user_task = task

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
                        self._integrate_completion(event.task, event.result, graph, state)
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

        yield PlanningTriggeredEvent(trigger=state.trigger)

        # The trigger active at the moment we decide to call the
        # planner is what caused *this* invocation. We stash it before
        # the call so the rendered panel and memory step both reflect
        # why planning is happening now. ``None`` means the first
        # call, which was forced by the empty-graph bootstrap rather
        # than by any trigger firing.
        invoked_by_trigger = None if state.planning_calls == 1 else state.trigger

        # Wall-clock timings captured around the LLM call so the
        # rendered panel can show how the planning phase overlaps
        # with already-running tasks.
        plan_start = time.time()
        result = self.planner.plan(task=task, memory=self.memory, graph=graph)
        plan_end = time.time()

        graph.add_tasks(result.new_tasks)
        # Attach the "what caused this call" metadata so downstream
        # consumers (e.g. the final summary, saved memory dumps) can
        # reconstruct the planning lineage.
        result.planning_step.invoked_by = (
            invoked_by_trigger.to_dict() if invoked_by_trigger is not None else None
        )
        self.memory.steps.append(result.planning_step)
        self._finalize_step(result.planning_step)

        state.trigger = result.next_trigger
        state.completions_since_last_plan = 0

        self._log_planning_panel(
            call_index=state.planning_calls,
            plan_start=plan_start,
            plan_end=plan_end,
            new_tasks=result.new_tasks,
            next_trigger=result.next_trigger,
            invoked_by=invoked_by_trigger,
        )
        self.logger.log(graph.to_yaml(), level=LogLevel.DEBUG)

        yield result.planning_step

    def _log_planning_panel(
        self,
        *,
        call_index: int,
        plan_start: float,
        plan_end: float,
        new_tasks: list[Task],
        next_trigger: NextPlanningTrigger,
        invoked_by: NextPlanningTrigger | None,
    ) -> None:
        """Render the outcome of one planner LLM call as a bounding box."""
        body = Text()
        body.append("invoked: ", style="dim")
        body.append(f"{_describe_trigger(invoked_by)}\n", style="bold")
        body.append("next planning: ", style="dim")
        body.append(f"{_describe_trigger(next_trigger)}\n", style="bold")
        body.append(f"added {len(new_tasks)} new task(s)\n", style="bold")
        if not new_tasks:
            body.append("(no new tasks)\n", style="dim")
        for t in new_tasks:
            deps = t.dependencies if t.dependencies else []
            runtime = f"{t.expected_runtime_s:g}s" if t.expected_runtime_s is not None else "?"
            body.append(f"  + {t.id} ", style="bold cyan")
            body.append(f"(deps={deps or '[]'}, ~{runtime}): ", style="dim")
            body.append(f"{t.goal}\n")

        title = (
            f"[bold]Parallel planning · call #{call_index} · "
            f"start={_fmt_ts(plan_start)} · end={_fmt_ts(plan_end)} · "
            f"duration={plan_end - plan_start:.2f}s"
        )
        self.logger.log(
            Panel(
                body,
                title=title,
                border_style=YELLOW_HEX,
                title_align="left",
                box=box.ROUNDED,
            ),
            level=LogLevel.INFO,
        )

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
        result: "TaskResult | None",
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
        # Intentionally do NOT call ``self._finalize_step`` here: the
        # default ``Monitor.update_metrics`` callback (registered for
        # ActionStep) would print lines like
        # ``[Step 3: Duration 0.42 seconds]`` that falsely imply a
        # sequential step counter. Parallel tasks have per-task
        # timings that we render separately in the summary table.
        memory_step.timing.end_time = time.time()
        state.completions_since_last_plan += 1

        # In thread mode each step was already printed live via
        # ``_emit_step_panels_live`` from the worker thread. Process
        # mode has no live channel (closures aren't picklable), so we
        # fall back to the batch path that replays all action steps
        # here.
        if result is not None and self.executor_kind != "thread":
            self._emit_task_panels(task, result)

        # Finally, show what actually lands in the outer AgentMemory.
        # This is the view the planner will see on its next round via
        # ``ParallelPlanner._summarize_memory``, so printing it here
        # makes the planner's next input legible.
        self._emit_memory_panel(memory_step)

    # ------------------------------------------------------------------
    # Per-task rendering
    # ------------------------------------------------------------------
    def _emit_task_panels(self, task: Task, result: "TaskResult") -> None:
        """Replay all inner steps of a completed task as panels.

        Used only in process mode; thread mode prints each step live
        as the worker finishes it.
        """
        for raw in result.action_steps:
            step = raw if isinstance(raw, dict) else _action_step_to_dict(raw)
            if step is None:
                continue
            self._emit_step_panels(task.id, step)

    def _emit_step_panels_live(self, task_id: str, memory_step: Any) -> None:
        """Step callback installed on inner agents in thread mode.

        Runs on the worker thread. Converts the finished ``ActionStep``
        to a dict and prints its panels under ``_print_lock`` so the
        three phase boxes for a single step never interleave with
        output from a concurrent worker.
        """
        try:
            step_dict = memory_step.dict()
        except Exception:
            return
        with self._print_lock:
            self._emit_step_panels(task_id, step_dict)

    def _emit_step_panels(self, task_id: str, step: dict) -> None:
        """Print up to three bounding boxes for one inner action step.

        ``step`` is the dict form of a worker's :class:`ActionStep`.
        We skip entries that carry no renderable phase (e.g. the
        initial ``TaskStep`` that leads the worker's memory) so the
        visible ``step=N`` label counts only action-bearing steps.
        """
        model_output = step.get("model_output")
        code_action = step.get("code_action")
        observations = step.get("observations")
        action_output = step.get("action_output")
        error = step.get("error")
        if not (
            model_output
            or code_action
            or observations
            or action_output is not None
            or error
        ):
            return

        visible_step = self._visible_steps_by_task.get(task_id, 0) + 1
        self._visible_steps_by_task[task_id] = visible_step

        timing = step.get("timing") or {}
        s_start = timing.get("start_time")
        s_end = timing.get("end_time")
        # Titles follow a fixed order so every panel is easy to scan:
        # task-id · step · phase · start · end.
        def _title(phase: str) -> str:
            return (
                f"[bold]{task_id} · step={visible_step} · {phase} · "
                f"start={_fmt_ts(s_start)} · end={_fmt_ts(s_end)}"
            )

        if model_output:
            self.logger.log(
                Panel(
                    Text(_short_result(model_output, 4000), overflow="fold"),
                    title=_title("LLM response"),
                    border_style="cyan",
                    title_align="left",
                    box=box.ROUNDED,
                ),
                level=LogLevel.INFO,
            )

        if code_action:
            self.logger.log(
                Panel(
                    Syntax(
                        code_action,
                        lexer="python",
                        theme="monokai",
                        word_wrap=True,
                    ),
                    title=_title("Code execution"),
                    border_style="magenta",
                    title_align="left",
                    box=box.ROUNDED,
                ),
                level=LogLevel.INFO,
            )

        obs_text: str | None = None
        if observations:
            obs_text = _short_result(observations, 2000)
        elif action_output is not None:
            obs_text = _short_result(action_output, 2000)

        if obs_text or error:
            body_parts: list[Any] = []
            if obs_text:
                body_parts.append(Text(obs_text, overflow="fold"))
            if error:
                body_parts.append(
                    Text(f"error: {error}", style="bold red", overflow="fold")
                )
            body = body_parts[0] if len(body_parts) == 1 else Text("\n").join(body_parts)
            self.logger.log(
                Panel(
                    body,
                    title=_title("Observation"),
                    border_style="green" if not error else "red",
                    title_align="left",
                    box=box.ROUNDED,
                ),
                level=LogLevel.INFO,
            )

    def _emit_memory_panel(self, step: ParallelTaskStep) -> None:
        """Render the outer-memory view of a just-completed task.

        The planner consumes ``AgentMemory`` on its next call, and for
        a :class:`ParallelTaskStep` it only sees the task id, goal,
        dependencies, and a truncated observation (see
        :meth:`ParallelPlanner._summarize_memory`). Printing exactly
        those fields here makes the hand-off from execution to the
        next planning round explicit.
        """
        body = Text()
        body.append("task_id: ", style="dim")
        body.append(f"{step.task_id}\n", style="bold")
        body.append("goal: ", style="dim")
        body.append(f"{_short_result(step.task_goal, 300)}\n")
        deps = ", ".join(step.dependencies) if step.dependencies else "-"
        body.append("dependencies: ", style="dim")
        body.append(f"{deps}\n")
        if step.action_output is not None:
            body.append("action_output: ", style="dim")
            body.append(f"{_short_result(step.action_output, 400)}\n")
        obs = step.observations or ""
        if obs:
            body.append("observations: ", style="dim")
            body.append(f"{_short_result(obs, 400)}\n")
        body.append("planner will see: ", style="dim")
        body.append(
            f"- completed {step.task_id}: "
            f"{_short_result(step.task_goal, 120)} -> "
            f"{_short_result(obs or step.action_output, 160)}",
            style="italic",
        )

        panel = Panel(
            body,
            title=f"[bold]{step.task_id} · stored in AgentMemory as ParallelTaskStep",
            border_style=YELLOW_HEX,
            title_align="left",
            box=box.ROUNDED,
        )
        # Take the shared lock so the panel doesn't interleave with a
        # concurrent worker's live step panels in thread mode.
        with self._print_lock:
            self.logger.log(panel, level=LogLevel.INFO)

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
            created_at=task.created_at,
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

        self._log_task_summary(graph)
        # Keep the plain-YAML snapshot around at DEBUG for users who
        # want a machine-readable dump of the whole graph.
        self.logger.log(graph.to_yaml(), level=LogLevel.DEBUG)

        final_answer_step = FinalAnswerStep(output=handle_agent_output_types(final_answer))
        self._finalize_step(final_answer_step)
        yield final_answer_step

    # ------------------------------------------------------------------
    # Final summary table
    # ------------------------------------------------------------------
    def _log_task_summary(self, graph: TaskGraph) -> None:
        """Render a per-task summary table with creation/start/end times."""
        table = Table(
            title="Parallel task summary",
            box=box.ROUNDED,
            title_style="bold",
            show_lines=False,
            header_style="bold",
        )
        table.add_column("Task")
        table.add_column("Status")
        table.add_column("Deps")
        table.add_column("Goal")
        table.add_column("Created")
        table.add_column("Started")
        table.add_column("Finished")
        table.add_column("Duration", justify="right")
        table.add_column("Expected", justify="right")

        status_styles = {
            TaskStatus.COMPLETED: "green",
            TaskStatus.FAILED: "red",
            TaskStatus.RUNNING: "yellow",
            TaskStatus.PENDING: "dim",
            TaskStatus.READY: "cyan",
        }

        for t in graph.all_tasks():
            goal = t.goal.strip().splitlines()[0] if t.goal else ""
            if len(goal) > 72:
                goal = goal[:71] + "…"
            actual = t.actual_runtime_s
            expected = t.expected_runtime_s
            table.add_row(
                t.id,
                Text(t.status.value, style=status_styles.get(t.status, "")),
                ",".join(t.dependencies) if t.dependencies else "-",
                goal,
                _fmt_ts(t.created_at),
                _fmt_ts(t.started_at),
                _fmt_ts(t.finished_at),
                f"{actual:.2f}s" if actual is not None else "-",
                f"~{expected:.0f}s" if expected is not None else "-",
            )

        self.logger.log(table, level=LogLevel.INFO)

    # ------------------------------------------------------------------
    # Worker spec
    # ------------------------------------------------------------------
    def _build_worker_spec(self, task: Task, graph: TaskGraph) -> TaskWorkerSpec:
        dependency_results: dict[str, Any] = {}
        for dep_id in task.dependencies:
            dep = graph.get(dep_id)
            dependency_results[dep_id] = dep.result

        # Live step callback is only safe in thread mode — process
        # pools would need to pickle the closure, which won't work.
        step_callback: Callable[[Any], None] | None = None
        if self.executor_kind == "thread":
            task_id = task.id

            def on_inner_step(memory_step: Any) -> None:
                self._emit_step_panels_live(task_id, memory_step)

            step_callback = on_inner_step

        return TaskWorkerSpec(
            task_id=task.id,
            task_goal=task.goal,
            task_dependencies=list(task.dependencies),
            dependency_results=dependency_results,
            user_task=getattr(self, "_current_user_task", "") or "",
            model=self.model,
            tools=[t for t in self.tools.values() if t.name != "final_answer"] + [self.tools["final_answer"]],
            additional_authorized_imports=list(self.additional_authorized_imports),
            max_steps=self.max_task_steps,
            code_block_tags=self.code_block_tags,
            use_structured_outputs_internally=self._use_structured_outputs_internally,
            instructions=self.instructions,
            step_callback=step_callback,
        )


def _short_result(result: Any, max_len: int = 500) -> str:
    if result is None:
        return ""
    text = str(result)
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text


def _fmt_ts(ts: float | None) -> str:
    if ts is None:
        return "--:--:--.---"
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3]


def _action_step_to_dict(step: Any) -> dict | None:
    """Fallback serializer when the worker returned raw step objects."""
    try:
        return step.dict()
    except Exception:
        return None


def _describe_trigger(
    trigger: "NextPlanningTrigger | dict | None",
) -> str:
    """Render a ``NextPlanningTrigger`` (or its serialized dict form)
    as a short human-readable phrase.

    Accepts either the live dataclass (as used by the agent) or the
    serialized dict form (as stored on :class:`ParallelPlanningStep`)
    so the same helper can drive the live panel and post-run
    summaries.
    """
    if trigger is None:
        return "initial planning (empty graph)"
    d = trigger.to_dict() if hasattr(trigger, "to_dict") else dict(trigger)
    kind = d.get("kind", "never")
    if kind == "never":
        return "never — this is a terminal plan"
    if kind == "after_task":
        tid = d.get("task_id") or "?"
        return f"after task {tid} completes"
    if kind == "after_n_completions":
        n = d.get("n")
        if isinstance(n, int):
            noun = "completion" if n == 1 else "completions"
            return f"after {n} more task {noun}"
        return "after N more task completions"
    return str(d)
