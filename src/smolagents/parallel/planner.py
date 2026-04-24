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
"""Incremental planner for parallel agent runs.

A single ``plan()`` call wraps one LLM request that proposes *new*
tasks to add to the graph. There is no separate "initial" branch —
the first call simply sees an empty graph. See design doc §7 for the
full contract.
"""
from __future__ import annotations

import importlib.resources
import re
import textwrap
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import yaml
from jinja2 import StrictUndefined, Template

from smolagents.memory import ParallelPlanningStep
from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import Timing, TokenUsage
from smolagents.parallel.task_graph import Task, TaskGraph
from smolagents.utils import AgentParsingError


if TYPE_CHECKING:
    from smolagents.memory import AgentMemory
    from smolagents.models import Model
    from smolagents.monitoring import AgentLogger


__all__ = [
    "NextPlanningTrigger",
    "PlanningResult",
    "ParallelPlanner",
    "load_default_planning_prompt",
]


TriggerKind = Literal["after_task", "after_n_completions", "never"]


@dataclass
class NextPlanningTrigger:
    """When the runtime should invoke the planner again.

    ``on_failure`` is intentionally omitted here — it will arrive with
    the failure-handling slice (see design doc §14).
    """

    kind: TriggerKind = "never"
    task_id: str | None = None
    n: int | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind}
        if self.task_id is not None:
            out["task_id"] = self.task_id
        if self.n is not None:
            out["n"] = self.n
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "NextPlanningTrigger":
        if not data:
            return cls(kind="never")
        kind = data.get("kind", "never")
        if kind not in ("after_task", "after_n_completions", "never"):
            raise ValueError(f"Unknown next_planning.kind {kind!r}.")
        return cls(
            kind=kind,
            task_id=data.get("task_id"),
            n=data.get("n"),
        )


@dataclass
class PlanningResult:
    """Output of one :meth:`ParallelPlanner.plan` call."""

    new_tasks: list[Task]
    next_trigger: NextPlanningTrigger
    reasoning: str
    raw_output: str
    planning_step: ParallelPlanningStep
    added_task_ids: list[str] = field(default_factory=list)


def load_default_planning_prompt() -> str:
    """Load the default parallel planning prompt bundled with the package."""
    text = (
        importlib.resources.files("smolagents.prompts")
        .joinpath("parallel_planning.yaml")
        .read_text(encoding="utf-8")
    )
    doc = yaml.safe_load(text)
    return doc["planning"]


class ParallelPlanner:
    """Wraps one incremental planning LLM call.

    Args:
        model: Any smolagents :class:`~smolagents.models.Model`.
        logger: The outer agent's logger, used for warnings and debug
            output.
        prompt_template: Optional Jinja2 template string. If omitted,
            the bundled ``parallel_planning.yaml`` template is used.
        max_retries: Number of additional attempts allowed on malformed
            LLM output. Defaults to 1 (so at most two LLM calls per
            planning step).
    """

    def __init__(
        self,
        model: "Model",
        logger: "AgentLogger",
        prompt_template: str | None = None,
        max_retries: int = 1,
    ) -> None:
        self.model = model
        self.logger = logger
        self.prompt_template = prompt_template or load_default_planning_prompt()
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plan(
        self,
        task: str,
        memory: "AgentMemory | None",
        graph: TaskGraph,
    ) -> PlanningResult:
        """Run one incremental planning call.

        Raises :class:`AgentParsingError` if the LLM never returns
        well-formed YAML after ``max_retries + 1`` attempts.
        """
        start_time = time.time()

        memory_summary = self._summarize_memory(memory)
        graph_yaml = graph.to_yaml()

        messages = self._build_messages(
            task=task,
            memory_summary=memory_summary,
            graph_yaml=graph_yaml,
            strict_reminder=False,
        )

        last_error: Exception | None = None
        raw_output = ""
        reasoning = ""
        new_tasks: list[Task] = []
        trigger = NextPlanningTrigger(kind="never")
        chat_message: ChatMessage | None = None
        success = False

        for attempt in range(self.max_retries + 1):
            try:
                chat_message = self.model.generate(messages)
                raw_output = chat_message.content or ""
                parsed = self._parse_yaml(raw_output)
                reasoning = str(parsed.get("reasoning", "")).strip()
                new_tasks = self._parse_new_tasks(parsed.get("new_tasks"), graph)
                trigger = NextPlanningTrigger.from_dict(parsed.get("next_planning"))
                self._validate_trigger(trigger, graph, new_tasks)
                success = True
                break
            except (AgentParsingError, yaml.YAMLError, ValueError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                # Reset messages with a stricter reminder for the retry.
                messages = self._build_messages(
                    task=task,
                    memory_summary=memory_summary,
                    graph_yaml=graph_yaml,
                    strict_reminder=True,
                )

        if not success or chat_message is None:
            raise AgentParsingError(
                f"Planner failed to return a valid plan after "
                f"{self.max_retries + 1} attempt(s). Last error: {last_error}",
                self.logger,
            )

        input_tokens = 0
        output_tokens = 0
        if chat_message.token_usage is not None:
            input_tokens = chat_message.token_usage.input_tokens
            output_tokens = chat_message.token_usage.output_tokens

        planning_step = ParallelPlanningStep(
            model_input_messages=messages,
            plan=reasoning or raw_output,
            model_output_message=chat_message,
            token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            timing=Timing(start_time=start_time, end_time=time.time()),
            graph_snapshot_yaml=graph_yaml,
            new_task_ids=[t.id for t in new_tasks],
            next_trigger=trigger.to_dict(),
        )

        return PlanningResult(
            new_tasks=new_tasks,
            next_trigger=trigger,
            reasoning=reasoning,
            raw_output=raw_output,
            planning_step=planning_step,
            added_task_ids=[t.id for t in new_tasks],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_messages(
        self,
        task: str,
        memory_summary: str,
        graph_yaml: str,
        strict_reminder: bool,
    ) -> list[ChatMessage]:
        rendered = Template(self.prompt_template, undefined=StrictUndefined).render(
            task=task,
            memory_summary=memory_summary,
            graph_yaml=graph_yaml,
        )
        if strict_reminder:
            rendered += (
                "\n\nREMINDER: Your previous response was not valid YAML or did "
                "not follow the schema. Return ONLY the YAML document with keys "
                "`reasoning`, `new_tasks`, `next_planning`. No prose, no fenced "
                "code blocks."
            )
        return [
            ChatMessage(
                role=MessageRole.USER,
                content=[{"type": "text", "text": rendered}],
            )
        ]

    @staticmethod
    def _parse_yaml(raw: str) -> dict[str, Any]:
        """Parse YAML from an LLM response, tolerating fenced code blocks."""
        text = raw.strip()
        # Strip a surrounding ```yaml ... ``` fence if present.
        fence_match = re.match(
            r"^```(?:ya?ml)?\s*(.*?)\s*```$", text, re.DOTALL | re.IGNORECASE
        )
        if fence_match:
            text = fence_match.group(1)
        if not text:
            raise ValueError("Empty planner output.")
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(
                f"Planner output must be a YAML mapping, got {type(data).__name__}."
            )
        return data

    def _parse_new_tasks(self, raw_tasks: Any, graph: TaskGraph) -> list[Task]:
        if raw_tasks is None:
            return []
        if not isinstance(raw_tasks, list):
            raise ValueError("`new_tasks` must be a YAML list.")

        parsed: list[Task] = []
        new_ids: set[str] = set()
        for idx, entry in enumerate(raw_tasks):
            if not isinstance(entry, dict):
                raise ValueError(f"new_tasks[{idx}] must be a YAML mapping.")

            task_id = entry.get("id")
            goal = entry.get("goal")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"new_tasks[{idx}].id must be a non-empty string.")
            if not isinstance(goal, str) or not goal.strip():
                raise ValueError(f"new_tasks[{idx}].goal must be a non-empty string.")
            if task_id in graph:
                raise ValueError(
                    f"Planner returned duplicate task id {task_id!r} (already in graph)."
                )
            if task_id in new_ids:
                raise ValueError(f"Planner returned duplicate task id {task_id!r} in this batch.")
            new_ids.add(task_id)

            deps_raw = entry.get("dependencies", []) or []
            if not isinstance(deps_raw, list) or not all(isinstance(d, str) for d in deps_raw):
                raise ValueError(
                    f"new_tasks[{idx}].dependencies must be a list of strings."
                )

            expected_runtime = entry.get("expected_runtime_s")
            if expected_runtime is not None and not isinstance(expected_runtime, (int, float)):
                raise ValueError(
                    f"new_tasks[{idx}].expected_runtime_s must be numeric or null."
                )

            resources_raw = entry.get("resources", []) or []
            if not isinstance(resources_raw, list) or not all(isinstance(r, str) for r in resources_raw):
                raise ValueError(
                    f"new_tasks[{idx}].resources must be a list of strings."
                )

            parsed.append(
                Task(
                    id=task_id,
                    goal=goal.strip(),
                    dependencies=list(deps_raw),
                    expected_runtime_s=float(expected_runtime) if expected_runtime is not None else None,
                    resources=list(resources_raw),
                )
            )

        # Validate dependency ids: each must reference either an
        # existing graph task or another new task from this batch.
        known = {t.id for t in graph.all_tasks()} | new_ids
        for task in parsed:
            for dep in task.dependencies:
                if dep not in known:
                    raise ValueError(
                        f"Task {task.id!r} depends on unknown id {dep!r}."
                    )
                if dep == task.id:
                    raise ValueError(f"Task {task.id!r} cannot depend on itself.")

        return parsed

    def _validate_trigger(
        self,
        trigger: NextPlanningTrigger,
        graph: TaskGraph,
        new_tasks: list[Task],
    ) -> None:
        if trigger.kind == "after_task":
            if not trigger.task_id:
                raise ValueError("`next_planning.kind == after_task` requires `task_id`.")
            known = {t.id for t in graph.all_tasks()} | {t.id for t in new_tasks}
            if trigger.task_id not in known:
                raise ValueError(
                    f"`next_planning.task_id` refers to unknown task {trigger.task_id!r}."
                )
        elif trigger.kind == "after_n_completions":
            if trigger.n is None or trigger.n <= 0:
                raise ValueError(
                    "`next_planning.kind == after_n_completions` requires positive integer `n`."
                )

    @staticmethod
    def _summarize_memory(memory: "AgentMemory | None") -> str:
        """Render a compact textual summary of the agent memory.

        We deliberately keep this small: only step numbers, task
        headings, and observation snippets. The planner does not need
        the full system prompt or raw tool outputs.
        """
        if memory is None or not getattr(memory, "steps", None):
            return "(no prior steps)"

        lines: list[str] = []
        for step in memory.steps:
            cls = step.__class__.__name__
            if cls in ("TaskStep",):
                text = getattr(step, "task", "")
                lines.append(f"- task: {_truncate(text, 200)}")
            elif cls in ("ParallelTaskStep",):
                goal = getattr(step, "task_goal", "")
                tid = getattr(step, "task_id", "")
                obs = getattr(step, "observations", "") or ""
                action_out = getattr(step, "action_output", None)
                summary = obs.strip().splitlines()[-1] if obs.strip() else ""
                if action_out is not None and not summary:
                    summary = str(action_out)
                lines.append(
                    f"- completed {tid}: {_truncate(goal, 120)} -> {_truncate(summary, 160)}"
                )
            elif cls in ("ActionStep",):
                obs = getattr(step, "observations", "") or ""
                lines.append(f"- action: {_truncate(obs, 160)}")
            elif cls in ("PlanningStep", "ParallelPlanningStep"):
                lines.append("- (planning step)")
        if not lines:
            return "(no prior steps)"
        return "\n".join(lines)


def _truncate(text: str, n: int) -> str:
    text = textwrap.shorten(str(text).replace("\n", " "), width=n, placeholder="…")
    return text
