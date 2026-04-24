# coding=utf-8
# Copyright 2024 HuggingFace Inc.
"""End-to-end tests for :class:`ParallelCodeAgent` with a mock model."""
from __future__ import annotations

import textwrap

from smolagents.agents import CodeAgent
from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.monitoring import TokenUsage
from smolagents.parallel import ParallelCodeAgent


class ScriptedPlannerAndCoderModel(Model):
    """Mock model that plays two roles based on prompt content.

    - Planning prompts (detected via the fixed header from the bundled
      prompt template) get a scripted planning YAML.
    - Worker prompts (the inner Thought/Code/Observation loop) get a
      scripted code block that calls ``final_answer``.
    """

    def __init__(self, planning_outputs: list[str], worker_outputs: dict[str, str]):
        super().__init__()
        self._planning_outputs = list(planning_outputs)
        self._worker_outputs = dict(worker_outputs)
        self.generate_calls = 0

    def generate(self, messages, stop_sequences=None, **kwargs):
        self.generate_calls += 1
        prompt = _stringify(messages)
        if "planning assistant for a parallel agent runtime" in prompt:
            content = self._planning_outputs.pop(0)
        else:
            task_id = _extract_task_id(prompt)
            content = self._worker_outputs.get(task_id)
            assert content is not None, f"No scripted worker output for {task_id!r}. Prompt: {prompt[:400]}"
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        )


def _stringify(messages) -> str:
    out = []
    for m in messages:
        content = getattr(m, "content", None)
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    out.append(str(part.get("text", "")))
                else:
                    out.append(str(part))
        else:
            out.append(str(content))
    return "\n".join(out)


def _extract_task_id(prompt: str) -> str:
    # Inner task text is built as: "... sub-task `task_N` ..."
    import re

    m = re.search(r"sub-task `([^`]+)`", prompt)
    return m.group(1) if m else ""


SINGLE_TASK_PLAN = textwrap.dedent(
    """
    reasoning: Compute once and call final_answer.
    new_tasks:
      - id: task_1
        goal: Compute 17 times 23 and return the result via final_answer.
        dependencies: []
        expected_runtime_s: 5
        resources: []
    next_planning:
      kind: never
    """
).strip()


TWO_TASK_PLAN = textwrap.dedent(
    """
    reasoning: Two independent computations, then aggregate.
    new_tasks:
      - id: task_1
        goal: Return the integer 2.
        dependencies: []
        expected_runtime_s: 1
        resources: []
      - id: task_2
        goal: Return the integer 3.
        dependencies: []
        expected_runtime_s: 1
        resources: []
      - id: task_3
        goal: Add task_1 and task_2 and call final_answer.
        dependencies: [task_1, task_2]
        expected_runtime_s: 1
        resources: []
    next_planning:
      kind: never
    """
).strip()


WORKER_MUL = """
Thought: Simple multiplication.
<code>
final_answer(17 * 23)
</code>
"""


WORKER_RETURN_2 = """
Thought: Return two.
<code>
final_answer(2)
</code>
"""


WORKER_RETURN_3 = """
Thought: Return three.
<code>
final_answer(3)
</code>
"""


WORKER_ADD = """
Thought: Add the two upstream results.
<code>
final_answer(task_1 + task_2)
</code>
"""


class TestParallelAgentEndToEnd:
    def test_single_task_happy_path(self):
        model = ScriptedPlannerAndCoderModel(
            planning_outputs=[SINGLE_TASK_PLAN],
            worker_outputs={"task_1": WORKER_MUL},
        )
        agent = ParallelCodeAgent(
            tools=[],
            model=model,
            max_parallel_tasks=2,
            max_task_steps=3,
            executor_kind="thread",
        )
        result = agent.run("What is 17 * 23?")
        assert result == 17 * 23

    def test_parallel_tasks_with_aggregator(self):
        model = ScriptedPlannerAndCoderModel(
            planning_outputs=[TWO_TASK_PLAN],
            worker_outputs={
                "task_1": WORKER_RETURN_2,
                "task_2": WORKER_RETURN_3,
                "task_3": WORKER_ADD,
            },
        )
        agent = ParallelCodeAgent(
            tools=[],
            model=model,
            max_parallel_tasks=2,
            max_task_steps=3,
            executor_kind="thread",
        )
        result = agent.run("Add two plus three.")
        assert result == 5

    def test_worker_prompt_includes_user_task(self):
        """Workers must receive the full user task as read-only context.

        This guards against regressions of the "the requested
        dictionary" failure mode where the planner's terse goal
        back-references a schema only the user task spells out.
        """
        seen_prompts: list[str] = []

        class RecordingModel(ScriptedPlannerAndCoderModel):
            def generate(self, messages, stop_sequences=None, **kwargs):
                prompt = _stringify(messages)
                if "planning assistant for a parallel agent runtime" not in prompt:
                    seen_prompts.append(prompt)
                return super().generate(messages, stop_sequences=stop_sequences, **kwargs)

        model = RecordingModel(
            planning_outputs=[SINGLE_TASK_PLAN],
            worker_outputs={"task_1": WORKER_MUL},
        )
        agent = ParallelCodeAgent(
            tools=[],
            model=model,
            max_parallel_tasks=1,
            max_task_steps=3,
            executor_kind="thread",
        )
        user_task = "What is 17 * 23?"
        agent.run(user_task)

        assert seen_prompts, "Worker was never invoked."
        worker_prompt = seen_prompts[0]
        assert user_task in worker_prompt, (
            "Worker prompt must include the original user task verbatim.\n"
            f"Prompt was:\n{worker_prompt[:800]}"
        )
        assert "Overall user task" in worker_prompt

    def test_parallel_execution_flag_on_codeagent(self):
        model = ScriptedPlannerAndCoderModel(
            planning_outputs=[SINGLE_TASK_PLAN],
            worker_outputs={"task_1": WORKER_MUL},
        )
        agent = CodeAgent(
            tools=[],
            model=model,
            parallel_execution=True,
            max_parallel_tasks=1,
            max_task_steps=3,
            executor_kind="thread",
        )
        assert isinstance(agent, ParallelCodeAgent)
        result = agent.run("What is 17 * 23?")
        assert result == 17 * 23
