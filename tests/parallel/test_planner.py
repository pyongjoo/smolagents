# coding=utf-8

"""Unit tests for :mod:`smolagents.parallel.planner`."""
from __future__ import annotations

import pytest

from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import AgentLogger, LogLevel, TokenUsage
from smolagents.parallel.planner import (
    NextPlanningTrigger,
    ParallelPlanner,
)
from smolagents.parallel.task_graph import Task, TaskGraph
from smolagents.utils import AgentParsingError


def make_logger() -> AgentLogger:
    return AgentLogger(level=LogLevel.OFF)


class FakeModel:
    """Minimal stand-in for :class:`smolagents.models.Model` in planner tests."""

    def __init__(self, contents: list[str]):
        self._contents = list(contents)
        self.calls = 0

    def generate(self, messages, **kwargs):
        self.calls += 1
        content = self._contents.pop(0)
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        )


VALID_YAML = """
reasoning: |
  Two independent tasks then an aggregator.
new_tasks:
  - id: task_1
    goal: Do the first thing.
    dependencies: []
    expected_runtime_s: 10
    resources: []
  - id: task_2
    goal: Do the second thing.
    dependencies: []
    expected_runtime_s: 10
    resources: []
  - id: task_3
    goal: Combine results and call final_answer.
    dependencies: [task_1, task_2]
    expected_runtime_s: 5
    resources: []
next_planning:
  kind: after_task
  task_id: task_1
"""


FENCED_YAML = "```yaml\n" + VALID_YAML.strip() + "\n```"


INVALID_THEN_VALID = ("this is not yaml: [::", VALID_YAML)


class TestHappyPath:
    def test_parses_valid_yaml(self):
        model = FakeModel([VALID_YAML])
        planner = ParallelPlanner(model=model, logger=make_logger())
        result = planner.plan(task="demo", memory=None, graph=TaskGraph())

        assert [t.id for t in result.new_tasks] == ["task_1", "task_2", "task_3"]
        assert result.new_tasks[2].dependencies == ["task_1", "task_2"]
        assert result.next_trigger == NextPlanningTrigger(kind="after_task", task_id="task_1")
        assert result.planning_step.new_task_ids == ["task_1", "task_2", "task_3"]
        assert result.planning_step.next_trigger["kind"] == "after_task"

    def test_tolerates_markdown_fence(self):
        model = FakeModel([FENCED_YAML])
        planner = ParallelPlanner(model=model, logger=make_logger())
        result = planner.plan(task="demo", memory=None, graph=TaskGraph())
        assert len(result.new_tasks) == 3

    def test_retries_once_on_parse_error(self):
        model = FakeModel(list(INVALID_THEN_VALID))
        planner = ParallelPlanner(model=model, logger=make_logger(), max_retries=1)
        result = planner.plan(task="demo", memory=None, graph=TaskGraph())
        assert model.calls == 2
        assert len(result.new_tasks) == 3

    def test_gives_up_after_retries_exhausted(self):
        # Strings that parse as scalars (not mappings) trigger our
        # "must be a YAML mapping" guard.
        model = FakeModel(["just a string", "another scalar"])
        planner = ParallelPlanner(model=model, logger=make_logger(), max_retries=1)
        with pytest.raises(AgentParsingError):
            planner.plan(task="demo", memory=None, graph=TaskGraph())


class TestValidation:
    def test_rejects_unknown_dependency(self):
        bad_yaml = """
reasoning: "x"
new_tasks:
  - id: task_1
    goal: g
    dependencies: [does_not_exist]
    expected_runtime_s: 1
    resources: []
next_planning:
  kind: never
"""
        model = FakeModel([bad_yaml, bad_yaml])
        planner = ParallelPlanner(model=model, logger=make_logger(), max_retries=1)
        with pytest.raises(AgentParsingError):
            planner.plan(task="demo", memory=None, graph=TaskGraph())

    def test_rejects_duplicate_id_with_existing_graph(self):
        graph = TaskGraph()
        graph.add_tasks([Task(id="task_1", goal="pre", dependencies=[])])

        dup_yaml = """
reasoning: ""
new_tasks:
  - id: task_1
    goal: dup
    dependencies: []
    expected_runtime_s: 1
    resources: []
next_planning:
  kind: never
"""
        model = FakeModel([dup_yaml, dup_yaml])
        planner = ParallelPlanner(model=model, logger=make_logger(), max_retries=1)
        with pytest.raises(AgentParsingError):
            planner.plan(task="demo", memory=None, graph=graph)

    def test_rejects_trigger_to_unknown_task(self):
        bad_yaml = """
reasoning: ""
new_tasks: []
next_planning:
  kind: after_task
  task_id: ghost
"""
        model = FakeModel([bad_yaml, bad_yaml])
        planner = ParallelPlanner(model=model, logger=make_logger(), max_retries=1)
        with pytest.raises(AgentParsingError):
            planner.plan(task="demo", memory=None, graph=TaskGraph())

    def test_incremental_dependency_on_existing_task(self):
        graph = TaskGraph()
        graph.add_tasks([Task(id="task_1", goal="pre", dependencies=[])])
        graph.mark_running("task_1")
        graph.mark_completed("task_1", result="done")

        yaml_doc = """
reasoning: ""
new_tasks:
  - id: task_2
    goal: depends on task_1
    dependencies: [task_1]
    expected_runtime_s: 5
    resources: []
next_planning:
  kind: never
"""
        model = FakeModel([yaml_doc])
        planner = ParallelPlanner(model=model, logger=make_logger())
        result = planner.plan(task="demo", memory=None, graph=graph)
        assert result.new_tasks[0].dependencies == ["task_1"]
