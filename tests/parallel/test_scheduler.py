# coding=utf-8

"""Tests for :mod:`smolagents.parallel.scheduler` with fake workers."""
from __future__ import annotations

import time
from dataclasses import dataclass

import pytest

from smolagents.parallel.events import (
    TaskCompletedEvent,
)
from smolagents.parallel.scheduler import (
    ParallelScheduler,
    SchedulerFailure,
    TaskResult,
)
from smolagents.parallel.task_graph import Task, TaskGraph


@dataclass
class FakeSpec:
    task_id: str
    sleep_s: float = 0.0
    fail: bool = False
    output: object = None


def _fake_worker(spec: FakeSpec) -> TaskResult:
    if spec.sleep_s:
        time.sleep(spec.sleep_s)
    if spec.fail:
        raise RuntimeError(f"simulated failure for {spec.task_id}")
    return TaskResult(task_id=spec.task_id, output=spec.output, action_steps=[])


def _spec_builder(sleep_map: dict[str, float] | None = None, fail_set: set[str] | None = None):
    sleep_map = sleep_map or {}
    fail_set = fail_set or set()

    def _build(task: Task, graph: TaskGraph) -> FakeSpec:
        return FakeSpec(
            task_id=task.id,
            sleep_s=sleep_map.get(task.id, 0.0),
            fail=task.id in fail_set,
            output=f"out:{task.id}",
        )

    return _build


def _make_graph(edges: list[tuple[str, list[str]]]) -> TaskGraph:
    g = TaskGraph()
    g.add_tasks([Task(id=tid, goal=f"goal {tid}", dependencies=list(deps)) for tid, deps in edges])
    return g


class TestSchedulerThreaded:
    def test_simple_run_completes_all_tasks(self):
        graph = _make_graph([("a", []), ("b", [])])

        with ParallelScheduler(
            build_spec=_spec_builder(),
            worker=_fake_worker,
            max_workers=2,
            executor_kind="thread",
        ) as scheduler:
            sched_events = scheduler.submit_ready(graph)
            started_events = list(scheduler.drain_started_events())
            completions = []
            while scheduler.running_count:
                completions.extend(scheduler.wait_for_any(timeout=5.0))

        assert {e.task.id for e in sched_events} == {"a", "b"}
        assert {e.task.id for e in started_events} == {"a", "b"}
        assert len(completions) == 2
        assert all(isinstance(e, TaskCompletedEvent) for e in completions)
        # The scheduler transitions tasks to RUNNING; completion on the
        # graph is the outer agent's responsibility — see the agent
        # integration test.
        assert {e.task.id for e in completions} == {"a", "b"}
        assert graph.get("a").status.value == "running"
        assert graph.get("b").status.value == "running"

    def test_dependent_task_waits_for_parent(self):
        graph = _make_graph([("a", []), ("b", ["a"])])

        with ParallelScheduler(
            build_spec=_spec_builder(),
            worker=_fake_worker,
            max_workers=4,
            executor_kind="thread",
        ) as scheduler:
            first_batch = scheduler.submit_ready(graph)
            assert [e.task.id for e in first_batch] == ["a"]

            scheduler.wait_for_any(timeout=5.0)
            # After a is done, but we haven't marked it completed on the
            # graph yet (agent does that); simulate the agent step:
            graph.mark_completed("a", result="done-a")

            second_batch = scheduler.submit_ready(graph)
            assert [e.task.id for e in second_batch] == ["b"]
            scheduler.wait_for_any(timeout=5.0)

    def test_pool_capacity_respected(self):
        graph = _make_graph([("a", []), ("b", []), ("c", [])])

        with ParallelScheduler(
            build_spec=_spec_builder(sleep_map={"a": 0.2, "b": 0.2, "c": 0.2}),
            worker=_fake_worker,
            max_workers=2,
            executor_kind="thread",
        ) as scheduler:
            first = scheduler.submit_ready(graph)
            assert len(first) == 2  # third blocked by pool capacity
            assert scheduler.running_count == 2
            scheduler.wait_for_any(timeout=5.0)
            second = scheduler.submit_ready(graph)
            assert len(second) == 1

    def test_failure_surfaces_scheduler_failure(self):
        graph = _make_graph([("a", [])])

        with ParallelScheduler(
            build_spec=_spec_builder(fail_set={"a"}),
            worker=_fake_worker,
            max_workers=1,
            executor_kind="thread",
        ) as scheduler:
            scheduler.submit_ready(graph)
            with pytest.raises(SchedulerFailure) as exc:
                scheduler.wait_for_any(timeout=5.0)
            assert exc.value.task_id == "a"

    def test_cannot_submit_outside_context(self):
        graph = _make_graph([("a", [])])
        scheduler = ParallelScheduler(
            build_spec=_spec_builder(),
            worker=_fake_worker,
            executor_kind="thread",
        )
        with pytest.raises(RuntimeError):
            scheduler.submit_ready(graph)
