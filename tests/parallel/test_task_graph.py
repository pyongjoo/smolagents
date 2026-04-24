# coding=utf-8

# Licensed under the Apache License, Version 2.0 (the "License").
"""Unit tests for :mod:`smolagents.parallel.task_graph`."""
from __future__ import annotations

import pytest
import yaml

from smolagents.parallel.task_graph import Task, TaskGraph, TaskStatus


def make_task(task_id: str, *, deps: list[str] | None = None, goal: str = "do stuff") -> Task:
    return Task(id=task_id, goal=goal, dependencies=list(deps or []))


class TestTaskGraphBasics:
    def test_empty_graph_is_done(self):
        g = TaskGraph()
        assert g.is_done()
        assert g.ready_tasks() == []
        assert g.completed_tasks() == []

    def test_add_tasks_preserves_insertion_order(self):
        g = TaskGraph()
        g.add_tasks([make_task("a"), make_task("b")])
        g.add_tasks([make_task("c", deps=["a"])])
        assert [t.id for t in g.all_tasks()] == ["a", "b", "c"]

    def test_duplicate_ids_rejected(self):
        g = TaskGraph()
        g.add_tasks([make_task("a")])
        with pytest.raises(ValueError):
            g.add_tasks([make_task("a")])

    def test_same_batch_duplicate_ids_rejected(self):
        g = TaskGraph()
        with pytest.raises(ValueError):
            g.add_tasks([make_task("a"), make_task("a")])

    def test_unknown_dep_rejected(self):
        g = TaskGraph()
        with pytest.raises(ValueError):
            g.add_tasks([make_task("a", deps=["nope"])])

    def test_self_dep_rejected(self):
        g = TaskGraph()
        with pytest.raises(ValueError):
            g.add_tasks([make_task("a", deps=["a"])])

    def test_failed_batch_is_atomic(self):
        g = TaskGraph()
        with pytest.raises(ValueError):
            g.add_tasks([make_task("a"), make_task("b", deps=["unknown"])])
        assert len(g) == 0


class TestReadinessAndTransitions:
    def test_ready_tasks_respects_dependencies(self):
        g = TaskGraph()
        g.add_tasks([make_task("a"), make_task("b", deps=["a"])])
        ready = g.ready_tasks()
        assert [t.id for t in ready] == ["a"]

    def test_mark_running_then_completed(self):
        g = TaskGraph()
        g.add_tasks([make_task("a"), make_task("b", deps=["a"])])
        g.mark_running("a")
        assert g.get("a").status == TaskStatus.RUNNING
        assert g.ready_tasks() == []
        g.mark_completed("a", result=42)
        assert g.get("a").status == TaskStatus.COMPLETED
        assert g.get("a").result == 42
        assert [t.id for t in g.ready_tasks()] == ["b"]

    def test_illegal_transition_rejected(self):
        g = TaskGraph()
        g.add_tasks([make_task("a")])
        with pytest.raises(ValueError):
            g.mark_completed("a", result=None)  # must be RUNNING first

    def test_is_done_after_all_complete(self):
        g = TaskGraph()
        g.add_tasks([make_task("a"), make_task("b", deps=["a"])])
        g.mark_running("a")
        g.mark_completed("a", result=1)
        g.mark_running("b")
        g.mark_completed("b", result=2)
        assert g.is_done()

    def test_is_not_done_with_failed_intermediate_but_pending_children(self):
        # In this iteration, a failed parent leaves children PENDING
        # forever; is_done() returns True when nothing is ready/running.
        g = TaskGraph()
        g.add_tasks([make_task("a"), make_task("b", deps=["a"])])
        g.mark_running("a")
        g.mark_failed("a", "boom")
        # b is still pending, but no way to progress -> is_done True
        assert g.is_done()
        assert g.failed_tasks() and g.failed_tasks()[0].id == "a"


class TestYamlSnapshot:
    def test_yaml_roundtrip_structure(self):
        g = TaskGraph()
        g.add_tasks(
            [
                Task(id="t1", goal="g1", dependencies=[], expected_runtime_s=5, resources=["./x"]),
                Task(id="t2", goal="g2", dependencies=["t1"], expected_runtime_s=10),
            ]
        )
        g.mark_running("t1")
        g.mark_completed("t1", result="hello world")

        snapshot = yaml.safe_load(g.to_yaml())
        assert list(snapshot.keys()) == ["tasks"]
        assert len(snapshot["tasks"]) == 2

        t1, t2 = snapshot["tasks"]
        assert t1["id"] == "t1"
        assert t1["status"] == "completed"
        assert t1["resources"] == ["./x"]
        assert "actual_runtime_s" in t1
        assert t1["result_summary"] == "hello world"

        assert t2["status"] == "pending"
        assert t2["dependencies"] == ["t1"]

    def test_result_summary_truncated(self):
        g = TaskGraph()
        g.add_tasks([make_task("t1")])
        g.mark_running("t1")
        g.mark_completed("t1", result="x" * 1000)
        snap = yaml.safe_load(g.to_yaml())
        assert len(snap["tasks"][0]["result_summary"]) <= 301  # 300 + ellipsis
