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
"""Lifecycle events emitted by the parallel execution pipeline.

Only the three task-lifecycle events plus a planning-trigger event are
defined here, matching the design doc §9.3. ``TaskProgressEvent``,
``TaskFailedEvent``, and ``TaskCancelledEvent`` will land alongside the
follow-up progress-reporting and failure-handling slices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from smolagents.parallel.planner import NextPlanningTrigger
    from smolagents.parallel.scheduler import TaskResult
    from smolagents.parallel.task_graph import Task


__all__ = [
    "TaskScheduledEvent",
    "TaskStartedEvent",
    "TaskCompletedEvent",
    "PlanningTriggeredEvent",
    "ParallelEvent",
]


@dataclass
class TaskScheduledEvent:
    """Emitted after a task has been submitted to the process pool."""

    task: "Task"


@dataclass
class TaskStartedEvent:
    """Emitted when a worker has started executing a task.

    In the current implementation ``TaskScheduledEvent`` and
    ``TaskStartedEvent`` are emitted back-to-back; they are kept
    distinct so the upcoming progress-reporting slice can make the
    started-event fire only when the worker actually picks the task
    up.
    """

    task: "Task"


@dataclass
class TaskCompletedEvent:
    """Emitted when a task finished successfully (status = COMPLETED).

    ``result`` carries the full worker return value (output, trace,
    token usage, etc.) so the outer agent can render the task's trace
    atomically and merge token counts into the monitor.
    """

    task: "Task"
    result: "TaskResult | None" = None


@dataclass
class PlanningTriggeredEvent:
    """Emitted right before the agent invokes the planner again."""

    trigger: "NextPlanningTrigger"


ParallelEvent = (
    TaskScheduledEvent
    | TaskStartedEvent
    | TaskCompletedEvent
    | PlanningTriggeredEvent
)
