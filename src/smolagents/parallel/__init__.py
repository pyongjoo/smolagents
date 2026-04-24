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
"""Parallel execution primitives for smolagents.

Opt-in module. See ``ASYNC_EXECUTION_DESIGN.md`` at the repo root for
the design rationale.
"""
from smolagents.parallel.agent import (
    ParallelCodeAgent,
    TaskWorkerSpec,
    run_task_worker,
)
from smolagents.parallel.events import (
    ParallelEvent,
    PlanningTriggeredEvent,
    TaskCompletedEvent,
    TaskScheduledEvent,
    TaskStartedEvent,
)
from smolagents.parallel.planner import (
    NextPlanningTrigger,
    ParallelPlanner,
    PlanningResult,
    load_default_planning_prompt,
)
from smolagents.parallel.scheduler import (
    ParallelScheduler,
    SchedulerFailure,
    TaskResult,
)
from smolagents.parallel.task_graph import Task, TaskGraph, TaskStatus


__all__ = [
    "NextPlanningTrigger",
    "ParallelCodeAgent",
    "ParallelEvent",
    "ParallelPlanner",
    "ParallelScheduler",
    "PlanningResult",
    "PlanningTriggeredEvent",
    "SchedulerFailure",
    "Task",
    "TaskCompletedEvent",
    "TaskGraph",
    "TaskResult",
    "TaskScheduledEvent",
    "TaskStartedEvent",
    "TaskStatus",
    "TaskWorkerSpec",
    "load_default_planning_prompt",
    "run_task_worker",
]
