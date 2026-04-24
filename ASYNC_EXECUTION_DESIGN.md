# Async / Parallel LLM Execution Design for smolagents

## 1. Motivation

The current smolagents framework executes agents in a strictly sequential
Thought → Code → Observation loop. Every step blocks on a single LLM call,
followed by a single Python execution. This is simple and correct, but
leaves significant time on the table when a task naturally decomposes into
independent sub-tasks (e.g. fetching multiple web pages, running multiple
searches, analyzing several files, etc.).

This design introduces an **opt-in parallel execution mode** that:

- Lets a *planning* LLM call decompose work into a DAG of tasks, where
  each task is described only by a **high-level goal** — not by concrete
  code or tool calls.
- Runs independent tasks concurrently in separate processes. Each task
  internally runs its **own Thought → Code → Observation ReAct loop**
  (with its own LLM calls) to elaborate the steps needed to achieve its
  goal, exactly like today's `CodeAgent` does for a full run.
- Re-invokes planning incrementally as tasks complete, based on
  hints produced by the previous planning call.
- Keeps the existing sequential loop fully intact when the flag is off.

There are therefore **two levels of LLM calls**:

1. **Planner-level** calls — coarse, infrequent, operate on the task
   graph, produce new task *goals*.
2. **Task-level** calls — fine-grained, one per ReAct step inside a
   running task, produce concrete code to execute.

The planner never writes code or prescribes tool sequences; that job
belongs to the task-level LLM.

Several pieces are **explicitly out of scope** for this first iteration
and tracked in §14:

- **Resource conflict detection.** Tasks declare the resources they
  *expect* to modify, but the scheduler does not yet enforce mutual
  exclusion.
- **Progress reporting.** Only task-lifecycle events
  (scheduled / started / completed) are emitted; mid-task progress is
  not streamed live.
- **Failure handling.** The first task failure raises
  `AgentExecutionError`; there is no retry, cascading cancellation, or
  failure-driven replanning yet.

These are deferred deliberately so the initial slice stays small and
reviewable.

## 2. Design Principles

1. **Additive, not invasive.** The new mechanism lives in new modules and
   new classes; existing `MultiStepAgent`, `CodeAgent`, and
   `ToolCallingAgent` code paths are untouched unless the new flag is set.
2. **Flag-gated.** A single constructor flag (e.g.
   `parallel_execution=True`) or a new `run(..., parallel=True)` kwarg
   selects the parallel path. Default behavior is unchanged.
3. **Incremental planning.** There is no distinction between "initial"
   and "update" planning. The first call simply runs against an empty
   task graph. Planning is always incremental: it may add new tasks,
   but never mutates completed ones.
4. **Graph-structured state.** A standalone module owns the task graph;
   agents, planners, and executors interact with it through a narrow API.
5. **Reuse existing memory.** Completed tasks produce standard
   `ActionStep` entries that append to `AgentMemory`, so downstream
   tooling (replay, monitoring, serialization) keeps working.
6. **Goals, not scripts.** Planning output specifies *what* each task
   should accomplish, never *how*. A task's internal ReAct loop is
   responsible for figuring out the concrete thoughts, code, and tool
   calls required to meet its goal.

## 3. High-Level Architecture

```
                          ┌──────────────────────┐
                          │  ParallelCodeAgent   │
                          │   (new subclass)     │
                          └──────────┬───────────┘
                                     │ owns
            ┌────────────────────────┼─────────────────────────┐
            ▼                        ▼                         ▼
     ┌─────────────┐          ┌──────────────┐          ┌──────────────┐
     │ TaskGraph   │◀────────▶│   Planner    │          │  Scheduler   │
     │  (state)    │          │ (LLM calls)  │          │ (ProcessPool)│
     └─────────────┘          └──────────────┘          └──────┬───────┘
            ▲                                                  │
            │                   results / failures             │
            └──────────────────────────────────────────────────┘
```

- **TaskGraph** – owns the DAG of pending / running / completed / failed
  tasks. Produces a YAML view for prompting.
- **Planner** – wraps the planning LLM call. Consumes `AgentMemory` +
  `TaskGraph` and returns new tasks + a trigger describing when to be
  called again.
- **Scheduler** – maintains a process pool, decides which ready tasks to
  launch, collects results, and writes them back into the `TaskGraph`
  and `AgentMemory`.
- **ParallelCodeAgent** – thin subclass of `CodeAgent` that orchestrates
  the three pieces above when the parallel flag is on.

### 3.1 Two-Level LLM Model

It is important to keep the two levels distinct:

```
 Planner LLM (coarse, infrequent)
 └─ produces Task(goal="Summarize paper A", deps=[...], ...)
     │
     ▼
 Task execution (one process per task)
   └─ Inner CodeAgent ReAct loop:
        Thought  ──▶  Code  ──▶  Observation   (Task-level LLM call, step 1)
        Thought  ──▶  Code  ──▶  Observation   (Task-level LLM call, step 2)
        ...
        final_answer(...)  ──▶  result bubbles up to the graph
```

- The planner **never** emits code, tool-call sequences, or inner-step
  plans. Its output is purely a list of goals, dependencies, estimated
  runtimes, and declarative resources.
- Each task is itself a miniature `CodeAgent`: it runs a bounded
  Thought → Code → Observation loop to *work out* how to meet its goal.
  Every iteration of that loop makes its own LLM call — these are the
  calls that actually run in parallel across the process pool.
- When a task finishes, its inner `ActionStep`s are merged back into the
  outer `AgentMemory` under a wrapping `ParallelTaskStep`, so the full
  thought/code/observation trace is preserved and visible to the next
  planner call (in summarized form).

This separation of concerns is what lets planning stay cheap and
globally-aware while still letting each task reason flexibly about its
own execution.

## 4. New Files

To respect the "new file over in-place update" preference, all non-trivial
new logic lives in dedicated modules:

| File | Purpose |
| --- | --- |
| `src/smolagents/parallel/__init__.py` | Public re-exports. |
| `src/smolagents/parallel/task_graph.py` | `Task`, `TaskStatus`, `TaskGraph`. |
| `src/smolagents/parallel/planner.py` | `ParallelPlanner` wrapping the LLM planning call. |
| `src/smolagents/parallel/scheduler.py` | `ParallelScheduler` using `ProcessPoolExecutor`. |
| `src/smolagents/parallel/events.py` | Event dataclasses for progress reporting. |
| `src/smolagents/parallel/agent.py` | `ParallelCodeAgent` (and optionally `ParallelToolCallingAgent`). |
| `src/smolagents/prompts/parallel_planning.yaml` | Planning prompt with examples. |
| `tests/parallel/test_task_graph.py` | Unit tests for the graph. |
| `tests/parallel/test_planner.py` | Tests with a mock model. |
| `tests/parallel/test_scheduler.py` | Tests with fake tasks. |
| `tests/parallel/test_parallel_agent.py` | End-to-end tests with a mock model. |

## 5. Touch-Points in Existing Code

Kept deliberately minimal:

1. `src/smolagents/__init__.py` – re-export `ParallelCodeAgent` and the
   parallel submodule.
2. `src/smolagents/memory.py` – add two small dataclasses for task-level
   memory entries (see §8). No changes to existing classes.
3. `src/smolagents/agents.py` – *optional*, only to expose a
   `parallel_execution` kwarg on `CodeAgent` that dispatches to
   `ParallelCodeAgent`. If we prefer zero changes here, users import
   `ParallelCodeAgent` directly.

Everything else (ReAct loop, python executor, tools, prompt templates for
the classical agent) is untouched.

## 6. Task Graph Module

### 6.1 Data Model

```python
class TaskStatus(str, Enum):
    PENDING   = "pending"     # Not yet runnable (deps unmet) or waiting in queue
    READY     = "ready"       # Dependencies satisfied, not yet scheduled
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"      # Populated only to surface the terminal error

@dataclass
class Task:
    id: str                                 # unique within a run, e.g. "task_3"
    goal: str                               # natural language description
    dependencies: list[str]                 # parent task ids
    expected_runtime_s: float | None        # LLM estimate
    resources: list[str]                    # declarative only, not enforced yet
    status: TaskStatus = TaskStatus.PENDING
    result: Any | None = None
    error: str | None = None
    created_at: float = ...
    started_at: float | None = None
    finished_at: float | None = None
    memory_step: ActionStep | None = None   # populated on completion
```

`CANCELLED` and a `progress` field are intentionally omitted — they
will be added together with failure handling and progress reporting
(§14).

### 6.2 `TaskGraph` API

```python
class TaskGraph:
    def add_tasks(self, tasks: Iterable[Task]) -> None
    def get(self, task_id: str) -> Task
    def ready_tasks(self) -> list[Task]              # deps completed, status==PENDING/READY
    def running_tasks(self) -> list[Task]
    def pending_tasks(self) -> list[Task]            # not yet ready (deps unmet)
    def completed_tasks(self) -> list[Task]
    def failed_tasks(self) -> list[Task]
    def is_done(self) -> bool                        # no pending/ready/running left
    def mark_running(self, task_id: str) -> None
    def mark_completed(self, task_id: str, result, memory_step) -> None
    def mark_failed(self, task_id: str, error: str) -> None
    def to_yaml(self) -> str                         # full snapshot for prompts
```

`mark_failed` exists so a terminal error can be recorded on the graph
for logging, but in this iteration the outer agent raises immediately
on the first failure and does not attempt to make further progress.

Invariants:

- Task ids are unique and never reused.
- Dependencies must point to tasks already in the graph.
- Completed tasks are immutable.
- Status transitions only along the state machine:
  `PENDING → READY → RUNNING → {COMPLETED | FAILED}`.

### 6.3 YAML Snapshot Format

This is the canonical shape fed back to the planner and used in logs:

```yaml
tasks:
  - id: task_1
    goal: Search the web for recent papers on X.
    status: completed
    dependencies: []
    expected_runtime_s: 15
    actual_runtime_s: 12.4
    resources: []
    result_summary: "Found 5 papers, ids stored in `papers_list`."
  - id: task_2
    goal: Download PDF for each paper in `papers_list`.
    status: running
    dependencies: [task_1]
    expected_runtime_s: 40
    resources: ["./downloads/"]
  - id: task_3
    goal: Summarize each downloaded paper.
    status: pending
    dependencies: [task_2]
    expected_runtime_s: 60
    resources: []
```

Only small, plan-relevant fields are serialized; raw tool outputs stay in
memory but aren't shipped back to the planner by default.

## 7. Planner Module

### 7.1 Responsibility

A single method, `plan()`, performs one incremental planning LLM call.

**The planner's only output is task *goals*, not task *implementations*.**
Each goal is a natural-language description of what the task should
achieve. The concrete Thought → Code → Observation steps required to
meet that goal are decided at task execution time by the task's own
inner ReAct loop (see §9.2), which makes its own LLM calls. The planner
must not embed code, tool names, or step-by-step instructions in a
task's `goal` field.

```python
class ParallelPlanner:
    def __init__(self, model, prompt_templates, logger): ...

    def plan(
        self,
        task: str,
        memory: AgentMemory,
        graph: TaskGraph,
    ) -> PlanningResult: ...
```

`PlanningResult` is:

```python
@dataclass
class PlanningResult:
    new_tasks: list[Task]
    next_trigger: NextPlanningTrigger
    reasoning: str
    raw_output: str
    planning_step: PlanningStep   # for memory

@dataclass
class NextPlanningTrigger:
    kind: Literal["after_task", "after_n_completions", "on_failure", "never"]
    task_id: str | None = None
    n: int | None = None
```

There is no separate "initial" branch. The first call simply receives an
empty `TaskGraph` and produces the initial tasks. Subsequent calls see the
current graph (completed, running, pending) plus the updated memory and
produce *additional* tasks (or none, if planning is done).

Rules enforced in code:

- The planner must not reference task ids that don't already exist or
  aren't being created in this call.
- Dependencies of new tasks must be existing tasks (in any non-cancelled
  state) or other new tasks from the same call.
- If the LLM returns malformed YAML, we raise `AgentParsingError` and the
  outer agent falls back to a single retry with a stricter "return YAML
  only" reminder before giving up.

### 7.2 Prompt Template

Location: `src/smolagents/prompts/parallel_planning.yaml`.

Must include **worked examples**, per the user's request. Sketch:

```yaml
planning: |-
  You are a planning assistant for a parallel agent runtime.
  You are given:
  - The user's overall task.
  - The agent's memory so far (thought/code/observation history).
  - The current task graph (completed, running, pending).

  Your job is to propose *new* tasks that should be added to the graph.
  You may propose zero tasks if nothing more is needed right now. You must
  also say when you should be called again.

  Rules:
  1. Output valid YAML, and nothing else.
  2. Each new task must have:
     - id (unique, new)
     - goal (concise but complete natural-language description of *what*
       the task should accomplish — NEVER include code, tool names, or
       step-by-step instructions; the task itself will figure that out
       via its own Thought/Code/Observation loop)
     - dependencies (list of existing or newly-proposed task ids; [] if none)
     - expected_runtime_s (your best numeric guess, in seconds)
     - resources (list of string identifiers for things this task will
       read or modify, e.g. file paths, variable names; [] if none)
  3. `next_planning` tells the runtime when to call you again:
     - kind: after_task | after_n_completions | on_failure | never
     - task_id (if kind == after_task)
     - n (if kind == after_n_completions)

  Prefer wide, independent subtasks when possible — tasks without
  dependencies on each other can run in parallel. Keep each goal
  focused enough that a single sub-agent can complete it within a
  handful of Thought → Code → Observation iterations.

  ---
  ## Example 1 — Initial planning, empty graph

  User task: "Summarize the top 3 news stories about AI today."

  Current graph:
  tasks: []

  Output:
  ```yaml
  reasoning: |
    I need to search for today's AI news, then for each of the top 3
    stories fetch the article and summarize it. Fetching and summarizing
    each story is independent, so they can run in parallel once the search
    is done.
  new_tasks:
    - id: task_1
      goal: Search the web for today's top AI news stories and return
        the top 3 URLs.
      dependencies: []
      expected_runtime_s: 10
      resources: []
    - id: task_2
      goal: Fetch and summarize the first URL from task_1.
      dependencies: [task_1]
      expected_runtime_s: 30
      resources: []
    - id: task_3
      goal: Fetch and summarize the second URL from task_1.
      dependencies: [task_1]
      expected_runtime_s: 30
      resources: []
    - id: task_4
      goal: Fetch and summarize the third URL from task_1.
      dependencies: [task_1]
      expected_runtime_s: 30
      resources: []
  next_planning:
    kind: after_task
    task_id: task_1
  ```

  ---
  ## Example 2 — Incremental planning, some tasks done

  User task: same as above.

  Current graph:
  ```yaml
  tasks:
    - id: task_1
      status: completed
      result_summary: "URLs: [url_a, url_b, url_c]"
    - id: task_2
      status: running
    - id: task_3
      status: running
    - id: task_4
      status: running
  ```

  Output:
  ```yaml
  reasoning: |
    Search succeeded; summarization tasks are running. Once all three
    summaries are back I should produce a final combined answer.
  new_tasks:
    - id: task_5
      goal: Combine the three summaries from task_2, task_3, task_4
        into a single answer and call final_answer with it.
      dependencies: [task_2, task_3, task_4]
      expected_runtime_s: 15
      resources: []
  next_planning:
    kind: never
  ```

  ---
  Now produce your YAML for:

  User task:
  ```
  {{task}}
  ```

  Agent memory (summary):
  ```
  {{memory_summary}}
  ```

  Current task graph:
  ```yaml
  {{graph_yaml}}
  ```
```

All `{{...}}` slots are populated by the existing Jinja2 renderer.

## 8. Memory Integration

Two new memory step dataclasses in `memory.py`:

```python
@dataclass
class ParallelPlanningStep(PlanningStep):
    graph_snapshot_yaml: str
    new_task_ids: list[str]
    next_trigger: dict

@dataclass
class ParallelTaskStep(ActionStep):
    task_id: str
    task_goal: str
    dependencies: list[str]
    expected_runtime_s: float | None
    resources: list[str]
```

`ParallelTaskStep` inherits all ActionStep fields (code, observation,
token usage, etc.), so it plugs straight into `AgentMemory.steps` without
breaking any existing consumers. Replay and serialization just work.

## 9. Scheduler Module

### 9.1 Responsibilities

- Keep a `ProcessPoolExecutor` with configurable `max_workers`.
- On each tick:
  1. Ask the graph for ready tasks.
  2. Launch up to `max_workers` of them, moving them to `RUNNING`.
  3. Poll futures; for each that has finished, update the graph and
     emit events.
- Provide a blocking `wait_for_next_event(timeout)` for the agent loop.

In this first iteration the scheduler emits only three lifecycle
events per task — `TaskScheduledEvent`, `TaskStartedEvent`, and
`TaskCompletedEvent` — and does **not** report mid-task progress or
handle failures gracefully. See §9.3 and §9.4 for what is deferred
and why.

### 9.2 Running a Task

Each task runs in a worker process as a **mini CodeAgent that executes
its own bounded Thought → Code → Observation loop**. This is where
task-level LLM calls happen: every ReAct iteration inside the worker
makes one LLM call, and because workers run in parallel across the
process pool, these task-level calls run concurrently.

Concretely, the worker is configured with:

- A fresh `LocalPythonExecutor` seeded with:
  - The parent agent's tools and managed agents.
  - A read-only snapshot of the parent's Python state.
  - The results of this task's completed dependencies, bound to
    variables named after the dependency task ids (e.g. `task_1`).
- A prompt constructed from:
  - The system prompt of the parent agent (so the inner loop knows the
    available tools and code conventions).
  - A short prefix: "You are executing sub-task `{id}`. Goal: …",
    carrying only the planner's high-level goal — no concrete steps.
  - The parent memory (summary_mode, so lightweight).
- `max_steps` bounded, typically much lower than the outer agent's cap
  (tasks are expected to be focused, so a handful of ReAct iterations
  should suffice).

The worker terminates when either `final_answer(...)` is called inside
the loop or `max_steps` is reached. It then returns a `TaskResult`
containing:

- Final answer / return value (what `final_answer` was called with, or
  the last expression value).
- The full list of captured inner `ActionStep`s, serialized for
  cross-process transport. These get wrapped into a `ParallelTaskStep`
  (see §8) when merged back into the outer `AgentMemory`, so the
  planner on subsequent rounds can see a summary of *how* the task
  was solved, not just its final answer.
- Logs and observations (truncated).
- Token usage (task-level tokens; tracked separately from planner
  tokens for observability).

### 9.3 Progress Reporting (Deferred)

**Deferred to a future iteration.** For now, the scheduler only reports
*lifecycle* events — when a task is scheduled, started, and completed —
but does not surface mid-task progress. The worker's inner ReAct steps
are captured and returned in `TaskResult` on completion, so the full
trace is still available after the fact; it is just not streamed live.

Events present in this iteration live in `parallel/events.py`:

```python
@dataclass
class TaskScheduledEvent:    task: Task
@dataclass
class TaskStartedEvent:      task: Task
@dataclass
class TaskCompletedEvent:    task: Task
@dataclass
class PlanningTriggeredEvent: trigger: NextPlanningTrigger
```

These extend the existing `StreamEvent` union so streaming consumers
can pattern-match on them. `TaskProgressEvent`, `TaskFailedEvent`, and
`TaskCancelledEvent` will be added when progress reporting and failure
handling land (see §14).

### 9.4 Failure Handling (Deferred)

**Deferred to a future iteration.** In this first cut the behaviour on
any task failure is deliberately minimal:

1. If a task raises, or its worker process crashes, the scheduler
   records the exception on the `Task` (status stays conceptually
   "failed" for logs) and stops submitting new work.
2. Any dependent tasks simply never become ready, so they remain
   `PENDING` in the final graph snapshot.
3. The outer `ParallelCodeAgent.run(...)` surfaces the first failure by
   raising `AgentExecutionError`, matching how a failing step is
   handled in the sequential agent today.

Retry budgets, dependency-cascade cancellation, re-planning on failure,
and crash-vs-exception differentiation are explicitly out of scope for
now and tracked in §14. The planner prompt also does not need a
failure-recovery example yet; it will be added when failure handling
is introduced.

## 10. Parallel Agent (`ParallelCodeAgent`)

A subclass of `CodeAgent` that overrides `_run_stream`.

```python
class ParallelCodeAgent(CodeAgent):
    def __init__(
        self,
        *args,
        max_parallel_tasks: int = 4,
        max_task_steps: int = 6,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.planner   = ParallelPlanner(self.model, self.prompt_templates, self.logger)
        self.graph     = TaskGraph()
        self.scheduler = ParallelScheduler(
            max_workers=max_parallel_tasks,
            build_worker=self._build_worker_spec,
        )
        ...
```

Retry / failure-handling knobs are intentionally omitted here; they
will be introduced together with failure handling (§14).

### 10.1 Execution Loop (replaces `_run_stream` under the flag)

Pseudocode:

```python
def _run_stream(self, task, max_steps, images):
    # Always start with one planning call against an empty graph.
    yield from self._run_planning(task)

    while not self.graph.is_done():
        # Launch whatever is ready.
        for t in self.graph.ready_tasks():
            self.scheduler.submit(t)
            yield TaskScheduledEvent(t)

        # Block until something happens (completion, failure, progress).
        for event in self.scheduler.drain_events(timeout=...):
            yield event
            self._apply_event_to_graph(event)

        # Decide whether to re-plan.
        if self._should_replan():
            yield from self._run_planning(task)

    # Derive final answer. Typical pattern: the planner creates a terminal
    # task that calls `final_answer`. If none did, we fall back to a
    # `provide_final_answer` LLM call, as today.
    yield self._finalize(task)
```

### 10.2 `_should_replan`

Returns True when any of the following holds:

- The most recent `NextPlanningTrigger` is satisfied (target task
  completed, or `n` completions since last planning, etc.).
- The graph has no ready or running tasks but `is_done()` is still
  False (sanity guard: avoid deadlock — in practice this shouldn't
  happen, but if it does, we re-plan instead of hanging).

A failure-driven replan condition will be added alongside failure
handling (§14).

### 10.3 Final Answer

Two supported styles, both documented in the prompt examples:

1. The planner schedules a terminal aggregator task that calls
   `final_answer(...)`. Its return value becomes the run's output.
2. If no terminal task exists when the graph is done, the agent runs
   the existing `provide_final_answer()` path over the enriched memory.

## 11. Flag Surface

```python
# Option A — explicit class
from smolagents import ParallelCodeAgent
agent = ParallelCodeAgent(
    tools=[...],
    model=model,
    max_parallel_tasks=4,
)
agent.run("…")

# Option B — convenience flag on CodeAgent (thin dispatch)
agent = CodeAgent(tools=[...], model=model, parallel_execution=True)
```

Both are trivial to support; Option B just instantiates the parallel
subclass under the hood when the flag is True. Default remains `False`,
so no existing user is affected.

## 12. Observability

- The lifecycle event types in §9.3 are streamable; existing
  `stream=True` users get them for free.
- `TaskGraph.to_yaml()` is logged at `LogLevel.DEBUG` after every
  planning call and at `INFO` on completion.
- `Monitor` is extended with a minimal parallel counter
  (`tasks_completed`). Richer metrics — failure rates, predicted-vs-
  actual runtime ratios, etc. — are deferred along with progress
  reporting and failure handling.

## 13. Testing Strategy

- **Unit:** TaskGraph state machine, YAML round-trip, trigger
  evaluation.
- **Planner:** use `InferenceClientModel` mocks that return canned YAML;
  assert that new tasks are merged, invalid YAML triggers the retry,
  dependency validation rejects bad ids.
- **Scheduler:** run fake tasks (pure Python callables) to verify
  parallelism and basic lifecycle events — no LLM involved.
- **End-to-end:** mock model that returns scripted planning YAML and
  scripted code for a toy multi-fetch task; assert that
  `ParallelCodeAgent.run(...)` completes with the expected final answer
  on the happy path and that task timings indicate actual parallelism.

Failure-path and progress-event tests will be added together with the
corresponding features (§14).

## 14. Out of Scope (for now)

- **Progress reporting.** Only task-lifecycle events are streamed in
  this iteration; mid-task progress updates from the worker's inner
  ReAct loop are not forwarded. A follow-up will add a
  `multiprocessing.Queue`-backed pipeline, a `TaskProgressEvent`, and
  `Task.progress` updates on the graph.
- **Failure handling.** The current design fails fast: on the first
  task exception or worker crash, the run raises
  `AgentExecutionError`. A follow-up will add retry budgets,
  dependency-cascade cancellation (`TaskStatus.CANCELLED`,
  `graph.cancel_descendants`), re-plan-on-failure, a
  `max_total_failures` global budget, and dedicated
  `TaskFailedEvent` / `TaskCancelledEvent` events. The planner prompt
  will grow a failure-recovery example at that point.
- **Resource conflict detection / locking.** Tasks declare `resources`,
  but the scheduler does not yet enforce exclusion. A follow-up design
  will turn this into real locks or conflict-serialization.
- **Cross-task shared Python state.** Workers get a read-only snapshot
  of parent state; writes happen only on merge when a task completes.
  Richer sharing (e.g. shared variables) is deferred.
- **Async (asyncio) model APIs.** We use process parallelism because the
  bottleneck is LLM latency + Python execution, which process workers
  handle well without touching the `Model` API. An async path can be
  added later without changing the graph / planner.
- **Streaming inside a sub-task to the outer UI.** Token streams from
  workers are not piped to the parent live view in this iteration.
  (Tied to progress reporting above.)

## 15. Rollout Plan

Initial slice (this design):

1. Land `task_graph.py` + tests. No behavioral change.
2. Land `events.py` (lifecycle events only) and `scheduler.py` with
   fake-task tests.
3. Land `planner.py` + prompt template + tests with mocked model.
4. Land `parallel/agent.py` and wire up `ParallelCodeAgent`
   (happy-path only; fail-fast on any task exception).
5. Add `parallel_execution` convenience flag on `CodeAgent`.
6. Docs + examples in `playground/`.

Follow-up slices (not part of this design, each tracked in §14):

7. Progress reporting — `multiprocessing.Queue` pipeline,
   `TaskProgressEvent`, `Task.progress` field, prompt surfacing.
8. Failure handling — retries, `TaskStatus.CANCELLED`,
   `graph.cancel_descendants`, `TaskFailedEvent` /
   `TaskCancelledEvent`, replan-on-failure, `max_total_failures`,
   failure-recovery example in the planner prompt.
9. Resource conflict detection / locking.

Each step is independently reviewable and ships behind the off-by-default
flag, so main stays green throughout.
