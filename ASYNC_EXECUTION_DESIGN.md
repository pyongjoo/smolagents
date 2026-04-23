# Async Execution Design for Smolagents

## Overview

This document describes the design and implementation plan for adding asynchronous execution capabilities to the smolagents CodeAgent. This feature enables the LLM's thought and code generation to proceed while the previous step's code is still executing.

**Key Constraint:** Executions are sequential - only one code execution runs at a time. However, while code is executing in the background, the LLM can proceed with generating thoughts and code for subsequent steps.

## Motivation

**Current Limitation:** Steps execute sequentially in a completely blocking manner:
```
Step 1: Thought → Code → Execute → Observe (blocks here)
Step 2: Thought → Code → Execute → Observe (blocks here)
Step 3: ...
```

**With Async Execution:** LLM thinking/coding overlaps with execution:
```
Step 1: Thought → Code → Execute (execution running in background)
              ↓ (LLM doesn't block)
Step 2: Thought → Code → [Queued - waiting for Step 1 execution to finish]
Step 1: Execution completes → Observation available
Step 2: Execute (now runs)
              ↓ (LLM doesn't block)
Step 3: Thought → Code → [Queued - waiting for Step 2 execution to finish]
...
```

**Key Insight:** While code executes (often I/O bound operations like API calls, file operations), the LLM can be thinking about and generating code for the next step. This overlapping reduces total wall-clock time.

**Benefits:**
- Faster task completion by overlapping LLM time with execution time
- Better resource utilization (LLM + executor both working)
- More natural pipelining of work
- Simpler than full parallelism: only one execution at a time means no state conflicts

## Core Architecture

### Three-Phase Step Execution

Each step is broken into three distinct phases:

1. **Phase 1: Thought Generation** (streaming, yields immediately)
   - LLM generates reasoning about what to do next
   - Streams character-by-character via model.generate_stream()
   - Stops at code block opening tag
   - Yields: `ThoughtStreamDelta` events, then `ThoughtComplete`

2. **Phase 2: Code Generation** (streaming, yields immediately)
   - LLM continues generating the code block
   - Streams character-by-character
   - Stops at code block closing tag
   - Yields: `CodeStreamDelta` events, then `CodeComplete`

3. **Phase 3: Code Execution** (non-blocking, process-based)
   - Code submitted to separate process for execution
   - Returns immediately without waiting
   - Yields: `ExecutionStarted` event
   - Later yields: `ExecutionComplete` when done

### Wait State Mechanism

**Key Innovation:** The LLM has multiple strategies to handle the fact that the previous execution may still be running:

**Strategy A: Explicit Wait Signal** (RECOMMENDED when you need results from previous step)
1. LLM returns special marker in thought: `"no work to do"` or `"WAIT"`
2. Central coordinator detects this marker
3. Coordinator waits for the currently executing step to complete
4. When execution completes, coordinator re-prompts LLM with results
5. LLM can now proceed with new information

**Strategy B: Queue Code for Later** (LIMITED - see Error Handling section)
1. LLM generates code that will need results from previous step
2. Code is queued and will execute after the current execution finishes
3. ⚠️ LIMITATION: The code won't have access to variables from previous step due to process isolation
4. Only works if previous step writes to external resources (files, databases)

**Strategy C: Independent Work** (BEST when results not needed yet)
1. LLM recognizes the previous execution is running
2. Generates code for work that doesn't depend on previous results
3. This code can execute immediately after current execution finishes
4. Maximizes pipeline efficiency

**Example Flow (Strategy A - Explicit Wait):**
```
Step 1: "Fetch data from API" → Code → Executing in background
Step 2: "Process the fetched data" → LLM thinks: "I need data from Step 1"
        → Returns: "no work to do: waiting for Step 1 to complete"
Coordinator: Detects wait state → Blocks until Step 1 execution completes
Step 1: Completes → Observation available
Coordinator: Re-prompts LLM with Step 1 results
Step 2: "Now I can process data" → Code → Execute (now runs)
```

**Example Flow (Strategy C - Independent Work):**
```
Step 1: "Fetch user data from API" → Code → Executing in background
Step 2: LLM thinks: "Step 1 is fetching data. I can prepare analysis config independently"
        → Generates code: "config = {'method': 'stats', 'metrics': ['mean', 'std']}"
        → Code queued (will run after Step 1)
Step 1: Completes → Observation available
Step 2: Code executes (config setup)
Step 3: LLM thinks: "Now I have both user data and config"
        → Generates code: "analyze(user_data, config)"
        → Code queued
Step 2: Completes → config available
Step 3: Code executes (analysis runs)
```

**Important:** Since only one execution runs at a time, there are no state conflicts. Each execution has exclusive access to modify state.

## Component Design

### 1. AsyncStepManager (`async_execution.py`)

**Purpose:** Orchestrates sequential step execution using a single process

**Key Responsibilities:**
- Submit code to a separate process for execution
- Track the currently executing step (only one at a time)
- Check if execution is complete
- Manage state snapshots and merging
- Maintain queue of pending code to execute
- Handle process lifecycle

**Main Methods:**
```python
submit_code_execution(code, step_number, state_snapshot, tools) -> Future
register_inflight_step(step_number, thought, code, future)
get_current_execution() -> AsyncStepState | None
is_execution_complete() -> bool
get_completed_execution() -> tuple[step_number, output, logs, state_changes] | None
wait_for_completion() -> tuple[step_number, output, logs, state_changes]
has_pending_execution() -> bool
shutdown()
```

**AsyncStepState Data Structure:**
```python
@dataclass
class AsyncStepState:
    step_number: int
    status: "executing" | "completed" | "failed"
    thought: str
    code: str
    started_at: float
    completed_at: float | None
    execution_future: Future | None
    result: tuple[output, logs, state_changes] | None
    error: Exception | None
    elapsed_seconds: float  # computed property
```

### 2. Process-Based Code Execution

**Why Processes (not threads):**
- Complete isolation between executions
- No GIL contention
- Clean state management
- No shared memory concerns
- Safe for untrusted code

**Execution Function:**
```python
def _execute_code_in_process(
    code: str,
    state_snapshot: dict,
    tools: dict,
    authorized_imports: list[str],
    max_print_outputs_length: int
) -> tuple[output, logs, state_changes]:
    # Runs in separate process
    # Creates isolated LocalPythonExecutor
    # Restores state from snapshot
    # Executes code
    # Returns results and state changes
```

**Process Pool Configuration:**
- Uses `multiprocessing.ProcessPoolExecutor`
- Context: `spawn` (cleanest isolation)
- Max workers: **1** (only one execution at a time)

### 3. Modified CodeAgent (`agents.py`)

**New Initialization Parameters:**
```python
enable_async_execution: bool = False  # Enable async mode
```

**Key Methods:**

#### `_run_stream_async()`
Main async orchestration loop:
```python
def _run_stream_async(self, task, max_steps, images):
    async_manager = AsyncStepManager()  # Single worker
    
    while not returned_final_answer and self.step_number <= max_steps:
        # Check if previous execution is still running
        current_execution = async_manager.get_current_execution()
        
        # Check if previous execution completed
        if async_manager.is_execution_complete():
            step_num, output, logs, state_changes = async_manager.get_completed_execution()
            # Merge state changes
            self.python_executor.merge_state_changes(state_changes, step_num)
            # Yield completion event
            yield ExecutionComplete(step_num, output, logs, state_changes)
            # Add to memory
            memory.add_observation(step_num, output, logs)
        
        # Generate next step (thought + code)
        wait_detected = False
        for output in self._step_stream_async(action_step, async_manager, current_execution):
            yield output
            
            # Check for wait state
            if isinstance(output, ThoughtComplete):
                if self._is_wait_state(output.content):
                    wait_detected = True
                    break
        
        # If wait state detected, block until current execution completes
        if wait_detected:
            self.logger.log("Agent is waiting for execution to complete...")
            step_num, output, logs, state_changes = async_manager.wait_for_completion()
            self.python_executor.merge_state_changes(state_changes, step_num)
            yield ExecutionComplete(step_num, output, logs, state_changes)
            memory.add_observation(step_num, output, logs)
            # Continue loop without incrementing step_number
            # This will re-prompt LLM with new context
            continue
        
        self.step_number += 1
    
    # Wait for final execution if any
    if async_manager.has_pending_execution():
        step_num, output, logs, state_changes = async_manager.wait_for_completion()
        self.python_executor.merge_state_changes(state_changes, step_num)
        yield ExecutionComplete(step_num, output, logs, state_changes)
    
    async_manager.shutdown()
```

#### `_step_stream_async()`
Three-phase step execution:
```python
def _step_stream_async(self, memory_step, async_manager, current_execution):
    # Prepare messages with current execution context
    input_messages = self._augment_messages_with_execution_context(
        self.write_memory_to_messages(),
        current_execution
    )
    
    # Phase 1: Stream thought generation
    thought_content = ""
    for delta in self.model.generate_stream(
        input_messages,
        stop_sequences=[self.code_block_tags[0]]
    ):
        thought_content += delta.content
        yield ThoughtStreamDelta(content=delta.content, step=self.step_number)
    
    yield ThoughtComplete(content=thought_content, step=self.step_number)
    memory_step.thought = thought_content
    
    # Check if this is a wait state
    if self._is_wait_state(thought_content):
        return  # Don't generate code, just return
    
    # Phase 2: Stream code generation
    code_content = ""
    for delta in self.model.generate_stream(
        input_messages,
        stop_sequences=[self.code_block_tags[1]]
    ):
        code_content += delta.content
        yield CodeStreamDelta(content=delta.content, step=self.step_number)
    
    code_action = parse_code_blobs(code_content, self.code_block_tags)
    code_action = fix_final_answer_code(code_action)
    
    yield CodeComplete(content=code_action, step=self.step_number)
    memory_step.code_action = code_action
    
    # Phase 3: Submit to background execution (will queue if previous still running)
    execution_future = async_manager.submit_code_execution(
        code=code_action,
        step_number=self.step_number,
        state_snapshot=self.python_executor.get_state_snapshot(),
        tools={**self.tools, **self.managed_agents}
    )
    
    async_manager.register_inflight_step(
        step_number=self.step_number,
        thought=thought_content,
        code=code_action,
        future=execution_future
    )
    
    yield ExecutionStarted(
        code=code_action,
        step=self.step_number,
        started_at=time.time()
    )
```

#### `_is_wait_state()`
Detect wait state from thought content:
```python
def _is_wait_state(self, thought: str) -> bool:
    """Check if LLM indicated it needs to wait."""
    thought_lower = thought.lower()
    wait_markers = [
        "no work to do",
        "cannot proceed",
        "waiting for",
        "must wait",
        "need to wait",
        "[wait]",
    ]
    return any(marker in thought_lower for marker in wait_markers)
```

#### `_augment_messages_with_execution_context()`
Add current execution information to prompt:
```python
def _augment_messages_with_execution_context(
    self, 
    messages: list[ChatMessage], 
    current_execution: AsyncStepState | None
) -> list[ChatMessage]:
    """Inject current execution context into system prompt."""
    if not current_execution:
        return messages
    
    # Build execution context
    execution_context = self._build_execution_context(current_execution)
    
    # Insert into system message
    system_message = messages[0]
    augmented_content = system_message.content + "\n\n" + execution_context
    messages[0] = ChatMessage(
        role=system_message.role,
        content=augmented_content
    )
    
    return messages
```

### 4. Prompt Template (`prompts/code_agent_async.yaml`)

**New File:** Dedicated prompt template for async execution mode.

**Key Sections:**

#### System Prompt Modifications
```yaml
system_prompt: |-
  You are an expert assistant who can solve any task using code blobs.
  
  **IMPORTANT: ASYNC EXECUTION MODE**
  You are operating in async mode where code execution happens in background.
  **KEY CONSTRAINT:** Only ONE execution runs at a time (sequential execution).
  However, while code executes, you can think and generate code for the next step.
  
  At each step, follow this protocol:
  
  1. **Check if previous execution is still running** (see "Current Execution" below)
  
  2. **Decide your strategy:**
  
     **Option A - Explicit Wait (RECOMMENDED when you need previous results):**
     - Write: "Thought: no work to do - waiting for Step X to complete"
     - DO NOT write any code block
     - The system will wait for Step X and re-prompt you
     - Use this when you need variables created by the previous step
     
     **Option B - Queue Independent Code:**
     - Write normal thought explaining your approach
     - Generate code for work that doesn't depend on previous execution
     - Your code will execute after the current execution finishes
     - ⚠️  CANNOT access variables from previous step (process isolation)
     - Can work with: external resources, independent calculations
     
     **Option C - Queue Code That Assumes Completion:**
     - Write thought explaining you're generating code for after current execution
     - Generate code that will run after previous completes
     - Only works if previous step writes to external resources (files, DBs)
     - Include checks: `if os.path.exists(...)` for safety
  
  3. **Write your Thought:** explaining your reasoning and chosen strategy
  
  4. **Write your Code:** (unless using Option A explicit wait)
  
  {%- if current_execution %}
  
  ## Current Execution:
  
  The following operation is CURRENTLY RUNNING:
  
  ### Step {{current_execution.step_number}} (running for {{current_execution.elapsed_seconds | round(1)}}s):
  
  **Thought:** {{current_execution.thought | truncate(200)}}
  
  **Code executing:**
  ```python
  {{current_execution.code}}
  ```
  
  **Status:** {{current_execution.status}}
  
  **Decision Guide:**
  - **Strategy A - Explicit Wait:** Use if you need results from Step {{current_execution.step_number}}
  - **Strategy B - Independent Code:** Use if your work doesn't depend on above
  - **Strategy C - Queue Dependent Code:** Use if Step {{current_execution.step_number}} writes to files/DBs you can read
  - ⚠️  Your code will run AFTER Step {{current_execution.step_number}} completes
  - ⚠️  You CANNOT access variables from Step {{current_execution.step_number}} (process isolation)
  
  {%- else %}
  
  ## Current Execution:
  None - the previous execution has completed. You can proceed freely.
  
  {%- endif %}
  
  [Rest of standard instructions...]
  
  **Rules for Async Mode:**
  1. Always check "Current Execution" section
  2. Remember: Only ONE execution runs at a time
  3. Your generated code will queue and run after current execution finishes
  4. Choose your strategy:
     - Option A: "no work to do" ← USE THIS when you need previous results
     - Option B: Generate independent code ← USE THIS to maximize pipeline efficiency
     - Option C: Queue code assuming previous completes ← ONLY for external resources
  5. ⚠️  Process isolation: You CANNOT check variables with `if 'var' in dir()`
  6. Use unique variable names to avoid confusion
  7. State merges after each execution completes
  
  Now Begin!
```

#### Planning Modifications
```yaml
planning:
  initial_plan: |-
    [Standard planning prompt...]
    
    **Additional requirement for async mode:**
    In your plan, consider that executions run sequentially but LLM thinking can overlap.
    Identify which steps can have their code generated while a previous step executes.
```

### 5. State Management (`local_python_executor.py`)

**New Methods:**

#### `get_state_snapshot()`
```python
def get_state_snapshot(self) -> dict:
    """
    Create serializable snapshot of current state.
    This will be passed to separate process.
    
    Filters out:
    - Non-picklable objects (with warning)
    - Internal variables (starting with _)
    - Tool instances (will be re-sent separately)
    """
    import copy
    snapshot = {}
    
    for key, value in self.state.items():
        # Skip internal variables
        if key.startswith('_') and key != '_print_outputs':
            continue
            
        try:
            # Test if picklable
            snapshot[key] = copy.deepcopy(value)
        except Exception as e:
            self.logger.log(
                f"Cannot snapshot variable '{key}' (type: {type(value).__name__}): {e}",
                level=LogLevel.DEBUG
            )
    
    return snapshot
```

#### `merge_state_changes()`
```python
def merge_state_changes(
    self,
    state_changes: dict,
    step_number: int,
    conflict_strategy: Literal["error", "overwrite", "skip"] = "error"
) -> dict[str, str]:
    """
    Merge state changes from completed async execution.
    
    Detects conflicts: variables modified both locally and by async step.
    
    Args:
        state_changes: New/modified variables from async execution
        step_number: Which step produced these changes
        conflict_strategy: How to handle conflicts
            - "error": Raise exception on conflict
            - "overwrite": Remote changes win
            - "skip": Local changes win
    
    Returns:
        Dictionary of detected conflicts (empty if none)
    """
    conflicts = {}
    
    for key, new_value in state_changes.items():
        if key == "_print_outputs":
            # Special handling: append print outputs
            if key in self.state:
                self.state[key] += "\n" + new_value
            else:
                self.state[key] = new_value
            continue
        
        if key in self.state:
            old_value = self.state[key]
            # Simple equality check for conflict detection
            # More sophisticated: hash-based or deep comparison
            if old_value != new_value:
                conflict_msg = (
                    f"Variable '{key}' was modified both locally and by Step {step_number}. "
                    f"Local: {type(old_value).__name__}, Remote: {type(new_value).__name__}"
                )
                conflicts[key] = conflict_msg
                
                if conflict_strategy == "error":
                    raise AsyncConflictError(conflict_msg)
                elif conflict_strategy == "skip":
                    self.logger.log(f"Conflict on '{key}': keeping local value", LogLevel.WARNING)
                    continue
                # else: overwrite (fall through)
        
        self.state[key] = new_value
    
    if conflicts and conflict_strategy != "error":
        self.logger.log(
            f"Resolved {len(conflicts)} conflicts using strategy: {conflict_strategy}",
            LogLevel.WARNING
        )
    
    return conflicts
```

### 6. Memory System (`memory.py`)

**New Output Classes:**

```python
@dataclass
class ThoughtStreamDelta(MemoryStep):
    """Yielded during thought streaming"""
    content: str
    step: int
    
    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        return []

@dataclass
class ThoughtComplete(MemoryStep):
    """Yielded when thought generation completes"""
    content: str
    step: int
    is_wait_state: bool = False  # Flagged if this is a wait
    
    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        # Only include in messages if not a wait state
        if self.is_wait_state:
            return []
        return [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=[{"type": "text", "text": f"Thought: {self.content}"}]
            )
        ]

@dataclass
class CodeStreamDelta(MemoryStep):
    """Yielded during code streaming"""
    content: str
    step: int
    
    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        return []

@dataclass
class CodeComplete(MemoryStep):
    """Yielded when code generation completes"""
    content: str
    step: int
    
    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        return []

@dataclass
class ExecutionStarted(MemoryStep):
    """Yielded when code execution starts (non-blocking)"""
    code: str
    step: int
    started_at: float
    
    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        return []

@dataclass
class ExecutionComplete(MemoryStep):
    """Yielded when background execution completes"""
    step: int
    output: Any
    logs: str
    state_changes: dict
    elapsed_time: float
    
    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        observation_text = f"Observation from Step {self.step}:\n"
        if self.logs:
            observation_text += f"Execution logs:\n{self.logs}\n"
        observation_text += f"Last output: {self.output}"
        
        return [
            ChatMessage(
                role=MessageRole.USER,
                content=[{"type": "text", "text": observation_text}]
            )
        ]
```

**Modified ActionStep:**
```python
@dataclass
class ActionStep(MemoryStep):
    step_number: int
    timing: Timing
    thought: str | None = None  # NEW: store thought separately
    model_input_messages: list[ChatMessage] | None = None
    tool_calls: list[ToolCall] | None = None
    error: AgentError | None = None
    model_output_message: ChatMessage | None = None
    model_output: str | list[dict[str, Any]] | None = None
    code_action: str | None = None
    observations: str | None = None
    observations_images: list["PIL.Image.Image"] | None = None
    action_output: Any = None
    token_usage: TokenUsage | None = None
    is_final_answer: bool = False
    is_wait_state: bool = False  # NEW: flag for wait states
```

## Implementation Plan

### Phase 1: Core Infrastructure
**Files to create:**
- `src/smolagents/async_execution.py`
  - `AsyncStepState` dataclass
  - `_execute_code_in_process()` function
  - `AsyncStepManager` class

**Files to modify:**
- `src/smolagents/utils.py`
  - Add `AsyncConflictError` exception

### Phase 2: Executor State Management
**Files to modify:**
- `src/smolagents/local_python_executor.py`
  - Add `get_state_snapshot()` method
  - Add `merge_state_changes()` method
  - Update `__call__()` to support process execution

### Phase 3: Memory System
**Files to modify:**
- `src/smolagents/memory.py`
  - Add new output dataclasses (ThoughtStreamDelta, etc.)
  - Modify `ActionStep` to include `thought` and `is_wait_state`
  - Update type aliases

### Phase 4: Prompt Template
**Files to create:**
- `src/smolagents/prompts/code_agent_async.yaml`
  - Complete async-aware prompt with wait state instructions
  - Inflight steps context section
  - Async-specific rules and examples

### Phase 5: Agent Async Logic
**Files to modify:**
- `src/smolagents/agents.py`
  - Update `CodeAgent.__init__()` with async parameters
  - Add `_run_stream_async()` method
  - Add `_step_stream_async()` method
  - Add `_is_wait_state()` helper
  - Add `_augment_messages_with_inflight_context()` helper
  - Update `run()` to route to async when enabled

### Phase 6: Testing & Validation
**Files to create:**
- `tests/test_async_execution.py`
  - Test wait state detection
  - Test state snapshot/merge
  - Test sequential execution with pipeline overlap
  - Test that only one execution runs at a time
- `playground/async_agent_demo.py`
  - Demonstration script

## Usage Example

```python
from smolagents import CodeAgent, LiteLLMModel

model = LiteLLMModel(model_id="gpt-4")

# Create agent with async execution enabled
agent = CodeAgent(
    tools=[],
    model=model,
    enable_async_execution=True
)

# Run agent - LLM will generate thoughts/code while executions run
result = agent.run(
    "Fetch weather data for 3 cities and analyze the patterns",
    stream=True
)

# Stream will yield:
# - ThoughtStreamDelta events (as thought is generated)
# - ThoughtComplete events (complete thought)
# - CodeStreamDelta events (as code is generated)
# - CodeComplete events (complete code)
# - ExecutionStarted events (code submitted to background)
# - ExecutionComplete events (when background execution finishes)
# - ActionStep events (complete steps)
# - FinalAnswerStep (final result)

for event in result:
    if isinstance(event, ThoughtComplete):
        print(f"Thought: {event.content}")
    elif isinstance(event, ExecutionStarted):
        print(f"Started executing step {event.step}")
    elif isinstance(event, ExecutionComplete):
        print(f"Step {event.step} completed: {event.output}")
```

## Behavior Examples

### Example 1: Independent Steps (Sequential Execution with Pipeline Overlap)
```
User: "Calculate 100! and also compute fib(30)"

Step 1:
  Thought: "I'll calculate 100 factorial first"
  Code: result_factorial = math.factorial(100)
  → Submits to background process (starts executing)

Step 2:
  Checks: Step 1 execution still running
  Thought: "Step 1 is calculating factorial. I can generate code for fibonacci now"
  Code: result_fib = fibonacci(30)
  → Queued (will execute after Step 1 completes)

Step 1 execution completes → result_factorial available

Step 2 execution starts → calculates fibonacci

Step 2 execution completes → result_fib available

Step 3:
  Checks: No execution running
  Thought: "Both results are now available, I can report them"
  Code: final_answer(f"Factorial: {result_factorial}, Fib: {result_fib}")
  → Executes

Note: While Step 1 was executing, the LLM generated code for Step 2. This pipeline
overlap saves the time it would take to generate Step 2's code after Step 1 finishes.
```

### Example 2: Dependent Steps (with Explicit Wait)
```
User: "Fetch user data from API, then calculate average age"

Step 1:
  Thought: "I'll fetch the user data"
  Code: users = fetch_users_from_api()
  → Submits to background process

Step 2:
  Checks: Step 1 still running
  Thought: "I need the users data from Step 1 to calculate average age"
  Thought: "no work to do - waiting for Step 1 to complete"
  → No code generated

Coordinator: Detects wait state → Blocks until Step 1 completes

Step 1 completes → users data now available
Coordinator: Re-prompts LLM with completed Step 1

Step 2 (retry):
  Checks: No running steps
  Thought: "Step 1 is complete, users data is available. I can now calculate average"
  Code: avg_age = sum(u['age'] for u in users) / len(users)
        final_answer(f"Average age: {avg_age}")
  → Executes
```

### Example 2B: Dependent Steps (with Code-Based Wait - External Resource)
```
User: "Fetch user data from API and save it, then calculate average age"

Step 1:
  Thought: "I'll fetch the user data and save to file for other steps"
  Code: 
    import json
    users = fetch_users_from_api()
    with open('/tmp/users.json', 'w') as f:
        json.dump(users, f)
    print("Users data saved to /tmp/users.json")
  → Submits to background process

Step 2:
  Checks: Step 1 still running
  Thought: "I need users data from Step 1. Since it's saved to file, I can poll for it"
  Code: 
    import time, os, json
    
    # Wait for file to appear
    timeout = 30
    elapsed = 0
    while not os.path.exists('/tmp/users.json') and elapsed < timeout:
        print(f"Waiting for users data file... ({elapsed}s)")
        time.sleep(1)
        elapsed += 1
    
    if not os.path.exists('/tmp/users.json'):
        raise Exception("Timeout waiting for users data")
    
    # Load and calculate average age
    with open('/tmp/users.json') as f:
        users = json.load(f)
    avg_age = sum(u['age'] for u in users) / len(users)
    print(f"Average age: {avg_age}")
    final_answer(f"Average age: {avg_age}")
  → Submits to background process (will poll file system)

Step 1 completes → writes /tmp/users.json
Step 2's code detects file → loads and calculates average age

Note: Strategy A (explicit wait) would be simpler for this case!
```

### Example 3: Mixed Strategies (Sequential Execution)
```
User: "Download file A, download file B, then merge them and analyze"

Step 1:
  Thought: "Download file A to /tmp/file_a.dat"
  Code: 
    file_a_data = download('url_a')
    with open('/tmp/file_a.dat', 'wb') as f:
        f.write(file_a_data)
    print("Downloaded file A")
  → Executing in background

Step 2:
  Checks: Step 1 execution running
  Thought: "Step 1 downloading. I'll generate code for file B download (Strategy B - queue independent work)"
  Code: 
    file_b_data = download('url_b')
    with open('/tmp/file_b.dat', 'wb') as f:
        f.write(file_b_data)
    print("Downloaded file B")
  → Queued (will run after Step 1)

Step 1: Completes → /tmp/file_a.dat available
Step 2: Execution starts → downloading file B

Step 3:
  Checks: Step 2 execution running (downloading file B)
  Thought: "Step 2 downloading B. I can generate merge code (Strategy C - queue dependent code)"
  Code:
    import time, os
    
    # Wait for both files (defensive check)
    timeout = 60
    start = time.time()
    while (not os.path.exists('/tmp/file_a.dat') or 
           not os.path.exists('/tmp/file_b.dat')) and \
          (time.time() - start) < timeout:
        time.sleep(1)
    
    if not os.path.exists('/tmp/file_a.dat') or \
       not os.path.exists('/tmp/file_b.dat'):
        raise Exception("Timeout waiting for files")
    
    # Load and merge
    with open('/tmp/file_a.dat', 'rb') as f:
        file_a_data = f.read()
    with open('/tmp/file_b.dat', 'rb') as f:
        file_b_data = f.read()
    
    merged = merge_files(file_a_data, file_b_data)
    with open('/tmp/merged.dat', 'wb') as f:
        f.write(merged)
    print(f"Files merged: {len(merged)} bytes")
  → Queued (will run after Step 2)

Step 2: Completes → /tmp/file_b.dat available
Step 3: Execution starts → merging files

Step 4:
  Checks: Step 3 execution running (merging)
  Thought: "Step 3 merging. I can prepare analysis config (Strategy B - independent)"
  Code: 
    import json
    config = {
        'method': 'statistical',
        'metrics': ['mean', 'median', 'std'],
        'visualize': True
    }
    with open('/tmp/analysis_config.json', 'w') as f:
        json.dump(config, f)
    print("Analysis config ready")
  → Queued (will run after Step 3)

Step 3: Completes → /tmp/merged.dat available
Step 4: Execution starts → config setup

Step 4: Completes → /tmp/analysis_config.json available

Step 5:
  Checks: No execution running
  Thought: "All inputs ready. I can analyze now"
  Code:
    import json
    
    with open('/tmp/analysis_config.json') as f:
        config = json.load(f)
    
    with open('/tmp/merged.dat', 'rb') as f:
        merged = f.read()
    
    results = analyze(merged, config)
    final_answer(f"Analysis complete: {results}")
  → Executes

Note: While each execution ran sequentially, the LLM was generating code for
subsequent steps during each execution, creating a pipeline effect.
```

## Error Handling

### No State Conflicts (Sequential Execution Benefit!)

```python
# With sequential execution, this scenario CANNOT happen:
Step 1: x = fetch_data_slow()  # Takes 10s
Step 2: x = 42                  # Would run AFTER Step 1 completes

# Step 2's code is queued and only executes after Step 1 finishes
# Each execution has exclusive access to state
# No conflicts possible!
```

**Benefit of Sequential Execution:**
- No concurrent state modifications
- No need for complex conflict resolution
- Simpler state management
- Predictable execution order

### Code-Based Waiting Considerations

**Process isolation still applies:**
```python
# Step 2 code runs in its own process AFTER Step 1
# But it gets a state snapshot from AFTER Step 1 completed
# So it WILL see variables created by Step 1!
```

**This works because execution is sequential:**
When Step 2 starts executing:
1. Step 1 has already completed
2. Step 1's state changes are merged into main executor state
3. Step 2 gets a fresh snapshot with Step 1's variables
4. Step 2 can now access those variables

**Revised Example 2B - This NOW works:**
```python
# Step 1 runs first
users = fetch_users_from_api()

# After Step 1 completes, Step 2 gets snapshot with 'users'
# This check would work:
if 'users' not in dir():
    raise Exception("Users not found")

avg_age = sum(u['age'] for u in users) / len(users)
```

However, **explicit wait (Strategy A) is still simpler** because:
- Coordinator handles it automatically
- No need for defensive checks
- Clearer code intent

### Process Failures
```python
# If execution process crashes
Step 1: Execute code → Process dies

async_manager.poll_completed() returns:
  (step_num=1, output=None, logs="Process terminated", state_changes={})

Agent: Handles error in memory, continues or retries
```

### Timeouts
```python
# If step takes too long
Step 1: Execute infinite loop

async_manager.wait_any(timeout=30.0)
  → Raises TimeoutError after 30s

Agent: Can cancel step, log error, continue
```

## Configuration

### Agent Initialization
```python
agent = CodeAgent(
    tools=tools,
    model=model,
    
    # Async configuration
    enable_async_execution=True,        # Enable async mode
    
    # Executor configuration (must be local for async)
    executor_type="local",
    
    # Other standard parameters
    max_steps=20,
    verbosity_level=LogLevel.INFO,
)
```

### Environment Variables
```python
# Optional: Control multiprocessing behavior
os.environ['SMOLAGENTS_ASYNC_METHOD'] = 'spawn'  # or 'fork' (not recommended)
os.environ['SMOLAGENTS_ASYNC_TIMEOUT'] = '60'    # Default timeout in seconds
```

## Limitations and Constraints

### 1. Executor Compatibility
- **Only works with `executor_type="local"`**
- Remote executors (E2B, Modal, etc.) not supported initially
- Reason: Need control over state management

### 2. State Serialization
- Only picklable objects can be passed between processes
- File handles, network connections, etc. won't work
- Tools must be serializable

### 3. Managed Agents
- Managed agents (sub-agents) may not work in async mode
- Reason: Complex state management across processes

### 4. Performance Overhead
- Process creation has overhead (~100ms per process)
- State serialization/deserialization cost
- Best for long-running operations (>1 second)
- **Sequential execution means no true parallelism** - benefit comes from pipeline overlap

### 5. Determinism
- Execution order is fully deterministic (sequential)
- Results are predictable and reproducible
- Much simpler than fully concurrent execution

## Future Enhancements

### 1. Smart Dependency Tracking
- Automatically detect variable dependencies
- Build dependency graph
- Optimize execution order

### 2. State Diffing
- More sophisticated state tracking
- Track which lines modified which variables
- Better debugging of state changes

### 3. Execution Prediction
- Estimate execution time for steps
- Better wait vs. proceed decisions
- Optimize pipeline efficiency

### 4. Remote Executor Support
- Extend async support to E2B, Modal
- Requires async-aware remote execution

### 5. Checkpoint/Resume
- Save state during long runs
- Resume from checkpoints
- Fault tolerance

## Testing Strategy

### Unit Tests
- `AsyncStepManager` state tracking
- State snapshot/merge logic
- Wait state detection
- Prompt augmentation

### Integration Tests
- End-to-end async execution
- Sequential execution with pipeline overlap
- Wait state handling
- Error recovery

### Performance Tests
- Speedup measurement (vs sync)
- Pipeline efficiency measurement
- Overhead quantification

### Stress Tests
- Long execution chains
- Large state sizes
- Process failures
- Timeout handling

## Documentation Updates

### User Guide
- When to use async mode
- How to write async-friendly tasks
- Performance tuning tips
- Debugging async execution

### API Reference
- New parameters documented
- New output types explained
- Examples for each feature

### Migration Guide
- Converting sync agents to async
- Common pitfalls
- Performance comparison

## Open Questions

1. **How to handle tool calls that modify global state?**
   - With sequential execution, this is less of an issue
   - But still relevant for external resources (files, databases)
   - Possible solution: Tool-level awareness of resource usage

2. **Should we support async for ToolCallingAgent too?**
   - Similar architecture could apply
   - Would also use sequential execution model
   - Pipeline LLM reasoning with tool execution

3. **How to visualize async execution for debugging?**
   - Timeline view showing pipeline overlap
   - Show when LLM is generating vs when execution is running
   - State change visualization

4. **Should wait state be explicit or implicit?**
   - Current: Explicit "no work to do" marker
   - Alternative: Analyze code for dependencies automatically
   - Current approach gives LLM more control

5. **Rate limiting for LLM calls?**
   - Rapid re-prompting on wait states could hit rate limits
   - Need backoff or throttling
   - Possibly queue multiple generations during long executions

## Conclusion

This async execution design enables the CodeAgent to achieve overlapping of LLM reasoning/code generation with code execution. By separating thought generation, code generation, and code execution into distinct phases, and by enforcing sequential execution, we create a system that can pipeline work efficiently while maintaining simplicity and correctness.

### The Sequential Execution Model:

**Only one code execution runs at a time**, but the LLM can generate thoughts and code for subsequent steps while the current execution is running. This creates a pipeline effect:

```
Timeline:
|--- Step 1: Thought+Code (LLM) ---|--- Step 1: Execute ---|
                                    |--- Step 2: Thought+Code (LLM) ---|--- Step 2: Execute ---|
                                                                        |--- Step 3: Thought+Code (LLM) ---|
```

### The Three Strategies:

1. **Strategy A - Explicit Wait (RECOMMENDED)**: The LLM signals "no work to do" when it needs results from the running execution. The coordinator waits for completion and re-prompts with updated context. This is the simplest and most reliable approach.

2. **Strategy B - Queue Independent Code**: The LLM generates code for work that doesn't depend on the current execution's results. This code queues and runs after the current execution finishes.

3. **Strategy C - Queue Code Assuming Completion**: The LLM generates code that will run after the current execution, using external resources (files, DBs) for communication between steps.

### Key Benefits of Sequential Execution:

- **No state conflicts**: Only one execution modifies state at a time
- **Deterministic**: Execution order is predictable and reproducible
- **Simpler state management**: No complex merging or conflict resolution
- **Process isolation still protects**: Each execution in its own process
- **Pipeline efficiency**: LLM time overlaps with execution time

### Important Characteristics:

- **Process isolation ensures safety**: Each execution is completely isolated
- **State merges after each execution**: Results become available for next step
- **Variables ARE accessible**: Since execution is sequential, Step N+1 sees Step N's variables
- **Best for I/O-bound operations**: While waiting for API calls, file I/O, the LLM is thinking

### Trade-offs:

- **No true parallelism**: Independent operations still run sequentially
- **Benefit is pipeline overlap**: Time saved = LLM generation time during execution
- **Best when LLM time ≈ execution time**: Maximum pipeline efficiency
- **Less benefit if execution is instant**: Overhead may not be worth it

Implementation will be done incrementally, with careful testing at each phase. The design prioritizes **correctness, simplicity, and safety** while enabling meaningful pipeline efficiency for appropriate tasks.
