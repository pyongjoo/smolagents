"""Example: run a task with parallel planning + execution.

This script mirrors ``hello_agent.py``'s Azure OpenAI setup but opts in
to the new parallel execution mode. The task is intentionally shaped
so the planner has three obviously-independent subtasks it can fan out
in parallel, plus a terminal aggregator that calls ``final_answer``.

Run from the repo root after ``source venv/bin/activate``::

    python playground/parallel_agent.py
"""
import os
import time
from datetime import datetime

import litellm
from dotenv import load_dotenv

from smolagents import LiteLLMModel, ParallelCodeAgent
from smolagents.monitoring import LogLevel
from smolagents.parallel.agent import _describe_trigger


load_dotenv(override=True)
litellm.suppress_debug_info = True

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

model = LiteLLMModel(
    model_id=f"azure/{deployment_name}",
    api_base=endpoint,
    api_key=api_key,
    api_version=api_version,
)

# A task with three clearly-independent computations plus an
# aggregation step. A well-prompted planner will emit 3 parallel
# worker tasks and one aggregator that depends on all three.
TASK = """\
Solve these three independent problems, then combine the answers.

  A. Compute the 25th Fibonacci number (with F(1) = F(2) = 1).
  B. Compute 10! (ten factorial).
  C. Compute the sum of all integers from 1 to 100 inclusive.

Finally, return a single dict {"fib_25": ..., "fact_10": ..., "sum_1_to_100": ...}
as the final answer.
"""

agent = ParallelCodeAgent(
    tools=[],
    model=model,
    max_parallel_tasks=4,
    max_task_steps=4,
    # "thread" avoids pickling the LiteLLM model across process
    # boundaries on macOS; switch to "process" for true CPU parallelism
    # when your tools and model are picklable.
    executor_kind="thread",
    verbosity_level=LogLevel.INFO,
)

print(f"Running parallel agent on task:\n{TASK}\n{'=' * 60}")
start = time.time()
result = agent.run(TASK)
elapsed = time.time() - start

print("=" * 60)
print(f"Final answer: {result}")
print(f"Wall-clock time: {elapsed:.2f}s")
