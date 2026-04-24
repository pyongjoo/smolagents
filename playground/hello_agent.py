import os

import litellm
from dotenv import load_dotenv

from smolagents import CodeAgent, LiteLLMModel


# Load environment variables from .env file (override any existing ones)
load_dotenv(override=True)

# Suppress litellm debug messages
litellm.suppress_debug_info = True

# Azure Foundry configuration from environment variables
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")


# Using Azure OpenAI (requires 'smolagents[litellm]')
model = LiteLLMModel(
    model_id=f"azure/{deployment_name}",
    api_base=endpoint,
    api_key=api_key,
    api_version=api_version
)

# Create an agent with no tools
agent = CodeAgent(tools=[], model=model)

# Run the agent with a task
result = agent.run("Calculate the sum of numbers from 1 to 10")
print(result)
