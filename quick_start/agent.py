from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from dp.agent.adapter.adk import CalculationMCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams, StreamableHTTPServerParams

import os, json

from abacusagent.prompt import EXAMPLE_ABACUS_AGENT_INSTRUCTION

# Set the secret key in ~/.abacusagent/env.json or as an environment variable, or modify the code to set it directly.
env_file = os.path.expanduser("~/.abacusagent/env.json")
if os.path.isfile(env_file):
    env = json.load(open(env_file, "r"))
else:
    env = {}
abacusagent_host = env.get("ABACUSAGENT_HOST", os.environ.get("ABACUSAGENT_HOST", ""))
abacusagent_port = env.get("ABACUSAGENT_PORT", os.environ.get("ABACUSAGENT_PORT", 50001))
model_name = env.get("LLM_MODEL", os.environ.get("LLM_MODEL", ""))
model_api_key = env.get("LLM_API_KEY", os.environ.get("LLM_API_KEY", ""))
model_base_url = env.get("LLM_BASE_URL", os.environ.get("LLM_BASE_URL", ""))

toolset = CalculationMCPToolset(
    connection_params=SseServerParams(
        url=f"http://{abacusagent_host}:{abacusagent_port}/sse", # Or any other MCP server URL
        sse_read_timeout=10800,  # Set SSE timeout to 3000 seconds
    ),
)

root_agent = Agent(
    name='agent',
    model=LiteLlm(
        model=model_name,
        base_url=model_base_url,
        api_key=model_api_key
    ),
    description=(
        "Do ABACUS calculations."
    ),
    instruction=EXAMPLE_ABACUS_AGENT_INSTRUCTION,
    tools=[toolset]
)