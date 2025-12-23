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
model_name = env.get("LLM_MODEL", os.environ.get("LLM_MODEL", ""))
model_api_key = env.get("LLM_API_KEY", os.environ.get("LLM_API_KEY", ""))
model_base_url = env.get("LLM_BASE_URL", os.environ.get("LLM_BASE_URL", ""))
bohrium_username = env.get("BOHRIUM_USERNAME", os.environ.get("BOHRIUM_USERNAME", ""))
bohrium_password = env.get("BOHRIUM_PASSWORD", os.environ.get("BOHRIUM_PASSWORD", ""))
bohrium_project_id = env.get("BOHRIUM_PROJECT_ID", os.environ.get("BOHRIUM_PROJECT_ID", ""))

executor = {
    "bohr": {
        "type": "dispatcher",
        "machine": {
            "batch_type": "Bohrium",
            "context_type": "Bohrium",
            "remote_profile": {
                "email": bohrium_username,
                "password": bohrium_password,
                "program_id": bohrium_project_id,
                "input_data": {
                    "image_name": "registry.dp.tech/dptech/dp/native/prod-22618/abacus-agent-tools:v0.2",
                    "job_type": "container",
                    "platform": "ali",
                    "scass_type": "c32_m64_cpu",
                },
            },
        }
    },
    "local": {"type": "local",}
}

EXECUTOR_MAP = {
    "generate_bulk_structure": executor["local"],
    "generate_molecule_structure": executor["local"],
    "abacus_prepare": executor["local"],
    "abacus_modify_input": executor["local"],
    "abacus_modify_stru": executor["local"],
    "abacus_collect_data": executor["local"],
    "abacus_prepare_inputs_from_relax_results": executor["local"],
    "generate_bulk_structure_from_wyckoff_position": executor["local"],
}

STORAGE = {
    "type": "https",
    "plugin":{
        "type": "bohrium",
        "username": bohrium_username,
        "password": bohrium_password,
        "project_id": bohrium_project_id,
    }
}

toolset = CalculationMCPToolset(
    connection_params=SseServerParams(
        url="http://localhost:50001/sse", # Or any other MCP server URL
        sse_read_timeout=3000,  # Set SSE timeout to 3000 seconds
    ),
    executor_map = EXECUTOR_MAP,
    executor=executor["local"],
    storage=STORAGE,
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