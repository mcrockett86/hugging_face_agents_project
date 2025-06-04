from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings


# load the tools
from agents.tools.basic_tools import sort_list_tool, reverse_text_tool, add_numbers_tool, subtract_numbers_tool, multiply_numbers_tool, divide_numbers_tool

import os

# Load the system prompt
with open('agents/prompts/system_prompt_agent_basic_tools.txt') as f:
    system_prompt_agent_basic_tools = f.read()

# Define the Settings with LLM and Local Embedding Model
Settings.llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), request_timeout=360.0)
Settings.embed_model = "local:BAAI/bge-base-en-v1.5"

# Create the Basisc Tools agsent
agent_basic_tools = FunctionAgent(
    name="BasicToolsAgent",
    description="Useful for finding the answer to a prompt with basic tools.",
    tools=[sort_list_tool, reverse_text_tool, add_numbers_tool, subtract_numbers_tool, multiply_numbers_tool, divide_numbers_tool],
    llm=Settings.llm,
    system_prompt=system_prompt_agent_basic_tools,
    can_handoff_to= ['DuckDuckGoAgent', 'WikipediaAgent', 'VisionAgent']
)

