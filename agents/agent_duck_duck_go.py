from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
import os

# load the tools
from agents.tools.search_duckduckgo import search_tools_duckduckgo
from agents.tools.basic_tools import sort_list_tool, add_numbers_tool, subtract_numbers_tool, multiply_numbers_tool, divide_numbers_tool


# Load the system prompt
with open('agents/prompts/system_prompt_duck_duck_go.txt') as f:
    system_prompt_duck_duck_go = f.read()


# Define the Settings with LLM and Local Embedding Model
Settings.llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), request_timeout=360.0)
Settings.embed_model = "local:BAAI/bge-base-en-v1.5"

# Create the DuckDuckGo agent
agent_duck_duck_go = FunctionAgent(
    name="DuckDuckGoAgent",
    description="Usefusl for searching the web for answer to a prompt.",
    tools=[*search_tools_duckduckgo, sort_list_tool, add_numbers_tool, subtract_numbers_tool, multiply_numbers_tool, divide_numbers_tool],
    llm=Settings.llm,
    system_prompt=system_prompt_duck_duck_go,
    can_handoff_to= ['WikipediaAgent', 'VisionAgent']
)

