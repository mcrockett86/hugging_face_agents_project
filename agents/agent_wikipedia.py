from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
import os

# load the tools
from agents.tools.search_tool_wikipedia import search_tools_wikipedia
from agents.tools.basic_tools import sort_list_tool, add_numbers_tool, subtract_numbers_tool, multiply_numbers_tool, divide_numbers_tool


# Load the system prompt
with open('agents/prompts/system_prompt_wikipedia.txt') as f:
    system_prompt_wikipedia = f.read()

# Define the Settings with LLM and Local Embedding Model
Settings.llm = OpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"), request_timeout=360.0)
Settings.embed_model = "local:BAAI/bge-base-en-v1.5"

# Create the Wikipedia agent
agent_wikipedia = FunctionAgent(
    name="WikipediaAgent",
    description="Useful for searching the wikipedia for answer to a prompt.",
    tools=[*search_tools_wikipedia, sort_list_tool, add_numbers_tool, subtract_numbers_tool, multiply_numbers_tool, divide_numbers_tool],
    llm=Settings.llm,
    system_prompt=system_prompt_wikipedia,
    can_handoff_to= ['DuckDuckGoAgent', 'WikipediaAgent', 'VisionAgent']
)
