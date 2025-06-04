from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
import os

# load the tools
from agents.tools.basic_tools import sort_list_tool, add_numbers_tool, subtract_numbers_tool, multiply_numbers_tool, divide_numbers_tool


# Load the system prompt
with open('agents/prompts/system_prompt_vision.txt') as f:
    system_prompt_vision = f.read()

# Define the Settings with LLM and Local Embedding Model
Settings.llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), request_timeout=360.0)
Settings.embed_model = "local:BAAI/bge-base-en-v1.5"

# Create the Vision agent
agent_vision = FunctionAgent(
    name="VisionAgent",
    description="Useful for analyzing image content.",
    tools=[sort_list_tool, add_numbers_tool, subtract_numbers_tool, multiply_numbers_tool, divide_numbers_tool],
    llm=Settings.llm,
    system_prompt=system_prompt_vision,
    can_handoff_to= ['WikipediaAgent', 'DuckDuckGoAgent']
)
