from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from llama_index.core.tools.tool_spec.load_and_search import LoadAndSearchToolSpec

# Initialize the tool spec
tool_spec = DuckDuckGoSearchToolSpec()

# get the duckduckgo full search tool
tool = tool_spec.to_tool_list()[1]

search_tools_duckduckgo = LoadAndSearchToolSpec.from_defaults(tool).to_tool_list()