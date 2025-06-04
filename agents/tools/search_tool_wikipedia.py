from llama_index.core.tools.tool_spec.load_and_search import LoadAndSearchToolSpec
from llama_index.tools.wikipedia import WikipediaToolSpec

# Initialize the tool spec
tool_spec = WikipediaToolSpec()

# Get the search wikipedia tool
tool = tool_spec.to_tool_list()[1]

search_tools_wikipedia = LoadAndSearchToolSpec.from_defaults(tool).to_tool_list()