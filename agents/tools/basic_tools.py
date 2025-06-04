from llama_index.core.tools import FunctionTool


def sort_list(input_list: list) -> list:
    """Useful for sorting a list of words or numbers."""
    return sorted(input_list)

sort_list_tool = FunctionTool.from_defaults(
    sort_list,
    name="sort_list_tool",
    description="Useful for sorting a list of words or numbers."
)


def reverse_text(text: str) -> str:
    """Useful for reversing a string of text."""
    return text[::-1]

reverse_text_tool = FunctionTool.from_defaults(
    reverse_text,
    name="reverse_text_tool",
    description="Useful for reversing a string of text."
)

# Example usage: reverse_text_tool
#print(tool.call('.rewsna eht sa "tfel" drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI'))


# Define some tools
def add(a: int, b: int) -> int:
    """Useful for adding two numbers."""
    return a + b

add_numbers_tool = FunctionTool.from_defaults(
    add,
    name="add_numbers_tool",
    description="Useful for adding two numbers."
)


def subtract(a: int, b: int) -> int:
    """Useful for subtracting one number from another number."""
    return a - b

subtract_numbers_tool = FunctionTool.from_defaults(
    subtract,
    name="subtract_numbers_tool",
    description="Useful for subtracting one number from another number."
)


def multiply(a: int, b: int) -> int:
    """Useful for multiplying two numbers."""
    return a * b

multiply_numbers_tool = FunctionTool.from_defaults(
    multiply,
    name="multiply_numbers_tool",
    description="Useful for multiplying two numbers."
)


def divide(a: int, b: int) -> int:
    """Useful for dividing two numbers."""
    return a / b

divide_numbers_tool = FunctionTool.from_defaults(
    divide,
    name="divide_numbers_tool",
    description="Useful for dividing one number by another number."
)