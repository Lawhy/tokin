from ..tool_parsers.hermes import HermesToolParser
from .template import ChatTemplate


class Qwen25ChatTemplate(ChatTemplate):
    """Qwen2.5 through Qwen3, which write tool calls as JSON inside `<tool_call>`."""

    name = "qwen25"
    turn_end = "<|im_end|>\n"
    reasoning_start = "<think>"
    reasoning_end = "</think>"
    tool_parser = HermesToolParser()
    stop = ("<|im_end|>", "<|endoftext|>")
    kwargs = ("enable_thinking",)
    models = (
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-3B",
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen2.5-14B",
        "Qwen/Qwen2.5-32B",
        "Qwen/Qwen2.5-72B",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-235B-A22B",
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "Qwen/Qwen3-4B-Thinking-2507",
        "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "Qwen/Qwen3-235B-A22B-Thinking-2507",
    )
