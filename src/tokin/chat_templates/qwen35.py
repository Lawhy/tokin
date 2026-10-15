from ..tool_parsers.qwen3_coder import Qwen3CoderToolParser
from .template import ChatTemplate


class Qwen35ChatTemplate(ChatTemplate):
    """Qwen3.5 onwards, which write tool calls as `<function=…>` XML inside `<tool_call>`."""

    name = "qwen35"
    turn_end = "<|im_end|>\n"
    reasoning_start = "<think>"
    reasoning_end = "</think>"
    tool_parser = Qwen3CoderToolParser()
    stop = ("<|im_end|>", "<|endoftext|>")
    kwargs = ("enable_thinking", "preserve_thinking", "add_vision_id")
    models = (
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.5-27B",
        "Qwen/Qwen3.5-35B-A3B",
        "Qwen/Qwen3.5-122B-A10B",
        "Qwen/Qwen3.5-397B-A17B",
        "Qwen/Qwen3.6-27B",
        "Qwen/Qwen3.6-35B-A3B",
        "Qwen/Qwen3.8-27B",
        "Qwen/Qwen3.8-2.4T-A95B",
    )
