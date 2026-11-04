from .template import ChatTemplate


class Qwen38ChatTemplate(ChatTemplate):
    """Qwen3.8 onwards, which keep every reply's reasoning by default and state a reasoning effort to the model."""

    name = "qwen38"
    turn_end = "<|im_end|>\n"
    reasoning_start = "<think>"
    reasoning_end = "</think>"
    tool_call_parser = "qwen3_coder"
    stop = ("<|im_end|>", "<|endoftext|>")
    kwargs = ("enable_thinking", "preserve_thinking", "reasoning_effort", "add_vision_id")
    models = (
        "Qwen/Qwen3.8-27B",
        "Qwen/Qwen3.8-2.4T-A95B",
    )
