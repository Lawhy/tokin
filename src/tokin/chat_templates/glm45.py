from ..tool_parsers.glm45 import GLM45ToolParser
from .template import ChatTemplate


class GLM45ChatTemplate(ChatTemplate):
    name = "glm45"
    reasoning_start = "<think>"
    reasoning_end = "</think>"
    tool_parser = GLM45ToolParser()
    stop = ("<|user|>", "<|observation|>", "<|endoftext|>")
    kwargs = ("enable_thinking", "clear_thinking", "reasoning_effort")
    models = (
        "zai-org/GLM-4.5",
        "zai-org/GLM-4.5-Air",
        "zai-org/GLM-4.6",
        "zai-org/GLM-4.7",
        "zai-org/GLM-4.7-Flash",
        "zai-org/GLM-5",
        "zai-org/GLM-5.1",
        "zai-org/GLM-5.2",
        "zai-org/GLM-5.3",
        "zai-org/GLM-5.3-Flash",
    )
