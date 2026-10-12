import json

from ..messages import FunctionToolCall, ToolSchema
from .tool_parser import TOOL_CALL, ToolParser


class HermesToolParser(ToolParser):
    """Hermes 2 Pro's grammar; Qwen2.5 and Qwen3 write it.

    `<tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call>`
    """

    name = "hermes"
    block = TOOL_CALL

    def parse_block(self, body: str, tools: list[ToolSchema] | None) -> list[FunctionToolCall] | None:
        try:
            call = json.loads(body)
            arguments = call.get("arguments") if isinstance(call, dict) else None
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
        except ValueError:
            return None
        if isinstance(call, dict) and isinstance(call.get("name"), str) and isinstance(arguments, dict):
            return [self.tool_call(call["name"], arguments)]
        return None
