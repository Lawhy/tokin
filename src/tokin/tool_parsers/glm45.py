import re

from ..messages import FunctionToolCall, ToolSchema
from .tool_parser import TOOL_CALL, ToolParser


class GLM45ToolParser(ToolParser):
    """GLM-4.5's grammar; every GLM since writes it, one tag per line before 4.7.

    `<tool_call>f<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>`
    """

    name = "glm45"
    block = TOOL_CALL
    identifier = re.compile(r"[\w.\-]+")
    pair = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)

    def parse_block(self, body: str, tools: list[ToolSchema] | None) -> list[FunctionToolCall] | None:
        name, _, arguments = body.partition("<arg_key>")
        name = name.strip()
        if not self.identifier.fullmatch(name):
            return None
        pairs = self.pair.findall("<arg_key>" + arguments) if arguments else []
        return [self.tool_call(name, {k.strip(): self.coerce(v, tools, name, k.strip()) for k, v in pairs})]
