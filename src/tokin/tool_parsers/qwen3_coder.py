import re

from ..messages import FunctionToolCall, ToolSchema
from .tool_parser import TOOL_CALL, ToolParser


class Qwen3CoderToolParser(ToolParser):
    """Qwen3-Coder's grammar; Qwen3.5 onwards writes it, one tag per line.

    `<tool_call><function=f><parameter=x>1</parameter></function></tool_call>`
    """

    name = "qwen3_coder"
    block = TOOL_CALL
    function = re.compile(r"\A<function=([^>\n]+)>(.*)</function>\Z", re.DOTALL)
    parameter = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL)

    def parse_block(self, body: str, tools: list[ToolSchema] | None) -> list[FunctionToolCall] | None:
        if not (m := self.function.match(body.strip())):
            return None
        name, parameters = m.groups()
        arguments = {k: self.coerce(v, tools, name, k) for k, v in self.parameter.findall(parameters)}
        return [self.tool_call(name, arguments)]
