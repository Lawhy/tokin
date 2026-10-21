from __future__ import annotations

import json
import re
import uuid
from typing import Any, ClassVar

from ..messages import FunctionToolCall, ToolSchema

# A block opens at the last `<tool_call>` before its `</tool_call>`, so one the model abandoned stays text, and ends
# at the end of the text when the model never closed it.
TOOL_CALL = re.compile(r"<tool_call>\n?((?:(?!<tool_call>).)*?)\n?(?:</tool_call>|\Z)", re.DOTALL)


class ToolParser:
    """One tool-call grammar: `block` finds the calls in a response, `parse_block` turns a block's body into calls.

    `name_of` reads the tool's name from a block `parse_block` cannot read.
    """

    name: ClassVar[str] = ""
    block: ClassVar[re.Pattern[str]]
    name_of: ClassVar[re.Pattern[str] | None] = None

    def parse_block(self, body: str, tools: list[ToolSchema] | None) -> list[FunctionToolCall] | None:
        """The calls in one block, or `None` when the body is not in this grammar."""
        raise NotImplementedError

    def read(self, body: str, tools: list[ToolSchema] | None) -> list[FunctionToolCall] | None:
        """The calls in one block, or `None` when it names no tool.

        A block `parse_block` cannot read is a call to the tool `name_of` reads, its arguments the block as the model
        wrote it: a harness refuses arguments that are not JSON with an error the model can correct, where text would
        end its turn.
        """
        found = self.parse_block(body, tools)
        if found is None and self.name_of and (name := self.name_of.search(body)):
            found = [self.tool_call(name.group(1).strip(), body)]
        return found

    def parse(self, text: str, tools: list[ToolSchema] | None) -> tuple[str, list[FunctionToolCall]]:
        """Cut every block `read` takes for calls out of `text`; one that names no tool stays as the model wrote it."""
        calls: list[FunctionToolCall] = []

        def replace(m: re.Match[str]) -> str:
            if (found := self.read(m.group(1), tools)) is None:
                return m.group(0)
            calls.extend(found)
            return ""

        return self.block.sub(replace, text), calls

    def coerce(self, value: str, tools: list[ToolSchema] | None, name: str, key: str) -> Any:
        """A tag body typed by the tool's schema: strings stay text, anything else is JSON when it parses as JSON."""
        for tool in tools or []:
            if tool["function"]["name"] == name:
                properties = tool["function"].get("parameters", {}).get("properties")
                if isinstance(properties, dict) and properties.get(key, {}).get("type") == "string":
                    return value
        try:
            return json.loads(value)
        except ValueError:
            return value

    def tool_call(self, name: str, arguments: dict[str, Any] | str, id: str | None = None) -> FunctionToolCall:
        """An OpenAI tool call, a string its arguments verbatim; `id` is the model's when the grammar carries one."""
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        return FunctionToolCall(
            id=id or f"call_{uuid.uuid4().hex[:24]}", type="function", function={"name": name, "arguments": arguments}
        )
