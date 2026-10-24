from __future__ import annotations

import json
import re
import uuid
from typing import Any, ClassVar

from ..messages import FunctionToolCall, ToolSchema


class ToolParser:
    """One tool-call grammar: `call_pattern` finds the calls in a response, `parse_block` reads a block's body.

    `name_pattern` reads the tool's name from a block `parse_block` cannot read.
    """

    name: ClassVar[str] = ""
    call_pattern: ClassVar[re.Pattern[str]] = re.compile(
        r"<tool_call>\n?((?:(?!<tool_call>).)*?)\n?(?:</tool_call>|\Z)", re.DOTALL
    )
    name_pattern: ClassVar[re.Pattern[str] | None] = None

    def parse_block(self, body: str, tools: list[ToolSchema] | None) -> list[FunctionToolCall] | None:
        """The calls in one block, or `None` when the body is not in this grammar."""
        raise NotImplementedError

    def parse(self, text: str, tools: list[ToolSchema] | None) -> tuple[str, list[FunctionToolCall]]:
        """The text with every block cut out, and the calls those blocks hold.

        A block `parse_block` cannot read is a call to the tool `name_pattern` reads, or with no name where it reads
        none, its arguments the block as the model wrote it: a harness refuses it with an error the model can correct,
        where text would end its turn. No name rather than a made-up one, which a harness could hold a tool by.
        """
        calls: list[FunctionToolCall] = []

        def replace(m: re.Match[str]) -> str:
            body = m.group(1)
            if (found := self.parse_block(body, tools)) is None:
                name = self.name_pattern.search(body) if self.name_pattern else None
                found = [self.tool_call(name.group(1).strip() if name else "", body)]
            calls.extend(found)
            return ""

        return self.call_pattern.sub(replace, text), calls

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
