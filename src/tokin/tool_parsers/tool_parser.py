from __future__ import annotations

import json
import re
import uuid
from typing import Any, ClassVar

from ..messages import FunctionToolCall, ToolSchema

TOOL_CALL = re.compile(r"<tool_call>\n?(.*?)\n?</tool_call>", re.DOTALL)


class ToolParser:
    """One tool-call grammar: `block` finds the calls in a response, `parse_block` turns a block's body into calls."""

    name: ClassVar[str] = ""
    block: ClassVar[re.Pattern[str]]

    def parse_block(self, body: str, tools: list[ToolSchema] | None) -> list[FunctionToolCall] | None:
        """The calls in one block, or `None` when the body is not in this grammar."""
        raise NotImplementedError

    def parse(self, text: str, tools: list[ToolSchema] | None) -> tuple[str, list[FunctionToolCall]]:
        """Cut every block `parse_block` understands out of `text`; one it does not stays as the model wrote it."""
        calls: list[FunctionToolCall] = []

        def replace(m: re.Match[str]) -> str:
            if (found := self.parse_block(m.group(1), tools)) is None:
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

    def tool_call(self, name: str, arguments: dict[str, Any], id: str | None = None) -> FunctionToolCall:
        """An OpenAI tool call; `id` is the model's when the grammar carries one, as Mistral's does, minted otherwise."""
        return FunctionToolCall(
            id=id or f"call_{uuid.uuid4().hex[:24]}",
            type="function",
            function={"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        )
