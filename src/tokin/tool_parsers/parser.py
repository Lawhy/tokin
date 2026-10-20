from __future__ import annotations

import json
import re
import uuid
from typing import Any, ClassVar

from ..messages import FunctionToolCall, ToolSchema

OPEN, CLOSE = "<tool_call>", "</tool_call>"
# A block opens at the last `<tool_call>` before its `</tool_call>`: one the model opened and abandoned stays text.
TOOL_CALL = re.compile(r"<tool_call>\n?((?:(?!<tool_call>).)*?)\n?</tool_call>", re.DOTALL)


class ToolParser:
    """One tool-call grammar: `block` finds the calls in a response, `parse_block` turns a block's body into calls.

    `name_of` reads the tool's name from a block `parse_block` cannot read, so the call is handed over all the same.
    """

    name: ClassVar[str] = ""
    block: ClassVar[re.Pattern[str]]
    name_of: ClassVar[re.Pattern[str] | None] = None

    def parse_block(self, body: str, tools: list[ToolSchema] | None) -> list[FunctionToolCall] | None:
        """The calls in one block, or `None` when the body is not in this grammar."""
        raise NotImplementedError

    def unreadable(self, body: str) -> list[FunctionToolCall] | None:
        """The call in a block `parse_block` cannot read: its tool's name, with no arguments.

        No arguments rather than the body: templates that render arguments read a mapping, and the model's own text
        is in its context already.
        """
        if self.name_of is None or (name := self.name_of.search(body)) is None:
            return None
        return [self.tool_call(name.group(1).strip(), {})]

    def parse(self, text: str, tools: list[ToolSchema] | None) -> tuple[str, list[FunctionToolCall]]:
        """Cut every block that names a tool out of `text` as calls; one that names none stays as the model wrote it.

        A block `parse_block` cannot read, or one left open at the end, is still a call: a harness answers a call
        missing its arguments with an error the model can correct, where text would end its turn.
        """
        calls: list[FunctionToolCall] = []

        def read(body: str) -> list[FunctionToolCall] | None:
            return self.parse_block(body, tools) or self.unreadable(body)

        def replace(m: re.Match[str]) -> str:
            if (found := read(m.group(1))) is None:
                return m.group(0)
            calls.extend(found)
            return ""

        # A block left open at the end: its opening comes after the last close.
        cut, tail = text.rfind(OPEN), None
        if cut > text.rfind(CLOSE):
            text, tail = text[:cut], text[cut + len(OPEN) :]
        text = self.block.sub(replace, text)
        if tail is not None:
            if found := read(tail.strip("\n")):
                calls.extend(found)
            else:
                text += OPEN + tail
        return text, calls

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
        """An OpenAI tool call; `id` is the model's when the grammar carries one, as Mistral's, minted otherwise."""
        return FunctionToolCall(
            id=id or f"call_{uuid.uuid4().hex[:24]}",
            type="function",
            function={"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        )
