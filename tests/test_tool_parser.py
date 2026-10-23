import pytest

from tokin.tool_parsers import ToolParser, get_tool_parser
from tokin.tool_parsers.hermes import HermesToolParser

PROPERTIES = {"s": {"type": "string"}, "n": {"type": "integer"}}
TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": PROPERTIES}}}]


class Shouting(ToolParser):
    def parse_block(self, body, tools):
        return [self.tool_call(body, {})] if body.isupper() else None


def test_parse_cuts_every_block_and_names_none_it_cannot_read():
    text = "a <tool_call>F</tool_call> b <tool_call>bad</tool_call> c <tool_call>worse"
    rest, found = Shouting().parse(text, None)
    assert rest == "a  b  c " and [c["function"] for c in found] == [
        {"name": "F", "arguments": "{}"},
        {"name": "", "arguments": "bad"},
        {"name": "", "arguments": "worse"},
    ]


def test_parse_reads_a_block_left_open_at_the_end():
    rest, found = Shouting().parse("a <tool_call>F</tool_call> b <tool_call>G", None)
    assert rest == "a  b " and [c["function"]["name"] for c in found] == ["F", "G"]


def test_parse_opens_a_block_at_its_last_opening_tag():
    rest, found = Shouting().parse("a <tool_call>dropped b <tool_call>F</tool_call>", None)
    assert rest == "a <tool_call>dropped b " and [c["function"]["name"] for c in found] == ["F"]


def test_tool_call_keeps_the_grammars_id_and_mints_the_rest():
    assert ToolParser().tool_call("f", {}, "abc")["id"] == "abc"
    minted = ToolParser().tool_call("f", {"x": "é"})
    assert minted["id"].startswith("call_") and minted["function"] == {"name": "f", "arguments": '{"x": "é"}'}


class TestCoerce:
    def test_string_schema_keeps_text(self):
        assert ToolParser().coerce("42", TOOLS, "f", "s") == "42"

    def test_other_schema_parses_json(self):
        assert ToolParser().coerce("42", TOOLS, "f", "n") == 42

    def test_no_schema_parses_json_when_it_can(self):
        assert ToolParser().coerce("[1]", TOOLS, "g", "k") == [1]
        assert ToolParser().coerce("plain", None, "f", "n") == "plain"


class TestGetToolParser:
    def test_registered_name(self):
        assert get_tool_parser("hermes") is HermesToolParser

    def test_module_path(self):
        assert get_tool_parser("tokin.tool_parsers.hermes:HermesToolParser") is HermesToolParser

    def test_unknown_name(self):
        with pytest.raises(ValueError, match="unknown tool parser"):
            get_tool_parser("pythonic")

    def test_non_parser_path(self):
        with pytest.raises(TypeError, match="not a ToolParser"):
            get_tool_parser("tokin.tool_parsers.parser:FunctionToolCall")
