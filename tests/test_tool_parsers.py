import json

from tokin.tool_parsers.glm45 import GLM45ToolParser
from tokin.tool_parsers.hermes import HermesToolParser
from tokin.tool_parsers.qwen3_coder import Qwen3CoderToolParser

PROPERTIES = {"s": {"type": "string"}, "n": {"type": "integer"}, "o": {"type": "object"}}
TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": PROPERTIES}}}]


def calls(found):
    return [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in found]


class TestHermesToolParser:
    def test_cuts_calls_out_of_text(self):
        text = (
            'Checking.\n<tool_call>\n{"name": "f", "arguments": {"n": 1}}\n</tool_call>\n'
            '<tool_call>\n{"name": "f", "arguments": {"n": 2}}\n</tool_call>'
        )
        rest, found = HermesToolParser().parse(text, None)
        assert (rest, calls(found)) == ("Checking.\n\n", [("f", {"n": 1}), ("f", {"n": 2})])

    def test_arguments_given_as_a_string(self):
        text = '<tool_call>{"name": "f", "arguments": "{\\"n\\": 1}"}</tool_call>'
        rest, found = HermesToolParser().parse(text, None)
        assert (rest, calls(found)) == ("", [("f", {"n": 1})])

    def test_bad_blocks_stay(self):
        text = (
            '<tool_call>\n{"name": "f", "arguments": [1]}\n</tool_call> '
            '<tool_call>\nnot json\n</tool_call> <tool_call>\n{"name": "f"'
        )
        assert HermesToolParser().parse(text, None) == (text, [])


class TestQwen3CoderToolParser:
    def test_types_values_by_schema(self):
        text = (
            "<tool_call>\n<function=f>\n<parameter=s>\n42\n</parameter>\n<parameter=n>\n42\n</parameter>\n"
            '<parameter=o>\n{"a": [1]}\n</parameter>\n</function>\n</tool_call>'
        )
        rest, found = Qwen3CoderToolParser().parse(text, TOOLS)
        assert (rest, calls(found)) == ("", [("f", {"s": "42", "n": 42, "o": {"a": [1]}})])

    def test_value_keeps_inner_newlines(self):
        text = "<tool_call>\n<function=f>\n<parameter=s>\nline1\nline2\n</parameter>\n</function>\n</tool_call>"
        rest, found = Qwen3CoderToolParser().parse(text, TOOLS)
        assert (rest, calls(found)) == ("", [("f", {"s": "line1\nline2"})])

    def test_block_without_function_stays(self):
        text = "<tool_call>\n<parameter=s>\nx\n</parameter>\n</tool_call>"
        assert Qwen3CoderToolParser().parse(text, TOOLS) == (text, [])


class TestGLM45ToolParser:
    def test_pairs_with_and_without_newlines(self):
        glm45 = (
            "<tool_call>f\n<arg_key>s</arg_key>\n<arg_value>42</arg_value>\n"
            "<arg_key>n</arg_key>\n<arg_value>42</arg_value>\n</tool_call>"
        )
        glm47 = "<tool_call>f<arg_key>s</arg_key><arg_value>42</arg_value><arg_key>n</arg_key><arg_value>42</arg_value></tool_call>"
        for text in (glm45, glm47):
            rest, found = GLM45ToolParser().parse(text, TOOLS)
            assert (rest, calls(found)) == ("", [("f", {"s": "42", "n": 42})])

    def test_name_only(self):
        rest, found = GLM45ToolParser().parse("<tool_call>f</tool_call>", None)
        assert (rest, calls(found)) == ("", [("f", {})])

    def test_bad_name_stays(self):
        text = "<tool_call>not a name<arg_key>s</arg_key><arg_value>x</arg_value></tool_call>"
        assert GLM45ToolParser().parse(text, None) == (text, [])
