from __future__ import annotations

import pytest
from conftest import CHATML, END, START, ChatML, FakeTokenizer

from tokin.chat_templates import ChatTemplateError, get_chat_template
from tokin.rollout import FinishReason, Generation

USER = {"role": "user", "content": "hi"}
SYSTEM = {"role": "system", "content": "be terse"}
TOOLS = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}]
CALLS = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}}]
TOOL = {"role": "tool", "tool_call_id": "c1", "content": "18C"}
# A generation prompt that opens the think block itself, as Qwen3.5's does, unless thinking is off.
OPEN_THINK = CHATML.replace("<think></think>{% endif %}", "<think></think>{% else %}<think>\n{% endif %}")


def sampled(text, finish=FinishReason.STOP):
    return Generation([ord(c) for c in text] + [ord(END)] * (finish is FinishReason.STOP), finish)


def test_conform_parses_string_arguments_without_touching_the_input():
    assistant = {"role": "assistant", "content": "", "tool_calls": CALLS}
    conformed = ChatML(FakeTokenizer()).conform([USER, assistant])
    assert conformed[1]["tool_calls"][0]["function"]["arguments"] == {"x": 1}
    assert CALLS[0]["function"]["arguments"] == '{"x": 1}'
    assert conformed[0] == USER


def test_conform_joins_text_parts():
    parts = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert ChatML(FakeTokenizer()).conform([{"role": "user", "content": parts}]) == [{"role": "user", "content": "ab"}]


def test_conform_refuses_non_text_content():
    image = {"type": "image_url", "image_url": {"url": "u"}}
    with pytest.raises(ChatTemplateError, match="text"):
        ChatML(FakeTokenizer()).conform([{"role": "user", "content": [image]}])


def test_conform_refuses_arguments_that_are_not_json():
    call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{x: 1"}}
    with pytest.raises(ChatTemplateError, match="'c1' has arguments that are not JSON"):
        ChatML(FakeTokenizer()).conform([USER, {"role": "assistant", "content": "", "tool_calls": [call]}])


class TestOrderToolResults:
    def calls(self, *ids):
        return {"role": "assistant", "content": "", "tool_calls": [{**CALLS[0], "id": i} for i in ids]}

    def result(self, id):
        return {**TOOL, "tool_call_id": id}

    def test_puts_results_in_their_calls_order(self):
        messages = [USER, self.calls("c1", "c2"), self.result("c2"), self.result("c1")]
        ChatML(FakeTokenizer()).order_tool_results(messages)
        assert [m["tool_call_id"] for m in messages[2:]] == ["c1", "c2"]

    def test_refuses_a_result_out_of_its_run(self):
        messages = [USER, self.calls("c1"), self.result("c1"), USER, self.result("c1")]
        with pytest.raises(ChatTemplateError, match="does not follow"):
            ChatML(FakeTokenizer()).order_tool_results(messages)

    def test_refuses_a_missing_result(self):
        with pytest.raises(ChatTemplateError, match="one each"):
            ChatML(FakeTokenizer()).order_tool_results([USER, self.calls("c1", "c2"), self.result("c2")])

    def test_refuses_a_duplicate_result(self):
        messages = [USER, self.calls("c1"), self.result("c1"), self.result("c1")]
        with pytest.raises(ChatTemplateError, match="one each"):
            ChatML(FakeTokenizer()).order_tool_results(messages)


def test_apply_is_the_tokenizers_own_render():
    tokenizer = FakeTokenizer()
    assert ChatML(tokenizer).apply([SYSTEM, USER]) == tokenizer.apply_chat_template(
        [SYSTEM, USER], add_generation_prompt=True, tokenize=False
    )


def test_apply_turns_a_template_refusal_into_a_chat_template_error():
    strict = CHATML.replace(
        "{%- for m in messages %}",
        "{%- for m in messages %}{% if m.role == 'system' and not loop.first %}{{ raise_exception('system first') }}{% endif %}",
    )
    with pytest.raises(ChatTemplateError, match="system first"):
        ChatML(FakeTokenizer(strict)).apply([USER, SYSTEM])


def test_kwargs_reach_the_template():
    assert ChatML(FakeTokenizer(), enable_thinking=False).apply([USER]).endswith("<think></think>")


def test_unknown_kwarg_is_rejected():
    with pytest.raises(ValueError, match="enable_thinkin"):
        ChatML(FakeTokenizer(), enable_thinkin=False)


def test_stop_tokens_must_be_single_ids():
    with pytest.raises(ValueError, match="not one"):
        get_chat_template("qwen25")(FakeTokenizer())


class TestApplyIncrement:
    def test_tool_results_then_user(self):
        got = ChatML(FakeTokenizer()).apply_increment([TOOL, USER], TOOLS, tool_calls=CALLS)
        assert got == f"\n{START}tool\n18C<id>c1</id>{END}\n{START}user\nhi{END}\n{START}assistant\n"

    def test_tool_results_need_their_calls(self):
        with pytest.raises(ChatTemplateError, match="tool_calls"):
            ChatML(FakeTokenizer()).apply_increment([TOOL], TOOLS)

    def test_tool_results_come_first(self):
        with pytest.raises(ChatTemplateError, match="before"):
            ChatML(FakeTokenizer()).apply_increment([USER, TOOL], TOOLS, tool_calls=CALLS)

    def test_opening_token_must_be_a_stop(self):
        class NoTurnEnd(ChatML):
            turn_end = ""

        with pytest.raises(ChatTemplateError, match="never stops on"):
            NoTurnEnd(FakeTokenizer()).apply_increment([USER])

    def test_rewriting_earlier_turns_is_refused(self):
        hoisting = "{%- for m in messages if m.role == 'system' %}[{{ m.content }}]{% endfor %}" + CHATML.replace(
            "{%- for m in messages %}", "{%- for m in messages if m.role != 'system' %}"
        )
        with pytest.raises(ChatTemplateError, match="rewrites"):
            ChatML(FakeTokenizer(hoisting)).apply_increment([SYSTEM])


def test_opens_reasoning_follows_the_generation_prompt():
    assert not ChatML(FakeTokenizer()).opens_reasoning
    assert ChatML(FakeTokenizer(OPEN_THINK)).opens_reasoning
    assert not ChatML(FakeTokenizer(OPEN_THINK), enable_thinking=False).opens_reasoning


class TestParse:
    def test_think_block(self):
        got = ChatML(FakeTokenizer()).parse(sampled("<think>\nhmm\n</think>\n\nok"))
        assert got == {"role": "assistant", "content": "ok", "reasoning_content": "hmm"}

    def test_reasoning_left_open_by_the_prompt(self):
        got = ChatML(FakeTokenizer(OPEN_THINK)).parse(sampled("hmm\n</think>\n\nok"))
        assert got == {"role": "assistant", "content": "ok", "reasoning_content": "hmm"}

    def test_cut_short_keeps_its_last_token(self):
        got = ChatML(FakeTokenizer(OPEN_THINK)).parse(sampled("hmm", FinishReason.LENGTH))
        assert got == {"role": "assistant", "content": None, "reasoning_content": "hmm"}

    def test_empty_think_block(self):
        assert ChatML(FakeTokenizer()).parse(sampled("<think>\n\n</think>\n\nok")) == {
            "role": "assistant",
            "content": "ok",
        }

    def test_unclosed_think(self):
        got = ChatML(FakeTokenizer()).parse(sampled("<think>\nhmm"))
        assert got == {"role": "assistant", "content": None, "reasoning_content": "hmm"}

    def test_no_think(self):
        assert ChatML(FakeTokenizer()).parse(sampled("ok")) == {"role": "assistant", "content": "ok"}

    def test_stray_close_tag_stays_in_content(self):
        assert ChatML(FakeTokenizer()).parse(sampled("ok </think> x")) == {
            "role": "assistant",
            "content": "ok </think> x",
        }

    def test_tool_calls(self):
        got = ChatML(FakeTokenizer()).parse(
            sampled('Checking.\n<tool_call>\n{"name": "f", "arguments": {"x": 1}}\n</tool_call>')
        )
        [call] = got["tool_calls"]
        assert got["content"] == "Checking." and call["id"].startswith("call_") and call["type"] == "function"
        assert call["function"] == {"name": "f", "arguments": '{"x": 1}'}

    def test_tool_calls_alone_leave_no_content(self):
        got = ChatML(FakeTokenizer()).parse(sampled('<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>'))
        assert got["content"] is None and len(got["tool_calls"]) == 1

    def test_tool_call_drafted_in_reasoning_is_not_a_call(self):
        text = '<think>\n<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>\n</think>\n\nok'
        got = ChatML(FakeTokenizer()).parse(sampled(text))
        assert "tool_calls" not in got and got["content"] == "ok"

    def test_tool_call_in_unclosed_reasoning_is_draft(self):
        got = ChatML(FakeTokenizer()).parse(
            sampled('<think>\nhmm\n<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>')
        )
        assert "tool_calls" not in got and got["content"] is None

    def test_family_without_reasoning_block(self):
        class NoThink(ChatML):
            reasoning_start = reasoning_end = ""

        got = NoThink(FakeTokenizer()).parse(sampled("<think>hmm</think>ok"))
        assert got == {"role": "assistant", "content": "<think>hmm</think>ok"}


class TestGetTemplate:
    def test_unknown_name(self):
        with pytest.raises(ValueError, match="unknown chat template"):
            get_chat_template("gpt2")

    def test_module_path(self):
        assert get_chat_template("tokin.chat_templates.qwen25:Qwen25ChatTemplate") is get_chat_template("qwen25")

    def test_non_template_path(self):
        with pytest.raises(TypeError, match="not a ChatTemplate"):
            get_chat_template("tokin.chat_templates:ChatTemplateError")
