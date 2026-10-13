import pytest
from openai.types.chat.completion_create_params import CompletionCreateParamsStreaming
from pydantic import ValidationError

from tokin.harness_apis import ChatCompletionRequest
from tokin.rollout import FinishReason, Generation

# OpenAI's keys `tokin` does not serve: each is declared when a harness is seen sending it.
UNSERVED = {
    "audio",
    "frequency_penalty",
    "function_call",
    "functions",
    "logit_bias",
    "logprobs",
    "metadata",
    "modalities",
    "moderation",
    "n",
    "parallel_tool_calls",
    "prediction",
    "presence_penalty",
    "prompt_cache_key",
    "prompt_cache_options",
    "prompt_cache_retention",
    "reasoning_effort",
    "response_format",
    "safety_identifier",
    "seed",
    "service_tier",
    "stop",
    "store",
    "stream",
    "stream_options",
    "tool_choice",
    "top_logprobs",
    "user",
    "verbosity",
    "web_search_options",
}

CALL = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}


def request(**over):
    return ChatCompletionRequest.model_validate({"model": "m", "messages": [{"role": "user", "content": "hi"}], **over})


def test_every_openai_key_is_served_or_listed_unserved():
    # A bump of `openai` that adds a key fails here, so the new key gets a decision rather than a default.
    upstream = CompletionCreateParamsStreaming.__required_keys__ | CompletionCreateParamsStreaming.__optional_keys__
    served = {
        key
        for name, field in ChatCompletionRequest.model_fields.items()
        for key in (field.validation_alias.choices if field.validation_alias else [name])
    }
    assert served <= upstream
    assert upstream - served == UNSERVED


def test_undeclared_key_is_refused_by_name():
    with pytest.raises(ValidationError, match="verbosity"):
        request(verbosity="low")


def test_messages_drop_unknown_keys():
    got = request(messages=[{"role": "user", "content": "hi", "provider_specific_fields": {}}])
    assert got.messages == [{"role": "user", "content": "hi"}]


def test_tools_keep_the_key_order_they_came_in():
    tool = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
    assert list(request(tools=[tool]).tools[0]) == ["type", "function"]


def test_max_tokens_below_one_is_refused():
    with pytest.raises(ValidationError, match="max_tokens"):
        request(max_tokens=0)


def test_max_completion_tokens_is_max_tokens():
    assert request(max_completion_tokens=5).max_tokens == 5


def test_chat_params_leave_out_what_the_harness_did_not_set():
    assert request(temperature=0.7).chat()[2] == {"temperature": 0.7}


def test_lazy_fields_are_read_into_lists():
    parts = [{"type": "text", "text": "hi"}]
    got = request(messages=[{"role": "user", "content": parts}, {"role": "assistant", "tool_calls": [CALL]}])
    assert got.messages[0]["content"] == parts and got.messages[1]["tool_calls"] == [CALL]


def test_respond_reports_a_stop_with_tool_calls_as_tool_calls():
    message = {"role": "assistant", "content": None, "tool_calls": [CALL]}
    assert request().respond(message, Generation([1, 2], FinishReason.STOP), 3).choices[0].finish_reason == "tool_calls"
