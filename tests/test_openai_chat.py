from http import HTTPStatus

import pytest
from openai.types.chat.completion_create_params import CompletionCreateParamsStreaming
from pydantic import ValidationError

from tokin.gateway import ChatResponse
from tokin.harness_apis import OpenAIChatRequest
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
    "top_logprobs",
    "user",
    "verbosity",
    "web_search_options",
}

CALL = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}


def request(**over):
    return OpenAIChatRequest.model_validate({"model": "m", "messages": [{"role": "user", "content": "hi"}], **over})


def test_every_openai_key_is_served_or_listed_unserved():
    # A bump of `openai` that adds a key fails here, so the new key gets a decision rather than a default.
    upstream = CompletionCreateParamsStreaming.__required_keys__ | CompletionCreateParamsStreaming.__optional_keys__
    served = {
        key
        for name, field in OpenAIChatRequest.model_fields.items()
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


def test_a_bad_lazy_item_is_located_in_its_message():
    no_id = {"type": "function", "function": {"name": "f", "arguments": "{}"}}
    with pytest.raises(ValidationError) as e:
        request(messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "tool_calls": [no_id]}])
    assert e.value.errors()[0]["loc"][:4] == ("messages", 1, "tool_calls", 0)


def test_error_names_the_message_validation_failed_in():
    with pytest.raises(ValidationError) as e:
        request(messages=[{"role": "user", "content": "hi"}, {"role": "user"}])
    assert OpenAIChatRequest.error(e.value, HTTPStatus.BAD_REQUEST)["error"]["param"] == "messages[1]"


def test_tool_choice_other_than_auto_is_refused():
    with pytest.raises(ValidationError, match="tool_choice"):
        request(tool_choice="required")


def test_parallel_tool_calls_false_is_refused():
    with pytest.raises(ValidationError, match="parallel_tool_calls"):
        request(parallel_tool_calls=False)


def test_store_true_is_refused():
    with pytest.raises(ValidationError, match="store"):
        request(store=True)


def test_max_tokens_below_one_is_refused():
    with pytest.raises(ValidationError, match="max_tokens"):
        request(max_tokens=0)


def test_max_completion_tokens_is_max_tokens():
    assert request(max_completion_tokens=5).max_tokens == 5


def test_chat_request_params_leave_out_what_the_harness_did_not_set():
    assert request(temperature=0.7).chat_request().params == {"temperature": 0.7}


def test_lazy_fields_are_read_into_lists():
    parts = [{"type": "text", "text": "hi"}]
    got = request(messages=[{"role": "user", "content": parts}, {"role": "assistant", "tool_calls": [CALL]}])
    assert got.messages[0]["content"] == parts and got.messages[1]["tool_calls"] == [CALL]


def test_response_reports_a_stop_with_tool_calls_as_tool_calls():
    message = {"role": "assistant", "content": None, "tool_calls": [CALL]}
    assert (
        request()
        .response(ChatResponse(message, Generation([1, 2], FinishReason.STOP), 3, "chatcmpl-1"))
        .choices[0]
        .finish_reason
        == "tool_calls"
    )


def test_response_carries_the_turns_id():
    reply = ChatResponse({"role": "assistant", "content": "ok"}, Generation([1], FinishReason.STOP), 3, "chatcmpl-7")
    assert request().response(reply).id == "chatcmpl-7"
