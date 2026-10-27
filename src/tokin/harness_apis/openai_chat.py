from __future__ import annotations

import time
from collections.abc import Iterator
from http import HTTPStatus
from typing import Annotated, Any, Literal, cast

from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionStreamOptionsParam
from pydantic import (
    AliasChoices,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    ValidatorFunctionWrapHandler,
    field_validator,
)
from pydantic_core import InitErrorDetails

from ..gateway import ChatRequest, ChatResponse, ContextLengthExceededError
from ..messages import Message, ToolSchema
from ..rollout import FinishReason, GenerationParams
from .request import HarnessRequest

# `extra="forbid"` would refuse unknown keys in messages too; they are dropped instead, since no template reads them.
_MESSAGES_VALIDATOR: TypeAdapter[list[Message]] = TypeAdapter(
    list[Annotated[Message, Field(discriminator="role")]], config=ConfigDict(extra="ignore")
)


class OpenAIChatRequest(HarnessRequest, extra="forbid"):
    """The harness's `/v1/chat/completions` body, as much of OpenAI's as `tokin` serves.

    Undeclared keys are refused by name, not ignored: an ignored key changes the response unannounced.
    """

    model: str
    messages: list[Message]
    tools: list[ToolSchema] | None = None
    max_tokens: int | None = Field(None, ge=1, validation_alias=AliasChoices("max_completion_tokens", "max_tokens"))
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    stream_options: ChatCompletionStreamOptionsParam | None = None
    store: Literal[False] | None = None
    tool_choice: Literal["auto"] | None = None
    parallel_tool_calls: Literal[True] | None = None

    @field_validator("tools", mode="wrap")
    @classmethod
    def validate_tools(cls, value: Any, handler: ValidatorFunctionWrapHandler) -> Any:
        """Validated, but kept as sent.

        Templates print tools whole with `tojson`; `pydantic` rebuilds a TypedDict in declared order, `function`
        before `type`, where harnesses and servers put `type` first.
        """
        handler(value)
        return value

    @field_validator("messages", mode="plain")
    @classmethod
    def validate_messages(cls, value: Any) -> list[Message]:
        """Validated by `_MESSAGES_VALIDATOR`, with OpenAI's `Iterable` fields read into lists.

        `pydantic` leaves them as one-pass iterators validating items when read: as lists they copy and reread,
        and a bad item fails here, with status 400.
        """
        messages = _MESSAGES_VALIDATOR.validate_python(value)
        for i, m in enumerate(cast(list[dict[str, Any]], messages)):
            for key, field in m.items():
                if isinstance(field, Iterator):
                    try:
                        m[key] = list(field)
                    except ValidationError as e:
                        # The iterator knows only its own items, not which message and field it is.
                        errors = [{**err, "loc": (i, key, *err["loc"])} for err in e.errors()]
                        raise ValidationError.from_exception_data(e.title, cast(list[InitErrorDetails], errors)) from e
        return messages

    def chat_request(self) -> ChatRequest:
        # A field named as a `GenerationParams` key means the same to the engine.
        params = self.model_dump(include=set(GenerationParams.__optional_keys__), exclude_none=True)
        return ChatRequest(self.messages, self.tools, cast(GenerationParams, params))

    def response(self, reply: ChatResponse) -> ChatCompletion:
        """The `ChatCompletion` for `reply`; a stop with tool calls reports `tool_calls`, which harnesses branch on."""
        stopped = reply.generation.finish_reason is FinishReason.STOP
        reason = "tool_calls" if stopped and reply.message.get("tool_calls") else reply.generation.finish_reason
        return ChatCompletion.model_validate(
            {
                "id": reply.id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model,
                "choices": [{"index": 0, "message": reply.message, "finish_reason": reason}],
                "usage": {
                    "prompt_tokens": reply.input_len,
                    "completion_tokens": len(reply.generation),
                    "total_tokens": reply.input_len + len(reply.generation),
                },
            }
        )

    def events(self, reply: ChatResponse) -> list[str]:
        """The response as OpenAI's stream: the whole reply in one chunk, usage in another if asked, then `[DONE]`.

        One chunk, as the turn is done before it is sent. Tool calls carry their `index` there: some clients
        assemble arguments split across chunks wrongly.
        """
        completion = self.response(reply)
        choice = completion.choices[0]
        delta = choice.message.model_dump(exclude_none=True)
        if calls := delta.get("tool_calls"):
            delta["tool_calls"] = [{**call, "index": i} for i, call in enumerate(calls)]
        head = {
            "id": completion.id,
            "object": "chat.completion.chunk",
            "created": completion.created,
            "model": self.model,
        }
        chunks = [{**head, "choices": [{"index": 0, "delta": delta, "finish_reason": choice.finish_reason}]}]
        if self.stream_options and self.stream_options.get("include_usage") and completion.usage:
            chunks.append({**head, "choices": [], "usage": completion.usage.model_dump()})
        lines = [ChatCompletionChunk.model_validate(c).model_dump_json(exclude_none=True) for c in chunks]
        return [f"data: {line}\n\n" for line in [*lines, "[DONE]"]]

    @classmethod
    def error(cls, exc: Exception, status: HTTPStatus) -> dict[str, Any]:
        """OpenAI's error body; a context overflow is worded and coded as OpenAI's: harnesses match on both."""
        message, code, param = str(exc), None, None
        match exc:
            case ValidationError():
                errors = exc.errors()
                message = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in errors)
                loc = errors[0]["loc"]
                if loc and isinstance(loc[0], str):
                    param = f"{loc[0]}[{loc[1]}]" if len(loc) > 1 and isinstance(loc[1], int) else loc[0]
            case ContextLengthExceededError():
                message = (
                    f"This model's maximum context length is {exc.context_length} tokens. "
                    f"However, your messages resulted in {exc.prompt_len} tokens."
                )
                code, param = "context_length_exceeded", "messages"
        kind = "invalid_request_error" if status < HTTPStatus.INTERNAL_SERVER_ERROR else "server_error"
        return {"error": {"message": message, "type": kind, "param": param, "code": code}}
