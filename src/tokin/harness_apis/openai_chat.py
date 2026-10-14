from __future__ import annotations

import time
from collections.abc import Iterator
from http import HTTPStatus
from typing import Annotated, Any, cast
from uuid import uuid4

from openai.types.chat import ChatCompletion
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

from ..gateway import ChatRequest, ChatResponse, ContextLengthExceededError, StaleRequestError
from ..messages import Message, ToolSchema
from ..rollout import FinishReason, GenerationParams
from .harness_request import HarnessRequest

# `extra="forbid"` would refuse unknown keys in messages too; they are dropped instead, since no template reads them.
_MESSAGES_VALIDATOR: TypeAdapter[list[Message]] = TypeAdapter(
    list[Annotated[Message, Field(discriminator="role")]], config=ConfigDict(extra="ignore")
)


class OpenAIChatRequest(HarnessRequest, extra="forbid"):
    """The harness's `/v1/chat/completions` body, as much of OpenAI's as `tokin` serves.

    A key not declared is refused by name rather than dropped, since dropping it would change
    what the harness gets back without telling it; serving one more is declaring it.
    """

    model: str
    messages: list[Message]
    tools: list[ToolSchema] | None = None
    max_tokens: int | None = Field(None, ge=1, validation_alias=AliasChoices("max_completion_tokens", "max_tokens"))
    temperature: float | None = None
    top_p: float | None = None

    @field_validator("tools", mode="wrap")
    @classmethod
    def validate_tools(cls, value: Any, handler: ValidatorFunctionWrapHandler) -> Any:
        """Validated, but kept as the harness sent them.

        Templates print tools whole with `tojson`, and `pydantic` rebuilds a TypedDict in its declared
        order, which for OpenAI's tool type puts `function` before `type`; harnesses and servers put
        `type` first.
        """
        handler(value)
        return value

    @field_validator("messages", mode="plain")
    @classmethod
    def validate_messages(cls, value: Any) -> list[Message]:
        """Validated by `_MESSAGES_VALIDATOR`, with OpenAI's `Iterable` fields read into lists.

        `pydantic` leaves those fields as iterators, which read once and cannot be copied, and
        whose bad items fail only when read; read here, they fail with status 400.
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
        """The answer to this request; a stop that made tool calls is `tool_calls`, which harnesses branch on."""
        stopped = reply.generation.finish_reason is FinishReason.STOP
        reason = "tool_calls" if stopped and reply.message.get("tool_calls") else reply.generation.finish_reason
        return ChatCompletion.model_validate(
            {
                "id": f"chatcmpl-{uuid4().hex}",
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

    @classmethod
    def error(cls, exc: Exception, status: HTTPStatus) -> dict[str, Any]:
        """OpenAI's error body; a context overflow is worded and coded as OpenAI's, which harnesses match to trim history."""
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
            case StaleRequestError():
                param = "messages"
        kind = "invalid_request_error" if status < HTTPStatus.INTERNAL_SERVER_ERROR else "server_error"
        return {"error": {"message": message, "type": kind, "param": param, "code": code}}
