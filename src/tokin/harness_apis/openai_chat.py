from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Annotated, Any, cast
from uuid import uuid4

from openai.types.chat import ChatCompletion
from pydantic import AliasChoices, ConfigDict, Field, TypeAdapter, field_validator

from ..gateway import ChatRequest
from ..messages import AssistantMessage, Message, ToolSchema
from ..rollout import FinishReason, Generation, GenerationParams
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

    @field_validator("messages", mode="plain")
    @classmethod
    def validate_messages(cls, value: Any) -> list[Message]:
        """Validated by `_MESSAGES_VALIDATOR`, with OpenAI's `Iterable` fields read into lists so a bad item fails here as a 400."""
        messages = _MESSAGES_VALIDATOR.validate_python(value)
        for m in cast(list[dict[str, Any]], messages):
            for key, field in m.items():
                if isinstance(field, Iterator):
                    m[key] = list(field)
        return messages

    def chat_request(self) -> ChatRequest:
        # A field named as a `GenerationParams` key means the same to the engine.
        params = self.model_dump(include=set(GenerationParams.__optional_keys__), exclude_none=True)
        return ChatRequest(self.messages, self.tools, cast(GenerationParams, params))

    def response(self, message: AssistantMessage, generation: Generation, input_len: int) -> ChatCompletion:
        """The answer to this request; a stop that made tool calls is `tool_calls`, which harnesses branch on."""
        stopped = generation.finish_reason is FinishReason.STOP
        reason = "tool_calls" if stopped and message.get("tool_calls") else generation.finish_reason
        return ChatCompletion.model_validate(
            {
                "id": f"chatcmpl-{uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model,
                "choices": [{"index": 0, "message": message, "finish_reason": reason}],
                "usage": {
                    "prompt_tokens": input_len,
                    "completion_tokens": len(generation),
                    "total_tokens": input_len + len(generation),
                },
            }
        )
