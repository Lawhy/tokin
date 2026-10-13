from __future__ import annotations

import time
from typing import Annotated, Any, ClassVar, cast
from uuid import uuid4

from openai.types.chat import ChatCompletion
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidatorFunctionWrapHandler,
    field_validator,
)

from ..messages import AssistantMessage, Message, ToolSchema
from ..rollout import FinishReason, Generation, GenerationParams


class ChatRequest(BaseModel, extra="forbid"):
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

    # SDKs decorate messages with keys no template reads, so messages drop what the body refuses.
    messages_adapter: ClassVar[TypeAdapter[list[Message]]] = TypeAdapter(
        list[Annotated[Message, Field(discriminator="role")]], config=ConfigDict(extra="ignore")
    )

    @field_validator("tools", mode="wrap")
    @classmethod
    def as_sent(cls, value: Any, handler: ValidatorFunctionWrapHandler) -> Any:
        """Validated but kept as the harness sent them: templates print tools with `tojson`, so their key order is prompt."""
        handler(value)
        return value

    @field_validator("messages", mode="plain")
    @classmethod
    def lenient(cls, value: Any) -> list[Message]:
        """Validate `messages` by the adapter, reading OpenAI's `Iterable` fields into lists so a bad item fails here as a 400."""
        messages = cls.messages_adapter.validate_python(value)
        for m in (cast(dict[str, Any], m) for m in messages):
            for key in ("content", "tool_calls"):
                if not isinstance(value := m.get(key), str | None):
                    m[key] = list(value)
        return messages

    @property
    def params(self) -> GenerationParams:
        """What the harness asked of the engine, under `GenerationParams`' names; a field named as one of its keys means the same."""
        return cast(
            GenerationParams, self.model_dump(include=set(GenerationParams.__optional_keys__), exclude_none=True)
        )

    def respond(self, message: AssistantMessage, generation: Generation, input_len: int) -> ChatCompletion:
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
