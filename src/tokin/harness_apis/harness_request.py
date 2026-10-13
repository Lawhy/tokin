from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from ..messages import AssistantMessage, Message, ToolSchema
from ..rollout import Generation, GenerationParams


class HarnessRequest(BaseModel, ABC):
    """A harness's request in one wire API, and the way back to that API's response."""

    @abstractmethod
    def chat(self) -> tuple[list[Message], list[ToolSchema] | None, GenerationParams]:
        """What the request asks of the gateway, in tokin's vocabulary: messages, tools and params."""

    @abstractmethod
    def respond(self, message: AssistantMessage, generation: Generation, input_len: int) -> BaseModel:
        """The response this API gives for one turn the gateway ran."""
