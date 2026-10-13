from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from ..gateway import ChatRequest
from ..messages import AssistantMessage
from ..rollout import Generation


class HarnessRequest(BaseModel, ABC):
    """A harness's request in one wire API, and the way back to that API's response."""

    @abstractmethod
    def chat_request(self) -> ChatRequest: ...

    @abstractmethod
    def response(self, message: AssistantMessage, generation: Generation, input_len: int) -> BaseModel:
        """The response this API gives for one turn the gateway ran."""
