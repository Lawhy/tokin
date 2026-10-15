from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

from ..rollout import Generation, GenerationParams


class InferenceEngineError(RuntimeError):
    """The inference server answered outside what the protocol allows."""


class InferenceEngine(ABC):
    """An inference server spoken to in token ids only."""

    # Where the engine spells a `GenerationParams` key differently; a key not listed keeps its name.
    param_map: ClassVar[dict[str, str]] = {}
    # How a value is reshaped for the engine's wire format; a key not listed goes as it is.
    param_convert: ClassVar[dict[str, Callable[[Any], Any]]] = {}

    def translate(self, params: GenerationParams) -> dict[str, Any]:
        """The params under the engine's names and shapes; two keys landing on one name is refused."""
        out: dict[str, Any] = {}
        for key, value in params.items():
            name = self.param_map.get(key, key)
            if name in out:
                raise ValueError(f"{key!r} lands on {name!r}, which another key already set")
            out[name] = self.param_convert[key](value) if key in self.param_convert else value
        return out

    @abstractmethod
    async def ready(self) -> None:
        """Return once the engine can generate, however long that takes; the caller bounds the wait."""
        raise NotImplementedError

    @abstractmethod
    async def context_length(self) -> int:
        """The most tokens a prompt and its generation may total; `InferenceEngineError` when the engine cannot say."""
        raise NotImplementedError

    @abstractmethod
    async def generate(
        self,
        token_ids: list[int],
        params: GenerationParams,
    ) -> Generation:
        raise NotImplementedError
