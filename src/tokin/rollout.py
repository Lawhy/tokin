from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypedDict
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from .messages import Message, ToolSchema


@dataclass(frozen=True, slots=True)
class Prompt:
    """Ids the harness's messages rendered to, as the engine received them."""

    token_ids: list[int]

    def __len__(self) -> int:
        return len(self.token_ids)


class GenerationParams(TypedDict, total=False):
    """Parameters for a single generation request; a key left out takes the model's own default."""

    # Identity: names the request to the inference engine.
    request_id: str

    # Length and stopping.
    max_tokens: int
    stop_ids: frozenset[int]

    # Distribution: how the next token is drawn.
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    seed: int

    # Penalties on tokens already in the sequence.
    frequency_penalty: float
    presence_penalty: float
    repetition_penalty: float

    # Constrained decoding / Structured output.
    response_schema: dict[str, Any]

    # Additional data to return besides the ids
    return_logprobs: bool
    return_routed_experts: bool
    routed_experts_start: int


class FinishReason(StrEnum):
    """Why a generation ended, as the engine reports it."""

    STOP = "stop"  # on one of the template's stop ids
    LENGTH = "length"  # `max_tokens`
    ABORT = "abort"  # the engine cut it short, as on a weight update in asynchronous training


@dataclass(frozen=True, slots=True)
class Generation:
    """What the model sampled for one prompt, as ids, including the stop id it ended on.

    `routed_experts` is the engine's buffer as it came: int32, C order, `(positions, layers, top_k)` flattened,
    positions from `routed_experts_start` to the last sampled token, which predicts nothing.
    """

    token_ids: list[int]
    finish_reason: FinishReason
    logprobs: list[float] | None = None
    routed_experts: bytes | None = None

    def __post_init__(self) -> None:
        # Whether logprobs are wanted is the trainer's call; alignment is not.
        if self.logprobs is not None and len(self.logprobs) != len(self.token_ids):
            raise ValueError(f"logprobs has {len(self.logprobs)} entries but token_ids has {len(self.token_ids)}")

    def __len__(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True, slots=True)
class Turn:
    """One generation and the prompt rendered for it: a node of a session's tree.

    `messages` are those the request added past its parent's path, then the reply read from `generation`. `parent` is
    the index in `Session.turns` of the turn it continues, `None` for a root. A root holds the `tools` it was rendered
    with, which hold for every turn under it. `id` is the completion's, as the harness got it.
    """

    prompt: Prompt
    generation: Generation
    messages: list[Message]
    parent: int | None = None
    tools: list[ToolSchema] | None = None
    id: str = field(default_factory=lambda: f"chatcmpl-{uuid4().hex}")


@dataclass(frozen=True, slots=True)
class Rollout:
    """One conversation: a session's turns from a root down to one of them, as token ids and as messages."""

    turns: list[Turn] = field(default_factory=list)

    @property
    def messages(self) -> list[Message]:
        """What the harness sent, with the reply read from each generation."""
        return [m for t in self.turns for m in t.messages]

    @property
    def token_ids(self) -> list[int]:
        """Every turn's prompt and generation concatenated — the prefix the next prompt extends."""
        return [i for t in self.turns for i in (*t.prompt.token_ids, *t.generation.token_ids)]

    @property
    def loss_mask(self) -> list[int]:
        """1 at each id the model sampled and 0 at each one `tokin` rendered, aligned with `token_ids`."""
        return [m for t in self.turns for m in [0] * len(t.prompt) + [1] * len(t.generation)]

    @property
    def logprobs(self) -> list[float]:
        """Each sampled id's logprob and 0.0 at each rendered one, aligned with `token_ids`."""
        logprobs: list[float] = []
        for i, turn in enumerate(self.turns):
            if turn.generation.logprobs is None:
                raise ValueError(f"turn {i} was generated without its logprobs")
            logprobs += [0.0] * len(turn.prompt) + turn.generation.logprobs
        return logprobs

    def routed_experts(self, layers: int, top_k: int) -> NDArray[np.int32]:
        """Every position's expert routing so far, `(len(self) - 1, layers, top_k)`, joined from each turn's slice.

        Each generation's slice must run from where the previous one ended to its own second-to-last
        token; the engine sometimes appends one extra row for the final token, which is dropped.
        """
        covered, tokens, chunks = 0, 0, []
        for i, turn in enumerate(self.turns):
            tokens += len(turn.prompt) + len(turn.generation)
            if turn.generation.routed_experts is None:
                raise ValueError(f"turn {i} was generated without its expert routing")
            # A token's routing is produced when it is fed in, so every token so far has a row but the newest.
            positions = tokens - 1
            x = np.frombuffer(turn.generation.routed_experts, dtype=np.int32)
            rows, rest = divmod(x.size, layers * top_k)
            # +1 is sglang having fed the final token once more, as after an abort; the next turn owns it.
            if rest or rows not in (positions - covered, positions - covered + 1):
                raise ValueError(f"turn {i} carries {rows} routing rows for {positions - covered} positions")
            chunks.append(x.reshape(rows, layers, top_k)[: positions - covered])
            covered = positions
        return np.concatenate(chunks) if chunks else np.empty((0, layers, top_k), dtype=np.int32)

    def __len__(self) -> int:
        return sum(len(t.prompt) + len(t.generation) for t in self.turns)

    def __repr__(self) -> str:
        generated = sum(len(t.generation) for t in self.turns)
        return f"Rollout(tokens={len(self)}, generated={generated}, turns={len(self.turns)})"
