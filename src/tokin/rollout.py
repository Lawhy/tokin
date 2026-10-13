from __future__ import annotations

import asyncio
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypedDict
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from .messages import Message


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
    stop_ids: Collection[int]

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

    `routed_experts` is the engine's buffer as it came: int32, C order, shape `(positions, layers, top_k)`
    flattened, positions running from `routed_experts_start` up to the last sampled token, which predicts
    nothing. The trainer reshapes it with its model's layer count and top-k; MoE training replays the routing.
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


@dataclass
class Rollout:
    """One conversation, as token ids and as messages.

    Attributes:
        segments (list[Prompt | Generation]): The token ids in engine order: the prompts `tokin`
            rendered and the generations the model sampled. Concatenated they are the full
            sequence; which segment an id falls in is its loss mask.
        messages (list[Message]): The same conversation as OpenAI messages: what the harness
            sent, plus the assistant message parsed from each generation. Not one to one with
            `segments`, since one prompt renders every message added since the last generation.
    """

    segments: list[Prompt | Generation] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)

    @property
    def token_ids(self) -> list[int]:
        """Every segment's ids concatenated — the prefix the next prompt extends."""
        return [i for s in self.segments for i in s.token_ids]

    @property
    def loss_mask(self) -> list[int]:
        """1 at each id the model sampled and 0 at each one `tokin` rendered, aligned with `token_ids`."""
        return [int(isinstance(s, Generation)) for s in self.segments for _ in s.token_ids]

    @property
    def logprobs(self) -> list[float]:
        """Each sampled id's logprob and 0.0 at each rendered one, aligned with `token_ids`."""
        logprobs: list[float] = []
        for i, segment in enumerate(self.segments):
            if isinstance(segment, Prompt):
                logprobs += [0.0] * len(segment)
            elif segment.logprobs is None:
                raise ValueError(f"segment {i} was generated without its logprobs")
            else:
                logprobs += segment.logprobs
        return logprobs

    def append(self, segment: Prompt | Generation, messages: Sequence[Message]) -> None:
        """Add a segment with the messages it carries: those a prompt rendered, or the reply a generation was read as.

        A rollout opens with a prompt. An empty segment adds no segment, but its messages are still
        part of the conversation: a generation of no tokens is still a reply the harness got.
        """
        if segment.token_ids:
            if not self.segments and isinstance(segment, Generation):
                raise RuntimeError("a rollout cannot open with a generation")
            self.segments.append(segment)
        self.messages.extend(messages)

    def routed_experts(self, layers: int, top_k: int) -> NDArray[np.int32]:
        """Every position's expert routing so far, shape `(len(self) - 1, layers, top_k)`, joined from the per-call slices.

        Each generation's slice must run from where the previous one ended to its own second-to-last
        token; the engine sometimes appends one extra row for the final token, which is dropped.
        """
        covered, tokens, chunks = 0, 0, []
        for i, segment in enumerate(self.segments):
            tokens += len(segment)
            if not isinstance(segment, Generation):
                continue
            if segment.routed_experts is None:
                raise ValueError(f"segment {i} was generated without its expert routing")
            # A token's routing is produced when it is fed in, so every token so far has a row but the newest.
            positions = tokens - 1
            x = np.frombuffer(segment.routed_experts, dtype=np.int32)
            rows, rest = divmod(x.size, layers * top_k)
            # +1 is sglang having fed the final token once more, as after an abort; the next turn owns it.
            if rest or rows not in (positions - covered, positions - covered + 1):
                raise ValueError(f"segment {i} carries {rows} routing rows for {positions - covered} positions")
            chunks.append(x.reshape(rows, layers, top_k)[: positions - covered])
            covered = positions
        return np.concatenate(chunks) if chunks else np.empty((0, layers, top_k), dtype=np.int32)

    def __len__(self) -> int:
        return sum(len(s) for s in self.segments)

    def __repr__(self) -> str:
        generated = sum(len(s) for s in self.segments if isinstance(s, Generation))
        return (
            f"Rollout(tokens={len(self)}, generated={generated}, "
            f"segments={len(self.segments)}, messages={len(self.messages)})"
        )


@dataclass
class Session:
    """One agent run, as the rollouts it produced; turns append to `current` in place.

    A list although a run has one rollout for now: a run whose history is rewritten
    continues in a fresh one, and every rollout of a run shares its outcome.
    """

    id: str = field(default_factory=lambda: uuid4().hex)
    rollouts: list[Rollout] = field(default_factory=lambda: [Rollout()])
    # One turn at a time; the gateway holds it from prompt to parse.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, compare=False)

    @property
    def current(self) -> Rollout:
        """The rollout new turns append to."""
        return self.rollouts[-1]

    def __len__(self) -> int:
        return sum(len(rollout) for rollout in self.rollouts)

    def __repr__(self) -> str:
        return f"Session(id={self.id!r}, rollouts={len(self.rollouts)}, tokens={len(self)})"
