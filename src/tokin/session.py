from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from uuid import uuid4

import httpx2
import numpy as np
from safetensors.numpy import load, save

from .rollout import FinishReason, Generation, GenerationParams, Prompt, Rollout


@dataclass
class Session:
    """One agent run: a rollout per history its harness kept, and the params its turns run with.

    A turn extends the rollout whose messages its own begin with; one that rewrote what was served, by dropping a
    reply or compacting, starts a rollout of its own. Each rollout holds at least one turn.
    """

    id: str = field(default_factory=lambda: uuid4().hex)
    rollouts: list[Rollout] = field(default_factory=list)
    params: GenerationParams = field(default_factory=GenerationParams)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, compare=False)

    def __len__(self) -> int:
        return sum(len(rollout) for rollout in self.rollouts)

    def __repr__(self) -> str:
        return f"Session(id={self.id!r}, rollouts={len(self.rollouts)}, tokens={len(self)})"

    def pack(self) -> bytes:
        """The session as one safetensors payload, the format a trainer collects and `unpack` reads.

        Each segment `s` of rollout `r` is stored as it was recorded:

            `{r}.{s}.token_ids`       `int32`    its ids
            `{r}.{s}.logprobs`        `float32`  a generation's logprobs, when it carries them
            `{r}.{s}.routed_experts`  `int32`    a generation's routing, the engine's buffer as it came

        `metadata` is UTF-8 JSON as uint8, a tensor because `safetensors.numpy.load` drops header metadata:

            `{"id": ..., "params": {...}, "rollouts": [{"messages": [...], "segments": ["prompt", "stop", ...]}]}`

        where each entry of `segments` is `"prompt"` or the generation's finish reason. Loss mask, flat logprobs
        and routing are left out: a `Rollout` derives them.
        """
        tensors: dict[str, np.ndarray] = {}
        rollouts = []
        for r, rollout in enumerate(self.rollouts):
            for s, segment in enumerate(rollout.segments):
                tensors[f"{r}.{s}.token_ids"] = np.array(segment.token_ids, dtype=np.int32)
                if isinstance(segment, Generation) and segment.logprobs is not None:
                    tensors[f"{r}.{s}.logprobs"] = np.array(segment.logprobs, dtype=np.float32)
                if isinstance(segment, Generation) and segment.routed_experts is not None:
                    tensors[f"{r}.{s}.routed_experts"] = np.frombuffer(segment.routed_experts, dtype=np.int32)
            kinds = [s.finish_reason if isinstance(s, Generation) else "prompt" for s in rollout.segments]
            rollouts.append({"messages": rollout.messages, "segments": kinds})
        meta = json.dumps({"id": self.id, "params": self.params, "rollouts": rollouts}).encode()
        tensors["metadata"] = np.frombuffer(meta, dtype=np.uint8)
        return save(tensors)

    @classmethod
    def unpack(cls, payload: bytes) -> Session:
        """The session `pack` wrote."""
        tensors = load(payload)
        meta = json.loads(tensors["metadata"].tobytes())
        rollouts = []
        for r, rollout in enumerate(meta["rollouts"]):
            segments: list[Prompt | Generation] = []
            for s, kind in enumerate(rollout["segments"]):
                token_ids = tensors[f"{r}.{s}.token_ids"].tolist()
                if kind == "prompt":
                    segments.append(Prompt(token_ids))
                    continue
                logprobs, experts = tensors.get(f"{r}.{s}.logprobs"), tensors.get(f"{r}.{s}.routed_experts")
                segments.append(
                    Generation(
                        token_ids,
                        FinishReason(kind),
                        logprobs=None if logprobs is None else logprobs.tolist(),
                        routed_experts=None if experts is None else experts.tobytes(),
                    )
                )
            rollouts.append(Rollout(segments, rollout["messages"]))
        return cls(meta["id"], rollouts, meta["params"])

    @classmethod
    async def open(cls, tokin: httpx2.AsyncClient, params: GenerationParams | None = None) -> str:
        """Open a session on the tokin `tokin` reaches and return its id: a harness takes `/sessions/<id>/v1`."""
        answer = await tokin.post("/sessions", json={"params": params or GenerationParams()})
        return str(answer.raise_for_status().json()["id"])

    @classmethod
    async def collect(cls, tokin: httpx2.AsyncClient, sid: str) -> Session:
        """The session `sid` on the tokin `tokin` reaches, then deleted there, whether or not it could be read."""
        try:
            return cls.unpack((await tokin.get(f"/sessions/{sid}")).raise_for_status().content)
        finally:
            await tokin.delete(f"/sessions/{sid}")
