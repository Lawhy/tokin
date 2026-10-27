from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import cast
from uuid import uuid4

import httpx2
import numpy as np
from safetensors.numpy import load, save

from .messages import AssistantMessage, Message, ToolSchema
from .rollout import FinishReason, Generation, GenerationParams, Prompt, Rollout, Turn


@dataclass
class Session:
    """One agent run: a tree of the turns its harness was served, and the params its turns run with.

    A turn continues the deepest turn whose path the request begins with, so a request that drops a reply or edits
    what followed it branches off where it left the path, keeping the ids sampled before. One that shares no such path
    with any turn, as a compaction or a subagent does, starts a root. Each leaf ends a rollout.
    """

    id: str = field(default_factory=lambda: uuid4().hex)
    turns: list[Turn] = field(default_factory=list)
    params: GenerationParams = field(default_factory=GenerationParams)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, compare=False)

    def __len__(self) -> int:
        return sum(len(turn.prompt) + len(turn.generation) for turn in self.turns)

    def __repr__(self) -> str:
        return f"Session(id={self.id!r}, turns={len(self.turns)}, rollouts={len(self.rollouts)}, tokens={len(self)})"

    def rollout(self, turn: int | None) -> Rollout:
        """The rollout from a root down to `turn`, the empty one for `None`."""
        path = []
        while turn is not None:
            path.append(self.turns[turn])
            turn = self.turns[turn].parent
        return Rollout(path[::-1])

    @property
    def rollouts(self) -> list[Rollout]:
        """The rollout of every leaf, in the order the leaves were served."""
        parents = {turn.parent for turn in self.turns}
        return [self.rollout(i) for i in range(len(self.turns)) if i not in parents]

    def attach(self, messages: list[Message], tools: list[ToolSchema] | None) -> int | None:
        """The turn `messages` continue, or `None` for a new root.

        It is the deepest turn whose path they begin with and add to, the latest on a tie:

        - a root matches only the `tools` it was rendered with: they render into its first prompt;
        - a reply matches by its text, reasoning and tool call ids, an empty text however written: harnesses echo
          those as served;
        - a call's name and arguments are not compared: harnesses rewrite them (pi `{}`, opencode `invalid`), and the
          ids the model sampled stand for them;
        - a turn cut at `length` continues nothing: it ends before its stop id, so a request that keeps its reply
          renders it as prompt under its parent.
        """

        def served(kept: Message, sent: Message) -> bool:
            if kept["role"] != sent["role"]:
                return False
            if kept["role"] != "assistant":
                return kept == sent
            echo = cast(AssistantMessage, sent)
            return (
                (kept.get("content") or None) == (echo.get("content") or None)
                and kept.get("reasoning_content") == echo.get("reasoning_content")
                and [c["id"] for c in kept.get("tool_calls") or []] == [c["id"] for c in echo.get("tool_calls") or []]
            )

        children: dict[int | None, list[int]] = {}
        for i, turn in enumerate(self.turns):
            children.setdefault(turn.parent, []).append(i)
        best: tuple[int, int] | None = None
        stack = [(i, 0) for i in children.get(None, []) if (self.turns[i].tools or None) == (tools or None)]
        while stack:
            i, start = stack.pop()
            turn = self.turns[i]
            end = start + len(turn.messages)
            if len(messages) <= end:
                # The request ends within or at this turn, so it leaves nothing past it to generate.
                continue
            if not all(map(served, turn.messages, messages[start:end])):
                continue
            if turn.generation.finish_reason is FinishReason.LENGTH:
                # Cut before its stop id: an increment after it would open on a stop id the model never sampled.
                continue
            if best is None or (end, i) > best:
                best = (end, i)
            stack.extend((child, end) for child in children.get(i, []))
        return None if best is None else best[1]

    def pack(self) -> bytes:
        """The session as one safetensors payload, the format a trainer collects and `unpack` reads.

        Each turn `t` is stored once, however many rollouts run through it:

            `{t}.prompt`          `int32`    the ids rendered for it
            `{t}.generation`      `int32`    the ids sampled
            `{t}.logprobs`        `float32`  the generation's logprobs, when it carries them
            `{t}.routed_experts`  `int32`    the generation's routing, the engine's buffer as it came

        `metadata` is UTF-8 JSON as uint8, a tensor because `safetensors.numpy.load` drops header metadata:

            `{"id": ..., "params": {...}, "turns": [{"parent": ..., "id": ..., "finish_reason": ..., "messages": [...],
            "tools": [...]}]}`

        in the order the turns were served, so `parent` indexes `turns`. Rollouts, loss masks, flat logprobs and routing
        are left out: `Session.rollout` and a `Rollout` derive them.
        """
        tensors: dict[str, np.ndarray] = {}
        turns = []
        for t, turn in enumerate(self.turns):
            tensors[f"{t}.prompt"] = np.array(turn.prompt.token_ids, dtype=np.int32)
            tensors[f"{t}.generation"] = np.array(turn.generation.token_ids, dtype=np.int32)
            if turn.generation.logprobs is not None:
                tensors[f"{t}.logprobs"] = np.array(turn.generation.logprobs, dtype=np.float32)
            if turn.generation.routed_experts is not None:
                tensors[f"{t}.routed_experts"] = np.frombuffer(turn.generation.routed_experts, dtype=np.int32)
            turns.append(
                {
                    "parent": turn.parent,
                    "id": turn.id,
                    "finish_reason": turn.generation.finish_reason,
                    "messages": turn.messages,
                    "tools": turn.tools,
                }
            )
        meta = json.dumps({"id": self.id, "params": self.params, "turns": turns}).encode()
        tensors["metadata"] = np.frombuffer(meta, dtype=np.uint8)
        return save(tensors)

    @classmethod
    def unpack(cls, payload: bytes) -> Session:
        """The session `pack` wrote."""
        tensors = load(payload)
        meta = json.loads(tensors["metadata"].tobytes())
        if "turns" not in meta:
            raise ValueError("a payload of rollouts, which tokin wrote before sessions were trees of turns")
        turns = []
        for t, turn in enumerate(meta["turns"]):
            logprobs, experts = tensors.get(f"{t}.logprobs"), tensors.get(f"{t}.routed_experts")
            generation = Generation(
                tensors[f"{t}.generation"].tolist(),
                FinishReason(turn["finish_reason"]),
                logprobs=None if logprobs is None else logprobs.tolist(),
                routed_experts=None if experts is None else experts.tobytes(),
            )
            prompt = Prompt(tensors[f"{t}.prompt"].tolist())
            turns.append(Turn(prompt, generation, turn["messages"], turn["parent"], turn["tools"], turn["id"]))
        return cls(meta["id"], turns, meta["params"])

    @classmethod
    async def open(cls, client: httpx2.AsyncClient, params: GenerationParams | None = None) -> str:
        """Open a session on the tokin `client` reaches and return its id: a harness takes `/sessions/<id>/v1`."""
        answer = await client.post("/sessions", json={"params": params or GenerationParams()})
        return str(answer.raise_for_status().json()["id"])

    @classmethod
    async def collect(cls, client: httpx2.AsyncClient, sid: str) -> Session:
        """The session `sid` on the tokin `client` reaches, then deleted there, whether or not it could be read."""
        try:
            return cls.unpack((await client.get(f"/sessions/{sid}")).raise_for_status().content)
        finally:
            await client.delete(f"/sessions/{sid}")
