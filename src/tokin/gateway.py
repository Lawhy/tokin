from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import ClassVar, cast

from .chat_templates import ChatTemplate
from .inference_engines import InferenceEngine, InferenceEngineError
from .messages import AssistantMessage, Message, PromptMessage, ToolSchema
from .rollout import FinishReason, Generation, GenerationParams, Prompt, Rollout, Session


class GatewayError(Exception):
    """A request the gateway refuses; the subclass says with which status."""

    status_code: ClassVar[HTTPStatus] = HTTPStatus.INTERNAL_SERVER_ERROR


class ContextLengthExceededError(GatewayError):
    status_code = HTTPStatus.BAD_REQUEST


class GenerationAbortedError(GatewayError):
    """The engine cut the generation short; nothing was recorded, so the harness may send the turn again."""

    status_code = HTTPStatus.SERVICE_UNAVAILABLE


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """One turn asked of the gateway, in tokin's vocabulary whichever API the harness speaks.

    Attributes:
        messages (list[Message]): The conversation as the harness sends it, whole; only those
            past what the session has served are rendered.
        tools (list[ToolSchema] | None): The tools the model may call.
        params (GenerationParams): What the caller asks of the engine. `max_tokens` is capped
            to the context left, which it is when unset.
    """

    messages: list[Message]
    tools: list[ToolSchema] | None = None
    params: GenerationParams = field(default_factory=GenerationParams)


class Gateway:
    """tokin's core as a plain object: one turn on a session the caller keeps.

    Free of any web framework, so a trainer can drive it in-process and tests
    exercise it without a client. What it refuses, it refuses with a
    `GatewayError` whose subclass names the HTTP status, and the HTTP layer
    renders that once instead of choosing a status route by route.
    """

    def __init__(self, template: ChatTemplate, engine: InferenceEngine) -> None:
        self.template = template
        self.engine = engine
        self.context_length: int | None = None
        self._context_length_lock = asyncio.Lock()

    def render(self, rollout: Rollout, messages: list[Message], tools: list[ToolSchema] | None) -> str:
        """The text `messages` add to `rollout`: all of them on the first turn, those past what was served after.

        It only reads `rollout`, so it can preview or count a turn without running it.
        """
        if not rollout.segments:
            return self.template.apply(messages, tools)
        answered = cast(AssistantMessage, rollout.messages[-1]).get("tool_calls")
        tail = cast(list[PromptMessage], messages[len(rollout.messages) :])
        return self.template.apply_increment(tail, tools, tool_calls=answered)

    async def chat(self, session: Session, request: ChatRequest) -> tuple[AssistantMessage, Generation, int]:
        """Run `request` as the next turn of `session`, holding its lock for the whole turn.

        Returns:
            `(message, generation, input_len)`: the reply parsed, the generation it was parsed
            from, and how many tokens the engine was given, which is the whole rollout and not only
            this turn's prompt.
        """
        if self.context_length is None:
            # Only ask the engine once, however many first turns arrive together.
            async with self._context_length_lock:
                if self.context_length is None:
                    self.context_length = await self.engine.context_length()
        async with session.lock:
            rollout = session.current
            # 1. Render: what this turn adds to the rollout, as ids.
            text = self.render(rollout, request.messages, request.tools)
            prompt = Prompt(self.template.encode(text))
            # 2. Budget: what the context leaves for this turn's generation.
            room = self.context_length - len(rollout) - len(prompt)
            if room < 1:
                raise ContextLengthExceededError(
                    f"{len(rollout) + len(prompt)} prompt tokens leave no room in {self.context_length}"
                )
            params: GenerationParams = {**request.params, "stop_ids": self.template.stop_ids}
            # Cut to the room left: a ceiling, so lowering it changes no generation that could fit.
            params["max_tokens"] = min(params.get("max_tokens", room), room)
            if params.get("return_routed_experts"):
                params["routed_experts_start"] = max(0, len(rollout) - 1)
            # 3. Generate: the engine sees the rollout and the prompt, as ids only.
            generation = await self.engine.generate(rollout.token_ids + prompt.token_ids, params)
            # 4. Check: a stop must end on a stop id, since the next increment drops that token as already sampled.
            if generation.finish_reason is FinishReason.ABORT:
                raise GenerationAbortedError("the engine aborted the generation")
            stopped = generation.finish_reason is FinishReason.STOP
            if stopped and (not generation.token_ids or generation.token_ids[-1] not in self.template.stop_ids):
                raise InferenceEngineError(
                    f"the engine reported a stop ending on {generation.token_ids[-1:]}, not a stop id"
                )
            # 5. Parse: the reply the harness gets.
            message = self.template.parse(generation, request.tools)
            # 6. Record: tokens and messages together, so a turn that failed above leaves the rollout as it was.
            rollout.append(prompt, request.messages[len(rollout.messages) :])
            rollout.append(generation, [message])
            return message, generation, len(rollout) - len(generation)
