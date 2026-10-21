from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, ClassVar, cast

from .chat_templates import ChatTemplate
from .inference_engines import InferenceEngine, InferenceEngineError
from .messages import AssistantMessage, Message, PromptMessage, ToolSchema
from .rollout import FinishReason, Generation, GenerationParams, Prompt, Rollout
from .session import Session

logger = logging.getLogger(__name__)


class GatewayError(Exception):
    """A request the gateway refuses; the subclass says with which status."""

    status_code: ClassVar[HTTPStatus] = HTTPStatus.INTERNAL_SERVER_ERROR


class ContextLengthExceededError(GatewayError):
    status_code = HTTPStatus.BAD_REQUEST

    def __init__(self, prompt_len: int, context_length: int) -> None:
        super().__init__(f"{prompt_len} prompt tokens leave no room in {context_length}")
        self.prompt_len, self.context_length = prompt_len, context_length


class GenerationAbortedError(GatewayError):
    """The engine cut the generation short; nothing was recorded, so the harness may send the turn again."""

    status_code = HTTPStatus.SERVICE_UNAVAILABLE


class ReservedParamsError(GatewayError):
    """The params name a key the gateway sets itself for each turn."""

    status_code = HTTPStatus.BAD_REQUEST


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """The input of `Gateway.chat`: one turn, independent of the harness's wire API.

    Attributes:
        messages (list[Message]): The conversation as the harness sends it, whole; only those
            past what the session has served are rendered.
        tools (list[ToolSchema] | None): The tools the model may call.
        params (GenerationParams): What the caller asks of the engine. The session's params fill
            what it leaves out and overrule what it sets otherwise. `max_tokens` is capped to the
            context left, which it is when unset; `return_logprobs` is on unless set off.
    """

    messages: list[Message]
    tools: list[ToolSchema] | None = None
    params: GenerationParams = field(default_factory=GenerationParams)


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """One turn the gateway ran, as recorded in the session's rollout.

    Attributes:
        message (AssistantMessage): The reply, parsed from `generation`.
        generation (Generation): What the engine sampled for this turn.
        input_len (int): How many tokens the engine was given: the whole rollout so far, not only
            this turn's prompt. Read under the session's lock, since a later turn grows the rollout.
    """

    message: AssistantMessage
    generation: Generation
    input_len: int


class Gateway:
    """Runs turns of sessions: renders what a turn adds, generates on the engine, parses the reply, records both.

    No web framework: callable in-process. A refusal is a `GatewayError` subclass carrying its HTTP status.
    """

    reserved: ClassVar[frozenset[str]] = frozenset({"stop_ids", "routed_experts_start"})

    @classmethod
    def refuse_reserved(cls, params: GenerationParams) -> None:
        if names := sorted(cls.reserved & params.keys()):
            raise ReservedParamsError(f"{names} are set by the gateway for each turn")

    def __init__(self, template: ChatTemplate, engine: InferenceEngine) -> None:
        self.template = template
        self.engine = engine
        self.context_length: int | None = None
        self._context_length_lock = asyncio.Lock()
        self._overruled_logged = False

    def route(self, session: Session, messages: list[Message]) -> Rollout:
        """The rollout `messages` extend, or a new one, not yet in `session`, when they extend none.

        At most one can match: a request that extended a rollout was routed to it rather than starting another. A
        request sent again, or another that opens as an earlier one did, as two subagents with one task do, extends
        none and gets a rollout of its own: a server answers it again.
        """
        return next((rollout for rollout in session.rollouts if rollout.extends(messages)), Rollout())

    def render(self, rollout: Rollout, messages: list[Message], tools: list[ToolSchema] | None) -> str:
        """The text `messages` add to `rollout`, which they extend: all of them to a new one, else those it lacks.

        Reads `rollout` only: usable to preview or count a turn without running it.
        """
        if not rollout.segments:
            return self.template.apply(messages, tools)
        answered = cast(AssistantMessage, rollout.messages[-1]).get("tool_calls")
        tail = cast(list[PromptMessage], messages[len(rollout.messages) :])
        return self.template.apply_increment(tail, tools, tool_calls=answered)

    async def chat(
        self, session: Session, request: ChatRequest, generating: asyncio.Event | None = None
    ) -> ChatResponse:
        """Run `request` as the next turn of `session`, holding its lock for the whole turn.

        `generating` is set once the turn has passed every check before the engine and only generation is left.
        """
        self.refuse_reserved(request.params)
        fixed, asked = cast(dict[str, Any], session.params), cast(dict[str, Any], request.params)
        # Session params win: the policy trains under the trainer's sampling, not the harness's.
        clash = sorted(k for k in fixed.keys() & asked.keys() if fixed[k] != asked[k])
        if clash and not self._overruled_logged:
            self._overruled_logged = True
            logger.warning(
                "requests ask %s, their sessions fix %s; the sessions' are used (logged once)",
                {k: asked[k] for k in clash},
                {k: fixed[k] for k in clash},
            )
        if self.context_length is None:
            # Only ask the engine once, however many first turns arrive together.
            async with self._context_length_lock:
                if self.context_length is None:
                    self.context_length = await self.engine.context_length()
        async with session.lock:
            # 1. Render: what this turn adds to the rollout it extends, as ids.
            rollout = self.route(session, request.messages)
            text = self.render(rollout, request.messages, request.tools)
            prompt = Prompt(self.template.encode(text))
            # 2. Budget: what the context leaves for this turn's generation.
            room = self.context_length - len(rollout) - len(prompt)
            if room < 1:
                raise ContextLengthExceededError(len(rollout) + len(prompt), self.context_length)
            params: GenerationParams = {
                "return_logprobs": True,
                **request.params,
                **session.params,
                "stop_ids": self.template.stop_ids,
            }
            # Cut to the room left: a ceiling, so lowering it changes no generation that could fit.
            params["max_tokens"] = min(params.get("max_tokens", room), room)
            if params.get("return_routed_experts"):
                params["routed_experts_start"] = max(0, len(rollout) - 1)
            # 3. Generate: the engine sees the rollout and the prompt, as ids only.
            if generating:
                generating.set()
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
            # 6. Record: tokens and messages together, so a turn that failed above leaves the session as it was.
            if not rollout.messages:
                session.rollouts.append(rollout)
            rollout.append(prompt, request.messages[len(rollout.messages) :])
            rollout.append(generation, [message])
            return ChatResponse(message, generation, len(rollout) - len(generation))
