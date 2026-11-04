from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace
from http import HTTPStatus
from typing import Any, ClassVar, cast

from .chat_templates import ChatTemplate
from .inference_engines import InferenceEngine, InferenceEngineError
from .messages import AssistantMessage, Message, PromptMessage, ToolSchema
from .rollout import FinishReason, Generation, GenerationParams, Prompt, Rollout, Turn
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
    """One turn the gateway ran, as recorded in the session.

    Attributes:
        message (AssistantMessage): The reply, parsed from `generation`.
        generation (Generation): What the engine sampled for this turn.
        input_len (int): How many tokens the engine was given: the whole path so far, not only
            this turn's prompt.
        id (str): The recorded turn's, for the harness's completion: the key that joins its log to the session.
    """

    message: AssistantMessage
    generation: Generation
    input_len: int
    id: str


class Gateway:
    """Runs turns of sessions: renders what a turn adds, generates on the engine, parses the reply, records both.

    No web framework: callable in-process. A refusal is a `GatewayError` subclass carrying its HTTP status.
    """

    reserved: ClassVar[frozenset[str]] = frozenset({"stop_ids", "routed_experts_start", "session_id"})

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

    def render(self, rollout: Rollout, messages: list[Message], tools: list[ToolSchema] | None) -> str:
        """The text `messages` add to `rollout`, which they extend: all of them to a new one, else those it lacks.

        Reads `rollout` only: usable to preview or count a turn without running it.
        """
        if not rollout.turns:
            return self.template.apply(messages, tools)
        answered = cast(AssistantMessage, rollout.messages[-1]).get("tool_calls")
        tail = cast(list[PromptMessage], messages[len(rollout.messages) :])
        return self.template.apply_increment(tail, tools, tool_calls=answered)

    async def resume(self, ids: list[int], sampled: Generation, params: GenerationParams) -> Generation:
        """The generation of `ids` that `sampled`, aborted, continues into, joined as if never cut.

        `params` are those `sampled` ran with. It comes back aborted again if the engine aborts once more, for `chat`
        to resume again. Asking again at once does not spin: sglang holds a request sent while it pauses until it
        continues.
        """
        # 1. Nothing sampled: the engine aborted before the first id; ask again whole.
        if not sampled.token_ids:
            return await self.engine.generate(ids, params)
        # 2. Budget spent: aborted on reaching `max_tokens`, the generation ends there as if cut by it.
        left = params["max_tokens"] - len(sampled)
        if left < 1:
            return replace(sampled, finish_reason=FinishReason.LENGTH)
        # 3. Routing: a row per id fed so far, any extra one dropped; the continuation feeds the last sampled id.
        more: GenerationParams = {**params, "max_tokens": left}
        routing = sampled.routed_experts
        if routing is not None:
            layers, top_k = await self.engine.routing()
            fed = len(ids) + len(sampled) - 1
            rows, row = fed - params["routed_experts_start"], 4 * layers * top_k  # int32 expert ids
            if len(routing) not in (rows * row, (rows + 1) * row):
                raise InferenceEngineError(
                    f"an aborted generation carries {len(routing)} routing bytes for {rows} rows"
                )
            routing, more["routed_experts_start"] = routing[: rows * row], fed
        # 4. Continue: from the ids sampled so far; aborted before its first id, it adds nothing to join.
        rest = await self.engine.generate(ids + sampled.token_ids, more)
        if rest.finish_reason is FinishReason.ABORT and not rest.token_ids:
            return sampled
        # 5. Join: one generation, ending as the continuation did.
        return Generation(
            token_ids=sampled.token_ids + rest.token_ids,
            finish_reason=rest.finish_reason,
            logprobs=None if sampled.logprobs is None else sampled.logprobs + (rest.logprobs or []),
            routed_experts=None if routing is None else routing + (rest.routed_experts or b""),
            support=sampled.support + rest.support if sampled.support and rest.support else sampled.support,
            top_logprobs=sampled.top_logprobs + rest.top_logprobs
            if sampled.top_logprobs and rest.top_logprobs
            else sampled.top_logprobs,
            weight_versions=sampled.weight_versions
            + tuple((v, start + len(sampled), end + len(sampled)) for v, start, end in rest.weight_versions),
        )

    async def chat(
        self, session: Session, request: ChatRequest, generating: asyncio.Event | None = None
    ) -> ChatResponse:
        """Run `request` as the next turn of `session`.

        Turns of one session generate concurrently, as a harness's subagents send them: the session's lock is held to
        attach and render a turn and to record it, never while it generates. Two that continue the same turn are
        recorded as its siblings. `generating` is set once the turn has passed every check before the engine and only
        generation is left.
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
            # 1. Render: what this turn adds to the path of the turn it continues, as ids.
            parent = session.attach(request.messages, request.tools)
            rollout = session.rollout(parent)
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
                "session_id": session.id,
            }
            # Cut to the room left: a ceiling, so lowering it changes no generation that could fit.
            params["max_tokens"] = min(params.get("max_tokens", room), room)
            if params.get("return_routed_experts"):
                params["routed_experts_start"] = max(0, len(rollout) - 1)
        # 3. Generate: the engine sees the rollout and the prompt, as ids only.
        if generating:
            generating.set()
        ids = rollout.token_ids + prompt.token_ids
        generation = await self.engine.generate(ids, params)
        while generation.finish_reason is FinishReason.ABORT and session.resume_aborted:
            generation = await self.resume(ids, generation, params)
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
        async with session.lock:
            # 6. Record: tokens and messages together, so a turn that failed above leaves the session as it was.
            added = [*request.messages[len(rollout.messages) :], message]
            turn = Turn(prompt, generation, added, parent, request.tools if parent is None else None)
            session.turns.append(turn)
        return ChatResponse(message, generation, len(rollout) + len(prompt), turn.id)
