import asyncio

import pytest
from conftest import said

from tokin.gateway import (
    ChatRequest,
    ContextLengthExceededError,
    GenerationAbortedError,
    ReservedParamsError,
    StaleRequestError,
)
from tokin.inference_engines import InferenceEngineError
from tokin.rollout import FinishReason, Generation, Session

USER = {"role": "user", "content": "hi"}
REPLY = {"role": "assistant", "content": "ok"}
AGAIN = {"role": "user", "content": "again"}


class TestChat:
    async def test_first_turn_renders_in_full(self, gateway, chatml):
        g = gateway(said("ok"))
        reply = await g.chat(Session(), ChatRequest([USER]))
        assert g.engine.calls[0][0] == chatml.encode(chatml.apply([USER])) and reply.input_len == len(
            g.engine.calls[0][0]
        )
        assert reply.message == {"role": "assistant", "content": "ok"}
        assert reply.generation.finish_reason is FinishReason.STOP

    async def test_later_turn_extends_the_rollout_by_the_increment(self, gateway, chatml):
        g = gateway(said("ok"), said("ok"))
        s = Session()
        await g.chat(s, ChatRequest([USER]))
        served = s.rollouts[0].token_ids
        await g.chat(s, ChatRequest([USER, REPLY, AGAIN]))
        assert g.engine.calls[1][0] == served + chatml.encode(chatml.apply_increment([AGAIN]))

    async def test_prompt_past_the_context_is_refused(self, gateway):
        with pytest.raises(ContextLengthExceededError):
            await gateway(context_length=5).chat(Session(), ChatRequest([USER]))

    async def test_max_tokens_are_capped_to_the_room_left(self, gateway, chatml):
        g = gateway(said("ok"), context_length=len(chatml.encode(chatml.apply([USER]))) + 3)
        await g.chat(Session(), ChatRequest([USER], params={"max_tokens": 100}))
        assert g.engine.calls[0][1]["max_tokens"] == 3

    async def test_unset_max_tokens_take_the_room_left(self, gateway, chatml):
        g = gateway(said("ok"), context_length=len(chatml.encode(chatml.apply([USER]))) + 3)
        await g.chat(Session(), ChatRequest([USER]))
        assert g.engine.calls[0][1]["max_tokens"] == 3

    async def test_logprobs_are_on_unless_turned_off(self, gateway):
        g = gateway(said("ok"), said("ok"))
        await g.chat(Session(), ChatRequest([USER]))
        await g.chat(Session(), ChatRequest([USER], params={"return_logprobs": False}))
        assert [p["return_logprobs"] for _, p in g.engine.calls] == [True, False]

    async def test_session_params_fill_what_the_request_leaves_out(self, gateway):
        g = gateway(said("ok"))
        await g.chat(Session(params={"temperature": 1.0, "top_k": 5}), ChatRequest([USER], params={"temperature": 1.0}))
        assert {k: g.engine.calls[0][1][k] for k in ("temperature", "top_k")} == {"temperature": 1.0, "top_k": 5}

    async def test_session_params_overrule_the_request_logged_once(self, gateway, caplog):
        g, asked = gateway(said("ok"), said("ok")), ChatRequest([USER], params={"temperature": 0.7})
        for _ in range(2):
            await g.chat(Session(params={"temperature": 1.0}), asked)
        assert [p["temperature"] for _, p in g.engine.calls] == [1.0, 1.0]
        assert [r.getMessage() for r in caplog.records] == [
            "requests ask {'temperature': 0.7}, their sessions fix {'temperature': 1.0}; the sessions' are used (logged once)"
        ]

    async def test_params_the_gateway_sets_are_refused(self, gateway):
        with pytest.raises(ReservedParamsError, match="set by the gateway"):
            await gateway().chat(Session(), ChatRequest([USER], params={"stop_ids": frozenset({1})}))

    async def test_routed_experts_start_where_the_rollout_has_them(self, gateway):
        g = gateway(said("ok"), said("ok"))
        s = Session()
        await g.chat(s, ChatRequest([USER], params={"return_routed_experts": True}))
        served = len(s.rollouts[0])
        await g.chat(s, ChatRequest([USER, REPLY, AGAIN], params={"return_routed_experts": True}))
        assert [p["routed_experts_start"] for _, p in g.engine.calls] == [0, served - 1]

    async def test_the_engine_is_asked_its_context_length_once(self, gateway):
        g = gateway(*[said("ok")] * 3)
        g.engine.asked = 0
        length = g.engine.context_length

        async def counted():
            g.engine.asked += 1
            await asyncio.sleep(0)  # let the other first turns arrive while this one waits on the engine
            return await length()

        g.engine.context_length = counted
        await asyncio.gather(*(g.chat(Session(), ChatRequest([USER])) for _ in range(3)))
        assert g.engine.asked == 1

    async def test_stop_off_a_stop_id_is_an_engine_error(self, gateway):
        with pytest.raises(InferenceEngineError, match="stop id"):
            await gateway(Generation([ord("o")], FinishReason.STOP)).chat(Session(), ChatRequest([USER]))

    async def test_engine_abort_is_503_and_records_nothing(self, gateway):
        g = gateway(said("o", FinishReason.ABORT))
        s = Session()
        with pytest.raises(GenerationAbortedError) as e:
            await g.chat(s, ChatRequest([USER]))
        assert e.value.status_code == 503 and not s.rollouts

    async def test_a_served_turn_sent_again_is_409(self, gateway):
        g, s = gateway(said("ok")), Session()
        await g.chat(s, ChatRequest([USER]))
        served = len(s)
        with pytest.raises(StaleRequestError, match="all 1 messages were served already") as e:
            await g.chat(s, ChatRequest([USER]))
        assert e.value.status_code == 409 and len(s) == served

    async def test_a_rewritten_history_opens_a_rollout_rendered_in_full(self, gateway, chatml):
        # As a harness does that drops a reply it could not use and asks again.
        g, s = gateway(said("ok"), said("ok")), Session()
        await g.chat(s, ChatRequest([USER]))
        await g.chat(s, ChatRequest([USER, AGAIN]))
        assert g.engine.calls[1][0] == chatml.encode(chatml.apply([USER, AGAIN]))
        assert [r.messages for r in s.rollouts] == [[USER, REPLY], [USER, AGAIN, REPLY]]

    async def test_a_failed_rewrite_opens_no_rollout(self, gateway):
        g, s = gateway(said("ok"), said("o", FinishReason.ABORT)), Session()
        await g.chat(s, ChatRequest([USER]))
        with pytest.raises(GenerationAbortedError):
            await g.chat(s, ChatRequest([USER, AGAIN]))
        assert [r.messages for r in s.rollouts] == [[USER, REPLY]]

    async def test_a_turn_extends_the_rollout_it_continues(self, gateway, chatml):
        g, s = gateway(*[said("ok")] * 3), Session()
        await g.chat(s, ChatRequest([USER]))
        await g.chat(s, ChatRequest([AGAIN]))
        first = s.rollouts[0].token_ids
        await g.chat(s, ChatRequest([USER, REPLY, AGAIN]))
        assert g.engine.calls[2][0] == first + chatml.encode(chatml.apply_increment([AGAIN]))
        assert [len(r.messages) for r in s.rollouts] == [4, 2]

    async def test_cancelled_turn_records_nothing(self, gateway):
        # A server cancels the turn when the harness disconnects mid-generation.
        g, s, started = gateway(), Session(), asyncio.Event()

        async def generate(token_ids, params):
            started.set()
            await asyncio.Event().wait()

        g.engine.generate = generate
        turn = asyncio.create_task(g.chat(s, ChatRequest([USER])))
        await started.wait()
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn
        assert not s.rollouts and not s.lock.locked()
