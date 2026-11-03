import asyncio

import numpy as np
import pytest
from conftest import END, said

from tokin.gateway import (
    ChatRequest,
    ContextLengthExceededError,
    GenerationAbortedError,
    ReservedParamsError,
)
from tokin.inference_engines import InferenceEngineError
from tokin.rollout import FinishReason, Generation, Support, TopLogprobs
from tokin.session import Session

USER = {"role": "user", "content": "hi"}
REPLY = {"role": "assistant", "content": "ok"}
AGAIN = {"role": "user", "content": "again"}
OTHER = {"role": "user", "content": "call a tool"}


def rows(*values):
    return np.array(values, dtype=np.int32).tobytes()


def widths(*counts):
    # Support of these widths at successive positions.
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype("<i4")
    return Support(bytes(4 * sum(counts)), bytes(4 * sum(counts)), offsets.tobytes())


def heads(positions, k=2):
    return TopLogprobs(bytes(4 * k * positions), bytes(4 * k * positions), k)


class TestResume:
    @pytest.fixture
    def resumed(self, gateway):
        async def run(*replies, routing=None, **params):
            g = gateway(*replies, routing=routing)
            s = Session(resume_aborted=True)
            await g.chat(s, ChatRequest([USER], params=params))
            return g, s

        return run

    async def test_continues_from_the_ids_it_sampled(self, resumed):
        g, s = await resumed(
            Generation(
                [ord("o")],
                FinishReason.ABORT,
                [-1.0],
                support=widths(2),
                top_logprobs=heads(1),
                weight_versions=(("1", 0, 1),),
            ),
            Generation(
                [ord("k"), ord(END)],
                FinishReason.STOP,
                [-2.0, -3.0],
                support=widths(1, 3),
                top_logprobs=heads(2),
                weight_versions=(("2", 0, 2),),
            ),
        )
        (first, asked), (then, more) = g.engine.calls
        assert then == first + [ord("o")] and more["max_tokens"] == asked["max_tokens"] - 1
        assert s.turns[0].generation == Generation(
            [ord("o"), ord("k"), ord(END)],
            FinishReason.STOP,
            [-1.0, -2.0, -3.0],
            support=widths(2, 1, 3),
            top_logprobs=heads(3),
            weight_versions=(("1", 0, 1), ("2", 1, 3)),
        )
        assert s.turns[0].messages[-1] == REPLY

    async def test_joins_routing_without_the_extra_row(self, resumed, chatml):
        # The aborted call routes every id fed, its last sampled one included; the continuation owns that row.
        n = len(chatml.encode(chatml.apply([USER])))
        g, s = await resumed(
            Generation([ord("o")], FinishReason.ABORT, routed_experts=rows(*range(n), 99)),
            Generation([ord("k"), ord(END)], FinishReason.STOP, routed_experts=rows(n, n + 1)),
            routing=(1, 1),
            return_routed_experts=True,
        )
        assert g.engine.calls[1][1]["routed_experts_start"] == n
        assert s.rollouts[0].routed_experts(layers=1, top_k=1).ravel().tolist() == list(range(n + 2))

    async def test_a_continuation_that_samples_nothing_is_asked_again(self, resumed):
        g, s = await resumed(
            Generation([ord("o")], FinishReason.ABORT, [-1.0]),
            Generation([], FinishReason.ABORT),
            Generation([ord("k"), ord(END)], FinishReason.STOP, [-2.0, -3.0]),
        )
        assert g.engine.calls[2] == g.engine.calls[1] and s.turns[0].generation.logprobs == [-1.0, -2.0, -3.0]

    async def test_an_abort_at_the_budget_ends_as_cut_by_it(self, resumed):
        g, s = await resumed(Generation([ord("o")], FinishReason.ABORT), max_tokens=1)
        assert len(g.engine.calls) == 1 and s.turns[0].generation.finish_reason is FinishReason.LENGTH

    async def test_asks_again_whole_when_nothing_was_sampled(self, resumed):
        g, _ = await resumed(Generation([], FinishReason.ABORT), said("ok"))
        assert g.engine.calls[1] == g.engine.calls[0]

    async def test_refuses_routing_of_the_wrong_size(self, resumed):
        with pytest.raises(InferenceEngineError, match="routing bytes"):
            await resumed(
                Generation([ord("o")], FinishReason.ABORT, routed_experts=rows(0)),
                routing=(1, 1),
                return_routed_experts=True,
            )


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

    async def test_a_served_turn_sent_again_opens_a_rollout_of_its_own(self, gateway, chatml):
        # As two subagents given one task open, or a harness resends a turn whose reply it lost.
        g, s = gateway(said("ok"), said("ok")), Session()
        await g.chat(s, ChatRequest([USER]))
        await g.chat(s, ChatRequest([USER]))
        assert g.engine.calls[1][0] == chatml.encode(chatml.apply([USER]))
        assert [r.messages for r in s.rollouts] == [[USER, REPLY], [USER, REPLY]]

    async def test_a_rewritten_history_opens_a_rollout_rendered_in_full(self, gateway, chatml):
        # As a harness does that compacts, keeping none of the turns it was served.
        g, s = gateway(said("ok"), said("ok")), Session()
        await g.chat(s, ChatRequest([USER]))
        await g.chat(s, ChatRequest([USER, AGAIN]))
        assert g.engine.calls[1][0] == chatml.encode(chatml.apply([USER, AGAIN]))
        assert [r.messages for r in s.rollouts] == [[USER, REPLY], [USER, AGAIN, REPLY]]

    async def test_a_dropped_reply_branches_from_the_turn_before_it(self, gateway, chatml):
        # As mini-swe-agent does, swapping a reply with no tool call for a reminder to make one.
        g, s = gateway(*[said("ok")] * 3), Session()
        await g.chat(s, ChatRequest([USER]))
        kept = s.rollouts[0].token_ids
        await g.chat(s, ChatRequest([USER, REPLY, AGAIN]))
        await g.chat(s, ChatRequest([USER, REPLY, OTHER]))
        assert g.engine.calls[2][0] == kept + chatml.encode(chatml.apply_increment([OTHER]))
        assert [r.messages for r in s.rollouts] == [[USER, REPLY, AGAIN, REPLY], [USER, REPLY, OTHER, REPLY]]

    async def test_a_branch_renders_the_replies_its_history_kept(self, gateway, chatml):
        # As a harness that edits an earlier reply with tool calls, and keeps its result: the increment holds both.
        call = '<tool_call>{"name": "f", "arguments": {}}</tool_call>'
        g, s = gateway(said(call), said(call), said("ok")), Session()
        tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
        await g.chat(s, ChatRequest([USER], tools))
        first = s.turns[0].messages[-1]
        result = {"role": "tool", "tool_call_id": first["tool_calls"][0]["id"], "content": "1"}
        await g.chat(s, ChatRequest([USER, first, result], tools))
        edited = {**s.turns[1].messages[-1], "content": "edited"}
        again = {"role": "tool", "tool_call_id": edited["tool_calls"][0]["id"], "content": "2"}
        await g.chat(s, ChatRequest([USER, first, result, edited, again], tools))
        tail = chatml.apply_increment([result, edited, again], tools, tool_calls=first["tool_calls"])
        assert g.engine.calls[2][0] == s.rollout(0).token_ids + chatml.encode(tail)
        assert [t.parent for t in s.turns] == [None, 0, 0]

    async def test_a_turn_cut_inside_a_call_renders_as_the_model_wrote_it(self, gateway, chatml):
        # As a harness that keeps the cut call and answers it: the next turn renders that reply as prompt.
        cut = '<tool_call>{"name": "f", "arguments": {"x": 1'
        g, s = gateway(said(cut, FinishReason.LENGTH), said("ok")), Session()
        await g.chat(s, ChatRequest([USER]))
        reply = s.turns[0].messages[-1]
        result = {"role": "tool", "tool_call_id": reply["tool_calls"][0]["id"], "content": "bad arguments"}
        await g.chat(s, ChatRequest([USER, reply, result]))
        assert cut in "".join(map(chr, g.engine.calls[1][0])) and s.turns[1].parent is None

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
        assert [len(r.messages) for r in s.rollouts] == [2, 4]

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
        assert not s.turns and not s.lock.locked()

    async def test_turns_of_one_session_generate_concurrently(self, gateway):
        # As a harness's subagents do: neither waits on the other's generation.
        g, s, started, both = gateway(), Session(), [], asyncio.Event()

        async def generate(token_ids, params):
            started.append(token_ids)
            if len(started) == 2:
                both.set()
            await both.wait()
            return said("ok")

        g.engine.generate = generate
        await asyncio.wait_for(asyncio.gather(*(g.chat(s, ChatRequest([m])) for m in (USER, AGAIN))), 1)
        assert sorted(len(r.messages) for r in s.rollouts) == [2, 2]
