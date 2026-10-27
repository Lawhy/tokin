import json

import httpx2
import numpy as np
import pytest
from conftest import said
from safetensors.numpy import load, save

from tokin.rollout import FinishReason, Generation, Prompt, Turn
from tokin.server import Server
from tokin.session import Session

USER = {"role": "user", "content": "hi"}
REPLY = {"role": "assistant", "content": "ok"}
RESULT = {"role": "tool", "tool_call_id": "c1", "content": "done"}
AGAIN = {"role": "user", "content": "again"}
OTHER = {"role": "user", "content": "call a tool"}
TOOL = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}


def turn(prompt, generation, messages, parent=None, tools=None, finish=FinishReason.STOP, **kwargs):
    return Turn(Prompt(prompt), Generation(generation, finish, **kwargs), messages, parent, tools)


def test_unpack_restores_what_pack_wrote():
    routed = np.arange(3 * 2, dtype=np.int32).tobytes()  # positions 0..2 of 4 ids, in 2 layers, top-1
    carried = turn([1, 2], [3, 4], [USER, REPLY], tools=[TOOL], logprobs=[-0.5, -0.25], routed_experts=routed)
    bare = turn([5], [6], [RESULT, REPLY], parent=0, finish=FinishReason.LENGTH)
    session = Session("s1", [carried, bare], {"temperature": 1.0})
    assert Session.unpack(session.pack()) == session


def test_unpack_refuses_a_payload_of_rollouts():
    meta = np.frombuffer(json.dumps({"id": "s1", "params": {}, "rollouts": []}).encode(), dtype=np.uint8)
    with pytest.raises(ValueError, match="before sessions were trees of turns"):
        Session.unpack(save({"metadata": meta}))


def test_a_shared_turn_is_packed_once():
    session = Session(
        turns=[turn([1], [2], [USER, REPLY]), turn([3], [4], [AGAIN, REPLY], 0), turn([5], [6], [OTHER, REPLY], 0)]
    )
    assert len([k for k in load(session.pack()) if k.endswith(".prompt")]) == 3


def test_rollouts_are_the_paths_to_the_leaves():
    session = Session(
        turns=[turn([1], [2], [USER, REPLY]), turn([3], [4], [AGAIN, REPLY], 0), turn([5], [6], [OTHER, REPLY], 0)]
    )
    assert [r.token_ids for r in session.rollouts] == [[1, 2, 3, 4], [1, 2, 5, 6]]
    assert session.rollouts[1].messages == [USER, REPLY, OTHER, REPLY]


def test_session_repr_summarises_without_dumping_tokens():
    s = Session(turns=[turn([1, 2], [3], [USER, REPLY])])
    assert repr(s) == f"Session(id={s.id!r}, turns=1, rollouts=1, tokens=3)"


def test_sessions_do_not_share_their_turns():
    a, b = Session(), Session()
    a.turns.append(turn([1], [2], [USER, REPLY]))
    assert not b.turns


CALLED = {
    "role": "assistant",
    "content": "ok",
    "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
}


class TestAttach:
    def test_continues_the_turn_its_path_ends_at(self):
        s = Session(turns=[turn([1], [2], [USER, CALLED])])
        assert s.attach([USER, CALLED, RESULT], None) == 0

    def test_matches_a_reply_whose_calls_were_rewritten(self):
        # As pi sends unreadable arguments back as `{}` and opencode renames such a call `invalid`.
        call = {**CALLED["tool_calls"][0], "function": {"name": "invalid", "arguments": "{}"}}
        s = Session(turns=[turn([1], [2], [USER, CALLED])])
        assert s.attach([USER, {**CALLED, "tool_calls": [call]}, RESULT], None) == 0

    def test_matches_an_empty_text_however_written(self):
        s = Session(turns=[turn([1], [2], [USER, {**CALLED, "content": None}])])
        assert (
            s.attach([USER, {k: v for k, v in CALLED.items() if k != "content"} | {"content": ""}, RESULT], None) == 0
        )

    def test_a_reply_of_other_text_opens_a_root(self):
        assert (
            Session(turns=[turn([1], [2], [USER, REPLY])]).attach([USER, {**REPLY, "content": "edited"}, AGAIN], None)
            is None
        )

    def test_a_reply_of_other_calls_opens_a_root(self):
        other = {**CALLED, "tool_calls": [{**CALLED["tool_calls"][0], "id": "c2"}]}
        assert Session(turns=[turn([1], [2], [USER, CALLED])]).attach([USER, other, RESULT], None) is None

    def test_a_dropped_reply_branches_from_the_turn_before(self):
        s = Session(turns=[turn([1], [2], [USER, REPLY]), turn([3], [4], [AGAIN, REPLY], 0)])
        assert s.attach([USER, REPLY, OTHER], None) == 0

    def test_takes_the_deepest_turn(self):
        s = Session(turns=[turn([1], [2], [USER, REPLY]), turn([3], [4], [AGAIN, REPLY], 0)])
        assert s.attach([USER, REPLY, AGAIN, REPLY, OTHER], None) == 1

    def test_takes_the_latest_of_equal_turns(self):
        # Two turns from one request sent twice: the second is the one the harness went on with.
        s = Session(turns=[turn([1], [2], [USER, REPLY]), turn([1], [3], [USER, REPLY])])
        assert s.attach([USER, REPLY, AGAIN], None) == 1

    def test_messages_sent_again_open_a_root(self):
        assert Session(turns=[turn([1], [2], [USER, REPLY])]).attach([USER, REPLY], None) is None

    def test_a_cut_turn_is_continued_from_its_parent(self):
        s = Session(
            turns=[turn([1], [2], [USER, REPLY]), turn([3], [4], [AGAIN, REPLY], 0, finish=FinishReason.LENGTH)]
        )
        assert s.attach([USER, REPLY, AGAIN, REPLY, OTHER], None) == 0

    def test_other_tools_open_a_root(self):
        s = Session(turns=[turn([1], [2], [USER, REPLY], tools=[TOOL])])
        assert s.attach([USER, REPLY, AGAIN], None) is None
        assert s.attach([USER, REPLY, AGAIN], [TOOL]) == 0


def test_sessions_do_not_share_a_lock():
    assert Session().lock is not Session().lock


def test_session_id_is_fresh_unless_given():
    assert Session().id != Session().id and len(Session().id) == 32
    assert Session(id="episode-7").id == "episode-7"


@pytest.fixture
def tokin(gateway):
    server = Server(gateway(said("ok")))
    return server, httpx2.AsyncClient(transport=httpx2.ASGITransport(app=server.app), base_url="http://tokin")


async def test_an_opened_session_serves_and_is_collected_then_deleted(tokin):
    server, http = tokin
    sid = await Session.open(http, {"temperature": 1.0})
    turn = await http.post(f"/sessions/{sid}/v1/chat/completions", json={"model": "m", "messages": [USER]})
    got = await Session.collect(http, sid)
    assert turn.status_code == 200 and not server.sessions
    assert got.id == sid and got.params == {"temperature": 1.0}
    assert [r.messages for r in got.rollouts] == [[USER, {"role": "assistant", "content": "ok"}]]


async def test_a_session_that_cannot_be_read_is_still_deleted(tokin, monkeypatch):
    server, http = tokin
    sid = await Session.open(http)

    def unreadable(cls, payload):
        raise ValueError("bad")

    monkeypatch.setattr(Session, "unpack", classmethod(unreadable))
    with pytest.raises(ValueError, match="bad"):
        await Session.collect(http, sid)
    assert not server.sessions
