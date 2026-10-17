import httpx2
import numpy as np
import pytest
from conftest import said

from tokin.rollout import FinishReason, Generation, Prompt, Rollout
from tokin.server import Server
from tokin.session import Session

USER = {"role": "user", "content": "hi"}
REPLY = {"role": "assistant", "content": "ok"}
RESULT = {"role": "tool", "tool_call_id": "c1", "content": "done"}


def test_unpack_restores_what_pack_wrote():
    routed = np.arange(3 * 2, dtype=np.int32).tobytes()  # positions 0..2 of 4 ids, in 2 layers, top-1
    carried = Rollout()
    carried.append(Prompt([1, 2]), [USER])
    carried.append(Generation([3, 4], FinishReason.STOP, logprobs=[-0.5, -0.25], routed_experts=routed), [REPLY])
    bare = Rollout()
    bare.append(Prompt([5]), [USER, RESULT])
    bare.append(Generation([6], FinishReason.LENGTH), [REPLY])
    session = Session("s1", [carried, bare], {"temperature": 1.0})
    assert Session.unpack(session.pack()) == session


def test_session_repr_summarises_without_dumping_tokens():
    s = Session(rollouts=[Rollout([Prompt([1, 2, 3])])])
    assert repr(s) == f"Session(id={s.id!r}, rollouts=1, tokens=3)"


def test_sessions_do_not_share_their_rollouts():
    a, b = Session(), Session()
    a.rollouts.append(Rollout())
    assert not b.rollouts


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
