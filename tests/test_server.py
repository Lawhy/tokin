import json

import httpx2
import numpy as np
import pytest
from conftest import said
from openai import AsyncOpenAI, BadRequestError, ConflictError, InternalServerError, NotFoundError
from safetensors.numpy import load

from tokin.rollout import FinishReason, Generation, Prompt, Session
from tokin.server import Server

USER = {"role": "user", "content": "hi"}


@pytest.fixture
def serve(gateway):
    def make(*replies, context_length=1000):
        return over(Server(gateway(*replies, context_length=context_length)))

    return make


def over(server):
    return httpx2.AsyncClient(transport=httpx2.ASGITransport(app=server.app), base_url="http://tokin")


# A session recorded without a harness, to collect what the fake engine never returns.
def recorded(server, **generated):
    session = Session("s1")
    session.current.append(Prompt([1, 2]), [USER])
    session.current.append(Generation([3, 4], FinishReason.STOP, **generated), [{"role": "assistant", "content": "ok"}])
    server.sessions[session.id] = session
    return session.id


# The OpenAI SDK a harness would use, pointed at one session; retries off so each test sees one answer.
async def harness(http, sid=None):
    sid = sid or (await http.post("/sessions")).json()["id"]
    return AsyncOpenAI(base_url=f"http://tokin/sessions/{sid}/v1", api_key="none", http_client=http, max_retries=0)


async def test_a_session_serves_chat_completions(serve):
    completion = await (await harness(serve(said("ok")))).chat.completions.create(model="m", messages=[USER])
    assert completion.choices[0].message.content == "ok" and completion.choices[0].finish_reason == "stop"


async def test_an_unknown_session_is_404(serve):
    with pytest.raises(NotFoundError, match="no session 'nope'"):
        await (await harness(serve(), "nope")).chat.completions.create(model="m", messages=[USER])


async def test_an_invalid_body_names_its_param(serve):
    with pytest.raises(BadRequestError) as e:
        await (await harness(serve())).chat.completions.create(model="m", messages=[USER], n=2)
    assert e.value.param == "n" and e.value.type == "invalid_request_error"


async def test_a_context_overflow_is_worded_and_coded_as_openais(serve):
    with pytest.raises(BadRequestError, match="maximum context length is 5 tokens") as e:
        await (await harness(serve(context_length=5))).chat.completions.create(model="m", messages=[USER])
    assert e.value.code == "context_length_exceeded"


async def test_a_stale_turn_is_409_and_not_retried(serve):
    client = await harness(serve(said("ok")))
    await client.chat.completions.create(model="m", messages=[USER])
    with pytest.raises(ConflictError) as e:
        await client.chat.completions.create(model="m", messages=[USER])
    assert e.value.response.headers["x-should-retry"] == "false"


async def test_an_engine_failure_is_502(serve):
    off_stop = Generation([ord("o")], FinishReason.STOP)
    with pytest.raises(InternalServerError) as e:
        await (await harness(serve(off_stop))).chat.completions.create(model="m", messages=[USER])
    assert e.value.status_code == 502 and e.value.type == "server_error"


async def test_a_collected_session_aligns_with_its_rollout(gateway):
    http = over(server := Server(gateway(said("ok"))))
    await (await harness(http, sid := (await http.post("/sessions")).json()["id"])).chat.completions.create(
        model="m", messages=[USER]
    )
    got = load((await http.get(f"/sessions/{sid}")).content)
    rollout = server.sessions[sid].current
    assert got["token_ids.0"].tolist() == rollout.token_ids and got["loss_mask.0"].tolist() == rollout.loss_mask
    meta = json.loads(got["metadata"].tobytes())
    assert meta["session_id"] == sid and meta["rollouts"][0]["finish_reasons"] == ["stop"]
    assert sorted(got) == ["loss_mask.0", "metadata", "token_ids.0"]


async def test_logprobs_and_routing_are_collected_when_carried(gateway):
    http = over(server := Server(gateway()))
    rows = np.arange(3 * 2, dtype=np.int32).tobytes()  # positions 0..2 of 4 tokens, in the engine's 2 layers, top-1
    got = load((await http.get(f"/sessions/{recorded(server, logprobs=[-0.5, -0.25], routed_experts=rows)}")).content)
    assert got["logprobs.0"].tolist() == [0, 0, -0.5, -0.25] and got["routed_experts.0"].shape == (3, 2, 1)


async def test_a_deleted_session_is_gone(serve):
    http = serve()
    sid = (await http.post("/sessions")).json()["id"]
    assert (await http.delete(f"/sessions/{sid}")).status_code == 204
    assert (await http.get(f"/sessions/{sid}")).status_code == 404
    with pytest.raises(NotFoundError):
        await (await harness(http, sid)).chat.completions.create(model="m", messages=[USER])
