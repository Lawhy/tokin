import asyncio
import json

import httpx2
import numpy as np
import pytest
from conftest import said
from openai import AsyncOpenAI, BadRequestError, ConflictError, InternalServerError, NotFoundError
from safetensors.numpy import load

from tokin.inference_engines import InferenceEngineError
from tokin.rollout import FinishReason, Generation, Prompt, Rollout, Session
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
    rollout = Rollout()
    rollout.append(Prompt([1, 2]), [USER])
    rollout.append(Generation([3, 4], FinishReason.STOP, **generated), [{"role": "assistant", "content": "ok"}])
    session = Session("s1", [rollout])
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
    (rollout,) = server.sessions[sid].rollouts
    assert got["token_ids.0"].tolist() == rollout.token_ids and got["loss_mask.0"].tolist() == rollout.loss_mask
    meta = json.loads(got["metadata"].tobytes())
    assert meta["session_id"] == sid and meta["rollouts"][0]["finish_reasons"] == ["stop"]
    assert sorted(got) == ["loss_mask.0", "metadata", "token_ids.0"]


def config_beside(tokenizer, directory, **config):
    (directory / "config.json").write_text(json.dumps(config))
    tokenizer.name_or_path = str(directory)


async def test_logprobs_and_routing_are_collected_when_carried(gateway, chatml, tmp_path):
    config_beside(chatml.tokenizer, tmp_path, model_type="qwen3_moe", num_hidden_layers=2, num_experts_per_tok=1)
    http = over(server := Server(gateway()))
    rows = np.arange(3 * 2, dtype=np.int32).tobytes()  # positions 0..2 of 4 tokens, in the engine's 2 layers, top-1
    got = load((await http.get(f"/sessions/{recorded(server, logprobs=[-0.5, -0.25], routed_experts=rows)}")).content)
    assert got["logprobs.0"].tolist() == [0, 0, -0.5, -0.25] and got["routed_experts.0"].shape == (3, 2, 1)


async def test_routing_beside_a_dense_model_is_refused(gateway, chatml, tmp_path):
    config_beside(chatml.tokenizer, tmp_path, model_type="qwen2", num_hidden_layers=2)
    http = over(server := Server(gateway()))
    with pytest.raises(RuntimeError, match="has no expert routing"):
        await http.get(f"/sessions/{recorded(server, routed_experts=b'\\0' * 24)}")


async def test_routing_without_a_model_config_is_refused(gateway, chatml, tmp_path):
    chatml.tokenizer.name_or_path = str(tmp_path / "elsewhere")
    http = over(server := Server(gateway()))
    with pytest.raises(RuntimeError, match="no model config at"):
        await http.get(f"/sessions/{recorded(server, routed_experts=b'\\0' * 24)}")


async def test_a_deleted_session_is_gone(serve):
    http = serve()
    sid = (await http.post("/sessions")).json()["id"]
    assert (await http.delete(f"/sessions/{sid}")).status_code == 204
    assert (await http.get(f"/sessions/{sid}")).status_code == 404
    with pytest.raises(NotFoundError):
        await (await harness(http, sid)).chat.completions.create(model="m", messages=[USER])


async def test_a_harness_hanging_up_cancels_its_turn(gateway):
    # Driven over raw ASGI, since an HTTP client cannot hang up mid-request on cue.
    server, started, cancelled, hang_up, sent = Server(g := gateway()), asyncio.Event(), [], asyncio.Event(), []

    async def generate(token_ids, params):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    g.engine.generate = generate
    sid = (await over(server).post("/sessions")).json()["id"]
    body = json.dumps({"model": "m", "messages": [USER]}).encode()
    messages = iter([{"type": "http.request", "body": body, "more_body": False}])

    async def receive():
        if (message := next(messages, None)) is not None:
            return message
        await hang_up.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": f"/sessions/{sid}/v1/chat/completions",
        "headers": [(b"content-type", b"application/json")],
        "query_string": b"",
        "http_version": "1.1",
        "scheme": "http",
        "server": ("tokin", 80),
        "client": ("harness", 1),
    }
    call = asyncio.create_task(server.app(scope, receive, send))
    await started.wait()
    hang_up.set()
    await call
    assert cancelled and sent[0]["status"] == 499 and not server.sessions[sid].rollouts


class TestStream:
    async def stream(self, serve, *replies, **params):
        client = await harness(serve(*replies))
        return [
            c async for c in await client.chat.completions.create(model="m", messages=[USER], stream=True, **params)
        ]

    async def test_carries_the_whole_reply(self, serve):
        chunks = await self.stream(serve, said("ok"))
        assert [c.choices[0].delta.content for c in chunks] == ["ok"] and chunks[0].choices[0].finish_reason == "stop"

    async def test_carries_tool_calls_with_their_index(self, serve):
        tool = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
        call = said('<tool_call>\n{"name": "f", "arguments": {"x": 1}}\n</tool_call>')
        [chunk] = await self.stream(serve, call, tools=[tool])
        [delta_call] = chunk.choices[0].delta.tool_calls
        assert (
            delta_call.index == 0 and delta_call.function.name == "f" and chunk.choices[0].finish_reason == "tool_calls"
        )

    async def test_sends_usage_when_asked(self, serve):
        *_, last = await self.stream(serve, said("ok"), stream_options={"include_usage": True})
        assert last.choices == [] and last.usage.completion_tokens == 3


async def test_a_session_runs_its_turns_with_the_params_it_was_opened_with(gateway):
    http = over(Server(g := gateway(said("ok"))))
    sid = (await http.post("/sessions", json={"params": {"temperature": 1.0}})).json()["id"]
    await (await harness(http, sid)).chat.completions.create(model="m", messages=[USER], temperature=0.7)
    assert g.engine.calls[0][1]["temperature"] == 1.0


async def test_opening_a_session_with_a_param_the_gateway_sets_is_400(serve):
    refused = await serve().post("/sessions", json={"params": {"stop_ids": [1]}})
    assert refused.status_code == 400 and "set by the gateway" in refused.json()["error"]["message"]


class TestStartup:
    async def test_waits_for_the_engine_then_learns_its_context(self, gateway):
        server, asked = Server(g := gateway(context_length=4096)), []

        async def ready():
            asked.append("ready")

        g.engine.ready = ready
        async with server.app.router.lifespan_context(server.app):
            assert asked == ["ready"] and g.context_length == 4096

    async def test_an_engine_never_ready_fails_it(self, gateway):
        server = Server(g := gateway(), ready_timeout=0.01)
        g.engine.ready = asyncio.Event().wait
        with pytest.raises(InferenceEngineError, match="could not generate within"):
            async with server.app.router.lifespan_context(server.app):
                pass

    async def test_health_answers_once_started(self, serve):
        assert (await serve().get("/health")).json() == {"status": "ok"}
