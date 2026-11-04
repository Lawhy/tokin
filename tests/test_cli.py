import asyncio
import json

import httpx2
import pytest
from click.testing import CliRunner
from conftest import END, ChatML, FakeTokenizer

from tokin import cli
from tokin.rollout import FinishReason, Generation
from tokin.server import Server


@pytest.fixture(autouse=True)
def chatml_family(monkeypatch):
    known = cli.get_chat_template
    monkeypatch.setattr(cli, "get_chat_template", lambda name: ChatML if name == "chatml" else known(name))
    monkeypatch.setattr(cli.AutoTokenizer, "from_pretrained", lambda path: FakeTokenizer())


@pytest.fixture
def served(monkeypatch):
    runs = []
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **settings: runs.append((app, settings)))
    return runs


@pytest.fixture
def tokin(monkeypatch, gateway):
    """A server `tokin open` and `tokin collect` reach in process, whose engine samples "ok" with its logprobs."""
    server = Server(
        gateway(Generation([ord(c) for c in "ok" + END], FinishReason.STOP, logprobs=[-0.5, -0.25, -0.125]))
    )
    client = httpx2.AsyncClient
    transport = httpx2.ASGITransport(app=server.app)
    monkeypatch.setattr(cli.httpx2, "AsyncClient", lambda base_url: client(transport=transport, base_url=base_url))
    return server


def invoke(*args, **env):
    return CliRunner().invoke(cli.main, args, env=env)


def serve(*args, **env):
    return invoke("serve", *args, **env)


def test_serves_the_app_with_its_settings(served):
    result = serve("--model-path", "m", "--chat-template", "chatml", "--port", "9100", "--timeout-keep-alive", "30")
    [(app, settings)] = served
    assert result.exit_code == 0
    assert settings == {
        "host": "127.0.0.1",
        "port": 9100,
        "log_level": "info",
        "access_log": False,
        "timeout_keep_alive": 30,
    }
    assert {r.path for r in app.routes} >= {"/health", "/sessions", "/sessions/{sid}/v1/chat/completions"}


def test_options_come_from_the_environment(served):
    assert serve(TOKIN_MODEL_PATH="m", TOKIN_CHAT_TEMPLATE="chatml", TOKIN_PORT="9200").exit_code == 0
    assert served[0][1]["port"] == 9200


def test_build_reads_the_template_and_engine():
    server = cli.build("m", "chatml", '{"enable_thinking": false}', "http://e:1/", 5)
    assert isinstance(server.gateway.template, ChatML) and str(server.gateway.engine.client.base_url) == "http://e:1/"
    assert server.gateway.template.apply([{"role": "user", "content": "hi"}]).endswith("<think></think>")


def test_template_kwargs_must_be_json(served):
    result = serve("--model-path", "m", "--chat-template", "chatml", "--chat-template-kwargs", "{oops")
    assert result.exit_code == 2 and "not JSON" in result.output


def test_a_model_no_family_lists_needs_its_template_named(served):
    result = serve("--model-path", "/models/local")
    assert isinstance(result.exception, ValueError) and "unknown chat template '/models/local'" in str(result.exception)


def test_open_prints_the_session_base_url(tokin):
    result = invoke("open", "--params", '{"temperature": 0.5}')
    [(sid, session)] = tokin.sessions.items()
    assert result.output == f"http://127.0.0.1:8000/sessions/{sid}/v1\n" and session.params == {"temperature": 0.5}


def test_collect_writes_each_rollout_and_deletes_the_session(tokin):
    url = invoke("open", "--params", '{"return_logprobs": true}').output.strip()

    async def turn():
        async with cli.httpx2.AsyncClient(base_url=url) as client:
            reply = await client.post(
                "/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]}
            )
            return reply.raise_for_status()

    asyncio.run(turn())
    [rollout] = next(iter(tokin.sessions.values())).rollouts
    [line] = invoke("collect", url).output.splitlines()
    assert json.loads(line) == {
        "token_ids": rollout.token_ids,
        "loss_mask": rollout.loss_mask,
        "logprobs": rollout.logprobs,
    }
    assert not tokin.sessions


def test_collect_refuses_a_url_that_is_not_a_sessions():
    result = invoke("collect", "http://127.0.0.1:8000/v1")
    assert result.exit_code == 2 and "not a session's base URL" in result.output
