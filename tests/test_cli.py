import pytest
from click.testing import CliRunner
from conftest import ChatML, FakeTokenizer

from tokin import cli


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


def invoke(*args, **env):
    return CliRunner().invoke(cli.main, args, env=env)


def test_serves_the_app_with_its_settings(served):
    result = invoke("--model-path", "m", "--chat-template", "chatml", "--port", "9100", "--timeout-keep-alive", "30")
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
    assert invoke(TOKIN_MODEL_PATH="m", TOKIN_CHAT_TEMPLATE="chatml", TOKIN_PORT="9200").exit_code == 0
    assert served[0][1]["port"] == 9200


def test_build_reads_the_template_and_engine():
    server = cli.build("m", "chatml", '{"enable_thinking": false}', "http://e:1/", 5)
    assert isinstance(server.gateway.template, ChatML) and str(server.gateway.engine.client.base_url) == "http://e:1/"
    assert server.gateway.template.apply([{"role": "user", "content": "hi"}]).endswith("<think></think>")


def test_template_kwargs_must_be_json(served):
    result = invoke("--model-path", "m", "--chat-template", "chatml", "--chat-template-kwargs", "{oops")
    assert result.exit_code == 2 and "not JSON" in result.output


def test_a_model_no_family_lists_needs_its_template_named(served):
    result = invoke("--model-path", "/models/local")
    assert isinstance(result.exception, ValueError) and "unknown chat template '/models/local'" in str(result.exception)
