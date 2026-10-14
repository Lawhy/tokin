import httpx2
import pytest
from conftest import said
from openai import AsyncOpenAI, BadRequestError, ConflictError, InternalServerError, NotFoundError

from tokin.rollout import FinishReason, Generation
from tokin.server import create_app

USER = {"role": "user", "content": "hi"}


@pytest.fixture
def serve(gateway):
    def make(*replies, context_length=1000):
        app = create_app(gateway(*replies, context_length=context_length))
        return httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="http://tokin")

    return make


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
