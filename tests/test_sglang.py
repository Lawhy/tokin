import base64
import json

import httpx2
import pytest

from tokin.inference_engines import InferenceEngineError, SGLangEngine
from tokin.rollout import Generation

REPLY = {
    "text": "ok",
    "output_ids": [5, 6, 3],
    "meta_info": {
        "finish_reason": {"type": "stop", "matched": 3},
        "output_token_logprobs": [[-0.1, 5, None], [-0.2, 6, None], [-0.3, 3, None]],
    },
}


class FakeSGLang:
    """Answers `/generate` from a script: an int is a status code for `reply`, an exception is raised instead."""

    def __init__(self, reply=REPLY, script=(200,)):
        self.requests = []
        self.reply, self.script = reply, list(script)

    def handle(self, request):
        assert request.url.path == "/generate"
        self.requests.append(json.loads(request.content))
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(step, Exception):
            raise step
        return httpx2.Response(step, json=self.reply)


async def call(fake, retries=3, **params):
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(fake.handle)) as client:
        engine = SGLangEngine("http://sglang/", client, retries=retries, retry_delay=0)
        return await engine.generate([1, 2], {"stop_ids": frozenset({7, 3}), **params})


def models(max_model_len):
    reply = {"object": "list", "data": [{"id": "m", "max_model_len": max_model_len}]}
    return httpx2.AsyncClient(transport=httpx2.MockTransport(lambda r: httpx2.Response(200, json=reply)))


async def test_context_length_is_the_model_cards():
    async with models(32768) as client:
        assert await SGLangEngine("http://sglang/", client).context_length() == 32768


async def test_context_length_is_retried():
    card = {"object": "list", "data": [{"id": "m", "max_model_len": 32768}]}
    answers = [httpx2.ConnectError("refused"), httpx2.Response(503), httpx2.Response(200, json=card)]

    def handle(request):
        if isinstance(step := answers.pop(0), Exception):
            raise step
        return step

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        assert await SGLangEngine("http://sglang/", client, retries=3, retry_delay=0).context_length() == 32768


async def test_ready_polls_until_sglang_generates():
    answers, paths = [httpx2.ConnectError("refused"), httpx2.Response(503), httpx2.Response(200)], []

    def handle(request):
        paths.append(request.url.path)
        if isinstance(step := answers.pop(0), Exception):
            raise step
        return step

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        await SGLangEngine("http://sglang/", client, retry_delay=0).ready()
    assert paths == ["/health_generate"] * 3


async def test_missing_context_length_is_an_error():
    async with models(None) as client:
        with pytest.raises(InferenceEngineError, match="no context length"):
            await SGLangEngine("http://sglang/", client).context_length()


class TestGenerate:
    async def test_speaks_ids_and_keeps_the_stop_token(self):
        fake = FakeSGLang()
        got = await call(fake, max_tokens=8, temperature=0.5)
        assert got == Generation(token_ids=[5, 6, 3], finish_reason="stop", logprobs=None)
        [body] = fake.requests
        assert body["input_ids"] == [1, 2] and body["stream"] is False
        assert "return_logprob" not in body and "rid" not in body
        # Only what was set travels; the engine keeps its own defaults for the rest.
        assert body["sampling_params"] == {"max_new_tokens": 8, "temperature": 0.5, "stop_token_ids": [3, 7]}

    async def test_optional_knobs_travel_only_when_set(self):
        fake = FakeSGLang()
        await call(fake, max_tokens=8, top_k=20, seed=7, response_schema={"type": "object"}, request_id="s1:t3")
        [body] = fake.requests
        assert body["rid"] == "s1:t3"
        params = body["sampling_params"]
        assert (params["top_k"], params["sampling_seed"], params["json_schema"]) == (20, 7, '{"type": "object"}')

    async def test_logprobs_take_the_first_of_each_triple(self):
        fake = FakeSGLang()
        got = await call(fake, max_tokens=8, return_logprobs=True)
        assert got.logprobs == [-0.1, -0.2, -0.3] and fake.requests[0]["return_logprob"] is True

    async def test_length_passes_through(self):
        reply = {**REPLY, "meta_info": {**REPLY["meta_info"], "finish_reason": {"type": "length", "length": 3}}}
        assert (await call(FakeSGLang(reply), max_tokens=3)).finish_reason == "length"

    async def test_abort_keeps_what_was_sampled(self):
        finish = {"type": "abort", "message": "Aborted", "status_code": None, "err_type": None}
        reply = {**REPLY, "output_ids": [5], "meta_info": {**REPLY["meta_info"], "finish_reason": finish}}
        assert await call(FakeSGLang(reply), max_tokens=3) == Generation(token_ids=[5], finish_reason="abort")

    async def test_unknown_finish_reason_is_refused(self):
        reply = {**REPLY, "meta_info": {**REPLY["meta_info"], "finish_reason": {"type": "retracted"}}}
        with pytest.raises(InferenceEngineError, match="retracted"):
            await call(FakeSGLang(reply), max_tokens=3)

    async def test_routed_experts_come_back_as_the_raw_buffer(self):
        buffer = bytes(range(24))
        reply = {**REPLY, "meta_info": {**REPLY["meta_info"], "routed_experts": base64.b64encode(buffer).decode()}}
        fake = FakeSGLang(reply)
        got = await call(fake, max_tokens=8, return_routed_experts=True, routed_experts_start=9)
        [body] = fake.requests
        assert body["return_routed_experts"] is True and body["routed_experts_start_len"] == 9
        assert "routed_experts_start_len" not in body["sampling_params"] and got.routed_experts == buffer

    async def test_unknown_keys_are_sampling_params(self):
        fake = FakeSGLang()
        await call(fake, max_tokens=8, top_n_sigma=1.5)
        assert fake.requests[0]["sampling_params"]["top_n_sigma"] == 1.5

    async def test_every_failure_is_retried(self):
        fake = FakeSGLang(script=(httpx2.ConnectError("refused"), 400, 503, 200))
        assert (await call(fake, retries=4, max_tokens=3)).finish_reason == "stop"
        assert len(fake.requests) == 4

    async def test_gives_up_after_the_retries(self):
        fake = FakeSGLang(script=(503,))
        with pytest.raises(InferenceEngineError, match="/generate failed 3 times, last: .*503"):
            await call(fake, retries=3, max_tokens=3)
        assert len(fake.requests) == 3
