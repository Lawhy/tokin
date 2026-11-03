import base64
import json
from types import SimpleNamespace

import httpx2
import numpy as np
import pytest

from tokin.inference_engines import InferenceEngineError, SGLangEngine, sglang
from tokin.inference_engines.client import ShardedTransport
from tokin.rollout import Generation, Support, TopLogprobs

REPLY = {
    "text": "ok",
    "output_ids": [5, 6, 3],
    "meta_info": {
        "finish_reason": {"type": "stop", "matched": 3},
        "output_token_logprobs": [[-0.1, 5, None], [-0.2, 6, None], [-0.3, 3, None]],
    },
}


def b64(array):
    return base64.b64encode(array.tobytes()).decode()


TOPK = {
    "output_topk_token_ids": b64(np.arange(6, dtype="<i4")),
    "output_topk_log_probs": b64(np.full(6, -1.0, dtype="<f4")),
    "output_topk_shape": [3, 2],
}
KEPT = {
    "top_p_token_ids": b64(np.arange(4, dtype="<i4")),
    "top_p_log_probs": b64(np.zeros(4, dtype="<f4")),
    "top_p_token_offsets": b64(np.array([0, 1, 3, 4], dtype="<i4")),
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


# An engine whose client a fake sglang answers instead of the network.
def against(handle, **kwargs):
    engine = SGLangEngine("http://sglang/", retry_delay=0, **kwargs)
    engine.client = httpx2.AsyncClient(base_url="http://sglang/", transport=httpx2.MockTransport(handle))
    return engine


async def call(fake, retries=3, **params):
    return await against(fake.handle, retries=retries).generate([1, 2], {"stop_ids": frozenset({7, 3}), **params})


def server_info(max_req_input_len):
    reply = {"version": "0.5.21", "max_req_input_len": max_req_input_len}

    def handle(request):
        assert request.url.path == "/get_server_info"
        return httpx2.Response(200, json=reply)

    return handle


async def test_context_length_is_the_longest_input_the_scheduler_takes():
    # sglang reports 131066 for a 131072 model: its scheduler refuses an input of that length or more.
    assert await against(server_info(131066)).context_length() == 131066


class TestRouting:
    @pytest.fixture
    def served(self, monkeypatch):
        configs = {"moe": {"num_hidden_layers": 46, "num_experts_per_tok": 8}, "dense": {"num_hidden_layers": 28}}
        read = []

        def from_pretrained(path):
            read.append(path)
            return SimpleNamespace(get_text_config=lambda: SimpleNamespace(**configs[path]))

        monkeypatch.setattr(sglang.AutoConfig, "from_pretrained", from_pretrained)

        def engine(path):
            return against(lambda request: httpx2.Response(200, json={"model_path": path})), read

        return engine

    async def test_is_read_once_from_the_served_models_config(self, served):
        engine, read = served("moe")
        assert [await engine.routing(), await engine.routing()] == [(46, 8)] * 2 and read == ["moe"]

    async def test_a_model_without_experts_is_refused(self, served):
        with pytest.raises(InferenceEngineError, match="routes to no experts"):
            await served("dense")[0].routing()


async def test_context_length_is_retried():
    info = {"max_req_input_len": 32762}
    answers = [httpx2.ConnectError("refused"), httpx2.Response(503), httpx2.Response(200, json=info)]

    def handle(request):
        if isinstance(step := answers.pop(0), Exception):
            raise step
        return step

    assert await against(handle, retries=3).context_length() == 32762


async def test_ready_polls_until_sglang_generates():
    answers, paths = [httpx2.ConnectError("refused"), httpx2.Response(503), httpx2.Response(200)], []

    def handle(request):
        paths.append(request.url.path)
        if isinstance(step := answers.pop(0), Exception):
            raise step
        return step

    await against(handle).ready()
    assert paths == ["/health_generate"] * 3


async def test_sharded_transport_takes_requests_in_turn():
    served = []
    pools = [httpx2.MockTransport(lambda r, i=i: served.append(i) or httpx2.Response(200)) for i in range(3)]
    async with httpx2.AsyncClient(transport=ShardedTransport(pools)) as client:
        for _ in range(5):
            await client.get("http://sglang/")
    assert served == [0, 1, 2, 0, 1]


async def test_missing_context_length_is_an_error():
    with pytest.raises(InferenceEngineError, match="no context length"):
        await against(server_info(None)).context_length()


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

    async def test_weight_versions_come_back_as_runs(self):
        runs = [{"version": "3", "start": 0, "end": 1}, {"version": "4", "start": 1, "end": 3}]
        reply = {**REPLY, "meta_info": {**REPLY["meta_info"], "weight_versions": runs}}
        assert (await call(FakeSGLang(reply), max_tokens=3)).weight_versions == (("3", 0, 1), ("4", 1, 3))

    async def test_top_logprobs_come_back_k_per_position(self):
        fake = FakeSGLang({**REPLY, "meta_info": {**REPLY["meta_info"], **TOPK}})
        got = (await call(fake, max_tokens=3, top_logprobs=2)).top_logprobs
        assert fake.requests[0]["top_logprobs_num"] == 2
        assert got == TopLogprobs(np.arange(6, dtype="<i4").tobytes(), np.full(6, -1.0, dtype="<f4").tobytes(), 2)

    async def test_a_support_asks_sglang_for_logprobs_but_keeps_none_unasked(self):
        fake = FakeSGLang({**REPLY, "meta_info": {**REPLY["meta_info"], **KEPT}})
        got = await call(fake, max_tokens=3, top_p=0.9, return_support=True)
        assert fake.requests[0]["return_logprob"] is True and got.logprobs is None and got.support is not None

    async def test_both_come_back_when_both_are_asked(self):
        fake = FakeSGLang({**REPLY, "meta_info": {**REPLY["meta_info"], **TOPK, **KEPT}})
        got = await call(fake, max_tokens=3, top_p=0.9, return_support=True, top_logprobs=2)
        assert got.support is not None and got.top_logprobs is not None

    @pytest.mark.parametrize(
        ("asked", "reason"), [({"return_support": True}, "no top-p support"), ({"top_logprobs": 2}, "as text")]
    )
    async def test_asked_for_and_missing_is_refused(self, asked, reason):
        with pytest.raises(InferenceEngineError, match=reason):
            await call(FakeSGLang(), max_tokens=3, **asked)

    async def test_the_ids_top_p_kept_come_back_as_the_support(self):
        fake = FakeSGLang({**REPLY, "meta_info": {**REPLY["meta_info"], **KEPT}})
        got = (await call(fake, max_tokens=3, top_p=0.9, return_support=True)).support
        custom = fake.requests[0]["sampling_params"]["custom_params"]
        assert custom == {"return_top_p_token_ids": True, "return_top_p_log_probs": True}
        offsets = np.array([0, 1, 3, 4], dtype="<i4").tobytes()
        assert got == Support(np.arange(4, dtype="<i4").tobytes(), np.zeros(4, dtype="<f4").tobytes(), offsets)

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
        assert (await call(fake, retries=3, max_tokens=3)).finish_reason == "stop"
        assert len(fake.requests) == 4

    async def test_gives_up_after_the_retries(self):
        fake = FakeSGLang(script=(503,))
        with pytest.raises(InferenceEngineError, match="/generate failed 3 times, last: .*503"):
            await call(fake, retries=2, max_tokens=3)
        assert len(fake.requests) == 3

    async def test_no_retries_is_one_attempt(self):
        fake = FakeSGLang(script=(503,))
        with pytest.raises(InferenceEngineError, match="failed 1 times"):
            await call(fake, retries=0, max_tokens=3)
        assert len(fake.requests) == 1
