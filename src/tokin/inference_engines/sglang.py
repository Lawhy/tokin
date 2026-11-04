from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Callable
from typing import Any, ClassVar, cast

import httpx2
from transformers import AutoConfig

from ..rollout import FinishReason, Generation, GenerationParams, GenerationUsage, Support, TopLogprobs
from .client import engine_client
from .engine import InferenceEngine, InferenceEngineError

logger = logging.getLogger(__name__)


class SGLangEngine(InferenceEngine):
    """A running sglang server, spoken to over its native `/generate`."""

    # In the order of `GenerationParams`.
    param_map: ClassVar[dict[str, str]] = {
        "request_id": "rid",
        "max_tokens": "max_new_tokens",
        "stop_ids": "stop_token_ids",
        "seed": "sampling_seed",
        "response_schema": "json_schema",
        "return_logprobs": "return_logprob",
        "top_logprobs": "top_logprobs_num",
        "return_support": "custom_params",
        "routed_experts_start": "routed_experts_start_len",
    }
    param_convert: ClassVar[dict[str, Callable[[Any], Any]]] = {
        "stop_ids": sorted,
        "response_schema": json.dumps,
        # The ids top-p kept come from the sampler patch slime's images carry, asked for through `custom_params`.
        "return_support": lambda on: {"return_top_p_token_ids": on, "return_top_p_log_probs": on},
    }
    # These sit at the top level of the `/generate` body; every other key goes into `sampling_params`.
    top_level_keys: ClassVar[frozenset[str]] = frozenset(
        {"rid", "return_logprob", "top_logprobs_num", "return_routed_experts", "routed_experts_start_len"}
    )

    def __init__(self, base_url: str, *, retries: int = 60, retry_delay: float = 1.0) -> None:
        self.client = engine_client(base_url)
        self.retries = retries
        self.retry_delay = retry_delay
        self._routing: tuple[int, int] | None = None

    async def request(
        self, method: str, path: str, payload: dict[str, Any] | None = None, headers: dict[str, str] | None = None
    ) -> Any:
        """The JSON sglang answers at `path`, retried `retries` times through a transient failure."""
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self.client.request(method, path, json=payload, headers=headers)
                response.raise_for_status()
                # Off the event loop: a generation's support can run to megabytes.
                return await asyncio.to_thread(response.json)
            except (httpx2.HTTPError, ValueError) as e:
                if attempt > self.retries:
                    raise InferenceEngineError(f"sglang {path} failed {attempt} times, last: {e!r}") from e
                await asyncio.sleep(self.retry_delay)

    async def ready(self) -> None:
        """Poll `/health_generate`, which generates one token when sglang is idle, until it answers 200."""
        logger.info("waiting for sglang at %s to generate", self.client.base_url)
        while True:
            try:
                # Bounded per attempt, so one hung connection does not stall the polling.
                if (await self.client.get("/health_generate", timeout=10)).status_code == 200:
                    return
            except httpx2.HTTPError:
                pass
            await asyncio.sleep(self.retry_delay)

    async def context_length(self) -> int:
        infos = [await self.request("GET", "/get_server_info")]
        if infos[0].get("router_manager"):
            # sglang's router answers about itself; a request may land on any of its workers, so the least of theirs.
            workers = (await self.request("GET", "/workers")).get("workers") or []
            infos = await asyncio.gather(*(self.request("GET", f"{w['url']}/get_server_info") for w in workers))
            if not infos:
                raise InferenceEngineError("sglang's router lists no workers")
        try:
            # The scheduler refuses an input of `max_req_input_len` ids or more, 6 below `max_model_len` or further
            # with a small KV pool; a total within it also stays within the scheduler's cap on a generation.
            return min(int(info["max_req_input_len"]) for info in infos)
        except (KeyError, TypeError, ValueError) as e:
            raise InferenceEngineError(f"sglang reports no context length: {infos}") from e

    async def routing(self) -> tuple[int, int]:
        """Read once, from the config of the model sglang serves."""
        if self._routing is None:
            path = (await self.request("GET", "/get_model_info"))["model_path"]
            config = (await asyncio.to_thread(AutoConfig.from_pretrained, path)).get_text_config()
            if not hasattr(config, "num_experts_per_tok"):
                raise InferenceEngineError(f"sglang serves {path}, which routes to no experts")
            # sglang sizes its routing row from these two, for every hidden layer, dense ones included.
            self._routing = (config.num_hidden_layers, config.num_experts_per_tok)
        return self._routing

    async def generate(self, token_ids: list[int], params: GenerationParams) -> Generation:
        payload: dict[str, Any] = {"input_ids": token_ids, "stream": False, "sampling_params": {}}
        sent = cast(GenerationParams, {k: v for k, v in params.items() if k != "session_id"})
        for name, value in self.translate(sent).items():
            target = payload if name in self.top_level_keys else payload["sampling_params"]
            target[name] = value
        # sglang's router keeps a session on one engine by this header, so its prefix stays cached there.
        headers = {"X-SMG-Routing-Key": params["session_id"]} if "session_id" in params else None
        # sglang computes a support and top logprobs only alongside the sampled logprobs, kept or not.
        if params.get("return_support") or params.get("top_logprobs"):
            payload["return_logprob"] = True
        out = await self.request("POST", "/generate", payload, headers)
        return await asyncio.to_thread(self.read, out, params)

    def read(self, out: dict[str, Any], params: GenerationParams) -> Generation:
        """The generation in a `/generate` answer; the support and top logprobs arrive in base64."""
        meta = out["meta_info"]
        try:
            finish_reason = FinishReason(meta["finish_reason"]["type"])
        except ValueError as e:
            raise InferenceEngineError(f"sglang finished with {meta['finish_reason']}") from e
        sampled = out["output_ids"]
        # Asked for and missing: an sglang without slime's top-p patch, or no cut to report; never a fallback.
        if params.get("return_support") and sampled and "top_p_token_ids" not in meta:
            raise InferenceEngineError(
                "sglang returned no top-p support; it needs slime's sampler patch and `top_p` below 1"
            )
        if params.get("top_logprobs") and sampled and "output_topk_token_ids" not in meta:
            raise InferenceEngineError("sglang returned its top logprobs as text, not the binary `output_topk_*`")
        support = top_logprobs = None
        if "top_p_token_ids" in meta:
            support = Support(
                *(base64.b64decode(meta[key]) for key in ("top_p_token_ids", "top_p_log_probs", "top_p_token_offsets"))
            )
        if "output_topk_token_ids" in meta:
            top_logprobs = TopLogprobs(
                base64.b64decode(meta["output_topk_token_ids"]),
                base64.b64decode(meta["output_topk_log_probs"]),
                meta["output_topk_shape"][1],
            )
        return Generation(
            token_ids=sampled,
            finish_reason=finish_reason,
            logprobs=[lp for lp, _, _ in meta["output_token_logprobs"]] if params.get("return_logprobs") else None,
            routed_experts=base64.b64decode(meta["routed_experts"]) if "routed_experts" in meta else None,
            support=support,
            top_logprobs=top_logprobs,
            weight_versions=tuple((str(r["version"]), r["start"], r["end"]) for r in meta.get("weight_versions", ())),
            # sglang reports the speculative counts only when it decodes speculatively.
            usage=GenerationUsage(
                prompt_tokens=meta["prompt_tokens"],
                cached_tokens=meta["cached_tokens"],
                draft_tokens=meta.get("spec_num_proposed_drafts", 0),
                accepted_tokens=meta.get("spec_num_correct_drafts", 0),
                verify_steps=meta.get("spec_verify_ct", 0),
            ),
        )
