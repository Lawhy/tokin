from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Callable
from typing import Any, ClassVar

import httpx2

from ..rollout import FinishReason, Generation, GenerationParams
from .client import engine_client
from .inference_engine import InferenceEngine, InferenceEngineError

logger = logging.getLogger(__name__)


class SGLangEngine(InferenceEngine):
    """A running sglang server, spoken to over its native `/generate`."""

    param_map: ClassVar[dict[str, str]] = {
        "max_tokens": "max_new_tokens",
        "stop_ids": "stop_token_ids",
        "seed": "sampling_seed",
        "response_schema": "json_schema",
        "return_logprobs": "return_logprob",
        "routed_experts_start": "routed_experts_start_len",
        "request_id": "rid",
    }
    param_convert: ClassVar[dict[str, Callable[[Any], Any]]] = {"stop_ids": sorted, "response_schema": json.dumps}
    # These sit at the top level of the `/generate` body; every other key goes into `sampling_params`.
    top_level_keys: ClassVar[frozenset[str]] = frozenset(
        {"return_logprob", "return_routed_experts", "routed_experts_start_len", "rid"}
    )

    def __init__(self, base_url: str, *, retries: int = 60, retry_delay: float = 1.0) -> None:
        self.client = engine_client(base_url)
        self.retries = retries
        self.retry_delay = retry_delay

    async def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        """The JSON sglang answers at `path`, retried `retries` times through a transient failure."""
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self.client.request(method, path, json=payload)
                response.raise_for_status()
                return response.json()
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
        card = await self.request("GET", "/v1/models")
        try:
            # The model card has the length sglang resolved and checks against; `/server_info` only echoes the flag.
            return int(card["data"][0]["max_model_len"])
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise InferenceEngineError(f"sglang reports no context length: {card}") from e

    async def generate(self, token_ids: list[int], params: GenerationParams) -> Generation:
        payload: dict[str, Any] = {"input_ids": token_ids, "stream": False, "sampling_params": {}}
        for name, value in self.translate(params).items():
            target = payload if name in self.top_level_keys else payload["sampling_params"]
            target[name] = value
        out = await self.request("POST", "/generate", payload)
        meta = out["meta_info"]
        try:
            finish_reason = FinishReason(meta["finish_reason"]["type"])
        except ValueError as e:
            raise InferenceEngineError(f"sglang finished with {meta['finish_reason']}") from e
        return Generation(
            token_ids=out["output_ids"],
            finish_reason=finish_reason,
            logprobs=[lp for lp, _, _ in meta["output_token_logprobs"]] if params.get("return_logprobs") else None,
            routed_experts=base64.b64decode(meta["routed_experts"]) if "routed_experts" in meta else None,
        )
