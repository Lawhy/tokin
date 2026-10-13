from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Callable
from typing import Any, ClassVar

import httpx2

from ..rollout import FinishReason, Generation
from .inference_engine import GenerationParams, InferenceEngine, InferenceEngineError


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

    def __init__(self, url: str, client: httpx2.AsyncClient, *, retries: int = 60, retry_delay: float = 1.0) -> None:
        self.url = url.rstrip("/")
        self.client = client
        self.retries = retries
        self.retry_delay = retry_delay

    async def context_length(self) -> int:
        response = await self.client.get(f"{self.url}/v1/models")
        response.raise_for_status()
        # The model card has the length sglang resolved and checks requests against; `/server_info` only echoes the flag.
        if (length := response.json()["data"][0].get("max_model_len")) is None:
            raise InferenceEngineError("sglang reports no context length")
        return int(length)

    async def post(self, payload: dict[str, Any]) -> Any:
        """The JSON `/generate` answers, after however many retries it takes."""
        for attempt in range(1, self.retries + 1):
            try:
                response = await self.client.post(f"{self.url}/generate", json=payload)
                response.raise_for_status()
                return response.json()
            except (httpx2.HTTPError, ValueError) as e:
                if attempt == self.retries:
                    raise InferenceEngineError(f"sglang failed {attempt} times, last: {e}") from e
                await asyncio.sleep(self.retry_delay)

    async def generate(self, token_ids: list[int], params: GenerationParams) -> Generation:
        payload: dict[str, Any] = {"input_ids": token_ids, "stream": False, "sampling_params": {}}
        for name, value in self.translate(params).items():
            target = payload if name in self.top_level_keys else payload["sampling_params"]
            target[name] = value
        out = await self.post(payload)
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
