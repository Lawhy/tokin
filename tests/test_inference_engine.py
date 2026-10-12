import json
from typing import ClassVar

import pytest

from tokin.inference_engines import InferenceEngine
from tokin.rollout import Generation


class Stub(InferenceEngine):
    param_map: ClassVar = {
        "max_tokens": "max_new_tokens",
        "stop_ids": "stop_token_ids",
        "response_schema": "json_schema",
    }
    param_convert: ClassVar = {"stop_ids": sorted, "response_schema": json.dumps}

    async def context_length(self):
        return None

    async def generate(self, token_ids, params):
        return Generation(token_ids=[], finish_reason="stop")


def test_translate_renames_and_converts_and_passes_the_rest():
    got = Stub().translate(
        {"max_tokens": 8, "stop_ids": frozenset({7, 3}), "response_schema": {"a": 1}, "top_n_sigma": 1.5}
    )
    assert got == {"max_new_tokens": 8, "stop_token_ids": [3, 7], "json_schema": '{"a": 1}', "top_n_sigma": 1.5}


def test_translate_refuses_two_keys_on_one_name():
    with pytest.raises(ValueError, match="max_new_tokens"):
        Stub().translate({"max_tokens": 8, "stop_ids": [], "max_new_tokens": 5})
