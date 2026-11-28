# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model directories register --kv-cache-dtype names for their architectures."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.layers.adapter.attention import (
    TPU_STR_DTYPE_TO_TORCH_DTYPE,
    _is_fp8_kv_cache_dtype,
)
from vllm_torchtpu.layers.core import model_kv_cache_dtypes
from vllm_torchtpu.platforms.tpu_block_size_utils import _resolve_tpu_cache_dtype
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

# A name vLLM accepts for --kv-cache-dtype that is not a built-in TPU name.
_NAME = "fp8_per_token_head"


@pytest.fixture(autouse=True)
def kv_dtypes(monkeypatch):
    registered: dict[str, torch.dtype] = {}
    monkeypatch.setattr(model_kv_cache_dtypes, "MODEL_KV_CACHE_DTYPES", registered)
    monkeypatch.setattr(model_kv_cache_dtypes, "_ARCHITECTURES", {})
    monkeypatch.setattr(
        model_kv_cache_dtypes.TPU_KV_CACHE_DTYPES,
        "maps",
        [model_kv_cache_dtypes._BUILTIN_KV_CACHE_DTYPES, registered],
    )


def test_model_kv_cache_dtype_admitted_for_its_architectures_only() -> None:
    model_kv_cache_dtypes.register_model_kv_cache_dtype(
        _NAME, torch.float8_e4m3fn, "ArchA"
    )
    ok = TpuPlatform.is_kv_cache_dtype_supported
    assert ok(_NAME, SimpleNamespace(architectures=["ArchA"]))
    assert not ok(_NAME, SimpleNamespace(architectures=["ArchB"]))
    assert ok("fp8", SimpleNamespace(architectures=["ArchB"]))


def test_registered_name_resolves_everywhere() -> None:
    model_kv_cache_dtypes.register_model_kv_cache_dtype(
        _NAME, torch.float8_e4m3fn, "ArchA"
    )
    # Shared attention layers, worker and runner.
    assert TPU_STR_DTYPE_TO_TORCH_DTYPE[_NAME] == torch.float8_e4m3fn
    # Attention quantizes K/V into a registered fp8-backed cache.
    assert _is_fp8_kv_cache_dtype(_NAME)
    assert not _is_fp8_kv_cache_dtype("bfloat16")
    # Unified-pool block sizing.
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype=_NAME),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )
    assert _resolve_tpu_cache_dtype(vllm_config) == torch.float8_e4m3fn


def test_model_kv_cache_dtype_conflict_is_rejected() -> None:
    model_kv_cache_dtypes.register_model_kv_cache_dtype(
        _NAME, torch.float8_e4m3fn, "ArchA"
    )
    with pytest.raises(ValueError, match="already registered"):
        model_kv_cache_dtypes.register_model_kv_cache_dtype(
            _NAME, torch.bfloat16, "ArchA"
        )


@pytest.mark.parametrize(
    ("name", "match"),
    [
        ("int8", "built-in"),
        ("FP8_PER_TOKEN_HEAD", "lowercase"),
        ("not_a_vllm_dtype", "vLLM does not accept"),
    ],
)
def test_invalid_names_are_rejected(name: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        model_kv_cache_dtypes.register_model_kv_cache_dtype(
            name, torch.bfloat16, "ArchA"
        )
    assert model_kv_cache_dtypes.MODEL_KV_CACHE_DTYPES == {}
