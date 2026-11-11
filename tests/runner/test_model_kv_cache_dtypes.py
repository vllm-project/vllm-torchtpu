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

from vllm_torchtpu.layers.adapter.attention import TPU_STR_DTYPE_TO_TORCH_DTYPE
from vllm_torchtpu.layers.core import model_kv_cache_dtypes
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform


@pytest.fixture
def kv_dtypes(monkeypatch):
    monkeypatch.setitem(model_kv_cache_dtypes._ARCHITECTURES, "_test_int4", set())
    yield
    model_kv_cache_dtypes.MODEL_KV_CACHE_DTYPES.pop("_test_int4", None)


def test_model_kv_cache_dtype_admitted_for_its_architectures_only(kv_dtypes) -> None:
    model_kv_cache_dtypes.register_model_kv_cache_dtype(
        "_test_int4", torch.float8_e4m3fn, "ArchA"
    )
    ok = TpuPlatform.is_kv_cache_dtype_supported
    assert ok("_test_int4", SimpleNamespace(architectures=["ArchA"]))
    assert not ok("_test_int4", SimpleNamespace(architectures=["ArchB"]))
    assert ok("fp8", SimpleNamespace(architectures=["ArchB"]))
    # Shared attention layers resolve the registered name.
    assert TPU_STR_DTYPE_TO_TORCH_DTYPE["_test_int4"] == torch.float8_e4m3fn
    assert TPU_STR_DTYPE_TO_TORCH_DTYPE.get("_test_int4") == torch.float8_e4m3fn


def test_model_kv_cache_dtype_conflict_is_rejected(kv_dtypes) -> None:
    model_kv_cache_dtypes.register_model_kv_cache_dtype(
        "_test_int4", torch.float8_e4m3fn, "ArchA"
    )
    with pytest.raises(ValueError, match="already registered"):
        model_kv_cache_dtypes.register_model_kv_cache_dtype(
            "_test_int4", torch.bfloat16, "ArchA"
        )
