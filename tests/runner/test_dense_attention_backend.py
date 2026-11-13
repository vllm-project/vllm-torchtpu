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
"""Plain attention caches take a dense backend next to model-owned MLA caches."""

from unittest.mock import patch

from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackend,
    PallasMLAttentionBackend,
)
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform
from vllm_torchtpu.runner import kv_cache_manager


def test_dense_attention_backend_skips_mla_family() -> None:
    with (
        patch.object(
            TpuPlatform, "_find_non_ssm_backend", return_value=PallasMLAttentionBackend
        ),
        patch.object(
            TpuPlatform,
            "_find_non_ssm_backends",
            return_value=[PallasMLAttentionBackend, PallasAttentionBackend],
        ),
    ):
        assert kv_cache_manager.dense_attention_backend(None) is PallasAttentionBackend
    with (
        patch.object(
            TpuPlatform, "_find_non_ssm_backend", return_value=PallasMLAttentionBackend
        ),
        patch.object(
            TpuPlatform,
            "_find_non_ssm_backends",
            return_value=[PallasMLAttentionBackend],
        ),
    ):
        # Pure MLA keeps the first backend.
        assert (
            kv_cache_manager.dense_attention_backend(None) is PallasMLAttentionBackend
        )
