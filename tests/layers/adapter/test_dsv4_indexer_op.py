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

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_indexer import \
    VllmDeepseekV4IndexerCache


@pytest.mark.parametrize("block_size, expected_ratio", [(1024, 4), (2, 2)])
def test_indexer_cache_spec_is_raw_uint8(block_size, expected_ratio):
    """The indexer cache is a packed byte record, not KV-dtype features.

    Any dtype but uint8 is retyped by `kv_cache_spec_normalizer`; the compress
    ratio is clamped so it can never exceed the page it is packed into.
    """
    cache = VllmDeepseekV4IndexerCache.__new__(VllmDeepseekV4IndexerCache)
    cache.cache_config = SimpleNamespace(block_size=block_size)
    cache.compress_ratio = 4

    spec = cache.get_kv_cache_spec(SimpleNamespace())

    assert isinstance(spec, MLAAttentionSpec)
    assert spec.dtype == torch.uint8
    assert spec.block_size == block_size
    assert spec.num_kv_heads == 1
    # 128 record bytes + 1 scale byte, rounded to the 128-byte lane.
    assert spec.head_size == 256
    assert spec.tokens_per_state == expected_ratio
