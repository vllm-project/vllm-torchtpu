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
from unittest.mock import patch

import pytest
import torch
from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_compressor import (
    VllmCompressorStateCache,
    VllmDeepseekCompressor,
)


def _state_cache(compress_ratio: int, head_dim: int = 512):
    """A state cache with the attributes its geometry depends on.

    The base `__init__` registers into the vLLM forward context, which needs a
    live config; the geometry under test does not. `state_dim` follows vLLM's
    own `2 * coff * head_dim` (kv_state + score_state).
    """
    cache = VllmCompressorStateCache.__new__(VllmCompressorStateCache)
    coff = 1 + (compress_ratio == 4)
    cache.prefix = "model.layers.0.self_attn.compressor.state_cache"
    cache.compress_ratio = compress_ratio
    cache.head_dim = head_dim
    cache.state_dim = 2 * coff * head_dim
    cache.sliding_window = coff * compress_ratio
    cache.block_size = 0
    return cache


@pytest.mark.parametrize(
    "compress_ratio, head_dim, cache_block_size, expected",
    [
        # CSA: its state overlays its own compressed-KV page.
        (4, 512, 1024, 16),
        (4, 512, 512, 8),
        # HCA state rows are hosted on a *CSA* page, not on HCA's own (which
        # holds two token states), so the page is far larger than HCA's
        # compression ratio alone would suggest.
        (128, 512, 1024, 32),
        (128, 512, 512, 16),
        # The lightning indexer: 256-lane records. Its own page holds 32
        # token states, but it is floored to CSA's 16 so the two share a
        # `(block_size, sliding_window)` cache group.
        (4, 128, 1024, 16),
    ],
)
def test_state_block_size_follows_the_host_page(
    compress_ratio, head_dim, cache_block_size, expected
):
    """Block size must come from the kernel's own layout model.

    It is not a closed form -- HCA writes two rows per record, the indexer
    array is 256 lanes wide, HCA's state is hosted on a CSA page, and CSA and
    the indexer are floored to a shared value -- so this pins the values
    `compress_and_store.config` actually produces, which is what the
    compressor kernel indexes the block table with.
    """
    cache = _state_cache(compress_ratio, head_dim)
    assert cache._derive_block_size(cache_block_size) == expected


def test_state_block_size_refuses_a_page_holding_no_row():
    """A page too small to hold one compressed row must not be emitted."""
    cache = _state_cache(128)
    with pytest.raises(ValueError, match="cannot be paged"):
        cache._derive_block_size(64)


def test_state_cache_spec_is_raw_uint8_at_the_host_page_size():
    """The spec must survive `kv_cache_spec_normalizer` unretyped.

    Any dtype but uint8 is rewritten to the model's KV dtype, which truncates
    the raw f32 state; the block size must come from the final cache config,
    not the base class's CUDA constant.
    """
    cache = _state_cache(4)
    assert cache.block_size == 0, "fixture starts unset"
    spec = cache.get_kv_cache_spec(
        SimpleNamespace(cache_config=SimpleNamespace(block_size=1024))
    )

    assert spec.dtype == torch.uint8
    assert spec.block_size == 16
    # `_build_compressor_op` bakes this in, so the object must be updated too.
    assert cache.block_size == 16
    # 2048 f32 words = 8192 B, already a multiple of 128.
    assert spec.head_size == 8192
    assert spec.num_kv_heads == 1
    assert spec.sliding_window == 8
    # vLLM's DSv4 grouping is isinstance-based, so the base type is load-bearing.
    assert isinstance(spec, SlidingWindowMLASpec)


def test_compressor_op_is_built_lazily_and_cached():
    """Building in `__init__` bakes the pre-KV-init `state_block_size`.

    That constant is the base class's CUDA value until `get_kv_cache_spec`
    replaces it, so an eagerly built op traces against the wrong page geometry.
    """
    assert isinstance(VllmDeepseekCompressor.__dict__["compressor_op"], property), (
        "compressor_op must stay a property, not an __init__ field"
    )

    compressor = VllmDeepseekCompressor.__new__(VllmDeepseekCompressor)
    sentinel = object()
    with patch.object(
        VllmDeepseekCompressor, "_build_compressor_op", return_value=sentinel
    ) as build:
        assert compressor.compressor_op is sentinel
        assert compressor.compressor_op is sentinel
    build.assert_called_once()
