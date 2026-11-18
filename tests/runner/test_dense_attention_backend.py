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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackend,
    PallasMLAttentionBackend,
)
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform
from vllm_torchtpu.runner import kv_cache_manager
from vllm_torchtpu.runner.kv_cache_manager import KVCacheManager

BLOCK_SIZE = 128
DTYPE = torch.bfloat16


class _Mamba(MambaBase):
    def get_kv_cache_spec(self, vllm_config):
        return MambaSpec(
            block_size=16,
            shapes=[(4, 128)],
            dtypes=[torch.bfloat16],
        )

    def get_state_dtype(self):
        return (torch.bfloat16,)

    def get_state_shape(self):
        return ((4, 128),)

    @property
    def mamba_type(self):
        return "dummy"


def _patch_backends(*backends):
    return (
        patch.object(TpuPlatform, "_find_non_ssm_backend", return_value=backends[0]),
        patch.object(
            TpuPlatform, "_find_non_ssm_backends", return_value=list(backends)
        ),
    )


@pytest.fixture
def mla_then_dense():
    """The target declares the MLA backend; a drafter adds plain Attention."""
    first, all_ = _patch_backends(PallasMLAttentionBackend, PallasAttentionBackend)
    with first, all_:
        yield


def _mla_layer():
    layer = MagicMock(spec=MLAAttention)
    layer.num_kv_heads = 1
    layer.head_size = 576
    layer.use_sparse = False
    return layer


def _attn_layer(num_kv_heads=8, head_size=128):
    layer = MagicMock(spec=Attention)
    layer.num_kv_heads = num_kv_heads
    layer.head_size = head_size
    return layer


def _runner_stub():
    return SimpleNamespace(
        block_size=BLOCK_SIZE,
        kv_cache_dtype=DTYPE,
        cache_config=SimpleNamespace(mamba_page_size_padded=None),
        _hybrid_uniform_page_size_bytes=None,
        _mamba_num_blocks=None,
        vllm_config=MagicMock(),
    )


def test_dense_attention_backend_skips_mla_family(mla_then_dense) -> None:
    assert kv_cache_manager.dense_attention_backend(None) is PallasAttentionBackend


def test_dense_attention_backend_keeps_first_for_pure_mla() -> None:
    first, all_ = _patch_backends(PallasMLAttentionBackend)
    with first, all_:
        assert (
            kv_cache_manager.dense_attention_backend(None) is PallasMLAttentionBackend
        )


def test_dense_attention_backend_keeps_dense_first_backend() -> None:
    first, all_ = _patch_backends(PallasAttentionBackend, PallasMLAttentionBackend)
    with first, all_:
        assert kv_cache_manager.dense_attention_backend(None) is PallasAttentionBackend


def test_attention_page_padding_sizes_plain_attention_with_dense_backend(
    mla_then_dense,
) -> None:
    stub = _runner_stub()
    layers = {"target.mla": _mla_layer(), "draft.attn": _attn_layer()}

    KVCacheManager(stub)._update_attention_page_size_padded(layers, BLOCK_SIZE)

    mla_page = PallasMLAttentionBackend.get_kv_cache_page_size_bytes(
        BLOCK_SIZE, 1, 576, DTYPE
    )
    dense_page = PallasAttentionBackend.get_kv_cache_page_size_bytes(
        BLOCK_SIZE, 8, 128, DTYPE
    )
    mla_formula_page = PallasMLAttentionBackend.get_kv_cache_page_size_bytes(
        BLOCK_SIZE, 8, 128, DTYPE
    )
    assert dense_page != mla_formula_page
    assert stub._hybrid_uniform_page_size_bytes == max(mla_page, dense_page)


def test_mamba_page_padding_sizes_first_mla_layer_with_mla_backend(
    mla_then_dense,
) -> None:
    """MLA target + Mamba + plain drafter Attention: the first (MLA) layer's
    page must keep the MLA geometry, not the dense formula."""
    stub = _runner_stub()
    layers = {
        "target.mla": _mla_layer(),
        "draft.attn": _attn_layer(),
        **{f"mamba_{i}": _Mamba() for i in range(3)},
    }
    manager = KVCacheManager(stub)

    with (
        patch.object(manager, "_maybe_select_compact_mamba_layout") as compact,
        patch.object(manager, "_maybe_set_num_blocks_override"),
    ):
        manager._update_mamba_page_size_padded(layers, BLOCK_SIZE)

    mla_page = PallasMLAttentionBackend.get_kv_cache_page_size_bytes(
        BLOCK_SIZE, 1, 576, DTYPE
    )
    assert mla_page != PallasAttentionBackend.get_kv_cache_page_size_bytes(
        BLOCK_SIZE, 1, 576, DTYPE
    )
    assert compact.call_args.args[0] == mla_page
    # 2 attention + 3 mamba layers: group_size 2, so 1 attention group and
    # 2 mamba groups of 4 * 128 bf16 = 1024 bytes each.
    assert stub._hybrid_uniform_page_size_bytes == mla_page + 2 * 1024
