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
"""MiniMax M3 DSpark draft KV pages at long context, on CPU.

The target's owned caches report `PallasMLAttentionBackend`, which is the
first backend in layer order. Taking it for the drafter's plain attention
layers allocated each draft layer as MLA rows, `(num_blocks, 32, 4, 128)` at
fp8, which the RPA adapter reads as 32-token pages under 128-token block
tables. The kernel then needs `cdiv(kv_len, 32)` table entries per request:
beyond a quarter of max_model_len (65536 tokens) it reads past the request's
row, and for the last row past the table, so the page ids it DMAs are
garbage (SparseCore halt in the draft forward on TPU). The drafter's pages
must use the dense RPA geometry, one 128-token page per block.
"""

import math
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.math_utils import cdiv

from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import (
    dynamic_validate_inputs,
)
from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackend,
    PallasMLAttentionBackend,
)
from vllm_torchtpu.models.minimax_m3.attention import MiniMaxM3Cache
from vllm_torchtpu.runner.kv_cache_manager import dense_attention_backend
from vllm_torchtpu.spec_decode.dflash import DFlashProposer

MAX_MODEL_LEN = 262144
BLOCK_SIZE = 128  # M3 sparse block = scheduler block
HEAD_DIM = 128
KV_HEADS_PER_RANK = 1  # 4 draft KV heads over TP4
NUM_Q_HEADS = 16  # 64 over TP4
K = 7  # dspark_block_size: dense layout, 7 query tokens
WINDOW = 4096
DTYPE = torch.float8_e4m3fn


class _DraftAttention(AttentionLayerBase):
    """Stands in for a drafter `Attention` layer: only its backend matters."""

    def get_attn_backend(self):
        return PallasAttentionBackend

    def get_kv_cache_spec(self, vllm_config):
        raise NotImplementedError


def _deployment() -> SimpleNamespace:
    """Target owned caches first, then the drafter layers, as registered."""
    owned = MiniMaxM3Cache.__new__(MiniMaxM3Cache)
    context = {"model.layers.0.self_attn.kv_cache": owned}
    for i in range(5):
        context[f"draft_model.model.layers.{i}.self_attn.attn"] = _DraftAttention()
    return SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=context)
    )


def _kernel_page_size(pool_shape: tuple[int, ...]) -> int:
    """Tokens per RPA page, as `_pallas_rpa_kernel_impl` views the pool:
    a 4-D (MLA-row) pool becomes native pages of `pool_shape[1]` tokens."""
    return pool_shape[1]


def test_owned_mla_target_does_not_set_the_draft_page_geometry():
    vllm_config = _deployment()
    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

    # The first backend is the target's: this is what used to size the drafter.
    assert TpuPlatform._find_non_ssm_backend(vllm_config) is PallasMLAttentionBackend
    assert dense_attention_backend(vllm_config) is PallasAttentionBackend


def test_dense_backend_is_unchanged_without_a_dense_layer():
    owned = MiniMaxM3Cache.__new__(MiniMaxM3Cache)
    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={"model.layers.0.self_attn.kv_cache": owned}
        )
    )
    assert dense_attention_backend(vllm_config) is PallasMLAttentionBackend


@pytest.mark.parametrize(
    "backend,covers",
    [(PallasMLAttentionBackend, False), (PallasAttentionBackend, True)],
)
def test_draft_pages_cover_max_model_len(backend, covers):
    pool = backend.get_kv_cache_shape(1, BLOCK_SIZE, KV_HEADS_PER_RANK, HEAD_DIM, DTYPE)
    page_size = _kernel_page_size(tuple(pool))
    # One block-table entry per 128-token block (the runner's table width).
    pages_per_seq = cdiv(MAX_MODEL_LEN, BLOCK_SIZE)
    needed = cdiv(MAX_MODEL_LEN, page_size)
    assert (needed <= pages_per_seq) is covers
    if not covers:
        # MLA rows: 32-token pages; the table runs out at a quarter of
        # max_model_len.
        assert page_size == 32
        assert pages_per_seq * page_size == MAX_MODEL_LEN // 4


def _draft_block(base_pos: list[int]):
    """The draft forward's (positions, seq_lens) for anchors `base_pos`."""
    stub = SimpleNamespace(_max_model_len=MAX_MODEL_LEN)
    n = len(base_pos)
    base_tokens = torch.zeros(n, dtype=torch.int32)
    base = torch.tensor(base_pos, dtype=torch.int32).unsqueeze(1)
    _, positions, seq_lens = DFlashProposer._layout_draft_block(
        stub, base_tokens, base, K, n * K, 200058
    )
    return positions, seq_lens


def _validate(page_size: int, seq_lens: torch.Tensor, num_blocks: int = 64):
    """Run the RPA kernel's runtime index checks on the draft forward's
    metadata: one table row of `cdiv(max_model_len, 128)` entries per request,
    every entry a live block."""
    num_reqs = seq_lens.shape[0]
    pages_per_seq = cdiv(MAX_MODEL_LEN, BLOCK_SIZE)
    total_pages = num_blocks * (BLOCK_SIZE // page_size)
    rng = np.random.default_rng(0)
    page_indices = rng.integers(1, num_blocks, num_reqs * pages_per_seq)
    tokens = num_reqs * K
    q = jnp.zeros((tokens, NUM_Q_HEADS, HEAD_DIM), jnp.bfloat16)
    # K/V arrive quantized to the cache dtype.
    kv = jnp.zeros((tokens, KV_HEADS_PER_RANK, HEAD_DIM), jnp.float8_e4m3fn)
    kv_cache = jnp.zeros((total_pages, page_size, 1, 4, HEAD_DIM), jnp.float8_e4m3fn)
    dynamic_validate_inputs(
        q,
        kv,
        kv,
        kv_cache,
        jnp.asarray(seq_lens.numpy(), jnp.int32),
        jnp.asarray(page_indices, jnp.int32),
        jnp.arange(num_reqs + 1, dtype=jnp.int32) * K,
        jnp.asarray([0, 0, num_reqs], jnp.int32),
        use_causal_mask=False,
        sliding_window=WINDOW,
    )


# Anchors of a long AgentX context, one past a quarter of max_model_len and
# one at the clamp.
LONG_ANCHORS = [1000, 70000, 200000, MAX_MODEL_LEN - 3]


def test_draft_block_stays_inside_max_model_len():
    positions, seq_lens = _draft_block(LONG_ANCHORS)
    assert int(positions.max()) <= MAX_MODEL_LEN - 1
    assert int(seq_lens.max()) == MAX_MODEL_LEN
    assert seq_lens.tolist()[:3] == [a + K for a in LONG_ANCHORS[:3]]


def test_long_context_draft_forward_indexes_inside_the_table():
    _, seq_lens = _draft_block(LONG_ANCHORS)
    dense_pool = PallasAttentionBackend.get_kv_cache_shape(
        1, BLOCK_SIZE, KV_HEADS_PER_RANK, HEAD_DIM, DTYPE
    )
    assert math.prod(dense_pool) == BLOCK_SIZE * 4 * HEAD_DIM
    _validate(_kernel_page_size(tuple(dense_pool)), seq_lens)


def test_mla_row_geometry_overruns_the_table_past_a_quarter_context():
    mla_pool = PallasMLAttentionBackend.get_kv_cache_shape(
        1, BLOCK_SIZE, KV_HEADS_PER_RANK, HEAD_DIM, DTYPE
    )
    page_size = _kernel_page_size(tuple(mla_pool))
    _, short = _draft_block([1000, 60000])
    _validate(page_size, short)  # short contexts (GSM8K) stay in bounds
    _, long = _draft_block(LONG_ANCHORS)
    with pytest.raises(ValueError, match="page_cnt"):
        _validate(page_size, long)
