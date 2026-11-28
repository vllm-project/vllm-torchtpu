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
"""CPU tests for main's #2438 fixes ported into the model-owned GDN v3 copy.

`triangular_block_size` is a power of two dividing every chunk, the blocked
triangular solves reject a non-dividing block instead of truncating, and the
static (KDA) mixed tile rounds down to a power of two. The served Qwen3.8
geometry (48 value heads at TP8, 6 per rank) keeps B = 16.
"""

import jax.numpy as jnp
import pytest

from vllm_torchtpu.models.qwen4_exp.experimental.kernels.gdn_v3 import (
    compute_gdn,
    compute_kda,
    config,
    tiling,
)


def _cfg(n_v, chunk_size, attention_mode=config.AttentionMode.GDN):
    n_kq = max(1, n_v // 4)
    return config.GDNConfig(
        mode=config.GDNMode.PER_SEQ,
        dtypes=config.Dtypes(
            act_in=jnp.dtype(jnp.bfloat16),
            act_out=jnp.dtype(jnp.bfloat16),
            compute=jnp.dtype(jnp.float32),
            recurrent_state=jnp.dtype(jnp.float32),
            conv_state=jnp.dtype(jnp.bfloat16),
        ),
        batch_size=chunk_size,
        dim_size=2 * n_kq * 128 + n_v * 128,
        kernel_size=4,
        tile_size=chunk_size,
        num_kq_heads=n_kq,
        num_v_heads=n_v,
        kq_head_dim=128,
        v_head_dim=128,
        attention_mode=attention_mode,
    )


@pytest.mark.parametrize("n_v", [1, 2, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64, 128])
@pytest.mark.parametrize("attention_mode", list(config.AttentionMode))
def test_triangular_block_size_divides_every_chunk(n_v, attention_mode):
    for chunk_size in tiling.MIXED_TILE_SIZES:
        b = _cfg(n_v, chunk_size, attention_mode).triangular_block_size
        assert b >= 2 and b & (b - 1) == 0
        assert chunk_size % min(b, chunk_size) == 0


@pytest.mark.parametrize(
    "n_v,expected",
    # 6 = served Qwen3.8 per-rank value heads (48 at TP8).
    [(6, 16), (8, 16), (16, 8), (24, 4), (32, 4), (40, 2), (48, 2), (64, 2)],
)
def test_gdn_triangular_block_size(n_v, expected):
    assert _cfg(n_v, 128).triangular_block_size == expected


def test_invert_triangular_matrix_rejects_non_dividing_block():
    t = jnp.tril(jnp.ones((2, 128, 128), dtype=jnp.float32))
    with pytest.raises(AssertionError, match="divisible"):
        compute_gdn.invert_triangular_matrix(t, block_size=5)
    for block_size in (16, 8, 4, 2):
        inv = compute_gdn.invert_triangular_matrix(t, block_size=block_size)
        assert inv.shape == t.shape


def _kda_inputs(chunk):
    return dict(
        q_large=jnp.zeros((2, chunk, 128), dtype=jnp.float32),
        k_large=jnp.zeros((2, chunk, 128), dtype=jnp.float32),
        v_large=jnp.zeros((8, chunk, 128), dtype=jnp.float32),
        gating_log=jnp.zeros((8, chunk, 128), dtype=jnp.float32),
        beta=jnp.zeros((1, chunk, 8), dtype=jnp.float32),
        state_prev=jnp.zeros((8, 128, 128), dtype=jnp.float32),
    )


def test_chunked_kda_rejects_non_dividing_chunk():
    bad = _cfg(8, 24, config.AttentionMode.KDA)
    assert bad.triangular_block_size == 16
    with pytest.raises(AssertionError, match="divisible"):
        compute_kda.chunked_kda_per_seq(**_kda_inputs(24), cfg=bad)
    out, state = compute_kda.chunked_kda_per_seq(
        **_kda_inputs(32), cfg=_cfg(8, 32, config.AttentionMode.KDA)
    )
    assert out.shape == (8, 32, 128)
    assert state.shape == (8, 128, 128)


@pytest.mark.parametrize(
    "mixed,batch,expected",
    [(64, 24, 16), (64, 48, 32), (64, 64, 64), (64, 200, 64), (64, 1, 1), (3, 8, 2)],
)
def test_clip_mixed_tile_size(mixed, batch, expected):
    assert tiling.clip_mixed_tile_size(mixed, batch) == expected
