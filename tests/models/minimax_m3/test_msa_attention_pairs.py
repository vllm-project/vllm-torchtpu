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
"""Page-major MSA attention (interpret mode) against the tile kernel and a
dense reference."""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.models.minimax_m3.test_msa_attention_lse import (
    HEAD_DIM,
    PAGES,
    Q_PER_KV,
    SCALE,
    _fp4_cache,
    _reference,
)
from vllm_torchtpu.models.minimax_m3.kernels import (
    msa_attention_lse,
    msa_attention_pairs,
)
from vllm_torchtpu.models.minimax_m3.kernels.msa_attention_pairs import (
    msa_sparse_attention_pairs,
)

TOPK = 6


def _inputs(rng, num_tokens, num_real, crowd):
    """Selections of `num_real` tokens; with `crowd`, every token picks from
    the same few pages (long runs of pairs per page, cut into blocks)."""
    q = jnp.asarray(rng.standard_normal((num_tokens, Q_PER_KV, HEAD_DIM))).astype(
        jnp.bfloat16
    )
    table = rng.permutation(PAGES).astype(np.int32)
    keys = np.full((num_tokens, TOPK), -1, np.int32)
    local_key = np.full(num_tokens, -1, np.int32)
    local_limit = np.zeros(num_tokens, np.int32)
    pool = 4 if crowd else 10  # keys 10, 11: nobody selects
    for t in range(num_real):
        chosen = rng.choice(pool, rng.integers(0, min(TOPK, pool) + 1), replace=False)
        keys[t, : len(chosen)] = chosen
        if len(chosen):
            local_key[t] = chosen[0]
            local_limit[t] = rng.integers(1, 129)
    if num_real > 1:
        keys[1, -1] = keys[1, 0]  # a page listed twice counts once
    return q, table, keys, local_key, local_limit


@pytest.mark.parametrize("num_tokens,num_real,window", [(70, 67, 32), (40, 40, 1024)])
@pytest.mark.parametrize("crowd", [False, True])
def test_pairs_match_reference_and_tiles(num_tokens, num_real, window, crowd):
    rng = np.random.default_rng(num_tokens + crowd)
    q, table, keys, local_key, local_limit = _inputs(rng, num_tokens, num_real, crowd)
    kv, scales, rows = _fp4_cache(rng, 1)
    scales[table[10:]] = np.nan  # scale pages nobody selects
    args = (
        q,
        kv,
        jnp.asarray(keys),
        jnp.asarray(local_key),
        jnp.asarray(local_limit),
        jnp.asarray(table),
    )
    got = msa_sparse_attention_pairs(
        *args,
        sm_scale=SCALE,
        scale_cache=jnp.asarray(scales),
        window=window,
        interpret=True,
    )
    got = np.asarray(got.astype(jnp.float32))
    assert got.shape == (Q_PER_KV, num_tokens, HEAD_DIM)
    assert np.isfinite(got).all()
    dedup = keys.copy()
    dedup[1, -1] = -1
    want, _ = _reference(
        np.asarray(q.astype(jnp.float32)), rows, dedup, local_key, local_limit, table, 0
    )
    np.testing.assert_allclose(got, want.transpose(1, 0, 2), atol=2e-2)
    tiles, _ = msa_attention_lse.msa_sparse_attention_lse(
        *args[:2],
        jnp.asarray(dedup),
        *args[3:],
        sm_scale=SCALE,
        scale_cache=jnp.asarray(scales),
        head_major=True,
        interpret=True,
        block_q=64,
        pages_per_step=32,
    )
    # Same products; only the order of the f32 sums differs.
    np.testing.assert_allclose(got, np.asarray(tiles.astype(jnp.float32)), atol=1e-2)


def test_window_fits_vmem():
    """More query heads take a smaller token window; none fitting -> no support."""
    assert msa_attention_pairs._max_window(16, HEAD_DIM) == msa_attention_pairs.WINDOW
    assert msa_attention_pairs._max_window(32, HEAD_DIM) == 400
    assert msa_attention_pairs.supports(32, HEAD_DIM, 128, 2048)
    assert not msa_attention_pairs.supports(4096, HEAD_DIM, 128, 2048)
