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
"""TPU_MINIMAX_M3_MSA_PAGE_CACHE_SCATTER: the presence-table page cache
table equals the sorted one, and the attention keeps its bits (interpret)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import msa_attention_lse, msa_fp4

ENV = "TPU_MINIMAX_M3_MSA_PAGE_CACHE_SCATTER"


def _unions(rng, tiles, width, num_keys, max_union):
    rows = np.full((tiles, width), -1, np.int32)
    for t in range(tiles):
        n = int(rng.integers(0, max_union + 1))
        rows[t, :n] = np.sort(rng.choice(num_keys, n, replace=False))
    return jnp.asarray(rows.reshape(-1))


@pytest.mark.parametrize(
    "tiles,width,num_keys,max_union,cache,tile_pages",
    [
        (8, 256, 300, 40, 512, 32),  # tile unions past the sorted width: streams
        (8, 256, 300, 30, 512, 32),
        (16, 256, 2000, 60, 64, 64),  # more distinct keys than slots
        (16, 64, 50, 50, 512, 128),
        (4, 256, 9, 0, 16, 32),  # no pages at all
    ],
)
def test_table_matches_sorted(monkeypatch, tiles, width, num_keys, max_union, cache, tile_pages):
    monkeypatch.setattr(msa_attention_lse, "PAGE_CACHE_TILE_PAGES", tile_pages)
    rng = np.random.default_rng(tiles * 7 + max_union)
    for _ in range(3):
        pages = _unions(rng, tiles, width, num_keys, max_union)
        ref = msa_attention_lse._page_cache(pages, tiles, cache)
        got = msa_attention_lse._page_cache_scatter(pages, tiles, cache, num_keys)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


@pytest.mark.parametrize("slots", [256, 8])
def test_attention_bits(monkeypatch, slots):
    T, H, D, S, P = 64, 8, 128, 128, 40
    rng = np.random.default_rng(slots)
    q = jnp.asarray(rng.standard_normal((T, H, D)) * 0.5, jnp.bfloat16)
    nib = jax.random.randint(jax.random.PRNGKey(1), (P, S, 2 * D), 0, 16, jnp.int32)
    kv = msa_fp4.pack_rows(nib.astype(jnp.uint8))
    sc = jax.random.uniform(
        jax.random.PRNGKey(2), (P, msa_fp4.SCALE_ROWS, S), jnp.float32, 0.01, 0.03
    )
    sel = np.full((T, 16), -1, np.int32)
    for t in range(T):
        sel[t, :12] = np.sort(rng.choice(P, 12, replace=False))[::-1]
    args = (q, kv, jnp.asarray(sel), jnp.asarray(sel[:, 0]), jnp.full((T,), 70, jnp.int32),
            jnp.asarray(rng.permutation(P).astype(np.int32)))
    kw = dict(sm_scale=D**-0.5, head=0, scale_cache=sc, head_major=True, interpret=True,
              page_cache_slots=slots, step_widths=(4, 8))
    outs = []
    for flag in ("0", "1"):
        monkeypatch.setenv(ENV, flag)
        jax.clear_caches()
        outs.append(msa_attention_lse.msa_sparse_attention_lse(*args, **kw))
    for a, b in zip(*outs):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("block_q", [16, 8])
def test_mask_bcast_bits(monkeypatch, block_q):
    T, H, D, S, P = 64, 8, 128, 128, 40
    rng = np.random.default_rng(block_q)
    q = jnp.asarray(rng.standard_normal((T, H, D)) * 0.5, jnp.bfloat16)
    nib = jax.random.randint(jax.random.PRNGKey(3), (P, S, 2 * D), 0, 16, jnp.int32)
    kv = msa_fp4.pack_rows(nib.astype(jnp.uint8))
    sc = jax.random.uniform(
        jax.random.PRNGKey(4), (P, msa_fp4.SCALE_ROWS, S), jnp.float32, 0.01, 0.03
    )
    sel = np.full((T, 16), -1, np.int32)
    for t in range(T):
        sel[t, :10] = np.sort(rng.choice(P, 10, replace=False))[::-1]
    args = (q, kv, jnp.asarray(sel), jnp.asarray(sel[:, 0]), jnp.full((T,), 50, jnp.int32),
            jnp.arange(P, dtype=jnp.int32))
    kw = dict(sm_scale=D**-0.5, head=0, scale_cache=sc, head_major=True, interpret=True,
              block_q=block_q, page_cache_slots=256, step_widths=(4, 8))
    outs = []
    for flag in ("0", "1"):
        monkeypatch.setenv("TPU_MINIMAX_M3_MSA_MASK_BCAST", flag)
        jax.clear_caches()
        outs.append(msa_attention_lse.msa_sparse_attention_lse(*args, **kw))
    for a, b in zip(*outs):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
