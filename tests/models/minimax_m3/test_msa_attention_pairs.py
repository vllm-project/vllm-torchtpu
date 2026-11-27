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


def test_window_fits_vmem_and_sort_key():
    """More query heads, or page keys, take a smaller token window; none
    fitting -> no support."""
    window = msa_attention_pairs._max_window
    assert window(16, HEAD_DIM, 128, 2048) == msa_attention_pairs.WINDOW
    assert 8 <= window(32, HEAD_DIM, 128, 2048) < msa_attention_pairs.WINDOW
    assert window(16, HEAD_DIM, 128, 4096) == 1024
    assert not msa_attention_pairs.supports(16, HEAD_DIM, 128, 8192)
    assert window(16, HEAD_DIM, 128, 8192, window=152) % 8 == 0
    assert msa_attention_pairs.supports(32, HEAD_DIM, 128, 2048)
    assert not msa_attention_pairs.supports(4096, HEAD_DIM, 128, 2048)
    # Every page's scale rows stay in VMEM: a large cache takes window room.
    assert window(16, HEAD_DIM, 128, 2048, num_pages=16384) < msa_attention_pairs.WINDOW
    assert not msa_attention_pairs.supports(16, HEAD_DIM, 128, 2048, num_pages=48000)


def test_pairs_many_blocks_of_mixed_sizes():
    """More blocks than a unit and the copy ring hold, of every width: many
    page keys (aliasing the few physical pages) picked by a handful of tokens
    each, and hot keys whose runs are cut into several blocks."""
    rng = np.random.default_rng(7)
    num_tokens, num_keys, hot = 300, 90, 3
    q = jnp.asarray(rng.standard_normal((num_tokens, Q_PER_KV, HEAD_DIM))).astype(
        jnp.bfloat16
    )
    table = rng.integers(0, PAGES, num_keys).astype(np.int32)
    keys = np.full((num_tokens, TOPK), -1, np.int32)
    local_key = np.full(num_tokens, -1, np.int32)
    local_limit = np.zeros(num_tokens, np.int32)
    for t in range(num_tokens):
        chosen = np.concatenate(
            [rng.choice(hot, 2, replace=False), hot + rng.choice(num_keys - hot, 3)]
        )
        keys[t, : len(chosen)] = chosen
        local_key[t] = chosen[-1]
        local_limit[t] = rng.integers(1, 129)
    kv, scales, rows = _fp4_cache(rng, 1)
    got = msa_sparse_attention_pairs(
        q,
        kv,
        jnp.asarray(keys),
        jnp.asarray(local_key),
        jnp.asarray(local_limit),
        jnp.asarray(table),
        sm_scale=SCALE,
        scale_cache=jnp.asarray(scales),
        window=152,
        interpret=True,
    )
    got = np.asarray(got.astype(jnp.float32))
    dedup = keys.copy()
    for t in range(num_tokens):  # a page listed twice counts once
        _, first = np.unique(keys[t], return_index=True)
        dup = np.ones(TOPK, bool)
        dup[first] = False
        dedup[t, dup] = -1
    want, _ = _reference(
        np.asarray(q.astype(jnp.float32)), rows, dedup, local_key, local_limit, table, 0
    )
    np.testing.assert_allclose(got, want.transpose(1, 0, 2), atol=2e-2)


def test_pairs_window_on_page_key_zero_only():
    """A window whose few pairs all select page key 0: its block holds
    those pairs only, not the padding past them (key 0 too)."""
    rng = np.random.default_rng(3)
    num_tokens = 40
    q = jnp.asarray(rng.standard_normal((num_tokens, Q_PER_KV, HEAD_DIM))).astype(
        jnp.bfloat16
    )
    table = rng.permutation(PAGES).astype(np.int32)
    keys = np.full((num_tokens, TOPK), -1, np.int32)
    keys[:3, 0] = 0
    local_key = np.full(num_tokens, -1, np.int32)
    local_limit = np.zeros(num_tokens, np.int32)
    local_key[1], local_limit[1] = 0, 70
    kv, scales, rows = _fp4_cache(rng, 1)
    got = msa_sparse_attention_pairs(
        q,
        kv,
        jnp.asarray(keys),
        jnp.asarray(local_key),
        jnp.asarray(local_limit),
        jnp.asarray(table),
        sm_scale=SCALE,
        scale_cache=jnp.asarray(scales),
        window=32,
        interpret=True,
    )
    got = np.asarray(got.astype(jnp.float32))
    want, _ = _reference(
        np.asarray(q.astype(jnp.float32)), rows, keys, local_key, local_limit, table, 0
    )
    assert np.abs(got[:, :3]).max() > 0
    np.testing.assert_allclose(got, want.transpose(1, 0, 2), atol=2e-2)


def _work_reference(keys, local_key, local_limit, window, page_size):
    """`msa_attention_pairs._work`'s outputs, in numpy: each window's pairs
    sorted by (page key, token), cut into runs per page and blocks of at most
    the widest width."""
    widest, widths = msa_attention_pairs._WIDTHS[-1], len(msa_attention_pairs._WIDTHS)
    blocks, words, num_pairs, width_blocks = [], [], [], []
    length = window * keys.shape[1]
    pad = -(length + 128) % 1024 + 128
    for w in range(keys.shape[0] // window):
        pairs = set()
        for t in range(w * window, (w + 1) * window):
            for k in set(keys[t][keys[t] >= 0].tolist()):
                vis = local_limit[t] if k == local_key[t] else page_size
                pairs.add((k, t - w * window, vis))
        pairs = sorted(pairs)
        b = np.zeros(length + pad, np.int64)
        b[len(pairs) : length] = 1 << 24  # empty words: blocks of one
        wd = np.zeros(length + pad, np.int64)
        counts = np.zeros(widths, np.int64)
        i = 0
        while i < len(pairs):
            run = 1
            while i + run < len(pairs) and pairs[i + run][0] == pairs[i][0]:
                run += 1
            for j in range(run):
                size = min(run - j, widest)
                b[i + j] = pairs[i][0] | size << 24
                if j % widest == 0:
                    counts[max((size - 1) >> 2, 1) - 1] += 1
            i += run
        for i, (_, tok, vis) in enumerate(pairs):
            wd[i] = tok | vis << 16
        blocks.append(b)
        words.append(wd)
        num_pairs.append(len(pairs))
        width_blocks.append(counts)
    return [
        np.concatenate(blocks),
        np.concatenate(words),
        np.array(num_pairs),
        np.concatenate(width_blocks),
    ]


@pytest.mark.parametrize(
    "window,num_keys", [(32, 12), (152, 90), (48, 2000), (688, 300)]
)
def test_work_list_matches_reference(window, num_keys):
    """The work list's kernel (sort, runs, blocks, counts) against numpy, in
    windows whose pairs do not fill a power of two, with pages listed twice
    and tokens selecting nothing; 688 tokens hold their pairs in eight tiles
    (cross-tile compares, neighbours and scan carries)."""
    rng = np.random.default_rng(window)
    num_tokens = 2 * window
    keys = rng.integers(-1, num_keys, (num_tokens, TOPK)).astype(np.int32)
    keys[::5, 1] = keys[::5, 0]
    keys[3] = -1
    local_key = keys[:, 0].copy()
    local_limit = rng.integers(1, 129, num_tokens).astype(np.int32)
    got = msa_attention_pairs._work(
        jnp.asarray(keys),
        jnp.asarray(local_key),
        jnp.asarray(local_limit),
        window,
        128,
        interpret=True,
    )
    want = _work_reference(keys, local_key, local_limit, window, 128)
    for g, w in zip(got, want):
        np.testing.assert_array_equal(np.asarray(g), w)
