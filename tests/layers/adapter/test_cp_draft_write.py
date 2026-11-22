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
"""The drafter's run-wise KV write against the XLA scatter it replaces."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.layers.adapter.cp_draft_attention import (
    _token_layout,
    write_kv,
    write_kv_layers,
    write_pairs,
    write_runs,
    write_runs_batched,
)


@pytest.mark.parametrize("write", [write_runs, write_runs_batched])
@pytest.mark.parametrize(
    "shape, dtype",
    [((40, 128, 2, 4, 128), jnp.float8_e4m3fn), ((30, 128, 1, 4, 128), jnp.bfloat16)],
)
def test_write_runs_matches_scatter(write, shape, dtype):
    rng = np.random.default_rng(0)
    num_pages = shape[0]
    cache = jnp.asarray(rng.standard_normal(shape), jnp.float32).astype(dtype)
    # A 700-token prefill from position 300, then five 8-token decode rows;
    # pages past 36 and every other block of the decode rows are dropped.
    pages = [7 + (300 + t) // 128 for t in range(700)]
    slots = [(300 + t) % 128 for t in range(700)]
    for s in range(5):
        base = int(rng.integers(0, 3 * 128))
        for t in range(8):
            pos = base + t
            pages.append(1 + 2 * s if (pos // 128) % 2 == 0 else num_pages)
            slots.append(pos % 128)
    pages = jnp.asarray(pages, jnp.int32)
    pages = jnp.where(pages > 36, num_pages, pages)
    slots = jnp.asarray(slots, jnp.int32)
    rows = jnp.asarray(
        rng.standard_normal((slots.shape[0], *shape[2:])), jnp.float32
    ).astype(dtype)
    expected = cache.at[pages, slots].set(rows, mode="drop")
    got = write(
        cache, rows, pages, slots, max_runs=slots.shape[0] // 128 + 16, interpret=True
    )
    np.testing.assert_array_equal(
        np.asarray(got.astype(jnp.float32)), np.asarray(expected.astype(jnp.float32))
    )


def test_write_pairs_matches_half_word_scatter():
    """Half-word rows (two token-heads per 8-bit word) at odd and even run
    starts and ends: equal to a scatter of the halves, except each run's
    half after an odd end (the position after the run, filled with the next
    row), which stays finite."""
    rng = np.random.default_rng(1)
    num_pages, page_words = 24, 256
    cache = jnp.asarray(
        rng.standard_normal((num_pages, page_words, 4, 128)), jnp.float32
    ).astype(jnp.float8_e4m3fn)
    # (page, first half, length) runs; one is dropped (page out of range).
    runs = [
        (3, 0, 300),
        (5, 7, 1),
        (6, 8, 1),
        (7, 13, 6),
        (num_pages, 2, 5),
        (9, 511, 1),
        (11, 100, 129),
    ]
    pages = np.concatenate([np.full(n, p) for p, _, n in runs])
    slots = np.concatenate([np.arange(lo, lo + n) for _, lo, n in runs])
    rows = jnp.asarray(
        rng.standard_normal((slots.shape[0], 2, 128)), jnp.float32
    ).astype(jnp.float8_e4m3fn)
    halves = cache.reshape(num_pages, page_words, 2, 2, 128)
    expected = halves.at[pages, slots // 2, slots % 2].set(rows, mode="drop")
    got = write_pairs(
        cache,
        rows,
        jnp.asarray(pages, jnp.int32),
        jnp.asarray(slots, jnp.int32),
        max_runs=16,
        interpret=True,
    ).reshape(halves.shape)
    keep = np.ones(halves.shape[:3], bool)
    for page, lo, n in runs:
        if page < num_pages and (lo + n) % 2:
            keep[page, (lo + n) // 2, 1] = False
    got, expected = np.asarray(got, np.float32), np.asarray(expected, np.float32)
    np.testing.assert_array_equal(got[keep], expected[keep])
    assert np.isfinite(got).all()


def test_token_layout_matches_search():
    """Each real token's sequence and position equal a search of the query
    starts (empty and trailing padding requests included)."""
    q_lens = np.asarray([3, 0, 1, 7, 0, 2, 0, 0])
    seq_lens = np.asarray([10, 4, 1, 300, 0, 9, 0, 0])
    cu = np.concatenate([[0], np.cumsum(q_lens)])
    num_tokens = 16
    seq, pos, valid = _token_layout(
        jnp.asarray(seq_lens, jnp.int32), jnp.asarray(cu, jnp.int32), num_tokens
    )
    tok = np.arange(num_tokens)
    want_seq = np.searchsorted(cu[1:], tok, side="right")
    real = tok < cu[-1]
    np.testing.assert_array_equal(np.asarray(valid), real)
    np.testing.assert_array_equal(np.asarray(seq)[real], want_seq[real])
    s = want_seq[real]
    want_pos = seq_lens[s] - q_lens[s] + tok[real] - cu[s]
    np.testing.assert_array_equal(np.asarray(pos)[real], want_pos)
    np.testing.assert_array_equal(np.asarray(pos)[~real], 0)


@pytest.mark.parametrize("dcp, head_ranks", [(2, 2), (8, 2), (8, 1)])
def test_write_kv_layers_matches_each_layer(dcp, head_ranks):
    """Every layer's draft-block write from one call (one kernel for a
    decode batch's runs) equals `write_kv` layer by layer, on every rank."""
    rng = np.random.default_rng(3)
    layers, num_pages, head_dim = 3, 40, 128
    kv_heads = dcp // head_ranks
    words = -(-2 * kv_heads // 4)
    ctx_lens = [1501, 300, 1033, 127]  # page crossings, a block edge
    num_reqs, block = len(ctx_lens), 7
    tables = jnp.asarray(
        rng.permutation(np.arange(1, num_pages))[: num_reqs * 8].reshape(num_reqs, 8),
        jnp.int32,
    )
    num_tokens = 32  # a padded decode bucket
    seq_lens = jnp.asarray(ctx_lens, jnp.int32) + block
    cu = jnp.asarray(np.arange(num_reqs + 1) * block, jnp.int32)
    keys = jnp.asarray(
        rng.standard_normal((layers, num_tokens, 1, head_dim)), jnp.bfloat16
    )
    values = jnp.asarray(
        rng.standard_normal((layers, num_tokens, 1, head_dim)), jnp.bfloat16
    )
    for rank in range(dcp):
        caches = tuple(
            jnp.asarray(
                rng.standard_normal((num_pages, 128, words, 4, head_dim)), jnp.float32
            ).astype(jnp.float8_e4m3fn)
            for _ in range(layers)
        )
        common = dict(dcp=dcp, head_ranks=head_ranks, rank=jnp.int32(rank))
        want = [
            write_kv(
                c, keys[i], values[i], seq_lens, tables, cu, interpret=True, **common
            )
            for i, c in enumerate(caches)
        ]
        got = write_kv_layers(
            caches, keys, values, seq_lens, tables, cu, interpret=True, **common
        )
        for w, g in zip(want, got, strict=True):
            np.testing.assert_array_equal(
                np.asarray(g.astype(jnp.float32)), np.asarray(w.astype(jnp.float32))
            )
