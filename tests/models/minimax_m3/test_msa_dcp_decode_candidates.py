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
"""DCP decode candidates in one kernel, and the cache write folded into it
(interpret mode), against the write + block scores + local top-k path they
replace, bit for bit."""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.models.minimax_m3.packing import pack, pack_index, pack_index_fp4, rounded
from vllm_torchtpu.models.minimax_m3.kernels import msa, msa_dcp, msa_fp4, msa_indexer

PAGE = 128
HEAD_DIM = 128
TOPK = 16
MAX_SEQS = 4


def _case(q_lens, ctx_lens, pages_per_seq, cache_kind, pad, seed, max_seqs=MAX_SEQS):
    rng = np.random.default_rng(seed)
    num_seqs = len(q_lens)
    num_real = sum(q_lens)
    num_pages = max_seqs * pages_per_seq + 8
    seq_lens = np.zeros(max_seqs, np.int32)
    seq_lens[:num_seqs] = np.array(ctx_lens) + np.array(q_lens)
    cu_q = np.full(max_seqs + 1, num_real, np.int32)
    cu_q[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    tables = rng.permutation(np.arange(1, num_pages))[: max_seqs * pages_per_seq]
    raw = rng.standard_normal((num_pages, PAGE, HEAD_DIM))
    if cache_kind == "fp4":
        nib, scale = msa_fp4.quantize(jnp.asarray(raw, jnp.float32))
        cache = pack_index_fp4(nib, scale)
    else:
        cache = pack_index(rounded(raw, cache_kind), cache_kind)
    index_q = jnp.asarray(rng.standard_normal((num_real + pad, 1, HEAD_DIM))).astype(
        jnp.bfloat16
    )
    return (
        index_q,
        cache,
        jnp.asarray(seq_lens),
        jnp.asarray(tables.astype(np.int32)),
        jnp.asarray(cu_q),
        jnp.int32(num_seqs),
        num_real,
    )


@pytest.mark.parametrize("cache_kind", [jnp.float8_e4m3fn, "fp4", jnp.bfloat16])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize(
    "q_lens,ctx_lens,pages_per_seq,init_blocks,pad",
    [
        ([8], [7_000], 40, 0, 0),  # one decode request (c4 b1)
        ([8, 8, 8], [70_000, 40_000, 3], 300, 0, 0),  # 256-page groups, 3 of 4 live
        ([8, 23, 8], [9_000, 4_000, 300], 40, 1, 5),  # unaligned tile starts
        ([120, 8], [5_000, 9_000], 40, 0, 0),  # the most rows of one request
        # bf16 pages score in 64-row tiles: the last runs past the 128 rows.
        ([56, 64, 8], [5_000, 9_000, 700], 40, 0, 0),
    ],
)
def test_matches_scores_then_picks(
    q_lens, ctx_lens, pages_per_seq, init_blocks, pad, rank, cache_kind
):
    dcp = 2
    index_q, cache, seq_lens, tables, cu_q, num_seqs, num_real = _case(
        q_lens, ctx_lens, pages_per_seq, cache_kind, pad, sum(ctx_lens) + rank
    )
    num_tokens = index_q.shape[0]
    seq_id, pos, _ = msa.token_positions(num_tokens, seq_lens, cu_q)
    local_lens = msa_dcp.local_seq_lens(seq_lens, block_size=PAGE, rank=rank, dcp=dcp)
    picks = dict(topk=TOPK, init_blocks=init_blocks, local_blocks=1, rank=rank, dcp=dcp)

    got = msa_dcp.decode_candidates(
        index_q,
        cache,
        local_lens,
        tables,
        cu_q,
        num_seqs,
        seq_id,
        pos,
        **picks,
        interpret=True,
    )
    scores = msa._block_scores(index_q, cache, local_lens, tables, cu_q, num_seqs, True)
    want = msa_dcp.pack_candidates(
        *msa_dcp.local_candidates(scores, pos, block_size=PAGE, **picks, interpret=True)
    )

    assert got.shape == want.shape
    np.testing.assert_array_equal(
        np.asarray(got)[:num_real], np.asarray(want)[:num_real]
    )


@pytest.mark.parametrize("cache_kind", [jnp.float8_e4m3fn, "fp4"])
def test_long_request_in_decode_step_scores_bf16_queries(cache_kind):
    """A request longer than a small tile in a decode-sized step (a short
    chunk beside the decodes) is scored in small tiles with bf16 queries:
    the candidates of `msa_indexer.block_scores` with every request in the
    small class (where the token-order path scores it in an e4m3 large tile,
    so not bit for bit)."""
    rank, dcp, max_seqs = 0, 2, 24
    q_lens = [8] * 6 + [136]
    index_q, cache, seq_lens, tables, cu_q, num_seqs, num_real = _case(
        q_lens, [3_000 + 977 * i for i in range(7)], 40, cache_kind, 3, 7, max_seqs
    )
    num_tokens = index_q.shape[0]
    seq_id, pos, _ = msa.token_positions(num_tokens, seq_lens, cu_q)
    local_lens = msa_dcp.local_seq_lens(seq_lens, block_size=PAGE, rank=rank, dcp=dcp)
    picks = dict(topk=TOPK, init_blocks=0, local_blocks=1, rank=rank, dcp=dcp)
    got = msa_dcp.decode_candidates(
        index_q, cache, local_lens, tables, cu_q, num_seqs, seq_id, pos, **picks,
        interpret=True,
    )  # fmt: skip
    options = msa.block_score_options()
    scores = msa_indexer.block_scores(
        index_q, cache, local_lens, tables, cu_q, num_seqs,
        block_q=2 * options["block_q"],
        small_pages_per_step=options["small_pages_per_step"],
        transposed_tile_q=options["transposed_tile_q"],
        interpret=True,
    )  # fmt: skip
    want = msa_dcp.pack_candidates(
        *msa_dcp.local_candidates(scores, pos, block_size=PAGE, **picks, interpret=True)
    )
    np.testing.assert_array_equal(
        np.asarray(got)[:num_real], np.asarray(want)[:num_real]
    )


def test_wide_4bit_groups_match_256_page_groups():
    """4-bit decode steps over 1024-page tables score 512-page groups
    (`msa._wide_decode_scores`): the same candidates as 256-page groups."""
    rank, dcp = 1, 2
    index_q, cache, seq_lens, tables, cu_q, num_seqs, num_real = _case(
        [8, 8, 8], [140_000, 9_000, 300], 1024, "fp4", 0, 11
    )
    assert msa._wide_decode_scores(index_q, cache, seq_lens, tables, 1) == 4
    num_tokens = index_q.shape[0]
    seq_id, pos, _ = msa.token_positions(num_tokens, seq_lens, cu_q)
    local_lens = msa_dcp.local_seq_lens(seq_lens, block_size=PAGE, rank=rank, dcp=dcp)
    picks = dict(topk=TOPK, init_blocks=0, local_blocks=1, rank=rank, dcp=dcp)
    got = msa_dcp.decode_candidates(
        index_q, cache, local_lens, tables, cu_q, num_seqs, seq_id, pos, **picks,
        interpret=True,
    )  # fmt: skip
    options = msa.block_score_options(1, 2)
    scores = msa_indexer.block_scores(
        index_q, cache, local_lens, tables, cu_q, num_seqs,
        small_pages_per_step=options["small_pages_per_step"],
        transposed_tile_q=options["transposed_tile_q"],
        interpret=True,
    )  # fmt: skip
    want = msa_dcp.pack_candidates(
        *msa_dcp.local_candidates(scores, pos, block_size=PAGE, **picks, interpret=True)
    )
    np.testing.assert_array_equal(
        np.asarray(got)[:num_real], np.asarray(want)[:num_real]
    )


def _write_caches(cache_kind, num_pages, rng):
    if cache_kind == "fp4":
        nibbles = rng.integers(0, 16, (num_pages, PAGE, 3 * HEAD_DIM))
        kv = msa_fp4.pack_rows(jnp.asarray(nibbles[..., : 2 * HEAD_DIM], jnp.uint8))
        index = pack_index_fp4(
            jnp.asarray(nibbles[..., 2 * HEAD_DIM :], jnp.uint8),
            jnp.asarray(rng.uniform(0.5, 1.5, (num_pages, PAGE)), jnp.float32),
        )
        scales = jnp.asarray(
            rng.uniform(0.5, 1.5, (num_pages, msa_fp4.scale_rows(1), PAGE)),
            jnp.float32,
        )
        return kv, index, scales
    dtype = cache_kind
    kv = pack(
        rounded(rng.standard_normal((num_pages, PAGE, 2 * HEAD_DIM)), dtype), dtype
    )
    index = pack_index(
        rounded(rng.standard_normal((num_pages, PAGE, HEAD_DIM)), dtype), dtype
    )
    return kv, index, None


@pytest.mark.parametrize("cache_kind", [jnp.float8_e4m3fn, "fp4", jnp.bfloat16])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize(
    "q_lens,ctx_lens,max_seqs",
    [
        ([8], [70_000], 4),  # one decode request, its last of 3 page groups
        ([8, 8, 8], [3 * 256 + 124, 9_000, 3], 4),  # one straddles a block boundary
        ([8, 23, 8], [9_000, 4_000, 300], 4),  # unaligned tile starts
        ([8] * 12, [5_000 + 997 * i for i in range(12)], 16),  # 12 runs, 96 rows
        # A decode step of more rows than a small tile (c48's 192-row buckets).
        ([8] * 20, [3_000 + 211 * i for i in range(20)], 24),
    ],
)
def test_folded_write_matches_write_then_candidates(
    q_lens, ctx_lens, max_seqs, rank, cache_kind, monkeypatch
):
    dcp, pages_per_seq, pad = 2, 300, 3
    rng = np.random.default_rng(sum(ctx_lens) + rank)
    num_seqs = len(q_lens)
    num_real = sum(q_lens)
    num_tokens = num_real + pad
    num_pages = max_seqs * pages_per_seq + 8
    seq_lens = np.zeros(max_seqs, np.int32)
    seq_lens[:num_seqs] = np.array(ctx_lens) + np.array(q_lens)
    cu_q = np.full(max_seqs + 1, num_real, np.int32)
    cu_q[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    tables = rng.permutation(np.arange(1, num_pages))[: max_seqs * pages_per_seq]
    meta = [
        jnp.asarray(a.astype(np.int32))
        for a in (seq_lens, tables, cu_q, np.array([0, 0, num_seqs]))
    ]
    kv, index, scales = _write_caches(cache_kind, num_pages, rng)
    k, v, index_q = (
        jnp.asarray(rng.standard_normal((num_tokens, 1, HEAD_DIM)), jnp.bfloat16)
        for _ in range(3)
    )
    index_k = jnp.asarray(rng.standard_normal((num_tokens, HEAD_DIM)), jnp.bfloat16)

    def run():
        return msa_dcp.msa_write_and_candidates_dcp(
            k, v, index_q, index_k, kv, index, *meta, topk=TOPK, init_blocks=1,
            local_blocks=1, rank=rank, dcp=dcp, scale_cache=scales, interpret=True,
        )  # fmt: skip

    got = run()
    monkeypatch.setattr(msa_dcp, "decodes_in_one_kernel", lambda *_: False)
    want = run()

    assert len(got) == len(want)
    for g, w in zip(got[:-1], want[:-1]):  # the caches
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))
    np.testing.assert_array_equal(
        np.asarray(got[-1])[:num_real], np.asarray(want[-1])[:num_real]
    )


_THREE_GROUPS = 2 * 128 * 128 * 4 + 9_000  # 3 page groups of 128 on a DCP 4 rank


@pytest.mark.parametrize(
    "cache_kind,rank,dcp,q_lens,ctx_lens,max_seqs,pad",
    [
        ("int4", 3, 4, [8, 5, 8], [_THREE_GROUPS, 9_000, 300], 4, 3),
        ("int4", 0, 4, [8, 5, 8], [_THREE_GROUPS, 9_000, 300], 4, 3),
        (jnp.float8_e4m3fn, 1, 4, [8, 5, 8], [_THREE_GROUPS, 9_000, 300], 4, 3),
        ("int4", 1, 2, [8, 5, 8], [_THREE_GROUPS // 2, 9_000, 300], 4, 3),
        # 256 rows: picked in 256-row blocks.
        ("int4", 2, 4, [8] * 16, [3_000 + 4_000 * i for i in range(16)], 16, 0),
    ],
)
def test_two_heads_match_partial_block_scores_then_candidates(
    cache_kind, rank, dcp, q_lens, ctx_lens, max_seqs, pad, monkeypatch
):
    """Two KV heads per rank (c64: DCP 4): their rows fold into one kernel,
    against `msa.partial_block_scores` + `local_candidates`."""
    from tests.models.minimax_m3.test_msa_dcp_classes import _caches

    heads, pages_per_seq = 2, 300
    rng = np.random.default_rng(10 * dcp + rank)
    num_seqs = len(q_lens)
    num_real = sum(q_lens)
    num_tokens = num_real + pad
    num_pages = max_seqs * pages_per_seq + 8
    seq_lens = np.zeros(max_seqs, np.int32)
    seq_lens[:num_seqs] = np.array(ctx_lens) + np.array(q_lens)
    cu_q = np.full(max_seqs + 1, num_real, np.int32)
    cu_q[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    tables = rng.permutation(np.arange(1, num_pages))[: max_seqs * pages_per_seq]
    meta = [
        jnp.asarray(a.astype(np.int32))
        for a in (seq_lens, tables, cu_q, np.array([num_seqs, 0, num_seqs]))
    ]
    kv, index, scales = _caches(cache_kind, num_pages, rng, heads)
    k, v, index_q = (
        jnp.asarray(rng.standard_normal((num_tokens, heads, HEAD_DIM)), jnp.bfloat16)
        for _ in range(3)
    )
    index_k = jnp.asarray(rng.standard_normal((num_tokens, HEAD_DIM)), jnp.bfloat16)
    assert not msa_indexer.is_prefill_step(num_tokens * heads, max_seqs)
    assert msa_dcp.decodes_in_one_kernel(
        num_tokens * heads, index, max_seqs, pages_per_seq, heads
    )

    def run():
        return msa_dcp.msa_write_and_candidates_dcp(
            k, v, index_q, index_k, kv, index, *meta, topk=TOPK, init_blocks=0,
            local_blocks=1, rank=rank, dcp=dcp, scale_cache=scales, interpret=True,
        )  # fmt: skip

    got = run()
    monkeypatch.setattr(msa_dcp, "decodes_in_one_kernel", lambda *_: False)
    want = run()

    assert len(got) == len(want)
    for g, w in zip(got[:-1], want[:-1]):  # the caches
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))
    np.testing.assert_array_equal(
        np.asarray(got[-1])[:num_real], np.asarray(want[-1])[:num_real]
    )
