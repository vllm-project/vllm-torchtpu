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
"""MSA DCP candidates: a decode step's block scores come back as the indexer's
token rows, picking from those rows equals picking from the gathered scores,
and the merge of the ranks' candidates does not
depend on their order.

``interpret=True`` runs the kernel body on the host; it does not gate Mosaic
lowering.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.minimax_m3.kernels import (
    msa,
    msa_attention,
    msa_attention_lse,
    msa_dcp,
    msa_fp4,
    msa_indexer,
    msa_write,
)

_TOPK = 16
_BLOCK = 128


def _pick(scores, pos, rank, *, init_blocks=1, local_blocks=2, **kwargs):
    return msa_dcp.local_candidates(
        scores,
        pos,
        block_size=_BLOCK,
        topk=_TOPK,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        rank=rank,
        dcp=2,
        interpret=True,
        **kwargs,
    )


# Row-major blocks (8, 24 tokens) and a transposed one (128).
@pytest.mark.parametrize("tokens", [8, 24, 128])
@pytest.mark.parametrize("rank", [0, 1])
def test_token_rows_match_gathered_scores(tokens: int, rank: int) -> None:
    rng = np.random.default_rng(tokens + rank)
    num_rows, cols = 3 * tokens + 8, 256
    # Few distinct values, so most picks break ties.
    scores = jnp.asarray(rng.integers(-3, 4, (num_rows, cols)), jnp.float32)
    rows = jnp.asarray(rng.permutation(num_rows)[:tokens], jnp.int32)
    pos = jnp.asarray(rng.integers(0, 2 * _BLOCK * cols, tokens), jnp.int32)
    want = _pick(scores[rows][:, None, :], pos, rank)
    got = _pick(scores, pos, rank, rows=rows)
    for w, g in zip(want, got):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))


def test_decode_step_returns_token_rows() -> None:
    rng = np.random.default_rng(0)
    seqs, pages_per_seq, pages, dim = 4, 8, 96, 128
    q_lens = np.array([8, 5, 8, 0])
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]), jnp.int32)
    seq_lens = jnp.asarray(np.where(q_lens > 0, [900, 300, 1024, 0], 0), jnp.int32)
    args = (
        jnp.asarray(rng.standard_normal((24, 1, dim)), jnp.bfloat16),
        jnp.asarray(rng.integers(-2, 3, (pages, 32, 4, dim)), jnp.float8_e4m3fn),
        seq_lens,
        jnp.asarray(rng.permutation(pages)[: seqs * pages_per_seq], jnp.int32),
        cu,
        jnp.asarray(3, jnp.int32),
    )
    options = dict(**msa.block_score_options(), interpret=True)
    rows = msa_indexer.block_scores(*args, token_rows=True, **options)
    assert isinstance(rows, msa_indexer.TokenRows)
    gathered = msa_indexer.block_scores(*args, **options)
    np.testing.assert_array_equal(
        np.asarray(rows.scores)[np.asarray(rows.rows)], np.asarray(gathered)[:, 0]
    )


# Transposed picks: four column blocks up to 32 rows, two up to 64, one above
# (4 and 24 rows leave lanes spare, 32 and 64 fill them).
@pytest.mark.parametrize("width", [256, 512, 768, 1024])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("tokens", [4, 24, 32, 48, 64, 72])
@pytest.mark.parametrize("ties", [True, False])
def test_packed_pick_matches_local_candidates(width, rank, tokens, ties) -> None:
    rng = np.random.default_rng(width + rank + tokens)
    # Few distinct values, so most picks break ties.
    values = (
        rng.integers(-3, 4, (tokens, width))
        if ties
        else rng.normal(size=(tokens, width))
    )
    scores = jnp.asarray(values, jnp.float32)
    pos = jnp.asarray(rng.integers(0, 2 * _BLOCK * width, tokens), jnp.int32)
    want = msa_dcp.pack_candidates(*_pick(scores[:, None, :], pos, rank))
    got = _run_pick(msa_dcp._PackedPick(_BLOCK, _TOPK, 1, 2, 2), scores, pos, rank)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def _run_pick(pick, scores, pos, rank):
    """`pick` in an interpret-mode kernel, as the select kernel runs it."""
    run = _compiled_pick(pick, *scores.shape)
    return run(jnp.asarray([rank], jnp.int32), scores, pos[:, None])


@functools.cache
def _compiled_pick(pick, rows, width):
    def call(rank, scores, pos):
        def kernel(rank_ref, s_ref, pos_ref, o_ref):
            o_ref[...] = pick(s_ref[...], pos_ref[...], rank_ref[0])

        return pl.pallas_call(
            kernel,
            in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)] + [pl.BlockSpec()] * 2,
            out_shape=jax.ShapeDtypeStruct((rows, 2 * pick.topk), jnp.float32),
            interpret=True,
        )(rank, scores, pos)

    args = (
        jax.ShapeDtypeStruct((1,), jnp.int32),
        jax.ShapeDtypeStruct((rows, width), jnp.float32),
        jax.ShapeDtypeStruct((rows, 1), jnp.int32),
    )
    # XLA:CPU's fusion copies the sorting networks' shared operands into each
    # consumer: ~10x the ops and minutes of compile per shape.
    return (
        jax.jit(call)
        .lower(*args)
        .compile(compiler_options={"xla_disable_hlo_passes": "fusion"})
    )


# Served options (no forced initial block, one local block) at the decode
# buckets' rows and more, with signed zeros (equal scores of other bits) and
# rows that see nothing, one block, or every block.
@pytest.mark.parametrize("width", [256, 512, 768, 1024])
@pytest.mark.parametrize("tokens", [4, 8, 16, 24, 40, 48, 64, 72])
@pytest.mark.parametrize("values", ["ties", "zeros", "normal"])
def test_packed_pick_matches_row_major_pick(width, tokens, values) -> None:
    kinds = ["ties", "zeros", "normal"]
    rng = np.random.default_rng(3 * width * tokens + kinds.index(values))
    shape = (tokens, width)
    scores = {
        "ties": lambda: rng.integers(-3, 4, shape).astype(np.float32),
        "zeros": lambda: np.where(
            rng.random(shape) < 0.3,
            np.float32(-1),
            np.where(rng.random(shape) < 0.5, np.float32(-0.0), np.float32(0.0)),
        ),
        "normal": lambda: rng.normal(size=shape).astype(np.float32),
    }[values]()
    assert values != "zeros" or np.signbit(scores[scores == 0]).any()
    scores = jnp.asarray(scores)
    pos = rng.integers(0, 2 * _BLOCK * width, tokens)
    pos[:3] = 0, 2 * _BLOCK * 5, 2 * _BLOCK * width - 1
    pos = jnp.asarray(pos, jnp.int32)
    for rank in (0, 1):
        want = msa_dcp.pack_candidates(
            *msa_dcp._pick(
                scores,
                pos[:, None] // _BLOCK + 1,
                rank,
                topk=_TOPK,
                init_blocks=0,
                local_blocks=1,
                dcp=2,
            )
        )
        pick = msa_dcp._PackedPick(_BLOCK, _TOPK, 0, 1, 2)
        got = np.asarray(_run_pick(pick, scores, pos, rank))
        want = np.asarray(want)
        # Scores as values (the pick returns a candidate's own zero, `_pick`
        # its reduction's), the blocks bit for bit.
        np.testing.assert_array_equal(got[:, :_TOPK], want[:, :_TOPK])
        np.testing.assert_array_equal(
            got[:, _TOPK:].view(np.uint32), want[:, _TOPK:].view(np.uint32)
        )


# One vreg of picks (folded), three (not a power of two: unfolded), four, and
# eight in a row narrower than four blocks of them (two blocks).
@pytest.mark.parametrize("topk, width", [(8, 512), (24, 512), (32, 512), (64, 128)])
@pytest.mark.parametrize("rank", [0, 1])
def test_packed_pick_other_topk(topk, width, rank) -> None:
    rng = np.random.default_rng(topk + rank)
    tokens = 8
    scores = jnp.asarray(rng.integers(-3, 4, (tokens, width)), jnp.float32)
    pos = jnp.asarray(rng.integers(0, 2 * _BLOCK * width, tokens), jnp.int32)
    want = msa_dcp.pack_candidates(
        *msa_dcp._pick(
            scores,
            pos[:, None] // _BLOCK + 1,
            rank,
            topk=topk,
            init_blocks=1,
            local_blocks=2,
            dcp=2,
        )
    )
    got = _run_pick(msa_dcp._PackedPick(_BLOCK, topk, 1, 2, 2), scores, pos, rank)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


# One page group, two (the second partly live), a whole one (pages read in
# 16-bit pairs), and an odd block-table width (packed rows straddle words).
@pytest.mark.parametrize("pages_per_seq", [8, 160, 128, 9])
@pytest.mark.parametrize("rank", [0, 1])
def test_decode_write_and_candidates_match_separate_calls(pages_per_seq, rank):
    """A decode step writes, scores and picks in one kernel: the same caches
    and candidates as the write, the token-row scores and their picks."""
    rng = np.random.default_rng(pages_per_seq + rank)
    seqs, dim, tokens, dcp = 4, 128, 24, 2
    pages = seqs * pages_per_seq + 8
    # Request 2 starts mid-tile, so its tile's first rows are request 1's;
    # requests 0 and 2 straddle a block: their first tokens' page is scored by
    # their last tokens, whose own block is the other rank's.
    q_lens = np.array([8, 5, 8, 0])
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]), jnp.int32)
    full = dcp * _BLOCK * pages_per_seq
    # Slot 3 is past the batch, with an earlier request's length (as served).
    seq_lens = jnp.asarray(
        [
            _BLOCK * (full // _BLOCK // 3) + 3,
            300,
            _BLOCK * (full // _BLOCK - 2) + 3,
            full,
        ],
        jnp.int32,
    )
    block_tables = jnp.asarray(
        rng.permutation(pages)[: seqs * pages_per_seq], jnp.int32
    )
    bf16 = functools.partial(
        lambda *s: jnp.asarray(rng.standard_normal(s), jnp.bfloat16)
    )
    k, v, index_q = bf16(tokens, 1, dim), bf16(tokens, 1, dim), bf16(tokens, 1, dim)
    index_k = bf16(tokens, dim)
    fp8 = jnp.float8_e4m3fn
    kv_cache = jnp.asarray(rng.integers(-2, 3, (pages, 32, 4, 2 * dim)), fp8)
    # Stale keys far above the new ones: a page scored before its write lands
    # scores higher than written.
    index_cache = jnp.asarray(rng.integers(-2, 3, (pages, 32, 4, dim)) * 64, fp8)
    meta = (seq_lens, block_tables, cu, jnp.asarray([3, 3, 3], jnp.int32))
    got = msa_dcp.msa_write_and_candidates_dcp(
        k,
        v,
        index_q,
        index_k,
        kv_cache,
        index_cache,
        *meta,
        topk=_TOPK,
        init_blocks=0,
        local_blocks=1,
        rank=rank,
        dcp=dcp,
        interpret=True,
    )

    seq_id, pos, valid = msa.token_positions(tokens, seq_lens, cu)
    page = msa_dcp.owner_pages(
        block_tables.reshape(seqs, -1),
        seq_id,
        pos,
        valid,
        block_size=_BLOCK,
        rank=rank,
        dcp=dcp,
    )
    kv_rows = jnp.concatenate([k, v], axis=-1).reshape(tokens, -1)
    want_kv, want_index = msa.write_caches(
        [msa_write.Rows(kv_cache, kv_rows), msa_write.Rows(index_cache, index_k)],
        page,
        pos,
        max_num_seqs=seqs,
        interpret=True,
    )
    local_lens = msa_dcp.local_seq_lens(seq_lens, block_size=_BLOCK, rank=rank, dcp=dcp)
    rows = msa_indexer.block_scores(
        index_q,
        want_index,
        local_lens,
        block_tables,
        cu,
        meta[3][2],
        token_rows=True,
        **msa.block_score_options(),
        interpret=True,
    )
    assert isinstance(rows, msa_indexer.TokenRows)
    # As served (`sparse_local_block` 1): the straddled block's score counts.
    want = msa_dcp.pack_candidates(
        *_pick(rows.scores, pos, rank, rows=rows.rows, init_blocks=0, local_blocks=1)
    )
    np.testing.assert_array_equal(np.asarray(got[0]), np.asarray(want_kv))
    np.testing.assert_array_equal(np.asarray(got[1]), np.asarray(want_index))
    live = int(cu[-1])
    np.testing.assert_array_equal(np.asarray(got[2])[:live], np.asarray(want)[:live])


def test_decode_step_with_no_page_on_rank() -> None:
    """The runner's dummy batches: one-token requests of length 1 in every
    slot, every block-table entry page 0. Rank 1 holds no page of any
    request; its candidates are all missing, not unset."""
    rng = np.random.default_rng(1)
    # Every request slot in use (the bucket's dummy batch of 16 tokens).
    seqs, dim, tokens = 16, 128, 16
    q_lens = np.ones(seqs, np.int64)
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]), jnp.int32)
    bf16 = functools.partial(
        lambda *s: jnp.asarray(rng.standard_normal(s), jnp.bfloat16)
    )
    fp8 = jnp.float8_e4m3fn
    got = msa_dcp.msa_write_and_candidates_dcp(
        bf16(tokens, 1, dim),
        bf16(tokens, 1, dim),
        bf16(tokens, 1, dim),
        bf16(tokens, dim),
        jnp.asarray(rng.integers(-2, 3, (16, 32, 4, 2 * dim)), fp8),
        jnp.asarray(rng.integers(-2, 3, (16, 32, 4, dim)), fp8),
        jnp.ones((seqs,), jnp.int32),
        jnp.zeros((seqs * 8,), jnp.int32),
        cu,
        jnp.asarray([tokens] * 3, jnp.int32),
        topk=_TOPK,
        init_blocks=0,
        local_blocks=1,
        rank=1,
        dcp=2,
        interpret=True,
    )[2]
    np.testing.assert_array_equal(np.asarray(got[:, :_TOPK]), -np.inf)
    np.testing.assert_array_equal(np.asarray(got[:, _TOPK:]), -1)


def _exact_block_maxima(index_q, index_cache, block_tables, pages, pages_per_seq):
    """``[T, pages_per_seq]`` block maxima of integer-valued queries and keys
    (exact in f32 in any order), per token over its request's first `pages`
    entries (0 past them); `index_q` ``[T, D]``, `pages` per token."""
    q = np.asarray(index_q, np.float32)
    keys = np.asarray(index_cache.astype(jnp.float32)).reshape(
        index_cache.shape[0], -1, index_cache.shape[-1]
    )
    out = np.zeros((q.shape[0], pages_per_seq), np.float32)
    for t, (row, n) in enumerate(pages):
        k = keys[
            np.asarray(block_tables)[row * pages_per_seq : row * pages_per_seq + n]
        ]
        out[t, :n] = (k @ q[t]).max(axis=1)
    return out


def _integers(rng, shape, dtype, bound):
    return jnp.asarray(rng.integers(-bound, bound + 1, shape), dtype)


def test_partial_group_scores_ignore_stale_pages() -> None:
    """A request's last page group, partly live, scores its live pages
    exactly, whatever its ring slot holds past them: here the NaN pages a
    group three steps earlier left there (the score ring has 3 slots). A
    32-page chunk scores pages j and j + 16 in one contraction, so unmasked
    stale keys would spoil live pages' scores."""
    rng = np.random.default_rng(7)
    dim, pages_per_seq = 128, 512
    # 3 whole groups, then 100 live pages: past 64, so the last group's first
    # half is known live; its pages 112.. (pairs of 96..) hold stale NaN. The
    # NaN pages fill a chunk: a page paired with a NaN page is spoilt.
    live = 3 * 128 + 100
    pages = pages_per_seq + 8
    block_tables = jnp.asarray(rng.permutation(pages)[:pages_per_seq], jnp.int32)
    index_cache = _integers(rng, (pages, 32, 4, dim), jnp.float8_e4m3fn, 2)
    stale = np.asarray(block_tables)[96:128]
    index_cache = index_cache.at[stale].set(jnp.nan)
    index_q = _integers(rng, (8, 1, dim), jnp.bfloat16, 3)
    cu = jnp.asarray([0, 8], jnp.int32)
    rows = msa_indexer.block_scores(
        index_q,
        index_cache,
        jnp.asarray([128 * live], jnp.int32),
        block_tables,
        cu,
        jnp.asarray(1, jnp.int32),
        token_rows=True,
        **msa.block_score_options(),
        interpret=True,
    )
    got = np.asarray(rows.scores)[np.asarray(rows.rows)][:, :live]
    want = _exact_block_maxima(
        index_q[:, 0], index_cache, block_tables, [(0, live)] * 8, pages_per_seq
    )[:, :live]
    # The NaN pages' own maxima are not compared (a max may skip NaN).
    fresh = np.r_[0:96, 128:live]
    assert not np.isnan(want[:, fresh]).any()
    np.testing.assert_array_equal(got[:, fresh], want[:, fresh])


# Requests' last groups: 100 / 75 / 72 / 74 live pages (past 64), 40 and 1
# (within 64), and whole groups.
@pytest.mark.parametrize(
    "local_pages", [[484], [484, 75, 1, 128, 168, 330]], ids=["t8", "t48"]
)
@pytest.mark.parametrize("rank", [0, 1])
def test_decode_candidates_match_exact_scores(local_pages, rank):
    """The decode step's write, block scores and pick in one kernel equal the
    pick of exact block maxima of the written cache (integer-valued queries
    and keys: their f32 dot products are exact in any order)."""
    rng = np.random.default_rng(len(local_pages) + rank)
    seqs, dim, dcp, pages_per_seq = len(local_pages), 128, 2, 512
    tokens = 8 * seqs
    pages = seqs * pages_per_seq + 8
    cu = jnp.asarray(np.arange(seqs + 1) * 8, jnp.int32)
    # `n` blocks on each rank, the last one partly written.
    seq_lens = jnp.asarray([_BLOCK * (2 * n - 1) + 5 for n in local_pages], jnp.int32)
    block_tables = jnp.asarray(
        rng.permutation(pages)[: seqs * pages_per_seq], jnp.int32
    )
    fp8, bf16 = jnp.float8_e4m3fn, jnp.bfloat16
    index_q = _integers(rng, (tokens, 1, dim), bf16, 3)
    got = msa_dcp.msa_write_and_candidates_dcp(
        _integers(rng, (tokens, 1, dim), bf16, 2),
        _integers(rng, (tokens, 1, dim), bf16, 2),
        index_q,
        _integers(rng, (tokens, dim), bf16, 2),
        _integers(rng, (pages, 32, 4, 2 * dim), fp8, 2),
        _integers(rng, (pages, 32, 4, dim), fp8, 2),
        seq_lens,
        block_tables,
        cu,
        jnp.asarray([tokens] * 3, jnp.int32),
        topk=_TOPK,
        init_blocks=0,
        local_blocks=1,
        rank=rank,
        dcp=dcp,
        interpret=True,
    )
    local = msa_dcp.local_seq_lens(seq_lens, block_size=_BLOCK, rank=rank, dcp=dcp)
    local = np.asarray(-(-local // _BLOCK))
    seq_id, pos, _ = msa.token_positions(tokens, seq_lens, cu)
    owner = [(int(r), int(local[r])) for r in np.asarray(seq_id)]
    scores = _exact_block_maxima(
        index_q[:, 0], got[1], block_tables, owner, pages_per_seq
    )
    want = msa_dcp.pack_candidates(
        *_pick(
            jnp.asarray(scores)[:, None, :], pos, rank, init_blocks=0, local_blocks=1
        )
    )
    np.testing.assert_array_equal(np.asarray(got[2]), np.asarray(want))


def _reference_merge(candidates: np.ndarray) -> np.ndarray:
    """Per token: the real candidates by score descending, then block."""
    out = -np.ones(candidates.shape[1:2] + (_TOPK,), np.int32)
    for t in range(candidates.shape[1]):
        s = candidates[:, t, :_TOPK].reshape(-1)
        b = candidates[:, t, _TOPK:].reshape(-1).astype(np.int32)
        order = sorted((-s[i], b[i]) for i in range(s.size) if s[i] > -np.inf)
        picked = [blk for _, blk in order[:_TOPK]]
        out[t, : len(picked)] = picked
    return out


@pytest.mark.parametrize("tokens", [8, 48])
def test_merge_ignores_rank_order(tokens: int) -> None:
    rng = np.random.default_rng(tokens)
    ranks = []
    for rank in range(2):
        scores = rng.integers(-2, 3, (tokens, _TOPK)).astype(np.float32)
        scores[rng.random((tokens, _TOPK)) < 0.2] = -np.inf
        blocks = 2 * rng.permuted(np.tile(np.arange(64), (tokens, 1)), axis=1)
        blocks = (blocks[:, :_TOPK] + rank).astype(np.float32)
        blocks[scores == -np.inf] = -1
        ranks.append(np.concatenate([scores, blocks], axis=1))
    ordered = np.stack(ranks)
    want = _reference_merge(ordered)
    for candidates in (ordered, ordered[::-1]):
        got = msa_dcp.merge_head_candidates(jnp.asarray(candidates), _TOPK)
        np.testing.assert_array_equal(np.asarray(got)[:, 0], want)


def _rank_candidates(rng, tokens, pages_per_seq, rank, dcp, shared):
    """Packed candidates of one rank: distinct blocks of the rank, scores with
    ties (few values), some rows short of topk (-inf, -1). `shared`: every
    row picks the same blocks but for a few, as a request's rows do served."""
    first = rng.permutation(pages_per_seq)
    blocks = []
    for row in range(tokens):
        mine = rng.permutation(pages_per_seq) if not shared or row % 3 == 2 else first
        blocks.append(mine[:_TOPK])
    blocks = np.stack(blocks) * dcp + rank
    scores = rng.integers(-3, 4, (tokens, _TOPK)).astype(np.float32)
    short = rng.random((tokens, _TOPK)) < 0.15
    scores[short] = -np.inf
    blocks[short] = -1
    return msa_dcp.pack_candidates(jnp.asarray(scores), jnp.asarray(blocks))


def _attend_case(tokens, rank, shared, seed, pages_per_seq=128):
    rng = np.random.default_rng(seed)
    seqs, dim, dcp, heads = 8, 128, 2, 16
    if tokens == 48:  # a request of 5 rows, a padding tile
        q_lens = np.array([8, 8, 5, 8, 8, 0, 0, 0])
    else:
        q_lens = np.array([8] * (tokens // 8) + [0] * (seqs - tokens // 8))
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]), jnp.int32)
    full = dcp * _BLOCK * pages_per_seq
    seq_lens = jnp.asarray(
        np.where(q_lens > 0, rng.integers(2000, full, seqs), 0), jnp.int32
    )
    pages = seqs * pages_per_seq + 8
    block_tables = jnp.asarray(
        rng.permutation(pages)[: seqs * pages_per_seq], jnp.int32
    )
    candidates = jnp.stack(
        [
            _rank_candidates(rng, tokens, pages_per_seq, r, dcp, shared)
            for r in (rank, 1 - rank)
        ]
    )
    kv_cache = jnp.asarray(
        rng.integers(-2, 3, (pages, 32, 4, 2 * dim)), jnp.float8_e4m3fn
    )
    q = jnp.asarray(rng.standard_normal((heads, tokens, dim)), jnp.bfloat16)
    common = dict(
        sm_scale=dim**-0.5,
        topk=_TOPK,
        rank=rank,
        dcp=dcp,
        q_head_major=True,
        interpret=True,
    )
    return (q, kv_cache, candidates, seq_lens, block_tables, cu), common


# One tile (8 tokens; 4 padded to it) and several (16 to `_PAIR_WORK_MAX_TOKENS`);
# a one-vreg table (16 x 40 entries) and the served one (16 x 1,024: 8 vregs).
@pytest.mark.parametrize("tokens", [4, 8, 16, 48, 64])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("pages_per_seq", [40, 1024])
def test_pair_work_matches_xla(tokens, shared, pages_per_seq):
    """`pair_work`'s kept page keys and work list row equal
    `kept_own_candidates`' keys and `_packed_work` of them, bit for bit; with
    the packed page table the row goes on with the unions' physical pages."""
    rng = np.random.default_rng(tokens + shared + pages_per_seq)
    seqs = 16
    candidates = [
        _rank_candidates(rng, tokens, pages_per_seq, r, 2, shared) for r in (0, 1)
    ]
    # Page ids up to 2**16 - 1 (both 16-bit halves of a word, high bit set);
    # two logical blocks share a physical page (a shared prefix): the union
    # stays one of keys.
    num_pages = 1 << 16
    table = rng.permutation(num_pages)[: seqs * pages_per_seq].astype(np.int32)
    table[pages_per_seq + 3] = table[3]
    table = jnp.asarray(table)
    seq_id = jnp.asarray(np.minimum(np.arange(tokens) // 8, seqs - 1), jnp.int32)
    valid = jnp.asarray(rng.random(tokens) < 0.9)
    blocks, kept = msa_dcp.kept_own_candidates(jnp.stack(candidates), _TOPK)
    own_keys = seq_id[:, None] * pages_per_seq + jnp.maximum(blocks, 0) // 2
    want_keys = jnp.where(kept & valid[:, None], own_keys, -1)
    pages_per_step = msa_dcp._small_pages_per_step(tokens)
    padded = jnp.pad(want_keys, ((0, -tokens % 8), (0, 0)), constant_values=-1)
    want_work, _ = msa_attention_lse._packed_work(
        padded[None],
        8,
        pages_per_step,
        align=msa_attention_lse._merged_store_rows(8, 16, jnp.bfloat16),
        in_kernel=padded.shape[0] > 8,
    )
    unions = want_work[:, : padded.shape[0] // 8 * 128]
    want_pages = jnp.where(unions >= 0, table[jnp.maximum(unions, 0)], -1)
    key_base = jnp.where(valid, seq_id * pages_per_seq, -1)[:, None]
    common = dict(pages_per_step=pages_per_step, interpret=True)
    keys, work = msa_dcp.pair_work(*candidates, key_base, **common)
    np.testing.assert_array_equal(np.asarray(keys), np.asarray(want_keys))
    np.testing.assert_array_equal(np.asarray(work), np.asarray(want_work))
    keys, work = msa_dcp.pair_work(
        *candidates,
        key_base,
        page_table=msa_attention.pack_table16(table, num_pages),
        **common,
    )
    np.testing.assert_array_equal(np.asarray(keys), np.asarray(want_keys))
    np.testing.assert_array_equal(
        np.asarray(work), np.concatenate([want_work, want_pages], axis=1)
    )


# Tokens 8 (one tile), 24 (three tiles, one per step), 48 (a request of 5 rows
# and a padding tile: tiles of several requests and of none); one tile also
# with rows of 40 pages.
@pytest.mark.parametrize(
    "tokens, pages_per_seq", [(8, 128), (8, 40), (24, 128), (48, 128)]
)
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("shared", [False, True])
def test_attention_of_kept_own_candidates(tokens, pages_per_seq, rank, shared):
    """A decode step attends the pages of this rank's kept candidates
    (`own_first`): the same outputs, bit for bit, as over its merged
    selection."""
    args, common = _attend_case(
        tokens,
        rank,
        shared,
        seed=tokens * 4 + rank * 2 + shared,
        pages_per_seq=pages_per_seq,
    )
    want = msa_dcp.msa_attend_dcp(*args, **common)
    got = msa_dcp.msa_attend_dcp(*args, own_first=True, **common)
    for w, g in zip(want, got):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))


@pytest.mark.parametrize("tokens", [8, 24, 48])
@pytest.mark.parametrize("rank", [0, 1])
def test_attention_of_word_pages(tokens, rank, monkeypatch):
    """A decode step's fp8 pages read as 32-bit words (`_words_ring`) give the
    outputs of the packed-rows read, bit for bit. In interpret mode the host
    computes the bf16 x fp8 products: this checks the slot order, masks and
    hit tests, not the MXU's fp8 weights (on device: kbench j-a248f63b)."""
    args, common = _attend_case(tokens, rank, True, seed=tokens + 7 * rank)
    assert msa_attention_lse._words_ring(args[1], 8, False, 256)
    got = msa_dcp.msa_attend_dcp(*args, own_first=True, **common)
    monkeypatch.setattr(msa_attention_lse, "_words_ring", lambda *_: False)
    jax.clear_caches()
    want = msa_dcp.msa_attend_dcp(*args, own_first=True, **common)
    jax.clear_caches()
    for w, g in zip(want, got):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))


@pytest.mark.parametrize("tokens", [8, 24, 48])
@pytest.mark.parametrize("rank", [0, 1])
def test_attention_of_query_halves(tokens, rank):
    """A pair's query heads passed as (this rank's, the other rank's) halves
    give the outputs of the rank-ordered heads, bit for bit."""
    args, common = _attend_case(tokens, rank, True, seed=tokens + rank)
    q, *rest = args
    half = q.shape[0] // 2
    own, peer = q[rank * half : (rank + 1) * half], q[(1 - rank) * half :][:half]
    want = msa_dcp.msa_attend_dcp(*args, own_first=True, **common)
    got = msa_dcp.msa_attend_dcp((own, peer), *rest, own_first=True, **common)
    for w, g in zip(want, got):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))


def test_attention_of_kept_own_candidates_other_topk():
    """A topk outside `pair_work`'s envelope (its tile's keys fill the lanes
    at topk 16) attends through the XLA work list, as over the merge."""
    args, common = _attend_case(8, 0, True, seed=5)
    q, kv_cache, candidates, *meta = args
    topk = 8
    candidates = jnp.concatenate(
        [candidates[..., :topk], candidates[..., _TOPK : _TOPK + topk]], axis=-1
    )
    common = dict(common, topk=topk)
    want = msa_dcp.msa_attend_dcp(q, kv_cache, candidates, *meta, **common)
    got = msa_dcp.msa_attend_dcp(
        q, kv_cache, candidates, *meta, own_first=True, **common
    )
    for w, g in zip(want, got):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))


def test_kept_own_candidates_comparison_catches_a_changed_selection():
    """Positive control for the test above: one candidate that loses its
    place in the global top-k changes the outputs it compares."""
    args, common = _attend_case(16, 0, True, seed=3)
    q, kv_cache, candidates, *meta = args
    want = msa_dcp.msa_attend_dcp(*args, own_first=True, **common)
    # Row 0's best own candidate drops below every real one.
    best = int(jnp.argmax(candidates[0, 0, :_TOPK]))
    fault = candidates.at[0, 0, best].set(-1e9)
    got = msa_dcp.msa_attend_dcp(q, kv_cache, fault, *meta, own_first=True, **common)
    assert not np.array_equal(np.asarray(got[0]), np.asarray(want[0]))


def _pack(out, lse, block):
    """Reference `merge_pair` pack of head block `block` of a head-major
    partial (``out [2 * Hl, T, D]`` 16-bit, ``lse [2 * Hl, T]`` f32): its
    heads' bits, then a row of each token's LSE bits, high halves on lanes
    ``[0, Hl)``, low halves on ``[Hl, 2 Hl)``, zeros after."""
    heads, dim = out.shape[0] // 2, out.shape[2]
    rows = slice(block * heads, (block + 1) * heads)
    words = np.asarray(jax.lax.bitcast_convert_type(out[rows], jnp.uint16))
    bits = np.asarray(lse, np.float32)[rows].T.view(np.uint32)  # [T, Hl]
    halves = np.zeros((bits.shape[0], dim), np.uint16)
    halves[:, :heads] = bits >> 16
    halves[:, heads : 2 * heads] = bits & 0xFFFF
    return np.concatenate([words, halves[None]])


# Heads per rank 8 as served, 4 (an LSE block of fewer than 8 rows).
@pytest.mark.parametrize("tokens, heads", [(8, 8), (48, 8), (384, 8), (24, 4)])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("rotated", [False, True])
def test_merge_pair_matches_merge_partials(tokens, heads, rank, rotated):
    """A pair's merge from this rank's and the other rank's packs of this
    rank's heads equals `merge_partials` of both ranks' blocks in rank order,
    bit for bit, rows no rank attended included."""
    rng = np.random.default_rng(tokens + rank + 2 * rotated)
    dim = 128
    outs, lses = [], []
    for _ in range(2):
        lse = rng.standard_normal((2 * heads, tokens)).astype(np.float32) * 4
        lse[:, ::7] = -np.inf  # this rank attended nothing for these tokens
        out = rng.standard_normal((2 * heads, tokens, dim)).astype(np.float32)
        out[:, ::7] = 0
        outs.append(jnp.asarray(out, jnp.bfloat16))
        lses.append(jnp.asarray(lse))
    lses[1] = lses[1].at[:, ::14].set(-np.inf)  # neither rank: the output is 0
    hadamard = (
        jnp.asarray(rng.choice([-1.0, 1.0], (dim, dim)), jnp.bfloat16)
        if rotated
        else None
    )
    block = slice(rank * heads, (rank + 1) * heads)
    parts = jnp.stack([o[block] for o in outs])
    stacked = jnp.stack([x[block].T for x in lses])
    want = msa_dcp.merge_partials(parts, stacked, hadamard, interpret=True)
    got = msa_dcp.merge_pair(
        jnp.asarray(_pack(outs[rank], lses[rank], rank)),
        jnp.asarray(_pack(outs[1 - rank], lses[1 - rank], rank)),
        jnp.asarray([rank], jnp.int32),
        hadamard,
        dtype=jnp.bfloat16,
        interpret=True,
    )
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def _halves(q, rank):
    half = q.shape[0] // 2
    return q[rank * half : (rank + 1) * half], q[(1 - rank) * half :][:half]


# One tile, 3 tiles per grid step, two steps (48: a request of 5 rows, a
# padding tile).
@pytest.mark.parametrize("tokens", [8, 24, 48])
@pytest.mark.parametrize("rank", [0, 1])
def test_attention_pair_packs_are_the_partial(tokens, rank):
    """`pair_packs` writes this rank's and the other rank's packs (`_pack`)
    of the partial and LSEs the plain call returns, bit for bit."""
    args, common = _attend_case(tokens, rank, False, seed=tokens + 3 * rank)
    q, *rest = args
    out, lse = msa_dcp.msa_attend_dcp(*args, own_first=True, **common)
    got = msa_dcp.msa_attend_dcp(
        _halves(q, rank), *rest, own_first=True, pair_packs=True, **common
    )
    assert np.isfinite(np.asarray(lse)).any()
    for g, block in zip(got, (rank, 1 - rank)):
        assert g.dtype == jnp.uint16
        np.testing.assert_array_equal(np.asarray(g), _pack(out, lse, block))


@pytest.mark.parametrize("tokens", [8, 24])
@pytest.mark.parametrize("rank", [0, 1])
def test_attention_pair_packs_of_a_4bit_cache(tokens, rank):
    """`pair_packs` over a 4-bit cache (`attend` with scale pages, not the
    words ring) writes the packs of the plain call's partial."""
    args, common = _attend_case(tokens, rank, True, seed=tokens + rank)
    q, kv_cache, *meta = args
    rng = np.random.default_rng(tokens)
    pages, rows, packing, width = kv_cache.shape
    kv_cache = jnp.asarray(
        rng.integers(0, 256, (pages, rows * packing // 8, 4, width)), jnp.uint8
    )
    scale_cache = jnp.asarray(
        rng.uniform(0.01, 0.1, (pages, msa_fp4.SCALE_ROWS, msa_fp4.SCALE_LANES)),
        jnp.float32,
    )
    common = dict(common, scale_cache=scale_cache)
    out, lse = msa_dcp.msa_attend_dcp(q, kv_cache, *meta, own_first=True, **common)
    got = msa_dcp.msa_attend_dcp(
        _halves(q, rank), kv_cache, *meta, own_first=True, pair_packs=True, **common
    )
    for g, block in zip(got, (rank, 1 - rank)):
        np.testing.assert_array_equal(np.asarray(g), _pack(out, lse, block))


@pytest.mark.parametrize("tokens", [8, 16, 24, 48])
def test_pair_merge_of_packs_matches_merge_partials(tokens):
    """Both ranks of a pair, the permute simulated: each rank's merge of its
    own pack and the other rank's equals `merge_partials` of the two ranks'
    plain partials of its heads, bit for bit."""
    args, common = _attend_case(tokens, 0, True, seed=tokens)
    q, kv_cache, candidates, *meta = args
    rng = np.random.default_rng(tokens + 1)
    caches = (
        kv_cache,
        jnp.asarray(rng.integers(-2, 3, kv_cache.shape), kv_cache.dtype),
    )
    plain, packs = [], []
    for r in (0, 1):
        rank_args = (caches[r], candidates[::-1] if r else candidates, *meta)
        rank_common = dict(common, rank=r)
        plain.append(
            msa_dcp.msa_attend_dcp(q, *rank_args, own_first=True, **rank_common)
        )
        packs.append(
            msa_dcp.msa_attend_dcp(
                _halves(q, r),
                *rank_args,
                own_first=True,
                pair_packs=True,
                **rank_common,
            )
        )
    heads = q.shape[0] // 2
    for r in (0, 1):
        block = slice(r * heads, (r + 1) * heads)
        want = msa_dcp.merge_partials(
            jnp.stack([out[block] for out, _ in plain]),
            jnp.stack([lse[block].T for _, lse in plain]),
            interpret=True,
        )
        got = msa_dcp.merge_pair(
            packs[r][0],
            packs[1 - r][1],
            jnp.asarray([r], jnp.int32),
            dtype=q.dtype,
            interpret=True,
        )
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
