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

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import msa, msa_dcp, msa_indexer, msa_write

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


# Transposed picks (up to 768 columns) and row-major ones.
@pytest.mark.parametrize("width", [256, 1024])
@pytest.mark.parametrize("rank", [0, 1])
def test_packed_pick_matches_local_candidates(width: int, rank: int) -> None:
    rng = np.random.default_rng(width + rank)
    tokens = 24
    # Few distinct values, so most picks break ties.
    scores = jnp.asarray(rng.integers(-3, 4, (tokens, width)), jnp.float32)
    pos = jnp.asarray(rng.integers(0, 2 * _BLOCK * width, tokens), jnp.int32)
    want = msa_dcp.pack_candidates(*_pick(scores[:, None, :], pos, rank))
    pick = msa_dcp._PackedPick(_BLOCK, _TOPK, 1, 2, 2)
    got = pick(scores, pos[:, None], jnp.asarray(rank, jnp.int32))
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


# One page group, and two (the second partly live).
@pytest.mark.parametrize("pages_per_seq", [8, 160])
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
        [_BLOCK * (full // _BLOCK // 3) + 3, 300, _BLOCK * (full // _BLOCK - 2) + 3, full],
        jnp.int32,
    )
    block_tables = jnp.asarray(
        rng.permutation(pages)[: seqs * pages_per_seq], jnp.int32
    )
    bf16 = functools.partial(lambda *s: jnp.asarray(rng.standard_normal(s), jnp.bfloat16))
    k, v, index_q = bf16(tokens, 1, dim), bf16(tokens, 1, dim), bf16(tokens, 1, dim)
    index_k = bf16(tokens, dim)
    fp8 = jnp.float8_e4m3fn
    kv_cache = jnp.asarray(rng.integers(-2, 3, (pages, 32, 4, 2 * dim)), fp8)
    # Stale keys far above the new ones: a page scored before its write lands
    # scores higher than written.
    index_cache = jnp.asarray(rng.integers(-2, 3, (pages, 32, 4, dim)) * 64, fp8)
    meta = (seq_lens, block_tables, cu, jnp.asarray([3, 3, 3], jnp.int32))
    got = msa_dcp.msa_write_and_candidates_dcp(
        k, v, index_q, index_k, kv_cache, index_cache, *meta, topk=_TOPK,
        init_blocks=0, local_blocks=1, rank=rank, dcp=dcp, interpret=True,
    )

    seq_id, pos, valid = msa.token_positions(tokens, seq_lens, cu)
    page = msa_dcp.owner_pages(
        block_tables.reshape(seqs, -1), seq_id, pos, valid,
        block_size=_BLOCK, rank=rank, dcp=dcp,
    )
    kv_rows = jnp.concatenate([k, v], axis=-1).reshape(tokens, -1)
    want_kv, want_index = msa.write_caches(
        [msa_write.Rows(kv_cache, kv_rows), msa_write.Rows(index_cache, index_k)],
        page, pos, max_num_seqs=seqs, interpret=True,
    )
    local_lens = msa_dcp.local_seq_lens(seq_lens, block_size=_BLOCK, rank=rank, dcp=dcp)
    rows = msa_indexer.block_scores(
        index_q, want_index, local_lens, block_tables, cu, meta[3][2],
        token_rows=True, **msa.block_score_options(), interpret=True,
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
    bf16 = functools.partial(lambda *s: jnp.asarray(rng.standard_normal(s), jnp.bfloat16))
    fp8 = jnp.float8_e4m3fn
    got = msa_dcp.msa_write_and_candidates_dcp(
        bf16(tokens, 1, dim), bf16(tokens, 1, dim), bf16(tokens, 1, dim),
        bf16(tokens, dim),
        jnp.asarray(rng.integers(-2, 3, (16, 32, 4, 2 * dim)), fp8),
        jnp.asarray(rng.integers(-2, 3, (16, 32, 4, dim)), fp8),
        jnp.ones((seqs,), jnp.int32), jnp.zeros((seqs * 8,), jnp.int32), cu,
        jnp.asarray([tokens] * 3, jnp.int32), topk=_TOPK, init_blocks=0,
        local_blocks=1, rank=1, dcp=2, interpret=True,
    )[2]
    np.testing.assert_array_equal(np.asarray(got[:, :_TOPK]), -np.inf)
    np.testing.assert_array_equal(np.asarray(got[:, _TOPK:]), -1)


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
