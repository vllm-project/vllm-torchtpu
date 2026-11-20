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

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import msa, msa_dcp, msa_indexer

_TOPK = 16
_BLOCK = 128


def _pick(scores, pos, rank, **kwargs):
    return msa_dcp.local_candidates(
        scores,
        pos,
        block_size=_BLOCK,
        topk=_TOPK,
        init_blocks=1,
        local_blocks=2,
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
