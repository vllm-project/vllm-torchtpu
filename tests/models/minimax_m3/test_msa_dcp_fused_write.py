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
"""A decode step's cache write fused into its block scores
(`msa_indexer.block_scores_and_write`) against the write kernel followed by
the block scores (interpret mode)."""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.models.minimax_m3.packing import pack
from vllm_torchtpu.models.minimax_m3.kernels import msa, msa_dcp, msa_write

PAGE = 128
HEAD_DIM = 128
TOPK = 16
DCP = 2


def _separate(k, v, iq, ik, kv, index, seq_lens, block_tables, cu_q, dist, **opts):
    """The step as two kernels: `msa_write.write_runs`, then the scores."""
    num_tokens = k.shape[0]
    max_num_seqs = seq_lens.shape[0]
    seq_id, pos, valid = msa.token_positions(num_tokens, seq_lens, cu_q)
    pages = msa_dcp.owner_pages(
        block_tables.reshape(max_num_seqs, -1),
        seq_id,
        pos,
        valid,
        block_size=PAGE,
        rank=opts["rank"],
        dcp=DCP,
    )
    rows = jnp.concatenate([k, v], axis=-1).reshape(num_tokens, -1)
    kv, index = msa.write_caches(
        [msa_write.Rows(kv, rows), msa_write.Rows(index, ik)],
        pages,
        pos,
        max_num_seqs=max_num_seqs,
        interpret=True,
    )
    local_lens = msa_dcp.local_seq_lens(
        seq_lens, block_size=PAGE, rank=opts["rank"], dcp=DCP
    )
    scores = msa._block_scores(iq, index, local_lens, block_tables, cu_q, dist[2], True)
    picked = msa_dcp.local_candidates(
        scores, pos, block_size=PAGE, dcp=DCP, interpret=True, **opts
    )
    return kv, index, msa_dcp.pack_candidates(*picked)


@pytest.mark.parametrize("dtype", [jnp.float8_e4m3fn, jnp.bfloat16])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("local_blocks", [0, 1])
def test_fused_write_matches_separate_kernels(dtype, rank, local_blocks):
    rng = np.random.default_rng(10 * rank + local_blocks)
    # Verify groups of 8 rows (two straddle a block boundary, one opens a
    # block) and a short request; one padding token.
    q_lens = [8, 8, 8, 5]
    ctx_lens = [3 * PAGE - 3, 6 * PAGE, 9 * PAGE + 50, 2 * PAGE + 125]
    num_seqs = len(q_lens)
    max_num_seqs = num_seqs + 1
    num_real = sum(q_lens)
    num_tokens = num_real + 3
    pages_per_seq = 8
    num_pages = max_num_seqs * pages_per_seq + 3
    block_tables = rng.permutation(num_pages)[: max_num_seqs * pages_per_seq]
    seq_lens = np.zeros(max_num_seqs, np.int32)
    seq_lens[:num_seqs] = np.array(ctx_lens) + np.array(q_lens)
    cu_q = np.full(max_num_seqs + 1, num_real, np.int32)
    cu_q[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    dist = np.array([num_seqs, num_seqs, num_seqs], np.int32)
    kv = pack(rng.standard_normal((num_pages, PAGE, 2 * HEAD_DIM)), dtype)
    index = pack(rng.standard_normal((num_pages, PAGE, HEAD_DIM)), dtype)
    k, v, iq = (
        jnp.asarray(rng.standard_normal((num_tokens, 1, HEAD_DIM)), jnp.bfloat16)
        for _ in range(3)
    )
    # Large new index keys, so the rewritten pages win the selection.
    ik = jnp.asarray(8 * rng.standard_normal((num_tokens, HEAD_DIM)), jnp.bfloat16)
    meta = tuple(
        jnp.asarray(a, jnp.int32) for a in (seq_lens, block_tables, cu_q, dist)
    )
    opts = dict(topk=TOPK, init_blocks=0, local_blocks=local_blocks, rank=rank)
    want = _separate(k, v, iq, ik, kv, index, *meta, **opts)
    got = msa_dcp.msa_write_and_candidates_dcp(
        k, v, iq, ik, kv, index, *meta, dcp=DCP, interpret=True, **opts
    )
    for g, w in zip(got[:2], want[:2]):
        np.testing.assert_array_equal(
            np.asarray(g).view(np.uint8), np.asarray(w).view(np.uint8)
        )
    np.testing.assert_array_equal(
        np.asarray(got[2])[:num_real], np.asarray(want[2])[:num_real]
    )
