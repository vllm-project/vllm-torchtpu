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
"""DCP candidates of a mixed (decode + prefill chunk) step, the large tiles
picked inside their indexer kernel, against the token-order path (interpret
mode), bit for bit: several large tiles of several page groups each, an
unaligned chunk start, padding rows, two large requests."""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.models.minimax_m3.test_msa_dcp_classes import HEAD_DIM, TOPK, _caches
from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp, msa_indexer


def _mixed_step(dtype, rank, dcp, heads, q_lens=None, ctx_lens=None):
    """A run of `msa_write_and_candidates_dcp` over a mixed step, and its
    real rows."""
    rng = np.random.default_rng(10 * dcp + rank)
    # Two decode requests, then a chunk of 3+ large tiles starting at row 21;
    # the chunk's context spans 2 page groups of 256 pages (3 of 128) on
    # every rank.
    q_lens = q_lens or [8, 13, 1100]
    ctx_lens = ctx_lens or [9_000, 300, 2 * 128 * 128 * dcp + 5_000]
    num_seqs = len(q_lens)
    max_num_seqs = 4
    num_real = sum(q_lens)
    num_tokens = num_real + 13
    assert msa_indexer.is_prefill_step(num_tokens * heads, max_num_seqs)
    pages_per_seq = 300
    num_pages = max_num_seqs * pages_per_seq + 1
    perm = rng.permutation(np.arange(1, num_pages))
    block_tables = perm[: max_num_seqs * pages_per_seq].astype(np.int32)
    seq_lens = np.zeros(max_num_seqs, np.int32)
    seq_lens[:num_seqs] = np.array(ctx_lens) + np.array(q_lens)
    cu_q = np.full(max_num_seqs + 1, num_real, np.int32)
    cu_q[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    dist = np.array([2, 0, num_seqs], np.int32)
    kv, index, scales = _caches(dtype, num_pages, rng, heads)
    x = [
        jnp.asarray(rng.standard_normal((num_tokens, heads, HEAD_DIM)), jnp.bfloat16)
        for _ in range(3)
    ]
    ik = jnp.asarray(rng.standard_normal((num_tokens, HEAD_DIM)), jnp.bfloat16)

    def run():
        return msa_dcp.msa_write_and_candidates_dcp(
            x[0],
            x[1],
            x[2],
            ik,
            kv,
            index,
            *(jnp.asarray(a) for a in (seq_lens, block_tables, cu_q, dist)),
            topk=TOPK,
            init_blocks=0,
            local_blocks=1,
            rank=rank,
            dcp=dcp,
            scale_cache=scales,
            interpret=True,
        )[-1]

    return run, num_real


@pytest.mark.parametrize(
    "dtype,rank,dcp,heads",
    [
        (jnp.float8_e4m3fn, 0, 2, 1),
        ("int4", 1, 2, 1),
        (jnp.bfloat16, 1, 2, 1),
        # DCP 4 at 2 KV heads (c64): the heads fold into the query rows.
        ("int4", 3, 4, 2),
        (jnp.float8_e4m3fn, 2, 4, 2),
    ],
)
def test_mixed_step_matches_token_order(dtype, rank, dcp, heads, monkeypatch):
    run, num_real = _mixed_step(dtype, rank, dcp, heads)
    got = np.asarray(run())
    monkeypatch.setattr(msa_indexer, "is_prefill_step", lambda *_: False)
    want = np.asarray(run())
    np.testing.assert_array_equal(got[:num_real], want[:num_real])
    # Padding rows hold the picks of position 0: block 0 on rank 0.
    blocks = got[num_real:, TOPK : 2 * TOPK]
    assert np.all((blocks == -1) | (blocks == 0))


def test_two_large_requests_match_token_order(monkeypatch):
    """A second large request with fewer pages than the first: its first
    tile refills the 4-bit pages' key cache, its later tiles read it."""
    run, num_real = _mixed_step(
        "int4", 1, 2, 1, q_lens=[8, 1100, 700], ctx_lens=[9_000, 70_000, 40_000]
    )
    got = np.asarray(run())
    monkeypatch.setattr(msa_indexer, "is_prefill_step", lambda *_: False)
    want = np.asarray(run())
    np.testing.assert_array_equal(got[:num_real], want[:num_real])


def test_over_vmem_budget_picks_in_their_own_kernel(monkeypatch):
    """A step past `_PREFILL_VMEM_BYTES` picks its large tiles in their own
    kernel and gathers the classes' picks: the same candidates."""
    run, num_real = _mixed_step("int4", 1, 4, 2)
    got = np.asarray(run())
    monkeypatch.setattr(msa_dcp, "_PREFILL_VMEM_BYTES", 0)
    want = np.asarray(run())
    np.testing.assert_array_equal(got[:num_real], want[:num_real])
