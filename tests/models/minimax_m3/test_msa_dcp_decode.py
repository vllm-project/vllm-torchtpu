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
"""DCP candidates of a decode step picked inside the block-score kernel
(`msa_dcp.decode_candidates`) against the scores + candidates-kernel path,
bit for bit (interpret mode)."""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.models.minimax_m3.packing import pack
from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp

PAGE = 128
HEAD_DIM = 128
TOPK = 16
PAGES_PER_SEQ = 300  # 3 groups of 128 pages, the last one partial


def _step(dtype, q_lens, ctx_lens, rank, init_blocks, seed, max_num_seqs=None):
    rng = np.random.default_rng(seed)
    num_seqs = len(q_lens)
    max_num_seqs = max_num_seqs or num_seqs + 1
    num_real = sum(q_lens)
    num_tokens = -(-num_real // 8) * 8 + 8  # padding rows past the last request
    num_pages = max_num_seqs * PAGES_PER_SEQ + 1
    block_tables = rng.permutation(np.arange(1, num_pages))[
        : max_num_seqs * PAGES_PER_SEQ
    ]
    seq_lens = np.zeros(max_num_seqs, np.int32)
    seq_lens[:num_seqs] = np.array(ctx_lens) + np.array(q_lens)
    cu_q = np.full(max_num_seqs + 1, num_real, np.int32)
    cu_q[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    meta = [
        jnp.asarray(a)
        for a in (
            seq_lens,
            block_tables.astype(np.int32),
            cu_q,
            np.array([0, num_seqs, num_seqs], np.int32),
        )
    ]
    kv = pack(rng.standard_normal((num_pages, PAGE, 2 * HEAD_DIM)), dtype)
    index = pack(rng.standard_normal((num_pages, PAGE, HEAD_DIM)), dtype)
    x = [
        jnp.asarray(rng.standard_normal((num_tokens, 1, HEAD_DIM)), jnp.bfloat16)
        for _ in range(3)
    ]
    ik = jnp.asarray(rng.standard_normal((num_tokens, HEAD_DIM)), jnp.bfloat16)

    def run():
        return msa_dcp.msa_write_and_candidates_dcp(
            *x,
            ik,
            kv,
            index,
            *meta,
            topk=TOPK,
            init_blocks=init_blocks,
            local_blocks=1,
            rank=rank,
            dcp=2,
            interpret=True,
        )

    return run, num_real


@pytest.mark.parametrize("dtype", [jnp.float8_e4m3fn, jnp.bfloat16])
@pytest.mark.parametrize(
    "q_lens,ctx_lens,rank,init_blocks",
    [
        ([8], [70_000], 0, 0),  # one verify window (the c4 decode step)
        ([8, 8, 8], [70_000, 5_000, 40_000], 1, 0),
        ([1, 5, 8], [3_000, 60_000, 200], 0, 1),  # unaligned starts share a tile
        ([20, 3], [30_000, 9_000], 1, 1),  # a request over three tiles
        # 12 verify windows, half of them across a block boundary: 18 runs
        ([8] * 12, [128 * i - 4 * (i % 2) for i in range(60, 72)], 0, 0),
    ],
)
def test_decode_candidates_match_the_scores_path(
    dtype, q_lens, ctx_lens, rank, init_blocks, monkeypatch
):
    run, num_real = _step(dtype, q_lens, ctx_lens, rank, init_blocks, seed=sum(q_lens))
    kv, index, got = run()
    monkeypatch.setattr(msa_dcp, "decode_candidates", lambda *a, **k: None)
    want_kv, want_index, want = run()
    got, want = np.asarray(got)[:num_real], np.asarray(want)[:num_real]
    assert got.shape == (num_real, 2 * TOPK)
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(np.asarray(kv), np.asarray(want_kv))
    np.testing.assert_array_equal(np.asarray(index), np.asarray(want_index))
    # Every row picked real candidates, not just the padding value.
    assert np.isfinite(got[:, :TOPK]).any(axis=1).all()


def test_runs_past_the_vmem_budget_take_the_plain_path(monkeypatch):
    """A step whose write buffers would pass `FUSED_WRITE_VMEM_BYTES` keeps
    `write_runs` + the scores path, with the same results."""
    run, num_real = _step(jnp.float8_e4m3fn, [8, 8], [50_000, 7_000], 0, 0, seed=3)
    want = [np.asarray(x) for x in run()]
    taken = []
    plain = msa_dcp.decode_candidates

    def spy(*args, **kwargs):
        taken.append(plain(*args, **kwargs))
        return taken[-1]

    monkeypatch.setattr(msa_dcp, "FUSED_WRITE_VMEM_BYTES", 0)
    monkeypatch.setattr(msa_dcp, "decode_candidates", spy)
    got = [np.asarray(x) for x in run()]
    assert taken == [None]
    for g, w in zip(got, want, strict=True):
        np.testing.assert_array_equal(g[:num_real], w[:num_real])
