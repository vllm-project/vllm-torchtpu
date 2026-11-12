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
"""DCP candidates of a prefill-sized step, picked per indexer tile class,
against the token-order path (interpret mode)."""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.models.minimax_m3.packing import pack
from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp, msa_fp4, msa_indexer

PAGE = 128
HEAD_DIM = 128
TOPK = 16


def _caches(dtype, num_pages, rng, heads=1):
    if dtype == "int4":
        kv = msa_fp4.pack_rows(
            jnp.asarray(
                rng.integers(0, 16, (num_pages, PAGE, heads * 2 * HEAD_DIM)), jnp.uint8
            )
        )
        index = msa_fp4.pack_rows(
            jnp.asarray(rng.integers(0, 16, (num_pages, PAGE, HEAD_DIM)), jnp.uint8),
            jnp.asarray(rng.uniform(0.5, 1.5, (num_pages, HEAD_DIM)), jnp.float32),
        )
        scales = jnp.asarray(rng.uniform(0.5, 1.5, (num_pages, 8, PAGE)), jnp.float32)
        return kv, index, scales
    kv = pack(rng.standard_normal((num_pages, PAGE, heads * 2 * HEAD_DIM)), dtype)
    index = pack(rng.standard_normal((num_pages, PAGE, HEAD_DIM)), dtype)
    return kv, index, None


@pytest.mark.parametrize(
    "dtype,rank,dcp,heads",
    [
        (jnp.bfloat16, 0, 2, 1),
        (jnp.bfloat16, 1, 2, 1),
        (jnp.float8_e4m3fn, 0, 2, 1),
        (jnp.float8_e4m3fn, 1, 2, 1),
        ("int4", 0, 2, 1),
        ("int4", 1, 2, 1),
        # DCP 8 at 4 KV heads: the heads fold into the query rows.
        (jnp.float8_e4m3fn, 5, 8, 4),
        ("int4", 3, 8, 4),
    ],
)
def test_class_candidates_match_token_order(dtype, rank, dcp, heads, monkeypatch):
    rng = np.random.default_rng(rank)
    q_lens, ctx_lens = [8, 9, 300], [3000, 700, 2100]
    num_seqs = len(q_lens)
    max_num_seqs = num_seqs + 1
    num_real = sum(q_lens)
    num_tokens = num_real + 13
    assert msa_indexer.is_prefill_step(num_tokens * heads, max_num_seqs)
    pages_per_seq = 16
    num_pages = max_num_seqs * pages_per_seq + 1
    perm = rng.permutation(np.arange(1, num_pages))
    block_tables = perm[: max_num_seqs * pages_per_seq].astype(np.int32)
    seq_lens = np.zeros(max_num_seqs, np.int32)
    seq_lens[:num_seqs] = np.array(ctx_lens) + np.array(q_lens)
    cu_q = np.full(max_num_seqs + 1, num_real, np.int32)
    cu_q[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    dist = np.array([1, 2, num_seqs], np.int32)
    kv, index, scales = _caches(dtype, num_pages, rng, heads)
    x = [
        jnp.asarray(rng.standard_normal((num_tokens, heads, HEAD_DIM)), jnp.bfloat16)
        for _ in range(4)
    ]
    ik = jnp.asarray(rng.standard_normal((num_tokens, HEAD_DIM)), jnp.bfloat16)

    def run():
        return msa_dcp.msa_write_and_candidates_dcp(
            x[0],
            x[1],
            x[3],
            ik,
            kv,
            index,
            *(jnp.asarray(a) for a in (seq_lens, block_tables, cu_q, dist)),
            topk=TOPK,
            init_blocks=1,
            local_blocks=1,
            rank=rank,
            dcp=dcp,
            scale_cache=scales,
            interpret=True,
        )[-1]

    got = np.asarray(run())
    monkeypatch.setattr(msa_indexer, "is_prefill_step", lambda *_: False)
    want = np.asarray(run())
    np.testing.assert_array_equal(got[:num_real], want[:num_real])
