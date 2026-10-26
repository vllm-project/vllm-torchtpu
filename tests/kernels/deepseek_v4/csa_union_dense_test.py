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
"""Union-dense CSA attention vs the sparse kernel, and the SparseCore union
plan vs its numpy reference, bit for bit."""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.kernels.deepseek_v4.sparse_mla_test import create_cache
from vllm_torchtpu.kernels.deepseek_v4.core_attention import (
    csa_union_dense,
    sparse_mla,
)

pytestmark = pytest.mark.tpu_test

HEAD_DIM, PAGE_SIZE, TOPK = 512, 256, 1024


def _case(rng, seqs, heads, correlated):
    """seqs: (q_len, token context before this step). Correlated top-k: each
    token keeps ~75% of its predecessor's picks (served traffic: 0.68-0.85)."""
    q_lens = np.array([q for q, _ in seqs])
    cu = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    n = int(cu[-1])
    T = 1 << max(9, (n - 1).bit_length())
    # Served kv_len = seq_len // 4; rc7's indexer also selects a token's own
    # still-open block, index == kv_len (the regression behind the E2E halt).
    kv_lens = np.array([(c + q) // 4 for q, c in seqs], np.int32)
    max_seqs = max(4, len(seqs))
    pps = int(-(-(kv_lens.max() + 1) // PAGE_SIZE))
    total_pages = max_seqs * pps + 5
    pages = rng.permutation(total_pages)[: max_seqs * pps].astype(np.int32)
    topk = np.full((T, TOPK), -1, np.int32)
    row = 0
    for (q, c), kv in zip(seqs, kv_lens):
        prev = None
        for t in range(q):
            vis = int(min(kv + 1, (c + t) // 4 + 1))
            if vis <= TOPK:
                sel = np.arange(vis)
            elif correlated and prev is not None and prev.max() < vis:
                keep = rng.choice(prev, int(0.75 * TOPK), replace=False)
                rest = np.setdiff1d(np.arange(vis), keep)
                sel = np.concatenate([keep, rng.choice(rest, TOPK - keep.size, replace=False)])
            else:
                sel = rng.choice(vis, TOPK, replace=False)
            prev = sel
            topk[row, : sel.size] = sel
            row += 1
    _, nope, rope = create_cache(rng, total_pages, PAGE_SIZE, HEAD_DIM, jnp.bfloat16)
    f32 = np.float32
    return dict(
        q=jnp.asarray(rng.standard_normal((T, heads, HEAD_DIM), f32)).astype(jnp.bfloat16),
        nope=nope, rope=rope, topk=jnp.asarray(topk), topk_np=topk,
        pages=jnp.asarray(pages), pages_np=pages.reshape(max_seqs, pps),
        cu=jnp.asarray(np.concatenate([cu, np.full(max_seqs - len(seqs), cu[-1])]).astype(np.int32)),
        cu_np=cu,
        kv=jnp.asarray(np.concatenate([kv_lens, np.zeros(max_seqs - len(seqs), np.int32)])),
        kv_np=kv_lens,
        dist=jnp.asarray(np.array([0, 0, len(seqs)], np.int32)),
        sinks=jnp.asarray(rng.standard_normal((heads,), f32) + 4.0),
        swa_acc=jnp.asarray(rng.standard_normal((T, heads, HEAD_DIM), f32) * 0.5).astype(jnp.bfloat16),
        swa_l=jnp.asarray(rng.random((T, heads), f32) * 4 + 0.5),
        swa_m=jnp.asarray(rng.standard_normal((T, heads), f32)),
    )  # fmt: skip


def _sparse(c):
    return sparse_mla.sparse_ragged_paged_attention(
        c["q"], c["nope"], c["rope"], c["topk"], c["pages"], c["cu"], c["dist"],
        c["sinks"], c["swa_acc"], c["swa_l"], c["swa_m"], sm_scale=0.044,
        gather_and_attention_chunk_size=64, attention_kernel_batch_size=16,
    )  # fmt: skip


def _union(c, bq):
    return csa_union_dense.attention(
        c["q"], c["nope"], c["rope"], c["topk"], c["kv"], c["pages"], c["cu"],
        c["dist"], c["sinks"], c["swa_acc"], c["swa_l"], c["swa_m"], sm_scale=0.044,
        bq=bq,
    )  # fmt: skip


@pytest.mark.parametrize(
    "seqs,heads,bq",
    [
        ([(7, 40000), (1017, 22000)], 16, 64),  # verify window + chunk, correlated
        ([(3, 2000), (5, 9000), (300, 6000)], 16, 64),  # short chunk, small context
        ([(7, 30000), (7, 9000), (498, 31000)], 32, 32),  # TP4 shape
        ([(130, 0)], 16, 64),  # everything visible, -1 padded rows
        ([(2048, 22528)], 16, 64),  # kv_len 6144 (multiple of 16), own block picked
        ([(7, 9000), (1024, 250000)], 16, 64),  # kv_len 62.7k, near s_cap
        # sparse prefix of 70 rows (a deep seq past s_cap in front): 70 % 16 = 6,
        # the regression behind the served SparseCore halt (page ids unwritten
        # for rows the sparse batches round up to)
        ([(70, 300000), (954, 22000)], 16, 64),
    ],
)
def test_union_matches_sparse(seqs, heads, bq):
    c = _case(np.random.default_rng(0), seqs, heads, correlated=True)
    n = int(c["cu_np"][-1])
    ref = np.asarray(_sparse(c), np.float32)[:n]
    out = np.asarray(_union(c, bq), np.float32)[:n]
    # Dense rows differ from the sparse kernel only in summation order.
    np.testing.assert_allclose(out, ref, atol=8e-3, rtol=0)


@pytest.mark.parametrize("bq", [32, 64])
def test_plan_matches_numpy(bq):
    c = _case(np.random.default_rng(1), [(7, 40000), (1017, 22000)], 16, correlated=True)
    T = c["topk_np"].shape[0]
    pre = [1]
    usz = []
    for q0 in range(int(c["cu_np"][1]), int(c["cu_np"][2]), bq):
        blk = c["topk_np"][q0 : min(int(c["cu_np"][2]), q0 + bq)]
        usz.append(len(np.unique(blk[blk >= 0])))
    tiles = sum(-(-u // 1024) for u in usz)
    u_cap = 16384
    ref = csa_union_dense.plan_host(
        c["topk_np"], c["cu_np"], 2, c["pages_np"], PAGE_SIZE, bq=bq,
        rows_cap=-(-tiles * 1024 // 32768) * 32768, mask_words=u_cap // 32,
        prefill_seq_ids=pre,
    )  # fmt: skip
    meta = ref["meta"]
    nb = ref["num_blocks"]
    sel_len = c["kv_np"] + 1  # the open block is selectable (see _case)
    rows = np.stack([meta[0], meta[1], meta[5], sel_len[meta[5]] * (meta[1] > 0)], 1)
    idx, mask, usize = csa_union_dense.build_union_plan(
        c["topk"], c["pages"], jnp.asarray(rows.astype(np.int32)),
        max_num_seqs=4, s_cap=65536, u_cap=u_cap, page_size=PAGE_SIZE,
    )  # fmt: skip
    idx, mask, usize = np.asarray(idx), np.asarray(mask), np.asarray(usize)
    _, _, ucount = csa_union_dense.build_union_plan(
        c["topk"], c["pages"], jnp.asarray(rows.astype(np.int32)),
        max_num_seqs=4, s_cap=65536, u_cap=u_cap, page_size=PAGE_SIZE, count_only=True,
    )  # fmt: skip
    np.testing.assert_array_equal(np.asarray(ucount)[:nb], usize[:nb])
    for b in range(nb):
        q0, qn, t0, nt, u = (int(x) for x in meta[:5, b])
        assert usize[b] == u
        np.testing.assert_array_equal(
            idx[b * u_cap : b * u_cap + nt * 1024], ref["indices"][t0 * 1024 : (t0 + nt) * 1024]
        )
        np.testing.assert_array_equal(mask[q0 : q0 + qn], ref["mask"][q0 : q0 + qn])
    assert T >= int(c["cu_np"][-1])


@pytest.mark.parametrize(
    "seqs,routed",
    [
        ([(7, 9000), (498, 100000)], False),  # chunk past the cap: plan skipped, all sparse
        ([(7, 100000), (498, 31000)], True),  # deep verify row in front, short chunk routed
        ([(300, 100000), (200, 9000)], True),  # deep chunk -> sparse prefix of 300 rows
    ],
)
def test_union_max_kv_tp4(seqs, routed):
    """TPU_DSV4_CSA_UNION_DENSE_MAX_KV_TP4: sequences whose compressed KV
    exceeds the cap stay on the sparse path; output matches the sparse kernel."""
    c = _case(np.random.default_rng(2), seqs, 32, correlated=True)
    n = int(c["cu_np"][-1])
    plan = csa_union_dense.prepare(
        c["topk"], c["kv"], c["pages"], c["cu"], c["dist"], page_size=PAGE_SIZE,
        bq=32, count_first=False, split=2, max_kv=16384,
    )  # fmt: skip
    long_rows = sum(q for q, ctx in seqs if (ctx + q) // 4 + 1 > 16384 and q >= 17)
    n_sparse = int(plan["n_sparse"])
    assert (n_sparse < n) == routed  # some sequence routed to union-dense
    if not routed:
        assert not bool(plan["use_union"])
    assert n_sparse >= long_rows
    ref = np.asarray(_sparse(c), np.float32)[:n]
    out = csa_union_dense.attention(
        c["q"], c["nope"], c["rope"], c["topk"], c["kv"], c["pages"], c["cu"],
        c["dist"], c["sinks"], c["swa_acc"], c["swa_l"], c["swa_m"], sm_scale=0.044,
        bq=32, plan=plan,
    )  # fmt: skip
    np.testing.assert_allclose(np.asarray(out, np.float32)[:n], ref, atol=8e-3, rtol=0)


@pytest.mark.parametrize(
    "seqs",
    [
        [(7, 30000), (7, 9000), (498, 31000)],  # TP4 shape, routed
        [(7, 30000), (7, 9000), (7, 8000)],  # verify rows only: nothing routed
        [(7, 9000), (498, 300000)],  # chunk past s_cap: nothing routed
    ],
)
def test_union_plan_unconditional(seqs):
    """prepare(unconditional=True) (the TP4 plan outside lax.cond, with the
    indexer token split): with nothing routed the plan's buffers are left
    unwritten and must never be read; output matches the sparse kernel."""
    c = _case(np.random.default_rng(3), seqs, 32, correlated=True)
    n = int(c["cu_np"][-1])
    plan = csa_union_dense.prepare(
        c["topk"], c["kv"], c["pages"], c["cu"], c["dist"], page_size=PAGE_SIZE,
        bq=32, count_first=False, split=2, unconditional=True,
    )  # fmt: skip
    ref = np.asarray(_sparse(c), np.float32)[:n]
    out = csa_union_dense.attention(
        c["q"], c["nope"], c["rope"], c["topk"], c["kv"], c["pages"], c["cu"],
        c["dist"], c["sinks"], c["swa_acc"], c["swa_l"], c["swa_m"], sm_scale=0.044,
        bq=32, plan=plan,
    )  # fmt: skip
    np.testing.assert_allclose(np.asarray(out, np.float32)[:n], ref, atol=8e-3, rtol=0)
