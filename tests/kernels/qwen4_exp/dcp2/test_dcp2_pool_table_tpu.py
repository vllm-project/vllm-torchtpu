# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_DCP_POOL_PAGE_TABLE on TPU (DCP2_HARNESS_DEVICE=tpu; skipped elsewhere).

The compressed-DCP owned-group decode indexer with the pool-page table (the
KV reader expands each owned block into its kernel pages) against the
kernel-page table, on every rank of a DCP group of 4, both with
VLLM_TPU_QSA_COMPRESSED_DCP_LOCAL_MERGE: the select step writes the same pool
bytes and returns the same local lists, and the attend step the same output and
LSE, bit for bit. Inputs as test_dcp2_local_merge_tpu.py.

Run it in the served decode env, e.g. through the bench recipe:
``BENCH=script "BENCH_CMD=env DCP2_HARNESS_DEVICE=tpu python -m pytest -q -s
tests/kernels/qwen4_exp/dcp2/test_dcp2_pool_table_tpu.py"``.
"""

import functools

import dcp2_harness
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_dcp2_op
from dcp2_harness import ON_TPU, from_raw, poison_tail_rows, project_rank_pool, rank_operand, raw
from test_dcp2_local_merge_tpu import (
    CONTEXTS,
    DCP,
    K_SCALE,
    QUERY_LENS,
    RATIO,
    SERVED_DECODE_ENV,
    TOPK,
    V_SCALE,
    _bits,
    _zero_most_keys,
)
from test_dcp2_op import ROTARY, _args, make_step

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa

pytestmark = pytest.mark.skipif(not ON_TPU, reason="DCP2_HARNESS_DEVICE=tpu only")

CASES = {
    # variant, main_slot, query windows of the kernel-page / pool-page leg
    "random-s0-w1": ("random", 0, (1, 1)),
    "zero_keys-s1-w2": ("zero_keys", 1, (2, 2)),
    # The served change: 128-row windows on the kernel-page table, one
    # window on the pool-page table.
    "random-s1-w2v1": ("random", 1, (2, 1)),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_tpu_pool_page_table_scores_the_same(monkeypatch, case):
    variant, main_slot, windows = CASES[case]
    monkeypatch.setattr(dcp2_harness, "D", DCP)
    monkeypatch.setattr(test_dcp2_op, "D", DCP)
    for name, value in SERVED_DECODE_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("VLLM_TPU_QSA_COMPRESSED_DCP_LOCAL_MERGE", "1")
    chunk_size = pooled_qsa.decode_query_chunk_size
    layout, full, keep, qgate, s, m, seq, starts = make_step(
        21 + windows[0] + 2 * (windows[0] != windows[1]), QUERY_LENS, CONTEXTS)
    if variant == "zero_keys":
        full = _zero_most_keys(full, m["compressed_table"], 7)
    pools = [
        poison_tail_rows(project_rank_pool(raw(full), layout, r, keep_pages=keep), layout, r, seq)
        for r in range(DCP)
    ]
    pair_qgate = jnp.concatenate(qgate, axis=1)
    opts = dict(compress_ratio=RATIO, token_topk=TOPK, norm_eps=1e-6, rotary_dim=ROTARY,
                k_scale=K_SCALE, v_scale=V_SCALE, main_slot=main_slot, max_query_len=4)
    att = dict(kv_heads=1, compress_ratio=RATIO, token_topk=TOPK, k_scale=K_SCALE,
               v_scale=V_SCALE, main_slot=main_slot, max_query_len=4, dcp_size=DCP,
               local_merge=True)

    def run(flag):
        monkeypatch.setenv("VLLM_TPU_QSA_DCP_POOL_PAGE_TABLE", str(flag))
        # Two 32-row windows over the 60 rows, or the SMEM-derived window.
        monkeypatch.setattr(pooled_qsa, "decode_query_chunk_size",
                            (lambda pages: 32) if windows[flag] == 2 else chunk_size)
        select = jax.jit(functools.partial(
            pooled_qsa.pooled_qsa_compressed_dcp_select_step, dcp_size=DCP, **opts))
        attend = jax.jit(functools.partial(pooled_qsa.pooled_qsa_compressed_dcp_attend_step, **att))
        sel = [select(*_args(pair_qgate, s, from_raw(pools[r]), m, layout.dcp_table),
                      rank_operand(r)) for r in range(DCP)]
        vals_pair = jnp.stack([x[6] for x in sel])
        pos_pair = jnp.stack([x[7] for x in sel])
        out = []
        for r in range(DCP):
            pool, q, gate, lengths, req, st = sel[r][:6]
            out.append([np.asarray(x) for x in attend(
                q, gate, lengths, req, st, jnp.asarray(layout.dcp_table), m["context"], pool,
                vals_pair, pos_pair, rank_operand(r))])
        return [[np.asarray(x) for x in row] for row in sel], out

    sel0, out0 = run(0)
    sel1, out1 = run(1)
    failures = []
    for r in range(DCP):
        for i, name in ((0, "select pool"), (6, "vals"), (7, "pos")):
            if not np.array_equal(_bits(sel1[r][i]), _bits(sel0[r][i])):
                failures.append(f"rank {r}: {name} differs")
        for i, name in ((0, "pool"), (1, "out"), (2, "lse"), (3, "selected")):
            if not np.array_equal(_bits(out1[r][i]), _bits(out0[r][i])):
                failures.append(f"rank {r}: attend {name} differs")
    filled = [int(np.sum(sel0[r][7] != np.iinfo(np.int32).max)) for r in range(DCP)]
    print(case, {"rows": int(starts[-1]), "filled per rank": filled, "failures": failures})
    assert not failures, failures
