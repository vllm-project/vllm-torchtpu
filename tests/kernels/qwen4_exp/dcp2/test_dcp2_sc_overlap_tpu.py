# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_DECODE_SC_OVERLAP on TPU (DCP2_HARNESS_DEVICE=tpu; skipped elsewhere).

One decode/verify step of ``pooled_qsa_dcp_step`` (rank 0 of a DCP pair, real
StreamIndex scoring, SparseCore top-k, cache writers and batched RPA) runs on
the same inputs with the flag off and on. The restructured program (main-KV
write after the scoring, no early-exit cond, unrolled windows) must write the
same pool bytes and select the same blocks; its attention output and LSE must
match the default program's.

Run it in the served decode env, e.g. through the bench recipe:
``BENCH=script "BENCH_CMD=env DCP2_HARNESS_DEVICE=tpu python -m pytest -q -s
tests/kernels/qwen4_exp/dcp2/test_dcp2_sc_overlap_tpu.py"``.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dcp2_harness import ON_TPU, from_raw, project_rank_pool, rank_operand, raw
from test_dcp2_op import DH, ROTARY, _args, make_step

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa

pytestmark = pytest.mark.skipif(not ON_TPU, reason="DCP2_HARNESS_DEVICE=tpu only")

RATIO = 4
TOPK = 2048  # tokens: the served 512 compressed groups
SEATS = 16
SERVED_DECODE_ENV = {
    "VLLM_TPU_QSA_DECODE_QUERY_BLOCK": "4",
    "VLLM_TPU_QSA_INDEXER_MIXED_ONLY": "1",
    "VLLM_TPU_QSA_INDEXER_SC_BOUNDED_COPY": "1",
}


def _contexts(seed):
    # Every seat past the 2048-token top-k, below the harness's 16384-row rope
    # table; one seat at the early-exit bound's side of it.
    rng = np.random.default_rng(seed)
    contexts = rng.integers(2100, 16000, SEATS - 1).tolist() + [1500]
    return [4] * SEATS, contexts


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("windows", [1, 2])
def test_tpu_sc_overlap_matches_default(monkeypatch, windows, mode):
    for k, v in SERVED_DECODE_ENV.items():
        monkeypatch.setenv(k, v)
    if windows == 2:
        # 64 rows in two 32-row windows (the 256-row bucket's shape at 128 each).
        monkeypatch.setattr(pooled_qsa, "decode_query_chunk_size", lambda pages: 32)
    query_lens, contexts = _contexts(windows)
    layout, full, keep, qgate, s, m, _, starts = make_step(7, query_lens, contexts)
    pool = from_raw(project_rank_pool(raw(full), layout, 0, keep_pages=keep))
    args = _args(jnp.concatenate(qgate, axis=1), s, pool, m, layout.dcp_table)
    opts = dict(compress_ratio=RATIO, token_topk=TOPK, norm_eps=1e-6, rotary_dim=ROTARY,
                k_scale=0.375, v_scale=2.5, max_query_len=4)

    def run(flag):
        monkeypatch.setenv("VLLM_TPU_QSA_DECODE_SC_OVERLAP", str(flag))
        step = jax.jit(functools.partial(pooled_qsa.pooled_qsa_dcp_step, **opts))
        return [np.asarray(x) for x in step(*args, rank_operand(0))]

    pool0, out0, lse0, sel0 = run(0)
    pool1, out1, lse1, sel1 = run(mode)
    tokens = int(starts[-1])
    assert np.array_equal(raw(pool1), raw(pool0)), "pool bytes differ"
    # Top-k order is unspecified: compare each row's selected token set.
    same_rows = np.all(np.sort(sel1, axis=1) == np.sort(sel0, axis=1), axis=1)
    out0, out1 = out0[:tokens].astype(np.float32), out1[:tokens].astype(np.float32)
    lse0, lse1 = lse0[:tokens], lse1[:tokens]
    finite = np.isfinite(lse0)
    scale = np.maximum(np.abs(out0).max(axis=-1, keepdims=True), 1e-6)
    print(
        f"windows {windows} mode {mode}: rows with identical selections "
        f"{int(same_rows.sum())}/{len(same_rows)}; out bit-equal "
        f"{float(np.mean(out1 == out0)):.6f}, max |d|/max|head| "
        f"{float(np.max(np.abs(out1 - out0) / scale)):.3g}; lse bit-equal "
        f"{float(np.mean(lse1[finite] == lse0[finite])):.6f}, max |d| "
        f"{float(np.max(np.abs(lse1[finite] - lse0[finite]), initial=0)):.3g}"
    )
    assert same_rows.all(), "selected tokens differ"
    np.testing.assert_array_equal(np.isfinite(lse1), finite)
    assert np.max(np.abs(out1 - out0) / scale) <= 4e-3
    assert np.max(np.abs(lse1[finite] - lse0[finite]), initial=0) <= 4e-3
    assert out1.shape[-1] == DH
