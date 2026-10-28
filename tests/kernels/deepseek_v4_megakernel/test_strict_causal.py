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
"""TPU_STREAMIDX_STRICT_CAUSAL in the megakernel's CSA indexer (interpret).

Row lengths must equal stock `streamindex_topk._effective_row_lengths` with
the same flag; top-k lists must be the SparseCore-contract top-k of the
kernel's own scores under those lengths (all -1 for a row whose bound is 0);
attention and caches must match the block references with the same flag.
Each check has a control with the other flag that must fail.
"""

import dataclasses
import os
import sys

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_cases as C  # noqa: E402
import attn_phase_harness as H  # noqa: E402
import test_step_impl as TSI  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk import (  # noqa: E402
    _effective_row_lengths,
)
from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_csa  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_phase as AP  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_ref  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import kernel_interface as KI  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_impl as SI  # noqa: E402

Q = 7
# seq 8: positions 1..7, strict bounds 0 0 1 1 1 1 2 (else 1 1 1 2 2 2 2);
# seq 1030: 256 / 257 rows, across the first page edge;
# seq 4103: 1024 / 1025 rows, a real top-1024.
SEQ = [8, 1030, 4103]


def stock_lens(seq_lens, q_len, strict):
    seq = jnp.asarray(seq_lens, jnp.int32)
    R = seq.shape[0]
    cu = jnp.arange(R + 1, dtype=jnp.int32) * q_len
    dist = jnp.array([0, R, R], jnp.int32)
    out = _effective_row_lengths(seq, cu, dist, R * q_len, 1 << 30, 4, strict_causal=strict)
    return np.asarray(out).reshape(R, q_len)


@pytest.mark.parametrize("q_len", [1, 7])
def test_visible_rows_match_stock(q_len):
    seq = np.arange(q_len, q_len + 40, dtype=np.int32)
    pos = seq[:, None] - q_len + np.arange(q_len)[None, :]
    for strict in (False, True):
        got = np.asarray(attn_csa.visible_rows(jnp.asarray(seq[:, None] // 4), jnp.asarray(pos),
                                               strict))
        np.testing.assert_array_equal(got, stock_lens(seq, q_len, strict))
        if q_len > 1:  # control (decode rows are the last row: both bounds agree)
            assert not np.array_equal(got, stock_lens(seq, q_len, not strict))
    if q_len > 1:
        assert (stock_lens(seq, q_len, True) == 0).any()


def test_kernel_config_reads_the_env(monkeypatch):
    cfg = TSI._cfg()
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, num_ranks=16))
    bucket = KI.Bucket(q=Q, rows=16, reqs=1)
    for value, want in (("1", True), ("0", False)):
        monkeypatch.setenv("TPU_STREAMIDX_STRICT_CAUSAL", value)
        geom = SI.kernel_config(cfg, bucket).geometry()
        assert geom.strict_causal is want and AP.dims_of(geom).strict_causal is want
    monkeypatch.delenv("TPU_STREAMIDX_STRICT_CAUSAL")
    assert SI.kernel_config(cfg, bucket).geometry().strict_causal is True


@pytest.mark.parametrize("strict", [False, True])
def test_standalone_indexer(strict):
    case = C.csa_case(np.random.default_rng(3), SEQ, heads=8)
    sc, rl = attn_csa.indexer_scores(
        case.q_idx, case.weights, case.idx_words, case.csa_table, case.win.seq_lens,
        max_len=max(SEQ), blk_pages=2, strict_causal=strict, interpret=True)
    ref, rl_ref = attn_ref.indexer_scores_reference(
        case.q_idx.astype(jnp.float32), case.weights, case.idx_words, case.csa_table,
        case.win.seq_lens, strict_causal=strict)
    np.testing.assert_array_equal(np.asarray(rl), stock_lens(SEQ, Q, strict))
    np.testing.assert_array_equal(np.asarray(rl_ref), stock_lens(SEQ, Q, strict))
    sc, ref = np.asarray(sc), np.asarray(ref)
    for r, s in enumerate(SEQ):
        a, b = sc[r, :, : s // 4], ref[r, :, : s // 4]
        fin = np.isfinite(b)
        assert np.array_equal(np.isfinite(a), fin)
        assert np.abs(a[fin] - b[fin]).max() <= 1e-6 * np.abs(b[fin]).max()
    assert not np.array_equal(np.asarray(rl), stock_lens(SEQ, Q, not strict))  # control


def _group_scores(debug_scores, r):
    """tc_topk group layout [S * 8, 128] of request r -> f32 [8, S * 128]."""
    g = np.asarray(debug_scores)[r]
    return g.reshape(-1, 8, 128).transpose(1, 0, 2).reshape(8, -1).view(np.float32)


def _assert_close(o, exp_o, R):
    o = o[:R, :Q * 128].reshape(R, Q, 128, 512)
    e = np.asarray(exp_o)[:R]
    err = np.abs(o - e)
    assert err.max() / np.abs(e).max() <= 1e-2 and err.mean() <= 1e-3, err.max()


@pytest.fixture(scope="module")
def csa_layer():
    rng = np.random.default_rng(21)
    ls = H.build(rng, ("hca", "csa"), SEQ, 8192)
    return ls, H.make_inputs(rng, ls, "csa")


@pytest.mark.parametrize("strict", [False, True])
def test_phase_topk_and_attention(csa_layer, strict):
    ls, inp = csa_layer
    caches, out, (tk, scores, _, _) = H.run_phase(ls, "csa", 1, inp, debug_topk=True,
                                               strict_causal=strict)
    tk = np.asarray(tk)[:, :Q]
    lens = stock_lens(SEQ, Q, strict)
    wrong = stock_lens(SEQ, Q, not strict)
    exp_c, exp_o, exp_tk = H.reference(ls, "csa", 1, inp, strict_causal=strict)
    exp_tk = np.asarray(exp_tk)
    differs = 0
    for r in range(len(SEQ)):
        sc = jnp.asarray(_group_scores(scores, r)[:Q])
        want = np.asarray(attn_ref.topk_reference(sc, jnp.asarray(lens[r]), AP.TOPK))
        np.testing.assert_array_equal(tk[r], want)
        alt = np.asarray(attn_ref.topk_reference(sc, jnp.asarray(wrong[r]), AP.TOPK))
        differs += int(np.sum(np.any(alt != tk[r], axis=-1)))
        for t in range(Q):
            if lens[r, t] <= AP.TOPK:  # every visible row selected: exact vs the reference
                np.testing.assert_array_equal(tk[r, t], exp_tk[r, t])
            if lens[r, t] == 0:
                assert (tk[r, t] == -1).all()
    assert differs > 0  # control: the other flag's lengths select differently
    if strict:
        assert (lens == 0).any()
    o = np.asarray(out, np.float32)
    assert np.isfinite(o).all()
    _assert_close(o, exp_o, len(SEQ))
    for c, x in zip(caches, exp_c):
        assert np.array_equal(np.asarray(c), x)
    _, bad_o, _ = H.reference(ls, "csa", 1, inp, strict_causal=not strict)
    with pytest.raises(AssertionError):  # control: the other flag's reference
        _assert_close(o, bad_o, len(SEQ))
