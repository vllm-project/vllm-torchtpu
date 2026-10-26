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
"""TPU tests for the union-of-window CSA path (TPU_DSV4_SMLA_UNION_VERIFY).

* L1 (`union_gather`, SparseCore) is bit-exact against the numpy plan and the
  `csa_gather` reference: union sizes, gathered nope/rope rows, keep masks.
* The whole `_attention_csa` path with union_verify on stays within the K1v2
  gate of the float64 oracle (max-rel <= max(2 x incumbent, incumbent + 2^-8)),
  on production-shaped cells (measured union distribution, strict-causal, shuffled
  pages) and the edge cells; poisoned unreferenced cache rows stay finite.
* Positive control: the same path with one window's union truncated by a tile
  must fail the gate.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.environ.get("K1V2_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "k1v2")))
import k1v2_common as C  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4.core_attention import (  # noqa: E402
    csa_window_union as WU,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4 import (  # noqa: E402
    deepseek_v4_attention_op as op,
)

pytestmark = pytest.mark.skipif(
    jax.devices()[0].platform != "tpu", reason="TPU (SparseCore) only"
)

L1_CELLS = ["c4T8-mid", "c8T32-ragged", "c16T32-last", "edge-tiny", "edge-pageb",
            "edge-maxb", "edge-maxb8"]
ACC_CELLS = ["c4T8-mid", "c8T16-mid", "c16T32-u90", "c16T32-last", "c8T32-last", "c16T8-last", "c8T32-ragged",
             "edge-short", "edge-tiny", "edge-pageb", "edge-maxb", "edge-sink", "edge-maxb8"]


def _rope_entries(rg, n):
    x = np.asarray(rg).view(np.uint32).reshape(-1, 128)
    rows = np.empty((x.shape[0] * 2, 128), np.uint32)
    rows[0::2], rows[1::2] = x & 0xFFFF, x >> 16
    t = rows.reshape(-1, 512, 128)
    return np.concatenate([t[:, :, :64], t[:, :, 64:]], axis=1).reshape(-1, 64)[:n]


@pytest.mark.parametrize("name", L1_CELLS)
def test_union_gather_bit_exact(name):
    inp = C.build_inputs(C.CELLS[name])
    d, _ = C.device_inputs(inp, cache="random")
    T, ms, s_cap = inp["T"], inp["ms"], inp["pps"] * C.PAGE
    plan = WU.plan_host(inp["tk"], inp["cu"], inp["dist"], inp["kv_lens"], inp["table"],
                        C.PAGE, s_cap=s_cap)
    meta = WU.window_meta(d["cu"], d["dist"], d["kv_lens"], num_tokens=T, s_cap=s_cap)
    ng, rg, mk, uc = WU.union_gather(meta, d["topk"], d["pages"], d["nope"], d["rope"],
                                     num_tokens=T, max_num_seqs=ms)
    rn, rr, _ = WU.reference_union_gather(d["nope"], d["rope"], plan, T)
    ng, mk, uc, rn = map(np.asarray, (ng, mk, uc, rn))
    re_, rr_ = _rope_entries(rg, T * WU.TILE), _rope_entries(rr, T * WU.TILE)
    assert int(np.asarray(meta)[0]) == len(plan["windows"])
    for w, (q0, qn, _, _) in enumerate(plan["windows"]):
        U = plan["U"][w]
        assert int(uc[w * 16]) == U
        r0 = q0 * WU.TILE
        np.testing.assert_array_equal(ng[r0:r0 + U], rn[r0:r0 + U])
        np.testing.assert_array_equal(re_[r0:r0 + U], rr_[r0:r0 + U])
        np.testing.assert_array_equal(mk[q0:q0 + qn], plan["mask"][q0:q0 + qn])


def _region(conc, union_verify, **extra):
    kw = C.kernel_tuning(conc)
    kw.update(union_verify=union_verify, **extra)

    def f(d, q):
        orig = op._run_swa
        op._run_swa = lambda **k: (d["swa_acc"], k["sw_cache"], d["swa_l"], d["swa_m"])
        try:
            dummy = jnp.zeros((1, 8, 128), jnp.int32)
            out, _ = op._attention_csa(
                q, jnp.zeros((q.shape[0], C.D), jnp.bfloat16), dummy, d["kv_lens"],
                d["pages"], d["cu"], d["dist"], d["nope"], d["kv_lens"], d["topk"],
                d["pages"], d["cu"], d["dist"], d["sinks"], d["rope"],
                sm_scale=C.SM_SCALE, sliding_window=128, logical_page_size=128,
                swa_only=False, two_caches_same_buffer=False, verify_q_len=C.VQ, **kw)
        finally:
            op._run_swa = orig
        return out
    return f


@pytest.mark.parametrize("poison", [False, True])
@pytest.mark.parametrize("name", ACC_CELLS)
def test_union_verify_matches_oracle(name, poison):
    cell = C.CELLS[name]
    inp = C.build_inputs(cell)
    d, deq = C.device_inputs(inp, cache="quant", poison=poison)
    want, rows = C.oracle(inp, d, deq)
    inc = np.asarray(jax.jit(_region(cell.conc, False))(d, d["q"]).astype(jnp.float32))
    cand = np.asarray(jax.jit(_region(cell.conc, True))(d, d["q"]).astype(jnp.float32))
    ref_err, _, _ = C.gate(inc[rows], want)
    err, ok, tol = C.gate(cand[rows], want, ref_err)
    assert np.isfinite(cand[rows]).all()
    assert ok, (name, err, tol, ref_err)


def test_positive_control_truncated_union_fails():
    """A candidate that drops each window's last union tile must fail the gate."""
    cell = C.CELLS["edge-pageb"]  # U = 1025 and 7168: the dropped tile carries real weight
    inp = C.build_inputs(cell)
    d, deq = C.device_inputs(inp, cache="quant")
    want, rows = C.oracle(inp, d, deq)
    inc = np.asarray(jax.jit(_region(cell.conc, False))(d, d["q"]).astype(jnp.float32))
    ref_err, _, _ = C.gate(inc[rows], want)
    orig = WU.union_gather

    def broken(*a, **k):
        ng, rg, mk, uc = orig(*a, **k)
        u = uc.reshape(-1, 16)
        u = jnp.where(u > WU.TILE, (u - 1) // WU.TILE * WU.TILE, u)
        return ng, rg, mk, u.reshape(-1)

    WU.union_gather = broken
    try:
        cand = np.asarray(jax.jit(_region(cell.conc, True))(d, d["q"]).astype(jnp.float32))
    finally:
        WU.union_gather = orig
    _, ok, _ = C.gate(cand[rows], want, ref_err)
    assert not ok


@pytest.mark.parametrize("name", ["c16T32-u90", "c8T16-mid"])
def test_positive_control_causal_leak_caught_by_poison(name):
    """Fault class `causal` (blind 4/6 in the oracle gate on production cells): a
    candidate that lets the last verify token attend its still-open compressed
    group reads a row no valid top-k entry references. With those rows poisoned
    (0xFF) the output must go non-finite or out of gate."""
    cell = C.CELLS[name]
    inp = C.build_inputs(cell)
    d, deq = C.device_inputs(inp, cache="quant", poison=True)
    want, rows = C.oracle(inp, d, deq)
    inc = np.asarray(jax.jit(_region(cell.conc, False))(d, d["q"]).astype(jnp.float32))
    ref_err, _, _ = C.gate(inc[rows], want)
    t6 = np.array([w * C.VQ + 6 for w in range(cell.R)])
    open_ids = jnp.asarray(((inp["pos"][t6] + 1) // C.RATIO).astype(np.int32))
    orig = WU.union_gather

    def broken(meta, topk, *a, **k):
        return orig(meta, topk.at[t6, C.TOPK - 1].set(open_ids), *a, **k)

    WU.union_gather = broken
    try:
        cand = np.asarray(jax.jit(_region(cell.conc, True))(d, d["q"]).astype(jnp.float32))
    finally:
        WU.union_gather = orig
    _, ok, _ = C.gate(cand[rows], want, ref_err)
    assert not (ok and np.isfinite(cand[rows]).all())


def test_qlen1_empty_window_and_pad_rows():
    """A 1-token request with no compressed rows (U = 0: the idle-replica slot shape)
    gets the SWA + sink result; rows outside every window return q bit-exactly."""
    cell = C.CELLS["c8T16-mid"]  # 2 verify windows in rows 0..13 of T = 16
    inp = C.build_inputs(cell)
    s = 2  # third sequence: one token at row 14, kv_len 0, top-k all -1
    inp["cu"][s + 1:] = inp["cu"][s] + 1
    inp["dist"] = np.array([0, cell.R, cell.R + 1], np.int32)
    inp["kv_lens"][s] = 0
    inp["seq_of"][14] = s
    inp["pos"][14] = 0
    inp["tk"][14] = -1
    inp["nseq"] = 3
    d, deq = C.device_inputs(inp, cache="quant")
    want, rows = C.oracle(inp, d, deq)
    assert 14 in rows and 15 not in rows
    inc = np.asarray(jax.jit(_region(cell.conc, False))(d, d["q"]).astype(jnp.float32))
    cand = np.asarray(jax.jit(_region(cell.conc, True))(d, d["q"]).astype(jnp.float32))
    ref_err, _, _ = C.gate(inc[rows], want)
    err, ok, tol = C.gate(cand[rows], want, ref_err)
    assert ok, (err, tol, ref_err)
    np.testing.assert_array_equal(cand[15], np.asarray(d["q"][15].astype(jnp.float32)))


def test_route_condition():
    assert WU.supported(128, 16) and WU.supported(32, 16) and WU.supported(8, 8)
    assert not WU.supported(128, 17)


def _eight_token_windows():
    """Two 8-token requests whose 8 rows select identical top-k lists: every union row has count 8 (0b1000
    in its nibble) — the L4 chunk shape of a short prefill tail in a decode bucket."""
    cell = C.CELLS["c8T16-mid"]
    inp = C.build_inputs(cell)
    tk = inp["tk"].copy()
    first1 = tk[7].copy()  # seq 1's first token list
    tk[0:8] = tk[0]
    tk[8:16] = first1
    inp["tk"] = tk
    inp["cu"][1] = 8
    inp["cu"][2:] = 16
    inp["seq_of"][:8], inp["seq_of"][8:] = 0, 1
    inp["pos"][:8] = inp["ctx"][0] + np.arange(8)
    inp["pos"][8:] = inp["ctx"][1] + np.arange(8)
    inp["kv_lens"][:2] += 1
    return cell, inp


def test_eight_token_window_count_8_bit_exact():
    cell, inp = _eight_token_windows()
    d, _ = C.device_inputs(inp, cache="random")
    T, ms, s_cap = inp["T"], inp["ms"], inp["pps"] * C.PAGE
    plan = WU.plan_host(inp["tk"], inp["cu"], inp["dist"], inp["kv_lens"], inp["table"], C.PAGE, s_cap=s_cap)
    assert plan["U"] == [1024, 1024]
    meta = WU.window_meta(d["cu"], d["dist"], d["kv_lens"], num_tokens=T, s_cap=s_cap)
    ng, rg, mk, uc = WU.union_gather(meta, d["topk"], d["pages"], d["nope"], d["rope"], num_tokens=T,
                                     max_num_seqs=ms, mask_sb_major=False)
    rn, _, _ = WU.reference_union_gather(d["nope"], d["rope"], plan, T)
    uc, ng, mk, rn = map(np.asarray, (uc, ng, mk, rn))
    assert [int(uc[0]), int(uc[16])] == [1024, 1024]
    np.testing.assert_array_equal(ng[:1024], rn[:1024])
    np.testing.assert_array_equal(ng[8 * 1024:9 * 1024], rn[8 * 1024:9 * 1024])
    np.testing.assert_array_equal(mk, plan["mask"])


def test_eight_token_window_matches_oracle():
    cell, inp = _eight_token_windows()
    d, deq = C.device_inputs(inp, cache="quant", poison=True)
    want, rows = C.oracle(inp, d, deq)
    inc = np.asarray(jax.jit(_region(cell.conc, False))(d, d["q"]).astype(jnp.float32))
    cand = np.asarray(jax.jit(_region(cell.conc, True))(d, d["q"]).astype(jnp.float32))
    ref_err, _, _ = C.gate(inc[rows], want)
    err, ok, tol = C.gate(cand[rows], want, ref_err)
    assert np.isfinite(cand[rows]).all() and ok, (err, tol, ref_err)
