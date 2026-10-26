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
"""TPU tests for the mixed-bucket union-of-window CSA route (TPU_DSV4_CSA_WINDOW_MIXED).

* L1 (`union_gather` in SparseCore waves: more windows than workers) is bit-exact
  against the numpy plan and the `csa_gather` reference on T512 / T2048 cells.
* `_attention_csa` with window_mixed on stays within the K1v2 gate of the float64
  oracle (max-rel <= max(2 x incumbent, incumbent + 2^-8)); poisoned unreferenced
  cache rows stay finite.
* Positive control: one window's union truncated by a tile must fail the gate.

Cells: real served top-k (KUDL_DATA dumps) when present, else synthetic K1v2 mixed cells.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "kudl"))
import kudl_common as K  # noqa: E402

C = K.C
from vllm_torchtpu.kernels.deepseek_v4.core_attention import (  # noqa: E402
    csa_window_union as WU,
)

@pytest.fixture(autouse=True)
def _tpu_only():
    # A fixture, not a module-level skipif: collection must not claim the TPU (tests/conftest.py).
    if jax.devices()[0].platform != "tpu":
        pytest.skip("TPU (SparseCore) only")

_REAL = os.path.exists(os.path.join(K.DATA, "dp1_kv6144.npz"))
L1_CELLS = (["m16-r22k", "m16-s470kL1", "m8-r22k"] if _REAL else []) + ["c16m512-dec", "c16m512-pf32k"]
ACC_CELLS = (["m16-r100kL1", "m16-s470k", "m8-s470k"] if _REAL else []) + ["c16m512-dec", "c16m512-pf127k"]


def _rope_entries(rg, n):
    x = np.asarray(rg).view(np.uint32).reshape(-1, 128)
    rows = np.empty((x.shape[0] * 2, 128), np.uint32)
    rows[0::2], rows[1::2] = x & 0xFFFF, x >> 16
    t = rows.reshape(-1, 512, 128)
    return np.concatenate([t[:, :, :64], t[:, :, 64:]], axis=1).reshape(-1, 64)[:n]


def test_route_condition():
    assert WU.supported_mixed(512, 16) and WU.supported_mixed(2048, 8)
    assert not WU.supported_mixed(4096, 8) and not WU.supported_mixed(516, 8)
    L, nc, ns = WU.sc_geometry()
    assert WU.waves_for(128, 16, nc * ns) == 1 and WU.waves_for(512, 16, nc * ns) == 3


@pytest.mark.parametrize("name", L1_CELLS)
def test_union_gather_waves_bit_exact(name):
    inp = K.build_inputs(name)
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
    assert len(plan["windows"]) > 32 or name == "c16m512-dec"  # waves exercised
    for w, (q0, qn, _, _) in enumerate(plan["windows"]):
        U = plan["U"][w]
        assert int(uc[w * 16]) == U, (w, int(uc[w * 16]), U)
        r0 = q0 * WU.TILE
        np.testing.assert_array_equal(ng[r0:r0 + U], rn[r0:r0 + U])
        np.testing.assert_array_equal(re_[r0:r0 + U], rr_[r0:r0 + U])
        np.testing.assert_array_equal(mk[q0:q0 + qn], plan["mask"][q0:q0 + qn])


@pytest.mark.parametrize("arm", ["win", "win_g4"])
@pytest.mark.parametrize("poison", [False, True])
@pytest.mark.parametrize("name", ACC_CELLS)
def test_window_mixed_matches_oracle(name, poison, arm):
    cell = K.cell_obj(name)
    inp = K.build_inputs(name)
    d, deq = C.device_inputs(inp, cache="quant", poison=poison)
    want, rows = C.oracle(inp, d, deq)
    inc = np.asarray(jax.jit(K.inc(cell.conc))(d, d["q"]).astype(jnp.float32))
    cand = np.asarray(jax.jit(K.ARMS[arm](cell.conc))(d, d["q"]).astype(jnp.float32))
    ref_err, _, _ = C.gate(inc[rows], want)
    err, ok, tol = C.gate(cand[rows], want, ref_err)
    assert np.isfinite(cand[rows]).all()
    assert ok, (name, arm, err, tol, ref_err)


@pytest.mark.parametrize("name", L1_CELLS)
def test_grouped_equals_ungrouped_bitwise(name):
    """Grouping only splits the windows over calls: same windows, same per-window order -> identical bits."""
    cell = K.cell_obj(name)
    inp = K.build_inputs(name)
    d, _ = C.device_inputs(inp, cache="quant")
    a = np.asarray(jax.jit(K.win(cell.conc))(d, d["q"]).astype(jnp.float32))
    for g in (K.win_g2, K.win_g4):
        b = np.asarray(jax.jit(g(cell.conc))(d, d["q"]).astype(jnp.float32))
        np.testing.assert_array_equal(a, b)


def test_positive_control_truncated_union_fails():
    """A union cut at one tile in every window must fail the same gate (the gate can see a lost tile)."""
    name = ACC_CELLS[0]
    cell = K.cell_obj(name)
    inp = K.build_inputs(name)
    d, deq = C.device_inputs(inp, cache="quant")
    want, rows = C.oracle(inp, d, deq)
    inc = np.asarray(jax.jit(K.inc(cell.conc))(d, d["q"]).astype(jnp.float32))
    ref_err, _, _ = C.gate(inc[rows], want)
    orig = WU.union_gather

    def truncated(*a, **k):
        ng, rg, mk, uc = orig(*a, **k)
        return ng, rg, mk, jnp.minimum(uc, WU.TILE)

    WU.union_gather = truncated
    try:
        cand = np.asarray(jax.jit(K.win(cell.conc))(d, d["q"]).astype(jnp.float32))
    finally:
        WU.union_gather = orig
    _, ok, _ = C.gate(cand[rows], want, ref_err)
    assert not ok


WQ16_CELLS = (["m8-r22k", "m8-r22k-p1m", "m16-s470kL1"] if _REAL else []) + ["c16m512-pf32k"]


@pytest.mark.parametrize("name", WQ16_CELLS)
def test_union_gather_wq16_bit_exact(name):
    """16-token windows (two nibble count halves): every group's windows bit-exact vs the numpy plan."""
    inp = K.build_inputs(name)
    d, _ = C.device_inputs(inp, cache="random")
    T, ms, s_cap = inp["T"], inp["ms"], inp["pps"] * C.PAGE
    plan = WU.plan_host(inp["tk"], inp["cu"], inp["dist"], inp["kv_lens"], inp["table"],
                        C.PAGE, s_cap=s_cap, wq=16)
    rn, rr, _ = WU.reference_union_gather(d["nope"], d["rope"], plan, T)
    rn = np.asarray(rn)
    rr_ = _rope_entries(rr, T * WU.TILE)
    groups = 2
    w_base = 0
    for g in range(groups):
        meta, tslot, nwc = WU.group_meta(d["cu"], d["dist"], d["kv_lens"], num_tokens=T, s_cap=s_cap,
                                         groups=groups, g=g, wq=16)
        ng, rg, mk, uc = WU.union_gather(meta, d["topk"], d["pages"], d["nope"], d["rope"], num_tokens=T,
                                         max_num_seqs=ms, n_tiles=nwc * 16, nw_cap_override=nwc, wq=16)
        ng, mk, uc, meta = map(np.asarray, (ng, mk, uc, meta))
        re_ = _rope_entries(rg, nwc * 16 * WU.TILE)
        n_g = int(meta[0])
        for i in range(n_g):
            q0, qn, _, _ = plan["windows"][w_base + i]
            U = plan["U"][w_base + i]
            assert int(uc[i * 16]) == U, (g, i, int(uc[i * 16]), U)
            r0, s0 = q0 * WU.TILE, i * 16 * WU.TILE
            np.testing.assert_array_equal(ng[s0:s0 + U], rn[r0:r0 + U])
            np.testing.assert_array_equal(re_[s0:s0 + U], rr_[r0:r0 + U])
            np.testing.assert_array_equal(mk[q0:q0 + qn], plan["mask"][q0:q0 + qn])
        w_base += n_g
    assert w_base == len(plan["windows"])


@pytest.mark.parametrize("arm", ["wq16", "wq16_g4"])
@pytest.mark.parametrize("name", ["m8-r22k", "m16-r100kL1"] if _REAL else ["c16m512-pf127k"])
def test_wq16_matches_oracle(name, arm):
    cell = K.cell_obj(name)
    inp = K.build_inputs(name)
    for poison in (False, True):
        d, deq = C.device_inputs(inp, cache="quant", poison=poison)
        want, rows = C.oracle(inp, d, deq)
        inc = np.asarray(jax.jit(K.inc(cell.conc))(d, d["q"]).astype(jnp.float32))
        cand = np.asarray(jax.jit(K.ARMS[arm](cell.conc))(d, d["q"]).astype(jnp.float32))
        ref_err, _, _ = C.gate(inc[rows], want)
        err, ok, tol = C.gate(cand[rows], want, ref_err)
        assert np.isfinite(cand[rows]).all()
        assert ok, (name, arm, poison, err, tol, ref_err)
