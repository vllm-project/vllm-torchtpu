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
"""VLLM_TPU_MK_QSA_LANE_MAJOR_SELECT selects exactly what the flag-off selection does:
`thresholds` on int16 keys (plain and early-exit value passes, tie passes), `compact_row` /
`compact_sel` / `compact_union` / `compact_union_chunk` lane-major over the slots. The
flag-off formulations (frozen copies below, and the in-tree flag-off paths) are the
reference (plain jnp, CPU) on random, tie-heavy, all-zero, duplicated, full-range and
low-bits-only scores; the TPU lowerings compile deviceless for tpu7x, and on a TPU the
kernel matches interpret mode."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.wholestep import indexer as ix
from vllm_torchtpu.megakernel.wholestep import union_attn as ua

I32, F32, BF16 = jnp.int32, jnp.float32, jnp.bfloat16
S, G, K = 512, 65536, 512


# ------------------------------------------------------- previous formulations (reference)
def _thresholds_i32(bits, ids, nv, K, G):
    cnt = lambda m: jnp.sum(m.astype(I32), axis=(1, 2), keepdims=True)  # noqa: E731
    over = nv > K
    lo = jnp.zeros(nv.shape, I32)
    for b in range(30, -1, -1):
        cand = lo | (1 << b)
        lo = jnp.where(cnt(bits >= cand) >= K, cand, lo)
    thr = jnp.where(over, lo, 0)
    eq = bits == thr
    need = K - cnt(bits > thr)
    t = jnp.zeros(nv.shape, I32)
    for b in range(int(G - 1).bit_length() - 1, -1, -1):
        cand = t | (1 << b)
        t = jnp.where(cnt(eq & (ids < cand)) < need, cand, t)
    return thr, jnp.where(over & (cnt(eq) > need), t, G)


def _compact_row_sublane_major(b, th, tm, nv, K):
    Sx = b.shape[0]
    li = jax.lax.broadcasted_iota(I32, (128, 128), 0)
    lj = jax.lax.broadcasted_iota(I32, (128, 128), 1)
    U, Us = (li <= lj).astype(BF16), (li < lj).astype(BF16)
    jcol = jax.lax.broadcasted_iota(I32, (K, 1), 0)
    srow = jax.lax.broadcasted_iota(I32, (1, Sx), 1)
    lid = srow.reshape(Sx, 1) * 128 + jax.lax.broadcasted_iota(I32, (Sx, 128), 1)
    ov = nv > K
    sel = (ov & ((b > th) | ((b == th) & (lid <= tm)))) | (~ov & (b >= 0))
    P = jnp.dot(sel.astype(F32).astype(BF16), U, preferred_element_type=F32)
    c_row = jnp.transpose(jnp.broadcast_to(P[:, 127:128], (Sx, 128)))[0:1, :]
    O_row = ix.exclusive_prefix(c_row, Us)
    E_row = O_row + c_row
    jf = jcol.astype(F32)
    r_j = jnp.sum((E_row <= jf).astype(I32), axis=1, keepdims=True)
    oh = srow == r_j
    Pg = jnp.dot(oh.astype(F32).astype(BF16), P.astype(BF16), preferred_element_type=F32)
    O_rj = jnp.sum(jnp.where(oh, O_row, 0.0), axis=1, keepdims=True)
    lane_j = jnp.sum((Pg <= jf - O_rj).astype(I32), axis=1, keepdims=True)
    idc = jnp.where(jcol < jnp.minimum(nv, K), r_j * 128 + lane_j, 0).astype(F32)
    return jnp.transpose(jnp.broadcast_to(idc, (K, 128)))[0:1, :].astype(I32)


def _compact_union_sublane_major(code, KU):
    Sx = code.shape[0]
    W = min(KU, ua.UNION_CHUNK)
    li = jax.lax.broadcasted_iota(I32, (128, 128), 0)
    lj = jax.lax.broadcasted_iota(I32, (128, 128), 1)
    U, Us = (li <= lj).astype(BF16), (li < lj).astype(BF16)
    jcol0 = jax.lax.broadcasted_iota(I32, (W, 1), 0)
    srow = jax.lax.broadcasted_iota(I32, (1, Sx), 1)
    lane = jax.lax.broadcasted_iota(I32, (W, 128), 1)
    sel = code > 0
    P = jnp.dot(sel.astype(F32).astype(BF16), U, preferred_element_type=F32)
    c_row = jnp.transpose(jnp.broadcast_to(P[:, 127:128], (Sx, 128)))[0:1, :]
    O_row = ix.exclusive_prefix(c_row, Us)
    E_row = O_row + c_row
    Pb, codeb = P.astype(BF16), code.astype(F32).astype(BF16)
    n = jnp.sum(sel.astype(I32), axis=(0, 1), keepdims=True)
    ids_out, codes_out = [], []
    for j0 in range(0, KU, W):
        jcol = jcol0 + j0
        jf = jcol.astype(F32)
        r_j = jnp.sum((E_row <= jf).astype(I32), axis=1, keepdims=True)
        hit = srow == r_j
        oh = hit.astype(F32).astype(BF16)
        Pg = jnp.dot(oh, Pb, preferred_element_type=F32)
        O_rj = jnp.sum(jnp.where(hit, O_row, 0.0), axis=1, keepdims=True)
        lane_j = jnp.sum((Pg <= jf - O_rj).astype(I32), axis=1, keepdims=True)
        Cg = jnp.dot(oh, codeb, preferred_element_type=F32)
        code_j = jnp.sum(jnp.where(lane == lane_j, Cg, 0.0), axis=1, keepdims=True)
        live = jcol < n
        ids = jnp.where(live, r_j * 128 + lane_j, 0).astype(F32)
        codes = jnp.where(live, code_j, 0.0)
        ids_out.append(jnp.transpose(jnp.broadcast_to(ids, (W, 128)))[0:1, :].astype(I32))
        codes_out.append(jnp.transpose(jnp.broadcast_to(codes, (W, 128)))[0:1, :].astype(I32))
    return jnp.concatenate(ids_out, axis=1), jnp.concatenate(codes_out, axis=1), n


# --------------------------------------------------------------------------- inputs
def _scores(rng, kind: str, shape: tuple) -> np.ndarray:
    if kind == "rand":
        return np.maximum(rng.standard_normal(shape), 0).astype(np.float32) * 7
    if kind == "ties":  # few distinct values: truncated ties at the threshold
        return rng.integers(0, 6, shape).astype(np.float32) * 0.5
    if kind == "zero":
        return np.zeros(shape, np.float32)
    if kind == "dups":  # each value 16 times (duplicated keys)
        n = int(np.prod(shape))
        base = np.maximum(rng.standard_normal(n // 16), 0).astype(np.float32)
        return rng.permutation(np.repeat(base, 16)).reshape(shape)
    if kind == "wide":  # every finite non-negative exponent, denormals included
        return rng.integers(0, 0x7F800000, shape, dtype=np.int64).astype(np.uint32).view(np.float32)
    assert kind == "lowbits"  # one high half: everything is settled by the low 16 bits
    return (0x3F800000 + rng.integers(0, 1 << 16, shape)).astype(np.uint32).view(np.float32)


KINDS = ["rand", "ties", "zero", "dups", "wide", "lowbits"]
NVS = [0, 1, K - 1, K, K + 1, 2000, 26000, G - 5, G]


def _rows(kind: str, seed: int, n: int = 4):
    rng = np.random.default_rng(seed)
    sc = _scores(rng, kind, (n, S, 128))
    nv = rng.choice(NVS, n).astype(np.int32).reshape(n, 1, 1)
    ids = np.arange(G, dtype=np.int32).reshape(1, S, 128)
    bits = np.where(ids < nv, sc.view(np.int32), -1).astype(np.int32)
    return jnp.asarray(bits), jnp.asarray(np.broadcast_to(ids, bits.shape)), jnp.asarray(nv)


# ------------------------------------------------------------------------ thresholds
@pytest.mark.parametrize("kind", KINDS)
def test_thresholds_int16_keys_bit_identical(kind):
    """Same `(thr, tmax)` as the int32 passes, truncated ties included (`tmax < G`)."""
    for seed in range(3):
        bits, ids, nv = _rows(kind, seed)
        want = jax.jit(_thresholds_i32, static_argnums=(3, 4))(bits, ids, nv, K, G)
        got = jax.jit(functools.partial(ix.thresholds, packed16=True),
                      static_argnums=(3, 4))(bits, ids, nv, K, G)
        for a, b in zip(got, want):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=(kind, seed))


def test_thresholds_int16_keys_control(monkeypatch):
    """Positive control: the low-half passes without `count(hi > p)` give other thresholds."""
    bits, ids, nv = _rows("rand", 0)
    want = _thresholds_i32(bits, ids, nv, K, G)

    def no_above(bits, K):
        hi = (bits >> 16).astype(ix.I16)
        lo = jnp.zeros((bits.shape[0], 1, 1), I32)
        for b in range(30, 15, -1):
            cand = lo | (1 << b)
            lo = jnp.where(ix._count16(hi >= (cand >> 16).astype(ix.I16)) >= K, cand, lo)
        p = (lo >> 16).astype(ix.I16)
        low = jnp.where(hi == p, ((bits & 0xFFFF) - 32768).astype(ix.I16), ix.I16(-32768))
        for b in range(15, -1, -1):
            cand = lo | (1 << b)
            c = ((cand & 0xFFFF) - 32768).astype(ix.I16)
            lo = jnp.where(ix._count16(low >= c) >= K, cand, lo)
        return lo

    monkeypatch.setattr(ix, "_value_passes16", no_above)
    got = ix.thresholds(bits, ids, nv, K, G, packed16=True)
    assert not np.array_equal(np.asarray(got[0]), np.asarray(want[0]))


# ------------------------------------------------------------------------ compaction
@pytest.mark.parametrize("Sx", [8, 64, 512])
@pytest.mark.parametrize("kind", KINDS)
def test_compact_row_lane_major_bit_identical(Sx, kind):
    """Same ids and order as the sublane-major compaction: top-K rows (ties truncated by
    `tmax`), select-all rows (`nv <= K`) and the pool-path form (`th = 0, tm = G`)."""
    rng = np.random.default_rng(Sx)
    Gx = Sx * 128
    consts = ix.compact_consts(Sx, K, True)
    new = jax.jit(lambda *a: ix.compact_row(*a, K, consts, True))
    old = jax.jit(lambda *a: _compact_row_sublane_major(*a, K))
    lid = np.arange(Gx).reshape(Sx, 128)
    for nvis in (0, 1, K - 1, K, K + 1, Gx // 2, Gx):
        sc = _scores(rng, kind, (Sx, 128))
        b = jnp.asarray(np.where(lid < nvis, sc.view(np.int32), -1).astype(np.int32))
        nv = jnp.full((1, 1), nvis, I32)
        th, tm = ix.thresholds(b[None], jnp.asarray(lid, I32)[None], nv[None], K, Gx)
        for args in ((b, th[0], tm[0], nv),
                     (jnp.where(rng.random((Sx, 128)) < 0.003, jnp.maximum(b, 0), -1),
                      jnp.zeros((1, 1), I32), jnp.full((1, 1), Gx, I32), nv)):
            if args[0] is not b:  # pool-path form: nv = the count of kept groups
                args = (*args[:3], jnp.sum(args[0] >= 0).reshape(1, 1).astype(I32))
            np.testing.assert_array_equal(np.asarray(new(*args)), np.asarray(old(*args)),
                                          err_msg=(Sx, kind, nvis))


@pytest.mark.parametrize("Sx,KU", [(512, 2048), (64, 2048), (8, 512), (512, 512)])
def test_compact_union_lane_major_bit_identical(Sx, KU):
    """Same union ids, codes and count, including more coded lanes than union slots."""
    rng = np.random.default_rng(KU + Sx)
    new = jax.jit(lambda c: ua.compact_union(c, KU, ua.union_consts(Sx, KU, True), True))
    old = jax.jit(lambda c: _compact_union_sublane_major(c, KU))
    for n in (0, 1, 7, 100, KU // 2, KU, KU + 300):
        n = min(n, Sx * 128)
        for pos in (rng.choice(Sx * 128, n, replace=False), np.arange(n),
                    Sx * 128 - 1 - np.arange(n)):
            code = np.zeros((Sx, 128), np.int32)
            code.reshape(-1)[pos] = rng.integers(1, 16, n)
            for a, b in zip(new(jnp.asarray(code)), old(jnp.asarray(code))):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=(Sx, KU, n))


# ---------------------------------------------------------------- TPU lowering / device
def _select_kernel(nv_ref, s_ref, code_ref, th_ref, tm_ref, id_ref, u_ref, lm=True, thr_kw=None):
    thr_kw = thr_kw or {}
    nv = jnp.zeros((4, 1, 1), I32)
    rid = jax.lax.broadcasted_iota(I32, (4, 1, 1), 0)
    for i in range(4):
        nv = jnp.where(rid == i, nv_ref[i], nv)
    ids = jax.lax.broadcasted_iota(I32, (4, S, 128), 1) * 128 + jax.lax.broadcasted_iota(
        I32, (4, S, 128), 2)
    bits = jnp.where(ids < nv, pltpu.bitcast(s_ref[...], I32), -1)
    thr, tmax = ix.thresholds(bits, ids, nv, K, G, packed16=lm, **thr_kw)
    th_ref[...] = jnp.broadcast_to(thr, (4, 1, 128))
    tm_ref[...] = jnp.broadcast_to(tmax, (4, 1, 128))
    consts = ix.compact_consts(S, K, lm)
    for r in range(4):
        id_ref[r] = ix.compact_row(bits[r], thr[r], tmax[r], nv[r], K, consts, lm)
    u_ids, u_codes, _ = ua.compact_union(code_ref[...], 2048, ua.union_consts(S, 2048, lm), lm)
    u_ref[0:1] = u_ids
    u_ref[1:2] = u_codes


def _select_call(interpret: bool, lm: bool = True, **thr_kw):
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    return pl.pallas_call(
        functools.partial(_select_kernel, lm=lm, thr_kw=thr_kw),
        out_shape=(jax.ShapeDtypeStruct((4, 1, 128), I32), jax.ShapeDtypeStruct((4, 1, 128), I32),
                   jax.ShapeDtypeStruct((4, 1, K), I32), jax.ShapeDtypeStruct((2, 2048), I32)),
        in_specs=[smem, vmem, vmem], out_specs=(vmem,) * 4, interpret=interpret)


def _select_args(seed: int):
    bits, _, nv = _rows("ties" if seed % 2 else "rand", seed)
    sc = jax.lax.bitcast_convert_type(jnp.maximum(bits, 0), F32)
    code = np.zeros((S, 128), np.int32)
    code.reshape(-1)[np.random.default_rng(seed).choice(S * 128, 1500, replace=False)] = 3
    return jnp.asarray(nv).reshape(4), sc, jnp.asarray(code)


@pytest.mark.parametrize("thr_kw", [{}, dict(tie_skip=True, early_exit=(19, 2))])
def test_select_tpu_path_compiles(thr_kw):
    """int16 count passes (plain and early exit) and the lane-major compactions lower for
    tpu7x."""
    from jax.experimental import topologies

    dev = topologies.get_topology_desc(platform="tpu", topology_name="tpu7x:2x2x1").devices[0]
    on = jax.sharding.SingleDeviceSharding(dev)
    sds = lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=on)  # noqa: E731
    jax.jit(_select_call(False, True, **thr_kw)).lower(*map(sds, _select_args(0))).compile()


def test_select_on_tpu_matches_interpret():
    """On device the int16 passes and the lane-major compactions give interpret's bits and
    the flag-off kernel's."""
    if jax.devices()[0].platform != "tpu":
        pytest.skip("needs a TPU")
    for seed in range(4):
        args = _select_args(seed)
        got = jax.jit(_select_call(False))(*args)
        want = jax.jit(_select_call(True))(*args)
        off = jax.jit(_select_call(False, False))(*args)
        for a, b in zip(got, off):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=seed)
        for a, b in zip(got, want):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=seed)


# ------------------------------------------------------------- early exit / tie skip
FAST = [dict(early_exit=(19, 2)), dict(tie_skip=True, early_exit=(19, 2)),
        dict(early_exit=(15, 1)), dict(early_exit=(23, 4)), dict(early_exit=(30, 1)),
        dict(early_exit=(3, 4)), dict(tie_skip=True)]


@pytest.mark.parametrize("kw", FAST, ids=str)
@pytest.mark.parametrize("kind", KINDS)
def test_thresholds_int16_early_exit_bit_identical(kind, kw):
    """`packed16` under VLLM_TPU_MK_QSA_THR_FAST's early exit / tie skip: the same `(thr,
    tmax)` as the int32 passes with the same options (the same stopping trip), incl.
    the tier widths S / 2 and 3S / 4."""
    for seed in range(2):
        bits, ids, nv = _rows(kind, seed)
        for Sx in (S, 3 * S // 4, S // 2):
            b, i = bits[:, :Sx], ids[:, :Sx]
            want = ix.thresholds(b, i, nv, K, G, **kw)
            got = ix.thresholds(b, i, nv, K, G, packed16=True, **kw)
            for x, y in zip(got, want):
                np.testing.assert_array_equal(np.asarray(x), np.asarray(y),
                                              err_msg=(kind, seed, Sx, kw))


def test_packed_early_exit_keeps_trips():
    """The int16 early exit runs only where its trips are the int32 one's."""
    assert ix._packed_early_ok(19, 2) and ix._packed_early_ok(15, 1)
    assert ix._packed_early_ok(23, 4) and ix._packed_early_ok(31 - 16, 16)
    assert not ix._packed_early_ok(3, 4)  # static passes below bit 16
    assert not ix._packed_early_ok(20, 3)  # 16 % 3: a trip straddles bit 16


def test_thresholds_negative_zero_scores():
    """-0.0 scores (bit pattern < 0) count nowhere, in both formulations."""
    bits, ids, nv = _rows("rand", 5)
    neg0 = np.int32(np.uint32(0x80000000).view(np.int32))
    bits = jnp.where((ids % 7 == 0) & (bits >= 0), neg0, bits)
    for kw in ({}, dict(early_exit=(19, 2))):
        want = ix.thresholds(bits, ids, nv, K, G, **kw)
        got = ix.thresholds(bits, ids, nv, K, G, packed16=True, **kw)
        for x, y in zip(got, want):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=str(kw))


# ------------------------------------------------------- pool-path / chunked compactions
@pytest.mark.parametrize("kind", ["rand", "ties", "zero"])
def test_compact_sel_pool_path_matches_compact_row(kind):
    """`scan_row`'s pool path: `compact_sel(sel & (lid < g_a), kept)` (flag on) gives the ids
    of `compact_row(b2, 0, G, kept)` (flag off)."""
    rng = np.random.default_rng(11)
    lid = np.arange(G).reshape(S, 128)
    off_c, on_c = ix.compact_consts(S, K), ix.compact_consts(S, K, True)
    for nvis in (0, 3, K, K + 1, 26000, G):
        sc = _scores(rng, kind, (S, 128))
        b = jnp.asarray(np.where(lid < nvis, sc.view(np.int32), -1).astype(np.int32))
        nv = jnp.full((1, 1), nvis, I32)
        th, tm = ix.thresholds(b[None], jnp.asarray(lid, I32)[None], nv[None], K, G)
        th, tm = th[0], tm[0]
        ov = nv > K
        L = jnp.asarray(lid, I32)
        sel = (ov & ((b > th) | ((b == th) & (L <= tm)))) | (~ov & (b >= 0))
        for ga in (0, 1, nvis // 4, max(nvis - 1, 0), nvis):
            keep = sel & (L < ga)
            kept = jnp.sum(keep.astype(I32)).reshape(1, 1)
            b2 = jnp.where(keep, jnp.where(L < ga, b, -1), -1)
            want = ix.compact_row(b2, jnp.zeros((1, 1), I32), jnp.full((1, 1), G, I32), kept, K,
                                  off_c)
            got = ix.compact_sel(keep, kept, K, on_c)
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want),
                                          err_msg=(kind, nvis, ga))


def _lane_major_chunks(monkeypatch):
    """Route `test_qsa_select_nvis`'s chunk loops through the lane-major chunk compaction."""
    on_consts = functools.partial(ua.union_chunk_consts, lane_major=True)
    on_chunk = functools.partial(ua.compact_union_chunk, lane_major=True)
    monkeypatch.setattr(ua, "union_chunk_consts", on_consts)
    monkeypatch.setattr(ua, "compact_union_chunk", on_chunk)


@pytest.mark.parametrize("nv,D,dense", [(5000, 1, 0.3), (25000, 2, 0.05), (9000, 2, 0.9),
                                         (16384, 1, 0.12), (0, 1, 0.5), (65536, 1, 0.03)])
def test_union_chunks_lane_major_match_full(monkeypatch, nv, D, dense):
    """`compact_union_chunk(lane_major=True)` over the visible rows equals the flag-off
    `compact_union` over all rows: same ids, codes and count."""
    from tests.megakernel import test_qsa_select_nvis as t

    _lane_major_chunks(monkeypatch)
    t.test_union_chunks_match_full(nv, D, dense)


@pytest.mark.parametrize("nv,gpp,dense", [(25000, 32, 0.05), (9000, 32, 0.9), (0, 32, 0.5),
                                          (65536, 32, 0.004), (25000, 128, 0.05),
                                          (50000, 128, 0.02), (9000, 128, 0.9),
                                          (65536, 128, 0.004)])
def test_union_lanepack_lane_major_matches_masked(monkeypatch, nv, gpp, dense):
    """VLLM_TPU_MK_QSA_UNION_LANEPACK with the lane-major chunk compaction (`id_of` on `[1, W]`
    lanes): each pair rank's union equals the flag-off owner-masked one."""
    from tests.megakernel import test_qsa_select_nvis as t

    _lane_major_chunks(monkeypatch)
    t.test_union_lanepack_matches_masked(nv, gpp, dense)


def test_union_chunk_lane_major_control(monkeypatch):
    """Negative control: a lane-major chunk loop with an off-by-one `tot` is caught."""
    from tests.megakernel import test_qsa_select_nvis as t

    orig = functools.partial(ua.compact_union_chunk, lane_major=True)
    monkeypatch.setattr(ua, "union_chunk_consts",
                        functools.partial(ua.union_chunk_consts, lane_major=True))
    monkeypatch.setattr(ua, "compact_union_chunk",
                        lambda code, tot, *a, **k: orig(code, tot + 1, *a, **k))
    with pytest.raises(AssertionError):
        t.test_union_chunks_match_full(25000, 1, 0.05)


@pytest.mark.parametrize("nv,gpp,dense", [(25000, 128, 0.05), (50000, 32, 0.02)])
def test_full_union_lanepack_lane_major(nv, gpp, dense):
    """`full_union` under lane-pack: lane-major `compact_union` over the packed rows, ids
    unpacked, equals the flag-off owner-masked union."""
    Sx, KU = 512, 2048
    rng = np.random.default_rng(nv + gpp)
    code = rng.integers(1, 16, (Sx, 128)) * (rng.random((Sx, 128)) < dense)
    gid = np.arange(Sx)[:, None] * 128 + np.arange(128)[None]
    code = np.where(gid < nv, code, 0).astype(np.int32)
    for r in range(2):
        own = np.where((gid // gpp) % 2 == r, code, 0)
        want = ua.compact_union(jnp.asarray(own), KU, ua.union_consts(Sx, KU))
        if gpp == 128:
            packed = jnp.asarray(code[r::2])
        else:
            packed = ua.lane_pack(jnp.asarray(code[0::2]), jnp.asarray(code[1::2]), r, gpp,
                                  jnp.roll)
        ids, codes, n = ua.compact_union(packed, KU, ua.union_consts(Sx // 2, KU, True), True)
        np.testing.assert_array_equal(ua.unpack_union_ids(ids, n, r, gpp), want[0])
        np.testing.assert_array_equal(codes, want[1])
        assert int(np.sum(n)) == int(np.sum(want[2]))


def test_union_chunk_lane_major_tpu_lowering():
    """The lane-major chunk compaction (with `pl.when` windows and a stored output) lowers
    for tpu7x."""
    from jax.experimental import topologies

    CR, KU = 128, 2048
    dev = topologies.get_topology_desc(platform="tpu", topology_name="tpu7x:2x2x1").devices[0]
    on = jax.sharding.SingleDeviceSharding(dev)

    def kernel(tot_ref, code_ref, ids_ref, cd_ref):
        ids_ref[...] = jnp.zeros_like(ids_ref)
        cd_ref[...] = jnp.zeros_like(cd_ref)

        def store(j0, ids, codes):
            W = ids.shape[1]
            ids_ref[:, pl.ds(j0, W)] = ids_ref[:, pl.ds(j0, W)] + ids
            cd_ref[:, pl.ds(j0, W)] = cd_ref[:, pl.ds(j0, W)] + codes

        consts = ua.union_chunk_consts(CR, KU, True)
        tot = tot_ref[0]
        ua.compact_union_chunk(code_ref[...], tot, lambda rj: rj + 8, KU, consts, store,
                               lane_major=True)

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    call = pl.pallas_call(
        kernel, out_shape=(jax.ShapeDtypeStruct((1, KU), I32),) * 2,
        in_specs=[smem, vmem], out_specs=(vmem, vmem))
    jax.jit(call).lower(jax.ShapeDtypeStruct((1,), I32, sharding=on),
                        jax.ShapeDtypeStruct((CR, 128), I32, sharding=on)).compile()
