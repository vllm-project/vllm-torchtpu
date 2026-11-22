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
"""The selection chain's cheaper formulations select exactly what the previous ones did:
`thresholds` on int16 keys, `compact_row` / `compact_union` lane-major over the slots, and the
key-stream's word read. The previous code is kept here as the reference (plain jnp, CPU) on
random, tie-heavy, all-zero, duplicated, full-range and low-bits-only scores; the TPU
lowerings compile deviceless for tpu7x, and on a TPU the kernel matches interpret mode."""

from __future__ import annotations

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
        got = jax.jit(ix.thresholds, static_argnums=(3, 4))(bits, ids, nv, K, G)
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
    got = ix.thresholds(bits, ids, nv, K, G)
    assert not np.array_equal(np.asarray(got[0]), np.asarray(want[0]))


# ------------------------------------------------------------------------ compaction
@pytest.mark.parametrize("Sx", [8, 64, 512])
@pytest.mark.parametrize("kind", KINDS)
def test_compact_row_lane_major_bit_identical(Sx, kind):
    """Same ids and order as the sublane-major compaction: top-K rows (ties truncated by
    `tmax`), select-all rows (`nv <= K`) and the pool-path form (`th = 0, tm = G`)."""
    rng = np.random.default_rng(Sx)
    Gx = Sx * 128
    consts = ix.compact_consts(Sx, K)
    new = jax.jit(lambda *a: ix.compact_row(*a, K, consts))
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
    new = jax.jit(lambda c: ua.compact_union(c, KU, ua.union_consts(Sx, KU)))
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
def _select_kernel(nv_ref, s_ref, code_ref, th_ref, tm_ref, id_ref, u_ref, interpret=False):
    nv = jnp.zeros((4, 1, 1), I32)
    rid = jax.lax.broadcasted_iota(I32, (4, 1, 1), 0)
    for i in range(4):
        nv = jnp.where(rid == i, nv_ref[i], nv)
    ids = jax.lax.broadcasted_iota(I32, (4, S, 128), 1) * 128 + jax.lax.broadcasted_iota(
        I32, (4, S, 128), 2)
    bits = jnp.where(ids < nv, pltpu.bitcast(s_ref[...], I32), -1)
    thr, tmax = ix.thresholds(bits, ids, nv, K, G)
    th_ref[...] = jnp.broadcast_to(thr, (4, 1, 128))
    tm_ref[...] = jnp.broadcast_to(tmax, (4, 1, 128))
    consts = ix.compact_consts(S, K)
    for r in range(4):
        id_ref[r] = ix.compact_row(bits[r], thr[r], tmax[r], nv[r], K, consts)
    u_ids, u_codes, _ = ua.compact_union(code_ref[...], 2048, ua.union_consts(S, 2048))
    u_ref[0:1] = u_ids
    u_ref[1:2] = u_codes


def _select_call(interpret: bool):
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    return pl.pallas_call(
        _select_kernel,
        out_shape=(jax.ShapeDtypeStruct((4, 1, 128), I32), jax.ShapeDtypeStruct((4, 1, 128), I32),
                   jax.ShapeDtypeStruct((4, 1, K), I32), jax.ShapeDtypeStruct((2, 2048), I32)),
        in_specs=[smem, vmem, vmem], out_specs=(vmem,) * 4, interpret=interpret)


def _select_args(seed: int):
    bits, _, nv = _rows("ties" if seed % 2 else "rand", seed)
    sc = jax.lax.bitcast_convert_type(jnp.maximum(bits, 0), F32)
    code = np.zeros((S, 128), np.int32)
    code.reshape(-1)[np.random.default_rng(seed).choice(S * 128, 1500, replace=False)] = 3
    return jnp.asarray(nv).reshape(4), sc, jnp.asarray(code)


def test_select_tpu_path_compiles():
    """int16 count passes, lane-major compactions and the FP8 word read lower for tpu7x."""
    from jax.experimental import topologies

    from vllm_torchtpu.megakernel.wholestep.qsa_full_b import GPK

    dev = topologies.get_topology_desc(platform="tpu", topology_name="tpu7x:2x2x1").devices[0]
    on = jax.sharding.SingleDeviceSharding(dev)
    sds = lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=on)  # noqa: E731
    jax.jit(_select_call(False)).lower(*map(sds, _select_args(0))).compile()
    for lanes in (128, 256):  # `_consume`'s fp8 read, served ck_lanes and full carrier rows
        CH = 32

        def read(ring, out_ref, lanes=lanes):
            rows = ring.bitcast(I32).reshape(CH * 16 * (lanes // 128), 128)
            words = rows[pl.ds(0, CH * 16, stride=lanes // 128)] if lanes > 128 else rows[...]
            out_ref[...] = pltpu.bitcast(words, BF16).astype(F32)

        call = pl.pallas_call(read, out_shape=jax.ShapeDtypeStruct((CH * GPK, 128), F32))
        jax.jit(call).lower(
            jax.ShapeDtypeStruct((CH, 16, 4, lanes), jnp.float8_e4m3fn, sharding=on)).compile()


def test_select_on_tpu_matches_interpret():
    """On device the int16 passes and the lane-major compactions give interpret's bits."""
    if jax.devices()[0].platform != "tpu":
        pytest.skip("needs a TPU")
    for seed in range(4):
        args = _select_args(seed)
        got = jax.jit(_select_call(False))(*args)
        want = jax.jit(_select_call(True))(*args)
        for a, b in zip(got, want):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=seed)
