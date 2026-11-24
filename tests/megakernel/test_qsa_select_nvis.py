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
"""`select_nvis` (VLLM_TPU_MK_QSA_SELECT_NVIS) pieces vs the full-row selection: the
chunked threshold bisection and compaction give the same thresholds and ids (plain jnp,
production row width S = 512, plus the compressed-key DCP score layout)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.megakernel.wholestep import indexer as ix

I32, F32 = jnp.int32, jnp.float32


def _gid(p, S, gpp, D):
    """`_cdcp_gid` of score positions `p` (D = 1: identity)."""
    if D == 1:
        return p
    Gh = S * 128 // D
    loc = p % Gh
    return (loc // gpp) * (D * gpp) + (p // Gh) * gpp + loc % gpp


def _local_nv(nv, rk, gpp, D):
    return (nv // (D * gpp)) * gpp + np.clip(nv % (D * gpp) - rk * gpp, 0, gpp)


def _plan(nv, CH, S, gpp, D):
    span = CH * 128
    if D == 1:
        return min(-(-nv // span), S // CH), lambda i: i * CH
    Sh = S // D
    m0, m1 = (min(-(-int(_local_nv(nv, rk, gpp, D)) // span), Sh // CH) for rk in range(D))
    return m0 + m1, lambda i: jnp.where(i < m0, i * CH, Sh + (i - m0) * CH)


def _scores(key, n, S, ties):
    s = jax.random.uniform(jax.random.PRNGKey(key), (n, S, 128), F32)
    if ties:  # few distinct values: many ties at the threshold
        s = jnp.floor(s * 7) / 7
    return jax.lax.bitcast_convert_type(s, I32)


CASES = [  # (nv per row, K, ties, D)
    ([5000, 5001, 5001, 5002], 512, False, 1),
    ([5000, 5001, 5001, 5002], 512, True, 1),
    ([100, 100, 101, 101], 512, False, 1),  # n_vis <= K: select all
    ([8192, 8192, 8193, 8193], 512, True, 1),  # exact chunk boundary
    ([65536 - 3, 65536 - 2, 65536 - 1, 65536], 512, True, 1),  # near max_groups
    ([0, 0, 1, 2], 512, False, 1),  # empty rows
    ([25000, 25001, 25001, 25002], 512, True, 2),
    ([6000, 6001, 6002, 6003], 512, False, 2),
    ([127, 128, 129, 600], 512, True, 2),
]


@pytest.mark.parametrize("digit", [1, 3])
@pytest.mark.parametrize("nvs,K,ties,D", CASES)
def test_chunked_selection_matches_full(nvs, K, ties, D, digit):
    S, G, n, gpp, CH, CM = 512, 65536, len(nvs), 128, 32, 64
    bits_raw = _scores(len(nvs) + sum(nvs) + K, n, S, ties)
    p = jnp.arange(S)[:, None] * 128 + jnp.arange(128)[None]
    ids = jnp.broadcast_to(_gid(p, S, gpp, D), (n, S, 128))
    nv = jnp.asarray(nvs, I32).reshape(n, 1, 1)
    bits = jnp.where(ids < nv, bits_raw, -1)
    thr, tmax = ix.thresholds(bits, ids, nv, K, G)

    m, base = _plan(max(nvs), CH, S, gpp, D)
    m = m if max(nvs) > K else 0

    def chunk(a, c):
        return jax.lax.dynamic_slice(a, (0, base(c), 0), (n, CH, 128))

    thr2, tmax2 = ix.thresholds_chunks(
        lambda c: chunk(bits, c), lambda c: chunk(ids, c), m, nv, K, G, (n, CH, 128),
        digit=digit,
    )
    np.testing.assert_array_equal(thr2, thr)
    np.testing.assert_array_equal(tmax2, tmax)

    consts = ix.compact_consts(S, K)
    nconsts = ix.chunk_consts(2 * CM, K)
    lid = _gid(consts[-1], S, gpp, D)
    for r in range(n):
        b = bits[r]
        ov = jnp.bool_(nvs[r] > K)
        sel = (ov & ((b > thr[r]) | ((b == thr[r]) & (lid <= tmax[r])))) | (~ov & (b >= 0))
        kept = int(jnp.sum(sel))
        b2 = jnp.where(sel, b, -1)
        want = ix.compact_row(b2, jnp.zeros((1, 1), I32), jnp.zeros((1, 1), I32) + G,
                              jnp.zeros((1, 1), I32) + kept, K, consts)
        mr, base_r = _plan(nvs[r], CM, S, gpp, D)
        acc, tot = jnp.zeros((1, K), F32), jnp.zeros((1, 1), F32)
        for i in range((mr + 1) // 2):
            parts, bases = [], []
            for k in range(2):
                c = 2 * i + k
                bs = base_r(c) if c < mr else 0
                parts.append(jax.lax.dynamic_slice(sel, (bs, 0), (CM, 128)) & (c < mr))
                bases.append(bs)
            part, tot = ix.compact_chunk(
                jnp.concatenate(parts, 0), tot,
                lambda rj, bases=bases: jnp.where(rj < CM, bases[0] + rj, bases[1] + rj - CM),
                nconsts,
            )
            acc = acc + part
        np.testing.assert_array_equal(acc.astype(I32), want, err_msg=f"row {r}")
        assert int(tot[0, 0]) == kept


@pytest.mark.parametrize("digit", [2, 3, 4])
def test_digit_thresholds_match(digit):
    """Several bits per bisection pass: same thresholds on tie-heavy and plain rows."""
    S, G, n, K, CH = 64, 8192, 4, 64, 8
    for key, ties in ((1, True), (2, False)):
        bits_raw = _scores(key, n, S, ties)
        ids = jnp.broadcast_to(jnp.arange(S)[:, None] * 128 + jnp.arange(128)[None], (n, S, 128))
        nv = jnp.asarray([3000, 3001, 8192, 40], I32).reshape(n, 1, 1)
        bits = jnp.where(ids < nv, bits_raw, -1)
        want = ix.thresholds(bits, ids, nv, K, G)

        def chunk(a, c):
            return jax.lax.dynamic_slice(a, (0, c * CH, 0), (n, CH, 128))

        got = ix.thresholds_chunks(lambda c: chunk(bits, c), lambda c: chunk(ids, c), S // CH,
                                   nv, K, G, (n, CH, 128), digit=digit)
        np.testing.assert_array_equal(got[0], want[0])
        np.testing.assert_array_equal(got[1], want[1])


def test_off_by_one_chunk_bound_is_caught():
    """Mutation: one bisection chunk fewer than n_vis needs must change the thresholds."""
    S, G, n, K, CH = 512, 65536, 4, 512, 32
    nvs = [8193, 8193, 8193, 8193]  # one group into the 3rd chunk
    bits_raw = _scores(3, n, S, False)
    ids = jnp.broadcast_to(jnp.arange(S)[:, None] * 128 + jnp.arange(128)[None], (n, S, 128))
    nv = jnp.asarray(nvs, I32).reshape(n, 1, 1)
    bits = jnp.where(ids < nv, bits_raw, -1)
    bits = bits.at[:, 64, 0].set(jnp.int32(0x7F000000))  # the last visible group: top score
    thr, tmax = ix.thresholds(bits, ids, nv, K, G)
    m = -(-max(nvs) // (CH * 128)) - 1

    def chunk(a, c):
        return jax.lax.dynamic_slice(a, (0, c * CH, 0), (n, CH, 128))

    thr2, tmax2 = ix.thresholds_chunks(
        lambda c: chunk(bits, c), lambda c: chunk(ids, c), m, nv, K, G, (n, CH, 128)
    )
    assert not (np.array_equal(thr2, thr) and np.array_equal(tmax2, tmax))


@pytest.mark.parametrize("nv,D,dense", [(5000, 1, 0.3), (25000, 2, 0.05), (9000, 2, 0.9),
                                         (16384, 1, 0.12), (0, 1, 0.5)])
def test_union_chunks_match_full(nv, D, dense):
    """`compact_union_chunk` over the visible rows (owner-masked for main-KV DCP) gives the
    same union ids, codes and count as `compact_union` over all rows."""
    from vllm_torchtpu.megakernel.wholestep import union_attn as ua

    S, KU, CM, gpp = 512, 2048, 64, 128
    rng = np.random.default_rng(nv + D)
    code = rng.integers(1, 16, (S, 128)) * (rng.random((S, 128)) < dense)
    gid = np.arange(S)[:, None] * 128 + np.arange(128)[None]
    code = np.where(gid < nv, code, 0).astype(np.int32)
    for r in range(D):
        own = code if D == 1 else np.where((gid // gpp) % D == r, code, 0)
        if (own > 0).sum() > KU:  # the union is at most T * budget wide
            keep = np.cumsum((own > 0).ravel()).reshape(S, 128) <= KU
            own = np.where(keep, own, 0)
        want = ua.compact_union(jnp.asarray(own), KU, ua.union_consts(S, KU))
        m, base = _plan(nv, CM, S, gpp, 1)
        out = {"ids": np.zeros((1, KU), np.int32), "codes": np.zeros((1, KU), np.int32)}

        def store(j0, ids, codes):
            W = ids.shape[1]
            out["ids"][:, j0:j0 + W] += np.asarray(ids)
            out["codes"][:, j0:j0 + W] += np.asarray(codes)

        tot = jnp.zeros((), jnp.int32)
        consts = ua.union_chunk_consts(2 * CM, KU)
        for i in range((m + 1) // 2):
            parts, bases = [], []
            for k in range(2):
                c = 2 * i + k
                bs = int(base(c)) if c < m else 0
                parts.append(own[bs:bs + CM] * (c < m))
                bases.append(bs)
            tot = _union_chunk_eager(jnp.asarray(np.concatenate(parts, 0)), tot, bases, CM,
                                     KU, consts, store)
        np.testing.assert_array_equal(out["ids"], want[0])
        np.testing.assert_array_equal(out["codes"], want[1])
        assert int(tot) == int(np.sum(want[2]))


def _union_chunk_eager(code, tot, bases, CM, KU, consts, store, row_of=None, id_of=None):
    """`compact_union_chunk` outside a kernel: `pl.when` as a Python condition."""
    from jax.experimental import pallas as pl

    from vllm_torchtpu.megakernel.wholestep import union_attn as ua

    orig = pl.when

    def when(cond):
        return (lambda f: f()) if bool(cond) else (lambda f: None)

    pl.when = when
    try:
        row_of = row_of or (lambda rj: jnp.where(rj < CM, bases[0] + rj, bases[1] + rj - CM))
        return ua.compact_union_chunk(code, tot, row_of, KU, consts, store, id_of)
    finally:
        pl.when = orig


def _lanepack_union(code, nv, r, gpp, KU, CM, S, rr=None):
    """`_nvis_union`'s lane-packed chunk loop outside a kernel (`jnp.roll` for `pltpu.roll`).
    `rr`: rank whose ids the packed lanes are mapped back with (negative control)."""
    from vllm_torchtpu.megakernel.wholestep import union_attn as ua

    rr = r if rr is None else rr
    H = CM // 2
    m, base = _plan(nv, CM, S, gpp, 1)
    out = {"ids": np.zeros((1, KU), np.int32), "codes": np.zeros((1, KU), np.int32)}

    def store(j0, ids, codes):
        W = ids.shape[1]
        out["ids"][:, j0:j0 + W] += np.asarray(ids)
        out["codes"][:, j0:j0 + W] += np.asarray(codes)

    tot = jnp.zeros((), jnp.int32)
    consts = ua.union_chunk_consts(2 * CM, KU)
    for i in range((m + 3) // 4):
        parts, bases = [], []
        for k in range(4):
            c = 4 * i + k
            bs = int(base(c)) if c < m else 0
            if gpp == 128:  # whole rows: rank r's are 2p + r
                parts.append(code[bs + r:bs + CM:2] * (c < m))
            else:
                ev = jnp.asarray(code[bs:bs + CM:2])
                od = jnp.asarray(code[bs + 1:bs + CM:2])
                parts.append(np.asarray(ua.lane_pack(ev, od, r, gpp, jnp.roll)) * (c < m))
            bases.append(bs)

        def row_of(rj, bases=bases):
            b = bases[3]
            for k in (2, 1, 0):
                b = jnp.where(rj < (k + 1) * H, bases[k], b)
            return b + 2 * jax.lax.rem(rj, H) + (rr if gpp == 128 else 0)

        tot = _union_chunk_eager(
            jnp.asarray(np.concatenate(parts, 0)), tot, bases, CM, KU, consts, store,
            row_of=row_of, id_of=None if gpp == 128 else (
                lambda rj, lj, row_of=row_of: ua.lane_pack_id(row_of(rj), lj, rr, gpp)),
        )
    return out["ids"], out["codes"], int(tot)


@pytest.mark.parametrize("nv,gpp,dense", [(25000, 32, 0.05), (50000, 32, 0.02),
                                          (9000, 32, 0.9), (16384, 16, 0.1),
                                          (40000, 64, 0.03), (0, 32, 0.5), (65536, 32, 0.004),
                                          (25000, 128, 0.05), (50000, 128, 0.02),
                                          (9000, 128, 0.9), (65536, 128, 0.004)])
def test_union_lanepack_matches_masked(nv, gpp, dense):
    """VLLM_TPU_MK_QSA_UNION_LANEPACK: each DCP2 pair rank's union over its lane-packed rows
    (half the chunks) equals its owner-masked union: same ids, codes, order and count."""
    from vllm_torchtpu.megakernel.wholestep import union_attn as ua

    S, KU, CM, D = 512, 2048, 64, 2
    rng = np.random.default_rng(nv + gpp)
    code = rng.integers(1, 16, (S, 128)) * (rng.random((S, 128)) < dense)
    gid = np.arange(S)[:, None] * 128 + np.arange(128)[None]
    code = np.where(gid < nv, code, 0).astype(np.int32)
    for r in range(D):
        own = np.where((gid // gpp) % D == r, code, 0)
        if (own > 0).sum() > KU:  # the union is at most T * budget wide
            keep = np.cumsum((own > 0).ravel()).reshape(S, 128) <= KU
            own = np.where(keep, own, 0)
            code = np.where((gid // gpp) % D == r, own, code)
        want = ua.compact_union(jnp.asarray(own), KU, ua.union_consts(S, KU))
        ids, codes, tot = _lanepack_union(code, nv, r, gpp, KU, CM, S)
        np.testing.assert_array_equal(ids, want[0])
        np.testing.assert_array_equal(codes, want[1])
        assert tot == int(np.sum(want[2]))


@pytest.mark.parametrize("gpp", [32, 128])
def test_union_lanepack_wrong_rank_ids_caught(gpp):
    """Negative control: mapping rank 1's packed rows back with rank 0's ids is caught."""
    from vllm_torchtpu.megakernel.wholestep import union_attn as ua

    S, KU, CM, nv = 512, 2048, 64, 25000
    rng = np.random.default_rng(7)
    gid = np.arange(S)[:, None] * 128 + np.arange(128)[None]
    code = rng.integers(1, 16, (S, 128)) * (rng.random((S, 128)) < 0.05)
    code = np.where(gid < nv, code, 0).astype(np.int32)
    own = np.where((gid // gpp) % 2 == 1, code, 0)
    want = ua.compact_union(jnp.asarray(own), KU, ua.union_consts(S, KU))
    ids, codes, _ = _lanepack_union(code, nv, 1, gpp, KU, CM, S, rr=0)
    np.testing.assert_array_equal(codes, want[1])  # same codes, same order
    assert not np.array_equal(ids, want[0])


@pytest.mark.parametrize("nv,gpp,dense", [(25000, 128, 0.05), (65536, 128, 0.004),
                                          (50000, 32, 0.02), (9000, 16, 0.9)])
def test_full_union_lanepack_matches_masked(nv, gpp, dense):
    """VLLM_TPU_MK_QSA_UNION_LANEPACK on the full-row union (`full_union`): `compact_union`
    over a pair rank's S / 2 packed rows, ids unpacked, equals its owner-masked union over S."""
    from vllm_torchtpu.megakernel.wholestep import union_attn as ua

    S, KU = 512, 2048
    rng = np.random.default_rng(nv + 3 * gpp)
    code = rng.integers(1, 16, (S, 128)) * (rng.random((S, 128)) < dense)
    gid = np.arange(S)[:, None] * 128 + np.arange(128)[None]
    code = np.where(gid < nv, code, 0).astype(np.int32)
    for r in range(2):
        own = np.where((gid // gpp) % 2 == r, code, 0)
        want = ua.compact_union(jnp.asarray(own), KU, ua.union_consts(S, KU))
        if gpp == 128:
            packed = jnp.asarray(code[r::2])
        else:
            packed = ua.lane_pack(jnp.asarray(code[0::2]), jnp.asarray(code[1::2]), r, gpp,
                                  jnp.roll)
        ids, codes, n = ua.compact_union(packed, KU, ua.union_consts(S // 2, KU))
        np.testing.assert_array_equal(ua.unpack_union_ids(ids, n, r, gpp), want[0])
        np.testing.assert_array_equal(codes, want[1])
        assert int(np.sum(n)) == int(np.sum(want[2]))
        if r == 1 and int(np.sum(n)):  # negative control: unpacked as the other rank
            assert not np.array_equal(ua.unpack_union_ids(ids, n, 0, gpp), want[0])
