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
"""q38 L2-UNPACK (X-UA-UNPACK, wholestep/union_unpack.py): the union tile body's strided full-vreg ring read is bitwise (CPU).

Part A, the reader (TPU interpret mode, one kernel): `union_unpack.unpack_kv(slot, 0)` against `qsa_attn.block._unpack_kv(slot,
0)` and a numpy reference (K = pack row 0, V = pack row 1 of every token), byte for byte, at a dynamic ring slot, on random
FP8 rings whose K and V rows are per-token permutations of all 256 byte codes (so every code, the e4m3fn NaN codes 0x7F / 0xFF
included, sits in both 128-lane halves; e4m3fn has no infinity encoding), with bytes 2 / 3 different from bytes 0 / 1 at
every position (an offset-1 read differs everywhere: the control). The offset guard accepts only the static 0.
Part B, the union tile body (one rank's union attention through `union_attn.attend_request`, test_l2_uaissue.py's walk at
D14 geometry on real 102K unions, offset 0): UNPACK=1 against UNPACK=0, bit for bit, on MM's walk and on X-UA-ISSUE level
2, at b16 / b32 / b48 / b64 DCP 4 and b32 DCP 1 with the whole step's 3 union ring slots (b32 DCP 4 also at 8), the reader
engaged (call count); and a pool of uniformly random byte codes (NaN K / V reach the scores and values; compared as f32 bit
patterns) at b32 DCP 4 / 1.
Part C, the whole step (test_wholestep.py's D08 S1 identity harness, every QSA layer's offset 0): `union_unpack` alone and
with X-UA-ISSUE / the S1 stages, DCP 1 / 2 / 4, bitwise MM's / D14's block, the reader engaged.
Part D, no kernel: block selection, scratch unchanged by the option, env plumbing, and the whole-step call's refusal of a
non-zero main-KV offset.
"""

from __future__ import annotations

import importlib.util
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.blocks.qsa_attn import block as qblock
from vllm_torchtpu.megakernel.ops import F32, I32
from vllm_torchtpu.megakernel.wholestep import union_attn as ua
from vllm_torchtpu.megakernel.wholestep import union_issue as ui
from vllm_torchtpu.megakernel.wholestep import union_unpack as uu
from vllm_torchtpu.megakernel.wholestep.qsa_full_b import UNION_SLOTS

HERE = os.path.dirname(os.path.abspath(__file__))
F8 = jnp.float8_e4m3fn
TILE = ua.TILE
NAN_CODES = (0x7F, 0xFF)  # e4m3fn: S.1111.111 (no infinities)


def _ut():
    """test_l2_uaissue.py, loaded by path (its D14 walk geometry and real-union requests)."""
    spec = importlib.util.spec_from_file_location("q38_unpack_ut", os.path.join(HERE, "test_l2_uaissue.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tw():
    """test_wholestep.py, loaded by path (its D08 S1 identity harness)."""
    spec = importlib.util.spec_from_file_location("q38_unpack_tw", os.path.join(HERE, "test_wholestep.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _counting(monkeypatch) -> list:
    """Wrap `union_unpack.unpack_kv` with a trace-time call counter (union_attn imports it when `unpack` is on)."""
    calls, real = [], uu.unpack_kv

    def wrapped(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(uu, "unpack_kv", wrapped)
    return calls


# ------------------------------------------------------------------------------------------------ Part A
def _ring_codes(seed: int, slots: int = 3) -> np.ndarray:
    """`[slots, TILE, 1, 4, 256]` uint8: rows 0 / 1 (K / V) are per-token permutations of all 256 codes; rows 2 / 3 are
    rows 0 / 1 XOR 0x55 / 0xAA (never equal to them)."""
    rng = np.random.default_rng(seed)
    codes = np.broadcast_to(np.arange(256, dtype=np.uint8), (TILE, 256))
    b = np.zeros((slots, TILE, 1, 4, 256), np.uint8)
    for s in range(slots):
        b[s, :, 0, 0] = rng.permuted(codes, axis=1)
        b[s, :, 0, 1] = rng.permuted(codes, axis=1)
        b[s, :, 0, 2] = b[s, :, 0, 0] ^ 0x55
        b[s, :, 0, 3] = b[s, :, 0, 1] ^ 0xAA
    return b


def _reader_run(ring_u8: np.ndarray, s: int) -> list:
    """Bytes `[TILE, 256]` of ring slot `s` (an SMEM scalar: a dynamic slot index, as the tile body's `ring.at[s]`):
    (K, V) from `unpack_kv(.., 0)`, from `_unpack_kv(.., 0)`, and from `_unpack_kv(.., 1)` (the control)."""
    readers = ((uu.unpack_kv, 0), (qblock._unpack_kv, 0), (qblock._unpack_kv, 1))

    def kernel(s_ref, ring, *outs):
        slot = ring.at[s_ref[0]]
        for i, (fn, off) in enumerate(readers):
            k, v = fn(slot, off, True)
            outs[2 * i][...] = jax.lax.bitcast_convert_type(k, jnp.uint8)
            outs[2 * i + 1][...] = jax.lax.bitcast_convert_type(v, jnp.uint8)

    S = jax.ShapeDtypeStruct((TILE, 256), jnp.uint8)
    f = pl.pallas_call(
        kernel,
        out_shape=(S,) * 6,
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM), pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=tuple(pl.BlockSpec(memory_space=pltpu.VMEM) for _ in range(6)),
        interpret=pltpu.InterpretParams(),
    )
    ring = jax.lax.bitcast_convert_type(jnp.asarray(ring_u8), F8)
    return [np.asarray(x) for x in f(jnp.asarray([s], jnp.int32), ring)]


@pytest.mark.parametrize("seed", [0, 1])
def test_unpack_reader_bytes_identical(seed):
    """`unpack_kv(slot, 0)` == `_unpack_kv(slot, 0)` == (pack row 0, pack row 1), byte for byte, at every slot of a
    3-slot ring (dynamic index); every code (NaN included) in both lane halves of K and V; the offset-1 read differs at
    every position (control: the offset is load-bearing)."""
    ring = _ring_codes(seed)
    for row in (0, 1):  # coverage: every code in lanes 0-127 and in lanes 128-255 of K and of V
        for lo in (0, 128):
            assert len(np.unique(ring[:, :, 0, row, lo : lo + 128])) == 256, (row, lo)
    for s in range(ring.shape[0]):
        k1, v1, k0, v0, k01, v01 = _reader_run(ring, s)
        np.testing.assert_array_equal(k0, ring[s, :, 0, 0], err_msg=f"slot {s}: _unpack_kv K vs pack row 0")
        np.testing.assert_array_equal(v0, ring[s, :, 0, 1], err_msg=f"slot {s}: _unpack_kv V vs pack row 1")
        np.testing.assert_array_equal(k1, k0, err_msg=f"slot {s}: unpack_kv K")
        np.testing.assert_array_equal(v1, v0, err_msg=f"slot {s}: unpack_kv V")
        assert (k01 != k1).all() and (v01 != v1).all(), f"slot {s}: the offset-1 control does not differ"
        nan_pos = int(np.isin(k1, NAN_CODES).sum() + np.isin(v1, NAN_CODES).sum())
        assert nan_pos == 4 * TILE, nan_pos  # two NaN codes per token row, in K and in V
        print(f"L2-UNPACK READER seed {seed} slot {s}: K / V bytes identical to _unpack_kv(.., 0) and to pack rows 0 / 1 "
              f"({nan_pos} NaN-code bytes); offset-1 control differs at all {2 * TILE * 256} positions", flush=True)


def test_unpack_offset_guard():
    """Only the static offset 0: 1, a bool, a numpy int and a traced value are refused by `check_offset`, by `unpack_kv`
    (at trace time, in a kernel) and by `attend_request(unpack=True)`."""
    uu.check_offset(0)
    for bad in (1, 2, True, np.int32(0), 0.0):
        with pytest.raises(AssertionError):
            uu.check_offset(bad)
    with pytest.raises(AssertionError):
        jax.jit(lambda o: (uu.check_offset(o), o)[1])(jnp.int32(0))
    with pytest.raises(AssertionError):
        _reader_offset(1)
    ut = _ut()
    rows, D, rk, edges = 16, 4, 1, ("one",)
    d = ut._requests(rows // ut.T, D, rk, seed=3, edges=edges)
    with pytest.raises(AssertionError):
        _walk(ut, d, "base", D, unpack=True, off=1)


def _reader_offset(off):
    def kernel(ring, k_o, v_o):
        k, v = uu.unpack_kv(ring.at[0], off, True)
        k_o[...] = jax.lax.bitcast_convert_type(k, jnp.uint8)
        v_o[...] = jax.lax.bitcast_convert_type(v, jnp.uint8)

    S = jax.ShapeDtypeStruct((TILE, 256), jnp.uint8)
    f = pl.pallas_call(kernel, out_shape=(S, S), in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
                       out_specs=(pl.BlockSpec(memory_space=pltpu.VMEM),) * 2, interpret=pltpu.InterpretParams())
    return f(jax.lax.bitcast_convert_type(jnp.asarray(_ring_codes(5, 1)), F8))


# ------------------------------------------------------------------------------------------------ Part B
def _walk(ut, d: dict, arm: str, D: int, *, unpack: bool, off: int = 0, slots: int = UNION_SLOTS):
    """One rank's union attention over every request (test_l2_uaissue's walk, with the offset and the UNPACK switch):
    arm 'base' = MM's walk (table lookup per copy), 'ab' = X-UA-ISSUE level 2. Returns (attn, lse) as numpy."""
    nreq, T, NH, KU, PPS = d["n"].shape[0], ut.T, ut.NH, ut.KU, ut.PPS
    W = ui.slab_width(PPS, D)
    cols = ui.slab_cols(ut.MAX_GROUPS, ut.PER_PAGE, D, PPS)
    issue = arm == "ab"

    def kernel(n_ref, mtab, pool, uid, ucd, qv, gv, o_ref, l_ref, ring, sems, row, tabw, slab, pv, msem):
        if issue:
            cp = ui.slab_copy(mtab, 0, cols, slab, msem.at[0])
            cp.start()
            cp.wait()
            ui.prep(nreq=nreq, union_ids=lambda q: uid.at[q], n_union=lambda q: n_ref[q], slab=slab, cols=cols,
                    per_page=ut.PER_PAGE, dcp=D, pv=pv)
        else:
            cp = pltpu.make_async_copy(mtab.at[0], tabw, msem.at[0])
            cp.start()
            cp.wait()

        def req(q, carry):
            n_u = n_ref[q]
            src = pv.at[pl.ds(q, 1)] if issue else uid.at[q]
            c = pltpu.make_async_copy(src, row.at[pl.ds(0, 1), pl.ds(0, KU)], msem.at[2])
            c.start()
            c.wait()
            row[0, KU] = jnp.int32(0)
            ok = [jnp.zeros((1, 8), jnp.bool_) for _ in range(T)]
            res = ua.attend_request(
                T=T, nh=NH, pool=pool, ring=ring, sems=sems, ids=row, codes=ucd.at[q], n_union=n_u,
                table=tabw, table_row=q, qs=[qv[q, t] for t in range(T)], gates=[gv[q, t] for t in range(T)],
                has=[n_u > 0 for _ in range(T)], off=off, scale_k=0.375, scale_v=2.5, sm_scale=256 ** -0.5,
                interpret=True, recent=(jnp.zeros((8, 256), F32), jnp.zeros((8, 256), F32), ok,
                                        jnp.zeros((8, 1), jnp.bool_)),
                phys=issue, dcp=D, **({"fast": True} if issue else {}), **({"unpack": True} if unpack else {}),
            )
            for t in range(T):
                if D > 1:
                    att, lse = res[t]
                    o_ref[q, t] = att
                    l_ref[q, t] = jnp.broadcast_to(lse, (NH, 128))
                else:
                    o_ref[q, t] = res[t]
            return carry

        jax.lax.fori_loop(0, nreq, req, 0)

    smem, vmem = pltpu.SMEM, pltpu.VMEM
    f = pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct((nreq, T, NH, 256), F32), jax.ShapeDtypeStruct((nreq, T, NH, 128), F32)),
        in_specs=[pl.BlockSpec(memory_space=smem), pl.BlockSpec(memory_space=pl.ANY),
                  pl.BlockSpec(memory_space=pl.ANY)] + [pl.BlockSpec(memory_space=vmem)] * 4,
        out_specs=tuple(pl.BlockSpec(memory_space=vmem) for _ in range(2)),
        scratch_shapes=[
            vmem((slots, TILE, 1, 4, 256), F8), pltpu.SemaphoreType.DMA((slots,)), smem((1, KU + 128), I32),
            smem((nreq, PPS), I32), vmem((nreq, W), I32), vmem((nreq, KU), I32), pltpu.SemaphoreType.DMA((3,)),
        ],
        interpret=pltpu.InterpretParams(),
    )
    o, lse = f(jnp.asarray(d["n"]), jnp.asarray(d["mtab"]), d["pool"], jnp.asarray(d["uid"]), jnp.asarray(d["ucd"]),
               jnp.asarray(d["q"]), jnp.asarray(d["g"]))
    return np.asarray(o), np.asarray(lse)


def _bits_same(run, base, tag: str, finite: bool = True) -> None:
    """Bitwise on the f32 bit patterns (NaN payloads included)."""
    (o, lse), (ob, lb) = run, base
    if finite:
        assert np.isfinite(ob).all() and np.isfinite(o).all(), tag
    np.testing.assert_array_equal(o.view(np.uint32), ob.view(np.uint32), err_msg=f"{tag}: attention bits")
    np.testing.assert_array_equal(lse.view(np.uint32), lb.view(np.uint32), err_msg=f"{tag}: lse bits")


# (rows, DCP, rank, edge requests, ring slots): b16 / b32 / b48 / b64 at DCP 4 on different ranks, b32 at DCP 1.
GEOMS = {
    "b16_d4": (16, 4, 1, ("one", "full", "top"), UNION_SLOTS),
    "b32_d4": (32, 4, 3, ("top", "idle"), UNION_SLOTS),
    "b48_d4": (48, 4, 0, ("full",), UNION_SLOTS),
    "b64_d4": (64, 4, 2, ("one",), UNION_SLOTS),
    "b32_d1": (32, 1, 0, ("one",), UNION_SLOTS),
    "b32_d4_s8": (32, 4, 3, ("top", "idle"), 8),
}


@pytest.mark.parametrize("geom", list(GEOMS))
def test_unpack_walk_bit_identical(monkeypatch, geom):
    """UNPACK=1 vs UNPACK=0 (offset 0), bit for bit, on MM's walk and on X-UA-ISSUE level 2, at the geometry's real
    unions; the strided reader is engaged on the UNPACK=1 runs only."""
    assert UNION_SLOTS == 3, UNION_SLOTS
    ut = _ut()
    rows, D, rk, edges, slots = GEOMS[geom]
    d = ut._requests(rows // ut.T, D, rk, seed=rows + D + 100, edges=edges)
    ut._cover(d, geom)
    n = d["n"]
    assert ((n % 64) != 0).any() and (n > 64 * 3).sum() >= 1, n.tolist()
    calls = _counting(monkeypatch)
    for arm in ("base", "ab"):
        ref = _walk(ut, d, arm, D, unpack=False, slots=slots)
        assert not calls, (geom, arm, "the strided reader ran with UNPACK=0")
        got = _walk(ut, d, arm, D, unpack=True, slots=slots)
        assert calls, (geom, arm, "the strided reader did not run with UNPACK=1")
        calls.clear()
        _bits_same(got, ref, f"{geom} {arm}")
        print(f"L2-UNPACK WALK {geom} slots {slots} {arm}: UNPACK=1 bitwise identical to UNPACK=0", flush=True)


@pytest.mark.parametrize("geom", ["b32_d4", "b32_d1"])
def test_unpack_walk_all_codes_pool(geom):
    """A pool of uniformly random bytes (every FP8 code, NaN included, in K and V): UNPACK=1 vs UNPACK=0 on MM's walk and
    on level 2, compared as f32 bit patterns (non-finite outputs allowed and required to match bit for bit)."""
    ut = _ut()
    rows, D, rk, edges, slots = GEOMS[geom]
    d = ut._requests(rows // ut.T, D, rk, seed=rows + D + 200, edges=edges)
    rng = np.random.default_rng(rows + D)
    pool_u8 = rng.integers(0, 256, d["pool"].shape, dtype=np.uint8)
    d["pool"] = jax.lax.bitcast_convert_type(jnp.asarray(pool_u8), F8)
    assert np.isin(pool_u8[..., 0:2, :], NAN_CODES).any()
    for arm in ("base", "ab"):
        ref = _walk(ut, d, arm, D, unpack=False, slots=slots)
        got = _walk(ut, d, arm, D, unpack=True, slots=slots)
        _bits_same(got, ref, f"codes {geom} {arm}", finite=False)
        print(f"L2-UNPACK CODES {geom} {arm}: UNPACK=1 bitwise identical to UNPACK=0 (non-finite outputs "
              f"{int((~np.isfinite(ref[0])).sum())} of {ref[0].size})", flush=True)


# ------------------------------------------------------------------------------------------------ Part C
# (scenario of test_wholestep.S1_SCEN, S1 config or None, X-UA-ISSUE level); `union_unpack` on in every case.
WHOLE = [
    ("d4_r8_4", None, 0), ("d4_r8_4", "ALL", 2), ("d4_r4_64", "ALL", 2),
    ("d2_r2_16", "ALL", 2), ("d1_r2_8", None, 0), ("d1_r2_8", "B3", 1),
]


@pytest.mark.parametrize(("scen", "cfg", "lvl"), WHOLE, ids=[f"{s}-{c}-L{v}" for s, c, v in WHOLE])
def test_unpack_wholestep_identity(monkeypatch, scen, cfg, lvl):
    """The whole step with `union_unpack` (on the S1 config and X-UA-ISSUE level, or alone), every QSA layer at offset 0,
    is bitwise MM's / D14's block on the scenario (test_wholestep._s1_same); the strided reader is engaged."""
    tw = _tw()
    orig = tw.wi.random_wholestep_inputs
    monkeypatch.setattr(tw.wi, "random_wholestep_inputs", lambda *a, **k: orig(*a, **dict(k, off=0)))
    kw = tw.S1_SCEN[scen]
    D, KU_ = kw["dcp"], tw._s1_ku(kw["budget"])
    base = tw._s1_run(monkeypatch, scen, None)
    assert int(np.asarray(base[8]["off"]).max()) == 0
    tw._s1_cover(base, scen)
    calls = _counting(monkeypatch)
    lean = dict(tw.S1_CONFIGS[cfg]) if cfg else {}
    run = _s1_run_unpack(tw, monkeypatch, scen, lean, lvl)
    assert calls, (scen, cfg, lvl, "the strided reader did not run")
    tw._s1_same(run, base, D, KU_, f"{scen} {cfg} L{lvl} unpack")
    print(f"L2-UNPACK WHOLE {scen} S1={cfg} issue {lvl} unpack: bitwise identical to MM's block ({len(calls)} reader "
          f"traces)", flush=True)


def _s1_run_unpack(tw, monkeypatch, scen: str, lean: dict, lvl: int):
    """test_wholestep._s1_run with the block options `union_unpack` (and `union_issue` when lvl > 0) added."""
    kw = dict(tw.S1_SCEN[scen])
    dcp, ties, ring = kw.pop("dcp"), kw.pop("ties", False), kw.pop("ring", False)
    if dcp > 1:
        monkeypatch.setenv(tw._s1_pool_layout.MAIN_KV_DCP_ENV, str(dcp))
    else:
        monkeypatch.delenv(tw._s1_pool_layout.MAIN_KV_DCP_ENV, raising=False)
    opts = dict(tw.S1_PROBE, **lean, **({"union_issue": lvl} if lvl else {}), union_unpack=True)
    kw.setdefault("shard", True)
    return tw._run_c(long=True, rank_ids=True, union=True, shared_kv=dcp > 1, dcp=dcp, lean=opts,
                     ck_fn=tw._s1_ties if ties else None, ring_fn=tw._s1_ring if ring else None, **kw)


# ------------------------------------------------------------------------------------------------ Part D
def test_unpack_block_selection_scratch_env_and_guard(monkeypatch):
    """No kernel run. `union_unpack` selects the lean class (alone, or with X-UA-ISSUE / S1) and adds no scratch at b16 /
    32 / 48 / 64; the env maps to the option (merged with X-UA-ISSUE's and the S1 options), keys nothing when unset, and is
    refused without union attention; the whole-step call's slot check refuses a non-zero main-KV offset only when the option
    is on."""
    from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
    from vllm_torchtpu.megakernel.models import qwen38
    from vllm_torchtpu.megakernel.wholestep import qsa_full_lean as ql
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc

    MG, PPS, BUDGET = 65536, 65536 * 4 // 128, 512
    base = dict(max_groups=MG, pages_per_seq=PPS, carrier="fp8", n_layers=11, hbm_tables=True,
                per_request_scores=True, bounded_chunks=True, shard_index=True, union_attn=True, loop_requests=True)
    qsa = dict(budget=BUDGET)
    assert ql.lean_step_block(dict(base, dcp=4), qsa) is None
    blk = ql.lean_step_block(dict(base, dcp=4, union_unpack=True), qsa)
    assert type(blk) is ql.QSAFullStepGroupLeanBlock and blk.union_unpack and not blk.union_issue and not blk.slot_batch
    blk1 = ql.lean_step_block(dict(base, dcp=1, union_unpack=True), qsa)
    assert type(blk1) is ql.QSAFullStepLeanBlock and blk1.union_unpack
    full = dict(base, dcp=4, union_issue=2, union_bounded=True, union_lean=True, slot_batch=True)
    on, off = ql.lean_step_block(dict(full, union_unpack=True), qsa), ql.lean_step_block(full, qsa)
    assert on.union_unpack and not off.union_unpack and on.union_issue == 2 and on.slot_batch
    g = qwen38.FULL
    for B in (16, 32, 48, 64):
        ex = ExecSpec(R=8, B=B, T=4, ctx=128, tiles=Tiles(bank=1 << 20, nbd=10, nbe=8, ebank=512 << 10, vmem_mib=64))
        sig = lambda b: [(s.name, repr(s.spec)) for s in b.scratch(g, ex, None)]  # noqa: E731
        assert sig(on) == sig(off), B
    for e in ("VLLM_TPU_MK_UNION_UNPACK", "VLLM_TPU_MK_UNION_ISSUE", "VLLM_TPU_MK_UNION_BOUNDED", "VLLM_TPU_MK_UNION_LEAN",
              "VLLM_TPU_MK_SLOT_BATCH"):
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setattr(wsc, "UNION_ATTN", True)
    monkeypatch.setattr(wsc, "UNION_PHYS", False)
    paired = [(1, 0), (0, 1), (1, 0)]
    unpaired = [(-1, 0), (0, 1), (-1, 0)]
    assert wsc._uaunpack_options() == {} and wsc._d08s1_options() == {}
    assert wsc._uaunpack_slots(paired) == paired  # option off: no check, list unchanged
    monkeypatch.setenv("VLLM_TPU_MK_UNION_UNPACK", "1")
    assert wsc._uaunpack_options() == {"union_unpack": True}
    assert wsc._d08s1_options() == {"union_unpack": True}  # S1 and X-UA-ISSUE off: the option alone
    assert wsc._uaunpack_slots(unpaired) == unpaired
    with pytest.raises(ValueError, match="non-zero main-KV offset"):
        wsc._uaunpack_slots(paired)
    monkeypatch.setenv("VLLM_TPU_MK_UNION_ISSUE", "2")
    assert wsc._d08s1_options() == {"union_issue": 2, "union_unpack": True}
    monkeypatch.setenv("VLLM_TPU_MK_SLOT_BATCH", "1")
    o = wsc._d08s1_options()
    assert o["union_unpack"] is True and o["union_issue"] == 2 and o["slot_batch"] is True
    monkeypatch.setattr(wsc, "UNION_ATTN", False)
    monkeypatch.delenv("VLLM_TPU_MK_UNION_ISSUE")
    with pytest.raises(ValueError):
        wsc._uaunpack_options()
