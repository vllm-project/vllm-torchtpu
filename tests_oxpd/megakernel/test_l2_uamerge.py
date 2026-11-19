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
"""q38 L2-UAMERGE (q38-l2-uamerge): X-UA-ISSUE, X-UA-UNPACK and X-UA-PIPE (ii) composed, bitwise (CPU, TPU interpret mode).

Arms are (ISSUE level, UNPACK, PIPE mode); (0, 0, 0) is MM's walk (`union_attn.attend_request`, table lookup per copy).
Part A, the walk at D14 geometry (test_l2_uaissue.py's requests: T = 4 rows, the DCP group's 12 heads, 512-token FP8 pages,
max_groups 65536, real 102K unions plus pinned edge requests; offset 0, the head pair X-UA-UNPACK reads), with the whole
step's 3 union ring slots: every arm of {ISSUE 0 / 2} x {UNPACK 0 / 1} x {PIPE 0 / 2} against (0, 0, 0), bit for bit
(attention partials and LSE at DCP 4, gated rows at DCP 1, compared as f32 bit patterns), at b16 / b32 / b48 / b64 DCP 4
and b32 DCP 1. PIPE 2 drives `attend_request_pipe` through `pipe_request_loop` (three SMEM id rows, the ring position
carried across requests) as the whole step's `_PipeMixin` does; PIPE 0 drives `attend_request` per request. Engagement is
asserted per run: the strided reader ran iff UNPACK, `attend_request_pipe` (and the cross-request loop) iff PIPE, and
`attend_request` iff not PIPE. ISSUE level 1 inside the pipe is covered at b32 DCP 4. The same arms with every DMA landing
at its start must drain every DMA semaphore at kernel exit, and X-UA-ISSUE stage (b) inside the pipe moves no padded rows.
Part B, the whole step (test_wholestep.py's D08 S1 identity harness, every QSA layer at offset 0): `union_pipe` 2 with
`union_issue` / `union_unpack` on the S1 set, DCP 1 / 2 / 4, bitwise MM's / D14's block, engagement asserted.
Part C, no kernel: block selection, scratch, env plumbing and the flag-off option set.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.ops import F32, I32
from vllm_torchtpu.megakernel.wholestep import union_attn as ua
from vllm_torchtpu.megakernel.wholestep import union_issue as ui
from vllm_torchtpu.megakernel.wholestep import union_unpack as uu
from vllm_torchtpu.megakernel.wholestep.qsa_full_b import UNION_SLOTS

HERE = os.path.dirname(os.path.abspath(__file__))
F8 = jnp.float8_e4m3fn
TILE = ua.TILE
SENTINEL = 0x5A  # ring bytes before the walk (an ordinary finite FP8 value)
COMBOS = [(i, u, p) for i in (0, 2) for u in (0, 1) for p in (0, 2)]  # (ISSUE level, UNPACK, PIPE mode)
BASE = (0, 0, 0)
ARMS = [c for c in COMBOS if c != BASE]


def _tag(c: tuple) -> str:
    return f"i{c[0]}u{c[1]}p{c[2]}"


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ut():
    """test_l2_uaissue.py, loaded by path (its D14 walk geometry and real-union requests)."""
    return _load("q38_uamerge_ut", "test_l2_uaissue.py")


def _tw():
    """test_wholestep.py, loaded by path (its D08 S1 identity harness and L2-UAPIPE's scenarios)."""
    return _load("q38_uamerge_tw", "test_wholestep.py")


def _counters(monkeypatch) -> dict:
    """Trace-time call counters: the strided reader, `attend_request`, `attend_request_pipe`, and cross-request loops."""
    calls = {"reader": 0, "base": 0, "pipe": 0, "xloop": 0}
    real_r, real_b, real_p, real_l = uu.unpack_kv, ua.attend_request, ua.attend_request_pipe, ua.pipe_request_loop

    def r(*a, **k):
        calls["reader"] += 1
        return real_r(*a, **k)

    def b(**k):
        calls["base"] += 1
        return real_b(**k)

    def p(**k):
        calls["pipe"] += 1
        return real_p(**k)

    def lp(**k):
        calls["xloop"] += bool(k["pipe"] & ua.PIPE_XREQ)
        return real_l(**k)

    monkeypatch.setattr(uu, "unpack_kv", r)
    monkeypatch.setattr(ua, "attend_request", b)
    monkeypatch.setattr(ua, "attend_request_pipe", p)
    monkeypatch.setattr(ua, "pipe_request_loop", lp)
    return calls


def _engaged(calls: dict, combo: tuple, tag: str) -> None:
    _, unpack, pipe = combo
    assert (calls["reader"] > 0) == bool(unpack), (tag, "strided reader engagement", calls)
    assert (calls["pipe"] > 0) == bool(pipe) and (calls["base"] > 0) == (not pipe), (tag, "walk engagement", calls)
    assert (calls["xloop"] > 0) == bool(pipe & ua.PIPE_XREQ), (tag, "cross-request loop engagement", calls)
    for k in calls:
        calls[k] = 0


# ------------------------------------------------------------------------------------------------ Part A
def _walk(ut, d: dict, D: int, combo: tuple, *, slots: int = UNION_SLOTS, mode: str = "on_wait", fill: bool = False,
          off: int = 0):
    """One rank's union attention over every request of `d`, arm `combo` = (ISSUE level, UNPACK, PIPE mode). Returns
    (attn, lse, ring bytes) as numpy."""
    issue, unpack, pipe = combo
    nreq, T, NH, KU, PPS = d["n"].shape[0], ut.T, ut.NH, ut.KU, ut.PPS
    W = ui.slab_width(PPS, D)
    cols = ui.slab_cols(ut.MAX_GROUPS, ut.PER_PAGE, D, PPS)
    xreq = bool(pipe & ua.PIPE_XREQ)
    flags = dict(phys=bool(issue))
    if issue >= 2:
        flags["fast"] = True
    if unpack:
        flags["unpack"] = True

    def kernel(n_ref, mtab, pool, uid, ucd, qv, gv, qiz, o_ref, l_ref, r_ref, ring, sems, row, rows3, xsem, tabw, slab, pv,
               msem):
        if fill:
            ring[...] = jax.lax.bitcast_convert_type(jnp.full(ring.shape, SENTINEL, jnp.uint8), F8)
        if issue:  # X-UA-ISSUE stage (a): the layer's slab, then every request's physical ids
            cp = ui.slab_copy(mtab, 0, cols, slab, msem.at[0])
            cp.start()
            cp.wait()
            ui.prep(nreq=nreq, union_ids=lambda q: uid.at[q], n_union=lambda q: n_ref[q], slab=slab, cols=cols,
                    per_page=ut.PER_PAGE, dcp=D, pv=pv)
        else:  # the SMEM table window (`load_tables`)
            cp = pltpu.make_async_copy(mtab.at[0], tabw, msem.at[0])
            cp.start()
            cp.wait()

        def src(q):
            return pv.at[pl.ds(q, 1)] if issue else uid.at[q]

        def body(q, ctl, carry, ids=None):
            n_u = n_ref[q]
            ok = [jnp.zeros((1, 8), jnp.bool_) for _ in range(T)]
            common = dict(
                T=T, nh=NH, pool=pool, ring=ring, sems=sems, codes=ucd.at[q], n_union=n_u, table=tabw, table_row=q,
                qs=[qv[q, t] for t in range(T)], gates=[gv[q, t] for t in range(T)], has=[n_u > 0 for _ in range(T)],
                off=off, scale_k=0.375, scale_v=2.5, sm_scale=256 ** -0.5, interpret=True,
                recent=(jnp.zeros((8, 256), F32), jnp.zeros((8, 256), F32), ok, jnp.zeros((8, 1), jnp.bool_)), dcp=D,
                **flags,
            )
            if not pipe:
                res, nt = ua.attend_request(ids=ids, **common), 0
            elif xreq:
                q1 = jnp.minimum(q + 1, nreq - 1)
                res, nt = ua.attend_request_pipe(
                    ids=rows3, ids_row=ctl["ids_row"], pipe=pipe, base=ctl["base"], inflight=ctl["inflight"],
                    nxt=ua.PipeNext(ids_row=ctl["next_row"], n_union=n_ref[q1], table_row=q1, has=ctl["has_next"]),
                    **common,
                )
            else:
                res, nt = ua.attend_request_pipe(ids=ids, ids_row=0, pipe=pipe, **common)
            for t in range(T):
                if D > 1:
                    att, lse = res[t]
                    o_ref[q, t] = att
                    l_ref[q, t] = jnp.broadcast_to(lse, (NH, 128))
                else:
                    o_ref[q, t] = res[t]
            return carry, nt

        if xreq:  # the whole step's `_PipeMixin`: three id rows, one semaphore each, ring position carried

            def ids_copies(q, k):
                return ua.pipe_ids_copies(src(q), qiz, rows3, xsem, k)

            ua.pipe_request_loop(nreq=nreq, nslots=slots, pipe=pipe, ids_copies=ids_copies, body=body,
                                 init=jnp.int32(0))
        else:

            def req(q, carry):
                cs = ua.pipe_ids_copies(src(q), qiz, row, xsem, 0)  # the id row and its zero tail
                for c in cs:
                    c.start()
                for c in cs:
                    c.wait()
                return body(q, None, carry, ids=row)[0]

            jax.lax.fori_loop(0, nreq, req, jnp.int32(0))
        r_ref[...] = ring[...]

    smem, vmem = pltpu.SMEM, pltpu.VMEM
    f = pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct((nreq, T, NH, 256), F32), jax.ShapeDtypeStruct((nreq, T, NH, 128), F32),
                   jax.ShapeDtypeStruct((slots, TILE, 1, 4, 256), F8)),
        in_specs=[pl.BlockSpec(memory_space=smem), pl.BlockSpec(memory_space=pl.ANY),
                  pl.BlockSpec(memory_space=pl.ANY)] + [pl.BlockSpec(memory_space=vmem)] * 5,
        out_specs=tuple(pl.BlockSpec(memory_space=vmem) for _ in range(3)),
        scratch_shapes=[
            vmem((slots, TILE, 1, 4, 256), F8), pltpu.SemaphoreType.DMA((slots,)), smem((1, KU + 128), I32),
            smem((ua.PIPE_IDS_BUFS, KU + 128), I32), pltpu.SemaphoreType.DMA((ua.PIPE_IDS_BUFS,)),
            smem((nreq, PPS), I32), vmem((nreq, W), I32), vmem((nreq, KU), I32), pltpu.SemaphoreType.DMA((2,)),
        ],
        interpret=pltpu.InterpretParams(dma_execution_mode=mode),
    )
    o, lse, ringd = f(jnp.asarray(d["n"]), jnp.asarray(d["mtab"]), d["pool"], jnp.asarray(d["uid"]), jnp.asarray(d["ucd"]),
                      jnp.asarray(d["q"]), jnp.asarray(d["g"]), jnp.zeros((1, 128), jnp.int32))
    return np.asarray(o), np.asarray(lse), np.asarray(jax.lax.bitcast_convert_type(ringd, jnp.uint8))


def _walk_eager(ut, d: dict, D: int, combo: tuple, **kw):
    """`_walk` with every DMA executed at its start; returns (attn, lse, ring, the interpreter's 'non-zero count' lines:
    a copy started but never waited over-signals its semaphore and is reported at kernel exit)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        o, lse, ring = _walk(ut, d, D, combo, mode="eager", **kw)
    text = buf.getvalue()
    sys.stdout.write(text)
    return o, lse, ring, [x for x in text.splitlines() if "non-zero count" in x]


def _bits_same(run, base, tag: str) -> None:
    """Bitwise on the f32 bit patterns (attention and LSE); the base must be finite and not vacuous."""
    o, lse = run[0], run[1]
    ob, lb = base[0], base[1]
    assert np.isfinite(ob).all() and np.isfinite(o).all(), tag
    assert np.abs(ob).max() > 0.01, (tag, "vacuous base")
    np.testing.assert_array_equal(o.view(np.uint32), ob.view(np.uint32), err_msg=f"{tag}: attention bits")
    np.testing.assert_array_equal(lse.view(np.uint32), lb.view(np.uint32), err_msg=f"{tag}: lse bits")


# (rows, DCP, rank, edge requests): b16 / b32 / b48 / b64 at DCP 4 on different ranks, b32 at DCP 1 (L2-UNPACK's set).
GEOMS = {
    "b16_d4": (16, 4, 1, ("one", "full", "top")),
    "b32_d4": (32, 4, 3, ("top", "idle")),
    "b48_d4": (48, 4, 0, ("full",)),
    "b64_d4": (64, 4, 2, ("one",)),
    "b32_d1": (32, 1, 0, ("one",)),
}
_REQ_CACHE: dict = {}


def _requests(ut, geom: str):
    rows, D, rk, edges = GEOMS[geom]
    d = ut._requests(rows // ut.T, D, rk, seed=rows + D + 300, edges=edges)
    n = d["n"]
    assert ((n % 64) != 0).any() and (n > 64 * 3).sum() >= 1, n.tolist()
    return d, D


WALK = [(g, c) for g in GEOMS for c in ARMS]


@pytest.mark.parametrize(("geom", "combo"), WALK, ids=[f"{g}-{_tag(c)}" for g, c in WALK])
def test_uamerge_walk_bit_identical(monkeypatch, geom, combo):
    """Arm `combo` vs MM's walk (0, 0, 0), bit for bit, at the geometry's real unions with the whole step's 3 union ring
    slots; the coverage line is printed and the engagement of each part asserted on both runs."""
    assert UNION_SLOTS == 3, UNION_SLOTS
    ut = _ut()
    d, D = _requests(ut, geom)
    ut._cover(d, f"uamerge {geom}")
    calls = _counters(monkeypatch)
    base = _walk(ut, d, D, BASE)
    _engaged(calls, BASE, f"{geom} base")
    got = _walk(ut, d, D, combo)
    _engaged(calls, combo, f"{geom} {_tag(combo)}")
    _bits_same(got, base, f"{geom} {_tag(combo)}")
    print(f"L2-UAMERGE WALK {geom} slots {UNION_SLOTS} {_tag(combo)}: bitwise identical to MM's walk (engaged: "
          f"reader={bool(combo[1])} pipe={combo[2]} issue={combo[0]})", flush=True)


@pytest.mark.parametrize("combo", [(1, 0, 2), (1, 1, 2)], ids=["i1u0p2", "i1u1p2"])
def test_uamerge_walk_issue1_in_pipe(monkeypatch, combo):
    """X-UA-ISSUE level 1 (physical ids, MM's 64 copies and whole-slot wait) inside the pipe, with and without UNPACK,
    bitwise MM's walk at b32 DCP 4."""
    ut = _ut()
    d, D = _requests(ut, "b32_d4")
    calls = _counters(monkeypatch)
    base = _walk(ut, d, D, BASE)
    _engaged(calls, BASE, "b32_d4 base")
    got = _walk(ut, d, D, combo)
    _engaged(calls, combo, f"b32_d4 {_tag(combo)}")
    _bits_same(got, base, f"b32_d4 {_tag(combo)}")
    print(f"L2-UAMERGE WALK b32_d4 slots {UNION_SLOTS} {_tag(combo)}: bitwise identical to MM's walk", flush=True)


@pytest.mark.parametrize("combo", [c for c in ARMS if c[2]] + [(2, 1, 0)], ids=lambda c: _tag(c))
def test_uamerge_walk_eager_drained(combo):
    """The arm with every DMA landing at its start (b32 DCP 4, 3 slots): bitwise MM's walk, and no DMA semaphore is
    left over-signaled at kernel exit (every started copy is waited by exactly its bytes)."""
    ut = _ut()
    d, D = _requests(ut, "b32_d4")
    base = _walk(ut, d, D, BASE)
    o, lse, _, left = _walk_eager(ut, d, D, combo)
    assert not left, (_tag(combo), "semaphores not drained at kernel exit", left[:4])
    _bits_same((o, lse), base, f"eager {_tag(combo)}")
    print(f"L2-UAMERGE EAGER b32_d4 {_tag(combo)}: bitwise identical, every DMA semaphore drained at kernel exit", flush=True)


def test_uamerge_eager_base_control():
    """Controls of the eager leg: MM's walk drains every semaphore with eager DMAs and its outputs do not depend on
    the DMA mode."""
    ut = _ut()
    d, D = _requests(ut, "b32_d4")
    base = _walk(ut, d, D, BASE)
    o, lse, _, left = _walk_eager(ut, d, D, BASE)
    assert not left, left[:4]
    _bits_same((o, lse), base, "eager base")
    print("L2-UAMERGE CONTROL b32_d4: MM's walk drains every semaphore with eager DMAs; outputs independent of the mode",
          flush=True)


def test_uamerge_pipe_last_tile_moves_no_padded_rows():
    """X-UA-ISSUE stage (b) inside the pipe copies only the union's slots: one request of 4 tiles + 1 slot, ring
    pre-filled with a sentinel byte, 8 ring slots (tile 4 sits in slot 4: the request is the layer's first, ring
    position 0), every DMA landing at its start. Under (2, u, 2) the slot holds data in its first 4 token rows and the
    sentinel in the other 252, and every semaphore drains; MM's walk fills all 256 rows (its 63 padded copies)."""
    ut = _ut()
    D = 4
    d = ut._requests(1, D, 1, seed=7, edges=("one",))
    assert int(d["n"][0]) == 64 * 4 + 1
    slot = 4
    got = {c: _walk_eager(ut, d, D, c, slots=8, fill=True) for c in (BASE, (2, 0, 2), (2, 1, 2), (2, 0, 0))}
    for c, (o, lse, ring, left) in got.items():
        assert not left, (_tag(c), left[:4])
        live, pad = ring[slot, : ua.MICRO], ring[slot, ua.MICRO:]
        assert (live != SENTINEL).any(), _tag(c)
        if c[0] == 2:
            assert (pad == SENTINEL).all(), f"{_tag(c)}: X-UA-ISSUE stage (b) wrote a padded ring row"
        else:
            assert (pad != SENTINEL).any(axis=(1, 2, 3)).all(), "MM's walk does not copy the padded rows"
        _bits_same(got[c], got[BASE], f"sentinel {_tag(c)}")
    print("L2-UAMERGE PADDED: X-UA-ISSUE (b) inside the pipe leaves 252 padded rows unwritten (i2u0p2, i2u1p2, i2u0p0); "
          "MM's walk writes all 256", flush=True)


def test_uamerge_pipe_offset_guard():
    """X-UA-UNPACK inside the pipe takes only the static offset 0 (refused at trace time otherwise)."""
    ut = _ut()
    d, D = _requests(ut, "b16_d4")
    for combo in ((0, 1, 2), (2, 1, 2)):
        with pytest.raises(AssertionError):
            _walk(ut, d, D, combo, off=1)
    o, lse, _ = _walk(ut, d, D, (2, 0, 2), off=1)  # control: without UNPACK the pipe reads head pair 1
    assert np.isfinite(o).all()
    print("L2-UAMERGE OFFSET: the pipe with X-UA-UNPACK refuses offset 1; without it offset 1 runs", flush=True)


# ------------------------------------------------------------------------------------------------ Part B
# (scenario of test_wholestep.UAPIPE_SCEN, S1 config, ISSUE level, UNPACK, PIPE mode)
WHOLE = [
    ("d4_r4_16_b512", "ALL", 2, 1, 2), ("d4_r4_16_b512", "ALL", 2, 0, 2), ("d4_r4_16_b512", "ALL", 0, 1, 2),
    ("d4_r4_64", "ALL", 2, 1, 2), ("d2_r2_16", "ALL", 2, 1, 2), ("d1_r2_8", "ALL", 2, 1, 2), ("d4_r8_4", "ALL", 1, 1, 2),
]


def _whole_run(tw, monkeypatch, scen: str, cfg: str | None, lvl: int, unpack: int, pipe: int):
    """test_wholestep._s1_run on UAPIPE_SCEN with the block options added (none: MM's / D14's block with the probe)."""
    kw = dict(tw.UAPIPE_SCEN[scen])
    dcp, ties, ring = kw.pop("dcp"), kw.pop("ties", False), kw.pop("ring", False)
    if dcp > 1:
        monkeypatch.setenv(tw._s1_pool_layout.MAIN_KV_DCP_ENV, str(dcp))
    else:
        monkeypatch.delenv(tw._s1_pool_layout.MAIN_KV_DCP_ENV, raising=False)
    lean = dict(tw.S1_PROBE, **(tw.S1_CONFIGS[cfg] if cfg else {}))
    if lvl:
        lean["union_issue"] = lvl
    if unpack:
        lean["union_unpack"] = True
    if pipe:
        lean["union_pipe"] = pipe
    kw.setdefault("shard", True)
    return tw._run_c(long=True, rank_ids=True, union=True, shared_kv=dcp > 1, dcp=dcp, lean=lean,
                     ck_fn=tw._s1_ties if ties else None, ring_fn=tw._s1_ring if ring else None, **kw)


@pytest.mark.parametrize(("scen", "cfg", "lvl", "unpack", "pipe"), WHOLE,
                         ids=[f"{s}-{c}-i{v}u{u}p{p}" for s, c, v, u, p in WHOLE])
def test_uamerge_wholestep_identity(monkeypatch, scen, cfg, lvl, unpack, pipe):
    """The whole step with the composed options on the S1 set, every QSA layer at offset 0, is bitwise MM's / D14's
    block on the scenario (test_wholestep._s1_same: outputs and the union state attention reads); engagement asserted."""
    tw = _tw()
    orig = tw.wi.random_wholestep_inputs
    monkeypatch.setattr(tw.wi, "random_wholestep_inputs", lambda *a, **k: orig(*a, **dict(k, off=0)))
    kw = tw.UAPIPE_SCEN[scen]
    D, KU_ = kw["dcp"], tw._s1_ku(kw["budget"])
    calls = _counters(monkeypatch)
    base = _whole_run(tw, monkeypatch, scen, None, 0, 0, 0)
    assert int(np.asarray(base[8]["off"]).max()) == 0
    _engaged(calls, BASE, f"{scen} base")
    run = _whole_run(tw, monkeypatch, scen, cfg, lvl, unpack, pipe)
    eng = dict(calls)
    _engaged(calls, (lvl, unpack, pipe), f"{scen} i{lvl}u{unpack}p{pipe}")
    n = tw._s1_union(base, D, KU_)[0]
    tiles = [-(-int(n[r, o, q, r % D]) * 4 // 256) for r in range(n.shape[0]) for o in range(n.shape[1])
             for q in range(n.shape[2])]
    print(f"L2-UAMERGE WCOVER {scen} {cfg} i{lvl}u{unpack}p{pipe}: per-rank tiles per request-layer: empty "
          f"{tiles.count(0)}, 1-tile {tiles.count(1)}, >= 2 {sum(1 for x in tiles if x >= 2)}, max {max(tiles)}; "
          f"calls {eng}", flush=True)
    tw._s1_same(run, base, D, KU_, f"{scen} {cfg} i{lvl}u{unpack}p{pipe}")
    print(f"L2-UAMERGE WHOLE {scen} S1={cfg} issue {lvl} unpack {unpack} pipe {pipe}: bitwise identical to MM's block",
          flush=True)


# ------------------------------------------------------------------------------------------------ Part C
def test_uamerge_block_selection_scratch_env(monkeypatch):
    """No kernel run. With `union_pipe` and the X-UA-ISSUE / X-UA-UNPACK options the pipe class is built on the lean
    class with both options live; (ii)'s three id rows replace B3's two and X-UA-ISSUE's slab / physical ids stay, and
    UNPACK adds no scratch; wholephys and (ii) without B3's ids stay refused. The env maps to the three options; with
    every flag unset no option is present and `wholestep_spec` builds MM's / D14's block (flag-off option set)."""
    from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
    from vllm_torchtpu.megakernel.models import qwen38
    from vllm_torchtpu.megakernel.wholestep import qsa_full_lean as ql
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc

    base = dict(max_groups=65536, pages_per_seq=2048, carrier="fp8", n_layers=11, hbm_tables=True,
                per_request_scores=True, bounded_chunks=True, shard_index=True, union_attn=True, loop_requests=True)
    qsa = dict(budget=512)
    full = dict(slot_batch=True, union_bounded=True, union_lean=True)
    for dcp, cls in ((1, ql.QSAFullStepPipeBlock), (2, ql.QSAFullStepPipeBlock), (4, ql.QSAFullStepGroupPipeBlock)):
        blk = ql.pipe_step_block(dict(base, dcp=dcp, **full, union_pipe=2, union_issue=2, union_unpack=True), qsa)
        assert type(blk) is cls and blk.union_pipe == 2 and blk.union_issue == 2 and blk.union_unpack and blk._bu
    g = qwen38.FULL
    for B in (16, 32, 48, 64):
        ex = ExecSpec(R=8, B=B, T=4, ctx=128, tiles=Tiles(bank=1 << 20, nbd=10, nbe=8, ebank=512 << 10, vmem_mib=64))

        def names(**kw):
            blk = ql.pipe_step_block(dict(base, dcp=4, **kw), qsa) or ql.lean_step_block(dict(base, dcp=4, **kw), qsa)
            return {s.name: repr(s.spec) for s in blk.scratch(g, ex, None)}

        lean_i = names(**full, union_issue=2)
        piu = names(**full, union_issue=2, union_unpack=True, union_pipe=2)
        assert set(piu) == (set(lean_i) - {"qu_rowb", "qb_usem"}) | {"qx_rows", "qx_usem"}, sorted(set(piu) ^ set(lean_i))
        assert {"qx_slab", "qx_pv", "qx_sem"} <= set(piu)
        assert piu == names(**full, union_issue=2, union_pipe=2), B  # UNPACK adds no scratch
    for bad in (dict(base, dcp=4, union_pipe=2, union_issue=2), dict(base, dcp=1, union_phys=True, union_pipe=1),
                dict(base, dcp=4, **full, union_pipe=2, union_issue=3)):
        with pytest.raises((AssertionError, ValueError)):
            ql.pipe_step_block(bad, qsa)
    envs = ("VLLM_TPU_MK_UNION_PIPE", "VLLM_TPU_MK_UNION_ISSUE", "VLLM_TPU_MK_UNION_UNPACK", "VLLM_TPU_MK_UNION_BOUNDED",
            "VLLM_TPU_MK_UNION_LEAN", "VLLM_TPU_MK_SLOT_BATCH")
    for e in envs:
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setattr(wsc, "UNION_ATTN", True)
    monkeypatch.setattr(wsc, "UNION_PHYS", False)
    off = wsc.indexed_options()
    assert not {"union_pipe", "union_issue", "union_unpack", "slot_batch"} & set(off), sorted(off)
    assert ql.pipe_step_block(off, qsa) is None and ql.lean_step_block(off, qsa) is None
    for e, v in zip(envs, ("2", "2", "1", "1", "1", "1")):
        monkeypatch.setenv(e, v)
    on = wsc.indexed_options()
    assert on["union_pipe"] == 2 and on["union_issue"] == 2 and on["union_unpack"] is True and on["slot_batch"] is True
    print(f"L2-UAMERGE SELECT: pipe class with X-UA-ISSUE 2 + X-UA-UNPACK at DCP 1 / 2 / 4; flag-off options "
          f"{sorted(k for k in off if k.startswith('union_'))}", flush=True)
