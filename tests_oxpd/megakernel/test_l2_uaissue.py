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
"""q38 L2-UAISSUE (X-UA-ISSUE, wholestep/union_issue.py): exactness of the union walk's cheap DMA descriptors (CPU).

Part A, the walk at D14 geometry (TPU interpret mode, one rank's union attention through `union_attn.attend_request`):
T = 4 verify rows, the DCP group's 12 heads (3 per rank x 4), 512-token pool pages (per_page 128), main-KV DCP 4
ownership (page `g // 128` on rank `page % 4`), max_groups 65536 (the 262K cap: pages_per_seq 2048, slab 128 of 512
planned columns), 3 ring slots (qsa_full_b.UNION_SLOTS, the whole step's union ring: tile t + 2 issues inside tile t's
loop iteration) and 8 (the block default: up to 7 tiles issue before the loop). Requests per rank 4 / 8 / 12 / 16 = b16 / b32 / b48 / b64. Unions are real 102K
selections (scratch/attn/sel_xover_pref, the four rows of a request = four consecutive positions, this rank's owned
groups: 4.75-5.50 tiles per request-layer on average) plus pinned edge requests: a last tile with 1 live slot, a full
last tile, an idle slot (union 0) and a request whose union reaches the last slab column (ids up to 65535, so the lane
compare's bound is load-bearing). Four arms must agree bit for bit (partials and LSE at DCP > 1, gated rows at DCP 1):
  base = MM's walk (table lookup per copy), phys = MM's `wholephys` (per-request table-row DMA, compare over all 2048
  columns), a = level 1 (slab + bounded compare, union_issue.prep), ab = level 2 (+ F0a issue: predicated copies, flat
  rows on device, byte-sized wait).
Part B, the whole step (the D08 S1 identity harness of test_wholestep.py): the block option `union_issue` 1 / 2 with
D08 S1 off / B3 / all on, DCP 1 / 2 / 4, bitwise MM's block (outputs and the union state attention reads).
Part C, no kernel: the slab bounds, the block selection, the env plumbing and the scratch.
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

from vllm_torchtpu.megakernel.ops import F32, I32
from vllm_torchtpu.megakernel.wholestep.qsa_full_b import UNION_SLOTS
from vllm_torchtpu.megakernel.wholestep import union_attn as ua
from vllm_torchtpu.megakernel.wholestep import union_issue as ui

HERE = os.path.dirname(os.path.abspath(__file__))
SEL_DIR = os.environ.get("Q38_UAISSUE_SEL", os.path.join(HERE, "data_uaissue"))
SEL_FILES = ("x102k_blocks_q38_sel_p26864_c00085.npz", "x102k_code_q38_sel_p775_c00169.npz")
F8 = jnp.float8_e4m3fn
T, NH, PT, NP, SLOTS, BUDGET = 4, 12, 512, 24, UNION_SLOTS, 512  # rows, heads (3 x DCP 4), page tokens, pool pages, ring, K
PER_PAGE = PT // ua.MICRO
MAX_GROUPS = 65536
PPS = MAX_GROUPS * 4 // 128
KU = -(-(T * BUDGET) // 128) * 128
ARMS = ("base", "phys", "a", "ab")
SENTINEL = 0x5A  # ring bytes before the walk (an ordinary finite FP8 value)


# ------------------------------------------------------------------------------------------------ Part A inputs
def _sel_rows():
    out = []
    for f in SEL_FILES:
        p = os.path.join(SEL_DIR, f)
        assert os.path.exists(p), f"real selection dump missing: {p} (set Q38_UAISSUE_SEL)"
        out.append(np.load(p, allow_pickle=False)["blocks"])
    return out


def _union(groups_per_row: list, D: int, rk: int):
    """Ascending union of this rank's owned groups and their row codes (bit t: row t selected it)."""
    code = {}
    for t, gs in enumerate(groups_per_row):
        for g in np.asarray(gs).tolist():
            if ((g // PER_PAGE) % D) == rk:
                code[g] = code.get(g, 0) | (1 << t)
    ids = np.array(sorted(code), np.int32)
    return ids, np.array([code[g] for g in ids.tolist()], np.int32)


def _owned(rng, D: int, rk: int, lo: int, hi: int, n: int):
    """`n` distinct owned groups in [lo, hi), ascending."""
    pool = np.arange(lo, hi, dtype=np.int64)
    pool = pool[((pool // PER_PAGE) % D) == rk]
    return np.sort(rng.choice(pool, n, replace=False)).astype(np.int32)


def _edge(kind: str, rng, D: int, rk: int):
    """Pinned edge unions: 'one' (last tile 1 live slot), 'full' (full last tile), 'idle' (0), 'top' (reaches the
    last slab column: ids up to MAX_GROUPS - 1)."""
    if kind == "idle":
        return np.zeros(0, np.int32), np.zeros(0, np.int32)
    if kind == "top":
        ids = np.concatenate([_owned(rng, D, rk, 0, 1 << 14, 150), _owned(rng, D, rk, MAX_GROUPS - 4096, MAX_GROUPS, 150)])
        hi = MAX_GROUPS - PER_PAGE * D + rk * PER_PAGE + PER_PAGE - 1  # the last owned group of the last column
        ids = np.unique(np.concatenate([ids, [hi]])).astype(np.int32)
    else:
        n = 64 * 4 + 1 if kind == "one" else 64 * 5
        ids = _owned(rng, D, rk, 0, 25600, n)
    codes = rng.integers(1, 1 << T, len(ids)).astype(np.int32)
    return ids, codes


def _requests(nreq: int, D: int, rk: int, seed: int, edges: tuple = ()):
    rng = np.random.default_rng(seed)
    sel = _sel_rows()
    reqs = []
    for q in range(nreq):
        if q < len(edges):
            reqs.append(_edge(edges[q], rng, D, rk))
            continue
        b = sel[q % len(sel)]
        r0 = int(rng.integers(0, len(b) - T))
        if D == 1:  # DCP 1: thin the real selection so the union stays under KU
            reqs.append(_union([b[r0 + t][: BUDGET // 4] for t in range(T)], 1, 0))
        else:
            reqs.append(_union([b[r0 + t] for t in range(T)], D, rk))
    n = np.array([len(i) for i, _ in reqs], np.int32)
    assert n.max() <= KU, n
    uid = np.zeros((nreq, 1, KU), np.int32)
    ucd = np.zeros((nreq, 1, KU), np.int32)
    for q, (i, c) in enumerate(reqs):
        uid[q, 0, : len(i)], ucd[q, 0, : len(c)] = i, c
    mtab = rng.integers(0, NP, (1, nreq, PPS)).astype(np.int32)  # every column live: a wrong column reads other data
    pool = (rng.standard_normal((NP, PT, 1, 4, 256)) * 2.0).astype(np.float32)
    q_ = (rng.standard_normal((nreq, T, NH, 256)) * 0.2).astype(np.float32)
    g_ = rng.standard_normal((nreq, T, NH, 256)).astype(np.float32)
    return dict(n=n, uid=uid, ucd=ucd, mtab=mtab, pool=jnp.asarray(pool, F8), q=q_, g=g_)


# ------------------------------------------------------------------------------------------------ Part A kernel
def _walk(d: dict, arm: str, D: int, mode: str = "on_wait", fill: bool = False, slots: int = SLOTS):
    """One rank's union attention over every request in a request loop (`fori_loop`, as the block), arm `arm`."""
    nreq = d["n"].shape[0]
    W = ui.slab_width(PPS, D)
    cols = ui.slab_cols(MAX_GROUPS, PER_PAGE, D, PPS)
    phys, fast = arm != "base", arm == "ab"

    def kernel(n_ref, mtab, pool, uid, ucd, qv, gv, o_ref, l_ref, r_ref, ring, sems, row, tabw, qtab, qpv, slab, pv, msem):
        if fill:
            ring[...] = jax.lax.bitcast_convert_type(jnp.full(ring.shape, SENTINEL, jnp.uint8), F8)
        if arm == "base":  # the SMEM table window (`load_tables`)
            cp = pltpu.make_async_copy(mtab.at[0], tabw, msem.at[0])
            cp.start()
            cp.wait()
        if arm in ("a", "ab"):  # level 1: the layer's slab, then every request's physical ids
            cp = ui.slab_copy(mtab, 0, cols, slab, msem.at[0])
            cp.start()
            cp.wait()
            ui.prep(nreq=nreq, union_ids=lambda q: uid.at[q], n_union=lambda q: n_ref[q], slab=slab, cols=cols,
                    per_page=PER_PAGE, dcp=D, pv=pv)

        def req(q, carry):
            n_u = n_ref[q]
            if arm == "phys":  # MM's wholephys: the request's table row, compare over all pps columns
                cp = pltpu.make_async_copy(mtab.at[0, pl.ds(q, 1)], qtab, msem.at[1])
                cp.start()
                cp.wait()
                ui.phys_ids(uid.at[q], qtab[...], n_u, PER_PAGE, D, qpv)
                src = qpv
            elif arm in ("a", "ab"):
                src = pv.at[pl.ds(q, 1)]
            else:
                src = uid.at[q]
            cs = [pltpu.make_async_copy(src, row.at[pl.ds(0, 1), pl.ds(0, KU)], msem.at[2])]
            for c in cs:
                c.start()
            for c in cs:
                c.wait()
            row[0, KU] = jnp.int32(0)
            ok = [jnp.zeros((1, 8), jnp.bool_) for _ in range(T)]
            res = ua.attend_request(
                T=T, nh=NH, pool=pool, ring=ring, sems=sems, ids=row, codes=ucd.at[q], n_union=n_u,
                table=tabw, table_row=q, qs=[qv[q, t] for t in range(T)], gates=[gv[q, t] for t in range(T)],
                has=[n_u > 0 for _ in range(T)], off=1, scale_k=0.375, scale_v=2.5, sm_scale=256 ** -0.5,
                interpret=True, recent=(jnp.zeros((8, 256), F32), jnp.zeros((8, 256), F32), ok,
                                        jnp.zeros((8, 1), jnp.bool_)),
                phys=phys, dcp=D, **({"fast": True} if fast else {}),
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
        r_ref[...] = ring[...]

    smem, vmem, anyspace = pltpu.SMEM, pltpu.VMEM, pl.ANY
    f = pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct((nreq, T, NH, 256), F32), jax.ShapeDtypeStruct((nreq, T, NH, 128), F32),
                   jax.ShapeDtypeStruct((slots, ua.TILE, 1, 4, 256), F8)),
        in_specs=[pl.BlockSpec(memory_space=smem), pl.BlockSpec(memory_space=anyspace),
                  pl.BlockSpec(memory_space=anyspace)] + [pl.BlockSpec(memory_space=vmem)] * 4,
        out_specs=tuple(pl.BlockSpec(memory_space=vmem) for _ in range(3)),
        scratch_shapes=[
            vmem((slots, ua.TILE, 1, 4, 256), F8), pltpu.SemaphoreType.DMA((slots,)), smem((1, KU + 128), I32),
            smem((nreq, PPS), I32), vmem((1, PPS), I32), vmem((1, KU), I32), vmem((nreq, W), I32),
            vmem((nreq, KU), I32), pltpu.SemaphoreType.DMA((3,)),
        ],
        interpret=pltpu.InterpretParams(dma_execution_mode=mode),
    )
    o, lse, ringd = f(jnp.asarray(d["n"]), jnp.asarray(d["mtab"]), d["pool"], jnp.asarray(d["uid"]),
                      jnp.asarray(d["ucd"]), jnp.asarray(d["q"]), jnp.asarray(d["g"]))
    return np.asarray(o), np.asarray(lse), np.asarray(jax.lax.bitcast_convert_type(ringd, jnp.uint8))


def _cover(d: dict, tag: str) -> None:
    n = d["n"]
    tiles = -(-n // 64)
    last = n - 64 * np.maximum(tiles - 1, 0)
    print(f"L2-UAISSUE COVER {tag}: requests {len(n)} union n {n.tolist()} tiles {tiles.tolist()} last-tile live "
          f"{last.tolist()} (partial {int(((last > 0) & (last < 64)).sum())}, full {int((last == 64).sum())}, idle "
          f"{int((n == 0).sum())}); max id {int(d['uid'].max())}", flush=True)


def _same(run, base, tag: str) -> None:
    o, lse, _ = run
    ob, lb, _ = base
    assert np.isfinite(ob).all() and np.isfinite(o).all(), tag
    np.testing.assert_array_equal(o, ob, err_msg=f"{tag}: attention")
    np.testing.assert_array_equal(lse, lb, err_msg=f"{tag}: lse")


# (rows, DCP, rank, edge requests): b16 / b32 / b48 / b64 at DCP 4 on different ranks; DCP 2 and 1 for the shifts.
GEOMS = {
    "b16_d4": (16, 4, 1, ("one", "full", "top")),
    "b32_d4": (32, 4, 3, ("top", "idle")),
    "b48_d4": (48, 4, 0, ("full",)),
    "b64_d4": (64, 4, 2, ("one",)),
    "b16_d2": (16, 2, 1, ("top",)),
    "b16_d1": (16, 1, 0, ("one",)),
}


@pytest.mark.parametrize("slots", [UNION_SLOTS, 8])
@pytest.mark.parametrize("geom", list(GEOMS))
def test_uaissue_walk_bit_identical(geom, slots):
    """The four arms (module docstring) give bit-identical outputs on the geometry's requests, with the whole step's 3
    union ring slots (steady-state starts inside the tile loop, slots reused every 3 tiles) and with 8; the coverage line
    is printed and its point asserted (real unions of 4-6 tiles, partial last tiles, the edge requests)."""
    assert UNION_SLOTS == 3, UNION_SLOTS
    rows, D, rk, edges = GEOMS[geom]
    d = _requests(rows // T, D, rk, seed=rows + D, edges=edges)
    _cover(d, geom)
    n = d["n"]
    assert ((n % 64) != 0).any() and (n > 64 * 3).sum() >= max(1, len(n) - len(edges) - 1), n.tolist()
    if "top" in edges:  # the bound is load-bearing: a union id sits in the slab's last column
        cshift = PER_PAGE.bit_length() - 1 + D.bit_length() - 1
        assert int(d["uid"].max()) >> cshift == ui.slab_cols(MAX_GROUPS, PER_PAGE, D, PPS) - 1
    base = _walk(d, "base", D, slots=slots)
    for arm in ARMS[1:]:
        _same(_walk(d, arm, D, slots=slots), base, f"{geom} slots {slots} {arm}")
        print(f"L2-UAISSUE WALK {geom} slots {slots} {arm}: bitwise identical to MM's walk", flush=True)


def test_uaissue_walk_dma_modes():
    """Level 2 under both TPU-interpret DMA execution modes (copies complete at wait / at start): the byte-sized wait
    and the predicated copies leave no race the eager order would hide (b32, DCP 4, the whole step's 3 ring slots)."""
    rows, D, rk, edges = GEOMS["b32_d4"]
    d = _requests(rows // T, D, rk, seed=rows + D, edges=edges)
    base = _walk(d, "base", D)
    for mode in ("on_wait", "eager"):
        _same(_walk(d, "ab", D, mode=mode), base, f"ab {mode}")
        print(f"L2-UAISSUE MODES ab {mode}: bitwise identical", flush=True)


def test_uaissue_last_tile_moves_no_padded_rows():
    """Level 2 copies only the union's slots: one request of 4 tiles + 1 slot, ring pre-filled with a sentinel byte.
    After the walk the last tile's slot holds data in its first 4 token rows and the sentinel in the other 252 under
    level 2, while MM's walk (and level 1) fill all 256 rows (its 63 padded copies). 8 ring slots, so tile 4's slot is
    fresh (with the whole step's 3 slots it would hold tile 1's rows; exactness at 3 slots is the walk test's)."""
    D = 4
    d = _requests(1, D, 1, seed=7, edges=("one",))
    assert int(d["n"][0]) == 64 * 4 + 1
    slot = 4  # tile 4 -> ring slot 4 (the ring restarts per request)
    got = {arm: _walk(d, arm, D, fill=True, slots=8) for arm in ("base", "a", "ab")}
    for arm, (o, lse, ring) in got.items():
        live, pad = ring[slot, : ua.MICRO], ring[slot, ua.MICRO:]
        assert (live != SENTINEL).any(), arm
        if arm == "ab":
            assert (pad == SENTINEL).all(), "level 2 wrote a padded ring row"
        else:
            assert (pad != SENTINEL).any(axis=(1, 2, 3)).all(), f"{arm}: padded rows not copied as MM's walk does"
    _same(got["ab"], got["base"], "sentinel ab")
    print("L2-UAISSUE PADDED: level 2 leaves 252 padded rows unwritten; MM's walk and level 1 write all 256", flush=True)


# ------------------------------------------------------------------------------------------------ Part C
def test_uaissue_slab_bounds():
    """The slab columns: exactly the columns a union id can address (128 at D14: max_groups 65536, 512-token pages,
    DCP 4), within the planned width (pps / D = 512), and every id < max_groups lands inside."""
    assert ui.slab_cols(65536, 128, 4, 2048) == 128 and ui.slab_width(2048, 4) == 512
    assert ui.slab_cols(65536, 128, 2, 2048) == 256 and ui.slab_cols(65536, 128, 1, 2048) == 512
    assert ui.slab_cols(1024, 32, 4, 32) == 32 and ui.slab_width(32, 4) == 32  # the whole-step tests' geometry
    for mg, pp, dcp, pps in ((65536, 128, 4, 2048), (32768, 32, 2, 1024), (1024, 32, 1, 32), (4096, 4, 4, 256)):
        cols, shift = ui.slab_cols(mg, pp, dcp, pps), pp.bit_length() - 1 + dcp.bit_length() - 1
        assert ((mg - 1) >> shift) < cols <= pps, (mg, pp, dcp, pps)
        if pp >= 32:
            assert cols <= ui.slab_width(pps, dcp)
    with pytest.raises(AssertionError):  # a table narrower than max_groups needs is refused, not silently bounded
        ui.slab_cols(4096, 4, 4, 128)


def test_uaissue_block_selection_and_scratch(monkeypatch):
    """No kernel run. `union_issue` selects the lean class (with every S1 stage off) and adds the slab / ids scratch
    at b64 DCP 4 (16 x 512 and 16 x 2048 int32); absent, MM's / D14's block; refused with wholephys, without union
    attention, or at an unknown level. The env maps to the option, keys nothing when unset, and refuses misuse."""
    from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
    from vllm_torchtpu.megakernel.models import qwen38
    from vllm_torchtpu.megakernel.wholestep import qsa_full_lean as ql
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc

    base = dict(max_groups=MAX_GROUPS, pages_per_seq=PPS, carrier="fp8", n_layers=11, hbm_tables=True,
                per_request_scores=True, bounded_chunks=True, shard_index=True, union_attn=True, loop_requests=True)
    qsa = dict(budget=BUDGET)
    assert ql.lean_step_block(dict(base, dcp=4), qsa) is None
    blk = ql.lean_step_block(dict(base, dcp=4, union_issue=2), qsa)
    assert type(blk) is ql.QSAFullStepGroupLeanBlock and blk.union_issue == 2 and not blk.slot_batch
    blk1 = ql.lean_step_block(dict(base, dcp=4, union_issue=1, slot_batch=True), qsa)
    assert blk1.union_issue == 1 and blk1.slot_batch and blk1._bu
    for bad in (dict(base, dcp=4, union_issue=3), dict(base, dcp=4, union_issue=1, union_phys=True)):
        with pytest.raises(AssertionError):
            ql.lean_step_block(bad, qsa)
    g = qwen38.FULL
    ex = ExecSpec(R=8, B=64, T=4, ctx=128, tiles=Tiles(bank=1 << 20, nbd=10, nbe=8, ebank=512 << 10, vmem_mib=64))
    names = {s.name: s for s in blk.scratch(g, ex, None)}
    assert names["qx_slab"].spec.shape == (16, 512) and names["qx_pv"].spec.shape == (16, KU), names["qx_slab"]
    assert "qx_slab" not in {s.name for s in ql.lean_step_block(dict(base, dcp=4, union_bounded=True), qsa).scratch(g, ex, None)}
    monkeypatch.delenv("VLLM_TPU_MK_UNION_ISSUE", raising=False)
    monkeypatch.setattr(wsc, "UNION_ATTN", True)
    monkeypatch.setattr(wsc, "UNION_PHYS", False)
    assert wsc._uaissue_options() == {}
    for lvl in ("1", "2"):
        monkeypatch.setenv("VLLM_TPU_MK_UNION_ISSUE", lvl)
        assert wsc._uaissue_options() == {"union_issue": int(lvl)}
        assert wsc._d08s1_options() == {"union_issue": int(lvl)}  # S1 off: the issue option alone
    monkeypatch.setenv("VLLM_TPU_MK_SLOT_BATCH", "1")
    assert wsc._d08s1_options()["union_issue"] == 2 and wsc._d08s1_options()["slot_batch"] is True
    monkeypatch.delenv("VLLM_TPU_MK_SLOT_BATCH")
    for env, attr in (("3", None), ("1", "UNION_PHYS"), ("1", "UNION_ATTN")):
        monkeypatch.setenv("VLLM_TPU_MK_UNION_ISSUE", env)
        if attr == "UNION_PHYS":
            monkeypatch.setattr(wsc, "UNION_PHYS", True)
        if attr == "UNION_ATTN":
            monkeypatch.setattr(wsc, "UNION_PHYS", False)
            monkeypatch.setattr(wsc, "UNION_ATTN", False)
        with pytest.raises(ValueError):
            wsc._uaissue_options()


# ------------------------------------------------------------------------------------------------ Part B
def _tw():
    """test_wholestep.py, loaded by path (its D08 S1 identity harness)."""
    spec = importlib.util.spec_from_file_location("q38_uaissue_tw", os.path.join(HERE, "test_wholestep.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# (scenario of test_wholestep.S1_SCEN, S1 config or None = MM's / D14's block, level)
WHOLE = [
    ("d4_r8_4", "ALL", 2), ("d4_r8_4", None, 1), ("d4_r8_4", "B3", 2),
    ("d2_r2_16", "ALL", 2), ("d1_r2_8", None, 2), ("d1_r2_8", "B3", 1),
    ("d4_r4_64", "ALL", 2),
]


@pytest.mark.parametrize(("scen", "cfg", "lvl"), WHOLE, ids=[f"{s}-{c}-L{v}" for s, c, v in WHOLE])
def test_uaissue_wholestep_identity(monkeypatch, scen, cfg, lvl):
    """The whole step with `union_issue` = lvl (on the S1 config, or alone) is bitwise MM's / D14's block on the
    scenario: residual, ck_new, ring_new, index queries, kr_new / ir_new, union stats and the union state
    attention reads (test_wholestep._s1_same)."""
    tw = _tw()
    kw = tw.S1_SCEN[scen]
    D, KU_ = kw["dcp"], tw._s1_ku(kw["budget"])
    base = tw._s1_run(monkeypatch, scen, None)
    tw._s1_cover(base, scen)
    lean = dict(tw.S1_CONFIGS[cfg]) if cfg else {}
    run = _s1_run_issue(tw, monkeypatch, scen, lean, lvl)
    tw._s1_same(run, base, D, KU_, f"{scen} {cfg} L{lvl}")
    print(f"L2-UAISSUE WHOLE {scen} S1={cfg} level {lvl}: bitwise identical to MM's block", flush=True)


def _s1_run_issue(tw, monkeypatch, scen: str, lean: dict, lvl: int):
    """test_wholestep._s1_run with the block option `union_issue` added to the lean options."""
    kw = dict(tw.S1_SCEN[scen])
    dcp, ties, ring = kw.pop("dcp"), kw.pop("ties", False), kw.pop("ring", False)
    if dcp > 1:
        monkeypatch.setenv(tw._s1_pool_layout.MAIN_KV_DCP_ENV, str(dcp))
    else:
        monkeypatch.delenv(tw._s1_pool_layout.MAIN_KV_DCP_ENV, raising=False)
    opts = dict(tw.S1_PROBE, **lean, union_issue=lvl)
    kw.setdefault("shard", True)
    return tw._run_c(long=True, rank_ids=True, union=True, shared_kv=dcp > 1, dcp=dcp, lean=opts,
                     ck_fn=tw._s1_ties if ties else None, ring_fn=tw._s1_ring if ring else None, **kw)
