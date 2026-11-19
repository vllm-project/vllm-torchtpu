# SPDX-License-Identifier: Apache-2.0
"""q38 L1-L47F (L47-FOLD-S1, VLLM_TPU_MK_L47_FOLD, default off): model layer 47 folded into the whole step.

Kernel (CPU interpret): the whole step at L 5 with `first="qsa"` (QSA, GDN, GDN, GDN, QSA: the 3..47 kernel's partial last
cycle, 2 QSA layers) with the D08 S1 lean block and `l47_sink`, at main-KV DCP 1 / 2 / 4 and 8 - 64 rows:
  * closeness of the residual to the stock-semantics reference (`wholestep.inputs.wholestep_reference`, L 5) within the
    bound stated in scratch/overflow/l1_l47fold_status.md BEFORE the run: (B1) every rank, every row: mean |kernel - ref| <
    3e-2 k scale (dongruizi a18f2ea44's L 5 bound; k = sqrt(2 L / 8), scale = mean |ref|); (B2) every rank, the rows of
    every select-all request (all rows n_vis <= K): test_wholestep `_assert_within_budget(layers=5)` (mean 1e-2 k, q999
    6e-2 k, max 1e-1 k of scale). Control: the same kernel output against the L 4 reference (no layer 47) must FAIL B2.
  * the sink: the kernel's export (`qsa_l47_sink_ids`), gathered over the ranks and expanded by `qwen38_l47fold.wrap_call`
    (the served wrapper, run on the CPU mesh), equals on EVERY rank, row by row and column by column, the stock `selected`
    layout (`qsa_rpa.selected_tokens`) of the stock selection from the kernel's own last-layer queries and completed-group
    key (`indexer.topk_reference`: top K, ties to the lower id, over every visible group, the step's completed group
    included), its blocks ascending (the stock path's own order when a row's groups all fit; for selecting rows the stock
    order is unspecified, decode_indexer.py `select_decode_blocks`). The MTP draft's reader (`pooled_qsa.shared_blocks`,
    what H3's DraftAttn and the stock draft call) then sees the same groups.
  * the export does not perturb the kernel: with and without `l47_sink`, residual / ck_new / kr_new / ir_new bitwise.
Unit / wiring (no kernel): the planner's partial last cycle, the block routing, `sink_rows` vs the stock layout, the call
wrapper's gather and operand tail, the module's layer list / pool slots / L, the forward's sink write, `forward_wholestep`
skipping the stock layer 47, the env and the engagement lines.
"""
from __future__ import annotations

import dataclasses
import logging
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tests.megakernel import test_l1idx as tl
from tests.megakernel import test_wholestep as tw

FOLD_ENV = "VLLM_TPU_MK_L47_FOLD"
DCP_ENV = "VLLM_TPU_QSA_MAIN_KV_DCP"
S1 = dict(tl.S1)
IDX_ENVS = tl.IDX_ENVS
T = 4
RT_LOGGER = "vllm.vllm_torchtpu.megakernel_runtime.qwen38_l47fold"
WS_LOGGER = "vllm.vllm_torchtpu.megakernel_runtime.qwen38_wholestep"
SCEN = {
    # 8 rows, DCP 4 over two groups (R 8): 2 live requests + 2 padded slots, ranks 2-7 own no live request; per-layer tables.
    "d4_r8_8": dict(R=8, nreq=2, dcp=4, loop=True, budget=64, ctx=[3070, 100], per_layer=True),
    # 16 rows, DCP 2 (R 4: two pairs), budget 512 (two 256-slot passes per row), ring pinning MM's sum order, per-layer.
    "d2_r4_16": dict(R=4, nreq=4, dcp=2, loop=True, budget=512, ctx=[3001, 37, 2047, 254], ring=True, per_layer=True),
    # 16 rows, DCP 4, 512-token pool blocks, L1-IDX QSPLIT 1 + radix 2 (the `_qsplit_compact` export path), per-layer.
    "d4_r4_16q": dict(R=4, nreq=4, dcp=4, loop=True, budget=64, ctx=tw.CTX_D4, page512=True, per_layer=True,
                      cfg={"VLLM_TPU_MK_UNION_QSPLIT": "1", "VLLM_TPU_MK_SELECT_RADIX": "2"}),
    # 32 rows, DCP 4, unrolled requests, tie-heavy keys.
    "d4_r4_32": dict(R=4, nreq=8, dcp=4, loop=False, budget=64, ctx=tw.CTX_D4, ties=True),
    # 64 rows, DCP 4, request loop (4 requests per rank), per-layer tables.
    "d4_r4_64": dict(R=4, nreq=16, dcp=4, loop=True, budget=64, ctx=tw.CTX_D4, per_layer=True),
    # 64 rows, DCP 2, unrolled, tie-heavy keys.
    "d2_r4_64": dict(R=4, nreq=16, dcp=2, loop=False, budget=64, ctx=tw.CTX_D4, ties=True),
    # 8 rows, DCP 1 (R 2, replicated KV), per-layer tables, ring.
    "d1_r2_8": dict(R=2, nreq=2, dcp=1, loop=True, budget=64, ctx=tw.CTX_C, ring=True, per_layer=True),
}


# ------------------------------------------------------------------ kernel runner
def _fold_run(monkeypatch, scen: str, *, sink: bool = True, key: int = 17) -> dict:
    """One whole-step kernel run (CPU interpret) of scenario `scen` at L 5, first="qsa", D08 S1 on (the served indexed
    options from the envs, as test_l1idx `_idx_run`), with or without `l47_sink`."""
    from vllm_torchtpu.megakernel import kernel
    from vllm_torchtpu.megakernel.blocks.hc import RESIDUAL_OUT
    from vllm_torchtpu.megakernel.exec_spec import ExecSpec
    from vllm_torchtpu.megakernel.models.qwen38_pack import shard_step_inputs
    from vllm_torchtpu.megakernel.wholestep import inputs as wi
    from vllm_torchtpu.megakernel.wholestep.qsa_full_b import DBG_IQ
    from vllm_torchtpu.megakernel.wholestep.qsa_full_c import CK_NEW, IR_NEW, KR_NEW
    from vllm_torchtpu.megakernel.wholestep.qsa_full_l47f import SINK_IDS, lean_step_block
    from vllm_torchtpu.megakernel.wholestep.spec import wholestep_spec
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc

    kw = dict(SCEN[scen])
    R, nreq, dcp, K = kw["R"], kw["nreq"], kw["dcp"], kw["budget"]
    for e in tuple(S1) + tuple(IDX_ENVS) + (FOLD_ENV,):
        monkeypatch.delenv(e, raising=False)
    for e, v in dict(S1, **kw.get("cfg", {})).items():
        monkeypatch.setenv(e, v)
    if dcp > 1:
        monkeypatch.setenv(DCP_ENV, str(dcp))
    else:
        monkeypatch.delenv(DCP_ENV, raising=False)
    Lk, MG = 5, tw.MAX_GROUPS
    Gx = tw.G if R == 2 else dataclasses.replace(tw.G, NH=3 * R)
    if R > 4:
        Gx = dataclasses.replace(Gx, KH=R, VH=3 * R)
    ex = ExecSpec(R=R, B=nreq * T, T=T, ctx=128, interpret=True, tiles=tw._TILES)
    natt = len(wi.att_layers(Gx, Lk, "qsa"))
    assert natt == 2
    Lq = 4 * natt  # add_indexer_inputs / add_step_inputs count QSA layers with the default first="gdn"
    d = wi.random_wholestep_inputs(key, Gx, ex, Lk, budget=K, n_pages=tw.PAGES, identical_experts=True, first="qsa")
    assert d.get("first") == "qsa"
    d = wi.add_indexer_inputs(key, Gx, ex, d, Lq, max_groups=MG, ctx=kw["ctx"])
    if kw.get("ties"):
        d["ck"] = tw._s1_ties(d["ck"])
    d = wi.add_step_inputs(key, Gx, ex, d, Lq, ctx=kw["ctx"])
    if kw.get("ring"):
        d["ring"] = tw._s1_ring(d["ring"], d["ctxq"])
    if kw.get("per_layer"):
        d = wi.add_per_layer(key, Gx, ex, d, Lk)
    d = wi.add_carrier(key, d)
    d["hbm_tables"] = True
    if dcp > 1:  # the DCP group shares one KV head: same K / V columns, same pool
        kvw = 2 * Gx.HD
        d["qkvg"] = d["qkvg"].at[1:, ..., -kvw:].set(
            jnp.broadcast_to(d["qkvg"][:1, ..., -kvw:], d["qkvg"][1:, ..., -kvw:].shape))
        d["pool"] = jnp.broadcast_to(d["pool"][:1], d["pool"].shape)
    ref, _ = wi.wholestep_reference(Gx, ex, d, Lk)
    steps = d.pop("_step_out")
    ref4, _ = wi.wholestep_reference(Gx, ex, dict(d), 4)  # control: no layer 47 (the last QSA layer and its MoE)
    if kw.get("page512"):
        tl._page512(d)
    if dcp > 1:
        tw._dcp_split(d, dcp)
    pps = d["mtab"].shape[-1]
    indexed = dict(max_groups=MG, pages_per_seq=pps, debug=True, carrier="bf16", chunk_pages=8, hbm_tables=True,
                   per_request_scores=True, bounded_chunks=True, shard_index=True, union_attn=True, union_stats=False,
                   union_phys=False, loop_requests=kw["loop"], dcp=dcp, compressed_dcp=False)
    if kw.get("per_layer"):
        indexed["n_layers"] = natt
    indexed.update(wsc._d08s1_options())
    indexed.update(wsc._idx_options())
    if sink:
        indexed["l47_sink"] = True
    blk = lean_step_block(dict(indexed), dict(budget=K))
    spec = wholestep_spec(Gx, R, L=Lk, qsa=dict(budget=K), step=True, indexed=indexed, first="qsa")
    w_mk, states = wi.pack_wholestep(spec, Gx, ex, d, Lk)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(mesh, ex, d["inp"].pos, d["inp"].x, w_mk, states)
    step, _ = kernel.make_step(mesh, spec, ex)
    _, outs = step(pos_mk, hid_mk, w_mk, st_mk, *wi.extras_c(d, mesh, R))
    B = ex.B
    out = dict(
        R=R, B=B, K=K, dcp=dcp, natt=natt, nreq=nreq, mesh=mesh, d=d, steps=steps, block=type(blk).__name__,
        res=np.asarray(outs[RESIDUAL_OUT].reshape(R, ex.M, -1)[:, :B], np.float32),
        ref=np.asarray(ref, np.float32), ref4=np.asarray(ref4, np.float32),
        x0=np.asarray(d["inp"].x[:B], np.float32),
        iq=np.asarray(outs[DBG_IQ], np.float32).reshape(R, natt, ex.M, -1)[0, :, :B],
        ck_new=np.asarray(outs[CK_NEW], np.float32).reshape(R, natt, ex.M // T, -1)[0],
        kr=np.asarray(outs[KR_NEW], np.float32), ir=np.asarray(outs[IR_NEW], np.float32),
    )
    if sink:
        out["export"] = outs[SINK_IDS]
        out["rows"] = blk.sink_rows(ex)
    return out


def _sink_reference(run: dict) -> tuple[np.ndarray, dict, int]:
    """Per live row: the stock `selected` row (blocks ascending) of the stock selection from the kernel's own last-layer
    queries and completed-group key; plus how many rows' sets equal the stock-semantics reference's own selection."""
    from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import selected_tokens
    from vllm_torchtpu.megakernel.ops import BF16
    from vllm_torchtpu.megakernel.wholestep import inputs as wi
    from vllm_torchtpu.megakernel.wholestep.indexer import score_rows, topk_reference

    d, K, o = run["d"], run["K"], run["natt"] - 1
    dl = wi.layer_view(d, o)
    rows, sets, agree = [], {}, 0
    for r in range(run["B"]):
        q, t = divmod(r, T)
        c0 = int(d["ctxq"][q])
        p3 = c0 + (3 - c0 % 4) % 4
        qr = jnp.asarray(run["iq"][o, r].reshape(4, -1), BF16)
        st = np.array(score_rows(qr, dl["ck"][q]))[0]
        st[p3 // 4] = np.asarray(
            score_rows(qr, jnp.broadcast_to(jnp.asarray(run["ck_new"][o, q], BF16), (8, run["ck_new"].shape[-1]))))[0, 0]
        n = int(d["nvis"][r])
        want = sorted(topk_reference(st, n, K))
        sets[r] = set(want)
        agree += int(set(want) == set(run["steps"][o]["sel"][r]))
        blocks = jnp.asarray([want + [-1] * (K - len(want))], jnp.int32)
        rows.append(np.asarray(selected_tokens(blocks, jnp.asarray([c0 + t + 1], jnp.int32), K * 4, 4))[0])
    return np.stack(rows), sets, agree


def _fold_sink(run: dict) -> np.ndarray:
    """The served path: the export through `qwen38_l47fold.wrap_call` (a stand-in `call` returning it as the third
    output, the real operand tail) on the run's CPU mesh -> every rank's `[B, width]` (shape [R, B, width])."""
    from vllm_torchtpu.megakernel_runtime.qwen38_l47fold import wrap_call

    R, B, natt, dcp, nreq = run["R"], run["B"], run["natt"], run["dcp"], run["nreq"]
    tp = P("tp")
    starts = jnp.arange(nreq + 1, dtype=jnp.int32) * T
    context = jnp.asarray(np.asarray(run["d"]["ctxq"])[:nreq], jnp.int32)
    tail = [jnp.arange(B, dtype=jnp.int32), starts, context, jnp.zeros((8, 128), jnp.float32)]
    tail += [jnp.zeros((256,), jnp.float32)] * (2 * natt) + ([jnp.zeros((R,), jnp.int32)] if dcp > 1 else [])
    export = run["export"]
    call_l47f, _, _, _ = wrap_call(lambda *a: (a[0], a[0], export), (), (tp, tp, tp), (), mesh=run["mesh"], B=B,
                                   n_att=natt, dcp=dcp, gather=True, budget=run["K"])
    out = call_l47f(jnp.zeros((R * B, 1), jnp.float32), *tail)[2]
    return np.asarray(out).reshape(R, B, -1)


def _close(res: np.ndarray, ref: np.ndarray, sel_all_rows: list, Lk: int = 5) -> dict:
    """(B1) every rank, all rows: mean < 3e-2 k scale; (B2) every rank, select-all rows: `_assert_within_budget(layers=5)`.
    Raises AssertionError on a miss; returns the measured ratios (of scale)."""
    scale, k = float(np.abs(ref).mean()), np.sqrt(2 * Lk / 8)
    m = {"mean_all": max(float(np.abs(res[r] - ref).mean()) / scale for r in range(res.shape[0]))}
    sa = np.asarray(sel_all_rows, np.int64)
    dd = np.abs(res[:, sa] - ref[sa][None])
    m.update(sa_mean=float(dd.mean(axis=(1, 2)).max()) / scale, sa_q999=float(max(np.quantile(dd[r], 0.999)
             for r in range(res.shape[0]))) / scale, sa_max=float(dd.max()) / scale, k=k)
    assert m["mean_all"] < 3e-2 * k, ("B1", m)
    tw._assert_within_budget(res[:, sa], ref[sa], layers=Lk)
    return m


@pytest.mark.parametrize("scen", list(SCEN))
def test_l47f_kernel_close_and_sink(monkeypatch, scen):
    """The folded kernel (L 5, first="qsa", S1 + `l47_sink`): its residual is within the stated bound of the stock-semantics
    reference (and the L 4 reference, without layer 47, is not), and its sink equals the stock `selected` of the stock
    selection on every rank, every live row, every column."""
    from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import shared_blocks

    run = _fold_run(monkeypatch, scen)
    kw, K, B = SCEN[scen], run["K"], run["B"]
    nvis = np.asarray(run["d"]["nvis"])[:B]
    sel_all = [r for r in range(B) if all(nvis[(r // T) * T : (r // T + 1) * T] <= K)]
    assert sel_all and len(sel_all) < B, (scen, "the scenario needs select-all and selecting requests", sel_all)
    assert float(np.std(run["ref"] - run["x0"])) > 0.01
    # sink first (reported even if the closeness bound misses)
    want, sets, agree = _sink_reference(run)
    got = _fold_sink(run)
    width = 4 * K + 3
    assert got.shape == (run["R"], B, width), got.shape
    lens = jnp.asarray([int(run["d"]["ctxq"][r // T]) + r % T + 1 for r in range(B)], jnp.int32)
    bad = [(rk, r) for rk in range(run["R"]) for r in range(B) if not np.array_equal(got[rk, r], want[r])]
    ga_rows = sum(1 for r in range(B) if (int(run["d"]["ctxq"][r // T]) // 4) in sets[r])
    fits = int(sum(1 for r in range(B) if nvis[r] <= K))
    draft_ok = bool(np.array_equal(np.asarray(shared_blocks(jnp.asarray(got[0]), lens, 4, 4 * K)),
                                   np.asarray(shared_blocks(jnp.asarray(want), lens, 4, 4 * K))))
    print(f"L1-L47F SINK {scen}: block {run['block']}; {B} rows x {width} columns on {run['R']} ranks: "
          f"{run['R'] * B - len(bad)} / {run['R'] * B} rank-rows exact vs the stock selected of the stock selection on the "
          f"kernel's queries; rows with the step's completed group selected {ga_rows}; rows whose groups all fit {fits}; "
          f"draft reader (shared_blocks) equal {draft_ok}; agreement with the reference's own selection {agree} / {B}; "
          f"export rows per rank {run['rows']}", flush=True)
    sink_ok = not bad and draft_ok
    close_err = None
    try:
        m = _close(run["res"], run["ref"], sel_all)
    except AssertionError as e:  # noqa: PERF203
        close_err, m = e, None
    scale = float(np.abs(run["ref"]).mean())
    e5 = max(float(np.abs(run["res"][r] - run["ref"]).mean()) for r in range(run["R"])) / scale
    e4 = max(float(np.abs(run["res"][r] - run["ref4"]).mean()) for r in range(run["R"])) / scale
    ctl_caught = False
    try:
        tw._assert_within_budget(run["res"][:, sel_all], run["ref4"][sel_all], layers=5)
    except AssertionError:
        ctl_caught = True
    print(f"L1-L47F CLOSE {scen}: DCP {run['dcp']} rows {B} budget {K}: {m if m else 'MISS ' + repr(close_err)[:300]}; "
          f"select-all rows {len(sel_all)}; mean |kernel - ref| L5 {e5:.3e} vs L4 (no layer 47) {e4:.3e} of scale; "
          f"control L4 reference caught by B2: {ctl_caught}", flush=True)
    assert sink_ok, (scen, "sink", bad[:8], draft_ok)
    assert close_err is None, (scen, "closeness", close_err)
    assert ctl_caught, (scen, "control: the L 4 reference passed B2 (the bound does not see layer 47)")
    assert run["block"].endswith(("L47FLeanBlock", "L47FIdxBlock")), run["block"]
    if kw.get("cfg", {}).get("VLLM_TPU_MK_UNION_QSPLIT"):
        assert run["block"].endswith("L47FIdxBlock"), run["block"]


def test_l47f_export_does_not_perturb_the_kernel(monkeypatch):
    """`l47_sink` only adds the export: residual, ck_new, kr_new and ir_new bitwise equal with and without it."""
    a = _fold_run(monkeypatch, "d4_r8_8", sink=True)
    b = _fold_run(monkeypatch, "d4_r8_8", sink=False)
    for k in ("res", "ck_new", "kr", "ir", "iq"):
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)
    print(f"L1-L47F NOPERTURB d4_r8_8: residual / ck_new / kr_new / ir_new / queries bitwise equal with and without "
          f"l47_sink (blocks {a['block']} / {b['block']})", flush=True)


# ------------------------------------------------------------------ unit (no kernel)
def test_l47f_planner_partial_last_cycle():
    """L 45 (first="qsa"): 12 QSA + 33 GDN layers, the stream's tiles = 11 cycles + the QSA layer; L 44 / 5 / 1 as the
    formula; Qwen38's inner geometry counts the partial QSA layer."""
    from vllm_torchtpu.megakernel import planner
    from vllm_torchtpu.megakernel.exec_spec import ExecSpec
    from vllm_torchtpu.megakernel.models import qwen38
    from vllm_torchtpu.megakernel.wholestep.spec import wholestep_spec
    from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_call import wholestep_tiles

    g, R = qwen38.FULL, 8
    for L, nq, ng in ((44, 11, 33), (45, 12, 33), (5, 2, 3), (1, 1, 0)):
        spec = wholestep_spec(g, R, L=L, first="qsa", pool_gdn=True, qsa=dict(budget=512), step=True,
                              indexed=dict(max_groups=65536, pages_per_seq=2048, carrier="fp8", n_layers=nq))
        p = planner.plan(spec, ExecSpec(R=R, B=16, T=4, ctx=128, tiles=wholestep_tiles(16, 65536)))
        cnt = p.block_layer_counts
        qn = [n for n in cnt if n.startswith("qsa")]
        gn = [n for n in cnt if "gdn" in n]
        assert len(qn) == 1 and cnt[qn[0]] == nq, (L, cnt)
        assert (not gn and ng == 0) or (len(gn) == 1 and cnt[gn[0]] == ng), (L, cnt)
        want = (L // 4) * p.t_cycle + (p.branch_tiles[0] if L % 4 == 1 and L > 1 else 0) if L > 1 else p.T_TOTAL
        assert p.T_TOTAL == want, (L, p.T_TOTAL, want, p.t_cycle, p.branch_tiles)
        gi = g.inner(R, L)
        assert (gi.NATT, gi.NGDN) == (nq, ng), (L, gi.NATT, gi.NGDN)
        print(f"L1-L47F PLAN L {L}: counts {dict(cnt)} T_TOTAL {p.T_TOTAL} t_cycle {p.t_cycle} branch tiles "
              f"{p.branch_tiles}; inner NATT {gi.NATT} NGDN {gi.NGDN}", flush=True)
    with pytest.raises(AssertionError):
        wholestep_spec(g, R, L=46, first="qsa", pool_gdn=True, qsa=dict(budget=512))


def test_l47f_block_routing_and_decls(monkeypatch):
    """`qsa_full_l47f.lean_step_block`: without `l47_sink` the same object type as `qsa_full_idx.lean_step_block`; with it
    the L47F class of the lean / L1-IDX class at DCP 1 / 2 / 4; refused without a D08 S1 key; the export output (rows =
    run x T, or all rows unsharded) and scratch are declared only with it; `block_layers` reads the partial cycle."""
    from vllm_torchtpu.megakernel.exec_spec import ExecSpec
    from vllm_torchtpu.megakernel.models import qwen38
    from vllm_torchtpu.megakernel.wholestep import qsa_full_idx, qsa_full_l47f as lf

    base = dict(max_groups=1024, pages_per_seq=32, carrier="bf16", chunk_pages=8, hbm_tables=True, per_request_scores=True,
                bounded_chunks=True, union_attn=True)
    s1 = dict(union_bounded=True, union_lean=True, slot_batch=True)
    ex = ExecSpec(R=4, B=64, T=4, ctx=128, interpret=True, tiles=tw._TILES)
    for dcp in (1, 2, 4):
        for extra, want in (({}, "L47FLeanBlock"), ({"radix_bits": 2}, "L47FIdxBlock")):
            for shard in (True, False):
                opts = dict(base, shard_index=shard, dcp=dcp, **s1, **extra)
                plain = lf.lean_step_block(dict(opts), dict(budget=64))
                assert type(plain) is type(qsa_full_idx.lean_step_block(dict(opts), dict(budget=64)))
                blk = lf.lean_step_block(dict(opts, l47_sink=True), dict(budget=64))
                assert type(blk).__name__.endswith(want) and ("Group" in type(blk).__name__) == (dcp > 2), type(blk)
                assert isinstance(blk, type(plain)) and blk.l47_sink and not getattr(plain, "l47_sink", False)
                g = qwen38.TINY.inner(4, 5)
                names = [s.name for s in blk.outputs(g, ex)]
                assert lf.SINK_IDS in names and lf.SINK_IDS not in [s.name for s in plain.outputs(g, ex)]
                decl = [s for s in blk.outputs(g, ex) if s.name == lf.SINK_IDS][0]
                rows = (-(-(ex.MB // 4) // 4)) * 4 if shard else ex.MB
                assert decl.shape == (rows, 1, 64) and decl.per_layer is False and decl.access == "w", decl
    with pytest.raises(ValueError):
        lf.lean_step_block(dict(base, shard_index=True, dcp=1, l47_sink=True), dict(budget=64))
    sp = types.SimpleNamespace(L=45, period=4, branch_counts=(1, 3), block_branch_idx={"qsa_full": 0, "gdn": 1})
    assert lf.block_layers(sp, "qsa_full") == 12 and lf.block_layers(sp, "gdn") == 33
    sp44 = types.SimpleNamespace(L=44, period=4, branch_counts=(1, 3), block_branch_idx={"qsa_full": 0, "gdn": 1})
    assert lf.block_layers(sp44, "qsa_full") == 11
    sp1 = types.SimpleNamespace(L=1, period=1, branch_counts=(1,), block_branch_idx={"qsa_full": None})
    assert lf.block_layers(sp1, "qsa_full") == 1


def _sink_case(seed: int, R: int, nreq: int, slots: int, K: int):
    """Synthetic step: `nreq` live requests of T rows in a bucket of `slots` request slots (idle slots have empty ranges),
    contexts mixing rows that fit (n_vis <= K) and rows that select; ids per row ascending (the export's form)."""
    rng = np.random.default_rng(seed)
    B = slots * T
    ctx = rng.integers(0, 6 * K, slots).astype(np.int32)
    ctx[0], ctx[1] = 4 * K - 3, 4 * K + 9  # one request at the fit boundary, one just past it
    starts = np.zeros(slots + 1, np.int32)
    starts[1 : nreq + 1] = np.arange(1, nreq + 1) * T
    starts[nreq + 1 :] = nreq * T
    ids = np.zeros((B, K), np.int32)
    for r in range(nreq * T):
        nv = (int(ctx[r // T]) + r % T + 1) // 4
        if nv <= K:
            ids[r, :nv] = np.arange(nv)
        else:
            ids[r] = np.sort(rng.choice(nv, size=K, replace=False))
    return B, ctx, starts, ids


def _stock_selected(ids, ctx, starts, B, K):
    """The stock request-shard path's `selected` for the same selection: rows past the live tokens -1 blocks
    (`request_shard_blocks`), lengths from `pooled_qsa.qsa_cache_writes`' rule, `visible_blocks` when every row fits
    (`_shard_attend`), else the selection, then `selected_tokens`."""
    from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import visible_blocks
    from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import selected_tokens

    lens = np.zeros(B, np.int32)
    for q in range(len(ctx)):
        for t in range(starts[q], starts[q + 1]):
            lens[t] = ctx[q] + t - starts[q] + 1
    blocks = np.where(np.arange(K)[None] < np.minimum(lens // 4, K)[:, None], ids, -1)
    blocks = np.where((np.arange(B) < starts[-1])[:, None], blocks, -1)
    if (lens // 4).max() <= K:
        blocks = np.asarray(visible_blocks(jnp.asarray(lens), 4, K))
    return np.asarray(selected_tokens(jnp.asarray(blocks), jnp.asarray(lens), 4 * K, 4))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_l47f_sink_rows_match_stock_layout(seed):
    """`qwen38_l47fold.sink_rows` (count mask from the stock lengths, then `selected_tokens`) equals the stock path's
    `selected` for the same per-row selection, idle request slots and padded rows included; and `stock_lengths` is the
    stock rule (context + t + 1 on live rows, 0 elsewhere)."""
    from vllm_torchtpu.megakernel_runtime.qwen38_l47fold import sink_rows, sink_width, stock_lengths

    K = 64
    B, ctx, starts, ids = _sink_case(seed, 4, 3, 4, K)
    got = np.asarray(sink_rows(jnp.asarray(ids), jnp.asarray(starts), jnp.asarray(ctx), B=B, budget=K))
    assert got.shape == (B, sink_width(K)) == (B, 4 * K + 3)
    np.testing.assert_array_equal(got, _stock_selected(ids, ctx, starts, B, K))
    lens = np.asarray(stock_lengths(B, jnp.asarray(starts), jnp.asarray(ctx)))
    want = np.array([ctx[r // T] + r % T + 1 if r < starts[-1] else 0 for r in range(B)], np.int32)
    np.testing.assert_array_equal(lens, want)
    assert (got[starts[-1]:] == -1).all() and (got[: starts[-1], 0] >= 0).all()
    # all rows fit: the stock order is visible_blocks (ascending), the export's order too
    ctx2 = np.full_like(ctx, 37)
    _, _, _, ids2 = _sink_case(seed, 4, 3, 4, K)
    for r in range(B):
        nv = (37 + r % T + 1) // 4
        ids2[r, :nv] = np.arange(nv)
    np.testing.assert_array_equal(
        np.asarray(sink_rows(jnp.asarray(ids2), jnp.asarray(starts), jnp.asarray(ctx2), B=B, budget=K)),
        _stock_selected(ids2, ctx2, starts, B, K))


@pytest.mark.parametrize("dcp", [1, 4])
def test_l47f_wrap_call_gathers_runs_and_reads_the_tail(dcp):
    """`wrap_call` on the CPU mesh (R 4): each rank's export run is gathered in rank order (run r = rows [r rows, (r + 1)
    rows)), `starts` / `context` are read from the call's operand tail (with the DCP rank operand when DCP > 1), and every
    rank's output equals `sink_rows` of the concatenated runs; without gathering (unsharded indexer) each rank reads its
    own rows."""
    from vllm_torchtpu.megakernel_runtime.qwen38_l47fold import sink_rows, wrap_call

    R, K, natt, rows = 4, 64, 12, 8
    B, ctx, starts, ids = _sink_case(5, R, 7, 8, K)
    full = np.zeros((R * rows, K), np.int32)
    full[:B] = ids
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    tp = P("tp")
    export = jax.device_put(jnp.asarray(full.reshape(R * rows, 1, K)), jax.sharding.NamedSharding(mesh, tp))
    tail = [jnp.arange(B, dtype=jnp.int32), jnp.asarray(starts), jnp.asarray(ctx), jnp.full((8, 128), 7.0)]
    tail += [jnp.full((256,), 3.0)] * (2 * natt) + ([jnp.arange(R, dtype=jnp.int32)] if dcp > 1 else [])
    want = np.asarray(sink_rows(jnp.asarray(full), jnp.asarray(starts), jnp.asarray(ctx), B=B, budget=K))
    fn, ins, outs, don = wrap_call(lambda *a: (a[0], a[0], export), ("i",), (tp, tp, tp), (9,), mesh=mesh, B=B, n_att=natt,
                                   dcp=dcp, gather=True, budget=K)
    assert ins == ("i",) and outs == (tp, tp, tp) and don == (9,)
    got = np.asarray(fn(jnp.zeros((R * B, 1)), *tail)[2]).reshape(R, B, -1)
    for rk in range(R):
        np.testing.assert_array_equal(got[rk], want, err_msg=f"rank {rk}")
    fn2, _, _, _ = wrap_call(lambda *a: (a[0], a[0], jnp.tile(export[: rows], (R, 1, 1))), (), (tp, tp, tp), (),
                             mesh=mesh, B=rows, n_att=natt, dcp=dcp, gather=False, budget=K)
    got2 = np.asarray(fn2(jnp.zeros((R * rows, 1)), *tail)[2]).reshape(R, rows, -1)
    want2 = np.asarray(sink_rows(jnp.asarray(full[:rows]), jnp.asarray(starts), jnp.asarray(ctx), B=rows, budget=K))
    for rk in range(R):
        np.testing.assert_array_equal(got2[rk], want2, err_msg=f"ungathered rank {rk}")


def _collect(name: str) -> tuple[logging.Logger, list]:
    lg = logging.getLogger(name)
    recs: list = []

    class _H(logging.Handler):
        def emit(self, record):
            recs.append(record)

    lg.addHandler(_H())
    lg.setLevel(logging.INFO)
    return lg, recs


def test_l47f_env_options_and_call_builder(monkeypatch):
    """VLLM_TPU_MK_L47_FOLD: registered, default off; off -> no option / no output / the builder returns
    `build_wholestep_call`'s tuple itself; on -> {"l47_sink": True} only at L % 4 == 1 (else refused), the sink output name,
    the call's third output, refused with wholestats, and the builder wraps the call (n_att 12, gather = the sharded
    indexer) and logs its engagement line once."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.megakernel.wholestep.qsa_full_l47f import SINK_IDS
    from vllm_torchtpu.megakernel_runtime import qwen38_l47fold as lf
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc

    monkeypatch.delenv(FOLD_ENV, raising=False)
    assert envs.VLLM_TPU_MK_L47_FOLD is False and lf.l47f_on() is False
    assert lf.l47f_options(45, 4) == {} and wsc._l47f_options(45, 4) == {}
    assert wsc._l47f_outs(44, 4) == () and wsc._l47f_sink_out(44, 4) is False
    sentinel = (lambda *a: a, ("in",), (P("tp"),) * 2, (3,))
    rec: list = []
    monkeypatch.setattr(wsc, "build_wholestep_call", lambda *a, **k: rec.append((a, k)) or sentinel)
    g = types.SimpleNamespace(att_every=4)
    assert lf.build_wholestep_call_l47f(None, g, 16, None, [], L=44, budget=512) is sentinel
    monkeypatch.setenv(FOLD_ENV, "1")
    assert envs.VLLM_TPU_MK_L47_FOLD is True and lf.l47f_on() is True
    assert lf.l47f_options(45, 4) == {"l47_sink": True} == wsc._l47f_options(45, 4)
    assert wsc._l47f_outs(45, 4) == (SINK_IDS,) and wsc._l47f_sink_out(45, 4) is True
    with pytest.raises(AssertionError):
        lf.l47f_options(44, 4)
    monkeypatch.setattr(wsc, "UNION_ATTN", True)
    monkeypatch.setattr(wsc, "UNION_STATS_EVERY", 5)
    with pytest.raises(AssertionError):
        wsc._l47f_sink_out(45, 4)
    monkeypatch.setattr(wsc, "UNION_STATS_EVERY", 0)
    seen: dict = {}
    monkeypatch.setattr(wsc, "build_wholestep_call", lambda *a, **k: (lambda *x: x, ("in",), (P("tp"),) * 3, (3,)))
    monkeypatch.setattr(lf, "wrap_call", lambda *a, **k: seen.update(k, args=a) or ("wrapped",))
    monkeypatch.setattr(wsc, "SHARD_INDEX", True)
    lf._LOGGED.discard("call")
    lg, recs = _collect(RT_LOGGER)
    assert lf.build_wholestep_call_l47f(types.SimpleNamespace(size=8), g, 16, None, [], L=45, budget=512) == ("wrapped",)
    assert lf.build_wholestep_call_l47f(types.SimpleNamespace(size=8), g, 16, None, [], L=45, budget=512) == ("wrapped",)
    assert seen["n_att"] == 12 and seen["B"] == 16 and seen["gather"] is True and seen["budget"] == 512, seen
    lines = [r.getMessage() for r in recs if "q38 L1-L47F whole step" in r.getMessage()]
    assert len(lines) == 1 and "layers 3..47" in lines[0] and "12 QSA layers" in lines[0], lines
    print(f"L1-L47F ENGAGEMENT call builder: {lines[0]}", flush=True)


def _layers(n: int = 48):
    """Stub decoder layers: QSA at i % 4 == 3, GDN otherwise; each QSA attention with its own pool slots (compressed
    slot i, main slot 100 + i) so a wrong layer's slots are visible."""
    out = []
    for i in range(n):
        if i % 4 == 3:
            attn = types.SimpleNamespace(
                _select_keywords={"compressed_slot": i, "main_slot": 100 + i, "rotary_dim": 64}, fp8_kv=False,
                q_norm=types.SimpleNamespace(variance_epsilon=1e-6), prefix=f"model.layers.{i}.self_attn")
            out.append(types.SimpleNamespace(layer_type="full_attention", self_attn=attn))
        else:
            out.append(types.SimpleNamespace(layer_type="linear_attention",
                                             linear_attn=types.SimpleNamespace(prefix=f"model.layers.{i}.linear_attn")))
    return out


def _module(monkeypatch, fold: bool, tag: str):
    """`Qwen38WholeStep` over stub layers (FULL geometry, R 8, bucket 16), with the operand packer recorded instead of run."""
    from vllm_torchtpu.megakernel.models import qwen38
    from vllm_torchtpu.megakernel_runtime import operand_hold, qwen38_wholestep_weights
    from vllm_torchtpu.megakernel_runtime.qwen38_wholestep import Qwen38WholeStep

    for e in tuple(S1) + tuple(IDX_ENVS) + (FOLD_ENV, DCP_ENV):
        monkeypatch.delenv(e, raising=False)
    if fold:
        monkeypatch.setenv(FOLD_ENV, "1")
    rec: dict = {}

    def pack(model, g, spec, ex, ids):
        rec.update(L=spec.config.L, ids=list(ids))
        return [torch.zeros(2)]

    monkeypatch.setattr(qwen38_wholestep_weights, "pack_wholestep_operands", pack)
    monkeypatch.setattr(operand_hold, "host_round_trip_small", lambda ops: 0)
    monkeypatch.setattr(operand_hold, "hbm_summary", lambda ops, model: "stub")
    model = types.SimpleNamespace(model=types.SimpleNamespace(layers=_layers()))
    ws = Qwen38WholeStep(model, qwen38.FULL, types.SimpleNamespace(size=8), (16,), budget=512, max_groups=65536, tag=tag)
    return ws, rec


@pytest.mark.parametrize("fold", [False, True])
def test_l47f_module_layers_slots_and_L(monkeypatch, fold):
    """`Qwen38WholeStep`: fold off -> layers 3..46 (L 44, 11 QSA layers, the stock layer 47 left to the model); on -> 3..47
    (L 45, 12 QSA layers, the last one model layer 47), `l47f` set, the engagement line logged; `bind_pool` hands the call
    one cache entry per QSA layer with layer 47's own pool slots last; the call is built at L 45; the write-free shadow is
    refused under the fold."""
    from vllm_torchtpu.megakernel_runtime import exported_bridge, qwen38_l47fold
    from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_cache import QSALayerCache

    lg, recs = _collect(WS_LOGGER)
    ws, rec = _module(monkeypatch, fold, f"l47f_mod_{int(fold)}")
    n_q = 12 if fold else 11
    assert rec["L"] == (45 if fold else 44) and rec["ids"] == list(range(3, 48 if fold else 47)), rec
    assert ws.l47f is fold and len(ws.qsa_ids) == n_q and len(ws.gdn_ids) == 33 and ws.qsa_ids[-1] == (47 if fold else 43)
    lines = [r.getMessage() for r in recs if "q38 L1-L47F whole step" in r.getMessage()]
    assert len(lines) == int(fold), lines
    ws.bind_pool(torch.zeros(4), types.SimpleNamespace())
    assert ws.kind.l47f is fold and len(ws.kind.qsa) == n_q
    last = ws.qsa_ids[-1]
    assert ws.kind.qsa[-1] == QSALayerCache(compressed_slot=last, main_slot=100 + last, k_scale=1.0, v_scale=1.0)
    assert [c.compressed_slot for c in ws.kind.qsa] == ws.qsa_ids
    built: dict = {}
    monkeypatch.setattr(qwen38_l47fold, "build_wholestep_call_l47f",
                        lambda *a, **k: built.update(k, n=len(a[4])) or (None, (), (), ()))
    monkeypatch.setattr(exported_bridge, "make_typed_step_wrapper", lambda *a, **k: "typed")
    monkeypatch.setattr(exported_bridge, "sharded_jax_callable", lambda *a, **k: "callable")
    assert ws.kind.callable_for(16) == "callable"
    assert built["L"] == (45 if fold else 44) and built["n"] == n_q, built
    if fold:
        print(f"L1-L47F ENGAGEMENT module: {lines[0]}", flush=True)
        ws.write = False
        with pytest.raises(ValueError):
            ws.bind_pool(torch.zeros(4), types.SimpleNamespace())


@pytest.mark.parametrize("case", ["fold", "off", "nosink"])
def test_l47f_module_forward_writes_the_sink(monkeypatch, case):
    """`Qwen38WholeStep.forward`: under the fold the call's third output goes to layer 47's `_selection_sink` through the
    stock row update, once per step; fold off (or no sink bound) -> no write."""
    import vllm.forward_context as fc
    from torch import nn

    from vllm_torchtpu.layers.adapter.custom_ops import rows_update_op
    from vllm_torchtpu.megakernel_runtime.qwen38_wholestep import Qwen38WholeStep

    B, W = 8, 4 * 512 + 3
    sink = torch.full((32, W), -1, dtype=torch.int32)
    md = {}
    for suffix in (".raw_ring", "", ".c"):
        md["l47" + suffix] = types.SimpleNamespace(block_tables=torch.zeros((2, 4), dtype=torch.int32))
    attn = types.SimpleNamespace(prefix="l47", main_cache_name="l47", compressed_cache_name="l47.c",
                                 k_norm=types.SimpleNamespace(weight=torch.ones(4)), main_kv_dcp=1,
                                 indexer=types.SimpleNamespace(k_layernorm=types.SimpleNamespace(weight=torch.ones(4))),
                                 rope_cache=torch.zeros(4))
    if case != "nosink":
        attn._selection_sink = sink
    sel = torch.arange(B * W, dtype=torch.int32).reshape(B, W)
    ops_seen: list = []

    def op(hidden, args):
        ops_seen.append(len(args))
        return [hidden + 1, torch.zeros(4), sel]

    ws = Qwen38WholeStep.__new__(Qwen38WholeStep)
    nn.Module.__init__(ws)
    ws.layers = [None] * 47 + [types.SimpleNamespace(self_attn=attn)]
    ws.qsa_ids, ws.gdn_ids, ws.gdn_shared, ws.hold, ws._ops, ws.n_w = [47], [], False, "list", [], 0
    ws.pool, ws.union_stats, ws.l47f = torch.zeros(4), None, case != "off"
    ws.kind = types.SimpleNamespace(op=op)
    monkeypatch.setattr(fc, "get_forward_context", lambda: types.SimpleNamespace(attn_metadata=md))
    writes: list = []
    monkeypatch.setattr(rows_update_op, "update_rows_", lambda buf, rows: writes.append((buf, rows)))
    out = ws.forward(torch.zeros((B, 2)), torch.arange(B), torch.arange(3), torch.zeros(2))
    assert torch.equal(out, torch.ones((B, 2))) and ops_seen
    if case == "fold":
        assert len(writes) == 1 and writes[0][0] is sink and writes[0][1] is sel, writes
    else:
        assert not writes, writes


@pytest.mark.parametrize("fold", [False, True])
def test_l47f_forward_wholestep_skips_the_stock_layer47(fold):
    """`Qwen4ExpBackbone.forward_wholestep`: fold off -> layer 47's stock HC mix, `attend` and the last segment kernel run
    on the whole step's output; on (`whole.l47f`) -> none of them, the final HC mixer gets the whole step's output, and
    the MTP hidden (`emit_mtp_hidden`) is that output."""
    from vllm_torchtpu.models.vllm.qwen4_exp import model as m

    calls: list = []
    H = 4

    class _Whole:
        l47f = fold

        def __call__(self, hidden, positions, qsl, ctx):
            calls.append("whole")
            return hidden + 10

    def _attend(block, pos, md, request_shard=True):
        # q38 STACK-D v5 (L1-V5): the composed forward_wholestep passes O1SD14's request_shard=False (model.py); pinned here.
        assert request_shard is False, "O1SD14: the whole step's layer 47 must not take O1s"
        calls.append("attend")
        return block * 0 + 5

    last_attn = types.SimpleNamespace(attend=_attend, select_for_kernel=lambda *a: calls.append("select"))
    layers = [types.SimpleNamespace(layer_type="linear_attention") for _ in range(47)]
    layers.append(types.SimpleNamespace(layer_type="full_attention", self_attn=last_attn,
                                        attn_hyper_connection=types.SimpleNamespace(
                                            mix=lambda h: (calls.append("mix47") or h, None))))

    class _Seg:
        head_groups: list = []
        attention_rows = 0

        def __call__(self, s, h, a):
            calls.append(("seg", s))
            return h + 100

    for emit in (False, True):
        calls.clear()
        stub = types.SimpleNamespace(
            host_ple_index=None, stage=list(range(48)), layers=layers, hc_count=1, emit_mtp_hidden=emit,
            embed_tokens=lambda ids: torch.zeros((ids.shape[0], H)),
            hyper_connection_mixer=types.SimpleNamespace(mix=lambda h: (calls.append(("mixer", float(h[0, 0]))) or h * 2,
                                                                        None)))
        ids = torch.zeros(4, dtype=torch.int64)
        out = m.Qwen4ExpBackbone.forward_wholestep(
            stub, ids, torch.arange(4), _Seg(), _Whole(), query_start_loc=torch.arange(2), ngram_context=None,
            state_slots=None, context_lens=torch.zeros(1), valid_tokens=None, main_block_table=None,
            compressed_block_table=None)
        mixed, hidden = out if emit else (out, None)
        if fold:
            assert calls == ["whole", ("mixer", 10.0)], calls
            assert float(mixed[0, 0]) == 20.0 and (not emit or float(hidden[0, 0]) == 10.0)
        else:
            assert calls == ["whole", "mix47", "attend", ("seg", 0), ("mixer", 110.0)], calls
            assert float(mixed[0, 0]) == 220.0 and (not emit or float(hidden[0, 0]) == 110.0)


def test_l47f_no_such_node():
    """Guard control node target (G0 selects a name that collects nothing)."""
