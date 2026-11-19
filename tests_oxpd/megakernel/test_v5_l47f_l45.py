# SPDX-License-Identifier: Apache-2.0
"""STACK-D v5 (q38 L1-V5): L47-FOLD-S1 at FULL depth (the L1 successor's ruling: L47's own kernel legs ran at L 5 only).

The whole step at L 45 (model layers 3..47, `first="qsa"`: 12 QSA + 33 GDN layers, the partial last cycle = layer 47) in CPU
interpret, main-KV DCP 4, 16 rows (R 4, 4 requests), with D08 S1 + L1-IDX QSPLIT 1 + radix 2 (L47's own scenario d4_r4_16q:
512-token pool blocks, per-layer tables) and `l47_sink`. Rules, L47-FOLD-S1's own, stated before its first data
(scratch/overflow/l1_l47fold_status.md) and L-generic (k = sqrt(2 L / 8)):
  * SINK: the export, gathered and expanded by the served `qwen38_l47fold.wrap_call` on the CPU mesh, equals on EVERY rank,
    every live row, every column the stock `selected` layout of the stock selection computed from the kernel's own
    last-QSA-layer queries and completed-group key (L47's `_sink_reference`), and the MTP draft's reader
    (`pooled_qsa.shared_blocks`) sees the same groups;
  * CLOSE: (B1) every rank, every row: mean |kernel - ref| < 3e-2 k scale; (B2) every rank, select-all rows:
    test_wholestep `_assert_within_budget(layers=45)`; control: the same output against the L 44 reference (no layer 47) is
    caught by B2;
  * the block is the L47F IDX block (QSPLIT routes `_qsplit_compact`).
The runner is L47's `_fold_run` (tests/megakernel/test_l1l47f.py) with its depth as a parameter (its body copied, credit
L1-L47F); the sink reference, the served-wrapper expansion and the closeness check are L47's own functions, imported.
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from tests.megakernel import test_l1l47f as L
from tests.megakernel import test_l1idx as tl
from tests.megakernel import test_wholestep as tw

LK = 45
SCEN = "d4_r4_16q"


def _fold_run_depth(monkeypatch, scen: str, Lk: int, key: int = 17) -> dict:
    """L47's `_fold_run` (sink on) at depth `Lk` (its body, Lk as a parameter; the control reference at Lk - 1)."""
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

    kw = dict(L.SCEN[scen])
    R, nreq, dcp, K = kw["R"], kw["nreq"], kw["dcp"], kw["budget"]
    for e in tuple(L.S1) + tuple(L.IDX_ENVS) + (L.FOLD_ENV,):
        monkeypatch.delenv(e, raising=False)
    for e, v in dict(L.S1, **kw.get("cfg", {})).items():
        monkeypatch.setenv(e, v)
    if dcp > 1:
        monkeypatch.setenv(L.DCP_ENV, str(dcp))
    else:
        monkeypatch.delenv(L.DCP_ENV, raising=False)
    MG = tw.MAX_GROUPS
    Gx = tw.G if R == 2 else dataclasses.replace(tw.G, NH=3 * R)
    if R > 4:
        Gx = dataclasses.replace(Gx, KH=R, VH=3 * R)
    ex = ExecSpec(R=R, B=nreq * L.T, T=L.T, ctx=128, interpret=True, tiles=tw._TILES)
    natt = len(wi.att_layers(Gx, Lk, "qsa"))
    assert natt == -(-Lk // Gx.att_every), (natt, Lk)
    Lq = 4 * natt
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
    if dcp > 1:
        kvw = 2 * Gx.HD
        d["qkvg"] = d["qkvg"].at[1:, ..., -kvw:].set(
            jnp.broadcast_to(d["qkvg"][:1, ..., -kvw:], d["qkvg"][1:, ..., -kvw:].shape))
        d["pool"] = jnp.broadcast_to(d["pool"][:1], d["pool"].shape)
    ref, _ = wi.wholestep_reference(Gx, ex, d, Lk)
    steps = d.pop("_step_out")
    ref_ctl, _ = wi.wholestep_reference(Gx, ex, dict(d), Lk - 1)  # control: no layer 47
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
    indexed["l47_sink"] = True
    blk = lean_step_block(dict(indexed), dict(budget=K))
    spec = wholestep_spec(Gx, R, L=Lk, qsa=dict(budget=K), step=True, indexed=indexed, first="qsa")
    w_mk, states = wi.pack_wholestep(spec, Gx, ex, d, Lk)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(mesh, ex, d["inp"].pos, d["inp"].x, w_mk, states)
    step, _ = kernel.make_step(mesh, spec, ex)
    _, outs = step(pos_mk, hid_mk, w_mk, st_mk, *wi.extras_c(d, mesh, R))
    B = ex.B
    return dict(
        R=R, B=B, K=K, dcp=dcp, natt=natt, nreq=nreq, mesh=mesh, d=d, steps=steps, block=type(blk).__name__,
        res=np.asarray(outs[RESIDUAL_OUT].reshape(R, ex.M, -1)[:, :B], np.float32),
        ref=np.asarray(ref, np.float32), ref4=np.asarray(ref_ctl, np.float32),
        x0=np.asarray(d["inp"].x[:B], np.float32),
        iq=np.asarray(outs[DBG_IQ], np.float32).reshape(R, natt, ex.M, -1)[0, :, :B],
        ck_new=np.asarray(outs[CK_NEW], np.float32).reshape(R, natt, ex.M // L.T, -1)[0],
        kr=np.asarray(outs[KR_NEW], np.float32), ir=np.asarray(outs[IR_NEW], np.float32),
        export=outs[SINK_IDS], rows=blk.sink_rows(ex),
    )


def test_l47f_full_depth_l45_sink_and_close(monkeypatch):
    from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import shared_blocks

    run = _fold_run_depth(monkeypatch, SCEN, LK)
    K, B, R = run["K"], run["B"], run["R"]
    assert (R, run["dcp"], B, run["natt"]) == (4, 4, 16, 12), (R, run["dcp"], B, run["natt"])
    nvis = np.asarray(run["d"]["nvis"])[:B]
    sel_all = [r for r in range(B) if all(nvis[(r // L.T) * L.T : (r // L.T + 1) * L.T] <= K)]
    assert sel_all and len(sel_all) < B, ("the scenario needs select-all and selecting requests", sel_all)
    want, sets, agree = L._sink_reference(run)
    got = L._fold_sink(run)
    assert got.shape == (R, B, 4 * K + 3), got.shape
    lens = jnp.asarray([int(run["d"]["ctxq"][r // L.T]) + r % L.T + 1 for r in range(B)], jnp.int32)
    bad = [(rk, r) for rk in range(R) for r in range(B) if not np.array_equal(got[rk, r], want[r])]
    draft_ok = bool(np.array_equal(np.asarray(shared_blocks(jnp.asarray(got[0]), lens, 4, 4 * K)),
                                   np.asarray(shared_blocks(jnp.asarray(want), lens, 4, 4 * K))))
    close_err, m = None, None
    try:
        m = L._close(run["res"], run["ref"], sel_all, Lk=LK)
    except AssertionError as e:
        close_err = e
    scale = float(np.abs(run["ref"]).mean())
    e45 = max(float(np.abs(run["res"][r] - run["ref"]).mean()) for r in range(R)) / scale
    e44 = max(float(np.abs(run["res"][r] - run["ref4"]).mean()) for r in range(R)) / scale
    ctl_caught = False
    try:
        tw._assert_within_budget(run["res"][:, sel_all], run["ref4"][sel_all], layers=LK)
    except AssertionError:
        ctl_caught = True
    print(f"V5 L47F L45 SINK {SCEN}: block {run['block']}; {R * B - len(bad)} / {R * B} rank-rows exact; draft reader "
          f"equal {draft_ok}; agreement with the reference's own selection {agree} / {B}", flush=True)
    print(f"V5 L47F L45 CLOSE {SCEN}: {m if m else 'MISS ' + repr(close_err)[:300]}; select-all rows {len(sel_all)}; mean "
          f"|kernel - ref| L45 {e45:.3e} vs L44 (no layer 47) {e44:.3e} of scale; control caught {ctl_caught}", flush=True)
    assert not bad and draft_ok, ("sink", bad[:8], draft_ok)
    assert close_err is None, ("closeness", close_err)
    assert ctl_caught, "control: the L 44 reference passed B2"
    assert run["block"].endswith("L47FIdxBlock"), run["block"]
