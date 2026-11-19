# SPDX-License-Identifier: Apache-2.0
"""q38 L1-V6 (STACK-D v6): composed CPU tests on q38-L1-v6 (CDK CPU pod only; never the Mac or the VM).

C1  SELLOOP x X-UA (whole step, CPU interpret, test_l2_xuav5's `_run` = test_l1idx's `_idx_run` with the served
    `_d08s1_options()` + `_idx_options()` plumbing, plus `_selloop_options()` merged in as test_l1selloop does): F1's IDX
    family at test scale (scorer chunk / lanes / manager pages, radix 2, UNION_QSPLIT 1, TABLE_SLIM 2 = the prefetch)
    + DCP_QX_EARLY 1 + X-UA ISSUE 2 + UNPACK 1 + SELECT_OWNED (the v6 kernel option set) is BITWISE the IDX-on base (X-UA
    off, SELLOOP off) at 16 and 48 rows, main-KV DCP 4 (`test_wholestep._s1_same`). The block is
    QSAFullStepGroupIdxQxBlock with select_owned engaged (the composed `qsa_full_idx.lean_step_block` takes the option
    out before the QX routing); SELLOOP's trace record shows the owned-slot loop for the run and nothing for the base.
    (X-UA x QSPLIT=1 alone is test_l2_xuav5 Part A: every arm there carries UNION_QSPLIT 1.)
C2  KVW x IDX (TABLE_SLIM 2): with the same v6 kernel option set, the kernel's own QSA cache values (qsa_kv_new key rows,
    qsa_ck_new, qsa_ring_new) equal what the post-kernel writer recomputes from the kernel's qsa_kr_new / qsa_ir_new
    (`produce_kv`, `_indexer_updates`), bit for bit, on every rank, QSA layer, row and request, at 16 and 48 rows, both
    settings of VLLM_TPU_QSA_INDEXER_UPDATE_SHIFT. That is QSA-WRITE-FOLD stage A's premise (KVAL bytes == stock bytes
    given test_qwen38_qsa_write_kval's writer proofs). A positive control (keys recomputed one position late) must
    differ. One RoPE table feeds every RoPE row and the norm weights are BF16 (served-like, as the kval test's kernel run).
C3  KVW-ALL alone through the composed dispatch (`wholestep_qsa_writes`, no kernel values): with
    VLLM_TPU_MK_QSA_WRITE_ALL off the stock `_batched_writes` runs (no KVW trace record, no write-all call); on, the
    all-layer writer runs (KVW_TRACES "all" + its log line, one write-all call); both write the stock bytes.
C4  SELECT_OWNED together with L47_FOLD is refused by `_selloop_options()` (the layer-47 fold rebuilds the block class).
C5  Engagement lines: each armed v6 member's build-time line (SELLOOP, X-UA ISSUE / UNPACK, KVW-ALL with and without
    KVAL) is emitted on vLLM's handlers ("vllm." prefix) with the regexes of scratch/decL1/s7/v6/engagement_markers.tsv.
Run: JAX_PLATFORMS=cpu PYTHONPATH=src python -m pytest -q -p no:cacheprovider -rfEs -s tests/megakernel/test_l1v6_compose.py
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pytest

from tests.megakernel import test_l1idx as li
from tests.megakernel import test_l2_xuav5 as xu
from tests.megakernel import test_wholestep as tw
from tests.megakernel_runtime import test_qwen38_qsa_write_kval as kt
from tests.megakernel_runtime.test_qwen38_qsa_write_kval import counters  # noqa: F401  (fixture)

SEL_ENV, L47_ENV = "VLLM_TPU_MK_SELECT_OWNED", "VLLM_TPU_MK_L47_FOLD"
V6_SELECT_OWNED = 1
MARKERS = {
    "selloop": r"q38 L1-SELLOOP whole step: VLLM_TPU_MK_SELECT_OWNED=1 ",
    "uaissue": r"q38 UAISSUE whole step: VLLM_TPU_MK_UNION_ISSUE=2 ",
    "uaunpack": r"q38 UAUNPACK whole step: VLLM_TPU_MK_UNION_UNPACK=1 ",
    "wfold": r"q38 WFOLD whole step: QSA cache writes take the kernel's values \(VLLM_TPU_MK_QSA_WRITE_ALL=1 "
             r"VLLM_TPU_MK_QSA_WRITE_KVAL=1\)",
    "kvwall": r"q38 KVW-ALL whole step: QSA cache writes as one all-layer Pallas call \(VLLM_TPU_MK_QSA_WRITE_ALL=1\)",
    "kvwtrace": r"q38 L1-KVW wholestep_qsa_writes traced: all-layer writer \(batched_writes_all\), kernel values (on|off)",
}
C1_ARMS = [("d4_r4_16", V6_SELECT_OWNED), ("d4_r4_48", V6_SELECT_OWNED)]


def _selloop(monkeypatch, mode: int) -> None:
    """SELECT_OWNED = `mode` (0 = unset) merged into the served indexed options, as test_l1selloop's `_run`."""
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc

    monkeypatch.delenv(L47_ENV, raising=False)
    if mode:
        monkeypatch.setenv(SEL_ENV, str(mode))
    else:
        monkeypatch.delenv(SEL_ENV, raising=False)
    idx = wsc.__dict__.get("_q38v6_idx_orig") or wsc._idx_options
    monkeypatch.setattr(wsc, "_q38v6_idx_orig", idx, raising=False)
    monkeypatch.setattr(wsc, "_idx_options", lambda: {**idx(), **wsc._selloop_options()})


def _v6_run(monkeypatch, scen: str, mode: int):
    """The v6 kernel option set (XP = IDX ALL + QX, X-UA ISSUE 2 + UNPACK 1, SELECT_OWNED `mode`); returns the run and
    SELLOOP's trace-record delta."""
    from vllm_torchtpu.megakernel.wholestep import qsa_full_lean as ql

    _selloop(monkeypatch, mode)
    before = dict(ql.SELLOOP_TRACES)
    run = xu._run(monkeypatch, scen, "XP", 2, 1)
    traced = {k: v - before.get(k, 0) for k, v in ql.SELLOOP_TRACES.items() if v != before.get(k, 0)}
    return run, traced


@pytest.mark.parametrize(("scen", "mode"), C1_ARMS, ids=[f"{s}-sel{m}" for s, m in C1_ARMS])
def test_v6_selloop_x_xua_idx_identity(monkeypatch, scen, mode):
    """C1: the v6 kernel option set vs the IDX-on base (X-UA off, SELLOOP off), bitwise."""
    from vllm_torchtpu.megakernel.wholestep import qsa_full_lean as ql

    kw = xu.SCEN[scen]
    D, KU = kw["dcp"], li._ku(kw["budget"])
    _selloop(monkeypatch, 0)
    before = dict(ql.SELLOOP_TRACES)
    base = xu._run(monkeypatch, scen, "XP", 0, 0)
    assert dict(ql.SELLOOP_TRACES) == before, "SELLOOP traced with the flag off"
    xu._check_options(base, "XP", 0, 0, f"{scen} base")
    assert "select_owned" not in base[8]["_opts"], base[8]["_opts"]
    run, traced = _v6_run(monkeypatch, scen, mode)
    xu._check_options(run, "XP", 2, 1, f"{scen} v6")
    assert run[8]["_opts"].get("select_owned") == mode, run[8]["_opts"]
    assert traced and all(v > 0 for v in traced.values()), ("SELLOOP's owned-slot loop was not traced", traced)
    tw._s1_same(run, base, D, KU, f"{scen} v6 (X-UA 2/1 + SELECT_OWNED {mode}) vs IDX-on base")
    print(f"L1-V6 C1 {scen}: X-UA ISSUE 2 + UNPACK 1 + SELECT_OWNED {mode} on IDX ALL (QSPLIT 1, TABLE_SLIM 2) + QX: bitwise "
          f"the IDX-on base; block {run[8]['_block']}; SELLOOP traced {sorted(traced)}", flush=True)


def _capture(monkeypatch) -> dict:
    """Record the kernel's outputs and make the step inputs served-like (BF16 norm weights, one RoPE table at the
    rows' positions), as test_qwen38_qsa_write_kval's `kernel_run`."""
    import jax.numpy as jnp

    from vllm_torchtpu.megakernel import kernel
    from vllm_torchtpu.megakernel.ops import BF16, F32
    from vllm_torchtpu.megakernel.wholestep import inputs as wi

    cap: dict = {}
    orig_ms = kernel.make_step

    def make_step(mesh, spec, ex):
        step, info = orig_ms(mesh, spec, ex)

        def recorded(*a, **k):
            out = step(*a, **k)
            cap["outs"], cap["ex"] = out[1], ex
            return out

        return recorded, info

    monkeypatch.setattr(kernel, "make_step", make_step)
    orig_asi = wi.add_step_inputs

    def add_step_inputs(key, g, ex, d, L, **k):
        d = orig_asi(key, g, ex, d, L, **k)
        ctxq = np.asarray(d["ctxq"])
        T = ex.T
        positions = (ctxq[:, None] + np.arange(T)[None]).reshape(-1).astype(np.int32)
        trig = kt.trig_table(int(ctxq.max()) + 2 * T + 1)
        j0 = ctxq + (3 - ctxq % 4) % 4 - 3
        d["kn"] = d["kn"].astype(BF16).astype(F32)
        d["ikn"] = d["ikn"].astype(BF16).astype(F32)
        d["rope"] = jnp.asarray(trig[positions], F32)
        d["rope_ck"] = jnp.asarray(trig[np.maximum(j0, 0)], F32)[:, None, :]
        cap.update(positions=positions, trig=trig, ctx=ctxq.astype(np.int32))
        return d

    monkeypatch.setattr(wi, "add_step_inputs", add_step_inputs)
    return cap


@pytest.mark.parametrize("scen", ["d4_r4_16", "d4_r4_48"])
def test_v6_kvw_kernel_values_under_idx(monkeypatch, scen):
    """C2: the v6 kernel's own cache values == the writer's recomputation from the kernel's projections, bitwise."""
    import jax
    import jax.numpy as jnp

    from vllm_torchtpu import envs
    from vllm_torchtpu.kernels.qwen4_exp.qsa import _indexer_updates, produce_kv
    from vllm_torchtpu.megakernel.wholestep.qsa_full import KV_NEW
    from vllm_torchtpu.megakernel.wholestep.qsa_full_c import CK_NEW, IR_NEW, KR_NEW, RING_NEW

    cap = _capture(monkeypatch)
    run, traced = _v6_run(monkeypatch, scen, V6_SELECT_OWNED)
    xu._check_options(run, "XP", 2, 1, f"{scen} v6 C2")
    assert traced, "SELLOOP not traced"
    d, ex, outs = run[8], cap["ex"], cap["outs"]
    R, M, T = ex.R, ex.M, ex.T
    nreq = M // T
    n_att = tw.L // tw.G.att_every  # as test_l1idx `_idx_run` (Gx differs from tw.G only in head counts)
    B = int(np.asarray(cap["positions"]).shape[0])
    B = min(B, M)
    g = {k: np.asarray(outs[n]).reshape(R, n_att, *s) for k, n, s in (
        ("kv", KV_NEW, (M, -1)), ("kr", KR_NEW, (M, -1)), ("ir", IR_NEW, (M, -1)),
        ("ck", CK_NEW, (nreq, 1, -1)), ("ring", RING_NEW, (nreq, kt.RING_ROWS, -1)))}
    hd = g["kv"].shape[-1] // 2
    kn, ikn = np.asarray(d["kn"]), np.asarray(d["ikn"])
    ring_in = np.asarray(d["ring"])
    ctx, positions, trig = cap["ctx"][:nreq], jnp.asarray(cap["positions"][:B]), cap["trig"]
    nb = B // T
    lens = np.full(nb, T, np.int32)
    starts = jnp.asarray(np.concatenate([[0], np.cumsum(lens)]).astype(np.int32))
    context = jnp.asarray(ctx[:nb])
    slots = jnp.where(starts[1:] > starts[:-1], jnp.arange(nb, dtype=jnp.int32), -1)
    tags = jnp.asarray(kt.coherent_tags(ctx[:nb]))
    kfn = jax.jit(lambda kr, v, nk, pos, tr: produce_kv(kr, v, nk, pos, tr, kt.EPS, kt.ROTARY)[0])
    ifn = jax.jit(lambda raw, ring, tg, nik, pos, tr: _indexer_updates(
        raw, pos, ring, tg, nik, tr, starts, context, slots, kt.RATIO, kt.EPS, kt.ROTARY))
    n_rows = n_req = bad_k = bad_c = bad_r = ctl = 0
    for shift in (False, True):
        monkeypatch.setattr(envs, kt.SHIFT, shift)
        jax.clear_caches()
        for r in range(R):
            for o in range(n_att):
                kv = g["kv"][r, o, :B]
                nk = jnp.asarray(kn[o] if kn.ndim > 1 else kn, jnp.bfloat16)
                k_ref = np.asarray(kfn(jnp.asarray(g["kr"][r, o, :B]), jnp.asarray(kv[:, hd:]), nk, positions,
                                       trig)).reshape(B, -1)
                bad_k += int((kt.bits(kv[:, :hd]) != kt.bits(k_ref)).sum())
                n_rows += B
                if r == 0 and o == 0 and not shift:  # positive control: one position late must differ
                    k_bad = np.asarray(kfn(jnp.asarray(g["kr"][r, o, :B]), jnp.asarray(kv[:, hd:]), nk, positions + 1,
                                           trig)).reshape(B, -1)
                    ctl = int((kt.bits(kv[:, :hd]) != kt.bits(k_bad)).sum())
                ri = ring_in[o] if ring_in.ndim == 4 else ring_in
                nik = jnp.asarray(ikn[o] if ikn.ndim > 1 else ikn, jnp.bfloat16)
                completed, complete, req, _, ring, _ = (np.asarray(x) for x in ifn(
                    jnp.asarray(g["ir"][r, o, :B]), jnp.asarray(ri[:nb]), tags, nik, positions, trig))
                ends = np.nonzero(complete)[0]
                for t in ends:
                    q = int(req[t])
                    bad_c += int((kt.bits(g["ck"][r, o, q, 0]) != kt.bits(completed[t])).sum())
                    n_req += 1
                for q in range(nb):
                    bad_r += int((kt.bits(g["ring"][r, o, q]) != kt.bits(ring[q])).sum())
    print("L1-V6 C2 KVW_X_IDX " + str(dict(scen=scen, ranks=R, layers=n_att, rows=B, requests=nb, rows_checked=n_rows,
                                            group_ends=n_req, key_bits_diff=bad_k, ck_bits_diff=bad_c,
                                            ring_bits_diff=bad_r, control_bits_diff=ctl, block=d["_block"])), flush=True)
    assert n_rows > 0 and n_req > 0 and ctl > 0, (n_rows, n_req, ctl)
    assert bad_k == 0 and bad_c == 0 and bad_r == 0, (bad_k, bad_c, bad_r)


def test_v6_kvw_dispatch_write_all_alone(monkeypatch, counters):  # noqa: F811
    """C3: VLLM_TPU_MK_QSA_WRITE_ALL alone through `wholestep_qsa_writes`: off = stock path, on = the all-layer writer
    (trace record + log line + one write-all call); both write the stock bytes."""
    import jax

    from vllm_torchtpu import envs
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_cache as wc

    recs: list = []
    h = logging.Handler()
    h.emit = recs.append  # type: ignore[method-assign]
    lg = logging.getLogger("vllm." + wc.__name__)
    lg.addHandler(h)
    old = lg.level
    lg.setLevel(logging.INFO)
    try:
        monkeypatch.setattr(wc, "KVW_TRACES", {})
        case = kt.make_case(711, 16, "served", n_layers=kt.L, dcp=4)
        with kt.interpret():
            monkeypatch.setattr(envs, "VLLM_TPU_MK_QSA_WRITE_ALL", False)
            n0 = len(counters["stats"])
            off = kt.run_dispatch(case, case["pool"], 1, None)
            jax.block_until_ready(off)
            jax.effects_barrier()
            assert wc.KVW_TRACES == {} and len(counters["stats"]) == n0, (wc.KVW_TRACES, counters["stats"][n0:])
            monkeypatch.setattr(envs, "VLLM_TPU_MK_QSA_WRITE_ALL", True)
            on = kt.run_dispatch(case, case["pool"], 1, None)
            jax.block_until_ready(on)
            jax.effects_barrier()
            assert wc.KVW_TRACES == {"all": 1}, wc.KVW_TRACES
            assert len(counters["stats"]) == n0 + 1, counters["stats"][n0:]
            stock = kt.run(wc._batched_writes, case, case["pool"], 1)
        a = kt.u8(stock)
        assert not np.array_equal(a, kt.u8(case["pool"])), "the stock writer wrote nothing"
        np.testing.assert_array_equal(kt.u8(off), a, err_msg="WRITE_ALL off != stock")
        np.testing.assert_array_equal(kt.u8(on), a, err_msg="WRITE_ALL on != stock")
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)
    msgs = [r.getMessage() for r in recs]
    assert any(re.search(MARKERS["kvwtrace"], m) and "kernel values off" in m for m in msgs), msgs
    print(f"L1-V6 C3 KVW-ALL dispatch: off -> stock (no record), on -> all-layer writer {counters['stats'][-1]}; "
          f"bytes == stock; line {[m for m in msgs if 'L1-KVW' in m][:1]}", flush=True)


def test_v6_selloop_refused_with_l47_fold(monkeypatch):
    """C4: SELECT_OWNED with L47_FOLD raises; either alone builds its usual options."""
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc

    monkeypatch.setenv(SEL_ENV, "1")
    monkeypatch.setenv(L47_ENV, "1")
    with pytest.raises(ValueError, match="refused together with VLLM_TPU_MK_L47_FOLD=1"):
        wsc._selloop_options()
    monkeypatch.delenv(L47_ENV)
    assert wsc._selloop_options() == {"select_owned": 1}
    monkeypatch.setenv(L47_ENV, "1")
    monkeypatch.delenv(SEL_ENV)
    assert wsc._selloop_options() == {}
    print("L1-V6 C4 SELECT_OWNED x L47_FOLD: refused together; each alone as before", flush=True)


def test_v6_engagement_lines(monkeypatch):
    """C5: the armed members' build-time engagement lines, on the call module's "vllm." logger."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc

    recs: list = []
    h = logging.Handler()
    h.emit = recs.append  # type: ignore[method-assign]
    lg = logging.getLogger("vllm." + wsc.__name__)
    lg.addHandler(h)
    old = lg.level
    lg.setLevel(logging.INFO)
    try:
        for g in ("_SELLOOP_LOGGED", "_UAISSUE_LOGGED", "_UAUNPACK_LOGGED", "_KVW_LOGGED"):
            monkeypatch.setattr(wsc, g, False)
        monkeypatch.setattr(wsc, "UNION_ATTN", True)
        monkeypatch.setattr(wsc, "UNION_PHYS", False)
        monkeypatch.delenv(L47_ENV, raising=False)
        for k, v in ((SEL_ENV, "1"), ("VLLM_TPU_MK_UNION_ISSUE", "2"), ("VLLM_TPU_MK_UNION_UNPACK", "1")):
            monkeypatch.setenv(k, v)
        monkeypatch.setattr(envs, "VLLM_TPU_MK_QSA_WRITE_ALL", True)
        monkeypatch.setattr(envs, "VLLM_TPU_MK_QSA_WRITE_KVAL", True)
        assert wsc._selloop_options() == {"select_owned": 1}
        assert wsc._uaissue_options() == {"union_issue": 2}
        assert wsc._uaunpack_options() == {"union_unpack": True}
        assert len(wsc._kvw_extra_outs()) == 2
        monkeypatch.setattr(wsc, "_KVW_LOGGED", False)
        monkeypatch.setattr(envs, "VLLM_TPU_MK_QSA_WRITE_KVAL", False)
        assert wsc._kvw_extra_outs() == ()
        monkeypatch.setattr(wsc, "_KVW_LOGGED", False)
        monkeypatch.setattr(envs, "VLLM_TPU_MK_QSA_WRITE_ALL", False)
        n = len(recs)
        assert wsc._kvw_extra_outs() == () and len(recs) == n, "a KVW line with both flags off"
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)
    msgs = [r.getMessage() for r in recs]
    for name in ("selloop", "uaissue", "uaunpack", "wfold", "kvwall"):
        hits = [m for m in msgs if re.search(MARKERS[name], m)]
        assert len(hits) == 1, (name, msgs)
        print(f"L1-V6 C5 MARKER {name}: {hits[0]}", flush=True)
