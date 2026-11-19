# SPDX-License-Identifier: Apache-2.0
"""q38 L1-SELLOOP: the whole step's in-kernel selection loops only over this rank's owned request slots
(VLLM_TPU_MK_SELECT_OWNED 1 / 2, default off; wholestep/qsa_full_lean.py `_LeanMixin._owned_slots`).

With the request loop (`wholeloop=1`), the sharded indexer (`wholeshard=1`) and D08 S1's B3 (`slot_batch`), `_select`'s
slot loop `fori_loop(0, nreq)` holds, for a slot this rank does not own, only two reads (`group`: `qb_ck[q]`, `ctx[q]`)
and a skipped `pl.when`. The flag replaces it by a loop over the owned slots q = run * npr + i (i < npr, q < nreq):
straight-line at npr = 1 (served <= 32 rows at TP 8), a fori_loop over the owned count (mode 1) or every owned slot
unrolled (mode 2) above. Bitwise by construction.

Kernel tests run the whole step in CPU interpret mode with the union probe through test_l1idx's `_idx_run` (the served
`_d08s1_options()` + `_idx_options()` plumbing, plus `_selloop_options()` here) and compare BITWISE with
`test_wholestep._s1_same` (residual, ck_new, ring_new, index queries, kr_new / ir_new, and on every rank the union
state attention reads: qu_n, this rank's union ids / codes below the count, qi_kvl, qc_sga, qi_z). Engagement: the
trace-time record `qsa_full_lean.SELLOOP_TRACES` must show the expected path (straight / loop / unrolled) and shape
for every flag-on run and nothing for the flag-off runs. Kernel slots = max(16, ceil(rows / 16) * 16) / 4, so the
served buckets 8 / 16 / 32 / 48 / 64 rows give 4 / 4 / 8 / 12 / 16 slots, i.e. npr 1 / 1 / 1 / 2 / 2 at R = 8.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from tests.megakernel import test_l1idx as ti
from tests.megakernel import test_wholestep as tw

SEL_ENV = "VLLM_TPU_MK_SELECT_OWNED"
S1 = dict(ti.S1)
ALL = dict(ti.CFG["ALL"])  # S1 + S2P (test chunk 32 / lanes 128 / MP 4) + radix 2 + QSPLIT 1 + TSLIM 2 (prefetch)
CFG = {
    "SL_BASE": dict(S1),
    "SL_SEL1": dict(S1, **{SEL_ENV: "1"}),
    "SL_SEL2": dict(S1, **{SEL_ENV: "2"}),
    "SL_IBASE": dict(ALL),
    "SL_ISEL1": dict(ALL, **{SEL_ENV: "1"}),
    "SL_ISEL2": dict(ALL, **{SEL_ENV: "2"}),
}
_CTX = [3003, 37, 1021, 254, 2047, 511, 3070, 129]
_K = dict(loop=True, page512=True, per_layer=True)
SCEN = {
    # served shapes at TP 8 (R = 8), main-KV DCP 4: 8 / 16 / 32 / 48 / 64 rows
    "sd4_r8_8": dict(R=8, nreq=2, dcp=4, budget=128, ctx=_CTX, ring=True, **_K),
    "sd4_r8_16": dict(R=8, nreq=4, dcp=4, budget=64, ctx=_CTX, **_K),
    "sd4_r8_32": dict(R=8, nreq=8, dcp=4, budget=64, ctx=_CTX, ties=True, **_K),
    "sd4_r8_48": dict(R=8, nreq=12, dcp=4, budget=64, ctx=_CTX, **_K),
    "sd4_r8_64": dict(R=8, nreq=16, dcp=4, budget=64, ctx=_CTX, ties=True, **_K),
    # main-KV DCP 2
    "sd2_r8_16": dict(R=8, nreq=4, dcp=2, budget=64, ctx=_CTX, **_K),
    "sd2_r8_48": dict(R=8, nreq=12, dcp=2, budget=64, ctx=_CTX, ring=True, **_K),
    "sd2_r4_32": dict(R=4, nreq=8, dcp=2, budget=512, ctx=_CTX, **_K),
    "sd2_r4_48": dict(R=4, nreq=12, dcp=2, budget=64, ctx=_CTX, ties=True, **_K),
    # small controls (mutants): R = 2, DCP 2, 4 slots -> npr 2 (static loop / unrolled)
    "c_d2_r2_16": dict(R=2, nreq=4, dcp=2, budget=64, ctx=[3001, 37, 1023, 254], ring=True, **_K),
}


_ORIG_IDX: list = []


def _slots(sc: dict) -> tuple[int, int]:
    """(kernel request slots, slots per run) of a scenario, from the test's own arithmetic (not the kernel's)."""
    rows = max(16, -(-(4 * sc["nreq"]) // 16) * 16)
    nreq = rows // 4
    return nreq, -(-nreq // sc["R"])


def _note_line() -> str:
    from vllm_torchtpu.megakernel.wholestep import qsa_full_lean as ql

    src, start = inspect.getsourcelines(ql._LeanMixin._owned_slots)
    hits = [start + i for i, x in enumerate(src) if "_selloop_note(" in x]
    return f"{inspect.getsourcefile(ql._LeanMixin._owned_slots).split('/src/')[-1]}:{','.join(map(str, hits))}"


def _run(monkeypatch, scen: str, cfg: str, key: int = 17):
    """test_l1idx `_idx_run` of `scen` under `cfg`, with `_selloop_options()` merged into the indexed options (as
    `indexed_options()` does). Returns (run tuple, {trace key: count} of `_owned_slots` traces during the run)."""
    from vllm_torchtpu.megakernel.wholestep import qsa_full_lean as ql
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc

    for k, v in SCEN.items():
        monkeypatch.setitem(ti.SCEN, k, v)
    for k, v in CFG.items():
        monkeypatch.setitem(ti.CFG, k, v)
    monkeypatch.delenv(SEL_ENV, raising=False)
    if not _ORIG_IDX:  # the unpatched `_idx_options` (monkeypatch restores it after each test)
        _ORIG_IDX.append(wsc._idx_options)
    idx = _ORIG_IDX[0]
    monkeypatch.setattr(wsc, "_idx_options", lambda: {**idx(), **wsc._selloop_options()})
    before = dict(ql.SELLOOP_TRACES)
    r = ti._idx_run(monkeypatch, scen, cfg, key=key)
    traced = {k: v - before.get(k, 0) for k, v in ql.SELLOOP_TRACES.items() if v != before.get(k, 0)}
    return r, traced


def _check_engaged(scen: str, cfg: str, traced: dict) -> None:
    sc = SCEN[scen]
    mode = int(CFG[cfg].get(SEL_ENV, "0"))
    nreq, npr = _slots(sc)
    if not mode:
        assert traced == {}, (scen, cfg, "owned-slot loop traced with the flag off", traced)
        return
    path = "straight" if npr == 1 else ("loop" if mode == 1 else "unrolled")
    assert set(traced) == {(path, nreq, npr, sc["R"])}, (scen, cfg, "engagement", traced, (path, nreq, npr, sc["R"]))
    print(f"L1-SELLOOP ENGAGED {scen} {cfg}: {path}, {nreq} slots, {npr} per run over {sc['R']} runs, traced "
          f"{sum(traced.values())} times at {_note_line()}", flush=True)


# (scenario, base config, configs compared to it bitwise)
IDENTITY = [
    ("sd4_r8_8", "SL_BASE", ["SL_SEL1"]),
    ("sd4_r8_16", "SL_BASE", ["SL_SEL1", "SL_SEL2"]),
    ("sd4_r8_16", "SL_IBASE", ["SL_ISEL1"]),
    ("sd4_r8_32", "SL_BASE", ["SL_SEL1"]),
    ("sd4_r8_32", "SL_IBASE", ["SL_ISEL1"]),
    ("sd4_r8_48", "SL_BASE", ["SL_SEL1"]),
    ("sd4_r8_48", "SL_BASE", ["SL_SEL1", "SL_SEL2"]),
    ("sd4_r8_48", "SL_IBASE", ["SL_ISEL1", "SL_ISEL2"]),
    ("sd4_r8_64", "SL_BASE", ["SL_SEL1", "SL_SEL2"]),
    ("sd4_r8_64", "SL_IBASE", ["SL_ISEL1", "SL_ISEL2"]),
    ("sd2_r8_16", "SL_IBASE", ["SL_ISEL1"]),
    ("sd2_r8_48", "SL_BASE", ["SL_SEL1", "SL_SEL2"]),
    ("sd2_r4_32", "SL_IBASE", ["SL_ISEL1", "SL_ISEL2"]),
    ("sd2_r4_48", "SL_BASE", ["SL_SEL1", "SL_SEL2"]),
    ("c_d2_r2_16", "SL_BASE", ["SL_SEL1"]),
    ("c_d2_r2_16", "SL_BASE", ["SL_SEL1", "SL_SEL2"]),
]


@pytest.mark.parametrize(("scen", "base", "cfgs"), IDENTITY, ids=[f"{s}-{b}-{'+'.join(c)}" for s, b, c in IDENTITY])
def test_l1selloop_identity(monkeypatch, scen, base, cfgs):
    """Each owned-slot config is bitwise identical to its flag-off base on the scenario (test_wholestep `_s1_same`),
    with the expected owned-slot path traced (and none in the base)."""
    sc = SCEN[scen]
    D, KU = sc["dcp"], ti._ku(sc["budget"])
    b, tb = _run(monkeypatch, scen, base)
    _check_engaged(scen, base, tb)
    nreq, npr = _slots(sc)
    print(f"L1-SELLOOP BASE {scen} {base}: block {b[8]['_block']}; R {sc['R']} DCP {D} slots {nreq} per run {npr} "
          f"runs owning none {max(0, sc['R'] - -(-nreq // npr))}; options {b[8]['_opts']}", flush=True)
    ti._qcover(b, D, KU, scen)
    for cfg in cfgs:
        r, tr = _run(monkeypatch, scen, cfg)
        _check_engaged(scen, cfg, tr)
        assert r[8]["_block"] == b[8]["_block"], (cfg, r[8]["_block"], b[8]["_block"])
        tw._s1_same(r, b, D, KU, f"{scen} {cfg} vs {base}")
        print(f"L1-SELLOOP IDENTITY {scen} {cfg} vs {base}: bitwise identical (block {r[8]['_block']})", flush=True)


def test_l1selloop_comparator_control(monkeypatch):
    """Positive control of the comparison: the owned-slot run on different inputs (key 18 vs 17) is NOT identical to
    the base (the comparator and the probe are live), while the engagement record still shows the loop."""
    scen = "c_d2_r2_16"
    sc = SCEN[scen]
    b, _ = _run(monkeypatch, scen, "SL_BASE")
    r, tr = _run(monkeypatch, scen, "SL_SEL1", key=18)
    _check_engaged(scen, "SL_SEL1", tr)
    with pytest.raises(AssertionError):
        tw._s1_same(r, b, sc["dcp"], ti._ku(sc["budget"]), "control")
    print("L1-SELLOOP CONTROL c_d2_r2_16: other inputs differ (comparator live)", flush=True)


# ------------------------------------------------------------------ unit tests (no kernel)
def test_l1selloop_options_and_routing(monkeypatch):
    """No kernel run. Env unset: no option anywhere (indexed_options unchanged). "1" / "2": the option, engaged on the
    lean / L1-IDX block for the DCP size; bad values and every unsupported configuration are refused."""
    from vllm_torchtpu.megakernel.wholestep import qsa_full_idx as qi
    from vllm_torchtpu.megakernel.wholestep import qsa_full_lean as ql
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc

    for e in tuple(S1) + ti.IDX_ENVS + (SEL_ENV,):
        monkeypatch.delenv(e, raising=False)
    assert wsc._selloop_options() == {}
    assert "select_owned" not in wsc.indexed_options()
    for v in ("1", "2"):
        monkeypatch.setenv(SEL_ENV, v)
        assert wsc._selloop_options() == {"select_owned": int(v)}
        assert wsc.indexed_options()["select_owned"] == int(v)
    monkeypatch.setenv(SEL_ENV, "0")
    assert wsc._selloop_options() == {}
    monkeypatch.setenv(SEL_ENV, "3")
    with pytest.raises(ValueError):
        wsc._selloop_options()
    monkeypatch.delenv(SEL_ENV)
    base = dict(max_groups=65536, pages_per_seq=2048, carrier="fp8", n_layers=11, hbm_tables=True,
                per_request_scores=True, bounded_chunks=True, shard_index=True, union_attn=True, loop_requests=True,
                union_bounded=True, union_lean=True, slot_batch=True)
    qsa = dict(budget=512)
    for dcp, lean_cls, idx_cls in ((4, ql.QSAFullStepGroupLeanBlock, qi.QSAFullStepGroupIdxBlock),
                                   (2, ql.QSAFullStepLeanBlock, qi.QSAFullStepIdxBlock),
                                   (1, ql.QSAFullStepLeanBlock, qi.QSAFullStepIdxBlock)):
        off = qi.lean_step_block(dict(base, dcp=dcp), qsa)
        assert type(off) is lean_cls and off.select_owned == 0
        for mode in (1, 2):
            blk = qi.lean_step_block(dict(base, dcp=dcp, select_owned=mode), qsa)
            assert type(blk) is lean_cls and blk.select_owned == mode, (dcp, mode, type(blk))
            blk = qi.lean_step_block(dict(base, dcp=dcp, select_owned=mode, radix_bits=2), qsa)
            assert type(blk) is idx_cls and blk.select_owned == mode and blk.radix_bits == 2, (dcp, mode, type(blk))
        assert qi.lean_step_block(dict(base, dcp=dcp, select_owned=0), qsa).select_owned == 0
    refuse = {
        "no S1 block": dict({k: v for k, v in base.items() if k not in ("union_bounded", "union_lean", "slot_batch")}),
        "probe only": dict({k: v for k, v in base.items() if k not in ("union_bounded", "union_lean", "slot_batch")},
                           union_probe=True, debug=True),
        "slot_batch off": dict(base, slot_batch=False),
        "no request loop": dict(base, loop_requests=False),
        "no sharded indexer": dict(base, shard_index=False),
        "B3 newgrp only": dict(base, slot_batch_parts="newgrp"),
        "B3 recent only": dict(base, slot_batch_parts="recent"),
        "mode 3": dict(base, radix_bits=1),
    }
    for what, opts in refuse.items():
        mode = 3 if what == "mode 3" else 1
        with pytest.raises(AssertionError):
            qi.lean_step_block(dict(opts, dcp=4, select_owned=mode), qsa)
        print(f"L1-SELLOOP REFUSED {what}", flush=True)
    # the generated L1-IDX copy and the lean `_select` carry the same hook, once each, inside `_select`
    hook = "if not self.select_owned else self._owned_slots(ctx, nreq, npr, group, score_merge)"
    for cls in (ql._LeanMixin, qi._IdxMixin):
        assert inspect.getsource(cls._select).count(hook) == 1, cls
    print(f"L1-SELLOOP ROUTING OK; engagement record at {_note_line()}", flush=True)


def test_l1selloop_flag_changes_the_traced_kernel(monkeypatch):
    """No kernel run: the flag-on block traces `_select` through `_owned_slots` (the whole step's jaxpr, which holds the
    kernel body, at R 2, DCP 2, 16 rows), the flag-off block never does, and the two jaxprs differ while two flag-off
    traces are equal (the comparison is deterministic). The device lowering check is the AOT job's (tpu7x, d4_16 / d4_48)."""
    import jax

    from vllm_torchtpu.megakernel import kernel
    from vllm_torchtpu.megakernel.exec_spec import ExecSpec
    from vllm_torchtpu.megakernel.wholestep import inputs as wi
    from vllm_torchtpu.megakernel.wholestep import qsa_full_lean as ql
    from vllm_torchtpu.megakernel.wholestep.spec import wholestep_spec
    from jax.sharding import Mesh

    for e in tuple(S1) + ti.IDX_ENVS + (SEL_ENV,):
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv(ti.DCP_ENV, "2")
    R, T, nreq = 2, 4, 4
    ex = ExecSpec(R=R, B=nreq * T, T=T, ctx=128, interpret=True, tiles=tw._TILES)
    d = wi.random_wholestep_inputs(17, tw.G, ex, tw.L, budget=64, n_pages=tw.PAGES, identical_experts=True)
    d = wi.add_indexer_inputs(17, tw.G, ex, d, tw.L, max_groups=tw.MAX_GROUPS, ctx=[3001, 37, 1023, 254])
    d = wi.add_step_inputs(17, tw.G, ex, d, tw.L, ctx=[3001, 37, 1023, 254])
    d = wi.add_carrier(17, d)
    d["hbm_tables"] = True
    d.pop("_step_out", None)
    ti._page512(d)
    tw._dcp_split(d, 2)

    def traced(mode: int) -> tuple[str, dict]:
        indexed = dict(max_groups=tw.MAX_GROUPS, pages_per_seq=d["mtab"].shape[-1], carrier="bf16", chunk_pages=8,
                       hbm_tables=True, per_request_scores=True, bounded_chunks=True, shard_index=True,
                       union_attn=True, union_stats=False, union_phys=False, loop_requests=True, dcp=2,
                       compressed_dcp=False, union_bounded=True, union_lean=True, slot_batch=True)
        if mode:
            indexed["select_owned"] = mode
        spec = wholestep_spec(tw.G, R, L=tw.L, qsa=dict(budget=64), step=True, indexed=indexed)
        w_mk, states = wi.pack_wholestep(spec, tw.G, ex, d, tw.L)
        mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
        from vllm_torchtpu.megakernel.models.qwen38_pack import shard_step_inputs

        pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(mesh, ex, d["inp"].pos, d["inp"].x, w_mk, states)
        step, _ = kernel.make_step(mesh, spec, ex)
        before = dict(ql.SELLOOP_TRACES)
        txt = str(jax.make_jaxpr(step)(pos_mk, hid_mk, w_mk, st_mk, *wi.extras_c(d, mesh, R)))
        return txt, {k: v - before.get(k, 0) for k, v in ql.SELLOOP_TRACES.items() if v != before.get(k, 0)}

    off1, t_off1 = traced(0)
    off2, t_off2 = traced(0)
    on1, t_on1 = traced(1)
    assert t_off1 == {} and t_off2 == {}, (t_off1, t_off2)
    assert set(t_on1) == {("loop", 4, 2, 2)}, t_on1
    assert off1 == off2, "flag-off trace is not deterministic"
    assert on1 != off1, "the flag does not change the traced kernel"
    import hashlib

    h = lambda s: hashlib.sha256(s.encode()).hexdigest()[:16]  # noqa: E731
    print(f"L1-SELLOOP JAXPR R 2 DCP 2 16 rows: off {h(off1)} (x2 equal, {len(off1)} chars), on mode 1 {h(on1)} "
          f"({len(on1)} chars); traced {t_on1}", flush=True)
