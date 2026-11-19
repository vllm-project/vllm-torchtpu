# SPDX-License-Identifier: Apache-2.0
"""q38 L1-QX (X-DCP-QX-EARLY, cycle 6): the whole step's DCP q exchange starts right after `_project`, default off.

VLLM_TPU_MK_DCP_QX_EARLY=1 -> kernel option `qx_early` (qwen38_wholestep_call._qx_options; main-KV DCP 2 / 4 only) -> the
X-DCP-QX-EARLY subclass of the block `wholestep_spec` would build otherwise (wholestep/qsa_full_idx.py `_QxEarlyMixin`, routed
by its `lean_step_block`): `_project` starts the q copies (MM's pair copy at DCP 2, D14's group copies at DCP 4) once `qf_q`
is written, and `_dcp_attend` waits for them (recv, then send) just before the union attention instead of start-then-wait.

* Kernel (CPU interpret, union probe on): each scenario / config with QX on is bitwise identical to the same config off
  (test_l1idx `_idx_run` + test_wholestep `_s1_same`: residual, ck_new, ring_new, index queries, kr_new / ir_new and the
  union state on every rank), in the interpreter DMA mode the node names: "eager" (a DMA moves when it starts, so a copy
  started before `qf_q` is written sends stale data) or "on_wait" (it moves only when a semaphore is waited, so a wait moved
  after the attention, or dropped, leaves the peers' q stale). Both runs must print the same interpreter 'non-zero count at
  kernel exit' semaphore lines (a dropped wait leaves a semaphore raised). "full" nodes also check the trace-time order per
  QSA layer: project, D - 1 starts, select, D - 1 wait_recv, D - 1 wait_send, attend (off: project, select, attend and no QX
  copies); "data" nodes (the mutants' controls) skip the order check, so only the data / semaphore checks can kill a mutant.
* Routing / options / engagement log (no kernel).
"""
from __future__ import annotations

import dataclasses
import inspect
import logging
import os
import re

import pytest

from tests.megakernel import test_l1idx as tl
from tests.megakernel import test_wholestep as tw

QX_ENV = "VLLM_TPU_MK_DCP_QX_EARLY"
WSC_LOGGER = "vllm.vllm_torchtpu.megakernel_runtime.qwen38_wholestep_call"
QX_LINE = "q38 L1-QX whole step: DCP q exchange starts after _project (main-KV DCP"
QX_OFF_LINE = "q38 L1-QX whole step: VLLM_TPU_MK_DCP_QX_EARLY=1 inactive"
NZ = re.compile(r"Semaphore \d+ has non-zero count for \d+ \(global core \d+\) at kernel exit: -?\d+")
# Extra scenarios in test_l1idx.SCEN's format (added per test with monkeypatch.setitem).
SCEN_QX = {
    # 8 rows at DCP 4 (2 requests), ring that pins the f32 sum order.
    "d4_r4_8": dict(R=4, nreq=2, dcp=4, loop=True, budget=64, ctx=tw.CTX_D4, ring=True, page512=True),
    # 48 rows at DCP 4 (12 requests), request loop, tie-heavy keys.
    "d4_r4_48": dict(R=4, nreq=12, dcp=4, loop=True, budget=64, ctx=tw.CTX_D4, ties=True, page512=True),
    # 16 rows at DCP 4 over two groups (R 8), ring.
    "d4_r8_16": dict(R=8, nreq=4, dcp=4, loop=True, budget=128, ctx=[3003, 37, 1021, 254], ring=True, page512=True),
    # 8 rows at DCP 2 (one pair), ring (the DCP-2 mutant's control; small).
    "d2_r2_8": dict(R=2, nreq=2, dcp=2, loop=True, budget=128, ctx=[3001, 3003, 102, 255], ring=True, page512=True),
}
EVENTS = ("project", "start", "select", "wait_recv", "wait_send", "attend")


def _scen(monkeypatch, scen: str) -> dict:
    if scen in SCEN_QX:
        monkeypatch.setitem(tl.SCEN, scen, SCEN_QX[scen])
    return tl.SCEN[scen]


def _dma_mode(monkeypatch, mode: str) -> list:
    """The interpreter DMA mode for this test: kernel.py builds `pltpu.InterpretParams()` when ExecSpec.interpret is True,
    so a subclass whose default mode is `mode` is put in its place; returns the list of the modes of every instance built
    (the engagement check of the mode)."""
    from jax.experimental.pallas import tpu as pltpu

    real = pltpu.InterpretParams
    made: list = []

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class _Mode(real):
        dma_execution_mode: str = mode

        def __post_init__(self) -> None:
            super().__post_init__()
            made.append(self.dma_execution_mode)

    monkeypatch.setattr(pltpu, "InterpretParams", _Mode)
    return made


class _Proxy:
    """A q-copy descriptor that records start / wait_recv / wait_send at trace time and forwards the call."""

    def __init__(self, d, ev: list) -> None:
        self._d, self._ev = d, ev

    def start(self):
        self._ev.append("start")
        return self._d.start()

    def wait_recv(self):
        self._ev.append("wait_recv")
        return self._d.wait_recv()

    def wait_send(self):
        self._ev.append("wait_send")
        return self._d.wait_send()


def _spy(monkeypatch, cur: list) -> None:
    """Trace-time events into the list `cur[0]`: 'project' when the base projection returns, 'select' / 'attend' when the
    stage-C selection / union attention starts (wrapped where each is defined, so the QX subclass and its base resolve to the
    same wrapper), and the QX copies' start / wait_recv / wait_send."""
    from vllm_torchtpu.megakernel.wholestep import qsa_full as qf
    from vllm_torchtpu.megakernel.wholestep import qsa_full_c as qc
    from vllm_torchtpu.megakernel.wholestep import qsa_full_idx as qi
    from vllm_torchtpu.megakernel.wholestep import qsa_full_lean as ql

    real_proj = qf.QSAFullBlock._project

    def proj(self, *a, **k):
        out = real_proj(self, *a, **k)
        cur[0].append("project")
        return out

    monkeypatch.setattr(qf.QSAFullBlock, "_project", proj)
    for owner, name, tag in ((qc.QSAFullStepBlock, "_select", "select"), (ql._LeanMixin, "_select", "select"),
                             (qi._IdxMixin, "_select", "select"), (qc.QSAFullStepBlock, "_attend_unions", "attend"),
                             (ql._ProbeMixin, "_attend_unions", "attend")):
        real = owner.__dict__[name]

        def wrap(self, *a, _real=real, _tag=tag, **k):
            cur[0].append(_tag)
            return _real(self, *a, **k)

        monkeypatch.setattr(owner, name, wrap)
    real_copies = qi._QxEarlyMixin._qx_copies
    monkeypatch.setattr(qi._QxEarlyMixin, "_qx_copies", lambda self, ctx: [_Proxy(d, cur[0]) for d in real_copies(self, ctx)])


def _dedup(ev: list) -> list:
    out: list = []
    for e in ev:
        if not out or out[-1] != e:
            out.append(e)
    return out


def _check_order(ev_off: list, ev_on: list, D: int) -> int:
    """Off: (project, select, attend) repeated, no QX copy. On: (project, start, select, wait_recv, wait_send, attend)
    repeated, with exactly D - 1 of each copy event per layer. Returns the number of traced QSA layers."""
    off = _dedup(ev_off)
    assert not {"start", "wait_recv", "wait_send"} & set(ev_off), ("QX copies with the flag off", ev_off[:12])
    n = len(off) // 3
    assert n >= 1 and off == ["project", "select", "attend"] * n, ("flag-off order", off[:12])
    on = _dedup(ev_on)
    m = len(on) // 6
    assert m == n and on == list(EVENTS) * m, ("flag-on order", on[:14], n)
    for e in ("start", "wait_recv", "wait_send"):
        assert ev_on.count(e) == m * (D - 1), (e, ev_on.count(e), m, D)
    return n


def _run(monkeypatch, capfd, scen: str, cfg: str, on: bool):
    if on:
        monkeypatch.setenv(QX_ENV, "1")
    else:
        monkeypatch.delenv(QX_ENV, raising=False)
    capfd.readouterr()
    r = tl._idx_run(monkeypatch, scen, cfg)
    out, err = capfd.readouterr()
    nz = sorted(NZ.findall(out + err))
    return r, nz


# (scenario, base config of test_l1idx.CFG, interpreter DMA mode, checks): QX on vs off, bitwise.
IDENT = [
    ("d4_r4_8", "BASE", "eager", "full"),  # DCP 4, 8 rows, S1
    ("d4_r4_8", "ALLNP", "on_wait", "full"),  # + the IDX package (S2P, radix 2, QSPLIT 1, TSLIM 1)
    ("d4_r4_16", "ALL", "eager", "full"),  # 16 rows, per-layer tables + TSLIM 2 prefetch
    ("d4_r4_16", "BASE", "on_wait", "full"),
    ("d4_r4_32", "ALLNP", "eager", "full"),  # 32 rows unrolled, tie-heavy
    ("d4_r4_32", "BASE", "on_wait", "full"),
    ("d4_r4_48", "ALLNP", "on_wait", "full"),  # 48 rows
    ("d4_r4_64", "BASE", "eager", "full"),  # 64 rows, request loop
    ("d4_r4_64", "ALLNP", "on_wait", "full"),
    ("d4_r8_16", "ALLNP", "on_wait", "full"),  # R 8 (two DCP groups), 16 rows
    ("d4_r8_4", "MM", "on_wait", "full"),  # D14 without S1 (the probe class)
    ("d2_r2_16", "ALL", "eager", "full"),  # DCP 2 (MM's pair), 16 rows, per-layer tables, budget 512
    ("d2_r2_16", "BASE", "on_wait", "full"),
    ("d2_r4_64", "ALLNP", "on_wait", "full"),  # DCP 2 over two pairs, 64 rows
    ("d2_r4_64", "BASE", "eager", "full"),
    # the mutants' controls (data + semaphore checks only)
    ("d4_r8_4", "BASE", "eager", "data"),
    ("d4_r8_4", "BASE", "on_wait", "data"),
    ("d2_r2_8", "BASE", "on_wait", "data"),
]


@pytest.mark.parametrize(("scen", "cfg", "mode", "chk"), IDENT, ids=[f"{s}-{c}-{m}-{k}" for s, c, m, k in IDENT])
def test_qx_identity(monkeypatch, capfd, scen, cfg, mode, chk):
    """QX on vs off on config `cfg`, same DMA mode: bitwise identical outputs and union state, the QX class built only with
    the env and `qx_early` reaching the block, the DMA mode engaged, the same exit-semaphore lines; "full": the trace order."""
    kw = _scen(monkeypatch, scen)
    D, KU = kw["dcp"], tl._ku(kw["budget"])
    made = _dma_mode(monkeypatch, mode)
    ev0: list = []
    ev1: list = []
    cur = [ev0]
    if chk == "full":
        _spy(monkeypatch, cur)
    b, nz0 = _run(monkeypatch, capfd, scen, cfg, False)
    cur[0] = ev1
    r, nz1 = _run(monkeypatch, capfd, scen, cfg, True)
    assert made and set(made) == {mode}, (mode, made)
    assert "Qx" not in b[8]["_block"] and "qx_early" not in b[8]["_opts"], (b[8]["_block"], b[8]["_opts"])
    assert r[8]["_block"].endswith("QxBlock") and r[8]["_opts"].get("qx_early") is True, (r[8]["_block"], r[8]["_opts"])
    tl._qcover(b, D, KU, scen)
    tw._s1_same(r, b, D, KU, f"{scen} {cfg} {mode} QX on vs off")
    assert nz1 == nz0, ("exit-semaphore lines differ", nz0[:4], nz1[:4])
    n = _check_order(ev0, ev1, D) if chk == "full" else 0
    print(f"L1-QX IDENTITY {scen} {cfg} {mode} ({chk}): bitwise identical (block {r[8]['_block']} vs {b[8]['_block']}); "
          f"DMA mode {mode} engaged ({len(made)} InterpretParams built); exit-semaphore lines {len(nz0)} both runs"
          + (f"; trace order project, {D - 1} x start, select, {D - 1} x wait_recv, {D - 1} x wait_send, attend on {n} "
             f"traced QSA layer(s) (off: project, select, attend)" if chk == "full" else ""), flush=True)


# ------------------------------------------------------------------ routing / options / engagement (no kernel)
def test_qx_options_routing_and_log(monkeypatch):
    """Flag off: no option anywhere. QX=1 at main-KV DCP 2 / 4: {"qx_early": True} from `_idx_options()` (alone or with the
    S1 / IDX envs) and `indexed_options()`, exactly one engagement line per process; at DCP 1 and under compressed-key DCP no
    option and the inactive line. Routing: for every option set and DCP 2 / 4 the block is the QX subclass of exactly the
    class built without `qx_early` (spec.py's rule), `qx_early` False keeps the subclass with the option off, and DCP 1 with
    the option is refused. Bad env values are refused."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.megakernel.wholestep import qsa_full_c as qc
    from vllm_torchtpu.megakernel.wholestep import qsa_full_idx as qi
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc

    def clear():
        for e in tuple(tl.S1) + tl.IDX_ENVS + (QX_ENV, tl.DCP_ENV, "VLLM_TPU_QSA_COMPRESSED_DCP"):
            monkeypatch.delenv(e, raising=False)
        monkeypatch.setattr(wsc, "_QX_LOGGED", False)

    clear()
    monkeypatch.setenv(tl.DCP_ENV, "4")
    assert wsc._qx_options() == {} and wsc._idx_options() == {}
    assert "qx_early" not in wsc.indexed_options()
    base = dict(max_groups=65536, pages_per_seq=2048, carrier="fp8", n_layers=11, hbm_tables=True,
                per_request_scores=True, bounded_chunks=True, shard_index=True, union_attn=True, loop_requests=True)
    qsa = dict(budget=512)
    sets = {"MM": {}, "PROBE": dict(union_probe=True, debug=True), "S1": dict(union_bounded=True, union_lean=True, slot_batch=True),
            "B1": dict(union_bounded=True), "IDX": dict(union_lean=True, radix_bits=2, union_qsplit=1),
            "IDXMM": dict(union_qsplit=1)}
    seen = set()
    for dcp in (2, 4):
        for name, extra in sets.items():
            opts = dict(base, dcp=dcp, **extra)
            ref = qi.lean_step_block(dict(opts), qsa) or (
                qc.QSAFullStepGroupBlock if dcp > 2 else qc.QSAFullStepBlock)(**opts, **qsa)
            assert qi.qx_base_class(opts) is type(ref), (dcp, name, qi.qx_base_class(opts), type(ref))
            for flag in (True, False):
                blk = qi.lean_step_block(dict(opts, qx_early=flag), qsa)
                mro = type(blk).__mro__
                assert mro[1] is qi._QxEarlyMixin and mro[2:] == type(ref).__mro__ \
                    and type(blk) is qi.QX_CLASSES[type(ref)], (dcp, name, flag, type(blk), type(ref))
                assert blk.qx_early is flag and blk.dcp == dcp, (dcp, name, flag)
                for k in ("radix_bits", "union_qsplit", "union_bounded", "union_lean", "slot_batch", "union_probe"):
                    if hasattr(ref, k):
                        assert getattr(blk, k) == getattr(ref, k), (dcp, name, k)
            seen.add(type(ref).__name__)
    assert len(seen) == 8 and len(qi.QX_CLASSES) == 8, sorted(seen)
    with pytest.raises(AssertionError):
        qi.lean_step_block(dict(base, dcp=1, union_lean=True, qx_early=True), qsa)

    class _H(logging.Handler):  # on the module's own logger: vLLM's "vllm" logger need not propagate to the root
        def __init__(self) -> None:
            super().__init__()
            self.msgs: list = []

        def emit(self, record: logging.LogRecord) -> None:
            self.msgs.append((record.name, record.getMessage()))

    lg, h = logging.getLogger(WSC_LOGGER), _H()
    old_level = lg.level
    lg.addHandler(h)
    lg.setLevel(logging.INFO)
    try:
        for dcp in ("4", "2"):
            clear()
            monkeypatch.setenv(QX_ENV, "1")
            monkeypatch.setenv(tl.DCP_ENV, dcp)
            h.msgs.clear()
            assert wsc._qx_options() == {"qx_early": True}
            assert wsc._idx_options() == {"qx_early": True}
            for e, v in tl.S1.items():
                monkeypatch.setenv(e, v)
            monkeypatch.setenv("VLLM_TPU_MK_SELECT_RADIX", "2")
            monkeypatch.setenv("VLLM_TPU_MK_UNION_QSPLIT", "1")
            assert wsc._idx_options() == {"qx_early": True, "radix_bits": 2, "union_qsplit": 1}
            io = wsc.indexed_options()
            assert io["qx_early"] is True and io["dcp"] == int(dcp) and io["union_lean"] is True, io
            lines = [m for n, m in h.msgs if n == WSC_LOGGER and QX_LINE in m]
            assert len(lines) == 1 and lines[0].endswith(f"DCP {dcp})"), h.msgs
        for dcp, cd in (("1", "0"), ("2", "1")):
            clear()
            monkeypatch.setenv(QX_ENV, "1")
            if dcp != "1":
                monkeypatch.setenv(tl.DCP_ENV, dcp)
            monkeypatch.setenv("VLLM_TPU_QSA_COMPRESSED_DCP", cd)
            h.msgs.clear()
            assert wsc._qx_options() == {} and "qx_early" not in wsc._idx_options()
            assert [m for n, m in h.msgs if QX_OFF_LINE in m] and not [m for n, m in h.msgs if QX_LINE in m], h.msgs
    finally:
        lg.removeHandler(h)
        lg.setLevel(old_level)
    src, start = inspect.getsourcelines(wsc._qx_options)
    at = start + next(i for i, x in enumerate(src) if '"q38 L1-QX whole step: DCP q exchange starts after _project' in x)
    src2, start2 = inspect.getsourcelines(qi._QxEarlyMixin._project)
    at2 = start2 + next(i for i, x in enumerate(src2) if "d.start()" in x)
    src3, start3 = inspect.getsourcelines(qi._QxEarlyMixin._dcp_attend)
    at3 = start3 + next(i for i, x in enumerate(src3) if "d.wait_recv()" in x)
    print(f"L1-QX ENGAGEMENT line {lines[0]!r} logged once per process at "
          f"src/vllm_torchtpu/megakernel_runtime/qwen38_wholestep_call.py:{at} (logger {WSC_LOGGER}); copies start at "
          f"src/vllm_torchtpu/megakernel/wholestep/qsa_full_idx.py:{at2} (_QxEarlyMixin._project), waited at :{at3} "
          f"(_QxEarlyMixin._dcp_attend)", flush=True)
    clear()
    monkeypatch.setenv(QX_ENV, "0")
    monkeypatch.setenv(tl.DCP_ENV, "4")
    assert wsc._qx_options() == {}
    monkeypatch.setenv(QX_ENV, "2")
    with pytest.raises(ValueError):
        getattr(envs, QX_ENV)
    assert os.path.basename(qi.__file__) == "qsa_full_idx.py"
