# SPDX-License-Identifier: Apache-2.0
"""q38 L2-XUAV5: X-UA (X-UA-ISSUE + X-UA-UNPACK; X-UA-PIPE refused here) on STACK-D v5 F-v5c with the in-kernel indexer on.

Part A (whole step, CPU interpret; tests/megakernel/test_l1idx.py's `_idx_run`, so the indexed options are the served ones
built from the envs by `_d08s1_options()` + `_idx_options()`): with F1's IDX key set (scorer chunk / lanes / manager
pages, radix 2, QSPLIT 1, TABLE_SLIM 2 = the prefetched per-layer SMEM windows) plus VLLM_TPU_MK_DCP_QX_EARLY=1, the block
is `QSAFullStepGroupIdxQxBlock`; each X-UA arm ({ISSUE 0 / 2} x {UNPACK 0 / 1}) is bitwise identical to the IDX-on base
(X-UA off) at 16 and 48 rows, main-KV DCP 4 (test_wholestep._s1_same: outputs and the union state attention reads on every
rank). The same with TABLE_SLIM unset (no prefetch: X-UA-ISSUE's own `load_tables` starts the slab) x ISSUE 2. Every QSA
layer reads head pair 0 (inputs at offset 0, as X-UA-UNPACK needs). X-UA-ISSUE's slab copies are counted at trace time
through a proxy of `union_issue.slab_copy`: every wait follows exactly one start of the same layer's copy (a wait with no
start, or a second start, raises at trace time), and under the prefetch every start is `_issue_prep`'s own.

Part B (no kernel): the per-layer start record of `_LeanMixin.load_tables` / `_issue_prep` on a stub, over three layers:
load_tables reached for every layer (one start each, by load_tables), for none (one start each, by `_issue_prep`), and
mixed (layer 1 not reached: still exactly one start per layer).

Part C (no kernel): VLLM_TPU_MK_UNION_PIPE is refused with every in-kernel indexer / early-q env (ValueError from
`indexed_options()`), accepted alone; with every X-UA env unset no X-UA option is present under IDX all on.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tests.megakernel import test_l1idx as li
from tests.megakernel import test_wholestep as tw

XUA_ENVS = ("VLLM_TPU_MK_UNION_ISSUE", "VLLM_TPU_MK_UNION_UNPACK", "VLLM_TPU_MK_UNION_PIPE", "VLLM_TPU_MK_DCP_QX_EARLY")
QX = {"VLLM_TPU_MK_DCP_QX_EARLY": "1"}
# F1's IDX family at test scale (li.CFG docstring: chunk 32 at max_groups 1024 stands for the served 128) + QX-EARLY.
XCFG = {
    "XP": dict(li.CFG["ALL"], **QX),  # TABLE_SLIM 2: the prefetch (F1's served value)
    "XN": dict({k: v for k, v in li.CFG["ALL"].items() if k != "VLLM_TPU_MK_TABLE_SLIM"}, **QX),  # TABLE_SLIM unset
}
SCEN = {
    "d4_r4_16": dict(li.SCEN["d4_r4_16"]),  # 16 rows, DCP 4, per-layer tables, request loop
    "d4_r4_48": dict(R=4, nreq=12, dcp=4, loop=True, budget=64, ctx=tw.CTX_D4, page512=True, per_layer=True),  # 48 rows
}
# (scenario, config, ISSUE level, UNPACK): each compared bitwise with (scenario, config, 0, 0)
ARMS = [(s, c, i, u) for s in SCEN for c, iu in (("XP", ((2, 0), (0, 1), (2, 1))), ("XN", ((2, 0), (2, 1)))) for i, u in iu]


class _Slab:
    """Trace-time proxy of one `union_issue.slab_copy` descriptor: starts / waits are counted and balanced."""

    def __init__(self, real, book: dict) -> None:
        self.real, self.book = real, book

    def start(self) -> None:
        self.book["starts"] += 1
        self.book["open"] += 1
        assert self.book["open"] == 1, ("X-UA slab copy started twice before its wait", dict(self.book))
        self.real.start()

    def wait(self) -> None:
        self.book["waits"] += 1
        self.book["open"] -= 1
        assert self.book["open"] == 0, ("X-UA slab copy waited without a start", dict(self.book))
        self.real.wait()


def _instrument(monkeypatch) -> dict:
    from vllm_torchtpu.megakernel.wholestep import qsa_full_lean as ql
    from vllm_torchtpu.megakernel.wholestep import union_issue as ui

    book = {"starts": 0, "waits": 0, "open": 0, "self": 0}
    real = ui.slab_copy
    monkeypatch.setattr(ui, "slab_copy", lambda *a, **k: _Slab(real(*a, **k), book))
    real_prep = ql._LeanMixin._issue_prep

    def prep(self, ctx, l, union, rk):  # noqa: E741
        book["self"] += int(getattr(ctx, "_xua_slab_l", None) is not l)  # `_issue_prep` will start this layer's copy
        return real_prep(self, ctx, l, union, rk)

    monkeypatch.setattr(ql._LeanMixin, "_issue_prep", prep)
    return book


def _run(monkeypatch, scen: str, cfg: str, lvl: int, unpack: int):
    """`li._idx_run` of `scen` under `cfg` + the X-UA envs (every QSA layer at head-pair offset 0)."""
    from vllm_torchtpu.megakernel.wholestep import inputs as wi
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc

    monkeypatch.setitem(li.SCEN, scen, SCEN[scen])
    monkeypatch.setitem(li.CFG, cfg, XCFG[cfg])
    for e in XUA_ENVS:
        monkeypatch.delenv(e, raising=False)
    for e, v in QX.items():
        monkeypatch.setenv(e, v)
    if lvl:
        monkeypatch.setenv("VLLM_TPU_MK_UNION_ISSUE", str(lvl))
    if unpack:
        monkeypatch.setenv("VLLM_TPU_MK_UNION_UNPACK", "1")
    monkeypatch.setattr(wsc, "UNION_ATTN", True)
    monkeypatch.setattr(wsc, "UNION_PHYS", False)
    orig = wi.random_wholestep_inputs
    monkeypatch.setattr(wi, "random_wholestep_inputs", lambda *a, **k: orig(*a, **dict(k, off=0)))
    orig_pl = wi.add_per_layer

    def per_layer0(*a, **k):  # per-layer tables keep their own keys / pages, every layer at head pair 0
        out = orig_pl(*a, **k)
        out["off"] = out["off"] * 0
        return out

    monkeypatch.setattr(wi, "add_per_layer", per_layer0)
    return li._idx_run(monkeypatch, scen, cfg)


def _check_options(run: tuple, cfg: str, lvl: int, unpack: int, tag: str) -> None:
    opts, blk = run[8]["_opts"], run[8]["_block"]
    assert blk == "QSAFullStepGroupIdxQxBlock", (tag, blk)
    assert opts.get("radix_bits") == 2 and opts.get("union_qsplit") == 1 and opts.get("qx_early") is True, (tag, opts)
    assert bool(opts.get("table_prefetch")) == (cfg == "XP"), (tag, opts)
    assert opts.get("union_issue", 0) == lvl and bool(opts.get("union_unpack", False)) == bool(unpack), (tag, opts)
    assert "union_pipe" not in opts, (tag, opts)


@pytest.mark.parametrize(("scen", "cfg", "lvl", "unpack"), ARMS, ids=[f"{s}-{c}-i{i}u{u}" for s, c, i, u in ARMS])
def test_xuav5_idx_wholestep_identity(monkeypatch, scen, cfg, lvl, unpack):
    """X-UA arm vs the IDX-on base (X-UA off), bitwise; the class, the options and (ISSUE) the slab copies' start / wait
    balance are asserted, and under the prefetch every slab start is `_issue_prep`'s own."""
    kw = SCEN[scen]
    D, KU = kw["dcp"], li._ku(kw["budget"])
    base = _run(monkeypatch, scen, cfg, 0, 0)
    _check_options(base, cfg, 0, 0, f"{scen} {cfg} base")
    assert int(np.asarray(base[8]["off"]).max()) == 0, "every QSA layer must read head pair 0 (X-UA-UNPACK)"
    li._qcover(base, D, KU, scen)
    book = _instrument(monkeypatch)
    run = _run(monkeypatch, scen, cfg, lvl, unpack)
    _check_options(run, cfg, lvl, unpack, f"{scen} {cfg} i{lvl}u{unpack}")
    if lvl:
        assert book["starts"] >= 1 and book["starts"] == book["waits"] and book["open"] == 0, book
        assert book["self"] == (book["starts"] if cfg == "XP" else 0), book  # prefetch: `_issue_prep` starts each copy
    else:
        assert book["starts"] == book["waits"] == 0, book
    tw._s1_same(run, base, D, KU, f"{scen} {cfg} i{lvl}u{unpack} vs base")
    print(f"L2-XUAV5 IDENTITY {scen} {cfg} issue {lvl} unpack {unpack}: bitwise identical to the IDX-on base (block "
          f"{run[8]['_block']}, prefetch {cfg == 'XP'}, slab starts {book['starts']} waits {book['waits']} self-started "
          f"{book['self']})", flush=True)


def test_xuav5_idx_wholestep_raw_prefetch(monkeypatch):
    """The prefetch arm ISSUE 2 + UNPACK 1 at 16 rows WITHOUT the trace-time proxy (the plain program): bitwise the IDX-on
    base. (The m1 mutant's real failure mode is read from this node: a wait on a copy that was never started.)"""
    kw = SCEN["d4_r4_16"]
    D, KU = kw["dcp"], li._ku(kw["budget"])
    base = _run(monkeypatch, "d4_r4_16", "XP", 0, 0)
    assert int(np.asarray(base[8]["off"]).max()) == 0
    run = _run(monkeypatch, "d4_r4_16", "XP", 2, 1)
    _check_options(run, "XP", 2, 1, "raw prefetch i2u1")
    tw._s1_same(run, base, D, KU, "d4_r4_16 XP i2u1 raw vs base")
    print("L2-XUAV5 RAW d4_r4_16 XP issue 2 unpack 1: bitwise identical to the IDX-on base (no proxy)", flush=True)


# ------------------------------------------------------------------------------------------------ Part B
class _TabBase:
    def load_tables(self, ctx, l):  # noqa: E741
        ctx.log.append(("tables", l))


def _stub(monkeypatch):
    from vllm_torchtpu.megakernel.wholestep import qsa_full_lean as ql
    from vllm_torchtpu.megakernel.wholestep import union_issue as ui

    class Stub(ql._LeanMixin, _TabBase):
        pass

    s = object.__new__(Stub)
    s.union_issue = 2

    class Rec:
        def __init__(self, ctx, l):  # noqa: E741
            self.ctx, self.l = ctx, l

        def start(self):
            self.ctx.log.append(("start", self.l))

        def wait(self):
            self.ctx.log.append(("wait", self.l))

    s._issue_slab = lambda ctx, l: Rec(ctx, l)  # noqa: E731
    s._issue_cols = lambda ctx: 128  # noqa: E731
    monkeypatch.setattr(ui, "prep", lambda **k: None)
    ctx = SimpleNamespace(ex=SimpleNamespace(MB=16, T=4), scr={"qu_n": None, "qx_slab": None, "qx_pv": None},
                          extra={ql.POOL: SimpleNamespace(shape=(1, 512))}, log=[])
    return s, ctx


@pytest.mark.parametrize("reach", ["all", "none", "mixed"])
def test_xuav5_issue_slab_start_per_layer(monkeypatch, reach):
    """Three layers; `load_tables` reached for every layer / none (an override that skips it, like the prefetch) / layers
    0 and 2 only. Each layer's slab copy is started exactly once, before its wait (by `load_tables` when reached, else by
    `_issue_prep`), and the record never carries over to the next layer."""
    s, ctx = _stub(monkeypatch)
    layers = [object(), object(), object()]
    reached = {"all": (0, 1, 2), "none": (), "mixed": (0, 2)}[reach]
    want = []
    for i, l in enumerate(layers):  # noqa: E741
        if i in reached:
            s.load_tables(ctx, l)
            want += [("tables", l), ("start", l)]
        s._issue_prep(ctx, l, None, 0)
        want += ([] if i in reached else [("start", l)]) + [("wait", l)]
    assert ctx.log == want, (reach, [(k, layers.index(x)) for k, x in ctx.log])
    for l in layers:  # noqa: E741
        assert [k for k, x in ctx.log if x is l].count("start") == 1, reach
    print(f"L2-XUAV5 SLAB-START {reach}: one start per layer before its wait "
          f"({sum(1 for k, _ in ctx.log if k == 'start')} starts over 3 layers)", flush=True)


# ------------------------------------------------------------------------------------------------ Part C
REFUSED = [("VLLM_TPU_MK_SCORE_CHUNK", "128"), ("VLLM_TPU_MK_SCORE_LANES", "128"), ("VLLM_TPU_MK_SCORE_MP", "4"),
           ("VLLM_TPU_MK_SELECT_RADIX", "2"), ("VLLM_TPU_MK_UNION_QSPLIT", "1"), ("VLLM_TPU_MK_TABLE_SLIM", "2"),
           ("VLLM_TPU_MK_TABLE_SLIM", "1"), ("VLLM_TPU_MK_RING_S1", "32=7:10"), ("VLLM_TPU_MK_DCP_QX_EARLY", "1")]


def _clean(monkeypatch):
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc

    for e in XUA_ENVS + li.IDX_ENVS:
        monkeypatch.delenv(e, raising=False)
    for e, v in li.S1.items():
        monkeypatch.setenv(e, v)
    monkeypatch.setenv(li.DCP_ENV, "4")
    monkeypatch.setattr(wsc, "UNION_ATTN", True)
    monkeypatch.setattr(wsc, "UNION_PHYS", False)
    return wsc


@pytest.mark.parametrize(("env", "val"), REFUSED, ids=[f"{e}={v}" for e, v in REFUSED])
def test_xuav5_pipe_refused_with_idx_or_qx(monkeypatch, env, val):
    """VLLM_TPU_MK_UNION_PIPE=2 with one in-kernel indexer / early-q env set: `indexed_options()` raises ValueError naming
    the env; with VLLM_TPU_MK_UNION_PIPE unset the same env builds its options."""
    wsc = _clean(monkeypatch)
    monkeypatch.setenv(env, val)
    off = wsc.indexed_options()
    assert "union_pipe" not in off
    monkeypatch.setenv("VLLM_TPU_MK_UNION_PIPE", "2")
    with pytest.raises(ValueError, match="VLLM_TPU_MK_UNION_PIPE=2 is not composed"):
        wsc.indexed_options()
    print(f"L2-XUAV5 PIPE-REFUSED {env}={val}", flush=True)


def test_xuav5_pipe_alone_and_flag_off_options(monkeypatch):
    """VLLM_TPU_MK_UNION_PIPE=2 with no indexer / early-q env still builds `union_pipe` (S1 on, X-UA-PIPE (ii) needs
    B3); VLLM_TPU_MK_SELECT_RADIX=1 (its default) is not refused; with every X-UA env unset and F1's IDX family + QX on
    there is no X-UA option, and with ISSUE 2 + UNPACK 1 exactly those two appear."""
    wsc = _clean(monkeypatch)
    monkeypatch.setenv("VLLM_TPU_MK_UNION_PIPE", "2")
    monkeypatch.setenv("VLLM_TPU_MK_SELECT_RADIX", "1")
    assert wsc.indexed_options()["union_pipe"] == 2
    monkeypatch.delenv("VLLM_TPU_MK_UNION_PIPE")
    for e, v in XCFG["XP"].items():
        monkeypatch.setenv(e, v)
    off = wsc.indexed_options()
    assert not {"union_issue", "union_unpack", "union_pipe"} & set(off), sorted(off)
    assert off["table_prefetch"] is True and off["union_qsplit"] == 1 and off["radix_bits"] == 2 and off["qx_early"]
    monkeypatch.setenv("VLLM_TPU_MK_UNION_ISSUE", "2")
    monkeypatch.setenv("VLLM_TPU_MK_UNION_UNPACK", "1")
    on = wsc.indexed_options()
    assert on["union_issue"] == 2 and on["union_unpack"] is True and "union_pipe" not in on
    assert {k: v for k, v in on.items() if k not in ("union_issue", "union_unpack")} == off
    print(f"L2-XUAV5 OPTIONS: flag-off X-UA options none under IDX + QX; xua adds exactly union_issue=2 union_unpack=True",
          flush=True)
