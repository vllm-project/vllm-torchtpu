# SPDX-License-Identifier: Apache-2.0
"""q38 L6-GATE: the batch-size gates of the CARRY and STEADY levers
(VLLM_TPU_L6_CARRY_MIN_REQS / VLLM_TPU_L6_STEADY_MIN_REQS, core/l6_ab.py
GATES / gate_step, runner/tpu_runner.py TPUModelRunner._l6_gate_step).

Host-only; runs in a CDK CPU pod. A gated step runs the lever off for that
step only, exactly as a step of an off window does, so:

A. l6_ab: a lever is off on a step whose scheduled batch has fewer than N
   requests and on again at N and above; gated steps are counted; the window
   line still names the window's levers; a window-off lever is never counted.
B. the runner hook: execute_model calls it after begin_step and before
   _update_states; it counts this step's scheduled requests (not the batch the
   previous step left); it drops STEADY's caches exactly when it turns STEADY
   off.
C. CARRY + STEADY on, through the real _prepare_inputs / _modify_prev_results
   (l6_carry_harness.py) over admissions and finishes that move the batch
   across N upward and downward many times: every output that leaves
   _prepare_inputs equals the all-off run at every step, for each gate alone
   and both; gated steps run no CARRY code and make the stock uploads; steps
   at or above N carry where the batch allows it.
D. STEADY's fast paths on real BlockTables / builders (test_l6_steady.py's
   world) under the gate: metadata, reorder + chunk count, upstream reorder
   and adapter tables equal the all-off results; gated steps take the stock
   path with the caches dropped.
"""

from __future__ import annotations

import ast
import os
import pathlib
import random
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

import l6_carry_harness as H
import test_l6_steady as S
from vllm_torchtpu.core import l6_ab
from vllm_torchtpu.runner import decode_carry, l6_steady, tpu_runner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

P = 20000  # A/B window period, ms
NS_PER_MS = 1_000_000
CARRY = "VLLM_TPU_L6_CARRY_MIN_REQS"
STEADY = "VLLM_TPU_L6_STEADY_MIN_REQS"


def _t(window: int) -> int:
    return int((window + 0.5) * P * NS_PER_MS)


@pytest.fixture
def g(monkeypatch):
    """Clean L6 env; `setup(**env)` sets the flags and re-resolves l6_ab."""

    def clear():
        for k in list(os.environ):
            if k.startswith("VLLM_TPU_L6_") or k in (
                "VLLM_TPU_HOST_AB_PERIOD_MS",
                "VLLM_TPU_D1N_AB_PERIOD_MS",
                "VLLM_TPU_SERVED_AB",
            ):
                monkeypatch.delenv(k, raising=False)

    clear()
    monkeypatch.setenv("VLLM_TPU_POSTFWD_FUSED", "1")
    lines: list[str] = []
    monkeypatch.setattr(l6_ab.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    clock = {"t": _t(100)}
    monkeypatch.setattr(l6_ab, "_now_ns", lambda: clock["t"])
    l6_steady._CHECK[0] = 0

    def setup(**env):
        clear()
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        l6_ab._resolve()
        return l6_ab

    yield NS(setup=setup, lines=lines, clock=clock, monkeypatch=monkeypatch)
    clear()
    l6_ab._resolve()
    l6_steady._CHECK[0] = None


def _levers(**extra):
    return dict(VLLM_TPU_L6_DECODE_CARRY="1", VLLM_TPU_L6_STEADY_STEP="1", **extra)


def _so(n: int):
    return NS(num_scheduled_tokens={f"s{i}": 4 for i in range(n)})


def _bare_runner(num_reqs: int = 0):
    r = TPUModelRunner.__new__(TPUModelRunner)
    r.input_batch = NS(num_reqs=num_reqs, req_ids=[f"p{i}" for i in range(num_reqs)])
    return r


# ---------------------------------------------------------------- A. l6_ab


def test_gate_flags_parse_default_off_and_reject_negatives(g):
    m = g.setup()
    assert m.GATE == 0 and m.MIN_REQS == {}
    assert m.GATES == {"carry": CARRY, "steady": STEADY}
    m = g.setup(**{CARRY: "0", STEADY: ""})
    assert m.GATE == 0 and m.MIN_REQS == {}
    m = g.setup(**{CARRY: "9"})
    assert m.MIN_REQS == {"carry": 9} and m.GATE == 9
    m = g.setup(**{CARRY: "3", STEADY: "7"})
    assert m.MIN_REQS == {"carry": 3, "steady": 7} and m.GATE == 7
    g.monkeypatch.setenv(STEADY, "-1")
    with pytest.raises(ValueError):
        l6_ab._resolve()


def test_gate_turns_a_lever_off_below_n_for_that_step_only(g):
    m = g.setup(**_levers(VLLM_TPU_L6_ENVS_CACHE="1", **{CARRY: "4", STEADY: "6"}))
    assert m.ACTIVE and m.MIN_REQS == {"carry": 4, "steady": 6} and m.GATE == 6
    seq = [0, 1, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 4, 6, 3, 8, 3, 3, 4, 4, 5, 5, 6, 6]
    for n in seq:
        m.begin_step()
        off = m.gate_step(n)
        assert m.on("carry") is (n >= 4), n
        assert m.on("steady") is (n >= 6), n
        assert m.on("envcache") is True, n  # an ungated lever
        assert set(off) == {lv for lv, k in (("carry", 4), ("steady", 6)) if n < k}, (n, off)
    counts = m._state["counts"]
    assert counts.get("carry.why_minreqs", 0) == sum(n < 4 for n in seq), counts
    assert counts.get("steady.minreqs", 0) == sum(n < 6 for n in seq), counts


def test_window_line_names_the_window_levers_and_counts_gated_steps(g):
    m = g.setup(**_levers(**{CARRY: "4", STEADY: "4"}))
    for n in (5, 2, 3):  # the window's last step is gated
        m.begin_step()
        m.gate_step(n)
    assert not m.on("carry") and not m.on("steady")
    g.clock["t"] += P * NS_PER_MS
    m.begin_step()
    assert len(g.lines) == 1, g.lines
    assert g.lines[0].endswith(
        "class=uniform steps=3 on=carry+steady counts=carry.why_minreqs=2,steady.minreqs=2"
    ), g.lines[0]
    m.gate_step(6)
    assert m.on("carry") and m.on("steady")


def test_gate_counts_only_levers_their_window_turns_on(g):
    m = g.setup(
        VLLM_TPU_L6_AB_PERIOD_MS=str(P),
        VLLM_TPU_L6_DECODE_CARRY="ab:1",
        VLLM_TPU_L6_STEADY_STEP="1",
        **{CARRY: "4", STEADY: "4"},
    )
    for window in (100, 101):  # class 0: carry off; class 1: carry on
        g.clock["t"] = _t(window)
        for n in (2, 5, 2, 5, 2):
            m.begin_step()
            m.gate_step(n)
            assert m.on("carry") is (window == 101 and n >= 4), (window, n)
            assert m.on("steady") is (n >= 4), (window, n)
    g.clock["t"] = _t(102)
    m.begin_step()
    assert len(g.lines) == 2, g.lines
    assert g.lines[0].endswith("class=0 steps=5 on=steady counts=steady.minreqs=3"), g.lines[0]
    assert g.lines[1].endswith(
        "class=1 steps=5 on=carry+steady counts=carry.why_minreqs=3,steady.minreqs=3"
    ), g.lines[1]


def test_gate_without_a_lever_is_inert(g):
    m = g.setup(**{CARRY: "4", STEADY: "4"})
    assert not m.ACTIVE and m.GATE == 4
    m.begin_step()
    assert m.gate_step(1) == () and m._state["counts"] == {}
    assert not m.on("carry") and not m.on("steady")


# ---------------------------------------------------------------- B. the runner hook


def _execute_model_ast():
    tree = ast.parse(pathlib.Path(tpu_runner.__file__).read_text())
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "execute_model")


def test_execute_model_runs_the_gate_after_begin_step_and_before_update_states():
    fn = _execute_model_ast()
    outer = [
        (i, n) for i, n in enumerate(fn.body)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "_l6_ab.ACTIVE or _l6_envcache.CENSUS"
    ]
    assert len(outer) == 1, [ast.unparse(n.test) for _, n in outer]
    i_outer, block = outer[0]
    stmts = [ast.unparse(s) for s in block.body]
    i_begin = next(i for i, s in enumerate(stmts) if s.startswith("_l6_ab.begin_step("))
    i_drop = next(i for i, s in enumerate(stmts) if "_l6_steady.drop(self)" in s)
    gates = [i for i, s in enumerate(block.body) if isinstance(s, ast.If) and ast.unparse(s.test) == "_l6_ab.GATE"]
    assert len(gates) == 1, stmts
    gate = block.body[gates[0]]
    assert [ast.unparse(s) for s in gate.body] == ["self._l6_gate_step(scheduler_output)"] and not gate.orelse
    assert i_begin < i_drop < gates[0], stmts
    upd = next(
        i for i, n in enumerate(fn.body) if "self._update_states(scheduler_output)" in ast.unparse(n)
    )
    assert i_outer < upd, (i_outer, upd)


def test_gate_hook_counts_this_steps_scheduled_requests(g):
    g.setup(**_levers(**{CARRY: "4", STEADY: "4"}))
    # Before _update_states the batch still holds the previous step's 2 requests; this step schedules 5.
    r = _bare_runner(2)
    l6_ab.begin_step()
    r._l6_gate_step(_so(5))
    assert l6_ab.on("carry") and l6_ab.on("steady")
    # The previous step's 6 requests, 3 of which finished: this step schedules 3.
    r = _bare_runner(6)
    l6_ab.begin_step()
    r._l6_gate_step(_so(3))
    assert not l6_ab.on("carry") and not l6_ab.on("steady")
    # Exactly N scheduled: on.
    r = _bare_runner(3)
    l6_ab.begin_step()
    r._l6_gate_step(_so(4))
    assert l6_ab.on("carry") and l6_ab.on("steady")


def test_gate_hook_drops_steady_caches_only_when_it_turns_steady_off(g):
    g.setup(**_levers(**{STEADY: "4"}))
    r = _bare_runner()
    am, ro = object(), object()
    r._l6_steady_am, r._l6_steady_ro = am, ro
    l6_ab.begin_step()
    r._l6_gate_step(_so(5))
    assert r._l6_steady_am is am and r._l6_steady_ro is ro
    l6_ab.begin_step()
    r._l6_gate_step(_so(3))
    assert "_l6_steady_am" not in r.__dict__ and "_l6_steady_ro" not in r.__dict__
    # A CARRY-only gate leaves STEADY and its caches alone.
    g.setup(**_levers(**{CARRY: "4"}))
    r._l6_steady_am, r._l6_steady_ro = am, ro
    l6_ab.begin_step()
    r._l6_gate_step(_so(2))
    assert not l6_ab.on("carry") and l6_ab.on("steady")
    assert r._l6_steady_am is am and r._l6_steady_ro is ro


# ---------------------------------------------------------------- C. CARRY + STEADY through _prepare_inputs

SHAPES = {
    "b3_padded": dict(nreq=3, max_num_reqs=8, paddings=(16, 32, 64), mamba=True, comb=True),
    "pooled": dict(nreq=3, max_num_reqs=8, paddings=(16, 32, 64), mamba=False, comb=True, pooled=True),
}
CONFIGS = {"carry": {CARRY: 4}, "steady": {STEADY: 4}, "both": {CARRY: 4, STEADY: 5}}
# Batch-size targets as (target, steps): the batch moves one request a step toward the target (a join or a
# leave) and dwells there, crossing N = 4 upward 9 times and downward 8 times, with the immediate turn-around
# (4 -> 3 -> 4) and the low end (1, 2) included.
_SEGMENTS = [(3, 2), (4, 5), (3, 2), (5, 6), (3, 4), (4, 5), (2, 4), (4, 6), (3, 1), (4, 5), (6, 7), (3, 4),
             (1, 4), (4, 7), (3, 2), (4, 5), (5, 4), (2, 5), (4, 6), (3, 2), (4, 5)]
_TARGETS = [t for t, d in _SEGMENTS for _ in range(d)]
STREAM_STEPS = len(_TARGETS)  # 91


def _stream(nreq0: int):
    """(events, sizes): the harness events per step and the batch size after each step's edits."""
    events, sizes, n = {}, [], nreq0
    for s, tgt in enumerate(_TARGETS):
        if n < tgt:
            events[s] = ("join",)
            n += 1
        elif n > tgt:
            events[s] = ("leave",)
            n -= 1
        sizes.append(n)
    return events, sizes


def _run_carry(g, shape, env, events, *, seed):
    """One harness engine with `env` set; returns (digests, sim, notes, calls, bound)."""
    g.setup(**env)
    notes: list[tuple[int, str, str]] = []
    calls: list[tuple[int, str]] = []
    bound: list[int] = []
    cur = {"s": -1}
    real_note = l6_ab.note
    g.monkeypatch.setattr(l6_ab, "note", lambda lever, key, n=1: (notes.append((cur["s"], lever, key)),
                                                                   real_note(lever, key, n))[1])
    for fn in ("begin", "_check", "fused_prepare", "record_rejected", "reuse_upload", "late_upload", "end"):
        real = getattr(decode_carry, fn)
        g.monkeypatch.setattr(decode_carry, fn,
                              lambda *a, real=real, fn=fn, **k: (calls.append((cur["s"], fn)), real(*a, **k))[1])
    real_bound = l6_steady.seq_lens_bound
    g.monkeypatch.setattr(l6_steady, "seq_lens_bound",
                          lambda *a, **k: (bound.append(cur["s"]), real_bound(*a, **k))[1])

    def on_step(sim):
        cur["s"] = sim.n

    digests, sim = H.run(shape, STREAM_STEPS, seed=seed, events=events, on_step=on_step)
    g.monkeypatch.setattr(l6_ab, "note", real_note)
    return digests, sim, notes, calls, bound


_CASES = [(c, s) for c in sorted(CONFIGS) for s in sorted(SHAPES)]


@pytest.mark.parametrize(("config", "shape"), _CASES, ids=[f"{c}-{s}" for c, s in _CASES])
def test_gated_stream_matches_all_off(g, config, shape):
    undo = H.install_eager_programs()
    try:
        _gated_stream(g, CONFIGS[config], SHAPES[shape], seed=31 + sorted(SHAPES).index(shape))
    finally:
        undo()


def _gated_stream(g, gates, shape, *, seed):
    events, sizes = _stream(shape["nreq"])
    n_c, n_s = gates.get(CARRY, 0), gates.get(STEADY, 0)
    up = sum(1 for a, b in zip(sizes, sizes[1:]) if a < 4 <= b)
    down = sum(1 for a, b in zip(sizes, sizes[1:]) if a >= 4 > b)
    assert up >= 8 and down >= 8 and min(sizes) == 1 and max(sizes) == 6, (up, down, sizes)

    base, bsim, _, _, _ = _run_carry(g, shape, {}, events, seed=seed)
    env = _levers(**{k: str(v) for k, v in gates.items()})
    run, rsim, notes, calls, bound = _run_carry(g, shape, env, events, seed=seed)

    # Every output that leaves _prepare_inputs equals the all-off run, at every step.
    assert H.first_difference(base, run) is None, H.first_difference(base, run)

    steps = range(STREAM_STEPS)
    gated_c = {s for s in steps if sizes[s] < n_c}
    gated_s = {s for s in steps if sizes[s] < n_s}
    carry_notes = {s: {k for t, lv, k in notes if t == s and lv == "carry"} for s in steps}
    steady_min = sorted(t for t, lv, k in notes if lv == "steady" and k == "minreqs")
    called = {}
    for s, fn in calls:
        called.setdefault(s, set()).add(fn)

    # CARRY: exactly the steps below N are gated; they run no CARRY code and make the stock uploads.
    assert {s for s in steps if "why_minreqs" in carry_notes[s]} == gated_c
    for s in gated_c:
        assert carry_notes[s] == {"why_minreqs"}, (s, carry_notes[s])
        assert s not in called, (s, called.get(s))
        assert rsim.uploads[s] == bsim.uploads[s], (s, rsim.uploads[s], bsim.uploads[s])
    # Steps at or above N run the lever: they carry unless the batch changed this step, a request joined the
    # step before (its first scheduled step has no draft source), or the step before did not run the lever.
    ungated = lambda s: s not in gated_c  # noqa: E731
    join = lambda s: "join" in events.get(s, ())  # noqa: E731
    leave = lambda s: "leave" in events.get(s, ())  # noqa: E731
    want_carried = {
        s for s in steps
        if s >= 1 and ungated(s) and ungated(s - 1) and not join(s - 1) and not join(s) and not leave(s)
    }
    got_carried = {s for s in steps if "carried" in carry_notes[s]}
    assert got_carried == want_carried, (sorted(got_carried ^ want_carried), sizes)
    for s in set(steps) - gated_c - want_carried:
        assert "uploaded" in carry_notes[s] and "begin" in called.get(s, ()), (s, carry_notes[s])
        whys = {k for k in carry_notes[s] if k.startswith("why_")}
        assert len(whys) == 1 and whys <= {"why_stale", "why_nostate", "why_batch"}, (s, carry_notes[s])
    assert len(want_carried) >= 20, len(want_carried)
    if n_c:
        assert len(gated_c) >= 15, len(gated_c)

    # STEADY: exactly the steps below its N are gated (counted, and _prepare_inputs computes no steady bound).
    assert steady_min == sorted(gated_s), (steady_min, sorted(gated_s))
    assert sorted(bound) == sorted(set(steps) - gated_s), (sorted(bound), sorted(gated_s))
    if n_s:
        assert len(gated_s) >= 15, len(gated_s)
    print(f"GATE_STREAM gates={gates} carried={len(got_carried)} carry_gated={len(gated_c)} "
          f"steady_gated={len(gated_s)} up={up} down={down}")


# ---------------------------------------------------------------- D. STEADY fast paths under the gate

_D_TARGETS = [5] * 10 + [2] * 8 + [6] * 12 + [3] * 6 + [4] * 10 + [7] * 10 + [1] * 6 + [4] * 8 + [3] * 3 + [4] * 5


def _gstep(p):
    """One worker step as execute_model and _prepare_inputs run it on the lever-on runner p.a (switches, the
    lever-off drop, the gate hook, then the metadata walk with the steady bound when STEADY is on), against
    the all-off replica p.b. Returns (outcome, gated)."""
    p.t += 1
    l6_ab.begin_step()
    if not l6_steady.on(p.a):
        l6_steady.drop(p.a)
    p.a._l6_gate_step(NS(num_scheduled_tokens={q["id"]: 1 + S.K for q in p.wa.reqs}))
    steady = l6_steady.on(p.a)
    if not steady:
        assert "_l6_steady_am" not in p.a.__dict__ and "_l6_steady_ro" not in p.a.__dict__
    before = S._counts()
    sa = S.Step(p.wa, p.t)
    sb = S.Step(p.wb, p.t)
    sb.positions, sb.seq_lens, sb.qsl, sb.rd, sb.mrd, sb.offsets = (
        sa.positions, sa.seq_lens, sa.qsl, sa.rd, sa.mrd, sa.offsets
    )
    out_a, ctx_a = S._call(p.a, sa, steady)
    out_b, ctx_b = S._call(p.b, sb, False)
    S._same(out_a, ctx_a, out_b, ctx_b)
    S._step_fields(out_a, p.a, ctx_a)
    if p.prev_a is not None:
        for name in out_a:
            assert out_a[name] is not p.prev_a[name], name
    p.prev_a = out_a
    after = S._counts()
    got = [k for k in ("fast", "full", "check") if after.get(f"steady.{k}", 0) > before.get(f"steady.{k}", 0)]
    return (got[0] if got else None), not steady


@pytest.mark.parametrize("config", ["steady", "both", "carry"])
def test_steady_metadata_under_the_gate_matches_all_off(g, config):
    gates = {"steady": {STEADY: "4"}, "both": {CARRY: "3", STEADY: "4"}, "carry": {CARRY: "4"}}[config]
    g.setup(**_levers(**gates))
    S._flags(g.monkeypatch, True)
    n_s = int(gates.get(STEADY, 0))
    rng = random.Random(77)
    p = S.Pair()
    for _ in range(3):
        p.both("admit", rng.randint(20, 300))
    outcomes, sizes, gated = [], [], []
    for step in range(300):
        tgt = _D_TARGETS[step % len(_D_TARGETS)]
        n = len(p.wa.reqs)
        if n < tgt and n < S.MAXR:
            p.both("admit", rng.randint(20, 300))
        elif n > tgt:
            p.both("finish", rng.randrange(n))
        elif n > 1 and rng.random() < 0.05:
            p.both("swap", *rng.sample(range(n), 2))
        accepted = [rng.randint(0, S.K) for _ in p.wa.reqs]
        for row in reversed(range(len(p.wa.reqs))):
            if p.wa.reqs[row]["s"] + 2 * (1 + S.K) + S.K > S.WIDTH * S.BS - S.BS:
                p.both("finish", row)
                accepted.pop(row)
        p.both("advance", accepted)
        sizes.append(len(p.wa.reqs))
        out, gd = _gstep(p)
        outcomes.append(out)
        gated.append(gd)
    want_gated = [n < n_s for n in sizes]
    assert gated == want_gated, [i for i, (a, b) in enumerate(zip(gated, want_gated)) if a != b]
    for i, (out, gd) in enumerate(zip(outcomes, gated)):
        if gd:
            assert out is None, (i, out)  # the stock walk: no steady accounting
        else:
            assert out in ("fast", "full"), (i, out)
            if i and gated[i - 1]:
                assert out == "full", i  # the caches were dropped on the gated step
    counts = S._counts()
    assert counts.get("steady.minreqs", 0) == sum(gated), counts
    assert counts.get("steady.mismatch", 0) == 0, counts
    fast = outcomes.count("fast")
    if n_s:
        assert sum(gated) >= 40 and fast >= 60, (sum(gated), fast)
        assert sum(1 for a, b in zip(gated, gated[1:]) if a and not b) >= 10
    else:
        assert not any(gated) and fast >= 100, fast
        assert counts.get("carry.why_minreqs", 0) == sum(n < 4 for n in sizes) > 0, counts
    print(f"GATE_STEADY config={config} fast={fast} full={outcomes.count('full')} gated={sum(gated)}")


def test_reorder_and_count_under_the_gate_matches_the_walks(g):
    g.setup(**_levers(**{STEADY: "4"}))
    rng = random.Random(41)
    ids = [f"q{i}" for i in range(3)]
    a, b = S._ro_runner(ids), S._ro_runner(ids)
    kind = {r: "verify" for r in ids}
    nxt = len(ids)
    fast = gated_steps = after_gate_fast = 0
    prev_gated = False
    for step in range(400):
        l6_ab.begin_step()
        new, finished = [], []
        live = a.input_batch.req_ids
        tgt = _D_TARGETS[step % len(_D_TARGETS)]
        if len(live) < tgt:
            r = f"q{nxt}"
            nxt += 1
            kind[r] = "verify"
            new = [NS(req_id=r)]
            for rr in (a, b):
                rr.input_batch.req_ids.append(r)
        elif len(live) > tgt:
            r = rng.choice(live)
            finished = [r]
            for rr in (a, b):
                rr.input_batch.req_ids.remove(r)
        elif rng.random() < 0.05:
            kind[rng.choice(live)] = rng.choice(["verify", "decode"])
        live = a.input_batch.req_ids
        sched = {r: {"verify": 1 + S.K, "decode": 1}[kind[r]] for r in live}
        so = S._so(sched, [r for r in live if kind[r] == "verify"], new=new, finished=finished)
        # execute_model: the lever-off drop, the gate hook, then the steady path or the walks.
        if not l6_steady.on(a):
            l6_steady.drop(a)
        a._l6_gate_step(so)
        before = S._counts()
        if l6_steady.on(a):
            got = l6_steady.reorder_and_count(a, so)
        else:
            assert len(live) < 4 and "_l6_steady_ro" not in a.__dict__
            got = TPUModelRunner._reorder_batch_for_rpa(a, so) + TPUModelRunner._count_input_chunks(a, so)
        want = TPUModelRunner._reorder_batch_for_rpa(b, so) + TPUModelRunner._count_input_chunks(b, so)
        assert got == want, step
        assert a.input_batch.req_ids == b.input_batch.req_ids, step
        after = S._counts()
        is_fast = after.get("steady.ro_fast", 0) > before.get("steady.ro_fast", 0)
        is_gated = len(live) < 4
        assert not (is_gated and is_fast), step
        if is_gated:
            assert after.get("steady.ro_full", 0) == before.get("steady.ro_full", 0), step
        fast += is_fast
        gated_steps += is_gated
        after_gate_fast += prev_gated and is_fast
        prev_gated = is_gated
    assert after_gate_fast == 0
    assert fast >= 150 and gated_steps >= 60, (fast, gated_steps)
    assert S._counts().get("steady.minreqs", 0) == gated_steps


def test_upstream_reorder_runs_on_a_gated_step(g):
    calls = S._spy_upstream(g.monkeypatch)
    g.setup(**_levers(**{STEADY: "3"}))
    so2 = NS(num_scheduled_tokens={"a": 3, "b": 1})
    r = S._up_runner(S.UpBatch(["a", "b"], [100, 200], [50, 60]))
    l6_ab.begin_step()
    r._l6_gate_step(so2)
    TPUModelRunner._may_reorder_batch(r, so2)
    assert len(calls) == 1
    assert "steady.up_skip" not in S._counts() and "steady.up_run" not in S._counts()
    so3 = NS(num_scheduled_tokens={"a": 3, "b": 1, "c": 3})
    r3 = S._up_runner(S.UpBatch(["a", "b", "c"], [100, 200, 300], [50, 60, 70]))
    l6_ab.begin_step()
    r3._l6_gate_step(so3)
    TPUModelRunner._may_reorder_batch(r3, so3)
    assert len(calls) == 1 and S._counts().get("steady.up_skip") == 1


def test_adapter_tables_take_the_stock_path_on_a_gated_step(g):
    g.setup(**_levers(**{STEADY: "3"}))
    ad = S._adapter()
    r = _bare_runner()
    main, comp = torch.arange(12, dtype=torch.int32), torch.arange(12, dtype=torch.int32) + 50
    l6_ab.begin_step()
    r._l6_gate_step(_so(4))
    first = ad._block_tables(S._snap(main, comp), 4)
    again = ad._block_tables(S._snap(main, comp), 4)
    assert again[0] is first[0] and S._counts().get("steady.tbl_reuse") == 1
    l6_ab.begin_step()
    r._l6_gate_step(_so(2))
    third = ad._block_tables(S._snap(main, comp), 4)
    assert third[0] is not first[0] and torch.equal(third[0], first[0])
    assert S._counts().get("steady.tbl_reuse") == 1


# ---------------------------------------------------------------- E / F. staging and registration


def test_imports_the_staged_tree():
    """Lead rule: the run imports the tree under test, not an installed copy."""
    want = os.environ.get("Q38_L6GATE_EXPECT_ROOT", "")
    assert want, "Q38_L6GATE_EXPECT_ROOT must name the src root under test"
    import vllm_torchtpu

    got = pathlib.Path(vllm_torchtpu.__file__).resolve()
    root = pathlib.Path(want).resolve()
    assert root in got.parents, (str(got), str(root))
    assert torch.__version__ and np.__version__


def test_gate_flags_are_registered_and_compile_ignored():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED

    for lever, name in (("carry", CARRY), ("steady", STEADY)):
        assert l6_ab.GATES[lever] == name
        assert name in envs.environment_variables and name in _TPU_COMPILE_ENV_IGNORED
