# SPDX-License-Identifier: Apache-2.0
"""q38 L6 levers clean (runner/l6_carry_lean.py, H2 CARRY-LEAN) and pshort
(runner/l6_prep_short.py, H2 PREP-SHORT). Host-only: runs in a CDK CPU pod.

Every stream test drives the real ``TPUModelRunner._prepare_inputs`` (and the
real ``_modify_prev_results`` / ``_update_placeholder``, and STEADY's real
reorder verdict) through l6_pshort_harness.py twice from the same seed, once
with STEADY + CARRY on (the stock path) and once with the lever(s) added, and
compares at every step, byte for byte: every output that leaves
``_prepare_inputs`` for the device (the CARRY harness digests), the spec
metadata (its tensors and its ``draft_token_ids`` at return), every host mirror
(positions, sequence lengths, query starts, the mamba-slot and combined
staging tensors, the slot map and free list, the seeded set, the layout plan;
``input_ids_cpu`` only where noted), the carry state the step leaves, the
compiled-program shapes called, the upload count, and the CARRY decisions
(carried / uploaded / why_*). Streams mix admissions, finishes, preemptions
with resumption, order changes, schedule changes, remapped and discarded
previous results, skewed counts, flushes and DP padding.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import os
import pathlib
import re
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

import l6_carry_harness as H
import l6_pshort_harness as S
from vllm_torchtpu.core import l6_ab
from vllm_torchtpu.runner import decode_carry, l6_carry_lean, l6_prep_short, l6_ptrims, l6_steady

ENV_KEYS = (
    "VLLM_TPU_L6_CARRY_LEAN",
    "VLLM_TPU_L6_CARRY_LEAN_CHECK",
    "VLLM_TPU_L6_PREP_SHORT",
    "VLLM_TPU_L6_PREP_SHORT_CHECK",
    "VLLM_TPU_L6_PREP_TRIMS",
    "VLLM_TPU_L6_PREP_TRIMS_CHECK",
    "VLLM_TPU_L6_DECODE_CARRY",
    "VLLM_TPU_L6_DECODE_CARRY_CHECK",
    "VLLM_TPU_L6_STEADY_STEP",
    "VLLM_TPU_L6_STEADY_STEP_CHECK",
    "VLLM_TPU_L6_ENVS_CACHE",
    "VLLM_TPU_L6_CHECK_STEPS",
    "VLLM_TPU_L6_AB_PERIOD_MS",
    "VLLM_TPU_L6_AB_STEPS",
    "VLLM_TPU_L6_AB_CLASSES",
    "VLLM_TPU_L6_AB_CONFIRM",
    "VLLM_TPU_HOST_AB_PERIOD_MS",
    "VLLM_TPU_D1N_AB_PERIOD_MS",
)
BASE = {"VLLM_TPU_L6_STEADY_STEP": "1", "VLLM_TPU_L6_DECODE_CARRY": "1"}
LEAN = dict(BASE, VLLM_TPU_L6_CARRY_LEAN="1")
SHORT = dict(LEAN, VLLM_TPU_L6_PREP_SHORT="1")
PSA = dict(BASE, VLLM_TPU_L6_PREP_SHORT="1")
PT = {"VLLM_TPU_L6_PREP_TRIMS": "1"}
SHAPES = S.SHAPES
STEPS = 120
STREAMS = [(name, seed) for name in sorted(SHAPES) for seed in range(5)]  # the 25 streams


@pytest.fixture
def env(monkeypatch):
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VLLM_TPU_POSTFWD_FUSED", "1")
    undo = H.install_eager_programs()
    monkeypatch.setattr(l6_ab, "_now_ns", lambda: 10**18)  # one wall-clock window
    monkeypatch.setattr(l6_ab.logger, "info", lambda *a, **k: None)
    errors: list[str] = []
    monkeypatch.setattr(l6_carry_lean.logger, "error", lambda fmt, *a: errors.append(fmt % a))
    monkeypatch.setattr(l6_prep_short.logger, "error", lambda fmt, *a: errors.append(fmt % a))
    lnotes: list = []
    real_note = l6_ab.note

    def note(lever, key, n=1):
        lnotes.append((l6_ab._state.get("seq", 0), lever, key))
        real_note(lever, key, n)

    monkeypatch.setattr(l6_ab, "note", note)

    def lever(arm=None, **extra):
        for k in ENV_KEYS:
            monkeypatch.delenv(k, raising=False)
        for k, v in dict(arm or {}, **extra).items():
            monkeypatch.setenv(k, v)
        l6_ab._resolve()
        for mod in (l6_steady, l6_ptrims, l6_carry_lean, l6_prep_short):
            mod._CHECK[0] = None
        lnotes.clear()

    yield NS(lever=lever, errors=errors, monkeypatch=monkeypatch, lnotes=lnotes)
    undo()
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    l6_ab._resolve()
    for mod in (l6_steady, l6_ptrims, l6_carry_lean, l6_prep_short):
        mod._CHECK[0] = None


def counts():
    return dict(l6_ab._state.get("counts", {}))


def per_step(lnotes, lever, key) -> set[int]:
    """The l6_ab steps on which `lever.key` was noted."""
    return {s for s, lv, k in lnotes if lv == lever and k == key}


def pair(env, base_arm, arm, shape, steps, *, seed, **kw):
    """(base rows, base sim, base counts, rows, sim, counts, lever notes of the second run)."""
    env.lever(base_arm)
    b, bsim = S.run_stream(shape, steps, seed=seed, **kw)
    bc = counts()
    env.lever(arm)
    t, tsim = S.run_stream(shape, steps, seed=seed, **kw)
    return b, bsim, bc, t, tsim, counts(), list(env.lnotes)


def strip(c, *prefixes):
    return {k: v for k, v in c.items() if not k.startswith(prefixes)}


# ---------------------------------------------------------------- the 25 streams


@pytest.mark.parametrize("name,seed", STREAMS)
def test_lean_matches_stock(env, name, seed):
    """CARRY-LEAN: identical carry decisions, outputs, mirrors and carry state."""
    b, _, bc, t, _, c, _ = pair(env, BASE, LEAN, SHAPES[name], STEPS, seed=seed)
    assert S.first_difference(b, t) is None
    assert strip(c, "clean.") == bc  # carry / steady counts identical
    assert c.get("clean.fast", 0) > STEPS // 4, c
    assert c.get("clean.inplace", 0) > 0 and c.get("clean.mismatch", 0) == 0


@pytest.mark.parametrize("name,seed", STREAMS)
def test_short_matches_stock(env, name, seed):
    """PREP-SHORT with CARRY-LEAN: identical everything but the stale input_ids_cpu."""
    b, _, bc, t, tsim, c, _ = pair(env, BASE, SHORT, SHAPES[name], STEPS, seed=seed)
    assert S.first_difference(b, t, skip=S.SKIPPED_MIRRORS) is None
    assert strip(c, "clean.", "pshort.") == bc
    assert c.get("pshort.mismatch", 0) == 0 and c.get("pshort.materialize", 0) == 0
    if SHAPES[name]["comb"]:
        assert c.get("pshort.short", 0) > STEPS // 4, c
    else:
        assert "pshort.short" not in c and c.get("pshort.why_config", 0) > 0


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_pshort_alone_matches_stock(env, name):
    """PREP-SHORT without CARRY-LEAN (CARRY's stock check inside the short path)."""
    b, _, bc, t, _, c, _ = pair(env, BASE, PSA, SHAPES[name], STEPS, seed=7)
    assert S.first_difference(b, t, skip=S.SKIPPED_MIRRORS) is None
    assert strip(c, "clean.", "pshort.") == bc
    assert not any(k.startswith("clean.") and k != "clean.stamp" for k in c)
    assert (c.get("pshort.short", 0) > STEPS // 4) == SHAPES[name]["comb"], c


@pytest.mark.parametrize("name", ["s04_served", "s11_served"])
def test_short_with_ptrims_matches_ptrims(env, name):
    """Next to ptrims (on in both runs): ptrims keeps the non-short steps."""
    b, _, bc, t, _, c, _ = pair(env, dict(BASE, **PT), dict(SHORT, **PT), SHAPES[name], STEPS, seed=3)
    assert S.first_difference(b, t, skip=S.SKIPPED_MIRRORS) is None
    assert c.get("pshort.short", 0) > STEPS // 4
    assert 0 < c.get("ptrims.steps", 0) < bc.get("ptrims.steps", 0) == STEPS


def test_steady_stream_engages_every_step(env):
    """No events: every carried step is a lean accept, and every step from the
    second on takes the short path (the first records the state and the epoch)."""
    shape = SHAPES["s11_served"]
    b, _, bc, t, _, c, ln = pair(env, BASE, LEAN, shape, 60, seed=2, rand=False)
    assert S.first_difference(b, t) is None
    assert bc.get("carry.carried") == 59
    assert c.get("clean.fast") == 59 and c.get("clean.full") == 1, c
    assert c.get("clean.inplace") == 59 and c.get("clean.stamp") == 1, c
    for arm in (SHORT, PSA):
        b, _, bc, t, _, c, ln = pair(env, BASE, arm, shape, 60, seed=2, rand=False)
        assert S.first_difference(b, t, skip=S.SKIPPED_MIRRORS) is None
        assert c.get("pshort.short") == 59, c
        assert per_step(ln, "pshort", "short") == set(range(2, 61))


# ---------------------------------------------------------------- lean: each guard, alone

def _lean_perturbations():
    def stale(r, st, a):
        # The state and its rejection record both from the step before last (a step in
        # between recorded nothing): only the step pairing refuses it (the vector test
        # can pass, e.g. two 1-token advances against a 2-token expectation).
        rec = r._l6_carry_rej
        st.step -= 1
        r._l6_carry_rej = (rec[0] - 1, rec[1], rec[2])

        def undo():
            st.step += 1
            r._l6_carry_rej = rec
        return undo

    def no_epoch(r, st, a):
        ro = st.ro
        st.ro = None
        return lambda: setattr(st, "ro", ro)

    def epoch(r, st, a):
        c = r._l6_steady_ro
        r._l6_steady_ro = copy.copy(c)  # same contents and verdict stamp, another cache object
        return lambda: setattr(r, "_l6_steady_ro", c)

    def verdict(r, st, a):
        c = r._l6_steady_ro
        c.fast_seq -= 1  # the verdict was not this step's (a CHECK step's or a full step's)
        return lambda: setattr(c, "fast_seq", c.fast_seq + 1)

    def remap(r, st, a):
        pre = r._pre_async_results
        m = pre.req_id_to_index_copy
        m2 = dict(m)
        k = st.req_ids[0]
        m2[k] = m2[k] + 1
        pre.req_id_to_index_copy = m2
        return lambda: setattr(pre, "req_id_to_index_copy", m)

    def rej_step(r, st, a):
        rec = r._l6_carry_rej
        r._l6_carry_rej = (rec[0] - 1, rec[1], rec[2])
        return lambda: setattr(r, "_l6_carry_rej", rec)

    def rej_tag(r, st, a):
        rec = r._l6_carry_rej
        r._l6_carry_rej = (rec[0], rec[1] + 1000, rec[2])
        return lambda: setattr(r, "_l6_carry_rej", rec)

    def rej_missing(r, st, a):
        rec = r._l6_carry_rej
        r._l6_carry_rej = (rec[0], rec[1], {k: v for k, v in rec[2].items() if k != st.req_ids[-1]})
        return lambda: setattr(r, "_l6_carry_rej", rec)

    def rej_value(r, st, a):
        rec = r._l6_carry_rej
        d = dict(rec[2])
        d[st.req_ids[-1]] += 1
        r._l6_carry_rej = (rec[0], rec[1], d)
        return lambda: setattr(r, "_l6_carry_rej", rec)

    def seq(r, st, a):
        st.S[0] += 1
        return lambda: st.S.__setitem__(0, st.S[0] - 1)

    def pre_same(r, st, a):
        pre = r._pre_async_results
        had = "_l6_carry_tag" in pre.__dict__
        old = pre.__dict__.get("_l6_carry_tag")
        pre._l6_carry_tag = st.src_tag

        def undo():
            if had:
                pre._l6_carry_tag = old
            else:
                del pre._l6_carry_tag
        return undo

    def arg(i, delta):
        def f(r, st, a):
            a[i] += delta
            return lambda: None
        return f

    def postfwd_off(r, st, a):
        os.environ.pop("VLLM_TPU_POSTFWD_FUSED")
        return lambda: os.environ.__setitem__("VLLM_TPU_POSTFWD_FUSED", "1")

    def flush(r, st, a):
        pre = r._pre_async_results
        r._pre_async_results = None
        return lambda: setattr(r, "_pre_async_results", pre)

    # fast(runner, start_index, num_reqs, total, padded_total, target, layout_plan)
    return {
        "stale_step": stale, "no_epoch": no_epoch, "epoch_replaced": epoch, "verdict_not_fast": verdict,
        "map_changed": remap, "rej_step": rej_step, "rej_tag": rej_tag, "rej_missing": rej_missing,
        "rej_value": rej_value, "seq_advance": seq, "pre_not_advanced": pre_same,
        "chunk_start": arg(1, 1), "total": arg(3, 1), "padded_total": arg(4, 16), "target": arg(5, 1),
        "postfwd_off": postfwd_off, "no_pre": flush,
    }


LEAN_PERTURB = _lean_perturbations()


@pytest.mark.parametrize("name", sorted(LEAN_PERTURB))
def test_lean_refuses_each_change(env, name):
    """At step 3 of a steady stream (heterogeneous rejections) the lean test
    accepts the state; with one change applied it refuses."""
    perturb = LEAN_PERTURB[name]
    real = l6_carry_lean.fast
    seen = []

    def fast(*args):
        if l6_ab.step_seq() == 3 and not seen:
            pargs = list(args)
            undo = perturb(args[0], args[0]._l6_carry_state, pargs)
            try:
                seen.append(real(*pargs))
            finally:
                undo()
            control = real(*args)
            seen.append(control)
            return control
        return real(*args)

    env.monkeypatch.setattr(l6_carry_lean, "fast", fast)
    env.lever(LEAN)
    _, sim = S.run_stream(SHAPES["s04_served"], 5, seed=11, rand=False)
    assert len(seen) == 2, "the perturbed lean test never ran"
    assert seen[0] is None, name
    assert seen[1] is not None, "the unperturbed control did not accept"


def test_lean_control_rejections_vary(env):
    """The perturbation tests' step 3 sees different rejected counts across
    requests (so a gather from the wrong request fails the control)."""
    seen = []
    real = l6_carry_lean.fast

    def fast(*args):
        if l6_ab.step_seq() == 3:
            r, st = args[0], args[0]._l6_carry_state
            seen.append([r._l6_carry_rej[2][q] for q in st.req_ids])
        return real(*args)

    env.monkeypatch.setattr(l6_carry_lean, "fast", fast)
    env.lever(LEAN)
    S.run_stream(SHAPES["s04_served"], 5, seed=11, rand=False)
    assert seen and len(set(seen[0])) > 1, seen


# ---------------------------------------------------------------- short: each guard, end to end


def _around_at(k, fn):
    def around(sim, call, so, start, nd, nw):
        if sim.n == k:
            return fn(sim, call, so, start, nd, nw)
        return call(so, start, nd, nw)
    return around


def _swap_slots(sim, call, so, start, nd, nw):
    r = sim.r
    a, b = r.input_batch.req_ids[0], r.input_batch.req_ids[1]
    m = r._mamba_slot_by_req_id
    m[a], m[b] = m[b], m[a]
    return call(so, start, nd, nw)


def _windowed(sim, call, so, start, nd, nw):
    return call(so, start, nd, nw - 1)


def _chunk(sim, call, so, start, nd, nw):
    sim._adapter = lambda ctx, ids, n: {}  # the adapter digest models one chunk per step
    return call(so, 1, nd, nw)


def _epoch(sim, call, so, start, nd, nw):
    r = sim.r
    r._l6_steady_ro = copy.copy(r._l6_steady_ro)
    return call(so, start, nd, nw)


def _spec_key(sim, call, so, start, nd, nw):
    sim.r.spec_decode_manager._meta_cache_key = None
    return call(so, start, nd, nw)


SHORT_GUARDS = {
    "mamba_slot_changed": (_around_at(9, _swap_slots), None),
    "comb_values_changed": (_around_at(9, _windowed), None),
    "chunk_start": (_around_at(9, _chunk), None),
    "epoch_replaced": (_around_at(9, _epoch), None),
    "spec_key_changed": (_around_at(9, _spec_key), None),
    "bucket_changed": (None, {9: ("pad",)}),
}


@pytest.mark.parametrize("arm", ["short", "alone"])
@pytest.mark.parametrize("name", sorted(SHORT_GUARDS))
def test_short_refuses_each_change(env, name, arm):
    """A change at step 9 of a steady stream, applied to both runs: the short
    path refuses that step (the full path runs) and every step equals stock;
    with CARRY-LEAN on ('short') and without it ('alone', CARRY's stock check
    inside the short path, which knows no epoch or verdict)."""
    around, ev = SHORT_GUARDS[name]
    b, _, _, t, _, c, ln = pair(env, BASE, SHORT if arm == "short" else PSA, SHAPES["s11_served"], 16,
                                seed=4, rand=False, around=around, extra_events=ev)
    assert S.first_difference(b, t, skip=S.SKIPPED_MIRRORS) is None
    short = per_step(ln, "pshort", "short")
    assert 10 not in short, sorted(short)  # step 9 of the sim is l6_ab step 10
    assert {4, 5, 6, 7, 8, 9} <= short, sorted(short)


def test_check_verdict_is_refused_even_when_the_epoch_survives(env):
    """STEADY CHECK steps (VLLM_TPU_L6_STEADY_STEP_CHECK=4), with STEADY patched to
    keep its cache object across a passing check: neither the lean test nor the
    short path accepts a 'check' verdict; both runs equal stock."""
    real = l6_steady.reorder_record

    def keep(runner, so, result, attempt=None):
        c = runner.__dict__.get("_l6_steady_ro")
        real(runner, so, result, attempt)
        new = runner.__dict__.get("_l6_steady_ro")
        if isinstance(attempt, tuple) and attempt and attempt[0] == "check" and c is not None and new is not None:
            c.since_check = new.since_check
            runner._l6_steady_ro = c

    env.monkeypatch.setattr(l6_steady, "reorder_record", keep)
    extra = {"VLLM_TPU_L6_STEADY_STEP_CHECK": "4"}
    for arm in (LEAN, SHORT, PSA):
        b, _, _, t, _, c, ln = pair(env, dict(BASE, **extra), dict(arm, **extra), SHAPES["s11_served"], 30,
                                    seed=5, rand=False)
        assert S.first_difference(b, t, skip=S.SKIPPED_MIRRORS) is None
        checks = per_step(ln, "steady", "check")
        assert len(checks) >= 5, checks
        took = per_step(ln, "clean", "fast") | per_step(ln, "pshort", "short")
        assert took and not (took & checks), (sorted(took), sorted(checks))
        # The epoch object did survive each check (the patch works), so only the verdict refused.
        assert c.get("steady.mismatch", 0) == 0


# ---------------------------------------------------------------- CHECK modes


def test_lean_check_mode_finds_no_mismatch(env):
    b, _, _, t, _, c, _ = pair(env, BASE, dict(LEAN, VLLM_TPU_L6_CARRY_LEAN_CHECK="1"),
                               SHAPES["s11_served"], 60, seed=8)
    assert S.first_difference(b, t) is None
    assert c.get("clean.check") == c.get("clean.fast") > 10
    assert c.get("clean.mismatch", 0) == 0 and not env.errors


def _broken_fast(real):
    """A lean test that accepts any state of the previous step (no other test)."""
    def fast(runner, *a):
        out = real(runner, *a)
        st = runner.__dict__.get("_l6_carry_state")
        if out is None and st is not None and getattr(st, "ro", None) is not None \
                and st.step == l6_ab.step_seq() - 1 and a[0] == 0:
            st.map_same = True
            st.lean_step = l6_ab.step_seq()
            return st
        return out
    return fast


@pytest.mark.parametrize("check", [None, "1"])
def test_lean_check_catches_a_broken_lean_test(env, check):
    """A skewed count at step 6 (the stock check refuses: seq) under a lean test
    that accepts anything: without CHECK the outputs go wrong; with CHECK=1 the
    stock result is used, one ERROR is logged and the lean test turns off."""
    env.monkeypatch.setattr(l6_carry_lean, "fast", _broken_fast(l6_carry_lean.fast))
    arm = dict(LEAN, VLLM_TPU_L6_CARRY_LEAN_CHECK=check) if check else LEAN
    b, _, _, t, tsim, c, _ = pair(env, BASE, arm, SHAPES["s04_served"], 14, seed=9, rand=False,
                                  extra_events={6: ("skew",)})
    if check is None:
        assert S.first_difference(b, t) is not None
        return
    assert S.first_difference(b, t) is None
    assert c.get("clean.mismatch") == 1 and len(env.errors) == 1 and "CHECK mismatch" in env.errors[0]
    assert tsim.r._l6_clean_off is True


def test_short_check_mode_finds_no_mismatch(env):
    """CHECK=3: on a steady stream every third eligible step (29 eligible: steps 2..30)
    runs the full path and compares; on a random stream no compare finds a mismatch."""
    arm = dict(SHORT, VLLM_TPU_L6_PREP_SHORT_CHECK="3")
    b, _, _, t, _, c, _ = pair(env, BASE, arm, SHAPES["s11_served"], 30, seed=8, rand=False)
    assert S.first_difference(b, t, skip=S.SKIPPED_MIRRORS) is None
    assert c.get("pshort.check") == 9 and c.get("pshort.short") == 20, c
    b, _, _, t, _, c, _ = pair(env, BASE, arm, SHAPES["s11_served"], 60, seed=8)
    assert S.first_difference(b, t, skip=S.SKIPPED_MIRRORS) is None
    assert c.get("pshort.check", 0) >= 4 and c.get("pshort.short", 0) >= 8, c
    assert c.get("pshort.mismatch", 0) == 0 and not env.errors


@pytest.mark.parametrize("check", [None, "1"])
def test_short_check_catches_a_broken_prediction(env, check):
    """A short path that reuses a wrong mamba-slot tensor: without CHECK the
    outputs go wrong; with CHECK=1 the full result is used, one ERROR is logged
    and the short path turns off."""
    real = l6_prep_short._plan

    def plan(*a):
        p = real(*a)
        if p is not None and p.mamba_dev is not None:
            p.mamba_dev = p.mamba_dev.flip(0).clone()
        return p

    env.monkeypatch.setattr(l6_prep_short, "_plan", plan)
    arm = dict(SHORT, VLLM_TPU_L6_PREP_SHORT_CHECK=check) if check else SHORT
    b, _, _, t, tsim, c, _ = pair(env, BASE, arm, SHAPES["s04_served"], 14, seed=9, rand=False)
    if check is None:
        assert S.first_difference(b, t, skip=S.SKIPPED_MIRRORS) is not None
        return
    assert S.first_difference(b, t, skip=S.SKIPPED_MIRRORS) is None
    assert c.get("pshort.mismatch") == 1 and len(env.errors) == 1 and "CHECK mismatch" in env.errors[0]
    assert tsim.r._l6_pshort_off is True and "pshort.short" not in c


def test_check_steps_bound_both_checks(env):
    """VLLM_TPU_L6_CHECK_STEPS=8 with both CHECK cadences at 1: no compare after step 8."""
    env.lever(dict(SHORT, VLLM_TPU_L6_CARRY_LEAN_CHECK="1", VLLM_TPU_L6_PREP_SHORT_CHECK="1",
                   VLLM_TPU_L6_CHECK_STEPS="8"))
    S.run_stream(SHAPES["s11_served"], 20, seed=12, rand=False)
    ln = env.lnotes
    checks = per_step(ln, "clean", "check") | per_step(ln, "pshort", "check")
    assert checks and max(checks) <= 8, sorted(checks)
    assert per_step(ln, "pshort", "short") >= set(range(9, 21))


# ---------------------------------------------------------------- off / windows / fallbacks


def test_levers_off_run_no_lever_code(env):
    """Unset or '0' (with STEADY + CARRY on, or no L6 lever at all): no lean test,
    no in-place record, no stamp, no short-path plan runs."""

    def boom(*a, **k):
        raise AssertionError("lever code ran with the lever off")

    for mod, name in ((l6_carry_lean, "fast"), (l6_carry_lean, "record"), (l6_carry_lean, "stamp"),
                      (l6_carry_lean, "fast_tail"), (l6_carry_lean, "cadence"), (l6_prep_short, "_plan")):
        env.monkeypatch.setattr(mod, name, boom)
    for arm in (None, BASE, dict(BASE, VLLM_TPU_L6_CARRY_LEAN="0", VLLM_TPU_L6_PREP_SHORT="0")):
        env.lever(arm)
        _, sim = S.run_stream(SHAPES["s11_served"], 20, seed=1)
        assert not any(k.startswith(("clean.", "pshort.")) for k in counts())
        st = sim.r.__dict__.get("_l6_carry_state")
        assert st is None or getattr(st, "ro", "unset") == "unset"


def test_ab_windows_switch_both_levers(env):
    """ab:1 with step windows of 6: lean accepts and short steps only in class-1
    windows (never on a window's first step), and every step equals stock."""
    win = dict(VLLM_TPU_L6_AB_STEPS="6", VLLM_TPU_L6_AB_CLASSES="2", VLLM_TPU_L6_AB_CONFIRM="2")
    arm = dict(BASE, VLLM_TPU_L6_CARRY_LEAN="ab:1", VLLM_TPU_L6_PREP_SHORT="ab:1", **win)
    b, _, _, t, _, c, ln = pair(env, dict(BASE, **win), arm, SHAPES["s11_served"], 48, seed=6, rand=False)
    assert S.first_difference(b, t, skip=S.SKIPPED_MIRRORS) is None
    took = per_step(ln, "clean", "fast") | per_step(ln, "pshort", "short")
    assert len(took) >= 12, sorted(took)
    assert all((j // 6) % 2 == 1 and j % 6 != 0 for j in took), sorted(took)


def test_materialize_before_carry_check_upload(env):
    """CARRY's CHECK (VLLM_TPU_L6_DECODE_CARRY_CHECK=1) uploads the host inputs on
    every carried step: on a short step StepCtx.materialize writes input_ids_cpu
    first, so even input_ids_cpu equals stock."""
    extra = {"VLLM_TPU_L6_DECODE_CARRY_CHECK": "1"}
    b, _, bc, t, _, c, _ = pair(env, dict(BASE, **extra), dict(SHORT, **extra), SHAPES["s11_served"], 40,
                                seed=10, rand=False)
    assert S.first_difference(b, t) is None  # input_ids_cpu included
    assert c.get("pshort.materialize") == c.get("pshort.short") > 30, c
    assert c.get("carry.checked") == bc.get("carry.checked") and "carry.mismatch" not in c


def test_short_carry_key_follows_its_arguments(env):
    """The short path reuses the previous carried call's shape key only while every
    argument is the same object or has the same shape: after warm-up, a longer token
    source at step 9 (a shape no carry program was warmed for) makes both runs take
    the cold-shape fallback (upload path), and the short path's upload is materialized."""

    def hook(sim):
        r = sim.r
        if sim.n == 6:
            r.warmup_done = True  # serving: never compile a new carry shape
        if sim.n == 8:
            pre = r._pre_async_results
            nt = pre.next_tokens_tpu
            pre.next_tokens_tpu = torch.cat([nt, torch.zeros(H.STRIDE * 4, dtype=nt.dtype)])

    for arm in (SHORT, PSA):
        notes_b, notes_t = [], []
        env.lever(BASE)
        b, _ = S.run_stream(SHAPES["s04_served"], 12, seed=14, rand=False, on_step=hook, notes=notes_b)
        env.lever(arm)
        t, _ = S.run_stream(SHAPES["s04_served"], 12, seed=14, rand=False, on_step=hook, notes=notes_t)
        c, ln = counts(), env.lnotes
        assert notes_b == notes_t
        assert S.first_difference(b, t, skip=S.SKIPPED_MIRRORS) is None
        assert {"why_cold", "uploaded"} <= {k for s, k in notes_b if s == 8}, notes_b
        assert 9 in per_step(ln, "pshort", "short") and c.get("pshort.materialize") == 1, c


def test_kstep_burst_head_and_shifted_rerecord(env):
    """K-step compatibility (reports/kstep_design_v1.md §10 item 1). (a) A burst
    head: every count advanced by an extra 1 + K at step 8: the stock check refuses
    (seq) and so do the lean test and the short path. (b) A shifted re-record of the
    stock fields (S and pos advanced by the same extra 1 + K, as rung 2's re-record):
    the next step carries in both runs and the lean test / short path accept it."""
    extra = H.STRIDE

    def burst(sim):
        if sim.n == 8:
            for rid in list(sim.computed):
                sim.computed[rid] += extra

    def rerecord(sim):
        if sim.n == 12:
            st = sim.r._l6_carry_state
            st.S += extra
            st.pos += extra
            for rid in list(sim.computed):
                sim.computed[rid] += extra

    for hook, k, carried in ((burst, 9, False), (rerecord, 13, True)):
        for arm in (LEAN, SHORT):
            notes_b, notes_t = [], []
            env.lever(BASE)
            b, _ = S.run_stream(SHAPES["s04_served"], 16, seed=13, rand=False, on_step=hook, notes=notes_b)
            env.lever(arm)
            t, _ = S.run_stream(SHAPES["s04_served"], 16, seed=13, rand=False, on_step=hook, notes=notes_t)
            ln = env.lnotes
            assert notes_b == notes_t
            assert S.first_difference(b, t, skip=S.SKIPPED_MIRRORS) is None
            at = {key for s, key in notes_b if s == k - 1}
            took = per_step(ln, "clean", "fast") | per_step(ln, "pshort", "short")
            if carried:
                assert "carried" in at and k in took, (at, sorted(took))
            else:
                assert "why_seq" in at and k not in took, (at, sorted(took))


# ---------------------------------------------------------------- source scans


def _src_root() -> pathlib.Path:
    import vllm_torchtpu

    return pathlib.Path(vllm_torchtpu.__file__).resolve().parents[1]


MIRROR_RE = re.compile(
    r"\b(input_ids_cpu|mamba_state_indices_cpu|_combined_request_distribution_cpu|deferred_draft_token_ids_cpu"
    r"|upload_deferred_draft_token_ids|positions_np|positions_cpu|seq_lens_np|seq_lens_cpu|query_start_loc_np"
    r"|query_start_loc_cpu)\b")
# Every occurrence of the mirrors the lean test reasons about (positions / sequence lengths: their writers)
# and the short path writes or skips (input_ids_cpu and the deferred host drafts: their readers), per file.
MIRROR_PIN = {
    "kernels/experimental/pcp_streaming_rpa/pcp_layout.py:input_ids_cpu": 4,
    "kernels/experimental/pcp_streaming_rpa/pcp_layout.py:positions_cpu": 4,
    "layers/core/attention_metadata.py:query_start_loc_cpu": 2,
    "layers/core/attention_metadata.py:seq_lens_cpu": 2,
    "layers/core/pcp_sequence_layout.py:input_ids_cpu": 6,
    "layers/core/pcp_sequence_layout.py:positions_cpu": 5,
    "layers/core/pcp_sequence_layout.py:positions_np": 2,
    "layers/core/pcp_sequence_layout.py:seq_lens_np": 1,
    "megakernel_runtime/driver.py:_combined_request_distribution_cpu": 1,
    "megakernel_runtime/driver.py:query_start_loc_cpu": 2,
    "megakernel_runtime/driver.py:query_start_loc_np": 1,
    "models/vllm/qwen4_exp/inputs.py:input_ids_cpu": 2,
    "models/vllm/qwen4_exp/inputs.py:positions_cpu": 2,
    "models/vllm/qwen4_exp/inputs.py:query_start_loc_cpu": 2,
    "models/vllm/qwen4_exp/inputs.py:seq_lens_cpu": 2,
    "models/vllm/qwen4_exp/pooled_cache.py:_combined_request_distribution_cpu": 1,
    "models/vllm/qwen4_exp/pooled_cache.py:seq_lens_cpu": 2,
    # q38 L6-V6 compose: ADC's precompile reads only these two mirrors' dtypes at warm-up (adapt_carry.precompile).
    "runner/adapt_carry.py:mamba_state_indices_cpu": 1,
    "runner/adapt_carry.py:query_start_loc_cpu": 1,
    "runner/decode_carry.py:input_ids_cpu": 1,
    "runner/decode_carry.py:positions_cpu": 1,
    "runner/decode_carry.py:positions_np": 4,
    "runner/decode_carry.py:query_start_loc_cpu": 1,
    "runner/decode_carry.py:seq_lens_cpu": 1,
    "runner/decode_carry.py:seq_lens_np": 4,
    "runner/l6_carry_lean.py:positions_np": 2,
    "runner/l6_carry_lean.py:seq_lens_np": 5,
    "runner/l6_prep_short.py:_combined_request_distribution_cpu": 1,
    "runner/l6_prep_short.py:deferred_draft_token_ids_cpu": 2,
    "runner/l6_prep_short.py:input_ids_cpu": 3,
    "runner/l6_prep_short.py:mamba_state_indices_cpu": 1,
    "runner/l6_prep_short.py:positions_np": 7,
    "runner/l6_prep_short.py:query_start_loc_np": 7,
    "runner/l6_prep_short.py:seq_lens_np": 6,
    "runner/l6_ptrims.py:input_ids_cpu": 4,
    "runner/l6_ptrims.py:positions_np": 9,
    "runner/speculative_decoding_manager.py:deferred_draft_token_ids_cpu": 7,
    "runner/speculative_decoding_manager.py:input_ids_cpu": 3,
    "runner/speculative_decoding_manager.py:upload_deferred_draft_token_ids": 1,
    "runner/tpu_runner.py:_combined_request_distribution_cpu": 3,
    "runner/tpu_runner.py:input_ids_cpu": 9,
    "runner/tpu_runner.py:mamba_state_indices_cpu": 2,
    "runner/tpu_runner.py:positions_cpu": 4,
    "runner/tpu_runner.py:positions_np": 10,
    "runner/tpu_runner.py:query_start_loc_cpu": 5,
    "runner/tpu_runner.py:query_start_loc_np": 8,
    "runner/tpu_runner.py:seq_lens_cpu": 12,
    "runner/tpu_runner.py:seq_lens_np": 7,
    "runner/tpu_runner.py:upload_deferred_draft_token_ids": 1,
    "spec_decode/eagle3.py:query_start_loc_np": 8,
    "spec_decode/utils.py:query_start_loc_np": 1,
}


def test_skipped_mirrors_have_no_new_reader():
    """A new reader or writer of a mirror the two levers reason about (anywhere in
    src) fails here: review runner/l6_carry_lean.py (writers of positions_np /
    seq_lens_np between two steps) and runner/l6_prep_short.py (readers of
    input_ids_cpu and of the deferred host drafts on a short step), then re-pin.
    The deferred host drafts' only upload runs for a chunk with nothing substituted."""
    root = _src_root() / "vllm_torchtpu"
    got = {}
    for f in sorted(root.rglob("*.py")):
        rel = f.relative_to(root).as_posix()
        for m in MIRROR_RE.finditer(f.read_text()):
            k = f"{rel}:{m.group(1)}"
            got[k] = got.get(k, 0) + 1
    assert got == MIRROR_PIN, sorted(set(got.items()) ^ set(MIRROR_PIN.items()))
    src = (root / "runner" / "tpu_runner.py").read_text()
    i = src.index("spec_decode_metadata.upload_deferred_draft_token_ids(self.device)")
    head = src[:i]
    j = head.rindex("elif (")
    assert "and spec_decode_metadata.draft_token_ids is None" in head[j:]
    k = head.rindex("and len(cur_input_indices) > 0")
    assert k < j and head.rindex("if postfwd_out is not None:\n                pass") < k


def _prepare_inputs_source() -> str:
    src = (_src_root() / "vllm_torchtpu" / "runner" / "tpu_runner.py").read_text()
    lines = src.splitlines(keepends=True)
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef) and node.name == "TPUModelRunner":
            for b in node.body:
                if isinstance(b, ast.FunctionDef) and b.name == "_prepare_inputs":
                    return "".join(lines[b.lineno - 1 : b.end_lineno])
    raise AssertionError("_prepare_inputs not found")


# q38 L6-V6 compose: re-pinned on q38-l6-v6 after porting the composed full path's edits of _prepare_inputs
# (L6-INSTR's sub-marks; ADC's merge context before _postfwd_fused_prepare; L6-H2's fskip marks: the carried
# mark, the region's upload marks issue nothing on a short-eligible step) to runner/l6_prep_short.py.
# q38-l6-pshort 033b1feb5 had body d5d9fc6b15b927f9, tail 8fbeae0cbb0daff7; v6 before L6-H2 847f6d366b9d0f3f.
PREP_BODY_SHA = "fe450a23176c7dcf"
PREP_TAIL_SHA = "779087a43813a144"


def test_full_path_text_is_pinned():
    """The short path re-implements the full path's carried-step result and runs a
    copy of its tail (from the prep_host_and_uploads mark to the return). Any edit
    of _prepare_inputs fails here: port it to runner/l6_prep_short.py (or refuse the
    short path for it), then re-pin."""
    body = _prepare_inputs_source()
    tail = body[body.index('host_phases.mark("prep_host_and_uploads")'):]
    assert hashlib.sha256(tail.encode()).hexdigest()[:16] == PREP_TAIL_SHA
    assert hashlib.sha256(body.encode()).hexdigest()[:16] == PREP_BODY_SHA
    # The short path's hook is the only new statement: guarded, right after the fence.
    assert body.count("_l6_pshort.run(") == 1
    assert body.index('host_phases.sub("pi_fence")') < body.index("_l6_pshort.run(") < body.index(
        "_l6_ptrims.begin(self)")


# ---------------------------------------------------------------- registration / staging


def test_levers_are_registered_and_compile_ignored():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import tpu_compiler

    assert l6_ab.LEVERS["clean"] == "VLLM_TPU_L6_CARRY_LEAN"
    assert l6_ab.LEVERS["pshort"] == "VLLM_TPU_L6_PREP_SHORT"
    names = ("VLLM_TPU_L6_CARRY_LEAN", "VLLM_TPU_L6_CARRY_LEAN_CHECK", "VLLM_TPU_L6_PREP_SHORT",
             "VLLM_TPU_L6_PREP_SHORT_CHECK")
    for name in names:
        assert name in envs.environment_variables
        assert name in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    reg = pathlib.Path(__file__).resolve().parents[1] / "compilation" / "test_tpu_compile_key_registration.py"
    text = reg.read_text()
    for row in ('("VLLM_TPU_L6_CARRY_LEAN", "1", "")', '("VLLM_TPU_L6_CARRY_LEAN_CHECK", "200", 0)',
                '("VLLM_TPU_L6_PREP_SHORT", "1", "")', '("VLLM_TPU_L6_PREP_SHORT_CHECK", "200", 0)'):
        assert row in text


def test_import_resolves_to_the_tree_under_test():
    """Guard against a staging that silently imports the pod's installed package."""
    import vllm_torchtpu

    want = os.environ.get("Q38_L6_EXPECT_TREE")
    if not want:
        pytest.skip("Q38_L6_EXPECT_TREE not set (the CPU harness sets it)")
    got = os.path.realpath(vllm_torchtpu.__file__)
    assert got.startswith(os.path.realpath(want) + os.sep), (got, want)
