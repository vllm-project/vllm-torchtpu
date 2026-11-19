# SPDX-License-Identifier: Apache-2.0
"""q38 L6-HOST lever carry (runner/decode_carry.py; port of IX PR14's decode
carry). Host-only: runs in a CDK CPU pod.

Every test drives the real ``TPUModelRunner._prepare_inputs`` (and the real
``_modify_prev_results`` / ``_update_placeholder``) through the harness in
l6_carry_harness.py, once with the lever unset (the upload path) and once with
it on, from the same seed, and compares every output that leaves
``_prepare_inputs`` for the device byte for byte.
"""

from __future__ import annotations

import inspect
import os
import pathlib
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

import l6_carry_harness as H
from vllm_torchtpu.core import l6_ab
from vllm_torchtpu.runner import decode_carry
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

P = 20000  # A/B window period, ms
ENV_KEYS = (
    "VLLM_TPU_L6_DECODE_CARRY",
    "VLLM_TPU_L6_DECODE_CARRY_CHECK",
    "VLLM_TPU_L6_AB_PERIOD_MS",
    "VLLM_TPU_L6_AB_CLASSES",
    "VLLM_TPU_L6_ENVS_CACHE",
    "VLLM_TPU_HOST_AB_PERIOD_MS",
    "VLLM_TPU_D1N_AB_PERIOD_MS",
)
SHAPES = {
    # one request, request padding 1 -> 8, token padding 4 -> 16
    "b1_mamba_comb": dict(nreq=1, max_num_reqs=8, paddings=(16, 32, 64), mamba=True, comb=True),
    # three requests: token padding 12 -> 16, request padding 3 -> 8
    "b3_padded": dict(nreq=3, max_num_reqs=8, paddings=(16, 32, 64), mamba=True, comb=True),
    # pooled cache (no compact mamba indices), 20 -> 32 tokens
    "b5_pooled": dict(nreq=5, max_num_reqs=8, paddings=(16, 32, 64), mamba=False, comb=True,
                      pooled=True),
    # a full bucket (32 tokens, 8 requests), no GDN read offsets
    "b8_full_nocomb": dict(nreq=8, max_num_reqs=8, paddings=(32, 64), mamba=True, comb=False),
    # exactly one token bucket (8), request bucket 16
    "b2_exact_bucket": dict(nreq=2, max_num_reqs=16, paddings=(8, 16, 64), mamba=True, comb=True),
}
STEPS = 40


@pytest.fixture
def env(monkeypatch):
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VLLM_TPU_POSTFWD_FUSED", "1")
    undo = H.install_eager_programs()
    notes: list[tuple[int, str]] = []
    cur = {"step": -1}
    real_note = decode_carry._note

    def note(key):
        notes.append((cur["step"], key))
        real_note(key)

    monkeypatch.setattr(decode_carry, "_note", note)

    def lever(value=None, check=None, **extra):
        for k in ENV_KEYS:
            monkeypatch.delenv(k, raising=False)
        if value is not None:
            monkeypatch.setenv("VLLM_TPU_L6_DECODE_CARRY", value)
        if check is not None:
            monkeypatch.setenv("VLLM_TPU_L6_DECODE_CARRY_CHECK", str(check))
        for k, v in extra.items():
            monkeypatch.setenv(k, v)
        l6_ab._resolve()

    def track(sim):
        cur["step"] = sim.n

    yield NS(lever=lever, notes=notes, track=track, cur=cur, monkeypatch=monkeypatch)
    undo()
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    l6_ab._resolve()


def decisions(notes) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {}
    for step, key in notes:
        out.setdefault(step, set()).add(key)
    return out


def carried_steps(notes) -> set[int]:
    return {s for s, k in notes if k == "carried"}


def both(env, shape, steps, *, seed, events=None, value="1", check=None, carry_hook=None,
         **lever_extra):
    """(base digests, base sim, carry digests, carry sim): the same engine run
    with the lever unset and with it set to `value`."""
    env.lever(None)
    base, bsim = H.run(shape, steps, seed=seed, events=events)
    env.lever(value, check=check, **lever_extra)
    env.notes.clear()

    def on_step(sim):
        env.track(sim)
        if carry_hook is not None:
            carry_hook(sim)

    carry, csim = H.run(shape, steps, seed=seed, events=events, on_step=on_step)
    return base, bsim, carry, csim


# ---------------------------------------------------------------- carry == upload path


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_carry_matches_upload_path_over_random_rejections(env, name):
    shape = SHAPES[name]
    base, bsim, carry, csim = both(env, shape, STEPS, seed=sorted(SHAPES).index(name))
    assert H.first_difference(base, carry) is None
    # Every step after the first carries (steady decode, random rejections).
    assert carried_steps(env.notes) == set(range(1, STEPS))
    rej = csim.r._pre_async_results.spec_decode_num_rejected_tokens
    assert rej is not None
    # Each carried step skips input_ids / positions / seq_lens, plus the
    # unchanged mamba-index and request-distribution uploads.
    saved = 3 + int(shape["mamba"]) + int(shape["comb"])
    assert bsim.uploads[0] == csim.uploads[0]
    assert [b - c for b, c in zip(bsim.uploads[1:], csim.uploads[1:])] == [saved] * (STEPS - 1)
    # A carried steady step makes no upload at all in _prepare_inputs here.
    assert set(csim.uploads[1:]) == {0} if shape["comb"] else True


def test_rejections_vary_in_the_run(env):
    """The comparison above is not vacuous: rejected counts 0..K all occur and
    the corrected device seq_lens differ from the optimistic host values."""
    env.lever("1")
    r = H.make_runner(**{k: v for k, v in SHAPES["b3_padded"].items() if k != "pooled"})
    sim = H.Sim(r, nreq=3, seed=5)
    seen, diffs = set(), 0
    for _ in range(STEPS):
        env.track(sim)
        d = sim.step()
        diffs += d["seq_lens"] != H._b(r.seq_lens_cpu[: r.num_reqs_max_model_len])
        seen.update(r._pre_async_results.spec_decode_num_rejected_tokens[:3].tolist())
    assert seen == set(range(H.K + 1)), seen
    assert diffs >= STEPS // 2, diffs
    assert carried_steps(env.notes) == set(range(1, STEPS))


# ---------------------------------------------------------------- refusals in a run

EVENTS = {
    "request_joins": ("join", "batch", 8),
    "request_leaves": ("leave", "batch", 7),
    "reorder": ("swap", "batch", 7),
    "partial_prefill": ("partial", "batch", 8),
    "non_spec_step": ("nonspec", "batch", 8),
    "padding_change": ("pad", "batch", 8),
    "host_count_skew": ("skew", "seq", 7),
    "results_flushed": ("flush", "pre", 8),
}


@pytest.mark.parametrize("name", sorted(EVENTS))
def test_host_check_refuses_each_change(env, name):
    event, why, resumed = EVENTS[name]
    base, _, carry, _ = both(env, SHAPES["b3_padded"], 14, seed=7, events={6: (event,)})
    assert H.first_difference(base, carry) is None
    d = decisions(env.notes)
    assert all("carried" in d[s] for s in range(1, 6))
    assert "carried" not in d[6] and "uploaded" in d[6], d[6]
    assert "why_" + why in d[6], d[6]
    assert all("carried" in d[s] for s in range(resumed, 14)), {s: d[s] for s in range(6, 14)}


def test_discarded_request_is_refused_by_the_mapping(env):
    base, _, carry, _ = both(env, SHAPES["b3_padded"], 14, seed=8, events={6: ("discard",)})
    assert H.first_difference(base, carry) is None
    d = decisions(env.notes)
    assert "carried" in d[6]
    assert "why_map" in d[7] and "carried" not in d[7], d[7]
    assert all("carried" in d[s] for s in range(9, 14))


def test_new_source_positions_rebuild_the_indices(env):
    """The previous results put requests at other draft-source rows: the carry
    still holds (same host counts) but the subtraction and substitution
    indices are rebuilt, not reused."""
    base, _, carry, _ = both(env, SHAPES["b3_padded"], 14, seed=9, events={6: ("remap",)})
    assert H.first_difference(base, carry) is None
    assert carried_steps(env.notes) == set(range(1, 14))


def test_a_lever_off_window_leaves_no_usable_state(env):
    """ab windows: the lever is off for step 6 only; step 7 must not carry from
    step 5's state."""
    clock = {"t": 0}
    env.monkeypatch.setattr(l6_ab, "_now_ns", lambda: clock["t"])
    w = 4_000_001  # an odd window: class 1

    def hook(sim):
        k = w + 1 if sim.n == 6 else w + (0 if sim.n < 6 else 2)
        clock["t"] = int((k + 0.5) * P * 1_000_000)

    base, _, carry, _ = both(
        env, SHAPES["b3_padded"], 12, seed=10, value="ab:1", carry_hook=hook,
        VLLM_TPU_L6_AB_PERIOD_MS=str(P),
    )
    assert H.first_difference(base, carry) is None
    d = decisions(env.notes)
    assert 6 not in d  # lever off: the carry code did not run
    assert "why_stale" in d[7], d.get(7)
    assert all("carried" in d[s] for s in (1, 2, 3, 4, 5, 8, 9, 10, 11))


# ---------------------------------------------------------------- each host check, alone

def _perturbations():
    def stale(r, st, args):
        st.step -= 1
        return lambda: setattr(st, "step", st.step + 1)

    def rej_step(r, st, args):
        rec = r._l6_carry_rej
        r._l6_carry_rej = (rec[0] - 1, rec[1], rec[2])
        return lambda: setattr(r, "_l6_carry_rej", rec)

    def rej_tag(r, st, args):
        rec = r._l6_carry_rej
        r._l6_carry_rej = (rec[0], rec[1] + 1000, rec[2])
        return lambda: setattr(r, "_l6_carry_rej", rec)

    def rej_missing(r, st, args):
        rec = r._l6_carry_rej
        r._l6_carry_rej = (rec[0], rec[1], {k: v for k, v in rec[2].items() if k != st.req_ids[0]})
        return lambda: setattr(r, "_l6_carry_rej", rec)

    def pre_same(r, st, args):
        pre = r._pre_async_results
        old = pre._l6_carry_tag
        pre._l6_carry_tag = st.src_tag
        return lambda: setattr(pre, "_l6_carry_tag", old)

    def positions(r, st, args):
        r.positions_np[0] += 1
        return lambda: r.positions_np.__setitem__(0, r.positions_np[0] - 1)

    def positions_pad(r, st, args):
        t = st.total
        r.positions_np[t] += 7
        return lambda: r.positions_np.__setitem__(t, r.positions_np[t] - 7)

    def seq_pad(r, st, args):
        n = len(st.req_ids)
        r.seq_lens_np[n] = 7
        return lambda: r.seq_lens_np.__setitem__(n, 0)

    def seq_count(r, st, args):
        st.S[0] += 1
        return lambda: st.S.__setitem__(0, st.S[0] - 1)

    def unmapped(r, st, args):
        m = r._pre_async_results.req_id_to_index_copy
        rid = st.req_ids[0]
        v = m.pop(rid)
        return lambda: m.__setitem__(rid, v)

    def postfwd_off(r, st, args):
        os.environ.pop("VLLM_TPU_POSTFWD_FUSED")
        return lambda: os.environ.__setitem__("VLLM_TPU_POSTFWD_FUSED", "1")

    def pcp(r, st, args):
        r._pcp_enabled = True
        return lambda: setattr(r, "_pcp_enabled", False)

    def mrope(r, st, args):
        r.uses_mrope = True
        return lambda: setattr(r, "uses_mrope", False)

    def second_chunk(r, st, args):
        args[1] = 1
        return lambda: None

    def shape(r, st, args):
        ids = st.input_ids
        st.input_ids = ids[:-1]
        return lambda: setattr(st, "input_ids", ids)

    def nsched(r, st, args):
        b = st.nsched_b
        st.nsched_b = b[::-1] + b"x"
        return lambda: setattr(st, "nsched_b", b)

    def req_order(r, st, args):
        ids = st.req_ids
        st.req_ids = ids[::-1]
        return lambda: setattr(st, "req_ids", ids)

    return {
        "stale": (stale, "stale"),
        "rej_step": (rej_step, "rej"),
        "rej_tag": (rej_tag, "rej"),
        "rej_missing": (rej_missing, "rej"),
        "pre_not_advanced": (pre_same, "pre"),
        "positions": (positions, "pos"),
        "positions_padding": (positions_pad, "pad"),
        "seq_lens_padding": (seq_pad, "pad"),
        "seq_lens_advance": (seq_count, "seq"),
        "unmapped_request": (unmapped, "map"),
        "postfwd_off": (postfwd_off, "config"),
        "pcp": (pcp, "config"),
        "mrope": (mrope, "config"),
        "second_chunk": (second_chunk, "chunk"),
        "shape": (shape, "shape"),
        "scheduled_tokens": (nsched, "batch"),
        "request_order": (req_order, "batch"),
    }


PERTURB = _perturbations()


@pytest.mark.parametrize("name", sorted(PERTURB))
def test_each_host_check_refuses_alone(env, name):
    perturb, why = PERTURB[name]
    real = decode_carry._check
    seen = []

    def check(*args):
        args = list(args)
        control = real(*args)
        if env.cur["step"] == 3 and not seen:
            st = control[0]
            assert st is not None, f"control did not carry: {control[1]}"
            undo = perturb(args[0], st, args)
            try:
                seen.append(real(*args))
            finally:
                undo()
        return control

    env.monkeypatch.setattr(decode_carry, "_check", check)
    env.lever("1")
    H.run(SHAPES["b3_padded"], 5, seed=11, on_step=env.track)
    assert seen, "the perturbed check never ran"
    st, reason = seen[0]
    assert st is None and reason == why, (st, reason)


# ---------------------------------------------------------------- lever off


def test_lever_off_runs_no_carry_code(env):
    def boom(*a, **k):
        raise AssertionError("decode_carry ran with the lever off")

    for fn in ("begin", "_check", "fused_prepare", "record_rejected", "reuse_upload",
               "late_upload", "end", "precompile"):
        env.monkeypatch.setattr(decode_carry, fn, boom)
    events = {4: ("join",), 7: ("swap",), 9: ("partial",), 11: ("nonspec",), 13: ("leave",)}
    env.lever(None)
    _, sim = H.run(SHAPES["b3_padded"], 16, seed=12, events=events)
    # The base path: input_ids, positions, seq_lens, mamba indices and the
    # request distribution, every steady step.
    assert sim.uploads[1:4] == [5, 5, 5], sim.uploads
    # ab:1 in a class-0 window: also off.
    env.monkeypatch.setattr(l6_ab, "_now_ns", lambda: int((4_000_000 + 0.5) * P * 1_000_000))
    env.lever("ab:1", VLLM_TPU_L6_AB_PERIOD_MS=str(P))
    _, sim2 = H.run(SHAPES["b3_padded"], 16, seed=12, events=events)
    assert sim2.uploads == sim.uploads


def test_refused_next_to_another_windowed_ab(env):
    """With D1e's windowed A/B also set, l6_ab refuses every ab: lever, so the
    carry stays off (uniform 1 still works)."""
    env.lever("ab:1", VLLM_TPU_L6_AB_PERIOD_MS=str(P), VLLM_TPU_HOST_AB_PERIOD_MS="20000")
    assert l6_ab.REFUSED
    H.run(SHAPES["b3_padded"], 6, seed=13, on_step=env.track)
    assert env.notes == []


# ---------------------------------------------------------------- CHECK mode


def test_check_mode_compares_every_carried_step(env):
    base, _, carry, _ = both(env, SHAPES["b3_padded"], 20, seed=14, check=1)
    assert H.first_difference(base, carry) is None
    keys = [k for _, k in env.notes]
    assert keys.count("checked") == keys.count("carried") == 19
    assert "mismatch" not in keys


@pytest.mark.parametrize("check", [None, 1])
def test_check_mode_catches_a_corrupted_carry(env, check):
    """A device tensor the carry holds is changed behind its back (as an
    in-place write by a consumer would). Without CHECK the outputs go wrong;
    with CHECK the step takes the upload path's values and the carry restarts."""

    def corrupt(sim):
        if sim.n == 10:
            st = sim.r._l6_carry_state
            st.seq_lens = st.seq_lens + 1

    base, _, carry, _ = both(env, SHAPES["b3_padded"], 14, seed=15, check=check,
                             carry_hook=corrupt)
    d = decisions(env.notes)
    if check is None:
        assert H.first_difference(base, carry) is not None
        return
    assert H.first_difference(base, carry) is None
    assert "mismatch" in d[10] and "carried" not in d[10], d[10]
    assert "why_nostate" in d[11]
    assert all("carried" in d[s] for s in (12, 13))


# ---------------------------------------------------------------- warmup / cold shapes


def test_serving_never_compiles_a_carry_shape_and_precompile_warms_all(env):
    compiled = []
    eager = decode_carry.carry_subtract_substitute_extract

    def recorder(step_delta, *args):
        compiled.append((step_delta, tuple((tuple(a.shape), a.dtype) for a in args)))
        return eager(step_delta, *args)

    env.monkeypatch.setattr(decode_carry, "carry_fused", recorder)
    state = {}

    def hook(sim):
        r = sim.r
        if sim.n == 3:  # warmup over; no carry shape is warm
            r.warmup_done = True
            r._l6_carry_warm_keys = set()
        if sim.n == 6:
            compiled.clear()
            state["n"] = decode_carry.precompile(r)
            state["fused_keys"] = set(r._postfwd_warm_keys)

    base, _, carry, _ = both(env, SHAPES["b3_padded"], 10, seed=16, carry_hook=hook)
    assert H.first_difference(base, carry) is None
    d = decisions(env.notes)
    assert "why_cold" in d[3] and "carried" not in d[3]
    assert state["n"] == len(state["fused_keys"]) >= 1
    pre = [c for c in compiled[: state["n"]]]
    assert {k for _, k in pre} == state["fused_keys"] and {s for s, _ in pre} == {H.STRIDE}
    assert all("carried" in d[s] for s in range(6, 10))


def test_flag_set_and_warmup_hook(env):
    env.lever(None)
    assert not decode_carry.flag_set()
    env.lever("0")
    assert not decode_carry.flag_set()
    env.lever("1")
    assert decode_carry.flag_set()
    env.lever("ab:1", VLLM_TPU_L6_AB_PERIOD_MS=str(P))
    assert decode_carry.flag_set()
    src = inspect.getsource(TPUModelRunner.capture_model)
    tail = src[src.index("self._warmup_spec_decode()"):]
    assert tail.index("if _decode_carry.flag_set():") < tail.index("_decode_carry.precompile(self)")


# ---------------------------------------------------------------- engagement line and registration


def test_engagement_line(env):
    lines = []
    clock = {"t": int(4_000_000.5 * 20000 * 1_000_000)}
    env.monkeypatch.setattr(l6_ab, "_now_ns", lambda: clock["t"])
    env.monkeypatch.setattr(l6_ab.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    env.lever("1")
    H.run(SHAPES["b3_padded"], 6, seed=17, on_step=env.track)
    clock["t"] += 20000 * 1_000_000
    l6_ab.begin_step()
    assert len(lines) == 1, lines
    assert lines[0].endswith(
        "class=uniform steps=6 on=carry counts=carry.carried=5,carry.comb_reused=5,"
        "carry.comb_uploaded=1,carry.mamba_reused=5,carry.mamba_uploaded=1,"
        "carry.uploaded=1,carry.why_nostate=1"
    ), lines[0]


def test_registration():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED

    assert l6_ab.LEVERS["carry"] == "VLLM_TPU_L6_DECODE_CARRY"
    for name in ("VLLM_TPU_L6_DECODE_CARRY", "VLLM_TPU_L6_DECODE_CARRY_CHECK"):
        assert name in envs.environment_variables
        assert name in _TPU_COMPILE_ENV_IGNORED


def test_imports_the_staged_tree():
    """Lead rule: the run imports the tree under test, not an installed copy."""
    want = os.environ.get("Q38_L6CARRY_EXPECT_ROOT", "")
    assert want, "Q38_L6CARRY_EXPECT_ROOT must name the src root under test"
    import vllm_torchtpu

    got = pathlib.Path(vllm_torchtpu.__file__).resolve()
    root = pathlib.Path(want).resolve()
    assert root in got.parents, (str(got), str(root))
    assert torch.__version__ and np.__version__


# ---------------------------------------------------------------- STEADY and CARRY together (STACK-D v4)


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_carry_with_steady_matches_both_off(env, name):
    """v4 runs STEADY and CARRY in overlapping windows: with both on, every output that leaves _prepare_inputs equals
    the all-off run bit for bit, CARRY carries every step after the first, and STEADY counts no mismatch."""
    shape = SHAPES[name]
    env.monkeypatch.delenv("VLLM_TPU_L6_STEADY_STEP", raising=False)
    counts: dict[str, int] = {}
    real = l6_ab.note

    def note(lever, key, n=1):
        counts[f"{lever}.{key}"] = counts.get(f"{lever}.{key}", 0) + n
        real(lever, key, n)

    env.monkeypatch.setattr(l6_ab, "note", note)
    base, bsim, comb, csim = both(env, shape, STEPS, seed=100 + sorted(SHAPES).index(name),
                                  VLLM_TPU_L6_STEADY_STEP="1")
    assert H.first_difference(base, comb) is None
    assert carried_steps(env.notes) == set(range(1, STEPS))
    assert counts.get("steady.mismatch", 0) == 0, counts
    print(f"COMBO {name} steady/carry counts {dict(sorted(counts.items()))}")
