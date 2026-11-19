# SPDX-License-Identifier: Apache-2.0
"""q38 L6-HOST lever adcarry (runner/adapt_carry.py): H0-1 ADAPT-CARRY (a)-(c) and
POSTFWD-MERGE. Host-only: runs in a CDK CPU pod.

Every stream test drives the real ``TPUModelRunner._prepare_inputs`` and the real
``Qwen4ExpAdapter.begin_step`` / ``prepare_inputs`` (l6_adc_harness.py) once with
every L6 lever unset (today's two programs, today's uploads) and once with the
lever on, from the same seed, and compares every device input of the forward
byte for byte: the attention-metadata context, positions, input ids, drafts,
indices, spec metadata, every adapter kwarg, both n-gram records and both row
maps. The compiled programs run as the eager functions they wrap.
"""

from __future__ import annotations

import inspect
import os
import pathlib
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

import l6_adc_harness as A
from vllm_torchtpu.core import l6_ab
from vllm_torchtpu.models.vllm.qwen4_exp import inputs as qinputs
from vllm_torchtpu.runner import adapt_carry, decode_carry
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

P = 20000  # A/B window period, ms
L6_KEYS = (
    "VLLM_TPU_L6_ADAPT_CARRY",
    "VLLM_TPU_L6_ADAPT_CARRY_CHECK",
    "VLLM_TPU_L6_DECODE_CARRY",
    "VLLM_TPU_L6_DECODE_CARRY_CHECK",
    "VLLM_TPU_L6_STEADY_STEP",
    "VLLM_TPU_L6_AB_PERIOD_MS",
    "VLLM_TPU_L6_AB_CLASSES",
    "VLLM_TPU_L6_AB_STEPS",
    "VLLM_TPU_L6_CHECK_STEPS",
    "VLLM_TPU_L6_ENVS_CACHE",
    "VLLM_TPU_HOST_AB_PERIOD_MS",
    "VLLM_TPU_D1N_AB_PERIOD_MS",
)
SHAPES = {
    # compact mamba slots (state indices from the runner), 3 requests in an 8-row bucket
    "b3_compact": dict(nreq=3, max_num_reqs=8, paddings=(16, 32, 64), mamba=True, comb=True),
    # pooled cache (state slots = rows), 5 requests
    "b5_pooled": dict(nreq=5, max_num_reqs=8, paddings=(16, 32, 64), mamba=False, comb=True, pooled=True),
    # a full request bucket, no GDN read offsets
    "b8_full": dict(nreq=8, max_num_reqs=8, paddings=(32, 64), mamba=True, comb=False),
    # short contexts (one-token prompts) and n-gram 8: the window rule binds for the first steps
    "b2_short_ng8": dict(nreq=2, max_num_reqs=8, paddings=(16, 32, 64), mamba=True, comb=True, prompt=1, ngram=8),
    # pooled, n-gram 5, a 16-row bucket
    "b4_pooled_ng5": dict(nreq=4, max_num_reqs=16, paddings=(16, 32, 64), mamba=False, comb=True, pooled=True,
                          ngram=5),
}
MODES = {
    "adc": {"VLLM_TPU_L6_ADAPT_CARRY": "1"},
    "adc_carry": {"VLLM_TPU_L6_ADAPT_CARRY": "1", "VLLM_TPU_L6_DECODE_CARRY": "1"},
    "adc_carry_steady": {"VLLM_TPU_L6_ADAPT_CARRY": "1", "VLLM_TPU_L6_DECODE_CARRY": "1",
                         "VLLM_TPU_L6_STEADY_STEP": "1"},
}
STEPS = 48


@pytest.fixture
def env(monkeypatch):
    for k in L6_KEYS + A.ADAPTER_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VLLM_TPU_POSTFWD_FUSED", "1")
    for k, v in A.adapter_env().items():
        monkeypatch.setenv(k, v)
    undo = A.install_eager_programs()
    notes: list[tuple[int, str, str]] = []
    cur = {"step": -1}
    real_note = l6_ab.note

    def note(lever, key, n=1):
        notes.append((cur["step"], lever, key))
        real_note(lever, key, n)

    monkeypatch.setattr(l6_ab, "note", note)

    def lever(extra=None, **kw):
        for k in L6_KEYS:
            monkeypatch.delenv(k, raising=False)
        for k, v in dict(extra or {}, **kw).items():
            monkeypatch.setenv(k, v)
        l6_ab._resolve()

    def track(sim):
        cur["step"] = sim.n

    yield NS(lever=lever, notes=notes, track=track, cur=cur, monkeypatch=monkeypatch)
    undo()
    for k in L6_KEYS:
        monkeypatch.delenv(k, raising=False)
    l6_ab._resolve()


def adc_notes(env, step=None) -> list[str]:
    return [k for s, lv, k in env.notes if lv == "adcarry" and (step is None or s == step)]


def per_step(env, lever="adcarry") -> dict[int, set[str]]:
    out: dict[int, set[str]] = {}
    for s, lv, k in env.notes:
        if lv == lever:
            out.setdefault(s, set()).add(k)
    return out


def both(env, shape, steps, *, seed, events=None, mode="adc_carry", hook=None, force_rej=None, extra=None):
    """(base digests, base sim, lever digests, lever sim)."""
    env.lever()
    base, bsim = A.run(shape, steps, seed=seed, events=events, force_rej=force_rej)
    env.lever(dict(MODES[mode], **(extra or {})))
    env.notes.clear()

    def on_step(sim):
        env.track(sim)
        if hook is not None:
            hook(sim)

    head, hsim = A.run(shape, steps, seed=seed, events=events, on_step=on_step, force_rej=force_rej)
    return base, bsim, head, hsim


def random_events(seed: int, steps: int, shape: dict) -> dict[int, tuple]:
    """A served-shaped stream: steady stretches with admissions, finishes,
    preemption and resume, reorders, partial / non-spec steps, padding changes,
    host skew, remapped sources, discards, flushed results and idle steps."""
    rng = np.random.default_rng(seed)
    n, pre, held = shape.get("nreq", 3), 0, False
    cap = shape.get("max_num_reqs", 8)
    short = shape.get("prompt", 40) < 6
    ev: dict[int, tuple] = {}
    for s in range(3, steps - 2):
        if rng.random() >= 0.3:
            continue
        choices = ["swap", "skew", "remap", "partial", "nonspec", "discard", "flush", "idle_begin", "pad"]
        choices += ["hold2_off"] if held else ["hold2"]
        if n < cap:
            choices += ["join_short"] if short else ["join", "join_short"]
        if n > 1:
            choices += ["leave", "preempt"]
        if pre and n < cap:
            choices += ["resume"]
        e = str(rng.choice(choices))
        ev[s] = (e,)
        if e in ("join", "join_short"):
            n += 1
        elif e == "leave":
            n -= 1
        elif e == "preempt":
            n, pre = n - 1, pre + 1
        elif e == "resume":
            n, pre = n + 1, pre - 1
        elif e == "hold2":
            held = True
        elif e == "hold2_off":
            held = False
    return ev


CASES = [(s, m) for s in sorted(SHAPES) for m in sorted(MODES)]


def tag_precompile(env) -> dict:
    """flag["pc"] is True while a lever's warm-up precompile runs."""
    flag = {"pc": False}
    for mod in (adapt_carry, decode_carry):
        real = mod.precompile

        def wrapped(runner, _real=real):
            flag["pc"] = True
            try:
                return _real(runner)
            finally:
                flag["pc"] = False

        env.monkeypatch.setattr(mod, "precompile", wrapped)
    return flag


# ---------------------------------------------------------------- folded path == today's path


@pytest.mark.parametrize(("name", "mode"), CASES)
def test_folded_path_matches_today_over_random_streams(env, name, mode):
    shape = SHAPES[name]
    seed = 31 * sorted(SHAPES).index(name) + sorted(MODES).index(mode)
    events = random_events(seed, STEPS, shape)
    base, bsim, head, hsim = both(env, shape, STEPS, seed=seed, events=events, mode=mode)
    assert A.first_difference(base, head) is None
    keys = adc_notes(env)
    merged = keys.count("merged_post") + keys.count("merged_carry")
    assert merged >= 10, keys
    assert keys.count("taken") == merged and "dropped" not in keys and "mismatch" not in keys
    assert keys.count("rows_reused") >= 5
    if "carry" in mode:
        assert keys.count("merged_carry") >= 5
        assert keys.count("win_reused") >= 3, sorted(set(keys))
    else:
        assert "merged_carry" not in keys and "win_reused" not in keys
    print(f"STREAM {name} {mode} events={len(events)} merged_post={keys.count('merged_post')} "
          f"merged_carry={keys.count('merged_carry')} win_reused={keys.count('win_reused')} "
          f"rows_reused={keys.count('rows_reused')} uploads base={sum(bsim.uploads)} head={sum(hsim.uploads)}")


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_steady_carried_steps_merge_and_skip_the_window_upload(env, name):
    """Steady decode with lever carry: every step after warm-up runs one merged
    program, reuses the row map and (once the window rule allows) the window,
    and makes one upload fewer than carry alone."""
    shape = SHAPES[name]
    env.lever(MODES["adc_carry"])
    env.notes.clear()
    head, hsim = A.run(shape, 30, seed=5, on_step=env.track)
    env.lever({"VLLM_TPU_L6_DECODE_CARRY": "1"})
    carry, csim = A.run(shape, 30, seed=5)
    assert A.first_difference(carry, head) is None
    d = per_step(env)
    win = [s for s in range(2, 30) if "win_reused" in d.get(s, ())]
    assert all({"merged_carry", "taken", "rows_reused"} <= d[s] for s in range(2, 30)), {s: d.get(s) for s in range(30)}
    if shape.get("prompt", 40) >= 6:
        assert win == list(range(2, 30)), win
    else:  # one-token prompts: the window rule refuses the first carried steps (context <= 9 < 10 at step 2)
        assert win and win[0] > 2 and win == list(range(win[0], 30)), win
    for s in win:
        assert csim.uploads[s] - hsim.uploads[s] == 1, (s, csim.uploads, hsim.uploads)


def test_one_program_instead_of_two_on_a_merged_step(env):
    """The program census: today's step runs the post-forward (or carry) program
    and the adapter's program; a merged step runs one merged program and no
    separate adapter program."""
    calls: list[tuple[int, str]] = []
    cur = env.cur
    real_fused = qinputs.fused_step_inputs
    flag = tag_precompile(env)

    def rec(name, fn):
        def wrapped(*a, **k):
            if not flag["pc"]:
                calls.append((cur["step"], name))
            return fn(*a, **k)
        return wrapped

    def install(sim):
        env.track(sim)
        if sim.n == 0:
            sim.adapter._fused = rec("adapter", real_fused)
            env.monkeypatch.setattr(adapt_carry, "program_postfwd", rec("merged", adapt_carry.merged_postfwd))
            env.monkeypatch.setattr(adapt_carry, "program_carry", rec("merged", adapt_carry.merged_carry))
            env.monkeypatch.setattr(decode_carry, "carry_fused", rec("carry", decode_carry.carry_fused))
            from vllm_torchtpu.runner import tpu_runner

            env.monkeypatch.setattr(tpu_runner, "_postfwd_subst_fused", rec("postfwd", tpu_runner._postfwd_subst_fused))

    env.lever(MODES["adc_carry"])
    A.run(SHAPES["b3_compact"], 12, seed=6, on_step=install, warm_at=4)
    by = {s: sorted(n for t, n in calls if t == s) for s in range(12)}
    assert by[1] == ["adapter", "carry"] and by[3] == ["adapter", "carry"], by  # warm-up: today's two programs
    assert all(by[s] == ["merged"] for s in range(4, 12)), by


def test_every_adapter_program_input_matches(env):
    """Every device input of the adapter's program (starts, seq_lens, state
    indices, scalars, both records, input ids) is the same bits in the folded
    path. The window differs only where (a) reused it: same shape and zero
    padded rows; its live rows are never read (the history matches)."""
    logs: dict[str, list] = {"base": [], "head": []}
    real = qinputs.fused_step_inputs
    arm = {"name": "base"}
    flag = tag_precompile(env)

    def recorder(starts, seq_lens, state, store, scalars, previous, current, ids, **kw):
        if flag["pc"]:
            return real(starts, seq_lens, state, store, scalars, previous, current, ids, **kw)
        logs[arm["name"]].append(dict(
            starts=starts.clone(), seq_lens=seq_lens.clone(), state=state.clone(), store=store.clone(),
            scalars=scalars.clone(), previous=tuple(t.clone() for t in previous),
            current=tuple(t.clone() for t in current), ids=ids.clone(), kw=dict(kw, functional=None)))
        return real(starts, seq_lens, state, store, scalars, previous, current, ids, **kw)

    env.monkeypatch.setattr(qinputs, "fused_step_inputs", recorder)
    env.monkeypatch.setattr(adapt_carry, "_BODIES", {})
    shape = SHAPES["b3_compact"]
    events = {6: ("join",), 9: ("swap",), 12: ("leave",), 15: ("preempt",), 18: ("resume",)}

    def hook(sim):
        sim.adapter._fused = recorder

    env.lever()
    A.run(shape, 24, seed=7, events=events, on_step=hook, prewarm=False)  # the scratch warm-up runners would log
    arm["name"] = "head"
    env.lever(MODES["adc_carry"])
    env.notes.clear()
    A.run(shape, 24, seed=7, events=events, on_step=lambda s: (env.track(s), hook(s)), prewarm=False)
    reused = {s for s, lv, k in env.notes if lv == "adcarry" and k == "win_reused"}
    b, h = logs["base"], logs["head"]
    assert len(b) == len(h) == 24 and len(reused) >= 8
    for s, (x, y) in enumerate(zip(b, h)):
        for k in ("starts", "seq_lens", "state", "scalars", "ids"):
            assert torch.equal(x[k], y[k]), (s, k)
        for k in ("previous", "current"):
            assert all(torch.equal(p, q) for p, q in zip(x[k], y[k])), (s, k)
        assert x["kw"] == y["kw"], s
        n = int(x["scalars"][1])
        assert x["store"].shape == y["store"].shape, s
        if s in reused:
            assert torch.equal(x["store"][n:], y["store"][n:]) and not y["store"][n:].any(), s
        else:
            assert torch.equal(x["store"], y["store"]), s


# ---------------------------------------------------------------- the stream scenarios behind each guard


def test_admitted_row_never_reads_a_reused_window(env):
    """Admissions (and resumed preempted requests) are new rows: the step is
    not carried and its window is uploaded."""
    events = {6: ("join",), 10: ("join_short",), 14: ("preempt",), 17: ("resume",)}
    base, _, head, _ = both(env, SHAPES["b3_compact"], 22, seed=8, events=events)
    assert A.first_difference(base, head) is None
    d = per_step(env)
    for s in (6, 10, 17):
        assert "win_uploaded" in d[s] and "win_reused" not in d[s], (s, d[s])


def test_resumed_row_on_a_merged_step_uploads_the_window(env):
    """A preempted request resumes (a new row, previous row -1) on a step whose
    shape is warm: the merged post-forward program runs, and the window is
    uploaded, not reused (the step is not carried)."""
    base, _, head, _ = both(env, SHAPES["b3_compact"], 12, seed=26, events={6: ("preempt",), 8: ("resume",)})
    assert A.first_difference(base, head) is None
    d = per_step(env)
    assert {"merged_post", "rows_built", "win_uploaded"} <= d[8] and "win_reused" not in d[8], d[8]


def test_idle_adapter_step_refuses_row_map_and_window(env):
    """An adapter begin_step with no chunk (the previous rows are empty): the
    carry still holds, but neither the row map nor the window may be reused."""
    base, _, head, _ = both(env, SHAPES["b3_compact"], 14, seed=9, events={7: ("idle_begin",)})
    assert A.first_difference(base, head) is None
    d = per_step(env)
    assert "merged_carry" in d[7] and "rows_built" in d[7] and "win_uploaded" in d[7], d[7]
    assert {"rows_reused", "win_reused"} <= d[9]


def test_skewed_host_counts_refuse_the_window(env):
    """Host counts skewed by one with zero rejections: the device context moves
    by 1 + K + 1, past the previous record, so the window is read. The carry
    refuses the step; so does (a)."""
    base, _, head, _ = both(env, SHAPES["b3_compact"], 14, seed=10, events={7: ("skew",)},
                            force_rej={5: 0, 6: 0, 7: 0})
    assert A.first_difference(base, head) is None
    d = per_step(env)
    assert "merged_post" in d[7] and "win_uploaded" in d[7], d[7]


def test_reorder_rebuilds_the_scalars(env):
    base, _, head, _ = both(env, SHAPES["b5_pooled"], 14, seed=11, events={7: ("swap",)}, mode="adc")
    assert A.first_difference(base, head) is None
    d = per_step(env)
    assert "rows_built" in d[7] and "rows_reused" in d[8], (d[7], d[8])


def test_short_context_rows_upload_the_window(env):
    """One-token prompts with n-gram 8 (width = K + 7 = 10): while a host context
    is below the width the window is uploaded, on carried steps too (the brief's
    window rule; conservative: on a carried step the previous record covers every
    non-negative n-gram column and negative columns are EOS from either source)."""
    env.lever(MODES["adc_carry"])
    env.notes.clear()
    ctxs = {}

    def hook(sim):
        env.track(sim)
        ctxs[sim.n] = min(sim.computed[rid] for rid in sim.r.input_batch.req_ids[: sim.r.input_batch.num_reqs])

    rej = {0: 3, 1: 3}  # device contexts 1 -> 2 -> 3: n-gram columns below 7 read the window
    head, hsim = A.run(SHAPES["b2_short_ng8"], 16, seed=12, on_step=hook, force_rej=rej)
    env.lever()
    base, _ = A.run(SHAPES["b2_short_ng8"], 16, seed=12, force_rej=rej)
    assert A.first_difference(base, head) is None
    d = per_step(env)
    short_carried = [s for s in range(2, 16) if "merged_carry" in d[s] and "win_uploaded" in d[s]]
    assert short_carried, d
    for s in range(2, 16):
        if "win_reused" in d[s]:
            assert ctxs[s] >= 10, (s, ctxs[s])


def test_two_token_rows_carry_and_still_match(env):
    """Request 0 schedules 2 tokens (one draft) for a stretch: no step of it may
    reuse the window (its record is 2 tokens, not spec_window)."""
    base, _, head, _ = both(env, SHAPES["b3_compact"], 20, seed=13, events={6: ("hold2",), 14: ("hold2_off",)})
    assert A.first_difference(base, head) is None
    d = per_step(env)
    assert all("win_reused" not in d[s] for s in range(6, 15)), {s: d[s] for s in range(6, 15)}


def test_unmerged_steps_keep_the_row_map_reuse(env):
    """Padding-bucket changes after warm-up are cold for the post-forward
    program (today's separate calls run), and the adapter's own program runs
    with (c)."""
    base, _, head, _ = both(env, SHAPES["b3_compact"], 14, seed=14, events={7: ("pad",), 8: ("pad",)}, mode="adc")
    assert A.first_difference(base, head) is None
    d = per_step(env)
    assert "merged_post" not in d.get(8, set()) and "rows_reused" in d[8], d.get(8)


# ---------------------------------------------------------------- lever off / windows


def test_lever_off_runs_no_adcarry_code(env):
    def boom(*a, **k):
        raise AssertionError("adapt_carry ran with the lever off")

    for fn in ("begin", "run_merged", "adapter_prepare", "stage", "commit", "discard"):
        env.monkeypatch.setattr(adapt_carry, fn, boom)
    events = random_events(3, 20, SHAPES["b3_compact"])
    env.lever()
    base, bsim = A.run(SHAPES["b3_compact"], 20, seed=15, events=events)
    # another lever on, adcarry unset
    env.lever({"VLLM_TPU_L6_DECODE_CARRY": "1"})
    _, csim = A.run(SHAPES["b3_compact"], 20, seed=15, events=events)
    # ab:1 in a class-0 window: also off
    env.monkeypatch.setattr(l6_ab, "_now_ns", lambda: int((4_000_000 + 0.5) * P * 1_000_000))
    env.lever({"VLLM_TPU_L6_ADAPT_CARRY": "ab:1", "VLLM_TPU_L6_AB_PERIOD_MS": str(P)})
    off, osim = A.run(SHAPES["b3_compact"], 20, seed=15, events=events)
    assert A.first_difference(base, off) is None and osim.uploads == bsim.uploads


def test_lever_off_window_leaves_no_usable_state(env):
    """ab windows: the lever is off for step 6 only; step 7 must rebuild the
    row map and upload the window, then reuse resumes."""
    clock = {"t": 0}
    env.monkeypatch.setattr(l6_ab, "_now_ns", lambda: clock["t"])
    w = 4_000_001  # an odd window: class 1

    def hook(sim):
        k = w + 1 if sim.n == 6 else w + (0 if sim.n < 6 else 2)
        clock["t"] = int((k + 0.5) * P * 1_000_000)

    base, _, head, _ = both(env, SHAPES["b3_compact"], 12, seed=16, hook=hook,
                            mode="adc_carry", extra={"VLLM_TPU_L6_ADAPT_CARRY": "ab:1",
                                                     "VLLM_TPU_L6_DECODE_CARRY": "ab:1",
                                                     "VLLM_TPU_L6_AB_PERIOD_MS": str(P)})
    assert A.first_difference(base, head) is None
    d = per_step(env)
    assert 6 not in d
    assert "rows_built" in d[7] and "win_uploaded" in d[7], d[7]
    assert all({"rows_reused", "win_reused", "merged_carry"} <= d[s] for s in (9, 10, 11)), d


# ---------------------------------------------------------------- CHECK mode


def test_check_mode_compares_every_engaged_step(env):
    events = {6: ("join",), 9: ("swap",), 12: ("pad",)}
    base, _, head, _ = both(env, SHAPES["b3_compact"], 20, seed=17, events=events,
                            extra={"VLLM_TPU_L6_ADAPT_CARRY_CHECK": "1"})
    assert A.first_difference(base, head) is None
    keys = adc_notes(env)
    merged = keys.count("merged_post") + keys.count("merged_carry")
    assert merged >= 10 and "mismatch" not in keys
    assert keys.count("checked") >= merged


@pytest.mark.parametrize("check", [None, "1"])
def test_check_mode_catches_corrupted_cached_scalars(env, check):
    """The cached identity scalars are corrupted behind the lever's back: without
    CHECK the outputs go wrong; with CHECK the step uses today's values and the
    caches restart."""

    def corrupt(sim):
        if sim.n == 9:
            cache = sim.adapter._adc_idscalars
            for k in list(cache):
                cache[k] = cache[k].clone()
                cache[k][2] = cache[k][2] + 1

    extra = {} if check is None else {"VLLM_TPU_L6_ADAPT_CARRY_CHECK": check}
    base, _, head, _ = both(env, SHAPES["b3_compact"], 14, seed=18, hook=corrupt, extra=extra)
    if check is None:
        assert A.first_difference(base, head) is not None
        return
    assert A.first_difference(base, head) is None
    d = per_step(env)
    assert "mismatch" in d[9], d[9]
    assert "win_uploaded" in d[10] and "mismatch" not in d[10], d[10]  # the caches restarted


def test_check_steps_bound_the_check_turns(env):
    base, _, head, _ = both(env, SHAPES["b3_compact"], 16, seed=19,
                            extra={"VLLM_TPU_L6_ADAPT_CARRY_CHECK": "1", "VLLM_TPU_L6_CHECK_STEPS": "6"})
    assert A.first_difference(base, head) is None
    d = per_step(env)
    assert any("checked" in d[s] for s in range(3, 6))
    assert all("checked" not in d[s] for s in range(7, 16)), d


# ---------------------------------------------------------------- warm-up and cold shapes


def test_serving_never_compiles_a_merged_program_and_precompile_warms_all(env):
    """precompile compiles both merged programs for every warmed post-forward
    shape; after warm-up every merged call is one the warm-up compiled; a shape
    the warm-up did not compile (removed here) and a flipped window mode are
    refused (today's programs run)."""
    calls: list = []
    state: dict = {}
    flag = tag_precompile(env)

    def recorder(kind, fn):
        def wrapped(*args, **statics):
            tens = args[1:] if kind == "carry" else args
            sd = args[0] if kind == "carry" else None
            calls.append((flag["pc"], env.cur["step"], kind, sd,
                          tuple((tuple(a.shape), a.dtype) for a in tens[:11]), tuple(sorted(statics.items()))))
            return fn(*args, **statics)
        return wrapped

    env.monkeypatch.setattr(adapt_carry, "program_postfwd", recorder("postfwd", adapt_carry.merged_postfwd))
    env.monkeypatch.setattr(adapt_carry, "program_carry", recorder("carry", adapt_carry.merged_carry))

    def hook(sim):
        env.track(sim)
        r = sim.r
        if sim.n == 3 and "adc_keys" not in state and hasattr(r, "_adc_warm_keys"):
            state["post_keys"] = set(r._postfwd_warm_keys)
            state["adc_keys"] = set(r._adc_warm_keys)
        if sim.n == 8 and hasattr(r, "_adc_warm_keys"):
            for k in list(r._adc_warm_keys):
                if isinstance(k[0][0], int):  # carry keys start with the step delta
                    r._adc_warm_keys.discard(k)
        if sim.n == 10:
            if hasattr(r, "_adc_warm_keys"):
                r._adc_warm_keys |= state["adc_keys"]
            env.monkeypatch.setenv("VLLM_TPU_QWEN4_NGRAM_WINDOW", "")
        if sim.n == 12:
            env.monkeypatch.setenv("VLLM_TPU_QWEN4_NGRAM_WINDOW", "1")

    env.lever()
    base, _ = A.run(SHAPES["b3_compact"], 14, seed=20, on_step=hook)
    env.lever(MODES["adc_carry"])
    env.notes.clear()
    head, _ = A.run(SHAPES["b3_compact"], 14, seed=20, on_step=hook)
    assert A.first_difference(base, head) is None
    warm_calls = [c for c in calls if c[0]]
    assert len(warm_calls) == 2 * len(state["post_keys"]) >= 2
    assert {c[2] for c in warm_calls} == {"postfwd", "carry"}
    warmed = {c[2:] for c in warm_calls}
    served = [c for c in calls if not c[0]]
    assert served and all(c[1] >= 2 and c[2:] in warmed for c in served), served
    d = per_step(env)
    assert all("why_cold" in d[s] and "merged_carry" not in d[s] for s in (8, 9)), {s: d[s] for s in (8, 9)}
    assert all("why_cold" in d[s] and "taken" not in d[s] for s in (10, 11)), {s: d[s] for s in (10, 11)}
    assert "merged_carry" in d[13]


def test_warmup_steps_run_todays_programs(env):
    env.lever(MODES["adc_carry"])
    env.notes.clear()
    A.run(SHAPES["b3_compact"], 6, seed=21, on_step=env.track, warm_at=4)
    d = per_step(env)
    assert all("why_warmup" in d[s] and "taken" not in d[s] for s in range(0, 4)), d
    assert "merged_carry" in d[4]


def test_flag_set_and_warmup_hook(env):
    env.lever()
    assert not adapt_carry.flag_set()
    env.lever({"VLLM_TPU_L6_ADAPT_CARRY": "0"})
    assert not adapt_carry.flag_set()
    env.lever({"VLLM_TPU_L6_ADAPT_CARRY": "1"})
    assert adapt_carry.flag_set()
    env.lever({"VLLM_TPU_L6_ADAPT_CARRY": "ab:1", "VLLM_TPU_L6_AB_PERIOD_MS": str(P)})
    assert adapt_carry.flag_set()
    src = inspect.getsource(TPUModelRunner.capture_model)
    tail = src[src.index("self._warmup_spec_decode()"):]
    assert tail.index("if _adapt_carry.flag_set():") < tail.index("_adapt_carry.precompile(self)")


# ---------------------------------------------------------------- each guard clause, alone


def test_rows_guard_clauses():
    d = {"a": 0, "b": 1}
    rec = (5, ("a", "b"), d)
    assert adapt_carry.rows_reusable(rec, ("a", "b"), d, 0, True)
    assert not adapt_carry.rows_reusable(rec, ("a", "b"), d, 0, False)  # not a full chunk
    assert not adapt_carry.rows_reusable(rec, ("a", "b"), d, 2, True)  # not the first row
    assert not adapt_carry.rows_reusable(None, ("a", "b"), d, 0, True)  # no record
    assert not adapt_carry.rows_reusable(rec, ("b", "a"), d, 0, True)  # other order
    assert not adapt_carry.rows_reusable(rec, ("a", "b"), dict(d), 0, True)  # an equal but other dict


def test_window_guard_clauses():
    win = ((8, 6), 3, torch.zeros(8, 6, dtype=torch.int64))
    ctx = np.array([5, 9, 30])
    ok = dict(win=win, rows=8, width=5, n=3, ctx=ctx, nsched=np.array([4, 4, 4]), spec_window=4, carried=True,
              rows_ok=True)
    assert adapt_carry.window_reusable(**ok)
    for field, bad in (("carried", False), ("rows_ok", False), ("win", None),
                       ("win", ((16, 6), 3, win[2])), ("win", ((8, 7), 3, win[2])), ("win", ((8, 6), 4, win[2])),
                       ("nsched", None), ("nsched", np.array([4, 2, 4])), ("ctx", np.array([5, 4, 30]))):
        assert not adapt_carry.window_reusable(**dict(ok, **{field: bad})), field


def test_identity_scalars_are_keyed_by_rows_and_count():
    a = NS(_host_keep=())
    a._scalars_upload = lambda host: torch.from_numpy(host.copy())
    s1 = adapt_carry._identity_scalars(a, 3, 8)
    s2 = adapt_carry._identity_scalars(a, 3, 16)
    s3 = adapt_carry._identity_scalars(a, 4, 8)
    assert s1.tolist() == [0, 3, 0, 1, 2] + [-1] * 5
    assert s2.tolist() == [0, 3, 0, 1, 2] + [-1] * 13
    assert s3.tolist() == [0, 4, 0, 1, 2, 3] + [-1] * 4
    assert adapt_carry._identity_scalars(a, 3, 8) is s1


def test_commit_records_only_full_chunks():
    a = type("Ad", (), {})()
    a._current_rows = {}
    r = NS(input_batch=NS(num_reqs=3))
    st = NS(pending={"a": 0, "b": 1, "c": 2}, ids=("a", "b", "c"))
    adapt_carry.commit(a, r, st, 0, 3)
    assert a._adc_rows[1] == ("a", "b", "c") and a._adc_rows[2] is a._current_rows
    assert a._current_rows == {"a": 0, "b": 1, "c": 2}
    a2 = type("Ad", (), {})()
    a2._current_rows = {}
    a2._adc_rows = (1, ("a",), {})
    adapt_carry.commit(a2, r, NS(pending={"a": 0, "b": 1}, ids=("a", "b")), 0, 2)
    assert not hasattr(a2, "_adc_rows")


def _snap(snap, **changes):
    fields = ("token_store", "row_start", "num_requests", "num_tokens", "attention_metadata", "input_ids")
    return NS(**dict({f: getattr(snap, f) for f in fields}, **changes))


def _copy_parked(p):
    q = adapt_carry.Parked()
    for f in adapt_carry.Parked.__slots__:
        setattr(q, f, getattr(p, f))
    return q


def _perturbations():
    def step(p, snap, ctx):
        p.step -= 1

    def row_start(p, snap, ctx):
        return dict(snap=_snap(snap, row_start=1))

    def num_requests(p, snap, ctx):
        p.n += 1

    def ids(p, snap, ctx):
        p.ids = p.ids.clone()

    def num_tokens(p, snap, ctx):
        return dict(snap=_snap(snap, num_tokens=snap.num_tokens + 1))

    def seq_lens(p, snap, ctx):
        p.seq_lens = p.seq_lens.clone()

    def starts(p, snap, ctx):
        p.starts = p.starts.clone()

    def state(p, snap, ctx):
        p.state = p.state.clone()

    def window(p, snap, ctx):
        p.window = not p.window

    return {"step": step, "row_start": row_start, "num_requests": num_requests, "input_ids": ids,
            "num_tokens": num_tokens, "seq_lens": seq_lens, "query_start_loc": starts, "state_indices": state,
            "window": window}


PERTURB = _perturbations()


@pytest.mark.parametrize("name", sorted(PERTURB))
def test_each_parked_field_refuses_alone(env, name):
    """The parked merged result is taken only by its own chunk: each field of
    the match, changed alone on a real parked result, refuses it."""
    real = adapt_carry.parked_matches
    seen = []

    def check(p, adapter, snap, ctx, window):
        control = real(p, adapter, snap, ctx, window)
        if env.cur["step"] == 6 and not seen:
            assert control, "control did not match"
            q = _copy_parked(p)
            got = PERTURB[name](q, snap, ctx) or {}
            seen.append(real(q, adapter, got.get("snap", snap), ctx, window))
        return control

    env.monkeypatch.setattr(adapt_carry, "parked_matches", check)
    env.lever(MODES["adc_carry"])
    A.run(SHAPES["b3_compact"], 8, seed=22, on_step=env.track)
    assert seen == [False], seen


def test_token_store_identity_refuses_in_store_mode(env):
    """Without the n-gram window the merged program reads the step's device
    token store: a parked result from another store is refused."""
    env.monkeypatch.setenv("VLLM_TPU_QWEN4_NGRAM_WINDOW", "")
    real = adapt_carry.parked_matches
    seen = []

    def check(p, adapter, snap, ctx, window):
        control = real(p, adapter, snap, ctx, window)
        if env.cur["step"] == 6 and not seen:
            q = _copy_parked(p)
            q.store = q.store.clone()
            seen.append((control, real(q, adapter, snap, ctx, window)))
        return control

    env.monkeypatch.setattr(adapt_carry, "parked_matches", check)
    base, _, head, _ = both(env, SHAPES["b3_compact"], 10, seed=23)
    assert A.first_difference(base, head) is None
    assert seen == [(True, False)], seen


def test_carry_check_mismatch_discards_the_parked_result(env):
    """A carried device tensor corrupted behind the carry's back with carry
    CHECK on: the carry takes the upload path's values, so the merged
    adapter outputs (computed from the carry's values) must be dropped."""

    def corrupt(sim):
        if sim.n == 9:
            st = sim.r._l6_carry_state
            st.seq_lens = st.seq_lens + 1

    parked_at_entry = {}
    real = adapt_carry.adapter_prepare

    def spy(adapter, snapshot):
        parked_at_entry[env.cur["step"]] = adapter.__dict__.get("_adc_parked") is not None
        return real(adapter, snapshot)

    env.monkeypatch.setattr(adapt_carry, "adapter_prepare", spy)
    base, _, head, _ = both(env, SHAPES["b3_compact"], 13, seed=24, hook=corrupt,
                            extra={"VLLM_TPU_L6_DECODE_CARRY_CHECK": "1"})
    assert A.first_difference(base, head) is None
    d = per_step(env)
    assert "dropped" in d[9] and "taken" not in d[9], d[9]
    # the carry CHECK itself dropped it (the parked input-ids identity would also refuse it)
    assert parked_at_entry[9] is False and parked_at_entry[8] is True and parked_at_entry[10] is True, parked_at_entry


# ---------------------------------------------------------------- engagement line and registration


def test_engagement_line(env):
    lines = []
    clock = {"t": int(4_000_000.5 * 20000 * 1_000_000)}
    env.monkeypatch.setattr(l6_ab, "_now_ns", lambda: clock["t"])
    env.monkeypatch.setattr(l6_ab.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    env.lever(MODES["adc_carry"])
    A.run(SHAPES["b3_compact"], 6, seed=25, on_step=env.track)
    clock["t"] += 20000 * 1_000_000
    l6_ab.begin_step()
    assert len(lines) == 1, lines
    line = lines[0]
    print("ENGAGEMENT", line)
    assert " on=adcarry+carry " in line, line
    counts = dict(kv.split("=") for kv in line.split(" counts=")[1].split(","))
    got = {k: int(v) for k, v in counts.items() if k.startswith("adcarry.")}
    # steps 0-1 warm-up (today's programs; the adapter's own program with (c) from step 1),
    # steps 2-5 merged carry programs with the row map and the window reused
    assert got == {"adcarry.merged_carry": 4, "adcarry.rows_built": 1, "adcarry.rows_reused": 5,
                   "adcarry.separate": 2, "adcarry.taken": 4, "adcarry.why_warmup": 2,
                   "adcarry.win_reused": 4, "adcarry.win_uploaded": 2}, got


def test_lever_is_registered_and_compile_ignored():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED

    assert l6_ab.LEVERS["adcarry"] == "VLLM_TPU_L6_ADAPT_CARRY"
    for name in ("VLLM_TPU_L6_ADAPT_CARRY", "VLLM_TPU_L6_ADAPT_CARRY_CHECK"):
        assert name in envs.environment_variables
        assert name in _TPU_COMPILE_ENV_IGNORED


def test_imports_the_staged_tree():
    """Lead rule: the run imports the tree under test, not an installed copy."""
    want = os.environ.get("Q38_L6ADC_EXPECT_ROOT", "")
    assert want, "Q38_L6ADC_EXPECT_ROOT must name the src root under test"
    import vllm_torchtpu

    got = pathlib.Path(vllm_torchtpu.__file__).resolve()
    root = pathlib.Path(want).resolve()
    assert root in got.parents, (str(got), str(root))
