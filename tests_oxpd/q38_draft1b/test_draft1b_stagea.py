# SPDX-License-Identifier: Apache-2.0
"""q38 L4 DRAFT1-B stage A (VLLM_TPU_DRAFT_ASM_CARRY_LOOP, H1 card DH-B) through the real
``Eagle3Proposer.propose`` and the real async spec substitution (tests/q38_draft1b/draft1b_harness.py), CPU.

A. Bitwise over 8 serving steps: stage A == DRAFT-ASM == the base path for the drafts, step 1's seq_lens /
   positions / first token / hidden carry at the loop call, the source, the rejected count and the index
   dicts; rows 4 / 8 / 12 / 16 / 32 / 48 / 64 (p < capacity: fresh carry; p == capacity at 64: plain
   carry), K = 3, fallback rows on and off; requests with fewer drafts.
B. Census per served step: stage A runs 3 compiled draft programs (seed, step 0, carry + loop) where
   DRAFT-ASM runs 4; 0 eager ops, 0 deferred loop inputs, no spec tail.
C. Fallbacks: a never-warmed key, p above the capacity, several chunks, a prefill in the chunk.
D. SAB dual-path: the warm-up runs the base path for results and the configured folded variants beside
   it; DRAFT-ASM's own carry + loop are warmed only when the stage window can be off; a window switching
   per step among stage A / DRAFT-ASM / base never compiles and stays bitwise; the superset seed's
   precompile covers every steady decode call site.
E. CHECK compares stage A (and DRAFT-ASM when kept warm) on its steps and catches perturbed outputs.
F. Markers. G. eagle3 builds DraftOne only with a stage flag; a stage window hard off is DRAFT-ASM.
H. The VFY-ONE value 2 hooks name the superset seed.
"""

import sys
from collections import Counter
from types import ModuleType, SimpleNamespace

import pytest
import torch

import draft1b_harness as h
from vllm_torchtpu.spec_decode import draft_asm, draft_one

ROWS = [4, 8, 12, 16, 32, 48, 64]
DRAFT_PROGRAMS = ("seed", "seed_asm", "seed_cl", "draft_fwd0", "carry", "carry_fresh", "carry_asm",
                  "carry_fresh_asm", "loop", "loop_asm", "carry_loop", "one")


def _draft_calls(calls):
    return sum(v for k, v in calls.items() if k in DRAFT_PROGRAMS)


def _variant(monkeypatch, name, cap=64, **kw):
    """A fresh proposer of variant `name`: base (fold off), dha (DRAFT-ASM), cl (+ stage A)."""
    h.install(monkeypatch, fold=name != "base", cl=name == "cl", **kw)
    return h.make_proposer(fold=name != "base", cap=cap)


def _serve(monkeypatch, name, seeds, R, warm_each=False, cap=64, step_kw=None, **kw):
    """Warm-up once (or before every step), then one serving step per seed; the StepOuts."""
    step_kw = step_kw or {}
    p = _variant(monkeypatch, name, cap=cap, **kw)
    if not warm_each:
        h.run_step(p, h.make_step(seeds[0] - 1, R, cap=cap, **step_kw), serving=False)
    outs = []
    for s in seeds:
        if warm_each:
            h.install(monkeypatch, fold=name != "base", cl=name == "cl", **kw)
            h.run_step(p, h.make_step(s, R, cap=cap, **step_kw), serving=False)
        outs.append(h.run_step(p, h.make_step(s, R, cap=cap, **step_kw), serving=True))
    return outs, p


# ---------------------------------------------------------------- A. bitwise over 8 steps


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("R", ROWS)
def test_stage_a_is_bitwise_draft_asm_and_base_over_8_steps(monkeypatch, R, fallback):
    seeds = [11000 + 100 * R + 10 * fallback + i for i in range(8)]
    kw = {"step_kw": {"fallback": fallback}}
    base, _ = _serve(monkeypatch, "base", seeds, R, **kw)
    dha, _ = _serve(monkeypatch, "dha", seeds, R, **kw)
    cl, cp = _serve(monkeypatch, "cl", seeds, R, **kw)
    for i, (b, d, c) in enumerate(zip(base, dha, cl)):
        assert not b.folded and d.folded and c.folded, (i, b.folded, d.folded, c.folded, cp._draft_asm.counts())
        assert c.calls["carry_loop"] == 1 and c.calls["loop_asm"] == 0, c.calls
        h.assert_same_outputs(c, b, f"stage A vs base R={R} step {i}")
        h.assert_same_outputs(d, b, f"DRAFT-ASM vs base R={R} step {i}")
    assert cp._draft_asm.counts()["folded_cl"] == 8
    carry = "plain" if R == 64 else "fresh"
    print(f"\nDRAFT1B_BITWISE stageA R={R} fallback={int(fallback)} steps=8 carry={carry} drafts "
          f"{tuple(cl[0].drafts.shape)} seq_lens/positions/token0/hidden at the loop call, source[{cl[0].src.numel()}], "
          f"rejected[{cl[0].rejected.numel()}], index dicts equal to DRAFT-ASM and base on every step")


@pytest.mark.parametrize("R", [12, 32])
def test_stage_a_bitwise_with_fewer_drafts(monkeypatch, R):
    seeds = [12000 + 100 * R + i for i in range(8)]
    kw = {"warm_each": True, "step_kw": {"short": True}}
    base, _ = _serve(monkeypatch, "base", seeds, R, **kw)
    cl, cp = _serve(monkeypatch, "cl", seeds, R, **kw)
    for i, (b, c) in enumerate(zip(base, cl)):
        assert c.folded, (i, cp._draft_asm.counts())
        h.assert_same_outputs(c, b, f"short R={R} step {i}")
    print(f"\nDRAFT1B_BITWISE stageA_short R={R} steps=8 fewer-draft requests equal to base")


def test_carry_rows_of_the_folded_step_equal_both_eagle3_carries(monkeypatch):
    """The prologue's hidden carry (recorded by the region by plain slicing) at p < capacity (fresh carry)
    and p == capacity (plain carry) equals the base loop's hidden input; covered per step in A, here with
    an explicit carry-path label from the base path's own call census."""
    for R, want in ((20, "carry_fresh"), (64, "carry")):
        (b,), _ = _serve(monkeypatch, "base", [12900 + R], R)
        (c,), _ = _serve(monkeypatch, "cl", [12900 + R], R)
        assert b.calls[want] == 1, b.calls
        h.same(c.loop_entry["hidden"], b.loop_entry["hidden"], f"hidden carry R={R} ({want})")
        h.same(c.loop_entry["input_ids"], b.loop_entry["input_ids"], f"token0 R={R} ({want})")


# ---------------------------------------------------------------- B. census


@pytest.mark.parametrize("R", [8, 32, 48, 64])
def test_census_three_draft_programs(monkeypatch, R):
    (d,), _ = _serve(monkeypatch, "dha", [13000 + R], R)
    (c,), _ = _serve(monkeypatch, "cl", [13000 + R], R)
    assert c.eager == [] and c.deferred == [], (c.eager, c.deferred)
    assert c.calls["spec_tail"] == 0 and c.calls["seed_cl"] == 1 and c.calls["draft_fwd0"] == 1
    assert c.calls["carry_loop"] == 1
    for k in ("seed", "seed_asm", "carry", "carry_fresh", "carry_asm", "carry_fresh_asm", "loop", "loop_asm"):
        assert c.calls[k] == 0, (k, c.calls)
    nd, nc = _draft_calls(d.calls), _draft_calls(c.calls)
    assert nd == 4 and nc == 3, (d.calls, c.calls)
    assert sum(c.calls.values()) == sum(d.calls.values()) - 1
    print(f"\nDRAFT1B_CENSUS R={R} DRAFT-ASM draft_programs={nd} ({dict(sorted(d.calls.items()))}) "
          f"stageA draft_programs={nc} ({dict(sorted(c.calls.items()))}) eager=0 deferred=0 spec_tail=0")


# ---------------------------------------------------------------- C. fallbacks


def test_never_warmed_key_takes_the_base_path_without_a_folded_call(monkeypatch):
    (b,), _ = _serve(monkeypatch, "base", [14001], 12)
    p = _variant(monkeypatch, "cl")
    c = h.run_step(p, h.make_step(14001, 12), serving=True)  # no warm-up
    assert not c.folded and c.calls["seed_cl"] == 0 and c.calls["carry_loop"] == 0
    assert p._draft_asm.reasons["cold_seed"] == 1
    h.assert_same_outputs(c, b, "cold")


def test_cold_carry_loop_key_falls_to_draft_asm_when_kept_warm(monkeypatch):
    """Stage A windowed (DRAFT-ASM kept warm): a step whose carry-loop key was never warmed (the key set
    cleared after warm-up) folds through DRAFT-ASM's carry + loop, bitwise."""
    (b,), _ = _serve(monkeypatch, "base", [14101], 20)
    p = _variant(monkeypatch, "cl", cl_ab="1")
    p._draft_asm.dha_needed = True  # as with a windowed stage
    h.run_step(p, h.make_step(14100, 20), serving=False)
    p._draft_asm.warm["carry_loop"].clear()
    p._draft_asm.cl_agreed = True
    c = h.run_step(p, h.make_step(14101, 20), serving=True)
    assert c.folded and c.calls["carry_loop"] == 0 and c.calls["loop_asm"] == 1, c.calls
    assert p._draft_asm.reasons["cold_carry_loop"] >= 1
    h.assert_same_outputs(c, b, "cold carry-loop -> DRAFT-ASM")


@pytest.mark.parametrize("dha", [False, True])
def test_loop_bucket_above_capacity_keeps_draft_asm_or_base_with_no_eager_op(monkeypatch, dha):
    """Capacity 40, 40 requests: the loop bucket is 48 > 40, so stage A (agreed on) cannot run. DRAFT-ASM
    not warmed: the base path. DRAFT-ASM kept warm: DRAFT-ASM's fold, with DRAFT-ASM's own eager ops at this
    shape and no more (eagle3's two [cap] -> [p] pads of step 0's outputs, which DRAFT-ASM also runs)."""
    R, cap = 40, 40
    (b,), _ = _serve(monkeypatch, "base", [14201], R, cap=cap)
    (d,), _ = _serve(monkeypatch, "dha", [14201], R, cap=cap)
    p = _variant(monkeypatch, "cl", cap=cap)
    p._draft_asm.dha_needed = dha
    h.run_step(p, h.make_step(14200, R, cap=cap), serving=False)
    c = h.run_step(p, h.make_step(14201, R, cap=cap), serving=True)
    assert p._draft_asm.cl_agreed and c.calls["carry_loop"] == 0, (p._draft_asm.counts(), c.calls)
    h.assert_same_outputs(c, b, f"p > capacity dha={dha}")
    if dha:
        assert d.folded and c.folded and c.calls["loop_asm"] == 1, (c.calls, c.eager)
        assert Counter(c.eager) == Counter(d.eager), (c.eager, d.eager)
        print(f"\nDRAFT1B_CENSUS R={R} cap={cap} p>capacity DRAFT-ASM eager={dict(Counter(d.eager))} "
              f"DraftOne(stage A ineligible) eager={dict(Counter(c.eager))}")
    else:
        assert not c.folded


def test_two_chunks_and_a_prefill_keep_the_base_path(monkeypatch):
    for make, reason in ((lambda: [h.make_step(14301, 6), h.make_step(14302, 5)], "chunks"),
                         (lambda: h.make_step(14303, 9, mixed=True), "mixed")):
        h.install(monkeypatch, fold=False)
        bp = h.make_proposer(fold=False)
        h.run_step(bp, make(), serving=False)
        b = h.run_step(bp, make(), serving=True)
        p = _variant(monkeypatch, "cl")
        h.run_step(p, make(), serving=False)
        c = h.run_step(p, make(), serving=True)
        assert not c.folded and c.calls["seed_cl"] == 0 and c.calls["carry_loop"] == 0
        h.same(c.drafts, b.drafts, reason)
        h.same(c.src, b.src, reason)
        assert p._draft_asm.reasons[reason] == 1


# ---------------------------------------------------------------- D. SAB dual-path


def test_warmup_runs_base_and_warms_only_the_configured_variants(monkeypatch):
    p = _variant(monkeypatch, "cl")
    w = h.run_step(p, h.make_step(15001, 10), serving=False)
    assert not w.folded
    assert w.calls["seed"] == 1 and w.calls["loop"] == 2 and w.calls["spec_tail"] == 1
    assert w.calls["seed_cl"] == 1 and w.calls["carry_loop"] == 2  # first call: trace, then compiled
    assert w.calls["carry_fresh_asm"] == 0 and w.calls["loop_asm"] == 0 and w.calls["seed_asm"] == 0
    assert {k: len(v) for k, v in p._draft_asm.warm.items()} == {
        "seed": 1, "carry": 0, "loop": 0, "carry_loop": 1, "one": 0}
    h.install(monkeypatch, fold=False)
    bp = h.make_proposer(fold=False)
    bw = h.run_step(bp, h.make_step(15001, 10), serving=False)
    h.assert_same_outputs(w, bw, "warm-up step")


class _FakeL6:
    def __init__(self, levers, steps=1, on=None):
        self.LEVERS = {k: "x" for k in levers}
        self.STEPS = steps
        self.on_fn = on or (lambda lever: True)
        self.notes = Counter()

    def install(self, monkeypatch):
        import vllm_torchtpu.core as core

        mod = ModuleType("vllm_torchtpu.core.l6_ab")
        mod.LEVERS, mod.STEPS = self.LEVERS, self.STEPS
        mod.on = lambda lever: self.on_fn(lever)
        mod.note = lambda lever, key, n=1: self.notes.update({(lever, key): n})
        monkeypatch.setitem(sys.modules, "vllm_torchtpu.core.l6_ab", mod)
        monkeypatch.setattr(core, "l6_ab", mod, raising=False)
        monkeypatch.setattr(draft_asm, "_L6_AB", [])


def test_windowed_stage_keeps_draft_asm_warm(monkeypatch):
    _FakeL6(["draftasmcl"]).install(monkeypatch)
    p = _variant(monkeypatch, "cl", cl_ab="ab:1")
    _FakeL6(["draftasmcl"]).install(monkeypatch)
    assert p._draft_asm.dha_needed and p._draft_asm.cl_conf
    w = h.run_step(p, h.make_step(15101, 10), serving=False)
    assert w.calls["carry_loop"] == 2 and w.calls["carry_fresh_asm"] == 1 and w.calls["loop_asm"] == 2
    assert {k: len(v) for k, v in p._draft_asm.warm.items()} == {
        "seed": 1, "carry": 1, "loop": 1, "carry_loop": 1, "one": 0}


def test_window_switching_per_step_never_compiles_and_stays_bitwise(monkeypatch):
    """Boundary every call (l6_ab step window 1): steps cycle stage A / DRAFT-ASM (stage window off) /
    base (fold window off) over 1..64 requests; every serving key was warmed and every step is bitwise."""
    sched = {"cl": True}
    fake = _FakeL6(["draftasmcl"], steps=1, on=lambda lever: sched["cl"])
    fake.install(monkeypatch)
    p = _variant(monkeypatch, "cl", cl_ab="ab:1")
    fake.install(monkeypatch)
    p._draft_asm.precompile()
    rs = list(range(1, h.MAX_REQS + 1))
    warm_keys = set()
    for R in rs:
        warm_keys.update(h.run_step(p, h.make_step(16000 + R, R), serving=False).keys)
    warm_keys.update(("seed_cl", k) for k in p._draft_asm.warm["seed"])
    warm_keys.update(("carry_fresh_asm", k) for k in p._draft_asm.warm["carry"])
    warm_keys.update(("carry_asm", k) for k in p._draft_asm.warm["carry"])
    h.install(monkeypatch, fold=False)
    bp = h.make_proposer(fold=False)
    h.run_step(bp, h.make_step(16001, 1), serving=False)
    new, used = set(), Counter()
    for i, R in enumerate(rs):
        kind = ("cl", "dha", "base")[i % 3]
        sched["cl"] = kind == "cl"
        p._draft_asm.window_on = kind != "base"
        h.install(monkeypatch, fold=True, cl=True, cl_ab="ab:1")
        fake.install(monkeypatch)
        f = h.run_step(p, h.make_step(17000 + R, R), serving=True)
        h.install(monkeypatch, fold=False)
        b = h.run_step(bp, h.make_step(17000 + R, R), serving=True)
        h.assert_same_outputs(f, b, f"window step R={R} ({kind})")
        got = "base" if not f.folded else ("cl" if f.calls["carry_loop"] else "dha")
        assert got == kind, (R, kind, got, f.calls)
        used[got] += 1
        new.update(k for k in f.keys if k not in warm_keys)
    assert not new, sorted(new)[:4]
    c = p._draft_asm.counts()
    assert c["cold"] == 0 and c["folded_cl"] == used["cl"] and c["folded"] == used["cl"] + used["dha"]
    print(f"\nDRAFT1B_COMPILE stageA warm_keys={len(warm_keys)} new_after_warmup={len(new)} steps={len(rs)} "
          f"stageA={used['cl']} draftasm={used['dha']} base={used['base']} warm={c['warm_keys']}")


def test_superset_seed_precompile_covers_every_decode_call_site(monkeypatch):
    p = _variant(monkeypatch, "cl")
    p._draft_asm.precompile()
    n_seed, n_carry = len(p._draft_asm.warm["seed"]), len(p._draft_asm.warm["carry"])
    assert n_carry == 0  # stage A unconditional: DRAFT-ASM's carries are not precompiled
    for R in range(1, h.MAX_REQS + 1):
        h.run_step(p, h.make_step(18000 + R, R), serving=True)
    r = dict(p._draft_asm.reasons)
    assert r.get("cold_seed", 0) == 0, r
    p2 = _variant(monkeypatch, "cl", cl_ab="ab:1")
    p2._draft_asm.precompile()
    assert len(p2._draft_asm.warm["carry"]) > 0  # windowed: DRAFT-ASM's carries precompiled too
    print(f"\nDRAFT1B_PRECOMPILE superset_seed={n_seed} draftasm_carry={n_carry} (unconditional) / "
          f"{len(p2._draft_asm.warm['carry'])} (windowed); steady call sites cold_seed=0")


# ---------------------------------------------------------------- E. CHECK


@pytest.mark.parametrize("dha", [False, True])
def test_check_mode_compares_then_arms_stage_a(monkeypatch, dha):
    rec = {"info": [], "error": []}
    h.install(monkeypatch, fold=True, cl=True, check_steps=3)
    monkeypatch.setattr(draft_asm, "AGREE_EVERY", 4)
    with h.quiet_logs(monkeypatch, rec):
        p = h.make_proposer()
        p._draft_asm.dha_needed = dha
        h.run_step(p, h.make_step(19001, 20), serving=False)
        outs = [h.run_step(p, h.make_step(19002 + i, 20), serving=True) for i in range(5)]
    assert [o.folded for o in outs] == [False, False, False, True, True]
    assert all(o.calls["carry_loop"] == 1 and o.calls["spec_tail"] == 1 for o in outs[:3])
    if dha:
        assert all(o.calls["loop_asm"] == 1 for o in outs[:3])
    c = p._draft_asm.counts()
    assert c["checked"] == 3 and c["mismatches"] == 0 and not p._draft_asm.failed, (c, rec["error"])
    assert c["folded_cl"] == 2
    print(f"\nDRAFT1B_CHECK stageA clean (draftasm_also={int(dha)}): checked 3 mismatches 0, then folded 2")


@pytest.mark.parametrize("what", ["source", "token0", "loop_token", "positions_step1"])
def test_check_mode_catches_a_perturbed_stage_a_output(monkeypatch, what):
    rec = {"info": [], "error": []}
    h.install(monkeypatch, fold=True, cl=True, check_steps=2)
    monkeypatch.setattr(draft_asm, "AGREE_EVERY", 3)
    if what == "source":
        real = draft_one.fold_source
        monkeypatch.setattr(draft_one, "fold_source", lambda *a: real(*a).index_fill(0, torch.tensor([1]), -5))
    elif what == "token0":
        real_t = draft_one.propose_tokens
        calls = []

        def bad(mtp, hidden, dvp):
            calls.append(1)
            return real_t(mtp, hidden, dvp) + 1

        monkeypatch.setattr(draft_one, "propose_tokens", bad)
    elif what == "loop_token":
        real_b = draft_one.carry_loop_body
        monkeypatch.setattr(draft_one, "carry_loop_body",
                            lambda *a, **k: (lambda o: (o[0], o[1] + 1, *o[2:]))(real_b(*a, **k)))
    else:
        real_s = draft_one._seed_cl_impl
        monkeypatch.setattr(draft_one, "_seed_cl_compiled", h.region(
            "seed_cl", lambda *a: (lambda o: (*o[:9], o[9] + 1))(real_s(*a))))
    with h.quiet_logs(monkeypatch, rec):
        p = h.make_proposer()
        h.run_step(p, h.make_step(19101, 16), serving=False)
        outs = [h.run_step(p, h.make_step(19102 + i, 16), serving=True) for i in range(4)]
    assert not any(o.folded for o in outs), [o.folded for o in outs]
    c = p._draft_asm.counts()
    assert c["mismatches"] >= 2 and p._draft_asm.failed, c
    errs = rec["error"]
    assert "CHECK mismatch" in errs[0] and "every rank keeps the base path" in errs[-1], errs
    print(f"\nDRAFT1B_CHECK stageA perturbed {what}: caught ({errs[0][:90]}), fold off for the run")


# ---------------------------------------------------------------- F. markers, G. init and identity


def test_engagement_markers(monkeypatch):
    rec = {"info": [], "error": []}
    h.install(monkeypatch, fold=True, cl=True)
    monkeypatch.setattr(draft_asm, "LOG_EVERY", 3)
    with h.quiet_logs(monkeypatch, rec):
        p = h.make_proposer()
        h.run_step(p, h.make_step(19201, 5), serving=False)
        for i in range(7):
            h.run_step(p, h.make_step(19202 + i, 5), serving=True)
    eng = [m for m in rec["info"] if m.startswith("q38 DRAFT1-B stage A carry-loop: on")]
    assert eng == ["q38 DRAFT1-B stage A carry-loop: on (TP rank 0)"], rec["info"]
    agr = [m for m in rec["info"] if m.startswith("q38 DRAFT1-B agreement")]
    assert agr and "stage A on" in agr[0], agr
    per = [m for m in rec["info"] if m.startswith("q38 DRAFT1-B: ")]
    assert per and per[-1].startswith("q38 DRAFT1-B: 6 folded = 0 DH-D + 6 stage A + 0 DRAFT-ASM"), per
    print(f"\nDRAFT1B_MARKER {eng[0]} | {agr[0]} | {per[-1]}")


def test_init_builds_draft_one_only_with_a_stage_flag(monkeypatch):
    from unittest import mock

    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    def make():
        sc = SimpleNamespace(draft_tensor_parallel_size=1, method="mtp", draft_model_config=None)
        vc = SimpleNamespace(speculative_config=sc, parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1))
        return Eagle3Proposer(runner=mock.MagicMock(_pcp_enabled=False, _mtp_enabled=True, _mtp_prefill_only=False),
                              vllm_config=vc)

    for name in ("VLLM_TPU_DRAFT_ASM_FOLD",) + h.CLEAR_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLLM_TPU_DRAFT_ASM_CARRY_LOOP", "1")
    assert make()._draft_asm is None  # a stage flag without the fold builds nothing
    monkeypatch.setenv("VLLM_TPU_DRAFT_ASM_FOLD", "1")
    assert type(make()._draft_asm) is draft_one.DraftOne
    monkeypatch.delenv("VLLM_TPU_DRAFT_ASM_CARRY_LOOP")
    assert type(make()._draft_asm) is draft_asm.DraftAsm
    monkeypatch.setenv("VLLM_TPU_DRAFT_ASM_ONE", "1")
    assert type(make()._draft_asm) is draft_one.DraftOne


def test_stage_window_hard_off_is_draft_asm(monkeypatch):
    """CARRY_LOOP=1 with its window "0": DraftOne with no stage configured runs DRAFT-ASM's programs and
    outputs, call for call."""
    (d,), _ = _serve(monkeypatch, "dha", [19301], 24)
    (c,), cp = _serve(monkeypatch, "cl", [19301], 24, cl_ab="0")
    assert type(cp._draft_asm) is draft_one.DraftOne and not cp._draft_asm.cl_conf
    assert Counter(c.calls) == Counter(d.calls), (c.calls, d.calls)
    h.assert_same_outputs(c, d, "stage window off")


# ---------------------------------------------------------------- H. VFY-ONE value 2 hooks


def test_vfy_hooks_name_the_superset_seed(monkeypatch):
    vfy = ModuleType("vllm_torchtpu.runner.vfy_one")
    vfy.VfySeedSpec = lambda impl, args, base: SimpleNamespace(impl=impl, args=args, base=base)
    monkeypatch.setitem(sys.modules, "vllm_torchtpu.runner.vfy_one", vfy)
    p = _variant(monkeypatch, "cl")
    taken = []

    def vfy_args(num_draft, qsl, num_reqs, input_ids, capacity, extras):
        return (num_draft, qsl, torch.tensor(num_reqs, dtype=torch.int64), input_ids,
                p._loop_bucket(num_reqs), capacity)

    def take(chunk, nt, impl, rest):
        taken.append((impl, rest))
        return impl(nt, *rest)

    p._vfy_seed_args = vfy_args
    p._vfy_seed_take = take
    step = h.make_step(19401, 9)
    spec = p._draft_asm.vfy_seed_spec(step.chunk, 9)
    assert spec.impl is draft_one._seed_cl_impl
    assert spec.args[4] is step.chunk.attn_ctx.seq_lens and spec.args[5] is step.chunk.position_ids
    warm = p._draft_asm.vfy_seed_warm_spec(64, 16, 64, 9)
    assert warm.impl is draft_one._seed_cl_impl and warm.args[5].shape == (64,)
    h.run_step(p, h.make_step(19400, 9), serving=False)
    f = h.run_step(p, step, serving=True)
    assert f.folded and taken and taken[-1][0] is draft_one._seed_cl_impl
    assert taken[-1][1][5] is step.chunk.position_ids
    assert f.calls["seed_cl"] == 0  # the verify's superset was taken


# ---------------------------------------------------------------- I. fail-stop and sub-marks


def test_a_carry_loop_build_failure_stops(monkeypatch):
    p = _variant(monkeypatch, "cl")

    def broken(**kw):
        raise ValueError("support_torch_compile refused the module")

    monkeypatch.setattr(draft_one, "build_mtp_fused_loop_carry", broken)
    with pytest.raises(RuntimeError, match="could not build the stage A carry-loop program"):
        h.run_step(p, h.make_step(19501, 7), serving=False)


def test_host_sub_marks_name_the_folded_programs(monkeypatch):
    """Log-only HOSTSUBS sub-marks (runner/host_phases.sub): stage A's program is ~dl_fold and DRAFT-ASM's
    folded carry under DraftOne is ~dc_fold, each after a ~dc_rest boundary (L6-INSTR's convention)."""
    names = []
    monkeypatch.setattr(draft_one.host_phases, "sub", lambda name: names.append(name))
    (c,), _ = _serve(monkeypatch, "cl", [19601], 9)
    assert c.folded and names[-2:] == ["dc_rest", "dl_fold"], names
    names.clear()
    p = _variant(monkeypatch, "cl", cl_ab="1")
    p._draft_asm.dha_needed = True
    h.run_step(p, h.make_step(19700, 9), serving=False)
    p._draft_asm.warm["carry_loop"].clear()
    names.clear()
    d = h.run_step(p, h.make_step(19701, 9), serving=True)
    assert d.folded and d.calls["carry_fresh_asm"] == 1 and names == ["dc_rest", "dc_fold"], (names, d.calls)
    print(f"\nDRAFT1B_MARKER sub-marks stage A {['dc_rest', 'dl_fold']} DRAFT-ASM carry {names}")
