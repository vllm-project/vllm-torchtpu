# SPDX-License-Identifier: Apache-2.0
"""q38 L4 DRAFT-ASM through the real ``Eagle3Proposer.propose`` and the real async spec substitution
assembler (tests/q38_draftasm/draftasm_harness.py), CPU only.

A. Bitwise: flag on (after warm-up) == flag off for the drafts, step 1's seq_lens / positions / tokens /
   hidden at the loop call, the source, the rejected count and the index dicts; rows 4 / 8 / 12 / 16 / 32 /
   48 / 64, K = 3, fallback rows on and off, requests with fewer drafts.
B. Census per served step: flag off runs the eager seq_lens + 1 - rejected, positions + 1 and stack + cat
   (add 2, sub 1, cat 2 aten ops), blocks on deferred loop inputs and calls the spec tail; flag on runs 0
   eager ops, 0 deferred loop inputs and no spec tail.
C. Steps that must keep the base path do (several chunks, a prefill in the chunk, structured output,
   REJ_PAD off, a never-warmed shape); their outputs equal the flag-off path's.
D. SAB dual-path: warm-up runs the base path for results and warms the folded programs beside it; after
   it, a window switching per step never compiles (every serving program key was warmed) and stays bitwise;
   the precompile covers the folded seed / carry at every decode call site; the l6_ab lever reads as
   documented.
E. CHECK compares both paths on its steps, arms the fold at the next agreement when clean, and keeps the
   base path for the rest of the run when a folded output is perturbed. F. Engagement markers. G. __init__ builds the controller
   only with the flag.
"""

from collections import Counter
from types import ModuleType, SimpleNamespace

import pytest
import torch

import draftasm_harness as h
from vllm_torchtpu.spec_decode import draft_asm
from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

ROWS = [4, 8, 12, 16, 32, 48, 64]


def _pair(monkeypatch, seed, R, **kw):
    """(flag-off serving step, flag-on serving step) on equal inputs; each proposer warmed once first."""
    h.install(monkeypatch, fold=False)
    base_p = h.make_proposer(False)
    h.run_step(base_p, h.make_step(seed, R, **kw), serving=False)
    b = h.run_step(base_p, h.make_step(seed, R, **kw), serving=True)
    h.install(monkeypatch, fold=True)
    fold_p = h.make_proposer(True)
    h.run_step(fold_p, h.make_step(seed, R, **kw), serving=False)
    f = h.run_step(fold_p, h.make_step(seed, R, **kw), serving=True)
    return b, f, fold_p


def _norm(ops):
    return Counter("cat" if o == "stack" else o for o in ops)


# ---------------------------------------------------------------- A. bitwise


@pytest.mark.parametrize("short", [False, True])
@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("R", ROWS)
def test_fold_is_bitwise_the_base_path(monkeypatch, R, fallback, short):
    b, f, fold_p = _pair(monkeypatch, 1000 + R, R, fallback=fallback, short=short)
    assert not b.folded and f.folded, (b.folded, f.folded, fold_p._draft_asm.counts())
    h.assert_same_outputs(f, b, f"R={R}")
    print(f"\nDRAFTASM_BITWISE propose R={R} fallback={int(fallback)} short={int(short)} drafts {tuple(f.drafts.shape)} "
          f"seq_lens/positions/tokens/hidden at the loop call, source[{f.src.numel()}], "
          f"rejected[{f.rejected.numel()}] equal; folded={int(f.folded)}")


# ---------------------------------------------------------------- B. census


@pytest.mark.parametrize("R", [8, 32, 48, 64])
def test_census_eager_defer_and_programs(monkeypatch, R):
    b, f, _ = _pair(monkeypatch, 2000 + R, R)
    assert _norm(b.eager) == Counter({"add": 2, "sub": 1, "cat": 2}), b.eager
    assert set(b.deferred) == {"positions", "seq_lens", "md.seq_lens"}, b.deferred
    assert b.calls["spec_tail"] == 1 and b.calls["loop"] == 1 and b.calls["seed"] == 1
    assert f.eager == [], f.eager
    assert f.deferred == [], f.deferred
    assert f.calls["spec_tail"] == 0 and f.calls["loop"] == 0 and f.calls["seed"] == 0
    assert f.calls["seed_asm"] == 1 and f.calls["loop_asm"] == 1
    carry_b = b.calls["carry_fresh"] + b.calls["carry"]
    carry_f = f.calls["carry_fresh_asm"] + f.calls["carry_asm"]
    assert carry_b == 1 and carry_f == 1
    nb, nf = sum(b.calls.values()), sum(f.calls.values())
    assert nf == nb - 1, (b.calls, f.calls)
    fmt = lambda c: "+".join(f"{k}{v}" for k, v in sorted(_norm(c).items())) or "0"
    print(f"\nDRAFTASM_CENSUS R={R} flag=off eager={fmt(b.eager)} deferred={len(b.deferred)} "
          f"programs={nb} spec_tail={b.calls['spec_tail']} ({dict(sorted(b.calls.items()))})")
    print(f"\nDRAFTASM_CENSUS R={R} flag=on eager={fmt(f.eager)} deferred={len(f.deferred)} "
          f"programs={nf} spec_tail={f.calls['spec_tail']} ({dict(sorted(f.calls.items()))})")


# ---------------------------------------------------------------- C. standalone paths


def _base_and_flag_on(monkeypatch, make, serving_warm=True, env=None):
    def setenv():
        for k, v in (env or {}).items():
            monkeypatch.setenv(k, v)

    h.install(monkeypatch, fold=False)
    setenv()
    base_p = h.make_proposer(False)
    h.run_step(base_p, make(), serving=False)
    b = h.run_step(base_p, make(), serving=True)
    h.install(monkeypatch, fold=True)
    setenv()
    fold_p = h.make_proposer(True)
    if serving_warm:
        h.run_step(fold_p, make(), serving=False)
    f = h.run_step(fold_p, make(), serving=True)
    return b, f, fold_p


def test_two_chunks_keep_the_base_path(monkeypatch):
    make = lambda: [h.make_step(3001, 6), h.make_step(3002, 5)]
    b, f, fp = _base_and_flag_on(monkeypatch, make)
    assert not f.folded and f.calls["seed_asm"] == 0 and f.calls["loop_asm"] == 0
    assert f.calls["spec_tail"] == b.calls["spec_tail"]
    h.same(f.drafts, b.drafts, "drafts")
    h.same(f.src, b.src, "source")
    h.same(f.rejected, b.rejected, "rejected")
    assert fp._draft_asm.reasons["chunks"] == 1
    print("\nDRAFTASM_STANDALONE two_chunks base path kept, outputs equal")


def test_prefill_in_the_chunk_keeps_the_base_path(monkeypatch):
    make = lambda: h.make_step(3003, 9, mixed=True)
    b, f, fp = _base_and_flag_on(monkeypatch, make)
    assert not f.folded and f.calls["seed_asm"] == 0 and f.calls["spec_tail"] == 1
    h.assert_same_outputs(f, b, "mixed")
    assert fp._draft_asm.reasons["mixed"] == 1
    print("\nDRAFTASM_STANDALONE mixed base path kept, outputs equal")


def test_structured_output_keeps_the_base_path(monkeypatch):
    h.install(monkeypatch, fold=True)
    fp = h.make_proposer(True)
    h.run_step(fp, h.make_step(3004, 7), serving=False)
    f = h.run_step(fp, h.make_step(3004, 7), serving=True, structured=True)
    assert not f.folded and f.calls["spec_tail"] == 1 and f.calls["seed_asm"] == 0
    assert fp._draft_asm.reasons["structured_output"] == 1
    print("\nDRAFTASM_STANDALONE structured_output base path kept")


def test_rejected_count_unpadded_keeps_the_base_path(monkeypatch):
    make = lambda: h.make_step(3005, 7)
    b, f, fp = _base_and_flag_on(monkeypatch, make, env={"VLLM_TPU_SPEC_TAIL_REJ_PAD": "0"})
    assert not f.folded and f.calls["seed_asm"] == 0 and f.calls["spec_tail"] == 1
    h.assert_same_outputs(f, b, "rej_pad_off")
    assert fp._draft_asm.reasons["tail_flags"] == 1
    print("\nDRAFTASM_STANDALONE rej_pad_off base path kept, outputs equal")


def test_never_warmed_shape_falls_back_without_a_folded_call(monkeypatch):
    make = lambda: h.make_step(3006, 12)
    b, f, fp = _base_and_flag_on(monkeypatch, make, serving_warm=False)
    assert not f.folded
    assert f.calls["seed_asm"] == 0 and f.calls["carry_fresh_asm"] == 0 and f.calls["loop_asm"] == 0
    assert fp._draft_asm.stats["cold"] == 1 and fp._draft_asm.reasons["cold_seed"] == 1
    h.assert_same_outputs(f, b, "cold")
    print("\nDRAFTASM_STANDALONE cold base path kept, no folded program called, outputs equal")


def test_runner_stores_the_fold_only_for_its_own_chunk(monkeypatch):
    """A folded result handed to the assembler with another verify output (or draft lengths, or request
    count) is not stored: the drafts are rebuilt eagerly and the spec tail runs."""
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    h.install(monkeypatch, fold=True)
    fp = h.make_proposer(True)
    h.run_step(fp, h.make_step(3101, 11), serving=False)
    step = h.make_step(3102, 11)
    f = h.run_step(fp, step, serving=True)
    assert f.folded
    res = draft_asm.DraftAsmResult(f.src, f.rejected, step.nt, step.chunk.draft_lengths, 11, [])
    runner = h.make_assembler_runner(fp)
    state = SimpleNamespace(num_reqs_list=[11], spec_decode_metadata_list=[step.md])
    tail, _ = fp._draft_asm.assemble_tail(runner, res, [step.nt], state)
    assert tail is not None and tail[0][0] is f.src
    calls = []
    monkeypatch.setattr(draft_asm.DraftAsmResult, "drafts", lambda self: calls.append(1) or f.drafts)
    for nts, st in (([step.nt.clone()], state),
                    ([step.nt], SimpleNamespace(num_reqs_list=[10], spec_decode_metadata_list=[step.md])),
                    ([step.nt], SimpleNamespace(num_reqs_list=[11], spec_decode_metadata_list=[
                        SimpleNamespace(draft_lengths=step.md.draft_lengths.clone(),
                                        draft_lengths_cpu=step.md.draft_lengths_cpu)]))):
        tail, drafts = fp._draft_asm.assemble_tail(runner, res, nts, st)
        assert tail is None and drafts is f.drafts
    assert len(calls) == 3 and fp._draft_asm.stats["tail_miss"] == 3
    h.CEN.reset()
    out = TPUModelRunner._assemble_async_spec_substitution(runner, res, [step.nt.clone()], state)
    assert h.CEN.calls["spec_tail"] == 1
    h.same(out[2], f.rejected, "rejected via the spec tail")


# ---------------------------------------------------------------- D. SAB dual-path


def test_warmup_runs_base_and_warms_the_folded_programs(monkeypatch):
    h.install(monkeypatch, fold=True)
    fp = h.make_proposer(True)
    w = h.run_step(fp, h.make_step(4001, 10), serving=False)
    # results from the base path, the folded trio beside it
    assert not w.folded
    # (each loop's first call runs twice: trace, then compiled)
    assert w.calls["seed"] == 1 and w.calls["loop"] == 2 and w.calls["spec_tail"] == 1
    assert w.calls["seed_asm"] == 1 and w.calls["carry_fresh_asm"] == 1 and w.calls["loop_asm"] == 2
    assert {k: len(v) for k, v in fp._draft_asm.warm.items()} == {"seed": 1, "carry": 1, "loop": 1}
    h.install(monkeypatch, fold=False)
    bp = h.make_proposer(False)
    b = h.run_step(bp, h.make_step(4001, 10), serving=False)
    h.assert_same_outputs(w, b, "warm-up step")


def test_window_switching_per_step_never_compiles_and_stays_bitwise(monkeypatch):
    h.install(monkeypatch, fold=True)
    monkeypatch.setattr(draft_asm, "AGREE_EVERY", 1)  # a boundary on every serving call
    fp = h.make_proposer(True)
    fp._draft_asm.precompile()
    rs = list(range(1, h.MAX_REQS + 1))
    warm_keys = set()
    for R in rs:
        out = h.run_step(fp, h.make_step(5000 + R, R), serving=False)
        warm_keys.update(out.keys)
    warm_keys.update(("seed_asm", k) for k in fp._draft_asm.warm["seed"])
    warm_keys.update(("carry_fresh_asm", k) for k in fp._draft_asm.warm["carry"])
    warm_keys.update(("carry_asm", k) for k in fp._draft_asm.warm["carry"])
    h.install(monkeypatch, fold=False)
    bp = h.make_proposer(False)
    h.run_step(bp, h.make_step(5001, 1), serving=False)
    h.install(monkeypatch, fold=True)
    # the warmed proposer keeps its programs; re-install the regions only
    serving_keys, new, folds = set(), set(), 0
    for i, R in enumerate(rs):
        fp._draft_asm.window_on = i % 2 == 0
        f = h.run_step(fp, h.make_step(6000 + R, R), serving=True)
        h.install(monkeypatch, fold=False)
        b = h.run_step(bp, h.make_step(6000 + R, R), serving=True)
        h.install(monkeypatch, fold=True)
        h.assert_same_outputs(f, b, f"window step R={R}")
        assert f.folded == (i % 2 == 0), (R, f.folded, fp._draft_asm.counts())
        folds += f.folded
        serving_keys.update(f.keys)
        new.update(k for k in f.keys if k not in warm_keys)
    assert not new, sorted(new)[:4]
    c = fp._draft_asm.counts()
    assert c["cold"] == 0 and c["folded"] == folds == len(rs) // 2 and c["window_off"] == len(rs) - folds
    print(f"\nDRAFTASM_COMPILE warm_keys={len(warm_keys)} serving_keys={len(serving_keys)} new_after_warmup={len(new)} "
          f"steps={len(rs)} folded={folds} window_off={c['window_off']} warm={c['warm_keys']}")


def test_precompile_covers_every_decode_call_site(monkeypatch):
    """With only DraftAsm.precompile's keys, no steady decode step (1..64 requests, K drafts each) finds its
    folded seed or carry cold (the loop is warmed by the warm-up, not here); a step whose requests carry
    fewer drafts finds them cold exactly when its token bucket is not the steady one, and runs the base path."""
    from vllm_torchtpu.runner.tpu_runner import _get_padded_token_len

    h.install(monkeypatch, fold=True)
    fp = h.make_proposer(True)
    fp._draft_asm.precompile()
    n_seed, n_carry = len(fp._draft_asm.warm["seed"]), len(fp._draft_asm.warm["carry"])
    for R in range(1, h.MAX_REQS + 1):
        h.run_step(fp, h.make_step(7000 + R, R), serving=True)
    r = dict(fp._draft_asm.reasons)
    assert r.get("cold_seed", 0) == 0 and r.get("cold_carry", 0) == 0, r
    assert r.get("cold_loop", 0) == h.MAX_REQS, r
    from vllm_torchtpu.runner.tpu_runner import _get_padded_num_reqs_with_upper_limit

    shapes = set(fp._draft_asm.decode_shapes())
    want_cold = 0
    for R in range(1, h.MAX_REQS + 1):
        step = h.make_step(7100 + R, R, short=True)
        key = (step.chunk.input_ids.shape[0], _get_padded_num_reqs_with_upper_limit(R, h.MAX_REQS),
               _get_padded_token_len(h.PADDINGS, R), 64)
        want_cold += key not in shapes  # cold exactly when its (T, width, p, cap) is no steady shape
        h.run_step(fp, step, serving=True)
    r2 = dict(fp._draft_asm.reasons)
    assert r2.get("cold_seed", 0) == want_cold and r2.get("cold_carry", 0) == 0, (r2, want_cold)
    rss = fp._draft_asm.precompile_rss
    per = "n/a" if None in rss[:2] else f"{(rss[1] - rss[0]) / max(1, rss[2]):.0f} B"
    print(f"\nDRAFTASM_PRECOMPILE seed={n_seed} carry={n_carry} programs (steady decode, 1..{h.MAX_REQS} requests); "
          f"steady call sites cold_seed=0 cold_carry=0; fewer-draft steps cold {want_cold} of {h.MAX_REQS} (base path); "
          f"CPU RSS per eager-warmed program {per} (eager CPU bodies, not TPU executables)")


class _FakeL6:
    def __init__(self, steps=200, on=False, registered=True):
        self.LEVERS = {"draftasm": "VLLM_TPU_DRAFT_ASM_AB"} if registered else {}
        self.STEPS = steps
        self._on = on
        self.notes = Counter()

    def on(self, lever):
        return self._on

    def note(self, lever, key, n=1):
        self.notes[(lever, key)] += n


def _with_l6(monkeypatch, fake, ab):
    import sys

    mod = ModuleType("vllm_torchtpu.core.l6_ab")
    for k in ("LEVERS", "STEPS"):
        setattr(mod, k, getattr(fake, k))
    mod.on, mod.note = fake.on, fake.note
    monkeypatch.setitem(sys.modules, "vllm_torchtpu.core.l6_ab", mod)
    import vllm_torchtpu.core as core

    monkeypatch.setattr(core, "l6_ab", mod, raising=False)
    monkeypatch.setattr(draft_asm, "_L6_AB", [])
    monkeypatch.setenv("VLLM_TPU_DRAFT_ASM_AB", ab)
    return draft_asm.DraftAsm(h.make_proposer(False))


def test_l6_ab_window_reading(monkeypatch):
    h.install(monkeypatch, fold=True)
    errs = []
    monkeypatch.setattr(draft_asm, "logger", SimpleNamespace(error=lambda *a: errs.append(a[0] % a[1:]),
                                                           info=lambda *a: None, warning=lambda *a, **k: None))
    assert _with_l6(monkeypatch, _FakeL6(on=False), "").window() is True  # uniform: the flag governs
    assert _with_l6(monkeypatch, _FakeL6(on=True), "0").window() is False
    assert _with_l6(monkeypatch, _FakeL6(on=False), "1").window() is True
    assert _with_l6(monkeypatch, _FakeL6(on=True), "ab:1").window() is True
    assert _with_l6(monkeypatch, _FakeL6(on=False), "ab:1").window() is False
    assert not errs
    assert _with_l6(monkeypatch, _FakeL6(on=True, steps=0), "ab:1").window() is False  # wall-clock windows
    assert _with_l6(monkeypatch, _FakeL6(on=True, registered=False), "ab:1").window() is False
    assert _with_l6(monkeypatch, _FakeL6(on=True, registered=False), "").window() is True
    assert len(errs) == 2 and "VLLM_TPU_L6_AB_STEPS" in errs[0] and "lever 'draftasm'" in errs[1], errs
    # no l6_ab at all: "" and "1" fold, "0" does not, "ab:" refuses with one ERROR
    import sys

    import vllm_torchtpu.core as core

    monkeypatch.delattr(core, "l6_ab", raising=False)
    monkeypatch.setitem(sys.modules, "vllm_torchtpu.core.l6_ab", None)
    for ab, want in (("", True), ("1", True), ("0", False), ("ab:1+3", False)):
        monkeypatch.setattr(draft_asm, "_L6_AB", [])
        monkeypatch.setenv("VLLM_TPU_DRAFT_ASM_AB", ab)
        assert draft_asm.DraftAsm(h.make_proposer(False)).window() is want, ab
    print("\nDRAFTASM_WINDOW l6_ab lever 'draftasm': ''/1/0/ab:<c> read as documented; wall-clock and unregistered "
          "windows refused with one ERROR each")


def test_l6_ab_windows_drive_the_fold_and_note_it(monkeypatch):
    h.install(monkeypatch, fold=True, ab="ab:1")
    fake = _FakeL6(on=True, steps=1)  # an agreement on every serving call
    _with_l6(monkeypatch, fake, "ab:1")
    fp = h.make_proposer(True)
    h.run_step(fp, h.make_step(8001, 9), serving=False)
    f1 = h.run_step(fp, h.make_step(8002, 9), serving=True)
    fake._on = False
    f2 = h.run_step(fp, h.make_step(8003, 9), serving=True)
    assert f1.folded and not f2.folded
    assert fake.notes[("draftasm", "fold")] == 1


# ---------------------------------------------------------------- E. CHECK


def _logs(monkeypatch):
    rec = {"info": [], "error": []}
    monkeypatch.setattr(draft_asm, "logger", SimpleNamespace(
        info=lambda *a: rec["info"].append(a[0] % a[1:]),
        error=lambda *a: rec["error"].append(a[0] % a[1:]),
        warning=lambda *a, **k: None,
    ))
    return rec


def test_check_mode_compares_then_arms_the_fold(monkeypatch):
    h.install(monkeypatch, fold=True, check_steps=3)
    monkeypatch.setattr(draft_asm, "AGREE_EVERY", 4)  # boundaries at serving calls 1, 4, 8
    rec = _logs(monkeypatch)
    fp = h.make_proposer(True)
    h.run_step(fp, h.make_step(9001, 20), serving=False)
    outs = [h.run_step(fp, h.make_step(9002 + i, 20), serving=True) for i in range(5)]
    assert [o.folded for o in outs] == [False, False, False, True, True]
    assert all(o.calls["seed_asm"] == 1 and o.calls["spec_tail"] == 1 for o in outs[:3])
    c = fp._draft_asm.counts()
    assert c["checked"] == 3 and c["mismatches"] == 0 and not fp._draft_asm.failed
    assert not rec["error"] and any("CHECK: 3 steps, 0 mismatches" in m for m in rec["info"])
    print(f"\nDRAFTASM_CHECK clean: checked {c['checked']} mismatches 0, then folded {c['folded']}")


@pytest.mark.parametrize("what", ["source", "seq_lens_step1", "positions_step1", "drafts"])
def test_check_mode_catches_a_perturbed_folded_output(monkeypatch, what):
    h.install(monkeypatch, fold=True, check_steps=2)
    monkeypatch.setattr(draft_asm, "AGREE_EVERY", 3)  # boundaries at serving calls 1, 3
    rec = _logs(monkeypatch)
    if what == "source":
        real = draft_asm.fold_source
        monkeypatch.setattr(draft_asm, "fold_source", lambda *a: real(*a).index_fill(0, torch.tensor([1]), -5))
    elif what == "seq_lens_step1":
        real_seed = draft_asm._async_seed_inputs_asm_impl
        monkeypatch.setattr(draft_asm, "_async_seed_inputs_asm_compiled", h.region(
            "seed_asm", lambda *a: (lambda o: (*o[:5], o[5] + 1, *o[6:]))(real_seed(*a))))
    elif what == "positions_step1":
        real_c = draft_asm._carry_token_fresh_asm_impl
        monkeypatch.setattr(draft_asm, "_carry_token_fresh_asm_compiled", h.region(
            "carry_fresh_asm", lambda *a: (lambda o: (*o[:3], o[3] + 1))(real_c(*a))))
    else:
        real_c = draft_asm._carry_token_fresh_asm_impl
        monkeypatch.setattr(draft_asm, "_carry_token_fresh_asm_compiled", h.region(
            "carry_fresh_asm", lambda *a: (lambda o: (o[0], o[1] + 1, *o[2:]))(real_c(*a))))
    fp = h.make_proposer(True)
    h.run_step(fp, h.make_step(9101, 16), serving=False)
    outs = [h.run_step(fp, h.make_step(9102 + i, 16), serving=True) for i in range(4)]
    assert not any(o.folded for o in outs), [o.folded for o in outs]
    c = fp._draft_asm.counts()
    assert c["mismatches"] >= 2 and fp._draft_asm.failed and c["failed"] == 2, c
    errs = rec["error"]
    assert len(errs) == 2 and "CHECK mismatch" in errs[0] and what.split("_")[0] in errs[0], errs
    assert "every rank keeps the base path" in errs[1] and "CHECK:" in errs[1], errs
    print(f"\nDRAFTASM_CHECK perturbed {what}: caught (mismatches {c['mismatches']}), fold off for the run")


# ---------------------------------------------------------------- F. markers, G. init, multi-step


def test_engagement_markers(monkeypatch):
    h.install(monkeypatch, fold=True)
    rec = _logs(monkeypatch)
    monkeypatch.setattr(draft_asm, "LOG_EVERY", 3)
    fp = h.make_proposer(True)
    h.run_step(fp, h.make_step(9201, 5), serving=False)
    for i in range(7):
        h.run_step(fp, h.make_step(9202 + i, 5), serving=True)
    eng = [m for m in rec["info"] if m.startswith("q38 DRAFT-ASM fold: on")]
    assert eng == ["q38 DRAFT-ASM fold: on (TP rank 0)"], rec["info"]
    per = [m for m in rec["info"] if m.startswith("q38 DRAFT-ASM: ")]
    assert len(per) == 2 and per[-1].startswith("q38 DRAFT-ASM: 6 folded, 0 base"), per
    print(f"\nDRAFTASM_MARKER {eng[0]} | {per[-1]}")


def test_init_builds_the_controller_only_with_the_flag(monkeypatch):
    from unittest import mock

    def make():
        sc = SimpleNamespace(draft_tensor_parallel_size=1, method="mtp", draft_model_config=None)
        vc = SimpleNamespace(speculative_config=sc, parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1))
        return Eagle3Proposer(runner=mock.MagicMock(_pcp_enabled=False, _mtp_enabled=True, _mtp_prefill_only=False),
                              vllm_config=vc)

    monkeypatch.delenv("VLLM_TPU_DRAFT_ASM_FOLD", raising=False)
    assert make()._draft_asm is None and Eagle3Proposer._draft_asm is None
    monkeypatch.setenv("VLLM_TPU_DRAFT_ASM_FOLD", "1")
    assert isinstance(make()._draft_asm, draft_asm.DraftAsm)


def test_each_step_stores_its_own_rejected_count(monkeypatch):
    h.install(monkeypatch, fold=True)
    fp = h.make_proposer(True)
    h.run_step(fp, h.make_step(9301, 24), serving=False)
    fold = [h.run_step(fp, h.make_step(9302 + i, 24, fallback=i % 2 == 1), serving=True) for i in range(3)]
    h.install(monkeypatch, fold=False)
    bp = h.make_proposer(False)
    h.run_step(bp, h.make_step(9301, 24), serving=False)
    base = [h.run_step(bp, h.make_step(9302 + i, 24, fallback=i % 2 == 1), serving=True) for i in range(3)]
    assert all(f.folded for f in fold)
    assert not torch.equal(base[0].rejected, base[1].rejected)
    for i, (f, b) in enumerate(zip(fold, base)):
        h.assert_same_outputs(f, b, f"step {i}")
