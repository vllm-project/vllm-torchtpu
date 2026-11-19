# SPDX-License-Identifier: Apache-2.0
"""q38 L4 VFY-ONE value 2 (DH-C) composed with DRAFT-ASM (DH-A) on q38-l4-vfy-v5. CDK CPU pod only.

Rule: one seed fold per step. The VFY verify program computes the drafter's seed once (with a DraftAsm, its
superset seed through DraftAsm.vfy_seed_spec); DraftAsm's fold-mode seed() and eagle3's base fused-seed block take
it (Eagle3Proposer._vfy_seed_take: full argument identity, unchanged in-place version counters) instead of running a
seed program. Driven through the real Eagle3Proposer.propose and the real async spec substitution assembler with
DRAFT-ASM's harness (tests/q38_draftasm/draftasm_harness.py; every compiled program is a counted region). The stash
the runner's _vfy_verify leaves is built here as the program computes it: spec.fn(next_tokens, *spec.args)
(_vfy_one_impl returns `seed_fn(tokens, *seed_args)`; its bitwise identity to the D83 chain is tests/l4vfy).

C1 no DraftAsm: the base block takes the verify seed; 0 seed programs; outputs bitwise the no-VFY run (5 row counts).
C2 DraftAsm folding: DraftAsm takes the superset; 0 seed programs of either kind; bitwise the flag-off run.
C3 DraftAsm present but not folding this step: the base block takes the superset's first five; bitwise.
C4 L4-E's seq_lens item: chunk.attn_ctx.seq_lens written in place after the verify -> DraftAsm runs its own seed on
   the written values (seed_written), bitwise the flag-off run on the same write.
C5 a seed input replaced after the verify (not the same object) -> the base block runs its own seed; bitwise.
C6 the warm-up seed spec (DraftAsm's superset) has the serving spec's key at a steady shape.
C7 a multi-step run with DRAFT-ASM's window switching per step: every step makes no seed program call.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "q38_draftasm"))
import draftasm_harness as h  # noqa: E402
from vllm_torchtpu.runner import vfy_one  # noqa: E402
from vllm_torchtpu.spec_decode import draft_asm, eagle3  # noqa: E402

ROWS = [8, 16, 32, 48, 64]
COUNTS: dict = {}


def _inject(p, step):
    """The stash the runner's VFY step leaves for this chunk (value 2), and the runner's controller."""
    runner = p.runner
    if runner.__dict__.get("_vfy") is None:
        runner._vfy = vfy_one.VfyOne(2, 0, "")
    spec = p.vfy_seed_spec(step.chunk, step.R)
    assert spec is not None
    outs = tuple(spec.fn(step.nt, *spec.args))
    runner._vfy_seed = (spec, step.nt, outs, vfy_one.versions((step.nt, *spec.args, *spec.base_args)))
    return spec, outs


def _seed_calls(out):
    return out.calls["seed"] + out.calls["seed_asm"]


def _base(monkeypatch, seed, R, write=None):
    """The flag-off run (no DRAFT-ASM, no VFY) of the same step."""
    h.install(monkeypatch, fold=False)
    bp = h.make_proposer(False)
    h.run_step(bp, h.make_step(seed, R), serving=False)
    s = h.make_step(seed, R)
    if write:
        write(s)
    return h.run_step(bp, s, serving=True)


@pytest.mark.parametrize("R", ROWS)
def test_c1_base_block_takes_the_verify_seed(monkeypatch, R):
    b = _base(monkeypatch, 7100 + R, R)
    h.install(monkeypatch, fold=False)
    p = h.make_proposer(False)
    h.run_step(p, h.make_step(7100 + R, R), serving=False)
    s = h.make_step(7100 + R, R)
    spec, outs = _inject(p, s)
    assert spec.fn is eagle3._async_seed_inputs_impl and len(outs) == 5
    v = h.run_step(p, s, serving=True)
    h.assert_same_outputs(v, b, f"C1 R={R}")
    assert b.calls["seed"] == 1 and _seed_calls(v) == 0, (b.calls, v.calls)
    assert p.runner._vfy.stats["seed_taken"] == 1 and "_vfy_seed" not in p.runner.__dict__
    COUNTS["c1"] = COUNTS.get("c1", 0) + 1


@pytest.mark.parametrize("R", ROWS)
def test_c2_draftasm_fold_takes_the_superset(monkeypatch, R):
    b = _base(monkeypatch, 7200 + R, R)
    h.install(monkeypatch, fold=True)
    fp = h.make_proposer(True)
    h.run_step(fp, h.make_step(7200 + R, R), serving=False)
    s = h.make_step(7200 + R, R)
    spec, outs = _inject(fp, s)
    assert spec.fn is draft_asm._async_seed_inputs_asm_impl and len(outs) == 8
    f = h.run_step(fp, s, serving=True)
    assert f.folded, fp._draft_asm.counts()
    h.assert_same_outputs(f, b, f"C2 R={R}")
    assert _seed_calls(f) == 0, f.calls  # one seed fold per step: the verify's
    assert f.calls["loop_asm"] == 1 and f.calls["spec_tail"] == 0  # DH-A's other folds stay
    assert fp.runner._vfy.stats["seed_taken"] == 1
    COUNTS["c2"] = COUNTS.get("c2", 0) + 1


def test_c3_draftasm_not_folding_the_base_block_takes_the_first_five(monkeypatch):
    R = 24
    b = _base(monkeypatch, 7300, R)
    h.install(monkeypatch, fold=True)
    fp = h.make_proposer(True)
    h.run_step(fp, h.make_step(7300, R), serving=False)
    fp._draft_asm.window_on = False  # the first serving agreement: the fold is off for this window
    s = h.make_step(7300, R)
    spec, outs = _inject(fp, s)
    f = h.run_step(fp, s, serving=True)
    assert not f.folded and _seed_calls(f) == 0, (f.folded, f.calls)
    h.assert_same_outputs(f, b, "C3")
    assert fp.runner._vfy.stats["seed_taken"] == 1
    COUNTS["c3"] = 1


def test_c4_seq_lens_written_in_place_after_the_verify_is_caught(monkeypatch):
    R = 32

    def write(s):
        s.chunk.attn_ctx.seq_lens.add_(3)
    b = _base(monkeypatch, 7400, R, write)
    h.install(monkeypatch, fold=True)
    fp = h.make_proposer(True)
    h.run_step(fp, h.make_step(7400, R), serving=False)
    s = h.make_step(7400, R)
    spec, stale = _inject(fp, s)
    write(s)  # between the verify and the seed
    f = h.run_step(fp, s, serving=True)
    assert f.folded and f.calls["seed_asm"] == 1 and f.calls["seed"] == 0, f.calls
    assert fp.runner._vfy.stats["seed_written"] == 1 and fp.runner._vfy.stats["seed_taken"] == 0
    h.assert_same_outputs(f, b, "C4")
    fresh = draft_asm._async_seed_inputs_asm_impl(s.nt, *spec.args)
    assert not torch.equal(stale[5], fresh[5])  # the stale step-1 seq_lens would have been wrong
    COUNTS["c4"] = 1


def test_c5_a_replaced_seed_input_is_not_taken(monkeypatch):
    R = 16

    def write(s):
        s.chunk.input_ids = s.chunk.input_ids.roll(1)  # a new tensor: another object, other values
    b = _base(monkeypatch, 7500, R, write)
    h.install(monkeypatch, fold=False)
    p = h.make_proposer(False)
    h.run_step(p, h.make_step(7500, R), serving=False)
    s = h.make_step(7500, R)
    _inject(p, s)
    write(s)
    v = h.run_step(p, s, serving=True)
    assert v.calls["seed"] == 1 and p.runner._vfy.stats["seed_miss"] == 1
    h.assert_same_outputs(v, b, "C5")
    COUNTS["c5"] = 1


def test_c6_warm_spec_has_the_serving_key(monkeypatch):
    h.install(monkeypatch, fold=True)
    fp = h.make_proposer(True)
    for R in (5, 16, 40):
        s = h.make_step(7600 + R, R)
        live = fp.vfy_seed_spec(s.chunk, R)
        T, B = s.chunk.input_ids.shape[0], s.chunk.draft_lengths.shape[0]
        cap = s.chunk.attn_ctx.seq_lens.shape[0]
        warm = fp.vfy_seed_warm_spec(T, B, cap, R)
        assert warm.fn is live.fn is draft_asm._async_seed_inputs_asm_impl
        assert vfy_one.key(warm.fn, *warm.args) == vfy_one.key(live.fn, *live.args), R
        assert vfy_one.key(*warm.base_args) == vfy_one.key(*live.base_args), R
    COUNTS["c6"] = 1


def test_c7_window_switching_run_makes_no_seed_program_call(monkeypatch):
    h.install(monkeypatch, fold=True)
    monkeypatch.setattr(draft_asm, "AGREE_EVERY", 1)  # a DRAFT-ASM boundary on every serving call
    fp = h.make_proposer(True)
    rs = [8, 13, 32, 47, 64, 9]
    for R in rs:
        h.run_step(fp, h.make_step(7700 + R, R), serving=False)
    h.install(monkeypatch, fold=False)
    bp = h.make_proposer(False)
    for R in rs:
        h.run_step(bp, h.make_step(7700 + R, R), serving=False)
    folds = 0
    for i, R in enumerate(rs):
        h.install(monkeypatch, fold=True)
        fp._draft_asm.window_on = i % 2 == 0
        s = h.make_step(7800 + R, R)
        _inject(fp, s)
        f = h.run_step(fp, s, serving=True)
        h.install(monkeypatch, fold=False)
        b = h.run_step(bp, h.make_step(7800 + R, R), serving=True)
        h.assert_same_outputs(f, b, f"C7 R={R}")
        assert _seed_calls(f) == 0, (R, f.calls)
        folds += f.folded
    assert folds == len(rs) // 2 and fp.runner._vfy.stats["seed_taken"] == len(rs)
    COUNTS["c7_steps"] = len(rs)


def test_c8_a_draftasm_that_can_never_fold_offers_no_seed(monkeypatch):
    """DraftAsm._static() non-empty: its hook returns None, so the verify folds no seed (eagle3's spec is None) and
    every seed program runs as without VFY-ONE."""
    h.install(monkeypatch, fold=True)
    fp = h.make_proposer(True)
    s = h.make_step(7900, 12)
    assert fp.vfy_seed_spec(s.chunk, 12) is not None
    fp._draft_asm.static_reason = "tail_flags"
    assert fp.vfy_seed_spec(s.chunk, 12) is None
    assert fp.vfy_seed_warm_spec(s.chunk.input_ids.shape[0], s.chunk.draft_lengths.shape[0], 64, 12) is None
    COUNTS["c8"] = 1


def test_zz_counts():
    print("Q38_L4VFY_COMPOSE_COUNTS " + " ".join(f"{k}={v}" for k, v in sorted(COUNTS.items())))
