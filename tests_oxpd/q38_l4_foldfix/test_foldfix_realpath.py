# SPDX-License-Identifier: Apache-2.0
"""q38 L4 FOLDFIX: the REAL draft path, propose -> SpeculativeDecodingManager.propose_draft_token_ids -> the runner's
sample_tokens consumer -> the next step's inputs, over CHECK -> fold. CDK CPU pod only. FOLDFIX-BUILD (helper of L4-E).

Harness: tests/q38_l4_foldfix/foldfix_harness.py (it reads the runner statements from this tree's tpu_runner.py; no
src-hash or line pins). Every comparison is bitwise against the flag-off base path (no DRAFT-ASM, no VFY-ONE) of the
same step through the same manager and runner code: drafts, the parked source and rejected count, both index dicts,
the placeholder advance and the next step's substituted input ids.

S0  the runner segment is this tree's: compiled from tpu_runner.py's sample_tokens, holding every consumer call.
F1  STACK-D v5 F1 (DRAFT-ASM family off, VFY-ONE value 2; L1 #5's render): VFY_ONE_CHECK absent (code default 0) over
    300 serving steps, and VFY_ONE_CHECK=256 (test-only) over 448 (VFY armed at the step-400 agreement): every step
    returns a tensor through the manager, VFY's seed is taken by the base seed block on every "vfy" step, bitwise base.
    Expected PASS on an unfixed tree too (with FOLD=0 the fold return is unreachable).
X   CHECK -> fold through the real manager (needs the FOLDFIX manager change; an unfixed tree fails at the first folded
    step with the manager's `assert isinstance(result, torch.Tensor | list)`):
      asm              DRAFT_ASM_FOLD=1, CHECK_STEPS=256, 448 steps (T81r's DRAFT-ASM arm)
      asm_vfy2         + VPS_VFY_ONE=2, VFY_ONE_CHECK=256 (T81r's DRAFT-ASM + VFY-ONE arm)
      dhd              DRAFT_ASM_FOLD=1, DRAFT_ASM_ONE=1 (DH-D), CHECK_STEPS=256
      v5full_check256  v5's whole intended env + CHECK_STEPS=256, VFY_ONE_CHECK=256 (test-only CHECK)
      v5full_car       v5's env exactly as rendered (no CHECK keys: fold and VFY from the first step), 300 steps
    The CHECK window covers serving steps 1-256, the agreement at engine step 400 arms the fold, so steps 400-448 fold
    (49 steps, at least 8 per row count 8 / 32 / 48 / 64). On every folded step: the manager hands the DraftAsmResult to
    the runner, the folded-source consumer (_assemble_async_spec_substitution -> DraftAsm.assemble_tail) parks its
    source as is (no spec-tail program, the parked tensor IS the fold's source, length max_num_reqs * (1 + K), not
    padded); 0 CHECK mismatches, 0 cold, 0 tail misses; with VFY the seed is taken on every "vfy" step.
R   each "not reachable" site of the consumer trace, asserted: (3) a KV producer with the token-copy fence never folds
    ("kv_fence") and the fence enqueues the drafts tensor; (4) a replicated drafter under TP > 1 never folds ("tp_pin");
    (5) a structured-output step never folds ("structured_output") and stages the eager drafts; (7) a call with no
    verify output never folds ("not_async_verify"); (11) a sync step never folds ("not_async_verify") and caches the
    host list.
M   the manager's hand-over on those steps if a DraftAsmResult ever reaches them: structured output -> the host copy and
    the consumer get result.drafts(); no verify output (prefill bootstrap) -> result.drafts(); bitwise the eager result.
Run: python -m pytest -q -p no:cacheprovider -p no:logging -rfEs -s tests/q38_l4_foldfix/test_foldfix_realpath.py
"""

import os
import pathlib
import sys
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import foldfix_harness as fh  # noqa: E402

from vllm_torchtpu.runner import tpu_runner  # noqa: E402
from vllm_torchtpu.spec_decode import draft_asm, draft_one  # noqa: E402

SEED0 = 90000
N_CHECK = 448  # CHECK steps 1-256; the engine-step-400 agreement arms the fold; steps 400-448 fold
N_NOCHECK = 300
FIRST_ARMED = 400  # the first agreement boundary after a 256-step CHECK window (AGREE_EVERY = 200)
FOLD, ONE, CAP, CL = "VLLM_TPU_DRAFT_ASM_FOLD", "VLLM_TPU_DRAFT_ASM_ONE", "VLLM_TPU_DRAFT_ASM_CAP", "VLLM_TPU_DRAFT_ASM_CARRY_LOOP"
CHECK, VFY, VFY_CHECK = "VLLM_TPU_DRAFT_ASM_CHECK_STEPS", "VLLM_TPU_VPS_VFY_ONE", "VLLM_TPU_VPS_VFY_ONE_CHECK"
BASE_ENV = {FOLD: None, ONE: None, CAP: None, CL: None, CHECK: None, VFY: None, VFY_CHECK: None}
REF: dict = {}
COUNTS: Counter = Counter()


def _ref(monkeypatch, keys):
    """The flag-off base path of each (seed, rows, fallback) through the same manager and runner code (cached)."""
    missing = [k for k in keys if k not in REF]
    if missing:
        R = fh.build(monkeypatch, env=BASE_ENV)
        assert R.drafter._draft_asm is None and R._vfy is None
        fh.warm(R, SEED0 - 100)
        for seed, r, fb in missing:
            REF[(seed, r, fb)] = fh.engine_step(R, fh.hc.make_step(seed, r, fallback=fb), serving=True)
    return [REF[k] for k in keys]


def _serve(R, n, tag):
    """`n` serving steps on the schedule of fh.step_kw; a failing step is named before the exception propagates."""
    out = []
    for i in range(n):
        r, fb = fh.step_kw(i)
        try:
            out.append(((SEED0 + i, r, fb), fh.engine_step(R, fh.hc.make_step(SEED0 + i, r, fallback=fb), serving=True)))
        except Exception as e:  # noqa: BLE001 - re-raised
            print(f"\nFOLDFIX_STEP_FAIL {tag} serving_step={i + 1} rows={r} fallback={int(fb)} "
                  f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}")
            raise
    return out


def test_s0_the_segment_is_this_trees_runner_code():
    fn = fh.segment()
    first, last = fn.lines
    assert fn.__code__.co_filename == tpu_runner.__file__
    assert 0 < first < last
    COUNTS["s0"] += 1
    print(f"\nFOLDFIX_SEGMENT {os.path.realpath(tpu_runner.__file__)} lines {first}-{last} "
          f"({last - first + 1} lines; every consumer call present)")


F1_LEGS = {
    "check_absent": (dict(fh.F1_ENV), 0, N_NOCHECK),
    "check256": (dict(fh.F1_ENV, **{VFY_CHECK: "256"}), 256, N_CHECK),
}


@pytest.mark.parametrize("leg", list(F1_LEGS))
def test_f1_vfy2_with_the_fold_off(monkeypatch, leg):
    env, vchk, n = F1_LEGS[leg]
    R = fh.build(monkeypatch, env=env)
    vfy = R._vfy
    assert R.drafter._draft_asm is None, "F1 builds no DRAFT-ASM controller"
    assert vfy is not None and vfy.mode == 2 and vfy.check_steps == vchk, (vfy and (vfy.mode, vfy.check_steps))
    fh.warm(R, SEED0 - 100)
    recs = _serve(R, n, f"F1 {leg}")
    refs = _ref(monkeypatch, [k for k, _ in recs])
    vsteps = 0
    for i, ((key, rec), ref) in enumerate(zip(recs, refs)):
        assert rec.result_type == "Tensor", (i, rec.result_type)
        assert rec.calls["spec_tail"] == 1 and not rec.stash_left, (i, rec.calls)
        if rec.vmode == "vfy":
            vsteps += 1
            assert rec.stashed and rec.seed_taken == 1, (i, rec.stashed, rec.seed_taken)
        fh.same_rec(rec, ref, f"F1 {leg} step {i + 1}")
    want_v = n if vchk == 0 else n - FIRST_ARMED + 1
    s = vfy.stats
    assert vfy.seed_reason == "" and vsteps == want_v and s["vfy"] == want_v and s["check"] == vchk, (vsteps, dict(s))
    assert s["seed_taken"] == want_v and s["seed_unused"] == 0 and s["seed_miss"] == 0 and s["seed_written"] == 0, dict(s)
    COUNTS["f1_legs"] += 1
    print(f"\nFOLDFIX_F1 leg={leg} VFY_ONE=2 VFY_ONE_CHECK={'absent(default 0)' if vchk == 0 else vchk} FOLD=0 "
          f"serving_steps={n} past_check={n - vchk} vfy_steps={vsteps} check_steps={s['check']} "
          f"seed_taken={s['seed_taken']} seed_unused=0 manager->tensor {n}/{n} bitwise_base={n}/{n} PASS")


# name -> (env, fold controller, CHECK steps, VFY CHECK steps or None (VFY off), serving steps)
FOLD_CFGS = {
    "asm": (dict(BASE_ENV, **{FOLD: "1", CHECK: "256"}), "DraftAsm", 256, None, N_CHECK),
    "asm_vfy2": (dict(BASE_ENV, **{FOLD: "1", CHECK: "256", VFY: "2", VFY_CHECK: "256"}), "DraftAsm", 256, 256, N_CHECK),
    "dhd": (dict(BASE_ENV, **{FOLD: "1", ONE: "1", CHECK: "256"}), "DraftOne", 256, None, N_CHECK),
    "v5full_check256": (dict(fh.V5_ENV, **{CHECK: "256", VFY_CHECK: "256"}), "DraftOne", 256, 256, N_CHECK),
    "v5full_car": (dict(fh.V5_ENV), "DraftOne", 0, 0, N_NOCHECK),
}


@pytest.mark.parametrize("cfg", list(FOLD_CFGS))
def test_fold_check_to_fold_through_the_manager(monkeypatch, cfg):
    env, kind, chk, vchk, n = FOLD_CFGS[cfg]
    R = fh.build(monkeypatch, env=env)
    ctl, vfy = R.drafter._draft_asm, R._vfy
    assert type(ctl).__name__ == kind and ctl.check_steps == chk, (type(ctl).__name__, ctl.check_steps)
    assert (vfy is None) == (vchk is None) and (vfy is None or vfy.check_steps == vchk)
    if cfg.startswith("v5full"):
        assert ctl.one_conf and ctl.one_cap, "v5: DH-D with DRAFT-CAP"
    fh.warm(R, SEED0 - 100)
    recs = _serve(R, n, f"X {cfg}")
    refs = _ref(monkeypatch, [k for k, _ in recs])
    first = FIRST_ARMED if chk else 1
    per_rows: Counter = Counter()
    vsteps = 0
    for i, (((seed, r, fb), rec), ref) in enumerate(zip(recs, refs)):
        tag = f"X {cfg} step {i + 1}"
        assert not rec.stash_left, tag
        if rec.ctl_folded:
            per_rows[r] += 1
            assert i + 1 >= first, (tag, "folded before the CHECK window closed")
            assert rec.result_type == "DraftAsmResult", (tag, rec.result_type)
            # the folded-source consumer: no spec-tail program; the parked source IS the fold's, unpadded
            assert rec.calls["spec_tail"] == 0, (tag, rec.calls)
            assert rec.src is rec.drafts_obj.src, (tag, "the parked source is not the fold's (re-made or padded)")
            assert tuple(rec.src.shape) == (fh.BOUND,), (tag, tuple(rec.src.shape))
        else:
            assert i + 1 < first, (tag, "a step after the CHECK window did not fold", dict(ctl.reasons))
            assert rec.result_type == "Tensor" and rec.calls["spec_tail"] == 1, (tag, rec.result_type, rec.calls)
        if rec.vmode == "vfy":
            vsteps += 1
            assert rec.stashed and rec.seed_taken == 1, (tag, rec.stashed, rec.seed_taken)
        fh.same_rec(rec, ref, tag)
    c = ctl.counts()
    nfold = n - first + 1
    assert c["folded"] == nfold and sum(per_rows.values()) == nfold, (c, dict(per_rows))
    assert all(per_rows[r] >= 8 for r in fh.ROWS), dict(per_rows)
    assert c["checked"] == min(chk, n) and c["mismatches"] == 0 and c["cold"] == 0 and c["tail_miss"] == 0, c
    assert c["failed"] == 0 and not ctl.failed, c
    if kind == "DraftOne":
        assert c["folded_one"] == nfold, c
    if vfy is not None:
        s = vfy.stats
        want_v = n if not vchk else n - FIRST_ARMED + 1
        assert vsteps == want_v and s["seed_taken"] == want_v and s["seed_unused"] == 0 and s["seed_miss"] == 0, \
            (vsteps, dict(s))
    COUNTS["fold_cfgs"] += 1
    COUNTS["folded_steps"] += nfold
    print(f"\nFOLDFIX_FOLD cfg={cfg} controller={kind} CHECK={chk} VFY_ONE={'2 CHECK=%s' % vchk if vfy else 'off'} "
          f"serving_steps={n} checked={c['checked']} first_fold={first} folded={nfold} "
          f"per_rows={dict(sorted(per_rows.items()))} vfy_steps={vsteps} manager->DraftAsmResult {nfold}/{nfold} "
          f"spec_tail_on_folded=0 src_len={fh.BOUND} mismatches=0 cold=0 tail_miss=0 bitwise_base={n}/{n} PASS")


ARMED = dict(BASE_ENV, **{FOLD: "1"})  # DRAFT-ASM with no CHECK: armed at the first serving step's agreement


def _arm(R):
    """One serving engine-step tick: DRAFT-ASM agrees and arms (no propose)."""
    R.warmup_done = True
    R._l4_agree.tick(R)
    assert R.drafter._draft_asm.agreed


def test_reach3_a_fenced_kv_producer_never_folds(monkeypatch):
    keys = [(SEED0 + i, *fh.step_kw(i)) for i in range(4)]
    refs = _ref(monkeypatch, keys)
    monkeypatch.setattr(tpu_runner.dist_utils, "get_stage3_register_ahead_enabled", lambda: True)
    synced, fenced = [], []
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda *a, **k: synced.append((a, k)))
    R = fh.build(monkeypatch, env=ARMED, fence=True)
    ctl = R.drafter._draft_asm
    assert R._fence_spec_kv_under_token_copy is True and ctl._static() == "kv_fence"
    fh.warm(R, SEED0 - 100)
    cs = SimpleNamespace(fence_pending_device_work=lambda: fenced.append(1))
    for j, ((seed, r, fb), ref) in enumerate(zip(keys, refs)):
        n0 = len(synced)
        rec = fh.engine_step(R, fh.hc.make_step(seed, r, fallback=fb), serving=True, copy_state=cs)
        assert rec.ctl_folded == 0 and rec.result_type == "Tensor", (j, rec.result_type)
        got = synced[n0:]
        assert len(got) == 1 and got[0][0][0][0] is rec.drafts_obj and got[0][1] == {"wait": False}, got
        fh.same_rec(rec, ref, f"R3 step {j}")
    assert len(fenced) == 4 and ctl.reasons["kv_fence"] == 4 and ctl.stats["folded"] == 0, dict(ctl.reasons)
    COUNTS["reach"] += 1
    print("\nFOLDFIX_REACH (3) fence: KV producer with the token-copy fence -> DraftAsm static 'kv_fence', 4/4 serving "
          "steps not folded, the fence enqueued the drafts tensor 4/4; bitwise base")


def test_reach4_a_replicated_drafter_under_tp_never_folds(monkeypatch):
    import vllm.distributed as vd

    monkeypatch.setattr(vd, "get_tensor_model_parallel_world_size", lambda: 2)
    R = fh.build(monkeypatch, env=ARMED)
    p = R.drafter
    p.speculative_config.draft_tensor_parallel_size = 1
    ctl = p._draft_asm
    assert ctl._static() == "tp_pin"
    _arm(R)
    s = fh.hc.make_step(SEED0 + 1, 32)
    ctl.begin([s.chunk], [s.nt], SimpleNamespace(has_structured_output_requests=False), True)
    assert ctl.step is None and ctl.reasons["tp_pin"] == 1, dict(ctl.reasons)
    COUNTS["reach"] += 1
    print("\nFOLDFIX_REACH (4) tp pin: replicated drafter under TP 2 -> DraftAsm static 'tp_pin', the armed controller "
          "refuses the verify step")


def test_reach5_a_structured_output_step_never_folds(monkeypatch):
    keys = [(SEED0 + 40 + i, *fh.step_kw(i)) for i in range(4)]
    refs = _ref(monkeypatch, keys)
    R = fh.build(monkeypatch, env=ARMED)
    ctl = R.drafter._draft_asm
    fh.warm(R, SEED0 - 100)
    structured = [False, True, True, False]
    recs = [fh.engine_step(R, fh.hc.make_step(seed, r, fallback=fb), serving=True, structured=st)
            for (seed, r, fb), st in zip(keys, structured)]
    assert [x.ctl_folded for x in recs] == [1, 0, 0, 1], [x.ctl_folded for x in recs]
    assert ctl.reasons["structured_output"] == 2, dict(ctl.reasons)
    for j, (rec, ref, st) in enumerate(zip(recs, refs, structured)):
        if st:
            assert rec.result_type == "Tensor", rec.result_type
            assert rec.host is not None and rec.host.req_ids == [f"r{i}" for i in range(rec.n)]
            assert rec.host.draft_token_ids == ref.drafts.tolist(), f"R5 step {j} staged host drafts"
        fh.same_rec(rec, ref, f"R5 step {j}")
    COUNTS["reach"] += 1
    print("\nFOLDFIX_REACH (5) structured output: armed DraftAsm refuses 2/2 structured steps ('structured_output'), the "
          "host copy stages the eager drafts; the steps around them fold; bitwise base")


def test_reach7_a_call_without_verify_output_never_folds(monkeypatch):
    R = fh.build(monkeypatch, env=ARMED)
    ctl = R.drafter._draft_asm
    _arm(R)
    s = fh.hc.make_step(SEED0 + 2, 48)
    n0 = ctl.reasons["not_async_verify"]
    ctl.begin([s.chunk], None, SimpleNamespace(has_structured_output_requests=False), True)
    assert ctl.step is None and ctl.reasons["not_async_verify"] == n0 + 1, dict(ctl.reasons)
    assert ctl.result([s.chunk], [[]], [None], True) is None
    COUNTS["reach"] += 1
    print("\nFOLDFIX_REACH (7) no verify output (the runner's prefill / non-spec device_seed call): the armed controller "
          "refuses it ('not_async_verify') and result() is None")


def test_reach11_a_sync_step_never_folds(monkeypatch):
    keys = [(SEED0 + 60 + i, *fh.step_kw(i)) for i in range(3)]
    refs = _ref(monkeypatch, keys)
    R = fh.build(monkeypatch, env=ARMED)
    ctl = R.drafter._draft_asm
    fh.warm(R, SEED0 - 100)
    sync = [False, True, False]
    n0 = ctl.reasons["not_async_verify"]
    recs = [fh.engine_step(R, fh.hc.make_step(seed, r, fallback=fb), serving=True, sync=sy)
            for (seed, r, fb), sy in zip(keys, sync)]
    assert [x.ctl_folded for x in recs] == [1, 0, 1], [x.ctl_folded for x in recs]
    assert ctl.reasons["not_async_verify"] == n0 + 1, dict(ctl.reasons)
    b = recs[1]
    assert b.drafts_obj is None and b.host is not None and b.host.draft_token_ids == refs[1].drafts.tolist()
    fh.same_rec(recs[0], refs[0], "R11 step 0")
    fh.same_rec(recs[2], refs[2], "R11 step 2")
    COUNTS["reach"] += 1
    print("\nFOLDFIX_REACH (11) sync: the armed controller refuses the sync step ('not_async_verify'), the manager caches "
          "the host list (= the eager drafts); the async steps around it fold; bitwise base")


def _real_fold_result(monkeypatch, seed, r):
    R = fh.build(monkeypatch, env=ARMED)
    fh.warm(R, SEED0 - 100)
    rec = fh.engine_step(R, fh.hc.make_step(seed, r), serving=True)
    assert rec.ctl_folded == 1 and isinstance(rec.drafts_obj, draft_asm.DraftAsmResult), rec.result_type
    return rec.drafts_obj


def test_m_forced_fold_result_on_a_structured_step_hands_over_eager_drafts(monkeypatch):
    seed, r = SEED0 + 80, 32
    (ref,) = _ref(monkeypatch, [(seed, r, False)])
    res = _real_fold_result(monkeypatch, seed, r)
    fp = fh.forced_proposer(res)
    R = fh.build(monkeypatch, env=BASE_ENV, proposer=fp)
    rec = fh.engine_step(R, fh.hc.make_step(seed, r), serving=True, structured=True)
    assert fp.seen and fp.seen[-1]["next_tokens_per_chunk"] is not None
    assert rec.result_type == "Tensor", rec.result_type
    fh.same(rec.drafts, res.drafts(), "M structured drafts")
    assert rec.host.draft_token_ids == res.drafts().tolist() == ref.drafts.tolist()
    fh.same_rec(rec, ref, "M structured")
    COUNTS["forced"] += 1
    print("\nFOLDFIX_FORCED structured: a DraftAsmResult on a structured-output verify step -> the manager hands over "
          "result.drafts(); host copy and consumer bitwise the eager path")


def test_m_forced_fold_result_without_verify_output_hands_over_eager_drafts(monkeypatch):
    seed, r = SEED0 + 90, 48
    res = _real_fold_result(monkeypatch, seed, r)
    fp = fh.forced_proposer(res)
    R = fh.build(monkeypatch, env=BASE_ENV, proposer=fp)
    rec = fh.engine_step(R, fh.hc.make_step(seed, r), serving=True, spec=False)
    kw = fp.seen[-1]
    assert kw["next_tokens_per_chunk"] is None and kw["device_seed"] is not None, kw
    assert rec.result_type == "Tensor", rec.result_type
    fe = fh.forced_proposer(res.drafts())
    Re = fh.build(monkeypatch, env=BASE_ENV, proposer=fe)
    ref = fh.engine_step(Re, fh.hc.make_step(seed, r), serving=True, spec=False)
    assert ref.result_type == "Tensor"
    fh.same_rec(rec, ref, "M prefill bootstrap")
    COUNTS["forced"] += 1
    print("\nFOLDFIX_FORCED prefill bootstrap: a DraftAsmResult on the non-spec device_seed call -> the manager hands "
          "over result.drafts(); the bootstrap source, indices and next-step ids bitwise the eager result's")


def test_zz_counts():
    print("\nFOLDFIX_COUNTS " + " ".join(f"{k}={v}" for k, v in sorted(COUNTS.items())))
