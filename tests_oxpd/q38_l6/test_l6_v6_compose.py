# SPDX-License-Identifier: Apache-2.0
"""q38 L6-V6 compose (q38-l6-v6): tests of the places where the composed pieces meet. Host-only: runs in a CDK
CPU pod, like every L6 harness.

1. CARRY-LEAN x K-step U2 (brief V6_COMPOSE item 3; scratch/kstep/U2_review_L6.md item 5; the coordination
   addendum of scratch/overflow/l6_h2_lean.md). U2's ``carry_rerecord`` (q38-kstep-io e5027f502,
   src/vllm_torchtpu/kstep/accounting.py:140-167, copied below byte for byte and pinned by sha256) shifts CARRY's
   record fields ``S`` / ``pos`` / ``seq_lens`` / ``positions`` by (K_sched - 1)(1 + k) after a burst. Applied to
   a record that lever ``clean`` wrote in place, the next head must carry exactly as it does on a stock record:
   the same CARRY decisions, outputs and carry state as STEADY + CARRY alone, at every step, and the head's
   sequence lengths and positions equal to the upload path's (every L6 lever unset).
2. PREP-SHORT x ADC (scratch/overflow/l6_h2_lean_status.md section 8 caveat 5). The short path builds lever
   adcarry's merge context as the full path does and passes it to ``_postfwd_fused_prepare``, so a carried short
   step runs the merged carry program exactly where the full path would: the same digests (every device input of
   the forward, the real Qwen3.8 adapter's included), the same uploads, and the merged carry program on the same
   steps, with and without the short path.
3. PREP-SHORT x L6-INSTR: the short path's copy of the full path's tail emits the tail's sub-marks
   (``~pi_subst2`` / ``~pi_cross``) in the same order as the full path.
4. ADMIT-INC x P-METER's admission census: every staged block-table upload in layers/core/attention_metadata.py,
   ADMIT-INC's count-change pack included, sits between ``~am_pack`` and ``~am_h2d``.
5. NODEFER's VfyCall wraps the two VPS verify compiled calls whether the census is on or off: with it off (every L6
   lever unset, or l6_ab active through another lever) a wrapped call returns the unwrapped call's very outputs, passes
   every positional and keyword argument through unchanged, leaves the arguments' bytes alone and counts nothing.
6. FENCE-SKIP x PREP-SHORT: the short path sets fskip's carried mark as the full path does on a carried chunk, so with
   fskip on the fence is skipped on exactly the steps the full path skips it, and never after an upload.
7. TOKEN-LISTS-ONCE with its flag off: get_output's result is byte-identical to the stock path's, the base tree's
   (b488e1cb0) own get_output, loaded from Q38_L6_BASE3_SRC.
"""

from __future__ import annotations

import ast
import hashlib
import os
import pathlib
from types import SimpleNamespace as NS
from typing import Any

import pytest
import torch

import l6_adc_harness as A
import l6_carry_harness as H
import l6_pshort_harness as S
from vllm_torchtpu.core import l6_ab
from vllm_torchtpu.runner import host_phases, l6_carry_lean, l6_prep_short, l6_ptrims, l6_steady

ENV_KEYS = (
    "VLLM_TPU_L6_CARRY_LEAN",
    "VLLM_TPU_L6_CARRY_LEAN_CHECK",
    "VLLM_TPU_L6_PREP_SHORT",
    "VLLM_TPU_L6_PREP_SHORT_CHECK",
    "VLLM_TPU_L6_PREP_TRIMS",
    "VLLM_TPU_L6_PREP_TRIMS_CHECK",
    "VLLM_TPU_L6_ADAPT_CARRY",
    "VLLM_TPU_L6_ADAPT_CARRY_CHECK",
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
ADC_REF = dict(BASE, VLLM_TPU_L6_ADAPT_CARRY="1")
ADC_SHORT = dict(ADC_REF, VLLM_TPU_L6_CARRY_LEAN="1", VLLM_TPU_L6_PREP_SHORT="1")

# ---------------------------------------------------------------- U2's carry_rerecord, byte for byte

U2_SRC = '''def _shift(t: torch.Tensor, idx: torch.Tensor, delta: int) -> torch.Tensor:
    """``t + delta`` on the slots ``idx >= 0`` marks real (the carry program's own add form)."""
    zero = t.new_zeros(())
    return t + torch.where(idx >= 0, zero + delta, zero)


def carry_rerecord(runner: Any, k_sched: int, stride: int) -> bool:
    """Shift this iteration's CARRY record by (K_sched - 1)(1 + k) (module docstring). Returns True if
    shifted. A record from another iteration is left alone (the next head refuses it as stale);
    a record that cannot be shifted is dropped (the next head uploads)."""
    st = runner.__dict__.get("_l6_carry_state")
    if st is None or k_sched < 2:
        return False
    from vllm_torchtpu.core import l6_ab

    if st.step != l6_ab.step_seq():
        return False
    if st.step_delta != stride or st.seq_idx_dev is None or st.pos_idx_dev is None:
        runner.__dict__["_l6_carry_state"] = None
        note(CARRY_CLEAR)
        return False
    delta = (k_sched - 1) * stride
    st.S = st.S + delta
    st.pos = st.pos + delta
    st.seq_lens = _shift(st.seq_lens, st.seq_idx_dev, delta)
    st.positions = _shift(st.positions, st.pos_idx_dev, delta)
    note(CARRY_SHIFT)
    return True
'''
# sha256 of q38-kstep-io e5027f502:src/vllm_torchtpu/kstep/accounting.py lines 140-167 (the two functions above).
U2_SHA = "4c971ea6924dbe7b"
U2_NOTES: list[str] = []
U2 = {"torch": torch, "Any": Any, "CARRY_SHIFT": "kstep.io_carry_shift", "CARRY_CLEAR": "kstep.io_carry_clear",
      "note": lambda name, n=1: U2_NOTES.append(name)}
exec(compile(U2_SRC, "kstep_accounting_u2_copy", "exec"), U2)  # noqa: S102 (a pinned test copy)


def test_u2_copy_is_byte_for_byte():
    assert hashlib.sha256(U2_SRC.encode()).hexdigest()[:16] == U2_SHA


# ---------------------------------------------------------------- fixture


@pytest.fixture
def env(monkeypatch):
    for k in ENV_KEYS + A.ADAPTER_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VLLM_TPU_POSTFWD_FUSED", "1")
    undo = A.install_eager_programs()
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

    def lever(arm=None):
        for k in ENV_KEYS:
            monkeypatch.delenv(k, raising=False)
        for k, v in dict(arm or {}).items():
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


def per_step(lnotes, lever, key) -> set[int]:
    """The l6_ab steps on which `lever.key` was noted."""
    return {s for s, lv, k in lnotes if lv == lever and k == key}


# ---------------------------------------------------------------- 1. CARRY-LEAN x U2 carry_rerecord

AT = 12  # the burst ends before sim step AT (its head); the record U2 shifts is sim step AT - 1's
K_SCHED = 2


@pytest.mark.parametrize("name", ["s04_served", "s11_served"])
def test_lean_record_shifted_by_u2_carry_rerecord_carries_at_the_next_head(env, name):
    """A burst of K_SCHED iterations ends after sim step AT - 1: U2 shifts that step's CARRY record by
    (K_SCHED - 1)(1 + k) and every request's count advances by the same amount. With lever clean (and with
    clean + pshort) the shifted record is the one clean wrote in place, and the head carries, leaving the same
    outputs, mirrors and carry state as STEADY + CARRY alone at every step; the carried head's sequence lengths
    and positions equal the upload path's."""
    shape = S.SHAPES[name]
    delta = (K_SCHED - 1) * H.STRIDE
    shifted: dict[str, list] = {}

    def burst(arm):
        def hook(sim):
            if sim.n == AT:
                shifted[arm].append(U2["carry_rerecord"](sim.r, K_SCHED, H.STRIDE))
                for rid in list(sim.computed):
                    sim.computed[rid] += delta
        return hook

    runs = {}
    for arm, cfg in (("off", None), ("base", BASE), ("lean", LEAN), ("short", SHORT)):
        shifted[arm] = []
        notes: list = []
        U2_NOTES.clear()
        env.lever(cfg)
        rows, _ = S.run_stream(shape, AT + 8, seed=21, rand=False, on_step=burst(arm), notes=notes)
        runs[arm] = (rows, notes, list(env.lnotes), list(U2_NOTES))
    head_seq = AT + 1  # l6_ab's step sequence is the sim step + 1
    assert shifted["off"] == [False] and runs["off"][3] == []
    base_rows, base_notes, _, base_u2 = runs["base"]
    assert shifted["base"] == [True] and base_u2 == ["kstep.io_carry_shift"]
    assert "carried" in {k for s, k in base_notes if s == AT}, sorted(base_notes)
    for arm in ("lean", "short"):
        rows, notes, ln, u2 = runs[arm]
        assert shifted[arm] == [True] and u2 == ["kstep.io_carry_shift"], (arm, shifted[arm], u2)
        # The record U2 shifted is the one clean updated in place at the step before the head.
        assert head_seq - 1 in per_step(ln, "clean", "inplace"), (arm, sorted(per_step(ln, "clean", "inplace")))
        took = per_step(ln, "clean", "fast") | per_step(ln, "pshort", "short")
        assert head_seq in took and head_seq + 1 in took, (arm, sorted(took))
        if arm == "short":
            assert head_seq in per_step(ln, "pshort", "short"), sorted(per_step(ln, "pshort", "short"))
        assert notes == base_notes, arm
        assert S.first_difference(base_rows, rows, skip=S.SKIPPED_MIRRORS) is None, arm
        assert not env.errors, env.errors
    off_rows = runs["off"][0]
    for arm in ("base", "lean", "short"):
        for i in range(len(off_rows)):
            for key in ("seq_lens", "positions"):
                assert off_rows[i][key] == runs[arm][0][i][key], (arm, i, key)


# ---------------------------------------------------------------- 2. PREP-SHORT x ADC

ADC_SHAPES = {
    "b3_compact": dict(nreq=3, max_num_reqs=8, paddings=(16, 32, 64), mamba=True, comb=True),
    "b5_pooled": dict(nreq=5, max_num_reqs=8, paddings=(16, 32, 64), mamba=False, comb=True, pooled=True),
}
ADC_STEPS = 30


def adc_steady_run(shape, steps, *, seed):
    """A.run with execute_model's STEADY reorder verdict before each _prepare_inputs (as l6_pshort_harness
    does; PREP-SHORT engages only on STEADY's ro_fast verdict). Steady decode, no events; warm-up ends before
    step 2. Returns (digests, sim)."""
    keys = A.prewarm_keys(shape, seed)
    sim = A._sim(shape, seed, shape.get("nreq", 3))
    r = sim.r
    r._reorder_batch_for_rpa = lambda so: (0, r.input_batch.num_reqs)
    r._count_input_chunks = lambda so: (1, r.input_batch.num_reqs)
    orig = r._prepare_inputs

    def prep(so, start, nd, nw):
        so.scheduled_new_reqs = []
        so.finished_req_ids = set()
        so.preempted_req_ids = set()
        so.scheduled_cached_reqs = NS(resumed_req_ids=set())
        if l6_ab.ACTIVE and not l6_steady.on(r):
            l6_steady.drop(r)
        if l6_ab.ACTIVE and l6_steady.on(r):
            nd, nw, _, _ = l6_steady.reorder_and_count(r, so)
        return orig(so, start, nd, nw)

    r._prepare_inputs = prep
    digests = []
    for _ in range(steps):
        if sim.n == 2:
            r.__dict__.setdefault("_postfwd_warm_keys", set()).update(keys)
            A.warm(sim)
        digests.append(sim.step())
    return digests, sim


@pytest.mark.parametrize("name", sorted(ADC_SHAPES))
def test_short_path_passes_the_adc_merge_context(env, name):
    """STEADY + CARRY + adcarry with and without clean + pshort: identical digests (every device input of the
    forward and the adapter's records) and uploads at every step; the short path engages on most steps, and the
    merged carry program runs on exactly the same steps in both runs (so on every short step)."""
    shape = ADC_SHAPES[name]
    for k, v in A.adapter_env().items():
        env.monkeypatch.setenv(k, v)
    env.lever(ADC_REF)
    ref, rsim = adc_steady_run(shape, ADC_STEPS, seed=7)
    rn = list(env.lnotes)
    env.lever(ADC_SHORT)
    head, hsim = adc_steady_run(shape, ADC_STEPS, seed=7)
    hn = list(env.lnotes)
    assert A.first_difference(ref, head) is None
    assert rsim.uploads == hsim.uploads, (rsim.uploads, hsim.uploads)
    short = per_step(hn, "pshort", "short")
    merged_ref = per_step(rn, "adcarry", "merged_carry")
    merged_head = per_step(hn, "adcarry", "merged_carry")
    # Warm-up ends before sim step 2 (l6_ab step 3): before it adcarry refuses (why_warmup) in both runs, on the
    # short path as on the full path; from it on every short step runs the merged carry program.
    warm_seq = 3
    assert per_step(hn, "adcarry", "why_warmup") == per_step(rn, "adcarry", "why_warmup") != set()
    short_warm = {s for s in short if s >= warm_seq}
    assert len(short_warm) >= ADC_STEPS // 2, sorted(short)
    assert len(merged_ref) >= ADC_STEPS // 2, sorted(merged_ref)
    assert merged_head == merged_ref, (sorted(merged_ref), sorted(merged_head))
    assert short_warm <= merged_head, sorted(short_warm - merged_head)
    assert not per_step(hn, "pshort", "mismatch") and not env.errors, env.errors
    print(f"ADC_SHORT {name} steps={ADC_STEPS} short={len(short)} merged_carry={len(merged_head)} "
          f"uploads={sum(hsim.uploads)}")


# ---------------------------------------------------------------- 3. PREP-SHORT x L6-INSTR tail sub-marks


def test_short_path_tail_emits_the_full_paths_sub_marks(env):
    """From the prep_attention_metadata mark to the return, a short step emits the full path's sub-marks, in
    order (L6-INSTR's ~pi_subst2 / ~pi_cross), on every step of a steady stream."""
    seen: list = []
    env.monkeypatch.setattr(host_phases, "sub", lambda name, *a, **k: seen.append(("sub", name)))
    env.monkeypatch.setattr(host_phases, "mark", lambda name, *a, **k: seen.append(("mark", name)))

    def tails(arm):
        per: list = []

        def hook(sim):
            per.append(len(seen))

        env.lever(arm)
        seen.clear()
        S.run_stream(S.SHAPES["s04_served"], 16, seed=3, rand=False, on_step=hook)
        per.append(len(seen))
        out = []
        for a, b in zip(per, per[1:]):
            step = seen[a:b]
            i = max(j for j, x in enumerate(step) if x == ("mark", "prep_attention_metadata"))
            out.append(step[i:])
        return out, list(env.lnotes)

    base, _ = tails(BASE)
    short, ln = tails(SHORT)
    took = per_step(ln, "pshort", "short")
    assert len(took) >= 12, sorted(took)
    want = [("mark", "prep_attention_metadata"), ("sub", "pi_subst2"), ("sub", "pi_cross")]
    assert all(t[:3] == want for t in base), base
    assert short == base, (short, base)


def test_import_resolves_to_the_tree_under_test():
    """Guard against a staging that silently imports the pod's installed package."""
    import vllm_torchtpu

    want = os.environ.get("Q38_L6_EXPECT_TREE")
    if not want:
        pytest.skip("Q38_L6_EXPECT_TREE not set (the CPU harness sets it)")
    got = os.path.realpath(vllm_torchtpu.__file__)
    assert got.startswith(os.path.realpath(want) + os.sep), (got, want)


# ---------------------------------------------------------------- 4. ADMIT-INC x the P-METER admission census


def test_every_staged_upload_is_bracketed_by_the_admission_census():
    """Each _inc_upload call (the full pack's, the incremental pack's and ADMIT-INC's count-change pack's) is the
    statement right after host_phases.sub("am_pack") and right before host_phases.sub("am_h2d")."""
    import vllm_torchtpu

    p = pathlib.Path(vllm_torchtpu.__file__).resolve().parent / "layers" / "core" / "attention_metadata.py"
    tree = ast.parse(p.read_text())
    sites = []
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        for node in ast.walk(fn):
            for field in ("body", "orelse", "finalbody"):
                body = getattr(node, field, None)
                if not isinstance(body, list):
                    continue
                for i, st in enumerate(body):
                    if isinstance(st, ast.Assign) and "_inc_upload(" in ast.unparse(st.value):
                        prev = ast.unparse(body[i - 1]) if i > 0 else ""
                        nxt = ast.unparse(body[i + 1]) if i + 1 < len(body) else ""
                        sites.append((fn.name, prev, nxt))
    assert len(sites) == 3, sites
    for fn, prev, nxt in sites:
        assert prev == "host_phases.sub('am_pack')" and nxt == "host_phases.sub('am_h2d')", (fn, prev, nxt)
    assert sorted(fn for fn, _, _ in sites) == ["_pack_admit", "_pack_and_upload", "_pack_incremental"], sites


# ---------------------------------------------------------------- 5. NODEFER's wrapper with the census off


@pytest.fixture
def nd_off(monkeypatch):
    from vllm_torchtpu.runner import l6_nodefer

    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(l6_ab.logger, "info", lambda *a, **k: None)
    monkeypatch.setattr(l6_ab, "_now_ns", lambda: 10**18)
    yield monkeypatch
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_"):
            monkeypatch.delenv(k, raising=False)
    l6_ab._resolve()
    l6_nodefer._resolve()


@pytest.mark.parametrize("which", ["_vps_verify_compiled", "_vps_verify_fb_compiled"])
@pytest.mark.parametrize("env", [{}, {"VLLM_TPU_L6_PMETER": "1"}], ids=["all-unset", "other-lever-on"])
def test_nodefer_wrapper_is_byte_identical_with_the_census_off(nd_off, which, env):
    """The runner's wrapped VPS verify call (census off) against the same compiled callable unwrapped, on the same
    arguments: byte-identical outputs (the very objects), every argument passed through by identity, the arguments'
    bytes unchanged, and no nodefer count."""
    import vllm_torchtpu.runner.tpu_runner as rm
    from vllm_torchtpu.runner import l6_nodefer

    for k, v in env.items():
        nd_off.setenv(k, v)
    l6_ab._resolve()
    l6_nodefer._resolve()
    assert not l6_ab.on("nodefer")
    if l6_ab.ACTIVE:
        l6_ab.begin_step()
    wrapped = getattr(rm, which)
    assert isinstance(wrapped, l6_nodefer.VfyCall)

    def compiled(*args, **kwargs):  # an input-dependent stand-in for the compiled program
        acc = sum(float(a.double().sum()) for a in args if isinstance(a, torch.Tensor))
        kw = float(sum(len(k) * (i + 1) for i, k in enumerate(sorted(kwargs))))
        return torch.tensor([acc, float(len(args)), kw], dtype=torch.float64), tuple(args), dict(kwargs)

    nd_off.setattr(wrapped, "fn", compiled)
    g = torch.Generator().manual_seed(5)
    args = (torch.randn(4, 8, generator=g), 3, torch.arange(12), torch.rand(4, generator=g), None)
    kwargs = {"fallback_rows": 2, "tag": "vps"}
    before = [a.clone() if isinstance(a, torch.Tensor) else a for a in args]
    got = wrapped(*args, **kwargs)
    ref = compiled(*args, **kwargs)
    assert got[0].numpy().tobytes() == ref[0].numpy().tobytes()
    assert len(got[1]) == len(args) and all(x is y for x, y in zip(got[1], args))
    assert got[2] == kwargs
    assert all((torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b) for a, b in zip(args, before))
    assert not {k for k in l6_ab._state.get("counts", {}) if k.startswith("nodefer.")}


# ---------------------------------------------------------------- 6. FENCE-SKIP x PREP-SHORT

FS = {"VLLM_TPU_L6_FENCE_SKIP": "1"}


class _Ev:
    records = 0

    def record(self):
        _Ev.records += 1

    def synchronize(self):
        pass


class _H2D:
    """Counts every Tensor.to(<torch.device>) and every copy_(..., non_blocking=True) (fskip's own harness counts
    the same two kinds), around the whole run."""

    def __init__(self):
        self.n = 0

    def __enter__(self):
        self._saved = {k: torch.Tensor.__dict__.get(k) for k in ("to", "copy_")}
        me, to0, copy0 = self, torch.Tensor.to, torch.Tensor.copy_

        def to(t, *a, **k):
            dev = a[0] if a else k.get("device")
            if isinstance(dev, torch.device):
                me.n += 1
            return to0(t, *a, **k)

        def copy_(t, src, *a, **k):
            if k.get("non_blocking") or (a and a[0] is True):
                me.n += 1
            return copy0(t, src, *a, **k)

        torch.Tensor.to, torch.Tensor.copy_ = to, copy_
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            if v is None:
                delattr(torch.Tensor, k)
            else:
                setattr(torch.Tensor, k, v)
        return False


@pytest.mark.parametrize("name", ["s04_served", "s11_served"])
def test_fence_skip_on_short_steps_is_the_full_paths(env, name):
    """STEADY + CARRY + fskip with and without clean + pshort, the real _record_input_staging_fence after every
    _prepare_inputs: the same fence decisions at every step, no skipped fence with an upload since the previous
    fence point, fences skipped on short steps, and the same outputs."""
    from vllm_torchtpu.runner import l6_fskip
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    env.monkeypatch.setattr(torch, "tpu", NS(Event=_Ev), raising=False)
    runs = {}
    for arm, cfg in (("full", dict(BASE, **FS)), ("short", dict(SHORT, **FS))):
        env.lever(cfg)
        l6_fskip.S.n, l6_fskip.S.carried = 0, False
        fences, last = [], [0]
        g = _H2D()

        def around(sim, call, so, start, nd, nw, g=g, fences=fences, last=last):
            out = call(so, start, nd, nw)
            since = g.n - last[0]
            rec0 = _Ev.records
            TPUModelRunner._record_input_staging_fence(sim.r)
            fences.append((sim.n, _Ev.records == rec0, since))
            last[0] = g.n
            return out

        with g:
            rows, _ = S.run_stream(S.SHAPES[name], 24, seed=17, rand=False, around=around)
        runs[arm] = (rows, fences, list(env.lnotes))
    (frows, ffen, _), (srows, sfen, sln) = runs["full"], runs["short"]
    for arm, fen in (("full", ffen), ("short", sfen)):
        bad = [(st, n) for st, skipped, n in fen if skipped and n]
        assert not bad, (arm, bad)
    assert [x[:2] for x in ffen] == [x[:2] for x in sfen], (ffen, sfen)
    short_steps = {s - 1 for s in per_step(sln, "pshort", "short")}  # l6_ab step = sim step + 1
    skipped_short = {st for st, skipped, _ in sfen if skipped} & short_steps
    assert len(skipped_short) >= 5, (sorted(short_steps), sfen)
    assert S.first_difference(frows, srows, skip=S.SKIPPED_MIRRORS) is None
    assert not env.errors, env.errors
    print(f"FSKIP_SHORT {name} fences={len(sfen)} skipped={sum(1 for x in sfen if x[1])} "
          f"skipped_on_short={len(skipped_short)}")


# ---------------------------------------------------------------- 7. TOKEN-LISTS-ONCE off: get_output vs the base


def _base_tao():
    root = os.environ.get("Q38_L6_BASE3_SRC", "")
    path = os.path.join(root, "vllm_torchtpu", "runner", "tpu_runner_async_output.py")
    assert root and os.path.exists(path), f"Q38_L6_BASE3_SRC must point at base3's src (b488e1cb0): {root!r}"
    import importlib.util
    import sys

    name = "q38_l6_v6_base3_tao"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


class _CopyEv:
    def record(self):
        pass

    def synchronize(self):
        pass


def test_token_lists_off_get_output_is_the_stock_paths(env):
    """Every L6 lever unset (lever toklists off): head's AsyncTPUModelRunnerOutput.get_output against base3's own
    (b488e1cb0, before TOKEN-LISTS-ONCE) on the same copy states over random served-shaped rows (1 / 2 / 4
    columns, INVALID padding and stray INVALID entries, 1-3 bucket-padded chunks, discarded rows): the same
    sampled_token_ids, byte for byte (pickled), and head never builds the Python rows."""
    import pickle
    import random

    from vllm_torchtpu.runner import tpu_runner_async_output as tao

    base = _base_tao()
    env.lever(None)
    assert not l6_ab.ACTIVE
    rng = random.Random(23)
    inv = tao.INVALID_TOKEN_ID
    n_cases = 0
    for n in (1, 4, 9, 11, 25):
        for cols in (1, 2, 4):
            for chunks in (1, 2, 3):
                for _ in range(6):
                    t = torch.full((n, cols), inv, dtype=torch.int32)
                    for i in range(n):
                        k = rng.randint(1, cols)
                        t[i, :k] = torch.tensor([rng.randint(0, 151000) for _ in range(k)], dtype=torch.int32)
                        if rng.random() < 0.08:
                            t[i, rng.randrange(cols)] = inv  # a stray INVALID, in any column (1-column rows too)
                    cuts = sorted(rng.sample(range(1, n), min(chunks - 1, n - 1))) if n > 1 else []
                    parts, start = [], 0
                    for c in cuts + [n]:
                        pad = torch.full((rng.randint(0, 4), cols), 7, dtype=torch.int32)
                        parts.append((torch.cat([t[start:c], pad]), c - start))
                        start = c
                    discard = sorted(rng.sample(range(n), rng.randint(0, min(2, n))))
                    outs = []
                    for mod in (tao, base):
                        st = mod.AsyncTPUCopyState(chunks_cpu=[p.clone() for p, _ in parts],
                                                   chunk_real_lens=[k for _, k in parts], copy_ready_event=_CopyEv())
                        mro = NS(sampled_token_ids=None, logprobs=None)
                        got = mod.AsyncTPUModelRunnerOutput(mro, st, list(discard)).get_output()
                        outs.append((pickle.dumps(got.sampled_token_ids), pickle.dumps(got.logprobs), st))
                    assert outs[0][0] == outs[1][0] and outs[0][1] == outs[1][1], (n, cols, chunks, discard)
                    assert outs[0][2].l6_lists is False and outs[0][2]._sampled_token_ids_list is None
                    n_cases += 1
    assert n_cases == 270
