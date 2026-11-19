# SPDX-License-Identifier: Apache-2.0
"""q38 L4 D83 (P3): the mamba read-offset update folded into the VPS verify (VLLM_TPU_VPS_FUSED_READ_OFFSETS).

L1-D83-SERVED (D83S-BUILD): ported from q38-l4-tail 0e475c3dc onto q38-l4-d83s (D83 rebased onto q38-l4-stack 11b0c44ef).
"base" below is 11b0c44ef (tests/l4tail/ref_11b0c44ef_vps.py.txt), not ee289379f; items 8 and 9 are new.

Torch CPU, no TPU and no vLLM install needed: the runner's own functions and methods are extracted from tpu_runner.py
with ast into the harness of tests/q38mm_dstock/test_vps_wiring.py (stub packages, TP1 VPS head, compiled bodies run
eagerly), as tests/decfb2/test_vps_fb_wiring.py does. Q38MM_SRC_ROOT points the extraction at a mutated copy.

1. Identity: `_sample_spec_verify_vps` with the fold leaves the same tokens, the same random stream, the same VPS
   fallback stats and the same read-offset buffer as the flag-off path followed by the separate program's body
   (`_update_read_offsets_impl`, what `_update_read_offsets_compiled` runs), across request counts, padded rows, slot
   widths (pad, equal, truncate), group counts, the fallback / plain VPS programs, full / candidate draws, fallback
   widths and synthetic / real acceptance; the fold marks the chunk so sample_tokens skips the separate program.
2. The in-program operands (`_read_offset_scatter_operands`) plus one index_put_ equal the separate impl on a random
   grid; dtypes and shapes are the buffer's.
3. Eager ops: outside the verify program the fold adds exactly one eager op, index_put_.
4. CHECK: clean checks count 0, the fault control is caught, a broken fold is seen, nothing runs when off.
5. Gate: the flag is read at init; flag off / no groups / no buffer keep the plain program.
6. Flags off: the VPS programs trace to the same graph as base's (make_fx), the flag-off runner path issues the
   same program calls as base's, and the stock fused-verify fold / separate update sources are unchanged.
7. Warm-up: flag off warms exactly what base warms; flag on warms only the fold variant, at every fallback width
   and every slot width the walk can produce, with distinct group tensors, and the serving call's full signature is
   among the warmed ones; the eager-scatter warm-up writes only zeros at the null slot; the geometry counts mamba
   builders (unified) or one shared slot tensor (compact).
8. EAGER-FIX E-b on the stack (the conflict the rebase resolved): with VLLM_TPU_VPS_WARM_DRAWS=N the warm-up issues the
   same warm draws (same calls, shapes and count) with the fold on as with it off, and the same as base, on the
   fallback and plain paths, served-only included.
9. Program census over a multi-step run through sample_tokens' own post-verify statement and the runner's own
   `_update_mamba_slot_read_offsets`: flag off = one verify + one `_update_read_offsets_compiled` per step; flag on =
   one fold-variant verify and zero separate updates per step; tokens, random stream, read-offset buffer, slot tensors
   and the offsets the next step reads are identical step by step.
Run: python -m pytest -q -p no:cacheprovider tests/l4tail/test_vps_ro_fold.py
"""

import ast
import collections
import json
import logging
import os
import sys
import re
import textwrap
import types

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._python_dispatch import TorchDispatchMode

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "q38mm_dstock"))
import test_vps_wiring as w  # noqa: E402

VP, VW, K, V = w.VP, w.VW, w.K, w.V
RUNNER = w.RUNNER
REF = os.path.join(HERE, "ref_11b0c44ef_vps.py.txt")
RUNNER_SRC = open(RUNNER).read()
COUNTS = collections.Counter()
CASES = set()


def _assign(src, name, ns):
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == name for t in n.targets))
    exec(ast.get_source_segment(src, node), ns)


def _ns():
    ns = dict(w.NS)
    ns["logger"] = logging.getLogger("t-l4tail")
    ns["logger"].warning_once = ns["logger"].warning
    ns["vps_wiring"] = VW
    ns["synchronize_tensors"] = lambda *a, **k: None
    return ns


NS = _ns()
# q38 L4 v5 composition (V5COMP-BUILD; tests only): VPSHOST's l6_ab lever reads (sampsnap, statsasync) in
# `_sample_spec_verify_vps` / `_vps_verify_fb` are off in this harness (tests/q38_l4_vpshost and tests/q38_l4_v5c
# run them on), as tests/decfb2/test_vps_fb_wiring.py does.
NS["_l6_ab"] = types.SimpleNamespace(on=lambda lever: False, note=lambda *a, **k: None)
w._fns(RUNNER, ["_update_read_offsets_impl", "_read_offset_scatter_operands", "_vps_verify_impl",
                "_vps_verify_fb_impl"], NS)
_assign(RUNNER_SRC, "_Q38_RO_FOLD_LOG_EVERY", NS)
NS["_vps_verify_compiled"] = NS["_vps_verify_impl"]
NS["_vps_verify_fb_compiled"] = NS["_vps_verify_fb_impl"]
METHODS = ["_sample_spec_verify_vps", "_vps_verify_fb", "_apply_folded_read_offsets", "_check_folded_read_offsets",
           "_precompile_vps_verify", "_read_offset_fold_geometry", "_vps_ro_fold_warmup_args", "_warm_folded_scatter",
           "_num_reqs_tensor", "_warm_vps_draws", "_log_vps_fallback"]
M = dict(zip(METHODS, w._fns(RUNNER, METHODS, NS, cls="TPUModelRunner")))


class _NoStatsInFlight:
    """q38 L4 T-1(i) (v5 composition, tests only): the runner's AsyncCounts with nothing in flight (statsasync off)."""

    pending = ()

    @staticmethod
    def ready(wait=False):
        return []

# base's functions (flags-off reference), in their own namespace.
BNS = _ns()
exec(open(REF).read(), BNS)
BNS["_vps_verify_compiled"] = BNS["_vps_verify_impl"]
BNS["_vps_verify_fb_compiled"] = BNS["_vps_verify_fb_impl"]


@pytest.fixture(autouse=True, scope="module")
def _harness_modules_installed():
    """test_vps_wiring's stubs (vllm.distributed's all-gather is the TP1 identity) only while these tests run."""
    with w.HARNESS.active():
        yield


class Runner:
    """The runner state the VPS verify, the fold and the warm-up read (as test_vps_fb_wiring's FbRunner)."""

    def __init__(self, B, *, fb=True, fb_max=8, cand=False, fold=False, checks=0, faults=0, synthetic=True,
                 nslots=4096, seed=0):
        self.rejection_sampler = types.SimpleNamespace(synthetic_conditional_rates=torch.tensor([1.0, 0.32, 0.0]),
                                                       synthetic_mode=synthetic)
        self.speculative_config = types.SimpleNamespace(num_speculative_tokens=K)
        self._vps_head = VW.VpsHead(None, None, 0, V, V, 1)
        self._vps_checks_left, self._vps_check_faults, self._vps_checks_done = 0, 0, 0
        self._vps_mismatches = None
        self._vps_fb_path, self._vps_fb_max, self._vps_tiegap, self._vps_cand_draws = fb, fb_max, 0, cand
        self._vps_fb_stats, self._vps_fb_chunks = None, 0
        self._vps_fb_async = _NoStatsInFlight()  # q38 L4 v5 composition: VPSHOST T-1(i)'s state (idle)
        self.device = torch.device("cpu")
        self._num_reqs_device = {}
        self._fused_read_offsets = (None, 0)
        self._fused_read_offsets_applied = False
        self._vps_ro_fold = fold
        self._vps_ro_checks_left, self._vps_ro_check_faults, self._vps_ro_checks_done = checks, faults, 0
        self._vps_ro_mismatches = None
        self._vps_ro_folds = 0
        self._vps_ro_warmed = set()
        g = torch.Generator().manual_seed(7000 + seed)
        buf = torch.randint(0, 9, (nslots,), generator=g, dtype=torch.int32)
        buf[0] = 0
        self.mamba_slot_read_offsets = buf
        self.top_k = torch.full((B,), 20, dtype=torch.int32)
        self.temps = torch.full((B,), 0.6)
        # warm-up geometry (unified pool, 3 mamba builders among 5; walk width 64)
        self._has_mamba_state = False
        mk = lambda m: types.SimpleNamespace(metadata_builders=[types.SimpleNamespace(is_mamba_group=m)])  # noqa: E731
        self.attn_groups = [[mk(True), mk(False)], [mk(True)], [mk(False), mk(True)]]
        self.num_reqs_max_model_len, self.most_model_len, self.num_reqs_most_model_len = 64, None, None
        self.max_num_reqs = 256

    def _padded_sampling_params(self, s, e, n, dtype, device):
        k = torch.where(self.top_k >= V, 0, self.top_k)
        return (self.temps.to(dtype).view(-1, 1)[:n], k.view(-1, 1)[:n].clone(),
                torch.full((n, 1), 0.95, dtype=torch.float32))

    def _sampler_topk_candidates(self, s, e, grammar):
        return 64

    def _upload_unless_unchanged(self, slot, host):
        return host


for _n, _f in M.items():
    setattr(Runner, _n, _f)


def problem(seed, B):
    g = torch.Generator().manual_seed(seed)
    N, T = B * (K + 1), B * K
    logits = torch.randn(N, V, generator=g) * 3
    logits.scatter_add_(1, torch.randint(0, V, (N, 6), generator=g), torch.rand(N, 6, generator=g) * 8)
    logits = logits.bfloat16()
    rows = torch.arange(N).view(B, K + 1)
    am = logits[rows[:, :K].reshape(-1)].float().argmax(-1)
    draft = torch.where(torch.rand(T, generator=g) < 0.6, am, torch.randint(0, V, (T,), generator=g)).to(torch.int32)
    md = types.SimpleNamespace(bonus_logits_indices=rows[:, K].contiguous(),
                               target_logits_indices=rows[:, :K].reshape(-1).contiguous(),
                               draft_token_ids=draft, draft_lengths=torch.full((B,), K, dtype=torch.int32),
                               segment_ids=torch.arange(B).repeat_interleave(K).to(torch.int32),
                               group_indices=torch.arange(K).repeat(B).to(torch.int32))
    return logits, md


def slot_groups(seed, G, W, R, nslots=4096):
    """G distinct-slot int32 tensors of width W; positions >= R hold the null slot 0, as the runner's do."""
    g = torch.Generator().manual_seed(9000 + seed)
    slots = (torch.randperm(nslots - 1, generator=g)[: G * W] + 1).to(torch.int32).view(G, W)
    slots[:, R:] = 0
    return [slots[i].clone() for i in range(G)]


def separate(buf, groups, tokens, R):
    """sample_tokens' separate update on TPU: `_update_read_offsets_compiled`'s body."""
    NS["_update_read_offsets_impl"](buf, list(groups), tokens, torch.tensor(R))


def run_vps(r, logits, md, groups, R, F, seed):
    r._fused_read_offsets = (groups, R)
    r._fused_read_offsets_applied = False
    gen = torch.Generator().manual_seed(1000 + seed)
    out = r._sample_spec_verify_vps(logits, md, 0, R, gen, F)
    applied = r._fused_read_offsets_applied
    if not applied and groups:  # sample_tokens: `if not self._fused_read_offsets_applied:` -> the separate program
        separate(r.mamba_slot_read_offsets, groups, out, R)
    return out, applied, gen.get_state()


# ---------------------------------------------------------------- 1. identity across the step's row counts

# (requests R, padded rows B, slot width W, groups G): the STEP cases 8 / 12 / 32 / 48 / 52 / 96 / 208 rows are
# R = 2 / 3 / 8 / 12 / 13 / 24 / 52 requests at K = 3 (B = 8 / 8 / 8 / 16 / 16 / 32 / 64); W is the walk's
# target_num_reqs (>= B in serving; W < B covers the truncate branch).
GRID = [(2, 8, 64, 3), (3, 8, 8, 1), (8, 8, 4, 2), (12, 16, 64, 2), (13, 16, 16, 1), (24, 32, 64, 3), (52, 64, 64, 4),
        (52, 64, 128, 2), (5, 16, 8, 3)]
MODES = [  # (fallback program, fallback rows F, candidate draws, synthetic acceptance)
    (True, 8, True, True), (True, 0, True, False), (True, 4, False, True), (False, 0, False, True),
    (False, 0, False, False)]


@pytest.mark.parametrize("R,B,W,G", GRID)
@pytest.mark.parametrize("fb,F,cand,synthetic", MODES)
def test_fold_equals_flag_off_plus_separate_program(R, B, W, G, fb, F, cand, synthetic):
    seed = R * 31 + B + W * 7 + G + F + 2 * cand + 3 * synthetic + 5 * fb
    logits, md = problem(seed, B)
    groups = slot_groups(seed, G, W, R)
    groups0 = [g.clone() for g in groups]
    got = {}
    for fold in (False, True):
        r = Runner(B, fb=fb, fb_max=8, cand=cand, fold=fold, checks=1 if fold else 0, synthetic=synthetic, seed=seed)
        buf0 = r.mamba_slot_read_offsets.clone()
        out, applied, state = run_vps(r, logits, md, groups, R, F, seed)
        got[fold] = (out, r.mamba_slot_read_offsets, applied, state, r._vps_fb_stats, buf0, r)
    (o0, b0, a0, s0, st0, i0, _), (o1, b1, a1, s1, st1, i1, r1) = got[False], got[True]
    assert torch.equal(i0, i1)
    assert torch.equal(o0, o1), "tokens differ"
    assert torch.equal(s0, s1), "random stream differs"
    assert (st0 is None) == (st1 is None) and (st0 is None or torch.equal(st0, st1))
    assert not a0 and a1, (a0, a1)
    assert torch.equal(b0, b1), f"read offsets differ in {int((b0 != b1).sum())} slots"
    assert not torch.equal(b1, i1) or R == 0, "the scatter is visible"
    # the state indices are inputs only; the null slot (every padded row's) keeps offset 0; the next step's GDN reads
    # each live request's checkpoint at read_offsets[slot]: identical, and equal to accepted - 1 on every row < R
    assert all(torch.equal(g, g0) for g, g0 in zip(groups, groups0)), "slot tensors changed"
    assert int(b1[0]) == 0
    n = min(R, W)  # W < R: the walk's width truncates (the GRID's (8, 8, 4, 2) case)
    acc = ((o1[:n] != -1).sum(dim=1) - 1).clamp(min=0).to(b1.dtype)
    for g in groups:
        assert torch.equal(b1[g[:n].long()], acc) and torch.equal(b0[g[:n].long()], acc)
    assert r1._vps_ro_mismatches.tolist() == [0, 0]  # the inline check agrees
    assert r1._vps_ro_folds == 1
    COUNTS["identity_cases"] += 1
    CASES.add((R, B, W, G))


# ---------------------------------------------------------------- 2. in-program operands on a random grid


def test_operands_plus_one_index_put_equal_the_separate_impl():
    g = torch.Generator().manual_seed(11)
    for case in range(300):
        rows = int(torch.randint(1, 300, (1,), generator=g))
        width = int(torch.randint(1, 300, (1,), generator=g))
        G = int(torch.randint(1, 6, (1,), generator=g))
        R = int(torch.randint(0, min(rows, width) + 1, (1,), generator=g))
        nt = torch.full((rows, K + 1), -1, dtype=torch.int32)
        acc = torch.randint(0, K + 1, (rows,), generator=g)
        for i in range(rows):
            nt[i, : int(acc[i]) + 1] = torch.randint(1, 150000, (int(acc[i]) + 1,), generator=g, dtype=torch.int32)
        # odd cases: every position holds a live slot (rows >= R are not hidden behind the null slot), so the
        # formula must match at every position, not only where the slot is real
        groups = slot_groups(case, G, width, width if case % 2 else R, nslots=8192)
        buf0 = torch.randint(0, 9, (8192,), generator=g, dtype=torch.int32)
        buf0[0] = 0
        num_reqs = torch.tensor(R)
        idx, vals = NS["_read_offset_scatter_operands"](nt, num_reqs, groups, buf0.dtype)
        assert idx.dtype == torch.int64 and vals.dtype == buf0.dtype
        assert idx.shape == (G * width,) and vals.shape == (G * width,)
        fold = buf0.clone()
        fold.index_put_((idx,), vals)
        sep = buf0.clone()
        NS["_update_read_offsets_impl"](sep, groups, nt, num_reqs)
        assert torch.equal(fold, sep), (case, rows, width, G, R)
        COUNTS["operand_cases"] += 1
        COUNTS["operand_truncate"] += rows > width
        COUNTS["operand_pad"] += rows < width


# ---------------------------------------------------------------- 3. the fold adds exactly one eager op


class _OpLog(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops, self.on = [], True

    def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
        if self.on:
            self.ops.append(func._schema.name)
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("fb", [True, False])
def test_fold_adds_exactly_one_eager_index_put(fb):
    B, R, W, G = 16, 12, 64, 3
    logits, md = problem(5, B)
    groups = slot_groups(5, G, W, R)
    ops = {}
    for fold in (False, True):
        r = Runner(B, fb=fb, fold=fold, seed=5)
        r._num_reqs_tensor(R)  # the cached scalar exists after the first step
        log = _OpLog()
        saved = {n: NS[n] for n in ("_vps_verify_compiled", "_vps_verify_fb_compiled")}

        def opaque(fn):
            def call(*a, **k):  # a compiled program: its ops are not eager ops
                log.on = False
                try:
                    return fn(*a, **k)
                finally:
                    log.on = True
            return call
        NS["_vps_verify_compiled"] = opaque(saved["_vps_verify_compiled"])
        NS["_vps_verify_fb_compiled"] = opaque(saved["_vps_verify_fb_compiled"])
        try:
            r._fused_read_offsets = (groups, R)
            with log:
                r._sample_spec_verify_vps(logits, md, 0, R, torch.Generator().manual_seed(3), 4 if fb else 0)
        finally:
            NS.update(saved)
        ops[fold] = collections.Counter(log.ops)
    added, removed = ops[True] - ops[False], ops[False] - ops[True]
    assert dict(added) == {"aten::index_put_": 1} and not removed, (dict(added), dict(removed))
    COUNTS["eager_ops_added_" + ("fb" if fb else "plain")] = sum(added.values())


# ---------------------------------------------------------------- 4. CHECK mode


def _chunks(r, n, B=8, R=5, W=64, G=2, F=0, seed=21):
    logits, md = problem(seed, B)
    for i in range(n):
        run_vps(r, logits, md, slot_groups(seed + i, G, W, R), R, F, seed)


def test_check_clean_and_fault_control(caplog):
    r = Runner(8, fold=True, checks=3, faults=1, seed=21)
    with caplog.at_level(logging.INFO, logger="t-l4tail"):
        _chunks(r, 4)
    assert r._vps_ro_mismatches.tolist() == [0, 1]
    assert (r._vps_ro_checks_left, r._vps_ro_checks_done, r._vps_ro_folds) == (0, 3, 4)
    lines = [m for m in caplog.messages if m.startswith("VPS_RO_FOLD_CHECK")]
    assert lines == ["VPS_RO_FOLD_CHECK checks=3 clean_mismatched_slots=0 faulted_checks=1 faulted_caught=1"], lines


def test_check_does_not_change_the_served_buffer():
    logits, md = problem(22, 8)
    groups = slot_groups(22, 2, 64, 6)
    bufs = []
    for checks in (0, 2):
        r = Runner(8, fold=True, checks=checks, faults=1, seed=22)
        run_vps(r, logits, md, groups, 6, 0, 22)
        bufs.append(r.mamba_slot_read_offsets)
    assert torch.equal(bufs[0], bufs[1])


def test_check_sees_a_wrong_fold():
    r = Runner(8, fold=True, checks=2, seed=23)
    real = NS["_read_offset_scatter_operands"]
    NS["_read_offset_scatter_operands"] = lambda *a: (lambda i, v: (i, v + 1))(*real(*a))  # a broken fold
    try:
        _chunks(r, 2, seed=23)
    finally:
        NS["_read_offset_scatter_operands"] = real
    clean, caught = r._vps_ro_mismatches.tolist()
    assert clean > 0 and caught == 0


def test_no_check_when_off():
    r = Runner(8, fold=True, checks=0, seed=24)
    clones = []
    real = torch.Tensor.clone
    torch.Tensor.clone = lambda self, *a, **k: clones.append(tuple(self.shape)) or real(self, *a, **k)
    try:
        _chunks(r, 2, seed=24)
    finally:
        torch.Tensor.clone = real
    assert r._vps_ro_mismatches is None and r._vps_ro_checks_done == 0
    assert (r.mamba_slot_read_offsets.shape[0],) not in clones  # no copy of the buffer is taken


def test_engagement_line(caplog):
    r = Runner(8, fold=True, seed=25)
    with caplog.at_level(logging.INFO, logger="t-l4tail"):
        _chunks(r, 1, G=2, W=64)
    assert caplog.messages[-1] == "q38 VPS_FUSED_READ_OFFSETS: folded 1 chunks (groups=2 width=64 warmed=False)"


# ---------------------------------------------------------------- 5. gate and flag plumbing


def _calls(r, logits, md, groups, R, F):
    seen = []
    saved = {n: NS[n] for n in ("_vps_verify_compiled", "_vps_verify_fb_compiled")}
    NS["_vps_verify_compiled"] = lambda *a: seen.append(("plain", len(a))) or saved["_vps_verify_compiled"](*a)
    NS["_vps_verify_fb_compiled"] = lambda *a: seen.append(("fb", len(a))) or saved["_vps_verify_fb_compiled"](*a)
    try:
        run_vps(r, logits, md, groups, R, F, 0)
    finally:
        NS.update(saved)
    return seen


@pytest.mark.parametrize("fb", [True, False])
def test_gate(fb):
    logits, md = problem(31, 8)
    groups = slot_groups(31, 2, 64, 5)
    n = 21 if fb else 17  # the plain signatures' argument counts
    assert _calls(Runner(8, fb=fb, fold=False), logits, md, groups, 5, 0) == [("fb" if fb else "plain", n)]
    assert _calls(Runner(8, fb=fb, fold=True), logits, md, groups, 5, 0) == [("fb" if fb else "plain", n + 3)]
    for empty in (None, []):
        r = Runner(8, fb=fb, fold=True)
        assert _calls(r, logits, md, empty, 5, 0) == [("fb" if fb else "plain", n)] and r._vps_ro_folds == 0
    r = Runner(8, fb=fb, fold=True)
    r.mamba_slot_read_offsets = None
    r._fused_read_offsets = (groups, 5)
    r._sample_spec_verify_vps(logits, md, 0, 5, torch.Generator().manual_seed(1), 0)
    assert not r._fused_read_offsets_applied


@pytest.mark.parametrize("fb", [True, False])
def test_flags_off_reads_no_fold_state(fb):
    """Flag off, `_sample_spec_verify_vps` reads none of D83's runner state nor the read-offset plumbing (as base
    does not): a runner without them (EAGER-FIX's test fake is one; the full-v2 regression j-f4b410b3 caught the
    unconditional `self._fused_read_offsets` read there) verifies to the same tokens and random stream."""
    B, R = 8, 5
    logits, md = problem(81, B)
    got = []
    for strip in (False, True):
        r = Runner(B, fb=fb, fold=False, seed=81)
        if strip:
            for a in ("_vps_ro_fold", "_vps_ro_checks_left", "_vps_ro_check_faults", "_vps_ro_checks_done",
                      "_vps_ro_mismatches", "_vps_ro_folds", "_vps_ro_warmed", "_fused_read_offsets",
                      "_fused_read_offsets_applied", "mamba_slot_read_offsets"):
                delattr(r, a)
        else:
            r._fused_read_offsets = (slot_groups(81, 2, 64, R), R)
        gen = torch.Generator().manual_seed(82)
        got.append((r._sample_spec_verify_vps(logits, md, 0, R, gen, 4 if fb else 0), gen.get_state()))
    assert torch.equal(got[0][0], got[1][0]) and torch.equal(got[0][1], got[1][1])


def _init_statements():
    tree = ast.parse(RUNNER_SRC)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_init_speculative_decoding")
    stmts = [n for n in fn.body if isinstance(n, (ast.Assign, ast.AnnAssign))
             and ast.unparse(n.targets[0] if isinstance(n, ast.Assign) else n.target).startswith("self._vps_ro_")]
    return "\n".join(ast.get_source_segment(RUNNER_SRC, n) for n in stmts)


def test_init_reads_the_flags(monkeypatch):
    envs = w._load("q38_l4tail_envs", os.path.join(w.PKG, "envs.py"))
    code = _init_statements()
    names = ("VLLM_TPU_VPS_FUSED_READ_OFFSETS", "VLLM_TPU_VPS_FUSED_READ_OFFSETS_CHECK",
             "VLLM_TPU_VPS_FUSED_READ_OFFSETS_CHECK_FAULT")
    for n in names:
        monkeypatch.delenv(n, raising=False)
    off = types.SimpleNamespace()
    exec(code, {"envs": envs, "torch": torch}, {"self": off})
    assert (off._vps_ro_fold, off._vps_ro_checks_left, off._vps_ro_check_faults) == (False, 0, 0)
    assert (off._vps_ro_checks_done, off._vps_ro_mismatches, off._vps_ro_folds, off._vps_ro_warmed) == (0, None, 0, set())
    for n, v in zip(names, ("1", "7", "2")):
        monkeypatch.setenv(n, v)
    on = types.SimpleNamespace()
    exec(code, {"envs": envs, "torch": torch}, {"self": on})
    assert (on._vps_ro_fold, on._vps_ro_checks_left, on._vps_ro_check_faults) == (True, 7, 2)


def test_flags_registered_default_off_and_check_is_host_only(monkeypatch):
    envs = w._load("q38_l4tail_envs2", os.path.join(w.PKG, "envs.py"))
    for n, on, default in (("VLLM_TPU_VPS_FUSED_READ_OFFSETS", "1", False),
                           ("VLLM_TPU_VPS_FUSED_READ_OFFSETS_CHECK", "5", 0),
                           ("VLLM_TPU_VPS_FUSED_READ_OFFSETS_CHECK_FAULT", "1", 0)):
        assert n in envs.environment_variables
        monkeypatch.delenv(n, raising=False)
        assert envs.environment_variables[n]() == default
        monkeypatch.setenv(n, on)
        assert envs.environment_variables[n]() != default
    tc = open(w.TC).read()
    node = next(n for n in ast.parse(tc).body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", None) == "_TPU_COMPILE_ENV_IGNORED")
    ignored = ast.literal_eval(node.value)
    assert "VLLM_TPU_VPS_FUSED_READ_OFFSETS" not in ignored
    assert {"VLLM_TPU_VPS_FUSED_READ_OFFSETS_CHECK", "VLLM_TPU_VPS_FUSED_READ_OFFSETS_CHECK_FAULT"} <= ignored


# ---------------------------------------------------------------- 6. flags off: same programs as base


def _fx_code(fn, tensors, build):
    # Real-mode trace; the fallback rows' one_hot validates its input with .item(), which make_fx refuses by default
    # (a value read, not control flow: the traced ops are the same either way), so allow it.
    try:
        code = make_fx(lambda *t: fn(*build(t)), tracing_mode="real", _error_on_data_dependent_ops=False)(*tensors).code
        COUNTS["fx_tracer_make_fx"] += 1
    except Exception:  # fall back to the graph dynamo hands a backend (what the TPU compile traces)
        graphs = []

        def capture(gm, inputs):
            graphs.append(gm.code)
            return gm.forward
        torch._dynamo.reset()
        torch.compile(lambda *t: fn(*build(t)), backend=capture, fullgraph=True, dynamic=False)(*tensors)
        code = graphs[0]
        COUNTS["fx_tracer_dynamo"] += 1
    return "\n".join(ln for ln in code.splitlines() if ln.strip() and not ln.strip().startswith("#"))


def _fb_inputs(B, F, cand, seed):
    logits, md = problem(seed, B)
    width = VP.VP_KC if cand else V
    g = torch.Generator().manual_seed(seed)
    T = md.target_logits_indices.shape[0]
    tensors = [logits, md.bonus_logits_indices, md.target_logits_indices,
               torch.full((B, 1), 0.6).bfloat16(), torch.full((B, 1), 20, dtype=torch.int32),
               torch.full((B, 1), 0.95), torch.rand(B, width, generator=g), torch.rand(md.draft_token_ids.shape, generator=g),
               torch.rand(T, width, generator=g), md.draft_token_ids, md.draft_lengths, md.segment_ids, md.group_indices,
               torch.tensor([1.0, 0.32, 0.0]), torch.tensor([B - 1], dtype=torch.int32)]
    if cand and F:
        tensors.append(torch.rand(F, V, generator=g))
    return tensors


@pytest.mark.parametrize("B,F,cand", [(8, 8, True), (16, 0, True), (8, 4, False)])
def test_flags_off_fb_program_traces_like_base(B, F, cand):
    t = _fb_inputs(B, F, cand, 41 + B + F)

    def build(x):
        fb_u = x[15] if len(x) > 15 else None
        return (x[0], 0, *x[1:14], K, True, x[14], F, cand, fb_u)
    new = _fx_code(NS["_vps_verify_fb_impl"], t, build)
    base = _fx_code(BNS["_vps_verify_fb_impl"], t, build)
    assert new == base
    a, b = NS["_vps_verify_fb_impl"](*build(t)), BNS["_vps_verify_fb_impl"](*build(t))
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    COUNTS["fx_identical_programs"] += 1


def test_flags_off_plain_program_traces_like_base():
    t = _fb_inputs(8, 0, False, 47)[:14]

    def build(x):
        return (x[0], 0, *x[1:14], K, True)
    assert _fx_code(NS["_vps_verify_impl"], t, build) == _fx_code(BNS["_vps_verify_impl"], t, build)
    assert torch.equal(NS["_vps_verify_impl"](*build(t)), BNS["_vps_verify_impl"](*build(t)))
    COUNTS["fx_identical_programs"] += 1


def _sig(args):
    out = []
    for a in args:
        if isinstance(a, torch.Tensor):
            out.append(("T", tuple(a.shape), str(a.dtype)))
        elif isinstance(a, (list, tuple)):
            out.append(("L", _sig(a), len({id(x) for x in a}) == len(a)))
        elif isinstance(a, torch.dtype):
            out.append(("D", str(a)))
        else:
            out.append(("V", a))
    return tuple(out)


@pytest.mark.parametrize("fb,F,cand", [(True, 8, True), (True, 4, False), (False, 0, False)])
def test_flags_off_runner_path_issues_base_calls(fb, F, cand):
    B, R = 16, 13
    logits, md = problem(51, B)
    groups = slot_groups(51, 2, 64, R)
    rec = {}
    for tag, ns in (("new", NS), ("base", BNS)):
        seen = []
        saved = {n: ns[n] for n in ("_vps_verify_compiled", "_vps_verify_fb_compiled")}
        ns["_vps_verify_compiled"] = lambda *a, s=saved: seen.append(_sig(a)) or s["_vps_verify_compiled"](*a)
        ns["_vps_verify_fb_compiled"] = lambda *a, s=saved: seen.append(_sig(a)) or s["_vps_verify_fb_compiled"](*a)
        r = Runner(B, fb=fb, cand=cand, fold=False, seed=51)
        if tag == "base":
            r._sample_spec_verify_vps = types.MethodType(BNS["_sample_spec_verify_vps"], r)
            r._vps_verify_fb = types.MethodType(BNS["_vps_verify_fb"], r)
        try:
            out, applied, state = run_vps(r, logits, md, groups, R, F, 51)
        finally:
            ns.update(saved)
        rec[tag] = (seen, out, applied, state, r.mamba_slot_read_offsets)
    assert rec["new"][0] == rec["base"][0] and len(rec["new"][0]) == 1
    assert torch.equal(rec["new"][1], rec["base"][1]) and torch.equal(rec["new"][3], rec["base"][3])
    assert rec["new"][2] is False and torch.equal(rec["new"][4], rec["base"][4])


def _src_of(src, name, cls=None):
    tree = ast.parse(src)
    body = tree.body
    if cls:
        body = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls).body
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    return textwrap.dedent(ast.get_source_segment(src, node, padded=True))


_SEEDFIX_BLOCK = re.compile(r'\n( *)seed_plan = getattr\(self, "_seed_plan", None\)\n\1if seed_plan is not None:\n(?:\1 .*\n)+')


def _strip_seedfix(src):
    """q38 L4 SEEDFIX (runner/seedfix.py): the seeded-row overlay block after a site's uniform draws (it overwrites
    seeded rows' slices only; unseeded identity is proven in tests/q38_l4_seedfix). Removed before the comparison,
    and it must occur exactly once in `_sample_spec_verify_fused`."""
    out, n = _SEEDFIX_BLOCK.subn("\n", src)
    return out, n


def test_stock_fold_and_separate_update_unchanged():
    ref = open(REF).read()
    for name, cls in (("_update_read_offsets_impl", None), ("_verify_fused_impl", None),
                      ("_sample_spec_verify_fused", "TPUModelRunner"),
                      ("_update_mamba_slot_read_offsets", "TPUModelRunner"),
                      ("_warm_vps_draws", "TPUModelRunner")):
        got, n = _strip_seedfix(_src_of(RUNNER_SRC, name, cls))
        assert n == (1 if name == "_sample_spec_verify_fused" else 0), (name, n)
        assert got.strip() == _src_of(ref, name).strip(), name
    assert RUNNER_SRC.count("                    if not self._fused_read_offsets_applied:\n"
                            "                        self._update_mamba_slot_read_offsets(\n") == 1


# ---------------------------------------------------------------- 7. warm-up


def _warm(r, B, served_only=False, ns=NS, pre=None):
    logits, md = problem(61, B)
    seen = []
    saved = {n: ns[n] for n in ("_vps_verify_compiled", "_vps_verify_fb_compiled")}
    ns["_vps_verify_compiled"] = lambda *a: seen.append(_sig(a)) or saved["_vps_verify_compiled"](*a)
    ns["_vps_verify_fb_compiled"] = lambda *a: seen.append(_sig(a)) or saved["_vps_verify_fb_compiled"](*a)
    try:
        (pre or M["_precompile_vps_verify"])(r, logits, md.bonus_logits_indices, md.target_logits_indices,
                                              md.draft_token_ids, md.draft_lengths, md.segment_ids, md.group_indices,
                                              torch.zeros(md.draft_token_ids.shape), served_only=served_only)
    finally:
        ns.update(saved)
    return seen


@pytest.mark.parametrize("fb,cand,served_only", [(True, True, False), (True, False, False), (True, True, True),
                                                 (False, False, False)])
def test_warmup_flag_off_is_base(fb, cand, served_only):
    new = _warm(Runner(8, fb=fb, cand=cand, fold=False), 8, served_only)
    base = _warm(Runner(8, fb=fb, cand=cand, fold=False), 8, served_only, BNS, BNS["_precompile_vps_verify"])
    assert new == base and new


def _serving_sig(r, B, R, W, G, F):
    logits, md = problem(61, B)
    seen = []
    saved = {n: NS[n] for n in ("_vps_verify_compiled", "_vps_verify_fb_compiled")}
    NS["_vps_verify_compiled"] = lambda *a: seen.append(_sig(a)) or saved["_vps_verify_compiled"](*a)
    NS["_vps_verify_fb_compiled"] = lambda *a: seen.append(_sig(a)) or saved["_vps_verify_fb_compiled"](*a)
    try:
        run_vps(r, logits, md, slot_groups(62, G, W, R), R, F, 62)
    finally:
        NS.update(saved)
    (s,) = seen
    return s


@pytest.mark.parametrize("fb,cand", [(True, True), (True, False), (False, False)])
@pytest.mark.parametrize("most", [None, 32])
def test_warmup_flag_on_covers_serving(fb, cand, most):
    r = Runner(8, fb=fb, cand=cand, fold=True)
    r.most_model_len, r.num_reqs_most_model_len = (None, None) if most is None else (4096, most)
    before = r.mamba_slot_read_offsets.clone()
    warmed = _warm(r, 8)
    G, widths = r._read_offset_fold_geometry()
    assert G == 3 and widths == sorted({64} | ({most} if most else set()))
    variants = VW.fallback_variants(r._vps_fb_max, r._vps_tiegap) if fb else [0]
    assert len(warmed) == len(variants) * len(widths)  # fold variants only, no plain variant
    for s in warmed:
        lst = s[-3]
        assert lst[0] == "L" and lst[2] is True and len(lst[1]) == G  # G distinct group tensors
        assert s[-1] == ("D", "torch.int32") and s[-2] == ("T", (), "torch.int64")
    assert {s[-3][1][0][1][0] for s in warmed} == set(widths)
    assert r._vps_ro_warmed == {(G, wd) for wd in widths}
    assert torch.equal(r.mamba_slot_read_offsets, before)  # the scatter warm-up wrote zeros at the null slot only
    for wd in widths:
        for F in variants:
            s = _serving_sig(Runner(8, fb=fb, cand=cand, fold=True), 8, 5, wd, G, F)
            assert s in warmed, (wd, F)
            COUNTS["warm_serving_keys_covered"] += 1
    COUNTS["warm_fold_programs"] += len(warmed)


def test_warm_scatter_writes_zeros_only():
    r = Runner(8, fold=True)
    before = r.mamba_slot_read_offsets.clone()
    r._warm_folded_scatter(torch.zeros(3 * 64, dtype=torch.int64), torch.full((3 * 64,), 5, dtype=torch.int32))
    assert torch.equal(r.mamba_slot_read_offsets, before) and int(r.mamba_slot_read_offsets[0]) == 0


def test_geometry():
    r = Runner(8)
    assert r._read_offset_fold_geometry() == (3, [64])
    r._has_mamba_state = True  # compact pool: one shared slot tensor
    assert r._read_offset_fold_geometry() == (1, [64])
    r._has_mamba_state = False
    r.most_model_len, r.num_reqs_most_model_len = 8192, 16
    assert r._read_offset_fold_geometry() == (3, [16, 64])
    r.attn_groups = [[types.SimpleNamespace(metadata_builders=[types.SimpleNamespace(is_mamba_group=False)])]]
    assert r._read_offset_fold_geometry()[0] == 0
    r._vps_ro_fold = True
    assert r._vps_ro_fold_warmup_args() == []  # no mamba group feeds the update: serving runs the plain variant


# ---------------------------------------------------------------- 8. EAGER-FIX E-b warm draws (the resolved conflict)

WARM_ROUNDS = 4  # the served value of VLLM_TPU_VPS_WARM_DRAWS


def _warm_draws(r, B, served_only, ns, pre):
    """The warm-up's compiled-call signatures, its `_warm_vps_draws` calls and the draws those calls issue, with
    VLLM_TPU_VPS_WARM_DRAWS=WARM_ROUNDS in the namespace the runner code reads `envs` from."""
    calls, rands = [], []
    real_draws = r._warm_vps_draws

    def draws(num_reqs, draft, target, width, fb_shapes):
        n0 = len(rands)
        real_draws(num_reqs, draft, target, width, fb_shapes)
        calls.append((num_reqs, tuple(draft.shape), tuple(target.shape), width, tuple(fb_shapes), len(rands) - n0))
    r._warm_vps_draws = draws
    real_rand = torch.rand

    def rand(*a, **k):
        # only the warm-up's own draws (its scratch generator); problem()'s seeded draws are not warm draws
        g = k.get("generator")
        if g is not None and g is r.__dict__.get("_vps_warm_generator"):
            rands.append((tuple(a[0]), k.get("dtype")))
        return real_rand(*a, **k)
    saved_envs = (NS["envs"], BNS["envs"])
    on = types.SimpleNamespace(**vars(NS["envs"]), VLLM_TPU_VPS_WARM_DRAWS=WARM_ROUNDS)
    NS["envs"] = BNS["envs"] = on
    torch.rand = rand
    try:
        sigs = _warm(r, B, served_only, ns, pre)
    finally:
        torch.rand = real_rand
        NS["envs"], BNS["envs"] = saved_envs
    return sigs, calls, rands


@pytest.mark.parametrize("fb,cand,served_only", [(True, True, False), (True, False, False), (True, True, True),
                                                 (False, False, False)])
def test_warmup_keeps_eagerfix_warm_draws(fb, cand, served_only):
    got = {}
    for tag in ("base", "off", "on"):
        r = Runner(8, fb=fb, cand=cand, fold=tag == "on")
        if tag == "base":
            r._warm_vps_draws = types.MethodType(BNS["_warm_vps_draws"], r)
            got[tag] = _warm_draws(r, 8, served_only, BNS, BNS["_precompile_vps_verify"])
        else:
            got[tag] = _warm_draws(r, 8, served_only, NS, None)
    variants = VW.fallback_variants(8, 0) if fb else [0]
    if served_only and fb:
        variants = [v for v in variants if v == VW.fallback_slots(0, 0, 8)]
    sig_b, calls_b, rands_b = got["base"]
    assert got["off"] == got["base"], "flag off: the warm-up differs from base"
    _, calls_on, rands_on = got["on"]
    assert calls_on == calls_b, ("the fold drops / changes EAGER-FIX's warm draws", calls_on, calls_b)
    assert rands_on == rands_b
    assert len(calls_b) == len(variants) >= 1
    for c, F in zip(calls_b, variants):
        assert c[5] == WARM_ROUNDS * (3 + (1 if cand and F > 0 else 0)), c
    assert all(dt == torch.float32 for _, dt in rands_b)
    COUNTS["warm_draw_calls"] += len(calls_on)
    COUNTS["warm_draws_issued"] += len(rands_on)


# ---------------------------------------------------------------- 9. program census over a multi-step run


def _sample_tokens_gate():
    """sample_tokens' own post-verify statement: the If whose test is `not self._fused_read_offsets_applied`."""
    fn = next(n for n in ast.walk(ast.parse(RUNNER_SRC)) if isinstance(n, ast.FunctionDef) and n.name == "sample_tokens")
    ifs = [n for n in ast.walk(fn) if isinstance(n, ast.If)
           and ast.unparse(n.test) == "not self._fused_read_offsets_applied"]
    assert len(ifs) == 1, len(ifs)
    return compile(ast.unparse(ifs[0]), "<sample_tokens gate>", "exec")


def _census_ns(counts):
    """The runner's own `_update_mamba_slot_read_offsets` with its TPU branch taken on CPU (device check `"tpu"` ->
    `"cpu"`, VLLM_TPU_FUSED_READ_OFFSET_UPDATE on, as served) and `_update_read_offsets_compiled` counted."""
    src = _src_of(RUNNER_SRC, "_update_mamba_slot_read_offsets", "TPUModelRunner")
    assert src.count('.device.type == "tpu"') == 1
    ns = dict(NS)
    ns["envs"] = types.SimpleNamespace(**{**vars(NS["envs"]), "VLLM_TPU_FUSED_READ_OFFSET_UPDATE": True})

    def separate_compiled(*a):
        counts["update_read_offsets_compiled"] += 1
        return NS["_update_read_offsets_impl"](*a)
    ns["_update_read_offsets_compiled"] = separate_compiled
    exec(src.replace('.device.type == "tpu"', '.device.type == "cpu"'), ns)
    return ns["_update_mamba_slot_read_offsets"]


@pytest.mark.parametrize("fb", [True, False])
def test_program_census_over_steps(fb):
    B, G, W = 16, 3, 64
    Rs = [12, 13, 5, 16, 1, 9, 16, 2]
    gate = _sample_tokens_gate()
    res = {}
    for fold in (False, True):
        counts = collections.Counter()
        r = Runner(B, fb=fb, fold=fold, seed=71)
        upd = _census_ns(counts)
        r._update_mamba_slot_read_offsets = lambda *a, _u=upd, _r=r: (
            counts.__setitem__("update_method", counts["update_method"] + 1), _u(_r, *a))[1]
        saved = {n: NS[n] for n in ("_vps_verify_compiled", "_vps_verify_fb_compiled")}

        def counted(name):
            def call(*a):
                counts[(name, len(a))] += 1
                return saved[name](*a)
            return call
        NS["_vps_verify_compiled"] = counted("_vps_verify_compiled")
        NS["_vps_verify_fb_compiled"] = counted("_vps_verify_fb_compiled")
        steps = []
        try:
            for i, R in enumerate(Rs):
                logits, md = problem(700 + i, B)
                groups = slot_groups(700 + i, G, W, R)
                groups0 = [g.clone() for g in groups]
                gen = torch.Generator().manual_seed(800 + i)
                # sample_tokens: set the chunk's slot groups, verify, then its own post-verify statement
                r._fused_read_offsets = (groups, R)
                r._fused_read_offsets_applied = False
                out = r._sample_spec_verify_vps(logits, md, 0, R, gen, 4 if fb else 0)
                exec(gate, NS, {"self": r, "mamba_indices": groups, "next_tokens": out, "num_reqs": R})
                assert all(torch.equal(g, g0) for g, g0 in zip(groups, groups0))
                read = torch.stack([r.mamba_slot_read_offsets[g[:R].long()] for g in groups])  # the next step's reads
                steps.append((out, gen.get_state(), r.mamba_slot_read_offsets.clone(), read))
        finally:
            NS.update(saved)
        res[fold] = (steps, counts, r._vps_ro_folds)
    n = len(Rs)
    name = "_vps_verify_fb_compiled" if fb else "_vps_verify_compiled"
    nargs = 21 if fb else 17
    (s_off, c_off, f_off), (s_on, c_on, f_on) = res[False], res[True]
    assert dict(c_off) == {(name, nargs): n, "update_method": n, "update_read_offsets_compiled": n}, dict(c_off)
    assert dict(c_on) == {(name, nargs + 3): n}, dict(c_on)  # the fold variant only; the separate program never runs
    assert (f_off, f_on) == (0, n)
    for (o0, g0, b0, r0), (o1, g1, b1, r1) in zip(s_off, s_on):
        assert torch.equal(o0, o1) and torch.equal(g0, g1) and torch.equal(b0, b1) and torch.equal(r0, r1)
    COUNTS["census_steps_" + ("fb" if fb else "plain")] = n
    COUNTS["census_separate_per_step_off"] = c_off["update_read_offsets_compiled"] // n
    COUNTS["census_separate_per_step_on"] = c_on["update_read_offsets_compiled"] // n


def test_zz_counts():
    COUNTS["identity_row_cases"] = len(CASES)
    out = os.environ.get("Q38_L4TAIL_COUNTS")
    if out:
        with open(out, "w") as f:
            json.dump(dict(COUNTS), f, sort_keys=True)
    print("Q38_L4TAIL_D83_COUNTS " + json.dumps(dict(COUNTS), sort_keys=True))
