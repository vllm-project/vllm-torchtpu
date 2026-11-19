# SPDX-License-Identifier: Apache-2.0
"""q38 L4 VFY-ONE (VLLM_TPU_VPS_VFY_ONE; H1 cards T-3 VFY-ONE and DH-C SEED-INTO-VERIFY). CDK CPU pod only.

Torch CPU, no TPU, no vLLM install: the runner's functions / methods (tpu_runner.py) and the drafter's seed
functions / methods (eagle3.py) are extracted with ast into the harness of tests/q38mm_dstock/test_vps_wiring.py
(stub packages, TP1 VPS head, compiled bodies run eagerly), as D83's tests/l4tail/test_vps_ro_fold.py does;
runner/vfy_one.py is loaded from the tree (tests/l4vfy/vfy_harness.py). Q38MM_SRC_ROOT points at a mutated copy.
The LM head is a fake vocab shard (hidden @ W^T in bf16) behind the real vps_wiring.local_logits.

1. Bitwise identity against the D83 chain (compute_selected_local_logits, then `_sample_spec_verify_vps` with
   D83's fold): tokens, random stream, fallback stats, read-offset buffer, the fold mark, chunk counts, at
   8 / 16 / 32 / 48 / 64 live requests, F in {0, 8}, full and candidate draws, values 1 and 2 (value 2: the seed
   outputs equal eagle3's `_async_seed_inputs_impl` on the chain's tokens); multi-step stats accumulation; the
   layout planner's pre-selected rows; non-synthetic acceptance.
2. CPU AOT census: D83 chain = 2 compiled calls + eager {_to_copy, add, index_put_}; VFY-ONE = 1 call + index_put_.
3. Fallbacks: a never-warmed key, no read-offset groups, no warm-up: the D83 chain, bitwise, counted.
4. Warm-up: every steady decode serving key (n = 1 .. max_num_reqs, every fallback width, every read-offset
   width, with / without the seed) is warmed; value 1 warms the seed variant too; counts exact.
5. Flags off: no controller; execute_model's statements keep the D83 chain; the seed block calls its program
   with the same arguments; the per-step hook and the end-of-step stash drop do nothing.
6. execute_model's real statements (ast-extracted): the defer gate (single chunk, spec metadata, VPS chunk) and
   the logits branch (VfyChunk instead of the local logits; the pre-selected path).
7. CHECK: results are the D83 chain's; a broken program / seed is counted.
8. DH-C take through the seed block's real statements: taken only on identity (census -1), misses fall back.
9. DH-A composition (one seed fold per step) with a DraftAsm stand-in that has the proposed hooks; without the
   hooks value 2 does not fold the seed.
Run: python -m pytest -q -p no:cacheprovider --noconftest tests/l4vfy/test_vfy_one.py
"""

import ast
import collections
import contextlib
import json
import logging
import os
import sys
import textwrap
import types

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "q38mm_dstock"))
sys.path.insert(0, HERE)
import test_vps_wiring as w  # noqa: E402
import vfy_harness as H  # noqa: E402

VP, VW, K, V = w.VP, w.VW, w.K, w.V
RUNNER, EAGLE = w.RUNNER, w.EAGLE
RUNNER_SRC, EAGLE_SRC = open(RUNNER).read(), open(EAGLE).read()
VF = H.load_vfy("main")
VF._L6_AB[:] = [None]
VF._tp_rank = lambda: 0
ENVS_MOD = w.HARNESS.mods["vllm_torchtpu.envs"]
COUNTS = collections.Counter()
HID = 64


def _assign(src, name, ns):
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == name for t in n.targets))
    exec(ast.get_source_segment(src, node), ns)


def _ns():
    ns = dict(w.NS)
    ns["logger"] = logging.getLogger("t-l4vfy")
    ns["logger"].warning_once = ns["logger"].warning
    ns["vps_wiring"] = VW
    ns["synchronize_tensors"] = lambda *a, **k: None
    ns["_vfy_one"] = VF
    ns["os"] = os
    ns["bisect"] = __import__("bisect")
    ns["envs"] = types.SimpleNamespace(**vars(w.NS["envs"]), VLLM_TPU_SPEC_VERIFY_PRECOMPILE_SERVED_ONLY=False,
                                       VLLM_TPU_VPS_WARM_DRAWS=0)
    return ns


NS = _ns()
# q38 L4 v5 composition (V5COMP-BUILD; tests only): VPSHOST's l6_ab lever reads (sampsnap, statsasync) in the stock
# chain and in VFY's copies of it are off in this harness; tests/q38_l4_v5c/test_v5c_compose.py runs them on.
NS["_l6_ab"] = types.SimpleNamespace(on=lambda lever: False, note=lambda *a, **k: None)
_assign(RUNNER_SRC, "MIN_NUM_SEQS", NS)
_assign(RUNNER_SRC, "_Q38_RO_FOLD_LOG_EVERY", NS)
w._fns(RUNNER, ["_update_read_offsets_impl", "_read_offset_scatter_operands", "_vps_verify_impl",
                "_vps_verify_fb_impl", "_vfy_one_impl", "_get_padded_token_len",
                "_get_padded_num_reqs_with_upper_limit", "_vfy_rss_bytes"], NS)
NS["_vps_verify_compiled"] = NS["_vps_verify_impl"]
NS["_vps_verify_fb_compiled"] = NS["_vps_verify_fb_impl"]
NS["_vfy_one_compiled"] = NS["_vfy_one_impl"]
METHODS = ["_sample_spec_verify_vps", "_vps_verify_fb", "_apply_folded_read_offsets", "_check_folded_read_offsets",
           "_read_offset_fold_geometry", "_vps_ro_fold_warmup_args", "_warm_folded_scatter", "_num_reqs_tensor",
           "compute_selected_local_logits", "compute_local_logits_from_hidden_states", "_vfy_chunk_logits",
           "_vfy_fb_draw", "_vfy_fb_tick", "_vfy_chain", "_vfy_seed_spec", "_vfy_verify", "_vfy_check",
           "_vfy_precompile", "_vfy_defer_logits", "_log_vps_fallback"]
M = dict(zip(METHODS, w._fns(RUNNER, METHODS, NS, cls="TPUModelRunner")))


class _NoStatsInFlight:
    """q38 L4 T-1(i) (v5 composition, tests only): the runner's AsyncCounts with nothing in flight (statsasync off)."""

    pending = ()

    @staticmethod
    def ready(wait=False):
        return []

DESC = object()  # DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR stand-in
ENS = {"torch": torch, "INVALID_TOKEN_ID": -1, "DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR": DESC, "DraftChunkInputs": object,
       "envs": types.SimpleNamespace(VLLM_TPU_FUSED_ASYNC_DRAFT_SEED=True, VLLM_TPU_DRAFT_SEED_FUSED=True,
                                     VLLM_TPU_DRAFT_SEED_EXTRAS=True, VLLM_TPU_PROPOSE_DEVICE_CONSTANTS=True)}
w._fns(EAGLE, ["_maybe_pad_dim0", "_async_draft_seed_impl", "_async_seed_inputs_impl"], ENS)
SEED_IMPL = ENS["_async_seed_inputs_impl"]
EMETHODS = ["vfy_seed_static", "_vfy_seed_args", "vfy_seed_spec", "vfy_seed_warm_spec", "_vfy_seed_take",
            "_vfy_seed_or_run", "_device_constant", "_draft_rows_len"]
EM = dict(zip(EMETHODS, w._fns(EAGLE, EMETHODS, ENS, cls="Eagle3Proposer")))


@pytest.fixture(autouse=True, scope="module")
def _harness_modules_installed():
    """test_vps_wiring's stubs, plus vllm_torchtpu.runner.vfy_one = the tree's vfy_one (the drafter's lazy imports)."""
    with w.HARNESS.active():
        pkg = sys.modules["vllm_torchtpu.runner"]
        had = hasattr(pkg, "vfy_one")
        pkg.vfy_one = VF
        with H.modules({"vllm_torchtpu.runner.vfy_one": VF}):
            yield
        if not had:
            del pkg.vfy_one


class Proc:
    """A TP1 vocab shard: the rank's local logits = hidden @ W^T (bf16), as LogitsProcessor._get_logits(skip_gather)."""

    def _get_logits(self, hidden, head, bias, skip_gather=False):
        assert bias is None and skip_gather
        return torch.nn.functional.linear(hidden, head.weight)


_gw = torch.Generator().manual_seed(4242)
HEAD = types.SimpleNamespace(weight=(torch.randn(V, HID, generator=_gw) * 0.5).bfloat16())
PROC = Proc()


class AlwaysWarm(set):
    def __contains__(self, k):
        return True


class Proposer:
    """The async eagle3 drafter's state the seed methods read (`_loop_bucket` as the runner pads)."""

    def __init__(self, runner, asm=None):
        self.runner = runner
        self._draft_rows = True
        self.draft_chunks = []
        if asm is not None:
            self._draft_asm = asm

    def _loop_bucket(self, n):
        return NS["_get_padded_token_len"](self.runner.num_tokens_paddings, n)


for _n, _f in EM.items():
    setattr(Proposer, _n, _f)


class Runner:
    """The runner state VFY-ONE, D83's fold, the VPS verify and the warm-up read (as D83's test Runner)."""

    def __init__(self, B=8, *, fb=True, fb_max=8, cand=False, fold=True, mode=0, check=0, synthetic=True,
                 nslots=4096, seed=0, drafter=True, asm=None):
        self.rejection_sampler = types.SimpleNamespace(synthetic_conditional_rates=torch.tensor([1.0, 0.32, 0.0]),
                                                       synthetic_mode=synthetic)
        self.speculative_config = types.SimpleNamespace(num_speculative_tokens=K)
        self._vps_head = VW.VpsHead(PROC, HEAD, 0, V, V, 1)
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
        self._vps_ro_checks_left, self._vps_ro_check_faults, self._vps_ro_checks_done = 0, 0, 0
        self._vps_ro_mismatches = None
        self._vps_ro_folds = 0
        self._vps_ro_warmed = set()
        g = torch.Generator().manual_seed(7000 + seed)
        buf = torch.randint(0, 9, (nslots,), generator=g, dtype=torch.int32)
        buf[0] = 0
        self.mamba_slot_read_offsets = buf
        self.top_k = torch.full((256,), 20, dtype=torch.int32)
        self.temps = torch.full((256,), 0.6)
        self._has_mamba_state = False
        mk = lambda m: types.SimpleNamespace(metadata_builders=[types.SimpleNamespace(is_mamba_group=m)])  # noqa: E731
        self.attn_groups = [[mk(True), mk(False)], [mk(True)], [mk(False), mk(True)]]
        self.num_reqs_max_model_len, self.most_model_len, self.num_reqs_most_model_len = 64, None, None
        self.max_num_reqs = 64
        self.num_tokens_paddings = [16, 32, 64, 128, 256]
        self.num_reqs_paddings = [8, 16, 32, 64]
        self._hidden_states_dtype = torch.bfloat16
        self.model_config = types.SimpleNamespace(get_hidden_size=lambda: HID)
        self.warmup_done = True
        self._vfy = VF.VfyOne(mode, check, "") if mode else None
        self._is_async_drafter = drafter
        self.drafter = self._async_drafter = Proposer(self, asm) if drafter else None

    def _padded_sampling_params(self, s, e, n, dtype, device):
        k = torch.where(self.top_k >= V, 0, self.top_k)
        return (self.temps.to(dtype).view(-1, 1)[:n], k.view(-1, 1)[:n].clone(),
                torch.full((n, 1), 0.95, dtype=torch.float32))

    def _upload_unless_unchanged(self, slot, host):
        return host

    def _precompile_timed(self, name):
        return contextlib.nullcontext()


for _n, _f in M.items():
    setattr(Runner, _n, _f)


def problem(seed, B, R=None, extra_rows=8):
    """A VPS step: hidden states (T rows), logits indices (N = B * (K + 1) of them), spec metadata, and a draft
    chunk whose seed inputs are consistent with R live requests carrying K drafts each."""
    R = B if R is None else R
    g = torch.Generator().manual_seed(seed)
    N = B * (K + 1)
    T = N + extra_rows
    hidden = torch.randn(T, HID, generator=g).bfloat16()
    idx = torch.randperm(T, generator=g)[:N].to(torch.int32)
    logits = PROC._get_logits(hidden[idx.long()], HEAD, None, skip_gather=True)
    rows = torch.arange(N).view(B, K + 1)
    am = logits[rows[:, :K].reshape(-1)].float().argmax(-1)
    draft = torch.where(torch.rand(B * K, generator=g) < 0.6, am,
                        torch.randint(0, V, (B * K,), generator=g)).to(torch.int32)
    md = types.SimpleNamespace(bonus_logits_indices=rows[:, K].contiguous(),
                               target_logits_indices=rows[:, :K].reshape(-1).contiguous(),
                               draft_token_ids=draft, draft_lengths=torch.full((B,), K, dtype=torch.int32),
                               segment_ids=torch.arange(B).repeat_interleave(K).to(torch.int32),
                               group_indices=torch.arange(K).repeat(B).to(torch.int32))
    cap = 64
    qsl = torch.full((cap + 1,), R * (K + 1), dtype=torch.int32)
    qsl[: R + 1] = torch.arange(R + 1, dtype=torch.int32) * (K + 1)
    chunk = types.SimpleNamespace(
        draft_lengths=md.draft_lengths, input_ids=torch.randint(0, V, (T,), generator=g, dtype=torch.int32),
        attn_ctx=types.SimpleNamespace(query_start_loc=qsl, use_max_model_len=True,
                                       seq_lens=torch.randint(5, 900, (cap,), generator=g, dtype=torch.int32)),
        sequence_layout_plan=types.SimpleNamespace(descriptor=DESC), num_reqs=R)
    return hidden, idx, md, chunk


def slot_groups(seed, G, W, R, nslots=4096):
    g = torch.Generator().manual_seed(9000 + seed)
    slots = (torch.randperm(nslots - 1, generator=g)[: G * W] + 1).to(torch.int32).view(G, W)
    slots[:, R:] = 0
    return [slots[i].clone() for i in range(G)]


def separate(r, groups, tokens, R):
    NS["_update_read_offsets_impl"](r.mamba_slot_read_offsets, list(groups), tokens, torch.tensor(R))


def run_step(r, hidden, idx, md, groups, R, F, seed, chunk=None, vfy=False, mode="vfy", preselected=False):
    """One spec step from sample_tokens' view: the chunk's slot groups, the verify (the D83 chain gets the local
    logits execute_model computed; VFY gets the VfyChunk), then sample_tokens' post-verify statement."""
    r._fused_read_offsets = (groups, R)
    r._fused_read_offsets_applied = False
    if chunk is not None and r.drafter is not None:
        r.drafter.draft_chunks = [chunk]
    gen = torch.Generator().manual_seed(1000 + seed)
    if vfy:
        r._vfy.step_mode = mode
        x = VF.VfyChunk(hidden, None if preselected else idx)
    else:
        x = (r.compute_local_logits_from_hidden_states(hidden) if preselected
             else r.compute_selected_local_logits(hidden, idx))
    out = r._sample_spec_verify_vps(x, md, 0, R, gen, F)
    if not r._fused_read_offsets_applied and groups:
        separate(r, groups, out, R)
    return out, gen.get_state()


def _pair(B, *, cand, mode, synthetic=True, seed=0, asm=None, flagged=True):
    """(D83-chain runner, VFY runner) on equal state. `flagged`: requests 1 and 5 have top_k disabled (>= V), so the
    VPS candidate path cannot cover them and the fallback stats are non-zero (covered with F > 0, else uncovered)."""
    base = Runner(B, cand=cand, synthetic=synthetic, seed=seed)
    vfy = Runner(B, cand=cand, mode=mode, synthetic=synthetic, seed=seed, asm=asm)
    if flagged:
        for r in (base, vfy):
            r.top_k[1] = r.top_k[5] = V
    vfy._vfy.logits_dtype = torch.bfloat16
    vfy._vfy.warm = AlwaysWarm()
    return base, vfy


def _same_state(a, b):
    assert (a._vps_fb_stats is None) == (b._vps_fb_stats is None)
    assert a._vps_fb_stats is None or torch.equal(a._vps_fb_stats, b._vps_fb_stats), (a._vps_fb_stats, b._vps_fb_stats)
    assert torch.equal(a.mamba_slot_read_offsets, b.mamba_slot_read_offsets), \
        int((a.mamba_slot_read_offsets != b.mamba_slot_read_offsets).sum())
    assert a._fused_read_offsets_applied == b._fused_read_offsets_applied
    assert (a._vps_fb_chunks, a._vps_ro_folds) == (b._vps_fb_chunks, b._vps_ro_folds)


# ---------------------------------------------------------------- 1. bitwise identity against the D83 chain

ROWS = [(8, 8), (16, 16), (32, 32), (48, 64), (64, 64)]  # (live requests R, padded rows B)


@pytest.mark.parametrize("R,B", ROWS)
@pytest.mark.parametrize("F", [0, 8])
@pytest.mark.parametrize("cand", [True, False])
@pytest.mark.parametrize("mode", [1, 2])
def test_vfy_equals_the_d83_chain(R, B, F, cand, mode):
    seed = R * 7 + B + F + 3 * cand + 11 * mode
    hidden, idx, md, chunk = problem(seed, B, R)
    groups = slot_groups(seed, 3, 64, R)
    base, vfy = _pair(B, cand=cand, mode=mode, seed=seed)
    o0, s0 = run_step(base, hidden, idx, md, groups, R, F, seed, chunk)
    o1, s1 = run_step(vfy, hidden, idx, md, groups, R, F, seed, chunk, vfy=True)
    assert torch.equal(o0, o1), "tokens differ"
    assert torch.equal(s0, s1), "random stream differs"
    _same_state(base, vfy)
    assert vfy._fused_read_offsets_applied and vfy._vfy.stats["vfy"] == 1 and vfy._vfy.stats["cold"] == 0
    stash = vfy.__dict__.get("_vfy_seed")
    if mode == 2:
        spec, tokens, outs, vers = stash
        assert tokens is o1 and spec.fn is SEED_IMPL and vers is not None
        ref = SEED_IMPL(o0, *spec.args)
        assert len(outs) == len(ref) == 5
        assert all(torch.equal(a, b) for a, b in zip(outs, ref)), "seed outputs differ"
        COUNTS["identity_seed_cases"] += 1
    else:
        assert stash is None
    COUNTS["identity_cases"] += 1


@pytest.mark.parametrize("mode", [1, 2])
def test_stats_accumulate_over_steps(mode):
    B = 16
    base, vfy = _pair(B, cand=True, mode=mode, seed=5)
    for i, (R, F) in enumerate([(12, 8), (16, 0), (9, 8), (16, 8)]):
        hidden, idx, md, chunk = problem(300 + i, B, R)
        groups = slot_groups(300 + i, 3, 64, R)
        o0, s0 = run_step(base, hidden, idx, md, groups, R, F, 300 + i, chunk)
        o1, s1 = run_step(vfy, hidden, idx, md, groups, R, F, 300 + i, chunk, vfy=True)
        vfy.__dict__.pop("_vfy_seed", None)
        assert torch.equal(o0, o1) and torch.equal(s0, s1)
        _same_state(base, vfy)
    assert vfy._vfy.stats["vfy"] == 4 and int(vfy._vps_fb_stats.sum()) == int(base._vps_fb_stats.sum())
    assert int(base._vps_fb_stats[0]) >= 8, base._vps_fb_stats  # 2 flagged rows x 4 steps: the accumulate is not vacuous
    COUNTS["accumulate_steps"] += 4


@pytest.mark.parametrize("synthetic,preselected", [(False, False), (True, True)])
def test_identity_preselected_rows_and_real_acceptance(synthetic, preselected):
    B, R, F = 16, 13, 8
    hidden, idx, md, chunk = problem(41, B, R)
    if preselected:
        hidden = hidden[idx.long()].contiguous()
    groups = slot_groups(41, 3, 64, R)
    base, vfy = _pair(B, cand=True, mode=1, synthetic=synthetic, seed=41)
    o0, s0 = run_step(base, hidden, idx, md, groups, R, F, 41, chunk, preselected=preselected)
    o1, s1 = run_step(vfy, hidden, idx, md, groups, R, F, 41, chunk, vfy=True, preselected=preselected)
    assert torch.equal(o0, o1) and torch.equal(s0, s1)
    _same_state(base, vfy)
    COUNTS["identity_cases"] += 1


# ---------------------------------------------------------------- 2. CPU AOT census


class _OpLog(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops, self.on = [], True

    def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
        if self.on:
            self.ops.append(func._schema.name)
        return func(*args, **(kwargs or {}))


def test_census_two_calls_and_three_eager_ops_become_one_call_and_index_put():
    B, R, F = 16, 12, 8
    hidden, idx, md, chunk = problem(5, B, R)
    groups = slot_groups(5, 3, 64, R)
    res = {}
    for tag in ("d83", "vfy"):
        r = Runner(B, cand=True, mode=1 if tag == "vfy" else 0, seed=5)
        if tag == "vfy":
            r._vfy.logits_dtype, r._vfy.warm = torch.bfloat16, AlwaysWarm()
        r._vps_fb_stats = torch.zeros(3, dtype=torch.int64)  # steady state: the accumulator exists
        r._num_reqs_tensor(R)
        log, calls = _OpLog(), collections.Counter()

        def opaque(name, fn):
            def call(*a, **k):  # a compiled program: its ops are not eager ops
                calls[name] += 1
                log.on = False
                try:
                    return fn(*a, **k)
                finally:
                    log.on = True
            return call
        saved = {n: NS[n] for n in ("_vps_verify_fb_compiled", "_vfy_one_compiled")}
        NS["_vps_verify_fb_compiled"] = opaque("_vps_verify_fb_compiled", saved["_vps_verify_fb_compiled"])
        NS["_vfy_one_compiled"] = opaque("_vfy_one_compiled", saved["_vfy_one_compiled"])
        r.compute_selected_local_logits = opaque("compute_selected_local_logits",
                                                 types.MethodType(M["compute_selected_local_logits"], r))
        try:
            r._fused_read_offsets = (groups, R)
            with log:
                if tag == "vfy":
                    r._vfy.step_mode = "vfy"
                    x = VF.VfyChunk(hidden, idx)
                else:
                    x = r.compute_selected_local_logits(hidden, idx)  # execute_model's call
                r._sample_spec_verify_vps(x, md, 0, R, torch.Generator().manual_seed(3), F)
        finally:
            NS.update(saved)
        res[tag] = (collections.Counter(log.ops), calls)
    (e0, c0), (e1, c1) = res["d83"], res["vfy"]
    assert dict(c0) == {"compute_selected_local_logits": 1, "_vps_verify_fb_compiled": 1}, dict(c0)
    assert dict(c1) == {"_vfy_one_compiled": 1}, dict(c1)
    removed, added = e0 - e1, e1 - e0
    assert dict(removed) == {"aten::_to_copy": 1, "aten::add": 1} and not added, (dict(removed), dict(added))
    assert e0["aten::index_put_"] == e1["aten::index_put_"] == 1
    COUNTS["census_calls_d83"] = sum(c0.values())
    COUNTS["census_calls_vfy"] = sum(c1.values())
    COUNTS["census_eager_removed"] = sum(removed.values())


# ---------------------------------------------------------------- 3. fallbacks to the D83 chain


@pytest.mark.parametrize("case", ["cold_key", "no_groups", "no_warmup"])
def test_fallbacks_run_the_d83_chain_bitwise(case):
    B, R, F = 16, 11, 8
    hidden, idx, md, chunk = problem(61, B, R)
    groups = None if case == "no_groups" else slot_groups(61, 3, 64, R)
    base, vfy = _pair(B, cand=True, mode=2, seed=61)
    if case == "cold_key":
        vfy._vfy.warm = set()
    if case == "no_warmup":
        vfy._vfy.logits_dtype = None
    called = []
    real = NS["_vfy_one_compiled"]
    NS["_vfy_one_compiled"] = lambda *a: called.append(1) or real(*a)
    try:
        o0, s0 = run_step(base, hidden, idx, md, groups, R, F, 61, chunk)
        o1, s1 = run_step(vfy, hidden, idx, md, groups, R, F, 61, chunk, vfy=True)
    finally:
        NS["_vfy_one_compiled"] = real
    assert torch.equal(o0, o1) and torch.equal(s0, s1)
    _same_state(base, vfy)
    assert not called and vfy.__dict__.get("_vfy_seed") is None
    want = {"cold_key": "cold", "no_groups": "no_ro", "no_warmup": "cold"}[case]
    assert vfy._vfy.stats[want] == 1 and vfy._vfy.stats["vfy"] == 0
    COUNTS["fallback_cases"] += 1


# ---------------------------------------------------------------- 4. warm-up covers the steady serving keys


def _steady(r, n, F, width, seed):
    """A steady decode step of n live requests at the shapes the VPS warm-up sweep mirrors."""
    T = NS["_get_padded_token_len"](r.num_tokens_paddings, n * (K + 1))
    B = NS["_get_padded_num_reqs_with_upper_limit"](n, r.max_num_reqs)
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(T, HID, generator=g).bfloat16()
    idx = torch.arange(T, dtype=torch.int32)
    t = torch.arange(T)
    md = types.SimpleNamespace(
        bonus_logits_indices=((torch.arange(B) * (K + 1) + K) % T).to(torch.int32),
        target_logits_indices=t.to(torch.int32), draft_token_ids=torch.randint(0, V, (T,), generator=g,
                                                                                dtype=torch.int32),
        draft_lengths=torch.where(torch.arange(B) < n, K, 0).to(torch.int32),
        segment_ids=(t // K).clamp(max=B - 1).to(torch.int64), group_indices=(t % K).to(torch.int32))
    qsl = torch.full((width + 1,), n * (K + 1), dtype=torch.int32)
    qsl[: n + 1] = torch.arange(n + 1, dtype=torch.int32) * (K + 1)
    chunk = types.SimpleNamespace(
        draft_lengths=md.draft_lengths, input_ids=torch.randint(0, V, (T,), generator=g, dtype=torch.int32),
        attn_ctx=types.SimpleNamespace(query_start_loc=qsl, use_max_model_len=width == r.num_reqs_max_model_len,
                                       seq_lens=torch.zeros(width, dtype=torch.int32)),
        sequence_layout_plan=types.SimpleNamespace(descriptor=DESC), num_reqs=n)
    return hidden, idx, md, chunk, slot_groups(seed, 3, width, n)


@pytest.mark.parametrize("mode,most", [(1, None), (2, None), (2, 16)])
def test_warmup_covers_every_steady_serving_key(mode, most):
    r = Runner(8, cand=True, mode=mode)
    r.max_num_reqs = 20
    r.num_tokens_paddings = [16, 32, 64, 128]
    r.num_reqs_paddings = [8, 16, 32]
    if most:
        r.most_model_len, r.num_reqs_most_model_len = 4096, most
    r._vfy_precompile()
    v = r._vfy
    assert v.logits_dtype == torch.bfloat16 and v.acc_zero is not None and v.programs == len(v.warm)
    fpos = lambda k: next((i for i, e in enumerate(k) if e[0] == "F"), None)  # noqa: E731
    seeded = {k[: fpos(k)] for k in v.warm if fpos(k) is not None}
    plain = {k for k in v.warm if fpos(k) is None}
    assert seeded == plain and plain  # value 1 warms the seed variant of every program too
    G, widths = r._read_offset_fold_geometry()
    variants = VW.fallback_variants(r._vps_fb_max, r._vps_tiegap)
    n_steps = 0
    for n in range(1, r.max_num_reqs + 1):
        for F in variants:
            for wd in widths:
                if n > wd:
                    continue  # a walk of width wd holds at most wd requests
                hidden, idx, md, chunk, groups = _steady(r, n, F, wd, 100 * n + F + wd)
                run_step(r, hidden, idx, md, groups, n, F, n, chunk, vfy=True)
                r.__dict__.pop("_vfy_seed", None)
                n_steps += 1
    assert v.stats["cold"] == 0 and v.stats["vfy"] == n_steps, dict(v.stats)
    COUNTS["warm_programs_" + ("m%d_w%d" % (mode, len(widths)))] = v.programs
    COUNTS["warm_serving_steps"] += n_steps


def test_warmup_respects_the_verify_precompile_cap():
    """VLLM_TPU_SPEC_VERIFY_PRECOMPILE_MAX_SEQS=8: VFY warms n <= 8 only (as the VPS sweep); n = 9 runs cold."""
    r = Runner(8, cand=True, mode=1)
    r.max_num_reqs, r.num_tokens_paddings, r.num_reqs_paddings = 20, [16, 32, 64, 128], [8, 16, 32]
    saved = NS["envs"]
    NS["envs"] = types.SimpleNamespace(**vars(saved), VLLM_TPU_SPEC_VERIFY_PRECOMPILE_MAX_SEQS=8)
    try:
        r._vfy_precompile()
    finally:
        NS["envs"] = saved
    assert r._vfy.programs == 12  # (T, B) in {(16, 8), (32, 8)} x F in {0, 4, 8} x {plain, seed}
    for n, want in ((8, "vfy"), (9, "cold")):
        hidden, idx, md, chunk, groups = _steady(r, n, 0, 64, 500 + n)
        run_step(r, hidden, idx, md, groups, n, 0, n, chunk, vfy=True)
        r.__dict__.pop("_vfy_seed", None)
        assert r._vfy.stats[want] >= 1, (n, dict(r._vfy.stats))
    COUNTS["warm_cap_programs"] = r._vfy.programs


def test_warmup_does_nothing_when_off_or_static_off():
    for r in (Runner(8, mode=0), Runner(8, mode=1, fold=False)):
        r._vfy_precompile()
        assert r._vfy is None or (not r._vfy.warm and r._vfy.static_reason == "no_d83_fold")


# ---------------------------------------------------------------- 5. flags off


def _stmt(fn_name, pred, cls="TPUModelRunner", src=RUNNER_SRC):
    tree = ast.parse(src)
    body = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(n for n in ast.walk(body) if isinstance(n, ast.FunctionDef) and n.name == fn_name)
    nodes = [n for n in ast.walk(fn) if pred(n)]
    assert len(nodes) == 1, (fn_name, len(nodes))
    return compile(textwrap.dedent(ast.get_source_segment(src, nodes[0], padded=True)), f"<{fn_name}>", "exec")


DEFER = _stmt("execute_model", lambda n: isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "vfy_defer")
LOGITS_IF = _stmt("execute_model", lambda n: isinstance(n, ast.If) and ast.unparse(n.test) == "vfy_defer")
HOOK = _stmt("execute_model", lambda n: isinstance(n, ast.If) and ast.unparse(n.test) == "_vfy is not None")
DROP = _stmt("sample_tokens", lambda n: isinstance(n, ast.If)
             and ast.unparse(n.test) == "self.__dict__.get('_vfy_seed') is not None")
INIT = _stmt("_init_speculative_decoding", lambda n: isinstance(n, ast.Assign)
             and ast.unparse(n.targets[0]) == "self._vfy")


def test_flag_off_constructs_no_controller(monkeypatch):
    for raw in (None, "0", ""):
        if raw is None:
            monkeypatch.delenv("VLLM_TPU_VPS_VFY_ONE", raising=False)
        else:
            monkeypatch.setenv("VLLM_TPU_VPS_VFY_ONE", raw)
        me = types.SimpleNamespace()
        exec(INIT, {"_vfy_one": VF, "envs": ENVS_MOD}, {"self": me})
        assert me._vfy is None
    for raw in ("1", "2"):
        monkeypatch.setenv("VLLM_TPU_VPS_VFY_ONE", raw)
        me = types.SimpleNamespace()
        exec(INIT, {"_vfy_one": VF, "envs": ENVS_MOD}, {"self": me})
        assert me._vfy.mode == int(raw)
    monkeypatch.setenv("VLLM_TPU_VPS_VFY_ONE", "3")
    with pytest.raises(ValueError):
        exec(INIT, {"_vfy_one": VF, "envs": ENVS_MOD}, {"self": types.SimpleNamespace()})


class _Plan:
    def __init__(self, pre=None):
        self.pre = pre

    def finalize_hidden_states(self, h, plan):
        return h


def _logits_branch(r, vfy_defer, vps_chunk=True, pre=None):
    hidden = torch.randn(40, HID).bfloat16()
    idx = torch.arange(32, dtype=torch.int32)
    calls = []
    r.compute_selected_local_logits = lambda h, i: calls.append("sel") or ("SEL", h, i)
    r.compute_local_logits_from_hidden_states = lambda h: calls.append("pre") or ("PRE", h)
    loc = {"self": r, "sample_hidden_states": pre, "sequence_layout_planner": _Plan(), "hidden_states": hidden,
           "layout_plan": None, "vfy_defer": vfy_defer, "vps_chunk": vps_chunk, "top_tokens": False,
           "logits_indices": idx}
    exec(LOGITS_IF, {"_vfy_one": VF, "torch": torch}, loc)
    return loc["logits"], calls, hidden, idx


def test_flag_off_statements_keep_the_d83_chain():
    r = Runner(8, mode=0)
    loc = {"self": r, "vps_chunk": True, "spec_decode_metadata": object(), "target_num_chunks": 1}
    exec(DEFER, {}, loc)
    assert loc["vfy_defer"] is False or loc["vfy_defer"] is None
    logits, calls, h, i = _logits_branch(r, loc["vfy_defer"])
    assert calls == ["sel"] and logits[0] == "SEL"
    exec(HOOK, {}, {"_vfy": r.__dict__.get("_vfy"), "self": r})  # no controller: nothing runs
    exec(DROP, {}, {"self": r})
    assert "_vfy_seed" not in r.__dict__


# ---------------------------------------------------------------- 6. execute_model's statements, flag on


def test_defer_gate_and_logits_branch():
    r = Runner(8, mode=1)
    r._vfy.engaged = True
    seen = []
    real = r._vfy.decide
    r._vfy.decide = lambda runner, single: seen.append(single) or real(runner, single)
    for vps_chunk, md, chunks, want in ((True, object(), 1, True), (True, object(), 2, False),
                                        (True, None, 1, False), (False, object(), 1, False)):
        loc = {"self": r, "vps_chunk": vps_chunk, "spec_decode_metadata": md, "target_num_chunks": chunks}
        exec(DEFER, {}, loc)
        assert bool(loc["vfy_defer"]) is want, (vps_chunk, md, chunks)
    assert seen == [True, False]  # decide sees single_chunk; never without a VPS chunk or spec metadata
    logits, calls, h, i = _logits_branch(r, True)
    assert type(logits) is VF.VfyChunk and logits.hidden is h and logits.indices is i and not calls
    pre = torch.randn(32, HID).bfloat16()
    logits, calls, _, _ = _logits_branch(r, True, pre=pre)
    assert type(logits) is VF.VfyChunk and logits.hidden is pre and logits.indices is None and not calls
    logits, calls, _, _ = _logits_branch(r, False, pre=pre)
    assert calls == ["pre"]


def test_step_hook_and_stash_drop():
    r = Runner(8, mode=1)
    begun = []
    r._vfy.begin_step = lambda runner: begun.append(runner)
    exec(HOOK, {}, {"_vfy": r._vfy, "self": r})
    assert begun == [r]
    r._vfy_seed = ("spec", None, ())
    exec(DROP, {}, {"self": r})
    assert "_vfy_seed" not in r.__dict__ and r._vfy.stats["seed_unused"] == 1


def test_engagement_line(caplog):
    B, R = 8, 6
    hidden, idx, md, chunk = problem(71, B, R)
    _, vfy = _pair(B, cand=True, mode=1, seed=71)
    with caplog.at_level(logging.INFO):
        run_step(vfy, hidden, idx, md, slot_groups(71, 3, 64, R), R, 8, 71, chunk, vfy=True)
    lines = [m for m in caplog.messages if m.startswith("q38 VPS_VFY_ONE engaged:")]
    assert len(lines) == 1 and "mode=1 seed=False vfy=1 check=0 cold=0" in lines[0], lines


# ---------------------------------------------------------------- 7. CHECK


@pytest.mark.parametrize("break_", [None, "ro", "seed", "tokens"])
def test_check_runs_the_d83_chain_and_counts_mismatches(break_):
    B, R, F = 16, 14, 8
    hidden, idx, md, chunk = problem(81, B, R)
    groups = slot_groups(81, 3, 64, R)
    base, vfy = _pair(B, cand=True, mode=2, seed=81)
    real = NS["_vfy_one_compiled"]
    calls = []

    def program(*a):
        calls.append(1)
        out = list(real(*a))
        if break_ == "ro":
            out[3] = out[3].roll(1)
        elif break_ == "seed":
            out[4] = out[4] + 1
        elif break_ == "tokens":
            out[0] = out[0].flip(0)
        return tuple(out)
    NS["_vfy_one_compiled"] = program
    try:
        o0, s0 = run_step(base, hidden, idx, md, groups, R, F, 81, chunk)
        o1, s1 = run_step(vfy, hidden, idx, md, groups, R, F, 81, chunk, vfy=True, mode="check")
    finally:
        NS["_vfy_one_compiled"] = real
    assert torch.equal(o0, o1) and torch.equal(s0, s1)  # the step's results are the D83 chain's
    _same_state(base, vfy)
    assert calls == [1] and vfy._vfy.stats["check"] == 1 and vfy.__dict__.get("_vfy_seed") is None
    if break_ is None:
        assert vfy._vfy.check_mismatches == 0
    else:
        assert vfy._vfy.check_mismatches > 0
    COUNTS["check_cases"] += 1


# ---------------------------------------------------------------- 8. DH-C: the seed block takes the verify's seed


def _seed_block():
    """The base fused seed block's seed statement: `if <VFY stash>: seed_out = self._vfy_seed_or_run(...)
    else: seed_out = _async_seed_inputs_compiled(...)` (the else arm is the base statement, verbatim)."""
    tree = ast.parse(EAGLE_SRC)
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "Eagle3Proposer")
    fn = next(n for n in ast.walk(cls) if isinstance(n, ast.FunctionDef) and n.name == "_prepare_draft_inputs")
    ifs = [n for n in ast.walk(fn) if isinstance(n, ast.If)
           and ast.unparse(n.test) == "self.runner.__dict__.get('_vfy_seed') is not None"]
    assert len(ifs) == 1
    return compile(textwrap.dedent(ast.get_source_segment(EAGLE_SRC, ifs[0], padded=True)),
                   "<eagle3 fused seed block>", "exec")


SEED_BLOCK = _seed_block()


def _vfy_seed_step(r, seed=91, B=16, R=13, F=8):
    hidden, idx, md, chunk = problem(seed, B, R)
    out, _ = run_step(r, hidden, idx, md, slot_groups(seed, 3, 64, R), R, F, seed, chunk, vfy=True)
    return out, chunk, R


def _exec_seed_block(p, chunk, nt, R):
    """Run the block's real seed statement for this chunk; (seed_out, the seed program's calls, the seed arguments
    after next_tokens as the block builds them)."""
    calls = []
    spy = lambda *a: calls.append(a) or SEED_IMPL(*a)  # noqa: E731
    saved = ENS.get("_async_seed_inputs_compiled")
    ENS["_async_seed_inputs_compiled"] = spy  # _vfy_seed_or_run's own global
    g = dict(ENS)  # one dict: the block's lambda reads num_reqs as a global
    g.update({"self": p, "nt": nt, "num_draft_full": chunk.draft_lengths, "chunk": chunk, "num_reqs": R, "extras": True})
    try:
        exec(SEED_BLOCK, g)
    finally:
        if saved is None:
            ENS.pop("_async_seed_inputs_compiled", None)
        else:
            ENS["_async_seed_inputs_compiled"] = saved
    args = p._vfy_seed_args(chunk.draft_lengths, chunk.attn_ctx.query_start_loc, R, chunk.input_ids,
                            p._draft_rows_len(chunk), True)
    return g["seed_out"], calls, args


def test_seed_block_takes_the_verify_seed_on_identity():
    _, r = _pair(16, cand=True, mode=2, seed=91)
    out, chunk, R = _vfy_seed_step(r)
    spec, tokens, outs, vers = r._vfy_seed
    seed_out, calls, args = _exec_seed_block(r.drafter, chunk, out, R)
    assert seed_out is outs and not calls  # census: the drafter's seed program does not run
    assert "_vfy_seed" not in r.__dict__ and r._vfy.stats["seed_taken"] == 1
    assert all(torch.equal(a, b) for a, b in zip(seed_out, SEED_IMPL(out, *args)))
    COUNTS["seed_taken"] += 1


@pytest.mark.parametrize("miss", ["tokens", "input_ids", "num_rows", "no_stash"])
def test_seed_block_misses_fall_back_to_its_program(miss):
    _, r = _pair(16, cand=True, mode=2, seed=92)
    out, chunk, R = _vfy_seed_step(r, seed=92)
    nt = out
    if miss == "tokens":
        nt = out.clone()
    elif miss == "input_ids":
        chunk.input_ids = chunk.input_ids.clone()
    elif miss == "num_rows":
        r.num_reqs_max_model_len = 32  # the chunk's draft-rows capacity changed after the verify
    elif miss == "no_stash":
        r.__dict__.pop("_vfy_seed")
    seed_out, calls, args = _exec_seed_block(r.drafter, chunk, nt, R)
    assert len(calls) == 1 and calls[0][0] is nt and all(x is y for x, y in zip(calls[0][1:], args) if
                                                          isinstance(x, torch.Tensor))
    assert all(torch.equal(a, b) for a, b in zip(seed_out, SEED_IMPL(nt, *args)))
    if miss != "no_stash":
        assert r._vfy.stats["seed_miss"] == 1 and "_vfy_seed" in r.__dict__
    COUNTS["seed_miss_cases"] += 1


@pytest.mark.parametrize("write", ["input_ids", "query_start_loc", "draft_lengths", "tokens"])
def test_seed_input_written_in_place_after_the_verify_is_caught(write):
    """An in-place write to a seed input (or the verify's tokens) between the verify and the seed block: the
    block must not take the stale stash (its version counter moved); it runs its own program on the written
    values, so the seed is still exact."""
    _, r = _pair(16, cand=True, mode=2, seed=98)
    out, chunk, R = _vfy_seed_step(r, seed=98)
    stale = r._vfy_seed[2]
    if write == "input_ids":
        chunk.input_ids[:] = chunk.input_ids.roll(1)
    elif write == "query_start_loc":
        chunk.attn_ctx.query_start_loc[1:] -= 1
    elif write == "draft_lengths":
        chunk.draft_lengths[:] = 0
    else:
        out[:, 1:] = -1
    seed_out, calls, args = _exec_seed_block(r.drafter, chunk, out, R)
    assert len(calls) == 1 and r._vfy.stats["seed_written"] == 1 and r._vfy.stats["seed_taken"] == 0
    fresh = SEED_IMPL(out, *args)
    assert all(torch.equal(a, b) for a, b in zip(seed_out, fresh))
    assert any(not torch.equal(a, b) for a, b in zip(stale, fresh)), "the write must matter for this test to bite"
    COUNTS["seed_written_cases"] += 1


def test_flag_off_seed_block_calls_its_program_with_the_same_arguments():
    r = Runner(16, mode=0)
    hidden, idx, md, chunk = problem(93, 16, 13)
    nt = torch.randint(-1, V, (16, K + 1), dtype=torch.int32)
    seed_out, calls, args = _exec_seed_block(r.drafter, chunk, nt, 13)
    n_dev = r.drafter._device_constant(("num_reqs", 13), None)
    want = (nt, chunk.draft_lengths, chunk.attn_ctx.query_start_loc, n_dev, chunk.input_ids, 16, 64)
    assert len(calls) == 1 and len(calls[0]) == len(want)
    assert all((x is y) if isinstance(y, torch.Tensor) else x == y for x, y in zip(calls[0], want))


# ---------------------------------------------------------------- 9. DH-A composition: one seed fold per step


def _asm_superset_impl(next_tokens, num_draft, query_start_loc, num_reqs, input_ids, seq_lens, p, num_rows, max_reqs):
    """A DRAFT-ASM-shaped superset seed body (q38-l4-draftasm's `_async_seed_inputs_asm_impl` shape: the base five
    outputs, then step 1's seq_lens and two tail rows); the real one lives on DRAFTASM-BUILD's branch."""
    out = SEED_IMPL(next_tokens, num_draft, query_start_loc, num_reqs, input_ids, p, num_rows)
    seq_lens_step1 = seq_lens + 1 - out[4]
    bonus = torch.nn.functional.pad(next_tokens[:, 0], (0, max(0, max_reqs - next_tokens.shape[0])))[:max_reqs]
    rej = torch.nn.functional.pad(out[2], (0, max(0, max_reqs - out[2].shape[0])))[:max_reqs]
    return (*out, seq_lens_step1, bonus, rej)


class FakeAsm:
    """DraftAsm with the two proposed hooks and its fold-mode seed taking the verify's superset."""

    def __init__(self):
        self.p, self.max_reqs, self.own = None, 64, 0

    def _args(self, base, seq_lens):
        return (*base[:4], seq_lens, base[4], base[5], self.max_reqs)

    def vfy_seed_spec(self, chunk, num_reqs):
        p = self.p
        base = p._vfy_seed_args(chunk.draft_lengths, chunk.attn_ctx.query_start_loc, num_reqs, chunk.input_ids,
                                p._draft_rows_len(chunk), True)
        return VF.VfySeedSpec(_asm_superset_impl, self._args(base, chunk.attn_ctx.seq_lens), base)

    def vfy_seed_warm_spec(self, T, B, cap, n):
        p = self.p
        base = p._vfy_seed_args(torch.zeros(B, dtype=torch.int32), torch.zeros(cap + 1, dtype=torch.int32), n,
                                torch.zeros(T, dtype=torch.int32), cap, True)
        return VF.VfySeedSpec(_asm_superset_impl, self._args(base, torch.zeros(cap, dtype=torch.int32)), base)

    def seed(self, chunk, nt, args):
        out = self.p._vfy_seed_take(chunk, nt, _asm_superset_impl, args)
        if out is None:
            self.own += 1
            out = _asm_superset_impl(nt, *args)
        return out


def _asm_runner(seed):
    asm = FakeAsm()
    _, r = _pair(16, cand=True, mode=2, seed=seed, asm=asm)
    asm.p = r.drafter
    return r, asm


def test_dha_fold_mode_adopts_the_verify_superset_seed():
    r, asm = _asm_runner(94)
    out, chunk, R = _vfy_seed_step(r, seed=94)
    spec, tokens, outs, vers = r._vfy_seed
    assert spec.fn is _asm_superset_impl and len(outs) == 8
    ref = _asm_superset_impl(out, *spec.args)
    assert all(torch.equal(a, b) for a, b in zip(outs, ref))
    got = asm.seed(chunk, out, spec.args)
    assert got is outs and asm.own == 0  # one seed fold per step: the verify's
    COUNTS["dha_adopted"] += 1


def test_dha_seq_lens_written_in_place_after_the_verify_is_caught():
    """L4-E's open item on the composed tree: step 1's seq_lens come from chunk.attn_ctx.seq_lens; an in-place write
    to it between the verify and DRAFT-ASM's seed makes the fold-mode take miss, and DraftAsm runs its own seed on
    the written values."""
    r, asm = _asm_runner(99)
    out, chunk, R = _vfy_seed_step(r, seed=99)
    spec, tokens, stale, vers = r._vfy_seed
    chunk.attn_ctx.seq_lens.add_(1)
    got = asm.seed(chunk, out, spec.args)
    assert asm.own == 1 and r._vfy.stats["seed_written"] == 1
    fresh = _asm_superset_impl(out, *spec.args)
    assert all(torch.equal(a, b) for a, b in zip(got, fresh)) and not torch.equal(stale[5], fresh[5])
    COUNTS["dha_seq_lens_caught"] += 1


def test_dha_base_path_takes_the_superset_first_five():
    r, asm = _asm_runner(95)
    out, chunk, R = _vfy_seed_step(r, seed=95)
    outs = r._vfy_seed[2]
    seed_out, calls, args = _exec_seed_block(r.drafter, chunk, out, R)  # DraftAsm not folding this step
    assert seed_out is outs and not calls
    assert all(torch.equal(a, b) for a, b in zip(seed_out[:5], SEED_IMPL(out, *args)))


def test_dha_without_hooks_keeps_its_own_seed():
    class OldAsm:  # today's DRAFTASM head: no vfy_seed_spec
        pass
    _, r = _pair(16, cand=True, mode=2, seed=96, asm=OldAsm())
    out, chunk, R = _vfy_seed_step(r, seed=96)
    assert r.__dict__.get("_vfy_seed") is None and r._vfy.seed_reason == "draft_asm_without_vfy_seed_hook"
    assert r._vfy.stats["vfy"] == 1  # still one verify program (value 1 behaviour)


def test_dha_warmup_warms_the_superset_variant():
    r, asm = _asm_runner(97)
    r.max_num_reqs = 8
    r.num_tokens_paddings = [16, 32, 64]
    r.num_reqs_paddings = [8]
    r._vfy.warm = set()
    r._vfy_precompile()
    names = {e[2] for k in r._vfy.warm for e in k if e[0] == "F"}
    assert names == {"_asm_superset_impl"}


def test_zz_counts():
    out = os.environ.get("Q38_L4VFY_COUNTS")
    if out:
        with open(out, "w") as f:
            json.dump(dict(COUNTS), f, sort_keys=True)
    print("Q38_L4VFY_COUNTS " + json.dumps(dict(COUNTS), sort_keys=True))
