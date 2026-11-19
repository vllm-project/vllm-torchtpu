# SPDX-License-Identifier: Apache-2.0
"""VPS wiring: runner/vps_wiring.py, the runner's VPS verify / check / dispatch, the draft's top-tokens path, and the
compile-cache keying. Torch CPU, no TPU, no vLLM install: packages are stubbed and the runner / draft methods are
extracted from the real source with ast.

The integration test runs the runner's own `_sample_spec_verify_vps` against its own `_sample_spec_verify_fused`
(both with eager bodies) from the same generator seed: same tokens and the same random stream.
Run: python -m pytest -q -p no:cacheprovider --noconftest tests/q38mm_dstock/test_vps_wiring.py
Mutation-checked by q38mm_vps_wiring_mutrun.py (Q38MM_SRC_ROOT points at a mutated copy).
"""

import ast
import contextlib
import importlib.util
import logging
import os
import sys
import textwrap
import types

import pytest
import torch

ROOT = os.environ.get("Q38MM_SRC_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
PKG = os.path.join(ROOT, "vllm_torchtpu")
RUNNER = os.path.join(PKG, "runner", "tpu_runner.py")
EAGLE = os.path.join(PKG, "spec_decode", "eagle3.py")
LOOP = os.path.join(PKG, "spec_decode", "mtp_fused_loop.py")
TC = os.path.join(PKG, "compilation", "tpu_compiler.py")


class HarnessModules:
    """The sys.modules entries this harness installs: stub vllm / vllm_torchtpu packages and the modules it loads from
    the tree. They are present only while the harness loads (module import) and while this file's tests run (autouse
    module fixture); otherwise the interpreter's own entries are back, so one pytest process can collect and run real
    vllm tests next to this file (INT-4's union job i5 died with 1165 setup errors when these stubs leaked).
    While the tests run, a stub package never shadows a real vllm package that is already imported (the conftest's
    autouse fixture imports vllm.logger around every test); leaf stubs (vllm.distributed) and the vllm_torchtpu
    entries do shadow, and everything is put back afterwards. Checked by tests/decfb2/test_vps_harness_no_leak.py."""

    TREES = ("vllm", "vllm_torchtpu")

    def __init__(self):
        self.mods = {}

    @classmethod
    def snapshot(cls):
        return {n: m for n, m in list(sys.modules.items()) if n.split(".")[0] in cls.TREES}

    @classmethod
    def restore(cls, saved):
        """Back to `saved` under the two trees: drop what was added, put back what was replaced."""
        for n in [n for n in list(sys.modules) if n.split(".")[0] in cls.TREES and n not in saved]:
            del sys.modules[n]
        sys.modules.update(saved)

    @contextlib.contextmanager
    def loading(self):
        saved = self.snapshot()
        try:
            yield
            self.mods = {n: m for n, m in self.snapshot().items() if saved.get(n) is not m}
        finally:
            self.restore(saved)

    @contextlib.contextmanager
    def active(self):
        saved = self.snapshot()
        for n, m in self.mods.items():
            real = saved.get(n)
            if n.split(".")[0] == "vllm" and getattr(m, "__path__", None) == [] and getattr(real, "__file__", None):
                continue                    # keep the imported real vllm package; only its leaves are stubbed
            sys.modules[n] = m
        try:
            yield
        finally:
            self.restore(saved)


def _pkg(name):
    m = types.ModuleType(name)
    m.__path__ = []
    sys.modules[name] = m
    return m


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


GATHER = {"fn": lambda x, dim=-1: x}      # TP1: the all-gather is the identity


def _setup():
    for n in ("vllm", "vllm.v1", "vllm.v1.spec_decode", "vllm_torchtpu", "vllm_torchtpu.layers",
              "vllm_torchtpu.layers.adapter", "vllm_torchtpu.layers.adapter.sample", "vllm_torchtpu.runner"):
        _pkg(n)
    utils = types.ModuleType("vllm.v1.spec_decode.utils")
    utils.unconditional_to_conditional_rates = lambda r: r
    sys.modules["vllm.v1.spec_decode.utils"] = utils
    dist = types.ModuleType("vllm.distributed")
    dist.tensor_model_parallel_all_gather = lambda x, dim=-1: GATHER["fn"](x, dim)
    sys.modules["vllm.distributed"] = dist
    real_compile = torch.compile
    torch.compile = lambda fn=None, **kw: fn if callable(fn) else (lambda f: f)
    try:
        s = os.path.join(PKG, "layers", "adapter", "sample")
        tk = _load("vllm_torchtpu.layers.adapter.sample.top_k_top_p", os.path.join(s, "top_k_top_p.py"))
        rs = _load("vllm_torchtpu.layers.adapter.sample.rejection_sampler", os.path.join(s, "rejection_sampler.py"))
        vp = _load("vllm_torchtpu.layers.adapter.sample.vp_sampler", os.path.join(s, "vp_sampler.py"))
        sys.modules["vllm_torchtpu.layers.adapter.sample"].vp_sampler = vp
        # PD stack: the VPS flags are registered in envs.py (stdlib only), which vps_wiring reads.
        sys.modules["vllm_torchtpu"].envs = _load("vllm_torchtpu.envs", os.path.join(PKG, "envs.py"))
        vw = _load("vllm_torchtpu.runner.vps_wiring", os.path.join(PKG, "runner", "vps_wiring.py"))
    finally:
        torch.compile = real_compile
    return tk, rs, vp, vw


HARNESS = HarnessModules()
with HARNESS.loading():
    TK, RS, VP, VW = _setup()


@pytest.fixture(autouse=True, scope="module")
def _harness_modules_installed():
    """The runner's lazy imports (vllm.distributed, vllm_torchtpu.envs, vp_sampler) resolve to the harness."""
    with HARNESS.active():
        yield


def _fns(path, names, ns, cls=None):
    src = open(path).read()
    tree = ast.parse(src)
    body = tree.body
    if cls:
        body = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls).body
    for name in names:
        node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(textwrap.dedent(ast.get_source_segment(src, node)), ns)
    return [ns[n] for n in names]


def _runner_ns():
    src = open(RUNNER).read()
    ns = {"torch": torch, "apply_top_k_top_p": TK.apply_top_k_top_p,
          "_random_rejection_sample_impl": RS._random_rejection_sample_impl, "INVALID_TOKEN_ID": -1,
          "SpecDecodeMetadata": object, "logger": logging.getLogger("t-vps"),
          "host_phases": types.SimpleNamespace(mark=lambda *_: None)}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "SAMPLING_EPS" for t in node.targets):
            exec(ast.get_source_segment(src, node), ns)
    _fns(RUNNER, ["_apply_temperature_impl", "_sample_from_logits_impl", "_verify_fused_impl", "_vps_verify_impl"], ns)
    ns["_verify_fused_compiled"] = ns["_verify_fused_impl"]
    ns["_vps_verify_compiled"] = ns["_vps_verify_impl"]
    ns["envs"] = types.SimpleNamespace(VLLM_TPU_VERIFY_FUSED=True, VLLM_TPU_VERIFY_FUSED_READ_OFFSETS=False,
                                       VLLM_TPU_FUSED_READ_OFFSET_UPDATE=False, VLLM_TPU_VERIFY_FUSED_CHECK_FAULT=0)
    return ns


NS = _runner_ns()
FUSED, VPSV, CHECK, CHUNK = _fns(RUNNER, ["_sample_spec_verify_fused", "_sample_spec_verify_vps", "_check_vps",
                                          "_sample_spec_verify_chunk"], NS, cls="TPUModelRunner")

B, K, V = 8, 3, 10240


def problem(seed):
    g = torch.Generator().manual_seed(seed)
    N, T = B * (K + 1), B * K
    logits = (torch.randn(N, V, generator=g) * 3)
    logits.scatter_add_(1, torch.randint(0, V, (N, 6), generator=g), torch.rand(N, 6, generator=g) * 8)
    logits = logits.bfloat16()
    rows = torch.arange(N).view(B, K + 1)
    am = logits[rows[:, :K].reshape(-1)].float().argmax(-1)
    draft = torch.where(torch.rand(T, generator=g) < 0.5, am, torch.randint(0, V, (T,), generator=g)).to(torch.int32)
    md = types.SimpleNamespace(bonus_logits_indices=rows[:, K].contiguous(), target_logits_indices=rows[:, :K].reshape(-1).contiguous(),
                               draft_token_ids=draft, draft_lengths=torch.full((B,), K, dtype=torch.int32),
                               segment_ids=torch.arange(B).repeat_interleave(K).to(torch.int32),
                               group_indices=torch.arange(K).repeat(B).to(torch.int32))
    return logits, md


class FakeRunner:
    def __init__(self, checks=0, faults=0):
        self.rejection_sampler = types.SimpleNamespace(synthetic_conditional_rates=torch.tensor([1.0, 0.32, 0.0]),
                                                       synthetic_mode=True)
        self.speculative_config = types.SimpleNamespace(num_speculative_tokens=K)
        self._fused_read_offsets = (None, 0)
        self.mamba_slot_read_offsets = None
        self._verify_fused_checks_left = 0
        self._vps_head = VW.VpsHead(None, None, 0, V, V, 1)
        self._vps_checks_left, self._vps_check_faults, self._vps_checks_done = checks, faults, 0
        self._vps_mismatches = None
        # PD DEC-FB-2 fallback / draw flags, off (the PD tests in tests/decfb2 turn them on).
        self._vps_fb_path, self._vps_fb_max, self._vps_tiegap, self._vps_cand_draws = False, 0, 0, False
        self._vps_fb_stats, self._vps_fb_chunks = None, 0
        self.calls = []

    def _padded_sampling_params(self, s, e, n, dtype, device):
        return (torch.full((n, 1), 0.6, dtype=dtype), torch.full((n, 1), 20, dtype=torch.int32),
                torch.full((n, 1), 0.95, dtype=torch.float32))

    def _sampler_topk_candidates(self, s, e, grammar):
        return 64

    _sample_spec_verify_fused = FUSED
    _sample_spec_verify_vps = VPSV
    _check_vps = CHECK


# ---- runner: the VPS verify equals the stock fused verify, tokens and random stream ----
@pytest.mark.parametrize("seed", range(4))
def test_vps_verify_equals_fused_and_keeps_the_rng_stream(seed):
    logits, md = problem(seed)
    r = FakeRunner()
    g1, g2 = torch.Generator().manual_seed(100 + seed), torch.Generator().manual_seed(100 + seed)
    a = r._sample_spec_verify_fused(logits, md, 0, B, g1)
    b = r._sample_spec_verify_vps(logits, md, 0, B, g2)
    assert torch.equal(a, b)
    assert torch.equal(g1.get_state(), g2.get_state())     # same draws, same order, same widths


def test_vps_uniforms_use_the_full_vocab_width():
    """The draws are at the full vocabulary width even though the chunk's logits are one shard."""
    logits, md = problem(9)
    r = FakeRunner()
    seen = []
    real = torch.rand

    def spy(*a, **kw):
        seen.append(tuple(a[0]) if a and isinstance(a[0], (tuple, torch.Size)) else a)
        return real(*a, **kw)
    torch.rand = spy
    try:
        r._vps_head = VW.VpsHead(None, None, 0, V, 2 * V, 1)    # one shard of a vocabulary twice as wide
        r._sample_spec_verify_vps(logits, md, 0, B, torch.Generator().manual_seed(1))
    finally:
        torch.rand = real
    assert seen == [(B, 2 * V), tuple(md.draft_token_ids.shape), (B * K, 2 * V)]


# ---- runner: the identity check ----
def test_check_clean_and_fault_control():
    logits, md = problem(3)
    r = FakeRunner(checks=3, faults=1)
    for _ in range(3):
        r._sample_spec_verify_vps(logits, md, 0, B, torch.Generator().manual_seed(5))
    clean, caught = r._vps_mismatches.tolist()
    assert clean == 0 and caught == 1 and r._vps_checks_left == 0 and r._vps_checks_done == 3


def test_check_sees_a_wrong_verify(monkeypatch):
    logits, md = problem(4)
    r = FakeRunner(checks=1, faults=0)
    real = NS["_vps_verify_compiled"]
    NS["_vps_verify_compiled"] = lambda *a, **k: real(*a, **k) * 0 + 7      # a broken VPS program
    try:
        r._sample_spec_verify_vps(logits, md, 0, B, torch.Generator().manual_seed(5))
    finally:
        NS["_vps_verify_compiled"] = real
    clean, caught = r._vps_mismatches.tolist()
    assert clean > 0 and caught == 0


def test_no_check_when_off():
    logits, md = problem(5)
    r = FakeRunner(checks=0)
    r._sample_spec_verify_vps(logits, md, 0, B, torch.Generator().manual_seed(5))
    assert r._vps_mismatches is None


# ---- runner: chunk dispatch ----
def test_chunk_dispatch():
    logits, md = problem(6)
    r = FakeRunner()
    r._sample_spec_verify_vps = lambda *a, **k: "VPS"
    out = CHUNK(r, logits, md, None, None, 0, B, False, torch.Generator().manual_seed(1), False, vps=True)
    assert out == "VPS"

    def boom(*a, **k):
        raise AssertionError("vps path taken with vps=False")
    r._sample_spec_verify_vps = boom
    r.rejection_sampler = types.SimpleNamespace(synthetic_mode=True, synthetic_conditional_rates=torch.tensor([1.0, .32, 0.]))
    r.spec_gather_bonus_and_target_logits = lambda lg, bi, ti: (lg[bi], lg[ti])
    r._build_padded_sampling_params = lambda s, e, lg: r._padded_sampling_params(s, e, lg.shape[0], lg.dtype, None)
    r.sample_from_logits_func = lambda *a, **k: torch.zeros(B, 1, dtype=torch.int64)
    r.rejection_sampler.__call__ = None
    called = {}

    class RS_:
        synthetic_mode = True
        synthetic_conditional_rates = torch.tensor([1.0, .32, 0.])

        def __call__(self, **kw):
            called["stock"] = True
            return "STOCK"
    r.rejection_sampler = RS_()
    assert CHUNK(r, logits, md, None, None, 0, B, False, torch.Generator().manual_seed(1), False) == "STOCK"
    assert called.get("stock")


# ---- vps_wiring helpers ----
def test_flags(monkeypatch):
    for n in (VW.VPS_VERIFY_ENV, VW.DRAFT_TOP_TOKENS_ENV, VW.VPS_CHECK_ENV, VW.VPS_CHECK_FAULT_ENV):
        monkeypatch.delenv(n, raising=False)
    assert not VW.verify_enabled() and not VW.draft_top_tokens_enabled() and VW.check_counts() == (0, 0)
    monkeypatch.setenv(VW.VPS_VERIFY_ENV, "1")
    monkeypatch.setenv(VW.VPS_CHECK_ENV, "200")
    monkeypatch.setenv(VW.VPS_CHECK_FAULT_ENV, "2")
    assert VW.verify_enabled() and VW.check_counts() == (200, 2)
    monkeypatch.setenv(VW.DRAFT_TOP_TOKENS_ENV, "1")
    assert VW.draft_top_tokens_enabled()              # q38 DEC-FB-3 lever 2: the draft sites are ported
    monkeypatch.setenv(VW.VPS_VERIFY_ENV, "maybe")
    with pytest.raises(ValueError):
        VW.verify_enabled()
    monkeypatch.setenv(VW.VPS_VERIFY_ENV, "1")
    monkeypatch.setenv(VW.VPS_CHECK_ENV, "-1")
    with pytest.raises(ValueError):
        VW.check_counts()


class _Proc:
    def __init__(self, scale=1.0, soft_cap=None, logits_as_input=False, org=248320):
        self.scale, self.soft_cap, self.logits_as_input, self.org_vocab_size = scale, soft_cap, logits_as_input, org
        self.calls = []

    def get_top_tokens(self, head, h):
        return None

    def _get_logits(self, h, head, bias, skip_gather=False):
        self.calls.append(skip_gather)
        return h


def _head(start=31040, width=31040, tp=8, pad=0):
    si = types.SimpleNamespace(org_vocab_start_index=start, org_vocab_end_index=start + width, num_org_vocab_padding=pad)
    return types.SimpleNamespace(shard_indices=si, tp_size=tp)


def _model(proc, head, plain=True):
    class M(torch.nn.Module):
        plain_compute_logits = plain

        def compute_logits(self, h):
            return h
    m = M()
    m.logits_processor, m.lm_head = proc, head
    return m


def test_resolve_head_ok_and_each_refusal():
    vh, why = VW.resolve_vps_head(_model(_Proc(), _head()), 160, 32)
    assert vh is not None and why == "" and (vh.shard_offset, vh.shard_width, vh.full_vocab, vh.tp_size) == (31040, 31040, 248320, 8)
    bad = [_model(_Proc(soft_cap=30.0), _head()), _model(_Proc(scale=0.5), _head()), _model(_Proc(logits_as_input=True), _head()),
           _model(_Proc(), _head(tp=1)), _model(_Proc(), _head(pad=8)), _model(_Proc(), _head(width=31000)),
           _model(_Proc(), _head(width=160 * 31))]
    for m in bad:
        vh, why = VW.resolve_vps_head(m, 160, 32)
        assert vh is None and why
    m = _model(_Proc(), _head(), plain=False)
    type(m).compute_logits.__module__ = "vllm_torchtpu.models.x"
    assert VW.resolve_vps_head(m, 160, 32)[0] is None
    assert VW.resolve_vps_head(torch.nn.Linear(2, 2), 160, 32)[0] is None


def test_local_logits_skips_the_gather():
    p = _Proc()
    vh = VW.VpsHead(p, _head(), 0, 31040, 248320, 8)
    VW.local_logits(vh, torch.zeros(2, 4))
    assert p.calls == [True]


def test_step_eligible():
    req = lambda so=None: types.SimpleNamespace(sampling_params=types.SimpleNamespace(structured_outputs=so))  # noqa: E731
    ib = lambda g=False, lp=None: types.SimpleNamespace(all_greedy=g, max_num_logprobs=lp, req_ids=["a", None, "b"])  # noqa: E731
    reqs = {"a": req(), "b": req()}
    assert VW.step_eligible(ib(), reqs)
    assert not VW.step_eligible(ib(g=True), reqs)
    assert not VW.step_eligible(ib(lp=5), reqs)
    assert not VW.step_eligible(ib(), {"a": req(), "b": req(so=object())})


def test_draft_top_tokens_ok():
    good = _model(_Proc(), _head())
    assert VW.draft_top_tokens_ok(good)
    assert not VW.draft_top_tokens_ok(_model(_Proc(), _head(), plain=False))
    d2t = _model(_Proc(), _head())
    d2t.draft_id_to_target_id = torch.zeros(1)
    assert not VW.draft_top_tokens_ok(d2t)
    assert not VW.draft_top_tokens_ok(None)


# ---- draft dispatch (eagle3 and the fused loop) ----
class _DraftProc:
    def get_top_tokens(self, head, h):
        return torch.full((h.shape[0],), 7, dtype=torch.int64)


def _draft(dvp):
    dm = types.SimpleNamespace(draft_id_to_target_id=None, logits_processor=_DraftProc(), lm_head=object(),
                               compute_logits=lambda h: torch.nn.functional.one_hot(torch.full((h.shape[0],), 3), 10).float())
    return types.SimpleNamespace(draft_model=dm, _dvp=dvp)


def test_eagle_propose_sites():
    # q38 DEC-FB-3 lever 2: PD's _draft_propose_token delegates to _propose_token_body (dongruizi 48745d9b0), and
    # lever 3's _carry_and_token_body is a second copy of the carry body; all three token sites switch.
    ns = {"torch": torch, "vps_wiring": VW, "_static_shape_args": lambda f: f}
    prop, carry, body = _fns(EAGLE, ["_propose_token_body", "_draft_rows_carry_and_token", "_carry_and_token_body"],
                             ns, cls="Eagle3Proposer")
    h = torch.zeros(4, 8)
    assert prop(_draft(True), h).tolist() == [7] * 4 and prop(_draft(True), h).dtype == torch.int32
    assert prop(_draft(False), h).tolist() == [3] * 4
    pos, idx = torch.arange(4), torch.arange(4)
    for fn in (carry, body):
        assert fn(_draft(True), pos, idx, h)[1].tolist() == [7] * 4
        assert fn(_draft(True), pos, idx, h)[1].dtype == torch.int32
        assert fn(_draft(False), pos, idx, h)[1].tolist() == [3] * 4
    # a proposer-like object without _dvp (the port tests' stubs) keeps the stock argmax
    bare = types.SimpleNamespace(draft_model=_draft(True).draft_model)
    assert prop(bare, h).tolist() == [3] * 4
    assert carry(bare, pos, idx, h)[1].tolist() == [3] * 4 and body(bare, pos, idx, h)[1].tolist() == [3] * 4


def test_fused_loop_forward():
    ns = {"torch": torch, "IntermediateTensors": object}
    (fwd,) = _fns(LOOP, ["forward"], ns, cls="_MtpFusedLoop")

    class Mtp:
        logits_processor = _DraftProc()
        lm_head = object()

        def forward(self, input_ids, positions, hidden, it, emb, spec_step_idx=0):
            return hidden, hidden

        def compute_logits(self, h):
            return torch.nn.functional.one_hot(torch.full((h.shape[0],), 3), 10).float()
    for dvp, want in ((True, 7), (False, 3)):
        me = types.SimpleNamespace(mtp=Mtp(), num_steps=2, dvp=dvp)
        toks = fwd(me, torch.zeros(4, dtype=torch.int32), torch.zeros(4), torch.zeros(4, 8), torch.zeros(4))
        assert [t.tolist() for t in toks] == [[want] * 4] * 2


# ---- the get_top_tokens reduction is argmax over the full row (vLLM 0.32.0's code, replicated) ----
def _vllm_top_tokens(shards, starts):
    """vllm/model_executor/layers/logits_processor.py get_top_tokens (v0.32.0 :231-251), simulated over TP shards."""
    pairs = []
    for lg, st in zip(shards, starts):
        v, i = lg.max(dim=-1)
        pairs.append(torch.stack([v.float(), (i + st).float()], dim=-1))
    gathered = torch.cat(pairs, dim=-1).view(shards[0].shape[0], len(shards), 2)
    r = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
    return gathered[:, :, 1].gather(dim=-1, index=r).squeeze(-1).to(torch.int64)


def test_top_tokens_reduction_is_exact_including_ties():
    g = torch.Generator().manual_seed(0)
    full = (torch.randn(16, 8 * 1280, generator=g) * 3).bfloat16()
    m = full.float().max() + 4
    full[0, 3 * 1280 + 5] = m; full[0, 1 * 1280 + 9] = m          # cross-shard tie: the lower id wins
    full[1, 77] = m; full[1, 78] = m                               # within-shard tie
    shards = list(full.split(1280, dim=-1))
    got = _vllm_top_tokens(shards, [r * 1280 for r in range(8)])
    assert torch.equal(got, full.argmax(dim=-1))
    assert got[0].item() == 1280 + 9 and got[1].item() == 77


# ---- compile-cache factors ----
def _real_factors():
    """`_tpu_compile_env_factors` and `_TPU_COMPILE_ENV_IGNORED` from the tree's tpu_compiler, over the real envs."""
    src = open(TC).read()
    tree = ast.parse(src)
    ns = {"envs": sys.modules["vllm_torchtpu.envs"], "logger": logging.getLogger("t"), "os": os, "Any": object}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "_TPU_COMPILE_ENV_IGNORED" for t in node.targets):
            exec(ast.get_source_segment(src, node), ns)
    (f,) = _fns(TC, ["_tpu_compile_env_factors"], ns)
    return f


def test_factors_vps_registered(monkeypatch):
    """PD stack: the program-changing VPS flags are registered and hashed (with their defaults when unset); the
    diagnostics are registered and ignored."""
    f = _real_factors()
    for n in VW.HASHED_ENVS + VW.IGNORED_ENVS:
        monkeypatch.delenv(n, raising=False)
    got = f()
    assert {n: got.get(n, "ABSENT") for n in VW.HASHED_ENVS} == {
        "VLLM_TPU_VPS_VERIFY": False, "VLLM_TPU_VPS_FALLBACK_ROWS": 0, "VLLM_TPU_VPS_TIEGAP_SLOTS": 0,
        "VLLM_TPU_VPS_DRAWS": "full", "VLLM_TPU_DRAFT_TOP_TOKENS": False}
    assert not any(n in got for n in VW.IGNORED_ENVS)
    monkeypatch.setenv("VLLM_TPU_VPS_VERIFY", "1")
    monkeypatch.setenv("VLLM_TPU_VPS_DRAWS", "candidate")
    monkeypatch.setenv("VLLM_TPU_VPS_CHECK", "64")
    got = f()
    assert got["VLLM_TPU_VPS_VERIFY"] is True and got["VLLM_TPU_VPS_DRAWS"] == "candidate"
    assert "VLLM_TPU_VPS_CHECK" not in got


# ---- execute_model's VPS decisions, evaluated from the real source ----
def _exec_model_nodes():
    src = open(RUNNER).read()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "execute_model")
    assigns = {}
    for n in ast.walk(fn):      # the first (outermost) assignment; PD's fallback re-assigns vps_chunk inside an if
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("vps_step", "vps_chunk"):
            assigns.setdefault(n.targets[0].id, "(" + ast.get_source_segment(src, n.value) + ")")
    disp = next(n for n in ast.walk(fn) if isinstance(n, ast.If) and ast.get_source_segment(src, n.test) == "sample_hidden_states is None")
    lines = src.splitlines()[disp.lineno - 1:disp.end_lineno]
    return assigns, textwrap.dedent("\n".join(lines))


ASSIGNS, DISPATCH = _exec_model_nodes()


def test_vps_step_expression():
    ib = types.SimpleNamespace(all_greedy=False, max_num_logprobs=None, req_ids=[])
    for head, top, want in ((object(), False, True), (None, False, False), (object(), True, False)):
        me = types.SimpleNamespace(_vps_head=head, input_batch=ib, requests={})
        assert eval(ASSIGNS["vps_step"], {"getattr": getattr, "vps_wiring": VW}, {"self": me, "top_tokens": top}) is want


def test_vps_chunk_expression():
    md = object()
    for step, smd, cands, want in ((True, md, 64, True), (False, md, 64, False), (True, None, 64, False), (True, md, 0, False)):
        me = types.SimpleNamespace(_sampler_topk_candidates=lambda s, e, g, c=cands: c)
        # q38 L4 T-2 stage 0: vps_skip_topk is False off the fallback path (tests/q38_l4_vpshost covers it on).
        got = eval(ASSIGNS["vps_chunk"], {}, {"self": me, "vps_step": step, "spec_decode_metadata": smd, "start_index": 0,
                                              "end_index": 8, "vps_skip_topk": False})
        assert bool(got) is want


@pytest.mark.parametrize("sel_none", [True, False])
@pytest.mark.parametrize("vps_chunk,top_tokens,want", [(True, False, "local"), (False, True, "top"), (False, False, "full")])
def test_logits_dispatch(sel_none, vps_chunk, top_tokens, want):
    calls = []
    me = types.SimpleNamespace(
        compute_selected_local_logits=lambda h, i: calls.append("local_sel") or "local",
        compute_selected_top_tokens=lambda h, i: calls.append("top_sel") or "top",
        compute_selected_logits=lambda h, i: calls.append("full_sel") or "full",
        compute_local_logits_from_hidden_states=lambda h: calls.append("local_h") or "local",
        compute_top_tokens_from_hidden_states=lambda h: calls.append("top_h") or "top",
        compute_logits_from_hidden_states=lambda h: calls.append("full_h") or "full")
    planner = types.SimpleNamespace(finalize_hidden_states=lambda h, lp: h)
    loc = {"self": me, "sample_hidden_states": None if sel_none else "SH", "hidden_states": "H", "layout_plan": None,
           "sequence_layout_planner": planner, "logits_indices": "I", "vps_chunk": vps_chunk, "top_tokens": top_tokens}
    exec(DISPATCH, {}, loc)
    assert loc["logits"] == want and len(calls) == 1
