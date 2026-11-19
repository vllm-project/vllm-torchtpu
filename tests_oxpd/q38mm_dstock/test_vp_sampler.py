# SPDX-License-Identifier: Apache-2.0
"""VPS: layers/adapter/sample/vp_sampler.py against the PRODUCTION stock verify, token for token (torch CPU).

The stock side is the served code itself: top_k_top_p.py and rejection_sampler.py loaded from the tree, and
runner/tpu_runner.py's `_verify_fused_impl` / `_sample_from_logits_impl` / `_apply_temperature_impl` extracted with
ast (importing tpu_runner needs the whole TPU stack). Packages are stubbed so no torch_tpu / vllm import happens.
Run: python -m pytest -q -p no:cacheprovider --noconftest tests/q38mm_dstock/test_vp_sampler.py
Mutation-checked by tests/q38mm_dstock/q38mm_vps_mutrun.py (Q38MM_SRC_ROOT points at a mutated copy).
"""

import ast
import contextlib
import importlib.util
import os
import sys
import textwrap
import types

import pytest
import torch

ROOT = os.environ.get("Q38MM_SRC_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
SAMPLE = os.path.join(ROOT, "vllm_torchtpu", "layers", "adapter", "sample")
RUNNER = os.path.join(ROOT, "vllm_torchtpu", "runner", "tpu_runner.py")


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


def _setup():
    for n in ("vllm", "vllm.v1", "vllm.v1.spec_decode", "vllm_torchtpu", "vllm_torchtpu.layers",
              "vllm_torchtpu.layers.adapter", "vllm_torchtpu.layers.adapter.sample"):
        _pkg(n)
    utils = types.ModuleType("vllm.v1.spec_decode.utils")
    utils.unconditional_to_conditional_rates = lambda r: [p / q if q > 0 else 0.0 for p, q in zip(r, [1.0, *r[:-1]])]
    sys.modules["vllm.v1.spec_decode.utils"] = utils
    real_compile = torch.compile
    # The served modules decorate with torch.compile(backend="tpu"); on CPU the eager body is the reference.
    torch.compile = lambda fn=None, **kw: fn if callable(fn) else (lambda f: f)
    try:
        tk = _load("vllm_torchtpu.layers.adapter.sample.top_k_top_p", os.path.join(SAMPLE, "top_k_top_p.py"))
        rs = _load("vllm_torchtpu.layers.adapter.sample.rejection_sampler", os.path.join(SAMPLE, "rejection_sampler.py"))
        vp = _load("vllm_torchtpu.layers.adapter.sample.vp_sampler", os.path.join(SAMPLE, "vp_sampler.py"))
    finally:
        torch.compile = real_compile
    src = open(RUNNER).read()
    tree = ast.parse(src)
    ns = {"torch": torch, "apply_top_k_top_p": tk.apply_top_k_top_p,
          "_random_rejection_sample_impl": rs._random_rejection_sample_impl, "INVALID_TOKEN_ID": -1}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "SAMPLING_EPS" for t in node.targets):
            exec(ast.get_source_segment(src, node), ns)
    for fn in ("_apply_temperature_impl", "_sample_from_logits_impl", "_verify_fused_impl"):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn)
        exec(textwrap.dedent(ast.get_source_segment(src, node)), ns)
    assert ns["SAMPLING_EPS"] == vp.SAMPLING_EPS
    return tk, rs, vp, ns["_verify_fused_impl"]


HARNESS = HarnessModules()
with HARNESS.loading():
    TK, RS, VP, STOCK = _setup()


@pytest.fixture(autouse=True, scope="module")
def _harness_modules_installed():
    """The runner's lazy imports (vllm.distributed, vllm_torchtpu.envs, vp_sampler) resolve to the harness."""
    with HARNESS.active():
        yield
GOLDEN = torch.tensor([1.0, 0.32, 0.0])     # conditional rates at golden AL 2.32, k=3 (PD DEC-FB, C)


def problem(seed, B=8, V=10240, k=3, temp=0.6, top_k=20, top_p=0.95, greedy_every=0, scale=3.0, invalid_every=0):
    g = torch.Generator().manual_seed(seed)
    N, T = B * (k + 1), B * k
    logits = (torch.randn(N, V, generator=g) * scale)
    hot = torch.randint(0, V, (N, 6), generator=g)          # a few peaked tokens per row, as real logits have
    logits.scatter_add_(1, hot, torch.rand(N, 6, generator=g) * 8.0)
    logits = logits.to(torch.bfloat16)
    rows = torch.arange(N).view(B, k + 1)
    bonus_idx, target_idx = rows[:, k].contiguous(), rows[:, :k].reshape(-1).contiguous()
    temps = torch.full((B, 1), temp, dtype=torch.bfloat16)
    if greedy_every:
        temps[::greedy_every] = 0.0
    tk = torch.full((B, 1), top_k, dtype=torch.int32)
    tp = torch.full((B, 1), top_p, dtype=torch.float32)
    am = logits[target_idx].float().argmax(-1)
    rnd = torch.randint(0, V, (T,), generator=g)
    draft = torch.where(torch.rand(T, generator=g) < 0.5, am, rnd).to(torch.int32)
    lengths = torch.full((B,), k, dtype=torch.int32)
    if invalid_every:
        for b in range(0, B, invalid_every):
            n = b % (k + 1)
            lengths[b] = n
            draft[b * k + n:(b + 1) * k] = -1
    seg = torch.arange(B).repeat_interleave(k).to(torch.int32)
    grp = torch.arange(k).repeat(B).to(torch.int32)
    bonus_u = torch.rand(B, V, generator=g)
    accept_u = torch.rand(T, generator=g)
    recover_u = torch.rand(T, V, generator=g)
    return dict(logits=logits, bonus_idx=bonus_idx, target_idx=target_idx, temps=temps, top_k=tk, top_p=tp,
                bonus_u=bonus_u, accept_u=accept_u, recover_u=recover_u, draft=draft, lengths=lengths, seg=seg,
                grp=grp, k=k)


def stock(p, rates=GOLDEN, synthetic=True, num_reqs=None):
    args = (p["logits"], p["bonus_idx"], p["target_idx"], p["temps"], p["top_k"], p["top_p"], p["bonus_u"],
            p["accept_u"], p["recover_u"], p["draft"], p["lengths"], p["seg"], p["grp"], rates, p["k"], synthetic, 64)
    return STOCK(*args, num_reqs) if num_reqs is not None else STOCK(*args)


def shards(p, tp, kc, block):
    V = p["logits"].shape[1]
    vs = V // tp
    parts = [VP.local_candidates(p["logits"][:, r * vs:(r + 1) * vs], r * vs, kc, block) for r in range(tp)]
    return torch.cat([a for a, _ in parts], -1), torch.cat([b for _, b in parts], -1)


def vp(p, rates=GOLDEN, synthetic=True, tp=8, kc=32, block=40, num_reqs=None):
    cv, ci = shards(p, tp, kc, block)
    return VP.vp_verify(cv, ci, p["bonus_idx"], p["target_idx"], p["temps"], p["top_k"], p["top_p"], p["bonus_u"],
                        p["accept_u"], p["recover_u"], p["draft"], p["lengths"], p["seg"], p["grp"], rates, p["k"],
                        synthetic, num_reqs)


# ---- 1. local candidates: exact top-kc, ties included ----
def _ref_topk(x, kc):
    idx = torch.sort(x.float(), dim=-1, descending=True, stable=True).indices[:, :kc]
    return torch.gather(x, 1, idx), idx


@pytest.mark.parametrize("seed", range(4))
def test_local_candidates_exact(seed):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(16, 1280, generator=g) * 3).to(torch.bfloat16)
    x[3] = torch.round(x[3])                     # heavy ties
    x[5, :] = 1.0                                # all equal
    x[6, 40:80] = x[6].max() + 1                 # a whole block tied at the max
    v, i = VP.local_candidates(x, 0, 32, 40)
    rv, ri = _ref_topk(x, 32)
    assert torch.equal(v, rv) and torch.equal(i.long(), ri)
    v2, i2 = VP.local_candidates(x, 31040, 32, 40)
    assert torch.equal(i2.long(), ri + 31040)


def test_local_candidates_shape_guard():
    with pytest.raises(ValueError):
        VP.local_candidates(torch.zeros(2, 1000, dtype=torch.bfloat16), 0, 32, 40)


# ---- 2. merged argmax = torch.argmax over the full row (first occurrence) ----
def test_merged_argmax_first_occurrence_across_shards():
    p = problem(1)
    lg = p["logits"].clone()
    m = lg.float().max() + 4
    lg[0, 3 * 1280 + 7] = m                      # shard 3
    lg[0, 1 * 1280 + 9] = m                      # shard 1: the lower id must win
    lg[1, 5] = lg[1, 7000] = m                   # shard 0 and shard 5
    p["logits"] = lg
    cv, ci = shards(p, 8, 32, 40)
    got = VP._argmax_min_id(cv, ci).long()
    assert torch.equal(got, lg.float().argmax(-1))
    assert got[0].item() == 1280 + 9 and got[1].item() == 5


# ---- 3. tokens equal stock ----
@pytest.mark.parametrize("seed", range(12))
def test_tokens_equal_stock_synthetic_golden(seed):
    p = problem(seed)
    assert torch.equal(vp(p), stock(p))


@pytest.mark.parametrize("seed", range(6))
def test_tokens_equal_stock_real_acceptance(seed):
    p = problem(100 + seed)
    assert torch.equal(vp(p, rates=None, synthetic=False), stock(p, rates=None, synthetic=False))


@pytest.mark.parametrize("rates", [torch.tensor([1.0, 1.0, 1.0]), torch.tensor([0.0, 0.0, 0.0])])
def test_tokens_equal_stock_all_accept_all_reject(rates):
    p = problem(7)
    assert torch.equal(vp(p, rates=rates), stock(p, rates=rates))


def test_tokens_equal_stock_greedy_rows_and_invalid_drafts():
    p = problem(11, greedy_every=3, invalid_every=2)
    assert torch.equal(vp(p), stock(p))
    assert torch.equal(vp(p, rates=None, synthetic=False), stock(p, rates=None, synthetic=False))


@pytest.mark.parametrize("top_k,top_p,temp", [(1, 1.0, 1.0), (32, 0.9, 1.0), (20, 1.0, 0.3), (5, 0.5, 1.5)])
def test_tokens_equal_stock_sampling_params(top_k, top_p, temp):
    p = problem(21, top_k=top_k, top_p=top_p, temp=temp)
    assert torch.equal(vp(p), stock(p))


@pytest.mark.skipif(STOCK.__code__.co_argcount < 18, reason='PD tree: _verify_fused_impl has no num_reqs / read offsets')
def test_read_offsets_equal_stock():
    p = problem(5)
    nr = torch.tensor(6)
    a, ra = vp(p, num_reqs=nr)
    b, rb = stock(p, num_reqs=nr)
    assert torch.equal(a, b) and torch.equal(ra, rb)


def test_read_offsets_rule_pd():
    """PD tree (no num_reqs in the stock verify): vp_verify's read offsets follow the MM runner's rule on the stock
    tokens, the last valid position of each real row and 0 for padding rows."""
    p = problem(5, invalid_every=3)
    nr = torch.tensor(6)
    a, ra = vp(p, num_reqs=nr)
    b = stock(p)
    assert torch.equal(a, b)
    num_valid = (b != -1).sum(dim=1)
    want = [max(int(n) - 1, 0) if i < 6 else 0 for i, n in enumerate(num_valid)]
    assert ra.tolist() == want and max(want) > 0


def test_served_geometry_tp8():
    """V = 248,320 split TP8 into 31,040-wide shards, block 160, KC 32 (the served constants)."""
    p = problem(3, B=4, V=248320)
    assert torch.equal(vp(p, block=VP.VP_BLOCK, kc=VP.VP_KC), stock(p))


def test_from_local_matches_merged():
    p = problem(9)
    V, tp = p["logits"].shape[1], 8
    vs = V // tp
    parts = [VP.local_candidates(p["logits"][:, r * vs:(r + 1) * vs], r * vs, 32, 40) for r in range(tp)]
    calls = iter([torch.cat([a for a, _ in parts], -1), torch.cat([b for _, b in parts], -1)])
    out = VP.vp_verify_from_local(p["logits"][:, :vs], 0, lambda x: next(calls), p["bonus_idx"], p["target_idx"],
                                  p["temps"], p["top_k"], p["top_p"], p["bonus_u"], p["accept_u"], p["recover_u"],
                                  p["draft"], p["lengths"], p["seg"], p["grp"], GOLDEN, p["k"], True, kc=32, block=40)
    assert torch.equal(out, stock(p))


def test_positive_control_uniforms_matter():
    """The comparison can see a token change: perturbing the uniforms changes VP's tokens."""
    p = problem(13)
    base = vp(p)
    q = dict(p)
    q["bonus_u"] = torch.rand(p["bonus_u"].shape, generator=torch.Generator().manual_seed(999))
    q["recover_u"] = torch.rand(p["recover_u"].shape, generator=torch.Generator().manual_seed(998))
    assert not torch.equal(vp(q), base)
    assert torch.equal(vp(q), stock(q))


# ---- 4. the copied assembly block has not drifted from production ----
def test_assembly_block_is_verbatim():
    rs = open(os.path.join(SAMPLE, "rejection_sampler.py")).read()
    vs = open(os.path.join(SAMPLE, "vp_sampler.py")).read()
    start = "    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)\n"
    end = "    return selected_tokens[:batch_size]\n"
    fn = rs[rs.index("def _random_rejection_sample_impl("):]
    prod = fn[fn.index(start):fn.index(end) + len(end)]
    mine = vs[vs.index(start):vs.index(end) + len(end)]
    assert mine == prod


def test_invalid_draft_inside_declared_length():
    """A placeholder draft (-1) inside the declared length must never be accepted (stock: accepted & valid)."""
    p = problem(17)
    p["draft"][1] = -1                       # sequence 0, slot 1, length still k
    p["draft"][4] = -1                       # sequence 1, slot 1
    rates = torch.tensor([1.0, 1.0, 1.0])
    assert torch.equal(vp(p, rates=rates), stock(p, rates=rates))


def test_greedy_row_with_tied_max_rejects_the_second_token():
    """Greedy rows accept by argmax identity, not by probability: with two tokens tied at the max, a draft equal
    to the higher-id one is rejected even though its probability is about 0.5."""
    p = problem(19, greedy_every=1)
    lg = p["logits"].clone()
    for t in range(p["target_idx"].shape[0]):
        row = p["target_idx"][t].item()
        m = lg[row].float().max() + 2
        lg[row, 100], lg[row, 9000] = m, m
        p["draft"][t] = 9000
    p["logits"] = lg
    p["accept_u"] = torch.full_like(p["accept_u"], 0.1)
    a = vp(p, rates=None, synthetic=False)
    assert torch.equal(a, stock(p, rates=None, synthetic=False))
    assert (a[:, 0] == 100).all()            # rejected at slot 0, recovered to the first tied token
