# SPDX-License-Identifier: Apache-2.0
"""q38 L4 SAMP2 on the composed v5 tree (q38-l4-samp2-vfy; port of tests/q38_l4_samp2/test_l4_samp2.py at c152b0199:
the composed tree has no runner/l4_vpshost.py, its `_vps_verify_fb` takes D83's `ro`). CDK CPU pod only.

S2 (VLLM_TPU_VPS_TOPK_CUTOFF): vp_sampler._candidate_mask's top-k cutoff from one top-k instead of
top_k_top_p.topk_mask's 32-step bisection. Class B (bitwise): `_topk_mask_cutoff` equals topk_mask bit for bit for
every k (k <= 0, 1..W, > W), on smooth, tie-heavy and special (NaN, +-inf, +-0, +-FLT_MAX, denormal) rows, bf16 and
f32; whole VPS verifies give the same tokens with the flag on as off.

VRNG-VPS (VLLM_TPU_VPS_INGRAPH_RNG): the VPS verify's uniforms drawn in-graph from a device counter
(layers/adapter/sample/vps_rng.py, dongruizi's counter_uniform credited; runner/l4_samp2.py controller). Class R:
  * bitwise plumbing: the VRNG verify equals today's VPS verify fed with the counter's full arrays (vps_rng.full_draws),
    at vp_sampler level and through the runner's `_vps_verify_rng_impl`; the element indices are vps_rng's layout;
  * distribution: >= 4M uniforms (KS, 1024-bin chi-square, moments, 24 bit frequencies, 2-sample KS against
    torch.rand), independence (lag-1, consecutive counters, regions, 2-D chi-square), and >= 1e6 accept / reject
    outcomes of the real verify against torch.rand's and against the analytic target (acceptance per position,
    accepted-length histogram, recovered-token and bonus-token distributions);
  * seeded reproducibility, one counter step per VRNG call, warm-up never touching the serving counter;
  * SAB dual path: cold keys, seeded chunks, the counter wrap and disagreement all take the base path; 8 simulated TP
    ranks (threads, barrier all_gather) keep equal counters and tokens; a one-rank counter offset is caught;
  * AOT census: eager RNG dispatches per VPS call 3 (4 with fb_u) -> 0; no RNG node in the fullgraph capture;
  * AL canary: the program's [accepted, drafted, rows] equal the tokens' own counts; the timed line.
"""

import ast
import logging
import math
import os
import random
import subprocess
import sys
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import vllm_torchtpu.runner.tpu_runner as tr
from vllm_torchtpu.runner import l4_agree  # q38 L4 AGREE port (v5 composition): VRNG agrees on the runner's clock
from vllm_torchtpu.runner import l4_vpshost  # v5 composition: VPSHOST T-1(i)'s (idle) async-copy state
from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.core import l6_ab
from vllm_torchtpu.layers.adapter.sample import counter_uniform as cu
from vllm_torchtpu.layers.adapter.sample import top_k_top_p as tk
from vllm_torchtpu.layers.adapter.sample import vp_sampler as vp
from vllm_torchtpu.layers.adapter.sample import vps_rng
from vllm_torchtpu.runner import l4_samp2

HERE = os.path.dirname(os.path.abspath(__file__))
K = 3
TV = 10240                # 64 VP_BLOCKs of 160: a TP1 VPS head
V = 248320                # Qwen3.8 vocabulary (served draw width)
F32_MAX = float(torch.finfo(torch.float32).max)
INF = float("inf")
CPU = torch.device("cpu")


# ---- import origin ------------------------------------------------------------------------------------------------
def test_imports_the_tree_under_test():
    """The modules under test resolve inside this tree's src (or Q38_SAMP2_EXPECT_ROOT); the CPU image also carries an
    installed vllm_torchtpu, which must not be the one imported."""
    root = os.path.realpath(os.environ.get("Q38_SAMP2_EXPECT_ROOT") or os.path.join(HERE, "..", "..", "src"))
    for mod in (tr, tpu_compiler, l6_ab, cu, tk, vp, vps_rng, l4_samp2):  # the composed tree's modules
        where = os.path.realpath(mod.__file__)
        assert where.startswith(root + os.sep), f"{mod.__name__} imported from {where}, not under {root}"


# ---- registration -------------------------------------------------------------------------------------------------
def test_flags_registered_and_keyed():
    """S2 and VRNG are graph flags (registered, hashed); the VRNG path switch is host only (ignored). No on-value
    holds a ',' (the served decode env is ','-split)."""
    from vllm_torchtpu import envs

    for name in ("VLLM_TPU_VPS_TOPK_CUTOFF", "VLLM_TPU_VPS_INGRAPH_RNG"):
        assert name in envs.environment_variables and name not in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    assert "VLLM_TPU_VPS_INGRAPH_RNG_AB" in envs.environment_variables
    assert "VLLM_TPU_VPS_INGRAPH_RNG_AB" in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    for value in ("1", "0", "alt:400", "alt:2"):
        assert "," not in value
        l4_samp2.parse_ab(value)
    assert l4_samp2.parse_ab("alt:400") == ("alt", 400) and l4_samp2.parse_ab("") == ("on", l4_samp2.AGREE_EVERY)
    for bad in ("alt:1", "alt:x", "ab:1+3", "2"):
        with pytest.raises(ValueError):
            l4_samp2.parse_ab(bad)


# =================================================== S2 ============================================================
def _bits(x):
    return x.view(torch.int16) if x.dtype == torch.bfloat16 else x.view(torch.int32)


def _s2_rows(kind, rows, w, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    if kind == "smooth":
        x = torch.randn(rows, w, generator=g) * 4
    elif kind == "ties":
        pool = torch.tensor([-INF, -3.0, -1.5, -0.0, 0.0, 0.5, 1.25, 2.0, 7.0])
        x = pool[torch.randint(0, pool.numel(), (rows, w), generator=g)]
    elif kind == "special":
        pool = torch.tensor([float("nan"), INF, -INF, F32_MAX, -F32_MAX, 1e-40, -1e-40, 0.0, -0.0, 1.0, -2.0, 3.5])
        x = pool[torch.randint(0, pool.numel(), (rows, w), generator=g)]
    elif kind == "sparse":
        # mostly -inf rows with a few finite / +inf entries: k-th value is -inf (or +inf) on many rows
        x = torch.full((rows, w), -INF)
        n = max(1, w // 8)
        idx = torch.randint(0, w, (rows, n), generator=g)
        vals = torch.where(torch.rand(rows, n, generator=g) < 0.1, torch.full((rows, n), INF), torch.randn(rows, n, generator=g))
        x.scatter_(1, idx, vals)
    else:
        raise AssertionError(kind)
    return x.to(dtype)


def _ks_for(w):
    return [-2, -1, 0, *range(1, w + 1), w + 1, w + 50, V]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("w", [256, 33, 1])
@pytest.mark.parametrize("kind", ["smooth", "ties", "special", "sparse"])
def test_s2_topk_mask_cutoff_bitwise(kind, w, dtype):
    """`_topk_mask_cutoff` == topk_mask bit for bit, for every k in [-2, W + 50] and the vocabulary size."""
    ks = _ks_for(w)
    reps = 3
    x = _s2_rows(kind, len(ks) * reps, w, dtype, seed=w * 7 + len(kind))
    k = torch.tensor(ks * reps, dtype=torch.int32).unsqueeze(-1)
    want = tk.topk_mask(x, k)
    got = vp._topk_mask_cutoff(x, k)
    assert got.dtype == want.dtype and got.shape == want.shape
    bad = (_bits(got) != _bits(want)).any(dim=-1).nonzero().flatten().tolist()
    assert bad == [], [(i, ks[i % len(ks)]) for i in bad[:8]]
    # vacuity: the special branches really occur
    kk = k.squeeze(-1)
    assert bool((kk <= 0).any()) and bool((kk > w).any())
    if kind in ("special", "sparse"):
        assert bool(torch.isinf(x.float()).any())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_s2_k_above_width_bitwise(dtype):
    """k > W (W + 1, W + 50, the vocabulary): bitwise topk_mask on rows with no -inf (the clamped index's smallest
    value is finite, every finite entry passes), rows with -inf, rows with NaN, rows that are all -inf and rows with
    +inf; the cutoff path has no k > W branch."""
    w = 256
    g = torch.Generator().manual_seed(41)
    rows = []
    rows.append(torch.randn(6, w, generator=g) * 4)                                  # no -inf
    x = torch.randn(6, w, generator=g)
    x[:, ::7] = -INF
    rows.append(x)                                                                     # some -inf
    x = torch.randn(6, w, generator=g)
    x[:, ::5] = float("nan")
    rows.append(x)                                                                     # NaN
    rows.append(torch.full((6, w), -INF))                                              # all -inf
    x = torch.randn(6, w, generator=g)
    x[:, :3] = INF
    rows.append(x)                                                                     # +inf
    x = torch.cat(rows).to(dtype)
    k = torch.tensor([w + 1, w + 50, V] * (x.shape[0] // 3), dtype=torch.int32).unsqueeze(-1)
    assert torch.equal(_bits(vp._topk_mask_cutoff(x, k)), _bits(tk.topk_mask(x, k)))
    assert not bool(torch.isinf(x[:6].float()).any()) and bool(torch.isneginf(x[6:12].float()).any())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("kind", ["smooth", "ties", "special"])
def test_s2_apply_top_k_top_p_bitwise(kind, dtype):
    """The whole `_apply_top_k_top_p_cutoff` == apply_top_k_top_p (topk_candidates=0) bit for bit, top_p in
    {1, 0.95, 0.8, 0.5, 0.05} with every k."""
    w = 256
    ks = _ks_for(w)
    ps = [1.0, 0.95, 0.8, 0.5, 0.05]
    n = len(ks) * len(ps)
    x = _s2_rows(kind, n, w, dtype, seed=11 + len(kind))
    top_k = torch.tensor([k for k in ks for _ in ps], dtype=torch.int32).unsqueeze(-1)
    top_p = torch.tensor([p for _ in ks for p in ps], dtype=torch.float32).unsqueeze(-1)
    want = tk.apply_top_k_top_p(x, top_k, top_p)
    got = vp._apply_top_k_top_p_cutoff(x, top_k, top_p)
    assert torch.equal(_bits(got), _bits(want))


def test_s2_candidate_mask_flag_on_off(monkeypatch):
    """`_candidate_mask` is bitwise the same with the flag on as off; on, it takes the cutoff path (and never
    apply_top_k_top_p), off the reverse (vacuity)."""
    calls = {"cut": 0, "stock": 0}
    real_cut, real_stock = vp._apply_top_k_top_p_cutoff, vp.apply_top_k_top_p

    def cut(*a):
        calls["cut"] += 1
        return real_cut(*a)

    def stock(*a, **k):
        calls["stock"] += 1
        return real_stock(*a, **k)

    monkeypatch.setattr(vp, "_apply_top_k_top_p_cutoff", cut)
    monkeypatch.setattr(vp, "apply_top_k_top_p", stock)
    g = torch.Generator().manual_seed(5)
    for dtype in (torch.bfloat16, torch.float32):
        cand = (torch.randn(64, 256, generator=g) * 3).to(dtype)
        top_k = torch.randint(0, 40, (64, 1), generator=g).to(torch.int32)
        top_p = torch.tensor([1.0, 0.95, 0.8, 0.6])[torch.randint(0, 4, (64,), generator=g)].unsqueeze(-1)
        monkeypatch.setattr(vp, "_TOPK_CUTOFF", False)
        off = vp._candidate_mask(cand, top_k, top_p)
        assert calls == {"cut": 0, "stock": 1}
        monkeypatch.setattr(vp, "_TOPK_CUTOFF", True)
        on = vp._candidate_mask(cand, top_k, top_p)
        assert calls == {"cut": 1, "stock": 1}
        assert torch.equal(_bits(on), _bits(off))
        calls.update(cut=0, stock=0)


def _ident(x):
    return x


def vprob(P, seed, vocab=TV, n_range=1, temps=(0.6, 0.7, 1.0, 0.0), dl=None):
    """A VPS verify problem: logits [P*(K+1), vocab] bf16 with peaks, metadata, padded sampling params. n_range
    requests carry an out-of-range top_k (0, 40 or 100: fallback rows)."""
    g = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    N, T = P * (K + 1), P * K
    logits = torch.randn(N, vocab, generator=g) * 3
    logits.scatter_add_(1, torch.randint(0, vocab, (N, 6), generator=g), torch.rand(N, 6, generator=g) * 8)
    logits = logits.bfloat16()
    rows = torch.arange(N).view(P, K + 1)
    am = logits[rows[:, :K].reshape(-1)].float().argmax(-1)
    draft = torch.where(torch.rand(T, generator=g) < 0.6, am, torch.randint(0, vocab, (T,), generator=g)).to(torch.int32)
    lens = dl if dl is not None else [K if rng.random() < 0.8 else rng.randint(0, K) for _ in range(P)]
    md = NS(bonus_logits_indices=rows[:, K].contiguous().to(torch.int32),
            target_logits_indices=rows[:, :K].reshape(-1).contiguous().to(torch.int32),
            draft_token_ids=draft,
            draft_lengths=torch.tensor(lens, dtype=torch.int32),
            draft_lengths_cpu=np.array(lens, dtype=np.int32),
            segment_ids=torch.arange(P).repeat_interleave(K).to(torch.int32),
            group_indices=torch.arange(K).repeat(P).to(torch.int32))
    rr = set(rng.sample(range(P), min(n_range, P)))
    t = [rng.choice([0.6, 0.7, 1.0]) if i in rr else rng.choice(temps) for i in range(P)]
    kk = [rng.choice([0, 40, 100]) if i in rr else rng.choice([20, 20, 1, 5, 32]) for i in range(P)]
    pp = [rng.choice([1.0, 0.95, 0.8]) for _ in range(P)]
    params = (torch.tensor(t, dtype=torch.bfloat16).unsqueeze(-1), torch.tensor(kk, dtype=torch.int32).unsqueeze(-1),
              torch.tensor(pp, dtype=torch.float32).unsqueeze(-1))
    return logits, md, params


def common(md, params, bu, au, ru, synth=False, rates=None):
    temps, top_k, top_p = params
    return (md.bonus_logits_indices, md.target_logits_indices, temps, top_k, top_p, bu, au, ru, md.draft_token_ids,
            md.draft_lengths, md.segment_ids, md.group_indices, rates, K, synth)


@pytest.mark.parametrize("fb", [None, 0, 4, 8])
def test_s2_vps_verify_tokens_bitwise(monkeypatch, fb):
    """Whole VPS verifies (TP1; fallback widths F = 0 / 4 / 8 or the plain verify), full-vocabulary torch.rand draws:
    tokens and stats are the same with S2 on as off (the cutoff path ran: vacuity)."""
    seen = [0]
    real = vp._topk_mask_cutoff
    monkeypatch.setattr(vp, "_topk_mask_cutoff", lambda *a: seen.__setitem__(0, seen[0] + 1) or real(*a))
    for trial in range(4):
        logits, md, params = vprob(8, 100 + trial, n_range=trial % 3)
        g = torch.Generator().manual_seed(trial)
        bu, au, ru = torch.rand(8, TV, generator=g), torch.rand(8 * K, generator=g), torch.rand(8 * K, TV, generator=g)
        out = {}
        for flag in (False, True):
            monkeypatch.setattr(vp, "_TOPK_CUTOFF", flag)
            if fb is None:
                out[flag] = (vp.vp_verify_from_local(logits, 0, _ident, *common(md, params, bu, au, ru)),)
            else:
                out[flag] = vp.vp_verify_fb(logits, 0, _ident, *common(md, params, bu, au, ru),
                                            torch.tensor([8], dtype=torch.int32), fallback_rows=fb)
        assert all(torch.equal(a, b) for a, b in zip(out[False], out[True])), trial
    assert seen[0] > 0


def test_s2_flag_read_at_import():
    """The module-level read of VLLM_TPU_VPS_TOPK_CUTOFF (env_bool spellings), in fresh processes."""
    path = os.path.join(HERE, "..", "..", "src", "vllm_torchtpu", "layers", "adapter", "sample", "vp_sampler.py")
    code = ("import ast, os, sys; src = open(sys.argv[1]).read(); tree = ast.parse(src); "
            "ns = {'os': os}; node = [n for n in tree.body if isinstance(n, ast.Assign) and "
            "any(getattr(t, 'id', None) == '_TOPK_CUTOFF' for t in n.targets)][0]; "
            "exec(ast.get_source_segment(src, node), ns); print('CUT', ns['_TOPK_CUTOFF'])")
    env = {k: v for k, v in os.environ.items() if k != "VLLM_TPU_VPS_TOPK_CUTOFF"}
    for extra, want in (({}, "CUT False"), ({"VLLM_TPU_VPS_TOPK_CUTOFF": "1"}, "CUT True"),
                        ({"VLLM_TPU_VPS_TOPK_CUTOFF": "on"}, "CUT True"), ({"VLLM_TPU_VPS_TOPK_CUTOFF": "0"}, "CUT False")):
        out = subprocess.run([sys.executable, "-I", "-c", code, path], env={**env, **extra},
                             capture_output=True, text=True, timeout=60)
        assert out.returncode == 0 and out.stdout.strip() == want, (extra, out.stdout, out.stderr)
    # and the real module, imported fresh with the flag on
    src = os.path.join(HERE, "..", "..", "src")
    out = subprocess.run([sys.executable, "-c", "import vllm_torchtpu.layers.adapter.sample.vp_sampler as v; "
                          "print('CUT', v._TOPK_CUTOFF)"], cwd=os.path.join(HERE, "..", ".."),
                         env={**env, "VLLM_TPU_VPS_TOPK_CUTOFF": "1", "PYTHONPATH": src, "JAX_PLATFORMS": "cpu"},
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0 and out.stdout.strip().splitlines()[-1] == "CUT True", out.stderr[-2000:]


# =================================================== VRNG: the hash and the layout ==================================
def _fmix_py(x):
    x &= 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    x ^= x >> 16
    return x


def _u_py(counter, i):
    """An independent pure-Python oracle of counter_uniform's element i."""
    c = counter & 0xFFFFFFFF
    k1 = _fmix_py(c * 0x9E3779B9)
    k2 = _fmix_py(c ^ 0x5BD1E995)
    h = _fmix_py((_fmix_py((i & 0xFFFFFFFF) ^ k1) + k2) & 0xFFFFFFFF)
    return ((h >> 8) & 0xFFFFFF) / 16777216.0


COUNTERS = [0, 1, -1, 2**31 - 1, -2**31, 12345, cu.initial_counter(0), vps_rng.vps_initial_counter(0),
            vps_rng.vps_initial_counter(7)]


def test_counter_uniform_at_equals_counter_uniform_and_oracle():
    g = torch.Generator().manual_seed(3)
    for c in COUNTERS:
        ct = torch.tensor(c, dtype=torch.int32)
        n = 1 << 16
        assert torch.equal(vps_rng.counter_uniform_at(ct, torch.arange(n, dtype=torch.int32)),
                           cu.counter_uniform((n,), ct))
        idx = torch.randint(0, 2**31 - 1, (64,), generator=g, dtype=torch.int64).to(torch.int32)
        got = vps_rng.counter_uniform_at(ct, idx).tolist()
        want = [_u_py(c, int(i)) for i in idx.tolist()]
        assert got == want, c
    assert vps_rng.vps_initial_counter(0) != cu.initial_counter(0)
    assert all(-2**31 <= vps_rng.vps_initial_counter(s) < 2**31 for s in (0, 1, 2**40, -5))


def test_vrng_indices_follow_the_layout():
    """With `at` returning the element index itself, the in-graph draws read exactly vps_rng's layout: bonus r*V + id,
    accept B*V + t, recover B*V + T_a + t*V + id, whole fallback rows at the region bases; regions disjoint."""
    B, T, W = 16, 48, 256
    g = torch.Generator().manual_seed(1)
    bci = torch.randint(0, V, (B, W), generator=g, dtype=torch.int32)
    tci = torch.randint(0, V, (T, W), generator=g, dtype=torch.int32)
    draft = torch.zeros(T, dtype=torch.int32)
    at = lambda c, idx: idx.to(torch.int64)  # noqa: E731
    draws = (torch.tensor(0, dtype=torch.int32), V, at)
    b, a, r = vp._counter_draws(draws, bci, tci, draft)
    rb = torch.arange(B).unsqueeze(-1) * V
    rt = torch.arange(T).unsqueeze(-1) * V
    assert torch.equal(b, rb + bci) and torch.equal(a, B * V + torch.arange(T))
    assert torch.equal(r, B * V + T + rt + tci)
    a0, r0, total = vps_rng.layout(B, T, T, V)
    assert (a0, r0, total) == (B * V, B * V + T, (B + T) * V + T)
    assert int(b.max()) < a0 <= int(a.min()) and int(a.max()) < r0 <= int(r.min()) and int(r.max()) < total
    rows_b = vp._counter_rows(draws, 0, torch.tensor([0, 5, 15]))
    rows_t = vp._counter_rows(draws, r0, torch.tensor([0, 47]))
    assert torch.equal(rows_b[1], 5 * V + torch.arange(V)) and torch.equal(rows_t[1], r0 + 47 * V + torch.arange(V))
    # the full draws split at the same bases
    ct = torch.tensor(99, dtype=torch.int32)
    fb, fa, fr = vps_rng.full_draws(ct, 2, (3,), 3, 7)
    u = cu.counter_uniform((2 * 7 + 3 + 3 * 7,), ct)
    assert torch.equal(fb.reshape(-1), u[:14]) and torch.equal(fa, u[14:17]) and torch.equal(fr.reshape(-1), u[17:])


def test_vrng_int32_guard():
    with pytest.raises(ValueError):
        vps_rng.layout(6000, 3000, 3000, V)          # (6000 + 3000) x 248,320 + 3000 > 2^31
    vps_rng.layout(5000, 3000, 3000, V)              # 1.99e9 < 2^31: allowed
    big = torch.zeros(6000, 1, dtype=torch.int32)
    with pytest.raises(ValueError):
        vp._counter_draws((torch.tensor(0, dtype=torch.int32), V, vps_rng.counter_uniform_at), big,
                          torch.zeros(3000, 1, dtype=torch.int32), torch.zeros(3000, dtype=torch.int32))


@pytest.mark.parametrize("synth", [False, True])
@pytest.mark.parametrize("fb", [None, 0, 4, 8])
def test_vrng_equals_full_draw_reference(fb, synth):
    """vp_sampler level: the in-graph draws give bitwise the tokens (and fallback stats) of today's
    VLLM_TPU_VPS_DRAWS=full verify fed with the counter's full arrays, over counters at the int32 edges."""
    rates = torch.tensor([1.0, 0.32, 0.0]) if synth else None
    covered = 0
    for trial, c in enumerate(COUNTERS[:6]):
        logits, md, params = vprob(8, 200 + trial, n_range=1 + trial % 3)
        ct = torch.tensor(c, dtype=torch.int32)
        bu, au, ru = vps_rng.full_draws(ct, 8, tuple(md.draft_token_ids.shape), 8 * K, TV)
        draws = (ct, TV, vps_rng.counter_uniform_at)
        n = torch.tensor([8], dtype=torch.int32)
        if fb is None:
            ref = (vp.vp_verify_from_local(logits, 0, _ident, *common(md, params, bu, au, ru, synth, rates)),)
            got = (vp.vp_verify_from_local(logits, 0, _ident, *common(md, params, None, None, None, synth, rates),
                                           draws=draws),)
        else:
            ref = vp.vp_verify_fb(logits, 0, _ident, *common(md, params, bu, au, ru, synth, rates), n, fallback_rows=fb)
            got = vp.vp_verify_fb(logits, 0, _ident, *common(md, params, None, None, None, synth, rates), n,
                                  fallback_rows=fb, draws=draws)
            covered += int(ref[1][1])
        assert all(torch.equal(a, b) for a, b in zip(ref, got)), (trial, c)
    if fb:
        assert covered > 0          # the fallback rows' whole-row draws were exercised


@pytest.fixture
def ident_gather(monkeypatch):
    import vllm.distributed

    monkeypatch.setattr(vllm.distributed, "tensor_model_parallel_all_gather", lambda x, dim=-1: x)


def _al_expect(out, md, n):
    emitted = (out != -1).sum(dim=1)
    acc = dra = rows = 0
    for i in range(n):
        d = int(md.draft_lengths[i])
        dra += d
        if d > 0:
            rows += 1
            acc += max(int(emitted[i]) - 1, 0)
    return [acc, dra, rows]


@pytest.mark.parametrize("fb", [False, True])
def test_vrng_runner_impl_equals_base_impl(ident_gather, fb):
    """Runner level: `_vps_verify_rng_impl` == `_vps_verify_fb_impl` / `_vps_verify_impl` on the counter's full draws
    (tokens, stats); it returns counter + 1 and the AL counts of its own tokens over the active rows."""
    for trial, c in enumerate([0, 2**31 - 1, -7]):
        logits, md, params = vprob(8, 300 + trial, n_range=1)
        ct = torch.tensor(c, dtype=torch.int32)
        bu, au, ru = vps_rng.full_draws(ct, 8, tuple(md.draft_token_ids.shape), 8 * K, TV)
        n_act = 6
        na = torch.tensor([n_act], dtype=torch.int32)
        args = common(md, params, bu, au, ru)
        F = 4 if fb else 0
        if fb:
            ref_out, ref_stats = tr._vps_verify_fb_impl(logits, 0, *args, na, F, False, None)
        else:
            ref_out, ref_stats = tr._vps_verify_impl(logits, 0, *args), torch.zeros(3, dtype=torch.int32)
        out, stats, al, nxt = tr._vps_verify_rng_impl(logits, 0, *args[:5], ct, *args[8:], na, F, fb, TV)
        assert torch.equal(out, ref_out) and torch.equal(stats, ref_stats)
        assert int(nxt) == (c + 1 if c != 2**31 - 1 else -2**31) and nxt.dtype == torch.int32
        assert al.tolist() == _al_expect(out, md, n_act), trial


# =================================================== VRNG: distribution =============================================
def _ks(u):
    u = torch.sort(u.double()).values
    n = u.numel()
    i = torch.arange(1, n + 1, dtype=torch.float64)
    return float(torch.maximum(i / n - u, u - (i - 1) / n).max()) * math.sqrt(n)


def _chi2(counts, expected):
    counts = counts.double()
    return float(((counts - expected) ** 2 / expected).sum())


def _served_uniforms(n_counters=16, per=1 << 18, seed=0):
    """>= 4M uniforms the served program would read: for consecutive counters, bonus / accept / recover elements at
    candidate-like ids of the served layout (B 16, T 48, V 248,320)."""
    g = torch.Generator().manual_seed(seed)
    B, T = 16, 48
    a0, r0, _ = vps_rng.layout(B, T, T, V)
    c0 = vps_rng.vps_initial_counter(seed)
    out = []
    for j in range(n_counters):
        ct = torch.tensor(((c0 + j + 2**31) % 2**32) - 2**31, dtype=torch.int32)
        # distinct elements only (sampled without replacement), as one call reads them
        half = per // 2
        bi = torch.randperm(B * V, generator=g)[:half]
        ai = a0 + torch.arange(T)
        ri = r0 + torch.randperm(T * V, generator=g)[:per - half - T]
        idx = torch.cat([bi, ai, ri]).to(torch.int32)
        out.append(vps_rng.counter_uniform_at(ct, idx))
    return torch.cat(out)


def test_vrng_uniforms_distribution():
    """KS (D sqrt(n) < 1.95), 1024-bin chi-square (< df + 6 sqrt(2 df)), mean / variance within 5 SE, every value in
    [0, 1) on the 2^-24 grid, 24 bit frequencies within 5 SE; torch.rand passes the same bars (control) and the
    2-sample KS against it is < 1.95."""
    u = _served_uniforms()
    n = u.numel()
    assert n >= 4_000_000
    ref = torch.rand(n, generator=torch.Generator().manual_seed(17))
    rows = {}
    for name, x in (("vrng", u), ("rand", ref)):
        ks = _ks(x)
        counts = torch.bincount((x * 1024).long().clamp(max=1023), minlength=1024)
        chi = _chi2(counts, n / 1024)
        mean, var = float(x.double().mean()), float(x.double().var())
        rows[name] = (ks, chi, mean, var)
        assert ks < 1.95, (name, ks)
        assert chi < 1023 + 6 * math.sqrt(2 * 1023), (name, chi)
        assert abs(mean - 0.5) < 5 * math.sqrt(1 / 12 / n), (name, mean)
        assert abs(var - 1 / 12) < 5 * math.sqrt(1 / 180 / n), (name, var)
    assert float(u.min()) >= 0.0 and float(u.max()) < 1.0
    q = u.double() * 16777216.0
    assert torch.equal(q, torch.floor(q))
    words = (u.double() * 16777216.0).long()
    bits_z = [abs(float(((words >> b) & 1).double().mean()) - 0.5) / math.sqrt(0.25 / n) for b in range(24)]
    assert max(bits_z) < 5, bits_z
    m = 1 << 20
    a, b = torch.sort(u[:m].double()).values, torch.sort(ref[:m].double()).values
    allv = torch.cat([a, b])
    cdf_a = torch.searchsorted(a, allv, right=True).double() / m
    cdf_b = torch.searchsorted(b, allv, right=True).double() / m
    ks2 = float((cdf_a - cdf_b).abs().max()) * math.sqrt(m * m / (2 * m))
    print(f"\nVRNG_DIST n={n} vrng ks={rows['vrng'][0]:.3f} chi2={rows['vrng'][1]:.1f} mean={rows['vrng'][2]:.6f} "
          f"var={rows['vrng'][3]:.6f} | rand ks={rows['rand'][0]:.3f} chi2={rows['rand'][1]:.1f} | 2sample_ks={ks2:.3f} "
          f"max_bit_z={max(bits_z):.2f}")
    assert ks2 < 1.95
    # the accept stream as the verify reads it: slots a0 + t of 4096 consecutive counters (t < 1024), and the
    # fraction below the acceptance probabilities the outcome test meets (z < 5)
    B, T = 1024, 1024
    a0, _, _ = vps_rng.layout(B, T, T, 5120)
    c0 = vps_rng.vps_initial_counter(5)
    acc = torch.cat([vps_rng.counter_uniform_at(torch.tensor(((c0 + i + 2**31) % 2**32) - 2**31, dtype=torch.int32),
                                                a0 + torch.arange(T, dtype=torch.int32)) for i in range(4096)])
    na = acc.numel()
    ks_a = _ks(acc)
    zs = {p: abs(float((acc < p).double().mean()) - p) / math.sqrt(p * (1 - p) / na) for p in (0.05, 0.1355, 0.25, 0.5)}
    print(f"VRNG_DIST accept_stream n={na} ks={ks_a:.3f} " + " ".join(f"z@{p}={z:.2f}" for p, z in zs.items()))
    assert ks_a < 1.95 and max(zs.values()) < 5, (ks_a, zs)


def _corr(a, b):
    a, b = a.double() - a.double().mean(), b.double() - b.double().mean()
    return float((a * b).sum() / math.sqrt(float((a * a).sum()) * float((b * b).sum())))


def test_vrng_uniforms_independence():
    """|corr| < 6 / sqrt(n) for: lag-1 within a vocabulary row, consecutive counters (c, c + d) at the same element
    for d in {1, 2, 256, 65536}, bonus vs recover region at the same (row, token); and a 16 x 16 chi-square of the
    pairs (u(c, i), u(c + 1, i)) (< df + 6 sqrt(2 df))."""
    n = 1 << 20
    c0 = vps_rng.vps_initial_counter(1)
    ct = lambda c: torch.tensor(((c + 2**31) % 2**32) - 2**31, dtype=torch.int32)  # noqa: E731
    idx = torch.arange(n, dtype=torch.int32)
    u0 = vps_rng.counter_uniform_at(ct(c0), idx)
    bar = 6 / math.sqrt(n)
    res = {"lag1": _corr(u0[:-1], u0[1:])}
    for d in (1, 2, 256, 65536):
        res[f"ctr+{d}"] = _corr(u0, vps_rng.counter_uniform_at(ct(c0 + d), idx))
    B, T = 16, 48
    _, r0, _ = vps_rng.layout(B, T, T, V)
    g = torch.Generator().manual_seed(2)
    rows, toks = torch.randint(0, B, (n,), generator=g), torch.randint(0, V, (n,), generator=g)
    ub = vps_rng.counter_uniform_at(ct(c0), (rows * V + toks).to(torch.int32))
    ur = vps_rng.counter_uniform_at(ct(c0), (r0 + rows * V + toks).to(torch.int32))
    res["bonus_vs_recover"] = _corr(ub, ur)
    u1 = vps_rng.counter_uniform_at(ct(c0 + 1), idx)
    cell = (u0 * 16).long().clamp(max=15) * 16 + (u1 * 16).long().clamp(max=15)
    chi = _chi2(torch.bincount(cell, minlength=256), n / 256)
    # stream overlap: the next counter's stream (or the recover region) must not be this stream shifted; under
    # independence an exact match has probability 2^-24 per element (expected ~0.06 of 2^20)
    overlap = {}
    for d in (-2, -1, 0, 1, 2):
        lo, hi = max(0, -d), n - max(0, d)
        overlap[f"ctr+1@{d:+d}"] = float((u1[lo:hi] == u0[lo + d:hi + d]).double().mean())
    overlap["recover@bonus"] = float((ur == ub).double().mean())
    print(f"\nVRNG_INDEP n={n} bar={bar:.5f} " + " ".join(f"{k}={v:+.5f}" for k, v in res.items()) + f" pair_chi2={chi:.1f}"
          f" max_overlap={max(overlap.values()):.2e}")
    assert all(abs(v) < bar for v in res.values()), res
    assert chi < 255 + 6 * math.sqrt(2 * 255)
    assert max(overlap.values()) < 1e-4, overlap


PEAKS_V2 = [5.0, 4.7, 4.2, 3.8, 3.2, 2.6, 2.0, 1.5]       # acceptance ~0.24 (v2 onward)
PEAKS_V1 = [3.2, 2.9, 2.5, 2.2, 1.8, 1.5, 1.2, 1.0]       # smoke v1's regime, acceptance ~0.135


def _outcome_problem(P, peaks=PEAKS_V2):
    """One request's rows (K target rows + bonus row) repeated P times; drafts = each position's 2nd-ranked token."""
    vocab = 5120                       # 32 VP_BLOCKs of 160: the smallest TP1 VPS head
    g = torch.Generator().manual_seed(77)
    req = torch.randn(K + 1, vocab, generator=g) * 0.5
    for j in range(K + 1):
        top = torch.randperm(vocab, generator=g)[:8]
        req[j, top] = torch.tensor(peaks) + 0.1 * j
    req = req.bfloat16()
    logits = req.repeat(P, 1)
    rows = torch.arange(P * (K + 1)).view(P, K + 1)
    second = [int(torch.sort(req[j].float(), descending=True, stable=True).indices[1]) for j in range(K)]
    md = NS(bonus_logits_indices=rows[:, K].contiguous().to(torch.int32),
            target_logits_indices=rows[:, :K].reshape(-1).contiguous().to(torch.int32),
            draft_token_ids=torch.tensor(second * P, dtype=torch.int32),
            draft_lengths=torch.full((P,), K, dtype=torch.int32),
            segment_ids=torch.arange(P).repeat_interleave(K).to(torch.int32),
            group_indices=torch.arange(K).repeat(P).to(torch.int32))
    params = (torch.ones(P, 1, dtype=torch.bfloat16), torch.full((P, 1), 20, dtype=torch.int32),
              torch.full((P, 1), 0.95, dtype=torch.float32))
    return vocab, req, logits, md, params, second


def _analytic(req, params, second):
    """Per target position j: the target distribution over tokens (the verify's own masked softmax), and the bonus
    row's sampling distribution."""
    temps, top_k, top_p = (p[:1] for p in params)
    cv, ci = vp.local_candidates(req, 0)
    pt = []
    for j in range(K):
        m = vp._candidate_mask(vp._scaled_target(cv[j:j + 1], temps), top_k, top_p)
        pt.append(dict(zip(ci[j].tolist(), torch.softmax(m, dim=-1)[0].tolist())))
    mb = vp._candidate_mask(cv[K:K + 1] / temps, top_k, top_p).float()
    pb = dict(zip(ci[K].tolist(), torch.softmax(mb, dim=-1)[0].tolist()))
    return pt, pb


def _gof(counts: dict, probs: dict, total: int):
    """Chi-square of observed token counts against probabilities (expected < 5 pooled); (stat, df)."""
    keys = [k for k, p in probs.items() if p * total >= 5]
    stat, obs_rest, exp_rest = 0.0, total, total
    for k in keys:
        e = probs[k] * total
        stat += (counts.get(k, 0) - e) ** 2 / e
        obs_rest -= counts.get(k, 0)
        exp_rest -= e
    df = len(keys) - 1
    if exp_rest >= 5:
        stat += (obs_rest - exp_rest) ** 2 / exp_rest
        df += 1
    return stat, max(df, 1)


def test_vrng_accept_reject_outcomes(ident_gather):
    """>= 1e6 accept / reject decisions per arm of the real verify (vp_verify on the request's candidates; served
    params temperature 1, top_k 20, top_p 0.95): the VRNG arm (counter + 1 per call) and the torch.rand arm (today's
    full-vocabulary draws) each match the analytic target -- acceptance per position (z < 5), accepted-length
    histogram, recovered token at position 0, bonus token (chi-square < df + 6 sqrt(2 df) + 10) -- and each other
    (2-sample chi-square on the accepted-length histogram)."""
    _outcome_run("v2", PEAKS_V2, 2000, al_bar=K + 6 * math.sqrt(2 * K) + 10, z_bar=5.0)


def test_vrng_accept_reject_outcomes_v1_regime(ident_gather):
    """L4-E's confirmatory run: smoke v1's exact outcome setup (peaks PEAKS_V1, acceptance ~0.135) at 2x its rows
    (2000 calls x 1024 = 2.05M rows per arm), under the rule pre-stated before v2 read: accepted-length chi-square
    p >= 0.01 (< 11.34 on df 3) for VRNG and torch.rand, the 2-sample chi-square < 11.34, per-position |z| < 3.
    Run on its own (run_cpu.sh mode outcome1); P1 deselects it."""
    _outcome_run("v1x2", PEAKS_V1, 2000, al_bar=11.34, z_bar=3.0, two_bar=11.34)


def _outcome_run(regime, peaks, calls, al_bar, z_bar, two_bar=None):
    P = 1024
    vocab, req, logits, md, params, second = _outcome_problem(P, peaks)
    cv, ci = vp.local_candidates(logits, 0)
    pt, pb = _analytic(req, params, second)
    pacc = [pt[j].get(second[j], 0.0) for j in range(K)]
    al_p = [math.prod(pacc[:a]) * (1 - pacc[a]) for a in range(K)] + [math.prod(pacc)]
    T = P * K
    gen = torch.Generator().manual_seed(2024)
    c0 = vps_rng.vps_initial_counter(5)
    res = {}
    for arm in ("vrng", "rand"):
        hist = [0] * (K + 1)
        reached, accepted = [0] * K, [0] * K
        rec0, bonus = {}, {}
        for i in range(calls):
            if arm == "vrng":
                ct = torch.tensor(((c0 + i + 2**31) % 2**32) - 2**31, dtype=torch.int32)
                out = vp.vp_verify(cv, ci, *common(md, params, None, None, None), draws=(ct, vocab, vps_rng.counter_uniform_at))
            else:
                bu = torch.rand(P, vocab, generator=gen)
                au = torch.rand(T, generator=gen)
                ru = torch.rand(T, vocab, generator=gen)
                out = vp.vp_verify(cv, ci, *common(md, params, bu, au, ru))
            d = torch.tensor(second, dtype=torch.int32)
            acc_mask = out[:, :K] == d            # accepted positions (a recovered token never equals its draft)
            first_rej = torch.where(~acc_mask, torch.arange(K).expand(P, K), torch.full((P, K), K)).min(dim=1).values
            al = torch.bincount(first_rej, minlength=K + 1).tolist()
            hist = [h + a for h, a in zip(hist, al)]
            for j in range(K):
                reached[j] += int((first_rej >= j).sum())
                accepted[j] += int((first_rej > j).sum())
            r0 = out[first_rej == 0, 0].tolist()
            for t in r0:
                rec0[t] = rec0.get(t, 0) + 1
            for t in out[first_rej == K, K].tolist():
                bonus[t] = bonus.get(t, 0) + 1
        n = P * calls
        hist_chi = sum((h - n * p) ** 2 / (n * p) for h, p in zip(hist, al_p))
        z = [abs(accepted[j] - reached[j] * pacc[j]) / math.sqrt(reached[j] * pacc[j] * (1 - pacc[j])) for j in range(K)]
        q0 = {t: p / (1 - pacc[0]) for t, p in pt[0].items() if t != second[0]}
        rec_chi, rec_df = _gof(rec0, q0, hist[0])
        bon_chi, bon_df = _gof(bonus, pb, hist[K])
        res[arm] = (hist, z, rec_chi, rec_df, bon_chi, bon_df)
        print(f"\nVRNG_OUTCOME regime={regime} arm={arm} rows={n} decisions={sum(reached)} pacc={[round(p, 6) for p in pacc]} "
              f"al_hist={hist} expect={[round(n * p) for p in al_p]} al_chi2={hist_chi:.2f} z={[round(x, 2) for x in z]} "
              f"rec0_chi2={rec_chi:.1f}/df{rec_df} bonus_chi2={bon_chi:.1f}/df{bon_df}")
        assert n >= 1_000_000 and sum(reached) >= 1_000_000
        assert hist_chi < al_bar, (arm, hist, hist_chi)
        assert max(z) < z_bar, (arm, z)
        assert rec_chi < rec_df + 6 * math.sqrt(2 * rec_df) + 10, (arm, rec_chi, rec_df)
        assert bon_chi < bon_df + 6 * math.sqrt(2 * bon_df) + 10, (arm, bon_chi, bon_df)
    ha, hb = res["vrng"][0], res["rand"][0]
    tot = sum(ha) + sum(hb)
    two = 0.0
    for a, b in zip(ha, hb):
        col = a + b
        for obs, rowt in ((a, sum(ha)), (b, sum(hb))):
            e = rowt * col / tot
            two += (obs - e) ** 2 / e
    print(f"VRNG_OUTCOME regime={regime} two_sample_al_chi2={two:.2f} df={K}")
    assert two < (K + 6 * math.sqrt(2 * K) + 10 if two_bar is None else two_bar)


# =================================================== VRNG: runner, controller =======================================
class FakeBatch:
    def __init__(self, params, P):
        temps, top_k, top_p = params
        self.req_ids = [f"r{i}" for i in range(P)]
        self.temperature_cpu_tensor = temps.squeeze(-1).float().clone()
        self.top_k_cpu_tensor = top_k.squeeze(-1).clone()
        self.top_p_cpu_tensor = top_p.squeeze(-1).clone()
        self.generators = {}


class Runner:
    """The runner's real VPS verify methods on a fake runner (TP1 head of width TV; compiled programs replaced by
    their eager bodies in the `rt` fixture)."""

    _sample_spec_verify_vps = tr.TPUModelRunner._sample_spec_verify_vps
    _vps_verify_vrng = tr.TPUModelRunner._vps_verify_vrng
    _vps_verify_fb = tr.TPUModelRunner._vps_verify_fb
    _vrng_ro_key = tr.TPUModelRunner._vrng_ro_key
    _padded_sampling_params = tr.TPUModelRunner._padded_sampling_params
    _upload_unless_unchanged = tr.TPUModelRunner._upload_unless_unchanged
    _upload_keyed = tr.TPUModelRunner._upload_keyed
    _precompile_vps_vrng = tr.TPUModelRunner._precompile_vps_vrng
    # v5 composition (V5COMP-BUILD; tests only): VPSHOST T-1(i)'s VPS_FALLBACK helper (the stock _vps_verify_fb uses it)
    _log_vps_fallback = tr.TPUModelRunner._log_vps_fallback

    def __init__(self, params, P, fb=True, cand=False, vrng=None, warm=True):
        self.input_batch = FakeBatch(params, P)
        self.vocab_size = TV
        self.device = CPU
        self.speculative_config = NS(num_speculative_tokens=K)
        self.rejection_sampler = NS(synthetic_conditional_rates=None, synthetic_mode=False)
        self._vps_head = NS(full_vocab=TV, tp_size=1, shard_offset=0, shard_width=TV)
        self._vps_fb_max, self._vps_tiegap, self._vps_cand_draws = (8, 0, cand) if fb else (0, 0, False)
        self._vps_fb_path = fb
        self._vps_fb_stats, self._vps_fb_chunks, self._vps_checks_left = None, 0, 0
        self._h2d_reuse = {}
        self._vfy = None
        self._vrng = vrng
        self._vps_fb_async = l4_vpshost.AsyncCounts()  # v5 composition: VPSHOST T-1(i)'s state (statsasync off)
        if vrng is not None:
            # q38 L4 AGREE port (v5 composition, V5COMP-BUILD): as the runner registers VRNG-VPS at its build
            l4_agree.register(self, vrng)
        self.warmup_done = warm
        self.checks = []

    def _check_vps(self, out, local_logits, args, s, e, force_full=False):
        self.checks.append((args, force_full))


@pytest.fixture
def rt(monkeypatch, ident_gather):
    """Compiled programs -> eager bodies; every L4 / L6 lever env unset; log lines captured."""
    for k in list(os.environ):
        if k.startswith(("VLLM_TPU_L6_", "VLLM_TPU_L4_", "VLLM_TPU_VPS_INGRAPH")):
            monkeypatch.delenv(k)
    l6_ab._resolve()
    monkeypatch.setattr(tr, "_vps_verify_fb_compiled", tr._vps_verify_fb_impl)
    monkeypatch.setattr(tr, "_vps_verify_compiled", tr._vps_verify_impl)
    monkeypatch.setattr(tr, "_vps_verify_rng_compiled", tr._vps_verify_rng_impl)
    lines = []
    monkeypatch.setattr(l4_samp2.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    monkeypatch.setattr(l4_samp2.logger, "error", lambda fmt, *a: lines.append("ERROR " + fmt % a))
    yield NS(lines=lines)
    l6_ab._resolve()


def mkctl(seed=0, ab="1"):
    ctl = l4_samp2.VrngCtl(seed, CPU, ab=ab)
    ctl.event_fn = lambda: NS(query=lambda: True)     # torch.tpu.Event stand-in (CPU)
    return ctl


def warm_for(ctl, logits, md, fb, F):
    ctl.warm_key(ctl.key(logits, md.bonus_logits_indices.shape[0], md.target_logits_indices.shape[0],
                         tuple(md.draft_token_ids.shape), fb, F))


def run_calls(r, problems, gen, F=4, start=0):
    """One VPS verify per engine step: the runner's agreement clock ticks first (execute_model's hook; q38 L4 AGREE
    port, v5 composition)."""
    outs = []
    for logits, md, P in problems:
        agree = r.__dict__.get("_l4_agree")
        if agree is not None:
            agree.tick(r)
        outs.append(r._sample_spec_verify_vps(logits, md, start, P, gen, F))
    return outs


def problems_for(seed, n, P=8, n_range=1):
    out = []
    for i in range(n):
        logits, md, params = vprob(P, seed * 1000 + i, n_range=n_range)
        out.append((logits, md, P))
    return out, params


def same_params_problems(seed, n, P=8):
    """n problems sharing one set of sampling params (the fake batch holds one set)."""
    _, _, params = vprob(P, seed, n_range=1)
    out = []
    for i in range(n):
        logits, md, _ = vprob(P, seed * 1000 + i, n_range=1)
        out.append((logits, md, P))
    return out, params


@pytest.mark.parametrize("fb", [True, False])
def test_vrng_path_tokens_and_generator(rt, fb):
    """Flag on with warmed keys: every serving call runs the VRNG program, its tokens equal the base verify's on the
    counter's full draws (computed here), the counter advances once per call, and the sampling generator is never
    advanced; flag off advances the generator by the three draws per call."""
    probs, params = same_params_problems(4, 6)
    F = 4 if fb else 0
    ctl = mkctl(seed=3)
    r = Runner(params, 8, fb=fb, vrng=ctl)
    for logits, md, _ in probs:
        warm_for(ctl, logits, md, fb, F)
    gen = torch.Generator().manual_seed(9)
    s0 = gen.get_state().clone()
    for i, (logits, md, P) in enumerate(probs):
        c_before = ctl.counter_word()
        r._l4_agree.tick(r)  # q38 L4 AGREE port (v5 composition): the engine step's clock tick before its VPS call
        out = r._sample_spec_verify_vps(logits, md, 0, P, gen, F)
        ct = torch.tensor(c_before, dtype=torch.int32)
        bu, au, ru = vps_rng.full_draws(ct, P, tuple(md.draft_token_ids.shape), P * K, TV)
        temps, top_k, top_p = r._padded_sampling_params(0, P, P, logits.dtype, CPU)
        args = common(md, (temps, top_k, top_p), bu, au, ru)
        if fb:
            ref, _ = tr._vps_verify_fb_impl(logits, 0, *args, torch.tensor([P], dtype=torch.int32), F, False, None)
        else:
            ref = tr._vps_verify_impl(logits, 0, *args)
        assert torch.equal(out, ref), i
        assert (ctl.counter_word() - vps_rng.vps_initial_counter(3)) % 2**32 == i + 1
        assert int(ctl.counter) == ctl.counter_word()
    assert torch.equal(gen.get_state(), s0)
    assert ctl.stats["vrng"] == len(probs) and ctl.vrng_calls == len(probs)
    off = Runner(params, 8, fb=fb)
    gen2 = torch.Generator().manual_seed(9)
    run_calls(off, probs, gen2, F)
    ref_gen = torch.Generator().manual_seed(9)
    for logits, md, P in probs:
        torch.rand(P, TV, generator=ref_gen), torch.rand(md.draft_token_ids.shape, generator=ref_gen)
        torch.rand(P * K, TV, generator=ref_gen)
    assert torch.equal(gen2.get_state(), ref_gen.get_state())


def test_vrng_seeded_reproducibility(rt):
    """Same seed -> the same token sequence and counters; another seed -> different tokens; identical inputs on
    successive calls draw different uniforms (the counter moves)."""
    probs, params = same_params_problems(5, 12)
    seqs = {}
    for name, seed in (("a", 11), ("b", 11), ("c", 12)):
        ctl = mkctl(seed=seed)
        r = Runner(params, 8, vrng=ctl)
        for logits, md, _ in probs:
            warm_for(ctl, logits, md, True, 4)
        seqs[name] = (run_calls(r, probs, torch.Generator().manual_seed(0)), ctl.counter_word(), int(ctl.counter))
    assert all(torch.equal(x, y) for x, y in zip(seqs["a"][0], seqs["b"][0])) and seqs["a"][1:] == seqs["b"][1:]
    assert any(not torch.equal(x, y) for x, y in zip(seqs["a"][0], seqs["c"][0]))
    ctl = mkctl(seed=11)
    r = Runner(params, 8, vrng=ctl)
    logits, md, P = probs[0]
    warm_for(ctl, logits, md, True, 4)
    rep = run_calls(r, [probs[0]] * 6, torch.Generator().manual_seed(0))
    assert len({tuple(o.reshape(-1).tolist()) for o in rep}) > 1


def test_vrng_warmup_calls_and_precompile_keep_the_serving_counter(rt):
    """Before warmup_done every call takes the base path and never counts; `_precompile_vps_vrng` runs the program on
    a dummy counter and records the warmed key without creating or moving the serving counter."""
    probs, params = same_params_problems(6, 2)
    ctl = mkctl()
    r = Runner(params, 8, vrng=ctl, warm=False)
    logits, md, P = probs[0]
    temps, top_k, top_p = r._padded_sampling_params(0, P, P, logits.dtype, CPU)
    args = common(md, (temps, top_k, top_p), None, None, None)
    r._precompile_vps_vrng(logits, args, P, 4, None)
    assert ctl.key(logits, P, P * K, tuple(md.draft_token_ids.shape), True, 4) in ctl.warm
    assert ctl.counter is None and ctl.vrng_calls == 0
    gen = torch.Generator().manual_seed(1)
    run_calls(r, probs, gen)
    assert ctl.serving_calls == 0 and ctl.vrng_calls == 0 and ctl.counter is None and ctl.boundaries == 0


def test_vrng_cold_key_takes_the_base_path(rt):
    probs, params = same_params_problems(7, 3)
    ctl = mkctl()
    r = Runner(params, 8, vrng=ctl)
    logits, md, P = probs[0]
    warm_for(ctl, logits, md, True, 0)          # warmed at F = 0 only
    gen = torch.Generator().manual_seed(1)
    s0 = gen.get_state().clone()
    run_calls(r, probs, gen, F=4)
    assert ctl.stats["cold"] == 3 and ctl.vrng_calls == 0 and not torch.equal(gen.get_state(), s0)
    run_calls(r, probs[:1], gen, F=0)
    assert ctl.stats["vrng"] == 1 and ctl.vrng_calls == 1


def test_vrng_seeded_request_runs_the_base_path(rt):
    """A chunk holding a request with its own SamplingParams.seed takes the base path (counted `seeded`), so its
    tokens and the generator are bitwise those of the flag-off run; a chunk without one runs VRNG."""
    probs, params = same_params_problems(8, 5)
    ctl = mkctl()
    on = Runner(params, 8, vrng=ctl)
    on.input_batch.generators = {3: torch.Generator().manual_seed(42)}
    for logits, md, _ in probs:
        warm_for(ctl, logits, md, True, 4)
    off = Runner(params, 8)
    g_on, g_off = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    a = run_calls(on, probs, g_on)
    b = run_calls(off, probs, g_off)
    assert all(torch.equal(x, y) for x, y in zip(a, b)) and torch.equal(g_on.get_state(), g_off.get_state())
    assert ctl.stats["seeded"] == len(probs) and ctl.vrng_calls == 0 and ctl.counter is None
    on.input_batch.generators = {9: torch.Generator()}          # outside the chunk [0, 8)
    run_calls(on, probs[:1], g_on)
    assert ctl.stats["vrng"] == 1
    assert l4_samp2.seeded_chunk(NS(generators={2: None}), 0, 8) and not l4_samp2.seeded_chunk(NS(generators={}), 0, 8)
    assert not l4_samp2.seeded_chunk(NS(generators={8: None}), 0, 8) and l4_samp2.seeded_chunk(NS(generators={8: None}), 4, 9)


def test_vrng_counter_wrap(rt):
    """The int32 device counter wraps like the host word (2^31 - 1 -> -2^31); after 2^32 VRNG calls (every counter
    value used once) the run takes the base path, counted `wrapped`, with one line: no (counter, element) pair is
    reused within a run."""
    probs, params = same_params_problems(9, 4)
    ctl = mkctl()
    ctl.counter0 = 2**31 - 1
    r = Runner(params, 8, vrng=ctl)
    for logits, md, _ in probs:
        warm_for(ctl, logits, md, True, 4)
    gen = torch.Generator().manual_seed(1)
    run_calls(r, probs[:1], gen)
    assert int(ctl.counter) == -2**31 == ctl.counter_word()
    ctl2 = mkctl()
    ctl2.vrng_calls = 2**32 - 2
    r2 = Runner(params, 8, vrng=ctl2)
    for logits, md, _ in probs:
        warm_for(ctl2, logits, md, True, 4)
    used = []
    for p in probs[:3]:
        used.append(ctl2.counter_word())
        run_calls(r2, [p], gen)
        assert ctl2.counter is None or int(ctl2.counter) == ctl2.counter_word()
    assert ctl2.vrng_calls == 2**32 and ctl2.stats["vrng"] == 2 and ctl2.stats["wrapped"] == 1
    assert used[0] == ((ctl2.counter0 - 2 + 2**31) % 2**32) - 2**31
    assert sum("counter's 2**32 values are used" in ln for ln in rt.lines) == 1
    run_calls(r2, probs[3:], gen)
    assert ctl2.stats["wrapped"] == 2 and sum("counter's 2**32 values are used" in ln for ln in rt.lines) == 1


def test_vrng_alt_windows_and_modes(rt, monkeypatch):
    """alt:4 on the runner's engine-step clock (q38 L4 AGREE port: window k = n // 4) -> steps 1-3 VRNG, 4-7 base,
    8-11 VRNG, 12-15 base, 16 VRNG, an agreement at step 1 and every window start (4, 8, 12, 16); '0' never runs
    VRNG; '1' always (agreement every AGREE_EVERY steps); the l6_ab window line counts vrng.calls."""
    probs, params = same_params_problems(10, 16)
    for ab, want in (("alt:4", [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1]), ("0", [0] * 16), ("1", [1] * 16)):
        # Any l6_ab lever arms l6_ab (note counts). The composed tree's LEVERS has no VPSHOST lever, so the port of
        # c152b0199's VLLM_TPU_L4_VPS_STATS_ASYNC armed nothing (smoke v1 j-07b875aa); GC_FREEZE is a lever here and
        # only core/l6_jitter.py reads it, which this test never calls.
        monkeypatch.setenv("VLLM_TPU_L6_GC_FREEZE", "1")
        monkeypatch.setenv("VLLM_TPU_L6_AB_STEPS", "1000000")
        l6_ab._resolve()
        l6_ab.begin_step()
        ctl = mkctl(ab=ab)
        r = Runner(params, 8, vrng=ctl)
        for logits, md, _ in probs:
            warm_for(ctl, logits, md, True, 4)
        got = []
        for p in probs:
            before = ctl.vrng_calls
            run_calls(r, [p], torch.Generator().manual_seed(0))
            got.append(ctl.vrng_calls - before)
        assert got == want, (ab, got)
        assert ctl.boundaries == (5 if ab == "alt:4" else 1)
        assert int(l6_ab._state["counts"].get("vrng.calls", 0)) == sum(want)
        monkeypatch.delenv("VLLM_TPU_L6_GC_FREEZE")
        monkeypatch.delenv("VLLM_TPU_L6_AB_STEPS")
        l6_ab._resolve()
    assert sum("window 1: vrng off" in ln for ln in rt.lines) >= 1


def test_vrng_agreement_rules(rt):
    """Rows from every rank decide: a wish of 0 -> base for the window (not failed); a local disable, an unequal
    warm-key fingerprint or an unequal counter -> base for the rest of the run, one ERROR."""
    def fresh():
        c = mkctl()
        c.serving_calls = 1
        return c

    c = fresh()
    c.apply([[1, 0, 5, 7], [0, 0, 5, 7]])
    assert not c.agreed and not c.failed
    c.apply([[1, 0, 5, 7], [1, 0, 5, 7]])
    assert c.agreed and not c.failed
    for bad in ([[1, 1, 5, 7], [1, 0, 5, 7]], [[1, 0, 5, 7], [1, 0, 6, 7]], [[1, 0, 5, 7], [1, 0, 5, 8]]):
        c = fresh()
        n0 = len([ln for ln in rt.lines if ln.startswith("ERROR")])
        c.apply(bad)
        c.apply([[1, 0, 5, 7], [1, 0, 5, 7]])
        assert c.failed and not c.agreed, bad
        assert len([ln for ln in rt.lines if ln.startswith("ERROR")]) == n0 + 1
    c = mkctl()
    assert c.local_vector() == [1, 0, c.fingerprint(), c.counter_word()]


def test_vrng_gather_failure_stops(rt, monkeypatch):
    """A failing CPU all_gather raises (fail-stop) instead of letting ranks diverge."""
    import vllm.distributed.parallel_state as ps

    monkeypatch.setattr(ps, "get_tp_group", lambda: NS(world_size=2, cpu_group=None))

    def boom(*a, **k):
        raise RuntimeError("gloo down")

    monkeypatch.setattr(torch.distributed, "all_gather", boom)
    with pytest.raises(RuntimeError, match="could not agree"):
        l4_agree.gather([1, 0, 3, 4])  # q38 L4 AGREE port: VRNG's gather is the runner's one agreement gather
    assert not hasattr(l4_samp2, "_gather")


class Exchange:
    """An all_gather between threads: every rank's vector, in rank order, to every rank."""

    def __init__(self, n):
        self.n, self.slots, self.bar = n, [None] * n, threading.Barrier(n)
        self.tl = threading.local()

    def gather(self, vec):
        self.slots[self.tl.rank] = list(vec)
        self.bar.wait(timeout=600)
        rows = [list(s) for s in self.slots]
        self.bar.wait(timeout=600)
        return rows


def run_ranks(monkeypatch, nranks, ncalls, ab, offset=None, drift=None):
    """`nranks` simulated TP ranks (threads) each run `ncalls` VPS verifies of the same chunks; returns per rank
    (tokens per call, controller, generator state)."""
    ex = Exchange(nranks)
    monkeypatch.setattr(l4_agree, "gather", ex.gather)  # q38 L4 AGREE port: the runner's one agreement gather
    monkeypatch.setattr(l4_samp2, "_tp_rank", lambda: ex.tl.rank)
    probs, params = same_params_problems(12, ncalls)
    res, errs = [None] * nranks, []

    def body(rank):
        try:
            ex.tl.rank = rank
            ctl = mkctl(seed=21, ab=ab)
            if offset is not None and rank == offset:
                ctl.counter0 += 1
            r = Runner(params, 8, vrng=ctl)
            for logits, md, _ in probs:
                warm_for(ctl, logits, md, True, 4)
            gen = torch.Generator().manual_seed(1234)
            outs = []
            for i, p in enumerate(probs):
                if drift is not None and rank == drift[0] and i == drift[1]:
                    ctl.counter = ctl.device_counter() + 1
                    ctl.vrng_calls += 1
                outs.append(run_calls(r, [p], gen)[0])
            res[rank] = (outs, ctl, gen.get_state())
        except BaseException as exc:  # noqa: BLE001
            errs.append((rank, repr(exc)))
            ex.bar.abort()

    nthreads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        ths = [threading.Thread(target=body, args=(k,)) for k in range(nranks)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
    finally:
        torch.set_num_threads(nthreads)
    assert errs == []
    return res


def test_vrng_eight_ranks_equal_counters_and_tokens(rt, monkeypatch):
    """8 simulated TP ranks, alt:8 windows over 40 engine steps (q38 L4 AGREE port: window k = n // 8, so steps 1-7,
    16-23 and 32-39 run VRNG: 23 calls; agreements at steps 1, 8, 16, 24, 32, 40): every call's tokens are equal on
    every rank, the device counters and host words are equal at the end, and the generators agree."""
    res = run_ranks(monkeypatch, 8, 40, "alt:8")
    outs0, ctl0, g0 = res[0]
    for rank, (outs, ctl, g) in enumerate(res):
        assert all(torch.equal(a, b) for a, b in zip(outs, outs0)), rank
        assert int(ctl.counter) == int(ctl0.counter) == ctl.counter_word() and torch.equal(g, g0)
        assert ctl.vrng_calls == 23 and not ctl.failed and ctl.boundaries == 6
    print(f"\nVRNG_RANKS ranks=8 calls=40 vrng_calls={ctl0.vrng_calls} counter={int(ctl0.counter)} "
          f"boundaries={ctl0.boundaries}")


def test_vrng_one_rank_counter_offset_is_caught(rt, monkeypatch):
    """One rank's counter offset by 1 from the start: the first agreement sees unequal counters, every rank logs one
    ERROR and keeps the base path for the run, so the tokens stay equal across ranks. An offset that appears mid-run
    (a missed advance on rank 3 at call 11) is caught at the next window start (q38 L4 AGREE port: engine step 16)."""
    res = run_ranks(monkeypatch, 8, 24, "alt:8", offset=5)
    outs0 = res[0][0]
    for rank, (outs, ctl, _) in enumerate(res):
        assert ctl.failed and ctl.vrng_calls == 0 and all(torch.equal(a, b) for a, b in zip(outs, outs0)), rank
    assert sum(ln.startswith("ERROR") and "unequal counters" in ln for ln in rt.lines) == 8
    res = run_ranks(monkeypatch, 8, 32, "alt:8", drift=(3, 10))
    for rank, (outs, ctl, _) in enumerate(res):
        assert ctl.failed and ctl.stats["failed"] == 17, (rank, dict(ctl.stats))     # steps 16..32
        assert all(torch.equal(a, b) for a, b in zip(outs[15:], res[0][0][15:])), rank


class RngCount(TorchDispatchMode):
    """Counts RNG dispatches, split into eager (outside the program stand-ins) and in-program."""

    NAMES = ("uniform", "rand", "bernoulli", "normal", "exponential", "random_", "multinomial", "poisson")

    def __init__(self):
        super().__init__()
        self.eager = self.inprog = 0
        self.inside = False
        self.seen = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        if any(t in name for t in self.NAMES):
            self.seen.append(name)
            if self.inside:
                self.inprog += 1
            else:
                self.eager += 1
        return func(*args, **(kwargs or {}))


def _program(mode, fn):
    def wrapped(*a, **k):
        mode.inside = True
        try:
            return fn(*a, **k)
        finally:
            mode.inside = False
    return wrapped


@pytest.mark.parametrize("cfg", [("fb_full_F0", True, False, 0), ("fb_full_F4", True, False, 4),
                                 ("fb_cand_F4", True, True, 4), ("plain", False, False, 0)])
def test_vrng_aot_census(rt, monkeypatch, cfg):
    """Eager RNG dispatches per VPS call: today 3 (4 with the fallback rows' fb_u under candidate draws), VRNG 0; no
    RNG op inside either program; and the fullgraph capture of `_vps_verify_rng_impl` (one graph, no break) holds no
    RNG node."""
    name, fb, cand, F = cfg
    probs, params = same_params_problems(13, 3)
    counts = {}
    for arm in ("base", "vrng"):
        mode = RngCount()
        monkeypatch.setattr(tr, "_vps_verify_fb_compiled", _program(mode, tr._vps_verify_fb_impl))
        monkeypatch.setattr(tr, "_vps_verify_compiled", _program(mode, tr._vps_verify_impl))
        monkeypatch.setattr(tr, "_vps_verify_rng_compiled", _program(mode, tr._vps_verify_rng_impl))
        ctl = mkctl() if arm == "vrng" else None
        r = Runner(params, 8, fb=fb, cand=cand, vrng=ctl)
        if ctl is not None:
            for logits, md, _ in probs:
                warm_for(ctl, logits, md, fb, F)
        with mode:
            run_calls(r, probs, torch.Generator().manual_seed(0), F=F)
        counts[arm] = (mode.eager / len(probs), mode.inprog)
        if ctl is not None:
            assert ctl.vrng_calls == len(probs)
    want_base = 4 if (cand and F > 0) else 3
    assert counts["base"] == (want_base, 0) and counts["vrng"] == (0, 0), counts
    graphs = []

    def capture(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    logits, md, P = probs[0]
    temps, top_k, top_p = Runner(params, 8)._padded_sampling_params(0, P, P, logits.dtype, CPU)
    args = common(md, (temps, top_k, top_p), None, None, None)
    torch._dynamo.reset()
    fn = torch.compile(tr._vps_verify_rng_impl, backend=capture, fullgraph=True, dynamic=False)
    fn(logits, 0, *args[:5], torch.tensor(5, dtype=torch.int32), *args[8:], torch.tensor([P], dtype=torch.int32),
       F, fb, TV)
    nodes = [n for g in graphs for n in g.graph.nodes if n.op in ("call_function", "call_method")]
    rng_nodes = [str(n.target) for n in nodes if any(t in str(n.target) for t in RngCount.NAMES)]
    print(f"\nVRNG_CENSUS cfg={name} base_eager_rng_per_call={counts['base'][0]:g} vrng_eager_rng_per_call="
          f"{counts['vrng'][0]:g} in_program_rng={counts['base'][1] + counts['vrng'][1]} aot_graphs={len(graphs)} "
          f"aot_nodes={len(nodes)} aot_rng_nodes={len(rng_nodes)}")
    assert len(graphs) == 1 and nodes and rng_nodes == []


def test_vrng_al_canary(rt, monkeypatch):
    """The program's per-call [accepted, drafted, rows] summed on device equal the tokens' own counts; the timed line
    carries both paths' host counters and the device AL once its copy has landed."""
    probs, params = same_params_problems(14, 8)
    ctl = mkctl(ab="alt:4")
    ctl.event_fn = lambda: NS(query=lambda: True)
    r = Runner(params, 8, vrng=ctl)
    for logits, md, _ in probs:
        warm_for(ctl, logits, md, True, 4)
    t = [1000.0]
    monkeypatch.setattr(l4_samp2.time, "monotonic", lambda: t[0])
    ctl.t_log = t[0]
    want = [0, 0, 0]
    gen = torch.Generator().manual_seed(0)
    vrng_steps = (0, 1, 2, 7)  # q38 L4 AGREE port: alt:4 on engine steps 1-8 -> VRNG at steps 1-3 and 8
    for i, p in enumerate(probs):
        out = run_calls(r, [p], gen)[0]
        if i in vrng_steps:
            want = [a + b for a, b in zip(want, _al_expect(out, p[1], 8))]
    assert ctl.al_dev.tolist() == want and want[2] > 0
    t[0] += 11
    ctl.maybe_log()          # issues the copy
    t[0] += 11
    ctl.maybe_log()          # logs the landed counts
    al_lines = [ln for ln in rt.lines if "VRNG-VPS AL" in ln]
    assert len(al_lines) == 2
    last = al_lines[-1]
    drafted = int(sum(int(x) for i, p in enumerate(probs) if i in vrng_steps for x in p[1].draft_lengths_cpu))
    base_drafted = int(sum(int(x) for i, p in enumerate(probs) if i not in vrng_steps for x in p[1].draft_lengths_cpu))
    assert f"vrng calls=4 rows=32 drafted={drafted} | base calls=4 rows=32 drafted={base_drafted}" in last, last
    assert f"device(vrng) accepted={want[0]} drafted={want[1]} rows={want[2]} al={1 + want[0] / want[2]:.4f}" in last
    print(f"\nVRNG_AL {last}")


def test_vrng_check_mode_uses_the_counter_draws(rt):
    """VLLM_TPU_VPS_CHECK under VRNG: the check gets the counter's full draws (force_full); a device counter that
    disagrees with the host count disables the rank (taking effect at the next agreement)."""
    probs, params = same_params_problems(15, 3)
    ctl = mkctl(ab="alt:3")  # q38 L4 AGREE port: window k = n // 3 -> steps 1-2 VRNG, step 3 opens window 1
    r = Runner(params, 8, vrng=ctl)
    r._vps_checks_left = 5
    for logits, md, _ in probs:
        warm_for(ctl, logits, md, True, 4)
    logits, md, P = probs[0]
    c_before = ctl.counter_word()
    run_calls(r, [probs[0]], torch.Generator().manual_seed(0))
    (args, force_full), = r.checks
    bu, au, ru = vps_rng.full_draws(torch.tensor(c_before, dtype=torch.int32), P, tuple(md.draft_token_ids.shape),
                                    P * K, TV)
    assert force_full and torch.equal(args[5], bu) and torch.equal(args[6], au) and torch.equal(args[7], ru)
    ctl.counter = ctl.counter + 5
    run_calls(r, [probs[1]], torch.Generator().manual_seed(0))
    assert ctl.local_disabled == ["device counter != host count"]
    run_calls(r, [probs[2]], torch.Generator().manual_seed(0))       # step 3 opens window 1: agreement
    assert ctl.failed


def test_vrng_runner_sites():
    """The runner builds the controller only with the flag and a VPS head, and the base programs keep their names."""
    src = open(tr.__file__).read()
    assert "if envs.VLLM_TPU_VPS_INGRAPH_RNG:\n                    self._vrng = _l4_samp2.VrngCtl(" in src
    assert "_vps_verify_rng_compiled = _static_shape_args(_tpu_compile(_vps_verify_rng_impl))" in src
    tree = ast.parse(src)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert {"_vps_verify_impl", "_vps_verify_fb_impl", "_vps_verify_rng_impl", "_vps_verify_vrng",
            "_precompile_vps_vrng"} <= names


logging.getLogger("vllm_torchtpu").setLevel(logging.WARNING)
