# SPDX-License-Identifier: Apache-2.0
"""q38 L4 S1 (VLLM_TPU_VPS_TOPK_KEYS): VPS local_candidates' "stable descending sort, first kc" steps as an
exact top-kc over tie-free int32 keys. The candidates (values and global ids, in order) must be bitwise those of
the sort path, ties and signed zeros included; f32 logits and the flag off keep the sort path.

vp_sampler is loaded through the VPS test harness (tests/q38mm_dstock/test_vp_sampler.py: stub packages, no
torch_tpu / vllm import), exactly as the existing VPS tests load it.
"""

import importlib.util
import os
import subprocess
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "q38_samp1_vps_harness", os.path.join(_HERE, "..", "q38mm_dstock", "test_vp_sampler.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)
VP = H.VP


@pytest.fixture(autouse=True)
def _harness():
    with H.HARNESS.active():
        yield


def _tie_heavy(rows, n, levels, seed):
    """bf16 rows drawn from a few levels, incl. both signed zeros, negatives and -inf: many exact ties."""
    g = torch.Generator().manual_seed(seed)
    pool = torch.tensor([-float("inf"), -3.0, -1.5, -0.0, 0.0, 0.5, 1.25, 2.0, 7.0][:levels], dtype=torch.float32)
    idx = torch.randint(0, pool.numel(), (rows, n), generator=g)
    return pool[idx].to(torch.bfloat16)


def _sorted_first(x, kc):
    return torch.sort(x, dim=-1, descending=True, stable=True).indices[:, :kc]


@pytest.mark.parametrize("n,bits,kc", [(194, 8, 32), (194, 8, 33), (5120, 13, 32), (5280, 13, 33)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_first_kc_equals_stable_sort(n, bits, kc, seed):
    g = torch.Generator().manual_seed(100 + seed)
    smooth = (torch.randn(16, n, generator=g) * 4).to(torch.bfloat16)
    for x in (smooth, _tie_heavy(16, n, 9, seed), _tie_heavy(16, n, 3, seed)):
        assert torch.equal(VP._first_kc_desc(x, kc, bits), _sorted_first(x, kc))


def test_signed_zeros_compare_equal_like_the_sort():
    x = torch.tensor([[-0.0, 0.0, -1.0], [0.0, -0.0, -1.0]], dtype=torch.bfloat16)
    assert torch.equal(VP._first_kc_desc(x, 1, 8), _sorted_first(x, 1))   # index 0 in both rows
    assert torch.equal(VP._first_kc_desc(x, 2, 8), _sorted_first(x, 2))


@pytest.mark.parametrize("kc", [32, 33])
@pytest.mark.parametrize("kind", ["smooth", "ties"])
def test_local_candidates_bitwise_equal_sort_path(monkeypatch, kc, kind):
    rows, vs, block = 16, 194 * 160, 160
    g = torch.Generator().manual_seed(kc)
    x = (torch.randn(rows, vs, generator=g) * 3).to(torch.bfloat16) if kind == "smooth" else _tie_heavy(rows, vs, 9, kc)
    monkeypatch.setattr(VP, "_TOPK_KEYS", False)
    v0, i0 = VP.local_candidates(x, 31040 * 3, kc, block)
    monkeypatch.setattr(VP, "_TOPK_KEYS", True)
    v1, i1 = VP.local_candidates(x, 31040 * 3, kc, block)
    assert torch.equal(i0, i1) and torch.equal(v0.view(torch.int16), v1.view(torch.int16))


def test_flag_off_and_f32_keep_the_sort(monkeypatch):
    calls = []
    real = VP._first_kc_desc
    monkeypatch.setattr(VP, "_first_kc_desc", lambda *a: calls.append(1) or real(*a))
    x16 = (torch.randn(4, 194 * 160) * 2).to(torch.bfloat16)
    monkeypatch.setattr(VP, "_TOPK_KEYS", False)
    VP.local_candidates(x16, 0, 32, 160)
    assert calls == []
    monkeypatch.setattr(VP, "_TOPK_KEYS", True)
    v32, i32 = VP.local_candidates(x16.float(), 0, 32, 160)
    assert calls == []
    monkeypatch.setattr(VP, "_TOPK_KEYS", False)
    v32b, i32b = VP.local_candidates(x16.float(), 0, 32, 160)
    assert torch.equal(i32, i32b) and torch.equal(v32, v32b)
    monkeypatch.setattr(VP, "_TOPK_KEYS", True)
    VP.local_candidates(x16, 0, 32, 160)
    assert calls == [1, 1]


def test_flag_read_at_import_in_a_fresh_process():
    path = os.path.join(_HERE, "..", "..", "src", "vllm_torchtpu", "layers", "adapter", "sample", "vp_sampler.py")
    code = ("import ast, os, sys; src = open(sys.argv[1]).read(); tree = ast.parse(src); "
            "ns = {'os': os}; node = [n for n in tree.body if isinstance(n, ast.Assign) and "
            "any(getattr(t, 'id', None) == '_TOPK_KEYS' for t in n.targets)][0]; "
            "exec(ast.get_source_segment(src, node), ns); print('KEYS', ns['_TOPK_KEYS'])")
    env = {k: v for k, v in os.environ.items() if k != "VLLM_TPU_VPS_TOPK_KEYS"}
    for extra, want in (({}, "KEYS False"), ({"VLLM_TPU_VPS_TOPK_KEYS": "1"}, "KEYS True"),
                        ({"VLLM_TPU_VPS_TOPK_KEYS": "on"}, "KEYS True"), ({"VLLM_TPU_VPS_TOPK_KEYS": "0"}, "KEYS False")):
        out = subprocess.run([sys.executable, "-I", "-c", code, path], env={**env, **extra},
                             capture_output=True, text=True, timeout=60)
        assert out.returncode == 0 and out.stdout.strip() == want, (extra, out.stdout, out.stderr)
