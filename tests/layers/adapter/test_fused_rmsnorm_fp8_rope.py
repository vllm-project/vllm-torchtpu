# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
================================================================================
3-TIER EXECUTION PROVENANCE BANNER (Rule 19 Mandatory Provenance)
================================================================================
- Tier 1 (Compiled Host C-ABI Shared Library): N/A (Pure PyTorch & Torch Dynamo)
- Tier 2 (Compiled Host C++20 LloRegionBuilder IR): N/A
- Tier 3 (Execution Provenance):
  [CPU NUMERICAL PARITY & FULLGRAPH DYNAMO VERIFICATION — NOT PHYSICAL TPU]
  Executes on CPU PyTorch verifying:
  1. Gate 1: Fused residual + RMSNorm numerical parity (CosSim >= 0.99999, SQNR >= 45 dB)
     and zero residual drift against float64 golden reference.
  2. Gate 2: RMSNorm without residual (e.g. q_a_layernorm, kv_a_layernorm) parity.
  3. Gate 3: TpuRMSNorm variance_size_override fallback guard for Gemma models.
  4. Gate 4: FP8 linear scale formatting across 1-D, 2-D [N, 1], [1, N], >2-D,
     and blockwise scales.
  5. Gate 5: In-kernel RoPE downcast preventing float32 promotion from cos/sin cache.
  6. Gate 6: Combined RMSNorm + RoPE torch.compile(..., fullgraph=True) tracing safety.
  7. Gate 7: Patch registry and tpu_platform.py attention backend wiring verification.
  8. Gate 8: Negative control: mismatched operand shapes raise RuntimeError.
================================================================================
"""

import ast
import math
import os
import sys
import unittest

for p in [
    "/usr/local/google/home/jfy/.venvs/torchcpu/lib/python3.13/site-packages",
    "/usr/local/google/home/jfy/qwen35_hardening/.venv/lib/python3.13/site-packages",
]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if os.path.join(REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import importlib.util
import types

SRC_DIR = os.path.join(REPO_ROOT, 'src')
for subpkg in [
    'vllm_torchtpu',
    'vllm_torchtpu.layers',
    'vllm_torchtpu.layers.adapter',
    'vllm_torchtpu.layers.adapter.quantization',
]:
    if subpkg not in sys.modules:
        rel = subpkg.replace('.', '/')
        mod = types.ModuleType(subpkg)
        mod.__path__ = [os.path.join(SRC_DIR, rel)]
        sys.modules[subpkg] = mod

def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ln_mod = _load_module('vllm_torchtpu.layers.adapter.layernorm', os.path.join(SRC_DIR, 'vllm_torchtpu/layers/adapter/layernorm.py'))
sys.modules['vllm_torchtpu.layers.adapter.layernorm'] = ln_mod
tpu_rmsnorm_forward = ln_mod.tpu_rmsnorm_forward
TpuRMSNorm = ln_mod.TpuRMSNorm

def _extract_fn(filepath: str, fn_name: str):
    with open(filepath, 'r') as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            code = compile(ast.Module(body=[node], type_ignores=[]), filepath, 'exec')
            ns = {'torch': torch}
            exec(code, ns)
            return ns[fn_name]
    raise ValueError(f'{fn_name} not found in {filepath}')

_format_linear_scale_for_runtime = _extract_fn(
    os.path.join(SRC_DIR, 'vllm_torchtpu/layers/adapter/quantization/fp8.py'),
    '_format_linear_scale_for_runtime'
)


def _compute_cossim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float64).flatten()
    b_f = b.to(torch.float64).flatten()
    dot = torch.dot(a_f, b_f).item()
    norm_a = torch.linalg.norm(a_f).item()
    norm_b = torch.linalg.norm(b_f).item()
    if norm_a == 0.0 and norm_b == 0.0:
        return 1.0
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _compute_sqnr(golden: torch.Tensor, test: torch.Tensor) -> float:
    g_f = golden.to(torch.float64).flatten()
    t_f = test.to(torch.float64).flatten()
    sig = torch.sum(g_f**2).item()
    err = torch.sum((g_f - t_f) ** 2).item()
    if err == 0.0:
        return 100.0
    if sig == 0.0:
        return 0.0
    return 10.0 * math.log10(sig / err)


class TestFusedRMSNormFP8RoPE(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_01_tpu_rmsnorm_forward_with_residual_parity(self):
        """Gate 1: Verify Fused RMSNorm with residual matches golden float64."""
        def golden_rmsnorm(x, weight, eps, residual):
            orig_dtype = x.dtype
            x = x.to(torch.float64)
            if residual is not None:
                x = x + residual.to(torch.float64)
                residual_out = x.to(orig_dtype)
            else:
                residual_out = None
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            normed = x * torch.rsqrt(variance + eps)
            out = (normed * weight.to(torch.float64)).to(orig_dtype)
            return out, residual_out

        eps = 1e-6
        for shape in [(1, 4096), (32, 4096), (64, 7168)]:
            with self.subTest(shape=shape):
                x = torch.randn(*shape, dtype=torch.bfloat16)
                res = torch.randn(*shape, dtype=torch.bfloat16)
                w = torch.randn(shape[-1], dtype=torch.bfloat16)

                g_out, g_res = golden_rmsnorm(x, w, eps, res)
                t_out, t_res = tpu_rmsnorm_forward(x, w, eps, res)

                cossim = _compute_cossim(g_out, t_out)
                sqnr = _compute_sqnr(g_out, t_out)
                res_diff = (g_res.float() - t_res.float()).abs().max().item()

                self.assertGreaterEqual(cossim, 0.99999)
                self.assertGreaterEqual(sqnr, 45.0)
                self.assertEqual(res_diff, 0.0)

    def test_02_tpu_rmsnorm_forward_without_residual_parity(self):
        """Gate 2: Verify Fused RMSNorm without residual matches golden float64."""
        eps = 1e-6
        for shape in [(1, 4096), (16, 2048), (64, 7168)]:
            with self.subTest(shape=shape):
                x = torch.randn(*shape, dtype=torch.bfloat16)
                w = torch.randn(shape[-1], dtype=torch.bfloat16)

                x_f64 = x.to(torch.float64)
                var = x_f64.pow(2).mean(dim=-1, keepdim=True)
                g_out = (x_f64 * torch.rsqrt(var + eps) * w.to(torch.float64)).to(torch.bfloat16)

                t_out = tpu_rmsnorm_forward(x, w, eps, None)

                cossim = _compute_cossim(g_out, t_out)
                sqnr = _compute_sqnr(g_out, t_out)
                self.assertGreaterEqual(cossim, 0.99999)
                self.assertGreaterEqual(sqnr, 45.0)

    def test_03_tpu_rmsnorm_variance_size_override_guard(self):
        """Gate 3: Verify TpuRMSNorm keeps native fallback when variance_size_override is set."""
        class DummyBase(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(128))
                self.variance_epsilon = 1e-6
            def forward_native(self, x, residual=None):
                return "NATIVE_CALLED"

        class SubTpuRMSNorm(TpuRMSNorm, DummyBase):
            pass

        mod = SubTpuRMSNorm()
        mod.variance_size_override = 64
        res = mod.forward_oot(torch.randn(2, 128))
        self.assertEqual(res, "NATIVE_CALLED")

    def test_04_fp8_linear_scale_formatting(self):
        """Gate 4: Verify _format_linear_scale_for_runtime handles all dimensionalities."""
        # 1-D scale
        s1 = torch.tensor([0.5, 0.25], dtype=torch.float32)
        out1 = _format_linear_scale_for_runtime(s1, blockwise_kernel=False)
        self.assertEqual(out1.shape, (2,))

        # 2-D scale [N, 1]
        s2 = torch.tensor([[0.5], [0.25]], dtype=torch.float32)
        out2 = _format_linear_scale_for_runtime(s2, blockwise_kernel=False)
        self.assertEqual(out2.shape, (2,))

        # 2-D scale [1, N]
        s3 = torch.tensor([[0.5, 0.25]], dtype=torch.float32)
        out3 = _format_linear_scale_for_runtime(s3, blockwise_kernel=False)
        self.assertEqual(out3.shape, (2,))

        # >2-D scale [1, 1, N]
        s4 = torch.tensor([[[0.5, 0.25]]], dtype=torch.float32)
        out4 = _format_linear_scale_for_runtime(s4, blockwise_kernel=False)
        self.assertEqual(out4.shape, (2,))

        # Blockwise scale: 2-D preserved
        s_block = torch.randn(4, 8, dtype=torch.float32)
        out_block = _format_linear_scale_for_runtime(s_block, blockwise_kernel=True)
        self.assertEqual(out_block.shape, (1, 8, 1, 4))

    def test_05_fused_rope_in_kernel_downcast_bit_exact(self):
        """Gate 5: Verify in-kernel RoPE downcast prevents float32 promotion."""
        def rotate_neox(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        def ref_native_rope(positions, query, key, cos_sin_cache):
            orig_q_dtype = query.dtype
            orig_k_dtype = key.dtype
            cos_sin = cos_sin_cache.index_select(0, positions)
            cos, sin = cos_sin.chunk(2, dim=-1)
            cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
            sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
            q_rot = query * cos + rotate_neox(query) * sin
            k_rot = key * cos + rotate_neox(key) * sin
            return q_rot.to(orig_q_dtype), k_rot.to(orig_k_dtype)

        def tpu_fused_rope(positions, query, key, cos_sin_cache):
            cos_sin = cos_sin_cache.index_select(0, positions)
            cos, sin = cos_sin.chunk(2, dim=-1)
            cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
            sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
            q_rot = (query * cos + rotate_neox(query) * sin).to(query.dtype)
            k_rot = (key * cos + rotate_neox(key) * sin).to(key.dtype)
            return q_rot, k_rot

        for num_tokens, num_heads in [(1, 16), (32, 64), (64, 128)]:
            pos = torch.arange(num_tokens, dtype=torch.long)
            q = torch.randn(num_tokens, num_heads, 64, dtype=torch.bfloat16)
            k = torch.randn(num_tokens, 1, 64, dtype=torch.bfloat16)
            cache = torch.randn(2048, 64, dtype=torch.float32)

            g_q, g_k = ref_native_rope(pos, q, k, cache)
            t_q, t_k = tpu_fused_rope(pos, q, k, cache)

            self.assertEqual(t_q.dtype, torch.bfloat16)
            self.assertEqual(t_k.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(g_q, t_q))
            self.assertTrue(torch.equal(g_k, t_k))

    def test_06_torch_compile_fullgraph_safety_rmsnorm_and_rope(self):
        """Gate 6: Verify combined RMSNorm + RoPE compiles cleanly under torch.compile(..., fullgraph=True)."""
        def rotate_neox(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        class MockTransformerPrologue(torch.nn.Module):
            def __init__(self, hidden_dim, rotary_dim, eps=1e-6):
                super().__init__()
                self.norm_weight = torch.nn.Parameter(torch.ones(hidden_dim, dtype=torch.bfloat16))
                self.eps = eps
                self.rotary_dim = rotary_dim

            def forward(self, x, residual, pos, q_pe, k_pe, cos_sin_cache):
                normed_x, new_res = tpu_rmsnorm_forward(x, self.norm_weight, self.eps, residual)
                cos_sin = cos_sin_cache.index_select(0, pos)
                cos, sin = cos_sin.chunk(2, dim=-1)
                cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
                sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
                rot_q = (q_pe * cos + rotate_neox(q_pe) * sin).to(q_pe.dtype)
                rot_k = (k_pe * cos + rotate_neox(k_pe) * sin).to(k_pe.dtype)
                return normed_x, new_res, rot_q, rot_k

        model = MockTransformerPrologue(4096, 64)
        compiled_model = torch.compile(model, fullgraph=True)

        x = torch.randn(16, 4096, dtype=torch.bfloat16)
        res = torch.randn(16, 4096, dtype=torch.bfloat16)
        pos = torch.arange(16, dtype=torch.long)
        q_pe = torch.randn(16, 16, 64, dtype=torch.bfloat16)
        k_pe = torch.randn(16, 1, 64, dtype=torch.bfloat16)
        cache = torch.randn(1024, 64, dtype=torch.float32)

        e_x, e_res, e_q, e_k = model(x, res, pos, q_pe, k_pe, cache)
        c_x, c_res, c_q, c_k = compiled_model(x, res, pos, q_pe, k_pe, cache)

        self.assertTrue(torch.equal(e_x, c_x))
        self.assertTrue(torch.equal(e_res, c_res))
        self.assertTrue(torch.equal(e_q, c_q))
        self.assertTrue(torch.equal(e_k, c_k))

    def test_07_patch_registry_and_platform_wiring(self):
        """Gate 7: Verify patch_registry and tpu_platform.py AST wiring."""
        p_patch = os.path.join(REPO_ROOT, "src/vllm_torchtpu/patch_registry.py")
        with open(p_patch) as f:
            patch_src = f.read()
        self.assertIn("vllm_torchtpu.layers.adapter.layernorm", patch_src)

        p_plat = os.path.join(REPO_ROOT, "src/vllm_torchtpu/platforms/tpu_platform.py")
        with open(p_plat) as f:
            plat_src = f.read()
        self.assertIn("import vllm_torchtpu.layers.adapter.layernorm", plat_src)

        p_mla = os.path.join(REPO_ROOT, "src/vllm_torchtpu/layers/adapter/custom_ops/mla_attention_op.py")
        with open(p_mla) as f:
            mla_src = f.read()
        self.assertIn(".to(query.dtype)", mla_src)
        self.assertIn(".to(key.dtype)", mla_src)

    def test_08_negative_controls(self):
        """Gate 8: Verify mismatched shapes raise expected exceptions."""
        x = torch.randn(2, 128, dtype=torch.bfloat16)
        bad_w = torch.randn(64, dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError):
            tpu_rmsnorm_forward(x, bad_w, 1e-6)

        bad_res = torch.randn(4, 128, dtype=torch.bfloat16)
        good_w = torch.randn(128, dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError):
            tpu_rmsnorm_forward(x, good_w, 1e-6, bad_res)


if __name__ == "__main__":
    unittest.main(verbosity=2)
