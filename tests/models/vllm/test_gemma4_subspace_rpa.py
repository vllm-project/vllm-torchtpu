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
"""Unit tests for Gemma 4 ``Subspace-RPA`` global attention decomposition."""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import get_rope

from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.models.vllm import gemma4_subrpa_patch as gsub


def _sqnr_db(ref: torch.Tensor, cand: torch.Tensor) -> float:
    err = (ref.float() - cand.float()).pow(2).sum().item()
    sig = ref.float().pow(2).sum().item()
    if err == 0.0:
        return float("inf")
    return 10.0 * math.log10(sig / err)


class _TestRMSNorm(nn.Module):
    def __init__(
        self, dim: int, has_weight: bool = True, dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.has_weight = has_weight
        if has_weight:
            self.weight = nn.Parameter(
                torch.ones(dim, dtype=dtype), requires_grad=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x.float().pow(2).mean(-1, keepdim=True)
        y = x.float() * torch.rsqrt(v + 1e-6)
        if self.has_weight:
            y = y * self.weight.float()
        return y.to(x.dtype)


@torch.library.custom_op("pallas::subrpa_mock_fp8_linear", mutates_args=())
def _subrpa_mock_fp8_linear(
    x: torch.Tensor, w: torch.Tensor, w_scale: torch.Tensor
) -> torch.Tensor:
    return ((x.float() @ w.float()) * w_scale).to(x.dtype)


@_subrpa_mock_fp8_linear.register_fake
def _(x: torch.Tensor, w: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    del w_scale
    return torch.empty(x.shape[0], w.shape[-1], dtype=x.dtype, device=x.device)


class _FlippedLinear(nn.Module):
    def __init__(self, in_f: int, out_f: int, dtype: torch.dtype):
        super().__init__()
        w_f32 = torch.randn(in_f, out_f) * 0.05
        if dtype == torch.float8_e4m3fn:
            scale = w_f32.abs().amax(dim=0).clamp(min=1e-6) / 448.0
            w_q = (w_f32 / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            self.weight = nn.Parameter(w_q, requires_grad=False)
            self.register_buffer("weight_scale", scale.to(torch.float32))
        else:
            self.weight = nn.Parameter(w_f32.to(dtype), requires_grad=False)
            self.weight_scale = None
        setattr(self, WEIGHT_FLIPPED_ATTR, True)
        self.output_size_per_partition = out_f
        self.bias = None
        self.reduce_results = True

    def forward(self, x: torch.Tensor):
        w = self.weight
        ws = self.weight_scale
        if ws is not None:
            return _subrpa_mock_fp8_linear(x, w, ws), None
        return (x.float() @ w.float()).to(x.dtype), None


@pytest.fixture
def python_attention(monkeypatch) -> None:
    """Send the kernel call to the module's Python attention stand-in."""

    def _call(attn_layer, query_3d, k_rot_3d, value_3d):
        return attn_layer(
            query_3d.flatten(-2, -1),
            k_rot_3d.flatten(-2, -1),
            value_3d.flatten(-2, -1),
        )

    monkeypatch.setattr(gsub, "call_subspace_unified_attention", _call)


def _quant_dequant_fp8(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    xq = (x.float() / scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)
    return xq.float() * scale


def _build_gemma4_global_attn_pair(dtype: torch.dtype, quantize_kv_fp8: bool = False):
    torch.manual_seed(42)
    old_default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        hidden = 256
        num_heads = 4
        num_kv_heads = 1
        head_dim = 512
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim

        lin_dtype = torch.float8_e4m3fn if dtype == torch.bfloat16 else torch.float32
        qkv = _FlippedLinear(hidden, q_size + 2 * kv_size, lin_dtype)
        with torch.no_grad():
            qkv.weight.data[:, q_size + kv_size :] = qkv.weight.data[
                :, q_size : q_size + kv_size
            ]
            if qkv.weight_scale is not None:
                qkv.weight_scale.data[q_size + kv_size :] = qkv.weight_scale.data[
                    q_size : q_size + kv_size
                ]

        o_proj = _FlippedLinear(q_size, hidden, lin_dtype)
        q_norm = _TestRMSNorm(head_dim, has_weight=True, dtype=dtype)
        k_norm = _TestRMSNorm(head_dim, has_weight=True, dtype=dtype)
        v_norm = _TestRMSNorm(head_dim, has_weight=False, dtype=dtype)
        with torch.no_grad():
            # Gemma 4 folds 1/sqrt(D) into q_norm/k_norm, and the positional
            # channels of k_norm carry much larger weights than the rest.
            q_norm.weight.copy_((torch.randn(head_dim) * 0.15 + 0.25).to(dtype))
            k_w = torch.randn(head_dim).abs() * 0.35 + 0.15
            k_w[:64] *= 8.0
            k_w[256:320] *= 8.0
            k_norm.weight.copy_(k_w.to(dtype))

        with set_current_vllm_config(
            VllmConfig(device_config=DeviceConfig(device="cpu"))
        ):
            rope = get_rope(
                head_dim,
                max_position=4096,
                rope_parameters={
                    "rope_type": "proportional",
                    "rope_theta": 1_000_000.0,
                    "partial_rotary_factor": 0.25,
                },
                is_neox_style=True,
            )
    finally:
        torch.set_default_dtype(old_default)

    def _ref_sdpa(
        q_2d: torch.Tensor, k_2d: torch.Tensor, v_2d: torch.Tensor
    ) -> torch.Tensor:
        t = q_2d.shape[0]
        qh = q_2d.view(t, num_heads, head_dim).float()
        kh = k_2d.view(t, num_kv_heads, head_dim).float()
        vh = v_2d.view(t, num_kv_heads, head_dim).float()
        if quantize_kv_fp8:
            kh = _quant_dequant_fp8(kh, 1.0)
            vh = _quant_dequant_fp8(vh, 1.0)
        kh = kh.repeat_interleave(num_heads, dim=1)
        vh = vh.repeat_interleave(num_heads, dim=1)
        scores = torch.einsum("thd,shd->hts", qh, kh)
        causal = torch.tril(torch.ones((t, t), dtype=torch.bool, device=q_2d.device))
        scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("hts,shd->thd", probs, vh).to(q_2d.dtype)
        return out.reshape(t, q_size)

    def _subspace_sdpa(
        q_2d: torch.Tensor, k_rot_2d: torch.Tensor, v_2d: torch.Tensor
    ) -> torch.Tensor:
        t = q_2d.shape[0]
        qh = q_2d.view(t, num_heads, head_dim).float()
        # HEAD_ALONG_SUBLANE Subspace-RPA keeps k_rot in BF16 and quantizes V.
        kr = k_rot_2d.view(t, num_kv_heads, 128).to(torch.bfloat16).float()
        vh = v_2d.view(t, num_kv_heads, head_dim).float()
        if quantize_kv_fp8:
            vh = _quant_dequant_fp8(vh, 1.0)
        kr = kr.repeat_interleave(num_heads, dim=1)
        vh = vh.repeat_interleave(num_heads, dim=1)
        scores = torch.einsum("thd,shd->hts", qh[..., :128], kr) + torch.einsum(
            "thd,shd->hts", qh[..., 128:], vh[..., 128:]
        )
        causal = torch.tril(torch.ones((t, t), dtype=torch.bool, device=q_2d.device))
        scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("hts,shd->thd", probs, vh).to(q_2d.dtype)
        return out.reshape(t, q_size)

    class Gemma4Attention(nn.Module):
        def __init__(self, r_emb):
            super().__init__()
            self.is_sliding = False
            self.is_kv_shared_layer = False
            self.config = SimpleNamespace(attention_k_eq_v=True)
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            self.q_size = q_size
            self.kv_size = kv_size
            self.qkv_proj = qkv
            self.o_proj = o_proj
            self.q_norm = q_norm
            self.k_norm = k_norm
            self.v_norm = v_norm
            self.rotary_emb = r_emb
            self.attn = SimpleNamespace(
                impl=SimpleNamespace(kv_cache_quantized_dtype=torch.float8_e4m3fn)
            )

    ref_mod = Gemma4Attention(rope)
    # Give sub_mod its own deep-copied rope so ref_mod's 512-dim cache stays untouched
    sub_mod = copy.deepcopy(ref_mod)
    ref_mod.attn = _ref_sdpa  # type: ignore[assignment]
    assert gsub.conjugate_gemma4_subspace_attn_module(sub_mod) is True
    sub_mod.attn = _subspace_sdpa  # type: ignore[assignment]
    return ref_mod, sub_mod


def _run_stock_attn(mod, pos: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    qkv, _ = mod.qkv_proj(x)
    q, k, v = qkv.split([mod.q_size, mod.kv_size, mod.kv_size], dim=-1)
    q = mod.q_norm(q.unflatten(-1, (mod.num_heads, 512))).flatten(-2, -1)
    k = mod.k_norm(k.unflatten(-1, (mod.num_kv_heads, 512))).flatten(-2, -1)
    q, k = mod.rotary_emb(pos, q, k)
    v = mod.v_norm(v.unflatten(-1, (mod.num_kv_heads, 512))).flatten(-2, -1)
    out, _ = mod.o_proj(mod.attn(q, k, v))
    return out


def test_subspace_rpa_exact_algebraic_equivalence_fp32_and_bf16(python_attention):
    """The rewritten layer matches the stock layer in FP32 and BF16."""
    ref32, sub32 = _build_gemma4_global_attn_pair(torch.float32, quantize_kv_fp8=False)

    def _exact_f32_sdpa(q_2d, kr_2d, v_2d):
        t = q_2d.shape[0]
        qh = q_2d.view(t, 4, 512).float()
        kr = kr_2d.view(t, 1, 128).float().repeat_interleave(4, dim=1)
        vh = v_2d.view(t, 1, 512).float().repeat_interleave(4, dim=1)
        scores = torch.einsum("thd,shd->hts", qh[..., :128], kr) + torch.einsum(
            "thd,shd->hts", qh[..., 128:], vh[..., 128:]
        )
        causal = torch.tril(torch.ones((t, t), dtype=torch.bool, device=q_2d.device))
        scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        return torch.einsum("hts,shd->thd", probs, vh).reshape(t, 2048)

    # Keep k_rot in FP32 here so the comparison checks the algebra alone.
    sub32.attn = _exact_f32_sdpa  # type: ignore[assignment]

    assert sub32.qkv_proj.weight.shape[1] == sub32.q_size + sub32.kv_size
    assert sub32.k_norm.weight.shape == (128,)
    assert sub32.rotary_emb.cos_sin_cache.shape[1] == 128

    x32 = torch.randn(37, 256, dtype=torch.float32)
    pos = torch.arange(37, dtype=torch.long)
    out_ref32 = _run_stock_attn(ref32, pos, x32)
    out_sub32 = gsub.subspace_attention_partial(sub32, pos, x32)
    sqnr32 = _sqnr_db(out_ref32, out_sub32)
    assert sqnr32 > 110.0, f"Expected >110 dB FP32 exactness, got {sqnr32:.2f} dB"

    ref16, sub16 = _build_gemma4_global_attn_pair(torch.bfloat16, quantize_kv_fp8=False)
    x16 = torch.randn(37, 256, dtype=torch.bfloat16)
    out_ref16 = _run_stock_attn(ref16, pos, x16)
    out_sub16 = gsub.subspace_attention_partial(sub16, pos, x16)
    sqnr16 = _sqnr_db(out_ref16, out_sub16)
    assert sqnr16 > 40.0, f"Expected >40 dB BF16 equivalence, got {sqnr16:.2f} dB"


def test_subspace_rpa_mixed_precision_kv_quantization_sqnr_advantage(
    python_attention,
):
    """Verify Subspace-RPA (BF16-K_rot + FP8-V) beats stock FP8 K+V SQNR."""
    ref_exact, _ = _build_gemma4_global_attn_pair(torch.bfloat16, quantize_kv_fp8=False)
    ref_fp8_kv, sub_fp8_kv = _build_gemma4_global_attn_pair(
        torch.bfloat16, quantize_kv_fp8=True
    )
    x = torch.randn(64, 256, dtype=torch.bfloat16)
    pos = torch.arange(64, dtype=torch.long)

    y_exact = _run_stock_attn(ref_exact, pos, x)
    y_stock_fp8 = _run_stock_attn(ref_fp8_kv, pos, x)
    y_sub_fp8 = gsub.subspace_attention_partial(sub_fp8_kv, pos, x)

    sqnr_stock = _sqnr_db(y_exact, y_stock_fp8)
    sqnr_sub = _sqnr_db(y_exact, y_sub_fp8)
    # K_rot stays BF16 and the unrotated k_norm weights move into q_norm, so
    # the k_norm scale is never FP8-quantized.
    assert sqnr_sub >= sqnr_stock + 1.5, (
        f"Expected Subspace-RPA SQNR ({sqnr_sub:.2f} dB) to beat stock FP8 KV "
        f"({sqnr_stock:.2f} dB) by >= 1.5 dB"
    )


def test_subspace_rpa_kv_cache_shape_unification_and_gates(monkeypatch) -> None:
    """Verify LBNHC page unification with 256-dim SWA and patch eligibility gates."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackend

    monkeypatch.setattr(envs, "TPU_SUBSPACE_RPA", False)
    stock_global = PallasAttentionBackend.get_kv_cache_shape(
        10, 256, 1, 512, "fp8_e4m3"
    )
    swa_shape = PallasAttentionBackend.get_kv_cache_shape(10, 256, 2, 256, "fp8_e4m3")
    assert stock_global == (10, 256, 1, 4, 512)
    assert swa_shape == (10, 256, 1, 4, 256)

    monkeypatch.setattr(envs, "TPU_SUBSPACE_RPA", True)
    sub_global = PallasAttentionBackend.get_kv_cache_shape(10, 256, 1, 512, "fp8_e4m3")
    assert sub_global == swa_shape == (10, 256, 1, 4, 256)
    assert PallasAttentionBackend.get_kv_cache_page_size_bytes(
        256, 1, 512, "fp8_e4m3"
    ) == PallasAttentionBackend.get_kv_cache_page_size_bytes(256, 2, 256, "fp8_e4m3")

    monkeypatch.setattr(envs, "TPU_SUBSPACE_RPA", False)
    assert gsub.maybe_patch_gemma4_subspace_rpa(None) is None
    monkeypatch.setattr(envs, "TPU_SUBSPACE_RPA", True)
    assert gsub.maybe_patch_gemma4_subspace_rpa(None) is False
    non_gemma = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="llama", attention_k_eq_v=False)
    )
    assert gsub.maybe_patch_gemma4_subspace_rpa(non_gemma) is False


def test_call_subspace_unified_attention_layer_name_fullgraph(monkeypatch) -> None:
    """Ensure the real vLLM ``layer_name`` branch compiles under ``fullgraph=True``."""

    @torch.library.custom_op(
        "pallas::subrpa_mock_unified_attn_out", mutates_args=("output",)
    )
    def _mock_unified_out(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        del k
        output.copy_(q + v.repeat_interleave(q.shape[1] // v.shape[1], dim=1))

    @_mock_unified_out.register_fake
    def _(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        del q, k, v, output

    def _fake_direct(q, k, v, out, layer_name, kv_cache_dummy_dep=None):
        del layer_name, kv_cache_dummy_dep
        _mock_unified_out(q, k, v, out)

    monkeypatch.setattr(
        gsub.vllm_attn_layer, "unified_attention_with_output", _fake_direct
    )
    attn_layer = SimpleNamespace(
        layer_name="model.layers.5.self_attn.attn",
        num_heads=4,
        head_size_v=512,
        use_direct_call=True,
    )

    def _fn(q: torch.Tensor, kr: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return gsub.call_subspace_unified_attention(attn_layer, q, kr, v)

    torch._dynamo.reset()
    compiled = torch.compile(_fn, backend="eager", fullgraph=True)
    q = torch.randn(8, 4, 512, dtype=torch.bfloat16)
    kr = torch.randn(8, 1, 128, dtype=torch.bfloat16)
    v = torch.randn(8, 1, 512, dtype=torch.bfloat16)
    out = compiled(q, kr, v)
    assert out.shape == (8, 2048)
