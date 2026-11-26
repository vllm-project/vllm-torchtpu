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
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.attention import AttentionConfig
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding, get_rope
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackend,
    PallasAttentionBackendImpl,
    PallasBatchedRPAAttentionBackend,
    PallasBatchedRPAAttentionBackendImpl,
)
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    _format_linear_scale_for_runtime,
)
from vllm_torchtpu.layers.core.quantization import quantize_tensor
from vllm_torchtpu.models.vllm import gemma4_subrpa_patch as gsub
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform


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
    if w_scale.ndim == 4 and w_scale.shape[1] > 1:
        n_blocks = w_scale.shape[1]
        block_sz = w.shape[0] // n_blocks
        scale_exp = w_scale.squeeze(0).squeeze(1).repeat_interleave(block_sz, dim=0)
        w_deq = w.float() * scale_exp
        return (x.float() @ w_deq).to(x.dtype)
    return ((x.float() @ w.float()) * w_scale).to(x.dtype)


@_subrpa_mock_fp8_linear.register_fake
def _(x: torch.Tensor, w: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    del w_scale
    return torch.empty(x.shape[0], w.shape[-1], dtype=x.dtype, device=x.device)


@torch.library.custom_op(
    "pallas::subrpa_mock_unified_attn_out", mutates_args=("output",)
)
def _subrpa_mock_unified_attn_out(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
) -> None:
    del k
    output.copy_(q + v.repeat_interleave(q.shape[1] // v.shape[1], dim=1))


@_subrpa_mock_unified_attn_out.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
) -> None:
    del q, k, v, output


class _FlippedLinear(nn.Module):
    def __init__(
        self,
        in_f: int,
        out_f: int,
        dtype: torch.dtype,
        block_size: int | None = None,
    ):
        super().__init__()
        w_f32 = torch.randn(out_f, in_f) * 0.05
        if dtype == torch.float8_e4m3fn:
            w_q, scale = quantize_tensor(
                w_f32,
                quant_dtype=torch.float8_e4m3fn,
                axis=-1,
                block_size=block_size,
            )
            scale_rt = _format_linear_scale_for_runtime(
                scale, blockwise_kernel=block_size is not None
            )
            self.weight = nn.Parameter(
                w_q.transpose(0, 1).contiguous(), requires_grad=False
            )
            self.register_buffer("weight_scale", scale_rt.contiguous())
        else:
            self.weight = nn.Parameter(
                w_f32.transpose(0, 1).to(dtype).contiguous(), requires_grad=False
            )
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


def _make_batched_rpa_impl(
    kv_dtype: torch.dtype | None = torch.float8_e4m3fn,
) -> PallasBatchedRPAAttentionBackendImpl:
    impl = object.__new__(PallasBatchedRPAAttentionBackendImpl)
    impl.kv_cache_quantized_dtype = kv_dtype
    return impl


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


def _build_gemma4_global_attn_pair(
    dtype: torch.dtype,
    quantize_kv_fp8: bool = False,
    oproj_block_size: int | None = None,
    use_tpu_compacted_rope: bool = True,
):
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

        o_proj = _FlippedLinear(q_size, hidden, lin_dtype, block_size=oproj_block_size)
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
        if not use_tpu_compacted_rope:
            # Exercise the uncompacted 512-column upstream cos_sin_cache path
            # on a private clone without mutating get_rope's memoized entry.
            rope = copy.deepcopy(rope)
            full_cache = rope._compute_cos_sin_cache().to(dtype).contiguous()
            rope.register_buffer("cos_sin_cache", full_cache, persistent=False)
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

    class _MockGemma4Attention(nn.Module):
        def __init__(self, r_emb):
            super().__init__()
            self.is_sliding = False
            self.is_kv_shared_layer = False
            self.config = SimpleNamespace(
                attention_k_eq_v=True,
                num_kv_shared_layers=0,
                attention_bias=False,
            )
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
            self.attn = SimpleNamespace(impl=_make_batched_rpa_impl())

    ref_mod = _MockGemma4Attention(rope)
    sub_mod = copy.deepcopy(ref_mod)
    # Intentionally share the exact same rotary_emb object to verify the
    # rewrite never mutates get_rope's memoized RotaryEmbedding instance.
    sub_mod.rotary_emb = rope
    orig_rope_cache = rope.cos_sin_cache.clone()
    orig_rotary_dim = rope.rotary_dim

    ref_mod.attn = _ref_sdpa  # type: ignore[assignment]
    assert gsub.rewrite_gemma4_attn_for_subspace_rpa(sub_mod) is True
    assert rope.rotary_dim == orig_rotary_dim
    assert torch.equal(rope.cos_sin_cache, orig_rope_cache)
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


def test_subspace_rpa_exact_algebraic_equivalence_fp32(python_attention) -> None:
    """The rewritten layer matches the stock layer to FP32 precision (>110 dB)."""
    ref32, sub32 = _build_gemma4_global_attn_pair(
        torch.float32,
        quantize_kv_fp8=False,
        use_tpu_compacted_rope=False,
    )

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

    sub32.attn = _exact_f32_sdpa  # type: ignore[assignment]

    assert sub32.qkv_proj.weight.shape[1] == sub32.q_size + sub32.kv_size
    assert sub32.qkv_proj.output_size_per_partition == sub32.q_size + sub32.kv_size
    assert sub32.k_norm.weight.shape == (128,)
    assert sub32._subrpa_cos_sin_cache.shape[1] == 128
    assert sub32.rotary_emb.cos_sin_cache.shape[1] == 512

    x32 = torch.randn(37, 256, dtype=torch.float32)
    pos = torch.arange(37, dtype=torch.long)
    out_ref32 = _run_stock_attn(ref32, pos, x32)
    out_sub32 = gsub.subspace_attention_partial(sub32, pos, x32)
    sqnr32 = _sqnr_db(out_ref32, out_sub32)
    assert sqnr32 > 110.0, f"Expected >110 dB FP32 exactness, got {sqnr32:.2f} dB"


@pytest.mark.parametrize("use_tpu_compacted_rope", [False, True])
def test_subspace_rpa_algebraic_equivalence_bf16(
    python_attention, use_tpu_compacted_rope: bool
) -> None:
    """Rewritten layer matches stock layer in BF16 with both RoPE cache forms."""
    ref16, sub16 = _build_gemma4_global_attn_pair(
        torch.bfloat16,
        quantize_kv_fp8=False,
        use_tpu_compacted_rope=use_tpu_compacted_rope,
    )
    expected_rope_cols = 128 if use_tpu_compacted_rope else 512
    assert sub16.rotary_emb.cos_sin_cache.shape[1] == expected_rope_cols
    assert sub16._subrpa_cos_sin_cache.shape[1] == 128
    pos = torch.arange(37, dtype=torch.long)
    x16 = torch.randn(37, 256, dtype=torch.bfloat16)
    out_ref16 = _run_stock_attn(ref16, pos, x16)
    out_sub16 = gsub.subspace_attention_partial(sub16, pos, x16)
    sqnr16 = _sqnr_db(out_ref16, out_sub16)
    assert sqnr16 > 40.0, f"Expected >40 dB BF16 equivalence, got {sqnr16:.2f} dB"


def test_subspace_rpa_blockwise_fp8_oproj_rewrite(python_attention) -> None:
    """Blockwise 4D FP8 o_proj.weight_scale is dequantized and requantized."""
    ref16, sub16 = _build_gemma4_global_attn_pair(
        torch.bfloat16, quantize_kv_fp8=False, oproj_block_size=128
    )
    assert sub16.o_proj.weight_scale.shape == (1, 16, 1, 256)
    pos = torch.arange(37, dtype=torch.long)
    x16 = torch.randn(37, 256, dtype=torch.bfloat16)
    out_ref = _run_stock_attn(ref16, pos, x16)
    out_sub = gsub.subspace_attention_partial(sub16, pos, x16)
    sqnr = _sqnr_db(out_ref, out_sub)
    assert sqnr > 34.0, (
        f"Expected >34 dB blockwise FP8 o_proj parity, got {sqnr:.2f} dB"
    )

    bad_oproj = _FlippedLinear(2048, 256, torch.float8_e4m3fn)
    bad_oproj.register_buffer(
        "weight_scale", torch.ones((16, 256), dtype=torch.float32)
    )
    with pytest.raises(ValueError, match="o_proj.weight_scale"):
        gsub._permute_oproj_input_channels(bad_oproj, torch.arange(2048), [])


def test_subspace_rpa_mixed_precision_kv_quantization_sqnr_advantage(
    python_attention,
):
    """Verify the analytical BF16-K_rot + FP8-V decomposition beats stock FP8 K+V
    in Python attention (real TPU kernel storage parity is tested in
    ``test_subspace_rpa.py``)."""
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
    assert sqnr_sub >= sqnr_stock + 1.5, (
        f"Expected Subspace-RPA SQNR ({sqnr_sub:.2f} dB) to beat stock FP8 KV "
        f"({sqnr_stock:.2f} dB) by >= 1.5 dB"
    )


def _make_vllm_cfg(model_cfg, layout_env: str | None = None) -> VllmConfig:
    cfg = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        attention_config=AttentionConfig(backend=AttentionBackendEnum.CUSTOM),
        cache_config=CacheConfig(cache_dtype="fp8"),
    )
    cfg.model_config = model_cfg
    if layout_env == "HND":
        cfg.cache_config.kv_cache_layout = "HND"
    resolve_kv_cache_layout(cfg, [["LBNHC", "LBHNC"]])
    return cfg


def _make_gemma4_model_cfg(
    *,
    attention_k_eq_v: bool = True,
    num_kv_shared_layers: int = 0,
    attention_bias: bool = False,
    partial_rotary_factor: float = 0.25,
    sliding_head_dim: int = 256,
    global_head_dim: int = 512,
) -> SimpleNamespace:
    text_cfg = Gemma4TextConfig(
        architectures=["Gemma4ForConditionalGeneration"],
        num_hidden_layers=6,
        head_dim=sliding_head_dim,
        global_head_dim=global_head_dim,
        attention_k_eq_v=attention_k_eq_v,
        num_kv_shared_layers=num_kv_shared_layers,
        attention_bias=attention_bias,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": partial_rotary_factor,
                "rope_theta": 1_000_000.0,
            },
        },
    )
    return SimpleNamespace(
        architectures=["Gemma4ForConditionalGeneration"],
        hf_config=text_cfg,
        hf_text_config=text_cfg,
    )


def test_subspace_rpa_kv_cache_shape_unification_and_gates(monkeypatch) -> None:
    """Verify CUSTOM vs FLASH_ATTN backend shapes, model gates, and platform checks."""
    assert "TPU_SUBSPACE_RPA" in TpuPlatform.additional_env_vars
    gemma4_cfg = _make_gemma4_model_cfg()
    kv_shared_gemma4_cfg = _make_gemma4_model_cfg(num_kv_shared_layers=2)
    biased_gemma4_cfg = _make_gemma4_model_cfg(attention_bias=True)
    wrong_rope_gemma4_cfg = _make_gemma4_model_cfg(partial_rotary_factor=0.5)
    sliding_512_gemma4_cfg = _make_gemma4_model_cfg(sliding_head_dim=512)
    llama_text = SimpleNamespace(model_type="llama")
    non_gemma_cfg = SimpleNamespace(
        architectures=["LlamaForCausalLM"],
        hf_config=llama_text,
        hf_text_config=llama_text,
    )

    with set_current_vllm_config(_make_vllm_cfg(gemma4_cfg)):
        monkeypatch.setattr(envs, "TPU_SUBSPACE_RPA", False)
        stock_global = PallasBatchedRPAAttentionBackend.get_kv_cache_shape(
            10, 256, 1, 512, "fp8_e4m3"
        )
        swa_shape = PallasBatchedRPAAttentionBackend.get_kv_cache_shape(
            10, 256, 2, 256, "fp8_e4m3"
        )
        assert stock_global == (10, 256, 1, 4, 512)
        assert swa_shape == (10, 256, 1, 4, 256)

        monkeypatch.setattr(envs, "TPU_SUBSPACE_RPA", True)
        sub_global = PallasBatchedRPAAttentionBackend.get_kv_cache_shape(
            10, 256, 1, 512, "fp8_e4m3"
        )
        assert sub_global == swa_shape == (10, 256, 1, 4, 256)
        assert PallasBatchedRPAAttentionBackend.get_kv_cache_page_size_bytes(
            256, 1, 512, "fp8_e4m3"
        ) == PallasBatchedRPAAttentionBackend.get_kv_cache_page_size_bytes(
            256, 2, 256, "fp8_e4m3"
        )
        # FLASH_ATTN (RPA v3) never allocates Subspace-RPA pages.
        assert PallasAttentionBackend.get_kv_cache_shape(
            10, 256, 1, 512, "fp8_e4m3"
        ) == (10, 256, 1, 4, 512)

    with set_current_vllm_config(_make_vllm_cfg(gemma4_cfg, layout_env="HND")):
        monkeypatch.setattr(envs, "TPU_SUBSPACE_RPA", True)
        assert PallasBatchedRPAAttentionBackend.get_kv_cache_shape(
            10, 128, 1, 512, "fp8_e4m3"
        ) == (10, 1, 160, 4, 128)

    # Non-float 8-bit KV dtypes (int8/uint8 packed FP4) never take Subspace-RPA.
    with set_current_vllm_config(_make_vllm_cfg(gemma4_cfg)):
        for non_fp8 in ("int8", "uint8"):
            assert PallasBatchedRPAAttentionBackend.get_kv_cache_shape(
                10, 256, 1, 512, non_fp8
            ) == (10, 256, 1, 4, 512)

    # Ineligible model configs keep stock BatchedRPA shapes and skip patching.
    for ineligible_cfg in (
        non_gemma_cfg,
        kv_shared_gemma4_cfg,
        biased_gemma4_cfg,
        wrong_rope_gemma4_cfg,
        sliding_512_gemma4_cfg,
    ):
        with set_current_vllm_config(_make_vllm_cfg(ineligible_cfg)):
            assert PallasBatchedRPAAttentionBackend.get_kv_cache_shape(
                10, 256, 1, 512, "fp8_e4m3"
            ) == (10, 256, 1, 4, 512)
        assert gsub.maybe_patch_gemma4_subspace_rpa(ineligible_cfg) is False

    monkeypatch.setattr(envs, "TPU_SUBSPACE_RPA", False)
    assert gsub.maybe_patch_gemma4_subspace_rpa(None) is None
    monkeypatch.setattr(envs, "TPU_SUBSPACE_RPA", True)
    assert gsub.maybe_patch_gemma4_subspace_rpa(None) is False

    # Layer-level eligibility rejects FLASH_ATTN impl, BF16 KV, and non-Gemma4 RoPE.
    ref_mod, _ = _build_gemma4_global_attn_pair(torch.bfloat16)
    flash_impl = object.__new__(PallasAttentionBackendImpl)
    flash_impl.kv_cache_quantized_dtype = torch.float8_e4m3fn
    ref_mod.attn = SimpleNamespace(impl=flash_impl)
    assert gsub.is_subspace_eligible_gemma4_attn(ref_mod) is False

    for bad_kv_dtype in (None, torch.int8, torch.uint8):
        ref_mod.attn = SimpleNamespace(
            impl=_make_batched_rpa_impl(kv_dtype=bad_kv_dtype)
        )
        assert gsub.is_subspace_eligible_gemma4_attn(ref_mod) is False

    # Target layer of a KV-shared Gemma 4 checkpoint is also rejected.
    ref_mod.attn = SimpleNamespace(impl=_make_batched_rpa_impl())
    ref_mod.config.num_kv_shared_layers = 2
    assert gsub.is_subspace_eligible_gemma4_attn(ref_mod) is False
    ref_mod.config.num_kv_shared_layers = 0

    # Empty cos_sin_cache (as installed by tpu_rope_in_graph) fails fast.
    empty_rope_mod = copy.deepcopy(ref_mod)
    empty_rope_mod.attn = SimpleNamespace(impl=_make_batched_rpa_impl())
    empty_rope_mod.rotary_emb.register_buffer(
        "cos_sin_cache", torch.empty((0, 128), dtype=torch.bfloat16), persistent=False
    )
    with pytest.raises(ValueError, match="cos_sin_cache"):
        gsub.rewrite_gemma4_attn_for_subspace_rpa(empty_rope_mod)

    # FLASH_ATTN backend forward rejects 128-dim Subspace-RPA keys directly.
    flash_impl.head_size = 512
    flash_impl.num_heads = 4
    flash_impl.num_kv_heads = 1
    with pytest.raises(
        NotImplementedError, match="PallasBatchedRPAAttentionBackendImpl"
    ):
        flash_impl.forward(
            layer=SimpleNamespace(),
            query=torch.zeros((2, 4 * 512), dtype=torch.bfloat16),
            key=torch.zeros((2, 128), dtype=torch.bfloat16),
            value=torch.zeros((2, 512), dtype=torch.bfloat16),
            kv_cache=torch.zeros((1, 256, 1, 4, 256), dtype=torch.float8_e4m3fn),
            attn_metadata=SimpleNamespace(),
        )

    ref_mod.attn = SimpleNamespace(impl=_make_batched_rpa_impl())
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        ref_mod.rotary_emb = RotaryEmbedding(
            512, 512, 4096, 10000.0, True, torch.bfloat16
        )
    assert gsub.is_subspace_eligible_gemma4_attn(ref_mod) is False

    # Platform fail-fast checks when model_config is present.
    monkeypatch.setattr(envs, "TPU_SUBSPACE_RPA", False)
    bad_backend_cfg = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        attention_config=AttentionConfig(backend=AttentionBackendEnum.FLASH_ATTN),
        cache_config=CacheConfig(cache_dtype="fp8"),
    )
    bad_backend_cfg.model_config = gemma4_cfg
    bad_dtype_cfg = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        attention_config=AttentionConfig(backend=AttentionBackendEnum.CUSTOM),
        cache_config=CacheConfig(cache_dtype="auto"),
    )
    bad_dtype_cfg.model_config = gemma4_cfg
    bad_model_cfg = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        attention_config=AttentionConfig(backend=AttentionBackendEnum.CUSTOM),
        cache_config=CacheConfig(cache_dtype="fp8"),
    )
    bad_model_cfg.model_config = non_gemma_cfg

    monkeypatch.setattr(envs, "TPU_SUBSPACE_RPA", True)
    with pytest.raises(NotImplementedError, match="--attention-backend CUSTOM"):
        TpuPlatform.check_and_update_config(bad_backend_cfg)
    with pytest.raises(NotImplementedError, match="--kv-cache-dtype fp8"):
        TpuPlatform.check_and_update_config(bad_dtype_cfg)
    with pytest.raises(NotImplementedError, match="Gemma 4"):
        TpuPlatform.check_and_update_config(bad_model_cfg)


@pytest.mark.parametrize(
    "case, match",
    [
        ("pcp", "context parallelism"),
        ("dcp", "context parallelism"),
        ("block_major", "VLLM_TPU_BLOCK_MAJOR_KV"),
        ("rope_in_graph", "tpu_rope_in_graph"),
        ("spec_decode", "speculative decoding"),
    ],
)
def test_platform_gate_rejects_unsupported_modes(monkeypatch, case: str, match: str):
    """TpuPlatform.check_and_update_config rejects unsupported Subspace-RPA modes."""
    monkeypatch.setattr(envs, "TPU_SUBSPACE_RPA", False)
    monkeypatch.setattr(envs, "VLLM_TPU_BLOCK_MAJOR_KV", False)
    cfg = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        attention_config=AttentionConfig(backend=AttentionBackendEnum.CUSTOM),
        cache_config=CacheConfig(cache_dtype="fp8"),
        additional_config={"tpu_rope_in_graph": True}
        if case == "rope_in_graph"
        else {},
    )
    cfg.model_config = _make_gemma4_model_cfg()
    if case == "pcp":
        cfg.parallel_config.prefill_context_parallel_size = 2
    elif case == "dcp":
        cfg.parallel_config.decode_context_parallel_size = 2
    elif case == "spec_decode":
        cfg.speculative_config = SimpleNamespace(method="mtp")
    if case == "block_major":
        monkeypatch.setattr(envs, "VLLM_TPU_BLOCK_MAJOR_KV", True)

    monkeypatch.setattr(envs, "TPU_SUBSPACE_RPA", True)
    with pytest.raises(NotImplementedError, match=match):
        TpuPlatform.check_and_update_config(cfg)


def test_maybe_patch_gemma4_subspace_rpa_hook_and_forward_dispatch(
    monkeypatch, python_attention
) -> None:
    """Eligible maybe_patch_gemma4_subspace_rpa hooks loader and dispatches forward."""
    from vllm.model_executor.model_loader import base_loader
    from vllm.model_executor.models.gemma4 import Gemma4Attention

    monkeypatch.setattr(envs, "TPU_SUBSPACE_RPA", True)
    monkeypatch.setattr(
        base_loader,
        "process_weights_after_loading",
        lambda model, model_config, target_device: "loaded",
    )
    orig_calls: list[str] = []

    def _orig_forward(self, positions, hidden_states, **kwargs):
        del positions, kwargs
        orig_calls.append(self.tag)
        return hidden_states

    monkeypatch.setattr(Gemma4Attention, "forward", _orig_forward)
    monkeypatch.delattr(Gemma4Attention, gsub._PATCHED_ATTN_CLS_ATTR, raising=False)

    gemma4_cfg = _make_gemma4_model_cfg()
    assert gsub.maybe_patch_gemma4_subspace_rpa(gemma4_cfg) is True

    # Build a fake Gemma4Attention subclass with one eligible global layer and
    # one ineligible sliding layer so both dispatch branches are exercised.
    ref_mod, _ = _build_gemma4_global_attn_pair(torch.bfloat16)

    class _FakeGemma4Attn(Gemma4Attention):
        def __init__(self, tag: str, is_sliding: bool):
            nn.Module.__init__(self)
            self.tag = tag
            self.is_sliding = is_sliding
            self.is_kv_shared_layer = False
            self.config = gemma4_cfg.hf_text_config
            self.num_heads = ref_mod.num_heads
            self.num_kv_heads = ref_mod.num_kv_heads
            self.head_dim = ref_mod.head_dim
            self.q_size = ref_mod.q_size
            self.kv_size = ref_mod.kv_size
            self.qkv_proj = copy.deepcopy(ref_mod.qkv_proj)
            self.o_proj = copy.deepcopy(ref_mod.o_proj)
            self.q_norm = copy.deepcopy(ref_mod.q_norm)
            self.k_norm = copy.deepcopy(ref_mod.k_norm)
            self.v_norm = copy.deepcopy(ref_mod.v_norm)
            self.rotary_emb = ref_mod.rotary_emb
            self.attn = SimpleNamespace(impl=_make_batched_rpa_impl())

    eligible = _FakeGemma4Attn("global", is_sliding=False)
    sliding = _FakeGemma4Attn("sliding", is_sliding=True)
    model = nn.ModuleList([eligible, sliding])

    res = base_loader.process_weights_after_loading(model, gemma4_cfg, "cpu")
    assert res == "loaded"
    assert getattr(eligible, gsub._SUBRPA_ACTIVE_ATTR, False) is True
    assert getattr(sliding, gsub._SUBRPA_ACTIVE_ATTR, False) is False

    subspace_calls: list[str] = []

    def _spy_subspace(attn, positions, x):
        del positions
        subspace_calls.append(attn.tag)
        return x + 1.0

    monkeypatch.setattr(gsub, "subspace_attention_partial", _spy_subspace)

    x = torch.zeros((4, 256), dtype=torch.bfloat16)
    pos = torch.arange(4, dtype=torch.long)
    out_global = eligible(pos, x)
    out_sliding = sliding(pos, x)

    assert subspace_calls == ["global"]
    assert orig_calls == ["sliding"]
    torch.testing.assert_close(out_global, x + 1.0)
    torch.testing.assert_close(out_sliding, x)


@pytest.mark.parametrize("use_direct_call", [True, False])
def test_call_subspace_unified_attention_fullgraph(
    monkeypatch, use_direct_call: bool
) -> None:
    """Compile call_subspace_unified_attention under fullgraph=True and check values."""
    # Verify the pinned torch.ops.vllm.unified_attention_with_output schema
    # accepts kv_cache_dummy_dep=None as passed in the non-direct branch.
    op_schema = str(torch.ops.vllm.unified_attention_with_output.default._schema)
    assert "kv_cache_dummy_dep" in op_schema

    def _fake_direct(q, k, v, out, layer_name, kv_cache_dummy_dep=None):
        del layer_name, kv_cache_dummy_dep
        _subrpa_mock_unified_attn_out(q, k, v, out)

    if use_direct_call:
        monkeypatch.setattr(
            gsub.vllm_attn_layer, "unified_attention_with_output", _fake_direct
        )
    else:
        monkeypatch.setattr(
            torch.ops.vllm, "unified_attention_with_output", _fake_direct
        )

    attn_layer = SimpleNamespace(
        layer_name="model.layers.5.self_attn.attn",
        num_heads=4,
        head_size_v=512,
        use_direct_call=use_direct_call,
    )

    def _fn(q: torch.Tensor, kr: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return gsub.call_subspace_unified_attention(attn_layer, q, kr, v)

    torch._dynamo.reset()
    compiled = torch.compile(_fn, backend="eager", fullgraph=True)
    q = torch.randn(8, 4, 512, dtype=torch.bfloat16)
    kr = torch.randn(8, 1, 128, dtype=torch.bfloat16)
    v = torch.randn(8, 1, 512, dtype=torch.bfloat16)
    out = compiled(q, kr, v)
    expected = (q + v.repeat_interleave(4, dim=1)).reshape(8, 2048)
    torch.testing.assert_close(out, expected)


@pytest.mark.tpu_test
@pytest.mark.parametrize("layout_env", [None, "HND"], ids=["LBNHC", "LBHNC"])
def test_batched_rpa_impl_forward_subspace_rpa_both_layouts(
    monkeypatch, layout_env: str | None
) -> None:
    """Exercise PallasBatchedRPAAttentionBackendImpl.forward with 128-dim K on TPU.

    Passes unquantized BF16 K_rot and V with non-unit k_scale/v_scale through
    the real adapter forward path and real BatchedRPA Pallas kernel across
    prefill and decode, verifying V-only quantization for HEAD_ALONG_SUBLANE
    and K_rot+V quantization for SEQ_ALONG_LANE against an independent reference.
    """
    import jax
    import numpy as np
    from jax.sharding import Mesh

    from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        set_vllm_model_wrapper_context,
    )

    monkeypatch.setattr(envs, "TPU_SUBSPACE_RPA", True)
    monkeypatch.setattr(PallasAttentionBackendImpl, "_kernel_registry", {})
    gemma4_cfg = _make_gemma4_model_cfg()
    vllm_cfg = _make_vllm_cfg(gemma4_cfg, layout_env=layout_env)

    num_heads = 4
    num_kv_heads = 1
    head_dim = 512
    rot_dim = 128
    page_size = 128
    num_blocks = 4
    scale = head_dim**-0.5
    k_scale = 0.75
    v_scale = 1.25
    tpu = torch.device("tpu")

    with set_current_vllm_config(vllm_cfg):
        cache_shape = PallasBatchedRPAAttentionBackend.get_kv_cache_shape(
            num_blocks, page_size, num_kv_heads, head_dim, torch.float8_e4m3fn
        )
        impl = PallasBatchedRPAAttentionBackendImpl(
            num_heads=num_heads,
            head_size=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8",
            kv_cache_quantized_dtype=torch.float8_e4m3fn,
        )

    assert impl._pool_is_seq_along_lane is (layout_env == "HND")

    prefill_len = page_size + 24
    total_len = prefill_len + 1
    torch.manual_seed(29)
    q_all = (torch.randn(total_len, num_heads, head_dim) * 0.5).to(torch.bfloat16)
    k_all = (torch.randn(total_len, num_kv_heads, rot_dim) * 0.6).to(torch.bfloat16)
    v_all = (torch.randn(total_len, num_kv_heads, head_dim) * 0.6).to(torch.bfloat16)

    kv_cache = torch.zeros(cache_shape, dtype=torch.float8_e4m3fn, device=tpu)
    block_tables = torch.tensor([2, 0], dtype=torch.int32, device=tpu)
    layer = SimpleNamespace(_k_scale_float=k_scale, _v_scale_float=v_scale)
    mesh = Mesh(np.asarray(jax.devices("tpu")[:1]), ("model",))

    def _run_step(sl: slice, seq_len: int, is_decode: bool) -> torch.Tensor:
        q_step = q_all[sl].reshape(-1, num_heads * head_dim).to(tpu)
        k_step = k_all[sl].reshape(-1, num_kv_heads * rot_dim).to(tpu)
        v_step = v_all[sl].reshape(-1, num_kv_heads * head_dim).to(tpu)
        q_len = q_step.shape[0]
        dist = [1, 1, 1] if is_decode else [0, 0, 1]
        meta = AttentionMetadata(
            input_positions=torch.arange(
                seq_len - q_len, seq_len, dtype=torch.int32, device=tpu
            ),
            block_tables=block_tables,
            seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=tpu),
            query_start_loc=torch.tensor([0, q_len], dtype=torch.int32, device=tpu),
            request_distribution=torch.tensor(dist, dtype=torch.int32, device=tpu),
        )
        with set_vllm_model_wrapper_context(mesh=mesh, vllm_config=vllm_cfg):
            out = impl.forward(
                layer=layer,
                query=q_step,
                key=k_step,
                value=v_step,
                kv_cache=kv_cache,
                attn_metadata=meta,
            )
        return out.cpu().float().reshape(q_len, num_heads, head_dim)

    out_pre = _run_step(slice(0, prefill_len), prefill_len, is_decode=False)
    out_dec = _run_step(slice(prefill_len, total_len), total_len, is_decode=True)

    def _ref(q_sl: slice, kv_len: int) -> torch.Tensor:
        q_f = q_all[q_sl].float()
        v_f = _quant_dequant_fp8(v_all[:kv_len], v_scale)
        if layout_env == "HND":
            k_rot_f = _quant_dequant_fp8(k_all[:kv_len], k_scale)
        else:
            k_rot_f = k_all[:kv_len].float()
        k_full = torch.cat([k_rot_f, v_f[..., rot_dim:]], dim=-1)
        k_rep = k_full.repeat_interleave(num_heads // num_kv_heads, dim=1)
        v_rep = v_f.repeat_interleave(num_heads // num_kv_heads, dim=1)
        scores = torch.einsum("thd,shd->hts", q_f, k_rep) * scale
        q_len = q_f.shape[0]
        q_pos = (kv_len - q_len) + torch.arange(q_len)[:, None]
        k_pos = torch.arange(kv_len)[None, :]
        scores = scores.masked_fill((k_pos > q_pos).unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        return torch.einsum("hts,shd->thd", probs, v_rep).to(torch.bfloat16).float()

    ref_pre = _ref(slice(0, prefill_len), prefill_len)
    ref_dec = _ref(slice(prefill_len, total_len), total_len)
    assert _sqnr_db(ref_pre, out_pre) > 35.0
    assert _sqnr_db(ref_dec, out_dec) > 35.0
    np.testing.assert_allclose(out_pre.numpy(), ref_pre.numpy(), atol=3e-2, rtol=2e-2)
    np.testing.assert_allclose(out_dec.numpy(), ref_dec.numpy(), atol=3e-2, rtol=2e-2)
