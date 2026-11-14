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
"""Parity, tile-selection, AST-discovery, and fullgraph tests for RMSNorm+FP8."""

from types import SimpleNamespace
from unittest import mock

import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm import envs as vllm_envs
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.models import gemma4 as gemma4_mod
from vllm.model_executor.models import llama as llama_mod
from vllm.model_executor.models import qwen2 as qwen2_mod

from vllm_torchtpu.kernels import rmsnorm_fp8 as kernel
from vllm_torchtpu.kernels.quantized_matmul.util import (
    quantize_tensor,
    xla_quantized_matmul,
)
from vllm_torchtpu.layers.adapter import rmsnorm_fp8 as adapter
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    WEIGHT_FLIPPED_ATTR,
    quantized_matmul,
)
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU


class _FakeTpuTensor(torch.Tensor):
    @property
    def device(self):  # type: ignore[override]
        return SimpleNamespace(type="tpu", index=0)


def _reference_rmsnorm_fp8_matmul(
    x: jax.Array,
    gamma: jax.Array,
    w_q: jax.Array,
    w_s: jax.Array,
    eps: float = 1e-6,
    gamma_offset: float = 0.0,
    *,
    quantize_activation: bool = True,
) -> tuple[jax.Array, jax.Array]:
    xf = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(xf), axis=-1, keepdims=True)
    gf = gamma.astype(jnp.float32) + gamma_offset
    norm_bf16 = (xf * jax.lax.rsqrt(var + eps) * gf).astype(x.dtype)
    return (
        xla_quantized_matmul(
            norm_bf16, w_q, w_s, quantize_activation=quantize_activation
        ),
        norm_bf16,
    )


def _fp32_oracle(norm_bf16: jax.Array, w_q: jax.Array, w_s: jax.Array) -> jax.Array:
    w_deq = w_q.astype(jnp.float32) * w_s[None, :]
    return (norm_bf16.astype(jnp.float32) @ w_deq).astype(norm_bf16.dtype)


def _make_fp8_linear(
    cls: type[torch.nn.Module],
    k: int,
    n_out: int,
    *,
    w8a16_max_rows: int = 0,
    prefix: str = "model.layers.0.mlp.gate_up_proj",
    flipped: bool = True,
    bias_tensor: torch.Tensor | None = None,
    skip_bias_add: bool = False,
    return_bias: bool = True,
    weight_tensor: torch.Tensor | None = None,
    scale_tensor: torch.Tensor | None = None,
):
    lin = cls.__new__(cls)
    torch.nn.Module.__init__(lin)
    qm = VllmFp8LinearMethodTPU.__new__(VllmFp8LinearMethodTPU)
    qm.w8a16_max_rows = w8a16_max_rows
    lin.quant_method = qm
    lin.prefix = prefix
    lin.gather_output = False
    lin.skip_bias_add = skip_bias_add
    lin.return_bias = return_bias
    if flipped:
        setattr(lin, WEIGHT_FLIPPED_ATTR, True)
    if bias_tensor is None:
        lin.register_parameter("bias", None)
    else:
        lin.register_parameter(
            "bias", torch.nn.Parameter(bias_tensor, requires_grad=False)
        )
    w = (
        torch.empty((k, n_out), dtype=torch.float8_e4m3fn, device="meta")
        if weight_tensor is None
        else weight_tensor
    )
    s = (
        torch.empty((n_out,), dtype=torch.float32, device="meta")
        if scale_tensor is None
        else scale_tensor
    )
    lin.register_parameter("weight", torch.nn.Parameter(w, requires_grad=False))
    lin.register_parameter("weight_scale", torch.nn.Parameter(s, requires_grad=False))
    return lin


def test_select_block_m_and_should_use_pallas_gating_cpu():
    """CPU check of VMEM tile bounds, bucket gating, admission errors, and W8A16."""
    v7x_vmem = int(
        pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1).vmem_capacity_bytes
    )
    assert kernel.vmem_limit_bytes(v7x_vmem) == 56 * 1024 * 1024

    # k=5376 (Gemma 4 31B): bm=512 fits within 56 MiB (42.5 MiB with f32 scratch).
    assert kernel.select_block_m(128, 5376, v7x_vmem) == 128
    assert kernel.select_block_m(216, 5376, v7x_vmem) == 216
    assert kernel.select_block_m(256, 5376, v7x_vmem) == 256
    assert kernel.select_block_m(512, 5376, v7x_vmem) == 512
    assert kernel.select_block_m(16384, 5376, v7x_vmem) == 512

    # k=8192 (70B hidden size): bm=512 would exceed 56 MiB (64.5 MiB) -> steps to 256.
    assert kernel.select_block_m(16384, 8192, v7x_vmem) == 256

    # Alignment and empty/invalid bounds:
    assert kernel.select_block_m(0, 5376, v7x_vmem) is None
    assert kernel.select_block_m(128, 64, v7x_vmem) is None
    assert kernel.select_block_m(128, 130, v7x_vmem) is None
    assert kernel.select_block_m(130, 5376, v7x_vmem) is None
    assert kernel.select_block_m(600, 5376, v7x_vmem) is None
    assert kernel.select_block_m(256, 5376, 0) is None

    for n_out, m, w8a16, expected in [
        (8192, 64, 0, False),  # below FP8_RMSNORM_MIN_TOKENS=128
        (8192, 128, 0, True),
        (8192, 216, 0, True),
        (8192, 256, 0, True),
        (8192, 256, 256, False),  # W8A16 bucket stays on native XLA W8A16
        (8192, 512, 256, True),  # above w8a16_max_rows -> Pallas FP8
        (21504, 16384, 0, True),  # TP=2 gate_up_proj full prefill
        (10752, 16384, 0, True),  # TP=4 gate_up_proj full prefill
        (5376, 4096, 0, True),  # TP=8 gate_up_proj 4K prefill
        (5376, 8192, 0, True),  # TP=8 gate_up_proj 8K prefill
        (5376, 16384, 0, False),  # TP=8 narrow gate_up_proj (N < 6144) at 16K prefill
        (8192, 130, 0, False),  # M % 8 != 0
        (0, 256, 0, False),  # invalid N <= 0
    ]:
        assert (
            kernel.should_use_pallas_rmsnorm_fp8(
                m, 5376, n_out, v7x_vmem, w8a16_max_rows=w8a16
            )
            is expected
        )

    # Exact admission checks on kernel entry points:
    x_f32 = jnp.ones((128, 128), dtype=jnp.float32)
    x_bf16 = jnp.ones((128, 128), dtype=jnp.bfloat16)
    g_bf16 = jnp.ones((128,), dtype=jnp.bfloat16)
    wq_fp8 = jnp.ones((128, 256), dtype=jnp.float8_e4m3fn)
    ws_f32 = jnp.ones((256,), dtype=jnp.float32)

    with pytest.raises(ValueError, match="2D bfloat16"):
        kernel.fused_rmsnorm_fp8_matmul(
            x_f32, g_bf16, wq_fp8, ws_f32, vmem_capacity_bytes=v7x_vmem
        )
    with pytest.raises(ValueError, match="gamma of shape"):
        kernel.fused_rmsnorm_fp8_matmul(
            x_bf16,
            jnp.ones((64,), dtype=jnp.bfloat16),
            wq_fp8,
            ws_f32,
            vmem_capacity_bytes=v7x_vmem,
        )
    with pytest.raises(ValueError, match="1-byte floating FP8"):
        kernel.fused_rmsnorm_fp8_matmul(
            x_bf16,
            g_bf16,
            jnp.ones((128, 256), dtype=jnp.int8),
            ws_f32,
            vmem_capacity_bytes=v7x_vmem,
        )
    with pytest.raises(ValueError, match="Invalid tile config"):
        kernel.fused_rmsnorm_fp8_quant(
            x_bf16,
            g_bf16,
            jnp.float8_e4m3fn,
            bm=48,
            vmem_capacity_bytes=v7x_vmem,
        )

    # CPU JAX check of fallback paths (m < 128 W8A8 and w8a16_max_rows W8A16):
    k1, k2, k3 = jax.random.split(jax.random.key(42), 3)
    x_small = jax.random.normal(k1, (64, 128), jnp.float32).astype(jnp.bfloat16)
    gamma = (1.0 + 0.2 * jax.random.normal(k2, (128,), jnp.float32)).astype(
        jnp.bfloat16
    )
    w_raw = jax.random.normal(k3, (128, 256), jnp.float32) * 0.1
    wq, ws = quantize_tensor(w_raw.T, jnp.float8_e4m3fn)
    wq, ws = wq.T, ws.reshape(-1)

    out_m64 = kernel.fused_rmsnorm_fp8_matmul(
        x_small, gamma, wq, ws, eps=1e-6, gamma_offset=0.0, vmem_capacity_bytes=v7x_vmem
    )
    ref_m64, _ = _reference_rmsnorm_fp8_matmul(
        x_small, gamma, wq, ws, eps=1e-6, gamma_offset=0.0, quantize_activation=True
    )
    np.testing.assert_array_equal(np.asarray(out_m64), np.asarray(ref_m64))

    x_256 = jax.random.normal(k1, (256, 128), jnp.float32).astype(jnp.bfloat16)
    out_w8a16 = kernel.fused_rmsnorm_fp8_matmul(
        x_256,
        gamma,
        wq,
        ws,
        eps=1e-6,
        gamma_offset=1.0,
        w8a16_max_rows=256,
        vmem_capacity_bytes=v7x_vmem,
    )
    ref_w8a16, _ = _reference_rmsnorm_fp8_matmul(
        x_256, gamma, wq, ws, eps=1e-6, gamma_offset=1.0, quantize_activation=False
    )
    ref_w8a8, _ = _reference_rmsnorm_fp8_matmul(
        x_256, gamma, wq, ws, eps=1e-6, gamma_offset=1.0, quantize_activation=True
    )
    np.testing.assert_array_equal(np.asarray(out_w8a16), np.asarray(ref_w8a16))
    # Must-fail control: W8A16 skips activation quantization and differs from W8A8.
    assert not np.array_equal(np.asarray(out_w8a16), np.asarray(ref_w8a8))


def test_can_fuse_controls_and_ast_discovery_cpu(monkeypatch):
    """Verify MergedColumnParallelLinear scoping, must-fail guards, and AST matching."""
    v7x_vmem = int(
        pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1).vmem_capacity_bytes
    )

    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        rms = RMSNorm(5376, eps=1e-6)
        rms_no_w = RMSNorm(5376, eps=1e-6, has_weight=False)
        rms_var_ovr = RMSNorm(5376, eps=1e-6, var_hidden_size=2048)
        gemma_rms = GemmaRMSNorm(5376, eps=1e-6)

    assert adapter._resolve_gamma_offset(rms) == 0.0
    assert adapter._resolve_gamma_offset(gemma_rms) == 1.0
    assert adapter._resolve_gamma_offset(rms_no_w) is None
    assert adapter._resolve_gamma_offset(rms_var_ovr) is None

    x_256 = torch.zeros((256, 5376), dtype=torch.bfloat16).as_subclass(_FakeTpuTensor)
    x_64 = torch.zeros((64, 5376), dtype=torch.bfloat16).as_subclass(_FakeTpuTensor)

    lin_merged = _make_fp8_linear(MergedColumnParallelLinear, 5376, 21504)
    assert adapter.can_fuse_rmsnorm_fp8_linear(x_256, rms, lin_merged, v7x_vmem)
    assert adapter.can_fuse_rmsnorm_fp8_linear(x_256, gemma_rms, lin_merged, v7x_vmem)
    # Dynamo-level check stays guard-free across M < 128 and w8a16_max_rows > 0:
    assert adapter.can_fuse_rmsnorm_fp8_linear(x_64, rms, lin_merged, v7x_vmem)
    lin_w8a16 = _make_fp8_linear(
        MergedColumnParallelLinear, 5376, 21504, w8a16_max_rows=256
    )
    assert adapter.can_fuse_rmsnorm_fp8_linear(x_256, rms, lin_w8a16, v7x_vmem)

    # Attention QKVParallelLinear, plain ColumnParallelLinear, RowParallelLinear,
    # and subclasses of MergedColumnParallelLinear must stay unfused:
    class _SubMergedLinear(MergedColumnParallelLinear):
        pass

    for rejected_cls in (
        QKVParallelLinear,
        ColumnParallelLinear,
        RowParallelLinear,
        _SubMergedLinear,
    ):
        lin_rej = _make_fp8_linear(rejected_cls, 5376, 8192)
        assert not adapter.can_fuse_rmsnorm_fp8_linear(x_256, rms, lin_rej, v7x_vmem)

    # Norm, layout, unflipped square weight, drafter prefix, and batch-invariant guards:
    assert not adapter.can_fuse_rmsnorm_fp8_linear(
        x_256, rms_no_w, lin_merged, v7x_vmem
    )
    assert not adapter.can_fuse_rmsnorm_fp8_linear(
        x_256, rms_var_ovr, lin_merged, v7x_vmem
    )

    setattr(lin_merged, KEEP_VLLM_LAYOUT_ATTR, True)
    assert not adapter.can_fuse_rmsnorm_fp8_linear(x_256, rms, lin_merged, v7x_vmem)
    delattr(lin_merged, KEEP_VLLM_LAYOUT_ATTR)

    lin_unflipped_sq = _make_fp8_linear(
        MergedColumnParallelLinear, 5376, 5376, flipped=False
    )
    assert not adapter.can_fuse_rmsnorm_fp8_linear(
        x_256, rms, lin_unflipped_sq, v7x_vmem
    )

    lin_drafter = _make_fp8_linear(
        MergedColumnParallelLinear,
        5376,
        21504,
        prefix="draft_model.layers.0.mlp.gate_up_proj",
    )
    assert not adapter.can_fuse_rmsnorm_fp8_linear(x_256, rms, lin_drafter, v7x_vmem)

    lin_block_scale = _make_fp8_linear(
        MergedColumnParallelLinear,
        5376,
        21504,
        scale_tensor=torch.empty((1, 42, 1, 21504), dtype=torch.float32, device="meta"),
    )
    assert not adapter.can_fuse_rmsnorm_fp8_linear(
        x_256, rms, lin_block_scale, v7x_vmem
    )

    monkeypatch.setattr(vllm_envs, "VLLM_BATCH_INVARIANT", True)
    assert not adapter.can_fuse_rmsnorm_fp8_linear(x_256, rms, lin_merged, v7x_vmem)
    monkeypatch.setattr(vllm_envs, "VLLM_BATCH_INVARIANT", False)

    # Real upstream vLLM class AST discovery checks:
    assert (
        "pre_feedforward_layernorm",
        "mlp",
        0,
    ) in adapter._find_fusible_norm_submod_pairs(gemma4_mod.Gemma4DecoderLayer.forward)
    assert adapter._find_fusible_norm_submod_pairs(
        llama_mod.LlamaDecoderLayer.forward
    ) == [("post_attention_layernorm", "mlp", 0)]
    assert adapter._find_fusible_norm_submod_pairs(
        qwen2_mod.Qwen2DecoderLayer.forward
    ) == [("post_attention_layernorm", "mlp", 0)]
    assert adapter._entry_projection_attr(gemma4_mod.Gemma4MLP.forward, 0) == (
        "gate_up_proj"
    )
    assert adapter._entry_projection_attr(qwen2_mod.Qwen2MLP.forward, 0) == (
        "gate_up_proj"
    )
    # Gemma4Attention branching (`q_proj` vs `qkv_proj`) is intentionally rejected:
    assert (
        adapter._entry_projection_attr(
            gemma4_mod.Gemma4Attention.forward, "hidden_states"
        )
        is None
    )

    class _MultiUseNormLayer(torch.nn.Module):
        def forward(self, hidden_states):
            normed = self.pre_feedforward_layernorm(hidden_states)
            out = self.mlp(normed)
            return out + normed

    class _LeakyRebindMLP(torch.nn.Module):
        def forward(self, x):
            gate_up, _ = self.gate_up_proj(x)
            x = self.act_fn(gate_up) + x
            return x

    assert adapter._find_fusible_norm_submod_pairs(_MultiUseNormLayer.forward) == []
    assert adapter._entry_projection_attr(_LeakyRebindMLP.forward, 0) is None

    assert adapter.maybe_patch_rmsnorm_fp8(None) is False
    monkeypatch.setattr(adapter, "get_chip_version", lambda: pltpu.ChipVersion.TPU_V6E)
    assert (
        adapter.maybe_patch_rmsnorm_fp8(
            SimpleNamespace(architectures=["Gemma4ForCausalLM"], model_impl="auto")
        )
        is None
    )


def test_decoder_layer_binding_and_torch_compile_fullgraph_cpu(monkeypatch):
    """Verify Gemma4 (1-arg) and Llama/Qwen (2-arg) MLP binding under fullgraph=True."""
    k, n_qkv, n_gu = 128, 256, 512
    vmem = int(
        pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1).vmem_capacity_bytes
    )
    rng = torch.Generator().manual_seed(17)

    def _cpu_fp8_proj(cls: type[torch.nn.Module], n_out: int, prefix: str):
        w_q = (torch.randn((k, n_out), generator=rng) * 0.05).to(torch.float8_e4m3fn)
        w_s = torch.full((n_out,), 0.04, dtype=torch.float32)
        bias = (torch.randn((n_out,), generator=rng) * 0.01).to(torch.bfloat16)
        proj = _make_fp8_linear(
            cls,
            k,
            n_out,
            prefix=prefix,
            bias_tensor=bias,
            weight_tensor=w_q,
            scale_tensor=w_s,
        )
        return proj

    def _cpu_quant_mm(
        x: torch.Tensor,
        gamma: torch.Tensor,
        w_q: torch.Tensor,
        w_s: torch.Tensor,
        eps: float = 1e-6,
        gamma_offset: float = 0.0,
        w8a16_max_rows: int = 0,
        vmem_capacity_bytes: int = 0,
    ) -> torch.Tensor:
        del w8a16_max_rows, vmem_capacity_bytes
        raw_x = x.as_subclass(torch.Tensor) if type(x) is _FakeTpuTensor else x
        xf = raw_x.float()
        inv_rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
        normed = (xf * inv_rms * (gamma.float() + gamma_offset)).to(raw_x.dtype)
        w_deq = w_q.float() * w_s.unsqueeze(0)
        out = (normed.float() @ w_deq).to(raw_x.dtype)
        return out.as_subclass(_FakeTpuTensor) if type(x) is _FakeTpuTensor else out

    op_spy = mock.Mock(side_effect=_cpu_quant_mm)
    monkeypatch.setattr(adapter, "_rmsnorm_fp8_mm_op", op_spy)
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8.quantized_matmul",
        lambda x, w, ws, w8a16_max_rows=0: (
            (
                (
                    x.as_subclass(torch.Tensor) if type(x) is _FakeTpuTensor else x
                ).float()
                @ (w.float() * ws.unsqueeze(0))
            )
            .to(x.dtype)
            .as_subclass(_FakeTpuTensor if type(x) is _FakeTpuTensor else torch.Tensor)
        ),
    )

    class _StubAttn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv_proj = _cpu_fp8_proj(
                QKVParallelLinear, n_qkv, "model.layers.0.self_attn.qkv_proj"
            )

        def forward(self, positions, hidden_states):
            del positions
            qkv, _ = self.qkv_proj(hidden_states)
            return qkv[..., :k]

    class _StubMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = _cpu_fp8_proj(
                MergedColumnParallelLinear, n_gu, "model.layers.0.mlp.gate_up_proj"
            )

        def forward(self, x):
            gate_up, _ = self.gate_up_proj(x)
            x = gate_up[..., :k]
            return x

    class _Gemma4StyleDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = GemmaRMSNorm(k, eps=1e-6)
            self.self_attn = _StubAttn()
            self.pre_feedforward_layernorm = GemmaRMSNorm(k, eps=1e-6)
            self.mlp = _StubMLP()

        def forward(self, positions, hidden_states):
            residual = hidden_states
            hidden_states = self.input_layernorm(residual)
            hidden_states = self.self_attn(
                positions=positions, hidden_states=hidden_states
            )
            hidden_states = hidden_states + residual
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            return hidden_states

    class _LlamaStyleDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = RMSNorm(k, eps=1e-6)
            self.self_attn = _StubAttn()
            self.post_attention_layernorm = RMSNorm(k, eps=1e-6)
            self.mlp = _StubMLP()

        def forward(self, positions, hidden_states, residual):
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            hidden_states = self.self_attn(
                positions=positions, hidden_states=hidden_states
            )
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
            hidden_states = self.mlp(hidden_states)
            return hidden_states, residual

    for cls in (_Gemma4StyleDecoder, _LlamaStyleDecoder):
        pairs = adapter._find_fusible_norm_submod_pairs(cls.forward)
        cls.__init__ = adapter._make_patched_decoder_init(  # type: ignore[method-assign]
            cls.__init__, pairs, vmem
        )

    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        gemma_layer = _Gemma4StyleDecoder()
        llama_layer = _LlamaStyleDecoder()

    # Attention QKVParallelLinear stays unfused (owned by fused_attn_proj);
    # only pre-MLP MergedColumnParallelLinear is bound!
    assert not getattr(gemma_layer.input_layernorm, "_tpu_rmsnorm_fp8_bound", False)
    assert not getattr(gemma_layer.self_attn.qkv_proj, "_tpu_rmsnorm_fp8_bound", False)
    assert getattr(
        gemma_layer.pre_feedforward_layernorm, "_tpu_rmsnorm_fp8_bound", False
    )
    assert getattr(gemma_layer.mlp.gate_up_proj, "_tpu_rmsnorm_fp8_bound", False)

    assert not getattr(llama_layer.input_layernorm, "_tpu_rmsnorm_fp8_bound", False)
    assert not getattr(llama_layer.self_attn.qkv_proj, "_tpu_rmsnorm_fp8_bound", False)
    assert getattr(
        llama_layer.post_attention_layernorm, "_tpu_rmsnorm_fp8_bound", False
    )
    assert getattr(llama_layer.mlp.gate_up_proj, "_tpu_rmsnorm_fp8_bound", False)

    # Runtime CPU-tensor fallback check (verifies single normalization, no double norm,
    # and info_once diagnostic against an unpatched reference layer):
    unpatched_ref = _LlamaStyleDecoder.__new__(_LlamaStyleDecoder)
    torch.nn.Module.__init__(unpatched_ref)
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        unpatched_ref.input_layernorm = RMSNorm(k, eps=1e-6)
        unpatched_ref.self_attn = _StubAttn()
        unpatched_ref.post_attention_layernorm = RMSNorm(k, eps=1e-6)
        unpatched_ref.mlp = _StubMLP()
    unpatched_ref.load_state_dict(llama_layer.state_dict())

    logged: list[str] = []
    monkeypatch.setattr(
        adapter.logger,
        "info_once",
        lambda msg, *args, **kwargs: logged.append(msg % args if args else msg),
    )
    h_cpu = torch.randn((8, k), generator=rng, dtype=torch.bfloat16)
    r_cpu = torch.randn((8, k), generator=rng, dtype=torch.bfloat16)
    out_fb, res_fb = llama_layer(torch.arange(8), h_cpu.clone(), r_cpu.clone())
    out_ref, res_ref = unpatched_ref(torch.arange(8), h_cpu.clone(), r_cpu.clone())
    assert op_spy.call_count == 0
    assert any("Falling back to unfused pre-MLP" in m for m in logged)
    torch.testing.assert_close(res_fb, res_ref, rtol=0, atol=0)
    torch.testing.assert_close(out_fb, out_ref, rtol=1e-2, atol=2e-2)

    # 1. Eager _FakeTpuTensor check: bound MLP dispatches to _rmsnorm_fp8_mm_op
    # while unbound QKVParallelLinear stays on quantized_matmul:
    for m_eager in (16, 256):
        pos_e = torch.arange(m_eager)
        h_e = torch.randn(
            (m_eager, k), generator=rng, dtype=torch.bfloat16
        ).as_subclass(_FakeTpuTensor)
        r_e = torch.randn(
            (m_eager, k), generator=rng, dtype=torch.bfloat16
        ).as_subclass(_FakeTpuTensor)

        before = op_spy.call_count
        g_e = gemma_layer(pos_e, h_e)
        assert op_spy.call_count == before + 1
        assert op_spy.call_args.args[5] == 1.0  # GemmaRMSNorm gamma_offset=1.0
        assert g_e.shape == (m_eager, k)

        before = op_spy.call_count
        l_e, lr_e = llama_layer(pos_e, h_e, r_e)
        assert op_spy.call_count == before + 1
        assert op_spy.call_args.args[5] == 0.0  # RMSNorm gamma_offset=0.0
        assert l_e.shape == (m_eager, k) and lr_e.shape == (m_eager, k)

    # 2. Fullgraph dynamic compilation across M in {16, 128, 256, 512, 1024} and 3D
    # [B, S, K] on both bound decoder layers and fused_rmsnorm_fp8_linear:
    compiled_graphs: list[int] = []

    def _counting_backend(gm: torch.fx.GraphModule, _example_inputs):
        compiled_graphs.append(1)
        return gm.forward

    def _traceable_mm(
        x: torch.Tensor,
        gamma: torch.Tensor,
        w_q: torch.Tensor,
        w_s: torch.Tensor,
        eps: float = 1e-6,
        gamma_offset: float = 0.0,
        w8a16_max_rows: int = 0,
        vmem_capacity_bytes: int = 0,
    ) -> torch.Tensor:
        del w8a16_max_rows, vmem_capacity_bytes
        xf = x.float()
        inv_rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
        normed = (xf * inv_rms * (gamma.float() + gamma_offset)).to(x.dtype)
        return (normed.float() @ (w_q.float() * w_s.unsqueeze(0))).to(x.dtype)

    monkeypatch.setattr(adapter, "_rmsnorm_fp8_mm_op", _traceable_mm)
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8.quantized_matmul",
        lambda x, w, ws, w8a16_max_rows=0: (
            x.float() @ (w.float() * ws.unsqueeze(0))
        ).to(x.dtype),
    )
    gemma_compiled = torch.compile(
        gemma_layer, backend="eager", fullgraph=True, dynamic=True
    )
    llama_compiled = torch.compile(
        llama_layer, backend="eager", fullgraph=True, dynamic=True
    )
    fused_linear_compiled = torch.compile(
        lambda x: adapter.fused_rmsnorm_fp8_linear(
            x,
            gemma_layer.pre_feedforward_layernorm,
            gemma_layer.mlp.gate_up_proj,
            vmem,
        ),
        backend=_counting_backend,
        fullgraph=True,
        dynamic=True,
    )
    with torch.no_grad():
        for m in (16, 128, 256, 512, 1024):
            pos = torch.arange(m)
            h_in = torch.randn((m, k), generator=rng, dtype=torch.bfloat16)
            r_in = torch.randn((m, k), generator=rng, dtype=torch.bfloat16)

            g_out = gemma_compiled(pos, h_in)
            assert g_out.shape == (m, k)

            for res_arg in (None, r_in):
                l_out, l_res = llama_compiled(pos, h_in.clone(), res_arg)
                assert l_out.shape == (m, k) and l_res.shape == (m, k)

            gu_out, gu_bias = fused_linear_compiled(h_in)
            assert gu_out.shape == (m, n_gu) and gu_bias is None

        # Single dynamic FX graph across all 5 2D token buckets (no SymInt split):
        assert len(compiled_graphs) == 1

        h_3d = torch.randn((2, 64, k), generator=rng, dtype=torch.bfloat16)
        gu_3d, _ = fused_linear_compiled(h_3d)
        assert gu_3d.shape == (2, 64, n_gu)


@pytest.mark.tpu_test
@pytest.mark.parametrize("m", [128, 216, 256, 512, 1024, 4096, 16384])
@pytest.mark.parametrize("k,n", [(5376, 8192), (1024, 512)])
@pytest.mark.parametrize("gamma_offset", [0.0, 1.0])
def test_pallas_rmsnorm_fp8_parity_tpu(m, k, n, gamma_offset):
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU")

    vmem = int(pltpu.get_tpu_info().vmem_capacity_bytes)
    bm = kernel.select_block_m(m, k, vmem)
    if bm is None:
        pytest.skip("shape not eligible")

    k1, k2, k3 = jax.random.split(jax.random.key(m + k + int(gamma_offset * 10)), 3)
    x = jax.random.normal(k1, (m, k), jnp.float32).astype(jnp.bfloat16)
    gamma_dtype = jnp.float32 if gamma_offset == 1.0 else jnp.bfloat16
    gamma = (0.3 * jax.random.normal(k2, (k,), jnp.float32)).astype(gamma_dtype)
    if gamma_offset == 0.0:
        gamma = (1.0 + gamma.astype(jnp.float32)).astype(jnp.bfloat16)
    w = jax.random.normal(k3, (k, n), jnp.float32) * 0.02
    w_q, w_s = quantize_tensor(w.T, jnp.float8_e4m3fn)
    w_q, w_s = w_q.T, w_s.reshape(-1)

    if m == 256 and k == 1024:
        x_zero_row = x.at[3].set(0)
        xq_z, xs_z = jax.jit(
            lambda xi, gi: kernel.fused_rmsnorm_fp8_quant(
                xi,
                gi,
                jnp.float8_e4m3fn,
                eps=1e-6,
                gamma_offset=gamma_offset,
                bm=bm,
                vmem_capacity_bytes=vmem,
            )
        )(x_zero_row, gamma)
        assert float(xs_z[3, 0]) == 1.0
        assert not bool(jnp.any(jnp.isnan(xq_z.astype(jnp.float32))))

    fused = jax.jit(
        lambda xi, gi, wq, ws: kernel.fused_rmsnorm_fp8_matmul(
            xi, gi, wq, ws, eps=1e-6, gamma_offset=gamma_offset
        )
    )(x, gamma, w_q, w_s)
    ref, norm_bf16 = jax.jit(
        lambda xi, gi, wq, ws: _reference_rmsnorm_fp8_matmul(
            xi, gi, wq, ws, eps=1e-6, gamma_offset=gamma_offset
        )
    )(x, gamma, w_q, w_s)
    oracle = jax.jit(_fp32_oracle)(norm_bf16, w_q, w_s)

    assert fused.shape == ref.shape and fused.dtype == ref.dtype
    f = np.asarray(fused.astype(jnp.float32))
    r = np.asarray(ref.astype(jnp.float32))
    o = np.asarray(oracle.astype(jnp.float32))

    cos_fo = float(np.sum(f * o) / (np.linalg.norm(f) * np.linalg.norm(o)))
    cos_ro = float(np.sum(r * o) / (np.linalg.norm(r) * np.linalg.norm(o)))
    assert cos_fo > 0.9995, f"fused vs fp32 oracle cosine too low: {cos_fo:.6f}"

    if kernel.should_use_pallas_rmsnorm_fp8(m, k, n, vmem):
        cos_fr = float(np.sum(f * r) / (np.linalg.norm(f) * np.linalg.norm(r)))
        rms_rel_fr = float(np.linalg.norm(f - r) / np.linalg.norm(r))
        assert cos_fr > 0.9999, f"fused vs ref cosine too low: {cos_fr:.6f}"
        assert rms_rel_fr < 0.005, (
            f"fused vs ref RMS rel diff too high: {rms_rel_fr:.6f}"
        )
        assert abs(cos_fo - cos_ro) < 2e-4
        # Must-fail control: perturbing fused output trips the RMS check.
        f_bad = f.copy()
        f_bad[0] += 10.0
        assert float(np.linalg.norm(f_bad - r) / np.linalg.norm(r)) >= 0.005
    else:
        ref_inlined = jax.jit(
            lambda xi, gi, wq, ws: _reference_rmsnorm_fp8_matmul(
                xi, gi, wq, ws, eps=1e-6, gamma_offset=gamma_offset
            )[0]
        )(x, gamma, w_q, w_s)
        np.testing.assert_array_equal(np.asarray(fused), np.asarray(ref_inlined))


@pytest.mark.tpu_test
def test_decoder_layer_rmsnorm_fp8_torch_compile_fullgraph_tpu():
    """Verify fullgraph=True on TPU leaves QKVParallelLinear unfused and fuses MLP."""
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU")

    k, n_qkv, n_gu = 1024, 2048, 4096
    vmem = int(
        pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1).vmem_capacity_bytes
    )

    def _make_tpu_proj(cls: type[torch.nn.Module], n_out: int, prefix: str):
        w_q = (torch.randn((k, n_out), dtype=torch.float32) * 0.02).to(
            torch.float8_e4m3fn
        )
        w_s = torch.full((n_out,), 0.02, dtype=torch.float32)
        bias = (torch.randn((n_out,), dtype=torch.bfloat16) * 0.01).to("tpu")
        return _make_fp8_linear(
            cls,
            k,
            n_out,
            prefix=prefix,
            bias_tensor=bias,
            weight_tensor=w_q.to("tpu"),
            scale_tensor=w_s.to("tpu"),
        )

    class _ToyAttn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv_proj = _make_tpu_proj(
                QKVParallelLinear, n_qkv, "model.layers.0.self_attn.qkv_proj"
            )

        def forward(self, positions, hidden_states):
            del positions
            qkv, _ = self.qkv_proj(hidden_states)
            return qkv[..., :k]

    class _ToyMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = _make_tpu_proj(
                MergedColumnParallelLinear, n_gu, "model.layers.0.mlp.gate_up_proj"
            )

        def forward(self, x):
            gate_up, _ = self.gate_up_proj(x)
            x = gate_up[..., :k]
            return x

    class _ToyDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = RMSNorm(k, eps=1e-6).to("tpu")
            self.self_attn = _ToyAttn()
            self.post_attention_layernorm = RMSNorm(k, eps=1e-6).to("tpu")
            self.mlp = _ToyMLP()

        def forward(self, positions, hidden_states, residual):
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            hidden_states = self.self_attn(
                positions=positions, hidden_states=hidden_states
            )
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
            hidden_states = self.mlp(hidden_states)
            return hidden_states, residual

    pairs = adapter._find_fusible_norm_submod_pairs(_ToyDecoderLayer.forward)
    assert pairs == [("post_attention_layernorm", "mlp", 0)]
    _ToyDecoderLayer.__init__ = adapter._make_patched_decoder_init(  # type: ignore[method-assign]
        _ToyDecoderLayer.__init__, pairs, vmem
    )

    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        layer = _ToyDecoderLayer()

    assert not getattr(layer.input_layernorm, "_tpu_rmsnorm_fp8_bound", False)
    assert not getattr(layer.self_attn.qkv_proj, "_tpu_rmsnorm_fp8_bound", False)
    assert getattr(layer.post_attention_layernorm, "_tpu_rmsnorm_fp8_bound", False)
    assert getattr(layer.mlp.gate_up_proj, "_tpu_rmsnorm_fp8_bound", False)

    def _manual_ref(h: torch.Tensor, res: torch.Tensor | None):
        def _rn(t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            tf = t.float()
            inv_rms = torch.rsqrt(tf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            return (tf * inv_rms * w.float()).to(t.dtype)

        r1 = h if res is None else (h.float() + res.float()).to(h.dtype)
        n1 = _rn(r1, layer.input_layernorm.weight)
        qkv_proj = layer.self_attn.qkv_proj
        qkv = (
            quantized_matmul(n1, qkv_proj.weight, qkv_proj.weight_scale) + qkv_proj.bias
        )
        a_out = qkv[..., :k]
        r2 = (a_out.float() + r1.float()).to(h.dtype)
        n2 = _rn(r2, layer.post_attention_layernorm.weight)
        gu_proj = layer.mlp.gate_up_proj
        gu = quantized_matmul(n2, gu_proj.weight, gu_proj.weight_scale) + gu_proj.bias
        return gu[..., :k], r2

    compiled = torch.compile(layer, backend="eager", fullgraph=True, dynamic=True)
    with torch.no_grad():
        for m in (16, 128, 256, 512, 1024):
            pos = torch.arange(m, device="tpu")
            h_in = torch.randn((m, k), dtype=torch.bfloat16).to("tpu")
            r_in = torch.randn((m, k), dtype=torch.bfloat16).to("tpu")

            for res_arg in (None, r_in):
                out_c, res_c = compiled(pos, h_in, res_arg)
                out_r, res_r = _manual_ref(h_in, res_arg)
                torch.testing.assert_close(res_c.cpu(), res_r.cpu(), rtol=0, atol=0)
                oc = out_c.cpu().float()
                orr = out_r.cpu().float()
                cos = float(
                    torch.sum(oc * orr)
                    / (torch.linalg.norm(oc) * torch.linalg.norm(orr))
                )
                assert cos > 0.9999, f"m={m} res={res_arg is not None} cos={cos:.6f}"
