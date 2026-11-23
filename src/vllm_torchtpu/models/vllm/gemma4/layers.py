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
"""Shared RoPE, FP8 linear, collective, and feedforward helpers for Gemma 4."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.distributed import _functional_collectives as funcol
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_torchtpu.layers.adapter.attention import PallasBatchedRPAAttentionBackendImpl
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

GELU_COLUMN_ALIGNMENT = 128
MIN_FUSED_GELU_COLS = 2048
FUSED_QUANT_DTYPE = torch.float8_e4m3fn


def model_supports_multimodal(model_config: Any) -> bool:
    """Return True when ``model_config`` accepts multimodal inputs at runtime."""
    if model_config is None:
        return False
    if hasattr(model_config, "supports_multimodal_inputs"):
        return bool(model_config.supports_multimodal_inputs)
    return bool(getattr(model_config, "is_multimodal_model", False))


def waited(t: torch.Tensor) -> torch.Tensor:
    return funcol.wait_tensor(t) if isinstance(t, funcol.AsyncCollectiveTensor) else t


def all_gather_rows(x: torch.Tensor) -> torch.Tensor:
    return waited(funcol.all_gather_single(x, 0, get_tp_group().device_group))


def reduce_scatter_rows(x: torch.Tensor) -> torch.Tensor:
    return waited(
        funcol.reduce_scatter_single(x, "sum", 0, get_tp_group().device_group)
    )


def rotary_cos_sin(
    rope: Any, x: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos_sin = rope._match_cos_sin_cache_dtype(x).index_select(0, positions.reshape(-1))
    cos, sin = (half.contiguous() for half in cos_sin.chunk(2, dim=-1))
    return cos, sin


def split_norm_rotary_qkv(
    attn: Any,
    positions: torch.Tensor,
    qkv: torch.Tensor,
    *,
    num_heads: int | None = None,
    num_kv_heads: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    heads = attn.num_heads if num_heads is None else num_heads
    kv_heads = attn.num_kv_heads if num_kv_heads is None else num_kv_heads
    d = attn.head_dim
    q, k, v = qkv.split([heads * d, kv_heads * d, kv_heads * d], dim=-1)
    q = attn.q_norm(q.unflatten(-1, (heads, d))).flatten(-2, -1)
    k = attn.k_norm(k.unflatten(-1, (kv_heads, d))).flatten(-2, -1)
    q, k = attn.rotary_emb(positions, q, k)
    v = attn.v_norm(v.unflatten(-1, (kv_heads, d))).flatten(-2, -1)
    return q, k, v


def supports_qkv_prologue(attn: Any, impl: Any) -> bool:
    if not isinstance(impl, PallasBatchedRPAAttentionBackendImpl):
        return False
    eps = attn.q_norm.variance_epsilon
    return (
        bool(attn.rotary_emb.is_neox_style)
        and not attn.v_norm.has_weight
        and attn.k_norm.variance_epsilon == eps
        and attn.v_norm.variance_epsilon == eps
        and not impl.qkv_prologue_unsupported()
    )


def runtime_fp8_weight_dtype(linear: Any) -> torch.dtype | None:
    """Return the post-load runtime FP8 weight dtype for per-channel W8A8."""
    method = getattr(linear, "quant_method", None)
    if (
        not isinstance(method, VllmFp8LinearMethodTPU)
        or method.online_fp8
        or method.block_quant
        or method.w8a16_max_rows
        or linear.bias is not None
    ):
        return None
    _, dtype, requant_block_size, _ = method._linear_quant_config
    if requant_block_size is not None:
        return None
    if dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        return None
    return dtype


def is_per_token_fp8(linear: Any, *, expected_dtype: torch.dtype | None = None) -> bool:
    dtype = runtime_fp8_weight_dtype(linear)
    return dtype is not None and (expected_dtype is None or dtype == expected_dtype)


def prequantized_linear(
    linear: Any,
    x_bytes: torch.Tensor,
    x_scale: torch.Tensor,
    *,
    quant_dtype: torch.dtype = FUSED_QUANT_DTYPE,
) -> torch.Tensor:
    method = linear.quant_method
    rows = x_bytes.shape[0]
    if not (
        isinstance(method, VllmFp8LinearMethodTPU)
        and method.supports_quantized_input(linear, rows)
        and linear.weight.dtype == quant_dtype
        and x_bytes.dtype == torch.int8
    ):
        raise NotImplementedError(
            f"{linear.prefix} cannot consume pre-quantized {quant_dtype} rows "
            f"of shape {tuple(x_bytes.shape)}."
        )
    return method.apply_quantized_input(linear, x_bytes.view(quant_dtype), x_scale)


def apply_o_proj(o_proj: Any, out: Any) -> torch.Tensor:
    """Run ``o_proj`` on bf16 or per-token FP8 attention output rows."""
    if not isinstance(out, tuple):
        return o_proj(out)[0]
    x_q, x_scale = out
    method = o_proj.quant_method
    if not (
        isinstance(method, VllmFp8LinearMethodTPU)
        and x_q.dtype == o_proj.weight.dtype
        and method.supports_quantized_input(o_proj, x_q.shape[0])
    ):
        raise NotImplementedError(
            f"{o_proj.prefix} cannot consume pre-quantized {x_q.dtype} rows."
        )
    y = method.apply_quantized_input(o_proj, x_q, x_scale)
    if o_proj.reduce_results and o_proj.tp_size > 1:
        y = tensor_model_parallel_all_reduce(y)
    return y


def should_fuse_gelu_quant(act: Any, down: Any, gate_up: torch.Tensor) -> bool:
    if (
        gate_up.device.type != "tpu"
        or gate_up.dim() != 2
        or gate_up.dtype != torch.bfloat16
        or not isinstance(act, GeluAndMul)
        or act.approximate != "tanh"
    ):
        return False
    method = getattr(down, "quant_method", None)
    if not isinstance(method, VllmFp8LinearMethodTPU) or method.online_fp8:
        return False
    if down.bias is not None:
        return False
    cols = down.input_size_per_partition
    if cols < MIN_FUSED_GELU_COLS or cols % GELU_COLUMN_ALIGNMENT != 0:
        return False
    if down.weight.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        return False
    return bool(method.supports_quantized_input(down, gate_up.shape[0]))


def down_proj_from_gate_up(
    mlp: Any, gate_up: torch.Tensor, *, act_fn: Any = None
) -> torch.Tensor:
    """Compute ``mlp.down_proj(act_fn(gate_up))[0]``, fusing GeGLU+FP8 when safe."""
    act = mlp.act_fn if act_fn is None else act_fn
    down = mlp.down_proj
    if should_fuse_gelu_quant(act, down, gate_up):
        from vllm_torchtpu.layers.adapter.linear_common import (
            gelu_tanh_and_mul_quantize,
        )

        x_q, x_scale = gelu_tanh_and_mul_quantize(gate_up, down.weight)
        out = down.quant_method.apply_quantized_input(down, x_q, x_scale)
        if down.reduce_results and down.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out
    return down(act(gate_up))[0]


def router_weight(proj: Any) -> torch.Tensor:
    return proj.weight.t() if getattr(proj, WEIGHT_FLIPPED_ATTR, False) else proj.weight


def _layer_norms_uniform_eps(layer: Any) -> float | None:
    norms = [
        layer.input_layernorm,
        layer.post_attention_layernorm,
        layer.pre_feedforward_layernorm,
        layer.post_feedforward_layernorm,
    ]
    if layer.enable_moe_block:
        norms.extend(
            [
                layer.post_feedforward_layernorm_1,
                layer.pre_feedforward_layernorm_2,
                layer.post_feedforward_layernorm_2,
            ]
        )
    eps_set = {n.variance_epsilon for n in norms}
    if len(eps_set) != 1:
        return None
    if any(not n.has_weight or n.variance_size_override is not None for n in norms):
        return None
    return float(eps_set.pop())


def configure_layer_fused_norm_quant(layer: Any) -> bool:
    eps = _layer_norms_uniform_eps(layer)
    if eps is None:
        return False
    for linear in (layer.self_attn.qkv_proj, layer.mlp.gate_up_proj):
        if not is_per_token_fp8(linear, expected_dtype=FUSED_QUANT_DTYPE):
            return False
    from vllm_torchtpu.layers.adapter.norm_quant import (
        add_rms_norm_quantize_op,
        moe_tail_rms_norm_quantize_op,
    )

    layer.fused_norm_quant = add_rms_norm_quantize_op(eps, FUSED_QUANT_DTYPE)
    if layer.enable_moe_block:
        layer.fused_moe_tail = moe_tail_rms_norm_quantize_op(eps, FUSED_QUANT_DTYPE)
    return True


def configure_layer_fused_moe_head(layer: Any) -> bool:
    if not layer.enable_moe_block:
        return True
    router = layer.router
    if router is None:
        return False
    proj = router.proj
    eps = layer.post_attention_layernorm.variance_epsilon
    if (
        router.norm.has_weight
        or router.norm.variance_size_override is not None
        or router.norm.variance_epsilon != eps
        or proj.bias is not None
        or not isinstance(proj.quant_method, UnquantizedLinearMethod)
    ):
        return False
    from vllm_torchtpu.layers.adapter.norm_quant import moe_head_rms_norm_quantize_op

    root = float(torch.tensor(router.hidden_size**-0.5).to(proj.params_dtype))
    layer.fused_moe_head = moe_head_rms_norm_quantize_op(eps, FUSED_QUANT_DTYPE, root)
    return True


def decoder_feedforward_tail(
    layer: Any,
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    *,
    mlp_mod: Any = None,
    experts_mod: Any = None,
    project_gate_up: Callable[[Any, torch.Tensor], torch.Tensor] | None = None,
    project_prequant_gate_up: (
        Callable[[Any, torch.Tensor, torch.Tensor], torch.Tensor] | None
    ) = None,
    reduce_rows: Callable[[torch.Tensor], torch.Tensor] | None = None,
    gather_moe_inputs: bool = False,
    fused: bool = False,
    use_fused_moe_head: bool = False,
    next_input_norm: Any = None,
) -> Any:
    """Execute post-attention norm, MLP, optional MoE, and residual scalar tail."""
    mlp = layer.mlp if mlp_mod is None else mlp_mod
    experts = layer.experts if experts_mod is None else experts_mod
    moe_rows = router_logits = None
    if fused:
        w_post = layer.post_attention_layernorm.weight.view(-1)
        w_next = layer.pre_feedforward_layernorm.weight.view(-1)
        if (
            use_fused_moe_head
            and layer.enable_moe_block
            and getattr(layer, "fused_moe_head", None) is not None
        ):
            router = layer.router
            residual, q_mlp, s_mlp, moe_rows, router_logits = layer.fused_moe_head(
                attn_out,
                residual,
                w_post,
                w_next,
                layer.pre_feedforward_layernorm_2.weight.view(-1),
                router_weight(router.proj),
                router.scale,
            )
        else:
            residual, q_mlp, s_mlp = layer.fused_norm_quant(
                attn_out, residual, w_post, attn_out.new_ones(1), w_next
            )
        assert project_prequant_gate_up is not None
        gate_up = project_prequant_gate_up(mlp.gate_up_proj, q_mlp, s_mlp)
    else:
        residual = layer.post_attention_layernorm(attn_out) + residual
        normed = layer.pre_feedforward_layernorm(residual)
        gate_up = (
            mlp.gate_up_proj(normed)[0]
            if project_gate_up is None
            else project_gate_up(mlp.gate_up_proj, normed)
        )

    mlp_out = down_proj_from_gate_up(mlp, gate_up, act_fn=layer.mlp.act_fn)
    if reduce_rows is not None:
        mlp_out = reduce_rows(mlp_out)

    fused_tail = fused and next_input_norm is not None
    if layer.enable_moe_block:
        mlp_norm = None if fused_tail else layer.post_feedforward_layernorm_1(mlp_out)
        w_mlp = layer.post_feedforward_layernorm_1.weight.view(-1)
        if moe_rows is None:
            moe_rows = layer.pre_feedforward_layernorm_2(residual)
            router_logits = layer.router(residual)
            if gather_moe_inputs:
                moe_rows = all_gather_rows(moe_rows)
                router_logits = all_gather_rows(router_logits)
        experts_out = experts(moe_rows, router_logits)
        if reduce_rows is not None:
            experts_out = reduce_rows(experts_out)
        if fused_tail:
            return layer.fused_moe_tail(
                mlp_out,
                experts_out,
                residual,
                w_mlp,
                layer.post_feedforward_layernorm_2.weight.view(-1),
                layer.post_feedforward_layernorm.weight.view(-1),
                layer.layer_scalar.view(1),
                next_input_norm.weight.view(-1),
            )
        assert mlp_norm is not None
        mlp_out = mlp_norm + layer.post_feedforward_layernorm_2(experts_out)
    elif fused_tail:
        return layer.fused_norm_quant(
            mlp_out,
            residual,
            layer.post_feedforward_layernorm.weight.view(-1),
            layer.layer_scalar.view(1),
            next_input_norm.weight.view(-1),
        )
    return (layer.post_feedforward_layernorm(mlp_out) + residual) * layer.layer_scalar


def _patched_mlp_forward(self: Any, x: torch.Tensor) -> torch.Tensor:
    gate_up, _ = self.gate_up_proj(x)
    return down_proj_from_gate_up(self, gate_up)


def apply_gemma4_mlp_patch(gemma4_mod: Any) -> bool:
    """Install the fused GeGLU+FP8 forward on ``Gemma4MLP`` idempotently."""
    mlp_cls = gemma4_mod.Gemma4MLP
    if getattr(mlp_cls, "_TPU_GEMMA4_MLP_PATCHED", False):
        return False
    mlp_cls._tpu_orig_forward = mlp_cls.forward
    mlp_cls.forward = _patched_mlp_forward
    mlp_cls._TPU_GEMMA4_MLP_PATCHED = True
    logger.info(
        "Applied TPU patch: Gemma 4 MLP fuses GeGLU and FP8 down-proj quantization "
        "for per-rank widths >= %d.",
        MIN_FUSED_GELU_COLS,
    )
    return True
