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
"""Fused MLP GeGLU quantization and QKV attention prologue for Gemma 4."""

from __future__ import annotations

from typing import Any

import torch
from vllm.config import get_current_vllm_config
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.models.gemma4 import Gemma4Attention, Gemma4MLP

from vllm_torchtpu.kernels.gelu_quant import COLUMN_ALIGNMENT
from vllm_torchtpu.layers.adapter.attention import PallasBatchedRPAAttentionBackendImpl
from vllm_torchtpu.layers.adapter.linear_common import gelu_and_mul_quantize_op
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_orig_attention_init = Gemma4Attention.__init__


def down_proj_from_gate_up(
    mlp: Any,
    gate_up: torch.Tensor,
    *,
    act_fn: Any = None,
) -> torch.Tensor:
    """Run ``mlp.down_proj(act_fn(gate_up))[0]``, fusing GeGLU and FP8 quantization.

    When ``act_fn`` is ``GeluAndMul(approximate="tanh")`` and ``down_proj``
    uses per-channel W8A8 FP8 quantization on the current token bucket
    (``supports_quantized_input`` is ``True``) with lane-aligned intermediate
    width, ``gelu_and_mul_quantize_op`` produces the FP8 rows and scales in one
    pass. Small W8A16 decode buckets and unquantized layers fall back to
    ``down_proj(act_fn(gate_up))[0]``.
    """
    act = mlp.act_fn if act_fn is None else act_fn
    down = mlp.down_proj
    method = down.quant_method
    rows = gate_up.shape[0]
    if (
        isinstance(act, GeluAndMul)
        and act.approximate == "tanh"
        and isinstance(method, VllmFp8LinearMethodTPU)
        and down.bias is None
        and down.input_size_per_partition % COLUMN_ALIGNMENT == 0
        and down.weight.dtype == torch.float8_e4m3fn
        and gate_up.dtype == torch.bfloat16
        and method.supports_quantized_input(down, rows)
    ):
        x_q, x_scale = gelu_and_mul_quantize_op(gate_up)
        out = method.apply_quantized_input(down, x_q, x_scale)
        if down.reduce_results and down.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out
    return down(act(gate_up))[0]


def _patched_mlp_forward(self: Gemma4MLP, x: torch.Tensor) -> torch.Tensor:
    gate_up, _ = self.gate_up_proj(x)
    return down_proj_from_gate_up(self, gate_up)


def _o_proj_quant_dtype(o_proj: Any) -> torch.dtype | None:
    """Return the runtime FP8 weight dtype of ``o_proj`` when per-token W8A8."""
    method = o_proj.quant_method
    if (
        not isinstance(method, VllmFp8LinearMethodTPU)
        or method.online_fp8
        or method.block_quant
        or o_proj.bias is not None
        or not o_proj.input_is_parallel
    ):
        return None
    _, dtype, block_size, _ = method._linear_quant_config
    if block_size is not None or dtype not in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        return None
    return dtype


def _upstream_attention(
    attn: Any, positions: torch.Tensor, qkv: torch.Tensor
) -> torch.Tensor:
    """Run upstream Gemma 4 Q/K/V RMSNorm, rotary embedding, and attention."""
    q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    q = attn.q_norm(q.unflatten(-1, (attn.num_heads, attn.head_dim))).flatten(-2, -1)
    k = attn.k_norm(k.unflatten(-1, (attn.num_kv_heads, attn.head_dim)))
    q, k = attn.rotary_emb(positions, q, k.flatten(-2, -1))
    v = attn.v_norm(v.unflatten(-1, (attn.num_kv_heads, attn.head_dim)))
    return attn.attn(q, k, v.flatten(-2, -1))


def attention_from_qkv(
    attn: Any,
    positions: torch.Tensor,
    qkv: torch.Tensor,
    *,
    o_proj_rows: int | None = None,
) -> Any:
    """Run Gemma 4 attention from fused ``qkv_proj`` rows.

    Uses ``PallasBatchedRPAAttentionBackendImpl.forward_from_qkv`` when the
    layer's QKV prologue was enabled at construction time, and emits quantized
    attention output rows whenever ``o_proj`` accepts quantized inputs for
    ``o_proj_rows`` (defaulting to ``qkv.shape[0]``).
    """
    impl = attn.attn.impl
    if (
        not isinstance(impl, PallasBatchedRPAAttentionBackendImpl)
        or impl.qkv_prologue_eps is None
    ):
        return _upstream_attention(attn, positions, qkv)

    rope = attn.rotary_emb
    cos_sin = rope._match_cos_sin_cache_dtype(qkv).index_select(0, positions.flatten())
    cos, sin = (half.contiguous() for half in cos_sin.chunk(2, dim=-1))
    attn_metadata, layer, kv_cache, _ = get_attention_context(attn.attn.layer_name)

    target_rows = qkv.shape[0] if o_proj_rows is None else o_proj_rows
    o_proj = attn.o_proj
    method = o_proj.quant_method
    quantize_output = (
        impl.qkv_output_quant_dtype is not None
        and isinstance(method, VllmFp8LinearMethodTPU)
        and method.supports_quantized_input(o_proj, target_rows)
    )
    return impl.forward_from_qkv(
        layer,
        qkv,
        cos,
        sin,
        attn.q_norm.weight,
        attn.k_norm.weight,
        attn.q_norm.variance_epsilon,
        kv_cache,
        attn_metadata,
        quantize_output=quantize_output,
    )


def o_projection(attn: Any, out: Any) -> torch.Tensor:
    """Apply ``attn.o_proj`` to ``attention_from_qkv`` output rows."""
    o_proj = attn.o_proj
    if not isinstance(out, tuple):
        return o_proj(out)[0]
    x_q, x_scale = out
    method = o_proj.quant_method
    y = method.apply_quantized_input(o_proj, x_q, x_scale)
    if o_proj.reduce_results and o_proj.tp_size > 1:
        y = tensor_model_parallel_all_reduce(y)
    return y


def _patched_attention_init(self: Gemma4Attention, *args: Any, **kwargs: Any) -> None:
    _orig_attention_init(self, *args, **kwargs)
    if self.is_kv_shared_layer or not self.rotary_emb.is_neox_style:
        return
    kv_layout_name = (
        get_current_vllm_config().cache_config.get_resolved_kv_cache_layout()
    )
    if kv_layout_name == "HND":
        logger.info_once(
            "Skipping Gemma 4 TP QKV attention prologue under HND KV cache layout."
        )
        return
    impl = self.attn.impl
    if not isinstance(impl, PallasBatchedRPAAttentionBackendImpl):
        return
    unsupported = impl.qkv_prologue_unsupported()
    if unsupported:
        logger.info_once(
            "Skipping Gemma 4 QKV attention prologue on %s: %s.",
            self.attn.layer_name,
            ", ".join(unsupported),
        )
        return
    impl.qkv_prologue_eps = self.q_norm.variance_epsilon
    impl.qkv_v_rms_norm = True
    impl.qkv_output_quant_dtype = _o_proj_quant_dtype(self.o_proj)


def _patched_attention_forward(
    self: Gemma4Attention,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    if self.is_kv_shared_layer:
        q, _ = self.q_proj(hidden_states)
        q = self.q_norm(q.unflatten(-1, (self.num_heads, self.head_dim))).flatten(
            -2, -1
        )
        q, _ = self.rotary_emb(positions, q, None)
        return self.o_proj(self.attn(q, None, None))[0]
    qkv, _ = self.qkv_proj(hidden_states)
    return o_projection(self, attention_from_qkv(self, positions, qkv))


def apply_gemma4_layer_patches() -> None:
    """Install idempotent Gemma 4 MLP and attention forward patches."""
    if not getattr(Gemma4MLP, "_TPU_PATCHED", False):
        Gemma4MLP.forward = _patched_mlp_forward
        Gemma4MLP._TPU_PATCHED = True
    if not getattr(Gemma4Attention, "_TPU_PATCHED", False):
        Gemma4Attention.__init__ = _patched_attention_init
        Gemma4Attention.forward = _patched_attention_forward
        Gemma4Attention._TPU_PATCHED = True
