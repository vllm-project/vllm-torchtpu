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
"""Gemma4 decode buckets: the post-attention block as one weight stream
(TPU_GEMMA4_DECODE_FUSED_MLP_MAX_ROWS).

On a token bucket of at most that many rows, a dense Gemma4 decoder layer runs
everything between its two all-reduces (post-attention norm, residual add,
pre-feedforward norm, the W8A8 gate/up projection, GeGLU and the W8A8 down
projection) as `linear_common.decode_geglu_mlp`, one Pallas kernel
(kernels/quantized_matmul/decode_mlp.py). Other buckets, and layers the kernel
cannot take, keep the layer's forward. It is a model patch because upstream
exposes nothing between the attention's all-reduce and the MLP's.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.models.gemma4 import Gemma4DecoderLayer

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.kernels.quantized_matmul import decode_mlp
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    decode_geglu_mlp,
)
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

# TPU_GEMMA4_DECODE_FUSED_MLP_MAX_ROWS and the init and forward it wraps (the
# earlier Gemma4 patches' or upstream's), set when the patch is applied.
_max_rows = 0
_inner_init: Callable[..., None] | None = None
_inner_forward: Callable[..., Any] | None = None


def _refusal(layer: Gemma4DecoderLayer) -> str | None:
    """Why the kernel cannot run this layer's post-attention block, if so."""
    mlp: Any = layer.mlp
    if layer.enable_moe_block:
        return "MoE layers"
    if layer.per_layer_input_gate is not None:
        return "per-layer embeddings"
    act = mlp.act_fn
    if not isinstance(act, GeluAndMul) or act.approximate != "tanh":
        return f"activation {type(act).__name__}"
    for proj in (mlp.gate_up_proj, mlp.down_proj):
        if not isinstance(proj.quant_method, VllmFp8LinearMethodTPU):
            return f"{type(proj.quant_method).__name__} projections"
        if proj.bias is not None or getattr(proj, KEEP_VLLM_LAYOUT_ATTR, False):
            return "projection biases or vLLM-layout weights"
    norms = (layer.post_attention_layernorm, layer.pre_feedforward_layernorm)
    if any(not n.has_weight for n in norms):
        return "weightless norms"
    if norms[0].variance_epsilon != norms[1].variance_epsilon:
        return "norms with different epsilons"
    return None


def _fits(layer: Gemma4DecoderLayer, rows: Any) -> bool:
    """Whether the kernel takes this layer's weights, in their runtime layout:
    FP8 W8A8 projections that quantize these rows per token (not W8A16), and
    VMEM for every bucket up to the threshold (it grows with the rows)."""
    mlp: Any = layer.mlp
    gate_up, down = mlp.gate_up_proj, mlp.down_proj
    hidden, inter = down.weight.shape[1], down.weight.shape[0]
    return (
        all(
            p.quant_method.supports_quantized_input(p, rows)
            and p.weight.dtype.is_floating_point
            for p in (gate_up, down)
        )
        and gate_up.weight.shape == (hidden, 2 * inter)
        and decode_mlp.supports(_max_rows, hidden, inter)
    )


def _decode_bucket(rows: Any) -> bool:
    """Whether a bucket of `rows` tokens takes the fused path.

    vLLM traces the backbone with the token count symbolic; the comparison
    leaves a shape guard, and only the backbone warmup gives each side of it
    its own trace (compilation/shape_variants.py). A graph compiled outside it
    keeps the layer's forward.
    """
    if torch.compiler.is_compiling() and shape_variants.aot_cache_tag() is None:
        return False
    return rows % 16 == 0 and 0 < rows <= _max_rows


def _decode_forward(
    self: Gemma4DecoderLayer,
    positions: torch.Tensor,
    hidden_states: Any,
    residual: torch.Tensor | None,
    per_layer_input: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[Any, None]:
    assert _inner_forward is not None
    rows = hidden_states.shape[0] if isinstance(hidden_states, torch.Tensor) else 0
    if (
        kwargs.get("sequence_parallel")
        or self._decode_mlp_refusal is not None
        or not _decode_bucket(rows)
        or not _fits(self, rows)
    ):
        return _inner_forward(
            self, positions, hidden_states, residual, per_layer_input, **kwargs
        )
    kwargs.pop("sequence_parallel", None)
    # Gemma4 carries the residual inside hidden_states (residual is None).
    del residual, per_layer_input
    attn: Any = self.self_attn
    mlp: Any = self.mlp
    residual = hidden_states
    hidden_states = attn(
        positions=positions,
        hidden_states=self.input_layernorm(residual),
        **kwargs,
    )
    if not attn.o_proj.reduce_results and attn.o_proj.tp_size > 1:
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)
    gate_up, down = mlp.gate_up_proj, mlp.down_proj
    hidden_states, residual = decode_geglu_mlp(
        hidden_states,
        residual,
        self.post_attention_layernorm.weight,
        self.pre_feedforward_layernorm.weight,
        gate_up.weight,
        gate_up.weight_scale,
        down.weight,
        down.weight_scale,
        self.pre_feedforward_layernorm.variance_epsilon,
    )
    if down.tp_size > 1:
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)
    hidden_states = self.post_feedforward_layernorm(hidden_states) + residual
    return hidden_states * self.layer_scalar, None


def _patched_layer_init(self: Gemma4DecoderLayer, *args: Any, **kwargs: Any) -> None:
    assert _inner_init is not None
    _inner_init(self, *args, **kwargs)
    self._decode_mlp_refusal = _refusal(self)
    if self._decode_mlp_refusal is not None:
        logger.info_once(
            "TPU_GEMMA4_DECODE_FUSED_MLP_MAX_ROWS: %s keep the layer's forward.",
            self._decode_mlp_refusal,
        )


def maybe_patch_gemma4_decode_mlp(model_config: ModelConfig | None) -> bool | None:
    """Returns False until a Gemma4 text model config is available."""
    if not envs.TPU_GEMMA4_DECODE_FUSED_MLP_MAX_ROWS:
        return None
    if model_config is None or model_config.hf_text_config.model_type != "gemma4_text":
        return False
    global _max_rows, _inner_init, _inner_forward
    _max_rows = envs.TPU_GEMMA4_DECODE_FUSED_MLP_MAX_ROWS
    if _max_rows % 16:
        raise ValueError(
            "TPU_GEMMA4_DECODE_FUSED_MLP_MAX_ROWS must be a multiple of 16, got "
            f"{_max_rows}"
        )
    _inner_init = Gemma4DecoderLayer.__init__
    _inner_forward = Gemma4DecoderLayer.forward
    Gemma4DecoderLayer.__init__ = _patched_layer_init
    Gemma4DecoderLayer.forward = _decode_forward
    logger.info(
        "Applied TPU patch: Gemma4 decode buckets of at most %d rows run the "
        "post-attention block (norms, GeGLU MLP, FP8 projections) as one "
        "Pallas weight stream.",
        _max_rows,
    )
    return None
