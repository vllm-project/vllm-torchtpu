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
"""Sequence-parallel residual stream and fused RMSNorm+FP8 tails for Gemma 4."""

from __future__ import annotations

from itertools import islice
from typing import Any

import torch
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.gemma4.attention import (
    attention_from_qkv,
    has_active_mm_prefix,
    o_projection,
)
from vllm_torchtpu.models.vllm.gemma4.layers import (
    all_gather_rows,
    configure_layer_fused_moe_head,
    configure_layer_fused_norm_quant,
    decoder_feedforward_tail,
    model_supports_multimodal,
    prequantized_linear,
    reduce_scatter_rows,
    router_weight,
)

__all__ = [
    "MIN_SP_SHARD_BYTES",
    "all_gather_rows",
    "apply_gemma4_sequence_parallel_patch",
    "configure_layer_fused_moe_head",
    "configure_layer_fused_norm_quant",
    "reduce_scatter_rows",
    "router_weight",
    "should_enable_sequence_parallel",
    "use_sequence_parallel",
]

logger = init_logger(__name__)

MIN_SP_SHARD_BYTES = 1 << 20


def use_sequence_parallel(num_tokens: int, row_bytes: int, tp_size: int) -> bool:
    """Return True when ``num_tokens`` splits evenly into >= 1 MiB TP shards."""
    if tp_size <= 1:
        return False
    if torch.compiler.is_compiling() and shape_variants.aot_cache_tag() is None:
        return False
    return (
        num_tokens % tp_size == 0
        and (num_tokens // tp_size) * row_bytes >= MIN_SP_SHARD_BYTES
    )


def _layer_supports_sp(layer: Any) -> bool:
    attn = layer.self_attn
    mlp = layer.mlp
    if attn.is_kv_shared_layer or layer.per_layer_input_gate is not None:
        return False
    projections = (attn.qkv_proj, attn.o_proj, mlp.gate_up_proj, mlp.down_proj)
    if any(p.bias is not None for p in projections) or attn.o_proj.tp_size <= 1:
        return False
    if layer.enable_moe_block:
        moe_cfg = layer.experts.moe_config
        if moe_cfg.dp_size > 1 or moe_cfg.is_sequence_parallel:
            return False
    return True


def should_enable_sequence_parallel(model: Any, vllm_config: Any) -> bool:
    """Return True when ``model`` can safely run sequence-parallel prefill."""
    if (
        vllm_config is None
        or model_supports_multimodal(vllm_config.model_config)
        or get_tensor_model_parallel_world_size() <= 1
        or get_pp_group().world_size > 1
        or model.fast_prefill_enabled
        or envs.TPU_PRECOMPILE_PEER_REPLAY
        or vllm_config.speculative_config is not None
    ):
        return False
    layers = list(islice(model.layers, model.start_layer, model.end_layer))
    return bool(layers) and all(_layer_supports_sp(layer) for layer in layers)


def _gathered_projection(linear: Any, x_local: torch.Tensor) -> torch.Tensor:
    method = linear.quant_method
    rows = x_local.shape[0] * linear.tp_size
    if isinstance(method, VllmFp8LinearMethodTPU) and method.supports_quantized_input(
        linear, rows
    ):
        x_q, x_scale = method.quantize_input(linear, x_local)
        rows_q = all_gather_rows(x_q.view(torch.int8)).view(x_q.dtype)
        return method.apply_quantized_input(linear, rows_q, all_gather_rows(x_scale))
    out, _ = linear(all_gather_rows(x_local))
    return out


def _gathered_prequantized(
    linear: Any, x_bytes: torch.Tensor, x_scale: torch.Tensor
) -> torch.Tensor:
    return prequantized_linear(
        linear, all_gather_rows(x_bytes), all_gather_rows(x_scale)
    )


def _sp_layer_forward(
    self: Any, positions: torch.Tensor, state: Any, next_input_norm: Any, *, fused: bool
) -> Any:
    attn = self.self_attn
    if isinstance(state, tuple):
        residual, q, s = state
        qkv = _gathered_prequantized(attn.qkv_proj, q, s)
    else:
        residual = state
        qkv = _gathered_projection(attn.qkv_proj, self.input_layernorm(residual))
    o_rows = reduce_scatter_rows(
        o_projection(attn, attention_from_qkv(attn, positions, qkv))
    )
    return decoder_feedforward_tail(
        self,
        o_rows,
        residual,
        project_gate_up=_gathered_projection,
        project_prequant_gate_up=_gathered_prequantized,
        reduce_rows=reduce_scatter_rows,
        gather_moe_inputs=True,
        fused=fused,
        next_input_norm=next_input_norm,
    )


def _patched_layer_forward(
    self: Any,
    positions: torch.Tensor,
    hidden_states: Any,
    residual: torch.Tensor | None = None,
    per_layer_input: torch.Tensor | None = None,
    sequence_parallel: bool = False,
    next_input_norm: Any = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    if not getattr(self, "_tpu_sp_configured", False) or per_layer_input is not None:
        return type(self)._tpu_orig_forward(
            self, positions, hidden_states, residual, per_layer_input, **kwargs
        )
    del residual
    if sequence_parallel:
        fused = getattr(self, "fused_norm_quant", None) is not None
        return (
            _sp_layer_forward(
                self, positions, hidden_states, next_input_norm, fused=fused
            ),
            None,
        )
    attn_out = tensor_model_parallel_all_reduce(
        self.self_attn(
            positions=positions,
            hidden_states=self.input_layernorm(hidden_states),
            **kwargs,
        )
    )
    return (
        decoder_feedforward_tail(
            self, attn_out, hidden_states, reduce_rows=tensor_model_parallel_all_reduce
        ),
        None,
    )


def _patched_model_init(self: Any, *args: Any, **kwargs: Any) -> None:
    type(self)._tpu_orig_init(self, *args, **kwargs)
    self._tpu_sp_enabled = False
    self._tpu_sp_fused_norms = False
    if not should_enable_sequence_parallel(self, getattr(self, "vllm_config", None)):
        return
    layers = list(islice(self.layers, self.start_layer, self.end_layer))
    fused_all = True
    for layer in layers:
        layer.self_attn.o_proj.reduce_results = False
        layer.mlp.down_proj.reduce_results = False
        if layer.enable_moe_block:
            layer.experts.moe_config.skip_final_all_reduce = True
        layer._tpu_sp_configured = True
        if not configure_layer_fused_norm_quant(layer):
            fused_all = False
    if not fused_all:
        for layer in layers:
            layer.fused_norm_quant = None
            layer.fused_moe_tail = None
    self._tpu_sp_enabled = True
    self._tpu_sp_fused_norms = fused_all
    logger.info(
        "Gemma 4 sequence-parallel residual stream enabled (min shard %d bytes, "
        "fused RMSNorm+FP8 tails: %s).",
        MIN_SP_SHARD_BYTES,
        fused_all,
    )


def _model_has_active_mm_prefix(model: Any) -> bool:
    first = next(islice(model.layers, model.start_layer, model.end_layer), None)
    return first is not None and has_active_mm_prefix(first.self_attn)


def _patched_model_forward(
    self: Any,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: Any = None,
    inputs_embeds: torch.Tensor | None = None,
    per_layer_inputs: torch.Tensor | None = None,
    **kwargs: Any,
) -> Any:
    if (
        not getattr(self, "_tpu_sp_enabled", False)
        or input_ids is None
        or inputs_embeds is not None
        or per_layer_inputs is not None
        or intermediate_tensors is not None
        or getattr(self, "aux_hidden_state_layers", ())
        or kwargs
        or _model_has_active_mm_prefix(self)
    ):
        return type(self)._tpu_orig_forward(
            self,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            per_layer_inputs,
            **kwargs,
        )
    num_tokens = positions.shape[-1]
    torch._check(input_ids.shape[0] == num_tokens)
    tp_size = get_tensor_model_parallel_world_size()
    row_bytes = self.config.hidden_size * self.normalizer.dtype.itemsize
    if not use_sequence_parallel(num_tokens, row_bytes, tp_size):
        return type(self)._tpu_orig_forward(
            self,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            per_layer_inputs,
        )
    state: Any = sequence_parallel_chunk(self.embed_input_ids(input_ids))
    layers = list(islice(self.layers, self.start_layer, self.end_layer))
    fused = self._tpu_sp_fused_norms
    for i, layer in enumerate(layers):
        next_norm = (
            layers[i + 1].input_layernorm if fused and i + 1 < len(layers) else None
        )
        state, _ = layer(
            positions, state, None, sequence_parallel=True, next_input_norm=next_norm
        )
    return all_gather_rows(self.norm(state))


def apply_gemma4_sequence_parallel_patch(gemma4_mod: Any) -> bool:
    """Install sequence-parallel hooks on ``Gemma4DecoderLayer`` and ``Gemma4Model``."""
    layer_cls = gemma4_mod.Gemma4DecoderLayer
    model_cls = gemma4_mod.Gemma4Model
    if getattr(model_cls, "_TPU_GEMMA4_SP_PATCHED", False):
        return False
    layer_cls._tpu_orig_forward = layer_cls.forward
    layer_cls.forward = _patched_layer_forward
    model_cls._tpu_orig_init = model_cls.__init__
    model_cls._tpu_orig_forward = model_cls.forward
    model_cls.__init__ = _patched_model_init
    model_cls.forward = _patched_model_forward
    layer_cls._TPU_GEMMA4_SP_PATCHED = True
    model_cls._TPU_GEMMA4_SP_PATCHED = True
    return True
