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
"""Gemma 4: sequence parallelism for dense tensor-parallel decoder layers.

When `use_sequence_parallel` admits a padded token bucket (`>=
MIN_TOKENS_PER_RANK` tokens per TP rank, TP-divisible, during backbone warmup),
each sublayer reduce-scatters its `o_proj` / `down_proj` partial sum along the
token axis, runs RMSNorms, residual adds, and FP8 activation quantization on
`num_tokens // tp_size` tokens per rank, and all-gathers 1-byte FP8 rows plus
f32 row scales into `qkv_proj` / `gate_up_proj`. Layer 0 slices its token shard
from the replicated embedding output and the final layer all-gathers the
residual back to `[num_tokens, H]`.

When `TPU_DENSE_SP_HIER_RS=bf16|fp8` is set on a 4- or 8-TensorCore tpu7x TP
group, eligible buckets fuse each reduce-scatter and its residual/RMSNorm/FP8
epilogue into a single hierarchical Pallas kernel call.

`_attention_partial_sum`, `_sequence_parallel_forward`, and
`_fused_sequence_parallel_forward` are adapted from upstream vLLM
`vllm/model_executor/models/gemma4.py` (`Gemma4Attention` and
`Gemma4DecoderLayer` at `v0.33.0`) with these changes:
- `o_proj` and `down_proj` keep their upstream `reduce_results` so small
  buckets delegate directly to upstream `Gemma4DecoderLayer.forward`, while
  the sequence-parallel forwards invoke `quant_method.apply(..., None)`
  followed by `reduce_scatter` (or the fused hierarchical kernel).
- `qkv_proj` and `gate_up_proj` consume pre-quantized FP8 rows gathered across
  TP ranks (`k_eq_v` global-attention layers load `K` weights into both `K` and
  `V` slots of `QKVParallelLinear` upstream, so `[q_size, kv_size, kv_size]`
  holds).
- Layer 0 applies `replicated_tokens` (all-reduce) or `slice_token_shard`
  (sequence-parallel) using a per-rank int32 buffer `_tpu_sp_tp_rank` so all
  paths lift identical Dynamo symbolic inputs and all TP ranks trace one graph
  fingerprint for peer replay, and `_fused_sequence_parallel_forward` carries
  each non-final layer's un-reduced `down_proj` tail via `Gemma4Model`'s
  `residual` slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
from vllm.config import get_current_vllm_config_or_none
from vllm.distributed import tensor_model_parallel_reduce_scatter
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.gemma4 import (
    Gemma4Attention,
    Gemma4DecoderLayer,
    Gemma4MLP,
)

from vllm_torchtpu import dense_sequence_parallel_flag, envs
from vllm_torchtpu.layers.adapter import dense_sequence_parallel
from vllm_torchtpu.layers.adapter.custom_ops import (
    fp8_linear_op,
    fused_rs_epilogue_op,
)
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

logger = init_logger(__name__)

_orig_layer_init = Gemma4DecoderLayer.__init__
_orig_layer_forward = Gemma4DecoderLayer.forward
_CarriedSublayerTail = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class _DenseSpLayerState:
    streaming_collectives: fused_rs_epilogue_op.FusedSublayerCollectives | None = None


def _validate_layer_norms(layer: Gemma4DecoderLayer) -> None:
    norms = (
        layer.input_layernorm,
        layer.post_attention_layernorm,
        layer.pre_feedforward_layernorm,
        layer.post_feedforward_layernorm,
    )
    for norm in norms:
        if not isinstance(norm, RMSNorm):
            raise NotImplementedError(
                "tpu_dense_sequence_parallel requires vLLM RMSNorm decoder norms, "
                f"got {type(norm).__name__}"
            )
    epsilons = {norm.variance_epsilon for norm in norms}
    if len(epsilons) != 1:
        raise NotImplementedError(
            "tpu_dense_sequence_parallel requires matching variance_epsilon across "
            f"decoder norms, got {sorted(epsilons)}"
        )


def _configure_layer(
    layer: Gemma4DecoderLayer, *, num_hidden_layers: int | None = None
) -> None:
    if layer.mlp.down_proj.tp_size <= 1:
        raise ValueError("tpu_dense_sequence_parallel requires TP > 1")
    if (
        layer.enable_moe_block
        or layer.self_attn.is_kv_shared_layer
        or layer.per_layer_input_gate is not None
    ):
        # TODO: support MoE, KV-shared (q_proj-only), and per-layer-embedding layers.
        raise NotImplementedError(
            "tpu_dense_sequence_parallel supports dense Gemma 4 layers without KV "
            "sharing or per-layer embeddings"
        )
    for proj in (layer.self_attn.o_proj, layer.mlp.down_proj):
        if proj.bias is not None:
            raise NotImplementedError(
                "tpu_dense_sequence_parallel requires bias-free o_proj and down_proj"
            )
        if not proj.input_is_parallel or proj.quant_method is None:
            raise NotImplementedError(
                "tpu_dense_sequence_parallel requires row-parallel o_proj and "
                "down_proj with input_is_parallel=True and a quant_method"
            )
    dense_sequence_parallel.validate_fp8_linear_for_sp(layer.self_attn.qkv_proj)
    dense_sequence_parallel.validate_fp8_linear_for_sp(layer.mlp.gate_up_proj)
    _validate_layer_norms(layer)
    fp8_linear_op.register_fp8_linear_ops()
    if layer.layer_idx == 0:
        layer.register_buffer(
            "_tpu_sp_tp_rank",
            torch.tensor(
                [layer.self_attn.o_proj.tp_rank],
                dtype=torch.int32,
                device=layer.layer_scalar.device,
            ),
            persistent=False,
        )
    payload = envs.TPU_DENSE_SP_HIER_RS
    streaming_collectives = None
    if payload != "off":
        hidden = layer.input_layernorm.weight.shape[0]
        if fused_rs_epilogue_op.supports_streaming_sublayer(
            fused_rs_epilogue_op.MIN_STREAMING_RS_TOKENS, hidden
        ):
            streaming_collectives = (
                fused_rs_epilogue_op.build_streaming_sublayer_collectives(
                    layer.input_layernorm.variance_epsilon,
                    fp8_wire=payload == "fp8",
                )
            )
        else:
            logger.warning_once(
                "TPU_DENSE_SP_HIER_RS=%s is set, but hidden=%d does not fit "
                "the streaming epilogue; keeping XLA reduce-scatter.",
                payload,
                hidden,
            )
    cast("Any", layer)._tpu_dense_sp_num_layers = (
        num_hidden_layers
        if num_hidden_layers is not None
        else layer.self_attn.config.num_hidden_layers
    )
    cast("Any", layer)._tpu_dense_sp = _DenseSpLayerState(
        streaming_collectives=streaming_collectives
    )


def _layer_state(layer: Gemma4DecoderLayer) -> _DenseSpLayerState:
    return cast("_DenseSpLayerState", cast("Any", layer)._tpu_dense_sp)


def _is_last_layer(layer: Gemma4DecoderLayer) -> bool:
    num_layers = cast("int", cast("Any", layer)._tpu_dense_sp_num_layers)
    return layer.layer_idx == num_layers - 1


def _patched_layer_init(
    self: Gemma4DecoderLayer,
    config: Any,
    cache_config: Any = None,
    quant_config: Any = None,
    prefix: str = "",
) -> None:
    _orig_layer_init(
        self,
        config,
        cache_config=cache_config,
        quant_config=quant_config,
        prefix=prefix,
    )
    _configure_layer(self, num_hidden_layers=config.num_hidden_layers)


def _attention_partial_sum(
    attn: Gemma4Attention, positions: torch.Tensor, qkv: torch.Tensor
) -> torch.Tensor:
    # Copied from vLLM Gemma4Attention.forward (v0.33.0, non-KV-shared branch),
    # taking pre-projected `qkv` and returning unreduced `o_proj` output.
    q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)

    q = q.unflatten(-1, (attn.num_heads, attn.head_dim))
    q = attn.q_norm(q)
    q = q.flatten(-2, -1)

    k = k.unflatten(-1, (attn.num_kv_heads, attn.head_dim))
    k = attn.k_norm(k)
    k = k.flatten(-2, -1)
    q, k = attn.rotary_emb(positions, q, k)

    v = v.unflatten(-1, (attn.num_kv_heads, attn.head_dim))
    v = attn.v_norm(v)
    v = v.flatten(-2, -1)
    attn_output = attn.attn(q, k, v)

    assert attn.o_proj.quant_method is not None
    return attn.o_proj.quant_method.apply(attn.o_proj, attn_output, None)


def _mlp_partial_sum(mlp: Gemma4MLP, gate_up: torch.Tensor) -> torch.Tensor:
    assert mlp.down_proj.quant_method is not None
    return mlp.down_proj.quant_method.apply(mlp.down_proj, mlp.act_fn(gate_up), None)


def _sequence_parallel_forward(
    layer: Gemma4DecoderLayer, positions: torch.Tensor, hidden_states: torch.Tensor
) -> torch.Tensor:
    num_tokens = positions.shape[0]
    attn = layer.self_attn
    if layer.layer_idx == 0:
        hidden_states = dense_sequence_parallel.slice_token_shard(
            hidden_states,
            num_tokens,
            cast("torch.Tensor", cast("Any", layer)._tpu_sp_tp_rank),
            attn.o_proj.tp_size,
        )
    residual = hidden_states
    qkv = dense_sequence_parallel.quantize_all_gather_linear(
        attn.qkv_proj, layer.input_layernorm(residual), num_tokens
    )
    hidden_states = tensor_model_parallel_reduce_scatter(
        _attention_partial_sum(attn, positions, qkv), 0
    )
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = hidden_states + residual
    residual = hidden_states

    gate_up = dense_sequence_parallel.quantize_all_gather_linear(
        layer.mlp.gate_up_proj,
        layer.pre_feedforward_layernorm(hidden_states),
        num_tokens,
    )
    hidden_states = tensor_model_parallel_reduce_scatter(
        _mlp_partial_sum(layer.mlp, gate_up), 0
    )
    hidden_states = layer.post_feedforward_layernorm(hidden_states)
    hidden_states = hidden_states + residual
    hidden_states = hidden_states * layer.layer_scalar
    if _is_last_layer(layer):
        hidden_states = dense_sequence_parallel.all_gather_tokens(
            hidden_states, num_tokens
        )
    return hidden_states


def _fused_sequence_parallel_forward(
    layer: Gemma4DecoderLayer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    carried: _CarriedSublayerTail | None,
    collectives: fused_rs_epilogue_op.FusedSublayerCollectives,
) -> tuple[torch.Tensor, _CarriedSublayerTail | None]:
    """Sequence-parallel forward fusing each sublayer's RS + residual epilogue.

    Because `down_proj`'s fused epilogue consumes the next layer's
    `input_layernorm.weight`, non-final layers return the un-reduced `down_proj`
    partial sum plus `carried = (residual, post_feedforward_layernorm.weight,
    layer_scalar)` via `Gemma4Model`'s `residual` carry slot.
    """
    num_tokens = positions.shape[0]
    attn = layer.self_attn
    if carried is None:
        residual = dense_sequence_parallel.slice_token_shard(
            hidden_states,
            num_tokens,
            cast("torch.Tensor", cast("Any", layer)._tpu_sp_tp_rank),
            attn.o_proj.tp_size,
        )
        qkv = dense_sequence_parallel.quantize_all_gather_linear(
            attn.qkv_proj, layer.input_layernorm(residual), num_tokens
        )
    else:
        residual, post_norm_weight, scale = carried
        residual, x_q, x_scale = collectives.norm_add_scale_next_norm(
            hidden_states,
            residual,
            post_norm_weight,
            scale,
            layer.input_layernorm.weight,
        )
        qkv = dense_sequence_parallel.all_gather_prequantized_fp8_linear(
            attn.qkv_proj, x_q, x_scale, num_tokens
        )

    residual, x_q, x_scale = collectives.norm_add_next_norm(
        _attention_partial_sum(attn, positions, qkv),
        residual,
        layer.post_attention_layernorm.weight,
        layer.pre_feedforward_layernorm.weight,
    )
    gate_up = dense_sequence_parallel.all_gather_prequantized_fp8_linear(
        layer.mlp.gate_up_proj, x_q, x_scale, num_tokens
    )
    mlp_partial = _mlp_partial_sum(layer.mlp, gate_up)
    post_norm_weight = layer.post_feedforward_layernorm.weight
    if not _is_last_layer(layer):
        return mlp_partial, (residual, post_norm_weight, layer.layer_scalar)
    residual = collectives.norm_add_scale(
        mlp_partial, residual, post_norm_weight, layer.layer_scalar
    )
    return dense_sequence_parallel.all_gather_tokens(residual, num_tokens), None


def _require_no_carried_tail(residual: _CarriedSublayerTail | None) -> None:
    if residual is not None:
        raise NotImplementedError(
            "tpu_dense_sequence_parallel cannot carry a fused tail into a "
            "layer without matching fused collectives"
        )


def _dense_sequence_parallel_forward(
    self: Gemma4DecoderLayer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: _CarriedSublayerTail | None,
    per_layer_input: torch.Tensor | None = None,
    **kwargs: object,
) -> tuple[torch.Tensor, _CarriedSublayerTail | torch.Tensor | None]:
    num_tokens = positions.shape[0]
    if dense_sequence_parallel.use_sequence_parallel(
        num_tokens, self.mlp.down_proj.tp_size
    ):
        state = _layer_state(self)
        if (
            state.streaming_collectives is not None
            and fused_rs_epilogue_op.supports_streaming_sublayer(
                num_tokens, hidden_states.shape[-1]
            )
        ):
            return _fused_sequence_parallel_forward(
                self,
                positions,
                hidden_states,
                residual,
                state.streaming_collectives,
            )
        _require_no_carried_tail(residual)
        return _sequence_parallel_forward(self, positions, hidden_states), None
    _require_no_carried_tail(residual)
    if self.layer_idx == 0:
        hidden_states = dense_sequence_parallel.replicated_tokens(
            hidden_states,
            num_tokens,
            cast("torch.Tensor", cast("Any", self)._tpu_sp_tp_rank),
        )
    return _orig_layer_forward(
        self,
        positions,
        hidden_states,
        None,
        per_layer_input=per_layer_input,
        **kwargs,
    )


def maybe_patch_gemma4_dense_sequence_parallel(
    model_config: ModelConfig | None,
    vllm_config: VllmConfig | None = None,
) -> bool | None:
    """Patches `Gemma4DecoderLayer` only when `tpu_dense_sequence_parallel` is on.

    Returns `False` when the flag is off or the config is missing / non-Gemma 4,
    leaving `Gemma4DecoderLayer` untouched for default runs and leaving the
    registry hook eligible if an opted-in Gemma 4 model loads later.
    """
    if vllm_config is None:
        vllm_config = get_current_vllm_config_or_none()
    enabled = dense_sequence_parallel_flag.dense_sequence_parallel_enabled(vllm_config)
    dense_sequence_parallel.validate_sequence_parallel_env(enabled)
    if not enabled:
        return False
    if model_config is None:
        return False
    if model_config.hf_text_config.model_type != "gemma4_text":
        return False
    Gemma4DecoderLayer.__init__ = _patched_layer_init
    # Upstream annotates return as tuple[Tensor, Tensor] while returning None.
    Gemma4DecoderLayer.forward = cast("Any", _dense_sequence_parallel_forward)
    logger.info("Applied TPU patch: Gemma 4 dense sequence parallelism.")
    return None
