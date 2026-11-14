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
"""Gemma 4: token-sharded residual stream under tensor parallelism.

Upstream keeps the full residual ``[tokens, hidden]`` on every TP rank and
all-reduces the ``o_proj`` and ``down_proj`` partial sums twice per layer.
With ``TPU_SHARDED_RESIDUAL`` set (`"fp8"` or `"bf16"`), padded token buckets
at or above ``_MIN_SHARDED_TOKENS`` shard the residual along the token axis:

* ``o_proj`` and ``down_proj`` partial sums are reduce-scattered across TP
  ranks so each rank keeps ``tokens // tp`` rows;
* RMSNorms, residual adds, per-layer embeddings, and ``layer_scalar`` run on
  those local rows only;
* normed rows are all-gathered right before ``qkv_proj`` and ``gate_up_proj``.
  In ``bf16`` mode the gather moves bf16 activations; in ``fp8`` mode each rank
  quantizes its local rows first and all-gathers 1-byte FP8 rows plus per-row
  scales for ``prequantized_linear``.

Layer 0 slices this rank's rows from the replicated embedding output and the
final layer all-gathers the residual back to ``[tokens, hidden]``. Smaller or
unaligned buckets stay on the replicated all-reduce path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.distributed.dcp import get_dcp_world_size
from vllm_torchtpu.distributed.pcp import get_pcp_world_size
from vllm_torchtpu.layers.adapter import prequantized_linear as pq
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

MODE_BF16 = "bf16"
MODE_FP8 = "fp8"
_VALID_MODES = (MODE_BF16, MODE_FP8)
_PATCHED_ATTR = "_tpu_sharded_residual_patched"

# Padded compile_sizes token bucket threshold: decode buckets (< 1024) keep the
# replicated all-reduce path; prefill and chunked-prefill buckets (>= 1024)
# shard the residual stream across TP ranks.
_MIN_SHARDED_TOKENS = 1024

_Gathered = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


def _get_pp_world_size() -> int:
    try:
        return int(get_pp_group().world_size)
    except AssertionError:
        return 1


def _all_gather_rows(x: torch.Tensor, num_tokens: Any) -> torch.Tensor:
    return tensor_model_parallel_all_gather(x, 0).narrow(0, 0, num_tokens)


def _reduce_scatter_rows(x: torch.Tensor) -> torch.Tensor:
    return tensor_model_parallel_reduce_scatter(x, 0)


def _own_rows(x: torch.Tensor, layer: Any, num_tokens: Any) -> torch.Tensor:
    rows = num_tokens // layer._tpu_sr_tp
    return x.narrow(0, layer._tpu_sr_rank * rows, rows)


def _gather_for_matmul(layer: Any, x_local: torch.Tensor, num_tokens: Any) -> _Gathered:
    if layer._tpu_sr_mode == MODE_FP8:
        x_q, x_scale = pq.quantize_rows(x_local)
        return (
            _all_gather_rows(x_q, num_tokens),
            _all_gather_rows(x_scale, num_tokens),
        )
    return _all_gather_rows(x_local, num_tokens)


def _column_linear(linear: Any, x: _Gathered) -> torch.Tensor:
    if isinstance(x, tuple):
        pq.check_prequantized_linear(linear)
        return pq.prequantized_linear(linear, x[0], x[1])
    out, _ = linear(x)
    return out


def _attention_partial(
    attn: Any, positions: torch.Tensor, x: _Gathered
) -> torch.Tensor:
    # Adapted from Gemma4Attention.forward in vllm/model_executor/models/gemma4.py
    # (v0.33.0): inlined so qkv_proj can consume pre-gathered (x_q, x_scale) and
    # o_proj returns this rank's partial sum (reduce_results=False).
    qkv = _column_linear(attn.qkv_proj, x)
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

    output, _ = attn.o_proj(attn_output)
    return output


def _mlp_partial(mlp: Any, x: _Gathered) -> torch.Tensor:
    # Adapted from Gemma4MLP.forward in vllm/model_executor/models/gemma4.py
    # (v0.33.0): inlined so gate_up_proj can consume pre-gathered (x_q, x_scale)
    # and down_proj returns this rank's partial sum (reduce_results=False).
    gate_up = _column_linear(mlp.gate_up_proj, x)
    out, _ = mlp.down_proj(mlp.act_fn(gate_up))
    return out


def _per_layer_embedding(
    layer: Any, hidden_states: torch.Tensor, per_layer_input: torch.Tensor | None
) -> torch.Tensor:
    # Adapted from Gemma4DecoderLayer.forward in vllm/model_executor/models/gemma4.py
    # (v0.33.0): runs identically on replicated or row-sharded hidden_states.
    if per_layer_input is None or layer.per_layer_input_gate is None:
        return hidden_states
    gate = layer.per_layer_input_gate(hidden_states)
    gate = torch.nn.functional.gelu(gate, approximate="tanh")
    contribution = layer.per_layer_projection(gate * per_layer_input)
    return hidden_states + layer.post_per_layer_input_norm(contribution)


def _use_sharded(layer: Any, num_tokens: Any) -> bool:
    min_rows = layer._tpu_sr_min_rows
    if min_rows <= 0:
        return bool(num_tokens % layer._tpu_sr_tp == 0)
    # Token-count branches receive separate traces only during backbone warmup
    # (see compilation/shape_variants.py); anything compiled outside warmup
    # stays on the replicated path.
    in_warmup = (
        not torch.compiler.is_compiling() or shape_variants.aot_cache_tag() is not None
    )
    return bool(
        in_warmup and num_tokens >= min_rows and num_tokens % layer._tpu_sr_tp == 0
    )


def _replicated_forward(
    layer: Any,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    per_layer_input: torch.Tensor | None,
) -> torch.Tensor:
    # Adapted from Gemma4DecoderLayer.forward in vllm/model_executor/models/gemma4.py
    # (v0.33.0): explicit all-reduces replace o_proj/down_proj's disabled ones.
    if layer._tpu_sr_first:
        # Lift positions.shape[0] at the same point as _sharded_forward so all
        # bucket variants share one FX placeholder signature under shape_variants.
        hidden_states = hidden_states.narrow(0, 0, positions.shape[0])
    residual = hidden_states
    hidden_states = layer.input_layernorm(residual)
    hidden_states = tensor_model_parallel_all_reduce(
        _attention_partial(layer.self_attn, positions, hidden_states)
    )
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = hidden_states + residual
    residual = hidden_states

    hidden_states = layer.pre_feedforward_layernorm(hidden_states)
    hidden_states = tensor_model_parallel_all_reduce(
        _mlp_partial(layer.mlp, hidden_states)
    )
    hidden_states = layer.post_feedforward_layernorm(hidden_states)
    hidden_states = hidden_states + residual

    hidden_states = _per_layer_embedding(layer, hidden_states, per_layer_input)
    return hidden_states * layer.layer_scalar


def _sharded_forward(
    layer: Any,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    per_layer_input: torch.Tensor | None,
) -> torch.Tensor:
    # Adapted from Gemma4DecoderLayer.forward in vllm/model_executor/models/gemma4.py
    # (v0.33.0): residual and norms stay sharded along tokens across TP ranks.
    num_tokens = positions.shape[0]
    if layer._tpu_sr_first:
        hidden_states = _own_rows(hidden_states, layer, num_tokens)
    residual = hidden_states

    hidden_states = layer.input_layernorm(residual)
    gathered = _gather_for_matmul(layer, hidden_states, num_tokens)
    hidden_states = _reduce_scatter_rows(
        _attention_partial(layer.self_attn, positions, gathered)
    )
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = hidden_states + residual
    residual = hidden_states

    hidden_states = layer.pre_feedforward_layernorm(hidden_states)
    gathered = _gather_for_matmul(layer, hidden_states, num_tokens)
    hidden_states = _reduce_scatter_rows(_mlp_partial(layer.mlp, gathered))
    hidden_states = layer.post_feedforward_layernorm(hidden_states)
    hidden_states = hidden_states + residual

    if per_layer_input is not None and layer.per_layer_input_gate is not None:
        per_layer_input = _own_rows(per_layer_input, layer, num_tokens)
    hidden_states = _per_layer_embedding(layer, hidden_states, per_layer_input)
    hidden_states = hidden_states * layer.layer_scalar

    if layer._tpu_sr_last:
        hidden_states = _all_gather_rows(hidden_states, num_tokens)
    return hidden_states


def sharded_residual_forward(
    self: Any,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    per_layer_input: torch.Tensor | None = None,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    del residual, kwargs
    if _use_sharded(self, positions.shape[0]):
        out = _sharded_forward(self, positions, hidden_states, per_layer_input)
    else:
        out = _replicated_forward(self, positions, hidden_states, per_layer_input)
    return out, None


def configure_layer(
    layer: Any,
    num_hidden_layers: int,
    mode: str,
    min_rows: int = _MIN_SHARDED_TOKENS,
) -> None:
    """Configure one ``Gemma4DecoderLayer`` instance for token-sharded residual."""
    norm_mode = mode.lower()
    if norm_mode not in _VALID_MODES:
        raise ValueError(
            f"Unsupported TPU_SHARDED_RESIDUAL mode {mode!r}; expected {_VALID_MODES}"
        )
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        raise ValueError("TPU_SHARDED_RESIDUAL requires tensor parallelism (TP > 1)")
    if _get_pp_world_size() > 1:
        raise NotImplementedError(
            "TPU_SHARDED_RESIDUAL does not support pipeline parallelism (PP > 1)"
        )
    if get_pcp_world_size() > 1 or get_dcp_world_size() > 1:
        raise NotImplementedError(
            "TPU_SHARDED_RESIDUAL does not support context parallelism (PCP/DCP > 1)"
        )
    if layer.enable_moe_block:
        raise NotImplementedError(
            "TPU_SHARDED_RESIDUAL does not support Gemma 4 MoE layers"
        )
    if layer.self_attn.is_kv_shared_layer:
        raise NotImplementedError(
            "TPU_SHARDED_RESIDUAL does not support KV-shared Gemma 4 layers"
        )
    for proj in (layer.self_attn.o_proj, layer.mlp.down_proj):
        if proj.bias is not None:
            raise NotImplementedError(
                "TPU_SHARDED_RESIDUAL does not support row-parallel bias"
            )
        proj.reduce_results = False
    if norm_mode == MODE_FP8:
        pq.ensure_ops()
    layer._tpu_sr_mode = norm_mode
    layer._tpu_sr_min_rows = min_rows
    layer._tpu_sr_tp = tp_size
    layer._tpu_sr_rank = get_tensor_model_parallel_rank()
    layer._tpu_sr_first = layer.layer_idx == 0
    layer._tpu_sr_last = layer.layer_idx == num_hidden_layers - 1


def maybe_patch_gemma4_sharded_residual(
    model_config: ModelConfig | None = None,
) -> bool | None:
    """Patch ``Gemma4DecoderLayer`` when ``TPU_SHARDED_RESIDUAL`` is enabled."""
    raw_mode = envs.TPU_SHARDED_RESIDUAL
    if not raw_mode:
        return None
    mode = raw_mode.lower()
    if model_config is None:
        return False
    text_config = model_config.hf_text_config
    if "gemma4" not in str(getattr(text_config, "model_type", "")).lower():
        return False
    if getattr(text_config, "num_kv_shared_layers", 0) > 0:
        raise NotImplementedError(
            "TPU_SHARDED_RESIDUAL does not support KV-shared Gemma 4 layers"
        )

    from vllm.model_executor.models.gemma4 import Gemma4DecoderLayer

    if getattr(Gemma4DecoderLayer, _PATCHED_ATTR, False):
        return None

    # Upstream Gemma4DecoderLayer.forward returns (hidden_states, None) despite
    # its tuple[Tensor, Tensor] return annotation.
    target_cls: Any = Gemma4DecoderLayer
    orig_init = target_cls.__init__

    def _patched_init(self: Any, config: Any, *args: Any, **kwargs: Any) -> None:
        orig_init(self, config, *args, **kwargs)
        configure_layer(self, config.num_hidden_layers, mode, _MIN_SHARDED_TOKENS)

    target_cls.__init__ = _patched_init
    target_cls.forward = sharded_residual_forward
    setattr(target_cls, _PATCHED_ATTR, True)
    logger.info(
        "Applied TPU patch: Gemma 4 token-sharded residual (gather=%s, min_tokens=%d).",
        mode,
        _MIN_SHARDED_TOKENS,
    )
    return None
