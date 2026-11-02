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
"""Gemma4 at TP > 1: a token-sharded residual stream for large token buckets.

Upstream all-reduces every row-parallel output (o_proj, down_proj and, on MoE
layers, the routed experts), then each TP rank runs the norms, the residual
adds, ``layer_scalar`` and the next projection's activation quantization on all
T tokens. With TPU_GEMMA4_SEQUENCE_PARALLEL, a token bucket whose per-rank
shard is large enough reduce-scatters those outputs over tokens instead,
runs that per-token work on the rank's T/tp rows, and all-gathers the input of
the next column-parallel projection: as FP8 rows plus per-token scales when the
projection's method quantizes its input per token anyway (the bytes it would
have produced itself), in bf16 otherwise. The residual stream stays sharded
from the embedding to the final norm, and one all-gather returns ``[T, H]`` to
the runner. Smaller buckets, decode among them, keep upstream's all-reduces.
"""

from __future__ import annotations

from itertools import islice
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
from vllm.model_executor.models.gemma4 import Gemma4DecoderLayer, Gemma4Model

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

_orig_layer_init = Gemma4DecoderLayer.__init__
_orig_model_init = Gemma4Model.__init__
_orig_model_forward = Gemma4Model.forward

# Below this many residual-stream bytes per rank, the reduce-scatter and
# all-gather that replace an all-reduce are latency-bound and cost more than it.
_MIN_SHARD_BYTES = 1 << 20


def _use_sequence_parallel(num_tokens: int, row_bytes: int, tp_size: int) -> bool:
    # Only the backbone warmup gives each side of this branch its own trace
    # (shape_variants); a graph compiled outside it keeps the all-reduces.
    if torch.compiler.is_compiling() and shape_variants.aot_cache_tag() is None:
        return False
    return (
        num_tokens % tp_size == 0
        and num_tokens // tp_size * row_bytes >= _MIN_SHARD_BYTES
    )


def _configure_layer(layer: Gemma4DecoderLayer) -> None:
    attn: Any = layer.self_attn
    if attn.is_kv_shared_layer:
        raise ValueError("TPU_GEMMA4_SEQUENCE_PARALLEL does not support KV sharing")
    if layer.per_layer_input_gate is not None:
        raise ValueError(
            "TPU_GEMMA4_SEQUENCE_PARALLEL does not support per-layer embeddings"
        )
    mlp: Any = layer.mlp
    projections = (attn.qkv_proj, attn.o_proj, mlp.gate_up_proj, mlp.down_proj)
    if any(p.bias is not None for p in projections):
        raise ValueError("TPU_GEMMA4_SEQUENCE_PARALLEL does not support biases")
    if attn.o_proj.tp_size <= 1:
        raise ValueError("TPU_GEMMA4_SEQUENCE_PARALLEL requires TP > 1")
    # Both branches of the forward place the reduction themselves.
    attn.o_proj.reduce_results = False
    mlp.down_proj.reduce_results = False
    if layer.experts is not None:
        experts: Any = layer.experts
        if experts.moe_config.dp_size > 1:
            # The runner would gather tokens over DP around the experts.
            raise ValueError(
                "TPU_GEMMA4_SEQUENCE_PARALLEL does not support MoE with DP > 1"
            )
        experts.moe_config.skip_final_all_reduce = True


def _patched_layer_init(self: Gemma4DecoderLayer, *args: Any, **kwargs: Any) -> None:
    _orig_layer_init(self, *args, **kwargs)
    _configure_layer(self)


def _patched_model_init(self: Gemma4Model, *args: Any, **kwargs: Any) -> None:
    _orig_model_init(self, *args, **kwargs)
    if get_pp_group().world_size > 1:
        raise ValueError("TPU_GEMMA4_SEQUENCE_PARALLEL does not support PP")
    if self.fast_prefill_enabled:
        raise ValueError(
            "TPU_GEMMA4_SEQUENCE_PARALLEL does not support kv_sharing_fast_prefill"
        )
    # A buffer, not a Python int: every TP rank keeps one identical program.
    self.register_buffer(
        "_tpu_sp_rank",
        torch.tensor(get_tensor_model_parallel_rank(), dtype=torch.int64),
        persistent=False,
    )


def _local_rows(x: torch.Tensor, rank: torch.Tensor, tp_size: int) -> torch.Tensor:
    """This rank's contiguous 1/tp of the rows of replicated ``x``."""
    rows = x.shape[0] // tp_size
    index = torch.arange(rows, device=x.device) + rank * rows
    return x.index_select(0, index)


def _gathered_projection(linear: Any, x_local: torch.Tensor) -> torch.Tensor:
    """Column-parallel ``linear`` on the all-gather of token-sharded rows."""
    method = linear.quant_method
    rows = x_local.shape[0] * linear.tp_size
    supports = getattr(method, "supports_quantized_input", None)
    if supports is not None and supports(linear, rows):
        x_q, x_scale = method.quantize_input(linear, x_local)
        # Collectives move FP8 as raw bytes.
        rows_q = tensor_model_parallel_all_gather(x_q.view(torch.int8), 0)
        x_scale = tensor_model_parallel_all_gather(x_scale, 0)
        return method.apply_quantized_input(linear, rows_q.view(x_q.dtype), x_scale)
    out, _ = linear(tensor_model_parallel_all_gather(x_local, 0))
    return out


def _attention_partial(
    attn: Any, positions: torch.Tensor, qkv: torch.Tensor
) -> torch.Tensor:
    """``Gemma4Attention.forward`` from the qkv projection on; o_proj partial."""
    q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    q = attn.q_norm(q.unflatten(-1, (attn.num_heads, attn.head_dim))).flatten(-2, -1)
    k = attn.k_norm(k.unflatten(-1, (attn.num_kv_heads, attn.head_dim)))
    q, k = attn.rotary_emb(positions, q, k.flatten(-2, -1))
    v = attn.v_norm(v.unflatten(-1, (attn.num_kv_heads, attn.head_dim)))
    out, _ = attn.o_proj(attn.attn(q, k, v.flatten(-2, -1)))
    return out


def _sequence_parallel_layer(
    self: Gemma4DecoderLayer, positions: torch.Tensor, hidden_states: torch.Tensor
) -> torch.Tensor:
    """One decoder layer on this rank's rows of the residual stream."""
    residual = hidden_states
    qkv = _gathered_projection(self.self_attn.qkv_proj, self.input_layernorm(residual))
    attn_out = tensor_model_parallel_reduce_scatter(
        _attention_partial(self.self_attn, positions, qkv), 0
    )
    hidden_states = self.post_attention_layernorm(attn_out) + residual
    residual = hidden_states

    mlp: Any = self.mlp
    gate_up = _gathered_projection(
        mlp.gate_up_proj, self.pre_feedforward_layernorm(hidden_states)
    )
    mlp_out, _ = mlp.down_proj(mlp.act_fn(gate_up))
    if self.enable_moe_block:
        assert self.post_feedforward_layernorm_1 is not None
        assert self.pre_feedforward_layernorm_2 is not None
        assert self.router is not None
        assert self.experts is not None
        assert self.post_feedforward_layernorm_2 is not None
        # gmm_v2 quantizes its lhs per 512-column block, so the experts take
        # bf16 rows; every rank routes all T tokens, as upstream does.
        experts_in = tensor_model_parallel_all_gather(
            self.pre_feedforward_layernorm_2(residual), 0
        )
        router_logits = tensor_model_parallel_all_gather(self.router(residual), 0)
        experts_out = self.experts(experts_in, router_logits)
        hidden_states = self.post_feedforward_layernorm_1(
            tensor_model_parallel_reduce_scatter(mlp_out, 0)
        ) + self.post_feedforward_layernorm_2(
            tensor_model_parallel_reduce_scatter(experts_out, 0)
        )
    else:
        hidden_states = tensor_model_parallel_reduce_scatter(mlp_out, 0)
    hidden_states = self.post_feedforward_layernorm(hidden_states) + residual
    return hidden_states * self.layer_scalar


def _all_reduce_layer(
    self: Gemma4DecoderLayer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    """Upstream's decoder layer, with its all-reduces placed explicitly."""
    residual = hidden_states
    hidden_states = self.input_layernorm(residual)
    hidden_states = tensor_model_parallel_all_reduce(
        self.self_attn(positions=positions, hidden_states=hidden_states, **kwargs)
    )
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = hidden_states + residual
    residual = hidden_states

    hidden_states = self.pre_feedforward_layernorm(hidden_states)
    hidden_states = tensor_model_parallel_all_reduce(self.mlp(hidden_states))
    if self.enable_moe_block:
        assert self.post_feedforward_layernorm_1 is not None
        assert self.pre_feedforward_layernorm_2 is not None
        assert self.router is not None
        assert self.experts is not None
        assert self.post_feedforward_layernorm_2 is not None
        hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)
        hidden_states_2 = self.pre_feedforward_layernorm_2(residual)
        router_logits = self.router(residual)
        hidden_states_2 = tensor_model_parallel_all_reduce(
            self.experts(hidden_states_2, router_logits)
        )
        hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)
        hidden_states = hidden_states_1 + hidden_states_2

    hidden_states = self.post_feedforward_layernorm(hidden_states)
    hidden_states = hidden_states + residual
    return hidden_states * self.layer_scalar


def _patched_layer_forward(
    self: Gemma4DecoderLayer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    per_layer_input: torch.Tensor | None = None,
    sequence_parallel: bool = False,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    # Gemma4 carries the residual inside hidden_states (residual is always
    # None), and _configure_layer refused per-layer embeddings.
    del residual, per_layer_input
    if sequence_parallel:
        return _sequence_parallel_layer(self, positions, hidden_states), None
    return _all_reduce_layer(self, positions, hidden_states, **kwargs), None


def _patched_model_forward(
    self: Gemma4Model,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: Any,
    inputs_embeds: torch.Tensor | None = None,
    per_layer_inputs: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    # The residual stream has the hidden size and the dtype of the embedding.
    row_bytes = self.config.hidden_size * self.normalizer.dtype.itemsize
    if not _use_sequence_parallel(positions.shape[-1], row_bytes, tp_size):
        return _orig_model_forward(
            self,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            per_layer_inputs,
            **kwargs,
        )
    if self.aux_hidden_state_layers:
        raise NotImplementedError(
            "TPU_GEMMA4_SEQUENCE_PARALLEL does not return auxiliary hidden states"
        )
    if inputs_embeds is None:
        inputs_embeds = self.embed_input_ids(input_ids)
    hidden_states = _local_rows(inputs_embeds, self._tpu_sp_rank, tp_size)
    for layer in islice(self.layers, self.start_layer, self.end_layer):
        hidden_states, _ = layer(
            positions, hidden_states, None, sequence_parallel=True, **kwargs
        )
    return tensor_model_parallel_all_gather(self.norm(hidden_states), 0)


def maybe_patch_gemma4_sequence_parallel(
    model_config: ModelConfig | None,
) -> bool | None:
    """Returns False until a Gemma4 text model config is available."""
    if not envs.TPU_GEMMA4_SEQUENCE_PARALLEL:
        return None
    if model_config is None or model_config.hf_text_config.model_type != "gemma4_text":
        return False
    Gemma4DecoderLayer.__init__ = _patched_layer_init
    Gemma4DecoderLayer.forward = _patched_layer_forward
    Gemma4Model.__init__ = _patched_model_init
    Gemma4Model.forward = _patched_model_forward
    logger.info(
        "Applied TPU patch: Gemma4 sequence-parallel residual stream for token "
        "buckets whose per-rank shard is at least %d bytes; FP8 per-token "
        "projections all-gather FP8 rows, the rest bf16.",
        _MIN_SHARD_BYTES,
    )
    return None
