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
"""Sequence-parallel residual stream for Gemma 4 under TPU_GEMMA4_SEQUENCE_PARALLEL."""

from __future__ import annotations

from itertools import islice
from typing import Any

import torch
from torch.distributed import _functional_collectives as funcol
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.models.gemma4 import Gemma4DecoderLayer, Gemma4Model
from vllm.model_executor.models.utils import sequence_parallel_chunk

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.gemma4.layers import (
    attention_from_qkv,
    down_proj_from_gate_up,
    o_projection,
)

logger = init_logger(__name__)

_orig_layer_init = Gemma4DecoderLayer.__init__
_orig_model_init = Gemma4Model.__init__
_orig_model_forward = Gemma4Model.forward

# Below 1 MiB per TP rank, reduce-scatter + all-gather latency exceeds all-reduce.
_MIN_SHARD_BYTES = 1 << 20
_MAX_MICROBATCHES = 2
_AUX_HIDDEN_STATE_METHODS = ("eagle3", "extract_hidden_states", "dflash", "dspark")


def _shards_fit(num_tokens: int, row_bytes: int, num_shards: int) -> bool:
    return (
        num_tokens % num_shards == 0
        and (num_tokens // num_shards) * row_bytes >= _MIN_SHARD_BYTES
    )


def _use_sequence_parallel(num_tokens: int, row_bytes: int, tp_size: int) -> bool:
    # Only backbone warmup traces both branches via shape_variants.
    if torch.compiler.is_compiling() and shape_variants.aot_cache_tag() is None:
        return False
    return _shards_fit(num_tokens, row_bytes, tp_size)


def _num_microbatches(num_tokens: int, row_bytes: int, tp_size: int) -> int:
    """Return 2 microbatches when each shard meets ``_MIN_SHARD_BYTES``, else 1."""
    return (
        _MAX_MICROBATCHES
        if _shards_fit(num_tokens, row_bytes, _MAX_MICROBATCHES * tp_size)
        else 1
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
    attn.o_proj.reduce_results = False
    mlp.down_proj.reduce_results = False
    if layer.experts is not None:
        experts: Any = layer.experts
        if experts.moe_config.dp_size > 1:
            raise ValueError(
                "TPU_GEMMA4_SEQUENCE_PARALLEL does not support MoE with DP > 1"
            )
        if not experts.moe_config.moe_parallel_config.is_sequence_parallel:
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
    speculative = self.vllm_config.speculative_config
    if speculative is not None and speculative.method in _AUX_HIDDEN_STATE_METHODS:
        raise ValueError(
            "TPU_GEMMA4_SEQUENCE_PARALLEL does not support speculative method "
            f"{speculative.method!r}: its drafter reads auxiliary hidden states"
        )


def _waited(t: torch.Tensor) -> torch.Tensor:
    return funcol.wait_tensor(t) if isinstance(t, funcol.AsyncCollectiveTensor) else t


def _all_gather_rows(x: torch.Tensor) -> torch.Tensor:
    return _waited(funcol.all_gather_single(x, 0, get_tp_group().device_group))


def _reduce_scatter_rows(x: torch.Tensor) -> torch.Tensor:
    return _waited(
        funcol.reduce_scatter_single(x, "sum", 0, get_tp_group().device_group)
    )


def _gathered_projection(linear: Any, x_local: torch.Tensor) -> torch.Tensor:
    """Run column-parallel ``linear`` on the all-gather of token-sharded rows."""
    method = linear.quant_method
    rows = x_local.shape[0] * linear.tp_size
    if isinstance(method, VllmFp8LinearMethodTPU) and method.supports_quantized_input(
        linear, rows
    ):
        x_q, x_scale = method.quantize_input(linear, x_local)
        rows_q = _all_gather_rows(x_q.view(torch.int8))
        x_scale = _all_gather_rows(x_scale)
        return method.apply_quantized_input(linear, rows_q.view(x_q.dtype), x_scale)
    out, _ = linear(_all_gather_rows(x_local))
    return out


def _token_chunks(x: torch.Tensor, chunks: int) -> list[torch.Tensor]:
    # View + unbind avoids passing a symbolic split size that torch_tpu rejects.
    return list(x.view(chunks, -1, *x.shape[1:]).unbind(0)) if chunks > 1 else [x]


def _attention_chunks(
    attn: Any, positions: torch.Tensor, qkv: torch.Tensor, chunks: int
) -> list[Any]:
    out = attention_from_qkv(attn, positions, qkv, o_proj_rows=qkv.shape[0] // chunks)
    if isinstance(out, tuple):
        return list(zip(*(_token_chunks(t, chunks) for t in out)))
    return _token_chunks(out, chunks)


def _sequence_parallel_layer(
    self: Gemma4DecoderLayer,
    positions: torch.Tensor,
    hidden_states: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Run one Gemma 4 decoder layer on token-sharded residual chunks."""
    attn: Any = self.self_attn
    mlp: Any = self.mlp
    residual = hidden_states
    qkv = torch.cat(
        [_gathered_projection(attn.qkv_proj, self.input_layernorm(c)) for c in residual]
    )
    attn_out = _attention_chunks(attn, positions, qkv, len(residual))
    residual = [
        self.post_attention_layernorm(_reduce_scatter_rows(o_projection(attn, a))) + r
        for a, r in zip(attn_out, residual)
    ]

    mlp_out = [
        down_proj_from_gate_up(
            mlp,
            _gathered_projection(mlp.gate_up_proj, self.pre_feedforward_layernorm(r)),
        )
        for r in residual
    ]
    if self.enable_moe_block:
        assert self.post_feedforward_layernorm_1 is not None
        assert self.pre_feedforward_layernorm_2 is not None
        assert self.router is not None
        assert self.experts is not None
        assert self.post_feedforward_layernorm_2 is not None
        mlp_out = [
            self.post_feedforward_layernorm_1(_reduce_scatter_rows(m)) for m in mlp_out
        ]
        if self.experts.moe_config.moe_parallel_config.is_sequence_parallel:
            experts_out = [
                self.experts(self.pre_feedforward_layernorm_2(r), self.router(r))
                for r in residual
            ]
            hidden_states = [
                m + self.post_feedforward_layernorm_2(e)
                for m, e in zip(mlp_out, experts_out)
            ]
        else:
            experts_out = [
                self.experts(
                    _all_gather_rows(self.pre_feedforward_layernorm_2(r)),
                    _all_gather_rows(self.router(r)),
                )
                for r in residual
            ]
            hidden_states = [
                m + self.post_feedforward_layernorm_2(_reduce_scatter_rows(e))
                for m, e in zip(mlp_out, experts_out)
            ]
    else:
        hidden_states = [_reduce_scatter_rows(m) for m in mlp_out]
    return [
        (self.post_feedforward_layernorm(h) + r) * self.layer_scalar
        for h, r in zip(hidden_states, residual)
    ]


def _all_reduce_layer(
    self: Gemma4DecoderLayer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    """Run upstream Gemma 4 decoder layer with explicit TP reductions."""
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
        if self.experts.moe_config.moe_parallel_config.is_sequence_parallel:
            sharded_out = self.experts(
                sequence_parallel_chunk(hidden_states_2),
                sequence_parallel_chunk(router_logits),
            )
            hidden_states_2 = _all_gather_rows(sharded_out)
        else:
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
    hidden_states: Any,
    residual: torch.Tensor | None,
    per_layer_input: torch.Tensor | None = None,
    sequence_parallel: bool = False,
    **kwargs: Any,
) -> tuple[Any, None]:
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
    num_tokens = positions.shape[-1]
    tokens = inputs_embeds if input_ids is None else input_ids
    torch._check(tokens.shape[0] == num_tokens)
    tp_size = get_tensor_model_parallel_world_size()
    row_bytes = self.config.hidden_size * self.normalizer.dtype.itemsize
    if not _use_sequence_parallel(num_tokens, row_bytes, tp_size):
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
    chunks = _num_microbatches(num_tokens, row_bytes, tp_size)
    hidden_states = [
        sequence_parallel_chunk(c) for c in _token_chunks(inputs_embeds, chunks)
    ]
    for layer in islice(self.layers, self.start_layer, self.end_layer):
        hidden_states, _ = layer(
            positions, hidden_states, None, sequence_parallel=True, **kwargs
        )
    return torch.cat([_all_gather_rows(self.norm(h)) for h in hidden_states])


def apply_gemma4_sequence_parallel_patch() -> None:
    """Install idempotent Gemma 4 sequence-parallel residual patches."""
    if not envs.TPU_GEMMA4_SEQUENCE_PARALLEL:
        return
    if envs.TPU_PRECOMPILE_PEER_REPLAY:
        raise ValueError(
            "TPU_GEMMA4_SEQUENCE_PARALLEL does not support TPU_PRECOMPILE_PEER_REPLAY"
        )
    if getattr(Gemma4Model, "_TPU_SP_PATCHED", False):
        return
    Gemma4DecoderLayer.__init__ = _patched_layer_init
    Gemma4DecoderLayer.forward = _patched_layer_forward
    Gemma4Model.__init__ = _patched_model_init
    Gemma4Model.forward = _patched_model_forward
    Gemma4DecoderLayer._TPU_SP_PATCHED = True
    Gemma4Model._TPU_SP_PATCHED = True
    logger.info(
        "Applied TPU patch: Gemma 4 sequence-parallel residual stream "
        "(min shard %d bytes, up to %d microbatches).",
        _MIN_SHARD_BYTES,
        _MAX_MICROBATCHES,
    )
