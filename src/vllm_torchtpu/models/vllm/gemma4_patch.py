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
from vllm_torchtpu.models.vllm.gemma4_attention_patch import attention_from_qkv

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

_orig_layer_init = Gemma4DecoderLayer.__init__
_orig_model_init = Gemma4Model.__init__
_orig_model_forward = Gemma4Model.forward

# Below this many residual-stream bytes per rank, the reduce-scatter and
# all-gather that replace an all-reduce are latency-bound and cost more than it.
_MIN_SHARD_BYTES = 1 << 20

# TPU_GEMMA4_SP_MICROBATCHES, read when the patch is applied: the traced
# forward branches on it, and Dynamo guards on a module global by value.
_microbatches = 1


def _shards_fit(num_tokens: int, row_bytes: int, num_shards: int) -> bool:
    return (
        num_tokens % num_shards == 0
        and num_tokens // num_shards * row_bytes >= _MIN_SHARD_BYTES
    )


def _use_sequence_parallel(num_tokens: int, row_bytes: int, tp_size: int) -> bool:
    # Only the backbone warmup gives each side of this branch its own trace
    # (shape_variants); a graph compiled outside it keeps the all-reduces.
    if torch.compiler.is_compiling() and shape_variants.aot_cache_tag() is None:
        return False
    return _shards_fit(num_tokens, row_bytes, tp_size)


def _num_microbatches(num_tokens: int, row_bytes: int, tp_size: int) -> int:
    """Token chunks a sequence-parallel step runs as independent dataflows.

    With two, one chunk's reduce-scatters and all-gathers have the other
    chunk's projections and norms to overlap with; attention still takes all T.
    """
    chunks = _microbatches
    return chunks if _shards_fit(num_tokens, row_bytes, chunks * tp_size) else 1


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
        if experts.moe_config.dp_size > 1 or experts.moe_config.is_sequence_parallel:
            # The runner would gather tokens around the experts itself.
            raise ValueError(
                "TPU_GEMMA4_SEQUENCE_PARALLEL does not support MoE with DP > 1 or "
                "with the MoE's own sequence parallelism"
            )
        experts.moe_config.skip_final_all_reduce = True


def _patched_layer_init(self: Gemma4DecoderLayer, *args: Any, **kwargs: Any) -> None:
    _orig_layer_init(self, *args, **kwargs)
    _configure_layer(self)


# Speculative methods whose drafter reads the target's intermediate hidden states
# (vLLM's SpeculativeConfig lists them in its compile hash factors).
_AUX_HIDDEN_STATE_METHODS = ("eagle3", "extract_hidden_states", "dflash", "dspark")


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


# Functional collectives: vLLM's all_gather and reduce_scatter allocate their
# output with torch.empty at the traced token count, which torch_tpu cannot
# lower while that count is symbolic (the fused EP MoE gathers the same way).
# A traced funcol call already ends in its wait_tensor, and torch_tpu refuses a
# wait on a wait; only an eager AsyncCollectiveTensor is waited here.
def _waited(t: torch.Tensor) -> torch.Tensor:
    return funcol.wait_tensor(t) if isinstance(t, funcol.AsyncCollectiveTensor) else t


def _all_gather_rows(x: torch.Tensor) -> torch.Tensor:
    return _waited(funcol.all_gather_single(x, 0, get_tp_group().device_group))


def _reduce_scatter_rows(x: torch.Tensor) -> torch.Tensor:
    return _waited(
        funcol.reduce_scatter_single(x, "sum", 0, get_tp_group().device_group)
    )


def _gathered_projection(linear: Any, x_local: torch.Tensor) -> torch.Tensor:
    """Column-parallel ``linear`` on the all-gather of token-sharded rows."""
    method = linear.quant_method
    rows = x_local.shape[0] * linear.tp_size
    if isinstance(method, VllmFp8LinearMethodTPU) and method.supports_quantized_input(
        linear, rows
    ):
        x_q, x_scale = method.quantize_input(linear, x_local)
        # Collectives move FP8 as raw bytes.
        rows_q = _all_gather_rows(x_q.view(torch.int8))
        x_scale = _all_gather_rows(x_scale)
        return method.apply_quantized_input(linear, rows_q.view(x_q.dtype), x_scale)
    out, _ = linear(_all_gather_rows(x_local))
    return out


def _token_chunks(x: torch.Tensor, chunks: int) -> list[torch.Tensor]:
    # Not x.chunk: a split takes the symbolic chunk length as an argument,
    # which torch_tpu cannot lower; a view and integer selects need none.
    return list(x.view(chunks, -1, *x.shape[1:]).unbind(0)) if chunks > 1 else [x]


_attention = attention_from_qkv


def _sequence_parallel_layer(
    self: Gemma4DecoderLayer,
    positions: torch.Tensor,
    hidden_states: list[torch.Tensor],
) -> list[torch.Tensor]:
    """One decoder layer on this rank's rows of each contiguous token chunk.

    Chunks share only the attention call; the rest of the layer is one
    independent dataflow per chunk, so the scheduler may overlap one chunk's
    collectives with another chunk's projections.
    """
    attn: Any = self.self_attn
    mlp: Any = self.mlp
    residual = hidden_states
    qkv = torch.cat(
        [_gathered_projection(attn.qkv_proj, self.input_layernorm(h)) for h in residual]
    )
    attn_out = _token_chunks(_attention(attn, positions, qkv), len(residual))
    hidden_states = [
        self.post_attention_layernorm(_reduce_scatter_rows(attn.o_proj(a)[0])) + r
        for a, r in zip(attn_out, residual)
    ]
    residual = hidden_states

    mlp_out = [
        mlp.down_proj(
            mlp.act_fn(
                _gathered_projection(
                    mlp.gate_up_proj, self.pre_feedforward_layernorm(h)
                )
            )
        )[0]
        for h in hidden_states
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
        # gmm_v2 quantizes its lhs per 512-column block, so the experts take
        # bf16 rows; every rank routes all of a chunk's tokens, as upstream does.
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
    hidden_states: Any,
    residual: torch.Tensor | None,
    per_layer_input: torch.Tensor | None = None,
    sequence_parallel: bool = False,
    **kwargs: Any,
) -> tuple[Any, None]:
    """Upstream's signature; with ``sequence_parallel``, ``hidden_states`` is
    the list of this rank's rows of each token chunk (see the model forward)."""
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
    num_tokens = positions.shape[-1]
    # shape_variants hands a bucket's executable between the traces of the two
    # branches by input position, and inputs are ordered by first use: stating
    # the token count's identity first gives both traces the same leading
    # symbolic inputs, wherever each branch would first use them.
    tokens = inputs_embeds if input_ids is None else input_ids
    torch._check(tokens.shape[0] == num_tokens)
    tp_size = get_tensor_model_parallel_world_size()
    # The residual stream has the hidden size and the dtype of the embedding.
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


def maybe_patch_gemma4_sequence_parallel(
    model_config: ModelConfig | None,
) -> bool | None:
    """Returns False until a Gemma4 text model config is available."""
    if not envs.TPU_GEMMA4_SEQUENCE_PARALLEL:
        return None
    if model_config is None or model_config.hf_text_config.model_type != "gemma4_text":
        return False
    if envs.TPU_PRECOMPILE_PEER_REPLAY:
        # sequence_parallel_chunk bakes each rank's rows into its program while
        # the FX graph, which peer replay fingerprints, is the same on every rank.
        raise ValueError(
            "TPU_GEMMA4_SEQUENCE_PARALLEL does not support TPU_PRECOMPILE_PEER_REPLAY"
        )
    microbatches = envs.TPU_GEMMA4_SP_MICROBATCHES
    if microbatches < 1:
        raise ValueError(f"TPU_GEMMA4_SP_MICROBATCHES must be >= 1, got {microbatches}")
    global _microbatches
    _microbatches = microbatches
    Gemma4DecoderLayer.__init__ = _patched_layer_init
    Gemma4DecoderLayer.forward = _patched_layer_forward
    Gemma4Model.__init__ = _patched_model_init
    Gemma4Model.forward = _patched_model_forward
    logger.info(
        "Applied TPU patch: Gemma4 sequence-parallel residual stream for token "
        "buckets whose per-rank shard is at least %d bytes, in up to %d token "
        "chunks; FP8 per-token projections all-gather FP8 rows, the rest bf16.",
        _MIN_SHARD_BYTES,
        _microbatches,
    )
    return None
