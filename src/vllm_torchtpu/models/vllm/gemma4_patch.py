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
"""Gemma4 at TP > 1: prefill buckets restructured around the TP group.

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

With TPU_GEMMA4_CP_MIN_TOKENS, a bucket of at least that many tokens is a
context-parallel program of its own instead: each rank holds two zigzag pieces
of the step's one prompt (``cp_prefill.ZigzagPlan``) and runs every layer on
them with a whole copy of the projections, so nothing is reduced; attention
over the whole prompt, and the layer's TP KV cache, come from
``cp_prefill_attention``. The embedding is the vocab shards' lookup
reduce-scattered to each rank's rows, and an all-gather of the final hidden
states returns ``[T, H]`` in token order.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Iterable, Iterator
from itertools import islice
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.distributed import _functional_collectives as funcol
from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import set_current_vllm_config
from vllm.config.utils import getattr_iter
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers import vocab_parallel_embedding
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.models.gemma4 import (
    Gemma4DecoderLayer,
    Gemma4ForCausalLM,
    Gemma4Model,
)
from vllm.model_executor.models.utils import maybe_prefix, sequence_parallel_chunk

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.kernels.experimental.batched_rpa import cp_prefill
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.gemma4_attention_patch import attention_from_qkv
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    get_vllm_model_wrapper_context,
)

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

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


# ---------------------------------------------------------------------------
# Context parallelism (TPU_GEMMA4_CP_MIN_TOKENS)
# ---------------------------------------------------------------------------

_orig_causal_lm_init = Gemma4ForCausalLM.__init__
_orig_causal_lm_forward = Gemma4ForCausalLM.forward
_orig_model_load_weights = Gemma4Model.load_weights

# TPU_GEMMA4_CP_MIN_TOKENS, read when the patch is applied.
_cp_min_tokens = 0

# Gemma4Model's submodule holding every layer's projections whole.
_REPLICA = "cp_replica"
# The checkpoint tensors (names as Gemma4Model.load_weights receives them) that
# the replica takes too.
_REPLICATED_TENSOR = re.compile(
    r"layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj|experts)\."
)


class _ReplicaAttention(nn.Module):
    def __init__(self, attn: Any, quant_config: Any, prefix: str) -> None:
        super().__init__()
        self.qkv_proj = QKVParallelLinear(
            attn.hidden_size,
            attn.head_dim,
            attn.total_num_heads,
            attn.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            disable_tp=True,
        )
        self.o_proj = RowParallelLinear(
            attn.total_num_heads * attn.head_dim,
            attn.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            disable_tp=True,
        )


class _ReplicaMLP(nn.Module):
    def __init__(
        self, mlp: Any, hidden_size: int, quant_config: Any, prefix: str
    ) -> None:
        super().__init__()
        intermediate_size = mlp.down_proj.input_size
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            disable_tp=True,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            disable_tp=True,
        )


class _ReplicaLayer(nn.Module):
    def __init__(self, layer: Any, config: Any, quant_config: Any, prefix: str) -> None:
        super().__init__()
        self.self_attn = _ReplicaAttention(
            layer.self_attn, quant_config, f"{prefix}.self_attn"
        )
        self.mlp = _ReplicaMLP(
            layer.mlp, layer.hidden_size, quant_config, f"{prefix}.mlp"
        )
        if layer.enable_moe_block:
            # Upstream's experts (Gemma4DecoderLayer), every one on this rank.
            self.experts = FusedMoEFactory(
                num_experts=config.num_experts,
                top_k=config.top_k_experts,
                hidden_size=config.hidden_size,
                intermediate_size=getattr_iter(
                    config, ("moe_intermediate_size", "expert_intermediate_size")
                ),
                renormalize=True,
                quant_config=quant_config,
                tp_size=1,
                dp_size=1,
                prefix=f"{prefix}.experts",
                custom_routing_function=layer.router.routing_function,
                activation="gelu_tanh",
            )


class Gemma4ProjectionReplica(nn.Module):
    """Every decoder layer's projections, whole on this rank.

    The module paths mirror Gemma4Model's (``layers.<i>.self_attn.qkv_proj``
    and so on), so the model's own weights mapper stacks the checkpoint
    tensors into the replica as it does into the TP shards.
    """

    def __init__(self, model: Gemma4Model, quant_config: Any, prefix: str) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            _ReplicaLayer(layer, model.config, quant_config, f"{prefix}.layers.{i}")
            for i, layer in enumerate(model.layers)
        )


def _piece_rows(x: torch.Tensor, plan: Any, piece: int) -> torch.Tensor:
    size = plan.piece_tokens
    return x[piece * size : (piece + 1) * size]


def _rank_rows(x: torch.Tensor, plan: Any, rank: int) -> torch.Tensor:
    """The rows of `rank`'s two zigzag pieces, in the order the rank holds them."""
    return torch.cat([_piece_rows(x, plan, p) for p in plan.rank_pieces(rank)])


def _rank_major(x: torch.Tensor, plan: Any) -> torch.Tensor:
    """Every rank's rows, rank after rank, as a reduce-scatter deals them out."""
    return torch.cat([_rank_rows(x, plan, r) for r in range(plan.world)])


def _token_order(rows: torch.Tensor, plan: Any) -> torch.Tensor:
    """Inverts `_rank_major`: the all-gathered rows of every rank, by token."""
    size, blocks = plan.piece_tokens, []
    for piece in range(plan.num_pieces):
        rank, slot = plan.piece_owner(piece)
        start = (2 * rank + slot) * size
        blocks.append(rows[start : start + size])
    return torch.cat(blocks)


def _vocab_shard_embedding(embed: Any, input_ids: torch.Tensor) -> torch.Tensor:
    """VocabParallelEmbedding.forward before its all-reduce: the rows of this
    rank's vocab shard, zero for ids outside it."""
    # The shard bounds as the TPU forward reads them: the buffers the platform
    # registers (_patch_vllm_vocab_parallel_embedding) where they exist.
    bounds = [
        getattr(embed, name, getattr(embed.shard_indices, name))
        for name in (
            "org_vocab_start_index",
            "org_vocab_end_index",
            "num_org_vocab_padding",
            "added_vocab_start_index",
            "added_vocab_end_index",
        )
    ]
    ids, outside = vocab_parallel_embedding.get_masked_input_and_mask(
        input_ids, *bounds
    )
    rows = embed.quant_method.embedding(embed, ids.long())
    return rows.masked_fill(outside.unsqueeze(-1), 0)


# (attn, q, k, v, plan, rank) -> attention of the rank's rows over the prompt.
CPAttention = Callable[..., torch.Tensor]


def _context_parallel_layer(
    layer: Any,
    replica: Any,
    positions: torch.Tensor,
    x: torch.Tensor,
    plan: Any,
    rank: int,
    attention: CPAttention,
) -> torch.Tensor:
    """Upstream's decoder layer on this rank's zigzag rows, every head local."""
    attn = layer.self_attn
    heads, kv_heads, head_dim = (
        attn.total_num_heads,
        attn.total_num_kv_heads,
        attn.head_dim,
    )
    residual = x
    qkv, _ = replica.self_attn.qkv_proj(layer.input_layernorm(x))
    q, k, v = qkv.split(
        [heads * head_dim, kv_heads * head_dim, kv_heads * head_dim], dim=-1
    )
    q = attn.q_norm(q.unflatten(-1, (heads, head_dim))).flatten(-2, -1)
    k = attn.k_norm(k.unflatten(-1, (kv_heads, head_dim)))
    q, k = attn.rotary_emb(positions, q, k.flatten(-2, -1))
    v = attn.v_norm(v.unflatten(-1, (kv_heads, head_dim))).flatten(-2, -1)
    out, _ = replica.self_attn.o_proj(attention(attn.attn, q, k, v, plan, rank))
    x = layer.post_attention_layernorm(out) + residual
    residual = x

    gate_up, _ = replica.mlp.gate_up_proj(layer.pre_feedforward_layernorm(x))
    out, _ = replica.mlp.down_proj(layer.mlp.act_fn(gate_up))
    if layer.enable_moe_block:
        experts = replica.experts(
            layer.pre_feedforward_layernorm_2(residual), layer.router(residual)
        )
        out = layer.post_feedforward_layernorm_1(out)
        out = out + layer.post_feedforward_layernorm_2(experts)
    return (layer.post_feedforward_layernorm(out) + residual) * layer.layer_scalar


def _context_parallel_forward(
    model: Any,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    plan: Any,
    rank: int,
    attention: CPAttention,
) -> torch.Tensor:
    """Gemma4Model.forward of one prompt, on this rank's zigzag rows."""
    shards = _vocab_shard_embedding(model.embed_tokens, _rank_major(input_ids, plan))
    # As upstream: the normalizer scales the reduced embedding.
    x = _reduce_scatter_rows(shards) * model.normalizer
    positions = _rank_rows(positions, plan, rank)
    for layer, replica in zip(model.layers, getattr(model, _REPLICA).layers):
        x = _context_parallel_layer(layer, replica, positions, x, plan, rank, attention)
    return _token_order(_all_gather_rows(model.norm(x)), plan)


@support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": 0})
class Gemma4ContextParallelProgram(nn.Module):
    """The forward of one context-parallel token bucket.

    A compiled module of its own, built for that bucket only: its zigzag plan
    is a constant of the trace, which only slices the token dimension, so the
    bucket size remains a free symbol. It runs the TP model's embedding
    shards, norms, rotary embeddings and attention layers, and its projection
    replica.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        model: Gemma4Model,
        num_tokens: int,
    ) -> None:
        super().__init__()
        from vllm_torchtpu.layers.adapter.cp_prefill_attention import (
            cp_prefill_attention,
        )

        self.model = model
        self.plan = cp_prefill.make_zigzag_plan(
            num_tokens,
            get_tensor_model_parallel_world_size(),
            vllm_config.cache_config.block_size,
        )
        self.rank = get_tensor_model_parallel_rank()
        self.attention: CPAttention = cp_prefill_attention

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return _context_parallel_forward(
            self.model, input_ids, positions, self.plan, self.rank, self.attention
        )


def _check_context_parallel(model: Gemma4Model, vllm_config: VllmConfig) -> None:
    """Refuses a configuration the context-parallel programs cannot serve."""
    what = "TPU_GEMMA4_CP_MIN_TOKENS"
    parallel = vllm_config.parallel_config
    unsupported = [
        name
        for name, present in (
            ("PP", parallel.pipeline_parallel_size > 1),
            ("DP", parallel.data_parallel_size > 1),
            ("LoRA", vllm_config.lora_config is not None),
            ("speculative decoding", vllm_config.speculative_config is not None),
            # Sliding-window layers cache only the final window of the prompt.
            ("prefix caching", vllm_config.cache_config.enable_prefix_caching),
            ("kv_sharing_fast_prefill", model.fast_prefill_enabled),
            # Programs differ per rank while their FX graphs need not.
            ("TPU_PRECOMPILE_PEER_REPLAY", envs.TPU_PRECOMPILE_PEER_REPLAY),
            # One block-table shape per bucket.
            ("VLLM_TPU_MOST_MODEL_LEN", envs.VLLM_TPU_MOST_MODEL_LEN is not None),
        )
        if present
    ]
    moe = any(layer.enable_moe_block for layer in model.layers)
    for layer in model.layers:
        attn: Any = layer.self_attn
        for name, present in (
            ("KV sharing", attn.is_kv_shared_layer),
            ("per-layer embeddings", layer.per_layer_input_gate is not None),
            ("attention biases", attn.o_proj.bias is not None),
            # The replica holds every expert: the loader must read them all.
            ("the EP weight filter", moe and parallel.enable_ep_weight_filter),
            ("EPLB", moe and parallel.enable_eplb),
        ):
            if present and name not in unsupported:
                unsupported.append(name)
    if unsupported:
        raise ValueError(f"{what} does not support {', '.join(unsupported)}")
    if parallel.tensor_parallel_size < 3:
        raise ValueError(f"{what} needs TP >= 3, got {parallel.tensor_parallel_size}")

    # Every step padded into a context-parallel bucket must be one whole
    # prompt: the solo-prefill scheduler's steps of more than N tokens are,
    # and its other steps hold at most N, so N's bucket must be the last one
    # below the threshold.
    sizes = sorted(int(s) for s in vllm_config.compilation_config.compile_sizes)
    solo = envs.VLLM_TPU_SOLO_PREFILL_OVER_TOKENS
    above = [s for s in sizes if s >= solo] if solo else []
    if len(above) < 2 or not above[0] < _cp_min_tokens <= above[1]:
        raise ValueError(
            f"{what}={_cp_min_tokens} needs VLLM_TPU_SOLO_PREFILL_OVER_TOKENS=N "
            "whose token bucket is the last one below it; got "
            f"N={solo}, buckets {sizes}"
        )


def _build_context_parallel_programs(
    model: Gemma4Model, vllm_config: VllmConfig
) -> dict[int, Gemma4ContextParallelProgram]:
    """One compiled program per bucket at or above the threshold.

    Each compiles under its own copy of the compilation config, holding only
    its bucket (its cache directory hashes the files its own trace reads), and
    the TP model keeps the buckets below.
    """
    config = vllm_config.compilation_config
    sizes = sorted(int(s) for s in config.compile_sizes)
    config.compile_sizes = [s for s in sizes if s < _cp_min_tokens]
    programs = {}
    for size in (s for s in sizes if s >= _cp_min_tokens):
        program_config = copy.copy(vllm_config)
        program_config.compilation_config = copy.copy(config)
        program_config.compilation_config.compile_sizes = [size]
        program_config.compilation_config.traced_files = set()
        program_config.compilation_config.cache_dir = ""
        program_config.compilation_config.local_cache_dir = ""
        with (
            set_current_vllm_config(program_config),
            set_model_tag(f"gemma4_cp{size}"),
        ):
            programs[size] = Gemma4ContextParallelProgram(
                vllm_config=program_config, model=model, num_tokens=size
            )
    return programs


def _patched_causal_lm_init(
    self: Gemma4ForCausalLM, *, vllm_config: VllmConfig, prefix: str = ""
) -> None:
    _orig_causal_lm_init(self, vllm_config=vllm_config, prefix=prefix)
    model: Any = self.model
    _check_context_parallel(model, vllm_config)
    replica = Gemma4ProjectionReplica(
        model, vllm_config.quant_config, f"{maybe_prefix(prefix, 'model')}.{_REPLICA}"
    )
    for layer, copied in zip(model.layers, replica.layers):
        pairs = [
            (layer.self_attn.qkv_proj, copied.self_attn.qkv_proj),
            (layer.self_attn.o_proj, copied.self_attn.o_proj),
            (layer.mlp.gate_up_proj, copied.mlp.gate_up_proj),
            (layer.mlp.down_proj, copied.mlp.down_proj),
        ]
        if layer.enable_moe_block:
            pairs.append((layer.experts.routed_experts, copied.experts.routed_experts))
        for tp_module, module in pairs:
            if type(module.quant_method) is not type(tp_module.quant_method):
                raise ValueError(
                    f"{module.prefix} would run {type(module.quant_method).__name__}"
                    f", its TP shard {type(tp_module.quant_method).__name__}"
                )
    setattr(model, _REPLICA, replica)
    # A plain dict: the programs are not submodules, so nothing that walks the
    # model's compiled modules (a re-trace, a profiling reset) reaches them.
    self._cp_programs = _build_context_parallel_programs(model, vllm_config)
    self._cp_attention_prepared = False
    logger.info(
        "Gemma4 context-parallel programs for token buckets %s; the TP model "
        "compiles %s",
        sorted(self._cp_programs),
        vllm_config.compilation_config.compile_sizes,
    )


def _profiling(model: Any) -> bool:
    """Whether this is the memory-profiling forward, run before KV caches exist."""
    attn = model.layers[0].self_attn.attn
    return get_attention_context(attn.layer_name)[2].numel() == 0


def _check_solo_prompt(num_tokens: int) -> None:
    context = get_vllm_model_wrapper_context()
    query_start_loc, seq_lens = context.query_start_loc_cpu, context.seq_lens_cpu
    if seq_lens is None:
        return  # a warmup forward: no request to check
    if len(seq_lens) != 1 or seq_lens[0] != query_start_loc[1]:
        raise RuntimeError(
            f"A context-parallel step of {num_tokens} tokens must hold one prompt, "
            "whole and without cached tokens; got query_start_loc "
            f"{query_start_loc.tolist()}, seq_lens {seq_lens.tolist()}"
        )


def _prepare_context_parallel_attention(model: Any) -> None:
    from vllm_torchtpu.layers.adapter.cp_prefill_attention import (
        prepare_cp_prefill_attention,
    )

    for layer in model.layers:
        attn = layer.self_attn
        prepare_cp_prefill_attention(
            attn.attn,
            num_heads=attn.total_num_heads,
            num_kv_heads=attn.total_num_kv_heads,
            rank=get_tensor_model_parallel_rank(),
            world=get_tensor_model_parallel_world_size(),
        )


def _patched_causal_lm_forward(
    self: Gemma4ForCausalLM,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: Any = None,
    inputs_embeds: torch.Tensor | None = None,
    **kwargs: Any,
) -> Any:
    program = self._cp_programs.get(positions.shape[-1])
    # The profiling forward measures the largest bucket's activations on the
    # TP model, whose are the larger: all T rows rather than T/tp.
    if program is None or _profiling(self.model):
        return _orig_causal_lm_forward(
            self, input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )
    if input_ids is None or intermediate_tensors is not None or kwargs:
        raise NotImplementedError(
            "A context-parallel step takes token ids only, got "
            f"inputs_embeds={inputs_embeds is not None}, "
            f"intermediate_tensors={intermediate_tensors is not None}, {list(kwargs)}"
        )
    _check_solo_prompt(positions.shape[-1])
    if not self._cp_attention_prepared:
        # Builds the attention ops outside any trace, once caches and KV scales
        # exist.
        _prepare_context_parallel_attention(self.model)
        self._cp_attention_prepared = True
    return program(input_ids, positions)


def _patched_model_load_weights(
    self: Gemma4Model, weights: Iterable[tuple[str, torch.Tensor]]
) -> set[str]:
    if not hasattr(self, _REPLICA):
        return _orig_model_load_weights(self, weights)

    def with_replica() -> Iterator[tuple[str, torch.Tensor]]:
        for name, tensor in weights:
            yield name, tensor
            if _REPLICATED_TENSOR.match(name):
                yield f"{_REPLICA}.{name}", tensor

    return _orig_model_load_weights(self, with_replica())


def maybe_patch_gemma4_context_parallel(
    model_config: ModelConfig | None,
) -> bool | None:
    """Returns False until a Gemma4 text model config is available."""
    if not envs.TPU_GEMMA4_CP_MIN_TOKENS:
        return None
    if model_config is None or model_config.hf_text_config.model_type != "gemma4_text":
        return False
    if model_config.architecture != "Gemma4ForCausalLM":
        raise ValueError(
            "TPU_GEMMA4_CP_MIN_TOKENS needs Gemma4ForCausalLM, got "
            f"{model_config.architecture} (--hf-overrides "
            '\'{"architectures": ["Gemma4ForCausalLM"]}\' serves the text model)'
        )
    global _cp_min_tokens
    _cp_min_tokens = envs.TPU_GEMMA4_CP_MIN_TOKENS
    Gemma4ForCausalLM.__init__ = _patched_causal_lm_init
    Gemma4ForCausalLM.forward = _patched_causal_lm_forward
    Gemma4Model.load_weights = _patched_model_load_weights
    logger.info(
        "Applied TPU patch: Gemma4 context-parallel prefill for token buckets of "
        "at least %d tokens.",
        _cp_min_tokens,
    )
    return None
