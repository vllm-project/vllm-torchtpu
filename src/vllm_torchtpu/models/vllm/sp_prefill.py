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
"""Sequence-parallel prefill with replicated dense weights (Qwen3-VL backbone).

A TP prefill all-reduces the [tokens, hidden] activation twice per layer, about
half its device time on v7x-8. Here each TP rank runs the dense layers on its
1/tp of the rows with every rank's weight shard -- its own, plus a replica of
the others' -- and the ranks exchange only around attention: one all-to-all
gives each rank its own head group for all rows, which is exactly the TP
attention input, so the attention kernel, its metadata and the head-sharded KV
pools stay as they are; a second all-to-all returns the output rows to their
owners. That moves 9x fewer bytes per layer than the two all-reduces, at the
price of the replica, (tp - 1) times the dense weight shard per rank, which
comes out of the KV cache.

Opt in with ``TPU_SP_PREFILL_MIN_TOKENS``: token buckets at or above it run this
way and smaller ones (decode, short prompts) keep the TP graph, since streaming
the full weights once costs more than the all-reduces of a short step. The
choice is a branch on the token count, so each bucket compiles one of the two
graphs (see ``vllm_torchtpu.compilation.shape_variants``). The TP graph's
inputs are a subset of this one's, which is what lets the two share one live
graph.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional, cast

import torch
from torch import nn

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.sequence_parallel import (
    RowExchange,
    column,
    gather_other_shards,
    in_out,
    row,
)
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer
    from vllm.model_executor.models.qwen3_vl import Qwen3LLMModel
    from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)

_UPSTREAM_FORWARD = None


def maybe_patch_qwen3_vl_sp_prefill(
    model_config: Optional["ModelConfig"] = None,
) -> bool | None:
    """Route the Qwen3-VL text backbone's large buckets to sequence parallel."""
    global _UPSTREAM_FORWARD
    if not envs.TPU_SP_PREFILL_MIN_TOKENS:
        return None
    if model_config is None:
        return False
    from vllm.model_executor.models import qwen3_vl

    from vllm_torchtpu.models.vllm.qwen3_vl_patch import _has_qwen3_vl_vision

    if not _has_qwen3_vl_vision(model_config):
        return False
    if _UPSTREAM_FORWARD is None:
        _UPSTREAM_FORWARD = qwen3_vl.Qwen3LLMModel.forward
        qwen3_vl.Qwen3LLMModel.forward = _backbone_forward
        logger.info("Applied TPU patch: Qwen3-VL sequence-parallel prefill.")
    return None


def _backbone_forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: Optional["IntermediateTensors"] = None,
    inputs_embeds: torch.Tensor | None = None,
    deepstack_input_embeds: Optional["IntermediateTensors"] = None,
):
    # A module-level function, not a closure: vLLM swaps the compiled bytecode
    # into this function's __code__, which must keep its free variables.
    sp = getattr(self, "sp_prefill", None)
    if sp is not None and sp.serves(positions.shape[-1]):
        return sp(self, input_ids, positions, inputs_embeds, deepstack_input_embeds)
    assert _UPSTREAM_FORWARD is not None  # Set before this forward is installed.
    return _UPSTREAM_FORWARD(
        self,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds,
        deepstack_input_embeds,
    )


def maybe_install(model: nn.Module, token_buckets: Sequence[int]) -> None:
    """Give the loaded model's backbone its weight replica; no-op when off."""
    min_tokens = envs.TPU_SP_PREFILL_MIN_TOKENS
    if not min_tokens:
        return
    from vllm.model_executor.models.qwen3_vl import Qwen3LLMModel

    backbones = [m for m in model.modules() if isinstance(m, Qwen3LLMModel)]
    # Qwen3-VL-MoE passes the patch's vision check but has no Qwen3LLMModel.
    if _UPSTREAM_FORWARD is None or len(backbones) != 1:
        raise ValueError(
            "TPU_SP_PREFILL_MIN_TOKENS is set, but the model is not a Qwen3-VL "
            "dense text backbone (Qwen3-VL-MoE is not supported)."
        )
    sp = SequenceParallelPrefill(backbones[0], min_tokens, token_buckets)
    backbones[0].sp_prefill = sp
    logger.info(
        "Sequence-parallel prefill from %d tokens: %.1f GiB of replicated "
        "dense weights per rank.",
        min_tokens,
        sum(b.numel() * b.element_size() for b in sp.buffers()) / 2**30,
    )


class SequenceParallelPrefill(nn.Module):
    """The backbone forward over this rank's rows with every rank's weights."""

    def __init__(
        self,
        backbone: "Qwen3LLMModel",
        min_tokens: int,
        token_buckets: Sequence[int],
    ) -> None:
        super().__init__()
        from vllm.distributed import get_pp_group

        self.exchange = RowExchange(next(backbone.parameters()).device)
        if get_pp_group().world_size != 1 or self.exchange.world == 1:
            raise ValueError(
                "TPU_SP_PREFILL_MIN_TOKENS needs TP > 1 and no pipeline parallelism."
            )
        if getattr(backbone, "aux_hidden_state_layers", ()):
            raise ValueError(
                "TPU_SP_PREFILL_MIN_TOKENS does not return auxiliary hidden states."
            )
        self.min_tokens = min_tokens
        # A later TP-graph bucket would get a trace that cannot serve SP ones.
        if odd := [n for n in token_buckets if n >= min_tokens and not self.serves(n)]:
            raise ValueError(
                f"TPU_SP_PREFILL_MIN_TOKENS={min_tokens} needs every token bucket "
                f"at or above it to be a multiple of TP={self.exchange.world}, "
                f"got {odd}. Raise it or change --max-num-batched-tokens."
            )
        self.layers = nn.ModuleList(
            _OtherShards(layer, self.exchange.others)
            for layer in cast("Sequence[Qwen3DecoderLayer]", backbone.layers)
        )

    def serves(self, num_tokens: int) -> bool:
        return num_tokens >= self.min_tokens and num_tokens % self.exchange.world == 0

    def forward(
        self,
        backbone: "Qwen3LLMModel",
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        deepstack_input_embeds: Optional["IntermediateTensors"],
    ) -> torch.Tensor:
        # Mirrors Qwen3LLMModel.forward (no pipeline, no auxiliary states).
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = backbone.embed_input_ids(input_ids)
        exchange = self.exchange
        hidden_states, residual = exchange.own_rows(inputs_embeds), None
        layers = cast("Sequence[Qwen3DecoderLayer]", backbone.layers)
        replicas = cast("Sequence[_OtherShards]", self.layers)
        for layer_idx, (layer, replica) in enumerate(
            zip(layers, replicas, strict=True)
        ):
            hidden_states, residual = self._layer(
                layer, replica, positions, hidden_states, residual
            )
            if deepstack_input_embeds is not None and layer_idx < len(
                deepstack_input_embeds
            ):
                hidden_states = hidden_states + exchange.own_rows(
                    deepstack_input_embeds.tensors[
                        f"deepstack_input_embeds_{layer_idx}"
                    ]
                )
        hidden_states, _ = backbone.norm(hidden_states, residual)
        return exchange.group.all_gather(hidden_states, dim=0)

    def _layer(
        self,
        layer: "Qwen3DecoderLayer",
        replica: "_OtherShards",
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Mirrors Qwen3DecoderLayer.forward and Qwen3Attention.forward, with
        # every rank's projections and the reshards around attention.
        if residual is None:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
        else:
            hidden_states, residual = layer.input_layernorm(hidden_states, residual)
        attn, mlp = layer.self_attn, layer.mlp
        qkv = self.exchange.rows_to_heads(
            *column(attn.qkv_proj, replica.qkv, hidden_states)
        )
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // attn.head_dim, attn.head_dim)
        q = attn.q_norm(q_by_head).view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // attn.head_dim, attn.head_dim)
        k = attn.k_norm(k_by_head).view(k.shape)
        q, k = attn.rotary_emb(positions, q, k)
        hidden_states = row(
            attn.o_proj, replica.o, *self.exchange.heads_to_rows(attn.attn(q, k, v))
        )
        hidden_states, residual = layer.post_attention_layernorm(
            hidden_states, residual
        )
        # The MLP pairs each rank's gate-up block with its down block, in the
        # [own, *others] order they come in; no exchange, so no reordering.
        own, rest = column(mlp.gate_up_proj, replica.gate_up, hidden_states)
        hidden_states = row(
            mlp.down_proj, replica.down, mlp.act_fn(own), mlp.act_fn(rest)
        )
        return hidden_states, residual


class _OtherShards(nn.Module):
    """The other TP ranks' shards of one decoder layer's projections, stacked
    in rank order as [tp - 1, in, out]."""

    qkv: torch.Tensor
    o: torch.Tensor
    gate_up: torch.Tensor
    down: torch.Tensor

    def __init__(self, layer: "Qwen3DecoderLayer", others: list[int]) -> None:
        super().__init__()
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        def gather(linear: nn.Module) -> torch.Tensor:
            if linear.bias is not None or not isinstance(
                linear.quant_method, UnquantizedLinearMethod
            ):
                raise ValueError(
                    "TPU_SP_PREFILL_MIN_TOKENS supports unquantized projections "
                    f"without bias, not {linear.prefix}."
                )
            return gather_other_shards(in_out(linear).data, others)

        attn, mlp = layer.self_attn, layer.mlp
        for name, linear in (
            ("qkv", attn.qkv_proj),
            ("o", attn.o_proj),
            ("gate_up", mlp.gate_up_proj),
            ("down", mlp.down_proj),
        ):
            self.register_buffer(name, gather(linear), persistent=False)
