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

from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.collectives import all_to_all
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig
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
    return _UPSTREAM_FORWARD(
        self,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds,
        deepstack_input_embeds,
    )


def maybe_install(model: nn.Module) -> None:
    """Give the loaded model's backbone its weight replica; no-op when off."""
    min_tokens = envs.TPU_SP_PREFILL_MIN_TOKENS
    if not min_tokens:
        return
    if _UPSTREAM_FORWARD is None:
        raise ValueError(
            "TPU_SP_PREFILL_MIN_TOKENS is set, but the model is not a Qwen3-VL "
            "dense text backbone."
        )
    from vllm.model_executor.models.qwen3_vl import Qwen3LLMModel

    backbones = [m for m in model.modules() if isinstance(m, Qwen3LLMModel)]
    if len(backbones) != 1:
        raise ValueError(
            f"TPU_SP_PREFILL_MIN_TOKENS needs one Qwen3LLMModel, got {len(backbones)}."
        )
    backbone = backbones[0]
    backbone.sp_prefill = SequenceParallelPrefill(backbone, min_tokens)
    logger.info(
        "Sequence-parallel prefill from %d tokens: %.1f GiB of replicated "
        "dense weights per rank.",
        min_tokens,
        sum(b.numel() * b.element_size() for b in backbone.sp_prefill.buffers())
        / 2**30,
    )


class SequenceParallelPrefill(nn.Module):
    """The backbone forward over this rank's rows with every rank's weights."""

    def __init__(self, backbone: nn.Module, min_tokens: int) -> None:
        super().__init__()
        from vllm.distributed import get_pp_group, get_tp_group

        self.group = get_tp_group()
        self.world = self.group.world_size
        if get_pp_group().world_size != 1 or self.world == 1:
            raise ValueError(
                "TPU_SP_PREFILL_MIN_TOKENS needs TP > 1 and no pipeline parallelism."
            )
        if getattr(backbone, "aux_hidden_state_layers", ()):
            raise ValueError(
                "TPU_SP_PREFILL_MIN_TOKENS does not return auxiliary hidden states."
            )
        self.min_tokens = min_tokens
        rank = self.group.rank_in_group
        others = [r for r in range(self.world) if r != rank]
        device = backbone.norm.weight.device

        def index(values: list[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int32, device=device)

        # Device tensors, not Python ints: every rank must compile the same
        # program, or torch_tpu pairs their collectives wrongly.
        self.register_buffer("rank", index([rank]), persistent=False)
        self.register_buffer("others", index(others), persistent=False)
        # Where destination rank d's block sits in [own, *others].
        send_order = [
            0 if d == rank else 1 + others.index(d) for d in range(self.world)
        ]
        self.register_buffer("send_order", index(send_order), persistent=False)
        self.layers = nn.ModuleList(
            _OtherShards(layer, others) for layer in backbone.layers
        )

    def serves(self, num_tokens: int) -> bool:
        return num_tokens >= self.min_tokens and num_tokens % self.world == 0

    def forward(
        self,
        backbone: nn.Module,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        deepstack_input_embeds: Optional["IntermediateTensors"],
    ) -> torch.Tensor:
        # Mirrors Qwen3LLMModel.forward (no pipeline, no auxiliary states).
        if inputs_embeds is None:
            inputs_embeds = backbone.embed_input_ids(input_ids)
        hidden_states, residual = self._own_rows(inputs_embeds), None
        for layer_idx, (layer, replica) in enumerate(
            zip(backbone.layers, self.layers, strict=True)
        ):
            hidden_states, residual = self._layer(
                layer, replica, positions, hidden_states, residual
            )
            if deepstack_input_embeds is not None and layer_idx < len(
                deepstack_input_embeds
            ):
                hidden_states = hidden_states + self._own_rows(
                    deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"]
                )
        hidden_states, _ = backbone.norm(hidden_states, residual)
        return self.group.all_gather(hidden_states, dim=0)

    def _layer(
        self,
        layer: nn.Module,
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
        qkv = self._rows_to_heads(*_column(attn.qkv_proj, replica.qkv, hidden_states))
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // attn.head_dim, attn.head_dim)
        q = attn.q_norm(q_by_head).view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // attn.head_dim, attn.head_dim)
        k = attn.k_norm(k_by_head).view(k.shape)
        q, k = attn.rotary_emb(positions, q, k)
        hidden_states = _row(
            attn.o_proj, replica.o, *self._heads_to_rows(attn.attn(q, k, v))
        )
        hidden_states, residual = layer.post_attention_layernorm(
            hidden_states, residual
        )
        # The MLP pairs each rank's gate-up block with its down block, in the
        # [own, *others] order they come in; no exchange, so no reordering.
        own, rest = _column(mlp.gate_up_proj, replica.gate_up, hidden_states)
        hidden_states = _row(
            mlp.down_proj, replica.down, mlp.act_fn(own), mlp.act_fn(rest)
        )
        return hidden_states, residual

    def _own_rows(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's block of the token rows."""
        return x.unflatten(0, (self.world, -1)).index_select(0, self.rank)[0]

    def _rows_to_heads(self, own: torch.Tensor, rest: torch.Tensor) -> torch.Tensor:
        """Head groups of this rank's rows -> this rank's head group of all rows."""
        blocks = torch.cat([own[None], rest]).index_select(0, self.send_order)
        return all_to_all(blocks, self.group).flatten(0, 1)

    def _heads_to_rows(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """This rank's head group of all rows -> every head group of its rows,
        as (own group, the others')."""
        blocks = all_to_all(x.unflatten(0, (self.world, -1)), self.group)
        return blocks.index_select(0, self.rank)[0], blocks.index_select(0, self.others)


def _in_out(linear: nn.Module) -> torch.Tensor:
    """A linear's own weight shard as [in, out]; the TPU linear method keeps
    most weights flipped that way."""
    weight = linear.weight
    return weight if getattr(linear, WEIGHT_FLIPPED_ATTR, False) else weight.t()


def _column(
    linear: nn.Module, rest: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A column-parallel projection: [rows, out] own, [tp - 1, rows, out] rest."""
    return torch.matmul(x, _in_out(linear)), torch.matmul(x, rest)


def _row(
    linear: nn.Module, rest: torch.Tensor, x_own: torch.Tensor, x_rest: torch.Tensor
) -> torch.Tensor:
    """A row-parallel projection of every rank's input block, summed."""
    rest_out = torch.matmul(x_rest.transpose(0, 1).flatten(1), rest.flatten(0, 1))
    return torch.matmul(x_own, _in_out(linear)) + rest_out


class _OtherShards(nn.Module):
    """The other TP ranks' shards of one decoder layer's projections, stacked
    in rank order as [tp - 1, in, out]."""

    def __init__(self, layer: nn.Module, others: list[int]) -> None:
        super().__init__()
        from vllm.distributed import tensor_model_parallel_all_gather
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        def gather(linear: nn.Module) -> torch.Tensor:
            if linear.bias is not None or not isinstance(
                linear.quant_method, UnquantizedLinearMethod
            ):
                raise ValueError(
                    "TPU_SP_PREFILL_MIN_TOKENS supports unquantized projections "
                    f"without bias, not {linear.prefix}."
                )
            shard = _in_out(linear).data.contiguous()
            every = tensor_model_parallel_all_gather(shard[None], dim=0)
            return every[others].contiguous()

        attn, mlp = layer.self_attn, layer.mlp
        for name, linear in (
            ("qkv", attn.qkv_proj),
            ("o", attn.o_proj),
            ("gate_up", mlp.gate_up_proj),
            ("down", mlp.down_proj),
        ):
            self.register_buffer(name, gather(linear), persistent=False)
