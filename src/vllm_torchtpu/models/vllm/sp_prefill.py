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
1/tp of the rows with every rank's weight shard, and the ranks exchange only
around attention: one all-to-all gives each rank its own head group for all
rows, which is exactly the TP attention input, so the attention kernel, its
metadata and the head-sharded KV pools stay as they are; a second all-to-all
returns the output rows to their owners. That moves 9x fewer bytes per layer
than the two all-reduces, at the price of (tp - 1) / tp more dense weight HBM
per rank, which comes out of the KV cache.

Opt in with ``TPU_SP_PREFILL_MIN_TOKENS``: token buckets at or above it run this
way and smaller ones (decode, short prompts) keep the TP graph, since streaming
the full weights once costs more than the all-reduces of a short step. The
choice is a branch on the token count, so each bucket compiles one of the two
graphs (see ``vllm_torchtpu.compilation.shape_variants``).
"""

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
from torch import nn

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.collectives import all_to_all
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
        device = backbone.norm.weight.device
        # A device tensor, not a Python int: every rank must compile the same
        # program, or torch_tpu pairs their collectives wrongly.
        self.register_buffer(
            "rank",
            torch.tensor([self.group.rank_in_group], dtype=torch.int32, device=device),
            persistent=False,
        )
        self.layers = nn.ModuleList(
            _DenseReplica(layer, self.world) for layer in backbone.layers
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
        replica: "_DenseReplica",
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Mirrors Qwen3DecoderLayer.forward and Qwen3Attention.forward, with the
        # projections on the replica and the reshards around attention.
        if residual is None:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
        else:
            hidden_states, residual = layer.input_layernorm(hidden_states, residual)
        attn = layer.self_attn
        qkv = self._rows_to_heads(F.linear(hidden_states, replica.qkv))
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // attn.head_dim, attn.head_dim)
        q = attn.q_norm(q_by_head).view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // attn.head_dim, attn.head_dim)
        k = attn.k_norm(k_by_head).view(k.shape)
        q, k = attn.rotary_emb(positions, q, k)
        hidden_states = F.linear(self._heads_to_rows(attn.attn(q, k, v)), replica.o)
        hidden_states, residual = layer.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = layer.mlp.act_fn(F.linear(hidden_states, replica.gate_up))
        return F.linear(hidden_states, replica.down), residual

    def _own_rows(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's block of the token rows."""
        return x.unflatten(0, (self.world, -1)).index_select(0, self.rank)[0]

    def _rows_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[rows / tp, tp * width] -> [rows, width]: rank d gets head group d."""
        blocks = x.unflatten(-1, (self.world, -1)).transpose(0, 1)
        return all_to_all(blocks, self.group).flatten(0, 1)

    def _heads_to_rows(self, x: torch.Tensor) -> torch.Tensor:
        """[rows, width] -> [rows / tp, tp * width]: rank d gets row block d."""
        blocks = all_to_all(x.unflatten(0, (self.world, -1)), self.group)
        return blocks.transpose(0, 1).flatten(1)


class _DenseReplica(nn.Module):
    """Every TP rank's shard of one decoder layer's projections.

    Column-parallel weights stack the shards along the output rows, so the
    output columns come rank by rank, as ``_rows_to_heads`` splits them;
    row-parallel weights stack them along the input columns, as
    ``_heads_to_rows`` and the gate-up activation concatenate them.
    """

    def __init__(self, layer: nn.Module, world: int) -> None:
        super().__init__()
        from vllm.distributed import tensor_model_parallel_all_gather
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        attn, mlp = layer.self_attn, layer.mlp
        for linear in (attn.qkv_proj, attn.o_proj, mlp.gate_up_proj, mlp.down_proj):
            if linear.bias is not None or not isinstance(
                linear.quant_method, UnquantizedLinearMethod
            ):
                raise ValueError(
                    "TPU_SP_PREFILL_MIN_TOKENS supports unquantized projections "
                    f"without bias, not {linear.prefix}."
                )

        def gather(weight: torch.Tensor, dim: int) -> torch.Tensor:
            return tensor_model_parallel_all_gather(weight.data, dim=dim)

        # Each rank's gate-up rows are [gate; up]; regroup them as all gates,
        # then all ups, for the activation that splits its input in halves.
        gate_up = gather(mlp.gate_up_proj.weight, 0)
        gate_up = gate_up.unflatten(0, (world, 2, -1)).transpose(0, 1).flatten(0, 2)
        self.register_buffer("qkv", gather(attn.qkv_proj.weight, 0), persistent=False)
        self.register_buffer("o", gather(attn.o_proj.weight, 1), persistent=False)
        self.register_buffer("gate_up", gate_up.contiguous(), persistent=False)
        self.register_buffer("down", gather(mlp.down_proj.weight, 1), persistent=False)
