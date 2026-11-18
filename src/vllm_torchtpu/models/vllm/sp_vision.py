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
"""Sequence-parallel Qwen3-VL vision tower.

The TP tower replicates every patch row on all ranks and all-reduces the
[patches, 1152] activation twice per layer: on a 20900-patch image that is 37%
of its 96 ms, and the elementwise work is done 8 times over. Here each rank
embeds and runs only its 1/tp of the rows, with every rank's weight shards
(`distributed.sequence_parallel`), and the ranks exchange only around
attention, which stays the TP kernel on the rank's two heads of every row with
the same per-image segments. The patch merger groups 4 consecutive rows, so
row blocks of whole merge groups keep it local; the merged features are
all-gathered once at the end. Each rank also moves only its rows of the pixels
to the device.

Opt in with ``TPU_SP_VISION``; it serves every tower call, eager and the
compiled encoder buckets alike. A one-layer bench on v7x-8 has it ahead of TP
at every size, from 66 vs 125 us per layer (excluding attention) at 1024
patches to 536 vs 1550 at 20992. Every rank holds every rank's shard of the
tower's projections, its own included, stacked in rank order: a projection's
output is then the all-to-all's input as is.
"""

from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.sequence_parallel import (
    RowExchange,
    gather_shards,
    in_out,
    row_parallel,
)
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

_UPSTREAM_FORWARD = None


def maybe_patch_qwen3_vl_sp_vision(
    model_config: Optional["ModelConfig"] = None,
) -> bool | None:
    """Route the Qwen3-VL vision tower to sequence parallel."""
    global _UPSTREAM_FORWARD
    if not envs.TPU_SP_VISION:
        return None
    if model_config is None:
        return False
    from vllm.model_executor.models import qwen3_vl

    from vllm_torchtpu.models.vllm.qwen3_vl_patch import _has_qwen3_vl_vision

    if not _has_qwen3_vl_vision(model_config):
        return False
    if _UPSTREAM_FORWARD is None:
        _UPSTREAM_FORWARD = qwen3_vl.Qwen3_VisionTransformer.forward
        qwen3_vl.Qwen3_VisionTransformer.forward = _vision_forward
        logger.info("Applied TPU patch: Qwen3-VL sequence-parallel vision tower.")
    return None


def _vision_forward(
    self,
    x: torch.Tensor,
    grid_thw: torch.Tensor | list[list[int]],
    *,
    encoder_metadata: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    sp = getattr(self, "sp_vision", None)
    if sp is not None:
        return sp(self, x, grid_thw, encoder_metadata)
    return _UPSTREAM_FORWARD(self, x, grid_thw, encoder_metadata=encoder_metadata)


def maybe_install(model: nn.Module) -> None:
    """Give the loaded model's vision tower its weight replica; no-op when off."""
    if not envs.TPU_SP_VISION:
        return
    if _UPSTREAM_FORWARD is None:
        raise ValueError(
            "TPU_SP_VISION is set, but the model has no Qwen3-VL vision tower."
        )
    from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer

    towers = [m for m in model.modules() if isinstance(m, Qwen3_VisionTransformer)]
    if len(towers) != 1:
        raise ValueError(
            f"TPU_SP_VISION needs one Qwen3-VL vision tower, got {len(towers)}."
        )
    tower = towers[0]
    tower.sp_vision = SequenceParallelVision(tower)
    logger.info(
        "Sequence-parallel vision tower: %.2f GiB of replicated weights per rank.",
        sum(b.numel() * b.element_size() for b in tower.sp_vision.buffers()) / 2**30,
    )


class SequenceParallelVision(nn.Module):
    """The vision tower forward over this rank's rows with every rank's weights."""

    def __init__(self, tower: nn.Module) -> None:
        super().__init__()
        self.exchange = RowExchange(tower.device)
        if self.exchange.world == 1 or tower.tp_size != self.exchange.world:
            raise ValueError(
                "TPU_SP_VISION needs a TP-sharded vision tower with TP > 1."
            )
        # Row blocks of whole merge groups, so the patch merger stays local.
        self.row_multiple = self.exchange.world * tower.spatial_merge_unit
        self.blocks = nn.ModuleList(_BlockShards(b) for b in tower.blocks)
        self.merger = _MergerShards(tower.merger)
        self.deepstack_mergers = nn.ModuleList(
            _MergerShards(m) for m in tower.deepstack_merger_list
        )

    def forward(
        self,
        tower: nn.Module,
        x: torch.Tensor,
        grid_thw: torch.Tensor | list[list[int]] | None,
        metadata: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        # Mirrors Qwen3_VisionTransformer.forward on this rank's rows.
        exchange = self.exchange
        if metadata is None:
            assert grid_thw is not None
            grid = grid_thw if isinstance(grid_thw, list) else grid_thw.tolist()
            metadata = tower.prepare_encoder_metadata(grid)
        num_patches = x.shape[0]
        padded = -(-num_patches // self.row_multiple) * self.row_multiple

        def pad(t: torch.Tensor) -> torch.Tensor:
            return F.pad(t, (0, 0, 0, padded - num_patches))

        x = pad(x)
        if x.device.type == "cpu":
            # A host slice, so each rank moves only its rows to the device. A
            # blocking copy: the padded host tensor dies with this call, and a
            # TorchTPU non-blocking copy does not keep its source alive.
            rows = padded // exchange.world
            start = exchange.group.rank_in_group * rows
            x = x[start : start + rows].to(tower.device)
        else:
            x = exchange.own_rows(x)
        hidden_states = tower.patch_embed(tower.input_norm(x, tower.dtype))
        hidden_states = hidden_states + exchange.own_rows(pad(metadata["pos_embeds"]))
        hidden_states = hidden_states.unsqueeze(1)
        # Padding rows get a zero rotation and their own attention segment
        # (past the last cu_seqlens entry), so they never touch real rows.
        attention_inputs = dict(
            cu_seqlens=metadata["cu_seqlens"],
            rotary_pos_emb_cos=pad(metadata["rotary_pos_emb_cos"]),
            rotary_pos_emb_sin=pad(metadata["rotary_pos_emb_sin"]),
            max_seqlen=metadata["max_seqlen"],
            sequence_lengths=metadata.get("sequence_lengths"),
        )
        merged = []
        for layer_num, (block, shards) in enumerate(
            zip(tower.blocks, self.blocks, strict=True)
        ):
            hidden_states = self._block(block, shards, hidden_states, attention_inputs)
            if layer_num in tower.deepstack_visual_indexes:
                idx = tower.deepstack_visual_indexes.index(layer_num)
                merged.append(
                    _merge(
                        tower.deepstack_merger_list[idx],
                        self.deepstack_mergers[idx],
                        hidden_states,
                    )
                )
        merged.insert(0, _merge(tower.merger, self.merger, hidden_states))
        out = exchange.group.all_gather(torch.cat(merged, dim=1), dim=0)
        return out[: num_patches // tower.spatial_merge_unit]

    def _block(
        self,
        block: nn.Module,
        shards: "_BlockShards",
        hidden_states: torch.Tensor,
        attention_inputs: dict[str, Any],
    ) -> torch.Tensor:
        # Mirrors Qwen3_VisionBlock.forward and the projections of
        # Qwen2_5_VisionAttention.forward; rows are [rows, 1, hidden].
        attn, mlp = block.attn, block.mlp
        x = block.norm1(hidden_states)[:, 0]
        qkv = torch.matmul(x, shards.qkv) + shards.qkv_bias
        context = _attend(attn, self.exchange.rows_to_heads(qkv), attention_inputs)
        out = row_parallel(self.exchange.heads_to_rows(context), shards.proj)
        hidden_states = hidden_states + (out + attn.proj.bias)[:, None]
        x = block.norm2(hidden_states)[:, 0]
        x = mlp.act_fn(torch.matmul(x, shards.fc1) + shards.fc1_bias)
        out = row_parallel(x, shards.fc2) + mlp.linear_fc2.bias
        return hidden_states + out[:, None]


def _attend(
    attn: nn.Module, qkv: torch.Tensor, attention_inputs: dict[str, Any]
) -> torch.Tensor:
    """Qwen2_5_VisionAttention.forward between its qkv and proj projections:
    [seq, 3 * heads * head_dim] -> [seq, heads * head_dim]. q, k and v stay
    views of qkv ("s (three head head_dim)"), rotated and read in place."""
    heads = attn.num_attention_heads_per_partition
    head_dim = attn.hidden_size_per_attention_head
    q, k, v = qkv.unflatten(-1, (3, heads, head_dim))[None].unbind(2)
    cos = attention_inputs["rotary_pos_emb_cos"]
    sin = attention_inputs["rotary_pos_emb_sin"]
    context = attn.attn(
        query=attn.apply_rotary_emb(q, cos, sin),
        key=attn.apply_rotary_emb(k, cos, sin),
        value=v,
        cu_seqlens=attention_inputs["cu_seqlens"],
        max_seqlen=attention_inputs["max_seqlen"],
        sequence_lengths=attention_inputs["sequence_lengths"],
    )
    return context[0].flatten(1)


def _merge(
    merger: nn.Module, shards: "_MergerShards", hidden_states: torch.Tensor
) -> torch.Tensor:
    """Qwen3_VisionPatchMerger.forward on this rank's rows."""
    if merger.use_postshuffle_norm:
        x = merger.norm(hidden_states.view(-1, merger.hidden_size))
    else:
        x = merger.norm(hidden_states).view(-1, merger.hidden_size)
    x = merger.act_fn(torch.matmul(x, shards.fc1) + shards.fc1_bias)
    return row_parallel(x, shards.fc2) + merger.linear_fc2.bias


def _replicate(module: nn.Module, **linears: nn.Module) -> None:
    """Register every TP rank's shard of each linear, in rank order, as
    `<name>` [tp, in, out], and a column-parallel one's bias shards as
    `<name>_bias` [tp, 1, out]: the column-parallel output is then every head
    group of the rows in the order the all-to-all sends them."""
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        RowParallelLinear,
        UnquantizedLinearMethod,
    )

    for name, linear in linears.items():
        if not isinstance(linear.quant_method, UnquantizedLinearMethod):
            raise ValueError(
                f"TPU_SP_VISION supports unquantized projections, not {linear.prefix}."
            )
        if isinstance(linear, ColumnParallelLinear):
            bias = gather_shards(linear.bias.data)[:, None]
            module.register_buffer(f"{name}_bias", bias, persistent=False)
        elif not isinstance(linear, RowParallelLinear):
            raise ValueError(f"{linear.prefix} is not a TP-parallel linear.")
        module.register_buffer(
            name, gather_shards(in_out(linear).data), persistent=False
        )


class _BlockShards(nn.Module):
    """Every TP rank's shard of one vision block's projections."""

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        attn, mlp = block.attn, block.mlp
        _replicate(
            self,
            qkv=attn.qkv,
            proj=attn.proj,
            fc1=mlp.linear_fc1,
            fc2=mlp.linear_fc2,
        )


class _MergerShards(nn.Module):
    """Every TP rank's shard of one patch merger's projections."""

    def __init__(self, merger: nn.Module) -> None:
        super().__init__()
        _replicate(self, fc1=merger.linear_fc1, fc2=merger.linear_fc2)
