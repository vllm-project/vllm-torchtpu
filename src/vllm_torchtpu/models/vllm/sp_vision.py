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
patches to 536 vs 1550 at 20992. The replica costs (tp - 1)/tp of the tower's
projection weights per rank.
"""

from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.sequence_parallel import (
    RowExchange,
    column,
    gather_other_shards,
    in_out,
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
        others = self.exchange.others
        self.blocks = nn.ModuleList(_BlockShards(b, others) for b in tower.blocks)
        self.merger = _MergerShards(tower.merger, others)
        self.deepstack_mergers = nn.ModuleList(
            _MergerShards(m, others) for m in tower.deepstack_merger_list
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
        own, rest = column(attn.qkv, shards.qkv, x)
        qkv = self.exchange.rows_to_heads(own + attn.qkv.bias, rest + shards.qkv_bias)
        context = _attend(attn, qkv[:, None], attention_inputs)[:, 0]
        own, rest = self.exchange.heads_to_rows(context)
        hidden_states = (
            hidden_states + _row(shards.proj, own, rest, attn.proj.bias)[:, None]
        )
        x = block.norm2(hidden_states)[:, 0]
        own, rest = column(mlp.linear_fc1, shards.fc1, x)
        own = mlp.act_fn(own + mlp.linear_fc1.bias)
        rest = mlp.act_fn(rest + shards.fc1_bias)
        return hidden_states + _row(shards.fc2, own, rest, mlp.linear_fc2.bias)[:, None]


def _attend(
    attn: nn.Module, qkv: torch.Tensor, attention_inputs: dict[str, Any]
) -> torch.Tensor:
    """Qwen2_5_VisionAttention.forward between its qkv and proj projections:
    [seq, 1, 3 * heads * head_dim] -> [seq, 1, heads * head_dim]."""
    heads = attn.num_attention_heads_per_partition
    head_dim = attn.hidden_size_per_attention_head
    seq_len, batch_size, _ = qkv.shape
    # "s b (three head head_dim) -> b s three head head_dim"
    qkv = qkv.view(seq_len, batch_size, 3, heads, head_dim).permute(1, 0, 2, 3, 4)
    qk, v = qkv[:, :, :2], qkv[:, :, 2]
    # "b s two head head_dim -> (two b) s head head_dim"
    qk = qk.permute(2, 0, 1, 3, 4).reshape(2 * batch_size, seq_len, heads, head_dim)
    qk = attn.apply_rotary_emb(
        qk.contiguous(),
        attention_inputs["rotary_pos_emb_cos"],
        attention_inputs["rotary_pos_emb_sin"],
    )
    q, k = qk.view(2, batch_size, seq_len, heads, head_dim).unbind(dim=0)
    context = attn.attn(
        query=q,
        key=k,
        value=v,
        cu_seqlens=attention_inputs["cu_seqlens"],
        max_seqlen=attention_inputs["max_seqlen"],
        sequence_lengths=attention_inputs["sequence_lengths"],
    )
    # "b s h d -> s b (h d)"
    return context.permute(1, 0, 2, 3).reshape(seq_len, batch_size, heads * head_dim)


def _merge(
    merger: nn.Module, shards: "_MergerShards", hidden_states: torch.Tensor
) -> torch.Tensor:
    """Qwen3_VisionPatchMerger.forward on this rank's rows."""
    if merger.use_postshuffle_norm:
        x = merger.norm(hidden_states.view(-1, merger.hidden_size))
    else:
        x = merger.norm(hidden_states).view(-1, merger.hidden_size)
    own, rest = column(merger.linear_fc1, shards.fc1, x)
    own = merger.act_fn(own + merger.linear_fc1.bias)
    rest = merger.act_fn(rest + shards.fc1_bias)
    return _row(shards.fc2, own, rest, merger.linear_fc2.bias)


def _row(
    weight: torch.Tensor, own: torch.Tensor, rest: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """A row-parallel projection of every rank's input block as one matmul over
    the whole input, so it is rounded once, like the unsharded layer."""
    x = torch.cat([own, rest.transpose(0, 1).flatten(1)], dim=-1)
    return torch.matmul(x, weight) + bias


def _replicate(module: nn.Module, others: list[int], **linears: nn.Module) -> None:
    """Register each linear's weight shards for this rank's rows: a
    column-parallel one's other shards as `<name>` [tp - 1, in, out] and their
    biases as `<name>_bias`; a row-parallel one's whole weight as `<name>` [in,
    out], in [own, *others] shard order."""
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
        own = in_out(linear).data
        rest = gather_other_shards(own, others)
        if isinstance(linear, ColumnParallelLinear):
            bias = gather_other_shards(linear.bias.data, others)
            module.register_buffer(name, rest, persistent=False)
            module.register_buffer(f"{name}_bias", bias[:, None], persistent=False)
        elif isinstance(linear, RowParallelLinear):
            whole = torch.cat([own[None], rest]).flatten(0, 1)
            module.register_buffer(name, whole, persistent=False)
        else:
            raise ValueError(f"{linear.prefix} is not a TP-parallel linear.")


class _BlockShards(nn.Module):
    """The other TP ranks' shards of one vision block's projections."""

    def __init__(self, block: nn.Module, others: list[int]) -> None:
        super().__init__()
        attn, mlp = block.attn, block.mlp
        _replicate(
            self,
            others,
            qkv=attn.qkv,
            proj=attn.proj,
            fc1=mlp.linear_fc1,
            fc2=mlp.linear_fc2,
        )


class _MergerShards(nn.Module):
    """The other TP ranks' shards of one patch merger's projections."""

    def __init__(self, merger: nn.Module, others: list[int]) -> None:
        super().__init__()
        _replicate(self, others, fc1=merger.linear_fc1, fc2=merger.linear_fc2)
