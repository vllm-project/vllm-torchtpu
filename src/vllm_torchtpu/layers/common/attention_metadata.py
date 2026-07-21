import functools
from dataclasses import dataclass, field
from typing import Any

import jax
import torch
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import \
    AttentionMetadataBuilder as BaseAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.cp_utils import get_total_cp_world_size

from vllm_torchtpu.layers.common.sequence_layout import (
    DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR, DEFAULT_SEQUENCE_LAYOUT_PROTOCOL,
    SequenceLayoutDescriptor, SequenceLayoutKind)


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "input_positions",
        "block_tables",
        "seq_lens",
        "query_start_loc",
        "request_distribution",
        "mamba_state_indices",
    ],
    meta_fields=[
        "sequence_layout_kind",
        "sequence_layout_protocol",
        "sequence_layout_version",
    ],
    drop_fields=["query_start_loc_cpu", "seq_lens_cpu"],
)
@dataclass
class AttentionMetadata(object):
    # (padded_total_num_scheduled_tokens,)
    input_positions: jax.Array
    # (max_num_seqs * max_num_blocks_per_req,)
    block_tables: jax.Array = None
    # (max_num_seqs,)
    seq_lens: jax.Array = None
    # (max_num_seqs + 1,)
    query_start_loc: jax.Array = None
    # (3,)
    request_distribution: jax.Array = None
    # (max_num_seqs,) int32 - physical slot id in the mamba kv-cache for the
    # request currently in each persistent-batch position. Compact-mamba uses
    # an explicit slot from its independent pool; unified align mode derives
    # the slot from the current entry in the device block table. None for
    # attention groups, non-mamba models, and Mamba modes whose native fallback
    # addresses block_tables[:, 0].
    mamba_state_indices: jax.Array | None = None
    sequence_layout_kind: str = SequenceLayoutKind.ALL.value
    sequence_layout_protocol: str = DEFAULT_SEQUENCE_LAYOUT_PROTOCOL
    sequence_layout_version: int = 1

    query_start_loc_cpu: Any = field(init=False)
    seq_lens_cpu: Any = field(init=False)


@dataclass
class AttentionMetadataBuilderContext:
    """Per-call inputs the TPU metadata builder reads from the runner.

    Stashed on TPUModelRunner before invoking _build_attention_metadata so
    builder.build() can pick them up. Only fields TPU populates differently from
    the parent runner (seq_lens / query_start_loc) live in TPU's own _cpu
    staging tensors, not parent's CpuGpuBuffers) or that have no
    `CommonAttentionMetadata` equivalent (num_reqs, use_max_model_len,
    start_index, request_distribution, position_ids_override) live here.
    """
    num_reqs: int
    start_index: int
    use_max_model_len: bool
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    request_distribution: torch.Tensor
    position_ids_override: torch.Tensor | None = None
    # Compact-mamba per-request recurrent-slot ids (device int32, length =
    # target_num_reqs). Only the mamba group's builder reads it; None when the
    # model has no mamba layers. See AttentionMetadata.mamba_state_indices.
    mamba_state_indices: torch.Tensor | None = None
    sequence_layout_descriptor: SequenceLayoutDescriptor = (
        DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR)


class AttentionMetadataBuilder(BaseAttentionMetadataBuilder):
    """Single shared metadata builder for every TPU attention group.
    """

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device, runner,
                 kv_cache_group_id):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.runner = runner
        self.kv_cache_group_id = kv_cache_group_id
        self.is_mamba_group = isinstance(kv_cache_spec, MambaSpec)
        self.target_block_size = getattr(self.kv_cache_spec, "block_size",
                                         runner.block_size)
        if self.is_mamba_group:
            self.target_block_size *= get_total_cp_world_size()
        # Only mamba/GDN layers consume physical state slot ids; attention
        # groups leave AttentionMetadata.mamba_state_indices None.

        block_table_obj = runner.input_batch.block_table[
            self.kv_cache_group_id]
        self.block_tables_cpu = torch.zeros(
            (runner.max_num_reqs, block_table_obj.max_num_blocks_per_req),
            dtype=torch.int32,
            device="cpu")

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        runner = self.runner
        ctx = runner._attn_metadata_builder_ctx
        target_num_reqs = common_attn_metadata.num_reqs
        block_table_obj = runner.input_batch.block_table[
            self.kv_cache_group_id]
        if ctx.use_max_model_len:
            target_num_blocks = block_table_obj.max_num_blocks_per_req
        else:
            assert runner.most_model_len is not None
            target_num_blocks = cdiv(runner.most_model_len,
                                     self.target_block_size)

        # `position_ids` is only used for the dummy run in dummy runs, where we
        # want to use fixed position IDs instead of copying from the CPU tensor
        # that gets updated every step.
        if ctx.position_ids_override is not None:
            block_tables_dev = torch.zeros(
                (target_num_reqs * target_num_blocks, ),
                dtype=torch.int32).to(runner.device)
            input_positions = ctx.position_ids_override
        else:
            block_tables = self.block_tables_cpu[:target_num_reqs, :
                                                 target_num_blocks]
            block_tables.zero_()
            source_block_tables = block_table_obj.get_cpu_tensor()
            block_tables[:ctx.num_reqs, :target_num_blocks] = (
                source_block_tables[ctx.start_index:ctx.start_index +
                                    ctx.num_reqs, :target_num_blocks])
            # Flatten on CPU before H2D to avoid device-side as_strided/reshape
            # materialization on every decode step.
            block_tables_dev = block_tables.reshape(-1).to(runner.device,
                                                           non_blocking=True)
            input_positions = runner.position_ids

        if not self.is_mamba_group:
            mamba_state_indices = None
        elif ctx.mamba_state_indices is not None:
            # Default compact-mamba path: the runner provides a per-request
            # physical slot id from the compact slot pool.
            mamba_state_indices = ctx.mamba_state_indices
        elif runner._unified_kv_layout:
            # The pool keys mamba state by vLLM block ids in every cache
            # mode (it has no compact slot pool). Derive the current physical
            # state
            # slot on device from the zero-padded `block_tables_dev` (padded
            # tail rows resolve to the null block, never a stale id — the
            # GDN op scans the full length every step), avoiding a
            # D2H -> CPU gather -> H2D dependency.
            block_tables_2d = block_tables_dev.reshape(target_num_reqs,
                                                       target_num_blocks)
            state_block_offsets = torch.clamp(
                (ctx.seq_lens - 1) // self.target_block_size,
                min=0,
                max=target_num_blocks - 1,
            ).to(torch.int64)
            mamba_state_indices = torch.gather(
                block_tables_2d,
                dim=1,
                index=state_block_offsets.unsqueeze(1),
            ).squeeze(1)
        else:
            mamba_state_indices = None

        return AttentionMetadata(
            input_positions=input_positions,
            block_tables=block_tables_dev,
            seq_lens=ctx.seq_lens,
            query_start_loc=ctx.query_start_loc,
            request_distribution=ctx.request_distribution,
            mamba_state_indices=mamba_state_indices,
            sequence_layout_kind=ctx.sequence_layout_descriptor.kind.value,
            sequence_layout_protocol=(ctx.sequence_layout_descriptor.protocol),
            sequence_layout_version=ctx.sequence_layout_descriptor.version,
        )
