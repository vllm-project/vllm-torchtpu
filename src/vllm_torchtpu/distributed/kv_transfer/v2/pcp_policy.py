from __future__ import annotations

from dataclasses import dataclass

from .common import LayerType, check_non_negative, check_positive, head_range
from .layout import KVCacheRegion
from .metadata import (ConnectorMetadataV2, HeadMapping, KVParallelLayout,
                       LocalDecodeAllocation, PullMeta, SourceBlockRef)


@dataclass(frozen=True)
class PcpTokenTransfer:
    source_block: int
    source_token: int
    destination_block: int
    destination_token: int
    num_tokens: int

    def __post_init__(self) -> None:
        check_non_negative("source_block", self.source_block)
        check_non_negative("source_token", self.source_token)
        check_non_negative("destination_block", self.destination_block)
        check_non_negative("destination_token", self.destination_token)
        check_positive("num_tokens", self.num_tokens)


class PcpReshardingPolicy:
    """Full-attention PCP policy for rank/block-owner based resharding."""

    @staticmethod
    def is_enabled(layout: KVParallelLayout) -> bool:
        return layout.full_attn_pcp_size > 1

    @staticmethod
    def validate_layouts(
        source_layout: KVParallelLayout,
        destination_layout: KVParallelLayout,
    ) -> None:
        if source_layout.linear_attn_pcp_size != 1:
            raise ValueError("linear attention PCP is not supported")
        if PcpReshardingPolicy.is_enabled(source_layout):
            if source_layout.full_attn_pcp_size != source_layout.full_attn_tp_size:
                raise ValueError("source full attention PCP requires "
                                 "full_attn_pcp_size == full_attn_tp_size")
        if (destination_layout.full_attn_pcp_size != 1
                or destination_layout.linear_attn_pcp_size != 1):
            raise ValueError("decode-side PCP layout is not supported")

    @staticmethod
    def build_head_mappings(
        total_heads: int,
        source_layout: KVParallelLayout,
        destination_layout: KVParallelLayout,
        destination_tp_rank: int,
    ) -> dict[int, tuple[HeadMapping, ...]]:
        if total_heads == 0:
            return {}

        destination_range = head_range(
            total_heads,
            destination_layout.full_attn_tp_size,
            destination_tp_rank,
        )
        mappings = tuple(
            HeadMapping(
                global_head=head,
                source_head=head,
                destination_head=head - destination_range.start,
            ) for head in destination_range)
        return {
            p_rank: mappings
            for p_rank in range(source_layout.full_attn_pcp_size)
        }

    @staticmethod
    def build_block_refs(
        source_layout: KVParallelLayout,
        block_ids: tuple[int, ...],
    ) -> dict[int, tuple[SourceBlockRef, ...]]:
        block_refs_by_rank: dict[int, list[SourceBlockRef]] = {
            p_rank: []
            for p_rank in range(source_layout.full_attn_pcp_size)
        }
        for block_index, block_id in enumerate(block_ids):
            for p_rank in range(source_layout.full_attn_pcp_size):
                logical_block_index = (
                    block_index * source_layout.full_attn_pcp_size + p_rank)
                block_refs_by_rank[p_rank].append(
                    SourceBlockRef(
                        logical_block_index=logical_block_index,
                        block_id=block_id,
                    ))
        return {
            p_rank: tuple(block_refs)
            for p_rank, block_refs in block_refs_by_rank.items()
        }

    @staticmethod
    def build_full_attention_token_transfers(
        metadata: ConnectorMetadataV2,
        destination: LocalDecodeAllocation,
        pull_meta: PullMeta,
        p_rank: int,
        source_region: KVCacheRegion,
        destination_region: KVCacheRegion,
    ) -> tuple[PcpTokenTransfer, ...]:
        if source_region.layer_type != LayerType.FULL_ATTN:
            raise ValueError(f"PCP policy only supports full attention, got "
                             f"{source_region.layer_type}")
        if destination_region.layer_type != LayerType.FULL_ATTN:
            raise ValueError(f"destination region must be full attention, got "
                             f"{destination_region.layer_type}")

        source_num_tokens = metadata.fa_num_tokens
        if source_num_tokens is None:
            source_num_tokens = (len(metadata.fa_block_ids) *
                                 metadata.kv_source_layout.full_attn_pcp_size *
                                 source_region.lowering_units_per_block)
        destination_num_tokens = destination.fa_num_tokens
        if destination_num_tokens is None:
            destination_num_tokens = source_num_tokens
        if source_num_tokens != destination_num_tokens:
            raise ValueError(
                "source and destination token counts must match, got "
                f"{source_num_tokens} and {destination_num_tokens}")

        token_transfers: list[PcpTokenTransfer] = []
        source_block_size = source_region.lowering_units_per_block
        destination_block_size = destination_region.lowering_units_per_block
        source_token_offset = metadata.fa_token_offset
        destination_token_offset = destination.fa_token_offset
        source_window_start = source_token_offset
        source_window_end = source_token_offset + source_num_tokens
        for block_ref in pull_meta.fa_block_refs_by_rank.get(p_rank, ()):
            block_start = block_ref.logical_block_index * source_block_size
            block_end = block_start + source_block_size
            logical_token_start = max(block_start, source_window_start)
            logical_token_end = min(block_end, source_window_end)
            if logical_token_start >= logical_token_end:
                continue

            destination_logical_token = (
                destination_token_offset +
                (logical_token_start - source_token_offset))
            destination_block_index = (destination_logical_token //
                                       destination_block_size)
            if destination_block_index >= len(destination.fa_block_ids):
                raise ValueError(
                    "destination block ids do not cover num_tokens")
            destination_token = (destination_logical_token %
                                 destination_block_size)
            destination_block = destination.fa_block_ids[
                destination_block_index]

            token_transfers.append(
                PcpTokenTransfer(
                    source_block=block_ref.block_id,
                    source_token=logical_token_start - block_start,
                    destination_block=destination_block,
                    destination_token=destination_token,
                    num_tokens=logical_token_end - logical_token_start,
                ))

        return tuple(token_transfers)
