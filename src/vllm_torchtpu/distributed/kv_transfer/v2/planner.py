from __future__ import annotations

from dataclasses import dataclass

from vllm_torchtpu.logger import init_logger

from .common import (LayerType, TensorLayout, check_non_negative,
                     check_positive, head_range, is_linear_state_layer,
                     linear_rank)
from .layout import HeadSegment, KVCacheRegion
from .metadata import (ConnectorMetadataV2, HeadMapping, LocalDecodeAllocation,
                       PullMeta, RankTransferPlan, StridedSegmentOp,
                       TpKVTopology)
from .pcp_policy import PcpReshardingPolicy

logger = init_logger(__name__)


@dataclass(frozen=True)
class BlockSelection:
    source_blocks: tuple[int, ...]
    destination_blocks: tuple[int, ...]
    num_tokens: int
    source_token_offset: int = 0
    destination_token_offset: int = 0


@dataclass(frozen=True)
class LayoutOffset:
    offset_bytes: int
    stride_bytes: int
    segment_bytes: int


class TPTransferPlanner:
    """Interface for heterogeneous TP pull-meta planning and pointer lowering."""

    def build_pull_meta(
        self,
        metadata: ConnectorMetadataV2,
        topology: TpKVTopology,
    ) -> PullMeta:
        raise NotImplementedError

    def lower(
        self,
        metadata: ConnectorMetadataV2,
        topology: TpKVTopology,
        destination: LocalDecodeAllocation,
        pull_meta: PullMeta,
    ) -> dict[int, RankTransferPlan]:
        raise NotImplementedError


class ContiguousHeadTPTransferPlanner(TPTransferPlanner):
    """Planner for contiguous global-head ownership.

    This planner assumes each TP rank owns a contiguous global-head range.
    PCP ranks are treated as token partitions, so every PCP rank that owns the
    intersected TP head range is included in the pull plan.
    """

    def build_pull_meta(
        self,
        metadata: ConnectorMetadataV2,
        topology: TpKVTopology,
    ) -> PullMeta:
        PcpReshardingPolicy.validate_layouts(
            metadata.kv_source_layout,
            topology.local_layout,
        )
        if PcpReshardingPolicy.is_enabled(metadata.kv_source_layout):
            fa_mappings = PcpReshardingPolicy.build_head_mappings(
                total_heads=topology.total_num_kv_heads,
                source_layout=metadata.kv_source_layout,
                destination_layout=topology.local_layout,
                destination_tp_rank=topology.full_attn_tp_rank,
            )
            fa_block_refs = PcpReshardingPolicy.build_block_refs(
                source_layout=metadata.kv_source_layout,
                block_ids=metadata.fa_block_ids,
            )
        else:
            fa_mappings = self._map_heads_to_source_ranks(
                total_heads=topology.total_num_kv_heads,
                source_pcp_size=metadata.kv_source_layout.full_attn_pcp_size,
                source_tp_size=metadata.kv_source_layout.full_attn_tp_size,
                destination_tp_size=topology.local_layout.full_attn_tp_size,
                destination_tp_rank=topology.full_attn_tp_rank,
            )
            fa_block_refs = {}
        mamba_key_mappings = self._map_heads_to_source_ranks(
            total_heads=topology.total_num_mamba_key_heads,
            source_pcp_size=metadata.kv_source_layout.linear_attn_pcp_size,
            source_tp_size=metadata.kv_source_layout.linear_attn_tp_size,
            destination_tp_size=topology.local_layout.linear_attn_tp_size,
            destination_tp_rank=topology.linear_attn_tp_rank,
        )
        mamba_value_mappings = self._map_heads_to_source_ranks(
            total_heads=topology.total_num_mamba_heads,
            source_pcp_size=metadata.kv_source_layout.linear_attn_pcp_size,
            source_tp_size=metadata.kv_source_layout.linear_attn_tp_size,
            destination_tp_size=topology.local_layout.linear_attn_tp_size,
            destination_tp_rank=topology.linear_attn_tp_rank,
        )
        fa_heads = self._global_heads_by_rank(fa_mappings)
        mamba_key_heads = self._global_heads_by_rank(mamba_key_mappings)
        mamba_value_heads = self._global_heads_by_rank(mamba_value_mappings)
        p_ranks = tuple(
            sorted(
                set(fa_mappings) | set(mamba_key_mappings)
                | set(mamba_value_mappings)))

        pull_meta = PullMeta(
            req_id=metadata.req_id,
            p_ranks=p_ranks,
            fa_heads_by_rank=fa_heads,
            mamba_key_heads_by_rank=mamba_key_heads,
            mamba_value_heads_by_rank=mamba_value_heads,
            fa_head_mappings_by_rank=fa_mappings,
            mamba_key_head_mappings_by_rank=mamba_key_mappings,
            mamba_value_head_mappings_by_rank=mamba_value_mappings,
            fa_source_block_ids=metadata.fa_block_ids,
            mamba_source_block_ids=metadata.mamba_block_ids,
            fa_block_refs_by_rank=fa_block_refs,
        )
        logger.info(
            "TPUConnectorV2 logical pull meta built | req_id=%s | "
            "d_tp_rank=%s | "
            "p_ranks=%s | fa_blocks=%s | mamba_blocks=%s | "
            "fa_heads_by_rank=%s | "
            "mamba_state0_q_key_heads_by_rank=%s | "
            "mamba_state0_k_key_heads_by_rank=%s | "
            "mamba_state0_v_value_heads_by_rank=%s | "
            "mamba_state1_value_heads_by_rank=%s | "
            "fa_block_refs_by_rank=%s",
            metadata.req_id,
            topology.tp_rank,
            pull_meta.p_ranks,
            self._block_ids_summary(pull_meta.fa_source_block_ids),
            self._block_ids_summary(pull_meta.mamba_source_block_ids),
            pull_meta.fa_heads_by_rank,
            pull_meta.mamba_key_heads_by_rank,
            pull_meta.mamba_key_heads_by_rank,
            pull_meta.mamba_value_heads_by_rank,
            pull_meta.mamba_value_heads_by_rank,
            self._block_refs_summary(pull_meta.fa_block_refs_by_rank),
        )
        return pull_meta

    def lower(
        self,
        metadata: ConnectorMetadataV2,
        topology: TpKVTopology,
        destination: LocalDecodeAllocation,
        pull_meta: PullMeta,
    ) -> dict[int, RankTransferPlan]:
        if topology.block_size != destination.block_size:
            raise ValueError(
                "topology.block_size must match destination.block_size, got "
                f"{topology.block_size} and {destination.block_size}")
        ContiguousHeadTPTransferPlanner._validate_block_size_relation(
            metadata.block_size, destination.block_size)

        plans = {
            p_rank: self._lower_rank(metadata, destination, pull_meta, p_rank)
            for p_rank in pull_meta.p_ranks
        }
        (
            fa_ops_by_head,
            mamba_state0_q_ops_by_key_head,
            mamba_state0_k_ops_by_key_head,
            mamba_state0_v_ops_by_value_head,
            mamba_state1_ops_by_value_head,
        ) = self._ops_by_layer_type_and_head(plans)
        logger.info(
            "TPUConnectorV2 physical lowering summary | req_id=%s | "
            "d_rank=%s | "
            "d_tp_rank=%s | p_ranks=%s | total_ops=%d | ops_by_p_rank=%s | "
            "fa_ops_by_head=%s | "
            "mamba_state0_q_ops_by_key_head=%s | "
            "mamba_state0_k_ops_by_key_head=%s | "
            "mamba_state0_v_ops_by_value_head=%s | "
            "mamba_state1_ops_by_value_head=%s",
            metadata.req_id,
            destination.rank,
            topology.tp_rank,
            tuple(plans),
            sum(len(plan.ops) for plan in plans.values()),
            {
                p_rank: len(plan.ops)
                for p_rank, plan in plans.items()
            },
            fa_ops_by_head,
            mamba_state0_q_ops_by_key_head,
            mamba_state0_k_ops_by_key_head,
            mamba_state0_v_ops_by_value_head,
            mamba_state1_ops_by_value_head,
        )
        return plans

    @staticmethod
    def _validate_block_size_relation(
        source_block_size: int,
        destination_block_size: int,
    ) -> None:
        if destination_block_size < source_block_size:
            raise ValueError(
                "destination block_size must be >= source block_size")
        if destination_block_size % source_block_size != 0:
            raise ValueError(
                "destination block_size must be divisible by source block_size"
            )

    @staticmethod
    def _validate_region_block_size_relation(
        source_region: KVCacheRegion,
        destination_region: KVCacheRegion,
    ) -> None:
        if source_region.block_size is None or destination_region.block_size is None:
            return
        ContiguousHeadTPTransferPlanner._validate_block_size_relation(
            source_region.block_size, destination_region.block_size)

    @staticmethod
    def _validate_full_attention_layout(region: KVCacheRegion) -> None:
        if region.layer_type != LayerType.FULL_ATTN:
            return
        if region.block_size is None:
            raise ValueError("full attention cache requires block_size")
        if region.layout == TensorLayout.TOKEN_FIRST:
            assert region.token_first_layout is not None
            return
        if region.layout == TensorLayout.HEAD_FIRST:
            _ = region.token_head_bytes
            return
        if region.layout == TensorLayout.BLOCKS_FIRST:
            if region.num_heads != 1:
                raise ValueError(
                    "BLOCKS_FIRST full attention head slicing is not supported"
                )
            return
        raise ValueError(f"unsupported full attention layout: {region.layout}")

    @staticmethod
    def _validate_full_attention_layout_pair(
        source_region: KVCacheRegion,
        destination_region: KVCacheRegion,
    ) -> None:
        ContiguousHeadTPTransferPlanner._validate_full_attention_layout(
            source_region)
        ContiguousHeadTPTransferPlanner._validate_full_attention_layout(
            destination_region)

    def _map_heads_to_source_ranks(
        self,
        total_heads: int,
        source_pcp_size: int,
        source_tp_size: int,
        destination_tp_size: int,
        destination_tp_rank: int,
    ) -> dict[int, tuple[HeadMapping, ...]]:
        if total_heads == 0:
            return {}

        destination_range = head_range(total_heads, destination_tp_size,
                                       destination_tp_rank)
        mappings_by_rank: dict[int, tuple[HeadMapping, ...]] = {}
        source_tp_ranks = self._source_tp_ranks_for_destination_heads(
            total_heads=total_heads,
            source_tp_size=source_tp_size,
            destination_tp_size=destination_tp_size,
            destination_tp_rank=destination_tp_rank,
            destination_range=destination_range,
        )

        for source_tp_rank in source_tp_ranks:
            source_range = head_range(total_heads, source_tp_size,
                                      source_tp_rank)
            overlap = self._range_intersection(destination_range, source_range)
            if not overlap:
                continue

            mappings = tuple(
                HeadMapping(
                    global_head=head,
                    source_head=head - source_range.start,
                    destination_head=head - destination_range.start,
                ) for head in overlap)
            for pcp_rank in range(source_pcp_size):
                p_rank = linear_rank(pcp_rank, source_tp_rank, source_tp_size)
                mappings_by_rank[p_rank] = mappings

        return mappings_by_rank

    @staticmethod
    def _source_tp_ranks_for_destination_heads(
        *,
        total_heads: int,
        source_tp_size: int,
        destination_tp_size: int,
        destination_tp_rank: int,
        destination_range: range,
    ) -> tuple[int, ...]:
        if total_heads >= source_tp_size:
            return tuple(range(source_tp_size))

        if source_tp_size % total_heads != 0:
            raise ValueError(f"tp_size={source_tp_size} is not divisible by "
                             f"total_heads={total_heads}")

        if source_tp_size == destination_tp_size:
            return (destination_tp_rank, )

        tp_ranks_per_head = source_tp_size // total_heads
        return tuple(head * tp_ranks_per_head for head in destination_range)

    @staticmethod
    def _global_heads_by_rank(
        mappings_by_rank: dict[int, tuple[HeadMapping, ...]]
    ) -> dict[int, tuple[int, ...]]:
        return {
            rank: tuple(mapping.global_head for mapping in mappings)
            for rank, mappings in mappings_by_rank.items()
        }

    @staticmethod
    def _block_ids_summary(
            block_ids: tuple[int, ...]) -> dict[str, int | None]:
        if not block_ids:
            return {"count": 0, "first": None, "last": None}
        return {
            "count": len(block_ids),
            "first": block_ids[0],
            "last": block_ids[-1],
        }

    @staticmethod
    def _block_refs_summary(
        block_refs_by_rank: dict[int, tuple],
    ) -> dict[int, dict[str, tuple[int, int] | int | None]]:
        summary = {}
        for rank, block_refs in block_refs_by_rank.items():
            if not block_refs:
                summary[rank] = {"count": 0, "first": None, "last": None}
                continue
            summary[rank] = {
                "count":
                len(block_refs),
                "first": (
                    block_refs[0].logical_block_index,
                    block_refs[0].block_id,
                ),
                "last": (
                    block_refs[-1].logical_block_index,
                    block_refs[-1].block_id,
                ),
            }
        return summary

    @staticmethod
    def _ops_by_layer_type_and_head(
        plans: dict[int, RankTransferPlan],
    ) -> tuple[dict[int, int], dict[int, int], dict[int, int], dict[int, int],
               dict[int, int]]:
        fa_ops_by_head: dict[int, int] = {}
        mamba_state0_q_ops_by_key_head: dict[int, int] = {}
        mamba_state0_k_ops_by_key_head: dict[int, int] = {}
        mamba_state0_v_ops_by_value_head: dict[int, int] = {}
        mamba_state1_ops_by_value_head: dict[int, int] = {}
        for plan in plans.values():
            for op in plan.ops:
                if op.global_head is None:
                    continue
                if op.layer_type == LayerType.FULL_ATTN:
                    fa_ops_by_head[op.global_head] = (
                        fa_ops_by_head.get(op.global_head, 0) + 1)
                elif is_linear_state_layer(op.layer_type):
                    if op.segment_name == "q":
                        target = mamba_state0_q_ops_by_key_head
                    elif op.segment_name == "k":
                        target = mamba_state0_k_ops_by_key_head
                    elif op.segment_name == "v":
                        target = mamba_state0_v_ops_by_value_head
                    else:
                        target = mamba_state1_ops_by_value_head
                    target[op.global_head] = target.get(op.global_head, 0) + 1
        return (
            dict(sorted(fa_ops_by_head.items())),
            dict(sorted(mamba_state0_q_ops_by_key_head.items())),
            dict(sorted(mamba_state0_k_ops_by_key_head.items())),
            dict(sorted(mamba_state0_v_ops_by_value_head.items())),
            dict(sorted(mamba_state1_ops_by_value_head.items())),
        )

    @staticmethod
    def _range_intersection(left: range, right: range) -> range:
        start = max(left.start, right.start)
        stop = min(left.stop, right.stop)
        if start >= stop:
            return range(0)
        return range(start, stop)

    def _lower_rank(
        self,
        metadata: ConnectorMetadataV2,
        destination: LocalDecodeAllocation,
        pull_meta: PullMeta,
        p_rank: int,
    ) -> RankTransferPlan:
        source_regions = metadata.kv_caches[p_rank]
        ops: list[StridedSegmentOp] = []
        pcp_enabled = PcpReshardingPolicy.is_enabled(metadata.kv_source_layout)
        for layer_name, source_region in source_regions.items():
            destination_region = destination.kv_caches[layer_name]
            if pcp_enabled and source_region.layer_type == LayerType.FULL_ATTN:
                self._validate_full_attention_layout_pair(
                    source_region, destination_region)
                self._validate_region_block_size_relation(
                    source_region, destination_region)
                head_mappings = pull_meta.fa_head_mappings_by_rank.get(
                    p_rank, ())
                if not head_mappings:
                    continue
                token_transfers = (
                    PcpReshardingPolicy.build_full_attention_token_transfers(
                        metadata=metadata,
                        destination=destination,
                        pull_meta=pull_meta,
                        p_rank=p_rank,
                        source_region=source_region,
                        destination_region=destination_region,
                    ))
                for transfer in token_transfers:
                    for head_mapping in head_mappings:
                        ops.append(
                            self._lower_segment(
                                layer_name=layer_name,
                                source_region=source_region,
                                destination_region=destination_region,
                                source_block=transfer.source_block,
                                source_token=transfer.source_token,
                                destination_block=transfer.destination_block,
                                destination_token=transfer.destination_token,
                                num_tokens=transfer.num_tokens,
                                head_mapping=head_mapping,
                            ))
                continue

            block_selection = self._select_blocks(
                metadata,
                destination,
                source_region,
                destination_region,
            )
            head_mappings = self._select_head_mappings(
                pull_meta,
                source_region.layer_type,
                p_rank,
            )
            if source_region.head_segments:
                head_mappings = self._head_mappings_from_segments(
                    source_region.head_segments, destination_region)
            if not head_mappings:
                continue

            for token_range in self._token_ranges(
                    source_blocks=block_selection.source_blocks,
                    source_block_size=source_region.lowering_units_per_block,
                    destination_blocks=block_selection.destination_blocks,
                    destination_block_size=destination_region.
                    lowering_units_per_block,
                    num_tokens=block_selection.num_tokens,
                    source_token_offset=block_selection.source_token_offset,
                    destination_token_offset=(
                        block_selection.destination_token_offset),
            ):
                source_block, source_token, dest_block, dest_token, count = (
                    token_range)
                for head_mapping in head_mappings:
                    ops.append(
                        self._lower_segment(
                            layer_name=layer_name,
                            source_region=source_region,
                            destination_region=destination_region,
                            source_block=source_block,
                            source_token=source_token,
                            destination_block=dest_block,
                            destination_token=dest_token,
                            num_tokens=count,
                            head_mapping=head_mapping,
                        ))

        return RankTransferPlan(p_rank=p_rank, ops=tuple(ops))

    def _select_blocks(
        self,
        metadata: ConnectorMetadataV2,
        destination: LocalDecodeAllocation,
        source_region: KVCacheRegion,
        destination_region: KVCacheRegion,
    ) -> BlockSelection:
        if source_region.layer_type != destination_region.layer_type:
            raise ValueError(
                "source and destination layer types must match, got "
                f"{source_region.layer_type} and {destination_region.layer_type}"
            )
        self._validate_full_attention_layout_pair(source_region,
                                                  destination_region)
        self._validate_region_block_size_relation(source_region,
                                                  destination_region)

        if is_linear_state_layer(source_region.layer_type):
            source_blocks = metadata.mamba_block_ids
            destination_blocks = destination.mamba_block_ids
            source_num_tokens = metadata.mamba_num_tokens
            destination_num_tokens = destination.mamba_num_tokens
            source_token_offset = 0
            destination_token_offset = 0

            if source_region.block_id_group_index is not None:
                if (destination_region.block_id_group_index
                        != source_region.block_id_group_index):
                    raise ValueError(
                        "source and destination block_id_group_index must "
                        f"match, got {source_region.block_id_group_index} "
                        f"and {destination_region.block_id_group_index}")
                block_id_index = source_region.block_id_group_index
                if block_id_index >= len(source_blocks):
                    raise ValueError(
                        f"source mamba_block_ids has no index {block_id_index}"
                    )
                if block_id_index >= len(destination_blocks):
                    raise ValueError(
                        "destination mamba_block_ids has no index "
                        f"{block_id_index}")
                source_blocks = (source_blocks[block_id_index], )
                destination_blocks = (destination_blocks[block_id_index], )
                source_num_tokens = source_region.lowering_units_per_block
                destination_num_tokens = (
                    destination_region.lowering_units_per_block)
        else:
            source_blocks = metadata.fa_block_ids
            destination_blocks = destination.fa_block_ids
            source_num_tokens = metadata.fa_num_tokens
            destination_num_tokens = destination.fa_num_tokens
            source_token_offset = metadata.fa_token_offset
            destination_token_offset = destination.fa_token_offset

        if source_num_tokens is None:
            source_num_tokens = (len(source_blocks) *
                                 source_region.lowering_units_per_block)
        if destination_num_tokens is None:
            destination_num_tokens = source_num_tokens
        if source_num_tokens != destination_num_tokens:
            raise ValueError(
                "source and destination token counts must match, got "
                f"{source_num_tokens} and {destination_num_tokens}")
        return BlockSelection(
            source_blocks=source_blocks,
            destination_blocks=destination_blocks,
            num_tokens=source_num_tokens,
            source_token_offset=source_token_offset,
            destination_token_offset=destination_token_offset,
        )

    @staticmethod
    def _select_head_mappings(
        pull_meta: PullMeta,
        layer_type: LayerType,
        p_rank: int,
    ) -> tuple[HeadMapping, ...]:
        if is_linear_state_layer(layer_type):
            return pull_meta.mamba_value_head_mappings_by_rank.get(p_rank, ())
        return pull_meta.fa_head_mappings_by_rank.get(p_rank, ())

    @staticmethod
    def _head_mappings_from_segments(
        segments: tuple[HeadSegment, ...],
        destination_region: KVCacheRegion,
    ) -> tuple[HeadMapping, ...]:
        mappings: list[HeadMapping] = []
        for segment in segments:
            for head in segment.global_heads:
                if (ContiguousHeadTPTransferPlanner._find_head_segment(
                        destination_region, head, segment.name) is None):
                    continue
                mappings.append(
                    HeadMapping(
                        global_head=head,
                        source_head=0,
                        destination_head=0,
                        segment_name=segment.name,
                    ))
        return tuple(mappings)

    @staticmethod
    def _token_ranges(
        source_blocks: tuple[int, ...],
        source_block_size: int,
        destination_blocks: tuple[int, ...],
        destination_block_size: int,
        num_tokens: int,
        source_token_offset: int = 0,
        destination_token_offset: int = 0,
    ) -> tuple[tuple[int, int, int, int, int], ...]:
        check_positive("source_block_size", source_block_size)
        check_positive("destination_block_size", destination_block_size)
        check_non_negative("num_tokens", num_tokens)
        check_non_negative("source_token_offset", source_token_offset)
        check_non_negative("destination_token_offset",
                           destination_token_offset)

        ranges: list[tuple[int, int, int, int, int]] = []
        cursor = 0
        while cursor < num_tokens:
            source_cursor = source_token_offset + cursor
            destination_cursor = destination_token_offset + cursor
            source_index = source_cursor // source_block_size
            destination_index = destination_cursor // destination_block_size
            if source_index >= len(source_blocks):
                raise ValueError("source block ids do not cover num_tokens")
            if destination_index >= len(destination_blocks):
                raise ValueError(
                    "destination block ids do not cover num_tokens")

            source_token = source_cursor % source_block_size
            destination_token = destination_cursor % destination_block_size
            count = min(
                source_block_size - source_token,
                destination_block_size - destination_token,
                num_tokens - cursor,
            )
            ranges.append((
                source_blocks[source_index],
                source_token,
                destination_blocks[destination_index],
                destination_token,
                count,
            ))
            cursor += count

        return tuple(ranges)

    def _lower_segment(
        self,
        layer_name: str,
        source_region: KVCacheRegion,
        destination_region: KVCacheRegion,
        source_block: int,
        source_token: int,
        destination_block: int,
        destination_token: int,
        num_tokens: int,
        head_mapping: HeadMapping,
    ) -> StridedSegmentOp:
        source_offset = self._layout_offset(
            region=source_region,
            block_id=source_block,
            token_offset=source_token,
            num_tokens=num_tokens,
            local_head=head_mapping.source_head,
            global_head=head_mapping.global_head,
            segment_name=head_mapping.segment_name,
        )
        destination_offset = self._layout_offset(
            region=destination_region,
            block_id=destination_block,
            token_offset=destination_token,
            num_tokens=num_tokens,
            local_head=head_mapping.destination_head,
            global_head=head_mapping.global_head,
            segment_name=head_mapping.segment_name,
        )
        if source_offset.segment_bytes != destination_offset.segment_bytes:
            raise ValueError(
                "source and destination segment bytes must match, got "
                f"{source_offset.segment_bytes} and "
                f"{destination_offset.segment_bytes}")

        return StridedSegmentOp(
            src_offset_bytes=source_offset.offset_bytes,
            dst_offset_bytes=destination_offset.offset_bytes,
            segment_bytes=source_offset.segment_bytes,
            src_stride_bytes=source_offset.stride_bytes,
            dst_stride_bytes=destination_offset.stride_bytes,
            num_segments=self._num_segments_for_region(
                source_region,
                head_mapping.global_head,
                head_mapping.segment_name,
                num_tokens,
            ),
            layer_name=layer_name,
            layer_type=source_region.layer_type,
            source_region_id=str(source_region.physical_region_id),
            destination_region_id=str(destination_region.physical_region_id),
            source_base_addr=source_region.base_addr,
            destination_base_addr=destination_region.base_addr,
            global_head=head_mapping.global_head,
            segment_name=head_mapping.segment_name,
        )

    @staticmethod
    def _layout_offset(
        region: KVCacheRegion,
        block_id: int,
        token_offset: int,
        num_tokens: int,
        local_head: int,
        global_head: int,
        segment_name: str | None = None,
    ) -> LayoutOffset:
        check_non_negative("block_id", block_id)
        check_non_negative("token_offset", token_offset)
        check_positive("num_tokens", num_tokens)
        segment = ContiguousHeadTPTransferPlanner._find_head_segment(
            region, global_head, segment_name)
        if segment is not None:
            local_segment_head = segment.global_heads.index(global_head)
            block_offset = block_id * region.physical_block_stride_bytes
            stride_bytes = segment.stride_bytes or segment.head_bytes
            return LayoutOffset(
                offset_bytes=(block_offset + segment.base_offset_bytes +
                              local_segment_head * segment.head_bytes),
                stride_bytes=stride_bytes,
                segment_bytes=segment.head_bytes,
            )
        if local_head >= region.num_heads:
            raise ValueError(f"local_head={local_head} is out of range for "
                             f"num_heads={region.num_heads}")

        block_offset = block_id * region.physical_block_stride_bytes
        if region.layout == TensorLayout.HEAD_FIRST:
            assert region.head_bytes is not None
            head_offset = local_head * region.head_bytes
            token_offset_bytes = token_offset * region.token_head_bytes
            segment_bytes = num_tokens * region.token_head_bytes
            return LayoutOffset(
                offset_bytes=block_offset + head_offset + token_offset_bytes,
                stride_bytes=segment_bytes,
                segment_bytes=segment_bytes,
            )

        if region.layout == TensorLayout.TOKEN_FIRST:
            token_offset_bytes = token_offset * region.token_bytes
            head_offset = local_head * region.token_head_stride_bytes
            return LayoutOffset(
                offset_bytes=block_offset + token_offset_bytes + head_offset,
                stride_bytes=region.token_bytes,
                segment_bytes=region.token_head_bytes,
            )

        if region.layout == TensorLayout.BLOCKS_FIRST:
            if region.block_size is None:
                if token_offset != 0 or num_tokens != 1:
                    raise ValueError(
                        "opaque BLOCKS_FIRST state expects one whole payload")
                assert region.head_bytes is not None
                head_offset = local_head * region.head_bytes
                return LayoutOffset(
                    offset_bytes=block_offset + head_offset,
                    stride_bytes=region.head_bytes,
                    segment_bytes=region.head_bytes,
                )
            if region.num_heads != 1:
                raise NotImplementedError(
                    "BLOCKS_FIRST head slicing needs an explicit inner layout")
            token_offset_bytes = token_offset * region.token_bytes
            segment_bytes = num_tokens * region.token_bytes
            return LayoutOffset(
                offset_bytes=block_offset + token_offset_bytes,
                stride_bytes=segment_bytes,
                segment_bytes=segment_bytes,
            )

        raise ValueError(f"Unsupported tensor layout: {region.layout}")

    @staticmethod
    def _num_segments_for_layout(layout: TensorLayout, num_tokens: int) -> int:
        if layout == TensorLayout.TOKEN_FIRST:
            return num_tokens
        return 1

    @staticmethod
    def _find_head_segment(
        region: KVCacheRegion,
        global_head: int,
        segment_name: str | None = None,
    ) -> HeadSegment | None:
        for segment in region.head_segments:
            if segment_name is not None and segment.name != segment_name:
                continue
            if global_head in segment.global_heads:
                return segment
        return None

    @staticmethod
    def _num_segments_for_region(
        region: KVCacheRegion,
        global_head: int,
        segment_name: str | None,
        num_tokens: int,
    ) -> int:
        segment = ContiguousHeadTPTransferPlanner._find_head_segment(
            region, global_head, segment_name)
        if segment is not None:
            return segment.num_segments
        return ContiguousHeadTPTransferPlanner._num_segments_for_layout(
            region.layout, num_tokens)
