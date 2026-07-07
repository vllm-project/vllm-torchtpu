from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .common import (LayerType, check_non_negative, check_positive,
                     check_power_of_two, linear_rank)
from .layout import KVCacheRegion


def _coerce_block_ids_by_group(
    block_ids_by_group: tuple[tuple[int, ...], ...]
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(block_id) for block_id in group)
        for group in block_ids_by_group)


@dataclass(frozen=True)
class KVParallelLayout:
    """KV parallel layout for full-attention and linear-attention cache."""

    full_attn_pcp_size: int
    full_attn_tp_size: int
    linear_attn_pcp_size: int
    linear_attn_tp_size: int

    def __post_init__(self) -> None:
        check_positive("full_attn_pcp_size", self.full_attn_pcp_size)
        check_positive("full_attn_tp_size", self.full_attn_tp_size)
        check_positive("linear_attn_pcp_size", self.linear_attn_pcp_size)
        check_positive("linear_attn_tp_size", self.linear_attn_tp_size)
        check_power_of_two("full_attn", self.full_attn_tp_size)
        check_power_of_two("linear_attn", self.linear_attn_tp_size)

    @property
    def full_attn_world_size(self) -> int:
        return self.full_attn_pcp_size * self.full_attn_tp_size

    @property
    def linear_attn_world_size(self) -> int:
        return self.linear_attn_pcp_size * self.linear_attn_tp_size

    def full_attn_ranks_for_tp_rank(self, tp_rank: int) -> tuple[int, ...]:
        return tuple(
            linear_rank(pcp_rank, tp_rank, self.full_attn_tp_size)
            for pcp_rank in range(self.full_attn_pcp_size))

    def linear_attn_ranks_for_tp_rank(self, tp_rank: int) -> tuple[int, ...]:
        return tuple(
            linear_rank(pcp_rank, tp_rank, self.linear_attn_tp_size)
            for pcp_rank in range(self.linear_attn_pcp_size))

    def to_dict(self) -> dict[str, int]:
        return {
            "full_attn_pcp_size": self.full_attn_pcp_size,
            "full_attn_tp_size": self.full_attn_tp_size,
            "linear_attn_pcp_size": self.linear_attn_pcp_size,
            "linear_attn_tp_size": self.linear_attn_tp_size,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "KVParallelLayout":
        return cls(
            full_attn_pcp_size=int(data["full_attn_pcp_size"]),
            full_attn_tp_size=int(data["full_attn_tp_size"]),
            linear_attn_pcp_size=int(data["linear_attn_pcp_size"]),
            linear_attn_tp_size=int(data["linear_attn_tp_size"]),
        )


@dataclass(frozen=True)
class ConnectorMetadataV2:
    """Scheduler-to-worker metadata for one completed prefill request."""

    req_id: int
    block_size: int
    kv_source_layout: KVParallelLayout
    kv_caches: dict[int, dict[str, KVCacheRegion]]
    fa_block_ids: tuple[int, ...]
    mamba_block_ids: tuple[int, ...]
    block_ids_by_group: tuple[tuple[int, ...], ...] = ()
    fa_num_tokens: int | None = None
    mamba_num_tokens: int | None = None
    fa_token_offset: int = 0

    def __post_init__(self) -> None:
        check_positive("block_size", self.block_size)
        check_non_negative("fa_token_offset", self.fa_token_offset)
        object.__setattr__(self, "fa_block_ids", tuple(self.fa_block_ids))
        object.__setattr__(self, "mamba_block_ids",
                           tuple(self.mamba_block_ids))
        object.__setattr__(
            self,
            "block_ids_by_group",
            _coerce_block_ids_by_group(self.block_ids_by_group),
        )
        if self.fa_num_tokens is not None:
            check_non_negative("fa_num_tokens", self.fa_num_tokens)
        if self.mamba_num_tokens is not None:
            check_non_negative("mamba_num_tokens", self.mamba_num_tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "req_id":
            self.req_id,
            "block_size":
            self.block_size,
            "kv_source_layout":
            self.kv_source_layout.to_dict(),
            "kv_caches": {
                int(rank): {
                    layer_name: region.to_dict()
                    for layer_name, region in regions.items()
                }
                for rank, regions in self.kv_caches.items()
            },
            "fa_block_ids":
            list(self.fa_block_ids),
            "mamba_block_ids":
            list(self.mamba_block_ids),
            "block_ids_by_group":
            [list(group) for group in self.block_ids_by_group],
            "fa_num_tokens":
            self.fa_num_tokens,
            "mamba_num_tokens":
            self.mamba_num_tokens,
            "fa_token_offset":
            self.fa_token_offset,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ConnectorMetadataV2":
        return cls(
            req_id=int(data["req_id"]),
            block_size=int(data["block_size"]),
            kv_source_layout=KVParallelLayout.from_mapping(
                data["kv_source_layout"]),
            kv_caches={
                int(rank): {
                    str(layer_name): KVCacheRegion.from_mapping(region)
                    for layer_name, region in regions.items()
                }
                for rank, regions in data["kv_caches"].items()
            },
            fa_block_ids=tuple(int(block) for block in data["fa_block_ids"]),
            mamba_block_ids=tuple(
                int(block) for block in data["mamba_block_ids"]),
            block_ids_by_group=tuple(
                tuple(int(block) for block in group)
                for group in data["block_ids_by_group"]),
            fa_num_tokens=(None if data.get("fa_num_tokens") is None else int(
                data["fa_num_tokens"])),
            mamba_num_tokens=(None if data.get("mamba_num_tokens") is None else
                              int(data["mamba_num_tokens"])),
            fa_token_offset=int(data.get("fa_token_offset", 0)),
        )


@dataclass(frozen=True)
class LocalDecodeAllocation:
    """Decode-side local destination selected during block allocation."""

    rank: int
    block_size: int
    kv_caches: dict[str, KVCacheRegion]
    fa_block_ids: tuple[int, ...]
    mamba_block_ids: tuple[int, ...]
    block_ids_by_group: tuple[tuple[int, ...], ...] = ()
    fa_num_tokens: int | None = None
    mamba_num_tokens: int | None = None
    fa_token_offset: int = 0

    def __post_init__(self) -> None:
        check_non_negative("rank", self.rank)
        check_positive("block_size", self.block_size)
        check_non_negative("fa_token_offset", self.fa_token_offset)
        object.__setattr__(self, "fa_block_ids", tuple(self.fa_block_ids))
        object.__setattr__(self, "mamba_block_ids",
                           tuple(self.mamba_block_ids))
        object.__setattr__(
            self,
            "block_ids_by_group",
            _coerce_block_ids_by_group(self.block_ids_by_group),
        )
        if self.fa_num_tokens is not None:
            check_non_negative("fa_num_tokens", self.fa_num_tokens)
        if self.mamba_num_tokens is not None:
            check_non_negative("mamba_num_tokens", self.mamba_num_tokens)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LocalDecodeAllocation":
        return cls(
            rank=int(data["rank"]),
            block_size=int(data["block_size"]),
            kv_caches={
                str(layer_name): KVCacheRegion.from_mapping(region)
                for layer_name, region in data["kv_caches"].items()
            },
            fa_block_ids=tuple(int(block) for block in data["fa_block_ids"]),
            mamba_block_ids=tuple(
                int(block) for block in data["mamba_block_ids"]),
            block_ids_by_group=tuple(
                tuple(int(block) for block in group)
                for group in data["block_ids_by_group"]),
            fa_num_tokens=(None if data.get("fa_num_tokens") is None else int(
                data["fa_num_tokens"])),
            mamba_num_tokens=(None if data.get("mamba_num_tokens") is None else
                              int(data["mamba_num_tokens"])),
            fa_token_offset=int(data.get("fa_token_offset", 0)),
        )


@dataclass(frozen=True)
class TpKVTopology:
    """Local TP view used by a decode worker to build a pull plan."""

    local_layout: KVParallelLayout
    block_size: int
    tp_rank: int
    total_num_kv_heads: int
    total_num_mamba_key_heads: int
    total_num_mamba_heads: int

    def __post_init__(self) -> None:
        check_positive("block_size", self.block_size)
        check_non_negative("tp_rank", self.tp_rank)
        check_non_negative("total_num_kv_heads", self.total_num_kv_heads)
        check_non_negative("total_num_mamba_key_heads",
                           self.total_num_mamba_key_heads)
        check_non_negative("total_num_mamba_heads", self.total_num_mamba_heads)

    @property
    def full_attn_tp_rank(self) -> int:
        return self.tp_rank % self.local_layout.full_attn_tp_size

    @property
    def linear_attn_tp_rank(self) -> int:
        return self.tp_rank % self.local_layout.linear_attn_tp_size

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TpKVTopology":
        return cls(
            local_layout=KVParallelLayout.from_mapping(data["local_layout"]),
            block_size=int(data["block_size"]),
            tp_rank=int(data["tp_rank"]),
            total_num_kv_heads=int(data["total_num_kv_heads"]),
            total_num_mamba_key_heads=int(data["total_num_mamba_key_heads"]),
            total_num_mamba_heads=int(data["total_num_mamba_heads"]),
        )


@dataclass(frozen=True)
class HeadMapping:
    """One global head mapped from a producer-local head to a D-local head."""

    global_head: int
    source_head: int
    destination_head: int
    segment_name: str | None = None

    def __post_init__(self) -> None:
        check_non_negative("global_head", self.global_head)
        check_non_negative("source_head", self.source_head)
        check_non_negative("destination_head", self.destination_head)


@dataclass(frozen=True)
class SourceBlockRef:
    """One producer block with its global logical block index."""

    logical_block_index: int
    block_id: int

    def __post_init__(self) -> None:
        check_non_negative("logical_block_index", self.logical_block_index)
        check_non_negative("block_id", self.block_id)


@dataclass(frozen=True)
class PullMeta:
    """Logical P-rank/head relationship for the current D rank."""

    req_id: int
    p_ranks: tuple[int, ...]
    fa_heads_by_rank: dict[int, tuple[int, ...]]
    mamba_key_heads_by_rank: dict[int, tuple[int, ...]]
    mamba_value_heads_by_rank: dict[int, tuple[int, ...]]
    fa_head_mappings_by_rank: dict[int, tuple[HeadMapping, ...]]
    mamba_key_head_mappings_by_rank: dict[int, tuple[HeadMapping, ...]]
    mamba_value_head_mappings_by_rank: dict[int, tuple[HeadMapping, ...]]
    fa_source_block_ids: tuple[int, ...]
    mamba_source_block_ids: tuple[int, ...]
    fa_block_refs_by_rank: dict[int, tuple[SourceBlockRef, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "p_ranks", tuple(self.p_ranks))
        object.__setattr__(self, "fa_source_block_ids",
                           tuple(self.fa_source_block_ids))
        object.__setattr__(self, "mamba_source_block_ids",
                           tuple(self.mamba_source_block_ids))
        object.__setattr__(
            self,
            "fa_block_refs_by_rank",
            {
                rank: tuple(block_refs)
                for rank, block_refs in self.fa_block_refs_by_rank.items()
            },
        )


@dataclass(frozen=True)
class StridedSegmentOp:
    """A future raiden-friendly copy descriptor.

    The H2H payload is still compact bytes. The strides only describe how
    D2H packs source HBM and how H2D scatters into destination HBM.
    """

    src_offset_bytes: int
    dst_offset_bytes: int
    segment_bytes: int
    src_stride_bytes: int
    dst_stride_bytes: int
    num_segments: int
    source_region_id: str
    destination_region_id: str
    layer_name: str
    layer_type: LayerType
    source_base_addr: int = 0
    destination_base_addr: int = 0
    global_head: int | None = None
    segment_name: str | None = None

    def __post_init__(self) -> None:
        if not self.source_region_id:
            raise ValueError("source_region_id must be non-empty")
        if not self.destination_region_id:
            raise ValueError("destination_region_id must be non-empty")
        if not self.layer_name:
            raise ValueError("layer_name must be non-empty")
        object.__setattr__(self, "source_region_id",
                           str(self.source_region_id))
        object.__setattr__(self, "destination_region_id",
                           str(self.destination_region_id))
        object.__setattr__(self, "layer_name", str(self.layer_name))
        check_non_negative("src_offset_bytes", self.src_offset_bytes)
        check_non_negative("dst_offset_bytes", self.dst_offset_bytes)
        check_positive("segment_bytes", self.segment_bytes)
        check_positive("src_stride_bytes", self.src_stride_bytes)
        check_positive("dst_stride_bytes", self.dst_stride_bytes)
        check_positive("num_segments", self.num_segments)
        check_non_negative("source_base_addr", self.source_base_addr)
        check_non_negative("destination_base_addr", self.destination_base_addr)
        if self.global_head is not None:
            check_non_negative("global_head", self.global_head)

    @property
    def total_bytes(self) -> int:
        return self.segment_bytes * self.num_segments

    @property
    def src_addr(self) -> int:
        return self.source_base_addr + self.src_offset_bytes

    @property
    def dst_addr(self) -> int:
        return self.destination_base_addr + self.dst_offset_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "src_offset_bytes": self.src_offset_bytes,
            "dst_offset_bytes": self.dst_offset_bytes,
            "segment_bytes": self.segment_bytes,
            "src_stride_bytes": self.src_stride_bytes,
            "dst_stride_bytes": self.dst_stride_bytes,
            "num_segments": self.num_segments,
            "layer_name": self.layer_name,
            "layer_type": self.layer_type.value,
            "source_region_id": self.source_region_id,
            "destination_region_id": self.destination_region_id,
            "global_head": self.global_head,
            "segment_name": self.segment_name,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "StridedSegmentOp":
        return cls(
            src_offset_bytes=int(data["src_offset_bytes"]),
            dst_offset_bytes=int(data["dst_offset_bytes"]),
            segment_bytes=int(data["segment_bytes"]),
            src_stride_bytes=int(data["src_stride_bytes"]),
            dst_stride_bytes=int(data["dst_stride_bytes"]),
            num_segments=int(data["num_segments"]),
            layer_name=str(data["layer_name"]),
            layer_type=LayerType(data["layer_type"]),
            source_region_id=str(data["source_region_id"]),
            destination_region_id=str(data["destination_region_id"]),
            global_head=(None if data.get("global_head") is None else int(
                data["global_head"])),
            segment_name=data.get("segment_name"),
        )


@dataclass(frozen=True)
class RankTransferPlan:
    """Lowered transfer ops from one producer rank."""

    p_rank: int
    ops: tuple[StridedSegmentOp, ...]

    @property
    def total_bytes(self) -> int:
        return sum(op.total_bytes for op in self.ops)
