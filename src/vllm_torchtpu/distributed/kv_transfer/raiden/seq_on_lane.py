# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Seq-along-lane FA pool byte spans (RESHARD_DP4TP2_SEQ_ON_LANE_PLAN.md, D1).

The batched-RPA ``SEQ_ALONG_LANE`` cache is
``[pages, kv_rows, packed_rows, packing, page_tokens]`` with
``kv_rows = num_kv_heads * 2`` (K and V of every local KV head, K before V,
head-major), ``packed_rows = head_dim // packing`` and the tokens of one
kernel page on the lane axis. Measured on v7x: minor-to-major
``(4, 3, 2, 1, 0)`` with tiles ``((4, 128), (4, 1))``, so one 32-bit word
packs ``packing`` consecutive head-dim elements of one token, one
``(row, word-row)`` of a page is ``packing * page_tokens`` contiguous bytes,
and one K/V row of one page is a contiguous ``slab``. The KV rows of one
destination head shard are therefore one contiguous run per kernel page --
which is what lets the raw byte-span transport route a producer's full-head
pages to head-sharded (TP > 1) consumers without any data transform.

This module is the stand-alone declaration of that layout: geometry from
the live tensor shape (fail closed on anything else), the manifest's FA
regions, the producer's FA lowering (one span per owned kernel page per
destination shard, ``dst_unit_ordinal`` = shard), the GDN fan-in routing per
destination shard (layout-independent), and the E0' fingerprint that refuses
a token-major peer. The token-major lowering is untouched.

Geometry contract (enforced): the kernel page is a whole number of 128-lane
tiles, ``cp_kv_cache_interleave_size == page_tokens`` for PCP producers (one
owned chunk == one kernel page), and both peers run the same
``page_tokens`` (it is part of the fingerprint).
"""

import dataclasses
import hashlib
import importlib.metadata
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.distributed.kv_transfer.raiden.byte_spans import (
    PoolByteSpan,
    PoolSpanRegistration,
    _region_value,
    lower_gdn_state_shard_spans,
    owned_token_ranges,
)
from vllm_torchtpu.distributed.kv_transfer.raiden.pool_manifest import (
    FaLayoutHook,
    GdnHeadGeometry,
    PoolManifest,
    RegionSpec,
    build_qwen35_pool_manifest,
)
from vllm_torchtpu.distributed.kv_transfer.raiden.tags import (
    TAG_FA,
    TAG_GDN_CONV,
    TAG_GDN_SSM,
    class_tag,
)
from vllm_torchtpu.kernels.gdn.head_geometry import derive_gdn_head_geometry

# One vreg lane row: the DMA / tile granule of the layout.
LANE_WIDTH = 128
# fp8 packs four head-dim elements per 32-bit word.
PACKING = 4
# One (row, word-row) of one lane tile: the physical word-row.
WORD_ROW_BYTES = PACKING * LANE_WIDTH
SOL_LAYOUT_TOKEN = "seq-along-lane"
SOL_FINGERPRINT_SCHEMA = "qwen35-fa-raw-layout-fingerprint-v2"
SOL_REGION_NAME = "fa_sol"
# Smallest head_dim the batched-RPA kernel materializes (8 sublanes x packing).
MIN_HEAD_DIM = 32
# The pooled GDN kernel stores conv state as BF16 rows of carrier lanes.
GDN_CONV_ITEMSIZE = 2
EXPECTED_SOL_MINOR_TO_MAJOR = (4, 3, 2, 1, 0)
EXPECTED_SOL_TILES = ((4, 128), (4, 1))


class SeqOnLaneError(ValueError):
    """The admitted geometry is not the seq-on-lane layout this module
    lowers (fail closed)."""


@dataclasses.dataclass(frozen=True)
class SolFaGeometry:
    """Byte geometry of one seq-on-lane FA kernel page, derived from the live
    tensor shape ``[pages, kv_rows, packed_rows, packing, page_tokens]``."""

    kv_rows: int  # num_kv_heads * 2 (K and V rows of every local KV head)
    packed_rows: int  # head_dim // packing
    page_tokens: int  # tokens on the lane axis of one kernel page
    packing: int = PACKING

    @property
    def num_kv_heads(self) -> int:
        return self.kv_rows // 2

    @property
    def head_dim(self) -> int:
        return self.packed_rows * self.packing

    @property
    def slab_bytes(self) -> int:
        """One K or V row of one kernel page: contiguous (measured)."""
        return self.packed_rows * self.packing * self.page_tokens

    @property
    def pair_bytes(self) -> int:
        """K and V of one KV head in one kernel page (adjacent rows)."""
        return 2 * self.slab_bytes

    @property
    def kernel_page_bytes(self) -> int:
        return self.kv_rows * self.slab_bytes

    @property
    def bytes_per_token(self) -> int:
        return self.kv_rows * self.head_dim

    def shard_bytes(self, dst_shards: int) -> int:
        """Bytes of one destination head shard inside one kernel page."""
        if dst_shards <= 0 or self.num_kv_heads % dst_shards:
            raise SeqOnLaneError(
                "destination shards must divide the local KV head count: "
                f"num_kv_heads={self.num_kv_heads}, dst_shards={dst_shards}"
            )
        return self.kernel_page_bytes // dst_shards


def sol_fa_geometry_from_shape(
    shape: Sequence[int], *, itemsize: int = 1
) -> SolFaGeometry:
    """Validates ``[pages, kv_rows, packed_rows, 4, page_tokens]`` fp8 and
    returns its geometry; any other shape (notably the token-major
    ``[pages, page_tokens, groups, 4, head_dim]``) is refused."""
    dims = tuple(int(d) for d in shape)
    if len(dims) != 5:
        raise SeqOnLaneError(f"seq-on-lane FA cache must be rank-5: got shape {dims}")
    if itemsize != 1:
        raise SeqOnLaneError(
            f"seq-on-lane FA cache must be fp8 (itemsize 1): got {itemsize}"
        )
    pages, kv_rows, packed_rows, packing, page_tokens = dims
    if packing != PACKING:
        raise SeqOnLaneError(
            f"seq-on-lane FA cache packing must be {PACKING}: got {packing} "
            f"(shape {dims})"
        )
    if page_tokens <= 0 or page_tokens % LANE_WIDTH:
        raise SeqOnLaneError(
            "seq-on-lane FA cache kernel page must be a whole number of "
            f"{LANE_WIDTH}-lane tiles: got {page_tokens} (shape {dims})"
        )
    if pages <= 0 or kv_rows <= 0 or packed_rows <= 0:
        raise SeqOnLaneError(f"seq-on-lane FA cache has an empty dim: {dims}")
    # The kernel pads head_dim to a multiple of sublanes*packing (32); a
    # token-major fp8 cache [pages, page_tokens, groups, 4, head_dim] read as
    # this layout would put its 1-2 packed groups here.
    if packed_rows * packing < MIN_HEAD_DIM:
        raise SeqOnLaneError(
            "seq-on-lane FA cache head_dim is implausibly small (token-major "
            f"shape?): {packed_rows * packing} < {MIN_HEAD_DIM} (shape {dims})"
        )
    if kv_rows % 2:
        raise SeqOnLaneError(
            "seq-on-lane FA cache kv_rows must be even (K and V per KV "
            f"head): got {kv_rows}"
        )
    return SolFaGeometry(
        kv_rows=kv_rows, packed_rows=packed_rows, page_tokens=page_tokens
    )


def sol_fa_regions(
    *, block_size_tokens: int, geometry: SolFaGeometry
) -> tuple[RegionSpec, ...]:
    """One dense region per manager block: ``num_units`` kernel pages of
    ``num_kv_heads`` K+V head pairs each (compact-live == physical)."""
    if block_size_tokens <= 0 or block_size_tokens % geometry.page_tokens:
        raise SeqOnLaneError(
            "seq-on-lane manager block must be a whole number of kernel "
            f"pages: block_size={block_size_tokens}, "
            f"kernel_page={geometry.page_tokens}"
        )
    kernel_pages = block_size_tokens // geometry.page_tokens
    return (
        RegionSpec(
            name=SOL_REGION_NAME,
            offset_bytes=0,
            stride_bytes=geometry.kernel_page_bytes,
            unit_bytes=geometry.pair_bytes,
            num_units=kernel_pages,
            units_per_stride=geometry.num_kv_heads,
        ),
    )


class SolFaLayout(FaLayoutHook):
    """``build_qwen35_pool_manifest`` hook for the seq-on-lane FA shape."""

    name = SOL_LAYOUT_TOKEN

    def total_tokens(self, shape: Sequence[int]) -> int:
        geometry = sol_fa_geometry_from_shape(shape)
        return int(shape[0]) * geometry.page_tokens

    def regions(
        self,
        *,
        block_size_tokens: int,
        shape: Sequence[int],
        nbytes: int,
        num_kv_heads: int,
        head_size: int,
        itemsize: int,
    ) -> tuple[RegionSpec, ...]:
        geometry = sol_fa_geometry_from_shape(shape, itemsize=itemsize)
        if geometry.kv_rows != 2 * int(num_kv_heads):
            raise SeqOnLaneError(
                "seq-on-lane FA cache kv_rows disagree with the cache spec: "
                f"shape has {geometry.kv_rows}, spec num_kv_heads="
                f"{num_kv_heads}"
            )
        # head_size may be padded up to a multiple of 8*packing (=32) by the
        # kernel; the live tensor is authoritative, the spec only bounds it.
        if geometry.head_dim < int(head_size):
            raise SeqOnLaneError(
                "seq-on-lane FA cache head_dim is smaller than the spec's: "
                f"shape {geometry.head_dim} < spec {head_size}"
            )
        total = int(shape[0]) * geometry.page_tokens
        if nbytes != total * geometry.bytes_per_token:
            raise SeqOnLaneError(
                "seq-on-lane FA cache byte size disagrees with its shape: "
                f"nbytes={nbytes}, tokens={total}, "
                f"bytes_per_token={geometry.bytes_per_token}"
            )
        return sol_fa_regions(block_size_tokens=block_size_tokens, geometry=geometry)


def build_qwen35_pool_manifest_sol(
    *,
    named_kv_caches: Mapping[str, Any],
    kv_cache_groups: Sequence[Any],
    raw_tensors: Sequence[Any],
    gdn_geometry: GdnHeadGeometry,
    mamba_group_ordinal_by_layer: Mapping[str, int] | None = None,
    per_layer_tags: bool = False,
) -> PoolManifest:
    """The canonical Qwen3.5 manifest with seq-on-lane FA regions; GDN state
    pools derive exactly as in the token-major manifest (byte-level: SSM at
    offset 0 of the manager page, conv right after it)."""
    return build_qwen35_pool_manifest(
        named_kv_caches=named_kv_caches,
        kv_cache_groups=kv_cache_groups,
        raw_tensors=raw_tensors,
        gdn_geometry=gdn_geometry,
        mamba_group_ordinal_by_layer=mamba_group_ordinal_by_layer,
        fa_layout=SolFaLayout(),
        per_layer_tags=per_layer_tags,
    )


def sol_fa_geometry_from_manifest(manifest: PoolManifest) -> SolFaGeometry:
    """The admitted FA geometry (validated against the FA storage shape)."""
    fa_pools = [pool for pool in manifest.pools if class_tag(pool.tag) == TAG_FA]
    if not fa_pools:
        raise SeqOnLaneError("manifest has no FA pool")
    storage = manifest.storages[fa_pools[0].storage_index]
    shape = tuple(int(d) for d in getattr(storage, "shape", ()))
    itemsize = int(storage.element_size()) if hasattr(storage, "element_size") else 1
    geometry = sol_fa_geometry_from_shape(shape, itemsize=itemsize)
    for pool in fa_pools:
        if len(pool.regions) != 1 or pool.regions[0].name != SOL_REGION_NAME:
            raise SeqOnLaneError(
                "FA pools were not admitted with seq-on-lane regions "
                f"({pool.layer_name}: {[r.name for r in pool.regions]})"
            )
        region = pool.regions[0]
        if (
            region.stride_bytes != geometry.kernel_page_bytes
            or region.unit_bytes != geometry.pair_bytes
            or region.units_per_stride != geometry.num_kv_heads
        ):
            raise SeqOnLaneError(
                f"FA pool {pool.layer_name} regions disagree with the "
                "seq-on-lane geometry"
            )
    return geometry


def sol_fa_page_tokens(manifest: PoolManifest) -> int:
    """Manager page tokens of every FA pool (kernel pages x page tokens); the
    seq-on-lane counterpart of ``layout_fingerprint.fa_page_tokens``."""
    geometry = sol_fa_geometry_from_manifest(manifest)
    page_tokens: set[int] = set()
    for pool in manifest.pools:
        if class_tag(pool.tag) != TAG_FA:
            continue
        page_tokens.add(int(pool.regions[0].num_units) * geometry.page_tokens)
    if len(page_tokens) != 1:
        raise SeqOnLaneError(
            f"FA page geometry differs across pools: {sorted(page_tokens)}"
        )
    return next(iter(page_tokens))


def sol_fa_skip_bytes(
    skip_tokens: int, *, geometry: SolFaGeometry, dst_shards: int
) -> int:
    """Prefix-aware clip in one destination shard's compact byte space: a
    locally cached prefix can only be skipped in whole kernel pages."""
    if skip_tokens <= 0:
        return 0
    if skip_tokens % geometry.page_tokens:
        raise SeqOnLaneError(
            "seq-on-lane prefix clip must be a whole number of kernel pages: "
            f"skip_tokens={skip_tokens}, page_tokens={geometry.page_tokens}"
        )
    return (skip_tokens // geometry.page_tokens) * geometry.shard_bytes(dst_shards)


def _tp_head_range(heads: int, degree: int, rank: int) -> tuple[int, int]:
    """Global KV-head interval, including TP replication when heads < TP."""
    if heads <= 0 or degree <= 0:
        raise SeqOnLaneError("head count and TP degree must be positive")
    if not 0 <= rank < degree:
        raise SeqOnLaneError(f"TP rank {rank} is outside degree {degree}")
    unique = min(heads, degree)
    if heads % unique or degree % unique:
        raise SeqOnLaneError(
            f"KV heads and TP degree must divide evenly: {heads=}, {degree=}"
        )
    width = heads // unique
    start = (rank // (degree // unique)) * width
    return start, start + width


def lower_fa_spans_sol(
    *,
    num_tokens: int,
    transfer_rank: int,
    parallelism: int,
    interleave_tokens: int,
    page_tokens: int,
    geometry: SolFaGeometry,
    dst_shards: int,
    block_ids: Sequence[int],
    producer_tp_size: int = 1,
    producer_tp_rank: int = 0,
    total_kv_heads: int | None = None,
    dst_dcp_size: int = 1,
) -> PoolSpanRegistration:
    """Intersect source PCP/TP and destination TP/DCP ownership in HND.

    Source: the rank's owned chunks (``pcp_layout``) packed dense rank-major
    at kernel-page granularity over a prefix of its manager blocks. The
    intersection of source and destination heads is contiguous inside each
    kernel page. Destination offsets address rank-local compact byte space
    (``dst_space_version=1``), which the planner splits at the destination's
    manager page boundaries. The last owned kernel page ships whole
    even when the request ends inside it (lanes past the request are never
    read before later tokens rewrite them).

    ``page_tokens`` is the manager block (a whole number of kernel pages).
    ``dst_shards`` counts destination TP workers, including DCP lanes. DCP
    groups are contiguous TP ranks holding identical KV heads. Global page
    j goes to lane j % DCP at compact local page j // DCP. Each DCP group
    of ranks holding the same KV heads receives one full copy. Copies are
    dealt round-robin across the replicas of a source head, so each replica
    ships its local bytes at most once when the destination needs no more
    copies than the source has replicas (the controller rejects a
    registration whose span sum exceeds ``declared_bytes``). Replicas that
    are dealt no copy still register their local bytes and participate in
    GDN state transfer.

    The controller requires equal coverage on all destination lanes. In
    the final DCP stripe, copy the last source page into missing lanes as
    padding. Those bytes are outside the real sequence length and are
    overwritten before decode can attend to them. Source block ownership
    and the request's token count remain unchanged.

    TP1/DCP1 use the same ownership intersection. total_kv_heads is required
    for a TP producer because its local shape cannot distinguish sharding
    from replication.
    """
    if num_tokens <= 0:
        raise SeqOnLaneError("num_tokens must be positive")
    if parallelism <= 0 or not 0 <= transfer_rank < parallelism:
        raise SeqOnLaneError(
            f"transfer_rank {transfer_rank} is outside parallelism {parallelism}"
        )
    kp_tokens = geometry.page_tokens
    if parallelism > 1 and interleave_tokens != kp_tokens:
        raise SeqOnLaneError(
            "seq-on-lane PCP lowering requires cp_kv_cache_interleave_size "
            f"== kernel page ({kp_tokens}) (decision D1): got "
            f"{interleave_tokens}"
        )
    if page_tokens <= 0 or page_tokens % kp_tokens:
        raise SeqOnLaneError(
            "seq-on-lane manager page must be a whole number of kernel "
            f"pages: page_tokens={page_tokens}, kernel_page={kp_tokens}"
        )
    heads = (
        geometry.num_kv_heads * producer_tp_size
        if total_kv_heads is None
        else total_kv_heads
    )
    if producer_tp_size > 1 and total_kv_heads is None:
        raise SeqOnLaneError("total_kv_heads is required for a TP producer")
    src_start, src_end = _tp_head_range(heads, producer_tp_size, producer_tp_rank)
    if src_end - src_start != geometry.num_kv_heads:
        raise SeqOnLaneError("source head geometry disagrees with TP ownership")
    if dst_dcp_size <= 0 or dst_shards <= 0 or dst_shards % dst_dcp_size:
        raise SeqOnLaneError("DCP size must divide destination TP degree")
    destinations = [
        _tp_head_range(heads, dst_shards, rank) for rank in range(dst_shards)
    ]
    for base in range(0, dst_shards, dst_dcp_size):
        if any(
            head_range != destinations[base]
            for head_range in destinations[base : base + dst_dcp_size]
        ):
            raise SeqOnLaneError("DCP group ranks must own identical KV heads")
    source_replicas = producer_tp_size // min(heads, producer_tp_size)
    source_replica = producer_tp_rank % source_replicas
    dst_replicas = dst_shards // min(heads, dst_shards)
    # Destination shards whose copy of its heads this replica ships.
    is_copy_owner = [
        (shard % dst_replicas) // dst_dcp_size % source_replicas == source_replica
        for shard in range(dst_shards)
    ]
    kps_per_block = page_tokens // kp_tokens
    ranges = owned_token_ranges(
        num_tokens=num_tokens,
        transfer_rank=transfer_rank,
        parallelism=parallelism,
        interleave_tokens=interleave_tokens,
    )
    spans: list[PoolByteSpan] = []
    final_kp = (num_tokens - 1) // kp_tokens
    local_kp = 0
    for start, end in ranges:
        if start % kp_tokens:
            raise SeqOnLaneError(
                "PCP chunk does not start on a kernel page boundary: "
                f"start={start}, kernel_page={kp_tokens}"
            )
        first_kp = start // kp_tokens
        last_kp = (end + kp_tokens - 1) // kp_tokens  # exclusive; tail whole
        for global_kp in range(first_kp, last_kp):
            block_ordinal, in_block_kp = divmod(local_kp, kps_per_block)
            for shard, (dst_start, dst_end) in enumerate(destinations):
                head_start, head_end = max(src_start, dst_start), min(src_end, dst_end)
                lane = shard % dst_dcp_size
                owns_page = lane == global_kp % dst_dcp_size
                pads_tail = global_kp == final_kp and lane > final_kp % dst_dcp_size
                if (
                    not is_copy_owner[shard]
                    or head_start >= head_end
                    or not (owns_page or pads_tail)
                ):
                    continue
                dst_page_bytes = (dst_end - dst_start) * geometry.pair_bytes
                spans.append(
                    PoolByteSpan(
                        src_block_ordinal=block_ordinal,
                        src_offset_bytes=(
                            in_block_kp * geometry.kernel_page_bytes
                            + (head_start - src_start) * geometry.pair_bytes
                        ),
                        dst_block_index=0,
                        dst_offset_bytes=(
                            (global_kp // dst_dcp_size) * dst_page_bytes
                            + (head_start - dst_start) * geometry.pair_bytes
                        ),
                        size_bytes=(head_end - head_start) * geometry.pair_bytes,
                        dst_unit_ordinal=(shard if dst_shards > 1 else None),
                    )
                )
            local_kp += 1
    expected_blocks = (local_kp + kps_per_block - 1) // kps_per_block
    if len(block_ids) != expected_blocks:
        raise SeqOnLaneError(
            "block_ids do not match dense kernel-page packing: "
            f"got {len(block_ids)}, expected {expected_blocks} for "
            f"{local_kp} owned kernel pages"
        )
    return PoolSpanRegistration(
        tag=TAG_FA,
        block_ids=tuple(int(block_id) for block_id in block_ids),
        spans=tuple(spans),
        declared_bytes=sum(span.size_bytes * span.count for span in spans),
        dst_space_version=1,
    )


def lower_gdn_state_shard_spans_sol(
    *,
    tag: str,
    block_id: int,
    transfer_rank: int,
    parallelism: int,
    regions: Sequence[object],
    dst_shards: int,
    producer_tp_size: int = 1,
    producer_tp_rank: int = 0,
    physical_granule_bytes: int = 1024,
    num_key_heads: int | None = None,
    num_value_heads: int | None = None,
    conv_lane_bytes: int | None = None,
) -> PoolSpanRegistration:
    """One rank's GDN head shard into its destination shard's full state.

    Destination shard ``d`` holds the heads of producer ranks
    ``[d*F, (d+1)*F)`` with ``F = parallelism // dst_shards`` (contiguous
    head blocks on both sides), so the existing fan-in lowering applies with
    ``(transfer_rank mod F, F)`` and every span routed to destination ``d``.
    ``dst_shards == 1`` is exactly the existing lowering. A full-width
    producer (``parallelism == 1``, e.g. a DP prefill engine) splits its
    state instead: shard ``d`` takes head block ``d`` of every region
    (``_lower_gdn_state_split_spans``). A sharded producer with more
    destination shards than ranks (``dst_shards % parallelism == 0``) feeds
    several destinations from each rank
    (``lower_gdn_state_fanout_spans_sol``, which needs the model's head
    counts and ``conv_lane_bytes``). Layout-independent: GDN state bytes
    are word-row separable under both FA layouts.

    ``physical_granule_bytes`` is the measured state carrier's word-row
    width, not the logical state's element size. The existing 256-lane
    carrier uses 1024 bytes; a 128-lane HND carrier uses 512. Both source
    and destination must have the same admitted carrier layout.
    """
    # Worker enumeration is PCP-major, but GDN head ownership follows the
    # (TP, PCP) state axis in gdn_attention._gdn_pcp_tp_state_axis. DCP only
    # partitions FA history; it does not partition GDN recurrent states.
    if (
        parallelism <= 0
        or not 0 <= transfer_rank < parallelism
        or producer_tp_size <= 0
        or not 0 <= producer_tp_rank < producer_tp_size
    ):
        raise SeqOnLaneError("invalid producer PCP/TP rank or degree")
    transfer_rank += producer_tp_rank * parallelism
    parallelism *= producer_tp_size
    if dst_shards <= 0:
        raise SeqOnLaneError("destination shards must be positive")
    if parallelism == 1 and dst_shards > 1:
        return _lower_gdn_state_split_spans(
            tag=tag,
            block_id=block_id,
            regions=regions,
            dst_shards=dst_shards,
            physical_granule_bytes=physical_granule_bytes,
        )
    if dst_shards > parallelism and dst_shards % parallelism == 0:
        if num_key_heads is None or num_value_heads is None:
            raise SeqOnLaneError(
                "GDN fan-out to more destination shards than producer ranks "
                "requires the model's key and value head counts"
            )
        return lower_gdn_state_fanout_spans_sol(
            tag=tag,
            block_id=block_id,
            transfer_rank=transfer_rank,
            parallelism=parallelism,
            regions=regions,
            dst_shards=dst_shards,
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            conv_lane_bytes=conv_lane_bytes,
            physical_granule_bytes=physical_granule_bytes,
        )
    if parallelism % dst_shards:
        raise SeqOnLaneError(
            "destination shards must divide the producer parallelism: "
            f"parallelism={parallelism}, dst_shards={dst_shards}"
        )
    if dst_shards == 1:
        return lower_gdn_state_shard_spans(
            tag=tag,
            block_id=block_id,
            transfer_rank=transfer_rank,
            parallelism=parallelism,
            regions=regions,
            physical_granule_bytes=physical_granule_bytes,
        )
    fan_in = parallelism // dst_shards
    shard, rank_in_shard = divmod(int(transfer_rank), fan_in)
    registration = lower_gdn_state_shard_spans(
        tag=tag,
        block_id=block_id,
        transfer_rank=rank_in_shard,
        parallelism=fan_in,
        regions=regions,
        physical_granule_bytes=physical_granule_bytes,
    )
    return dataclasses.replace(
        registration,
        spans=tuple(
            dataclasses.replace(span, dst_unit_ordinal=shard)
            for span in registration.spans
        ),
    )


def _lower_gdn_state_split_spans(
    *,
    tag: str,
    block_id: int,
    regions: Sequence[object],
    dst_shards: int,
    physical_granule_bytes: int = 1024,
) -> PoolSpanRegistration:
    """Full-width producer state -> ``dst_shards`` head-sharded destination
    states (the converse of the fan-in lowering): every region (QK block, V
    block, SSM) is split into ``dst_shards`` equal contiguous head blocks;
    shard ``d`` receives block ``d`` at offset 0 of its own (narrower) region.
    """
    full = lower_gdn_state_shard_spans(
        tag=tag,
        block_id=block_id,
        transfer_rank=0,
        parallelism=1,
        regions=regions,
        physical_granule_bytes=physical_granule_bytes,
    )
    spans: list[PoolByteSpan] = []
    segments = list(full.spans)  # conv: [QK, V] strided x taps; ssm: [one]
    if not segments:
        raise SeqOnLaneError(f"no state spans to split for {tag}")
    local_row = segments[0].src_stride_bytes if len(segments) > 1 else 0
    shard_row = local_row // dst_shards if local_row else 0
    dst_segment_base = 0
    for span in segments:
        if span.size_bytes % dst_shards:
            raise SeqOnLaneError(
                f"{tag} segment of {span.size_bytes} B does not split into "
                f"{dst_shards} shards"
            )
        part = span.size_bytes // dst_shards
        if part % physical_granule_bytes:
            raise SeqOnLaneError(
                f"{tag} shard segment {part} B is not a whole physical "
                f"granule ({physical_granule_bytes} B)"
            )
        for shard in range(dst_shards):
            spans.append(
                PoolByteSpan(
                    src_block_ordinal=0,
                    src_offset_bytes=span.src_offset_bytes + shard * part,
                    dst_block_index=0,
                    dst_offset_bytes=dst_segment_base,
                    size_bytes=part,
                    src_stride_bytes=span.src_stride_bytes,
                    dst_stride_bytes=shard_row if span.count > 1 else 0,
                    count=span.count,
                    dst_unit_ordinal=shard,
                )
            )
        dst_segment_base += part
    return dataclasses.replace(full, spans=tuple(spans))


def sol_gdn_conv_lane_bytes(geometry: SolFaGeometry) -> int:
    """Lane width in bytes of the pooled GDN kernel's conv rows on the HND
    carrier: ``page_tokens`` lanes of BF16 conv state."""
    return geometry.page_tokens * GDN_CONV_ITEMSIZE


_STATE_REGION_FIELDS = (
    "offset_bytes",
    "stride_bytes",
    "unit_bytes",
    "num_units",
    "units_per_stride",
)


def _state_region_map(
    tag: str, regions: Sequence[object]
) -> dict[str, tuple[int, ...]]:
    """``name -> (offset, stride, unit, num_units, units_per_stride)``."""
    result: dict[str, tuple[int, ...]] = {}
    for region in regions:
        name = str(_region_value(region, "name"))
        if name in result:
            raise SeqOnLaneError(f"duplicate GDN state region {name!r} ({tag})")
        values = tuple(
            int(_region_value(region, field)) for field in _STATE_REGION_FIELDS
        )
        if min(values[:2]) < 0 or min(values[2:]) <= 0:
            raise SeqOnLaneError(f"invalid GDN state region {name!r} ({tag})")
        result[name] = values
    return result


def lower_gdn_state_fanout_spans_sol(
    *,
    tag: str,
    block_id: int,
    transfer_rank: int,
    parallelism: int,
    regions: Sequence[object],
    dst_shards: int,
    num_key_heads: int,
    num_value_heads: int,
    conv_lane_bytes: int | None = None,
    physical_granule_bytes: int = 1024,
) -> PoolSpanRegistration:
    """One rank's GDN head shard into the narrower states of the
    ``F = dst_shards // parallelism`` destination shards holding its heads.

    Both sides own heads as ``derive_gdn_head_geometry`` describes: rank
    ``r`` of degree ``N`` holds value-head block ``r`` and key-head block
    ``r // R`` (``R`` = Q/K replication at degree ``N``). Destination ``d``
    holds a subset of producer rank ``d // F``'s heads and receives its
    whole state from that rank, routed by ``dst_unit_ordinal``.
    Destinations sharing a key head each receive a copy of its Q/K rows.

    Conv, QK pair-blocked layout: per tap the QK block stores ``[Q, K]``
    row pairs in row order, followed by the V heads. A row is
    ``gcd(conv row, conv_lane_bytes, Q segment)`` bytes, the pooled
    kernel's logical row width. With equal row widths on both sides, each
    key head is a contiguous run of row pairs, so a destination takes one
    QK run and one V run per tap. The split q/k/v layout takes one run
    per segment and tap. SSM: one run of the destination's value heads.

    ``declared_bytes`` counts every routed byte, so replicated Q/K rows
    count once per destination.
    """
    if parallelism <= 1 or dst_shards <= parallelism or dst_shards % parallelism:
        raise SeqOnLaneError(
            "GDN fan-out requires a sharded producer and a whole number of "
            f"destination shards per rank: parallelism={parallelism}, "
            f"dst_shards={dst_shards}"
        )
    if not 0 <= transfer_rank < parallelism:
        raise SeqOnLaneError(
            f"transfer_rank {transfer_rank} is outside parallelism {parallelism}"
        )
    if block_id < 0 or physical_granule_bytes <= 0:
        raise SeqOnLaneError("GDN state block id and physical granule must be valid")
    try:
        source = derive_gdn_head_geometry(num_key_heads, num_value_heads, parallelism)
        destination = derive_gdn_head_geometry(
            num_key_heads, num_value_heads, dst_shards
        )
    except ValueError as exc:
        raise SeqOnLaneError(f"GDN heads do not shard: {exc}") from exc
    fan_out = dst_shards // parallelism
    src_keys, src_values = source.local_num_kq_heads, source.local_num_v_heads
    dst_keys, dst_values = (
        destination.local_num_kq_heads,
        destination.local_num_v_heads,
    )
    src_key0 = source.kq_shard_index(transfer_rank) * src_keys
    src_value0 = transfer_rank * src_values
    # (destination shard, first key head, first value head), local to this
    # rank's head blocks.
    targets = []
    for shard in range(transfer_rank * fan_out, (transfer_rank + 1) * fan_out):
        key = destination.kq_shard_index(shard) * dst_keys - src_key0
        value = shard * dst_values - src_value0
        if not (
            0 <= key <= src_keys - dst_keys and 0 <= value <= src_values - dst_values
        ):
            raise SeqOnLaneError(
                f"destination shard {shard} heads lie outside producer rank "
                f"{transfer_rank}'s heads"
            )
        targets.append((shard, key, value))

    layout = _state_region_map(tag, regions)
    is_conv = tag == TAG_GDN_CONV or tag.startswith(f"{TAG_GDN_CONV}.")
    is_ssm = tag == TAG_GDN_SSM or tag.startswith(f"{TAG_GDN_SSM}.")
    names = set(layout)
    if is_ssm:
        if names != {"gdn_ssm"}:
            raise SeqOnLaneError(
                f"GDN SSM layout requires exactly gdn_ssm, got {sorted(names)} ({tag})"
            )
        offset, stride, unit, heads, per_stride = layout["gdn_ssm"]
        head_bytes = unit * per_stride
        if offset != 0 or stride != head_bytes or heads != src_values:
            raise SeqOnLaneError(
                "GDN SSM compact layout must be dense and head-major over "
                f"the rank's {src_values} value heads ({tag})"
            )
        taps, src_row, dst_row = 1, 0, 0
        runs = [
            (shard, [(value * head_bytes, 0, dst_values * head_bytes)])
            for shard, _, value in targets
        ]
    elif is_conv and names == {"gdn_conv_qk", "gdn_conv_v"}:
        qk, v = layout["gdn_conv_qk"], layout["gdn_conv_v"]
        taps = qk[3]
        qk_bytes = qk[2] * qk[4]
        value_bytes = v[2]
        src_row = qk_bytes + value_bytes * v[4]
        pair_bytes = qk_bytes // src_keys  # Q and K rows of one key head
        if (
            qk[0] != 0
            or v[0] != qk_bytes
            or qk[1] != src_row
            or v[1] != src_row
            or v[3] != taps
            or v[4] != src_values
            or qk_bytes % (2 * src_keys)
        ):
            raise SeqOnLaneError(
                "GDN conv compact layout must be dense tap-major [QK | V] "
                f"over the rank's heads ({tag})"
            )
        dst_qk = dst_keys * pair_bytes
        dst_row = dst_qk + dst_values * value_bytes
        if conv_lane_bytes is None or conv_lane_bytes <= 0:
            raise SeqOnLaneError(
                "QK pair-blocked GDN fan-out requires the conv lane width"
            )
        src_width = math.gcd(src_row, conv_lane_bytes, qk_bytes // 2)
        dst_width = math.gcd(dst_row, conv_lane_bytes, dst_qk // 2)
        if src_width != dst_width or (pair_bytes // 2) % src_width:
            raise SeqOnLaneError(
                "producer and destination QK row pairs differ: "
                f"row bytes {src_width} vs {dst_width}, key head bytes "
                f"{pair_bytes // 2} ({tag})"
            )
        runs = [
            (
                shard,
                [
                    (key * pair_bytes, 0, dst_qk),
                    (qk_bytes + value * value_bytes, dst_qk, dst_values * value_bytes),
                ],
            )
            for shard, key, value in targets
        ]
    elif is_conv and names == {"gdn_conv_q", "gdn_conv_k", "gdn_conv_v"}:
        q, k, v = layout["gdn_conv_q"], layout["gdn_conv_k"], layout["gdn_conv_v"]
        taps = q[3]
        key_bytes, value_bytes = q[2], v[2]
        src_q = key_bytes * src_keys
        src_row = 2 * src_q + value_bytes * src_values
        if (
            q[0] != 0
            or k[0] != src_q
            or v[0] != 2 * src_q
            or k[2] != key_bytes
            or q[4] != src_keys
            or k[4] != src_keys
            or v[4] != src_values
            or k[3] != taps
            or v[3] != taps
            or any(region[1] != src_row for region in (q, k, v))
        ):
            raise SeqOnLaneError(
                "GDN conv compact layout must be dense tap-major "
                f"[Q | K | V] over the rank's heads ({tag})"
            )
        dst_q = dst_keys * key_bytes
        dst_row = 2 * dst_q + dst_values * value_bytes
        runs = [
            (
                shard,
                [
                    (key * key_bytes, 0, dst_q),
                    (src_q + key * key_bytes, dst_q, dst_q),
                    (
                        2 * src_q + value * value_bytes,
                        2 * dst_q,
                        dst_values * value_bytes,
                    ),
                ],
            )
            for shard, key, value in targets
        ]
    else:
        raise SeqOnLaneError(
            f"unsupported GDN state tag or regions: {tag!r} {sorted(names)}"
        )

    spans: list[PoolByteSpan] = []
    for shard, shard_runs in runs:
        for src_offset, dst_offset, size in shard_runs:
            for name, extent in (
                ("source offset", src_offset),
                ("destination offset", dst_offset),
                ("size", size),
                ("source row", src_row),
                ("destination row", dst_row),
            ):
                if extent % physical_granule_bytes:
                    raise SeqOnLaneError(
                        f"GDN fan-out {name} {extent} B is not a whole "
                        f"physical granule ({physical_granule_bytes} B) ({tag})"
                    )
            spans.append(
                PoolByteSpan(
                    src_block_ordinal=0,
                    src_offset_bytes=src_offset,
                    dst_block_index=0,
                    dst_offset_bytes=dst_offset,
                    size_bytes=size,
                    src_stride_bytes=src_row,
                    dst_stride_bytes=dst_row,
                    count=taps,
                    dst_unit_ordinal=shard,
                )
            )
    return PoolSpanRegistration(
        tag=tag,
        block_ids=(int(block_id),),
        spans=tuple(spans),
        declared_bytes=sum(span.size_bytes * span.count for span in spans),
    )


def registration_source_bytes(registration: PoolSpanRegistration) -> int:
    """Distinct source bytes a registration reads: the union over every span
    repeat, so source bytes copied to several destinations count once."""
    intervals = sorted(
        (
            span.src_block_ordinal,
            span.src_offset_bytes + repeat * span.src_stride_bytes,
            span.src_offset_bytes + repeat * span.src_stride_bytes + span.size_bytes,
        )
        for span in registration.spans
        for repeat in range(span.count)
    )
    total = 0
    current: tuple[int, int, int] | None = None
    for ordinal, start, end in intervals:
        if current is not None and current[0] == ordinal and start <= current[2]:
            current = (ordinal, current[1], max(current[2], end))
            continue
        if current is not None:
            total += current[2] - current[1]
        current = (ordinal, start, end)
    if current is not None:
        total += current[2] - current[1]
    return total


def _default_layout_getter(tensor: Any) -> Any:
    from torch_tpu._internal.compile import tpu_torch_compile

    return tpu_torch_compile.get_device_layout_if_materialized(tensor)


def canonical_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def measured_sol_layout_fingerprint(
    manifest: PoolManifest,
    *,
    layout_getter: Callable[[Any], Any] | None = None,
    package_version: Callable[[str], str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """E0' identity of the admitted seq-on-lane FA storage.

    Same tile / minor-to-major / element gates as the token-major
    fingerprint; the payload additionally pins the layout family and the
    kernel page so a token-major peer (identical byte sizes at TP1!) or a
    peer on another page size fails closed at plan time, and deliberately
    excludes per-role facts (KV row count) so a producer and its TP2
    consumers hash identically.
    """
    geometry = sol_fa_geometry_from_manifest(manifest)
    fa_pool = next(pool for pool in manifest.pools if class_tag(pool.tag) == TAG_FA)
    tensor = manifest.storages[fa_pool.storage_index]
    getter = layout_getter or _default_layout_getter
    layout = getter(tensor)
    if layout is None:
        raise RuntimeError(
            "admitted seq-on-lane FA storage has no materialized TPU layout"
        )
    raw_minor_to_major, raw_tiles, raw_element_bits = layout
    minor_to_major = tuple(int(dim) for dim in raw_minor_to_major)
    tiles = tuple(tuple(int(dim) for dim in tile) for tile in raw_tiles)
    element_bits = int(raw_element_bits or 8)
    if minor_to_major != EXPECTED_SOL_MINOR_TO_MAJOR:
        raise RuntimeError(
            "seq-on-lane FA layout failed E0' minor-to-major "
            f"gate: measured={minor_to_major} "
            f"expected={EXPECTED_SOL_MINOR_TO_MAJOR}"
        )
    if tiles != EXPECTED_SOL_TILES:
        raise RuntimeError(
            "seq-on-lane FA layout failed E0' tile gate: "
            f"measured={tiles} expected={EXPECTED_SOL_TILES}"
        )
    if element_bits != 8:
        raise RuntimeError(
            "seq-on-lane FA layout failed E0' element-size "
            f"gate: measured={element_bits} expected=8"
        )
    version = package_version or importlib.metadata.version
    qk_pair = bool(getattr(tpu_envs, "TPU_GDN_CONV_QK_PAIR_LAYOUT", True))
    payload = {
        "schema": SOL_FINGERPRINT_SCHEMA,
        "torch_tpu": version("torch_tpu"),
        "libtpu": version("libtpu"),
        "minor_to_major": list(minor_to_major),
        "tiles": [list(tile) for tile in tiles],
        "element_size_in_bits": element_bits,
        "fa_kv_layout": SOL_LAYOUT_TOKEN,
        "fa_kernel_page_tokens": geometry.page_tokens,
        "fa_packing": geometry.packing,
        "pool_lane_width": LANE_WIDTH,
        "gdn_conv_layout": "qk-pair-v1" if qk_pair else "legacy-split-qk",
    }
    return canonical_fingerprint(payload), payload


__all__ = [
    "EXPECTED_SOL_MINOR_TO_MAJOR",
    "EXPECTED_SOL_TILES",
    "GDN_CONV_ITEMSIZE",
    "LANE_WIDTH",
    "PACKING",
    "SOL_FINGERPRINT_SCHEMA",
    "SOL_LAYOUT_TOKEN",
    "SOL_REGION_NAME",
    "WORD_ROW_BYTES",
    "SeqOnLaneError",
    "SolFaGeometry",
    "SolFaLayout",
    "build_qwen35_pool_manifest_sol",
    "canonical_fingerprint",
    "lower_fa_spans_sol",
    "lower_gdn_state_fanout_spans_sol",
    "lower_gdn_state_shard_spans_sol",
    "measured_sol_layout_fingerprint",
    "registration_source_bytes",
    "sol_fa_geometry_from_manifest",
    "sol_fa_geometry_from_shape",
    "sol_fa_page_tokens",
    "sol_fa_regions",
    "sol_fa_skip_bytes",
    "sol_gdn_conv_lane_bytes",
]
