# SPDX-License-Identifier: Apache-2.0
"""Whole-page transfer descriptors bound to live Qwen pool owners.

This describes logical byte extents. Admission must still measure the TPU
layout and compare peer fingerprints before registering these descriptors.
"""

import dataclasses
import importlib.metadata
import math
import os
from collections.abc import Callable
from typing import Any

import torch
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_torchtpu.distributed.kv_transfer.raiden.layout_fingerprint import (
    EXPECTED_FA_MINOR_TO_MAJOR,
    EXPECTED_FA_TILES,
    canonical_layout_fingerprint,
)
from vllm_torchtpu.distributed.kv_transfer.raiden.pool_manifest import (
    BINDING_ALIASED_RAW,
    ManifestError,
    PoolEntry,
    PoolManifest,
    RegionSpec,
    verify_storage_binding,
)
from vllm_torchtpu.kv_cache_materializer import (
    MaterializedKVCache,
    layer_to_pool_index,
)

from .pd import build_transfer_groups


def compressed_live_bytes(layout) -> int | None:
    """Bytes of an unpacked compressed-key page that hold keys, or None.

    One layer's keys for a page's tokens fill its first block_size / 128
    kernel pages (409,600 of 3,276,800 bytes on a TP1 stage's 3200-token
    page); the rest is padding a transfer need not carry.
    """
    if layout is None or getattr(layout, "compressed_slots", 1) != 1:
        return None
    if getattr(layout, "merged_compressed", False):
        # Merged keys are whole blocks of their own pool (no padding to trim).
        return None
    live = (layout.block_size // layout.kernel_block_size) * layout.compressed_kernel_bytes
    return live if 0 < live < layout.page_bytes else None


def pool_tag(group_index: int, name: str, *, pcp_rank: int = 0, split: bool = False) -> str:
    """Raiden pool tag of a cache owner's pages.

    A PCP producer is one transfer unit per PCP rank of a stage. Its main-KV
    owners hold different tokens on every rank (``split``), so their tags
    carry the rank; the other owners are identical on every rank, so only
    rank 0 lists them, under the plain tag (``build_transfer_manifest`` drops
    them from the other ranks' manifests)."""
    tag = f"qwen4.g{group_index}.{name}"
    if split or pcp_rank > 0:
        tag += f".pcp{pcp_rank}"
    return tag


def qwen_staging_hints(
    pools, block_tokens: int, max_model_len: int, compressed_slots: int = 1
) -> list[int]:
    """Bounded host staging hints for a Qwen PD producer's pools, in pool order.

    tpu_sync sizes each storage's host arena as ``leases x max(hint)`` slots of
    one storage page, and a transfer leases one slot per distinct source block
    id it touches on that storage. The i-th layer of every cache group shares
    carrier i, and each group has its own block table, so a max-length
    transfer touches, per storage, ceil(max_model_len / block_tokens) pages of
    each main-KV and compressed-key group there and one page of each ring,
    PLE and GDN group. (Compressed-key sub-pages, ``compressed_slots`` > 1,
    pack that many blocks per physical page.)

    Every pool then reports the manager's largest storage sum (uniform):
    tpu_sync grows every host layer to the largest arena anyway, so a smaller
    hint saves no host memory, while a storage holding only GDN layers would
    otherwise get an arena of N x 2 slots and admit only N transfers at once.
    A pool of unknown kind gives all zeros, which keeps the full mirror.
    """
    from .pd_staging import owner_kind

    pools = list(pools)
    if not pools:
        return []
    if block_tokens <= 0 or max_model_len <= 0 or compressed_slots <= 0:
        raise ValueError("staging hints need positive block, length and slots")
    pages = math.ceil(max_model_len / block_tokens)
    per_kind = {
        "main": pages,
        "compressed": math.ceil(pages / compressed_slots),
        "ring": 1,
        "ple": 1,
        "gdn": 1,
    }
    # storage -> {group: pages}; a group's layers sit on distinct storages
    by_storage: dict[int, dict[str, int]] = {}
    for pool in pools:
        try:
            kind = owner_kind(str(pool.layer_name))
        except ValueError:
            return [0] * len(pools)
        group = str(pool.tag).split(".", 2)[1] if str(pool.tag).startswith("qwen4.") else str(pool.tag)
        groups = by_storage.setdefault(int(pool.storage_index), {})
        groups[group] = max(groups.get(group, 0), per_kind[kind])
    hint = max(sum(groups.values()) for groups in by_storage.values())
    return [hint] * len(pools)


def page_regions(name: str, page_bytes: int, compressed_live: int | None):
    """A pool's live region: whole pages, or a compressed-key page's keys."""
    if compressed_live is not None and name.endswith(".compressed_k"):
        return (RegionSpec("keys", 0, page_bytes, compressed_live, 1),)
    return (RegionSpec("page", 0, page_bytes, page_bytes, 1),)


def _leading_page_region(pool) -> bool:
    """One region at the start of each page: the whole page, or (compressed
    keys) its leading live bytes."""
    if len(pool.regions) != 1:
        return False
    region = pool.regions[0]
    stride = pool.block_stride_bytes
    return (
        region.offset_bytes == 0
        and region.stride_bytes == stride
        and region.num_units == 1
        and region.units_per_stride == 1
        and 0 < region.unit_bytes <= stride
        and (region.name == "page") == (region.unit_bytes == stride)
    )


def _group_of(groups, name: str) -> int:
    for group in groups:
        if name in group.layer_names:
            return group.group_index
    raise ManifestError(f"Qwen PD owner {name} has no group")


def merged_key_owner(name: str, owners) -> str | None:
    """The main-KV owner whose carrier region holds merged keys ``name``."""
    if name.endswith(".compressed_k") and name not in owners:
        main = name.removesuffix(".compressed_k")
        if main in owners:
            return main
    return None


def build_transfer_manifest(
    config: KVCacheConfig, materialized: MaterializedKVCache, layout=None
) -> PoolManifest:
    """Bind every scheduler group to full pages in the actual FP8 carrier.

    Distinct groups may share a carrier but use different block tables. Tags
    therefore identify the group and layer; transport must filter by group.
    Copying entire pages preserves all overlaid state bytes without a dtype
    conversion. This first path requires matching layer-major peer layouts.
    """
    if materialized.bm_pool_layout is not None:
        raise NotImplementedError("Qwen PD initially requires layer-major pools")
    if config.num_blocks <= 0:
        raise ManifestError("Qwen PD needs a positive pool capacity")
    groups = build_transfer_groups(config.kv_cache_groups)
    owners = {name for group in groups for name in group.layer_names}
    if owners != set(materialized.kv_caches) or owners != set(
        materialized.layer_to_raw
    ):
        raise ManifestError("Qwen PD must bind every live cache owner exactly once")
    placements = layer_to_pool_index(config)
    storages = materialized.raw_tensors
    if len({id(raw) for raw in storages}) != len(storages):
        raise ManifestError("Qwen PD raw storage list repeats a carrier")
    pools = []
    referenced = set()
    compressed_live = compressed_live_bytes(layout)
    from vllm_torchtpu.distributed.pcp import get_pcp_rank, get_pcp_world_size

    pcp = get_pcp_world_size()
    pcp_rank = get_pcp_rank() if pcp > 1 else 0
    split_owners = set(getattr(layout, "main_dcp_owners", ())) if pcp > 1 else set()
    for group in groups:
        page_bytes = config.kv_cache_groups[
            group.group_index
        ].kv_cache_spec.page_size_bytes
        used = set()
        for name in group.layer_names:
            if pcp_rank > 0 and name not in split_owners:
                # A PCP rank past 0 holds the same pages as rank 0 for every
                # owner but the split main KV; the transport pushes every pool
                # a unit lists, and the decode registers only rank 0's copy.
                continue
            index = placements[name]
            if index < 0 or index >= len(storages):
                raise ManifestError(f"Qwen PD owner {name} has no raw carrier")
            raw = storages[index]
            cache = materialized.kv_caches[name]
            tensors = cache if isinstance(cache, list) else [cache]
            if (
                materialized.layer_to_raw[name] is not raw
                or len(tensors) != 1
                or tensors[0] is not raw
            ):
                raise ManifestError(f"Qwen PD owner {name} is not its live carrier")
            if index in used:
                raise ManifestError("Two owners in one group overlap the same carrier")
            used.add(index)
            referenced.add(index)
            region_pages = (
                layout.compressed_region_pages(config.num_blocks)
                if getattr(layout, "merged_compressed", False)
                else 0
            )
            if (
                raw.dtype != torch.float8_e4m3fn
                or raw.storage_offset() != 0
                or not raw.is_contiguous()
                or raw.numel() * raw.element_size()
                != (config.num_blocks + region_pages) * page_bytes
            ):
                raise ManifestError(f"Qwen PD owner {name} has incompatible FP8 pages")
            pools.append(
                PoolEntry(
                    tag=pool_tag(
                        group.group_index, name, pcp_rank=pcp_rank, split=name in split_owners
                    ),
                    layer_name=name,
                    storage_index=index,
                    base_offset_bytes=0,
                    block_stride_bytes=page_bytes,
                    num_blocks=config.num_blocks,
                    regions=page_regions(name, page_bytes, compressed_live),
                    dtype_tag="qwen4-whole-fp8-page-v1",
                )
            )
    merged = bool(getattr(layout, "merged_compressed", False))
    split_keys = merged and bool(getattr(layout, "split_compressed", False))
    if merged and (pcp_rank == 0 or split_keys):
        # QWEN4_EXP_MERGE_COMPRESSED_K: each QSA layer's compressed keys sit in
        # a region of its main-KV carrier after the scheduler's pages, one
        # `merged_key_span` pages' worth of keys per main-KV block ID. Expose
        # them as their own pool on the main-KV group's IDs; the keys are the
        # same on every PCP rank, so only rank 0 lists them -- unless the ranks
        # split them by page owner (QWEN4_EXP_PCP_SPLIT_COMPRESSED_K), when
        # every rank lists its own, under a per-rank tag.
        span = layout.merged_key_span
        key_stride = span * layout.page_bytes // layout.compressed_columns_per_page
        main_pools = [p for p in pools if p.layer_name.endswith(".self_attn")]
        for main in main_pools:
            name = main.layer_name + ".compressed_k"
            pools.append(
                PoolEntry(
                    tag=pool_tag(
                        _group_of(groups, main.layer_name), name,
                        pcp_rank=pcp_rank, split=split_keys,
                    ),
                    layer_name=name,
                    storage_index=main.storage_index,
                    base_offset_bytes=config.num_blocks * main.block_stride_bytes,
                    block_stride_bytes=key_stride,
                    num_blocks=config.num_blocks,
                    regions=(RegionSpec("page", 0, key_stride, key_stride, 1),),
                    dtype_tag="qwen4-whole-fp8-page-v1",
                )
            )
    if pcp_rank == 0 and referenced != set(range(len(storages))):
        raise ManifestError("Qwen PD contains an unreferenced raw carrier")
    unit_storages = list(storages)
    if pcp_rank > 0:
        # Only the split owners' carriers, in the order the pools first use
        # them, so pool order and storage order agree: a tpu_sync sender that
        # resolves a pool's host buffer by position (builds before tpu-raiden
        # 4e61045) otherwise copies from another carrier's buffer, and the
        # device DMA faults.
        order = list(dict.fromkeys(pool.storage_index for pool in pools))
        position = {index: i for i, index in enumerate(order)}
        unit_storages = [storages[index] for index in order]
        pools = [
            dataclasses.replace(pool, storage_index=position[pool.storage_index])
            for pool in pools
        ]
    manifest = PoolManifest(BINDING_ALIASED_RAW, unit_storages, pools)
    verify_storage_binding(manifest, materialized.kv_caches, storages)
    return manifest


def _device_layout(tensor: torch.Tensor) -> Any:
    from torch_tpu._internal.compile import tpu_torch_compile

    return tpu_torch_compile.get_device_layout_if_materialized(tensor)


def measure_transfer_layout(
    manifest: PoolManifest,
    *,
    layout_getter: Callable[[torch.Tensor], Any] = _device_layout,
    package_version: Callable[[str], str] = importlib.metadata.version,
) -> tuple[str, dict[str, Any]]:
    """Fail closed unless every live carrier has whole, unpadded native pages.

    Matching peers may have different capacities, but must have the same page
    encoding. Model/checkpoint, scales and rank ownership are separate admission
    checks. This observation does not fence future device writes or transfers.
    """
    if not manifest.storages or not manifest.pools:
        raise ManifestError("Qwen PD layout needs live pools")
    carriers = []
    for index, raw in enumerate(manifest.storages):
        layout = layout_getter(raw)
        if layout is None:
            raise ManifestError("Qwen PD carrier has no materialized TPU layout")
        order, tiles, bits = layout
        order = tuple(int(x) for x in order)
        tiles = tuple(tuple(int(x) for x in tile) for tile in tiles)
        shape = tuple(raw.shape)
        if (
            raw.dtype != torch.float8_e4m3fn
            or len(shape) != len(EXPECTED_FA_MINOR_TO_MAJOR)
            or order != EXPECTED_FA_MINOR_TO_MAJOR
            or tiles != EXPECTED_FA_TILES
            or bits not in (0, 8)
        ):
            raise ManifestError("Qwen PD carrier has an unsupported native layout")
        # The admitted nested tiling only tiles the two innermost axes. Both
        # must be full tiles so physical padding cannot alter page strides.
        if any(dim % tile for dim, tile in zip(shape[-2:], tiles[0])):
            raise ManifestError("Qwen PD carrier would have native tile padding")
        row_bytes = math.prod(shape[1:])
        token_row = row_bytes // shape[1]
        pools = [p for p in manifest.pools if p.storage_index == index]
        if not pools:
            raise ManifestError("Qwen PD layout has an unreferenced carrier")
        for pool in pools:
            # Whole pages from offset 0; merged compressed keys
            # (QWEN4_EXP_MERGE_COMPRESSED_K) add whole-row blocks in the region
            # after them. Either way a pool stays inside its carrier.
            whole = pool.base_offset_bytes == 0 and pool.block_stride_bytes % row_bytes == 0
            keys = (
                pool.layer_name.endswith(".compressed_k")
                and pool.block_stride_bytes % token_row == 0
                and pool.base_offset_bytes % row_bytes == 0
            )
            if (
                pool.block_stride_bytes <= 0
                or not (whole or keys)
                or pool.base_offset_bytes + pool.num_blocks * pool.block_stride_bytes
                > raw.nbytes
                or not _leading_page_region(pool)
            ):
                raise ManifestError("Qwen PD pages do not partition the native carrier")
        carriers.append(
            dict(
                shape_per_kernel_page=list(shape[1:]),
                minor_to_major=list(order),
                tiles=[list(t) for t in tiles],
                element_bits=8,
            )
        )
    if any(
        p.storage_index < 0 or p.storage_index >= len(carriers) for p in manifest.pools
    ):
        raise ManifestError("Qwen PD pool references an unknown carrier")
    # TPU_RAIDEN_FINGERPRINT_VERSION_OVERRIDE=torch_tpu=X,libtpu=Y states the
    # runtime versions a peer on another runtime build reports, for legs that
    # run different torch_tpu builds over the same page encoding (a PP x PCP
    # producer needs a torch_tpu with stage meshes; its decode may not).
    versions = {
        "torch_tpu": package_version("torch_tpu"),
        "libtpu": package_version("libtpu"),
    }
    override = os.environ.get("TPU_RAIDEN_FINGERPRINT_VERSION_OVERRIDE", "")
    for item in filter(None, (part.strip() for part in override.split(","))):
        name, _, value = item.partition("=")
        if name not in versions or not value:
            raise ManifestError(
                f"TPU_RAIDEN_FINGERPRINT_VERSION_OVERRIDE entry {item!r}: "
                "expected torch_tpu=<version> or libtpu=<version>"
            )
        versions[name] = value
    payload = dict(
        schema="qwen4-whole-fp8-page-layout-v1",
        torch_tpu=versions["torch_tpu"],
        libtpu=versions["libtpu"],
        carriers=carriers,
        pools=[
            dict(
                tag=p.tag,
                storage_index=p.storage_index,
                block_stride_bytes=p.block_stride_bytes,
                dtype_tag=p.dtype_tag,
            )
            for p in manifest.pools
        ],
    )
    return canonical_layout_fingerprint(payload), payload
