# SPDX-License-Identifier: Apache-2.0
"""Whole-page transfer descriptors bound to live Qwen pool owners.

This describes logical byte extents. Admission must still measure the TPU
layout and compare peer fingerprints before registering these descriptors.
"""

import importlib.metadata
import math
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


def build_transfer_manifest(
    config: KVCacheConfig, materialized: MaterializedKVCache
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
    for group in groups:
        page_bytes = config.kv_cache_groups[
            group.group_index
        ].kv_cache_spec.page_size_bytes
        used = set()
        for name in group.layer_names:
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
            if (
                raw.dtype != torch.float8_e4m3fn
                or raw.storage_offset() != 0
                or not raw.is_contiguous()
                or raw.numel() * raw.element_size() != config.num_blocks * page_bytes
            ):
                raise ManifestError(f"Qwen PD owner {name} has incompatible FP8 pages")
            pools.append(
                PoolEntry(
                    tag=f"qwen4.g{group.group_index}.{name}",
                    layer_name=name,
                    storage_index=index,
                    base_offset_bytes=0,
                    block_stride_bytes=page_bytes,
                    num_blocks=config.num_blocks,
                    regions=(RegionSpec("page", 0, page_bytes, page_bytes, 1),),
                    dtype_tag="qwen4-whole-fp8-page-v1",
                )
            )
    if referenced != set(range(len(storages))):
        raise ManifestError("Qwen PD contains an unreferenced raw carrier")
    manifest = PoolManifest(BINDING_ALIASED_RAW, list(storages), pools)
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
        pools = [p for p in manifest.pools if p.storage_index == index]
        if not pools:
            raise ManifestError("Qwen PD layout has an unreferenced carrier")
        for pool in pools:
            if (
                pool.base_offset_bytes != 0
                or pool.block_stride_bytes <= 0
                or pool.block_stride_bytes % row_bytes
                or pool.num_blocks * pool.block_stride_bytes != raw.nbytes
                or pool.regions
                != (
                    RegionSpec(
                        "page", 0, pool.block_stride_bytes, pool.block_stride_bytes, 1
                    ),
                )
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
    payload = dict(
        schema="qwen4-whole-fp8-page-layout-v1",
        torch_tpu=package_version("torch_tpu"),
        libtpu=package_version("libtpu"),
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
