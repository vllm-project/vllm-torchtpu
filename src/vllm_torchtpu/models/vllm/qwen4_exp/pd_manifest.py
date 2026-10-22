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
)

from .pd import build_transfer_groups


def build_transfer_manifest(
    config: KVCacheConfig, materialized: MaterializedKVCache
) -> PoolManifest:
    """Bind scheduler owners and co-owned compressed keys to live byte pages."""
    if materialized.bm_pool_layout is not None:
        raise NotImplementedError("Qwen PD initially requires layer-major pools")
    if config.num_blocks <= 0:
        raise ManifestError("Qwen PD needs a positive pool capacity")
    groups = build_transfer_groups(config.kv_cache_groups)
    owners = {name for group in groups for name in group.layer_names}
    aliases = {name + ".compressed_k" for name in owners if name.endswith(".self_attn")}
    if owners != set(materialized.layer_to_raw) or owners | aliases != set(
        materialized.kv_caches
    ):
        raise ManifestError("Qwen PD must bind every live cache owner exactly once")
    storages = materialized.raw_tensors
    indices = {id(raw): index for index, raw in enumerate(storages)}
    if len(indices) != len(storages):
        raise ManifestError("Qwen PD raw storage list repeats a carrier")
    pools = []
    referenced = set()
    for group in groups:
        used = set()
        for owner in group.layer_names:
            names = (
                (owner, owner + ".compressed_k")
                if owner.endswith(".self_attn")
                else (owner,)
            )
            for name in names:
                cache = materialized.kv_caches[name]
                tensors = cache if isinstance(cache, list) else [cache]
                if len(tensors) != 1 or id(tensors[0]) not in indices:
                    raise ManifestError(f"Qwen PD owner {name} is not its live carrier")
                raw = tensors[0]
                if name == owner and materialized.layer_to_raw[owner] is not raw:
                    raise ManifestError(f"Qwen PD owner {name} is not its live carrier")
                index = indices[id(raw)]
                if index in used:
                    raise ManifestError(
                        "Two owners in one group overlap the same carrier"
                    )
                used.add(index)
                referenced.add(index)
                is_index = name.endswith((".compressed_k", ".raw_ring"))
                dtype = torch.bfloat16 if is_index else torch.float8_e4m3fn
                if (
                    raw.dtype != dtype
                    or raw.storage_offset() != 0
                    or not raw.is_contiguous()
                    or len(raw.shape) != len(EXPECTED_FA_MINOR_TO_MAJOR)
                    or raw.shape[0] != config.num_blocks
                    or raw.nbytes % config.num_blocks
                ):
                    raise ManifestError(
                        f"Qwen PD owner {name} has incompatible carrier pages"
                    )
                # The scheduler spec is main+index *budget*, not either stride.
                page_bytes = raw.nbytes // config.num_blocks
                if page_bytes <= 0:
                    raise ManifestError("Qwen PD needs nonempty carrier pages")
                role = "index" if is_index else "main"
                pools.append(
                    PoolEntry(
                        tag=f"qwen4.g{group.group_index}.{name}",
                        layer_name=name,
                        storage_index=index,
                        base_offset_bytes=0,
                        block_stride_bytes=page_bytes,
                        num_blocks=config.num_blocks,
                        regions=(RegionSpec("page", 0, page_bytes, page_bytes, 1),),
                        dtype_tag=f"qwen4-{role}-{dtype}-page-v2",
                    )
                )
    if referenced != set(range(len(storages))):
        raise ManifestError("Qwen PD contains an unreferenced raw carrier")
    # A compressed alias and ring must name the same index carrier, but retain
    # main and ring block-table lifetimes respectively.
    for name in aliases:
        ring = name.removesuffix(".compressed_k") + ".raw_ring"
        if (
            ring not in owners
            or materialized.kv_caches[name] is not materialized.kv_caches[ring]
        ):
            raise ManifestError("Qwen PD compressed/ring bindings disagree")
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
        # RPA packs one 32-bit word on the penultimate axis for both dtypes.
        packing = torch.int32.itemsize // raw.element_size()
        if (
            raw.dtype not in (torch.float8_e4m3fn, torch.bfloat16)
            or len(shape) != len(EXPECTED_FA_MINOR_TO_MAJOR)
            or order != EXPECTED_FA_MINOR_TO_MAJOR
            or tiles
            != (
                (packing, EXPECTED_FA_TILES[0][1]),
                (packing, 1),
            )
            or bits not in (0, raw.element_size() * 8)
        ):
            raise ManifestError("Qwen PD carrier has an unsupported native layout")
        # The admitted nested tiling only tiles the two innermost axes. Both
        # must be full tiles so physical padding cannot alter page strides.
        if any(dim % tile for dim, tile in zip(shape[-2:], tiles[0])):
            raise ManifestError("Qwen PD carrier would have native tile padding")
        row_bytes = math.prod(shape[1:]) * raw.element_size()
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
                element_bits=raw.element_size() * 8,
                dtype=str(raw.dtype),
            )
        )
    if any(
        p.storage_index < 0 or p.storage_index >= len(carriers) for p in manifest.pools
    ):
        raise ManifestError("Qwen PD pool references an unknown carrier")
    payload = dict(
        schema="qwen4-dual-carrier-page-layout-v2",
        torch_tpu=package_version("torch_tpu"),
        libtpu=package_version("libtpu"),
        carriers=carriers,
        pools=[
            dict(
                tag=p.tag,
                storage_index=p.storage_index,
                layer_name=p.layer_name,
                regions=[r.to_dict() for r in p.regions],
                block_stride_bytes=p.block_stride_bytes,
                dtype_tag=p.dtype_tag,
            )
            for p in manifest.pools
        ],
    )
    return canonical_layout_fingerprint(payload), payload
