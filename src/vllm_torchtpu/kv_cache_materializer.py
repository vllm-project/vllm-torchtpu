import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.utils import AttentionGroup

from vllm_torchtpu.block_major_pool import BlockMajorPoolLayout

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

type LayerKVCache = torch.Tensor | list[torch.Tensor]


@dataclass(frozen=True)
class MaterializedKVCache:
    kv_caches: dict[str, LayerKVCache]
    raw_tensors: list[torch.Tensor]
    # Maps each unified-pool layer to its backing raw pool tensor; empty when
    # layers use dedicated per-layer caches.
    layer_to_raw: dict[str, torch.Tensor] = field(default_factory=dict)
    # Populated when unified pools are merged into a single block-major pool
    # (see `vllm_torchtpu.block_major_pool`), in which case `raw_tensors`
    # contains only that merged pool.
    bm_pool_layout: BlockMajorPoolLayout | None = None


def _storage_ptr(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def _raw_tensor_index(
    tensor: torch.Tensor,
    raw_tensors: Sequence[torch.Tensor],
) -> int | None:
    storage_ptr = _storage_ptr(tensor)
    for idx, raw in enumerate(raw_tensors):
        if _storage_ptr(raw) == storage_ptr:
            return idx
    return None


def _tensor_layout_summary(
    tensor: torch.Tensor,
    raw_tensors: Sequence[torch.Tensor],
) -> str:
    raw_idx = _raw_tensor_index(tensor, raw_tensors)
    storage_offset_bytes = tensor.storage_offset() * tensor.element_size()
    return (
        f"shape={tuple(tensor.shape)} stride={tuple(tensor.stride())} "
        f"dtype={tensor.dtype} contiguous={tensor.is_contiguous()} "
        f"storage_offset={tensor.storage_offset()} "
        f"storage_offset_bytes={storage_offset_bytes} "
        f"numel={tensor.numel()} nbytes={tensor.nbytes} raw_idx={raw_idx}"
    )


def format_kv_cache_layout_summary(
    *,
    kv_cache_config: KVCacheConfig,
    kv_caches: dict[str, LayerKVCache],
    raw_tensors: Sequence[torch.Tensor],
    attn_groups: Sequence[Sequence[AttentionGroup]] | None = None,
) -> str:
    """Format stable KV/mamba cache layout diagnostics for TPU validation."""
    group_backend_names: dict[int, str] = {}
    if attn_groups is not None:
        for group_list in attn_groups:
            for group in group_list:
                group_backend_names[group.kv_cache_group_id] = group.backend.__name__

    lines = [
        "TPU KV cache layout summary:",
        f"  num_blocks={kv_cache_config.num_blocks} "
        f"num_groups={len(kv_cache_config.kv_cache_groups)} "
        f"num_raw_tensors={len(raw_tensors)}",
    ]
    for idx, raw in enumerate(raw_tensors):
        lines.append(
            f"  raw[{idx}]: shape={tuple(raw.shape)} dtype={raw.dtype} "
            f"numel={raw.numel()} nbytes={raw.nbytes}"
        )

    for gid, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        spec_name = type(spec).__name__
        backend_name = group_backend_names.get(gid, "<none>")
        lines.append(
            f"  group[{gid}]: spec={spec_name} "
            f"backend={backend_name} "
            f"block_size={spec.block_size} "
            f"page_size_bytes={spec.page_size_bytes} "
            f"layers={list(group.layer_names)}"
        )
        for layer_name in group.layer_names:
            cache = kv_caches.get(layer_name)
            if cache is None:
                lines.append(f"    {layer_name}: <missing>")
                continue
            if isinstance(cache, torch.Tensor):
                lines.append(
                    f"    {layer_name}: attention "
                    f"{_tensor_layout_summary(cache, raw_tensors)}"
                )
            else:
                lines.append(f"    {layer_name}: mamba states={len(cache)}")
                for state_idx, state in enumerate(cache):
                    lines.append(
                        f"      state[{state_idx}]: "
                        f"{_tensor_layout_summary(state, raw_tensors)}"
                    )
    return "\n".join(lines)


def _pool_attention_geometry(
    kv_cache_config: KVCacheConfig,
    attn_groups: Sequence[Sequence[AttentionGroup]],
    kernel_block_size_by_gid: dict[int, int],
) -> tuple[AttentionSpec, Any, int] | None:
    """(attention spec, backend, kernel block size) of the unified pool,
    or None when the config has no hybrid attention+mamba sharing."""
    has_mamba = any(
        isinstance(group.kv_cache_spec, MambaSpec)
        for group in kv_cache_config.kv_cache_groups
    )
    if not has_mamba:
        return None
    for group_list in attn_groups:
        for group in group_list:
            spec = group.kv_cache_spec
            if isinstance(spec, AttentionSpec) and not isinstance(
                spec, EncoderOnlyAttentionSpec
            ):
                kernel_block_size = kernel_block_size_by_gid.get(
                    group.kv_cache_group_id, spec.block_size
                )
                return spec, group.backend, kernel_block_size
    return None


def layer_to_pool_index(kv_cache_config: KVCacheConfig) -> dict[str, int]:
    """Maps native vLLM layer placements to disjoint TPU pool region indices.

    vLLM's descriptors all describe one backing allocation. The TPU runtime
    owns one native attention tensor per disjoint layer region instead; equal
    byte ranges across cache groups must remain the exact same pool object.

    Supports both placement orders emitted by vLLM and resolves them to the
    same region index:
    - Layer-outermost (`block_stride == page_bytes`): each region is a
      contiguous `num_blocks * page_bytes` span.
    - Block-outermost (`block_stride == num_regions * page_bytes`): each
      region occupies a `page_bytes` slot within every scheduler block row.
    """
    page_bytes = max(
        group.kv_cache_spec.page_size_bytes for group in kv_cache_config.kv_cache_groups
    )
    pool_bytes = kv_cache_config.num_blocks * page_bytes
    sizes = {tensor.size for tensor in kv_cache_config.kv_cache_tensors}
    assert len(sizes) == 1, "KV placements must share one backing allocation"
    backing_bytes = sizes.pop()
    assert backing_bytes % pool_bytes == 0, (backing_bytes, pool_bytes)
    num_regions = backing_bytes // pool_bytes
    result = {}
    for tensor in kv_cache_config.kv_cache_tensors:
        if tensor.block_stride == page_bytes:
            region_stride = pool_bytes
        else:
            assert tensor.block_stride == num_regions * page_bytes, (
                "Unified TPU pools require layer-compact or block-compact pages",
                tensor,
                page_bytes,
                num_regions,
            )
            region_stride = page_bytes
        for index, name in enumerate(tensor.layers):
            offset = tensor.offset + index * tensor.layer_stride
            assert offset % region_stride == 0, (name, offset, region_stride)
            region = offset // region_stride
            assert 0 <= region < num_regions, (name, tensor, num_regions)
            result[name] = region
    return result


def resolve_block_major_pool_layout(
    kv_cache_config: KVCacheConfig, kernel_block_size: int
) -> BlockMajorPoolLayout:
    """Derives the merged block-major pool geometry from `KVCacheConfig` alone.

    Shared by buffer materialization, the KV offload contract, and Raiden
    pool registration so that the scheduler process (which never allocates
    device memory) and worker processes agree on exact pool geometry.
    """
    spec = next(
        group.kv_cache_spec
        for group in kv_cache_config.kv_cache_groups
        if isinstance(group.kv_cache_spec, AttentionSpec)
    )
    assert spec.block_size % kernel_block_size == 0, (
        spec.block_size,
        kernel_block_size,
    )
    pool_indices = layer_to_pool_index(kv_cache_config)
    num_pools = len(set(pool_indices.values()))
    assert set(pool_indices.values()) == set(range(num_pools)), pool_indices
    return BlockMajorPoolLayout(
        num_blocks=kv_cache_config.num_blocks,
        num_pools=num_pools,
        split=spec.block_size // kernel_block_size,
        pool_index_by_layer=pool_indices,
    )


def allocate_raw_kv_cache_tensors(
    kv_cache_config: KVCacheConfig,
    device: torch.device,
    pool_geometry: tuple[AttentionSpec, Any, int],
    cache_dtype: str | torch.dtype = "auto",
    block_major: bool = False,
) -> tuple[dict[str, torch.Tensor], list[torch.Tensor], BlockMajorPoolLayout | None]:
    layer_to_raw: dict[str, torch.Tensor] = {}
    raw_tensors: list[torch.Tensor] = []
    pool_page_bytes = max(
        group.kv_cache_spec.page_size_bytes for group in kv_cache_config.kv_cache_groups
    )
    pool_indices = layer_to_pool_index(kv_cache_config)
    pool_bytes = kv_cache_config.num_blocks * pool_page_bytes
    num_pools = kv_cache_config.kv_cache_tensors[0].size // pool_bytes
    # The unified pool is born as the stock attention-shaped KV cache.
    # Attention consumes it natively; Mamba adapters address their byte
    # regions inside each manager page without creating typed aliases.
    spec, attn_backend, kernel_block_size = pool_geometry
    assert spec.block_size % kernel_block_size == 0, (
        spec.block_size,
        kernel_block_size,
    )
    split = spec.block_size // kernel_block_size
    num_blocks = kv_cache_config.num_blocks
    region_shape = tuple(
        attn_backend.get_kv_cache_shape(
            num_blocks * split,
            kernel_block_size,
            spec.num_kv_heads,
            spec.head_size,
            cache_dtype_str=cache_dtype,
        )
    )
    page_shape = region_shape[1:]
    fa_page_bytes = math.prod(page_shape) * spec.dtype.itemsize
    if fa_page_bytes * split != pool_page_bytes:
        group_pages = [
            (
                type(g.kv_cache_spec).__name__,
                g.kv_cache_spec.block_size,
                g.kv_cache_spec.page_size_bytes,
            )
            for g in kv_cache_config.kv_cache_groups
        ]
        raise AssertionError(
            f"pool page mismatch: fa_page_bytes={fa_page_bytes} "
            f"split={split} pool_page_bytes={pool_page_bytes} "
            f"kernel_block_size={kernel_block_size} "
            f"region_shape={region_shape} group_pages={group_pages}"
        )
    layout = None
    if block_major:
        layout = resolve_block_major_pool_layout(kv_cache_config, kernel_block_size)
        assert layout.num_pools == num_pools, (layout, num_pools)
        # Merge all `P` region pools into a single `(num_blocks, P * split, *page)`
        # allocation where row `b` stores scheduler block `b` across all regions
        # contiguously (see `vllm_torchtpu.block_major_pool`).
        shapes = [(num_blocks, num_pools * split) + page_shape]
    else:
        shapes = [region_shape] * num_pools
    for shape in shapes:
        if spec.dtype in _FP8_DTYPES:
            raw = torch.empty(shape, dtype=spec.dtype, device=device)
        else:
            raw = torch.zeros(shape, dtype=spec.dtype, device=device)
        raw_tensors.append(raw)
    for layer_name, pool_index in pool_indices.items():
        layer_to_raw[layer_name] = raw_tensors[0 if block_major else pool_index]
    return layer_to_raw, raw_tensors, layout


def make_attention_cache_tensor(
    *,
    spec: AttentionSpec,
    attn_backend: Any,
    num_blocks: int,
    kernel_block_size: int,
    cache_dtype: str | torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    num_blocks_per_kv_block = spec.block_size // kernel_block_size
    kernel_num_blocks = num_blocks * num_blocks_per_kv_block
    shape = attn_backend.get_kv_cache_shape(
        kernel_num_blocks,
        kernel_block_size,
        spec.num_kv_heads,
        spec.head_size,
        cache_dtype_str=cache_dtype,
    )
    return torch.zeros(shape, dtype=spec.dtype, device=device)


def can_share_attention_cache(
    specs: Sequence[KVCacheSpec],
    attn_backend: Any,
) -> bool:
    if not specs:
        return False
    first = specs[0]
    if not isinstance(first, AttentionSpec):
        return False
    for spec in specs:
        if (
            not isinstance(spec, AttentionSpec)
            or isinstance(spec, MLAAttentionSpec)
            or spec.dtype != first.dtype
            or spec.block_size != first.block_size
            or spec.page_size_bytes != first.page_size_bytes
        ):
            return False
        shape = attn_backend.get_kv_cache_shape(
            1, spec.block_size, spec.num_kv_heads, spec.head_size, spec.dtype
        )
        if spec.page_size_bytes % (math.prod(shape) * spec.dtype.itemsize):
            return False
    return True


def materialize_shared_attention_cache(
    *,
    layer_specs: Sequence[tuple[str, AttentionSpec]],
    tensor_size: int,
    num_blocks: int,
    attn_backend: Any,
    device: torch.device,
) -> torch.Tensor:
    """Allocate one tensor for attention groups sharing scheduler pages."""
    if not layer_specs:
        raise ValueError("layer_specs must contain an attention layer")
    spec = layer_specs[0][1]
    page_elements = tensor_size // (num_blocks * spec.dtype.itemsize)
    shapes = []
    for name, layer_spec in layer_specs:
        if (
            layer_spec.dtype != spec.dtype
            or layer_spec.block_size != spec.block_size
            or layer_spec.page_size_bytes * num_blocks != tensor_size
        ):
            raise ValueError(f"Incompatible shared KV cache spec for {name}")
        shape = tuple(
            attn_backend.get_kv_cache_shape(
                num_blocks,
                spec.block_size,
                layer_spec.num_kv_heads,
                layer_spec.head_size,
                spec.dtype,
            )
        )
        if page_elements % math.prod(shape[1:]):
            raise ValueError(
                f"Shared pool page must contain whole native pages for {name}"
            )
        shapes.append(shape)
    pool_shape = shapes[0]
    if (
        any(shape != pool_shape for shape in shapes)
        or math.prod(pool_shape[1:]) != page_elements
    ):
        # Keep the packed dtype axis separate when reinterpreting head shapes.
        packing = 4 // spec.dtype.itemsize
        pool_shape = (
            num_blocks,
            spec.block_size,
            packing,
            page_elements // (spec.block_size * packing),
        )
    return torch.zeros(pool_shape, dtype=spec.dtype, device=device)


def _can_allocate_attention_cache_directly(
    kv_cache_config: KVCacheConfig,
) -> bool:
    return all(
        isinstance(group.kv_cache_spec, AttentionSpec)
        for group in kv_cache_config.kv_cache_groups
    )


def _select_representative_kv_cache_spec(kv_cache_spec: KVCacheSpec) -> KVCacheSpec:
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        try:
            return next(iter(kv_cache_spec.kv_cache_specs.values()))
        except StopIteration as exc:
            raise ValueError(
                "UniformTypeKVCacheSpecs must contain at least one KV cache spec"
            ) from exc
    return kv_cache_spec


def build_kernel_block_size_by_group_id(
    *,
    kv_cache_config: KVCacheConfig,
    kernel_block_sizes: Sequence[int],
) -> dict[int, int]:
    kernel_block_size_by_gid: dict[int, int] = {}
    next_kernel_block_size_index = 0
    for gid, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = _select_representative_kv_cache_spec(group.kv_cache_spec)
        if isinstance(spec, EncoderOnlyAttentionSpec):
            continue
        if not isinstance(spec, (AttentionSpec, MambaSpec)):
            raise NotImplementedError(
                f"Unsupported KV cache spec: {type(group.kv_cache_spec)!r}"
            )
        if next_kernel_block_size_index >= len(kernel_block_sizes):
            raise ValueError(
                f"kernel_block_sizes is missing an entry for KV cache group {gid}"
            )
        kernel_block_size = int(kernel_block_sizes[next_kernel_block_size_index])
        if kernel_block_size <= 0:
            raise ValueError(
                "kernel_block_sizes must contain positive "
                f"values, got {kernel_block_size}"
            )
        kernel_block_size_by_gid[gid] = kernel_block_size
        next_kernel_block_size_index += 1
    if next_kernel_block_size_index != len(kernel_block_sizes):
        raise ValueError(
            "kernel_block_sizes contains unused entries: "
            f"used={next_kernel_block_size_index} "
            f"total={len(kernel_block_sizes)}"
        )
    return kernel_block_size_by_gid


def materialize_kv_cache_tensors(
    *,
    kv_cache_config: KVCacheConfig,
    attn_groups: Sequence[Sequence[AttentionGroup]],
    kernel_block_sizes: Sequence[int],
    device: torch.device,
    cache_dtype: str | torch.dtype,
    block_major: bool = False,
) -> MaterializedKVCache:
    kernel_block_size_by_gid = build_kernel_block_size_by_group_id(
        kv_cache_config=kv_cache_config,
        kernel_block_sizes=kernel_block_sizes,
    )
    if _can_allocate_attention_cache_directly(kv_cache_config):
        placements = {
            name: (tensor.offset + index * tensor.layer_stride, tensor.block_stride)
            for tensor in kv_cache_config.kv_cache_tensors
            for index, name in enumerate(tensor.layers)
        }
        region_caches: dict[tuple[int, int], torch.Tensor] = {}
        kv_caches: dict[str, LayerKVCache] = {}
        for group_list in attn_groups:
            for group in group_list:
                gid = group.kv_cache_group_id
                spec = group.kv_cache_spec
                kernel_block_size = kernel_block_size_by_gid.get(gid)
                if kernel_block_size is None:
                    continue
                assert isinstance(spec, AttentionSpec)
                for layer_name in group.layer_names:
                    num_blocks = kv_cache_config.num_blocks
                    region = placements[layer_name]
                    assert region[1] == spec.page_size_bytes, (
                        "Direct TPU attention caches require compact pages",
                        layer_name,
                        region,
                    )
                    if region in region_caches:
                        cache = region_caches[region]
                        expected_shape = group.backend.get_kv_cache_shape(
                            num_blocks * spec.block_size // kernel_block_size,
                            kernel_block_size,
                            spec.num_kv_heads,
                            spec.head_size,
                            cache_dtype_str=cache_dtype,
                        )
                        if (
                            tuple(cache.shape) != tuple(expected_shape)
                            or cache.dtype != spec.dtype
                        ):
                            raise NotImplementedError(
                                "Aliased TPU attention layers require identical "
                                f"native cache geometry: {layer_name} expects "
                                f"{expected_shape} {spec.dtype}, existing region "
                                f"has {tuple(cache.shape)} {cache.dtype}"
                            )
                    else:
                        region_caches[region] = make_attention_cache_tensor(
                            spec=spec,
                            attn_backend=group.backend,
                            num_blocks=num_blocks,
                            kernel_block_size=kernel_block_size,
                            cache_dtype=cache_dtype,
                            device=device,
                        )
                    kv_caches[layer_name] = region_caches[region]
        return MaterializedKVCache(kv_caches=kv_caches, raw_tensors=[])

    pool_geometry = _pool_attention_geometry(
        kv_cache_config, attn_groups, kernel_block_size_by_gid
    )
    if pool_geometry is None:
        raise ValueError("unified hybrid KV layout requires an attention pool")
    layer_to_raw, raw_tensors, layout = allocate_raw_kv_cache_tensors(
        kv_cache_config,
        device,
        pool_geometry=pool_geometry,
        cache_dtype=cache_dtype,
        block_major=block_major,
    )
    kv_caches: dict[str, LayerKVCache] = {}

    # Every layer sharing a hybrid buffer holds the exact same pool object.
    # This preserves the native attention layout and gives torch.compile one
    # persistent graph input instead of per-layer typed aliases or clones.
    for group_list in attn_groups:
        for group in group_list:
            gid = group.kv_cache_group_id
            if kernel_block_size_by_gid.get(gid) is None:
                continue
            spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                raw = layer_to_raw[layer_name]
                if isinstance(spec, MambaSpec):
                    kv_caches[layer_name] = [raw]
                elif isinstance(spec, AttentionSpec):
                    kv_caches[layer_name] = raw
                else:
                    raise NotImplementedError(
                        f"Unsupported KV cache spec: {type(spec)!r}"
                    )

    return MaterializedKVCache(
        kv_caches=kv_caches,
        raw_tensors=raw_tensors,
        layer_to_raw=layer_to_raw,
        bm_pool_layout=layout,
    )
