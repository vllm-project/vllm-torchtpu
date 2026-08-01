from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch
from vllm.v1.kv_cache_interface import (AttentionSpec,
                                        EncoderOnlyAttentionSpec,
                                        KVCacheConfig, KVCacheSpec, MambaSpec,
                                        UniformTypeKVCacheSpecs)
from vllm.v1.worker.utils import AttentionGroup

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

LayerKVCache: TypeAlias = torch.Tensor | list[torch.Tensor]


@dataclass(frozen=True)
class MaterializedKVCache:
    kv_caches: dict[str, LayerKVCache]
    raw_tensors: list[torch.Tensor]


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
    return (f"shape={tuple(tensor.shape)} stride={tuple(tensor.stride())} "
            f"dtype={tensor.dtype} contiguous={tensor.is_contiguous()} "
            f"storage_offset={tensor.storage_offset()} "
            f"storage_offset_bytes={storage_offset_bytes} "
            f"numel={tensor.numel()} nbytes={tensor.nbytes} raw_idx={raw_idx}")


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
                group_backend_names[group.kv_cache_group_id] = (
                    group.backend.__name__)

    lines = [
        "TPU KV cache layout summary:",
        f"  num_blocks={kv_cache_config.num_blocks} "
        f"num_groups={len(kv_cache_config.kv_cache_groups)} "
        f"num_raw_tensors={len(raw_tensors)}",
    ]
    for idx, raw in enumerate(raw_tensors):
        lines.append(
            f"  raw[{idx}]: shape={tuple(raw.shape)} dtype={raw.dtype} "
            f"numel={raw.numel()} nbytes={raw.nbytes}")

    for gid, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        spec_name = type(spec).__name__
        backend_name = group_backend_names.get(gid, "<none>")
        lines.append(f"  group[{gid}]: spec={spec_name} "
                     f"backend={backend_name} "
                     f"block_size={spec.block_size} "
                     f"page_size_bytes={spec.page_size_bytes} "
                     f"layers={list(group.layer_names)}")
        for layer_name in group.layer_names:
            cache = kv_caches.get(layer_name)
            if cache is None:
                lines.append(f"    {layer_name}: <missing>")
                continue
            if isinstance(cache, torch.Tensor):
                lines.append(f"    {layer_name}: attention "
                             f"{_tensor_layout_summary(cache, raw_tensors)}")
            else:
                lines.append(f"    {layer_name}: mamba states={len(cache)}")
                for state_idx, state in enumerate(cache):
                    lines.append(
                        f"      state[{state_idx}]: "
                        f"{_tensor_layout_summary(state, raw_tensors)}")
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
        for group in kv_cache_config.kv_cache_groups)
    if not has_mamba:
        return None
    for group_list in attn_groups:
        for group in group_list:
            spec = group.kv_cache_spec
            if isinstance(spec, AttentionSpec) and not isinstance(
                    spec, EncoderOnlyAttentionSpec):
                kernel_block_size = kernel_block_size_by_gid.get(
                    group.kv_cache_group_id, spec.block_size)
                return spec, group.backend, kernel_block_size
    return None


def allocate_raw_kv_cache_tensors(
    kv_cache_config: KVCacheConfig,
    device: torch.device,
    pool_geometry: tuple[AttentionSpec, Any, int],
    cache_dtype: str | torch.dtype = "auto",
) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
    layer_to_raw: dict[str, torch.Tensor] = {}
    raw_tensors: list[torch.Tensor] = []
    pool_page_bytes = max(group.kv_cache_spec.page_size_bytes
                          for group in kv_cache_config.kv_cache_groups)
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        # The unified pool is born as the stock attention-shaped KV cache.
        # Attention consumes it natively; Mamba adapters address their byte
        # regions inside each manager page without creating typed aliases.
        spec, attn_backend, kernel_block_size = pool_geometry
        assert spec.block_size % kernel_block_size == 0, (spec.block_size,
                                                          kernel_block_size)
        split = spec.block_size // kernel_block_size
        assert kv_cache_tensor.size % pool_page_bytes == 0, (
            kv_cache_tensor.size, pool_page_bytes)
        num_blocks = kv_cache_tensor.size // pool_page_bytes
        shape = attn_backend.get_kv_cache_shape(
            num_blocks * split,
            kernel_block_size,
            spec.num_kv_heads,
            spec.head_size,
            cache_dtype_str=cache_dtype,
        )
        if spec.dtype in _FP8_DTYPES:
            raw = torch.empty(tuple(shape), dtype=spec.dtype, device=device)
        else:
            raw = torch.zeros(tuple(shape), dtype=spec.dtype, device=device)
        fa_page_bytes = (raw.numel() * raw.element_size() //
                         (num_blocks * split))
        assert fa_page_bytes * split == pool_page_bytes, (fa_page_bytes, split,
                                                          pool_page_bytes)
        raw_tensors.append(raw)
        for layer_name in kv_cache_tensor.shared_by:
            layer_to_raw[layer_name] = raw
    return layer_to_raw, raw_tensors


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


def _can_allocate_attention_cache_directly(
    kv_cache_config: KVCacheConfig, ) -> bool:
    return all(
        isinstance(group.kv_cache_spec, AttentionSpec)
        for group in kv_cache_config.kv_cache_groups) and all(
            len(kv_cache_tensor.shared_by) == 1
            for kv_cache_tensor in kv_cache_config.kv_cache_tensors)


def _select_representative_kv_cache_spec(
        kv_cache_spec: KVCacheSpec) -> KVCacheSpec:
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        try:
            return next(iter(kv_cache_spec.kv_cache_specs.values()))
        except StopIteration as exc:
            raise ValueError("UniformTypeKVCacheSpecs must contain at least "
                             "one KV cache spec") from exc
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
                f"Unsupported KV cache spec: {type(group.kv_cache_spec)!r}")
        if next_kernel_block_size_index >= len(kernel_block_sizes):
            raise ValueError("kernel_block_sizes is missing an entry for "
                             f"KV cache group {gid}")
        kernel_block_size = int(
            kernel_block_sizes[next_kernel_block_size_index])
        if kernel_block_size <= 0:
            raise ValueError("kernel_block_sizes must contain positive "
                             f"values, got {kernel_block_size}")
        kernel_block_size_by_gid[gid] = kernel_block_size
        next_kernel_block_size_index += 1
    if next_kernel_block_size_index != len(kernel_block_sizes):
        raise ValueError("kernel_block_sizes contains unused entries: "
                         f"used={next_kernel_block_size_index} "
                         f"total={len(kernel_block_sizes)}")
    return kernel_block_size_by_gid


def materialize_kv_cache_tensors(
    *,
    kv_cache_config: KVCacheConfig,
    attn_groups: Sequence[Sequence[AttentionGroup]],
    kernel_block_sizes: Sequence[int],
    device: torch.device,
    cache_dtype: str | torch.dtype,
) -> MaterializedKVCache:
    kernel_block_size_by_gid = build_kernel_block_size_by_group_id(
        kv_cache_config=kv_cache_config,
        kernel_block_sizes=kernel_block_sizes,
    )
    if _can_allocate_attention_cache_directly(kv_cache_config):
        layer_to_tensor_size = {
            kv_cache_tensor.shared_by[0]: kv_cache_tensor.size
            for kv_cache_tensor in kv_cache_config.kv_cache_tensors
        }
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
                    tensor_size = layer_to_tensor_size[layer_name]
                    assert tensor_size % spec.page_size_bytes == 0
                    num_blocks = tensor_size // spec.page_size_bytes
                    kv_caches[layer_name] = make_attention_cache_tensor(
                        spec=spec,
                        attn_backend=group.backend,
                        num_blocks=num_blocks,
                        kernel_block_size=kernel_block_size,
                        cache_dtype=cache_dtype,
                        device=device,
                    )
        return MaterializedKVCache(kv_caches=kv_caches, raw_tensors=[])

    pool_geometry = _pool_attention_geometry(kv_cache_config, attn_groups,
                                             kernel_block_size_by_gid)
    if pool_geometry is None:
        raise ValueError("unified hybrid KV layout requires an attention pool")
    layer_to_raw, raw_tensors = allocate_raw_kv_cache_tensors(
        kv_cache_config,
        device,
        pool_geometry=pool_geometry,
        cache_dtype=cache_dtype)
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
                        f"Unsupported KV cache spec: {type(spec)!r}")

    return MaterializedKVCache(kv_caches=kv_caches, raw_tensors=raw_tensors)
