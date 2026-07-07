from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
from typing import Any, TypeAlias

import torch
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (AttentionSpec,
                                        EncoderOnlyAttentionSpec,
                                        KVCacheConfig, KVCacheSpec, MambaSpec,
                                        UniformTypeKVCacheSpecs)
from vllm.v1.worker.utils import AttentionGroup

from vllm_torchtpu import envs as tpu_envs

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


def allocate_raw_kv_cache_tensors(
    kv_cache_config: KVCacheConfig,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
    layer_to_raw: dict[str, torch.Tensor] = {}
    raw_tensors: list[torch.Tensor] = []
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        raw = torch.zeros(kv_cache_tensor.size,
                          dtype=torch.int8,
                          device=device)
        raw_tensors.append(raw)
        for layer_name in kv_cache_tensor.shared_by:
            layer_to_raw[layer_name] = raw
    return layer_to_raw, raw_tensors


def _kv_cache_alias_fallback_enabled() -> bool:
    return tpu_envs.TPU_VLLM_KV_CACHE_ALIAS_FALLBACK


def _typed_kv_cache_view(
    raw_int8: torch.Tensor,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    stride: tuple[int, ...] | None = None,
    storage_offset_bytes: int = 0,
) -> torch.Tensor:
    if raw_int8.dtype != torch.int8:
        raise TypeError(f"typed KV cache view expects int8 raw storage, got "
                        f"{raw_int8.dtype}")
    if _kv_cache_alias_fallback_enabled():
        if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # torch_tpu cannot materialize fp8 zero_ on TPU. Cache slots that
            # use fp8 are overwritten before use, so allocate them uninitialized
            # while preserving zero initialization for supported dtypes.
            return torch.empty(shape, dtype=dtype, device=raw_int8.device)
        return torch.zeros(shape, dtype=dtype, device=raw_int8.device)

    dtype_size = get_dtype_size(dtype)
    base_offset_bytes = raw_int8.storage_offset() * raw_int8.element_size()
    absolute_offset_bytes = base_offset_bytes + storage_offset_bytes
    if absolute_offset_bytes % dtype_size != 0:
        raise ValueError(
            f"storage offset {absolute_offset_bytes} bytes is not aligned "
            f"to dtype {dtype} size {dtype_size}")

    if stride is None:
        stride = torch.empty(shape, dtype=dtype).stride()
    num_elements = _num_elements_reachable(shape, stride)
    if storage_offset_bytes + num_elements * dtype_size > raw_int8.numel():
        raise ValueError(
            f"typed KV cache view shape={shape} stride={stride} dtype={dtype} "
            f"offset_bytes={storage_offset_bytes} exceeds raw int8 buffer "
            f"with {raw_int8.numel()} bytes")

    base = raw_int8.view(dtype)
    storage_offset = absolute_offset_bytes // dtype_size
    return base.as_strided(shape, stride, storage_offset)


def _num_elements_reachable(
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> int:
    if len(shape) == 0:
        return 1
    if 0 in shape:
        return 0
    if stride == torch.empty(shape).stride():
        return prod(shape)
    return 1 + sum((size - 1) * step for size, step in zip(shape, stride))


def make_attention_cache_view(
    *,
    raw: torch.Tensor,
    spec: AttentionSpec,
    attn_backend: Any,
    num_blocks: int,
    kernel_block_size: int,
    cache_dtype: str | torch.dtype,
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
    return _typed_kv_cache_view(raw, spec.dtype, tuple(shape))


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


def make_mamba_cache_views(
    *,
    raw: torch.Tensor,
    spec: MambaSpec,
    num_blocks: int,
) -> list[torch.Tensor]:
    states: list[torch.Tensor] = []
    storage_offset_bytes = 0
    for shape, dtype in zip(spec.shapes, spec.dtypes):
        dtype_size = get_dtype_size(dtype)
        num_element_per_page = spec.page_size_bytes // dtype_size
        target_shape = (num_blocks, *shape)
        dense_stride = torch.empty(target_shape, dtype=dtype).stride()
        target_stride = (num_element_per_page, *dense_stride[1:])
        state = _typed_kv_cache_view(
            raw,
            dtype,
            target_shape,
            stride=target_stride,
            storage_offset_bytes=storage_offset_bytes,
        )
        states.append(state)
        storage_offset_bytes += dense_stride[0] * dtype_size
    return states


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
    mamba_num_blocks: int | None = None,
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

    layer_to_raw, raw_tensors = allocate_raw_kv_cache_tensors(
        kv_cache_config, device)
    kv_caches: dict[str, LayerKVCache] = {}

    for group_list in attn_groups:
        for group in group_list:
            gid = group.kv_cache_group_id
            kernel_block_size = kernel_block_size_by_gid.get(gid)
            if kernel_block_size is None:
                continue
            spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                raw = layer_to_raw[layer_name]
                assert raw.numel() % spec.page_size_bytes == 0
                num_blocks = raw.numel() // spec.page_size_bytes
                if isinstance(spec, AttentionSpec):
                    kv_caches[layer_name] = make_attention_cache_view(
                        raw=raw,
                        spec=spec,
                        attn_backend=group.backend,
                        num_blocks=num_blocks,
                        kernel_block_size=kernel_block_size,
                        cache_dtype=cache_dtype,
                    )
                elif isinstance(spec, MambaSpec):
                    kv_caches[layer_name] = make_mamba_cache_views(
                        raw=raw,
                        spec=spec,
                        num_blocks=mamba_num_blocks
                        if mamba_num_blocks is not None else num_blocks,
                    )
                else:
                    raise NotImplementedError(
                        f"Unsupported KV cache spec: {type(spec)!r}")

    return MaterializedKVCache(kv_caches=kv_caches, raw_tensors=raw_tensors)
