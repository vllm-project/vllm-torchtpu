import math
from dataclasses import replace

import torch
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import (AttentionSpec, KVCacheSpec, MambaSpec,
                                        MLAAttentionSpec)

from vllm_torchtpu.layers.vllm.attention import (PallasAttentionBackend,
                                                 PallasMLAttentionBackend)


def normalize_kv_cache_specs_for_tpu(
    kv_cache_specs: dict[str, KVCacheSpec],
    kv_cache_dtype: str | torch.dtype,
    *,
    enable_unified_kv_layout: bool = False,
    exempt_layers: set[str] | None = None,
    attention_backend: type[AttentionBackend] | None = None,
) -> dict[str, KVCacheSpec]:
    """Normalize KV cache specs to the TPU's real page geometry.

    `exempt_layers` pass through untouched: their packed, compressed layout is
    not modelled by the page-size formulas here (`is_cache_for_ds_v4`).
    """
    exempt = exempt_layers or set()
    normalized: dict[str, KVCacheSpec] = {}
    for layer_name, spec in kv_cache_specs.items():
        if layer_name in exempt:
            normalized[layer_name] = spec
            continue
        normalized[layer_name] = _normalize_one_spec(
            spec,
            kv_cache_dtype,
            attention_backend=attention_backend,
        )
    if enable_unified_kv_layout:
        non_exempt_specs = {
            k: v
            for k, v in normalized.items() if k not in exempt
        }
        if _has_hybrid_attention_and_mamba(non_exempt_specs):
            page_size = max(
                _required_page_size_bytes(spec)
                for spec in non_exempt_specs.values())
            normalized = {
                layer_name: (spec if layer_name in exempt else _pad_page_size(
                    spec, page_size))
                for layer_name, spec in normalized.items()
            }
    return normalized


def _normalize_one_spec(
    spec: KVCacheSpec,
    kv_cache_dtype: str | torch.dtype,
    *,
    attention_backend: type[AttentionBackend] | None = None,
) -> KVCacheSpec:
    if not isinstance(spec, AttentionSpec):
        return spec

    if isinstance(spec, MLAAttentionSpec) and not spec.dtype.is_floating_point:
        # An integer-typed MLA cache is a packed layout: quantized float KV
        # values and their scale factors share one byte row (DSA indexer K
        # rows). That layout is fixed by the kernel reading it, so the fp8/bf16
        # KV normalization must not retype it -- keep the declared dtype.
        # Regular attention specs marked uint8 (vLLM's fp8-KV convention) still
        # normalize to the TPU fp8 dtype below.
        kv_cache_dtype = spec.dtype

    if isinstance(spec, MLAAttentionSpec):
        page_size = PallasMLAttentionBackend.get_kv_cache_page_size_bytes(
            spec.block_size,
            spec.num_kv_heads,
            spec.head_size,
            kv_cache_dtype,
        )
    else:
        backend = attention_backend or PallasAttentionBackend
        page_size = backend.get_kv_cache_page_size_bytes(
            spec.block_size,
            spec.num_kv_heads,
            spec.head_size,
            kv_cache_dtype,
        )
    target_spec = replace(spec, dtype=kv_cache_dtype)
    padded_page_size = spec.page_size_padded or 0
    page_size = max(page_size, target_spec.real_page_size_bytes,
                    padded_page_size)
    if spec.page_size_padded == page_size and spec.dtype == kv_cache_dtype:
        return spec
    return replace(target_spec, page_size_padded=page_size)


def _has_hybrid_attention_and_mamba(
    kv_cache_specs: dict[str, KVCacheSpec], ) -> bool:
    has_attention = any(
        isinstance(spec, AttentionSpec) for spec in kv_cache_specs.values())
    has_mamba = any(
        isinstance(spec, MambaSpec) for spec in kv_cache_specs.values())
    return has_attention and has_mamba


def _mamba_unpadded_page_size_bytes(spec: MambaSpec) -> int:
    return sum(
        math.prod(shape) * dtype.itemsize
        for shape, dtype in zip(spec.shapes, spec.dtypes))


def _required_page_size_bytes(spec: KVCacheSpec) -> int:
    if isinstance(spec, MambaSpec):
        return max(_mamba_unpadded_page_size_bytes(spec), spec.page_size_padded
                   or 0)
    return spec.page_size_bytes


def _pad_page_size(spec: KVCacheSpec, page_size: int) -> KVCacheSpec:
    if isinstance(spec, AttentionSpec):
        if spec.page_size_bytes == page_size:
            return spec
        return replace(spec, page_size_padded=page_size)
    if isinstance(spec, MambaSpec):
        target = max(page_size, _mamba_unpadded_page_size_bytes(spec))
        if spec.page_size_padded == target:
            return spec
        return replace(spec, page_size_padded=target)
    return spec
