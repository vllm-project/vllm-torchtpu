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
import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionType, MultipleOf
from vllm.v1.core.kv_cache_planning import unify_kv_cache_spec_page_size
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout as VllmKVCacheLayout

from vllm_torchtpu import envs
from vllm_torchtpu.gdn_pool_layout import pooled_gdn_state_dtypes
from vllm_torchtpu.kv_cache_materializer import (
    can_share_attention_cache,
    compresses_rows,
    stored_rows,
)
from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackend,
    PallasMLAttentionBackend,
    VllmTPUDeepseekV32IndexerBackend,
    get_tpu_min_page_size,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def normalize_kv_cache_specs_for_tpu(
    kv_cache_specs: dict[str, KVCacheSpec],
    kv_cache_dtype: torch.dtype,
    *,
    enable_unified_kv_layout: bool = False,
    exempt_layers: set[str] | None = None,
    attention_backend: type[PallasAttentionBackend] | None,
    pin_page_size: bool = True,
) -> dict[str, KVCacheSpec]:
    """Normalize KV cache specs to the TPU's real page geometry.

    `pin_page_size=False` leaves `page_size_padded` unset on specs that came
    in without one: `customize_spec` already publishes the exact TPU page, and
    vLLM's `unify_kv_cache_spec_page_size` can only rescale `block_size` on
    specs that are not pinned (see `packed_attention_block_sizes`).

    `attention_backend` must be the backend that will actually allocate: page
    geometry varies by layout, so a fixed base class would floor the spec at a
    page size nobody allocates.

    `exempt_layers` pass through untouched: their packed, compressed layout is
    not modelled by the page-size formulas here (`is_cache_for_ds_v4`).
    """
    exempt = exempt_layers or set()
    row_compressing = sorted(
        name
        for name, spec in kv_cache_specs.items()
        if name not in exempt and compresses_rows(spec)
    )
    if row_compressing and not (
        enable_unified_kv_layout
        and _has_hybrid_attention_and_mamba(
            {k: v for k, v in kv_cache_specs.items() if k not in exempt}
        )
    ):
        # Only the unified attention+mamba pool allocates the stored rows
        # (`_pool_attention_geometry`); the other layouts allocate
        # `block_size` rows per page.
        raise NotImplementedError(
            "attention specs whose page stores block_size // tokens_per_state "
            "rows are supported only in the unified attention+mamba KV pool: "
            f"{row_compressing}"
        )
    normalized: dict[str, KVCacheSpec] = {}
    for layer_name, spec in kv_cache_specs.items():
        if layer_name in exempt:
            normalized[layer_name] = spec
            continue
        normalized[layer_name] = _normalize_one_spec(
            spec,
            kv_cache_dtype,
            attention_backend=attention_backend,
            pin_page_size=pin_page_size,
        )
    if enable_unified_kv_layout:
        non_exempt_specs = {k: v for k, v in normalized.items() if k not in exempt}
        if _has_hybrid_attention_and_mamba(non_exempt_specs):
            # The pool stores the mamba state regions in its fixed physical
            # dtypes (bf16 conv -- see gdn_pool_layout), not the layer's
            # declared dense-cache dtypes. A layer that declares an fp32 conv
            # state (KDA does, for the fused kernel's sake) would otherwise
            # size its page past the attention-shaped page the pool is born
            # with; worse, a padded page below the declared-dtype page trips
            # MambaSpec.page_size_bytes' own invariant. Retype the spec to
            # the physical layout before any page math.
            non_exempt_specs = {
                name: _pooled_physical_mamba_spec(spec)
                for name, spec in non_exempt_specs.items()
            }
            normalized.update(non_exempt_specs)
            page_size = max(
                _required_page_size_bytes(spec) for spec in non_exempt_specs.values()
            )
            normalized = {
                layer_name: (
                    spec if layer_name in exempt else _pad_page_size(spec, page_size)
                )
                for layer_name, spec in normalized.items()
            }
    return normalized


def packed_attention_block_sizes(
    specs: Sequence[AttentionSpec],
    attention_backend: type[PallasAttentionBackend],
) -> dict[int, int] | None:
    """Return `{page_size_bytes: block_size}` after vLLM unifies `specs`.

    `specs` are unpinned non-MLA attention specs (one per layer geometry) at
    the kernel block size with integer-multiple page sizes. Returns None when
    vLLM groups `specs` as `UniformTypeKVCacheSpecs` (skipping `unify`), or
    when the scaled specs do not match TPU page bytes or cannot share one pool.
    """
    named: dict[str, KVCacheSpec] = {str(i): s for i, s in enumerate(specs)}
    if UniformTypeKVCacheSpecs.is_uniform_type(named):
        return None
    unified = unify_kv_cache_spec_page_size(named)
    block_sizes: dict[int, int] = {}
    for original, spec in zip(specs, unified.values()):
        assert isinstance(spec, AttentionSpec)
        if spec.page_size_bytes != attention_backend.get_kv_cache_page_size_bytes(
            spec.block_size, spec.num_kv_heads, spec.head_size, spec.dtype
        ):
            return None
        block_sizes[original.page_size_bytes] = spec.block_size
    if not can_share_attention_cache(
        list(unified.values()), attention_backend, min(block_sizes.values())
    ):
        return None
    return block_sizes


def _subpage_packing_blocker(
    vllm_config: VllmConfig,
    attention_backend: type[PallasAttentionBackend],
) -> str | None:
    """Return why sub-page packing is off for `vllm_config`, or None."""
    parallel_config = vllm_config.parallel_config
    if not attention_backend.supports_subpage_packing():
        return f"the {attention_backend.get_name()} backend on this TPU generation"
    if envs.VLLM_TPU_BLOCK_MAJOR_KV:
        return "VLLM_TPU_BLOCK_MAJOR_KV=1"
    if vllm_config.speculative_config is not None:
        return "speculative decoding"
    if vllm_config.kv_transfer_config is not None:
        return "KV transfer"
    if (
        parallel_config.prefill_context_parallel_size > 1
        or parallel_config.decode_context_parallel_size > 1
    ):
        return "PCP/DCP > 1"
    resolved_layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
    if resolved_layout != VllmKVCacheLayout.LBNHC:
        return f"the {resolved_layout.value} KV cache layout"
    return None


def _is_supported_kernel_block_size(
    block_size: int,
    min_tpu_page: int,
    supported: Sequence[int | MultipleOf],
) -> bool:
    if block_size < min_tpu_page:
        return False
    return any(
        (isinstance(entry, int) and block_size == entry)
        or (isinstance(entry, MultipleOf) and block_size % entry.base == 0)
        for entry in supported
    )


def _subpage_base_block_size(
    specs: Sequence[AttentionSpec],
    vllm_config: VllmConfig,
    attention_backend: type[PallasAttentionBackend],
    block_size: int,
) -> int:
    """Return the kernel block size the unpinned specs are built at.

    vLLM only scales smaller-page specs up. With an auto-selected `block_size`
    (e.g. 256 at `2048 < max_model_len <= 8192`) whose oversized layers are all
    `SlidingWindowSpec`, scaling the smaller group up would coarsen
    `scheduler_block_size` without relieving per-request KV reservation. Start
    from the smaller block instead, so the scaled groups land back on
    `block_size`. The smaller kernel page must be supported and fit the SMEM
    per-request page-table bound.
    """
    if vllm_config.cache_config.user_specified_block_size:
        return block_size
    min_page = min(s.page_size_bytes for s in specs)
    if not all(
        isinstance(s, SlidingWindowSpec) for s in specs if s.page_size_bytes > min_page
    ):
        return block_size
    ratio = max(s.page_size_bytes for s in specs) // min_page
    if block_size % ratio:
        return block_size
    base = block_size // ratio
    supported = attention_backend.get_supported_kernel_block_sizes()
    min_tpu_page = get_tpu_min_page_size(vllm_config)
    if not _is_supported_kernel_block_size(base, min_tpu_page, supported):
        return block_size
    return base


def _unpinned_attention_spec(
    module: Attention,
    kv_cache_dtype: torch.dtype,
    attention_backend: type[PallasAttentionBackend],
    block_size: int,
) -> AttentionSpec:
    spec: AttentionSpec
    if module.sliding_window is not None:
        spec = SlidingWindowSpec(
            block_size=block_size,
            num_kv_heads=module.num_kv_heads,
            head_size=module.head_size,
            dtype=kv_cache_dtype,
            sliding_window=module.sliding_window,
        )
    else:
        spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=module.num_kv_heads,
            head_size=module.head_size,
            dtype=kv_cache_dtype,
        )
    return attention_backend.customize_spec(spec)


def resolve_subpage_packing_block_size(
    layers: dict[str, AttentionLayerBase],
    vllm_config: VllmConfig,
    kv_cache_dtype: torch.dtype,
    attention_backend: type[PallasAttentionBackend],
    block_size: int,
) -> int | None:
    """Return the kernel block size to build unpinned attention specs at, or None.

    Applies to hybrid pure-attention models whose layers have unequal TPU page
    sizes that are integer multiples of each other. Unpinned specs keep their
    native page size so vLLM's `unify_kv_cache_spec_page_size` scales the
    smaller-page layers' `block_size`, packing sub-pages into one shared 4D pool.
    """
    # The backend's page and spec helpers read the current vLLM config.
    with set_current_vllm_config(vllm_config):
        return _resolve_subpage_packing_block_size(
            layers, vllm_config, kv_cache_dtype, attention_backend, block_size
        )


def _resolve_subpage_packing_block_size(
    layers: dict[str, AttentionLayerBase],
    vllm_config: VllmConfig,
    kv_cache_dtype: torch.dtype,
    attention_backend: type[PallasAttentionBackend],
    block_size: int,
) -> int | None:
    geometries: dict[tuple[int, int, int | None], AttentionSpec] = {}
    for module in layers.values():
        if not isinstance(module, Attention):
            return None
        if (
            module.kv_sharing_target_layer_name is not None
            or module.attn_type != AttentionType.DECODER
        ):
            continue
        key = (module.num_kv_heads, module.head_size, module.sliding_window)
        geometries[key] = _unpinned_attention_spec(
            module, kv_cache_dtype, attention_backend, block_size
        )
    if len({s.page_size_bytes for s in geometries.values()}) <= 1:
        return None

    specs = list(geometries.values())
    reason = _subpage_packing_blocker(vllm_config, attention_backend)
    if reason is None and UniformTypeKVCacheSpecs.is_uniform_type(
        {str(i): s for i, s in enumerate(specs)}
    ):
        reason = "uniform-type attention layers"
    if reason is None:
        max_page = max(s.page_size_bytes for s in specs)
        if any(max_page % s.page_size_bytes for s in specs):
            reason = "page sizes that are not integer multiples"
    if reason is None:
        base = _subpage_base_block_size(
            specs, vllm_config, attention_backend, block_size
        )
        if base != block_size:
            specs = [
                attention_backend.customize_spec(replace(s, block_size=base))
                for s in specs
            ]
        block_sizes = packed_attention_block_sizes(specs, attention_backend)
        if block_sizes is not None:
            logger.info_once(
                "Pure-attention HMA sub-page packing: page bytes -> block "
                "size %s (kernel_block_size=%d, scheduler_block_size=%d).",
                str(block_sizes),
                base,
                math.lcm(*block_sizes.values()),
            )
            return base
        reason = "pages that cannot share one pool"
    logger.info_once(
        "Pure-attention HMA sub-page packing is off because of %s; "
        "attention pages are padded to the largest page instead.",
        reason,
    )
    return None


def _as_torch_dtype(dtype: object) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    return getattr(torch, str(dtype).removeprefix("torch."))


def _pooled_physical_mamba_spec(spec: KVCacheSpec) -> KVCacheSpec:
    """Retype a two-state (conv, ssm) mamba spec to the dtypes the unified
    pool physically stores; anything else passes through."""
    if not isinstance(spec, MambaSpec):
        return spec
    if len(spec.shapes) != 2 or len(spec.dtypes) != 2:
        return spec
    conv_dtype, ssm_dtype = pooled_gdn_state_dtypes(spec.dtypes)
    dtypes = (_as_torch_dtype(conv_dtype), _as_torch_dtype(ssm_dtype))
    if tuple(spec.dtypes) == dtypes:
        return spec
    return replace(spec, dtypes=dtypes)


def _is_deepseek_v32_indexer_spec(spec: KVCacheSpec) -> bool:
    """True for packed uint8 DSA indexer K-cache (`DeepseekV32IndexerCache`)."""
    return (
        isinstance(spec, MLAAttentionSpec)
        and not spec.dtype.is_floating_point
        and spec.tokens_per_state == 1
        and spec.cache_dtype_str != "fp8_ds_mla"
        and getattr(spec, "model_version", None) != "deepseek_v4"
    )


def _normalize_one_spec(
    spec: KVCacheSpec,
    kv_cache_dtype: torch.dtype,
    *,
    attention_backend: type[PallasAttentionBackend] | None,
    pin_page_size: bool = True,
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

    if _is_deepseek_v32_indexer_spec(spec):
        backend: type[Any] = VllmTPUDeepseekV32IndexerBackend
    elif isinstance(spec, MLAAttentionSpec):
        backend = PallasMLAttentionBackend
    else:
        assert attention_backend is not None
        backend = attention_backend
    page_size = backend.get_kv_cache_page_size_bytes(
        stored_rows(spec),
        spec.num_kv_heads,
        spec.head_size,
        kv_cache_dtype,
    )
    target_spec = backend.customize_spec(replace(spec, dtype=kv_cache_dtype))
    if not pin_page_size and spec.page_size_padded is None:
        return target_spec
    padded_page_size = spec.page_size_padded or 0
    page_size = max(page_size, target_spec.real_page_size_bytes, padded_page_size)
    if target_spec.page_size_padded == page_size:
        return target_spec
    return replace(target_spec, page_size_padded=page_size)


def _has_hybrid_attention_and_mamba(
    kv_cache_specs: dict[str, KVCacheSpec],
) -> bool:
    has_attention = any(
        isinstance(spec, AttentionSpec) for spec in kv_cache_specs.values()
    )
    has_mamba = any(isinstance(spec, MambaSpec) for spec in kv_cache_specs.values())
    return has_attention and has_mamba


def _mamba_unpadded_page_size_bytes(spec: MambaSpec) -> int:
    return sum(
        math.prod(shape) * dtype.itemsize
        for shape, dtype in zip(spec.shapes, spec.dtypes)
    )


def _required_page_size_bytes(spec: KVCacheSpec) -> int:
    if isinstance(spec, MambaSpec):
        return max(_mamba_unpadded_page_size_bytes(spec), spec.page_size_padded or 0)
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
