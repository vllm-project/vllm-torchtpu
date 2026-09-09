# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-cache spec derivation, page-size padding, budgeting and allocation.

Moved verbatim out of `TPUModelRunner`; see vllm-project/vllm-torchtpu#713.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import torch
from vllm.config import get_layers_from_vllm_config, set_current_vllm_config
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.model_executor.layers.attention import (
    Attention,
    ChunkedLocalAttention,
    MLAAttention,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.models.deepseek_v4.attention import (
    DeepseekV4Attention,
    DeepseekV4IndexerCache,
)
from vllm.models.deepseek_v4.compressor import CompressorStateCache
from vllm.v1.attention.backend import AttentionBackend, AttentionType
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.utils import (
    AttentionGroup,
    add_kv_sharing_layers_to_kv_cache_groups,
    prepare_kernel_block_sizes,
)

from vllm_torchtpu import envs, utils
from vllm_torchtpu.kv_cache_materializer import (
    build_kernel_block_size_by_group_id,
    can_share_attention_cache,
    format_kv_cache_layout_summary,
    materialize_kv_cache_tensors,
    materialize_shared_attention_cache,
)
from vllm_torchtpu.kv_cache_spec_normalizer import normalize_kv_cache_specs_for_tpu
from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackend,
    PallasMLAttentionBackend,
)
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadataBuilder
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform
from vllm_torchtpu.runner.kv_cache_dsv4 import DsV4KVCacheAllocator
from vllm_torchtpu.utils import synchronize_tensors

if TYPE_CHECKING:
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)


def is_cache_for_ds_v4(attn_module: AttentionLayerBase) -> bool:
    """Whether this module owns one of DeepSeek-V4's custom KV caches.

    These build their own specs in the TPU's packed layout, so the specs must
    reach vLLM verbatim rather than through `_normalize_one_spec`.
    """
    return isinstance(
        attn_module,
        (
            DeepseekV4Attention,
            DeepseekV4SWACache,
            DeepseekV4IndexerCache,
            CompressorStateCache,
        ),
    )


def check_kv_caches_cover_block_ids(
    kv_caches: dict[str, torch.Tensor],
    kv_cache_tensors: Iterable[KVCacheTensor],
    num_blocks: int,
    spec_for_layer: Callable[[str], KVCacheSpec],
) -> None:
    """Check block coverage for dense caches in scheduler-page layout."""
    for kv_cache_tensor in kv_cache_tensors:
        for layer_name in kv_cache_tensor.layers:
            spec = spec_for_layer(layer_name)
            if not isinstance(spec, AttentionSpec) or isinstance(
                spec, MLAAttentionSpec
            ):
                continue
            kv_cache = kv_caches.get(layer_name)
            if kv_cache is None:
                raise ValueError(f"Missing KV cache for attention layer {layer_name}")
            if kv_cache.shape[0] < num_blocks:
                raise ValueError(
                    f"KV cache for {layer_name} holds {kv_cache.shape[0]} "
                    f"blocks, but the scheduler can issue block IDs through "
                    f"{num_blocks - 1} ({num_blocks} blocks; "
                    f"{len(kv_cache_tensor.layers)} layer(s) in this "
                    f"{kv_cache_tensor.size}-byte backing allocation). "
                    "TPU kernels do not bounds-check these "
                    "indices."
                )


def _warn_if_kv_cache_is_padded(
    backend: type[AttentionBackend],
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
) -> None:
    """Warn when a page costs more than its shape and dtype imply.

    The baseline is one key and one value vector per head per token. Layouts
    exceed it by rounding the packing dimension up to a 32-bit word and
    head_size up to a 128-lane register; either wastes HBM.

    Read through `get_kv_cache_shape` so any backend works, but MLA has no
    per-head key and value and so is not described by this baseline.
    """
    page_shape = backend.get_kv_cache_shape(
        1, block_size, num_kv_heads, head_size, dtype
    )
    actual = math.prod(page_shape) * dtype.itemsize
    unpadded = block_size * 2 * num_kv_heads * head_size * dtype.itemsize
    if actual == unpadded:
        return
    logger.warning_once(
        "KV cache pages are %.2fx larger than num_kv_heads=%d at "
        "head_size=%d with dtype %s requires: the %s layout pads head_size to "
        "a whole 128-lane register and the K/V vectors to a whole 32-bit word, "
        "so HBM use exceeds a shape-and-dtype estimate. Where head_size is a "
        "multiple of 128, lowering tensor parallelism (more KV heads per rank) "
        "or enabling attention data parallelism avoids it.",
        actual / unpadded,
        num_kv_heads,
        head_size,
        dtype,
        backend.get_name(),
    )


class KVCacheManager:
    """Owns the KV-cache spec/sizing half of `TPUModelRunner`."""

    def __init__(self, runner: TPUModelRunner) -> None:
        self.runner = runner
        self.ds_v4 = DsV4KVCacheAllocator(runner)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """
        layers = get_layers_from_vllm_config(
            self.runner.vllm_config,
            (AttentionLayerBase, MambaBase),  # type: ignore[type-abstract]
        )
        backend_cls = TpuPlatform._find_non_ssm_backend(self.runner.vllm_config)
        block_size = self.runner.vllm_config.cache_config.block_size
        cache_dtype_str = self.runner.vllm_config.cache_config.cache_dtype

        has_attention = any(
            isinstance(m, (Attention, MLAAttention)) for m in layers.values()
        )
        has_mamba = any(isinstance(m, MambaBase) for m in layers.values())
        if has_attention and not self.runner._unified_kv_layout:
            self._update_attention_page_size_padded(layers, block_size)
            if has_mamba:
                self._update_mamba_page_size_padded(layers)

        hma_enabled = not self.runner.scheduler_config.disable_hybrid_kv_cache_manager

        kv_cache_spec: dict[str, KVCacheSpec] = {}
        # DSv4 layers whose specs must reach vLLM verbatim; see
        # `is_cache_for_ds_v4`.
        ds_v4_layers: set[str] = set()
        for layer_name, attn_module in layers.items():
            # Linear Attention path
            if isinstance(attn_module, MambaBase):
                spec = attn_module.get_kv_cache_spec(self.runner.vllm_config)
                if spec is not None:
                    kv_cache_spec[layer_name] = spec
            # DSv4's attention, SWA, compressor and indexer caches all build
            # their own specs, as the reference does.
            elif is_cache_for_ds_v4(attn_module):
                ds_v4_layers.add(layer_name)
                spec = attn_module.get_kv_cache_spec(self.runner.vllm_config)
                if spec is not None:
                    kv_cache_spec[layer_name] = spec
            # Classic Attention path
            elif isinstance(attn_module, Attention):
                if (
                    kv_tgt_layer := attn_module.kv_sharing_target_layer_name
                ) is not None:
                    if kv_tgt_layer not in layers:
                        raise ValueError(
                            f"Layer {layer_name} reuses KV cache from missing "
                            f"target layer {kv_tgt_layer}."
                        )
                    target_module = layers[kv_tgt_layer]
                    if not isinstance(target_module, Attention):
                        raise ValueError(
                            f"Layer {layer_name} reuses KV cache from "
                            f"non-attention target layer {kv_tgt_layer}."
                        )
                    if target_module.kv_sharing_target_layer_name is not None:
                        raise ValueError(
                            f"Layer {layer_name} reuses KV cache from "
                            f"{kv_tgt_layer}, which is itself a shared KV "
                            "layer."
                        )
                    self._validate_shared_kv_cache_layout(
                        layer_name,
                        attn_module,
                        kv_tgt_layer,
                        target_module,
                        hma_enabled,
                    )
                    # The layer doesn't need its own KV cache and will use that of
                    # the target layer. We skip creating a KVCacheSpec for it, so
                    # that KV cache management logic will act as this layer does
                    # not exist, and doesn't allocate KV cache for the layer. This
                    # enables the memory saving of cross-layer kv sharing, allowing
                    # a given amount of memory to accommodate longer context lengths
                    # or enable more requests to be processed simultaneously.
                    self.runner.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                    continue

                if attn_module.attn_type == AttentionType.DECODER:
                    if isinstance(attn_module, ChunkedLocalAttention):
                        logger.warning_once(
                            "Using irope in Pallas is not supported yet, it "
                            "will fall back to global attention for long context."
                        )
                    # The layout can pad a page past what its shape and dtype
                    # imply, with no other signal that it did. Take the backend
                    # off this layer so it always matches the dimensions below.
                    _warn_if_kv_cache_is_padded(
                        backend_cls,
                        block_size,
                        attn_module.num_kv_heads,
                        attn_module.head_size,
                        self.runner.kv_cache_dtype,
                    )
                    page_size_padded = (
                        self.runner._hybrid_uniform_page_size_bytes
                        if self.runner._hybrid_uniform_page_size_bytes is not None
                        else backend_cls.get_kv_cache_page_size_bytes(
                            block_size,
                            attn_module.num_kv_heads,
                            attn_module.head_size,
                            self.runner.kv_cache_dtype,
                        )
                    )
                    if attn_module.sliding_window is not None:
                        kv_cache_spec[layer_name] = SlidingWindowSpec(
                            block_size=block_size,
                            num_kv_heads=attn_module.num_kv_heads,
                            head_size=attn_module.head_size,
                            dtype=self.runner.kv_cache_dtype,
                            page_size_padded=page_size_padded,
                            sliding_window=attn_module.sliding_window,
                        )
                    else:
                        kv_cache_spec[layer_name] = FullAttentionSpec(
                            block_size=block_size,
                            num_kv_heads=attn_module.num_kv_heads,
                            head_size=attn_module.head_size,
                            dtype=self.runner.kv_cache_dtype,
                            page_size_padded=page_size_padded,
                        )
                elif attn_module.attn_type in (
                    AttentionType.ENCODER,
                    AttentionType.ENCODER_ONLY,
                ):
                    # encoder-only attention does not need KV cache.
                    continue
                elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                    raise NotImplementedError
                else:
                    raise ValueError(f"Unknown attention type: {attn_module.attn_type}")
            # MLAAttention path
            elif isinstance(attn_module, MLAAttention):
                if layer_name in kv_cache_spec:
                    continue
                page_size_padded = (
                    self.runner._hybrid_uniform_page_size_bytes
                    if self.runner._hybrid_uniform_page_size_bytes is not None
                    else PallasMLAttentionBackend.get_kv_cache_page_size_bytes(
                        block_size,
                        1,
                        attn_module.head_size,
                        self.runner.kv_cache_dtype,
                    )
                )
                kv_cache_spec[layer_name] = MLAAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=attn_module.head_size,
                    dtype=self.runner.kv_cache_dtype,
                    cache_dtype_str=cache_dtype_str,
                    page_size_padded=page_size_padded,
                )
            elif isinstance(attn_module, DeepseekV32IndexerCache):
                # DSA indexer K cache: the module declares its own uint8 spec
                # (head_dim fp8 bytes + 1 e8m0 scale byte per token).
                kv_cache_spec[layer_name] = attn_module.get_kv_cache_spec(
                    self.runner.vllm_config
                )
            else:
                continue

        # Note: that each shared layer's target actually owns a KV cache (and
        # that the shared layer was not itself allocated one) is enforced later
        # against the concrete KVCacheConfig in
        # `_maybe_add_kv_sharing_layers_to_kv_cache_groups`.

        # Cross-model KV sharing (Gemma-4-style MTP) rides the same path as
        # within-model sharing. The drafter is loaded before `get_kv_cache_spec`
        # runs, so its `Attention` modules are already in the static forward
        # context with `kv_sharing_target_layer_name` set and the loop above has
        # registered them. Verify that rather than re-registering: a draft layer
        # reaching this point unregistered would be allocated its own KV cache
        # and would silently write there instead of into the target's.
        drafter = getattr(self.runner, "drafter", None)
        if (
            self.runner._is_async_drafter
            and getattr(drafter, "draft_model", None) is not None
        ):
            for draft_idx, layer in enumerate(drafter.draft_model.model.layers):
                attn = getattr(getattr(layer, "self_attn", None), "attn", None)
                if attn is None or attn.kv_sharing_target_layer_name is None:
                    continue
                name = f"draft_model.layers.{draft_idx}.self_attn.attn"
                target = attn.kv_sharing_target_layer_name
                if self.runner.shared_kv_cache_layers.get(name) != target:
                    raise RuntimeError(
                        f"Draft layer {name} declares KV sharing with {target} "
                        "but was not registered as a shared layer. It would be "
                        "given its own KV cache and write there instead of the "
                        "target's."
                    )
                if name in kv_cache_spec:
                    raise RuntimeError(
                        f"Draft layer {name} shares KV with {target} but still "
                        "has its own KVCacheSpec."
                    )

        kv_cache_spec = normalize_kv_cache_specs_for_tpu(
            kv_cache_spec,
            self.runner.kv_cache_dtype,
            enable_unified_kv_layout=self.runner._unified_kv_layout,
            exempt_layers=ds_v4_layers,
            attention_backend=backend_cls,
        )
        if (
            hma_enabled
            and has_attention
            and not has_mamba
            and not ds_v4_layers
            and not self.runner._unified_kv_layout
        ):
            kv_cache_spec = self._pad_native_attention_regions(
                kv_cache_spec, backend_cls
            )
        return kv_cache_spec

    def _pad_native_attention_regions(
        self,
        kv_cache_specs: dict[str, KVCacheSpec],
        backend_cls: type[AttentionBackend],
    ) -> dict[str, KVCacheSpec]:
        if can_share_attention_cache(list(kv_cache_specs.values()), backend_cls):
            return kv_cache_specs
        # Layouts that cannot share a packed pool need separate native arrays;
        # identical geometries can still share an array across
        # groups because each block ID belongs to only one group at a time.
        groups = get_kv_cache_groups(self.runner.vllm_config, kv_cache_specs.copy())
        if len(groups) <= 1:
            return kv_cache_specs
        native_pages = {}
        for group in groups:
            spec = group.kv_cache_spec
            assert isinstance(spec, AttentionSpec), (
                "Pure-attention HMA groups must have one attention spec",
                spec,
            )
            backend = (
                PallasMLAttentionBackend
                if isinstance(spec, MLAAttentionSpec)
                else backend_cls
            )
            shape = tuple(
                backend.get_kv_cache_shape(
                    1, spec.block_size, spec.num_kv_heads, spec.head_size, spec.dtype
                )
            )
            native_pages[shape, spec.dtype] = math.prod(shape) * spec.dtype.itemsize
        if len(native_pages) <= 1:
            return kv_cache_specs
        # Each layer position can contain at most one array of each native
        # geometry. Repeated groups with the same geometry add no HBM cost.
        page_bytes = max(
            sum(native_pages.values()),
            max(spec.page_size_bytes for spec in kv_cache_specs.values()),
        )
        logger.info(
            "Pure-attention HMA: budgeting %d native cache geometries per "
            "layer region, %d bytes per block",
            len(native_pages),
            page_bytes,
        )
        return {
            name: dataclasses.replace(spec, page_size_padded=page_bytes)
            for name, spec in kv_cache_specs.items()
        }

    @staticmethod
    def _validate_shared_kv_cache_layout(
        layer_name: str,
        attn_module: Attention,
        target_layer_name: str,
        target_module: Attention,
        hma_enabled: bool,
    ) -> None:
        # When the hybrid KV cache manager is disabled, all specs are unified to
        # full attention: we store the full KV cache and apply each layer's own
        # window mask on read, so sliding_window doesn't affect storage and need
        # not match between the shared layer and its target. When the hybrid
        # manager is enabled (e.g. a mamba + sliding window + full attn model),
        # sliding-window layers get a smaller, window-sized cache, so a window
        # mismatch would mean the layers genuinely disagree on storage and the
        # share is unsafe -- enforce sliding_window in that case.
        fields = ["attn_type", "num_kv_heads", "head_size"]
        if hma_enabled:
            fields.append("sliding_window")
        mismatches = []
        for field_name in fields:
            if getattr(attn_module, field_name, None) != getattr(
                target_module, field_name, None
            ):
                mismatches.append(field_name)
        if mismatches:
            raise ValueError(
                f"Layer {layer_name} cannot reuse KV cache from "
                f"{target_layer_name}: incompatible "
                f"{', '.join(mismatches)}."
            )

        # With fp8 KV cache the shared layer reads the target's cached K/V and
        # dequantizes with its OWN scales, so mismatched scales/dtype would
        # silently corrupt attention.
        attn_quant = attn_module.impl.kv_cache_quantized_dtype
        target_quant = target_module.impl.kv_cache_quantized_dtype
        # Trigger the check if EITHER side is fp8: the asymmetric case (only one
        # side quantized) is just as unsafe -- e.g. the shared layer reads the
        # target's packed fp8 bytes as bf16 -- and must error rather than
        # silently corrupt, so None != "fp8_e4m3" correctly fails below.
        if attn_quant or target_quant:
            attn_kv_layout = (
                attn_module._k_scale_float,
                attn_module._v_scale_float,
                attn_quant,
            )
            target_kv_layout = (
                target_module._k_scale_float,
                target_module._v_scale_float,
                target_quant,
            )
            if attn_kv_layout != target_kv_layout:
                raise ValueError(
                    f"Layer {layer_name} reuses the KV cache of "
                    f"{target_layer_name} but their kv cache dtype or k/v "
                    f"scales differ (shared={attn_kv_layout}, "
                    f"target={target_kv_layout}); the shared layer would "
                    "read/dequantize the target's cached K/V incorrectly. "
                    "Cross-layer KV sharing requires a matching kv cache dtype "
                    "and matching k/v scales."
                )

    def _maybe_add_kv_sharing_layers_to_kv_cache_groups(
        self, kv_cache_config: KVCacheConfig
    ) -> None:
        if not self.runner.shared_kv_cache_layers:
            return
        group_layer_names = {
            layer_name
            for group in kv_cache_config.kv_cache_groups
            for layer_name in group.layer_names
        }
        allocated_layer_names = {
            layer_name
            for kv_cache_tensor in kv_cache_config.kv_cache_tensors
            for layer_name in kv_cache_tensor.layers
        }
        for layer_name, target_layer_name in self.runner.shared_kv_cache_layers.items():
            if target_layer_name not in group_layer_names:
                raise ValueError(
                    f"Layer {layer_name} reuses KV cache from "
                    f"{target_layer_name}, but the target layer is missing "
                    "from KV cache groups."
                )
            if layer_name in group_layer_names or layer_name in allocated_layer_names:
                raise ValueError(
                    f"Shared KV layer {layer_name} must not have an "
                    "independent KV cache allocation."
                )
        runner_only_attn_layers = self.runner.runner_only_attn_layers
        add_kv_sharing_layers_to_kv_cache_groups(
            self.runner.shared_kv_cache_layers,
            kv_cache_config.kv_cache_groups,
            runner_only_attn_layers,
        )

    def _add_shared_kv_cache_aliases(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if not self.runner.shared_kv_cache_layers:
            return
        for layer_name in self.runner.shared_kv_cache_layers:
            if layer_name in kv_caches:
                raise ValueError(
                    f"Shared KV layer {layer_name} was allocated its own KV cache."
                )
        # `get_kv_cache_spec` already rejects a target that is itself a shared
        # layer, so the mapping is a flat shared -> owner relation (no chains or
        # cycles) and can be aliased directly.
        for layer_name, target_layer_name in self.runner.shared_kv_cache_layers.items():
            if target_layer_name not in kv_caches:
                raise ValueError(
                    f"Layer {layer_name} reuses KV cache from "
                    f"{target_layer_name}, but the target cache was not "
                    "allocated."
                )
            kv_caches[layer_name] = kv_caches[target_layer_name]

    def _update_attention_page_size_padded(
        self, layers: dict[str, AttentionLayerBase], block_size: int
    ) -> None:
        """Pad attention page sizes so vLLM's num_blocks matches what
        the TPU allocates per layer.

        If every attention layer already has the same natural TPU page
        size, there's nothing to compensate for: leave
        `mamba_page_size_padded` as a no-op default.

        If they differ, vLLM's own `unify_kv_cache_spec_page_size` will
        rescale the smaller layer's `block_size` to match but leaves
        its `page_size_padded` at the old, now-too-small
        value, which trips `AttentionSpec.page_size_bytes`'s own
        `page_size_padded >= real_page_size_bytes` assertion. Instead,
        pin every layer's `page_size_padded` to the max natural size
        up front (mirroring the Mamba-hybrid case below), so all
        layers already report equal `page_size_bytes` and
        `unify_kv_cache_spec_page_size` is a no-op.

        NOTE: Cannot leave `page_size_padded=None` on every layer, because
        vLLM core uses `page_size_padded` to size `num_blocks` against the HBM
        budget. With `page_size_padded=None`, that sizing falls back to
        generic page-size formula (https://github.com/vllm-project/vllm/blob/48aa8d8d7529d2314858d8487cc0a21789fc7ec1/vllm/v1/kv_cache_interface.py#L204-L218)
        which has no notion of TPU's packing/alignment rules (e.g.
        fp8 packs 4 elements per 32-bit lane, so `num_kv_heads * 2` gets
        rounded up to a multiple of 4 in the actual TPU tensor -- see
        `get_kv_cache_shape`). Where that rounding changes the byte count --
        e.g. a GQA layer TP-sharded down to a single KV head -- the real TPU
        allocation ends up larger than vLLM's generic formula accounted for,
        so `num_blocks` is silently oversized and HBM usage silently exceeds
        `gpu_memory_utilization`'s target, instead of failing loudly the way
        this assertion does. `page_size_padded` has to stay pinned to the
        true, packing-aware size on every layer; this function's job is only
        to make sure every layer is pinned to the *same* one.
        """
        attn_page_sizes = set()
        backend_cls = TpuPlatform._find_non_ssm_backend(self.runner.vllm_config)
        for m in layers.values():
            if isinstance(m, Attention):
                attn_page_sizes.add(
                    backend_cls.get_kv_cache_page_size_bytes(
                        block_size,
                        m.num_kv_heads,
                        m.head_size,
                        self.runner.kv_cache_dtype,
                    )
                )
            elif isinstance(m, MLAAttention):
                attn_page_sizes.add(
                    PallasMLAttentionBackend.get_kv_cache_page_size_bytes(
                        block_size,
                        1,
                        m.head_size,
                        self.runner.kv_cache_dtype,
                    )
                )

        if not attn_page_sizes:
            return
        elif len(attn_page_sizes) > 1:
            uniform_page_size_bytes = max(attn_page_sizes)
            self.runner._hybrid_uniform_page_size_bytes = uniform_page_size_bytes
            self.runner.cache_config.mamba_page_size_padded = uniform_page_size_bytes
            logger.info(
                "Pure-attention hybrid KV cache: padding every layer "
                "spec to %d bytes (max of native sizes %s). Avoids "
                "vLLM's unify_kv_cache_spec_page_size leaving a stale "
                "page_size_padded behind when layer head_dims differ.",
                uniform_page_size_bytes,
                sorted(attn_page_sizes),
            )
        else:
            self.runner.cache_config.mamba_page_size_padded = attn_page_sizes.pop()

    def _update_mamba_page_size_padded(
        self, layers: dict[str, AttentionLayerBase]
    ) -> None:
        """Pad attention and mamba page sizes so vLLM's num_blocks matches
        what the TPU allocates per layer.

        vLLM overlays cache groups in the same backing allocation: one
        layer position from every group shares each region's block IDs.
        TPU's non-unified path keeps separate typed arrays for those layers.
        Report a uniform page equal to the sum of their physical pages
        (`num_attn_groups * attn_page + num_mamba_groups * mamba_unpadded`),
        so the scheduler's block count fits every array within the HBM budget.
        Attention bytes include TPU dtype packing and head/lane padding.

        Args:
            layers: A dictionary mapping layer names to their corresponding
                attention module instances (e.g., `MambaBase`, `Attention`).
        """
        attn_modules = [
            m for m in layers.values() if isinstance(m, (Attention, MLAAttention))
        ]
        if not attn_modules:
            return

        first_attn_module = attn_modules[0]
        num_kv_heads = (
            first_attn_module.num_kv_heads
            if isinstance(first_attn_module, Attention)
            else 1
        )
        attention_backend = TpuPlatform._find_non_ssm_backend(self.runner.vllm_config)
        attn_page_size_bytes = attention_backend.get_kv_cache_page_size_bytes(
            self.runner.block_size,
            num_kv_heads,
            first_attn_module.head_size,
            self.runner.kv_cache_dtype,
        )

        mamba_modules = [m for m in layers.values() if isinstance(m, MambaBase)]
        if not mamba_modules:
            # Not hybrid; set `mamba_page_size_padded` to the attention
            # page size as a no-op default (vLLM's platform interface sets
            # this too when it detects hybrid). No layer duplication will
            # happen without mamba layers, so no block-ID mismatch to fix.
            self.runner.cache_config.mamba_page_size_padded = attn_page_size_bytes
            return

        # Compute the unpadded mamba page size from an actual mamba module's
        # spec (shapes × dtype-size), ignoring any existing padding.
        first_mamba_spec = mamba_modules[0].get_kv_cache_spec(self.runner.vllm_config)
        assert isinstance(first_mamba_spec, MambaSpec)
        unpadded_mamba_page_size = dataclasses.replace(
            first_mamba_spec, page_size_padded=None
        ).page_size_bytes

        # Derive vLLM's kv-cache group layout. vLLM splits each type into
        # equal-sized groups of `group_size` layers. Their native placements
        # overlay the same `group_size` regions, so each region backs one
        # layer from every group.
        #
        # Choosing `group_size` trades off padding vs. number of groups:
        #   * group_size = max_count → fewer groups (often 1 per type),
        #     but the smaller side pads its group up to max_count layers
        #     (wastes space if max ≫ min).
        #   * group_size = min_count → no padding, but the larger side
        #     splits into `ceil(max/min)` groups.
        # vLLM's rule: pick max_count only when counts are close enough
        # that the padding is minor (max < 1.5 × min), else min_count.
        #   e.g. 12 sliding-window + 13 full-attn → max (1 group each)
        #   e.g. 10 full-attn      + 30 mamba     → min (1 attn + 3 mamba)
        #
        # This duplicates the heuristic from
        # `vllm/v1/core/kv_cache_utils.py::_get_kv_cache_groups_uniform_page_size`.
        # We can't call it directly because vLLM's grouping needs a fully
        # populated spec dict, while we need the group layout *before* we
        # can finish creating the specs (padding depends on grouping,
        # spec creation depends on padding). Keep in sync if that
        # heuristic ever changes — it has been stable since the hybrid
        # allocator landed.
        num_attn = len(attn_modules)
        num_mamba = len(mamba_modules)
        min_count = min(num_attn, num_mamba)
        max_count = max(num_attn, num_mamba)

        # Match vLLM exactly: float comparison, no int() truncation (matters
        # at e.g. min=3, max=4, where 4 < 4.5 but 4 < int(4.5)==4 differs).
        if max_count < min_count * 1.5:
            group_size = max_count
        else:
            group_size = min_count

        num_attn_groups = (num_attn + group_size - 1) // group_size
        num_mamba_groups = (num_mamba + group_size - 1) // group_size

        uniform_page_size_bytes = (
            num_attn_groups * attn_page_size_bytes
            + num_mamba_groups * unpadded_mamba_page_size
        )

        logger.info(
            "Hybrid KV cache: padding every layer spec to %d bytes "
            "(num_attn_groups=%d × attn_page=%d + "
            "num_mamba_groups=%d × mamba_unpadded=%d). This makes vLLM's "
            "num_blocks match per-layer TPU allocation when mamba layers "
            "cannot be truly shared.",
            uniform_page_size_bytes,
            num_attn_groups,
            attn_page_size_bytes,
            num_mamba_groups,
            unpadded_mamba_page_size,
        )

        self.runner._hybrid_uniform_page_size_bytes = int(uniform_page_size_bytes)
        self.runner.cache_config.mamba_page_size_padded = int(uniform_page_size_bytes)

        # Prefer compact-mamba sizing: cap each mamba layer at
        # `max_num_reqs + 1` recurrent slots and give the freed HBM to the
        # attention pool. Mamba state is recurrent — one slot per active
        # request — so the uniform layout wastes `num_blocks - max_num_reqs`
        # mamba slots forever. On success this pins
        # `num_gpu_blocks_override` to the (larger) attention block count and
        # sets `_mamba_num_blocks`.
        self._maybe_set_compact_mamba_num_blocks_override(
            attn_page_size_bytes,
            int(unpadded_mamba_page_size),
            num_attn_groups,
            num_mamba_groups,
            group_size,
        )

        # Fallback (compact sizing skipped, e.g. CPU-only tests or a
        # user-pinned num_gpu_blocks_override): pin vLLM's num_blocks via the two-step
        # flooring that keeps peak HBM within `gpu_memory_utilization ×
        # total_hbm` at high utilization. See `_maybe_set_num_blocks_override`
        # for the formula and rationale; the short version is that vLLM's
        # single-step `floor(avail / (uniform × group_size))` can land one
        # block higher than the two-step value, and that extra block ×
        # group_size × uniform bytes is enough to push past the budget against
        # imprecision in vLLM's `avail` estimate.
        if self.runner._mamba_num_blocks is None:
            self._maybe_set_num_blocks_override(
                attn_page_size_bytes, int(uniform_page_size_bytes), group_size
            )

    def _available_kv_cache_hbm(self) -> int:
        """KV-cache HBM budget for the block-count override paths.

        Matches `TPUWorker.determine_available_memory()`: the
        `utils.compute_hbm_budget` result (which reserves the
        `gpu_memory_utilization` cap minus
        `utils.estimate_kv_connector_hbm_reserve` (HBM a connector allocates
        after profile_run, such as the offload H2D
        staging pool). Sizing overrides against this budget keeps them from
        filling the connector reserve back up with KV blocks and defeating
        the worker-side subtraction.
        """
        if kv_cache_memory_bytes := self.runner.cache_config.kv_cache_memory_bytes:
            return kv_cache_memory_bytes
        budget = utils.compute_hbm_budget(
            [self.runner.device], self.runner.cache_config.gpu_memory_utilization
        )
        return budget.available - utils.estimate_kv_connector_hbm_reserve(
            self.runner.vllm_config
        )

    def _maybe_set_compact_mamba_num_blocks_override(
        self,
        attn_page_size_bytes: int,
        unpadded_mamba_page_size_bytes: int,
        num_attn_groups: int,
        num_mamba_groups: int,
        group_size: int,
    ) -> None:
        """Cap mamba layers at `max_num_reqs + 1` recurrent slots and pin
        `cache_config.num_gpu_blocks_override` so the freed HBM grows the
        attention pool.

        Tradeoff vs. the uniform num_blocks layout
        ------------------------------------------
        Mamba state is recurrent: one slot per *active* request, regardless of
        context length. The uniform layout (set by
        `_maybe_set_num_blocks_override`) gives every layer the same
        `num_blocks`, leaving `num_blocks - max_num_reqs` mamba slots idle
        forever. The compact layout caps mamba at `max_num_reqs + 1` (the `+1`
        is the null/sentinel slot), which is strictly better for any model
        where `num_blocks > max_num_reqs` — i.e. all production hybrid configs
        we run. Cost: the GDN op must index mamba state by per-request slot id
        (`AttentionMetadata.mamba_state_indices`) rather than by
        `block_tables[:, 0]`, since the mamba leading dim is now smaller than
        the attention pool.

        Sizing math
        -----------
        vLLM allocates `group_size` layer regions, each shared across
        `num_attn_groups` attention layers + `num_mamba_groups` mamba layers.
        With per-tensor budget `B = avail / group_size` and
        `N_mamba = max_num_reqs + 1`,
            N_attn = floor((B - num_mamba_groups × N_mamba × mamba_unpadded)
                            / (num_attn_groups × attn_page)).

        We do NOT round to a sharding divisor: our single-process
        TP layout does not shard the mamba/attention block (leading) dim — TP
        sharding is on the head dims via the model — so any positive block
        count is valid.

        Args:
            attn_page_size_bytes: TPU-actual bytes per block per attention
                layer (accounts for dtype packing like fp8).
            unpadded_mamba_page_size_bytes: bytes per slot per mamba layer
                (`prod(shape) × dtype_size`, no padding).
            num_attn_groups: # attention layers backed by each KVCacheTensor.
            num_mamba_groups: # mamba layers backed by each KVCacheTensor.
            group_size: # layer regions vLLM allocates (= layers per
                kv-cache group); the same value passed to
                `_maybe_set_num_blocks_override`.

        On success: sets `cache_config.num_gpu_blocks_override` (attention
        block count) and `_mamba_num_blocks`. Raises when the KV budget cannot
        hold the mamba slots; the remaining precondition-fail paths leave both
        unset so the caller falls back to uniform sizing.
        """
        if self.runner._uniform_mamba_layout:
            logger.info("Compact mamba sizing skipped.")
            return
        cache_config = self.runner.cache_config
        if cache_config.num_gpu_blocks_override is not None:
            return
        if group_size <= 0:
            return

        avail = self._available_kv_cache_hbm()
        if avail <= 0:
            return

        # +1 reserves slot 0 as the null/sentinel block (never handed to a
        # request); padded tail positions in `mamba_state_indices` also point
        # here, so their writes can never corrupt an active request's state.
        # With speculative decoding each request owns a *group* of
        # `num_spec + 1` consecutive slots so the GDN kernel can checkpoint
        # the state after every speculative window position (see
        # `_init_mamba_slot_pool` for the rollback scheme).
        mamba_num_blocks = self.runner.max_num_reqs * self.runner._mamba_slot_stride + 1

        avail_per_tensor = avail // group_size
        mamba_per_tensor = (
            num_mamba_groups * mamba_num_blocks * unpadded_mamba_page_size_bytes
        )
        # Falling back to the uniform layout here would pad every block to the
        # mamba page and silently shrink the pool by ~50x, which surfaces as a
        # throughput collapse rather than a misconfiguration. Fail loudly.
        attn_per_tensor_avail = avail_per_tensor - mamba_per_tensor
        if attn_per_tensor_avail <= 0:
            raise ValueError(
                f"Compact-mamba KV sizing does not fit: mamba slots alone need "
                f"{mamba_per_tensor} B per layer region (mamba_num_blocks="
                f"{mamba_num_blocks} x num_mamba_groups={num_mamba_groups} x "
                f"mamba_unpadded={unpadded_mamba_page_size_bytes}), but the "
                f"per-tensor KV budget is {avail_per_tensor} B. Raise "
                f"`gpu_memory_utilization` or lower `max_num_seqs`."
            )

        attn_num_blocks = attn_per_tensor_avail // (
            num_attn_groups * attn_page_size_bytes
        )
        if attn_num_blocks <= 0:
            raise ValueError(
                f"Compact-mamba KV sizing does not fit: no attention blocks "
                f"remain (avail_per_tensor={avail_per_tensor} B, "
                f"mamba_per_tensor={mamba_per_tensor} B). Raise "
                f"`gpu_memory_utilization` or lower `max_num_seqs`."
            )

        cache_config.num_gpu_blocks_override = int(attn_num_blocks)
        self.runner._mamba_num_blocks = int(mamba_num_blocks)

        # Total HBM = group_size regions, each holding num_attn_groups
        # attention blocks + num_mamba_groups mamba slots.
        attn_bytes = (
            group_size * num_attn_groups * attn_num_blocks * attn_page_size_bytes
        )
        mamba_bytes = (
            group_size
            * num_mamba_groups
            * mamba_num_blocks
            * unpadded_mamba_page_size_bytes
        )
        logger.info(
            "Compact-mamba KV cache: num_gpu_blocks_override=%d (attn), "
            "_mamba_num_blocks=%d. HBM split: attn=%.2f GiB; "
            "mamba=%.2f GiB; total=%.2f GiB / avail=%.2f GiB.",
            attn_num_blocks,
            mamba_num_blocks,
            attn_bytes / (2**30),
            mamba_bytes / (2**30),
            (attn_bytes + mamba_bytes) / (2**30),
            avail / (2**30),
        )

    def _maybe_set_num_blocks_override(
        self, attn_page_size_bytes: int, uniform_page_size_bytes: int, group_size: int
    ) -> None:
        """Pin `cache_config.num_gpu_blocks_override` to the two-step
        flooring value that keeps peak HBM within the user-set
        `gpu_memory_utilization` budget at high utilization.

        Formula:
          `num_blocks_attn = floor(avail / (attn_page × group_size))`
          `num_blocks_tpu  = floor(attn_page × num_blocks_attn / uniform)`

        The two-step flooring can land 1 block lower than vLLM's
        single-step `floor(avail / (uniform × group_size))`. Since
        `uniform > attn_page`, that 1-block gap costs `group_size × uniform`
        bytes of HBM, which — against the imprecision in vLLM's `avail`
        estimate — is enough to tip high-utilization configurations into
        OOM. Pinning to `num_blocks_tpu` preserves the headroom the
        single-step formula silently removes.

        `avail` comes from `_available_kv_cache_hbm`, so the block count
        pinned here matches the KV-cache budget
        `TPUWorker.determine_available_memory()` hands vLLM.

        Skipped only if the user has explicitly set `num_gpu_blocks_override`.

        Args:
            attn_page_size_bytes: TPU-actual bytes per block for one
                attention layer, from `get_attention_page_size_bytes`
                (accounts for dtype packing like fp8).
            uniform_page_size_bytes: bytes per block for one overlaid layer region
                shared across `num_attn_groups + num_mamba_groups` layers
                (the `_hybrid_uniform_page_size_bytes` value set above).
            group_size: number of layers per vLLM kv-cache group, used by
                vLLM to compute `num_blocks` from the attention tensor size.

        Returns:
            None. Side effect: sets `cache_config.num_gpu_blocks_override`
            if all preconditions hold; otherwise leaves it unset.
        """
        cache_config = self.runner.cache_config
        if cache_config.num_gpu_blocks_override is not None:
            return

        avail = self._available_kv_cache_hbm()
        if avail <= 0:
            return

        naive_vllm_num_blocks = avail // (attn_page_size_bytes * group_size)
        if naive_vllm_num_blocks <= 0:
            return
        naive_tensor_size = attn_page_size_bytes * naive_vllm_num_blocks
        num_blocks_tpu = naive_tensor_size // uniform_page_size_bytes
        if num_blocks_tpu <= 0:
            return

        cache_config.num_gpu_blocks_override = int(num_blocks_tpu)
        logger.info(
            "Hybrid KV cache: setting num_gpu_blocks_override=%d to align "
            "the scheduler's block pool with per-layer TPU allocation "
            "(avail=%d, naive_vllm_num_blocks=%d).",
            num_blocks_tpu,
            avail,
            naive_vllm_num_blocks,
        )

    def get_kv_prewarm_shapes(self) -> list[int]:
        if not has_kv_transfer_group():
            return []
        kv_connector = get_kv_transfer_group()
        cw = getattr(kv_connector, "connector_worker", None)
        if (
            cw is None
            or not hasattr(cw, "spec")
            or not hasattr(cw.spec, "prewarm_shapes")
        ):
            return []
        return cw.spec.prewarm_shapes

    def prewarm_kv_offload_shape(self, p: int) -> None:
        if not has_kv_transfer_group():
            return
        kv_connector = get_kv_transfer_group()
        cw = getattr(kv_connector, "connector_worker", None)
        if cw is not None and hasattr(cw, "spec") and hasattr(cw.spec, "prewarm_shape"):
            cw.spec.prewarm_shape(p)

    def _initialize_unified_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        self.runner.attn_groups = []
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            layer_names = list(group.layer_names)
            backend = self.runner._resolve_tpu_group_backend(
                layer_names, group.kv_cache_spec
            )
            if isinstance(group.kv_cache_spec, AttentionSpec) and self.runner.use_spmd:
                num_kv_heads = group.kv_cache_spec.num_kv_heads
                parallel_config = self.runner.parallel_config
                tp_size = parallel_config.tensor_parallel_size
                assert num_kv_heads % tp_size == 0, (
                    f"num_kv_heads {num_kv_heads} must be divisible by "
                    f"tp_size {tp_size} under SPMD mode"
                )
            self.runner.attn_groups.append(
                [
                    AttentionGroup(
                        backend=backend,
                        layer_names=layer_names,
                        kv_cache_spec=group.kv_cache_spec,
                        kv_cache_group_id=gid,
                        metadata_builders=[],
                    )
                ]
            )

        # The HND batched-RPA backend distinguishes its generic 128-token page
        # from the wider page sizes supported by PCP streaming via the active
        # vLLM config. Cache initialization normally runs outside that context.
        with set_current_vllm_config(self.runner.vllm_config):
            kernel_block_sizes = prepare_kernel_block_sizes(
                kv_cache_config, self.runner.attn_groups
            )
        self.runner._kernel_block_sizes = kernel_block_sizes
        kernel_block_size_by_gid = build_kernel_block_size_by_group_id(
            kv_cache_config=kv_cache_config,
            kernel_block_sizes=kernel_block_sizes,
        )
        self.runner.may_reinitialize_input_batch(kv_cache_config, kernel_block_sizes)

        for group_list in self.runner.attn_groups:
            for group in group_list:
                kv_cache_spec = group.kv_cache_spec
                kernel_block_size = kernel_block_size_by_gid.get(
                    group.kv_cache_group_id
                )
                if (
                    isinstance(kv_cache_spec, AttentionSpec)
                    and kernel_block_size is not None
                ):
                    kv_cache_spec = kv_cache_spec.copy_with_new_block_size(
                        kernel_block_size
                    )
                    if isinstance(kv_cache_spec, FullAttentionSpec):
                        # The schedule check sizes the batched kernel's
                        # table from this group's page size and block
                        # table width.
                        self.runner._attention_kernel_block_size = kernel_block_size
                        self.runner._attention_kv_cache_group_id = (
                            group.kv_cache_group_id
                        )
                builder = AttentionMetadataBuilder(
                    kv_cache_spec,
                    group.layer_names,
                    self.runner.vllm_config,
                    self.runner.device,
                    runner=self.runner,
                    kv_cache_group_id=group.kv_cache_group_id,
                )
                group.metadata_builders.append(builder)

        for group_id in range(len(kv_cache_config.kv_cache_groups)):
            assert (
                self.runner.block_table_cpu.dtype
                == self.runner.input_batch.block_table[group_id].get_cpu_tensor().dtype
            )

        materialized = materialize_kv_cache_tensors(
            kv_cache_config=kv_cache_config,
            attn_groups=self.runner.attn_groups,
            kernel_block_sizes=kernel_block_sizes,
            device=self.runner.device,
            cache_dtype=self.runner.kv_cache_dtype,
        )
        kv_caches = materialized.kv_caches
        self.runner.kv_cache_raw_tensors = materialized.raw_tensors
        if self.runner._unified_kv_layout:
            self.runner._build_mamba_copy_plan(
                kv_cache_config, materialized.raw_tensors
            )

        for layer_name, target_layer_name in self.runner.shared_kv_cache_layers.items():
            logger.debug("%s reuses KV cache of %s", layer_name, target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

        if self.runner._unified_kv_layout:
            # Flush the pool zero-fill before any compiled execution. PJRT
            # only donates quiescent buffers: a pending write at enqueue time
            # makes every donated pool parameter fall back to a fresh
            # pool-sized output copy (a 2x-pool transient that OOMs at high
            # gpu_memory_utilization).
            if self.runner.kv_cache_raw_tensors:
                synchronize_tensors(self.runner.kv_cache_raw_tensors)
            # Same rule for the seed-copy program: compile it while the pools
            # are quiescent, before any dummy forward leaves writes pending.
            self.runner._precompile_mamba_state_seed_copies()

        logger.info(
            "%s",
            format_kv_cache_layout_summary(
                kv_cache_config=kv_cache_config,
                kv_caches=kv_caches,
                raw_tensors=self.runner.kv_cache_raw_tensors,
                attn_groups=self.runner.attn_groups,
            ),
        )

        self.runner.kv_caches = []
        utils.tpu_bind_kv_cache(
            kv_caches,
            self.runner.vllm_config.compilation_config.static_forward_context,
            self.runner.kv_caches,
        )

        if has_kv_transfer_group():
            kv_connector = get_kv_transfer_group()
            kv_connector.register_kv_caches(kv_caches)
            if hasattr(kv_connector, "set_host_xfer_buffer_ops"):
                kv_connector.set_host_xfer_buffer_ops(copy_kv_blocks)
            if hasattr(kv_connector, "register_runner"):
                kv_connector.register_runner(self.runner)

        # For hybrid models with spec decoding on the unified pool, keep a
        # per-block device buffer of mamba read offsets (num_accepted - 1
        # from each request's last verify step), indexed by manager block
        # id. The GDN kernel reads a request's initial state from checkpoint
        # `offset` of its state block, which is how rejected draft tokens
        # are rolled back (by selecting the checkpoint of the last accepted
        # token, never by copying state). The offsets follow the state
        # block through the align-mode seed copies (see
        # `_collect_mamba_state_seed_copies`).
        if (
            self.runner._unified_kv_layout
            and kv_cache_config.has_mamba_layers
            and self.runner.speculative_config is not None
        ):
            self.runner.mamba_slot_read_offsets = torch.zeros(
                kv_cache_config.num_blocks, dtype=torch.int32
            ).to(self.runner.device)

        if not self.runner.enforce_eager:
            self.runner._precompile_substitute_placeholder_token()

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        assert kv_cache_config.num_blocks is not None, (
            "KVCacheConfig.num_blocks must be resolved by the scheduler"
        )
        kv_cache_config = copy.deepcopy(kv_cache_config)
        # Mirror GPUModelRunner.initialize_kv_cache: needed by inherited
        # _update_states -> _may_reorder_batch which reads kv_cache_config.
        self.runner.kv_cache_config = kv_cache_config
        self._maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)

        # Dummy slot mapping, not used anywhere in the TPU code flow. But needed
        #  for upstream `_build_attention_metadata` call.
        self.runner.empty_slot_mappings = {
            gid: torch.empty(0, device=self.runner.device)
            for gid in range(len(self.runner.kv_cache_config.kv_cache_groups))
        }
        if self.runner._unified_kv_layout:
            # Fail closed: unified pool allocation bypasses the block-major gate below,
            # which would incorrectly register a block-major contract over unified-pool memory.
            if envs.VLLM_TPU_BLOCK_MAJOR_KV:
                raise NotImplementedError(
                    "VLLM_TPU_BLOCK_MAJOR_KV=1: the unified KV block pool "
                    "layout is not supported by the block-major KV bundle"
                )
            self._initialize_unified_kv_cache(kv_cache_config)
            return
        backend_cls = TpuPlatform._find_non_ssm_backend(self.runner.vllm_config)

        for group in kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                # DSv4: each group's spec wraps that type's per-layer specs
                # (one wrapper for MLA, one per SWA window class); unwrap one
                # for the type checks. Block size differs across groups
                # (SWA 256 vs MLA 1024), so it is not asserted uniform here.
                spec = next(iter(spec.kv_cache_specs.values()))
            if isinstance(spec, MambaSpec):
                # We can safely ignore block size for Mamba layers since they only use a single cache state per sequence.
                continue
            if (
                not isinstance(spec, AttentionSpec)
                and len(kv_cache_config.kv_cache_groups) > 1
            ):
                raise NotImplementedError(
                    "Only AttentionSpec and MambaSpec are supported in KV cache groups > 1."
                )

        block_sizes = [
            group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups
        ]

        self.runner.may_reinitialize_input_batch(kv_cache_config, block_sizes)

        # Populate self.attn_groups directly with a single shared TPU builder
        # per group; bypasses parent's initialize_attn_backend (per-backend
        # get_builder_cls dispatch, cudagraph-mode resolution, cp compat
        # checks) — all GPU-relevant and unused on TPU.
        self.runner.attn_groups = []
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            builder = AttentionMetadataBuilder(
                group.kv_cache_spec,
                group.layer_names,
                self.runner.vllm_config,
                self.runner.device,
                runner=self.runner,
                kv_cache_group_id=gid,
            )
            self.runner.attn_groups.append(
                [
                    AttentionGroup(
                        backend=None,
                        layer_names=list(group.layer_names),
                        kv_cache_spec=group.kv_cache_spec,
                        kv_cache_group_id=gid,
                        metadata_builders=[builder],
                    )
                ]
            )

        # Verify dtype compatibility between block_table_cpu and every
        # per-group block table that downstream code may index.
        for group_id in range(len(kv_cache_config.kv_cache_groups)):
            assert (
                self.runner.block_table_cpu.dtype
                == self.runner.input_batch.block_table[group_id].get_cpu_tensor().dtype
            )

        layer_name_to_spec = {}
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner.shared_kv_cache_layers:
                    continue
                if hasattr(group.kv_cache_spec, "kv_cache_specs"):
                    layer_name_to_spec[layer_name] = group.kv_cache_spec.kv_cache_specs[
                        layer_name
                    ]
                else:
                    layer_name_to_spec[layer_name] = group.kv_cache_spec

        kv_caches: dict[str, torch.Tensor] = {}
        # Actual leading-dim block count used for mamba state arrays, captured
        # at allocation time so the slot pool below is sized to exactly what
        # was allocated (compact `_mamba_num_blocks`, or uniform `num_blocks`).
        allocated_mamba_num_blocks: int | None = None

        def _per_layer_spec(layer_name: str) -> KVCacheSpec:
            spec = layer_name_to_spec[layer_name]
            if isinstance(spec, UniformTypeKVCacheSpecs):
                return spec.kv_cache_specs[layer_name]
            return spec

        # DSv4's packed layout and cache overlays are handled entirely in
        # `_initialize_ds_v4_kv_cache`, so the loop below stays DSv4-free.
        # TODO(patemotter): replace per-model flags (`_unified_kv_layout`,
        # `_is_ds_v4`) with per-layout allocation dispatch around a shared tail.
        _is_ds_v4 = any(
            is_cache_for_ds_v4(module)
            for module in self.runner.vllm_config.compilation_config.static_forward_context.values()
        )

        # ---- Block-major KV Cache Initialization (VLLM_TPU_BLOCK_MAJOR_KV) ----
        # Bundles all attention layer fragments into a single contiguous array in HBM:
        #   [num_blocks, num_layers, page_size, K2/p, p, head_dim]
        #
        # Because num_blocks is the outermost dimension, all layer fragments for a logical
        # block are contiguous in memory. This collapses save, load, and transfer operations
        # from F independent DMAs into a single hardware DMA.
        #
        # During forward execution, attention layers route through the bundled RPA kernel
        # via VllmModelWrapperContext. Bound per-layer strided views share storage with
        # the bundle, enabling deduplication during connector registration.
        #
        # Fail closed on unsupported model topologies to prevent cross-layout cache corruption.
        self.runner._kv_cache_bundle = None
        self.runner._kv_cache_bundle_layer_index = {}
        bundle_view_by_name: dict[str, torch.Tensor] = {}
        use_block_major = bool(envs.VLLM_TPU_BLOCK_MAJOR_KV)
        if use_block_major:
            unsupported = None
            if self.runner.use_spmd:
                unsupported = "SPMD"
            elif kv_cache_config.has_mamba_layers:
                unsupported = "hybrid attention+mamba models"
            elif _is_ds_v4:
                unsupported = "the DSv4 packed KV layout"
            elif self.runner.parallel_config.prefill_context_parallel_size > 1:
                unsupported = "prefill context parallelism"
            elif self.runner.parallel_config.pipeline_parallel_size > 1:
                # Per-stage bundles require per-stage offload namespace isolation.
                unsupported = "pipeline parallelism"
            elif self.runner.speculative_config is not None:
                # Speculative decode propose paths execute outside the bundle wrapper context.
                unsupported = "speculative decoding"
            elif self.runner.model_config.get_head_size() == 64:
                # Bundled RPA kernel is not supported for head_size=64 architectures.
                unsupported = "head_size=64 models (no bundled hd64 kernel)"
            if unsupported is not None:
                raise NotImplementedError(
                    f"VLLM_TPU_BLOCK_MAJOR_KV=1: {unsupported} is not "
                    "supported by the block-major KV bundle"
                )
            attn_specs = [
                s for s in layer_name_to_spec.values() if isinstance(s, AttentionSpec)
            ]
            attn_keys = {
                (s.block_size, s.num_kv_heads, s.head_size, s.dtype) for s in attn_specs
            }
            if len(attn_keys) != 1:
                raise NotImplementedError(
                    "VLLM_TPU_BLOCK_MAJOR_KV=1: heterogeneous attention "
                    f"specs {attn_keys} cannot share one bundle"
                )
            for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
                layer_names = kv_cache_tensor.layers
                if not all(
                    isinstance(layer_name_to_spec[n], AttentionSpec)
                    for n in layer_names
                ):
                    raise NotImplementedError(
                        "VLLM_TPU_BLOCK_MAJOR_KV=1: kv_cache_tensor "
                        f"containing non-attention layers {layer_names}"
                    )

            if len(kv_cache_config.kv_cache_groups) != 1:
                raise NotImplementedError(
                    "VLLM_TPU_BLOCK_MAJOR_KV=1 requires a single uniform "
                    "attention group"
                )
            from vllm_torchtpu.offload.block_major_layout import (
                block_major_layer_indices,
            )

            sample_spec = attn_specs[0]
            page_shape = PallasAttentionBackend.get_kv_cache_shape(
                1,
                sample_spec.block_size,
                sample_spec.num_kv_heads,
                sample_spec.head_size,
                self.runner.kv_cache_dtype,
            )[1:]
            page_bytes = math.prod(page_shape) * sample_spec.dtype.itemsize
            indices = block_major_layer_indices(kv_cache_config, page_bytes)
            bundle = torch.zeros(
                (kv_cache_config.num_blocks, len(set(indices.values())), *page_shape),
                dtype=sample_spec.dtype,
                device=self.runner.device,
            )
            # Kernels donate the full bundle; views serve metadata and DMA
            # registration, retaining the native descriptor's byte offsets.
            self.runner._kv_cache_bundle = bundle
            synchronize_tensors(bundle, wait=True)
            for name, index in indices.items():
                self.runner._kv_cache_bundle_layer_index[name] = index
                bundle_view_by_name[name] = bundle[:, index]
            logger.info(
                "Block-major KV cache: allocated native bundle %s %s",
                tuple(bundle.shape),
                bundle.dtype,
            )

        if _is_ds_v4:
            self.ds_v4._initialize_ds_v4_kv_cache(
                kv_cache_config, _per_layer_spec, kv_caches, kv_cache_config.num_blocks
            )

        # Pure-attention HMA groups overlay native placement regions. The
        # non-unified Mamba path instead budgets separate typed arrays in
        # _update_mamba_page_size_padded and must keep those allocations.
        share_attention_regions = not kv_cache_config.has_mamba_layers
        attention_regions: dict[
            tuple[int, int],
            dict[
                tuple[tuple[tuple[int, ...], torch.dtype], ...],
                tuple[torch.Tensor, ...],
            ],
        ] = {}

        def allocate_attention_region(
            region: tuple[int, int],
            geometry: list[tuple[tuple[int, ...], torch.dtype]],
        ) -> tuple[torch.Tensor, ...]:
            if share_attention_regions:
                native_caches = attention_regions.setdefault(region, {})
                key = tuple(geometry)
                if key in native_caches:
                    return native_caches[key]
                allocated_bytes = sum(
                    cache.nbytes
                    for caches in native_caches.values()
                    for cache in caches
                )
                required_bytes = sum(
                    math.prod(shape) * dtype.itemsize for shape, dtype in geometry
                )
                budget = kv_cache_config.num_blocks * region[1]
                assert allocated_bytes + required_bytes <= budget, (
                    "Native TPU attention arrays exceed the padded region "
                    f"budget: region={region}, allocated={allocated_bytes}, "
                    f"required={required_bytes}, budget={budget}"
                )
            caches = tuple(
                torch.zeros(shape, dtype=dtype).to(self.runner.device)
                for shape, dtype in geometry
            )
            if share_attention_regions:
                native_caches[key] = caches
            return caches

        shared_attention_caches: dict[tuple[int, int], torch.Tensor] = {}
        if share_attention_regions and not use_block_major and not _is_ds_v4:
            region_specs: dict[tuple[int, int], list[tuple[str, KVCacheSpec]]] = {}
            for tensor in kv_cache_config.kv_cache_tensors:
                for index, name in enumerate(tensor.layers):
                    region = (
                        tensor.offset + index * tensor.layer_stride,
                        tensor.block_stride,
                    )
                    region_specs.setdefault(region, []).append(
                        (name, _per_layer_spec(name))
                    )
            for region, layer_specs in region_specs.items():
                attention_specs = [
                    (name, spec)
                    for name, spec in layer_specs
                    if isinstance(spec, AttentionSpec)
                    and not isinstance(spec, MLAAttentionSpec)
                ]
                if (
                    len(attention_specs) <= 1
                    or len(attention_specs) != len(layer_specs)
                    or attention_specs[0][1].page_size_bytes != region[1]
                    or not can_share_attention_cache(
                        [spec for _, spec in attention_specs], backend_cls
                    )
                ):
                    continue
                shared_attention_caches[region] = materialize_shared_attention_cache(
                    layer_specs=attention_specs,
                    tensor_size=kv_cache_config.num_blocks * region[1],
                    num_blocks=kv_cache_config.num_blocks,
                    attn_backend=backend_cls,
                    device=self.runner.device,
                )

        for kv_cache_tensor in [] if _is_ds_v4 else kv_cache_config.kv_cache_tensors:
            # Each descriptor names layers within the whole backing allocation;
            # its size is not a per-layer budget. TPU's separate typed caches
            # use the scheduler's block count and the normalized page geometry.
            num_blocks = kv_cache_config.num_blocks
            for layer_index, layer_name in enumerate(kv_cache_tensor.layers):
                kv_cache_spec = _per_layer_spec(layer_name)
                region = (
                    kv_cache_tensor.offset + layer_index * kv_cache_tensor.layer_stride,
                    kv_cache_tensor.block_stride,
                )

                if isinstance(kv_cache_spec, MambaSpec):
                    # Compact-mamba: allocate only `_mamba_num_blocks`
                    # recurrent slots (= max_num_reqs + 1) when the override
                    # succeeded; otherwise fall back to the uniform
                    # `num_blocks`. Attention layers always keep `num_blocks`.
                    mamba_num_blocks = (
                        self.runner._mamba_num_blocks
                        if self.runner._mamba_num_blocks is not None
                        else num_blocks
                    )
                    allocated_mamba_num_blocks = mamba_num_blocks
                    mamba_states = []
                    for _, (shape, dtype) in enumerate(
                        zip(kv_cache_spec.shapes, kv_cache_spec.dtypes)
                    ):
                        cache_shape = (mamba_num_blocks, *shape)
                        mamba_states.append(
                            torch.zeros(cache_shape, dtype=dtype).to(self.runner.device)
                        )
                    kv_caches[layer_name] = tuple(mamba_states)
                elif isinstance(kv_cache_spec, MLAAttentionSpec):
                    attn_module = self.runner.vllm_config.compilation_config.static_forward_context.get(
                        layer_name
                    )
                    if getattr(attn_module, "use_sparse", False):
                        # Split (nope, rope) cache; each half's shape and dtype
                        # come from its own layout spec.
                        specs = PallasMLAttentionBackend.get_sparse_kv_cache_specs(
                            num_blocks,
                            kv_cache_spec.block_size,
                            kv_cache_spec.head_size,
                            kv_cache_spec.dtype,
                        )
                        attn_module.mla_kv_spec = specs
                        kv_caches[layer_name] = allocate_attention_region(
                            region,
                            [(tuple(spec.shape), spec.torch_dtype) for spec in specs],
                        )
                        continue
                    # SPMD Cache Invariance Details for Multi-Head Latent Attention (MLA):
                    # Because MLA maps all attention heads onto a single joint compressed latent key-value
                    # representation (`num_kv_heads=1`), the physical KV cache dimension never splits
                    # across tensor parallel ranks (`tp_size`) during SPMD graph execution (`self.use_spmd`).
                    # Each device partition consistently retains a complete, unsliced replication of the
                    # compressed latent cache structure across multi-chip execution loops.
                    kv_cache_shape = PallasMLAttentionBackend.get_kv_cache_shape(
                        num_blocks,
                        kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                        kv_cache_spec.dtype,
                    )
                    dtype = kv_cache_spec.dtype
                    kv_caches[layer_name] = allocate_attention_region(
                        region, [(tuple(kv_cache_shape), dtype)]
                    )[0]
                elif isinstance(kv_cache_spec, AttentionSpec):
                    if self.runner.use_spmd:
                        num_kv_heads = kv_cache_spec.num_kv_heads
                        parallel_config = self.runner.parallel_config
                        tp_size = parallel_config.tensor_parallel_size
                        # TODO: Handle kv cache duplication under SPMD mode.
                        assert num_kv_heads % tp_size == 0, (
                            f"num_kv_heads {num_kv_heads} must be divisible by "
                            f"tp_size {tp_size} under SPMD mode"
                        )
                    if region in shared_attention_caches:
                        kv_caches[layer_name] = shared_attention_caches[region]
                    elif use_block_major and layer_name in bundle_view_by_name:
                        # Bind the strided per-layer view sharing storage with self._kv_cache_bundle.
                        kv_caches[layer_name] = bundle_view_by_name[layer_name]
                    else:
                        kv_cache_shape = backend_cls.get_kv_cache_shape(
                            num_blocks,
                            kv_cache_spec.block_size,
                            kv_cache_spec.num_kv_heads,
                            kv_cache_spec.head_size,
                            kv_cache_spec.dtype,
                        )
                        dtype = kv_cache_spec.dtype
                        kv_caches[layer_name] = allocate_attention_region(
                            region, [(tuple(kv_cache_shape), dtype)]
                        )[0]
                else:
                    raise NotImplementedError

        self._add_shared_kv_cache_aliases(kv_caches)

        # Mark KV cache buffers as donation candidates outside torch.compile
        # regions to avoid Dynamo tracing through pybind calls.
        # TODO(geyuhao): Comment out for now as TorchTPU does not support this right now
        # for kv_cache in kv_caches.values():
        #     pallas.set_buffer_donor_(kv_cache, True)

        # Reset kv_caches list (tpu_bind_kv_cache expects empty list)
        self.runner.kv_caches = []

        if not use_block_major and not _is_ds_v4:
            check_kv_caches_cover_block_ids(
                kv_caches,
                kv_cache_config.kv_cache_tensors,
                kv_cache_config.num_blocks,
                _per_layer_spec,
            )

        # Use tpu_bind_kv_cache to bind KV caches to attention layers using layer names
        # This is the native vLLM pattern and avoids the 'layer_id' attribute error
        utils.tpu_bind_kv_cache(
            kv_caches,
            self.runner.vllm_config.compilation_config.static_forward_context,
            self.runner.kv_caches,
        )

        # Pre-build and cache bundled RPA kernels across all attention layers before torch.compile tracing.
        if self.runner._kv_cache_bundle is not None:
            from vllm_torchtpu.layers.adapter.attention import (
                PallasAttentionBackendImpl,
            )

            layers = get_layers_from_vllm_config(self.runner.vllm_config, Attention)
            # Only enter wrapper context when eligible attention layers are present.
            eligible = [
                (name, attn_layer)
                for name, attn_layer in layers.items()
                if name in self.runner._kv_cache_bundle_layer_index
                and isinstance(attn_layer.impl, PallasAttentionBackendImpl)
            ]
            if eligible:
                with set_vllm_model_wrapper_context(
                    mesh=self.runner.mesh,
                    kv_cache_bundle=self.runner._kv_cache_bundle,
                ):
                    for name, attn_layer in eligible:
                        attn_layer.impl.setup_bundled(
                            self.runner._kv_cache_bundle_layer_index[name],
                            self.runner._kv_cache_bundle.device,
                            attn_layer,
                        )

        if self.runner.use_spmd:
            # Shard KV Cache
            for cache in self.runner.kv_caches:
                continue
                # xs.mark_sharding(cache, self.mesh, (None, "x", None, None))

        if has_kv_transfer_group():
            kv_connector = get_kv_transfer_group()
            # Register KV caches with connector. TPURaidenOffloadingConnector deduplicates strided
            # views by underlying storage, registering only the single canonical bundle.
            kv_connector.register_kv_caches(kv_caches)
            # TPUConnector reads runner.kv_caches lazily and doesn't need
            # set_host_xfer_buffer_ops; only call it on connectors that
            # expose it.
            if hasattr(kv_connector, "set_host_xfer_buffer_ops"):
                kv_connector.set_host_xfer_buffer_ops(copy_kv_blocks)
            if hasattr(kv_connector, "register_runner"):
                kv_connector.register_runner(self.runner)

        # Initialize the compact-mamba slot allocator now that the true mamba
        # block count is known. When compact sizing was skipped, mamba shares
        # the attention `num_blocks`, so the pool spans that range instead.
        if (
            allocated_mamba_num_blocks is not None
            and not self.runner._uniform_mamba_layout
        ):
            self.runner._init_mamba_slot_pool(allocated_mamba_num_blocks)

        # Compact pool counterpart of the read-offset buffer allocated in
        # `initialize_kv_cache`, indexed over the mamba slot pool rather
        # than the unified pool's blocks: here checkpoint `offset` of a
        # request is the slot `base_slot + offset`.
        if (
            allocated_mamba_num_blocks is not None
            and self.runner.speculative_config is not None
        ):
            if self.runner._uniform_mamba_layout:
                raise NotImplementedError(
                    "Speculative decoding with mamba layers requires the "
                    "compact mamba slot layout (unsupported with "
                    "kv_transfer_config / uniform mamba layout)."
                )
            self.runner.mamba_slot_read_offsets = torch.zeros(
                allocated_mamba_num_blocks, dtype=torch.int32
            ).to(self.runner.device)

        # Precompile after KV cache allocation so XLA's buffer assignment sees
        # the same HBM pressure as runtime.
        # Rebuild custom attention ops here too: only now are block sizes final.
        self.runner._initialize_attention_kernels(force=True)

        if not self.runner.enforce_eager:
            self.runner._precompile_substitute_placeholder_token()
