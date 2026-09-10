# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-cache spec derivation, page-size padding and block-count budgeting.

Moved verbatim out of `TPUModelRunner`; see vllm-project/vllm-torchtpu#713.
"""
from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import torch
from vllm.config import get_layers_from_vllm_config
from vllm.model_executor.layers.attention import (Attention,
                                                  ChunkedLocalAttention,
                                                  MLAAttention)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.models.deepseek_v4.attention import (DeepseekV4Attention,
                                               DeepseekV4IndexerCache)
from vllm.models.deepseek_v4.compressor import CompressorStateCache
from vllm.v1.attention.backend import AttentionBackend, AttentionType
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec, MambaSpec,
                                        MLAAttentionSpec, SlidingWindowSpec)
from vllm.v1.worker.utils import add_kv_sharing_layers_to_kv_cache_groups

from vllm_torchtpu import utils
from vllm_torchtpu.kv_cache_spec_normalizer import \
    normalize_kv_cache_specs_for_tpu
from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackend
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

if TYPE_CHECKING:
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)


def is_cache_for_ds_v4(attn_module: AttentionLayerBase) -> bool:
    """Whether this module owns one of DeepSeek-V4's custom KV caches.

    These build their own specs in the TPU's packed layout, so the specs must
    reach vLLM verbatim rather than through `_normalize_one_spec`.
    """
    return isinstance(attn_module,
                      (DeepseekV4Attention, DeepseekV4SWACache,
                       DeepseekV4IndexerCache, CompressorStateCache))


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
    page_shape = backend.get_kv_cache_shape(1, block_size, num_kv_heads,
                                            head_size, dtype)
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
        "or enabling attention data parallelism avoids it.", actual / unpadded,
        num_kv_heads, head_size, dtype, backend.get_name())


class KVCacheManager:
    """Owns the KV-cache spec/sizing half of `TPUModelRunner`."""

    def __init__(self, runner: TPUModelRunner) -> None:
        self.runner = runner

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
        backend_cls = TpuPlatform._find_non_ssm_backend(
            self.runner.vllm_config)
        block_size = self.runner.vllm_config.cache_config.block_size
        cache_dtype_str = self.runner.vllm_config.cache_config.cache_dtype

        has_attention = any(
            isinstance(m, (Attention, MLAAttention)) for m in layers.values())
        has_mamba = any(isinstance(m, MambaBase) for m in layers.values())
        if has_attention and not self.runner._unified_kv_layout:
            self._update_attention_page_size_padded(layers, block_size)
            if has_mamba:
                self._update_mamba_page_size_padded(layers)

        hma_enabled = (
            not self.runner.scheduler_config.disable_hybrid_kv_cache_manager)

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
                if (kv_tgt_layer :=
                        attn_module.kv_sharing_target_layer_name) is not None:
                    if kv_tgt_layer not in layers:
                        raise ValueError(
                            f"Layer {layer_name} reuses KV cache from missing "
                            f"target layer {kv_tgt_layer}.")
                    target_module = layers[kv_tgt_layer]
                    if not isinstance(target_module, Attention):
                        raise ValueError(
                            f"Layer {layer_name} reuses KV cache from "
                            f"non-attention target layer {kv_tgt_layer}.")
                    if target_module.kv_sharing_target_layer_name is not None:
                        raise ValueError(
                            f"Layer {layer_name} reuses KV cache from "
                            f"{kv_tgt_layer}, which is itself a shared KV "
                            "layer.")
                    self._validate_shared_kv_cache_layout(
                        layer_name, attn_module, kv_tgt_layer, target_module,
                        hma_enabled)
                    # The layer doesn't need its own KV cache and will use that of
                    # the target layer. We skip creating a KVCacheSpec for it, so
                    # that KV cache management logic will act as this layer does
                    # not exist, and doesn't allocate KV cache for the layer. This
                    # enables the memory saving of cross-layer kv sharing, allowing
                    # a given amount of memory to accommodate longer context lengths
                    # or enable more requests to be processed simultaneously.
                    self.runner.shared_kv_cache_layers[
                        layer_name] = kv_tgt_layer
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
                        self.runner._hybrid_uniform_page_size_bytes if
                        self.runner._hybrid_uniform_page_size_bytes is not None
                        else backend_cls.get_kv_cache_page_size_bytes(
                            block_size,
                            attn_module.num_kv_heads,
                            attn_module.head_size,
                            self.runner.kv_cache_dtype,
                        ))
                    if attn_module.sliding_window is not None:
                        kv_cache_spec[layer_name] = SlidingWindowSpec(
                            block_size=block_size,
                            num_kv_heads=attn_module.num_kv_heads,
                            head_size=attn_module.head_size,
                            dtype=self.runner.kv_cache_dtype,
                            page_size_padded=page_size_padded,
                            sliding_window=attn_module.sliding_window,
                            indexes_kv_by_block_stride=True,
                        )
                    else:
                        kv_cache_spec[layer_name] = FullAttentionSpec(
                            block_size=block_size,
                            num_kv_heads=attn_module.num_kv_heads,
                            head_size=attn_module.head_size,
                            dtype=self.runner.kv_cache_dtype,
                            page_size_padded=page_size_padded,
                            indexes_kv_by_block_stride=True,
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
                    raise ValueError(
                        f"Unknown attention type: {attn_module.attn_type}")
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
                    ))
                kv_cache_spec[layer_name] = MLAAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=attn_module.head_size,
                    dtype=self.runner.kv_cache_dtype,
                    cache_dtype_str=cache_dtype_str,
                    page_size_padded=page_size_padded,
                    indexes_kv_by_block_stride=True,
                )
            elif isinstance(attn_module, DeepseekV32IndexerCache):
                # DSA indexer K cache: the module declares its own uint8 spec
                # (head_dim fp8 bytes + 1 e8m0 scale byte per token).
                kv_cache_spec[layer_name] = attn_module.get_kv_cache_spec(
                    self.runner.vllm_config)
            else:
                continue

        # Note: that each shared layer's target actually owns a KV cache (and
        # that the shared layer was not itself allocated one) is enforced later
        # against the concrete KVCacheConfig in
        # `_maybe_add_kv_sharing_layers_to_kv_cache_groups`.
        return normalize_kv_cache_specs_for_tpu(
            kv_cache_spec,
            self.runner.kv_cache_dtype,
            enable_unified_kv_layout=self.runner._unified_kv_layout,
            exempt_layers=ds_v4_layers,
            attention_backend=backend_cls,
        )

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
            if getattr(attn_module, field_name,
                       None) != getattr(target_module, field_name, None):
                mismatches.append(field_name)
        if mismatches:
            raise ValueError(f"Layer {layer_name} cannot reuse KV cache from "
                             f"{target_layer_name}: incompatible "
                             f"{', '.join(mismatches)}.")

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
                    "and matching k/v scales.")

    def _maybe_add_kv_sharing_layers_to_kv_cache_groups(
            self, kv_cache_config: KVCacheConfig) -> None:
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
            for layer_name in kv_cache_tensor.shared_by
        }
        for layer_name, target_layer_name in self.runner.shared_kv_cache_layers.items(
        ):
            if target_layer_name not in group_layer_names:
                raise ValueError(
                    f"Layer {layer_name} reuses KV cache from "
                    f"{target_layer_name}, but the target layer is missing "
                    "from KV cache groups.")
            if (layer_name in group_layer_names
                    or layer_name in allocated_layer_names):
                raise ValueError(
                    f"Shared KV layer {layer_name} must not have an "
                    "independent KV cache allocation.")
        runner_only_attn_layers = self.runner.runner_only_attn_layers
        add_kv_sharing_layers_to_kv_cache_groups(
            self.runner.shared_kv_cache_layers,
            kv_cache_config.kv_cache_groups,
            runner_only_attn_layers,
        )

    def _add_shared_kv_cache_aliases(
            self, kv_caches: dict[str, torch.Tensor]) -> None:
        if not self.runner.shared_kv_cache_layers:
            return
        for layer_name in self.runner.shared_kv_cache_layers:
            if layer_name in kv_caches:
                raise ValueError(
                    f"Shared KV layer {layer_name} was allocated its own KV "
                    "cache.")
        # `get_kv_cache_spec` already rejects a target that is itself a shared
        # layer, so the mapping is a flat shared -> owner relation (no chains or
        # cycles) and can be aliased directly.
        for layer_name, target_layer_name in self.runner.shared_kv_cache_layers.items(
        ):
            if target_layer_name not in kv_caches:
                raise ValueError(
                    f"Layer {layer_name} reuses KV cache from "
                    f"{target_layer_name}, but the target cache was not "
                    "allocated.")
            kv_caches[layer_name] = kv_caches[target_layer_name]

    def _update_attention_page_size_padded(self,
                                           layers: dict[str,
                                                        AttentionLayerBase],
                                           block_size: int) -> None:
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
        backend_cls = TpuPlatform._find_non_ssm_backend(
            self.runner.vllm_config)
        for m in layers.values():
            if isinstance(m, Attention):
                attn_page_sizes.add(
                    backend_cls.get_kv_cache_page_size_bytes(
                        block_size,
                        m.num_kv_heads,
                        m.head_size,
                        self.runner.kv_cache_dtype,
                    ))
            elif isinstance(m, MLAAttention):
                attn_page_sizes.add(
                    PallasMLAttentionBackend.get_kv_cache_page_size_bytes(
                        block_size,
                        1,
                        m.head_size,
                        self.runner.kv_cache_dtype,
                    ))

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
                uniform_page_size_bytes, sorted(attn_page_sizes))
        else:
            self.runner.cache_config.mamba_page_size_padded = attn_page_sizes.pop(
            )

    def _update_mamba_page_size_padded(
            self, layers: dict[str, AttentionLayerBase]) -> None:
        """Pad attention and mamba page sizes so vLLM's num_blocks matches
        what the TPU allocates per layer.

        For hybrid attention+mamba models, vLLM groups a tensor's memory so
        that one `KVCacheTensor` is `shared_by` one layer from each kv-cache
        group (e.g., Qwen3.5: 1 full-attn + 3 linear-attn per shared_by).
        vLLM's scheduler assumes these layers share a single physical
        tensor at the byte level — each layer's block_table indexes into
        disjoint slots of the same backing allocation, and device kernels
        reinterpret the bytes as attention KV or mamba state depending on
        which layer is accessing the slot.

        TPU `jax.Array`s are strongly typed, so we cannot overlay an
        attention tensor and a mamba tensor on the same bytes.
        `initialize_kv_cache` therefore allocates one physical array per
        layer in the `shared_by` group, carving the group's byte budget
        into separate per-layer tensors. Without the compensation done
        here, vLLM's block pool would hold `num_shared_layers`× more
        block IDs than each per-layer array has slots — the scheduler
        would hand out block IDs beyond a layer's leading dimension,
        JAX's indexed writes would silently clip them, and multiple
        requests' mamba recurrent states would collapse onto the same
        slot (corrupted state → gibberish generation).

        The fix: set every layer's reported `page_size_padded` equal to the
        full per-`shared_by` footprint — `num_attn_groups × attn_page +
        num_mamba_groups × mamba_unpadded`, where `attn_page` is the
        TPU-actual per-block bytes (from `get_attention_page_size_bytes`,
        which accounts for dtype packing like fp8) and `mamba_unpadded` is
        the natural `prod(shape) × dtype_size`. vLLM then computes a
        smaller `num_blocks` that exactly matches what we allocate per layer
        on the TPU side. HBM usage is unchanged; only the block-ID
        accounting lines up.

        Args:
            layers: A dictionary mapping layer names to their corresponding
                attention module instances (e.g., `MambaBase`, `Attention`).
        """
        attn_modules = [
            m for m in layers.values()
            if isinstance(m, (Attention, MLAAttention))
        ]
        if not attn_modules:
            return

        first_attn_module = attn_modules[0]
        num_kv_heads = first_attn_module.num_kv_heads if isinstance(
            first_attn_module, Attention) else 1
        attention_backend = TpuPlatform._find_non_ssm_backend(
            self.runner.vllm_config)
        attn_page_size_bytes = attention_backend.get_kv_cache_page_size_bytes(
            self.runner.block_size, num_kv_heads, first_attn_module.head_size,
            self.runner.kv_cache_dtype)

        mamba_modules = [
            m for m in layers.values() if isinstance(m, MambaBase)
        ]
        if not mamba_modules:
            # Not hybrid; set `mamba_page_size_padded` to the attention
            # page size as a no-op default (vLLM's platform interface sets
            # this too when it detects hybrid). No layer duplication will
            # happen without mamba layers, so no block-ID mismatch to fix.
            self.runner.cache_config.mamba_page_size_padded = attn_page_size_bytes
            return

        # Compute the unpadded mamba page size from an actual mamba module's
        # spec (shapes × dtype-size), ignoring any existing padding.
        first_mamba_spec = mamba_modules[0].get_kv_cache_spec(
            self.runner.vllm_config)
        assert isinstance(first_mamba_spec, MambaSpec)
        unpadded_mamba_page_size = dataclasses.replace(
            first_mamba_spec, page_size_padded=None).page_size_bytes

        # Derive vLLM's kv-cache group layout. vLLM splits each type into
        # equal-sized groups of `group_size` layers, then allocates
        # `group_size` `KVCacheTensor`s, each `shared_by` one layer from
        # every group — so each tensor covers `num_attn_groups +
        # num_mamba_groups` layers.
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

        uniform_page_size_bytes = (num_attn_groups * attn_page_size_bytes +
                                   num_mamba_groups * unpadded_mamba_page_size)

        logger.info(
            "Hybrid KV cache: padding every layer spec to %d bytes "
            "(num_attn_groups=%d × attn_page=%d + "
            "num_mamba_groups=%d × mamba_unpadded=%d). This makes vLLM's "
            "num_blocks match per-layer TPU allocation when mamba layers "
            "cannot be truly shared.", uniform_page_size_bytes,
            num_attn_groups, attn_page_size_bytes, num_mamba_groups,
            unpadded_mamba_page_size)

        self.runner._hybrid_uniform_page_size_bytes = int(
            uniform_page_size_bytes)
        self.runner.cache_config.mamba_page_size_padded = int(
            uniform_page_size_bytes)

        # Prefer compact-mamba sizing: cap each mamba layer at
        # `max_num_reqs + 1` recurrent slots and give the freed HBM to the
        # attention pool. Mamba state is recurrent — one slot per active
        # request — so the uniform layout wastes `num_blocks - max_num_reqs`
        # mamba slots forever. On success this pins
        # `num_gpu_blocks_override` to the (larger) attention block count and
        # sets `_mamba_num_blocks`.
        self._maybe_set_compact_mamba_num_blocks_override(
            attn_page_size_bytes, int(unpadded_mamba_page_size),
            num_attn_groups, num_mamba_groups, group_size)

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
            self._maybe_set_num_blocks_override(attn_page_size_bytes,
                                                int(uniform_page_size_bytes),
                                                group_size)

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
            [self.runner.device],
            self.runner.cache_config.gpu_memory_utilization)
        return budget.available - utils.estimate_kv_connector_hbm_reserve(
            self.runner.vllm_config)

    def _maybe_set_compact_mamba_num_blocks_override(
            self, attn_page_size_bytes: int,
            unpadded_mamba_page_size_bytes: int, num_attn_groups: int,
            num_mamba_groups: int, group_size: int) -> None:
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
        vLLM allocates `group_size` KVCacheTensors, each shared across
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
            group_size: # KVCacheTensors vLLM allocates (= layers per
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
        mamba_per_tensor = (num_mamba_groups * mamba_num_blocks *
                            unpadded_mamba_page_size_bytes)
        # Falling back to the uniform layout here would pad every block to the
        # mamba page and silently shrink the pool by ~50x, which surfaces as a
        # throughput collapse rather than a misconfiguration. Fail loudly.
        attn_per_tensor_avail = avail_per_tensor - mamba_per_tensor
        if attn_per_tensor_avail <= 0:
            raise ValueError(
                f"Compact-mamba KV sizing does not fit: mamba slots alone need "
                f"{mamba_per_tensor} B per KVCacheTensor (mamba_num_blocks="
                f"{mamba_num_blocks} x num_mamba_groups={num_mamba_groups} x "
                f"mamba_unpadded={unpadded_mamba_page_size_bytes}), but the "
                f"per-tensor KV budget is {avail_per_tensor} B. Raise "
                f"`gpu_memory_utilization` or lower `max_num_seqs`.")

        attn_num_blocks = attn_per_tensor_avail // (num_attn_groups *
                                                    attn_page_size_bytes)
        if attn_num_blocks <= 0:
            raise ValueError(
                f"Compact-mamba KV sizing does not fit: no attention blocks "
                f"remain (avail_per_tensor={avail_per_tensor} B, "
                f"mamba_per_tensor={mamba_per_tensor} B). Raise "
                f"`gpu_memory_utilization` or lower `max_num_seqs`.")

        cache_config.num_gpu_blocks_override = int(attn_num_blocks)
        self.runner._mamba_num_blocks = int(mamba_num_blocks)

        # Total HBM = group_size KVCacheTensors, each holding num_attn_groups
        # attention blocks + num_mamba_groups mamba slots.
        attn_bytes = (group_size * num_attn_groups * attn_num_blocks *
                      attn_page_size_bytes)
        mamba_bytes = (group_size * num_mamba_groups * mamba_num_blocks *
                       unpadded_mamba_page_size_bytes)
        logger.info(
            "Compact-mamba KV cache: num_gpu_blocks_override=%d (attn), "
            "_mamba_num_blocks=%d. HBM split: attn=%.2f GiB; "
            "mamba=%.2f GiB; total=%.2f GiB / avail=%.2f GiB.",
            attn_num_blocks, mamba_num_blocks, attn_bytes / (2**30),
            mamba_bytes / (2**30), (attn_bytes + mamba_bytes) / (2**30),
            avail / (2**30))

    def _maybe_set_num_blocks_override(self, attn_page_size_bytes: int,
                                       uniform_page_size_bytes: int,
                                       group_size: int) -> None:
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
            uniform_page_size_bytes: bytes per block for one `KVCacheTensor`
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
            "(avail=%d, naive_vllm_num_blocks=%d).", num_blocks_tpu, avail,
            naive_vllm_num_blocks)
