# SPDX-License-Identifier: Apache-2.0
import copy
import dataclasses
from typing import Any, cast

import torch
from vllm.config import get_layers_from_vllm_config
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import (AttentionSpec, KVCacheConfig,
                                        KVCacheSpec, MambaSpec,
                                        MLAAttentionSpec,
                                        UniformTypeKVCacheSpecs)
from vllm.v1.worker.kv_connector_model_runner_mixin import \
    KVConnectorModelRunnerMixin
from vllm.v1.worker.utils import (AttentionGroup,
                                  add_kv_sharing_layers_to_kv_cache_groups,
                                  prepare_kernel_block_sizes)

from vllm_torchtpu import envs, utils
from vllm_torchtpu.kv_cache_materializer import (
    build_kernel_block_size_by_group_id, format_kv_cache_layout_summary,
    materialize_kv_cache_tensors)
from vllm_torchtpu.layers.common.attention_metadata import \
    AttentionMetadataBuilder
from vllm_torchtpu.layers.vllm.attention import (PallasAttentionBackend,
                                                 PallasMLAttentionBackend)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    set_vllm_model_wrapper_context
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform
from vllm_torchtpu.runner.ds_v4_kv_cache import is_cache_for_ds_v4
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)


class KvCacheInitMixin:
    """KV cache allocation and layer binding."""

    def _maybe_add_kv_sharing_layers_to_kv_cache_groups(
            self, kv_cache_config: KVCacheConfig) -> None:
        if not self.shared_kv_cache_layers:
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
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items(
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
        runner_only_attn_layers = self.runner_only_attn_layers
        add_kv_sharing_layers_to_kv_cache_groups(
            self.shared_kv_cache_layers,
            kv_cache_config.kv_cache_groups,
            runner_only_attn_layers,
        )

    def _add_shared_kv_cache_aliases(
            self, kv_caches: dict[str, torch.Tensor]) -> None:
        if not self.shared_kv_cache_layers:
            return
        for layer_name in self.shared_kv_cache_layers:
            if layer_name in kv_caches:
                raise ValueError(
                    f"Shared KV layer {layer_name} was allocated its own KV "
                    "cache.")
        # `get_kv_cache_spec` already rejects a target that is itself a shared
        # layer, so the mapping is a flat shared -> owner relation (no chains or
        # cycles) and can be aliased directly.
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items(
        ):
            if target_layer_name not in kv_caches:
                raise ValueError(
                    f"Layer {layer_name} reuses KV cache from "
                    f"{target_layer_name}, but the target cache was not "
                    "allocated.")
            kv_caches[layer_name] = kv_caches[target_layer_name]

    def get_kv_prewarm_shapes(self) -> list[int]:
        if not has_kv_transfer_group():
            return []
        kv_connector = get_kv_transfer_group()
        cw = getattr(kv_connector, "connector_worker", None)
        if (cw is None or not hasattr(cw, "spec")
                or not hasattr(cw.spec, "prewarm_shapes")):
            return []
        return cw.spec.prewarm_shapes

    def prewarm_kv_offload_shape(self, p: int) -> None:
        if not has_kv_transfer_group():
            return
        kv_connector = get_kv_transfer_group()
        cw = getattr(kv_connector, "connector_worker", None)
        if (cw is not None and hasattr(cw, "spec")
                and hasattr(cw.spec, "prewarm_shape")):
            cw.spec.prewarm_shape(p)

    def _resolve_tpu_group_backend(
        self,
        layer_names: list[str],
        kv_cache_spec: KVCacheSpec,
    ) -> type[Any]:
        layer_type = cast(type[Any], AttentionLayerBase)
        layers = get_layers_from_vllm_config(self.vllm_config, layer_type,
                                             layer_names)
        if layer_names and layer_names[0] in layers:
            return cast(type[Any], layers[layer_names[0]].get_attn_backend())
        if isinstance(kv_cache_spec, (AttentionSpec, MambaSpec)):
            return cast(type[Any], PallasAttentionBackend)
        raise NotImplementedError(
            f"Unsupported KV cache spec: {type(kv_cache_spec)!r}")

    def _initialize_unified_kv_cache(self,
                                     kv_cache_config: KVCacheConfig) -> None:
        self.attn_groups = []
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            layer_names = list(group.layer_names)
            backend = self._resolve_tpu_group_backend(layer_names,
                                                      group.kv_cache_spec)
            if isinstance(group.kv_cache_spec,
                          AttentionSpec) and self.use_spmd:
                num_kv_heads = group.kv_cache_spec.num_kv_heads
                parallel_config = self.parallel_config
                tp_size = parallel_config.tensor_parallel_size
                assert num_kv_heads % tp_size == 0, (
                    f"num_kv_heads {num_kv_heads} must be divisible by "
                    f"tp_size {tp_size} under SPMD mode")
            self.attn_groups.append([
                AttentionGroup(
                    backend=backend,
                    layer_names=layer_names,
                    kv_cache_spec=group.kv_cache_spec,
                    kv_cache_group_id=gid,
                    metadata_builders=[],
                )
            ])

        kernel_block_sizes = prepare_kernel_block_sizes(
            kv_cache_config, self.attn_groups)
        self._kernel_block_sizes = kernel_block_sizes
        kernel_block_size_by_gid = build_kernel_block_size_by_group_id(
            kv_cache_config=kv_cache_config,
            kernel_block_sizes=kernel_block_sizes,
        )
        self.may_reinitialize_input_batch(kv_cache_config, kernel_block_sizes)

        for group_list in self.attn_groups:
            for group in group_list:
                kv_cache_spec = group.kv_cache_spec
                kernel_block_size = kernel_block_size_by_gid.get(
                    group.kv_cache_group_id)
                if (isinstance(kv_cache_spec, AttentionSpec)
                        and kernel_block_size is not None):
                    kv_cache_spec = kv_cache_spec.copy_with_new_block_size(
                        kernel_block_size)
                builder = AttentionMetadataBuilder(
                    kv_cache_spec,
                    group.layer_names,
                    self.vllm_config,
                    self.device,
                    runner=self,
                    kv_cache_group_id=group.kv_cache_group_id,
                )
                group.metadata_builders.append(builder)

        for group_id in range(len(kv_cache_config.kv_cache_groups)):
            assert (self.block_table_cpu.dtype == self.input_batch.
                    block_table[group_id].get_cpu_tensor().dtype)

        materialized = materialize_kv_cache_tensors(
            kv_cache_config=kv_cache_config,
            attn_groups=self.attn_groups,
            kernel_block_sizes=kernel_block_sizes,
            device=self.device,
            cache_dtype=self.kv_cache_dtype,
        )
        kv_caches = materialized.kv_caches
        self.kv_cache_raw_tensors = materialized.raw_tensors
        if self._unified_kv_layout:
            self._build_mamba_copy_plan(kv_cache_config,
                                        materialized.raw_tensors)
        if kv_cache_config.has_mamba_layers:
            self._update_hybrid_attention_mamba_layout(kv_caches,
                                                       kernel_block_sizes)

        for layer_name, target_layer_name in self.shared_kv_cache_layers.items(
        ):
            logger.debug("%s reuses KV cache of %s", layer_name,
                         target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

        if self._unified_kv_layout:
            # Flush the pool zero-fill before any compiled execution. PJRT
            # only donates quiescent buffers: a pending write at enqueue time
            # makes every donated pool parameter fall back to a fresh
            # pool-sized output copy (a 2x-pool transient that OOMs at high
            # gpu_memory_utilization).
            if self.kv_cache_raw_tensors:
                synchronize_tensors(self.kv_cache_raw_tensors)
            # Same rule for the seed-copy program: compile it while the pools
            # are quiescent, before any dummy forward leaves writes pending.
            self._precompile_mamba_state_seed_copies()

        logger.info(
            "%s",
            format_kv_cache_layout_summary(
                kv_cache_config=kv_cache_config,
                kv_caches=kv_caches,
                raw_tensors=self.kv_cache_raw_tensors,
                attn_groups=self.attn_groups,
            ),
        )

        self.kv_caches = []
        utils.tpu_bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches,
        )

        if has_kv_transfer_group():
            kv_connector = get_kv_transfer_group()
            kv_connector.register_kv_caches(kv_caches)
            if hasattr(kv_connector, "set_host_xfer_buffer_ops"):
                kv_connector.set_host_xfer_buffer_ops(copy_kv_blocks)
            if hasattr(kv_connector, "register_runner"):
                kv_connector.register_runner(self)

        # For hybrid models with spec decoding on the unified pool, keep a
        # per-block device buffer of mamba read offsets (num_accepted - 1
        # from each request's last verify step), indexed by manager block
        # id. The GDN kernel reads a request's initial state from checkpoint
        # `offset` of its state block, which is how rejected draft tokens
        # are rolled back (by selecting the checkpoint of the last accepted
        # token, never by copying state). The offsets follow the state
        # block through the align-mode seed copies (see
        # `_collect_mamba_state_seed_copies`).
        if (self._unified_kv_layout and kv_cache_config.has_mamba_layers
                and self.speculative_config is not None):
            self.mamba_slot_read_offsets = torch.zeros(
                kv_cache_config.num_blocks, dtype=torch.int32).to(self.device)

        if not self.enforce_eager:
            self._precompile_substitute_placeholder_token()

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        assert kv_cache_config.num_blocks is not None, (
            "KVCacheConfig.num_blocks must be resolved by the scheduler")
        kv_cache_config = copy.deepcopy(kv_cache_config)
        # Mirror GPUModelRunner.initialize_kv_cache: needed by inherited
        # _update_states -> _may_reorder_batch which reads kv_cache_config.
        self.kv_cache_config = kv_cache_config
        self._maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)

        # Dummy slot mapping, not used anywhere in the TPU code flow. But needed
        #  for upstream `_build_attention_metadata` call.
        self.empty_slot_mappings = {
            gid: torch.empty(0, device=self.device)
            for gid in range(len(self.kv_cache_config.kv_cache_groups))
        }
        if self._unified_kv_layout:
            # Fail closed: unified pool allocation bypasses the block-major gate below,
            # which would incorrectly register a block-major contract over unified-pool memory.
            if envs.VLLM_TPU_BLOCK_MAJOR_KV:
                raise NotImplementedError(
                    "VLLM_TPU_BLOCK_MAJOR_KV=1: the unified KV block pool "
                    "layout is not supported by the block-major KV bundle")
            self._initialize_unified_kv_cache(kv_cache_config)
            return
        backend_cls = TpuPlatform._find_non_ssm_backend(self.vllm_config)

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
            if not isinstance(spec, AttentionSpec) and len(
                    kv_cache_config.kv_cache_groups) > 1:
                raise NotImplementedError(
                    "Only AttentionSpec and MambaSpec are supported in KV cache groups > 1."
                )

        block_sizes = [
            group.kv_cache_spec.block_size
            for group in kv_cache_config.kv_cache_groups
        ]

        self.may_reinitialize_input_batch(kv_cache_config, block_sizes)

        # Populate self.attn_groups directly with a single shared TPU builder
        # per group; bypasses parent's initialize_attn_backend (per-backend
        # get_builder_cls dispatch, cudagraph-mode resolution, cp compat
        # checks) — all GPU-relevant and unused on TPU.
        self.attn_groups = []
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            builder = AttentionMetadataBuilder(group.kv_cache_spec,
                                               group.layer_names,
                                               self.vllm_config,
                                               self.device,
                                               runner=self,
                                               kv_cache_group_id=gid)
            self.attn_groups.append([
                AttentionGroup(
                    backend=None,
                    layer_names=list(group.layer_names),
                    kv_cache_spec=group.kv_cache_spec,
                    kv_cache_group_id=gid,
                    metadata_builders=[builder],
                )
            ])

        # Verify dtype compatibility between block_table_cpu and every
        # per-group block table that downstream code may index.
        for group_id in range(len(kv_cache_config.kv_cache_groups)):
            assert (self.block_table_cpu.dtype == self.input_batch.
                    block_table[group_id].get_cpu_tensor().dtype)

        layer_name_to_spec = {}
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.shared_kv_cache_layers:
                    continue
                if hasattr(group.kv_cache_spec, "kv_cache_specs"):
                    layer_name_to_spec[
                        layer_name] = group.kv_cache_spec.kv_cache_specs[
                            layer_name]
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
            is_cache_for_ds_v4(module) for module in self.vllm_config.
            compilation_config.static_forward_context.values())

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
        self._kv_cache_bundle = None
        self._kv_cache_bundle_layer_index = {}
        bundle_view_by_name: dict[str, torch.Tensor] = {}
        use_block_major = bool(envs.VLLM_TPU_BLOCK_MAJOR_KV)
        if use_block_major:
            unsupported = None
            if self.use_spmd:
                unsupported = "SPMD"
            elif kv_cache_config.has_mamba_layers:
                unsupported = "hybrid attention+mamba models"
            elif _is_ds_v4:
                unsupported = "the DSv4 packed KV layout"
            elif self.parallel_config.prefill_context_parallel_size > 1:
                unsupported = "prefill context parallelism"
            elif self.parallel_config.pipeline_parallel_size > 1:
                # Per-stage bundles require per-stage offload namespace isolation.
                unsupported = "pipeline parallelism"
            elif self.speculative_config is not None:
                # Speculative decode propose paths execute outside the bundle wrapper context.
                unsupported = "speculative decoding"
            elif self.model_config.get_head_size() == 64:
                # Bundled RPA kernel is not supported for head_size=64 architectures.
                unsupported = "head_size=64 models (no bundled hd64 kernel)"
            if unsupported is not None:
                raise NotImplementedError(
                    f"VLLM_TPU_BLOCK_MAJOR_KV=1: {unsupported} is not "
                    "supported by the block-major KV bundle")
            attn_specs = [
                s for s in layer_name_to_spec.values()
                if isinstance(s, AttentionSpec)
            ]
            attn_keys = {(s.block_size, s.num_kv_heads, s.head_size, s.dtype)
                         for s in attn_specs}
            if len(attn_keys) != 1:
                raise NotImplementedError(
                    "VLLM_TPU_BLOCK_MAJOR_KV=1: heterogeneous attention "
                    f"specs {attn_keys} cannot share one bundle")
            for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
                shared_by = kv_cache_tensor.shared_by
                if not all(
                        isinstance(layer_name_to_spec[n], AttentionSpec)
                        for n in shared_by):
                    raise NotImplementedError(
                        "VLLM_TPU_BLOCK_MAJOR_KV=1: kv_cache_tensor "
                        f"shared by non-attention layers {shared_by}")

            # Allocate via vLLM's uniform cross-layer KV machinery. Given the Pallas backend's
            # stride order (num_blocks outermost, num_layers second), the returned cross-layer
            # tensor is identical to the canonical block-major bundle.
            # Validate uniform cross-layer KV requirements (single uniform attention group,
            # cross-layer connector preference, and no per-token-head KV quantization).
            if not KVConnectorModelRunnerMixin.use_uniform_kv_cache(
                    self.attn_groups):
                raise NotImplementedError(
                    "VLLM_TPU_BLOCK_MAJOR_KV=1: "
                    "KVConnectorModelRunnerMixin.use_uniform_kv_cache "
                    "rejected this configuration (needs a single uniform "
                    "attention group, a connector preferring cross-layer "
                    "blocks, and no per-token-head KV quantization)")
            attn_group = self.attn_groups[0][0]
            # Bind PallasAttentionBackend explicitly for uniform KV cache allocation.
            attn_group.backend = PallasAttentionBackend
            sample_spec = next(iter(attn_specs))
            # Pass resolved KV datatype (e.g. torch.bfloat16) to ensure exact memory tiling.
            uniform_views, bundle, _ = (
                KVConnectorModelRunnerMixin.allocate_uniform_kv_caches(
                    kv_cache_config,
                    self.attn_groups,
                    self.kv_cache_dtype,
                    self.device,
                    kernel_block_sizes=[sample_spec.block_size],
                ))
            # Bundle shape: (num_blocks, num_layers, block_size, K2/p, p, H).
            # The bundled RPA kernel executes directly on the top-level bundle to preserve JAX buffer donation.
            # Bound per-layer strided views are retained for vLLM metadata probing and storage deduplication.
            logger.info(
                "Block-major KV cache: allocated bundle of shape %s "
                "dtype=%s via allocate_uniform_kv_caches; attention "
                "routes through the bundled RPA kernel with dynamic "
                "layer_idx.", tuple(bundle.shape), bundle.dtype)
            self._kv_cache_bundle = bundle
            # Synchronize HBM allocation immediately to prevent deadlocks in Raiden's KVCacheManager
            # during connector registration prior to forward execution.
            synchronize_tensors(self._kv_cache_bundle, wait=True)
            # Map layer names to bundle layer dimension indices matching the layout contract.
            for idx, kv_cache_tensor in enumerate(
                    kv_cache_config.kv_cache_tensors):
                for name in kv_cache_tensor.shared_by:
                    self._kv_cache_bundle_layer_index[name] = idx
                    bundle_view_by_name[name] = uniform_views[name]

        if _is_ds_v4:
            self._initialize_ds_v4_kv_cache(kv_cache_config, _per_layer_spec,
                                            kv_caches,
                                            kv_cache_config.num_blocks)

        for kv_cache_tensor in ([] if _is_ds_v4 else
                                kv_cache_config.kv_cache_tensors):
            # If the KV cache tensor is shared by multiple layers, then we
            # duplicate cache for each layer and `num_blocks` is calculated
            # based on the total size of the shared cache.
            # Otherwise, `num_blocks` is calculated based on the size of the
            # single layer's KV cache spec.
            tensor_size = kv_cache_tensor.size
            shared_by = kv_cache_tensor.shared_by
            if len(shared_by) > 1:
                total_group_page_size = 0
                for name in shared_by:
                    spec = layer_name_to_spec[name]
                    # Use the per-layer *TPU-actual* per-block bytes so the
                    # sum equals the `page_size_padded` that
                    # `update_mamba_page_size_padded` installed on every
                    # spec (== attn_page + N × mamba_unpadded). For
                    # attention, the TPU-actual size includes dtype-
                    # specific packing (e.g., fp8 KV packs 4 elements per
                    # 32-bit word) which `spec.real_page_size_bytes`
                    # doesn't account for — on fp8 models they differ by
                    # 2×, which would break the num_blocks match here.
                    if isinstance(spec, MambaSpec):
                        total_group_page_size += dataclasses.replace(
                            spec, page_size_padded=None).page_size_bytes
                    elif isinstance(spec, AttentionSpec):
                        total_group_page_size += PallasAttentionBackend.get_kv_cache_page_size_bytes(
                            spec.block_size,
                            spec.num_kv_heads,
                            spec.head_size,
                            spec.dtype,
                        )
                    else:
                        raise NotImplementedError
                num_blocks = tensor_size // total_group_page_size
            else:
                num_blocks = kv_cache_config.num_blocks

            for layer_name in shared_by:
                kv_cache_spec = _per_layer_spec(layer_name)

                if isinstance(kv_cache_spec, MambaSpec):
                    # Compact-mamba: allocate only `_mamba_num_blocks`
                    # recurrent slots (= max_num_reqs + 1) when the override
                    # succeeded; otherwise fall back to the uniform
                    # `num_blocks`. Attention layers always keep `num_blocks`.
                    mamba_num_blocks = (self._mamba_num_blocks
                                        if self._mamba_num_blocks is not None
                                        else num_blocks)
                    allocated_mamba_num_blocks = mamba_num_blocks
                    mamba_states = []
                    for _, (shape, dtype) in enumerate(
                            zip(kv_cache_spec.shapes, kv_cache_spec.dtypes)):
                        cache_shape = (mamba_num_blocks, *shape)
                        mamba_states.append(
                            torch.zeros(cache_shape,
                                        dtype=dtype).to(self.device))
                    kv_caches[layer_name] = tuple(mamba_states)
                elif isinstance(kv_cache_spec, MLAAttentionSpec):
                    attn_module = (self.vllm_config.compilation_config.
                                   static_forward_context.get(layer_name))
                    if getattr(attn_module, "use_sparse", False):
                        # Split (nope, rope) cache; each half's shape and dtype
                        # come from its own layout spec.
                        specs = (
                            PallasMLAttentionBackend.get_sparse_kv_cache_specs(
                                num_blocks,
                                kv_cache_spec.block_size,
                                kv_cache_spec.head_size,
                                kv_cache_spec.dtype,
                            ))
                        attn_module.mla_kv_spec = specs
                        kv_caches[layer_name] = tuple(
                            torch.zeros(spec.shape, dtype=spec.dtype).to(
                                self.device) for spec in specs)
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
                    tpu_kv_cache = torch.zeros(kv_cache_shape,
                                               dtype=dtype).to(self.device)

                    kv_caches[layer_name] = tpu_kv_cache
                elif isinstance(kv_cache_spec, AttentionSpec):
                    if self.use_spmd:
                        num_kv_heads = kv_cache_spec.num_kv_heads
                        parallel_config = self.parallel_config
                        tp_size = parallel_config.tensor_parallel_size
                        # TODO: Handle kv cache duplication under SPMD mode.
                        assert num_kv_heads % tp_size == 0, (
                            f"num_kv_heads {num_kv_heads} must be divisible by "
                            f"tp_size {tp_size} under SPMD mode")
                    if use_block_major and layer_name in bundle_view_by_name:
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
                        tpu_kv_cache = torch.zeros(kv_cache_shape,
                                                   dtype=dtype).to(self.device)
                        kv_caches[layer_name] = tpu_kv_cache
                else:
                    raise NotImplementedError

        self._add_shared_kv_cache_aliases(kv_caches)

        # Mark KV cache buffers as donation candidates outside torch.compile
        # regions to avoid Dynamo tracing through pybind calls.
        # TODO(geyuhao): Comment out for now as TorchTPU does not support this right now
        # for kv_cache in kv_caches.values():
        #     pallas.set_buffer_donor_(kv_cache, True)

        # Reset kv_caches list (tpu_bind_kv_cache expects empty list)
        self.kv_caches = []

        # Use tpu_bind_kv_cache to bind KV caches to attention layers using layer names
        # This is the native vLLM pattern and avoids the 'layer_id' attribute error
        utils.tpu_bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches,
        )

        # Pre-build and cache bundled RPA kernels across all attention layers before torch.compile tracing.
        if self._kv_cache_bundle is not None:
            from vllm_torchtpu.layers.vllm.attention import \
                PallasAttentionBackendImpl
            layers = get_layers_from_vllm_config(self.vllm_config, Attention)
            # Only enter wrapper context when eligible attention layers are present.
            eligible = [
                (name, attn_layer) for name, attn_layer in layers.items()
                if name in self._kv_cache_bundle_layer_index
                and isinstance(attn_layer.impl, PallasAttentionBackendImpl)
            ]
            if eligible:
                with set_vllm_model_wrapper_context(
                        mesh=self.mesh,
                        kv_cache_bundle=self._kv_cache_bundle,
                ):
                    for name, attn_layer in eligible:
                        attn_layer.impl.setup_bundled(
                            self._kv_cache_bundle_layer_index[name],
                            self._kv_cache_bundle.device,
                            attn_layer,
                        )

        if self.use_spmd:
            # Shard KV Cache
            for cache in self.kv_caches:
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
                kv_connector.register_runner(self)

        # Initialize the compact-mamba slot allocator now that the true mamba
        # block count is known. When compact sizing was skipped, mamba shares
        # the attention `num_blocks`, so the pool spans that range instead.
        if (allocated_mamba_num_blocks is not None
                and not self._uniform_mamba_layout):
            self._init_mamba_slot_pool(allocated_mamba_num_blocks)

        # Compact pool counterpart of the read-offset buffer allocated in
        # `initialize_kv_cache`, indexed over the mamba slot pool rather
        # than the unified pool's blocks: here checkpoint `offset` of a
        # request is the slot `base_slot + offset`.
        if (allocated_mamba_num_blocks is not None
                and self.speculative_config is not None):
            if self._uniform_mamba_layout:
                raise NotImplementedError(
                    "Speculative decoding with mamba layers requires the "
                    "compact mamba slot layout (unsupported with "
                    "kv_transfer_config / uniform mamba layout).")
            self.mamba_slot_read_offsets = torch.zeros(
                allocated_mamba_num_blocks, dtype=torch.int32).to(self.device)

        # Precompile after KV cache allocation so XLA's buffer assignment sees
        # the same HBM pressure as runtime.
        # Rebuild custom attention ops here too: only now are block sizes final.
        self._initialize_attention_kernels(force=True)

        if not self.enforce_eager:
            self._precompile_substitute_placeholder_token()
