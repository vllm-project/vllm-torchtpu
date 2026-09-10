# SPDX-License-Identifier: Apache-2.0
from typing import Any, cast

import torch
from vllm.config import get_layers_from_vllm_config
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import (AttentionSpec, KVCacheConfig,
                                        KVCacheSpec, MambaSpec)
from vllm.v1.worker.utils import (AttentionGroup,
                                  add_kv_sharing_layers_to_kv_cache_groups,
                                  prepare_kernel_block_sizes)

from vllm_torchtpu import utils
from vllm_torchtpu.kv_cache_materializer import (
    build_kernel_block_size_by_group_id, format_kv_cache_layout_summary,
    materialize_kv_cache_tensors)
from vllm_torchtpu.layers.common.attention_metadata import \
    AttentionMetadataBuilder
from vllm_torchtpu.layers.vllm.attention import PallasAttentionBackend
from vllm_torchtpu.logger import init_logger
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
