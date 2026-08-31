# SPDX-License-Identifier: Apache-2.0
"""TPU implementation of vLLM's ``DecodeBenchConnector``.

The public connector name intentionally remains the upstream name. The TPU
platform supplies this module path to vLLM's connector factory, so existing
command lines work unchanged while the implementation here can use TPU-native,
donated-buffer fills and the runner's compact recurrent-state slots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import \
    DecodeBenchConnector as VllmDecodeBenchConnector
from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import \
    DecodeBenchConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import \
    DecodeBenchConnectorScheduler as VllmDecodeBenchConnectorScheduler
from vllm.utils.math_utils import cdiv
from vllm.v1.core import kv_cache_utils
from vllm.v1.kv_cache_interface import (AttentionSpec, MambaSpec,
                                        UniformTypeKVCacheSpecs)
from vllm.v1.worker.block_table import BlockTable

from vllm_torchtpu.gdn_pool_layout import (pooled_gdn_state_dtypes,
                                           pooled_gdn_state_itemsize)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.platforms.tpu_block_size_utils import \
    get_tpu_pooled_gdn_state_layout

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)


def _read_fill_config(vllm_config: "VllmConfig") -> tuple[float, float]:
    transfer_config = vllm_config.kv_transfer_config
    if transfer_config is None:
        raise ValueError("DecodeBenchConnector requires kv_transfer_config")
    if not vllm_config.cache_config.enable_prefix_caching:
        raise ValueError(
            "TPU DecodeBenchConnector requires prefix caching; pass "
            "--enable-prefix-caching")

    try:
        fill_mean = float(
            transfer_config.get_from_extra_config("fill_mean", 0.015))
        fill_std = float(transfer_config.get_from_extra_config(
            "fill_std", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "DecodeBenchConnector fill_mean and fill_std must be finite "
            "numbers") from exc

    if not math.isfinite(fill_mean) or not math.isfinite(fill_std):
        raise ValueError(
            "DecodeBenchConnector fill_mean and fill_std must be finite")
    if fill_std < 0.0:
        raise ValueError(
            "DecodeBenchConnector fill_std must be non-negative, got "
            f"{fill_std}")
    return fill_mean, fill_std


def _fill_slots_in_place(cache: torch.Tensor, indices: torch.Tensor,
                         fill_value: torch.Tensor) -> None:
    # Import lazily: the same connector module is loaded in the API and
    # scheduler processes, neither of which should initialize the TPU/JAX
    # kernel bridge.
    from vllm_torchtpu.kernels.partial_kv_update import pallas_fill_slots_
    pallas_fill_slots_(cache, indices, fill_value)


def _normal_fill_slots_in_place(cache: torch.Tensor, indices: torch.Tensor,
                                mean_std: torch.Tensor,
                                seed: torch.Tensor) -> None:
    from vllm_torchtpu.kernels.partial_kv_update import \
        pallas_normal_fill_slots_
    pallas_normal_fill_slots_(cache, indices, mean_std, seed)


def _uses_upstream_block_fill(cache: torch.Tensor) -> bool:
    """Whether this cache has the known Kimi MLA/Pallas layout mismatch."""
    return (cache.dtype == torch.bfloat16
            and tuple(cache.shape[1:]) == (16, 1, 640))


def _upstream_fill_slots_in_place(cache: torch.Tensor, indices: torch.Tensor,
                                  *, mean: float, std: float) -> None:
    """Fill selected rows exactly like pinned vLLM's worker implementation."""
    shape = (indices.shape[0], ) + tuple(cache.shape[1:])
    if std > 0.0:
        values = torch.normal(mean=mean,
                              std=std,
                              size=shape,
                              dtype=cache.dtype,
                              device=cache.device)
    else:
        values = torch.full(shape,
                            mean,
                            dtype=cache.dtype,
                            device=cache.device)
    cache[indices] = values


def _fill_hybrid_pool_in_place(cache: torch.Tensor, indices: torch.Tensor,
                               fill_kinds: torch.Tensor,
                               fill_value: torch.Tensor, *, split: int,
                               ssm_bytes: int, ssm_itemsize: int,
                               conv_bytes: int) -> None:
    from vllm_torchtpu.kernels.partial_kv_update import \
        pallas_fill_hybrid_slots_
    pallas_fill_hybrid_slots_(cache,
                              indices,
                              fill_kinds,
                              fill_value,
                              split=split,
                              ssm_bytes=ssm_bytes,
                              ssm_itemsize=ssm_itemsize,
                              conv_bytes=conv_bytes)


def _normal_fill_hybrid_pool_in_place(cache: torch.Tensor,
                                      indices: torch.Tensor,
                                      fill_kinds: torch.Tensor,
                                      mean_std: torch.Tensor,
                                      seed: torch.Tensor, *, split: int,
                                      ssm_bytes: int, ssm_itemsize: int,
                                      conv_bytes: int) -> None:
    from vllm_torchtpu.kernels.partial_kv_update import \
        pallas_normal_fill_hybrid_slots_
    pallas_normal_fill_hybrid_slots_(cache,
                                     indices,
                                     fill_kinds,
                                     mean_std,
                                     seed,
                                     split=split,
                                     ssm_bytes=ssm_bytes,
                                     ssm_itemsize=ssm_itemsize,
                                     conv_bytes=conv_bytes)


def _synchronize_mutated(tensors: list[torch.Tensor]) -> None:
    from vllm_torchtpu.utils import synchronize_tensors
    synchronize_tensors(tensors)


@dataclass
class _UnifiedPoolFill:
    cache: torch.Tensor
    attention_blocks: set[int]
    state_blocks: set[int]
    layer_name: str
    ssm_bytes: int | None = None
    ssm_itemsize: int | None = None
    conv_bytes: int | None = None


class DecodeBenchConnector(VllmDecodeBenchConnector):
    """Pinned vLLM DecodeBenchConnector with a TPU-native worker."""

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole,
                 kv_cache_config: "KVCacheConfig"):
        # Keep the upstream facade, wire metadata, and lifecycle behavior. Only
        # the scheduler's block geometry and worker's materialization differ on
        # TPU.
        super().__init__(vllm_config, role, kv_cache_config)
        logger.info("Using TPU DecodeBenchConnector for role %s", role)
        _read_fill_config(vllm_config)
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = DecodeBenchConnectorScheduler(
                vllm_config, kv_cache_config)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = DecodeBenchConnectorWorker(
                vllm_config, kv_cache_config)

    def register_runner(self, runner: "TPUModelRunner") -> None:
        assert self.connector_worker is not None
        self.connector_worker.register_runner(runner)


class DecodeBenchConnectorScheduler(VllmDecodeBenchConnectorScheduler):
    """Upstream scheduler with prefix-aware TPU block geometry."""

    def __init__(self, vllm_config: "VllmConfig",
                 kv_cache_config: "KVCacheConfig"):
        super().__init__(vllm_config)
        self.kv_cache_config = kv_cache_config

    @staticmethod
    def _group_specs(spec: Any) -> tuple[Any, ...]:
        if isinstance(spec, UniformTypeKVCacheSpecs):
            return tuple(spec.kv_cache_specs.values())
        return (spec, )

    def _manager_block_size(self, spec: Any) -> int:
        """Token span represented by one scheduler/manager block ID."""
        leaves = self._group_specs(spec)
        parallel_config = self.vllm_config.parallel_config
        dcp = int(parallel_config.decode_context_parallel_size or 1)
        pcp = int(parallel_config.prefill_context_parallel_size or 1)
        has_attention = any(isinstance(leaf, AttentionSpec) for leaf in leaves)
        has_mamba = any(isinstance(leaf, MambaSpec) for leaf in leaves)
        cache_is_hybrid = (self.kv_cache_config.has_mamba_layers and any(
            any(
                isinstance(leaf, AttentionSpec)
                for leaf in self._group_specs(group.kv_cache_spec))
            for group in self.kv_cache_config.kv_cache_groups))

        block_size = int(spec.block_size)
        if pcp > 1 and cache_is_hybrid and (has_attention or has_mamba):
            return block_size * pcp
        if dcp > 1 and has_attention:
            return block_size * dcp
        return block_size

    def update_state_after_alloc(self, request: Any, blocks: Any,
                                 num_external_tokens: int) -> None:
        if num_external_tokens == 0:
            # A fully local hit already owns valid attention KV. Unified pools
            # can still have a divergent recurrent miss, while compact slots
            # are always request-local and freshly allocated. Preserve every
            # attention block and materialize only the final recurrent block
            # (compact workers ignore its ID and fill their request slot).
            if self.kv_cache_config.has_mamba_layers:
                block_groups = blocks.get_unhashed_block_ids_all_groups()
                groups = self.kv_cache_config.kv_cache_groups
                if len(block_groups) != len(groups):
                    raise ValueError(
                        "DecodeBenchConnector received block IDs for "
                        f"{len(block_groups)} groups, but KV cache has "
                        f"{len(groups)}")
                selected: list[list[int]] = []
                for group_blocks, group in zip(block_groups, groups):
                    specs = self._group_specs(group.kv_cache_spec)
                    if all(isinstance(spec, MambaSpec) for spec in specs):
                        selected.append(
                            [] if not group_blocks else [group_blocks[-1]])
                    else:
                        selected.append([])
                self._pending_fills[request.request_id] = (
                    tuple(selected),
                    0,
                )
                self._filled_requests.add(request.request_id)
            return

        block_groups = blocks.get_block_ids()
        specs = self.kv_cache_config.kv_cache_groups
        if len(block_groups) != len(specs):
            raise ValueError(
                "DecodeBenchConnector received block IDs for "
                f"{len(block_groups)} groups, but KV cache has {len(specs)}")

        local_prefix_tokens = (request.num_tokens - num_external_tokens - 1)
        if local_prefix_tokens < 0:
            raise ValueError(
                "DecodeBenchConnector external token count exceeds request "
                f"length: request={request.num_tokens}, "
                f"external={num_external_tokens}")
        # Resolve through the module at scheduling time. TorchTPU patches this
        # binding for hybrid PCP before EngineCore starts; importing the
        # function by value here would retain the unpatched upstream resolver.
        scheduler_block_size, _ = kv_cache_utils.resolve_kv_cache_block_sizes(
            self.kv_cache_config, self.vllm_config)
        if local_prefix_tokens % scheduler_block_size:
            raise ValueError(
                "DecodeBenchConnector requires a scheduler-block-aligned local "
                f"prefix: local={local_prefix_tokens}, "
                f"scheduler_block_size={scheduler_block_size}")

        selected: list[list[int]] = []
        for group_index, (group_blocks,
                          group) in enumerate(zip(block_groups, specs)):
            manager_block_size = self._manager_block_size(group.kv_cache_spec)
            start = local_prefix_tokens // manager_block_size
            end = cdiv(local_prefix_tokens + num_external_tokens,
                       manager_block_size)
            if len(group_blocks) < end:
                raise ValueError(
                    "DecodeBenchConnector group "
                    f"{group_index} needs blocks through index {end - 1} for "
                    f"local={local_prefix_tokens}, "
                    f"external={num_external_tokens}, got "
                    f"{len(group_blocks)} blocks")
            selected.append(list(group_blocks[start:end]))

        request_id = request.request_id
        self._pending_fills[request_id] = (tuple(selected),
                                           num_external_tokens)
        self._filled_requests.add(request_id)


class DecodeBenchConnectorWorker:
    """Materialize dummy attention KV and compact KDA recurrent state."""

    def __init__(self, vllm_config: "VllmConfig",
                 kv_cache_config: "KVCacheConfig"):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.fill_mean, self.fill_std = _read_fill_config(vllm_config)
        self.kv_caches: dict[str, Any] | None = None
        self.runner: TPUModelRunner | None = None
        self._attention_block_splits: dict[int, int] = {}
        self._pool_block_split = 1
        self._fill_values: dict[torch.device, torch.Tensor] = {}
        self._normal_params: dict[torch.device, torch.Tensor] = {}
        self._pooled_state_bytes_by_token_size: dict[int, tuple[int, int]] = {}
        self._next_random_seed: int | None = None
        if self.fill_std > 0.0:
            self._next_random_seed = int(
                torch.randint(0, 2**31 - 1, (1, ), dtype=torch.int64).item())
        self._logged_fill_summary = False

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        for layer_name, cache in kv_caches.items():
            tensors: tuple[torch.Tensor, ...] = ()
            if isinstance(cache, torch.Tensor):
                tensors = (cache, )
            elif (isinstance(cache, (tuple, list)) and cache and all(
                    isinstance(tensor, torch.Tensor) for tensor in cache)):
                tensors = tuple(cache)
            for tensor in tensors:
                if not tensor.is_floating_point():
                    raise TypeError(
                        "TPU DecodeBenchConnector cannot synthesize numeric "
                        "values for raw or integer cache encodings: layer "
                        f"{layer_name!r} has dtype {tensor.dtype}")
        self.kv_caches = kv_caches

    def register_runner(self, runner: "TPUModelRunner") -> None:
        self.runner = runner
        pool_split = getattr(runner, "_pool_block_split", 1)
        if isinstance(pool_split, int) and pool_split > 0:
            self._pool_block_split = pool_split
        kernel_sizes = getattr(runner, "_kernel_block_size_by_group_id", None)
        if not isinstance(kernel_sizes, dict):
            kernel_sizes = {}
        for group_index, group in enumerate(
                self.kv_cache_config.kv_cache_groups):
            specs = self._group_specs(group.kv_cache_spec)
            if not all(isinstance(spec, AttentionSpec) for spec in specs):
                continue
            manager_block_size = group.kv_cache_spec.block_size
            split: int | None = None
            input_batch = getattr(runner, "input_batch", None)
            block_tables = getattr(input_batch, "block_table", None)
            if block_tables is not None:
                try:
                    table_split = block_tables[group_index].blocks_per_kv_block
                except (IndexError, KeyError, TypeError):
                    table_split = None
                if isinstance(table_split, int) and table_split > 0:
                    split = table_split
            if split is None:
                kernel_block_size = kernel_sizes.get(group_index,
                                                     manager_block_size)
                if (not isinstance(kernel_block_size, int)
                        or kernel_block_size <= 0
                        or manager_block_size % kernel_block_size != 0):
                    raise ValueError(
                        "DecodeBenchConnector requires each attention kernel "
                        "block size to divide its manager block size: "
                        f"group={group_index}, manager={manager_block_size}, "
                        f"kernel={kernel_block_size}")
                split = manager_block_size // kernel_block_size
            self._attention_block_splits[group_index] = split

    @staticmethod
    def _group_specs(spec: Any) -> tuple[Any, ...]:
        if isinstance(spec, UniformTypeKVCacheSpecs):
            specs = tuple(spec.kv_cache_specs.values())
            if not specs:
                raise ValueError(
                    "DecodeBenchConnector received an empty composite KV "
                    "cache spec")
            return specs
        return (spec, )

    def _physical_attention_rows(self, group_index: int,
                                 manager_ids: set[int]) -> set[int]:
        split = self._attention_block_splits.get(group_index, 1)
        if not manager_ids:
            return set()
        manager_array = np.asarray(sorted(manager_ids), dtype=np.int64)
        offsets = np.arange(split, dtype=np.int64).reshape(1, -1)
        return set(
            BlockTable.map_to_kernel_blocks(manager_array, split,
                                            offsets).tolist())

    @staticmethod
    def _storage_key(cache: torch.Tensor) -> tuple[str, int]:
        ptr = cache.untyped_storage().data_ptr()
        # Meta/fake tensors have no backing allocation. Object identity keeps
        # their unit-test aliases distinct while real caches deduplicate by the
        # storage actually donated to XLA.
        return str(cache.device), ptr if ptr else id(cache)

    @staticmethod
    def _cache_tensors(cache: Any, *, layer_name: str,
                       recurrent: bool) -> tuple[torch.Tensor, ...]:
        if isinstance(cache, torch.Tensor) and not recurrent:
            return (cache, )
        if (isinstance(cache, (tuple, list)) and cache
                and all(isinstance(tensor, torch.Tensor) for tensor in cache)):
            return tuple(cache)
        kind = "recurrent" if recurrent else "attention"
        expected = "a non-empty tensor tuple/list" if recurrent else (
            "a tensor or a non-empty tensor tuple/list")
        raise TypeError(f"DecodeBenchConnector {kind} layer {layer_name!r} "
                        f"must use {expected}, got {type(cache).__name__}")

    def _pooled_state_bytes(self, cache: torch.Tensor) -> tuple[int, int]:
        token_bytes = math.prod(cache.shape[2:]) * cache.element_size()
        cached = self._pooled_state_bytes_by_token_size.get(token_bytes)
        if cached is not None:
            return cached
        layout = get_tpu_pooled_gdn_state_layout(self.vllm_config, token_bytes)
        if layout is None:
            raise ValueError(
                "unified DecodeBenchConnector requires a pooled GDN model")
        result = (layout.ssm_bytes, layout.conv_bytes)
        self._pooled_state_bytes_by_token_size[token_bytes] = result
        return result

    @staticmethod
    def _pooled_ssm_itemsize(specs: tuple[Any, ...]) -> int:
        itemsizes: set[int] = set()
        for spec in specs:
            dtypes = tuple(spec.dtypes)
            if len(dtypes) != 2:
                raise ValueError(
                    "unified DecodeBenchConnector requires conv and SSM "
                    f"dtypes, got {dtypes}")
            _, ssm_dtype = pooled_gdn_state_dtypes(dtypes)
            itemsizes.add(pooled_gdn_state_itemsize(ssm_dtype))
        if len(itemsizes) != 1:
            raise ValueError(
                "pooled GDN layers in one cache group have different SSM "
                f"item sizes: {sorted(itemsizes)}")
        return itemsizes.pop()

    def _fill_value_for(self, cache: torch.Tensor) -> torch.Tensor:
        value = self._fill_values.get(cache.device)
        if value is None:
            value = torch.tensor([self.fill_mean],
                                 dtype=torch.float32,
                                 device=cache.device)
            self._fill_values[cache.device] = value
        return value

    def _normal_params_for(self, cache: torch.Tensor) -> torch.Tensor:
        params = self._normal_params.get(cache.device)
        if params is None:
            params = torch.tensor([self.fill_mean, self.fill_std],
                                  dtype=torch.float32,
                                  device=cache.device)
            self._normal_params[cache.device] = params
        return params

    def _random_seed_for(self, cache: torch.Tensor) -> torch.Tensor:
        seed = self._next_random_seed
        if seed is None:
            raise RuntimeError("random seed requested for a constant fill")
        # Give every physical cache invocation an independent stream while
        # retaining torch.manual_seed control over the worker's initial seed.
        self._next_random_seed = (seed + 1) % (2**31 - 1)
        return torch.tensor([seed], dtype=torch.int32, device=cache.device)

    @staticmethod
    def _validate_slots(cache: torch.Tensor, slots: set[int],
                        layer_name: str) -> None:
        if cache.ndim < 3:
            raise ValueError(
                "TPU DecodeBenchConnector only supports cache tensors with "
                f"rank >= 3; layer {layer_name!r} has shape {tuple(cache.shape)}"
            )
        invalid = [
            slot for slot in slots if slot < 0 or slot >= cache.shape[0]
        ]
        if invalid:
            raise IndexError(
                f"DecodeBenchConnector slots {invalid[:8]} are outside layer "
                f"{layer_name!r} with {cache.shape[0]} slots")

    def start_fill_kv(self, metadata: DecodeBenchConnectorMetadata) -> None:
        if not metadata.reqs_to_fill:
            return
        if self.kv_caches is None:
            raise RuntimeError(
                "register_kv_caches() must run before DecodeBenchConnector fill"
            )

        groups = self.kv_cache_config.kv_cache_groups
        request_ids = list(metadata.reqs_to_fill)
        for request_id, (block_ids, _) in metadata.reqs_to_fill.items():
            if len(block_ids) != len(groups):
                raise ValueError(
                    f"request {request_id!r} has metadata for "
                    f"{len(block_ids)} groups, expected {len(groups)}")

        has_mamba = any(
            all(
                isinstance(spec, MambaSpec)
                for spec in self._group_specs(group.kv_cache_spec))
            for group in groups)
        if has_mamba and self.runner is None:
            raise RuntimeError(
                "register_runner() must run before hybrid decode bench fill")
        unified_layout = bool(has_mamba and self.runner is not None
                              and self.runner._unified_kv_layout)
        slots_by_request: dict[str, int] = {}
        if has_mamba and not unified_layout:
            assert self.runner is not None
            slots_by_request = self.runner.ensure_mamba_slots_for_requests(
                request_ids)

        # Aggregate aliases before invoking the donation kernel. A shared
        # storage must be donated exactly once per fill step.
        attention_fills: dict[tuple[str, int], tuple[torch.Tensor, set[int],
                                                     str]] = {}
        mamba_fills: dict[tuple[str, int], tuple[torch.Tensor, set[int],
                                                 str]] = {}
        unified_fills: dict[tuple[str, int], _UnifiedPoolFill] = {}
        attention_manager_blocks: set[int] = set()

        def unified_fill_for(cache: torch.Tensor,
                             layer_name: str) -> _UnifiedPoolFill:
            key = self._storage_key(cache)
            fill = unified_fills.get(key)
            if fill is None:
                fill = _UnifiedPoolFill(cache=cache,
                                        attention_blocks=set(),
                                        state_blocks=set(),
                                        layer_name=layer_name)
                unified_fills[key] = fill
            return fill

        for group_index, group in enumerate(groups):
            specs = self._group_specs(group.kv_cache_spec)
            if all(isinstance(spec, AttentionSpec) for spec in specs):
                manager_ids = {
                    int(block_id)
                    for block_ids, _ in metadata.reqs_to_fill.values()
                    for block_id in block_ids[group_index]
                }
                # Block zero is the shared null page and must stay deterministic.
                manager_ids.discard(0)
                attention_manager_blocks.update(manager_ids)
                for layer_name in group.layer_names:
                    cache = self.kv_caches.get(layer_name)
                    cache_tensors = self._cache_tensors(cache,
                                                        layer_name=layer_name,
                                                        recurrent=False)
                    # KVCacheConfig.num_blocks is a logical manager capacity
                    # and can differ slightly from an individual tensor under
                    # HMA padding. Expansion therefore comes from the runner's
                    # manager/kernel block-size contract, never tensor shape.
                    for cache_tensor in cache_tensors:
                        if unified_layout:
                            split = self._attention_block_splits.get(
                                group_index, 1)
                            if split != self._pool_block_split:
                                raise ValueError(
                                    "unified DecodeBenchConnector block split "
                                    "does not match the runner pool: "
                                    f"group={group_index}, attention={split}, "
                                    f"pool={self._pool_block_split}")
                            unified_fill_for(
                                cache_tensor,
                                layer_name).attention_blocks.update(
                                    manager_ids)
                            continue
                        physical_rows = self._physical_attention_rows(
                            group_index, manager_ids)
                        key = self._storage_key(cache_tensor)
                        if key in attention_fills:
                            attention_fills[key][1].update(physical_rows)
                        else:
                            attention_fills[key] = (cache_tensor,
                                                    set(physical_rows),
                                                    layer_name)
            elif all(isinstance(spec, MambaSpec) for spec in specs):
                ssm_itemsize = (self._pooled_ssm_itemsize(specs)
                                if unified_layout else None)
                if unified_layout:
                    state_ids = {
                        int(block_id)
                        for block_ids, _ in metadata.reqs_to_fill.values()
                        for block_id in block_ids[group_index]
                    }
                    state_ids.discard(0)
                else:
                    state_ids = set(slots_by_request.values())
                for layer_name in group.layer_names:
                    cache = self.kv_caches.get(layer_name)
                    state_tensors = self._cache_tensors(cache,
                                                        layer_name=layer_name,
                                                        recurrent=True)
                    for state_tensor in state_tensors:
                        if unified_layout:
                            ssm_bytes, conv_bytes = self._pooled_state_bytes(
                                state_tensor)
                            fill = unified_fill_for(state_tensor, layer_name)
                            fill.state_blocks.update(state_ids)
                            existing = (fill.ssm_bytes, fill.ssm_itemsize,
                                        fill.conv_bytes)
                            state_layout = (ssm_bytes, ssm_itemsize,
                                            conv_bytes)
                            if existing != (None, None,
                                            None) and existing != state_layout:
                                raise ValueError(
                                    "pooled GDN layers sharing raw storage have "
                                    "different state layouts: "
                                    f"{existing} vs {state_layout}")
                            fill.ssm_bytes = ssm_bytes
                            fill.ssm_itemsize = ssm_itemsize
                            fill.conv_bytes = conv_bytes
                            continue
                        key = self._storage_key(state_tensor)
                        if key in mamba_fills:
                            mamba_fills[key][1].update(state_ids)
                        else:
                            mamba_fills[key] = (state_tensor, set(state_ids),
                                                layer_name)
            else:
                raise TypeError(
                    "TPU DecodeBenchConnector does not support KV cache group "
                    f"{group_index} spec {type(group.kv_cache_spec).__name__}")

        mutated: list[torch.Tensor] = []

        def fill_selected(cache: torch.Tensor, slots: set[int],
                          layer_name: str, *, use_upstream: bool) -> None:
            if not slots:
                return
            self._validate_slots(cache, slots, layer_name)
            indices = torch.tensor(
                sorted(slots),
                dtype=(torch.long if use_upstream else torch.int32),
                device=cache.device)
            if use_upstream:
                _upstream_fill_slots_in_place(cache,
                                              indices,
                                              mean=self.fill_mean,
                                              std=self.fill_std)
            elif self.fill_std > 0.0:
                _normal_fill_slots_in_place(cache, indices,
                                            self._normal_params_for(cache),
                                            self._random_seed_for(cache))
            else:
                _fill_slots_in_place(cache, indices,
                                     self._fill_value_for(cache))
            mutated.append(cache)

        for cache, slots, layer_name in attention_fills.values():
            fill_selected(cache,
                          slots,
                          layer_name,
                          use_upstream=_uses_upstream_block_fill(cache))
        for cache, slots, layer_name in mamba_fills.values():
            fill_selected(cache, slots, layer_name, use_upstream=False)

        for fill in unified_fills.values():
            manager_ids = fill.attention_blocks | fill.state_blocks
            if not manager_ids:
                continue
            invalid = [
                block_id for block_id in manager_ids
                if block_id < 0 or (block_id + 1) *
                self._pool_block_split > fill.cache.shape[0]
            ]
            if invalid:
                raise IndexError(
                    f"DecodeBenchConnector manager blocks {invalid[:8]} are "
                    f"outside unified layer {fill.layer_name!r} with "
                    f"{fill.cache.shape[0]} physical rows and split "
                    f"{self._pool_block_split}")
            ordered = sorted(manager_ids)
            indices = torch.tensor(ordered,
                                   dtype=torch.int32,
                                   device=fill.cache.device)
            fill_kinds = torch.tensor(
                [(1 if block_id in fill.attention_blocks else 0) |
                 (2 if block_id in fill.state_blocks else 0)
                 for block_id in ordered],
                dtype=torch.int32,
                device=fill.cache.device)
            ssm_bytes = fill.ssm_bytes or 0
            ssm_itemsize = fill.ssm_itemsize or 0
            conv_bytes = fill.conv_bytes or 0
            if fill.state_blocks and not (ssm_bytes and ssm_itemsize
                                          and conv_bytes):
                raise ValueError(
                    f"unified layer {fill.layer_name!r} has state blocks but no "
                    "pooled GDN layout")
            if self.fill_std > 0.0:
                _normal_fill_hybrid_pool_in_place(
                    fill.cache,
                    indices,
                    fill_kinds,
                    self._normal_params_for(fill.cache),
                    self._random_seed_for(fill.cache),
                    split=self._pool_block_split,
                    ssm_bytes=ssm_bytes,
                    ssm_itemsize=ssm_itemsize,
                    conv_bytes=conv_bytes)
            else:
                _fill_hybrid_pool_in_place(fill.cache,
                                           indices,
                                           fill_kinds,
                                           self._fill_value_for(fill.cache),
                                           split=self._pool_block_split,
                                           ssm_bytes=ssm_bytes,
                                           ssm_itemsize=ssm_itemsize,
                                           conv_bytes=conv_bytes)
            mutated.append(fill.cache)

        if mutated:
            _synchronize_mutated(mutated)

        if not self._logged_fill_summary:
            external_tokens = sum(value[1]
                                  for value in metadata.reqs_to_fill.values())
            logger.info(
                "TPU DecodeBenchConnector filled %d external tokens, %d "
                "attention manager blocks, %d compact KDA slots, and %d "
                "unified pools", external_tokens,
                len(attention_manager_blocks),
                len(set(slots_by_request.values())), len(unified_fills))
            self._logged_fill_summary = True
