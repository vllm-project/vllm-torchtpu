# Copyright 2025 Google LLC
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
import importlib
import os
import time
from collections.abc import Callable, Mapping, Sequence
from functools import wraps
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import jax
import torch
from jax._src import dtypes
from jax.sharding import Mesh
from vllm import utils

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

GBYTES = 1024 * 1024 * 1024
TPU_HEAD_SIZE_ALIGNMENT = 128
TPU_SECOND_LAST_MINOR = 8

_megacore = False
logger = init_logger(__name__)


def align_to(unpadded_dim, pad_multiple):
    return (unpadded_dim + pad_multiple - 1) // pad_multiple * pad_multiple


def enable_megacore() -> None:
    global _megacore
    _megacore = True


def get_megacore() -> bool:
    return _megacore


def largest_divisor(x: int, cap: int) -> int:
    """Largest divisor of `x` that is <= `cap`."""
    for candidate in range(min(x, cap), 0, -1):
        if x % candidate == 0:
            return candidate
    return 1


def get_num_kv_heads_by_tp(num_kv_heads: int, tp_size: int) -> int:
    if tp_size <= num_kv_heads:
        assert num_kv_heads % tp_size == 0
        return num_kv_heads
    else:
        assert tp_size % num_kv_heads == 0
        return tp_size


def get_dp_size(parallel_config) -> int:
    return (
        int(os.environ.get("TORCH_TPU_DP_SIZE", "0"))
        or parallel_config.data_parallel_size
    )


def min_across_ep_dp(parallel_config, value: int) -> int:
    """Return the minimum of `value` across the DP engines coupled by EP.

    EP-coupled engines must compile identical programs, so a value baked into
    them (such as the KV cache size) has to agree; uncoupled engines keep theirs.
    """
    if get_dp_size(parallel_config) == 1 or not parallel_config.enable_expert_parallel:
        return value
    from vllm.distributed.parallel_state import get_dp_group

    t = torch.tensor([value], dtype=torch.int64, device="cpu")
    torch.distributed.all_reduce(
        t, op=torch.distributed.ReduceOp.MIN, group=get_dp_group().cpu_group
    )
    return int(t.item())


def hbm_usage_bytes(devices: Any) -> list[tuple[int, int]]:
    usage = []
    multihost_backend = envs.TPU_MULTIHOST_BACKEND
    if multihost_backend == "ray":
        # MemoryStats is only supported for addressable PjRt devices.
        # Assume all the devices have similar memory usage for now.
        # TODO(ranlihao): find a proper way to get the memory usage of each device.
        for device in devices:
            try:
                hbm_used = device.memory_stats()["bytes_in_use"]
                hbm_limit = device.memory_stats()["bytes_limit"]
                logger.info(
                    "Get memory stats for device %s. "
                    "Assuming all devices have the same usage.",
                    device,
                )
                usage.extend([(hbm_used, hbm_limit)] * len(devices))
                break
            except Exception as e:
                logger.warning(
                    "Failed to get memory stats for device %s: %s. ", device, e
                )
    else:
        for device in devices:
            hbm_used = device.memory_stats()["bytes_in_use"]
            hbm_limit = device.memory_stats()["bytes_limit"]
            usage.append((hbm_used, hbm_limit))

    return usage


def get_device_name(num_devices: int | None = None):
    kind = jax.devices()[0].device_kind
    if "TPU" not in kind:
        raise RuntimeError("Expected TPU devices")
    suffix = ""
    if kind.endswith(" lite"):
        kind = kind[: -len(" lite")]
        suffix = "e"
    elif kind.endswith("e"):
        kind = kind[:-1]
        suffix = "e"
    elif kind.endswith("p"):
        kind = kind[:-1]
        suffix = "p"
    elif kind.endswith("i"):
        kind = kind[:-1]
        suffix = "i"

    if kind.startswith("TPU7"):
        kind = "TPU v7"
    elif kind.startswith("TPU8"):
        kind = "TPU v8"
    assert kind[:-1] == "TPU v", kind
    kind += suffix
    if num_devices is not None:
        kind += f"-{num_devices}"
    return kind


def get_mesh_shape_product(
    mesh: Mesh,
    axes: str | list[str] | None,
) -> int:
    """
    Get the product of mesh dimensions for one or more axes.

    Examples:
        # Single axis (defaults to 1 if not present)
        get_mesh_shape_product(mesh, "model")

        # Multiple axes - computes product of their sizes
        get_mesh_shape_product(mesh, ["model", "attn_dp"])

        # None means no sharding on this dimension
        get_mesh_shape_product(mesh, None)  # returns 1
    """
    if axes is None:
        return 1

    if isinstance(axes, str):
        axes = [axes]

    product = 1
    for axis in axes:
        product *= mesh.shape.get(axis, 1)

    return product


def hbm_usage_gb(devices: Any) -> list[tuple[float, float]]:
    usage = hbm_usage_bytes(devices)
    usage = [
        (round(used / GBYTES, 2), round(limit / GBYTES, 2)) for used, limit in usage
    ]
    return usage


class HbmBudget(NamedTuple):
    """HBM accounting for the KV-cache budget, summed across devices.

    All fields are bytes:
      total_limit: HBM the device(s) expose.
      total_used:  HBM already resident (weights, compiled graphs, ...).
      cap:         total_limit * gpu_memory_utilization.
      available:   cap - total_used - headroom; the budget for the KV cache.
    """

    total_limit: int
    total_used: int
    cap: int
    available: int


def compute_hbm_budget(devices: Any, gpu_memory_utilization: float) -> HbmBudget:
    """Compute the HBM budget available for the KV cache."""
    total_limit = total_used = 0
    for device in devices:
        free_memory, limit_memory = torch.accelerator.get_memory_info(device)
        total_used += limit_memory - free_memory
        total_limit += limit_memory
    cap = int(total_limit * gpu_memory_utilization)
    available = cap - total_used

    return HbmBudget(
        total_limit=total_limit, total_used=total_used, cap=cap, available=available
    )


def estimate_kv_connector_hbm_reserve(vllm_config: Any) -> int:
    """Return HBM allocated after profiling by the configured KV connector.

    Offloading specs can advertise an estimate_hbm_reserve_bytes classmethod
    through the same metadata used by vLLM's spec factory. Keep this lookup
    shared so worker budgeting and runner block-count overrides cannot
    disagree about the available KV-cache memory.
    """
    kv_tc = vllm_config.kv_transfer_config
    if kv_tc is None:
        return 0
    extra = kv_tc.kv_connector_extra_config or {}
    spec_name = extra.get("spec_name")
    spec_module_path = extra.get("spec_module_path")
    if not spec_name or not spec_module_path:
        return 0
    try:
        spec_module = importlib.import_module(spec_module_path)
        spec_cls = getattr(spec_module, spec_name)
    except (ImportError, AttributeError):
        return 0
    estimator = getattr(spec_cls, "estimate_hbm_reserve_bytes", None)
    if estimator is None:
        return 0
    return int(estimator(vllm_config))


def get_padded_head_dim(head_dim: int) -> int:
    """Pads head_dim up to the nearest multiple of 128 for kernel performance."""
    # When head_dim == 64, we use kernel specificly optimized for it which does
    # not require any padding.
    if head_dim == 64:
        return 64
    return (head_dim + 127) // 128 * 128


def get_padded_num_heads(num_heads: int, sharding_size: int) -> int:
    if num_heads >= sharding_size:
        assert num_heads % sharding_size == 0
    else:
        assert sharding_size % num_heads == 0
        num_heads = sharding_size
    return num_heads


def get_dtype_packing(dtype):
    bits = dtypes.itemsize_bits(dtype)
    return 32 // bits


def get_hash_fn_by_name(hash_fn_name: str) -> Callable[[Any], bytes | int]:
    """
    A wrapper function of vllm.utils.hashing.get_hash_fn_by_name to support builtin
    """
    if hash_fn_name == "builtin":
        return hash
    return utils.hashing.get_hash_fn_by_name(hash_fn_name)


def time_function(func):
    """
    A decorator to measure the execution time of a function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        logger.debug(
            "Function '%s' executed in %.4f seconds.", func.__name__, execution_time
        )
        return result

    return wrapper


RawCachePool = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


class NativeKVCacheMaterialization(NamedTuple):
    """Materialized KV cache buffers and pool descriptors for native TPU layers.

    Unlike `kv_cache_materializer.MaterializedKVCache` (which tracks unified
    single-tensor byte pools and `BlockMajorPoolLayout`), this tuple carries
    typed `RawCachePool` entries (including split-page `(nope, rope)` tuples),
    per-layer axis-order tuples, and optional companion RoPE pools consumed by
    `tpu_bind_kv_cache`.
    """

    kv_caches: dict[str, torch.Tensor]
    raw_tensors: list[torch.Tensor]
    native_raw_cache: Mapping[str, RawCachePool] | None = None
    native_cache_layouts: Mapping[str, tuple[Any, ...]] | None = None
    rope_pools: Mapping[str, RawCachePool] | None = None


class NativeKVCacheLayer:
    """Protocol mixin for layers that materialize native TPU KV pools.

    All KV-bearing layers in a model using `NativeKVCacheLayer` must inherit
    from this mixin. During worker memory profiling and runner KV cache
    initialization, the first `NativeKVCacheLayer` encountered in
    `compilation_config.static_forward_context` iteration order is invoked once
    on behalf of the whole model:
    - `reserve_companion_memory` adjusts the available HBM budget before block
      count derivation (e.g. for companion pools not sized by `KVCacheSpec`).
    - `materialize_kv_caches` allocates the backing pools for all native layers
      and returns a `NativeKVCacheMaterialization` whose `kv_caches` mapping
      covers every layer name in `kv_cache_config`, and whose optional
      `native_raw_cache` and `native_cache_layouts` mappings (when provided)
      contain an entry for every bound `NativeKVCacheLayer` name (`rope_pools`
      maps only layers that own a companion RoPE pool).
    """

    def bind_kv_cache(
        self,
        cache: Any,
        *,
        raw_cache: Any = None,
        layout: Any = None,
        rope_pool: Any = None,
    ) -> None:
        """Bind materialized KV cache views and raw pool descriptors to this layer.

        Args:
            cache: Primary per-layer KV cache tensor or sequence of views.
            raw_cache: Optional `RawCachePool` backing tensor(s) for custom ops.
            layout: Optional axis-order descriptor tuple for `raw_cache`.
            rope_pool: Optional companion RoPE `RawCachePool` for split-page MLA.
        """
        self.kv_cache = cache
        self._tpu_raw_cache = raw_cache
        self._tpu_cache_layout = layout
        self._tpu_rope_pool = rope_pool

    def get_kv_cache_spec(self, vllm_config: "VllmConfig") -> "KVCacheSpec | None":
        """Return the vLLM `KVCacheSpec` for this layer, or `None` if uncached."""
        raise NotImplementedError

    def materialize_kv_caches(
        self,
        kv_cache_config: "KVCacheConfig",
        device: torch.device,
        layout: Any,
    ) -> NativeKVCacheMaterialization:
        """Allocate backing KV cache pools for all native layers in the model."""
        raise NotImplementedError

    def reserve_companion_memory(
        self,
        vllm_config: "VllmConfig",
        available: int,
        *,
        auto: bool = False,
    ) -> int:
        """Adjust available HBM bytes (`available`) for companion cache pools."""
        return available

    def zero_blocks(self, block_ids: Sequence[int]) -> None:
        """Zero newly claimed logical KV cache block IDs across backing pools."""
        del block_ids


class RpaSmemPageTableModel:
    """Mixin for models that configure RPA SMEM page-table request caps.

    Attributes:
        uses_rpa_smem_page_table: Whether `TPUModelRunner` should cap
            `num_reqs_max_model_len` and `num_reqs_most_model_len` by
            `PallasAttentionBackend.get_max_num_seqs_for_pages` (`False` when
            attention gathers KV pages directly via VMEM/SparseCore without an
            SMEM `[max_num_seqs, pages_per_seq]` page table).
    """

    uses_rpa_smem_page_table: bool = True


class TokenLookbackModel:
    """Mixin for models with request-level host token lookback.

    Attributes:
        token_lookback_depth: Number of preceding token IDs (`> 0`) gathered per
            request into `lookback_token_ids` and forwarded to `model(...)` each
            step (`0` disables lookback buffer allocation).
    """

    token_lookback_depth: int = 0


def tpu_bind_kv_cache(
    kv_caches: Mapping[str, torch.Tensor | Sequence[torch.Tensor]],
    forward_context: dict[str, Any],
    runner_kv_caches: list[torch.Tensor],
    num_attn_module: int = 1,
    native_raw_cache: Mapping[str, RawCachePool] | None = None,
    native_cache_layouts: Mapping[str, tuple[Any, ...]] | None = None,
    native_rope_pools: Mapping[str, RawCachePool] | None = None,
) -> None:
    """Bind materialized `kv_caches` to `runner_kv_caches` and `forward_context`.

    For layers implementing `NativeKVCacheLayer`, also forwards the per-layer
    `native_raw_cache`, `native_cache_layouts`, and optional `native_rope_pools`
    entries via `layer.bind_kv_cache(...)`.
    """
    assert len(runner_kv_caches) == 0

    from collections import defaultdict

    from vllm.v1.worker.utils import extract_layer_index

    index2name = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        for layer_name in sorted(index2name[layer_index]):
            cache = kv_caches[layer_name]
            if isinstance(cache, (list, tuple)):
                runner_kv_caches.extend(cache)
            else:
                runner_kv_caches.append(cast(torch.Tensor, cache))

    for layer_name, kv_cache in kv_caches.items():
        layer = forward_context[layer_name]
        if isinstance(layer, NativeKVCacheLayer):
            if native_raw_cache is not None:
                assert layer_name in native_raw_cache, (
                    f"Missing native_raw_cache entry for {layer_name}"
                )
                raw_pool = native_raw_cache[layer_name]
            else:
                raw_pool = None
            if native_cache_layouts is not None:
                assert layer_name in native_cache_layouts, (
                    f"Missing native_cache_layouts entry for {layer_name}"
                )
                layout = native_cache_layouts[layer_name]
            else:
                layout = None
            rope_pool = (
                native_rope_pools.get(layer_name)
                if native_rope_pools is not None
                else None
            )
            layer.bind_kv_cache(
                kv_cache,
                raw_cache=raw_pool,
                layout=layout,
                rope_pool=rope_pool,
            )
        else:
            layer.kv_cache = kv_cache


def synchronize_device() -> None:
    """Synchronize the TPU hardware execution stream (Device Stream Barrier).

    Differences vs. `synchronize_tensors(tensors=...)`:
    - `synchronize_device()` is equivalent to `torch.cuda.synchronize()`. It waits for
      enqueued hardware stream operations to finish on PJRT, but does NOT trigger
      deferred graph compilation or materialization for uninstantiated
      PyTorch Eager tensors.
    - Use this for stream barriers, Dynamo resets, or device-level synchronization after
      a computation.
    """
    torch.accelerator.synchronize()


def synchronize_tensors(
    tensors: torch.Tensor | Sequence[torch.Tensor] | None = None,
    wait: bool = True,
) -> None:
    """Synchronize TPU execution and force materialization of deferred tensors
    (Graph Sync).

    Differences vs. `synchronize_device()`:
    - `synchronize_tensors(tensors=...)` forces PyTorch TPU Eager mode to traverse the
      deferred tensor graph producing `tensors`, compile and enqueue the graph to PJRT,
      and materialize the resulting buffers in device memory.
    - `synchronize_device()` only waits on already-enqueued PJRT hardware commands
      without building/compiling pending deferred graphs.
    - Use `synchronize_tensors(tensors=...)` when only a subset of active tensors should
      be synchronized (e.g., weight materialization across multiple tensors). Prefer
      `synchronize_device()` when synchronizing immediately after a computation or
      across the entire device.
    """
    from torch_tpu._internal import sync

    sync.synchronize(tensors=tensors, wait=wait)


def fresh_buffer(t: torch.Tensor) -> torch.Tensor:
    """Return a copy of `t` backed by a fresh device buffer of its own shape.

    torch_tpu hands a compiled executable the buffer behind each argument. If a
    persistent tensor's storage is a buffer of a different shape (e.g. it is a
    view, or was filled by `copy_` from a view), torch_tpu re-materializes it as
    a standalone `tt_jit_as_strided` program before every forward. Calling this
    once at load time removes that per-step program.

    `torch.empty(...).copy_(t)` is not enough: torch_tpu may satisfy a
    same-shape, same-dtype `copy_` by aliasing the destination's storage to
    `t`'s base buffer, which keeps the shape mismatch. On TPU the `+ 0` forces
    a real op whose output buffer has `t`'s logical shape. Other devices get a
    plain contiguous clone (CPU has no `add` kernel for float8 dtypes).
    """
    out = torch.clone(t, memory_format=torch.contiguous_format)
    if out.device.type == "tpu":
        out = out + 0
    return out
