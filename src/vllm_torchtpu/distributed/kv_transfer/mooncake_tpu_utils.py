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
"""Shared TorchTPU <-> Mooncake TENT buffer bridge and transfer utilities.

Both `TPUMooncakeOffloadingConnector` (MooncakeDistributedStore) and
`TPUMooncakeConnector` (direct P2P TransferEngine) register TorchTPU unified
block pool (UBP) storages with Mooncake TENT through `mooncake.tpu` and
`tpu_sync.solutions.mooncake.torch.mooncake_provider`.

Key invariants enforced here:
- Every exported tensor is a contiguous `int8` view starting at storage offset 0
  and spanning the full underlying storage (required by `mooncake_provider`).
- Each logical block page is 32-byte aligned (required by TPU DMA).
- `mooncake.tpu.prepare(...)` runs BEFORE initializing `TransferEngine` or
  `MooncakeDistributedStore`, because TENT reads TPU topology once at engine
  construction time.
- Consecutive `(remote_block_id, local_block_id)` pairs are coalesced into
  contiguous spans to avoid small-transfer overhead on TPU DMA and HP-TCP.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# TPU DMA requires 32-byte-aligned device offsets (see tpu-sync mooncake_provider.h).
TPU_DMA_ALIGN_BYTES = 32


@dataclass(frozen=True)
class TpuKvPool:
    """One deduplicated TPU KV pool storage as a block-indexed `int8` byte view."""

    tensor: torch.Tensor
    page_size_bytes: int


def canonicalize_tpu_kv_pools(
    kv_caches: Mapping[str, torch.Tensor | Sequence[torch.Tensor]],
    num_blocks: int,
    *,
    align_bytes: int = TPU_DMA_ALIGN_BYTES,
) -> list[TpuKvPool]:
    """Deduplicate TPU KV cache storages into zero-copy `(num_blocks, page_bytes)`
    `int8` views starting at storage offset 0.

    Supports both single-tensor attention layers and list/tuple state views
    (hybrid Mamba / DSv4 layouts) sharing unified block pool storages.
    """
    if num_blocks <= 0:
        raise ValueError(f"num_blocks must be positive, got {num_blocks}")
    if align_bytes <= 0:
        raise ValueError(f"align_bytes must be positive, got {align_bytes}")

    pool_storages: dict[int, tuple[torch.UntypedStorage, torch.device]] = {}
    for layer_kv_cache in kv_caches.values():
        state_tensors = (
            [layer_kv_cache]
            if isinstance(layer_kv_cache, torch.Tensor)
            else layer_kv_cache
        )
        for tensor in state_tensors:
            storage = tensor.untyped_storage()
            pool_storages.setdefault(storage.data_ptr(), (storage, tensor.device))

    if not pool_storages:
        raise ValueError("kv_caches contains no backing storages to register")

    pools: list[TpuKvPool] = []
    for storage, device in pool_storages.values():
        storage_bytes = storage.nbytes()
        if storage_bytes == 0 or storage_bytes % num_blocks != 0:
            raise ValueError(
                "KV pool storage must hold whole scheduler blocks: "
                f"storage_bytes={storage_bytes}, num_blocks={num_blocks}"
            )
        page_size_bytes = storage_bytes // num_blocks
        if page_size_bytes % align_bytes != 0:
            raise ValueError(
                "Mooncake TPU DMA requires block page sizes to be multiples of "
                f"{align_bytes} bytes, got page_size_bytes={page_size_bytes}"
            )
        int8_view = (
            torch.empty(0, dtype=torch.int8, device=device)
            .set_(storage)
            .view(num_blocks, page_size_bytes)
        )
        pools.append(TpuKvPool(tensor=int8_view, page_size_bytes=page_size_bytes))

    return pools


def coalesce_block_pairs(
    src_block_ids: Sequence[int],
    dst_block_ids: Sequence[int],
) -> list[tuple[int, int, int]]:
    """Merge consecutive `(src, dst)` block pairs into `(src_start, dst_start, count)`
    runs so Mooncake issues fewer, larger transfers."""
    if len(src_block_ids) != len(dst_block_ids):
        raise ValueError(
            "src_block_ids and dst_block_ids must have equal length, got "
            f"{len(src_block_ids)} and {len(dst_block_ids)}"
        )
    if not src_block_ids:
        return []

    runs: list[tuple[int, int, int]] = []
    run_src = int(src_block_ids[0])
    run_dst = int(dst_block_ids[0])
    run_len = 1
    for raw_src, raw_dst in zip(src_block_ids[1:], dst_block_ids[1:]):
        src_id = int(raw_src)
        dst_id = int(raw_dst)
        if src_id == run_src + run_len and dst_id == run_dst + run_len:
            run_len += 1
        else:
            runs.append((run_src, run_dst, run_len))
            run_src = src_id
            run_dst = dst_id
            run_len = 1
    runs.append((run_src, run_dst, run_len))
    return runs


def build_p2p_transfer_plan(
    local_pool_addrs: Sequence[int],
    remote_pool_addrs: Sequence[int],
    page_sizes: Sequence[int],
    local_block_ids: Sequence[int],
    remote_block_ids: Sequence[int],
) -> tuple[list[int], list[int], list[int]]:
    """Build coalesced `(local_addrs, remote_addrs, lengths)` vectors for
    `TransferEngine.batch_transfer_sync_read` across all registered KV pools."""
    if not (len(local_pool_addrs) == len(remote_pool_addrs) == len(page_sizes)):
        raise ValueError(
            "Mismatched KV pool metadata lengths: "
            f"local={len(local_pool_addrs)}, "
            f"remote={len(remote_pool_addrs)}, "
            f"page_sizes={len(page_sizes)}"
        )
    runs = coalesce_block_pairs(remote_block_ids, local_block_ids)
    local_ptrs: list[int] = []
    remote_ptrs: list[int] = []
    lengths: list[int] = []
    for local_base, remote_base, page_size in zip(
        local_pool_addrs, remote_pool_addrs, page_sizes
    ):
        for remote_start, local_start, count in runs:
            local_ptrs.append(int(local_base) + local_start * int(page_size))
            remote_ptrs.append(int(remote_base) + remote_start * int(page_size))
            lengths.append(count * int(page_size))
    return local_ptrs, remote_ptrs, lengths


def build_store_block_buffers(
    shard_addrs: Sequence[int],
    page_sizes: Sequence[int],
    block_ids: Sequence[int],
) -> tuple[list[list[int]], list[list[int]]]:
    """Build `(all_buffer_ptrs, all_sizes)` for MooncakeDistributedStore
    `batch_put_from_multi_buffers` / `batch_get_into_multi_buffers`."""
    if len(shard_addrs) != len(page_sizes):
        raise ValueError(
            "shard_addrs and page_sizes must have equal length, got "
            f"{len(shard_addrs)} and {len(page_sizes)}"
        )
    all_ptrs: list[list[int]] = []
    all_sizes: list[list[int]] = []
    int_sizes = [int(sz) for sz in page_sizes]
    int_bases = [int(addr) for addr in shard_addrs]
    for raw_bid in block_ids:
        bid = int(raw_bid)
        all_ptrs.append([base + bid * size for base, size in zip(int_bases, int_sizes)])
        all_sizes.append(list(int_sizes))
    return all_ptrs, all_sizes


def store_protocol_for_tent(tent_protocol: str) -> str:
    """Return the protocol string accepted by `MooncakeDistributedStore.setup`.

    `MooncakeDistributedStore` validates its setup protocol string against
    host-segment protocols (`"tcp"`, `"rdma"`, ...), while TENT selects
    `hp_tcp` from `MC_TENT_CONF`.
    """
    normalized = tent_protocol.strip().lower()
    if normalized == "hp_tcp":
        return "tcp"
    return normalized or "tcp"


def build_tent_tpu_config(
    protocol: str = "hp_tcp",
    connections: int = 0,
    extra_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct the JSON object for `MC_TENT_CONF` enabling TENT TPU transport."""
    proto = protocol.strip().lower() or "hp_tcp"
    transport_cfg: dict[str, Any] = {"enable": True}
    if connections > 0:
        transport_cfg["connections"] = int(connections)

    transports: dict[str, Any] = {
        "tpu": {"enable": True},
        proto: transport_cfg,
    }
    config: dict[str, Any] = {"transports": transports}
    if extra_config:
        for key, value in extra_config.items():
            if key == "transports" and isinstance(value, Mapping):
                for t_name, t_val in value.items():
                    if isinstance(t_val, Mapping) and isinstance(
                        transports.get(t_name), dict
                    ):
                        transports[t_name] = {**transports[t_name], **t_val}
                    else:
                        transports[t_name] = t_val
            else:
                config[key] = value
    return config


def configure_tent_tpu_env(
    protocol: str = "hp_tcp",
    connections: int = 0,
    extra_tent_config: Mapping[str, Any] | None = None,
) -> str:
    """Export Mooncake TENT TPU environment variables before engine/store init.

    Preserves an operator-supplied `MC_TENT_CONF` when already set.
    """
    os.environ["MC_USE_TENT"] = "1"
    os.environ["MC_STORE_MEMCPY"] = "1" if tpu_envs.TPU_MOONCAKE_STORE_MEMCPY else "0"

    existing_conf = tpu_envs.MC_TENT_CONF
    if existing_conf:
        return existing_conf

    conf_json = json.dumps(
        build_tent_tpu_config(
            protocol=protocol,
            connections=connections,
            extra_config=extra_tent_config,
        )
    )
    os.environ["MC_TENT_CONF"] = conf_json
    return conf_json


def resolve_torch_mooncake_provider() -> Any:
    """Import tpu-sync's TorchTPU buffer provider for Mooncake TENT."""
    try:
        from tpu_sync.solutions.mooncake.torch import mooncake_provider
    except ImportError as exc:
        raise ImportError(
            "Mooncake TPU connectors require tpu-sync's TorchTPU buffer "
            "provider (`tpu_sync.solutions.mooncake.torch.mooncake_provider`). "
            "Install a `tpu-sync-torch` wheel that includes the Mooncake "
            "provider."
        ) from exc
    return mooncake_provider


def _import_mooncake_tpu() -> Any:
    try:
        from mooncake import tpu as mc_tpu
    except ImportError as exc:
        raise ImportError(
            "Mooncake TPU connectors require `mooncake.tpu` (Mooncake built "
            "with `-DUSE_TENT=ON -DUSE_TPU=ON`)."
        ) from exc
    return mc_tpu


def export_tpu_kv_pools(
    pool_tensors: Sequence[torch.Tensor],
    *,
    provider: Any = None,
    mc_tpu_module: Any = None,
) -> list[Any]:
    """Export TorchTPU KV pool tensors to Mooncake's TPU PJRT adapter.

    Must be called BEFORE creating `TransferEngine` or `MooncakeDistributedStore`.
    Each pool tensor on a TorchTPU worker rank owns one local TPU shard.
    """
    mc_tpu = mc_tpu_module if mc_tpu_module is not None else _import_mooncake_tpu()
    buf_provider = (
        provider if provider is not None else resolve_torch_mooncake_provider()
    )
    per_pool_shards = mc_tpu.prepare(list(pool_tensors), provider=buf_provider)

    if len(per_pool_shards) != len(pool_tensors):
        flat = [s for group in per_pool_shards for s in group]
        if flat:
            mc_tpu.unregister(flat)
        raise RuntimeError(
            f"mooncake.tpu.prepare returned {len(per_pool_shards)} pool entries "
            f"for {len(pool_tensors)} KV pool tensors"
        )

    shards: list[Any] = []
    for idx, pool_shards in enumerate(per_pool_shards):
        if len(pool_shards) != 1:
            flat = [s for group in per_pool_shards for s in group]
            if flat:
                mc_tpu.unregister(flat)
            raise RuntimeError(
                f"Expected 1 local TPU shard for KV pool {idx}, got {len(pool_shards)}"
            )
        shards.append(pool_shards[0])
    return shards


def unregister_tpu_kv_pools(
    shards: Sequence[Any],
    *,
    mc_tpu_module: Any = None,
) -> None:
    """Unregister TPU shards from Mooncake's TPU adapter on shutdown."""
    if not shards:
        return
    mc_tpu = mc_tpu_module if mc_tpu_module is not None else _import_mooncake_tpu()
    mc_tpu.unregister(list(shards))


def enqueue_pool_writes(pool_tensors: list[torch.Tensor]) -> None:
    """Asynchronously dispatch deferred TorchTPU writes to KV pool buffers."""
    from torch_tpu._internal import sync

    sync.synchronize(pool_tensors, wait=False)


def record_fence_event() -> Any:
    """Record a TPU stream event ordered behind enqueued KV pool writes."""
    event = torch.tpu.Event()
    event.record()
    return event


def event_is_retired(event: Any) -> bool:
    """Poll fence event completion without blocking and surface deferred errors."""
    if not event.query():
        return False
    event.synchronize()
    return True


def synchronize_fence_event(event: Any) -> None:
    """Block until a TPU fence event retires and surface deferred errors."""
    event.synchronize()
