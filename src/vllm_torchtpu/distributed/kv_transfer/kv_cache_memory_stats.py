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
"""Counts HBM KV cache blocks by state for ``vllm:kv_cache_blocks_by_state``."""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.outputs import KVConnectorOutput

KV_CACHE_BLOCKS_BY_STATE_METRIC_NAME = "vllm:kv_cache_blocks_by_state"
KV_CACHE_BLOCKS_BY_STATE_DOC = (
    "Number of HBM KV cache blocks in each state, excluding the null block. "
    "States are exclusive and sum to num_gpu_blocks - 1. Sampled about every "
    "10 seconds while the engine is stepping."
)
STATS_KEY_BLOCKS_BY_STATE = "kv_cache_blocks_by_state"

# Order of the counts in a snapshot.
BLOCK_STATES = (
    "active",
    "loading_kv",
    "held_for_send",
    "cached_evictable",
    "empty_free",
)

# A full scan takes tens of ms at 200k blocks on the engine thread, so sample
# rarely.
SCAN_INTERVAL_S = 10.0


# Last gauge created per gauge class (one class per registry).
_kv_cache_blocks_by_state_gauges: dict[Any, Any] = {}


def get_or_create_kv_cache_blocks_by_state_gauge(
    gauge_cls: Any, labelnames: list[str]
) -> Any:
    """Registers `vllm:kv_cache_blocks_by_state`, with no series until sampled.

    Under `TPUMultiConnector`, the TPU and offloading connectors' Prometheus
    metrics both register this gauge in one registry, so the second one reuses
    the first. A new stat logger unregisters vLLM metrics first, so its first
    registration succeeds and replaces the cached gauge.
    """
    try:
        gauge = gauge_cls(
            name=KV_CACHE_BLOCKS_BY_STATE_METRIC_NAME,
            documentation=KV_CACHE_BLOCKS_BY_STATE_DOC,
            labelnames=[*labelnames, "state"],
            multiprocess_mode="mostrecent",
        )
        _kv_cache_blocks_by_state_gauges[gauge_cls] = gauge
    except ValueError:  # Already registered.
        gauge = _kv_cache_blocks_by_state_gauges[gauge_cls]
    return gauge


def count_blocks_by_state(
    kv_cache_manager: KVCacheManager,
    loading_req_ids: Iterable[str],
    held_req_ids: Iterable[str],
) -> list[int]:
    """Counts the blocks of the manager's pool per state, in BLOCK_STATES order.

    A block counts once. A block shared with a loading request counts as
    ``loading_kv``, then one shared with a held request as ``held_for_send``,
    even when a running request also uses it.
    """
    coordinator = kv_cache_manager.coordinator

    def block_ids(req_ids: Iterable[str]) -> set[int]:
        ids: set[int] = set()
        for req_id in req_ids:
            for blocks in coordinator.get_blocks(req_id):
                ids.update(b.block_id for b in blocks if not b.is_null)
        return ids

    loading = block_ids(loading_req_ids)
    held = block_ids(held_req_ids) - loading
    in_use = cached = empty = 0
    for block in kv_cache_manager.block_pool.blocks:
        if block.is_null:
            continue
        if block.ref_cnt > 0:
            in_use += 1
        elif block.block_hash is not None:
            cached += 1
        else:
            empty += 1
    return [in_use - len(loading) - len(held), len(loading), len(held), cached, empty]


class KVBlockStateTracker:
    """Tracks requests whose blocks wait on a connector, and samples counts.

    Connectors report scheduler hooks here. ``TPUMultiConnector`` gives all
    its TPU children one tracker, so the counts cover every child's requests
    and are sampled once per interval.
    """

    def __init__(self) -> None:
        # Requests with an external load in flight.
        self.loading_req_ids: set[str] = set()
        # Finished requests whose blocks a connector still sends from.
        self.held_req_ids: set[str] = set()
        self._kv_cache_manager: KVCacheManager | None = None
        self._next_scan_time = 0.0
        self._idle_published = False

    def bind(self, kv_cache_manager: KVCacheManager) -> None:
        self._kv_cache_manager = kv_cache_manager

    def on_alloc(self, req_id: str, num_external_tokens: int) -> None:
        if num_external_tokens > 0:
            self.loading_req_ids.add(req_id)

    def on_scheduled(self, num_scheduled_tokens: dict[str, int]) -> None:
        # A request scheduled with an external hit loads synchronously, before
        # its forward pass, so the load is done when this step's stats are
        # read. Under async scheduling that read can belong to the previous
        # step, so a synchronous load may show as `active` for one step.
        if self.loading_req_ids:
            self.loading_req_ids.difference_update(num_scheduled_tokens)

    def on_finished(self, req_id: str, delay_free: bool) -> None:
        if delay_free:
            self.held_req_ids.add(req_id)

    def on_connector_output(self, connector_output: KVConnectorOutput) -> None:
        # A request aborted while loading keeps its blocks until its load
        # reports in finished_recving.
        self.loading_req_ids.difference_update(connector_output.finished_recving or ())
        self.held_req_ids.difference_update(connector_output.finished_sending or ())

    def maybe_count(self) -> list[int] | None:
        """Counts blocks by state if a sample is due, else returns None.

        A sample is due on the first call after bind, then every
        SCAN_INTERVAL_S, and on entering idle (all usable blocks free): the
        engine stops stepping once idle, so that sample is the one left
        published.
        """
        manager = self._kv_cache_manager
        if manager is None:
            # Nothing to count before the scheduler binds its cache manager.
            return None
        pool = manager.block_pool
        # The null block is never in the free queue.
        idle = pool.get_num_free_blocks() == pool.num_gpu_blocks - 1
        if not idle:
            self._idle_published = False
        now = time.monotonic()
        if now < self._next_scan_time and (not idle or self._idle_published):
            return None
        self._next_scan_time = now + SCAN_INTERVAL_S
        self._idle_published = idle
        return count_blocks_by_state(manager, self.loading_req_ids, self.held_req_ids)


class TracksKVBlockStates:
    """A connector that reports its scheduler hooks to ``kv_block_states``."""

    kv_block_states: KVBlockStateTracker
