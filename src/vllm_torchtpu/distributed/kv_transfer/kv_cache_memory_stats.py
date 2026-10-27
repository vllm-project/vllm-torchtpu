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
"""Scheduler-side KV-cache block state scanner and Prometheus gauge."""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

from vllm.v1.request import RequestStatus

from vllm_torchtpu import envs

ALL_KV_GROUPS = "all"

BLOCK_STATES: tuple[str, ...] = (
    "active_shared",
    "active_private",
    "held_for_send",
    "loading_kv",
    "waiting_hold",
    "transient_pin",
    "orphan_suspect",
    "cached_evictable",
    "free_unhashed",
)

_STATE_PRIORITY: dict[str, int] = {name: i for i, name in enumerate(BLOCK_STATES)}

STATS_BLOCKS_BY_STATE = "kv_cache_blocks_by_state"

KV_CACHE_BLOCKS_BY_STATE_METRIC_NAME = "vllm:kv_cache_blocks_by_state"
KV_CACHE_BLOCKS_BY_STATE_DOC = (
    "KV cache blocks per kv_group and exclusive priority state"
)

_SHARED_BLOCKS_BY_STATE_GAUGES: dict[tuple[Any, tuple[str, ...]], Any] = {}


def get_or_create_kv_cache_blocks_by_state_gauge(
    gauge_cls: Any, labelnames: list[str]
) -> Any:
    """Create or retrieve the shared ``vllm:kv_cache_blocks_by_state`` gauge."""
    full_labelnames = [*labelnames, "kv_group", "state"]
    key = (gauge_cls, tuple(full_labelnames))
    gauge = _SHARED_BLOCKS_BY_STATE_GAUGES.get(key)
    if gauge is not None:
        return gauge
    try:
        try:
            gauge = gauge_cls(
                name=KV_CACHE_BLOCKS_BY_STATE_METRIC_NAME,
                documentation=KV_CACHE_BLOCKS_BY_STATE_DOC,
                labelnames=full_labelnames,
                multiprocess_mode="mostrecent",
            )
        except TypeError:
            gauge = gauge_cls(
                name=KV_CACHE_BLOCKS_BY_STATE_METRIC_NAME,
                documentation=KV_CACHE_BLOCKS_BY_STATE_DOC,
                labelnames=full_labelnames,
            )
    except ValueError:
        import prometheus_client

        registry = (getattr(gauge_cls, "keywords", None) or {}).get(
            "registry", prometheus_client.REGISTRY
        )
        existing = getattr(registry, "_names_to_collectors", {}).get(
            KV_CACHE_BLOCKS_BY_STATE_METRIC_NAME
        )
        if existing is None:
            raise
        gauge = existing
    _SHARED_BLOCKS_BY_STATE_GAUGES[key] = gauge
    return gauge


def init_kv_cache_blocks_by_state_gauge(
    gauge: Any, per_engine_labelvalues: dict[int, list[object]]
) -> None:
    """Initialize all 9 block-state gauge label series to 0."""
    for lv in per_engine_labelvalues.values():
        for block_state in BLOCK_STATES:
            gauge.labels(*lv, ALL_KV_GROUPS, block_state).set(0)


class KVBlockStateScanner:
    """Periodic scanner classifying KV cache blocks into 9 exclusive states."""

    def __init__(self, interval_s: float | None = None) -> None:
        if interval_s is None:
            interval_s = float(envs.TPU_KV_BLOCK_STATE_INTERVAL_S)
        self._interval_s = float(interval_s)
        self._last_scan_ts: float | None = None

    def maybe_scan(
        self,
        scheduler: Any,
        finished_loading_req_ids: set[str] | None = None,
        now: float | None = None,
    ) -> dict[str, int] | None:
        if now is None:
            now = time.monotonic()
        if (
            self._last_scan_ts is not None
            and (now - self._last_scan_ts) < self._interval_s
        ):
            return None
        self._last_scan_ts = now
        return self.scan(scheduler, finished_loading_req_ids)

    @staticmethod
    def scan(
        scheduler: Any,
        finished_loading_req_ids: set[str] | None = None,
    ) -> dict[str, int] | None:
        kv_cache_manager = getattr(scheduler, "kv_cache_manager", None)
        coordinator = getattr(kv_cache_manager, "coordinator", None)
        block_pool = getattr(kv_cache_manager, "block_pool", None)
        if block_pool is None and coordinator is not None:
            block_pool = getattr(coordinator, "block_pool", None)
        managers = (
            getattr(coordinator, "single_type_managers", None)
            if coordinator is not None
            else None
        )
        if block_pool is None or not managers:
            return None
        all_blocks = getattr(block_pool, "blocks", None)
        if all_blocks is None:
            return None

        running_reqs = getattr(scheduler, "running", ()) or ()
        running_req_ids = {r.request_id for r in running_reqs}
        requests_map: dict[str, Any] = getattr(scheduler, "requests", None) or {}
        waiting_for_remote_status = getattr(
            RequestStatus, "WAITING_FOR_REMOTE_KVS", None
        )
        aborted_loading = finished_loading_req_ids or set()

        state: dict[int, str] = {}
        num_running_refs: dict[int, int] = {}

        def _mark(block: Any, candidate: str) -> None:
            if getattr(block, "is_null", False):
                return
            block_pool_attr = getattr(block, "pool", None)
            if block_pool_attr not in (None, block_pool):
                return
            bid = block.block_id
            if not (0 <= bid < len(all_blocks)) or all_blocks[bid] is not block:
                return
            cur = state.get(bid)
            if cur is None or _STATE_PRIORITY[candidate] < _STATE_PRIORITY[cur]:
                state[bid] = candidate

        def _classify_non_running_holder(req_id: str, req: Any | None) -> str:
            if req is None:
                return "loading_kv" if req_id in aborted_loading else "held_for_send"
            status = getattr(req, "status", None)
            if (
                waiting_for_remote_status is not None
                and status == waiting_for_remote_status
            ):
                return "loading_kv"
            is_finished_fn = getattr(req, "is_finished", None)
            if callable(is_finished_fn) and is_finished_fn():
                return "loading_kv" if req_id in aborted_loading else "held_for_send"
            return "waiting_hold"

        for manager in managers:
            req_to_blocks = getattr(manager, "req_to_blocks", None) or {}
            for req_id, blocks in req_to_blocks.items():
                if req_id in running_req_ids:
                    for b in blocks:
                        if getattr(b, "is_null", False):
                            continue
                        block_pool_attr = getattr(b, "pool", None)
                        if block_pool_attr not in (None, block_pool):
                            continue
                        bid = b.block_id
                        if not (0 <= bid < len(all_blocks)) or all_blocks[bid] is not b:
                            continue
                        num_running_refs[bid] = num_running_refs.get(bid, 0) + 1
                else:
                    holder_state = _classify_non_running_holder(
                        req_id, requests_map.get(req_id)
                    )
                    for b in blocks:
                        _mark(b, holder_state)

        for bid, n_refs in num_running_refs.items():
            state[bid] = "active_shared" if n_refs > 1 else "active_private"

        for _step, blocks in getattr(scheduler, "deferred_frees", ()) or ():
            for b in blocks:
                _mark(b, "transient_pin")

        counts: dict[str, int] = dict.fromkeys(BLOCK_STATES, 0)
        cached_by_id = getattr(block_pool, "cached_block_hashes_by_block", None) or {}
        for b in all_blocks:
            if getattr(b, "is_null", False):
                continue
            bid = b.block_id
            assigned = state.get(bid)
            if assigned is not None:
                counts[assigned] += 1
            elif b.ref_cnt > 0:
                counts["orphan_suspect"] += 1
            elif getattr(b, "block_hash", None) is not None or bid in cached_by_id:
                counts["cached_evictable"] += 1
            else:
                counts["free_unhashed"] += 1
        return counts


class KVCacheMemoryCollector:
    """Scheduler-side collector that periodically scans KV block states."""

    def __init__(self, scheduler: Any, interval_s: float | None = None) -> None:
        self._scheduler = scheduler
        self._scanner = KVBlockStateScanner(interval_s=interval_s)
        self._finished_loading_req_ids: set[str] = set()
        self._wrap_finish_requests(scheduler)

    def _wrap_finish_requests(self, scheduler: Any) -> None:
        orig_finish = getattr(scheduler, "finish_requests", None)
        if not callable(orig_finish):
            return

        loading_sets: list[set[str]] | None = getattr(
            scheduler, "_tpu_finished_loading_sets", None
        )
        if not isinstance(loading_sets, list):
            loading_sets = []
            scheduler._tpu_finished_loading_sets = loading_sets
        if self._finished_loading_req_ids not in loading_sets:
            loading_sets.append(self._finished_loading_req_ids)

        if getattr(orig_finish, "_tpu_kv_memory_wrapped", None) is True:
            return

        waiting_status = getattr(RequestStatus, "WAITING_FOR_REMOTE_KVS", None)

        def _wrapped_finish_requests(
            request_ids: Any, *args: Any, **kwargs: Any
        ) -> Any:
            requests_map = getattr(scheduler, "requests", None) or {}
            if request_ids is None:
                target_ids: Iterable[Any] = tuple(requests_map)
            elif isinstance(request_ids, str):
                target_ids = (request_ids,)
            else:
                target_ids = request_ids
            for req_id in target_ids:
                req = requests_map.get(req_id)
                if (
                    req is not None
                    and waiting_status is not None
                    and getattr(req, "status", None) == waiting_status
                ):
                    for loading_set in loading_sets:
                        loading_set.add(req_id)
            return orig_finish(request_ids, *args, **kwargs)

        _wrapped_finish_requests.__dict__["_tpu_kv_memory_wrapped"] = True
        scheduler.finish_requests = _wrapped_finish_requests

    def _prune_finished_loading_req_ids(self) -> None:
        if not self._finished_loading_req_ids:
            return
        kv_cache_manager = getattr(self._scheduler, "kv_cache_manager", None)
        coordinator = getattr(kv_cache_manager, "coordinator", None)
        managers = (
            getattr(coordinator, "single_type_managers", None)
            if coordinator is not None
            else None
        )
        if not managers:
            self._finished_loading_req_ids.clear()
            return
        held_req_ids: set[str] = set()
        for manager in managers:
            req_to_blocks = getattr(manager, "req_to_blocks", None) or {}
            held_req_ids.update(req_to_blocks.keys())
        self._finished_loading_req_ids &= held_req_ids

    def collect(self, now: float | None = None) -> dict[str, Any]:
        self._prune_finished_loading_req_ids()
        block_counts = self._scanner.maybe_scan(
            self._scheduler, self._finished_loading_req_ids, now=now
        )
        if block_counts is None:
            return {}
        return {STATS_BLOCKS_BY_STATE: block_counts}


class KVCacheMemoryProm:
    """Frontend Prometheus observer for ``vllm:kv_cache_blocks_by_state``."""

    def __init__(
        self,
        gauge_cls: Any,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> None:
        self._per_engine_labelvalues = per_engine_labelvalues
        self.blocks_by_state = get_or_create_kv_cache_blocks_by_state_gauge(
            gauge_cls, labelnames
        )
        init_kv_cache_blocks_by_state_gauge(
            self.blocks_by_state, per_engine_labelvalues
        )

    def observe(self, stats_data: dict[str, Any], engine_idx: int = 0) -> None:
        if not stats_data:
            return
        lv = self._per_engine_labelvalues.get(engine_idx)
        if lv is None:
            return
        blocks_by_state = stats_data.get(STATS_BLOCKS_BY_STATE)
        if blocks_by_state:
            for block_state, count in blocks_by_state.items():
                self.blocks_by_state.labels(*lv, ALL_KV_GROUPS, block_state).set(count)
