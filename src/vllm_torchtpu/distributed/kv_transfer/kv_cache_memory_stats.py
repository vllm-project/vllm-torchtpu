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
"""HBM KV cache memory metrics, collected on the scheduler side.

`TPUConnector` binds a `KVCacheMemoryCollector` to the vLLM scheduler (see
`_patch_vllm_scheduler_binds_kv_connector`) and drains it into its stats every
step; `KVCacheMemoryProm` exports them from `TpuKVConnectorPromMetrics`:

- `vllm:kv_cache_blocks_by_state{kv_group,state}`: every usable block in one
  state, from a periodic scan (`TPU_KV_BLOCK_STATE_INTERVAL_S`).
- `vllm:kv_orphan_block_info{kv_group,block_id,last_owner}`: blocks referenced
  more times than any table accounts for (suspected leaks).
- `vllm:kv_admission_reserved_blocks{kind}`: free blocks held back from
  admissions.
- `vllm:kv_admission_refusals{reason}`: waiting requests refused blocks.
- `vllm:kv_token_slots{kv_group,kind}`: token slots in the blocks of running
  and loading requests.

Ratios worth charting:

- resident HBM: 1 - free_unhashed / sum(kv_cache_blocks_by_state)
- admissible free blocks:
  cached_evictable + free_unhashed - sum(kv_admission_reserved_blocks)
- tail slack: tail_slack / (filled + tail_slack)
"""

from __future__ import annotations

import itertools
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from vllm.v1.core.single_type_kv_cache_manager import (
    CrossAttentionManager,
    MambaManager,
)
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.single_type_kv_cache_manager import (
        SingleTypeKVCacheManager,
    )
    from vllm.v1.request import Request

# Admission refusal reasons, from the most to the least fundamental.
EXCEEDS_POOL_CAPACITY = "exceeds_pool_capacity"
INSUFFICIENT_FREE_BLOCKS = "insufficient_free_blocks"
WATERMARK = "watermark"
INFLIGHT_LOAD_RESERVE = "inflight_load_reserve"
HEAD_OF_LINE = "head_of_line"
REFUSAL_REASONS = (
    EXCEEDS_POOL_CAPACITY,
    INSUFFICIENT_FREE_BLOCKS,
    WATERMARK,
    INFLIGHT_LOAD_RESERVE,
    HEAD_OF_LINE,
)

# Kinds of reserved free blocks.
RESERVED_WATERMARK = "watermark"
RESERVED_INFLIGHT_LOAD = "inflight_load"
RESERVED_KINDS = (RESERVED_WATERMARK, RESERVED_INFLIGHT_LOAD)

# Block states, in priority order: a block matching several states counts as
# the first one.
ACTIVE_SHARED = "active_shared"
ACTIVE_PRIVATE = "active_private"
HELD_FOR_SEND = "held_for_send"
LOADING_KV = "loading_kv"
WAITING_HOLD = "waiting_hold"
TRANSIENT_PIN = "transient_pin"
ORPHAN_SUSPECT = "orphan_suspect"
CACHED_EVICTABLE = "cached_evictable"
FREE_UNHASHED = "free_unhashed"
BLOCK_STATES = (
    ACTIVE_SHARED,
    ACTIVE_PRIVATE,
    HELD_FOR_SEND,
    LOADING_KV,
    WAITING_HOLD,
    TRANSIENT_PIN,
    ORPHAN_SUSPECT,
    CACHED_EVICTABLE,
    FREE_UNHASHED,
)
# Holder states by priority; lower wins. Active is split into shared/private
# after counting running referrers.
_ACTIVE = "active"
_HOLDER_STATES = (_ACTIVE, HELD_FOR_SEND, LOADING_KV, WAITING_HOLD, TRANSIENT_PIN)
_PRIORITY = {state: i for i, state in enumerate(_HOLDER_STATES)}

# Kinds of token slots.
SLOTS_FILLED = "filled"
SLOTS_TAIL_SLACK = "tail_slack"
SLOTS_RESERVED = "reserved"

MAX_ORPHANS_REPORTED = 16

# The KV cache groups share one block pool, so block states have one series.
ALL_KV_GROUPS = "all"

# Keys in `TpuKVConnectorStats.data`. Snapshot keys hold one entry per step;
# the newest entry wins.
STATS_REFUSALS = "kv_admission_refusals"
STATS_RESERVED_BLOCKS = "kv_admission_reserved_blocks"
STATS_TOKEN_SLOTS = "kv_token_slots"
STATS_BLOCKS_BY_STATE = "kv_blocks_by_state"
STATS_ORPHAN_BLOCKS = "kv_orphan_blocks"
STATS_KEYS = (
    STATS_REFUSALS,
    STATS_RESERVED_BLOCKS,
    STATS_TOKEN_SLOTS,
    STATS_BLOCKS_BY_STATE,
    STATS_ORPHAN_BLOCKS,
)

_WAITING_ADMISSION = (RequestStatus.WAITING, RequestStatus.PREEMPTED)


def classify_refusal(
    num_required: int,
    num_free: int,
    watermark_blocks: int,
    reserved_blocks: int,
    capacity: int,
) -> str:
    """Return why an allocation of `num_required` blocks was refused.

    The reasons are checked from the most to the least fundamental: an
    allocation refused for the watermark would fit without it, and one refused
    for the in-flight load reserve would fit without that reserve.
    """
    del reserved_blocks  # The remaining reason once the others are ruled out.
    if num_required > capacity:
        return EXCEEDS_POOL_CAPACITY
    if num_required > num_free:
        return INSUFFICIENT_FREE_BLOCKS
    if num_required + watermark_blocks > num_free:
        return WATERMARK
    return INFLIGHT_LOAD_RESERVE


class KVAdmissionRefusalTracker:
    """Counts refused admissions of waiting requests, once per episode.

    The head of the waiting queue is retried every step, so a refusal counts
    only when a request starts being refused for a reason. Requests queued
    behind a refused head count once as `head_of_line` when the head's
    episode starts.
    """

    def __init__(self) -> None:
        self._reason_by_req_id: dict[str, str] = {}
        self._blocked_head_req_id: str | None = None
        self._counts: Counter[str] = Counter()

    def record_refusal(self, req_id: str, reason: str, num_queued_behind: int) -> None:
        if self._reason_by_req_id.get(req_id) != reason:
            self._reason_by_req_id[req_id] = reason
            self._counts[reason] += 1
        if self._blocked_head_req_id != req_id:
            self._blocked_head_req_id = req_id
            if num_queued_behind > 0:
                self._counts[HEAD_OF_LINE] += num_queued_behind

    def record_admission(self, req_id: str) -> None:
        self._reason_by_req_id.pop(req_id, None)
        if self._blocked_head_req_id == req_id:
            self._blocked_head_req_id = None

    def prune(self, live_req_ids: Iterable[str]) -> None:
        """Forget requests that are gone."""
        live = set(live_req_ids)
        for req_id in self._reason_by_req_id.keys() - live:
            self.record_admission(req_id)
        if self._blocked_head_req_id not in live:
            self._blocked_head_req_id = None

    def take_counts(self) -> dict[str, int]:
        counts = dict(self._counts)
        self._counts.clear()
        return counts


def _is_token_group(manager: SingleTypeKVCacheManager) -> bool:
    # A Mamba state block holds one state whatever its size, and
    # cross-attention blocks hold encoder tokens; neither has token slots.
    return not isinstance(manager, (MambaManager, CrossAttentionManager))


def compute_token_slots(
    managers: Iterable[SingleTypeKVCacheManager],
    requests: Iterable[Request],
) -> list[tuple[str, str, int]]:
    """Token slots in the blocks held by `requests`, per KV cache group.

    Returns (kv_group, kind, slots) with kinds:
    - filled: computed tokens still held (tokens outside a sliding window,
      whose blocks are released, are excluded),
    - tail_slack: unused slots in each request's last computed block,
    - reserved: slots in blocks allocated past the computed tokens
      (scheduled, lookahead and speculative slots).

    Shared prefix blocks count once per request that references them.
    """
    requests = list(requests)
    slots: list[tuple[str, str, int]] = []
    for group_id, manager in enumerate(managers):
        if not _is_token_group(manager):
            continue
        block_size = manager.block_size
        filled = tail_slack = reserved = 0
        for request in requests:
            blocks = manager.req_to_blocks.get(request.request_id)
            if not blocks:
                continue
            num_computed = request.num_computed_tokens
            num_computed_blocks = -(-num_computed // block_size)
            skipped = manager.get_num_skipped_tokens(num_computed)
            filled += num_computed - skipped // block_size * block_size
            tail_slack += num_computed_blocks * block_size - num_computed
            reserved += max(len(blocks) - num_computed_blocks, 0) * block_size
        group = str(group_id)
        slots += [
            (group, SLOTS_FILLED, filled),
            (group, SLOTS_TAIL_SLACK, tail_slack),
            (group, SLOTS_RESERVED, reserved),
        ]
    return slots


class KVBlockStateScanner:
    """Classifies every block of the HBM pool, at most once per interval.

    A scan walks the block tables of all requests that hold blocks and the
    scheduler's pending frees, then every block of the pool. It costs time
    proportional to the referenced blocks plus the pool size, which is why
    it runs periodically rather than every step.

    A block whose ref_cnt exceeds the references found is an orphan suspect.
    References a connector takes outside the block tables would look like
    leaks, so a block is reported only when the same excess is seen on two
    consecutive scans.
    """

    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self._next_scan = 0.0
        self._prev_owner: dict[int, str] = {}
        self._prev_excess: dict[int, int] = {}
        self._orphans: dict[int, str] = {}

    def maybe_scan(
        self,
        block_pool: BlockPool,
        managers: Iterable[SingleTypeKVCacheManager],
        requests: Iterable[Request],
        finished_loading_req_ids: set[str],
        pinned_blocks: Iterable[KVCacheBlock],
    ) -> tuple[dict[str, int], list[tuple[int, str]]] | None:
        now = time.monotonic()
        if now < self._next_scan:
            return None
        self._next_scan = now + self.interval_s
        return self.scan(
            block_pool, managers, requests, finished_loading_req_ids, pinned_blocks
        )

    def scan(
        self,
        block_pool: BlockPool,
        managers: Iterable[SingleTypeKVCacheManager],
        requests: Iterable[Request],
        finished_loading_req_ids: set[str],
        pinned_blocks: Iterable[KVCacheBlock],
    ) -> tuple[dict[str, int], list[tuple[int, str]]]:
        """Return the block count per state and up to 16 orphan suspects.

        `pinned_blocks` are blocks the scheduler holds outside the block
        tables until a step completes (deferred frees).
        """
        managers = list(managers)
        state: dict[int, str] = {}
        owner: dict[int, str] = {}
        num_refs: Counter[int] = Counter()
        num_running_refs: Counter[int] = Counter()

        def hold(block: KVCacheBlock, block_state: str, owner_id: str) -> None:
            # Deferred frees may hold blocks of another pool.
            if block.is_null or block.pool not in (None, block_pool):
                return
            block_id = block.block_id
            num_refs[block_id] += 1
            if block_state == _ACTIVE:
                num_running_refs[block_id] += 1
            prev = state.get(block_id)
            if prev is None or _PRIORITY[block_state] < _PRIORITY[prev]:
                state[block_id] = block_state
                owner[block_id] = owner_id

        for request in requests:
            req_id = request.request_id
            if request.status == RequestStatus.RUNNING:
                request_state = _ACTIVE
            elif request.is_finished():
                request_state = (
                    LOADING_KV if req_id in finished_loading_req_ids else HELD_FOR_SEND
                )
            elif request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                request_state = LOADING_KV
            else:
                request_state = WAITING_HOLD
            for manager in managers:
                for block in manager.req_to_blocks.get(req_id, ()):
                    hold(block, request_state, req_id)
        for block in pinned_blocks:
            hold(block, TRANSIENT_PIN, "none")

        counts = dict.fromkeys(BLOCK_STATES, 0)
        excess: dict[int, int] = {}
        for block in block_pool.blocks:
            if block.is_null:
                continue
            block_id = block.block_id
            if block.ref_cnt == 0:
                is_cached = (
                    block.block_hash is not None
                    or block_id in block_pool.cached_block_hashes_by_block
                )
                counts[CACHED_EVICTABLE if is_cached else FREE_UNHASHED] += 1
                continue
            num_excess = block.ref_cnt - num_refs[block_id]
            if num_excess > 0:
                excess[block_id] = num_excess
            block_state = state.get(block_id)
            if block_state is None:
                counts[ORPHAN_SUSPECT] += 1
            elif block_state == _ACTIVE:
                shared = num_running_refs[block_id] > 1
                counts[ACTIVE_SHARED if shared else ACTIVE_PRIVATE] += 1
            else:
                counts[block_state] += 1

        # Report a leak once its excess is stable across two scans.
        for block_id in list(self._orphans):
            if block_id not in excess:
                del self._orphans[block_id]
        for block_id, num_excess in excess.items():
            if (
                self._prev_excess.get(block_id) == num_excess
                and block_id not in self._orphans
            ):
                self._orphans[block_id] = self._prev_owner.get(block_id, "none")
        self._prev_excess = excess
        self._prev_owner = owner
        orphans = list(self._orphans.items())[-MAX_ORPHANS_REPORTED:]
        return counts, orphans


class KVCacheMemoryCollector:
    """Collects the HBM KV cache memory metrics of one vLLM scheduler.

    Wraps, on the scheduler's own instances only, `allocate_slots` and the
    coordinator's block count to classify refused admissions, and
    `finish_requests` to tell requests aborted while loading KV (whose blocks
    stay held until the load ends) from finished ones held for a send.
    """

    def __init__(self, scheduler: Scheduler, interval_s: float) -> None:
        self._scheduler = scheduler
        self._refusals = KVAdmissionRefusalTracker()
        self._scanner = KVBlockStateScanner(interval_s)
        self._finished_loading_req_ids: set[str] = set()
        self._wrap_allocate_slots()
        self._wrap_finish_requests()

    def _wrap_allocate_slots(self) -> None:
        manager = self._scheduler.kv_cache_manager
        coordinator = manager.coordinator
        orig_get_num_blocks = coordinator.get_num_blocks_to_allocate
        orig_allocate_slots = manager.allocate_slots
        num_required = 0

        def get_num_blocks_to_allocate(*args, **kwargs) -> int:
            nonlocal num_required
            num_required = orig_get_num_blocks(*args, **kwargs)
            return num_required

        def allocate_slots(request: Request, *args, **kwargs):
            blocks = orig_allocate_slots(request, *args, **kwargs)
            if request.status in _WAITING_ADMISSION:
                if blocks is None:
                    self._record_refusal(request, num_required, kwargs)
                else:
                    self._refusals.record_admission(request.request_id)
            return blocks

        coordinator.get_num_blocks_to_allocate = get_num_blocks_to_allocate  # type: ignore[method-assign]
        manager.allocate_slots = allocate_slots  # type: ignore[method-assign]

    def _record_refusal(
        self, request: Request, num_required: int, kwargs: Mapping[str, Any]
    ) -> None:
        scheduler = self._scheduler
        manager = scheduler.kv_cache_manager
        block_pool = manager.block_pool
        # As in `KVCacheManager.allocate_slots`: the watermark applies to
        # waiting admissions once anything is scheduled.
        watermark_blocks = (
            manager.watermark_blocks if kwargs.get("has_scheduled_reqs", True) else 0
        )
        reason = classify_refusal(
            num_required,
            block_pool.get_num_free_blocks(),
            watermark_blocks,
            kwargs.get("reserved_blocks", 0),
            capacity=block_pool.num_gpu_blocks - 1,
        )
        num_queued = len(scheduler.waiting) + len(scheduler.kv_holding_waiting)
        self._refusals.record_refusal(request.request_id, reason, num_queued - 1)

    def _wrap_finish_requests(self) -> None:
        scheduler = self._scheduler
        orig_finish_requests = scheduler.finish_requests

        def finish_requests(request_ids, finished_status):
            if isinstance(request_ids, str):
                request_ids = (request_ids,)
            elif request_ids is not None:
                request_ids = list(request_ids)
            for req_id in request_ids or list(scheduler.requests):
                request = scheduler.requests.get(req_id)
                if (
                    request is not None
                    and request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
                    and req_id not in scheduler.finished_recving_kv_req_ids
                ):
                    self._finished_loading_req_ids.add(req_id)
            return orig_finish_requests(request_ids, finished_status)

        scheduler.finish_requests = finish_requests  # type: ignore[method-assign]

    def record(self, stats: Any) -> None:
        """Add this step's values to `stats` (a `TpuKVConnectorStats`)."""
        scheduler = self._scheduler
        manager = scheduler.kv_cache_manager
        managers = manager.coordinator.single_type_managers
        live = scheduler.requests
        self._finished_loading_req_ids.intersection_update(live)
        self._refusals.prune(live)

        refusals = self._refusals.take_counts()
        if refusals:
            stats.data[STATS_REFUSALS].extend(
                [reason, count] for reason, count in refusals.items()
            )
        stats.data[STATS_RESERVED_BLOCKS].append(
            [
                [
                    RESERVED_WATERMARK,
                    manager.watermark_blocks if scheduler.running else 0,
                ],
                [
                    RESERVED_INFLIGHT_LOAD,
                    scheduler._inflight_prefill_reserved_blocks(),
                ],
            ]
        )
        loading = (
            request
            for request in scheduler.kv_holding_waiting
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        stats.data[STATS_TOKEN_SLOTS].append(
            [
                list(slot)
                for slot in compute_token_slots(
                    managers, itertools.chain(scheduler.running, loading)
                )
            ]
        )
        scan = self._scanner.maybe_scan(
            manager.block_pool,
            managers,
            live.values(),
            self._finished_loading_req_ids,
            itertools.chain.from_iterable(
                blocks for _, blocks in scheduler.deferred_frees
            ),
        )
        if scan is not None:
            counts, orphans = scan
            stats.data[STATS_BLOCKS_BY_STATE].append(
                [[block_state, n] for block_state, n in counts.items()]
            )
            stats.data[STATS_ORPHAN_BLOCKS].append(
                [[block_id, last_owner] for block_id, last_owner in orphans]
            )


class KVCacheMemoryProm:
    """Prometheus metrics for the values `KVCacheMemoryCollector` records."""

    def __init__(
        self,
        gauge_cls: Any,
        counter_cls: Any,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        self.per_engine_labelvalues = per_engine_labelvalues

        self.gauge_blocks_by_state = gauge_cls(
            name="vllm:kv_cache_blocks_by_state",
            documentation=(
                "KV cache blocks in each state; the states sum to the usable "
                "blocks. Refreshed by a periodic scan "
                "(TPU_KV_BLOCK_STATE_INTERVAL_S)."
            ),
            multiprocess_mode="mostrecent",
            labelnames=[*labelnames, "kv_group", "state"],
        )
        self.gauge_reserved_blocks = gauge_cls(
            name="vllm:kv_admission_reserved_blocks",
            documentation=(
                "Free KV cache blocks the scheduler holds back from admissions: "
                "the watermark, and blocks reserved for in-flight async KV loads."
            ),
            multiprocess_mode="mostrecent",
            labelnames=[*labelnames, "kind"],
        )
        self.counter_refusals = counter_cls(
            name="vllm:kv_admission_refusals",
            documentation=(
                "Waiting requests refused KV cache blocks, once per request and "
                "reason; head_of_line counts requests queued behind a refused "
                "request."
            ),
            labelnames=[*labelnames, "reason"],
        )
        self.gauge_token_slots = gauge_cls(
            name="vllm:kv_token_slots",
            documentation=(
                "Token slots in the KV cache blocks of running and loading "
                "requests: filled, unused slots in each request's last block "
                "(tail_slack), and slots allocated ahead of computed tokens "
                "(reserved)."
            ),
            multiprocess_mode="mostrecent",
            labelnames=[*labelnames, "kv_group", "kind"],
        )
        self.gauge_orphan_block_info = gauge_cls(
            name="vllm:kv_orphan_block_info",
            documentation=(
                "KV cache blocks referenced more times than any request or "
                "scheduler table accounts for (suspected leaks), with the "
                "request that last held them; at most 16 per engine."
            ),
            multiprocess_mode="mostrecent",
            labelnames=[*labelnames, "kv_group", "block_id", "last_owner"],
        )
        self._orphan_labels: dict[int, set[tuple[str, str]]] = {
            idx: set() for idx in per_engine_labelvalues
        }

        # Create the fixed series so they exist at 0 before the first event.
        for labelvalues in per_engine_labelvalues.values():
            for block_state in BLOCK_STATES:
                self.gauge_blocks_by_state.labels(
                    *labelvalues, ALL_KV_GROUPS, block_state
                )
            for kind in RESERVED_KINDS:
                self.gauge_reserved_blocks.labels(*labelvalues, kind)
            for reason in REFUSAL_REASONS:
                self.counter_refusals.labels(*labelvalues, reason)

    def observe(self, data: Mapping[str, Any], engine_idx: int = 0) -> None:
        labelvalues = self.per_engine_labelvalues[engine_idx]
        for reason, count in data.get(STATS_REFUSALS, ()):
            self.counter_refusals.labels(*labelvalues, reason).inc(count)
        if reserved := data.get(STATS_RESERVED_BLOCKS):
            for kind, num_blocks in reserved[-1]:
                self.gauge_reserved_blocks.labels(*labelvalues, kind).set(num_blocks)
        if token_slots := data.get(STATS_TOKEN_SLOTS):
            for kv_group, kind, num_slots in token_slots[-1]:
                self.gauge_token_slots.labels(*labelvalues, kv_group, kind).set(
                    num_slots
                )
        if blocks_by_state := data.get(STATS_BLOCKS_BY_STATE):
            for block_state, num_blocks in blocks_by_state[-1]:
                self.gauge_blocks_by_state.labels(
                    *labelvalues, ALL_KV_GROUPS, block_state
                ).set(num_blocks)
        if orphans := data.get(STATS_ORPHAN_BLOCKS):
            self._set_orphans(engine_idx, labelvalues, orphans[-1])

    def _set_orphans(
        self,
        engine_idx: int,
        labelvalues: list[object],
        orphans: Iterable[tuple[int, str] | list[Any]],
    ) -> None:
        current = {(str(block_id), str(last_owner)) for block_id, last_owner in orphans}
        gauge = self.gauge_orphan_block_info
        for block_id, last_owner in self._orphan_labels[engine_idx] - current:
            series = (*labelvalues, ALL_KV_GROUPS, block_id, last_owner)
            if hasattr(gauge, "remove"):
                gauge.remove(*series)
            else:  # Ray metrics cannot drop a series; zero it instead.
                gauge.labels(*series).set(0)
        for block_id, last_owner in current:
            gauge.labels(*labelvalues, ALL_KV_GROUPS, block_id, last_owner).set(1)
        self._orphan_labels[engine_idx] = current
