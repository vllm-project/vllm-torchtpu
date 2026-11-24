# SPDX-License-Identifier: Apache-2.0
"""Arrival-time prefix touch (VLLM_TPU_ARRIVAL_TOUCH, engine core).

A request's prefix is looked up when the scheduler takes it from the waiting
queue, so the time it spends waiting counts as age in both prefix caches. On the
B1 prefill at c227 the queue wait (median 45.8 s) outlived the HBM pool's
retention (about 22 s): returners arrived with their prefix still in HBM and lost
it while they waited, so 0.1 % of the reusable tokens reached their lookup
younger than 20 s against 59.7 % at c200 (reports/hunt_iter19_b1_cliff_1121.md).
The local hit then collapses, the prefill loads and computes more, and the queue
grows further.

With the flag, when ``Scheduler.add_request`` puts a new request in the waiting
queue:

- HBM: the request's cached prefix blocks are pinned (``BlockPool.touch``, one
  reference per waiting request) so the LRU cannot evict them while it waits.
  The blocks are the ones the scheduler's own lookup would adopt now: the
  convergent hit (``find_longest_cache_hit``) and, when the connector takes
  divergent hybrid hits, the attention groups' longer run
  (``divergent_hybrid_hits.divergent_local_hit``). A pin is all or nothing and is
  granted only while (a) the distinct blocks pinned for waiting requests stay at
  most ``VLLM_TPU_ARRIVAL_TOUCH_HBM_FRAC`` of the pool and (b) the pool keeps at
  least ``VLLM_TPU_ARRIVAL_TOUCH_MIN_FREE_FRAC`` of its blocks free. Before every
  schedule pass, pins older than ``VLLM_TPU_ARRIVAL_TOUCH_TTL_S`` are released,
  and the newest pins are released while the pool is below its free floor. After
  the pass, the pin of every request that left the fresh-waiting state is
  released (scheduled, loading, finished); ``finish_requests`` releases the pins
  of the requests it ends. A released block keeps its hash and goes back to the
  free queue's most recently used end, as any freed cached block does.
- Host tier: the entries a load would read for the request now, past its local
  hit, are refreshed in the store's LRU (``TPURaidenOffloadingConnector
  .arrival_touch``: one pinned lookup and one release per tier class, which puts
  them at the most recently used end; nothing is loaded and no pin is kept).

Placement only: no block is written, allocated or copied, and every hit still
reads the bytes the cache holds. Without the flag nothing is installed.
``VLLM_TPU_ARRIVAL_TOUCH_HBM_FRAC=0`` turns the HBM pins off and
``VLLM_TPU_ARRIVAL_TOUCH_TIER=0`` the host-tier refresh.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_PATCH_FLAG = "_tpu_arrival_touch_patch"
_STATE_ATTR = "_tpu_arrival_touch"
_LOG_INTERVAL_S = 60.0
_clock = time.monotonic


def enabled() -> bool:
    return bool(envs.VLLM_TPU_ARRIVAL_TOUCH)


@dataclass
class _Pin:
    """One waiting request's pinned blocks, per KV cache group in position order."""

    groups: list[list[Any]]
    t: float
    unique_new: int


@dataclass
class ArrivalPins:
    """Pins of waiting requests' prefix blocks and the counters the log line reports."""

    hbm_frac: float
    min_free_frac: float
    ttl_s: float
    tier: bool
    pins: dict[str, _Pin] = field(default_factory=dict)
    # id(block) -> waiting requests pinning it.
    refs: dict[int, int] = field(default_factory=dict)
    # id(pool) -> distinct blocks pinned in that pool.
    unique: dict[int, int] = field(default_factory=dict)
    stats: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(
            (
                "arrivals",
                "pinned",
                "pinned_blocks",
                "no_hit",
                "denied_budget",
                "denied_floor",
                "rel_sched",
                "rel_finish",
                "rel_ttl",
                "rel_pressure",
                "adopted_blocks",
                "tier_requests",
                "tier_keys",
                "tier_errors",
                "peak_unique",
                "us",
            ),
            0,
        )
    )
    tier_off: bool = False
    # Set when an arrival raised: no further pins (held ones are released).
    off: bool = False
    log_t: float = 0.0

    # -- pin / release ------------------------------------------------------

    def _pools(self, kvm) -> list[Any]:
        return [m.block_pool for m in kvm.coordinator.single_type_managers]

    def pin(self, kvm, request_id: str, groups: Sequence[Sequence[Any]], now: float) -> bool:
        """Pins ``groups`` (per group, position order) for a waiting request, all or nothing."""
        pools = self._pools(kvm)
        chosen: list[list[Any]] = []
        seen: set[int] = set()
        new_unique: dict[int, int] = {}
        leaving_free: dict[int, int] = {}
        for gid, blocks in enumerate(groups):
            keep = []
            for block in blocks:
                if block is None or block.is_null or id(block) in seen:
                    continue
                seen.add(id(block))
                keep.append(block)
                pool = pools[gid]
                if id(block) not in self.refs:
                    new_unique[id(pool)] = new_unique.get(id(pool), 0) + 1
                if block.ref_cnt == 0:
                    leaving_free[id(pool)] = leaving_free.get(id(pool), 0) + 1
            chosen.append(keep)
        if not seen:
            self.stats["no_hit"] += 1
            return False
        by_id = {id(p): p for p in pools}
        for pid, n in new_unique.items():
            pool = by_id[pid]
            if self.unique.get(pid, 0) + n > self.hbm_frac * pool.num_gpu_blocks:
                self.stats["denied_budget"] += 1
                return False
        for pid, n in leaving_free.items():
            pool = by_id[pid]
            if pool.get_num_free_blocks() - n < self.min_free_frac * pool.num_gpu_blocks:
                self.stats["denied_floor"] += 1
                return False
        for gid, keep in enumerate(chosen):
            if keep:
                pools[gid].touch(keep)
        for gid, keep in enumerate(chosen):
            for block in keep:
                self.refs[id(block)] = self.refs.get(id(block), 0) + 1
        for pid, n in new_unique.items():
            self.unique[pid] = self.unique.get(pid, 0) + n
        self.pins[request_id] = _Pin(chosen, now, sum(new_unique.values()))
        self.stats["pinned"] += 1
        self.stats["pinned_blocks"] += len(seen)
        self.stats["peak_unique"] = max(self.stats["peak_unique"], sum(self.unique.values()))
        return True

    def release(self, kvm, request_id: str, reason: str) -> bool:
        """Drops a request's pins; blocks nobody else holds go back to the free queue."""
        pin = self.pins.pop(request_id, None)
        if pin is None:
            return False
        from vllm.v1.core.single_type_kv_cache_manager import MambaManager

        managers = kvm.coordinator.single_type_managers
        pools = self._pools(kvm)
        if reason == "rel_sched":
            owned = {id(b) for b in _request_blocks(kvm, request_id)}
            self.stats["adopted_blocks"] += sum(
                1 for keep in pin.groups for b in keep if id(b) in owned
            )
        # Free tail first, attention before state (the retention patch's
        # eviction order), one call per pool.
        attention = [k for gid, k in enumerate(pin.groups) if not isinstance(managers[gid], MambaManager)]
        state = [k for gid, k in enumerate(pin.groups) if isinstance(managers[gid], MambaManager)]
        pool_of = {id(b): pools[gid] for gid, keep in enumerate(pin.groups) for b in keep}
        per_pool: dict[int, list[Any]] = {}
        for block in _tail_first(attention) + _tail_first(state):
            pool = pool_of[id(block)]
            per_pool.setdefault(id(pool), []).append(block)
            n = self.refs[id(block)] - 1
            if n:
                self.refs[id(block)] = n
            else:
                del self.refs[id(block)]
                self.unique[id(pool)] -= 1
        by_id = {id(p): p for p in pools}
        for pid, blocks in per_pool.items():
            by_id[pid].free_blocks(blocks)
        self.stats[reason] += 1
        return True

    def release_all(self, kvm, reason: str) -> None:
        for request_id in list(self.pins):
            self.release(kvm, request_id, reason)

    # -- schedule-pass hooks ------------------------------------------------

    def before_schedule(self, kvm, now: float) -> None:
        """TTL, then the free floor: newest pins go first."""
        for request_id, pin in list(self.pins.items()):
            if now - pin.t > self.ttl_s:
                self.release(kvm, request_id, "rel_ttl")
        if not self.pins:
            return
        for pool in {id(p): p for p in self._pools(kvm)}.values():
            floor = self.min_free_frac * pool.num_gpu_blocks
            while self.pins and pool.get_num_free_blocks() < floor:
                newest = max(self.pins, key=lambda r: self.pins[r].t)
                self.release(kvm, newest, "rel_pressure")

    def after_schedule(self, scheduler) -> None:
        """Releases the pin of every request no longer fresh in the waiting queue."""
        from vllm.v1.request import RequestStatus

        kvm = scheduler.kv_cache_manager
        for request_id in list(self.pins):
            request = scheduler.requests.get(request_id)
            if request is None or request.is_finished():
                self.release(kvm, request_id, "rel_finish")
            elif request.status != RequestStatus.WAITING or request.num_computed_tokens > 0:
                self.release(kvm, request_id, "rel_sched")

    def line(self) -> str:
        s = self.stats
        return (
            "arrival-touch: arrivals=%d pinned=%d pinned_blocks=%d no_hit=%d denied_budget=%d "
            "denied_floor=%d rel_sched=%d rel_finish=%d rel_ttl=%d rel_pressure=%d adopted_blocks=%d "
            "held=%d unique=%d peak_unique=%d tier_requests=%d tier_keys=%d tier_errors=%d us=%d"
            % (
                s["arrivals"], s["pinned"], s["pinned_blocks"], s["no_hit"], s["denied_budget"],
                s["denied_floor"], s["rel_sched"], s["rel_finish"], s["rel_ttl"], s["rel_pressure"],
                s["adopted_blocks"], len(self.pins), sum(self.unique.values()), s["peak_unique"],
                s["tier_requests"], s["tier_keys"], s["tier_errors"], s["us"],
            )
        )

    def maybe_log(self, now: float) -> None:
        if now - self.log_t >= _LOG_INTERVAL_S:
            self.log_t = now
            logger.info("%s", self.line())


def _tail_first(lists: Sequence[Sequence[Any]]) -> list[Any]:
    """Position-major across groups, highest position first."""
    order: list[Any] = []
    for pos in range(max((len(b) for b in lists), default=0) - 1, -1, -1):
        order.extend(b[pos] for b in lists if pos < len(b))
    return order


def _request_blocks(kvm, request_id: str) -> list[Any]:
    out: list[Any] = []
    for manager in kvm.coordinator.single_type_managers:
        out.extend(manager.req_to_blocks.get(request_id, ()))
    return out


def _tier_member(connector) -> Any | None:
    """The scheduler-side connector that refreshes host-tier entries: the connector
    itself or a member of a multi connector; None without one."""
    if connector is None:
        return None
    members = getattr(connector, "sub_connectors", None)
    for member in [connector] if members is None else members:
        if callable(getattr(member, "arrival_touch", None)):
            return member
    return None


def arrival_blocks(kvm, connector, request) -> tuple[list[list[Any]], int]:
    """The blocks the scheduler's local lookup would adopt for ``request`` now, per
    group, and the hit it would report (tokens). Reads the cache only.

    The convergent hit is ``get_computed_blocks``'s. When the connector takes
    divergent hybrid hits (VLLM_TPU_OFFLOAD_DIVERGENT_HITS) on a hybrid with several
    token groups, the attention groups' run (``divergent_local_hit``) is added, so a
    pin keeps both the attention run and the state checkpoint the convergent
    fallback would resume from."""
    if not kvm.enable_caching or not kvm.prefix_cache_lookup_enabled(request):
        return [], 0
    coordinator = kvm.coordinator
    blocks, local, _ = coordinator.find_longest_cache_hit(
        request.block_hashes, request.num_tokens - 1
    )
    groups = [list(b) for b in blocks]
    if connector is not None and getattr(connector, "supports_divergent_local_hybrid_hits", False):
        from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator

        from vllm_torchtpu.core.divergent_hybrid_hits import _token_groups, divergent_local_hit

        config = kvm.kv_cache_config
        if (
            config.has_mamba_layers
            and isinstance(coordinator, HybridKVCacheCoordinator)
            and len(_token_groups(config)) > 1
        ):
            dblocks, dhit, _, _ = divergent_local_hit(kvm, request)
            if dhit > local:
                for gid, extra in enumerate(dblocks.blocks):
                    have = {id(b) for b in groups[gid]}
                    groups[gid].extend(b for b in extra if id(b) not in have)
                local = dhit
    return groups, local


def on_arrival(scheduler, request, pins: ArrivalPins) -> None:
    """Pins the request's HBM prefix and refreshes its host-tier entries."""
    t0 = time.perf_counter()
    pins.stats["arrivals"] += 1
    kvm = scheduler.kv_cache_manager
    groups, local = arrival_blocks(kvm, scheduler.connector, request)
    if pins.hbm_frac > 0 and groups:
        pins.pin(kvm, request.request_id, groups, _clock())
    if pins.tier and not pins.tier_off:
        member = _tier_member(scheduler.connector)
        if member is not None:
            block = scheduler.block_size
            try:
                keys = int(member.arrival_touch(request, local - local % block))
            except Exception:
                pins.tier_off = True
                pins.stats["tier_errors"] += 1
                logger.exception(
                    "arrival-touch: host-tier refresh failed; HBM pins only from now on."
                )
                keys = 0
            if keys:
                pins.stats["tier_requests"] += 1
                pins.stats["tier_keys"] += keys
    pins.stats["us"] += int((time.perf_counter() - t0) * 1e6)


def _state(scheduler) -> ArrivalPins:
    pins = scheduler.__dict__.get(_STATE_ATTR)
    if pins is None:
        pins = ArrivalPins(
            hbm_frac=float(envs.VLLM_TPU_ARRIVAL_TOUCH_HBM_FRAC),
            min_free_frac=float(envs.VLLM_TPU_ARRIVAL_TOUCH_MIN_FREE_FRAC),
            ttl_s=float(envs.VLLM_TPU_ARRIVAL_TOUCH_TTL_S),
            tier=bool(envs.VLLM_TPU_ARRIVAL_TOUCH_TIER),
        )
        if not 0.0 <= pins.hbm_frac <= 1.0 or not 0.0 <= pins.min_free_frac < 1.0 or pins.ttl_s <= 0:
            raise ValueError(
                "VLLM_TPU_ARRIVAL_TOUCH_HBM_FRAC must be in [0, 1], "
                "VLLM_TPU_ARRIVAL_TOUCH_MIN_FREE_FRAC in [0, 1) and "
                f"VLLM_TPU_ARRIVAL_TOUCH_TTL_S > 0; got {pins.hbm_frac}, "
                f"{pins.min_free_frac}, {pins.ttl_s}"
            )
        scheduler.__dict__[_STATE_ATTR] = pins
        logger.info(
            "arrival-touch: on (HBM pins %s of the pool, free floor %s, ttl %g s; host tier %s)",
            pins.hbm_frac,
            pins.min_free_frac,
            pins.ttl_s,
            "refreshed" if pins.tier else "not refreshed",
        )
    return pins


def patch_arrival_touch() -> None:
    """Wraps Scheduler.add_request, schedule and finish_requests. Installs nothing
    unless VLLM_TPU_ARRIVAL_TOUCH is set."""
    if not enabled():
        return
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import RequestStatus

    if getattr(Scheduler.add_request, _PATCH_FLAG, False):
        return
    original_add = Scheduler.add_request
    original_schedule = Scheduler.schedule
    original_finish = Scheduler.finish_requests

    def add_request(self, request):
        new = request.request_id not in self.requests
        original_add(self, request)
        if (
            new
            and request.status == RequestStatus.WAITING
            and request.num_computed_tokens == 0
            and self.requests.get(request.request_id) is request
        ):
            pins = _state(self)
            if pins.off:
                return
            try:
                on_arrival(self, request, pins)
            except Exception:
                # A placement aid must not fail a request: stop pinning.
                pins.off = True
                pins.release_all(self.kv_cache_manager, "rel_finish")
                logger.exception("arrival-touch: arrival failed; arrival touch off from now on.")

    def schedule(self, *args, **kwargs):
        pins = _state(self)
        now = _clock()
        pins.before_schedule(self.kv_cache_manager, now)
        out = original_schedule(self, *args, **kwargs)
        pins.after_schedule(self)
        pins.maybe_log(now)
        return out

    def finish_requests(self, request_ids, finished_status):
        out = original_finish(self, request_ids, finished_status)
        pins = self.__dict__.get(_STATE_ATTR)
        if pins is not None and pins.pins:
            for request in out:
                pins.release(self.kv_cache_manager, request.request_id, "rel_finish")
        return out

    for fn in (add_request, schedule, finish_requests):
        setattr(fn, _PATCH_FLAG, True)
    Scheduler.add_request = add_request
    Scheduler.schedule = schedule
    Scheduler.finish_requests = finish_requests
    logger.info("Applied TPU patch: arrival-time prefix touch (VLLM_TPU_ARRIVAL_TOUCH).")
