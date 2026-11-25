# SPDX-License-Identifier: Apache-2.0
"""Pin the cached prefix of waiting requests (``VLLM_TPU_PIN_WAITING_PREFIX=1``).

Why: on a prefill-bound P/D leg the waiting queue grows with load (q38 1P1D, merged keys, no host tier:
mean TTFT 15 s at c192, 65 s at c224, almost all of it queueing). A request's cached prefix is free
(refcount 0) while it waits, so every block written for the requests ahead of it can evict it. The
longer the queue, the more queued prefixes are evicted, the more each one costs to compute, the
longer the queue: the prefix hit rate falls from 82% (c192) to 59% (c224) against a 94% ceiling and
throughput falls with it instead of flattening.

What: at the start of each ``Scheduler.schedule`` call,

1. requests that have left the waiting states (scheduled, so their own allocation now holds the
   blocks, or finished) give their pins back (``BlockPool.free_blocks``);
2. waiting requests not seen yet, in queue order, pin their current cached prefix
   (``coordinator.find_longest_cache_hit``, which leaves the prefix-cache stats alone, then
   ``BlockPool.touch``) while the pinned blocks stay under ``VLLM_TPU_PIN_WAITING_MAX_FRAC`` of the
   pool (0.5 by default). A request is tried once.

A pinned block is still cached, so the scheduler's own lookup finds it when the request is
scheduled and takes its own reference before the pin is released one step later. Engine-core only;
inert unless the flag is set.
"""

from __future__ import annotations

import os
import time

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

ENV = "VLLM_TPU_PIN_WAITING_PREFIX"


def enabled() -> bool:
    return os.environ.get(ENV, "0").strip().lower() in ("1", "true", "yes", "on")


class _Pins:
    def __init__(self, max_frac: float) -> None:
        self.max_frac = max_frac
        self.by_req: dict[str, list] = {}
        self.tried: set[str] = set()
        self.blocks = 0
        self.counts = dict.fromkeys(
            ("pinned_reqs", "pinned_blocks", "released_reqs", "zero_hit", "over_budget"), 0
        )
        self.last_log = time.monotonic()

    def maybe_log(self) -> None:
        now = time.monotonic()
        if now - self.last_log < 60.0:
            return
        self.last_log = now
        logger.info(
            "pin-waiting: holding %d reqs / %d blocks; %s",
            len(self.by_req),
            self.blocks,
            " ".join(f"{k}={v}" for k, v in self.counts.items()),
        )


def _waiting(status) -> bool:
    from vllm.v1.request import RequestStatus

    return status in (
        RequestStatus.WAITING,
        RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
        RequestStatus.WAITING_FOR_REMOTE_KVS,
        RequestStatus.WAITING_FOR_STREAMING_REQ,
    )


def release_done(pins: _Pins, requests: dict, block_pool) -> None:
    """Give back the pins of requests no longer waiting."""
    for req_id in [r for r in pins.by_req if not (r in requests and _waiting(requests[r].status))]:
        blocks = pins.by_req.pop(req_id)
        pins.blocks -= len(blocks)
        block_pool.free_blocks(blocks)
        pins.counts["released_reqs"] += 1
    # Forget requests that are gone, so the tried set stays bounded.
    if len(pins.tried) > 4 * max(1, len(requests)):
        pins.tried &= set(requests)


def keep_reserve(pins: _Pins, waiting, block_pool, reserve_frac: float) -> None:
    """Release pins, latest-queued first, until the pool has ``reserve_frac`` of its blocks free.

    Pins must never squeeze the requests being scheduled: a short pool preempts running
    requests, and a preempted P/D producer request that replays after its transfer was
    registered is fatal (j-20064e16-a87b-4171-85c2: "Conflicting replay after terminal
    Stage-3 producer registration" with half the pool pinned).
    """
    need = int(reserve_frac * block_pool.num_gpu_blocks)
    if block_pool.get_num_free_blocks() >= need or not pins.by_req:
        return
    order = [r.request_id for r in waiting if r.request_id in pins.by_req]
    rest = [rid for rid in pins.by_req if rid not in set(order)]
    for rid in reversed(rest + order):
        if block_pool.get_num_free_blocks() >= need:
            break
        blocks = pins.by_req.pop(rid)
        pins.blocks -= len(blocks)
        block_pool.free_blocks(blocks)
        pins.counts["reserve_released"] = pins.counts.get("reserve_released", 0) + 1


def pin_new(pins: _Pins, waiting, manager, reserve_frac: float = 0.0) -> None:
    """Pin the cached prefix of waiting requests not tried yet, in queue order, within budget."""
    pool = manager.block_pool
    budget = int(pins.max_frac * pool.num_gpu_blocks)
    for request in waiting:
        rid = request.request_id
        if rid in pins.tried:
            continue
        if pins.blocks >= budget or pool.get_num_free_blocks() < int(
            reserve_frac * pool.num_gpu_blocks
        ):
            pins.counts["over_budget"] += 1
            return
        pins.tried.add(rid)
        hashes = getattr(request, "block_hashes", None)
        if not hashes or not manager.enable_caching:
            pins.counts["zero_hit"] += 1
            continue
        groups, num_tokens, _ = manager.coordinator.find_longest_cache_hit(
            hashes, request.num_tokens - 1
        )
        blocks = [b for g in groups for b in g if not b.is_null]
        if not num_tokens or not blocks:
            pins.counts["zero_hit"] += 1
            continue
        pool.touch(blocks)
        pins.by_req[rid] = blocks
        pins.blocks += len(blocks)
        pins.counts["pinned_reqs"] += 1
        pins.counts["pinned_blocks"] += len(blocks)


def patch_pin_waiting_prefix() -> None:
    """Install the hook. Idempotent; a no-op unless VLLM_TPU_PIN_WAITING_PREFIX is set."""
    if not enabled():
        return
    from vllm.v1.core.sched.scheduler import Scheduler

    if Scheduler.__dict__.get("_tpu_pin_waiting_patch", False):
        return
    max_frac = float(os.environ.get("VLLM_TPU_PIN_WAITING_MAX_FRAC", "0.35"))
    # Free blocks (incl. evictable cached ones) kept for the requests being scheduled.
    reserve_frac = float(os.environ.get("VLLM_TPU_PIN_WAITING_RESERVE_FRAC", "0.25"))
    orig_schedule = Scheduler.schedule

    def schedule(self, *args, **kwargs):
        pins = self.__dict__.get("_tpu_pin_waiting")
        if pins is None:
            pins = self.__dict__["_tpu_pin_waiting"] = _Pins(max_frac)
        manager = self.kv_cache_manager
        release_done(pins, self.requests, manager.block_pool)
        keep_reserve(pins, self.waiting, manager.block_pool, reserve_frac)
        pin_new(pins, self.waiting, manager, reserve_frac)
        pins.maybe_log()
        return orig_schedule(self, *args, **kwargs)

    schedule.__wrapped__ = orig_schedule
    # Keep the flag the same-step-hit patch checks on Scheduler.schedule.
    if getattr(orig_schedule, "_tpu_same_step_prefix_hit_patch", False):
        schedule._tpu_same_step_prefix_hit_patch = True
    Scheduler.schedule = schedule
    Scheduler._tpu_pin_waiting_patch = True
    logger.info(
        "pin-waiting: on (max %.0f%% of the pool, keep %.0f%% free)",
        100 * max_frac,
        100 * reserve_frac,
    )
