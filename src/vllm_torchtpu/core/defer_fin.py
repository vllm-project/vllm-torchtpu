# SPDX-License-Identifier: Apache-2.0
"""Frees a finished request's KV blocks, and drops the request, in the engine core's token wait (VLLM_TPU_DEFER_FIN).

When a decode request finishes, vLLM's ``Scheduler.update_from_output`` frees it on the engine core's main thread, on the
path between a step's sampled tokens and the next step's dispatch:
``_free_request`` -> ``_free_blocks`` -> ``_free_request_blocks`` (the KV cache manager frees every group's blocks,
about 0.67 ms per request at B1 decode rows), and when ``update_from_output`` returns, the last reference to the
``Request`` goes away and its token lists (~100k ints) are deallocated (about 1.7 ms per finish step; q38 SV-EC, T64n
probe, descriptive). Both sit on the critical path of the finish steps (~17 % of B1 steps).

With the key on, a ``_free_request_blocks`` call made inside ``update_from_output`` on the main thread is queued with the
request instead of run. The queue is worked off while the main thread is blocked anyway:

* one queued request per ``SpinCondition.wait()`` call once the main thread's current message-queue wait has lasted
  ``OPEN_AFTER_NS`` (the long wait for a step's tokens; at B1 rows 90 % of waits that reach 2 ms last 8 ms more);
* everything, before the main thread blocks on an empty input queue;
* at the start of ``schedule()``, every request queued ``MAX_LAG`` or more schedules ago, and any request whose id is
  back in ``scheduler.requests`` (so blocks never lag more than ``MAX_LAG`` steps).

Working off an entry calls the stock ``_free_request_blocks`` (all installed free-path patches included), then drops
the queue's reference, so the request is deallocated there. Everything else in ``_free_request`` runs as before:
the connector's ``request_finished`` (its transfer params go out with the request's last output), ``finished_req_ids``
and ``del self.requests[...]``. Frees made outside ``update_from_output`` (preemption, aborts) are not deferred.

What moves: the finished request's blocks return to the block pool up to ``MAX_LAG`` schedules later, so a later
allocation or eviction can pick different blocks (schedule-only; no request's tokens change); the request object is
deallocated later (no effect). Host scheduling only: no device program, worker or model input changes.

``VLLM_TPU_DEFER_FIN=1`` turns it on (default off: nothing is patched). ``VLLM_TPU_DEFER_FIN_AB_MS=<P>`` (0 = off):
in-process A/B in wall-clock windows (on in even ``time_ns // P ms`` windows; requests finishing in odd windows are
freed the stock way). Both keys are host-only and excluded from the compile key. Engine-core process only.
"""

import collections
import threading
import time

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

ENABLED = envs.VLLM_TPU_DEFER_FIN
AB_MS = envs.VLLM_TPU_DEFER_FIN_AB_MS
# A message-queue wait starts working the queue off once it has lasted this long.
OPEN_AFTER_NS = 2_000_000
# A queued free runs at the latest at the schedule() this many schedules after it was queued.
MAX_LAG = 2
LOG_EVERY = 500

_PATCH_FLAG = "_tpu_defer_fin_patch"
_WAIT_START = "_tpu_defer_fin_wait_start_ns"
_MAIN = threading.main_thread()
# (scheduler, request, schedule count when queued)
PENDING = collections.deque()
_ORIG = {}
_state = {"in_update": 0, "seq": 0, "ab_window": None}
stats = {"deferred": 0, "stock_ab": 0, "in_wait": 0, "idle": 0, "stale": 0, "readded": 0}


def enabled() -> bool:
    return bool(ENABLED)


def ab_on(now_ns: int | None = None) -> bool:
    """Whether the lever is on in the current A/B window (always, without AB)."""
    if AB_MS <= 0:
        return True
    k = (time.time_ns() if now_ns is None else now_ns) // (AB_MS * 1_000_000)
    if k != _state["ab_window"]:
        _state["ab_window"] = k
        logger.info(
            "DEFER_FIN AB window=%d on=%d deferred=%d stock=%d pending=%d",
            k, int(k % 2 == 0), stats["deferred"], stats["stock_ab"], len(PENDING),
        )
    return k % 2 == 0


def _work_off(cause: str) -> None:
    sched, request, _ = PENDING.popleft()
    _ORIG["free"](sched, request)
    stats[cause] += 1
    del sched, request  # the request is deallocated here when this was its last reference


def flush_all(cause: str = "idle") -> None:
    while PENDING:
        _work_off(cause)


def flush_due(sched) -> None:
    """At schedule() start: entries queued MAX_LAG or more schedules ago, and re-added request ids."""
    while PENDING and _state["seq"] - PENDING[0][2] >= MAX_LAG:
        _work_off("stale")
    if PENDING and any(r.request_id in sched.requests for _, r, _ in PENDING):
        keep = collections.deque()
        while PENDING:
            item = PENDING.popleft()
            if item[1].request_id in sched.requests:
                _ORIG["free"](item[0], item[1])
                stats["readded"] += 1
            else:
                keep.append(item)
        PENDING.extend(keep)


def note_wait(cond) -> None:
    """At each SpinCondition.wait(): once the main thread's wait has lasted OPEN_AFTER_NS, work off one entry."""
    if threading.current_thread() is not _MAIN:
        return
    now = time.monotonic_ns()
    start = getattr(cond, _WAIT_START, 0)
    if not start:
        setattr(cond, _WAIT_START, now)
    elif PENDING and now - start >= OPEN_AFTER_NS:
        _work_off("in_wait")


def note_read(cond) -> None:
    if threading.current_thread() is _MAIN:
        setattr(cond, _WAIT_START, 0)


def wrap_input_queue(q) -> None:
    """Works everything off before the main thread blocks on an empty input queue."""
    if getattr(q.get, _PATCH_FLAG, False):
        return
    orig_get = q.get

    def get(block=True, timeout=None):
        if block and threading.current_thread() is _MAIN and PENDING and q.empty():
            flush_all("idle")
        return orig_get(block, timeout)

    setattr(get, _PATCH_FLAG, True)
    get.__wrapped__ = orig_get
    q.get = get


def make_free_request_blocks(orig):
    def _free_request_blocks(self, request):
        if _state["in_update"] and threading.current_thread() is _MAIN:
            if ab_on():
                PENDING.append((self, request, _state["seq"]))
                stats["deferred"] += 1
                if stats["deferred"] % LOG_EVERY == 0:
                    logger.info(
                        "DEFER_FIN deferred=%d stock_ab=%d in_wait=%d idle=%d stale=%d readded=%d pending=%d",
                        stats["deferred"], stats["stock_ab"], stats["in_wait"], stats["idle"], stats["stale"],
                        stats["readded"], len(PENDING),
                    )
                return
            stats["stock_ab"] += 1
        return orig(self, request)

    setattr(_free_request_blocks, _PATCH_FLAG, True)
    _free_request_blocks.__wrapped__ = orig
    return _free_request_blocks


def make_update_from_output(orig):
    def update_from_output(self, *args, **kwargs):
        _state["in_update"] += 1
        try:
            return orig(self, *args, **kwargs)
        finally:
            _state["in_update"] -= 1

    setattr(update_from_output, _PATCH_FLAG, True)
    update_from_output.__wrapped__ = orig
    return update_from_output


def make_schedule(orig):
    def schedule(self, *args, **kwargs):
        _state["seq"] += 1
        if PENDING:
            flush_due(self)
        return orig(self, *args, **kwargs)

    setattr(schedule, _PATCH_FLAG, True)
    schedule.__wrapped__ = orig
    return schedule


def patch_defer_fin() -> None:
    """engine_core stage; a no-op unless VLLM_TPU_DEFER_FIN=1."""
    if not enabled():
        return
    from vllm.distributed.device_communicators import shm_broadcast
    from vllm.distributed.device_communicators.shm_broadcast import SpinCondition
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.core import EngineCoreProc

    if getattr(Scheduler._free_request_blocks, _PATCH_FLAG, False):
        return
    if getattr(shm_broadcast, "SPINLOOP_EXT_ENABLED", False):
        # Reads spin in C before the first wait(); the queue would only drain at schedule() / idle.
        logger.warning("VLLM_TPU_DEFER_FIN refused: VLLM_USE_SPINLOOP_EXT is on; finished requests are freed as before")
        return
    _ORIG["free"] = Scheduler._free_request_blocks
    Scheduler._free_request_blocks = make_free_request_blocks(Scheduler._free_request_blocks)
    Scheduler.update_from_output = make_update_from_output(Scheduler.update_from_output)
    Scheduler.schedule = make_schedule(Scheduler.schedule)

    orig_wait = SpinCondition.wait
    orig_record_read = SpinCondition.record_read

    def wait(self, timeout_ms=None):
        note_wait(self)
        return orig_wait(self, timeout_ms)

    def record_read(self):
        note_read(self)
        return orig_record_read(self)

    for fn, orig in ((wait, orig_wait), (record_read, orig_record_read)):
        setattr(fn, _PATCH_FLAG, True)
        fn.__wrapped__ = orig
    SpinCondition.wait = wait
    SpinCondition.record_read = record_read

    init = EngineCoreProc.__init__

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        wrap_input_queue(self.input_queue)
        # The engagement line: logged by the engine core process once its engine is built (patch-time lines are logged
        # before its logging reaches the served log).
        logger.info(
            "DEFER_FIN engaged: finished requests freed in the engine core's waits "
            "(after %.1f ms of a wait, at idle, or within %d schedules; AB %d ms)",
            OPEN_AFTER_NS / 1e6, MAX_LAG, AB_MS,
        )

    setattr(__init__, _PATCH_FLAG, True)
    __init__.__wrapped__ = init
    EngineCoreProc.__init__ = __init__
    logger.info(
        "VLLM_TPU_DEFER_FIN=1: finished requests are freed in the engine core's waits "
        "(after %.1f ms of a wait, at idle, or within %d schedules; AB %d ms)",
        OPEN_AFTER_NS / 1e6, MAX_LAG, AB_MS,
    )
