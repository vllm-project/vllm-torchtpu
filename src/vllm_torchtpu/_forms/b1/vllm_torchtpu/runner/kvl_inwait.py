# SPDX-License-Identifier: Apache-2.0
"""Repack planning inside the worker's host waits, Stage-3 submit pass first (q38 SV-KVGC KVL-INWAIT).

A staged PD consumer (``pd_staging.StagedConsumer``) lands each load in a
staging carrier and repacks it into the pool in the step the scheduler names
it in (QWEN4_EXP_PD_SYNC_REPACK; after the forward with LAND-OVERLAP). With
QWEN4_EXP_PD_DEFER_LOAD_SUBMIT that step's ``kv_load_submit`` runs, on the main
thread between the forward's dispatch and the previous-results wait: the
deferred repacks (planning, ``plan_repack``: numpy index rows and chunks; then
the dispatch, ``run_repack``), then the Stage-3 submit pass, whose batched
admission is a gloo MIN all-reduce over the TP CPU group. When that burst is
longer than the wait it overruns into the next forward's launch.

``VLLM_TPU_KVL_INWAIT=1`` (default off), on every rank of a staged consumer:

- **Plan units.** A load that landed on this rank (``_stage3_finish_destination``)
  is queued; a unit plans one queued load (``plan_repack_ahead``) on the main
  thread inside a host wait the worker already makes: the previous-results
  copy wait while its event is pending (``AsyncTPUCopyState._wait``), the
  message-queue wait for the next RPC (``SpinCondition.wait``, one unit per
  call) and the batched admission's all-reduce while it is in flight
  (``stage3_admission.prepare(wait_hook=)``). The named step's repack runs
  the stored plan (``repack_planned``) at today's point; a load without a
  valid plan plans there, as today. A plan is valid only for the very
  ``_pending_repacks`` entry it was made from (an aborted, dropped or
  re-landed load is planned again).
- **Submit first.** ``submit_deferred_loads`` runs the step's Stage-3 submit
  pass before its deferred repacks, so no rank's repack delays the pass's
  collective for the others and the step's named loads can be planned while
  that collective is in flight. Released staging pages stay quarantined
  (``QWEN4_EXP_PD_STAGING_QUARANTINE_STEPS`` >= 1, required), so the pass sees
  the same free map in either order; it reserves only free pages, which no
  landed load holds; it enqueues no device work. The loads' transfers start
  earlier, never later.

What is the same as without the key (C): every scheduler decision, every
collective and its order on every rank, every device program, its inputs and
its order on the device (the repack programs are dispatched at the same point
of the same step from the same plan). What differs: when this rank computes a
plan (its own host time; no other rank can observe it) and the order of two
host blocks inside ``submit_deferred_loads``. Refused (ValueError at engine
start) under QWEN4_EXP_PD_SHARED_COMPRESSED (its landing runs TP collectives),
without QWEN4_EXP_PD_SYNC_REPACK, or with a zero quarantine.

Lines, plain per-rank logging: ``q38 KVL_INWAIT engaged ...`` at engine start
(end of the connector's native registration; warm compile caches included)
and ``q38 KVL_INWAIT rank=...`` every ``STATS_EVERY`` connector steps.
"""

import os
import threading
import time
from collections import deque

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.gc_at_waits import event_pending

logger = init_logger(__name__)

ENABLED = envs.VLLM_TPU_KVL_INWAIT
# Set by ``engage`` in the engaged worker process only.
ACTIVE = False
STATS_EVERY = 2000
_MAIN = threading.main_thread()
_PATCH_FLAG = "_tpu_kvl_inwait_patch"
WINDOWS = ("copy", "mq", "adm")


class _State:
    def __init__(self) -> None:
        self.worker = None
        self.rank = -1
        self.tp_rank = -1
        self.queue: deque = deque()
        # req_id -> (the _pending_repacks entry planned, PlannedRepack)
        self.plans: dict = {}
        self.steps = 0
        self.landed = 0
        self.planned = dict.fromkeys(WINDOWS, 0)
        self.overrun = dict.fromkeys(WINDOWS, 0)
        self.plan_ns = 0
        self.plan_max_ns = 0
        self.used = 0
        self.inline = 0
        self.stale = 0
        self.submit_first = 0


_S = _State()


def refusals(worker) -> list[str]:
    """Why a staged consumer cannot run the key (empty: it can)."""
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_staging import quarantine_steps

    why = []
    staging = getattr(worker, "_staging", None)
    if getattr(staging, "shared", None) is not None:
        why.append("QWEN4_EXP_PD_SHARED_COMPRESSED lands loads with TP collectives")
    if not getattr(worker, "_sync_repack", False):
        why.append("needs QWEN4_EXP_PD_SYNC_REPACK (repack in the named step)")
    if quarantine_steps() <= 0:
        why.append(
            "needs QWEN4_EXP_PD_STAGING_QUARANTINE_STEPS >= 1 (the submit pass "
            "must see the same free map before and after the step's repacks)"
        )
    return why


def engage(worker) -> str | None:
    """Engine start, every rank (the end of the Qwen PD worker's native
    registration). Same decision on every rank: the key and the configuration
    are. Returns the engagement line, or None when the key is off."""
    global ACTIVE
    if not ENABLED:
        return None
    rank = _global_rank()
    tp_rank = int(getattr(worker, "tp_rank", -1))
    if getattr(worker, "_staging", None) is None:
        line = (
            "q38 KVL_INWAIT engaged=0 rank=%d tp_rank=%d pid=%d reason=not a staged consumer"
            % (rank, tp_rank, os.getpid())
        )
        logger.info("%s", line)
        return line
    why = refusals(worker)
    if why:
        raise ValueError("VLLM_TPU_KVL_INWAIT is refused: " + "; ".join(why))
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_staging import quarantine_steps

    _S.worker, _S.rank, _S.tp_rank = worker, rank, tp_rank
    install_mq_hook()
    ACTIVE = True
    line = (
        "q38 KVL_INWAIT engaged=1 rank=%d tp_rank=%d pid=%d windows=%s submit_first=1 "
        "quarantine=%d batched_admission=%d landx=%d defer_submit=%d"
        % (
            rank,
            tp_rank,
            os.getpid(),
            ",".join(WINDOWS),
            quarantine_steps(),
            int(bool(envs.QWEN4_EXP_PD_BATCHED_ADMISSION)),
            int(bool(envs.QWEN4_EXP_PD_LANDX_DEFER_REPACK)),
            int(bool(envs.QWEN4_EXP_PD_DEFER_LOAD_SUBMIT)),
        )
    )
    logger.info("%s", line)
    return line


def _global_rank() -> int:
    try:
        import torch.distributed as dist

        return int(dist.get_rank()) if dist.is_initialized() else -1
    except Exception:
        return -1


# --- the queue and the plans -------------------------------------------------


def enqueue(req_id) -> None:
    """A load landed on this rank and waits for its named step."""
    _S.landed += 1
    _S.queue.append(req_id)


def prioritize(names) -> None:
    """This step's deferred names (start_load_kv) go to the front of the queue,
    so the admission collective's wait plans them before their repack."""
    front = [n for n in names if n not in _S.plans]
    if not front:
        return
    rest = [n for n in _S.queue if n not in front]
    _S.queue = deque(front + rest)


def _plan_one(req_id, window: str) -> bool:
    w = _S.worker
    pending = w._pending_repacks.get(req_id)
    if pending is None:
        _S.plans.pop(req_id, None)
        return False
    held = _S.plans.get(req_id)
    if held is not None and held[0] is pending:
        return False
    uuid, tables, num_tokens = pending
    t0 = time.perf_counter_ns()
    planned = w._staging.plan_repack_ahead(uuid, tables, num_tokens)
    ns = time.perf_counter_ns() - t0
    _S.plans[req_id] = (pending, planned)
    _S.planned[window] += 1
    _S.plan_ns += ns
    _S.plan_max_ns = max(_S.plan_max_ns, ns)
    return True


def run_units(more, window: str, limit: int | None = None) -> int:
    """Plan units on the main thread: before each, ``more()`` (None: always)
    must still be true; at most ``limit``. A unit that ends when ``more()`` has
    turned false is counted as an overrun of its window."""
    if not ACTIVE or threading.current_thread() is not _MAIN:
        return 0
    done = 0
    while _S.queue and (limit is None or done < limit):
        if more is not None and not more():
            break
        if not _plan_one(_S.queue.popleft(), window):
            continue
        done += 1
        if more is not None and not more():
            _S.overrun[window] += 1
    return done


def take(req_id, pending):
    """The named step's repack (``repack_landed``): the stored plan of this very
    ``_pending_repacks`` entry, or None (the caller plans inline, as today)."""
    held = _S.plans.pop(req_id, None)
    if held is None:
        _S.inline += 1
        return None
    if held[0] is not pending:
        _S.stale += 1
        return None
    _S.used += 1
    return held[1]


# --- the windows -------------------------------------------------------------


def at_copy_wait(event) -> None:
    """``AsyncTPUCopyState._wait``: units while the copy's event is pending."""
    run_units(lambda: event_pending(event), "copy")


def admission_wait_hook():
    """``stage3_admission.prepare``'s wait hook: one unit per call."""
    if not ACTIVE:
        return None

    def hook() -> bool:
        return run_units(None, "adm", limit=1) > 0

    return hook


def install_mq_hook() -> None:
    """``SpinCondition.wait`` (a message-queue read with no message ready): one
    unit per call, in this worker process."""
    from vllm.distributed.device_communicators.shm_broadcast import SpinCondition

    orig = SpinCondition.wait
    if getattr(orig, _PATCH_FLAG, False):
        return

    def wait(self, timeout_ms=None):
        if ACTIVE and _S.queue:
            run_units(None, "mq", limit=1)
        return orig(self, timeout_ms)

    setattr(wait, _PATCH_FLAG, True)
    wait.__wrapped__ = orig
    SpinCondition.wait = wait


# --- per step ----------------------------------------------------------------


def note_submit_first() -> None:
    _S.submit_first += 1


def step() -> None:
    """Once per connector step (start_load_kv), every rank."""
    _S.steps += 1
    if _S.steps % 64 == 0 and _S.plans:
        pending = _S.worker._pending_repacks
        for req_id in [r for r, (p, _) in _S.plans.items() if pending.get(r) is not p]:
            del _S.plans[req_id]
    if _S.steps % STATS_EVERY == 0:
        log_stats()


def log_stats() -> str:
    planned = sum(_S.planned.values())
    line = (
        "q38 KVL_INWAIT rank=%d tp_rank=%d pid=%d steps=%d landed=%d planned_copy=%d "
        "planned_mq=%d planned_adm=%d used=%d inline=%d stale=%d plan_ms_mean=%.3f "
        "plan_ms_max=%.3f overrun_copy=%d overrun_adm=%d submit_first=%d queue=%d plans=%d"
        % (
            _S.rank,
            _S.tp_rank,
            os.getpid(),
            _S.steps,
            _S.landed,
            _S.planned["copy"],
            _S.planned["mq"],
            _S.planned["adm"],
            _S.used,
            _S.inline,
            _S.stale,
            _S.plan_ns / max(1, planned) / 1e6,
            _S.plan_max_ns / 1e6,
            _S.overrun["copy"],
            _S.overrun["adm"],
            _S.submit_first,
            len(_S.queue),
            len(_S.plans),
        )
    )
    logger.info("%s", line)
    return line
