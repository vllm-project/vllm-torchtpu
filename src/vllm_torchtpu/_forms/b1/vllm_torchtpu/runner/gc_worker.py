# SPDX-License-Identifier: Apache-2.0
"""The engine core's generation-2 policy in the worker processes (q38 SV-KVGC WORKER-GC).

``runner/gc_at_waits.py`` installs its generation-2 parts (threshold, passes
at waits, untracked token lists) only in the engine core: ``start_engine_gc``
runs from ``EngineCoreProc.__init__``. A worker freezes its warmed heap
(``TPUWorker.compile_or_warm_up_model``), but keeps CPython's generation-2
threshold of 10, and the generation-1 passes that ``collect_due`` runs at its
waits advance generation 2's count. So the next automatic collection, wherever
an allocation crosses generation 0's threshold, mid-step and on any thread,
becomes a generation-2 pass. Each rank does that on its own steps, and every
pass holds all ranks at the step's next collective.

``VLLM_TPU_GC_WORKER=<N>`` (default 0, off), in every worker:

- at the end of warmup, right after the worker's freeze: ``gc.freeze()``
  again (whatever warmup's tail allocated) and generation 2's threshold set to
  ``VLLM_TPU_GC_GEN2_THRESHOLD`` (1000 when that is 0), so automatic
  generation-2 passes become rare;
- a new or resumed request's ``output_token_ids`` and ``prompt_token_ids``
  lists leave the collector (``gc_at_waits.untrack_int_list``; ints only, as
  the engine core's ``VLLM_TPU_GC_UNTRACK_TOKEN_LISTS`` does for ``Request``);
- one generation-2 pass every N steps, inside the previous-results copy wait
  (``_modify_prev_results``, before ``wait_for_copy``), on the main thread.
  The step is counted there, a call every rank makes for the same steps, so
  every rank runs its pass in the same step: their stalls overlap instead of
  each rank stalling the others in turn. The engine core's period in seconds
  is not used here: each rank would reach it in a different step.

``VLLM_TPU_GC_WORKER_LOG=1`` installs only the counters and the lines (no
policy change). Lines, plain per-rank logging (never ``info_once``):
``q38 GCWORKER engaged ...`` once per worker at the end of warmup (it runs on
warm compile-cache starts too), and ``q38 GCWORKER rank=...`` every
``STATS_EVERY`` steps with every collection of each generation (count,
milliseconds, main thread or not, scheduled or automatic).
"""

import gc
import os
import threading
import time

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner import gc_at_waits

logger = init_logger(__name__)

PERIOD_STEPS = envs.VLLM_TPU_GC_WORKER
ON = PERIOD_STEPS > 0
LOG = ON or envs.VLLM_TPU_GC_WORKER_LOG
DEFAULT_GEN2_THRESHOLD = 1000
STATS_EVERY = 2000
_MAIN = threading.main_thread()


class _WorkerGC:
    """Per-worker state; stays empty unless ``start`` ran."""

    def __init__(self) -> None:
        self.started = False
        self.policy = False
        self.rank = -1
        self.local_rank = -1
        self.tp_rank = -1
        self.threshold = 0
        self.steps = 0
        self.scheduled = 0
        self.in_scheduled = False
        self.untracked = 0
        # Per generation: passes, ns, max ns, passes on the main thread,
        # passes the schedule ran.
        self.n = [0, 0, 0]
        self.ns = [0, 0, 0]
        self.max_ns = [0, 0, 0]
        self.main = [0, 0, 0]
        self.sched = [0, 0, 0]
        self.start_ns = 0
        self.start_gen = -1


_W = _WorkerGC()


def _ranks(worker) -> tuple[int, int, int]:
    rank = int(getattr(worker, "rank", -1))
    local_rank = int(getattr(worker, "local_rank", -1))
    try:
        from vllm.distributed import parallel_state

        tp_rank = int(parallel_state.get_tensor_model_parallel_rank())
    except Exception:
        tp_rank = -1
    return rank, local_rank, tp_rank


def start(worker) -> str | None:
    """End of warmup, after the worker's freeze: the policy (VLLM_TPU_GC_WORKER)
    or the counters only (VLLM_TPU_GC_WORKER_LOG), once. Returns the
    engagement line (None when off)."""
    if not LOG or _W.started:
        return None
    _W.started = True
    _W.policy = ON
    _W.rank, _W.local_rank, _W.tp_rank = _ranks(worker)
    if ON:
        gc.freeze()
        _W.threshold = envs.VLLM_TPU_GC_GEN2_THRESHOLD or DEFAULT_GEN2_THRESHOLD
        t0, t1, _ = gc.get_threshold()
        gc.set_threshold(t0, t1, _W.threshold)
    gc.callbacks.append(_callback)
    line = (
        "q38 GCWORKER engaged rank=%d local_rank=%d tp_rank=%d pid=%d policy=%d "
        "period_steps=%d gen2_threshold=%d thresholds=%s frozen=%d"
        % (
            _W.rank,
            _W.local_rank,
            _W.tp_rank,
            os.getpid(),
            int(ON),
            PERIOD_STEPS,
            gc.get_threshold()[2],
            ",".join(str(t) for t in gc.get_threshold()),
            gc.get_freeze_count(),
        )
    )
    logger.info("%s", line)
    return line


def _callback(phase: str, info: dict) -> None:
    gen = info.get("generation", -1)
    if not 0 <= gen <= 2:
        return
    now = time.perf_counter_ns()
    if phase == "start":
        _W.start_ns, _W.start_gen = now, gen
        return
    if _W.start_gen != gen:
        return
    ns = now - _W.start_ns
    _W.start_gen = -1
    _W.n[gen] += 1
    _W.ns[gen] += ns
    _W.max_ns[gen] = max(_W.max_ns[gen], ns)
    _W.main[gen] += threading.current_thread() is _MAIN
    _W.sched[gen] += _W.in_scheduled


def untrack_new(requests, scheduler_output) -> None:
    """Takes the token lists of this step's new and resumed requests out of the
    collector (``TPUModelRunner._update_states``, after the base update built
    their ``CachedRequestState``)."""
    if not _W.policy:
        return
    ids = [r.req_id for r in scheduler_output.scheduled_new_reqs]
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    ids += list(getattr(cached, "resumed_req_ids", ()) or ())
    for req_id in ids:
        state = requests.get(req_id)
        if state is None:
            continue
        for values in (state.output_token_ids, state.prompt_token_ids):
            if type(values) is list and gc.is_tracked(values):
                gc_at_waits.untrack_int_list(values)
                _W.untracked += 1


def step_wait() -> None:
    """One worker step, at the previous-results copy wait (main thread, before
    ``wait_for_copy``): the scheduled generation-2 pass when due."""
    if not _W.started or threading.current_thread() is not _MAIN:
        return
    _W.steps += 1
    if _W.policy and _W.steps % PERIOD_STEPS == 0:
        _W.in_scheduled = True
        try:
            gc.collect(2)
        finally:
            _W.in_scheduled = False
        _W.scheduled += 1
    if _W.steps % STATS_EVERY == 0:
        log_stats()


def log_stats() -> str:
    def gen(i: int) -> str:
        n = _W.n[i]
        return "g%d=%d/%.1fms/max%.2f/main%d/sched%d" % (
            i,
            n,
            _W.ns[i] / 1e6,
            _W.max_ns[i] / 1e6,
            _W.main[i],
            _W.sched[i],
        )

    line = (
        "q38 GCWORKER rank=%d tp_rank=%d pid=%d steps=%d policy=%d scheduled=%d "
        "auto_gen2=%d untracked=%d %s %s %s count=%s"
        % (
            _W.rank,
            _W.tp_rank,
            os.getpid(),
            _W.steps,
            int(_W.policy),
            _W.scheduled,
            _W.n[2] - _W.sched[2],
            _W.untracked,
            gen(0),
            gen(1),
            gen(2),
            ",".join(str(c) for c in gc.get_count()),
        )
    )
    logger.info("%s", line)
    return line
