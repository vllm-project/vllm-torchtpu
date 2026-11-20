# SPDX-License-Identifier: Apache-2.0
"""Runs due garbage collections while the main thread waits (VLLM_TPU_GC_AT_WAITS).

CPython collects generation 0 when its allocation count passes the threshold
(700), wherever that allocation happens. In the decode, that is mostly on the
critical path between a step's output and the next step's forward: each young
96k-token request list makes a generation-0 pass cost about 1-3 ms, and every
tenth pass is a generation-1 pass of about 20 ms. Both processes on that path
also wait every step: the engine core for the worker's output, a worker for
the next SchedulerOutput and for the previous step's sampled-token copy.

At each of those waits, when the generation-0 count has reached half the
threshold, this runs the collection CPython would run next (generation 1 when
its count is one short of its threshold, else generation 0). The waits keep
the count below the threshold, so automatic collections become rare; they
stay enabled, and generation 2 stays with CPython's own policy, which the
generation-1 passes here advance as usual. Only the main thread collects: a
collection on another thread holds the GIL against the main thread.

Waits: ``SpinCondition.wait`` (a shared-memory message queue reader with no
message ready; engine core and workers, patched here) and, on workers,
``AsyncTPUCopyState.wait`` when its ready event has not fired (it calls
``collect_due`` itself).
"""

import functools
import gc
import threading

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

ENABLED = envs.VLLM_TPU_GC_AT_WAITS
_PATCH_FLAG = "_tpu_gc_at_waits_patch"
_MAIN = threading.main_thread()
# Collections run here, by generation.
counts = [0, 0]


def collect_due() -> None:
    """Runs the next due collection when called on the main thread."""
    c0, c1, _ = gc.get_count()
    t0, t1, _ = gc.get_threshold()
    if t0 == 0 or c0 < t0 // 2 or threading.current_thread() is not _MAIN:
        return
    generation = 1 if c1 + 1 >= t1 else 0
    gc.collect(generation)
    counts[generation] += 1


def event_pending(event) -> bool:
    """Whether a torch.tpu.Event has not completed; False when unknown."""
    try:
        return not event.query()
    except Exception:
        return False


def patch_gc_at_waits() -> None:
    if not ENABLED:
        return
    from vllm.distributed.device_communicators.shm_broadcast import SpinCondition

    if getattr(SpinCondition.wait, _PATCH_FLAG, False):
        return
    orig_wait = SpinCondition.wait

    def wait(self, timeout_ms=None):
        collect_due()
        return orig_wait(self, timeout_ms)

    functools.update_wrapper(wait, orig_wait)
    setattr(wait, _PATCH_FLAG, True)
    SpinCondition.wait = wait
    logger.info("VLLM_TPU_GC_AT_WAITS: due collections run at message-queue waits")
