# SPDX-License-Identifier: Apache-2.0
"""Ship a new request's prompt token ids to the workers as an int32 numpy array (q38mm DM lane, P15).

Default off. Enable with ``VLLM_TPU_PROMPT_IDS_NDARRAY=1``.

Why: when a step admits a new request, its ``NewRequestData`` carries the full prompt as a Python list (about
94K ints at Qwen3.8 c48). The workers then pay for it twice on the step's critical path: the shm broadcast's
``pickle.loads`` builds the list (about 5.3 ms per decode->mixed boundary), and
``InputBatch.add_request`` copies the list into ``token_ids_cpu`` (``gpu_input_batch.py:371``, about 4.9 ms).
Measured on the c48-mix xprof j-ed3c3078 (20 decode->mixed boundaries, worker main thread). With an int32
array both become buffer copies.

What changes, and what does not:
  * Engine side: the engine core's input thread (``EngineCore.preprocess_add_request``) converts the request's
    prompt once, at arrival, and caches it on the ``Request`` as ``_q38_prompt_ids_np``. The ``Request``'s own
    ``prompt_token_ids`` list is untouched, so block hashing, prefix lookups, KV connectors and output
    processing see exactly what they saw before.
  * Scheduler output: after ``schedule()``, each ``NewRequestData.prompt_token_ids`` (which is the request's list
    object) is replaced by the cached array. A missing or stale cache (length or last token differs) is rebuilt
    from the list, so the shipped array always equals the list.
  * Workers: ``CachedRequestState.prompt_token_ids`` holds the array. vLLM 0.32's worker-side uses are len(),
    ``is None`` checks, integer indexing (``get_token_id``) and the ``token_ids_cpu`` row copy, all of which accept
    an array. ``token_ids_cpu`` contents are identical, so every traced program sees identical inputs.
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np

try:  # vLLM's logger when available (engine process); plain logging otherwise (tools, tests)
    from vllm.logger import init_logger

    logger = init_logger("vllm." + __name__)
except Exception:  # noqa: BLE001
    import logging

    logger = logging.getLogger(__name__)

ENV = "VLLM_TPU_PROMPT_IDS_NDARRAY"
ATTR = "_q38_prompt_ids_np"
_DTYPE = np.int32
#: Tokens per conversion chunk on the engine core's input thread. Between chunks the thread yields the GIL
#: (time.sleep(0)), so the engine core's busy loop waits at most about one chunk (~0.3 ms on e2) instead of the
#: whole conversion (5.1 ms for 94K tokens with np.asarray on e2-standard-32, M: CPU job j-0aeb11f8).
CHUNK = 8192

_lock = threading.Lock()
_installed = False
stats = {"attached": 0, "substituted": 0, "rebuilt": 0}


def enabled() -> bool:
    return os.environ.get(ENV, "0") == "1"


def to_array(prompt_token_ids) -> np.ndarray:
    """int32 copy of a prompt. numpy raises OverflowError for an id outside int32 rather than wrapping it."""
    return np.fromiter(prompt_token_ids, dtype=_DTYPE, count=len(prompt_token_ids))


def to_array_chunked(prompt_token_ids, chunk: int = CHUNK) -> np.ndarray:
    """``to_array`` in chunks, yielding the GIL between chunks (for the engine core's input thread)."""
    n = len(prompt_token_ids)
    if n <= chunk:
        return to_array(prompt_token_ids)
    out = np.empty(n, dtype=_DTYPE)
    for i in range(0, n, chunk):
        part = prompt_token_ids[i:i + chunk]
        out[i:i + len(part)] = np.fromiter(part, dtype=_DTYPE, count=len(part))
        time.sleep(0)
    return out


def attach(request) -> None:
    """Cache the int32 prompt on an engine-side Request (called once, at arrival)."""
    ids = getattr(request, "prompt_token_ids", None)
    if ids is None or isinstance(ids, np.ndarray):
        return
    setattr(request, ATTR, to_array_chunked(ids))
    stats["attached"] += 1


def _matches(arr, ids) -> bool:
    n = len(ids)
    return arr is not None and len(arr) == n and (n == 0 or int(arr[-1]) == int(ids[-1]))


def _is_tail_truncation(arr, ids) -> bool:
    """The list is the array's prefix after a tail pop (PD's Stage-3 producer): same id at the list's last position."""
    n = len(ids)
    return arr is not None and 0 < n < len(arr) and int(arr[n - 1]) == int(ids[-1])


def substitute(scheduler, scheduler_output) -> int:
    """Replace each new request's prompt list in ``scheduler_output`` by the int32 array. Returns the count."""
    n = 0
    requests = getattr(scheduler, "requests", {})
    for nrd in scheduler_output.scheduled_new_reqs:
        ids = nrd.prompt_token_ids
        if not isinstance(ids, list):
            # None, already an array, or another engine-side stand-in: PD decode's ADM-P PromptTail
            # (VLLM_TPU_PD_PROMPT_TAIL, set in NewRequestData.from_request) carries only the tail and refuses
            # reads below it. Ship it as it is: replacing it would put the whole prompt back on the wire.
            continue
        req = requests.get(nrd.req_id)
        arr = getattr(req, ATTR, None) if req is not None else None
        if not _matches(arr, ids) and _is_tail_truncation(arr, ids):
            # PD producer: the Stage-3 P side popped the prompt's tail after arrival
            # (tpu_connector, keep = the published prefix); the arrival array's prefix is the list.
            arr = arr[: len(ids)]
            if req is not None:
                setattr(req, ATTR, arr)
            stats["sliced"] = stats.get("sliced", 0) + 1
        if not _matches(arr, ids):
            arr = to_array(ids)
            if req is not None:
                setattr(req, ATTR, arr)
            stats["rebuilt"] += 1
        nrd.prompt_token_ids = arr
        n += 1
        if stats["substituted"] + n == 1 or (stats["substituted"] + n) % 1000 == 0:
            # Engagement line (one at the first substitution, then every 1000).
            logger.info(
                "%s engaged: substituted=%d attached_at_arrival=%d rebuilt_at_schedule=%d (last prompt %d tokens)",
                ENV, stats["substituted"] + n, stats["attached"], stats["rebuilt"], len(arr),
            )
    stats["substituted"] += n
    return n


def install_engine_hook() -> bool:
    """Wrap ``EngineCore.preprocess_add_request`` so the conversion runs in the engine core's input thread.

    Idempotent. Returns True if the hook is (now) installed."""
    global _installed
    with _lock:
        if _installed:
            return True
        try:
            from vllm.v1.engine.core import EngineCore
        except Exception:  # noqa: BLE001 - no engine core in this process (tests, tools)
            return False
        orig = EngineCore.preprocess_add_request

        def preprocess_add_request(self, request):
            out = orig(self, request)
            try:
                attach(out[0])
            except Exception:  # noqa: BLE001 - the scheduler-side rebuild still guarantees correctness
                pass
            return out

        preprocess_add_request.__wrapped__ = orig
        EngineCore.preprocess_add_request = preprocess_add_request
        _installed = True
        return True


_SCHED_FLAG = "_q38_prompt_ids_ndarray_patch"


def patch_pd_schedulers() -> None:
    """q38 PD port (P-PIPE-3, from MM's P15): with the flag on, wrap vLLM's base ``Scheduler.schedule`` so every PD
    scheduler (the PP producer's TpuPipelineChunkScheduler and the decode's vLLM scheduler both reach it through
    ``super().schedule``) ships new requests' prompts as int32 arrays, and convert at arrival on the engine core's
    input thread. Flag off: nothing is wrapped (the flag is read once, at engine-core start)."""
    if not enabled():
        return
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler, _SCHED_FLAG, False):
        return
    orig = Scheduler.schedule

    def schedule(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        substitute(self, out)
        return out

    schedule.__wrapped__ = orig
    Scheduler.schedule = schedule
    setattr(Scheduler, _SCHED_FLAG, True)
    hooked = install_engine_hook()
    logger.info("%s active on PD schedulers (engine input-thread conversion hook: %s)", ENV,
                "installed" if hooked else "not available, converting at schedule time")
