# SPDX-License-Identifier: Apache-2.0
"""Builds a new request in the engine core's token wait (VLLM_TPU_PREPROC_IN_WAIT).

The engine core's input thread builds every new ``Request`` (vLLM
``EngineCore.preprocess_add_request``): it copies the prompt and hashes it into
prefix-cache block hashes, about 6.4 ms (p50) and 17.6 ms (p90) of pure Python
for a ~100k-token prompt (q38 SV-EC, T64n probe, descriptive). It holds the
GIL while it does that, so whatever the main thread runs at the same time
(update_from_output, schedule, reading the workers' replies) runs about two
to three times slower. At B1 decode rows that happens on about 8 % of steps
and lengthens the engine-core path that feeds the next forward by about 8 ms
on those steps.

The main thread spends most of every step blocked: about 19 ms in the
message-queue wait for the step's sampled tokens, and all of its idle time on
an empty input queue. With the key on, the input thread builds a request only
while the main thread is in one of those waits:

* The window opens when the main thread's message-queue wait has lasted
  ``OPEN_AFTER_NS`` (the short reads of the other ranks' replies never open
  it) or while it blocks on an empty input queue, and closes when that read
  returns (``SpinCondition.record_read``) or the queue yields a request.
* ``preprocess_add_request`` on the input thread first waits for the window,
  then builds the request with the engine's own block hasher applied in
  slices of ``CHUNK_TOKENS`` prompt tokens. Between slices it sleeps
  ``YIELD_S`` (releasing the GIL, so a main thread whose read just completed
  takes it at once) and waits again if the window has closed.
* The time a request spends waiting for the window is capped at the key's
  value in ms; after that the remaining slices run without waiting. The main
  thread never waits for the input thread, so nothing can deadlock.

What stays the same (checked by tests/core/test_q38_preproc_in_wait.py):
the hasher is vLLM's own ``request_block_hasher``, which is incremental by
design (it starts at ``len(request.block_hashes)``), so the slices produce the
same block hashes; the finished ``Request`` keeps the stock hasher, so it equals
the stock request field by field; the request reaches the input queue only
after it is fully built, once, in arrival order (the input thread is the same
single thread). What moves: a request can reach the scheduler later, by at
most the cap plus the slice sleeps (schedule-only). Host scheduling only:
no device program, no worker and no model input changes.

``VLLM_TPU_PREPROC_IN_WAIT=<ms>`` turns it on (0 = off, the default; nothing is
patched). ``VLLM_TPU_PREPROC_IN_WAIT_AB_MS=<P>`` (0 = off): an in-process A/B
whose windows follow wall-clock time, so the lever is on in even windows
(``time_ns // P ms`` even) and requests take the stock path in odd ones. P should
divide 86,400,000 with an even quotient. Twice the host-knobs AB period
gives a 2x2 design with ``VLLM_TPU_HOST_AB_PERIOD_MS``. Both keys are host-only
and excluded from the compile key. Engine-core process only.
"""

import threading
import time

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

MAX_DEFER_MS = envs.VLLM_TPU_PREPROC_IN_WAIT
AB_MS = envs.VLLM_TPU_PREPROC_IN_WAIT_AB_MS
# A message-queue wait opens the window once it has lasted this long.
OPEN_AFTER_NS = 500_000
# Prompt tokens hashed between two looks at the window (~0.1-0.2 ms of work).
CHUNK_TOKENS = 2048
# Sleep between slices: long enough to hand the GIL to a waiting main thread.
YIELD_S = 50e-6
# Requests between two stats lines.
LOG_EVERY = 500

_PATCH_FLAG = "_tpu_preproc_in_wait_patch"
# SpinCondition attribute: when the current wait's first wait() call ran.
_WAIT_START = "_tpu_preproc_wait_start_ns"
_MAIN = threading.main_thread()
# Set while the engine core's main thread is blocked (see the module docstring).
WINDOW = threading.Event()

stats = {
    "requests": 0,
    "gated": 0,
    "stock_ab": 0,
    "waits": 0,
    "deferred_ns": 0,
    "budget_out": 0,
    "slices": 0,
    "ab_window": None,
}


def enabled() -> bool:
    return MAX_DEFER_MS > 0


def ab_on(now_ns: int | None = None) -> bool:
    """Whether the lever is on in the current A/B window (always, without AB)."""
    if AB_MS <= 0:
        return True
    k = (time.time_ns() if now_ns is None else now_ns) // (AB_MS * 1_000_000)
    if k != stats["ab_window"]:
        stats["ab_window"] = k
        logger.info(
            "PREPROC_IN_WAIT AB window=%d on=%d requests=%d gated=%d stock=%d",
            k, int(k % 2 == 0), stats["requests"], stats["gated"], stats["stock_ab"],
        )
    return k % 2 == 0


# --- main thread: the window -------------------------------------------------


def note_wait(cond) -> None:
    """Called at each SpinCondition.wait(): opens the window once the main
    thread's current wait has lasted OPEN_AFTER_NS."""
    if threading.current_thread() is not _MAIN:
        return
    now = time.monotonic_ns()
    start = getattr(cond, _WAIT_START, 0)
    if not start:
        setattr(cond, _WAIT_START, now)
    elif now - start >= OPEN_AFTER_NS and not WINDOW.is_set():
        WINDOW.set()


def note_read(cond) -> None:
    """Called at SpinCondition.record_read(): the main thread's read returned."""
    if threading.current_thread() is not _MAIN:
        return
    setattr(cond, _WAIT_START, 0)
    WINDOW.clear()


def wrap_input_queue(q) -> None:
    """Opens the window while the main thread blocks on an empty input queue."""
    if getattr(q.get, _PATCH_FLAG, False):
        return
    orig_get = q.get

    def get(block=True, timeout=None):
        if block and threading.current_thread() is _MAIN:
            WINDOW.set()
            try:
                return orig_get(block, timeout)
            finally:
                WINDOW.clear()
        return orig_get(block, timeout)

    setattr(get, _PATCH_FLAG, True)
    get.__wrapped__ = orig_get
    q.get = get


# --- input thread: the gated build -------------------------------------------


class Gate:
    """One request's wait budget for the window."""

    def __init__(self, budget_ns: int) -> None:
        self.left = budget_ns
        self.waited = 0
        self.waits = 0

    def wait(self) -> None:
        if WINDOW.is_set() or self.left <= 0:
            return
        t0 = time.monotonic_ns()
        WINDOW.wait(self.left / 1e9)
        dt = time.monotonic_ns() - t0
        self.left -= dt
        self.waited += dt
        self.waits += 1

    def between_slices(self) -> None:
        time.sleep(YIELD_S)
        self.wait()


class _Capped:
    """A request seen by the block hasher as if it held only ``num_tokens``
    tokens and the hashes computed so far; every other attribute is the
    request's own."""

    __slots__ = ("_r", "block_hashes", "num_tokens")

    def __init__(self, request, block_hashes, num_tokens) -> None:
        object.__setattr__(self, "_r", request)
        object.__setattr__(self, "block_hashes", block_hashes)
        object.__setattr__(self, "num_tokens", num_tokens)

    def __getattr__(self, name):
        return getattr(self._r, name)


def sliced_hashes(real, request, gate: Gate) -> list:
    """``real(request)`` computed in slices of CHUNK_TOKENS tokens."""
    total = request.num_tokens
    out = list(request.block_hashes)
    cap = 0
    while cap < total:
        if cap:
            gate.between_slices()
        cap = min(total, cap + CHUNK_TOKENS)
        out.extend(real(_Capped(request, out, cap)))
        stats["slices"] += 1
    # The rest of the build (prompt array, GC untrack) runs right after: in a window.
    gate.between_slices()
    return out[len(request.block_hashes):]


class _GatedHasher:
    def __init__(self, real, gate: Gate) -> None:
        self.real = real
        self.gate = gate

    def __call__(self, request) -> list:
        return sliced_hashes(self.real, request, self.gate)


class _EngineView:
    """The engine as ``preprocess_add_request`` sees it, with a gated block hasher.
    Reads and writes of every other attribute go to the engine."""

    __slots__ = ("_engine", "request_block_hasher")

    def __init__(self, engine, hasher) -> None:
        object.__setattr__(self, "_engine", engine)
        object.__setattr__(self, "request_block_hasher", hasher)

    def __getattr__(self, name):
        return getattr(self._engine, name)

    def __setattr__(self, name, value):
        setattr(self._engine, name, value)


def make_preprocess(orig):
    def preprocess_add_request(self, request):
        real = self.request_block_hasher
        if (
            real is None
            or threading.current_thread() is _MAIN
            or not ab_on()
        ):
            if real is not None and threading.current_thread() is not _MAIN:
                stats["stock_ab"] += 1
            return orig(self, request)
        gate = Gate(MAX_DEFER_MS * 1_000_000)
        gate.wait()
        req, wave = orig(_EngineView(self, _GatedHasher(real, gate)), request)
        req._block_hasher = real
        stats["requests"] += 1
        stats["gated"] += 1
        stats["waits"] += gate.waits
        stats["deferred_ns"] += gate.waited
        stats["budget_out"] += int(gate.left <= 0)
        if stats["gated"] % LOG_EVERY == 0:
            logger.info(
                "PREPROC_IN_WAIT gated=%d stock_ab=%d waits=%d deferred_ms_mean=%.3f "
                "budget_out=%d slices=%d",
                stats["gated"], stats["stock_ab"], stats["waits"],
                stats["deferred_ns"] / 1e6 / max(1, stats["gated"]),
                stats["budget_out"], stats["slices"],
            )
        return req, wave

    setattr(preprocess_add_request, _PATCH_FLAG, True)
    preprocess_add_request.__wrapped__ = orig
    return preprocess_add_request


# --- install -----------------------------------------------------------------


def patch_preproc_in_wait() -> None:
    """engine_core stage; a no-op unless VLLM_TPU_PREPROC_IN_WAIT > 0."""
    if not enabled():
        return
    from vllm.distributed.device_communicators import shm_broadcast
    from vllm.distributed.device_communicators.shm_broadcast import SpinCondition
    from vllm.v1.engine.core import EngineCore, EngineCoreProc

    if getattr(EngineCore.preprocess_add_request, _PATCH_FLAG, False):
        return
    if getattr(shm_broadcast, "SPINLOOP_EXT_ENABLED", False):
        # Reads spin in C for up to SPINLOOP_TIMEOUT_SECONDS before the first
        # wait(), so the window would rarely open: requests would only wait.
        logger.warning(
            "VLLM_TPU_PREPROC_IN_WAIT refused: VLLM_USE_SPINLOOP_EXT is on; "
            "requests are built as before"
        )
        return
    EngineCore.preprocess_add_request = make_preprocess(
        EngineCore.preprocess_add_request
    )

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

    setattr(__init__, _PATCH_FLAG, True)
    __init__.__wrapped__ = init
    EngineCoreProc.__init__ = __init__
    logger.info(
        "VLLM_TPU_PREPROC_IN_WAIT=%d ms: new requests are built in the engine "
        "core's waits (slices of %d tokens, window after %.1f ms, AB %d ms)",
        MAX_DEFER_MS, CHUNK_TOKENS, OPEN_AFTER_NS / 1e6, AB_MS,
    )
