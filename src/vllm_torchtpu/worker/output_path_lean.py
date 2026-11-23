# SPDX-License-Identifier: Apache-2.0
"""Leaner worker output path on the output rank (VLLM_TPU_OUTPUT_PATH_LEAN=1).

Under async scheduling the output rank (TP rank 0) hands every RPC result to
``WorkerProc.handle_output``, which puts it on ``async_output_queue``. The
``WorkerAsyncOutputCopy`` thread runs ``WorkerProc.enqueue_output`` on each
item: ``AsyncModelRunnerOutput.get_output()`` for ``sample_tokens`` results,
then ``MessageQueue.enqueue`` (pickle + shared-memory write). That thread holds
the GIL while the step thread works. This module makes two cuts. Both send the
engine the same responses, in the same order, with the same bytes.

1. ``execute_model``'s ``None`` result goes inline, on the step thread, when
   the output thread is idle. In every other case it is queued as before.

   Ordering argument. The engine reads one worker's responses in FIFO order:
   ``MultiprocExecutor.collective_rpc`` makes one ``FutureWrapper`` per RPC,
   and ``FutureWrapper.result`` drains the futures oldest first, each with one
   ``MessageQueue.dequeue``. So the n-th message on the response ring is the
   n-th RPC's answer. Skipping a message would hand every later RPC the
   previous RPC's answer, so it is not allowed. The only writers of the ring
   are the step thread (inline) and the output thread. The step thread is the
   only producer of queue items, and it counts them (``puts``). The output
   thread counts an item (``done``) only after ``enqueue_output`` has written
   it to the ring. If ``puts == done`` on the step thread, every earlier
   response is already on the ring and the output thread is blocked in
   ``get()``. No other item can appear meanwhile, because only this thread
   puts. Writing the ``None`` response now therefore puts it right after all
   earlier responses and before every later one: the same position it would
   have had in the queue. If ``puts != done`` (for example the previous
   step's ``sample_tokens`` output is still waiting for its device copy), the
   ``None`` is queued behind it as before. Each counter has one writer and
   ints are replaced atomically under the GIL, so no lock is needed. If the
   output thread dies on an exception, ``done`` stops, so every later item is
   queued, as before. If the thread runs vLLM's own loop (the patch came too
   late), ``loop_ok`` stays False and nothing goes inline.

2. ``MessageQueue.enqueue`` for the worker's response ring is replaced by a
   writer that keeps its pickler and output buffer across calls, sends the
   constant ``(SUCCESS, None)`` frame from a cache, and takes the free-slot
   fast path of ``acquire_write`` without its two nested context managers.
   It writes the same bytes, and sets the ring flags in the same order with
   the same memory fences. Every case outside the fast path calls vLLM's
   original ``enqueue``: payloads that do not fit a ring chunk (overflow over
   the socket), out-of-band pickle buffers (>= 1 MiB), remote readers, the
   shared-memory tensor arena, a full ring slot (the original waits, spins,
   times out and warns), and a concurrent caller on the same queue. The
   reused output buffer is truncated before every pickle, so a short payload
   after a long one has only its own bytes, and the buffer is dropped after a
   payload larger than ``_KEEP_BYTES``. The pickler's memo is cleared before
   every dump, and the dispatch table is rebuilt when
   ``copyreg.dispatch_table`` changes, so the pickle is the one a new pickler
   would write.

Installed only when the vLLM sources it replaces have the sha256 prefixes in
``_WORKER_SOURCE_SHA`` / ``_MQ_SOURCE_SHA`` (vLLM v0.31.0 and v0.32.0).
Otherwise it logs once and leaves that part unpatched. Flag off: nothing is
installed.
"""

import copyreg
import functools
import hashlib
import inspect
import io
import os
import pickle
import threading
from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_MARK = "_tpu_output_path_lean"
_STATE = "_tpu_lean_state"
_MQ_STATE = "_tpu_lean_mq_state"

# sha256 prefixes of the vLLM sources this module stands in for. The
# WorkerProc methods are the same in vLLM v0.31.0 and v0.32.0;
# MessageQueue.enqueue differs only in the v0.32.0 tensor arena, which the
# fast path never takes (an arena queue falls back to the original).
_WORKER_SOURCE_SHA = {
    "enqueue_output": ("ddc1dfcb053153dd",),
    "handle_output": ("213f7745442a9acf",),
    "async_output_busy_loop": ("ad786ba52689de18",),
}
_MQ_SOURCE_SHA = {
    ("MessageQueue", "enqueue"): ("8d00db0328000505", "57f1032296f82718"),
    ("MessageQueue", "acquire_write"): ("3f36794888c221a6",),
    ("ShmRingBuffer", "get_data"): ("16a9b202e5681f92",),
    ("ShmRingBuffer", "get_metadata"): ("e1354eb3e88b076b",),
}

# Keep the reused output buffer only up to this size after a dump.
_KEEP_BYTES = 256 * 1024
# vLLM's MessageQueue.enqueue: buffers below this are pickled in-band.
_OOB_INLINE_BYTES = 1024 * 1024
# One INFO line every this many queued sample_tokens items (rank 0 only).
_LOG_EVERY = 50_000


def enabled() -> bool:
    from vllm_torchtpu import envs

    try:
        return bool(envs.VLLM_TPU_OUTPUT_PATH_LEAN)
    except Exception:
        return False


def _source_sha(fn: Any) -> str | None:
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return None
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def worker_sources_match(cls: type) -> bool:
    for name, want in _WORKER_SOURCE_SHA.items():
        fn = cls.__dict__.get(name)
        # Another patch already wrapped it: do not stack on top.
        if fn is None or hasattr(fn, "__wrapped__") or _source_sha(fn) not in want:
            return False
    return True


def mq_sources_match() -> bool:
    try:
        from vllm.distributed.device_communicators import shm_broadcast
    except Exception:
        return False
    for (cls_name, name), want in _MQ_SOURCE_SHA.items():
        cls = getattr(shm_broadcast, cls_name, None)
        fn = getattr(cls, name, None)
        if fn is None or _source_sha(fn) not in want:
            return False
    # The original enqueue must not be wrapped already (we call it as the
    # fallback and replicate it on the fast path).
    return not hasattr(
        shm_broadcast.MessageQueue.__dict__.get("enqueue"), "__wrapped__"
    )


# --------------------------------------------------------------------------
# (2) MessageQueue writer
# --------------------------------------------------------------------------


class _MqWriter:
    """Per-queue state of the lean writer. One writer at a time (``lock``,
    taken non-blocking; a second concurrent caller uses the original)."""

    __slots__ = (
        "lock",
        "bio",
        "pickler",
        "base_table",
        "oob",
        "none_frame",
        "none_key",
        "fast",
        "fallback",
    )

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.bio: io.BytesIO | None = None
        self.pickler: pickle.Pickler | None = None
        self.base_table: dict | None = None
        self.oob = False
        self.none_frame: bytes | None = None
        self.none_key: Any = None
        self.fast = 0
        self.fallback = 0

    def _buffer_callback(self, buf: pickle.PickleBuffer) -> bool:
        # Same rule as vLLM's oob_callback: in-band below 1 MiB. A larger
        # buffer makes this dump unusable; the caller re-sends it through
        # the original enqueue, which emits it out of band.
        if len(buf.raw()) < _OOB_INLINE_BYTES:
            return True
        self.oob = True
        return True

    def _ensure_pickler(self, shm_broadcast: Any) -> pickle.Pickler:
        base = copyreg.dispatch_table
        if self.pickler is None or base != self.base_table:
            import torch

            # The same table vLLM builds per call.
            table = dict(base)
            table[torch.Tensor] = shm_broadcast._reduce_tensor
            self.bio = io.BytesIO()
            pickler = pickle.Pickler(
                self.bio,
                protocol=pickle.HIGHEST_PROTOCOL,
                buffer_callback=self._buffer_callback,
            )
            pickler.dispatch_table = table
            self.pickler = pickler
            self.base_table = dict(base)
            self.none_frame = None
        return self.pickler

    def dumps(self, obj: Any, shm_broadcast: Any) -> bytes | None:
        """Pickle ``obj`` exactly as vLLM's enqueue does. None if it needs
        out-of-band buffers."""
        pickler = self._ensure_pickler(shm_broadcast)
        bio = self.bio
        bio.seek(0)
        bio.truncate()
        self.oob = False
        pickler.clear_memo()
        pickler.dump(obj)
        if self.oob:
            self._drop_if_large()
            return None
        data = bio.getvalue()
        self._drop_if_large()
        return data

    def _drop_if_large(self) -> None:
        if self.bio is not None and self.bio.tell() > _KEEP_BYTES:
            # Rebuilt on the next call; a huge payload is not kept alive.
            self.pickler = None
            self.bio = None


def _frame(payload: bytes) -> bytes:
    # vLLM's in-ring layout for one buffer: byte 0 = 0 (not overflow),
    # bytes 1-2 = buffer count (1), then the buffer preceded by its 4-byte
    # big-endian length.
    return b"\x00\x00\x01" + len(payload).to_bytes(4, "big") + payload


def _writer_of(mq: Any) -> "_MqWriter | None":
    w = mq.__dict__.get(_MQ_STATE)
    if w is None:
        ok = (
            getattr(mq, "_is_writer", False)
            and getattr(mq, "n_local_reader", 0) > 0
            and getattr(mq, "n_remote_reader", 1) == 0
            and getattr(mq, "tensor_arena", None) is None
            and getattr(mq, "buffer", None) is not None
            and getattr(mq, "_spin_condition", None) is not None
        )
        w = _MqWriter() if ok else False
        mq.__dict__[_MQ_STATE] = w
    return w or None


_shm_broadcast: Any = None


def _load_shm_broadcast() -> Any:
    global _shm_broadcast
    from vllm.distributed.device_communicators import shm_broadcast

    _shm_broadcast = shm_broadcast
    return shm_broadcast


def lean_enqueue(
    mq: Any, obj: Any, timeout: float | None = None, none_status: Any = None
) -> None:
    """``MessageQueue.enqueue(obj, timeout)`` with the same bytes on the ring.

    ``none_status``: when set, ``obj`` is ``(none_status, None)`` and its frame
    is served from a cache.
    """
    shm_broadcast = _shm_broadcast or _load_shm_broadcast()
    orig = shm_broadcast.MessageQueue.enqueue
    w = _writer_of(mq)
    if w is None or not w.lock.acquire(blocking=False):
        return orig(mq, obj, timeout)
    try:
        frame = None
        if none_status is not None:
            # Valid while the pickler (and so the dispatch table) is.
            w._ensure_pickler(shm_broadcast)
            if w.none_frame is not None and w.none_key is none_status:
                frame = w.none_frame
        if frame is None:
            payload = w.dumps(obj, shm_broadcast)
            # vLLM's overflow test: 6 header bytes + payload.
            if payload is None or 6 + len(payload) >= mq.buffer.max_chunk_bytes:
                w.fallback += 1
                return orig(mq, obj, timeout)
            frame = _frame(payload)
            if none_status is not None:
                w.none_frame = frame
                w.none_key = none_status
        if not _write_fast(mq, frame, shm_broadcast):
            # Slot not free yet: the original waits for it (spin, sched_yield,
            # timeout, warning) and writes the same bytes.
            w.fallback += 1
            return orig(mq, obj, timeout)
        w.fast += 1
    finally:
        w.lock.release()


def _write_fast(mq: Any, frame: bytes, shm_broadcast: Any) -> bool:
    """``acquire_write``'s no-wait path plus the frame write. False (nothing
    written) when the current slot is still unread."""
    ring = mq.buffer
    shm_buf = ring.shared_memory.buf
    if shm_buf is None:
        return False
    fence = shm_broadcast.memory_fence
    idx = mq.current_idx
    n_reader = ring.n_reader
    m0 = ring.metadata_offset + idx * ring.metadata_size
    md = shm_buf[m0 : m0 + ring.metadata_size]
    try:
        fence()
        if md[0] and sum(md[1:]) != n_reader:
            return False
        # mark the block as not written
        md[0] = 0
        d0 = ring.data_offset + idx * ring.max_chunk_bytes
        n = len(frame)
        with shm_buf[d0 : d0 + n] as data:
            data[:] = frame
        # first the read flags, then the written flag (see acquire_write)
        for i in range(1, n_reader + 1):
            md[i] = 0
        fence()
        md[0] = 1
        fence()
        mq.current_idx = (idx + 1) % ring.max_chunks
    finally:
        md.release()
    mq._spin_condition.notify()
    return True


# --------------------------------------------------------------------------
# (1) WorkerProc methods
# --------------------------------------------------------------------------


class _LeanState:
    __slots__ = ("puts", "done", "loop_ok", "inline", "queued_none", "queued", "rank")

    def __init__(self, rank: Any) -> None:
        self.puts = 0  # written by the step thread only
        self.done = 0  # written by the output thread only
        self.loop_ok = False
        self.inline = 0
        self.queued_none = 0
        self.queued = 0
        self.rank = rank


def _state_of(self: Any) -> _LeanState:
    st = self.__dict__.get(_STATE)
    if st is None:
        st = self.__dict__.setdefault(_STATE, _LeanState(getattr(self, "rank", "?")))
    return st


def _log_stats(self: Any, st: _LeanState) -> None:
    w = None
    mq = getattr(self, "worker_response_mq", None)
    if mq is not None:
        w = mq.__dict__.get(_MQ_STATE) or None
    logger.info(
        "VLLM_TPU_OUTPUT_PATH_LEAN rank=%s: none inline=%d queued=%d; queued items=%d; "
        "mq fast=%s fallback=%s",
        st.rank,
        st.inline,
        st.queued_none,
        st.queued,
        w.fast if w else "-",
        w.fallback if w else "-",
    )


def wrap_worker_proc(cls: type, use_lean_mq: bool) -> bool:
    """Replaces ``cls.enqueue_output`` / ``handle_output`` /
    ``async_output_busy_loop``. False, after one log line, when the class
    shape differs."""
    if getattr(cls, _MARK, False):
        return False
    if not worker_sources_match(cls):
        logger.info(
            "VLLM_TPU_OUTPUT_PATH_LEAN: %s methods differ from the vLLM source this "
            "replicates; not installed",
            getattr(cls, "__name__", cls),
        )
        return False

    import vllm.v1.executor.multiproc_executor as mpe
    from vllm.v1.outputs import AsyncModelRunnerOutput

    response_status = cls.ResponseStatus
    success = response_status.SUCCESS
    failure = response_status.FAILURE
    vllm_logger = mpe.logger

    @functools.wraps(cls.__dict__["enqueue_output"])
    def enqueue_output(self, output: Any):
        # Same steps as vLLM's enqueue_output; only the final write differs.
        if isinstance(output, AsyncModelRunnerOutput):
            try:
                output = output.get_output()
            except Exception as e:
                vllm_logger.exception("Error getting async model runner output")
                output = e

        if isinstance(output, Exception):
            result = (failure, str(output))
        else:
            result = (success, output)
        if (response_mq := self.worker_response_mq) is not None:
            if use_lean_mq:
                lean_enqueue(
                    response_mq, result, none_status=success if output is None else None
                )
            else:
                response_mq.enqueue(result)

    @functools.wraps(cls.__dict__["handle_output"])
    def handle_output(self, output: Any):
        if not self.use_async_scheduling:
            self.enqueue_output(output)
            return
        st = _state_of(self)
        if output is None and st.loop_ok and st.puts == st.done:
            # Output thread idle and every earlier response on the ring:
            # this is the position the queue would give it (module doc).
            st.inline += 1
            self.enqueue_output(output)
            return
        if output is None:
            st.queued_none += 1
        else:
            st.queued += 1
            if st.queued % _LOG_EVERY == 0:
                _log_stats(self, st)
        st.puts += 1
        self.async_output_queue.put(output)

    @functools.wraps(cls.__dict__["async_output_busy_loop"])
    def async_output_busy_loop(self):
        # Same prelude as vLLM's loop.
        from vllm.platforms import current_platform

        if hasattr(self.worker, "device"):
            current_platform.set_device(self.worker.device)

        st = _state_of(self)
        st.loop_ok = True
        q = self.async_output_queue
        while True:
            output = q.get()
            self.enqueue_output(output)
            # After the write: puts == done means the ring holds every item.
            # Not reached when enqueue_output raises (the thread dies as in
            # vLLM, and nothing goes inline from then on).
            st.done += 1

    cls.enqueue_output = enqueue_output
    cls.handle_output = handle_output
    cls.async_output_busy_loop = async_output_busy_loop
    setattr(cls, _MARK, True)
    logger.info(
        "VLLM_TPU_OUTPUT_PATH_LEAN installed on %s (pid %d, lean mq=%s)",
        cls.__name__,
        os.getpid(),
        use_lean_mq,
    )
    return True


def patch_output_path_lean() -> None:
    """patch_registry callback (platform_activation, worker_init); a no-op
    unless VLLM_TPU_OUTPUT_PATH_LEAN is set. Must run before WorkerProc
    starts its output thread (worker_init runs inside WorkerProc.__init__,
    before the thread is created)."""
    if not enabled():
        return
    try:
        from vllm.v1.executor.multiproc_executor import WorkerProc
    except Exception as exc:  # pragma: no cover - vLLM shape differs
        logger.info("VLLM_TPU_OUTPUT_PATH_LEAN: WorkerProc unavailable (%s)", exc)
        return
    if getattr(WorkerProc, _MARK, False):
        return
    use_lean_mq = mq_sources_match()
    if not use_lean_mq:
        logger.info(
            "VLLM_TPU_OUTPUT_PATH_LEAN: MessageQueue differs from the vLLM source this "
            "replicates; response writes use the original enqueue"
        )
    wrap_worker_proc(WorkerProc, use_lean_mq)
