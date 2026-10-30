# SPDX-License-Identifier: Apache-2.0
"""Debug evidence for TPU tensors freed on torch_tpu compile threads.

The MoE schedule HLO pass runs inside XLA compilation on torch_tpu's compile
pool, whose threads must not touch thread-local context state. A cyclic
collection there can free an unrelated TPU tensor, and torch_tpu 0.2.0's
storage deleter then aborts with ``Check failed: allow_context_state_access``.

VLLM_TPU_DEBUG_COMPILE_THREAD_GC selects a mode; every mode other than "off"
logs, tagged ``[compile-thread-gc]``, the OS name of each thread that runs the
pass and every collection that starts on a non-main thread (any generation on
a pass thread, generation >= 1 elsewhere).

- "observe" keeps the pass's GC guard and, after each precompile phase, logs
  the TPU tensors a main-thread collection finds among unreachable objects,
  with the unreachable objects that refer to them. That collection frees them
  on the main thread, so this mode cannot also reproduce the abort.
- "collect_in_pass" drops the guard and runs a full collection inside the
  pass, so any TPU tensor then held only by a cycle is freed on the compile
  thread. Expect the abort if such a tensor exists.
"""

import collections
import gc
import threading
from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

TAG = "[compile-thread-gc]"

_lock = threading.Lock()
_pass_threads: set[int] = set()
_watch_installed = False


def thread_name() -> str:
    """OS thread name (e.g. ``tf_tt_compile``), not the Python one."""
    tid = threading.get_native_id()
    try:
        with open(f"/proc/self/task/{tid}/comm") as f:
            return f.read().strip()
    except OSError:
        return threading.current_thread().name


def note_pass_thread() -> None:
    """Log the first pass invocation on each thread."""
    tid = threading.get_native_id()
    with _lock:
        if tid in _pass_threads:
            return
        _pass_threads.add(tid)
    logger.info(
        "%s HLO pass on thread %s tid=%d main=%s",
        TAG,
        thread_name(),
        tid,
        threading.current_thread() is threading.main_thread(),
    )


def _on_gc(phase: str, info: dict[str, Any]) -> None:
    if phase != "start" or threading.current_thread() is threading.main_thread():
        return
    tid = threading.get_native_id()
    if info["generation"] == 0 and tid not in _pass_threads:
        return
    logger.info(
        "%s gen%d collection on thread %s tid=%d pass_thread=%s",
        TAG,
        info["generation"],
        thread_name(),
        tid,
        tid in _pass_threads,
    )


def collect_in_pass() -> int:
    """Full collection on the calling (pass) thread, logged around it."""
    logger.info("%s forcing collection on thread %s", TAG, thread_name())
    collected = gc.collect()
    logger.info("%s forced collection freed %d objects", TAG, collected)
    return collected


def install_gc_watch() -> None:
    global _watch_installed
    with _lock:
        if not _watch_installed:
            gc.callbacks.append(_on_gc)
            _watch_installed = True


def _describe(obj: Any) -> str:
    import torch

    if isinstance(obj, torch.Tensor):
        return f"{type(obj).__name__}{tuple(obj.shape)} {obj.dtype} {obj.device}"
    code = getattr(obj, "f_code", None)
    if code is not None:
        return f"frame {code.co_filename}:{obj.f_lineno} {code.co_name}"
    if isinstance(obj, dict):
        return f"dict keys={list(obj)[:8]}"
    qualname = getattr(obj, "__qualname__", None)
    if callable(obj) and qualname:
        return f"{type(obj).__name__} {getattr(obj, '__module__', '?')}.{qualname}"
    return type(obj).__qualname__


def dump_tpu_garbage(
    phase: str,
    *,
    device_type: str = "tpu",
    depth: int = 4,
    limit: int = 8,
    max_referrer_scans: int = 64,
) -> int:
    """Log device tensors among unreachable objects; returns how many."""
    if threading.current_thread() is not threading.main_thread():
        return 0
    import torch

    previous = gc.get_debug()
    gc.set_debug(previous | gc.DEBUG_SAVEALL)
    try:
        gc.collect()
        garbage = list(gc.garbage)
        gc.garbage.clear()
    finally:
        gc.set_debug(previous)
    try:
        ids = {id(o) for o in garbage}
        tensors = [
            o
            for o in garbage
            if isinstance(o, torch.Tensor) and o.device.type == device_type
        ]
        types = collections.Counter(type(o).__qualname__ for o in garbage)
        logger.info(
            "%s phase=%s unreachable=%d %s_tensors=%d top_types=%s",
            TAG,
            phase,
            len(garbage),
            device_type,
            len(tensors),
            types.most_common(8),
        )
        # gc.get_referrers scans every tracked object, so cap the scans.
        scans = 0
        for tensor in tensors[:limit]:
            logger.info("%s   %s", TAG, _describe(tensor))
            frontier, seen = [tensor], {id(tensor)}
            for level in range(depth):
                parents = []
                for obj in frontier:
                    if scans >= max_referrer_scans:
                        break
                    scans += 1
                    for ref in gc.get_referrers(obj):
                        if id(ref) in ids and id(ref) not in seen:
                            seen.add(id(ref))
                            parents.append(ref)
                            logger.info(
                                "%s   %s<- %s", TAG, "  " * level, _describe(ref)
                            )
                frontier = parents
        return len(tensors)
    finally:
        # Free the saved cycles here, on the main thread, not at the next
        # automatic collection on whatever thread triggers it.
        del garbage
        gc.collect()
