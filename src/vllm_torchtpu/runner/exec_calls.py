"""Count calls per loaded torch_tpu executable, to find buckets serving never
runs.

Every loaded executable keeps tens of MB of host memory in torch_tpu's tier-1
compilation cache for the life of the worker, so a bucket that serving never
calls is pure host cost. ``VLLM_TPU_EXEC_CALL_COUNT=1`` sizes each executable
when it is created (growth of tcmalloc's live bytes since the previous one, a
rough size), labels it on its first call (the innermost vllm_torchtpu frame
and its tensor shapes) and counts its calls. Rank 0 logs one ``[execcalls]`` line per
executable at warmup end (labels and sizes) and, every
``VLLM_TPU_EXEC_CALL_COUNT_S`` seconds, the calls each one received since
warmup ended. Off by default.
"""

import ctypes
import sys
import threading
import time

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_labels: dict[int, str] = {}
_sizes: dict[int, int] = {}
_calls: dict[int, int] = {}
_base: dict[int, int] = {}
_state = {"live": 0, "installed": False}


def _live_bytes() -> int:
    process = ctypes.CDLL(None)
    if not hasattr(process, "MallocExtension_GetNumericProperty"):
        return 0
    get = process.MallocExtension_GetNumericProperty
    get.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t)]
    value = ctypes.c_size_t(0)
    get(b"generic.current_allocated_bytes", ctypes.byref(value))
    return value.value


def _label() -> str:
    frame = sys._getframe(2)
    while frame is not None:
        path = frame.f_code.co_filename
        if "/vllm_torchtpu/" in path and not path.endswith("exec_calls.py"):
            shapes = [
                f"{name}={tuple(value.shape)}"
                for name, value in frame.f_locals.items()
                if isinstance(value, torch.Tensor)
            ][:6]
            return (
                f"{path.rsplit('/', 1)[-1]}:{frame.f_lineno} {frame.f_code.co_name}"
                + (f" [{' '.join(shapes)}]" if shapes else "")
            )
        frame = frame.f_back
    return "?"


def install() -> None:
    """Patch the executable class; call before the first compile."""
    if _state["installed"] or not envs.VLLM_TPU_EXEC_CALL_COUNT:
        return
    try:
        from torch_tpu._internal.compile.torch_tpu_compiled_executable import (
            TorchTpuCompiledExecutable as cls,
        )
    except ImportError:
        logger.warning("[execcalls] torch_tpu executable class not found")
        return
    init, call = cls.__init__, cls.__call__

    def counted_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        key = id(self)
        live = _live_bytes()
        _sizes[key] = live - _state["live"]
        _state["live"] = live
        _calls[key] = 0

    def counted_call(self, *args, **kwargs):
        key = id(self)
        n = _calls.get(key, 0)
        if not n:
            # Label on the first call: creation may run on a compile thread.
            _labels[key] = _label()
        _calls[key] = n + 1
        return call(self, *args, **kwargs)

    cls.__init__ = counted_init
    cls.__call__ = counted_call
    _state["live"] = _live_bytes()
    _state["installed"] = True
    logger.info("[execcalls] counting calls per torch_tpu executable")


def _dump(tag: str) -> None:
    rows = sorted(_calls.items(), key=lambda kv: kv[1] - _base.get(kv[0], 0))
    for i, (key, total) in enumerate(rows):
        logger.info(
            "[execcalls] %s #%d calls=%d warm=%d size=%.1fM %s",
            tag,
            i,
            total - _base.get(key, 0),
            _base.get(key, 0),
            _sizes.get(key, 0) / 2**20,
            _labels.get(key, "?"),
        )


def _loop(period: float) -> None:
    start = time.monotonic()
    while True:
        time.sleep(period)
        try:
            _dump(f"t={time.monotonic() - start:.0f}s")
        except Exception:
            logger.exception("[execcalls] dump failed; stopping")
            return


def after_warmup(rank: int) -> None:
    if not _state["installed"]:
        return
    _base.update(_calls)
    if rank != 0:
        return
    logger.info(
        "[execcalls] %d executables at warmup end, size sum %.1fG",
        len(_calls),
        sum(_sizes.values()) / 2**30,
    )
    period = envs.VLLM_TPU_EXEC_CALL_COUNT_S
    if period > 0:
        threading.Thread(
            target=_loop, args=(period,), name="execcalls", daemon=True
        ).start()
