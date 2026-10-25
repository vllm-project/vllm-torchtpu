# SPDX-License-Identifier: Apache-2.0
"""Report, and optionally drop, the host memory warmup leaves behind.

A warmed qwen4_exp worker holds ~35 GiB more host memory than before compile
(~0.65 GiB per compiled bucket). This says where it sits:

- Python: large bytes objects, compiled executables, bundled AOTAutograd
  entries and the runnables that keep them alive through ``.serialize``
  (torch_tpu's TpuBackend attaches ``serialize = lambda: entry`` to every
  runnable it returns; only the compile that persists it needs it).
- Native: what is left in RSS once Python's share is gone, i.e. the runtime's
  executable copies or allocator pages.

``VLLM_TPU_POST_WARMUP_HOSTMEM=report`` logs it once warmup is done;
``drop`` also removes the ``.serialize`` attributes, collects and returns the
freed pages, then logs it again. Off by default.
"""

import ctypes
import gc
import time
from typing import Any

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_BIG = 1 << 20


def _status() -> str:
    fields = {}
    with open("/proc/self/status") as f:
        for line in f:
            key, _, value = line.partition(":")
            if key in ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmLck"):
                fields[key] = int(value.split()[0]) / 1048576
    tc = ""
    process = ctypes.CDLL(None)
    if hasattr(process, "MallocExtension_GetNumericProperty"):
        get = process.MallocExtension_GetNumericProperty
        get.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t)]
        parts = []
        for name in (
            b"generic.current_allocated_bytes",
            b"tcmalloc.pageheap_free_bytes",
            b"tcmalloc.pageheap_unmapped_bytes",
        ):
            out = ctypes.c_size_t(0)
            if get(name, ctypes.byref(out)):
                parts.append(f"{name.decode().rsplit('.', 1)[-1]}={out.value >> 30}G")
        tc = " tcmalloc " + " ".join(parts)
    return " ".join(f"{k}={v:.1f}G" for k, v in fields.items()) + tc


def _census() -> tuple[str, list[Any]]:
    """Summarize large Python-held objects; return the runnables with .serialize."""
    objs = gc.get_objects()
    by_type: dict[str, int] = {}
    runnables = []
    executables = []
    for o in objs:
        name = type(o).__name__
        if name in (
            "TorchTpuCompiledExecutable",
            "GraphModule",
            "BundledAOTAutogradResult",
            "BundledAOTAutogradCacheEntry",
            "CompiledFunction",
        ):
            by_type[name] = by_type.get(name, 0) + 1
            if name == "TorchTpuCompiledExecutable":
                executables.append(o)
        try:
            if callable(o) and "serialize" in (getattr(o, "__dict__", None) or ()):
                runnables.append(o)
        except Exception:
            pass
    # bytes are not GC-tracked: reach them through their tracked holders.
    seen: set[int] = set()
    big_n = big_bytes = 0
    # Untracked containers (a dict of only bytes, say) are scanned one level down.
    untracked = []
    for batch in (objs, untracked):
        for i in range(0, len(batch), 4096):
            for r in gc.get_referents(*batch[i : i + 4096]):
                if isinstance(r, (bytes, bytearray, memoryview)):
                    n = r.nbytes if isinstance(r, memoryview) else len(r)
                    if n >= _BIG and id(r) not in seen:
                        seen.add(id(r))
                        big_n += 1
                        big_bytes += n
                elif batch is objs and isinstance(r, (dict, tuple, list)):
                    if not gc.is_tracked(r):
                        untracked.append(r)
    exe_size = ""
    if executables:
        try:
            from torch_tpu._internal.compile import torch_tpu_compiled_executable as m

            blob = m.tpu_torch_compile.serialize_executable(executables[0]._executable)
            exe_size = f" one_executable_serialized={len(blob) >> 20}M"
            del blob
        except Exception as e:  # diagnostic only
            exe_size = f" one_executable_serialized=? ({type(e).__name__})"
    types = " ".join(f"{k}={v}" for k, v in sorted(by_type.items()))
    summary = (
        f"objects={len(objs)} {types} serialize_attrs={len(runnables)} "
        f"bytes>=1M: n={big_n} total={big_bytes >> 20}M{exe_size}"
    )
    del objs, executables
    return summary, runnables


def _release_heap() -> None:
    process = ctypes.CDLL(None)
    if hasattr(process, "MallocExtension_ReleaseFreeMemory"):
        process.MallocExtension_ReleaseFreeMemory()
    elif hasattr(process, "malloc_trim"):
        process.malloc_trim(0)


def report_and_maybe_drop() -> None:
    """Called at the end of compile_or_warm_up_model, before the GC freeze."""
    mode = envs.VLLM_TPU_POST_WARMUP_HOSTMEM
    if mode not in ("report", "drop"):
        return
    start = time.perf_counter()
    before = _status()
    summary, runnables = _census()
    logger.info("[hostmem] after warmup: %s | %s", before, summary)
    if mode != "drop":
        return
    for fn in runnables:
        try:
            del fn.__dict__["serialize"]
        except (KeyError, TypeError):
            pass
    dropped = len(runnables)
    del runnables
    collected = gc.collect()
    _release_heap()
    after = _status()
    summary, rest = _census()
    del rest
    logger.info(
        "[hostmem] dropped %d serialize attrs, gc collected %d, %.1f s: %s | %s",
        dropped,
        collected,
        time.perf_counter() - start,
        after,
        summary,
    )
