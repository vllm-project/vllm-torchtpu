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
    big_str_n = big_str = 0
    # Untracked containers (a dict of only bytes, say) are scanned one level down.
    untracked = []
    for batch in (objs, untracked):
        for i in range(0, len(batch), 4096):
            for r in gc.get_referents(*batch[i : i + 4096]):
                # A dead weakref.proxy raises ReferenceError on isinstance(); a
                # released memoryview raises ValueError on .nbytes.
                try:
                    if isinstance(r, (bytes, bytearray, memoryview, str)):
                        n = r.nbytes if isinstance(r, memoryview) else len(r)
                        if n >= _BIG and id(r) not in seen:
                            seen.add(id(r))
                            if isinstance(r, str):
                                big_str_n += 1
                                big_str += n
                            else:
                                big_n += 1
                                big_bytes += n
                    elif batch is objs and isinstance(r, (dict, tuple, list)):
                        if not gc.is_tracked(r):
                            untracked.append(r)
                except (ReferenceError, ValueError):
                    continue
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
    if envs.VLLM_TPU_POST_WARMUP_HOSTMEM_DETAIL:
        _log_detail(objs)
    summary = (
        f"objects={len(objs)} {types} serialize_attrs={len(runnables)} "
        f"bytes>=1M: n={big_n} total={big_bytes >> 20}M "
        f"str>=1M: n={big_str_n} total={big_str >> 20}M{exe_size}"
    )
    del objs, executables
    return summary, runnables


def _log_detail(objs: list[Any]) -> None:
    """Python's share of the heap by type, and who holds a bundled entry."""
    import sys

    sizes: dict[str, list[int]] = {}
    for o in objs:
        t = type(o)
        try:
            n = sys.getsizeof(o)
        except Exception:
            continue
        rec = sizes.setdefault(t.__name__, [0, 0])
        rec[0] += 1
        rec[1] += n
    total = sum(v[1] for v in sizes.values())
    top = sorted(sizes.items(), key=lambda kv: -kv[1][1])[:20]
    logger.info(
        "[hostmem] python tracked objects shallow total=%.1fG top: %s",
        total / (1 << 30),
        " ".join(f"{k}={v[0]}/{v[1] >> 20}M" for k, v in top),
    )
    entry = next(
        (o for o in objs if type(o).__name__ == "BundledAOTAutogradResult"), None
    )
    if entry is None:
        return
    chain = []
    level = [entry]
    for _ in range(4):
        nxt = []
        for x in level:
            for r in gc.get_referrers(x):
                if r is objs or r is level or r is nxt or type(r).__name__ == "frame":
                    continue
                nxt.append(r)
        desc = {}
        for r in nxt:
            name = type(r).__name__
            if name == "dict":
                keys = [k for k in r if isinstance(k, str)][:4]
                name = f"dict{keys}"
            elif name == "function":
                name = f"fn:{getattr(r, '__qualname__', '?')}"
            elif name == "cell":
                name = "cell"
            desc[name] = desc.get(name, 0) + 1
        chain.append(" ".join(f"{k}x{v}" for k, v in list(desc.items())[:8]))
        level = nxt[:16]
        if not level:
            break
    logger.info("[hostmem] referrers of one bundled entry: %s", " <- ".join(chain))


def _release_heap() -> None:
    process = ctypes.CDLL(None)
    if hasattr(process, "MallocExtension_ReleaseFreeMemory"):
        process.MallocExtension_ReleaseFreeMemory()
    elif hasattr(process, "malloc_trim"):
        process.malloc_trim(0)


def report_and_maybe_drop() -> None:
    """Called at the end of compile_or_warm_up_model, before the GC freeze.

    Diagnostic only: a failure is logged and never takes the worker down.
    """
    mode = envs.VLLM_TPU_POST_WARMUP_HOSTMEM
    if mode not in ("report", "drop", "clearjax"):
        return
    try:
        _report_and_maybe_drop(mode)
    except Exception:
        logger.exception("[hostmem] report failed; continuing")


def _heap_prefix() -> str:
    import os

    prefix = os.environ.get("VLLM_TPU_HEAPDUMP_PREFIX", "")
    return f"{prefix}_r{os.environ.get('RANK', os.getpid())}" if prefix else ""


def start_heap_profiler() -> None:
    """Start gperftools' heap profiler for this worker (compile and warmup
    only, so weight loading is not slowed). Needs libtcmalloc (not _minimal)
    preloaded; HEAP_PROFILE_INUSE_INTERVAL etc. are read at start."""
    if not envs.VLLM_TPU_POST_WARMUP_HEAPDUMP or not _heap_prefix():
        return
    try:
        process = ctypes.CDLL(None)
        if hasattr(process, "HeapProfilerStart"):
            process.HeapProfilerStart(_heap_prefix().encode())
            logger.info("[hostmem] heap profiler started: %s", _heap_prefix())
    except Exception:
        logger.exception("[hostmem] heap profiler start failed; continuing")


def _heap_dump() -> None:
    """Dump the live heap now and copy every dump of this worker next to the
    job outputs."""
    import glob
    import os
    import shutil

    process = ctypes.CDLL(None)
    prefix = _heap_prefix()
    if not hasattr(process, "HeapProfilerDump") or not prefix:
        logger.info("[hostmem] heap profiler not running; no heap dump")
        return
    process.HeapProfilerDump(b"warm")
    out = os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp/cdk_output"), "hprof")
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(f"{prefix}.*.heap"))
    for f in files:
        shutil.copy(f, os.path.join(out, os.path.basename(f)))
    with open(f"/proc/{os.getpid()}/maps") as f, open(
        os.path.join(out, os.path.basename(prefix) + ".maps"), "w"
    ) as g:
        g.write(f.read())
    logger.info("[hostmem] heap dumps %s -> %s", files, out)


def _report_and_maybe_drop(mode: str) -> None:
    if envs.VLLM_TPU_POST_WARMUP_HEAPDUMP:
        try:
            _heap_dump()
        except Exception:
            logger.exception("[hostmem] heap dump failed; continuing")
    start = time.perf_counter()
    before = _status()
    summary, runnables = _census()
    logger.info("[hostmem] after warmup: %s | %s", before, summary)
    if mode == "clearjax":
        # Diagnostic: how much of the warmed heap JAX's tracing/lowering caches
        # (Pallas kernels trace through jax) keep alive.
        import jax

        jax.clear_caches()
        collected = gc.collect()
        _release_heap()
        logger.info(
            "[hostmem] jax.clear_caches + gc (%d) + release, %.1f s: %s",
            collected,
            time.perf_counter() - start,
            _status(),
        )
        return
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
