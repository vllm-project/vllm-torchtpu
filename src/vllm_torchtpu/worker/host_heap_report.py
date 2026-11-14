# SPDX-License-Identifier: Apache-2.0
"""Where a worker's host heap goes after warm-up (TPU_HOST_HEAP_REPORT).

Diagnostic only. Level 1 logs, at a few points of start-up, the process's
anonymous RSS, tcmalloc's in-use / free byte counters and the largest
anonymous mappings. Level 2 also takes a census of the Python heap (objects
by type, large byte buffers, CPU tensors). Level 3 also measures what dropping
start-up-only references frees: the bundled cache entry each compiled graph
keeps for `serialize()`, and JAX's tracing caches.
"""

from __future__ import annotations

import collections
import ctypes
import gc
import sys
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_GIB = float(1 << 30)
_TCMALLOC_PROPS = (
    "generic.current_allocated_bytes",
    "generic.heap_size",
    "tcmalloc.pageheap_free_bytes",
    "tcmalloc.pageheap_unmapped_bytes",
    "tcmalloc.central_cache_free_bytes",
    "tcmalloc.thread_cache_free_bytes",
    "tcmalloc.transfer_cache_free_bytes",
)


def _status_kib(field: str) -> int:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except OSError:
        pass
    return -1


def rss_anon_gib() -> float:
    return _status_kib("RssAnon") / (1 << 20)


def _tcmalloc() -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        lib = ctypes.CDLL(None)
        fn = lib.MallocExtension_GetNumericProperty
    except (OSError, AttributeError):
        return out
    fn.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t)]
    fn.restype = ctypes.c_int
    for name in _TCMALLOC_PROPS:
        v = ctypes.c_size_t(0)
        if fn(name.encode(), ctypes.byref(v)):
            out[name.split(".", 1)[1]] = v.value / _GIB
    return out


def _top_anon_mappings(n: int = 8) -> list[tuple[str, float, int]]:
    """(name, anon GiB, mapping count) of the largest anonymous holders."""
    agg: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0])
    name = "?"
    try:
        with open("/proc/self/smaps") as f:
            for line in f:
                first = line.split(None, 1)[0]
                if "-" in first and not first.endswith(":"):
                    parts = line.split()
                    name = parts[5] if len(parts) > 5 else "[anon]"
                    agg[name][1] += 1
                elif first == "Anonymous:":
                    agg[name][0] += int(line.split()[1]) / (1 << 20)
    except OSError:
        return []
    rows = sorted(agg.items(), key=lambda kv: -kv[1][0])[:n]
    return [(k, v[0], int(v[1])) for k, v in rows]


def report(tag: str, rank: int) -> None:
    tc = _tcmalloc()
    tcs = " ".join(f"{k}={v:.2f}" for k, v in tc.items()) or "n/a"
    maps = " ".join(f"{k}:{g:.2f}/{c}" for k, g, c in _top_anon_mappings())
    logger.info(
        "HOSTHEAP %s rank=%d rss_anon=%.2f GiB tcmalloc[%s] maps[%s]",
        tag,
        rank,
        rss_anon_gib(),
        tcs,
        maps,
    )


def _buffer_nbytes(o: Any) -> int:
    if isinstance(o, bytes | bytearray):
        return len(o)
    if isinstance(o, str):
        return len(o)  # approx for ascii
    if isinstance(o, memoryview):
        return o.nbytes
    return 0


def census(tag: str, rank: int, top: int = 25) -> None:
    """Python heap: objects by type (count, shallow GiB), big buffers, CPU tensors."""
    import torch

    objs = gc.get_objects()
    count: collections.Counter[str] = collections.Counter()
    size: collections.Counter[str] = collections.Counter()
    big: dict[int, tuple[int, str, str]] = {}
    cpu_storages: dict[int, tuple[int, str]] = {}
    for o in objs:
        t = type(o)
        tn = f"{t.__module__}.{t.__qualname__}"
        count[tn] += 1
        try:
            size[tn] += sys.getsizeof(o)
        except Exception:
            pass
        if isinstance(o, torch.Tensor):
            try:
                if o.device.type == "cpu":
                    st = o.untyped_storage()
                    cpu_storages[st.data_ptr()] = (st.nbytes(), tn)
            except Exception:
                pass
            continue
        try:
            refs = gc.get_referents(o)
        except Exception:
            continue
        for r in refs:
            nb = _buffer_nbytes(r)
            if nb >= (1 << 20) and id(r) not in big:
                big[id(r)] = (nb, type(r).__name__, tn)
    n_obj = len(objs)
    del objs
    total_shallow = sum(size.values()) / _GIB
    by_size = ", ".join(
        f"{k}={size[k] / _GIB:.2f}G/{count[k]}" for k, _ in size.most_common(top)
    )
    by_count = ", ".join(f"{k}={v}" for k, v in count.most_common(top))
    big_total = sum(v[0] for v in big.values()) / _GIB
    holder: collections.Counter[str] = collections.Counter()
    for nb, bt, tn in big.values():
        holder[f"{bt}<-{tn}"] += nb
    big_s = ", ".join(f"{k}={v / _GIB:.2f}G" for k, v in holder.most_common(12))
    cpu_total = sum(v[0] for v in cpu_storages.values()) / _GIB
    cpu_top = sorted(cpu_storages.values(), key=lambda v: -v[0])[:8]
    cpu_s = ", ".join(f"{nb / _GIB:.3f}G" for nb, _ in cpu_top)
    exe = {k: v for k, v in count.items() if "Executable" in k or "Artifact" in k}
    logger.info(
        "HOSTHEAP-CENSUS %s rank=%d objects=%d shallow=%.2f GiB | big buffers "
        "(>=1MiB) %d = %.2f GiB: %s | cpu tensor storages %d = %.2f GiB "
        "(top %s) | executables %s",
        tag,
        rank,
        n_obj,
        total_shallow,
        len(big),
        big_total,
        big_s,
        len(cpu_storages),
        cpu_total,
        cpu_s,
        exe,
    )
    logger.info("HOSTHEAP-CENSUS %s rank=%d by_size: %s", tag, rank, by_size)
    logger.info("HOSTHEAP-CENSUS %s rank=%d by_count: %s", tag, rank, by_count)


def _settle(release) -> float:
    gc.collect()
    release()
    return rss_anon_gib()


def experiments(rank: int, release) -> None:
    """Measure what start-up-only references hold (changes process state)."""
    base = _settle(release)
    # 1. The bundled AOTAutograd entry each compiled runnable keeps for
    #    `serialize()`; vLLM has already written it to its cache.
    dropped = 0
    for o in gc.get_objects():
        try:
            d = getattr(o, "__dict__", None)
            if d is not None and callable(o) and "serialize" in d:
                del d["serialize"]
                dropped += 1
        except Exception:
            continue
    after_ser = _settle(release)
    # 2. JAX tracing / lowering caches (Pallas kernels lowered into the
    #    compiled graphs at trace time).
    jax_ok = False
    try:
        import jax

        jax.clear_caches()
        jax_ok = True
    except Exception:
        pass
    after_jax = _settle(release)
    logger.info(
        "HOSTHEAP-EXP rank=%d base=%.2f GiB; drop %d serialize entries -> "
        "%.2f (%.2f); jax.clear_caches(%s) -> %.2f (%.2f)",
        rank,
        base,
        dropped,
        after_ser,
        after_ser - base,
        jax_ok,
        after_jax,
        after_jax - after_ser,
    )
