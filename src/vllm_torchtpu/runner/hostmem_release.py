"""Host memory after warmup: return the allocator's free pages, and sample
where host RSS goes while serving.

``VLLM_TPU_POST_WARMUP_HOSTMEM=release`` collects and returns tcmalloc's free
page heap to the OS once warmup is done (~6 GiB per worker; about a second).

``VLLM_TPU_HOSTMEM_SAMPLE_S=<s>`` logs, every ``s`` seconds, this worker's RSS
split (anon / file / shmem) next to tcmalloc's live and free bytes and
pymalloc's block count, so growth can be told apart: tcmalloc live = native
heap, blocks = Python objects, RSS outside tcmalloc = mmaps (e.g. a host KV
pool filling in). Rank 0 also logs every process in the container and the
cgroup's memory.stat. Off by default.
"""

import ctypes
import gc
import os
import sys
import threading
import time

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_GIB = 1 << 30


def _proc_status(pid: str = "self") -> dict[str, float]:
    fields = {}
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            key, _, value = line.partition(":")
            if key in ("VmRSS", "RssAnon", "RssFile", "RssShmem"):
                fields[key] = int(value.split()[0]) / 1048576
    return fields


def _tcmalloc() -> dict[str, float]:
    process = ctypes.CDLL(None)
    if not hasattr(process, "MallocExtension_GetNumericProperty"):
        return {}
    get = process.MallocExtension_GetNumericProperty
    get.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t)]
    out = {}
    for name in (
        b"generic.current_allocated_bytes",
        b"tcmalloc.pageheap_free_bytes",
    ):
        value = ctypes.c_size_t(0)
        if get(name, ctypes.byref(value)):
            out[name.decode().rsplit(".", 1)[-1]] = value.value / _GIB
    return out


def _status() -> str:
    fields = {**_proc_status(), **_tcmalloc()}
    return " ".join(f"{k}={v:.1f}G" for k, v in fields.items())


def _release_heap() -> None:
    process = ctypes.CDLL(None)
    if hasattr(process, "MallocExtension_ReleaseFreeMemory"):
        process.MallocExtension_ReleaseFreeMemory()
    elif hasattr(process, "malloc_trim"):
        process.malloc_trim(0)


def _release() -> None:
    start = time.perf_counter()
    before = _status()
    collected = gc.collect()
    _release_heap()
    logger.info(
        "[hostmem] release: gc (%d) + release, %.1f s: %s -> %s",
        collected,
        time.perf_counter() - start,
        before,
        _status(),
    )


def _all_processes() -> str:
    rows = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            fields = _proc_status(pid)
            with open(f"/proc/{pid}/comm") as f:
                comm = f.read().strip()
        except (OSError, ValueError):
            continue
        if fields.get("VmRSS", 0.0) >= 1.0:
            rows.append((fields["VmRSS"], pid, comm, fields))
    rows.sort(reverse=True)
    return " | ".join(
        f"{comm}[{pid}] rss={rss:.1f} anon={f.get('RssAnon', 0):.1f}"
        f" shmem={f.get('RssShmem', 0):.1f}"
        for rss, pid, comm, f in rows
    )


def _cgroup() -> str:
    try:
        with open("/sys/fs/cgroup/memory.current") as f:
            current = int(f.read()) / _GIB
        stat = {}
        with open("/sys/fs/cgroup/memory.stat") as f:
            for line in f:
                key, value = line.split()
                if key in ("anon", "file", "shmem", "kernel"):
                    stat[key] = int(value) / _GIB
    except (OSError, ValueError):
        return "n/a"
    return f"current={current:.1f}G " + " ".join(
        f"{k}={v:.1f}G" for k, v in stat.items()
    )


def _sample_loop(rank: int, period: float) -> None:
    while True:
        time.sleep(period)
        try:
            logger.info(
                "[hostmem] sample r%d: %s pyblocks=%d",
                rank,
                _status(),
                sys.getallocatedblocks(),
            )
            if rank == 0:
                logger.info(
                    "[hostmem] sample cgroup %s || %s", _cgroup(), _all_processes()
                )
        except Exception:
            logger.exception("[hostmem] sample failed; stopping the sampler")
            return


def after_warmup(rank: int) -> None:
    """Called at the end of compile_or_warm_up_model. Never fails the worker."""
    try:
        if envs.VLLM_TPU_POST_WARMUP_HOSTMEM == "release":
            _release()
        period = envs.VLLM_TPU_HOSTMEM_SAMPLE_S
        if period > 0:
            logger.info("[hostmem] sample r%d at warmup end: %s", rank, _status())
            threading.Thread(
                target=_sample_loop,
                args=(rank, period),
                name="hostmem-sampler",
                daemon=True,
            ).start()
    except Exception:
        logger.exception("[hostmem] after_warmup failed; continuing")
