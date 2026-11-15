# SPDX-License-Identifier: Apache-2.0
"""Periodic host memory report (TPU_HOST_MEMORY_LOG_S).

A host that runs out of memory during compile loses a worker without an error
line: the kernel kills the process, and the executor only reports that the
worker died. With TPU_HOST_MEMORY_LOG_S=N (N > 0), one worker per host logs
the memory of its container every N seconds:

  * the memory cgroup's usage, limit and peak (cgroup v2, else v1), split into
    anonymous, page cache (and its mapped part), shared memory (/dev/shm is
    shared memory) and unevictable (mlocked or pinned) pages;
  * the cgroup's OOM-kill count, as a warning when it grew since the last
    report;
  * the processes with the largest resident set, by pid, so a log line can be
    matched to the worker, EngineCore or API server that holds the memory.

The workers of a host compete for an flock on a file in the temp directory.
The holder reports. The others retry every interval, and one of them takes
over within an interval when the holder exits, because the kernel drops an
flock with its process. Reading only: nothing is allocated on the TPU and no
step waits on the thread. Any failure is logged at debug level and the thread
keeps going.
"""

from __future__ import annotations

import fcntl
import os
import tempfile
import threading
from dataclasses import dataclass

from vllm_torchtpu.logger import init_logger

# vllm_torchtpu.logger names this logger "vllm.<module>", so it uses vLLM's
# handler. A logger outside the "vllm" tree (vllm.logger.init_logger(__name__))
# falls through to Python's last-resort handler, which drops INFO lines.
logger = init_logger(__name__)

_GIB = float(1 << 30)
# cgroup v1 reports "no limit" as the largest page-aligned 64-bit value.
_V1_NO_LIMIT = 1 << 62
LOCK_FILE_NAME = "vllm_torchtpu_host_memory.lock"


@dataclass(frozen=True)
class CgroupMemory:
    """One reading of a memory cgroup, in bytes. None: not reported."""

    current: int
    limit: int | None
    peak: int | None
    anon: int | None
    file: int | None
    shmem: int | None
    oom_kills: int | None
    # Page cache that processes map, and memory the kernel cannot reclaim
    # (mlocked or pinned pages): the page cache counts toward the limit, but
    # only its unmapped, evictable part is free to reclaim before an OOM kill.
    file_mapped: int | None = None
    unevictable: int | None = None


def _read_int(path: str) -> int | None:
    """An integer file such as memory.current; None if missing or 'max'."""
    try:
        with open(path) as f:
            text = f.read().strip()
    except OSError:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _read_keyed(path: str) -> dict[str, int]:
    """A file of 'key value' lines such as memory.stat or memory.events."""
    values: dict[str, int] = {}
    try:
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) != 2:
                    continue
                try:
                    values[parts[0]] = int(parts[1])
                except ValueError:
                    continue
    except OSError:
        pass
    return values


def _read_v2(directory: str) -> CgroupMemory | None:
    current = _read_int(os.path.join(directory, "memory.current"))
    if current is None:
        return None
    stat = _read_keyed(os.path.join(directory, "memory.stat"))
    events = _read_keyed(os.path.join(directory, "memory.events"))
    return CgroupMemory(
        current=current,
        limit=_read_int(os.path.join(directory, "memory.max")),
        peak=_read_int(os.path.join(directory, "memory.peak")),
        anon=stat.get("anon"),
        file=stat.get("file"),
        shmem=stat.get("shmem"),
        oom_kills=events.get("oom_kill"),
        file_mapped=stat.get("file_mapped"),
        unevictable=stat.get("unevictable"),
    )


def _read_v1(directory: str) -> CgroupMemory | None:
    current = _read_int(os.path.join(directory, "memory.usage_in_bytes"))
    if current is None:
        return None
    limit = _read_int(os.path.join(directory, "memory.limit_in_bytes"))
    if limit is not None and limit >= _V1_NO_LIMIT:
        limit = None
    stat = _read_keyed(os.path.join(directory, "memory.stat"))
    oom_control = _read_keyed(os.path.join(directory, "memory.oom_control"))
    return CgroupMemory(
        current=current,
        limit=limit,
        peak=_read_int(os.path.join(directory, "memory.max_usage_in_bytes")),
        anon=stat.get("total_rss", stat.get("rss")),
        file=stat.get("total_cache", stat.get("cache")),
        shmem=stat.get("total_shmem", stat.get("shmem")),
        oom_kills=oom_control.get("oom_kill"),
        file_mapped=stat.get("total_mapped_file", stat.get("mapped_file")),
        unevictable=stat.get("total_unevictable", stat.get("unevictable")),
    )


def _own_cgroup_paths(self_cgroup: str) -> tuple[str | None, str | None]:
    """This process's (v2, v1 memory) cgroup paths from /proc/self/cgroup."""
    v2 = v1 = None
    try:
        with open(self_cgroup) as f:
            for line in f:
                parts = line.rstrip("\n").split(":", 2)
                if len(parts) != 3:
                    continue
                if parts[0] == "0" and parts[1] == "":
                    v2 = parts[2]
                elif "memory" in parts[1].split(","):
                    v1 = parts[2]
    except OSError:
        pass
    return v2, v1


def read_cgroup_memory(
    root: str = "/sys/fs/cgroup", self_cgroup: str = "/proc/self/cgroup"
) -> CgroupMemory | None:
    """The memory cgroup of this process's container, cgroup v2 first, then
    v1. The process's own path from ``self_cgroup`` comes first: it is "/"
    under a private cgroup namespace (the mount root is the container's
    cgroup), and the full path without one (the mount root is the host's top
    cgroup). None if nothing is readable."""
    v2_path, v1_path = _own_cgroup_paths(self_cgroup)
    v1_root = os.path.join(root, "memory")
    candidates = [
        (_read_v2, os.path.join(root, v2_path.lstrip("/")) if v2_path else None),
        (_read_v2, root),
        (_read_v1, os.path.join(v1_root, v1_path.lstrip("/")) if v1_path else None),
        (_read_v1, v1_root),
    ]
    for reader, directory in candidates:
        if directory is not None:
            memory = reader(directory)
            if memory is not None:
                return memory
    return None


def largest_processes(
    proc: str = "/proc", count: int = 4
) -> list[tuple[int, str, int]]:
    """The ``count`` processes with the largest resident set, largest first,
    as (pid, name, bytes). Kernel threads (no VmRSS) are skipped."""
    try:
        entries = os.listdir(proc)
    except OSError:
        return []
    found = []
    for entry in entries:
        if not entry.isdigit():
            continue
        name, rss = "?", None
        try:
            with open(os.path.join(proc, entry, "status")) as f:
                for line in f:
                    if line.startswith("Name:"):
                        name = line.split(None, 1)[1].strip()
                    elif line.startswith("VmRSS:"):
                        rss = int(line.split()[1]) * 1024  # reported in kB
                        break
        except (OSError, ValueError, IndexError):
            continue  # the process exited, or a malformed line
        if rss is not None:
            found.append((int(entry), name, rss))
    found.sort(key=lambda item: item[2], reverse=True)
    return found[:count]


def format_report(
    memory: CgroupMemory | None, processes: list[tuple[int, str, int]]
) -> str:
    """One log line for a reading and its largest processes."""

    def gib(value: int) -> str:
        return f"{value / _GIB:.1f}"

    parts = []
    if memory is None:
        parts.append("no readable memory cgroup")
    else:
        used = f"cgroup {gib(memory.current)} GiB"
        if memory.limit:
            used += (
                f" of {gib(memory.limit)} GiB"
                f" ({100.0 * memory.current / memory.limit:.1f}%)"
            )
        parts.append(used)
        if memory.peak is not None:
            parts.append(f"peak {gib(memory.peak)} GiB")
        kinds = [
            f"{kind} {gib(value)}"
            for kind, value in (
                ("anon", memory.anon),
                ("file", memory.file),
                ("mapped", memory.file_mapped),
                ("shmem", memory.shmem),
                ("unevictable", memory.unevictable),
            )
            if value is not None
        ]
        if kinds:
            parts.append(", ".join(kinds) + " GiB")
        if memory.oom_kills is not None:
            parts.append(f"oom_kill {memory.oom_kills}")
    if processes:
        parts.append(
            "largest RSS: "
            + ", ".join(
                f"pid {pid} {name} {gib(rss)} GiB" for pid, name, rss in processes
            )
        )
    return "Host memory: " + "; ".join(parts)


class HostMemoryReporter:
    """Reports host memory while it holds the host's lock file."""

    def __init__(
        self,
        interval_s: float,
        lock_path: str,
        cgroup_root: str = "/sys/fs/cgroup",
        self_cgroup: str = "/proc/self/cgroup",
        proc: str = "/proc",
    ) -> None:
        self.interval_s = interval_s
        self.lock_path = lock_path
        self.cgroup_root = cgroup_root
        self.self_cgroup = self_cgroup
        self.proc = proc
        self._lock_fd: int | None = None
        # OOM kills seen by this process's reports. Starting from 0 makes a
        # reporter that took over after an OOM kill still warn about it.
        self._oom_kills_seen = 0

    @property
    def holds_lock(self) -> bool:
        return self._lock_fd is not None

    def try_acquire(self) -> bool:
        """Take the host's lock file if no other process holds it."""
        if self._lock_fd is not None:
            return True
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        self._lock_fd = fd
        return True

    def release(self) -> None:
        if self._lock_fd is not None:
            os.close(self._lock_fd)  # closing the descriptor drops the flock
            self._lock_fd = None

    def report_once(self) -> str:
        """Log one report and return its line."""
        memory = read_cgroup_memory(self.cgroup_root, self.self_cgroup)
        line = format_report(memory, largest_processes(self.proc))
        kills = memory.oom_kills if memory is not None else None
        if kills is not None and kills > self._oom_kills_seen:
            logger.warning(
                "%s (the cgroup OOM killer fired %d time(s) since the last report)",
                line,
                kills - self._oom_kills_seen,
            )
            self._oom_kills_seen = kills
        else:
            logger.info("%s", line)
        return line

    def run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                if self.try_acquire():
                    self.report_once()
            except Exception:  # reporting only: never take the worker down
                logger.debug("Host memory report failed", exc_info=True)
            stop.wait(self.interval_s)


_start_lock = threading.Lock()
_started = False


def maybe_start(interval_s: int) -> None:
    """Start this process's reporter thread when ``interval_s`` > 0 (at most
    once per process)."""
    global _started
    if interval_s <= 0:
        return
    with _start_lock:
        if _started:
            return
        _started = True
    reporter = HostMemoryReporter(
        interval_s, os.path.join(tempfile.gettempdir(), LOCK_FILE_NAME)
    )
    threading.Thread(
        target=reporter.run,
        args=(threading.Event(),),
        name="tpu-host-memory",
        daemon=True,
    ).start()
