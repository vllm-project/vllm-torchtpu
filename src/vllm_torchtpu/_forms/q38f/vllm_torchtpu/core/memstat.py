# SPDX-License-Identifier: Apache-2.0
"""Container memory telemetry (VLLM_TPU_MEMSTAT_INTERVAL_S=N; 0 = off).

q38 HOSTTIER-D D1 telemetry. A decode pod's working set climbs for the whole
serving window (22-24 GiB/h on O1, no plateau), and on O4 the growth sits
outside RSS. The pod's /tmp, /dev/shm, /root/.cache and /var/tmp are
memory-backed emptyDirs, so tmpfs pages are charged to the cgroup and cannot be
reclaimed. ``kubectl top`` and kubelet stats only split working set from RSS.

With the flag on, one daemon thread per pod (the engine core of local DP rank
0) logs a line ``Q38_MEMSTAT {json}`` at start and then every N seconds. Each
line holds:

- ``cgroup``: memory.current / max / peak and memory.stat's anon, file, shmem,
  kernel, slab, sock, file_mapped, active_file, inactive_file and pagetables
  (cgroup v2; v1 field names are mapped onto the same keys);
- ``tmpfs``: used bytes of every tmpfs mount, from statvfs (O(1), no walk);
- ``procs``: the top processes by RSS with RssAnon / RssFile / RssShmem, plus
  container totals.

Host-only and log-only: it never touches a compiled program or the step loop,
so the flag is in ``_TPU_COMPILE_ENV_IGNORED``. Telemetry must never stop the
engine: every read is guarded and a failed read is reported in ``errors``.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from typing import Any

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

LINE_PREFIX = "Q38_MEMSTAT "
TOP_PROCS = 12

# memory.stat keys reported, cgroup v2 name -> v1 name (bytes in both).
_STAT_KEYS = {
    "anon": "rss",
    "file": "cache",
    "shmem": "shmem",
    "kernel": None,
    "slab": None,
    "sock": None,
    "file_mapped": "mapped_file",
    "active_file": "active_file",
    "inactive_file": "inactive_file",
    "pagetables": None,
}

_thread: threading.Thread | None = None
_lock = threading.Lock()


def _read_int(path: str) -> int | None:
    with open(path) as f:
        text = f.read().strip()
    return None if text == "max" else int(text)


def _parse_stat(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            out[parts[0]] = int(parts[1])
    return out


def read_cgroup(root: str = "/sys/fs/cgroup") -> dict[str, Any]:
    """The pod cgroup's usage and memory.stat, in bytes (v2, else v1)."""
    v2 = os.path.join(root, "memory.stat")
    if os.path.exists(v2):
        with open(v2) as f:
            raw = _parse_stat(f.read())
        out: dict[str, Any] = {"version": 2}
        for key, name in (
            ("current", "memory.current"),
            ("max", "memory.max"),
            ("peak", "memory.peak"),
        ):
            path = os.path.join(root, name)
            out[key] = _read_int(path) if os.path.exists(path) else None
        for key in _STAT_KEYS:
            out[key] = raw.get(key)
        return out
    v1 = os.path.join(root, "memory", "memory.stat")
    with open(v1) as f:
        raw = _parse_stat(f.read())
    base = os.path.join(root, "memory")
    out = {"version": 1}
    for key, name in (
        ("current", "memory.usage_in_bytes"),
        ("max", "memory.limit_in_bytes"),
        ("peak", "memory.max_usage_in_bytes"),
    ):
        path = os.path.join(base, name)
        out[key] = _read_int(path) if os.path.exists(path) else None
    for key, v1_name in _STAT_KEYS.items():
        out[key] = raw.get(v1_name) if v1_name else None
    return out


def read_tmpfs(
    mounts_path: str = "/proc/self/mounts",
    statvfs: Callable[[str], Any] = os.statvfs,
) -> dict[str, int]:
    """Used bytes of every tmpfs mount (memory-backed emptyDirs included)."""
    out: dict[str, int] = {}
    with open(mounts_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3 or parts[2] != "tmpfs":
                continue
            mount = parts[1].replace("\\040", " ")
            if mount in out:
                continue
            st = statvfs(mount)
            out[mount] = int((st.f_blocks - st.f_bfree) * st.f_frsize)
    return out


def read_procs(proc_root: str = "/proc", top: int = TOP_PROCS) -> dict[str, Any]:
    """Container processes by RSS (bytes), with the anon / file / shmem split."""
    procs = []
    for pid in os.listdir(proc_root):
        if not pid.isdigit():
            continue
        try:
            with open(os.path.join(proc_root, pid, "status")) as f:
                fields = dict(
                    line.split(":", 1) for line in f.read().splitlines() if ":" in line
                )
        except OSError:  # the process exited between listdir and open
            continue

        def kb(name: str) -> int:
            value = fields.get(name, "0 kB").split()
            return int(value[0]) * 1024 if value else 0

        rss = kb("VmRSS")
        if rss == 0:
            continue
        procs.append(
            {
                "pid": int(pid),
                "name": fields.get("Name", "?").strip(),
                "rss": rss,
                "anon": kb("RssAnon"),
                "file": kb("RssFile"),
                "shmem": kb("RssShmem"),
            }
        )
    procs.sort(key=lambda p: p["rss"], reverse=True)
    totals = {k: sum(p[k] for p in procs) for k in ("rss", "anon", "file", "shmem")}
    return {"n": len(procs), "totals": totals, "top": procs[:top]}


def snapshot(
    cgroup_root: str = "/sys/fs/cgroup",
    mounts_path: str = "/proc/self/mounts",
    proc_root: str = "/proc",
    statvfs: Callable[[str], Any] = os.statvfs,
) -> dict[str, Any]:
    """One telemetry record; a failed part is named in ``errors``."""
    snap: dict[str, Any] = {"t": round(time.time(), 3), "errors": []}
    for key, fn in (
        ("cgroup", lambda: read_cgroup(cgroup_root)),
        ("tmpfs", lambda: read_tmpfs(mounts_path, statvfs)),
        ("procs", lambda: read_procs(proc_root)),
    ):
        try:
            snap[key] = fn()
        except Exception as exc:  # telemetry must never stop the engine
            snap[key] = None
            snap["errors"].append(f"{key}: {type(exc).__name__}: {exc}")
    return snap


def format_line(snap: dict[str, Any]) -> str:
    return LINE_PREFIX + json.dumps(snap, separators=(",", ":"), sort_keys=True)


def _loop(interval_s: int, stop: threading.Event, emit: Callable[[str], None]) -> None:
    while True:
        try:
            emit(format_line(snapshot()))
        except Exception:  # pragma: no cover - defensive
            logger.exception("Memstat: snapshot failed; continuing")
        if stop.wait(interval_s):
            return


def _is_local_dp_leader(vllm_config: Any) -> bool:
    pc = getattr(vllm_config, "parallel_config", None)
    if pc is None:
        return True
    local = getattr(pc, "data_parallel_rank_local", None)
    if local is None:
        size_local = int(getattr(pc, "data_parallel_size_local", 1) or 1)
        local = int(getattr(pc, "data_parallel_rank", 0) or 0) % max(size_local, 1)
    return int(local) == 0


def maybe_start_memstat(
    vllm_config: Any = None,
    emit: Callable[[str], None] | None = None,
    stop: threading.Event | None = None,
) -> threading.Thread | None:
    """Start the pod's memstat thread; idempotent; no-op unless the flag is on
    and this engine core is local DP rank 0. Called from the engine-core
    process entry point."""
    global _thread
    interval = int(envs.VLLM_TPU_MEMSTAT_INTERVAL_S)
    if interval <= 0 or not _is_local_dp_leader(vllm_config):
        return None
    with _lock:
        if _thread is not None and _thread.is_alive():
            return _thread
        out = emit or logger.info
        _thread = threading.Thread(
            target=_loop,
            args=(interval, stop or threading.Event(), out),
            name="q38-memstat",
            daemon=True,
        )
        _thread.start()
    logger.info("Memstat: on (VLLM_TPU_MEMSTAT_INTERVAL_S=%d)", interval)
    return _thread
