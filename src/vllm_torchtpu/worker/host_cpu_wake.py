# SPDX-License-Identifier: Apache-2.0
"""Keep the worker's host CPUs out of deep idle states while serving.

A decode step's target forward is one long device program (about 9 ms at
low concurrency). The programs that consume its outputs are launched only
after the worker's runtime threads handle the forward's completion. Those
threads sleep through the forward, so their CPU can enter a deep C-state,
and leaving it costs hundreds of microseconds. Each worker pays this wake
latency independently, and the first cross-core collective after the forward
(the drafter) waits for the slowest worker, so every step pays the worst of
eight wake latencies.

``TPU_HOST_CPU_WAKE`` (modes joined by ``,`` or ``+``, default off) acts after warm-up:

* ``pin``: pin all of the worker's threads to ``TPU_HOST_CPU_WAKE_CORES``
  whole physical cores (default 4) on the worker's NUMA node, disjoint from
  the other workers' cores.
* ``spin``: ``pin`` plus one ``SCHED_IDLE`` busy loop on each pinned CPU. The
  loop keeps the CPUs from halting and yields at once to any runnable worker
  thread, so a wake-up costs a context switch instead of an idle exit.
* ``dma``: hold ``/dev/cpu_dma_latency`` at 0 us (PM QoS) when it exists.
* ``cycle``: diagnostic A/B inside one server. Every worker steps through
  off -> pin -> spin in wall-clock aligned windows of
  ``TPU_HOST_CPU_WAKE_PERIOD_S`` seconds (default 90), and logs each switch.
* ``stats``: every ``TPU_HOST_CPU_WAKE_STATS_S`` seconds (default 60) log the
  worker's busiest threads (CPU share, voluntary and involuntary context
  switches, last CPU) and, on local rank 0, the host's steal time.
"""

import threading
import time

import atexit
import os
import struct
import subprocess
import sys

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_dma_fd: int | None = None
_spinner: subprocess.Popen | None = None

# One process per CPU, forked from one interpreter: each child pins itself,
# drops to SCHED_IDLE and dies with the worker.
_SPIN_SRC = r"""
import ctypes, os, signal, sys
cpus = [int(c) for c in sys.argv[1].split(",")]
expect_parent = int(sys.argv[2])
mine = cpus[0]
for c in cpus[1:]:
    root = os.getpid()
    if os.fork() == 0:
        mine, expect_parent = c, root
        break
ctypes.CDLL(None).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
if os.getppid() != expect_parent:
    os._exit(0)
os.sched_setaffinity(0, [mine])
os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
while True:
    pass
"""


def _read_list(path: str) -> list[int]:
    out: list[int] = []
    with open(path) as f:
        for part in f.read().strip().split(","):
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-")
                out.extend(range(int(lo), int(hi) + 1))
            else:
                out.append(int(part))
    return out


def _numa_nodes() -> list[list[int]]:
    base = "/sys/devices/system/node"
    try:
        names = sorted(
            (n for n in os.listdir(base) if n.startswith("node") and n[4:].isdigit()),
            key=lambda n: int(n[4:]),
        )
    except OSError:
        names = []
    nodes = [_read_list(f"{base}/{n}/cpulist") for n in names]
    return [n for n in nodes if n] or [sorted(os.sched_getaffinity(0))]


def _physical_cores(cpus: list[int]) -> list[list[int]]:
    seen: dict[tuple[int, ...], None] = {}
    for c in cpus:
        try:
            sib = tuple(
                _read_list(
                    f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list"
                )
            )
        except OSError:
            sib = (c,)
        seen.setdefault(sib, None)
    return [list(s) for s in seen]


def pick_cpus(local_rank: int, local_world: int, cores_per_rank: int) -> list[int]:
    """CPUs for one worker: ``cores_per_rank`` whole physical cores on the
    NUMA node that holds the worker's share of the local ranks. Core 0 of each
    node is left to the system."""
    allowed = os.sched_getaffinity(0)
    nodes = [[c for c in n if c in allowed] for n in _numa_nodes()]
    nodes = [n for n in nodes if n]
    node = nodes[min(local_rank * len(nodes) // max(local_world, 1), len(nodes) - 1)]
    ranks_on_node = max(1, -(-local_world // len(nodes)))
    slot = local_rank % ranks_on_node
    cores = _physical_cores(node)[1:]
    chosen = cores[slot * cores_per_rank : (slot + 1) * cores_per_rank]
    if not chosen:
        chosen = cores[:cores_per_rank]
    return sorted(c for core in chosen for c in core if c in allowed)


def _pin_all_threads(cpus) -> int:
    n = 0
    for tid in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(tid), cpus)
            n += 1
        except OSError:
            pass
    return n


def _start_spinner(cpus: list[int]) -> None:
    global _spinner
    if _spinner is None:
        _spinner = subprocess.Popen(
            [sys.executable, "-c", _SPIN_SRC, ",".join(map(str, cpus)), str(os.getpid())],
            start_new_session=True,
        )


def _stop_spinner() -> None:
    global _spinner
    if _spinner is not None:
        try:
            os.killpg(_spinner.pid, 9)
        except OSError:
            pass
        try:
            _spinner.wait(timeout=5)
        except Exception:
            pass
        _spinner = None


def _stop() -> None:
    global _dma_fd
    _stop_spinner()
    if _dma_fd is not None:
        try:
            os.close(_dma_fd)
        except OSError:
            pass
        _dma_fd = None


def _set_state(state: str, cpus: list[int], all_cpus) -> str:
    """state: off | pin | spin."""
    if state == "off":
        _stop_spinner()
        n = _pin_all_threads(all_cpus)
    else:
        n = _pin_all_threads(cpus)
        if state == "spin":
            _start_spinner(cpus)
        else:
            _stop_spinner()
    return f"{state} pinned_threads={n}"


def _thread_sample() -> dict[int, tuple[str, int, int, int, int]]:
    out = {}
    for tid in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{tid}/stat") as f:
                st = f.read()
            comm = st[st.index("(") + 1 : st.rindex(")")]
            fields = st[st.rindex(")") + 2 :].split()
            ticks = int(fields[11]) + int(fields[12])
            cpu = int(fields[36])
            vol = invol = 0
            with open(f"/proc/self/task/{tid}/status") as f:
                for line in f:
                    if line.startswith("voluntary_ctxt_switches"):
                        vol = int(line.split()[1])
                    elif line.startswith("nonvoluntary_ctxt_switches"):
                        invol = int(line.split()[1])
            out[int(tid)] = (comm, ticks, vol, invol, cpu)
        except (OSError, ValueError, IndexError):
            pass
    return out


def _steal() -> tuple[int, int]:
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    return v[7], sum(v)


def _loop(rank: int, local_rank: int, modes: set[str], cpus: list[int], all_cpus) -> None:
    period = float(os.environ.get("TPU_HOST_CPU_WAKE_PERIOD_S", "90"))
    stats_s = float(os.environ.get("TPU_HOST_CPU_WAKE_STATS_S", "60"))
    hz = os.sysconf("SC_CLK_TCK")
    states = ("off", "pin", "spin")
    cur = None
    prev = _thread_sample() if "stats" in modes else None
    prev_t = time.time()
    prev_steal = _steal() if ("stats" in modes and local_rank == 0) else None
    next_stats = prev_t + stats_s
    while True:
        now = time.time()
        if "cycle" in modes:
            want = states[int(now // period) % len(states)]
            if want != cur:
                msg = _set_state(want, cpus, all_cpus)
                logger.info(
                    "TPU_HOST_CPU_WAKE cycle rank %d window=%d t=%.3f state=%s",
                    rank, int(now // period), now, msg,
                )
                cur = want
        if prev is not None and now >= next_stats:
            cur_s = _thread_sample()
            dt = max(now - prev_t, 1e-3)
            rows = []
            for tid, (comm, ticks, vol, invol, cpu) in cur_s.items():
                if tid in prev:
                    p = prev[tid]
                    rows.append((ticks - p[1], tid, comm, vol - p[2], invol - p[3], cpu))
            rows.sort(reverse=True)
            top = " ".join(
                f"{tid}:{comm}:{100.0 * d / hz / dt:.0f}%:v{vol}:i{invol}:c{cpu}"
                for d, tid, comm, vol, invol, cpu in rows[:8]
            )
            extra = ""
            if prev_steal is not None:
                s = _steal()
                extra = f" host_steal={100.0 * (s[0] - prev_steal[0]) / max(s[1] - prev_steal[1], 1):.3f}%"
                prev_steal = s
            logger.info(
                "TPU_HOST_CPU_WAKE stats rank %d t=%.3f state=%s%s top=%s",
                rank, now, cur, extra, top,
            )
            prev, prev_t = cur_s, now
            next_stats = now + stats_s
        wake = next_stats if prev is not None else now + 3600
        if "cycle" in modes:
            wake = min(wake, (int(now // period) + 1) * period + 0.01)
        time.sleep(max(0.05, wake - time.time()))


def apply(rank: int, local_rank: int, local_world: int) -> None:
    """Apply ``TPU_HOST_CPU_WAKE`` once, after warm-up. No-op when unset."""
    global _dma_fd
    # "+" also separates modes: recipe env lists are comma-separated.
    raw = os.environ.get("TPU_HOST_CPU_WAKE", "").replace("+", ",")
    modes = {m.strip() for m in raw.split(",") if m.strip()}
    if not modes:
        return
    all_cpus = sorted(os.sched_getaffinity(0))
    cpus = pick_cpus(local_rank, local_world, int(os.environ.get("TPU_HOST_CPU_WAKE_CORES", "4")))
    done: list[str] = [f"cpus={cpus}"]
    if "dma" in modes and _dma_fd is None:
        try:
            fd = os.open("/dev/cpu_dma_latency", os.O_RDWR)
            os.write(fd, struct.pack("i", 0))
            _dma_fd = fd
            done.append("dma=0us")
        except OSError as e:
            done.append(f"dma=unavailable({e.strerror})")
    if "cycle" not in modes:
        if "spin" in modes:
            done.append(_set_state("spin", cpus, all_cpus))
        elif "pin" in modes:
            done.append(_set_state("pin", cpus, all_cpus))
    atexit.register(_stop)
    if modes & {"cycle", "stats"}:
        threading.Thread(
            target=_loop,
            args=(rank, local_rank, modes, cpus, all_cpus),
            name="tpu-host-cpu-wake",
            daemon=True,
        ).start()
    logger.info(
        "TPU_HOST_CPU_WAKE=%s active on rank %d (local %d/%d): %s",
        ",".join(sorted(modes)), rank, local_rank, local_world, "; ".join(done),
    )
