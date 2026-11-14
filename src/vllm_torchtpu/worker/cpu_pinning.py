# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pin a TPU worker's threads to host CPUs near its chip (VLLM_TPU_CPU_PIN).

A tpu7x 2x2x1 host has two NUMA nodes; two chips (four TensorCore PCI
functions) hang off each. Without pinning, the kernel runs a worker's host
threads anywhere, so a worker can drive a chip from the far socket or share a
physical core with another worker's hot thread. After warmup:

  ``numa``   every thread of the worker may use any CPU of its chip's node.
  ``slice``  each worker gets its own physical cores (with their SMT siblings)
             on its chip's node; the node's last ``VLLM_TPU_CPU_PIN_RESERVE``
             cores stay free for the engine core, API server and the kernel.

The chip's node comes from the worker's open /dev/vfio/<group> file, so the
binding follows whatever device libtpu opened. VLLM_TPU_CPU_PROBE_S=N logs
each worker's thread placement and context switches every N seconds.
"""

import glob
import os
import threading
import time

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _read(path: str) -> str:
    with open(path) as f:
        return f.read().strip()


def _cpulist(text: str) -> list[int]:
    cpus: list[int] = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            cpus.extend(range(int(lo), int(hi) + 1))
        elif part:
            cpus.append(int(part))
    return cpus


def _group_devices(group: str) -> list[tuple[int, str]]:
    """(NUMA node, PCI address) of the devices in an IOMMU group."""
    out = []
    for dev in sorted(glob.glob(f"/sys/kernel/iommu_groups/{group}/devices/*")):
        try:
            out.append((int(_read(f"{dev}/numa_node")), os.path.basename(dev)))
        except (OSError, ValueError):
            continue
    return out


def my_chip_devices() -> list[tuple[int, str]]:
    """(node, PCI address) behind this process's open /dev/vfio/<group> files."""
    devices = set()
    for fd in glob.glob("/proc/self/fd/*"):
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        group = os.path.basename(target)
        if target.startswith("/dev/vfio/") and group.isdigit():
            devices.update(_group_devices(group))
    return sorted(devices)


def all_chip_devices() -> list[tuple[int, str]]:
    """(node, PCI address) of every TPU vfio group on the host."""
    devices = set()
    for path in glob.glob("/dev/vfio/[0-9]*"):
        devices.update(_group_devices(os.path.basename(path)))
    return sorted(devices)


def physical_cores(node: int) -> list[tuple[int, ...]]:
    """The node's physical cores, each as its sorted SMT sibling tuple."""
    cores = []
    for cpu in _cpulist(_read(f"/sys/devices/system/node/node{node}/cpulist")):
        sib = tuple(
            _cpulist(
                _read(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
            )
        )
        if sib not in cores:
            cores.append(sib)
    return sorted(cores)


def plan(
    mode: str,
    node: int,
    slot: int,
    slots: int,
    reserve: int,
) -> set[int]:
    """CPUs for the ``slot``-th of ``slots`` workers on ``node``."""
    cores = physical_cores(node)
    if mode == "numa":
        return {cpu for core in cores for cpu in core}
    usable = cores[: max(slots, len(cores) - reserve)]
    per = max(1, len(usable) // slots)
    mine = usable[slot * per : (slot + 1) * per]
    return {cpu for core in mine for cpu in core}


def pin_all_threads(cpus: set[int]) -> int:
    """Set the affinity of every existing thread; new ones inherit it."""
    pinned = 0
    failed = 0
    first_err: OSError | None = None
    for task in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(task), cpus)
            pinned += 1
        except ProcessLookupError:
            continue
        except OSError as err:
            failed += 1
            if first_err is None:
                first_err = err
    if failed:
        logger.warning(
            "CPU pin failed on %d thread(s) for cpus %s: %s",
            failed,
            sorted(cpus),
            first_err,
        )
    return pinned


def _probe(interval: float, rank: int) -> None:
    last: dict[int, tuple[int, int]] = {}
    while True:
        time.sleep(interval)
        busy = []
        migrated = 0
        for task in os.listdir("/proc/self/task"):
            try:
                stat = _read(f"/proc/self/task/{task}/stat").rsplit(")", 1)[1].split()
                status = _read(f"/proc/self/task/{task}/status")
            except OSError:
                continue
            cpu, ticks = int(stat[36]), int(stat[11]) + int(stat[12])
            nvcsw = int(status.rsplit("nonvoluntary_ctxt_switches:", 1)[1].split()[0])
            prev = last.get(int(task))
            last[int(task)] = (cpu, ticks)
            if prev is not None:
                migrated += prev[0] != cpu
                if ticks - prev[1] > 0:
                    busy.append((ticks - prev[1], int(task), cpu, nvcsw))
        busy.sort(reverse=True)
        logger.info(
            "CPUPROBE rank=%d threads=%d migrated=%d top(ticks,tid,cpu,nvcsw)=%s",
            rank,
            len(last),
            migrated,
            busy[:6],
        )


def apply(rank: int, local_rank: int) -> None:
    mode = envs.VLLM_TPU_CPU_PIN
    if envs.VLLM_TPU_CPU_PROBE_S > 0:
        threading.Thread(
            target=_probe,
            args=(envs.VLLM_TPU_CPU_PROBE_S, rank),
            name="cpu-probe",
            daemon=True,
        ).start()
    if mode == "off" or mode is None:
        return
    if mode not in ("numa", "slice"):
        raise ValueError(f"VLLM_TPU_CPU_PIN={mode!r}: expected off, numa or slice")
    mine = my_chip_devices()
    nodes = {node for node, _ in mine}
    if len(nodes) != 1 or min(nodes) < 0:
        logger.warning("CPU pin skipped: rank %d chip devices %s", rank, mine)
        return
    node = nodes.pop()
    # Slot = this worker's device among the node's TPU devices, so the workers
    # sharing a node split its cores whatever their local ranks are.
    peers = [d for d in all_chip_devices() if d[0] == node]
    slots = max(1, len(peers))
    slot = peers.index(mine[0]) if mine[0] in peers else local_rank % slots
    cpus = plan(mode, node, slot, slots, envs.VLLM_TPU_CPU_PIN_RESERVE)
    n = pin_all_threads(cpus)
    logger.info(
        "CPU pin: rank %d local_rank %d node %d mode %s slot %d/%d "
        "cpus %s (%d threads)",
        rank,
        local_rank,
        node,
        mode,
        slot,
        slots,
        sorted(cpus),
        n,
    )
