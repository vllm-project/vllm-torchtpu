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
"""Pin a TPU worker's threads to the host CPUs of its chip's NUMA node.

Off by default: a model directory turns it on for its architectures with
`register_cpu_pinning("numa", *architectures)` from its registration hook.

A tpu7x 2x2x1 host has two NUMA nodes; two chips (four TensorCore PCI
functions) hang off each. Without pinning, the kernel runs a worker's host
threads anywhere, so a worker can drive its chip from the far socket. After
warmup, ``numa`` lets every thread of the worker use any CPU of its chip's
node that the process may already use (`os.sched_getaffinity`), so a
`taskset`/`numactl` or cgroup CPU set chosen by the user is never widened.

The chip's node comes from the worker's open /dev/vfio/<group> file, so the
binding follows whatever device libtpu opened.

Why not upstream's ``--numa-bind``: it binds the worker subprocess to a node
(CPUs and memory) when the process starts, before libtpu has opened the chip,
and for the whole lifetime, including warmup. Pinning here runs after warmup,
so the warmup's many parallel compiles keep every CPU of the host, and it
leaves memory placement alone. When ``--numa-bind`` is set, the worker skips
this pinning so the two never fight.
"""

import glob
import os

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


# PCI vendor ID of Google TPU chips.
_TPU_PCI_VENDOR = "0x1ae0"


def _group_devices(group: str) -> list[tuple[int, str]]:
    """(NUMA node, PCI address) of the TPU devices in an IOMMU group.

    Other vfio passthrough devices (another vendor) are left out, so they
    cannot decide this worker's node.
    """
    out = []
    for dev in sorted(glob.glob(f"/sys/kernel/iommu_groups/{group}/devices/*")):
        try:
            if _read(f"{dev}/vendor") != _TPU_PCI_VENDOR:
                continue
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


def node_cpus(node: int) -> set[int]:
    """Every CPU (SMT siblings included) of NUMA node ``node``."""
    return set(_cpulist(_read(f"/sys/devices/system/node/node{node}/cpulist")))


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


_MODES = ("numa",)
_POLICIES: dict[str, str] = {}


def register_cpu_pinning(mode: str, *architectures: str) -> None:
    """Pin the workers of these architectures near their chip after warmup.

    Called from a model directory's registration hook. ``mode`` is "numa"
    (the CPUs of the chip's NUMA node). Unregistered architectures are not
    pinned.
    """
    if mode not in _MODES:
        raise ValueError(f"CPU pinning mode {mode!r}: expected one of {_MODES}")
    for arch in architectures:
        _POLICIES[arch] = mode


def unregister_cpu_pinning(*architectures: str) -> None:
    for arch in architectures:
        _POLICIES.pop(arch, None)


def policy_for(architecture: str) -> str | None:
    """The registered mode for ``architecture``, else None."""
    return _POLICIES.get(architecture)


def apply(rank: int, local_rank: int, mode: str) -> None:
    """Pin this worker's threads per ``mode``; warns and skips when it cannot."""
    if mode not in _MODES:
        raise ValueError(f"CPU pinning mode {mode!r}: expected one of {_MODES}")
    mine = my_chip_devices()
    nodes = {node for node, _ in mine}
    if not mine:
        reason = "no TPU vfio device is open in this process"
    elif len(nodes) != 1:
        reason = "its TPU devices are on several NUMA nodes"
    elif min(nodes) < 0:
        reason = "its TPU devices report no NUMA node (-1)"
    else:
        reason = None
    if reason is not None:
        logger.warning(
            "CPU pin skipped: rank %d: %s (chip devices %s)", rank, reason, mine
        )
        return
    node = nodes.pop()
    try:
        on_node = node_cpus(node)
    except (OSError, ValueError) as err:
        logger.warning(
            "CPU pin skipped: rank %d: cannot read node %d CPUs: %s", rank, node, err
        )
        return
    # Never widen the CPU set the process was started with (taskset, numactl,
    # a cgroup cpuset): pin only within it.
    allowed = os.sched_getaffinity(0)
    cpus = on_node & allowed
    if not cpus:
        logger.warning(
            "CPU pin skipped: rank %d: node %d CPUs are outside the allowed set %s",
            rank,
            node,
            sorted(allowed),
        )
        return
    n = pin_all_threads(cpus)
    logger.info(
        "CPU pin: rank %d local_rank %d node %d mode %s cpus %s (%d threads)",
        rank,
        local_rank,
        node,
        mode,
        sorted(cpus),
        n,
    )
