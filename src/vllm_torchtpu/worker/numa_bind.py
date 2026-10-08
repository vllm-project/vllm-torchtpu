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
"""Bind a TPU worker to the NUMA node its chip is attached to.

Called before the runtime starts so its threads inherit the node's CPU set
and its host buffers prefer the node's memory. The inherited CPU set is only
ever narrowed, so a launcher's reservation (taskset, numactl, a cgroup) stays
in force. Single-node hosts are left alone.
"""

import ctypes
import glob
import os

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_PCI_DEVICES = "/sys/bus/pci/devices"
_GOOGLE_VENDOR_ID = "0x1ae0"
# PCI class prefix of the TPU accelerator function.
_TPU_CLASS_PREFIX = "0xff00"


def parse_cpulist(text: str) -> list[int]:
    """Parse a sysfs cpulist: "0-3,8,10-11" -> [0, 1, 2, 3, 8, 10, 11]."""
    cpus: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        cpus.extend(range(int(lo), int(hi or lo) + 1))
    return cpus


def _read(path: str) -> str:
    with open(path) as f:
        return f.read().strip()


def tpu_pci_devices() -> list[str]:
    """PCI addresses of this host's TPU chips, in libtpu's chip order (PCI
    address order of the TPU accelerator functions)."""
    devices = []
    for path in glob.glob(os.path.join(_PCI_DEVICES, "*")):
        try:
            vendor = _read(os.path.join(path, "vendor"))
            pci_class = _read(os.path.join(path, "class"))
        except OSError:
            continue
        if vendor == _GOOGLE_VENDOR_ID and pci_class.startswith(_TPU_CLASS_PREFIX):
            devices.append(os.path.basename(path))
    return sorted(devices)


def chip_for_local_rank(
    local_rank: int, local_ranks: int, num_chips: int
) -> int | None:
    """The chip a host-local rank runs on, ranks spread evenly over chips
    (8 ranks on 4 chips -> 0,0,1,1,2,2,3,3). None if they do not divide."""
    if num_chips <= 0 or local_ranks % num_chips or not 0 <= local_rank < local_ranks:
        return None
    return local_rank // (local_ranks // num_chips)


def pci_numa_node(device: str) -> int | None:
    """NUMA node of a PCI device, None if sysfs does not say."""
    try:
        node = int(_read(os.path.join(_PCI_DEVICES, device, "numa_node")))
    except OSError:
        return None
    return node if node >= 0 else None


def bind_to_chip_numa_node(local_rank: int, local_ranks: int) -> None:
    """Narrow this process's CPU set to its chip's NUMA node and prefer the
    node's memory. `local_rank` is this worker's index among the host's
    `local_ranks` workers. Every failure is logged and leaves placement
    unchanged."""
    devices = tpu_pci_devices()
    chip = chip_for_local_rank(local_rank, local_ranks, len(devices))
    if chip is None:
        logger.info(
            "NUMA bind: local rank %d of %d does not map onto %d TPU chips; "
            "placement unchanged",
            local_rank,
            local_ranks,
            len(devices),
        )
        return
    device = devices[chip]
    node = pci_numa_node(device)
    if node is None:
        logger.info(
            "NUMA bind: chip %d (%s) has no NUMA node; placement unchanged",
            chip,
            device,
        )
        return
    try:
        node_cpus = set(
            parse_cpulist(_read(f"/sys/devices/system/node/node{node}/cpulist"))
        )
        inherited = os.sched_getaffinity(0)
        cpus = node_cpus & inherited
        if cpus:
            os.sched_setaffinity(0, cpus)
        else:
            # The launcher left no CPU on this node: keep its set and let only
            # the memory policy follow the chip.
            logger.info(
                "NUMA bind: chip %d (%s) node %d shares no CPU with the "
                "inherited set of %d; CPU set unchanged",
                chip,
                device,
                node,
                len(inherited),
            )
            cpus = inherited
    except OSError as e:
        logger.warning(
            "NUMA bind: chip %d (%s) node %d: CPU set not applied (%s)",
            chip,
            device,
            node,
            e,
        )
        return
    memory = "memory preferred there"
    try:
        ctypes.CDLL("libnuma.so.1").numa_set_preferred(ctypes.c_int(node))
    except (OSError, AttributeError):
        memory = "memory policy unchanged (libnuma unavailable)"
    logger.info(
        "NUMA bind: chip %d (%s) -> node %d, %d CPUs, %s",
        chip,
        device,
        node,
        len(cpus),
        memory,
    )
