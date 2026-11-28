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
"""Bind a TPU worker process to the NUMA node of its TPU chip."""

import ctypes
import glob
import os

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _parse_cpulist(text: str) -> list[int]:
    """Parse a sysfs CPU list: "0-3,8,10-11" -> [0, 1, 2, 3, 8, 10, 11]."""
    cpus: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        first, _, last = part.partition("-")
        cpus.extend(range(int(first), int(last or first) + 1))
    return cpus


def _tpu_chip_numa_node(chip_index: int) -> tuple[str | None, int | None]:
    """Return the PCI address and NUMA node of TPU chip `chip_index`.

    TPU chips are the PCI devices with Google's vendor id and the
    accelerator device class, in PCI address order, which matches the
    runtime's chip numbering. Either value is None when sysfs lacks it.
    """
    tpu_devices: list[str] = []
    for device_dir in glob.glob("/sys/bus/pci/devices/*"):
        try:
            with open(os.path.join(device_dir, "vendor")) as f:
                vendor = f.read().strip()
            with open(os.path.join(device_dir, "class")) as f:
                device_class = f.read().strip()
        except OSError:
            continue
        if vendor == "0x1ae0" and device_class.startswith("0xff00"):
            tpu_devices.append(os.path.basename(device_dir))
    tpu_devices.sort()
    if chip_index >= len(tpu_devices):
        return None, None
    pci_address = tpu_devices[chip_index]
    try:
        with open(f"/sys/bus/pci/devices/{pci_address}/numa_node") as f:
            node = int(f.read().strip())
    except OSError:
        return pci_address, None
    return pci_address, (node if node >= 0 else None)


def bind_worker_to_chip_numa_node(chip_index: int) -> None:
    """Bind this worker to the NUMA node its TPU chip is attached to.

    Runtime threads started later inherit the node's CPU set, and later
    allocations prefer its memory. This avoids remote-socket memory traffic
    and reduces host launch skew between workers. The inherited CPU set is
    only ever narrowed, so a launcher's reservation (taskset, numactl, a
    cgroup) stays in force. Hosts without NUMA information are left as is.
    """
    pci_address, node = _tpu_chip_numa_node(chip_index)
    if node is None:
        logger.info(
            "Worker NUMA bind: chip %d (%s) has no NUMA node; placement left "
            "to the OS scheduler.",
            chip_index,
            pci_address,
        )
        return
    try:
        with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
            node_cpus = set(_parse_cpulist(f.read()))
        inherited_cpus = os.sched_getaffinity(0)
        cpus = node_cpus & inherited_cpus
        if cpus:
            os.sched_setaffinity(0, cpus)
        else:
            # The launcher left this worker no CPU on its chip's node: keep
            # the inherited set and only move the memory policy.
            logger.info(
                "Worker NUMA bind: chip %d (%s) node %d shares no CPU with "
                "the inherited set of %d; CPU set left as inherited.",
                chip_index,
                pci_address,
                node,
                len(inherited_cpus),
            )
            cpus = inherited_cpus
    except OSError as e:
        logger.warning(
            "Worker NUMA bind: chip %d (%s) node %d: CPU set not applied (%s).",
            chip_index,
            pci_address,
            node,
            e,
        )
        return
    memory_policy = "memory preferred on that node"
    try:
        ctypes.CDLL("libnuma.so.1").numa_set_preferred(ctypes.c_int(node))
    except (OSError, AttributeError):
        memory_policy = "memory policy unchanged (libnuma unavailable)"
    logger.info(
        "Worker NUMA bind: chip %d (%s) -> node %d, %d CPUs, %s.",
        chip_index,
        pci_address,
        node,
        len(cpus),
        memory_policy,
    )
