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
"""NUMA-local CPU affinity pinning for TPU worker serving threads."""

from __future__ import annotations

import contextlib
import os
import re
import threading

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_VFIO_GROUP = re.compile(r"^/dev/vfio/(\d+)$")
_NUMA_NODE = re.compile(r"^node(\d+)$")
_OS_HOUSEKEEPING_CORES = 2


def parse_cpulist(text: str) -> list[int]:
    """Parse sysfs cpulist syntax `'0-3,8,10-11'` into `[0, 1, 2, 3, 8, 10, 11]`."""
    out: list[int] = []
    for raw in text.strip().split(","):
        part = raw.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def _open_vfio_groups(proc_fd: str = "/proc/self/fd") -> set[str]:
    groups: set[str] = set()
    try:
        fds = os.listdir(proc_fd)
    except OSError:
        return groups
    for fd in fds:
        try:
            m = _VFIO_GROUP.match(os.readlink(os.path.join(proc_fd, fd)))
        except OSError:
            continue
        if m:
            groups.add(m.group(1))
    return groups


def _group_devices(group: str, sys_root: str = "/sys") -> list[tuple[int, str]]:
    devdir = os.path.join(sys_root, "kernel/iommu_groups", group, "devices")
    try:
        devs = os.listdir(devdir)
    except OSError:
        return []
    out: list[tuple[int, str]] = []
    for dev in devs:
        try:
            node = int(_read(os.path.join(devdir, dev, "numa_node")))
        except (OSError, ValueError):
            continue
        if node >= 0:
            out.append((node, dev))
    return out


def tpu_numa_nodes(proc_fd: str = "/proc/self/fd", sys_root: str = "/sys") -> set[int]:
    """Return the NUMA nodes of the `/dev/vfio` groups held open by this process."""
    return {
        node
        for group in _open_vfio_groups(proc_fd)
        for node, _ in _group_devices(group, sys_root)
    }


def vfio_group_count(dev_vfio: str = "/dev/vfio") -> int:
    """Count numeric `/dev/vfio` group entries visible in this container/host."""
    try:
        return sum(1 for name in os.listdir(dev_vfio) if name.isdigit())
    except OSError:
        return 0


def _cpu_numa_nodes(sys_root: str = "/sys") -> set[int]:
    node_root = os.path.join(sys_root, "devices/system/node")
    try:
        entries = os.listdir(node_root)
    except OSError:
        return set()
    nodes: set[int] = set()
    for name in entries:
        m = _NUMA_NODE.match(name)
        if not m:
            continue
        try:
            if parse_cpulist(_read(os.path.join(node_root, name, "cpulist"))):
                nodes.add(int(m.group(1)))
        except (OSError, ValueError):
            continue
    return nodes


def resolve_node_rank(
    node: int,
    local_rank: int,
    local_world_size: int,
    proc_fd: str = "/proc/self/fd",
    dev_vfio: str = "/dev/vfio",
    sys_root: str = "/sys",
) -> tuple[int, int]:
    """Return `(node_local_rank, node_rank_width)` on `node`."""
    visible_by_node: dict[int, set[str]] = {}
    try:
        visible_groups = [g for g in os.listdir(dev_vfio) if g.isdigit()]
    except OSError:
        visible_groups = []
    for group in visible_groups:
        for dev_node, bdf in _group_devices(group, sys_root):
            visible_by_node.setdefault(dev_node, set()).add(bdf)
    total_visible = sum(len(bdfs) for bdfs in visible_by_node.values())
    node_bdfs = sorted(visible_by_node.get(node, ()))
    if node_bdfs and total_visible >= local_world_size:
        open_bdfs = sorted(
            {
                bdf
                for group in _open_vfio_groups(proc_fd)
                for dev_node, bdf in _group_devices(group, sys_root)
                if dev_node == node
            }
        )
        width = len(node_bdfs)
        if len(open_bdfs) == 1 and open_bdfs[0] in node_bdfs:
            return node_bdfs.index(open_bdfs[0]), width
        return local_rank % width, width
    cpu_nodes = _cpu_numa_nodes(sys_root)
    num_sockets = (
        len(cpu_nodes)
        if node in cpu_nodes and len(cpu_nodes) > 1
        else max(1, len(visible_by_node))
    )
    ranks_per_node = max(1, (local_world_size + num_sockets - 1) // num_sockets)
    visible_count = (
        vfio_group_count() if dev_vfio == "/dev/vfio" else vfio_group_count(dev_vfio)
    )
    width = max(ranks_per_node, visible_count)
    return local_rank % width, width


def node_physical_cores(
    node: int, allowed: set[int], sys_root: str = "/sys"
) -> list[tuple[int, ...]]:
    """Return whole physical cores on `node` whose SMT siblings are all in `allowed`."""
    cpus = parse_cpulist(_read(f"{sys_root}/devices/system/node/node{node}/cpulist"))
    seen: set[int] = set()
    cores: list[tuple[int, ...]] = []
    for cpu in sorted(cpus):
        if cpu in seen:
            continue
        sib = parse_cpulist(
            _read(
                f"{sys_root}/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
            )
        )
        seen.update(sib)
        if all(s in allowed for s in sib):
            cores.append(tuple(sorted(sib)))
    return cores


def plan(
    cores: list[tuple[int, ...]],
    local_rank: int,
    local_world_size: int,
    skip: int = _OS_HOUSEKEEPING_CORES,
) -> tuple[set[int], set[int]] | None:
    """Return `(serving_cpus, other_cpus)` for `local_rank`, or `None`."""
    if local_rank < 0 or local_rank >= local_world_size:
        return None
    if skip + local_world_size >= len(cores):
        return None
    reserved = cores[skip + local_rank]
    taken = {c for core in cores[skip : skip + local_world_size] for c in core}
    others = {c for core in cores for c in core} - taken
    return set(reserved), others


class WorkerCpuPinner:
    """Pins the calling TPU worker serving thread on its first `maybe_pin()` call."""

    def __init__(
        self,
        enabled: bool,
        local_rank: int,
        local_world_size: int,
        skip: int = _OS_HOUSEKEEPING_CORES,
        sweep_s: float = 5.0,
    ) -> None:
        self.enabled = enabled
        self.local_rank = local_rank
        self.local_world_size = local_world_size
        self.skip = skip
        self.sweep_s = sweep_s
        self.node: int = -1
        self.pins: int = 0
        self.serving_tid: int | None = None
        self.serving_cpus: set[int] | None = None
        self.other_cpus: set[int] | None = None
        self._failed = False
        self._seen: set[int] = set()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._sweeper: threading.Thread | None = None

    def close(self) -> None:
        self._stop_event.set()
        sweeper = self._sweeper
        if (
            sweeper is not None
            and sweeper.is_alive()
            and sweeper is not threading.current_thread()
        ):
            sweeper.join(timeout=1.0)
        self._sweeper = None

    def maybe_pin(self) -> None:
        if not self.enabled or self._failed or self._stop_event.is_set():
            return
        tid = threading.get_native_id()
        if tid == self.serving_tid:
            return
        try:
            self._pin(tid)
        except (OSError, RuntimeError, ValueError) as exc:
            self._failed = True
            logger.info_once("Skipping worker CPU affinity pin: %s", exc)

    def _pin(self, tid: int) -> None:
        if self.serving_cpus is None:
            nodes = tpu_numa_nodes()
            if len(nodes) != 1:
                raise RuntimeError(
                    f"TPU NUMA nodes {sorted(nodes)} (expected exactly one)"
                )
            node = next(iter(nodes))
            allowed = os.sched_getaffinity(0)
            cores = node_physical_cores(node, allowed)
            node_rank, width = resolve_node_rank(
                node, self.local_rank, self.local_world_size
            )
            planned = plan(cores, node_rank, width, self.skip)
            if planned is None:
                raise RuntimeError(
                    f"node {node} has {len(cores)} allowed cores for "
                    f"node_rank={node_rank}/{width} (skip={self.skip})"
                )
            self.serving_cpus, self.other_cpus = planned
            self.node = node
        assert self.serving_cpus is not None and self.other_cpus is not None
        if self.serving_tid is not None:
            logger.info(
                "Worker serving thread changed %d -> %d; re-pinning",
                self.serving_tid,
                tid,
            )
            with contextlib.suppress(OSError):
                os.sched_setaffinity(self.serving_tid, self.other_cpus)
        self.serving_tid = tid
        moved = self._sweep()
        os.sched_setaffinity(0, self.serving_cpus)
        self.pins += 1
        log_fn = logger.info if self.pins == 1 else logger.debug
        log_fn(
            "Pinned TPU worker CPU affinity: local_rank=%d node=%d serving_tid=%d "
            "serving_cpus=%s other_cpus=%d threads_moved=%d",
            self.local_rank,
            self.node,
            tid,
            sorted(self.serving_cpus),
            len(self.other_cpus),
            moved,
        )
        if self._sweeper is None and self.sweep_s > 0:
            self._sweeper = threading.Thread(
                target=self._sweep_loop, name="tpu-cpu-pin-sweep", daemon=True
            )
            self._sweeper.start()

    def _sweep(self) -> int:
        moved = 0
        with self._lock:
            try:
                tids = [int(t) for t in os.listdir("/proc/self/task")]
            except OSError:
                return 0
            if self.other_cpus is None:
                return 0
            for t in tids:
                if t == self.serving_tid or t in self._seen:
                    continue
                with contextlib.suppress(OSError):
                    os.sched_setaffinity(t, self.other_cpus)
                    moved += 1
                self._seen.add(t)
            self._seen &= set(tids)
        return moved

    def _sweep_loop(self) -> None:
        assert self.other_cpus is not None
        with contextlib.suppress(OSError):
            os.sched_setaffinity(0, self.other_cpus)
        while not self._stop_event.wait(self.sweep_s):
            self._sweep()
