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
"""Worker CPU placement (TPU_WORKER_CPU_PIN).

A TPU worker's serving thread (the thread that runs execute_model,
sample_tokens and the draft loop) is single-thread CPU-bound for most of a
decode step, and every rank of the slice waits for the slowest one. With no
affinity set, the kernel may run it on the SMT sibling of a busy thread, which
halves its speed while the other thread runs, or on the socket that is not
attached to the worker's TPU.

With the flag on, the first execute_model call of each worker:

  * finds the NUMA node of the TPU chip the process opened (its /dev/vfio/<group>
    descriptors -> /sys/kernel/iommu_groups/<group>/devices/*/numa_node);
  * gives the serving thread one whole physical core on that node (both SMT
    siblings); the core is picked by LOCAL_RANK, so the workers of one host
    never share one, and the block reserved per node covers every TPU device
    of the host (max(LOCAL_WORLD_SIZE, /dev/vfio groups)), so no pinned thread
    of any worker lands on another worker's core. Processes that are not TPU
    workers (API server, EngineCore, Ray) are not moved;
  * moves every other thread of the process onto the rest of that node's CPUs
    (the node minus every worker's reserved core), and keeps doing so for
    threads created later (a daemon sweeper, every few seconds).

The serving thread is pinned at most once per thread; nothing runs on the
per-step path after that. Placement only: no computation changes. Any failure
(no vfio descriptor, unknown node, affinity syscall refused) logs once and
leaves the process unpinned.
"""

from __future__ import annotations

import contextlib
import os
import re
import threading

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_VFIO_GROUP = re.compile(r"^/dev/vfio/(\d+)$")


def parse_cpulist(text: str) -> list[int]:
    """'0-3,8,10-11' -> [0, 1, 2, 3, 8, 10, 11]."""
    out: list[int] = []
    for part in text.strip().split(","):
        part = part.strip()
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


def tpu_numa_nodes(proc_fd: str = "/proc/self/fd", sys_root: str = "/sys") -> set[int]:
    """NUMA nodes of the vfio devices this process holds open."""
    nodes: set[int] = set()
    try:
        fds = os.listdir(proc_fd)
    except OSError:
        return nodes
    groups = set()
    for fd in fds:
        try:
            m = _VFIO_GROUP.match(os.readlink(os.path.join(proc_fd, fd)))
        except OSError:
            continue
        if m:
            groups.add(m.group(1))
    for g in groups:
        devdir = os.path.join(sys_root, "kernel/iommu_groups", g, "devices")
        try:
            devs = os.listdir(devdir)
        except OSError:
            continue
        for dev in devs:
            try:
                node = int(_read(os.path.join(devdir, dev, "numa_node")))
            except (OSError, ValueError):
                continue
            if node >= 0:
                nodes.add(node)
    return nodes


def vfio_group_count(dev_vfio: str = "/dev/vfio") -> int:
    """Number of TPU vfio groups on this host (one per device a worker opens)."""
    try:
        return sum(1 for n in os.listdir(dev_vfio) if n.isdigit())
    except OSError:
        return 0


def node_physical_cores(
    node: int, allowed: set[int], sys_root: str = "/sys"
) -> list[tuple[int, ...]]:
    """Physical cores of `node` as tuples of their allowed SMT siblings,
    ordered by lowest CPU id. Cores with any sibling outside `allowed` are
    dropped, so a reserved core is always reserved whole."""
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
    cores: list[tuple[int, ...]], local_rank: int, local_world_size: int, skip: int
) -> tuple[set[int], set[int]] | None:
    """(serving-thread CPUs, other-thread CPUs) for one worker, or None.

    Worker `local_rank` reserves core `skip + local_rank`; the other threads
    get the node minus the reserved cores of all `local_world_size` workers
    (whichever node each actually lands on, its core index is unique)."""
    if local_rank < 0 or local_rank >= local_world_size:
        return None
    if skip + local_world_size >= len(cores):
        return None  # leave at least one core for everything else
    reserved = cores[skip + local_rank]
    taken = {c for core in cores[skip : skip + local_world_size] for c in core}
    others = {c for core in cores for c in core} - taken
    return set(reserved), others


class WorkerCpuPinner:
    """Pins the calling (serving) thread on first `maybe_pin()`."""

    def __init__(
        self,
        enabled: bool,
        local_rank: int,
        local_world_size: int,
        skip: int = 2,
        sweep_s: float = 5.0,
    ):
        self.enabled = enabled
        self.local_rank = local_rank
        self.local_world_size = local_world_size
        self.skip = skip
        self.sweep_s = sweep_s
        self.serving_tid: int | None = None
        self.serving_cpus: set[int] | None = None
        self.other_cpus: set[int] | None = None
        self._failed = False
        self._seen: set[int] = set()
        self._lock = threading.Lock()
        self._sweeper: threading.Thread | None = None

    def maybe_pin(self) -> None:
        if not self.enabled or self._failed:
            return
        tid = threading.get_native_id()
        if tid == self.serving_tid:
            return
        try:
            self._pin(tid)
        except Exception as e:  # placement only: never fail the step
            self._failed = True
            logger.warning("TPU_WORKER_CPU_PIN: not pinned (%s)", e)

    def _pin(self, tid: int) -> None:
        if self.serving_cpus is None:
            nodes = tpu_numa_nodes()
            if len(nodes) != 1:
                raise RuntimeError(f"TPU NUMA nodes {sorted(nodes)} (need exactly one)")
            node = next(iter(nodes))
            allowed = os.sched_getaffinity(0)
            cores = node_physical_cores(node, allowed)
            # TPU_LOCAL_RANK_OFFSET servers sharing a host each see only their
            # own LOCAL_WORLD_SIZE; reserving per host device keeps their
            # reserved blocks out of each other's "other" CPUs.
            width = max(self.local_world_size, vfio_group_count())
            p = plan(cores, self.local_rank, width, self.skip)
            if p is None:
                raise RuntimeError(
                    f"node {node} has {len(cores)} usable cores for local rank "
                    f"{self.local_rank}/{self.local_world_size} (skip {self.skip})"
                )
            self.serving_cpus, self.other_cpus = p
            self.node = node
        assert self.serving_cpus is not None and self.other_cpus is not None
        if self.serving_tid is not None:
            logger.warning(
                "TPU_WORKER_CPU_PIN: serving thread changed %d -> %d; re-pinning",
                self.serving_tid,
                tid,
            )
            # The old serving thread may have exited (OSError).
            with contextlib.suppress(OSError):
                os.sched_setaffinity(self.serving_tid, self.other_cpus)
        # Other threads first, so nothing new inherits the serving core after
        # the serving thread is pinned (threads spawned later by the serving
        # thread itself are caught by the sweeper).
        self.serving_tid = tid
        moved = self._sweep()
        os.sched_setaffinity(0, self.serving_cpus)
        self.pins = getattr(self, "pins", 0) + 1
        (logger.warning if self.pins == 1 else logger.debug)(
            "TPU_WORKER_CPU_PIN local_rank=%d node=%d serving_tid=%d "
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
        """Move every thread except the serving thread to the other CPUs.
        Returns how many threads were moved."""
        moved = 0
        with self._lock:
            try:
                tids = [int(t) for t in os.listdir("/proc/self/task")]
            except OSError:
                return 0
            if self.other_cpus is None:
                return 0  # not planned yet
            for t in tids:
                if t == self.serving_tid or t in self._seen:
                    continue
                try:
                    os.sched_setaffinity(t, self.other_cpus)
                    moved += 1
                except OSError:
                    pass  # exited
                self._seen.add(t)
            self._seen &= set(tids)
        return moved

    def _sweep_loop(self) -> None:
        import time

        assert self.other_cpus is not None  # started after the first pin
        os.sched_setaffinity(0, self.other_cpus)
        while True:
            time.sleep(self.sweep_s)
            self._sweep()
