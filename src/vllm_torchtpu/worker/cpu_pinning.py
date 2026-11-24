# SPDX-License-Identifier: Apache-2.0
"""Pin a TPU worker's threads to host CPUs near its chip (VLLM_TPU_CPU_PIN).

A tpu7x 2x2x1 host has two NUMA nodes; two chips (four TensorCore PCI
functions) hang off each. Without pinning, the kernel runs a worker's host
threads anywhere, so a worker can drive a chip from the far socket or share a
physical core with another worker's hot thread. After warmup:

  ``numa``   every thread of the worker may use any CPU of its chip's node.
  ``slice``  each worker gets its own physical cores (with their SMT siblings)
             on its chip's node; the node's last ``VLLM_TPU_CPU_PIN_RESERVE``
             cores stay free for the engine core, API server and the kernel.
  ``core``   the step thread (the thread that runs warmup and every step)
             alone on one physical core of its node: it runs on the core's
             first CPU and nothing is pinned to the SMT sibling. Every other
             thread of the worker may use the node's remaining CPUs; the
             local-rank-0 worker also keeps the engine core and API server
             processes off all workers' step cores. A watcher re-pins threads
             the step thread spawns later (they inherit its single CPU).

``physical_cores`` intersects the node's ``cpulist`` with the process's initial
CPU affinity mask (``os.sched_getaffinity(0)``) so container cpuset limits are
honored. On multi-tenant sub-host TPU slices whose containers share host NUMA
node CPUs without cpuset partitioning, ``node_peers`` indexes slots across all
host TPU PCI functions on that NUMA node so co-located slices select disjoint
CPU slices and step cores.

VLLM_TPU_CPU_PIN_MIGRATE=1 also moves the worker's private pages to its node
(migrate_pages) and prefers the node for the step thread's allocations.

The chip's node comes from the worker's open /dev/vfio/<group> file, so the
binding follows whatever device libtpu opened. VLLM_TPU_CPU_PROBE_S=N logs
each worker's thread placement and context switches every N seconds.
"""

import glob
import os
import threading
import time

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_INITIAL_CPUS: set[int] | None = None


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


def allowed_cpus() -> set[int] | None:
    """Initial CPU affinity mask of this process before worker pinning."""
    if _INITIAL_CPUS is not None:
        return _INITIAL_CPUS
    try:
        return os.sched_getaffinity(0)
    except (AttributeError, OSError):
        return None


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
    """(node, PCI address) of every TPU vfio group visible to this container."""
    devices = set()
    for path in glob.glob("/dev/vfio/[0-9]*"):
        devices.update(_group_devices(os.path.basename(path)))
    return sorted(devices)


def host_chip_devices(
    visible: list[tuple[int, str]] | None = None,
) -> list[tuple[int, str]]:
    """(node, PCI address) of every host PCI function matching visible TPUs."""
    base = all_chip_devices() if visible is None else visible
    ids: set[tuple[str, str]] = set()
    for _, bdf in base:
        try:
            ids.add(
                (
                    _read(f"/sys/bus/pci/devices/{bdf}/vendor"),
                    _read(f"/sys/bus/pci/devices/{bdf}/device"),
                )
            )
        except OSError:
            continue
    if not ids:
        return base
    devices = set(base)
    for dev in glob.glob("/sys/bus/pci/devices/*"):
        try:
            key = (_read(f"{dev}/vendor"), _read(f"{dev}/device"))
            if key in ids:
                devices.add((int(_read(f"{dev}/numa_node")), os.path.basename(dev)))
        except (OSError, ValueError):
            continue
    return sorted(devices)


def node_peers(node: int) -> list[tuple[int, str]]:
    """Peer TPU devices sharing ``node``'s usable CPU pool with this worker."""
    visible = all_chip_devices()
    vfio_peers = [d for d in visible if d[0] == node]
    try:
        raw_node_cpus = set(
            _cpulist(_read(f"/sys/devices/system/node/node{node}/cpulist"))
        )
    except (OSError, ValueError):
        return vfio_peers
    allowed = allowed_cpus()
    if allowed is not None and raw_node_cpus and not (raw_node_cpus <= allowed):
        return vfio_peers
    host_peers = [d for d in host_chip_devices(visible) if d[0] == node]
    return host_peers if len(host_peers) > len(vfio_peers) else vfio_peers


def physical_cores(node: int) -> list[tuple[int, ...]]:
    """The node's usable physical cores, each as its sorted SMT sibling tuple."""
    node_cpus = _cpulist(_read(f"/sys/devices/system/node/node{node}/cpulist"))
    allowed = allowed_cpus()
    if allowed is not None:
        node_cpus = [cpu for cpu in node_cpus if cpu in allowed]
    cores: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for cpu in node_cpus:
        sib = tuple(
            c
            for c in _cpulist(
                _read(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
            )
            if allowed is None or c in allowed
        )
        if sib and sib not in seen:
            seen.add(sib)
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


def step_cores(node: int, slots: int) -> list[tuple[int, ...]]:
    """The physical cores reserved for the step threads of a node's workers:
    cores 1..slots (core 0 takes most device interrupts and housekeeping)."""
    cores = physical_cores(node)
    first = 1 if len(cores) > slots + 1 else 0
    return cores[first : first + slots]


def _all_step_cpus() -> set[int]:
    """Every CPU (with SMT siblings) of every active node's step cores."""
    nodes = sorted({node for node, _ in all_chip_devices() if node >= 0})
    return {
        cpu
        for node in nodes
        for core in step_cores(node, max(1, len(node_peers(node))))
        for cpu in core
    }


def _pin_process(pid: int, cpus: set[int]) -> int:
    """Set the affinity of every thread of another process (same user)."""
    pinned = 0
    try:
        tasks = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return 0
    for task in tasks:
        try:
            os.sched_setaffinity(int(task), cpus)
            pinned += 1
        except OSError:
            continue
    return pinned


def _parent_pid(pid: int) -> int:
    try:
        return int(_read(f"/proc/{pid}/stat").rsplit(")", 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        return 0


def _vllm_ancestors() -> list[int]:
    """The engine core and API server: this worker's parent and grandparent,
    while they look like vLLM / Python processes."""
    out = []
    pid = os.getppid()
    for _ in range(2):
        if pid <= 1:
            break
        try:
            cmd = _read(f"/proc/{pid}/cmdline").replace("\x00", " ")
        except OSError:
            break
        if "python" not in cmd and "vllm" not in cmd:
            break
        out.append(pid)
        pid = _parent_pid(pid)
    return out


def _repin_new_threads(main_tid: int, others: set[int], period: float) -> None:
    """Threads the step thread starts after pinning inherit its single CPU;
    move them to ``others``. Only unseen thread ids are touched."""
    seen = set(os.listdir("/proc/self/task"))
    while True:
        time.sleep(period)
        try:
            now = set(os.listdir("/proc/self/task"))
        except OSError:
            continue
        for task in now - seen:
            tid = int(task)
            if tid == main_tid:
                continue
            try:
                if os.sched_getaffinity(tid) != others:
                    os.sched_setaffinity(tid, others)
            except OSError:
                continue
        seen = now


_MPOL_PREFERRED = 1
_SYS_SET_MEMPOLICY = 238  # x86_64
_SYS_MIGRATE_PAGES = 256  # x86_64


def migrate_to_node(node: int) -> str:
    """Move this process's private pages to ``node`` and prefer ``node`` for
    the calling thread's new allocations. Returns a short status string."""
    import ctypes
    import platform

    if platform.machine() != "x86_64":
        return "skipped (not x86_64)"
    nodes = sorted(
        int(os.path.basename(p)[4:])
        for p in glob.glob("/sys/devices/system/node/node[0-9]*")
    )
    others = [n for n in nodes if n != node]
    if not others:
        return "skipped (one node)"
    maxnode = max(nodes) + 2
    words = (maxnode + 63) // 64
    mask_t = ctypes.c_ulong * words

    def mask(ns):
        m = mask_t()
        for n in ns:
            m[n // 64] |= 1 << (n % 64)
        return m

    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    t = time.monotonic()
    left = libc.syscall(
        ctypes.c_long(_SYS_MIGRATE_PAGES),
        ctypes.c_long(0),
        ctypes.c_ulong(maxnode),
        mask(others),
        mask([node]),
    )
    err = ctypes.get_errno() if left < 0 else 0
    pref = libc.syscall(
        ctypes.c_long(_SYS_SET_MEMPOLICY),
        ctypes.c_int(_MPOL_PREFERRED),
        mask([node]),
        ctypes.c_ulong(maxnode),
    )
    return (
        f"migrate_pages={left} errno={err} set_mempolicy={pref} "
        f"took {time.monotonic() - t:.1f}s"
    )


def pin_all_threads(cpus: set[int]) -> int:
    """Set the affinity of every existing thread; new ones inherit it."""
    pinned = 0
    for task in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(task), cpus)
            pinned += 1
        except OSError:
            continue
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
    from vllm_torchtpu import envs

    global _INITIAL_CPUS
    mode = envs.VLLM_TPU_CPU_PIN
    if envs.VLLM_TPU_CPU_PROBE_S > 0:
        threading.Thread(
            target=_probe,
            args=(envs.VLLM_TPU_CPU_PROBE_S, rank),
            name="cpu-probe",
            daemon=True,
        ).start()
    if mode == "off":
        return
    if mode not in ("numa", "slice", "core"):
        raise ValueError(
            f"VLLM_TPU_CPU_PIN={mode!r}: expected off, numa, slice or core"
        )
    mine = my_chip_devices()
    nodes = {node for node, _ in mine}
    if len(nodes) != 1 or min(nodes) < 0:
        logger.warning("CPU pin skipped: rank %d chip devices %s", rank, mine)
        return
    if _INITIAL_CPUS is None:
        _INITIAL_CPUS = allowed_cpus()
    node = nodes.pop()
    # Slot = this worker's device among the node's peer TPU devices, so workers
    # sharing a node split its usable cores whatever their local ranks are.
    peers = node_peers(node)
    slots = max(1, len(peers))
    slot = peers.index(mine[0]) if mine[0] in peers else local_rank % slots
    if mode == "core":
        _apply_core(rank, local_rank, node, slot, slots)
    else:
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
    if envs.VLLM_TPU_CPU_PIN_MIGRATE:
        logger.info(
            "CPU pin: rank %d pages -> node %d: %s", rank, node, migrate_to_node(node)
        )


def _apply_core(rank: int, local_rank: int, node: int, slot: int, slots: int):
    mine = step_cores(node, slots)[slot]
    reserved = _all_step_cpus()
    node_cpus = {cpu for core in physical_cores(node) for cpu in core}
    others = node_cpus - reserved
    n = pin_all_threads(others)
    main_tid = threading.get_native_id()
    try:
        os.sched_setaffinity(0, {mine[0]})
    except OSError as e:
        logger.warning("CPU pin core: rank %d cannot use cpu %d (%s)", rank, mine[0], e)
        return
    threading.Thread(
        target=_repin_new_threads,
        args=(main_tid, others, 10.0),
        name="cpu-repin",
        daemon=True,
    ).start()
    ancestors = []
    if local_rank == 0:
        allowed = allowed_cpus()
        everything = (
            set(allowed) if allowed is not None else set(range(os.cpu_count() or 1))
        )
        try:
            host_all = {
                cpu
                for p in glob.glob("/sys/devices/system/node/node[0-9]*")
                for cpu in _cpulist(_read(f"{p}/cpulist"))
            }
            if host_all:
                everything = host_all if allowed is None else (host_all & allowed)
        except OSError:
            everything = everything
        rest = everything - reserved
        for pid in _vllm_ancestors():
            ancestors.append((pid, _pin_process(pid, rest)))
    logger.info(
        "CPU pin: rank %d local_rank %d node %d mode core slot %d/%d step thread "
        "%d -> cpu %d (core %s); %d other threads -> %d cpus; ancestors %s",
        rank,
        local_rank,
        node,
        slot,
        slots,
        main_tid,
        mine[0],
        mine,
        n,
        len(others),
        ancestors,
    )
