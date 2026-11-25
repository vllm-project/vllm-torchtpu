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

The chip's node comes from the worker's open /dev/vfio/<group> file, so the
binding follows whatever device libtpu opened. VLLM_TPU_CPU_PROBE_S=N logs
each worker's thread placement and context switches every N seconds.

``slice`` is strict (q38 SV-PAIR): it raises SlicePinError at warmup instead of
skipping or pinning wrongly when (R1) the worker does not hold exactly one TPU
device on exactly one node, (R2) that device is not among its node's devices,
(R3) the node has fewer than slots + RESERVE physical cores, (R4) a planned
core's SMT siblings leave the node, (R5) the plan is not inside the process's
allowed CPUs, or (R6) after pinning, any thread's read-back mask differs from
the plan. ``numa`` and ``off`` behave exactly as before.

VLLM_TPU_CPU_PIN_ENGINE (q38 SV-PAIR; slice only): ``reserved`` pins the
EngineCore process (after its executor is up, so the workers it spawned keep
their own affinity) and each API-server process (at its first chat or
completion request) to the cores the worker slices leave free on every node
with TPU devices; ``reserved:<node>`` to that node's free cores. ``off``
(default) changes nothing and imports nothing.
"""

import glob
import os
import threading
import time

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
    for task in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(task), cpus)
            pinned += 1
        except OSError:
            pass
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


# q38 SERVE r0pin mechanism check (VLLM_TPU_CPU_IRQ_DUMP_S): read-only, log-only.
IRQ_DUMP_FILES = ("/proc/interrupts", "/proc/softirqs")
IRQ_DUMP_CHUNK = 4000


def irq_dump_lines(
    rank: int,
    seq: int,
    paths: tuple[str, ...] = IRQ_DUMP_FILES,
    chunk: int = IRQ_DUMP_CHUNK,
    now: float | None = None,
) -> list[str]:
    """'CPU IRQ DUMP' log lines for one snapshot of each file: the raw bytes,
    gzip-compressed (mtime 0) and base64-encoded, split into parts of at most
    ``chunk`` characters, each part tagged with seq / time / source / part i/n /
    raw byte count / raw md5. Reads the files only; an unreadable file gives one
    UNREADABLE line."""
    import base64
    import gzip
    import hashlib

    t = time.time() if now is None else now
    out = []
    for path in paths:
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except OSError as exc:
            out.append(f"CPU IRQ DUMP rank={rank} seq={seq} t={t:.3f} src={path} UNREADABLE {exc}")
            continue
        b64 = base64.b64encode(gzip.compress(raw, 6, mtime=0)).decode()
        parts = [b64[i : i + chunk] for i in range(0, len(b64), chunk)] or [""]
        md5 = hashlib.md5(raw).hexdigest()
        for i, part in enumerate(parts):
            out.append(
                f"CPU IRQ DUMP rank={rank} seq={seq} t={t:.3f} src={path} part={i + 1}/{len(parts)} "
                f"bytes={len(raw)} md5={md5} b64={part}"
            )
    return out


def _irq_dump(interval: float, rank: int) -> None:
    seq = 0
    while True:
        for line in irq_dump_lines(rank, seq):
            logger.info("%s", line)
        seq += 1
        time.sleep(interval)


def apply(rank: int, local_rank: int) -> None:
    from vllm_torchtpu import envs

    mode = envs.VLLM_TPU_CPU_PIN
    if envs.VLLM_TPU_CPU_PROBE_S > 0:
        threading.Thread(
            target=_probe,
            args=(envs.VLLM_TPU_CPU_PROBE_S, rank),
            name="cpu-probe",
            daemon=True,
        ).start()
    if envs.VLLM_TPU_CPU_IRQ_DUMP_S > 0 and rank == 0:  # q38 SERVE r0pin mechanism check
        threading.Thread(
            target=_irq_dump,
            args=(envs.VLLM_TPU_CPU_IRQ_DUMP_S, rank),
            name="cpu-irq-dump",
            daemon=True,
        ).start()
    if mode == "off":
        return
    if mode not in ("numa", "slice"):
        raise ValueError(f"VLLM_TPU_CPU_PIN={mode!r}: expected off, numa or slice")
    if mode == "slice":
        _apply_slice(rank, local_rank, envs.VLLM_TPU_CPU_PIN_RESERVE)
        return
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
        "CPU pin: rank %d local_rank %d node %d mode %s slot %d/%d cpus %s (%d threads)",
        rank,
        local_rank,
        node,
        mode,
        slot,
        slots,
        sorted(cpus),
        n,
    )


# ---------------------------------------------------------------------------
# q38 SV-PAIR: strict slice (R1-R6) and the EngineCore / API-server option.
# ---------------------------------------------------------------------------


class SlicePinError(RuntimeError):
    """VLLM_TPU_CPU_PIN=slice (or VLLM_TPU_CPU_PIN_ENGINE) refused to pin."""


def node_cpus(node: int) -> set[int]:
    return set(_cpulist(_read(f"/sys/devices/system/node/node{node}/cpulist")))


def skip_cpus() -> set[int]:
    """CPUs named by VLLM_TPU_CPU_PIN_SKIP_CPUS (q38 SERVE r0pin; empty when unset)."""
    from vllm_torchtpu import envs

    text = envs.VLLM_TPU_CPU_PIN_SKIP_CPUS.strip()
    return set(_cpulist(text)) if text else set()


def slice_cores(
    node: int, slots: int, reserve: int, skip: set[int] | None = None
) -> tuple[list[tuple[int, ...]], int]:
    """The node's physical cores and the cores per slot, checking R3 and R4.

    q38 SERVE r0pin: a physical core holding any CPU in ``skip`` (default:
    VLLM_TPU_CPU_PIN_SKIP_CPUS) is removed AFTER the cores per slot are fixed, so
    every slot keeps its size; the node's slices shift past the skipped core and
    the remainder (the engine reserve) shrinks by it. No slice or reserve holds a
    skipped core. Empty ``skip`` (the default) changes nothing.
    """
    cores = physical_cores(node)
    inside = node_cpus(node)
    for core in cores:
        if not set(core) <= inside:  # R4
            raise SlicePinError(
                f"R4: node {node} core {core} has SMT siblings outside the node's cpulist"
            )
    if slots < 1 or len(cores) - reserve < slots:  # R3
        raise SlicePinError(
            f"R3: node {node} has {len(cores)} physical cores for {slots} slots + "
            f"VLLM_TPU_CPU_PIN_RESERVE={reserve}"
        )
    per = (len(cores) - reserve) // slots
    if per < 1:  # R3
        raise SlicePinError(f"R3: node {node} gives {per} cores per slot")
    skip = skip_cpus() if skip is None else skip
    if skip:
        cores = [core for core in cores if not set(core) & skip]
        if len(cores) < slots * per:  # R3 after the skip
            raise SlicePinError(
                f"R3: node {node} keeps {len(cores)} physical cores after "
                f"VLLM_TPU_CPU_PIN_SKIP_CPUS for {slots} slots of {per}"
            )
    return cores, per


def slice_plan(
    node: int, slot: int, slots: int, reserve: int, skip: set[int] | None = None
) -> set[int]:
    """The ``slot``-th worker's CPUs under the checks; equal to plan("slice", ...)
    when no CPU is skipped."""
    cores, per = slice_cores(node, slots, reserve, skip)
    mine = cores[slot * per : (slot + 1) * per]
    return {cpu for core in mine for cpu in core}


def reserved_cpus(
    node: int, slots: int, reserve: int, skip: set[int] | None = None
) -> set[int]:
    """The node's CPUs that no worker slice takes (the reserve and the remainder),
    minus any skipped core."""
    cores, per = slice_cores(node, slots, reserve, skip)
    return {cpu for core in cores[slots * per :] for cpu in core}


def pin_exact(cpus: set[int], passes: int = 3) -> int:
    """Pin every thread to ``cpus`` and read every mask back (R6).

    A thread that exits while we walk is skipped. Threads created meanwhile by a
    thread not yet pinned are caught by the next pass. Any refused
    sched_setaffinity, or any read-back mask other than ``cpus`` after the last
    pass (for example a cgroup cpuset the kernel intersected with), raises.
    Returns the number of threads read back."""
    want = set(cpus)
    for _ in range(passes):
        for task in os.listdir("/proc/self/task"):
            tid = int(task)
            try:
                if os.sched_getaffinity(tid) != want:
                    os.sched_setaffinity(tid, want)
            except ProcessLookupError:
                continue
            except OSError as exc:
                raise SlicePinError(f"R6: sched_setaffinity({tid}) failed: {exc}") from exc
        bad, seen = [], 0
        for task in os.listdir("/proc/self/task"):
            try:
                got = os.sched_getaffinity(int(task))
            except ProcessLookupError:
                continue
            seen += 1
            if got != want:
                bad.append((int(task), sorted(got)[:8]))
        if not bad:
            return seen
    raise SlicePinError(f"R6: {len(bad)} threads read back a mask other than the plan, e.g. {bad[:3]}")


def _apply_slice(rank: int, local_rank: int, reserve: int) -> None:
    mine = my_chip_devices()
    nodes = {node for node, _ in mine}
    if len(mine) != 1 or len(nodes) != 1 or min(nodes) < 0:  # R1
        raise SlicePinError(
            f"R1: rank {rank} must hold exactly one TPU device on one node, holds {mine}"
        )
    node = mine[0][0]
    peers = [d for d in all_chip_devices() if d[0] == node]
    if mine[0] not in peers:  # R2
        raise SlicePinError(f"R2: rank {rank} device {mine[0]} is not among node {node}'s {peers}")
    slots = len(peers)
    slot = peers.index(mine[0])
    _, per = slice_cores(node, slots, reserve)
    cpus = slice_plan(node, slot, slots, reserve)
    allowed = os.sched_getaffinity(0)
    if not cpus <= allowed:  # R5
        raise SlicePinError(
            f"R5: rank {rank} plan {sorted(cpus)} is not inside the allowed CPUs {sorted(allowed)}"
        )
    n = pin_exact(cpus)
    skipped = skip_cpus()
    if skipped:
        logger.info(
            "CPU pin skip: rank %d node %d VLLM_TPU_CPU_PIN_SKIP_CPUS=%s; plan holds none of them: %s",
            rank,
            node,
            sorted(skipped),
            not (cpus & skipped),
        )
    logger.info(
        "CPU pin: rank %d local_rank %d node %d mode slice slot %d/%d cores %d cpus %s "
        "(%d threads read back)",
        rank,
        local_rank,
        node,
        slot,
        slots,
        per,
        sorted(cpus),
        n,
    )


def engine_target() -> set[int] | None:
    """The CPUs VLLM_TPU_CPU_PIN_ENGINE asks for, or None when it is off."""
    from vllm_torchtpu import envs

    value = (envs.VLLM_TPU_CPU_PIN_ENGINE or "off").strip()
    if value in ("", "off"):
        return None
    if envs.VLLM_TPU_CPU_PIN != "slice":
        raise SlicePinError(
            f"VLLM_TPU_CPU_PIN_ENGINE={value!r} needs VLLM_TPU_CPU_PIN=slice "
            f"(got {envs.VLLM_TPU_CPU_PIN!r})"
        )
    if value == "reserved":
        want = None
    elif value.startswith("reserved:") and value[len("reserved:") :].isdigit():
        want = int(value[len("reserved:") :])
    else:
        raise SlicePinError(
            f"VLLM_TPU_CPU_PIN_ENGINE={value!r}: expected off, reserved or reserved:<node>"
        )
    per_node: dict[int, int] = {}
    for node, _ in all_chip_devices():
        if node >= 0:
            per_node[node] = per_node.get(node, 0) + 1
    if want is not None and want not in per_node:
        raise SlicePinError(f"VLLM_TPU_CPU_PIN_ENGINE: node {want} has no TPU devices ({per_node})")
    targets = sorted(per_node) if want is None else [want]
    if not targets:
        raise SlicePinError("VLLM_TPU_CPU_PIN_ENGINE: no TPU devices on this host")
    reserve = envs.VLLM_TPU_CPU_PIN_RESERVE
    cpus: set[int] = set()
    for node in targets:
        free = reserved_cpus(node, per_node[node], reserve)
        if not free:
            raise SlicePinError(f"VLLM_TPU_CPU_PIN_ENGINE: node {node} has no reserved cores")
        if not free <= node_cpus(node):
            raise SlicePinError(f"VLLM_TPU_CPU_PIN_ENGINE: reserved {sorted(free)} not on node {node}")
        cpus |= free
    return cpus


_ENGINE_PINNED = threading.Lock()
_ENGINE_DONE: dict[str, int] = {}


def pin_engine_process(role: str, cpus: set[int]) -> None:
    """Pin this whole process (every thread) to ``cpus`` once per role."""
    with _ENGINE_PINNED:
        if role in _ENGINE_DONE:
            return
        allowed = os.sched_getaffinity(0)
        if not cpus <= allowed:  # R5 for the process
            raise SlicePinError(f"R5: {role} target {sorted(cpus)} is not inside the allowed CPUs")
        n = pin_exact(cpus)
        _ENGINE_DONE[role] = os.getpid()
    logger.info("CPU pin engine: role=%s pid=%d cpus %s (%d threads read back)", role, os.getpid(), sorted(cpus), n)


def patch_engine_core_cpu_pin() -> None:
    """engine_core stage: pin the EngineCore process once its executor (and so its workers) is up.

    Same install shape as q38 SV-KNOB's patch_engine_core_for_host_knobs (core/host_knobs.py, credited): the wrapper
    runs after EngineCore.__init__, whose executor has spawned and warmed the workers (which have already pinned
    themselves), so they never inherit this mask. Off by default: nothing is imported or patched."""
    target = engine_target()
    if target is None:
        return
    if not target <= os.sched_getaffinity(0):  # R5, checked at startup
        raise SlicePinError(f"R5: engine_core target {sorted(target)} is not inside the allowed CPUs")
    from vllm.v1.engine.core import EngineCore

    if getattr(EngineCore, "_q38_cpu_pin_engine", False):
        return
    orig_init = EngineCore.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        pin_engine_process("engine_core", target)

    EngineCore.__init__ = __init__
    EngineCore._q38_cpu_pin_engine = True
    logger.info("VLLM_TPU_CPU_PIN_ENGINE: patched EngineCore.__init__ (target %s)", sorted(target))


def patch_api_server_cpu_pin() -> None:
    """platform_activation / worker_init: pin each API-server process at its first chat or completion request.

    Same hook as q38 SV-TTFT's api_loop_lag (OpenAIServingChat.create_chat_completion, credited), plus
    OpenAIServingCompletion.create_completion. By the first request the EngineCore process already exists, so it
    never inherits this mask. Off by default: nothing is imported or patched."""
    target = engine_target()
    if target is None:
        return
    if not target <= os.sched_getaffinity(0):  # R5, checked at startup
        raise SlicePinError(f"R5: api_server target {sorted(target)} is not inside the allowed CPUs")
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion

    for cls, name in ((OpenAIServingChat, "create_chat_completion"), (OpenAIServingCompletion, "create_completion")):
        orig = getattr(cls, name)
        if getattr(orig, "_q38_cpu_pin_api", False):
            continue

        def make(orig=orig):
            async def wrapped(self, *args, **kwargs):
                if "api_server" not in _ENGINE_DONE:
                    pin_engine_process("api_server", target)
                return await orig(self, *args, **kwargs)

            wrapped._q38_cpu_pin_api = True  # type: ignore[attr-defined]
            wrapped.__wrapped__ = orig  # type: ignore[attr-defined]
            return wrapped

        setattr(cls, name, make())
    logger.info(
        "VLLM_TPU_CPU_PIN_ENGINE: patched OpenAIServingChat.create_chat_completion and "
        "OpenAIServingCompletion.create_completion (target %s)",
        sorted(target),
    )
