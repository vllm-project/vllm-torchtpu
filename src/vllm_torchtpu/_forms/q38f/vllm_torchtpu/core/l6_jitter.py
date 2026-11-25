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
"""q38 L6-HOST JITTER levers ``gcfreeze``, ``switch`` and ``mainpin`` (worker process; windowable).

Once the decode is host-bound, the step pays the slowest of the 8 ranks on every
step. On T85m c16 the last rank launches the forward 0.41 / 1.47 / 5.2 ms (p50 /
p90 / p99) after the median rank (M, L6 rank-max read of SKEW-READ's VLOG table),
and the late rank rotates, which is the signature of random host stalls (cyclic
GC, GIL hand-offs). IX's Qwen3.5 track-3 trims (PR11: ``sys.setswitchinterval``
200 µs and ``gc.freeze``, credited) cut that on its band-H serving.

* ``gcfreeze`` (``VLLM_TPU_L6_GC_FREEZE``): on a window where the lever turns on,
  ``gc.freeze()`` moves every tracked object to the permanent generation, so
  cyclic collections stop walking the long-lived heap; where it turns off,
  ``gc.unfreeze()`` puts them back. Unlike D1e's one-shot freeze this toggles,
  so it follows the L6 A/B windows. No ``gc.collect()`` is forced at a toggle
  (that would put a long pause into the on-window). Refused (lever off, one
  ERROR) when D1e's ``VLLM_TPU_HOST_GC=freeze`` is set, which owns the freeze.
* ``switch`` (``VLLM_TPU_L6_SWITCH``; interval ``VLLM_TPU_L6_SWITCH_US``, default
  200): ``sys.setswitchinterval`` while on, the interval found at the toggle while
  off (D1e's knob, if it set one, applies first in the same step).

* ``mainpin`` (``VLLM_TPU_L6_MAIN_PIN``): CPU isolation of the worker's main
  (Python) thread. The ``world`` highest CPUs of the process's allowed set are
  reserved for the ranks' main threads and rank r takes the (r mod world)-th
  highest, so the workers of one NUMA node pin to distinct CPUs and keep all of
  them free of their other threads (``VLLM_TPU_L6_MAIN_PIN_CPU`` overrides the CPU
  for a single-rank process only). Each worker runs about 2,000 threads on its NUMA node's CPUs
  (M, T85m), and the main thread is not pinned. On a window where the lever turns
  on, the main thread is pinned to its CPU and every other thread of the process
  is moved off every reserved CPU; where it turns off,
  every saved affinity is restored. The scan repeats at each on-window, so
  threads created meanwhile are covered at the next window. Linux only; any
  OSError leaves the lever off with one ERROR.

* ``syncgc`` (``VLLM_TPU_L6_SYNC_GC``; ``VLLM_TPU_L6_SYNC_GC_EVERY=N``, default 100;
  ``VLLM_TPU_L6_SYNC_GC_VALVE=V``, default 100000): H3-L3 SYNC-GC
  (reports/host_levers_h1.md T-5(b)). Cyclic GC runs at whatever allocation crosses the
  gen-0 threshold, so each rank pauses on its own step and the step pays the slowest
  rank. With the lever on, automatic collection is off (``gc.disable()``) and every rank
  runs ``gc.collect`` at the same worker step index instead: the first draft ``propose``
  (just before HOSTPHASES' ``dp_entry`` mark) of a step whose l6_ab step count j is a
  multiple of N (or the first after it, if a step had no propose). j is l6_ab's
  per-process step count (``step_seq``), the same on every TP rank, so the pauses
  line up. The generation follows CPython's own counters and thresholds (gen 1 when
  ``gc.get_count()[1]`` exceeds threshold 1, gen 2 likewise), so objects that survive
  a gen-0 pass are still reclaimed (CPython's long-lived-ratio gate for gen 2 is not
  applied). Safety valve: at each step start and each propose, a gen-0 count
  (``gc.get_count()[0]``: GC-tracked objects allocated minus freed since the last
  collection) above V collects at once, counted and logged (first 3, then every
  1,000th). On the first engagement after warm-up ``gc.freeze()`` runs once (then
  ``gc.disable()``); it is never undone, because ``gc.unfreeze()`` would also thaw the
  warm-up heap that the worker froze with vLLM's ``freeze_gc_heap()``
  (worker/tpu_worker.py), which every class keeps frozen. Where the lever turns off,
  ``gc.enable()`` restores automatic collection (with D1e's threshold, if set); on any
  error too, with one ERROR, and the lever stays off. Refused (one ERROR, lever off)
  next to D1e's ``VLLM_TPU_HOST_GC=freeze`` and next to ``gcfreeze``; composes with
  D1e's ``threshold`` mode (served); needs gc enabled when it engages. With the lever
  set, a ``gc.callbacks`` timer puts every collection's pause in the L6AB line
  (``gc.us`` / ``gc.n`` / ``gc.g2``) in every window, on or off. Counts:
  ``syncgc.collect`` (lockstep collections), ``syncgc.valve``, ``syncgc.us`` (µs inside
  those collections), ``syncgc.g1`` / ``g2`` (escalated ones), ``syncgc.on`` / ``off`` /
  ``freeze``.
* ``mainpin2`` (``VLLM_TPU_L6_MAIN_PIN2``): MAINPIN-2 (host_levers_h1.md), replacing
  ``mainpin``. The worker's NUMA node (the one node its allowed CPUs touch, else its
  chip's node from the open /dev/vfio group) has its physical cores read from sysfs
  (``thread_siblings_list``); the ``world`` highest cores are reserved with all their
  SMT siblings, and rank r's main thread takes the (r mod world)-th highest core's
  first CPU, leaving its sibling(s) idle. Once after warm-up (and after
  ``VLLM_TPU_CPU_PIN``), in every class, every other thread of the process is moved off
  every reserved CPU (a static exclusion, never repeated). Per l6_ab window only the
  main thread's mask toggles: its core's CPU while on, its own saved mask while off.
  At each toggle, and only for threads not seen before, newly created threads get the
  same exclusion (a thread born on the pinned CPU gets the process's allowed set back);
  known threads are not touched again. Any OSError restores every saved mask (main and
  excluded threads) with one ERROR and the lever stays off. Refused (one ERROR) next to
  ``mainpin`` and next to ``VLLM_TPU_CPU_PIN=slice``. Counts: ``mainpin2.pinned`` (steps
  run with the main thread pinned), ``mainpin2.toggle``, ``mainpin2.fail``,
  ``mainpin2.excl`` (threads moved by the static exclusion), ``mainpin2.newthr``. The
  rest of the host (``vllm serve`` with its engine core and API server, and the log
  ``ring_tee``) belongs on the complement of the reserved CPUs, which
  ``python3 -m vllm_torchtpu.core.l6_jitter complement <world>`` prints for a launcher
  ``taskset -c``. ``VLLM_TPU_L6_NUMA_BIND`` (worker/numa_bind.py) binds each worker to
  its chip's node before the runtime starts.

None of them changes what the device computes. Applied from the worker's execute_model
after ``l6_ab.begin_step()``; toggles are counted in the L6AB window line. SYNC-GC and
MAINPIN-2 engage only after ``mark_warm()`` (the worker calls it when warm-up and
``VLLM_TPU_CPU_PIN`` are done).
"""

from __future__ import annotations

import functools
import gc
import glob
import os
import sys
import threading
import time

from vllm_torchtpu.logger import init_logger

from . import l6_ab

logger = init_logger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


SWITCH_US = _env_int("VLLM_TPU_L6_SWITCH_US", 200)
PIN_CPU = _env_int("VLLM_TPU_L6_MAIN_PIN_CPU", -1)
D1E_FREEZE = os.environ.get("VLLM_TPU_HOST_GC", "").strip().lower() == "freeze"
_state: dict = {"frozen": False, "switch_on": False, "switch_prev": None, "warned": False,
                "pinned": False, "pin_saved": {}, "pin_broken": False}
SYNC_EVERY = _env_int("VLLM_TPU_L6_SYNC_GC_EVERY", 100)
SYNC_VALVE = _env_int("VLLM_TPU_L6_SYNC_GC_VALVE", 100000)
CPU_PIN_MODE = os.environ.get("VLLM_TPU_CPU_PIN", "off").strip().lower()
SYS = "/sys/devices/system"  # sysfs root of the CPU / node topology (tests point it at a fake tree)
_warm = {"on": False}
_sg: dict = {}
_p2: dict = {}


def _sg_reset() -> None:
    _sg.clear()
    _sg.update(checked=False, refused=False, broken=False, armed=False, frozen=False, due=0, timer=False,
               t0=0, valve_n=0, wrapped=False, fallback=False)


def _p2_reset() -> None:
    _p2.clear()
    _p2.update(checked=False, broken=False, excluded=False, pinned=False, main=None, main_saved=None, cpu=None,
               idle=frozenset(), reserved=frozenset(), node=None, node_cpus=set(), saved={}, known=set())


_sg_reset()
_p2_reset()


def mark_warm() -> None:
    """Warm-up (and VLLM_TPU_CPU_PIN) is done: SYNC-GC and MAINPIN-2 may engage from the next step."""
    _warm["on"] = True


def _configured(lever: str) -> bool:
    return l6_ab._cfg.get(lever, (False, None)) != (False, None)


def _tids() -> list[int]:
    return [int(t) for t in os.listdir("/proc/self/task")]


def _pick(allowed: set[int], rank: int, world: int) -> tuple[int, set[int]]:
    """(this rank's main-thread CPU, the CPUs reserved for every rank's main thread).
    The `world` highest CPUs of the allowed set are reserved; rank r takes the
    (r mod world)-th highest. Workers on one NUMA node share the node's allowed set
    (VLLM_TPU_CPU_PIN=numa), so their ranks pick distinct CPUs and every worker keeps
    all of them free of its other threads. VLLM_TPU_L6_MAIN_PIN_CPU applies to a
    single-rank process only."""
    cpus = sorted(allowed)
    if PIN_CPU >= 0 and world <= 1:
        if PIN_CPU not in allowed or len(cpus) < 2:
            raise OSError(f"cannot isolate CPU {PIN_CPU} inside {cpus}")
        return PIN_CPU, {PIN_CPU}
    k = max(1, world)
    if len(cpus) < k + 1:
        raise OSError(f"cannot reserve {k} main-thread CPUs inside {cpus}")
    return cpus[-1 - (rank % k)], set(cpus[-k:])


def _pin_on() -> None:
    main = threading.get_native_id()
    saved = _state["pin_saved"]
    # the process's allowed set: the main thread's mask saved at the first pin (once
    # pinned, its live mask is the single pinned CPU)
    allowed = saved.get(main) or os.sched_getaffinity(main)
    rank, world = l6_ab._rank_world()  # the torch.distributed rank / world (also in uniform mode)
    cpu, reserved = _pick(allowed, rank, world)
    rest = allowed - reserved
    moved = 0
    for tid in _tids():
        try:
            if tid not in saved:
                cur = os.sched_getaffinity(tid)
                # a thread born on the pinned CPU (created by the main thread while pinned)
                # goes back to the process's allowed set, not to that CPU, when unpinned
                saved[tid] = allowed if cur <= reserved and tid != main else cur
            if tid == main:
                os.sched_setaffinity(tid, {cpu})
            else:
                os.sched_setaffinity(tid, (saved[tid] - reserved) or rest)
                moved += 1
        except ProcessLookupError:
            saved.pop(tid, None)  # the thread ended during the scan
    l6_ab.note("mainpin", "moved", moved)


def _pin_off() -> None:
    saved, _state["pin_saved"] = _state["pin_saved"], {}
    for tid, mask in saved.items():
        try:
            os.sched_setaffinity(tid, mask)
        except ProcessLookupError:
            pass


# ---------------------------------------------------------------- SYNC-GC

_gc_collect = gc.collect  # indirection for the error-path tests


def _gc_timer(phase: str, info: dict) -> None:
    """gc.callbacks timer: every collection's pause into the current L6AB window's counts."""
    if phase == "start":
        _sg["t0"] = time.perf_counter_ns()
    elif _sg["t0"]:
        us = (time.perf_counter_ns() - _sg["t0"]) // 1000
        _sg["t0"] = 0
        l6_ab.note("gc", "us", us)
        l6_ab.note("gc", "n")
        if info.get("generation") == 2:
            l6_ab.note("gc", "g2")


def _install_sync_point(cls=None) -> bool:
    """Run sync_point() at the start of every draft propose (just before HOSTPHASES' dp_entry mark;
    only cheap list setup lies between). A class-level wrapper, installed once when syncgc is set."""
    if cls is None:
        from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer as cls
    orig = cls.propose
    if getattr(orig, "_q38_syncgc", False):
        return True

    @functools.wraps(orig)
    def propose(self, *args, **kwargs):
        if _sg["armed"]:
            sync_point()
        return orig(self, *args, **kwargs)

    propose._q38_syncgc = True  # type: ignore[attr-defined]
    cls.propose = propose
    return True


def _sg_refusal() -> str | None:
    if D1E_FREEZE:
        return "D1e's VLLM_TPU_HOST_GC=freeze owns the freeze"
    if _configured("gcfreeze"):
        return "the gcfreeze lever (VLLM_TPU_L6_GC_FREEZE) toggles gc.freeze / gc.unfreeze"
    if SYNC_EVERY < 1 or SYNC_VALVE < 1:
        return f"VLLM_TPU_L6_SYNC_GC_EVERY={SYNC_EVERY} and VLLM_TPU_L6_SYNC_GC_VALVE={SYNC_VALVE} must be >= 1"
    return None


def _sg_collect(kind: str) -> None:
    c1, c2 = gc.get_count()[1:]
    _, t1, t2 = gc.get_threshold()
    gen = 2 if (t2 and c2 > t2) else (1 if (t1 and c1 > t1) else 0)
    t0 = time.perf_counter_ns()
    _gc_collect(gen)
    l6_ab.note("syncgc", kind)
    l6_ab.note("syncgc", "us", (time.perf_counter_ns() - t0) // 1000)
    if gen:
        l6_ab.note("syncgc", f"g{gen}")


def _sg_valve() -> None:
    n0 = gc.get_count()[0]
    if n0 > SYNC_VALVE:
        _sg_collect("valve")
        _sg["valve_n"] += 1
        if _sg["valve_n"] <= 3 or _sg["valve_n"] % 1000 == 0:
            logger.warning("q38 L6 syncgc valve #%d: gen-0 count %d > %d at worker step %d; collected now",
                           _sg["valve_n"], n0, SYNC_VALVE, l6_ab.step_seq())


def _sg_engage() -> None:
    if not gc.isenabled():
        raise RuntimeError("gc is already disabled (not by syncgc), so syncgc would not own the re-enable")
    if not _sg["frozen"]:
        gc.freeze()  # once, after warm-up: never undone (gc.unfreeze would thaw the warm-up freeze too)
        _sg["frozen"] = True
        l6_ab.note("syncgc", "freeze")
    gc.disable()
    j = l6_ab.step_seq()
    _sg["due"] = -(-j // SYNC_EVERY) * SYNC_EVERY  # the first multiple of N at or after this step
    _sg["armed"] = True
    l6_ab.note("syncgc", "on")


def _sg_release() -> None:
    gc.enable()
    _sg["armed"] = False
    l6_ab.note("syncgc", "off")


def _sg_fail(exc: BaseException) -> None:
    _sg["broken"] = True
    undo = _sg["armed"]
    if undo:
        gc.enable()
        _sg["armed"] = False
    logger.error("q38 L6 syncgc off for the rest of the run%s: %s",
                 " (automatic gc re-enabled)" if undo else "", exc)


def sync_point() -> None:
    """The lockstep collection point (draft propose entry, just before dp_entry)."""
    if not _sg["armed"]:
        return
    try:
        j = l6_ab.step_seq()
        if j >= _sg["due"]:
            _sg["due"] = (j // SYNC_EVERY + 1) * SYNC_EVERY
            _sg_collect("collect")
        else:
            _sg_valve()
    except Exception as exc:  # noqa: BLE001 - an optimisation must never stop the step
        _sg_fail(exc)


def _syncgc_step() -> None:
    if not _configured("syncgc") or _sg["broken"]:
        return
    if not _sg["timer"]:
        gc.callbacks.append(_gc_timer)  # regardless of VLLM_TPU_HOST_AB_PERIOD_MS
        _sg["timer"] = True
    if not _sg["checked"]:
        _sg["checked"] = True
        why = _sg_refusal()
        if why:
            _sg["broken"] = _sg["refused"] = True
            logger.error("q38 L6 syncgc refused: %s; lever off.", why)
            return
        try:
            _sg["wrapped"] = _install_sync_point()
        except Exception as exc:  # noqa: BLE001 - no eagle3 drafter: collect at the step start instead
            _sg["fallback"] = True
            logger.info("q38 L6 syncgc: no propose hook (%s); the lockstep collection runs at the step start", exc)
    want = l6_ab.on("syncgc") and _warm["on"]
    try:
        if want != _sg["armed"]:
            if want:
                _sg_engage()
            else:
                _sg_release()
        if _sg["armed"]:
            if _sg["fallback"]:
                sync_point()
            else:
                _sg_valve()
    except Exception as exc:  # noqa: BLE001
        _sg_fail(exc)


# ---------------------------------------------------------------- MAINPIN-2


def _read(path: str) -> str:
    with open(path) as f:
        return f.read().strip()


def _cpulist(text: str) -> list[int]:
    from vllm_torchtpu.worker.numa_bind import parse_cpulist

    return parse_cpulist(text)


def fmt_cpulist(cpus) -> str:
    """[0, 1, 2, 5, 7, 8] -> "0-2,5,7-8"."""
    out, run = [], []
    for c in sorted(cpus):
        if run and c == run[-1] + 1:
            run.append(c)
            continue
        if run:
            out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
        run = [c]
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ",".join(out)


def _nodes() -> list[int]:
    return sorted(int(os.path.basename(p)[4:]) for p in glob.glob(os.path.join(SYS, "node", "node[0-9]*")))


def node_cpus(node: int) -> set[int]:
    return set(_cpulist(_read(os.path.join(SYS, "node", f"node{node}", "cpulist"))))


def node_cores(node: int) -> list[tuple[int, ...]]:
    """The node's physical cores, each as its sorted SMT sibling tuple, in CPU order."""
    cores = set()
    for c in node_cpus(node):
        cores.add(tuple(sorted(_cpulist(_read(os.path.join(SYS, "cpu", f"cpu{c}", "topology",
                                                           "thread_siblings_list"))))))
    return sorted(cores)


def plan2(node: int, rank: int, world: int) -> tuple[int, frozenset[int], frozenset[int]]:
    """(rank's main-thread CPU, its idle SMT siblings, every reserved CPU) on `node`: the `world` highest
    physical cores are reserved with all their siblings; rank r takes the (r mod world)-th highest core."""
    cores = node_cores(node)
    k = max(1, world)
    if len(cores) < k + 1:
        raise OSError(f"node {node} has {len(cores)} physical cores; {k} reserved main-thread cores need {k + 1}")
    mine = cores[-1 - (rank % k)]
    reserved = frozenset(c for core in cores[-k:] for c in core)
    return mine[0], frozenset(mine[1:]), reserved


def complement(world: int) -> list[int]:
    """Every online CPU outside every node's reserved main-thread cores: the launcher's taskset for
    `vllm serve` (engine core, API server) and the log ring_tee."""
    online = set(_cpulist(_read(os.path.join(SYS, "cpu", "online"))))
    reserved: set[int] = set()
    for n in _nodes():
        if node_cpus(n):
            reserved |= plan2(n, 0, world)[2]
    return sorted(online - reserved)


def _chip_node() -> int | None:
    from vllm_torchtpu.worker import cpu_pinning

    nodes = {n for n, _ in cpu_pinning.my_chip_devices()}
    return nodes.pop() if len(nodes) == 1 and min(nodes) >= 0 else None


def _worker_node(allowed: set[int]) -> int:
    hit = [n for n in _nodes() if node_cpus(n) & allowed]
    if len(hit) == 1:
        return hit[0]
    node = _chip_node()
    if node is None:
        raise OSError(f"cannot tell this worker's NUMA node (allowed CPUs touch nodes {hit}, chip node unknown); "
                      "set VLLM_TPU_CPU_PIN=numa or VLLM_TPU_L6_NUMA_BIND=1")
    return node


def _main_tid() -> int:
    return threading.get_native_id()


def _getaff(tid: int) -> set[int]:
    return set(os.sched_getaffinity(tid))


def _setaff(tid: int, cpus) -> None:
    os.sched_setaffinity(tid, cpus)


def _p2_exclusion(base: set[int]) -> set[int]:
    return (base - _p2["reserved"]) or (_p2["node_cpus"] - _p2["reserved"])


def _p2_setup() -> None:
    """Plan, then the static exclusion of every other thread (once)."""
    main = _main_tid()
    allowed = _getaff(main)
    rank, world = l6_ab._rank_world()
    node = _worker_node(allowed)
    cpu, idle, reserved = plan2(node, rank, world)
    _p2.update(main=main, main_saved=allowed, cpu=cpu, idle=idle, reserved=reserved, node=node,
               node_cpus=node_cpus(node))
    moved = 0
    for tid in _tids():
        if tid == main:
            continue
        try:
            m = _getaff(tid)
            _p2["saved"][tid] = m
            _setaff(tid, _p2_exclusion(m))
            moved += 1
        except ProcessLookupError:
            _p2["saved"].pop(tid, None)  # the thread ended during the scan
    _p2["known"] = set(_p2["saved"]) | {main}
    _p2["excluded"] = True
    l6_ab.note("mainpin2", "excl", moved)
    logger.info("q38 L6 mainpin2: rank %d of %d, node %d: main thread CPU %d, idle sibling(s) %s, reserved %s; "
                "%d other threads excluded once", rank, world, node, cpu, sorted(idle), fmt_cpulist(reserved), moved)


def _p2_sweep() -> None:
    """Exclude threads created since the last look (only those); forget threads that ended."""
    tids = set(_tids())
    main, pinned = _p2["main"], {_p2["cpu"]}
    new = tids - _p2["known"]
    n = 0
    for tid in new:
        try:
            m = _getaff(tid)
            base = set(_p2["main_saved"]) if m <= pinned else m  # born on the pinned CPU: the allowed set
            _p2["saved"][tid] = base
            _setaff(tid, _p2_exclusion(base))
            n += 1
        except ProcessLookupError:
            pass
    gone = _p2["known"] - tids - {main}
    for tid in gone:
        _p2["saved"].pop(tid, None)
    _p2["known"] = (_p2["known"] | new) - gone
    if n:
        l6_ab.note("mainpin2", "newthr", n)


def _p2_set_main(on: bool) -> None:
    main = _p2["main"]
    if _main_tid() != main:
        raise OSError(f"the step thread changed ({_main_tid()} != {main})")
    if on:
        _setaff(main, {_p2["cpu"]})
        got = _getaff(main)
        if got != {_p2["cpu"]}:
            raise OSError(f"main thread mask {sorted(got)} after pinning it to CPU {_p2['cpu']}")
    else:
        _setaff(main, _p2["main_saved"])
    _p2["pinned"] = on
    l6_ab.note("mainpin2", "toggle")


def _p2_restore_all() -> None:
    if _p2["main"] is not None and _p2["main_saved"] is not None:
        try:
            _setaff(_p2["main"], _p2["main_saved"])
        except OSError:
            pass
    for tid, mask in list(_p2["saved"].items()):
        try:
            _setaff(tid, mask)
        except OSError:  # includes ProcessLookupError (the thread ended)
            pass
    _p2.update(excluded=False, pinned=False, saved={}, known=set())


def _p2_fail(exc: BaseException) -> None:
    _p2["broken"] = True
    l6_ab.note("mainpin2", "fail")
    logger.error("q38 L6 mainpin2 off for the rest of the run (every saved mask restored): %s", exc)
    _p2_restore_all()


def _mainpin2_step() -> None:
    if not _configured("mainpin2") or _p2["broken"]:
        return
    if not _p2["checked"]:
        _p2["checked"] = True
        why = None
        if _configured("mainpin"):
            why = "the old mainpin lever (VLLM_TPU_L6_MAIN_PIN) is set"
        elif CPU_PIN_MODE == "slice":
            why = "VLLM_TPU_CPU_PIN=slice gives each worker its own cores"
        if why:
            _p2["broken"] = True
            logger.error("q38 L6 mainpin2 refused: %s; lever off.", why)
            return
    if not _warm["on"]:
        return
    want = l6_ab.on("mainpin2")
    try:
        if not _p2["excluded"]:
            _p2_setup()
        if want != _p2["pinned"]:
            _p2_sweep()
            _p2_set_main(want)
        if _p2["pinned"]:
            l6_ab.note("mainpin2", "pinned")
    except OSError as exc:
        _p2_fail(exc)


def on_step() -> None:
    if not l6_ab.ACTIVE:
        return
    want = l6_ab.on("gcfreeze")
    if want and D1E_FREEZE:
        if not _state["warned"]:
            _state["warned"] = True
            logger.error("q38 L6 gcfreeze refused: D1e's VLLM_TPU_HOST_GC=freeze owns the freeze; lever off.")
        want = False
    if want != _state["frozen"]:
        if want:
            gc.freeze()
            l6_ab.note("gcfreeze", "freeze")
        else:
            gc.unfreeze()
            l6_ab.note("gcfreeze", "unfreeze")
        _state["frozen"] = want
    sw = l6_ab.on("switch")
    if sw != _state["switch_on"]:
        if sw:
            _state["switch_prev"] = sys.getswitchinterval()
            sys.setswitchinterval(SWITCH_US / 1e6)
            l6_ab.note("switch", "set")
        else:
            sys.setswitchinterval(_state["switch_prev"])
            l6_ab.note("switch", "restore")
        _state["switch_on"] = sw
    pin = l6_ab.on("mainpin") and not _state["pin_broken"]
    if pin != _state["pinned"] or (pin and l6_ab._state.get("steps") == 1):
        try:
            if pin:
                _pin_on()
                l6_ab.note("mainpin", "pin")
            else:
                _pin_off()
                l6_ab.note("mainpin", "unpin")
            _state["pinned"] = pin
        except OSError as exc:
            _state["pin_broken"] = True
            logger.error("q38 L6 mainpin off for the rest of the run: %s", exc)
            try:
                _pin_off()
            except OSError:
                pass
            _state["pinned"] = False
    _syncgc_step()
    _mainpin2_step()


def restore() -> None:
    """Undo every toggle (tests)."""
    if _state["frozen"]:
        gc.unfreeze()
    if _state["switch_on"] and _state["switch_prev"] is not None:
        sys.setswitchinterval(_state["switch_prev"])
    if _state["pinned"] or _state["pin_saved"]:
        _pin_off()
    _state.update(frozen=False, switch_on=False, switch_prev=None, warned=False,
                  pinned=False, pin_saved={}, pin_broken=False)
    if _sg["armed"]:
        gc.enable()
    if _gc_timer in gc.callbacks:
        gc.callbacks.remove(_gc_timer)
    _sg_reset()
    if _p2["excluded"] or _p2["pinned"]:
        _p2_restore_all()
    _p2_reset()
    _warm["on"] = False


def _main(argv: list[str]) -> int:
    """python3 -m vllm_torchtpu.core.l6_jitter complement <world>: the MAINPIN-2 complement cpulist."""
    if len(argv) != 2 or argv[0] != "complement" or not argv[1].isdigit():
        print("usage: python3 -m vllm_torchtpu.core.l6_jitter complement <world>", file=sys.stderr)
        return 2
    print(fmt_cpulist(complement(int(argv[1]))))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
