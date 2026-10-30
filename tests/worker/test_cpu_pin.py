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
"""TPU_WORKER_CPU_PIN: topology discovery, the per-worker core plan, and the
pinner's syscalls (all against a fake sysfs / procfs)."""

import os
import threading

from vllm_torchtpu.worker import cpu_pin


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _fake_host(root, nodes=2, cores_per_node=8):
    """tpu7x-like: node n owns cores [n*C, (n+1)*C) and their SMT siblings
    at +nodes*C (like 0-55,112-167 / 56-111,168-223)."""
    total = nodes * cores_per_node
    for n in range(nodes):
        lo, hi = n * cores_per_node, (n + 1) * cores_per_node - 1
        _write(
            f"{root}/devices/system/node/node{n}/cpulist",
            f"{lo}-{hi},{lo + total}-{hi + total}\n",
        )
    for c in range(total):
        sib = f"{c},{c + total}"
        _write(
            f"{root}/devices/system/cpu/cpu{c}/topology/thread_siblings_list",
            sib + "\n",
        )
        _write(
            f"{root}/devices/system/cpu/cpu{c + total}/topology/thread_siblings_list",
            sib + "\n",
        )
    return set(range(2 * total))


def test_parse_cpulist():
    assert cpu_pin.parse_cpulist("0-3,8,10-11\n") == [0, 1, 2, 3, 8, 10, 11]
    assert cpu_pin.parse_cpulist("5") == [5]
    assert cpu_pin.parse_cpulist("") == []


def test_tpu_numa_nodes_from_vfio_fds(tmp_path):
    sysr = str(tmp_path / "sys")
    fdd = tmp_path / "fd"
    fdd.mkdir()
    # group 3 -> node 0 device, group 7 -> node 1 device, group 9 unknown (-1)
    _write(f"{sysr}/kernel/iommu_groups/3/devices/0000:00:04.0/numa_node", "0\n")
    _write(f"{sysr}/kernel/iommu_groups/7/devices/0000:c0:02.1/numa_node", "1\n")
    _write(f"{sysr}/kernel/iommu_groups/9/devices/0000:aa:00.0/numa_node", "-1\n")
    os.symlink("/dev/vfio/7", fdd / "11")
    os.symlink("/dev/vfio/vfio", fdd / "10")  # the container fd, not a group
    os.symlink("/dev/null", fdd / "0")
    assert cpu_pin.tpu_numa_nodes(str(fdd), sysr) == {1}
    os.symlink("/dev/vfio/3", fdd / "12")
    assert cpu_pin.tpu_numa_nodes(str(fdd), sysr) == {0, 1}
    os.symlink("/dev/vfio/9", fdd / "13")
    assert cpu_pin.tpu_numa_nodes(str(fdd), sysr) == {0, 1}


def test_node_cores_whole_and_allowed(tmp_path):
    sysr = str(tmp_path)
    allowed = _fake_host(sysr, nodes=2, cores_per_node=8)
    cores = cpu_pin.node_physical_cores(1, allowed, sysr)
    assert cores == [(c, c + 16) for c in range(8, 16)]
    # a core whose sibling is not allowed is dropped whole
    cores = cpu_pin.node_physical_cores(1, allowed - {25}, sysr)
    assert (9, 25) not in cores and len(cores) == 7


def test_plan_reserved_cores_disjoint_and_others_exclude_all_reserved(tmp_path):
    sysr = str(tmp_path)
    allowed = _fake_host(sysr, nodes=2, cores_per_node=16)
    cores = cpu_pin.node_physical_cores(0, allowed, sysr)
    plans = [cpu_pin.plan(cores, r, 8, skip=2) for r in range(8)]
    serving = [p[0] for p in plans]
    # each worker gets a whole, distinct physical core
    assert all(len(s) == 2 for s in serving)
    assert len(set().union(*serving)) == 16
    # the other threads of every worker avoid every worker's reserved core,
    # and stay on the node
    node0 = {c for core in cores for c in core}
    for _s, o in plans:
        assert not (o & set().union(*serving))
        assert o <= node0
        assert len(o) == 2 * (16 - 8)  # the node minus the 8 reserved cores
    # skipped cores are left to the other threads
    assert set(cores[0]) <= plans[0][1] and set(cores[1]) <= plans[0][1]


def test_plan_refuses_when_node_too_small():
    cores = [(i, i + 100) for i in range(10)]
    assert cpu_pin.plan(cores, 0, 8, skip=2) is None  # nothing left for others
    assert cpu_pin.plan(cores, 7, 7, skip=2) is None  # rank out of range
    assert cpu_pin.plan(cores, 0, 7, skip=2) is not None


def test_pinner_off_does_nothing(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("off pinner must not touch affinity")

    monkeypatch.setattr(cpu_pin.os, "sched_setaffinity", boom)
    monkeypatch.setattr(cpu_pin, "tpu_numa_nodes", boom)
    p = cpu_pin.WorkerCpuPinner(False, 0, 8)
    for _ in range(3):
        p.maybe_pin()
    assert p.serving_tid is None


def _patched_pinner(monkeypatch, tmp_path, nodes=(1,), tids=(100, 101, 102)):
    sysr = str(tmp_path)
    allowed = _fake_host(sysr, nodes=2, cores_per_node=16)
    calls = []
    monkeypatch.setattr(cpu_pin, "tpu_numa_nodes", lambda: set(nodes))
    monkeypatch.setattr(cpu_pin, "vfio_group_count", lambda: 0)
    real_cores = cpu_pin.node_physical_cores
    monkeypatch.setattr(
        cpu_pin, "node_physical_cores", lambda n, a: real_cores(n, a, sysr)
    )

    def setaff(pid, cpus):
        if pid in dead:
            raise ProcessLookupError(pid)
        calls.append((pid, set(cpus)))

    dead = set()
    monkeypatch.setattr(cpu_pin.os, "sched_setaffinity", setaff)
    monkeypatch.setattr(
        cpu_pin.os, "sched_getaffinity", lambda pid: set(masks.get(pid, allowed))
    )
    masks = {}
    me = threading.get_native_id()
    task = list(tids) + [me]
    real_listdir = os.listdir
    monkeypatch.setattr(
        cpu_pin.os,
        "listdir",
        lambda p: (
            [str(t) for t in task] if p == "/proc/self/task" else real_listdir(p)
        ),
    )
    return calls, me, task, masks, dead


def test_pinner_pins_serving_thread_once_and_moves_others(monkeypatch, tmp_path):
    calls, _me, task, _masks, _dead = _patched_pinner(monkeypatch, tmp_path)
    p = cpu_pin.WorkerCpuPinner(True, 3, 8, skip=2, sweep_s=0)
    p.maybe_pin()
    serving = {16 + 2 + 3, 16 + 2 + 3 + 32}  # node 1 core index 5 and sibling
    assert p.serving_cpus == serving
    # others moved before the serving thread is pinned; serving thread last
    assert calls[-1] == (0, serving)
    moved = {pid for pid, _cpus in calls[:-1]}
    assert moved == {100, 101, 102}
    assert all(cpus == p.other_cpus for _pid, cpus in calls[:-1])
    assert not (p.other_cpus & serving)
    n = len(calls)
    for _ in range(5):
        p.maybe_pin()  # same thread: no further syscalls
    assert len(calls) == n
    # a thread created later is moved by the next sweep, seen ones are not
    task.append(103)
    p._sweep()
    assert calls[n:] == [(103, p.other_cpus)]


def test_pinner_failure_is_quiet_and_sticky(monkeypatch, tmp_path):
    calls, _me, _task, _masks, _dead = _patched_pinner(
        monkeypatch, tmp_path, nodes={0, 1}
    )
    p = cpu_pin.WorkerCpuPinner(True, 0, 8, sweep_s=0)
    p.maybe_pin()  # two NUMA nodes: refuse, never raise
    assert p.serving_tid is None and calls == []
    p.maybe_pin()
    assert calls == []


def test_reserved_block_covers_every_host_device(tmp_path, monkeypatch):
    """TPU_LOCAL_RANK_OFFSET: two servers on one host see LOCAL_WORLD_SIZE 2 and
    4; the reserved block follows the host's vfio group count instead, so one
    server's other threads never land on the other server's serving cores."""
    _calls, _me, _task, _masks, _dead = _patched_pinner(monkeypatch, tmp_path)
    monkeypatch.setattr(cpu_pin, "vfio_group_count", lambda: 4)
    a = cpu_pin.WorkerCpuPinner(True, 0, 2, skip=2, sweep_s=0)  # server A, ranks 0-1
    a.maybe_pin()
    b = cpu_pin.WorkerCpuPinner(
        True, 3, 4, skip=2, sweep_s=0
    )  # server B (offset 2), rank 3
    b._pin(999)
    assert not (a.other_cpus & b.serving_cpus) and not (b.other_cpus & a.serving_cpus)
