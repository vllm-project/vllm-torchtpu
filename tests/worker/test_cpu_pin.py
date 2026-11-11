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
"""Unit tests for NUMA-local TPU worker CPU affinity planning."""

from __future__ import annotations

import os
from pathlib import Path

from vllm_torchtpu.worker import cpu_pin


def test_parse_cpulist():
    assert cpu_pin.parse_cpulist("0-2, 6, 8-9\n") == [0, 1, 2, 6, 8, 9]
    assert cpu_pin.parse_cpulist("") == []


def test_node_physical_cores_drops_partial_smt_cores(tmp_path: Path):
    node_dir = tmp_path / "devices/system/node/node0"
    node_dir.mkdir(parents=True)
    (node_dir / "cpulist").write_text("0-3,8-11\n")
    siblings = {
        0: "0,8",
        1: "1,9",
        2: "2,10",
        3: "3,11",
        8: "0,8",
        9: "1,9",
        10: "2,10",
        11: "3,11",
    }
    for cpu, text in siblings.items():
        topo = tmp_path / f"devices/system/cpu/cpu{cpu}/topology"
        topo.mkdir(parents=True)
        (topo / "thread_siblings_list").write_text(f"{text}\n")

    allowed = {0, 8, 1, 2, 10, 3, 11}
    assert cpu_pin.node_physical_cores(0, allowed, str(tmp_path)) == [
        (0, 8),
        (2, 10),
        (3, 11),
    ]


def test_plan_partitions_ranks_disjointly_and_preserves_housekeeping():
    cores = [(i, i + 16) for i in range(12)]
    p0 = cpu_pin.plan(cores, local_rank=0, local_world_size=4, skip=2)
    p3 = cpu_pin.plan(cores, local_rank=3, local_world_size=4, skip=2)
    assert p0 is not None and p3 is not None
    s0, o0 = p0
    s3, o3 = p3
    assert s0 == {2, 18}
    assert s3 == {5, 21}
    assert s0.isdisjoint(s3)
    assert o0 == o3
    assert s0.isdisjoint(o0) and s3.isdisjoint(o0)
    assert {0, 16, 1, 17}.issubset(o0)
    assert cpu_pin.plan(cores[:6], local_rank=0, local_world_size=4, skip=2) is None


def test_tpu_numa_nodes_reads_open_vfio_groups(tmp_path: Path):
    proc_fd = tmp_path / "fd"
    proc_fd.mkdir()
    os.symlink("/dev/vfio/vfio", proc_fd / "3")
    os.symlink("/dev/vfio/17", proc_fd / "4")
    dev = tmp_path / "sys/kernel/iommu_groups/17/devices/0000:01:00.0"
    dev.mkdir(parents=True)
    (dev / "numa_node").write_text("1\n")

    assert cpu_pin.tpu_numa_nodes(str(proc_fd), str(tmp_path / "sys")) == {1}


def test_worker_cpu_pinner_disabled_and_single_shot(monkeypatch):
    calls: list[tuple[int, set[int]]] = []
    monkeypatch.setattr(cpu_pin, "tpu_numa_nodes", lambda: {0})
    monkeypatch.setattr(cpu_pin, "vfio_group_count", lambda: 2)
    monkeypatch.setattr(
        cpu_pin.os, "sched_getaffinity", lambda _pid: set(range(16)), raising=False
    )
    monkeypatch.setattr(
        cpu_pin,
        "node_physical_cores",
        lambda _node, _allowed: [(i, i + 8) for i in range(8)],
    )
    monkeypatch.setattr(
        cpu_pin.os,
        "sched_setaffinity",
        lambda pid, cpus: calls.append((pid, set(cpus))),
        raising=False,
    )
    monkeypatch.setattr(cpu_pin.os, "listdir", lambda _p: [])

    disabled = cpu_pin.WorkerCpuPinner(
        enabled=False, local_rank=0, local_world_size=2, sweep_s=0.0
    )
    disabled.maybe_pin()
    assert calls == []

    active = cpu_pin.WorkerCpuPinner(
        enabled=True, local_rank=1, local_world_size=2, sweep_s=0.05
    )
    active.maybe_pin()
    active.maybe_pin()
    assert active.pins == 1
    assert active._sweeper is not None and active._sweeper.is_alive()
    active.close()
    assert active._sweeper is None
    assert calls[0] == (0, {3, 11})


def test_resolve_node_rank_dual_socket_uses_socket_local_cores(tmp_path: Path):
    dev_vfio = tmp_path / "vfio"
    dev_vfio.mkdir()
    sys_root = tmp_path / "sys"
    for idx in range(8):
        (dev_vfio / str(10 + idx)).write_text("")
        node = 0 if idx < 4 else 1
        bdf = f"0000:{0x80 + idx if node else idx:02x}:00.0"
        dev = sys_root / f"kernel/iommu_groups/{10 + idx}/devices/{bdf}"
        dev.mkdir(parents=True)
        (dev / "numa_node").write_text(f"{node}\n")

    proc_fd = tmp_path / "fd"
    proc_fd.mkdir()
    os.symlink(f"/dev/vfio/{10 + 5}", proc_fd / "7")

    node_rank, width = cpu_pin.resolve_node_rank(
        1,
        local_rank=5,
        local_world_size=8,
        proc_fd=str(proc_fd),
        dev_vfio=str(dev_vfio),
        sys_root=str(sys_root),
    )
    assert (node_rank, width) == (1, 4)
    cores = [(i, i + 16) for i in range(12)]
    planned = cpu_pin.plan(cores, node_rank, width, skip=2)
    assert planned is not None and planned[0] == {3, 19}
