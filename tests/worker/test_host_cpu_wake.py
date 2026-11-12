# SPDX-License-Identifier: Apache-2.0
import os
import time

import pytest

try:
    from vllm_torchtpu.worker import host_cpu_wake
except ImportError:  # CPU-only checkout without the TPU stack: load the file alone.
    import importlib.util
    import logging
    import sys
    import types

    _logger_mod = types.ModuleType("vllm_torchtpu.logger")
    _logger_mod.init_logger = logging.getLogger
    sys.modules.setdefault("vllm_torchtpu", types.ModuleType("vllm_torchtpu"))
    sys.modules["vllm_torchtpu.logger"] = _logger_mod
    _path = os.path.join(
        os.path.dirname(__file__),
        "../../src/vllm_torchtpu/worker/host_cpu_wake.py",
    )
    _spec = importlib.util.spec_from_file_location("host_cpu_wake", _path)
    host_cpu_wake = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(host_cpu_wake)


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("TPU_HOST_CPU_WAKE", raising=False)
    before = os.sched_getaffinity(0)
    host_cpu_wake.apply(0, 0, 8)
    assert host_cpu_wake._spinner is None
    assert host_cpu_wake._dma_fd is None
    assert os.sched_getaffinity(0) == before


def test_pick_cpus_disjoint_per_rank():
    world = 8
    picks = [set(host_cpu_wake.pick_cpus(r, world, 1)) for r in range(world)]
    assert all(picks)
    allowed = os.sched_getaffinity(0)
    assert all(p <= allowed for p in picks)
    ncores = len(host_cpu_wake._physical_cores(sorted(allowed)))
    if ncores >= world + len(host_cpu_wake._numa_nodes()):
        for i in range(world):
            for j in range(i + 1, world):
                assert not (picks[i] & picks[j])


def test_spin_mode_pins_and_spawns_idle_spinners(monkeypatch):
    if not hasattr(os, "SCHED_IDLE"):
        pytest.skip("no SCHED_IDLE")
    before = os.sched_getaffinity(0)
    monkeypatch.setenv("TPU_HOST_CPU_WAKE", "spin")
    monkeypatch.setenv("TPU_HOST_CPU_WAKE_CORES", "1")
    try:
        cpus = set(host_cpu_wake.pick_cpus(0, 1, 1))
        host_cpu_wake.apply(0, 0, 1)
        assert os.sched_getaffinity(0) == cpus
        root = host_cpu_wake._spinner
        assert root is not None
        deadline = time.time() + 10
        pids = []
        while time.time() < deadline:
            pids = [
                int(p)
                for p in os.listdir("/proc")
                if p.isdigit() and _pgid(int(p)) == root.pid
            ]
            if len(pids) == len(cpus) and all(
                os.sched_getscheduler(p) == os.SCHED_IDLE for p in pids
            ):
                break
            time.sleep(0.1)
        assert len(pids) == len(cpus)
        assert all(os.sched_getscheduler(p) == os.SCHED_IDLE for p in pids)
        assert {c for p in pids for c in os.sched_getaffinity(p)} == cpus
    finally:
        host_cpu_wake._stop()
        os.sched_setaffinity(0, before)
    assert host_cpu_wake._spinner is None


def _pgid(pid):
    try:
        return os.getpgid(pid)
    except OSError:
        return None


def test_thread_sample_sees_this_thread():
    rows = host_cpu_wake._thread_sample()
    assert os.getpid() in rows  # main thread tid == pid
    comm, ticks, vol, invol, cpu = rows[os.getpid()]
    assert ticks >= 0 and vol >= 0 and invol >= 0 and cpu >= 0


def test_set_state_off_restores_affinity():
    before = sorted(os.sched_getaffinity(0))
    cpus = host_cpu_wake.pick_cpus(0, 1, 1)
    try:
        host_cpu_wake._set_state("pin", cpus, before)
        assert os.sched_getaffinity(0) == set(cpus)
        host_cpu_wake._set_state("off", cpus, before)
        assert os.sched_getaffinity(0) == set(before)
    finally:
        os.sched_setaffinity(0, before)
