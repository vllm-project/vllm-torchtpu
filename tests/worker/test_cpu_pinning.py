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

import errno
import logging
import os

import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.worker import cpu_pinning


@pytest.fixture
def capture_pin_logs():
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _ListHandler(level=logging.WARNING)
    cpu_pinning.logger.addHandler(handler)
    try:
        yield records
    finally:
        cpu_pinning.logger.removeHandler(handler)


def test_cpulist_parses_ranges_and_singles():
    assert cpu_pinning._cpulist("0-2,5,112-113") == [0, 1, 2, 5, 112, 113]


def _node0(monkeypatch, cores=56):
    monkeypatch.setattr(
        cpu_pinning, "physical_cores", lambda node: [(c, c + 112) for c in range(cores)]
    )


def test_numa_mode_gives_the_whole_node(monkeypatch):
    _node0(monkeypatch)
    assert len(cpu_pinning.plan("numa", 0, 1, 4, reserve=2)) == 112


def test_slices_are_disjoint_whole_cores_and_keep_the_reserve(monkeypatch):
    _node0(monkeypatch)
    slices = [cpu_pinning.plan("slice", 0, slot, 4, reserve=2) for slot in range(4)]
    assert all(len(s) == 26 for s in slices)  # 13 cores with their SMT siblings
    union = set().union(*slices)
    assert len(union) == 4 * 26
    assert all((c + 112 in s) == (c in s) for s in slices for c in range(56))
    assert not union & {54, 55, 166, 167}  # the reserved last two cores


def test_cpu_pin_env_rejects_invalid_value_and_is_compile_ignored(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_CPU_PIN", "bogus")
    with pytest.raises(ValueError, match="VLLM_TPU_CPU_PIN"):
        _ = envs.environment_variables["VLLM_TPU_CPU_PIN"]()

    for valid in ("off", "numa", "slice"):
        monkeypatch.setenv("VLLM_TPU_CPU_PIN", valid)
        assert envs.environment_variables["VLLM_TPU_CPU_PIN"]() == valid

    for name in (
        "VLLM_TPU_CPU_PIN",
        "VLLM_TPU_CPU_PIN_RESERVE",
        "VLLM_TPU_CPU_PROBE_S",
    ):
        assert name in tpu_compiler._TPU_COMPILE_ENV_IGNORED


def test_chip_device_discovery_from_vfio_and_iommu(monkeypatch, tmp_path):
    iommu = tmp_path / "iommu_groups"
    for group, node, pci in (
        ("12", "0", "0000:01:00.0"),
        ("15", "1", "0000:81:00.0"),
    ):
        dev_dir = iommu / group / "devices" / pci
        dev_dir.mkdir(parents=True)
        (dev_dir / "numa_node").write_text(f"{node}\n")

    fd_dir = tmp_path / "fd"
    fd_dir.mkdir()
    os.symlink("/dev/vfio/15", fd_dir / "7")
    os.symlink("/dev/vfio/vfio", fd_dir / "8")
    os.symlink("/dev/null", fd_dir / "9")

    vfio_dir = tmp_path / "vfio"
    vfio_dir.mkdir()
    (vfio_dir / "12").write_text("")
    (vfio_dir / "15").write_text("")

    orig_glob = cpu_pinning.glob.glob

    def fake_glob(pattern: str):
        if pattern.startswith("/sys/kernel/iommu_groups/"):
            rel = pattern.removeprefix("/sys/kernel/iommu_groups/")
            return orig_glob(str(iommu / rel))
        if pattern == "/proc/self/fd/*":
            return orig_glob(str(fd_dir / "*"))
        if pattern == "/dev/vfio/[0-9]*":
            return orig_glob(str(vfio_dir / "[0-9]*"))
        return orig_glob(pattern)

    monkeypatch.setattr(cpu_pinning.glob, "glob", fake_glob)

    assert cpu_pinning.my_chip_devices() == [(1, "0000:81:00.0")]
    assert cpu_pinning.all_chip_devices() == [
        (0, "0000:01:00.0"),
        (1, "0000:81:00.0"),
    ]


def test_pin_all_threads_ignores_exited_and_warns_on_oserror(
    monkeypatch, capture_pin_logs
):
    monkeypatch.setattr(cpu_pinning.os, "listdir", lambda _p: ["101", "102", "103"])
    calls: list[tuple[int, set[int]]] = []

    def fake_setaffinity(tid: int, cpus: set[int]) -> None:
        calls.append((tid, set(cpus)))
        if tid == 102:
            raise ProcessLookupError(errno.ESRCH, "No such process")
        if tid == 103:
            raise OSError(errno.EINVAL, "Invalid argument")

    monkeypatch.setattr(cpu_pinning.os, "sched_setaffinity", fake_setaffinity)

    pinned = cpu_pinning.pin_all_threads({2, 3})
    assert pinned == 1
    assert [tid for tid, _ in calls] == [101, 102, 103]
    assert len(capture_pin_logs) == 1
    assert "CPU pin failed on 1 thread(s)" in capture_pin_logs[0].getMessage()


def test_apply_assigns_disjoint_peer_slots_on_node(monkeypatch):
    _node0(monkeypatch, cores=10)
    peers = [
        (0, "0000:01:00.0"),
        (0, "0000:02:00.0"),
        (1, "0000:81:00.0"),
    ]
    monkeypatch.setattr(cpu_pinning, "all_chip_devices", lambda: peers)
    monkeypatch.setenv("VLLM_TPU_CPU_PIN", "slice")
    monkeypatch.setenv("VLLM_TPU_CPU_PIN_RESERVE", "2")
    monkeypatch.setenv("VLLM_TPU_CPU_PROBE_S", "0")

    pinned_sets: list[set[int]] = []
    monkeypatch.setattr(
        cpu_pinning,
        "pin_all_threads",
        lambda cpus: pinned_sets.append(set(cpus)) or 4,
    )

    monkeypatch.setattr(cpu_pinning, "my_chip_devices", lambda: [(0, "0000:01:00.0")])
    cpu_pinning.apply(rank=5, local_rank=3)

    monkeypatch.setattr(cpu_pinning, "my_chip_devices", lambda: [(0, "0000:02:00.0")])
    cpu_pinning.apply(rank=6, local_rank=0)

    assert len(pinned_sets) == 2
    assert pinned_sets[0].isdisjoint(pinned_sets[1])
    # 10 cores - 2 reserved = 8 usable cores -> 4 physical cores (8 logical CPUs) each
    assert len(pinned_sets[0]) == 8
    assert len(pinned_sets[1]) == 8


@pytest.mark.parametrize(
    "mine",
    [
        [],
        [(-1, "0000:01:00.0")],
        [(0, "0000:01:00.0"), (1, "0000:81:00.0")],
    ],
)
def test_apply_skips_with_warning_on_ambiguous_or_negative_node(
    monkeypatch, capture_pin_logs, mine
):
    monkeypatch.setenv("VLLM_TPU_CPU_PIN", "numa")
    monkeypatch.setenv("VLLM_TPU_CPU_PROBE_S", "0")
    monkeypatch.setattr(cpu_pinning, "my_chip_devices", lambda: mine)
    called = False

    def fail_if_called(_cpus):
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(cpu_pinning, "pin_all_threads", fail_if_called)

    cpu_pinning.apply(rank=0, local_rank=0)
    assert not called
    assert any("CPU pin skipped" in r.getMessage() for r in capture_pin_logs)
