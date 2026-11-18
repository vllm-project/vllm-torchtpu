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


def test_registry_is_off_by_default():
    arch = "ToyPinnedForCausalLM"
    assert not cpu_pinning.is_registered(arch)
    cpu_pinning.register_cpu_pinning(arch)
    try:
        assert cpu_pinning.is_registered(arch)
        assert not cpu_pinning.is_registered("OtherForCausalLM")
    finally:
        cpu_pinning.unregister_cpu_pinning(arch)
    assert not cpu_pinning.is_registered(arch)


def test_chip_device_discovery_from_vfio_and_iommu(monkeypatch, tmp_path):
    iommu = tmp_path / "iommu_groups"
    for group, node, pci, vendor in (
        ("12", "0", "0000:01:00.0", "0x1ae0"),
        ("15", "1", "0000:81:00.0", "0x1ae0"),
        # A non-TPU passthrough device does not count for this worker.
        ("16", "0", "0000:02:00.0", "0x10de"),
    ):
        dev_dir = iommu / group / "devices" / pci
        dev_dir.mkdir(parents=True)
        (dev_dir / "numa_node").write_text(f"{node}\n")
        (dev_dir / "vendor").write_text(f"{vendor}\n")

    fd_dir = tmp_path / "fd"
    fd_dir.mkdir()
    os.symlink("/dev/vfio/15", fd_dir / "7")
    os.symlink("/dev/vfio/16", fd_dir / "10")
    os.symlink("/dev/vfio/vfio", fd_dir / "8")
    os.symlink("/dev/null", fd_dir / "9")

    orig_glob = cpu_pinning.glob.glob

    def fake_glob(pattern: str):
        if pattern.startswith("/sys/kernel/iommu_groups/"):
            rel = pattern.removeprefix("/sys/kernel/iommu_groups/")
            return orig_glob(str(iommu / rel))
        if pattern == "/proc/self/fd/*":
            return orig_glob(str(fd_dir / "*"))
        return orig_glob(pattern)

    monkeypatch.setattr(cpu_pinning.glob, "glob", fake_glob)

    # The non-TPU device in group 16 is open too, but does not count.
    assert cpu_pinning.my_chip_devices() == [(1, "0000:81:00.0")]


def test_pin_all_threads_ignores_exited_and_warns_on_oserror(
    monkeypatch, capture_pin_logs
):
    monkeypatch.setattr(cpu_pinning.os, "listdir", lambda _p: ["101", "102", "103"])
    monkeypatch.setattr(cpu_pinning.os, "sched_getaffinity", lambda _tid: set(range(8)))
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


def test_pin_all_threads_keeps_each_thread_within_its_own_mask(monkeypatch):
    """A thread a library bound to fewer CPUs is narrowed, never widened."""
    masks = {201: set(range(16)), 202: {3, 12}, 203: {12, 13}}
    monkeypatch.setattr(
        cpu_pinning.os, "listdir", lambda _p: [str(t) for t in sorted(masks)]
    )
    monkeypatch.setattr(cpu_pinning.os, "sched_getaffinity", lambda tid: masks[tid])
    calls: dict[int, set[int]] = {}
    monkeypatch.setattr(
        cpu_pinning.os,
        "sched_setaffinity",
        lambda tid, cpus: calls.__setitem__(tid, set(cpus)),
    )

    pinned = cpu_pinning.pin_all_threads(set(range(8)))
    assert pinned == 2
    # 203 has no CPU of the node: left as it is.
    assert calls == {201: set(range(8)), 202: {3}}


def _pin_on_node0(monkeypatch, allowed):
    monkeypatch.setattr(cpu_pinning, "my_chip_devices", lambda: [(0, "0000:01:00.0")])
    monkeypatch.setattr(cpu_pinning, "node_cpus", lambda node: set(range(8)))
    monkeypatch.setattr(cpu_pinning.os, "sched_getaffinity", lambda _pid: allowed)
    pinned: list[set[int]] = []
    monkeypatch.setattr(
        cpu_pinning, "pin_all_threads", lambda cpus: pinned.append(set(cpus)) or 4
    )
    return pinned


def test_apply_pins_to_the_chip_node(monkeypatch):
    pinned = _pin_on_node0(monkeypatch, allowed=set(range(16)))
    cpu_pinning.apply(rank=5, local_rank=3)
    assert pinned == [set(range(8))]


def test_apply_never_widens_the_allowed_cpus(monkeypatch):
    """Started under taskset/numactl on part of the node: pin within it."""
    pinned = _pin_on_node0(monkeypatch, allowed={2, 3, 12})
    cpu_pinning.apply(rank=5, local_rank=3)
    assert pinned == [{2, 3}]


def test_apply_skips_when_the_node_is_outside_the_allowed_cpus(
    monkeypatch, capture_pin_logs
):
    pinned = _pin_on_node0(monkeypatch, allowed={12, 13})
    cpu_pinning.apply(rank=5, local_rank=3)
    assert pinned == []
    assert any("outside the allowed set" in r.getMessage() for r in capture_pin_logs)


def test_apply_skips_when_the_node_cpus_cannot_be_read(monkeypatch, capture_pin_logs):
    pinned = _pin_on_node0(monkeypatch, allowed=set(range(16)))

    def unreadable(node):
        raise OSError(errno.ENOENT, "No such file")

    monkeypatch.setattr(cpu_pinning, "node_cpus", unreadable)
    cpu_pinning.apply(rank=5, local_rank=3)
    assert pinned == []
    assert any("cannot read node 0 CPUs" in r.getMessage() for r in capture_pin_logs)


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
