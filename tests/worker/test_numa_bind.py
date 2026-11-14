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
"""The worker NUMA bind narrows the inherited CPU set to the chip's node and
never widens it."""

import builtins
import io
from unittest import mock

from vllm_torchtpu.worker import numa_bind


def _bind(inherited_cpus, node_cpulist, node=1):
    """Run the bind against fake sysfs; return the CPU sets it applied."""
    applied = []
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if str(path).endswith(f"/node{node}/cpulist"):
            return io.StringIO(node_cpulist)
        return real_open(path, *args, **kwargs)

    with (
        mock.patch.object(
            numa_bind, "_tpu_chip_numa_node", return_value=("0000:c0:00.0", node)
        ),
        mock.patch("os.sched_getaffinity", return_value=set(inherited_cpus)),
        mock.patch(
            "os.sched_setaffinity",
            side_effect=lambda pid, cpus: applied.append(set(cpus)),
        ),
        mock.patch("builtins.open", side_effect=fake_open),
        mock.patch("ctypes.CDLL", side_effect=OSError("no libnuma")),
    ):
        numa_bind.bind_worker_to_chip_numa_node(4)
    return applied


def test_unrestricted_worker_gets_the_whole_node():
    applied = _bind(range(0, 224), "56-111,168-223\n")
    assert applied == [set(range(56, 112)) | set(range(168, 224))]


def test_launcher_reservation_is_kept_by_intersection():
    # A launcher reserved CPUs 0-7 and 56-63 for itself.
    inherited = set(range(0, 224)) - set(range(0, 8)) - set(range(56, 64))
    applied = _bind(inherited, "56-111,168-223\n")
    assert applied == [set(range(64, 112)) | set(range(168, 224))]


def test_no_cpu_on_the_node_leaves_the_inherited_set():
    # Pinned by the launcher to the other node entirely: nothing applied.
    applied = _bind(range(0, 56), "56-111,168-223\n")
    assert applied == []


def test_chip_without_numa_node_changes_nothing():
    with (
        mock.patch.object(numa_bind, "_tpu_chip_numa_node", return_value=(None, None)),
        mock.patch("os.sched_setaffinity") as set_affinity,
    ):
        numa_bind.bind_worker_to_chip_numa_node(0)
    set_affinity.assert_not_called()


def test_tpu_chips_found_in_pci_address_order(tmp_path, monkeypatch):
    devices = {
        "0000:c1:00.0": ("0x1ae0", "0xff0000", "1"),
        "0000:00:01.0": ("0x8086", "0x060000", "0"),  # Not a TPU.
        "0000:40:00.0": ("0x1ae0", "0xff0000", "0"),
        "0000:c0:00.0": ("0x1ae0", "0xff0000", "-1"),  # No NUMA node.
    }
    for address, (vendor, device_class, node) in devices.items():
        device_dir = tmp_path / address
        device_dir.mkdir()
        (device_dir / "vendor").write_text(vendor + "\n")
        (device_dir / "class").write_text(device_class + "\n")
        (device_dir / "numa_node").write_text(node + "\n")
    real_glob = numa_bind.glob.glob
    real_open = builtins.open
    monkeypatch.setattr(
        numa_bind.glob, "glob", lambda pattern: real_glob(str(tmp_path / "*"))
    )
    monkeypatch.setattr(
        builtins,
        "open",
        lambda path, *a, **k: real_open(
            str(path).replace("/sys/bus/pci/devices", str(tmp_path)), *a, **k
        ),
    )
    assert numa_bind._tpu_chip_numa_node(0) == ("0000:40:00.0", 0)
    assert numa_bind._tpu_chip_numa_node(1) == ("0000:c0:00.0", None)
    assert numa_bind._tpu_chip_numa_node(2) == ("0000:c1:00.0", 1)
    assert numa_bind._tpu_chip_numa_node(3) == (None, None)


def test_parse_cpulist():
    assert numa_bind._parse_cpulist("0-3,8,10-11\n") == [0, 1, 2, 3, 8, 10, 11]
