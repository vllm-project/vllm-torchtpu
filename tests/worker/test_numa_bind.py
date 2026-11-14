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
"""Tests for the per-worker NUMA bind: it narrows the inherited CPU set to the
chip's node and never widens it."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_torchtpu.worker import numa_bind, worker_options

_NODE1_CPUS = "56-111,168-223"


def _bind(inherited, node=1, node_cpulist=_NODE1_CPUS):
    applied = []

    def fake_read(path):
        assert path == f"/sys/devices/system/node/node{node}/cpulist"
        return node_cpulist

    with (
        patch.object(numa_bind, "tpu_pci_devices", return_value=["0000:c0"]),
        patch.object(numa_bind, "pci_numa_node", return_value=node),
        patch.object(numa_bind, "_read", side_effect=fake_read),
        patch("os.sched_getaffinity", return_value=set(inherited)),
        patch("os.sched_setaffinity", side_effect=lambda _, c: applied.append(c)),
        patch("ctypes.CDLL", side_effect=OSError("no libnuma")),
    ):
        numa_bind.bind_to_chip_numa_node(0, 1)
    return applied


def test_unrestricted_worker_gets_the_whole_node():
    applied = _bind(inherited=range(224))
    assert applied == [set(range(56, 112)) | set(range(168, 224))]


def test_launcher_reservation_is_kept_by_intersection():
    inherited = set(range(224)) - set(range(8)) - set(range(56, 64))
    applied = _bind(inherited=inherited)
    assert applied == [set(range(64, 112)) | set(range(168, 224))]


def test_no_cpu_on_the_node_leaves_the_inherited_set():
    assert _bind(inherited=range(56)) == []


def test_chip_without_numa_node_is_left_alone():
    with (
        patch.object(numa_bind, "tpu_pci_devices", return_value=["0000:c0"]),
        patch.object(numa_bind, "pci_numa_node", return_value=None),
        patch("os.sched_setaffinity") as setaffinity,
    ):
        numa_bind.bind_to_chip_numa_node(0, 1)
    setaffinity.assert_not_called()


@pytest.mark.parametrize(
    "local_ranks, num_chips, expected",
    [
        (8, 4, [0, 0, 1, 1, 2, 2, 3, 3]),  # tpu7x: 2 TensorCores per chip
        (4, 4, [0, 1, 2, 3]),
        (1, 1, [0]),
        (6, 4, [None] * 6),  # ranks do not divide over the chips
        (4, 0, [None] * 4),  # no TPU found in sysfs
    ],
)
def test_chip_for_local_rank(local_ranks, num_chips, expected):
    chips = [
        numa_bind.chip_for_local_rank(r, local_ranks, num_chips)
        for r in range(local_ranks)
    ]
    assert chips == expected
    assert numa_bind.chip_for_local_rank(local_ranks, local_ranks, num_chips) is None


def test_rank_past_the_chips_is_left_unbound():
    with (
        patch.object(numa_bind, "tpu_pci_devices", return_value=["a", "b", "c", "d"]),
        patch.object(numa_bind, "pci_numa_node") as node,
        patch("os.sched_setaffinity") as setaffinity,
    ):
        numa_bind.bind_to_chip_numa_node(5, 6)
    node.assert_not_called()
    setaffinity.assert_not_called()


def test_rank_binds_to_its_chips_node():
    with (
        patch.object(numa_bind, "tpu_pci_devices", return_value=["a", "b", "c", "d"]),
        patch.object(numa_bind, "pci_numa_node", return_value=None) as node,
    ):
        numa_bind.bind_to_chip_numa_node(5, 8)
    node.assert_called_once_with("c")


def test_chip_lookup_orders_tpu_functions_by_pci_address(tmp_path):
    def device(addr, vendor, pci_class, node):
        d = tmp_path / addr
        d.mkdir()
        (d / "vendor").write_text(vendor + "\n")
        (d / "class").write_text(pci_class + "\n")
        (d / "numa_node").write_text(f"{node}\n")

    device("0000:c0:00.0", "0x1ae0", "0xff0000", 1)
    device("0000:05:00.0", "0x1ae0", "0xff0000", 0)
    device("0000:06:00.0", "0x1ae0", "0x020000", 0)  # Google NIC, not a TPU
    device("0000:07:00.0", "0x8086", "0xff0000", 0)  # another vendor
    device("0000:d0:00.0", "0x1ae0", "0xff0000", -1)  # no NUMA node
    with patch.object(numa_bind, "_PCI_DEVICES", str(tmp_path)):
        assert numa_bind.tpu_pci_devices() == [
            "0000:05:00.0",
            "0000:c0:00.0",
            "0000:d0:00.0",
        ]
        assert numa_bind.pci_numa_node("0000:05:00.0") == 0
        assert numa_bind.pci_numa_node("0000:c0:00.0") == 1
        assert numa_bind.pci_numa_node("0000:d0:00.0") is None
        assert numa_bind.pci_numa_node("0000:ff:00.0") is None


def test_parse_cpulist():
    assert numa_bind.parse_cpulist("0-3,8,10-11\n") == [0, 1, 2, 3, 8, 10, 11]


def _worker(section, *, local_world_size=4, dp_local=None):
    additional = {} if section is None else {"tpu_worker": section}
    return SimpleNamespace(
        vllm_config=SimpleNamespace(additional_config=additional),
        parallel_config=SimpleNamespace(
            local_world_size=local_world_size, data_parallel_size_local=dp_local
        ),
    )


@pytest.mark.parametrize("section", [None, {}, {"numa_bind": False}])
def test_bind_is_off_by_default(section):
    from vllm_torchtpu.worker.tpu_worker import TPUWorker

    with patch("vllm_torchtpu.worker.tpu_worker.bind_to_chip_numa_node") as bind:
        TPUWorker._maybe_bind_numa(_worker(section), 3, dp_size=1)
    bind.assert_not_called()


@pytest.mark.parametrize(
    "dp_size, dp_local, expected_ranks",
    [
        (1, None, 4),  # one engine: its local ranks
        (2, 2, 8),  # two host-local DP engines share the host's chips
        (4, None, 16),  # DP size stands in for an unset local DP size
    ],
)
def test_bind_counts_the_hosts_workers(dp_size, dp_local, expected_ranks):
    from vllm_torchtpu.worker.tpu_worker import TPUWorker

    worker = _worker({"numa_bind": True}, dp_local=dp_local)
    with patch("vllm_torchtpu.worker.tpu_worker.bind_to_chip_numa_node") as bind:
        TPUWorker._maybe_bind_numa(worker, 3, dp_size=dp_size)
    bind.assert_called_once_with(3, expected_ranks)


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"numa_bind": 1}, "must be a bool"),
        ({"numa": True}, "unknown keys"),
        ("on", "must be a JSON object"),
    ],
)
def test_worker_options_are_strict(raw, message):
    with pytest.raises(ValueError, match=message):
        worker_options.parse(raw)


def test_worker_section_is_kept_out_of_the_compile_cache_key():
    from vllm_torchtpu.runner.utils import (
        HASH_IGNORED_ADDITIONAL_CONFIG_KEYS,
        WORKER_OPTIONS_SECTION,
    )

    assert worker_options.SECTION == WORKER_OPTIONS_SECTION
    assert WORKER_OPTIONS_SECTION in HASH_IGNORED_ADDITIONAL_CONFIG_KEYS
