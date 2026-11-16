# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

from vllm_torchtpu.distributed.mesh_utils import device_chip_coords


def test_lists_id_coords_and_core():
    devices = [
        SimpleNamespace(id=0, coords=[0, 0, 0], core_on_chip=0),
        SimpleNamespace(id=1, coords=[0, 0, 0], core_on_chip=1),
    ]
    assert device_chip_coords(devices) == [(0, (0, 0, 0), 0), (1, (0, 0, 0), 1)]


def test_missing_attributes_are_marked_not_raised():
    devices = [SimpleNamespace(id=3)]
    assert device_chip_coords(devices) == [
        (3, "coords unavailable", "core unavailable")
    ]
