# SPDX-License-Identifier: Apache-2.0
"""The device order of the pipeline hand-off mesh."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
import torch
from vllm.distributed import parallel_state

from vllm_torchtpu.distributed import pp_shift


def _fake_mesh(devices, axis_names):
    return SimpleNamespace(devices=np.asarray(devices, dtype=object))


@pytest.mark.parametrize(
    ("world_size", "device_ids", "positions"),
    # Stages of one device: the pipeline group is the world, in device order.
    # Stages of two devices: the pipeline group is a standalone subgroup, whose
    # mesh lists its devices in rank order.
    [(2, (2, 6), (1, 0)), (4, (6, 2), (0, 1))],
    ids=["world", "subgroup"],
)
def test_hand_off_mesh_order(monkeypatch, world_size, device_ids, positions):
    monkeypatch.setattr(pp_shift, "_MESH", None)
    monkeypatch.setattr(
        parallel_state, "get_pp_group", lambda: SimpleNamespace(ranks=[0, 2])
    )
    monkeypatch.setattr(pp_shift, "_collect_rank_to_device_id", lambda *a: {0: 6, 2: 2})
    monkeypatch.setattr(pp_shift, "_get_current_global_rank", lambda: 0)
    monkeypatch.setattr(pp_shift, "_get_tpu_global_device_id", lambda: 6)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda *a: world_size)
    monkeypatch.setattr(
        jax, "devices", lambda: [SimpleNamespace(id=i) for i in range(8)]
    )
    monkeypatch.setattr(pp_shift, "Mesh", _fake_mesh)

    mesh = pp_shift.get_or_create_pp_mesh()
    assert tuple(device.id for device in mesh.devices) == device_ids
    assert pp_shift.pp_positions() == positions
