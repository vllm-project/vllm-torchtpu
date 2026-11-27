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
"""CPU unit tests for `vllm_torchtpu.distributed.tp_mesh`."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import jax.sharding
import numpy as np
import pytest
import torch_tpu._internal.pallas as tpu_pallas

from vllm_torchtpu.distributed import tp_mesh


@pytest.fixture(autouse=True)
def reset_tp_mesh_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tp_mesh, "_MESH_CACHE", {})
    monkeypatch.setattr(tp_mesh, "_GATHERED_DEVICE_IDS", {})
    monkeypatch.setattr(tp_mesh, "_PLACEMENT_KEY_READ", False)


def _fake_pallas_mesh(device_ids: list[int]) -> SimpleNamespace:
    devices = [
        SimpleNamespace(id=device_id, process_index=idx)
        for idx, device_id in enumerate(device_ids)
    ]
    return SimpleNamespace(devices=np.asarray(devices, dtype=object))


def _fake_jax_mesh(devices: np.ndarray, axis_names: tuple[str, ...]) -> SimpleNamespace:
    return SimpleNamespace(devices=devices, axis_names=axis_names)


def test_build_tp_mesh_returns_none_when_dist_off_or_subworld(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gather = Mock()
    monkeypatch.setattr(tp_mesh, "_collect_rank_to_device_id", gather)
    monkeypatch.setattr(tp_mesh.dist, "is_available", lambda: True)
    monkeypatch.setattr(tp_mesh.dist, "is_initialized", lambda: False)
    assert tp_mesh.build_tp_mesh() is None

    monkeypatch.setattr(tp_mesh.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(tp_mesh.dist, "get_world_size", lambda: 8)
    monkeypatch.setattr(
        tp_mesh.parallel_state,
        "get_tp_group",
        lambda: SimpleNamespace(ranks=[0, 1, 2, 3]),
    )
    assert tp_mesh.build_tp_mesh() is None
    assert tp_mesh.tp_device_placement_key() == ()
    gather.assert_not_called()


def test_build_tp_mesh_reorders_tpu7x_devices_and_caches_before_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dual-core tpu7x `[0, 1, 4, 5, 6, 7, 2, 3]` sorts to `0..7` and caches."""
    tpu7x_rank_devices = [0, 1, 4, 5, 6, 7, 2, 3]
    group = SimpleNamespace(ranks=list(range(8)))
    gather = Mock(return_value=dict(enumerate(tpu7x_rank_devices)))
    pallas_mesh_fn = Mock(
        side_effect=lambda axis_names: _fake_pallas_mesh(tpu7x_rank_devices)
    )

    monkeypatch.setattr(tp_mesh.dist, "is_available", lambda: True)
    monkeypatch.setattr(tp_mesh.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(tp_mesh.dist, "get_world_size", lambda: 8)
    monkeypatch.setattr(tp_mesh.parallel_state, "get_tp_group", lambda: group)
    monkeypatch.setattr(tp_mesh, "_get_current_global_rank", lambda: 2)
    monkeypatch.setattr(tp_mesh, "_get_tpu_global_device_id", lambda: 4)
    monkeypatch.setattr(tp_mesh, "_collect_rank_to_device_id", gather)
    monkeypatch.setattr(tpu_pallas, "get_pallas_mesh", pallas_mesh_fn)
    monkeypatch.setattr(jax.sharding, "Mesh", _fake_jax_mesh)

    built = tp_mesh.build_tp_mesh()
    assert built is not None
    mesh, order = built
    assert [int(d.id) for d in mesh.devices.flat] == list(range(8))
    assert mesh.axis_names == (tp_mesh.TP_AXIS_NAME,)
    assert order == (0, 1, 6, 7, 2, 3, 4, 5)
    assert tp_mesh.tp_device_placement_key() == (
        (tuple(range(8)), tuple(tpu7x_rank_devices)),
    )
    gather.assert_called_once_with(group, 2, 4)
    pallas_mesh_fn.assert_called_once_with(axis_names=(tp_mesh.TP_AXIS_NAME,))

    # Second call hits _MESH_CACHE before running _collect_rank_to_device_id.
    assert tp_mesh.build_tp_mesh() == (mesh, order)
    gather.assert_called_once()
    pallas_mesh_fn.assert_called_once()


def test_tp_placement_gathered_after_key_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gathering a new or changed TP placement after the cache key was read raises."""
    placement = {0: 2, 1: 3, 2: 0, 3: 1}
    monkeypatch.setattr(tp_mesh.dist, "is_available", lambda: True)
    monkeypatch.setattr(tp_mesh.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(tp_mesh.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(
        tp_mesh.parallel_state,
        "get_tp_group",
        lambda: SimpleNamespace(ranks=[0, 1, 2, 3]),
    )
    monkeypatch.setattr(tp_mesh, "_get_current_global_rank", lambda: 0)
    monkeypatch.setattr(tp_mesh, "_get_tpu_global_device_id", lambda: 2)
    monkeypatch.setattr(tp_mesh, "_collect_rank_to_device_id", lambda *_: placement)
    monkeypatch.setattr(
        tpu_pallas,
        "get_pallas_mesh",
        lambda axis_names: _fake_pallas_mesh([0, 1, 2, 3]),
    )
    monkeypatch.setattr(jax.sharding, "Mesh", _fake_jax_mesh)

    # First placement gathered after the key is read must raise.
    assert tp_mesh.tp_device_placement_key() == ()
    with pytest.raises(RuntimeError, match="after the compile cache key"):
        tp_mesh.build_tp_mesh()

    # Reset and gather before key read: same placement after key read succeeds,
    # changed placement after cache clear raises.
    tp_mesh._PLACEMENT_KEY_READ = False
    assert tp_mesh.build_tp_mesh() is not None
    assert tp_mesh.tp_device_placement_key() == (((0, 1, 2, 3), (2, 3, 0, 1)),)
    tp_mesh._MESH_CACHE.clear()
    assert tp_mesh.build_tp_mesh() is not None

    placement = {0: 0, 1: 1, 2: 2, 3: 3}
    tp_mesh._MESH_CACHE.clear()
    with pytest.raises(RuntimeError, match="after the compile cache key"):
        tp_mesh.build_tp_mesh()


@pytest.mark.parametrize(
    ("rank_map", "pallas_ids", "match"),
    [
        ({0: 0, 1: 1, 2: 2}, [0, 1, 2, 3], r"missing=\[3\]"),
        ({0: 0, 1: 2, 2: 2, 3: 3}, [0, 1, 2, 3], "same TPU global device id"),
        (
            {0: 0, 1: 1, 2: 2, 3: 3},
            [0, 1, 2, 9],
            "get_pallas_mesh device ids do not match",
        ),
    ],
    ids=["missing_rank", "duplicate_device_id", "pallas_mismatch"],
)
def test_build_tp_mesh_error_paths(
    monkeypatch: pytest.MonkeyPatch,
    rank_map: dict[int, int],
    pallas_ids: list[int],
    match: str,
) -> None:
    monkeypatch.setattr(tp_mesh.dist, "is_available", lambda: True)
    monkeypatch.setattr(tp_mesh.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(tp_mesh.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(
        tp_mesh.parallel_state,
        "get_tp_group",
        lambda: SimpleNamespace(ranks=[0, 1, 2, 3]),
    )
    monkeypatch.setattr(tp_mesh, "_get_current_global_rank", lambda: 0)
    monkeypatch.setattr(tp_mesh, "_get_tpu_global_device_id", lambda: 0)
    monkeypatch.setattr(tp_mesh, "_collect_rank_to_device_id", lambda *_: rank_map)
    monkeypatch.setattr(
        tpu_pallas, "get_pallas_mesh", lambda axis_names: _fake_pallas_mesh(pallas_ids)
    )
    monkeypatch.setattr(jax.sharding, "Mesh", _fake_jax_mesh)

    with pytest.raises(RuntimeError, match=match):
        tp_mesh.build_tp_mesh()
    assert tp_mesh._GATHERED_DEVICE_IDS == {}
