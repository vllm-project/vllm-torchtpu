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

import jax
import numpy as np
import pytest
import torch.distributed as dist
from jax.sharding import Mesh
from torch_tpu._internal import pallas

from vllm_torchtpu.distributed.pallas_shapes import get_kernel_mesh

pytestmark = pytest.mark.cpu_test


def _identity(value: jax.Array) -> jax.Array:
    return value


@pytest.mark.parametrize("world_size", [1, 8])
def test_rank_local_registration_in_distributed_world(monkeypatch, world_size):
    """A local kernel must register inside a larger EP process group."""
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: world_size)
    mesh = Mesh(np.array(jax.devices("cpu")[:1]), ("local",))
    if world_size > 1:
        with pytest.raises(ValueError, match="mesh size .* world size"):
            pallas.jax_op("runtime_mesh_test::invalid_local", _identity, mesh=mesh)
    pallas.jax_op(
        f"runtime_mesh_test::local_world_{world_size}",
        _identity,
        mesh=get_kernel_mesh(mesh),
    )


def test_partial_collective_mesh_still_rejected(monkeypatch):
    """Local dispatch must not bypass validation for collective meshes."""
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 8)
    mesh = Mesh(np.array([jax.devices("cpu")[0]] * 2), ("collective",))
    with pytest.raises(ValueError, match="mesh size .* world size"):
        pallas.jax_op(
            "runtime_mesh_test::invalid_partial",
            _identity,
            mesh=get_kernel_mesh(mesh),
        )


def test_full_collective_mesh_registration(monkeypatch):
    """A full-world mesh keeps the native distributed dispatch path."""
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 8)
    mesh = Mesh(np.array([jax.devices("cpu")[0]] * 8), ("collective",))
    pallas.jax_op(
        "runtime_mesh_test::full_world",
        _identity,
        mesh=get_kernel_mesh(mesh),
    )
