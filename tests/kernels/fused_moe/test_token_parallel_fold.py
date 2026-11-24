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
"""Exactness of the two replica-layout shortcuts the fused EP layer can take.

- `restore_with_partial` must equal `restore(rows)` plus the group sum of the
  partial, which is what a separate all-reduce of the partial computes.
- `local_dispatch_rows` must hand the kernel, through its positions table,
  exactly the rows a flat all-gather of every shard's share would.

Runs on an eight-device CPU mesh with
XLA_FLAGS=--xla_force_host_platform_device_count=8. Integer-valued data keeps
every sum exact, so equality is bitwise.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.fused_moe.v2.token_parallel import TokenReplicaLayout

GROUPS = [
    ((0, 1, 2, 3), (4, 5, 6, 7)),
    ((7, 1, 4, 2), (6, 0, 5, 3)),
    ((0, 1), (2, 3), (4, 5), (6, 7)),
]


def _mesh():
    if len(jax.devices()) != 8:
        pytest.skip("requires eight devices; CPU virtual devices are sufficient")
    return Mesh(np.asarray(jax.devices()), ("d",))


def _replicated(layout, rows, cols, seed):
    """[8 * rows, cols]: each rank's block is its group's rows."""
    rng = np.random.default_rng(seed)
    per_group = [rng.integers(-8, 8, (rows, cols)).astype(np.float32) for _ in layout.groups]
    return np.concatenate([per_group[layout.group_of[r]] for r in range(8)])


@pytest.mark.multichip
@pytest.mark.parametrize("groups", GROUPS)
@pytest.mark.parametrize("rows", [5, 24, 256])
def test_restore_with_partial_equals_restore_plus_group_sum(groups, rows):
    mesh = _mesh()
    layout = TokenReplicaLayout.create(ep=8, input_rows=rows, groups=groups, row_alignment=8)
    x = _replicated(layout, rows, 16, 1)
    rng = np.random.default_rng(2)
    partial = rng.integers(-8, 8, (8 * rows, 16)).astype(np.float32)
    sh = NamedSharding(mesh, P("d"))

    def body(x_l, p_l):
        rank = lax.axis_index("d")
        mine = layout.split(x_l, rank) * 3
        fused = layout.restore_with_partial(mine, p_l, "d")
        reference = layout.restore(mine, "d") + lax.psum(
            p_l, "d", axis_index_groups=layout.groups
        )
        return fused, reference

    fused, reference = jax.jit(
        jax.shard_map(body, mesh=mesh, in_specs=(P("d"), P("d")),
                      out_specs=(P("d"), P("d")), check_vma=False)
    )(jax.device_put(x, sh), jax.device_put(partial, sh))
    np.testing.assert_array_equal(np.asarray(fused), np.asarray(reference))


@pytest.mark.multichip
@pytest.mark.parametrize("groups", GROUPS[:2])
@pytest.mark.parametrize("rows", [5, 24, 256])
def test_local_dispatch_rows_match_the_flat_gather(groups, rows):
    mesh = _mesh()
    layout = TokenReplicaLayout.create(ep=8, input_rows=rows, groups=groups, row_alignment=8)
    t_local, width = layout.local_rows, layout.width
    x = _replicated(layout, rows, 16, 3)
    sh = NamedSharding(mesh, P("d"))

    def body(x_l):
        rank = lax.axis_index("d")
        mine = layout.split(x_l, rank)
        flat = lax.all_gather(mine, "d", axis=0, tiled=True)
        own = jnp.concatenate(
            [layout.split_slot(x_l, jnp.int32(s)) for s in range(width)]
        )
        rows_, positions = layout.local_dispatch_rows(own, mine, rank, "d")
        # What the kernel reads for token t of mesh rank m: block positions[m].
        via_positions = jnp.concatenate(
            [
                lax.dynamic_slice_in_dim(rows_, positions[m] * t_local, t_local)
                for m in range(8)
            ]
        )
        return flat, via_positions, rows_.shape[0] + jnp.zeros((1,), jnp.int32)

    flat, via, n = jax.jit(
        jax.shard_map(body, mesh=mesh, in_specs=(P("d"),),
                      out_specs=(P("d"), P("d"), P("d")), check_vma=False)
    )(jax.device_put(x, sh))
    np.testing.assert_array_equal(np.asarray(via), np.asarray(flat))
    assert set(np.asarray(n).tolist()) == {8 * t_local}


def test_local_dispatch_refuses_more_than_two_groups():
    layout = TokenReplicaLayout.create(ep=8, input_rows=8, groups=GROUPS[2], row_alignment=8)
    with pytest.raises(ValueError, match="two replica groups"):
        layout.local_dispatch_rows(jnp.zeros((8, 4)), jnp.zeros((4, 4)), jnp.int32(0), "d")


def test_group_tables():
    layout = TokenReplicaLayout.create(ep=8, input_rows=8, groups=GROUPS[1], row_alignment=8)
    assert layout.group_of == (1, 0, 0, 1, 0, 1, 1, 0)
    assert layout.slot_groups == ((7, 6), (1, 0), (4, 5), (2, 3))
    assert layout.width == 4
