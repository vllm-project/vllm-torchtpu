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
"""`rowbucket.aligned_row`, the row-code load of the row-bucketed expert loop:
bitwise against direct indexing for every row and bucket width (interpret
mode), and a deviceless tpu7x:2x2x1 compile of the same kernel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.blocks.moe.rowbucket import aligned_row

E = 256  # rows of `rbpreT`, one per expert
G = 16  # rows of the one-hot block (`rb_small`)
WIDTHS = (16, 32, 48, 64, 96, 128)  # MB of the whole-step buckets


def _call(n, interpret):
    """Row `e` of the table and the one-hot `P [G, n]` `gather_mat` builds."""

    def kernel(e_ref, table_ref, row_ref, onehot_ref):
        crow = aligned_row(table_ref, e_ref[0], n)
        row_ref[...] = jnp.broadcast_to(crow, (8, n))
        want = jax.lax.broadcasted_iota(jnp.int32, (G, n), 0).astype(jnp.float32)
        onehot_ref[...] = jnp.where(crow == want, 1.0, 0.0).astype(jnp.bfloat16)

    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((8, n), jnp.float32),
            jax.ShapeDtypeStruct((G, n), jnp.bfloat16),
        ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        interpret=interpret,
    )


@pytest.mark.parametrize("n", WIDTHS)
def test_aligned_row_matches_direct_indexing(n):
    rng = np.random.default_rng(n)
    table = rng.integers(0, 64, size=(E, 128)).astype(np.float32)
    table[rng.random((E, 128)) < 0.5] = -1.0  # not routed, as in `rb_tables`
    call = jax.jit(_call(n, interpret=True))
    for e in range(E):
        row, onehot = call(jnp.array([e], jnp.int32), jnp.asarray(table))
        want = table[e, :n]
        np.testing.assert_array_equal(
            np.asarray(row).view(np.uint32),
            np.broadcast_to(want, (8, n)).view(np.uint32),
        )
        np.testing.assert_array_equal(
            np.asarray(onehot, np.float32),
            (want[None, :] == np.arange(G)[:, None]).astype(np.float32),
        )


@pytest.mark.parametrize("n", WIDTHS[1:])
def test_aligned_row_compiles_for_tpu7x_2x2x1(n):
    """Deviceless Mosaic compile at the bucket widths that produced garbage."""
    from jax.experimental import topologies
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    devices = topologies.get_topology_desc(
        platform="tpu", topology_name="tpu7x:2x2x1"
    ).devices
    mesh = Mesh(np.array(devices[:1]), ("x",))
    rep = NamedSharding(mesh, PartitionSpec())
    args = (
        jax.ShapeDtypeStruct((1,), jnp.int32, sharding=rep),
        jax.ShapeDtypeStruct((E, 128), jnp.float32, sharding=rep),
    )
    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
        compiled = jax.jit(_call(n, interpret=False)).lower(*args).compile()
    assert compiled.as_text()
