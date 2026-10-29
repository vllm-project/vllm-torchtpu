# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""The GDN mixer's exchanges (``megakernel.qwen38.exchange``) on the host.

Runs on an eight-device CPU mesh with
XLA_FLAGS=--xla_force_host_platform_device_count=8; the interpreted kernels
block host threads in semaphore waits, so the host needs more cores than
devices.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.megakernel.qwen38.exchange import (  # noqa: E402
    project_to_heads,
    to_rows,
)

# Two row tiles; two weight tiles per destination and two gate tiles.
AXIS, CORES, CHUNK, BLOCKS, WIDTH = "pcp", 8, 16, 8, 1280
ROWS = BLOCKS * CHUNK


def _run(f, *args, replicated=(), outputs=1):
    """``f`` per core over ``args`` (sharded by rows, except ``replicated``)."""
    if len(jax.devices()) != CORES:
        pytest.skip("requires eight devices; CPU virtual devices are sufficient")
    mesh = Mesh(np.asarray(jax.devices()), (AXIS,))
    specs = tuple(P() if i in replicated else P(AXIS) for i in range(len(args)))
    placed = [
        jax.device_put(jnp.asarray(a), NamedSharding(mesh, spec))
        for a, spec in zip(args, specs)
    ]
    out_specs = P(AXIS) if outputs == 1 else (P(AXIS),) * outputs
    run = jax.shard_map(
        f, mesh=mesh, in_specs=specs, out_specs=out_specs, check_vma=False
    )
    return jax.tree.map(np.asarray, jax.jit(run)(*placed))


def _flat_order(owned):
    """``[cores, rows, ...]`` (core ``s``'s rows) -> the step's flat rows."""
    blocks = owned.reshape(CORES, BLOCKS, CHUNK, *owned.shape[2:])
    return np.swapaxes(blocks, 0, 1).reshape(CORES * ROWS, *owned.shape[2:])


def test_project_to_heads_sends_each_core_its_columns_and_keeps_the_gate():
    rng = np.random.default_rng(0)
    hidden, gate = 256, 2048
    # Small integers and power-of-two scales: every product and sum is exact.
    x = rng.integers(-4, 5, (CORES * ROWS, hidden)).astype(np.float32)
    weight = rng.integers(-4, 5, (CORES * WIDTH, hidden)).astype(np.float32)
    scale = 2.0 ** rng.integers(-6, 0, (1, CORES * WIDTH))
    z_weight = rng.integers(-4, 5, (gate, hidden)).astype(np.float32)
    z_scale = 2.0 ** rng.integers(-6, 0, (1, gate))
    fp8 = jnp.float8_e4m3fn
    out, z = _run(
        lambda x, w, s, zw, zs: project_to_heads(
            x, w, s, zw, zs, AXIS, CORES, CHUNK, interpret=True
        ),
        x.astype(fp8),
        weight.astype(fp8),
        scale.astype(np.float32),
        z_weight.astype(fp8),
        z_scale.astype(np.float32),
        replicated=(1, 2, 3, 4),
        outputs=2,
    )
    out = out.reshape(CORES, CORES * ROWS, WIDTH).astype(np.float32)
    product = (x @ weight.T * scale).astype(jnp.bfloat16).astype(np.float32)
    product = product.reshape(CORES, ROWS, CORES, WIDTH)  # [source, row, dest, col]
    for d in range(CORES):
        np.testing.assert_array_equal(out[d], _flat_order(product[:, :, d]))
    want_z = (x @ z_weight.T * z_scale).astype(jnp.bfloat16).astype(np.float32)
    np.testing.assert_array_equal(z.astype(np.float32), want_z)


def test_to_rows_returns_each_flat_block_to_its_owner():
    rng = np.random.default_rng(1)
    heads, head_dim = 2, 128
    # Core d's flat rows of its heads.
    core = rng.standard_normal((CORES, CORES * ROWS, heads, head_dim))
    core = core.astype(jnp.bfloat16)
    out = _run(
        lambda c: to_rows(c, AXIS, CORES, CHUNK, interpret=True),
        core.reshape(-1, heads, head_dim),
    )
    # [d, j, owner, row, heads * head_dim] -> [owner, row, d, heads * head_dim]
    blocks = core.reshape(CORES, BLOCKS, CORES, CHUNK, heads * head_dim)
    want = blocks.transpose(2, 1, 3, 0, 4).reshape(CORES, ROWS, -1)
    np.testing.assert_array_equal(out.reshape(want.shape), want)
