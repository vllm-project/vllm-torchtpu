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
"""The MoE prologue (``megakernel.qwen38.prologue``) on the host: the rows'
and metadata's all-gather with the shared expert computed under it, and the
plain all-gather.

The kernel runs on an eight-device CPU mesh with
XLA_FLAGS=--xla_force_host_platform_device_count=8; the interpreted kernel
blocks host threads in semaphore waits, so the host needs more cores than
devices.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.megakernel.qwen38.prologue import (  # noqa: E402
    all_gather,
    gather_and_share,
)

AXIS, CORES = "pcp", 8
RING = (0, 1, 2, 3, 6, 7, 4, 5)


def test_gather_and_share_gathers_every_core_and_runs_the_shared_expert():
    if len(jax.devices()) != CORES:
        pytest.skip("requires eight devices; CPU virtual devices are sufficient")
    rng = np.random.default_rng(0)
    rows, hidden, inter, k = 64, 1024, 256, 3
    slabs = rng.standard_normal((CORES * rows, 4, 128)).astype(np.float32)
    ids = rng.integers(0, 100, (CORES * rows * k,)).astype(np.int32)
    weights = rng.random((CORES * rows * k,)).astype(np.float32)
    x = rng.standard_normal((CORES * rows, hidden)).astype(np.float32) / 8
    gate_up = rng.standard_normal((2 * inter, hidden)).astype(np.float32) / 32
    down = rng.standard_normal((hidden, inter)).astype(np.float32) / 16

    mesh = Mesh(np.asarray(jax.devices()), (AXIS,))

    def run(slabs, ids, weights, x, gate_up, down):
        gathered, (ids_all, weights_all), shared = gather_and_share(
            slabs.astype(jnp.float8_e4m3fn),
            (ids, weights),
            x.astype(jnp.bfloat16),
            gate_up.astype(jnp.bfloat16),
            down.astype(jnp.bfloat16),
            RING,
            AXIS,
            interpret=True,
        )
        return gathered[None], ids_all[None], weights_all[None], shared

    sharded, replicated = P(AXIS), P()
    specs = (sharded, sharded, sharded, sharded, replicated, replicated)
    out = jax.jit(
        jax.shard_map(
            run,
            mesh=mesh,
            in_specs=specs,
            out_specs=(sharded, sharded, sharded, sharded),
            check_vma=False,
        )
    )(
        *(
            jax.device_put(jnp.asarray(a), NamedSharding(mesh, spec))
            for a, spec in zip((slabs, ids, weights, x, gate_up, down), specs)
        )
    )
    gathered, ids_all, weights_all, shared = (np.asarray(o, np.float32) for o in out)

    fp8 = np.asarray(jnp.asarray(slabs).astype(jnp.float8_e4m3fn), np.float32)
    for c in range(CORES):
        np.testing.assert_array_equal(gathered[c], fp8)
        np.testing.assert_array_equal(ids_all[c], ids)
        np.testing.assert_array_equal(weights_all[c], weights)

    def bf16(a):
        return np.asarray(jnp.asarray(a, jnp.float32).astype(jnp.bfloat16), np.float32)

    xb, gu, dn = bf16(x), bf16(gate_up), bf16(down)
    gate, up = xb @ gu[:inter].T, xb @ gu[inter:].T
    act = bf16(gate / (1 + np.exp(-gate)) * up)
    np.testing.assert_allclose(shared, bf16(act @ dn.T), rtol=2e-2, atol=2e-3)


def test_all_gather_gathers_every_core():
    if len(jax.devices()) != CORES:
        pytest.skip("requires eight devices; CPU virtual devices are sufficient")
    rows = np.random.default_rng(1).standard_normal((CORES * 128, 2048))
    mesh = Mesh(np.asarray(jax.devices()), (AXIS,))
    gathered = jax.jit(
        jax.shard_map(
            lambda r: all_gather(
                r.astype(jnp.float8_e4m3fn), RING, AXIS, interpret=True
            )[None],
            mesh=mesh,
            in_specs=P(AXIS),
            out_specs=P(AXIS),
            check_vma=False,
        )
    )(jax.device_put(jnp.asarray(rows, jnp.float32), NamedSharding(mesh, P(AXIS))))
    fp8 = np.asarray(
        jnp.asarray(rows, jnp.float32).astype(jnp.float8_e4m3fn), np.float32
    )
    for c in range(CORES):
        np.testing.assert_array_equal(np.asarray(gathered[c], np.float32), fp8)
