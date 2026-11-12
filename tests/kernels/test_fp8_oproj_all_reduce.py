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
"""The fused row-parallel FP8 matmul + all-reduce (TPU_FUSED_OPROJ_AR).

The kernel must give, bit for bit, what the two-call path gives: fp8_skinny's
matmul on each core, then the small all-reduce of the partials. The bitwise
tests run on the eight cores of one host; the geometry tests run anywhere.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.small_collectives import all_reduce
from vllm_torchtpu.kernels.quantized_matmul import fp8_oproj_all_reduce as k
from vllm_torchtpu.kernels.quantized_matmul import fp8_skinny

N = 8
AXIS = "d"
K_OPROJ, N_OPROJ = 1024, 6144


def test_tile_matches_fp8_skinny_o_proj_config():
    split, tile, nbuf = fp8_skinny.TUNED[(K_OPROJ, N_OPROJ)]
    assert split == "n" and k.TUNED[(K_OPROJ, N_OPROJ)] == (tile, nbuf)


@pytest.mark.parametrize(
    "m,kk,n,cores,ok",
    [
        (4, 1024, 6144, 16, True),
        (16, 1024, 6144, 8, True),
        (17, 1024, 6144, 16, False),  # above 16 rows
        (4, 2048, 16384, 16, False),  # not a tuned shape
        (4, 1024, 6144, 1, False),
        (4, 1024, 6144, 32, False),  # 192-lane owner chunks
    ],
)
def test_supports(m, kk, n, cores, ok):
    assert k.supports(m, kk, n, cores) is ok


def test_owner_waves_cover_every_owner_once_after_its_last_column():
    waves = k.owner_waves(16, 6144, 2048)
    assert waves == [list(range(0, 5)), list(range(5, 10)), list(range(10, 16))]


@pytest.fixture(scope="module")
def mesh():
    """The eight cores of one host, asked for inside the test so that pytest
    collection does not claim the chip."""
    if jax.default_backend() != "tpu" or jax.device_count() < N:
        pytest.skip("needs the 8 TPU cores of one host")
    return Mesh(np.array(jax.devices()[:N]), (AXIS,))


def _operands(rows, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N * rows, K_OPROJ), dtype=np.float32)
    w = rng.standard_normal((N * K_OPROJ, N_OPROJ), dtype=np.float32) * 64
    w_s = rng.uniform(1e-4, 1e-3, N * N_OPROJ).astype(np.float32)
    return (
        jnp.asarray(x, jnp.bfloat16),
        jnp.asarray(w, jnp.float8_e4m3fn),
        jnp.asarray(w_s),
        jnp.arange(N, dtype=jnp.int32),
    )


def _run(mesh, body, operands):
    spec = P(AXIS)
    args = [jax.device_put(a, NamedSharding(mesh, spec)) for a in operands]
    fn = jax.shard_map(
        body, mesh=mesh, in_specs=(spec,) * 4, out_specs=spec, check_vma=False
    )
    return np.asarray(jax.jit(fn)(*args))


@pytest.mark.multichip
@pytest.mark.parametrize("waves", [False, True])
@pytest.mark.parametrize("rows", [1, 4, 16])
def test_bitwise_the_two_call_path(mesh, rows, waves):
    def fused(x, w, w_s, rank):
        return k.fp8_matmul_all_reduce(
            x, w, w_s, rank, num_cores=N, axis=AXIS, waves=waves
        )

    def two_calls(x, w, w_s, rank):
        y = fp8_skinny.fp8_skinny_matmul(x, w, w_s)
        return all_reduce.small_all_reduce(y, num_cores=N, axis=AXIS, rank=rank)

    operands = _operands(rows, seed=rows + 17 * int(waves))
    got = _run(mesh, fused, operands)
    want = _run(mesh, two_calls, operands)
    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got.view(np.int16), want.view(np.int16))
