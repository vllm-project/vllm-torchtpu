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
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized
from jax import lax
from jax._src import test_util as jtu

from vllm_torchtpu.kernels.collectives import matmul_reduce_scatter

jax.config.parse_flags_with_absl()

pytestmark = pytest.mark.multichip

P = jax.sharding.PartitionSpec


def _make_inputs(mesh, axis_name, m, n, k, dtype, seed):
    k0, k1 = jax.random.split(jax.random.key(seed), 2)
    a = jax.device_put(jax.random.normal(k0, (m, n), dtype=dtype),
                       jax.sharding.NamedSharding(mesh, P(None, axis_name)))
    w = jax.device_put(jax.random.normal(k1, (n, k), dtype=dtype),
                       jax.sharding.NamedSharding(mesh, P(axis_name, None)))
    return a, w


def _xla_reference(mesh, axis_name, a, w):
    """The unfused path: local matmul + psum_scatter, wire in a.dtype."""

    def ref(a_shard, w_shard):
        part = jnp.dot(a_shard, w_shard,
                       preferred_element_type=jnp.float32).astype(
                           a_shard.dtype)
        return lax.psum_scatter(part,
                                axis_name,
                                scatter_dimension=0,
                                tiled=True)

    return jax.jit(
        jax.shard_map(ref,
                      mesh=mesh,
                      in_specs=(P(None, axis_name), P(axis_name, None)),
                      out_specs=P(axis_name, None),
                      check_vma=False))(a, w)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class MatmulReduceScatterTest(jtu.JaxTestCase):

    @parameterized.product(
        grid_ko=[1, 2],
        grid_nc=[1, 2],
        m=[256, 1024],
    )
    def test_matmul_reduce_scatter_f32_exact(self, grid_ko, grid_nc, m):
        # fp32 inputs make the wire fp32, so the ring's accumulation matches
        # the reference tightly — this validates the ring/schedule logic.
        if jax.device_count() != 8:
            self.skipTest("Not enough devices for test")

        axis_name = "x"
        num_devices = jax.device_count()
        mesh = jax.sharding.Mesh(np.asarray(jax.devices()), (axis_name, ))
        bk_out, bnc = 512, 256
        k, n = bk_out * grid_ko, bnc * grid_nc * num_devices

        # Run several times to expose race conditions.
        for i in range(10):
            a, w = _make_inputs(mesh, axis_name, m, n, k, jnp.float32,
                                1234 + i)
            output = matmul_reduce_scatter.matmul_reduce_scatter(a,
                                                                 w,
                                                                 mesh,
                                                                 axis_name,
                                                                 bk_out=bk_out,
                                                                 bnc=bnc)
            expected = _xla_reference(mesh, axis_name, a, w)
            self.assertAllClose(output, expected, atol=1e-4, rtol=1e-4)

    @parameterized.parameters(2592, 2784)
    def test_matmul_reduce_scatter_unaligned_n_per(self, n_per):
        # n // tp_size not a multiple of 128 takes the lane-padded path; a
        # mis-mapped pad column misses this tolerance by orders of magnitude.
        if jax.device_count() != 8:
            self.skipTest("Not enough devices for test")

        axis_name = "x"
        num_devices = jax.device_count()
        mesh = jax.sharding.Mesh(np.asarray(jax.devices()), (axis_name, ))
        m, n, k = 1024, n_per * num_devices, 1024

        # Run several times to expose race conditions.
        for i in range(10):
            a, w = _make_inputs(mesh, axis_name, m, n, k, jnp.float32,
                                1234 + i)
            output = matmul_reduce_scatter.matmul_reduce_scatter(
                a, w, mesh, axis_name)
            expected = _xla_reference(mesh, axis_name, a, w)
            self.assertAllClose(output, expected, atol=1e-4, rtol=1e-4)

    @parameterized.parameters((512, 2816, None), (512, 2784, None),
                              (1024, 2816, None), (512, 2816, 1408))
    def test_matmul_reduce_scatter_folded_small_m(self, m, n_per, bnc):
        # m // tp_size < 256 with a heavy contraction takes the folded
        # schedule; n_per = 2784 also exercises fold x lane-pad and bnc = 1408
        # the stage's multi-block accumulate. A wrong stage offset misses.
        if jax.device_count() != 8:
            self.skipTest("Not enough devices for test")

        axis_name = "x"
        num_devices = jax.device_count()
        mesh = jax.sharding.Mesh(np.asarray(jax.devices()), (axis_name, ))
        n, k = n_per * num_devices, 4096

        # Run several times to expose race conditions.
        for i in range(10):
            a, w = _make_inputs(mesh, axis_name, m, n, k, jnp.float32,
                                4321 + i)
            output = matmul_reduce_scatter.matmul_reduce_scatter(a,
                                                                 w,
                                                                 mesh,
                                                                 axis_name,
                                                                 bnc=bnc)
            expected = _xla_reference(mesh, axis_name, a, w)
            self.assertAllClose(output, expected, atol=1e-4, rtol=1e-4)

    def test_matmul_reduce_scatter_bf16_no_worse_than_xla(self):
        # bf16 rounds on every wire hop, and the kernel and the unfused path
        # round in different orders — compare both against an fp32 golden and
        # require the kernel to be no worse than the existing path.
        if jax.device_count() != 8:
            self.skipTest("Not enough devices for test")

        axis_name = "x"
        mesh = jax.sharding.Mesh(np.asarray(jax.devices()), (axis_name, ))
        m, n, k = 1024, 2048, 1024

        a, w = _make_inputs(mesh, axis_name, m, n, k, jnp.bfloat16, 1234)
        golden = jax.jit(lambda a, w: jnp.einsum(
            "mn,nk->mk", a.astype(jnp.float32), w.astype(jnp.float32)),
                         out_shardings=jax.sharding.NamedSharding(
                             mesh, P(axis_name, None)))(a, w)
        output = matmul_reduce_scatter.matmul_reduce_scatter(
            a, w, mesh, axis_name)
        reference = _xla_reference(mesh, axis_name, a, w)

        kernel_err = jnp.max(jnp.abs(output.astype(jnp.float32) - golden))
        xla_err = jnp.max(jnp.abs(reference.astype(jnp.float32) - golden))
        self.assertLessEqual(float(kernel_err), 1.5 * float(xla_err))


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
