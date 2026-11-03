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

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.sparse_core.ragged_gather import ragged_gather
from vllm_torchtpu.kernels.sparse_core.ragged_gather_v2 import (
    choose_num_row_subchunks,
    ragged_gather_v2,
)

jax.config.parse_flags_with_absl()

# v7x SparseCore: 16 SIMD lanes x 2 cores x 16 subcores.
_BASE_BLOCK = 512


def _padded_rows(out_size: int, subchunks: int, base_block: int = _BASE_BLOCK) -> int:
    block = base_block * subchunks
    return -(-out_size // block) * block


class SubchunkChoiceCpuTest(parameterized.TestCase):
    @parameterized.parameters(
        (1, 1),
        (512, 1),
        (1024, 1),
        (3072, 1),
        (4096, 1),
        (4608, 1),
        (5120, 2),
        (6144, 4),
        (8192, 4),
        (16384, 4),
        (4500, 4),
        (6000, 4),
    )
    def test_choose_num_row_subchunks(self, out_size, expected):
        self.assertEqual(choose_num_row_subchunks(out_size, _BASE_BLOCK), expected)

    @parameterized.parameters(*range(1, 65))
    def test_block_multiples_are_never_padded(self, num_blocks):
        out_size = num_blocks * _BASE_BLOCK
        subchunks = choose_num_row_subchunks(out_size, _BASE_BLOCK)
        self.assertIn(subchunks, (1, 2, 4))
        self.assertEqual(_padded_rows(out_size, subchunks), out_size)

    def test_small_outputs_pad_under_one_base_block_and_valid_sweep(self):
        # Outputs in 1..8*base_block use 1 subchunk on v7x, so padding never
        # reaches a full base block; every size up to 16384 picks in {1, 2, 4}.
        for out_size in range(1, 8 * _BASE_BLOCK + 1):
            subchunks = choose_num_row_subchunks(out_size, _BASE_BLOCK)
            self.assertEqual(subchunks, 1)
            pad = _padded_rows(out_size, subchunks) - out_size
            self.assertGreaterEqual(pad, 0)
            self.assertLess(pad, _BASE_BLOCK)
        for out_size in range(8 * _BASE_BLOCK + 1, 16385):
            self.assertIn(choose_num_row_subchunks(out_size, _BASE_BLOCK), (1, 2, 4))

    @parameterized.parameters((5, 512), (6, 256))
    def test_non_v7x_generations_preserve_min_four_subchunks(
        self, generation, base_block
    ):
        for out_size in (1, base_block, 3 * base_block, 6 * base_block, 9 * base_block):
            expected = max(1, min(4, -(-out_size // base_block)))
            self.assertEqual(
                choose_num_row_subchunks(out_size, base_block, generation=generation),
                expected,
            )

    @parameterized.parameters(2560, 3072, 3584)
    def test_decode_sizes_avoid_four_subchunk_padding(self, out_size):
        # Multiples of base_block <= 8 blocks avoid the 4-block alignment pad.
        four_subchunk_cap = min(4, -(-out_size // _BASE_BLOCK))
        self.assertGreater(_padded_rows(out_size, four_subchunk_cap), out_size)
        subchunks = choose_num_row_subchunks(out_size, _BASE_BLOCK)
        self.assertEqual(_padded_rows(out_size, subchunks), out_size)


@pytest.mark.tpu_test
@jtu.with_config(jax_numpy_dtype_promotion="standard")
class GatherTest(jtu.JaxTestCase):
    @parameterized.product(
        in_out_size=[(512, 400), (512, 1024)],
        start_end=[(3, 338), (10, 422)],
        hidden_size=[128, 512, 8192],
        dtype=[jnp.int4, jnp.int8, jnp.bfloat16, jnp.float32],
        kernel_version=[2],
    )
    def test_sc_gather(
        self, in_out_size, hidden_size, start_end, dtype, kernel_version
    ):
        in_size, out_size = in_out_size
        start, end = start_end
        start = min(start, out_size)
        end = min(end, out_size)
        key = jax.random.key(0)
        x = jax.random.normal(key, (in_size, hidden_size), jnp.float32)
        x = x.astype(dtype)
        indices = jax.random.randint(key, (out_size,), 0, in_size, jnp.int32)

        start_arr = jnp.array([start], jnp.int32)
        end_arr = jnp.array([end], jnp.int32)

        kernel = ragged_gather if kernel_version == 1 else ragged_gather_v2

        actual = kernel(x, indices, start_arr, end_arr)
        actual.block_until_ready()

        # Correctness check.
        actual = actual[start:end]
        desired = x[indices][start:end]

        self.assertArraysEqual(actual, desired)

    @parameterized.product(
        # 512 is one v7x base block (16 lanes x 32 subcores); 3072 = 6 blocks
        # uses one subchunk with no padding on v7x; 4608 / 5120 / 6144 =
        # 9 / 10 / 12 blocks use 1 / 2 / 4 subchunks; 4500 is not a multiple of
        # the base block and must be padded. `partial_end` exercises EP partial
        # gathers where only `[0:end]` rows are active.
        out_size=[512, 3072, 4500, 4608, 5120, 6144],
        hidden_size=[512, 7168],
        partial_end=[False, True],
    )
    def test_sc_gather_subchunk_choices(self, out_size, hidden_size, partial_end):
        tpu_info = pltpu.get_tpu_info()
        sc_info = tpu_info.sparse_core
        assert sc_info is not None
        base_block = sc_info.num_lanes * sc_info.num_cores * sc_info.num_subcores
        expected_rows = _padded_rows(
            out_size,
            choose_num_row_subchunks(
                out_size, base_block, generation=tpu_info.generation
            ),
            base_block=base_block,
        )

        in_size = 8192
        key = jax.random.key(1)
        x = jax.random.normal(key, (in_size, hidden_size), jnp.bfloat16)
        indices = jax.random.randint(key, (out_size,), 0, in_size, jnp.int32)
        end_val = out_size // 8 if partial_end else out_size
        start = jnp.array([0], jnp.int32)
        end = jnp.array([end_val], jnp.int32)

        actual = ragged_gather_v2(x, indices, start, end)

        self.assertEqual(actual.shape, (out_size, hidden_size))
        self.assertArraysEqual(actual[:end_val], x[indices[:end_val]])

        inner_jaxpr = (
            jax.make_jaxpr(ragged_gather_v2)(x, indices, start, end)
            .jaxpr.eqns[0]
            .params["jaxpr"]
            .jaxpr
        )
        prims = {eqn.primitive.name for eqn in inner_jaxpr.eqns} | {
            sub.primitive.name
            for eqn in inner_jaxpr.eqns
            if "jaxpr" in eqn.params
            for sub in eqn.params["jaxpr"].jaxpr.eqns
        }
        if expected_rows == out_size:
            self.assertNotIn("pad", prims)
            self.assertNotIn("slice", prims)
        else:
            self.assertIn("pad", prims)
            self.assertIn("slice", prims)

    @pytest.mark.nightly
    def test_benchmark(self):
        benchmark_shapes = [
            # in_size, out_size, hidden_size
            (512, 1024, 8192),
            (1024, 2048, 8192),
            (2048, 4096, 8192),
        ]
        dtype = jnp.bfloat16

        def _time_function(fn, *args, n_repeats=100):
            # Warmup
            for _ in range(10):
                fn(*args).block_until_ready()

            # Asynchronous dispatch to hide Python overhead
            start = time.perf_counter()
            results = [fn(*args) for _ in range(n_repeats)]
            results[-1].block_until_ready()
            end = time.perf_counter()

            return (end - start) / n_repeats

        for in_size, out_size, hidden_size in benchmark_shapes:
            print(
                f"\n=== Running shape: in={in_size}, out={out_size},"
                f" hidden={hidden_size} ==="
            )

            key = jax.random.key(0)
            x = jax.random.normal(key, (in_size, hidden_size), jnp.float32).astype(
                dtype
            )
            indices = jax.random.randint(key, (out_size,), 0, in_size, jnp.int32)

            # For benchmark, use full coverage to test peak performance
            start = 0
            end = out_size
            start_arr = jnp.array([start], jnp.int32)
            end_arr = jnp.array([end], jnp.int32)

            @jax.jit
            def run_v1(x, indices, start_arr, end_arr):
                return ragged_gather(x, indices, start_arr, end_arr)

            @jax.jit
            def run_v2(x, indices, start_arr, end_arr):
                return ragged_gather_v2(x, indices, start_arr, end_arr)

            @jax.jit
            def run_jax(x, indices):
                return x[indices]

            # Run JAX baseline first
            t_jax = _time_function(run_jax, x, indices)
            res_jax = run_jax(x, indices)
            res_jax_sliced = res_jax[start:end]

            print(f"JAX:       {t_jax * 1000:.3f} ms")

            # V1 Kernel
            t_v1_str = "FAILED"
            err_v1_str = "FAILED"
            try:
                t_v1 = _time_function(run_v1, x, indices, start_arr, end_arr)
                t_v1_str = f"{t_v1 * 1000:.3f} ms"
                res_v1 = run_v1(x, indices, start_arr, end_arr)
                res_v1_sliced = res_v1[start:end]
                err_v1 = jnp.max(jnp.abs(res_v1_sliced - res_jax_sliced))
                err_v1_str = f"{float(err_v1):.6f}"
                np.testing.assert_allclose(
                    res_v1_sliced, res_jax_sliced, atol=1e-2, rtol=1e-2
                )
            except Exception:  # pylint: disable=broad-except
                print(
                    f"[Warning] V1 Kernel failed for shape ({in_size}, {out_size},"
                    f" {hidden_size})"
                )

            # V2 Kernel
            t_v2_str = "FAILED"
            err_v2_str = "FAILED"
            try:
                t_v2 = _time_function(run_v2, x, indices, start_arr, end_arr)
                t_v2_str = f"{t_v2 * 1000:.3f} ms"
                res_v2 = run_v2(x, indices, start_arr, end_arr)
                res_v2_sliced = res_v2[start:end]
                err_v2 = jnp.max(jnp.abs(res_v2_sliced - res_jax_sliced))
                err_v2_str = f"{float(err_v2):.6f}"
                np.testing.assert_allclose(
                    res_v2_sliced, res_jax_sliced, atol=1e-2, rtol=1e-2
                )
            except Exception:  # pylint: disable=broad-except
                print(
                    f"[Warning] V2 Kernel failed for shape ({in_size}, {out_size},"
                    f" {hidden_size})"
                )

            print(f"V1 Kernel: {t_v1_str}")
            print(f"V2 Kernel: {t_v2_str}")
            print(f"V1 Kernel vs JAX Max Err: {err_v1_str}")
            print(f"V2 Kernel vs JAX Max Err: {err_v2_str}")


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
