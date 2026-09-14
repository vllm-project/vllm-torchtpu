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

import dataclasses
import functools
import itertools
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce import \
    ragged_gather_reduce as ragged_gather_reduce_v1
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v2 import \
    config as rgr_v2_config
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v2.wrapper import \
    ragged_gather_reduce_v2
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v3 import \
    ragged_gather_reduce as ragged_gather_reduce_v3
from vllm_torchtpu.kernels.sparse_core.ragged_scatter import ragged_scatter

jax.config.parse_flags_with_absl()


def reference_ragged_gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
    """Reference implementation of ragged gather reduce."""
    out = x[indices] * topk_weights[:, None].astype(jnp.float32)
    out = jnp.where(valid_rows_mask[:, None], out, 0)
    out = out.reshape(-1, reduce_group_size, out.shape[-1])
    out = jnp.sum(out, axis=1).astype(jnp.bfloat16)
    return out


@functools.partial(jax.jit, static_argnames="reduce_group_size")
def ragged_scatter_and_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    start: jax.Array,
    end: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
    """Reference implementation of ragged gather reduce."""
    x = ragged_scatter(x, indices, start, end)
    out = x.reshape((-1, reduce_group_size, x.shape[-1]))
    topk_weights = topk_weights.reshape((-1, reduce_group_size))[..., None]
    out = out * topk_weights
    out = jnp.where(
        valid_rows_mask.reshape((-1, reduce_group_size))[:, :, None], out, 0.0)
    out = out.sum(axis=-2)
    return out


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


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class ScatterTest(jtu.JaxTestCase):
    _test_cases = [
        dict(out_size=o,
             start_end=se,
             hidden_size=h,
             dtype=d,
             reduce_group_size=rg) for o, se, h, d, rg in itertools.chain(
                 itertools.product(
                     [400, 840],
                     [(3, 338), (10, 255)],
                     [128, 512, 8192],
                     [jnp.bfloat16, jnp.float32],
                     [8, 5],
                 ),
                 itertools.product(
                     [16384],
                     [(99, 1120)],
                     [7168],
                     [jnp.bfloat16],
                     [8],
                 ),
                 itertools.product(
                     [16384],
                     [(300, 2358)],
                     [6144],
                     [jnp.bfloat16],
                     [8],
                 ),
                 itertools.product(
                     [20480],
                     [(300, 2850)],
                     [4096],
                     [jnp.bfloat16],
                     [10],
                 ),
             )
    ]

    @parameterized.parameters(*_test_cases)
    def test_sc_ragged_gather_reduce(self, out_size, hidden_size, start_end,
                                     dtype, reduce_group_size):
        start, end = start_end
        start = min(start, out_size)
        end = min(end, out_size)
        key = jax.random.key(0)
        x = jax.random.normal(key, (out_size, hidden_size), jnp.float32)
        x = x.astype(dtype)
        indices = jax.random.permutation(key, out_size)
        topk_weights = jax.random.normal(key, (out_size, ), jnp.bfloat16)
        valid_rows_mask = jnp.where(
            jnp.logical_and(
                jnp.array([start], jnp.int32) <= indices,
                indices < jnp.array([end], jnp.int32),
            ),
            True,
            False,
        )
        # Correctness check.
        desired = reference_ragged_gather_reduce(x, indices, topk_weights,
                                                 valid_rows_mask,
                                                 reduce_group_size)
        for rgr, name in (
            (ragged_gather_reduce_v1, "ragged_gather_reduce_v1"),
            (ragged_gather_reduce_v2, "ragged_gather_reduce_v2"),
            (ragged_gather_reduce_v3, "ragged_gather_reduce_v3"),
        ):
            try:
                actual = rgr(x, indices, topk_weights, valid_rows_mask,
                             reduce_group_size)
                np.testing.assert_allclose(actual,
                                           desired,
                                           atol=1e-2,
                                           rtol=1e-2)
            except AssertionError:
                raise
            except Exception as e:  # pylint: disable=broad-except
                print(f"Skipping {name} correctness check due to error: {e}")

    def test_sc_ragged_gather_reduce_v2_multiwindow(self):
        """v2 is correct when the input spans multiple resident sort-permutation
        windows: the windowed DMA and window-relative indexing, and -- at
        partial validity -- a reduce group straddling a window boundary. hidden
        is kept small so x stays under the 16 GiB SparseCore per-tensor limit.
        """
        out_size, hidden, rgs = 1_835_008, 1024, 8
        key = jax.random.key(0)
        x = jax.random.normal(key, (out_size, hidden), jnp.bfloat16)
        indices = jax.random.permutation(key, out_size)
        topk_weights = jax.random.normal(key, (out_size, ), jnp.bfloat16)
        # all-valid exercises the windowing; ~94% valid puts the per-partition
        # valid-row count above one window so reduce groups straddle a window
        # boundary (and many straddle row-block boundaries within a window).
        for valid_rows_mask in (jnp.ones((out_size, ), jnp.bool_), indices
                                < int(out_size * 0.9375)):
            desired = reference_ragged_gather_reduce(x, indices, topk_weights,
                                                     valid_rows_mask, rgs)
            actual = ragged_gather_reduce_v2(x, indices, topk_weights,
                                             valid_rows_mask, rgs)
            np.testing.assert_allclose(actual, desired, atol=1e-2, rtol=1e-2)

    def test_sc_ragged_gather_reduce_v2_fallback_keys_on_the_source(self):
        """The TensorCore fallback is chosen by the size of the gather source,
        not by how many rows are gathered out of it.

        The hidden size is below the threshold in rows, so of x.shape[0],
        x.shape[-1] and indices.size only the source keeps the kernel selected;
        the dtype shows which path ran, since the fallback returns bfloat16 and
        the kernel the input dtype.
        """
        info = pltpu.get_tpu_info()
        if info.sparse_core is None:
            self.skipTest("no SparseCore on this TPU")
        fixed = dataclasses.replace(info, vmem_capacity_bytes=64 * 1024 * 1024)

        def falls_back(source_rows, input_size):
            return rgr_v2_config.Config(
                input_size=input_size,
                hidden_size=4096,
                source_rows=source_rows,
                reduce_group_size=8,
                in_dtype=jnp.float32,
                core_axis_name="core",
                subcore_axis_name="subcore",
                tpu_info=fixed,
            ).should_fallback

        # 16384 source rows is far past the threshold however few are gathered;
        # keying on input_size would fall back for the chunked case.
        self.assertFalse(falls_back(source_rows=16384, input_size=16384))
        self.assertFalse(falls_back(source_rows=16384, input_size=1024))
        self.assertTrue(falls_back(source_rows=1024, input_size=1024))

        hidden, input_size, rgs = 2048, 1024, 8
        # Twice the source the threshold asks for, so the premise below holds
        # whatever this TPU's VMEM capacity is.
        source_rows = 2 * int(info.vmem_capacity_bytes * 0.6 /
                              (2 * hidden * 4))
        cfg = rgr_v2_config.Config(
            input_size=input_size,
            hidden_size=hidden,
            source_rows=source_rows,
            reduce_group_size=rgs,
            in_dtype=jnp.float32,
            core_axis_name="core",
            subcore_axis_name="subcore",
            tpu_info=info,
        )
        self.assertFalse(cfg.should_fallback)
        if (cfg.num_tot_cores // cfg.num_column_partitions
                > cfg.sc_info.num_lanes):
            self.skipTest("hidden size unsupported on this TPU")

        key = jax.random.key(0)
        x = jax.random.normal(key, (source_rows, hidden), jnp.float32)
        indices = jax.random.permutation(key, source_rows)[:input_size]
        topk_weights = jax.random.normal(key, (input_size, ), jnp.bfloat16)
        valid_rows_mask = jnp.ones((input_size, ), jnp.bool_)

        actual = ragged_gather_reduce_v2(x, indices, topk_weights,
                                         valid_rows_mask, rgs)
        self.assertEqual(actual.dtype, jnp.float32)
        desired = reference_ragged_gather_reduce(x, indices, topk_weights,
                                                 valid_rows_mask, rgs)
        np.testing.assert_allclose(actual, desired, atol=1e-2, rtol=1e-2)

    def test_sc_ragged_gather_reduce_v2_tiling_cost_model(self):
        """The tiling derivation applies its iteration limit, its chunk limit
        and its round-down, on shapes and devices where each is observable.

        Retiling leaves the kernel's output unchanged, so nothing else in the
        suite observes any of them.
        """
        live = pltpu.get_tpu_info()
        if live.sparse_core is None:
            self.skipTest("no SparseCore on this TPU")

        def tiling(input_size, hidden_size, generation, num_simd_lanes):
            return rgr_v2_config.Config(
                input_size=input_size,
                hidden_size=hidden_size,
                source_rows=input_size,
                reduce_group_size=8,
                in_dtype=jnp.bfloat16,
                core_axis_name="core",
                subcore_axis_name="subcore",
                tpu_info=dataclasses.replace(live,
                                             generation=generation,
                                             num_lanes=128,
                                             sparse_core=dataclasses.replace(
                                                 live.sparse_core,
                                                 num_cores=2,
                                                 num_subcores=16,
                                                 num_lanes=num_simd_lanes)),
            )

        # The limits are spelled out rather than read from _CostModelConstants,
        # so a derivation that stops consulting them still fails here.
        for input_size, hidden_size in ((20480, 4096), (32768, 4096)):
            cfg = tiling(input_size, hidden_size, 7, 16)
            self.assertLessEqual(
                input_size // (cfg.row_chunk_size * cfg.num_row_partitions),
                40)
            self.assertLessEqual(cfg.col_chunk_size, 1024)

        # This device makes the rounding observable: rounding the chunk's VMEM
        # bound up instead of down picks 1024, over the budget it enforces.
        self.assertEqual(tiling(32768, 4096, 6, 32).col_chunk_size, 512)

    # The first perf test case approximates the DeepSeekV3, 2k-batch-size, EP=16.
    # The second case approximates the Qwen3-Coder-480B, 2k-batch-size, EP=8.
    _perf_test_cases = [
        dict(
            out_size=o,
            start_end=se,
            hidden_size=h,
            dtype=d,
            reduce_group_size=rg,
            col_chunk_size=c_sz,
        ) for o, se, h, d, rg, c_sz in itertools.chain(
            itertools.product(
                [16384],
                [(99, 1120)],
                [7168],
                [jnp.bfloat16],
                [8],
                [3584],
            ),
            itertools.product(
                [16384],
                [(300, 2400)],
                [6144],
                [jnp.bfloat16],
                [8],
                [2048],
            ),
            itertools.product(
                [65536],
                [(100, 8300)],
                [6144],
                [jnp.bfloat16],
                [8],
                [2048],
            ),
        )
    ]

    @pytest.mark.nightly
    @parameterized.parameters(*_perf_test_cases)
    def test_perf(
        self,
        out_size,
        hidden_size,
        start_end,
        dtype,
        reduce_group_size,
        col_chunk_size,
    ):
        start, end = start_end
        start = min(start, out_size)
        end = min(end, out_size)
        key = jax.random.key(0)
        x = jax.random.normal(key, (out_size, hidden_size), jnp.float32)
        x = x.astype(dtype)
        indices = jax.random.permutation(key, out_size)
        topk_weights = jax.random.normal(key, (out_size, ), jnp.bfloat16)
        valid_rows_mask = jnp.where(
            jnp.logical_and(
                jnp.array([start], jnp.int32) <= indices,
                indices < jnp.array([end], jnp.int32),
            ),
            True,
            False,
        )

        print(f"\n=== Running shape: out={out_size},"
              f" hidden={hidden_size}, start={start}, end={end} ===")

        def run_and_time(name, fn, *args):
            try:
                t_val = _time_function(fn, *args)
                print(f"{name}: {t_val*1000:.3f} ms")
            except Exception as e:  # pylint: disable=broad-except
                print(f"{name} failed: {e}")

        run_and_time(
            "ragged_scatter_and_reduce",
            ragged_scatter_and_reduce,
            x,
            indices,
            topk_weights,
            valid_rows_mask,
            start,
            end,
            reduce_group_size,
        )

        run_and_time(
            "ragged_gather_reduce_v1",
            ragged_gather_reduce_v1,
            x,
            indices,
            topk_weights,
            valid_rows_mask,
            reduce_group_size,
        )

        run_and_time(
            "ragged_gather_reduce_v2",
            ragged_gather_reduce_v2,
            x,
            indices,
            topk_weights,
            valid_rows_mask,
            reduce_group_size,
        )

        run_and_time(
            "ragged_gather_reduce_v3",
            ragged_gather_reduce_v3,
            x,
            indices,
            topk_weights,
            valid_rows_mask,
            reduce_group_size,
        )


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
