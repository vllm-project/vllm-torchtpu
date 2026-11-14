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
"""The small reduce-scatter kernel on the 8 cores of one host, against the
float64 sum of each core's block and ``jax.lax.psum_scatter``."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.small_collectives import \
    reduce_scatter as rs

from .oracle import (AXIS, N, bf16_ulp, blocks, exact_sum, run_sharded,
                     sum_tolerance, within)

pytestmark = pytest.mark.multichip


def _reduce_scatter(mesh, per_core, ranks, *, rank_operand=True):

    def body(x, r):
        return rs.small_reduce_scatter(x,
                                       num_cores=N,
                                       axis=AXIS,
                                       rank=r if rank_operand else None)

    return run_sharded(mesh, body, per_core, ranks)


def _own_blocks(per_core, core):
    """[N, rows, hidden]: every core's partial sums for ``core``'s rows."""
    rows = per_core.shape[1] // N
    return per_core[:, core * rows:(core + 1) * rows]


# 1, 3 and 12 rows are not multiples of the sublane tile: Mosaic refused
# them before the kernel padded each block.
@pytest.mark.parametrize("rows", [1, 3, 4, 12, 16, 24, 40, 64])
@pytest.mark.parametrize("rank_operand", [False, True])
def test_each_core_gets_the_exact_sum_of_its_block(mesh, rows, rank_operand):
    per_core = blocks((N, N * rows, 4096), seed=100 + rows)
    got = _reduce_scatter(mesh, per_core, range(N), rank_operand=rank_operand)
    for core in range(N):
        terms = _own_blocks(per_core, core)
        ok = within(got[core], exact_sum(terms), sum_tolerance(terms))
        assert ok.all(), f"core {core}: {np.count_nonzero(~ok)} elements off"


def test_agrees_with_psum_scatter_within_its_rounding(mesh):
    per_core = blocks((N, N * 16, 4096), seed=3)
    want = run_sharded(
        mesh,
        lambda x, r: jax.lax.psum_scatter(
            x, AXIS, scatter_dimension=0, tiled=True),
        per_core,
        range(N),
    ).astype(np.float64)
    got = _reduce_scatter(mesh, per_core, range(N)).astype(np.float64)
    for core in range(N):
        terms = np.abs(_own_blocks(per_core,
                                   core).astype(np.float64)).sum(axis=0)
        tolerance = (N - 1) / 2 * bf16_ulp(terms) + 2 * bf16_ulp(
            np.abs(want[core]))
        assert within(got[core], want[core], tolerance).all()


def test_a_wrong_rank_order_fails_the_oracle(mesh):
    """Positive control: with cores 0 and 1 told each other's index, their
    blocks land on the wrong core, and the oracle has to see it."""
    per_core = blocks((N, N * 16, 4096), seed=13)
    got = _reduce_scatter(mesh, per_core, [1, 0, *range(2, N)])
    for core in (0, 1):
        terms = _own_blocks(per_core, core)
        assert not within(got[core], exact_sum(terms),
                          sum_tolerance(terms)).all()


def test_refuses_a_width_off_the_lane_tile():
    with pytest.raises(ValueError, match="lanes"):
        rs.small_reduce_scatter(jnp.zeros((8, 4000), jnp.bfloat16),
                                num_cores=N,
                                axis=AXIS)


def test_refuses_rows_that_do_not_split():
    with pytest.raises(ValueError, match="split"):
        rs.small_reduce_scatter(jnp.zeros((12, 4096), jnp.bfloat16),
                                num_cores=N,
                                axis=AXIS)


@pytest.mark.parametrize("rows", [1, 4, 8, 16, 20, 24])
@pytest.mark.parametrize("rank_operand", [False, True])
@pytest.mark.parametrize("permuted", [False, True])
def test_all_gather_pair_padding_and_rank_placement(monkeypatch, rows,
                                                    rank_operand, permuted):
    with jax.default_device(jax.devices("cpu")[0]):
        n, hidden, num_experts = 8, 128, 512
        x = (jnp.arange(rows * hidden, dtype=jnp.float32).reshape(
            rows, hidden).astype(jnp.bfloat16))
        logits = jnp.arange(rows * num_experts,
                            dtype=jnp.float32).reshape(rows, num_experts)
        rank_of_core = np.array([0, 1, 6, 7, 2, 3, 4, 5], np.int32)
        mesh_core = 2
        own_rank = int(rank_of_core[mesh_core]) if permuted else mesh_core
        rank = jnp.array([mesh_core], jnp.int32) if rank_operand else None
        my_rank = jnp.array([own_rank], jnp.int32) if permuted else None
        calls = []

        def pallas_call(kernel, *, out_shape, scratch_shapes, compiler_params,
                        **kwargs):
            padded_rows = out_shape[0].shape[1]
            assert padded_rows % 8 == 0
            assert out_shape[0].shape == (n, padded_rows, hidden)
            assert out_shape[1].shape == (n, padded_rows, num_experts)
            assert scratch_shapes[0].shape == (n, padded_rows, hidden)
            assert scratch_shapes[1].shape == (n, padded_rows, num_experts)
            assert compiler_params.collective_id == rs.GATHER_COLLECTIVE_ID

            def run(*args):
                x_in, logits_in = args[-2], args[-1]
                np.testing.assert_array_equal(x_in[:rows], x)
                np.testing.assert_array_equal(x_in[rows:], 0)
                np.testing.assert_array_equal(logits_in[:rows], logits)
                np.testing.assert_array_equal(logits_in[rows:], 0)
                if rows % 8 == 0:
                    assert x_in is x
                    assert logits_in is logits
                if rank_operand:
                    np.testing.assert_array_equal(args[0], rank)
                if permuted:
                    np.testing.assert_array_equal(args[-3], my_rank)
                calls.append(True)
                out_x = jnp.zeros((n, padded_rows, hidden), dtype=x.dtype)
                out_l = jnp.zeros((n, padded_rows, num_experts),
                                  dtype=logits.dtype)
                return out_x.at[own_rank].set(x_in), out_l.at[own_rank].set(
                    logits_in)

            return run

        monkeypatch.setattr(rs.pl, "pallas_call", pallas_call)
        got_x, got_l = rs.small_all_gather_pair(x,
                                                logits,
                                                num_cores=n,
                                                rank=rank,
                                                my_rank=my_rank)
        assert len(calls) == 1
        assert got_x.shape == (n * rows, hidden)
        assert got_l.shape == (n * rows, num_experts)
        np.testing.assert_array_equal(
            got_x.reshape(n, rows, hidden)[own_rank], x)
        np.testing.assert_array_equal(
            got_l.reshape(n, rows, num_experts)[own_rank], logits)


def test_all_gather_pair_validation_and_single_core():
    with jax.default_device(jax.devices("cpu")[0]):
        x = jnp.ones((3, 128), dtype=jnp.bfloat16)
        logits = jnp.ones((3, 512), dtype=jnp.float32)
        got_x, got_l = rs.small_all_gather_pair(x, logits, num_cores=1)
        assert got_x is x
        assert got_l is logits
        with pytest.raises(ValueError, match="row mismatch"):
            rs.small_all_gather_pair(x,
                                     jnp.ones((4, 512), dtype=jnp.float32),
                                     num_cores=8)
        with pytest.raises(ValueError, match="not a multiple"):
            rs.small_all_gather_pair(
                jnp.ones((4, 100), dtype=jnp.bfloat16),
                jnp.ones((4, 512), dtype=jnp.float32),
                num_cores=8,
            )
        with pytest.raises(ValueError, match="not a multiple"):
            rs.small_all_gather_pair(
                jnp.ones((4, 128), dtype=jnp.bfloat16),
                jnp.ones((4, 500), dtype=jnp.float32),
                num_cores=8,
            )
