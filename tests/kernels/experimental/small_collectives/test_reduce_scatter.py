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

from vllm_torchtpu.kernels.experimental.small_collectives import (
    reduce_scatter as rs,
)

from .oracle import (
    AXIS,
    N,
    bf16_ulp,
    blocks,
    exact_sum,
    run_sharded,
    sum_tolerance,
    within,
)

pytestmark = pytest.mark.multichip


def _reduce_scatter(mesh, per_core, ranks, *, rank_operand=True):
    def body(x, r):
        return rs.small_reduce_scatter(
            x, num_cores=N, axis=AXIS, rank=r if rank_operand else None
        )

    return run_sharded(mesh, body, per_core, ranks)


def _own_blocks(per_core, core):
    """[N, rows, hidden]: every core's partial sums for ``core``'s rows."""
    rows = per_core.shape[1] // N
    return per_core[:, core * rows : (core + 1) * rows]


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
        lambda x, r: jax.lax.psum_scatter(x, AXIS, scatter_dimension=0, tiled=True),
        per_core,
        range(N),
    ).astype(np.float64)
    got = _reduce_scatter(mesh, per_core, range(N)).astype(np.float64)
    for core in range(N):
        terms = np.abs(_own_blocks(per_core, core).astype(np.float64)).sum(axis=0)
        tolerance = (N - 1) / 2 * bf16_ulp(terms) + 2 * bf16_ulp(np.abs(want[core]))
        assert within(got[core], want[core], tolerance).all()


def test_a_wrong_rank_order_fails_the_oracle(mesh):
    """Positive control: with cores 0 and 1 told each other's index, their
    blocks land on the wrong core, and the oracle has to see it."""
    per_core = blocks((N, N * 16, 4096), seed=13)
    got = _reduce_scatter(mesh, per_core, [1, 0, *range(2, N)])
    for core in (0, 1):
        terms = _own_blocks(per_core, core)
        assert not within(got[core], exact_sum(terms), sum_tolerance(terms)).all()


def test_refuses_a_width_off_the_lane_tile():
    with pytest.raises(ValueError, match="lanes"):
        rs.small_reduce_scatter(
            jnp.zeros((8, 4000), jnp.bfloat16), num_cores=N, axis=AXIS
        )


def test_refuses_rows_that_do_not_split():
    with pytest.raises(ValueError, match="split"):
        rs.small_reduce_scatter(
            jnp.zeros((12, 4096), jnp.bfloat16), num_cores=N, axis=AXIS
        )
