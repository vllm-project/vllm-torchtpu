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
"""A host-provided Mamba copy pair with an index outside the pool becomes the
`(0, 0)` padding pair before the copy kernel, which does no bounds checks, uses
it: in `pool_adapters.copy_blocks`, which both copy programs call, and in the
row copy before a manager row expands into kernel blocks. The programs run on
the host with the Pallas kernel replaced by a NumPy copy, and the tests check
the pool's bytes: each valid pair copies its row and every other row keeps its
bytes.
"""

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from tests.layers.adapter.mamba_state_copy_test_utils import (
    numpy_copy_blocks,
    numpy_copy_kernel,
    row_copy_program,
)
from vllm_torchtpu.kernels import pool_adapters
from vllm_torchtpu.layers.adapter.custom_ops import mamba_state_copy_op

POOL_ROWS = 10
# In range, then below zero and past the pool: the first row past it and the
# int32 extremes.
INDICES = [0, 5, 9, -1, POOL_ROWS, 2**31 - 1, -(2**31)]

# Two valid pairs (one reads the last row), then pairs that each join a live
# row to an index outside the pool, then the padding pair. Clamping the
# outside index into the pool would change row 0, 3, 8 or 9.
VALID = [(2, 7), (9, 5)]
OUTSIDE = [
    (POOL_ROWS, 3),
    (2**31 - 1, 3),
    (-1, 8),
    (6, POOL_ROWS),
    (6, -(2**31)),
    (6, 2**31 - 1),
]
PAIRS = VALID + OUTSIDE + [(0, 0)]
# A manager row whose `row * steps` wraps in int32 to kernel block 4 when a row
# has 4 steps, that is, onto manager row 1.
WRAPS = 2**30 + 1


def _indices(pairs):
    src = jnp.array([s for s, _ in pairs], dtype=jnp.int32)
    dst = jnp.array([d for _, d in pairs], dtype=jnp.int32)
    return src, dst


def _rows_pool():
    """A pool whose every int32 differs, so any moved byte shows."""
    return jnp.arange(POOL_ROWS * 4 * 3, dtype=jnp.int32).reshape(POOL_ROWS, 4, 3)


def _after_valid_copies(pool):
    out = np.array(pool)
    for s, d in VALID:
        out[d] = np.asarray(pool)[s]
    return out


def test_a_pair_with_an_index_outside_the_pool_becomes_the_null_pair():
    pairs = [(s, d) for s in INDICES for d in INDICES]
    src, dst = _indices(pairs)
    new_src, new_dst = pool_adapters.null_out_of_range_pairs(src, dst, POOL_ROWS)
    assert new_src.dtype == new_dst.dtype == jnp.int32
    in_pool = {0, 5, 9}
    expected = [(s, d) if {s, d} <= in_pool else (0, 0) for s, d in pairs]
    assert list(zip(new_src.tolist(), new_dst.tolist())) == expected


def test_the_block_copy_moves_its_valid_pairs_and_no_other_row():
    pool = _rows_pool()
    with numpy_copy_kernel():
        new_pool, _ = mamba_state_copy_op._copy_fn(pool, *_indices(PAIRS))
    np.testing.assert_array_equal(np.asarray(new_pool), _after_valid_copies(pool))


@pytest.mark.parametrize("step_rows", [1, 2, 4], ids=["steps=4", "steps=2", "steps=1"])
def test_the_row_copy_moves_its_valid_rows_and_no_other_row(step_rows):
    """Each manager row expands into `steps` kernel-block steps. A row outside
    the pool is replaced before that expansion, so one that would wrap in
    int32 back into the pool moves nothing either."""
    pool = _rows_pool()  # 2 regions x 2 kernel blocks per manager row
    pairs = PAIRS + [(WRAPS, 3), (6, WRAPS)]
    program = row_copy_program(num_pools=2, split=2, step_rows=step_rows)
    with numpy_copy_kernel():
        new_pool, _ = program(pool, *_indices(pairs))
    np.testing.assert_array_equal(np.asarray(new_pool), _after_valid_copies(pool))


def test_valid_pairs_reach_the_copy_kernel_unchanged():
    """With every pair inside the pool the guard changes no index, so both
    programs copy exactly what they copied without it."""
    seen = []

    def kernel(pool, src, dst):
        seen.append((src, dst))
        return numpy_copy_blocks(pool, src, dst)

    src, dst = _indices(VALID + [(0, 0), (9, 9)])
    with patch.object(pool_adapters, "_copy_blocks_unchecked", kernel):
        mamba_state_copy_op._copy_fn(_rows_pool(), src, dst)
        row_copy_program(num_pools=2, split=2, step_rows=2)(_rows_pool(), src, dst)
    (block_src, block_dst), (row_src, row_dst) = seen
    assert block_src.dtype == block_dst.dtype == jnp.int32
    assert (block_src.tolist(), block_dst.tolist()) == (src.tolist(), dst.tolist())

    def expanded(rows, steps=2):
        return [r * steps + o for r in rows.tolist() for o in range(steps)]

    assert (row_src.tolist(), row_dst.tolist()) == (expanded(src), expanded(dst))


@pytest.mark.parametrize("program", ["block", "row"])
def test_the_debug_check_reports_a_replaced_pair(program):
    if program == "block":
        run = mamba_state_copy_op._copy_fn
    else:
        run = row_copy_program(num_pools=2, split=2, step_rows=2)
    pool = _rows_pool()
    with numpy_copy_kernel():
        run(pool, *_indices(PAIRS))  # debug checks off: replaced, not reported
        jax.effects_barrier()
        with pl.enable_debug_checks(True):
            run(pool, *_indices(VALID))  # nothing replaced, nothing reported
            jax.effects_barrier()
            with pytest.raises(
                (RuntimeError, jax.errors.JaxRuntimeError),
                match=r"outside \[0, 10\) replaced by the \(0, 0\) padding pair",
            ):
                run(pool, *_indices(PAIRS))
                jax.effects_barrier()
