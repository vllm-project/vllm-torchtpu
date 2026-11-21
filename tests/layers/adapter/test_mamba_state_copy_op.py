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
"""Host-provided Mamba copy indices are clamped into the pool before the copy
kernel, which does no bounds checks, uses them: in both copy programs, for the
source and the destination alike. The programs run here as plain JAX with the
copy kernel replaced by a recorder.
"""

from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels import pool_adapters
from vllm_torchtpu.layers.adapter.custom_ops import mamba_state_copy_op

POOL_ROWS = 10
# In range, then below zero and past the pool, at the int32 extremes too.
INDICES = [0, 5, 9, -1, 2**30, -(2**31)]
CLAMPED = [0, 5, 9, 0, 9, 0]


def test_copy_indices_clamped_into_pool():
    idx = jnp.array(INDICES, dtype=jnp.int32)
    out = mamba_state_copy_op._clamp_rows(idx, POOL_ROWS)
    assert out.dtype == idx.dtype
    assert out.tolist() == CLAMPED


def _recorded_copy(program, *args):
    """Run a copy program with the copy kernel recording its indices."""
    seen = []

    def copy_blocks(pool, src, dst):
        seen.append((np.asarray(src).tolist(), np.asarray(dst).tolist()))
        return pool

    with patch.object(pool_adapters, "copy_blocks", copy_blocks):
        program(*args)
    (indices,) = seen
    return indices


def test_the_block_copy_clamps_src_and_dst():
    pool = jnp.zeros((POOL_ROWS, 4), jnp.float32)
    src = jnp.array(INDICES, dtype=jnp.int32)
    dst = jnp.array(INDICES[::-1], dtype=jnp.int32)
    assert _recorded_copy(mamba_state_copy_op._copy_fn, pool, src, dst) == (
        CLAMPED,
        CLAMPED[::-1],
    )


def _rows_fn(num_pools, split, step_rows):
    """The row copy's program for one geometry, without registering an op."""
    captured = {}

    def jax_op(_name, fn, **_kwargs):
        captured["fn"] = fn
        return fn

    with patch.object(mamba_state_copy_op.pallas, "jax_op", jax_op):
        mamba_state_copy_op._row_copy_fn.__wrapped__(num_pools, split, step_rows)
    return captured["fn"]


def test_the_row_copy_clamps_src_and_dst_before_expanding_them():
    """Each manager row expands into `steps` kernel-block steps; the clamped
    row, not the garbage one, is what is expanded."""
    num_pools, split, step_rows = 2, 2, 2
    steps = num_pools * split // step_rows
    pool = jnp.zeros((POOL_ROWS, num_pools * split, 3), jnp.float32)
    src = jnp.array(INDICES, dtype=jnp.int32)
    dst = jnp.array(INDICES[::-1], dtype=jnp.int32)

    def expanded(rows):
        return [r * steps + o for r in rows for o in range(steps)]

    program = _rows_fn(num_pools, split, step_rows)
    assert _recorded_copy(program, pool, src, dst) == (
        expanded(CLAMPED),
        expanded(CLAMPED[::-1]),
    )
