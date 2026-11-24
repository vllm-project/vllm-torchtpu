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
"""Unit lower-triangular solvers for chunked GDN."""

import jax
import jax.numpy as jnp


def solve_triangular_rhs(
    strict_lower: jax.Array, rhs: jax.Array, block_size: int
) -> jax.Array:
    """Solves the unit lower-triangular system (I + L) @ x = rhs in blocks.

    Args:
        strict_lower: L, the strictly lower-triangular part of the system
            matrix, [num_heads, chunk, chunk]. Its diagonal and upper triangle
            must be zero; the unit diagonal is implicit.
        rhs: Right-hand side [num_heads, chunk, n].
        block_size: Diagonal block size B. Blocks are solved in order; each
            block first removes the coupling to the already-solved rows with
            one matmul and then runs a dense in-block forward substitution.

    Returns:
        x with the dtype of `strict_lower`.
    """
    out_dtype = strict_lower.dtype
    chunk = strict_lower.shape[-1]
    block_size = min(block_size, chunk)
    num_blocks = (chunk + block_size - 1) // block_size
    rhs_f32 = rhs.astype(jnp.float32)

    def local_forward_sub(l_block: jax.Array, b_block: jax.Array) -> jax.Array:
        # Column-oriented (right-looking) substitution over the whole dense
        # [num_heads, B, n] block: after step j, row j is final and the rows
        # below it have column j's contribution removed. Rows <= j are
        # untouched because their coefficient in the strictly lower block is
        # zero, so each step is one multiply-subtract over B rows held densely
        # in sublanes. A row-by-row chain costs one vreg op per head per
        # (row, column) pair instead, which made the solve VALU-bound at 8-
        # and 16-row blocks (few V heads per core).
        x = b_block
        for j in range(l_block.shape[1] - 1):
            x = x - l_block[:, :, j : j + 1] * x[:, j : j + 1, :]
        return x

    x_blocks = []
    for i in range(num_blocks):
        start = i * block_size
        end = min((i + 1) * block_size, chunk)
        target_b = rhs_f32[:, start:end, :]
        if i > 0:
            solved_x = jnp.concatenate(x_blocks, axis=1)
            prev_sum = jax.lax.dot(
                strict_lower[:, start:end, :start],
                solved_x,
                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            target_b = target_b - prev_sum

        l_block = strict_lower[:, start:end, start:end].astype(jnp.float32)
        x_blocks.append(local_forward_sub(l_block, target_b).astype(out_dtype))

    return jnp.concatenate(x_blocks, axis=1)
