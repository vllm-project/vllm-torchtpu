# SPDX-License-Identifier: Apache-2.0
"""Merge BF16 residual branches with FP32 per-token scalar gates."""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _kernel(residual_ref, block_ref, gates_ref, output_ref, *, branches, hidden):
    block = block_ref[...].astype(jnp.float32)
    gates = gates_ref[...]
    for branch in range(branches):
        start = branch * hidden
        residual = residual_ref[:, start : start + hidden].astype(jnp.float32)
        output_ref[:, start : start + hidden] = (
            residual + block * gates[:, branch : branch + 1]
        ).astype(output_ref.dtype)


@functools.cache
def gated_residual_fn(
    branches: int, hidden: int, token_tile: int
) -> Callable[[jax.Array, jax.Array, jax.Array], jax.Array]:
    """Build a tiled residual merge with BF16 inputs/output and FP32 gates.

    Args:
        branches: Number of contiguous residual branches.
        hidden: Width of each branch, aligned to 128 TPU vector lanes.
        token_tile: Token rows per tile, a positive multiple of 128.

    Returns:
        A JAX function taking residual, block output, and scalar gates.

    Raises:
        ValueError: Geometry, operand shapes, or dtypes are unsupported.
    """
    if branches <= 0 or hidden <= 0 or hidden % 128:
        raise ValueError("Residual branches must be positive and width 128-aligned")
    if token_tile <= 0 or token_tile % 128:
        raise ValueError("Residual token tile must be a positive multiple of 128")

    def core(residual: jax.Array, block: jax.Array, gates: jax.Array) -> jax.Array:
        tokens = residual.shape[0]
        if (
            tokens == 0
            or tokens % token_tile
            or residual.shape != (tokens, branches * hidden)
            or block.shape != (tokens, hidden)
            or gates.shape != (tokens, branches)
            or residual.dtype != jnp.bfloat16
            or block.dtype != jnp.bfloat16
            or gates.dtype != jnp.float32
        ):
            raise ValueError("Residual merge requires full BF16 tiles and FP32 gates")
        return pl.pallas_call(
            functools.partial(_kernel, branches=branches, hidden=hidden),
            grid=(tokens // token_tile,),
            in_specs=(
                pl.BlockSpec((token_tile, branches * hidden), lambda i: (i, 0)),
                pl.BlockSpec((token_tile, hidden), lambda i: (i, 0)),
                pl.BlockSpec((token_tile, branches), lambda i: (i, 0)),
            ),
            out_specs=pl.BlockSpec((token_tile, branches * hidden), lambda i: (i, 0)),
            out_shape=jax.ShapeDtypeStruct(residual.shape, residual.dtype),
            name=f"gated_residual_b{branches}_h{hidden}_m{token_tile}",
        )(residual, block, gates)

    return core
