# SPDX-License-Identifier: Apache-2.0
"""BF16 linear head writing BF16 logits with full-K weight tiles."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Measured on v7x for the DSpark drafter's [240, 5120] x [129280, 5120]^T:
# 435 us against 530 us for the XLA dot (the HBM floor is 448 us). The 80-
# and 160-row slot-major draft buckets (16 and 32 requests per slot) reuse the
# tile: the weight stream is the same and their VMEM footprint is smaller.
# The target model's sampled rows (8-128), device time against the XLA dot:
# 405/469 us at 8 and 16 rows, 424/497 at 32, 435/518 at 64, 455/527 at 128.
TUNED_TILES = {
    (8, 5120, 129280): 1280,
    (16, 5120, 129280): 1280,
    (32, 5120, 129280): 1280,
    (64, 5120, 129280): 1280,
    (80, 5120, 129280): 1280,
    (128, 5120, 129280): 1280,
    (160, 5120, 129280): 1280,
    (240, 5120, 129280): 1280,
}

# `logits_head_slice` (the vocabulary-parallel heads) runs the 1280-column
# tile of the replicated head on 8 ranks' rows: each tile's logits are then
# the same dot, bit for bit (the row count does not change them; another tile
# width does).
SLICE_TILE = 1280
# Rows of 8 ranks' target rows (8-64 each) and DSpark slot blocks (5 x 8-32).
SLICE_ROWS = (64, 128, 256, 320, 512, 640, 1280)
SLICE_BLOCK_ROWS = 640
SLICE_VMEM_LIMIT_BYTES = 56 * 1024 * 1024


def is_supported_shape(rows: int, weight_shape: tuple[int, ...]) -> bool:
    return (
        len(weight_shape) == 2
        and (rows, weight_shape[1], weight_shape[0]) in TUNED_TILES
    )


def _kernel(hidden_ref, weight_ref, logits_ref):
    logits_ref[...] = jnp.dot(
        hidden_ref[...], weight_ref[...].T, preferred_element_type=jnp.float32
    ).astype(logits_ref.dtype)


def logits_head(hidden: jax.Array, weight: jax.Array) -> jax.Array:
    """``hidden @ weight.T`` in BF16 with FP32 accumulation, for [vocab, K] W."""
    rows, width = hidden.shape
    vocab, weight_width = weight.shape
    assert hidden.dtype == weight.dtype == jnp.bfloat16
    assert width == weight_width
    tile = TUNED_TILES[(rows, width, vocab)]

    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((rows, vocab), jnp.bfloat16),
        grid=(vocab // tile,),
        in_specs=(
            pl.BlockSpec((rows, width), lambda n: (0, 0)),
            pl.BlockSpec((tile, width), lambda n: (n, 0)),
        ),
        out_specs=pl.BlockSpec((rows, tile), lambda n: (0, n)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        name=f"logits_head_m{rows}_n{tile}",
    )(hidden, weight)


def is_supported_slice(rows: int, width: int) -> bool:
    return rows in SLICE_ROWS and width == 5120


def logits_head_slice(
    hidden: jax.Array, weight: jax.Array, *, tile_start: int = 0, num_tiles: int
) -> jax.Array:
    """Logits of ``num_tiles`` ``SLICE_TILE``-row tiles of [vocab, K] W from
    tile ``tile_start``: [rows, num_tiles * SLICE_TILE]."""
    rows, width = hidden.shape
    assert hidden.dtype == weight.dtype == jnp.bfloat16
    assert width == weight.shape[1] and is_supported_slice(rows, width)
    assert (tile_start + num_tiles) * SLICE_TILE <= weight.shape[0]
    tile = SLICE_TILE
    # Past SLICE_BLOCK_ROWS the rows go in blocks, each streaming the weight
    # tiles again: such row counts are MXU-bound, and the rows fit VMEM.
    block = min(rows, SLICE_BLOCK_ROWS)
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((rows, num_tiles * tile), jnp.bfloat16),
        grid=(rows // block, num_tiles),
        in_specs=(
            # One buffer: a row block is fetched once per pass over the tiles.
            pl.BlockSpec(
                (block, width), lambda r, n: (r, 0), pipeline_mode=pl.Buffered(1)
            ),
            pl.BlockSpec((tile, width), lambda r, n: (n + tile_start, 0)),
        ),
        out_specs=pl.BlockSpec((block, tile), lambda r, n: (r, n)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"),
            vmem_limit_bytes=SLICE_VMEM_LIMIT_BYTES,
        ),
        name=f"logits_head_slice_m{rows}_t{num_tiles}",
    )(hidden, weight)
