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


def is_supported_shape(rows: int, weight_shape: tuple[int, ...]) -> bool:
    return (
        len(weight_shape) == 2
        and (rows, weight_shape[1], weight_shape[0]) in TUNED_TILES
    )


def logits_head(hidden: jax.Array, weight: jax.Array) -> jax.Array:
    """``hidden @ weight.T`` in BF16 with FP32 accumulation, for [vocab, K] W."""
    rows, width = hidden.shape
    vocab, weight_width = weight.shape
    assert hidden.dtype == weight.dtype == jnp.bfloat16
    assert width == weight_width
    tile = TUNED_TILES[(rows, width, vocab)]

    def kernel(hidden_ref, weight_ref, logits_ref):
        logits_ref[...] = jnp.dot(
            hidden_ref[...], weight_ref[...].T, preferred_element_type=jnp.float32
        ).astype(logits_ref.dtype)

    return pl.pallas_call(
        kernel,
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
