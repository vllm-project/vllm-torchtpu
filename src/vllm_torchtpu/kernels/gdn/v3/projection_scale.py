# SPDX-License-Identifier: Apache-2.0
"""FP8 QKVZ scale layouts shared by the Torch bridge and PCP kernel."""

import math

import jax.numpy as jnp


def projection_scale_layout(weight_shape, scale_shape):
    """Validate N-major weights and return (layout, N blocks, K blocks).

    Accept tensor scales, channel vectors, N/K block grids, and the existing
    Linear GMM runtime layout [1, K_blocks, 1, N]. K block boundaries must
    align with the 128-column VMEM tile used by the FP8 projection operands.
    """
    n, k = weight_shape
    shape = tuple(scale_shape)
    if math.prod(shape) == 1:
        return "tensor", 1, 1
    if shape == (n, ):
        return "channel", n, 1
    if len(shape) == 2:
        nb, kb = shape
        layout = "grid"
    elif len(shape) == 4 and shape[0] == shape[2] == 1 and shape[3] == n:
        nb, kb = n, shape[1]
        layout = "gmm"
    else:
        raise ValueError("FP8 QKVZ scale must be scalar, per-channel [N], "
                         "block grid [N_blocks, K_blocks], or "
                         "GMM [1, K_blocks, 1, N].")
    if nb <= 0 or kb <= 0 or n % nb or k % kb:
        raise ValueError("FP8 QKVZ scale block counts must divide N and K "
                         f"exactly: weight={weight_shape}, scale={shape}.")
    if kb > 1 and (k // kb) % 128:
        raise ValueError("FP8 QKVZ K block size must be a multiple of 128 "
                         "for VMEM slicing.")
    return layout, nb, kb


def normalize_projection_scale(weight_shape, scale):
    """Return [N] for a full-K scale or [K_blocks, N] for blocked weights.

    Expands only scale metadata, never the FP8 weights. N block sharing is
    expanded before the kernel so Q/K/V/Z pieces can be gathered at arbitrary
    channel boundaries using the existing projection DMA schedule.
    """
    layout, nb, kb = projection_scale_layout(weight_shape, scale.shape)
    n, _ = weight_shape
    if layout == "tensor":
        return jnp.broadcast_to(scale.reshape(()), (n, ))
    if layout == "channel":
        return scale
    if layout == "grid":
        scale = jnp.repeat(scale, n // nb, axis=0).T
    else:
        scale = scale.reshape(kb, n)
    return scale.reshape(n) if kb == 1 else scale
