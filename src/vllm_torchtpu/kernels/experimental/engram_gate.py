# SPDX-License-Identifier: Apache-2.0
"""Engram gate/residual for the measured v7 decode geometry.

The caller guards shape, BF16 inputs, FP32 mask and model constants. Keep
embedding, projection and collective work outside this kernel.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

TOKENS, MULT, DIM = 288, 4, 5120
EPS, CLAMP = 1e-20, 1e-6


def gate_residual(
    hidden: jax.Array,
    kv: jax.Array,
    qw: jax.Array,
    kw: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    """Tile32 candidate with the actual model constants and BF16 inputs."""
    if hidden.shape[0] != TOKENS:
        h = hidden.astype(jnp.float32)
        keys = (
            kv[:, : MULT * DIM].reshape(hidden.shape[0], MULT, DIM).astype(jnp.float32)
        )
        values = kv[:, MULT * DIM :].astype(jnp.float32)
        dot = jnp.sum(h * qw.astype(jnp.float32) * kw.astype(jnp.float32) * keys, -1)
        dot *= lax.rsqrt(jnp.mean(h * h, axis=-1) + EPS)
        dot *= lax.rsqrt(jnp.mean(keys * keys, axis=-1) + EPS) * DIM**-0.5
        root = jnp.sqrt(jnp.maximum(jnp.abs(dot), CLAMP))
        gate = jax.nn.sigmoid(jnp.where(dot < 0, -root, root)) * mask[:, None]
        return (h + gate[:, :, None] * values[:, None, :]).astype(hidden.dtype)

    tile = 32

    def kernel(hr, kr, qr, wr, mr, output):
        h = hr[...].astype(jnp.float32)
        keys = kr[:, : MULT * DIM].reshape(tile, MULT, DIM).astype(jnp.float32)
        values = kr[:, MULT * DIM :].astype(jnp.float32)
        q, w = qr[...].astype(jnp.float32), wr[...].astype(jnp.float32)
        dot = jnp.sum(h * q * w * keys, axis=-1)
        dot *= lax.rsqrt(jnp.mean(h * h, axis=-1) + EPS)
        dot *= lax.rsqrt(jnp.mean(keys * keys, axis=-1) + EPS) * DIM**-0.5
        root = jnp.sqrt(jnp.maximum(jnp.abs(dot), CLAMP))
        gate = jax.nn.sigmoid(jnp.where(dot < 0, -root, root)) * mr[...]
        output[...] = (h + gate[:, :, None] * values[:, None, :]).astype(hidden.dtype)

    return pl.pallas_call(
        kernel,
        grid=(TOKENS // tile,),
        in_specs=[
            pl.BlockSpec((tile, MULT, DIM), lambda i: (i, 0, 0)),
            pl.BlockSpec((tile, (MULT + 1) * DIM), lambda i: (i, 0)),
            pl.BlockSpec(qw.shape, lambda i: (0, 0)),
            pl.BlockSpec(kw.shape, lambda i: (0, 0)),
            pl.BlockSpec((tile, 1), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((tile, MULT, DIM), lambda i: (i, 0, 0)),
        out_shape=jax.ShapeDtypeStruct(hidden.shape, hidden.dtype),
        name="dsv41_engram_torch_gate_residual",
    )(hidden, kv, qw, kw, mask[:, None])
