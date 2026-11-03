# SPDX-License-Identifier: Apache-2.0
"""Prototype the original mHC post arithmetic with native 3D residual refs."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def post3d(
    x: jax.Array,
    residual: jax.Array,
    post: jax.Array,
    comb: jax.Array,
) -> jax.Array:
    assert x.shape == (288, 5120) and residual.shape == (288, 4, 5120)
    assert post.shape == (288, 4, 1) and comb.shape == (288, 4, 4)
    assert x.dtype == residual.dtype == jnp.bfloat16
    assert post.dtype == comb.dtype == jnp.float32
    tile, mult, dim = 32, 4, 5120

    def kernel(x_ref, res_ref, post_ref, comb_ref, out_ref):
        # Preserve post_kernel._post_kernel's F32 order and BF16 boundary.
        value = x_ref[...].astype(jnp.float32)
        p, c = post_ref[...], comb_ref[...]
        streams = [res_ref[:, i, :].astype(jnp.float32) for i in range(mult)]
        for j in range(mult):
            acc = p[:, j : j + 1] * value
            for i in range(mult):
                k = i * mult + j
                acc = acc + c[:, k : k + 1] * streams[i]
            out_ref[:, j, :] = acc.astype(out_ref.dtype)

    return pl.pallas_call(
        kernel,
        grid=(288 // tile,),
        in_specs=(
            pl.BlockSpec((tile, dim), lambda i: (i, 0)),
            pl.BlockSpec((tile, mult, dim), lambda i: (i, 0, 0)),
            pl.BlockSpec((tile, mult), lambda i: (i, 0)),
            pl.BlockSpec((tile, mult * mult), lambda i: (i, 0)),
        ),
        out_specs=pl.BlockSpec((tile, mult, dim), lambda i: (i, 0, 0)),
        out_shape=jax.ShapeDtypeStruct(residual.shape, residual.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=100 * 1024 * 1024,
            disable_bounds_checks=True,
        ),
        name="dsv41_mhc_post_3d_layout",
    )(x, residual, post.reshape(288, mult), comb.reshape(288, mult * mult))
