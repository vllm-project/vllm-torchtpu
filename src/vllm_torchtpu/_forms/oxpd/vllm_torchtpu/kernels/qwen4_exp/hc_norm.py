# SPDX-License-Identifier: Apache-2.0
"""Optional prefill HC branch normalization with one BF16 output boundary."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _norm_kernel(input_ref, weight_ref, output_ref, *, hidden, branches, eps):
    values = input_ref[...].astype(jnp.float32)
    weight = weight_ref[...].astype(jnp.float32)
    pieces = []
    for branch in range(branches):
        start = branch * hidden
        segment = values[:, start : start + hidden]
        variance = jnp.mean(segment * segment, axis=-1, keepdims=True)
        # The current Torch/XLA prefill HLO rounds each normalized branch to
        # BF16, then rounds branch * weight to BF16 before adding it back.
        # Keep both boundaries while eliminating the branch materializations.
        normalized = (
            (segment * jax.lax.rsqrt(variance + eps))
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
        scaled = (
            (normalized * weight[None, start : start + hidden])
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
        pieces.append(normalized + scaled)
    output_ref[...] = jnp.concatenate(pieces, axis=-1).astype(jnp.bfloat16)


@functools.partial(jax.jit, static_argnames=("branches", "hidden", "eps", "token_tile"))
def _normalize(input_tensor, weight, *, branches, hidden, eps, token_tile):
    tokens, width = input_tensor.shape
    if (
        tokens % token_tile
        or width != branches * hidden
        or weight.shape != (width,)
        or input_tensor.dtype != jnp.bfloat16
        or weight.dtype != jnp.bfloat16
    ):
        raise ValueError("HC norm requires BF16 and whole token tiles")
    return pl.pallas_call(
        functools.partial(_norm_kernel, hidden=hidden, branches=branches, eps=eps),
        grid=(tokens // token_tile,),
        in_specs=[
            pl.BlockSpec((token_tile, width), lambda tile: (tile, 0)),
            pl.BlockSpec((width,), lambda tile: (0,)),
        ],
        out_specs=pl.BlockSpec((token_tile, width), lambda tile: (tile, 0)),
        out_shape=jax.ShapeDtypeStruct((tokens, width), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=60 << 20
        ),
        name=f"qwen4exp_hc_norm_m{token_tile}",
    )(input_tensor, weight)
