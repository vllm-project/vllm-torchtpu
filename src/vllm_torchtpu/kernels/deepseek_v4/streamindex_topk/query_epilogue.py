# SPDX-License-Identifier: Apache-2.0
"""Indexer RoPE, FP8 quantization and head-weight scale folding."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import rope


def _body(q_ref, cs, hw, qout, sout, *, softmax_scale):
    q = q_ref[...].astype(jnp.float32)
    cos, sin = rope._cos_sin_lanes(cs[...], rotary_dim=64, inverse=False)
    y = q * cos[:, None, :] + rope._rotate_gptj(q) * sin[:, None, :]
    amax = jnp.max(jnp.abs(y), axis=-1, keepdims=True)
    scale = amax / 448.0
    inv = jnp.where(amax == 0, 0.0, 1.0 / scale)
    qout[...] = (y * inv).astype(jnp.float8_e4m3fn)
    weights = hw[...]
    if hw.dtype == jnp.bfloat16:
        # The TPU XLA pointwise fusion folds the two scalar factors in
        # bf16 but carries the product with weights in fp32. Reproduce that
        # compiled behavior instead of rounding an intermediate to bf16.
        factor = jnp.asarray(softmax_scale * (64**-0.5), jnp.bfloat16).astype(
            jnp.float32
        )
        weights = weights.astype(jnp.float32) * factor
    else:
        weights = weights * softmax_scale * (64**-0.5)
    sout[...] = weights * scale[..., 0]


def _pipeline(pos, q, cs, hw, qout, sout, *, softmax_scale):
    """RoPE kernel plus final scale folding; existing projection stays intact."""
    t = q.shape[0]

    def spec(shape):
        return pl.BlockSpec(shape, lambda i: (0,) * len(shape))

    specs = [spec((t, 64, 128)), spec((t, 64)), spec((t, 64))]
    outs = [spec((t, 64, 128)), spec((t, 64))]
    allocations = (
        pltpu.BufferedRef.input(specs[0], jnp.bfloat16),
        rope.CosSinRef.create(
            spec=specs[1], dtype=jnp.float32, buffer_count=2, tile_n=t
        ),
        pltpu.BufferedRef.input(specs[2], hw.dtype),
        pltpu.BufferedRef.output(outs[0], jnp.float8_e4m3fn),
        pltpu.BufferedRef.output(outs[1], jnp.float32),
    )
    pipe = pltpu.emit_pipeline(
        functools.partial(_body, softmax_scale=softmax_scale),
        grid=(1,),
        in_specs=specs,
        out_specs=outs,
    )

    @pl.with_scoped(allocations=allocations)
    def run(allocations):
        pipe(q, (cs, pos), hw, qout, sout, allocations=allocations)

    run()


def rope_quant_fold_weights(
    q: jax.Array,
    pos: jax.Array,
    cs: jax.Array,
    head_weights: jax.Array,
    *,
    softmax_scale: float = 128**-0.5,
) -> tuple[jax.Array, jax.Array]:
    """Rotate/quantize H64 D128 queries and fold scales in the same launch."""
    return pl.pallas_call(
        functools.partial(_pipeline, softmax_scale=softmax_scale),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * 3,
            out_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * 2,
        ),
        out_shape=(
            jax.ShapeDtypeStruct((q.shape[0], 64, 128), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((q.shape[0], 64), jnp.float32),
        ),
        # C8 single-token H64/D128 used 51 KiB scoped VMEM. Leave headroom
        # while allowing subsequent compressor weights to remain resident.
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=(
                1024 * 1024 if q.shape == (1, 64, 128) else 100 * 1024 * 1024
            )
        ),
        name="indexer_rope_quant_fold_weights",
    )(pos, q, cs, head_weights)
