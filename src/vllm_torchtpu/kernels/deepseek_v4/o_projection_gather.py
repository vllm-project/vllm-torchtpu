# SPDX-License-Identifier: Apache-2.0
"""Gather cos/sin once and keep it in VMEM through all Wo_a groups."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.utils import get_device_name

from . import o_projection, rope


def _body(x, w, scale, out, expanded):
    o_projection._kernel(
        x,
        w,
        scale,
        expanded,
        out,
        tile_t=x.shape[0],
        num_sub_t=1,
        quantize_activations=True,
        head_dim=x.shape[-1],
    )


def _kernel(pos, x, w, scale, cs, out, raw, expanded, sem, *, b, groups, rank, d):
    for i in range(b):
        pltpu.make_async_copy(cs.at[pos[i]], raw.at[i], sem).start()
    pltpu.make_async_copy(raw, raw, sem).wait()
    c, s = rope._cos_sin_lanes(raw[...], rotary_dim=cs.shape[-1], inverse=True)
    expanded[...] = jnp.concatenate((c, s), axis=-1)
    specs = [
        pl.BlockSpec((b, 8, d), lambda g: (0, g, 0)),
        pl.BlockSpec((8 * d, rank), lambda g: (0, g)),
        pl.BlockSpec((1, rank), lambda g: (0, g)),
    ]
    outs = pl.BlockSpec((b, rank), lambda g: (0, g))
    pltpu.emit_pipeline(_body, grid=(groups,), in_specs=specs, out_specs=[outs])(
        x, w, scale, out, scratches=(expanded,)
    )


def _fused(x, positions, cs, w, scale):
    b, h, d = x.shape
    assert b in (1, 4, 8, 16, 32, 64) and d == 512 and w.shape[0] == 8 * d
    groups = h // 8
    rank = w.shape[1] // groups
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        functools.partial(_kernel, b=b, groups=groups, rank=rank, d=d),
        out_shape=jax.ShapeDtypeStruct((b, w.shape[1]), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[hbm] * 4,
            out_specs=hbm,
            scratch_shapes=[
                pltpu.VMEM((b, cs.shape[1]), jnp.float32),
                pltpu.VMEM((b, 256), jnp.float32),
                pltpu.SemaphoreType.DMA,
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=o_projection.DEFAULT_VMEM_LIMIT_BYTES
        ),
        name="wo_a_inline_cos_sin_once",
    )(positions, x, w, scale.reshape(1, -1), cs)


def fused(
    x: jax.Array,
    positions: jax.Array,
    cs: jax.Array,
    w: jax.Array,
    scale: jax.Array,
    *,
    head_dim: int,
) -> jax.Array:
    """Keep rotary coefficients in VMEM across Wo_a groups for measured shapes.

    Selection is by static shape; a prefill with the same shape also uses it.
    """
    supported = (
        x.ndim == 3
        and x.shape[0] in (1, 4, 8, 16, 32, 64)
        and x.shape[1:] == (64, 512)
        and head_dim == 512
        and x.dtype == jnp.bfloat16
        and w.shape == (4096, 8192)
        and w.dtype == jnp.float8_e4m3fn
        and scale.size == w.shape[1]
        and scale.dtype == jnp.float32
        and positions.shape == (x.shape[0],)
        and positions.dtype == jnp.int32
        and cs.ndim == 2
        and cs.shape[1] == 64
        and cs.dtype == jnp.float32
        and get_device_name() == "TPU v7"
    )
    if supported:
        return _fused(x, positions, cs, w, scale)
    return o_projection.fused_reverse_rope_wo_a_projection(
        x,
        positions,
        cs,
        w,
        scale.reshape(-1),
        head_dim=head_dim,
        inverse=True,
        quantize_activations=True,
    )
