# SPDX-License-Identifier: Apache-2.0
"""Experimental fused output projection retaining intermediates in VMEM.

Based on the frozen grouped Wo_a and quantized MXU Wo_b candidates.
Preserves the BF16 boundary and existing quantization; no production caller.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import o_projection_grouped as grouped
from vllm_torchtpu.kernels.deepseek_v4 import rope
from vllm_torchtpu.kernels.deepseek_v4.o_projection_tuned import WO_OUTPUT_TILES
from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor


def _b_body(w, ws, out, lhs, xs):
    acc = jax.lax.dot_general(
        lhs[...],
        w[...],
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )[:1]
    out[...] = ((acc * jnp.tile(xs[...], (1, w.shape[1] // 128))) * ws[...]).astype(
        jnp.bfloat16
    )


def _b_pipeline(w, ws, out, z, lhs, xs, *, tile):
    q, scale = quantize_tensor(z[...], w.dtype)
    lhs[...] = jnp.broadcast_to(q.astype(jnp.bfloat16), lhs.shape)
    xs[...] = jnp.broadcast_to(scale, xs.shape)
    pltpu.emit_pipeline(
        _b_body,
        grid=(4096 // tile,),
        in_specs=[
            pl.BlockSpec((8192, tile), lambda i: (0, i)),
            pl.BlockSpec((1, tile), lambda i: (0, i)),
        ],
        out_specs=pl.BlockSpec((1, tile), lambda i: (0, i)),
    )(w, ws, out, scratches=(lhs, xs))


def _a_body(x, w, ws, z, expanded):
    # VMEM output tile is a view of the full BF16 intermediate.
    grouped._packed_body(x, w, ws, z, expanded, gc=2, nk=False)


def _fused(pos, x, wa, sa, cs, wb, sb, out, raw, expanded, z, lhs, xs, sem, *, tile):
    copy = pltpu.make_async_copy(cs.at[pos[0]], raw.at[0], sem)
    copy.start()
    copy.wait()
    c, s = rope._cos_sin_lanes(raw[...], rotary_dim=64, inverse=True)
    expanded[...] = jnp.concatenate((c, s), axis=-1)
    pltpu.emit_pipeline(
        _a_body,
        grid=(4,),
        in_specs=[
            pl.BlockSpec((1, 16, 512), lambda g: (0, g, 0)),
            pl.BlockSpec((2, 4096, 1024), lambda g: (g, 0, 0)),
            pl.BlockSpec((1, 2048), lambda g: (0, g)),
        ],
        out_specs=pl.BlockSpec((1, 2048), lambda g: (0, g), memory_space=pltpu.VMEM),
    )(x, wa, sa, z, scratches=(expanded,))
    _b_pipeline(wb, sb, out, z, lhs, xs, tile=tile)


def output_projection(
    x: jax.Array,
    positions: jax.Array,
    cs: jax.Array,
    wa: jax.Array,
    sa: jax.Array,
    wb: jax.Array,
    sb: jax.Array,
) -> jax.Array:
    """Fuse inverse RoPE, grouped Wo_a and quantized Wo_b for TPU7 B1."""
    info = pltpu.get_tpu_info()
    assert info.generation == 7
    assert x.shape == (1, 64, 512) and x.dtype == jnp.bfloat16
    assert wa.shape == (8, 4096, 1024) and wa.dtype == jnp.float8_e4m3fn
    assert sa.shape == (8192,) and sa.dtype == jnp.float32
    assert wb.shape == (8192, 4096) and wb.dtype == jnp.float8_e4m3fn
    assert sb.shape == (4096,) and sb.dtype == jnp.float32
    assert positions.shape == (1,) and positions.dtype == jnp.int32
    assert cs.ndim == 2 and cs.shape[1] == 64 and cs.dtype == jnp.float32
    key = (info.generation, x.shape[0], *wb.shape)
    assert key in WO_OUTPUT_TILES, f"Unmeasured output projection shape: {key}"
    tile = WO_OUTPUT_TILES[key]
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        functools.partial(_fused, tile=tile),
        out_shape=jax.ShapeDtypeStruct((1, 4096), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[hbm] * 6,
            out_specs=hbm,
            scratch_shapes=[
                pltpu.VMEM((1, 64), jnp.float32),
                pltpu.VMEM((1, 256), jnp.float32),
                pltpu.VMEM((1, 8192), jnp.bfloat16),
                pltpu.VMEM((info.num_sublanes, 8192), jnp.bfloat16),
                pltpu.VMEM((1, 128), jnp.float32),
                pltpu.SemaphoreType.DMA,
            ],
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=info.vmem_capacity_bytes),
        name=f"output_projection_resident_n{tile}",
    )(positions, x, wa, sa.reshape(1, 8192), cs, wb, sb.reshape(1, 4096))
