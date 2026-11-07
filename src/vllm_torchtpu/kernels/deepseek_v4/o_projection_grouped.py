# SPDX-License-Identifier: Apache-2.0
"""Experimental grouped Wo_a with a load-time contiguous weight layout.

Compared against vLLM NVIDIA nvidia/ops/o_proj.py at
f47792af7fba76eb8e8f6a564b35cfad316d39f2.
The TPU per-group quantization and BF16 boundaries remain unchanged.
No production caller is enabled. Repacking belongs at weight load time.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import o_projection, rope
from vllm_torchtpu.kernels.deepseek_v4.o_projection_tuned import WO_A_GROUP_TILES
from vllm_torchtpu.layers.core.quantization import FP8_E4M3_MAX


def pack_weight(weight: jax.Array) -> jax.Array:
    """Reorder K-major [4096,8192] into [8,4096,1024] without quantization."""
    assert weight.shape == (4096, 8192) and weight.dtype == jnp.float8_e4m3fn
    return weight.reshape(4096, 8, 1024).transpose(1, 0, 2)


def _packed_body(x, w, scale, out, expanded, *, gc, nk):
    for g in range(gc):
        xx = o_projection._rope_heads(
            x[:, g * 8 : (g + 1) * 8], expanded[...], head_dim=512
        ).reshape(1, 4096)
        amax = jnp.max(jnp.abs(xx), axis=1, keepdims=True)
        inv = (FP8_E4M3_MAX / jnp.maximum(amax, jnp.bfloat16(1e-30))).astype(
            jnp.bfloat16
        )
        lhs = (xx * inv).astype(jnp.float8_e4m3fn)
        rhs = w[g]
        acc = jax.lax.dot_general(
            lhs,
            rhs,
            (((1,), (1 if nk else 0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        out[:, g * 1024 : (g + 1) * 1024] = (
            (acc * (1.0 / inv.astype(jnp.float32)))
            * scale[:, g * 1024 : (g + 1) * 1024]
        ).astype(jnp.bfloat16)


def _packed(pos, x, w, ws, cs, out, raw, expanded, sem, *, gc, nk):
    pltpu.make_async_copy(cs.at[pos[0]], raw.at[0], sem).start()
    pltpu.make_async_copy(raw, raw, sem).wait()
    c, s = rope._cos_sin_lanes(raw[...], rotary_dim=64, inverse=True)
    expanded[...] = jnp.concatenate((c, s), axis=-1)
    pltpu.emit_pipeline(
        functools.partial(_packed_body, gc=gc, nk=nk),
        grid=(8 // gc,),
        in_specs=[
            pl.BlockSpec((1, gc * 8, 512), lambda g: (0, g, 0)),
            pl.BlockSpec(
                (gc, 1024, 4096) if nk else (gc, 4096, 1024), lambda g: (g, 0, 0)
            ),
            pl.BlockSpec((1, gc * 1024), lambda g: (0, g)),
        ],
        out_specs=pl.BlockSpec((1, gc * 1024), lambda g: (0, g)),
    )(x, w, ws, out, scratches=(expanded,))


def wo_a_grouped(
    x: jax.Array,
    positions: jax.Array,
    cs: jax.Array,
    weight: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    """Run inverse RoPE, original FP8 quantization and two Wo_a groups per tile."""
    info = pltpu.get_tpu_info()
    assert x.shape == (1, 64, 512) and x.dtype == jnp.bfloat16
    assert weight.shape == (8, 4096, 1024) and weight.dtype == jnp.float8_e4m3fn
    assert scale.shape == (8192,) and scale.dtype == jnp.float32
    assert positions.shape == (1,) and positions.dtype == jnp.int32
    assert cs.ndim == 2 and cs.shape[1] == 64 and cs.dtype == jnp.float32
    key = (info.generation, x.shape[0], *weight.shape)
    assert key in WO_A_GROUP_TILES, f"Unmeasured grouped Wo_a shape: {key}"
    gc = WO_A_GROUP_TILES[key]
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        functools.partial(_packed, gc=gc, nk=False),
        out_shape=jax.ShapeDtypeStruct((1, 8192), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[hbm] * 4,
            out_specs=hbm,
            scratch_shapes=[
                pltpu.VMEM((1, 64), jnp.float32),
                pltpu.VMEM((1, 256), jnp.float32),
                pltpu.SemaphoreType.DMA,
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=min(
                info.vmem_capacity_bytes, o_projection.DEFAULT_VMEM_LIMIT_BYTES
            )
        ),
        name="wo_a_packed_fused_g2_kn",
    )(positions, x, weight, scale.reshape(1, 8192), cs)
