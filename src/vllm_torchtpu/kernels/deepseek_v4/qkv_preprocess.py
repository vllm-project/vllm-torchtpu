# SPDX-License-Identifier: Apache-2.0
"""Joint Q RMSNorm/RoPE and KV RoPE with one shared position gather."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.utils import get_device_name

from . import rope


def _kernel(pos, q, kv, cs, qo, ko, raw, sem, *, eps):
    for i in range(q.shape[0]):
        pltpu.make_async_copy(cs.at[pos[i]], raw.at[i], sem).start()
    x = q[...].astype(jnp.float32)
    normalized = x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    pltpu.make_async_copy(raw, raw, sem).wait()
    c, s = rope._cos_sin_lanes(raw[...], rotary_dim=cs.shape[1], inverse=False)
    tail = normalized[:, :, -rope.LANE :]
    tail = tail * c[:, None, :] + rope._rotate_gptj(tail) * s[:, None, :]
    qo[...] = jnp.concatenate((normalized[:, :, : -rope.LANE], tail), axis=-1).astype(
        q.dtype
    )
    x = kv[...]
    tail = x[:, -rope.LANE :].astype(jnp.float32)
    tail = (tail * c + rope._rotate_gptj(tail) * s).astype(kv.dtype)
    ko[...] = jnp.concatenate((x[:, : -rope.LANE], tail), axis=-1)


def _qkv_preprocess(q, kv, positions, cos_sin_cache, *, eps):
    assert q.shape[0] in (1, 4, 8, 16, 32, 64)
    assert q.shape[1:] == (64, 512) and kv.shape == (q.shape[0], 512)
    assert q.dtype == kv.dtype == jnp.bfloat16
    vm = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    # Reserve inputs, outputs and FP32 temporaries without claiming 100 MiB
    # for small batches. The lower budget also leaves room for DMA scheduling.
    resident_bytes = sum(x.size * x.dtype.itemsize for x in (q, kv))
    vmem_limit_bytes = min(
        rope.DEFAULT_VMEM_LIMIT_BYTES, max(16 * 1024 * 1024, 4 * resident_bytes)
    )
    return pl.pallas_call(
        functools.partial(_kernel, eps=eps),
        out_shape=(
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            jax.ShapeDtypeStruct(kv.shape, kv.dtype),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[vm, vm, hbm],
            out_specs=[vm, vm],
            scratch_shapes=[
                pltpu.VMEM((q.shape[0], cos_sin_cache.shape[1]), cos_sin_cache.dtype),
                pltpu.SemaphoreType.DMA,
            ],
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes),
        name="qkv_preprocess_joint",
    )(positions, q, kv, cos_sin_cache)


def qkv_preprocess(
    q: jax.Array,
    kv: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    *,
    eps: float,
) -> tuple[jax.Array, jax.Array]:
    """Share the position gather for measured TPU7 shapes.

    Selection is by static shape; a prefill with the same shape also uses it.
    """
    supported = (
        q.ndim == 3
        and q.shape[0] in (1, 4, 8, 16, 32, 64)
        and q.shape[1:] == (64, 512)
        and kv.shape == (q.shape[0], 512)
        and q.dtype == kv.dtype == jnp.bfloat16
        and positions.shape == (q.shape[0],)
        and positions.dtype == jnp.int32
        and cos_sin_cache.ndim == 2
        and cos_sin_cache.shape[1] == 64
        and cos_sin_cache.dtype == jnp.float32
        and get_device_name() == "TPU v7"
    )
    if supported:
        return _qkv_preprocess(q, kv, positions, cos_sin_cache, eps=eps)
    return (
        rope.qnorm_rope(q, positions, cos_sin_cache, eps=eps),
        rope.rope(kv, positions, cos_sin_cache),
    )
