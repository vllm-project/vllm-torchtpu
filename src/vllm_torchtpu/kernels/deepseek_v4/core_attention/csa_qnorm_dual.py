"""Q producer experiment: unchanged BF16 for SWA plus scale-one E4M3 for CSA."""
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as tpu
from vllm_torchtpu.kernels.deepseek_v4 import rope


def _body(x_ref, cs_ref, bf_ref, fp_ref, *, heads, rotary_dim, eps, direct_fp32):
    x = x_ref[...].reshape(x_ref.shape[0], heads, 512).astype(jnp.float32)
    y = x * jax.lax.rsqrt(jnp.mean(x*x, axis=-1, keepdims=True) + eps)
    tail = y[..., -128:]
    cos, sin = rope._cos_sin_lanes(cs_ref[...], rotary_dim=rotary_dim, inverse=False)
    tail = tail*cos[:, None, :] + rope._rotate_gptj(tail)*sin[:, None, :]
    y = jnp.concatenate([y[..., :-128], tail], axis=-1)
    bf = y.astype(jnp.bfloat16)
    bf_ref[...] = bf
    # Default preserves the validated BF16 -> FP8 rounding boundary.
    fp_ref[...] = jnp.clip(y if direct_fp32 else bf.astype(jnp.float32),
                          -448, 448).astype(jnp.float8_e4m3fn)


def _kernel(pos_ref, x_ref, cs_ref, bf_ref, fp_ref, *, tokens, heads, tile_n,
            rotary_dim, eps, direct_fp32):
    ins = pl.BlockSpec((tile_n, heads*512), lambda i: (i, 0), memory_space=tpu.VMEM)
    outs = pl.BlockSpec((tile_n, heads, 512), lambda i: (i, 0, 0), memory_space=tpu.VMEM)
    css = pl.BlockSpec((tile_n, rotary_dim), lambda i: (i, 0), memory_space=tpu.VMEM)
    allocs = (tpu.BufferedRef.input(ins, x_ref.dtype, buffer_count=2),
              rope.CosSinRef.create(spec=css, dtype=cs_ref.dtype, buffer_count=2, tile_n=tile_n),
              tpu.BufferedRef.output(outs, jnp.bfloat16, buffer_count=2),
              tpu.BufferedRef.output(outs, jnp.float8_e4m3fn, buffer_count=2))
    pipeline = tpu.emit_pipeline(
        functools.partial(_body, heads=heads, rotary_dim=rotary_dim, eps=eps,
                          direct_fp32=direct_fp32),
        grid=(tokens//tile_n,), in_specs=[ins, css], out_specs=[outs, outs])

    @pl.with_scoped(allocations=allocs)
    def run(allocations):
        pipeline(x_ref, (cs_ref, pos_ref), bf_ref, fp_ref, allocations=allocations)
    run()


@functools.partial(jax.jit, static_argnames=('eps', 'direct_fp32'))
def qnorm_rope_dual(x, positions, cos_sin_cache, *, eps=1e-6, direct_fp32=False):
    if (x.ndim != 3 or x.shape[-1] != 512 or x.dtype != jnp.bfloat16
            or positions.shape != x.shape[:1] or x.shape[0] == 0
            or cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != 64
            or cos_sin_cache.dtype != jnp.float32):
        raise ValueError('Expected BF16 [tokens,heads,512], positions and FP32 RoPE64')
    tokens, heads, _ = x.shape
    tile_n = rope.largest_divisor(tokens, cap=64)
    return pl.pallas_call(
        functools.partial(_kernel, tokens=tokens, heads=heads, tile_n=tile_n,
                          rotary_dim=64, eps=eps, direct_fp32=direct_fp32),
        out_shape=(jax.ShapeDtypeStruct(x.shape, jnp.bfloat16),
                   jax.ShapeDtypeStruct(x.shape, jnp.float8_e4m3fn)),
        grid_spec=tpu.PrefetchScalarGridSpec(num_scalar_prefetch=1,
            in_specs=(pl.BlockSpec(memory_space=tpu.HBM), pl.BlockSpec(memory_space=tpu.HBM)),
            out_specs=(pl.BlockSpec(memory_space=tpu.HBM), pl.BlockSpec(memory_space=tpu.HBM))),
        compiler_params=tpu.CompilerParams(vmem_limit_bytes=rope.DEFAULT_VMEM_LIMIT_BYTES,
                                           disable_bounds_checks=True),
        name='csa_qnorm_dual_direct' if direct_fp32 else 'csa_qnorm_dual_bf16_boundary',
    )(positions, x.reshape(tokens, heads*512), cos_sin_cache)
