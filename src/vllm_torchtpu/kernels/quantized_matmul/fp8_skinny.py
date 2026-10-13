# SPDX-License-Identifier: Apache-2.0
"""FP8 dense matmul for skinny activations (decode, M <= 16), weight streamed from HBM.

At decode a per-channel FP8 linear is a weight stream: 16 rows against a 6-34 MB weight.
XLA stages the whole weight into VMEM before the MXU starts on several of these sites
(or streams it in a few windows that each pay a fixed cost), so the call takes about
twice the weight's read time. This kernel keeps the weight in HBM and walks it through a
small VMEM ring with several DMAs in flight, in one grid step, computing each tile as it
lands.

Numerics: the activation is quantized in-kernel with `util.quantize_block`'s formula as
XLA compiles it -- scale = f32(amax|x|) * f32(1/fp8_max), scale == 0 -> 1,
x_q = fp8(clip(f32(x) * (1/scale))) -- bit-identical x_q and scale. x_q is then held as
bf16 (every e4m3 value is exact in bf16) and fed to a bf16 x fp8 MXU dot, whose f32
accumulation is exact. XLA's fp8 x fp8 dot accumulates below f32 precision (0.4-0.6%
from f64 at these shapes), so outputs differ from `xla_quantized_matmul` and sit closer
to the f64 product of the same quantized operands.

Tilings, both NN on a K-major `[K, N]` weight with N % 128 == 0 (a weight whose N is
not gets a zero-padded copy at load, `padded_n`):
  'n': tiles W[:, j*t:(j+1)*t] (strided windows); each tile's output columns are
       written once.
  'k': tiles W[j*t:(j+1)*t, :] (contiguous rows); f32 accumulator in VMEM.
An N-major weight would need an NT dot, whose transposed tile is relaid through the XLU
before the push and costs about as much as the stream itself (fp8-gemm probe 04).
The ring follows tokamax `fused_moe_rs`: the prologue starts `nbuf` copies; the steady
state waits slot j, computes, and starts tile j + nbuf into the same slot. The tile loop
is unrolled, so slots and lane offsets are static.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

MAX_M = 16
LANE = 128

# Measured per site on tpu7x (GLM-5.2-FP8, TP16 q-replicate decode, cold weight)
# in write-tpu-kernels fp8-gemm probes 03-04: fastest config, ties to the smaller ring.
TUNED = {  # (k, n) -> (split, tile, nbuf)
    (6144, 2688): ("k", 1024, 3),  # q_a + kv_a (padded copy of 2624)
    (2048, 16384): ("n", 1024, 4),  # q_b
    (2048, 8192): (
        "n",
        1024,
        4,
    ),  # q_b at half the heads (scaled microbenchmark twin), not tuned
    (1024, 6144): ("n", 2048, 3),  # o_proj
    (6144, 1536): ("k", 1536, 3),  # dense gate_up
}


def pick_config(k: int, n: int) -> tuple[str, int, int]:
    if (k, n) in TUNED:
        return TUNED[(k, n)]
    for t in (2048, 1024, 512, 256, 128):
        if n % t == 0 and n // t >= 3:
            return ("n", t, 3)
    return ("n", n, 1)


def supports(m: int, w_q: jax.Array, w_s: jax.Array) -> bool:
    """True when this kernel serves the call: M <= 16, per-channel fp8, and a (K, N)
    with a measured config. Other shapes stay on XLA: an untuned shape's ring could
    exceed VMEM, and nothing says it would win."""
    return (
        m <= MAX_M
        and w_q.ndim == 2
        and w_s.ndim == 1
        and w_q.dtype == jnp.float8_e4m3fn
        and tuple(w_q.shape) in TUNED
        and w_s.shape[0] == w_q.shape[1]
    )


def _kernel(
    idx_ref,
    x_ref,
    ws_ref,
    w_hbm,
    o_ref,
    xq_ref,
    xs_ref,
    wbuf,
    sem,
    *acc,
    split,
    t,
    n_tiles,
    nbuf,
    q_dtype,
    scale_dma,
):
    # Weight slice: 0 in production (a [1, ...] view); a stack index in the benchmark.
    s = idx_ref[0]

    ws_copy = None
    if scale_dma:
        # The per-channel scale arrives as an HBM ref; copy it to VMEM here, issued
        # ahead of the weight ring so its latency hides under the first tile's.
        ws_vmem, ws_sem = acc[-2], acc[-1]
        acc = acc[:-2]
        ws_copy = pltpu.make_async_copy(ws_ref, ws_vmem, ws_sem)
        ws_copy.start()
        ws_ref = ws_vmem

    def copy(j):
        if split == "n":
            src = w_hbm.at[s, :, pl.ds(j * t, t)]
        else:
            src = w_hbm.at[s, pl.ds(j * t, t), :]
        return pltpu.make_async_copy(src, wbuf.at[j % nbuf], sem.at[j % nbuf])

    for j in range(min(nbuf, n_tiles)):
        copy(j).start()

    q_max = float(jnp.finfo(q_dtype).max)
    x = x_ref[...]
    scale = jnp.max(jnp.abs(x), axis=1, keepdims=True).astype(
        jnp.float32
    ) * jnp.float32(1.0 / q_max)
    scale = jnp.where(scale == 0, jnp.float32(1.0), scale)
    x_q = jnp.clip(
        x.astype(jnp.float32) * (jnp.float32(1.0) / scale), -q_max, q_max
    ).astype(q_dtype)
    xq_ref[...] = x_q.astype(xq_ref.dtype)
    xs_ref[...] = scale

    dn = (((1,), (0,)), ((), ()))
    for j in range(n_tiles):
        copy(j).wait()
        w = wbuf[j % nbuf]
        if split == "n":
            cols = slice(j * t, (j + 1) * t)
            y = jax.lax.dot_general(
                xq_ref[...], w, dn, preferred_element_type=jnp.float32
            )
            if scale_dma and j == 0:
                assert ws_copy is not None
                ws_copy.wait()
            o_ref[:, cols] = (y * xs_ref[...] * ws_ref[:, cols]).astype(o_ref.dtype)
        else:
            y = jax.lax.dot_general(
                xq_ref[:, j * t : (j + 1) * t],
                w,
                dn,
                preferred_element_type=jnp.float32,
            )
            if j == 0:
                acc[0][...] = y
            else:
                acc[0][...] += y
        if j + nbuf < n_tiles:
            copy(j + nbuf).start()
    if split == "k":
        if scale_dma:
            assert ws_copy is not None
            ws_copy.wait()
        o_ref[...] = (acc[0][...] * xs_ref[...] * ws_ref[...]).astype(o_ref.dtype)


@functools.cache
def _build(m, k, n, n_stack, w_dtype, out_dtype, split, t, nbuf, scale_dma=False):
    full = k if split == "k" else n
    if full % t or t % LANE:
        raise ValueError(
            f"tile {t} does not divide {split}-extent {full} in lane multiples"
        )
    n_tiles = full // t
    nbuf = max(1, min(nbuf, n_tiles))
    tile = (k, t) if split == "n" else (t, n)
    scratch = [
        pltpu.VMEM((m, k), jnp.bfloat16),
        pltpu.VMEM((m, 1), jnp.float32),
        pltpu.VMEM((nbuf, *tile), w_dtype),
        pltpu.SemaphoreType.DMA((nbuf,)),
    ]
    if split == "k":
        scratch.append(pltpu.VMEM((m, n), jnp.float32))
    if scale_dma:
        scratch += [pltpu.VMEM((1, n), jnp.float32), pltpu.SemaphoreType.DMA]
    kern = functools.partial(
        _kernel,
        split=split,
        t=t,
        n_tiles=n_tiles,
        nbuf=nbuf,
        q_dtype=w_dtype,
        scale_dma=scale_dma,
    )
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(1,),
        in_specs=[
            pl.BlockSpec((m, k), lambda g, idx: (0, 0)),
            # Scale [1, n]: a VMEM block, or with `scale_dma` an HBM ref the kernel
            # copies itself (see `fp8_skinny_matmul_stacked`).
            pl.BlockSpec(memory_space=pltpu.HBM)
            if scale_dma
            else pl.BlockSpec((1, n), lambda g, idx: (0, 0)),
            # W [s, k, n], pinned to HBM (MSA cannot move it into VMEM).
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=pl.BlockSpec((m, n), lambda g, idx: (0, 0)),
        scratch_shapes=scratch,
    )
    sd = "_sd" if scale_dma else ""
    return pl.pallas_call(
        kern,
        out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        name=f"fp8_skinny_{split}{t}_b{nbuf}{sd}_m{m}_k{k}_n{n}",
    )


def fp8_skinny_matmul(
    x: jax.Array,
    w_q: jax.Array,
    w_s: jax.Array,
    *,
    n_logical: int | None = None,
    config: tuple[str, int, int] | None = None,
    scale_dma: bool = False,
) -> jax.Array:
    """x [m<=16, k] bf16, w_q fp8 [k, n] with n % 128 == 0, w_s f32 [n]
    -> [m, n_logical or n] in x.dtype. `scale_dma`: see `fp8_skinny_matmul_stacked`."""
    return fp8_skinny_matmul_stacked(
        x,
        w_q[None],
        jnp.zeros((1,), jnp.int32),
        w_s,
        n_logical=n_logical,
        config=config,
        scale_dma=scale_dma,
    )


@functools.partial(jax.jit, static_argnums=0)
def _call_scale_in_hbm(call, idx, x, w_s, w_stack):
    # The constraint sets the scale operand's aval memory space, which pallas_call
    # lowers to the custom call's `input_memory_space_colors` (HBM = 0). It only
    # binds a pallas_call input and cannot run eagerly, hence the jit around both.
    return call(idx, x, pltpu.with_memory_space_constraint(w_s, pltpu.HBM), w_stack)


def fp8_skinny_matmul_stacked(
    x, w_stack, idx, w_s, *, n_logical=None, config=None, scale_dma=False
):
    """As `fp8_skinny_matmul`; the kernel reads the weight from `w_stack[idx[0]]`.

    `scale_dma` (TPU_FP8_SKINNY_SCALE_DMA): pass the per-channel scale as an HBM
    operand and DMA it to VMEM inside the kernel, issued with the weight ring.
    Otherwise XLA stages the scale (o_proj's f32[6144], 24 KB) into VMEM with a copy
    its memory-space assignment places one instruction before the call, exposing the
    DMA issue latency (~0.8 us) every layer. Same values, same arithmetic:
    bit-identical outputs.
    """
    m, k = x.shape
    n_stack, _, n = w_stack.shape
    split, t, nbuf = config or pick_config(k, n)
    call = _build(m, k, n, n_stack, w_stack.dtype, x.dtype, split, t, nbuf, scale_dma)
    w_s2 = w_s.reshape(1, n).astype(jnp.float32)
    if scale_dma:
        out = _call_scale_in_hbm(call, idx, x, w_s2, w_stack)
    else:
        out = call(idx, x, w_s2, w_stack)
    if n_logical is not None and n_logical != n:
        out = out[:, :n_logical]
    return out


# Sites where TPU_FP8_SKINNY_SCALE_DMA is used in the model: those whose XLA scale
# copy was measured exposed on GLM-5.2-FP8 decode (twin j-69560162 / j-dae7fb62:
# o_proj's f32[6144] copy sits one instruction before the call, ~0.8 us per layer).
# q_b's scale copy is hidden there, and standalone the in-kernel copy measured
# +0.3-0.5 us per call at (2048, 16384), so it stays as today. A site's N must be a
# multiple of 1024: the scale's [n] -> [1, n] reshape is then a bitcast of the f32
# T(1024) layout and the HBM operand is the parameter itself (at gate_up's 1536 the
# reshape is a real op whose output XLA would write to HBM).
SCALE_DMA_SITES = {(1024, 6144)}  # o_proj


def scale_dma_applies(k: int, n: int) -> bool:
    return (k, n) in SCALE_DMA_SITES


def padded_n(n: int, multiple: int = LANE) -> int:
    """The output width a weight is padded to at load (zero outputs, unit scale) so
    the kernel tiles it exactly."""
    return -(-n // multiple) * multiple
