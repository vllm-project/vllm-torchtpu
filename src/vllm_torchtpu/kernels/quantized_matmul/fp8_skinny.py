# SPDX-License-Identifier: Apache-2.0
"""FP8 dense matmul for skinny activations (decode, M <= MAX_M), weight streamed
from HBM.

At decode a per-channel FP8 linear is a weight stream: a few rows against a 5-15 MB
weight. This kernel keeps the weight in HBM and walks it through a VMEM ring with
several DMAs in flight, in one grid step, computing each tile as it lands. The
activation quantization runs inside the kernel instead of as separate XLA fusions.

Numerics: the activation is quantized with `util.quantize_block`'s formula as XLA
compiles it -- scale = f32(amax|x|) * f32(1/fp8_max), scale == 0 -> 1,
x_q = fp8(clip(f32(x) * (1/scale))) -- giving the same x_q and scale. x_q is held as
bf16 (every e4m3 value is exact in bf16) and fed to a bf16 x fp8 dot with f32
accumulation, while XLA's fp8 x fp8 dot accumulates below f32 precision. Outputs
therefore differ from `xla_quantized_matmul` and sit closer to the f64 product of the
same quantized operands.

Tilings, both on a K-major `[K, N]` weight with N % 128 == 0:
  'n': tiles W[:, j*t:(j+1)*t] (strided windows); each tile's output columns are
       written once.
  'k': tiles W[j*t:(j+1)*t, :] (contiguous rows); f32 accumulator in VMEM.
The prologue starts `nbuf` tile copies (each as `ndma` row-band DMAs); the steady
state waits slot j, computes, and starts tile j + nbuf into the same slot. The tile
loop is unrolled, so slots and offsets are static.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

MAX_M = 128
# Rows above this use the *_LARGE_M tables (decode with several requests x 7 tokens).
SMALL_M = 32
LANE = 128
FP8_SUBLANE_TILE = 32  # rows per fp8 (32, 128) HBM tile

Config = tuple[str, int, int, int]  # (split, tile, nbuf, ndma)

# Fastest config per DeepSeek-V4 TP8 decode shape on tpu7x at M=16, all 8 cores
# streaming at once, 16 distinct weights per chain. `TUNED`: the weight operand is
# left to XLA's memory-space assignment; `TUNED_HBM`: constrained to HBM (`hbm_pin`).
TUNED: dict[tuple[int, int], Config] = {
    (7168, 2048): ("k", 1792, 4, 1),  # attn.fused_wqa_wkv
    (1536, 8192): ("k", 256, 2, 1),  # attn.wq_b, attn.indexer.wq_b (TP8)
    (2048, 7168): ("k", 512, 2, 1),  # attn.wo_b (TP8)
    (7168, 768): ("k", 3584, 2, 1),  # shared_experts.gate_up_proj (TP8)
    # TP4 (c16) per-rank shapes: not tuned unpinned; same as TUNED_HBM.
    (1536, 16384): ("k", 128, 8, 4),  # attn.wq_b (TP4)
    (4096, 7168): ("k", 128, 8, 1),  # attn.wo_b (TP4)
    (7168, 1536): ("k", 512, 8, 4),  # shared_experts.gate_up_proj (TP4)
}
# HBM-pinned: fastest config per shape on tpu7x, one core, chain of 8 distinct
# weights, M = 8/16 (fp8-linears close-gaps sweep, j-325ac3c7). Within ~1% of each
# other across the top configs; all stream at the same rate.
TUNED_HBM: dict[tuple[int, int], Config] = {
    (7168, 2048): ("k", 512, 6, 1),
    (1536, 8192): ("k", 128, 6, 2),
    (2048, 7168): ("k", 256, 4, 2),
    (7168, 768): ("k", 1024, 6, 1),
    (1536, 16384): ("k", 128, 8, 4),
    (4096, 7168): ("k", 128, 8, 1),
    (7168, 1536): ("k", 512, 8, 4),
}
# HBM-pinned, 32 < M <= 128 (sweep j-b29381de / j-af7fb5fa, M = 64 / 128). At these
# rows the per-tile dot is no longer free and the n-split wins on some shapes.
TUNED_HBM_LARGE_M: dict[tuple[int, int], Config] = {
    (7168, 2048): ("k", 512, 6, 2),
    (1536, 8192): ("n", 1024, 6, 1),
    (2048, 7168): ("n", 1024, 6, 2),
    (7168, 768): ("k", 1792, 4, 2),
    (1536, 16384): ("n", 1024, 4, 2),
    (4096, 7168): ("n", 512, 4, 1),
    (7168, 1536): ("k", 512, 8, 2),
}
assert TUNED.keys() == TUNED_HBM.keys() == TUNED_HBM_LARGE_M.keys()


def pick_config(m: int, k: int, n: int, hbm_pin: bool) -> Config:
    if m > SMALL_M:
        # Only measured HBM-pinned; the unpinned path uses the same tiling.
        return TUNED_HBM_LARGE_M[(k, n)]
    return (TUNED_HBM if hbm_pin else TUNED)[(k, n)]


def supports(m: int, w_q: jax.Array, w_s: jax.Array) -> bool:
    """True when this kernel serves the call: M <= MAX_M, per-channel fp8, and a
    (K, N) with a measured config. Other shapes stay on XLA: nothing says an
    unmeasured shape would win."""
    return (
        m <= MAX_M
        and w_q.ndim == 2
        and w_s.ndim == 1
        and w_q.dtype == jnp.float8_e4m3fn
        and tuple(w_q.shape) in TUNED
        and w_s.shape[0] == w_q.shape[1]
    )


def _kernel(
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
    ndma,
    q_dtype,
    scale_dma=False,
):
    rows = wbuf.shape[1] // ndma  # each tile moves as `ndma` row-band DMAs
    ws_copy = None
    if scale_dma:
        # The per-channel scale arrives as an HBM ref: copy it here, issued ahead of
        # the weight ring, so its latency hides under the first tiles' instead of
        # sitting in an XLA copy right before the call.
        ws_vmem, ws_sem = acc[-2], acc[-1]
        acc = acc[:-2]
        ws_copy = pltpu.make_async_copy(ws_ref, ws_vmem, ws_sem)
        ws_copy.start()
        ws_ref = ws_vmem

    def copies(j):
        slot = j % nbuf
        out = []
        for r in range(ndma):
            if split == "n":
                src = w_hbm.at[pl.ds(r * rows, rows), pl.ds(j * t, t)]
            else:
                src = w_hbm.at[pl.ds(j * t + r * rows, rows), :]
            dst = wbuf.at[slot, pl.ds(r * rows, rows), :]
            out.append(pltpu.make_async_copy(src, dst, sem.at[slot]))
        return out

    for j in range(min(nbuf, n_tiles)):
        for c in copies(j):
            c.start()

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
        for c in copies(j):
            c.wait()
        w = wbuf[j % nbuf]
        if split == "n":
            cols = slice(j * t, (j + 1) * t)
            y = jax.lax.dot_general(
                xq_ref[...], w, dn, preferred_element_type=jnp.float32
            )
            if ws_copy is not None and j == 0:
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
            for c in copies(j + nbuf):
                c.start()
    if split == "k":
        if ws_copy is not None:
            ws_copy.wait()
        o_ref[...] = (acc[0][...] * xs_ref[...] * ws_ref[...]).astype(o_ref.dtype)


def _vmem_limit(m, k, n, tile, nbuf, split):
    """Scoped-VMEM budget: the weight ring plus double-buffered x / scale / out
    blocks, the bf16 x_q and the 'k' accumulator, with 4 MiB headroom."""
    ring = nbuf * tile[0] * tile[1]
    blocks = 2 * (m * k * 2 + n * 4 + m * n * 4)
    extra = m * k * 2 + (m * n * 4 if split == "k" else 0)
    return min(ring + blocks + extra + (4 << 20), 60 << 20)


@functools.cache
def _build(m, k, n, w_dtype, out_dtype, split, t, nbuf, ndma, scale_dma=False):
    full = k if split == "k" else n
    if full % t or t % LANE:
        raise ValueError(
            f"tile {t} does not divide {split}-extent {full} in lane multiples"
        )
    tile_rows = k if split == "n" else t
    if tile_rows % ndma or (tile_rows // ndma) % FP8_SUBLANE_TILE:
        raise ValueError(f"{ndma} DMAs per tile do not split {tile_rows} rows in 32s")
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
        scratch += [pltpu.VMEM((1, n), jnp.float32), pltpu.SemaphoreType.DMA(())]
    kern = functools.partial(
        _kernel,
        split=split,
        t=t,
        n_tiles=n_tiles,
        nbuf=nbuf,
        ndma=ndma,
        q_dtype=w_dtype,
        scale_dma=scale_dma,
    )
    return pl.pallas_call(
        kern,
        out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
        grid=(1,),
        in_specs=[
            pl.BlockSpec((m, k), lambda g: (0, 0)),
            pl.BlockSpec(memory_space=pltpu.HBM)
            if scale_dma
            else pl.BlockSpec((1, n), lambda g: (0, 0)),
            # The weight is an HBM ref inside the kernel. This alone does not stop
            # XLA staging the operand in VMEM first; see `hbm_pin`.
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=pl.BlockSpec((m, n), lambda g: (0, 0)),
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=_vmem_limit(m, k, n, tile, nbuf, split),
        ),
        name=f"fp8_skinny_{split}{t}_b{nbuf}_d{ndma}{'_sd' if scale_dma else ''}_m{m}_k{k}_n{n}",
    )


@functools.partial(jax.jit, static_argnums=(0, 4))
def _call_hbm_pinned(call, x, w_s, w_q, scale_in_hbm=False):
    # The constraint sets the operand aval's memory space, which pallas_call lowers
    # to the custom call's input memory space. It only binds a pallas_call input
    # and cannot run eagerly, hence the jit around both.
    if scale_in_hbm:
        w_s = pltpu.with_memory_space_constraint(w_s, pltpu.HBM)
    return call(x, w_s, pltpu.with_memory_space_constraint(w_q, pltpu.HBM))


def fp8_skinny_matmul(
    x: jax.Array,
    w_q: jax.Array,
    w_s: jax.Array,
    *,
    config: Config | None = None,
    hbm_pin: bool = False,
    scale_dma: bool = False,
) -> jax.Array:
    """x [m, k] bf16, w_q fp8 [k, n] with n % 128 == 0, w_s f32 [n] -> [m, n] in
    x.dtype. `config` defaults to the measured table for (k, n).

    `hbm_pin` constrains the weight operand to HBM. The in_spec's memory space only
    types the ref inside the kernel; XLA's memory-space assignment reads the
    operand's aval, so without the constraint it may stage the weight into VMEM
    with a copy before the call. Same kernel, same arithmetic: bit-identical.

    `scale_dma` (with `hbm_pin`, N % 1024 == 0 so the [1, N] view of the 1-D scale
    is a bitcast): the scale is an HBM operand copied in-kernel with the weight
    ring, instead of a VMEM operand XLA stages with a copy just before the call
    (~0.7 us exposed per call in the DSv4 decode twin). Bit-identical.
    """
    m, k = x.shape
    n = w_q.shape[1]
    if config is None:
        config = pick_config(m, k, n, hbm_pin)
    split, t, nbuf, ndma = config
    scale_dma = bool(scale_dma and hbm_pin and n % 1024 == 0)
    call = _build(m, k, n, w_q.dtype, x.dtype, split, t, nbuf, ndma, scale_dma)
    w_s2 = w_s.reshape(1, n).astype(jnp.float32)
    if hbm_pin:
        return _call_hbm_pinned(call, x, w_s2, w_q, scale_dma)
    return call(x, w_s2, w_q)
