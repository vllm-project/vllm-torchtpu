# SPDX-License-Identifier: Apache-2.0
"""Per-token activation quantize fused into an fp8 per-channel matmul.

Computes what `util.xla_quantized_matmul` computes for fp8 weights in one
kernel: each row's abs-max, its scale, the fp8 quantize, the fp8 MXU matmul
with f32 accumulation and the `x_scale * w_scale` epilogue. The XLA path runs
an abs-max reduction and a quantize pass over HBM before its dot instead.

The grid walks output column tiles. The first step copies the activation into
VMEM in row chunks and quantizes each chunk as it lands; the quantized rows
stay resident for every later tile.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.pallas.mosaic.primitives import AsyncCopyDescriptor
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.quantized_matmul.tuned_block_sizes import (
    get_device_vmem_limit,
)

# Fewest rows the kernel takes. Both paths stream the weight once; what the
# kernel saves is the activation's extra HBM passes (abs-max read, quantize
# read and write, fp8 re-read), which shrink with the rows, so smaller
# (decode) buckets keep XLA's path.
MIN_ROWS = 512
# Rows per activation copy and quantize chunk.
SUB_M = 128
# Lanes quantized per store; wider chunks spill the f32 quotients.
QUANT_LANES = 128
# Byte budgets that cap the output tile width: weight tile (K x tile_n fp8)
# and output tile (M x tile_n).
W_TILE_BYTES = 4 * 1024 * 1024
OUT_TILE_BYTES = 4 * 1024 * 1024
# Share of the VMEM limit the buffers below may take; Mosaic needs the rest.
VMEM_BUDGET_FRACTION = 0.9

LANE = 128


def _vmem_bytes(m: int, k: int, n: int, tile_n: int, itemsize: int) -> int:
    """VMEM the kernel holds at one output tile width."""
    resident = m * k * (itemsize + 1)  # activation and its fp8 copy
    resident += m * LANE * 4  # row scales, one per sublane row
    resident += 2 * n * 4  # weight scales
    tiles = 2 * k * tile_n + 2 * m * tile_n * itemsize  # double-buffered
    return resident + tiles + 2 * m * tile_n * 4  # f32 dot result, epilogue


def output_tile_n(
    m: int,
    k: int,
    n: int,
    itemsize: int,
    vmem_limit: int,
    *,
    multiple_of: int = LANE,
    extra_vmem_bytes: int = 0,
) -> int | None:
    """Widest multiple of `multiple_of` dividing n that fits the tile and VMEM
    budgets, with `extra_vmem_bytes` of the VMEM budget held by the caller."""
    cap = min(W_TILE_BYTES // k, OUT_TILE_BYTES // (m * itemsize), n)
    for tile_n in range(cap - cap % multiple_of, 0, -multiple_of):
        if (
            n % tile_n == 0
            and _vmem_bytes(m, k, n, tile_n, itemsize) + extra_vmem_bytes
            <= VMEM_BUDGET_FRACTION * vmem_limit
        ):
            return tile_n
    return None


def applies(x: jax.Array, w_q: jax.Array, w_scale: jax.Array) -> bool:
    """Whether `fused_quantize_matmul` takes these operands on this chip."""
    m, k = x.shape
    n = w_q.shape[1]
    return (
        w_q.dtype == jnp.float8_e4m3fn
        and x.dtype == jnp.bfloat16
        and w_scale.ndim == 1
        and m >= MIN_ROWS
        and m % SUB_M == 0
        and k % QUANT_LANES == 0
        # The kernel's fp8 x fp8 dot needs an MXU that multiplies fp8.
        and pltpu.get_tpu_info().fp8_ops_per_second > 0
        and output_tile_n(m, k, n, x.dtype.itemsize, get_device_vmem_limit())
        is not None
    )


def _row_scale(x_ref: jax.Ref, rows: pl.Slice, quant_dtype: jnp.dtype):
    """`util.quantize_block`'s per-row scale, as XLA compiles it.

    XLA folds `abs_max / dtype_max` into a multiply by the f32 reciprocal
    constant.
    """
    dtype_max = np.float32(jnp.finfo(quant_dtype).max)
    abs_max = jnp.max(jnp.abs(x_ref[rows, :]), axis=1, keepdims=True)
    scale = abs_max.astype(jnp.float32) * (np.float32(1.0) / dtype_max)
    return jnp.where(scale == 0, jnp.float32(1.0), scale)


def _quantize_rows(x_ref: jax.Ref, x_q_ref: jax.Ref, scale: jax.Array, rows: pl.Slice):
    """`util.quantize_block`'s divide, clip and cast; the divide stays one."""
    info = jnp.finfo(x_q_ref.dtype)
    for c in range(x_ref.shape[1] // QUANT_LANES):
        lanes = pl.ds(c * QUANT_LANES, QUANT_LANES)
        x = x_ref[rows, lanes].astype(jnp.float32)
        x_q_ref[rows, lanes] = jnp.clip(
            x / scale, np.float32(info.min), np.float32(info.max)
        ).astype(x_q_ref.dtype)


def start_activation_copies(
    x_hbm_ref: jax.Ref,  # (M, K), HBM
    x_ref: jax.Ref,  # (M, K), VMEM
    sems: jax.Ref,  # (M // SUB_M,) DMA
) -> list[AsyncCopyDescriptor]:
    """Starts copying the activation into VMEM, one DMA per SUB_M rows."""
    copies = [
        pltpu.make_async_copy(
            x_hbm_ref.at[pl.ds(s * SUB_M, SUB_M)],
            x_ref.at[pl.ds(s * SUB_M, SUB_M)],
            sems.at[s],
        )
        for s in range(x_ref.shape[0] // SUB_M)
    ]
    for copy in copies:
        copy.start()
    return copies


def quantize_activation(
    x_ref: jax.Ref,  # (M, K)
    x_q_ref: jax.Ref,  # (M, K) quantized
    x_scale_ref: jax.Ref,  # (M, 1) f32
    copies: list[AsyncCopyDescriptor],
) -> None:
    """`util.quantize_tensor` of the activation `start_activation_copies` is
    copying in, each chunk quantized as it lands."""
    num_sub = len(copies)
    # Chunk s+1's row scales are reduced while chunk s is cast, which
    # hides the cross-lane reduction's latency.
    chunks = [pl.ds(s * SUB_M, SUB_M) for s in range(num_sub)]
    copies[0].wait()
    scales = [_row_scale(x_ref, chunks[0], x_q_ref.dtype)]
    for s in range(num_sub):
        if s + 1 < num_sub:
            copies[s + 1].wait()
            scales.append(_row_scale(x_ref, chunks[s + 1], x_q_ref.dtype))
        _quantize_rows(x_ref, x_q_ref, scales[s], chunks[s])
        x_scale_ref[chunks[s], :] = scales[s]


def _kernel(
    x_hbm_ref: jax.Ref,  # (M, K) bf16, HBM
    w_ref: jax.Ref,  # (K, tile_n) fp8
    w_scale_ref: jax.Ref,  # (N,) f32, resident
    out_ref: jax.Ref,  # (M, tile_n)
    *refs: jax.Ref,
    return_quantized: bool,
):
    if return_quantized:
        # The (M, K) fp8 and (M, 1) f32 outputs, and their (2,) DMA semaphores.
        x_q_out_ref, x_scale_out_ref, *refs, out_sems = refs
    x_ref, x_q_ref, x_scale_ref, sems = refs

    def quantized_copies():
        return [
            pltpu.make_async_copy(x_q_ref, x_q_out_ref, out_sems.at[0]),
            pltpu.make_async_copy(x_scale_ref, x_scale_out_ref, out_sems.at[1]),
        ]

    @pl.when(pl.program_id(0) == 0)
    def _():
        copies = start_activation_copies(x_hbm_ref, x_ref, sems)
        quantize_activation(x_ref, x_q_ref, x_scale_ref, copies)
        if return_quantized:
            # Written back under the remaining steps' matmuls.
            for copy in quantized_copies():
                copy.start()

    acc = jax.lax.dot_general(
        x_q_ref[...],
        w_ref[...],
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    acc = acc * x_scale_ref[...]
    tile_n = out_ref.shape[1]
    cols = pl.ds(pl.multiple_of(pl.program_id(0) * tile_n, LANE), tile_n)
    acc = acc * w_scale_ref[cols][None, :]
    out_ref[...] = acc.astype(out_ref.dtype)

    if return_quantized:

        @pl.when(pl.program_id(0) == pl.num_programs(0) - 1)
        def _():
            for copy in quantized_copies():
                copy.wait()


@functools.partial(jax.jit, static_argnames=("return_quantized",))
def fused_quantize_matmul(
    x: jax.Array,  # [M, K] bf16
    w_q: jax.Array,  # [K, N] fp8
    w_scale: jax.Array,  # [N]
    *,
    return_quantized: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array, jax.Array]:
    """`util.xla_quantized_matmul(x, w_q, w_scale)` where `applies` holds.

    With `return_quantized`, also returns `util.quantize_tensor(x, w_q.dtype)`
    -- the fp8 activation and its (M, 1) row scales -- so another matmul of
    the same activation can skip its quantize.
    """
    assert applies(x, w_q, w_scale), (x.shape, w_q.shape, w_scale.shape)
    m, k = x.shape
    n = w_q.shape[1]
    vmem_limit = get_device_vmem_limit()
    tile_n = output_tile_n(m, k, n, x.dtype.itemsize, vmem_limit)
    assert tile_n is not None
    w_scale = w_scale.astype(jnp.float32)
    out_shape = [jax.ShapeDtypeStruct((m, n), x.dtype)]
    out_specs = [pl.BlockSpec((m, tile_n), lambda j: (0, j))]
    scratch_shapes = [
        pltpu.VMEM((m, k), x.dtype),
        pltpu.VMEM((m, k), w_q.dtype),
        pltpu.VMEM((m, 1), jnp.float32),
        pltpu.SemaphoreType.DMA((m // SUB_M,)),
    ]
    if return_quantized:
        out_shape += [
            jax.ShapeDtypeStruct((m, k), w_q.dtype),
            jax.ShapeDtypeStruct((m, 1), jnp.float32),
        ]
        out_specs += [pl.BlockSpec(memory_space=pltpu.HBM)] * 2
        scratch_shapes.append(pltpu.SemaphoreType.DMA((2,)))
    outs = pl.pallas_call(
        functools.partial(_kernel, return_quantized=return_quantized),
        out_shape=out_shape,
        grid=(n // tile_n,),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec((k, tile_n), lambda j: (0, j)),
            # Whole: a rank-1 block must be a power of two or 1024-aligned.
            pl.BlockSpec((n,), lambda j: (0,)),
        ],
        out_specs=out_specs,
        scratch_shapes=scratch_shapes,
        compiler_params=pltpu.CompilerParams(
            # Steps after the first read the activation the first quantized.
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit,
        ),
        name="fused_quantize_matmul",
    )(x, w_q, w_scale)
    return tuple(outs) if return_quantized else outs[0]
