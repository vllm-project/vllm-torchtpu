# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""One-pass (fused add +) RMSNorm over row tiles (Pallas).

XLA lowers ``vllm.ir.ops.fused_add_rms_norm`` at prefill shapes as two
fusions: a row reduction (which also writes the new residual) and an apply
pass that re-reads and re-adds both inputs. With the full hidden row resident
in VMEM the whole op is one pass: read ``x`` and ``residual`` once, write the
new residual and the normalized output once.

Numerics follow ``vllm/ir/ops/layernorm.py`` step for step:

    s        = f32(x) + f32(residual)        # unrounded
    residual = bf16(s)
    y        = bf16(bf16(s * rsqrt(mean(s^2) + eps)) * weight)

The only freedom is the f32 summation order of ``mean(s^2)``.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Rows per grid step. One [rows, 6144] bf16 tile is 3 MiB at 256 rows; measured
# 74 us vs 81 us at 128 rows for 4096 x 6144 on v7x (j-97d7757a).
DEFAULT_BLOCK_ROWS = 256

# Rows per slice of the slice bodies: one fp8 vreg holds 32 rows, one bf16 vreg 16.
SLICE_ROWS = 32
# Columns per group inside a slice: four lane blocks, one byte-transpose group.
GROUP_COLS = 512


def _kernel(*refs, eps: float, has_residual: bool):
    if has_residual:
        x_ref, r_ref, w_ref, y_ref, ro_ref = refs
        s = x_ref[...].astype(jnp.float32) + r_ref[...].astype(jnp.float32)
        ro_ref[...] = s.astype(ro_ref.dtype)
    else:
        x_ref, w_ref, y_ref = refs
        s = x_ref[...].astype(jnp.float32)
    var = jnp.mean(s * s, axis=-1, keepdims=True)
    normed = (s * jax.lax.rsqrt(var + eps)).astype(y_ref.dtype)
    y_ref[...] = (normed.astype(jnp.float32) * w_ref[...].astype(jnp.float32)).astype(
        y_ref.dtype
    )


def _for_each_slice(tile_rows: int, slice_rows: int, body) -> None:
    """Calls ``body(start)`` for the first row of each slice of the tile.

    Unrolled, so the scheduler overlaps one slice's row reductions with the
    next slice's loads and arithmetic. A ``fori_loop`` stalls at every
    reduction, and its dynamic row offsets do not lower for the MoE body's
    strided stores."""
    for start in range(0, tile_rows, slice_rows):
        body(start)


def _column_groups(hidden: int) -> list[slice]:
    return [slice(c, c + GROUP_COLS) for c in range(0, hidden, GROUP_COLS)]


def _lane_blocks(a):
    return [a[:, c : c + 128] for c in range(0, a.shape[1], 128)]


def _slice_mean_square(x_ref, r_ref, ro_ref, sum_ref, rows):
    """Writes the unrounded ``s`` of one slice to ``sum_ref`` (and ``bf16(s)``
    to ``ro_ref``) one column group at a time; returns ``mean(s^2)``. The
    squares are added lane block by lane block from the left, then across
    lanes: the order Mosaic gives the whole-tile bodies' row reduction."""
    acc = None
    for cols in _column_groups(x_ref.shape[1]):
        s = x_ref[rows, cols].astype(jnp.float32)
        if r_ref is not None:
            s = s + r_ref[rows, cols].astype(jnp.float32)
            ro_ref[rows, cols] = s.astype(ro_ref.dtype)
            sum_ref[:, cols] = s
        for block in _lane_blocks(s * s):
            acc = block if acc is None else acc + block
    return jnp.sum(acc, axis=-1, keepdims=True) / jnp.float32(x_ref.shape[1])


def _slice_norm(x_ref, sum_ref, w_ref, rows, cols, inv_rms, dtype):
    """``y`` of one column group of a slice; ``s`` comes from ``sum_ref``, or
    from ``x`` itself without a residual."""
    if sum_ref is None:
        s = x_ref[rows, cols].astype(jnp.float32)
    else:
        s = sum_ref[:, cols]
    normed = (s * inv_rms).astype(dtype)
    return (normed.astype(jnp.float32) * w_ref[:, cols].astype(jnp.float32)).astype(
        dtype
    )


def _row_peak(peak, a):
    """``peak`` updated with the lane blocks of ``a``; max is exact, so any
    order gives the whole-row max."""
    for block in _lane_blocks(a):
        peak = block if peak is None else jnp.maximum(peak, block)
    return peak


def _sliced_kernel(*refs, eps: float, has_residual: bool, slice_rows: int):
    """``_kernel`` computed one ``slice_rows`` slice of the tile at a time."""
    if has_residual:
        x_ref, r_ref, w_ref, y_ref, ro_ref, sum_ref = refs
    else:
        x_ref, w_ref, y_ref = refs
        r_ref = ro_ref = sum_ref = None

    def body(start):
        rows = pl.ds(start, slice_rows)
        var = _slice_mean_square(x_ref, r_ref, ro_ref, sum_ref, rows)
        inv_rms = jax.lax.rsqrt(var + eps)
        for cols in _column_groups(x_ref.shape[1]):
            y_ref[rows, cols] = _slice_norm(
                x_ref, sum_ref, w_ref, rows, cols, inv_rms, y_ref.dtype
            )

    _for_each_slice(x_ref.shape[0], slice_rows, body)


def _check_slices(block_rows: int, slice_rows: int) -> None:
    if slice_rows and block_rows % slice_rows:
        raise ValueError(f"{block_rows=} must be a multiple of {slice_rows=}")


def _slice_scratch(slice_rows: int, hidden: int) -> list:
    if not slice_rows:
        return []
    return [pltpu.VMEM((slice_rows, hidden), jnp.float32)]


def _slice_suffix(slice_rows: int) -> str:
    return f"_s{slice_rows}" if slice_rows else ""


def supports(num_rows: int, hidden: int, block_rows: int = DEFAULT_BLOCK_ROWS) -> bool:
    return num_rows >= block_rows and num_rows % block_rows == 0 and hidden % 128 == 0


@functools.partial(
    jax.jit, static_argnames=("eps", "block_rows", "slice_rows", "interpret")
)
def fused_add_rms_norm(
    x: jax.Array,
    residual: jax.Array | None,
    weight: jax.Array,
    *,
    eps: float,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    slice_rows: int = 0,
    interpret: bool = False,
):
    """Returns ``(y, new_residual)``, or ``y`` alone when ``residual`` is None.

    ``x``/``residual`` are ``[rows, hidden]`` of one dtype; ``weight`` is
    ``[hidden]`` of that dtype. ``rows`` must be a multiple of ``block_rows``.
    ``slice_rows`` (a divisor of ``block_rows``) computes each tile one slice
    of rows at a time, with the same values; 0 computes the whole tile at once.
    """
    rows, hidden = x.shape
    if rows % block_rows:
        raise ValueError(f"{rows=} must be a multiple of {block_rows=}")
    _check_slices(block_rows, slice_rows)
    has_residual = residual is not None
    row_spec = pl.BlockSpec((block_rows, hidden), lambda i: (i, 0))
    w_spec = pl.BlockSpec((1, hidden), lambda i: (0, 0))
    out_shape = [jax.ShapeDtypeStruct((rows, hidden), x.dtype)]
    out_specs = [row_spec]
    in_specs = [row_spec]
    args = [x]
    if has_residual:
        in_specs.append(row_spec)
        args.append(residual)
        out_shape.append(jax.ShapeDtypeStruct((rows, hidden), x.dtype))
        out_specs.append(row_spec)
    in_specs.append(w_spec)
    args.append(weight.reshape(1, hidden))
    # 2 (or 4) row tiles double-buffered plus ~4 f32 tile temporaries.
    tile = block_rows * hidden * 4
    vmem_limit = min(100 * 1024 * 1024, max(32 * 1024 * 1024, 12 * tile))
    if slice_rows:
        body = functools.partial(
            _sliced_kernel, eps=eps, has_residual=has_residual, slice_rows=slice_rows
        )
    else:
        body = functools.partial(_kernel, eps=eps, has_residual=has_residual)
    outs = pl.pallas_call(
        body,
        grid=(rows // block_rows,),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        scratch_shapes=_slice_scratch(slice_rows if has_residual else 0, hidden),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=vmem_limit,
        ),
        name=f"fused_add_rmsnorm_r{int(has_residual)}_b{block_rows}_h{hidden}"
        f"{_slice_suffix(slice_rows)}",
        interpret=interpret,
    )(*args)
    if has_residual:
        return outs[0], outs[1]
    return outs[0]


def _quant_kernel(*refs, eps: float, emit_y: bool, q_max: float):
    """``_kernel`` plus the per-row activation quant of
    ``quantized_matmul.util.quantize_block`` on the bf16-rounded ``y``:

        amax  = max|y|                       (exact: y is bf16)
        scale = f32(amax) / q_max;  0 -> 1
        y_q   = fp8(clip(f32(y) / scale, -q_max, q_max))
    """
    if emit_y:
        x_ref, r_ref, w_ref, y_ref, q_ref, s_ref, ro_ref = refs
    else:
        x_ref, r_ref, w_ref, q_ref, s_ref, ro_ref = refs
        y_ref = None
    s = x_ref[...].astype(jnp.float32) + r_ref[...].astype(jnp.float32)
    ro_ref[...] = s.astype(ro_ref.dtype)
    var = jnp.mean(s * s, axis=-1, keepdims=True)
    normed = (s * jax.lax.rsqrt(var + eps)).astype(x_ref.dtype)
    y = (normed.astype(jnp.float32) * w_ref[...].astype(jnp.float32)).astype(
        x_ref.dtype
    )
    if emit_y:
        assert y_ref is not None
        y_ref[...] = y
    yf = y.astype(jnp.float32)
    amax = jnp.max(jnp.abs(yf), axis=-1, keepdims=True)
    scale = amax / jnp.float32(q_max)
    scale = jnp.where(scale == 0, jnp.float32(1.0), scale)
    s_ref[...] = scale
    q_ref[...] = jnp.clip(yf / scale, -q_max, q_max).astype(q_ref.dtype)


def _sliced_quant_kernel(*refs, eps: float, emit_y: bool, q_max: float, slice_rows: int):
    """``_quant_kernel`` computed one ``slice_rows`` slice of the tile at a
    time. The quant pass after the row amax reads ``y`` back from its output
    block, or from a bf16 scratch without one."""
    if emit_y:
        x_ref, r_ref, w_ref, y_ref, q_ref, s_ref, ro_ref, sum_ref = refs
    else:
        x_ref, r_ref, w_ref, q_ref, s_ref, ro_ref, sum_ref, y_ref = refs
    groups = _column_groups(x_ref.shape[1])

    def body(start):
        rows = pl.ds(start, slice_rows)
        y_rows = rows if emit_y else slice(None)
        var = _slice_mean_square(x_ref, r_ref, ro_ref, sum_ref, rows)
        inv_rms = jax.lax.rsqrt(var + eps)
        peak = None
        for cols in groups:
            y = _slice_norm(x_ref, sum_ref, w_ref, rows, cols, inv_rms, x_ref.dtype)
            y_ref[y_rows, cols] = y
            peak = _row_peak(peak, jnp.abs(y.astype(jnp.float32)))
        amax = jnp.max(peak, axis=-1, keepdims=True)
        scale = amax / jnp.float32(q_max)
        scale = jnp.where(scale == 0, jnp.float32(1.0), scale)
        s_ref[rows, :] = scale
        for cols in groups:
            yf = y_ref[y_rows, cols].astype(jnp.float32)
            q_ref[rows, cols] = jnp.clip(yf / scale, -q_max, q_max).astype(q_ref.dtype)

    _for_each_slice(x_ref.shape[0], slice_rows, body)


@functools.partial(
    jax.jit,
    static_argnames=(
        "eps",
        "quant_dtype",
        "emit_y",
        "block_rows",
        "slice_rows",
        "interpret",
    ),
)
def fused_add_rms_norm_quant(
    x: jax.Array,
    residual: jax.Array,
    weight: jax.Array,
    *,
    eps: float,
    quant_dtype=jnp.float8_e4m3fn,
    emit_y: bool = True,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    slice_rows: int = 0,
    interpret: bool = False,
):
    """``TPU_FUSED_ADD_RMSNORM_QUANT``: the add + RMSNorm pass that also emits
    the consumer's per-row fp8 activation, so the fp8 matmul behind it skips
    its amax and convert passes over ``y``.

    Returns ``(y | None, y_q [rows, hidden], scale f32 [rows, 1], residual)``;
    ``(y_q, scale)`` are bit for bit ``quantize_tensor(y, quant_dtype)``.
    ``y`` is written only with ``emit_y`` (a second consumer needs bf16).
    ``slice_rows`` as in ``fused_add_rms_norm``.
    """
    if slice_rows:
        body = functools.partial(_sliced_quant_kernel, slice_rows=slice_rows)
    else:
        body = _quant_kernel
    return _quant_call(
        x,
        residual,
        weight,
        eps=eps,
        quant_dtype=quant_dtype,
        emit_y=emit_y,
        block_rows=block_rows,
        interpret=interpret,
        body=body,
        slice_rows=slice_rows,
    )


def _quant_call(
    x,
    residual,
    weight,
    *,
    eps,
    quant_dtype,
    emit_y,
    block_rows,
    interpret,
    body,
    slice_rows=0,
):
    """The wrapper; ``body`` is a kernel with ``_quant_kernel``'s signature
    (tests pass fault-injected bodies through the same grid and specs), or
    ``_sliced_quant_kernel``'s with ``slice_rows``."""
    rows, hidden = x.shape
    if rows % block_rows:
        raise ValueError(f"{rows=} must be a multiple of {block_rows=}")
    _check_slices(block_rows, slice_rows)
    q_max = float(jnp.finfo(quant_dtype).max)
    row_spec = pl.BlockSpec((block_rows, hidden), lambda i: (i, 0))
    w_spec = pl.BlockSpec((1, hidden), lambda i: (0, 0))
    s_spec = pl.BlockSpec((block_rows, 1), lambda i: (i, 0))
    out_shape = []
    out_specs = []
    if emit_y:
        out_shape.append(jax.ShapeDtypeStruct((rows, hidden), x.dtype))
        out_specs.append(row_spec)
    out_shape += [
        jax.ShapeDtypeStruct((rows, hidden), quant_dtype),
        jax.ShapeDtypeStruct((rows, 1), jnp.float32),
        jax.ShapeDtypeStruct((rows, hidden), x.dtype),
    ]
    out_specs += [row_spec, s_spec, row_spec]
    scratch_shapes = _slice_scratch(slice_rows, hidden)
    if slice_rows and not emit_y:
        scratch_shapes.append(pltpu.VMEM((slice_rows, hidden), x.dtype))
    tile = block_rows * hidden * 4
    vmem_limit = min(100 * 1024 * 1024, max(32 * 1024 * 1024, 14 * tile))
    outs = pl.pallas_call(
        functools.partial(body, eps=eps, emit_y=emit_y, q_max=q_max),
        grid=(rows // block_rows,),
        in_specs=[row_spec, row_spec, w_spec],
        out_specs=out_specs,
        out_shape=out_shape,
        scratch_shapes=scratch_shapes,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=vmem_limit,
        ),
        name=f"fused_add_rmsnorm_quant_y{int(emit_y)}_b{block_rows}_h{hidden}"
        f"{_slice_suffix(slice_rows)}",
        interpret=interpret,
    )(x, residual, weight.reshape(1, hidden))
    if emit_y:
        return outs[0], outs[1], outs[2], outs[3]
    return None, outs[0], outs[1], outs[2]


def _moe_quant_kernel(
    x_ref, r_ref, w_ref, y_ref, q_ref, s_ref, ro_ref, *, eps: float, q_max: float
):
    """``_kernel`` plus the fused MoE prefill kernel's own activation quant
    (the stage-A build's ``fused_moe_prefill_kernel_v2._quantize_rows``) on the
    bf16 ``y``, op for op:

        amax  = max|y|                          (bf16, exact)
        scale = f32(amax) / q_max;  usable = finite & > 0;  scale = usable ? scale : 0
        inv   = usable ? 1 / scale : 0
        x_q   = fp8(usable ? y * bf16(inv) : 0)  (bf16 multiply)

    written straight into the [rows, hidden / 128, 128] per-row layout the
    expert kernel gathers rows from: the natural fp8 tile viewed as u32 (four
    rows per word) is byte-transposed four lane blocks at a time and stored
    with sublane-strided u32 stores (the inverse of the expert kernel's LHS
    build)."""
    from vllm_torchtpu.kernels.experimental.fused_moe_prefill.fused_moe_prefill_kernel import (  # noqa: E501
        _byte_transpose4,
    )

    lanes = 128
    s = x_ref[...].astype(jnp.float32) + r_ref[...].astype(jnp.float32)
    ro_ref[...] = s.astype(ro_ref.dtype)
    var = jnp.mean(s * s, axis=-1, keepdims=True)
    normed = (s * jax.lax.rsqrt(var + eps)).astype(x_ref.dtype)
    y = (normed.astype(jnp.float32) * w_ref[...].astype(jnp.float32)).astype(
        x_ref.dtype
    )
    y_ref[...] = y
    amax = jnp.max(jnp.abs(y), axis=-1, keepdims=True)
    scale = amax.astype(jnp.float32) / q_max
    usable = jnp.isfinite(scale) & (scale > 0)
    scale = jnp.where(usable, scale, 0.0)
    inv = jnp.where(usable, 1.0 / jnp.where(usable, scale, 1.0), 0.0)
    q = jnp.where(usable, y * inv.astype(y.dtype), jnp.zeros_like(y)).astype(
        q_ref.dtype
    )
    s_ref[...] = scale
    rows, hidden = y.shape
    lane_blocks = hidden // lanes
    wpr = lane_blocks // 4
    # byte t of word (s, c) of qn is row 4s + t, column c
    qn = pltpu.bitcast(q, jnp.uint32)
    # word-row r * wpr + k of q32 holds lane blocks 4k..4k+3 of row r
    # (block 4k + j in byte j)
    q32 = q_ref.bitcast(jnp.uint32).reshape(rows * wpr, lanes)
    for k in range(wpr):
        a = [qn[:, (4 * k + j) * lanes : (4 * k + j + 1) * lanes] for j in range(4)]
        b = _byte_transpose4(a)  # b[t]: byte j = row 4s + t, lane block 4k + j
        for t in range(4):
            q32[pl.ds(t * wpr + k, rows // 4, stride=lane_blocks)] = b[t]


def _sliced_moe_quant_kernel(
    *refs, eps: float, q_max: float, slice_rows: int, router: bool
):
    """``_moe_quant_kernel`` computed one ``slice_rows`` slice of the tile at
    a time. The quant pass after the row amax reads ``y`` back from its output
    block, one byte-transpose group of columns at a time, and stores the same
    u32 words to the same rows of the per-row layout as the whole-tile body.

    With ``router``, ``y`` goes to a bf16 scratch of the whole tile instead of
    an output, and the tile ends with the router logits ``y @ gate.T`` from
    it: bf16 operands, f32 accumulation, rounded to bf16 as the gate's XLA
    matmul rounds them."""
    from vllm_torchtpu.kernels.experimental.fused_moe_prefill.fused_moe_prefill_kernel import (  # noqa: E501
        _byte_transpose4,
    )

    if router:
        (x_ref, r_ref, w_ref, gate_ref, q_ref, s_ref, ro_ref, logits_ref) = refs[:8]
        sum_ref, y_ref = refs[8:]
    else:
        x_ref, r_ref, w_ref, y_ref, q_ref, s_ref, ro_ref, sum_ref = refs

    lanes = 128
    tile_rows, hidden = x_ref.shape
    lane_blocks = hidden // lanes
    wpr = lane_blocks // 4
    q32 = q_ref.bitcast(jnp.uint32).reshape(tile_rows * wpr, lanes)
    groups = _column_groups(hidden)

    def body(start):
        rows = pl.ds(start, slice_rows)
        var = _slice_mean_square(x_ref, r_ref, ro_ref, sum_ref, rows)
        inv_rms = jax.lax.rsqrt(var + eps)
        peak = None
        for cols in groups:
            y = _slice_norm(x_ref, sum_ref, w_ref, rows, cols, inv_rms, x_ref.dtype)
            y_ref[rows, cols] = y
            peak = _row_peak(peak, jnp.abs(y))
        amax = jnp.max(peak, axis=-1, keepdims=True)
        scale = amax.astype(jnp.float32) / q_max
        usable = jnp.isfinite(scale) & (scale > 0)
        scale = jnp.where(usable, scale, 0.0)
        inv = jnp.where(usable, 1.0 / jnp.where(usable, scale, 1.0), 0.0)
        s_ref[rows, :] = scale
        first = start * wpr
        for k, cols in enumerate(groups):
            y = y_ref[rows, cols]
            q = jnp.where(usable, y * inv.astype(y.dtype), jnp.zeros_like(y)).astype(
                q_ref.dtype
            )
            b = _byte_transpose4(_lane_blocks(pltpu.bitcast(q, jnp.uint32)))
            for t in range(4):
                q32[
                    pl.ds(first + t * wpr + k, slice_rows // 4, stride=lane_blocks)
                ] = b[t]

    _for_each_slice(tile_rows, slice_rows, body)
    if router:
        logits_ref[...] = jax.lax.dot_general(
            y_ref[...],
            gate_ref[...],
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(logits_ref.dtype)


@functools.partial(
    jax.jit, static_argnames=("eps", "block_rows", "slice_rows", "interpret")
)
def fused_add_rms_norm_quant_moe(
    x: jax.Array,
    residual: jax.Array,
    weight: jax.Array,
    gate: jax.Array | None = None,
    *,
    eps: float,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    slice_rows: int = 0,
    interpret: bool = False,
):
    """The add + RMSNorm pass that also emits the fused MoE prefill kernel's
    activation rows, so the op skips its own amax / convert passes over ``y``
    (``TPU_FUSED_ADD_RMSNORM_QUANT_MOE``, #1935 trial 4; pass the results to
    fused_moe_prefill_kernel_v2's ``x_q`` / ``x_scale``).

    Returns ``(y [rows, hidden], x_q fp8 [rows, hidden / 128, 128],
    x_scale f32 [rows], new_residual)``; ``(x_q, x_scale)`` are bit for bit
    ``fused_moe_prefill_kernel_v2._quantize_rows(y)``. ``y`` is still written:
    the router reads it. ``slice_rows`` as in ``fused_add_rms_norm``.

    With ``gate`` (the router weight, ``[experts, hidden]`` of x's dtype; needs
    ``slice_rows``), the kernel computes the router logits from ``y`` itself
    and does not write ``y``: returns ``(None, x_q, x_scale, new_residual,
    logits [rows, experts])``, the logits in x's dtype, the gate matmul's
    ``bf16(y @ gate.T)``. The weight stays resident in VMEM.
    """
    rows, hidden = x.shape
    if rows % block_rows or hidden % 512:
        raise ValueError(f"{rows=} {hidden=} {block_rows=}")
    _check_slices(block_rows, slice_rows)
    if gate is not None:
        return _quant_moe_router_call(
            x,
            residual,
            weight,
            gate,
            eps=eps,
            block_rows=block_rows,
            slice_rows=slice_rows,
            interpret=interpret,
        )
    q_dtype = jnp.float8_e4m3fn
    q_max = float(jnp.finfo(q_dtype).max)
    lane_blocks = hidden // 128
    row_spec = pl.BlockSpec((block_rows, hidden), lambda i: (i, 0))
    q_spec = pl.BlockSpec((block_rows, lane_blocks, 128), lambda i: (i, 0, 0))
    s_spec = pl.BlockSpec((block_rows, 1), lambda i: (i, 0))
    w_spec = pl.BlockSpec((1, hidden), lambda i: (0, 0))
    tile = block_rows * hidden * 4
    if slice_rows:
        body = functools.partial(
            _sliced_moe_quant_kernel,
            eps=eps,
            q_max=q_max,
            slice_rows=slice_rows,
            router=False,
        )
    else:
        body = functools.partial(_moe_quant_kernel, eps=eps, q_max=q_max)
    y, x_q, scale, new_res = pl.pallas_call(
        body,
        grid=(rows // block_rows,),
        in_specs=[row_spec, row_spec, w_spec],
        out_specs=[row_spec, q_spec, s_spec, row_spec],
        out_shape=[
            jax.ShapeDtypeStruct((rows, hidden), x.dtype),
            jax.ShapeDtypeStruct((rows, lane_blocks, 128), q_dtype),
            jax.ShapeDtypeStruct((rows, 1), jnp.float32),
            jax.ShapeDtypeStruct((rows, hidden), x.dtype),
        ],
        scratch_shapes=_slice_scratch(slice_rows, hidden),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=min(100 * 1024 * 1024, max(32 * 1024 * 1024, 14 * tile)),
        ),
        name=f"fused_add_rmsnorm_quant_moe_b{block_rows}_h{hidden}"
        f"{_slice_suffix(slice_rows)}",
        interpret=interpret,
    )(x, residual, weight.reshape(1, hidden))
    return y, x_q, scale.reshape(-1), new_res


def _quant_moe_router_call(
    x, residual, weight, gate, *, eps, block_rows, slice_rows, interpret
):
    rows, hidden = x.shape
    experts = gate.shape[0]
    if not slice_rows or gate.shape != (experts, hidden) or experts % 128:
        raise ValueError(f"{slice_rows=} {gate.shape=} {hidden=}")
    q_dtype = jnp.float8_e4m3fn
    q_max = float(jnp.finfo(q_dtype).max)
    lane_blocks = hidden // 128
    row_spec = pl.BlockSpec((block_rows, hidden), lambda i: (i, 0))
    q_spec = pl.BlockSpec((block_rows, lane_blocks, 128), lambda i: (i, 0, 0))
    s_spec = pl.BlockSpec((block_rows, 1), lambda i: (i, 0))
    w_spec = pl.BlockSpec((1, hidden), lambda i: (0, 0))
    gate_spec = pl.BlockSpec(
        (experts, hidden), lambda i: (0, 0), pipeline_mode=pl.Buffered(1)
    )
    logits_spec = pl.BlockSpec((block_rows, experts), lambda i: (i, 0))
    tile = block_rows * hidden * 4
    x_q, scale, new_res, logits = pl.pallas_call(
        functools.partial(
            _sliced_moe_quant_kernel,
            eps=eps,
            q_max=q_max,
            slice_rows=slice_rows,
            router=True,
        ),
        grid=(rows // block_rows,),
        in_specs=[row_spec, row_spec, w_spec, gate_spec],
        out_specs=[q_spec, s_spec, row_spec, logits_spec],
        out_shape=[
            jax.ShapeDtypeStruct((rows, lane_blocks, 128), q_dtype),
            jax.ShapeDtypeStruct((rows, 1), jnp.float32),
            jax.ShapeDtypeStruct((rows, hidden), x.dtype),
            jax.ShapeDtypeStruct((rows, experts), x.dtype),
        ],
        scratch_shapes=[
            *_slice_scratch(slice_rows, hidden),
            pltpu.VMEM((block_rows, hidden), x.dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=min(100 * 1024 * 1024, max(32 * 1024 * 1024, 14 * tile)),
        ),
        name=f"fused_add_rmsnorm_quant_moe_router_b{block_rows}_h{hidden}_e{experts}"
        f"{_slice_suffix(slice_rows)}",
        interpret=interpret,
    )(x, residual, weight.reshape(1, hidden), gate)
    return None, x_q, scale.reshape(-1), new_res, logits


def reference(x, residual, weight, eps):
    """``vllm.ir.ops`` numerics in JAX (XLA)."""
    orig = x.dtype
    s = x.astype(jnp.float32)
    new_res = None
    if residual is not None:
        s = s + residual.astype(jnp.float32)
        new_res = s.astype(orig)
    var = jnp.mean(s * s, axis=-1, keepdims=True)
    y = (s * jax.lax.rsqrt(var + eps)).astype(weight.dtype) * weight
    y = y.astype(orig)
    return (y, new_res) if residual is not None else y
