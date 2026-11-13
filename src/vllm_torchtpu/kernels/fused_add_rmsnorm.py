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

The op adds ``x`` to the residual stream, writes the new residual, and writes
the RMS-normalized and scaled sum. For a ``[rows, hidden]`` input with many
rows, XLA lowers it as two fusions: a row reduction, which also writes the new
residual, and an apply pass, which reads both inputs again and repeats the add.
This kernel keeps whole rows in VMEM, so it reads ``x`` and ``residual`` once
and writes the new residual and the output once.

The numerics follow the ``native`` provider of ``vllm/ir/ops/layernorm.py``
step by step:

    s        = f32(x) + f32(residual)        # not rounded
    residual = bf16(s)
    y        = bf16(bf16(s * rsqrt(mean(s^2) + eps)) * weight)

Only the f32 summation order of ``mean(s^2)`` is free. So the new residual is
bit-identical to the upstream op, and ``y`` can differ by at most 2 bf16 ulp.

``fused_add_rms_norm_quant`` also writes the per-row fp8 quantization of
``y``, which the FP8 matmul that reads ``y`` would otherwise compute in two
more passes over ``y``.

Example for one row with hidden size 4 (the arithmetic only; the kernel needs
a hidden size that is a multiple of 128), ``eps`` = 1e-6 and a weight of ones:
``x`` = [1, 1, 1, 1] and ``residual`` = [1, -1, 3, -3] give ``s`` =
[2, 0, 4, -2]. The mean of the squares is 6, and rsqrt(6) = 0.40825. So the
new residual is [2, 0, 4, -2], and ``y`` is [0.8164, 0, 1.6328, -0.8164]
after bf16 rounding.

The bounds above hold against the ``native`` provider only. On TPU the default
provider of these ops is ``tpu`` (``layers/adapter/ir_ops.py``), which rounds
at other points: it normalizes the bf16-rounded sum, and it multiplies the
weight in f32 without the intermediate bf16 rounding. The difference between
this kernel and the ``tpu`` provider is not measured.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Rows per grid step. One [256, 6144] bf16 tile is 3 MiB. This value was tuned
# on TPU v7x at [4096, 6144] bf16, where 256 rows ran faster than 128.
DEFAULT_BLOCK_ROWS = 256

# Number of f32 [block_rows, hidden] values the kernel body keeps live: the sum,
# its square, the normalized value and the weighted product. An estimate, used
# only to reject blocks that cannot fit in VMEM.
_F32_TEMPORARIES = 4


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


def _vmem_working_set_bytes(
    block_rows: int, hidden: int, itemsize: int, has_residual: bool
) -> int:
    """Returns the estimated VMEM bytes of one grid step of ``fused_add_rms_norm``.

    The Pallas pipeline double-buffers each row block of the inputs and the
    outputs. The kernel body also keeps ``_F32_TEMPORARIES`` f32 values of
    the block size live. The ``[1, hidden]`` weight block is small and is not
    counted.

    Example: bf16 (itemsize 2) with a residual, hidden 6144 and 256 rows.
    One row block is 3 MiB. Four row blocks (x, residual, y, new residual),
    each double-buffered, take 24 MiB. Four f32 temporaries of 6 MiB each take
    24 MiB. The estimate is 48 MiB.

    Args:
        block_rows: Rows per grid step.
        hidden: Size of the normalized (last) dimension.
        itemsize: Bytes per element of ``x``.
        has_residual: Whether the call has a residual input and output.

    Returns:
        The estimate in bytes.
    """
    row_arrays = 4 if has_residual else 2
    block = block_rows * hidden
    return 2 * row_arrays * block * itemsize + _F32_TEMPORARIES * block * 4


# VMEM budget of ``supports()``: 64 MiB, the VMEM of one TPU v7x TensorCore,
# the smallest of the TPU generations this kernel was run on. ``supports()``
# cannot read the chip's capacity because it runs inside torch.compile traces.
SUPPORTS_VMEM_BYTES = 64 * 1024 * 1024


def _vmem_capacity_bytes() -> int | None:
    """Returns the VMEM capacity of one TensorCore, or None on a host without a TPU."""
    try:
        return pltpu.get_tpu_info().vmem_capacity_bytes
    except ValueError:  # Non-TPU host.
        return None


def _check_shape(rows: int, hidden: int, block_rows: int) -> None:
    """Raises ValueError for a ``[rows, hidden]`` input the row tiling rejects.

    Args:
        rows: Number of rows of ``x``.
        hidden: Size of the normalized (last) dimension.
        block_rows: Rows per grid step.

    Raises:
        ValueError: If ``rows`` is not a multiple of ``block_rows``, if
            ``hidden`` is not a multiple of 128 (the TPU lane width), or if
            ``block_rows`` is not a multiple of 8 (the TPU sublane count) and
            not equal to ``rows``.
    """
    if rows % block_rows:
        raise ValueError(f"{rows=} must be a multiple of {block_rows=}")
    if hidden % 128:
        raise ValueError(f"{hidden=} must be a multiple of 128, the TPU lane width")
    if block_rows % 8 and block_rows != rows:
        raise ValueError(f"{block_rows=} must be a multiple of 8 or equal {rows=}")


def supports(num_rows: int, hidden: int, block_rows: int = DEFAULT_BLOCK_ROWS) -> bool:
    """Returns whether ``fused_add_rms_norm`` accepts a ``[num_rows, hidden]`` input.

    The VMEM check assumes bf16 inputs with a residual, the larger working set,
    and compares it with ``SUPPORTS_VMEM_BYTES``, not with the chip's own
    capacity. Callers run this function inside ``torch.compile`` traces, where
    ``pltpu.get_tpu_info()`` cannot be traced.

    Args:
        num_rows: Number of rows of ``x``.
        hidden: Size of the normalized (last) dimension.
        block_rows: Rows per grid step.

    Returns:
        True when ``num_rows`` is a positive multiple of ``block_rows``,
        ``hidden`` is a multiple of 128 (the TPU lane width), ``block_rows``
        is a multiple of 8 or equals ``num_rows``, and the estimated VMEM
        working set fits in ``SUPPORTS_VMEM_BYTES``. For example,
        ``supports(4096, 6144)`` is True (48 MiB), ``supports(4100, 6144)``
        is False, and ``supports(4096, 16384)`` is False (128 MiB).
    """
    if num_rows < block_rows or num_rows % block_rows or hidden % 128:
        return False
    if block_rows % 8 and block_rows != num_rows:
        return False
    need = _vmem_working_set_bytes(block_rows, hidden, 2, has_residual=True)
    return need <= SUPPORTS_VMEM_BYTES


@functools.partial(jax.jit, static_argnames=("eps", "block_rows", "interpret"))
def fused_add_rms_norm(
    x: jax.Array,
    residual: jax.Array | None,
    weight: jax.Array,
    *,
    eps: float,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    interpret: bool = False,
):
    """Computes (fused add +) RMSNorm in one pass over HBM.

    Args:
        x: ``[rows, hidden]`` input, for example bf16.
        residual: ``[rows, hidden]`` residual stream of the same dtype as
            ``x``, or None for a plain RMSNorm of ``x``.
        weight: ``[hidden]`` scale of the same dtype as ``x``.
        eps: Constant added to the mean of the squares before the rsqrt.
        block_rows: Rows per grid step. ``rows`` must be a multiple of it.
            A value that is not a multiple of 8 is valid only when it equals
            ``rows``.
        interpret: Run the kernel in the Pallas interpreter, for tests on a
            host without a TPU.

    Returns:
        ``(y, new_residual)``, both ``[rows, hidden]`` in the dtype of ``x``,
        where ``new_residual`` is ``x + residual``. When ``residual`` is None,
        only ``y``.

    Raises:
        ValueError: If ``rows`` is not a multiple of ``block_rows``, if
            ``hidden`` is not a multiple of 128, if ``block_rows`` is not a
            multiple of 8 and not equal to ``rows``, or, when ``interpret``
            is False on a TPU host, if the estimated VMEM working set of one
            grid step is larger than the VMEM of one TensorCore.
    """
    rows, hidden = x.shape
    _check_shape(rows, hidden, block_rows)
    has_residual = residual is not None
    capacity = None if interpret else _vmem_capacity_bytes()
    need = _vmem_working_set_bytes(block_rows, hidden, x.dtype.itemsize, has_residual)
    if capacity is not None and need > capacity:
        raise ValueError(
            f"{block_rows=} and {hidden=} need about {need} bytes of VMEM, more "
            f"than the {capacity} bytes of one TensorCore; use a smaller block_rows"
        )
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
    # 2 (or 4) row tiles double-buffered plus about 4 f32 tile temporaries.
    tile = block_rows * hidden * 4
    vmem_limit = min(100 * 1024 * 1024, max(32 * 1024 * 1024, 12 * tile))
    outs = pl.pallas_call(
        functools.partial(_kernel, eps=eps, has_residual=has_residual),
        grid=(rows // block_rows,),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=vmem_limit,
        ),
        name=f"fused_add_rmsnorm_r{int(has_residual)}_b{block_rows}_h{hidden}",
        interpret=interpret,
    )(*args)
    if has_residual:
        return outs[0], outs[1]
    return outs[0]


def _quant_kernel(*refs, eps: float, emit_y: bool, q_max: float):
    """``_kernel`` plus a per-row quantization of the bf16-rounded ``y``.

    The quantization is the one of ``quantized_matmul.util.quantize_block``:

        amax  = max|y|                       (exact, because y is bf16)
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
        y_ref[...] = y
    yf = y.astype(jnp.float32)
    amax = jnp.max(jnp.abs(yf), axis=-1, keepdims=True)
    scale = amax / jnp.float32(q_max)
    scale = jnp.where(scale == 0, jnp.float32(1.0), scale)
    s_ref[...] = scale
    q_ref[...] = jnp.clip(yf / scale, -q_max, q_max).astype(q_ref.dtype)


@functools.partial(
    jax.jit, static_argnames=("eps", "quant_dtype", "emit_y", "block_rows", "interpret")
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
    interpret: bool = False,
):
    """Computes add + RMSNorm and also the per-row quantization of its output.

    An FP8 matmul that reads ``y`` first quantizes it per row, which reads
    ``y`` twice: once for the row maximum and once for the conversion. This
    kernel computes the same quantization while the row is still in VMEM.

    Example for one row: if ``y`` = [0.5, -2.0, 1.0] and ``quant_dtype`` is
    float8_e4m3fn (``q_max`` = 448), then ``scale`` = 2.0 / 448 = 0.0044643
    and ``y_q`` = [112, -448, 224].

    Args:
        x: ``[rows, hidden]`` input, for example bf16.
        residual: ``[rows, hidden]`` residual stream of the same dtype as
            ``x``.
        weight: ``[hidden]`` scale of the same dtype as ``x``.
        eps: Constant added to the mean of the squares before the rsqrt.
        quant_dtype: Dtype of ``y_q``, a floating dtype such as
            ``jnp.float8_e4m3fn``.
        emit_y: Also write ``y``, for callers with a second consumer of it.
        block_rows: Rows per grid step. ``rows`` must be a multiple of it.
            A value that is not a multiple of 8 is valid only when it equals
            ``rows``.
        interpret: Run the kernel in the Pallas interpreter.

    Returns:
        ``(y, y_q, scale, new_residual)``. ``y`` and ``new_residual`` are as
        in ``fused_add_rms_norm``; ``y`` is None when ``emit_y`` is False.
        ``y_q`` is ``[rows, hidden]`` in ``quant_dtype`` and ``scale`` is
        ``[rows, 1]`` f32. ``(y_q, scale)`` are bit for bit
        ``quantize_tensor(y, quant_dtype)``.

    Raises:
        ValueError: If ``rows`` is not a multiple of ``block_rows``, if
            ``hidden`` is not a multiple of 128, if ``block_rows`` is not a
            multiple of 8 and not equal to ``rows``, or if ``quant_dtype`` is
            not a floating dtype.
    """
    rows, hidden = x.shape
    _check_shape(rows, hidden, block_rows)
    if not jnp.issubdtype(quant_dtype, jnp.floating):
        raise ValueError(
            f"{quant_dtype=} is not supported; use a floating fp8 dtype such as "
            "jnp.float8_e4m3fn"
        )
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
    # 14 f32 tiles, counted as in fused_add_rms_norm: each row block counts as
    # one f32 tile and is double-buffered, so x, residual, new residual, y and
    # y_q give 10 tiles; the f32 temporaries give about 4 more. The bf16 and
    # fp8 blocks are smaller than an f32 tile, so the count is an upper bound.
    tile = block_rows * hidden * 4
    vmem_limit = min(100 * 1024 * 1024, max(32 * 1024 * 1024, 14 * tile))
    outs = pl.pallas_call(
        functools.partial(_quant_kernel, eps=eps, emit_y=emit_y, q_max=q_max),
        grid=(rows // block_rows,),
        in_specs=[row_spec, row_spec, w_spec],
        out_specs=out_specs,
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=vmem_limit,
        ),
        name=f"fused_add_rmsnorm_quant_y{int(emit_y)}_b{block_rows}_h{hidden}",
        interpret=interpret,
    )(x, residual, weight.reshape(1, hidden))
    if emit_y:
        return outs[0], outs[1], outs[2], outs[3]
    return None, outs[0], outs[1], outs[2]


def reference(x, residual, weight, eps):
    """Computes (fused add +) RMSNorm with plain JAX ops, lowered by XLA.

    The arithmetic is the one of ``fused_add_rms_norm``. On CPU, a test checks
    it against the bound of the kernel: the new residual is bit-identical to
    the host oracle of the kernel tests, and ``y`` differs by at most 2 bf16
    ulp on at most 0.1% of the elements. On TPU, XLA can keep f32 where the
    kernel rounds to bf16, so ``y`` can differ on more elements. No test
    checks that case.

    Args:
        x: ``[rows, hidden]`` input.
        residual: ``[rows, hidden]`` residual stream, or None.
        weight: ``[hidden]`` scale.
        eps: Constant added to the mean of the squares before the rsqrt.

    Returns:
        ``(y, new_residual)``, or ``y`` alone when ``residual`` is None.
    """
    orig = x.dtype
    s = x.astype(jnp.float32)
    if residual is not None:
        s = s + residual.astype(jnp.float32)
        new_res = s.astype(orig)
    var = jnp.mean(s * s, axis=-1, keepdims=True)
    y = (s * jax.lax.rsqrt(var + eps)).astype(weight.dtype) * weight
    y = y.astype(orig)
    return (y, new_res) if residual is not None else y
