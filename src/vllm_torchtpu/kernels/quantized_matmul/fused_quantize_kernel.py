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
"""FP8 matmul that quantizes its per-token activation in VMEM.

Computes what `util.xla_quantized_matmul` computes for an fp8 weight in the
(k, n) layout, without the separate abs-max and quantize passes over the
activation in HBM. The grid walks weight column tiles outermost; the first
column pass quantizes each row tile in the same basic block as its matmul, so
the quantize overlaps the MXU work, and keeps it in a scratch that every later
pass reuses.

`fused_rms_norm_quantized_matmul` also takes the RMSNorm that produces the
activation: the first pass scales the raw rows by their precomputed rsqrt and
the norm weight in VMEM, so the normalized activation is never read back from
HBM, and is written there only when the caller asks for it.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.quantized_matmul import util

# Tile sizes and the shape gate are tunables in `util` (FUSED_QUANTIZE_*), read
# at trace time.
# Columns per quantize step; rows per step are one fp8 vreg's 32 sublanes.
QUANT_COLS = 512
QUANT_ROWS = 32
VMEM_LIMIT_BYTES = 48 * 1024 * 1024


def _first_divisor(size: int, blocks: tuple[int, ...]) -> int | None:
    return next((block for block in blocks if size % block == 0), None)


def _tiles(m: int, k: int, n: int) -> tuple[int, int] | None:
    """(rows, weight columns) per grid step, or None if XLA should keep it."""
    if n < util.FUSED_QUANTIZE_MAX_OUT_PER_IN * k:
        tm = util.FUSED_QUANTIZE_TOKEN_BLOCK
        tm = tm if m % tm == 0 else None
        tn = _first_divisor(n, (util.FUSED_QUANTIZE_OUT_BLOCK, 512, 256))
    elif n >= util.FUSED_QUANTIZE_MIN_WIDE_OUT_PER_IN * k:
        tm = _first_divisor(
            m, (util.FUSED_QUANTIZE_WIDE_TOKEN_BLOCK, util.FUSED_QUANTIZE_TOKEN_BLOCK)
        )
        tn = _first_divisor(
            n,
            (util.FUSED_QUANTIZE_WIDE_OUT_BLOCK, util.FUSED_QUANTIZE_OUT_BLOCK, 512),
        )
    else:
        return None
    if tm is None or tn is None or _vmem_bytes(m, k, tm, tn) > VMEM_LIMIT_BYTES:
        return None
    return tm, tn


def _vmem_bytes(m: int, k: int, tm: int, tn: int) -> int:
    x = 2 * tm * k * 2  # bf16, double-buffered
    x_q = m * k + m * 128 * 4  # fp8 rows and their lane-padded f32 scales
    w = 2 * k * tn
    out = 2 * tm * tn * 2
    acc = 2 * tm * tn * 4  # f32 dot result and its scaled copy
    return x + x_q + w + out + acc


def _rms_norm_vmem_bytes(k: int, tm: int) -> int:
    """VMEM the RMSNorm adds: the normalized tile, its rsqrt and the weight."""
    normed = 2 * tm * k * 2  # bf16, two output buffers or scratch slots
    rstd = 2 * tm * 128 * 4  # lane-padded f32, double-buffered
    gamma = 2 * 8 * k * 4  # sublane-padded f32
    return normed + rstd + gamma


def _rms_norm_tiles(m: int, k: int, n: int) -> tuple[int, int] | None:
    tiles = _tiles(m, k, n)
    if tiles is None:
        return None
    tm, tn = tiles
    vmem = _vmem_bytes(m, k, tm, tn) + _rms_norm_vmem_bytes(k, tm)
    return tiles if vmem <= VMEM_LIMIT_BYTES else None


def supports(x: jax.Array, w_q: jax.Array, w_scale: jax.Array) -> bool:
    """Whether `fused_quantized_matmul` should take these operands."""
    m, k = x.shape
    n = w_q.shape[1]
    return (
        x.dtype == jnp.bfloat16
        and w_q.dtype == jnp.float8_e4m3fn
        and w_scale.shape == (n,)
        and k % QUANT_COLS == 0
        and _tiles(m, k, n) is not None
    )


def supports_rms_norm(
    x: jax.Array, gamma: jax.Array, w_q: jax.Array, w_scale: jax.Array
) -> bool:
    """Whether `fused_rms_norm_quantized_matmul` should take these operands."""
    m, k = x.shape
    return (
        gamma.dtype in (jnp.bfloat16, jnp.float32)
        and supports(x, w_q, w_scale)
        and _rms_norm_tiles(m, k, w_q.shape[1]) is not None
    )


def _normalize_rows(x_ref, rstd_ref, gamma_ref, x_normed_ref, r, weight_dtype):
    """Writes the RMSNorm of rows [r, r + QUANT_ROWS) of the x tile.

    Rounds `x * rstd` to the weight's dtype before the weight multiply, as
    vLLM's `rms_norm` spells it (`x.to(weight.dtype) * weight`); the served
    program keeps that rounding. The product of the two rounded factors is
    exact in f32, so the final cast rounds it once, like a bf16 multiply.
    """
    rows = pl.ds(r, QUANT_ROWS)
    rstd = rstd_ref[rows, :]
    for c in range(x_ref.shape[1] // QUANT_COLS):
        cols = pl.ds(c * QUANT_COLS, QUANT_COLS)
        x = x_ref[rows, cols].astype(jnp.float32)
        x_scaled = (x * rstd).astype(weight_dtype).astype(jnp.float32)
        x_normed = x_scaled * gamma_ref[:, cols]
        x_normed_ref[rows, cols] = x_normed.astype(x_normed_ref.dtype)


def _quantize_rows(x_ref, x_q_ref, x_scale_ref, i, r, f8_max):
    """Quantizes rows [r, r + QUANT_ROWS) of the x tile into slot i.

    The same arithmetic as `util.quantize_block` as XLA compiles it.
    """
    rows = pl.ds(r, QUANT_ROWS)
    num_cols = x_ref.shape[1] // QUANT_COLS

    def cols(c):
        return pl.ds(c * QUANT_COLS, QUANT_COLS)

    # Two streaming passes over the tile in VMEM, so a row slab is never held
    # in registers whole.
    abs_max = jnp.abs(x_ref[rows, cols(0)])
    for c in range(1, num_cols):
        abs_max = jnp.maximum(abs_max, jnp.abs(x_ref[rows, cols(c)]))
    abs_max = jnp.max(abs_max, axis=1, keepdims=True)
    # XLA folds `abs_max / f8_max` into a multiply by the f32 reciprocal;
    # spelling it out keeps the scale bit-identical.
    scale = abs_max.astype(jnp.float32) * jnp.float32(1.0 / f8_max)
    scale = jnp.where(scale == 0, 1.0, scale)
    x_scale_ref[i, rows, :] = scale
    for c in range(num_cols):
        x = x_ref[rows, cols(c)].astype(jnp.float32)
        x_q = jnp.clip(x / scale, -f8_max, f8_max)
        x_q_ref[i, rows, cols(c)] = x_q.astype(x_q_ref.dtype)


def _kernel(x_ref, w_ref, w_scale_ref, out_ref, x_q_ref, x_scale_ref, *, f8_max):
    _quantized_matmul_body(
        x_ref, w_ref, w_scale_ref, out_ref, x_q_ref, x_scale_ref, f8_max
    )


def _rms_norm_kernel(
    x_ref,
    rstd_ref,
    gamma_ref,
    w_ref,
    w_scale_ref,
    out_ref,
    x_normed_ref,
    x_q_ref,
    x_scale_ref,
    *,
    f8_max,
    weight_dtype,
):
    def normalize(r):
        _normalize_rows(x_ref, rstd_ref, gamma_ref, x_normed_ref, r, weight_dtype)

    _quantized_matmul_body(
        x_normed_ref,
        w_ref,
        w_scale_ref,
        out_ref,
        x_q_ref,
        x_scale_ref,
        f8_max,
        normalize,
    )


def _quantized_matmul_body(
    x_ref, w_ref, w_scale_ref, out_ref, x_q_ref, x_scale_ref, f8_max, normalize=None
):
    """Quantizes the x tile on the first column pass, then multiplies.

    `normalize(r)`, when given, writes rows [r, r + QUANT_ROWS) of `x_ref`
    just before they are quantized.
    """
    j, i = pl.program_id(0), pl.program_id(1)

    def matmul():
        acc = jnp.dot(x_q_ref[i], w_ref[...], preferred_element_type=jnp.float32)
        acc = acc * x_scale_ref[i]
        acc = acc * w_scale_ref[...]
        out_ref[...] = acc.astype(out_ref.dtype)

    # Unrolled, and in the same branch as the matmul, so the scheduler can
    # issue the quantize in the VALU slots of the MXU-bound matmul.
    @pl.when(j == 0)
    def _first_pass():
        for r in range(0, x_ref.shape[0], QUANT_ROWS):
            if normalize is not None:
                normalize(r)
            _quantize_rows(x_ref, x_q_ref, x_scale_ref, i, r, f8_max)
        matmul()

    @pl.when(j != 0)
    def _later_pass():
        matmul()


@jax.jit
def fused_quantized_matmul(
    x: jax.Array,  # [m, k] bfloat16
    w_q: jax.Array,  # [k, n] float8_e4m3fn
    w_scale: jax.Array,  # [n]
) -> jax.Array:
    """`xla_quantized_matmul(x, w_q, w_scale)` with the quantize kept in VMEM."""
    m, k = x.shape
    n = w_q.shape[1]
    tm, tn = _tiles(m, k, n)
    num_i, num_j = m // tm, n // tn
    f8_max = float(jnp.finfo(w_q.dtype).max)

    def x_index(j, i):
        # Only the first column pass reads x; later passes keep the last tile.
        return jnp.where(j == 0, i, num_i - 1), 0

    return pl.pallas_call(
        functools.partial(_kernel, f8_max=f8_max),
        grid=(num_j, num_i),
        in_specs=[
            pl.BlockSpec((tm, k), x_index),
            pl.BlockSpec((k, tn), lambda j, i: (0, j)),
            pl.BlockSpec((1, tn), lambda j, i: (0, j)),
        ],
        out_specs=pl.BlockSpec((tm, tn), lambda j, i: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        scratch_shapes=[
            pltpu.VMEM((num_i, tm, k), w_q.dtype),
            pltpu.VMEM((num_i, tm, 1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"),
            vmem_limit_bytes=VMEM_LIMIT_BYTES,
        ),
        name="fused_quantized_matmul",
    )(x, w_q, w_scale.astype(jnp.float32).reshape(1, n))


@functools.partial(jax.jit, static_argnames=("emit_normed",))
def fused_rms_norm_quantized_matmul(
    x: jax.Array,  # [m, k] bfloat16, the RMSNorm's input
    rstd: jax.Array,  # [m, 1] float32, rsqrt(mean(x**2) + eps)
    gamma: jax.Array,  # [k] bfloat16, the RMSNorm weight
    w_q: jax.Array,  # [k, n] float8_e4m3fn
    w_scale: jax.Array,  # [n]
    *,
    emit_normed: bool,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """`xla_quantized_matmul` of `x` normalized in VMEM.

    Returns the matmul output, and with `emit_normed` also the normalized
    activation, `bf16(gamma.dtype(x * rstd) * gamma)`.
    """
    m, k = x.shape
    n = w_q.shape[1]
    tiles = _rms_norm_tiles(m, k, n)
    if tiles is None:
        raise ValueError(f"Unsupported shape {(m, k, n)}; see `supports_rms_norm`.")
    tm, tn = tiles
    num_i, num_j = m // tm, n // tn
    f8_max = float(jnp.finfo(w_q.dtype).max)

    def x_index(j, i):
        # Only the first column pass reads x; later passes keep the last tile.
        return jnp.where(j == 0, i, num_i - 1), 0

    out_spec = pl.BlockSpec((tm, tn), lambda j, i: (i, j))
    out_shape = jax.ShapeDtypeStruct((m, n), x.dtype)
    x_normed_shape = jax.ShapeDtypeStruct((m, k), x.dtype)
    scratch_shapes = [
        pltpu.VMEM((num_i, tm, k), w_q.dtype),
        pltpu.VMEM((num_i, tm, 1), jnp.float32),
    ]
    kernel = functools.partial(
        _rms_norm_kernel, f8_max=f8_max, weight_dtype=gamma.dtype
    )
    if emit_normed:
        out_specs = (out_spec, pl.BlockSpec((tm, k), x_index))
        out_shapes = (out_shape, x_normed_shape)
    else:
        out_specs, out_shapes = out_spec, out_shape
        scratch_shapes.append(pltpu.VMEM((2, tm, k), x.dtype))
        kernel = _normed_in_scratch(kernel)

    return pl.pallas_call(
        kernel,
        grid=(num_j, num_i),
        in_specs=[
            pl.BlockSpec((tm, k), x_index),
            pl.BlockSpec((tm, 1), x_index),
            pl.BlockSpec((1, k), lambda j, i: (0, 0)),
            pl.BlockSpec((k, tn), lambda j, i: (0, j)),
            pl.BlockSpec((1, tn), lambda j, i: (0, j)),
        ],
        out_specs=out_specs,
        out_shape=out_shapes,
        scratch_shapes=scratch_shapes,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"),
            vmem_limit_bytes=VMEM_LIMIT_BYTES,
        ),
        name="fused_rms_norm_quantized_matmul",
    )(
        x,
        rstd,
        gamma.astype(jnp.float32).reshape(1, k),
        w_q,
        w_scale.astype(jnp.float32).reshape(1, n),
    )


def _normed_in_scratch(kernel):
    """Hands the kernel a scratch slot where the normalized output would be.

    The slot alternates with the row tile: at one fixed address the compiler
    forwards the normalized rows from their store to the quantize's loads and
    spills (5983 against 4540 bundles per first-pass step at 1024x4096x1536).
    """

    def wrapped(*refs):
        *ins_out, x_q_ref, x_scale_ref, x_normed_ref = refs
        x_normed_ref = x_normed_ref.at[pl.program_id(1) % 2]
        return kernel(*ins_out, x_normed_ref, x_q_ref, x_scale_ref)

    return wrapped
