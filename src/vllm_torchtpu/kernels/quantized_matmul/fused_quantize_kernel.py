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

`fused_quantized_matmul_rope_quant` takes the consumer instead: it rotates and
quantizes each 128-channel head of the output in VMEM, as `rope.rope_quant`
does, and writes the quantized heads in the `[tokens, heads, 128]` layout.
`fused_quantized_matmul_qnorm_rope` likewise normalizes and rotates each head
of the output, as `rope.qnorm_rope` does, before it leaves VMEM.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import rope
from vllm_torchtpu.kernels.quantized_matmul import util

# Tile sizes and the shape gate are tunables in `util` (FUSED_QUANTIZE_*), read
# at trace time.
# Columns per quantize step; rows per step are one fp8 vreg's 32 sublanes.
QUANT_COLS = 512
QUANT_ROWS = 32
VMEM_LIMIT_BYTES = 48 * 1024 * 1024
# `fused_quantized_matmul_rope_quant` rotates and quantizes this many rows of a
# head at a time, and swaps the lane pairs of the heads this far ahead of them.
ROPE_ROWS = 128
ROTATE_AHEAD = 8


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


def _rope_quant_token_block(m: int) -> int | None:
    tm = util.FUSED_ROPE_QUANT_TOKEN_BLOCK
    while tm >= QUANT_ROWS:
        if m % tm == 0 and tm % QUANT_ROWS == 0:
            return tm
        tm //= 2
    return None


def _rope_quant_vmem_bytes(k: int, num_heads: int, tm: int) -> int:
    n = num_heads * rope.LANE
    x = 2 * tm * k * 2  # bf16, double-buffered
    x_q = tm * k + tm * 128 * 4  # fp8 rows and their lane-padded f32 scales
    w = 2 * k * n
    q = 2 * tm * n  # fp8 heads, double-buffered
    scales = 2 * tm * pl.cdiv(num_heads, 128) * 128 * 4
    heads = 2 * tm * n * 2  # bf16 heads and their lane-swapped copy
    stage = tm * (num_heads + 1) * rope.LANE * 4
    cos_sin = 2 * tm * 128 * 4 + 2 * tm * rope.LANE * 4
    acc = 2 * tm * util.FUSED_ROPE_QUANT_HEADS_PER_DOT * rope.LANE * 4
    return x + x_q + w + q + scales + heads + stage + cos_sin + acc


def supports_rope_quant(
    x: jax.Array, w_q: jax.Array, w_scale: jax.Array, num_heads: int
) -> bool:
    """Whether `fused_quantized_matmul_rope_quant` should take these operands."""
    m, k = x.shape
    n = w_q.shape[1]
    tm = _rope_quant_token_block(m)
    return (
        x.dtype == jnp.bfloat16
        and w_q.dtype == jnp.float8_e4m3fn
        and w_scale.shape == (n,)
        and k % QUANT_COLS == 0
        and n == num_heads * rope.LANE
        and num_heads % util.FUSED_ROPE_QUANT_HEADS_PER_DOT == 0
        and tm is not None
        and _rope_quant_vmem_bytes(k, num_heads, tm) <= VMEM_LIMIT_BYTES
    )


def _rope_quant_kernel(
    # scalar prefetch
    positions_ref,
    # inputs
    x_ref,
    w_ref,
    w_scale_ref,
    cos_sin_cache_ref,
    # outputs
    q_ref,
    scale_ref,
    # scratch
    x_q_ref,
    x_scale_ref,
    cos_sin_ref,
    cos_sin_sems,
    cos_ref,
    sin_ref,
    heads_ref,
    swapped_ref,
    stage_ref,
    *,
    f8_max,
    rotary_dim,
    quant_dtype,
):
    i, num_i = pl.program_id(0), pl.num_programs(0)
    tm, num_heads = q_ref.shape[:2]
    lane = rope.LANE
    heads_per_dot = util.FUSED_ROPE_QUANT_HEADS_PER_DOT
    rope_rows = math.gcd(ROPE_ROWS, tm)

    def fetch_cos_sin(tile, slot):
        # One row of the table per token, as `rope.CosSinRef` gathers them.
        @pl.loop(0, tm, unroll=True)
        def _(t):
            position = positions_ref[tile * tm + t]
            pltpu.make_async_copy(
                cos_sin_cache_ref.at[position, pl.ds(0, rotary_dim)],
                cos_sin_ref.at[slot, t, :],
                cos_sin_sems.at[slot],
            ).start()

    slot = i % 2

    @pl.when(i == 0)
    def _():
        fetch_cos_sin(0, 0)

    @pl.when(i + 1 < num_i)
    def _():
        fetch_cos_sin(i + 1, 1 - slot)

    for r in range(0, tm, QUANT_ROWS):
        _quantize_rows(x_ref, x_q_ref, x_scale_ref, 0, r, f8_max)

    cos_sin = cos_sin_ref.at[slot]
    pltpu.make_async_copy(cos_sin, cos_sin, cos_sin_sems.at[slot]).wait()
    cos, sin = rope._cos_sin_lanes(cos_sin[...], rotary_dim=rotary_dim, inverse=False)
    cos_ref[...] = cos
    # `rope._rotate_gptj` negates the even lanes of the swapped pairs; negating
    # sin there instead gives the same products, since rounding is symmetric.
    even = jax.lax.broadcasted_iota(jnp.int32, sin.shape, 1) % 2 == 0
    sin_ref[...] = sin * jnp.where(even, -1.0, 1.0).astype(sin.dtype)

    for g in range(num_heads // heads_per_dot):
        cols = pl.ds(g * heads_per_dot * lane, heads_per_dot * lane)
        acc = jnp.dot(x_q_ref[0], w_ref[:, cols], preferred_element_type=jnp.float32)
        acc = acc * x_scale_ref[0]
        acc = acc * w_scale_ref[:, cols]
        # rope_quant reads the bf16 heads `xla_quantized_matmul` returns.
        heads_ref[:, cols] = acc.astype(heads_ref.dtype)

    def swap_lane_pairs(rows, h):
        # On the packed bf16 words, so one lane permute moves two rows; doing
        # it ahead keeps it from queueing behind the abs-max lane reductions.
        head = pl.ds(h * lane, lane)
        words = pltpu.bitcast(heads_ref[rows, head], jnp.uint32)
        swapped = rope._lane_gather(words, jnp.arange(lane) ^ 1)
        swapped_ref[rows, head] = pltpu.bitcast(swapped, heads_ref.dtype)

    # Head h of token t goes to stage row t * stride + h. An odd stride puts the
    # 8 tokens of a vreg in 8 different sublanes, so each head is one strided
    # store and each token's heads one contiguous load.
    stride = num_heads + 1
    for r in range(0, tm, rope_rows):
        rows = pl.ds(r, rope_rows)
        for h in range(min(ROTATE_AHEAD, num_heads)):
            swap_lane_pairs(rows, h)
        for h in range(num_heads):
            if h + ROTATE_AHEAD < num_heads:
                swap_lane_pairs(rows, h + ROTATE_AHEAD)
            head = pl.ds(h * lane, lane)
            x_h = heads_ref[rows, head].astype(jnp.float32)
            swapped_h = swapped_ref[rows, head].astype(jnp.float32)
            # `rope._rope_body`'s rotation and quantize, one head at a time.
            y = x_h * cos_ref[rows, :] + swapped_h * sin_ref[rows, :]
            y_scaled, scale = rope._quantize_scale_rows(y, quant_dtype)
            stage_ref[pl.ds(r * stride + h, rope_rows, stride=stride), :] = y_scaled
            scale_ref[rows, h : h + 1] = scale
    for t in range(tm):
        q_ref[t] = stage_ref[pl.ds(t * stride, num_heads), :].astype(quant_dtype)


@functools.partial(jax.jit, static_argnames=("quant_dtype",))
def fused_quantized_matmul_rope_quant(
    x: jax.Array,  # [m, k] bfloat16
    w_q: jax.Array,  # [k, num_heads * 128] float8_e4m3fn
    w_scale: jax.Array,  # [num_heads * 128]
    positions: jax.Array,  # [m] int32
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    *,
    quant_dtype: jnp.dtype = jnp.float8_e4m3fn,
) -> tuple[jax.Array, jax.Array]:
    """`rope.rope_quant` of `xla_quantized_matmul(x, w_q, w_scale)` as heads.

    Returns the rotated heads quantized to `quant_dtype`, `[m, num_heads,
    128]`, and their per-head scales, `[m, num_heads]` float32.
    """
    m, k = x.shape
    n = w_q.shape[1]
    num_heads = n // rope.LANE
    rotary_dim = cos_sin_cache.shape[1]
    # What `rope.rope_quant` asserts, and `rope._cos_sin_lanes` relies on.
    assert rotary_dim % 2 == 0 and rotary_dim <= rope.LANE
    tm = _rope_quant_token_block(m)
    f8_max = float(jnp.finfo(w_q.dtype).max)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(m // tm,),
        in_specs=[
            pl.BlockSpec((tm, k), lambda i, positions: (i, 0)),
            pl.BlockSpec((k, n), lambda i, positions: (0, 0)),
            pl.BlockSpec((1, n), lambda i, positions: (0, 0)),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=[
            pl.BlockSpec((tm, num_heads, rope.LANE), lambda i, positions: (i, 0, 0)),
            pl.BlockSpec((tm, num_heads), lambda i, positions: (i, 0)),
        ],
        scratch_shapes=[
            pltpu.VMEM((1, tm, k), w_q.dtype),
            pltpu.VMEM((1, tm, 1), jnp.float32),
            pltpu.VMEM((2, tm, rotary_dim), cos_sin_cache.dtype),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.VMEM((tm, rope.LANE), cos_sin_cache.dtype),
            pltpu.VMEM((tm, rope.LANE), cos_sin_cache.dtype),
            pltpu.VMEM((tm, n), x.dtype),
            pltpu.VMEM((tm, n), x.dtype),
            pltpu.VMEM((tm * (num_heads + 1), rope.LANE), jnp.float32),
        ],
    )
    return pl.pallas_call(
        functools.partial(
            _rope_quant_kernel,
            f8_max=f8_max,
            rotary_dim=rotary_dim,
            quant_dtype=quant_dtype,
        ),
        grid_spec=grid_spec,
        out_shape=(
            jax.ShapeDtypeStruct((m, num_heads, rope.LANE), quant_dtype),
            jax.ShapeDtypeStruct((m, num_heads), jnp.float32),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=VMEM_LIMIT_BYTES,
            # The per-token cos/sin row copies are bounds-checked otherwise, at
            # ~5x the scalar work of issuing them (as in `rope_quant`).
            disable_bounds_checks=True,
        ),
        name="fused_quantized_matmul_rope_quant",
    )(positions, x, w_q, w_scale.astype(jnp.float32).reshape(1, n), cos_sin_cache)


def _qnorm_rope_vmem_bytes(m: int, k: int, tn: int, head_dim: int) -> int:
    rows = min(util.FUSED_QNORM_ROPE_ROWS, m)
    x = 2 * m * k * 2  # bf16, double-buffered
    x_q = m * k + m * 128 * 4  # fp8 rows and their lane-padded f32 scales
    w = util.FUSED_QNORM_ROPE_WEIGHT_SLOTS * k * tn
    out = 2 * m * tn * 2
    cos_sin = 3 * m * rope.LANE * 4  # gathered rows, cos and signed sin
    chunk = 4 * rows * head_dim * 4  # a row chunk of a head through the norm
    return x + x_q + w + out + cos_sin + chunk


def _qnorm_rope_out_block(m: int, k: int, n: int, head_dim: int) -> int | None:
    """The widest of the halvings of the out block that fits, in whole heads."""
    tn = util.FUSED_QNORM_ROPE_OUT_BLOCK
    while tn >= util.FUSED_QNORM_ROPE_MIN_OUT_BLOCK:
        if (
            tn % head_dim == 0
            and n % tn == 0
            and _qnorm_rope_vmem_bytes(m, k, tn, head_dim) <= VMEM_LIMIT_BYTES
        ):
            return tn
        tn //= 2
    return None


def supports_qnorm_rope(
    x: jax.Array, w_q: jax.Array, w_scale: jax.Array, head_dim: int
) -> bool:
    """Whether `fused_quantized_matmul_qnorm_rope` should take these operands."""
    m, k = x.shape
    n = w_q.shape[1]
    return (
        x.dtype == jnp.bfloat16
        and w_q.dtype == jnp.float8_e4m3fn
        and w_scale.shape == (n,)
        and k % QUANT_COLS == 0
        and k % util.FUSED_QNORM_ROPE_WEIGHT_COPIES == 0
        and m > 0
        and m % QUANT_ROWS == 0
        and m % min(util.FUSED_QNORM_ROPE_ROWS, m) == 0
        and head_dim % rope.LANE == 0
        and _qnorm_rope_out_block(m, k, n, head_dim) is not None
    )


def _qnorm_rope_kernel(
    # scalar prefetch
    positions_ref,
    # inputs
    x_ref,
    w_hbm_ref,
    w_scale_ref,
    cos_sin_cache_ref,
    # output
    out_ref,
    # scratch
    x_q_ref,
    x_scale_ref,
    cos_sin_ref,
    cos_sin_sem,
    cos_ref,
    sin_ref,
    w_slots_ref,
    w_sems,
    *,
    f8_max,
    rotary_dim,
    eps,
    head_dim,
    num_tiles,
):
    j = pl.program_id(0)
    m, tn = out_ref.shape
    k = x_ref.shape[1]
    lane = rope.LANE
    rows_per_chunk = min(util.FUSED_QNORM_ROPE_ROWS, m)
    num_slots = w_slots_ref.shape[0]
    w_ref = w_slots_ref.at[j % num_slots]

    # The weight streams through `num_slots` tiles, each fetched as several
    # copies: more copies in flight than the pipeline's one per block.
    def fetch_weight(tile):
        part = k // util.FUSED_QNORM_ROPE_WEIGHT_COPIES
        for p in range(util.FUSED_QNORM_ROPE_WEIGHT_COPIES):
            rows = pl.ds(p * part, part)
            pltpu.make_async_copy(
                w_hbm_ref.at[rows, pl.ds(tile * tn, tn)],
                w_slots_ref.at[tile % num_slots, rows],
                w_sems.at[tile % num_slots],
            ).start()

    def wait_weight():
        pltpu.make_async_copy(w_ref, w_ref, w_sems.at[j % num_slots]).wait()

    @pl.when(j == 0)
    def _():
        for tile in range(min(num_slots - 1, num_tiles)):
            fetch_weight(tile)

    @pl.when(j + num_slots - 1 < num_tiles)
    def _():
        fetch_weight(j + num_slots - 1)

    def matmul_qnorm_rope():
        # One head of a row chunk at a time, from the MXU result to the output,
        # so the scheduler overlaps each chunk's norm and rotation with the
        # next chunk's matmul.
        for h in range(tn // head_dim):
            cols = pl.ds(h * head_dim, head_dim)
            for r in range(0, m, rows_per_chunk):
                rows = pl.ds(r, rows_per_chunk)
                acc = jnp.dot(
                    x_q_ref[0, rows, :],
                    w_ref[:, cols],
                    preferred_element_type=jnp.float32,
                )
                acc = acc * x_scale_ref[0, rows, :]
                acc = acc * w_scale_ref[:, cols]
                # qnorm_rope reads the bf16 heads `xla_quantized_matmul`
                # returns; from here on this is `rope._qnorm_rope_body`.
                q = acc.astype(out_ref.dtype)
                x = q.astype(jnp.float32)
                rms = jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
                y = x * rms
                # The lane pairs are swapped on the packed bf16 words, so one
                # lane permute moves two rows; the norm's scaling commutes
                # with the swap.
                words = pltpu.bitcast(q[:, -lane:], jnp.uint32)
                swapped = rope._lane_gather(words, jnp.arange(lane) ^ 1)
                swapped = pltpu.bitcast(swapped, q.dtype).astype(jnp.float32)
                roped = (
                    y[:, -lane:] * cos_ref[rows, :] + (swapped * rms) * sin_ref[rows, :]
                )
                out_ref[rows, cols] = jnp.concatenate(
                    [y[:, :-lane], roped], axis=-1
                ).astype(out_ref.dtype)

    @pl.when(pl.program_id(0) == 0)
    def _first_pass():
        # One row of the table per token, as `rope.CosSinRef` gathers them.
        @pl.loop(0, m, unroll=True)
        def _(t):
            pltpu.make_async_copy(
                cos_sin_cache_ref.at[positions_ref[t], pl.ds(0, rotary_dim)],
                cos_sin_ref.at[t, :],
                cos_sin_sem,
            ).start()

        for r in range(0, m, QUANT_ROWS):
            _quantize_rows(x_ref, x_q_ref, x_scale_ref, 0, r, f8_max)

        pltpu.make_async_copy(cos_sin_ref, cos_sin_ref, cos_sin_sem).wait()
        cos, sin = rope._cos_sin_lanes(
            cos_sin_ref[...], rotary_dim=rotary_dim, inverse=False
        )
        cos_ref[...] = cos
        # `rope._rotate_gptj` negates the even lanes of the swapped pairs;
        # negating sin there instead gives the same products, since rounding
        # is symmetric.
        even = jax.lax.broadcasted_iota(jnp.int32, sin.shape, 1) % 2 == 0
        sin_ref[...] = sin * jnp.where(even, -1.0, 1.0).astype(sin.dtype)
        # After the quantize, which hides the first tile's fetch.
        wait_weight()
        matmul_qnorm_rope()

    @pl.when(pl.program_id(0) != 0)
    def _later_pass():
        wait_weight()
        matmul_qnorm_rope()


@functools.partial(jax.jit, static_argnames=("head_dim", "eps"))
def fused_quantized_matmul_qnorm_rope(
    x: jax.Array,  # [m, k] bfloat16
    w_q: jax.Array,  # [k, num_heads * head_dim] float8_e4m3fn
    w_scale: jax.Array,  # [num_heads * head_dim]
    positions: jax.Array,  # [m] int32
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    *,
    head_dim: int,
    eps: float,
) -> jax.Array:
    """`rope.qnorm_rope` of `xla_quantized_matmul(x, w_q, w_scale)`.

    Returns the normalized and rotated heads, `[m, num_heads * head_dim]`
    bfloat16.
    """
    m, k = x.shape
    n = w_q.shape[1]
    rotary_dim = cos_sin_cache.shape[1]
    # What `rope.qnorm_rope` asserts, and `rope._cos_sin_lanes` relies on.
    assert rotary_dim % 2 == 0 and rotary_dim <= rope.LANE
    if not supports_qnorm_rope(x, w_q, w_scale, head_dim):
        raise ValueError(f"Unsupported shape {(m, k, n)}; see `supports_qnorm_rope`.")
    tn = _qnorm_rope_out_block(m, k, n, head_dim)
    f8_max = float(jnp.finfo(w_q.dtype).max)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(n // tn,),
        in_specs=[
            pl.BlockSpec((m, k), lambda j, positions: (0, 0)),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec((1, tn), lambda j, positions: (0, j)),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=pl.BlockSpec((m, tn), lambda j, positions: (0, j)),
        scratch_shapes=[
            pltpu.VMEM((1, m, k), w_q.dtype),
            pltpu.VMEM((1, m, 1), jnp.float32),
            pltpu.VMEM((m, rotary_dim), cos_sin_cache.dtype),
            pltpu.SemaphoreType.DMA(()),
            pltpu.VMEM((m, rope.LANE), cos_sin_cache.dtype),
            pltpu.VMEM((m, rope.LANE), cos_sin_cache.dtype),
            pltpu.VMEM((util.FUSED_QNORM_ROPE_WEIGHT_SLOTS, k, tn), w_q.dtype),
            pltpu.SemaphoreType.DMA((util.FUSED_QNORM_ROPE_WEIGHT_SLOTS,)),
        ],
    )
    return pl.pallas_call(
        functools.partial(
            _qnorm_rope_kernel,
            f8_max=f8_max,
            rotary_dim=rotary_dim,
            eps=eps,
            head_dim=head_dim,
            num_tiles=n // tn,
        ),
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=VMEM_LIMIT_BYTES,
            # As in `fused_quantized_matmul_rope_quant`.
            disable_bounds_checks=True,
        ),
        name="fused_quantized_matmul_qnorm_rope",
    )(positions, x, w_q, w_scale.astype(jnp.float32).reshape(1, n), cos_sin_cache)
