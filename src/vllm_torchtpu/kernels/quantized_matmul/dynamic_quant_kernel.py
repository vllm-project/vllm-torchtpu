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
"""Per-channel FP8 matmul that quantizes the activation inside the kernel.

Computes exactly what `util.xla_quantized_matmul` computes for FP8 weights
(per-row absmax scale, `where(scale == 0, 1, scale)`, divide, clip, FP8 dot
with f32 accumulation, `acc * x_scale * w_scale`), but reads the bf16
activation from HBM once instead of twice and keeps its FP8 copy in VMEM
instead of round-tripping it through HBM. Row block `s` is quantized while the
MXU multiplies row block `s - 1`, so the quantization runs on the vector units
under the matmul.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.quantized_matmul.tuned_block_sizes import (
    DEVICE_VMEM_LIMIT,
)

# 512 x 512 output chunks keep the MXU busy at the served shapes; 32 rows is
# one fp8 sublane tile per quantization step.
ROW_BLOCK = 512
OUT_BLOCK = 512
QUANT_ROWS = 32
# Mosaic needs VMEM beyond the operand buffers for the matmul's f32 result and
# register spills.
VMEM_HEADROOM = 1.5


def _out_block(n: int, token_minor_cols: int) -> int | None:
    # Each chunk lands wholly in the row-major or the token-minor part.
    for block in range(OUT_BLOCK, 0, -128):
        if n % block == 0 and token_minor_cols % block == 0:
            return block
    return None


def _vmem_limit_bytes(k: int, n: int, out_block: int, token_minor_cols: int) -> int:
    x = 2 * ROW_BLOCK * k * 2  # double-buffered bf16 input block
    x_q = 2 * ROW_BLOCK * k  # two fp8 slots
    x_scale = 2 * ROW_BLOCK * 128 * 4  # (rows, 1) f32 pads to 128 lanes
    # Resident fp8 weight and its scales: (n_tm, 1) pads to 128 lanes,
    # (1, n_rm) to 8 sublanes.
    w = k * n + (8 * (n - token_minor_cols) + 128 * token_minor_cols) * 4
    out = 2 * ROW_BLOCK * n * 2  # double-buffered bf16 output blocks
    acc = ROW_BLOCK * out_block * 4
    return int(VMEM_HEADROOM * (x + x_q + x_scale + w + out + acc))


def supports(
    x: jax.Array, w_q: jax.Array, w_scale: jax.Array, token_minor_cols: int = 0
) -> bool:
    """Whether `dynamic_quant_matmul` takes these operands (static shapes)."""
    if x.ndim != 2 or w_q.ndim != 2 or w_scale.shape != w_q.shape[1:]:
        return False
    if x.dtype != jnp.bfloat16 or w_q.dtype != jnp.float8_e4m3fn:
        return False
    m, k = x.shape
    n = w_q.shape[1]
    if m == 0 or m % ROW_BLOCK or k % 128 or not 0 <= token_minor_cols <= n:
        return False
    out_block = _out_block(n, token_minor_cols)
    if out_block is None:
        return False
    vmem = _vmem_limit_bytes(k, n, out_block, token_minor_cols)
    return vmem <= min(DEVICE_VMEM_LIMIT.values())


def _quantize_rows(x_ref, x_q_ref, x_scale_ref, rows):
    q_dtype = x_q_ref.dtype
    q_max = float(jnp.finfo(q_dtype).max)
    q_min = float(jnp.finfo(q_dtype).min)
    abs_max = jnp.max(jnp.abs(x_ref[rows, :]), axis=1, keepdims=True)
    # XLA folds `abs_max / q_max` into this multiply; matching it keeps the
    # scales bit-identical.
    scale = abs_max.astype(jnp.float32) * jnp.float32(1.0 / q_max)
    scale = jnp.where(scale == 0, 1.0, scale)
    x_scale_ref[rows, :] = scale
    x = x_ref[rows, :].astype(jnp.float32)
    x_q_ref[rows, :] = jnp.clip(x / scale, q_min, q_max).astype(q_dtype)


def _kernel(x_ref, w_ref, *refs, out_block, n_rm, n_tm):
    # Operands present only for a non-empty part: the row-major part's (1, n_rm)
    # scale and (rows, n_rm) output, the token-minor part's (n_tm, 1) scale and
    # (n_tm, rows) output.
    refs = list(refs)
    rm_scale_ref = refs.pop(0) if n_rm else None
    tm_scale_ref = refs.pop(0) if n_tm else None
    rm_out_ref = refs.pop(0) if n_rm else None
    tm_out_ref = refs.pop(0) if n_tm else None
    x_q_ref, x_scale_ref = refs
    # w_ref is (n, k) when any part is token-minor, else (k, n).
    w_nk = n_tm > 0

    # Step s quantizes row block s into slot s % 2 and multiplies row block
    # s - 1 from the other slot.
    s = pl.program_id(0)
    quant_starts = range(0, x_ref.shape[0], QUANT_ROWS)

    def quantize(starts):
        for r in starts:
            _quantize_rows(
                x_ref, x_q_ref.at[s % 2], x_scale_ref.at[s % 2], pl.ds(r, QUANT_ROWS)
            )

    @pl.when(s == 0)
    def _():
        quantize(quant_starts)

    @pl.when(s > 0)
    def _():
        n_chunks = (n_rm + n_tm) // out_block
        per_chunk = -(-len(quant_starts) // n_chunks)
        x_q = x_q_ref.at[(s + 1) % 2]
        x_scale = x_scale_ref[(s + 1) % 2]
        if n_tm:
            # The (n_tm, rows) output block scales its columns, so the per-row
            # scales go lane-major once per block.
            rows = x_scale.shape[0]
            x_scale_t = jnp.transpose(jnp.broadcast_to(x_scale, (rows, 128)))[:1]
        # One basic block, so the scheduler interleaves the quantization of
        # the next row block with these matmuls. The last step re-quantizes
        # the last block into the free slot rather than branch.
        for j in range(n_chunks):
            cols = pl.ds(j * out_block, out_block)
            if j * out_block >= n_rm:
                # Contract both operands on k for an (out_block, rows) slice
                # of the transposed output.
                part = pl.ds(j * out_block - n_rm, out_block)
                acc = jax.lax.dot_general(
                    w_ref[cols, :],
                    x_q[...],
                    (((1,), (1,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
                acc = acc * x_scale_t
                acc = acc * tm_scale_ref[part, :]
                tm_out_ref[part, :] = acc.astype(tm_out_ref.dtype)
            else:
                if w_nk:
                    acc = jax.lax.dot_general(
                        x_q[...],
                        w_ref[cols, :],
                        (((1,), (1,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                else:
                    acc = jnp.dot(
                        x_q[...], w_ref[:, cols], preferred_element_type=jnp.float32
                    )
                acc = acc * x_scale
                acc = acc * rm_scale_ref[:, cols]
                rm_out_ref[:, cols] = acc.astype(rm_out_ref.dtype)
            quantize(quant_starts[j * per_chunk : (j + 1) * per_chunk])


@functools.partial(jax.jit, static_argnames=("token_minor_cols",))
def dynamic_quant_matmul(
    x: jax.Array,  # [m, k] bf16
    w_q: jax.Array,  # [k, n] fp8
    w_scale: jax.Array,  # [n]
    token_minor_cols: int = 0,
) -> jax.Array:
    """`x @ w_q` with per-row dynamic FP8 quantization of `x`, as [m, n].

    The last `token_minor_cols` output columns are written as an [n_tm, m]
    row-major array and returned transposed, so a consumer that reads those
    columns token-minor gets them through a layout change instead of a
    transposing copy; the other columns are written [m, n_rm] row-major. A
    split result is the concatenation of the two parts, which XLA folds away
    when the consumers slice at the split.
    """
    if not supports(x, w_q, w_scale, token_minor_cols):
        raise ValueError(f"unsupported operands {x.shape=} {x.dtype=} {w_q.shape=}")
    m, k = x.shape
    n = w_q.shape[1]
    n_tm = token_minor_cols
    n_rm = n - n_tm
    out_block = _out_block(n, n_tm)
    nb = m // ROW_BLOCK
    w_scale = w_scale.astype(jnp.float32)
    rm_out_spec = pl.BlockSpec((ROW_BLOCK, n_rm), lambda s: (jnp.maximum(s - 1, 0), 0))
    tm_out_spec = pl.BlockSpec((n_tm, ROW_BLOCK), lambda s: (0, jnp.maximum(s - 1, 0)))
    if n_tm:
        w_q = w_q.T
    in_specs = [
        pl.BlockSpec((ROW_BLOCK, k), lambda s: (jnp.minimum(s, nb - 1), 0)),
        pl.BlockSpec(w_q.shape, lambda s: (0, 0)),
    ]
    operands = [x, w_q]
    out_specs, out_shapes = [], []
    if n_rm:
        in_specs.append(pl.BlockSpec((1, n_rm), lambda s: (0, 0)))
        operands.append(w_scale[:n_rm].reshape(1, n_rm))
        out_specs.append(rm_out_spec)
        out_shapes.append(jax.ShapeDtypeStruct((m, n_rm), x.dtype))
    if n_tm:
        in_specs.append(pl.BlockSpec((n_tm, 1), lambda s: (0, 0)))
        operands.append(w_scale[n_rm:].reshape(n_tm, 1))
        out_specs.append(tm_out_spec)
        out_shapes.append(jax.ShapeDtypeStruct((n_tm, m), x.dtype))
    outs = pl.pallas_call(
        functools.partial(_kernel, out_block=out_block, n_rm=n_rm, n_tm=n_tm),
        grid=(nb + 1,),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shapes,
        scratch_shapes=[
            pltpu.VMEM((2, ROW_BLOCK, k), w_q.dtype),
            pltpu.VMEM((2, ROW_BLOCK, 1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=_vmem_limit_bytes(k, n, out_block, n_tm),
        ),
    )(*operands)
    parts = []
    if n_rm:
        parts.append(outs[0])
    if n_tm:
        parts.append(outs[-1].T)
    return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=1)
