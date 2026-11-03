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


def _out_block(n: int, split_cols: int) -> int | None:
    # Each chunk lands wholly in one output.
    for block in range(OUT_BLOCK, 0, -128):
        if n % block == 0 and split_cols % block == 0:
            return block
    return None


def _vmem_limit_bytes(k: int, n: int, out_block: int) -> int:
    x = 2 * ROW_BLOCK * k * 2  # double-buffered bf16 input block
    x_q = 2 * ROW_BLOCK * k  # two fp8 slots
    x_scale = 2 * ROW_BLOCK * 128 * 4  # (rows, 1) f32 pads to 128 lanes
    w = k * n + 8 * n * 4  # resident fp8 weight; its (1, n) scale pads to 8 rows
    out = 2 * ROW_BLOCK * n * 2  # double-buffered bf16 output blocks
    acc = ROW_BLOCK * out_block * 4
    return int(VMEM_HEADROOM * (x + x_q + x_scale + w + out + acc))


def supports(
    x: jax.Array, w_q: jax.Array, w_scale: jax.Array, split_cols: int = 0
) -> bool:
    """Whether `dynamic_quant_matmul` takes these operands (static shapes)."""
    if x.ndim != 2 or w_q.ndim != 2 or w_scale.shape != w_q.shape[1:]:
        return False
    if x.dtype != jnp.bfloat16 or w_q.dtype != jnp.float8_e4m3fn:
        return False
    m, k = x.shape
    n = w_q.shape[1]
    if m == 0 or m % ROW_BLOCK or k % 128 or not 0 <= split_cols < n:
        return False
    out_block = _out_block(n, split_cols)
    if out_block is None:
        return False
    return _vmem_limit_bytes(k, n, out_block) <= min(DEVICE_VMEM_LIMIT.values())


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


def _kernel(x_ref, w_ref, w_scale_ref, *refs, out_block, n_first, w_nk):
    # One output, or two when the trailing columns from `n_first` on are split
    # off. w_ref is (n, k) when `w_nk`, else (k, n).
    *out_refs, x_q_ref, x_scale_ref = refs

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
        n_chunks = w_scale_ref.shape[1] // out_block
        per_chunk = -(-len(quant_starts) // n_chunks)
        x_q = x_q_ref.at[(s + 1) % 2]
        x_scale = x_scale_ref[(s + 1) % 2]
        # One basic block, so the scheduler interleaves the quantization of
        # the next row block with these matmuls. The last step re-quantizes
        # the last block into the free slot rather than branch.
        for j in range(n_chunks):
            col = j * out_block
            cols = pl.ds(col, out_block)
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
            acc = acc * w_scale_ref[:, cols]
            if col < n_first:
                out_refs[0][:, cols] = acc.astype(out_refs[0].dtype)
            else:
                part = pl.ds(col - n_first, out_block)
                out_refs[1][:, part] = acc.astype(out_refs[1].dtype)
            quantize(quant_starts[j * per_chunk : (j + 1) * per_chunk])


@functools.partial(jax.jit, static_argnames=("split_cols", "weight_nk"))
def dynamic_quant_matmul(
    x: jax.Array,  # [m, k] bf16
    w_q: jax.Array,  # [k, n] fp8
    w_scale: jax.Array,  # [n]
    split_cols: int = 0,
    weight_nk: bool = False,
) -> jax.Array:
    """`x @ w_q` with per-row dynamic FP8 quantization of `x`, as [m, n].

    With `split_cols`, the last that many columns are written as their own
    [m, split_cols] array and the result is the concatenation of the two,
    which XLA folds away when the consumers slice at the split. `weight_nk`
    says `w_q` is the transpose of a stored [n, k] array, which the kernel
    then reads as is.
    """
    if not supports(x, w_q, w_scale, split_cols):
        raise ValueError(f"unsupported operands {x.shape=} {x.dtype=} {w_q.shape=}")
    m, k = x.shape
    n = w_q.shape[1]
    n_first = n - split_cols
    out_block = _out_block(n, split_cols)
    nb = m // ROW_BLOCK
    if weight_nk:
        w_q = w_q.T
    widths = [n_first, split_cols] if split_cols else [n]
    outs = pl.pallas_call(
        functools.partial(
            _kernel, out_block=out_block, n_first=n_first, w_nk=weight_nk
        ),
        grid=(nb + 1,),
        in_specs=[
            pl.BlockSpec((ROW_BLOCK, k), lambda s: (jnp.minimum(s, nb - 1), 0)),
            pl.BlockSpec(w_q.shape, lambda s: (0, 0)),
            pl.BlockSpec((1, n), lambda s: (0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((ROW_BLOCK, w), lambda s: (jnp.maximum(s - 1, 0), 0))
            for w in widths
        ],
        out_shape=[jax.ShapeDtypeStruct((m, w), x.dtype) for w in widths],
        scratch_shapes=[
            pltpu.VMEM((2, ROW_BLOCK, k), w_q.dtype),
            pltpu.VMEM((2, ROW_BLOCK, 1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=_vmem_limit_bytes(k, n, out_block),
        ),
    )(x, w_q, w_scale.astype(jnp.float32).reshape(1, n))
    return outs[0] if len(outs) == 1 else jnp.concatenate(outs, axis=1)
