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

Given an RMSNorm weight, the kernel takes the norm's input instead and
normalizes each row before quantizing it, so the normalized activation is
never written to HBM.
"""

import functools
import math

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
# Lanes per step of a fused RMSNorm's passes over a row chunk: a (32, 256) f32
# slice is 8 vregs, so a pass never holds a whole row in registers.
NORM_LANES = 256
# Mosaic needs VMEM beyond the operand buffers for the matmul's f32 result and
# register spills.
VMEM_HEADROOM = 1.5


def _out_block(n: int, split_cols: int) -> int | None:
    # Each chunk lands wholly in one output.
    for block in range(OUT_BLOCK, 0, -128):
        if n % block == 0 and split_cols % block == 0:
            return block
    return None


def _vmem_limit_bytes(
    k: int,
    n: int,
    out_block: int,
    split_cols: int = 0,
    split_head_dim: int = 0,
    norm: bool = False,
) -> int:
    x = 2 * ROW_BLOCK * k * 2  # double-buffered bf16 input block
    # A fused RMSNorm's resident (1, k) f32 weight, padded to 8 sublanes, and
    # its bf16 output for one quantization step.
    x_norm = 8 * k * 4 + QUANT_ROWS * k * 2 if norm else 0
    x_q = 2 * ROW_BLOCK * k  # two fp8 slots
    x_scale = 2 * ROW_BLOCK * 128 * 4  # (rows, 1) f32 pads to 128 lanes
    w = k * n + 8 * n * 4  # resident fp8 weight; its (1, n) scale pads to 8 rows
    if split_head_dim:
        # A (rows, heads, head_dim) bf16 block pads heads to 16 sublanes.
        heads = -(-split_cols // split_head_dim // 16) * 16
        n += heads * split_head_dim - split_cols
    out = 2 * ROW_BLOCK * n * 2  # double-buffered bf16 output blocks
    acc = ROW_BLOCK * out_block * 4
    return int(VMEM_HEADROOM * (x + x_norm + x_q + x_scale + w + out + acc))


def supports(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    split_cols: int = 0,
    split_head_dim: int = 0,
    norm_weight: jax.Array | None = None,
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
    if norm_weight is not None and norm_weight.shape != (k,):
        return False
    out_block = _out_block(n, split_cols)
    if out_block is None:
        return False
    if split_head_dim and (
        not split_cols or split_head_dim % 128 or out_block % split_head_dim
    ):
        return False
    vmem = _vmem_limit_bytes(
        k, n, out_block, split_cols, split_head_dim, norm_weight is not None
    )
    return vmem <= min(DEVICE_VMEM_LIMIT.values())


def _rms_norm_rows(x_ref, norm_weight_ref, out_ref, rows, eps):
    # `util.rms_norm` of x_ref[rows] into out_ref, rounded to bf16 as the
    # norm's output would be; returns the rows' abs-max.
    k = x_ref.shape[1]
    step = math.gcd(k, NORM_LANES)
    lanes = [pl.ds(c, step) for c in range(0, k, step)]
    sum_sq = None
    for cols in lanes:
        x = x_ref[rows, cols].astype(jnp.float32)
        sum_sq = x * x if sum_sq is None else sum_sq + x * x
    scale = jax.lax.rsqrt(jnp.sum(sum_sq, axis=1, keepdims=True) / k + eps)
    abs_max = None
    for cols in lanes:
        x = x_ref[rows, cols].astype(jnp.float32) * scale
        x = x * norm_weight_ref[:, cols]
        out_ref[:, cols] = x.astype(out_ref.dtype)
        a = jnp.abs(x)
        abs_max = a if abs_max is None else jnp.maximum(abs_max, a)
    # Rounding is monotonic, so this is the abs-max of the rounded rows.
    return jnp.max(abs_max, axis=1, keepdims=True).astype(out_ref.dtype)


def _quantize_normed_rows(normed_ref, abs_max, x_q_ref, x_scale_ref):
    q_dtype = x_q_ref.dtype
    q_max = float(jnp.finfo(q_dtype).max)
    q_min = float(jnp.finfo(q_dtype).min)
    scale = abs_max.astype(jnp.float32) * jnp.float32(1.0 / q_max)
    scale = jnp.where(scale == 0, 1.0, scale)
    x_scale_ref[...] = scale
    x = normed_ref[...].astype(jnp.float32)
    x_q_ref[...] = jnp.clip(x / scale, q_min, q_max).astype(q_dtype)


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


def _kernel(*refs, out_block, n_first, w_nk, head_dim, norm_eps):
    # One output, or two when the trailing columns from `n_first` on are split
    # off, the second as (rows, heads, head_dim) when `head_dim`. w_ref is
    # (n, k) when `w_nk`, else (k, n). A fused RMSNorm adds its (1, k) weight
    # after x and its (QUANT_ROWS, k) output scratch last.
    norm = norm_eps is not None
    refs = list(refs)
    x_ref = refs.pop(0)
    norm_weight_ref = refs.pop(0) if norm else None
    w_ref, w_scale_ref = refs.pop(0), refs.pop(0)
    normed_ref = refs.pop() if norm else None
    *out_refs, x_q_ref, x_scale_ref = refs

    # Step s quantizes row block s into slot s % 2 and multiplies row block
    # s - 1 from the other slot.
    s = pl.program_id(0)
    quant_starts = range(0, x_ref.shape[0], QUANT_ROWS)

    def quantize(starts):
        for r in starts:
            rows = pl.ds(r, QUANT_ROWS)
            x_q, x_scale = x_q_ref.at[s % 2], x_scale_ref.at[s % 2]
            if norm:
                abs_max = _rms_norm_rows(
                    x_ref, norm_weight_ref, normed_ref, rows, norm_eps
                )
                _quantize_normed_rows(
                    normed_ref, abs_max, x_q.at[rows], x_scale.at[rows]
                )
            else:
                _quantize_rows(x_ref, x_q, x_scale, rows)

    @pl.when(s == 0)
    def _():
        quantize(quant_starts)

    @pl.when(s > 0)
    def _():
        n_chunks = w_scale_ref.shape[1] // out_block
        n_quant = len(quant_starts)
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
            elif head_dim:
                heads = pl.ds((col - n_first) // head_dim, out_block // head_dim)
                acc = acc.astype(out_refs[1].dtype)
                out_refs[1][:, heads, :] = acc.reshape(
                    acc.shape[0], out_block // head_dim, head_dim
                )
            else:
                part = pl.ds(col - n_first, out_block)
                out_refs[1][:, part] = acc.astype(out_refs[1].dtype)
            # Spread evenly: ceil-sized shares would leave the last chunks
            # with none (16 steps over 5 chunks: 4, 4, 4, 4, 0).
            quantize(
                quant_starts[j * n_quant // n_chunks : (j + 1) * n_quant // n_chunks]
            )


@functools.partial(
    jax.jit,
    static_argnames=("split_cols", "weight_nk", "split_head_dim", "norm_eps"),
)
def dynamic_quant_matmul(
    x: jax.Array,  # [m, k] bf16
    w_q: jax.Array,  # [k, n] fp8
    w_scale: jax.Array,  # [n]
    split_cols: int = 0,
    weight_nk: bool = False,
    split_head_dim: int = 0,
    norm_weight: jax.Array | None = None,  # [k]
    norm_eps: float = 0.0,
) -> jax.Array:
    """`x @ w_q` with per-row dynamic FP8 quantization of `x`, as [m, n].

    With `norm_weight`, `x` is the input of an RMSNorm and the product is
    `util.rms_norm(x, norm_weight, norm_eps) @ w_q`.

    With `split_cols`, the last that many columns are written as their own
    [m, split_cols] array and the result is the concatenation of the two,
    which XLA folds away when the consumers slice at the split. With
    `split_head_dim` too, that array is written as [m, heads, split_head_dim],
    so a consumer that reshapes the split-off columns into heads reads the
    kernel's output as is rather than through a relayout. `weight_nk` says
    `w_q` is the transpose of a stored [n, k] array, which the kernel then
    reads as is.
    """
    if not supports(x, w_q, w_scale, split_cols, split_head_dim, norm_weight):
        raise ValueError(f"unsupported operands {x.shape=} {x.dtype=} {w_q.shape=}")
    m, k = x.shape
    n = w_q.shape[1]
    n_first = n - split_cols
    out_block = _out_block(n, split_cols)
    nb = m // ROW_BLOCK
    if weight_nk:
        w_q = w_q.T
    widths = [n_first, split_cols] if split_cols else [n]
    out_specs = [
        pl.BlockSpec((ROW_BLOCK, w), lambda s: (jnp.maximum(s - 1, 0), 0))
        for w in widths
    ]
    out_shape = [jax.ShapeDtypeStruct((m, w), x.dtype) for w in widths]
    if split_head_dim:
        heads = (split_cols // split_head_dim, split_head_dim)
        out_specs[1] = pl.BlockSpec(
            (ROW_BLOCK, *heads), lambda s: (jnp.maximum(s - 1, 0), 0, 0)
        )
        out_shape[1] = jax.ShapeDtypeStruct((m, *heads), x.dtype)
    norm = norm_weight is not None
    in_specs = [pl.BlockSpec((ROW_BLOCK, k), lambda s: (jnp.minimum(s, nb - 1), 0))]
    operands = [x]
    scratch_shapes = [
        pltpu.VMEM((2, ROW_BLOCK, k), w_q.dtype),
        pltpu.VMEM((2, ROW_BLOCK, 1), jnp.float32),
    ]
    if norm:
        in_specs.append(pl.BlockSpec((1, k), lambda s: (0, 0)))
        operands.append(norm_weight.astype(jnp.float32).reshape(1, k))
        scratch_shapes.append(pltpu.VMEM((QUANT_ROWS, k), x.dtype))
    in_specs += [
        pl.BlockSpec(w_q.shape, lambda s: (0, 0)),
        pl.BlockSpec((1, n), lambda s: (0, 0)),
    ]
    operands += [w_q, w_scale.astype(jnp.float32).reshape(1, n)]
    outs = pl.pallas_call(
        functools.partial(
            _kernel,
            out_block=out_block,
            n_first=n_first,
            w_nk=weight_nk,
            head_dim=split_head_dim,
            norm_eps=float(norm_eps) if norm else None,
        ),
        grid=(nb + 1,),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        scratch_shapes=scratch_shapes,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=_vmem_limit_bytes(
                k, n, out_block, split_cols, split_head_dim, norm
            ),
        ),
    )(*operands)
    if len(outs) == 1:
        return outs[0]
    return jnp.concatenate([outs[0], outs[1].reshape(m, split_cols)], axis=1)
