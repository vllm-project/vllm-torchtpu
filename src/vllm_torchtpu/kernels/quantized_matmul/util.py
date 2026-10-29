# Copyright 2025 Google LLC
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
"""Utility functions for quantized matmul kernel."""

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.quantized_matmul.tuned_block_sizes import TunedValue


def unfold_args(
    conditions: tuple[jax.Array | bool, ...],
    fn_conditions: tuple[bool, ...],
    fn: Callable[..., Any],
):
    """Minimize run-time branching of fn by converting jnp.bool to python bool."""
    if conditions:
        arg = conditions[0]
        if isinstance(arg, bool):
            unfold_args(conditions[1:], fn_conditions + (arg,), fn)
        else:
            assert arg.dtype == jnp.bool and arg.size == 1
            jax.lax.cond(
                arg,
                lambda: unfold_args(conditions[1:], fn_conditions + (True,), fn),
                lambda: unfold_args(conditions[1:], fn_conditions + (False,), fn),
            )
    else:
        fn(*fn_conditions)


def quantize_tensor(
    x: jax.Array, dtype: jnp.dtype, dim: int = -1, block_size: int | None = None
):
    if block_size is not None:
        # Flatten all leading dims into a single batch dim for block
        # quantization, then restore the original shape.
        orig_shape = x.shape
        k_dim = orig_shape[-1]
        x_flat = x.reshape(-1, k_dim)
        n_dim = x_flat.shape[0]
        x_reshaped = x_flat.reshape(n_dim, -1, block_size)
        x_q, scale = quantize_block(x_reshaped, axis=-1, target_dtype=dtype)

        x_q = x_q.reshape(orig_shape)

        return x_q, scale.transpose(1, 2, 0).astype(jnp.float32)
    data_q, scale = quantize_block(x, axis=dim, target_dtype=dtype)
    return data_q, scale.astype(jnp.float32)


# Rows of x per grid step of `quantize_rows`.
QUANTIZE_BLOCK_ROWS = 128
# Rows quantized together inside a step; the 8-bit output tile is 32 rows tall,
# so a dynamically indexed group must start on a multiple of 32.
QUANTIZE_GROUP_ROWS = 64
# Columns converted and stored per step of a row group.
QUANTIZE_LANE_CHUNK = 512
_LANES = 128
# Lanes of the row-scale output: one vreg row, so it is stored unrelaid.
_SCALE_LANES = _LANES


def _quantize_rows_kernel(x_ref, q_ref, s_ref, *, group_rows: int):
    k = x_ref.shape[1]

    def quantize_group(rows):
        # quantize_block's ops, with the group read twice from VMEM, column
        # chunk by chunk, instead of held in vregs between the max and the
        # convert. max is exact, so the lane blocks may be folded in any order.
        if k % QUANTIZE_LANE_CHUNK:
            x_q, scale = quantize_block(
                x_ref[rows, :], axis=1, target_dtype=q_ref.dtype
            )
            q_ref[rows, :] = x_q
        else:
            folded = jnp.abs(x_ref[rows, 0:_LANES])
            for c in range(_LANES, k, _LANES):
                folded = jnp.maximum(folded, jnp.abs(x_ref[rows, c : c + _LANES]))
            abs_max = jnp.max(folded, axis=1, keepdims=True)
            scale = scale_from_abs_max(abs_max, q_ref.dtype)
            for c in range(0, k, QUANTIZE_LANE_CHUNK):
                cols = slice(c, c + QUANTIZE_LANE_CHUNK)
                q_ref[rows, cols] = quantize_with_scale(
                    x_ref[rows, cols], scale, q_ref.dtype
                )
        s_ref[rows, :] = jnp.broadcast_to(scale, (scale.shape[0], _SCALE_LANES))

    num_groups = x_ref.shape[0] // group_rows
    if num_groups == 1:
        quantize_group(slice(None))
        return

    def body(g, carry):
        quantize_group(pl.ds(pl.multiple_of(g * group_rows, group_rows), group_rows))
        return carry

    jax.lax.fori_loop(0, num_groups, body, 0)


def quantize_rows(
    x: jax.Array, dtype: jnp.dtype, interpret: bool = False
) -> tuple[jax.Array, jax.Array]:
    """`quantize_tensor(x, dtype)` for a 2-D ``x``, in one pass over ``x``.

    Each row group runs `quantize_block`'s own ops in VMEM, so the values and
    scales are those `quantize_tensor` computes, while ``x`` is read from HBM
    once instead of once for the abs-max reduce and once for the convert.
    Returns ``x_q`` ``[m, k]`` and the f32 row scales ``[m, 1]``.
    """
    m, k = x.shape
    block_rows = min(QUANTIZE_BLOCK_ROWS, m)
    group_rows = QUANTIZE_GROUP_ROWS
    if block_rows % group_rows:
        group_rows = block_rows
    x_q, scale = pl.pallas_call(
        functools.partial(_quantize_rows_kernel, group_rows=group_rows),
        out_shape=(
            jax.ShapeDtypeStruct((m, k), dtype),
            jax.ShapeDtypeStruct((m, _SCALE_LANES), jnp.float32),
        ),
        grid=(pl.cdiv(m, block_rows),),
        in_specs=[pl.BlockSpec((block_rows, k), lambda i: (i, 0))],
        out_specs=(
            pl.BlockSpec((block_rows, k), lambda i: (i, 0)),
            pl.BlockSpec((block_rows, _SCALE_LANES), lambda i: (i, 0)),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=_quantize_rows_vmem_bytes(block_rows, k, x.dtype, dtype),
        ),
        name="quantize_rows",
        interpret=interpret,
    )(x)
    return x_q, scale[:, :1]


def _quantize_rows_vmem_bytes(rows: int, k: int, x_dtype, q_dtype) -> int:
    """Double-buffered blocks plus room for one group's f32 temporaries."""
    block = rows * (
        k * (dtypes.itemsize_bits(x_dtype) + dtypes.itemsize_bits(q_dtype)) // 8
        + _SCALE_LANES * 4
    )
    return 2 * block + 4 * QUANTIZE_GROUP_ROWS * k * 4 + (4 << 20)


def next_multiple(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple


def get_kernel_name(tuned_value: TunedValue):
    batch_block_size = tuned_value.batch_block_size
    out_block_size = tuned_value.out_block_size
    in_block_size = tuned_value.in_block_size
    return (
        f"quantized_matmul_kernel_{batch_block_size}_{out_block_size}_{in_block_size}"
    )


def xla_quantized_matmul(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    quantize_activation=True,
    out_dtype: jnp.dtype | None = None,
    quantize_in_one_pass: bool = False,
) -> jax.Array:
    """
    Reference (pure JAX) implementation of the quantized matmul kernel.

    Weights are in the canonical (k, n) layout, so this is a plain
    `(m, k) @ (k, n)` contraction. Callers holding an N-major
    `[n_output_features, n_input_features]` weight (the Pallas kernels and
    their tests) transpose at the call site.

    Args:
        x:  Activation.
        x: Activation. [num_tokens, n_input_features]
        w_q: Weight quantized array. [n_input_features, n_output_features]
        w_scale: Weight quantization scale. [n_output_features]
        out_dtype: Output dtype; defaults to ``x.dtype``.
        quantize_in_one_pass: Quantize ``x`` with the `quantize_rows` TPU
            kernel instead of XLA ops; same values, one read of ``x``.

    Returns:
        Output of the quantized matmul.
    """
    if quantize_activation:
        acc_dtype = jnp.float32
        if quantize_activation and jnp.issubdtype(w_q.dtype, jnp.integer):
            acc_dtype = jnp.int32

        if quantize_in_one_pass:
            x_q, x_scale = quantize_rows(x, w_q.dtype)
        else:
            x_q, x_scale = quantize_tensor(x, w_q.dtype)
        out = jax.lax.dot_general(
            x_q,
            w_q,
            dimension_numbers=(((1,), (0,)), ((), ())),
            preferred_element_type=acc_dtype,
        ).astype(jnp.float32)
        out *= x_scale
    else:
        out = jax.lax.dot_general(
            x,
            w_q,
            dimension_numbers=(((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
    out *= jnp.expand_dims(w_scale, 0)
    return out.astype(out_dtype or x.dtype)


def xla_quantized_batched_matmul(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    dimension_numbers: tuple,
    quantize_activation: bool = True,
) -> jax.Array:
    """Quantized matmul with batch dimensions via dot_general.

    Generalizes xla_quantized_matmul to support batch dimensions (axes
    shared between both operands that appear in the output). Uses
    jax.lax.dot_general with configurable dimension_numbers.

    Args:
        x: Activation tensor (e.g. [T, N, H]).
        w_q: Quantized weight (e.g. [A, N, H]).
        w_scale: Per-output-channel weight scale (e.g. [A]).
        dimension_numbers: ``((contracting_x, contracting_w),
            (batch_x, batch_w))`` for dot_general.
        quantize_activation: Whether to dynamically quantize activations.

    Returns:
        Output of the batched quantized matmul. Shape is determined by
        dot_general: batch dims first, then lhs free dims, then rhs free dims.
    """
    contract_dims, batch_dims = dimension_numbers

    if quantize_activation:
        acc_dtype = jnp.float32
        if jnp.issubdtype(w_q.dtype, jnp.integer):
            acc_dtype = jnp.int32

        x_q, x_scale = quantize_tensor(x, w_q.dtype)
        out = jax.lax.dot_general(
            x_q,
            w_q,
            dimension_numbers=(contract_dims, batch_dims),
            preferred_element_type=acc_dtype,
        ).astype(jnp.float32)
        # Permute x_scale to match dot_general output ordering.
        # dot_general output: batch dims, lhs free dims, rhs free dims.
        # x_scale has same shape as x but with contracting dim(s) as 1.
        contract_set = set(contract_dims[0])
        batch_set = set(batch_dims[0])
        lhs_free = [
            i for i in range(x.ndim) if i not in contract_set and i not in batch_set
        ]
        perm = list(batch_dims[0]) + lhs_free + list(contract_dims[0])
        if perm != list(range(x.ndim)):
            x_scale = jnp.transpose(x_scale, perm)
        out *= x_scale
    else:
        out = jax.lax.dot_general(
            x,
            w_q,
            dimension_numbers=(contract_dims, batch_dims),
            preferred_element_type=jnp.float32,
        )

    # Broadcast w_scale to match the output shape.
    # dot_general output order: batch dims, lhs free dims, rhs free dims.
    # w_scale is per-output-channel (rhs free dims), so expand leading dims.
    n_leading = out.ndim - w_scale.ndim
    for _ in range(n_leading):
        w_scale = jnp.expand_dims(w_scale, 0)
    out *= w_scale
    return out.astype(x.dtype)


def quantize_array(
    x: jax.Array,  # [bs_block_size, in_block_size]
    x_abs_max: jax.Array,  # [1, bs_block_size]
    quant_dtype: jnp.dtype,
):
    is_float = jnp.issubdtype(quant_dtype, jnp.floating)
    dtype_info = jnp.finfo(quant_dtype) if is_float else jnp.iinfo(quant_dtype)
    dtype_max = float(dtype_info.max)

    # TODO(kyuyeunk): Investigate performance gain from non xlu transpose.
    scale = jnp.transpose(x_abs_max / dtype_max)
    scale_inv = jnp.nan_to_num(1 / scale, dtype_max)
    return (x * scale_inv).astype(quant_dtype), scale.astype(jnp.float32)


def get_vmem_limit(
    n_batch: int,
    n_out: int,
    n_in: int,
    batch_block_size: int,
    out_block_size: int,
    in_block_size: int,
    x_dtype: jnp.dtype,
    x_q_dtype: jnp.dtype,
    w_q_dtype: jnp.dtype,
    scale_dtype: jnp.dtype,
    out_dtype: jnp.dtype,
    acc_dtype: jnp.dtype,
    save_acc: bool,
    save_x_q: bool,
    upper_limit_bytes: int,
):
    """Calculate VMEM limit for the kernel."""

    # Calculate in/out VMEM size.
    x_size = batch_block_size * in_block_size * dtypes.itemsize_bits(x_dtype)
    x_abs_max_size = batch_block_size * dtypes.itemsize_bits(scale_dtype)
    w_q_size = out_block_size * in_block_size * dtypes.itemsize_bits(w_q_dtype)
    w_scale_size = out_block_size * dtypes.itemsize_bits(scale_dtype)
    out_size = batch_block_size * out_block_size * dtypes.itemsize_bits(out_dtype)

    vmem_in_out = x_size + x_abs_max_size + w_q_size + w_scale_size + out_size
    vmem_in_out *= 2  # Account for compute and vreg spills.

    # Account for double buffering.
    # Double buffering is used only if there are multiple blocks per in/out.
    vmem_in_out += x_size if (n_batch > 1 or n_in > 1) else 0
    vmem_in_out += x_abs_max_size if (n_batch > 1) else 0
    vmem_in_out += w_q_size if (n_out > 1 or n_in > 1) else 0
    vmem_in_out += w_scale_size if (n_out > 1) else 0
    vmem_in_out += out_size if (n_batch > 1 or n_out > 1) else 0

    # Calculate scratch VMEM size.
    acc_size = batch_block_size * out_block_size * dtypes.itemsize_bits(acc_dtype)
    x_q_size = batch_block_size * in_block_size * dtypes.itemsize_bits(x_q_dtype)
    x_scale_size = batch_block_size * dtypes.itemsize_bits(scale_dtype)

    vmem_scratch = acc_size if save_acc else 0
    vmem_scratch += x_q_size + x_scale_size if save_x_q else 0
    vmem_scratch *= 2  # Account for compute and vreg spills.

    # Add in/out and scratch VMEM size.
    vmem_used = vmem_in_out + vmem_scratch
    vmem_used_bytes = vmem_used // 8  # Convert bits to bytes.
    # Specify upper limit. Defaults to 96MB.
    vmem_limit_bytes = min(vmem_used_bytes, upper_limit_bytes)

    return vmem_limit_bytes


def validate_inputs(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    x_abs_max: jax.Array | None,
    batch_block_size: int,
    out_block_size: int,
    in_block_size: int,
):
    """Verify inputs invoking the kernel."""

    # Verify input shapes.
    if x.shape[1] != w_q.shape[1]:
        raise ValueError(f"{x.shape[1]=} must be equal to {w_q.shape[1]=}")
    if w_q.shape[0] != w_scale.shape[1] and (
        w_scale.ndim == 3 and w_q.shape[0] != w_scale.shape[2]
    ):
        raise ValueError(f"{w_q.shape[0]=} must be equal to {w_scale.shape[1]=}")
    if x_abs_max is not None and x_abs_max.shape != (1, x.shape[0]):
        raise ValueError(f"{x_abs_max.shape=} must be equal to (1, {x.shape[0]=})")
    if x.shape[0] % batch_block_size != 0:
        raise ValueError(f"{x.shape[0]=} must be a multiple of {batch_block_size=}")
    if w_q.shape[0] % out_block_size != 0:
        raise ValueError(f"{w_q.shape[0]=} must be a multiple of {out_block_size=}")
    if x.shape[1] % in_block_size != 0:
        raise ValueError(f"{x.shape[1]=} must be a multiple of {in_block_size=}")


def get_max_min(target_dtype):
    if jnp.issubdtype(target_dtype, jnp.floating):
        return jnp.finfo(target_dtype).max.astype(jnp.float32), jnp.finfo(
            target_dtype
        ).min.astype(jnp.float32)
    else:
        return jnp.iinfo(target_dtype).max, jnp.iinfo(target_dtype).min


def quantize_block(data, axis, target_dtype):
    """Calculates scale and quantizes a block of data."""
    abs_max = jnp.max(
        jnp.abs(data),
        axis=axis,
        keepdims=True,
    )
    scale = scale_from_abs_max(abs_max, target_dtype)
    return quantize_with_scale(data, scale, target_dtype), scale


def scale_from_abs_max(abs_max, target_dtype):
    """`quantize_block`'s scale for a block whose largest magnitude is abs_max."""
    dtype_max, _ = get_max_min(target_dtype)
    scale = abs_max / dtype_max
    return jnp.where(scale == 0, 1.0, scale)


def quantize_with_scale(data, scale, target_dtype):
    """`quantize_block`'s values for data under its scale."""
    dtype_max, dtype_min = get_max_min(target_dtype)
    if jnp.issubdtype(target_dtype, jnp.floating):
        return (data / scale).clip(dtype_min, dtype_max).astype(target_dtype)
    return jnp.round(data / scale).astype(target_dtype)
