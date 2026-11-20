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
"""Attention output gate and per-token fp8 quantization in one pass.

For each token row: ``y = attn_out * sigmoid(gate)``, rounded to bf16 as the
eager gate returns it; then ``scale = max|y| / 448`` over the row (1 for an
all-zero row) and ``q = clip(y / scale)`` in float8_e4m3fn -- the rows and
scales the FP8 ``o_proj`` would compute for itself. The Qwen3.5 model patch
(``models/vllm/qwen3_next_patch.py``) runs it in front of the FP8 ``o_proj``.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def legal_tile(rows: int, cap: int) -> int:
    """The largest multiple of 32 up to ``cap`` dividing ``rows``, else ``rows``
    (a block equal to the full dimension is always legal)."""
    for t in range(min(cap, rows), 31, -1):
        if rows % t == 0 and t % 32 == 0:
            return t
    return rows


def select_vreg_step(cols: int) -> int:
    """An in-kernel row step that keeps the live fp32 rows in 64 VREGs."""
    vregs_per_row = max(1, (cols + 1023) // 1024)
    return max(16, (64 // vregs_per_row // 8) * 8)


# Two bf16 inputs and one fp8 output, double buffered: 10 bytes per element
# of a block. 16 MiB keeps the unsharded width (TP1 under pipeline
# parallelism, 8192 columns) inside the scoped VMEM limit.
_BLOCK_BYTES_PER_ELEMENT = 10
_BLOCK_VMEM_BUDGET = 16 << 20


def _fits_vmem_budget(rows: int, cols: int) -> bool:
    """Whether a [rows, cols] block, double buffered, fits the VMEM budget."""
    return rows * cols * _BLOCK_BYTES_PER_ELEMENT <= _BLOCK_VMEM_BUDGET


def select_tile(rows: int, cols: int) -> int:
    """Select a VMEM BlockSpec tile cap for [rows, cols] gate + quant: a
    multiple of 32, or ``rows`` for a step that runs as one block.

    Up to 2048 columns a decode/verify step (rows <= 256) caps the tile near
    rows / 6 (at least 32 rows) so double-buffered VMEM DMAs overlap with
    compute; wider, such a step runs as one block where it fits the VMEM
    budget (see below). For large prefill shapes (rows > 256), uses a larger
    VMEM block (up to 512 rows) while the kernel processes VREG-sized
    sub-tiles internally, avoiding large Pallas grid overhead at T=8192
    without spilling VREGs.
    """
    vreg_cap = select_vreg_step(cols)
    budget_cap = max(
        32, (_BLOCK_VMEM_BUDGET // (_BLOCK_BYTES_PER_ELEMENT * cols)) // 32 * 32
    )
    if rows <= 256:
        if vreg_cap < 32:
            # Wider than 2048 columns the VREG step (16 rows) is below the
            # smallest legal tile. A step whose block fits the budget runs as
            # one block: on TPU7x, 32-row blocks were 4-15% slower at 64 to
            # 256 rows. A larger step tiles at the budget cap (256 rows at
            # 8192 columns: two 128-row blocks, as fast as one).
            return rows if _fits_vmem_budget(rows, cols) else budget_cap
        dma_cap = max(32, (rows // 6 // 32) * 32)
        return min(vreg_cap, dma_cap, budget_cap)
    return min(max(vreg_cap, 512), budget_cap)


class BlockPlan(NamedTuple):
    """How `pallas_sigmoid_gate_quant` runs a step."""

    rows: int  # the step's rows after padding
    block_rows: int  # BlockSpec rows
    step_rows: int  # in-kernel row step


def plan_blocks(rows: int, cols: int) -> BlockPlan:
    """The padded rows, block and in-kernel step for a [rows, cols] step.

    A step pads to 8 rows, and to a multiple of 32 when no legal tile of its
    rows fits the VMEM budget, so every block fits it.
    """
    padded = max(rows, 8)
    block = legal_tile(padded, select_tile(padded, cols))
    if not _fits_vmem_budget(block, cols):
        padded = pl.cdiv(padded, 32) * 32
        block = legal_tile(padded, select_tile(padded, cols))
    if not _fits_vmem_budget(block, cols):
        raise NotImplementedError(
            f"pallas_sigmoid_gate_quant: a {block}-row block of {cols} columns "
            f"exceeds the {_BLOCK_VMEM_BUDGET >> 20} MiB VMEM budget; use "
            "sigmoid_gate_quant_reference at this width."
        )
    return BlockPlan(padded, block, legal_tile(block, select_vreg_step(cols)))


def _sigmoid_gate_quant_kernel(attn_ref, gate_ref, q_ref, s_ref, *, step_m: int):
    bm = attn_ref.shape[0]
    step = min(bm, step_m)
    for r in range(bm // step):
        sl = slice(r * step, (r + 1) * step)
        a = attn_ref[sl, :].astype(jnp.float32)
        g = gate_ref[sl, :].astype(jnp.float32)
        y = (a * jax.nn.sigmoid(g)).astype(jnp.bfloat16).astype(jnp.float32)
        amax = jnp.max(jnp.abs(y), axis=1, keepdims=True)
        scale = amax * jnp.float32(1.0 / 448.0)
        scale = jnp.where(scale == 0.0, 1.0, scale)
        scale_inv = jnp.float32(1.0) / scale
        q_ref[sl, :] = jnp.clip(y * scale_inv, -448.0, 448.0).astype(jnp.float8_e4m3fn)
        s_ref[sl, :] = scale


def pallas_sigmoid_gate_quant(
    attn_out: jax.Array,
    gate: jax.Array,
    *,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Fused attn_output * sigmoid(gate) + per-token FP8 quantization in Pallas."""
    orig_m, k = attn_out.shape
    m, bm, step_m = plan_blocks(orig_m, k)
    if m != orig_m:
        pad = m - orig_m
        attn_out = jnp.pad(attn_out, ((0, pad), (0, 0)))
        gate = jnp.pad(gate, ((0, pad), (0, 0)))
    q_spec = pl.BlockSpec((bm, k), lambda i: (i, 0))
    s_spec = pl.BlockSpec((bm, 1), lambda i: (i, 0))
    q, s = pl.pallas_call(
        functools.partial(_sigmoid_gate_quant_kernel, step_m=step_m),
        out_shape=[
            jax.ShapeDtypeStruct((m, k), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((m, 1), jnp.float32),
        ],
        grid=(m // bm,),
        in_specs=[q_spec, q_spec],
        out_specs=[q_spec, s_spec],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name="pallas_sigmoid_gate_quant",
        interpret=pltpu.InterpretParams() if interpret else False,
    )(attn_out, gate)
    if m != orig_m:
        return q[:orig_m], s[:orig_m]
    return q, s


def sigmoid_gate_quant_reference(
    attn_out: jax.Array,
    gate: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Pure-JAX reference for attn_output * sigmoid(gate) + per-token FP8 quant."""
    a = attn_out.astype(jnp.float32)
    g = gate.astype(jnp.float32)
    y = (a * jax.nn.sigmoid(g)).astype(jnp.bfloat16)
    y_f32 = y.astype(jnp.float32)
    amax = jnp.max(jnp.abs(y_f32), axis=-1, keepdims=True)
    scale = amax * jnp.float32(1.0 / 448.0)
    scale = jnp.where(scale == 0.0, 1.0, scale)
    scale_inv = jnp.float32(1.0) / scale
    q = jnp.clip(y_f32 * scale_inv, -448.0, 448.0).astype(jnp.float8_e4m3fn)
    return q, scale
