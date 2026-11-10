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
"""The fused-EP v2 layer's activation quantization as one Pallas pass.

`kernel.rowquant_fp8` (per-row dynamic fp8: the row's abs-max in bf16, the
scale and its inverse in f32, the rows scaled in bf16 and cast to fp8) runs in
XLA as three programs per MoE layer at the served shape: the abs-max reduce,
the quantizing fusion, and a copy that re-tiles the fp8 rows from the fusion's
layout into the per-token [32, 128] tiles the expert-parallel all-gather and
the FFN kernel's row gather read (28 ms per round at 8,704 tokens, base7).
This kernel reads each row once and writes the fp8 rows directly as
[tokens, hidden / 128, 128], so the re-tiling happens in VMEM, and returns the
same row scale [tokens, 1] f32.

`apply`: "bf16" scales the rows in bf16 exactly as rowquant_fp8 spells it
(x * sinv.astype(bf16), then the fp8 cast); "f32" scales in f32 and casts once.
The kernel bench decides which one is the serving program's arithmetic; the
other is the rounding-order mutant that must fail the bit-identity check.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.fused_moe.v2.host import FP8, FP8_MAX, HIDDEN_LANE_BLOCK

APPLY = "bf16"


def quant_block_rows(tokens):
    """The widest power-of-two row block (<= 256, >= 32) dividing `tokens`, or 0."""
    b = 256
    while b >= 32:
        if tokens % b == 0:
            return b
        b //= 2
    return 0


def quant_kernel_supported(tokens, hidden):
    return quant_block_rows(tokens) > 0 and hidden % HIDDEN_LANE_BLOCK == 0


def _body(x_ref, q_ref, s_ref, *, apply, lane_blocks):
    x = x_ref[...]  # [b, hidden] bf16
    amax = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
    scale = amax.astype(jnp.float32) / FP8_MAX
    usable = jnp.isfinite(scale) & (scale > 0)
    scale = jnp.where(usable, scale, 0.0)
    sinv = jnp.where(usable, 1.0 / jnp.where(usable, scale, 1.0), 0.0)
    if apply == "bf16":
        q = jnp.where(usable, x * sinv.astype(x.dtype), jnp.zeros_like(x)).astype(FP8)
    else:
        q = jnp.where(usable, x.astype(jnp.float32) * sinv, 0.0).astype(FP8)
    q_ref[...] = q.reshape(q.shape[0], lane_blocks, HIDDEN_LANE_BLOCK)
    s_ref[...] = scale


def rowquant_fp8_kernel(x, *, apply=None, interpret=None):
    """(q [tokens, hidden / 128, 128] fp8, scale [tokens, 1] f32) from bf16 rows x [tokens, hidden]."""
    tokens, hidden = x.shape
    if x.dtype != jnp.bfloat16 or not quant_kernel_supported(tokens, hidden):
        raise ValueError(f"quant kernel: unsupported rows {x.shape} {x.dtype}")
    apply = apply or (envs.MOE_FUSED_EP_V2_TEST_MUTANT[len("qk_"):]
                      if envs.MOE_FUSED_EP_V2_TEST_MUTANT.startswith("qk_") else APPLY)
    if apply not in ("bf16", "f32"):
        raise ValueError(f"quant kernel: apply {apply}")
    if interpret is None:
        interpret = envs.MOE_FUSED_EP_V2_INTERPRET
    b = quant_block_rows(tokens)
    lb = hidden // HIDDEN_LANE_BLOCK
    return pl.pallas_call(
        functools.partial(_body, apply=apply, lane_blocks=lb),
        grid=(tokens // b,),
        in_specs=[pl.BlockSpec((b, hidden), lambda i: (i, 0))],
        out_specs=(pl.BlockSpec((b, lb, HIDDEN_LANE_BLOCK), lambda i: (i, 0, 0)),
                   pl.BlockSpec((b, 1), lambda i: (i, 0))),
        out_shape=(jax.ShapeDtypeStruct((tokens, lb, HIDDEN_LANE_BLOCK), FP8),
                   jax.ShapeDtypeStruct((tokens, 1), jnp.float32)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=(pltpu.InterpretParams() if interpret else False),
        name="fused_ep_rowquant" + ("" if apply == APPLY else "_" + apply),
    )(x)
