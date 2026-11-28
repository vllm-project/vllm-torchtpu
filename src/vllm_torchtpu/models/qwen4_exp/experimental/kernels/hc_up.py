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

# Provenance: copied from src/vllm_torchtpu/kernels/qwen4_exp/hc_up.py
#   at 9061fd34883df781b65bfc0eaad0686a36d88799.
#   Owned by models/qwen4_exp from this copy on.
"""Prefill HC up projection with an in-tile gated branch reduction."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _up_kernel(mid_ref, norm_ref, weight_ref, mixed_ref, *, hidden, branches):
    mid = mid_ref[...]
    norm = norm_ref[...].astype(jnp.float32)
    # Checkpoint-native weight is [branches * hidden, lowrank].
    logits = jax.lax.dot_general(
        mid,
        weight_ref[...],
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    # F.linear(BF16, BF16) rounds to BF16 before the FP32 sigmoid.
    logits = logits.astype(jnp.bfloat16).astype(jnp.float32)
    weighted = jax.nn.sigmoid(logits) * norm
    mixed = weighted[:, :hidden]
    for branch in range(1, branches):
        start = branch * hidden
        mixed = mixed + weighted[:, start : start + hidden]
    mixed_ref[...] = (mixed / branches).astype(jnp.bfloat16)


@functools.partial(jax.jit, static_argnames=("branches", "hidden", "token_tile"))
def _up_epilogue(mid, norm, weight, *, branches, hidden, token_tile):
    tokens, lowrank = mid.shape
    width = branches * hidden
    if (
        tokens % token_tile
        or norm.shape != (tokens, width)
        or weight.shape != (width, lowrank)
        or mid.dtype != jnp.bfloat16
        or norm.dtype != jnp.bfloat16
        or weight.dtype != jnp.bfloat16
    ):
        raise ValueError("HC up epilogue requires BF16 and whole token tiles")
    return pl.pallas_call(
        functools.partial(_up_kernel, hidden=hidden, branches=branches),
        grid=(tokens // token_tile,),
        in_specs=[
            pl.BlockSpec((token_tile, lowrank), lambda tile: (tile, 0)),
            pl.BlockSpec((token_tile, width), lambda tile: (tile, 0)),
            pl.BlockSpec((width, lowrank), lambda tile: (0, 0)),
        ],
        out_specs=pl.BlockSpec((token_tile, hidden), lambda tile: (tile, 0)),
        out_shape=jax.ShapeDtypeStruct((tokens, hidden), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=60 << 20
        ),
        name=f"qwen4exp_hc_up_epilogue_m{token_tile}",
    )(mid, norm, weight)
