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
"""The combine: every token's top-k arrival rows gathered on the sparse
cores, then summed with the router weights on the TensorCore,

    out[t] = sum over k of row(t, k) * (scale(t, k) * weight(t, k))

the scale and the weight multiplied first, in f32. The arrival rows come
in layout.result_lane_blocks(hidden) blocks, a whole number of SUBLANES;
the sum writes the true `hidden` and never reads past it.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v3 import device
from vllm_torchtpu.kernels.fused_moe.v3 import row_gather_kernel as gather
from vllm_torchtpu.kernels.fused_moe.v3.config import (
    COMBINE_TOKENS_PER_TILE_DEFAULT,
    COMBINE_UNROLL_DEFAULT,
    GATHER_BLOCK_ROWS_DEFAULT,
    GATHER_RING_DEFAULT,
)
from vllm_torchtpu.kernels.fused_moe.v3.layout import LANES, SUBLANES


def _stage_align(dtype):
    """Sublanes per vector register of `dtype`: a staging-tile row offset
    that is a multiple of it is a whole number of registers."""
    return SUBLANES * 32 // jnp.finfo(dtype).bits


def _stage_dtype(dtype, lane_blocks):
    """The staging tile's dtype: the sum's own when a token's rows fill
    whole registers of it, else f32."""
    return dtype if lane_blocks % _stage_align(dtype) == 0 else jnp.float32


def _weighted_sum_body(
    *refs, tokens_per_tile, top_k, lane_blocks, hidden, unroll, with_scales, dtype
):
    """One tile of tokens. The tile's scales and weights, [tokens, top_k]
    each, arrive as vectors and are multiplied once into the coefficient
    table; then token by token the top_k rows (a row fills whole vector
    registers) accumulate in registers, k in order, each times its
    coefficient spread over the row, and land in the staging tile as
    [tokens * lane blocks, LANES], which viewed as [tokens, hidden] is
    the output. The sum runs in `dtype`; the coefficient is cast to it as
    a vector."""
    scales_ref = refs[0] if with_scales else None
    weights_ref, rows_ref, out_ref, coef_vmem, sum_vmem = refs[int(with_scales) :]
    coef = weights_ref[...]
    if scales_ref is not None:
        # A route of weight zero has an unwritten scale as well as an
        # unwritten row, and zero times a NaN or Inf scale is NaN, which
        # the per-term test below keeps: select, not scale.
        coef = jnp.where(coef != 0, coef * scales_ref[...], jnp.zeros_like(coef))
    coef_vmem[...] = coef

    def token(t, carry):
        c = coef_vmem[pl.ds(t, 1), :].astype(dtype)  # [1, top_k]
        acc = None
        for k in range(top_k):
            coef = c[:, k : k + 1]
            term = rows_ref[t, k].astype(dtype) * coef
            # A route of weight zero points at a row nobody wrote this
            # call: its term is dropped, not multiplied, so a NaN or Inf
            # there cannot reach the token.
            term = jnp.where(coef != 0, term, jnp.zeros_like(term))
            acc = term if acc is None else acc + term
        assert acc is not None
        row0 = pl.multiple_of(t * lane_blocks, _stage_align(sum_vmem.dtype))
        sum_vmem[pl.ds(row0, lane_blocks), :] = acc.astype(sum_vmem.dtype)
        return carry

    lax.fori_loop(0, tokens_per_tile, token, 0, unroll=unroll)
    staged = sum_vmem[...].reshape(tokens_per_tile, lane_blocks * sum_vmem.shape[1])
    if hidden != staged.shape[1]:
        staged = staged[:, :hidden]  # the zero blocks past the true width
    out_ref[...] = staged.astype(out_ref.dtype)


def weighted_sum(
    rows,
    scales,
    weights,
    *,
    tokens_per_shard,
    hidden,
    out_dtype,
    tokens_per_tile=COMBINE_TOKENS_PER_TILE_DEFAULT,
    unroll=COMBINE_UNROLL_DEFAULT,
    dtype=jnp.float32,
):
    """rows [tokens * top_k, lane blocks, LANES] fp8 or bf16 in pair order
    (token-major: pair t * top_k + k is token t's slot k), scales
    [tokens * top_k] f32 the same or None, weights [tokens, top_k] f32 ->
    [tokens, hidden] out_dtype. The rows' blocks are the true width padded
    to whole SUBLANES (zero blocks past `hidden`)."""
    n_rows, lane_blocks, lanes = rows.shape
    if not 0 < hidden <= lane_blocks * lanes:
        raise ValueError(
            f"hidden {hidden} does not fit in the gathered rows' "
            f"{lane_blocks} {lanes}-lane blocks"
        )
    if hidden % lanes:
        raise ValueError(
            f"hidden {hidden} is not a whole number of {lanes}-lane blocks"
        )
    if lanes != LANES:
        raise ValueError(
            f"gathered rows of {lanes} lanes. A row is staged as {LANES}-lane blocks"
        )
    if n_rows % tokens_per_shard:
        raise ValueError(
            f"{n_rows} gathered rows is not a whole number of "
            f"slots of {tokens_per_shard} tokens"
        )
    top_k = n_rows // tokens_per_shard
    with_scales = scales is not None
    if (scales is not None and scales.shape != (n_rows,)) or weights.shape != (
        tokens_per_shard,
        top_k,
    ):
        raise ValueError(
            f"scales {None if scales is None else scales.shape} "
            f"and weights {weights.shape} "
            f"do not match {n_rows} rows as {tokens_per_shard} "
            f"tokens x {top_k} slots"
        )
    if (
        scales is not None and scales.dtype != jnp.float32
    ) or weights.dtype != jnp.float32:
        raise ValueError(
            f"scales {None if scales is None else scales.dtype} "
            f"and weights {weights.dtype}: "
            "the coefficient is formed in float32 from both"
        )
    if tokens_per_shard % tokens_per_tile:
        raise ValueError(
            f"{tokens_per_shard} tokens is not a whole number of "
            f"{tokens_per_tile}-token tiles"
        )
    if lane_blocks % SUBLANES:
        raise ValueError(
            f"{lane_blocks} lane blocks per row: the staging tile "
            f"is stored {SUBLANES} sublanes at a time"
        )
    staged = lane_blocks * lanes
    stage_dtype = _stage_dtype(dtype, lane_blocks)
    # The double-buffered input and output blocks and the staging tile,
    # with headroom, never over the chip's budget.
    itemsize = jnp.dtype(out_dtype).itemsize
    needed = (
        2 * top_k * tokens_per_tile * staged
        + 2 * tokens_per_tile * hidden * itemsize
        + tokens_per_tile * staged * jnp.dtype(stage_dtype).itemsize
    )
    vmem_limit = min(2 * needed, device.vmem_limit())
    table = pl.BlockSpec((tokens_per_tile, top_k), lambda i: (i, 0))
    tables = [weights]
    if scales is not None:
        tables = [scales.reshape(tokens_per_shard, top_k)] + tables

    def body(*refs):
        return _weighted_sum_body(
            *refs,
            tokens_per_tile=tokens_per_tile,
            top_k=top_k,
            lane_blocks=lane_blocks,
            hidden=hidden,
            unroll=unroll,
            with_scales=with_scales,
            dtype=dtype,
        )

    rows = rows.reshape(tokens_per_shard, top_k, lane_blocks, lanes)
    rows_spec = pl.BlockSpec(
        (tokens_per_tile, top_k, lane_blocks, lanes), lambda i: (i, 0, 0, 0)
    )
    out_spec = pl.BlockSpec((tokens_per_tile, hidden), lambda i: (i, 0))
    coefficients = pltpu.VMEM((tokens_per_tile, top_k), jnp.float32)
    accumulator = pltpu.VMEM((tokens_per_tile * lane_blocks, lanes), stage_dtype)
    out_shape = jax.ShapeDtypeStruct((tokens_per_shard, hidden), out_dtype)
    program = pl.pallas_call(
        body,
        out_shape=out_shape,
        grid=(tokens_per_shard // tokens_per_tile,),
        in_specs=[table] * len(tables) + [rows_spec],
        out_specs=out_spec,
        scratch_shapes=[coefficients, accumulator],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit),
        name=f"fused_ep_combine_sum_t{tokens_per_tile}u{unroll}"
        f"{'' if with_scales else '_bf16'}"
        f"{'' if dtype == jnp.float32 else '_e16'}",
    )
    return program(*tables, rows)


def combine(
    arrivals,
    arrival_scales,
    arrival_rows,
    weights,
    out_dtype,
    *,
    hidden,
    block_rows=GATHER_BLOCK_ROWS_DEFAULT,
    ring=GATHER_RING_DEFAULT,
    tokens_per_tile=COMBINE_TOKENS_PER_TILE_DEFAULT,
    unroll=COMBINE_UNROLL_DEFAULT,
    dtype=jnp.float32,
):
    """The layer's output from the arrival rows as the FFN kernel returns
    it: arrivals [R, result blocks, LANES] fp8 or bf16 (the true `hidden`
    padded with zero blocks to whole SUBLANES), arrival_scales [R] f32 (a
    word per row) or None for bf16 rows, arrival_rows
    [tokens * top_k] int32 in pair order (token-major: pair t * top_k + k
    is token t's slot k), weights [tokens, top_k] -> [tokens, hidden]."""
    tokens_per_shard, top_k = weights.shape
    n_pairs = tokens_per_shard * top_k
    if arrival_rows.ndim != 1 or arrival_rows.shape[0] < n_pairs:
        raise ValueError(
            f"the arrival rows are {arrival_rows.shape}. One per "
            f"(token, slot) pair of the weights' {weights.shape} "
            "is expected, then any padding pairs"
        )
    rows, scales = gather.gather_rows_and_scales(
        arrivals, arrival_scales, arrival_rows, block_rows=block_rows, ring=ring
    )
    if arrival_rows.shape[0] > n_pairs:
        # The pair list was padded to whole gather blocks; the real pairs
        # come first (the padding sits past the last token's pairs).
        rows = rows[:n_pairs]
        scales = None if scales is None else scales[:n_pairs]
    return weighted_sum(
        rows,
        scales,
        weights.astype(jnp.float32),
        tokens_per_shard=tokens_per_shard,
        hidden=hidden,
        out_dtype=out_dtype,
        tokens_per_tile=tokens_per_tile,
        unroll=unroll,
        dtype=dtype,
    )
