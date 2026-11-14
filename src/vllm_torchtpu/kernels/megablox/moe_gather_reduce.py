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
"""TensorCore MoE combine: gather each token's routed rows by DMA, sum them.

Destination-major: each grid step owns a block of tokens, DMAs the
``topk`` expert-sorted rows of every token in it from HBM, and reduces them
weighted on the VPU. The next block's row DMAs are issued before the current
block is reduced, so the transfer overlaps the arithmetic.

A bf16 HBM array packs rows in pairs into 32-bit words, so a single row is
not addressable by DMA: the kernel views the source as uint32 row pairs,
fetches the pair that holds each routed row, and keeps the row's half.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Tokens per grid step.
_TOKEN_BLOCK = 128
# Token blocks whose row DMAs are in flight while one block is reduced. The
# kernel is bound by the DMAs, and more of them in flight hide their latency:
# 2 blocks ran the served 8192-token combine in 295 us against 375 us for 1.
_BLOCKS_AHEAD = 2
# Rows x columns of one VPU reduction chunk: one packed bf16 output tile row
# group, 4 f32 vregs per routed row.
_CHUNK_ROWS = 16
_CHUNK_COLS = 256
_HIGH_HALF = -65536  # 0xFFFF0000 as int32


def gather_reduce_vmem_bytes(hidden_size: int, topk: int) -> int:
    """VMEM of the in-flight row pairs and the double-buffered output."""
    rows = (_BLOCKS_AHEAD + 1) * topk * _TOKEN_BLOCK * hidden_size * 4
    out = 2 * _TOKEN_BLOCK * hidden_size * 2
    meta = 3 * 2 * _TOKEN_BLOCK * 128 * 4
    return rows + out + meta


def can_use_gather_reduce(x: jax.Array, num_routes: int, topk: int, tpu_info) -> bool:
    """Whether this combine shape fits the kernel on this TPU."""
    if x.ndim != 2 or x.dtype != jnp.bfloat16 or x.shape[0] % 2:
        return False
    if num_routes % topk:
        return False
    if x.shape[1] % _CHUNK_COLS:
        return False
    # The routed-row index table lives in SMEM; half of it is budgeted.
    num_tokens = num_routes // topk
    padded_routes = pl.cdiv(num_tokens, _TOKEN_BLOCK) * _TOKEN_BLOCK * topk
    if padded_routes * 4 > tpu_info.smem_capacity_bytes // 2:
        return False
    return gather_reduce_vmem_bytes(x.shape[1], topk) <= int(
        tpu_info.vmem_capacity_bytes * 0.75
    )


def _kernel(
    pair_ref,  # SMEM int32[padded_routes]: row-pair index per route
    shift_ref,  # VMEM int32[TB, topk]: 16 for an even row, 0 for an odd one
    keep_ref,  # VMEM int32[TB, topk]: high-half mask, 0 for an invalid route
    weight_ref,  # VMEM f32[TB, topk]
    x_hbm_ref,  # HBM bf16[num_rows, hidden]
    out_ref,  # VMEM bf16[TB, hidden]
    rows_vmem,  # VMEM uint32[ahead + 1, topk, TB, hidden]
    sems,  # DMA[ahead + 1]
    *,
    topk: int,
    hidden_size: int,
    token_block: int,
    ahead: int,
):
    step = pl.program_id(0)
    num_steps = pl.num_programs(0)
    num_slots = ahead + 1
    slot = step % num_slots
    x_pairs = x_hbm_ref.bitcast(jnp.uint32)  # [num_rows // 2, hidden]

    def issue_token(block, slot, t):
        for k in range(topk):
            pair = pair_ref[block * (token_block * topk) + t * topk + k]
            pltpu.make_async_copy(
                x_pairs.at[pl.ds(pair, 1)],
                rows_vmem.at[slot, k, pl.ds(t, 1)],
                sems.at[slot],
            ).start()

    def wait(slot):
        # Every row DMA of a slot signals the same semaphore with its bytes,
        # so one wait for the whole slot is exact in any landing order.
        pltpu.make_async_copy(
            rows_vmem.at[slot], rows_vmem.at[slot], sems.at[slot]
        ).wait()

    @pl.when(step == 0)
    def _():
        for block in range(ahead):

            def token(t, carry, block=block):
                issue_token(jnp.minimum(block, num_steps - 1), block, t)
                return carry

            lax.fori_loop(0, token_block, token, 0, unroll=4)

    wait(slot)
    # Block step + ahead goes out row group by row group, in the same basic
    # block as the group's arithmetic, so the scalar slots issue it while the
    # vector slots reduce. A branch would split the block, so the last steps
    # refetch the last block instead and drain it at the end.
    ahead_block = jnp.minimum(step + ahead, num_steps - 1)
    ahead_slot = (step + ahead) % num_slots

    def row_group(g, carry):
        row0 = pl.multiple_of(g * _CHUNK_ROWS, _CHUNK_ROWS)
        rows = pl.ds(row0, _CHUNK_ROWS)
        for t in range(_CHUNK_ROWS):
            issue_token(ahead_block, ahead_slot, row0 + t)

        shape = (_CHUNK_ROWS, _CHUNK_COLS)
        # Per-route lane broadcasts, invariant along the hidden dim.
        shift = [
            jnp.broadcast_to(shift_ref[rows, pl.ds(k, 1)], shape) for k in range(topk)
        ]
        keep = [
            jnp.broadcast_to(keep_ref[rows, pl.ds(k, 1)], shape) for k in range(topk)
        ]
        weight = [
            jnp.broadcast_to(weight_ref[rows, pl.ds(k, 1)], shape) for k in range(topk)
        ]
        for j in range(hidden_size // _CHUNK_COLS):
            cols = pl.ds(j * _CHUNK_COLS, _CHUNK_COLS)
            acc = jnp.zeros(shape, jnp.float32)
            for k in range(topk):
                words = pltpu.bitcast(rows_vmem[slot, k, rows, cols], jnp.int32)
                # The routed row's half, moved to the high bits of an f32. An
                # invalid route keeps nothing: rows past the GMM's computed
                # prefix are never written and may hold NaNs.
                bits = jnp.bitwise_and(jnp.left_shift(words, shift[k]), keep[k])
                acc += pltpu.bitcast(bits, jnp.float32) * weight[k]
            out_ref[rows, cols] = acc.astype(out_ref.dtype)
        return carry

    lax.fori_loop(0, token_block // _CHUNK_ROWS, row_group, 0)

    @pl.when(step + 1 == num_steps)
    def _():
        for i in range(1, num_slots):
            wait((step + i) % num_slots)


@functools.partial(
    jax.jit,
    static_argnames=("reduce_group_size", "token_block", "ahead"),
)
def gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
    token_block: int = _TOKEN_BLOCK,
    ahead: int = _BLOCKS_AHEAD,
) -> jax.Array:
    """``sum_k w[t, k] * x[indices[t * topk + k]]`` over valid routes, in f32.

    ``x`` holds the expert-sorted routed rows; ``indices`` gives each
    token-major route's row in it. Rows of invalid routes are not read into
    the sum, so they may hold anything.
    """
    topk = reduce_group_size
    num_rows, hidden_size = x.shape
    num_tokens = indices.size // topk
    padded_tokens = pl.cdiv(num_tokens, token_block) * token_block
    pad = (padded_tokens - num_tokens) * topk

    indices = jnp.pad(indices.astype(jnp.int32), (0, pad))
    valid = jnp.pad(valid_rows_mask, (0, pad)).reshape(padded_tokens, topk)
    keep = jnp.where(valid, _HIGH_HALF, 0).astype(jnp.int32)
    weights = topk_weights.astype(jnp.float32)
    weights = jnp.pad(weights, (0, pad)).reshape(padded_tokens, topk)
    pairs = jnp.right_shift(indices, 1)
    # bf16 packs row 2i in the low half of a word and row 2i + 1 in the high.
    shifts = jnp.where(jnp.bitwise_and(indices, 1) == 0, 16, 0).astype(jnp.int32)
    shifts = shifts.reshape(padded_tokens, topk)

    tpu_info = pltpu.get_tpu_info()
    meta_spec = pl.BlockSpec((token_block, topk), lambda i, *_: (i, 0))
    out = pl.pallas_call(
        functools.partial(
            _kernel,
            topk=topk,
            hidden_size=hidden_size,
            token_block=token_block,
            ahead=ahead,
        ),
        out_shape=jax.ShapeDtypeStruct((padded_tokens, hidden_size), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(padded_tokens // token_block,),
            in_specs=[
                meta_spec,
                meta_spec,
                meta_spec,
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec((token_block, hidden_size), lambda i, *_: (i, 0)),
            scratch_shapes=[
                pltpu.VMEM((ahead + 1, topk, token_block, hidden_size), jnp.uint32),
                pltpu.SemaphoreType.DMA((ahead + 1,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            # The row DMAs of step i + 1 are issued at step i.
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=int(tpu_info.vmem_capacity_bytes * 0.9),
            disable_bounds_checks=True,
        ),
        name=f"moe_gather_reduce-r_{num_rows}-h_{hidden_size}-k_{topk}",
    )(pairs, shifts, keep, weights, x)
    return out[:num_tokens]
