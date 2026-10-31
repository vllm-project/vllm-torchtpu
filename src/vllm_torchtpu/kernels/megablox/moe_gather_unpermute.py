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
"""TensorCore MoE combine: gather each token's routed rows and sum them.

Computes what the SparseCore ``ragged_gather_reduce_v2`` computes, bit for bit:
per token, the f32 products ``row * weight`` of its valid routes, summed left to
right in route order, zero for a token with no valid route, rounded to bf16.

The kernel is destination-major. A grid step owns ``TOKEN_BLOCK`` output tokens.
One row DMA per valid route lands the route's gmm2 row in VMEM at (slot k,
token t), where k is the route's rank among its token's valid routes, so the
sum is dense vector work over (tokens, hidden) tiles, one pass per slot. The
next block's rows are fetched while the current block is summed.

bf16 rows are packed in pairs into 32-bit words in HBM, so a row is fetched as
the 32-bit row holding its pair and its half is picked with a per-token shift.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Output tokens per grid step. The slot buffers hold 2 x topk x TOKEN_BLOCK
# 32-bit rows.
TOKEN_BLOCK = 128
# Tokens summed together: one f32 sublane tile. Each pass over a slot covers
# every token of the group, so a smaller group wastes fewer passes.
SUM_GROUP = 8
# Tokens per row-DMA list. Smaller lists cost XLA less to build; each list is
# one loop in the kernel.
LIST_TOKENS = 32
# Row DMAs issued per loop iteration. A list is padded to a multiple with DMAs
# into a spare row, so the scalar address chains of one iteration overlap.
ISSUE_UNROLL = 8

_LANES = 128
_SUBLANES = 8
_HIGH_HALF = 0xFFFF0000
# Packed list entry: destination slot row in the low bits, the route's index
# in the (slot, token) tables above them.
_DST_BITS = 12
# Rows of the per-token weight table, one per slot.
_TABLE_ROWS = 8
# Rows of the per-token route table: the route count, then a bit per slot set
# when the slot's row is even.
_COUNT_ROW = 0
_EVEN_ROW = 1
_ROUTE_ROWS = 2
_VMEM_LIMIT_BYTES = 48 * 1024 * 1024
_SMEM_BUDGET_BYTES = 512 * 1024


def _slot_rows(topk: int) -> int:
    return topk * TOKEN_BLOCK


def _spare_row(topk: int) -> int:
    """Destination of a list's padding DMAs, past every slot."""
    return _slot_rows(topk)


def can_use_gather_unpermute(
    route_output: jax.Array, topk_weights: jax.Array, num_tokens: int, topk: int
) -> bool:
    """Whether this combine shape fits the kernel."""
    num_routes, hidden_size = route_output.shape
    return (
        route_output.dtype == jnp.bfloat16
        and topk_weights.dtype == jnp.bfloat16
        and num_routes == num_tokens * topk
        and num_routes % 2 == 0
        and hidden_size % _LANES == 0
        and num_tokens % TOKEN_BLOCK == 0
        and TOKEN_BLOCK % LIST_TOKENS == 0
        and TOKEN_BLOCK % SUM_GROUP == 0
        and (LIST_TOKENS * topk) % ISSUE_UNROLL == 0
        and topk <= _TABLE_ROWS
        and _spare_row(topk) < (1 << _DST_BITS)
        and _TABLE_ROWS * num_tokens < (1 << (31 - _DST_BITS))
        # The row and entry tables in SMEM.
        and (_TABLE_ROWS + topk) * num_tokens * 4 <= _SMEM_BUDGET_BYTES
        # Slot buffers, output blocks and the per-block tables.
        and 4
        * hidden_size
        * (2 * (_slot_rows(topk) + _SUBLANES) + TOKEN_BLOCK + SUM_GROUP)
        + 4 * _LANES * TOKEN_BLOCK * (topk + 2)
        <= _VMEM_LIMIT_BYTES - 8 * 1024 * 1024
    )


def _route_tables(
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    *,
    num_tokens: int,
    topk: int,
):
    """Per-(slot, token) tables and per-list row-DMA tables.

    Built with tokens along lanes and no gathers: slot k of a token holds its
    k-th valid route, so the slots keep the routes' order.
    """
    valid = valid_rows_mask.reshape(num_tokens, topk).T
    src = indices.reshape(num_tokens, topk).T
    weight = topk_weights.astype(jnp.float32).reshape(num_tokens, topk).T
    valid_i = valid.astype(jnp.int32)
    slot_of_route = jnp.cumsum(valid_i, axis=0) - valid_i
    count = valid_i.sum(axis=0)

    # Rows past the last slot stay zero; the count row is filled below.
    rows = jnp.arange(_TABLE_ROWS, dtype=jnp.int32)[:, None]
    slot_src = jnp.zeros((_TABLE_ROWS, num_tokens), jnp.int32)
    weight_table = jnp.zeros((_TABLE_ROWS, num_tokens), jnp.float32)
    for j in range(topk):
        sel = valid[j][None, :] & (slot_of_route[j][None, :] == rows)
        slot_src = jnp.where(sel, src[j][None, :], slot_src)
        weight_table = jnp.where(sel, weight[j][None, :], weight_table)

    # Even rows sit in the low half of their 32-bit pair word.
    even_bits = jnp.left_shift((slot_src % 2 == 0).astype(jnp.int32), rows).sum(axis=0)
    route_table = jnp.stack([count, even_bits])
    pair_rows = (slot_src // 2).reshape(-1)

    # Row-DMA lists: the i-th valid (slot, token) of each list of LIST_TOKENS
    # tokens, found by comparing i with the running route count per token.
    num_lists = num_tokens // LIST_TOKENS
    list_count = count.reshape(num_lists, LIST_TOKENS)
    running = jnp.cumsum(list_count, axis=1)
    entry_id = jnp.arange(LIST_TOKENS * topk, dtype=jnp.int32)
    before = running[:, :, None] <= entry_id[None, None, :]
    token = before.sum(axis=1, dtype=jnp.int32)
    slot = entry_id[None, :] - jnp.where(before, list_count[:, :, None], 0).sum(
        axis=1, dtype=jnp.int32
    )
    list_id = jnp.arange(num_lists, dtype=jnp.int32)[:, None]
    lists_per_block = TOKEN_BLOCK // LIST_TOKENS
    dst_row = slot * TOKEN_BLOCK + (list_id % lists_per_block) * LIST_TOKENS + token
    route = slot * num_tokens + list_id * LIST_TOKENS + token
    list_len = running[:, -1]
    entries = jnp.where(
        entry_id[None, :] < list_len[:, None],
        dst_row | (route << _DST_BITS),
        _spare_row(topk),
    )
    list_steps = (list_len + ISSUE_UNROLL - 1) // ISSUE_UNROLL

    block_rows = (list_steps * ISSUE_UNROLL).reshape(-1, lists_per_block).sum(axis=1)
    group_max = count.reshape(-1, SUM_GROUP).max(axis=1)
    return (
        pair_rows,
        entries.reshape(-1),
        list_steps,
        block_rows,
        group_max,
        weight_table,
        route_table,
    )


def _kernel(
    # Scalar prefetch
    pair_rows_ref,
    entries_ref,
    list_steps_ref,
    block_rows_ref,
    group_max_ref,
    # In
    weight_ref,
    route_ref,
    rows_hbm_ref,
    # Out
    out_ref,
    # Scratch
    slots_ref,
    weight_lanes_ref,
    count_lanes_ref,
    even_lanes_ref,
    acc_ref,
    sems,
    *,
    topk: int,
    hidden_size: int,
):
    block = pl.program_id(0)
    num_blocks = pl.num_programs(0)
    buf = jnp.bitwise_and(block, 1)
    pair_rows_hbm = rows_hbm_ref.bitcast(jnp.uint32)
    lists_per_block = TOKEN_BLOCK // LIST_TOKENS
    list_entries = LIST_TOKENS * topk

    def fetch(blk, slot_buf):
        for i in range(lists_per_block):
            lst = blk * lists_per_block + i
            base = lst * list_entries

            def issue(step, carry, base=base):
                for u in range(ISSUE_UNROLL):
                    entry = entries_ref[base + step * ISSUE_UNROLL + u]
                    row = pair_rows_ref[entry >> _DST_BITS]
                    pltpu.make_async_copy(
                        pair_rows_hbm.at[row],
                        slots_ref.at[slot_buf, entry & ((1 << _DST_BITS) - 1)],
                        sems.at[slot_buf],
                    ).start()
                return carry

            lax.fori_loop(0, list_steps_ref[lst], issue, 0)

    @pl.when(block == 0)
    def _():
        fetch(block, buf)

    @pl.when(block + 1 < num_blocks)
    def _():
        fetch(block + 1, 1 - buf)

    # Per-token tables with tokens along sublanes, broadcast across lanes by
    # a matmul with one-hot rows. Every entry is exact in bf16 (a bf16 weight,
    # a count or a bit mask below 256), so the f32 results are exact.
    def broadcast_rows(table, num_rows):
        shape = (table.shape[0], num_rows * _LANES)
        one_hot = lax.broadcasted_iota(jnp.int32, shape, 0) == (
            lax.broadcasted_iota(jnp.int32, shape, 1) // _LANES
        )
        return jnp.dot(
            table.T.astype(jnp.bfloat16),
            one_hot.astype(jnp.bfloat16),
            preferred_element_type=jnp.float32,
        )

    weights = broadcast_rows(weight_ref[...], topk)
    for k in range(topk):
        weight_lanes_ref[k] = weights[:, k * _LANES : (k + 1) * _LANES]
    routes = broadcast_rows(route_ref[...].astype(jnp.float32), _ROUTE_ROWS)
    routes = routes.astype(jnp.int32)
    count_lanes_ref[...] = routes[:, _COUNT_ROW * _LANES : (_COUNT_ROW + 1) * _LANES]
    even_lanes_ref[...] = routes[:, _EVEN_ROW * _LANES : (_EVEN_ROW + 1) * _LANES]

    # DMA semaphores count bytes, so a wait for 8 rows covers any 8 of the
    # block's rows. A wait's row count must be static and tile aligned.
    block_len = block_rows_ref[block]

    def wait_rows(rows):
        def wait(i, carry):
            pltpu.make_async_copy(rows, rows, sems.at[buf]).wait()
            return carry

        return wait

    tile_rows = slots_ref.at[buf, pl.ds(0, _SUBLANES)]
    lax.fori_loop(0, lax.div(block_len, _SUBLANES), wait_rows(tile_rows), 0)
    lax.fori_loop(0, lax.rem(block_len, _SUBLANES), wait_rows(slots_ref.at[buf, 0]), 0)

    def sum_group(g, carry):
        row0 = pl.multiple_of(g * SUM_GROUP, SUM_GROUP)
        rows = pl.ds(row0, SUM_GROUP)
        group_max = group_max_ref[block * (TOKEN_BLOCK // SUM_GROUP) + g]
        count = count_lanes_ref[rows, :]
        even = even_lanes_ref[rows, :]

        def slot_pass(k, *, first: bool, last: bool):
            weight = weight_lanes_ref[k, rows, :]
            shift = (jnp.right_shift(even, k) & 1).astype(jnp.uint32) * 16
            valid = count > k
            slot_row0 = pl.multiple_of(k * TOKEN_BLOCK + row0, SUM_GROUP)
            for c in range(hidden_size // _LANES):
                cols = pl.ds(c * _LANES, _LANES)
                packed = slots_ref[buf, pl.ds(slot_row0, SUM_GROUP), cols]
                value = lax.bitcast_convert_type(
                    jnp.left_shift(packed, shift) & jnp.uint32(_HIGH_HALF),
                    jnp.float32,
                )
                product = value * weight
                if first:
                    total = jnp.where(valid, product, 0.0)
                else:
                    acc = acc_ref[:, cols]
                    total = jnp.where(valid, acc + product, acc)
                if last:
                    out_ref[rows, cols] = total.astype(out_ref.dtype)
                else:
                    acc_ref[:, cols] = total

        @pl.when(group_max == 0)
        def _():
            out_ref[rows, :] = jnp.zeros((SUM_GROUP, hidden_size), out_ref.dtype)

        @pl.when(group_max == 1)
        def _():
            slot_pass(0, first=True, last=True)

        @pl.when(group_max > 1)
        def _():
            slot_pass(0, first=True, last=False)

            def middle(k, c):
                slot_pass(k, first=False, last=False)
                return c

            lax.fori_loop(1, group_max - 1, middle, 0)
            slot_pass(group_max - 1, first=False, last=True)

        return carry

    lax.fori_loop(0, TOKEN_BLOCK // SUM_GROUP, sum_group, 0)


@functools.partial(jax.jit, static_argnames=("num_tokens", "topk", "interpret"))
def gather_unpermute(
    route_output: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    *,
    num_tokens: int,
    topk: int,
    interpret: bool = False,
) -> jax.Array:
    """Sum each token's valid routed rows, weighted, as ragged_gather_reduce_v2.

    Args:
      route_output: bf16 ``[num_tokens * topk, hidden]`` routed rows.
      indices: ``[num_tokens * topk]`` row of each (token, route).
      topk_weights: bf16 ``[num_tokens * topk]`` weight of each (token, route).
      valid_rows_mask: ``[num_tokens * topk]`` whether the route's row is valid.

    Returns:
      bf16 ``[num_tokens, hidden]``.
    """
    # The kernel broadcasts the weights through a bf16 matmul.
    if topk_weights.dtype != jnp.bfloat16:
        raise ValueError(f"topk_weights must be bf16, got {topk_weights.dtype}")
    hidden_size = route_output.shape[-1]
    (
        pair_rows,
        entries,
        list_steps,
        block_rows,
        group_max,
        weight_table,
        route_table,
    ) = _route_tables(
        indices,
        topk_weights,
        valid_rows_mask,
        num_tokens=num_tokens,
        topk=topk,
    )
    num_blocks = num_tokens // TOKEN_BLOCK
    token_block = lambda b, *_: (b, 0)  # noqa: E731
    token_columns = lambda b, *_: (0, b)  # noqa: E731
    return pl.pallas_call(
        functools.partial(_kernel, topk=topk, hidden_size=hidden_size),
        out_shape=jax.ShapeDtypeStruct((num_tokens, hidden_size), route_output.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            grid=(num_blocks,),
            in_specs=[
                pl.BlockSpec((_TABLE_ROWS, TOKEN_BLOCK), token_columns),
                pl.BlockSpec((_ROUTE_ROWS, TOKEN_BLOCK), token_columns),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ],
            out_specs=pl.BlockSpec((TOKEN_BLOCK, hidden_size), token_block),
            scratch_shapes=[
                pltpu.VMEM((2, _slot_rows(topk) + _SUBLANES, hidden_size), jnp.uint32),
                pltpu.VMEM((topk, TOKEN_BLOCK, _LANES), jnp.float32),
                pltpu.VMEM((TOKEN_BLOCK, _LANES), jnp.int32),
                pltpu.VMEM((TOKEN_BLOCK, _LANES), jnp.int32),
                pltpu.VMEM((SUM_GROUP, hidden_size), jnp.float32),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            # Each step consumes rows the previous step fetched.
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
        name="moe_gather_unpermute",
        interpret=interpret,
    )(
        pair_rows,
        entries,
        list_steps,
        block_rows,
        group_max,
        weight_table,
        route_table,
        route_output,
    )
