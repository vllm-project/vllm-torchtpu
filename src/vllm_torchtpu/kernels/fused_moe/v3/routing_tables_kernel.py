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
"""The routing tables' five O(pairs x experts) passes as one Pallas program.

The program's grid steps run in order on the core, so the step order is the
barrier between its three phases:

  histogram  per group of pair blocks: each block's expert histogram and each
           pair's rank among the block's earlier pairs of the same expert,
           as single-pass bf16 matrix-unit dots against ones (0/1 operands
           with sums below the block size are exact in the f32
           accumulator).
  prefix   once: the chain of exclusive sums routing_tables does with
           cumsum, as strict-triangular f32 matmuls (exact below 2**24).
  lookup     per group again: the three table lookups of every pair as one
           [9, E] @ [E, block] bf16 dot, each table value split into three
           exact 8-bit digits and recombined with an int32 shift-add.

shard_tables_kernel is the pallas mode's: it reads the pair
indices out of the gathered routing messages, emits every shard table the
FFN kernel takes from the prefix step, zeroes the row tables the
sparse-core scatter fills, and hosts the token transport's middle step.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v3 import layout, transport
from vllm_torchtpu.kernels.fused_moe.v3.config import (
    TABLES_BLOCKS_PER_STEP_MAX,
    TABLES_ZERO_WORDS_DEFAULT,
)
from vllm_torchtpu.kernels.fused_moe.v3.layout import (
    BYTE_BITS,
    BYTE_MASK,
    LANES,
    SUBLANES,
)
from vllm_torchtpu.kernels.fused_moe.v3.routing_tables import (
    COUNT_ROWS,
    MAX_ROUTING_BLOCK,
)

# Pair blocks per grid step: enough independent chains for the scheduler,
# and a sublane-legal block shape.
BLOCKS_PER_STEP = TABLES_BLOCKS_PER_STEP_MAX  # the most blocks a step takes
# The lookup phase's table values reach a few million and may be negative.
# Biased by
# this they lie in [0, 2**23), where three 8-bit digits represent each one
# exactly and each digit is exact in bf16.
LOOKUP_BIAS = 1 << 22
# The three lookup tables (block offset, packed shift, expert base) travel
# through the one-hot matmul as three bf16 digit rows each, one byte per
# row, and are reassembled from the picked digits.
LOOKUP_TABLES = 3
LOOKUP_BYTES = 3
# The row each lookup table occupies in the picked result.
LOOKUP_BLOCK_OFFSET, LOOKUP_PACKED_SHIFT, LOOKUP_EXPERT_BASE = range(LOOKUP_TABLES)
# The shard-tables kernel's outputs, in order: the three the layer takes
# whole, the seven FFN tables, and the four expert tables.
SHARD_TABLE_OUTPUTS = (
    "run_rows",
    "arrival_row",
    "routed_row",
    "run_start",
    "run_blocks",
    "outgoing_offset",
    "push_blocks",
    "push_destination",
    "push_rows",
    "arrival_offset",
    "expert_rows",
    "expert_base",
    "counts",
    "visit_order",
)
# The first outputs carry their own block specs (the run rows, the
# arrival rows, the routed rows); the rest are whole small tables.
SHARD_TABLE_BLOCKED_OUTPUTS = 3
SHARD_TABLE_SCRATCH = (
    "histogram",
    "rank",
    "block_offset",
    "packed_shift",
    "expert_base",
)


def _pair_block(message_ref, i, block):
    """Pair block i of a grid step's routing message rows, as one [block]
    vector: the
    block's lane rows laid end to end (a lane concat of whole rows)."""
    rows_per_block = block // LANES
    rows = message_ref[i * rows_per_block : (i + 1) * rows_per_block, :]
    return jnp.concatenate([rows[j : j + 1, :] for j in range(rows_per_block)], axis=1)[
        0
    ]


def _exclusive_cumsum_rows(x):
    """Exclusive cumsum over axis 0 (Pallas has no cumsum lowering)."""
    m = x.shape[0]
    r = lax.broadcasted_iota(jnp.int32, (m, m), 0)
    c = lax.broadcasted_iota(jnp.int32, (m, m), 1)
    lower = jnp.where(c < r, 1.0, 0.0)
    return jnp.dot(
        lower,
        x.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    ).astype(jnp.int32)


def _exclusive_cumsum_cols(x):
    """Exclusive cumsum over axis 1, the same construction."""
    m = x.shape[1]
    r = lax.broadcasted_iota(jnp.int32, (m, m), 0)
    c = lax.broadcasted_iota(jnp.int32, (m, m), 1)
    upper = jnp.where(r < c, 1.0, 0.0)
    return jnp.dot(
        x.astype(jnp.float32),
        upper,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    ).astype(jnp.int32)


# Pairs one grid step reads as a whole number of (SUBLANES, LANES) tiles of
# the routing message: a step's rows are a block of that many sublanes.
STEP_PAIRS_MULTIPLE = SUBLANES * LANES


def step_pairs_multiple(blocks_per_step):
    """The pairs a shard's count is a whole number of for `blocks_per_step`
    blocks per step: the step's blocks of LANES pairs, and at least one
    (SUBLANES, LANES) tile of the routing message."""
    return max(blocks_per_step * LANES, STEP_PAIRS_MULTIPLE)


def shard_tables_block(pairs_per_shard, blocks_per_step=BLOCKS_PER_STEP):
    """The widest routing block the shard-tables kernel takes for a shard's
    pairs: a power of two of at least LANES pairs, with the shard's pairs a
    whole number of blocks_per_step-block groups and a group a whole
    number of the routing message's tiles."""
    block = MAX_ROUTING_BLOCK
    while block >= LANES:
        group = blocks_per_step * block
        if pairs_per_shard % group == 0 and group % STEP_PAIRS_MULTIPLE == 0:
            return block
        block //= 2
    raise ValueError(
        f"{pairs_per_shard} routed pairs per shard are not a "
        f"whole number of groups of {blocks_per_step} blocks of "
        f"at least {LANES} pairs and {STEP_PAIRS_MULTIPLE} "
        "pairs a group"
    )


def _histogram_constants(block, num_experts):
    row_i = lax.broadcasted_iota(jnp.int32, (block, block), 0)
    col_i = lax.broadcasted_iota(jnp.int32, (block, block), 1)
    strictly_above = jnp.where(row_i < col_i, 1.0, 0.0)
    ones_row = jnp.ones((1, block), jnp.bfloat16)
    expert_iota = lax.broadcasted_iota(jnp.int32, (block, num_experts), 1)
    return dict(
        strictly_above=strictly_above, ones_row=ones_row, expert_iota=expert_iota
    )


def _histogram_and_rank(
    pair_experts, b, histogram_vmem, rank_vmem, *, strictly_above, ones_row, expert_iota
):
    """Pair block b: its expert histogram into histogram_vmem[b], each
    pair's rank among the block's earlier pairs of its expert into
    rank_vmem[b]. The one-hot is (pair, expert), pairs down the sublanes,
    so the histogram comes back as whole result tiles."""
    one_hot = jnp.where(pair_experts[:, None] == expert_iota, 1.0, 0.0).astype(
        jnp.bfloat16
    )
    histogram = jnp.dot(ones_row, one_hot, preferred_element_type=jnp.float32)
    histogram_vmem[pl.ds(b, 1)] = histogram.astype(jnp.int32)
    same = jnp.where(pair_experts[:, None] == pair_experts[None, :], 1.0, 0.0)
    rank = jnp.dot(
        ones_row,
        (same * strictly_above).astype(jnp.bfloat16),
        preferred_element_type=jnp.float32,
    )
    rank_vmem[pl.ds(b, 1)] = rank.astype(jnp.int32)


class _Prefix(NamedTuple):
    """The prefix step's results, for the pallas mode's shard tables."""

    run_rows: jax.Array
    run_rows_aligned: jax.Array
    run_start_aligned: jax.Array
    expert_rows_aligned: jax.Array
    expert_base: jax.Array
    rows_per_dest: jax.Array
    recv_base: jax.Array


def _prefix_chain(
    histogram,
    block_offset_vmem,
    run_rows_ref,
    packed_vmem,
    expert_base_vmem,
    *,
    num_experts,
    num_shards,
    n_blocks,
    routed_rows_per_shard,
    row_block,
    slot_field,
):
    """The prefix step: routing_tables' cumsum chain from the block
    histogram [n_blocks, E]. Writes the block offsets, the run sizes, the
    packed shift table (source-major, so the lookup phase indexes
    sublanes) and the
    expert bases. Returns the values the shard tables are sliced from."""
    experts_per_shard = num_experts // num_shards
    block_offset_vmem[...] = _exclusive_cumsum_rows(histogram).astype(jnp.float32)
    run_rows = (
        histogram.reshape(num_shards, n_blocks // num_shards, num_experts).sum(axis=1).T
    )
    run_rows_ref[...] = run_rows.astype(jnp.int32)
    run_start = _exclusive_cumsum_cols(run_rows)
    run_rows_aligned = (run_rows + (row_block - 1)) & jnp.int32(-row_block)
    run_start_aligned = _exclusive_cumsum_cols(run_rows_aligned)
    slot_shift = run_start_aligned - run_start
    expert_rows_aligned = run_rows_aligned.sum(axis=1)
    rows_by_shard = expert_rows_aligned.reshape(num_shards, experts_per_shard)
    local_base = _exclusive_cumsum_cols(rows_by_shard)
    shard_base = local_base + jnp.arange(num_shards, dtype=jnp.int32)[
        :, None
    ] * jnp.int32(routed_rows_per_shard)
    expert_base = jnp.concatenate([shard_base[i] for i in range(num_shards)])
    region_rows = run_rows_aligned.reshape(num_shards, experts_per_shard, num_shards)
    rows_per_dest = region_rows.transpose(2, 0, 1).reshape(
        num_shards, num_shards * experts_per_shard
    )
    recv_base = _exclusive_cumsum_cols(rows_per_dest).reshape(
        num_shards, num_shards, experts_per_shard
    )
    position_shift = (
        jnp.concatenate([recv_base[:, i, :].T for i in range(num_shards)], axis=0)
        - run_start
    )
    packed = (position_shift * slot_field + slot_shift).astype(jnp.float32)
    for i in range(num_shards):
        packed_vmem[i, :] = packed[:, i]
    expert_base_vmem[...] = expert_base.astype(jnp.float32)[None, :]
    return _Prefix(
        run_rows,
        run_rows_aligned,
        run_start_aligned,
        expert_rows_aligned,
        expert_base,
        rows_per_dest,
        recv_base,
    )


def _table_lookups(
    pair_experts,
    b,
    *,
    blocks_per_shard,
    expert_iota_t,
    block_offset_vmem,
    packed_vmem,
    expert_base_vmem,
    bias,
):
    """Pair block b's three table lookups as one dot: [3, block] int32 rows
    of every pair's block offset, packed shift and expert base."""
    one_hot_t = jnp.where(pair_experts[None, :] == expert_iota_t, 1.0, 0.0).astype(
        jnp.bfloat16
    )
    source = b // blocks_per_shard
    tables = jnp.concatenate(
        [
            block_offset_vmem[pl.ds(b, 1)],
            packed_vmem[pl.ds(source, 1)],
            expert_base_vmem[...],
        ],
        axis=0,
    )
    biased = tables.astype(jnp.int32) + bias
    digits = jnp.concatenate(
        [
            ((biased >> (BYTE_BITS * d)) & BYTE_MASK).astype(jnp.bfloat16)
            for d in range(LOOKUP_BYTES)
        ],
        axis=0,
    )
    picked = jnp.dot(digits, one_hot_t, preferred_element_type=jnp.float32)
    value = None
    for d in range(LOOKUP_BYTES):
        digit = picked[d * LOOKUP_TABLES : (d + 1) * LOOKUP_TABLES].astype(
            jnp.int32
        ) << (BYTE_BITS * d)
        value = digit if value is None else value + digit
    return value - bias


def _shard_tables_body(
    *refs,
    num_experts,
    num_shards,
    pairs_per_shard,
    routed_rows_per_shard,
    block,
    row_block,
    slot_field,
    n_groups,
    axis,
    hosts_transport,
    zero_blocks,
    blocks_per_step,
    shard_operand=False,
):
    """The pallas mode's front end: the three phases plus every shard table
    the FFN kernel takes, emitted from the prefix step. The pair indices
    read out of the gathered routing messages. The row tables zeroed on
    the first steps. With `hosts_transport`, the transport's middle step
    (wait round A, issue round B) on the last step."""
    it = iter(refs)
    # The shard's index: an SMEM operand or the mesh axis.
    shard_ref = next(it) if shard_operand else None

    def shard_of():
        return shard_ref[0] if shard_ref is not None else lax.axis_index(axis)

    ids_ref = next(it)
    if hosts_transport:
        _, send_sem, receive_sems = next(it), next(it), next(it)
    (
        run_rows_ref,
        arrival_ref,
        routed_row_ref,
        run_start_ref,
        run_blocks_ref,
        outgoing_offset_ref,
        push_blocks_ref,
        push_destination_ref,
        push_rows_ref,
        arrival_offset_ref,
        expert_rows_ref,
        expert_base_ref,
        counts_ref,
        visit_order_ref,
    ) = (next(it) for _ in SHARD_TABLE_OUTPUTS)
    if hosts_transport:
        gathered_rows_out = next(it)
    zero_refs = [next(it) for _ in zero_blocks]
    (histogram_vmem, rank_vmem, block_offset_vmem, packed_vmem, expert_base_vmem) = (
        next(it) for _ in SHARD_TABLE_SCRATCH
    )
    step = pl.program_id(0)
    n_blocks = n_groups * blocks_per_step
    blocks_per_shard = pairs_per_shard // block
    groups_per_shard = blocks_per_shard // blocks_per_step
    rows_per_block = block // LANES
    expert_iota_t = lax.broadcasted_iota(jnp.int32, (num_experts, block), 0)
    row_block_shift = row_block.bit_length() - 1
    slot_field_shift = slot_field.bit_length() - 1

    for zero_ref, n_zero in zip(zero_refs, zero_blocks):

        @pl.when(step < n_zero)
        def _zero(zero_ref=zero_ref):
            zero_ref[...] = jnp.zeros_like(zero_ref)

    @pl.when(step < n_groups)
    def _histogram_phase():
        constants = _histogram_constants(block, num_experts)
        for i in range(blocks_per_step):
            b = step * blocks_per_step + i
            _histogram_and_rank(
                _pair_block(ids_ref, i, block),
                b,
                histogram_vmem,
                rank_vmem,
                **constants,
            )

    @pl.when(step == n_groups)
    def _prefix():
        experts_per_shard = num_experts // num_shards
        shard = shard_of()
        prefix = _prefix_chain(
            histogram_vmem[...],
            block_offset_vmem,
            run_rows_ref,
            packed_vmem,
            expert_base_vmem,
            num_experts=num_experts,
            num_shards=num_shards,
            n_blocks=n_blocks,
            routed_rows_per_shard=routed_rows_per_shard,
            row_block=row_block,
            slot_field=slot_field,
        )
        # This shard's rows of the [E, ...] tables, picked by a one-hot f32
        # dot (dynamic_slice does not lower in Pallas. Exact below 2**24).
        local_index = lax.broadcasted_iota(
            jnp.int32, (experts_per_shard, num_experts), 0
        )
        expert_index = lax.broadcasted_iota(
            jnp.int32, (experts_per_shard, num_experts), 1
        )
        row_select = jnp.where(
            expert_index == shard * experts_per_shard + local_index, 1.0, 0.0
        )

        def own_rows(table):
            return jnp.dot(
                row_select,
                table.astype(jnp.float32),
                precision=lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
            ).astype(jnp.int32)

        run_rows_aligned = own_rows(prefix.run_rows_aligned)
        run_start_aligned = own_rows(prefix.run_start_aligned)
        run_rows = own_rows(prefix.run_rows)
        per_dest = run_rows_aligned.sum(axis=0, keepdims=True)
        dest_base = _exclusive_cumsum_rows(per_dest.T).T
        outgoing_offset = dest_base + _exclusive_cumsum_rows(run_rows_aligned)
        # recv_base[d, shard, g] for every d -> [G, d]
        recv_flat = prefix.recv_base.reshape(num_shards, num_shards * experts_per_shard)
        flat_index = lax.broadcasted_iota(
            jnp.int32, (num_shards * experts_per_shard, experts_per_shard), 0
        )
        local_column = lax.broadcasted_iota(
            jnp.int32, (num_shards * experts_per_shard, experts_per_shard), 1
        )
        col_select = jnp.where(
            flat_index == shard * experts_per_shard + local_column, 1.0, 0.0
        )
        arrival_base = jnp.dot(
            recv_flat.astype(jnp.float32),
            col_select,
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        ).astype(jnp.int32)  # [d, G]
        arrival_offset = arrival_base.T  # [G, d] rows

        def blocks(rows):
            return jnp.right_shift(rows, row_block_shift).astype(jnp.int32)

        run_start_ref[...] = blocks(run_start_aligned)
        run_blocks_ref[...] = blocks(run_rows_aligned)
        outgoing_offset_ref[...] = blocks(outgoing_offset)
        push_blocks_ref[...] = blocks(run_rows_aligned)
        push_destination_ref[...] = blocks(arrival_base.T)
        push_rows_ref[...] = run_rows.astype(jnp.int32)
        arrival_offset_ref[...] = arrival_offset.astype(jnp.int32)

        expert_rows = own_rows(prefix.expert_rows_aligned[:, None])[:, 0]
        expert_base = own_rows(prefix.expert_base[:, None])[:, 0]
        expert_base_ref[...] = (expert_base - expert_base[0]).astype(jnp.int32)[None, :]
        expert_rows_ref[...] = expert_rows.astype(jnp.int32)[None, :]

        mine = (jnp.arange(num_shards, dtype=jnp.int32) == shard).astype(jnp.int32)
        other = 1 - mine
        first = (jnp.arange(experts_per_shard, dtype=jnp.int32) == 0).astype(jnp.int32)[
            :, None
        ]
        received = prefix.run_rows.sum(axis=0).astype(jnp.int32)
        received_aligned = prefix.rows_per_dest.sum(axis=1).astype(jnp.int32)
        active = (expert_rows > 0).astype(jnp.int32)[:, None]
        counts_ref[...] = jnp.concatenate(
            [
                (run_rows * other[None, :]).sum(axis=0, keepdims=True),
                ((received * mine)[None, :] * first - run_rows * mine[None, :]).sum(
                    axis=0, keepdims=True
                ),
                (run_rows_aligned * other[None, :]).sum(axis=0, keepdims=True),
                (
                    (received_aligned * mine)[None, :] * first
                    - run_rows_aligned * mine[None, :]
                ).sum(axis=0, keepdims=True),
                (active * mine[None, :]).sum(axis=0, keepdims=True),
            ],
            axis=0,
        ).astype(jnp.int32)

        # The visit order without a sort: each expert's rank under (rows
        # descending, index ascending), inverted by a one-hot.
        order = jnp.arange(experts_per_shard, dtype=jnp.int32)
        key = order - expert_rows.astype(jnp.int32) * jnp.int32(experts_per_shard)
        rank = jnp.sum((key[None, :] < key[:, None]).astype(jnp.int32), axis=1)
        inverse = jnp.sum(
            jnp.where(rank[None, :] == order[:, None], order[None, :], 0), axis=1
        )
        visit_order_ref[...] = jnp.minimum(
            inverse, jnp.int32(experts_per_shard - 1)
        ).astype(jnp.int32)[None, :]

    @pl.when(step > n_groups)
    def _lookup_phase():
        # The two pair tables the layer reads: the arrival row of each of
        # this shard's own pairs (the combine's gather index, in pair
        # order) and every pair's row in this shard's routed rows,
        # rebased, with a pair of another shard sent to
        # routed_rows_per_shard (the scatter drops it).
        group = step - n_groups - 1
        bias = jnp.int32(LOOKUP_BIAS)
        shard = shard_of()
        shard_base = shard * jnp.int32(routed_rows_per_shard)
        first_own_group = shard * jnp.int32(groups_per_shard)
        own = jnp.logical_and(
            group >= first_own_group, group < first_own_group + groups_per_shard
        )
        arrival_rows = []
        for i in range(blocks_per_step):
            b = group * blocks_per_step + i
            pair_experts = _pair_block(ids_ref, i, block)
            unrouted = pair_experts[None, :] == jnp.int32(num_experts)
            picked = _table_lookups(
                pair_experts,
                b,
                blocks_per_shard=blocks_per_shard,
                expert_iota_t=expert_iota_t,
                block_offset_vmem=block_offset_vmem,
                packed_vmem=packed_vmem,
                expert_base_vmem=expert_base_vmem,
                bias=bias,
            )
            rank = (
                rank_vmem[pl.ds(b, 1)]
                + picked[LOOKUP_BLOCK_OFFSET : LOOKUP_BLOCK_OFFSET + 1]
            )
            shift = picked[LOOKUP_PACKED_SHIFT : LOOKUP_PACKED_SHIFT + 1]
            arrival = rank + jnp.right_shift(shift, slot_field_shift)
            routed_row = (
                picked[LOOKUP_EXPERT_BASE : LOOKUP_EXPERT_BASE + 1]
                + rank
                + (shift & jnp.int32(slot_field - 1))
            )
            # A pair that routes nowhere (index past the last expert) takes
            # the drop row and gathers arrival row zero at weight zero.
            arrival = jnp.where(unrouted, jnp.int32(0), arrival)
            local = jnp.where(
                jnp.logical_or(unrouted, routed_row < shard_base),
                jnp.int32(routed_rows_per_shard),
                routed_row - shard_base,
            )
            routed_row_ref[pl.ds(i * block, block)] = local[0]
            arrival_rows += [
                arrival[:, j * LANES : (j + 1) * LANES] for j in range(rows_per_block)
            ]

        # Own arrival rows land as whole tiles at a tile-aligned row: a row
        # stored alone at a run-time offset is a branch, and so a matrix
        # unit pipeline drain, per block on every step.
        @pl.when(own)
        def _():
            rows = blocks_per_step * rows_per_block
            row0 = pl.multiple_of((group - first_own_group) * rows, rows)
            arrival_ref[pl.ds(row0, rows), :] = jnp.concatenate(arrival_rows, axis=0)

    if hosts_transport:

        @pl.when(step == 2 * n_groups)
        def _transport_middle():
            shard = shard_of()
            transport.forward_middle_rounds(
                transport.Exchange(
                    gathered_rows_out,
                    send_sem,
                    receive_sems,
                    shard,
                    gathered_rows_out.shape[0] // num_shards,
                    num_shards,
                    shard_operand=shard_operand,
                )
            )


def shard_tables_kernel(
    routing_messages,
    *,
    num_experts,
    num_shards,
    pairs_per_shard,
    routed_rows_per_shard,
    block,
    row_block,
    slot_field,
    axis,
    transport_refs=None,
    zero_tables=(),
    interpret=False,
    blocks_per_step=BLOCKS_PER_STEP,
    zero_words_per_step=TABLES_ZERO_WORDS_DEFAULT,
    shard_index=None,
    side_effects=True,
):
    """The pallas mode's front end, from the gathered routing messages
    (`routing messages` [num_shards, rows, LANES] int32: shard s's pair indices
    flat in
    its first pairs_per_shard // LANES rows, a pair's number its word).

    Returns (run_rows [E, num_shards], arrival rows [pairs_per_shard // LANES,
    LANES] of this shard's own pairs in pair order, routed_row [n], the FFN
    kernel's 7-tuple of shard tables, expert_rows [G], expert_base [G],
    counts, visit_order [G]), then the gathered rows when the call hosts the
    transport's middle step (`transport_refs` = (gathered rows, send sem,
    receive
    sems) from the start call. The gathered rows come back aliased with round B
    issued), then one zeroed table per length in `zero_tables`. Every
    output is placed in HBM.
    """
    n = num_shards * pairs_per_shard
    n_blocks = n // block
    n_groups = n_blocks // blocks_per_step
    if (
        block % LANES
        or pairs_per_shard % (blocks_per_step * block)
        or (blocks_per_step * block) % STEP_PAIRS_MULTIPLE
    ):
        raise ValueError(
            f"the pallas front end takes {block}-pair blocks in "
            f"groups of {blocks_per_step}: a block is a whole "
            f"number of {LANES}-lane rows, a group a whole "
            f"number of {STEP_PAIRS_MULTIPLE}-pair tiles, and a "
            f"shard's {pairs_per_shard} pairs a whole number of "
            "groups"
        )
    if routing_messages.shape[0] != num_shards or routing_messages.dtype != jnp.int32:
        raise ValueError(
            f"the gathered routing messages is {routing_messages.shape} "
            f"{routing_messages.dtype}. One int32 block per shard "
            f"({num_shards}) is expected"
        )
    if routing_messages.shape[1] * LANES < pairs_per_shard:
        raise ValueError(
            f"a shard's routing-message block of "
            f"{routing_messages.shape[1]} rows "
            f"holds fewer words than its {pairs_per_shard} pairs"
        )
    if len(zero_tables) > 2:
        raise ValueError(
            f"{len(zero_tables)} tables to zero. The scatter behind this call fills two"
        )
    # Each table is zeroed in equal blocks over the grid's steps: at least
    # zero_words_per_step words a step, more when the table is longer than
    # the steps would otherwise cover.
    steps = 2 * n_groups + 1
    zero_words = tuple(
        max(
            zero_words_per_step,
            layout.align_up(-(-length // steps), zero_words_per_step),
        )
        for length in zero_tables
    )
    zero_blocks = tuple(
        -(-length // words) for length, words in zip(zero_tables, zero_words)
    )
    rows_per_group = blocks_per_step * block // LANES
    groups_per_shard = pairs_per_shard // (blocks_per_step * block)
    f32, i32 = jnp.float32, jnp.int32
    vmem = pltpu.MemorySpace.VMEM
    hbm = pltpu.MemorySpace.HBM
    experts_per_shard = num_experts // num_shards
    hosts = transport_refs is not None

    def in_map(step):
        group = jnp.where(step < n_groups, step, jnp.maximum(step - n_groups - 1, 0))
        return (group // groups_per_shard, group % groups_per_shard, 0)

    def out_map(step):
        return (jnp.where(step > n_groups, step - n_groups - 1, 0),)

    def small(shape):
        return hbm(shape, i32)

    transport_in_specs = (
        [
            pl.BlockSpec(memory_space=hbm),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ]
        if hosts
        else []
    )
    transport_out_specs = (pl.BlockSpec(memory_space=hbm),) if hosts else ()
    transport_out_shape = (
        (jax.ShapeDtypeStruct(transport_refs[0].shape, transport_refs[0].dtype),)
        if hosts
        else ()
    )
    zero_out_specs = tuple(
        pl.BlockSpec(
            (words,), lambda step, n_zero=n_zero: (jnp.minimum(step, n_zero - 1),)
        )
        for words, n_zero in zip(zero_words, zero_blocks)
    )
    zero_out_shape = tuple(hbm((length,), i32) for length in zero_tables)
    outs = pl.pallas_call(
        functools.partial(
            _shard_tables_body,
            num_experts=num_experts,
            num_shards=num_shards,
            pairs_per_shard=pairs_per_shard,
            routed_rows_per_shard=routed_rows_per_shard,
            block=block,
            row_block=row_block,
            slot_field=slot_field,
            n_groups=n_groups,
            axis=axis,
            hosts_transport=hosts,
            zero_blocks=zero_blocks,
            blocks_per_step=blocks_per_step,
            shard_operand=shard_index is not None,
        ),
        grid=(2 * n_groups + 1,),
        in_specs=(
            [pl.BlockSpec(memory_space=pltpu.SMEM)] if shard_index is not None else []
        )
        + [pl.BlockSpec((None, rows_per_group, LANES), in_map, memory_space=vmem)]
        + transport_in_specs,
        out_specs=(
            pl.BlockSpec(memory_space=vmem),
            pl.BlockSpec(memory_space=vmem),
            pl.BlockSpec((blocks_per_step * block,), out_map, memory_space=vmem),
        )
        + (pl.BlockSpec(memory_space=vmem),)
        * (len(SHARD_TABLE_OUTPUTS) - SHARD_TABLE_BLOCKED_OUTPUTS)
        + transport_out_specs
        + zero_out_specs,
        out_shape=(
            small((num_experts, num_shards)),
            small((pairs_per_shard // LANES, LANES)),
            small((n,)),
            small((experts_per_shard, num_shards)),
            small((experts_per_shard, num_shards)),
            small((experts_per_shard, num_shards)),
            small((experts_per_shard, num_shards)),
            small((experts_per_shard, num_shards)),
            small((experts_per_shard, num_shards)),
            small((experts_per_shard, num_shards)),
            small((1, experts_per_shard)),
            small((1, experts_per_shard)),
            small((COUNT_ROWS, num_shards)),
            small((1, experts_per_shard)),
        )
        + transport_out_shape
        + zero_out_shape,
        scratch_shapes=[
            pltpu.VMEM((n_blocks, num_experts), i32),  # histogram
            pltpu.VMEM((n_blocks, block), i32),  # rank
            pltpu.VMEM((n_blocks, num_experts), f32),  # block offset
            pltpu.VMEM((num_shards, num_experts), f32),  # packed shift
            pltpu.VMEM((1, num_experts), f32),  # expert base
        ],
        # The gathered rows (the first transport ref) come back aliased; the
        # shard-index operand, when present, sits in front of the messages.
        input_output_aliases=(
            {(2 if shard_index is not None else 1): len(SHARD_TABLE_OUTPUTS)}
            if hosts
            else {}
        ),
        compiler_params=pltpu.CompilerParams(has_side_effects=hosts and side_effects),
        interpret=interpret,
        name="fused_ep_shard_tables",
    )(
        *(() if shard_index is None else (shard_index,)),
        routing_messages,
        *(transport_refs if hosts else ()),
    )
    (
        run_rows,
        arrival,
        routed_row,
        run_start,
        run_blocks,
        outgoing_offset,
        push_blocks,
        push_destination,
        push_rows,
        arrival_offset,
        expert_rows,
        expert_base,
        counts,
        visit,
    ) = outs[: len(SHARD_TABLE_OUTPUTS)]
    ffn_tables = (
        run_start,
        run_blocks,
        outgoing_offset,
        push_blocks,
        push_destination,
        push_rows,
        arrival_offset,
    )
    return (
        run_rows,
        arrival,
        routed_row,
        ffn_tables,
        expert_rows.reshape(experts_per_shard),
        expert_base.reshape(experts_per_shard),
        counts,
        visit.reshape(experts_per_shard),
        *outs[len(SHARD_TABLE_OUTPUTS) :],
    )
