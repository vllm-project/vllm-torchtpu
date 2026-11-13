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
"""The fused expert-parallel MoE layer.

fused_ep_moe validates the operands, builds and caches the FFN kernel for
the layer's shape, and runs one shard_map over a power-of-two
expert-parallel mesh of 1 to 32 shards.

The per-shard body (_pallas_step) uses Pallas programs throughout. The
select kernel writes the routing message, the transport starts the token
exchange, and the shard-tables kernel builds the routing tables and
forwards the transport's middle rounds. A sparse-core scatter fills the
row tables. The FFN kernel computes the expert outputs and pushes them
to their destination shards, where a sparse-core gather and TensorCore
weighted sum combine the arrivals.
"""
import math
import threading

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.pallas import tpu as pltpu

from . import combine as combine_lib
from . import device, ffn, layout
from . import routing_tables as tables
from . import routing_tables_kernel as tables_kernel
from . import row_gather_kernel as gather
from . import row_tables_kernel, transport, vmem
from .config import (COMBINE_TOKENS_PER_TILE_DEFAULT,
                     GATHER_BLOCK_ROWS_DEFAULT, TABLES_BLOCKS_PER_STEP_MAX,
                     Config, sorted_rows_by_shape)
from .ffn_kernel import ffn_kernel
from .formats import WEIGHT_BLOCK_DEFAULT, WeightFormat, weight_form
from .layout import (LANES, MIB, ROW_BLOCK, SUBLANES, align_up,
                     row_lane_blocks, scale_table_rows)
from .rowquant import FP8
from .select_kernel import select_top_k


def _select(logits,
            *,
            top_k,
            tokens_per_shard,
            renormalize,
            drop_nan_rows,
            message_rows,
            pair_rows=None):
    """The router on one shard: (weights [tokens, top_k] f32, indices).

    The indices are [tokens, top_k] int32, or with `message_rows` the pallas
    mode's routing message, indices half, slot-major.
    """
    # The kernel's row blocks: whole sublane tiles, so a token count that
    # is not one is padded with rows of no finite logit (routed nowhere,
    # weight zero) and the outputs sliced back. The message form's caller
    # pads to the message's own multiple already.
    real_rows = tokens_per_shard
    tokens_per_shard = align_up(tokens_per_shard, SUBLANES)
    if tokens_per_shard > real_rows:
        if message_rows:
            raise ValueError(f"{real_rows} tokens per shard: the routing "
                             f"message form takes whole sublane tiles")
        logits = jnp.pad(logits, ((0, tokens_per_shard - real_rows), (0, 0)),
                         constant_values=jnp.nan)
    # One block of every row when the rows fit a routing block (a block
    # equal to the whole array is any size), else whole blocks of at least
    # a sublane tile, which a count of whole tiles always has.
    block_rows = tables.MAX_ROUTING_BLOCK
    if tokens_per_shard <= block_rows:
        block_rows = tokens_per_shard
    else:
        while tokens_per_shard % block_rows:
            block_rows //= 2
    if renormalize:
        weights, indices = select_top_k(logits,
                                        top_k=top_k,
                                        block_rows=block_rows,
                                        renormalize=True,
                                        drop_nan_rows=drop_nan_rows,
                                        message_rows=message_rows,
                                        pair_rows=pair_rows)
        return weights[:real_rows], (indices
                                     if message_rows else indices[:real_rows])
    # Without renormalization the weights are the full softmax's scores.
    scores = jax.nn.softmax(logits, axis=-1)
    weights, indices = select_top_k(scores,
                                    top_k=top_k,
                                    block_rows=block_rows,
                                    message_rows=message_rows,
                                    pair_rows=pair_rows)
    routes = jnp.any(jnp.isfinite(scores), axis=-1, keepdims=True)
    weights = jnp.where(routes, weights, 0.0)[:real_rows]
    return weights, (indices if message_rows else indices[:real_rows])


def _routing_from_caller(indices, weights, *, num_experts, message_tokens,
                         message_rows, pair_rows):
    """A caller's routing as the select kernel would have written it:
    (weights [message_tokens, top_k] f32, the routing message's index half
    [message_rows, LANES] int32, slot-major). An index outside the expert
    range, and every padding token, routes nowhere: index num_experts
    (the tables allocate it no routed row) and weight zero."""
    tokens, top_k = indices.shape
    indices = indices.astype(jnp.int32)
    weights = weights.astype(jnp.float32)
    routes = jnp.logical_and(indices >= 0, indices < num_experts)
    indices = jnp.where(routes, indices, jnp.int32(num_experts))
    weights = jnp.where(routes, weights, jnp.float32(0.0))
    if message_tokens > tokens:
        pad = ((0, message_tokens - tokens), (0, 0))
        indices = jnp.pad(indices, pad, constant_values=num_experts)
        weights = jnp.pad(weights, pad)
    index_rows = top_k * message_tokens // LANES
    message = indices.T.reshape(index_rows, LANES)
    parts = [message]
    if pair_rows > index_rows:
        parts.append(
            jnp.full((pair_rows - index_rows, LANES), num_experts, jnp.int32))
    if message_rows > pair_rows:
        parts.append(jnp.zeros((message_rows - pair_rows, LANES), jnp.int32))
    return weights, jnp.concatenate(parts, axis=0)


def _shard_biases(w1_bias, w2_bias, experts_per_shard, w1_cols, hidden):
    """The shard's per-expert biases as the kernel's [G, N] tables."""
    return (None if w1_bias is None else w1_bias.reshape(
        experts_per_shard, w1_cols), None
            if w2_bias is None else w2_bias.reshape(experts_per_shard, hidden))


def stream_blocks_for(inter, activation_block, weight_block=None):
    """The column-block widths a streamed expert may take, widest first:
    divisors of inter that are whole numbers of activation blocks, below
    the whole width. With block-scaled weights a block also covers a whole
    number of SUBLANES weight blocks: a streamed block's scale rows are
    copied as one slice of the [experts, blocks, columns] scale table, and
    a slice along its sublane-tiled block axis has to be whole tiles."""
    unit = activation_block
    if weight_block:
        unit = math.lcm(unit, weight_block * SUBLANES)
    return [b for b in range(inter - unit, 0, -unit) if inter % b == 0]


def expert_buffers_that_fit(experts_per_shard,
                            tile_rows,
                            hidden,
                            inter,
                            *,
                            config,
                            weight_format,
                            weight_block,
                            has_w1_bias,
                            has_w2_bias,
                            no_gate,
                            activation_block=0,
                            result_rows="fp8",
                            token_block=None,
                            token_rows="fp8"):
    """(weight_slots, weight_prefetch, stream_block, stream_rows): the
    expert buffers the kernel is built with. Whole-expert weight slots
    first, the most up to `config.weight_slots` whose VMEM estimate fits
    the chip's budget, the prefetch depth clipped under the count. When no
    count of whole-expert slots fits (or config.stream_block asks for
    it), the expert streams in column blocks: the widest block, then the
    most slots, then the most rows per group down from config.stream_rows,
    that fit. Streaming rounds the intermediate per activation block, so
    it needs one; without one the whole-expert refusal is raised here."""
    limit = device.vmem_limit(config.vmem_fraction)
    result_dtype = jnp.bfloat16 if result_rows == "bf16" else FP8
    form = weight_form(weight_format)

    def needed(slots, block, rows):
        return vmem.estimate_bytes(
            experts_per_shard,
            tile_rows,
            hidden,
            inter,
            weight_slots=slots,
            result_slots=config.result_slots,
            weight_format=weight_format,
            weight_block=weight_block,
            has_w1_bias=has_w1_bias,
            has_w2_bias=has_w2_bias,
            no_gate=no_gate,
            activation_block=activation_block,
            result_dtype=result_dtype,
            stream_block=block,
            stream_rows=rows,
            token_block=token_block,
            intermediate=config.intermediate,
            token_dtype=jnp.bfloat16 if token_rows == "bf16" else FP8)

    def group_rows():
        rows = config.stream_rows
        while rows >= tile_rows:
            yield rows
            rows //= 2

    slots_down = range(config.weight_slots, 1, -1)
    if config.stream_block:
        for slots in slots_down:
            for rows in group_rows():
                if needed(slots, config.stream_block, rows) <= limit:
                    return (slots, min(config.weight_prefetch,
                                       slots - 1), config.stream_block, rows)
        # Nothing fits: the kernel refuses with its own message.
        return (config.weight_slots, config.weight_prefetch,
                config.stream_block, config.stream_rows)
    for slots in slots_down:
        if needed(slots, 0, 0) <= limit:
            return slots, min(config.weight_prefetch, slots - 1), 0, 0
    whole = needed(2, 0, 0)
    if not activation_block:
        raise ValueError(
            f"whole-expert weight buffers of {experts_per_shard} local experts "
            f"of {hidden}x{inter} need {whole / MIB:.1f} MiB of VMEM at two "
            f"slots, over the {limit / MIB:.1f} MiB budget. The expert can "
            "stream in column blocks, which rounds the intermediate per "
            "activation block: set activation_block")
    blocks = stream_blocks_for(inter, activation_block,
                               weight_block if form.block_scaled else None)
    if not blocks:
        raise ValueError(
            f"whole-expert weight buffers of {experts_per_shard} local experts "
            f"of {hidden}x{inter} need {whole / MIB:.1f} MiB of VMEM at two "
            f"slots, over the {limit / MIB:.1f} MiB budget, and no column "
            "block below the whole width can stream them: a streamed block "
            f"covers a whole number of {SUBLANES} scale rows (one sublane "
            f"tile of the scale table) and inter={inter} in weight blocks of "
            f"{weight_block} has {inter // weight_block}")
    for block in blocks:
        for slots in slots_down:
            for rows in group_rows():
                if needed(slots, block, rows) <= limit:
                    return (slots, min(config.weight_prefetch,
                                       slots - 1), block, rows)
    return config.weight_slots, config.weight_prefetch, 0, 0


def pallas_token_multiple(num_shards):
    """The row count per shard the pallas mode's transport and FFN kernel
    take: whole ROW_BLOCK-row blocks and whole transport pieces. A shard
    with fewer tokens is padded up to it with zero rows that route
    nowhere. The select kernel and the combine work on the tokens padded
    to message_token_multiple (cheap programs), and the routed pairs are
    padded on their own (pallas_pair_multiple), so neither the tables
    kernels' step nor the gather's block bounds the rows."""
    return math.lcm(ROW_BLOCK, transport.token_multiple(num_shards))


def message_token_multiple(tokens_per_tile=COMBINE_TOKENS_PER_TILE_DEFAULT):
    """The tokens per shard the select kernel and the combine see: the
    select writes each top-k slot's pair words as whole 128-lane rows and
    the combine sums whole tiles of `tokens_per_tile` tokens, so both see
    the tokens padded to this; the padding tokens carry no finite logit
    and weight zero."""
    return math.lcm(LANES, tokens_per_tile)


# The tables steps the automatic choice considers: powers of two down
# from TABLES_BLOCKS_PER_STEP_MAX.
TABLES_STEP_CANDIDATES = tuple(
    2**i for i in range(TABLES_BLOCKS_PER_STEP_MAX.bit_length() - 1, -1, -1))
# The rank of a weight scale table: [experts, columns] per output channel,
# [experts, blocks, columns] per contraction block.
SCALES_RANK_PER_CHANNEL = 2
SCALES_RANK_PER_BLOCK = 3


def pallas_pair_multiple(blocks_per_step=tables_kernel.BLOCKS_PER_STEP,
                         gather_block_rows=GATHER_BLOCK_ROWS_DEFAULT):
    """The routed pairs per shard the tables kernels and the gather take: a
    whole number of tables steps and of the gather's subcore blocks. The
    pair list is padded up to it with pairs that route nowhere."""
    return math.lcm(tables_kernel.step_pairs_multiple(blocks_per_step),
                    gather.gather_rows_multiple(gather_block_rows))


def pallas_geometry(tokens_per_shard,
                    top_k,
                    num_shards,
                    blocks_per_step,
                    *,
                    tokens_per_tile=COMBINE_TOKENS_PER_TILE_DEFAULT,
                    gather_block_rows=GATHER_BLOCK_ROWS_DEFAULT):
    """(rows per shard, message tokens per shard, pair words per shard) the
    pallas mode's programs run at for a call: the rows padded to their
    multiple, the message tokens to message_token_multiple, then the pairs
    to theirs."""
    rows = align_up(tokens_per_shard, pallas_token_multiple(num_shards))
    message_tokens = align_up(rows, message_token_multiple(tokens_per_tile))
    pairs = align_up(message_tokens * top_k,
                     pallas_pair_multiple(blocks_per_step, gather_block_rows))
    return rows, message_tokens, pairs


def tables_blocks_per_step_for(tokens_per_shard, top_k, num_shards, config):
    """The tables kernels' blocks per step for one call: the configured
    count, or, at 0, the largest count up to TABLES_BLOCKS_PER_STEP_MAX
    whose pair padding is no larger than the smallest count's."""
    if config.tables_blocks_per_step:
        return config.tables_blocks_per_step
    counts = list(TABLES_STEP_CANDIDATES)
    padded = {
        c:
        pallas_geometry(tokens_per_shard,
                        top_k,
                        num_shards,
                        c,
                        tokens_per_tile=config.combine_tokens_per_tile,
                        gather_block_rows=config.gather_block_rows)[2]
        for c in counts
    }
    least = min(padded.values())
    return max(c for c in counts if padded[c] == least)


def _pallas_step(x,
                 logits,
                 w1,
                 w2,
                 w1_scales,
                 w2_scales,
                 w1_bias,
                 w2_bias,
                 routing_indices,
                 routing_weights,
                 *,
                 run_kernel,
                 axis,
                 top_k,
                 renormalize,
                 drop_nan_rows,
                 num_experts,
                 num_shards,
                 tokens_per_shard,
                 hidden,
                 routing_block,
                 routed_rows_per_shard,
                 row_scale_dtype,
                 experts_per_shard,
                 w1_cols,
                 arrival_rows,
                 blocks_per_step,
                 activation_block,
                 token_block,
                 result_rows,
                 elementwise,
                 combine_dtype,
                 config,
                 sorted_rows,
                 mesh_axes,
                 token_dtype,
                 shard_index=None):
    """One shard of the pallas mode. `config` supplies the programs'
    geometry (the gather, scatter and combine settings); `token_block` is
    the token rows' rounding block (their scale planes), `activation_block`
    the intermediate's."""
    real_tokens = tokens_per_shard
    n_scales = transport.scale_planes(hidden, token_block)
    tokens_per_shard, message_tokens, pairs_per_shard = pallas_geometry(
        real_tokens,
        top_k,
        num_shards,
        blocks_per_step,
        tokens_per_tile=config.combine_tokens_per_tile,
        gather_block_rows=config.gather_block_rows)
    # The rows are padded to whole blocks with zero rows, the logits to the
    # message's token count with no finite logit: the select kernel routes
    # a padding token nowhere, its weight is zero, and the output rows past
    # the real tokens are sliced off.
    if tokens_per_shard > real_tokens:
        x = jnp.pad(x, ((0, tokens_per_shard - real_tokens), (0, 0)))
    rows_bf16 = x.astype(jnp.bfloat16)
    message_rows = transport.message_rows(tokens_per_shard, pairs_per_shard,
                                          n_scales)
    if routing_indices is not None:
        weights, routing_message = _routing_from_caller(
            routing_indices,
            routing_weights,
            num_experts=num_experts,
            message_tokens=message_tokens,
            message_rows=message_rows,
            pair_rows=pairs_per_shard // LANES)
    else:
        if message_tokens > real_tokens:
            logits = jnp.pad(logits,
                             ((0, message_tokens - real_tokens), (0, 0)),
                             constant_values=jnp.nan)
        weights, routing_message = _select(logits,
                                           top_k=top_k,
                                           tokens_per_shard=message_tokens,
                                           renormalize=renormalize,
                                           drop_nan_rows=drop_nan_rows,
                                           message_rows=message_rows,
                                           pair_rows=pairs_per_shard // LANES)
    (gathered_rows, send_sem, receive_sems,
     routing_messages) = transport.start_transport(
         rows_bf16,
         routing_message,
         scale_base=transport.scale_row_offset(pairs_per_shard),
         num_shards=num_shards,
         mesh_axes=mesh_axes,
         token_dtype=token_dtype,
         scale_dtype=row_scale_dtype,
         activation_block=token_block,
         elementwise=elementwise,
         shard_index=shard_index)
    scale_rows = scale_table_rows(routed_rows_per_shard)
    scale_words = scale_rows * LANES
    plane_words = transport.quant_tiles(tokens_per_shard) * LANES
    shard_tables = tables_kernel.shard_tables_kernel(
        routing_messages,
        num_experts=num_experts,
        num_shards=num_shards,
        pairs_per_shard=pairs_per_shard,
        routed_rows_per_shard=routed_rows_per_shard,
        block=routing_block,
        row_block=ROW_BLOCK,
        slot_field=tables.alignment_slot_field(num_shards),
        axis=axis,
        transport_refs=(gathered_rows, send_sem, receive_sems),
        # The token table in whole LANES-row blocks: the FFN kernel reads it
        # a block-aligned window at a time.
        zero_tables=(align_up(routed_rows_per_shard,
                              LANES), n_scales * scale_words),
        blocks_per_step=blocks_per_step,
        zero_words_per_step=config.tables_zero_words,
        shard_index=shard_index)
    (_, arrival_rows_own, routed_row, ffn_tables, expert_rows, expert_base,
     counts, visit_order, gathered_rows, token_table,
     scale_table) = shard_tables
    token_of_row, token_scales = row_tables_kernel.scatter_row_tables(
        routed_row,
        routing_messages.reshape(-1),
        token_table,
        scale_table,
        top_k=top_k,
        tokens_per_shard=tokens_per_shard,
        message_words=routing_messages.shape[1] * routing_messages.shape[2],
        scale_offset=(transport.scale_row_offset(pairs_per_shard) *
                      transport.MESSAGE_LANES),
        n_scales=n_scales,
        plane_words=plane_words,
        pairs_per_shard=pairs_per_shard,
        message_tokens=message_tokens,
        chunk_max=config.scatter_chunk_pairs,
        ring=config.scatter_ring)
    token_scales = token_scales.reshape(n_scales, scale_rows, LANES)
    w1_bias, w2_bias = _shard_biases(w1_bias, w2_bias, experts_per_shard,
                                     w1_cols, hidden)
    if sorted_rows:
        # The routed rows sorted into expert order on the sparse cores, so
        # the FFN kernel fetches each tile as one copy. The landing has to
        # be complete before the gather reads the buffer, so the exchange's
        # finish runs here as its own program instead of in the kernel: it
        # reads the landing buffer where the arrivals land (without the
        # memory-space constraint the compiler prefetches a copy into its
        # alternate memory for the call, taken before the arrivals have
        # landed), after the scatter's wait (by then the arrivals have
        # landed), and returns a completion word the gather's index and
        # rows depend on.
        n_sorted = align_up(
            token_of_row.shape[0],
            gather.gather_rows_multiple(config.gather_block_rows))
        index = jnp.pad(token_of_row.astype(jnp.int32),
                        (0, n_sorted - token_of_row.shape[0]))
        landing = pltpu.with_memory_space_constraint(gathered_rows,
                                                     pltpu.MemorySpace.HBM)
        done = transport.finish_call(landing,
                                     send_sem,
                                     receive_sems,
                                     index[:LANES],
                                     num_shards=num_shards,
                                     mesh_axes=mesh_axes,
                                     rows_per_shard=gathered_rows.shape[0] //
                                     num_shards,
                                     shard_index=shard_index)
        landed, done = lax.optimization_barrier((gathered_rows, done))
        index = jnp.where(done[0, 0] > 0, index, 0)
        # A padding row (the tail of an (expert, shard) run, the tail slack,
        # the alignment) carries token zero in the table; thousands of
        # reads of one row serialize in the sparse cores, so every such row
        # reads a row of its own. The
        # live rows are the first push_rows of each run, the run starting
        # run_start blocks into its expert's rows, which start at
        # expert_base: +1 at each run's start, -1 at its end, summed along
        # the rows, a row is live where the running sum is positive.
        run_start, push_rows = ffn_tables[0], ffn_tables[5]
        starts = (expert_base.astype(jnp.int32)[:, None] +
                  run_start.astype(jnp.int32) * ROW_BLOCK).reshape(-1)
        ends = starts + push_rows.reshape(-1).astype(jnp.int32)
        # The table is sized to its bound (every shard's pairs); the FFN
        # kernel reads only the used prefix, so only that is sorted.
        used = jnp.max(
            expert_base.astype(jnp.int32) + expert_rows.astype(jnp.int32))
        n_blocks = n_sorted // ROW_BLOCK
        block_idx = jnp.arange(n_blocks, dtype=jnp.int32)
        block_live = jnp.where(block_idx * ROW_BLOCK < used,
                               jnp.int32(ROW_BLOCK), jnp.int32(0))
        rem = ends & (ROW_BLOCK - 1)
        tail_block = jnp.where(rem != 0, ends // ROW_BLOCK, n_blocks)
        block_live = block_live.at[tail_block].set(rem, mode="drop")
        live = (jnp.arange(ROW_BLOCK, dtype=jnp.int32)[None, :]
                < block_live[:, None]).reshape(-1)
        row = jnp.arange(n_sorted, dtype=jnp.int32)
        index = jnp.where(live, index, row % gathered_rows.shape[0])
        rows_in = gather.gather_rows_prefix(
            landed,
            index,
            gather.prefix_run(used, n_sorted, config.gather_block_rows),
            block_rows=config.gather_block_rows,
            ring=gather.ring_for(landed.shape[1],
                                 jnp.dtype(landed.dtype).itemsize,
                                 ring=config.gather_ring,
                                 block_rows=config.gather_block_rows))
        kernel_sems = None
    else:
        rows_in, kernel_sems = gathered_rows, (send_sem, receive_sems)
    arrivals, arrival_scales = run_kernel(ffn_tables,
                                          token_of_row,
                                          expert_rows,
                                          expert_base,
                                          visit_order,
                                          counts,
                                          rows_in,
                                          token_scales,
                                          w1,
                                          w2,
                                          w1_scales,
                                          w2_scales,
                                          w1_bias,
                                          w2_bias,
                                          arrival_rows=arrival_rows,
                                          transport_sems=kernel_sems,
                                          shard_index=shard_index)
    if result_rows == "bf16":
        arrival_scales = None
    # The gather's index: the arrival row of every (token, slot) pair the
    # combine sums, token-major (the tables hold it slot-major), in whole
    # gather blocks. A pair that routes nowhere (a padding token, a dropped
    # row, a padding pair) has weight zero and would take arrival row zero
    # from the tables; thousands of reads of one row serialize in the
    # sparse cores, so every such pair reads a row of its own instead.
    n_pairs = top_k * message_tokens
    n_gather = align_up(n_pairs,
                        gather.gather_rows_multiple(config.gather_block_rows))
    pair_index = (arrival_rows_own.reshape(-1)[:n_pairs].reshape(
        top_k, message_tokens).T.reshape(-1))
    pair_index = jnp.pad(pair_index, (0, n_gather - n_pairs))
    live = weights.reshape(-1) > 0
    live = jnp.pad(live, (0, n_gather - live.shape[0]))
    pair_index = jnp.where(
        live, pair_index,
        jnp.arange(n_gather, dtype=jnp.int32) % arrival_rows)
    out = combine_lib.combine(arrivals,
                              arrival_scales,
                              pair_index,
                              weights,
                              x.dtype,
                              hidden=hidden,
                              block_rows=config.gather_block_rows,
                              ring=gather.ring_for(
                                  arrivals.shape[1],
                                  jnp.dtype(arrivals.dtype).itemsize,
                                  ring=config.gather_ring,
                                  block_rows=config.gather_block_rows),
                              tokens_per_tile=config.combine_tokens_per_tile,
                              unroll=config.combine_unroll,
                              dtype=combine_dtype)
    return out[:real_tokens]


# One shard_map per layer configuration: a fresh body would re-trace the
# whole kernel for every layer of the model.
_LAYER_CACHE = {}
_LAYER_CACHE_LOCK = threading.Lock()

WRAP_IN_JIT = True


def _cached_layer_program(key, build):
    """The jitted shard_map for one layer shape, built once per process.
    Jitted so the layer is one program whether the caller is inside a jit
    or not (the memory-space constraints on the kernels' operands only
    lower under jit). A model's identical layers then trace the kernel once
    instead of once per layer."""
    body = _LAYER_CACHE.get(key)
    if body is None:
        with _LAYER_CACHE_LOCK:
            body = _LAYER_CACHE.get(key)
            if body is None:
                body = _LAYER_CACHE[key] = build()
    return body


def fused_ep_moe(x,
                 w1,
                 w2,
                 w1_scales,
                 w2_scales,
                 router_logits,
                 w1_bias=None,
                 w2_bias=None,
                 *,
                 top_k,
                 renormalize,
                 mesh,
                 tile_rows,
                 routing_block=None,
                 routed_rows_per_shard=None,
                 weight_format=WeightFormat.FP8,
                 weight_block=None,
                 activation="silu",
                 swiglu_limit=ffn.SWIGLU_LIMIT_DEFAULT,
                 drop_nan_rows=False,
                 no_gate=False,
                 routing=None,
                 config=None,
                 shard_index=None,
                 experts_per_shard=None):
    """One MoE layer under expert parallelism, through the fused kernel.

    Args:
      x: [tokens, hidden] activations, sharded over the mesh axis.
      w1: [experts, hidden, 2 * inter] (gate and up halves side by side.
        [experts, hidden, inter] with no_gate), in the format's dtype.
      w2: [experts, inter, hidden].
      w1_scales, w2_scales: f32, per output channel ([experts, N]) for fp8,
        per contraction block ([experts, blocks, N]) for fp4 and for
        fp8_block (fp8 weights with one scale per `weight_block` rows of
        the contraction, the DeepSeek-family checkpoints' form), None for
        bf16.
      router_logits: [tokens, experts], any float type. Selection runs in
        float32. None when `routing` is given.
      routing: the caller's routing instead of the kernel's top-k:
        (indices [tokens, top_k] int32, weights [tokens, top_k] float).
        For routers the select kernel does not compute (sigmoid scoring,
        an expert bias, group-limited top-k): the serving stack selects
        and weights, the kernel moves rows and sums. An index outside
        the expert range routes nowhere with weight zero. renormalize
        and drop_nan_rows do not apply.
      w1_bias, w2_bias: optional [experts, 1, N] f32, added to the post-scale
        halves before the activation and to the down projection before it
        travels between shards.
      top_k: selections per token, at least 1 and at most the expert count.
      renormalize: whether the selected weights are a softmax over the
        selected logits (True) or the full softmax's scores.
      mesh: the mesh. The experts are split over its axis named
        layout.MESH_AXIS; every other axis holds a replica of them (data
        parallelism over the rest of the machine), and the token rows
        split over all of them.
      tile_rows: rows per FFN tile, a multiple of 8.
      routing_block: pairs per routing block. Derived when None.
      routed_rows_per_shard: routed rows per shard. The no-drop bound
        when None.
      weight_format: "fp8", "fp8_block", "fp4", "int4" or "bf16".
      weight_block: contraction rows per fp4 scale. WEIGHT_BLOCK_DEFAULT
        when None.
      activation: "silu" or "swigluoai" ("silu", "gelu" or "relu" with
        no_gate).
      swiglu_limit: the clip of the clamped swiglu ("swigluoai"), the
        model's swiglu_limit (GPT-OSS ships 7.0); ignored by the other
        activations.
      drop_nan_rows: route a row with any NaN logit nowhere, as the stock
        layer does, at one boolean sweep of the logits per call. Off, only
        all-garbage rows are dropped.
      no_gate: a single projection with no gate half.
      config: a Config. None takes Config.for_shape: the defaults, with the
        fields TUNED_BY_SHAPE holds for this shape on top.

    Returns:
      [tokens, hidden] in x's dtype, sharded like x.
    """
    form = weight_form(weight_format)
    weight_format = form.name
    weight_block = (WEIGHT_BLOCK_DEFAULT
                    if weight_block is None else int(weight_block))
    for name, w in (("w1", w1), ("w2", w2)):
        if jnp.dtype(w.dtype) != jnp.dtype(form.weight_dtype):
            raise ValueError(
                f"weight format {weight_format!r} takes "
                f"{jnp.dtype(form.weight_dtype).name} expert "
                f"weights and {name} is {jnp.dtype(w.dtype).name}")
    tokens, hidden = x.shape
    num_experts = w1.shape[0]
    inter = w1.shape[2] // (1 if no_gate else 2)
    if not 1 <= top_k <= num_experts:
        raise ValueError(f"top_k {top_k}: the kernel takes 1 to "
                         f"{num_experts} selections per token")
    if routing is not None:
        routing_indices, routing_weights = routing
        for name, r in (("indices", routing_indices), ("weights",
                                                       routing_weights)):
            if r.shape != (tokens, top_k):
                raise ValueError(f"routing {name} is {tuple(r.shape)}; the "
                                 f"caller's routing is [tokens, top_k] = "
                                 f"{(tokens, top_k)}")
        if not jnp.issubdtype(routing_indices.dtype, jnp.integer):
            raise ValueError(f"routing indices are {routing_indices.dtype}; "
                             "expert ids are integers")
        if not jnp.issubdtype(routing_weights.dtype, jnp.floating):
            raise ValueError(f"routing weights are {routing_weights.dtype}; "
                             "a float type is expected")
    elif router_logits is None:
        raise ValueError("router_logits is None and no routing was given")
    w1_cols = inter if no_gate else 2 * inter
    ndim = (SCALES_RANK_PER_BLOCK
            if form.block_scaled else SCALES_RANK_PER_CHANNEL)
    for name, s, n in (("w1_scales", w1_scales, w1_cols), ("w2_scales",
                                                           w2_scales, hidden)):
        if not form.has_scales:
            if s is not None:
                raise ValueError(f"{name} is {tuple(s.shape)}. bf16 weights "
                                 "carry no scales, so None is expected")
            continue
        if (s is None or s.ndim != ndim or s.shape[0] != num_experts
                or s.shape[-1] != n):
            want = ((num_experts, "blocks", n) if form.block_scaled else
                    (num_experts, n))
            raise ValueError(
                f"{name} is {None if s is None else tuple(s.shape)}. "
                f"the {form.scale_layout} form {want} is expected")
    axis = transport.MESH_AXIS
    if axis not in mesh.axis_names:
        raise ValueError(f"the mesh's axes are {mesh.axis_names}: the experts "
                         f"are split over the axis named {axis!r}; every "
                         "other axis holds a replica of them")
    num_shards = mesh.shape[axis]
    replicas = 1
    for name in mesh.axis_names:
        if name != axis:
            replicas *= mesh.shape[name]
    if tokens % (num_shards * replicas):
        raise ValueError(f"{tokens} tokens do not split over the mesh's "
                         f"{num_shards * replicas} cores")
    if experts_per_shard is None:
        experts_per_shard = num_experts // num_shards
    elif num_experts < num_shards * experts_per_shard:
        raise ValueError(
            f"experts_per_shard {experts_per_shard} over {num_shards} shards needs {num_shards * experts_per_shard} "
            f"experts, the weights carry {num_experts}")
    else:
        # the weight arrays carry more experts per shard than the kernel routes over (an expert placement's copy
        # slots after the primaries): the routing tables and the FFN visit the first experts_per_shard of every
        # shard's block, the rest of the block is never read
        num_experts = num_shards * experts_per_shard
    tokens_per_shard = tokens // (num_shards * replicas)
    # shard_index: int32 [num_shards, 1], the shard's own index handed to the
    # transport and shard-tables programs as data (layout.SHARD_INDEX_OPERAND,
    # the torch bridge's path); None takes an arange, which is fine under a
    # plain jit with a mesh and not under the bridge (a jit-internal
    # constant sharded over the axis compiles to a partition-id there).
    with_shard = layout.SHARD_INDEX_OPERAND or shard_index is not None
    if with_shard:
        if replicas != 1:
            raise ValueError(
                "the shard-index operand takes a mesh whose only "
                f"axis is {axis!r}; this one has {mesh.axis_names}")
        if shard_index is None:
            shard_index = jnp.arange(num_shards,
                                     dtype=jnp.int32).reshape(num_shards, 1)
        if tuple(shard_index.shape) != (num_shards, 1):
            raise ValueError(f"shard_index is {tuple(shard_index.shape)}; "
                             f"one int32 per shard, [{num_shards}, 1], is "
                             "expected")
        shard_index = shard_index.astype(jnp.int32)
    if config is None:
        config = Config.for_shape(hidden=hidden,
                                  inter=inter,
                                  experts_per_shard=experts_per_shard,
                                  top_k=top_k)
    partition = jax.sharding.PartitionSpec
    token_dtype = config.token_dtype
    bound = tables.routed_rows_bound(tokens, top_k, num_experts, tile_rows)
    if routed_rows_per_shard is None:
        routed_rows_per_shard = bound
    transport.check_width(num_shards)
    blocks_per_step = tables_blocks_per_step_for(tokens_per_shard, top_k,
                                                 num_shards, config)
    if routing_block is None:
        _, _, pairs = pallas_geometry(
            tokens_per_shard,
            top_k,
            num_shards,
            blocks_per_step,
            tokens_per_tile=config.combine_tokens_per_tile,
            gather_block_rows=config.gather_block_rows)
        routing_block = tables_kernel.shard_tables_block(
            pairs, blocks_per_step)
    if routed_rows_per_shard % tile_rows:
        raise ValueError(
            f"routed_rows_per_shard {routed_rows_per_shard} is not a "
            f"multiple of tile_rows {tile_rows}. {bound} works")
    if routed_rows_per_shard < bound:
        raise ValueError(
            f"routed_rows_per_shard {routed_rows_per_shard} is below the "
            f"no-drop bound {bound}: one shard's routed rows could bleed "
            "into the next")
    has_w1_bias = w1_bias is not None
    has_w2_bias = w2_bias is not None
    activation_block = config.activation_block
    if activation_block and (hidden % activation_block
                             or inter % activation_block):
        raise ValueError(
            f"activation block {activation_block} does not divide both "
            f"hidden={hidden} and inter={inter}: every row of each matmul's "
            "input is a whole number of blocks")
    token_block = config.token_rows_block
    if token_block and hidden % token_block:
        raise ValueError(f"token block {token_block} does not divide "
                         f"hidden={hidden}: every token row is a whole "
                         "number of blocks")

    def buffers(token_block):
        return expert_buffers_that_fit(experts_per_shard,
                                       tile_rows,
                                       hidden,
                                       inter,
                                       config=config,
                                       weight_format=weight_format,
                                       weight_block=weight_block,
                                       has_w1_bias=has_w1_bias,
                                       has_w2_bias=has_w2_bias,
                                       no_gate=no_gate,
                                       activation_block=activation_block,
                                       result_rows=config.result_rows,
                                       token_block=token_block,
                                       token_rows=config.token_rows)

    weight_slots, weight_prefetch, stream_block, stream_rows = buffers(
        token_block)
    if stream_block and config.token_block is None and not (form.block_scaled):
        # A streamed expert needs the blocked intermediate and nothing
        # else: the token rows use one scale per row, allowing the up
        # projection to use a single matmul and rescale. An explicit
        # token_block is preserved; block-scaled weights require their block.
        token_block = 0
        weight_slots, weight_prefetch, stream_block, stream_rows = buffers(
            token_block)
    # Sorted rows fetch each tile as one copy from rows the sparse cores
    # sorted into expert order (Config.sorted_rows, or the choice by the
    # shard's shape); a streamed build fetches per row instead (its body
    # reads the token table a block at a time), and so does a width whose
    # lane blocks are not whole gather words (the gather moves rows as
    # 32-bit words): the sort is skipped there rather than refused. A table
    # whose sort program is over the sparse core's program size (a width
    # whose rows do not move as one copy per block, at a large table) is
    # skipped by the choice by shape and refused when the sort was asked
    # for.
    if config.token_rows == "bf16":
        if form.block_scaled:
            raise ValueError("bf16 token rows take fp8 weights with one scale "
                             "per channel or bf16 weights, not the "
                             f"block-scaled {form.name} form")
        if stream_block and form.has_scales:
            raise ValueError("bf16 token rows with fp8 weights take a "
                             "whole-expert build: the up matmul runs on a "
                             "per-expert bf16 copy of w1")
    sorted_rows = config.sorted_rows
    if sorted_rows is None:
        sorted_rows = sorted_rows_by_shape(experts_per_shard, tokens_per_shard)
    sorted_rows = (sorted_rows and not stream_block and gather.rows_gatherable(
        row_lane_blocks(hidden), token_dtype))
    if sorted_rows:
        n_sorted = align_up(
            routed_rows_per_shard,
            gather.gather_rows_multiple(config.gather_block_rows))
        per_subcore = n_sorted // gather.subcore_mesh()[1]
        bundles = gather.sort_program_bundles(per_subcore,
                                              row_lane_blocks(hidden),
                                              token_dtype,
                                              config.gather_block_rows)
        if bundles > gather.SORT_PROGRAM_BUNDLE_LIMIT:
            if config.sorted_rows:
                raise ValueError(
                    f"sorted rows: the sort program of a {n_sorted}-row table "
                    f"of {hidden}-wide rows is about {bundles} bundles, over "
                    f"the sparse core's {gather.SORT_PROGRAM_BUNDLE_LIMIT}. "
                    "Rows of a power of two words move as one copy per "
                    "block at 23 bundles; this width's blocks cost 92 and "
                    "1.25 a row. Fewer tokens a call, a wider gather block "
                    "(Config.gather_block_rows), or Config.sorted_rows=None "
                    "or False")
            sorted_rows = False
    run_kernel = ffn_kernel(
        experts_per_shard=experts_per_shard,
        tile_rows=tile_rows,
        hidden=hidden,
        inter=inter,
        num_shards=num_shards,
        mesh_axes=tuple(mesh.axis_names),
        weight_format=weight_format,
        weight_block=weight_block,
        routed_rows=routed_rows_per_shard,
        activation=activation,
        swiglu_limit=swiglu_limit,
        has_w1_bias=has_w1_bias,
        has_w2_bias=has_w2_bias,
        no_gate=no_gate,
        pallas_transport=not sorted_rows,
        weight_slots=weight_slots,
        weight_prefetch=weight_prefetch,
        result_slots=config.result_slots,
        row_fetch_ahead=config.row_fetch_ahead,
        bounds_checks=config.bounds_checks,
        activation_block=activation_block,
        result_rows=config.result_rows,
        elementwise_dtype=config.elementwise_dtype,
        stream_block=stream_block,
        stream_rows=stream_rows,
        vmem_fraction=config.vmem_fraction,
        activation_columns_per_pass=config.activation_columns_per_pass,
        weight_dma_priority=config.weight_dma_priority,
        token_block=token_block,
        intermediate=config.intermediate,
        sorted_rows=sorted_rows,
        token_rows=config.token_rows)
    # The arrival buffer: every routed pair of this shard's tokens plus the
    # row-block padding of every expert's run, in whole blocks.
    arrival_rows = align_up(
        tokens_per_shard * top_k + (ROW_BLOCK - 1) * num_experts, ROW_BLOCK)
    shared = dict(run_kernel=run_kernel,
                  axis=axis,
                  top_k=top_k,
                  renormalize=renormalize,
                  drop_nan_rows=drop_nan_rows,
                  num_experts=num_experts,
                  num_shards=num_shards,
                  tokens_per_shard=tokens_per_shard,
                  hidden=hidden,
                  routing_block=routing_block,
                  routed_rows_per_shard=routed_rows_per_shard,
                  experts_per_shard=experts_per_shard,
                  w1_cols=w1_cols,
                  arrival_rows=arrival_rows,
                  blocks_per_step=blocks_per_step,
                  activation_block=activation_block,
                  token_block=token_block,
                  result_rows=config.result_rows,
                  elementwise=config.elementwise,
                  combine_dtype=config.combine,
                  config=config,
                  sorted_rows=sorted_rows,
                  mesh_axes=tuple(mesh.axis_names),
                  token_dtype=token_dtype)

    handed_in = routing is not None

    def shard_body(x, w1, w2, *optional):
        it = iter(optional)
        if handed_in:
            logits, indices, weights = None, next(it), next(it)
        else:
            logits, indices, weights = next(it), None, None
        w1_scales = next(it) if form.has_scales else None
        w2_scales = next(it) if form.has_scales else None
        w1_bias = next(it) if has_w1_bias else None
        w2_bias = next(it) if has_w2_bias else None
        shard_l = next(it).reshape(-1) if with_shard else None
        return _pallas_step(x,
                            logits,
                            w1,
                            w2,
                            w1_scales,
                            w2_scales,
                            w1_bias,
                            w2_bias,
                            indices,
                            weights,
                            row_scale_dtype=config.row_scale_dtype,
                            shard_index=shard_l,
                            **shared)

    router = tuple(routing) if handed_in else (router_logits, )
    scales = (w1_scales, w2_scales) if form.has_scales else ()
    biases = tuple(b for b in (w1_bias, w2_bias) if b is not None)
    shard_args = (shard_index, ) if with_shard else ()
    args = (x, w1, w2) + router + scales + biases + shard_args
    # The token rows split over every mesh axis; the experts over the
    # expert-parallel axis alone, copied over the others.
    rows_spec = partition(tuple(mesh.axis_names))
    experts_spec = partition(axis)
    in_specs = ((rows_spec, experts_spec, experts_spec) +
                (rows_spec, ) * len(router) + (experts_spec, ) *
                (len(scales) + len(biases)) +
                (experts_spec, ) * len(shard_args))
    args = tuple(
        jax.device_put(a, jax.sharding.NamedSharding(mesh, spec))
        for a, spec in zip(args, in_specs))
    bias_dtypes = (w1_bias.dtype if has_w1_bias else None,
                   w2_bias.dtype if has_w2_bias else None)
    key = (mesh, tokens, hidden, num_experts, inter, top_k, bool(renormalize),
           tile_rows, routing_block, routed_rows_per_shard, weight_format,
           weight_block, activation, swiglu_limit, bool(no_gate), config,
           bool(drop_nan_rows), bool(with_shard), x.dtype, w1.dtype, w2.dtype,
           w1_scales.dtype if form.has_scales else None,
           tuple(r.dtype for r in router)) + bias_dtypes

    # WRAP_IN_JIT: the layer's own callers get one compiled program per shape.
    # A caller that is already inside a jit -- the torch bridge exports the
    # whole op as one jitted function -- sets it False. On the export path,
    # XLA's SPMD partitioner rejects a nested jit with "Side-effect HLO
    # must have sharding" for the transport program.
    def build():
        sm = jax.shard_map(shard_body,
                           mesh=mesh,
                           in_specs=in_specs,
                           out_specs=rows_spec,
                           check_vma=False)
        return jax.jit(sm) if WRAP_IN_JIT else sm

    body = _cached_layer_program(key, build)
    return body(*args)
