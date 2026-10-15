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
"""The expert FFN kernel: one Pallas TPU program computes one shard's experts
and pushes each result row to the shard that owns its token.

The program (kernel_body):

  prologue    the resident tables land (weight scales, biases). The weight
              matrices of the first `weight_prefetch` experts are issued. An
              all-shard barrier. In pallas mode, the transport's final
              waits, so every reader of the gathered token rows reads
              them complete. The first expert's first `row_fetch_ahead`
              tiles
              are issued.
  Visit loop  for each local expert with rows, most rows first:
                wait its weight matrices. Issue the refill for the expert
                `weight_prefetch` ahead into the slot the previous expert
                last read.
                For each tile of `tile_rows` rows, at result slot
                (tile count mod result_slots): wait the slot's earlier
                commits. Wait this tile's rows. Issue the tile
                `row_fetch_ahead` ahead. The two matmuls and the
                activation. The epilogue (row scale, weight scale, bias,
                fp8 quantization). Stage the rows and their scales in the
                slot. Commit the tile's part of every (expert, destination)
                run to the outgoing rows.
                Issue the next expert's first tiles. Drain the slots'
                commits. Push the expert's runs to their destination shards
                at true length (the own-shard run is copied locally).
  Drain       every push and commit still in flight. An all-shard barrier.
"""

import dataclasses
import logging
import threading
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v3 import device, ffn, layout, transport, vmem
from vllm_torchtpu.kernels.fused_moe.v3 import ffn_body as body
from vllm_torchtpu.kernels.fused_moe.v3 import ffn_expert_streamed as streamed
from vllm_torchtpu.kernels.fused_moe.v3 import ffn_expert_whole as whole
from vllm_torchtpu.kernels.fused_moe.v3.config import (
    ACTIVATION_COLUMNS_PER_PASS_DEFAULT,
    RESULT_SLOTS_DEFAULT,
    ROW_FETCH_AHEAD_DEFAULT,
    WEIGHT_DMA_PRIORITY_DEFAULT,
    WEIGHT_PREFETCH_DEFAULT,
    WEIGHT_SLOTS_DEFAULT,
)
from vllm_torchtpu.kernels.fused_moe.v3.formats import (
    PACKED_ROW_TILE,
    WEIGHT_BLOCK_DEFAULT,
    WeightFormat,
    weight_form,
)
from vllm_torchtpu.kernels.fused_moe.v3.layout import LANES, ROW_BLOCK, SUBLANES

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Tables:
    """The scalar-prefetch tables (SMEM), in operand order. Block units are
    ROW_BLOCK-row blocks. E indexes a local expert, d a destination shard."""

    run_start: Any  # [G, num_shards] block e's run for d starts at
    run_blocks: Any  # [G, num_shards] blocks in that run (padded)
    outgoing_offset: Any  # [G, num_shards] outgoing block it fills
    push_blocks: Any  # [G, num_shards] blocks the push to d carries
    push_destination: Any  # [G, num_shards] d's block it lands on
    push_rows: Any  # [G, num_shards] true rows of the run (rows)
    arrival_offset: Any  # [G, num_shards] d's arrival row it lands on
    expert_rows: Any  # [G] padded rows of each local expert
    expert_base: Any  # [G] routed row each local expert's rows start at
    visit_order: Any  # [G] local experts with rows, most rows first
    counts: Any  # [COUNT_ROWS, num_shards] the five counts, per shard


@dataclasses.dataclass(frozen=True)
class Operands:
    """The HBM operands, in operand order."""

    token_of_row: Any  # [routed rows / LANES, 1, LANES] i32 token each
    # routed row computes; read per tile into a
    # scalar-memory window (the whole table
    # outgrows scalar memory past 8192 tokens a call)
    tokens: Any  # [tokens, lane blocks, LANES] fp8 rows
    token_scales: Any  # [scale planes, scale rows, LANES] i32: f32
    # bits, one word per row per plane
    w1: Any  # [G, hidden, cols] weights (packed u32 for fp4)
    w2: Any  # [G, inter, hidden]
    w1_scales: Any  # [G, cols] or [G, blocks, cols] f32, or None
    w2_scales: Any  # [G, hidden] or [G, blocks, hidden] f32, or None
    w1_bias: Any  # [G, cols] f32 or None
    w2_bias: Any  # [G, hidden] f32 or None
    transport_send_sem: Any = None  # pallas mode: the start call's
    transport_receive_sems: Any = None  # two semaphores


@dataclasses.dataclass(frozen=True)
class Outputs:
    """The kernel's output refs: arrival and outgoing rows with scales."""

    arrivals: Any  # [arrival rows, result blocks, LANES] fp8
    arrival_scales: (
        object  # [arrival rows / ROW_BLOCK, 1, ROW_BLOCK] f32, a word per row
    )
    outgoing: Any  # [routed rows, result blocks, LANES] fp8
    outgoing_scales: Any  # [routed rows / ROW_BLOCK, 1, ROW_BLOCK] f32


@dataclasses.dataclass(frozen=True)
class Scratch:
    """The VMEM buffers, in vmem.scratch_arrays' order. A streamed build
    holds a group's rows and scale windows double-buffered (the leading
    axis 2 instead of the result slots), the up-projection scales and bias
    per weight slot, and the group's running sum."""

    rows: Any  # [result slots, tile rows, lane blocks, LANES] fp8
    w1: Any  # [weight slots, ...]
    w2: Any
    # [hidden, cols] fp8, the current expert widened (whole-expert fp4 builds),
    # or None
    w1_fp8: Any
    w2_fp8: Any  # [inter, hidden] fp8, the same
    # [inter, hidden] bf16, the current expert (intermediate "bf16"), or None
    w2_bf16: Any
    # [hidden, cols] bf16, the current expert (bf16 token rows on fp8 weights),
    # or None
    w1_bf16: Any
    w1_scales: Any  # [G, ...] resident, or [weight slots, ...]
    w2_scales: Any
    w1_bias: Any  # [G, cols] or [weight slots, 1, cols] or None
    w2_bias: Any  # [G, hidden] or None
    row_scales: Any  # [result slots, planes, window rows, LANES] i32
    result_rows: Any  # [result slots, tile rows, result blocks, LANES]
    result_scales: Any  # [result slots, tile blocks, 1, ROW_BLOCK] f32
    acc: Any = None  # [stream rows, hidden] f32, streamed builds
    # Scalar-memory windows of the routed-row token table: one per row
    # buffer (result slot, or group buffer when streamed), and the next
    # expert's first tiles (whole-expert builds).
    token_window: Any = None  # [row buffers, blocks, 1, LANES] i32, SMEM
    token_head: Any = None  # [row_fetch_ahead, blocks, 1, LANES] i32


@dataclasses.dataclass(frozen=True)
class Semaphores:
    """DMA semaphores. Waits count bytes per semaphore, so every stream that
    is waited separately has its own."""

    rows: Any  # [result slots] a tile's row stream
    w1: Any  # [weight slots]
    w2: Any  # [weight slots]
    copy: Any  # the resident-table copies
    commit: Any  # [result slots] a slot's commits
    push_send: Any
    push_receive: Any
    commit_scales: Any  # [result slots]
    push_send_scales: Any
    push_receive_scales: Any
    local_hop: Any  # the own-shard run's local copy
    local_hop_scales: Any
    token_window: Any  # [row buffers] the token windows
    token_head: Any  # [row_fetch_ahead] the next expert's first


def _check_build(
    *,
    tile_rows,
    hidden,
    inter,
    weight_format,
    weight_block,
    routed_rows,
    weight_slots,
    weight_prefetch,
    result_slots,
    row_fetch_ahead,
    activation_block,
    stream_block=0,
    stream_rows=0,
    token_block=None,
):
    """Raises ValueError for a build the kernel does not support, naming
    the operand."""
    form = weight_form(weight_format)
    if stream_block:
        if stream_block % LANES or inter % stream_block:
            raise ValueError(
                f"stream block {stream_block}: a whole number of "
                f"{LANES}-lane blocks dividing inter={inter}"
            )
        if not activation_block:
            raise ValueError(
                f"a streamed expert (stream block {stream_block}) rounds its "
                "intermediate per activation block, one block at a time, "
                "so activation_block has to be set; one scale per whole "
                "row cannot be formed "
                "before every block has been seen"
            )
        if stream_block % activation_block:
            raise ValueError(
                f"stream block {stream_block} is not a whole "
                f"number of activation blocks of "
                f"{activation_block}"
            )
        if stream_rows < tile_rows or stream_rows % tile_rows:
            raise ValueError(
                f"stream rows {stream_rows}: a whole number of {tile_rows}-row tiles"
            )
        if form.block_scaled and stream_block % weight_block:
            raise ValueError(
                f"stream block {stream_block} is not a whole "
                f"number of weight blocks of {weight_block}"
            )
        if form.block_scaled and (stream_block // weight_block) % SUBLANES:
            raise ValueError(
                f"stream block {stream_block} covers "
                f"{stream_block // weight_block} weight blocks of "
                f"{weight_block}; a streamed block-scaled expert copies its "
                "scale rows one sublane tile at a time, so that count has "
                f"to be a whole number of {SUBLANES}"
            )
    if activation_block:
        if activation_block % LANES:
            raise ValueError(
                f"activation block {activation_block} is not a "
                f"whole number of {LANES}-lane blocks"
            )
        if hidden % activation_block or inter % activation_block:
            raise ValueError(
                f"activation block {activation_block} does not "
                f"divide both hidden={hidden} and inter={inter}"
            )
        if form.block_scaled and activation_block != weight_block:
            raise ValueError(
                f"activation block {activation_block} against a weight block "
                f"of {weight_block}: the block-scaled path scales its "
                "contraction per weight block, so an activation block has "
                "to be that block"
            )
    # The token rows' own block, checked after the activation block so a
    # build wrong in both names the activation block first.
    token_block = activation_block if token_block is None else token_block
    if token_block:
        if token_block % LANES:
            raise ValueError(
                f"token block {token_block} is not a whole "
                f"number of {LANES}-lane blocks"
            )
        if hidden % token_block:
            raise ValueError(
                f"token block {token_block} does not divide hidden={hidden}"
            )
        if form.block_scaled and token_block != weight_block:
            raise ValueError(
                f"token block {token_block} against a weight block of "
                f"{weight_block}: the block-scaled path scales its contraction "
                "per weight block, so a nonzero token block has to be that "
                "block"
            )
    if not 1 <= weight_prefetch < weight_slots:
        raise ValueError(
            f"weight_prefetch {weight_prefetch} has to be between "
            f"1 and weight_slots - 1 = {weight_slots - 1}"
        )
    if not 1 <= row_fetch_ahead < result_slots:
        raise ValueError(
            f"row_fetch_ahead {row_fetch_ahead} has to be between "
            f"1 and result_slots - 1 = {result_slots - 1}"
        )
    if tile_rows % ROW_BLOCK:
        raise ValueError(
            f"tile height {tile_rows} is not a whole number of "
            f"{ROW_BLOCK}-row blocks, the unit every transport "
            "moves"
        )
    if form.block_scaled:
        if weight_block < 1:
            raise ValueError(
                f"weight block {weight_block}: a scale covers at "
                "least one contraction row"
            )
        if hidden % weight_block or inter % weight_block:
            raise ValueError(
                f"the weight block {weight_block} has to divide "
                f"both hidden={hidden} and inter={inter}"
            )
        if weight_block % PACKED_ROW_TILE:
            raise ValueError(
                f"the weight block {weight_block} is not a whole "
                f"number of the packed-weight row tile "
                f"({PACKED_ROW_TILE} rows)"
            )
    if hidden < LANES or hidden % LANES:
        raise ValueError(
            f"hidden {hidden} is not a whole number of "
            f"{LANES}-lane blocks. The VMEM estimate decides "
            "how wide a row may be"
        )
    if inter < LANES:
        raise ValueError(f"inter {inter} is narrower than one {LANES}-lane block")
    if routed_rows is None or routed_rows % ROW_BLOCK or routed_rows % tile_rows:
        raise ValueError(
            f"routed_rows {routed_rows} must be a whole number of "
            f"{ROW_BLOCK}-row blocks and of {tile_rows}-row "
            "tiles: a tail tile reads a full window"
        )


def build_ffn_kernel(
    *,
    experts_per_shard,
    tile_rows,
    hidden,
    inter,
    num_shards,
    mesh_axes=(layout.MESH_AXIS,),
    weight_format,
    weight_block=WEIGHT_BLOCK_DEFAULT,
    routed_rows,
    activation="silu",
    has_w1_bias=False,
    has_w2_bias=False,
    no_gate=False,
    pallas_transport=False,
    weight_slots=WEIGHT_SLOTS_DEFAULT,
    weight_prefetch=WEIGHT_PREFETCH_DEFAULT,
    result_slots=RESULT_SLOTS_DEFAULT,
    row_fetch_ahead=ROW_FETCH_AHEAD_DEFAULT,
    bounds_checks=False,
    activation_block=0,
    result_rows="fp8",
    elementwise_dtype="float32",
    stream_block=0,
    stream_rows=0,
    swiglu_limit=ffn.SWIGLU_LIMIT_DEFAULT,
    vmem_fraction=device.VMEM_FRACTION,
    activation_columns_per_pass=ACTIVATION_COLUMNS_PER_PASS_DEFAULT,
    weight_dma_priority=WEIGHT_DMA_PRIORITY_DEFAULT,
    token_block=None,
    intermediate="fp8",
    sorted_rows=False,
    token_rows="fp8",
    shard_operand=False,
):
    """Builds the program for one layer shape and returns the function that
    runs it (see `run` below). pallas_transport: the token buffer is the
    transport start call's gathered_rows buffer, still landing, and its two
    semaphores are operands. activation_block: values per fp8 scale of a
    row of each matmul's input (0: one scale per row); the row scales
    arrive as hidden // activation_block planes of the scale table.
    token_block: the token rows' own rounding block when it differs
    (None: activation_block; 0: one scale per row), so the up projection
    is one plain matmul while a streamed intermediate stays blocked.
    result_rows: "fp8" (rows with one scale per row) or "bf16" (rows
    alone; the scale buffers and their transfers are left out).
    elementwise_dtype: the element type of the vector math after each
    matmul (Config.elementwise_dtype). stream_block: 0 holds whole experts
    in the weight slots; a count streams each expert in column blocks of
    that many intermediate columns, with a group of stream_rows rows and
    their float32 running sum resident, the intermediate rounded per
    activation block (Config.stream_block, Config.stream_rows).
    shard_operand: the shard's index arrives as an operand (`shard_index`)
    instead of lax.axis_index."""
    form = weight_form(weight_format)
    if elementwise_dtype not in ("float32", "bfloat16"):
        raise ValueError(
            f"elementwise_dtype {elementwise_dtype!r} is float32 or bfloat16"
        )
    elementwise = jnp.float32 if elementwise_dtype == "float32" else jnp.bfloat16
    if result_rows not in ("fp8", "bf16"):
        raise ValueError(f"result_rows {result_rows!r} is fp8 or bf16")
    result_dtype = ffn.FP8 if result_rows == "fp8" else jnp.bfloat16
    if intermediate not in ("fp8", "bf16"):
        raise ValueError(f"intermediate {intermediate!r} is fp8 or bf16")
    if intermediate == "bf16" and (stream_block or form.block_scaled):
        raise ValueError(
            'intermediate "bf16" runs whole-expert builds with fp8 '
            "or bf16 weights: a streamed or block-scaled build "
            "keeps the fp8 intermediate"
        )
    has_result_scales = result_rows == "fp8"
    if sorted_rows and stream_block:
        raise ValueError(
            "sorted_rows fetches each tile as one copy from rows "
            "sorted into expert order: a whole-expert build; a "
            "streamed build reads its rows by group"
        )
    if pallas_transport:
        transport.check_width(num_shards)
    activations = ffn.NO_GATE_ACTIVATIONS if no_gate else ffn.ACTIVATIONS
    if activation not in activations:
        raise ValueError(f"activation {activation!r} is not one of {activations}")
    _check_build(
        tile_rows=tile_rows,
        hidden=hidden,
        inter=inter,
        weight_format=weight_format,
        weight_block=weight_block,
        routed_rows=routed_rows,
        weight_slots=weight_slots,
        weight_prefetch=weight_prefetch,
        result_slots=result_slots,
        row_fetch_ahead=row_fetch_ahead,
        activation_block=activation_block,
        stream_block=stream_block,
        stream_rows=stream_rows,
        token_block=token_block,
    )
    token_block = activation_block if token_block is None else token_block
    device.check_generation()
    token_dtype = jnp.bfloat16 if token_rows == "bf16" else jnp.float8_e4m3fn
    if form.packed:
        device.check_u32_sublane_tile()
    lane_blocks = layout.row_lane_blocks(hidden)
    result_blocks = layout.result_lane_blocks(hidden)
    scale_lanes = vmem.SCALE_WORDS  # words per block of scales
    tile_blocks = tile_rows // ROW_BLOCK
    window_rows = layout.scale_window_rows(tile_rows)
    scale_planes = hidden // token_block if token_block else 1
    scratch_shapes = vmem.scratch_arrays(
        experts_per_shard,
        tile_rows,
        hidden,
        inter,
        weight_slots=weight_slots,
        result_slots=result_slots,
        weight_format=weight_format,
        weight_block=weight_block,
        has_w1_bias=has_w1_bias,
        has_w2_bias=has_w2_bias,
        no_gate=no_gate,
        activation_block=activation_block,
        result_dtype=result_dtype,
        stream_block=stream_block,
        stream_rows=stream_rows,
        token_block=token_block,
        intermediate=intermediate,
        token_dtype=token_dtype,
    )
    needed = vmem.estimate_bytes(
        experts_per_shard,
        tile_rows,
        hidden,
        inter,
        weight_slots=weight_slots,
        result_slots=result_slots,
        weight_format=weight_format,
        weight_block=weight_block,
        has_w1_bias=has_w1_bias,
        has_w2_bias=has_w2_bias,
        no_gate=no_gate,
        activation_block=activation_block,
        result_dtype=result_dtype,
        stream_block=stream_block,
        stream_rows=stream_rows,
        token_block=token_block,
        intermediate=intermediate,
        token_dtype=token_dtype,
    )
    limit = device.vmem_limit(vmem_fraction)
    if needed > limit:
        raise ValueError(
            f"the kernel's VMEM buffers need "
            f"{needed / layout.MIB:.1f} MiB, over the "
            f"{limit / layout.MIB:.1f} MiB budget, for {experts_per_shard} local "
            f"experts of {hidden}x{inter} at tile {tile_rows}"
        )
    # One more scalar-prefetch table under the operand mode: the shard's
    # own index, read instead of the mesh axis.
    n_tables = len(dataclasses.fields(Tables)) + int(shard_operand)
    # Which optional operands and scratch buffers this build carries, in
    # the dataclasses' field order; the unpack below reads the flat ref
    # list by these, never by a count.
    operand_present = dict(
        token_of_row=True,
        tokens=True,
        token_scales=True,
        w1=True,
        w2=True,
        w1_scales=form.has_scales,
        w2_scales=form.has_scales,
        w1_bias=has_w1_bias,
        w2_bias=has_w2_bias,
        transport_send_sem=pallas_transport,
        transport_receive_sems=pallas_transport,
    )
    scratch_present = dict(
        rows=True,
        w1=True,
        w2=True,
        w1_fp8=form.packed and not stream_block,
        w2_fp8=form.packed and not stream_block,
        w2_bf16=intermediate == "bf16",
        w1_bf16=(token_rows == "bf16" and form.has_scales and not form.block_scaled),
        w1_scales=form.has_scales,
        w2_scales=form.has_scales,
        w1_bias=has_w1_bias,
        w2_bias=has_w2_bias,
        row_scales=True,
        result_rows=True,
        result_scales=True,
        acc=bool(stream_block),
        token_window=True,
        token_head=True,
    )
    for cls, present in ((Operands, operand_present), (Scratch, scratch_present)):
        assert [f.name for f in dataclasses.fields(cls)] == list(present), cls
    # The array operands take HBM specs; the transport's two semaphores
    # take semaphore specs of their own below.
    n_operands = sum(
        present
        for name, present in operand_present.items()
        if not name.startswith("transport_")
    )

    def take_fields(it, present):
        return {name: (next(it) if there else None) for name, there in present.items()}

    def unpack(refs):
        it = iter(refs)
        tables = Tables(*(next(it) for _ in dataclasses.fields(Tables)))
        shard_ref = next(it) if shard_operand else None
        operands = Operands(**take_fields(it, operand_present))
        if form.packed:
            # The four-bit weights as packed 32-bit words, so the DMA moves
            # half the bytes. This ref-level bitcast is a private jax
            # method with no public equivalent, so its absence is an error.
            if not hasattr(operands.w1, "bitcast"):
                raise NotImplementedError(
                    "the four-bit weight stream views a memory ref as packed "
                    f"32-bit words through the private "
                    f"{type(operands.w1).__name__}.bitcast, which jax "
                    f"{device.jax_version()} no longer carries"
                )
            operands = dataclasses.replace(
                operands,
                w1=operands.w1.bitcast(jnp.uint32),
                w2=operands.w2.bitcast(jnp.uint32),
            )
        outputs = Outputs(*(next(it) for _ in dataclasses.fields(Outputs)))
        scratch = Scratch(**take_fields(it, scratch_present))
        sems = Semaphores(*(next(it) for _ in dataclasses.fields(Semaphores)))
        return tables, operands, outputs, scratch, sems, shard_ref

    build = body.Build(
        tile_rows=tile_rows,
        hidden=hidden,
        inter=inter,
        num_shards=num_shards,
        mesh_axes=tuple(mesh_axes),
        token_dtype=token_dtype,
        experts_per_shard=experts_per_shard,
        form=form,
        weight_block=weight_block,
        has_w1_bias=has_w1_bias,
        has_w2_bias=has_w2_bias,
        no_gate=no_gate,
        activation=activation,
        swiglu_limit=swiglu_limit,
        pallas_transport=pallas_transport,
        weight_slots=weight_slots,
        weight_prefetch=weight_prefetch,
        result_slots=result_slots,
        row_fetch_ahead=row_fetch_ahead,
        activation_block=activation_block,
        token_block=token_block,
        result_dtype=result_dtype,
        elementwise=elementwise,
        has_result_scales=has_result_scales,
        stream_block=stream_block,
        stream_rows=stream_rows,
        activation_columns_per_pass=activation_columns_per_pass,
        weight_dma_priority=weight_dma_priority,
        intermediate=intermediate,
        sorted_rows=sorted_rows,
        lane_blocks=lane_blocks,
        result_blocks=result_blocks,
        tile_blocks=tile_blocks,
        scale_planes=scale_planes,
        scale_lanes=scale_lanes,
        window_rows=window_rows,
        shard_operand=shard_operand,
    )
    step = streamed if stream_block else whole

    def kernel_body(*refs):
        tables, operands, outputs, scratch, sems, shard_ref = unpack(refs)
        ctx = body.Context(
            b=build,
            tables=tables,
            operands=operands,
            outputs=outputs,
            scratch=scratch,
            sems=sems,
            n_visit=None,
            me=None,
        )
        ctx = dataclasses.replace(
            ctx,
            n_visit=body.count(ctx, body.COUNT_VISITS),
            me=(
                shard_ref[0]
                if shard_ref is not None
                else lax.axis_index(layout.MESH_AXIS)
            ),
        )
        # ---- prologue: the resident tables, the first weight refills ----
        body.resident_copies(ctx)
        step.first_refills(ctx)
        transport.all_shards_barrier(num_shards, build.mesh_axes, shard_operand)
        # ---- the program ----
        if pallas_transport:
            transport.finish(
                transport.Exchange(
                    operands.tokens,
                    operands.transport_send_sem,
                    operands.transport_receive_sems,
                    ctx.me,
                    operands.tokens.shape[0] // num_shards,
                    num_shards,
                    mesh_axes=build.mesh_axes,
                    shard_operand=shard_operand,
                )
            )
        step.issue_first(ctx)
        lax.fori_loop(
            0,
            ctx.n_visit,
            lambda visit_i, carry: step.visit(ctx, visit_i, carry),
            step.initial_carry(ctx),
        )
        body.drain_transport(ctx)
        transport.all_shards_barrier(num_shards, build.mesh_axes, shard_operand)

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    sem = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)
    scratch = [pltpu.VMEM(shape, dtype) for _, shape, dtype in scratch_shapes]
    # The routed-row token table's windows in scalar memory: the rows of one
    # row buffer (a group when streamed, a tile otherwise) per buffer, and
    # the next expert's first tiles.
    row_buffers, window_len = (
        (2, stream_rows) if stream_block else (result_slots, tile_rows)
    )
    table_rows = layout.align_up(routed_rows, LANES)  # the layer's table
    table_blocks = table_rows // LANES
    scratch += [
        pltpu.SMEM(
            (row_buffers, body.token_window_blocks(window_len, table_blocks), 1, LANES),
            jnp.int32,
        ),
        pltpu.SMEM(
            (
                row_fetch_ahead,
                body.token_window_blocks(tile_rows, table_blocks),
                1,
                LANES,
            ),
            jnp.int32,
        ),
    ]
    scratch += [
        pltpu.SemaphoreType.DMA((result_slots,)),  # rows
        pltpu.SemaphoreType.DMA((weight_slots,)),  # w1
        pltpu.SemaphoreType.DMA((weight_slots,)),  # w2
        pltpu.SemaphoreType.DMA,  # copy
        pltpu.SemaphoreType.DMA((result_slots,)),  # commit
        pltpu.SemaphoreType.DMA,  # push_send
        pltpu.SemaphoreType.DMA,  # push_receive
        pltpu.SemaphoreType.DMA((result_slots,)),  # commit_scales
        pltpu.SemaphoreType.DMA,  # push_send_scales
        pltpu.SemaphoreType.DMA,  # push_receive_scales
        pltpu.SemaphoreType.DMA,  # local_hop
        pltpu.SemaphoreType.DMA,  # local_hop_scales
        pltpu.SemaphoreType.DMA((row_buffers,)),  # token_window
        pltpu.SemaphoreType.DMA((row_fetch_ahead,)),  # token_head
    ]
    columns_part = (
        f"_p{activation_columns_per_pass}"
        if activation_columns_per_pass != ACTIVATION_COLUMNS_PER_PASS_DEFAULT
        else ""
    )
    priority_part = (
        f"_q{weight_dma_priority}"
        if weight_dma_priority != WEIGHT_DMA_PRIORITY_DEFAULT
        else ""
    )
    name = (
        "fused_ep_moe"
        f"{'' if form.name == WeightFormat.FP8 else '_' + form.name}"
        f"{'' if activation == 'silu' else '_' + activation}"
        f"{'_w1bias' if has_w1_bias else ''}{'_w2bias' if has_w2_bias else ''}"
        f"{'_pallas' if pallas_transport else ''}"
        f"{'_a' + str(activation_block) if activation_block else ''}"
        f"{'_k' + str(token_block) if token_block != activation_block else ''}"
        f"{'_bf16out' if not has_result_scales else ''}"
        f"{'_e16' if elementwise_dtype == 'bfloat16' else ''}"
        f"{'_c' + str(stream_block) + 'r' + str(stream_rows) if stream_block else ''}"
        f"{'_l' + str(swiglu_limit) if activation == 'swigluoai' else ''}"
        f"{columns_part}{priority_part}"
        f"{'_m16' if intermediate == 'bf16' else ''}"
        f"_g{experts_per_shard}_t{tile_rows}_w{weight_slots}_s{result_slots}"
    )

    def pallas_call(arrival_rows):
        if arrival_rows % ROW_BLOCK:
            raise ValueError(
                f"the arrival buffer height {arrival_rows} is not "
                f"a whole number of {ROW_BLOCK}-row blocks"
            )
        out_shape = [
            jax.ShapeDtypeStruct((arrival_rows, result_blocks, LANES), result_dtype),
            jax.ShapeDtypeStruct(
                (arrival_rows // ROW_BLOCK, 1, scale_lanes), jnp.float32
            ),
            jax.ShapeDtypeStruct((routed_rows, result_blocks, LANES), result_dtype),
            jax.ShapeDtypeStruct(
                (routed_rows // ROW_BLOCK, 1, scale_lanes), jnp.float32
            ),
        ]
        return pl.pallas_call(
            kernel_body,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=n_tables,
                in_specs=[hbm] * n_operands + [sem] * (2 * int(pallas_transport)),
                out_specs=[hbm] * len(out_shape),
                scratch_shapes=tuple(scratch),
                grid=(),
            ),
            out_shape=out_shape,
            compiler_params=pltpu.CompilerParams(
                collective_id=layout.COLLECTIVE_ID_FFN,
                vmem_limit_bytes=limit,
                disable_bounds_checks=not bounds_checks,
            ),
            name=name + ("_sorted" if sorted_rows else ""),
        )

    def run(
        tables,
        token_of_row,
        expert_rows,
        expert_base,
        visit_order,
        counts,
        tokens,
        token_scales,
        w1,
        w2,
        w1_scales,
        w2_scales,
        w1_bias=None,
        w2_bias=None,
        *,
        arrival_rows,
        transport_sems=None,
        shard_index=None,
    ):
        """Runs the program on one shard. `tables` is the 7-tuple of
        (expert, destination) tables. token_scales is the scale table in its
        dense [planes, rows, LANES] view (or [rows, LANES] with one plane).
        Returns (arrivals, arrival scales [arrival rows] f32, a word per
        row); the scales are meaningless with bf16 result rows."""
        for operand, present, name_ in (
            (w1_scales, form.has_scales, "w1_scales"),
            (w2_scales, form.has_scales, "w2_scales"),
            (w1_bias, has_w1_bias, "w1_bias"),
            (w2_bias, has_w2_bias, "w2_bias"),
        ):
            if (operand is not None) != present:
                raise ValueError(
                    f"the kernel was built for {weight_format} weights with "
                    f"has_w1_bias={has_w1_bias} and "
                    f"has_w2_bias={has_w2_bias}, "
                    f"and {name_} is "
                    f"{'present' if operand is not None else 'absent'}"
                )
        if pallas_transport != (transport_sems is not None):
            raise ValueError(
                "the pallas transport takes the start call's "
                "semaphore pair, and only it does"
            )
        if (shard_index is not None) != shard_operand:
            raise ValueError(
                "shard_index goes with a build with shard_operand and only then"
            )
        if token_of_row.shape != (table_rows,):
            raise ValueError(
                f"the token table is {token_of_row.shape}; the kernel reads "
                f"it in whole {LANES}-row blocks and was built for "
                f"({table_rows},): its {routed_rows} routed rows rounded up"
            )
        scales = tuple(
            s.astype(jnp.float32) for s in (w1_scales, w2_scales) if s is not None
        )
        biases = tuple(
            b.astype(jnp.float32) for b in (w1_bias, w2_bias) if b is not None
        )
        arrivals, arrival_scales, _, _ = pallas_call(arrival_rows)(
            *tables,
            expert_rows.astype(jnp.int32),
            expert_base.astype(jnp.int32),
            visit_order.astype(jnp.int32),
            counts.astype(jnp.int32),
            *(() if shard_index is None else (shard_index.astype(jnp.int32),)),
            token_of_row.astype(jnp.int32).reshape(-1, 1, LANES),
            tokens.reshape(-1, lane_blocks, LANES),
            token_scales.reshape(
                scale_planes, layout.scale_table_rows(routed_rows), LANES
            ),
            w1,
            w2,
            *scales,
            *biases,
            *(transport_sems if transport_sems is not None else ()),
        )
        return arrivals, arrival_scales.reshape(arrival_rows)

    return run


# Pallas keys its kernel cache on the body object, so the build is memoized.
# The miss path is serialized so two threads never build one key twice.
_BUILD_CACHE = {}
_BUILD_CACHE_LOCK = threading.Lock()


def ffn_kernel(**kwargs):
    """build_ffn_kernel, memoized on its arguments. The cache is unbounded:
    one program per (token bucket, format, bias set, activation, config)."""
    key = tuple(sorted(kwargs.items()))
    run = _BUILD_CACHE.get(key)
    if run is None:
        with _BUILD_CACHE_LOCK:
            run = _BUILD_CACHE.get(key)
            if run is None:
                run = build_ffn_kernel(**kwargs)
                _BUILD_CACHE[key] = run
                logger.debug(
                    "fused EP MoE: built program %d for %s",
                    len(_BUILD_CACHE),
                    ", ".join(f"{k}={v}" for k, v in key),
                )
    return run
