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
"""Top-level Pallas kernel entrypoint for Fused EP MoE v3 expert FFN execution."""

import dataclasses
import logging
import threading
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import device, ffn, layout, transport, vmem
from . import ffn_body as body
from . import ffn_expert_whole as whole
from .config import (
    ACTIVATION_COLUMNS_PER_PASS_DEFAULT,
    RESULT_SLOTS_DEFAULT,
    ROW_FETCH_AHEAD_DEFAULT,
    WEIGHT_DMA_PRIORITY_DEFAULT,
    WEIGHT_PREFETCH_DEFAULT,
    WEIGHT_SLOTS_DEFAULT,
)
from .formats import WeightFormat, weight_form
from .layout import LANES, ROW_BLOCK

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Tables:
    """The scalar-prefetch tables (SMEM), in operand order. Block units are
    ROW_BLOCK-row blocks. G is the shard's local expert count: row e of a
    [G, num_shards] table is local expert e, column d destination shard d."""

    run_start: object  # [G, num_shards] block e's run for d starts at
    run_blocks: object  # [G, num_shards] blocks in that run (padded)
    outgoing_offset: object  # [G, num_shards] outgoing block it fills
    push_blocks: object  # [G, num_shards] blocks the push to d carries
    push_destination: object  # [G, num_shards] d's block it lands on
    push_rows: object  # [G, num_shards] true rows of the run (rows)
    arrival_offset: object  # [G, num_shards] d's arrival row it lands on
    expert_rows: object  # [G] padded rows of each local expert
    expert_base: object  # [G] routed row each local expert's rows start at
    visit_order: object  # [G] local experts with rows, most rows first
    counts: object  # [COUNT_ROWS, num_shards] the five counts, per shard


@dataclasses.dataclass(frozen=True)
class Operands:
    """The HBM operands, in operand order."""

    token_of_row: object  # [routed rows / LANES, 1, LANES] i32 token each
    # routed row computes; read per tile into a scalar-memory window
    # to keep scalar-memory use bounded as the table grows
    tokens: object  # [tokens, lane blocks, LANES] fp8 rows
    token_scales: object  # [scale planes, scale rows, LANES] i32: f32
    # bits, one word per row per plane
    w1: object  # [G, hidden, cols] weights
    w2: object  # [G, inter, hidden]
    w1_scales: object  # [G, cols] f32, or None (bf16 weights)
    w2_scales: object  # [G, hidden] f32, or None
    w1_bias: object  # [G, cols] f32 or None
    w2_bias: object  # [G, hidden] f32 or None
    transport_send_sem: object = None  # pallas mode: the start call's
    transport_receive_sems: object = None  # two semaphores


@dataclasses.dataclass(frozen=True)
class Outputs:
    """The kernel's output refs: arrival and outgoing rows with scales. The
    rows are the result rows' dtype (fp8, or bf16 with result_rows="bf16",
    when the two scale outputs are allocated and never written)."""

    arrivals: object  # [arrival rows, result blocks, LANES]
    arrival_scales: (
        object  # [arrival rows / ROW_BLOCK, 1, ROW_BLOCK] f32, a word per row
    )
    outgoing: object  # [routed rows, result blocks, LANES]
    outgoing_scales: object  # [routed rows / ROW_BLOCK, 1, ROW_BLOCK] f32


@dataclasses.dataclass(frozen=True)
class Scratch:
    """The VMEM buffers, in vmem.scratch_arrays' order."""

    rows: object  # [result slots, tile rows, lane blocks, LANES] fp8
    w1: object  # [weight slots, hidden, cols]
    w2: object  # [weight slots, inter, hidden]
    w1_scales: object  # [G, cols] resident, or None
    w2_scales: object  # [G, hidden] resident, or None
    w1_bias: object  # [G, cols] or None
    w2_bias: object  # [G, hidden] or None
    row_scales: object  # [result slots, planes, window rows, LANES] i32
    result_rows: object  # [result slots, tile rows, result blocks, LANES]
    result_scales: object  # [result slots, tile blocks, 1, ROW_BLOCK] f32
    # Scalar-memory windows of the routed-row token table: one per result
    # slot, and the next expert's first tiles.
    token_window: object = None  # [row buffers, blocks, 1, LANES] i32, SMEM
    token_head: object = None  # [row_fetch_ahead, blocks, 1, LANES] i32


@dataclasses.dataclass(frozen=True)
class Semaphores:
    """DMA semaphores. Waits count bytes per semaphore, so every stream that
    is waited separately has its own."""

    rows: object  # [result slots] a tile's row stream
    w1: object  # [weight slots]
    w2: object  # [weight slots]
    copy: object  # the resident-table copies
    commit: object  # [result slots] a slot's commits
    push_send: object
    push_receive: object
    commit_scales: object  # [result slots]
    push_send_scales: object
    push_receive_scales: object
    local_hop: object  # the own-shard run's local copy
    local_hop_scales: object
    token_window: object  # [row buffers] the token windows
    token_head: object  # [row_fetch_ahead] the next expert's first


def _check_build(
    *,
    tile_rows,
    hidden,
    inter,
    routed_rows,
    weight_slots,
    weight_prefetch,
    result_slots,
    row_fetch_ahead,
    activation_block,
    token_block=None,
):
    """Raises ValueError for a build the kernel does not support, naming
    the operand."""
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
    vmem_fraction=device.VMEM_FRACTION,
    activation_columns_per_pass=ACTIVATION_COLUMNS_PER_PASS_DEFAULT,
    weight_dma_priority=WEIGHT_DMA_PRIORITY_DEFAULT,
    token_block=None,
    sorted_rows=False,
    shard_index_operand=False,
):
    """Builds the program for one layer shape and returns the function that
    runs it (see `run` below). pallas_transport: the token buffer is the
    transport start call's gathered_rows buffer, still landing, and its two
    semaphores are operands. activation_block: values per fp8 scale of a
    row of each matmul's input (0: one scale per row); the row scales
    arrive as hidden // activation_block planes of the scale table.
    shard_index_operand: the program takes the shard's own index as one
    more scalar-prefetch operand (run's `shard_index`) instead of reading
    the mesh axis, and addresses its peers in that form
    (layout.device_coords); the torch bridge's form.
    token_block: the token rows' own rounding block when it differs
    (None: activation_block; 0: one scale per row).
    result_rows: "fp8" (rows with one scale per row) or "bf16" (rows
    alone: the scale scratch and both scale outputs are still allocated,
    and nothing is written to them or transferred).
    elementwise_dtype: the element type of the vector math after each
    matmul (Config.elementwise_dtype). The weight slots hold whole
    experts."""
    form = weight_form(weight_format)
    if elementwise_dtype not in ("float32", "bfloat16"):
        raise ValueError(
            f"elementwise_dtype {elementwise_dtype!r} is float32 or bfloat16"
        )
    elementwise = jnp.float32 if elementwise_dtype == "float32" else jnp.bfloat16
    if result_rows not in ("fp8", "bf16"):
        raise ValueError(f"result_rows {result_rows!r} is fp8 or bf16")
    result_dtype = ffn.FP8 if result_rows == "fp8" else jnp.bfloat16
    has_result_scales = result_rows == "fp8"
    if pallas_transport:
        transport.check_width(num_shards)
    activations = ffn.NO_GATE_ACTIVATIONS if no_gate else ffn.ACTIVATIONS
    if activation not in activations:
        raise ValueError(f"activation {activation!r} is not one of {activations}")
    _check_build(
        tile_rows=tile_rows,
        hidden=hidden,
        inter=inter,
        routed_rows=routed_rows,
        weight_slots=weight_slots,
        weight_prefetch=weight_prefetch,
        result_slots=result_slots,
        row_fetch_ahead=row_fetch_ahead,
        activation_block=activation_block,
        token_block=token_block,
    )
    token_block = activation_block if token_block is None else token_block
    device.check_generation()
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
        has_w1_bias=has_w1_bias,
        has_w2_bias=has_w2_bias,
        no_gate=no_gate,
        activation_block=activation_block,
        result_dtype=result_dtype,
        token_block=token_block,
    )
    needed = vmem.estimate_bytes(
        experts_per_shard,
        tile_rows,
        hidden,
        inter,
        weight_slots=weight_slots,
        result_slots=result_slots,
        weight_format=weight_format,
        has_w1_bias=has_w1_bias,
        has_w2_bias=has_w2_bias,
        no_gate=no_gate,
        activation_block=activation_block,
        result_dtype=result_dtype,
        token_block=token_block,
    )
    limit = device.vmem_limit(vmem_fraction)
    if needed > limit:
        raise ValueError(
            f"the kernel's VMEM buffers need "
            f"{needed / layout.MIB:.1f} MiB, over the "
            f"{limit / layout.MIB:.1f} MiB budget, for {experts_per_shard} local "
            f"experts of {hidden}x{inter} at tile {tile_rows}"
        )
    # One more scalar-prefetch table in the operand form: the shard's own
    # index, read instead of the mesh axis.
    n_tables = len(dataclasses.fields(Tables)) + int(shard_index_operand)
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
        w1_scales=form.has_scales,
        w2_scales=form.has_scales,
        w1_bias=has_w1_bias,
        w2_bias=has_w2_bias,
        row_scales=True,
        result_rows=True,
        result_scales=True,
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
        shard_ref = next(it) if shard_index_operand else None
        operands = Operands(**take_fields(it, operand_present))
        outputs = Outputs(*(next(it) for _ in dataclasses.fields(Outputs)))
        scratch = Scratch(**take_fields(it, scratch_present))
        sems = Semaphores(*(next(it) for _ in dataclasses.fields(Semaphores)))
        return tables, operands, outputs, scratch, sems, shard_ref

    build = body.Build(
        tile_rows=tile_rows,
        hidden=hidden,
        num_shards=num_shards,
        mesh_axes=tuple(mesh_axes),
        shard_index_operand=shard_index_operand,
        experts_per_shard=experts_per_shard,
        form=form,
        has_w1_bias=has_w1_bias,
        has_w2_bias=has_w2_bias,
        no_gate=no_gate,
        activation=activation,
        weight_slots=weight_slots,
        weight_prefetch=weight_prefetch,
        result_slots=result_slots,
        row_fetch_ahead=row_fetch_ahead,
        activation_block=activation_block,
        token_block=token_block,
        result_dtype=result_dtype,
        elementwise=elementwise,
        has_result_scales=has_result_scales,
        activation_columns_per_pass=activation_columns_per_pass,
        weight_dma_priority=weight_dma_priority,
        sorted_rows=sorted_rows,
        lane_blocks=lane_blocks,
        result_blocks=result_blocks,
        tile_blocks=tile_blocks,
        scale_planes=scale_planes,
        window_rows=window_rows,
    )

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
        whole.first_refills(ctx)
        transport.all_shards_barrier(
            num_shards, build.mesh_axes, shard_index_operand=shard_index_operand
        )
        # ---- the program ----
        if pallas_transport:
            transport.finish(
                transport.Exchange(
                    operands.tokens,
                    operands.transport_send_sem,
                    operands.transport_receive_sems,
                    ctx.me,
                    cast(Any, operands.tokens).shape[0] // num_shards,
                    num_shards,
                    mesh_axes=build.mesh_axes,
                    shard_index_operand=shard_index_operand,
                )
            )
        whole.issue_first(ctx)
        lax.fori_loop(
            0,
            ctx.n_visit,
            lambda visit_i, carry: whole.visit(ctx, visit_i, carry),
            whole.initial_carry(ctx),
        )
        body.drain_transport(ctx)
        transport.all_shards_barrier(
            num_shards, build.mesh_axes, shard_index_operand=shard_index_operand
        )

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    sem = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)
    scratch = [pltpu.VMEM(shape, dtype) for _, shape, dtype in scratch_shapes]
    # The routed-row token table's windows in scalar memory: one tile's rows
    # per result slot, and the next expert's first tiles.
    row_buffers, window_len = result_slots, tile_rows
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
    columns_tag = (
        f"_p{activation_columns_per_pass}"
        if activation_columns_per_pass != ACTIVATION_COLUMNS_PER_PASS_DEFAULT
        else ""
    )
    priority_tag = (
        f"_q{weight_dma_priority}"
        if weight_dma_priority != WEIGHT_DMA_PRIORITY_DEFAULT
        else ""
    )
    name = (
        "fused_ep_moe"
        f"{'' if form.name == WeightFormat.FP8 else '_' + form.name}"
        f"{'_w1bias' if has_w1_bias else ''}{'_w2bias' if has_w2_bias else ''}"
        f"{'_pallas' if pallas_transport else ''}"
        f"{'_a' + str(activation_block) if activation_block else ''}"
        f"{'_k' + str(token_block) if token_block != activation_block else ''}"
        f"{'_bf16out' if not has_result_scales else ''}"
        f"{'_e16' if elementwise_dtype == 'bfloat16' else ''}"
        f"{columns_tag}{priority_tag}"
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
        if (shard_index is not None) != shard_index_operand:
            raise ValueError(
                "shard_index goes with a build made with shard_index_operand=True, "
                "and only with it"
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
            *(cast("tuple", transport_sems) if pallas_transport else ()),
        )
        return arrivals, arrival_scales.reshape(arrival_rows)

    return run


# Pallas keys its kernel cache on the body object, so the build is memoized.
# The miss path is serialized so two threads never build one key twice.
_BUILD_CACHE = {}
_BUILD_CACHE_LOCK = threading.Lock()


def ffn_kernel(*, shard_index_operand=False, **kwargs):
    """build_ffn_kernel, memoized on its arguments and on
    shard_index_operand, which the build reads (the operand mode adds
    one scalar-prefetch table and changes the unpacking), so a build made in
    one mode is never handed to the other. The cache is unbounded: one
    program per (token bucket, format, bias set, activation, config, mode)."""
    key = tuple(sorted(kwargs.items())) + (
        ("shard_index_operand", shard_index_operand),
    )
    run = _BUILD_CACHE.get(key)
    if run is None:
        with _BUILD_CACHE_LOCK:
            run = _BUILD_CACHE.get(key)
            if run is None:
                run = build_ffn_kernel(
                    **kwargs, shard_index_operand=shard_index_operand
                )
                _BUILD_CACHE[key] = run
                logger.debug(
                    "fused EP MoE: built program %d for %s",
                    len(_BUILD_CACHE),
                    ", ".join(f"{k}={v}" for k, v in key),
                )
    return run
