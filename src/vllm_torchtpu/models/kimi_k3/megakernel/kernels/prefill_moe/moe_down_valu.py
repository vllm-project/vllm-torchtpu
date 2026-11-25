# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

#
# Portions of this file are modified copies of code Copyright 2026 Google LLC,
# licensed under the Apache License, Version 2.0, from
# vllm_torchtpu/kernels/megablox/gmm_v2.py.
# Source provenance: inferact/k3-megakernel@30c9ba966b
#   src/vllm_torchtpu/megakernel/kimi_k3/kernels/prefill_moe/moe_down_valu.py
# Source provenance: vllm_torchtpu/kernels/megablox/gmm_v2.py@644426d851 (modified copy)
"""MXFP4 down, weighted token accumulation, and pipelined reduce-scatter.

While the MXU produces tile n, VALU accumulates tile n-1 in packed token
layout. Completed column-block subtotals feed the ring directly from VMEM;
the previous block's ring advances between tiles of the next down block.
Products and local subtotals round to BF16; token weighting uses FP32 and
the ring uses ``ring_dtype``.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.kimi_k3.megakernel.common.constants import (
    AXIS_SP,
    HIDDEN_SIZE,
    OVERFLOW_WINDOW_TOKENS,
    PREFILL_WINDOWS,
    STORE_TILE,
    TP_PEERS,
    TP_SIZE,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import prefill_megakernel as k

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from jax._src.pallas import core as pl_core

    from vllm_torchtpu.models.kimi_k3.megakernel.common.protocols import (
        AsyncCopyProtocol,
        PallasRefProtocol,
    )

RING_FAMILIES = len(k.CYCLES)
RING_BUFFERS = 2
VMEM_CAPACITY_FRACTION = 0.98
DEFAULT_DOWN_COLUMNS = HIDDEN_SIZE // 4
DEFAULT_DOWN_UNROLL = 8
_VALID_TILES = (k.STRIPE, STORE_TILE, 2 * STORE_TILE)
_VALID_TOKEN_WINDOWS = PREFILL_WINDOWS[-2:]
_DEFAULT_ROUTE_CACHE_ROWS = OVERFLOW_WINDOW_TOKENS
_REGION_TUPLE_WITH_COLS = 4
_RESIDENT_WEIGHT_NDIM = 4
_COLLECTIVE_ID = 77


@dataclasses.dataclass(frozen=True, slots=True)
class _RingStreamDescriptor:
    """Per-stream ring peer coordinates and source/destination slices."""

    target: tuple[jax.Array]
    source: tuple[jax.Array]
    rows: pl.Slice
    cols: pl.Slice
    source_rows: pl.Slice


@dataclasses.dataclass(frozen=True, slots=True)
class _RingStreamRefs:
    """Scratch and semaphore references shared across ring streams."""

    x_ref: PallasRefProtocol
    o_ref: PallasRefProtocol
    send_sems: PallasRefProtocol
    recv_sems: PallasRefProtocol
    credit_sems: PallasRefProtocol
    payload: PallasRefProtocol
    received: PallasRefProtocol
    buffers: int


@dataclasses.dataclass(frozen=True, slots=True)
class _DownValuSpec:
    """Configuration options for ``down_valu``."""

    tokens: int
    columns: int
    tile: int
    unroll: int
    layer: jax.Array | None
    stream_metadata: bool
    metadata_cache_rows: int
    metadata_row_capacity: int
    ring_dtype: jnp.dtype


@dataclasses.dataclass(frozen=True, slots=True)
class _DownKernelSpec:
    """Static tiling and streaming parameters for the ``down_valu`` kernel."""

    cfg: k.GmmConfigs
    regions: Sequence[tuple[int, ...]]
    tokens: int
    tn: int
    width: int
    sub: int
    unroll: int
    route_chunk: int
    route_span: int
    stream_metadata: bool
    metadata_cache_rows: int
    max_gm: int
    x_dtype: jnp.dtype
    ring_dtype: jnp.dtype


@dataclasses.dataclass(frozen=True, slots=True)
class _DownKernelContext:
    """Bound kernel references and static parameters for ``down_valu``."""

    spec: _DownKernelSpec
    metadata: k.MetadataRef
    sums: PallasRefProtocol
    packed: PallasRefProtocol
    cast_buf: PallasRefProtocol
    subtotal_vmem: PallasRefProtocol
    dest_tiles: PallasRefProtocol
    prob_tiles: PallasRefProtocol
    destinations_ref: PallasRefProtocol
    probabilities_ref: PallasRefProtocol
    route_sem: PallasRefProtocol
    route_tags: PallasRefProtocol
    state: PallasRefProtocol


def _stream_ring_descriptor(
    tables_ref: PallasRefProtocol,
    o_ref: PallasRefProtocol,
    region: tuple[int, ...],
    stream_step: tuple[int, jax.Array | int],
    position: jax.Array,
) -> _RingStreamDescriptor:
    """Build peer coordinates and slice descriptors for one ring stream."""
    stream, step = stream_step
    begin, size = region[:2]
    col_begin, width = (
        region[2:] if len(region) == _REGION_TUPLE_WITH_COLS else (0, o_ref.shape[1])
    )
    cols = pl.ds(col_begin, width)
    family = stream // 2 % RING_FAMILIES
    direction = -1 if stream % 2 == 0 else 1
    family_idx = jnp.int32(family)
    target = (tables_ref[family_idx, 1, (position + direction) % TP_SIZE],)
    source = (tables_ref[family_idx, 1, (position - direction) % TP_SIZE],)
    owner_mesh = tables_ref[
        family_idx, 1, (position - direction * (step + 1)) % TP_SIZE
    ]
    owner = tables_ref[family_idx, 2, owner_mesh]
    rows = pl.ds(begin, size)
    source_rows = pl.ds(owner * o_ref.shape[0] + begin, size)
    return _RingStreamDescriptor(
        target=target,
        source=source,
        rows=rows,
        cols=cols,
        source_rows=source_rows,
    )


def _ring_stream_hop(
    stream: int,
    descriptor: _RingStreamDescriptor,
    step: jax.Array | int,
    refs: _RingStreamRefs,
) -> None:
    """Execute one stream's wait, accumulate, and forward transfers for ``step``."""
    rows = descriptor.rows
    cols = descriptor.cols
    source_rows = descriptor.source_rows
    buffers = refs.buffers

    def local_value() -> jax.Array:
        """Read this rank's local subtotal slice for the current hop."""
        return (
            refs.x_ref.reshape(refs.x_ref.shape[0], refs.o_ref.shape[1])
            .at[source_rows, cols][...]
            .astype(refs.payload.dtype)
        )

    def remote(index: jax.Array | int) -> AsyncCopyProtocol:
        """Construct the remote peer copy descriptor for ``index``."""
        return cast(
            "AsyncCopyProtocol",
            pltpu.make_async_remote_copy(
                refs.payload.at[rows, cols],
                refs.received.at[index % buffers, rows, cols],
                refs.send_sems.at[stream],
                refs.recv_sems.at[index % buffers, stream],
                device_id=descriptor.target,
                device_id_type=pl.DeviceIdType.MESH,
            ),
        )

    @pl.when(step > 0)
    def wait() -> None:
        """Wait for the previous hop's incoming remote transfer to complete."""
        remote(step - 1).wait()

    @pl.when(step == 0)
    def seed() -> None:
        """Initialize the ring payload buffer with hop 0's local subtotal."""
        refs.payload.at[rows, cols][...] = local_value()

    @pl.when(step > 0)
    def add() -> None:
        """Accumulate the received peer shard and return flow-control credit."""
        refs.payload.at[rows, cols][...] = (
            local_value() + refs.received.at[(step - 1) % buffers, rows, cols][...]
        )
        pl.semaphore_signal(
            refs.credit_sems.at[stream],
            1,
            device_id=descriptor.source,
            device_id_type=pl.DeviceIdType.MESH,
        )

    @pl.when(step < TP_PEERS)
    def forward() -> None:
        """Acquire flow-control credit and start the outgoing remote transfer."""

        @pl.when(step >= buffers)
        def acquire() -> None:
            """Wait for downstream buffer credit before reusing a receive slot."""
            pl.semaphore_wait(refs.credit_sems.at[stream], 1)

        remote(step).start()

    @pl.when(step == TP_PEERS)
    def finish() -> None:
        """Drain all outstanding flow-control credits after the final hop."""
        pl.semaphore_wait(refs.credit_sems.at[stream], buffers)


def ring_step(
    tables_ref: PallasRefProtocol,
    x_ref: PallasRefProtocol,
    o_ref: PallasRefProtocol,
    send_sems: PallasRefProtocol,
    recv_sems: PallasRefProtocol,
    credit_sems: PallasRefProtocol,
    store_sem: PallasRefProtocol,
    payload: PallasRefProtocol,
    received: PallasRefProtocol,
    *,
    streams: Sequence[tuple[int, ...]],
    buffers: int,
    step: jax.Array | int,
) -> None:
    """Advance one hop of the 32-rank multi-stream reduce-scatter ring."""
    rank = lax.axis_index(AXIS_SP)
    positions = [tables_ref[f, 0, rank] for f in range(RING_FAMILIES)]

    @pl.when(step == 0)
    def enter_ring() -> None:
        """Synchronize all ranks on the ring barrier before hop 0."""
        k.ring_enter(tables_ref, rank)

    descriptors = [
        _stream_ring_descriptor(
            tables_ref,
            o_ref,
            region,
            (stream, step),
            positions[stream // 2 % RING_FAMILIES],
        )
        for stream, region in enumerate(streams)
    ]
    refs = _RingStreamRefs(
        x_ref=x_ref,
        o_ref=o_ref,
        send_sems=send_sems,
        recv_sems=recv_sems,
        credit_sems=credit_sems,
        payload=payload,
        received=received,
        buffers=buffers,
    )
    for stream, descriptor in enumerate(descriptors):
        _ring_stream_hop(stream, descriptor, step, refs)

    @pl.when(step == TP_PEERS)
    def store_complete_shard() -> None:
        """Write the fully reduced shard to the output buffer."""
        output = pltpu.make_async_copy(payload, o_ref, store_sem)
        output.start()
        output.wait()


def _prepare_down_valu_cfg(
    x: jax.Array,
    rhs: jax.Array,
    scales: jax.Array,
    counts: jax.Array,
    spec: _DownValuSpec,
) -> tuple[k.GmmConfigs, int, int, int, jax.Array, jax.Array]:
    """Validate parameters and construct the grouped-matmul config for ``down_valu``."""
    if (
        spec.tile not in _VALID_TILES
        or spec.unroll <= 0
        or spec.tokens not in _VALID_TOKEN_WINDOWS
    ):
        msg = (
            f"Invalid down_valu tile={spec.tile}, "
            f"unroll={spec.unroll}, or tokens={spec.tokens}"
        )
        raise ValueError(msg)
    rows, width = (spec.metadata_row_capacity or x.shape[0], rhs.shape[-1])
    if rows > x.shape[0] or width % spec.columns != 0:
        msg = (
            f"Invalid rows={rows} for x.shape={x.shape} "
            f"or width={width} % {spec.columns}"
        )
        raise ValueError(msg)
    if spec.metadata_cache_rows and (
        not spec.stream_metadata or spec.metadata_cache_rows % STORE_TILE != 0
    ):
        msg = "metadata_cache_rows requires stream_metadata and 128-row alignment"
        raise ValueError(msg)
    limit = int(pltpu.get_tpu_info().vmem_capacity_bytes * VMEM_CAPACITY_FRACTION)
    offset = jnp.zeros((1,), jnp.int32)
    family = rhs.ndim == _RESIDENT_WEIGHT_NDIM
    if family != (spec.layer is not None):
        msg = "Resident weights require a layer index"
        raise ValueError(msg)
    resolved_layer = offset if spec.layer is None else spec.layer
    cfg = k.make_gmm_configs(
        jax.ShapeDtypeStruct((rows, x.shape[-1]), x.dtype),
        jax.ShapeDtypeStruct(rhs.shape[1:] if family else rhs.shape, rhs.dtype),
        jax.ShapeDtypeStruct(
            scales.shape[1:] if family else scales.shape, scales.dtype
        ),
        counts,
        offset,
        tile_info=k.calculate_tiling,
        vmem_limit_bytes=limit,
        out_dtype=None,
        acc_dtype=None,
        fuse_act=None,
        rhs_quant_dtype=jnp.float4_e2m1fn,
    )
    cfg = dataclasses.replace(
        cfg, tiles=dataclasses.replace(cfg.tiles, tile_m=spec.tile, tile_n=spec.columns)
    )
    max_gm = cfg.dims.size_group + pl.cdiv(rows, spec.tile) - 1
    return cfg, rows, limit, max_gm, offset, resolved_layer


def _build_down_valu_grid_spec(spec: _DownKernelSpec) -> pltpu.PrefetchScalarGridSpec:
    """Build the ``PrefetchScalarGridSpec`` for ``down_valu``."""
    tm, tn = (spec.cfg.tiles.tile_m, spec.cfg.tiles.tile_n)
    localrows = spec.tokens // TP_SIZE
    streams = len(spec.regions)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    in_prefix = (hbm, hbm) if spec.stream_metadata else ()
    scratch_shapes = cast(
        "pl_core.ScratchShapeTree",
        (
            None,  # partial: packed_inner_kernel stores rows directly
            pltpu.VMEM((tm, tn), jnp.float32),
            k.MetadataRef(
                gm_id_to_group_id=pltpu.SMEM((spec.max_gm,), jnp.int32),
                gm_id_to_m_offset=pltpu.SMEM((spec.max_gm + 1,), jnp.int32),
            ),
            pltpu.VMEM((2, tm, 2, tn // 2), spec.x_dtype),
            pltpu.VMEM((spec.tokens, 2, tn // 2), jnp.float32),
            pltpu.VMEM((STORE_TILE, tn), spec.x_dtype),
            pltpu.SemaphoreType.DMA((streams,)),
            pltpu.SemaphoreType.DMA((2, streams)),
            pltpu.SemaphoreType.REGULAR((streams,)),
            pltpu.SemaphoreType.DMA,
            pltpu.VMEM((localrows, tn), spec.ring_dtype),
            pltpu.VMEM((2, localrows, tn), spec.ring_dtype),
            pltpu.SMEM((1,), jnp.int32),
            pltpu.VMEM((spec.tokens, tn), spec.x_dtype),
            (
                pltpu.SMEM((2, spec.route_span), jnp.int32)
                if spec.stream_metadata
                else None
            ),
            (
                pltpu.SMEM((2, spec.route_span), jnp.float32)
                if spec.stream_metadata
                else None
            ),
            pltpu.SemaphoreType.DMA if spec.stream_metadata else None,
            pltpu.SMEM((2,), jnp.int32) if spec.metadata_cache_rows else None,
        ),
    )
    return pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=4 if spec.stream_metadata else 6,
        in_specs=(*in_prefix, hbm, k.WeightsRef(weight=hbm, scale=hbm)),
        out_specs=hbm,
        scratch_shapes=scratch_shapes,
    )


def _route_entry(
    ctx: _DownKernelContext,
    tile_g: jax.Array,
    row_idx: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Read token destination and routing probability for row ``row_idx``."""
    if ctx.spec.stream_metadata:
        offsets = cast("PallasRefProtocol", ctx.metadata.gm_id_to_m_offset)
        chunk = offsets[tile_g] // ctx.spec.route_chunk
        begin = chunk * ctx.spec.route_chunk
        return (
            ctx.dest_tiles[chunk % 2, row_idx - begin],
            ctx.prob_tiles[chunk % 2, row_idx - begin],
        )
    return (ctx.destinations_ref[row_idx], ctx.probabilities_ref[row_idx])


def _consume_down_tile(
    ctx: _DownKernelContext,
    tile_g: jax.Array,
    source: PallasRefProtocol,
    first: int = 0,
) -> None:
    """Accumulate remaining rows of ``source`` for group tile ``tile_g``."""
    offsets = cast("PallasRefProtocol", ctx.metadata.gm_id_to_m_offset)
    start = offsets[tile_g]
    end = offsets[tile_g + 1]
    begin = start // ctx.spec.sub * ctx.spec.sub
    start = jnp.minimum(start + first, end)
    unroll = ctx.spec.unroll

    def update(row_idx: jax.Array) -> None:
        """Weight and accumulate a single row into its destination token."""
        dest, weight = _route_entry(ctx, tile_g, row_idx)
        value = source.at[pl.ds(row_idx - begin, 1)][...].astype(jnp.float32)
        target = ctx.sums.at[pl.ds(dest, 1)]
        target[...] = target[...] + value * weight

    full = (end - start) // unroll

    @pl.loop(0, full)
    def chunk(b: jax.Array) -> None:
        """Accumulate one unrolled batch of rows."""
        for u in range(unroll):
            update(start + b * unroll + u)

    @pl.loop(start + full * unroll, end)
    def tail(row_idx: jax.Array) -> None:
        """Accumulate any remainder rows after the unrolled loop."""
        update(row_idx)


def _load_streamed_routes(ctx: _DownKernelContext, g: jax.Array) -> None:
    """Prefetch streamed routing metadata when crossing an 8K chunk boundary."""
    if not ctx.spec.stream_metadata:
        return
    route_chunk = ctx.spec.route_chunk
    offsets = cast("PallasRefProtocol", ctx.metadata.gm_id_to_m_offset)
    chunk = offsets[g] // route_chunk
    previous_chunk = offsets[jnp.maximum(g - 1, 0)] // route_chunk
    missing = (
        ctx.route_tags[chunk % 2] != chunk
        if ctx.spec.metadata_cache_rows
        else (g == 0) | (chunk != previous_chunk)
    )

    @pl.when(missing)
    def load_routes() -> None:
        """Copy the current metadata chunk's destinations and weights."""
        begin = chunk * route_chunk
        copies = (
            pltpu.make_async_copy(
                ctx.destinations_ref.at[pl.ds(begin, ctx.spec.route_span)],
                ctx.dest_tiles.at[chunk % 2],
                ctx.route_sem,
            ),
            pltpu.make_async_copy(
                ctx.probabilities_ref.at[pl.ds(begin, ctx.spec.route_span)],
                ctx.prob_tiles.at[chunk % 2],
                ctx.route_sem,
            ),
        )
        for cp in copies:
            cp.start()
        for cp in copies:
            cp.wait()
        if ctx.spec.metadata_cache_rows:
            ctx.route_tags[chunk % 2] = chunk


def _emit_down_subtotals(ctx: _DownKernelContext) -> None:
    """Round completed FP32 token accumulators to BF16 in ``subtotal_vmem``."""

    @pl.loop(0, ctx.spec.tokens // STORE_TILE)
    def block(b: jax.Array) -> None:
        """Convert one 128-token slice of ``sums`` into BF16 ring input."""
        ctx.cast_buf[...] = (
            ctx.sums[pl.ds(b * STORE_TILE, STORE_TILE)]
            .reshape(STORE_TILE, ctx.spec.tn)
            .astype(jnp.bfloat16)
        )
        ctx.subtotal_vmem.at[pl.ds(b * STORE_TILE, STORE_TILE)][...] = ctx.cast_buf[...]


def _overlap_previous_tile(
    ctx: _DownKernelContext,
    g: jax.Array,
    previous_g: jax.Array,
    previous: PallasRefProtocol,
    panel: int,
) -> None:
    """Accumulate ``unroll`` rows of the previous tile after each column panel."""
    offsets = cast("PallasRefProtocol", ctx.metadata.gm_id_to_m_offset)
    start = offsets[previous_g]
    end = offsets[previous_g + 1]
    begin = start // ctx.spec.sub * ctx.spec.sub
    for row in range(ctx.spec.unroll):
        row_offset = panel * ctx.spec.unroll + row
        row_idx = jnp.minimum(start + row_offset, end - 1)
        dest, prob = _route_entry(ctx, previous_g, row_idx)
        weight = jnp.where((g > 0) & (start + row_offset < end), prob, 0)
        value = previous.at[pl.ds(row_idx - begin, 1)][...].astype(jnp.float32)
        target = ctx.sums.at[pl.ds(dest, 1)]
        target[...] = target[...] + value * weight


def _finish_down_column(
    ctx: _DownKernelContext,
    g: jax.Array,
    n: jax.Array,
    current: PallasRefProtocol,
    advance_fn: Callable[[jax.Array, jax.Array], None],
) -> None:
    """Drain the final group tile, emit subtotals, and finish the column ring."""
    _consume_down_tile(ctx, g, current)

    @pl.when(n > 0)
    def previous_ring() -> None:
        """Complete all remaining ring hops for column block ``n - 1``."""
        advance_fn(jnp.int32(TP_SIZE), n - 1)

    _emit_down_subtotals(ctx)
    ctx.state[0] = 0

    @pl.when(n == ctx.spec.width // ctx.spec.tn - 1)
    def last_ring() -> None:
        """Run all 32 ring hops for the final column block."""
        advance_fn(jnp.int32(TP_SIZE), n)


def _make_down_valu_inner(
    ctx: _DownKernelContext,
    active_ng: jax.Array,
    ng: jax.Array,
    advance_fn: Callable[[jax.Array, jax.Array], None],
) -> Callable[..., None]:
    """Build the ``emit_pipeline`` body closure for ``down_valu``."""
    panel_count = pl.cdiv(ctx.spec.tn, pltpu.get_tpu_info().mxu_column_size)

    def inner(
        x_tile: PallasRefProtocol,
        w_tile: k.RhsRef,
        _partial_tile: PallasRefProtocol | None,
        acc_tile: PallasRefProtocol,
        metadata_tile: k.MetadataRef,
    ) -> None:
        """Run one (column_block, group_tile) step of the down-projection pipeline."""
        g, n = (pl.program_id(1), pl.program_id(0))
        _load_streamed_routes(ctx, g)

        @pl.when(g == 0)
        def clear() -> None:
            """Zero the FP32 token accumulator at the start of each column block."""
            ctx.sums[...] = jnp.zeros(ctx.sums.shape, jnp.float32)

        @pl.when(n > 0)
        def progress() -> None:
            """Advance the previous column block's ring proportionally to tile ``g``."""
            advance_fn(jnp.minimum(TP_SIZE, (g + 1) * TP_SIZE // ng), n - 1)

        current = ctx.packed.at[g % 2]
        previous = ctx.packed.at[(g + 1) % 2]
        previous_g = jnp.maximum(g - 1, 0)

        # These callbacks issue VALU work between MXU panel issue/use.
        # Ping-pong output banks keep the previous tile live throughout.
        def overlap_work(panel: int) -> None:
            """Issue weighted row accumulation for ``panel`` of the previous tile."""
            _overlap_previous_tile(ctx, g, previous_g, previous, panel)

        @pl.when(g == 0)
        def initialize_previous() -> None:
            """Zero the previous ping-pong bank before group tile 0."""
            previous[...] = jnp.zeros(previous.shape, previous.dtype)

        @pl.when(active_ng > 0)
        def product() -> None:
            """Compute the current group tile's MXFP4 matmul while overlapping rows."""
            packed_inner_kernel(
                x_tile,
                w_tile,
                current,
                acc_tile,
                metadata_tile,
                cfgs=ctx.spec.cfg,
                overlap_work=overlap_work,
            )

        @pl.when(g > 0)
        def previous_tail() -> None:
            """Finish any remaining rows of the previous group tile."""
            _consume_down_tile(ctx, previous_g, previous, panel_count * ctx.spec.unroll)

        @pl.when(g == ng - 1)
        def finish() -> None:
            """Complete the final group tile and drain the ring."""
            _finish_down_column(ctx, g, n, current, advance_fn)

    return inner


def down_valu(
    x: jax.Array,
    rhs: jax.Array,
    scales: jax.Array,
    counts: jax.Array,
    destinations: jax.Array,
    probabilities: jax.Array,
    tables: jax.typing.ArrayLike,
    *,
    tokens: int,
    columns: int = DEFAULT_DOWN_COLUMNS,
    tile: int = STORE_TILE,
    unroll: int = DEFAULT_DOWN_UNROLL,
    layer: jax.Array | None = None,
    stream_metadata: bool = False,
    metadata_cache_rows: int = 0,
    metadata_row_capacity: int = 0,
    ring_dtype: jax.typing.DTypeLike = jnp.float32,
) -> jax.Array:
    """Combine packed BF16 expert products with FP32 weights and ring-reduce."""
    resolved_ring_dtype = jnp.dtype(ring_dtype)
    valu_spec = _DownValuSpec(
        tokens=tokens,
        columns=columns,
        tile=tile,
        unroll=unroll,
        layer=layer,
        stream_metadata=stream_metadata,
        metadata_cache_rows=metadata_cache_rows,
        metadata_row_capacity=metadata_row_capacity,
        ring_dtype=resolved_ring_dtype,
    )
    cfg, rows, limit, max_gm, offset, resolved_layer = _prepare_down_valu_cfg(
        x, rhs, scales, counts, valu_spec
    )
    width = rhs.shape[-1]
    tm, tn, localrows = (tile, columns, tokens // TP_SIZE)
    family = rhs.ndim == _RESIDENT_WEIGHT_NDIM
    sub = cfg.dims.size_lhs_sublane
    regions = k.regions(localrows, tn)
    # Cache adjacent 8K metadata chunks across product and column tiles.
    route_chunk = min(rows, metadata_cache_rows or _DEFAULT_ROUTE_CACHE_ROWS)
    route_span = route_chunk + tm
    kernel_spec = _DownKernelSpec(
        cfg=cfg,
        regions=regions,
        tokens=tokens,
        tn=tn,
        width=width,
        sub=sub,
        unroll=unroll,
        route_chunk=route_chunk,
        route_span=route_span,
        stream_metadata=stream_metadata,
        metadata_cache_rows=metadata_cache_rows,
        max_gm=max_gm,
        x_dtype=x.dtype,
        ring_dtype=resolved_ring_dtype,
    )

    def kernel(
        counts_ref: PallasRefProtocol,
        offset_ref: PallasRefProtocol,
        layer_ref: PallasRefProtocol,
        tables_ref: PallasRefProtocol,
        destinations_ref: PallasRefProtocol,
        probabilities_ref: PallasRefProtocol,
        x_ref: PallasRefProtocol,
        w: k.WeightsRef,
        out: PallasRefProtocol,
        partial: PallasRefProtocol | None,
        acc: PallasRefProtocol,
        metadata: k.MetadataRef,
        packed: PallasRefProtocol,
        sums: PallasRefProtocol,
        cast_buf: PallasRefProtocol,
        sends: PallasRefProtocol,
        receives: PallasRefProtocol,
        credit_sems: PallasRefProtocol,
        stores: PallasRefProtocol,
        payload: PallasRefProtocol,
        rx: PallasRefProtocol,
        state: PallasRefProtocol,
        subtotal_vmem: PallasRefProtocol,
        dest_tiles: PallasRefProtocol,
        prob_tiles: PallasRefProtocol,
        route_sem: PallasRefProtocol,
        route_tags: PallasRefProtocol,
    ) -> None:
        """Run the fused down-projection, row accumulation, and ring pipeline."""
        weight_ref = cast("PallasRefProtocol", w.weight)
        scale_ref = cast("PallasRefProtocol", w.scale)
        if family:
            w = dataclasses.replace(
                w,
                weight=weight_ref.at[layer_ref[0]],
                scale=scale_ref.at[layer_ref[0]],
            )
            weight_ref = cast("PallasRefProtocol", w.weight)
        w = dataclasses.replace(w, weight=weight_ref.bitcast(jnp.float4_e2m1fn))
        active_ng = k.fill_metadata(
            counts_ref,
            offset_ref,
            metadata,
            cfgs=cfg,
        )

        @pl.when(active_ng == 0)
        def empty_metadata() -> None:
            """Zero the first metadata slot when a rank has no active expert tiles."""
            groups = cast("PallasRefProtocol", metadata.gm_id_to_group_id)
            offsets = cast("PallasRefProtocol", metadata.gm_id_to_m_offset)
            groups[0] = 0
            offsets[0] = 0
            offsets[1] = 0

        # Empty ranks retain the same collective call sites as active ranks.
        # A separate empty-rank ring branch can allocate different semaphores.
        ng = jnp.maximum(active_ng, 1)
        (xs, ws), _ = k.generate_block_specs(metadata, cfg)
        state[0] = 0
        if metadata_cache_rows:
            route_tags[0] = -1
            route_tags[1] = -1

        def advance(stop: jax.Array, block: jax.Array) -> None:
            """Advance the reduce-scatter ring up to ``stop`` for ``block``."""

            @pl.loop(state[0], stop)
            def hop(step: jax.Array) -> None:
                """Execute one ring hop for the completed column block."""
                ring_step(
                    tables_ref,
                    subtotal_vmem,
                    out.at[:, pl.ds(block * tn, tn)],
                    sends,
                    receives,
                    credit_sems,
                    stores,
                    payload,
                    rx,
                    streams=regions,
                    buffers=RING_BUFFERS,
                    step=step,
                )

            state[0] = stop

        ctx = _DownKernelContext(
            spec=kernel_spec,
            metadata=metadata,
            sums=sums,
            packed=packed,
            cast_buf=cast_buf,
            subtotal_vmem=subtotal_vmem,
            dest_tiles=dest_tiles,
            prob_tiles=prob_tiles,
            destinations_ref=destinations_ref,
            probabilities_ref=probabilities_ref,
            route_sem=route_sem,
            route_tags=route_tags,
            state=state,
        )
        inner = _make_down_valu_inner(ctx, active_ng, ng, advance)
        pltpu.emit_pipeline(
            inner, grid=(width // tn, ng, 1), in_specs=(xs, ws), out_specs=()
        )(
            x_ref.reshape(-1, sub, x_ref.shape[-1]),
            w,
            scratches=(partial, acc, metadata),
        )

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((localrows, width), resolved_ring_dtype),
        grid_spec=_build_down_valu_grid_spec(kernel_spec),
        compiler_params=pltpu.CompilerParams(
            collective_id=_COLLECTIVE_ID, vmem_limit_bytes=limit
        ),
        name=f"down_valu_m{tm}_n{tn}_u{unroll}_mc{metadata_cache_rows}",
    )
    return call(
        counts,
        offset,
        resolved_layer,
        jnp.asarray(tables, jnp.int32),
        jnp.pad(destinations, (0, route_span)) if stream_metadata else destinations,
        jnp.pad(probabilities, (0, route_span)) if stream_metadata else probabilities,
        x,
        k.WeightsRef(weight=rhs, scale=scales),
    )


def _prepare_tiled_rhs(
    tiled_rhs_ref: k.RhsRef,
    cfgs: k.GmmConfigs,
) -> tuple[jax.Array, int, int]:
    """Load and optionally pre-dequantize the RHS weight tile."""
    tiled_rhs = tiled_rhs_ref.get_weight()
    rhs_tile_n = tiled_rhs.shape[1]
    rhs_qbs = cfgs.rhs_cfgs.quant_block_size
    if rhs_qbs is None:
        msg = "Packed inner kernel requires a quantized block size"
        raise ValueError(msg)
    if cfgs.rhs_cfgs.should_dequantize_before_matmul:
        tiled_rhs_scale = tiled_rhs_ref.get_scale(replicate_size=rhs_qbs).astype(
            cfgs.lhs_cfgs.dtype
        )
        num_blocks = cfgs.num_quant_blocks_per_tile_k
        tiled_rhs_dequant = tiled_rhs.astype(cfgs.lhs_cfgs.dtype).reshape(
            num_blocks, rhs_qbs, rhs_tile_n
        )
        tiled_rhs_dequant = tiled_rhs_dequant * tiled_rhs_scale
        tiled_rhs = tiled_rhs_dequant.reshape(cfgs.tiles.tile_k, rhs_tile_n)
        rhs_qbs = cfgs.tiles.tile_k
    return tiled_rhs, rhs_tile_n, rhs_qbs


def packed_inner_kernel(
    tiled_lhs_ref: jax.Array | PallasRefProtocol,
    tiled_rhs_ref: k.RhsRef,
    tiled_out_ref: PallasRefProtocol,
    acc_ref: jax.Array | PallasRefProtocol,
    metadata_ref: k.MetadataRef,
    *,
    cfgs: k.GmmConfigs,
    overlap_work: Callable[[int], None],
) -> None:
    """Compute a grouped-matmul tile using a direct packed final store.

    Variant of ``prefill_megakernel.inner_kernel`` with a direct packed final store.
    Arithmetic and row bucketing follow ``k.inner_kernel``, over the complete
    reduction dimension in one tile. Each valid expert row is consumed
    immediately, so no adjacent-group sublane carry is needed. Output is
    ``[tile_m, 2, tile_n // 2]``; ``partial_out_ref`` is deliberately unused and
    ``acc_ref`` only sets the accumulation dtype. Products round to BF16 before
    the caller's FP32 routing weighting. ``overlap_work(panel)`` runs after
    each MXU column panel is issued.
    """
    if cfgs.dims.size_k != cfgs.tiles.tile_k or cfgs.fuse_act is not None:
        msg = "packed_inner_kernel requires full-K tiling without fused activation"
        raise ValueError(msg)
    gm_id = pl.program_id(1)
    offsets = cast("PallasRefProtocol", metadata_ref.gm_id_to_m_offset)
    m_start = offsets[gm_id]
    m_end = offsets[gm_id + 1]
    m_offset = m_start - m_start % cfgs.dims.size_lhs_sublane
    m_start_local = m_start - m_offset
    m_end_local = m_end - m_offset

    def _matmul(bucket_m: int) -> None:
        """Compute one row-bucketed matmul and write packed BF16 rows."""
        mxu_size = pltpu.get_tpu_info().mxu_column_size
        tiled_lhs = tiled_lhs_ref.reshape(-1, cfgs.tiles.tile_k)[:bucket_m]
        tiled_rhs, rhs_tile_n, rhs_qbs = _prepare_tiled_rhs(tiled_rhs_ref, cfgs)
        acc_list = []
        for start_n in range(0, rhs_tile_n, mxu_size):
            end_n = min(rhs_tile_n, start_n + mxu_size)
            col_size = end_n - start_n
            acc_n = jnp.zeros((bucket_m, col_size), dtype=acc_ref.dtype)
            for start_k in range(0, cfgs.tiles.tile_k, rhs_qbs):
                end_k = min(cfgs.tiles.tile_k, start_k + rhs_qbs)
                block_acc = jnp.matmul(
                    tiled_lhs[:, start_k:end_k],
                    tiled_rhs[start_k:end_k, start_n:end_n],
                    preferred_element_type=jnp.float32,
                ).astype(acc_ref.dtype)
                if end_k == cfgs.tiles.tile_k:
                    overlap_work(start_n // mxu_size)
                if cfgs.rhs_cfgs.should_dequantize_after_matmul:
                    b_id = start_k // rhs_qbs
                    rhs_scale_replicated = tiled_rhs_ref.get_scale(
                        replicate_size=bucket_m
                    )[b_id, :, start_n : start_n + col_size]
                    block_acc *= rhs_scale_replicated.astype(acc_ref.dtype)
                acc_n += block_acc
            acc_list.append(acc_n)
        acc = jnp.concatenate(acc_list, axis=1)
        acc_m = acc.shape[0]
        iota = lax.broadcasted_iota(jnp.int32, acc.shape, 0)
        mask = jnp.logical_and(m_start_local <= iota, iota < m_end_local)
        acc_masked = jnp.where(mask, acc, 0)
        tiled_out_ref[:acc_m] = acc_masked.astype(tiled_out_ref.dtype).reshape(
            acc_m, 2, cfgs.tiles.tile_n // 2
        )

    def run_matmul_step(bucket_idx: int) -> None:
        """Dispatch ``_matmul`` for bucket ``bucket_idx`` under a named JAX scope."""
        bucket_m = cfgs.tiles.bucket_base * (bucket_idx + 1)
        with jax.named_scope(f"bm{bucket_m}_fragment"):
            _matmul(bucket_m=bucket_m)

    branches = [
        functools.partial(run_matmul_step, bucket_idx=bucket_idx)
        for bucket_idx in range(cfgs.tiles.tile_m // cfgs.tiles.bucket_base)
    ]
    bucket_idx = m_end_local // cfgs.tiles.bucket_base
    lax.switch(bucket_idx, branches)


def down_valu_compact(
    x: jax.Array,
    rhs: jax.Array,
    scales: jax.Array,
    counts: jax.Array,
    destinations: jax.Array,
    probabilities: jax.Array,
    tables: jax.typing.ArrayLike,
    *,
    metadata_prefix_rows: int,
    tokens: int,
    slice_input: bool = True,
    global_route_max: jax.Array | int | None = None,
    columns: int = DEFAULT_DOWN_COLUMNS,
    tile: int = STORE_TILE,
    unroll: int = DEFAULT_DOWN_UNROLL,
    layer: jax.Array | None = None,
    ring_dtype: jax.typing.DTypeLike = jnp.float32,
) -> jax.Array:
    """Select between the uniform compact-prefix fast path and exact overflow path.

    Uniform compact-prefix fast path with an exact, uncapped overflow path.
    All ranks must select the same collective-containing branch. The caller
    supplies the maximum derived from globally replicated route counts to
    avoid an additional collective. The full activation buffer stays in HBM;
    only the scalar metadata prefix is bounded. Overflow never drops routes.
    """
    capacity = metadata_prefix_rows
    peak = (
        lax.pmax(counts.sum(), AXIS_SP)
        if global_route_max is None
        else global_route_max
    )
    fits = peak <= capacity
    return lax.cond(
        fits,
        lambda: down_valu(
            x[:capacity] if slice_input else x,
            rhs,
            scales,
            counts,
            destinations[:capacity],
            probabilities[:capacity],
            tables,
            tokens=tokens,
            columns=columns,
            tile=tile,
            unroll=unroll,
            layer=layer,
            stream_metadata=False,
            metadata_cache_rows=0,
            metadata_row_capacity=capacity,
            ring_dtype=ring_dtype,
        ),
        lambda: down_valu(
            x,
            rhs,
            scales,
            counts,
            destinations,
            probabilities,
            tables,
            tokens=tokens,
            columns=columns,
            tile=tile,
            unroll=unroll,
            layer=layer,
            stream_metadata=True,
            metadata_cache_rows=_DEFAULT_ROUTE_CACHE_ROWS,
            ring_dtype=ring_dtype,
        ),
    )
