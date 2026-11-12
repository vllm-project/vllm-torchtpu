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
# src/vllm_torchtpu/megakernel/kimi_k3/kernels/prefill_moe/moe_down_valu.py
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
from typing import TYPE_CHECKING, Protocol, TypedDict, Unpack, cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...common.constants import STORE_TILE as STORE_TILE
from ...common.constants import TP_PEERS, TP_SIZE
from .. import prefill_megakernel as k

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_VALID_TILES = (64, 128, 256)
_VALID_TOKEN_WINDOWS = (4096, 8192)
_DEFAULT_ROUTE_CACHE_ROWS = 8192
_COLLECTIVE_ID = 77
_VMEM_CAPACITY_FRACTION = 0.98


class _AsyncCopy(Protocol):
    """Protocol for a Pallas async DMA copy descriptor."""

    def start(self) -> None:
        """Start the async DMA transfer."""
        ...

    def wait(self) -> None:
        """Wait for the async DMA transfer to complete."""
        ...


class _PallasIndexer(Protocol):
    """Protocol for ``ref.at[...]`` returning a sliced ``_PallasRef`` view."""

    def __getitem__(self, idx: object) -> _PallasRef:
        """Select a sub-reference view for ``idx``."""
        ...


class _PallasRef(Protocol):
    """Protocol for a Pallas memory reference supporting ``.at`` slicing."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the reference tensor shape."""
        ...

    @property
    def dtype(self) -> jnp.dtype:
        """Return the reference element dtype."""
        ...

    @property
    def at(self) -> _PallasIndexer:
        """Return an indexer producing sliced Pallas reference views."""
        ...

    def reshape(self, *shape: int) -> _PallasRef:
        """Return a reshaped view of the reference."""
        ...

    def bitcast(self, dtype: jax.typing.DTypeLike) -> _PallasRef:
        """Reinterpret the reference elements as ``dtype``."""
        ...

    def __getitem__(self, idx: object) -> jax.Array:
        """Read the reference or slice into a JAX array."""
        ...

    def __setitem__(self, idx: object, val: object) -> None:
        """Write ``val`` into the reference slice."""
        ...


class _RingStepOptions(TypedDict):
    """Keyword options for ``ring_step``."""

    streams: Sequence[tuple[int, ...]]
    buffers: int
    step: jax.Array | int


class _DownValuOptions(TypedDict, total=False):
    """Keyword options for ``down_valu``."""

    tokens: int
    columns: int
    tile: int
    unroll: int
    layer: jax.Array | None
    stream_metadata: bool
    metadata_cache_rows: int
    metadata_row_capacity: int
    ring_dtype: jax.typing.DTypeLike


class _DownValuCompactOptions(_DownValuOptions, total=False):
    """Keyword options for ``down_valu_compact``."""

    metadata_prefix_rows: int
    slice_input: bool
    global_route_max: jax.Array | int | None


@dataclasses.dataclass(frozen=True, slots=True)
class DownValuConfig:
    """Configuration for ``down_valu`` GMM tiling and metadata streaming."""

    tokens: int
    columns: int = 1792
    tile: int = 128
    unroll: int = 8
    layer: jax.Array | None = None
    stream_metadata: bool = False
    metadata_cache_rows: int = 0
    metadata_row_capacity: int = 0
    ring_dtype: jax.typing.DTypeLike = jnp.float32


@dataclasses.dataclass(frozen=True)
class _DownKernelContext:
    """Immutable references and tile geometry for ``down_valu`` kernel helpers."""

    cfg: k.GmmConfigs
    metadata: k.MetadataRef
    sums: _PallasRef
    packed: _PallasRef
    cast_buf: _PallasRef
    subtotal_vmem: _PallasRef
    dest_tiles: _PallasRef
    prob_tiles: _PallasRef
    destinations_ref: _PallasRef
    probabilities_ref: _PallasRef
    route_sem: _PallasRef
    route_tags: _PallasRef
    state: _PallasRef
    tokens: int
    tn: int
    width: int
    sub: int
    unroll: int
    route_chunk: int
    route_span: int
    stream_metadata: bool
    metadata_cache_rows: int
    family: bool
    regions: Sequence[tuple[int, ...]]


def _stream_ring_descriptor(
    tables_ref: _PallasRef,
    o_ref: _PallasRef,
    region: tuple[int, ...],
    stream_step: tuple[int, jax.Array | int],
    position: jax.Array,
) -> tuple[tuple[jax.Array], tuple[jax.Array], pl.Slice, pl.Slice, pl.Slice]:
    """Build peer coordinates and row/column slices for one ring stream and hop."""
    stream, step = stream_step
    begin, size = region[:2]
    col_begin, width = region[2:] if len(region) == 4 else (0, o_ref.shape[1])
    cols = pl.ds(col_begin, width)
    family = stream // 2 % 3
    direction = -1 if stream % 2 == 0 else 1
    target = (tables_ref[family, 1, (position + direction) % TP_SIZE],)
    source = (tables_ref[family, 1, (position - direction) % TP_SIZE],)
    owner_mesh = tables_ref[
        family,
        1,
        (position - direction * (step + 1)) % TP_SIZE,
    ]
    owner = tables_ref[family, 2, owner_mesh]
    rows = pl.ds(begin, size)
    source_rows = pl.ds(owner * o_ref.shape[0] + begin, size)
    return (target, source, rows, cols, source_rows)


def _ring_stream_hop(
    stream: int,
    descriptor: tuple[tuple[jax.Array], tuple[jax.Array], pl.Slice, pl.Slice, pl.Slice],
    step: jax.Array | int,
    buffers: int,
    *refs: _PallasRef,
) -> None:
    """Issue wait, accumulate, credit signal, and forward DMA for one ring stream."""
    x_ref, o_arr, send_sems, recv_sems, credit_sems, payload, received = refs
    target, source, rows, cols, source_rows = descriptor

    def local_value() -> jax.Array:
        """Load the local subtotal slice for this stream and ring hop."""
        return (
            x_ref.reshape(x_ref.shape[0], o_arr.shape[1])
            .at[source_rows, cols][...]
            .astype(payload.dtype)
        )

    def remote(index: jax.Array | int) -> _AsyncCopy:
        """Create the async remote DMA descriptor for hop ``index``."""
        return cast(
            "_AsyncCopy",
            pltpu.make_async_remote_copy(
                payload.at[rows, cols],
                received.at[index % buffers, rows, cols],
                send_sems.at[stream],
                recv_sems.at[index % buffers, stream],
                device_id=target,
                device_id_type=pl.DeviceIdType.MESH,
            ),
        )

    @pl.when(step > 0)
    def wait() -> None:
        """Wait for the previous hop's incoming remote DMA to complete."""
        remote(step - 1).wait()

    @pl.when(step == 0)
    def seed() -> None:
        """Initialize the ring payload buffer with hop 0's local subtotal."""
        payload.at[rows, cols][...] = local_value()

    @pl.when(step > 0)
    def add() -> None:
        """Accumulate the received peer shard and return credit."""
        payload.at[rows, cols][...] = (
            local_value() + received.at[(step - 1) % buffers, rows, cols][...]
        )
        pl.semaphore_signal(
            credit_sems.at[stream],
            1,
            device_id=source,
            device_id_type=pl.DeviceIdType.MESH,
        )

    @pl.when(step < TP_PEERS)
    def forward() -> None:
        """Acquire flow-control credit and start the outgoing remote DMA."""

        @pl.when(step >= buffers)
        def acquire() -> None:
            """Wait for downstream buffer credit before reusing a receive slot."""
            pl.semaphore_wait(credit_sems.at[stream], 1)

        remote(step).start()

    @pl.when(step == TP_PEERS)
    def finish() -> None:
        """Drain all outstanding flow-control credits after the final hop."""
        pl.semaphore_wait(credit_sems.at[stream], buffers)


def ring_step(
    tables_ref: object,
    x_ref: object,
    o_ref: object,
    *ring_refs: object,
    **ring_opts: Unpack[_RingStepOptions],
) -> None:
    """Advance one hop of the 32-rank multi-stream reduce-scatter ring."""
    send_sems, recv_sems, credit_sems, store_sem, payload, received = (
        cast("_PallasRef", r) for r in ring_refs
    )
    tables_pallas = cast("_PallasRef", tables_ref)
    x_pallas = cast("_PallasRef", x_ref)
    o_pallas = cast("_PallasRef", o_ref)
    streams = ring_opts["streams"]
    buffers = ring_opts["buffers"]
    step = ring_opts["step"]
    rank = lax.axis_index("k3_sp")

    positions = [tables_pallas[family_idx, 0, rank] for family_idx in range(3)]

    @pl.when(step == 0)
    def _enter_ring() -> None:
        """Synchronize all ranks on the ring barrier before hop 0."""
        k.ring_enter(cast("jax.Array", tables_ref), rank)

    # Every stream's descriptor is built before any hop is issued, as in the port
    # kernel, so the lowered op order matches it.
    descriptors = [
        _stream_ring_descriptor(
            tables_pallas,
            o_pallas,
            region,
            (stream, step),
            positions[stream // 2 % 3],
        )
        for stream, region in enumerate(streams)
    ]
    for stream, descriptor in enumerate(descriptors):
        _ring_stream_hop(
            stream,
            descriptor,
            step,
            buffers,
            x_pallas,
            o_pallas,
            send_sems,
            recv_sems,
            credit_sems,
            payload,
            received,
        )

    @pl.when(step == TP_PEERS)
    def store_complete_shard() -> None:
        """Write the fully reduced shard from VMEM to HBM output."""
        output = pltpu.make_async_copy(payload, o_pallas, store_sem)
        output.start()
        output.wait()


def _prepare_down_valu_cfg(
    x: jax.Array,
    rhs: jax.Array,
    scales: jax.Array,
    counts: jax.Array,
    spec: DownValuConfig,
) -> tuple[k.GmmConfigs, int, int, int, jax.Array, jax.Array]:
    """Validate parameters and construct the GMM config for ``down_valu``.

    Invariants:
        assert tile in (64, 128, 256) and unroll > 0 and tokens in (4096, 8192)
        assert rows <= x.shape[0] and width % columns == 0
        assert not metadata_cache_rows or stream_metadata
        assert metadata_cache_rows % 128 == 0
        assert family == (layer is not None)
    """
    tokens = spec.tokens
    columns = spec.columns
    tile = spec.tile
    unroll = spec.unroll
    layer = spec.layer
    stream_metadata = spec.stream_metadata
    metadata_cache_rows = spec.metadata_cache_rows
    metadata_row_capacity = spec.metadata_row_capacity

    if tile not in _VALID_TILES or unroll <= 0 or tokens not in _VALID_TOKEN_WINDOWS:
        msg = f"Invalid down_valu tile={tile}, unroll={unroll}, or tokens={tokens}"
        raise ValueError(msg)
    rows, width = (metadata_row_capacity or x.shape[0], rhs.shape[-1])
    if rows > x.shape[0] or width % columns != 0:
        msg = f"Invalid rows={rows} for x.shape={x.shape} or width={width} % {columns}"
        raise ValueError(msg)
    if metadata_cache_rows and (
        not stream_metadata or metadata_cache_rows % STORE_TILE != 0
    ):
        msg = "metadata_cache_rows requires stream_metadata and 128-row alignment"
        raise ValueError(msg)
    limit = int(pltpu.get_tpu_info().vmem_capacity_bytes * _VMEM_CAPACITY_FRACTION)
    offset = jnp.zeros((1,), jnp.int32)
    family = rhs.ndim == 4
    if family != (layer is not None):
        msg = "Resident weights require a layer index"
        raise ValueError(msg)
    resolved_layer = offset if layer is None else layer
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
        cfg, tiles=dataclasses.replace(cfg.tiles, tile_m=tile, tile_n=columns)
    )
    max_gm = cfg.dims.size_group + pl.cdiv(rows, tile) - 1
    return cfg, rows, limit, max_gm, offset, resolved_layer


def _build_down_valu_grid_spec(
    cfg: k.GmmConfigs,
    x_dtype: jnp.dtype,
    ring_dtype: jnp.dtype,
    tokens: int,
    *,
    max_gm: int,
    streams: int,
    route_span: int,
    stream_metadata: bool,
    metadata_cache_rows: int,
) -> pltpu.PrefetchScalarGridSpec:
    """Build the ``PrefetchScalarGridSpec`` for ``down_valu``."""
    tm, tn = (cfg.tiles.tile_m, cfg.tiles.tile_n)
    localrows = tokens // TP_SIZE
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    in_prefix = (hbm, hbm) if stream_metadata else ()
    return pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=4 if stream_metadata else 6,
        in_specs=(*in_prefix, hbm, k.WeightsRef(weight=hbm, scale=hbm)),
        out_specs=hbm,
        scratch_shapes=(
            None,  # partial: packed_inner_kernel stores rows directly
            pltpu.VMEM((tm, tn), jnp.float32),
            k.MetadataRef(
                gm_id_to_group_id=pltpu.SMEM((max_gm,), jnp.int32),
                gm_id_to_m_offset=pltpu.SMEM((max_gm + 1,), jnp.int32),
            ),
            pltpu.VMEM((2, tm, 2, tn // 2), x_dtype),
            pltpu.VMEM((tokens, 2, tn // 2), jnp.float32),
            pltpu.VMEM((STORE_TILE, tn), x_dtype),
            pltpu.SemaphoreType.DMA((streams,)),
            pltpu.SemaphoreType.DMA((2, streams)),
            pltpu.SemaphoreType.REGULAR((streams,)),
            pltpu.SemaphoreType.DMA,
            pltpu.VMEM((localrows, tn), ring_dtype),
            pltpu.VMEM((2, localrows, tn), ring_dtype),
            pltpu.SMEM((1,), jnp.int32),
            pltpu.VMEM((tokens, tn), x_dtype),
            pltpu.SMEM((2, route_span), jnp.int32) if stream_metadata else None,
            pltpu.SMEM((2, route_span), jnp.float32) if stream_metadata else None,
            pltpu.SemaphoreType.DMA if stream_metadata else None,
            pltpu.SMEM((2,), jnp.int32) if metadata_cache_rows else None,
        ),
    )


def _route_entry(
    ctx: _DownKernelContext,
    tile_g: jax.Array,
    row_idx: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Read token destination and routing probability for row ``row_idx``."""
    if ctx.stream_metadata:
        chunk = ctx.metadata.gm_id_to_m_offset[tile_g] // ctx.route_chunk
        begin = chunk * ctx.route_chunk
        return (
            ctx.dest_tiles[chunk % 2, row_idx - begin],
            ctx.prob_tiles[chunk % 2, row_idx - begin],
        )
    return (ctx.destinations_ref[row_idx], ctx.probabilities_ref[row_idx])


def _consume_down_tile(
    ctx: _DownKernelContext,
    tile_g: jax.Array,
    source: _PallasRef,
    first: int = 0,
) -> None:
    """Accumulate remaining rows of ``source`` for group tile ``tile_g``."""
    start = ctx.metadata.gm_id_to_m_offset[tile_g]
    end = ctx.metadata.gm_id_to_m_offset[tile_g + 1]
    begin = start // ctx.sub * ctx.sub
    start = jnp.minimum(start + first, end)

    def update(row_idx: jax.Array) -> None:
        """Weight and accumulate a single row into its destination token."""
        dest, weight = _route_entry(ctx, tile_g, row_idx)
        value = source.at[pl.ds(row_idx - begin, 1)][...].astype(jnp.float32)
        target = ctx.sums.at[pl.ds(dest, 1)]
        target[...] = target[...] + value * weight

    full = (end - start) // ctx.unroll

    @pl.loop(0, full)
    def chunk(b: jax.Array) -> None:
        """Accumulate one unrolled batch of rows."""
        for u in range(ctx.unroll):
            update(start + b * ctx.unroll + u)

    @pl.loop(start + full * ctx.unroll, end)
    def tail(row_idx: jax.Array) -> None:
        """Accumulate any remainder rows after the unrolled loop."""
        update(row_idx)


def _load_streamed_routes(ctx: _DownKernelContext, g: jax.Array) -> None:
    """Prefetch streamed routing metadata into SMEM when crossing an 8K chunk."""
    if not ctx.stream_metadata:
        return
    chunk = ctx.metadata.gm_id_to_m_offset[g] // ctx.route_chunk
    previous_chunk = (
        ctx.metadata.gm_id_to_m_offset[jnp.maximum(g - 1, 0)] // ctx.route_chunk
    )
    missing = (
        ctx.route_tags[chunk % 2] != chunk
        if ctx.metadata_cache_rows
        else (g == 0) | (chunk != previous_chunk)
    )

    @pl.when(missing)
    def load_routes() -> None:
        """DMA-copy the current metadata chunk's destinations and weights."""
        begin = chunk * ctx.route_chunk
        copies = (
            pltpu.make_async_copy(
                ctx.destinations_ref.at[pl.ds(begin, ctx.route_span)],
                ctx.dest_tiles.at[chunk % 2],
                ctx.route_sem,
            ),
            pltpu.make_async_copy(
                ctx.probabilities_ref.at[pl.ds(begin, ctx.route_span)],
                ctx.prob_tiles.at[chunk % 2],
                ctx.route_sem,
            ),
        )
        for cp in copies:
            cp.start()
        for cp in copies:
            cp.wait()
        if ctx.metadata_cache_rows:
            ctx.route_tags[chunk % 2] = chunk


def _emit_down_subtotals(ctx: _DownKernelContext) -> None:
    """Round completed FP32 token accumulators to BF16 in ``subtotal_vmem``."""

    @pl.loop(0, ctx.tokens // STORE_TILE)
    def block(b: jax.Array) -> None:
        """Convert one 128-token slice of ``sums`` into BF16 ring input."""
        ctx.cast_buf[...] = (
            ctx.sums[pl.ds(b * STORE_TILE, STORE_TILE)]
            .reshape(STORE_TILE, ctx.tn)
            .astype(jnp.bfloat16)
        )
        ctx.subtotal_vmem.at[pl.ds(b * STORE_TILE, STORE_TILE)][...] = ctx.cast_buf[...]


def _overlap_previous_tile(
    ctx: _DownKernelContext,
    g: jax.Array,
    previous_g: jax.Array,
    previous: _PallasRef,
    panel: int,
) -> None:
    """Accumulate ``unroll`` rows of the previous tile after each MXU panel."""
    start = ctx.metadata.gm_id_to_m_offset[previous_g]
    end = ctx.metadata.gm_id_to_m_offset[previous_g + 1]
    begin = start // ctx.sub * ctx.sub
    for row in range(ctx.unroll):
        row_offset = panel * ctx.unroll + row
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
    current: _PallasRef,
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

    @pl.when(n == ctx.width // ctx.tn - 1)
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
    panel_count = pl.cdiv(ctx.tn, pltpu.get_tpu_info().mxu_column_size)

    def inner(
        x_tile: jax.Array,
        w_tile: k.RhsRef,
        _partial: jax.Array,
        acc_tile: jax.Array,
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
            advance_fn(
                jnp.minimum(TP_SIZE, (g + 1) * TP_SIZE // ng),
                n - 1,
            )

        current = ctx.packed.at[g % 2]
        previous = ctx.packed.at[(g + 1) % 2]
        previous_g = jnp.maximum(g - 1, 0)

        # These callbacks issue VALU work between MXU panel issue/use. Ping-pong output
        # banks keep the previous tile live throughout.
        def overlap_work(panel: int) -> None:
            """Issue VALU accumulation for ``panel`` of the previous tile."""
            _overlap_previous_tile(ctx, g, previous_g, previous, panel)

        @pl.when(g == 0)
        def initialize_previous() -> None:
            """Zero the previous ping-pong bank before group tile 0."""
            previous[...] = jnp.zeros(previous.shape, previous.dtype)

        @pl.when(active_ng > 0)
        def product() -> None:
            """Compute the current group tile's MXFP4 matmul while overlapping VALU."""
            packed_inner_kernel(
                x_tile,
                w_tile,
                current,
                acc_tile,
                metadata_tile,
                cfgs=ctx.cfg,
                overlap_work=overlap_work,
            )

        @pl.when(g > 0)
        def previous_tail() -> None:
            """Finish any remaining rows of the previous group tile."""
            _consume_down_tile(ctx, previous_g, previous, panel_count * ctx.unroll)

        @pl.when(g == ng - 1)
        def finish() -> None:
            """Complete the final group tile and drain the ring."""
            _finish_down_column(ctx, g, n, current, advance_fn)

    return inner


def _make_down_valu_kernel(
    cfg: k.GmmConfigs,
    regions: Sequence[tuple[int, ...]],
    *,
    tokens: int,
    tn: int,
    width: int,
    sub: int,
    unroll: int,
    route_chunk: int,
    route_span: int,
    stream_metadata: bool,
    metadata_cache_rows: int,
    family: bool,
) -> Callable[..., None]:
    """Build the top-level ``pallas_call`` kernel closure for ``down_valu``."""

    def kernel(*refs: _PallasRef) -> None:
        """Run the fused down-projection, VALU accumulation, and ring pipeline."""
        (
            counts_ref,
            offset_ref,
            layer_ref,
            tables_ref,
            destinations_ref,
            probabilities_ref,
            x_ref,
            w_obj,
            out,
            partial,
            acc,
            metadata_obj,
            packed,
            sums,
            cast_buf,
            sends,
            receives,
            credit_sems,
            stores,
            payload,
            rx,
            state,
            subtotal_vmem,
            dest_tiles,
            prob_tiles,
            route_sem,
            route_tags,
        ) = refs
        w = cast("k.WeightsRef", w_obj)
        metadata = cast("k.MetadataRef", metadata_obj)
        scale_ref = cast("_PallasRef", w.scale)
        if family:
            w = dataclasses.replace(
                w,
                weight=w.weight.at[layer_ref[0]],
                scale=scale_ref.at[layer_ref[0]],
            )
        w = dataclasses.replace(w, weight=w.weight.bitcast(jnp.float4_e2m1fn))
        active_ng = k.fill_metadata(
            cast("jax.Array", counts_ref),
            cast("jax.Array", offset_ref),
            metadata,
            cfgs=cfg,
        )

        @pl.when(active_ng == 0)
        def empty_metadata() -> None:
            """Zero the first metadata slot when a rank has no active expert tiles."""
            metadata.gm_id_to_group_id[0] = 0
            metadata.gm_id_to_m_offset[0] = 0
            metadata.gm_id_to_m_offset[1] = 0

        # Empty ranks retain the same collective call sites as active ranks. A separate
        # empty-rank ring branch can allocate different semaphores.
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
                    buffers=2,
                    step=step,
                )

            state[0] = stop

        ctx = _DownKernelContext(
            cfg=cfg,
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
            tokens=tokens,
            tn=tn,
            width=width,
            sub=sub,
            unroll=unroll,
            # Cache adjacent 8K metadata chunks across product and column tiles.
            route_chunk=route_chunk,
            route_span=route_span,
            stream_metadata=stream_metadata,
            metadata_cache_rows=metadata_cache_rows,
            family=family,
            regions=regions,
        )
        inner = _make_down_valu_inner(ctx, active_ng, ng, advance)
        pltpu.emit_pipeline(
            inner,
            grid=(width // tn, ng, 1),
            in_specs=(xs, ws),
            out_specs=(),
        )(
            x_ref.reshape(-1, sub, x_ref.shape[-1]),
            w,
            scratches=(partial, acc, metadata),
        )

    return kernel


def down_valu(
    x: jax.Array,
    rhs: jax.Array,
    scales: jax.Array,
    counts: jax.Array,
    destinations: jax.Array,
    probabilities: jax.Array,
    tables_obj: object,
    **down_opts: Unpack[_DownValuOptions],
) -> jax.Array:
    """Complete BF16 products are immediately combined with FP32 routing weights.

    Destination metadata follows expert-major product rows. There is one writer
    per local token accumulator. Columns share one bounded accumulator; only
    complete local subtotals are rounded to BF16 for the ring, whose running
    sum and reduced output are ``ring_dtype``.
    """
    tokens = down_opts["tokens"]
    columns = down_opts.get("columns", 1792)
    tile = down_opts.get("tile", 128)
    unroll = down_opts.get("unroll", 8)
    stream_metadata = down_opts.get("stream_metadata", False)
    metadata_cache_rows = down_opts.get("metadata_cache_rows", 0)
    ring_dtype = jnp.dtype(down_opts.get("ring_dtype", jnp.float32))
    spec = DownValuConfig(
        tokens=tokens,
        columns=columns,
        tile=tile,
        unroll=unroll,
        layer=down_opts.get("layer"),
        stream_metadata=stream_metadata,
        metadata_cache_rows=metadata_cache_rows,
        metadata_row_capacity=down_opts.get("metadata_row_capacity", 0),
        ring_dtype=ring_dtype,
    )

    cfg, rows, limit, max_gm, offset, resolved_layer = _prepare_down_valu_cfg(
        x, rhs, scales, counts, spec
    )
    width = rhs.shape[-1]
    tm, tn, localrows = (tile, columns, tokens // TP_SIZE)
    family = rhs.ndim == 4
    sub = cfg.dims.size_lhs_sublane
    regions = k.regions(localrows, tn)
    streams = len(regions)
    route_chunk = min(rows, metadata_cache_rows or _DEFAULT_ROUTE_CACHE_ROWS)
    route_span = route_chunk + tm

    kernel = _make_down_valu_kernel(
        cfg,
        regions,
        tokens=tokens,
        tn=tn,
        width=width,
        sub=sub,
        unroll=unroll,
        route_chunk=route_chunk,
        route_span=route_span,
        stream_metadata=stream_metadata,
        metadata_cache_rows=metadata_cache_rows,
        family=family,
    )
    grid_spec = _build_down_valu_grid_spec(
        cfg,
        x.dtype,
        ring_dtype,
        tokens,
        max_gm=max_gm,
        streams=streams,
        route_span=route_span,
        stream_metadata=stream_metadata,
        metadata_cache_rows=metadata_cache_rows,
    )
    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((localrows, width), ring_dtype),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(
            collective_id=_COLLECTIVE_ID, vmem_limit_bytes=limit
        ),
        name=f"down_valu_m{tm}_n{tn}_u{unroll}_mc{metadata_cache_rows}",
    )
    return call(
        counts,
        offset,
        resolved_layer,
        jnp.asarray(tables_obj, jnp.int32),
        jnp.pad(destinations, (0, route_span)) if stream_metadata else destinations,
        jnp.pad(probabilities, (0, route_span)) if stream_metadata else probabilities,
        x,
        k.WeightsRef(weight=rhs, scale=scales),
    )


def _prepare_tiled_rhs(
    tiled_rhs_ref: k.RhsRef,
    cfgs: k.GmmConfigs,
) -> tuple[jax.Array, int, int]:
    """Load and optionally pre-dequantize the RHS weight tile.

    Invariants:
        assert rhs_qbs is not None
    """
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
    tiled_lhs_ref: jax.Array,
    tiled_rhs_ref: k.RhsRef,
    tiled_out_ref: object,
    *acc_and_metadata: jax.Array | k.MetadataRef,
    cfgs: k.GmmConfigs,
    overlap_work: Callable[[int], None],
) -> None:
    """Variant of ``prefill_megakernel.inner_kernel`` with a direct packed final store.

    Arithmetic and row bucketing follow k.inner_kernel, over the complete
    reduction dimension in one tile. Each valid expert row is consumed
    immediately, so no adjacent-group sublane carry is needed. Output is
    [tile_m, 2, tile_n // 2]; acc_ref only sets the accumulation dtype.
    Products round to BF16 before the caller's FP32 routing weighting.
    ``overlap_work(panel)`` runs after each MXU column panel is issued.

    Invariants:
        assert cfgs.dims.size_k == cfgs.tiles.tile_k
        assert cfgs.fuse_act is None
    """
    acc_obj, metadata_obj = acc_and_metadata
    acc_ref = cast("jax.Array", acc_obj)
    metadata_ref = cast("k.MetadataRef", metadata_obj)
    out_ref = cast("_PallasRef", tiled_out_ref)
    if cfgs.dims.size_k != cfgs.tiles.tile_k or cfgs.fuse_act is not None:
        msg = "packed_inner_kernel requires full-K tiling without fused activation"
        raise ValueError(msg)
    gm_id = pl.program_id(1)
    m_start = metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
    m_offset = m_start - m_start % cfgs.dims.size_lhs_sublane
    m_start_local = m_start - m_offset
    m_end_local = m_end - m_offset

    def _matmul(bucket_m: int) -> None:
        """Compute one row-bucketed MXU matmul and store packed BF16 outputs."""
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
        out_ref[:acc_m] = acc_masked.astype(out_ref.dtype).reshape(
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
    rhs: jax.Array | None,
    scales: jax.Array | None,
    counts: jax.Array,
    destinations: jax.Array,
    probabilities: jax.Array,
    tables_obj: object,
    **compact_spec: Unpack[_DownValuCompactOptions],
) -> jax.Array:
    """Uniform compact-prefix fast path with an exact, uncapped overflow path.

    All ranks must select the same collective-containing branch. The caller
    supplies the maximum derived from globally replicated route counts to
    avoid an additional collective. The full activation buffer stays in HBM;
    only the scalar metadata prefix is bounded. Overflow never drops routes.
    """
    rhs_arr = cast("jax.Array", rhs)
    scales_arr = cast("jax.Array", scales)
    capacity = compact_spec["metadata_prefix_rows"]
    slice_input = compact_spec.get("slice_input", True)
    global_route_max = compact_spec.get("global_route_max")
    tokens = compact_spec.get("tokens", 4096)
    compact_opts: _DownValuOptions = {
        "tokens": tokens,
        "columns": compact_spec.get("columns", 1792),
        "tile": compact_spec.get("tile", 128),
        "unroll": compact_spec.get("unroll", 8),
        "layer": compact_spec.get("layer"),
        "ring_dtype": compact_spec.get("ring_dtype", jnp.float32),
        "stream_metadata": False,
        "metadata_cache_rows": 0,
        "metadata_row_capacity": capacity,
    }
    overflow_opts: _DownValuOptions = {
        "tokens": tokens,
        "columns": compact_spec.get("columns", 1792),
        "tile": compact_spec.get("tile", 128),
        "unroll": compact_spec.get("unroll", 8),
        "layer": compact_spec.get("layer"),
        "ring_dtype": compact_spec.get("ring_dtype", jnp.float32),
        "stream_metadata": True,
        "metadata_cache_rows": _DEFAULT_ROUTE_CACHE_ROWS,
    }
    peak = (
        lax.pmax(counts.sum(), "k3_sp")
        if global_route_max is None
        else global_route_max
    )
    fits = peak <= capacity
    return lax.cond(
        fits,
        lambda: down_valu(
            x[:capacity] if slice_input else x,
            rhs_arr,
            scales_arr,
            counts,
            destinations[:capacity],
            probabilities[:capacity],
            tables_obj,
            **compact_opts,
        ),
        lambda: down_valu(
            x,
            rhs_arr,
            scales_arr,
            counts,
            destinations,
            probabilities,
            tables_obj,
            **overflow_opts,
        ),
    )
