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

# Source provenance: inferact/k3-megakernel@30c9ba966b
#   src/vllm_torchtpu/megakernel/kimi_k3/kernels/prefill_moe/fused.py
"""Fused EP16 x TP2 prefill MoE for 512- through 8192-token windows.

Schedule:
  1. Router/latent/shared projections overlap the incremental Top16, route sort,
     route exchange, packed latent ring and shared-output reduction.
  2. Compact routing metadata moves to scalar scratch. GU/SiTU and the first down
     column share a pipeline; later down columns overlap token combine and reduction.
  3. The reduced local shard is normalized, projected up and added to the shared
     expert and residual outputs.

The factory at the end is the only production entry point. The small-window, 4K
and 8K schedules are selected by window size; 8K skew overflow retains every route via
an uncapped fallback. Python generators split work at tracing time, allowing
independent vector/communication work between matrix panels without extra calls.
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
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.kimi_k3.megakernel import config as _config_mod
from vllm_torchtpu.models.kimi_k3.megakernel.common.constants import (
    AXIS_SP,
    DP2_GROUPS,
    HIDDEN_SIZE,
    MIB_BYTES,
    OVERFLOW_WINDOW_TOKENS,
    PREFILL_WINDOWS,
    ROUTED_EXPERTS,
    STORE_TILE,
    SUBLANE_ROWS,
    TP_PEERS,
    TP_SIZE,
    VREG_LANE_COUNT,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import prefill_megakernel as k
from vllm_torchtpu.models.kimi_k3.megakernel.kernels.prefill_moe.moe_down_valu import (
    DEFAULT_DOWN_COLUMNS,
    DEFAULT_DOWN_UNROLL,
    RING_BUFFERS,
    RING_FAMILIES,
    VMEM_CAPACITY_FRACTION,
    down_valu_compact,
)

from .indexed_ragged_gmm import PackedRows, indexed_gate_up

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from jax._src.pallas import core as pl_core
    from jax.sharding import Mesh

    from vllm_torchtpu.models.kimi_k3.megakernel.common.protocols import (
        AsyncCopyProtocol,
        PallasRefProtocol,
    )

TOP_K = 16
COMPACT_METADATA_PREFIX_ROWS = 32768
# 4096-token bitonic sort window and compact metadata envelope.
_COMPACT_WINDOW_THRESHOLD = PREFILL_WINDOWS[-2]
_SORT_WINDOW = _COMPACT_WINDOW_THRESHOLD
_SORT_WINDOW_ROWS = 32
_REGISTER_SWAP_DISTANCE = 1024
_SUBLANE_SWAP_DISTANCE = VREG_LANE_COUNT
_EP_SIZE = TP_SIZE // DP2_GROUPS
_LOCAL_EXPERTS = ROUTED_EXPERTS // _EP_SIZE
_EXPERT_GROUPS = 16
_SMALL_METADATA_TOKEN_THRESHOLD = _COMPACT_WINDOW_THRESHOLD
_PANEL_COLS = 2 * VREG_LANE_COUNT
_LATENT_HALF_COLS = DEFAULT_DOWN_COLUMNS
_LATENT_FULL_COLS = 2 * _LATENT_HALF_COLS
_LATENT_PROJ_COLS = 4096
_LATENT_PAIR_PANELS = _LATENT_FULL_COLS // _PANEL_COLS
_ROUTE_GATHER_STAGES = 5
_DIRECTED_RINGS = 2 * RING_FAMILIES
_RING_STREAMS = 2 * _DIRECTED_RINGS
_RING_PHASE_DONE = 2
# Final stream phase after draining all credit semaphores; the ring barrier stays
# outside the inner expert loop so both alternating receive semaphores drain cleanly
# before peer reuse.
_RING_PHASE_RELEASED = 3
_RING_META_STRIDE = 34
_SMALL_WINDOW_MAX_LOCAL_ROWS = _SMALL_METADATA_TOKEN_THRESHOLD // TP_SIZE
_MAX_LOCAL_ROWS = OVERFLOW_WINDOW_TOKENS // TP_SIZE
_VALID_LOCAL_ROW_COUNTS = tuple(window // TP_SIZE for window in PREFILL_WINDOWS)
_LAST_CREDIT_RETURN_STEP = TP_PEERS - 1
_STRIPE_MAX_ROWS = k.STRIPE
_EXPERT_INTERMEDIATE_COLS = 1536
_SHARED_INTERMEDIATE_COLS = 2 * _EXPERT_INTERMEDIATE_COLS
_SHARED_GU_PANELS = _SHARED_INTERMEDIATE_COLS // _PANEL_COLS
_SHARED_DOWN_PANELS = HIDDEN_SIZE // _PANEL_COLS
_FUSED_GU_WIDTH = 2 * _EXPERT_INTERMEDIATE_COLS
_COMPACT_DOWN_COLUMNS = DEFAULT_DOWN_COLUMNS // 2
_DOWN_8K_HALF_TILE_COLS = (_COMPACT_DOWN_COLUMNS + VREG_LANE_COUNT) // 2
_DEFAULT_DOWN_TILE = STORE_TILE
_COMPACT_DOWN_TILE = 2 * STORE_TILE
_COMPACT_DOWN_UNROLL = 32
_SPARSE_POLL_MIN_GROUPS = 72
_EPILOGUE_PANEL_COLS = 2 * _PANEL_COLS
_EPILOGUE_BANKS = 3
_RMS_EPS = 1e-5
_FUSED_VMEM_LIMIT_BYTES = 64 * MIB_BYTES
_NUM_LAYER_INPUT_SPECS = 13


@dataclasses.dataclass(frozen=True, slots=True)
class PreparedRoutes:
    """Prepared expert-major token routing buffers."""

    latent: jax.Array
    sorted_tokens: jax.Array
    sorted_probs: jax.Array
    counts: jax.Array


@dataclasses.dataclass(frozen=True, slots=True)
class RoutedExpertWeights:
    """Resident or layer-sliced routed expert MXFP4 weights and scales."""

    gate_up: jax.Array
    gate_up_scales: jax.Array
    down: jax.Array
    down_scales: jax.Array
    expert_layer: jax.Array | None = None


def _bitonic_partner(
    value: jax.Array,
    distance: int,
    rows: int,
    lane: jax.Array,
) -> jax.Array:
    """Permute register or lane elements by ``distance`` for bitonic compare-swap."""
    if distance >= _REGISTER_SWAP_DISTANCE:
        # Reorder register-sized chunks without a gather-index constant.
        chunk_idx = distance // _REGISTER_SWAP_DISTANCE
        return jnp.concatenate(
            [
                value[
                    SUBLANE_ROWS * (i ^ chunk_idx) : SUBLANE_ROWS * (i ^ chunk_idx)
                    + SUBLANE_ROWS
                ]
                for i in range(rows // SUBLANE_ROWS)
            ],
            axis=0,
        )
    if distance >= _SUBLANE_SWAP_DISTANCE:
        shape3 = (rows // SUBLANE_ROWS, SUBLANE_ROWS, VREG_LANE_COUNT)
        index = jax.lax.broadcasted_iota(jnp.int32, shape3, 1)
        return jnp.take_along_axis(
            value.reshape(shape3),
            index ^ (distance // _SUBLANE_SWAP_DISTANCE),
            axis=1,
            mode="promise_in_bounds",
        ).reshape((rows, VREG_LANE_COUNT))
    return jnp.take_along_axis(value, lane ^ distance, axis=1, mode="promise_in_bounds")


@dataclasses.dataclass(frozen=True, slots=True)
class _BitonicSortBuffers:
    """Key/probability references and index grids for in-place bitonic sorting."""

    keys: PallasRefProtocol
    probabilities: PallasRefProtocol
    positions: jax.Array
    lane: jax.Array
    size: int


def _flush_bitonic_pending(
    bufs: _BitonicSortBuffers,
    pending: list[tuple[int, int]],
) -> Iterator[None]:
    """Execute queued intra-window bitonic stages across all 4096-route windows."""
    if not pending:
        return
    for ordinal in range(bufs.size // _SORT_WINDOW):
        row_slice = pl.ds(ordinal * _SORT_WINDOW_ROWS, _SORT_WINDOW_ROWS)
        kr = bufs.keys.at[row_slice, :]
        pr = bufs.probabilities.at[row_slice, :]
        x, p = kr[...], pr[...]
        global_position = bufs.positions + ordinal * _SORT_WINDOW
        for distance, group in pending:
            y = _bitonic_partner(x, distance, _SORT_WINDOW_ROWS, bufs.lane)
            q = _bitonic_partner(p, distance, _SORT_WINDOW_ROWS, bufs.lane)
            descending = (global_position & group) != 0
            upper = (global_position & distance) != 0
            take_min = ~(descending ^ upper)
            keep = ~((x < y) ^ take_min)
            x, p = jnp.where(keep, x, y), jnp.where(keep, p, q)
        kr[...], pr[...] = x, p
        yield
    pending.clear()


def _sort_route_steps(
    keys: PallasRefProtocol,
    probabilities: PallasRefProtocol,
    *,
    size: int,
) -> Iterator[None]:
    """Yield after each 4096-route window of the in-place bitonic network.

    Keys encode (local expert, original route position), so all live keys are
    unique and the result has deterministic expert-major/token-major order.
    Probability values follow exactly the same swaps as their keys.
    """
    if size < _SORT_WINDOW or (size & (size - 1)) != 0:
        msg = f"Bitonic route sort requires power-of-two size >= 4096, got {size}"
        raise ValueError(msg)
    shape = (_SORT_WINDOW_ROWS, VREG_LANE_COUNT)
    row = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    lane = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
    bufs = _BitonicSortBuffers(
        keys=keys,
        probabilities=probabilities,
        positions=row * VREG_LANE_COUNT + lane,
        lane=lane,
        size=size,
    )
    pending: list[tuple[int, int]] = []
    for stage in range(1, size.bit_length()):
        for shift in reversed(range(stage)):
            distance, group = 1 << shift, 1 << stage
            if distance < _SORT_WINDOW:
                pending.append((distance, group))
                continue
            yield from _flush_bitonic_pending(bufs, pending)
            stride = distance // _SORT_WINDOW
            for ordinal in range(size // (2 * _SORT_WINDOW)):
                low = ordinal // stride * (2 * stride) + ordinal % stride
                high = low + stride
                low_slice = pl.ds(low * _SORT_WINDOW_ROWS, _SORT_WINDOW_ROWS)
                high_slice = pl.ds(high * _SORT_WINDOW_ROWS, _SORT_WINDOW_ROWS)
                kl, kh = keys.at[low_slice, :], keys.at[high_slice, :]
                plow = probabilities.at[low_slice, :]
                phigh = probabilities.at[high_slice, :]
                x, y, p, q = kl[...], kh[...], plow[...], phigh[...]
                ascending = (low * _SORT_WINDOW & group) == 0
                keep = ~((x < y) ^ ascending)
                kl[...], kh[...] = jnp.where(keep, x, y), jnp.where(keep, y, x)
                plow[...], phigh[...] = jnp.where(keep, p, q), jnp.where(keep, q, p)
                yield
    yield from _flush_bitonic_pending(bufs, pending)


class _Top16:
    """Incremental biased sigmoid Top16 selector breaking ties by lowest expert ID."""

    def __init__(self, logits: jax.Array, bias: jax.Array) -> None:
        """Initialize candidate scores and output accumulators from router logits."""
        rows, _experts = logits.shape
        self.raw = 1.0 / (1.0 + jnp.exp(-logits))
        self.scores = self.raw + bias
        self.experts = jax.lax.broadcasted_iota(jnp.int32, logits.shape, 1)
        self.slots = jax.lax.broadcasted_iota(jnp.int32, (rows, TOP_K), 1)
        self.selected = jnp.zeros((rows, TOP_K), jnp.int32)
        self.probabilities = jnp.zeros((rows, TOP_K), jnp.float32)
        self.index = 0

    def step(self) -> None:
        """Select the next highest biased-sigmoid expert per token row."""
        best = jnp.max(self.scores, axis=1, keepdims=True)
        winner = jnp.min(
            jnp.where(self.scores == best, self.experts, self.scores.shape[1]),
            axis=1,
            keepdims=True,
        )
        chosen = self.experts == winner
        probability = jnp.sum(jnp.where(chosen, self.raw, 0.0), axis=1, keepdims=True)
        self.selected = jnp.where(self.slots == self.index, winner, self.selected)
        self.probabilities = jnp.where(
            self.slots == self.index, probability, self.probabilities
        )
        self.scores = jnp.where(chosen, -jnp.inf, self.scores)
        self.index += 1

    def finish(self, valid: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Complete all 16 selections and normalize probabilities on valid rows."""
        while self.index < TOP_K:
            self.step()
        denominator = jnp.maximum(
            jnp.sum(self.probabilities, axis=1, keepdims=True), 1e-30
        )
        return self.selected, jnp.where(valid, self.probabilities / denominator, 0.0)


def _unpack_metadata_chunk(
    wire: PallasRefProtocol,
    chunk: int,
    tokens: int,
) -> tuple[jax.Array, jax.Array]:
    """Slice one 128-token chunk of expert IDs and probabilities from ``wire``."""
    if tokens < _SMALL_METADATA_TOKEN_THRESHOLD:
        records = wire[pl.ds(chunk * 32, 32), :].reshape(VREG_LANE_COUNT, 32)
        ids = records[:, :TOP_K]
        probs = jax.lax.bitcast_convert_type(records[:, TOP_K:], jnp.float32)
        return ids, probs
    # Keep IDs and probabilities in separate planes so 128-token
    # chunks need no interleaved-record reshape during metadata prep.
    chunks_per_rank = tokens // _SMALL_METADATA_TOKEN_THRESHOLD
    rank_base = (chunk // chunks_per_rank) * (tokens // VREG_LANE_COUNT)
    chunk_base = (chunk % chunks_per_rank) * TOP_K
    ids = wire[pl.ds(rank_base + chunk_base, TOP_K), :]
    probs = jax.lax.bitcast_convert_type(
        wire[pl.ds(rank_base + tokens // 256 + chunk_base, TOP_K), :],
        jnp.float32,
    )
    return ids, probs


def _metadata_steps(
    wire: PallasRefProtocol,
    keys: PallasRefProtocol,
    probabilities: PallasRefProtocol,
    token_ids: PallasRefProtocol,
    counts: PallasRefProtocol,
    maximum: PallasRefProtocol | None,
    *,
    tokens: int,
    rank: jax.Array,
    counts_ready: Callable[[], None],
) -> Iterator[None]:
    """Build stable local routing metadata, yielding one prep/sort chunk at a time.

    The all-gathered wire contains 16 expert IDs followed by 16 probability bit
    patterns per token. Every rank counts all 16 expert groups at 8K, making the
    compact-capacity decision uniform before any conditional collective runs.
    """
    bits = (tokens * TOP_K - 1).bit_length()
    shape = (
        (VREG_LANE_COUNT, TOP_K)
        if tokens < _SMALL_METADATA_TOKEN_THRESHOLD
        else (TOP_K, VREG_LANE_COUNT)
    )
    expert_index = jnp.arange(_LOCAL_EXPERTS, dtype=jnp.int32)
    group_index = jnp.arange(_EXPERT_GROUPS, dtype=jnp.int32)
    lanes = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
    rows = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    total_counts = jnp.zeros((_LOCAL_EXPERTS,), jnp.int32)
    total_groups = jnp.zeros((_EXPERT_GROUPS,), jnp.int32)
    for chunk in range(tokens // VREG_LANE_COUNT):
        ids, probs = _unpack_metadata_chunk(wire, chunk, tokens)
        local = ids - (rank // 2 * _LOCAL_EXPERTS)
        valid = (probs != 0) & (local >= 0) & (local < _LOCAL_EXPERTS)
        local = jnp.where(valid, local, _LOCAL_EXPERTS)
        original_position = chunk * 2048 + rows * shape[1] + lanes
        packed = (local << bits) | original_position
        keys[pl.ds(chunk * TOP_K, TOP_K), :] = packed.reshape(TOP_K, VREG_LANE_COUNT)
        probabilities[pl.ds(chunk * TOP_K, TOP_K), :] = probs.reshape(
            TOP_K, VREG_LANE_COUNT
        )
        local_flat = local.reshape(TOP_K, VREG_LANE_COUNT)
        ids_flat = ids.reshape(TOP_K, VREG_LANE_COUNT)
        live_flat = probs.reshape(TOP_K, VREG_LANE_COUNT) != 0
        tile_counts = jnp.zeros((_LOCAL_EXPERTS,), jnp.int32)
        for expert in range(_LOCAL_EXPERTS):
            count = jnp.sum((local_flat == expert).astype(jnp.int32))
            tile_counts = jnp.where(expert_index == expert, count, tile_counts)
        total_counts += tile_counts
        if tokens == OVERFLOW_WINDOW_TOKENS:
            tile_groups = jnp.zeros((_EXPERT_GROUPS,), jnp.int32)
            groups = ids_flat // _LOCAL_EXPERTS
            for group in range(_EXPERT_GROUPS):
                count = jnp.sum((live_flat & (groups == group)).astype(jnp.int32))
                tile_groups = jnp.where(group_index == group, count, tile_groups)
            total_groups += tile_groups
        yield
    counts[...] = total_counts
    if maximum is not None:
        maximum[...] = jnp.max(total_groups)[None]
    counts_ready()
    yield from _sort_route_steps(keys, probabilities, size=tokens * TOP_K)

    @pl.loop(0, tokens * TOP_K // _SORT_WINDOW)
    def extract(chunk: jax.Array) -> None:
        """Unpack sorted source token indices from the low bits of ``keys``."""
        index = pl.ds(chunk * _SORT_WINDOW_ROWS, _SORT_WINDOW_ROWS)
        token_ids[index, :] = (keys[index, :] & ((1 << bits) - 1)) // TOP_K


def _dense_panels(
    lhs: jax.Array,
    weights: PallasRefProtocol,
    buffers: Sequence[PallasRefProtocol],
    semaphores: Sequence[PallasRefProtocol],
    *,
    columns: int,
) -> Iterator[tuple[int, jax.Array]]:
    """Stream 256-column BF16 panels, with the next weight DMA in flight.

    The yielded FP32 product is deliberately not stored here. The caller places
    independent routing/vector/communication work before consuming it so the
    compiler can schedule that work alongside the MXU product.
    """

    def source(panel: int) -> PallasRefProtocol:
        """Select column slice ``panel`` from ``weights``."""
        return weights.at[:, pl.ds(panel * _PANEL_COLS, _PANEL_COLS)]

    pltpu.make_async_copy(source(0), buffers[0], semaphores[0]).start()
    for panel in range(columns // _PANEL_COLS):
        bank = panel % 2
        pltpu.make_async_copy(source(panel), buffers[bank], semaphores[bank]).wait()
        if panel + 1 < columns // _PANEL_COLS:
            pltpu.make_async_copy(
                source(panel + 1), buffers[1 - bank], semaphores[1 - bank]
            ).start()
        yield (
            panel * _PANEL_COLS,
            jnp.matmul(lhs, buffers[bank][...], preferred_element_type=jnp.float32),
        )


def _latent_half_pair_panels(
    lhs: jax.Array,
    weights: PallasRefProtocol,
    buffers: Sequence[PallasRefProtocol],
    semaphores: Sequence[PallasRefProtocol],
) -> Iterator[tuple[int, jax.Array]]:
    """Pair matching 256-column fragments of the two 1792-wide latent halves."""

    def source_column(panel: int) -> int:
        """Return starting column offset in ``weights`` for ``panel``."""
        return (panel // 2) * _PANEL_COLS + (panel % 2) * _LATENT_HALF_COLS

    def copy(panel: int) -> AsyncCopyProtocol:
        """Create async weight copy for ``panel``."""
        return cast(
            "AsyncCopyProtocol",
            pltpu.make_async_copy(
                weights.at[:, pl.ds(source_column(panel), _PANEL_COLS)],
                buffers[panel % 4],
                semaphores[panel % 4],
            ),
        )

    for panel in range(2):
        copy(panel).start()
    for panel in range(0, _LATENT_PAIR_PANELS, 2):
        for j in range(2):
            copy(panel + j).wait()
        for j in range(2):
            if panel + 2 + j < _LATENT_PAIR_PANELS:
                copy(panel + 2 + j).start()
        products = [
            jnp.matmul(
                lhs, buffers[(panel + j) % 4][...], preferred_element_type=jnp.float32
            )
            for j in range(2)
        ]
        for j, product in enumerate(products):
            yield source_column(panel + j), product


def _route_gather_steps(
    payload: PallasRefProtocol,
    output: PallasRefProtocol,
    send: PallasRefProtocol,
    recv: PallasRefProtocol,
    *,
    rank: jax.Array,
) -> Iterator[int]:
    """Run five recursive-doubling VMEM exchanges; one completion counter per round.

    Output ranges are immutable once received. A call-entry collective barrier
    is required, and callers must drain this generator before consuming output.
    """
    rows = payload.shape[0]
    output[pl.ds(rank * rows, rows), :] = payload[...]
    for stage in range(_ROUTE_GATHER_STAGES):
        peers = 1 << stage
        peer = rank ^ peers
        origin = rank // peers * (peers * rows)
        incoming = peer // peers * (peers * rows)
        source = output.at[pl.ds(origin, peers * rows), :]
        destination = output.at[pl.ds(origin, peers * rows), :]
        received = output.at[pl.ds(incoming, peers * rows), :]
        transfer = cast(
            "AsyncCopyProtocol",
            pltpu.make_async_remote_copy(
                source,
                destination,
                send.at[stage],
                recv.at[stage],
                device_id=peer,
                device_id_type=pl.DeviceIdType.LOGICAL,
            ),
        )
        transfer.start()
        yield stage
        transfer.wait_send()
        # The local receive address differs from the destination on our peer.
        cast(
            "AsyncCopyProtocol",
            pltpu.make_async_remote_copy(
                received,
                received,
                send.at[stage],
                recv.at[stage],
                device_id=peer,
                device_id_type=pl.DeviceIdType.LOGICAL,
            ),
        ).wait_recv()


def _enter_mesh() -> None:
    """Synchronize all 32 logical ranks on the collective barrier."""
    barrier = pltpu.get_barrier_semaphore()
    for peer in range(TP_SIZE):
        pl.semaphore_signal(
            barrier, 1, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL
        )
    pl.semaphore_wait(barrier, TP_SIZE)


@dataclasses.dataclass(frozen=True, slots=True)
class _LatentGatherRegion:
    """Row slice and peer topology metadata for one latent-gather stream."""

    begin: int
    size: int
    family: int
    direction: int
    position: jax.Array
    peer: jax.Array
    source_peer: jax.Array


@dataclasses.dataclass(frozen=True, slots=True)
class _LatentGatherCtx:
    """References and stream descriptors for the 12-stream latent all-gather."""

    tables: PallasRefProtocol
    payload: PallasRefProtocol
    received: PallasRefProtocol | None
    output: PallasRefProtocol
    send: PallasRefProtocol
    recv: PallasRefProtocol
    credit_sems: PallasRefProtocol
    stores: PallasRefProtocol
    rank: jax.Array
    rows: int
    regions: tuple[_LatentGatherRegion, ...]

    def destination(self, region: _LatentGatherRegion, step: int) -> PallasRefProtocol:
        """Return output slice owned by the rank arriving at ``step``."""
        owner = self.tables[
            region.family,
            1,
            (region.position + 2 * TP_SIZE - region.direction * step) % TP_SIZE,
        ]
        return self.output.at[
            pl.ds(owner * self.rows + region.begin, region.size), :, :
        ]


def _remote_logical_copy(
    source: PallasRefProtocol,
    target: PallasRefProtocol,
    send_sem: PallasRefProtocol,
    recv_sem: PallasRefProtocol,
    peer: jax.Array,
) -> AsyncCopyProtocol:
    """Construct a logical-device async remote copy descriptor."""
    return cast(
        "AsyncCopyProtocol",
        pltpu.make_async_remote_copy(
            source,
            target,
            send_sem,
            recv_sem,
            device_id=peer,
            device_id_type=pl.DeviceIdType.LOGICAL,
        ),
    )


def _latent_gather_small_window(ctx: _LatentGatherCtx) -> Iterator[None]:
    """Forward latent slices through double-buffered scratch for windows <= 4K."""
    received = cast("PallasRefProtocol", ctx.received)
    for step in range(TP_SIZE):
        for stream, region in enumerate(ctx.regions):
            piece = ctx.payload.at[pl.ds(region.begin, region.size), :, :]
            ss = ctx.send.at[stream]
            credit = ctx.credit_sems.at[stream]
            st = ctx.stores.at[stream]
            if step:
                old = received.at[
                    (step - 1) % RING_BUFFERS,
                    pl.ds(region.begin, region.size),
                    :,
                    :,
                ]
                transfer = _remote_logical_copy(
                    piece,
                    old,
                    ss,
                    ctx.recv.at[(step - 1) % RING_BUFFERS, stream],
                    region.peer,
                )
                transfer.wait_send()
                transfer.wait_recv()
                pltpu.make_async_copy(
                    piece, ctx.destination(region, step - 1), st
                ).wait()
                piece[...] = old[...]
                pl.semaphore_signal(
                    credit,
                    1,
                    device_id=region.source_peer,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )
            dest = ctx.destination(region, step)
            store = pltpu.make_async_copy(piece, dest, st)
            store.start()
            if step < TP_PEERS:
                if step >= RING_BUFFERS:
                    pl.semaphore_wait(credit, 1)
                future = received.at[
                    step % RING_BUFFERS,
                    pl.ds(region.begin, region.size),
                    :,
                    :,
                ]
                _remote_logical_copy(
                    piece,
                    future,
                    ss,
                    ctx.recv.at[step % RING_BUFFERS, stream],
                    region.peer,
                ).start()
            else:
                store.wait()
                pl.semaphore_wait(credit, RING_BUFFERS)
        yield


def _latent_gather_large_stream(
    ctx: _LatentGatherCtx,
    stream: int,
    region: _LatentGatherRegion,
    step: int,
    pending: list[tuple[AsyncCopyProtocol, PallasRefProtocol, int] | None],
) -> None:
    """Advance one 8K latent-gather stream at ``step``."""
    entry = pending[stream]
    if entry is not None:
        transfer, incoming, completed_step = entry
        transfer.wait_send()
        _remote_logical_copy(
            incoming,
            incoming,
            ctx.send.at[stream],
            ctx.recv.at[completed_step % RING_BUFFERS, stream],
            region.peer,
        ).wait_recv()
        if completed_step + RING_BUFFERS < TP_PEERS:
            pl.semaphore_signal(
                ctx.credit_sems.at[stream],
                1,
                device_id=region.source_peer,
                device_id_type=pl.DeviceIdType.LOGICAL,
            )
    if step < TP_PEERS:
        source = (
            ctx.payload.at[pl.ds(region.begin, region.size), :, :]
            if step == 0
            else ctx.destination(region, step)
        )
        dest = ctx.destination(region, step)
        incoming = ctx.destination(region, step + 1)
        if step >= RING_BUFFERS:
            pl.semaphore_wait(ctx.credit_sems.at[stream], 1)
        transfer = _remote_logical_copy(
            source,
            dest,
            ctx.send.at[stream],
            ctx.recv.at[step % RING_BUFFERS, stream],
            region.peer,
        )
        transfer.start()
        pending[stream] = (transfer, incoming, step)


def _latent_gather_steps(
    tables: PallasRefProtocol,
    payload: PallasRefProtocol,
    received: PallasRefProtocol | None,
    output: PallasRefProtocol,
    send: PallasRefProtocol,
    recv: PallasRefProtocol,
    credit_sems: PallasRefProtocol,
    stores: PallasRefProtocol,
    *,
    rank: jax.Array,
) -> Iterator[None]:
    """Run twelve directional streams, with two receive slots per stream.

    Windows up to 4K forward through VMEM, copying the completed receive slot into
    payload. 8K forwards immutable final HBM ranges to avoid that copy. The latter
    still returns credits before reusing receive counters; immutable data alone
    does not make reusing an unconsumed semaphore safe.
    """
    rows = payload.shape[0]
    if rows not in _VALID_LOCAL_ROW_COUNTS:
        msg = f"Unsupported local latent row count: {rows}"
        raise ValueError(msg)
    regions: list[_LatentGatherRegion] = []
    begin = 0
    for stream in range(_RING_STREAMS):
        size = rows // _RING_STREAMS + (stream < rows % _RING_STREAMS)
        family = stream // 2 % RING_FAMILIES
        direction = -1 if stream % 2 == 0 else 1
        position = tables[family, 0, rank]
        target = tables[family, 1, (position + TP_SIZE + direction) % TP_SIZE]
        source_peer = tables[family, 1, (position + TP_SIZE - direction) % TP_SIZE]
        regions.append(
            _LatentGatherRegion(
                begin=begin,
                size=size,
                family=family,
                direction=direction,
                position=position,
                peer=target,
                source_peer=source_peer,
            )
        )
        begin += size
    ctx = _LatentGatherCtx(
        tables=tables,
        payload=payload,
        received=received,
        output=output,
        send=send,
        recv=recv,
        credit_sems=credit_sems,
        stores=stores,
        rank=rank,
        rows=rows,
        regions=tuple(regions),
    )
    if rows <= _SMALL_WINDOW_MAX_LOCAL_ROWS:
        yield from _latent_gather_small_window(ctx)
        return
    own = output.at[pl.ds(rank * rows, rows), :, :]
    seed = pltpu.make_async_copy(payload, own, stores)
    seed.start()
    pending: list[tuple[AsyncCopyProtocol, PallasRefProtocol, int] | None] = [
        None
    ] * _RING_STREAMS
    for step in range(TP_SIZE):
        for stream, region in enumerate(ctx.regions):
            _latent_gather_large_stream(ctx, stream, region, step, pending)
        yield
    seed.wait()


def _latent_ready_init(
    tables: PallasRefProtocol,
    payload: PallasRefProtocol,
    output: PallasRefProtocol,
    send: PallasRefProtocol,
    recv: PallasRefProtocol,
    stores: PallasRefProtocol,
    state: PallasRefProtocol,
    meta: PallasRefProtocol,
    *,
    rank: jax.Array,
) -> None:
    """Seed twelve independently progressing streams of the 8K latent gather."""
    rows = payload.shape[0]
    if rows != _MAX_LOCAL_ROWS:
        msg = f"8K ready latent gather requires {_MAX_LOCAL_ROWS} rows, got {rows}"
        raise ValueError(msg)
    for directed in range(_DIRECTED_RINGS):
        family, direction = directed // 2, -1 if directed % 2 == 0 else 1
        position = tables[family, 0, rank]
        meta[directed * _RING_META_STRIDE] = tables[
            family, 1, (position + direction) & TP_PEERS
        ]
        meta[directed * _RING_META_STRIDE + 1] = tables[
            family, 1, (position - direction) & TP_PEERS
        ]

        @pl.loop(0, TP_SIZE)
        def owners(
            step: jax.Array,
            *,
            family: int = family,
            position: jax.Array = position,
            direction: int = direction,
            directed: int = directed,
        ) -> None:
            """Record owner row base for ``step`` along ``directed``."""
            owner = tables[family, 1, (position - direction * step) & TP_PEERS]
            meta[directed * _RING_META_STRIDE + 2 + step] = owner * rows

    pltpu.make_async_copy(
        payload, output.at[pl.ds(rank * rows, rows), :, :], stores
    ).start()
    state[0] = _RING_STREAMS
    begin = 0
    for stream in range(_RING_STREAMS):
        size = rows // _RING_STREAMS + (stream < rows % _RING_STREAMS)
        part = pl.ds(begin, size)
        pltpu.make_async_remote_copy(
            payload.at[part, :, :],
            output.at[pl.ds(rank * rows + begin, size), :, :],
            send.at[stream],
            recv.at[0, stream],
            device_id=meta[(stream % _DIRECTED_RINGS) * _RING_META_STRIDE],
            device_id_type=pl.DeviceIdType.LOGICAL,
        ).start()
        state[1 + stream] = 1
        state[1 + _RING_STREAMS + stream] = 0
        begin += size


@dataclasses.dataclass(frozen=True, slots=True)
class _LatentReadyCtx:
    """References for non-blocking 8K latent-gather progress polls."""

    output: PallasRefProtocol
    send: PallasRefProtocol
    recv: PallasRefProtocol
    credit_sems: PallasRefProtocol
    state: PallasRefProtocol
    meta: PallasRefProtocol


def _step_latent_ready_stream(
    ctx: _LatentReadyCtx,
    stream: int,
    start: int,
    size: int,
) -> None:
    """Poll and advance one stream of the non-blocking 8K latent gather."""
    base = (stream % _DIRECTED_RINGS) * _RING_META_STRIDE
    # TPUv7 DMA semaphore accounting: one unit per 32 logical bytes.
    units = size * 2 * _LATENT_HALF_COLS * 2 // 32
    step = ctx.state[1 + stream]
    phase_slot = 1 + _RING_STREAMS + stream

    @pl.when(ctx.state[phase_slot] == 0)
    def receive() -> None:
        """Consume completed incoming transfer and return credit if ready."""
        bank = (step - 1) & 1
        ready = (pl.semaphore_read(ctx.send.at[stream]) >= units) & (
            pl.semaphore_read(ctx.recv.at[bank, stream]) >= units
        )

        @pl.when(ready)
        def consume() -> None:
            """Complete transfer wait and transition stream state."""
            incoming = ctx.output.at[
                pl.ds(ctx.meta[base + 2 + step] + start, size), :, :
            ]
            # wait_send only consumes its completion count; the HBM view
            # has the same logical transfer size as the initial VMEM source.
            transfer = _remote_logical_copy(
                incoming,
                incoming,
                ctx.send.at[stream],
                ctx.recv.at[bank, stream],
                ctx.meta[base],
            )
            transfer.wait_send()
            transfer.wait_recv()

            @pl.when(step < _LAST_CREDIT_RETURN_STEP)
            def credit() -> None:
                """Signal upstream peer that receive semaphore slot is free."""
                pl.semaphore_signal(
                    ctx.credit_sems.at[stream],
                    1,
                    device_id=ctx.meta[base + 1],
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )

            ctx.state[phase_slot] = jnp.where(step == TP_PEERS, _RING_PHASE_DONE, 1)

            @pl.when(step == TP_PEERS)
            def complete() -> None:
                """Decrement active stream counter after final hop."""
                ctx.state[0] = ctx.state[0] - 1

    @pl.when(ctx.state[phase_slot] == 1)
    def forward() -> None:
        """Issue next outgoing hop once downstream credit is available."""
        available = (step < RING_BUFFERS) | (
            pl.semaphore_read(ctx.credit_sems.at[stream]) >= 1
        )

        @pl.when(available)
        def issue() -> None:
            """Consume credit and start remote copy for ``step``."""

            @pl.when(step >= RING_BUFFERS)
            def acquire() -> None:
                """Wait for downstream credit semaphore."""
                pl.semaphore_wait(ctx.credit_sems.at[stream], 1)

            source = ctx.output.at[pl.ds(ctx.meta[base + 2 + step] + start, size), :, :]
            _remote_logical_copy(
                source,
                source,
                ctx.send.at[stream],
                ctx.recv.at[step & 1, stream],
                ctx.meta[base],
            ).start()
            ctx.state[1 + stream] = step + 1
            ctx.state[phase_slot] = 0


def _latent_ready_progress(
    payload: PallasRefProtocol,
    output: PallasRefProtocol,
    send: PallasRefProtocol,
    recv: PallasRefProtocol,
    credit_sems: PallasRefProtocol,
    stores: PallasRefProtocol,
    state: PallasRefProtocol,
    meta: PallasRefProtocol,
    *,
    rank: jax.Array,
    drain: bool = False,
) -> None:
    """Advance ready streams without blocking unrelated compute or streams.

    A receive consumes both DMA completion counts. Credits protect the two
    alternating receive semaphores from reuse before the peer consumes them.
    The final drain completes every stream before routed experts read HBM.
    """
    rows = payload.shape[0]
    ctx = _LatentReadyCtx(
        output=output,
        send=send,
        recv=recv,
        credit_sems=credit_sems,
        state=state,
        meta=meta,
    )

    def sweep() -> None:
        """Poll all twelve streams once."""
        begin = 0
        for stream in range(_RING_STREAMS):
            size = rows // _RING_STREAMS + (stream < rows % _RING_STREAMS)
            start = begin
            begin += size
            _step_latent_ready_stream(ctx, stream, start, size)

    if drain:

        def body(i: jax.Array) -> jax.Array:
            """Poll streams until all twelve finish."""
            sweep()
            return i + 1

        jax.lax.while_loop(lambda _i: state[0] != 0, body, jnp.int32(0))
        pltpu.make_async_copy(
            payload, output.at[pl.ds(rank * rows, rows), :, :], stores
        ).wait()
    else:

        @pl.when(state[0] != 0)
        def progress() -> None:
            """Run one non-blocking sweep while streams remain active."""
            sweep()


def stripe_validity(rows: int, length: jax.Array, rank: jax.Array) -> jax.Array:
    """Return boolean mask of sequence-parallel token rows below ``length``."""
    row = jax.lax.broadcasted_iota(jnp.int32, (rows, 1), 0)
    stripe = min(rows, _STRIPE_MAX_ROWS)
    return (row // stripe * (TP_SIZE * stripe) + rank * stripe + row % stripe) < length


@dataclasses.dataclass(slots=True)
class _PrologueState:
    """Mutable generator and progress state across prologue overlap phases."""

    route_steps: Iterator[int]
    metadata_steps: Iterator[None]
    latent_steps: Iterator[None]
    route_started: bool = False
    metadata_started: bool = False
    latent_tick: int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class _PrologueCtx:
    """References and rank geometry for ``_prologue``."""

    tables: PallasRefProtocol
    hidden: PallasRefProtocol
    router: PallasRefProtocol
    bias: PallasRefProtocol
    latent_weight: PallasRefProtocol
    shared_gu: PallasRefProtocol
    shared_down: PallasRefProtocol
    shared: PallasRefProtocol
    latent_hbm: PallasRefProtocol
    token_ids: PallasRefProtocol
    probabilities: PallasRefProtocol
    counts: PallasRefProtocol
    maximum: PallasRefProtocol | None
    rows: int
    rank: jax.Array
    parity: jax.Array
    peer: jax.Array
    valid: jax.Array
    counts_ready: Callable[[], None]
    metadata_ready: Callable[[], None]

    def start_metadata(
        self,
        stage_state: dict[str, object],
        state: _PrologueState,
    ) -> None:
        """Initialize the metadata generator once all-gathered routes arrive."""
        if not state.metadata_started:
            state.metadata_started = True
            state.metadata_steps = _metadata_steps(
                cast("PallasRefProtocol", stage_state["route_wire"]),
                cast("PallasRefProtocol", stage_state["keys"]),
                self.probabilities,
                self.token_ids,
                self.counts,
                self.maximum,
                tokens=self.rows * TP_SIZE,
                rank=self.rank,
                counts_ready=self.counts_ready,
            )

    def metadata_progress(self, state: _PrologueState) -> None:
        """Advance a bounded slice of metadata preparation or bitonic sort."""
        steps = 2 if self.rows <= _SMALL_WINDOW_MAX_LOCAL_ROWS else 6
        for _ in range(steps):
            next(state.metadata_steps, None)

    def latent_progress(
        self,
        stage_state: dict[str, object],
        state: _PrologueState,
        *,
        drain: bool = False,
    ) -> None:
        """Advance or drain the all-gather of projected latent activations."""
        if self.rows == _MAX_LOCAL_ROWS:
            if state.latent_tick == 0:
                _latent_ready_init(
                    self.tables,
                    cast("PallasRefProtocol", stage_state["latent_payload"]),
                    self.latent_hbm,
                    cast("PallasRefProtocol", stage_state["latent_send_sems"]),
                    cast("PallasRefProtocol", stage_state["latent_recv_sems"]),
                    cast("PallasRefProtocol", stage_state["latent_stores"]),
                    cast("PallasRefProtocol", stage_state["latent_state"]),
                    cast("PallasRefProtocol", stage_state["latent_meta"]),
                    rank=self.rank,
                )
            else:
                _latent_ready_progress(
                    cast("PallasRefProtocol", stage_state["latent_payload"]),
                    self.latent_hbm,
                    cast("PallasRefProtocol", stage_state["latent_send_sems"]),
                    cast("PallasRefProtocol", stage_state["latent_recv_sems"]),
                    cast("PallasRefProtocol", stage_state["latent_credits"]),
                    cast("PallasRefProtocol", stage_state["latent_stores"]),
                    cast("PallasRefProtocol", stage_state["latent_state"]),
                    cast("PallasRefProtocol", stage_state["latent_meta"]),
                    rank=self.rank,
                    drain=drain,
                )
            state.latent_tick += 1
        elif drain:
            for _ in state.latent_steps:
                pass
        else:
            next(state.latent_steps, None)


def _prologue_setup_and_router(
    ctx: _PrologueCtx,
    stage_state: dict[str, object],
) -> tuple[
    AsyncCopyProtocol,
    tuple[PallasRefProtocol, ...],
    tuple[PallasRefProtocol, ...],
    jax.Array,
    _Top16,
]:
    """Load local inputs, start peer exchange, and launch initial router Top16."""
    pair_ref = cast("PallasRefProtocol", stage_state["pair"])
    input_sem = cast("PallasRefProtocol", stage_state["input_sem"])
    own = pair_ref.at[pl.ds(ctx.parity * ctx.rows, ctx.rows), :]
    copy = pltpu.make_async_copy(ctx.hidden, own, input_sem)
    copy.start()
    copy.wait()
    if ctx.rows <= _SMALL_WINDOW_MAX_LOCAL_ROWS:
        router_dst = cast("PallasRefProtocol", stage_state["router"])
        copy = pltpu.make_async_copy(ctx.router, router_dst, input_sem)
        copy.start()
        copy.wait()
    bias_dst = cast("PallasRefProtocol", stage_state["bias"])
    copy = pltpu.make_async_copy(ctx.bias, bias_dst, input_sem)
    copy.start()
    copy.wait()
    _enter_mesh()
    peer_credit = cast("PallasRefProtocol", stage_state["peer_credit"])
    pl.semaphore_signal(
        peer_credit, 1, device_id=ctx.peer, device_id_type=pl.DeviceIdType.LOGICAL
    )
    pl.semaphore_wait(peer_credit, 1)
    peer_input = _remote_logical_copy(
        own,
        own,
        cast("PallasRefProtocol", stage_state["peer_send"]),
        cast("PallasRefProtocol", stage_state["peer_recv"]),
        ctx.peer,
    )
    peer_input.start()
    gu_sems = cast("tuple[PallasRefProtocol, ...]", stage_state["gu_sems"])
    if ctx.rows == _MAX_LOCAL_ROWS:
        router_gu = cast("PallasRefProtocol", stage_state["router_gu"])
        buffers = tuple(
            router_gu.at[:, pl.ds(i * _PANEL_COLS, _PANEL_COLS)] for i in range(4)
        )
        router_v = router_gu.at[:, :ROUTED_EXPERTS]
        copy = pltpu.make_async_copy(ctx.router, router_v, gu_sems[0])
        copy.start()
        copy.wait()
    else:
        buffers = cast("tuple[PallasRefProtocol, ...]", stage_state["gu_buffers"])
        router_v = cast("PallasRefProtocol", stage_state["router"])
    hidden_tile = own[...]
    logits = jnp.matmul(hidden_tile, router_v[...], preferred_element_type=jnp.float32)
    top = _Top16(logits, bias_dst[...])
    return peer_input, buffers, gu_sems, hidden_tile, top


def _prologue_finish_top16_routes(
    ctx: _PrologueCtx,
    stage_state: dict[str, object],
    state: _PrologueState,
    top: _Top16,
) -> None:
    """Pack completed local Top16 selections and start recursive-doubling gather."""
    ids, probs = top.finish(ctx.valid)
    if ctx.rows < _SMALL_WINDOW_MAX_LOCAL_ROWS:
        wire = jnp.concatenate(
            (ids, jax.lax.bitcast_convert_type(probs, jnp.int32)), axis=1
        )
    else:
        wire = jnp.concatenate(
            (
                ids.reshape(ctx.rows // SUBLANE_ROWS, VREG_LANE_COUNT),
                jax.lax.bitcast_convert_type(probs, jnp.int32).reshape(
                    ctx.rows // SUBLANE_ROWS, VREG_LANE_COUNT
                ),
            ),
            axis=0,
        )
    route_payload = cast("PallasRefProtocol", stage_state["route_payload"])
    route_payload[...] = wire.reshape(ctx.rows // 4, VREG_LANE_COUNT)
    state.route_steps = _route_gather_steps(
        route_payload,
        cast("PallasRefProtocol", stage_state["route_wire"]),
        cast("PallasRefProtocol", stage_state["route_send"]),
        cast("PallasRefProtocol", stage_state["route_recv"]),
        rank=ctx.rank,
    )
    next(state.route_steps, None)
    state.route_started = True


def _write_latent_panel(
    ctx: _PrologueCtx,
    stage_state: dict[str, object],
    col: int,
    product: jax.Array,
    first_half: jax.Array | None,
) -> jax.Array | None:
    """Store one 256-column latent panel or pair two 8K half-panels."""
    if ctx.rows == _MAX_LOCAL_ROWS:
        fragment = product.astype(jnp.bfloat16)
        if col < _LATENT_HALF_COLS:
            return fragment
        if first_half is None:
            msg = "Missing first latent half before paired panel write"
            raise ValueError(msg)
        paired = jnp.concatenate((first_half, fragment), axis=1)
        latent_payload = cast("PallasRefProtocol", stage_state["latent_payload"])
        latent_payload[:, :, pl.ds(col - _LATENT_HALF_COLS, _PANEL_COLS)] = (
            paired.reshape(ctx.rows, 2, _PANEL_COLS)
        )
        return None
    latent_buf = cast("PallasRefProtocol", stage_state["latent"])
    latent_buf[:, pl.ds(col, _PANEL_COLS)] = product.astype(jnp.bfloat16)
    return first_half


def _prologue_project_latent(
    ctx: _PrologueCtx,
    stage_state: dict[str, object],
    state: _PrologueState,
    hidden_tile: jax.Array,
    top: _Top16,
) -> None:
    """Compute latent-down panels while overlapping Top16 selection and route gather."""
    gu_sems = cast("tuple[PallasRefProtocol, ...]", stage_state["gu_sems"])
    is_small = ctx.rows <= _SMALL_WINDOW_MAX_LOCAL_ROWS
    if ctx.rows == _MAX_LOCAL_ROWS:
        router_gu = cast("PallasRefProtocol", stage_state["router_gu"])
        latent_buffers = tuple(
            router_gu.at[:, pl.ds(i * _PANEL_COLS, _PANEL_COLS)] for i in range(4)
        )
        latent_products = _latent_half_pair_panels(
            hidden_tile, ctx.latent_weight, latent_buffers, gu_sems
        )
    else:
        gate_buffers = cast("tuple[PallasRefProtocol, ...]", stage_state["gu_buffers"])[
            :2
        ]
        latent_products = _dense_panels(
            hidden_tile,
            ctx.latent_weight,
            gate_buffers,
            gu_sems[:2],
            columns=_LATENT_PROJ_COLS,
        )
    latent_payload = cast("PallasRefProtocol", stage_state["latent_payload"])
    first_half: jax.Array | None = None
    for col, product in latent_products:
        if state.route_started and next(state.route_steps, None) is None and is_small:
            ctx.start_metadata(stage_state, state)
        if is_small and state.metadata_started:
            ctx.metadata_progress(state)
        first_half = _write_latent_panel(ctx, stage_state, col, product, first_half)
        for _ in range(min(2, TOP_K - top.index)):
            top.step()
        if top.index == TOP_K and not state.route_started:
            _prologue_finish_top16_routes(ctx, stage_state, state, top)
    for _ in state.route_steps:
        pass
    ctx.start_metadata(stage_state, state)
    if is_small:
        latent_buf = cast("PallasRefProtocol", stage_state["latent"])
        latent_payload[...] = latent_buf[:, :_LATENT_FULL_COLS].reshape(
            ctx.rows, 2, _LATENT_HALF_COLS
        )
    state.latent_steps = _latent_gather_steps(
        ctx.tables,
        latent_payload,
        cast("PallasRefProtocol | None", stage_state["latent_recv"]),
        ctx.latent_hbm,
        cast("PallasRefProtocol", stage_state["latent_send_sems"]),
        cast("PallasRefProtocol", stage_state["latent_recv_sems"]),
        cast("PallasRefProtocol", stage_state["latent_credits"]),
        cast("PallasRefProtocol", stage_state["latent_stores"]),
        rank=ctx.rank,
    )


def _prologue_shared_gate_up(
    ctx: _PrologueCtx,
    stage_state: dict[str, object],
    state: _PrologueState,
    buffers: tuple[PallasRefProtocol, ...],
    gu_sems: tuple[PallasRefProtocol, ...],
) -> None:
    """Compute shared-expert gate/up panels overlapped with latent/metadata progress."""
    gate_buffers, up_buffers = buffers[:2], buffers[2:]
    gate_sems, up_sems = gu_sems[:2], gu_sems[2:]
    pair_ref = cast("PallasRefProtocol", stage_state["pair"])
    middle_ref = cast("PallasRefProtocol", stage_state["middle"])

    def gu_copy(
        panel: int,
        offset: int,
        bufs: Sequence[PallasRefProtocol],
        sems: Sequence[PallasRefProtocol],
    ) -> AsyncCopyProtocol:
        """Create async weight copy for shared gate or up ``panel``."""
        bank = panel % 2
        return cast(
            "AsyncCopyProtocol",
            pltpu.make_async_copy(
                ctx.shared_gu.at[:, pl.ds(offset + panel * _PANEL_COLS, _PANEL_COLS)],
                bufs[bank],
                sems[bank],
            ),
        )

    gu_copy(0, 0, gate_buffers, gate_sems).start()
    gu_copy(0, _SHARED_INTERMEDIATE_COLS, up_buffers, up_sems).start()
    if ctx.rows == _MAX_LOCAL_ROWS:
        ctx.latent_progress(stage_state, state)
    for panel in range(_SHARED_GU_PANELS):
        if ctx.rows <= _SMALL_WINDOW_MAX_LOCAL_ROWS:
            ctx.latent_progress(stage_state, state)
        bank = panel % 2
        gu_copy(panel, 0, gate_buffers, gate_sems).wait()
        gu_copy(panel, _SHARED_INTERMEDIATE_COLS, up_buffers, up_sems).wait()
        if panel + 1 < _SHARED_GU_PANELS:
            gu_copy(panel + 1, 0, gate_buffers, gate_sems).start()
            gu_copy(panel + 1, _SHARED_INTERMEDIATE_COLS, up_buffers, up_sems).start()
        for row in (0, ctx.rows):
            lhs = pair_ref[pl.ds(row, ctx.rows), :]
            gate = jnp.matmul(
                lhs, gate_buffers[bank][...], preferred_element_type=jnp.float32
            )
            up = jnp.matmul(
                lhs, up_buffers[bank][...], preferred_element_type=jnp.float32
            )
            ctx.metadata_progress(state)
            activated = (
                (4.0 * jnp.tanh(gate / 4.0))
                * (1.0 / (1.0 + jnp.exp(-gate)))
                * (25.0 * jnp.tanh(up / 25.0))
            )
            middle_ref[
                pl.ds(row, ctx.rows), pl.ds(panel * _PANEL_COLS, _PANEL_COLS)
            ] = activated.astype(jnp.bfloat16)
            if ctx.rows == _MAX_LOCAL_ROWS:
                ctx.latent_progress(stage_state, state)


def _drain_prologue_tail(
    ctx: _PrologueCtx,
    stage_state: dict[str, object],
    state: _PrologueState,
) -> None:
    """Drain remaining metadata exchange and latent ring steps at prologue end."""
    if ctx.rows == _MAX_LOCAL_ROWS:
        for _ in state.metadata_steps:
            ctx.latent_progress(stage_state, state)
        # Start compact VMEM-to-SMEM transfers while the latent ring finishes.
        ctx.metadata_ready()
        ctx.latent_progress(stage_state, state, drain=True)
    else:
        ctx.latent_progress(stage_state, state, drain=True)
        for _ in state.metadata_steps:
            pass


def _prologue_shared_down(
    ctx: _PrologueCtx,
    stage_state: dict[str, object],
    state: _PrologueState,
) -> None:
    """Compute shared-expert down panels and TP2 peer reductions."""
    down_buffers = cast("tuple[PallasRefProtocol, ...]", stage_state["down_buffers"])
    down_sems = cast("tuple[PallasRefProtocol, ...]", stage_state["down_sems"])
    partial_refs = cast("tuple[PallasRefProtocol, ...]", stage_state["partial"])
    peer_y_refs = cast("tuple[PallasRefProtocol, ...]", stage_state["peer_y"])
    reduce_send = cast("tuple[PallasRefProtocol, ...]", stage_state["reduce_send"])
    reduce_recv = cast("tuple[PallasRefProtocol, ...]", stage_state["reduce_recv"])
    middle_ref = cast("PallasRefProtocol", stage_state["middle"])

    def down_copy(panel: int) -> AsyncCopyProtocol:
        """Create async copy for shared-down weight ``panel``."""
        return cast(
            "AsyncCopyProtocol",
            pltpu.make_async_copy(
                ctx.shared_down.at[:, pl.ds(panel * _PANEL_COLS, _PANEL_COLS)],
                down_buffers[panel % 2],
                down_sems[panel % 2],
            ),
        )

    def finish_reduction(pending_item: tuple[AsyncCopyProtocol, int, int]) -> None:
        """Complete TP2 peer exchange and store reduced BF16 shared output."""
        transfer, bank, panel = pending_item
        own = partial_refs[bank][pl.ds(ctx.parity * ctx.rows, ctx.rows), :]
        transfer.wait_send()
        transfer.wait_recv()
        ctx.shared[:, pl.ds(panel * _PANEL_COLS, _PANEL_COLS)] = (
            own + peer_y_refs[bank][...]
        ).astype(jnp.bfloat16)

    down_copy(0).start()
    pending: tuple[AsyncCopyProtocol, int, int] | None = None
    for panel in range(_SHARED_DOWN_PANELS):
        bank = panel % 2
        down_copy(panel).wait()
        if panel + 1 < _SHARED_DOWN_PANELS:
            down_copy(panel + 1).start()
        for row in (0, ctx.rows):
            product = jnp.matmul(
                middle_ref[pl.ds(row, ctx.rows), :],
                down_buffers[bank][...],
                preferred_element_type=jnp.float32,
            )
            ctx.metadata_progress(state)
            partial_refs[bank][pl.ds(row, ctx.rows), :] = product
            if row == ctx.rows:
                if pending is not None:
                    finish_reduction(pending)
                transfer = _remote_logical_copy(
                    partial_refs[bank].at[
                        pl.ds((1 - ctx.parity) * ctx.rows, ctx.rows), :
                    ],
                    peer_y_refs[bank],
                    reduce_send[bank],
                    reduce_recv[bank],
                    ctx.peer,
                )
                transfer.start()
                pending = (transfer, bank, panel)
            if ctx.rows in (_MAX_LOCAL_ROWS, row):
                ctx.latent_progress(stage_state, state)
    if pending is not None:
        finish_reduction(pending)
    _drain_prologue_tail(ctx, stage_state, state)


def _prologue(
    tables: PallasRefProtocol,
    control: PallasRefProtocol,
    hidden: PallasRefProtocol,
    router: PallasRefProtocol,
    bias: PallasRefProtocol,
    latent_weight: PallasRefProtocol,
    shared_gu: PallasRefProtocol,
    shared_down: PallasRefProtocol,
    shared: PallasRefProtocol,
    latent_hbm: PallasRefProtocol,
    token_ids: PallasRefProtocol,
    probabilities: PallasRefProtocol,
    counts: PallasRefProtocol,
    maximum: PallasRefProtocol | None,
    *,
    counts_ready: Callable[[], None],
    metadata_ready: Callable[[], None],
) -> None:
    """Run router, latent projection, and shared experts with incremental collectives.

    The caller retains shared/metadata outputs across this scope. All temporary
    projection and communication buffers die before routed-expert scratch is
    allocated. The initial HBM input load targets our half of the peer buffer;
    both the router and shared expert use this same copy.
    """
    rows, hidden_size = hidden.shape
    rank = jax.lax.axis_index(AXIS_SP)
    parity, peer = rank % 2, rank ^ 1
    family = control[1]
    ctx = _PrologueCtx(
        tables=tables,
        hidden=hidden,
        router=router.at[family],
        bias=bias.at[family],
        latent_weight=latent_weight.at[family],
        shared_gu=shared_gu.at[family],
        shared_down=shared_down.at[family],
        shared=shared,
        latent_hbm=latent_hbm,
        token_ids=token_ids,
        probabilities=probabilities,
        counts=counts,
        maximum=maximum,
        rows=rows,
        rank=rank,
        parity=parity,
        peer=peer,
        valid=stripe_validity(rows, control[0], rank),
        counts_ready=counts_ready,
        metadata_ready=metadata_ready,
    )
    vmem, dma = pltpu.VMEM, pltpu.SemaphoreType.DMA

    def body(stage_state: dict[str, object]) -> None:
        """Execute prologue stages inside scoped temporary allocations."""
        peer_input, buffers, gu_sems, hidden_tile, top = _prologue_setup_and_router(
            ctx, stage_state
        )
        state = _PrologueState(
            route_steps=iter(()),
            metadata_steps=iter(()),
            latent_steps=iter(()),
        )
        _prologue_project_latent(ctx, stage_state, state, hidden_tile, top)
        peer_input.wait_send()
        peer_input.wait_recv()
        _prologue_shared_gate_up(ctx, stage_state, state, buffers, gu_sems)
        _prologue_shared_down(ctx, stage_state, state)

    is_small = rows <= _SMALL_WINDOW_MAX_LOCAL_ROWS
    is_max = rows == _MAX_LOCAL_ROWS
    scratch = {
        "pair": vmem((2 * rows, hidden_size), jnp.bfloat16),
        "router": vmem((hidden_size, ROUTED_EXPERTS), jnp.bfloat16)
        if is_small
        else None,
        "router_gu": vmem((hidden_size, 1024), jnp.bfloat16) if is_max else None,
        "gu_buffers": (
            tuple(vmem((hidden_size, _PANEL_COLS), jnp.bfloat16) for _ in range(4))
            if is_small
            else ()
        ),
        "gu_sems": tuple(dma for _ in range(4)),
        "bias": vmem((1, ROUTED_EXPERTS), jnp.float32),
        "input_sem": dma,
        "peer_send": dma,
        "peer_recv": dma,
        "peer_credit": pltpu.SemaphoreType.REGULAR,
        "latent": vmem((rows, _LATENT_PROJ_COLS), jnp.bfloat16) if is_small else None,
        "route_payload": vmem((rows // 4, VREG_LANE_COUNT), jnp.int32),
        "route_wire": vmem((rows * 8, VREG_LANE_COUNT), jnp.int32),
        "route_send": dma((_ROUTE_GATHER_STAGES,)),
        "route_recv": dma((_ROUTE_GATHER_STAGES,)),
        "keys": vmem((rows * 4, VREG_LANE_COUNT), jnp.int32),
        "latent_payload": vmem((rows, 2, _LATENT_HALF_COLS), jnp.bfloat16),
        "latent_recv": (
            vmem((2, rows, 2, _LATENT_HALF_COLS), jnp.bfloat16) if is_small else None
        ),
        "latent_state": pltpu.SMEM((25,), jnp.int32) if is_max else None,
        "latent_meta": (
            pltpu.SMEM((_DIRECTED_RINGS * _RING_META_STRIDE,), jnp.int32)
            if is_max
            else None
        ),
        "latent_send_sems": dma((_RING_STREAMS,)),
        "latent_recv_sems": dma((RING_BUFFERS, _RING_STREAMS)),
        "latent_credits": pltpu.SemaphoreType.REGULAR((_RING_STREAMS,)),
        "latent_stores": dma((_RING_STREAMS,)) if is_small else dma,
        "middle": vmem((2 * rows, _SHARED_INTERMEDIATE_COLS), jnp.bfloat16),
        "down_buffers": tuple(
            vmem((_SHARED_INTERMEDIATE_COLS, _PANEL_COLS), jnp.bfloat16)
            for _ in range(2)
        ),
        "down_sems": (dma, dma),
        "partial": tuple(vmem((2 * rows, _PANEL_COLS), jnp.float32) for _ in range(2)),
        "peer_y": tuple(vmem((rows, _PANEL_COLS), jnp.float32) for _ in range(2)),
        "reduce_send": (dma, dma),
        "reduce_recv": (dma, dma),
    }
    pl.run_scoped(body, scratch)


@dataclasses.dataclass(frozen=True, slots=True)
class _FusedRoutedSpec:
    """Static tiling and schedule parameters for ``_routed_experts``."""

    cfg: k.GmmConfigs
    gu_cfg: k.GmmConfigs
    regions: Sequence[tuple[int, ...]]
    tokens: int
    tm: int
    tn: int
    width: int
    localrows: int
    packed_half: int
    sub: int
    max_gm: int
    unroll: int
    gu_unroll: int
    first_down_unroll: int
    x_dtype: jnp.dtype


@dataclasses.dataclass(frozen=True, slots=True)
class _FusedRoutedCtx:
    """Bound kernel references and configuration for ``_routed_experts``."""

    spec: _FusedRoutedSpec
    tables: PallasRefProtocol
    destinations: _FlatMetadata
    probabilities: _FlatMetadata
    latent: PallasRefProtocol
    middle: PallasRefProtocol
    out: PallasRefProtocol
    metadata: k.MetadataRef
    packed: PallasRefProtocol
    sums: PallasRefProtocol
    cast_buf: PallasRefProtocol
    sends: PallasRefProtocol
    receives: PallasRefProtocol
    credit_sems: PallasRefProtocol
    stores: PallasRefProtocol
    payload: PallasRefProtocol
    rx: PallasRefProtocol
    state: PallasRefProtocol
    ring_meta: PallasRefProtocol
    probabilities_ready: Callable[[], None]


def _build_fused_routed_spec(
    tokens: int,
    latent_dtype: jnp.dtype,
    weights: tuple[
        PallasRefProtocol, PallasRefProtocol, PallasRefProtocol, PallasRefProtocol
    ],
    counts_offset: tuple[PallasRefProtocol, PallasRefProtocol],
) -> _FusedRoutedSpec:
    """Validate shapes and build grouped-matmul configs for ``_routed_experts``."""
    if tokens not in PREFILL_WINDOWS:
        msg = f"Unsupported fused routed-expert token window: {tokens}"
        raise ValueError(msg)
    gu_weights, gu_scales, rhs, scales = weights
    counts, offset = counts_offset
    is_8k = tokens == OVERFLOW_WINDOW_TOKENS
    columns = _COMPACT_DOWN_COLUMNS if is_8k else DEFAULT_DOWN_COLUMNS
    tile = _COMPACT_DOWN_TILE if is_8k else _DEFAULT_DOWN_TILE
    unroll = _COMPACT_DOWN_UNROLL if is_8k else DEFAULT_DOWN_UNROLL
    gu_unroll = 8 if is_8k else 4
    first_down_unroll = 8 if is_8k else 2
    metadata_row_capacity = COMPACT_METADATA_PREFIX_ROWS if is_8k else 0
    x_shape = (tokens * TOP_K, _EXPERT_INTERMEDIATE_COLS)
    rows, width = (metadata_row_capacity or x_shape[0], rhs.shape[-1])
    if rows > x_shape[0] or width % columns != 0:
        msg = f"Invalid routed rows={rows} or width={width} for columns={columns}"
        raise ValueError(msg)
    tm, tn, localrows = (tile, columns, tokens // TP_SIZE)
    packed_half = _DOWN_8K_HALF_TILE_COLS if is_8k else tn // 2
    limit = int(pltpu.get_tpu_info().vmem_capacity_bytes * VMEM_CAPACITY_FRACTION)
    counts_arr = cast("jax.Array", counts)
    offset_arr = cast("jax.Array", offset)
    cfg = k.make_gmm_configs(
        jax.ShapeDtypeStruct((rows, x_shape[1]), latent_dtype),
        jax.ShapeDtypeStruct(rhs.shape[1:], rhs.dtype),
        jax.ShapeDtypeStruct(scales.shape[1:], scales.dtype),
        counts_arr,
        offset_arr,
        tile_info=k.calculate_tiling,
        vmem_limit_bytes=limit,
        out_dtype=None,
        acc_dtype=None,
        fuse_act=None,
        rhs_quant_dtype=jnp.float4_e2m1fn,
    )
    cfg = dataclasses.replace(
        cfg, tiles=dataclasses.replace(cfg.tiles, tile_m=tm, tile_n=tn)
    )
    sub = cfg.dims.size_lhs_sublane
    max_gm = cfg.dims.size_group + pl.cdiv(rows, tm) - 1
    regions = k.regions(localrows, tn)
    gu_cfg = k.make_gmm_configs(
        jax.ShapeDtypeStruct((rows, _LATENT_FULL_COLS), latent_dtype),
        jax.ShapeDtypeStruct(gu_weights.shape[1:], gu_weights.dtype),
        jax.ShapeDtypeStruct(gu_scales.shape[1:], gu_scales.dtype),
        counts_arr,
        offset_arr,
        tile_info=k.calculate_tiling,
        vmem_limit_bytes=limit,
        out_dtype=None,
        acc_dtype=None,
        fuse_act="situ:4:25",
        rhs_quant_dtype=jnp.float4_e2m1fn,
    )
    gu_cfg = dataclasses.replace(
        gu_cfg,
        tiles=dataclasses.replace(
            gu_cfg.tiles,
            tile_m=tm,
            tile_n=_EXPERT_INTERMEDIATE_COLS,
            tile_k=_LATENT_FULL_COLS,
        ),
    )
    if gu_cfg.dims.size_lhs_sublane != sub:
        msg = "Gate/up and down grouped-matmul sublane sizes disagree"
        raise ValueError(msg)
    for gmm_cfg in (cfg, gu_cfg):
        if (
            not gmm_cfg.rhs_cfgs.should_dequantize_before_matmul
            or gmm_cfg.rhs_cfgs.should_dequantize_after_matmul
        ):
            msg = "Fused routed experts require pre-matmul RHS dequantization"
            raise ValueError(msg)
    return _FusedRoutedSpec(
        cfg=cfg,
        gu_cfg=gu_cfg,
        regions=regions,
        tokens=tokens,
        tm=tm,
        tn=tn,
        width=width,
        localrows=localrows,
        packed_half=packed_half,
        sub=sub,
        max_gm=max_gm,
        unroll=unroll,
        gu_unroll=gu_unroll,
        first_down_unroll=first_down_unroll,
        x_dtype=latent_dtype,
    )


def _init_routed_ring_meta(
    tables: PallasRefProtocol,
    ring_meta: PallasRefProtocol,
    localrows: int,
) -> None:
    """Populate six directed-ring neighbor and source-row lookup tables."""
    # Six directed rings: [target, source, 32 local-contribution row bases].
    # One initialization per layer call, reused across all four columns.
    rank = lax.axis_index(AXIS_SP)
    for directed in range(_DIRECTED_RINGS):
        family, direction = directed // 2, -1 if directed % 2 == 0 else 1
        position = tables[family, 0, rank]
        ring_meta[directed * _RING_META_STRIDE] = tables[
            family, 1, (position + direction) & TP_PEERS
        ]
        ring_meta[directed * _RING_META_STRIDE + 1] = tables[
            family, 1, (position - direction) & TP_PEERS
        ]

        @pl.loop(0, TP_SIZE)
        def hop(
            step: jax.Array,
            *,
            family: int = family,
            position: jax.Array = position,
            direction: int = direction,
            directed: int = directed,
        ) -> None:
            """Store local contribution row base for ``step``."""
            owner_mesh = tables[
                family, 1, (position - direction * (step + 1)) & TP_PEERS
            ]
            owner = tables[family, 2, owner_mesh]
            ring_meta[directed * _RING_META_STRIDE + 2 + step] = owner * localrows


def _emit_fused_subtotals(
    ctx: _FusedRoutedCtx,
    subtotal_vmem: PallasRefProtocol,
) -> None:
    """Trim aligned accumulator padding and round completed subtotals to BF16."""
    tn = ctx.spec.tn
    packed_width = 2 * ctx.spec.packed_half

    @pl.loop(0, ctx.spec.tokens // STORE_TILE)
    def block(b: jax.Array) -> None:
        """Convert one 128-token slice of ``sums`` into BF16 ring input."""
        # Trim the aligned-layout padding before the existing ring.
        # The cut at tn=896 is lane aligned, unlike a 448-wide half.
        values = ctx.sums[pl.ds(b * STORE_TILE, STORE_TILE)].reshape(
            STORE_TILE, packed_width
        )
        ctx.cast_buf[...] = values[:, :tn].astype(jnp.bfloat16)
        subtotal_vmem.at[pl.ds(b * STORE_TILE, STORE_TILE)][...] = ctx.cast_buf[...]


def _consume_fused_routed_tile(
    ctx: _FusedRoutedCtx,
    tile_g: jax.Array,
    source: PallasRefProtocol,
    first: int,
    loop_unroll: int,
) -> None:
    """Weight and accumulate remaining rows of ``source`` for group ``tile_g``."""
    offsets = cast("PallasRefProtocol", ctx.metadata.gm_id_to_m_offset)
    start = offsets[tile_g]
    end = offsets[tile_g + 1]
    begin = start // ctx.spec.sub * ctx.spec.sub
    start = jnp.minimum(start + first, end)

    def update(i: jax.Array) -> None:
        """Accumulate single expert row ``i`` into its destination token."""
        dest = ctx.destinations[i]
        weight = ctx.probabilities[i]
        value = source.at[pl.ds(i - begin, 1)][...].astype(jnp.float32)
        target = ctx.sums.at[pl.ds(dest, 1)]
        target[...] = target[...] + value * weight

    full = (end - start) // loop_unroll

    @pl.loop(0, full)
    def chunk(b: jax.Array) -> None:
        """Accumulate one unrolled batch of expert rows."""
        for u in range(loop_unroll):
            update(start + b * loop_unroll + u)

    @pl.loop(start + full * loop_unroll, end)
    def tail(i: jax.Array) -> None:
        """Accumulate remainder expert rows."""
        update(i)


def _advance_fused_ring(
    ctx: _FusedRoutedCtx,
    ng: jax.Array,
    block: jax.Array | int,
    subtotal_vmem: PallasRefProtocol,
    poll_or_drain: tuple[jax.Array | int, bool],
) -> None:
    """Poll or drain the non-blocking 12-stream reduce-scatter ring."""
    poll_round, drain = poll_or_drain
    # Real-prompt tuning: short expert loops need all twelve streams per
    # poll; longer loops alternate six-stream groups to bound polling cost.
    _ready_reduce(
        subtotal_vmem,
        ctx.out.at[:, pl.ds(block * ctx.spec.tn, ctx.spec.tn)],
        ctx.sends,
        ctx.receives,
        ctx.credit_sems,
        ctx.stores,
        ctx.payload,
        ctx.rx,
        streams=ctx.spec.regions,
        buffers=RING_BUFFERS,
        state=ctx.state,
        ring_meta=ctx.ring_meta,
        drain=drain,
        poll_round=poll_round,
        sparse_poll=(ng >= _SPARSE_POLL_MIN_GROUPS),
    )


def _fused_overlap_previous_rows(
    ctx: _FusedRoutedCtx,
    g: jax.Array,
    base_offset: int,
    row_unroll: int,
) -> None:
    """Accumulate ``row_unroll`` rows of the previous expert tile."""
    previous_g = jnp.maximum(g - 1, 0)
    offsets = cast("PallasRefProtocol", ctx.metadata.gm_id_to_m_offset)
    start = offsets[previous_g]
    end = offsets[previous_g + 1]
    begin = start // ctx.spec.sub * ctx.spec.sub
    previous = ctx.packed.at[(g + 1) % 2]
    for row in range(row_unroll):
        offset = base_offset + row
        i = jnp.minimum(start + offset, end - 1)
        dest = ctx.destinations[i]
        prob = ctx.probabilities[i]
        weight = jnp.where((g > 0) & (start + offset < end), prob, 0)
        value = previous.at[pl.ds(i - begin, 1)][...].astype(jnp.float32)
        target = ctx.sums.at[pl.ds(dest, 1)]
        target[...] = target[...] + value * weight


def _finish_fused_later_column(
    ctx: _FusedRoutedCtx,
    ng: jax.Array,
    n: jax.Array | int,
    subtotal_vmem: PallasRefProtocol,
) -> None:
    """Drain previous column ring, emit subtotals, and drain final column."""
    _advance_fused_ring(ctx, ng, n - 1, subtotal_vmem, (0, True))
    _emit_fused_subtotals(ctx, subtotal_vmem)
    ctx.state[0] = 0

    @pl.when(n == ctx.spec.width // ctx.spec.tn - 1)
    def last() -> None:
        """Drain reduce-scatter ring for the last column block."""
        _advance_fused_ring(ctx, ng, n, subtotal_vmem, (0, True))


def _prepare_later_down_tile(
    ctx: _FusedRoutedCtx,
    ng: jax.Array,
    g: jax.Array,
    n: jax.Array | int,
    subtotal_vmem: PallasRefProtocol,
) -> None:
    """Reset accumulators at group 0 and poll previous column reduce-scatter."""

    @pl.when(g == 0)
    def clear() -> None:
        """Zero FP32 token accumulators at column start."""
        ctx.sums[...] = jnp.zeros(ctx.sums.shape, jnp.float32)

    _advance_fused_ring(ctx, ng, n - 1, subtotal_vmem, (g, False))

    @pl.when(g == 0)
    def initialize_previous() -> None:
        """Zero previous ping-pong buffer before group tile 0."""
        previous = ctx.packed.at[1]
        previous[...] = jnp.zeros(previous.shape, previous.dtype)


def _make_fused_down_inner(
    ctx: _FusedRoutedCtx,
    active_ng: jax.Array,
    ng: jax.Array,
    moved_rows: int,
) -> Callable[..., None]:
    """Build the down-projection tile body shared by column 0 and remaining columns."""
    panel_count = pl.cdiv(ctx.spec.tn, pltpu.get_tpu_info().mxu_column_size)

    def inner(
        x: PallasRefProtocol,
        w: k.RhsRef,
        metadata: k.MetadataRef,
        *,
        first: bool = False,
        subtotal_vmem: PallasRefProtocol | None = None,
    ) -> None:
        """Execute one down-projection tile with overlapped row combine and ring."""
        g, n = (pl.program_id(1), 0 if first else pl.program_id(0) + 1)
        current = ctx.packed.at[g % 2]
        previous = ctx.packed.at[(g + 1) % 2]
        previous_g = jnp.maximum(g - 1, 0)
        down_unroll = ctx.spec.first_down_unroll if first else ctx.spec.unroll
        row_prefix = moved_rows if first else 0
        tail_unroll = DEFAULT_DOWN_UNROLL if first else ctx.spec.unroll
        if not first:
            _prepare_later_down_tile(
                ctx, ng, g, n, cast("PallasRefProtocol", subtotal_vmem)
            )

        def overlap_work(panel: int) -> None:
            """Consume ``down_unroll`` rows of the previous expert tile."""
            _fused_overlap_previous_rows(
                ctx, g, row_prefix + panel * down_unroll, down_unroll
            )

        @pl.when(active_ng > 0)
        def product() -> None:
            """Compute packed MXFP4 down-projection tile."""
            _down_inner(
                x,
                w,
                current,
                metadata,
                cfgs=ctx.spec.cfg,
                overlap_work=overlap_work,
            )

        @pl.when(g > 0)
        def previous_tail() -> None:
            """Consume remaining rows of the previous expert tile."""
            _consume_fused_routed_tile(
                ctx,
                previous_g,
                previous,
                row_prefix + panel_count * down_unroll,
                tail_unroll,
            )

        @pl.when(g == ng - 1)
        def finish() -> None:
            """Consume final expert tile and advance column reductions."""
            _consume_fused_routed_tile(ctx, g, current, 0, tail_unroll)
            if not first:
                _finish_fused_later_column(
                    ctx, ng, n, cast("PallasRefProtocol", subtotal_vmem)
                )

    return inner


def _run_skewed_gu_pipeline(
    joint_fn: Callable[..., None],
    ng: jax.Array,
    in_specs: tuple[object, object],
    out_specs: object,
    refs_and_scratches: tuple[tuple[object, ...], tuple[object, ...]],
) -> None:
    """Run ``joint_fn`` with +256-column skewed gate/up buffer stride."""
    refs, scratches = refs_and_scratches
    pipeline, allocate = pltpu.emit_pipeline_with_allocations(
        joint_fn,
        grid=(1, ng, 1),
        in_specs=in_specs,
        out_specs=out_specs,
    )
    allocations = allocate(*refs)
    logical_gu = allocations[0]

    def is_buffer(value: object) -> bool:
        """Check whether ``value`` is a Pallas pipeline ``BufferedRef`` leaf."""
        return isinstance(value, pltpu.BufferedRef)

    def pad(buffer: pltpu.BufferedRef) -> pltpu.BufferedRef:
        """Pad the trailing dimension of ``buffer.window_ref`` by 256 elements."""
        window = cast("pl_core.MemoryRef | None", buffer.window_ref)
        if window is None:
            return buffer
        return dataclasses.replace(
            buffer,
            window_ref=cast(
                "jax.Ref",
                pltpu.VMEM(
                    (*window.shape[:-1], window.shape[-1] + _PANEL_COLS), window.dtype
                ),
            ),
        )

    padded_gu = jax.tree.map(pad, logical_gu, is_leaf=is_buffer)

    def run_pipeline(buffers: tuple[object, ...]) -> None:
        """Slice padded gate/up buffers back to logical width and run pipeline."""

        def trim(
            buffer: pltpu.BufferedRef,
            original: pltpu.BufferedRef,
        ) -> pltpu.BufferedRef:
            """Trim ``buffer.window_ref`` to ``original``'s trailing width."""
            win = cast("PallasRefProtocol | None", buffer.window_ref)
            orig_win = cast("PallasRefProtocol | None", original.window_ref)
            if win is None or orig_win is None:
                return buffer
            return dataclasses.replace(
                buffer,
                window_ref=cast("jax.Ref", win.at[..., pl.ds(0, orig_win.shape[-1])]),
            )

        selected = (
            jax.tree.map(trim, buffers[0], logical_gu, is_leaf=is_buffer),
            *buffers[1:],
        )
        pipeline(*refs, allocations=selected, scratches=scratches)

    pl.run_scoped(run_pipeline, (padded_gu, *allocations[1:]))


def _double_buffer_spec(spec: pl.BlockSpec) -> pl.BlockSpec:
    """Attach 2-buffer pipeline mode to ``spec``."""
    return dataclasses.replace(spec, pipeline_mode=pl.Buffered(buffer_count=2))


def _group_row_bounds(
    ctx: _FusedRoutedCtx,
    g: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return sublane-aligned row start and count for group ``g``."""
    sub = ctx.spec.sub
    offsets = cast("PallasRefProtocol", ctx.metadata.gm_id_to_m_offset)
    begin = offsets[g] // sub * sub
    end = pl.cdiv(offsets[g + 1], sub) * sub
    return (begin, end - begin)


def _start_group_row_gathers(
    ctx: _FusedRoutedCtx,
    g: jax.Array,
    bank: jax.Array,
    acts: PallasRefProtocol,
    act_sems: PallasRefProtocol,
) -> None:
    """Start indexed row gathers for group ``g`` into ``bank``."""
    begin, nrows = _group_row_bounds(ctx, g)

    @pl.loop(0, nrows)
    def row(i: jax.Array) -> None:
        """Start single-row gather for sorted route ``begin + i``."""
        source = ctx.destinations[begin + i]
        pltpu.make_async_copy(
            ctx.latent.at[pl.ds(source, 1)],
            acts.at[bank, pl.ds(i, 1)],
            act_sems.at[bank],
        ).start()


@dataclasses.dataclass(frozen=True, slots=True)
class _JointGateUpCtx:
    """Captured schedule state for the joint gate/up + first down-column body."""

    ctx: _FusedRoutedCtx
    active_ng: jax.Array
    ng: jax.Array
    inner_fn: Callable[..., None]


def _make_fused_joint_body(joint_context: _JointGateUpCtx) -> Callable[..., None]:
    """Build the pipeline body executing fused gate/up and down column 0."""
    ctx = joint_context.ctx
    gu_cfg = ctx.spec.gu_cfg

    def overlap_gu(g: jax.Array, panel: int) -> None:
        """Overlap previous-tile row combine between gate/up panels."""
        _fused_overlap_previous_rows(
            ctx, g, panel * ctx.spec.gu_unroll, ctx.spec.gu_unroll
        )

    def joint(
        gw: k.RhsRef,
        dw: k.RhsRef,
        gy: PallasRefProtocol,
        metadata: k.MetadataRef,
        gu_partial_ref: PallasRefProtocol,
        acts_ref: PallasRefProtocol,
        act_sems_ref: PallasRefProtocol,
    ) -> None:
        """Run one group tile of joint gate/up and first down column."""
        g = pl.program_id(1)
        bank = g % 2

        @pl.when(g == 0)
        def initialize_accumulation() -> None:
            """Drain probability copy and zero accumulators before group 0."""
            # emit_pipeline has already prefetched the first weights.
            # The <=4K probability DMA can overlap metadata and prefetch;
            # drain once here before any expert reads its probability.
            ctx.probabilities_ready()
            ctx.sums[...] = jnp.zeros(ctx.sums.shape, jnp.float32)
            ctx.packed.at[1][...] = jnp.zeros(ctx.packed.at[1].shape, ctx.packed.dtype)

        @pl.when(joint_context.active_ng > 0)
        def gate_up() -> None:
            """Gather activations and evaluate fused gate/up + SiTU."""

            @pl.when(g == 0)
            def initial() -> None:
                """Prefetch group 0 activations."""
                _start_group_row_gathers(ctx, g, bank, acts_ref, act_sems_ref)

            _, nrows = _group_row_bounds(ctx, g)
            active = acts_ref.at[bank, pl.ds(0, nrows)]
            pltpu.make_async_copy(active, active, act_sems_ref.at[bank]).wait()

            @pl.when(g + 1 < joint_context.ng)
            def following() -> None:
                """Prefetch group ``g + 1`` activations into alternate bank."""
                _start_group_row_gathers(ctx, g + 1, 1 - bank, acts_ref, act_sems_ref)

            _gu_inner(
                PackedRows(acts_ref.at[bank]),
                gw,
                gy,
                gu_partial_ref,
                metadata,
                cfgs=gu_cfg,
                overlap_work=functools.partial(overlap_gu, g),
            )

        joint_context.inner_fn(gy, dw, metadata, first=True)

    return joint


def _run_fused_first_column(
    ctx: _FusedRoutedCtx,
    active_ng: jax.Array,
    ng: jax.Array,
    weights_and_specs: tuple[k.FusedWeightsRef, k.WeightsRef, object],
    inner_fn: Callable[..., None],
) -> None:
    """Run the joint gate/up + SiTU + first down-column pipeline."""
    gu, w, ws = weights_and_specs
    sub = ctx.spec.sub
    gu_cfg = ctx.spec.gu_cfg
    (_, gu_ws), gu_ys = k.generate_block_specs(ctx.metadata, gu_cfg)
    fused_gu_ws = k.FusedWeightsRef(gate=gu_ws, up=gu_ws)
    joint = _make_fused_joint_body(_JointGateUpCtx(ctx, active_ng, ng, inner_fn))

    def is_block_spec(val: object) -> bool:
        """Return True when ``val`` is a ``pl.BlockSpec`` leaf."""
        return isinstance(val, pl.BlockSpec)

    joint_gu_ws, joint_down_ws = jax.tree.map(
        _double_buffer_spec, (fused_gu_ws, ws), is_leaf=is_block_spec
    )

    @functools.partial(
        pl.run_scoped,
        gu_partial=pltpu.VMEM((sub, _EXPERT_INTERMEDIATE_COLS), gu_cfg.out_dtype),
        acts=pltpu.VMEM((2, ctx.spec.tm, 2, _LATENT_HALF_COLS), ctx.spec.x_dtype),
        act_sems=pltpu.SemaphoreType.DMA((2,)),
    )
    def first_column(
        gu_partial: PallasRefProtocol,
        acts: PallasRefProtocol,
        act_sems: PallasRefProtocol,
    ) -> None:
        """Execute double-buffered indexed gate/up and column-0 down projection."""
        refs = (gu, w, ctx.middle.reshape(-1, sub, _EXPERT_INTERMEDIATE_COLS))
        scratches = (ctx.metadata, gu_partial, acts, act_sems)
        if ctx.spec.tokens == OVERFLOW_WINDOW_TOKENS:
            # The 8K schedule already fills VMEM; retain its compact buffers.
            pltpu.emit_pipeline(
                joint,
                grid=(1, ng, 1),
                in_specs=(joint_gu_ws, joint_down_ws),
                out_specs=gu_ys,
            )(*refs, scratches=scratches)
            return
        # Skew the physical GU weight/scale row stride to reduce concurrent
        # VMEM load conflicts. Logical tiles and arithmetic stay unchanged.
        _run_skewed_gu_pipeline(
            joint,
            ng,
            (joint_gu_ws, joint_down_ws),
            gu_ys,
            (refs, scratches),
        )


def _routed_experts(
    latent: PallasRefProtocol,
    gu_weights: PallasRefProtocol,
    gu_scales: PallasRefProtocol,
    rhs: PallasRefProtocol,
    scales: PallasRefProtocol,
    counts: PallasRefProtocol,
    destinations: _FlatMetadata,
    probabilities: _FlatMetadata,
    tables: PallasRefProtocol,
    layer: PallasRefProtocol,
    offset: PallasRefProtocol,
    middle: PallasRefProtocol,
    out: PallasRefProtocol,
    *,
    tokens: int,
    probabilities_ready: Callable[[], None],
) -> None:
    """Run joint gate/up/first-down pipeline followed by remaining down columns."""
    spec = _build_fused_routed_spec(
        tokens,
        latent.dtype,
        (gu_weights, gu_scales, rhs, scales),
        (counts, offset),
    )
    streams = len(spec.regions)

    def slice_gu_up(ref: object) -> PallasRefProtocol:
        """Slice the upper 1536 columns of ``ref`` for the fused up projection."""
        return cast("PallasRefProtocol", ref).at[..., _EXPERT_INTERMEDIATE_COLS:]

    def kernel(
        counts_ref: PallasRefProtocol,
        offset_ref: PallasRefProtocol,
        layer_ref: PallasRefProtocol,
        tables_ref: PallasRefProtocol,
        destinations_ref: _FlatMetadata,
        probabilities_ref: _FlatMetadata,
        latent_ref: PallasRefProtocol,
        gu: k.WeightsRef,
        w: k.WeightsRef,
        middle_ref: PallasRefProtocol,
        out_ref: PallasRefProtocol,
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
        ring_meta: PallasRefProtocol,
    ) -> None:
        """Execute fused expert gate/up, down-projection, and 12-stream ring."""
        w_weight = cast("PallasRefProtocol", w.weight)
        w_scale = cast("PallasRefProtocol", w.scale)
        sliced_w = dataclasses.replace(
            w,
            weight=w_weight.at[layer_ref[0]].bitcast(jnp.float4_e2m1fn),
            scale=w_scale.at[layer_ref[0]],
        )
        gu_weight = cast("PallasRefProtocol", gu.weight)
        gu_scale = cast("PallasRefProtocol", gu.scale)
        sliced_gu = dataclasses.replace(
            gu,
            weight=gu_weight.at[layer_ref[0]].bitcast(jnp.float4_e2m1fn),
            scale=gu_scale.at[layer_ref[0]],
        )
        fused_gu = k.FusedWeightsRef(
            gate=sliced_gu, up=jax.tree.map(slice_gu_up, sliced_gu)
        )
        _init_routed_ring_meta(tables_ref, ring_meta, spec.localrows)
        active_ng = k.fill_metadata(
            counts_ref,
            offset_ref,
            metadata,
            cfgs=spec.cfg,
        )

        @pl.when(active_ng == 0)
        def empty_metadata() -> None:
            """Zero initial metadata slots when rank has zero active expert tiles."""
            groups = cast("PallasRefProtocol", metadata.gm_id_to_group_id)
            offsets = cast("PallasRefProtocol", metadata.gm_id_to_m_offset)
            groups[0] = 0
            offsets[0] = 0
            offsets[1] = 0

        ng = jnp.maximum(active_ng, 1)
        (xs, ws), _ = k.generate_block_specs(metadata, spec.cfg)
        state[0] = 0
        ctx = _FusedRoutedCtx(
            spec=spec,
            tables=tables_ref,
            destinations=destinations_ref,
            probabilities=probabilities_ref,
            latent=latent_ref,
            middle=middle_ref,
            out=out_ref,
            metadata=metadata,
            packed=packed,
            sums=sums,
            cast_buf=cast_buf,
            sends=sends,
            receives=receives,
            credit_sems=credit_sems,
            stores=stores,
            payload=payload,
            rx=rx,
            state=state,
            ring_meta=ring_meta,
            probabilities_ready=probabilities_ready,
        )
        gu_panels = pl.cdiv(_FUSED_GU_WIDTH, pltpu.get_tpu_info().mxu_column_size)
        moved_rows = gu_panels * spec.gu_unroll
        inner = _make_fused_down_inner(ctx, active_ng, ng, moved_rows)
        _run_fused_first_column(ctx, active_ng, ng, (fused_gu, sliced_w, ws), inner)

        @functools.partial(
            pl.run_scoped,
            subtotal_vmem=pltpu.VMEM((tokens, spec.tn), spec.x_dtype),
        )
        def remaining_columns(subtotal_vmem: PallasRefProtocol) -> None:
            """Emit column-0 subtotals and pipeline remaining down columns."""
            _emit_fused_subtotals(ctx, subtotal_vmem)
            state[0] = 0

            def shifted(block_spec: pl.BlockSpec) -> pl.BlockSpec:
                """Advance column block index by 1 past the fused first column."""
                index_map = cast("Callable[..., object]", block_spec.index_map)
                return dataclasses.replace(
                    block_spec, index_map=lambda n, g, k_idx: index_map(n + 1, g, k_idx)
                )

            shifted_ws = jax.tree.map(
                shifted, ws, is_leaf=lambda v: isinstance(v, pl.BlockSpec)
            )

            def remaining(
                x_tile: PallasRefProtocol,
                w_tile: k.RhsRef,
                metadata_tile: k.MetadataRef,
            ) -> None:
                """Run one remaining down-column tile."""
                inner(x_tile, w_tile, metadata_tile, subtotal_vmem=subtotal_vmem)

            pltpu.emit_pipeline(
                remaining,
                grid=(spec.width // spec.tn - 1, ng, 1),
                in_specs=(xs, shifted_ws),
                out_specs=(),
            )(
                middle_ref.reshape(-1, spec.sub, _EXPERT_INTERMEDIATE_COLS),
                sliced_w,
                scratches=(metadata,),
            )

    def run(scratch_refs: dict[str, object]) -> None:
        """Unpack scoped scratch allocations and launch ``kernel``."""
        kernel(
            counts,
            offset,
            layer,
            tables,
            destinations,
            probabilities,
            latent,
            k.WeightsRef(weight=gu_weights, scale=gu_scales),
            k.WeightsRef(weight=rhs, scale=scales),
            middle,
            out,
            cast("k.MetadataRef", scratch_refs["metadata"]),
            cast("PallasRefProtocol", scratch_refs["packed"]),
            cast("PallasRefProtocol", scratch_refs["sums"]),
            cast("PallasRefProtocol", scratch_refs["cast"]),
            cast("PallasRefProtocol", scratch_refs["sends"]),
            cast("PallasRefProtocol", scratch_refs["receives"]),
            cast("PallasRefProtocol", scratch_refs["credits"]),
            cast("PallasRefProtocol", scratch_refs["stores"]),
            cast("PallasRefProtocol", scratch_refs["payload"]),
            cast("PallasRefProtocol", scratch_refs["rx"]),
            cast("PallasRefProtocol", scratch_refs["state"]),
            cast("PallasRefProtocol", scratch_refs["ring_meta"]),
        )

    ring_dtype = jnp.float32
    pl.run_scoped(
        run,
        {
            "metadata": k.MetadataRef(
                gm_id_to_group_id=pltpu.SMEM((spec.max_gm,), jnp.int32),
                gm_id_to_m_offset=pltpu.SMEM((spec.max_gm + 1,), jnp.int32),
            ),
            "packed": pltpu.VMEM((2, spec.tm, 2, spec.packed_half), spec.x_dtype),
            "sums": pltpu.VMEM((tokens, 2, spec.packed_half), jnp.float32),
            "cast": pltpu.VMEM((STORE_TILE, spec.tn), spec.x_dtype),
            "sends": pltpu.SemaphoreType.DMA((streams,)),
            "receives": pltpu.SemaphoreType.DMA((RING_BUFFERS, streams)),
            "credits": pltpu.SemaphoreType.REGULAR((streams,)),
            "stores": pltpu.SemaphoreType.DMA,
            "payload": pltpu.VMEM((spec.localrows, spec.tn), ring_dtype),
            "rx": pltpu.VMEM((RING_BUFFERS, spec.localrows, spec.tn), ring_dtype),
            "state": pltpu.SMEM((1 + 2 * streams,), jnp.int32),
            "ring_meta": pltpu.SMEM((_DIRECTED_RINGS * _RING_META_STRIDE,), jnp.int32),
        },
    )


def _column_rhs(
    rhs: k.RhsRef,
    first: int,
    size: int,
) -> k.RhsRef:
    """Slice weight and scale references along columns before dequantization."""

    def slice_ref(ref: k.WeightsRef, col_first: int, col_size: int) -> k.WeightsRef:
        """Slice one ``WeightsRef`` leaf along ``[col_first, col_first + col_size)``."""
        if ref.scale is None:
            msg = "Routed MXFP4 weights require non-None scale reference"
            raise ValueError(msg)
        weight_ref = cast("PallasRefProtocol", ref.weight)
        scale_ref = cast("PallasRefProtocol", ref.scale)
        return dataclasses.replace(
            ref,
            weight=weight_ref.at[:, pl.ds(col_first, col_size)],
            scale=scale_ref.at[..., pl.ds(col_first, col_size)],
        )

    if isinstance(rhs, k.FusedWeightsRef):
        # Gate/up columns alternate in lane-sized chunks. A complete pair
        # maps to half as many columns in each underlying reference.
        pair = 2 * pltpu.get_tpu_info().num_lanes
        if first % pair != 0 or size % pair != 0:
            msg = f"Fused gate/up slice ({first}, {size}) must align to {pair}"
            raise ValueError(msg)
        return k.FusedWeightsRef(
            slice_ref(rhs.gate, first // 2, size // 2),
            slice_ref(rhs.up, first // 2, size // 2),
        )
    return slice_ref(cast("k.WeightsRef", rhs), first, size)


def _expert_product(
    lhs: PackedRows | PallasRefProtocol,
    rhs: k.RhsRef,
    *,
    cfgs: k.GmmConfigs,
    bucket_rows: int,
    overlap_work: Callable[[int], None],
) -> jax.Array:
    """Dequantize MXFP4 weights, panel-locally for the 8K row tile.

    Both routed projections cover their complete reduction dimension in one
    tile. The callback consumes rows of the previous expert tile between MXU
    panels; it must not read this product's result.
    """
    if cfgs.dims.size_k != cfgs.tiles.tile_k:
        msg = "Fused expert product requires full-K single-tile contraction"
        raise ValueError(msg)
    columns = pltpu.get_tpu_info().mxu_column_size
    activations = lhs.reshape(-1, cfgs.tiles.tile_k)[:bucket_rows]
    width = (
        2 * cast("PallasRefProtocol", rhs.gate.weight).shape[-1]
        if isinstance(rhs, k.FusedWeightsRef)
        else cast("PallasRefProtocol", cast("k.WeightsRef", rhs).weight).shape[-1]
    )
    block = cfgs.rhs_cfgs.quant_block_size
    if block is None:
        msg = "Fused expert product requires quantized block size"
        raise ValueError(msg)
    # The 8K window uses a 256-row tile. Smaller windows use 128 rows;
    # their whole-tile conversion schedules better than panel-local loads.
    panel_local = cfgs.tiles.tile_m == _COMPACT_DOWN_TILE
    weights: jax.Array | None = None
    if not panel_local:
        raw_weights = rhs.get_weight()
        scales = rhs.get_scale(replicate_size=block).astype(cfgs.lhs_cfgs.dtype)
        reshaped = raw_weights.astype(cfgs.lhs_cfgs.dtype).reshape(
            cfgs.num_quant_blocks_per_tile_k, block, width
        )
        weights = (reshaped * scales).reshape(cfgs.tiles.tile_k, width)
    panels = []
    for first in range(0, width, columns):
        panel_width = min(columns, width - first)
        if panel_local:
            panel_rhs = _column_rhs(rhs, first, panel_width)
            panel_weights = (
                panel_rhs.get_weight()
                .astype(cfgs.lhs_cfgs.dtype)
                .reshape(cfgs.num_quant_blocks_per_tile_k, block, panel_width)
            )
            scales = panel_rhs.get_scale(replicate_size=block).astype(
                cfgs.lhs_cfgs.dtype
            )
            panel_weights = (panel_weights * scales).reshape(
                cfgs.tiles.tile_k, panel_width
            )
        else:
            if weights is None:
                msg = "Expected pre-dequantized full-tile weights"
                raise ValueError(msg)
            panel_weights = weights[:, first : first + panel_width]
        accumulator = jnp.zeros((bucket_rows, panel_width), cfgs.acc_dtype)
        product = jnp.matmul(
            activations, panel_weights, preferred_element_type=jnp.float32
        ).astype(cfgs.acc_dtype)
        overlap_work(first // columns)
        accumulator += product
        panels.append(accumulator)
    return jnp.concatenate(panels, axis=1)


def _gu_inner(
    lhs: PackedRows,
    rhs: k.RhsRef,
    output: PallasRefProtocol,
    partial: PallasRefProtocol,
    metadata: k.MetadataRef,
    *,
    cfgs: k.GmmConfigs,
    overlap_work: Callable[[int], None],
) -> None:
    """Compute GU/SiTU with the production row buckets and adjacent-expert carry.

    Expert boundaries need not align with the 16-row BF16 sublane. Mask outside
    the active expert, then preserve the preceding expert's partial first row.
    """
    group = pl.program_id(1)
    offsets = cast("PallasRefProtocol", metadata.gm_id_to_m_offset)
    start = offsets[group]
    end = offsets[group + 1]
    aligned = start - start % cfgs.dims.size_lhs_sublane
    first, last = start - aligned, end - aligned

    def product(bucket_index: int) -> None:
        """Evaluate ``bucket_index`` row bucket and merge sublane boundary carry."""
        rows = cfgs.tiles.bucket_base * (bucket_index + 1)
        value = _expert_product(
            lhs, rhs, cfgs=cfgs, bucket_rows=rows, overlap_work=overlap_work
        )
        value = k.apply_act_fn(value, cfgs.fuse_act)
        row = lax.broadcasted_iota(jnp.int32, value.shape, 0)
        value = jnp.where((first <= row) & (row < last), value, 0)
        output.reshape(-1, cfgs.tiles.tile_n)[:rows] = value.astype(output.dtype)
        zeros = jnp.zeros(partial.shape, partial.dtype)
        output[0] = output[0] + jnp.where(group == 0, zeros, partial[...])
        last_row = last // cfgs.dims.size_lhs_sublane
        partial[...] = jnp.where(
            last % cfgs.dims.size_lhs_sublane == 0, zeros, output[last_row]
        )

    # At 8K, share the 128-row product for both small buckets.
    # Keep route order, trading padded work for fewer instruction overlays.
    first_bucket = 1 if cfgs.tiles.tile_m == _COMPACT_DOWN_TILE else 0
    branches = [
        functools.partial(product, bucket_index=i)
        for i in range(first_bucket, cfgs.tiles.tile_m // cfgs.tiles.bucket_base)
    ]
    lax.switch(jnp.maximum(last // cfgs.tiles.bucket_base - first_bucket, 0), branches)


def _down_inner(
    lhs: PallasRefProtocol,
    rhs: k.RhsRef,
    output: PallasRefProtocol,
    metadata: k.MetadataRef,
    *,
    cfgs: k.GmmConfigs,
    overlap_work: Callable[[int], None],
) -> None:
    """Compute down projection stored as packed rows for token-wise VALU combine.

    Each valid expert row is consumed individually, so this output needs no
    adjacent-expert carry. BF16 rounding precedes FP32 routing weighting.
    """
    if cfgs.fuse_act is not None:
        msg = "Down inner kernel does not support fused activation"
        raise ValueError(msg)
    group = pl.program_id(1)
    offsets = cast("PallasRefProtocol", metadata.gm_id_to_m_offset)
    start = offsets[group]
    end = offsets[group + 1]
    aligned = start - start % cfgs.dims.size_lhs_sublane
    last = end - aligned

    # GU already masks inactive rows; consume() visits only live routes.
    def product(bucket_index: int) -> None:
        """Evaluate ``bucket_index`` row bucket and store packed BF16 rows."""
        rows = cfgs.tiles.bucket_base * (bucket_index + 1)
        with jax.named_scope(f"bm{rows}_fragment"):
            value = _expert_product(
                lhs, rhs, cfgs=cfgs, bucket_rows=rows, overlap_work=overlap_work
            )
            if cfgs.tiles.tile_n == _COMPACT_DOWN_COLUMNS:
                # Keep the half boundary on a complete 128-lane block.
                # Only the final 128 elements are padding; the consumer trims
                # them once per subtotal before communicating the 896 columns.
                cast_val = value.astype(output.dtype)
                padded = jnp.concatenate(
                    (cast_val, jnp.zeros((rows, VREG_LANE_COUNT), output.dtype)),
                    axis=1,
                )
                output[:rows] = padded.reshape(rows, 2, _DOWN_8K_HALF_TILE_COLS)
            else:
                output[:rows] = value.astype(output.dtype).reshape(
                    rows, 2, cfgs.tiles.tile_n // 2
                )

    branches = [
        functools.partial(product, bucket_index=i)
        for i in range(cfgs.tiles.tile_m // cfgs.tiles.bucket_base)
    ]
    lax.switch(last // cfgs.tiles.bucket_base, branches)


@dataclasses.dataclass(frozen=True, slots=True)
class _ReadyReduceCtx:
    """References and stream metadata for ``_ready_reduce``."""

    x_ref: PallasRefProtocol
    o_ref: PallasRefProtocol
    send_sems: PallasRefProtocol
    recv_sems: PallasRefProtocol
    credit_sems: PallasRefProtocol
    payload: PallasRefProtocol
    received: PallasRefProtocol
    streams: Sequence[tuple[int, ...]]
    buffers: int
    state: PallasRefProtocol
    ring_meta: PallasRefProtocol

    def remote(
        self,
        stream: int,
        slices: tuple[pl.Slice, pl.Slice],
        target: tuple[jax.Array],
        bank: jax.Array | int,
    ) -> AsyncCopyProtocol:
        """Construct mesh-indexed async remote copy for ``stream``."""
        rows, cols = slices
        return cast(
            "AsyncCopyProtocol",
            pltpu.make_async_remote_copy(
                self.payload.at[rows, cols],
                self.received.at[bank, rows, cols],
                self.send_sems.at[stream],
                self.recv_sems.at[bank, stream],
                device_id=target,
                device_id_type=pl.DeviceIdType.MESH,
            ),
        )


def _init_ready_reduce(ctx: _ReadyReduceCtx) -> None:
    """Barrier-synchronize ring neighbors and start hop-0 remote transfers."""
    count = len(ctx.streams)
    peers = [(ctx.ring_meta[d * _RING_META_STRIDE],) for d in range(_DIRECTED_RINGS)]
    sem = pltpu.get_barrier_semaphore()
    for peer in peers:
        pl.semaphore_signal(sem, 1, device_id=peer, device_id_type=pl.DeviceIdType.MESH)
    pl.semaphore_wait(sem, len(peers))

    @functools.partial(pl.run_scoped, ready=pltpu.SemaphoreType.REGULAR)
    def initialized(ready: PallasRefProtocol) -> None:
        """Confirm all ring peers have exited the entry barrier."""
        for peer in peers:
            pl.semaphore_signal(
                ready, 1, device_id=peer, device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(ready, len(peers))

    for stream, region in enumerate(ctx.streams):
        begin, size, col_begin, width = region
        base = (stream % _DIRECTED_RINGS) * _RING_META_STRIDE
        rows, cols = pl.ds(begin, size), pl.ds(col_begin, width)
        source_rows = pl.ds(
            pl.multiple_of(ctx.ring_meta[base + 2], ctx.o_ref.shape[0]) + begin, size
        )
        target = (ctx.ring_meta[base],)
        ctx.payload.at[rows, cols][...] = ctx.x_ref.at[source_rows, cols][...].astype(
            ctx.payload.dtype
        )
        ctx.remote(stream, (rows, cols), target, jnp.int32(0)).start()
        ctx.state[1 + stream] = 1
        ctx.state[1 + count + stream] = 0
    ctx.state[0] = count + 1


def _step_ready_reduce_stream(
    ctx: _ReadyReduceCtx,
    stream: int,
    region: tuple[int, ...],
) -> None:
    """Poll and advance a single stream of the non-blocking reduce-scatter ring."""
    count = len(ctx.streams)
    begin, size, col_begin, width = region
    base = (stream % _DIRECTED_RINGS) * _RING_META_STRIDE
    rows, cols = pl.ds(begin, size), pl.ds(col_begin, width)
    # TPUv7 DMA semaphore accounting: one unit per 32 logical bytes.
    units = size * width * 4 // 32
    step = ctx.state[1 + stream]
    phase_slot = 1 + count + stream

    @pl.when(ctx.state[phase_slot] == 0)
    def receive() -> None:
        """Accumulate received peer shard once completion counters reach ``units``."""
        bank = (step - 1) & 1
        ready = (pl.semaphore_read(ctx.send_sems.at[stream]) >= units) & (
            pl.semaphore_read(ctx.recv_sems.at[bank, stream]) >= units
        )

        @pl.when(ready)
        def reduce() -> None:
            """Wait on completed copy, add local shard, and return credit."""
            target = (ctx.ring_meta[base],)
            ctx.remote(stream, (rows, cols), target, bank).wait()
            source_rows = pl.ds(
                pl.multiple_of(ctx.ring_meta[base + 2 + step], ctx.o_ref.shape[0])
                + begin,
                size,
            )
            ctx.payload.at[rows, cols][...] = (
                ctx.x_ref.at[source_rows, cols][...].astype(ctx.payload.dtype)
                + ctx.received.at[bank, rows, cols][...]
            )
            pl.semaphore_signal(
                ctx.credit_sems.at[stream],
                1,
                device_id=(ctx.ring_meta[base + 1],),
                device_id_type=pl.DeviceIdType.MESH,
            )
            ctx.state[phase_slot] = jnp.where(step == TP_PEERS, _RING_PHASE_DONE, 1)

    @pl.when(ctx.state[phase_slot] == 1)
    def forward() -> None:
        """Send accumulated shard to downstream peer once credit is available."""
        available = (step < ctx.buffers) | (
            pl.semaphore_read(ctx.credit_sems.at[stream]) >= 1
        )

        @pl.when(available)
        def send() -> None:
            """Acquire credit when needed and start remote copy."""

            @pl.when(step >= ctx.buffers)
            def acquire() -> None:
                """Consume one downstream buffer credit."""
                pl.semaphore_wait(ctx.credit_sems.at[stream], 1)

            ctx.remote(stream, (rows, cols), (ctx.ring_meta[base],), step & 1).start()
            ctx.state[1 + stream] = step + 1
            ctx.state[phase_slot] = 0

    @pl.when(ctx.state[phase_slot] == _RING_PHASE_DONE)
    def finish() -> None:
        """Drain remaining credits after all 31 hops complete."""

        @pl.when(pl.semaphore_read(ctx.credit_sems.at[stream]) >= ctx.buffers)
        def release() -> None:
            """Wait for final credits and mark stream finished."""
            # Stay outside the inner expert loop so both alternating receive
            # semaphores drain cleanly before peers reuse the ring buffers.
            pl.semaphore_wait(ctx.credit_sems.at[stream], ctx.buffers)
            ctx.state[phase_slot] = _RING_PHASE_RELEASED
            ctx.state[0] = ctx.state[0] - 1


def _sweep_ready_reduce(
    ctx: _ReadyReduceCtx,
    selected: int | None = None,
) -> None:
    """Poll all twelve reduce-scatter streams or one 6-stream half-group."""
    for stream, region in enumerate(ctx.streams):
        if selected is not None and (stream // _DIRECTED_RINGS) != selected:
            continue
        _step_ready_reduce_stream(ctx, stream, region)


def _drain_ready_reduce(
    ctx: _ReadyReduceCtx,
    store_sem: PallasRefProtocol,
) -> None:
    """Spin until all streams finish credit release and store reduced payload."""

    def pending(_i: jax.Array) -> jax.Array:
        """Return True while any stream has not yet released final credits."""
        return ctx.state[0] != 1

    def iterate(i: jax.Array) -> jax.Array:
        """Poll all twelve streams once inside drain loop."""
        _sweep_ready_reduce(ctx)
        return i + 1

    jax.lax.while_loop(pending, iterate, jnp.int32(0))
    output = pltpu.make_async_copy(ctx.payload, ctx.o_ref, store_sem)
    output.start()
    output.wait()


def _ready_reduce(
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
    state: PallasRefProtocol,
    ring_meta: PallasRefProtocol,
    drain: bool = False,
    poll_round: jax.Array | int = 0,
    sparse_poll: jax.Array | bool = True,
) -> None:
    """Run the 12-stream protocol with cached descriptors and cheap failed polls.

    ring_meta has six directed rings, each with neighbor mesh IDs followed by
    32 source-row bases. SMEM offsets, not physical VMEM pointers, are cached.
    Neither a failed receive nor a failed credit poll needs a payload address.
    """
    if buffers != RING_BUFFERS:
        msg = f"_ready_reduce requires {RING_BUFFERS} ring buffers, got {buffers}"
        raise ValueError(msg)
    ctx = _ReadyReduceCtx(
        x_ref=x_ref,
        o_ref=o_ref,
        send_sems=send_sems,
        recv_sems=recv_sems,
        credit_sems=credit_sems,
        payload=payload,
        received=received,
        streams=streams,
        buffers=buffers,
        state=state,
        ring_meta=ring_meta,
    )

    @pl.when(state[0] == 0)
    def initialize() -> None:
        """Initialize ring barrier and hop-0 transfers on first invocation."""
        _init_ready_reduce(ctx)

    if drain:
        _drain_ready_reduce(ctx, store_sem)
        return

    @pl.when(state[0] != 1)
    def progress() -> None:
        """Poll alternating 6-stream groups or all streams when ``~sparse_poll``."""
        # Alternate two six-stream groups only for long expert loops.
        for group in range(2):

            @pl.when(((poll_round & 1) == group) | ~jnp.asarray(sparse_poll))
            def selected(*, group: int = group) -> None:
                """Sweep stream group ``group``."""
                _sweep_ready_reduce(ctx, group)


def _epilogue(
    control: PallasRefProtocol,
    reduced: PallasRefProtocol,
    shared: PallasRefProtocol,
    norm_weight: PallasRefProtocol,
    up_weight: PallasRefProtocol,
    prefix: PallasRefProtocol,
    output: PallasRefProtocol,
) -> None:
    """Normalize routed output and project up with three weight/prefix banks.

    Reduced FP32 rows remain in VMEM. Preserve the BF16 boundaries after
    normalization, after latent-up, and after adding the shared expert.
    """
    rows, width = reduced.shape
    family = control[1]
    norm_slice = norm_weight.at[family, :, :width]
    up_slice = up_weight.at[family]
    valid = stripe_validity(rows, control[0], jax.lax.axis_index(AXIS_SP))
    vmem, dma = pltpu.VMEM, pltpu.SemaphoreType.DMA

    def body(stage_state: dict[str, object]) -> None:
        """Stream 512-column latent-up panels across triple-buffered banks."""
        norm_buf = cast("PallasRefProtocol", stage_state["norm"])
        norm_sem = cast("PallasRefProtocol", stage_state["norm_sem"])
        normalized = cast("PallasRefProtocol", stage_state["normalized"])
        weights_bufs = cast("tuple[PallasRefProtocol, ...]", stage_state["weights"])
        tiles_bufs = cast("tuple[PallasRefProtocol, ...]", stage_state["tiles"])
        weight_sems = cast("tuple[PallasRefProtocol, ...]", stage_state["weight_sems"])
        prefix_sems = cast("tuple[PallasRefProtocol, ...]", stage_state["prefix_sems"])
        output_sems = cast("tuple[PallasRefProtocol, ...]", stage_state["output_sems"])

        copy = pltpu.make_async_copy(norm_slice, norm_buf, norm_sem)
        copy.start()
        copy.wait()
        panels = HIDDEN_SIZE // _EPILOGUE_PANEL_COLS

        def weight_copy(panel: int) -> AsyncCopyProtocol:
            """Create async copy for latent-up weight ``panel``."""
            return cast(
                "AsyncCopyProtocol",
                pltpu.make_async_copy(
                    up_slice.at[
                        :, pl.ds(panel * _EPILOGUE_PANEL_COLS, _EPILOGUE_PANEL_COLS)
                    ],
                    weights_bufs[panel % _EPILOGUE_BANKS],
                    weight_sems[panel % _EPILOGUE_BANKS],
                ),
            )

        def prefix_copy(panel: int) -> AsyncCopyProtocol:
            """Create async copy for residual prefix ``panel``."""
            return cast(
                "AsyncCopyProtocol",
                pltpu.make_async_copy(
                    prefix.at[
                        :, pl.ds(panel * _EPILOGUE_PANEL_COLS, _EPILOGUE_PANEL_COLS)
                    ],
                    tiles_bufs[panel % _EPILOGUE_BANKS],
                    prefix_sems[panel % _EPILOGUE_BANKS],
                ),
            )

        def output_copy(panel: int) -> AsyncCopyProtocol:
            """Create async copy storing completed output ``panel``."""
            return cast(
                "AsyncCopyProtocol",
                pltpu.make_async_copy(
                    tiles_bufs[panel % _EPILOGUE_BANKS],
                    output.at[
                        :, pl.ds(panel * _EPILOGUE_PANEL_COLS, _EPILOGUE_PANEL_COLS)
                    ],
                    output_sems[panel % _EPILOGUE_BANKS],
                ),
            )

        for panel in (0, 1):
            weight_copy(panel).start()
            prefix_copy(panel).start()
        routed = reduced[...]
        mean = jnp.sum(routed * routed, axis=1, keepdims=True) / width
        inverse = jax.lax.rsqrt(mean + _RMS_EPS)
        normalized[...] = ((routed * inverse) * norm_buf[...]).astype(jnp.bfloat16)
        for panel in range(panels):
            bank = panel % _EPILOGUE_BANKS
            weight_copy(panel).wait()
            prefix_copy(panel).wait()
            if panel + 2 < panels:
                if panel >= 1:
                    output_copy(panel - 1).wait()
                weight_copy(panel + 2).start()
                prefix_copy(panel + 2).start()
            product = jnp.matmul(
                normalized[...],
                weights_bufs[bank][...],
                preferred_element_type=jnp.float32,
            ).astype(jnp.bfloat16)
            col_slice = pl.ds(panel * _EPILOGUE_PANEL_COLS, _EPILOGUE_PANEL_COLS)
            value = product + shared[:, col_slice]
            value = jnp.where(valid, value, 0)
            tiles_bufs[bank][...] = tiles_bufs[bank][...] + value
            output_copy(panel).start()
        for panel in range(panels - _EPILOGUE_BANKS, panels):
            output_copy(panel).wait()

    pl.run_scoped(
        body,
        {
            "norm": vmem((1, width), jnp.float32),
            "norm_sem": dma,
            "normalized": vmem((rows, width), jnp.bfloat16),
            "weights": tuple(
                vmem((width, _EPILOGUE_PANEL_COLS), jnp.bfloat16)
                for _ in range(_EPILOGUE_BANKS)
            ),
            "tiles": tuple(
                vmem((rows, _EPILOGUE_PANEL_COLS), jnp.bfloat16)
                for _ in range(_EPILOGUE_BANKS)
            ),
            "weight_sems": (dma, dma, dma),
            "prefix_sems": (dma, dma, dma),
            "output_sems": (dma, dma, dma),
        },
    )


class _FlatMetadata:
    """Scalar expert-row access to the 2D SMEM layout used by metadata DMA.

    Keeping DMA source and destination two-dimensional avoids an unsupported
    1D VMEM memref reshape. Only scalar reads are needed by expert pipelines.
    """

    def __init__(self, ref: PallasRefProtocol) -> None:
        """Reshape ``ref`` into a single-row 2D view for scalar indexing."""
        self.ref = ref.reshape(1, ref.shape[0] * ref.shape[1])

    def __getitem__(self, index: jax.Array | int) -> jax.Array:
        """Read scalar element at ``index``."""
        return self.ref[0, index]


@dataclasses.dataclass(frozen=True, slots=True)
class _LayerKernelInputs:
    """Grouped HBM references and window parameters for ``_layer_kernel``."""

    tables: PallasRefProtocol
    control: PallasRefProtocol
    hidden: PallasRefProtocol
    router: PallasRefProtocol
    bias: PallasRefProtocol
    latent_down: PallasRefProtocol
    shared_gu: PallasRefProtocol
    shared_down: PallasRefProtocol
    latent: PallasRefProtocol
    fallbacks: tuple[
        PallasRefProtocol | None,
        PallasRefProtocol | None,
        PallasRefProtocol | None,
        PallasRefProtocol | None,
    ]
    maximum: PallasRefProtocol | None
    tokens: int
    capacity: int


def _export_layer_overflow(
    layer_inputs: _LayerKernelInputs,
    stage_state: dict[str, object],
    ids_v: PallasRefProtocol,
    probs_v: PallasRefProtocol,
) -> None:
    """Spill full routing metadata and shared expert output when 8K overflows."""
    max_smem = cast("PallasRefProtocol", stage_state["maximum"])
    export_sem = cast("PallasRefProtocol", stage_state["export_sem"])
    counts_v = cast("PallasRefProtocol", stage_state["counts_v"])
    shared_v = cast("PallasRefProtocol", stage_state["shared"])
    fb_ids, fb_probs, fb_counts, fb_shared = layer_inputs.fallbacks

    @pl.when(max_smem[0] > layer_inputs.capacity)
    def export_overflow() -> None:
        """Copy overflow buffers back to HBM."""
        for source, dest in (
            (ids_v, fb_ids),
            (probs_v, fb_probs),
            (counts_v, fb_counts),
            (shared_v, fb_shared),
        ):
            copy = pltpu.make_async_copy(
                source, cast("PallasRefProtocol", dest), export_sem
            )
            copy.start()
            copy.wait()


def _prepare_layer_prologue(
    layer_inputs: _LayerKernelInputs,
    stage_state: dict[str, object],
    ids_v: PallasRefProtocol,
    probs_v: PallasRefProtocol,
) -> None:
    """Run prologue projections and export fallback buffers if 8K overflows."""
    counts_v = cast("PallasRefProtocol", stage_state["counts_v"])
    counts_smem = cast("PallasRefProtocol", stage_state["counts"])
    count_sem = cast("PallasRefProtocol", stage_state["count_sem"])
    probs_smem = cast("PallasRefProtocol", stage_state["probs"])
    prob_sem = cast("PallasRefProtocol", stage_state["prob_sem"])
    ids_smem = cast("PallasRefProtocol", stage_state["ids"])
    index_sem = cast("PallasRefProtocol", stage_state["index_sem"])
    shared_v = cast("PallasRefProtocol", stage_state["shared"])
    is_8k = layer_inputs.tokens == OVERFLOW_WINDOW_TOKENS

    def count_copy() -> AsyncCopyProtocol:
        """Create copy moving expert route counts to scalar scratch."""
        return cast(
            "AsyncCopyProtocol",
            pltpu.make_async_copy(counts_v, counts_smem, count_sem),
        )

    def maximum_copy() -> AsyncCopyProtocol:
        """Create copy moving 8K global maximum count to scalar scratch."""
        return cast(
            "AsyncCopyProtocol",
            pltpu.make_async_copy(
                cast("PallasRefProtocol", layer_inputs.maximum),
                cast("PallasRefProtocol", stage_state["maximum"]),
                cast("PallasRefProtocol", stage_state["max_sem"]),
            ),
        )

    def counts_ready() -> None:
        """Start expert count and optional 8K maximum copies."""
        count_copy().start()
        if is_8k:
            maximum_copy().start()

    def probability_copy() -> AsyncCopyProtocol:
        """Create copy moving sorted probabilities prefix to scalar scratch."""
        return cast(
            "AsyncCopyProtocol",
            pltpu.make_async_copy(
                probs_v.at[: layer_inputs.capacity // VREG_LANE_COUNT, :],
                probs_smem,
                prob_sem,
            ),
        )

    indices = pltpu.make_async_copy(
        ids_v.at[: layer_inputs.capacity // VREG_LANE_COUNT, :], ids_smem, index_sem
    )

    def metadata_ready() -> None:
        """Start sorted token ID and probability prefix copies."""
        indices.start()
        probability_copy().start()

    _prologue(
        layer_inputs.tables,
        layer_inputs.control,
        layer_inputs.hidden,
        layer_inputs.router,
        layer_inputs.bias,
        layer_inputs.latent_down,
        layer_inputs.shared_gu,
        layer_inputs.shared_down,
        shared_v,
        layer_inputs.latent,
        ids_v,
        probs_v,
        counts_v,
        layer_inputs.maximum,
        counts_ready=counts_ready,
        metadata_ready=metadata_ready,
    )
    count_copy().wait()
    if is_8k:
        maximum_copy().wait()
        _export_layer_overflow(layer_inputs, stage_state, ids_v, probs_v)
    else:
        metadata_ready()
    indices.wait()
    if is_8k:
        probability_copy().wait()


def _layer_kernel(
    tables: PallasRefProtocol,
    control: PallasRefProtocol,
    hidden: PallasRefProtocol,
    router: PallasRefProtocol,
    bias: PallasRefProtocol,
    latent_down: PallasRefProtocol,
    shared_gu: PallasRefProtocol,
    shared_down: PallasRefProtocol,
    expert_gu: PallasRefProtocol,
    gu_scales: PallasRefProtocol,
    expert_down: PallasRefProtocol,
    down_scales: PallasRefProtocol,
    norm: PallasRefProtocol,
    latent_up: PallasRefProtocol,
    prefix: PallasRefProtocol,
    output: PallasRefProtocol,
    latent: PallasRefProtocol,
    middle: PallasRefProtocol,
    fallback_ids: PallasRefProtocol | None = None,
    fallback_probs: PallasRefProtocol | None = None,
    fallback_counts: PallasRefProtocol | None = None,
    fallback_shared: PallasRefProtocol | None = None,
    maximum: PallasRefProtocol | None = None,
) -> None:
    """Execute all compute, metadata, and communication on the fast path.

    Large projection scratch has a shorter lifetime than expert scratch. Only
    shared output and the scalar routing metadata survive that boundary. At 8K
    all ranks take the same overflow branch; no routes are dropped to fit SMEM.
    """
    rows = hidden.shape[0]
    tokens = rows * TP_SIZE
    is_8k = tokens == OVERFLOW_WINDOW_TOKENS
    capacity = COMPACT_METADATA_PREFIX_ROWS if is_8k else tokens * TOP_K
    shape = (tokens // SUBLANE_ROWS, VREG_LANE_COUNT)
    vmem, smem, dma = pltpu.VMEM, pltpu.SMEM, pltpu.SemaphoreType.DMA
    layer_inputs = _LayerKernelInputs(
        tables=tables,
        control=control,
        hidden=hidden,
        router=router,
        bias=bias,
        latent_down=latent_down,
        shared_gu=shared_gu,
        shared_down=shared_down,
        latent=latent,
        fallbacks=(fallback_ids, fallback_probs, fallback_counts, fallback_shared),
        maximum=maximum,
        tokens=tokens,
        capacity=capacity,
    )

    def body(stage_state: dict[str, object]) -> None:
        """Coordinate scoped prologue, compact fast path, and 8K overflow export."""
        offset_ref = cast("PallasRefProtocol", stage_state["offset"])
        family_ref = cast("PallasRefProtocol", stage_state["family"])
        offset_ref[0] = 0
        # Dense projections retain global family indices. Resident expert banks
        # correspond to first/group/final programs with 3/8/1 MoE families.
        family_ref[0] = k.expert_family_slot(expert_gu.shape[0], control[1])
        if is_8k:
            # The compact path completes both metadata DMAs before expert work.
            # Reclaim the full probability array (512 KiB) for expert scratch.
            pl.run_scoped(
                lambda ids_v, probs_v: _prepare_layer_prologue(
                    layer_inputs, stage_state, ids_v, probs_v
                ),
                vmem(shape, jnp.int32),
                vmem(shape, jnp.float32),
            )
        else:
            probs_v = cast("PallasRefProtocol", stage_state["probs_v"])
            pl.run_scoped(
                lambda ids_v: _prepare_layer_prologue(
                    layer_inputs, stage_state, ids_v, probs_v
                ),
                vmem(shape, jnp.int32),
            )

        def probabilities_ready() -> None:
            """Wait for <=4K probability copy before routed expert consumption."""
            if not is_8k:
                probs_v = cast("PallasRefProtocol", stage_state["probs_v"])
                pltpu.make_async_copy(
                    probs_v.at[: capacity // VREG_LANE_COUNT, :],
                    cast("PallasRefProtocol", stage_state["probs"]),
                    cast("PallasRefProtocol", stage_state["prob_sem"]),
                ).wait()

        def fast() -> None:
            """Run compact routed experts followed by normalization and latent-up."""
            _routed_experts(
                latent,
                expert_gu,
                gu_scales,
                expert_down,
                down_scales,
                cast("PallasRefProtocol", stage_state["counts"]),
                _FlatMetadata(cast("PallasRefProtocol", stage_state["ids"])),
                _FlatMetadata(cast("PallasRefProtocol", stage_state["probs"])),
                tables,
                family_ref,
                offset_ref,
                middle,
                cast("PallasRefProtocol", stage_state["reduced"]),
                tokens=tokens,
                probabilities_ready=probabilities_ready,
            )
            _epilogue(
                control,
                cast("PallasRefProtocol", stage_state["reduced"]),
                cast("PallasRefProtocol", stage_state["shared"]),
                norm,
                latent_up,
                prefix,
                output,
            )

        if not is_8k:
            fast()
        else:
            max_smem = cast("PallasRefProtocol", stage_state["maximum"])

            @pl.when(max_smem[0] <= capacity)
            def compact() -> None:
                """Execute compact fast path when global route peak fits prefix."""
                fast()

    pl.run_scoped(
        body,
        {
            "shared": vmem((rows, HIDDEN_SIZE), jnp.bfloat16),
            "reduced": vmem((rows, _LATENT_FULL_COLS), jnp.float32),
            "probs_v": None if is_8k else vmem(shape, jnp.float32),
            "counts_v": vmem((_LOCAL_EXPERTS,), jnp.int32),
            "counts": smem((_LOCAL_EXPERTS,), jnp.int32),
            "maximum": smem((1,), jnp.int32) if is_8k else None,
            "ids": smem((capacity // VREG_LANE_COUNT, VREG_LANE_COUNT), jnp.int32),
            "probs": smem((capacity // VREG_LANE_COUNT, VREG_LANE_COUNT), jnp.float32),
            "family": smem((1,), jnp.int32),
            "offset": smem((1,), jnp.int32),
            "count_sem": dma,
            "max_sem": dma if is_8k else None,
            "index_sem": dma,
            "prob_sem": dma,
            "export_sem": dma,
        },
    )


def routed_experts_prepared(
    routes: PreparedRoutes,
    weights: RoutedExpertWeights,
    *,
    reduce_tables: jax.typing.ArrayLike,
    global_route_max: jax.Array | int | None = None,
) -> jax.Array:
    """Consume prepared 8K expert-major metadata with indexed gate/up and down."""
    tokens = routes.latent.shape[0]
    if tokens != OVERFLOW_WINDOW_TOKENS:
        msg = (
            "routed_experts_prepared requires "
            f"tokens={OVERFLOW_WINDOW_TOKENS}, got tokens={tokens}"
        )
        raise ValueError(msg)
    expert_layer = weights.expert_layer
    if expert_layer is not None:
        expert_layer = k.expert_family_slot(weights.gate_up.shape[0], expert_layer)
    gu_scale = weights.gate_up_scales.astype(jnp.uint8)
    down_scale = weights.down_scales.astype(jnp.uint8)
    middle = indexed_gate_up(
        routes.latent,
        routes.sorted_tokens,
        weights.gate_up,
        routes.counts,
        gu_scale,
        expert_layer,
    )
    return down_valu_compact(
        middle,
        weights.down,
        down_scale,
        routes.counts,
        routes.sorted_tokens,
        routes.sorted_probs,
        reduce_tables,
        tile=_COMPACT_DOWN_TILE,
        columns=_COMPACT_DOWN_COLUMNS,
        unroll=_COMPACT_DOWN_UNROLL,
        metadata_prefix_rows=COMPACT_METADATA_PREFIX_ROWS,
        slice_input=False,
        global_route_max=global_route_max,
        tokens=tokens,
        layer=expert_layer,
        ring_dtype=jnp.float32,
    )


def make_indexed_moe_sp_ring(
    mesh: Mesh | None,
    *,
    seq_len: int = OVERFLOW_WINDOW_TOKENS,
) -> Callable[..., jax.Array]:
    """Build the fused EP16 x TP2 prefill MoE callable for ``seq_len``."""
    if seq_len not in PREFILL_WINDOWS:
        msg = "Fused prefill MoE supports 512, 1024, 2048, 4096 and 8192 token windows"
        raise ValueError(msg)
    if _config_mod.active().moe_reduce_dtype != "float32":
        msg = "Fused prefill MoE requires FP32 reduction; use the dispatcher for BF16"
        raise ValueError(msg)
    if mesh is None or mesh.size != TP_SIZE or tuple(mesh.axis_names) != (AXIS_SP,):
        msg = "Fused prefill MoE requires a flat 32-rank sequence-parallel mesh"
        raise ValueError(msg)

    rows = seq_len // TP_SIZE
    tables = k.ring_tables(list(mesh.devices.flat), tuple(range(TP_SIZE)))
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    shape = (seq_len // SUBLANE_ROWS, VREG_LANE_COUNT)
    bf16 = jnp.dtype(jnp.bfloat16)
    workspace_shapes: list[tuple[tuple[int, ...], jnp.dtype]] = [
        ((rows, HIDDEN_SIZE), bf16),
        ((seq_len, 2, _LATENT_HALF_COLS), bf16),
        ((seq_len * TOP_K, _EXPERT_INTERMEDIATE_COLS), bf16),
    ]
    output_specs: list[pl.BlockSpec] = [hbm] * 3
    if seq_len == OVERFLOW_WINDOW_TOKENS:
        # Only 8K can overflow compact SMEM routing capacity. At <=4K every
        # route fits, so omit the unused overflow outputs and maximum DMA.
        workspace_shapes += [
            (shape, jnp.dtype(jnp.int32)),
            (shape, jnp.dtype(jnp.float32)),
            ((_LOCAL_EXPERTS,), jnp.dtype(jnp.int32)),
            ((rows, HIDDEN_SIZE), bf16),
            ((1,), jnp.dtype(jnp.int32)),
        ]
        output_specs += [hbm] * 4 + [pl.BlockSpec()]
    kernel = pl.pallas_call(
        _layer_kernel,
        out_shape=tuple(
            jax.ShapeDtypeStruct(workspace_shape, workspace_dtype)
            for workspace_shape, workspace_dtype in workspace_shapes
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=(hbm,) * _NUM_LAYER_INPUT_SPECS,
            out_specs=tuple(output_specs),
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=0,
            vmem_limit_bytes=_FUSED_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
            shape_invariant_numerics=False,
        ),
        name=f"fused_moe_pallas_{seq_len}",
    )

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(P(AXIS_SP), P(AXIS_SP), P(AXIS_SP), P(), P()),
        out_specs=P(AXIS_SP),
        check_vma=False,
    )
    def local(
        hidden: jax.Array,
        prefix: jax.Array,
        families: dict[str, jax.Array],
        length: jax.Array,
        slot: jax.Array,
    ) -> jax.Array:
        """Run fused MoE on one shard and dispatch 8K overflow when needed."""
        w = {name: value[0] for name, value in families.items()}
        control = jnp.concatenate((length.reshape(1), slot.reshape(1)))
        outputs = kernel(
            jnp.asarray(tables),
            control,
            hidden[0],
            w["router"],
            w["router_bias"],
            w["latent_down"],
            w["shared_gu"],
            w["shared_down"],
            w["expert_gu"],
            w["expert_gus"],
            w["expert_down"],
            w["expert_ds"],
            w["latent_norm"],
            w["latent_up"],
            prefix[0],
        )
        if seq_len < OVERFLOW_WINDOW_TOKENS:
            return outputs[0][None]
        final, latent, _, ids, probs, counts, shared, maximum = outputs

        def overflow() -> jax.Array:
            """Execute uncapped fallback expert pipeline on 8K route overflow."""
            reduced = routed_experts_prepared(
                PreparedRoutes(
                    latent=latent,
                    sorted_tokens=ids.reshape(-1),
                    sorted_probs=probs.reshape(-1),
                    counts=counts,
                ),
                RoutedExpertWeights(
                    gate_up=w["expert_gu"],
                    gate_up_scales=w["expert_gus"],
                    down=w["expert_down"],
                    down_scales=w["expert_ds"],
                    expert_layer=slot,
                ),
                reduce_tables=tables,
                global_route_max=maximum[0],
            )
            selected = {name: w[name][slot[0]] for name in ("latent_norm", "latent_up")}
            return k.moe_sp_tail(
                reduced,
                shared,
                prefix[0],
                length,
                selected,
                axis_name=AXIS_SP,
                eps=_RMS_EPS,
                shared_reduced=True,
            )

        return jax.lax.cond(
            maximum[0] <= COMPACT_METADATA_PREFIX_ROWS, lambda: final, overflow
        )[None]

    return jax.jit(local)
