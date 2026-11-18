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

# Source provenance: inferact/k3-megakernel@30c9ba966b
# src/vllm_torchtpu/megakernel/kimi_k3/kernels/prefill_moe/fused.py
"""Fused EP16/TP2 prefill MoE for 512- through 8192-token windows.

Schedule:
  1. Router/latent/shared projections overlap the incremental Top16, route sort,
     route exchange, packed latent ring and shared-output reduction.
  2. Compact routing metadata moves to SMEM. GU/SiTU and the first down column
     share a pipeline; later down columns overlap token combine and reduction.
  3. The reduced VMEM shard is normalized, projected up and added to the shared
     expert and residual outputs.

The factory at the end is the only production entry point. The small-window, 4K
and 8K schedules are selected by window size; 8K skew overflow retains every route via
an uncapped fallback. Python generators split work at tracing time, allowing
independent vector/communication work between MXU panels without extra calls.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING, Protocol, TypedDict, Unpack, cast

import jax
import jax.numpy as jnp
import jax.typing
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from ... import config as _config_mod
from ...common.constants import OVERFLOW_WINDOW_TOKENS as OVERFLOW_WINDOW_TOKENS
from ...common.constants import SUBLANE_ROWS, TP_PEERS, TP_SIZE, VREG_LANE_COUNT
from .. import prefill_megakernel as k
from .indexed_ragged_gmm import PackedRows, indexed_gate_up
from .moe_down_valu import STORE_TILE, down_valu, down_valu_compact

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from jax.sharding import Mesh

    _PrologueArgsTuple = tuple[
        "_PallasRef",
        "_PallasRef",
        "_PallasRef",
        "_PallasRef",
        "_PallasRef",
        "_PallasRef",
        "_PallasRef",
        "_PallasRef",
        "_PallasRef",
        "_PallasRef",
        "_PallasRef",
        "_PallasRef",
        "_PallasRef | None",
        jax.Array,
        jax.Array,
        Callable[[], None],
        Callable[[], None],
    ]

_SORT_WINDOW = 4096
_SORT_ROWS = 32
_REG_CHUNK_DIST = 1024
_TOP_K = 16
_PROB_FLOOR = 1e-30
_SMALL_WINDOW_LIMIT = 4096
_LOCAL_EXPERTS = 56
_EXPERT_GROUPS = 16
_CHUNK_TOKENS = 128
_CHUNK_RECORDS = 32
_HALF_RECORDS = 16
_ROUTE_POS_STRIDE = 2048
_PANEL_COLS = 256
_LATENT_HALF_COLS = 1792
_LATENT_FULL_COLS = 3584
_LATENT_PADDED_COLS = 4096
_LATENT_PANELS = 14
_PAIR_BANKS = 4
_ROUTE_GATHER_STAGES = 5
_ROUTE_DONE = object()
_RING_WRAP_OFFSET = 64
_RING_STREAMS = 12
_DIRECTED_RINGS = 6
_RING_META_STRIDE = 34
_RING_MASK = 31
_RING_BUFFERS = 2
_VALID_LATENT_ROWS = (16, 32, 64, 128, 256)
_VMEM_GATHER_MAX_ROWS = 128
_ROWS_8K = 256
_DMA_BYTES_PER_UNIT = 32
_PHASE_COMPLETE = 2
_PHASE_DONE = 3
_MAX_STRIPE_ROWS = 64
_ROUTER_COLS = 896
_ROUTER_GU_COLS = 1024
_SHARED_INTERMEDIATE_COLS = 3072
_SHARED_GU_PANELS = 12
_SHARED_DOWN_PANELS = 28
_SITU_GATE_SCALE = 4.0
_SITU_UP_SCALE = 25.0
_COMPACT_METADATA_PREFIX_ROWS = 32768
_COMPACT_DOWN_TILE = 256
_COMPACT_DOWN_COLUMNS = 896
_COMPACT_DOWN_UNROLL = 32
_DEFAULT_DOWN_TILE = 128
_DEFAULT_DOWN_COLUMNS = 1792
_DEFAULT_DOWN_UNROLL = 8
_GU_UNROLL_DEFAULT = 4
_GU_UNROLL_8K = 8
_FIRST_DOWN_UNROLL_DEFAULT = 2
_FIRST_DOWN_UNROLL_8K = 8
_PACKED_HALF_8K = 512
_VMEM_LIMIT_FRACTION = 0.98
_EXPERT_INTERMEDIATE_COLS = 1536
_SPARSE_POLL_NG_THRESHOLD = 72
_EPILOGUE_HIDDEN_COLS = 7168
_EPILOGUE_PANEL_COLS = 512
_EPILOGUE_BANKS = 3
_RMS_EPS = 1e-5
_SUPPORTED_WINDOWS = (512, 1024, 2048, 4096, 8192)
_FUSED_VMEM_LIMIT_BYTES = 64 * 1024**2
_NUM_PROLOGUE_INPUT_SPECS = 13


class _AsyncCopy(Protocol):
    """Protocol for a Pallas async DMA copy descriptor."""

    def start(self) -> None:
        """Start the async DMA transfer."""
        ...

    def wait(self) -> None:
        """Wait for the async DMA transfer to complete."""
        ...


class _RemoteCopy(Protocol):
    """Protocol for a Pallas async remote DMA copy descriptor."""

    def start(self) -> None:
        """Start the async remote DMA transfer."""
        ...

    def wait(self) -> None:
        """Wait for both send and receive completion."""
        ...

    def wait_send(self) -> None:
        """Wait for the local send completion."""
        ...

    def wait_recv(self) -> None:
        """Wait for the local receive completion."""
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


@dataclasses.dataclass(frozen=True)
class _WindowedBuffer:
    """Pipeline buffer allocation wrapper carrying a ``window_ref``."""

    window_ref: _PallasRef | None = None


class _ReadyReduceOptions(TypedDict, total=False):
    """Keyword options for ``_ready_reduce``."""

    streams: Sequence[tuple[int, ...]]
    buffers: int
    state: _PallasRef
    ring_meta: _PallasRef
    drain: bool
    poll_round: jax.Array | int
    sparse_poll: jax.Array | bool


def _bitonic_partner(
    value: jax.Array,
    distance: int,
    lane: jax.Array,
) -> jax.Array:
    """Permute ``value`` across bitonic comparator distance ``distance``."""
    shape = (_SORT_ROWS, VREG_LANE_COUNT)
    if distance >= _REG_CHUNK_DIST:
        # Reorder register-sized chunks without a gather-index constant.
        step = distance // _REG_CHUNK_DIST
        return jnp.concatenate(
            [
                value[
                    SUBLANE_ROWS * (i ^ step) : SUBLANE_ROWS * (i ^ step) + SUBLANE_ROWS
                ]
                for i in range(_SORT_ROWS // SUBLANE_ROWS)
            ],
            axis=0,
        )
    if distance >= VREG_LANE_COUNT:
        shape3 = (_SORT_ROWS // SUBLANE_ROWS, SUBLANE_ROWS, VREG_LANE_COUNT)
        index = jax.lax.broadcasted_iota(jnp.int32, shape3, 1)
        return jnp.take_along_axis(
            value.reshape(shape3),
            index ^ (distance // VREG_LANE_COUNT),
            axis=1,
            mode="promise_in_bounds",
        ).reshape(shape)
    return jnp.take_along_axis(value, lane ^ distance, axis=1, mode="promise_in_bounds")


def _flush_sort_window(
    keys: _PallasRef,
    probabilities: _PallasRef,
    pending: list[tuple[int, int]],
    positions: jax.Array,
    lane: jax.Array,
) -> Iterator[None]:
    """Execute queued in-window bitonic stages across all 4096-route windows."""
    if not pending:
        return
    num_windows = (keys.shape[0] * VREG_LANE_COUNT) // _SORT_WINDOW
    for ordinal in range(num_windows):
        row_slice = pl.ds(ordinal * _SORT_ROWS, _SORT_ROWS)
        kr, pr = keys.at[row_slice, :], probabilities.at[row_slice, :]
        x, p = kr[...], pr[...]
        global_position = positions + ordinal * _SORT_WINDOW
        for distance, group in pending:
            y = _bitonic_partner(x, distance, lane)
            q = _bitonic_partner(p, distance, lane)
            descending = (global_position & group) != 0
            upper = (global_position & distance) != 0
            take_min = ~(descending ^ upper)
            keep = ~((x < y) ^ take_min)
            x, p = jnp.where(keep, x, y), jnp.where(keep, p, q)
        kr[...], pr[...] = x, p
        yield
    pending.clear()


def _sort_route_steps(
    keys: _PallasRef,
    probabilities: _PallasRef,
    *,
    size: int,
) -> Iterator[None]:
    """Yield after each 4096-route window of the in-place bitonic network.

    Keys encode (local expert, original route position), so all live keys are
    unique and the result has deterministic expert-major/token-major order.
    Probability values follow exactly the same swaps as their keys.

    Invariants:
        assert size >= window and size & (size - 1) == 0
    """
    if size < _SORT_WINDOW or (size & (size - 1)) != 0:
        msg = f"Bitonic sort requires power-of-two size >= {_SORT_WINDOW}: {size}"
        raise ValueError(msg)
    shape = (_SORT_ROWS, VREG_LANE_COUNT)
    row = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    lane = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
    positions = row * VREG_LANE_COUNT + lane

    def view(ref: _PallasRef, ordinal: int) -> _PallasRef:
        """Slice one 4096-route window at ``ordinal``."""
        return ref.at[pl.ds(ordinal * _SORT_ROWS, _SORT_ROWS), :]

    pending: list[tuple[int, int]] = []
    for stage in range(1, size.bit_length()):
        for shift in reversed(range(stage)):
            distance, group = 1 << shift, 1 << stage
            if distance < _SORT_WINDOW:
                pending.append((distance, group))
                continue
            yield from _flush_sort_window(keys, probabilities, pending, positions, lane)
            stride = distance // _SORT_WINDOW
            for ordinal in range(size // (2 * _SORT_WINDOW)):
                low = ordinal // stride * (2 * stride) + ordinal % stride
                high = low + stride
                kl, kh = view(keys, low), view(keys, high)
                plow, phigh = view(probabilities, low), view(probabilities, high)
                x, y, p, q = kl[...], kh[...], plow[...], phigh[...]
                ascending = (low * _SORT_WINDOW & group) == 0
                keep = ~((x < y) ^ ascending)
                kl[...], kh[...] = jnp.where(keep, x, y), jnp.where(keep, y, x)
                plow[...], phigh[...] = jnp.where(keep, p, q), jnp.where(keep, q, p)
                yield
    yield from _flush_sort_window(keys, probabilities, pending, positions, lane)


class _Top16:
    """Incremental biased sigmoid Top16; ties choose the lowest expert ID."""

    def __init__(self, logits: jax.Array, bias: jax.Array) -> None:
        """Initialize biased sigmoid scores and selection accumulators."""
        rows, _experts = logits.shape
        self.raw = 1.0 / (1.0 + jnp.exp(-logits))
        self.scores = self.raw + bias
        self.experts = jax.lax.broadcasted_iota(jnp.int32, logits.shape, 1)
        self.slots = jax.lax.broadcasted_iota(jnp.int32, (rows, _TOP_K), 1)
        self.selected = jnp.zeros((rows, _TOP_K), jnp.int32)
        self.probabilities = jnp.zeros((rows, _TOP_K), jnp.float32)
        self.index = 0

    def step(self) -> None:
        """Select the next highest-scoring expert across all rows."""
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
        """Complete remaining Top-16 steps and normalize selected probabilities."""
        while self.index < _TOP_K:
            self.step()
        denominator = jnp.maximum(
            jnp.sum(self.probabilities, axis=1, keepdims=True), _PROB_FLOOR
        )
        return self.selected, jnp.where(valid, self.probabilities / denominator, 0.0)


def _unpack_wire_chunk(
    wire: _PallasRef,
    chunk: int,
    tokens: int,
) -> tuple[jax.Array, jax.Array]:
    """Extract expert IDs and FP32 probabilities for one 128-token wire chunk."""
    if tokens < _SMALL_WINDOW_LIMIT:
        records = wire[pl.ds(chunk * _CHUNK_RECORDS, _CHUNK_RECORDS), :].reshape(
            _CHUNK_TOKENS, _CHUNK_RECORDS
        )
        ids = records[:, :_HALF_RECORDS]
        probs = jax.lax.bitcast_convert_type(records[:, _HALF_RECORDS:], jnp.float32)
        return ids, probs
    # Keep IDs and probabilities in separate planes so 128-token
    # chunks need no interleaved-record reshape during metadata prep.
    chunks_per_rank = tokens // _SMALL_WINDOW_LIMIT
    rank_base = (chunk // chunks_per_rank) * (tokens // _CHUNK_TOKENS)
    chunk_base = (chunk % chunks_per_rank) * _HALF_RECORDS
    ids = wire[pl.ds(rank_base + chunk_base, _HALF_RECORDS), :]
    probs = jax.lax.bitcast_convert_type(
        wire[pl.ds(rank_base + tokens // _PANEL_COLS + chunk_base, _HALF_RECORDS), :],
        jnp.float32,
    )
    return ids, probs


def _accumulate_chunk_counts(
    local_flat: jax.Array,
    ids_flat: jax.Array,
    live_flat: jax.Array,
    expert_index: jax.Array,
    group_index: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Count per-expert and per-group routes within one 128-token chunk."""
    tile_counts = jnp.zeros((_LOCAL_EXPERTS,), jnp.int32)
    for expert in range(_LOCAL_EXPERTS):
        count = jnp.sum((local_flat == expert).astype(jnp.int32))
        tile_counts = jnp.where(expert_index == expert, count, tile_counts)
    tile_groups = jnp.zeros((_EXPERT_GROUPS,), jnp.int32)
    groups = ids_flat // _LOCAL_EXPERTS
    for group in range(_EXPERT_GROUPS):
        count = jnp.sum((live_flat & (groups == group)).astype(jnp.int32))
        tile_groups = jnp.where(group_index == group, count, tile_groups)
    return tile_counts, tile_groups


def _extract_sorted_token_ids(
    keys: _PallasRef,
    token_ids: _PallasRef,
    tokens: int,
    bits: int,
) -> None:
    """Extract sorted token indices from the low bits of ``keys``."""

    @pl.loop(0, tokens * _TOP_K // _SORT_WINDOW)
    def extract(chunk: jax.Array) -> None:
        """Unpack one 4096-route window of token IDs."""
        index = pl.ds(chunk * _SORT_ROWS, _SORT_ROWS)
        token_ids[index, :] = (keys[index, :] & ((1 << bits) - 1)) // _TOP_K


def _metadata_steps(
    wire: _PallasRef,
    keys: _PallasRef,
    probabilities: _PallasRef,
    token_ids: _PallasRef,
    counts: _PallasRef,
    maximum: _PallasRef | None,
    *,
    tokens_and_rank: tuple[int, jax.Array],
    counts_ready: Callable[[], None],
) -> Iterator[None]:
    """Build stable local routing metadata, yielding one prep/sort chunk at a time.

    The all-gathered wire contains 16 expert IDs followed by 16 probability bit
    patterns per token. Every rank counts all 16 expert groups at 8K, making the
    compact-capacity decision uniform before any conditional collective runs.
    """
    tokens, rank = tokens_and_rank
    bits = (tokens * _TOP_K - 1).bit_length()
    shape = (
        (_CHUNK_TOKENS, _TOP_K)
        if tokens < _SMALL_WINDOW_LIMIT
        else (_TOP_K, _CHUNK_TOKENS)
    )
    expert_index = jnp.arange(_LOCAL_EXPERTS, dtype=jnp.int32)
    group_index = jnp.arange(_EXPERT_GROUPS, dtype=jnp.int32)
    lanes = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
    rows = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    total_counts = jnp.zeros((_LOCAL_EXPERTS,), jnp.int32)
    total_groups = jnp.zeros((_EXPERT_GROUPS,), jnp.int32)
    for chunk in range(tokens // _CHUNK_TOKENS):
        ids, probs = _unpack_wire_chunk(wire, chunk, tokens)
        local = ids - (rank // 2 * _LOCAL_EXPERTS)
        valid = (probs != 0) & (local >= 0) & (local < _LOCAL_EXPERTS)
        local = jnp.where(valid, local, _LOCAL_EXPERTS)
        original_position = chunk * _ROUTE_POS_STRIDE + rows * shape[1] + lanes
        packed = (local << bits) | original_position
        keys[pl.ds(chunk * _HALF_RECORDS, _HALF_RECORDS), :] = packed.reshape(
            _HALF_RECORDS, VREG_LANE_COUNT
        )
        probabilities[pl.ds(chunk * _HALF_RECORDS, _HALF_RECORDS), :] = probs.reshape(
            _HALF_RECORDS, VREG_LANE_COUNT
        )
        # Count routes on the packed 16x128 layout.
        local_flat = local.reshape(_HALF_RECORDS, VREG_LANE_COUNT)
        ids_flat = ids.reshape(_HALF_RECORDS, VREG_LANE_COUNT)
        live_flat = probs.reshape(_HALF_RECORDS, VREG_LANE_COUNT) != 0
        if tokens == OVERFLOW_WINDOW_TOKENS:
            tile_counts, tile_groups = _accumulate_chunk_counts(
                local_flat, ids_flat, live_flat, expert_index, group_index
            )
            total_counts += tile_counts
            total_groups += tile_groups
        else:
            tile_counts = jnp.zeros((_LOCAL_EXPERTS,), jnp.int32)
            for expert in range(_LOCAL_EXPERTS):
                count = jnp.sum((local_flat == expert).astype(jnp.int32))
                tile_counts = jnp.where(expert_index == expert, count, tile_counts)
            total_counts += tile_counts
        yield
    counts[...] = total_counts
    if maximum is not None:
        maximum[...] = jnp.max(total_groups)[None]
    counts_ready()
    yield from _sort_route_steps(keys, probabilities, size=tokens * _TOP_K)
    _extract_sorted_token_ids(keys, token_ids, tokens, bits)


def _dense_panels(
    lhs: jax.Array,
    weights: _PallasRef,
    buffers: Sequence[_PallasRef],
    semaphores: Sequence[_PallasRef],
    *,
    columns: int,
) -> Iterator[tuple[int, jax.Array]]:
    """Stream 256-column BF16 panels, with the next weight DMA in flight.

    The yielded FP32 product is deliberately not stored here. The caller places
    independent routing/vector/communication work before consuming it so the
    compiler can schedule that work alongside the MXU product.
    """

    def source(panel: int) -> _PallasRef:
        """Slice 256 columns of ``weights`` at ``panel``."""
        return weights.at[:, pl.ds(panel * _PANEL_COLS, _PANEL_COLS)]

    cast(
        "_AsyncCopy", pltpu.make_async_copy(source(0), buffers[0], semaphores[0])
    ).start()
    for panel in range(columns // _PANEL_COLS):
        bank = panel % 2
        cast(
            "_AsyncCopy",
            pltpu.make_async_copy(source(panel), buffers[bank], semaphores[bank]),
        ).wait()
        if panel + 1 < columns // _PANEL_COLS:
            cast(
                "_AsyncCopy",
                pltpu.make_async_copy(
                    source(panel + 1), buffers[1 - bank], semaphores[1 - bank]
                ),
            ).start()
        yield (
            panel * _PANEL_COLS,
            jnp.matmul(lhs, buffers[bank][...], preferred_element_type=jnp.float32),
        )


def _latent_half_pair_panels(
    lhs: jax.Array,
    weights: _PallasRef,
    buffers: Sequence[_PallasRef],
    semaphores: Sequence[_PallasRef],
) -> Iterator[tuple[int, jax.Array]]:
    """Pair matching 256-column fragments of the two 1792-wide halves."""

    def source_column(panel: int) -> int:
        """Map paired panel index ``panel`` to its starting column offset."""
        return (panel // 2) * _PANEL_COLS + (panel % 2) * _LATENT_HALF_COLS

    def copy(panel: int) -> _AsyncCopy:
        """Create the async weight DMA descriptor for ``panel``."""
        return cast(
            "_AsyncCopy",
            pltpu.make_async_copy(
                weights.at[:, pl.ds(source_column(panel), _PANEL_COLS)],
                buffers[panel % _PAIR_BANKS],
                semaphores[panel % _PAIR_BANKS],
            ),
        )

    for panel in range(2):
        copy(panel).start()
    for panel in range(0, _LATENT_PANELS, 2):
        for j in range(2):
            copy(panel + j).wait()
        for j in range(2):
            if panel + 2 + j < _LATENT_PANELS:
                copy(panel + 2 + j).start()
        products = [
            jnp.matmul(
                lhs,
                buffers[(panel + j) % _PAIR_BANKS][...],
                preferred_element_type=jnp.float32,
            )
            for j in range(2)
        ]
        for j, product in enumerate(products):
            yield source_column(panel + j), product


def _route_gather_steps(
    payload: _PallasRef,
    output: _PallasRef,
    send: _PallasRef,
    recv: _PallasRef,
    *,
    rank: jax.Array,
) -> Iterator[int]:
    """Five recursive-doubling VMEM exchanges; one completion counter per round.

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
            "_RemoteCopy",
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
            "_RemoteCopy",
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
    """Match collective lifetimes before any peer can write another's scratch."""
    barrier = pltpu.get_barrier_semaphore()
    for peer in range(TP_SIZE):
        pl.semaphore_signal(
            barrier, 1, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL
        )
    pl.semaphore_wait(barrier, TP_SIZE)


_LatentRegion = tuple[int, int, int, int, jax.Array, jax.Array, jax.Array]


def _build_latent_regions(
    tables: _PallasRef,
    rows: int,
    rank: jax.Array,
) -> list[_LatentRegion]:
    """Build the 12 directional stream descriptors for the latent ring."""
    regions: list[_LatentRegion] = []
    begin = 0
    for stream in range(_RING_STREAMS):
        size = rows // _RING_STREAMS + int(stream < rows % _RING_STREAMS)
        family = stream // 2 % 3
        direction = -1 if stream % 2 == 0 else 1
        position = tables[family, 0, rank]
        target = tables[family, 1, (position + TP_SIZE + direction) % TP_SIZE]
        source_peer = tables[family, 1, (position + TP_SIZE - direction) % TP_SIZE]
        regions.append((begin, size, family, direction, position, target, source_peer))
        begin += size
    return regions


def _latent_vmem_gather_steps(
    tables: _PallasRef,
    payload: _PallasRef,
    regions: Sequence[_LatentRegion],
    *ring_refs: _PallasRef,
) -> Iterator[None]:
    """Advance the <=4K VMEM-forwarded latent all-gather ring one hop at a time."""
    received, output, send, recv, credit_sems, stores = ring_refs
    rows = payload.shape[0]

    def destination(region: _LatentRegion, step: int) -> _PallasRef:
        """Return the HBM output slice owned by the peer for ``step``."""
        begin, size, family, direction, position, _, _ = region
        owner = tables[
            family, 1, (position + _RING_WRAP_OFFSET - direction * step) % TP_SIZE
        ]
        return output.at[pl.ds(owner * rows + begin, size), :, :]

    def remote(
        source: _PallasRef,
        target: _PallasRef,
        ss: _PallasRef,
        rs: _PallasRef,
        peer: jax.Array,
    ) -> _RemoteCopy:
        """Create an async remote DMA descriptor to ``peer``."""
        return cast(
            "_RemoteCopy",
            pltpu.make_async_remote_copy(
                source,
                target,
                ss,
                rs,
                device_id=peer,
                device_id_type=pl.DeviceIdType.LOGICAL,
            ),
        )

    for step in range(TP_SIZE):
        for stream, region in enumerate(regions):
            begin, size, _, _, _, peer, source_peer = region
            piece = payload.at[pl.ds(begin, size), :, :]
            ss = send.at[stream]
            credit = credit_sems.at[stream]
            st = stores.at[stream]
            if step:
                old = received.at[(step - 1) % 2, pl.ds(begin, size), :, :]
                transfer = remote(piece, old, ss, recv.at[(step - 1) % 2, stream], peer)
                transfer.wait_send()
                transfer.wait_recv()
                cast(
                    "_AsyncCopy",
                    pltpu.make_async_copy(piece, destination(region, step - 1), st),
                ).wait()
                piece[...] = old[...]
                pl.semaphore_signal(
                    credit,
                    1,
                    device_id=source_peer,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )
            dest = destination(region, step)
            store = cast("_AsyncCopy", pltpu.make_async_copy(piece, dest, st))
            store.start()
            if step < TP_PEERS:
                if step >= _RING_BUFFERS:
                    pl.semaphore_wait(credit, 1)
                future = received.at[step % 2, pl.ds(begin, size), :, :]
                remote(piece, future, ss, recv.at[step % 2, stream], peer).start()
            else:
                store.wait()
                pl.semaphore_wait(credit, _RING_BUFFERS)
        yield


def _latent_hbm_gather_steps(
    tables: _PallasRef,
    payload: _PallasRef,
    regions: Sequence[_LatentRegion],
    *ring_refs: _PallasRef,
    rank: jax.Array,
) -> Iterator[None]:
    """Advance the 8K HBM-forwarded latent all-gather ring one hop at a time."""
    output, send, recv, credit_sems, stores = ring_refs
    rows = payload.shape[0]

    def destination(region: _LatentRegion, step: int) -> _PallasRef:
        """Return the HBM output slice owned by the peer for ``step``."""
        begin, size, family, direction, position, _, _ = region
        owner = tables[
            family, 1, (position + _RING_WRAP_OFFSET - direction * step) % TP_SIZE
        ]
        return output.at[pl.ds(owner * rows + begin, size), :, :]

    def remote(
        source: _PallasRef,
        target: _PallasRef,
        ss: _PallasRef,
        rs: _PallasRef,
        peer: jax.Array,
    ) -> _RemoteCopy:
        """Create an async remote DMA descriptor to ``peer``."""
        return cast(
            "_RemoteCopy",
            pltpu.make_async_remote_copy(
                source,
                target,
                ss,
                rs,
                device_id=peer,
                device_id_type=pl.DeviceIdType.LOGICAL,
            ),
        )

    own = output.at[pl.ds(rank * rows, rows), :, :]
    seed = cast("_AsyncCopy", pltpu.make_async_copy(payload, own, stores))
    seed.start()
    pending: list[tuple[_RemoteCopy, _PallasRef, int] | None] = [None] * _RING_STREAMS
    for step in range(TP_SIZE):
        for stream, region in enumerate(regions):
            begin, size, _, _, _, peer, source_peer = region
            pending_entry = pending[stream]
            if pending_entry is not None:
                transfer, incoming, completed_step = pending_entry
                transfer.wait_send()
                remote(
                    incoming,
                    incoming,
                    send.at[stream],
                    recv.at[completed_step % 2, stream],
                    peer,
                ).wait_recv()
                if completed_step + _RING_BUFFERS < TP_PEERS:
                    pl.semaphore_signal(
                        credit_sems.at[stream],
                        1,
                        device_id=source_peer,
                        device_id_type=pl.DeviceIdType.LOGICAL,
                    )
            if step < TP_PEERS:
                source = (
                    payload.at[pl.ds(begin, size), :, :]
                    if step == 0
                    else destination(region, step)
                )
                dest = destination(region, step)
                incoming = destination(region, step + 1)
                if step >= _RING_BUFFERS:
                    pl.semaphore_wait(credit_sems.at[stream], 1)
                transfer = remote(
                    source, dest, send.at[stream], recv.at[step % 2, stream], peer
                )
                transfer.start()
                pending[stream] = (transfer, incoming, step)
        yield
    seed.wait()


def _latent_gather_steps(
    tables: _PallasRef,
    payload: _PallasRef,
    *ring_refs: _PallasRef,
    rank: jax.Array,
) -> Iterator[None]:
    """Twelve directional streams, with two receive slots per stream.

    Windows up to 4K forward through VMEM, copying completed receive slots
    into payload. 8K forwards immutable final HBM ranges to avoid that copy.
    The latter still returns credits before reusing receive counters; immutable
    data alone does not make reusing an unconsumed semaphore safe.

    Invariants:
        assert rows in (16, 32, 64, 128, 256)
    """
    received, output, send, recv, credit_sems, stores = ring_refs
    rows = payload.shape[0]
    if rows not in _VALID_LATENT_ROWS:
        msg = f"Unsupported latent payload row count: {rows}"
        raise ValueError(msg)
    regions = _build_latent_regions(tables, rows, rank)
    if rows <= _VMEM_GATHER_MAX_ROWS:
        yield from _latent_vmem_gather_steps(
            tables,
            payload,
            regions,
            received,
            output,
            send,
            recv,
            credit_sems,
            stores,
        )
        return
    yield from _latent_hbm_gather_steps(
        tables,
        payload,
        regions,
        output,
        send,
        recv,
        credit_sems,
        stores,
        rank=rank,
    )


def _init_latent_ready_owners(
    tables: _PallasRef,
    meta: _PallasRef,
    rows: int,
    ring_info: tuple[int, jax.Array, int, int],
) -> None:
    """Populate the 32 hop-owner HBM row offsets for one directed ring."""
    family, position, direction, directed = ring_info

    @pl.loop(0, TP_SIZE)
    def owners(step: jax.Array) -> None:
        """Cache the HBM row offset owned by ``step`` in SMEM."""
        owner = tables[family, 1, (position - direction * step) & _RING_MASK]
        meta[directed * _RING_META_STRIDE + 2 + step] = owner * rows


def _latent_ready_init(
    tables: _PallasRef,
    payload: _PallasRef,
    output: _PallasRef,
    *state_refs: _PallasRef,
    rank: jax.Array,
) -> None:
    """Seed twelve independently progressing streams of the 8K latent gather.

    Cache neighbor IDs and owner row offsets in SMEM once. Each stream tracks
    its hop and receive/forward/complete phase; state[0] counts active streams.

    Invariants:
        assert rows == 256
    """
    send, recv, stores, state, meta = state_refs
    rows = payload.shape[0]
    if rows != _ROWS_8K:
        msg = f"_latent_ready_init requires 256 rows, got {rows}"
        raise ValueError(msg)
    for directed in range(_DIRECTED_RINGS):
        family, direction = directed // 2, -1 if directed % 2 == 0 else 1
        position = tables[family, 0, rank]
        meta[directed * _RING_META_STRIDE] = tables[
            family, 1, (position + direction) & _RING_MASK
        ]
        meta[directed * _RING_META_STRIDE + 1] = tables[
            family, 1, (position - direction) & _RING_MASK
        ]
        _init_latent_ready_owners(
            tables, meta, rows, (family, position, direction, directed)
        )

    cast(
        "_AsyncCopy",
        pltpu.make_async_copy(
            payload, output.at[pl.ds(rank * rows, rows), :, :], stores
        ),
    ).start()
    state[0] = _RING_STREAMS
    begin = 0
    for stream in range(_RING_STREAMS):
        size = rows // _RING_STREAMS + int(stream < rows % _RING_STREAMS)
        part = pl.ds(begin, size)
        cast(
            "_RemoteCopy",
            pltpu.make_async_remote_copy(
                payload.at[part, :, :],
                output.at[pl.ds(rank * rows + begin, size), :, :],
                send.at[stream],
                recv.at[0, stream],
                device_id=meta[(stream % _DIRECTED_RINGS) * _RING_META_STRIDE],
                device_id_type=pl.DeviceIdType.LOGICAL,
            ),
        ).start()
        state[1 + stream] = 1
        state[1 + _RING_STREAMS + stream] = 0
        begin += size


def _poll_latent_ready_stream(
    output: _PallasRef,
    *state_refs: _PallasRef,
    stream_info: tuple[int, int, int, int, int],
) -> None:
    """Poll and advance one stream of the 8K ready-polled latent gather."""
    send, recv, credit_sems, state, meta = state_refs
    stream, start, size, base, units = stream_info
    step = state[1 + stream]

    @pl.when(state[1 + _RING_STREAMS + stream] == 0)
    def receive() -> None:
        """Consume completed remote DMA counters and return credit."""
        bank = (step - 1) & 1
        ready = (pl.semaphore_read(send.at[stream]) >= units) & (
            pl.semaphore_read(recv.at[bank, stream]) >= units
        )

        @pl.when(ready)
        def consume() -> None:
            """Complete the receive hop and transition to forward or done."""
            incoming = output.at[pl.ds(meta[base + 2 + step] + start, size), :, :]
            # wait_send only consumes its completion count; the HBM view
            # has the same logical transfer size as the initial VMEM source.
            transfer = cast(
                "_RemoteCopy",
                pltpu.make_async_remote_copy(
                    incoming,
                    incoming,
                    send.at[stream],
                    recv.at[bank, stream],
                    device_id=meta[base],
                    device_id_type=pl.DeviceIdType.LOGICAL,
                ),
            )
            transfer.wait_send()
            transfer.wait_recv()

            @pl.when(step < TP_PEERS - 1)
            def credit() -> None:
                """Return buffer credit to the upstream sender."""
                pl.semaphore_signal(
                    credit_sems.at[stream],
                    1,
                    device_id=meta[base + 1],
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )

            state[1 + _RING_STREAMS + stream] = jnp.where(
                step == TP_PEERS, _PHASE_COMPLETE, 1
            )

            @pl.when(step == TP_PEERS)
            def complete() -> None:
                """Decrement the active stream counter on final hop."""
                state[0] = state[0] - 1

    @pl.when(state[1 + _RING_STREAMS + stream] == 1)
    def forward() -> None:
        """Start the next outgoing hop once downstream credit is available."""
        available = (step < _RING_BUFFERS) | (
            pl.semaphore_read(credit_sems.at[stream]) >= 1
        )

        @pl.when(available)
        def issue() -> None:
            """Acquire credit if needed and issue the remote DMA."""

            @pl.when(step >= _RING_BUFFERS)
            def acquire() -> None:
                """Consume one downstream flow-control credit."""
                pl.semaphore_wait(credit_sems.at[stream], 1)

            source = output.at[pl.ds(meta[base + 2 + step] + start, size), :, :]
            cast(
                "_RemoteCopy",
                pltpu.make_async_remote_copy(
                    source,
                    source,
                    send.at[stream],
                    recv.at[step & 1, stream],
                    device_id=meta[base],
                    device_id_type=pl.DeviceIdType.LOGICAL,
                ),
            ).start()
            state[1 + stream] = step + 1
            state[1 + _RING_STREAMS + stream] = 0


def _latent_ready_progress(
    payload: _PallasRef,
    output: _PallasRef,
    *state_refs: _PallasRef,
    rank: jax.Array,
    drain: bool = False,
) -> None:
    """Advance ready streams without blocking unrelated compute or streams.

    A receive consumes both DMA completion counts. Credits protect the two
    alternating receive semaphores from reuse before the peer consumes them.
    The final drain completes every stream before routed experts read HBM.
    """
    send, recv, credit_sems, stores, state, meta = state_refs
    rows = payload.shape[0]

    def sweep() -> None:
        """Poll all 12 streams once."""
        begin = 0
        for stream in range(_RING_STREAMS):
            size = rows // _RING_STREAMS + int(stream < rows % _RING_STREAMS)
            start = begin
            begin += size
            base = (stream % _DIRECTED_RINGS) * _RING_META_STRIDE
            # TPUv7 DMA semaphore accounting: one unit per 32 logical bytes.
            units = size * 2 * _LATENT_HALF_COLS * 2 // _DMA_BYTES_PER_UNIT
            _poll_latent_ready_stream(
                output,
                send,
                recv,
                credit_sems,
                state,
                meta,
                stream_info=(stream, start, size, base, units),
            )

    if drain:

        def body(i: jax.Array) -> jax.Array:
            """Poll streams until all 12 reach completion."""
            sweep()
            return i + 1

        jax.lax.while_loop(lambda _i: state[0] != 0, body, jnp.int32(0))
        cast(
            "_AsyncCopy",
            pltpu.make_async_copy(
                payload, output.at[pl.ds(rank * rows, rows), :, :], stores
            ),
        ).wait()
    else:

        @pl.when(state[0] != 0)
        def progress() -> None:
            """Advance any ready streams without blocking."""
            sweep()


def _stripe_validity(
    rows: int,
    length: jax.Array,
    rank: jax.Array,
) -> jax.Array:
    """Return a boolean mask of valid sequence-parallel rows owned by ``rank``."""
    row = jax.lax.broadcasted_iota(jnp.int32, (rows, 1), 0)
    stripe = min(rows, _MAX_STRIPE_ROWS)
    token_pos = row // stripe * (TP_SIZE * stripe) + rank * stripe + row % stripe
    return token_pos < length


def _prologue_setup_peer_and_router(
    s: dict[str, object],
    hidden: _PallasRef,
    router: _PallasRef,
    bias: _PallasRef,
    rank: jax.Array,
) -> tuple[
    _RemoteCopy,
    tuple[_PallasRef, ...],
    tuple[_PallasRef, ...],
    tuple[_PallasRef, ...],
    tuple[_PallasRef, ...],
    jax.Array,
    _Top16,
]:
    """Load local hidden/router/bias, start TP2 peer exchange, and initialize Top16."""
    rows = hidden.shape[0]
    parity, peer = rank % 2, rank ^ 1
    pair_ref = cast("_PallasRef", s["pair"])
    input_sem = cast("_PallasRef", s["input_sem"])
    own = pair_ref.at[pl.ds(parity * rows, rows), :]
    copy = cast("_AsyncCopy", pltpu.make_async_copy(hidden, own, input_sem))
    copy.start()
    copy.wait()
    if rows <= _VMEM_GATHER_MAX_ROWS:
        router_buf = cast("_PallasRef", s["router"])
        copy = cast("_AsyncCopy", pltpu.make_async_copy(router, router_buf, input_sem))
        copy.start()
        copy.wait()
    bias_buf = cast("_PallasRef", s["bias"])
    copy = cast("_AsyncCopy", pltpu.make_async_copy(bias, bias_buf, input_sem))
    copy.start()
    copy.wait()
    _enter_mesh()
    peer_credit = cast("_PallasRef", s["peer_credit"])
    pl.semaphore_signal(
        peer_credit, 1, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL
    )
    pl.semaphore_wait(peer_credit, 1)
    peer_input = cast(
        "_RemoteCopy",
        pltpu.make_async_remote_copy(
            own,
            own,
            s["peer_send"],
            s["peer_recv"],
            device_id=peer,
            device_id_type=pl.DeviceIdType.LOGICAL,
        ),
    )
    peer_input.start()
    gu_sems = cast("tuple[_PallasRef, ...]", s["gu_sems"])
    if rows == _ROWS_8K:
        router_gu = cast("_PallasRef", s["router_gu"])
        buffers = tuple(
            router_gu.at[:, pl.ds(i * _PANEL_COLS, _PANEL_COLS)]
            for i in range(_PAIR_BANKS)
        )
        router_v = router_gu.at[:, :_ROUTER_COLS]
        copy = cast("_AsyncCopy", pltpu.make_async_copy(router, router_v, gu_sems[0]))
        copy.start()
        copy.wait()
    else:
        buffers = cast("tuple[_PallasRef, ...]", s["gu_buffers"])
        router_v = cast("_PallasRef", s["router"])
    gate_buffers, up_buffers = buffers[:2], buffers[2:]
    gate_sems, up_sems = gu_sems[:2], gu_sems[2:]
    hx = own[...]
    logits = jnp.matmul(hx, router_v[...], preferred_element_type=jnp.float32)
    top = _Top16(logits, bias_buf[...])
    return peer_input, gate_buffers, up_buffers, gate_sems, up_sems, hx, top


def _prologue_shared_gu(
    s: dict[str, object],
    shared_gu: _PallasRef,
    *buffers_and_sems: tuple[_PallasRef, ...],
    rows: int,
    callbacks: tuple[Callable[[], None], Callable[..., None]],
) -> None:
    """Compute the shared-expert gate/up SiTU panels while advancing collectives."""
    gate_buffers, up_buffers, gate_sems, up_sems = buffers_and_sems
    metadata_progress, latent_progress = callbacks
    pair_ref = cast("_PallasRef", s["pair"])
    middle_ref = cast("_PallasRef", s["middle"])

    def gu_copy(
        panel: int,
        offset: int,
        bufs: Sequence[_PallasRef],
        sems: Sequence[_PallasRef],
    ) -> _AsyncCopy:
        """Create the async DMA copy descriptor for shared gate/up ``panel``."""
        bank = panel % 2
        return cast(
            "_AsyncCopy",
            pltpu.make_async_copy(
                shared_gu.at[:, pl.ds(offset + panel * _PANEL_COLS, _PANEL_COLS)],
                bufs[bank],
                sems[bank],
            ),
        )

    gu_copy(0, 0, gate_buffers, gate_sems).start()
    gu_copy(0, _SHARED_INTERMEDIATE_COLS, up_buffers, up_sems).start()
    if rows == _ROWS_8K:
        latent_progress()
    for panel in range(_SHARED_GU_PANELS):
        if rows <= _VMEM_GATHER_MAX_ROWS:
            latent_progress()
        bank = panel % 2
        gu_copy(panel, 0, gate_buffers, gate_sems).wait()
        gu_copy(panel, _SHARED_INTERMEDIATE_COLS, up_buffers, up_sems).wait()
        if panel + 1 < _SHARED_GU_PANELS:
            gu_copy(panel + 1, 0, gate_buffers, gate_sems).start()
            gu_copy(panel + 1, _SHARED_INTERMEDIATE_COLS, up_buffers, up_sems).start()
        for row in (0, rows):
            lhs = pair_ref[pl.ds(row, rows), :]
            gate = jnp.matmul(
                lhs, gate_buffers[bank][...], preferred_element_type=jnp.float32
            )
            up = jnp.matmul(
                lhs, up_buffers[bank][...], preferred_element_type=jnp.float32
            )
            metadata_progress()
            activated = (_SITU_GATE_SCALE * jnp.tanh(gate / _SITU_GATE_SCALE)) * (
                1.0 / (1.0 + jnp.exp(-gate))
            )
            activated *= _SITU_UP_SCALE * jnp.tanh(up / _SITU_UP_SCALE)
            middle_ref[pl.ds(row, rows), pl.ds(panel * _PANEL_COLS, _PANEL_COLS)] = (
                activated.astype(jnp.bfloat16)
            )
            if rows == _ROWS_8K:
                latent_progress()


def _prologue_shared_down(
    s: dict[str, object],
    shared_down: _PallasRef,
    shared: _PallasRef,
    rank_and_rows: tuple[jax.Array, int],
    callbacks: tuple[Callable[[], None], Callable[..., None]],
) -> None:
    """Compute shared-expert down-projection panels and TP2 peer reduction."""
    rank, rows = rank_and_rows
    parity, peer = rank % 2, rank ^ 1
    metadata_progress, latent_progress = callbacks
    down_buffers = cast("tuple[_PallasRef, ...]", s["down_buffers"])
    down_sems = cast("tuple[_PallasRef, ...]", s["down_sems"])
    partial_bufs = cast("tuple[_PallasRef, ...]", s["partial"])
    peer_y_bufs = cast("tuple[_PallasRef, ...]", s["peer_y"])
    reduce_send = cast("tuple[_PallasRef, ...]", s["reduce_send"])
    reduce_recv = cast("tuple[_PallasRef, ...]", s["reduce_recv"])
    middle_ref = cast("_PallasRef", s["middle"])

    def down_copy(panel: int) -> _AsyncCopy:
        """Create the async DMA copy descriptor for shared down ``panel``."""
        return cast(
            "_AsyncCopy",
            pltpu.make_async_copy(
                shared_down.at[:, pl.ds(panel * _PANEL_COLS, _PANEL_COLS)],
                down_buffers[panel % 2],
                down_sems[panel % 2],
            ),
        )

    def finish_reduction(pending_item: tuple[_RemoteCopy, int, int]) -> None:
        """Complete TP2 peer reduction for a finished shared down panel."""
        transfer, bank, panel = pending_item
        own = partial_bufs[bank][pl.ds(parity * rows, rows), :]
        transfer.wait_send()
        transfer.wait_recv()
        shared[:, pl.ds(panel * _PANEL_COLS, _PANEL_COLS)] = (
            own + peer_y_bufs[bank][...]
        ).astype(jnp.bfloat16)

    down_copy(0).start()
    pending: tuple[_RemoteCopy, int, int] | None = None
    for panel in range(_SHARED_DOWN_PANELS):
        bank = panel % 2
        down_copy(panel).wait()
        if panel + 1 < _SHARED_DOWN_PANELS:
            down_copy(panel + 1).start()
        for row in (0, rows):
            product = jnp.matmul(
                middle_ref[pl.ds(row, rows), :],
                down_buffers[bank][...],
                preferred_element_type=jnp.float32,
            )
            metadata_progress()
            partial_bufs[bank][pl.ds(row, rows), :] = product
            if row == rows:
                if pending is not None:
                    finish_reduction(pending)
                transfer = cast(
                    "_RemoteCopy",
                    pltpu.make_async_remote_copy(
                        partial_bufs[bank].at[pl.ds((1 - parity) * rows, rows), :],
                        peer_y_bufs[bank],
                        reduce_send[bank],
                        reduce_recv[bank],
                        device_id=peer,
                        device_id_type=pl.DeviceIdType.LOGICAL,
                    ),
                )
                transfer.start()
                pending = (transfer, bank, panel)
            if rows in (_ROWS_8K, row):
                latent_progress()
    if pending is not None:
        finish_reduction(pending)


def _store_prologue_latent_panel(
    s: dict[str, object],
    rows: int,
    col: int,
    product: jax.Array,
    first_half: jax.Array | None,
) -> jax.Array | None:
    """Store one projected latent panel into VMEM and return paired half carry."""
    fragment = product.astype(jnp.bfloat16)
    if rows != _ROWS_8K:
        cast("_PallasRef", s["latent"])[:, pl.ds(col, _PANEL_COLS)] = fragment
        return first_half
    if col < _LATENT_HALF_COLS:
        return fragment
    if first_half is None:
        msg = "Expected first latent half before second half"
        raise ValueError(msg)
    paired = jnp.concatenate((first_half, fragment), axis=1)
    latent_payload = cast("_PallasRef", s["latent_payload"])
    latent_payload[:, :, pl.ds(col - _LATENT_HALF_COLS, _PANEL_COLS)] = paired.reshape(
        rows, 2, _PANEL_COLS
    )
    return first_half


def _start_prologue_route_gather(
    s: dict[str, object],
    top: _Top16,
    valid: jax.Array,
    rows: int,
    rank: jax.Array,
) -> Iterator[int]:
    """Finalize Top-16 selection, pack wire payload, and start route gather."""
    ids, probs = top.finish(valid)
    probs_bits = jax.lax.bitcast_convert_type(probs, jnp.int32)
    wire = (
        jnp.concatenate((ids, probs_bits), axis=1)
        if rows < _VMEM_GATHER_MAX_ROWS
        else jnp.concatenate(
            (
                ids.reshape(rows // SUBLANE_ROWS, VREG_LANE_COUNT),
                probs_bits.reshape(rows // SUBLANE_ROWS, VREG_LANE_COUNT),
            ),
            axis=0,
        )
    )
    route_payload = cast("_PallasRef", s["route_payload"])
    route_payload[...] = wire.reshape(rows // _PAIR_BANKS, VREG_LANE_COUNT)
    route_steps = _route_gather_steps(
        route_payload,
        cast("_PallasRef", s["route_wire"]),
        cast("_PallasRef", s["route_send"]),
        cast("_PallasRef", s["route_recv"]),
        rank=rank,
    )
    next(route_steps, None)
    return route_steps


def _prologue_run_latent_and_routes(
    s: dict[str, object],
    latent_products: Iterator[tuple[int, jax.Array]],
    top_and_valid: tuple[_Top16, jax.Array],
    *,
    rows_and_rank: tuple[int, jax.Array],
    metadata_fns: tuple[Callable[[], None], Callable[[], None]],
) -> None:
    """Overlap latent projection panels with Top-16 selection and route gather."""
    top, valid = top_and_valid
    rows, rank = rows_and_rank
    start_metadata, metadata_progress = metadata_fns
    route_steps: Iterator[int] = iter(())
    route_started = False
    first_half: jax.Array | None = None
    for col, product in latent_products:
        route_stage = next(route_steps, _ROUTE_DONE) if route_started else _ROUTE_DONE
        if (
            route_started
            and route_stage is _ROUTE_DONE
            and rows <= _VMEM_GATHER_MAX_ROWS
        ):
            start_metadata()
        if rows <= _VMEM_GATHER_MAX_ROWS:
            metadata_progress()
        first_half = _store_prologue_latent_panel(s, rows, col, product, first_half)
        for _ in range(min(2, _TOP_K - top.index)):
            top.step()
        if top.index == _TOP_K and not route_started:
            route_steps = _start_prologue_route_gather(s, top, valid, rows, rank)
            route_started = True
    for _ in route_steps:
        pass
    start_metadata()
    if rows <= _VMEM_GATHER_MAX_ROWS:
        latent_src = cast("_PallasRef", s["latent"])
        latent_payload = cast("_PallasRef", s["latent_payload"])
        latent_payload[...] = latent_src[:, :_LATENT_FULL_COLS].reshape(
            rows, 2, _LATENT_HALF_COLS
        )


def _make_prologue_latent_progress(
    s: dict[str, object],
    tables: _PallasRef,
    latent_hbm: _PallasRef,
    rows: int,
    rank: jax.Array,
) -> Callable[..., None]:
    """Create the ``latent_progress`` closure for ``_prologue``."""
    latent_payload = cast("_PallasRef", s["latent_payload"])
    latent_steps = _latent_gather_steps(
        tables,
        latent_payload,
        cast("_PallasRef", s["latent_recv"]),
        latent_hbm,
        cast("_PallasRef", s["latent_send_sems"]),
        cast("_PallasRef", s["latent_recv_sems"]),
        cast("_PallasRef", s["latent_credits"]),
        cast("_PallasRef", s["latent_stores"]),
        rank=rank,
    )
    latent_tick = 0

    def latent_progress(*, drain: bool = False) -> None:
        """Advance the active latent ring implementation."""
        nonlocal latent_tick
        if rows != _ROWS_8K:
            if drain:
                for _ in latent_steps:
                    pass
            else:
                next(latent_steps, None)
            return
        if latent_tick == 0:
            _latent_ready_init(
                tables,
                latent_payload,
                latent_hbm,
                cast("_PallasRef", s["latent_send_sems"]),
                cast("_PallasRef", s["latent_recv_sems"]),
                cast("_PallasRef", s["latent_stores"]),
                cast("_PallasRef", s["latent_state"]),
                cast("_PallasRef", s["latent_meta"]),
                rank=rank,
            )
        else:
            _latent_ready_progress(
                latent_payload,
                latent_hbm,
                cast("_PallasRef", s["latent_send_sems"]),
                cast("_PallasRef", s["latent_recv_sems"]),
                cast("_PallasRef", s["latent_credits"]),
                cast("_PallasRef", s["latent_stores"]),
                cast("_PallasRef", s["latent_state"]),
                cast("_PallasRef", s["latent_meta"]),
                rank=rank,
                drain=drain,
            )
        latent_tick += 1

    return latent_progress


def _drain_prologue_collectives(
    rows: int,
    metadata_steps: Iterator[None],
    metadata_ready: Callable[[], None],
    latent_progress: Callable[..., None],
) -> None:
    """Drain remaining latent ring and metadata sorting work at prologue end."""
    if rows == _ROWS_8K:
        for _ in metadata_steps:
            latent_progress()
        # Start compact VMEM-to-SMEM transfers while the latent ring finishes.
        metadata_ready()
        latent_progress(drain=True)
    else:
        latent_progress(drain=True)
        for _ in metadata_steps:
            pass


def _prologue_body(
    s: dict[str, object],
    *prologue_args: object,
) -> None:
    """Execute the scoped prologue projections, routing, and ring collectives."""
    (
        tables,
        hidden,
        router,
        bias,
        latent_weight,
        shared_gu,
        shared_down,
        shared,
        latent_hbm,
        token_ids,
        probabilities,
        counts,
        maximum,
        rank,
        valid,
        counts_ready,
        metadata_ready,
    ) = cast("_PrologueArgsTuple", prologue_args)
    rows = hidden.shape[0]
    (
        peer_input,
        gate_buffers,
        up_buffers,
        gate_sems,
        up_sems,
        hx,
        top,
    ) = _prologue_setup_peer_and_router(s, hidden, router, bias, rank)
    metadata_steps: Iterator[None] = iter(())
    metadata_started = False

    def start_metadata() -> None:
        """Initialize the metadata generator once route exchange completes."""
        nonlocal metadata_started, metadata_steps
        if not metadata_started:
            metadata_started = True
            metadata_steps = _metadata_steps(
                cast("_PallasRef", s["route_wire"]),
                cast("_PallasRef", s["keys"]),
                probabilities,
                token_ids,
                counts,
                maximum,
                tokens_and_rank=(rows * TP_SIZE, rank),
                counts_ready=counts_ready,
            )

    def metadata_progress() -> None:
        """Advance a bounded number of metadata prep/sort chunks."""
        if not metadata_started:
            return
        for _ in range(2 if rows <= _VMEM_GATHER_MAX_ROWS else 6):
            next(metadata_steps, None)

    if rows == _ROWS_8K:
        router_gu = cast("_PallasRef", s["router_gu"])
        latent_buffers = tuple(
            router_gu.at[:, pl.ds(i * _PANEL_COLS, _PANEL_COLS)]
            for i in range(_PAIR_BANKS)
        )
        latent_products = _latent_half_pair_panels(
            hx,
            latent_weight,
            latent_buffers,
            cast("tuple[_PallasRef, ...]", s["gu_sems"]),
        )
    else:
        latent_products = _dense_panels(
            hx,
            latent_weight,
            gate_buffers,
            gate_sems,
            columns=_LATENT_PADDED_COLS,
        )
    _prologue_run_latent_and_routes(
        s,
        latent_products,
        (top, valid),
        rows_and_rank=(rows, rank),
        metadata_fns=(start_metadata, metadata_progress),
    )
    latent_progress = _make_prologue_latent_progress(s, tables, latent_hbm, rows, rank)
    peer_input.wait_send()
    peer_input.wait_recv()
    _prologue_shared_gu(
        s,
        shared_gu,
        gate_buffers,
        up_buffers,
        gate_sems,
        up_sems,
        rows=rows,
        callbacks=(metadata_progress, latent_progress),
    )
    _prologue_shared_down(
        s,
        shared_down,
        shared,
        (rank, rows),
        (metadata_progress, latent_progress),
    )
    _drain_prologue_collectives(rows, metadata_steps, metadata_ready, latent_progress)


def _prologue(
    tables: _PallasRef,
    control: _PallasRef,
    hidden: _PallasRef,
    *weight_and_out_refs: _PallasRef | None,
    counts_ready: Callable[[], None],
    metadata_ready: Callable[[], None],
) -> None:
    """Router, latent projection and shared experts, with incremental collectives.

    The caller retains shared/metadata outputs across this scope. All temporary
    projection and communication buffers die before routed-expert scratch is
    allocated. The initial HBM input load targets our half of the peer buffer;
    both the router and shared expert use this same copy.

    Invariants:
        assert first_half is not None
    """
    (
        router_in,
        bias_in,
        latent_weight_in,
        shared_gu_in,
        shared_down_in,
        shared_in,
        latent_hbm_in,
        token_ids_in,
        probabilities_in,
        counts_in,
        maximum,
    ) = weight_and_out_refs
    rows, hidden_size = hidden.shape
    rank = jax.lax.axis_index("k3_sp")
    # Dense projections retain global family indices. Resident expert banks correspond
    # to first/group/final programs with 3/8/1 MoE families.
    family = control[1]
    router = cast("_PallasRef", router_in).at[family]
    bias = cast("_PallasRef", bias_in).at[family]
    latent_weight = cast("_PallasRef", latent_weight_in).at[family]
    shared_gu = cast("_PallasRef", shared_gu_in).at[family]
    shared_down = cast("_PallasRef", shared_down_in).at[family]
    valid = _stripe_validity(rows, control[0], rank)
    vmem, dma = pltpu.VMEM, pltpu.SemaphoreType.DMA

    scratch = {
        "pair": vmem((2 * rows, hidden_size), jnp.bfloat16),
        "router": (
            vmem((hidden_size, _ROUTER_COLS), jnp.bfloat16)
            if rows <= _VMEM_GATHER_MAX_ROWS
            else None
        ),
        "router_gu": (
            vmem((hidden_size, _ROUTER_GU_COLS), jnp.bfloat16)
            if rows == _ROWS_8K
            else None
        ),
        "gu_buffers": (
            tuple(
                vmem((hidden_size, _PANEL_COLS), jnp.bfloat16)
                for _ in range(_PAIR_BANKS)
            )
            if rows <= _VMEM_GATHER_MAX_ROWS
            else ()
        ),
        "gu_sems": tuple(dma for _ in range(_PAIR_BANKS)),
        "bias": vmem((1, _ROUTER_COLS), jnp.float32),
        "input_sem": dma,
        "peer_send": dma,
        "peer_recv": dma,
        "peer_credit": pltpu.SemaphoreType.REGULAR,
        "latent": (
            vmem((rows, _LATENT_PADDED_COLS), jnp.bfloat16)
            if rows <= _VMEM_GATHER_MAX_ROWS
            else None
        ),
        "route_payload": vmem((rows // _PAIR_BANKS, VREG_LANE_COUNT), jnp.int32),
        "route_wire": vmem((rows * SUBLANE_ROWS, VREG_LANE_COUNT), jnp.int32),
        "route_send": dma((_ROUTE_GATHER_STAGES,)),
        "route_recv": dma((_ROUTE_GATHER_STAGES,)),
        "keys": vmem((rows * _PAIR_BANKS, VREG_LANE_COUNT), jnp.int32),
        "latent_payload": vmem((rows, 2, _LATENT_HALF_COLS), jnp.bfloat16),
        "latent_recv": (
            vmem((2, rows, 2, _LATENT_HALF_COLS), jnp.bfloat16)
            if rows <= _VMEM_GATHER_MAX_ROWS
            else None
        ),
        "latent_state": (
            pltpu.SMEM((1 + 2 * _RING_STREAMS,), jnp.int32)
            if rows == _ROWS_8K
            else None
        ),
        "latent_meta": (
            pltpu.SMEM((_DIRECTED_RINGS * _RING_META_STRIDE,), jnp.int32)
            if rows == _ROWS_8K
            else None
        ),
        "latent_send_sems": dma((_RING_STREAMS,)),
        "latent_recv_sems": dma((2, _RING_STREAMS)),
        "latent_credits": pltpu.SemaphoreType.REGULAR((_RING_STREAMS,)),
        "latent_stores": (
            dma((_RING_STREAMS,)) if rows <= _VMEM_GATHER_MAX_ROWS else dma
        ),
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
    pl.run_scoped(
        lambda s: _prologue_body(
            s,
            tables,
            hidden,
            router,
            bias,
            latent_weight,
            shared_gu,
            shared_down,
            shared_in,
            latent_hbm_in,
            token_ids_in,
            probabilities_in,
            counts_in,
            maximum,
            rank,
            valid,
            counts_ready,
            metadata_ready,
        ),
        scratch,
    )


@dataclasses.dataclass(frozen=True)
class _RoutedExpertContext:
    """Immutable references and tile parameters for ``_routed_experts``."""

    cfg: k.GmmConfigs
    gu_cfg: k.GmmConfigs
    metadata: k.MetadataRef
    destinations: _FlatMetadata
    probabilities: _FlatMetadata
    latent: _PallasRef
    middle: _PallasRef
    out: _PallasRef
    packed: _PallasRef
    sums: _PallasRef
    cast_buf: _PallasRef
    tables: _PallasRef
    sends: _PallasRef
    receives: _PallasRef
    credit_sems: _PallasRef
    stores: _PallasRef
    payload: _PallasRef
    rx: _PallasRef
    state: _PallasRef
    ring_meta: _PallasRef
    regions: Sequence[tuple[int, ...]]
    tokens: int
    tm: int
    tn: int
    width: int
    sub: int
    packed_half: int
    unroll: int
    gu_unroll: int
    first_down_unroll: int
    probabilities_ready: Callable[[], None]


def _build_routed_gmm_configs(
    latent_dtype: jnp.dtype,
    gu_pair: tuple[_PallasRef, _PallasRef],
    down_pair: tuple[_PallasRef, _PallasRef],
    *,
    counts_and_offset: tuple[_PallasRef, _PallasRef],
    tokens: int,
) -> tuple[k.GmmConfigs, k.GmmConfigs, int, int, int, int]:
    """Construct down and gate/up GMM configurations for ``_routed_experts``.

    Invariants:
        assert tokens in (512, 1024, 2048, 4096, 8192)
        assert rows <= x.shape[0]
        assert width % tn == 0
        assert gu_cfg.dims.size_lhs_sublane == sub
        assert config.rhs_cfgs.should_dequantize_before_matmul
        assert not config.rhs_cfgs.should_dequantize_after_matmul
    """
    gu_weights, gu_scales = gu_pair
    rhs, scales = down_pair
    counts, offset = counts_and_offset
    if tokens not in _SUPPORTED_WINDOWS:
        msg = f"Unsupported routed_experts token window: {tokens}"
        raise ValueError(msg)
    columns = (
        _DEFAULT_DOWN_COLUMNS
        if tokens < OVERFLOW_WINDOW_TOKENS
        else _COMPACT_DOWN_COLUMNS
    )
    tile = _DEFAULT_DOWN_TILE if tokens < OVERFLOW_WINDOW_TOKENS else _COMPACT_DOWN_TILE
    metadata_row_capacity = (
        0 if tokens < OVERFLOW_WINDOW_TOKENS else _COMPACT_METADATA_PREFIX_ROWS
    )
    x_shape = (tokens * _TOP_K, _EXPERT_INTERMEDIATE_COLS)
    rows, width = (metadata_row_capacity or x_shape[0], rhs.shape[-1])
    if rows > x_shape[0] or width % columns != 0:
        msg = f"Invalid rows={rows} or width={width} for columns={columns}"
        raise ValueError(msg)
    limit = int(pltpu.get_tpu_info().vmem_capacity_bytes * _VMEM_LIMIT_FRACTION)
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
        cfg, tiles=dataclasses.replace(cfg.tiles, tile_m=tile, tile_n=columns)
    )
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
            tile_m=tile,
            tile_n=_EXPERT_INTERMEDIATE_COLS,
            tile_k=_LATENT_FULL_COLS,
        ),
    )
    sub = cfg.dims.size_lhs_sublane
    if gu_cfg.dims.size_lhs_sublane != sub:
        msg = "Gate/up and down sublane tiling mismatch"
        raise ValueError(msg)
    for config in (cfg, gu_cfg):
        if (
            not config.rhs_cfgs.should_dequantize_before_matmul
            or config.rhs_cfgs.should_dequantize_after_matmul
        ):
            msg = "Routed experts require pre-matmul MXFP4 dequantization"
            raise ValueError(msg)
    max_gm = cfg.dims.size_group + pl.cdiv(rows, tile) - 1
    return cfg, gu_cfg, tile, columns, width, max_gm


def _pad_and_run_first_column(
    joint: Callable[..., None],
    spec_args: tuple[tuple[int | jax.Array, ...], tuple[object, ...], object],
    refs: tuple[object, ...],
    scratches: tuple[object, ...],
) -> None:
    """Skew physical GU VMEM stride by +256 columns and run ``joint``."""
    grid, in_specs, out_specs = spec_args
    pipeline, allocate = pltpu.emit_pipeline_with_allocations(
        joint, grid=grid, in_specs=in_specs, out_specs=out_specs
    )
    allocations = allocate(*refs)
    logical_gu = allocations[0]

    def is_buffer(value: object) -> bool:
        """Return True if ``value`` is a pipeline buffer allocation."""
        return isinstance(value, pltpu.BufferedRef)

    def pad(buffer: _WindowedBuffer) -> _WindowedBuffer:
        """Pad the trailing dimension by 256 columns to skew VMEM banks."""
        window = buffer.window_ref
        if window is None:
            return buffer
        return dataclasses.replace(
            buffer,
            window_ref=cast(
                "_PallasRef",
                pltpu.VMEM(
                    (*window.shape[:-1], window.shape[-1] + _PANEL_COLS), window.dtype
                ),
            ),
        )

    padded_gu = jax.tree.map(pad, logical_gu, is_leaf=is_buffer)

    def run_pipeline(buffers: Sequence[object]) -> None:
        """Slice padded VMEM allocations back to logical tile shapes and run."""

        def trim(buffer: _WindowedBuffer, original: _WindowedBuffer) -> _WindowedBuffer:
            """Restore the original trailing dimension view on ``buffer``."""
            if buffer.window_ref is None or original.window_ref is None:
                return buffer
            return dataclasses.replace(
                buffer,
                window_ref=buffer.window_ref.at[
                    ..., pl.ds(0, original.window_ref.shape[-1])
                ],
            )

        selected = (
            jax.tree.map(trim, buffers[0], logical_gu, is_leaf=is_buffer),
            *buffers[1:],
        )
        pipeline(*refs, allocations=selected, scratches=scratches)

    pl.run_scoped(run_pipeline, (padded_gu, *allocations[1:]))


def _routed_tile_bounds(
    ctx: _RoutedExpertContext,
    g: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute sublane-aligned row start and count for tile ``g``."""
    begin = ctx.metadata.gm_id_to_m_offset[g] // ctx.sub * ctx.sub
    end = pl.cdiv(ctx.metadata.gm_id_to_m_offset[g + 1], ctx.sub) * ctx.sub
    return (begin, end - begin)


def _start_routed_tile_acts(
    ctx: _RoutedExpertContext,
    acts: _PallasRef,
    act_sems: _PallasRef,
    g: jax.Array,
    bank: jax.Array,
) -> None:
    """Prefetch sorted activation rows for tile ``g`` into ``bank``."""
    begin, nrows = _routed_tile_bounds(ctx, g)

    @pl.loop(0, nrows)
    def row(i: jax.Array) -> None:
        """Issue one single-row async DMA copy."""
        source = ctx.destinations[begin + i]
        cast(
            "_AsyncCopy",
            pltpu.make_async_copy(
                ctx.latent.at[pl.ds(source, 1)],
                acts.at[bank, pl.ds(i, 1)],
                act_sems.at[bank],
            ),
        ).start()


def _make_routed_first_column_joint(
    ctx: _RoutedExpertContext,
    acts: _PallasRef,
    act_sems: _PallasRef,
    *,
    ng_pair: tuple[jax.Array, jax.Array],
    callbacks: tuple[Callable[..., None], Callable[[jax.Array, int], None]],
) -> Callable[..., None]:
    """Create the ``joint`` callback for the first routed-expert column."""
    active_ng, ng = ng_pair
    inner_fn, gu_work_fn = callbacks

    def joint(
        gw: k.FusedWeightsRef,
        dw: k.WeightsRef,
        gy: _PallasRef,
        *joint_scratches: object,
    ) -> None:
        """Run one group tile of the joint GU + first down column."""
        metadata_s, gu_partial_s, acts_s, act_sems_s = joint_scratches
        metadata_ref = cast("k.MetadataRef", metadata_s)
        gu_partial_ref = cast("_PallasRef", gu_partial_s)
        acts_ref = cast("_PallasRef", acts_s)
        act_sems_ref = cast("_PallasRef", act_sems_s)
        g = pl.program_id(1)
        bank = g % 2

        @pl.when(g == 0)
        def initialize_accumulation() -> None:
            """Drain <=4K probability DMA and zero initial accumulators."""
            ctx.probabilities_ready()
            ctx.sums[...] = jnp.zeros(ctx.sums.shape, jnp.float32)
            ctx.packed.at[1][...] = jnp.zeros(ctx.packed.at[1].shape, ctx.packed.dtype)

        @pl.when(active_ng > 0)
        def gate_up() -> None:
            """Prefetch activations and execute the gate/up SiTU tile."""

            @pl.when(g == 0)
            def initial() -> None:
                """Start the first tile's activation gather."""
                _start_routed_tile_acts(ctx, acts, act_sems, g, bank)

            _, nrows = _routed_tile_bounds(ctx, g)
            active = acts_ref.at[bank, pl.ds(0, nrows)]
            cast(
                "_AsyncCopy",
                pltpu.make_async_copy(active, active, act_sems_ref.at[bank]),
            ).wait()

            @pl.when(g + 1 < ng)
            def following() -> None:
                """Prefetch the next tile's activations into the other bank."""
                _start_routed_tile_acts(ctx, acts, act_sems, g + 1, 1 - bank)

            _gu_inner(
                PackedRows(acts_ref.at[bank]),
                gw,
                gy,
                gu_partial_ref,
                metadata_ref,
                cfgs=ctx.gu_cfg,
                overlap_work=lambda panel: gu_work_fn(g, panel),
            )

        inner_fn(gy, dw, metadata_ref, first=True)

    return joint


def _run_routed_first_column(
    ctx: _RoutedExpertContext,
    weights_triple: tuple[k.FusedWeightsRef, k.WeightsRef, k.WeightsRef],
    *,
    ng_pair: tuple[jax.Array, jax.Array],
    callbacks: tuple[Callable[..., None], Callable[[jax.Array, int], None]],
) -> None:
    """Execute the joint gate/up + first down-column pipeline."""
    gu, w, ws = weights_triple
    _, ng = ng_pair
    (_, gu_ws_single), gu_ys = k.generate_block_specs(ctx.metadata, ctx.gu_cfg)
    gu_ws = k.FusedWeightsRef(gate=gu_ws_single, up=gu_ws_single)

    @functools.partial(
        pl.run_scoped,
        gu_partial=pltpu.VMEM(
            (ctx.sub, _EXPERT_INTERMEDIATE_COLS), ctx.gu_cfg.out_dtype
        ),
        acts=pltpu.VMEM((2, ctx.tm, 2, _LATENT_HALF_COLS), ctx.latent.dtype),
        act_sems=pltpu.SemaphoreType.DMA((2,)),
    )
    def first_column(
        gu_partial: _PallasRef,
        acts: _PallasRef,
        act_sems: _PallasRef,
    ) -> None:
        """Run the double-buffered joint first-column pipeline."""

        def buffering(spec: pl.BlockSpec) -> pl.BlockSpec:
            """Override pipeline buffering to 2 slots."""
            return dataclasses.replace(spec, pipeline_mode=pl.Buffered(buffer_count=2))

        joint_gu_ws, joint_down_ws = jax.tree.map(
            buffering, (gu_ws, ws), is_leaf=lambda v: isinstance(v, pl.BlockSpec)
        )
        joint = _make_routed_first_column_joint(
            ctx, acts, act_sems, ng_pair=ng_pair, callbacks=callbacks
        )
        grid = (1, ng, 1)
        in_specs = (joint_gu_ws, joint_down_ws)
        refs = (gu, w, ctx.middle.reshape(-1, ctx.sub, _EXPERT_INTERMEDIATE_COLS))
        scratches = (ctx.metadata, gu_partial, acts, act_sems)
        if ctx.tokens == OVERFLOW_WINDOW_TOKENS:
            pltpu.emit_pipeline(joint, grid=grid, in_specs=in_specs, out_specs=gu_ys)(
                *refs, scratches=scratches
            )
            return
        _pad_and_run_first_column(joint, (grid, in_specs, gu_ys), refs, scratches)


def _init_routed_ring_meta(
    tables_ref: _PallasRef,
    ring_meta: _PallasRef,
    localrows: int,
) -> None:
    """Initialize SMEM neighbor IDs and 32 hop source-row bases across 6 rings."""
    # Six directed rings: [target, source, 32 local-contribution row bases]. One
    # initialization per layer call, reused across all four columns.
    rank = lax.axis_index("k3_sp")
    for directed in range(_DIRECTED_RINGS):
        family, direction = directed // 2, -1 if directed % 2 == 0 else 1
        position = tables_ref[family, 0, rank]
        ring_meta[directed * _RING_META_STRIDE] = tables_ref[
            family, 1, (position + direction) & _RING_MASK
        ]
        ring_meta[directed * _RING_META_STRIDE + 1] = tables_ref[
            family, 1, (position - direction) & _RING_MASK
        ]

        @pl.loop(0, TP_SIZE)
        def hop(
            step: jax.Array,
            *,
            ring_info: tuple[int, jax.Array, int, int] = (
                family,
                position,
                direction,
                directed,
            ),
        ) -> None:
            """Cache source-row base offsets for hop ``step`` in SMEM."""
            fam, pos, dir_step, dir_idx = ring_info
            owner_mesh = tables_ref[fam, 1, (pos - dir_step * (step + 1)) & _RING_MASK]
            owner = tables_ref[fam, 2, owner_mesh]
            ring_meta[dir_idx * _RING_META_STRIDE + 2 + step] = owner * localrows


def _emit_routed_subtotals(
    ctx: _RoutedExpertContext,
    subtotal_vmem: _PallasRef,
) -> None:
    """Convert FP32 token sums into BF16 ring subtotals."""

    @pl.loop(0, ctx.tokens // STORE_TILE)
    def block(b: jax.Array) -> None:
        """Trim padding and round one 128-token block to BF16."""
        # Trim the aligned-layout padding before the existing ring. The cut at tn=896 is
        # lane aligned, unlike a 448-wide half.
        values = ctx.sums[pl.ds(b * STORE_TILE, STORE_TILE)].reshape(
            STORE_TILE, 2 * ctx.packed_half
        )
        ctx.cast_buf[...] = values[:, : ctx.tn].astype(jnp.bfloat16)
        subtotal_vmem.at[pl.ds(b * STORE_TILE, STORE_TILE)][...] = ctx.cast_buf[...]


def _consume_routed_tile(
    ctx: _RoutedExpertContext,
    tile_g: jax.Array,
    source: _PallasRef,
    first: int = 0,
    *,
    loop_unroll: int,
) -> None:
    """Accumulate remaining rows of ``source`` for tile ``tile_g``."""
    start = ctx.metadata.gm_id_to_m_offset[tile_g]
    end = ctx.metadata.gm_id_to_m_offset[tile_g + 1]
    begin = start // ctx.sub * ctx.sub
    start = jnp.minimum(start + first, end)

    def update(i: jax.Array) -> None:
        """Weight and accumulate row ``i`` into its token sum."""
        dest, weight = (ctx.destinations[i], ctx.probabilities[i])
        value = source.at[pl.ds(i - begin, 1)][...].astype(jnp.float32)
        target = ctx.sums.at[pl.ds(dest, 1)]
        target[...] = target[...] + value * weight

    full = (end - start) // loop_unroll

    @pl.loop(0, full)
    def chunk(b: jax.Array) -> None:
        """Accumulate one unrolled batch of rows."""
        for u in range(loop_unroll):
            update(start + b * loop_unroll + u)

    @pl.loop(start + full * loop_unroll, end)
    def tail(i: jax.Array) -> None:
        """Accumulate remainder rows."""
        update(i)


def _routed_gu_work(
    ctx: _RoutedExpertContext,
    g: jax.Array,
    panel: int,
) -> None:
    """Overlap previous tile's VALU accumulation during gate/up panels."""
    previous_g = jnp.maximum(g - 1, 0)
    start = ctx.metadata.gm_id_to_m_offset[previous_g]
    end = ctx.metadata.gm_id_to_m_offset[previous_g + 1]
    begin = start // ctx.sub * ctx.sub
    for row in range(ctx.gu_unroll):
        row_offset = panel * ctx.gu_unroll + row
        i = jnp.minimum(start + row_offset, end - 1)
        dest, prob = (ctx.destinations[i], ctx.probabilities[i])
        weight = jnp.where((g > 0) & (start + row_offset < end), prob, 0)
        previous = ctx.packed.at[(g + 1) % 2]
        value = previous.at[pl.ds(i - begin, 1)][...].astype(jnp.float32)
        target = ctx.sums.at[pl.ds(dest, 1)]
        target[...] = target[...] + value * weight


def _routed_down_overlap_work(
    ctx: _RoutedExpertContext,
    previous: _PallasRef,
    g_pair: tuple[jax.Array, jax.Array],
    *,
    panel_and_rows: tuple[int, int, int],
) -> None:
    """Accumulate ``down_unroll`` rows of the previous tile during down panels."""
    g, previous_g = g_pair
    panel, base_moved_rows, down_unroll = panel_and_rows
    start = ctx.metadata.gm_id_to_m_offset[previous_g]
    end = ctx.metadata.gm_id_to_m_offset[previous_g + 1]
    begin = start // ctx.sub * ctx.sub
    for row in range(down_unroll):
        row_offset = base_moved_rows + panel * down_unroll + row
        i = jnp.minimum(start + row_offset, end - 1)
        dest, prob = (ctx.destinations[i], ctx.probabilities[i])
        weight = jnp.where((g > 0) & (start + row_offset < end), prob, 0)
        value = previous.at[pl.ds(i - begin, 1)][...].astype(jnp.float32)
        target = ctx.sums.at[pl.ds(dest, 1)]
        target[...] = target[...] + value * weight


def _finish_routed_non_first_column(
    ctx: _RoutedExpertContext,
    n: jax.Array,
    subtotal_vmem: _PallasRef | None,
    advance_fn: Callable[..., None],
) -> None:
    """Drain the preceding column's ring, emit subtotals, and drain final block."""
    subtotal_ref = cast("_PallasRef", subtotal_vmem)
    advance_fn(n - 1, subtotal_ref, drain=True)
    _emit_routed_subtotals(ctx, subtotal_ref)
    ctx.state[0] = 0

    @pl.when(n == ctx.width // ctx.tn - 1)
    def last() -> None:
        """Drain the ring for the final column block."""
        advance_fn(n, subtotal_ref, drain=True)


def _make_routed_inner(
    ctx: _RoutedExpertContext,
    active_ng: jax.Array,
    ng: jax.Array,
    advance_fn: Callable[..., None],
) -> Callable[..., None]:
    """Create the down-projection pipeline callback for ``_routed_experts``."""
    gu_panels = pl.cdiv(_SHARED_INTERMEDIATE_COLS, pltpu.get_tpu_info().mxu_column_size)
    moved_rows = gu_panels * ctx.gu_unroll
    panel_count = pl.cdiv(ctx.tn, pltpu.get_tpu_info().mxu_column_size)

    def inner(
        x_tile: _PallasRef,
        w_tile: k.WeightsRef,
        metadata_tile: k.MetadataRef,
        *,
        first: bool = False,
        subtotal_vmem: _PallasRef | None = None,
    ) -> None:
        """Execute one down-projection tile and overlap VALU/ring work."""
        g, n = (pl.program_id(1), jnp.int32(0) if first else pl.program_id(0) + 1)
        current = ctx.packed.at[g % 2]
        previous = ctx.packed.at[(g + 1) % 2]
        previous_g = jnp.maximum(g - 1, 0)
        down_unroll = ctx.first_down_unroll if first else ctx.unroll
        base_moved = moved_rows if first else 0
        tail_unroll = _DEFAULT_DOWN_UNROLL if first else ctx.unroll
        if not first:

            @pl.when(g == 0)
            def clear() -> None:
                """Zero the FP32 token accumulator and previous bank at column start."""
                ctx.sums[...] = jnp.zeros(ctx.sums.shape, jnp.float32)
                previous[...] = jnp.zeros(previous.shape, previous.dtype)

            advance_fn(n - 1, cast("_PallasRef", subtotal_vmem), poll_round=g)

        @pl.when(active_ng > 0)
        def product() -> None:
            """Compute the down-projection matmul tile."""
            _down_inner(
                x_tile,
                w_tile,
                current,
                metadata_tile,
                cfgs=ctx.cfg,
                overlap_work=lambda panel: _routed_down_overlap_work(
                    ctx,
                    previous,
                    (g, previous_g),
                    panel_and_rows=(panel, base_moved, down_unroll),
                ),
            )

        @pl.when(g > 0)
        def previous_tail() -> None:
            """Drain remaining rows of the previous tile."""
            _consume_routed_tile(
                ctx,
                previous_g,
                previous,
                base_moved + panel_count * down_unroll,
                loop_unroll=tail_unroll,
            )

        @pl.when(g == ng - 1)
        def finish() -> None:
            """Drain the final group tile and complete the column ring."""
            _consume_routed_tile(ctx, g, current, loop_unroll=tail_unroll)
            if not first:
                _finish_routed_non_first_column(ctx, n, subtotal_vmem, advance_fn)

    return inner


def _routed_experts_kernel(
    ctx: _RoutedExpertContext,
    counts: _PallasRef,
    offset: _PallasRef,
    layer: _PallasRef,
    weights_pair: tuple[k.WeightsRef, k.WeightsRef],
) -> None:
    """Execute the routed-expert first-column and remaining-columns pipelines."""
    gu_in, w_in = weights_pair
    w_scale = cast("_PallasRef", w_in.scale)
    gu_scale = cast("_PallasRef", gu_in.scale)
    w = dataclasses.replace(
        w_in,
        weight=w_in.weight.at[layer[0]].bitcast(jnp.float4_e2m1fn),
        scale=w_scale.at[layer[0]],
    )
    gu_single = dataclasses.replace(
        gu_in,
        weight=gu_in.weight.at[layer[0]].bitcast(jnp.float4_e2m1fn),
        scale=gu_scale.at[layer[0]],
    )
    gu = k.FusedWeightsRef(
        gate=gu_single,
        up=jax.tree.map(lambda r: r.at[..., _EXPERT_INTERMEDIATE_COLS:], gu_single),
    )
    localrows = ctx.tokens // TP_SIZE
    _init_routed_ring_meta(ctx.tables, ctx.ring_meta, localrows)
    active_ng = k.fill_metadata(
        cast("jax.Array", counts),
        cast("jax.Array", offset),
        ctx.metadata,
        cfgs=ctx.cfg,
    )

    @pl.when(active_ng == 0)
    def empty_metadata() -> None:
        """Zero the first metadata entry when no expert tiles are active."""
        ctx.metadata.gm_id_to_group_id[0] = 0
        ctx.metadata.gm_id_to_m_offset[0] = 0
        ctx.metadata.gm_id_to_m_offset[1] = 0

    ng = jnp.maximum(active_ng, 1)
    (xs, ws), _ = k.generate_block_specs(ctx.metadata, ctx.cfg)
    ctx.state[0] = 0

    # Real-prompt tuning: short expert loops need all twelve streams per poll; longer
    # loops alternate six-stream groups to bound polling cost.
    def advance(
        block: jax.Array,
        subtotal_vmem: _PallasRef,
        *,
        drain: bool = False,
        poll_round: jax.Array | int = 0,
    ) -> None:
        """Poll or drain the reduce-scatter ring for ``block``."""
        _ready_reduce(
            ctx.tables,
            subtotal_vmem,
            ctx.out.at[:, pl.ds(block * ctx.tn, ctx.tn)],
            ctx.sends,
            ctx.receives,
            ctx.credit_sems,
            ctx.stores,
            ctx.payload,
            ctx.rx,
            streams=ctx.regions,
            buffers=_RING_BUFFERS,
            state=ctx.state,
            ring_meta=ctx.ring_meta,
            drain=drain,
            poll_round=poll_round,
            sparse_poll=(ng >= _SPARSE_POLL_NG_THRESHOLD),
        )

    inner = _make_routed_inner(ctx, active_ng, ng, advance)
    _run_routed_first_column(
        ctx,
        (gu, w, ws),
        ng_pair=(active_ng, ng),
        callbacks=(inner, lambda g, panel: _routed_gu_work(ctx, g, panel)),
    )

    @functools.partial(
        pl.run_scoped, subtotal_vmem=pltpu.VMEM((ctx.tokens, ctx.tn), ctx.latent.dtype)
    )
    def remaining_columns(subtotal_vmem: _PallasRef) -> None:
        """Run columns 1..N-1 while overlapping the previous column's ring."""
        _emit_routed_subtotals(ctx, subtotal_vmem)
        ctx.state[0] = 0

        def shifted(spec: pl.BlockSpec) -> pl.BlockSpec:
            """Shift the column index map by +1 past the first column."""
            index_map = spec.index_map
            if index_map is None:
                return spec
            return dataclasses.replace(
                spec, index_map=lambda n, g, k_idx: index_map(n + 1, g, k_idx)
            )

        shifted_ws = jax.tree.map(
            shifted, ws, is_leaf=lambda v: isinstance(v, pl.BlockSpec)
        )

        def remaining(
            x_tile: _PallasRef,
            w_tile: k.WeightsRef,
            metadata_tile: k.MetadataRef,
        ) -> None:
            """Invoke ``inner`` for a non-first column block."""
            inner(x_tile, w_tile, metadata_tile, subtotal_vmem=subtotal_vmem)

        pltpu.emit_pipeline(
            remaining,
            grid=(ctx.width // ctx.tn - 1, ng, 1),
            in_specs=(xs, shifted_ws),
            out_specs=(),
        )(
            ctx.middle.reshape(-1, ctx.sub, _EXPERT_INTERMEDIATE_COLS),
            w,
            scratches=(ctx.metadata,),
        )


def _routed_experts(
    latent: _PallasRef,
    *weight_and_out_refs: object,
    tokens: int,
    probabilities_ready: Callable[[], None],
) -> None:
    """Joint GU/SiTU/first-down pipeline followed by remaining columns."""
    (
        gu_weights_obj,
        gu_scales_obj,
        rhs_obj,
        scales_obj,
        counts_obj,
        destinations_obj,
        probabilities_obj,
        tables_obj,
        layer_obj,
        offset_obj,
        middle_obj,
        out_obj,
    ) = weight_and_out_refs
    gu_weights = cast("_PallasRef", gu_weights_obj)
    gu_scales = cast("_PallasRef", gu_scales_obj)
    rhs = cast("_PallasRef", rhs_obj)
    scales = cast("_PallasRef", scales_obj)
    counts = cast("_PallasRef", counts_obj)
    destinations = cast("_FlatMetadata", destinations_obj)
    probabilities = cast("_FlatMetadata", probabilities_obj)
    tables = cast("_PallasRef", tables_obj)
    layer = cast("_PallasRef", layer_obj)
    offset = cast("_PallasRef", offset_obj)
    middle = cast("_PallasRef", middle_obj)
    out = cast("_PallasRef", out_obj)

    cfg, gu_cfg, tm, tn, width, max_gm = _build_routed_gmm_configs(
        latent.dtype,
        (gu_weights, gu_scales),
        (rhs, scales),
        counts_and_offset=(counts, offset),
        tokens=tokens,
    )
    localrows = tokens // TP_SIZE
    packed_half = _PACKED_HALF_8K if tokens == OVERFLOW_WINDOW_TOKENS else tn // 2
    sub = cfg.dims.size_lhs_sublane
    regions = k.regions(localrows, tn)
    streams = len(regions)
    unroll = (
        _DEFAULT_DOWN_UNROLL
        if tokens < OVERFLOW_WINDOW_TOKENS
        else _COMPACT_DOWN_UNROLL
    )
    gu_unroll = _GU_UNROLL_DEFAULT if tokens < OVERFLOW_WINDOW_TOKENS else _GU_UNROLL_8K
    first_down_unroll = (
        _FIRST_DOWN_UNROLL_DEFAULT
        if tokens < OVERFLOW_WINDOW_TOKENS
        else _FIRST_DOWN_UNROLL_8K
    )

    def run(**scratch: object) -> None:
        """Build ``_RoutedExpertContext`` from scoped scratch and run kernel."""
        ctx = _RoutedExpertContext(
            cfg=cfg,
            gu_cfg=gu_cfg,
            metadata=cast("k.MetadataRef", scratch["metadata"]),
            destinations=destinations,
            probabilities=probabilities,
            latent=latent,
            middle=middle,
            out=out,
            packed=cast("_PallasRef", scratch["packed"]),
            sums=cast("_PallasRef", scratch["sums"]),
            cast_buf=cast("_PallasRef", scratch["cast_buf"]),
            tables=tables,
            sends=cast("_PallasRef", scratch["sends"]),
            receives=cast("_PallasRef", scratch["receives"]),
            credit_sems=cast("_PallasRef", scratch["credit_sems"]),
            stores=cast("_PallasRef", scratch["stores"]),
            payload=cast("_PallasRef", scratch["payload"]),
            rx=cast("_PallasRef", scratch["rx"]),
            state=cast("_PallasRef", scratch["state"]),
            ring_meta=cast("_PallasRef", scratch["ring_meta"]),
            regions=regions,
            tokens=tokens,
            tm=tm,
            tn=tn,
            width=width,
            sub=sub,
            packed_half=packed_half,
            unroll=unroll,
            gu_unroll=gu_unroll,
            first_down_unroll=first_down_unroll,
            probabilities_ready=probabilities_ready,
        )
        _routed_experts_kernel(
            ctx,
            counts,
            offset,
            layer,
            (
                k.WeightsRef(weight=gu_weights, scale=gu_scales),
                k.WeightsRef(weight=rhs, scale=scales),
            ),
        )

    ring_dtype = jnp.float32
    pl.run_scoped(
        run,
        metadata=k.MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_gm,), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_gm + 1,), jnp.int32),
        ),
        packed=pltpu.VMEM((2, tm, 2, packed_half), latent.dtype),
        sums=pltpu.VMEM((tokens, 2, packed_half), jnp.float32),
        cast_buf=pltpu.VMEM((STORE_TILE, tn), latent.dtype),
        sends=pltpu.SemaphoreType.DMA((streams,)),
        receives=pltpu.SemaphoreType.DMA((2, streams)),
        credit_sems=pltpu.SemaphoreType.REGULAR((streams,)),
        stores=pltpu.SemaphoreType.DMA,
        payload=pltpu.VMEM((localrows, tn), ring_dtype),
        rx=pltpu.VMEM((2, localrows, tn), ring_dtype),
        state=pltpu.SMEM((1 + 2 * streams,), jnp.int32),
        ring_meta=pltpu.SMEM((_DIRECTED_RINGS * _RING_META_STRIDE,), jnp.int32),
    )


def _column_rhs(
    rhs: k.RhsRef,
    first: int,
    size: int,
) -> k.RhsRef:
    """Slice references before dequantization to bound panel temporaries.

    Invariants:
        assert ref.scale is not None
        assert first % pair == 0 and size % pair == 0
    """

    def slice_ref(ref: k.WeightsRef, col_first: int, col_size: int) -> k.WeightsRef:
        """Slice weight and scale references across columns."""
        if ref.scale is None:
            msg = "Routed expert weights require non-null quantization scales"
            raise ValueError(msg)
        scale_ref = cast("_PallasRef", ref.scale)
        return dataclasses.replace(
            ref,
            weight=ref.weight.at[:, pl.ds(col_first, col_size)],
            scale=scale_ref.at[..., pl.ds(col_first, col_size)],
        )

    if isinstance(rhs, k.FusedWeightsRef):
        # Gate/up columns alternate in lane-sized chunks. A complete pair
        # maps to half as many columns in each underlying reference.
        pair = 2 * pltpu.get_tpu_info().num_lanes
        if first % pair != 0 or size % pair != 0:
            msg = f"Fused column slice ({first}, {size}) must align to {pair}"
            raise ValueError(msg)
        return k.FusedWeightsRef(
            slice_ref(rhs.gate, first // 2, size // 2),
            slice_ref(rhs.up, first // 2, size // 2),
        )
    return slice_ref(cast("k.WeightsRef", rhs), first, size)


def _expert_product(
    lhs: PackedRows | _PallasRef,
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

    Invariants:
        assert cfgs.dims.size_k == cfgs.tiles.tile_k
        assert weights is not None
    """
    if cfgs.dims.size_k != cfgs.tiles.tile_k:
        msg = "Routed expert product requires single-tile full-K reduction"
        raise ValueError(msg)
    columns = pltpu.get_tpu_info().mxu_column_size
    activations = lhs.reshape(-1, cfgs.tiles.tile_k)[:bucket_rows]
    width = (
        2 * rhs.gate.weight.shape[-1]
        if isinstance(rhs, k.FusedWeightsRef)
        else cast("k.WeightsRef", rhs).weight.shape[-1]
    )
    block = cfgs.rhs_cfgs.quant_block_size
    if block is None:
        msg = "Routed expert product requires a quantized block size"
        raise ValueError(msg)
    # The 8K window uses a 256-row tile. Smaller windows use 128 rows;
    # their whole-tile conversion schedules better than panel-local loads.
    panel_local = cfgs.tiles.tile_m == _COMPACT_DOWN_TILE
    weights: jax.Array | None = None
    if not panel_local:
        weights_raw = rhs.get_weight()
        scales = rhs.get_scale(replicate_size=block).astype(cfgs.lhs_cfgs.dtype)
        weights_deq = weights_raw.astype(cfgs.lhs_cfgs.dtype).reshape(
            cfgs.num_quant_blocks_per_tile_k, block, width
        )
        weights = (weights_deq * scales).reshape(cfgs.tiles.tile_k, width)
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
                msg = "Expected pre-dequantized weights for non-panel-local tile"
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
    rhs: k.FusedWeightsRef,
    output: _PallasRef,
    *partial_and_metadata: object,
    cfgs: k.GmmConfigs,
    overlap_work: Callable[[int], None],
) -> None:
    """GU/SiTU with the production row buckets and adjacent-expert carry.

    Expert boundaries need not align with the 16-row BF16 sublane. Mask outside
    the active expert, then preserve the preceding expert's partial first row.
    """
    partial_obj, metadata_obj = partial_and_metadata
    partial = cast("_PallasRef", partial_obj)
    metadata = cast("k.MetadataRef", metadata_obj)
    group = pl.program_id(1)
    start = metadata.gm_id_to_m_offset[group]
    end = metadata.gm_id_to_m_offset[group + 1]
    aligned = start - start % cfgs.dims.size_lhs_sublane
    first, last = start - aligned, end - aligned

    def product(bucket_index: int) -> None:
        """Compute one row-bucketed gate/up SiTU tile and update sublane carry."""
        rows = cfgs.tiles.bucket_base * (bucket_index + 1)
        value = _expert_product(
            lhs, rhs, cfgs=cfgs, bucket_rows=rows, overlap_work=overlap_work
        )
        value = k.apply_act_fn(value, cfgs.fuse_act)
        row = lax.broadcasted_iota(jnp.int32, value.shape, 0)
        value = jnp.where((first <= row) & (row < last), value, 0)
        output.reshape(-1, cfgs.tiles.tile_n)[:rows] = value.astype(output.dtype)
        zeros = jnp.zeros_like(partial[...])
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
    lhs: _PallasRef,
    rhs: k.WeightsRef,
    *output_and_metadata: object,
    cfgs: k.GmmConfigs,
    overlap_work: Callable[[int], None],
) -> None:
    """Down projection stored as packed rows for token-wise VALU combine.

    Each valid expert row is consumed individually, so this output needs no
    adjacent-expert carry. BF16 rounding precedes FP32 routing weighting.

    Invariants:
        assert cfgs.fuse_act is None
    """
    output_obj, metadata_obj = output_and_metadata
    output = cast("_PallasRef", output_obj)
    metadata = cast("k.MetadataRef", metadata_obj)
    if cfgs.fuse_act is not None:
        msg = "_down_inner does not support fused activation"
        raise ValueError(msg)
    group = pl.program_id(1)
    start = metadata.gm_id_to_m_offset[group]
    end = metadata.gm_id_to_m_offset[group + 1]
    aligned = start - start % cfgs.dims.size_lhs_sublane
    last = end - aligned

    # GU already masks inactive rows; consume() visits only live routes.
    def product(bucket_index: int) -> None:
        """Compute one row-bucketed down-projection tile and store packed rows."""
        rows = cfgs.tiles.bucket_base * (bucket_index + 1)
        with jax.named_scope(f"bm{rows}_fragment"):
            value = _expert_product(
                lhs, rhs, cfgs=cfgs, bucket_rows=rows, overlap_work=overlap_work
            )
            if cfgs.tiles.tile_n == _COMPACT_DOWN_COLUMNS:
                # Keep the half boundary on a complete 128-lane block.
                # Only the final 128 elements are padding; the consumer trims
                # them once per subtotal before communicating the 896 columns.
                value_cast = value.astype(output.dtype)
                padded = jnp.concatenate(
                    (value_cast, jnp.zeros((rows, VREG_LANE_COUNT), output.dtype)),
                    axis=1,
                )
                output[:rows] = padded.reshape(rows, 2, _PACKED_HALF_8K)
            else:
                output[:rows] = value.astype(output.dtype).reshape(
                    rows, 2, cfgs.tiles.tile_n // 2
                )

    branches = [
        functools.partial(product, bucket_index=i)
        for i in range(cfgs.tiles.tile_m // cfgs.tiles.bucket_base)
    ]
    lax.switch(last // cfgs.tiles.bucket_base, branches)


def _poll_ready_reduce_stream(
    x_ref: _PallasRef,
    o_ref: _PallasRef,
    *sem_and_buf_refs: _PallasRef,
    stream_region: tuple[int, tuple[int, ...], int, int],
) -> None:
    """Poll and advance one stream of the 12-stream reduce-scatter ring."""
    send_sems, recv_sems, credit_sems, payload, received, state, ring_meta = (
        sem_and_buf_refs
    )
    stream, region, count, buffers = stream_region
    begin, size, col_begin, width = region
    base = (stream % _DIRECTED_RINGS) * _RING_META_STRIDE
    rows, cols = pl.ds(begin, size), pl.ds(col_begin, width)
    units = size * width * 4 // _DMA_BYTES_PER_UNIT
    step = state[1 + stream]

    def remote(target: tuple[jax.Array], bank: jax.Array | int) -> _RemoteCopy:
        """Create the async remote DMA descriptor for ``stream`` and ``bank``."""
        return cast(
            "_RemoteCopy",
            pltpu.make_async_remote_copy(
                payload.at[rows, cols],
                received.at[bank, rows, cols],
                send_sems.at[stream],
                recv_sems.at[bank, stream],
                device_id=target,
                device_id_type=pl.DeviceIdType.MESH,
            ),
        )

    @pl.when(state[1 + count + stream] == 0)
    def receive() -> None:
        """Poll incoming DMA, accumulate local shard, and signal credit."""
        bank = (step - 1) & 1
        ready = (pl.semaphore_read(send_sems.at[stream]) >= units) & (
            pl.semaphore_read(recv_sems.at[bank, stream]) >= units
        )

        @pl.when(ready)
        def reduce() -> None:
            """Accumulate the received peer shard and return buffer credit."""
            target = (ring_meta[base],)
            remote(target, bank).wait()
            source_rows = pl.ds(
                pl.multiple_of(ring_meta[base + 2 + step], o_ref.shape[0]) + begin,
                size,
            )
            payload.at[rows, cols][...] = (
                x_ref.at[source_rows, cols][...].astype(payload.dtype)
                + received.at[bank, rows, cols][...]
            )
            pl.semaphore_signal(
                credit_sems.at[stream],
                1,
                device_id=(ring_meta[base + 1],),
                device_id_type=pl.DeviceIdType.MESH,
            )
            state[1 + count + stream] = jnp.where(step == TP_PEERS, _PHASE_COMPLETE, 1)

    @pl.when(state[1 + count + stream] == 1)
    def forward() -> None:
        """Poll for downstream credit and issue the next outgoing hop."""
        available = (step < buffers) | (pl.semaphore_read(credit_sems.at[stream]) >= 1)

        @pl.when(available)
        def send() -> None:
            """Acquire credit if needed and start the remote DMA."""

            @pl.when(step >= buffers)
            def acquire() -> None:
                """Wait for one flow-control credit."""
                pl.semaphore_wait(credit_sems.at[stream], 1)

            remote((ring_meta[base],), step & 1).start()
            state[1 + stream] = step + 1
            state[1 + count + stream] = 0

    @pl.when(state[1 + count + stream] == _PHASE_COMPLETE)
    def finish() -> None:
        """Drain final credits once all hops have completed."""

        @pl.when(pl.semaphore_read(credit_sems.at[stream]) >= buffers)
        def release() -> None:
            """Consume the remaining buffer credits and mark the stream done."""
            pl.semaphore_wait(credit_sems.at[stream], buffers)
            state[1 + count + stream] = _PHASE_DONE
            state[0] = state[0] - 1


def _initialize_ready_reduce(
    x_ref: _PallasRef,
    o_ref: _PallasRef,
    *sem_and_buf_refs: _PallasRef,
    streams: Sequence[tuple[int, ...]],
) -> None:
    """Synchronize neighbors on the ring barrier and seed hop 0."""
    send_sems, recv_sems, payload, received, state, ring_meta = sem_and_buf_refs
    count = len(streams)
    peers = [(ring_meta[d * _RING_META_STRIDE],) for d in range(_DIRECTED_RINGS)]
    sem = pltpu.get_barrier_semaphore()
    for peer in peers:
        pl.semaphore_signal(sem, 1, device_id=peer, device_id_type=pl.DeviceIdType.MESH)
    pl.semaphore_wait(sem, len(peers))

    @functools.partial(pl.run_scoped, ready=pltpu.SemaphoreType.REGULAR)
    def initialized(ready: _PallasRef) -> None:
        """Complete the two-phase neighbor readiness handshake."""
        for peer in peers:
            pl.semaphore_signal(
                ready, 1, device_id=peer, device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(ready, len(peers))

    for stream, region in enumerate(streams):
        begin, size, col_begin, width = region
        base = (stream % _DIRECTED_RINGS) * _RING_META_STRIDE
        rows, cols = pl.ds(begin, size), pl.ds(col_begin, width)
        source_rows = pl.ds(
            pl.multiple_of(ring_meta[base + 2], o_ref.shape[0]) + begin, size
        )
        target = (ring_meta[base],)
        payload.at[rows, cols][...] = x_ref.at[source_rows, cols][...].astype(
            payload.dtype
        )
        cast(
            "_RemoteCopy",
            pltpu.make_async_remote_copy(
                payload.at[rows, cols],
                received.at[jnp.int32(0), rows, cols],
                send_sems.at[stream],
                recv_sems.at[jnp.int32(0), stream],
                device_id=target,
                device_id_type=pl.DeviceIdType.MESH,
            ),
        ).start()
        state[1 + stream] = 1
        state[1 + count + stream] = 0
    state[0] = count + 1


def _sweep_ready_reduce(
    x_ref: _PallasRef,
    o_ref: _PallasRef,
    sem_and_buf_refs: tuple[_PallasRef, ...],
    streams: Sequence[tuple[int, ...]],
    *,
    buffers_and_selected: tuple[int, int | None],
) -> None:
    """Poll all streams or one 6-stream half-group."""
    send_sems, recv_sems, credit_sems, payload, received, state, ring_meta = (
        sem_and_buf_refs
    )
    buffers, selected = buffers_and_selected
    count = len(streams)
    for stream, region in enumerate(streams):
        if selected is not None and (stream // _DIRECTED_RINGS) != selected:
            continue
        _poll_ready_reduce_stream(
            x_ref,
            o_ref,
            send_sems,
            recv_sems,
            credit_sems,
            payload,
            received,
            state,
            ring_meta,
            stream_region=(stream, region, count, buffers),
        )


def _ready_reduce(
    _tables_ref: object,
    x_ref: _PallasRef,
    o_ref: _PallasRef,
    *sem_and_buf_refs: _PallasRef,
    **reduce_opts: Unpack[_ReadyReduceOptions],
) -> None:
    """Run the 12-stream protocol with cached descriptors and cheap failed polls.

    ring_meta has six directed rings, each with neighbor mesh IDs followed by
    32 source-row bases. SMEM offsets, not physical VMEM pointers, are cached.
    Neither a failed receive nor a failed credit poll needs a payload address.

    Invariants:
        assert buffers == 2
    """
    send_sems, recv_sems, credit_sems, store_sem, payload, received = sem_and_buf_refs
    streams = reduce_opts["streams"]
    buffers = reduce_opts["buffers"]
    state = reduce_opts["state"]
    ring_meta = reduce_opts["ring_meta"]
    drain = reduce_opts.get("drain", False)
    poll_round = reduce_opts.get("poll_round", 0)
    sparse_poll = reduce_opts.get("sparse_poll", True)
    if buffers != _RING_BUFFERS:
        msg = f"_ready_reduce requires {_RING_BUFFERS} buffers, got {buffers}"
        raise ValueError(msg)
    poll_refs = (
        send_sems,
        recv_sems,
        credit_sems,
        payload,
        received,
        state,
        ring_meta,
    )

    @pl.when(state[0] == 0)
    def initialize() -> None:
        """Seed hop 0 across all 12 streams."""
        _initialize_ready_reduce(
            x_ref,
            o_ref,
            send_sems,
            recv_sems,
            payload,
            received,
            state,
            ring_meta,
            streams=streams,
        )

    if drain:

        def pending(_i: jax.Array) -> jax.Array:
            """Return True while any stream has not yet completed."""
            return state[0] != 1

        def iterate(i: jax.Array) -> jax.Array:
            """Poll all 12 streams once during final drain."""
            _sweep_ready_reduce(
                x_ref,
                o_ref,
                poll_refs,
                streams,
                buffers_and_selected=(buffers, None),
            )
            return i + 1

        jax.lax.while_loop(pending, iterate, jnp.int32(0))
        output = cast("_AsyncCopy", pltpu.make_async_copy(payload, o_ref, store_sem))
        output.start()
        output.wait()
    else:

        @pl.when(state[0] != 1)
        def progress() -> None:
            """Alternate two six-stream groups only for long expert loops."""
            for group in range(2):

                @pl.when(((poll_round & 1) == group) | ~jnp.asarray(sparse_poll))
                def selected(*, grp: int = group) -> None:
                    """Poll the active 6-stream group."""
                    _sweep_ready_reduce(
                        x_ref,
                        o_ref,
                        poll_refs,
                        streams,
                        buffers_and_selected=(buffers, grp),
                    )


def _epilogue_body(
    s: dict[str, object],
    reduced: _PallasRef,
    shared: _PallasRef,
    valid: jax.Array,
    *epilogue_refs: _PallasRef,
) -> None:
    """Normalize ``reduced`` and stream 512-column latent-up panels."""
    norm_weight, up_weight, prefix, output = epilogue_refs
    width = reduced.shape[1]
    norm_buf = cast("_PallasRef", s["norm"])
    norm_sem = cast("_PallasRef", s["norm_sem"])
    normalized = cast("_PallasRef", s["normalized"])
    weights_bufs = cast("tuple[_PallasRef, ...]", s["weights"])
    tiles_bufs = cast("tuple[_PallasRef, ...]", s["tiles"])
    weight_sems = cast("tuple[_PallasRef, ...]", s["weight_sems"])
    prefix_sems = cast("tuple[_PallasRef, ...]", s["prefix_sems"])
    output_sems = cast("tuple[_PallasRef, ...]", s["output_sems"])

    copy = cast("_AsyncCopy", pltpu.make_async_copy(norm_weight, norm_buf, norm_sem))
    copy.start()
    copy.wait()
    panels = _EPILOGUE_HIDDEN_COLS // _EPILOGUE_PANEL_COLS

    def weight_copy(panel: int) -> _AsyncCopy:
        """Create the async DMA descriptor for latent-up weight ``panel``."""
        bank = panel % _EPILOGUE_BANKS
        return cast(
            "_AsyncCopy",
            pltpu.make_async_copy(
                up_weight.at[
                    :, pl.ds(panel * _EPILOGUE_PANEL_COLS, _EPILOGUE_PANEL_COLS)
                ],
                weights_bufs[bank],
                weight_sems[bank],
            ),
        )

    def prefix_copy(panel: int) -> _AsyncCopy:
        """Create the async DMA descriptor for residual prefix ``panel``."""
        bank = panel % _EPILOGUE_BANKS
        return cast(
            "_AsyncCopy",
            pltpu.make_async_copy(
                prefix.at[:, pl.ds(panel * _EPILOGUE_PANEL_COLS, _EPILOGUE_PANEL_COLS)],
                tiles_bufs[bank],
                prefix_sems[bank],
            ),
        )

    def output_copy(panel: int) -> _AsyncCopy:
        """Create the async DMA descriptor storing completed ``panel``."""
        bank = panel % _EPILOGUE_BANKS
        return cast(
            "_AsyncCopy",
            pltpu.make_async_copy(
                tiles_bufs[bank],
                output.at[:, pl.ds(panel * _EPILOGUE_PANEL_COLS, _EPILOGUE_PANEL_COLS)],
                output_sems[bank],
            ),
        )

    # Hide the first weight/residual loads behind the RMS reduction.
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
        value = (
            product
            + shared[:, pl.ds(panel * _EPILOGUE_PANEL_COLS, _EPILOGUE_PANEL_COLS)]
        )
        value = jnp.where(valid, value, 0)
        tiles_bufs[bank][...] = tiles_bufs[bank][...] + value
        output_copy(panel).start()
    for panel in range(panels - _EPILOGUE_BANKS, panels):
        output_copy(panel).wait()


def _epilogue(
    control: _PallasRef,
    reduced: _PallasRef,
    shared: _PallasRef,
    *weight_and_out_refs: _PallasRef,
) -> None:
    """Normalize routed output and project up with three weight/prefix banks.

    Reduced FP32 rows remain in VMEM. Preserve the BF16 boundaries after
    normalization, after latent-up, and after adding the shared expert.
    """
    norm_weight_in, up_weight_in, prefix, output = weight_and_out_refs
    rows, width = reduced.shape
    family = control[1]
    norm_weight = norm_weight_in.at[family, :, :width]
    up_weight = up_weight_in.at[family]
    valid = _stripe_validity(rows, control[0], jax.lax.axis_index("k3_sp"))
    vmem, dma = pltpu.VMEM, pltpu.SemaphoreType.DMA

    pl.run_scoped(
        lambda s: _epilogue_body(
            s, reduced, shared, valid, norm_weight, up_weight, prefix, output
        ),
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

    def __init__(self, ref: _PallasRef) -> None:
        """Reshape the 2D SMEM reference into a single-row view."""
        # Both views remain 2D; scalar consumers use one contiguous index.
        self.ref = ref.reshape(1, ref.shape[0] * ref.shape[1])

    def __getitem__(self, index: jax.Array | int) -> jax.Array:
        """Read scalar element ``index`` from the flattened SMEM view."""
        return self.ref[0, index]


def _prepare_layer_kernel(
    s: dict[str, object],
    tables: _PallasRef,
    control: _PallasRef,
    hidden: _PallasRef,
    *prep_refs: object,
) -> None:
    """Run ``_prologue`` and drain compact metadata DMAs into SMEM."""
    (
        ids_v_obj,
        probs_v_obj,
        router_obj,
        bias_obj,
        latent_down_obj,
        shared_gu_obj,
        shared_down_obj,
        latent_obj,
        fallback_ids_obj,
        fallback_probs_obj,
        fallback_counts_obj,
        fallback_shared_obj,
        maximum_obj,
        tokens_and_cap_obj,
    ) = prep_refs
    ids_v = cast("_PallasRef", ids_v_obj)
    probs_v = cast("_PallasRef", probs_v_obj)
    router = cast("_PallasRef | None", router_obj)
    bias = cast("_PallasRef | None", bias_obj)
    latent_down = cast("_PallasRef | None", latent_down_obj)
    shared_gu = cast("_PallasRef | None", shared_gu_obj)
    shared_down = cast("_PallasRef | None", shared_down_obj)
    latent = cast("_PallasRef | None", latent_obj)
    fallback_ids = cast("_PallasRef | None", fallback_ids_obj)
    fallback_probs = cast("_PallasRef | None", fallback_probs_obj)
    fallback_counts = cast("_PallasRef | None", fallback_counts_obj)
    fallback_shared = cast("_PallasRef | None", fallback_shared_obj)
    maximum = cast("_PallasRef | None", maximum_obj)
    tokens, capacity = cast("tuple[int, int]", tokens_and_cap_obj)
    counts_v = cast("_PallasRef", s["counts_v"])
    counts_s = cast("_PallasRef", s["counts"])
    ids_s = cast("_PallasRef", s["ids"])
    probs_s = cast("_PallasRef", s["probs"])
    shared_s = cast("_PallasRef", s["shared"])
    maximum_s = cast("_PallasRef", s["maximum"])
    count_copy = cast(
        "_AsyncCopy",
        pltpu.make_async_copy(counts_v, counts_s, s["count_sem"]),
    )
    maximum_copy = (
        cast(
            "_AsyncCopy",
            pltpu.make_async_copy(maximum, maximum_s, s["max_sem"]),
        )
        if tokens == OVERFLOW_WINDOW_TOKENS
        else None
    )
    indices = cast(
        "_AsyncCopy",
        pltpu.make_async_copy(
            ids_v.at[: capacity // VREG_LANE_COUNT, :], ids_s, s["index_sem"]
        ),
    )
    prob_copy = cast(
        "_AsyncCopy",
        pltpu.make_async_copy(
            probs_v.at[: capacity // VREG_LANE_COUNT, :], probs_s, s["prob_sem"]
        ),
    )

    def counts_ready() -> None:
        """Start moving expert route counts (and 8K max) to SMEM."""
        count_copy.start()
        if maximum_copy is not None:
            maximum_copy.start()

    def metadata_ready() -> None:
        """Start VMEM-to-SMEM transfers for sorted IDs and probabilities."""
        indices.start()
        prob_copy.start()

    _prologue(
        tables,
        control,
        hidden,
        router,
        bias,
        latent_down,
        shared_gu,
        shared_down,
        shared_s,
        latent,
        ids_v,
        probs_v,
        counts_v,
        maximum,
        counts_ready=counts_ready,
        metadata_ready=metadata_ready,
    )
    count_copy.wait()
    if maximum_copy is not None:
        maximum_copy.wait()
    if tokens != OVERFLOW_WINDOW_TOKENS:
        metadata_ready()
    if tokens == OVERFLOW_WINDOW_TOKENS:

        @pl.when(maximum_s[0] > capacity)
        def export_overflow() -> None:
            """Export full VMEM routing metadata and shared output on overflow."""
            for source, dest in (
                (ids_v, fallback_ids),
                (probs_v, fallback_probs),
                (counts_v, fallback_counts),
                (shared_s, fallback_shared),
            ):
                copy = cast(
                    "_AsyncCopy",
                    pltpu.make_async_copy(source, dest, s["export_sem"]),
                )
                copy.start()
                copy.wait()

    indices.wait()
    if tokens == OVERFLOW_WINDOW_TOKENS:
        prob_copy.wait()


def _layer_kernel_body(
    s: dict[str, object],
    tables: _PallasRef,
    control: _PallasRef,
    hidden: _PallasRef,
    *layer_refs: _PallasRef | None,
) -> None:
    """Execute prologue, optional 8K overflow export, routed experts, and epilogue."""
    (
        router,
        bias,
        latent_down,
        shared_gu,
        shared_down,
        expert_gu,
        gu_scales,
        expert_down,
        down_scales,
        norm,
        latent_up,
        prefix,
        output,
        latent,
        middle,
        fallback_ids,
        fallback_probs,
        fallback_counts,
        fallback_shared,
        maximum,
    ) = layer_refs
    rows = hidden.shape[0]
    tokens = rows * TP_SIZE
    capacity = (
        tokens * _TOP_K
        if tokens < OVERFLOW_WINDOW_TOKENS
        else _COMPACT_METADATA_PREFIX_ROWS
    )
    shape = (tokens // SUBLANE_ROWS, VREG_LANE_COUNT)
    vmem = pltpu.VMEM
    expert_gu_ref = cast("_PallasRef", expert_gu)
    offset_ref = cast("_PallasRef", s["offset"])
    family_ref = cast("_PallasRef", s["family"])
    counts_s = cast("_PallasRef", s["counts"])
    ids_s = cast("_PallasRef", s["ids"])
    probs_s = cast("_PallasRef", s["probs"])
    shared_s = cast("_PallasRef", s["shared"])
    reduced_s = cast("_PallasRef", s["reduced"])
    maximum_s = cast("_PallasRef", s["maximum"])
    offset_ref[0] = 0
    family = k.expert_family_slot(expert_gu_ref.shape[0], control[1])
    family_ref[0] = family

    prep_tail = (
        router,
        bias,
        latent_down,
        shared_gu,
        shared_down,
        latent,
        fallback_ids,
        fallback_probs,
        fallback_counts,
        fallback_shared,
        maximum,
        (tokens, capacity),
    )
    if tokens == OVERFLOW_WINDOW_TOKENS:
        pl.run_scoped(
            lambda ids_v, probs_v: _prepare_layer_kernel(
                s, tables, control, hidden, ids_v, probs_v, *prep_tail
            ),
            vmem(shape, jnp.int32),
            vmem(shape, jnp.float32),
        )
    else:
        probs_v_ref = cast("_PallasRef", s["probs_v"])
        pl.run_scoped(
            lambda ids_v: _prepare_layer_kernel(
                s, tables, control, hidden, ids_v, probs_v_ref, *prep_tail
            ),
            vmem(shape, jnp.int32),
        )

    def probabilities_ready() -> None:
        """Wait for the <=4K probability SMEM copy before expert consumption."""
        if tokens < OVERFLOW_WINDOW_TOKENS:
            probs_v = cast("_PallasRef", s["probs_v"])
            cast(
                "_AsyncCopy",
                pltpu.make_async_copy(
                    probs_v.at[: capacity // VREG_LANE_COUNT, :],
                    probs_s,
                    s["prob_sem"],
                ),
            ).wait()

    def fast() -> None:
        """Execute the in-kernel routed experts and epilogue."""
        _routed_experts(
            cast("_PallasRef", latent),
            expert_gu,
            gu_scales,
            expert_down,
            down_scales,
            counts_s,
            _FlatMetadata(ids_s),
            _FlatMetadata(probs_s),
            tables,
            family_ref,
            offset_ref,
            middle,
            reduced_s,
            tokens=tokens,
            probabilities_ready=probabilities_ready,
        )
        _epilogue(
            control,
            reduced_s,
            shared_s,
            cast("_PallasRef", norm),
            cast("_PallasRef", latent_up),
            cast("_PallasRef", prefix),
            cast("_PallasRef", output),
        )

    if tokens < OVERFLOW_WINDOW_TOKENS:
        fast()
    else:

        @pl.when(maximum_s[0] <= capacity)
        def compact() -> None:
            """Run the fast path when 8K route counts fit compact SMEM capacity."""
            fast()


def _layer_kernel(
    tables: _PallasRef,
    control: _PallasRef,
    hidden: _PallasRef,
    *layer_refs: _PallasRef | None,
) -> None:
    """One launch owns all compute, metadata and communication on the fast path.

    Large projection scratch has a shorter lifetime than expert scratch. Only
    shared output and the scalar routing metadata survive that boundary. At 8K
    all ranks take the same overflow branch; no routes are dropped to fit SMEM.
    """
    (
        router,
        bias,
        latent_down,
        shared_gu,
        shared_down,
        expert_gu,
        gu_scales,
        expert_down,
        down_scales,
        norm,
        latent_up,
        prefix,
        output,
        latent,
        middle,
        *fallback_refs,
    ) = layer_refs
    fallback_ids, fallback_probs, fallback_counts, fallback_shared, maximum = (
        fallback_refs if fallback_refs else (None, None, None, None, None)
    )
    rows = hidden.shape[0]
    tokens = rows * TP_SIZE
    capacity = (
        tokens * _TOP_K
        if tokens < OVERFLOW_WINDOW_TOKENS
        else _COMPACT_METADATA_PREFIX_ROWS
    )
    shape = (tokens // SUBLANE_ROWS, VREG_LANE_COUNT)
    vmem, smem, dma = pltpu.VMEM, pltpu.SMEM, pltpu.SemaphoreType.DMA

    pl.run_scoped(
        lambda s: _layer_kernel_body(
            s,
            tables,
            control,
            hidden,
            router,
            bias,
            latent_down,
            shared_gu,
            shared_down,
            expert_gu,
            gu_scales,
            expert_down,
            down_scales,
            norm,
            latent_up,
            prefix,
            output,
            latent,
            middle,
            fallback_ids,
            fallback_probs,
            fallback_counts,
            fallback_shared,
            maximum,
        ),
        {
            "shared": vmem((rows, _EPILOGUE_HIDDEN_COLS), jnp.bfloat16),
            "reduced": vmem((rows, _LATENT_FULL_COLS), jnp.float32),
            "probs_v": (
                vmem(shape, jnp.float32) if tokens < OVERFLOW_WINDOW_TOKENS else None
            ),
            "counts_v": vmem((_LOCAL_EXPERTS,), jnp.int32),
            "counts": smem((_LOCAL_EXPERTS,), jnp.int32),
            "maximum": (
                smem((1,), jnp.int32) if tokens == OVERFLOW_WINDOW_TOKENS else None
            ),
            "ids": smem((capacity // VREG_LANE_COUNT, VREG_LANE_COUNT), jnp.int32),
            "probs": smem((capacity // VREG_LANE_COUNT, VREG_LANE_COUNT), jnp.float32),
            "family": smem((1,), jnp.int32),
            "offset": smem((1,), jnp.int32),
            "count_sem": dma,
            "max_sem": dma if tokens == OVERFLOW_WINDOW_TOKENS else None,
            "index_sem": dma,
            "prob_sem": dma,
            "export_sem": dma,
        },
    )


def routed_experts_prepared(
    latent: jax.Array,
    *expert_inputs: jax.Array,
    reduce_tables: jax.typing.ArrayLike,
    expert_layer: jax.Array | None = None,
    global_route_max: jax.Array | int | None = None,
    reduce_dtype: jax.typing.DTypeLike | None = None,
) -> jax.Array:
    """Consume prepared expert-major metadata with the indexed GU/down path."""
    (
        sorted_tokens,
        sorted_probs,
        counts,
        expert_gate_up,
        expert_gate_up_scales,
        expert_down,
        expert_down_scales,
    ) = expert_inputs
    if expert_layer is not None:
        expert_layer = k.expert_family_slot(expert_gate_up.shape[0], expert_layer)
    tokens = latent.shape[0]
    gu_scale = expert_gate_up_scales.astype(jnp.uint8)
    down_scale = expert_down_scales.astype(jnp.uint8)
    middle = indexed_gate_up(
        latent, sorted_tokens, expert_gate_up, counts, gu_scale, expert_layer
    )
    ring_dtype = jnp.dtype(
        _config_mod.active().moe_reduce_dtype if reduce_dtype is None else reduce_dtype
    )
    if tokens == OVERFLOW_WINDOW_TOKENS:
        return down_valu_compact(
            middle,
            expert_down,
            down_scale,
            counts,
            sorted_tokens,
            sorted_probs,
            reduce_tables,
            tile=_COMPACT_DOWN_TILE,
            columns=_COMPACT_DOWN_COLUMNS,
            unroll=_COMPACT_DOWN_UNROLL,
            metadata_prefix_rows=_COMPACT_METADATA_PREFIX_ROWS,
            slice_input=False,
            global_route_max=global_route_max,
            tokens=tokens,
            layer=expert_layer,
            ring_dtype=ring_dtype,
        )
    return down_valu(
        middle,
        expert_down,
        down_scale,
        counts,
        sorted_tokens,
        sorted_probs,
        reduce_tables,
        tile=_DEFAULT_DOWN_TILE,
        columns=_DEFAULT_DOWN_COLUMNS,
        unroll=_DEFAULT_DOWN_UNROLL,
        tokens=tokens,
        layer=expert_layer,
        ring_dtype=ring_dtype,
    )


def make_indexed_moe_sp_ring(
    mesh: Mesh | None,
    *,
    seq_len: int = OVERFLOW_WINDOW_TOKENS,
) -> Callable[..., jax.Array]:
    """Build the fused EP16/TP2 prefill MoE for a supported serving window.

    Inputs follow the prefill layer interface: rank-sharded normalized hidden
    states, residual prefix, resident weight families, valid length and family
    index. All fast-path operations execute in one Pallas kernel. Rare 8K route
    overflow uses the uncapped production expert path with the same metadata.
    """
    if seq_len not in _SUPPORTED_WINDOWS:
        msg = "Fused prefill MoE supports 512, 1024, 2048, 4096 and 8192 token windows"
        raise ValueError(msg)
    if _config_mod.active().moe_reduce_dtype != "float32":
        msg = "Fused prefill MoE requires FP32 reduction; use the dispatcher for BF16"
        raise ValueError(msg)
    if mesh is None or mesh.size != TP_SIZE or tuple(mesh.axis_names) != ("k3_sp",):
        msg = "Fused prefill MoE requires a flat 32-rank k3_sp mesh"
        raise ValueError(msg)
    rows = seq_len // TP_SIZE
    tables = k.ring_tables(list(mesh.devices.flat), tuple(range(TP_SIZE)))
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    shape = (seq_len // SUBLANE_ROWS, VREG_LANE_COUNT)
    bf16 = jnp.bfloat16
    workspace_shapes: list[tuple[tuple[int, ...], jnp.dtype]] = [
        ((rows, _EPILOGUE_HIDDEN_COLS), jnp.dtype(bf16)),
        ((seq_len, 2, _LATENT_HALF_COLS), jnp.dtype(bf16)),
        ((seq_len * _TOP_K, _EXPERT_INTERMEDIATE_COLS), jnp.dtype(bf16)),
    ]
    output_specs: list[pl.BlockSpec] = [hbm] * 3
    if seq_len == OVERFLOW_WINDOW_TOKENS:
        # Only 8K can overflow compact SMEM routing capacity. At <=4K every
        # route fits, so omit the unused overflow outputs and maximum DMA.
        workspace_shapes += [
            (shape, jnp.dtype(jnp.int32)),
            (shape, jnp.dtype(jnp.float32)),
            ((_LOCAL_EXPERTS,), jnp.dtype(jnp.int32)),
            ((rows, _EPILOGUE_HIDDEN_COLS), jnp.dtype(bf16)),
            ((1,), jnp.dtype(jnp.int32)),
        ]
        output_specs += [hbm] * 4 + [pl.BlockSpec()]
    kernel = pl.pallas_call(
        _layer_kernel,
        out_shape=tuple(jax.ShapeDtypeStruct(s, d) for s, d in workspace_shapes),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=(hbm,) * _NUM_PROLOGUE_INPUT_SPECS,
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
        in_specs=(P("k3_sp"), P("k3_sp"), P("k3_sp"), P(), P()),
        out_specs=P("k3_sp"),
        check_vma=False,
    )
    def local(
        hidden: jax.Array,
        prefix: jax.Array,
        families: dict[str, jax.Array],
        length: jax.Array,
        slot: jax.Array,
    ) -> jax.Array:
        """Run the fused MoE kernel on one shard and handle 8K overflow."""
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
            """Execute the uncapped fallback expert pipeline on 8K route overflow."""
            reduced = routed_experts_prepared(
                latent,
                ids.reshape(-1),
                probs.reshape(-1),
                counts,
                w["expert_gu"],
                w["expert_gus"],
                w["expert_down"],
                w["expert_ds"],
                reduce_tables=tables,
                expert_layer=slot,
                global_route_max=maximum[0],
            )
            selected = {name: w[name][slot[0]] for name in ("latent_norm", "latent_up")}
            return k.moe_sp_tail(
                reduced,
                shared,
                prefix[0],
                length,
                selected,
                axis_name="k3_sp",
                eps=_RMS_EPS,
                shared_reduced=True,
            )

        return jax.lax.cond(
            maximum[0] <= _COMPACT_METADATA_PREFIX_ROWS, lambda: final, overflow
        )[None]

    return jax.jit(local)


stripe_validity = _stripe_validity
