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
# src/vllm_torchtpu/megakernel/kimi_k3/kernels/mla_dcp.py
"""Decode context parallelism (DCP) for Kimi-K3 MLA.

Without DCP every TP32 rank holds its 3 heads' expanded K|V for every token.
Under DCP the ``G`` ranks of a group (consecutive ranks, ``member = rank %
G``) each hold the compressed latent (576 of 640 lanes) of a disjoint 1/G of
the tokens and attend *all* the group's heads over them in latent space
(``q_abs = [q_nope @ W_UK^T | q_rope | 0]``, score scale folded in):

1. **q all-gather.** Each rank writes its ``[B, heads, 640]`` absorbed
   queries into slot ``member`` of every group peer's gather buffer.
2. **Local attention.** For each row, over the rank's own pages (global page
   ``j G + member`` is local page ``j``; tokens ``< position``), ``span``
   pages per step: ``Q @ C^T`` with each member's ``slots`` head slots
   (``packet_layout``) on sublanes, online softmax in FP32,
   ``O += P @ C[:, :512]``.
3. **Combine.** Each rank normalizes its partial (``o = O / l``, ``lse = m +
   log l``) and sends member ``d`` a BF16 ``[slots B, 640]`` packet of ``d``'s
   heads: row ``slots b + h``, ``o`` in lanes 0:512, ``lse`` as a high and a
   low BF16 part in lanes 512 and 513.
4. **Merge** on the owner: the log-sum-exp merge of the G partials plus the
   new token (replicated, so in no rank's pages yet), giving the latent-space
   output ``[B, heads, 512]`` (the caller applies W_UV).

Under attention DP (``heads`` = 6 for TP16 x DP2) the DCP group is the DP
group of ``32 / dp`` ranks, over that group's rows. Global page ``g`` lives on
member ``g % G``. Block tables are the same on every rank, in one of two
conventions (``KernelConfig.dcp_tables``):

* **local** (the default; vLLM's ``cp_kv_cache_interleave_size`` of one page):
  entry ``j`` covers G pages and every rank keeps its page in its block of that
  id; pools shrink by G, blocks are shared per G pages.
* **global**: entry ``g`` is page ``g``'s block, exactly vLLM's non-DCP table;
  only its owner fills that block, so every rank's pool has all the block ids.

The cache is vLLM's unified pools ``[blocks, 128, packing, 640]`` (packing 2 in
BF16, 4 in FP8); ``paged.commit_latent_rows`` commits new rows on their owners.
In-kernel pieces: :class:`DcpCache` (pool and table), :class:`DcpWorkspace`
(:func:`persistent_scratch`, which peers write into, so it must not alias other
scoped local memory, and :func:`scoped_scratch`) and :class:`DcpAttention`
(``prime`` the first page DMAs early, ``attend`` later).
"""

from __future__ import annotations

import contextlib
import dataclasses
from typing import TYPE_CHECKING, Protocol, cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from vllm_torchtpu.models.kimi_k3.megakernel.config import active as _config

from ..common.constants import (
    AXIS_TP,
    DP2_GROUPS,
    DP4_GROUPS,
    HEAD_DIM,
    MLA_LATENT_ROW_LANES,
    PAGE_TOKENS,
    SUBLANE_ROWS,
    TP_SIZE,
    VREG_LANE_COUNT,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from jax import typing as jax_typing

    from ..common.pool_alias import TransformedRef
    from ..common.protocols import PallasRefProtocol

    class _ScalarLookupProtocol(Protocol):
        """Structural protocol for 1D indexable scalar views."""

        def __getitem__(self, index: jax.Array | int, /) -> jax.Array:
            """Return scalar element at ``index``."""
            ...

    class _PageLoaderProtocol(Protocol):
        """Callable protocol reading lanes and optional token slices of a page slot."""

        def __call__(
            self,
            lanes: pl.Slice | slice,
            tokens: pl.Slice | slice | None = None,
            /,
        ) -> jax.Array:
            """Return page slice for ``lanes`` and optional ``tokens``."""
            ...


type _TensorOrRef = jax.Array | PallasRefProtocol
type _RowIndex = jax.Array | int
type _ScalarTable = _TensorOrRef | _ScalarLookupProtocol

LANES: int = MLA_LATENT_ROW_LANES  # latent 512 | rope 64 | zero 64
LATENT: int = 4 * HEAD_DIM
HEADS_PER_RANK: int = 3
NEG: float = -1e30  # finite "no tokens" log-sum-exp
# Batch rows of an unchunked call; a larger batch runs in chunks (see _attend_chunked).
MAX_ROWS: int = 8
HOST_LANES: int = 8  # DCP ranks per host (rank = host * 8 + lane)
DP4_HEADS: int = 12  # attention DP4's held heads per rank (four TP32 shards)
DP2_GROUP: int = TP_SIZE // DP2_GROUPS

_DP2_HEADS: int = DP2_GROUPS * HEADS_PER_RANK
_DP4_GROUP: int = TP_SIZE // DP4_GROUPS
_DCP_GROUPS: tuple[int, ...] = (_DP4_GROUP, DP2_GROUP, TP_SIZE)
_HOSTS_PER_DCP32: int = TP_SIZE // HOST_LANES

_DUAL_GROUP_BATCH: int = 2
_UNSCALED_SPAN_MAX_BATCH: int = 4
_GROUP_ROWS_16: int = 16
_GROUP_ROWS_24: int = 24
_GROUP_ROWS_32: int = 32
_GROUP_ROWS_48: int = 48
_GROUP_ROWS_64: int = 64
_DP4_LOOKUP_BATCHES: tuple[int, ...] = (
    _GROUP_ROWS_16,
    _GROUP_ROWS_24,
    _GROUP_ROWS_32,
)
_LOOKUP_BATCHES: tuple[int, ...] = (
    _GROUP_ROWS_32,
    _GROUP_ROWS_48,
    _GROUP_ROWS_64,
)
_PLAIN_FP8_DP2_BATCHES: tuple[int, ...] = (
    _GROUP_ROWS_16,
    _GROUP_ROWS_32,
    _GROUP_ROWS_48,
    _GROUP_ROWS_64,
)
_FP8_SPAN_BATCHES: tuple[int, ...] = (
    MAX_ROWS,
    _GROUP_ROWS_16,
    _GROUP_ROWS_32,
    _GROUP_ROWS_64,
)
_PAD_STORE_BATCHES: tuple[int, ...] = (
    _GROUP_ROWS_16,
    _GROUP_ROWS_32,
    _GROUP_ROWS_48,
    _GROUP_ROWS_64,
)

_NARROW_CHUNK_ROWS: int = 4
_WIDE_CHUNK_ROWS: int = 8
_RING_SLOTS_MIN: int = 2
_RING_SLOTS_STAGED: int = 3
_RING_SLOTS_QUAD: int = 4
_CHUNKED_BUFFERS: int = 3
_MIN_CHUNKED_SEQ_BUFFERS: int = 2
_NARROW_SPAN: int = 2
_PACKED_PAGE_NDIM: int = 3
_SCORE_TILE_ROW_ALIGN: int = 16

_DEFAULT_DECODE_KTILE: int = 1024
_WIDE_DECODE_KTILE: int = 2048
_FP8_PAGE_TOKENS: int = 2 * PAGE_TOKENS

_VERIFY_GROUP_ROWS: int = 4
_VERIFY_MAX_GROUP_ROWS: int = 8
_PEER_MERGE_UNROLL: int = 4
_MIN_STAGE2_DRAIN_CHUNKS: int = 2

_ROPE_LANES: int = HEAD_DIM // 2
_COMPRESSED_KV_LANES: int = LATENT + _ROPE_LANES
_BYTES_PER_FP32: int = 4
_BITS_PER_BYTE: int = 8
_BYTE_MASK: int = 255
_FP8_SCALE_END_LANE: int = _COMPRESSED_KV_LANES + _BYTES_PER_FP32
_FP8_E4M3_MAX: float = 448.0
_FP8_PROB_SCALE: int = 256
_LSE_PARTS: int = 2
_MIN_LSE_PADDING_SLOTS: int = 2


def _lookup_rows(batch: int, heads: int = HEADS_PER_RANK) -> bool:
    """Return whether page steps locate their row via binary search over scalar bounds.

    Applies to 32, 48, and 64 rows, plus attention DP4's 16, 24, and 32 group rows.
    """
    if heads == DP4_HEADS:
        return batch in _DP4_LOOKUP_BATCHES
    return batch in _LOOKUP_BATCHES


def chunk_layout(
    batch: int,
    heads: int = HEADS_PER_RANK,
    *,
    chunk_rows: int | None = None,
) -> tuple[int, int]:
    """Return ``(chunk rows, ring slots)`` for a chunked call (``batch > MAX_ROWS``).

    Uses 8-row chunks at B16/B32/B48/B64 and attention DP4's 24 group rows
    (amortizing exchanges), else 4 (also attention DP2's B128 and DP4's: 64
    group rows of six heads, 32 of twelve, for local scratch); ring slots
    4 at B32 with three heads per rank, 3 for 8-row chunks, else 2
    (``persistent_scratch`` uses 4 for FP8 query packets). ``chunk_rows``
    overrides the width and keeps at least three slots for a staged combine.
    """
    wide = batch <= _GROUP_ROWS_16 or (
        batch in _LOOKUP_BATCHES
        and not (batch == _GROUP_ROWS_64 and heads == _DP2_HEADS)
    )
    if heads == DP4_HEADS:
        wide = batch <= _GROUP_ROWS_24
    rows = (
        chunk_rows
        if chunk_rows is not None
        else _WIDE_CHUNK_ROWS
        if wide
        else _NARROW_CHUNK_ROWS
    )
    ring = (
        _RING_SLOTS_QUAD
        if batch == _GROUP_ROWS_32 and heads == HEADS_PER_RANK
        else _RING_SLOTS_STAGED
        if wide or chunk_rows is not None
        else _RING_SLOTS_MIN
    )
    if (
        rows not in (_NARROW_CHUNK_ROWS, _WIDE_CHUNK_ROWS)
        or batch % rows
        or rows > MAX_ROWS
        or ring not in (_RING_SLOTS_MIN, _RING_SLOTS_STAGED, _RING_SLOTS_QUAD)
    ):
        msg = (
            f"DCP chunks: {rows} rows must divide the batch {batch} "
            f"(<= {MAX_ROWS}), ring 2, 3 or 4"
        )
        raise ValueError(msg)
    return rows, ring


GROUP: int = _config().dcp_group
GLOBAL_PAGES: bool = _config().dcp_tables == "global"
SPAN: int = 4  # pages per step
BUFFERS: int = 4  # steps in flight


def page_plan(
    batch: int,
    heads: int = HEADS_PER_RANK,
    sequence_rows: int = 1,
    cache_dtype: jax_typing.DTypeLike = jnp.bfloat16,
    *,
    force_chunks: bool = False,
) -> dict[str, int]:
    """Return page span and in-flight buffer count for ``batch`` rows.

    The decoder's page buffers for ``batch`` rows of ``heads`` heads per rank
    (``sequence_rows`` per sequence): ``span`` pages per step and ``buffers``
    steps in flight, sized to fit VMEM. Up to four rows and plain decode:
    ``SPAN`` x ``BUFFERS`` (3 buffers chunked, except attention DP2's and
    DP4's FP8 B128, whose long-context page reads outrun three; FP8 B64 and
    B96 stage twice the span, see ``staged_folds``). Verifies above four rows:
    half the span, twice the buffers (4 at B64 and attention DP's B32), 3/4
    chunked.
    """
    if batch <= _UNSCALED_SPAN_MAX_BATCH:
        return {"span": SPAN, "buffers": BUFFERS}
    if sequence_rows == 1:
        deep = (
            (batch == _GROUP_ROWS_64 and heads == _DP2_HEADS)
            or (batch == _GROUP_ROWS_32 and heads == DP4_HEADS)
        ) and (cache_dtype == jnp.float8_e4m3fn)
        return {
            "span": SPAN,
            "buffers": _CHUNKED_BUFFERS if chunked(batch) and not deep else BUFFERS,
        }
    span = max(1, SPAN // 2)
    buffers = (
        BUFFERS
        if batch == _GROUP_ROWS_64
        or (batch == _GROUP_ROWS_32 and heads != HEADS_PER_RANK)
        else BUFFERS * SPAN // span
    )
    return {
        "span": span,
        "buffers": (
            buffers
            if not chunked(batch, force=force_chunks)
            else max(_MIN_CHUNKED_SEQ_BUFFERS, buffers * 3 // 4)
        ),
    }


MAX_GROUPS: int = 2  # row groups of a call (see packet_layout)


_B2_PACKED_HEAD_SLOTS: int = HEADS_PER_RANK + 1


def packet_layout(batch: int, heads: int = HEADS_PER_RANK) -> tuple[int, int]:
    """Return ``(head slots, row groups)`` for ``batch`` rows of ``heads`` heads.

    Head slots are the packet rows per rank and batch row. B2 runs two groups
    whose exchanges overlap the other group's attention (used at B2 only; above
    it the DMA count doubles); a group's packet slice must be a multiple of 8
    rows unless it has 4-row slots, so B2 pads its three heads with a zero
    slot. Otherwise one group of a slot per head.
    """
    return (
        (_B2_PACKED_HEAD_SLOTS, MAX_GROUPS)
        if batch == _DUAL_GROUP_BATCH and heads == HEADS_PER_RANK
        else (heads, 1)
    )


def owner_index(
    table: Sequence[int] | None,
    *,
    rank: jax.Array | int | None = None,
) -> jax.Array | int:
    """Return this rank's page-owner index from a static per-mesh-position table.

    ``table[p]`` gives the page-owner index for mesh position ``p`` (for
    example, its vLLM TP rank); ``None`` returns the mesh position. Works in a
    kernel or at the XLA level.
    """
    resolved_rank = jax.lax.axis_index(AXIS_TP) if rank is None else rank
    if table is None:
        return resolved_rank
    owner: jax.Array | int = 0
    for position, value in enumerate(table):
        owner = jnp.where(resolved_rank == position, value, owner)
    return owner


@dataclasses.dataclass(frozen=True)
class DcpCache:
    """Describe this rank's latent pages in HBM and block-table addressing.

    ``pool_ref`` holds the per-MLA-layer tuple of vLLM's unified pools
    (``[blocks, 128, packing, 640]``, token ``t`` of a block at
    ``[t // packing, t % packing]``), picked by the traced layer index
    ``layer``. ``table_ref`` is the flat ``[rows * pages]`` int32 block table
    in scalar memory (``pages`` entries per row), ``group`` the DCP group size,
    ``global_pages`` the table convention (see the module docstring).

    vLLM's pages hold stale bytes past a request's tokens and in lanes
    576:640 (freed blocks, KDA state that reads as NaN in bf16); the
    attention zeroes both before they reach the matrix unit.
    """

    pool_ref: Sequence[PallasRefProtocol]
    table_ref: _ScalarTable
    pages: int
    layer: _RowIndex | None = None
    group: int = GROUP
    global_pages: bool = GLOBAL_PAGES
    # Which pages this rank holds: global page g is on the rank whose owner
    # index is g % group. Default: its member index (mesh order); a P/D
    # decoder uses its vLLM TP rank (see owner_index), the order the KV
    # transfer's destination units follow.
    owner: _RowIndex | None = None
    # The table row of attention row ``b`` (default ``b``): attention DP
    # attends its group's rows of the batch-wide table.
    table_row: Callable[[_RowIndex], _RowIndex] | None = None

    @property
    def dtype(self) -> jnp.dtype:
        """Return element dtype of the KV-cache pools."""
        return self.pool_ref[0].dtype

    @property
    def page_tokens(self) -> int:
        """Return token capacity per physical KV-cache page."""
        ref = self.pool_ref[0]
        return ref.shape[-3] * ref.shape[-2]

    def start_page_copy(
        self,
        row: _RowIndex,
        local: _RowIndex,
        member: _RowIndex,
        destination: PallasRefProtocol,
        semaphore: PallasRefProtocol,
    ) -> None:
        """Start the DMA of ``row``'s local page ``local`` on ``member``."""
        layer = self.layer
        if layer is None:
            msg = "DcpCache.layer must be set before copying pages"
            raise ValueError(msg)
        entry = local * self.group + member if self.global_pages else local
        if self.table_row is not None:
            row = self.table_row(row)
        block = self.table_ref[row * self.pages + entry]
        for index, ref in enumerate(self.pool_ref):

            @pl.when(layer == index)
            def start(ref: PallasRefProtocol = ref) -> None:
                """Start async DMA from the active layer pool."""
                tpu.make_async_copy(ref.at[block], destination, semaphore).start()

    def start_native_copies(
        self,
        row: _RowIndex,
        locals_: Sequence[_RowIndex],
        member: _RowIndex,
        destinations: Sequence[PallasRefProtocol],
        semaphore: PallasRefProtocol,
    ) -> None:
        """Start retiling DMAs of ``row``'s local pages ``locals_`` on ``member``.

        Writes into ``[page_tokens, 640]`` ``destinations`` (see
        ``native_pages``): each DMA retiles a pool block into the buffer's
        native tile. Its descriptor setup is not predicated, so the layer's
        pool is picked by one binary tree of branches for all the pages, not
        per pool.
        """
        layer = self.layer
        if layer is None:
            msg = "DcpCache.layer must be set before copying pages"
            raise ValueError(msg)
        if self.table_row is not None:
            row = self.table_row(row)
        blocks = [
            self.table_ref[
                row * self.pages
                + (local * self.group + member if self.global_pages else local)
            ]
            for local in locals_
        ]

        def pick(first: int, end: int) -> None:
            """Select active pool via binary-tree predication and issue DMAs."""
            if end - first == 1:
                for block, destination in zip(blocks, destinations, strict=True):
                    tpu.make_async_copy(
                        self.pool_ref[first].at[block].reshape(self.page_tokens, LANES),
                        destination,
                        semaphore,
                    ).start()
                return
            middle = (first + end) // 2
            pl.when(layer < middle)(lambda: pick(first, middle))
            pl.when(layer >= middle)(lambda: pick(middle, end))

        pick(0, len(self.pool_ref))


@dataclasses.dataclass(frozen=True)
class DcpWorkspace:
    """Hold persistent peer-exchange and per-invocation scratch references for DCP."""

    # Persistent: peers' remote copies land here.
    gathered: PallasRefProtocol  # BF16 queries, or FP8 query/scale packets as uint8
    send: PallasRefProtocol  # [G, slots B, 640] bf16
    receive: PallasRefProtocol  # [2, G, slots B, 640] bf16
    gather_semaphores: (
        PallasRefProtocol  # DMA [2 (send, receive), 2 (parity), MAX_GROUPS]
    )
    combine_semaphores: PallasRefProtocol  # DMA [2, 2, MAX_GROUPS]
    # Scoped to one MLA invocation.
    queries: (
        PallasRefProtocol  # [B, head_rows, 640], cache dtype; head slots padded to 16
    )
    pages: PallasRefProtocol  # [buffers, span * page_rows, packing, 640], cache dtype
    page_semaphores: PallasRefProtocol  # DMA [buffers]
    maximum: PallasRefProtocol  # [B, head_rows, 1] f32 (per head slot)
    denominator: PallasRefProtocol  # [B, head_rows, 1] f32
    accumulator: PallasRefProtocol  # [B, head_rows, 512] f32
    own_rows: PallasRefProtocol  # [slots B, 640] f32 (chunked calls: chunk_rows_shape)
    new_rows: PallasRefProtocol  # [slots B, 640] f32 (chunked calls: chunk_rows_shape)
    merged: PallasRefProtocol  # chunked calls: chunk_rows_shape(512) merged outputs
    lookup_bounds: PallasRefProtocol
    lookup_counts: PallasRefProtocol
    query_scale: (
        PallasRefProtocol  # [held, heads, 1] FP32 dequantization scale for FP8 Q
    )


def chunked(batch: int, *, force: bool = False) -> bool:
    """Return whether a call of ``batch`` rows runs in chunks of ``MAX_ROWS``."""
    return batch > MAX_ROWS or bool(force)


GROUPED_SPEC_SHAPES: tuple[tuple[int, int], ...] = (
    (4, 4),
    (8, 4),
    (8, 8),
    (16, 8),
    (32, 8),
    (32, 4),
    (64, 8),
    (64, 4),
)


def _grouped_spec(
    batch: int,
    group: int,
    heads: int,
    sequence_rows: int,
) -> bool:
    """Return whether ``(batch, sequence_rows)`` uses grouped FP8 QK/PV transport.

    Supports DCP32 with three heads per rank, or attention DP2's DCP16 with six
    (the same 96-row score tile).
    """
    if group == DP2_GROUP and heads == _DP2_HEADS:
        return (batch, sequence_rows) in (
            *GROUPED_SPEC_SHAPES,
            (_GROUP_ROWS_16, _VERIFY_GROUP_ROWS),
        )
    return (
        group == TP_SIZE
        and heads == HEADS_PER_RANK
        and (batch, sequence_rows) in GROUPED_SPEC_SHAPES
    )


def persistent_scratch(
    batch: int,
    group: int = GROUP,
    heads: int = HEADS_PER_RANK,
    *,
    cache_dtype: jax_typing.DTypeLike = jnp.bfloat16,
    sequence_rows: int = 1,
    fp8_wire: bool = False,
    force_chunks: bool = False,
    chunk_rows: int | None = None,
) -> tuple[
    pl.MemoryRef,
    pl.MemoryRef,
    pl.MemoryRef,
    pl.MemoryRef,
    pl.MemoryRef,
]:
    """Build persistent peer-exchange buffer and semaphore specs for DCP."""
    # Send the receiver's identical FP8 operand and lossless FP32 scale
    # (``fp8_wire``: for any shape, e.g. the MLA DSpark draft's two heads).
    # Plain attention DP2 holds six heads for B/2 rows, DP4 twelve for B/4.
    # Speculative attention
    # retains full-precision owner queries for its causal new-token fold.
    resolved_dtype = jnp.dtype(cache_dtype)
    plain = sequence_rows == 1 and (
        (batch in (_GROUP_ROWS_32, _GROUP_ROWS_64) and heads == HEADS_PER_RANK)
        or (
            group == DP2_GROUP
            and batch in _PLAIN_FP8_DP2_BATCHES
            and heads == _DP2_HEADS
        )
        or (group == _DP4_GROUP and batch in _DP4_LOOKUP_BATCHES and heads == DP4_HEADS)
    )
    query_dtype = jnp.dtype(
        jnp.uint8
        if resolved_dtype == jnp.float8_e4m3fn
        and (plain or fp8_wire or _grouped_spec(batch, group, heads, sequence_rows))
        else jnp.bfloat16
    )
    bf16_dtype = jnp.dtype(jnp.bfloat16)
    if chunked(batch, force=force_chunks):
        chunk, ring = chunk_layout(batch, heads, chunk_rows=chunk_rows)
        rows = packet_layout(batch, heads)[0] * chunk
        if query_dtype == jnp.uint8 and not (group == DP2_GROUP and sequence_rows > 1):
            # The smaller query buffers make two-chunk lookahead fit at B64.
            # Attention DP's verifies keep one chunk: two change their outputs
            # at 128 rows.
            ring = _RING_SLOTS_QUAD
        return (
            tpu.VMEM((ring, group, rows, LANES), query_dtype),
            tpu.VMEM((group, rows, LANES), bf16_dtype),
            tpu.VMEM((ring, group, rows, LANES), bf16_dtype),
            tpu.SemaphoreType.DMA((2, ring)),
            tpu.SemaphoreType.DMA((2, ring)),
        )
    rows = packet_layout(batch, heads)[0] * batch
    return (
        tpu.VMEM((2, group, rows, LANES), query_dtype),
        tpu.VMEM((group, rows, LANES), bf16_dtype),
        tpu.VMEM((2, group, rows, LANES), bf16_dtype),
        tpu.SemaphoreType.DMA((2, 2, MAX_GROUPS)),
        tpu.SemaphoreType.DMA((2, 2, MAX_GROUPS)),
    )


PERSISTENT_NAMES: tuple[str, ...] = (
    "dcp gathered queries",
    "dcp send packets",
    "dcp received packets",
    "dcp gather semaphores",
    "dcp combine semaphores",
)


def native_pages(
    batch: int,
    heads: int = HEADS_PER_RANK,
    sequence_rows: int = 1,
    cache_dtype: jax_typing.DTypeLike = jnp.bfloat16,
) -> bool:
    """Return whether pages stage as ``[span * page_tokens, 640]`` native tiles.

    Avoids relayout loads from ``[span * 128, packing, 640]`` pool blocks on
    attention DP2's plain B96 and B128 (48 and 64 group rows of six heads) and
    FP8 B64 (32; not under the opt-in ``mla_decode_ktile=2048``), plus
    attention DP4's plain B96 and B128 and FP8 B64 (24, 32, and 16 rows).
    """
    resolved_dtype = jnp.dtype(cache_dtype)
    if heads == DP4_HEADS:
        return sequence_rows == 1 and (
            batch in (_GROUP_ROWS_24, _GROUP_ROWS_32)
            or (batch == _GROUP_ROWS_16 and resolved_dtype == jnp.float8_e4m3fn)
        )
    return (
        heads == _DP2_HEADS
        and sequence_rows == 1
        and (
            batch in (_GROUP_ROWS_48, _GROUP_ROWS_64)
            or (
                batch == _GROUP_ROWS_32
                and resolved_dtype == jnp.float8_e4m3fn
                and _config().mla_decode_ktile == _DEFAULT_DECODE_KTILE
            )
        )
    )


def staged_folds(
    batch: int,
    heads: int = HEADS_PER_RANK,
    sequence_rows: int = 1,
    cache_dtype: jax_typing.DTypeLike = jnp.bfloat16,
) -> int:
    """Return softmax folds executed per page step.

    Attention DP2's FP8 B64 and B96 (32 and 48 group rows) and DP4's (16 and
    24) stage two: one page wait, row lookup, pool pick, and QK per 2048
    tokens, folded as two 1024-token parts in order (preserving normalization
    and FP8 probability boundaries of 1024-token steps). Their rows end with a
    static tail (``DcpAttention._staged_tail``), which B128 lacks local memory
    for.
    """
    resolved_dtype = jnp.dtype(cache_dtype)
    if heads == DP4_HEADS:
        return (
            2
            if (
                native_pages(batch, heads, sequence_rows, resolved_dtype)
                and batch in (_GROUP_ROWS_16, _GROUP_ROWS_24)
                and resolved_dtype == jnp.float8_e4m3fn
            )
            else 1
        )
    return (
        2
        if (
            native_pages(batch, heads, sequence_rows, resolved_dtype)
            and batch in (_GROUP_ROWS_32, _GROUP_ROWS_48)
            and resolved_dtype == jnp.float8_e4m3fn
            and _config().mla_decode_ktile == _DEFAULT_DECODE_KTILE
        )
        else 1
    )


def scoped_scratch(
    batch: int,
    span: int = SPAN,
    buffers: int = BUFFERS,
    heads: int = HEADS_PER_RANK,
    group: int = GROUP,
    cache_dtype: jax_typing.DTypeLike = jnp.bfloat16,
    page_tokens: int = PAGE_TOKENS,
    *,
    sequence_rows: int = 1,
    force_chunks: bool = False,
    chunk_rows: int | None = None,
) -> tuple[pl.MemoryRef, ...]:
    """Build per-layer scoped attention buffer and semaphore specs for DCP."""
    rows = packet_layout(batch, heads)[0] * batch
    tile = head_rows(batch, group, heads)
    resolved_dtype = jnp.dtype(cache_dtype)
    f32_dtype = jnp.dtype(jnp.float32)
    i32_dtype = jnp.dtype(jnp.int32)
    packing = _BYTES_PER_FP32 // resolved_dtype.itemsize
    # FP8 doubles tokens per physical page. The opt-in mla_decode_ktile=2048
    # folds 2048 tokens per step for plain DP2 B64 (32 local rows).
    wide = _config().mla_decode_ktile == _WIDE_DECODE_KTILE
    if wide and (
        group != DP2_GROUP
        or heads != _DP2_HEADS
        or resolved_dtype != jnp.float8_e4m3fn
        or page_tokens != _FP8_PAGE_TOKENS
    ):
        msg = (
            "mla_decode_ktile=2048 requires DCP16, six heads per rank and "
            "FP8 e4m3fn KV with 512-token pages"
        )
        raise ValueError(msg)
    if resolved_dtype == jnp.float8_e4m3fn and batch in _FP8_SPAN_BATCHES:
        span = (
            SPAN
            if wide and batch == _GROUP_ROWS_32 and sequence_rows == 1
            else min(span, SPAN * PAGE_TOKENS // page_tokens)
        )
    else:
        span = span * PAGE_TOKENS // page_tokens
    span = max(1, span) * staged_folds(batch, heads, sequence_rows, resolved_dtype)
    pages = (
        (buffers, span * page_tokens, LANES)
        if native_pages(batch, heads, sequence_rows, resolved_dtype)
        else (buffers, span * page_tokens // packing, packing, LANES)
    )
    # Rows whose attention state is live at once.
    held = (
        chunk_layout(batch, heads, chunk_rows=chunk_rows)[0]
        if chunked(batch, force=force_chunks)
        else batch
    )
    row_specs = (
        (
            tpu.VMEM(
                chunk_rows_shape(batch, LANES, heads, chunk_rows=chunk_rows),
                f32_dtype,
            ),
            tpu.VMEM(
                chunk_rows_shape(batch, LANES, heads, chunk_rows=chunk_rows),
                f32_dtype,
            ),
            tpu.VMEM(
                chunk_rows_shape(batch, LATENT, heads, chunk_rows=chunk_rows),
                f32_dtype,
            ),
        )
        if chunked(batch, force=force_chunks)
        else (
            tpu.VMEM((rows, LANES), f32_dtype),
            tpu.VMEM((rows, LANES), f32_dtype),
            tpu.VMEM((SUBLANE_ROWS, VREG_LANE_COUNT), f32_dtype),
        )
    )
    lookup_size = batch if _lookup_rows(batch, heads) else 1
    return (
        tpu.VMEM((held, tile, LANES), resolved_dtype),
        tpu.VMEM(pages, resolved_dtype),
        tpu.SemaphoreType.DMA((buffers,)),
        tpu.VMEM((held, tile, 1), f32_dtype),
        tpu.VMEM((held, tile, 1), f32_dtype),
        tpu.VMEM((held, tile, LATENT), f32_dtype),
        *row_specs,
        tpu.SMEM((lookup_size,), i32_dtype),
        tpu.SMEM((lookup_size,), i32_dtype),
        tpu.VMEM((held, tile, 1), f32_dtype),
    )


def head_rows(
    batch: int,
    group: int = GROUP,
    heads: int = HEADS_PER_RANK,
) -> int:
    """Return score-tile row count, rounding the group's head slots up to 16."""
    slots = packet_layout(batch, heads)[0] * group
    return -(-slots // _SCORE_TILE_ROW_ALIGN) * _SCORE_TILE_ROW_ALIGN


def chunk_rows_shape(
    batch: int,
    lanes: int,
    heads: int = HEADS_PER_RANK,
    *,
    chunk_rows: int | None = None,
) -> tuple[int, int, int]:
    """Return ``[chunks, rows padded to 8, lanes]`` shape for chunked row buffers.

    A chunk's packet rows (``slots x chunk``) need not be a multiple of the
    float32 tile; chunks are indexed on the leading dimension.
    """
    chunk = chunk_layout(batch, heads, chunk_rows=chunk_rows)[0]
    rows = packet_layout(batch, heads)[0] * chunk
    return (batch // chunk, -(-rows // SUBLANE_ROWS) * SUBLANE_ROWS, lanes)


SCOPED_NAMES: tuple[str, ...] = (
    "dcp query columns",
    "dcp latent pages",
    "dcp page semaphores",
    "dcp maximum",
    "dcp denominator",
    "dcp accumulator",
    "dcp own rows",
    "dcp new rows",
    "dcp merged",
    "dcp lookup bounds",
    "dcp lookup counts",
    "dcp query scales",
)


def _validate_dcp_config(
    batch: int,
    heads: int,
    sequence_rows: int,
    group: int,
    chunk_rows: int | None,
) -> None:
    """Validate batch, chunk, sequence-row, and DCP group divisibility."""
    if batch > MAX_ROWS:
        chunk = chunk_layout(batch, heads, chunk_rows=chunk_rows)[0]  # validates
        if chunk % sequence_rows:
            msg = (
                f"DCP chunks of {chunk} rows must hold whole "
                f"{sequence_rows}-row sequences"
            )
            raise ValueError(msg)
    if sequence_rows < 1 or batch % sequence_rows:
        msg = (
            "sequence_rows must divide the batch, "
            f"got batch={batch}, sequence_rows={sequence_rows}"
        )
        raise ValueError(msg)
    if group not in _DCP_GROUPS:
        msg = f"DCP groups are 8, 16 or 32 consecutive ranks, got group={group}"
        raise ValueError(msg)


@dataclasses.dataclass(frozen=True, slots=True)
class _HostCombineBuffers:
    """Hold stage-two buffers, semaphores, and topology for host combine."""

    stage2_recv: PallasRefProtocol
    stage2_send: PallasRefProtocol
    stage2_sems: PallasRefProtocol
    hosts: int
    host: _RowIndex
    lane: _RowIndex


@dataclasses.dataclass(frozen=True, slots=True)
class _ChunkedAttendPlan:
    """Hold chunk geometry, ring schedule, and combine state for chunked DCP."""

    chunk: int
    count: int
    rows: int
    heads: int
    ring_slots: int
    lookahead: int
    layer: _RowIndex
    peers_bytes: int
    steps: tuple[_RowIndex, ...]
    keys: _TensorOrRef | None
    key_valid: _TensorOrRef | None
    host_combine: _HostCombineBuffers | None

    def ring(self, chunk_index: _RowIndex) -> _RowIndex:
        """Return persistent ring-buffer slot for ``chunk_index``."""
        return (self.layer * self.count + chunk_index) % self.ring_slots

    def first_step(self, first_row: _RowIndex) -> jax.Array:
        """Return cumulative page-step offset preceding ``first_row``."""
        total = jnp.int32(0)
        for batch_row, row_steps in enumerate(self.steps):
            total = total + jnp.where(batch_row < first_row, row_steps, 0)
        return total


class DcpAttention:
    """Execute one DCP attention invocation for ``batch`` rows at ``position_ref``."""

    # Profiling: named scopes inside attend (``trace`` their prefix; None: none).
    trace: str | None = None
    # Explicit-keys sequences whose last row is padding (the MLA DSpark
    # draft): partials carry their log-sum-exps in that row, and the combine
    # moves only the 512 latent lanes (a fifth fewer bytes).
    lse_in_padding: bool = False
    # Chunked explicit-keys calls (lse_in_padding, DCP32): the combine in two
    # stages, (stage-2 receive [ring, hosts, rows, 640], stage-2 send [hosts,
    # rows, 640], DMA semaphores [2, ring]) or None. Stage one stays on the
    # host: each rank sends a host-mate the partials of the owners on its
    # lane (one per host) and merges the eight it receives per owner; stage
    # two sends each merged partial to its owner across hosts, which merges
    # four. Cross-host combine bytes drop eightfold (the cross-host links
    # bound the exchange); the merged partials round to BF16 once more.
    host_combine: (
        tuple[
            PallasRefProtocol,
            PallasRefProtocol,
            PallasRefProtocol,
        ]
        | None
    ) = None
    # Run even one chunk through _attend_chunked (scratch: force_chunks).
    force_chunked: bool = False
    # FP8 explicit-keys sequences: QK/PV for several rows in one
    # dot (the grouped path's arithmetic, per row as the ungrouped fold).
    group_fold: bool = False
    # Packing for host_combine: send packet of peer p at (p % 8) hosts + p // 8.
    _lane_major: bool = False

    def __init__(
        self,
        cache: DcpCache,
        workspace: DcpWorkspace,
        position_ref: _ScalarTable,
        batch: int,
        *,
        skip: Iterable[str] = (),
        sequence_rows: int = 1,
        heads: int = HEADS_PER_RANK,
        chunk_rows: int | None = None,
        cache_verify_rows: bool = False,
    ) -> None:
        """Initialize per-layer DCP page schedule and workspace bindings."""
        _validate_dcp_config(batch, heads, sequence_rows, cache.group, chunk_rows)
        self.cache = cache
        self.ws = workspace
        self.position_ref = position_ref
        self.batch = batch
        # sequence_rows > 1 (speculative verify): each group of rows is one
        # sequence's consecutive tokens. All rows attend the pool below the
        # group's first position; its new tokens fold in causally after the merge.
        self.sequence_rows = sequence_rows
        self.chunk_rows = chunk_rows
        self.cache_verify_rows = cache_verify_rows
        self.heads = heads
        self.slots, self.groups = packet_layout(batch, heads)
        self.group = cache.group
        self.tile = workspace.queries.shape[1]  # score-tile rows (head_rows)
        if self.group * self.slots > self.tile:
            msg = (
                f"{self.group} ranks x {self.slots} head slots exceed {self.tile} rows"
            )
            raise ValueError(msg)
        self.buffers = workspace.pages.shape[0]
        self.page_tokens = cache.page_tokens
        self.fp8 = cache.dtype == jnp.float8_e4m3fn
        self.packing = _BYTES_PER_FP32 // jnp.dtype(cache.dtype).itemsize
        self.page_rows = self.page_tokens // self.packing
        self.native = workspace.pages.ndim == _PACKED_PAGE_NDIM  # native_pages
        self.span = workspace.pages.shape[1] // (
            self.page_tokens if self.native else self.page_rows
        )
        self.folds = staged_folds(batch, heads, sequence_rows, cache.dtype)
        self.fold_tokens = self.span * self.page_tokens // self.folds
        self.skip = set(skip)
        rank = jax.lax.axis_index(AXIS_TP)
        self.member = rank % self.group
        self.base = rank - self.member
        self.owner: _RowIndex = self.member if cache.owner is None else cache.owner
        # Local pages per row: global pages below ceil(limit / page_tokens)
        # congruent to ``owner``, ``span`` per step; the rows' steps form one
        # flat sequence. Shared (whole sequences per row group): a sequence's
        # first row owns its steps and each step folds into all its rows.
        self.shared = sequence_rows > 1 and all(
            first % sequence_rows == 0 and end % sequence_rows == 0
            for first, end in self.row_groups()
        )
        if cache_verify_rows and not (self.shared and self.fp8):
            msg = (
                "cached verify rows need shared FP8 sequences, "
                f"got shared={self.shared}, cache_dtype={cache.dtype}"
            )
            raise ValueError(msg)
        self.counts, self.starts = self._init_step_schedule()
        self.total: _RowIndex = self.starts[-1]

    def _init_step_schedule(self) -> tuple[list[_RowIndex], list[_RowIndex]]:
        """Compute per-row local page counts, step starts, and scalar lookup tables."""
        counts: list[_RowIndex] = []
        starts: list[_RowIndex] = [0]
        extra = self.sequence_rows - 1 if self.cache_verify_rows else 0
        for b in range(self.batch):
            limit = self._limit(b) + extra
            needed = (limit + self.page_tokens - 1) // self.page_tokens
            count = jnp.maximum(needed - self.owner + self.group - 1, 0) // self.group
            counts.append(
                count if not self.shared or b % self.sequence_rows == 0 else 0
            )
            starts.append(starts[-1] + (counts[-1] + self.span - 1) // self.span)
        if _lookup_rows(self.batch, self.heads):
            for b in range(self.batch):
                self.ws.lookup_bounds[b] = starts[b]
                self.ws.lookup_counts[b] = counts[b]
        return counts, starts

    def _limit(self, row: _RowIndex) -> jax.Array:
        """Return pool token upper bound visible to ``row`` (which may be traced).

        Equals ``row``'s position, or its sequence's first position when
        ``sequence_rows > 1``.
        """
        if self.sequence_rows == 1:
            return self.position_ref[row]
        return self.position_ref[row // self.sequence_rows * self.sequence_rows]

    def _locate(self, step: _RowIndex) -> tuple[_RowIndex, _RowIndex, _RowIndex]:
        """Resolve ``(row, local_step, page_count)`` for global page ``step``."""
        if _lookup_rows(self.batch, self.heads):
            row = jnp.int32(0)
            for bit in range((self.batch - 1).bit_length() - 1, -1, -1):
                candidate = jnp.minimum(row + (1 << bit), self.batch - 1)
                row = jnp.where(
                    step >= self.ws.lookup_bounds[candidate],
                    candidate,
                    row,
                )
            return (
                row,
                step - self.ws.lookup_bounds[row],
                self.ws.lookup_counts[row],
            )
        row_val: _RowIndex = 0
        first_val: _RowIndex = 0
        count_val: _RowIndex = self.counts[0]
        for b in range(1, self.batch):
            later = step >= self.starts[b]
            row_val = jnp.where(later, b, row_val)
            first_val = jnp.where(later, self.starts[b], first_val)
            count_val = jnp.where(later, self.counts[b], count_val)
        return row_val, step - first_val, count_val

    def _start_step(self, step: _RowIndex) -> None:
        """Issue ``span`` page DMAs for ``step``, clamping past the row's last page."""
        row, local_step, count = self._locate(step)
        slot = step % self.buffers
        if self.native:
            self.cache.start_native_copies(
                row,
                [
                    jnp.minimum(local_step * self.span + index, count - 1)
                    for index in range(self.span)
                ],
                self.owner,
                [
                    self.ws.pages.at[
                        slot,
                        pl.ds(index * self.page_tokens, self.page_tokens),
                    ]
                    for index in range(self.span)
                ],
                self.ws.page_semaphores.at[slot],
            )
            return
        for index in range(self.span):
            local = jnp.minimum(local_step * self.span + index, count - 1)
            self.cache.start_page_copy(
                row,
                local,
                self.owner,
                self.ws.pages.at[slot, pl.ds(index * self.page_rows, self.page_rows)],
                self.ws.page_semaphores.at[slot],
            )

    def _wait_step(self, step: _RowIndex) -> None:
        """Wait for the in-flight page DMAs of ``step`` to arrive."""
        slot = step % self.buffers
        tpu.make_async_copy(
            self.ws.pages.at[slot],
            self.ws.pages.at[slot],
            self.ws.page_semaphores.at[slot],
        ).wait()

    def _attend_step(self, step: _RowIndex, base: _RowIndex = 0) -> None:
        """Execute one local-attention step, prefetching ahead and folding scores.

        Running softmax state is indexed at ``row - base`` (the chunk's first row).
        """
        self._wait_step(step)

        @pl.when(step + self.buffers - 1 < self.total)
        def prefetch() -> None:
            """Prefetch the next pipeline-depth page step."""
            self._start_step(step + self.buffers - 1)

        if "pagecompute" in self.skip:
            return
        row, visible, load = self._step_pages(step)
        if self.cache_verify_rows:
            self._attend_cached_verify(step, row, load, base)
        elif self.shared:
            self._attend_shared_step(row, base, visible, load)
        else:
            self._fold_pages(load, visible, row - base)

    def _attend_shared_step(
        self,
        row: _RowIndex,
        base: _RowIndex,
        visible: jax.Array,
        load: _PageLoaderProtocol,
    ) -> None:
        """Dispatch shared multi-row sequence folds across FP8 and BF16 regimes."""
        if (
            self.fp8
            and self.group_fold
            and self.sequence_rows in (_VERIFY_GROUP_ROWS, _VERIFY_MAX_GROUP_ROWS)
        ):
            self._fold_explicit_group_rows(load, visible, row, base)
        elif self.fp8 and _grouped_spec(
            self.batch,
            self.group,
            self.heads,
            self.sequence_rows,
        ):
            self._fold_grouped_spec_rows(load, visible, row, base)
        elif self.fp8 or self.batch in (_GROUP_ROWS_16, _GROUP_ROWS_32):
            self._fold_pipelined_shared_rows(load, visible, row, base)
        else:
            # BF16 verifier: fold rows one at a time (no QK lookahead).
            @pl.loop(0, self.sequence_rows - (1 if self.lse_in_padding else 0))
            def fold_row(r: jax.Array) -> None:
                """Fold one shared sequence row into its softmax state."""
                self._fold_pages(load, visible, row - base + r)

    def _fold_explicit_group_rows(
        self,
        load: _PageLoaderProtocol,
        visible: jax.Array,
        row: _RowIndex,
        base: _RowIndex,
    ) -> None:
        """Fold 4-row or 8-row explicit-key sequences using grouped QK/PV dots."""
        active = self.sequence_rows - (1 if self.lse_in_padding else 0)
        first_rows = min(_VERIFY_GROUP_ROWS, active)
        first = self._group_scores(load, visible, row - base, first_rows)
        if active > _VERIFY_GROUP_ROWS:
            second = self._group_scores(
                load,
                visible,
                row - base + _VERIFY_GROUP_ROWS,
                active - _VERIFY_GROUP_ROWS,
            )
            self._group_fold(first, load, row - base, first_rows)
            self._group_fold(
                second,
                load,
                row - base + _VERIFY_GROUP_ROWS,
                active - _VERIFY_GROUP_ROWS,
            )
        else:
            self._group_fold(first, load, row - base, first_rows)

    def _fold_grouped_spec_rows(
        self,
        load: _PageLoaderProtocol,
        visible: jax.Array,
        row: _RowIndex,
        base: _RowIndex,
    ) -> None:
        """Fold grouped 4-query tiles with QK lookahead across sequence rows."""

        # Four independent queries fill the matrix row dimension together.
        # Preserve each query's softmax reduction and FP8 conversion; only
        # QK/PV dispatch is grouped. One/two groups cover four/eight rows.
        def fold_group(g: jax.Array, score: jax.Array) -> jax.Array:
            """Issue next 4-query QK and fold current 4-query group."""
            upcoming = self._group_scores(
                load,
                visible,
                row - base + (g + 1) * _VERIFY_GROUP_ROWS,
                _VERIFY_GROUP_ROWS,
            )
            self._group_fold(
                score,
                load,
                row - base + g * _VERIFY_GROUP_ROWS,
                _VERIFY_GROUP_ROWS,
            )
            return upcoming

        final = jax.lax.fori_loop(
            0,
            self.sequence_rows // _VERIFY_GROUP_ROWS - 1,
            fold_group,
            self._group_scores(load, visible, row - base, _VERIFY_GROUP_ROWS),
        )
        self._group_fold(
            final,
            load,
            row - base + self.sequence_rows - _VERIFY_GROUP_ROWS,
            _VERIFY_GROUP_ROWS,
        )

    def _fold_pipelined_shared_rows(
        self,
        load: _PageLoaderProtocol,
        visible: jax.Array,
        row: _RowIndex,
        base: _RowIndex,
    ) -> None:
        """Fold shared rows with single-row QK lookahead and peeled final PV."""

        # Each query has independent softmax state. Issue the next QK
        # before folding the current scores so its matrix work can overlap
        # softmax/PV. Keep this page buffer live until the final query's
        # PV completes; the caller refills it only in a later page step.
        # Peeling the last fold avoids an unused extra QK per page tile.
        def fold_row(r: jax.Array, score: jax.Array) -> jax.Array:
            """Issue next row QK and fold current row scores."""
            upcoming = self._scores(load, visible, row - base + r + 1)
            self._fold_scores(score, load, row - base + r)
            return upcoming

        # ``lse_in_padding``: the sequence's last row is padding; skip it.
        active = self.sequence_rows - (1 if self.lse_in_padding else 0)
        final = jax.lax.fori_loop(
            0,
            active - 1,
            fold_row,
            self._scores(load, visible, row - base),
        )
        self._fold_scores(final, load, row - base + active - 1)

    def _cache_verify_row(
        self,
        slot: _RowIndex,
        local_step: _RowIndex,
        row: _RowIndex,
    ) -> None:
        """Stage an earlier verify token's new latent into the in-flight page slot."""
        position = self.position_ref[row]
        page = position // self.page_tokens
        local_page = page // self.group - local_step * self.span

        @pl.when(
            (page % self.group == self.owner)
            & (local_page >= 0)
            & (local_page < self.span)
        )
        def write() -> None:
            """Write quantized verify latent into the matching sublane tile."""
            if chunked(self.batch, force=self.force_chunked):
                chunk = chunk_layout(
                    self.batch,
                    self.heads,
                    chunk_rows=self.chunk_rows,
                )[0]
                latent = self.ws.new_rows[row // chunk, self.slots * (row % chunk)]
            else:
                latent = self.ws.new_rows[self.slots * row]
            offset = local_page * self.page_tokens + position % self.page_tokens
            tile_tokens = SUBLANE_ROWS * self.packing
            first = pl.multiple_of(
                offset // tile_tokens * SUBLANE_ROWS,
                SUBLANE_ROWS,
            )
            words = (
                self.ws.pages.bitcast(jnp.uint32).at[slot]
                if self.native
                else self.ws.pages.bitcast(jnp.uint32)
                .at[slot]
                .reshape(self.span * self.page_rows, LANES)
            )
            block = tpu.bitcast(
                words[pl.ds(first, SUBLANE_ROWS)],
                self.cache.dtype,
            )
            row_iota = jax.lax.broadcasted_iota(jnp.int32, block.shape, 0)
            block = jnp.where(
                row_iota == offset % tile_tokens,
                latent[None, :].astype(self.cache.dtype),
                block,
            )
            words[pl.ds(first, SUBLANE_ROWS)] = tpu.bitcast(block, jnp.uint32)

    def _attend_cached_verify(
        self,
        step: _RowIndex,
        row: _RowIndex,
        load: _PageLoaderProtocol,
        base: _RowIndex,
    ) -> None:
        """Stage verify rows into ``step``'s page buffer and fold causal scores."""
        # Earlier verify tokens use the cache's FP8 QK/PV, like sequential decode.
        _, local_step, _ = self._locate(step)

        @pl.loop(0, self.sequence_rows - 1)
        def cache(r: jax.Array) -> None:
            """Insert verify row ``r`` into the current page buffer if owned."""
            self._cache_verify_row(step % self.buffers, local_step, row + r)

        offset = jax.lax.broadcasted_iota(
            jnp.int32,
            (1, self.span * self.page_tokens),
            1,
        )
        page = (
            local_step * self.span + offset // self.page_tokens
        ) * self.group + self.owner
        token_pos = page * self.page_tokens + offset % self.page_tokens
        if _grouped_spec(self.batch, self.group, self.heads, self.sequence_rows):
            for first in range(0, self.sequence_rows, _VERIFY_GROUP_ROWS):
                query_idx = (
                    jax.lax.broadcasted_iota(
                        jnp.int32,
                        (_VERIFY_GROUP_ROWS * self.tile, 1),
                        0,
                    )
                    // self.tile
                )
                visible = token_pos < self.position_ref[row] + first + query_idx
                score = self._group_scores(
                    load,
                    visible,
                    row - base + first,
                    _VERIFY_GROUP_ROWS,
                )
                self._group_fold(
                    score,
                    load,
                    row - base + first,
                    _VERIFY_GROUP_ROWS,
                )
        else:

            @pl.loop(0, self.sequence_rows)
            def query(r: jax.Array) -> None:
                """Fold causal page scores for verify row ``r``."""
                self._fold_pages(
                    load,
                    token_pos < self.position_ref[row + r],
                    row - base + r,
                )

    def _step_pages(
        self,
        step: _RowIndex,
    ) -> tuple[_RowIndex, jax.Array, _PageLoaderProtocol]:
        """Return step ``step``'s row, visible-token mask, and page reader.

        Clears stale page bytes before returning the reader (their stale bytes
        zeroed first).
        """
        row, local_step, _ = self._locate(step)
        # Tokens on lanes.
        offset = jax.lax.broadcasted_iota(
            jnp.int32,
            (1, self.span * self.page_tokens),
            1,
        )
        global_page = (
            local_step * self.span + offset // self.page_tokens
        ) * self.group + self.owner
        token_pos = global_page * self.page_tokens + offset % self.page_tokens
        visible = token_pos < self._limit(row)
        slot = step % self.buffers
        self._clean_step(slot, row, local_step)
        return row, visible, self._pages(slot)

    def _pages(self, slot: _RowIndex) -> _PageLoaderProtocol:
        """Return a ``load(lanes, tokens=None)`` reader for page buffer ``slot``."""
        if self.native:

            def native(
                lanes: pl.Slice | slice,
                tokens: pl.Slice | slice | None = None,
            ) -> jax.Array:
                """Read native-layout page lanes and optional token slice."""
                return self.ws.pages[
                    slot,
                    slice(None) if tokens is None else tokens,
                    lanes,
                ]

            return native
        words = (
            self.ws.pages.bitcast(jnp.uint32)
            .at[slot]
            .reshape(self.span * self.page_rows, LANES)
        )

        def packed(
            lanes: pl.Slice | slice,
            tokens: pl.Slice | slice | None = None,
        ) -> jax.Array:
            """Read packed-layout page lanes bitcast to the cache dtype."""
            if tokens is not None:
                msg = f"token ranges need native pages, got tokens={tokens!r}"
                raise ValueError(msg)
            return tpu.bitcast(words[:, lanes], self.cache.dtype)

        return packed

    def _uses_split_tail_range(self) -> bool:
        """Return whether ``_attend_range`` splits partial tail steps per row."""
        if "attend" in self.skip or "pagecompute" in self.skip:
            return False
        if self.folds > 1:
            return True
        dp2_32 = (
            self.group == DP2_GROUP
            and self.heads == _DP2_HEADS
            and self.batch == _GROUP_ROWS_32
        )
        valid_heads = self.heads == HEADS_PER_RANK or dp2_32
        valid_span = self.span == _NARROW_SPAN or (
            _config().mla_decode_ktile == _WIDE_DECODE_KTILE and self.span == SPAN
        )
        return (
            self.sequence_rows == 1
            and valid_heads
            and self.batch in (_GROUP_ROWS_32, _GROUP_ROWS_64)
            and self.ws.gathered.dtype == jnp.uint8
            and valid_span
        )

    def _fold_partial_tail_step(
        self,
        full_end: _RowIndex,
        held: _RowIndex,
        tail_pages: int,
    ) -> None:
        """Wait, prefetch, and fold a static ``tail_pages`` partial final step."""
        self._wait_step(full_end)

        @pl.when(full_end + self.buffers - 1 < self.total)
        def prefetch() -> None:
            """Prefetch ahead of the split-tail step."""
            self._start_step(full_end + self.buffers - 1)

        _, visible, load = self._step_pages(full_end)
        tail_tokens = tail_pages * self.page_tokens
        score = self._scores(load, visible, held, tail_tokens=tail_tokens)
        self._fold_scores(score, load, held, tail_tokens=tail_tokens)

    def _attend_row_with_tail(self, row: _RowIndex, base: _RowIndex) -> None:
        """Run full steps of ``row`` with peeled PV, followed by its partial tail."""
        begin = self.ws.lookup_bounds[row]
        count = self.ws.lookup_counts[row]
        full_end = begin + count // self.span
        self._attend_pipeline(begin, full_end, base, peel=True)
        if self.folds > 1:
            pl.when(count % self.span != 0)(
                lambda: self._staged_tail(full_end, row, count, base)
            )
            return
        for tail_pages in range(1, self.span):

            @pl.when(count % self.span == tail_pages)
            def tail(tail_pages: int = tail_pages) -> None:
                """Fold static partial tail of ``tail_pages`` pages."""
                self._fold_partial_tail_step(full_end, row - base, tail_pages)

    def _attend_range(
        self,
        first: _RowIndex,
        end: _RowIndex,
        base: _RowIndex,
    ) -> None:
        """Attend page steps ``first .. end - 1`` for chunk starting at ``base``."""
        # FP8-query plain decode at B32/B64 (and DP2's 32 local rows), and staged
        # steps: rows with a partial last step fold its 1 .. span - 1 pages with
        # static tail shapes.
        if not self._uses_split_tail_range():
            self._attend_pipeline(first, end, base)
            return
        chunk = chunk_layout(self.batch, self.heads, chunk_rows=self.chunk_rows)[0]

        def split() -> None:
            """Attend each row in the chunk with peeled full steps and static tail."""

            @pl.loop(0, chunk)
            def row_loop(j: jax.Array) -> None:
                """Attend chunk row ``j`` and its partial last step."""
                self._attend_row_with_tail(base + j, base)

        if self.folds > 1:
            # Staged chunks always run row by row (one inlined page loop): a
            # chunk's rows rarely all end on a whole step.
            split()
            return
        # Branch once per chunk so QK/PV shapes in the page loop stay static.
        remainder: _RowIndex = jnp.int32(0)
        for j in range(chunk):
            remainder = remainder | (self.ws.lookup_counts[base + j] % self.span)

        @pl.when(remainder == 0)
        def whole() -> None:
            """Run unified chunk pipeline when every row ends on a full step."""
            self._attend_pipeline(first, end, base)

        pl.when(remainder != 0)(split)

    def _staged_tail(
        self,
        step: _RowIndex,
        row: _RowIndex,
        count: _RowIndex,
        base: _RowIndex,
    ) -> None:
        """Fold a staged row's partial last step using static part and page shapes.

        Processes a whole 1024-token part if present, then a single-page
        remainder, matching 1024-token step normalization boundaries.
        """
        self._wait_step(step)

        @pl.when(step + self.buffers - 1 < self.total)
        def prefetch() -> None:
            """Prefetch ahead of the staged partial step."""
            self._start_step(step + self.buffers - 1)

        held, slot = row - base, step % self.buffers
        local_step, pages = count // self.span, count % self.span
        part_pages = self.span // self.folds
        self._clean_step(
            slot,
            row,
            local_step,
            (self.span - 1) * self.page_tokens,
            end=pages * self.page_tokens,
        )
        load = self._pages(slot)
        limit = self._limit(row)

        def visible(first_page: _RowIndex, tokens: int) -> jax.Array:
            """Compute visibility mask for ``tokens`` starting at ``first_page``."""
            offset = jax.lax.broadcasted_iota(jnp.int32, (1, tokens), 1)
            global_page = (
                local_step * self.span + first_page + offset // self.page_tokens
            ) * self.group + self.owner
            return global_page * self.page_tokens + offset % self.page_tokens < limit

        @pl.when(pages >= part_pages)
        def whole() -> None:
            """Fold the leading full 1024-token half-step."""
            tokens = pl.ds(0, self.fold_tokens)
            score = self._scores(
                lambda lanes: load(lanes, tokens),
                visible(0, self.fold_tokens),
                held,
            )
            self._fold_part(score, lambda: load(pl.ds(0, LATENT), tokens), held)

        @pl.when(pages % part_pages != 0)
        def remainder() -> None:
            """Fold the trailing single-page remainder."""
            first = pages - 1
            tokens = pl.ds(
                pl.multiple_of(
                    cast("jax.Array", first * self.page_tokens),
                    self.page_tokens,
                ),
                self.page_tokens,
            )
            score = self._scores(
                lambda lanes: load(lanes, tokens),
                visible(first, self.page_tokens),
                held,
            )
            score = jnp.pad(
                score,
                ((0, 0), (0, self.fold_tokens - self.page_tokens)),
                constant_values=-jnp.inf,
            )
            self._fold_part(
                score,
                lambda: load(pl.ds(0, LATENT), tokens),
                held,
                self.page_tokens,
            )

    def _attend_pipeline(
        self,
        first: _RowIndex,
        end: _RowIndex,
        base: _RowIndex,
        *,
        peel: bool = False,
    ) -> None:
        """Pipeline QK lookahead across steps ``first .. end - 1`` from ``base``.

        Plain decode issues step ``s + 1``'s scores before step ``s``'s softmax
        and PV so the next QK overlaps them, refilling a step's buffer only
        after its PV. Normally the last step recomputes its own scores (unused),
        avoiding a branch before QK. ``peel`` is static: the split-tail path
        finishes PV outside the loop before issuing the narrower tail QK/PV.
        """
        if self.sequence_rows > 1 or "pagecompute" in self.skip:
            pl.loop(first, end)(lambda step: self._attend_step(step, base=base))
            return

        def scores(step: _RowIndex) -> tuple[jax.Array, _RowIndex]:
            """Compute masked QK scores and owning row for ``step``."""
            row, visible, load = self._step_pages(step)
            return self._scores(load, visible, row - base), row

        @pl.when(first < end)
        def attend() -> None:
            """Run pipelined page loop over non-empty step range."""
            self._wait_step(first)

            @pl.when(first + self.buffers - 1 < self.total)
            def prefetch_first() -> None:
                """Prefetch ahead of the initial step."""
                self._start_step(first + self.buffers - 1)

            def fold(
                step: jax.Array,
                carry: tuple[jax.Array, _RowIndex],
            ) -> tuple[jax.Array, _RowIndex]:
                """Overlap next-step QK with current-step softmax and PV."""
                score, row = carry
                more = step + 1 < end
                if peel:
                    self._wait_step(step + 1)
                    upcoming = scores(step + 1)
                else:

                    @pl.when(more)
                    def wait_next() -> None:
                        """Wait for next step's page DMA when in range."""
                        self._wait_step(step + 1)

                    upcoming = scores(jnp.minimum(step + 1, end - 1))
                self._fold_scores(
                    score,
                    self._pages(step % self.buffers),
                    row - base,
                )

                @pl.when((True if peel else more) & (step + self.buffers < self.total))
                def prefetch() -> None:
                    """Refill completed page slot with future step."""
                    self._start_step(step + self.buffers)

                return upcoming

            carry = jax.lax.fori_loop(first, end - int(peel), fold, scores(first))
            if peel:
                score, row = carry
                self._fold_scores(
                    score,
                    self._pages((end - 1) % self.buffers),
                    row - base,
                )

    def _store_query(self, held: int, query: jax.Array) -> None:
        """Quantize or cast ``query`` once outside the streamed page loop.

        Scale each absorbed head independently; its latent/RoPE columns share
        one scale, so the QK dot needs only one FP32 multiply per score row.
        Padding heads are zero and get scale one. KV uses scale one.
        """
        if self.fp8:
            maximum = jnp.max(jnp.abs(query), axis=1, keepdims=True)
            scale = jnp.where(maximum > 0, maximum / _FP8_E4M3_MAX, 1.0)
            self.ws.queries[held] = (query / scale).astype(jnp.float8_e4m3fn)
            self.ws.query_scale[held] = scale
        else:
            self.ws.queries[held] = query.astype(jnp.bfloat16)

    def _group_scores(
        self,
        load: _PageLoaderProtocol,
        visible: jax.Array,
        held: _RowIndex,
        count: int,
    ) -> jax.Array:
        """Compute masked QK scores for ``count`` adjacent queries from ``held``."""
        query = jnp.concatenate(
            [self.ws.queries[held + i] for i in range(count)],
            axis=0,
        )
        scales = jnp.concatenate(
            [self.ws.query_scale[held + i] for i in range(count)],
            axis=0,
        )
        latent = load(pl.ds(0, LANES))
        score = jax.lax.dot_general(
            query,
            latent,
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        return jnp.where(visible, score * scales, -jnp.inf)

    def _group_fold(
        self,
        score: jax.Array,
        load: _PageLoaderProtocol,
        held: _RowIndex,
        count: int,
    ) -> None:
        """Fold grouped PV output back into each query's online softmax state."""
        ws = self.ws
        previous = jnp.concatenate(
            [ws.maximum[held + i] for i in range(count)],
            axis=0,
        )
        new_maximum = jnp.maximum(previous, jnp.max(score, axis=1, keepdims=True))
        scale = jnp.exp(previous - new_maximum)
        probabilities = jnp.exp(score - new_maximum)
        totals = jnp.sum(probabilities, axis=1, keepdims=True)
        high = (probabilities * _FP8_PROB_SCALE).astype(jnp.float8_e4m3fn)
        values = load(pl.ds(0, LATENT))
        weighted = jax.lax.dot_general(
            high,
            values,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        ) * (1.0 / _FP8_PROB_SCALE)
        for i in range(count):
            rows = slice(i * self.tile, (i + 1) * self.tile)
            ws.denominator[held + i] = (
                ws.denominator[held + i] * scale[rows] + totals[rows]
            )
            ws.maximum[held + i] = new_maximum[rows]
            ws.accumulator[held + i] = (
                ws.accumulator[held + i] * scale[rows] + weighted[rows]
            )

    def _fold_pages(
        self,
        load: _PageLoaderProtocol,
        visible: jax.Array,
        held: _RowIndex,
    ) -> None:
        """Compute and fold page scores into running softmax state row ``held``."""
        self._fold_scores(self._scores(load, visible, held), load, held)

    def _scores(
        self,
        load: Callable[[pl.Slice | slice], jax.Array],
        visible: jax.Array,
        held: _RowIndex,
        *,
        tail_tokens: int = 0,
    ) -> jax.Array:
        """Compute state row ``held``'s masked QK scores over one step's pages."""
        # Heads-major (flash-attention layout): scores [head_rows head slots,
        # span * page_tokens tokens] = Q C^T, so the probabilities feed P V as is.
        latent = load(pl.ds(0, LANES))
        if tail_tokens:
            latent = latent[:tail_tokens]
        score = jax.lax.dot_general(
            self.ws.queries[held],
            latent,
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        if self.fp8:
            score = score * self.ws.query_scale[held]
        if tail_tokens:
            score = jnp.pad(
                score,
                ((0, 0), (0, self.span * self.page_tokens - tail_tokens)),
                constant_values=-jnp.inf,
            )
        return jnp.where(visible, score, -jnp.inf)

    def _fold_scores(
        self,
        score: jax.Array,
        load: _PageLoaderProtocol,
        held: _RowIndex,
        *,
        tail_tokens: int = 0,
    ) -> None:
        """Fold ``score`` into state row ``held``, loading values after softmax.

        The values are loaded after the softmax (a load, not a live 320-vreg
        value across it). A staged step folds its parts in order.
        """
        if self.folds == 1:
            self._fold_part(
                score,
                lambda: load(pl.ds(0, LATENT)),
                held,
                tail_tokens,
            )
            return
        for first in range(0, self.span * self.page_tokens, self.fold_tokens):
            tokens = pl.ds(first, self.fold_tokens)
            self._fold_part(
                score[:, first : first + self.fold_tokens],
                lambda tokens=tokens: load(pl.ds(0, LATENT), tokens),
                held,
            )

    def _fold_part(
        self,
        score: jax.Array,
        values: Callable[[], jax.Array],
        held: _RowIndex,
        tail_tokens: int = 0,
    ) -> None:
        """Fold one score slice and lazy ``values()`` tile into state row ``held``."""
        ws = self.ws
        previous = ws.maximum[held]  # [head_rows, 1]
        new_maximum = jnp.maximum(previous, jnp.max(score, axis=1, keepdims=True))
        scale = jnp.exp(previous - new_maximum)
        probabilities = jnp.exp(score - new_maximum)  # [head_rows, span * page_tokens]
        ws.denominator[held] = ws.denominator[held] * scale + jnp.sum(
            probabilities,
            axis=1,
            keepdims=True,
        )
        ws.maximum[held] = new_maximum
        # Keep FP32 probabilities for the denominator; use scaled E4M3 for PV.
        high = (
            (probabilities * _FP8_PROB_SCALE).astype(jnp.float8_e4m3fn)
            if self.fp8
            else probabilities.astype(jnp.bfloat16)
        )
        loaded_values = values()
        if tail_tokens:
            loaded_values = loaded_values[:tail_tokens]
            high = high[:, :tail_tokens]
        weighted = jax.lax.dot_general(
            high,
            loaded_values,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        if self.fp8:
            weighted = weighted * (1.0 / _FP8_PROB_SCALE)
        ws.accumulator[held] = ws.accumulator[held] * scale + weighted

    def _clean_step(
        self,
        slot: _RowIndex,
        row: _RowIndex,
        local_step: _RowIndex,
        tokens: int | None = None,
        end: _RowIndex | None = None,
    ) -> PallasRefProtocol:
        """Zero stale padding lanes and out-of-bounds tokens in ``slot`` in place.

        Treats a step's pages as 32-bit rows ``[span * page_rows, 640]`` (each
        word packs ``packing`` tokens), zeroing lanes 576:640 of every token and
        every token at or past the row's limit (a step's visible offsets form a
        prefix). Stores directly into the buffer rather than materializing a
        masked copy of the whole tile. ``tokens`` (static) and ``end`` (traced,
        at most ``tokens``) bound the zeroing to the tokens a tail reads.
        """
        ws = self.ws
        rows = tokens or self.span * self.page_tokens
        words = (
            ws.pages.bitcast(jnp.uint32).at[slot]
            if self.native
            else ws.pages.bitcast(jnp.uint32)
            .at[slot]
            .reshape(self.span * self.page_rows, LANES)
        )
        if (
            self.batch in _PAD_STORE_BATCHES
            or self.heads != HEADS_PER_RANK
            or (self.batch, self.sequence_rows) == (MAX_ROWS, 1)
        ):
            pad_shape = (rows // self.packing, LANES - LATENT)
            pad_lane = jax.lax.broadcasted_iota(jnp.int32, pad_shape, 1)
            pad_ref = (
                words.at[:, pl.ds(LATENT, LANES - LATENT)]
                if tokens is None
                else words.at[
                    pl.ds(0, rows // self.packing),
                    pl.ds(LATENT, LANES - LATENT),
                ]
            )
            tpu.store(
                cast("TransformedRef", pad_ref),
                jnp.zeros(pad_shape, jnp.uint32),
                mask=pad_lane >= _ROPE_LANES,
            )
        else:
            # A word's two tokens share its lane.
            pad = words[:, pl.ds(LATENT, LANES - LATENT)]
            pad_lane = jax.lax.broadcasted_iota(jnp.int32, pad.shape, 1)
            words[:, pl.ds(LATENT, LANES - LATENT)] = jnp.where(
                pad_lane < _ROPE_LANES,
                pad,
                jnp.zeros((), pad.dtype),
            )
        limit = self._limit(row)
        cut: _RowIndex = 0  # visible offsets
        for k in range(rows // self.page_tokens):
            first_token = (
                (local_step * self.span + k) * self.group + self.owner
            ) * self.page_tokens
            cut = cut + jnp.clip(limit - first_token, 0, self.page_tokens)
        tile_tokens = SUBLANE_ROWS * self.packing
        tile = cut // tile_tokens  # one native eight-word-row tile

        @pl.when(tile < rows // tile_tokens)
        def boundary() -> None:  # the tile holding the cut
            """Mask stale tokens inside the partially visible boundary tile."""
            at = pl.ds(
                pl.multiple_of(
                    cast("jax.Array", tile * SUBLANE_ROWS),
                    SUBLANE_ROWS,
                ),
                SUBLANE_ROWS,
            )
            block = tpu.bitcast(words[at, :], self.cache.dtype)
            offset = tile * tile_tokens + jax.lax.broadcasted_iota(
                jnp.int32,
                block.shape,
                0,
            )
            words[at, :] = tpu.bitcast(
                jnp.where(offset < cut, block, jnp.zeros((), block.dtype)),
                jnp.uint32,
            )

        @pl.loop(tile + 1, (rows if end is None else end) // tile_tokens)
        def invisible(t: jax.Array) -> None:
            """Zero completely invisible trailing tiles."""
            words[
                pl.ds(pl.multiple_of(t * SUBLANE_ROWS, SUBLANE_ROWS), SUBLANE_ROWS),
                :,
            ] = jnp.zeros((SUBLANE_ROWS, LANES), jnp.uint32)

        return words

    def _pack_partials(self, first: int, end: int) -> None:
        """Normalize state rows ``first..end`` and write BF16 peer send packets."""
        ws, group = self.ws, self.group
        lses: dict[int, tuple[jax.Array, jax.Array]] = {}
        for held in range(first, end):
            total_weight = ws.denominator[held]  # [head_rows, 1]
            row_maximum = ws.maximum[held]
            has = total_weight > 0
            normalized = jnp.where(
                has,
                ws.accumulator[held] / jnp.where(has, total_weight, 1.0),
                0.0,
            )
            lse = jnp.where(
                has,
                row_maximum + jnp.log(jnp.where(has, total_weight, 1.0)),
                NEG,
            )
            target = pl.ds(self.slots * held, self.slots)
            lse_high = lse.astype(jnp.bfloat16).astype(jnp.float32)
            lane = jax.lax.broadcasted_iota(
                jnp.int32,
                (lse.shape[0], LANES - LATENT),
                1,
            )
            lse_lanes = jnp.where(
                lane == 0,
                lse_high,
                jnp.where(lane == 1, lse - lse_high, 0.0),
            )
            lses[held] = (lse_high, lse - lse_high)
            for peer in range(group):
                heads_of_peer = slice(self.slots * peer, self.slots * (peer + 1))
                ws.send[self._send_index(peer), target, :] = jnp.concatenate(
                    (normalized[heads_of_peer], lse_lanes[heads_of_peer]),
                    axis=1,
                ).astype(jnp.bfloat16)
        if self.lse_in_padding:
            self._pack_lse_rows(lses, first, end)

    def _pack_lse_rows(
        self,
        lses: dict[int, tuple[jax.Array, jax.Array]],
        first: int,
        end: int,
    ) -> None:
        """Pack each sequence's high/low LSE parts into its trailing padding row.

        Places packet row ``r = slots * j + h`` of the sequence at lane ``r``,
        high parts in the first packet row and low parts in the second (BF16,
        matching lanes 512:514). Each peer's ``[1, 128]`` lane vector is an
        exact 0/1 product.
        """
        ws, group, slots, rows = self.ws, self.group, self.slots, self.sequence_rows
        if (
            slots < _MIN_LSE_PADDING_SLOTS
            or slots * (rows - 1) > VREG_LANE_COUNT
            or (end - first) % rows
        ):
            msg = (
                "lse_in_padding takes two or more head slots, one lane each, "
                "and whole sequences"
            )
            raise ValueError(msg)
        tile_rows = lses[first][0].shape[0]
        slot_row = jax.lax.broadcasted_iota(
            jnp.int32,
            (tile_rows, VREG_LANE_COUNT),
            0,
        )
        lane = jax.lax.broadcasted_iota(
            jnp.int32,
            (tile_rows, VREG_LANE_COUNT),
            1,
        )
        index = jax.lax.broadcasted_iota(jnp.int32, (group, tile_rows), 0)  # send slot
        if self._lane_major:
            hosts = group // HOST_LANES
            index = (index % hosts) * HOST_LANES + index // hosts  # the slot's peer
        owner = (
            jax.lax.broadcasted_iota(jnp.int32, (group, tile_rows), 1) // slots == index
        ).astype(jnp.bfloat16)
        for start in range(first, end, rows):
            for part in range(_LSE_PARTS):  # high, low
                spread = jnp.zeros((tile_rows, VREG_LANE_COUNT), jnp.float32)
                for j in range(rows - 1):
                    value = lses[start + j][part]  # [head_rows, 1]
                    spread = jnp.where(
                        lane == slots * j + slot_row % slots,
                        value,
                        spread,
                    )
                vectors = jax.lax.dot_general(
                    owner,
                    spread.astype(jnp.bfloat16),
                    (((1,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )  # [group, 128]
                row = slots * (start + rows - 1) + part
                ws.send[:, pl.ds(row, 1), pl.ds(0, VREG_LANE_COUNT)] = vectors[
                    :, None, :
                ].astype(jnp.bfloat16)

    def _combine_lanes(self) -> pl.Slice:
        """Return lane slice transferred during partial combine."""
        return pl.ds(0, LATENT if self.lse_in_padding else LANES)

    def _send_index(self, peer: int) -> int:
        """Return ``ws.send`` slot index corresponding to ``peer``."""
        if not self._lane_major:
            return peer
        hosts = self.group // HOST_LANES
        return (peer % HOST_LANES) * hosts + peer // HOST_LANES

    def _scope(self, name: str) -> contextlib.AbstractContextManager[object]:
        """Return optional named profiling scope when ``self.trace`` is active."""
        return (
            jax.named_scope(self.trace + name)
            if self.trace
            else contextlib.nullcontext()
        )

    def prime(self) -> None:
        """Issue the initial pipeline-depth page DMAs as early as positions allow."""
        for step in range(self.buffers - 1):

            @pl.when(step < self.total)
            def prime(step: int = step) -> None:
                """Start initial page step ``step``."""
                self._start_step(step)

    def row_groups(self) -> list[tuple[int, int]]:
        """Return ``[(first row, end row)]`` partitioning ``batch`` into groups."""
        groups = self.groups
        bounds = [self.batch * g // groups for g in range(groups + 1)]
        return [(bounds[g], bounds[g + 1]) for g in range(groups)]

    def _unchunked_group_rows(self, group_index: int) -> pl.Slice:
        """Return packet-row slice corresponding to unchunked ``group_index``."""
        first, end = self.row_groups()[group_index]
        return pl.ds(self.slots * first, self.slots * (end - first))

    def _gather_unchunked_queries(
        self,
        queries: jax.Array,
        new_latents: jax.Array,
        parity: _RowIndex,
        between: Callable[[], None] | None,
    ) -> None:
        """Stage own queries/latents, launch query all-gather, and run ``between``."""
        ws, group, member, base, batch = (
            self.ws,
            self.group,
            self.member,
            self.base,
            self.batch,
        )
        groups = self.row_groups()
        ws.own_rows[...] = jnp.zeros(ws.own_rows.shape, jnp.float32)
        ws.new_rows[...] = jnp.zeros(ws.new_rows.shape, jnp.float32)
        for b in range(batch):
            ws.own_rows[pl.ds(self.slots * b, self.heads), :] = queries[b]
            ws.new_rows[pl.ds(self.slots * b, self.heads), :] = jnp.broadcast_to(
                new_latents[b][None, :],
                (self.heads, LANES),
            )
        # 1. q all-gather into slot `member` of every peer, group by group.
        ws.gathered[parity, member] = (
            _pack_fp8_query(ws.own_rows[...])
            if ws.gathered.dtype == jnp.uint8
            else ws.own_rows[...].astype(jnp.bfloat16)
        )
        for g in range(len(groups)):
            row_slice = self._unchunked_group_rows(g)
            for peer in range(0 if "gather" in self.skip else group):

                @pl.when(peer != member)
                def send_query(
                    peer: int = peer,
                    g: int = g,
                    row_slice: pl.Slice = row_slice,
                ) -> None:
                    """Send absorbed query slice to group ``peer``."""
                    tpu.make_async_remote_copy(
                        ws.gathered.at[parity, member, row_slice],
                        ws.gathered.at[parity, member, row_slice],
                        ws.gather_semaphores.at[0, parity, g],
                        ws.gather_semaphores.at[1, parity, g],
                        device_id=(base + peer,),
                        device_id_type=pl.DeviceIdType.MESH,
                    ).start()

        if between is not None:
            between()
        ws.maximum[...] = jnp.full(ws.maximum.shape, -jnp.inf, jnp.float32)
        ws.denominator[...] = jnp.zeros(ws.denominator.shape, jnp.float32)
        ws.accumulator[...] = jnp.zeros(ws.accumulator.shape, jnp.float32)

    def _run_unchunked_group(
        self,
        group_index: int,
        first: int,
        end: int,
        parity: _RowIndex,
    ) -> None:
        """Await gathered queries, attend local pages, and send partial outputs."""
        ws, group, member, base = self.ws, self.group, self.member, self.base
        heads = group * self.slots
        row_slice = pl.ds(self.slots * first, self.slots * (end - first))
        others = ws.gathered.at[parity, pl.ds(0, group - 1), row_slice]
        if "gather" not in self.skip:
            with self._scope("gather_wait"):
                tpu.make_async_copy(
                    others,
                    others,
                    ws.gather_semaphores.at[0, parity, group_index],
                ).wait()
                tpu.make_async_copy(
                    others,
                    others,
                    ws.gather_semaphores.at[1, parity, group_index],
                ).wait()
        with self._scope("local"):
            for b in range(first, end):
                # [G, slots, 640] -> [slots G, 640] (slot `slots` d + h is member
                # d's head h), padded to the tile's rows.
                gathered = ws.gathered[
                    parity,
                    :,
                    pl.ds(self.slots * b, self.slots),
                    :,
                ].reshape(heads, LANES)
                padding = ((0, self.tile - heads), (0, 0))
                if ws.gathered.dtype == jnp.uint8:
                    query, scale = _unpack_fp8_query(gathered)
                    ws.queries[b] = jnp.pad(query, padding)
                    ws.query_scale[b] = jnp.pad(scale, padding, constant_values=1)
                else:
                    self._store_query(b, jnp.pad(gathered.astype(jnp.float32), padding))

            # 2. Local attention over this rank's pages (the group's steps).
            # Attention DP's group rows pipeline the fold as chunked calls do.
            step_end = self.starts[first] if "attend" in self.skip else self.starts[end]
            if self.heads != HEADS_PER_RANK:
                self._attend_pipeline(self.starts[first], step_end, 0)
            else:

                @pl.loop(self.starts[first], step_end)
                def attend(step: jax.Array) -> None:
                    """Attend local page step ``step``."""
                    self._attend_step(step)

            # 3. Normalize and send each owner its heads' rows.
            self._pack_partials(first, end)
            for peer in range(0 if "combine" in self.skip else group):

                @pl.when(peer != member)
                def send_partial(
                    peer: int = peer,
                    g: int = group_index,
                ) -> None:
                    """Send normalized partial packet to owner ``peer``."""
                    tpu.make_async_remote_copy(
                        ws.send.at[peer, row_slice, self._combine_lanes()],
                        ws.receive.at[parity, member, row_slice, self._combine_lanes()],
                        ws.combine_semaphores.at[0, parity, g],
                        ws.combine_semaphores.at[1, parity, g],
                        device_id=(base + peer,),
                        device_id_type=pl.DeviceIdType.MESH,
                    ).start()

            ws.receive[parity, member, row_slice] = ws.send[member, row_slice]

    def _merge_unchunked_causal_sequences(
        self,
        partials: jax.Array,
        lse: jax.Array,
        new_latents: jax.Array,
    ) -> jax.Array:
        """Merge received partials with causal in-step multi-row ``new_latents``."""
        # Row b's head slots also attend the new tokens of rows j <= b of
        # its sequence (every rank holds all rows' new latents).
        scores = jax.lax.dot_general(
            self.ws.own_rows[...],
            new_latents,
            (((1,), (1,)), ((), ())),
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )  # [rows, B]
        slot_row = jax.lax.broadcasted_iota(jnp.int32, scores.shape, 0) // self.slots
        key_row = jax.lax.broadcasted_iota(jnp.int32, scores.shape, 1)
        causal = (key_row // self.sequence_rows == slot_row // self.sequence_rows) & (
            key_row <= slot_row
        )
        scores = jnp.where(causal, scores, -jnp.inf)
        top = jnp.maximum(jnp.max(lse, axis=0), jnp.max(scores, axis=1, keepdims=True))
        weights = jnp.exp(lse - top[None])
        new_weights = jnp.exp(scores - top)  # [rows, B]
        new_values = jax.lax.dot_general(
            new_weights,
            new_latents[:, :LATENT],
            (((1,), (0,)), ((), ())),
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        return (jnp.sum(weights * partials[:, :, :LATENT], axis=0) + new_values) / (
            jnp.sum(weights, axis=0) + jnp.sum(new_weights, axis=1, keepdims=True)
        )

    def _merge_unchunked_partials(
        self,
        new_latents: jax.Array,
        parity: _RowIndex,
        keys: _TensorOrRef | None,
        key_valid: _TensorOrRef | None,
    ) -> jax.Array:
        """Merge peer partials and new tokens across all unchunked rows."""
        ws, group, batch = self.ws, self.group, self.batch
        with self._scope("merge"):
            # 4. Log-sum-exp merge of the G partials and the new token.
            partials = ws.receive[parity].astype(jnp.float32)  # [G, rows, 640]
            lse = (
                partials[:, :, LATENT : LATENT + 1]
                + partials[:, :, LATENT + 1 : LATENT + 2]
            )
            if keys is not None:
                if key_valid is None:
                    msg = (
                        "key_valid is required when explicit keys are provided "
                        f"(batch={batch}, sequence_rows={self.sequence_rows})"
                    )
                    raise ValueError(msg)
                sequences = batch // self.sequence_rows
                rows = self.slots * self.sequence_rows
                merged = jnp.concatenate(
                    [
                        _merge_received_keys(
                            ws.receive.at[parity, :, pl.ds(rows * s, rows)],
                            ws.own_rows[pl.ds(rows * s, rows), :],
                            keys[s],
                            key_valid[s],
                            group,
                            lse_rows=self.slots if self.lse_in_padding else 0,
                        )
                        for s in range(sequences)
                    ],
                    axis=0,
                )
            elif self.sequence_rows == 1 or self.cache_verify_rows:
                new_score = jnp.sum(
                    ws.own_rows[...] * ws.new_rows[...],
                    axis=1,
                    keepdims=True,
                )  # [rows, 1]
                top = jnp.maximum(jnp.max(lse, axis=0), new_score)
                weights = jnp.exp(lse - top[None])
                new_weight = jnp.exp(new_score - top)
                merged = (
                    jnp.sum(weights * partials[:, :, :LATENT], axis=0)
                    + new_weight * ws.new_rows[:, :LATENT]
                ) / (jnp.sum(weights, axis=0) + new_weight)
            else:
                merged = self._merge_unchunked_causal_sequences(
                    partials,
                    lse,
                    new_latents,
                )
            return jnp.stack(
                [
                    merged[self.slots * b : self.slots * b + self.heads]
                    for b in range(batch)
                ]
            )

    def attend(
        self,
        queries: jax.Array,
        new_latents: jax.Array,
        parity: _RowIndex,
        between: Callable[[], None] | None = None,
        *,
        keys: _TensorOrRef | None = None,
        key_valid: _TensorOrRef | None = None,
    ) -> jax.Array:
        """Run DCP gather, local attention, combine, and merge for ``queries``.

        ``queries`` ``[B, heads, 640]`` FP32 are absorbed (scale folded in),
        ``new_latents`` is ``[B, 640]`` FP32, and ``parity`` alternates per
        invocation (MLA layer). ``between`` runs while the query gather is in
        flight. Returns the merged latent-space output ``[B, heads, 512]`` FP32.

        ``keys`` / ``key_valid`` (``sequence_rows > 1`` only) replace the
        sequences' causal new tokens: sequence ``s``'s rows all attend its
        ``K`` new keys ``keys[s]`` (``[sequences, K, 640]`` FP32 values, every
        rank holding all of them) where ``key_valid[s]`` (``[sequences, 1, K]``
        int32) is nonzero (``new_latents`` is then unused). A row sees at least
        one valid key.

        The rows run in groups (``row_groups``): every group's queries go out
        first; then per group, in order, its gather is awaited, its pages
        attended, and its partials sent, overlapping adjacent groups' transfers,
        followed by the final log-sum-exp merge.
        """
        if keys is not None and self.sequence_rows == 1:
            msg = (
                "explicit new keys need multi-row sequences, "
                f"got sequence_rows={self.sequence_rows}"
            )
            raise ValueError(msg)
        if chunked(self.batch, force=self.force_chunked):
            return self._attend_chunked(
                queries,
                new_latents,
                between,
                keys=keys,
                key_valid=key_valid,
            )
        self._gather_unchunked_queries(queries, new_latents, parity, between)
        groups = self.row_groups()
        for g, (first, end) in enumerate(groups):
            self._run_unchunked_group(g, first, end, parity)
        with self._scope("combine_wait"):
            for g in range(len(groups)):
                others = self.ws.receive.at[
                    parity,
                    pl.ds(0, self.group - 1),
                    self._unchunked_group_rows(g),
                    self._combine_lanes(),
                ]
                if "combine" not in self.skip:
                    tpu.make_async_copy(
                        others,
                        others,
                        self.ws.combine_semaphores.at[0, parity, g],
                    ).wait()
                    tpu.make_async_copy(
                        others,
                        others,
                        self.ws.combine_semaphores.at[1, parity, g],
                    ).wait()
        return self._merge_unchunked_partials(new_latents, parity, keys, key_valid)

    def _build_chunked_plan(
        self,
        keys: _TensorOrRef | None,
        key_valid: _TensorOrRef | None,
    ) -> _ChunkedAttendPlan:
        """Validate chunked options and assemble ``_ChunkedAttendPlan``."""
        chunk = chunk_layout(
            self.batch,
            self.heads,
            chunk_rows=self.chunk_rows,
        )[0]
        if keys is not None and chunk != self.sequence_rows:
            msg = (
                "explicit new keys need one sequence per chunk, "
                f"got chunk={chunk}, sequence_rows={self.sequence_rows}"
            )
            raise ValueError(msg)
        ring_slots = self.ws.gathered.shape[0]
        count = self.batch // chunk
        hier = self.host_combine is not None
        # The two-stage combine sends a chunk's merged partials a chunk later:
        # queries one chunk ahead leave its final merge three chunks behind.
        lookahead = 2 if ring_slots == _RING_SLOTS_QUAD and not hier else 1
        self._lane_major = hier
        host_combine_buffers: _HostCombineBuffers | None = None
        if self.host_combine is not None:
            if (
                self.group != _HOSTS_PER_DCP32 * HOST_LANES
                or not self.lse_in_padding
                or keys is None
                or "combine" in self.skip
            ):
                msg = (
                    "host_combine requires DCP32 explicit-keys sequences with "
                    f"lse_in_padding, got group={self.group}, "
                    f"lse_in_padding={self.lse_in_padding}, "
                    f"has_keys={keys is not None}, skip={sorted(self.skip)}"
                )
                raise ValueError(msg)
            stage2_recv, stage2_send, stage2_sems = self.host_combine
            host_combine_buffers = _HostCombineBuffers(
                stage2_recv=stage2_recv,
                stage2_send=stage2_send,
                stage2_sems=stage2_sems,
                hosts=self.group // HOST_LANES,
                host=self.member // HOST_LANES,
                lane=self.member % HOST_LANES,
            )
        peers_bytes = (
            (HOST_LANES - 1) * (self.group // HOST_LANES) if hier else self.group - 1
        )
        return _ChunkedAttendPlan(
            chunk=chunk,
            count=count,
            rows=self.slots * chunk,
            heads=self.group * self.slots,
            ring_slots=ring_slots,
            lookahead=lookahead,
            layer=self.cache.layer if self.cache.layer is not None else 0,
            peers_bytes=peers_bytes,
            steps=(),
            keys=keys,
            key_valid=key_valid,
            host_combine=host_combine_buffers,
        )

    def _stage_chunked_rows(
        self,
        plan: _ChunkedAttendPlan,
        queries: jax.Array,
        new_latents: jax.Array,
    ) -> None:
        """Stage per-chunk owned queries and new-token latents into workspace."""
        ws = self.ws
        ws.own_rows[...] = jnp.zeros(ws.own_rows.shape, jnp.float32)
        ws.new_rows[...] = jnp.zeros(ws.new_rows.shape, jnp.float32)
        for b in range(self.batch):
            c, j = divmod(b, plan.chunk)
            ws.own_rows[c, pl.ds(self.slots * j, self.heads), :] = queries[b]
            ws.new_rows[c, pl.ds(self.slots * j, self.heads), :] = jnp.broadcast_to(
                new_latents[b][None, :],
                (self.heads, LANES),
            )

    def _send_chunk_queries(
        self,
        plan: _ChunkedAttendPlan,
        chunk_index: _RowIndex,
    ) -> None:
        """Pack and remotely broadcast queries for ``chunk_index``."""
        ws, group, member, base = self.ws, self.group, self.member, self.base
        slot = plan.ring(chunk_index)
        own = ws.own_rows[chunk_index, : plan.rows, :]
        ws.gathered[slot, member] = (
            _pack_fp8_query(own)
            if ws.gathered.dtype == jnp.uint8
            else own.astype(jnp.bfloat16)
        )
        for peer in range(0 if "gather" in self.skip else group):

            @pl.when(peer != member)
            def send(peer: int = peer) -> None:
                """Send chunk query packet to ``peer``."""
                tpu.make_async_remote_copy(
                    ws.gathered.at[slot, member],
                    ws.gathered.at[slot, member],
                    ws.gather_semaphores.at[0, slot],
                    ws.gather_semaphores.at[1, slot],
                    device_id=(base + peer,),
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()

    def _wait_chunk_gather(
        self,
        plan: _ChunkedAttendPlan,
        chunk_index: _RowIndex,
    ) -> None:
        """Wait for ``chunk_index`` query gather transfers to complete."""
        if "gather" in self.skip:
            return
        slot = plan.ring(chunk_index)
        others = self.ws.gathered.at[slot, pl.ds(0, self.group - 1)]
        tpu.make_async_copy(
            others,
            others,
            self.ws.gather_semaphores.at[0, slot],
        ).wait()
        tpu.make_async_copy(
            others,
            others,
            self.ws.gather_semaphores.at[1, slot],
        ).wait()

    def _send_chunk_partials(
        self,
        plan: _ChunkedAttendPlan,
        chunk_index: _RowIndex,
    ) -> None:
        """Send normalized partial outputs of ``chunk_index`` to peers or host-mates."""
        ws, group, member, base = self.ws, self.group, self.member, self.base
        slot = plan.ring(chunk_index)
        host_combine = plan.host_combine
        if host_combine is not None:
            lanes512 = self._combine_lanes()
            # Stage one: each host-mate gets the partials of its lane's owners.
            for k in range(1, HOST_LANES):
                mate = (host_combine.lane + k) % HOST_LANES
                tpu.make_async_remote_copy(
                    ws.send.at[
                        pl.ds(mate * host_combine.hosts, host_combine.hosts),
                        :,
                        lanes512,
                    ],
                    ws.receive.at[
                        slot,
                        pl.ds(
                            host_combine.lane * host_combine.hosts, host_combine.hosts
                        ),
                        :,
                        lanes512,
                    ],
                    ws.combine_semaphores.at[0, slot],
                    ws.combine_semaphores.at[1, slot],
                    device_id=(base + host_combine.host * HOST_LANES + mate,),
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()
            ws.receive[
                slot, pl.ds(host_combine.lane * host_combine.hosts, host_combine.hosts)
            ] = ws.send[
                pl.ds(host_combine.lane * host_combine.hosts, host_combine.hosts)
            ]
            return
        for peer in range(0 if "combine" in self.skip else group):

            @pl.when(peer != member)
            def send(peer: int = peer) -> None:
                """Send partial packet of ``chunk_index`` to ``peer``."""
                tpu.make_async_remote_copy(
                    ws.send.at[peer, :, self._combine_lanes()],
                    ws.receive.at[slot, member, :, self._combine_lanes()],
                    ws.combine_semaphores.at[0, slot],
                    ws.combine_semaphores.at[1, slot],
                    device_id=(base + peer,),
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()

        ws.receive[slot, member] = ws.send[member]

    def _wait_chunk_sent(
        self,
        plan: _ChunkedAttendPlan,
        chunk_index: _RowIndex,
    ) -> None:  # this rank's partials of chunk_index have left the send buffer
        """Wait until ``chunk_index`` partials have drained from ``ws.send``."""
        if "combine" in self.skip:
            return
        others = self.ws.receive.at[
            plan.ring(chunk_index),
            pl.ds(0, plan.peers_bytes),
            :,
            self._combine_lanes(),
        ]
        tpu.make_async_copy(
            others,
            others,
            self.ws.combine_semaphores.at[0, plan.ring(chunk_index)],
        ).wait()

    def _wait_chunk_received(
        self,
        plan: _ChunkedAttendPlan,
        chunk_index: _RowIndex,
    ) -> None:
        """Wait until peer partials for ``chunk_index`` have arrived."""
        if "combine" in self.skip:
            return
        others = self.ws.receive.at[
            plan.ring(chunk_index),
            pl.ds(0, plan.peers_bytes),
            :,
            self._combine_lanes(),
        ]
        tpu.make_async_copy(
            others,
            others,
            self.ws.combine_semaphores.at[1, plan.ring(chunk_index)],
        ).wait()

    def _host_merge_chunk(
        self,
        plan: _ChunkedAttendPlan,
        host_combine: _HostCombineBuffers,
        chunk_index: _RowIndex,
    ) -> None:
        """Merge intra-host partials per owner lane and forward across hosts."""
        ws, group, base = self.ws, self.group, self.base
        slot = plan.ring(chunk_index)
        lanes512 = self._combine_lanes()
        # Every block's lse at once (packet row r at lane r of its padding rows).
        pad = plan.rows - self.slots
        tails = ws.receive[
            slot,
            :,
            pl.ds(pad, _LSE_PARTS),
            pl.ds(0, VREG_LANE_COUNT),
        ].astype(jnp.float32)  # [32, 2, 128]
        vectors = tails[:, 0:1, :] + tails[:, 1:2, :]  # [32, 1, 128]
        packet_row = jax.lax.broadcasted_iota(
            jnp.int32,
            (group, plan.rows, VREG_LANE_COUNT),
            1,
        )
        packet_lane = jax.lax.broadcasted_iota(
            jnp.int32,
            (group, plan.rows, VREG_LANE_COUNT),
            2,
        )
        lses = jnp.sum(
            jnp.where(packet_lane == packet_row, vectors, 0.0),
            axis=2,
            keepdims=True,
        )  # [32, rows, 1]
        lses = jnp.where(packet_row[:, :, :1] < pad, lses, NEG)
        for owner_host in range(host_combine.hosts):
            sources = [
                source * host_combine.hosts + owner_host for source in range(HOST_LANES)
            ]
            top = lses[sources[0]]
            for index in sources[1:]:
                top = jnp.maximum(top, lses[index])
            total = jnp.zeros_like(top)
            accumulator: jax.Array | None = None
            for index in sources:
                weight = jnp.exp(lses[index] - top)
                total = total + weight
                term = weight * ws.receive[slot, index, :, pl.ds(0, LATENT)].astype(
                    jnp.float32
                )
                accumulator = term if accumulator is None else accumulator + term
            if accumulator is None:
                msg = (
                    "host_merge requires at least one host-lane source, "
                    f"got host_lanes={HOST_LANES}"
                )
                raise ValueError(msg)
            host_combine.stage2_send[owner_host] = _lse_packet(
                accumulator / total,
                top + jnp.log(total),
                self.slots,
            )
        for k in range(1, host_combine.hosts):
            owner_host = (host_combine.host + k) % host_combine.hosts
            tpu.make_async_remote_copy(
                host_combine.stage2_send.at[owner_host, :, lanes512],
                host_combine.stage2_recv.at[slot, host_combine.host, :, lanes512],
                host_combine.stage2_sems.at[0, slot],
                host_combine.stage2_sems.at[1, slot],
                device_id=(base + owner_host * HOST_LANES + host_combine.lane,),
                device_id_type=pl.DeviceIdType.MESH,
            ).start()
        host_combine.stage2_recv[slot, host_combine.host] = host_combine.stage2_send[
            host_combine.host
        ]

    def _wait_stage2_chunk(
        self,
        plan: _ChunkedAttendPlan,
        host_combine: _HostCombineBuffers,
        chunk_index: _RowIndex,
        which: int,
    ) -> None:
        """Wait on stage-two semaphore ``which`` (0: send drained, 1: recv arrived)."""
        others = host_combine.stage2_recv.at[
            plan.ring(chunk_index),
            pl.ds(0, host_combine.hosts - 1),
            :,
            self._combine_lanes(),
        ]
        tpu.make_async_copy(
            others, others, host_combine.stage2_sems.at[which, plan.ring(chunk_index)]
        ).wait()

    def _chunk_merge_unroll(self) -> int:
        """Return peer-loop unroll factor for single-row or cached-verify merges."""
        dp2_wide = self.heads == _DP2_HEADS and (
            self.batch == _GROUP_ROWS_64 or (self.batch == _GROUP_ROWS_48 and self.fp8)
        )
        dp4_wide = self.heads == DP4_HEADS and (
            self.batch == _GROUP_ROWS_32 or (self.batch == _GROUP_ROWS_24 and self.fp8)
        )
        return _PEER_MERGE_UNROLL if dp2_wide or dp4_wide else 1

    def _merge_chunk(
        self,
        plan: _ChunkedAttendPlan,
        chunk_index: _RowIndex,
    ) -> None:
        """Merge received partials and new tokens/keys for ``chunk_index``."""
        if "merge" in self.skip:
            return
        ws = self.ws
        if self.sequence_rows == 1 or self.cache_verify_rows:
            # Attention DP2's and DP4's B128 and FP8 B96 merge four peers a step.
            ws.merged[chunk_index, : plan.rows, :] = _merge_received(
                ws.receive.at[plan.ring(chunk_index)],
                ws.own_rows[chunk_index, : plan.rows, :],
                ws.new_rows[chunk_index, : plan.rows, :],
                self.group,
                unroll=self._chunk_merge_unroll(),
            )
            return
        if plan.keys is not None:
            if plan.key_valid is None:
                msg = (
                    "key_valid is required when explicit keys are provided "
                    f"(chunk={plan.chunk}, sequence_rows={self.sequence_rows})"
                )
                raise ValueError(msg)
            # One sequence per chunk: its rows attend its explicit new keys.
            host_combine = plan.host_combine
            received, sources = (
                (
                    host_combine.stage2_recv.at[plan.ring(chunk_index)],
                    host_combine.hosts,
                )
                if host_combine is not None
                else (ws.receive.at[plan.ring(chunk_index)], self.group)
            )
            ws.merged[chunk_index, : plan.rows, :] = _merge_received_keys(
                received,
                ws.own_rows[chunk_index, : plan.rows, :],
                plan.keys[chunk_index],
                plan.key_valid[chunk_index],
                sources,
                lse_rows=self.slots if self.lse_in_padding else 0,
            )
            return
        # Whole sequences per chunk: fold each row's earlier in-step tokens too.
        latents = jnp.concatenate(
            [
                ws.new_rows[chunk_index, pl.ds(self.slots * j, 1), :]
                for j in range(plan.chunk)
            ],
            axis=0,
        )  # [chunk, 640]
        seq_unroll = (
            _PEER_MERGE_UNROLL
            if self.fp8
            and _grouped_spec(self.batch, self.group, self.heads, self.sequence_rows)
            else 1
        )
        ws.merged[chunk_index, : plan.rows, :] = _merge_received_sequences(
            ws.receive.at[plan.ring(chunk_index)],
            ws.own_rows[chunk_index, : plan.rows, :],
            latents,
            self.group,
            self.slots,
            self.sequence_rows,
            unroll=seq_unroll,
        )

    def _attend_chunk_local(
        self,
        plan: _ChunkedAttendPlan,
        chunk_index: jax.Array,
    ) -> None:
        """Unpack gathered queries and run local page attention for ``chunk_index``."""
        ws = self.ws
        with self._scope("gather_wait"):
            self._wait_chunk_gather(plan, chunk_index)
        with self._scope("local"):
            for j in range(0 if "build" in self.skip else plan.chunk):
                # [G, slots, 640] -> [slots G, 640] (slot `slots` d + h is member
                # d's head h), padded to the tile's rows.
                gathered = ws.gathered[
                    plan.ring(chunk_index),
                    :,
                    pl.ds(self.slots * j, self.slots),
                    :,
                ].reshape(plan.heads, LANES)
                padding = ((0, self.tile - plan.heads), (0, 0))
                if ws.gathered.dtype == jnp.uint8:
                    query, scale = _unpack_fp8_query(gathered)
                    ws.queries[j] = jnp.pad(query, padding)
                    ws.query_scale[j] = jnp.pad(scale, padding, constant_values=1)
                else:
                    self._store_query(j, jnp.pad(gathered.astype(jnp.float32), padding))
            ws.maximum[...] = jnp.full(ws.maximum.shape, -jnp.inf, jnp.float32)
            ws.denominator[...] = jnp.zeros(ws.denominator.shape, jnp.float32)
            ws.accumulator[...] = jnp.zeros(ws.accumulator.shape, jnp.float32)
            first = plan.chunk * chunk_index
            self._attend_range(
                plan.first_step(first),
                plan.first_step(first)
                if "attend" in self.skip
                else plan.first_step(first + plan.chunk),
                first,
            )

    def _run_chunk_step(
        self,
        plan: _ChunkedAttendPlan,
        chunk_index: jax.Array,
    ) -> None:
        """Execute one pipelined chunk iteration inside ``chunk_step``."""
        # Merge before issuing the query that reuses this slot.
        done = chunk_index + plan.lookahead - plan.ring_slots
        host_combine = plan.host_combine

        @pl.when(done >= 0)
        def merge_done() -> None:
            """Finalize combine wait and merge for completed slot ``done``."""
            with self._scope("combine_wait"):
                if host_combine is not None:
                    self._wait_stage2_chunk(plan, host_combine, done, 1)
                else:
                    self._wait_chunk_received(plan, done)
            with self._scope("merge"):
                self._merge_chunk(plan, done)

        @pl.when(chunk_index + plan.lookahead < plan.count)
        def send_next() -> None:
            """Launch lookahead query gather for ``chunk_index + lookahead``."""
            self._send_chunk_queries(plan, chunk_index + plan.lookahead)

        self._attend_chunk_local(plan, chunk_index)

        @pl.when(chunk_index >= 1)
        def free_send_buffer() -> None:
            """Ensure prior chunk's outbound partials have left ``ws.send``."""
            with self._scope("send_wait"):
                self._wait_chunk_sent(plan, chunk_index - 1)

        with self._scope("pack_send"):
            # The chunk's rows: state rows 0..chunk.
            self._pack_partials(0, 0 if "pack" in self.skip else plan.chunk)
            self._send_chunk_partials(plan, chunk_index)

        if host_combine is not None:

            @pl.when(chunk_index >= 1)
            def host_stage() -> None:
                """Run intra-host merge and inter-host send for ``chunk_index - 1``."""
                with self._scope("host_wait"):
                    self._wait_chunk_received(plan, chunk_index - 1)

                    @pl.when(chunk_index >= _MIN_STAGE2_DRAIN_CHUNKS)
                    def free_stage2() -> None:
                        """Wait for ``chunk_index - 2`` stage-two sends to drain."""
                        self._wait_stage2_chunk(plan, host_combine, chunk_index - 2, 0)

                with self._scope("host_merge"):
                    self._host_merge_chunk(plan, host_combine, chunk_index - 1)

    def _drain_chunked_tail(self, plan: _ChunkedAttendPlan) -> jax.Array:
        """Drain trailing in-flight chunk combines/merges and stack batch outputs."""
        tail_start = max(plan.count - plan.ring_slots + plan.lookahead, 0)
        host_combine = plan.host_combine
        if host_combine is not None:
            with self._scope("host_wait"):
                self._wait_chunk_received(plan, plan.count - 1)
                if plan.count >= _MIN_STAGE2_DRAIN_CHUNKS:
                    self._wait_stage2_chunk(plan, host_combine, plan.count - 2, 0)
            with self._scope("host_merge"):
                self._host_merge_chunk(plan, host_combine, plan.count - 1)
            self._wait_chunk_sent(plan, plan.count - 1)
            for c in range(tail_start, plan.count):
                with self._scope("combine_wait"):
                    self._wait_stage2_chunk(plan, host_combine, c, 1)
                with self._scope("merge"):
                    self._merge_chunk(plan, c)
            self._wait_stage2_chunk(plan, host_combine, plan.count - 1, 0)
            self._lane_major = False
            return jnp.stack(
                [
                    self.ws.merged[
                        b // plan.chunk,
                        pl.ds(self.slots * (b % plan.chunk), self.heads),
                        :,
                    ]
                    for b in range(self.batch)
                ]
            )
        if self.batch != _GROUP_ROWS_32:
            self._wait_chunk_sent(plan, plan.count - 1)
        for c in range(tail_start, plan.count):
            with self._scope("combine_wait"):
                self._wait_chunk_received(plan, c)
            with self._scope("merge"):
                self._merge_chunk(plan, c)
        if self.batch == _GROUP_ROWS_32:
            # Merges read receive/own/new buffers, leaving the final send free
            # to drain in parallel. Finish it before the next layer can pack.
            self._wait_chunk_sent(plan, plan.count - 1)
        return jnp.stack(
            [
                self.ws.merged[
                    b // plan.chunk,
                    pl.ds(self.slots * (b % plan.chunk), self.heads),
                    :,
                ]
                for b in range(self.batch)
            ]
        )

    def _attend_chunked(
        self,
        queries: jax.Array,
        new_latents: jax.Array,
        between: Callable[[], None] | None,
        *,
        keys: _TensorOrRef | None = None,
        key_valid: _TensorOrRef | None = None,
    ) -> jax.Array:
        """Run pipelined multi-chunk DCP attention across ``chunk_layout`` slots.

        Queries run two chunks ahead with four ring slots, else one; earlier
        chunks' combines overlap the current chunk's page attention. Packets
        rotate through ``R`` slots across layers (chunk ``c`` of MLA layer ``L``:
        slot ``(L * C + c) % R``). A query for chunk ``q`` leaves only after
        chunk ``q - R`` has merged so every peer has consumed the slot's
        queries. The chunk loop is rolled so code size stays O(1) in the batch.
        """
        plan = self._build_chunked_plan(keys, key_valid)
        self._stage_chunked_rows(plan, queries, new_latents)
        # Page steps before each row (scalars), traced after row staging.
        plan = dataclasses.replace(
            plan,
            steps=tuple(
                (self.counts[b] + self.span - 1) // self.span for b in range(self.batch)
            ),
        )
        for initial in range(min(plan.lookahead, plan.count)):
            self._send_chunk_queries(plan, initial)
        if between is not None:
            between()

        @pl.loop(0, plan.count)
        def chunk_step(c: jax.Array) -> None:
            """Run chunk ``c`` inside the rolled chunk pipeline."""
            self._run_chunk_step(plan, c)

        return self._drain_chunked_tail(plan)


def _merge_received(
    received: PallasRefProtocol,
    own_rows: jax.Array,
    new_rows: jax.Array,
    group: int,
    *,
    unroll: int = 1,
) -> jax.Array:
    """Merge ``group`` peers' partials in ``received`` with the new token online.

    ``received`` ``[G, rows, 640]`` holds normalized outputs and high/low LSE in
    lanes 512:514; ``own_rows`` holds queries and ``new_rows`` holds the new
    latent per head slot, returning ``[rows, 512]``. Loops online over peers
    (``unroll`` peers per step) to bound live register footprint.
    """
    # [rows, 1]: the new token's score.
    top = jnp.sum(own_rows * new_rows, axis=1, keepdims=True)
    accumulator = new_rows[:, :LATENT]
    denominator = jnp.ones_like(top)

    def fold(
        peer: jax.Array,
        carry: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Fold ``peer``'s partial output into the running log-sum-exp state."""
        cur_top, cur_acc, cur_den = carry
        partial = received[peer].astype(jnp.float32)  # [rows, 640]
        lse = partial[:, LATENT : LATENT + 1] + partial[:, LATENT + 1 : LATENT + 2]
        new_top = jnp.maximum(cur_top, lse)
        keep, weight = jnp.exp(cur_top - new_top), jnp.exp(lse - new_top)
        return (
            new_top,
            cur_acc * keep + weight * partial[:, :LATENT],
            cur_den * keep + weight,
        )

    _, accumulator, denominator = jax.lax.fori_loop(
        0,
        group,
        fold,
        (top, accumulator, denominator),
        unroll=unroll,
    )
    return accumulator / denominator


def _merge_received_sequences(
    received: PallasRefProtocol,
    own_rows: jax.Array,
    latents: jax.Array,
    group: int,
    slots: int,
    sequence_rows: int,
    *,
    unroll: int = 1,
) -> jax.Array:
    """Merge peer partials for a chunk of causal multi-row sequences.

    Each row's head slots causally fold in the new tokens of its sequence's
    earlier rows in the chunk (``latents`` ``[chunk, 640]``), then fold peers'
    partials online.
    """
    scores = jax.lax.dot_general(
        own_rows,
        latents,
        (((1,), (1,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )  # [rows, chunk]
    slot_row = jax.lax.broadcasted_iota(jnp.int32, scores.shape, 0) // slots
    key_row = jax.lax.broadcasted_iota(jnp.int32, scores.shape, 1)
    causal = (key_row // sequence_rows == slot_row // sequence_rows) & (
        key_row <= slot_row
    )
    scores = jnp.where(causal, scores, -jnp.inf)
    top = jnp.max(scores, axis=1, keepdims=True)  # finite: a row sees its own token
    weights = jnp.exp(scores - top)
    accumulator = jax.lax.dot_general(
        weights,
        latents[:, :LATENT],
        (((1,), (0,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    denominator = jnp.sum(weights, axis=1, keepdims=True)

    def fold(
        peer: jax.Array,
        carry: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Fold ``peer``'s partial output into the sequence merge state."""
        cur_top, cur_acc, cur_den = carry
        partial = received[peer].astype(jnp.float32)  # [rows, 640]
        lse = partial[:, LATENT : LATENT + 1] + partial[:, LATENT + 1 : LATENT + 2]
        new_top = jnp.maximum(cur_top, lse)
        keep, weight = jnp.exp(cur_top - new_top), jnp.exp(lse - new_top)
        return (
            new_top,
            cur_acc * keep + weight * partial[:, :LATENT],
            cur_den * keep + weight,
        )

    # Bounded unrolling exposes the next peer load without changing the
    # online merge order or materializing all 32 peers in FP32.
    def fold_group(
        peer: jax.Array,
        carry: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Fold ``unroll`` consecutive peers per outer loop step."""
        for i in range(unroll):
            carry = fold(peer * unroll + i, carry)
        return carry

    _, accumulator, denominator = jax.lax.fori_loop(
        0,
        group // unroll,
        fold_group,
        (top, accumulator, denominator),
    )
    return accumulator / denominator


def _lse_packet(
    normalized: jax.Array,
    lse: jax.Array,
    lse_rows: int,
) -> jax.Array:
    """Build a ``[rows, 640]`` BF16 ``lse_in_padding`` packet from a partial."""
    rows = normalized.shape[0]
    pad = rows - lse_rows
    high = lse.astype(jnp.bfloat16).astype(jnp.float32)
    low = (lse - high).astype(jnp.bfloat16).astype(jnp.float32)
    row = jax.lax.broadcasted_iota(jnp.int32, (rows, VREG_LANE_COUNT), 0)
    lane = jax.lax.broadcasted_iota(jnp.int32, (rows, VREG_LANE_COUNT), 1)
    high_vector = jnp.sum(
        jnp.where((lane == row) & (row < pad), high, 0.0),
        axis=0,
        keepdims=True,
    )
    low_vector = jnp.sum(
        jnp.where((lane == row) & (row < pad), low, 0.0),
        axis=0,
        keepdims=True,
    )
    packet = jnp.concatenate(
        (normalized, jnp.zeros((rows, LANES - LATENT), jnp.float32)),
        axis=1,
    )
    row640 = jax.lax.broadcasted_iota(jnp.int32, (rows, LANES), 0)
    pad_lanes = jnp.zeros((1, LANES - VREG_LANE_COUNT), jnp.float32)
    packet = jnp.where(
        row640 == pad,
        jnp.concatenate((high_vector, pad_lanes), axis=1),
        packet,
    )
    packet = jnp.where(
        row640 == pad + 1,
        jnp.concatenate((low_vector, pad_lanes), axis=1),
        packet,
    )
    return packet.astype(jnp.bfloat16)


def _merge_received_keys(
    received: PallasRefProtocol,
    own_rows: jax.Array,
    keys: jax.Array,
    key_valid: jax.Array,
    group: int,
    *,
    lse_rows: int = 0,
) -> jax.Array:
    """Merge ``group`` peers' partials with explicit ``keys`` in two passes.

    Every head-slot row attends valid entries of ``keys`` ``[K, 640]`` where
    ``key_valid`` ``[1, K]`` is nonzero (at least one), plus peers' partials.
    Loads all peers' LSE values first to take one global maximum before summing
    weighted partials without per-peer rescaling or exponentials.
    """
    keys_f32 = keys.astype(jnp.float32)
    scores = jax.lax.dot_general(
        own_rows,
        keys_f32,
        (((1,), (1,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )  # [rows, K]
    scores = jnp.where(key_valid != 0, scores, -jnp.inf)
    if lse_rows:
        # ``lse_in_padding``: the last ``lse_rows`` packet rows (a padding
        # row's head slots) hold the others' lse, packet row r at lane r.
        rows = received.shape[1]
        pad = rows - lse_rows
        vector = received[:, pad, pl.ds(0, VREG_LANE_COUNT)].astype(
            jnp.float32
        ) + received[:, pad + 1, pl.ds(0, VREG_LANE_COUNT)].astype(
            jnp.float32
        )  # [G, 128]
        row = jax.lax.broadcasted_iota(
            jnp.int32,
            (group, rows, VREG_LANE_COUNT),
            1,
        )
        lane = jax.lax.broadcasted_iota(
            jnp.int32,
            (group, rows, VREG_LANE_COUNT),
            2,
        )
        lse = jnp.sum(
            jnp.where(lane == row, vector[:, None, :], 0.0),
            axis=2,
            keepdims=True,
        )  # [G, rows, 1]
        lse = jnp.where(row[:, :, :1] < pad, lse, NEG)  # the padding row: no peers
    else:
        tails = received[:, :, LATENT : LATENT + VREG_LANE_COUNT].astype(
            jnp.float32
        )  # [G, rows, 128]
        lse = tails[:, :, 0:1] + tails[:, :, 1:2]  # [G, rows, 1]
    top = jnp.maximum(
        jnp.max(scores, axis=1, keepdims=True),
        jnp.max(lse, axis=0),
    )  # [rows, 1]
    weights = jnp.exp(scores - top)
    peer_weights = jnp.exp(lse - top[None])  # [G, rows, 1]
    accumulator = jax.lax.dot_general(
        weights,
        keys_f32[:, :LATENT],
        (((1,), (0,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    denominator = jnp.sum(weights, axis=1, keepdims=True) + jnp.sum(
        peer_weights,
        axis=0,
    )
    for peer in range(group):
        accumulator = accumulator + peer_weights[peer] * received[
            peer, :, :LATENT
        ].astype(jnp.float32)
    return accumulator / denominator


def _pack_fp8_query(query: jax.Array) -> jax.Array:
    """Pack FP8 query bytes with a lossless FP32 scale in padding lanes 576:580.

    Rounds to BF16 first, matching the BF16 query path. Integer transport
    preserves every scale bit, including bytes that would encode NaNs if treated
    as FP8 values. The owner's FP32 query stays live separately for the
    new-token score.
    """
    rounded = query.astype(jnp.bfloat16).astype(jnp.float32)
    maximum = jnp.max(jnp.abs(rounded), axis=1, keepdims=True)
    scale = jnp.where(maximum > 0, maximum / _FP8_E4M3_MAX, 1.0)
    # Perform masks/broadcasts in 32-bit lanes; packed-byte sublane broadcasts
    # are unsupported by Mosaic. Only the stored wire stays byte packed.
    payload = jax.lax.bitcast_convert_type(
        (rounded / scale).astype(jnp.float8_e4m3fn),
        jnp.uint8,
    ).astype(jnp.uint32)
    bits = jax.lax.bitcast_convert_type(scale, jnp.uint32)
    lanes = jax.lax.broadcasted_iota(jnp.int32, rounded.shape, 1)
    scale_bytes = (
        bits >> (((lanes - _COMPRESSED_KV_LANES) % _BYTES_PER_FP32) * _BITS_PER_BYTE)
    ) & _BYTE_MASK
    return jnp.where(lanes < _COMPRESSED_KV_LANES, payload, scale_bytes).astype(
        jnp.uint8
    )


def _unpack_fp8_query(packet: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Recover exact FP8 query and FP32 scale while clearing metadata lanes."""
    words_u32 = packet.astype(jnp.uint32)
    lanes = jax.lax.broadcasted_iota(jnp.int32, words_u32.shape, 1)
    # Reassemble the scale with a lane reduction; narrow byte slices lose bits on TPU.
    words = words_u32 << (
        ((lanes - _COMPRESSED_KV_LANES) % _BYTES_PER_FP32) * _BITS_PER_BYTE
    )
    bits = jnp.sum(
        jnp.where(
            (lanes >= _COMPRESSED_KV_LANES) & (lanes < _FP8_SCALE_END_LANE),
            words,
            jnp.uint32(0),
        ),
        axis=1,
        keepdims=True,
    )
    scale = jax.lax.bitcast_convert_type(bits, jnp.float32)
    payload = jnp.where(
        lanes < _COMPRESSED_KV_LANES,
        words_u32,
        jnp.uint32(0),
    ).astype(jnp.uint8)
    return jax.lax.bitcast_convert_type(payload, jnp.float8_e4m3fn), scale
