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
scoped VMEM, and :func:`scoped_scratch`) and :class:`DcpAttention` (``prime``
the first page DMAs early, ``attend`` later).
"""

from __future__ import annotations

import contextlib
import dataclasses
from typing import TYPE_CHECKING, TypedDict, Unpack

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from ..config import active as _config

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from types import SimpleNamespace

    from jax._src.lib import hlo as jax_hlo
    from jax._src.pallas import core as pl_core
    from jax._src.state import types as state_types

    type _TensorOrRef = (
        jax.Array
        | pl_core.MemoryRef
        | state_types.TransformedRef
        | tpu.SemaphoreType
        | SimpleNamespace
        | jax_hlo.HloModule
        | int
        | float
        | bool
        | None
    )
else:
    type _TensorOrRef = jax.Array

PAGE_TOKENS = 256
LANES = 640  # latent 512 | rope 64 | zero 64
LATENT = 512
ROPE_LANES = 64
ROPE_END_LANE = 576
SCALE_END_LANE = 580
SCALE_WORD_BYTES = 4
BYTE_SHIFT_BITS = 8
BYTE_MASK = 255
LSE_VECTOR_LANES = 128
SUBLANE_ROWS = 8
SCORE_TILE_ROUND = 16
HEADS_PER_RANK = 3
DP2_HEADS = 6
DP4_HEADS = 12  # attention DP4's held heads per rank (four TP32 shards)
DP4_GROUP = 8
DP2_GROUP = 16
FULL_TP_GROUP = 32
NEG = -1e30  # finite "no tokens" log-sum-exp
MAX_ROWS = 8  # batch rows of an unchunked call; a larger batch runs in chunks
HOST_LANES = 8  # DCP ranks per host (rank = host * 8 + lane)
B2_BATCH = 2
B4_BATCH = 4
B16_BATCH = 16
B24_BATCH = 24
B32_BATCH = 32
B48_BATCH = 48
B64_BATCH = 64
RING_SLOTS_DEFAULT = 2
RING_SLOTS_WIDE = 3
RING_SLOTS_MAX = 4
DEFAULT_KTILE = 1024
WIDE_KTILE = 2048
PACKED_FP8_PAGE_TOKENS = 512
FP8_MAX_FINITE = 448.0
FP8_PV_SCALE = 256
FP8_INV_PV_SCALE = 1.0 / 256
MIN_LSE_SLOTS = 2
LSE_PARTS = 2
PEER_UNROLL_WIDE = 4
WORD_BYTES = 4
PACKED_B2_SLOTS = 4


def _lookup_rows(batch: int, heads: int = HEADS_PER_RANK) -> bool:
    """Return whether page steps find their row by binary search over SMEM bounds."""
    if heads == DP4_HEADS:
        return batch in (B16_BATCH, B24_BATCH, B32_BATCH)
    return batch in (B32_BATCH, B48_BATCH, B64_BATCH)


def chunk_layout(
    batch: int,
    heads: int = HEADS_PER_RANK,
    *,
    chunk_rows: int | None = None,
) -> tuple[int, int]:
    """Return ``(chunk rows, ring slots)`` of a chunked call (``batch > MAX_ROWS``).

    Uses 8-row chunks at B16/B32/B48/B64 and attention DP4's 24 group rows
    (amortizing exchanges), else 4 (also attention DP2's B128 and DP4's: 64
    group rows of six heads, 32 of twelve, for VMEM); ring slots 4 at B32 with
    three heads per rank, 3 for 8-row chunks, else 2 (``persistent_scratch``
    uses 4 for FP8 query packets). ``chunk_rows`` overrides the width and keeps
    at least three slots for a staged combine.
    """
    wide = batch <= B16_BATCH or (
        batch in (B32_BATCH, B48_BATCH, B64_BATCH)
        and not (batch == B64_BATCH and heads == DP2_HEADS)
    )
    if heads == DP4_HEADS:
        wide = batch <= B24_BATCH
    rows = chunk_rows if chunk_rows is not None else (MAX_ROWS if wide else B4_BATCH)
    ring = (
        RING_SLOTS_MAX
        if (batch == B32_BATCH and heads == HEADS_PER_RANK)
        else (
            RING_SLOTS_WIDE if (wide or chunk_rows is not None) else RING_SLOTS_DEFAULT
        )
    )
    if (
        rows not in (B4_BATCH, MAX_ROWS)
        or batch % rows
        or rows > MAX_ROWS
        or ring not in (RING_SLOTS_DEFAULT, RING_SLOTS_WIDE, RING_SLOTS_MAX)
    ):
        msg = (
            f"DCP chunks: {rows} rows must divide the batch {batch} "
            f"(<= {MAX_ROWS}), ring 2, 3 or 4"
        )
        raise ValueError(msg)
    return rows, ring


GROUP = _config().dcp_group
GLOBAL_PAGES = _config().dcp_tables == "global"
SPAN = 4  # pages per step
BUFFERS = 4  # steps in flight


def page_plan(
    batch: int,
    heads: int = HEADS_PER_RANK,
    sequence_rows: int = 1,
    cache_dtype: _TensorOrRef = jnp.bfloat16,
    *,
    force_chunks: bool | None = None,
) -> dict[str, int]:
    """Return the decoder's page buffer configuration for ``batch`` rows."""
    if batch <= B4_BATCH:
        return {"span": SPAN, "buffers": BUFFERS}
    if sequence_rows == 1:
        deep = (
            (batch == B64_BATCH and heads == DP2_HEADS)
            or (batch == B32_BATCH and heads == DP4_HEADS)
        ) and cache_dtype == jnp.float8_e4m3fn
        return {
            "span": SPAN,
            "buffers": (RING_SLOTS_WIDE if chunked(batch) and not deep else BUFFERS),
        }
    span = max(1, SPAN // 2)
    buffers = (
        BUFFERS
        if batch == B64_BATCH or (batch == B32_BATCH and heads != HEADS_PER_RANK)
        else BUFFERS * SPAN // span
    )
    return {
        "span": span,
        "buffers": (
            buffers
            if not chunked(batch, force_chunks)
            else max(RING_SLOTS_DEFAULT, buffers * 3 // 4)
        ),
    }


MAX_GROUPS = 2  # row groups of a call (see packet_layout)


def packet_layout(batch: int, heads: int = HEADS_PER_RANK) -> tuple[int, int]:
    """Return ``(head slots, row groups)`` for ``batch`` rows of ``heads`` heads."""
    return (
        (PACKED_B2_SLOTS, MAX_GROUPS)
        if batch == B2_BATCH and heads == HEADS_PER_RANK
        else (heads, 1)
    )


def owner_index(table: _TensorOrRef) -> _TensorOrRef:
    """Return this rank's page-owner index from a static per-mesh-position table."""
    rank = jax.lax.axis_index("tp")
    if table is None:
        return rank
    owner: _TensorOrRef = 0
    for position, value in enumerate(table):
        owner = jnp.where(rank == position, value, owner)
    return owner


_REGISTRATION_STATE = {"done": False}


def _ensure_dataclasses_registered() -> None:
    """Register MLA DCP PyTree dataclasses with JAX lazily on first instantiation."""
    if _REGISTRATION_STATE["done"]:
        return
    jax.tree_util.register_dataclass(
        DcpCache,
        data_fields=["pool_ref", "table_ref", "layer", "owner"],
        meta_fields=["pages", "group", "global_pages", "table_row"],
    )
    jax.tree_util.register_dataclass(
        DcpWorkspace,
        data_fields=[
            "gathered",
            "send",
            "receive",
            "gather_semaphores",
            "combine_semaphores",
            "queries",
            "pages",
            "page_semaphores",
            "maximum",
            "denominator",
            "accumulator",
            "own_rows",
            "new_rows",
            "merged",
            "lookup_bounds",
            "lookup_counts",
            "query_scale",
        ],
        meta_fields=[],
    )
    _REGISTRATION_STATE["done"] = True


@dataclasses.dataclass(frozen=True)
class DcpCache:
    """This rank's latent pages in HBM and flat block table in SMEM."""

    pool_ref: _TensorOrRef
    table_ref: _TensorOrRef
    pages: int
    layer: _TensorOrRef = None
    group: int = GROUP
    global_pages: bool = GLOBAL_PAGES
    owner: _TensorOrRef = None
    table_row: Callable[[_TensorOrRef], _TensorOrRef] | None = None

    def __post_init__(self) -> None:
        """Ensure JAX PyTree registration is active when ``DcpCache`` is created."""
        _ensure_dataclasses_registered()

    @property
    def dtype(self) -> _TensorOrRef:
        """Return the element dtype of the underlying KV pool."""
        return self.pool_ref[0].dtype

    @property
    def page_tokens(self) -> int:
        """Return the number of tokens stored per physical page block."""
        ref = self.pool_ref[0]
        return int(ref.shape[-3] * ref.shape[-2])

    def start_page_copy(
        self,
        row: _TensorOrRef,
        local: _TensorOrRef,
        member: _TensorOrRef,
        destination: _TensorOrRef,
        semaphore: _TensorOrRef,
    ) -> None:
        """Start the DMA of ``row``'s local page ``local`` on ``member``."""
        entry = local * self.group + member if self.global_pages else local
        if self.table_row is not None:
            row = self.table_row(row)
        block = self.table_ref[row * self.pages + entry]
        for index, ref in enumerate(self.pool_ref):

            @pl.when(self.layer == index)
            def start(*, ref: _TensorOrRef = ref) -> None:
                """Issue the async DMA copy for the matching layer pool."""
                tpu.make_async_copy(ref.at[block], destination, semaphore).start()

    def start_native_copies(
        self,
        row: _TensorOrRef,
        locals_: list[_TensorOrRef],
        member: _TensorOrRef,
        destinations: list[_TensorOrRef],
        semaphore: _TensorOrRef,
    ) -> None:
        """Start DMAs of ``row``'s local pages ``locals_`` into native destinations."""
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
            """Select the active layer pool by binary tree recursion."""
            if end - first == 1:
                for block, destination in zip(blocks, destinations, strict=True):
                    tpu.make_async_copy(
                        self.pool_ref[first].at[block].reshape(self.page_tokens, LANES),
                        destination,
                        semaphore,
                    ).start()
                return
            middle = (first + end) // 2
            pl.when(self.layer < middle)(lambda: pick(first, middle))
            pl.when(self.layer >= middle)(lambda: pick(middle, end))

        pick(0, len(self.pool_ref))


@dataclasses.dataclass(frozen=True)
class DcpWorkspace:
    """Persistent and per-layer VMEM/SMEM workspace for DCP MLA."""

    gathered: _TensorOrRef
    send: _TensorOrRef
    receive: _TensorOrRef
    gather_semaphores: _TensorOrRef
    combine_semaphores: _TensorOrRef
    queries: _TensorOrRef
    pages: _TensorOrRef
    page_semaphores: _TensorOrRef
    maximum: _TensorOrRef
    denominator: _TensorOrRef
    accumulator: _TensorOrRef
    own_rows: _TensorOrRef
    new_rows: _TensorOrRef
    merged: _TensorOrRef
    lookup_bounds: _TensorOrRef
    lookup_counts: _TensorOrRef
    query_scale: _TensorOrRef

    def __post_init__(self) -> None:
        """Ensure JAX PyTree registration is active when ``DcpWorkspace`` is created."""
        _ensure_dataclasses_registered()


def chunked(batch: int, force: _TensorOrRef = None) -> bool:
    """Return whether a call of ``batch`` rows runs in chunks of ``MAX_ROWS``."""
    return batch > MAX_ROWS or bool(force)


GROUPED_SPEC_SHAPES = (
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
    """Return whether verifier shape supports grouped FP8 QK/PV and query transport."""
    if group == DP2_GROUP and heads == DP2_HEADS:
        return (batch, sequence_rows) in (*GROUPED_SPEC_SHAPES, (16, 4))
    return (
        group == FULL_TP_GROUP
        and heads == HEADS_PER_RANK
        and (batch, sequence_rows) in GROUPED_SPEC_SHAPES
    )


class _PersistentScratchKwargs(TypedDict, total=False):
    """Optional configuration keyword arguments for ``persistent_scratch``."""

    cache_dtype: _TensorOrRef
    sequence_rows: int
    fp8_wire: bool
    force_chunks: bool | int
    chunk_rows: int | None


def persistent_scratch(
    batch: int,
    group: int = GROUP,
    heads: int = HEADS_PER_RANK,
    **scratch_kwargs: Unpack[_PersistentScratchKwargs],
) -> tuple[_TensorOrRef, ...]:
    """Allocate persistent VMEM and DMA semaphore specs for DCP communication."""
    cache_dtype = scratch_kwargs.get("cache_dtype", jnp.bfloat16)
    sequence_rows = int(scratch_kwargs.get("sequence_rows", 1))
    fp8_wire = bool(scratch_kwargs.get("fp8_wire", False))
    force_chunks = bool(scratch_kwargs.get("force_chunks", False))
    chunk_rows = scratch_kwargs.get("chunk_rows")
    plain = sequence_rows == 1 and (
        (batch in (B32_BATCH, B64_BATCH) and heads == HEADS_PER_RANK)
        or (
            group == DP2_GROUP
            and batch in (B16_BATCH, B32_BATCH, B48_BATCH, B64_BATCH)
            and heads == DP2_HEADS
        )
        or (
            group == DP4_GROUP
            and batch in (B16_BATCH, B24_BATCH, B32_BATCH)
            and heads == DP4_HEADS
        )
    )
    query_dtype = (
        jnp.uint8
        if cache_dtype == jnp.float8_e4m3fn
        and (plain or fp8_wire or _grouped_spec(batch, group, heads, sequence_rows))
        else jnp.bfloat16
    )
    if chunked(batch, force_chunks):
        chunk, ring = chunk_layout(batch, heads, chunk_rows=chunk_rows)
        rows = packet_layout(batch, heads)[0] * chunk
        if query_dtype == jnp.uint8 and not (group == DP2_GROUP and sequence_rows > 1):
            ring = RING_SLOTS_MAX
        return (
            tpu.VMEM((ring, group, rows, LANES), query_dtype),
            tpu.VMEM((group, rows, LANES), jnp.bfloat16),
            tpu.VMEM((ring, group, rows, LANES), jnp.bfloat16),
            tpu.SemaphoreType.DMA((2, ring)),
            tpu.SemaphoreType.DMA((2, ring)),
        )
    rows = packet_layout(batch, heads)[0] * batch
    return (
        tpu.VMEM((2, group, rows, LANES), query_dtype),
        tpu.VMEM((group, rows, LANES), jnp.bfloat16),
        tpu.VMEM((2, group, rows, LANES), jnp.bfloat16),
        tpu.SemaphoreType.DMA((2, 2, MAX_GROUPS)),
        tpu.SemaphoreType.DMA((2, 2, MAX_GROUPS)),
    )


PERSISTENT_NAMES = (
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
    cache_dtype: _TensorOrRef = jnp.bfloat16,
) -> bool:
    """Return whether a call stages pages in the MXU's native 2D tile layout."""
    if heads == DP4_HEADS:
        return sequence_rows == 1 and (
            batch in (B24_BATCH, B32_BATCH)
            or (batch == B16_BATCH and cache_dtype == jnp.float8_e4m3fn)
        )
    return (
        heads == DP2_HEADS
        and sequence_rows == 1
        and (
            batch in (B48_BATCH, B64_BATCH)
            or (
                batch == B32_BATCH
                and cache_dtype == jnp.float8_e4m3fn
                and _config().mla_decode_ktile == DEFAULT_KTILE
            )
        )
    )


def staged_folds(
    batch: int,
    heads: int = HEADS_PER_RANK,
    sequence_rows: int = 1,
    cache_dtype: _TensorOrRef = jnp.bfloat16,
) -> int:
    """Return the number of softmax folds per page step."""
    if heads == DP4_HEADS:
        return (
            2
            if (
                native_pages(batch, heads, sequence_rows, cache_dtype)
                and batch in (B16_BATCH, B24_BATCH)
                and cache_dtype == jnp.float8_e4m3fn
            )
            else 1
        )
    return (
        2
        if (
            native_pages(batch, heads, sequence_rows, cache_dtype)
            and batch in (B32_BATCH, B48_BATCH)
            and cache_dtype == jnp.float8_e4m3fn
            and _config().mla_decode_ktile == DEFAULT_KTILE
        )
        else 1
    )


class _ScopedScratchKwargs(TypedDict, total=False):
    """Optional configuration keyword arguments for ``scoped_scratch``."""

    group: int
    cache_dtype: _TensorOrRef
    page_tokens: int
    sequence_rows: int
    force_chunks: bool | int
    chunk_rows: int | None


def scoped_scratch(
    batch: int,
    span: int = SPAN,
    buffers: int = BUFFERS,
    heads: int = HEADS_PER_RANK,
    **scratch_kwargs: Unpack[_ScopedScratchKwargs],
) -> tuple[_TensorOrRef, ...]:
    """Allocate per-MLA-invocation scoped VMEM/SMEM specs for DCP attention."""
    group = int(scratch_kwargs.get("group", GROUP))
    cache_dtype = scratch_kwargs.get("cache_dtype", jnp.bfloat16)
    page_tokens = int(scratch_kwargs.get("page_tokens", PAGE_TOKENS))
    sequence_rows = int(scratch_kwargs.get("sequence_rows", 1))
    force_chunks = bool(scratch_kwargs.get("force_chunks", False))
    chunk_rows = scratch_kwargs.get("chunk_rows")
    rows = packet_layout(batch, heads)[0] * batch
    tile = head_rows(batch, group, heads)
    packing = WORD_BYTES // jnp.dtype(cache_dtype).itemsize
    wide = _config().mla_decode_ktile == WIDE_KTILE
    if wide and (
        group != DP2_GROUP
        or heads != DP2_HEADS
        or cache_dtype != jnp.float8_e4m3fn
        or page_tokens != PACKED_FP8_PAGE_TOKENS
    ):
        msg = (
            "mla_decode_ktile=2048 requires DCP16, six heads per rank and "
            "FP8 e4m3fn KV with 512-token pages"
        )
        raise ValueError(msg)
    if cache_dtype == jnp.float8_e4m3fn and batch in (
        MAX_ROWS,
        B16_BATCH,
        B32_BATCH,
        B64_BATCH,
    ):
        span = (
            SPAN
            if wide and batch == B32_BATCH and sequence_rows == 1
            else min(span, SPAN * PAGE_TOKENS // page_tokens)
        )
    else:
        span = span * PAGE_TOKENS // page_tokens
    span = max(1, span) * staged_folds(batch, heads, sequence_rows, cache_dtype)
    pages = (
        (buffers, span * page_tokens, LANES)
        if native_pages(batch, heads, sequence_rows, cache_dtype)
        else (buffers, span * page_tokens // packing, packing, LANES)
    )
    held = (
        chunk_layout(batch, heads, chunk_rows=chunk_rows)[0]
        if chunked(batch, force_chunks)
        else batch
    )
    return (
        tpu.VMEM((held, tile, LANES), cache_dtype),
        tpu.VMEM(pages, cache_dtype),
        tpu.SemaphoreType.DMA((buffers,)),
        tpu.VMEM((held, tile, 1), jnp.float32),
        tpu.VMEM((held, tile, 1), jnp.float32),
        tpu.VMEM((held, tile, LATENT), jnp.float32),
        *(
            (
                tpu.VMEM(
                    chunk_rows_shape(batch, LANES, heads, chunk_rows=chunk_rows),
                    jnp.float32,
                ),
                tpu.VMEM(
                    chunk_rows_shape(batch, LANES, heads, chunk_rows=chunk_rows),
                    jnp.float32,
                ),
                tpu.VMEM(
                    chunk_rows_shape(batch, LATENT, heads, chunk_rows=chunk_rows),
                    jnp.float32,
                ),
            )
            if chunked(batch, force_chunks)
            else (
                tpu.VMEM((rows, LANES), jnp.float32),
                tpu.VMEM((rows, LANES), jnp.float32),
                tpu.VMEM((SUBLANE_ROWS, LSE_VECTOR_LANES), jnp.float32),
            )
        ),
        tpu.SMEM((batch if _lookup_rows(batch, heads) else 1,), jnp.int32),
        tpu.SMEM((batch if _lookup_rows(batch, heads) else 1,), jnp.int32),
        tpu.VMEM((held, tile, 1), jnp.float32),
    )


def head_rows(
    batch: int,
    group: int = GROUP,
    heads: int = HEADS_PER_RANK,
) -> int:
    """Return the score-tile row count, rounded up to a multiple of 16."""
    slots = packet_layout(batch, heads)[0] * group
    return -(-slots // SCORE_TILE_ROUND) * SCORE_TILE_ROUND


def chunk_rows_shape(
    batch: int,
    lanes: int,
    heads: int = HEADS_PER_RANK,
    *,
    chunk_rows: int | None = None,
) -> tuple[int, int, int]:
    """Return the per-chunk row buffer shape ``[chunks, rows_padded_to_8, lanes]``."""
    chunk = chunk_layout(batch, heads, chunk_rows=chunk_rows)[0]
    rows = packet_layout(batch, heads)[0] * chunk
    return (batch // chunk, -(-rows // SUBLANE_ROWS) * SUBLANE_ROWS, lanes)


SCOPED_NAMES = (
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


class _DcpAttentionKwargs(TypedDict, total=False):
    """Optional configuration keyword arguments for ``DcpAttention``."""

    skip: Iterable[str]
    sequence_rows: int
    heads: int
    chunk_rows: int | None
    cache_verify_rows: bool


class _AttendExplicitKeysKwargs(TypedDict, total=False):
    """Optional explicit-key arguments for ``DcpAttention.attend``."""

    keys: _TensorOrRef
    key_valid: _TensorOrRef


@dataclasses.dataclass(frozen=True, slots=True)
class ChunkLoopLayout:
    """Loop geometry and optional explicit-key tensors for chunked DCP attention."""

    chunk: int
    count: int
    ring_slots: int
    lookahead: int = 1
    rows: int = 0
    hier: bool = False
    keys: _TensorOrRef = None
    key_valid: _TensorOrRef = None
    host: _TensorOrRef = None
    lane: _TensorOrRef = None


@dataclasses.dataclass(frozen=True, slots=True)
class SequenceMergeSpec:
    """Configuration for merging multi-row sequence chunk partials."""

    group: int
    slots: int
    sequence_rows: int
    unroll: int = 1


@dataclasses.dataclass(frozen=True, slots=True)
class KeyMergeSpec:
    """Configuration for merging received peer partials with explicit keys."""

    group: int
    lse_rows: int = 0


class DcpAttention:
    """One DCP attention invocation for ``batch`` rows at ``position_ref``."""

    trace: str | None = None
    lse_in_padding: bool = False
    host_combine: _TensorOrRef = None
    force_chunked: bool = False
    group_fold: bool = False
    _lane_major: bool = False

    def __init__(
        self,
        cache: DcpCache,
        workspace: DcpWorkspace,
        position_ref: _TensorOrRef,
        batch: int,
        **attn_kwargs: Unpack[_DcpAttentionKwargs],
    ) -> None:
        """Initialize per-row page counts and bounds for one DCP attention call."""
        skip = attn_kwargs.get("skip", ())
        sequence_rows = int(attn_kwargs.get("sequence_rows", 1))
        heads = int(attn_kwargs.get("heads", HEADS_PER_RANK))
        chunk_rows = attn_kwargs.get("chunk_rows")
        cache_verify_rows = bool(attn_kwargs.get("cache_verify_rows", False))
        if batch > MAX_ROWS:
            chunk = chunk_layout(batch, heads, chunk_rows=chunk_rows)[0]
            if chunk % sequence_rows:
                msg = (
                    f"DCP chunks of {chunk} rows must hold whole "
                    f"{sequence_rows}-row sequences"
                )
                raise ValueError(msg)
        if sequence_rows < 1 or batch % sequence_rows:
            msg = "sequence_rows must divide the batch"
            raise ValueError(msg)
        if cache.group not in (DP4_GROUP, DP2_GROUP, FULL_TP_GROUP):
            msg = "DCP groups are 8, 16 or 32 consecutive ranks"
            raise ValueError(msg)
        self.cache = cache
        self.ws = workspace
        self.position_ref = position_ref
        self.batch = batch
        self.sequence_rows = sequence_rows
        self.chunk_rows = chunk_rows
        self.cache_verify_rows = cache_verify_rows
        self.heads = heads
        self.slots, self.groups = packet_layout(batch, heads)
        self.group = cache.group
        self.tile = int(workspace.queries.shape[1])
        if self.group * self.slots > self.tile:
            msg = (
                f"{self.group} ranks x {self.slots} head slots exceed {self.tile} rows"
            )
            raise ValueError(msg)
        self.buffers = int(workspace.pages.shape[0])
        self.page_tokens = cache.page_tokens
        self.fp8 = cache.dtype == jnp.float8_e4m3fn
        self.packing = WORD_BYTES // jnp.dtype(cache.dtype).itemsize
        self.page_rows = self.page_tokens // self.packing
        self.native = workspace.pages.ndim == RING_SLOTS_WIDE
        self.span = int(
            workspace.pages.shape[1]
            // (self.page_tokens if self.native else self.page_rows)
        )
        self.folds = staged_folds(batch, heads, sequence_rows, cache.dtype)
        self.fold_tokens = self.span * self.page_tokens // self.folds
        self.skip = set(skip)
        rank = jax.lax.axis_index("tp")
        self.member = rank % self.group
        self.base = rank - self.member
        self.owner = self.member if cache.owner is None else cache.owner
        self.shared = sequence_rows > 1 and all(
            first % sequence_rows == 0 and end % sequence_rows == 0
            for first, end in self.row_groups()
        )
        if cache_verify_rows and not (self.shared and self.fp8):
            msg = "cached verify rows need shared FP8 sequences"
            raise ValueError(msg)
        self._init_page_counts()

    def _init_page_counts(self) -> None:
        """Compute per-row local page counts, step offsets, and SMEM lookup tables."""
        self.counts: list[_TensorOrRef] = []
        starts: list[_TensorOrRef] = [0]
        for batch_idx in range(self.batch):
            limit = self._limit(batch_idx) + (
                self.sequence_rows - 1 if self.cache_verify_rows else 0
            )
            needed = (limit + self.page_tokens - 1) // self.page_tokens
            count = jnp.maximum(needed - self.owner + self.group - 1, 0) // self.group
            self.counts.append(
                count if not self.shared or batch_idx % self.sequence_rows == 0 else 0
            )
            starts.append(starts[-1] + (self.counts[-1] + self.span - 1) // self.span)
        self.starts = starts
        self.total = starts[-1]
        if _lookup_rows(self.batch, self.heads):
            for batch_idx in range(self.batch):
                self.ws.lookup_bounds[batch_idx] = starts[batch_idx]
                self.ws.lookup_counts[batch_idx] = self.counts[batch_idx]

    def _limit(self, row: _TensorOrRef) -> _TensorOrRef:
        """Return the pool token upper bound visible to ``row``."""
        if self.sequence_rows == 1:
            return self.position_ref[row]
        return self.position_ref[row // self.sequence_rows * self.sequence_rows]

    def _locate(
        self,
        step: _TensorOrRef,
    ) -> tuple[_TensorOrRef, _TensorOrRef, _TensorOrRef]:
        """Map flattened page step ``step`` to ``(row, local_step, page_count)``."""
        if _lookup_rows(self.batch, self.heads):
            row = jnp.int32(0)
            for bit in range((self.batch - 1).bit_length() - 1, -1, -1):
                candidate = jnp.minimum(row + (1 << bit), self.batch - 1)
                row = jnp.where(
                    step >= self.ws.lookup_bounds[candidate], candidate, row
                )
            return (
                row,
                step - self.ws.lookup_bounds[row],
                self.ws.lookup_counts[row],
            )
        row_val: _TensorOrRef = 0
        first_val: _TensorOrRef = 0
        count_val: _TensorOrRef = self.counts[0]
        for batch_idx in range(1, self.batch):
            later = step >= self.starts[batch_idx]
            row_val = jnp.where(later, batch_idx, row_val)
            first_val = jnp.where(later, self.starts[batch_idx], first_val)
            count_val = jnp.where(later, self.counts[batch_idx], count_val)
        return row_val, step - first_val, count_val

    def _start_step(self, step: _TensorOrRef) -> None:
        """Issue ``span`` page DMAs for ``step``; clamp out-of-range pages to last."""
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

    def _wait_step(self, step: _TensorOrRef) -> None:
        """Wait for the page DMAs of ``step`` to arrive in VMEM."""
        slot = step % self.buffers
        tpu.make_async_copy(
            self.ws.pages.at[slot],
            self.ws.pages.at[slot],
            self.ws.page_semaphores.at[slot],
        ).wait()

    def _attend_shared_group_fold(
        self,
        load: Callable[..., _TensorOrRef],
        visible: _TensorOrRef,
        row: _TensorOrRef,
        base: _TensorOrRef,
    ) -> None:
        """Fold grouped FP8 explicit-key sequences in chunks of up to 4 rows.

        Each use computes ``row - base`` again, so the lowered kernel matches the
        port op for op.
        """
        active = self.sequence_rows - (1 if self.lse_in_padding else 0)
        first_rows = min(B4_BATCH, active)
        first = self._group_scores(load, visible, row - base, first_rows)
        if active > B4_BATCH:
            second = self._group_scores(
                load, visible, row - base + B4_BATCH, active - B4_BATCH
            )
            self._group_fold(first, load, row - base, first_rows)
            self._group_fold(second, load, row - base + B4_BATCH, active - B4_BATCH)
        else:
            self._group_fold(first, load, row - base, first_rows)

    def _attend_shared_grouped_spec(
        self,
        load: Callable[..., _TensorOrRef],
        visible: _TensorOrRef,
        held_base: _TensorOrRef,
    ) -> None:
        """Fold grouped speculative-verify sequences 4 rows at a time."""

        def fold_group(
            group_idx: _TensorOrRef,
            score: _TensorOrRef,
        ) -> _TensorOrRef:
            """Issue the next 4-row QK group before folding the current scores."""
            upcoming = self._group_scores(
                load,
                visible,
                held_base + (group_idx + 1) * B4_BATCH,
                B4_BATCH,
            )
            self._group_fold(score, load, held_base + group_idx * B4_BATCH, B4_BATCH)
            return upcoming

        final = jax.lax.fori_loop(
            0,
            self.sequence_rows // B4_BATCH - 1,
            fold_group,
            self._group_scores(load, visible, held_base, B4_BATCH),
        )
        self._group_fold(
            final,
            load,
            held_base + self.sequence_rows - B4_BATCH,
            B4_BATCH,
        )

    def _attend_shared_pipelined(
        self,
        load: Callable[..., _TensorOrRef],
        visible: _TensorOrRef,
        held_base: _TensorOrRef,
    ) -> None:
        """Fold shared sequence rows with one-row QK lookahead."""

        def fold_row(
            row_offset: _TensorOrRef,
            score: _TensorOrRef,
        ) -> _TensorOrRef:
            """Issue the next row's QK before folding the current row's scores."""
            upcoming = self._scores(load, visible, held_base + row_offset + 1)
            self._fold_scores(score, load, held_base + row_offset)
            return upcoming

        active = self.sequence_rows - (1 if self.lse_in_padding else 0)
        final = jax.lax.fori_loop(
            0,
            active - 1,
            fold_row,
            self._scores(load, visible, held_base),
        )
        self._fold_scores(final, load, held_base + active - 1)

    def _attend_shared_step(
        self,
        load: Callable[..., _TensorOrRef],
        visible: _TensorOrRef,
        held_base: _TensorOrRef,
    ) -> None:
        """Dispatch a shared multi-row sequence step to its specialized fold."""
        if self.fp8 and _grouped_spec(
            self.batch, self.group, self.heads, self.sequence_rows
        ):
            self._attend_shared_grouped_spec(load, visible, held_base)
        elif self.fp8 or self.batch in (B16_BATCH, B32_BATCH):
            self._attend_shared_pipelined(load, visible, held_base)
        else:

            @pl.loop(0, self.sequence_rows - (1 if self.lse_in_padding else 0))
            def fold_row(row_offset: _TensorOrRef) -> None:
                """Fold each active sequence row sequentially."""
                self._fold_pages(load, visible, held_base + row_offset)

    def _attend_step(self, step: _TensorOrRef, base: _TensorOrRef = 0) -> None:
        """Execute one step of local page attention and prefetch ahead."""
        self._wait_step(step)

        @pl.when(step + self.buffers - 1 < self.total)
        def prefetch() -> None:
            """Issue the page DMA ``buffers - 1`` steps ahead."""
            self._start_step(step + self.buffers - 1)

        if "pagecompute" in self.skip:
            return
        row, visible, load = self._step_pages(step)
        if self.cache_verify_rows:
            self._attend_cached_verify(step, row, load, base)
        elif (
            self.shared
            and self.fp8
            and self.group_fold
            and self.sequence_rows in (B4_BATCH, MAX_ROWS)
        ):
            self._attend_shared_group_fold(load, visible, row, base)
        elif self.shared:
            self._attend_shared_step(load, visible, row - base)
        else:
            self._fold_pages(load, visible, row - base)

    def _cache_verify_row(
        self,
        slot: _TensorOrRef,
        local_step: _TensorOrRef,
        row: _TensorOrRef,
    ) -> None:
        """Write an in-flight verify row into the current page buffer if owned."""
        position = self.position_ref[row]
        page = position // self.page_tokens
        local_page = page // self.group - local_step * self.span

        @pl.when(
            (page % self.group == self.owner)
            & (local_page >= 0)
            & (local_page < self.span)
        )
        def write() -> None:
            """Overwrite the target token slot in VMEM with ``row``'s new latent."""
            if chunked(self.batch, self.force_chunked):
                chunk = chunk_layout(
                    self.batch, self.heads, chunk_rows=self.chunk_rows
                )[0]
                latent = self.ws.new_rows[row // chunk, self.slots * (row % chunk)]
            else:
                latent = self.ws.new_rows[self.slots * row]
            offset = local_page * self.page_tokens + position % self.page_tokens
            tile_tokens = SUBLANE_ROWS * self.packing
            first = pl.multiple_of(offset // tile_tokens * SUBLANE_ROWS, SUBLANE_ROWS)
            words = (
                self.ws.pages.bitcast(jnp.uint32).at[slot]
                if self.native
                else self.ws.pages.bitcast(jnp.uint32)
                .at[slot]
                .reshape(self.span * self.page_rows, LANES)
            )
            block = tpu.bitcast(words[pl.ds(first, SUBLANE_ROWS)], self.cache.dtype)
            token_idx = jax.lax.broadcasted_iota(jnp.int32, block.shape, 0)
            block = jnp.where(
                token_idx == offset % tile_tokens,
                latent[None, :].astype(self.cache.dtype),
                block,
            )
            words[pl.ds(first, SUBLANE_ROWS)] = tpu.bitcast(block, jnp.uint32)

    def _attend_cached_verify(
        self,
        step: _TensorOrRef,
        row: _TensorOrRef,
        load: Callable[..., _TensorOrRef],
        base: _TensorOrRef,
    ) -> None:
        """Insert earlier verify rows into the page buffer and attend causally."""
        _, local_step, _ = self._locate(step)

        @pl.loop(0, self.sequence_rows - 1)
        def cache(row_offset: _TensorOrRef) -> None:
            """Stage verify row ``row_offset`` into the page buffer."""
            self._cache_verify_row(step % self.buffers, local_step, row + row_offset)

        offset = jax.lax.broadcasted_iota(
            jnp.int32, (1, self.span * self.page_tokens), 1
        )
        page = (
            local_step * self.span + offset // self.page_tokens
        ) * self.group + self.owner
        token_pos = page * self.page_tokens + offset % self.page_tokens
        if _grouped_spec(self.batch, self.group, self.heads, self.sequence_rows):
            for first in range(0, self.sequence_rows, B4_BATCH):
                query = (
                    jax.lax.broadcasted_iota(jnp.int32, (B4_BATCH * self.tile, 1), 0)
                    // self.tile
                )
                visible = token_pos < self.position_ref[row] + first + query
                score = self._group_scores(load, visible, row - base + first, B4_BATCH)
                self._group_fold(score, load, row - base + first, B4_BATCH)
        else:

            @pl.loop(0, self.sequence_rows)
            def query(row_offset: _TensorOrRef) -> None:
                """Fold pages for verify row ``row_offset``."""
                self._fold_pages(
                    load,
                    token_pos < self.position_ref[row + row_offset],
                    row - base + row_offset,
                )

    def _step_pages(
        self,
        step: _TensorOrRef,
    ) -> tuple[_TensorOrRef, _TensorOrRef, Callable[..., _TensorOrRef]]:
        """Clean stale bytes in ``step``'s pages and return ``(row, visible, load)``."""
        row, local_step, _ = self._locate(step)
        offset = jax.lax.broadcasted_iota(
            jnp.int32, (1, self.span * self.page_tokens), 1
        )
        global_page = (
            local_step * self.span + offset // self.page_tokens
        ) * self.group + self.owner
        token_pos = global_page * self.page_tokens + offset % self.page_tokens
        visible = token_pos < self._limit(row)
        slot = step % self.buffers
        self._clean_step(slot, row, local_step)
        return row, visible, self._pages(slot)

    def _pages(self, slot: _TensorOrRef) -> Callable[..., _TensorOrRef]:
        """Return a reader ``load(lanes, tokens=None)`` for page buffer ``slot``."""
        if self.native:

            def native(
                lanes: _TensorOrRef,
                tokens: _TensorOrRef = None,
            ) -> _TensorOrRef:
                """Slice native 2D page buffer ``slot`` along tokens and lanes."""
                return self.ws.pages[
                    slot, slice(None) if tokens is None else tokens, lanes
                ]

            return native
        words = (
            self.ws.pages.bitcast(jnp.uint32)
            .at[slot]
            .reshape(self.span * self.page_rows, LANES)
        )

        def packed(
            lanes: _TensorOrRef,
            tokens: _TensorOrRef = None,
        ) -> _TensorOrRef:
            """Bitcast packed 32-bit page words to the cache dtype."""
            np.testing.assert_(tokens is None, "token ranges need native pages")
            return tpu.bitcast(words[:, lanes], self.cache.dtype)

        return packed

    def _needs_split_tail_range(self) -> bool:
        """Return whether ``_attend_range`` peels partial last steps per row."""
        if "attend" in self.skip or "pagecompute" in self.skip:
            return False
        if self.folds > 1:
            return True
        dp_match = self.heads == HEADS_PER_RANK or (
            self.group == DP2_GROUP
            and self.heads == DP2_HEADS
            and self.batch == B32_BATCH
        )
        span_match = self.span == RING_SLOTS_DEFAULT or (
            _config().mla_decode_ktile == WIDE_KTILE and self.span == SPAN
        )
        return (
            self.sequence_rows == 1
            and dp_match
            and self.batch in (B32_BATCH, B64_BATCH)
            and self.ws.gathered.dtype == jnp.uint8
            and span_match
        )

    def _attend_tail_step(
        self,
        full_end: _TensorOrRef,
        held_row: _TensorOrRef,
        tail_pages: int,
    ) -> None:
        """Fold a static-width ``tail_pages`` remainder step for ``held_row``."""
        self._wait_step(full_end)

        @pl.when(full_end + self.buffers - 1 < self.total)
        def prefetch() -> None:
            """Issue the prefetch DMA for the step after ``full_end``."""
            self._start_step(full_end + self.buffers - 1)

        _, visible, load = self._step_pages(full_end)
        tail_tokens = tail_pages * self.page_tokens
        score = self._scores(load, visible, held_row, tail_tokens=tail_tokens)
        self._fold_scores(score, load, held_row, tail_tokens=tail_tokens)

    def _attend_split_rows(self, chunk: int, base: _TensorOrRef) -> None:
        """Run ``chunk`` rows one by one, peeling each row's partial tail step."""

        @pl.loop(0, chunk)
        def row_loop(row_offset: _TensorOrRef) -> None:
            """Attend full steps and any partial tail step for ``base + row_offset``."""
            row = base + row_offset
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
                def tail(*, tail_pages: int = tail_pages) -> None:
                    """Fold the ``tail_pages`` partial step."""
                    self._attend_tail_step(full_end, row - base, tail_pages)

    def _attend_range(
        self,
        first: _TensorOrRef,
        end: _TensorOrRef,
        base: _TensorOrRef,
    ) -> None:
        """Attend page steps ``first .. end - 1`` for chunk starting at ``base``."""
        if not self._needs_split_tail_range():
            self._attend_pipeline(first, end, base)
            return
        chunk = chunk_layout(self.batch, self.heads, chunk_rows=self.chunk_rows)[0]
        if self.folds > 1:
            self._attend_split_rows(chunk, base)
            return
        remainder = jnp.int32(0)
        for row_offset in range(chunk):
            remainder = remainder | (
                self.ws.lookup_counts[base + row_offset] % self.span
            )

        @pl.when(remainder == 0)
        def whole() -> None:
            """Run the whole chunk through one pipelined loop when all rows align."""
            self._attend_pipeline(first, end, base)

        pl.when(remainder != 0)(lambda: self._attend_split_rows(chunk, base))

    def _staged_tail(
        self,
        step: _TensorOrRef,
        row: _TensorOrRef,
        count: _TensorOrRef,
        base: _TensorOrRef,
    ) -> None:
        """Fold a staged row's partial last step using static part shapes."""
        self._wait_step(step)

        @pl.when(step + self.buffers - 1 < self.total)
        def prefetch() -> None:
            """Prefetch the page step ``buffers - 1`` ahead."""
            self._start_step(step + self.buffers - 1)

        held = row - base
        slot = step % self.buffers
        local_step = count // self.span
        pages = count % self.span
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

        def visible(first_page: _TensorOrRef, tokens: int) -> _TensorOrRef:
            """Return visibility mask for ``tokens`` starting at ``first_page``."""
            offset = jax.lax.broadcasted_iota(jnp.int32, (1, tokens), 1)
            global_page = (
                local_step * self.span + first_page + offset // self.page_tokens
            ) * self.group + self.owner
            return global_page * self.page_tokens + offset % self.page_tokens < limit

        @pl.when(pages >= part_pages)
        def whole() -> None:
            """Fold the first full ``fold_tokens`` part when present."""
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
                pl.multiple_of(first * self.page_tokens, self.page_tokens),
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
        first: _TensorOrRef,
        end: _TensorOrRef,
        base: _TensorOrRef,
        *,
        peel: bool = False,
    ) -> None:
        """Pipeline QK and softmax/PV across steps ``first .. end - 1``."""
        if self.sequence_rows > 1 or "pagecompute" in self.skip:
            pl.loop(first, end)(lambda step: self._attend_step(step, base=base))
            return

        def scores(
            step: _TensorOrRef,
        ) -> tuple[_TensorOrRef, _TensorOrRef]:
            """Compute QK scores for ``step`` and return ``(score, row)``."""
            row, visible, load = self._step_pages(step)
            return self._scores(load, visible, row - base), row

        @pl.when(first < end)
        def attend() -> None:
            """Run the software-pipelined page loop when the range is non-empty."""
            self._wait_step(first)

            @pl.when(first + self.buffers - 1 < self.total)
            def prefetch_first() -> None:
                """Prefetch the step corresponding to ``first``."""
                self._start_step(first + self.buffers - 1)

            def fold(
                step: _TensorOrRef,
                carry: tuple[_TensorOrRef, _TensorOrRef],
            ) -> tuple[_TensorOrRef, _TensorOrRef]:
                """Issue step ``step + 1``'s QK before folding step ``step``."""
                score, row = carry
                more = step + 1 < end
                if peel:
                    self._wait_step(step + 1)
                    upcoming = scores(step + 1)
                else:

                    @pl.when(more)
                    def wait_next() -> None:
                        """Wait for the next step's page DMA when ``more`` holds."""
                        self._wait_step(step + 1)

                    upcoming = scores(jnp.minimum(step + 1, end - 1))
                self._fold_scores(score, self._pages(step % self.buffers), row - base)

                @pl.when((True if peel else more) & (step + self.buffers < self.total))
                def prefetch() -> None:
                    """Refill the buffer freed by ``step``."""
                    self._start_step(step + self.buffers)

                return upcoming

            carry = jax.lax.fori_loop(first, end - int(peel), fold, scores(first))
            if peel:
                score, row = carry
                self._fold_scores(
                    score, self._pages((end - 1) % self.buffers), row - base
                )

    def _store_query(self, held: _TensorOrRef, query: _TensorOrRef) -> None:
        """Quantize or cast ``query`` once and store into ``ws.queries[held]``."""
        if self.fp8:
            maximum = jnp.max(jnp.abs(query), axis=1, keepdims=True)
            scale = jnp.where(maximum > 0, maximum / FP8_MAX_FINITE, 1.0)
            self.ws.queries[held] = (query / scale).astype(jnp.float8_e4m3fn)
            self.ws.query_scale[held] = scale
        else:
            self.ws.queries[held] = query.astype(jnp.bfloat16)

    def _group_scores(
        self,
        load: Callable[..., _TensorOrRef],
        visible: _TensorOrRef,
        held: _TensorOrRef,
        count: int,
    ) -> _TensorOrRef:
        """Compute grouped QK scores for ``count`` adjacent queries from ``held``."""
        query = jnp.concatenate(
            [self.ws.queries[held + idx] for idx in range(count)], axis=0
        )
        scales = jnp.concatenate(
            [self.ws.query_scale[held + idx] for idx in range(count)], axis=0
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
        score: _TensorOrRef,
        load: Callable[..., _TensorOrRef],
        held: _TensorOrRef,
        count: int,
    ) -> None:
        """Fold grouped PV back into each query's online softmax state."""
        ws = self.ws
        previous = jnp.concatenate(
            [ws.maximum[held + idx] for idx in range(count)], axis=0
        )
        new_maximum = jnp.maximum(previous, jnp.max(score, axis=1, keepdims=True))
        scale = jnp.exp(previous - new_maximum)
        probabilities = jnp.exp(score - new_maximum)
        totals = jnp.sum(probabilities, axis=1, keepdims=True)
        high = (probabilities * FP8_PV_SCALE).astype(jnp.float8_e4m3fn)
        values = load(pl.ds(0, LATENT))
        weighted = (
            jax.lax.dot_general(
                high,
                values,
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            * FP8_INV_PV_SCALE
        )
        for idx in range(count):
            rows = slice(idx * self.tile, (idx + 1) * self.tile)
            ws.denominator[held + idx] = (
                ws.denominator[held + idx] * scale[rows] + totals[rows]
            )
            ws.maximum[held + idx] = new_maximum[rows]
            ws.accumulator[held + idx] = (
                ws.accumulator[held + idx] * scale[rows] + weighted[rows]
            )

    def _fold_pages(
        self,
        load: Callable[..., _TensorOrRef],
        visible: _TensorOrRef,
        held: _TensorOrRef,
    ) -> None:
        """Compute and fold one step's scores into state row ``held``."""
        self._fold_scores(self._scores(load, visible, held), load, held)

    def _scores(
        self,
        load: Callable[..., _TensorOrRef],
        visible: _TensorOrRef,
        held: _TensorOrRef,
        *,
        tail_tokens: int = 0,
    ) -> _TensorOrRef:
        """Return state row ``held``'s masked scores for a step's pages."""
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
        score: _TensorOrRef,
        load: Callable[..., _TensorOrRef],
        held: _TensorOrRef,
        *,
        tail_tokens: int = 0,
    ) -> None:
        """Fold ``score`` into state row ``held``'s running softmax."""
        if self.folds == 1:
            self._fold_part(score, lambda: load(pl.ds(0, LATENT)), held, tail_tokens)
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
        score: _TensorOrRef,
        values: Callable[[], _TensorOrRef],
        held: _TensorOrRef,
        tail_tokens: int = 0,
    ) -> None:
        """Fold one score slice and value callback into ``held``."""
        ws = self.ws
        previous = ws.maximum[held]
        new_maximum = jnp.maximum(previous, jnp.max(score, axis=1, keepdims=True))
        scale = jnp.exp(previous - new_maximum)
        probabilities = jnp.exp(score - new_maximum)
        ws.denominator[held] = ws.denominator[held] * scale + jnp.sum(
            probabilities, axis=1, keepdims=True
        )
        ws.maximum[held] = new_maximum
        high = (
            (probabilities * FP8_PV_SCALE).astype(jnp.float8_e4m3fn)
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
            weighted = weighted * FP8_INV_PV_SCALE
        ws.accumulator[held] = ws.accumulator[held] * scale + weighted

    def _clean_step(
        self,
        slot: _TensorOrRef,
        row: _TensorOrRef,
        local_step: _TensorOrRef,
        tokens: int | None = None,
        end: _TensorOrRef = None,
    ) -> _TensorOrRef:
        """Zero stale padding lanes and out-of-bounds tokens in buffer ``slot``."""
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
            self.batch in (B16_BATCH, B32_BATCH, B48_BATCH, B64_BATCH)
            or self.heads != HEADS_PER_RANK
            or (self.batch, self.sequence_rows) == (MAX_ROWS, 1)
        ):
            pad_shape = (rows // self.packing, LANES - LATENT)
            pad_lane = jax.lax.broadcasted_iota(jnp.int32, pad_shape, 1)
            tpu.store(
                words.at[:, pl.ds(LATENT, LANES - LATENT)]
                if tokens is None
                else words.at[
                    pl.ds(0, rows // self.packing),
                    pl.ds(LATENT, LANES - LATENT),
                ],
                jnp.zeros(pad_shape, jnp.uint32),
                mask=pad_lane >= ROPE_LANES,
            )
        else:
            pad = words[:, pl.ds(LATENT, LANES - LATENT)]
            pad_lane = jax.lax.broadcasted_iota(jnp.int32, pad.shape, 1)
            words[:, pl.ds(LATENT, LANES - LATENT)] = jnp.where(
                pad_lane < ROPE_LANES, pad, jnp.zeros((), pad.dtype)
            )
        limit = self._limit(row)
        cut: _TensorOrRef = 0
        for page_idx in range(rows // self.page_tokens):
            first_token = (
                (local_step * self.span + page_idx) * self.group + self.owner
            ) * self.page_tokens
            cut = cut + jnp.clip(limit - first_token, 0, self.page_tokens)
        tile_tokens = SUBLANE_ROWS * self.packing
        tile = cut // tile_tokens

        @pl.when(tile < rows // tile_tokens)
        def boundary() -> None:
            """Zero tokens at or past ``cut`` inside the boundary 8-word tile."""
            at = pl.ds(pl.multiple_of(tile * SUBLANE_ROWS, SUBLANE_ROWS), SUBLANE_ROWS)
            block = tpu.bitcast(words[at, :], self.cache.dtype)
            offset = tile * tile_tokens + jax.lax.broadcasted_iota(
                jnp.int32, block.shape, 0
            )
            words[at, :] = tpu.bitcast(
                jnp.where(offset < cut, block, jnp.zeros((), block.dtype)),
                jnp.uint32,
            )

        @pl.loop(tile + 1, (rows if end is None else end) // tile_tokens)
        def invisible(tile_idx: _TensorOrRef) -> None:
            """Zero whole 8-word tiles past the boundary tile."""
            words[
                pl.ds(
                    pl.multiple_of(tile_idx * SUBLANE_ROWS, SUBLANE_ROWS),
                    SUBLANE_ROWS,
                ),
                :,
            ] = jnp.zeros((SUBLANE_ROWS, LANES), jnp.uint32)

        return words

    def _pack_partials(self, first: int, end: int) -> None:
        """Normalize state rows ``first..end`` and write peer send packets."""
        ws = self.ws
        group = self.group
        lses: dict[int, tuple[_TensorOrRef, _TensorOrRef]] = {}
        for held in range(first, end):
            total_weight = ws.denominator[held]
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
                jnp.int32, (lse.shape[0], LANES - LATENT), 1
            )
            lse_lanes = jnp.where(
                lane == 0, lse_high, jnp.where(lane == 1, lse - lse_high, 0.0)
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
        lses: dict[int, tuple[_TensorOrRef, _TensorOrRef]],
        first: int,
        end: int,
    ) -> None:
        """Pack each sequence's log-sum-exps into its trailing padding row."""
        ws = self.ws
        group = self.group
        slots = self.slots
        rows = self.sequence_rows
        if (
            slots < MIN_LSE_SLOTS
            or slots * (rows - 1) > LSE_VECTOR_LANES
            or (end - first) % rows
        ):
            msg = (
                "lse_in_padding takes two or more head slots, one lane each, "
                "and whole sequences"
            )
            raise ValueError(msg)
        head_row_count = int(lses[first][0].shape[0])
        slot_row = jax.lax.broadcasted_iota(
            jnp.int32, (head_row_count, LSE_VECTOR_LANES), 0
        )
        lane = jax.lax.broadcasted_iota(
            jnp.int32, (head_row_count, LSE_VECTOR_LANES), 1
        )
        index = jax.lax.broadcasted_iota(jnp.int32, (group, head_row_count), 0)
        if self._lane_major:
            hosts = group // HOST_LANES
            index = (index % hosts) * HOST_LANES + index // hosts
        owner = (
            jax.lax.broadcasted_iota(jnp.int32, (group, head_row_count), 1) // slots
            == index
        ).astype(jnp.bfloat16)
        for start in range(first, end, rows):
            for part in range(LSE_PARTS):
                spread = jnp.zeros((head_row_count, LSE_VECTOR_LANES), jnp.float32)
                for row_offset in range(rows - 1):
                    value = lses[start + row_offset][part]
                    spread = jnp.where(
                        lane == slots * row_offset + slot_row % slots,
                        value,
                        spread,
                    )
                vectors = jax.lax.dot_general(
                    owner,
                    spread.astype(jnp.bfloat16),
                    (((1,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
                row = slots * (start + rows - 1) + part
                ws.send[:, pl.ds(row, 1), pl.ds(0, LSE_VECTOR_LANES)] = vectors[
                    :, None, :
                ].astype(jnp.bfloat16)

    def _combine_lanes(self) -> pl.Slice:
        """Return the lane slice transported during the partial combine."""
        return pl.ds(0, LATENT if self.lse_in_padding else LANES)

    def _send_index(self, peer: int) -> int:
        """Return the send packet index for ``peer``."""
        if not self._lane_major:
            return peer
        hosts = self.group // HOST_LANES
        return (peer % HOST_LANES) * hosts + peer // HOST_LANES

    def _scope(self, name: str) -> contextlib.AbstractContextManager[_TensorOrRef]:
        """Return a named JAX trace scope when ``self.trace`` is set."""
        return (
            jax.named_scope(self.trace + name)
            if self.trace
            else contextlib.nullcontext()
        )

    def prime(self) -> None:
        """Issue the initial ``buffers - 1`` page DMAs before projections."""
        for step in range(self.buffers - 1):

            @pl.when(step < self.total)
            def prime_step(*, step: int = step) -> None:
                """Start page DMA for initial ``step``."""
                self._start_step(step)

    def row_groups(self) -> list[tuple[int, int]]:
        """Return ``[(first_row, end_row)]`` partitioning the batch into groups."""
        groups = self.groups
        bounds = [self.batch * group_idx // groups for group_idx in range(groups + 1)]
        return [
            (bounds[group_idx], bounds[group_idx + 1]) for group_idx in range(groups)
        ]

    def _rows_of_group(
        self,
        groups: list[tuple[int, int]],
        group_idx: int,
    ) -> pl.Slice:
        """Return the packet row slice for row group ``group_idx``."""
        first, end = groups[group_idx]
        return pl.ds(self.slots * first, self.slots * (end - first))

    def _init_unchunked_rows(
        self,
        queries: _TensorOrRef,
        new_latents: _TensorOrRef,
        parity: _TensorOrRef,
        groups: list[tuple[int, int]],
    ) -> None:
        """Populate ``own_rows`` and ``new_rows`` and issue query all-gather."""
        ws = self.ws
        group = self.group
        member = self.member
        base = self.base
        ws.own_rows[...] = jnp.zeros(ws.own_rows.shape, jnp.float32)
        ws.new_rows[...] = jnp.zeros(ws.new_rows.shape, jnp.float32)
        for batch_idx in range(self.batch):
            ws.own_rows[pl.ds(self.slots * batch_idx, self.heads), :] = queries[
                batch_idx
            ]
            ws.new_rows[pl.ds(self.slots * batch_idx, self.heads), :] = (
                jnp.broadcast_to(new_latents[batch_idx][None, :], (self.heads, LANES))
            )
        ws.gathered[parity, member] = (
            _pack_fp8_query(ws.own_rows[...])
            if ws.gathered.dtype == jnp.uint8
            else ws.own_rows[...].astype(jnp.bfloat16)
        )
        for group_idx in range(len(groups)):
            for peer in range(0 if "gather" in self.skip else group):

                @pl.when(peer != member)
                def send_query(
                    *,
                    peer: int = peer,
                    group_idx: int = group_idx,
                ) -> None:
                    """Send this rank's queries for ``group_idx`` to ``peer``."""
                    row_slice = self._rows_of_group(groups, group_idx)
                    tpu.make_async_remote_copy(
                        ws.gathered.at[parity, member, row_slice],
                        ws.gathered.at[parity, member, row_slice],
                        ws.gather_semaphores.at[0, parity, group_idx],
                        ws.gather_semaphores.at[1, parity, group_idx],
                        device_id=(base + peer,),
                        device_id_type=pl.DeviceIdType.MESH,
                    ).start()

    def _attend_unchunked_group(
        self,
        parity: _TensorOrRef,
        groups: list[tuple[int, int]],
        group_idx: int,
    ) -> None:
        """Await gathered queries, attend pages, and send partials for ``group_idx``."""
        ws = self.ws
        group = self.group
        member = self.member
        base = self.base
        heads = group * self.slots
        first, end = groups[group_idx]
        row_slice = self._rows_of_group(groups, group_idx)
        others = ws.gathered.at[parity, pl.ds(0, group - 1), row_slice]
        if "gather" not in self.skip:
            with self._scope("gather_wait"):
                tpu.make_async_copy(
                    others,
                    others,
                    ws.gather_semaphores.at[0, parity, group_idx],
                ).wait()
                tpu.make_async_copy(
                    others,
                    others,
                    ws.gather_semaphores.at[1, parity, group_idx],
                ).wait()
        local_scope = self._scope("local")
        local_scope.__enter__()
        for batch_idx in range(first, end):
            gathered = ws.gathered[
                parity, :, pl.ds(self.slots * batch_idx, self.slots), :
            ].reshape(heads, LANES)
            padding = ((0, self.tile - heads), (0, 0))
            if ws.gathered.dtype == jnp.uint8:
                query, scale = _unpack_fp8_query(gathered)
                ws.queries[batch_idx] = jnp.pad(query, padding)
                ws.query_scale[batch_idx] = jnp.pad(scale, padding, constant_values=1)
            else:
                self._store_query(
                    batch_idx, jnp.pad(gathered.astype(jnp.float32), padding)
                )
        if self.heads != HEADS_PER_RANK:
            self._attend_pipeline(
                self.starts[first],
                self.starts[first] if "attend" in self.skip else self.starts[end],
                0,
            )
        else:

            @pl.loop(
                self.starts[first],
                self.starts[first] if "attend" in self.skip else self.starts[end],
            )
            def attend_loop(step: _TensorOrRef) -> None:
                """Attend one local page step for row group ``group_idx``."""
                self._attend_step(step)

        self._pack_partials(first, end)
        for peer in range(0 if "combine" in self.skip else group):

            @pl.when(peer != member)
            def send_partial(
                *,
                peer: int = peer,
                group_idx: int = group_idx,
            ) -> None:
                """Send normalized partial outputs for ``group_idx`` to ``peer``."""
                slice_g = self._rows_of_group(groups, group_idx)
                tpu.make_async_remote_copy(
                    ws.send.at[peer, slice_g, self._combine_lanes()],
                    ws.receive.at[parity, member, slice_g, self._combine_lanes()],
                    ws.combine_semaphores.at[0, parity, group_idx],
                    ws.combine_semaphores.at[1, parity, group_idx],
                    device_id=(base + peer,),
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()

        ws.receive[parity, member, row_slice] = ws.send[member, row_slice]
        local_scope.__exit__(None, None, None)

    def _merge_unchunked(
        self,
        parity: _TensorOrRef,
        new_latents: _TensorOrRef,
        keys: _TensorOrRef,
        key_valid: _TensorOrRef,
    ) -> _TensorOrRef:
        """Merge received partials with new tokens for an unchunked call."""
        ws = self.ws
        group = self.group
        batch = self.batch
        partials = ws.receive[parity].astype(jnp.float32)
        lse = (
            partials[:, :, LATENT : LATENT + 1]
            + partials[:, :, LATENT + 1 : LATENT + 2]
        )
        if keys is not None:
            np.testing.assert_(key_valid is not None, "key_valid is required")
            sequences = batch // self.sequence_rows
            rows = self.slots * self.sequence_rows
            merged = jnp.concatenate(
                [
                    _merge_received_keys(
                        ws.receive.at[parity, :, pl.ds(rows * seq_idx, rows)],
                        ws.own_rows[pl.ds(rows * seq_idx, rows), :],
                        keys[seq_idx],
                        key_valid[seq_idx],
                        KeyMergeSpec(
                            group=group,
                            lse_rows=self.slots if self.lse_in_padding else 0,
                        ),
                    )
                    for seq_idx in range(sequences)
                ],
                axis=0,
            )
        elif self.sequence_rows == 1 or self.cache_verify_rows:
            new_score = jnp.sum(
                ws.own_rows[...] * ws.new_rows[...], axis=1, keepdims=True
            )
            top = jnp.maximum(jnp.max(lse, axis=0), new_score)
            weights = jnp.exp(lse - top[None])
            new_weight = jnp.exp(new_score - top)
            merged = (
                jnp.sum(weights * partials[:, :, :LATENT], axis=0)
                + new_weight * ws.new_rows[:, :LATENT]
            ) / (jnp.sum(weights, axis=0) + new_weight)
        else:
            latents = new_latents
            scores = jax.lax.dot_general(
                ws.own_rows[...],
                latents,
                (((1,), (1,)), ((), ())),
                precision=jax.lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
            )
            slot_row = (
                jax.lax.broadcasted_iota(jnp.int32, scores.shape, 0) // self.slots
            )
            key_row = jax.lax.broadcasted_iota(jnp.int32, scores.shape, 1)
            causal = (
                key_row // self.sequence_rows == slot_row // self.sequence_rows
            ) & (key_row <= slot_row)
            scores = jnp.where(causal, scores, -jnp.inf)
            top = jnp.maximum(
                jnp.max(lse, axis=0), jnp.max(scores, axis=1, keepdims=True)
            )
            weights = jnp.exp(lse - top[None])
            new_weights = jnp.exp(scores - top)
            new_values = jax.lax.dot_general(
                new_weights,
                latents[:, :LATENT],
                (((1,), (0,)), ((), ())),
                precision=jax.lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
            )
            merged = (
                jnp.sum(weights * partials[:, :, :LATENT], axis=0) + new_values
            ) / (jnp.sum(weights, axis=0) + jnp.sum(new_weights, axis=1, keepdims=True))
        return jnp.stack(
            [
                merged[self.slots * batch_idx : self.slots * batch_idx + self.heads]
                for batch_idx in range(batch)
            ]
        )

    def attend(
        self,
        queries: _TensorOrRef,
        new_latents: _TensorOrRef,
        parity: _TensorOrRef,
        between: Callable[[], None] | None = None,
        **attend_kwargs: Unpack[_AttendExplicitKeysKwargs],
    ) -> _TensorOrRef:
        """Run DCP query all-gather, local page attention, combine, and merge."""
        keys = attend_kwargs.get("keys")
        key_valid = attend_kwargs.get("key_valid")
        if keys is not None and self.sequence_rows == 1:
            msg = "explicit new keys need multi-row sequences"
            raise ValueError(msg)
        if chunked(self.batch, self.force_chunked):
            return self._attend_chunked(
                queries, new_latents, between, keys=keys, key_valid=key_valid
            )
        ws = self.ws
        group = self.group
        groups = self.row_groups()
        self._init_unchunked_rows(queries, new_latents, parity, groups)
        if between is not None:
            between()
        ws.maximum[...] = jnp.full(ws.maximum.shape, -jnp.inf, jnp.float32)
        ws.denominator[...] = jnp.zeros(ws.denominator.shape, jnp.float32)
        ws.accumulator[...] = jnp.zeros(ws.accumulator.shape, jnp.float32)
        for group_idx in range(len(groups)):
            self._attend_unchunked_group(parity, groups, group_idx)
        with self._scope("combine_wait"):
            for group_idx in range(len(groups)):
                row_slice = self._rows_of_group(groups, group_idx)
                others = ws.receive.at[
                    parity,
                    pl.ds(0, group - 1),
                    row_slice,
                    self._combine_lanes(),
                ]
                if "combine" not in self.skip:
                    tpu.make_async_copy(
                        others,
                        others,
                        ws.combine_semaphores.at[0, parity, group_idx],
                    ).wait()
                    tpu.make_async_copy(
                        others,
                        others,
                        ws.combine_semaphores.at[1, parity, group_idx],
                    ).wait()
        merge_scope = self._scope("merge")
        merge_scope.__enter__()
        result = self._merge_unchunked(parity, new_latents, keys, key_valid)
        merge_scope.__exit__(None, None, None)
        return result

    def _chunk_ring_slot(
        self,
        chunk_idx: _TensorOrRef,
        count: int,
        ring_slots: int,
    ) -> _TensorOrRef:
        """Return the ring slot index for ``chunk_idx``."""
        layer = self.cache.layer if self.cache.layer is not None else 0
        return (layer * count + chunk_idx) % ring_slots

    def _send_chunk_queries(
        self,
        chunk_idx: _TensorOrRef,
        rows: int,
        count: int,
        ring_slots: int,
    ) -> None:
        """Pack and send query packet for ``chunk_idx`` to all group peers."""
        ws = self.ws
        group = self.group
        member = self.member
        base = self.base
        slot = self._chunk_ring_slot(chunk_idx, count, ring_slots)
        own = ws.own_rows[chunk_idx, :rows, :]
        ws.gathered[slot, member] = (
            _pack_fp8_query(own)
            if ws.gathered.dtype == jnp.uint8
            else own.astype(jnp.bfloat16)
        )
        for peer in range(0 if "gather" in self.skip else group):

            @pl.when(peer != member)
            def send(*, peer: int = peer) -> None:
                """Start remote query DMA to ``peer``."""
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
        chunk_idx: _TensorOrRef,
        count: int,
        ring_slots: int,
    ) -> None:
        """Wait for query all-gather of ``chunk_idx`` to complete."""
        if "gather" in self.skip:
            return
        slot = self._chunk_ring_slot(chunk_idx, count, ring_slots)
        others = self.ws.gathered.at[slot, pl.ds(0, self.group - 1)]
        tpu.make_async_copy(
            others, others, self.ws.gather_semaphores.at[0, slot]
        ).wait()
        tpu.make_async_copy(
            others, others, self.ws.gather_semaphores.at[1, slot]
        ).wait()

    def _send_chunk_partials(
        self,
        chunk_idx: _TensorOrRef,
        loop_layout: ChunkLoopLayout,
    ) -> None:
        """Send packed partial outputs of ``chunk_idx`` to group peers or host-mates."""
        ws = self.ws
        group = self.group
        member = self.member
        base = self.base
        count = loop_layout.count
        ring_slots = loop_layout.ring_slots
        slot = self._chunk_ring_slot(chunk_idx, count, ring_slots)
        if self.host_combine is not None:
            hosts = group // HOST_LANES
            host = loop_layout.host
            lane = loop_layout.lane
            lanes512 = self._combine_lanes()
            for host_step in range(1, HOST_LANES):
                mate = (lane + host_step) % HOST_LANES
                tpu.make_async_remote_copy(
                    ws.send.at[pl.ds(mate * hosts, hosts), :, lanes512],
                    ws.receive.at[slot, pl.ds(lane * hosts, hosts), :, lanes512],
                    ws.combine_semaphores.at[0, slot],
                    ws.combine_semaphores.at[1, slot],
                    device_id=(base + host * HOST_LANES + mate,),
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()
            ws.receive[slot, pl.ds(lane * hosts, hosts)] = ws.send[
                pl.ds(lane * hosts, hosts)
            ]
            return
        for peer in range(0 if "combine" in self.skip else group):

            @pl.when(peer != member)
            def send(*, peer: int = peer) -> None:
                """Start remote partial combine DMA to ``peer``."""
                tpu.make_async_remote_copy(
                    ws.send.at[peer, :, self._combine_lanes()],
                    ws.receive.at[slot, member, :, self._combine_lanes()],
                    ws.combine_semaphores.at[0, slot],
                    ws.combine_semaphores.at[1, slot],
                    device_id=(base + peer,),
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()

        ws.receive[slot, member] = ws.send[member]

    def _wait_chunk_combine(
        self,
        chunk_idx: _TensorOrRef,
        count: int,
        ring_slots: int,
        which: int,
    ) -> None:
        """Wait for combine DMA direction ``which`` (0=send, 1=recv) on chunk."""
        if "combine" in self.skip:
            return
        hier = self.host_combine is not None
        peers_bytes = (
            (HOST_LANES - 1) * (self.group // HOST_LANES) if hier else self.group - 1
        )
        # The ring slot is computed at each use, as in the port kernel.
        others = self.ws.receive.at[
            self._chunk_ring_slot(chunk_idx, count, ring_slots),
            pl.ds(0, peers_bytes),
            :,
            self._combine_lanes(),
        ]
        tpu.make_async_copy(
            others,
            others,
            self.ws.combine_semaphores.at[
                which, self._chunk_ring_slot(chunk_idx, count, ring_slots)
            ],
        ).wait()

    def _host_merge_chunk(
        self,
        chunk_idx: _TensorOrRef,
        loop_layout: ChunkLoopLayout,
    ) -> None:
        """Run stage two of ``host_combine`` for ``chunk_idx``."""
        ws = self.ws
        group = self.group
        base = self.base
        rows = loop_layout.rows
        stage2_recv, stage2_send, stage2_sems = self.host_combine
        hosts = group // HOST_LANES
        host = loop_layout.host
        lane = loop_layout.lane
        lanes512 = self._combine_lanes()
        slot = self._chunk_ring_slot(
            chunk_idx, loop_layout.count, loop_layout.ring_slots
        )
        pad = rows - self.slots
        tails = ws.receive[
            slot, :, pl.ds(pad, LSE_PARTS), pl.ds(0, LSE_VECTOR_LANES)
        ].astype(jnp.float32)
        vectors = tails[:, 0:1, :] + tails[:, 1:2, :]
        packet_row = jax.lax.broadcasted_iota(
            jnp.int32, (group, rows, LSE_VECTOR_LANES), 1
        )
        packet_lane = jax.lax.broadcasted_iota(
            jnp.int32, (group, rows, LSE_VECTOR_LANES), 2
        )
        lses = jnp.sum(
            jnp.where(packet_lane == packet_row, vectors, 0.0),
            axis=2,
            keepdims=True,
        )
        lses = jnp.where(packet_row[:, :, :1] < pad, lses, NEG)
        for owner_host in range(hosts):
            sources = [source * hosts + owner_host for source in range(HOST_LANES)]
            top = lses[sources[0]]
            for index in sources[1:]:
                top = jnp.maximum(top, lses[index])
            total = jnp.zeros_like(top)
            accumulator: _TensorOrRef = None
            for index in sources:
                weight = jnp.exp(lses[index] - top)
                total = total + weight
                term = weight * ws.receive[slot, index, :, pl.ds(0, LATENT)].astype(
                    jnp.float32
                )
                accumulator = term if accumulator is None else accumulator + term
            stage2_send[owner_host] = _lse_packet(
                accumulator / total, top + jnp.log(total), self.slots
            )
        for host_step in range(1, hosts):
            owner_host = (host + host_step) % hosts
            tpu.make_async_remote_copy(
                stage2_send.at[owner_host, :, lanes512],
                stage2_recv.at[slot, host, :, lanes512],
                stage2_sems.at[0, slot],
                stage2_sems.at[1, slot],
                device_id=(base + owner_host * HOST_LANES + lane,),
                device_id_type=pl.DeviceIdType.MESH,
            ).start()
        stage2_recv[slot, host] = stage2_send[host]

    def _stage2_wait_chunk(
        self,
        chunk_idx: _TensorOrRef,
        count: int,
        ring_slots: int,
        which: int,
    ) -> None:
        """Wait on stage-2 ``host_combine`` semaphore ``which`` for ``chunk_idx``."""
        stage2_recv, _, stage2_sems = self.host_combine
        hosts = self.group // HOST_LANES
        # The ring slot is computed at each use, as in the port kernel.
        others = stage2_recv.at[
            self._chunk_ring_slot(chunk_idx, count, ring_slots),
            pl.ds(0, hosts - 1),
            :,
            self._combine_lanes(),
        ]
        tpu.make_async_copy(
            others,
            others,
            stage2_sems.at[which, self._chunk_ring_slot(chunk_idx, count, ring_slots)],
        ).wait()

    def _merge_chunk(
        self,
        chunk_idx: _TensorOrRef,
        loop_layout: ChunkLoopLayout,
    ) -> None:
        """Merge received partials of ``chunk_idx`` into ``ws.merged[chunk_idx]``."""
        if "merge" in self.skip:
            return
        chunk = loop_layout.chunk
        count = loop_layout.count
        ring_slots = loop_layout.ring_slots
        keys = loop_layout.keys
        key_valid = loop_layout.key_valid
        ws = self.ws
        group = self.group
        batch = self.batch
        slots = self.slots
        rows = slots * chunk
        if self.sequence_rows == 1 or self.cache_verify_rows:
            unroll = (
                PEER_UNROLL_WIDE
                if self.heads == DP2_HEADS
                and (batch == B64_BATCH or (batch == B48_BATCH and self.fp8))
                else (
                    PEER_UNROLL_WIDE
                    if self.heads == DP4_HEADS
                    and (batch == B32_BATCH or (batch == B24_BATCH and self.fp8))
                    else 1
                )
            )
            ws.merged[chunk_idx, :rows, :] = _merge_received(
                ws.receive.at[self._chunk_ring_slot(chunk_idx, count, ring_slots)],
                ws.own_rows[chunk_idx, :rows, :],
                ws.new_rows[chunk_idx, :rows, :],
                group,
                unroll=unroll,
            )
            return
        if keys is not None:
            np.testing.assert_(key_valid is not None, "key_valid is required")
            hier = self.host_combine is not None
            hosts = group // HOST_LANES
            slot = self._chunk_ring_slot(chunk_idx, count, ring_slots)
            received, sources = (
                (self.host_combine[0].at[slot], hosts)
                if hier
                else (ws.receive.at[slot], group)
            )
            ws.merged[chunk_idx, :rows, :] = _merge_received_keys(
                received,
                ws.own_rows[chunk_idx, :rows, :],
                keys[chunk_idx],
                key_valid[chunk_idx],
                KeyMergeSpec(
                    group=sources,
                    lse_rows=self.slots if self.lse_in_padding else 0,
                ),
            )
            return
        latents = jnp.concatenate(
            [
                ws.new_rows[chunk_idx, pl.ds(slots * row_offset, 1), :]
                for row_offset in range(chunk)
            ],
            axis=0,
        )
        ws.merged[chunk_idx, :rows, :] = _merge_received_sequences(
            ws.receive.at[self._chunk_ring_slot(chunk_idx, count, ring_slots)],
            ws.own_rows[chunk_idx, :rows, :],
            latents,
            SequenceMergeSpec(
                group=group,
                slots=slots,
                sequence_rows=self.sequence_rows,
                unroll=(
                    PEER_UNROLL_WIDE
                    if self.fp8
                    and _grouped_spec(batch, group, self.heads, self.sequence_rows)
                    else 1
                ),
            ),
        )

    def _build_chunk_queries(
        self,
        chunk_idx: _TensorOrRef,
        chunk: int,
        count: int,
        ring_slots: int,
    ) -> None:
        """Unpack or store gathered queries for ``chunk_idx`` into ``ws.queries``."""
        ws = self.ws
        slots = self.slots
        heads = self.group * slots
        for row_offset in range(0 if "build" in self.skip else chunk):
            gathered = ws.gathered[
                self._chunk_ring_slot(chunk_idx, count, ring_slots),
                :,
                pl.ds(slots * row_offset, slots),
                :,
            ].reshape(heads, LANES)
            padding = ((0, self.tile - heads), (0, 0))
            if ws.gathered.dtype == jnp.uint8:
                query, scale = _unpack_fp8_query(gathered)
                ws.queries[row_offset] = jnp.pad(query, padding)
                ws.query_scale[row_offset] = jnp.pad(scale, padding, constant_values=1)
            else:
                self._store_query(
                    row_offset,
                    jnp.pad(gathered.astype(jnp.float32), padding),
                )

    def _finish_chunked_tail(
        self,
        loop_layout: ChunkLoopLayout,
    ) -> _TensorOrRef:
        """Drain remaining combines and merges after the rolled chunk loop."""
        ws = self.ws
        batch = self.batch
        slots = self.slots
        hier = self.host_combine is not None
        chunk = loop_layout.chunk
        count = loop_layout.count
        ring_slots = loop_layout.ring_slots
        lookahead = loop_layout.lookahead
        if hier:
            with self._scope("host_wait"):
                self._wait_chunk_combine(count - 1, count, ring_slots, 1)
                if count >= RING_SLOTS_DEFAULT:
                    self._stage2_wait_chunk(count - 2, count, ring_slots, 0)
            with self._scope("host_merge"):
                self._host_merge_chunk(count - 1, loop_layout)
            self._wait_chunk_combine(count - 1, count, ring_slots, 0)
            for chunk_idx in range(max(count - ring_slots + lookahead, 0), count):
                with self._scope("combine_wait"):
                    self._stage2_wait_chunk(chunk_idx, count, ring_slots, 1)
                with self._scope("merge"):
                    self._merge_chunk(chunk_idx, loop_layout)
            self._stage2_wait_chunk(count - 1, count, ring_slots, 0)
            self._lane_major = False
            return jnp.stack(
                [
                    ws.merged[
                        batch_idx // chunk,
                        pl.ds(slots * (batch_idx % chunk), self.heads),
                        :,
                    ]
                    for batch_idx in range(batch)
                ]
            )
        if batch != B32_BATCH:
            self._wait_chunk_combine(count - 1, count, ring_slots, 0)
        for chunk_idx in range(max(count - ring_slots + lookahead, 0), count):
            with self._scope("combine_wait"):
                self._wait_chunk_combine(chunk_idx, count, ring_slots, 1)
            with self._scope("merge"):
                self._merge_chunk(chunk_idx, loop_layout)
        if batch == B32_BATCH:
            self._wait_chunk_combine(count - 1, count, ring_slots, 0)
        return jnp.stack(
            [
                ws.merged[
                    batch_idx // chunk,
                    pl.ds(slots * (batch_idx % chunk), self.heads),
                    :,
                ]
                for batch_idx in range(batch)
            ]
        )

    def _validate_chunked_config(
        self,
        chunk: int,
        keys: _TensorOrRef | None,
        *,
        hier: bool,
    ) -> None:
        """Validate chunked explicit-keys and ``host_combine`` preconditions."""
        if keys is not None and chunk != self.sequence_rows:
            msg = "explicit new keys need one sequence per chunk"
            raise ValueError(msg)
        if hier and (
            self.group != B4_BATCH * HOST_LANES
            or not self.lse_in_padding
            or keys is None
            or "combine" in self.skip
        ):
            msg = "host_combine: DCP32 explicit-keys sequences with lse_in_padding"
            raise ValueError(msg)

    def _attend_chunk_local_and_send(
        self,
        chunk_idx: _TensorOrRef,
        loop_layout: ChunkLoopLayout,
        first_step: Callable[[_TensorOrRef], _TensorOrRef],
    ) -> None:
        """Run local page attention, pack partials, and issue combine for chunk."""
        chunk = loop_layout.chunk
        count = loop_layout.count
        ring_slots = loop_layout.ring_slots
        hier = loop_layout.hier
        ws = self.ws
        with self._scope("gather_wait"):
            self._wait_chunk_gather(chunk_idx, count, ring_slots)
        local_scope = self._scope("local")
        local_scope.__enter__()
        self._build_chunk_queries(chunk_idx, chunk, count, ring_slots)
        ws.maximum[...] = jnp.full(ws.maximum.shape, -jnp.inf, jnp.float32)
        ws.denominator[...] = jnp.zeros(ws.denominator.shape, jnp.float32)
        ws.accumulator[...] = jnp.zeros(ws.accumulator.shape, jnp.float32)
        first = chunk * chunk_idx
        self._attend_range(
            first_step(first),
            first_step(first) if "attend" in self.skip else first_step(first + chunk),
            first,
        )
        local_scope.__exit__(None, None, None)

        @pl.when(chunk_idx >= 1)
        def free_send_buffer() -> None:
            """Wait for the previous chunk's send buffer to drain."""
            with self._scope("send_wait"):
                self._wait_chunk_combine(chunk_idx - 1, count, ring_slots, 0)

        with self._scope("pack_send"):
            self._pack_partials(0, 0 if "pack" in self.skip else chunk)
            self._send_chunk_partials(chunk_idx, loop_layout)

        if hier:

            @pl.when(chunk_idx >= 1)
            def host_stage() -> None:
                """Run host-level combine for ``chunk_idx - 1``."""
                with self._scope("host_wait"):
                    self._wait_chunk_combine(chunk_idx - 1, count, ring_slots, 1)

                    @pl.when(chunk_idx >= RING_SLOTS_DEFAULT)
                    def free_stage2() -> None:
                        """Wait for stage-2 send buffer from ``chunk_idx - 2``."""
                        self._stage2_wait_chunk(chunk_idx - 2, count, ring_slots, 0)

                with self._scope("host_merge"):
                    self._host_merge_chunk(chunk_idx - 1, loop_layout)

    def _attend_chunked(
        self,
        queries: _TensorOrRef,
        new_latents: _TensorOrRef,
        between: Callable[[], None] | None = None,
        **attend_kwargs: Unpack[_AttendExplicitKeysKwargs],
    ) -> _TensorOrRef:
        """Run pipelined multi-chunk DCP attention over ``batch // chunk`` chunks."""
        keys = attend_kwargs.get("keys")
        key_valid = attend_kwargs.get("key_valid")
        ws = self.ws
        batch = self.batch
        chunk = chunk_layout(batch, self.heads, chunk_rows=self.chunk_rows)[0]
        slots = self.slots
        ring_slots = int(ws.gathered.shape[0])
        count = batch // chunk
        hier = self.host_combine is not None
        self._validate_chunked_config(chunk, keys, hier=hier)
        lookahead = (
            RING_SLOTS_DEFAULT if ring_slots == RING_SLOTS_MAX and not hier else 1
        )
        self._lane_major = hier
        host, lane = (
            (self.member // HOST_LANES, self.member % HOST_LANES)
            if hier
            else (None, None)
        )
        rows = slots * chunk
        loop_layout = ChunkLoopLayout(
            chunk=chunk,
            count=count,
            ring_slots=ring_slots,
            lookahead=lookahead,
            rows=rows,
            hier=hier,
            keys=keys,
            key_valid=key_valid,
            host=host,
            lane=lane,
        )
        ws.own_rows[...] = jnp.zeros(ws.own_rows.shape, jnp.float32)
        ws.new_rows[...] = jnp.zeros(ws.new_rows.shape, jnp.float32)
        for batch_idx in range(batch):
            chunk_idx, row_offset = divmod(batch_idx, chunk)
            ws.own_rows[chunk_idx, pl.ds(slots * row_offset, self.heads), :] = queries[
                batch_idx
            ]
            ws.new_rows[chunk_idx, pl.ds(slots * row_offset, self.heads), :] = (
                jnp.broadcast_to(new_latents[batch_idx][None, :], (self.heads, LANES))
            )
        steps = [
            (self.counts[batch_idx] + self.span - 1) // self.span
            for batch_idx in range(batch)
        ]

        def first_step(first: _TensorOrRef) -> _TensorOrRef:
            """Return the starting page step index for row ``first``."""
            total = jnp.int32(0)
            for batch_idx in range(batch):
                total = total + jnp.where(batch_idx < first, steps[batch_idx], 0)
            return total

        for initial in range(min(lookahead, count)):
            self._send_chunk_queries(initial, rows, count, ring_slots)
        if between is not None:
            between()

        @pl.loop(0, count)
        def chunk_step(chunk_idx: _TensorOrRef) -> None:
            """Advance one pipelined chunk step of local attention and combine."""
            done = chunk_idx + lookahead - ring_slots

            @pl.when(done >= 0)
            def merge_done() -> None:
                """Merge the completed chunk whose ring slot will be reused."""
                with self._scope("combine_wait"):
                    if hier:
                        self._stage2_wait_chunk(done, count, ring_slots, 1)
                    else:
                        self._wait_chunk_combine(done, count, ring_slots, 1)
                with self._scope("merge"):
                    self._merge_chunk(done, loop_layout)

            @pl.when(chunk_idx + lookahead < count)
            def send_next() -> None:
                """Send lookahead queries for ``chunk_idx + lookahead``."""
                self._send_chunk_queries(chunk_idx + lookahead, rows, count, ring_slots)

            self._attend_chunk_local_and_send(chunk_idx, loop_layout, first_step)

        return self._finish_chunked_tail(loop_layout)


def _merge_received(
    received: _TensorOrRef,
    own_rows: _TensorOrRef,
    new_rows: _TensorOrRef,
    group: int,
    *,
    unroll: int = 1,
) -> _TensorOrRef:
    """Merge ``group`` peers' partials in ``received`` with the new token online."""
    top = jnp.sum(own_rows * new_rows, axis=1, keepdims=True)
    accumulator = new_rows[:, :LATENT]
    denominator = jnp.ones_like(top)

    def fold(
        peer: _TensorOrRef,
        carry: tuple[_TensorOrRef, _TensorOrRef, _TensorOrRef],
    ) -> tuple[_TensorOrRef, _TensorOrRef, _TensorOrRef]:
        """Fold ``peer``'s partial output and LSE into the online accumulator."""
        top_acc, val_acc, den_acc = carry
        partial = received[peer].astype(jnp.float32)
        lse = partial[:, LATENT : LATENT + 1] + partial[:, LATENT + 1 : LATENT + 2]
        new_top = jnp.maximum(top_acc, lse)
        keep = jnp.exp(top_acc - new_top)
        weight = jnp.exp(lse - new_top)
        return (
            new_top,
            val_acc * keep + weight * partial[:, :LATENT],
            den_acc * keep + weight,
        )

    _, accumulator, denominator = jax.lax.fori_loop(
        0, group, fold, (top, accumulator, denominator), unroll=unroll
    )
    return accumulator / denominator


def _merge_received_sequences(
    received: _TensorOrRef,
    own_rows: _TensorOrRef,
    latents: _TensorOrRef,
    spec: SequenceMergeSpec,
) -> _TensorOrRef:
    """Merge multi-row sequence chunk partials and causal in-step new latents."""
    group = spec.group
    slots = spec.slots
    sequence_rows = spec.sequence_rows
    unroll = spec.unroll
    scores = jax.lax.dot_general(
        own_rows,
        latents,
        (((1,), (1,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    slot_row = jax.lax.broadcasted_iota(jnp.int32, scores.shape, 0) // slots
    key_row = jax.lax.broadcasted_iota(jnp.int32, scores.shape, 1)
    causal = (key_row // sequence_rows == slot_row // sequence_rows) & (
        key_row <= slot_row
    )
    scores = jnp.where(causal, scores, -jnp.inf)
    top = jnp.max(scores, axis=1, keepdims=True)
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
        peer: _TensorOrRef,
        carry: tuple[_TensorOrRef, _TensorOrRef, _TensorOrRef],
    ) -> tuple[_TensorOrRef, _TensorOrRef, _TensorOrRef]:
        """Fold ``peer``'s partial into the online sequence accumulator."""
        top_acc, val_acc, den_acc = carry
        partial = received[peer].astype(jnp.float32)
        lse = partial[:, LATENT : LATENT + 1] + partial[:, LATENT + 1 : LATENT + 2]
        new_top = jnp.maximum(top_acc, lse)
        keep = jnp.exp(top_acc - new_top)
        weight = jnp.exp(lse - new_top)
        return (
            new_top,
            val_acc * keep + weight * partial[:, :LATENT],
            den_acc * keep + weight,
        )

    def fold_group(
        peer: _TensorOrRef,
        carry: tuple[_TensorOrRef, _TensorOrRef, _TensorOrRef],
    ) -> tuple[_TensorOrRef, _TensorOrRef, _TensorOrRef]:
        """Unroll ``unroll`` consecutive peer folds per loop step."""
        for step_offset in range(unroll):
            carry = fold(peer * unroll + step_offset, carry)
        return carry

    _, accumulator, denominator = jax.lax.fori_loop(
        0, group // unroll, fold_group, (top, accumulator, denominator)
    )
    return accumulator / denominator


def _lse_packet(
    normalized: _TensorOrRef,
    lse: _TensorOrRef,
    lse_rows: int,
) -> _TensorOrRef:
    """Build a ``[rows, 640]`` BF16 ``lse_in_padding`` packet from ``normalized``."""
    rows = int(normalized.shape[0])
    pad = rows - lse_rows
    high = lse.astype(jnp.bfloat16).astype(jnp.float32)
    low = (lse - high).astype(jnp.bfloat16).astype(jnp.float32)
    row = jax.lax.broadcasted_iota(jnp.int32, (rows, LSE_VECTOR_LANES), 0)
    lane = jax.lax.broadcasted_iota(jnp.int32, (rows, LSE_VECTOR_LANES), 1)
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
        (normalized, jnp.zeros((rows, LANES - LATENT), jnp.float32)), axis=1
    )
    row640 = jax.lax.broadcasted_iota(jnp.int32, (rows, LANES), 0)
    pad_lanes = jnp.zeros((1, LANES - LSE_VECTOR_LANES), jnp.float32)
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
    received: _TensorOrRef,
    own_rows: _TensorOrRef,
    keys: _TensorOrRef,
    key_valid: _TensorOrRef,
    spec: KeyMergeSpec,
) -> _TensorOrRef:
    """Merge received peer partials with explicit new keys in two passes."""
    group = spec.group
    lse_rows = spec.lse_rows
    keys = keys.astype(jnp.float32)
    scores = jax.lax.dot_general(
        own_rows,
        keys,
        (((1,), (1,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    scores = jnp.where(key_valid != 0, scores, -jnp.inf)
    if lse_rows:
        rows = int(received.shape[1])
        pad = rows - lse_rows
        vector = received[:, pad, pl.ds(0, LSE_VECTOR_LANES)].astype(
            jnp.float32
        ) + received[:, pad + 1, pl.ds(0, LSE_VECTOR_LANES)].astype(jnp.float32)
        row = jax.lax.broadcasted_iota(jnp.int32, (group, rows, LSE_VECTOR_LANES), 1)
        lane = jax.lax.broadcasted_iota(jnp.int32, (group, rows, LSE_VECTOR_LANES), 2)
        lse = jnp.sum(
            jnp.where(lane == row, vector[:, None, :], 0.0),
            axis=2,
            keepdims=True,
        )
        lse = jnp.where(row[:, :, :1] < pad, lse, NEG)
    else:
        tails = received[:, :, LATENT : LATENT + LSE_VECTOR_LANES].astype(jnp.float32)
        lse = tails[:, :, 0:1] + tails[:, :, 1:2]
    top = jnp.maximum(jnp.max(scores, axis=1, keepdims=True), jnp.max(lse, axis=0))
    weights = jnp.exp(scores - top)
    peer_weights = jnp.exp(lse - top[None])
    accumulator = jax.lax.dot_general(
        weights,
        keys[:, :LATENT],
        (((1,), (0,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    denominator = jnp.sum(weights, axis=1, keepdims=True) + jnp.sum(
        peer_weights, axis=0
    )
    for peer in range(group):
        accumulator = accumulator + peer_weights[peer] * received[
            peer, :, :LATENT
        ].astype(jnp.float32)
    return accumulator / denominator


def _pack_fp8_query(query: _TensorOrRef) -> _TensorOrRef:
    """Pack FP8 query bytes and a lossless FP32 scale in padding lanes 576:580."""
    query = query.astype(jnp.bfloat16).astype(jnp.float32)
    maximum = jnp.max(jnp.abs(query), axis=1, keepdims=True)
    scale = jnp.where(maximum > 0, maximum / FP8_MAX_FINITE, 1.0)
    payload = jax.lax.bitcast_convert_type(
        (query / scale).astype(jnp.float8_e4m3fn), jnp.uint8
    ).astype(jnp.uint32)
    bits = jax.lax.bitcast_convert_type(scale, jnp.uint32)
    lanes = jax.lax.broadcasted_iota(jnp.int32, query.shape, 1)
    scale_bytes = (
        bits >> (((lanes - ROPE_END_LANE) % SCALE_WORD_BYTES) * BYTE_SHIFT_BITS)
    ) & BYTE_MASK
    return jnp.where(lanes < ROPE_END_LANE, payload, scale_bytes).astype(jnp.uint8)


def _unpack_fp8_query(
    packet: _TensorOrRef,
) -> tuple[_TensorOrRef, _TensorOrRef]:
    """Recover the FP8 query and FP32 scale and clear metadata lanes before QK."""
    packet = packet.astype(jnp.uint32)
    lanes = jax.lax.broadcasted_iota(jnp.int32, packet.shape, 1)
    words = packet << (((lanes - ROPE_END_LANE) % SCALE_WORD_BYTES) * BYTE_SHIFT_BITS)
    bits = jnp.sum(
        jnp.where(
            (lanes >= ROPE_END_LANE) & (lanes < SCALE_END_LANE),
            words,
            jnp.uint32(0),
        ),
        axis=1,
        keepdims=True,
    )
    scale = jax.lax.bitcast_convert_type(bits, jnp.float32)
    payload = jnp.where(lanes < ROPE_END_LANE, packet, jnp.uint32(0)).astype(jnp.uint8)
    return jax.lax.bitcast_convert_type(payload, jnp.float8_e4m3fn), scale
