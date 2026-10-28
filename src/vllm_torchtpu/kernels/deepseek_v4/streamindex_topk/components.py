# SPDX-License-Identifier: Apache-2.0
"""Static components for reusing StreamIndex's DMA and scoring pipeline."""

from dataclasses import dataclass
from typing import Any, Protocol

import jax

from .config import KVLayout
from .metadata import MetadataRef


class KeyReader(Protocol):
    """Decode a DMA window into keys and per-position scales in VMEM."""

    @property
    def packing(self) -> int: ...

    def validate(self, cache: jax.Array, width: int, layout: KVLayout) -> None: ...

    def read(self, ref: Any, rows: int, width: int) -> tuple[jax.Array, jax.Array]: ...


class IndexSchedule(Protocol):
    """Define visible positions and the scalar-prefetch window per chunk."""

    def validate(self, compression_ratio: int) -> None: ...

    def positions(self, positions: jax.Array) -> jax.Array: ...

    def visible_lengths(self, lengths: jax.Array) -> jax.Array: ...

    def chunk_metadata(self, meta: MetadataRef, chunk_idx: int) -> MetadataRef: ...


@dataclass(frozen=True)
class StreamIndexComponents:
    """Explicit opt-in; omitted components preserve the original kernel path."""

    keys: KeyReader
    schedule: IndexSchedule
    # Score a multi-token query block one token at a time with the exact
    # single-token (decode) expression, sharing one KV stream and one q @ k
    # matmul per block. Each token's scores are then bit-identical to scoring
    # it alone as a one-token sequence, so a request's verify tokens can share
    # its compressed-key stream without changing any top-k selection. Meant
    # for small blocks (speculative verify); the blocked prefill layout is
    # bypassed.
    per_token_scores: bool = False
    # (expansion, stride, offset): each `page_indices` entry stands for
    # `expansion` consecutive kernel pages; kernel page k of a sequence reads
    # physical page entry[k // expansion] * stride + offset + k % expansion.
    # A caller whose pages are runs inside larger storage pages passes the
    # storage table instead of an expanded one, so the scalar-prefetched table
    # is `expansion` times smaller. (1, 1, 0) is the plain table.
    table_expansion: tuple[int, int, int] = (1, 1, 0)
    # With table_expansion and a KV block of whole entries: one DMA per entry
    # (its `expansion` kernel pages are consecutive) instead of one per page.
    coalesce_entries: bool = False
