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
