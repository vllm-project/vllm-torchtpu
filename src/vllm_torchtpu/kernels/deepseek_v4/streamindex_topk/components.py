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
"""Static components for reusing StreamIndex's DMA and scoring pipeline."""

from dataclasses import dataclass
from typing import Any, Protocol

import jax

from .config import KVLayout
from .metadata import MetadataRef


class KeyReader(Protocol):
    """Decode a HEAD_ALONG_SUBLANE DMA window into VMEM keys and scales.

    Implementations are hashable static JIT arguments. ``packing`` is the number
    of positions per physical cache group. ``validate`` rejects unsupported
    cache shapes, dtypes, widths and layouts before kernel tracing. ``read``
    returns keys[rows, width] and FP32 scales[1, rows] from the supplied window.
    """

    @property
    def packing(self) -> int: ...

    def validate(self, cache: jax.Array, width: int, layout: KVLayout) -> None: ...

    def read(self, ref: Any, rows: int, width: int) -> tuple[jax.Array, jax.Array]: ...


class IndexSchedule(Protocol):
    """Define query positions, visible prefixes and per-chunk DMA steps.

    Implementations are hashable static JIT arguments. ``positions`` preserves
    the shape of uncompressed query coordinates. ``visible_lengths`` maps their
    causal compressed prefix lengths to admitted prefix lengths of the same
    shape. ``chunk_metadata`` returns one chunk's steps indexed from zero; it
    must cover every admitted score column. These hooks do not promise that
    score-free enumeration is equivalent to scoring and selection.
    """

    def validate(self, compression_ratio: int) -> None: ...

    def positions(self, positions: jax.Array) -> jax.Array: ...

    def visible_lengths(self, lengths: jax.Array) -> jax.Array: ...

    def chunk_metadata(self, meta: MetadataRef, chunk_idx: int) -> MetadataRef: ...


@dataclass(frozen=True)
class StreamIndexComponents:
    """Explicit opt-in; omitted components preserve the original kernel path."""

    keys: KeyReader
    schedule: IndexSchedule
