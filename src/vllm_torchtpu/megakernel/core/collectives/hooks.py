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
"""Hand-off types between a dense matmul and the collectives that overlap it:
`HalfSplitHook` (a hook between the two lane halves of the `K` tiles) and
`ColumnSplit` (output column groups handed to hooks). `dma.stream` takes them
as `dmatmul(hook=, split=)`; the all-reduce builds them."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HalfSplitHook:
    """`hook=` of `TwoRing.dmatmul` / `TwoRing.arp_mm`: `fn()` runs between the
    two lane halves of a dense matmul's `K` tiles.

    `K` tile `s` (for `s` in `tiles`) belongs to half 0 and tile `s + hpt` to
    half 1. `first` (0 or 1, may be traced) is the half whose tiles are summed
    before `fn()`; the other half's tiles are summed after it, so `fn()` may
    still write that half's lanes of the matmul input.
    """

    tiles: tuple[int, ...]
    hpt: int
    first: Any
    fn: Callable[[], None]


@dataclass(frozen=True)
class ColumnSplit:
    """`split=` of `TwoRing.dmatmul`: the output columns in groups whose sums go
    to hooks instead of `acc`.

    `pre()` runs before the matmul. Group `i` covers the `(c0, width)` column
    chunks `cols[i]` (`c0` may be traced); once its chunks are summed,
    `hooks[i](vals)` receives their f32 sums in `cols[i]` order. The groups
    must cover all `N` columns.
    """

    pre: Callable[[], None]
    cols: tuple[Sequence[tuple[Any, int]], ...]
    hooks: tuple[Callable[[list[Any]], None], ...]


__all__ = ["ColumnSplit", "HalfSplitHook"]
