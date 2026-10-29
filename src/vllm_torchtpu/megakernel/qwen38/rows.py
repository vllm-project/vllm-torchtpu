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
"""Where a prefill step's tokens sit on a stage's cores.

A step's scheduled tokens, concatenated request by request (the *flat*
order), are dealt to the stage's ``cores`` in chunks of ``chunk`` tokens:
flat chunk ``k`` goes to core ``k % cores``, so core ``c``'s local row ``l``
is flat row ``((l // chunk) * cores + c) * chunk + l % chunk``. Every core
holds the same number of rows (the flat length padded to whole rounds of
``cores * chunk`` tokens), however the requests' lengths fall, and the PP
hand-off exchanges each core's rows with the same core of the next stage.

Flat order is not KV page order: a request's KV page ``p`` (absolute
positions ``[p * chunk, (p + 1) * chunk)``) is stored on core ``p % cores``
(vLLM's PCP pool layout). The attention layers stream every new KV row to
every core anyway, and each core writes the rows of the pages it owns.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np


@dataclasses.dataclass(frozen=True)
class Span:
    """One request's tokens of the step: positions ``[start, end)``."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclasses.dataclass(frozen=True)
class RowPlan:
    """A step's rows in flat order (``flat_rows`` = ``cores * rows``, padded
    rows have ``request == -1``)."""

    cores: int
    chunk: int
    rows: int  # rows per core
    request: np.ndarray  # [flat_rows] int32: the row's index into the spans
    position: np.ndarray  # [flat_rows] int32: its absolute position
    flat_start: np.ndarray  # [requests] int32: each span's first flat row

    @property
    def flat_rows(self) -> int:
        return self.cores * self.rows

    def core_rows(self, core: int) -> np.ndarray:
        """Flat indices of ``core``'s local rows, in local order."""
        local = np.arange(self.rows)
        return (
            (local // self.chunk) * self.cores + core
        ) * self.chunk + local % self.chunk


def rows_per_core(tokens: int, cores: int, chunk: int) -> int:
    """Each core's rows for a step of ``tokens`` flat tokens."""
    round_tokens = cores * chunk
    return max(1, -(-tokens // round_tokens)) * chunk


def plan_rows(
    spans: Sequence[Span], cores: int, chunk: int, rows: int | None = None
) -> RowPlan:
    """The row plan of ``spans`` (in batch order), ``rows`` per core (at
    least :func:`rows_per_core`; None: exactly that)."""
    tokens = sum(span.length for span in spans)
    needed = rows_per_core(tokens, cores, chunk)
    if rows is None:
        rows = needed
    elif rows < needed or rows % chunk:
        raise ValueError(
            f"{rows} rows per core cannot hold {tokens} tokens in chunks of {chunk} "
            f"on {cores} cores (at least {needed}, a multiple of {chunk})"
        )
    flat_rows = cores * rows
    request = np.full(flat_rows, -1, np.int32)
    position = np.full(flat_rows, -1, np.int32)
    flat_start = np.zeros(len(spans), np.int32)
    offset = 0
    for i, span in enumerate(spans):
        flat_start[i] = offset
        request[offset : offset + span.length] = i
        position[offset : offset + span.length] = np.arange(span.start, span.end)
        offset += span.length
    return RowPlan(cores, chunk, rows, request, position, flat_start)
