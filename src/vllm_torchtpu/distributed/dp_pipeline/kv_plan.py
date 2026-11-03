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
"""Which K/V pages a helper of a pipelined prefill sends the owner.

The owner decodes alone, so before its first decode step it needs, for every
layer a helper ran, the K/V the decode reads: every prompt block of a
full-attention layer, and the blocks of a sliding-window layer's last window.
Both engines cut the prompt into the same chunks, so both derive the same
transfers from the plan, the chunk and the prompt length; a transfer names
logical blocks, which each engine maps through its own block table.

A helper sends each chunk's full-attention pages after computing the chunk,
and after its last chunk the pages of the final window. Every layer of a
transfer takes a fixed number of rows (``pages``), so the programs that
gather and scatter them keep one shape.
"""

from __future__ import annotations

from dataclasses import dataclass


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class KvGroup:
    """A KV cache group: its layers ``(index, name)`` in model order share
    block ids; ``sliding_window`` is None for full attention."""

    group_id: int
    layers: tuple[tuple[int, str], ...]
    sliding_window: int | None

    def names_in(self, layers: tuple[int, int]) -> tuple[str, ...]:
        start, end = layers
        return tuple(name for index, name in self.layers if start <= index < end)


@dataclass(frozen=True)
class KvEntry:
    """Pages of ``layer_names`` (one group) at logical ``blocks``."""

    group_id: int
    layer_names: tuple[str, ...]
    blocks: tuple[int, ...]


@dataclass(frozen=True)
class KvTransfer:
    """One hand-off of pages, identified on its ring by ``token``. Layer
    ``i`` of the flattened entries takes rows ``[i * pages, i * pages +
    len(blocks))``."""

    token: int
    pages: int
    entries: tuple[KvEntry, ...]

    @property
    def num_layers(self) -> int:
        return sum(len(entry.layer_names) for entry in self.entries)

    @property
    def rows(self) -> int:
        return self.num_layers * self.pages


def chunk_pages(max_chunk_tokens: int, block_size: int) -> int:
    """Rows per layer of a chunk transfer: the blocks a chunk of at most
    ``max_chunk_tokens`` touches, starting anywhere in a block."""
    return _cdiv(max_chunk_tokens, block_size) + 1


def window_pages(groups: tuple[KvGroup, ...], block_size: int) -> int:
    """Rows per layer of a window transfer (0 without sliding layers)."""
    windows = [g.sliding_window for g in groups if g.sliding_window is not None]
    return max((_cdiv(w - 1, block_size) + 1 for w in windows), default=0)


def chunk_transfer(
    groups: tuple[KvGroup, ...],
    layers: tuple[int, int],
    token: int,
    first: int,
    count: int,
    block_size: int,
    pages: int,
) -> KvTransfer | None:
    """The full-attention pages the chunk ``[first, first + count)`` wrote
    in ``layers``, or None when the range has no full-attention layer."""
    blocks = tuple(range(first // block_size, _cdiv(first + count, block_size)))
    if len(blocks) > pages:
        raise ValueError(f"a chunk of {count} tokens spans more than {pages} blocks")
    entries = tuple(
        KvEntry(g.group_id, names, blocks)
        for g in groups
        if g.sliding_window is None and (names := g.names_in(layers))
    )
    return KvTransfer(token, pages, entries) if entries else None


def window_transfer(
    groups: tuple[KvGroup, ...],
    layers: tuple[int, int],
    token: int,
    prompt_len: int,
    block_size: int,
) -> KvTransfer | None:
    """The sliding-window pages the first decode step reads in ``layers``:
    the blocks holding prompt positions ``[prompt_len - window + 1,
    prompt_len)``, which the sliding-window manager still holds then."""
    pages = window_pages(groups, block_size)
    entries = []
    for g in groups:
        if g.sliding_window is None or not (names := g.names_in(layers)):
            continue
        first = max(0, prompt_len - g.sliding_window + 1) // block_size
        blocks = tuple(range(first, _cdiv(prompt_len, block_size)))
        entries.append(KvEntry(g.group_id, names, blocks))
    return KvTransfer(token, pages, tuple(entries)) if entries else None


def max_ring_rows(
    groups: tuple[KvGroup, ...],
    stage_layers: list[tuple[int, int]],
    chunk_rows: int,
    block_size: int,
) -> int:
    """Rows a helper-to-owner KV ring slot needs for any helper range in
    ``stage_layers``."""
    rows = 0
    window_rows = window_pages(groups, block_size)
    for layers in stage_layers:
        full = sum(len(g.names_in(layers)) for g in groups if g.sliding_window is None)
        sliding = sum(
            len(g.names_in(layers)) for g in groups if g.sliding_window is not None
        )
        rows = max(rows, full * chunk_rows, sliding * window_rows)
    return rows
