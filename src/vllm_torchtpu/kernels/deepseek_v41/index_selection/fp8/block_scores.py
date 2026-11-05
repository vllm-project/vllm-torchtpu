# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate block maxima published by the dense FP8 scorer."""

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .cache import PAGE_RECORDS


def supports_candidate_publication(block_size: int) -> bool:
    """Block reductions fit independently in each packed-record vector."""
    return block_size > 0 and PAGE_RECORDS % block_size == 0


def block_maxima(pages, block):
    """Block maxima of up to `block` [rows, 128] pages, in block order.

    Page p is first rolled so its block j ends at lane j * block + p. A
    butterfly then halves the vector count per level: each pair swaps lanes
    across one bit and takes the maximum, so a level costs one roll per pair
    instead of one per page. Lane j * block + p ends with page p's block j
    maximum, and one lane gather restores block order.
    """
    assert pages, "Block maxima need at least one page"
    lanes = lax.broadcasted_iota(jnp.int32, (1, PAGE_RECORDS), 1)
    vectors = [
        pltpu.roll(page, (p - block + 1) % PAGE_RECORDS, axis=1)
        for p, page in enumerate(pages)
    ]
    # Missing pages repeat the first; their lanes are sliced off by the caller.
    vectors += vectors[:1] * (block - len(vectors))
    shift = 1
    while len(vectors) > 1:
        bit = (lanes & shift) != 0
        vectors = [
            jnp.maximum(
                jnp.where(bit, b, a), pltpu.roll(jnp.where(bit, a, b), shift, axis=1)
            )
            for a, b in zip(vectors[::2], vectors[1::2])
        ]
        shift *= 2
    merged = vectors[0]
    blocks = PAGE_RECORDS // block
    order = (lanes % blocks) * block + lanes // blocks
    return jnp.take_along_axis(merged, jnp.broadcast_to(order, merged.shape), axis=1)


class TileBlockWriter:
    """Writes one key tile's block maxima into a query slot's block row.

    Scaled score pages arrive in order; every `block` pages, or at the tile's
    last live page, their maxima are merged and written. The newest visible
    block of each row is forced to +inf so it is always selected.
    """

    def __init__(self, row, block, tile, key_tile, lengths, live_pages):
        self.row, self.block, self.live_pages = row, block, live_pages
        self.tile_start = tile * (key_tile // block)
        # A tile's blocks start at lane `shift` of the row.
        self.shift = self.tile_start % PAGE_RECORDS
        self.last = (lengths - 1) // block
        self.pending, self.carry = [], []

    def add(self, local_page, scaled):
        self.pending.append(scaled)
        if (local_page + 1) % self.block and local_page != self.live_pages - 1:
            return
        per_page = PAGE_RECORDS // self.block
        written = len(self.pending) * per_page
        packed = block_maxima(self.pending, self.block)[:, :written]
        start = self.tile_start + (local_page + 1) * per_page - written
        positions = start + jnp.arange(written)
        packed = jnp.where(positions[None, :] == self.last[:, None], jnp.inf, packed)
        self._write(start, packed, final=local_page == self.live_pages - 1)
        self.pending.clear()

    def _write(self, start, values, final):
        # Each rotated write fills the rest of its vector and carries its tail
        # into the next one in registers: only the tile's first and last
        # vectors touch VMEM outside a plain aligned store.
        rows, width = values.shape
        if width < PAGE_RECORDS:
            pad = jnp.full((rows, PAGE_RECORDS - width), -jnp.inf, values.dtype)
            values = jnp.concatenate([values, pad], axis=1)
        rolled = pltpu.roll(values, self.shift, axis=1)
        lane = lax.broadcasted_iota(jnp.int32, rolled.shape, 1)
        base = pl.multiple_of(start - self.shift, PAGE_RECORDS)
        low = self.carry.pop() if self.carry else self.row[:, pl.ds(base, PAGE_RECORDS)]
        self.row[:, pl.ds(base, PAGE_RECORDS)] = jnp.where(
            lane >= self.shift, rolled, low
        )
        if final:
            self.row[:, pl.ds(base + PAGE_RECORDS, PAGE_RECORDS)] = jnp.where(
                lane < self.shift, rolled, -jnp.inf
            )
        else:
            self.carry.append(rolled)
