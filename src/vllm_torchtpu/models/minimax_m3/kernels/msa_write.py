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
"""In-place row writes into the packed MSA caches.

An XLA scatter into the packed layout ``[pages, S / p, p, W]`` cannot index
the packed axis in place: XLA relayouts the whole cache to ``[rows, W]`` and
back around it, two full-cache copies per cache per layer. This kernel
instead rewrites only the pages that receive rows: per token run
(consecutive tokens of one request landing in one page) it copies the page in
as 32-bit words, merges the run's rows into their packing slots and copies
the page back over the aliased cache. One call writes every cache of a layer
(K/V rows, index keys, scales) per run, with up to `_DEPTH` runs' copies in
flight.
"""

import functools
from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4

_UINT_BY_BITS = {16: jnp.uint16, 8: jnp.uint8}
# Runs whose page copies are in flight at once: a run's merge is a few
# hundred bundles, so its copies' latency, not their bytes, bounds a
# one-run-at-a-time pipeline (~0.5 us per run and cache in the c48 twins,
# j-efdac848).
_DEPTH = 4
# Rows of a 32-bit sublane tile: row windows start at a multiple.
_ROW_TILE = 8
_LANES = 128


class Rows(NamedTuple):
    """One token row per slot of a packed ``[pages, S / p (+ 1), p, W]`` cache.

    ``rows [T, W]`` are in the cache dtype, or uint8 nibbles below 16 with
    ``bits=4`` (the 4-bit caches of `msa_fp4`, 8 slots per 32-bit word).
    `scales` (f32 ``[T]``) go to the last word row of a page with folded
    scales (`msa_fp4.scales_in_page`), lane = slot.
    """

    cache: jax.Array
    rows: jax.Array
    bits: int | None = None
    scales: jax.Array | None = None


class SlotValues(NamedTuple):
    """Per-slot f32 ``values [T, n]`` into rows ``[0, n)`` of an f32 ``[pages,
    rows, S]`` cache, lane = slot."""

    cache: jax.Array
    values: jax.Array


class _RowsSpec(NamedTuple):
    bits: int
    folds: int  # 128-lane blocks of a row
    data_rows: int  # word rows holding slots
    scale_lane: int | None  # lane of the lane values holding folded scales


class _ValuesSpec(NamedTuple):
    num_values: int  # lanes [0, num_values) of the lane values


def _kernel(
    # Scalar prefetch
    num_runs_ref,  # [1]
    run_page_ref,  # [R] page of each run
    run_lo_ref,  # [R] first slot of each run within its page
    run_hi_ref,  # [R] one past the last slot
    run_tok0_ref,  # [R] first token of each run
    *refs,
    specs: tuple,
    page_size: int,
    num_tokens: int,
    has_lanes: bool,
):
    """In (HBM): each `Rows` target's 32-bit slot values ``[F, T, 128]``
    (row ``t``'s lanes ``[128 f, 128 (f + 1))`` at ``[f, t]``), the lane
    values ``[T, 128]`` (if any), then the caches; out: the caches, aliased.

    A run's tokens ``tok0 + i`` take slots ``lo + i`` of its page: slot ``s``
    reads batch row ``tok0 - lo + s``. Each run copies in a window of
    ``S + 8`` batch rows holding all of its tokens (clamped into the batch;
    rows past its tokens are never selected) to row ``S`` of a ``3S + 8``-row
    buffer, so slot 0's row there is ``S + tok0 - lo - start`` >= 1."""
    num_rows = sum(isinstance(s, _RowsSpec) for s in specs)
    it = iter(refs)
    words_hbm = [next(it) for _ in range(num_rows)]
    lanes_hbm = next(it) if has_lanes else None
    for _ in specs:  # the aliased inputs: `caches` below are the same buffers
        next(it)
    caches = [next(it) for _ in specs]
    windows = [next(it) for _ in range(num_rows)]
    lane_window = next(it) if has_lanes else None
    in_bufs = [next(it) for _ in specs]
    out_bufs = [next(it) for _ in specs]
    in_sem, out_sem = next(it), next(it)
    window = page_size + _ROW_TILE

    def placement(r):
        base = run_tok0_ref[r] - run_lo_ref[r]
        start = jnp.minimum(
            jnp.maximum(base, 0) // _ROW_TILE * _ROW_TILE, num_tokens - window
        )
        return start, page_size + base - start

    def in_copies(r, slot):
        start, _ = placement(r)
        page = run_page_ref[r]
        rows = pl.ds(start, window)
        to = pl.ds(page_size, window)
        copies = [
            pltpu.make_async_copy(w.at[:, rows], buf.at[slot, :, to], in_sem.at[slot])
            for w, buf in zip(words_hbm, windows)
        ]
        if has_lanes:
            copies.append(
                pltpu.make_async_copy(
                    lanes_hbm.at[rows], lane_window.at[slot, to], in_sem.at[slot]
                )
            )
        copies += [
            pltpu.make_async_copy(cache.at[page], buf.at[slot], in_sem.at[slot])
            for cache, buf in zip(caches, in_bufs)
        ]
        return copies

    def out_copies(r, slot):
        page = run_page_ref[r]
        return [
            pltpu.make_async_copy(buf.at[slot], cache.at[page], out_sem.at[slot])
            for cache, buf in zip(caches, out_bufs)
        ]

    def merge(r, slot):
        lo, hi = run_lo_ref[r], run_hi_ref[r]
        _, row0 = placement(r)
        # [lane value, slot]: the run's lane values, one row per value.
        by_slot = lane_window[slot, pl.ds(row0, page_size), :].T if has_lanes else None
        windows_left = iter(windows)
        for spec, in_buf, out_buf in zip(specs, in_bufs, out_bufs):
            if isinstance(spec, _ValuesSpec):
                page = in_buf[slot]
                rows, lanes = page.shape
                row = jax.lax.broadcasted_iota(jnp.int32, page.shape, 0)
                lane = jax.lax.broadcasted_iota(jnp.int32, page.shape, 1)
                take = (row < spec.num_values) & (lane >= lo) & (lane < hi)
                values = pltpu.bitcast(by_slot[:rows], jnp.float32)
                out_buf[slot] = jnp.where(take, values, page)
                continue
            buf = next(windows_left)
            words = in_buf.bitcast(jnp.uint32)[slot, :, 0, :]
            rows, width = words.shape
            packing = 32 // spec.bits
            row = jax.lax.broadcasted_iota(jnp.int32, (rows, width), 0)
            for j in range(packing):
                slot_j = row * packing + j  # past the data rows: >= S, never taken
                take = (slot_j >= lo) & (slot_j < hi)
                keep = jnp.uint32(
                    ~(((1 << spec.bits) - 1) << (spec.bits * j)) & 0xFFFFFFFF
                )
                # Slots j, p + j, ...: rows at stride p.
                new = jnp.concatenate(
                    [
                        buf.at[slot, f][pl.ds(row0 + j, spec.data_rows, stride=packing)]
                        for f in range(spec.folds)
                    ],
                    axis=1,
                )
                if spec.data_rows < rows:
                    new = jnp.pad(new, ((0, rows - spec.data_rows), (0, 0)))
                merged = (words & keep) | (new << jnp.uint32(spec.bits * j))
                words = jnp.where(take, merged, words)
            if spec.scale_lane is not None:  # last word row: one scale per slot (lane)
                bits = by_slot[spec.scale_lane : spec.scale_lane + 1]
                if width > _LANES:
                    bits = jnp.pad(bits, ((0, 0), (0, width - _LANES)))
                lane = jax.lax.broadcasted_iota(jnp.int32, (rows, width), 1)
                take = (row == rows - 1) & (lane >= lo) & (lane < hi)
                words = jnp.where(take, bits[:, :width], words)
            out_buf.bitcast(jnp.uint32)[slot, :, 0, :] = words

    num_runs = num_runs_ref[0]

    def start(r, slot):
        for c in in_copies(r, slot):
            c.start()

    for i in range(_DEPTH):
        pl.when(i < num_runs)(functools.partial(start, i, i))

    def body(r, carry):
        slot = r % _DEPTH
        for c in in_copies(r, slot):
            c.wait()

        @pl.when(r >= _DEPTH)
        def _():  # the out buffer's previous run
            for c in out_copies(r - _DEPTH, slot):
                c.wait()

        merge(r, slot)
        for c in out_copies(r, slot):
            c.start()

        pl.when(r + _DEPTH < num_runs)(functools.partial(start, r + _DEPTH, slot))

        return carry

    jax.lax.fori_loop(0, num_runs, body, 0)

    def drain(r, carry):
        for c in out_copies(r, r % _DEPTH):
            c.wait()
        return carry

    jax.lax.fori_loop(jnp.maximum(num_runs - _DEPTH, 0), num_runs, drain, 0)


def _slot_words(target: Rows) -> tuple[jax.Array, _RowsSpec, int]:
    """A `Rows` target's 32-bit slot values ``[T, W]``, its spec (no scale
    lane yet) and its page's slot count."""
    cache = target.cache
    _, rows_per_page, word_bytes, width = cache.shape
    bits = target.bits or 32 // word_bytes
    folded = target.scales is not None
    if folded and not (
        bits == 4 and msa_fp4.scales_in_page(rows_per_page, cache.dtype)
    ):
        raise ValueError(
            f"scales need a 4-bit cache with a scale row, got {cache.shape}"
        )
    if width % _LANES:
        raise ValueError(f"cache rows must be whole 128-lane blocks, got {cache.shape}")
    if bits == 4:
        words = target.rows.astype(jnp.uint32) & 0xF
    else:
        words = jax.lax.bitcast_convert_type(target.rows, _UINT_BY_BITS[bits]).astype(
            jnp.uint32
        )
    data_rows = rows_per_page - 1 if folded else rows_per_page
    spec = _RowsSpec(bits, width // _LANES, data_rows, None)
    return words, spec, data_rows * 32 // bits


def write_runs(
    targets: Sequence[Rows | SlotValues],
    pages: jax.Array,
    slots: jax.Array,
    *,
    max_runs: int,
    interpret: bool = False,
) -> list[jax.Array]:
    """Write token ``t``'s rows / values of every target at ``(pages[t],
    slots[t])`` of its cache; returns the caches in `targets` order.

    The caches share the page geometry (same pages, slots per page). Tokens
    with a page outside the caches are dropped. Consecutive tokens of a run
    must take consecutive slots of one page, as a request's new tokens do;
    a run boundary is wherever that breaks, and no two runs may share a page
    (the kernel copies a run's page in while earlier runs' pages are still
    being written back; tokens map to pages by whole blocks, so a request's
    tokens in one page are one run).
    `max_runs` bounds the number of runs (see `max_write_runs`).
    """
    num_tokens = pages.shape[0]
    num_pages = targets[0].cache.shape[0]
    if any(t.cache.shape[0] != num_pages for t in targets):
        raise ValueError("write_runs targets must share their pages")
    num_runs, run_page, run_lo, run_hi, run_tok0 = token_runs(
        pages, slots, num_pages, max_runs
    )
    words, specs, scale_lanes = [], [], []
    lane_values = None
    page_size = None
    for target in targets:
        if isinstance(target, SlotValues):
            if lane_values is not None:
                raise ValueError("write_runs takes at most one SlotValues target")
            n = target.values.shape[1]
            specs.append(_ValuesSpec(n))
            lane_values = jax.lax.bitcast_convert_type(
                target.values.astype(jnp.float32), jnp.uint32
            )
            size = target.cache.shape[2]
        else:
            w, spec, size = _slot_words(target)
            if target.scales is not None:  # folded scales from the last lanes down
                lane = _LANES - 1 - len(scale_lanes)
                bits = jax.lax.bitcast_convert_type(
                    target.scales.astype(jnp.float32), jnp.uint32
                )
                scale_lanes.append((lane, bits))
                spec = spec._replace(scale_lane=lane)
            words.append(w)
            specs.append(spec)
        if page_size not in (None, size):
            raise ValueError("write_runs targets must share their page size")
        page_size = size
    has_lanes = lane_values is not None or bool(scale_lanes)
    if has_lanes and page_size != _LANES:
        raise ValueError(f"per-slot values need {_LANES}-slot pages, got {page_size}")
    window = page_size + _ROW_TILE
    padded = max(-(-num_tokens // _ROW_TILE) * _ROW_TILE, window)
    pad = ((0, padded - num_tokens), (0, 0))
    inputs = [
        jnp.pad(w, pad).reshape(padded, -1, _LANES).transpose(1, 0, 2) for w in words
    ]
    if has_lanes:  # one elementwise fusion: no column concatenates
        if lane_values is None:
            lanes = jnp.zeros((num_tokens, _LANES), jnp.uint32)
        else:
            lanes = jnp.pad(lane_values, ((0, 0), (0, _LANES - lane_values.shape[1])))
        lane = jnp.arange(_LANES, dtype=jnp.int32)
        for i, bits in scale_lanes:
            lanes = jnp.where(lane == i, bits[:, None], lanes)
        inputs.append(jnp.pad(lanes, pad))
    caches = [t.cache for t in targets]
    buffer_rows = 3 * page_size + _ROW_TILE
    scratch = [
        pltpu.VMEM((_DEPTH, s.folds, buffer_rows, _LANES), jnp.uint32)
        for s in specs
        if isinstance(s, _RowsSpec)
    ]
    if has_lanes:
        scratch.append(pltpu.VMEM((_DEPTH, buffer_rows, _LANES), jnp.uint32))
    pages_scratch = [pltpu.VMEM((_DEPTH,) + c.shape[1:], c.dtype) for c in caches]
    scratch += pages_scratch + pages_scratch
    scratch += [pltpu.SemaphoreType.DMA((_DEPTH,))] * 2
    vmem_bytes = (
        sum(
            _DEPTH
            * buffer_rows
            * _LANES
            * 4
            * (s.folds if isinstance(s, _RowsSpec) else 0)
            for s in specs
        )
        + _DEPTH * buffer_rows * _LANES * 4 * has_lanes
    )
    vmem_bytes += 2 * sum(_DEPTH * c[0].size * c.dtype.itemsize for c in caches)
    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    first_cache = 5 + len(inputs)
    return pl.pallas_call(
        functools.partial(
            _kernel,
            specs=tuple(specs),
            page_size=page_size,
            num_tokens=padded,
            has_lanes=has_lanes,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            grid=(1,),
            in_specs=[any_spec] * (len(inputs) + len(caches)),
            out_specs=[any_spec] * len(caches),
            scratch_shapes=scratch,
        ),
        out_shape=[jax.ShapeDtypeStruct(c.shape, c.dtype) for c in caches],
        input_output_aliases={first_cache + i: i for i in range(len(caches))},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=max(32 << 20, vmem_bytes + (16 << 20)),
        ),
        interpret=interpret,
        name="minimax_m3_msa_write_runs",
    )(num_runs[None], run_page, run_lo, run_hi, run_tok0, *inputs, *caches)


def token_runs(pages: jax.Array, slots: jax.Array, num_pages: int, max_runs: int):
    """Token runs: maximal spans of consecutive tokens writing one page;
    ``(count, page, first slot, end slot, first token)`` (unset past `count`).

    Compares of every run against every token (``[max_runs, T]``, fused into
    reductions) and two small gathers: XLA's scatters of each token to its run
    cost ~27 us each per step at 8K tokens (twins j-a4c2ab21, j-0a66fa2e).
    """
    num_tokens = pages.shape[0]
    if num_tokens == 0:
        zeros = jnp.zeros((max_runs,), jnp.int32)
        return jnp.int32(0), zeros, zeros, zeros, zeros
    valid = (pages >= 0) & (pages < num_pages)
    prev_page = jnp.concatenate([jnp.full((1,), -1, pages.dtype), pages[:-1]])
    prev_slot = jnp.concatenate([jnp.full((1,), -2, slots.dtype), slots[:-1]])
    start = valid & ((pages != prev_page) | (slots != prev_slot + 1))
    # Starts up to each token: token t is in run starts[t] - 1 (if valid).
    starts = jnp.cumsum(start.astype(jnp.int32))
    num_runs = jnp.minimum(starts[-1], max_runs)
    run = jnp.arange(max_runs, dtype=jnp.int32)[:, None]
    # Run r starts after every token with at most r starts up to it.
    run_tok0 = jnp.sum(starts[None, :] <= run, axis=1, dtype=jnp.int32)
    count = jnp.sum(
        valid[None, :] & (starts[None, :] == run + 1), axis=1, dtype=jnp.int32
    )
    first = jnp.minimum(run_tok0, num_tokens - 1)
    run_lo = slots[first].astype(jnp.int32)
    return num_runs, pages[first].astype(jnp.int32), run_lo, run_lo + count, run_tok0


@functools.partial(jax.jit, static_argnames=("max_runs", "bits", "interpret"))
def write_rows(
    cache: jax.Array,
    rows: jax.Array,
    pages: jax.Array,
    slots: jax.Array,
    *,
    max_runs: int,
    bits: int | None = None,
    scales: jax.Array | None = None,
    interpret: bool = False,
) -> jax.Array:
    """`write_runs` of one packed cache's `Rows`."""
    return write_runs(
        [Rows(cache, rows, bits, scales)],
        pages,
        slots,
        max_runs=max_runs,
        interpret=interpret,
    )[0]


@functools.partial(jax.jit, static_argnames=("max_runs", "interpret"))
def write_slot_scales(
    scale_cache: jax.Array,
    values: jax.Array,
    pages: jax.Array,
    slots: jax.Array,
    *,
    max_runs: int,
    interpret: bool = False,
) -> jax.Array:
    """`write_runs` of one f32 scale cache's `SlotValues`."""
    return write_runs(
        [SlotValues(scale_cache, values)],
        pages,
        slots,
        max_runs=max_runs,
        interpret=interpret,
    )[0]


def max_write_runs(num_tokens: int, max_num_seqs: int, page_size: int) -> int:
    """Runs of a batch: a request's q tokens span at most
    ``(q - 1) // page_size + 2`` pages."""
    return max(1, min(num_tokens, num_tokens // page_size + 2 * max_num_seqs))
