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
from collections.abc import Callable, Sequence
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
# j-efdac848). At 4 a 4-bit decode run still took ~0.37 us for a ~360-bundle
# body (c48 decode twin j-c22c589e: 37 runs, 13.9 us per layer).
_DEPTH = 8
# Rows of an 8-bit sublane tile: row windows start at a multiple.
_ROW_ALIGN = 32
_LANES = 128


class Rows(NamedTuple):
    """One token row per slot of a packed ``[pages, S / p (+ 1), p, W]`` cache.

    ``rows [T, W]`` are in the cache dtype, or uint8 nibbles below 16 with
    ``bits=4`` (the 4-bit caches of `msa_fp4`, 8 slots per 32-bit word).
    `scales` (f32 ``[T]``) go to the last word row of a page with folded
    scales (`msa_fp4.scales_in_page`), lane = slot.

    With `dim_major` the cache keeps its pages ``[W / p (+ 1), p, S]``: a
    token's row is the slot's lane, ``p`` of its values per 32-bit word
    (the index cache, `msa_indexer.dim_major_rows`).
    """

    cache: jax.Array
    rows: jax.Array
    bits: int | None = None
    scales: jax.Array | None = None
    dim_major: bool = False


class SlotValues(NamedTuple):
    """Per-slot f32 ``values [T, n]`` into rows ``[0, n)`` of an f32 ``[pages,
    rows, S]`` cache, lane = slot.

    `lanes` (uint32 ``[T, 128]``), when the producer of the values made it,
    is the write's whole per-slot lane block as `write_operands` lays it
    out: the bits of ``values`` in lanes ``[0, n)``, of the folded scales of
    the `Rows` targets from the last lane down, zeros elsewhere. It is then
    read as is."""

    cache: jax.Array
    values: jax.Array
    lanes: jax.Array | None = None


class _RowsSpec(NamedTuple):
    bits: int
    folds: int  # 128-lane blocks of a row
    data_rows: int  # word rows holding slots (dim-major: a slot's words)
    scale_lane: int | None  # lane of the lane values holding folded scales
    dim_major: bool = False  # slot = lane; a token's row comes packed in words


class _ValuesSpec(NamedTuple):
    num_values: int  # lanes [0, num_values) of the lane values


class RunOps(NamedTuple):
    """One write's per-run steps (`_run_ops`), each taking a run and the
    buffer slot it uses, and its merged pages."""

    in_copies: Callable  # the run's batch rows and pages, HBM -> VMEM
    out_copies: Callable  # the run's merged pages, VMEM -> its caches
    merge: Callable  # the run's rows merged into its pages
    out_bufs: list  # [slots, ...page] merged pages, one buffer per cache


def _run_ops(
    run_refs,
    refs,
    *,
    specs: tuple,
    page_size: int,
    num_tokens: int,
    has_lanes: bool,
):
    """`RunOps` of one write over its refs.

    ``run_refs`` are the run tables ``(page, lo, hi, tok0)`` (`token_runs`);
    ``refs`` the write's operands in order: in (HBM), each `Rows` target's
    rows ``[T, W]`` as unsigned ints of its slot width (8 bits for 4-bit
    slots), the lane values ``[T, 128]`` (if any), then the caches; out: the
    caches, aliased; then the scratch of `_write_operands`.

    A run's tokens ``tok0 + i`` take slots ``lo + i`` of its page: slot ``s``
    reads batch row ``tok0 - lo + s``. Each run copies in a window of
    ``S + 32`` batch rows holding all of its tokens (clamped into the batch;
    rows past its tokens are never selected), widened to 32 bits at row
    ``S`` of a ``[W / 128, 2S + window + 8, 128]`` staging buffer (Mosaic's
    strided loads need 32-bit 128-lane rows), so slot 0's row there is
    ``S + tok0 - lo - start`` >= 1."""
    run_page_ref, run_lo_ref, run_hi_ref, run_tok0_ref = run_refs
    num_rows = sum(isinstance(s, _RowsSpec) for s in specs)
    it = iter(refs)
    words_hbm = [next(it) for _ in range(num_rows)]
    lanes_hbm = next(it) if has_lanes else None
    for _ in specs:  # the aliased inputs: `caches` below are the same buffers
        next(it)
    caches = [next(it) for _ in specs]
    windows = [next(it) for _ in range(num_rows)]
    staging = [next(it) for _ in range(num_rows)]
    lane_window = next(it) if has_lanes else None
    in_bufs = [next(it) for _ in specs]
    out_bufs = [next(it) for _ in specs]
    in_sem, out_sem = next(it), next(it)
    window = _window(page_size)

    def placement(r):
        base = run_tok0_ref[r] - run_lo_ref[r]
        start = jnp.minimum(
            jnp.maximum(base, 0) // _ROW_ALIGN * _ROW_ALIGN, num_tokens - window
        )
        return pl.multiple_of(start, _ROW_ALIGN), page_size + base - start

    def in_copies(r, slot):
        start, _ = placement(r)
        page = run_page_ref[r]
        rows = pl.ds(start, window)
        copies = [
            pltpu.make_async_copy(w.at[rows], buf.at[slot], in_sem.at[slot])
            for w, buf in zip(words_hbm, windows)
        ]
        if has_lanes:
            copies.append(
                pltpu.make_async_copy(
                    lanes_hbm.at[rows],
                    lane_window.at[slot, pl.ds(page_size, window)],
                    in_sem.at[slot],
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
        windows_left = iter(zip(windows, staging))
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
            raw, buf = next(windows_left)
            new_rows = raw[slot].astype(jnp.uint32)
            if spec.bits == 4 and not spec.dim_major:
                new_rows = new_rows & 0xF
            for f in range(spec.folds):
                buf[f, pl.ds(page_size, window)] = new_rows[
                    :, f * _LANES : (f + 1) * _LANES
                ]
            words = in_buf.bitcast(jnp.uint32)[slot, :, 0, :]
            rows, width = words.shape
            packing = 32 // spec.bits
            row = jax.lax.broadcasted_iota(jnp.int32, (rows, width), 0)
            lane = jax.lax.broadcasted_iota(jnp.int32, (rows, width), 1)
            if spec.dim_major:
                # [slot, word] of the run's packed rows -> [word, slot].
                by_lane = buf[0, pl.ds(row0, page_size), :].T
                data = row < spec.data_rows
                words = jnp.where(data & (lane >= lo) & (lane < hi), by_lane[:rows], words)
                # Slots past the run are zeroed, as below.
                words = jnp.where(data & (lane >= hi), 0, words)
            for j in range(0 if spec.dim_major else packing):
                slot_j = row * packing + j  # past the data rows: >= S, never taken
                take = (slot_j >= lo) & (slot_j < hi)
                keep = jnp.uint32(
                    ~(((1 << spec.bits) - 1) << (spec.bits * j)) & 0xFFFFFFFF
                )
                # Slots j, p + j, ...: rows at stride p.
                new = jnp.concatenate(
                    [
                        buf.at[f][pl.ds(row0 + j, spec.data_rows, stride=packing)]
                        for f in range(spec.folds)
                    ],
                    axis=1,
                )
                if spec.data_rows < rows:
                    new = jnp.pad(new, ((0, rows - spec.data_rows), (0, 0)))
                merged = (words & keep) | (new << jnp.uint32(spec.bits * j))
                words = jnp.where(take, merged, words)
                # Slots past the run are zeroed: an unwritten slot may hold
                # any bytes (NaN), which readers then need not mask.
                past = (slot_j >= hi) & (row < spec.data_rows)
                words = jnp.where(past, words & keep, words)
            if spec.scale_lane is not None:  # last word row: one scale per slot (lane)
                bits = by_slot[spec.scale_lane : spec.scale_lane + 1]
                if width > _LANES:
                    bits = jnp.pad(bits, ((0, 0), (0, width - _LANES)))
                take = (row == rows - 1) & (lane >= lo) & (lane < hi)
                words = jnp.where(take, bits[:, :width], words)
                # Unwritten slots' scales too: the decode indexer multiplies
                # every slot's (zeroed) key score by its scale unmasked.
                words = jnp.where((row == rows - 1) & (lane >= hi), 0, words)
            out_buf.bitcast(jnp.uint32)[slot, :, 0, :] = words

    return RunOps(in_copies, out_copies, merge, out_bufs)


def _kernel(
    # Scalar prefetch
    num_runs_ref,  # [1]
    run_page_ref,  # [R] page of each run
    run_lo_ref,  # [R] first slot of each run within its page
    run_hi_ref,  # [R] one past the last slot
    run_tok0_ref,  # [R] first token of each run
    *refs,
    **kwargs,
):
    """`write_runs`: each run's page copied in, merged and copied back, with
    up to `_DEPTH` runs' copies in flight (`_run_ops`)."""
    in_copies, out_copies, merge, _ = _run_ops(
        (run_page_ref, run_lo_ref, run_hi_ref, run_tok0_ref), refs, **kwargs
    )
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


def _window(page_size: int) -> int:
    """Batch rows a run copies: its page's slots and the alignment slack,
    a multiple of `_ROW_ALIGN` so that a window clamped to the batch's end
    starts aligned too."""
    return -(-page_size // _ROW_ALIGN) * _ROW_ALIGN + _ROW_ALIGN


def _slot_words(target: Rows) -> tuple[jax.Array, _RowsSpec, int]:
    """A `Rows` target's rows ``[T, W]`` as unsigned ints of its slot width
    (uint8 nibbles for 4-bit slots; the kernel widens a run's window to 32
    bits, so XLA moves the rows at their own width), its spec (no scale lane
    yet) and its page's slot count."""
    cache = target.cache
    _, rows_per_page, word_bytes, width = cache.shape
    bits = target.bits or 32 // word_bytes
    folded = target.scales is not None
    data_rows = rows_per_page - 1 if folded else rows_per_page
    if folded and not (
        bits == 4 and msa_fp4.scales_in_page(rows_per_page, cache.dtype)
    ):
        raise ValueError(
            f"scales need a 4-bit cache with a scale row, got {cache.shape}"
        )
    if width % _LANES:
        raise ValueError(f"cache rows must be whole 128-lane blocks, got {cache.shape}")
    if target.dim_major:
        if width != _LANES:
            raise ValueError(f"dim-major pages hold {_LANES} slots, got {cache.shape}")
        words = _packed_words(target.rows, bits)
        return words, _RowsSpec(bits, 1, data_rows, None, True), width
    if bits == 4:
        words = target.rows.astype(jnp.uint8)
    else:
        words = jax.lax.bitcast_convert_type(target.rows, _UINT_BY_BITS[bits])
    spec = _RowsSpec(bits, width // _LANES, data_rows, None)
    return words, spec, data_rows * 32 // bits


def _packed_words(rows: jax.Array, bits: int) -> jax.Array:
    """Token rows ``[T, W]`` (uint8 nibbles for 4 bits) -> uint32 ``[T,
    128]``: value ``i`` in bits ``bits * (i % p)`` of word ``i // p`` (``p =
    32 / bits``), the words of a dim-major page column; lanes past the
    row's words are zero."""
    num_tokens, width = rows.shape
    packing = 32 // bits
    if bits != 4:
        rows = jax.lax.bitcast_convert_type(rows, _UINT_BY_BITS[bits])
    # Shifts and ors: one elementwise fusion (a reshaping bitcast to 32 bits
    # became one slice per packed value, ~15 us each for 3K fp8 rows).
    values = rows.astype(jnp.uint32).reshape(num_tokens, width // packing, packing)
    shifts = jnp.arange(packing, dtype=jnp.uint32) * bits
    words = jnp.bitwise_or.reduce(values << shifts, axis=-1)
    if words.shape[1] > _LANES:
        raise ValueError(f"a dim-major column holds {_LANES} words, got {words.shape}")
    return jnp.pad(words, ((0, 0), (0, _LANES - words.shape[1])))


class WriteOperands(NamedTuple):
    """A `write_runs` call's operands (`write_operands`)."""

    runs: tuple  # (count, page, lo, hi, tok0) of each run (`token_runs`)
    inputs: list  # HBM: each `Rows` target's rows, then the lane values if any
    caches: list  # the targets' caches (aliased to the outputs)
    scratch: list  # scratch shapes, `depth` buffer slots
    vmem_bytes: int  # VMEM the scratch takes
    kwargs: dict  # `_run_ops` keyword arguments


def write_operands(
    targets: Sequence[Rows | SlotValues],
    pages: jax.Array,
    slots: jax.Array,
    *,
    max_runs: int,
    depth: int = _DEPTH,
) -> WriteOperands:
    """`write_runs`' operands, with `depth` page buffer slots (arguments as
    `write_runs`), for its kernel or one that folds the write in
    (`_run_ops`)."""
    num_tokens = pages.shape[0]
    num_pages = targets[0].cache.shape[0]
    if any(t.cache.shape[0] != num_pages for t in targets):
        raise ValueError("write_runs targets must share their pages")
    num_runs, run_page, run_lo, run_hi, run_tok0 = token_runs(
        pages, slots, num_pages, max_runs
    )
    words, specs, scale_lanes = [], [], []
    lane_values = None
    built_lanes = None
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
            if target.lanes is not None:
                built_lanes = target.lanes
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
    window = _window(page_size)
    padded = max(-(-num_tokens // _ROW_ALIGN) * _ROW_ALIGN, window)
    pad = ((0, padded - num_tokens), (0, 0))
    inputs = [jnp.pad(w, pad) for w in words]
    if built_lanes is not None:
        # A producer's lanes place one folded-scale target's in the last lane.
        if len(scale_lanes) != 1:
            raise ValueError("built lanes hold one target's folded scales")
        if built_lanes.shape != (num_tokens, _LANES) or built_lanes.dtype != jnp.uint32:
            raise ValueError(
                f"uint32 lanes [{num_tokens}, {_LANES}], got {built_lanes.dtype} "
                f"{built_lanes.shape}"
            )
        inputs.append(jnp.pad(built_lanes, pad))
    elif has_lanes:  # one elementwise fusion: no column concatenates
        if lane_values is None:
            lanes = jnp.zeros((num_tokens, _LANES), jnp.uint32)
        else:
            lanes = jnp.pad(lane_values, ((0, 0), (0, _LANES - lane_values.shape[1])))
        lane = jnp.arange(_LANES, dtype=jnp.int32)
        for i, bits in scale_lanes:
            lanes = jnp.where(lane == i, bits[:, None], lanes)
        inputs.append(jnp.pad(lanes, pad))
    caches = [t.cache for t in targets]
    buffer_rows = 2 * page_size + window + 8
    scratch = [pltpu.VMEM((depth, window) + w.shape[1:], w.dtype) for w in words]
    scratch += [
        pltpu.VMEM((w.shape[1] // _LANES, buffer_rows, _LANES), jnp.uint32)
        for w in words
    ]
    if has_lanes:
        scratch.append(pltpu.VMEM((depth, buffer_rows, _LANES), jnp.uint32))
    pages_scratch = [pltpu.VMEM((depth,) + c.shape[1:], c.dtype) for c in caches]
    scratch += pages_scratch + pages_scratch
    scratch += [pltpu.SemaphoreType.DMA((depth,))] * 2
    vmem_bytes = sum(
        (depth * window * w.dtype.itemsize + buffer_rows * 4) * w.shape[1]
        for w in words
    )
    vmem_bytes += depth * buffer_rows * _LANES * 4 * has_lanes
    vmem_bytes += 2 * sum(depth * c[0].size * c.dtype.itemsize for c in caches)
    return WriteOperands(
        (num_runs, run_page, run_lo, run_hi, run_tok0),
        inputs,
        caches,
        scratch,
        vmem_bytes,
        dict(
            specs=tuple(specs),
            page_size=page_size,
            num_tokens=padded,
            has_lanes=has_lanes,
        ),
    )


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

    A `Rows` target's data slots past each run's last token are zeroed, and
    so are their folded scales: the 8-bit decode attention
    (`msa_attention_lse._kernel`) multiplies a page's unseen values by p = 0
    unmasked, its prefill tiles (`DIRECT_T`) feed the keys to the MXU
    unmasked, and the decode indexer (`msa_indexer._chunk_maxima_t`) scales
    every slot's score, so no slot they read may hold NaN bytes. Any other
    writer of these caches must keep that invariant.

    The caches share the page geometry (same pages, slots per page). Tokens
    with a page outside the caches are dropped. Consecutive tokens of a run
    must take consecutive slots of one page, as a request's new tokens do;
    a run boundary is wherever that breaks, and no two runs may share a page
    (the kernel copies a run's page in while earlier runs' pages are still
    being written back; tokens map to pages by whole blocks, so a request's
    tokens in one page are one run).
    `max_runs` bounds the number of runs (see `max_write_runs`).
    """
    ops = write_operands(targets, pages, slots, max_runs=max_runs)
    num_runs, *run_tables = ops.runs
    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    first_cache = 5 + len(ops.inputs)
    return pl.pallas_call(
        functools.partial(_kernel, **ops.kwargs),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            grid=(1,),
            in_specs=[any_spec] * (len(ops.inputs) + len(ops.caches)),
            out_specs=[any_spec] * len(ops.caches),
            scratch_shapes=ops.scratch,
        ),
        out_shape=[jax.ShapeDtypeStruct(c.shape, c.dtype) for c in ops.caches],
        input_output_aliases={first_cache + i: i for i in range(len(ops.caches))},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=max(32 << 20, ops.vmem_bytes + (16 << 20)),
        ),
        interpret=interpret,
        name="minimax_m3_msa_write_runs",
    )(num_runs[None], *run_tables, *ops.inputs, *ops.caches)


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


@functools.partial(
    jax.jit, static_argnames=("max_runs", "bits", "dim_major", "interpret")
)
def write_rows(
    cache: jax.Array,
    rows: jax.Array,
    pages: jax.Array,
    slots: jax.Array,
    *,
    max_runs: int,
    bits: int | None = None,
    scales: jax.Array | None = None,
    dim_major: bool = False,
    interpret: bool = False,
) -> jax.Array:
    """`write_runs` of one packed cache's `Rows`."""
    return write_runs(
        [Rows(cache, rows, bits, scales, dim_major)],
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
