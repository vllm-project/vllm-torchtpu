# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact SparseCore top-k that scans only the highest-maximum key blocks.

Block maxima, reduced on SparseCore, rank the blocks first. Every key scoring
above the `blocks`-th largest block maximum lies in the top `blocks` blocks,
and each of those blocks holds a key at or above it (publication forces one
to +inf). With `blocks > k` their keys therefore contain a valid top-k, so
only they are gathered and radix-selected instead of the whole score row.

Rows that fit SparseCore VMEM stay resident. Wider rows (1M-token contexts)
stream: maxima come from chunked score reads, and only the chosen blocks are
fetched from HBM.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental import xla_metadata
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.deepseek_v41.index_selection.publication import Radix
from vllm_torchtpu.kernels.deepseek_v41.index_selection.streaming_topk import (
    RADIX_BUCKETS,
    _ordered_bits,
)

# Score keys per streamed read; see `_chunk`.
_STREAM_CHUNK = 8192


def default_blocks(k: int, lanes: int, forced: bool = True) -> int:
    """Lane-aligned blocks whose maxima give k true keys; a forced block adds one."""
    return pl.cdiv(k + forced, lanes) * lanes


def _length_words(rows: int) -> int:
    """One-dimensional lengths, padded to whole lane-aligned HBM rows."""
    lanes = pltpu.get_tpu_info().num_lanes
    return pl.cdiv(rows, lanes) * lanes


def _chunk(width: int) -> int:
    """Score keys per streamed read when the row is not resident.

    The last read ends at the row's end and re-reduces the keys it shares with
    the previous read, so the chunk need not divide the width. A divisor can be
    tiny: width 148,224 read 256 keys at a time, 930 us against 51 us resident.
    """
    return min(width, _STREAM_CHUNK)


def _vmem_words(
    rows, width, block_width, k, blocks, block_size, buffers, resident, extra=0
):
    # Resident rows keep whole score rows; streamed rows keep two chunks and
    # the chosen blocks' fetched keys.
    rows_words = (
        buffers * width if resident else 2 * _chunk(width) + blocks * block_size
    )
    return (
        rows_words
        + buffers * block_width
        + RADIX_BUCKETS
        + blocks
        + blocks * block_size
        + k
        + _length_words(rows)
        + extra
    )


def _records_words(rows, k, table_words):
    """VMEM of the record stage: page table, row requests, records, counts."""
    lanes = plsc.get_sparse_core_info().num_lanes
    return _length_words(table_words) + _length_words(rows) + k + lanes + lanes


def _plan(rows, width, block_width, k, blocks, block_size, extra=0):
    """(resident, buffers): resident rows when they fit, double-buffered if room."""
    capacity = plsc.get_sparse_core_info().vmem_capacity_bytes
    sizes = (rows, width, block_width, k, blocks, block_size)
    for resident in (True, False):
        for buffers in (2, 1):
            if 4 * _vmem_words(*sizes, buffers, resident, extra) <= capacity:
                return resident, buffers
    return None


def records_fit(rows: int, width: int, k: int, block_size: int, table_words: int):
    """Whether the record stage leaves `block_topk`'s VMEM plan unchanged."""
    lanes = pltpu.get_tpu_info().num_lanes
    blocks = default_blocks(k, plsc.get_sparse_core_info().num_lanes)
    block_width = pl.cdiv(pl.cdiv(width, block_size), lanes) * lanes
    sizes = (rows, width, block_width, k, blocks, block_size)
    extra = _records_words(rows, k, table_words)
    return _plan(*sizes) == _plan(*sizes, extra=extra)


def is_supported(rows: int, width: int, k: int, block_size: int) -> bool:
    """Lane-aligned shapes whose block maxima fit SparseCore VMEM."""
    if jax.default_backend() != "tpu":
        return False
    info = pltpu.get_tpu_info()
    sc = info.sparse_core
    if sc is None or rows < 1 or block_size < 1 or sc.num_lanes % block_size:
        return False
    alignment = info.num_lanes
    if width % alignment or k % alignment or not 0 < k < width:
        return False
    blocks = default_blocks(k, sc.num_lanes)
    block_width = pl.cdiv(pl.cdiv(width, block_size), alignment) * alignment
    plan = _plan(rows, width, block_width, k, blocks, block_size)
    span = sc.num_lanes * block_size
    streamable = width % span == 0 and _chunk(width) % span == 0
    return blocks * block_size < width and plan is not None and (plan[0] or streamable)


# Not jitted: an inner jit boundary would drop the scheduling-group metadata.
def block_topk(
    scores: jax.Array,
    lengths: jax.Array,
    *,
    k: int,
    block_size: int,
    publish: int = 0,
    records: tuple[jax.Array, jax.Array, jax.Array, int] | None = None,
    scheduling_group_id: int | None = None,
) -> jax.Array | tuple[jax.Array, ...]:
    """Return ascending top-k key IDs per row, padded with -1.

    Args:
        scores: FP32 [rows, width]; keys at or past a row's length are -inf.
        lengths: Int32 [rows] visible keys, clipped to [0, width].
        k: Selected keys per row. Negative infinity is never selected.
        block_size: Keys per block maximum.
        publish: Also return the IDs of the `publish` highest block maxima,
            the newest visible block forced first; ascending, -1 padded, and
            ties keep the lowest IDs.
        records: Also return each row's physical records and valid count:
            (page table INT32 [requests, columns], row requests INT32
            [rows], visible INT32 [1], page size). A record is
            `table[request, id // page] * page + id % page`, -1 for -1 IDs;
            with `visible` nonzero the IDs are the row's visible keys
            `0..length-1` instead of the selection. Counts are [rows, lanes],
            each lane the row's count. `records_fit` bounds the table.
        scheduling_group_id: Optional XLA scheduling-group ID.
    """
    rows, width = scores.shape
    sc = plsc.get_sparse_core_info()
    lanes, alignment = sc.num_lanes, pltpu.get_tpu_info().num_lanes
    if (
        scores.dtype != jnp.float32
        or lengths.shape != (rows,)
        or not is_supported(rows, width, k, block_size)
        or publish % lanes
    ):
        raise ValueError("Unsupported block top-k shapes or dtypes")
    blocks = default_blocks(k, lanes, forced=bool(publish))
    candidates = blocks * block_size
    block_width = pl.cdiv(pl.cdiv(width, block_size), alignment) * alignment
    lengths = jnp.clip(lengths.astype(jnp.int32), 0, width)
    # 1-D: a [rows, lanes] buffer pads each row to 128 lanes in VMEM.
    token_lengths = jnp.pad(lengths, (0, _length_words(rows) - rows))
    if records is not None:
        table, row_requests, visible, page_size = records
        columns = table.shape[1]
        if (
            table.dtype != jnp.int32
            or row_requests.shape != (rows,)
            or visible.shape != (1,)
            or page_size & (page_size - 1)
            or not records_fit(rows, width, k, block_size, table.size)
        ):
            raise ValueError("Unsupported block top-k records")
        page_shift = page_size.bit_length() - 1
    plan = _plan(rows, width, block_width, k, blocks, block_size)
    assert plan is not None
    resident, buffers = plan
    chunk = _chunk(width)
    # Streamed rows read scores in two alternating chunks.
    score_words = buffers * width if resident else 2 * chunk
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core",
        subcore_axis_name="subcore",
        num_cores=min(sc.num_cores, pl.cdiv(rows, sc.num_subcores)),
    )
    out_types = (jax.ShapeDtypeStruct((rows, k), jnp.int32),)
    if publish:
        out_types += (jax.ShapeDtypeStruct((rows, publish), jnp.int32),)
    record_scratch = {}
    if records is not None:
        out_types += (
            jax.ShapeDtypeStruct((rows, k), jnp.int32),
            jax.ShapeDtypeStruct((rows, lanes), jnp.int32),
        )
        record_scratch = dict(
            table_words=pltpu.VMEM((_length_words(table.size),), jnp.int32),
            requests=pltpu.VMEM((_length_words(rows),), jnp.int32),
            record_row=pltpu.VMEM((k,), jnp.int32),
            count_row=pltpu.VMEM((lanes,), jnp.int32),
            visible_lanes=pltpu.VMEM((lanes,), jnp.int32),
            record_sems=pltpu.SemaphoreType.DMA((2,)),
            load_sems=pltpu.SemaphoreType.DMA((3,)),
        )

    @pl.kernel(
        out_type=out_types,
        mesh=mesh,
        scratch_types=dict(
            score_buffers=pltpu.VMEM((score_words,), jnp.float32),
            maxima=pltpu.VMEM((block_width,), jnp.float32),
            histogram=pltpu.VMEM((RADIX_BUCKETS,), jnp.int32),
            block_ids=pltpu.VMEM((blocks,), jnp.int32),
            # Ordered key bits; first holds the published block IDs.
            keys=pltpu.VMEM((max(candidates, publish),), jnp.int32),
            output=pltpu.VMEM((k,), jnp.int32),
            limits=pltpu.VMEM((_length_words(rows),), jnp.int32),
            # Streamed rows: the chosen blocks' keys, fetched from HBM.
            fetched=pltpu.VMEM((lanes if resident else candidates,), jnp.float32),
            score_sems=pltpu.SemaphoreType.DMA((2,)),
            output_sem=pltpu.SemaphoreType.DMA,
            length_sem=pltpu.SemaphoreType.DMA,
            publish_sem=pltpu.SemaphoreType.DMA,
            fetch_sem=pltpu.SemaphoreType.DMA,
            **record_scratch,
        ),
        compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
        name="block_topk",
    )
    def kernel(
        score_ref,
        length_ref,
        *refs,
        score_buffers,
        maxima,
        histogram,
        block_ids,
        keys,
        output,
        limits,
        fetched,
        score_sems,
        output_sem,
        length_sem,
        publish_sem,
        fetch_sem,
        **record_refs,
    ):
        if records is not None:
            table_ref, request_ref, visible_ref, *refs = refs
        out_ref, *refs = refs
        publish_refs = refs[:1] if publish else ()
        record_out, count_out = refs[len(publish_refs) :] or (None, None)
        radix = Radix(histogram, lanes)
        strided = [radix.lane_ids * block_size + j for j in range(block_size)]
        first = jax.lax.axis_index("core") * mesh.num_subcores
        first += jax.lax.axis_index("subcore")
        stride = mesh.num_cores * mesh.num_subcores

        span = lanes * block_size

        def block_max(window):
            """Lane j reduces block j of a `span`-key window. The stride-
            `block_size` offsets are constant; SC loads reject strided slices."""
            values = plsc.load_gather(window, [strided[0]])
            for offset in strided[1:]:
                values = jnp.maximum(values, plsc.load_gather(window, [offset]))
            return values

        def row_copy(row_id, slot):
            """The copy of a resident row's scores into its buffer slot."""
            return pltpu.make_async_copy(
                score_ref.at[row_id],
                score_buffers.at[pl.ds(slot * width, width)],
                score_sems.at[slot],
            )

        def start_row(row_id, slot):
            if resident:
                row_copy(row_id, slot).start()

        def stream_maxima(row_id, tokens):
            """Reduce the visible chunks' block maxima, reading two chunks ahead."""
            reads = pl.cdiv(tokens, chunk)

            def offset(c):
                # Span-aligned: streamed widths and chunks are span multiples.
                return jnp.minimum(c * chunk, width - chunk)

            def read(c, s):
                return pltpu.make_async_copy(
                    score_ref.at[row_id, pl.ds(offset(c), chunk)],
                    score_buffers.at[pl.ds(s * chunk, chunk)],
                    score_sems.at[s],
                )

            pl.when(reads > 0)(lambda: read(0, 0).start())

            @pl.loop(0, reads)
            def reduce_chunk(c):
                s = c % 2
                read(c, s).wait()
                pl.when(c + 1 < reads)(lambda: read(c + 1, 1 - s).start())

                @plsc.parallel_loop(0, chunk // span, unroll=8)
                def reduce(i):
                    window = score_buffers.at[pl.ds(s * chunk + i * span, span)]
                    start = offset(c) // block_size + i * lanes
                    maxima[pl.ds(start, lanes)] = block_max(window)

        def fetch_blocks(row_id, count):
            """DMA the keys of the first `count` chosen blocks into `fetched`."""

            def copy(b, owner):
                return pltpu.make_async_copy(
                    score_ref.at[row_id, pl.ds(owner * block_size, block_size)],
                    fetched.at[pl.ds(b * block_size, block_size)],
                    fetch_sem,
                )

            # One vector load of `lanes` owners per group instead of one
            # broadcast gather per block (528 blocks: 57.9 -> 52.3 us a row).
            @pl.loop(0, pl.cdiv(count, lanes))
            def start(group):
                owners = block_ids[pl.ds(group * lanes, lanes)]
                for j in range(lanes):

                    @pl.when(group * lanes + j < count)
                    def _(j=j):
                        copy(group * lanes + j, owners[j]).start()

            @pl.loop(0, count)
            def land(b):
                copy(0, 0).wait()

        length_copy = pltpu.async_copy(length_ref, limits, length_sem)
        pl.when(first < rows)(lambda: start_row(first, 0))
        if records is not None:
            loads = [
                pltpu.async_copy(src, record_refs[dst], record_refs["load_sems"].at[i])
                for i, (src, dst) in enumerate(
                    (
                        (table_ref, "table_words"),
                        (request_ref, "requests"),
                        (visible_ref, "visible_lanes"),
                    )
                )
            ]
        length_copy.wait()
        if records is not None:
            for load in loads:
                load.wait()

        def record_copies(row_id):
            return (
                pltpu.make_async_copy(
                    record_refs["record_row"],
                    record_out.at[row_id],
                    record_refs["record_sems"].at[0],
                ),
                pltpu.make_async_copy(
                    record_refs["count_row"],
                    count_out.at[row_id],
                    record_refs["record_sems"].at[1],
                ),
            )

        def emit_records(row_id, tokens):
            """Physical records of the row's IDs, as `prepare_main_records`."""
            request = plsc.load_gather(
                record_refs["requests"], [jnp.full(lanes, row_id, jnp.int32)]
            )
            visible = record_refs["visible_lanes"][...] != 0
            base = request * columns

            @plsc.parallel_loop(
                0, k // lanes, carry=jnp.zeros(lanes, jnp.int32), unroll=4
            )
            def lookup(i, count):
                positions = radix.positions(i)
                selected = output[pl.ds(i * lanes, lanes)]
                ids = jnp.where(
                    visible, jnp.where(positions < tokens, positions, -1), selected
                )
                valid = ids >= 0
                logical = jnp.maximum(ids, 0)
                page = jnp.minimum(logical >> page_shift, columns - 1)
                physical = plsc.load_gather(record_refs["table_words"], [base + page])
                record = (physical << page_shift) + (logical & (page_size - 1))
                record_refs["record_row"][pl.ds(i * lanes, lanes)] = jnp.where(
                    valid, record, -1
                )
                return count + plsc.all_reduce_population_count(valid)

            record_refs["count_row"][...] = lookup

        @pl.loop(first, rows, step=stride)
        def row(row_id):
            slot = ((row_id - first) // stride) % buffers
            if resident:
                row_copy(row_id, slot).wait()
            if buffers == 2:
                pl.when(row_id + stride < rows)(
                    lambda: start_row(row_id + stride, 1 - slot)
                )
            tokens = plsc.load_gather(limits, [jnp.full(lanes, row_id, jnp.int32)])[0]

            # Stage 1: the top `blocks` blocks by maximum, in block order.
            live_blocks = pl.cdiv(tokens, block_size)
            block_chunks = pl.cdiv(live_blocks, lanes)
            base = slot * width
            if not resident:
                stream_maxima(row_id, tokens)
            radix.clear()

            def block_values(i):
                if not resident:
                    return maxima[pl.ds(i * lanes, lanes)]
                # Blocks past the length read in-row keys; the caller masks them.
                return block_max(score_buffers.at[pl.ds(base + i * span, span)])

            @plsc.parallel_loop(0, block_chunks, unroll=8)
            def prepare_blocks(i):
                positions = radix.positions(i)
                values = block_values(i)
                if publish:
                    # Publication rule: the newest visible block always wins.
                    values = jnp.where(positions == live_blocks - 1, jnp.inf, values)
                live = positions < live_blocks
                bits = _ordered_bits(jnp.where(live, values, -jnp.inf))
                maxima[pl.ds(i * lanes, lanes)] = bits.view(jnp.float32)
                radix.add(bits)

            publish_copy = None
            if publish:
                (publish_ref,) = publish_refs
                # Staged in `keys`, which stage 2 overwrites after the copy.
                published = keys.at[pl.ds(0, publish)]
                top = radix.threshold(maxima, block_chunks, publish)
                radix.emit(maxima, block_chunks, publish, top, published)
                publish_copy = pltpu.async_copy(
                    published, publish_ref.at[row_id], publish_sem
                )
                # The selection consumed the histogram; refill its top byte.
                radix.clear()

                @plsc.parallel_loop(0, block_chunks, unroll=8)
                def refill(i):
                    radix.add(maxima[pl.ds(i * lanes, lanes)].view(jnp.uint32))

            chosen = radix.threshold(maxima, block_chunks, blocks)
            radix.emit(maxima, block_chunks, blocks, chosen, block_ids)

            # Stage 2: the chosen blocks' keys, from the resident row or HBM.
            def owners(i):
                candidate = radix.positions(i)
                owner = plsc.load_gather(block_ids, [candidate // block_size])
                return owner, owner * block_size + candidate % block_size

            if publish_copy is not None:
                publish_copy.wait()
            chosen_blocks = jnp.minimum(chosen[0], blocks)
            key_chunks = pl.cdiv(chosen_blocks * block_size, lanes)
            if not resident:
                fetch_blocks(row_id, chosen_blocks)
            radix.clear()

            @plsc.parallel_loop(0, key_chunks, unroll=4)
            def gather_keys(i):
                owner, ids = owners(i)
                valid = (owner >= 0) & (ids < tokens)
                if resident:
                    index = base + jnp.where(valid, ids, 0)
                    values = plsc.load_gather(score_buffers, [index])
                else:
                    values = fetched[pl.ds(i * lanes, lanes)]
                bits = _ordered_bits(jnp.where(valid, values, -jnp.inf))
                keys[pl.ds(i * lanes, lanes)] = bits.view(jnp.int32)
                radix.add(bits)

            selection = radix.threshold(keys, key_chunks, k)

            @pl.when(row_id > first)
            def wait_previous_output():
                pltpu.make_async_copy(
                    output, out_ref.at[row_id - stride], output_sem
                ).wait()

            radix.emit(keys, key_chunks, k, selection, output, lambda i: owners(i)[1])
            if records is not None:

                @pl.when(row_id > first)
                def wait_previous_records():
                    for copy in record_copies(row_id - stride):
                        copy.wait()

                emit_records(row_id, tokens)
                for copy in record_copies(row_id):
                    copy.start()
            pltpu.async_copy(output, out_ref.at[row_id], output_sem)
            if buffers == 1:
                pl.when(row_id + stride < rows)(lambda: start_row(row_id + stride, 0))

        @pl.when(first < rows)
        def drain():
            last_row = first + ((rows - 1 - first) // stride) * stride
            pltpu.make_async_copy(output, out_ref.at[last_row], output_sem).wait()
            if records is not None:
                for copy in record_copies(last_row):
                    copy.wait()

    inputs = (scores, token_lengths)
    if records is not None:
        inputs += (
            jnp.pad(table.reshape(-1), (0, _length_words(table.size) - table.size)),
            jnp.pad(row_requests, (0, _length_words(rows) - rows)),
            jnp.broadcast_to(visible, (lanes,)),
        )
    result = kernel(*inputs)
    if scheduling_group_id is not None:
        result = jax.tree.map(
            lambda x: xla_metadata.set_xla_metadata(
                x, _scheduling_group_id=scheduling_group_id
            ),
            result,
        )
    if records is not None:
        return result
    return result if publish else result[0]
