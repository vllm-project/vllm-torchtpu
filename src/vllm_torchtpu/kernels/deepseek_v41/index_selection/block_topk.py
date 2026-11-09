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

import math

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


def default_blocks(k: int, lanes: int, forced: bool = True) -> int:
    """Lane-aligned blocks whose maxima give k true keys; a forced block adds one."""
    return pl.cdiv(k + forced, lanes) * lanes


def _length_words(rows: int) -> int:
    """One-dimensional lengths, padded to whole lane-aligned HBM rows."""
    lanes = pltpu.get_tpu_info().num_lanes
    return pl.cdiv(rows, lanes) * lanes


def _chunk(width: int) -> int:
    """Score keys per streamed read when the row is not resident."""
    return math.gcd(width, 8192)


def _vmem_words(rows, width, block_width, k, blocks, block_size, buffers, resident):
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
    )


def _plan(rows, width, block_width, k, blocks, block_size):
    """(resident, buffers): resident rows when they fit, double-buffered if room."""
    capacity = plsc.get_sparse_core_info().vmem_capacity_bytes
    sizes = (rows, width, block_width, k, blocks, block_size)
    for resident in (True, False):
        for buffers in (2, 1):
            if 4 * _vmem_words(*sizes, buffers, resident) <= capacity:
                return resident, buffers
    return None


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
    streamable = _chunk(width) % (sc.num_lanes * block_size) == 0
    return blocks * block_size < width and plan is not None and (plan[0] or streamable)


# Not jitted: an inner jit boundary would drop the scheduling-group metadata.
def block_topk(
    scores: jax.Array,
    lengths: jax.Array,
    *,
    k: int,
    block_size: int,
    publish: int = 0,
    scheduling_group_id: int | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Return ascending top-k key IDs per row, padded with -1.

    Args:
        scores: FP32 [rows, width]; keys at or past a row's length are -inf.
        lengths: Int32 [rows] visible keys, clipped to [0, width].
        k: Selected keys per row. Negative infinity is never selected.
        block_size: Keys per block maximum.
        publish: Also return the IDs of the `publish` highest block maxima,
            the newest visible block forced first; ascending, -1 padded, and
            ties keep the lowest IDs.
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
        ),
        compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
        name="block_topk",
    )
    def kernel(
        score_ref,
        length_ref,
        out_ref,
        *publish_refs,
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
    ):
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

            def read(c, s):
                return pltpu.make_async_copy(
                    score_ref.at[row_id, pl.ds(c * chunk, chunk)],
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
                    start = c * (chunk // block_size) + i * lanes
                    maxima[pl.ds(start, lanes)] = block_max(window)

        def fetch_blocks(row_id, count):
            """DMA the keys of the first `count` chosen blocks into `fetched`."""

            def copy(b, owner):
                return pltpu.make_async_copy(
                    score_ref.at[row_id, pl.ds(owner * block_size, block_size)],
                    fetched.at[pl.ds(b * block_size, block_size)],
                    fetch_sem,
                )

            @pl.loop(0, count)
            def start(b):
                owner = plsc.load_gather(block_ids, [jnp.full(lanes, b, jnp.int32)])
                copy(b, owner[0]).start()

            @pl.loop(0, count)
            def land(b):
                copy(0, 0).wait()

        length_copy = pltpu.async_copy(length_ref, limits, length_sem)
        pl.when(first < rows)(lambda: start_row(first, 0))
        length_copy.wait()

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
            pltpu.async_copy(output, out_ref.at[row_id], output_sem)
            if buffers == 1:
                pl.when(row_id + stride < rows)(lambda: start_row(row_id + stride, 0))

        @pl.when(first < rows)
        def drain():
            last_row = first + ((rows - 1 - first) // stride) * stride
            pltpu.make_async_copy(output, out_ref.at[last_row], output_sem).wait()

    result = kernel(scores, token_lengths)
    if scheduling_group_id is not None:
        result = jax.tree.map(
            lambda x: xla_metadata.set_xla_metadata(
                x, _scheduling_group_id=scheduling_group_id
            ),
            result,
        )
    return result if publish else result[0]
