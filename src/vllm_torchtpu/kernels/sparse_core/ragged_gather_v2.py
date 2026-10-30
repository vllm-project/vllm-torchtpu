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

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.megablox.gmm_v2 import (
    LHS_SCALE_TRAILER_COLS,
    PREQUANTIZED_LHS_BLOCK_SIZE,
    QUANTIZE_LHS_BLOCK_ROWS,
    _quantize_lhs_rows_kernel,
)
from vllm_torchtpu.kernels.sparse_core import core_map_helper


def calculate_col_size(hidden_size: int, packing: int) -> int:
    """Calculates the max column size bounded by VMEM limits and
    hidden_size divisibility."""
    tpu_info = pltpu.get_tpu_info()
    sc_info = tpu_info.sparse_core
    assert sc_info is not None, "SparseCore info is missing."
    lanes = sc_info.num_lanes

    match tpu_info.generation:
        case 6:
            target_bytes = (256 * 1024) * 0.8
        case 7:
            target_bytes = (512 * 1024) * 0.8
        case _:
            target_bytes = (128 * 1024) * 0.8

    # Calculate max safe column size based on VMEM budget and aligne to 128.
    num_buffers = 2
    bytes_per_col = (lanes + lanes // packing) * 4 * num_buffers
    max_safe_col = int((target_bytes // bytes_per_col) // 128) * 128

    # Search for the largest divisor of hidden_size bounded by max_safe_col.
    # The first divisor found is the maximum.
    start_col = (min(hidden_size, max_safe_col) // 128) * 128
    for c in range(start_col, 127, -128):
        if hidden_size % c == 0:
            return c
    return max_safe_col


def main_kernel_v2(
    start_ref: jax.Ref,
    end_ref: jax.Ref,
    in_hbm_ref: jax.Ref,
    indices_hbm_ref: jax.Ref,
    out_hbm_ref: jax.Ref,
    start_vmem_ref: jax.Ref,
    end_vmem_ref: jax.Ref,
    sem_ref: jax.Ref,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_row_subchunks: int,
    row_dtype: jnp.dtype | None = None,
):
    # `row_dtype` set: the output is int32 words of `row_dtype` rows.
    tpu_info = pltpu.get_tpu_info()
    sc_info = tpu_info.sparse_core
    assert sc_info is not None
    num_simd_lanes = sc_info.num_lanes
    hidden_size = in_hbm_ref.shape[-1]
    dtype_bits = jax.dtypes.itemsize_bits(row_dtype or out_hbm_ref.dtype)
    packing = 32 // dtype_bits
    col_size = calculate_col_size(hidden_size, packing)

    assert isinstance(hidden_size, int), (
        f"hidden_size must be int, got {type(hidden_size)}"
    )
    num_cores = jax.lax.axis_size((core_axis_name, subcore_axis_name))
    row_subchunk_size = num_simd_lanes
    row_chunk_size = row_subchunk_size * num_row_subchunks
    block_size = row_chunk_size * num_cores

    recv_sem = sem_ref.at[0]

    copy_start = pltpu.make_async_copy(start_ref, start_vmem_ref.at[:1], recv_sem)
    copy_end = pltpu.make_async_copy(end_ref, end_vmem_ref.at[:1], recv_sem)
    copy_start.start()
    copy_end.start()
    copy_start.wait()
    copy_end.wait()

    start = start_vmem_ref[...][0]
    end = end_vmem_ref[...][0]

    block_start = start // block_size
    block_end = pl.cdiv(end, block_size)
    num_blocks = block_end - block_start
    num_blocks = jnp.where(end <= start, 0, num_blocks)

    num_cols = pl.cdiv(hidden_size, col_size)

    dtype = row_dtype or out_hbm_ref.dtype
    dtype_bits = jax.dtypes.itemsize_bits(dtype)
    packing = 32 // dtype_bits

    core_index = lax.axis_index((core_axis_name, subcore_axis_name))

    # SparseCore `.bitcast()` leverages hardware Row-Packing for
    # 16-bit -> 32-bit conversion.
    # The logical row count halves, while physical column dimensions remain unchanged.
    # A row-pair-packed int32 input (`pack_row_pairs`) is already that view.
    in_hbm_i32 = (
        in_hbm_ref if in_hbm_ref.dtype == jnp.int32 else in_hbm_ref.bitcast(jnp.int32)
    )
    out_hbm_i32 = (
        out_hbm_ref if row_dtype is not None else out_hbm_ref.bitcast(jnp.int32)
    )

    num_phys_cols = col_size

    # The outer pipeline runs on the Vector Core, hoisting the integer arithmetic
    # required to decode the row-packed indices. This prevents scalar-core instruction
    # starvation during the execution of the indirect `in_specs` block lambdas.

    # TODO(guoweij): The nested `emit_pipeline` design creates a pipeline bubble
    # (DMA wait) when the inner pipeline empties and restarts with new indices. This is
    # a known high-level API limitation, currently amortized by setting a large
    # num_row_subchunks and shouldn't be an issue for most use cases. We should still
    # monitor the bubble's impact on E2E performance and, if needed, manually
    # reimplement this using primitive operations to eliminate the bubble entirely.

    def col_loop(col_base, gather_ref, out_ref, idx_rem, unpack_col_chunk):
        col_slice = pl.ds(col_base, unpack_col_chunk)
        if packing == 1:
            gather_dt = gather_ref.bitcast(dtype)
            out_dt = out_ref.bitcast(dtype)
            out_dt[:, col_slice] = gather_dt[:, col_slice]
        else:
            # Manual bitwise extraction and packing for
            # packing >= 2 (bfloat16, int8, int4)
            # bf16: 0xFFFF, int8: 0xFF, int4: 0xF
            mask = (1 << dtype_bits) - 1
            shift_multiplier = dtype_bits.bit_length() - 1
            for i in range(num_simd_lanes // packing):
                packed_row = jnp.zeros((1, unpack_col_chunk), dtype=jnp.int32)
                for j in range(packing):
                    k = i * packing + j
                    dynamic_shift = jnp.left_shift(idx_rem[k], shift_multiplier)
                    val = jnp.bitwise_right_shift(
                        gather_ref[pl.ds(k, 1), col_slice], dynamic_shift
                    )
                    val = jnp.bitwise_and(val, mask)
                    pack_shift = j * dtype_bits
                    packed_row = jnp.bitwise_or(
                        packed_row, jnp.left_shift(val, pack_shift)
                    )
                out_ref[pl.ds(i, 1), col_slice] = packed_row

    def inner_pipeline(gather_ref, out_ref, idx_ref, unpack_col_chunk):
        row_slice = pl.ds(pl.program_id(0) * row_subchunk_size, row_subchunk_size)
        subchunk_idxs = idx_ref[row_slice]
        if packing > 1:
            # Equivalent to `subchunk_idxs % packing`
            idx_rem = jnp.bitwise_and(subchunk_idxs, packing - 1)
        else:
            idx_rem = jnp.zeros_like(subchunk_idxs)

        col_loop_fn = functools.partial(
            col_loop,
            gather_ref=gather_ref,
            out_ref=out_ref,
            idx_rem=idx_rem,
            unpack_col_chunk=unpack_col_chunk,
        )
        plsc.parallel_loop(0, num_phys_cols, step=unpack_col_chunk)(col_loop_fn)

    def outer_pipeline(idx_ref):
        b = pl.program_id(0)
        b_global = b + block_start

        unpack_col_chunk = 128
        assert num_phys_cols % unpack_col_chunk == 0
        shift_amount = packing.bit_length() - 1
        pltpu.emit_pipeline(
            functools.partial(
                inner_pipeline, idx_ref=idx_ref, unpack_col_chunk=unpack_col_chunk
            ),
            grid=(num_row_subchunks, num_cols),
            in_specs=pl.BlockSpec(
                (pl.Indirect(row_subchunk_size), num_phys_cols),
                lambda r, col_id: (
                    jnp.bitwise_right_shift(
                        idx_ref[pl.ds(r * row_subchunk_size, row_subchunk_size)],
                        shift_amount,
                    ),
                    col_id,
                ),
            ),
            out_specs=pl.BlockSpec(
                (row_subchunk_size // packing, num_phys_cols),
                lambda r, col_id: (
                    (b_global * num_cores + core_index) * num_row_subchunks + r,
                    col_id,
                ),
            ),
        )(in_hbm_i32, out_hbm_i32)

    pltpu.emit_pipeline(
        outer_pipeline,
        grid=(num_blocks,),
        in_specs=pl.BlockSpec(
            (row_chunk_size,),
            lambda b: ((b + block_start) * num_cores + core_index,),
        ),
    )(indices_hbm_ref)


# Rows per pack_row_pairs grid step: 2 MiB of bf16 in, 2 MiB of int32 out.
PACK_BLOCK_ROWS = 256


def _pack_row_pairs_kernel(x_ref, o_ref):
    o_ref[...] = pltpu.bitcast(x_ref[...], jnp.int32)


def pack_row_pairs(x: jax.Array, interpret: bool = False) -> jax.Array:
    """bf16 ``[2n, h]`` as ``[n, h]`` int32, row ``2i`` in the low half.

    This is the int32 view the kernel reads a 16-bit input through, so a
    packed input skips the kernel-side relayout. An int32 array also keeps its
    native tiling through a collective, where a bf16 one is gathered in the
    default tiling and relaid out afterwards at the gathered size. A Pallas
    bitcast, because XLA lowers the same packing as several full passes.
    """
    rows, hidden = x.shape
    assert x.dtype == jnp.bfloat16 and rows % 2 == 0, (x.dtype, rows)
    block = min(PACK_BLOCK_ROWS, rows)
    return pl.pallas_call(
        _pack_row_pairs_kernel,
        out_shape=jax.ShapeDtypeStruct((rows // 2, hidden), jnp.int32),
        grid=(pl.cdiv(rows, block),),
        in_specs=[pl.BlockSpec((block, hidden), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((block // 2, hidden), lambda i: (i, 0)),
        interpret=interpret,
        name="moe_pack_row_pairs",
    )(x)


def unpack_row_pairs(packed: jax.Array, dtype: jnp.dtype) -> jax.Array:
    """Inverse of `pack_row_pairs`."""
    pairs, hidden = packed.shape
    bits = jax.lax.bitcast_convert_type(packed, jnp.uint32)
    halves = jnp.stack([bits & 0xFFFF, jnp.right_shift(bits, 16)], axis=1)
    halves = jax.lax.bitcast_convert_type(halves.astype(jnp.uint16), dtype)
    return halves.reshape(2 * pairs, hidden)


def unpack_row_quads(packed: jax.Array) -> jax.Array:
    """int32 ``[n, h]`` holding row ``4i + j`` in byte ``j`` as int8 ``[4n, h]``."""
    quads, hidden = packed.shape
    bytes_ = jax.lax.bitcast_convert_type(packed, jnp.int8)  # [n, h, 4]
    return jnp.swapaxes(bytes_, 1, 2).reshape(4 * quads, hidden)


def pack_row_quads(rows: jax.Array) -> jax.Array:
    """Inverse of `unpack_row_quads`."""
    n, hidden = rows.shape
    quads = jnp.swapaxes(rows.reshape(n // 4, 4, hidden), 1, 2)
    return jax.lax.bitcast_convert_type(quads, jnp.int32)


# Token quads (`pack_token_quads`): int32 [n, token_cols + r] holding int8 rows
# 4i..4i+3 like row quads, except that row 4i+k's leading `token_cols` bytes
# fill words [k * token_cols / 4, (k + 1) * token_cols / 4) of the row, so a
# gather of one row reads only that row's bytes. Word 128 * b + p of that span
# holds the row's columns 512 * b + 128 * j + p in byte j. The last r columns
# stay row quads.
TOKEN_QUAD_LANES = 128
TOKEN_QUAD_COLS = 4 * TOKEN_QUAD_LANES
# Token-span words each transpose loop iteration covers.
TRANSPOSE_WORDS = 16
# Chunks whose gathers are in flight while one is transposed.
TOKEN_GATHER_AHEAD = 2


def transpose_byte_quads(
    w: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Byte ``k`` of output word ``j`` is byte ``j`` of input word ``k``.

    Its own inverse. Static shifts and masks, so it runs on a TensorCore and
    a SparseCore alike.
    """
    w0, w1, w2, w3 = w

    def swap(a, b, shift, mask):
        t = jnp.bitwise_and(jnp.bitwise_xor(jnp.right_shift(a, shift), b), mask)
        return jnp.bitwise_xor(a, jnp.left_shift(t, shift)), jnp.bitwise_xor(b, t)

    w0, w2 = swap(w0, w2, 16, 0x0000FFFF)
    w1, w3 = swap(w1, w3, 16, 0x0000FFFF)
    w0, w1 = swap(w0, w1, 8, 0x00FF00FF)
    w2, w3 = swap(w2, w3, 8, 0x00FF00FF)
    return w0, w1, w2, w3


def pack_token_quads(rows: jax.Array, token_cols: int) -> jax.Array:
    """int8 ``[n, h]`` as token quads, ``n`` a multiple of 4."""
    n, _ = rows.shape
    assert token_cols % TOKEN_QUAD_COLS == 0, token_cols
    data = rows[:, :token_cols].reshape(
        n // 4, 4, token_cols // TOKEN_QUAD_COLS, 4, TOKEN_QUAD_LANES
    )
    words = jax.lax.bitcast_convert_type(jnp.swapaxes(data, 3, 4), jnp.int32)
    return jnp.concatenate(
        [words.reshape(n // 4, token_cols), pack_row_quads(rows[:, token_cols:])],
        axis=1,
    )


def unpack_token_quads(packed: jax.Array, token_cols: int) -> jax.Array:
    """Inverse of `pack_token_quads`."""
    quads, _ = packed.shape
    data = jax.lax.bitcast_convert_type(packed[:, :token_cols], jnp.int8).reshape(
        quads, 4, token_cols // TOKEN_QUAD_COLS, TOKEN_QUAD_LANES, 4
    )
    return jnp.concatenate(
        [
            jnp.swapaxes(data, 3, 4).reshape(4 * quads, token_cols),
            unpack_row_quads(packed[:, token_cols:]),
        ],
        axis=1,
    )


def store_token_quads(quads_ref: jax.Ref, o_ref: jax.Ref, token_cols: int):
    """In a TensorCore kernel: store row quads ``quads_ref`` as token quads."""
    words = token_cols // 4
    lanes = TOKEN_QUAD_LANES
    for b in range(token_cols // TOKEN_QUAD_COLS):
        rows = transpose_byte_quads(
            tuple(
                quads_ref[:, pl.ds(TOKEN_QUAD_COLS * b + lanes * j, lanes)]
                for j in range(4)
            )
        )
        for k, row in enumerate(rows):
            o_ref[:, pl.ds(words * k + lanes * b, lanes)] = row
    o_ref[:, token_cols:] = quads_ref[:, token_cols:]


def _quantize_token_quads_kernel(x_ref, o_ref, quads_ref):
    _quantize_lhs_rows_kernel(x_ref, quads_ref)
    store_token_quads(quads_ref, o_ref, x_ref.shape[1])


def quantize_token_quads(x: jax.Array, interpret: bool = False) -> jax.Array:
    """`gmm_v2.quantize_lhs_rows`, stored as token quads.

    The same values and scale trailer, bit for bit: ``unpack_token_quads(
    quantize_token_quads(x), k)`` equals ``unpack_row_quads(quantize_lhs_rows(
    x))``. `ragged_gather_v2` with ``token_cols=k`` turns a gather of these
    rows back into the row quads `gmm_v2` reads.
    """
    rows, k = x.shape
    assert x.dtype == jnp.bfloat16 and rows % 4 == 0, (x.dtype, rows)
    assert k % PREQUANTIZED_LHS_BLOCK_SIZE == 0, k
    block = min(QUANTIZE_LHS_BLOCK_ROWS, rows)
    out_cols = k + LHS_SCALE_TRAILER_COLS
    return pl.pallas_call(
        _quantize_token_quads_kernel,
        out_shape=jax.ShapeDtypeStruct((rows // 4, out_cols), jnp.int32),
        grid=(pl.cdiv(rows, block),),
        in_specs=[pl.BlockSpec((block, k), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((block // 4, out_cols), lambda i: (i, 0)),
        scratch_shapes=[pltpu.VMEM((block // 4, out_cols), jnp.int32)],
        interpret=interpret,
        name="moe_quantize_token_quads",
    )(x)


def _balanced_range(n, part, num_parts):
    """Part ``part``'s [begin, begin + count) of ``n`` items split evenly over
    ``num_parts`` parts (counts differ by at most one)."""
    base = n // num_parts
    extra = n - base * num_parts
    count = base + (part < extra).astype(jnp.int32)
    begin = part * base + jnp.minimum(part, extra)
    return begin, count


def token_quads_kernel(
    start_ref: jax.Ref,
    end_ref: jax.Ref,
    in_hbm_ref: jax.Ref,
    indices_hbm_ref: jax.Ref,
    out_hbm_ref: jax.Ref,
    start_vmem_ref: jax.Ref,
    end_vmem_ref: jax.Ref,
    idx_vmem_ref: jax.Ref,
    data_vmem_ref: jax.Ref,
    trailer_vmem_ref: jax.Ref,
    out_vmem_ref: jax.Ref,
    sem_ref: jax.Ref,
    row_smem_ref: jax.Ref,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    token_cols: int,
):
    """Gathers token-quad rows into row quads, the rows `main_kernel_v2` would.

    The 16-row chunks covering [start, end) are split evenly over the
    subcores, and each subcore runs its chunks as one flat pipeline: a chunk
    DMAs every row's own token span and its quad's trailer into one of
    ``TOKEN_GATHER_AHEAD + 1`` slots, and its byte transposes overlap the
    gathers of the chunks ahead and the last chunk's write. The transpose loop
    itself issues the row DMAs of the chunk ``TOKEN_GATHER_AHEAD`` ahead.
    """
    sc_info = pltpu.get_tpu_info().sparse_core
    assert sc_info is not None
    lanes = sc_info.num_lanes
    token_words = token_cols // 4
    trailer_cols = in_hbm_ref.shape[-1] - token_cols
    out_rows = lanes // 4
    num_slots = data_vmem_ref.shape[0] // lanes
    # The first chunk's indices get their own semaphore: a wait on one shared
    # with the rest could be satisfied by the rest's bytes.
    idx_sem, out_sem = sem_ref.at[pl.ds(0, 2)], sem_ref.at[pl.ds(2, 2)]
    in_sem = sem_ref.at[pl.ds(4, num_slots)]

    copy_start = pltpu.make_async_copy(start_ref, start_vmem_ref.at[:1], in_sem.at[0])
    copy_end = pltpu.make_async_copy(end_ref, end_vmem_ref.at[:1], in_sem.at[0])
    copy_start.start()
    copy_end.start()
    copy_start.wait()
    copy_end.wait()
    start = start_vmem_ref[...][0]
    end = end_vmem_ref[...][0]

    first_chunk = start // lanes
    num_chunks = jnp.where(end > start, pl.cdiv(end, lanes) - first_chunk, 0)
    # Order the parts so the chunks left over alternate between SparseCores.
    num_cores = jax.lax.axis_size(core_axis_name)
    part = lax.axis_index(subcore_axis_name) * num_cores + lax.axis_index(
        core_axis_name
    )
    part_begin, my_chunks = _balanced_range(
        num_chunks, part, jax.lax.axis_size((core_axis_name, subcore_axis_name))
    )
    chunk0 = first_chunk + part_begin

    def idx_copy(first, count):
        return pltpu.make_async_copy(
            indices_hbm_ref.at[
                pl.ds(pl.multiple_of((chunk0 + first) * lanes, lanes), count * lanes)
            ],
            idx_vmem_ref.at[pl.ds(first * lanes, count * lanes)],
            idx_sem.at[min(first, 1)],
        )

    def chunk_indices(g):
        return idx_vmem_ref[pl.ds(pl.multiple_of(g * lanes, lanes), lanes)]

    def row_copy(g, r, row, cols):
        """Chunk ``g``'s row ``r``, index ``row``: its token span's ``cols``."""
        return pltpu.make_async_copy(
            in_hbm_ref.at[
                pl.ds(jnp.right_shift(row, 2), 1),
                pl.ds(jnp.bitwise_and(row, 3) * token_words + cols.start, cols.size),
            ],
            data_vmem_ref.at[pl.ds((g % num_slots) * lanes + r, 1), cols],
            in_sem.at[g % num_slots],
        )

    def start_trailers(g, idx):
        pltpu.make_async_copy(
            in_hbm_ref.at[jnp.right_shift(idx, 2), pl.ds(token_cols, trailer_cols)],
            trailer_vmem_ref.at[pl.ds((g % num_slots) * lanes, lanes)],
            in_sem.at[g % num_slots],
        ).start()

    def start_gathers(g):
        idx = chunk_indices(g)
        for r in range(lanes):
            row_copy(g, r, idx[r], pl.ds(0, token_words)).start()
        start_trailers(g, idx)

    def wait_gathers(g):
        # A wait takes its descriptor's size, here that of all the slot's copies.
        slot = g % num_slots
        for cols, dst in (
            (pl.ds(0, token_words), data_vmem_ref),
            (pl.ds(token_cols, trailer_cols), trailer_vmem_ref),
        ):
            pltpu.make_async_copy(
                in_hbm_ref.at[pl.ds(0, lanes), cols],
                dst.at[pl.ds(slot * lanes, lanes)],
                in_sem.at[slot],
            ).wait()

    def out_copy(g):
        slot = g % 2
        return pltpu.make_async_copy(
            out_vmem_ref.at[pl.ds(slot * out_rows, out_rows)],
            out_hbm_ref.at[
                pl.ds(pl.multiple_of((chunk0 + g) * out_rows, out_rows), out_rows)
            ],
            out_sem.at[slot],
        )

    iters = token_words // TRANSPOSE_WORDS
    assert token_words % TRANSPOSE_WORDS == 0 and TRANSPOSE_WORDS % lanes == 0
    assert iters % lanes == 0
    # Each transpose iteration DMAs one piece of one row of the chunk ahead.
    pieces = iters // lanes
    piece_words = token_words // pieces

    def transpose(g, ahead=None):
        in_row = (g % num_slots) * lanes
        out_row = (g % 2) * out_rows
        if ahead is not None:
            idx = chunk_indices(ahead)
            # A lane cannot be read at a traced index: the loop reads SMEM.
            for r in range(lanes):
                row_smem_ref[r] = idx[r]
            start_trailers(ahead, idx)

        # Word w = 128 b + p of a row's token span holds its columns
        # 512 b + 128 j + p, j its byte; that is output column 4 w - 3 p + 128 j.
        @plsc.parallel_loop(0, token_words, step=TRANSPOSE_WORDS)
        def _(w0):
            if ahead is not None:
                i = w0 // TRANSPOSE_WORDS
                r = i // pieces
                cols = pl.ds(
                    pl.multiple_of((i % pieces) * piece_words, lanes), piece_words
                )
                row_copy(ahead, r, row_smem_ref[r], cols).start()
            for v in range(0, TRANSPOSE_WORDS, lanes):
                w = pl.multiple_of(w0 + v, lanes)
                out_col = 4 * w - 3 * jnp.bitwise_and(w, TOKEN_QUAD_LANES - 1)
                # All loads ahead of all stores: the stores could alias them.
                words = [
                    tuple(
                        data_vmem_ref[in_row + 4 * m + k, pl.ds(w, lanes)]
                        for k in range(4)
                    )
                    for m in range(out_rows)
                ]
                for m in range(out_rows):
                    for j, word in enumerate(transpose_byte_quads(words[m])):
                        out_vmem_ref[
                            out_row + m,
                            pl.ds(
                                pl.multiple_of(out_col + TOKEN_QUAD_LANES * j, lanes),
                                lanes,
                            ),
                        ] = word

        # The trailer is row quads: pick each row's byte.
        shifts = jnp.left_shift(jnp.bitwise_and(chunk_indices(g), 3), 3)

        @plsc.parallel_loop(0, trailer_cols, step=lanes)
        def _(c):
            for m in range(out_rows):
                packed = jnp.zeros((lanes,), jnp.int32)
                for k in range(4):
                    r = 4 * m + k
                    byte = jnp.right_shift(
                        trailer_vmem_ref[in_row + r, pl.ds(c, lanes)], shifts[r]
                    )
                    byte = jnp.left_shift(jnp.bitwise_and(byte, 0xFF), 8 * k)
                    packed = jnp.bitwise_or(packed, byte)
                out_vmem_ref[out_row + m, pl.ds(token_cols + c, lanes)] = packed

    def step(g):
        wait_gathers(g)

        @pl.when(g >= 2)
        def _():
            out_copy(g - 2).wait()

        # The chunk ahead takes the slot the last chunk's transpose freed.
        ahead = g + num_slots - 1
        pl.when(ahead < my_chunks)(lambda: transpose(g, ahead))
        pl.when(ahead >= my_chunks)(lambda: transpose(g))
        out_copy(g).start()

    @pl.when(my_chunks > 0)
    def _():
        # The first chunk's indices alone gate the first gathers.
        idx_copy(0, 1).start()
        # A static single-chunk index buffer has no rest to load.
        has_rest = idx_vmem_ref.shape[0] > lanes
        rest = functools.partial(idx_copy, 1, my_chunks - 1)
        if has_rest:
            pl.when(my_chunks > 1)(lambda: rest().start())
        idx_copy(0, 1).wait()
        start_gathers(0)
        if has_rest:
            pl.when(my_chunks > 1)(lambda: rest().wait())
        for g in range(1, num_slots - 1):
            pl.when(g < my_chunks)(functools.partial(start_gathers, g))

        pl.loop(0, my_chunks)(step)

        out_copy(0).wait()
        pl.when(my_chunks > 1)(lambda: out_copy(1).wait())


def _unpack_rows(packed: jax.Array, dtype: jnp.dtype) -> jax.Array:
    if dtype == jnp.int8:
        return unpack_row_quads(packed)
    return unpack_row_pairs(packed, dtype)


@functools.partial(jax.jit, static_argnames=("out_dtype", "packed_out", "token_cols"))
def ragged_gather_v2(
    x: jax.Array,
    indices: jax.Array,
    start: jax.Array,
    end: jax.Array,
    out_dtype: jnp.dtype | None = None,
    packed_out: bool = False,
    token_cols: int | None = None,
) -> jax.Array:
    """Perform gather on indices within dynamic array start and end using BlockSpec.

    ``x`` may be a `pack_row_pairs` int32 array with ``out_dtype`` its 16-bit
    dtype, or int32 row quads of int8 rows (`unpack_row_quads`) with
    ``out_dtype`` int8; ``indices`` then still name unpacked rows.

    ``packed_out`` returns int8 rows as their int32 row quads, the layout a
    TensorCore kernel reads an 8-bit array in. ``token_cols`` says ``x`` holds
    the int8 rows as `pack_token_quads` token quads instead.
    """

    assert x.ndim == 2, "Ragged gather only supports 2d inputs."
    assert indices.ndim == 1, "Ragged gather only supports 1d indices."

    if jnp.isscalar(start):
        start = start[None]
    if jnp.isscalar(end):
        end = end[None]

    row_packed = out_dtype is not None and x.dtype != out_dtype
    if row_packed and not (
        x.dtype == jnp.int32 and out_dtype in (jnp.bfloat16, jnp.int8)
    ):
        raise ValueError(
            "a packed input must be int32 rows of bf16 or int8, got "
            f"{x.dtype} -> {out_dtype}"
        )
    dtype = out_dtype if row_packed else x.dtype
    if dtype not in (jnp.bfloat16, jnp.float32, jnp.int8, jnp.int4):
        raise ValueError(f"dtype must be f32, bf16, int8, or int4, but got {dtype}")

    if packed_out and (dtype != jnp.int8 or indices.size % 4):
        raise ValueError(f"packed_out takes int8 rows in fours, got {dtype}")

    if token_cols is not None and not (row_packed and packed_out):
        raise ValueError("token quads gather int8 rows into row quads")

    sc_info = pltpu.get_tpu_info().sparse_core
    if sc_info is None:
        if token_cols is not None:
            return pack_row_quads(unpack_token_quads(x, token_cols)[indices])
        rows = (_unpack_rows(x, dtype) if row_packed else x)[indices]
        return pack_row_quads(rows) if packed_out else rows

    hidden_size = x.shape[-1]
    out_size = indices.size

    dtype_bits = jax.dtypes.itemsize_bits(dtype)
    packing = 32 // dtype_bits
    col_size = calculate_col_size(hidden_size, packing)

    aligned_hidden_size = ((hidden_size + col_size - 1) // col_size) * col_size

    num_simd_lanes = sc_info.num_lanes
    num_cores = sc_info.num_cores * sc_info.num_subcores
    base_block_size = num_simd_lanes * num_cores

    # Calculate ideal num_row_subchunks to avoid too much padding overhead.
    num_row_subchunks = max(
        1, min(4, (out_size + base_block_size - 1) // base_block_size)
    )

    row_subchunk_size = num_simd_lanes
    row_chunk_size = row_subchunk_size * num_row_subchunks
    block_size = row_chunk_size * num_cores

    out_pad_size = ((out_size + block_size - 1) // block_size) * block_size - out_size
    out_packing = packing if packed_out else 1

    vector_mesh = plsc.VectorSubcoreMesh(
        num_cores=sc_info.num_cores,
        num_subcores=sc_info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    compiler_params = pltpu.CompilerParams(
        use_tc_tiling_on_sc=True,
        needs_layout_passes=True,
        disable_bounds_checks=True,
    )
    if token_cols is not None:
        slot_rows = (TOKEN_GATHER_AHEAD + 1) * num_simd_lanes
        # Whole 16-row chunks; a subcore's chunk range is at most this many.
        chunk_pad_size = -out_size % num_simd_lanes
        max_chunks = pl.cdiv((out_size + chunk_pad_size) // num_simd_lanes, num_cores)
        return core_map_helper.kernel(
            functools.partial(
                token_quads_kernel,
                core_axis_name=vector_mesh.core_axis_name,
                subcore_axis_name=vector_mesh.subcore_axis_name,
                token_cols=token_cols,
            ),
            out_type=jax.ShapeDtypeStruct(
                ((out_size + chunk_pad_size) // 4, hidden_size), jnp.int32
            ),
            compiler_params=compiler_params,
            scratch_types=[
                pltpu.VMEM((16,), jnp.int32),
                pltpu.VMEM((16,), jnp.int32),
                pltpu.VMEM((max_chunks * num_simd_lanes,), jnp.int32),
                pltpu.VMEM((slot_rows, token_cols // 4), jnp.int32),
                pltpu.VMEM((slot_rows, hidden_size - token_cols), jnp.int32),
                pltpu.VMEM((2 * num_simd_lanes // 4, hidden_size), jnp.int32),
                pltpu.SemaphoreType.DMA((5 + TOKEN_GATHER_AHEAD,)),
                pltpu.SMEM((num_simd_lanes,), jnp.int32),
            ],
            mesh=vector_mesh,
            name="sc_ragged_gather_v2",
        )(start, end, x, jnp.pad(indices, (0, chunk_pad_size)))[: out_size // 4]
    indices = jnp.pad(indices, ((0, out_pad_size)))
    return core_map_helper.kernel(
        functools.partial(
            main_kernel_v2,
            core_axis_name=vector_mesh.core_axis_name,
            subcore_axis_name=vector_mesh.subcore_axis_name,
            num_row_subchunks=num_row_subchunks,
            row_dtype=dtype if packed_out else None,
        ),
        out_type=jax.ShapeDtypeStruct(
            ((out_size + out_pad_size) // out_packing, aligned_hidden_size),
            jnp.int32 if packed_out else dtype,
        ),
        compiler_params=compiler_params,
        scratch_types=[
            pltpu.VMEM((16,), jnp.int32),
            pltpu.VMEM((16,), jnp.int32),
            pltpu.SemaphoreType.DMA((1,)),
        ],
        mesh=vector_mesh,
        name="sc_ragged_gather_v2",
    )(start, end, x, indices)[: out_size // out_packing, :hidden_size]
