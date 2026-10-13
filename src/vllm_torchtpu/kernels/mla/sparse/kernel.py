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
"""TPU-Friendly MLA Ragged Paged Attention kernel - GLM5.2 DeepSeek Sparse Attention."""

import functools
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.mla import kv_cache_utils
from vllm_torchtpu.kernels.mla.sparse import dsa_gather, native_sc_gather

DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024
TENSORCORE_CACHE_LAYOUT = "tensorcore"
SPARSECORE_CACHE_LAYOUT = "sparsecore"
DEFAULT_GATHER_AND_ATTENTION_CHUNK_SIZE = 64
DEFAULT_ATTENTION_KERNEL_BATCH_SIZE = 16
# The native SparseCore gather's unit, which `topk_cap` must be a multiple of.
GATHER_ATOM_ROWS = native_sc_gather.OCCURRENCES_PER_ATOM


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def get_dtype_bitwidth(dtype):
    return jax.dtypes.itemsize_bits(dtype)


def get_dtype_packing(dtype):
    bits = get_dtype_bitwidth(dtype)
    return 32 // bits


def _largest_divisor(x: int, cap: int) -> int:
    """Largest divisor of ``x`` that is <= ``cap``."""
    for candidate in range(min(x, cap), 0, -1):
        if x % candidate == 0:
            return candidate
    return 1


_GATHER_PAGE_CHUNK = 128


def _gather_page_ids_kernel(windows_ref, logical_ref, out_ref, *, num_chunks):
    logical = logical_ref[...]  # i32[block_tokens, topk]
    out = jnp.zeros_like(logical)
    for c in range(num_chunks):
        window_chunk = windows_ref[
            :, c * _GATHER_PAGE_CHUNK : (c + 1) * _GATHER_PAGE_CHUNK
        ]  # i32[block_tokens, 128]
        local = logical - c * _GATHER_PAGE_CHUNK
        gathered = jnp.take_along_axis(
            window_chunk, jnp.clip(local, 0, _GATHER_PAGE_CHUNK - 1), axis=1
        )
        out = jnp.where((local >= 0) & (local < _GATHER_PAGE_CHUNK), gathered, out)
    out_ref[...] = out


def gather_page_ids(
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    seq_page_ids: jax.Array,  # i32[num_tokens, topk]  (logical page within seq)
    seq_ids_segment: jax.Array,  # i32[num_tokens]  (token -> seq id)
    max_num_seqs: int,
    *,
    block_tokens: int = 8,
) -> jax.Array:
    """Gathers physical page ids for the CSA top-k tokens."""
    num_tokens, topk = seq_page_ids.shape
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    num_chunks = cdiv(pages_per_seq, _GATHER_PAGE_CHUNK)
    padded_pps = num_chunks * _GATHER_PAGE_CHUNK

    page_table = page_indices.reshape(max_num_seqs, pages_per_seq)
    if padded_pps != pages_per_seq:
        page_table = jnp.pad(page_table, ((0, 0), (0, padded_pps - pages_per_seq)))
    # Per-token page-table window. This is a whole-row gather.
    windows = page_table[seq_ids_segment]  # i32[num_tokens, padded_pps]
    logical = jnp.clip(seq_page_ids, 0, pages_per_seq - 1)

    padded_tokens = align_to(num_tokens, block_tokens)
    if padded_tokens != num_tokens:
        pad = padded_tokens - num_tokens
        windows = jnp.pad(windows, ((0, pad), (0, 0)))
        logical = jnp.pad(logical, ((0, pad), (0, 0)))

    out = pl.pallas_call(
        functools.partial(_gather_page_ids_kernel, num_chunks=num_chunks),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((block_tokens, padded_pps), lambda t: (t, 0)),
                pl.BlockSpec((block_tokens, topk), lambda t: (t, 0)),
            ],
            out_specs=pl.BlockSpec((block_tokens, topk), lambda t: (t, 0)),
            grid=(padded_tokens // block_tokens,),
        ),
        out_shape=jax.ShapeDtypeStruct((padded_tokens, topk), jnp.int32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=True,
        ),
        name="gather_page_ids",
    )(windows, logical)
    return out[:num_tokens]


def _floor_divmod_const(x: jax.Array, d: int) -> tuple[jax.Array, jax.Array]:
    """Computes exact integer `(x // d, x % d)` without hardware integer division.

    TPU vector units lack a native vector `idiv` instruction. This function
    emulates floor div/mod with maximum efficiency:
    - Powers of 2: Uses bitwise shift (`>>`) and bitmask (`&`).
    - Arbitrary integers: Uses a reciprocal float32 multiplication followed by
      a remainder refinement and a ±1 clamp to guarantee bit-exact results.

    Args:
        x: Input int32 tensor of arbitrary shape.
        d: Static positive integer divisor (must be > 0).

    Returns:
        A tuple `(quotient, remainder)` where `quotient = x // d` and
        `remainder = x % d`, both int32 arrays with exact floor semantics.
    """
    # Fast path: powers of 2 (e.g., page_size = 256) use bitwise operations.
    if d & (d - 1) == 0:
        shift = d.bit_length() - 1
        return lax.shift_right_arithmetic(x, jnp.int32(shift)), x & jnp.int32(d - 1)

    # General path: reciprocal float32 multiplication.
    inv = np.float32(1.0 / d)
    q = jnp.floor(x.astype(jnp.float32) * inv).astype(jnp.int32)
    r = x - q * jnp.int32(d)
    q2 = jnp.floor(r.astype(jnp.float32) * inv).astype(jnp.int32)
    q = q + q2
    r = r - q2 * jnp.int32(d)
    lo = r < 0
    q = jnp.where(lo, q - 1, q)
    r = jnp.where(lo, r + jnp.int32(d), r)
    hi = r >= d
    q = jnp.where(hi, q + 1, q)
    r = jnp.where(hi, r - jnp.int32(d), r)
    return q, r


def _capped_window_indices_kernel(
    live_windows_ref,  # i32[1] (scalar prefetch; sizes the grid)
    windows_ref,  # i32[block_tokens, padded_pps] page-table row per token
    topk_ref,  # i32[block_tokens, topk_cap] one window of the logical top-k
    out_ref,  # i32[block_tokens, topk_cap] its physical rows
    *,
    block_tokens: int,
    chunk_size: int,
    topk: int,
    topk_cap: int,
    page_size: int,
    pages_per_seq: int,
    total_num_pages: int,
):
    """Translates logical top-k slots to physical memory rows for one tile.

    Executes on TPU vector cores for a single (window `w`, chunk `c`, tile `tb`).
    - Valid slots: looks up their physical page in the page table.
    - Empty slots (-1): hashes them to pseudo-random physical rows using primes
      to prevent memory bank conflicts on SparseCore during gathering."""
    del live_windows_ref
    w = pl.program_id(0)
    c = pl.program_id(1)
    tb = pl.program_id(2)
    idx = topk_ref[...]
    is_padding = idx == -1

    # 1. Translate logical index -> (logical sequence page, token offset within page).
    seq_page, token_offset = _floor_divmod_const(idx, page_size)
    logical = jnp.clip(seq_page, 0, pages_per_seq - 1)

    # 2. Look up the physical page from the sequence's page table.
    # Processed in chunks of _GATHER_PAGE_CHUNK to stay within vector register limits.
    page = jnp.zeros_like(logical)
    for pc in range(windows_ref.shape[1] // _GATHER_PAGE_CHUNK):
        window_chunk = windows_ref[
            :, pc * _GATHER_PAGE_CHUNK : (pc + 1) * _GATHER_PAGE_CHUNK
        ]
        local = logical - pc * _GATHER_PAGE_CHUNK
        gathered = jnp.take_along_axis(
            window_chunk, jnp.clip(local, 0, _GATHER_PAGE_CHUNK - 1), axis=1
        )
        page = jnp.where((local >= 0) & (local < _GATHER_PAGE_CHUNK), gathered, page)

    # 3. Handle empty padding slots (-1) with pseudo-random scattering.
    # If all -1s read row 0, SparseCore would bottleneck on memory bank 0.
    # Multiplying unique slot coordinates by large primes hashes -1s uniformly.
    row = (
        c * chunk_size
        + tb * block_tokens
        + lax.broadcasted_iota(jnp.int32, idx.shape, 0)
    )
    col = w * topk_cap + lax.broadcasted_iota(jnp.int32, idx.shape, 1)
    flat_element_index = row * topk + col

    _, scattered_page = _floor_divmod_const(
        flat_element_index * 104729, total_num_pages
    )
    _, scattered_offset = _floor_divmod_const(flat_element_index * 15485863, page_size)

    # Use the real page/offset for valid slots; use scattered values for -1 padding.
    page = jnp.where(is_padding, scattered_page, page)
    token_offset = jnp.where(is_padding, scattered_offset, token_offset)

    # 4. Write physical row (page * page_size + offset) straight into output layout.
    out_ref[...] = page * page_size + token_offset


def _window_physical_rows(
    idx: jax.Array,  # i32[tokens, slots] logical top-k positions, -1 = empty
    windows_ref,  # i32[tokens, padded_pps] page-table row per token
    flat_element_index: jax.Array,  # i32[tokens, slots] row * topk + col
    *,
    page_size: int,
    pages_per_seq: int,
    total_num_pages: int,
) -> jax.Array:
    """Physical cache rows (page * page_size + offset) for a tile of slots.

    Valid slots look up their physical page in the token's page-table row;
    empty slots (-1) are hashed by their flat (row, col) coordinate to
    pseudo-random rows so the SparseCore gather does not hot-spot one bank.
    """
    is_padding = idx == -1

    # 1. Translate logical index -> (logical sequence page, token offset within page).
    seq_page, token_offset = _floor_divmod_const(idx, page_size)
    logical = jnp.clip(seq_page, 0, pages_per_seq - 1)

    # 2. Look up the physical page from the sequence's page table.
    # Processed in chunks of _GATHER_PAGE_CHUNK to stay within vector register limits.
    page = jnp.zeros_like(logical)
    for pc in range(windows_ref.shape[1] // _GATHER_PAGE_CHUNK):
        window_chunk = windows_ref[
            :, pc * _GATHER_PAGE_CHUNK : (pc + 1) * _GATHER_PAGE_CHUNK
        ]
        local = logical - pc * _GATHER_PAGE_CHUNK
        gathered = jnp.take_along_axis(
            window_chunk, jnp.clip(local, 0, _GATHER_PAGE_CHUNK - 1), axis=1
        )
        page = jnp.where((local >= 0) & (local < _GATHER_PAGE_CHUNK), gathered, page)

    # 3. Handle empty padding slots (-1) with pseudo-random scattering.
    # If all -1s read row 0, SparseCore would bottleneck on memory bank 0.
    # Multiplying unique slot coordinates by large primes hashes -1s uniformly.
    _, scattered_page = _floor_divmod_const(
        flat_element_index * 104729, total_num_pages
    )
    _, scattered_offset = _floor_divmod_const(flat_element_index * 15485863, page_size)

    # Use the real page/offset for valid slots; use scattered values for -1 padding.
    page = jnp.where(is_padding, scattered_page, page)
    token_offset = jnp.where(is_padding, scattered_offset, token_offset)

    # 4. Physical row (page * page_size + offset).
    return page * page_size + token_offset


def prepare_capped_window_indices(
    topk_indices: jax.Array,  # i32[num_tokens, topk]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    live_windows: jax.Array,  # i32[] in [1, topk // topk_cap]
    *,
    chunk_size: int,
    topk_cap: int,
    page_size: int,
    total_num_pages: int,
    block_tokens: int | None = None,
) -> jax.Array:
    """Launches the TPU kernel to translate logical slots to physical rows.

    This function configures and launches the `_capped_window_indices_kernel`.
    By sizing the Pallas grid dimension to `live_windows` (typically 1), it only
    translates slots for windows that will actually be read by the gather,
    leaving unneeded windows unwritten.

    Args:
        topk_indices: Shape [num_tokens, topk]. Logical token positions to gather.
        page_indices: Flattened page table mapping logical pages to physical pages.
        cu_q_lens: Cumulative query token count per sequence (ragged boundaries).
        live_windows: Scalar from `capped_live_windows` determining how many
          512-slot windows to translate (1 to 4).
        chunk_size: Attention chunk size (e.g. 256).
        topk_cap: Window capacity (e.g. 512).
        page_size: Tokens per physical page (e.g. 256).
        total_num_pages: Total physical cache pages allocated.
        block_tokens: Token sub-tile size inside each chunk (e.g. 16 or 32).

    Returns:
        Array of shape [num_chunks * num_windows * chunk_size, topk_cap]
        containing physical cache rows formatted directly in the layout
        expected by the downstream SparseCore gather.
    """
    num_tokens, topk = topk_indices.shape
    if num_tokens % chunk_size or topk % topk_cap:
        raise ValueError(
            f"num_tokens ({num_tokens}) and topk ({topk}) must be multiples of "
            f"chunk_size ({chunk_size}) and topk_cap ({topk_cap})"
        )
    if block_tokens is None:
        block_tokens = next((b for b in (32, 16, 8) if chunk_size % b == 0), chunk_size)
    if chunk_size % block_tokens:
        raise ValueError(
            f"block_tokens ({block_tokens}) must divide chunk_size ({chunk_size})"
        )
    num_windows = topk // topk_cap
    num_chunks = num_tokens // chunk_size
    blocks_per_chunk = chunk_size // block_tokens
    max_num_seqs = cu_q_lens.shape[0] - 1
    pages_per_seq = page_indices.shape[0] // max_num_seqs

    # Align page table width to vector register boundaries (_GATHER_PAGE_CHUNK).
    padded_pps = align_to(pages_per_seq, _GATHER_PAGE_CHUNK)
    page_table = page_indices.reshape(max_num_seqs, pages_per_seq)
    if padded_pps != pages_per_seq:
        page_table = jnp.pad(page_table, ((0, 0), (0, padded_pps - pages_per_seq)))

    # Associate each token with its sequence ID to look up its page table row.
    tokens_per_seq = cu_q_lens[1:] - cu_q_lens[:-1]
    seq_ids_segment = jnp.repeat(
        jnp.arange(max_num_seqs), tokens_per_seq, total_repeat_length=num_tokens
    )
    windows = page_table[seq_ids_segment]  # i32[num_tokens, padded_pps]

    def token_block(w, c, tb, lw):
        del w, lw
        return c * blocks_per_chunk + tb

    kernel = functools.partial(
        _capped_window_indices_kernel,
        block_tokens=block_tokens,
        chunk_size=chunk_size,
        topk=topk,
        topk_cap=topk_cap,
        page_size=page_size,
        pages_per_seq=pages_per_seq,
        total_num_pages=total_num_pages,
    )
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                # Spec 0: Page table for this token block
                pl.BlockSpec(
                    (block_tokens, padded_pps),
                    lambda w, c, tb, lw: (token_block(w, c, tb, lw), 0),
                ),
                # Spec 1: Top-k indices for this token block
                pl.BlockSpec(
                    (block_tokens, topk_cap),
                    lambda w, c, tb, lw: (token_block(w, c, tb, lw), w),
                ),
            ],
            # Direct-in-place write: places output straight into windowed chunk layout.
            out_specs=pl.BlockSpec(
                (block_tokens, topk_cap),
                lambda w, c, tb, lw: ((c * num_windows + w) * blocks_per_chunk + tb, 0),
            ),
            grid=(live_windows, num_chunks, blocks_per_chunk),
        ),
        out_shape=jax.ShapeDtypeStruct(
            (num_chunks * num_windows * chunk_size, topk_cap), jnp.int32
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary", "arbitrary"),
            disable_bounds_checks=True,
        ),
        name="capped_window_indices",
    )(jnp.reshape(live_windows, (1,)).astype(jnp.int32), windows, topk_indices)


def _capped_window_live_kernel(
    live_windows_ref,  # i32[1] (scalar prefetch; sizes the grid)
    tab_ref,  # i32[2 * G] (scalar prefetch) `live_group_table`
    windows_ref,  # i32[bz, padded_pps] page-table row per token
    topk_ref,  # i32[bz, topk_cap] one window of the logical top-k
    out_hbm_ref,  # i32[num_chunks * W * chunk * topk_cap // 128, 128] (HBM)
    stage_ref,  # VMEM i32[2, chunk * topk_cap // 128, 128] window-0 regions
    # VMEM i32[topk_cap // 128 * bz, 128]: 128-slot column tile j at rows
    # [j*bz, (j+1)*bz)
    phys_ref,
    tok_ref,  # VMEM i32[bz * topk_cap // 128, 128] token-major rows (w >= 1)
    sems,  # DMA[3]: window-0 region slots 0 / 1, token-major rows
    *,
    bz: int,
    chunk_size: int,
    topk: int,
    topk_cap: int,
    page_size: int,
    pages_per_seq: int,
    total_num_pages: int,
    num_windows: int,
    num_chunks: int,
):
    """`_capped_window_indices_kernel` for one (window w, chunk c, group tb),
    writing the 128-lane linear form, with window 0 laid out by atom block.

    Window 0 of chunk c is `R = chunk * topk_cap / (bz * 64)` atom blocks of
    one (8, 128) i32 tile each: block (g, a) holds the `bz` tokens of group g
    times their slots [64 a, 64 a + 64), token-major. The live blocks
    (a < A(g)) go to the front, block `off(g) + a`, so a chunk's live atoms
    are one prefix; the dead blocks fill the region from its end, so the
    region stays a permutation of valid physical rows (an overflowing chunk
    gathers all of it). A group with every atom live keeps its rows
    token-major at `off(g)` (the kernel then fetches it with one DMA).
    Windows w >= 1 keep the token-major order of
    `prepare_capped_window_indices`. The region is staged in VMEM (two slots)
    and written with one DMA per chunk.
    """
    del live_windows_ref
    w = pl.program_id(0)
    c = pl.program_id(1)
    tb = pl.program_id(2)
    gpc = chunk_size // bz
    atom = GATHER_ATOM_ROWS
    atoms_w = topk_cap // atom
    lane_rows = topk_cap // 128
    tile_rows = bz * atom // 128
    region_rows = chunk_size * topk_cap // 128
    region_tiles = region_rows // tile_rows
    num_groups = tab_ref.shape[0] // 2

    idx = topk_ref[...]
    row = c * chunk_size + tb * bz + lax.broadcasted_iota(jnp.int32, idx.shape, 0)
    col = w * topk_cap + lax.broadcasted_iota(jnp.int32, idx.shape, 1)
    phys = _window_physical_rows(
        idx,
        windows_ref,
        row * topk + col,
        page_size=page_size,
        pages_per_seq=pages_per_seq,
        total_num_pages=total_num_pages,
    )

    def region_copy(chunk, slot):
        return pltpu.make_async_copy(
            stage_ref.at[slot],
            out_hbm_ref.at[pl.ds(chunk * num_windows * region_rows, region_rows)],
            sems.at[slot],
        )

    @pl.when(w == 0)
    def _window0():
        slot = lax.rem(c, 2)

        @pl.when((tb == 0) & (c >= 2))
        def _reuse():
            region_copy(c - 2, slot).wait()

        # Strided loads need a 128-lane memref: stage column tiles row-stacked.
        for jj in range(lane_rows):
            phys_ref[pl.ds(jj * bz, bz), :] = phys[:, jj * 128 : (jj + 1) * 128]
        g = c * gpc + tb
        live = tab_ref[g]
        off = tab_ref[num_groups + g]
        dead_before = tb * atoms_w - off

        @pl.when(live == atoms_w)
        def _whole_group():
            # Every atom live: the group's rows token-major (as the
            # token-major prep lays them out), so the kernel fetches the
            # group with one DMA per cache.
            for jj in range(lane_rows):
                tok_ref[pl.ds(jj, bz, stride=lane_rows), :] = phys[
                    :, jj * 128 : (jj + 1) * 128
                ]
            start = pl.multiple_of(off * tile_rows, tile_rows)
            stage_ref[slot, pl.ds(start, bz * lane_rows), :] = tok_ref[...]

        @pl.when(live < atoms_w)
        def _blocks():
            lane = lax.broadcasted_iota(jnp.int32, (bz // 2, 128), 1)
            for a in range(atoms_w):
                j, h = divmod(a * atom, 128)
                even = phys_ref[pl.ds(j * bz, bz // 2, stride=2), :]
                odd = phys_ref[pl.ds(j * bz + 1, bz // 2, stride=2), :]
                if h == 0:
                    tile = jnp.where(lane < atom, even, pltpu.roll(odd, atom, 1))
                else:
                    tile = jnp.where(lane < atom, pltpu.roll(even, atom, 1), odd)
                pos = jnp.where(
                    a < live, off + a, region_tiles - 1 - (dead_before + a - live)
                )
                start = pl.multiple_of(pos * tile_rows, tile_rows)
                stage_ref[slot, pl.ds(start, tile_rows), :] = tile

        @pl.when(tb == gpc - 1)
        def _flush():
            region_copy(c, slot).start()

            @pl.when(c == num_chunks - 1)
            def _drain():
                region_copy(c, slot).wait()
                if num_chunks >= 2:
                    region_copy(c - 1, 1 - slot).wait()

    @pl.when(w > 0)
    def _token_major():
        for jj in range(lane_rows):
            tok_ref[pl.ds(jj, bz, stride=lane_rows), :] = phys[
                :, jj * 128 : (jj + 1) * 128
            ]
        base = ((c * num_windows + w) * chunk_size + tb * bz) * lane_rows
        cp = pltpu.make_async_copy(
            tok_ref, out_hbm_ref.at[pl.ds(base, bz * lane_rows)], sems.at[2]
        )
        cp.start()
        cp.wait()


def prepare_capped_window_live(
    topk_indices: jax.Array,  # i32[num_tokens, topk]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    live_windows: jax.Array,  # i32[] in [1, topk // topk_cap]
    tab: jax.Array,  # i32[2 * G] from `live_group_table`
    *,
    chunk_size: int,
    kernel_batch_size: int,
    topk_cap: int,
    page_size: int,
    total_num_pages: int,
) -> jax.Array:
    """`prepare_capped_window_indices` with window 0 in live-group order.

    Same physical rows, as one linear i32[num_chunks * W * chunk * topk_cap
    // 128, 128] array (the native gather's `linear_indices` operand; a chunk
    is a contiguous row range). Window 0 of each chunk holds group g's live
    atom blocks at `off(g) + a` (see `_capped_window_live_kernel`), so the
    chunk gathers only `num_live` of `live_group_table`. Windows past
    `live_windows` stay unwritten, as before.
    """
    num_tokens, topk = topk_indices.shape
    bz = kernel_batch_size
    if num_tokens % chunk_size or topk % topk_cap:
        raise ValueError(
            f"num_tokens ({num_tokens}) and topk ({topk}) must be multiples of "
            f"chunk_size ({chunk_size}) and topk_cap ({topk_cap})"
        )
    if not live_group_supported(chunk_size, bz, topk_cap):
        raise ValueError(
            f"live-group prep unsupported: chunk {chunk_size}, batch {bz}, "
            f"cap {topk_cap}"
        )
    num_windows = topk // topk_cap
    num_chunks = num_tokens // chunk_size
    gpc = chunk_size // bz
    max_num_seqs = cu_q_lens.shape[0] - 1
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    padded_pps = align_to(pages_per_seq, _GATHER_PAGE_CHUNK)
    page_table = page_indices.reshape(max_num_seqs, pages_per_seq)
    if padded_pps != pages_per_seq:
        page_table = jnp.pad(page_table, ((0, 0), (0, padded_pps - pages_per_seq)))
    tokens_per_seq = cu_q_lens[1:] - cu_q_lens[:-1]
    seq_ids_segment = jnp.repeat(
        jnp.arange(max_num_seqs), tokens_per_seq, total_repeat_length=num_tokens
    )
    windows = page_table[seq_ids_segment]  # i32[num_tokens, padded_pps]
    region_rows = chunk_size * topk_cap // 128
    kernel = functools.partial(
        _capped_window_live_kernel,
        bz=bz,
        chunk_size=chunk_size,
        topk=topk,
        topk_cap=topk_cap,
        page_size=page_size,
        pages_per_seq=pages_per_seq,
        total_num_pages=total_num_pages,
        num_windows=num_windows,
        num_chunks=num_chunks,
    )
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                pl.BlockSpec(
                    (bz, padded_pps), lambda w, c, tb, lw, tab: (c * gpc + tb, 0)
                ),
                pl.BlockSpec(
                    (bz, topk_cap), lambda w, c, tb, lw, tab: (c * gpc + tb, w)
                ),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((2, region_rows, 128), jnp.int32),
                pltpu.VMEM((topk_cap // 128 * bz, 128), jnp.int32),
                pltpu.VMEM((bz * topk_cap // 128, 128), jnp.int32),
                pltpu.SemaphoreType.DMA((3,)),
            ],
            grid=(live_windows, num_chunks, gpc),
        ),
        out_shape=jax.ShapeDtypeStruct(
            (num_chunks * num_windows * region_rows, 128), jnp.int32
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary", "arbitrary"),
            disable_bounds_checks=True,
        ),
        name="capped_window_indices_live",
    )(jnp.reshape(live_windows, (1,)).astype(jnp.int32), tab, windows, topk_indices)


def mixed_fused_prep_supported(
    num_tokens: int, topk: int, chunk_size: int, kernel_batch_size: int, topk_cap: int
) -> bool:
    """Whether `prepare_capped_window_live_fused` serves a mixed bucket: the
    live-group layout holds, and kv_lens fits whole 128-lane rows per chunk."""
    return (
        live_group_supported(chunk_size, kernel_batch_size, topk_cap)
        and num_tokens % chunk_size == 0
        and chunk_size % 128 == 0
        and topk % topk_cap == 0
        and topk % 128 == 0
    )


def _capped_window_live_fused_kernel(
    token_end_ref,  # SMEM i32[1] (scalar prefetch) end of real tokens
    windows_ref,  # VMEM i32[chunk, padded_pps] page-table row per token (chunk c)
    topk_ref,  # VMEM i32[chunk * topk // 128, 128] chunk c's top-k, 128-lane rows
    *refs,
    bz: int,
    chunk_size: int,
    topk: int,
    topk_cap: int,
    page_size: int,
    pages_per_seq: int,
    total_num_pages: int,
    num_chunks: int,
    fault: str | None,
):
    """kv_lens + `live_group_table` + `prepare_capped_window_live`, one chunk
    per grid step, reading the top-k in its 128-lane row form.

    Row ``t * (topk / 128) + j`` of the top-k operand holds slots
    ``[128 j, 128 j + 128)`` of token t: the SparseCore top-k's flat output
    as a bitcast, so no [T, topk] relayout feeds this call. Per chunk:

    1. kv_lens (live slots per token) over all `topk` slots, exact in f32;
    2. the group table (A(g), off(g)) and the chunk's gather bound exactly as
       `live_group_table` computes them, into SMEM outputs;
    3. window 0 in live-group order, exactly as `_capped_window_live_kernel`
       lays it out, and windows 1 .. W(c) - 1 token-major, where W(c) is the
       chunk's own window bound -- the only windows its gather reads
       (`live_group_table`'s `num_live`);
    4. one output array per chunk, so the chunk's gather takes it whole (no
       per-chunk slice copy of the [chunks * W * C * cap / 128, 128] array).

    Windows at or past W(c) stay unwritten; the gather never reads them.
    """
    outs = refs[:num_chunks]
    kv_ref, tab_ref, live_ref = refs[num_chunks : num_chunks + 3]
    stage_ref, phys_ref, cw_ref, ab_ref, sems = refs[num_chunks + 3 :]
    c = pl.program_id(0)
    gpc = chunk_size // bz
    atom = GATHER_ATOM_ROWS
    atoms_w = topk_cap // atom
    num_windows = topk // topk_cap
    lane_rows = topk_cap // 128  # 128-lane rows per (token, window)
    row_stride = topk // 128  # 128-lane rows per token in the top-k operand
    tile_rows = bz * atom // 128
    region_rows = chunk_size * topk_cap // 128
    region_tiles = region_rows // tile_rows
    num_groups = num_chunks * gpc
    slot = lax.rem(c, 2)
    token_end = token_end_ref[0]

    def region_copy(chunk_out, w, slot_, sem):
        return pltpu.make_async_copy(
            stage_ref.at[slot_, pl.ds(w * region_rows, region_rows)],
            chunk_out.at[pl.ds(w * region_rows, region_rows)],
            sems.at[sem],
        )

    def wait_chunk(cc, slot_):
        # Waits only need the semaphore and the byte count: any chunk output.
        region_copy(outs[0], 0, slot_, slot_ * num_windows).wait()
        for w in range(1, num_windows):

            @pl.when(w < cw_ref[cc])
            def _():
                region_copy(outs[0], w, slot_, slot_ * num_windows + w).wait()

    @pl.when(c >= 2)
    def _reuse():
        wait_chunk(c - 2, slot)

    # 1. kv_lens: tokens on sublanes, one strided load per 128-slot column.
    acc = jnp.zeros((chunk_size, 128), jnp.float32)
    for j in range(row_stride):
        acc = acc + (topk_ref[pl.ds(j, chunk_size, stride=row_stride), :] != -1).astype(
            jnp.float32
        )
    kv_col = jnp.sum(acc, axis=1, keepdims=True)  # f32[chunk, 1], exact
    for h in range(chunk_size // 128):
        blk = jnp.broadcast_to(kv_col[h * 128 : (h + 1) * 128, :], (128, 128))
        kv_row = jnp.transpose(blk)[0:1, :].astype(jnp.int32)
        if fault == "kv":  # test-only positive control
            kv_row = kv_row + (
                lax.broadcasted_iota(jnp.int32, kv_row.shape, 1) == 5
            ).astype(jnp.int32)
        kv_ref[pl.ds(c * (chunk_size // 128) + h, 1), :] = kv_row
    kv_i = kv_col.astype(jnp.int32)

    # 2. Group table and the chunk's gather bound (`live_group_table`).
    chunk_start = c * chunk_size
    valid_rows = jnp.clip(token_end - chunk_start, 0, chunk_size)
    grid_rows = (valid_rows + bz - 1) // bz * bz
    local = lax.broadcasted_iota(jnp.int32, (chunk_size, 1), 0)
    token_windows = jnp.where(
        local < grid_rows, (jnp.maximum(kv_i, 1) + topk_cap - 1) // topk_cap, 1
    )
    chunk_windows = jnp.clip(jnp.max(token_windows), 1, num_windows)
    cw_ref[c] = chunk_windows
    off = jnp.int32(0)
    for tb in range(gpc):
        kv_max = jnp.max(kv_i[tb * bz : (tb + 1) * bz, :])
        gridded = chunk_start + tb * bz < token_end
        a_live = jnp.where(
            gridded, jnp.minimum((kv_max + atom - 1) // atom, atoms_w), 0
        )
        g = c * gpc + tb
        tab_ref[g] = a_live
        tab_ref[num_groups + g] = off
        ab_ref[tb] = a_live
        ab_ref[gpc + tb] = off
        off = off + a_live
        if fault == "off" and tb == 0:  # test-only positive control
            off = off + 1
    live_ref[c] = jnp.where(
        chunk_windows == 1,
        off * (bz * atom),
        (chunk_windows - 1) * chunk_size * topk_cap + grid_rows * topk_cap,
    )

    def translate(w, jj):
        # Column tile jj of window w for every token of the chunk: one wide
        # vector pass (tokens on sublanes), not one per 16-token group.
        idx = topk_ref[pl.ds(w * lane_rows + jj, chunk_size, stride=row_stride), :]
        row = chunk_start + lax.broadcasted_iota(jnp.int32, idx.shape, 0)
        col = w * topk_cap + jj * 128 + lax.broadcasted_iota(jnp.int32, idx.shape, 1)
        return _window_physical_rows(
            idx,
            windows_ref,
            row * topk + col,
            page_size=page_size,
            pages_per_seq=pages_per_seq,
            total_num_pages=total_num_pages,
        )

    # 3a. Window 0 in live-group order (`_capped_window_live_kernel`); phys
    # column tile jj of token t at row jj * chunk + t.
    for jj in range(lane_rows):
        phys_ref[pl.ds(jj * chunk_size, chunk_size), :] = translate(0, jj)

    def group_body(tb, carry):
        live = ab_ref[tb]
        goff = ab_ref[gpc + tb]
        dead_before = tb * atoms_w - goff
        g0 = pl.multiple_of(tb * bz, bz)

        @pl.when(live == atoms_w)
        def _whole_group():
            start = pl.multiple_of(goff * tile_rows, tile_rows)
            for jj in range(lane_rows):
                stage_ref[slot, pl.ds(start + jj, bz, stride=lane_rows), :] = phys_ref[
                    pl.ds(jj * chunk_size + g0, bz), :
                ]

        @pl.when(live < atoms_w)
        def _blocks():
            lane = lax.broadcasted_iota(jnp.int32, (bz // 2, 128), 1)
            for a in range(atoms_w):
                j, hh = divmod(a * atom, 128)
                even = phys_ref[pl.ds(j * chunk_size + g0, bz // 2, stride=2), :]
                odd = phys_ref[pl.ds(j * chunk_size + g0 + 1, bz // 2, stride=2), :]
                if hh == 0:
                    tile = jnp.where(lane < atom, even, pltpu.roll(odd, atom, 1))
                else:
                    tile = jnp.where(lane < atom, pltpu.roll(even, atom, 1), odd)
                pos = jnp.where(
                    a < live, goff + a, region_tiles - 1 - (dead_before + a - live)
                )
                start = pl.multiple_of(pos * tile_rows, tile_rows)
                stage_ref[slot, pl.ds(start, tile_rows), :] = tile

        return carry

    lax.fori_loop(0, gpc, group_body, 0)

    # 3b. Windows 1 .. W(c) - 1, token-major (overflowing chunks only).
    for w in range(1, num_windows if fault != "win1" else 1):

        @pl.when(w < chunk_windows)
        def _window():
            for jj in range(lane_rows):
                stage_ref[
                    slot, pl.ds(w * region_rows + jj, chunk_size, stride=lane_rows), :
                ] = translate(w, jj)

    # 4. This chunk's own output: start its live windows' DMAs.
    for cc in range(num_chunks):

        @pl.when(c == cc)
        def _flush():
            region_copy(outs[cc], 0, slot, slot * num_windows).start()
            for w in range(1, num_windows):

                @pl.when(w < chunk_windows)
                def _():
                    region_copy(outs[cc], w, slot, slot * num_windows + w).start()

    @pl.when(c == num_chunks - 1)
    def _drain():
        wait_chunk(c, slot)
        if num_chunks >= 2:
            wait_chunk(c - 1, 1 - slot)


def prepare_capped_window_live_fused(
    topk_indices: jax.Array,  # i32[num_tokens, topk], -1 = empty slot
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    token_end: jax.Array,  # i32[]
    *,
    chunk_size: int,
    kernel_batch_size: int,
    topk_cap: int,
    page_size: int,
    total_num_pages: int,
    _fault: str | None = None,
) -> tuple[tuple[jax.Array, ...], jax.Array, jax.Array, jax.Array]:
    """kv_lens, `live_group_table` and `prepare_capped_window_live` in one call.

    Reads the top-k as 128-lane rows (a bitcast of the SparseCore top-k's flat
    output, so the [T, topk] relayout, the kv_lens reduce and the group-table
    fusions all go) and writes one window array per chunk (so the per-chunk
    slice fusion goes too).

    Returns:
      rows: per chunk, i32[W * chunk * topk_cap // 128, 128], the chunk's rows
        of `prepare_capped_window_live` on every window its gather reads
        (window 0 always; windows < W(c) when the chunk overflows).
      kv_lens: i32[num_tokens], `sum(topk_indices != -1, -1)`.
      tab, num_live: `live_group_table(kv_lens, token_end, ...)`.

    `_fault` is a test-only positive control ("kv", "off", "win1").
    """
    num_tokens, topk = topk_indices.shape
    bz = kernel_batch_size
    if not mixed_fused_prep_supported(num_tokens, topk, chunk_size, bz, topk_cap):
        raise ValueError(
            f"mixed fused prep unsupported: tokens {num_tokens}, topk {topk}, "
            f"chunk {chunk_size}, batch {bz}, cap {topk_cap}"
        )
    num_windows = topk // topk_cap
    num_chunks = num_tokens // chunk_size
    gpc = chunk_size // bz
    max_num_seqs = cu_q_lens.shape[0] - 1
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    padded_pps = align_to(pages_per_seq, _GATHER_PAGE_CHUNK)
    page_table = page_indices.reshape(max_num_seqs, pages_per_seq)
    if padded_pps != pages_per_seq:
        page_table = jnp.pad(page_table, ((0, 0), (0, padded_pps - pages_per_seq)))
    tokens_per_seq = cu_q_lens[1:] - cu_q_lens[:-1]
    seq_ids_segment = jnp.repeat(
        jnp.arange(max_num_seqs), tokens_per_seq, total_repeat_length=num_tokens
    )
    windows = page_table[seq_ids_segment]  # i32[num_tokens, padded_pps]
    region_rows = chunk_size * topk_cap // 128
    row_stride = topk // 128
    kernel = functools.partial(
        _capped_window_live_fused_kernel,
        bz=bz,
        chunk_size=chunk_size,
        topk=topk,
        topk_cap=topk_cap,
        page_size=page_size,
        pages_per_seq=pages_per_seq,
        total_num_pages=total_num_pages,
        num_chunks=num_chunks,
        fault=_fault,
    )
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    outs = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec((chunk_size, padded_pps), lambda c, te: (c, 0)),
                pl.BlockSpec((chunk_size * row_stride, 128), lambda c, te: (c, 0)),
            ],
            out_specs=[
                *([hbm] * num_chunks),
                pl.BlockSpec((num_tokens // 128, 128), lambda c, te: (0, 0)),
                smem,
                smem,
            ],
            scratch_shapes=[
                pltpu.VMEM((2, num_windows * region_rows, 128), jnp.int32),
                pltpu.VMEM((topk_cap // 128 * chunk_size, 128), jnp.int32),
                pltpu.SMEM((num_chunks,), jnp.int32),
                pltpu.SMEM((2 * gpc,), jnp.int32),
                pltpu.SemaphoreType.DMA((2 * num_windows,)),
            ],
            grid=(num_chunks,),
        ),
        out_shape=[
            *(
                [jax.ShapeDtypeStruct((num_windows * region_rows, 128), jnp.int32)]
                * num_chunks
            ),
            jax.ShapeDtypeStruct((num_tokens // 128, 128), jnp.int32),
            jax.ShapeDtypeStruct((2 * num_tokens // bz,), jnp.int32),
            jax.ShapeDtypeStruct((num_chunks,), jnp.int32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=True,
            vmem_limit_bytes=32 * 1024 * 1024,
        ),
        name="capped_window_indices_live_fused",
    )(
        jnp.reshape(token_end, (1,)).astype(jnp.int32),
        windows,
        topk_indices.reshape(num_tokens * row_stride, 128),
    )
    rows = tuple(outs[:num_chunks])
    kv_lens, tab, num_live = outs[num_chunks:]
    return rows, kv_lens.reshape(num_tokens), tab, num_live


# Decode buckets the fused window prep serves (one chunk, whole top-k block in
# VMEM: 32 tokens x 2048 x i32 = 256 KiB).
FUSED_PREP_MAX_TOKENS = 32
# Rows the fused prep appends after the window rows: one (8, 128) i32 tile
# whose first row is the native gather's live vector (lanes [0, 16) the live
# logical-batch count, [16, 32) the per-owner trip count), read by the gather
# from the same flat operand (`dsa_gather_native_sc(live_vec_offset=...)`).
FUSED_PREP_META_ROWS = 8


def _cdiv_const(x: jax.Array, d: int) -> jax.Array:
    """ceil(x / d) for a non-negative int32 vector and a static positive d."""
    if d & (d - 1) == 0:
        return lax.shift_right_logical(x + (d - 1), jnp.int32(d.bit_length() - 1))
    return _floor_divmod_const(x + (d - 1), d)[0]


def _capped_window_fused_kernel(
    token_end_ref,  # i32[1] (scalar prefetch) end of real tokens
    windows_ref,  # i32[T, padded_pps] page-table row per token
    topk_ref,  # i32[T, topk] logical top-k, -1 = empty slot
    rows_hbm_ref,  # i32[W * T * topk_cap // 128 + META, 128] out: rows + live vector
    rows_vmem_ref,  # i32[W * T * topk_cap // 128 + META, 128] scratch
    sems,  # DMA[W + 1]
    *,
    topk_cap: int,
    page_size: int,
    pages_per_seq: int,
    total_num_pages: int,
    kernel_batch_size: int,
    occurrences_per_batch: int,
    num_batches: int,
    num_owners: int,
    num_lanes: int,
):
    """One-call capped window prep for a single decode chunk.

    Counts each token's live top-k entries on the vector unit (the kv_lens
    the attention kernel gets from XLA, recomputed here so this call does not
    wait on it), derives the live-window bound (`capped_live_windows`) and the
    native SparseCore gather's live vector (`dsa_gather_native_sc`'s
    `num_live_indices` path) from it, and translates the live windows' slots
    to physical rows (`_window_physical_rows`). The rows are written
    window-major exactly as `prepare_capped_window_indices` lays them out, but
    as 128-lane rows (row ``(w * T + t) * k + j`` holds slots
    ``[128 j, 128 j + 128)`` of token t's window w, ``k = topk_cap / 128``):
    that 2-D array is the gather operand's linear layout, so XLA hands it over
    without a relayout. The live vector rides in a trailing tile of the same
    array, so the call has one output and every output DMA is in flight at
    once (each separately waited output costs a full DMA round trip). Windows
    past the live bound stay unwritten, as in the per-window prep; the gather
    never reads them.
    """
    num_tokens, topk = topk_ref.shape
    num_windows = topk // topk_cap
    lane_rows = topk_cap // 128  # 128-lane rows per (token, window)
    window_rows = num_tokens * lane_rows
    meta_row = num_windows * window_rows

    def copy(start, size, sem):
        return pltpu.make_async_copy(
            rows_vmem_ref.at[pl.ds(start, size)],
            rows_hbm_ref.at[pl.ds(start, size)],
            sems.at[sem],
        )

    def translate(w):
        idx = topk_ref[:, w * topk_cap : (w + 1) * topk_cap]
        row = lax.broadcasted_iota(jnp.int32, idx.shape, 0)
        col = w * topk_cap + lax.broadcasted_iota(jnp.int32, idx.shape, 1)
        phys = _window_physical_rows(
            idx,
            windows_ref,
            row * topk + col,
            page_size=page_size,
            pages_per_seq=pages_per_seq,
            total_num_pages=total_num_pages,
        )
        # Token t's 128-lane slice j -> row w * window_rows + t * k + j.
        for j in range(lane_rows):
            rows_vmem_ref[
                pl.ds(w * window_rows + j, num_tokens, stride=lane_rows), :
            ] = phys[:, j * 128 : (j + 1) * 128]
        copy(w * window_rows, window_rows, w).start()

    # Window 0 is always live: translate it before the bound is known.
    translate(0)

    # Live windows over the rows the attention grid reaches (exactly
    # `capped_live_windows`), then the gather's live bound (exactly
    # `sparse_attention_chunk`'s `num_live_capped` and the gather's live_b /
    # owner_b). Counts are exact in f32 (<= topk).
    valid_rows = jnp.clip(token_end_ref[0], 0, num_tokens)
    grid_rows = cdiv(valid_rows, kernel_batch_size) * kernel_batch_size
    kv = jnp.sum(
        (topk_ref[...] != -1).astype(jnp.float32), axis=1, keepdims=True
    ).astype(jnp.int32)
    token = lax.broadcasted_iota(jnp.int32, kv.shape, 0)
    token_windows = jnp.where(
        token < grid_rows, _cdiv_const(jnp.maximum(kv, 1), topk_cap), 1
    )
    live_windows = jnp.minimum(jnp.max(token_windows), num_windows)
    num_live = (live_windows - 1) * num_tokens * topk_cap + grid_rows * topk_cap
    live_batches = jnp.minimum(cdiv(num_live, occurrences_per_batch), num_batches)
    owner_batches = cdiv(live_batches, num_owners)
    lane = lax.broadcasted_iota(jnp.int32, (FUSED_PREP_META_ROWS, 128), 1)
    rows_vmem_ref[pl.ds(meta_row, FUSED_PREP_META_ROWS), :] = jnp.where(
        lane < num_lanes, live_batches, owner_batches
    )
    meta_copy = copy(meta_row, FUSED_PREP_META_ROWS, num_windows)
    meta_copy.start()

    for w in range(1, num_windows):
        pl.when(w < live_windows)(functools.partial(translate, w))

    copy(0, window_rows, 0).wait()
    for w in range(1, num_windows):

        @pl.when(w < live_windows)
        def _wait():
            copy(w * window_rows, window_rows, w).wait()

    meta_copy.wait()


def prepare_capped_window_fused(
    topk_indices: jax.Array,  # i32[num_tokens, topk], -1 = empty slot
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    token_end: jax.Array,  # i32[]
    *,
    topk_cap: int,
    page_size: int,
    total_num_pages: int,
    kernel_batch_size: int,
    gather_atoms: int,
) -> jax.Array:
    """Window indices + gather live vector for one decode chunk, in one call.

    Replaces `capped_live_windows` + `prepare_capped_window_indices` + the
    gather's live-count arithmetic and the index relayouts for a chunk that
    is the whole (decode) bucket. Needs no kv_lens operand.

    Returns:
        rows: i32[W * num_tokens * topk_cap // 128 + FUSED_PREP_META_ROWS, 128]:
          rows [0, W * num_tokens * topk_cap // 128) are the same physical
          rows as `prepare_capped_window_indices` (window-major), the native
          gather's linear operand; the next row's lanes [0, 2 * NUM_LANES)
          are its live vector, at flat offset ``num_tokens * topk``
          (`dsa_gather_native_sc(live_vec_offset=num_tokens * topk)`).
    """
    num_tokens, topk = topk_indices.shape
    if num_tokens > FUSED_PREP_MAX_TOKENS:
        raise ValueError(
            f"fused window prep serves <= {FUSED_PREP_MAX_TOKENS} tokens, "
            f"got {num_tokens}"
        )
    if topk % topk_cap or topk_cap % 128:
        raise ValueError(
            f"topk_cap ({topk_cap}) must be a multiple of 128 dividing topk ({topk})"
        )
    num_windows = topk // topk_cap
    max_num_seqs = cu_q_lens.shape[0] - 1
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    padded_pps = align_to(pages_per_seq, _GATHER_PAGE_CHUNK)
    page_table = page_indices.reshape(max_num_seqs, pages_per_seq)
    if padded_pps != pages_per_seq:
        page_table = jnp.pad(page_table, ((0, 0), (0, padded_pps - pages_per_seq)))
    tokens_per_seq = cu_q_lens[1:] - cu_q_lens[:-1]
    seq_ids_segment = jnp.repeat(
        jnp.arange(max_num_seqs), tokens_per_seq, total_repeat_length=num_tokens
    )
    windows = page_table[seq_ids_segment]  # i32[num_tokens, padded_pps]

    out_size = num_tokens * topk
    num_cores, num_subcores, _, occurrences_per_batch, num_batches = (
        native_sc_gather.live_vec_geometry(out_size, gather_atoms)
    )
    num_lanes = native_sc_gather.NUM_LANES
    if 2 * num_lanes > 128:
        raise ValueError(f"live vector ({2 * num_lanes}) must fit one 128-lane row")
    total_rows = num_windows * num_tokens * (topk_cap // 128) + FUSED_PREP_META_ROWS
    kernel = functools.partial(
        _capped_window_fused_kernel,
        topk_cap=topk_cap,
        page_size=page_size,
        pages_per_seq=pages_per_seq,
        total_num_pages=total_num_pages,
        kernel_batch_size=kernel_batch_size,
        occurrences_per_batch=occurrences_per_batch,
        num_batches=num_batches,
        num_owners=num_cores * num_subcores,
        num_lanes=num_lanes,
    )
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=[
                pl.BlockSpec((num_tokens, padded_pps), lambda i, te: (0, 0)),
                pl.BlockSpec((num_tokens, topk), lambda i, te: (0, 0)),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((total_rows, 128), jnp.int32),
                pltpu.SemaphoreType.DMA((num_windows + 1,)),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct((total_rows, 128), jnp.int32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=True,
        ),
        name="capped_window_indices_fused",
    )(
        jnp.reshape(token_end, (1,)).astype(jnp.int32),
        windows,
        topk_indices,
    )


def live_group_supported(
    chunk_size: int, kernel_batch_size: int, topk_cap: int
) -> bool:
    """Whether the live-group capped path (`prepare_capped_window_live`) serves
    a chunk: whole grid steps per chunk, a grid step's 64-slot atom block
    (`kernel_batch_size * 64` indices) is whole (8, 128) i32 tiles, and the
    window is whole 128-lane rows."""
    atom = GATHER_ATOM_ROWS
    tile = kernel_batch_size * atom
    return (
        chunk_size % kernel_batch_size == 0
        and tile % (8 * 128) == 0
        and topk_cap % 128 == 0
        and topk_cap % atom == 0
        and tile % topk_cap == 0
    )


def live_group_table(
    kv_lens: jax.Array,  # i32[num_tokens]
    token_end: jax.Array,  # i32[]
    *,
    chunk_size: int,
    kernel_batch_size: int,
    topk_cap: int,
    num_windows: int,
) -> tuple[jax.Array, jax.Array]:
    """Per grid step (`kernel_batch_size` tokens, "group") live atoms of window 0.

    Group g's live atoms A(g) = cdiv(max kv_len in g, 64), clipped to the
    window's atoms, and 0 for groups the attention grid never reaches. Its
    atom blocks a < A(g) sit at block `off(g) + a` of its chunk's window 0
    (`off` = exclusive running sum of A within the chunk), the layout
    `prepare_capped_window_live` writes.

    Returns:
      tab: i32[2 * G]: A at [0, G), off at [G, 2G) (G = num_tokens / bz).
      num_live: i32[num_chunks], the chunk gather's live-index bound: its
        live atom blocks (`sum A * bz * 64`) when no gridded token overflows
        window 0, else every window up to the last live one, exactly as the
        token-major path bounds it.
    """
    num_tokens = kv_lens.shape[0]
    bz = kernel_batch_size
    atom = GATHER_ATOM_ROWS
    atoms_w = topk_cap // atom
    num_groups = num_tokens // bz
    num_chunks = num_tokens // chunk_size
    gpc = chunk_size // bz
    group_start = np.arange(num_groups, dtype=np.int32) * bz
    gridded = group_start < token_end
    kv_max = jnp.max(kv_lens.reshape(num_groups, bz), axis=1)
    live_atoms = jnp.where(
        gridded, jnp.minimum((kv_max + atom - 1) // atom, atoms_w), 0
    ).astype(jnp.int32)
    per_chunk = live_atoms.reshape(num_chunks, gpc)
    off = (jnp.cumsum(per_chunk, axis=1) - per_chunk).reshape(num_groups)
    live_blocks = jnp.sum(per_chunk, axis=1)
    # Windows per chunk (gridded tokens only), as `capped_live_windows`.
    chunk_start = np.arange(num_chunks, dtype=np.int32) * chunk_size
    valid_rows = jnp.clip(token_end - chunk_start, 0, chunk_size)
    grid_rows = (valid_rows + bz - 1) // bz * bz
    local = np.arange(num_tokens, dtype=np.int32).reshape(num_chunks, chunk_size)
    local = local % chunk_size
    token_windows = jnp.where(
        local < grid_rows[:, None],
        (jnp.maximum(kv_lens.reshape(num_chunks, chunk_size), 1) + topk_cap - 1)
        // topk_cap,
        1,
    )
    chunk_windows = jnp.clip(jnp.max(token_windows, axis=1), 1, num_windows)
    num_live = jnp.where(
        chunk_windows == 1,
        live_blocks * (bz * atom),
        (chunk_windows - 1) * chunk_size * topk_cap + grid_rows * topk_cap,
    ).astype(jnp.int32)
    tab = jnp.concatenate([live_atoms, off.astype(jnp.int32)])
    return tab, num_live


def capped_live_windows(
    kv_lens: jax.Array,  # i32[num_tokens]
    token_end: jax.Array,  # i32[]
    *,
    chunk_size: int,
    kernel_batch_size: int,
    topk_cap: int,
    num_windows: int,
) -> jax.Array:
    """Calculates how many `topk_cap` (512-slot) windows the capped path must process.

    Instead of blindly gathering all possible windows (e.g., 4 windows = 2,048 slots),
    this function inspects all active query tokens in the batch and finds the
    maximum number of windows required by any single token.

    Args:
        kv_lens: Array of shape [num_tokens] containing the number of valid KV
            cache entries each query token owns on this DCP rank.
        token_end: Scalar integer marking the end of real tokens in the buffer.
            Tokens at or beyond this index are padding.
        chunk_size: Token chunk size processed per attention loop step (e.g., 256).
        kernel_batch_size: Hardware execution tile size (e.g., 8 or 16 rows)
            processed together by the TPU vector units.
        topk_cap: Capacity of a single window (e.g., 512 slots).
        num_windows: Maximum number of windows possible (e.g., 4 for 2,048 slots).

    Returns:
        A scalar i32 in the range [1, num_windows].
        - 1 if all active tokens fit within the first window (<= topk_cap keys).
        - >1 only if at least one active token overflows into later windows.
        The window prep translates this many windows per chunk, and no chunk's
        gather reaches past them.
    """
    num_tokens = kv_lens.shape[0]

    # Map each token to its offset and start position within its chunk.
    local = np.arange(num_tokens, dtype=np.int32) % chunk_size
    start = np.arange(num_tokens, dtype=np.int32) - local

    # Count how many real tokens fall within each chunk before hitting token_end.
    valid_rows = jnp.clip(token_end - start, 0, chunk_size)

    # Round up valid rows to the hardware tile size (kernel_batch_size)
    # because TPU vector units execute complete tiles, not individual rows.
    grid_rows = (valid_rows + kernel_batch_size - 1) // kernel_batch_size
    grid_rows = grid_rows * kernel_batch_size

    # Compute windows needed per token:
    # - If inside the active hardware grid: ceil(kv_len / topk_cap).
    # - Outside it (padding, rows the grid never reaches): 1, so they never raise
    #   the bound.
    token_windows = jnp.where(
        local < grid_rows,
        (jnp.maximum(kv_lens, 1) + topk_cap - 1) // topk_cap,
        1,
    )

    # The worst case over all active tokens, clamped to [1, num_windows].
    return jnp.clip(jnp.max(token_windows), 1, num_windows).astype(jnp.int32)


# `body_levers` the live-group body accepts (`TPU_DCP_MLA_MIXED_BODY`).
# In-kernel q rope (TPU_MLA_DECODE_PROLOGUE): the raw q head is nope 192 |
# rope 64 = 256 lanes; lanes 128:256 (one vreg column) hold q_pe at 64:128.
Q_ROPE_DIM = 64
Q_ROPE_HEAD_DIM = 256
# Test-only faults for the in-kernel q rope / kernel-owned lse.
PROLOGUE_FAULTS = (
    None,
    "skip_q_rope",
    "rope_swap",
    "rope_f32",
    "skip_lse_fill",
    # `row_widths`: attend one width below what the batch needs (test-only).
    "row_width_undersize",
)


def rope_q_block(x: jax.Array, tab: jax.Array, fault: str | None = None) -> jax.Array:
    """NeoX rope of the q_pe lanes of one raw head block, as `[q_pe | 0]`.

    `x` is `[heads, 128]` = lanes 128:256 of the raw q head (q_nope[128:192]
    at lanes 0:64, q_pe at 64:128), `tab` f32 `[2, 128]` = (C, S) with
    C[64:128] = [cos, cos] and S[64:128] = [-sin, sin]. Returns `[heads, 128]`
    in x's dtype: the roped q_pe in lanes 0:64 and zeros in 64:128 -- exactly
    `pad(concat(o1, o2))` with o1 = x1*cos - x2*sin, o2 = x2*cos + x1*sin.
    XLA on TPU evaluates that bf16 expression op by op -- each product is
    rounded to bf16 before the add (measured: 0 of 131k elements differ, vs
    32 % for a single rounding of the f32 sum) -- so the kernel does the same.
    """
    xf = x.astype(jnp.float32)
    lane = lax.broadcasted_iota(jnp.int32, xf.shape, 1)
    # rot(x)[64:96] = x2 = x[96:128] (roll by 96 == by -32);
    # rot(x)[96:128] = x1 = x[64:96] (roll by 32).
    lo, hi = (32, 96) if fault == "rope_swap" else (96, 32)
    rot = jnp.where(lane < 96, pltpu.roll(xf, lo, 1), pltpu.roll(xf, hi, 1))
    c = tab[0:1, :]
    s = tab[1:2, :]
    if fault == "skip_q_rope":
        r = xf
    elif fault == "rope_f32":
        # Positive control: one rounding of the f32 sum (excess precision).
        r = xf * c + rot * s
    else:
        # Each product rounded to x's dtype, then the add (rounded below).
        a = (xf * c).astype(x.dtype).astype(jnp.float32)
        b = (rot * s).astype(x.dtype).astype(jnp.float32)
        r = a + b
    r = pltpu.roll(r, 64, 1)
    return jnp.where(lane < 64, r, 0.0).astype(x.dtype)


MIXED_BODY_LEVERS = ("tr", "zu", "pv512", "noinf", "pbf16")
# The exact (bitwise) default set for TPU_DCP_MLA_MIXED_BODY=1.
MIXED_BODY_DEFAULT = ("tr", "zu", "pv512", "noinf")
# Buckets up to this many tokens keep the plain live body: at 64 tokens the
# levers measured within noise to +2.5 % (partly filled bucket).
MIXED_BODY_MAX_SKIP_TOKENS = 64


def _attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_tokens] valid top-k count per query token
    start_end_seq_idx_ref,  # [2] (start_seq_idx, end_seq_idx)
    sem_ids_ref,  # [2] (bi_sem_idx, bo_sem_idx)
    # Input
    q_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    # [num_tokens, topk * WORD_BYTES, TILE_LANE_BYTES] (gathered, raw tiles)
    cache_kv_nope_hbm_ref,
    cache_kv_rope_hbm_ref,  # [num_tokens, topk, rope_dim] (gathered)
    # Output (+ scratch) refs are positional and their count depends on
    # `return_lse`, so they arrive packed and are unpacked below.
    *refs,
    sm_scale: float,
    k_scale: float = 1.0,
    batch_size: int = 1,
    preserve_suffix: bool = False,
    return_lse: bool = False,
    sparsecore_layout: bool = False,
    capped_windows: int | None = None,
    split_q: bool = False,
    lse_lane_dense: bool = False,
    lse_mask_empty: bool = False,
    group_tab_ref=None,
    live_widths: tuple[int, ...] = (),
    body_levers: tuple[str, ...] = (),
    q_rope: bool = False,
    lse_fill: bool = False,
    fault: str | None = None,
    row_widths: tuple[int, ...] = (),
):
    # Row widths (`row_widths`, TPU_MLA_DECODE_ROW_WIDTHS; not live-group):
    # ascending static row counts (multiples of 128 below bkv_sz). A grid step
    # whose tokens all have kv_len <= w fetches only the first w rows of each
    # gathered tile and runs its body on them -- unpack, zero, QK, mask,
    # softmax and PV -- instead of all bkv_sz. Rows past kv_len are zeroed and
    # masked either way, so every result is the same bit for bit (the dropped
    # rows only ever add exact zeros). The width is chosen from the scalar
    # kv_lens, identically for a batch's fetch, wait and body.
    #
    # In-kernel q rope (`q_rope`, split-q only; TPU_MLA_DECODE_PROLOGUE):
    # `q_pe_hbm_ref` is the raw (un-roped) q head view `[tokens, heads, 256]`
    # (nope 192 | rope 64) and a second input `tab_hbm_ref` f32[tokens, 2, 128]
    # holds per-token C = [cos, cos, cos, cos], S = [-sin, sin, -sin, sin]
    # (each with its own VMEM double buffer, the scratch ref before q_pe's).
    # `load_bq` takes lanes 128:256 of each head (q_nope[128:192] | q_pe),
    # rotates the NeoX halves of q_pe with lane rolls, forms x*C + rot(x)*S
    # in f32 (bf16 products are exact there; one add, one RNE rounding) and
    # rolls the result down into the `[q_pe | 0]` block the caller's rope +
    # concat + pad used to build.
    #
    # Kernel-owned lse (`lse_fill`, lane-dense only): there is no aliased lse
    # input; grid step 0 DMAs the sentinel over every row of the output
    # (`fill_ref` / `fill_sem`, after the capped fix-up scratch) and waits for
    # it before its own first lse store. The grid is never empty (the caller
    # rounds it up to one step); a step past the live rows does nothing else.
    #
    # Split-q mode (`split_q`): `q_hbm_ref` holds q_nope only and the rope
    # part arrives as a separate input `q_pe_hbm_ref` (first of `refs`), with
    # its own VMEM double buffer (last scratch ref). The two are joined and
    # zero-padded in VMEM, so the scores see exactly the operand the caller's
    # concat used to build. The output is q_nope's width and aliases it.
    #
    # Lane-dense lse (`lse_lane_dense`): `lse_hbm_ref` is f32[tokens, heads]
    # and the blse buffer f32[2, bsz, heads]; each token's [heads, 1] column
    # is moved to a row exactly (a masked max) before it is stored.
    #
    # Empty-row lse (`lse_mask_empty`): a token whose kv_len is 0 (this DCP
    # rank owns none of its top-k) attends nothing -- every key is zeroed and
    # masked, so its output is 0 -- and its lse is written as -inf, the
    # weight-0 value the cross-rank merge expects.
    #
    # Capped mode (`capped_windows` = W set, SparseCore layout only): the
    # gathered buffers hold W windows of `bkv_sz` rows per token, window-major
    # (`[W * chunk, bkv_sz, ...]`), and the pipelined body attends window 0
    # exactly like the uncapped path. Tokens with more than `bkv_sz` live rows
    # are recomputed over all their windows by an online-softmax fix-up with
    # double-buffered window DMAs (two-slot scratch trails the usual scratch).
    q_pe_hbm_ref = None
    bq_pe_x2_ref = None
    tab_hbm_ref = None
    bq_tab_x2_ref = None
    if split_q:
        q_pe_hbm_ref = refs[0]  # [max_num_tokens, num_q_heads, rope_dim]
        bq_pe_x2_ref = refs[-1]  # [2, bsz, num_q_heads, rope_dim]
        refs = refs[1:-1]
        if q_rope:
            tab_hbm_ref = refs[0]  # f32[max_num_tokens, 2, 128]
            bq_tab_x2_ref = refs[-1]  # f32[2, bsz, 2, 128]
            refs = refs[1:-1]
    assert not q_rope or split_q
    fill_ref = None
    fill_sem = None
    if lse_fill:
        assert return_lse and lse_lane_dense
        fill_ref, fill_sem = refs[-2], refs[-1]
        refs = refs[:-2]
    # Live-group mode (`group_tab_ref` given; capped only): window 0 of the
    # gathered buffers is in live-group order (`prepare_capped_window_live`)
    # and arrives also as atom-block views (the two inputs after q_pe). Grid
    # step g fetches only its A(g) live 64-row atom blocks, one DMA per
    # (block, cache) for all its tokens; the rest of each token's VMEM tile
    # is stale and is zeroed / masked exactly like the `-1` tail.
    live_nope_hbm_ref = None
    live_rope_hbm_ref = None
    if group_tab_ref is not None:
        assert capped_windows is not None
        live_nope_hbm_ref = refs[0]  # [blocks, bsz, 64, 128]
        live_rope_hbm_ref = refs[1]  # [blocks, bsz, 16, 128]
        refs = refs[2:]
    # Live widths (`live_widths`, live-group mode only): ascending static row
    # counts (multiples of 128 below bkv_sz). A grid step whose live rows
    # (A(g) * 64) fit width w runs its whole 16-token body on the first w rows
    # of each tile -- unpack, zero, QK, mask, softmax and PV -- instead of all
    # bkv_sz; rows past a token's kv_len are zero / masked either way, so the
    # result is the same. Steps wider than every width run the full body.
    assert not live_widths or group_tab_ref is not None
    assert not row_widths or group_tab_ref is None
    # Body levers (`body_levers`, live-group mode only; `MIXED_BODY_LEVERS`):
    #   "tr"     unpack the NOPE words with four stride-4 row loads and a 4x4
    #            byte transpose on 32-bit words, then a native bitcast to fp8,
    #            instead of the u8 reshape relayout (same bytes, same layout).
    #   "zu"     zero keys past kv_len on the 32-bit NOPE words (per row) only;
    #            ROPE rows past kv_len are left as gathered -- they reach only
    #            QK scores that the kv_len mask replaces, while PV reads NOPE.
    #   "pv512"  PV contracts against the NOPE columns only (the output keeps
    #            just those), instead of all head_dim columns.
    #   "noinf"  drop the artificial prev_out -> q dependency.
    #   "pbf16"  feed P to the PV matmul as q's dtype (not bit-exact).
    assert not body_levers or group_tab_ref is not None
    assert set(body_levers) <= set(MIXED_BODY_LEVERS), body_levers
    fix_refs = None
    if capped_windows is not None:
        assert sparsecore_layout
        fix_refs = refs[-6:]
        refs = refs[:-6]
    if lse_fill:
        # No aliased lse input: the kernel defines every row itself.
        refs = (None, *refs)
    if return_lse:
        (
            _lse_in_hbm_ref,  # aliased with lse_hbm_ref; never read
            o_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
            lse_hbm_ref,  # f32[max_num_tokens, num_q_heads, 1]
            bkv_nope_x2_ref,  # [2, bsz, topk * WORD_BYTES, TILE_LANE_BYTES]
            bkv_rope_x2_ref,  # [2, bsz, topk, rope_dim]
            bq_x2_ref,  # [2, bsz, num_q_heads, head_dim]
            bo_x2_ref,  # [2, bsz, num_q_heads, head_dim]
            blse_x2_ref,  # f32[2, bsz, num_q_heads, 1]
            sems,  # [5, 2]
        ) = refs
    else:
        (
            o_hbm_ref,
            bkv_nope_x2_ref,
            bkv_rope_x2_ref,
            bq_x2_ref,
            bo_x2_ref,
            sems,  # [4, 2]
        ) = refs
        lse_hbm_ref = None
        blse_x2_ref = None

    assert o_hbm_ref is not None
    assert q_hbm_ref.shape == o_hbm_ref.shape

    num_tokens, num_q_heads, head_dim = q_hbm_ref.shape
    # Width of the stored output (q_nope's in split-q mode, else head_dim).
    o_dim = head_dim
    if split_q:
        assert q_pe_hbm_ref is not None
        pe_dim = Q_ROPE_DIM if q_rope else q_pe_hbm_ref.shape[-1]
        head_dim = align_to(head_dim + pe_dim, 128)
    nope_dim = kv_cache_utils.WORD_BYTES * kv_cache_utils.TILE_LANE_BYTES
    assert kv_lens_ref.shape[0] == num_tokens
    bkv_sz = (
        cache_kv_nope_hbm_ref.shape[1]
        if sparsecore_layout
        else cache_kv_nope_hbm_ref.shape[1] // kv_cache_utils.WORD_BYTES
    )

    q_dtype = q_hbm_ref.dtype
    q_packing = get_dtype_packing(q_dtype)
    # Validate against the KV dtype.
    assert o_hbm_ref.dtype == q_dtype

    assert head_dim % 128 == 0
    assert num_q_heads % q_packing == 0

    start_seq_idx = start_end_seq_idx_ref[0]
    end_seq_idx = start_end_seq_idx_ref[1]
    if preserve_suffix:
        prefix_end = start_end_seq_idx_ref[2]

    batch_start_seq_idx = start_seq_idx + pl.program_id(0) * batch_size
    batch_end_seq_idx = batch_start_seq_idx + batch_size - 1

    def attention_step1_qk_softmax(
        q,  # [num_q_heads, head_dim]
        kv,  # [bkv_sz, head_dim] <- Correspond to data from bkv_*_x2_ref
        kv_len,  # i32[] number of valid (non "-1") top-k rows in kv
    ):
        assert len(q.shape) == 2
        assert len(kv.shape) == 2
        assert q.shape[0] % num_q_heads == 0
        assert q.shape[1] == head_dim
        assert kv.shape == (bkv_sz, head_dim) or (
            (live_widths or row_widths)
            and kv.shape[1] == head_dim
            and kv.shape[0] in (*live_widths, *row_widths)
        )

        # Single-block softmax: each query attends exactly one gathered kv
        # block, so no FlashAttention running-max/rescale state (GLM-5.2 DSA
        # has no SWA or sinks). The step1/step2 split only interleaves VPU
        # and MXU work across batch entries.
        s = jnp.einsum("nd,md->nm", q, kv, preferred_element_type=jnp.float32)
        s *= sm_scale
        if k_scale != 1.0:
            s *= k_scale

        # Mask scores past kv_len (the "-1" tail): zeroing the keys alone is
        # not enough -- a zeroed key still scores 0 and inflates the softmax
        # denominator whenever kv_len << topk.
        kv_span = lax.broadcasted_iota(jnp.int32, s.shape, 1)
        s = jnp.where(kv_span < kv_len, s, jnp.finfo(jnp.float32).min)

        m = jnp.max(s, axis=1, keepdims=True)
        p = jnp.exp(s - m)
        l_sum = jnp.sum(p, axis=1, keepdims=True)

        # `m` is needed only to reconstruct the log-sum-exp for a cross-shard
        # merge; the single-device path drops it.
        return p, l_sum, m

    def attention_step2_pv(
        p,
        kv,
        l_sum,
        levers=(),
    ):
        if "pv512" in levers and split_q:
            kv = kv[:, :o_dim]
        if "pbf16" in levers:
            p = p.astype(q_dtype)
        pv = jnp.einsum("nm,md->nd", p, kv, preferred_element_type=jnp.float32)

        # We use `k_scale` for the PV step as well since MLA uses a shared
        # compressed latent for k/v and k_scale ~= v_scale.
        # TODO: Add a separate v_scale param and apply it to the
        # PV step.
        if k_scale != 1.0:
            pv *= k_scale

        out = (
            lax.div(pv, l_sum)
            if q_dtype == jnp.float32
            else (pv * pl.reciprocal(l_sum, approx=True)).astype(q_dtype)
        )
        return out

    def _async_copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_bkv_batch(seq_idx_start, bkv_sem_idx, *, wait=False, rows=None):
        sem_nope = sems.at[0, bkv_sem_idx]
        sem_rope = sems.at[3, bkv_sem_idx]

        bkv_nope_vmem_ref = bkv_nope_x2_ref.at[bkv_sem_idx, :]
        bkv_rope_vmem_ref = bkv_rope_x2_ref.at[bkv_sem_idx, :]

        # The index into cache_kv_hbm_ref should be relative to the current
        # chunk.
        page_idx_start = seq_idx_start - start_seq_idx
        if rows is not None:
            # `row_widths`: only the first `rows` rows of each token's tile.
            rope_rows = rows * cache_kv_rope_hbm_ref.shape[1] // bkv_sz
            nope_src = cache_kv_nope_hbm_ref.at[
                pl.ds(page_idx_start, batch_size), pl.ds(0, rows)
            ]
            rope_src = cache_kv_rope_hbm_ref.at[
                pl.ds(page_idx_start, batch_size), pl.ds(0, rope_rows)
            ]
            nope_dst = bkv_nope_vmem_ref.at[:, pl.ds(0, rows)]
            rope_dst = bkv_rope_vmem_ref.at[:, pl.ds(0, rope_rows)]
            if wait:
                _async_copy(nope_dst, nope_dst, sem_nope, True)
                _async_copy(rope_dst, rope_dst, sem_rope, True)
            else:
                _async_copy(nope_src, nope_dst, sem_nope, False)
                _async_copy(rope_src, rope_dst, sem_rope, False)
            return
        if group_tab_ref is not None:
            _fetch_live_blocks(
                seq_idx_start,
                bkv_nope_vmem_ref,
                bkv_rope_vmem_ref,
                sem_nope,
                sem_rope,
                wait,
            )
            return
        if not wait:
            _async_copy(
                cache_kv_nope_hbm_ref.at[pl.ds(page_idx_start, batch_size)],
                bkv_nope_vmem_ref,
                sem_nope,
                wait,
            )
            _async_copy(
                cache_kv_rope_hbm_ref.at[pl.ds(page_idx_start, batch_size)],
                bkv_rope_vmem_ref,
                sem_rope,
                wait,
            )
        else:
            dst_nope = bkv_nope_vmem_ref
            _async_copy(src=dst_nope, dst=dst_nope, sem=sem_nope, wait=True)
            dst_rope = bkv_rope_vmem_ref
            _async_copy(src=dst_rope, dst=dst_rope, sem=sem_rope, wait=True)

    def _fetch_live_blocks(seq_idx_start, nope_dst, rope_dst, sem_nope, sem_rope, wait):
        assert group_tab_ref is not None
        assert live_nope_hbm_ref is not None and live_rope_hbm_ref is not None
        num_groups = group_tab_ref.shape[0] // 2
        atom_rows = live_nope_hbm_ref.shape[2]
        rope_atom_rows = live_rope_hbm_ref.shape[2]
        g = seq_idx_start // batch_size
        live = group_tab_ref[g]
        off = group_tab_ref[num_groups + g]
        num_atoms = bkv_sz // atom_rows
        # A whole group (every atom live) sits token-major at block `off`:
        # the incumbent's one DMA per cache from its first token row.
        whole_row = off * (batch_size * atom_rows // bkv_sz)

        @pl.when(live == num_atoms)
        def _copy_whole():
            if wait:
                _async_copy(nope_dst, nope_dst, sem_nope, True)
                _async_copy(rope_dst, rope_dst, sem_rope, True)
            else:
                _async_copy(
                    cache_kv_nope_hbm_ref.at[pl.ds(whole_row, batch_size)],
                    nope_dst,
                    sem_nope,
                    False,
                )
                _async_copy(
                    cache_kv_rope_hbm_ref.at[pl.ds(whole_row, batch_size)],
                    rope_dst,
                    sem_rope,
                    False,
                )

        for a in range(num_atoms):

            @pl.when((a < live) & (live < num_atoms))
            def _copy_block():
                nope_slot = nope_dst.at[:, pl.ds(a * atom_rows, atom_rows)]
                rope_slot = rope_dst.at[:, pl.ds(a * rope_atom_rows, rope_atom_rows)]
                if wait:
                    _async_copy(nope_slot, nope_slot, sem_nope, True)
                    _async_copy(rope_slot, rope_slot, sem_rope, True)
                else:
                    _async_copy(
                        live_nope_hbm_ref.at[off + a], nope_slot, sem_nope, False
                    )
                    _async_copy(
                        live_rope_hbm_ref.at[off + a], rope_slot, sem_rope, False
                    )

    def _fetch_bq_batch(seq_idx_start, bq_sem_idx, *, wait=False):
        sem = sems.at[1, bq_sem_idx]
        bq_vmem_ref = bq_x2_ref.at[bq_sem_idx, :]

        if not wait:
            _async_copy(
                q_hbm_ref.at[pl.ds(seq_idx_start, batch_size)],
                bq_vmem_ref,
                sem,
                wait,
            )
            if split_q:
                # Same semaphore; the waits below mirror both starts.
                _async_copy(
                    q_pe_hbm_ref.at[pl.ds(seq_idx_start, batch_size)],
                    bq_pe_x2_ref.at[bq_sem_idx, :],
                    sem,
                    wait,
                )
            if q_rope:
                _async_copy(
                    tab_hbm_ref.at[pl.ds(seq_idx_start, batch_size)],
                    bq_tab_x2_ref.at[bq_sem_idx, :],
                    sem,
                    wait,
                )
        else:
            _async_copy(src=bq_vmem_ref, dst=bq_vmem_ref, sem=sem, wait=True)
            if split_q:
                bq_pe_vmem_ref = bq_pe_x2_ref.at[bq_sem_idx, :]
                _async_copy(src=bq_pe_vmem_ref, dst=bq_pe_vmem_ref, sem=sem, wait=True)
            if q_rope:
                bq_tab_vmem_ref = bq_tab_x2_ref.at[bq_sem_idx, :]
                _async_copy(
                    src=bq_tab_vmem_ref, dst=bq_tab_vmem_ref, sem=sem, wait=True
                )

    def _send_bo_batch(seq_idx_start, bo_sem_idx, *, wait=False):
        sem = sems.at[2, bo_sem_idx]
        vmem_ref = bo_x2_ref.at[bo_sem_idx, :]

        if preserve_suffix:
            # Split routing owns only the prefix, including in the last
            # rounded batch. Leave suffix queries untouched in HBM.
            rows = jnp.minimum(batch_size, prefix_end - seq_idx_start)
            _async_copy(
                vmem_ref.at[pl.ds(0, rows)],
                o_hbm_ref.at[pl.ds(seq_idx_start, rows)],
                sem,
                wait,
            )
            return

        if not wait:
            _async_copy(
                vmem_ref,
                o_hbm_ref.at[pl.ds(seq_idx_start, batch_size)],
                sem,
                wait,
            )
        else:
            _async_copy(src=vmem_ref, dst=vmem_ref, sem=sem, wait=True)

    def _send_blse_batch(seq_idx_start, bo_sem_idx, *, wait=False):
        sem = sems.at[4, bo_sem_idx]
        vmem_ref = blse_x2_ref.at[bo_sem_idx, :]

        if not wait:
            _async_copy(
                vmem_ref,
                lse_hbm_ref.at[pl.ds(seq_idx_start, batch_size)],
                sem,
                wait,
            )
        else:
            _async_copy(src=vmem_ref, dst=vmem_ref, sem=sem, wait=True)

    def fill_copy():
        return pltpu.make_async_copy(fill_ref, lse_hbm_ref, fill_sem)

    def start_fetch_bkv_batch(seq_idx_start, bkv_sem_idx, rows=None):
        return _fetch_bkv_batch(seq_idx_start, bkv_sem_idx, rows=rows)

    def wait_fetch_bkv_batch(seq_idx_start, bkv_sem_idx, rows=None):
        return _fetch_bkv_batch(seq_idx_start, bkv_sem_idx, wait=True, rows=rows)

    def with_row_width(seq_idx_start, fn):
        """`fn(rows)` for the batch at `seq_idx_start`: `rows` the smallest
        of `row_widths` covering every token's kv_len, None (all bkv_sz rows)
        when none does or there are no widths."""
        if not row_widths:
            fn(None)
            return
        need = kv_lens_ref[seq_idx_start]
        for b in range(1, batch_size):
            need = jnp.maximum(need, kv_lens_ref[seq_idx_start + b])
        widths = tuple(row_widths) + (None,)
        if fault == "row_width_undersize":
            # Positive control: every bucket above the first attends the
            # next-narrower width, dropping live rows.
            widths = (row_widths[0], *row_widths)
        lower = None
        for upper, width in zip((*row_widths, None), widths):
            if lower is None:
                fits = need <= upper
            elif upper is None:
                fits = need > lower
            else:
                fits = (need > lower) & (need <= upper)
            pl.when(fits)(functools.partial(fn, width))
            lower = upper

    def start_fetch_bq_batch(seq_idx_start, bq_sem_idx):
        return _fetch_bq_batch(seq_idx_start, bq_sem_idx)

    def wait_fetch_bq_batch(seq_idx_start, bq_sem_idx):
        return _fetch_bq_batch(seq_idx_start, bq_sem_idx, wait=True)

    def start_send_bo_batch(seq_idx_start, bo_sem_idx):
        return _send_bo_batch(seq_idx_start, bo_sem_idx)

    def wait_send_bo_batch(seq_idx_start, bo_sem_idx):
        return _send_bo_batch(seq_idx_start, bo_sem_idx, wait=True)

    def load_bq(bq_sem_idx, batch_idx):
        q = bq_x2_ref.at[bq_sem_idx, batch_idx][...]
        if q_rope:
            # The caller's rope + concat + zero pad, rebuilt in VMEM: lanes
            # 128:256 of the raw head are one aligned vreg column.
            raw = bq_pe_x2_ref.at[bq_sem_idx, batch_idx][:, 128:Q_ROPE_HEAD_DIM]
            tab = bq_tab_x2_ref.at[bq_sem_idx, batch_idx][...]
            q = jnp.concatenate([q, rope_q_block(raw, tab, fault)], -1)
        elif split_q:
            # The caller's concat + zero pad, rebuilt in VMEM.
            q = jnp.concatenate([q, bq_pe_x2_ref.at[bq_sem_idx, batch_idx][...]], -1)
            if q.shape[-1] < head_dim:
                q = jnp.pad(q, ((0, 0), (0, head_dim - q.shape[-1])))
        return q

    def store_blse(bo_sem_idx, batch_idx, lse_col):
        """Stores one token's f32[num_q_heads, 1] log-sum-exp column."""
        if not lse_lane_dense:
            blse_x2_ref.at[bo_sem_idx, batch_idx][...] = lse_col
            return
        # Heads from sublanes to lanes, exactly: each column of the masked
        # [heads, heads] tile holds one live entry, the rest -inf.
        eye_shape = (num_q_heads, num_q_heads)
        eye = lax.broadcasted_iota(jnp.int32, eye_shape, 0) == (
            lax.broadcasted_iota(jnp.int32, eye_shape, 1)
        )
        row = jnp.max(
            jnp.where(eye, jnp.broadcast_to(lse_col, eye_shape), -jnp.inf),
            axis=0,
            keepdims=True,
        )
        tile_ref = blse_x2_ref.at[bo_sem_idx]  # [bsz, num_q_heads]
        if isinstance(batch_idx, int):
            tile_ref[pl.ds(batch_idx, 1), :] = row
        else:
            pick = lax.broadcasted_iota(jnp.int32, tile_ref.shape, 0) == batch_idx
            tile_ref[...] = jnp.where(pick, row, tile_ref[...])

    def load_bkv_levers(bkv_sem_idx, batch_idx, rows, levers):
        """`load_bkv` for the live body with `body_levers` (see above)."""
        rows = bkv_sz if rows is None else rows
        nope_ref = bkv_nope_x2_ref.at[bkv_sem_idx, batch_idx]
        rope_ref = bkv_rope_x2_ref.at[bkv_sem_idx, batch_idx]
        rope_ref = rope_ref.at[pl.ds(0, rows * rope_ref.shape[0] // bkv_sz)]
        kv_len = kv_lens_ref[batch_start_seq_idx + batch_idx]
        zu = "zu" in levers
        if "tr" in levers:
            # Row r of the NOPE tile holds key r's 512 bytes as 128 words;
            # the fp8 tile wants each word to hold one byte column of four
            # consecutive keys. xs[j] = keys 4s + j (stride-4 rows).
            quarter = rows // 4
            xs = [nope_ref[pl.ds(j, quarter, stride=4), :] for j in range(4)]
            if zu:
                key4 = lax.broadcasted_iota(jnp.int32, xs[0].shape, 0) * 4
                xs = [
                    jnp.where(key4 < kv_len - j, x, jnp.uint32(0))
                    for j, x in enumerate(xs)
                ]
            lo16, hi16 = jnp.uint32(0x0000FFFF), jnp.uint32(0xFFFF0000)
            ev8, od8 = jnp.uint32(0x00FF00FF), jnp.uint32(0xFF00FF00)
            a0 = (xs[0] & lo16) | (xs[2] << 16)
            a2 = (xs[0] >> 16) | (xs[2] & hi16)
            a1 = (xs[1] & lo16) | (xs[3] << 16)
            a3 = (xs[1] >> 16) | (xs[3] & hi16)
            cols = [
                (a0 & ev8) | ((a1 & ev8) << 8),  # byte 0 of keys 4s..4s+3
                ((a0 >> 8) & ev8) | (a1 & od8),  # byte 1
                (a2 & ev8) | ((a3 & ev8) << 8),  # byte 2
                ((a2 >> 8) & ev8) | (a3 & od8),  # byte 3
            ]
            bkv_nope = jnp.concatenate(
                [pltpu.bitcast(c, jnp.float8_e4m3fn) for c in cols], axis=-1
            )
        else:
            words = nope_ref[pl.ds(0, rows)]
            if zu:
                key = lax.broadcasted_iota(jnp.int32, words.shape, 0)
                words = jnp.where(key < kv_len, words, jnp.uint32(0))
            bkv_nope = pltpu.bitcast(words, jnp.uint8).reshape(rows, nope_dim)
            bkv_nope = pltpu.bitcast(bkv_nope, jnp.float8_e4m3fn)
        bkv_rope = pltpu.bitcast(rope_ref[...], jnp.uint8).reshape(rows, 128)
        bkv_rope = pltpu.bitcast(bkv_rope, jnp.float8_e4m3fn)
        bkv = jnp.concatenate([bkv_nope, bkv_rope], axis=-1)
        if bkv.shape[-1] < head_dim:
            bkv = jnp.pad(bkv, ((0, 0), (0, head_dim - bkv.shape[-1])))
        if not zu:
            k_span = lax.broadcasted_iota(jnp.int32, bkv.shape, 0)
            bkv = jnp.where(k_span < kv_len, bkv, 0)
        return bkv

    def load_bkv(bkv_sem_idx, batch_idx, rows=None, levers=()):
        if levers:
            return load_bkv_levers(bkv_sem_idx, batch_idx, rows, levers)
        nope_ref = bkv_nope_x2_ref.at[bkv_sem_idx, batch_idx]
        rope_ref = bkv_rope_x2_ref.at[bkv_sem_idx, batch_idx]
        if rows is not None:
            nope_ref = nope_ref.at[pl.ds(0, rows)]
            rope_ref = rope_ref.at[pl.ds(0, rows * rope_ref.shape[0] // bkv_sz)]
        return unpack_bkv(
            nope_ref,
            rope_ref,
            lambda: kv_lens_ref[batch_start_seq_idx + batch_idx],
        )

    def unpack_bkv(bkv_nope_ref, bkv_rope_ref, get_kv_len):
        bkv_nope = bkv_nope_ref[...]
        if sparsecore_layout:
            # Native SC returns one uint32[128] row per token. Bitcasting
            # expands the packed byte lanes before folding them into the
            # logical 512-byte NOPE vector.
            bkv_nope = pltpu.bitcast(bkv_nope, jnp.uint8)
        rows = bkv_nope_ref.shape[0] if sparsecore_layout else bkv_sz
        bkv_nope = bkv_nope.reshape(rows, nope_dim)
        bkv_nope = pltpu.bitcast(bkv_nope, jnp.float8_e4m3fn)

        bkv_rope = bkv_rope_ref[...]
        if sparsecore_layout:
            # The native gather's delta-swap produces four-token grouped
            # uint32[128] rows. Expanding their byte lanes recovers one
            # contiguous 128-byte row per selected token.
            bkv_rope = pltpu.bitcast(bkv_rope, jnp.uint8)
            bkv_rope = bkv_rope.reshape(rows, 128)
        bkv_rope = pltpu.bitcast(bkv_rope, jnp.float8_e4m3fn)
        bkv = jnp.concatenate([bkv_nope, bkv_rope], axis=-1)

        # Pad kv (576) to q's 128-aligned head_dim (640); the padded dims are
        # no-ops in the einsums since q's padded dims are also zero.
        if bkv.shape[-1] < head_dim:
            bkv = jnp.pad(bkv, ((0, 0), (0, head_dim - bkv.shape[-1])))

        # Zero keys beyond kv_len: the "-1" tail is gathered from scattered
        # pages and can decode to fp8 NaN; score masking zeroes the weights,
        # but 0 * NaN = NaN in the PV einsum, so the data must be finite too.
        kv_len = get_kv_len()
        k_span = lax.broadcasted_iota(jnp.int32, bkv.shape, 0)
        bkv = jnp.where(k_span < kv_len, bkv, 0)
        return bkv

    def fixup_capped_overflow(bi_sem_idx, bo_sem_idx):
        """Recompute tokens with more than `bkv_sz` live rows (capped mode).

        The pipelined body attended only window 0 of such a token. This pass
        redoes it over windows `0 .. cdiv(kv_len, bkv_sz) - 1` with an online
        softmax: window 0 from the VMEM slot the body just read (the next
        batch prefetches into the other slot), window j >= 1 from gathered row
        `j * chunk + (seq - start)` into a two-slot scratch, the DMA of window
        j + 1 started before window j is computed. Overwrites the token's `bo`
        and `blse` entries before they are sent. Tokens within `bkv_sz` skip it.
        """
        assert fix_refs is not None and capped_windows is not None
        fix_nope_ref, fix_rope_ref, fix_m_ref, fix_l_ref, fix_acc_ref, fix_sems = (
            fix_refs
        )
        f32_min = jnp.finfo(jnp.float32).min
        chunk_rows = cache_kv_nope_hbm_ref.shape[0] // capped_windows

        def window_copies(local_idx, j, slot):
            row = j * chunk_rows + local_idx
            return (
                pltpu.make_async_copy(
                    cache_kv_nope_hbm_ref.at[row],
                    fix_nope_ref.at[slot],
                    fix_sems.at[0, slot],
                ),
                pltpu.make_async_copy(
                    cache_kv_rope_hbm_ref.at[row],
                    fix_rope_ref.at[slot],
                    fix_sems.at[1, slot],
                ),
            )

        def accumulate(q, kv, rem):
            s = jnp.einsum("nd,md->nm", q, kv, preferred_element_type=jnp.float32)
            s *= sm_scale
            if k_scale != 1.0:
                s *= k_scale
            kv_span = lax.broadcasted_iota(jnp.int32, s.shape, 1)
            s = jnp.where(kv_span < rem, s, f32_min)
            m_prev = fix_m_ref[...]
            m_new = jnp.maximum(m_prev, jnp.max(s, axis=1, keepdims=True))
            p = jnp.exp(s - m_new)
            alpha = jnp.exp(m_prev - m_new)
            fix_l_ref[...] = alpha * fix_l_ref[...] + jnp.sum(p, axis=1, keepdims=True)
            fix_acc_ref[...] = alpha * fix_acc_ref[...] + jnp.einsum(
                "nm,md->nd", p, kv, preferred_element_type=jnp.float32
            )
            fix_m_ref[...] = m_new

        def fix_token(batch_idx, carry):
            seq_idx = batch_start_seq_idx + batch_idx
            kv_len = kv_lens_ref[seq_idx]

            @pl.when(kv_len > bkv_sz)
            def _fix():
                local_idx = seq_idx - start_seq_idx
                num_windows = cdiv(kv_len, bkv_sz)
                for cp in window_copies(local_idx, 1, 1):
                    cp.start()
                q = load_bq(bi_sem_idx, batch_idx)
                fix_m_ref[...] = jnp.full(fix_m_ref.shape, f32_min, jnp.float32)
                fix_l_ref[...] = jnp.zeros(fix_l_ref.shape, jnp.float32)
                fix_acc_ref[...] = jnp.zeros(fix_acc_ref.shape, jnp.float32)
                accumulate(
                    q,
                    unpack_bkv(
                        bkv_nope_x2_ref.at[bi_sem_idx, batch_idx],
                        bkv_rope_x2_ref.at[bi_sem_idx, batch_idx],
                        lambda: kv_len,
                    ),
                    kv_len,
                )

                def window(j, window_carry):
                    slot = lax.rem(j, 2)

                    @pl.when(j + 1 < num_windows)
                    def _prefetch_next():
                        for cp in window_copies(local_idx, j + 1, 1 - slot):
                            cp.start()

                    for cp in window_copies(local_idx, j, slot):
                        cp.wait()
                    rem = kv_len - j * bkv_sz
                    kv = unpack_bkv(
                        fix_nope_ref.at[slot], fix_rope_ref.at[slot], lambda: rem
                    )
                    accumulate(q, kv, rem)
                    return window_carry

                lax.fori_loop(1, num_windows, window, 0)
                l_sum = fix_l_ref[...]
                pv = fix_acc_ref[...]
                if k_scale != 1.0:
                    pv *= k_scale
                out = (
                    lax.div(pv, l_sum)
                    if q_dtype == jnp.float32
                    else (pv * pl.reciprocal(l_sum, approx=True)).astype(q_dtype)
                )
                bo_x2_ref.at[bo_sem_idx, batch_idx][...] = out[:, :o_dim]
                if return_lse:
                    store_blse(bo_sem_idx, batch_idx, fix_m_ref[...] + jnp.log(l_sum))

            return carry

        lax.fori_loop(0, batch_size, fix_token, 0)

    def process():
        def get_next_seq_ids(seq_idx, bi_sem_idx):
            next_seq_idx = seq_idx + batch_size
            next_bi_sem_idx = lax.select(bi_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bi_sem_idx

        bi_sem_idx = sem_ids_ref[0]
        next_seq_idx, next_bi_sem_idx = get_next_seq_ids(
            batch_start_seq_idx, bi_sem_idx
        )

        # Prefetch next seq
        @pl.when(next_seq_idx < end_seq_idx)
        def prefetch_next_seq():
            sem_ids_ref[0] = next_bi_sem_idx
            start_fetch_bq_batch(next_seq_idx, next_bi_sem_idx)
            with_row_width(
                next_seq_idx,
                functools.partial(start_fetch_bkv_batch, next_seq_idx, next_bi_sem_idx),
            )

        bo_sem_idx = sem_ids_ref[1]
        sem_ids_ref[1] = lax.select(bo_sem_idx == 0, 1, 0)

        def write_blse(batch_idx, m, l_sum):
            # log-sum-exp in the *unshifted* frame, so a peer holding a
            # disjoint slice of the same token's top-k can merge with
            # o = sum_r softmax(lse_r) * o_r without knowing our max.
            if not return_lse:
                return
            lse_col = m + jnp.log(l_sum)
            if lse_mask_empty:
                kv_len = kv_lens_ref[batch_start_seq_idx + batch_idx]
                lse_col = lse_col + jnp.where(
                    kv_len == 0, jnp.float32(-jnp.inf), jnp.float32(0.0)
                )
            store_blse(bo_sem_idx, batch_idx, lse_col)

        # Wait for cur blocks if not ready yet
        wait_fetch_bq_batch(batch_start_seq_idx, bi_sem_idx)
        if not row_widths:
            wait_fetch_bkv_batch(batch_start_seq_idx, bi_sem_idx)

        @pl.when(pl.program_id(0) >= 2)
        def _wait_send():
            # This slot holds the transfer from two batches earlier.
            sent_start = (
                batch_start_seq_idx - 2 * batch_size
                if preserve_suffix
                else batch_start_seq_idx
            )
            wait_send_bo_batch(sent_start, bo_sem_idx)
            if return_lse:
                _send_blse_batch(batch_start_seq_idx, bo_sem_idx, wait=True)

        def run_steps(rows=None, levers=()):
            prev_p = None
            prev_bkv = None
            prev_l_sum = None
            prev_m = None
            prev_out = None
            for batch_idx in range(batch_size):
                kv_len = kv_lens_ref[batch_start_seq_idx + batch_idx]
                bkv = load_bkv(bi_sem_idx, batch_idx, rows, levers)
                bq = load_bq(bi_sem_idx, batch_idx)

                if prev_out is not None and "noinf" not in levers:
                    # Artificial dependency to force MXU/VPU interleaving by
                    # limiting LLO's QK runahead
                    # We use jnp.where to prevent XLA from optimizing the
                    # dependency away.
                    # prev_out won't be inf in practice.
                    dep = prev_out
                    if dep.shape != bq.shape:  # "pv512": a NOPE-wide output
                        dep = dep[:, :1]
                    bq = jnp.where(dep == jnp.inf, dep, bq)

                p, l_sum, m = attention_step1_qk_softmax(bq, bkv, kv_len)

                if prev_p is not None:
                    assert prev_bkv is not None
                    assert prev_l_sum is not None
                    out = attention_step2_pv(prev_p, prev_bkv, prev_l_sum, levers)

                    # Store output from acc to bo.
                    bo_x2_ref.at[bo_sem_idx, batch_idx - 1][...] = out[:, :o_dim]
                    write_blse(batch_idx - 1, prev_m, prev_l_sum)
                    prev_out = out

                prev_p = p
                prev_bkv = bkv
                prev_l_sum = l_sum
                prev_m = m

            # end of pipelining loop
            out = attention_step2_pv(prev_p, prev_bkv, prev_l_sum, levers)
            bo_x2_ref.at[bo_sem_idx, batch_size - 1][...] = out[:, :o_dim]
            write_blse(batch_size - 1, prev_m, prev_l_sum)

        if live_widths:
            assert live_nope_hbm_ref is not None
            atom_rows = live_nope_hbm_ref.shape[2]
            need = group_tab_ref[batch_start_seq_idx // batch_size] * atom_rows
            lower = None
            for width in tuple(live_widths) + (bkv_sz,):
                fits = (
                    need <= width if lower is None else (need > lower) & (need <= width)
                )
                pl.when(fits)(
                    functools.partial(
                        run_steps,
                        width if width < bkv_sz else None,
                        tuple(body_levers),
                    )
                )
                lower = width
        elif row_widths:

            def wait_and_run(rows):
                wait_fetch_bkv_batch(batch_start_seq_idx, bi_sem_idx, rows)
                run_steps(rows, tuple(body_levers))

            with_row_width(batch_start_seq_idx, wait_and_run)
        else:
            run_steps(None, tuple(body_levers))
        if capped_windows is not None:
            fixup_capped_overflow(bi_sem_idx, bo_sem_idx)
        start_send_bo_batch(batch_start_seq_idx, bo_sem_idx)
        if lse_fill:
            # The sentinel fill must land before this step's rows overwrite it.
            @pl.when(pl.program_id(0) == 0)
            def _():
                fill_copy().wait()

        if return_lse:
            _send_blse_batch(batch_start_seq_idx, bo_sem_idx)

    ### ------- Kernel start ------- ###

    # With `lse_fill` the grid has at least one step even when no row is
    # live; such a step only lays down the sentinel.
    has_work = end_seq_idx > start_seq_idx if lse_fill else True

    if lse_fill:

        @pl.when(pl.program_id(0) == 0)
        def _fill_lse():
            sentinel = -jnp.inf if lse_mask_empty else jnp.finfo(jnp.float32).min
            if fault == "skip_lse_fill":
                sentinel = 0.0  # positive control: a wrong sentinel
            assert fill_ref is not None
            fill_ref[...] = jnp.full(fill_ref.shape, sentinel, jnp.float32)
            fill_copy().start()

            @pl.when(jnp.logical_not(has_work))
            def _():
                fill_copy().wait()

    @pl.when(jnp.logical_and(batch_start_seq_idx == start_seq_idx, has_work))
    def prologue():
        start_fetch_bq_batch(batch_start_seq_idx, 0)
        with_row_width(
            batch_start_seq_idx,
            functools.partial(start_fetch_bkv_batch, batch_start_seq_idx, 0),
        )

    if lse_fill:
        pl.when(has_work)(process)
    else:
        process()

    @pl.when(jnp.logical_and(batch_end_seq_idx == end_seq_idx - 1, has_work))
    def epilogue():
        # Partial stores must wait for the same byte count as their send.
        last_even = (pl.num_programs(0) - 1) // 2 * 2
        sent_start = start_seq_idx + last_even * batch_size if preserve_suffix else 0
        wait_send_bo_batch(sent_start, 0)
        if return_lse:
            _send_blse_batch(0, 0, wait=True)

        @pl.when(pl.num_programs(0) >= 2)
        def _wait_1():
            last_odd = (pl.num_programs(0) - 2) // 2 * 2 + 1
            sent_start = start_seq_idx + last_odd * batch_size if preserve_suffix else 0
            wait_send_bo_batch(sent_start, 1)
            if return_lse:
                _send_blse_batch(0, 1, wait=True)

    ### ------- Kernel end ------- ###


def _chunk_kernel_batch_size(
    chunk_size: int,
    attention_kernel_batch_size: int,
    small_chunk_tokens: int,
    small_chunk_batch_size: int | None,
) -> tuple[bool, int]:
    """(is a small chunk, attention grid step) -- as `sparse_attention_chunk`."""
    small_chunk = small_chunk_tokens > 0 and chunk_size <= small_chunk_tokens
    batch_cap = attention_kernel_batch_size
    if small_chunk and small_chunk_batch_size is not None:
        batch_cap = small_chunk_batch_size
    return small_chunk, _largest_divisor(chunk_size, batch_cap)


def prepare_q_inputs(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim],
):
    _, actual_num_q_heads, actual_head_dim = q.shape
    q_packing = get_dtype_packing(q.dtype)
    num_q_heads = align_to(actual_num_q_heads, q_packing)
    head_dim = align_to(actual_head_dim, 128)
    q = jnp.pad(
        q,
        (
            (0, 0),
            (0, num_q_heads - actual_num_q_heads),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    )
    return q


def prepare_outputs(
    out,  # [max_num_tokens, num_q_heads, head_dim]
    actual_num_q_heads: int,
    actual_head_dim: int,
):
    return out[:, :actual_num_q_heads, :actual_head_dim]


# Main attention kernel for GLM-5.2 DSA (top-k gather and attention).
# The KV entries referenced by `topk_indices` (including the current batch's
# own tokens) must already be resident in the caches before this call;
# per-token kv_lens derive from the "-1" padding in `topk_indices`.
@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "k_scale",
        "attention_kernel_batch_size",
        "gather_and_attention_chunk_size",
        "vmem_limit_bytes",
        "return_lse",
        "cache_layout",
        "sparsecore_atoms_per_batch",
        "sparsecore_gather_wait_mode",
        "sparsecore_gather_pipelined",
        "small_chunk_tokens",
        "small_chunk_batch_size",
        "small_chunk_live_gather",
        "small_chunk_gather_atoms",
        "live_gather_all_chunks",
        "topk_cap",
        "lse_lane_dense",
        "lse_mask_empty",
        "fused_window_prep",
        "mixed_live_gather",
        "mixed_live_widths",
        "mixed_body_levers",
        "lse_in_kernel",
        "prologue_fault",
        "kv_write_fault",
        "mixed_fused_prep",
        "decode_row_widths",
    ),
)
def sparse_ragged_paged_attention(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, head_dim]
    # TensorCore: u8[pages, page_size, 4, 128].
    # SparseCore: u32[pages, page_size, 128].
    cache_kv_nope: jax.Array,
    # TensorCore: u8[pages, page_size/4, 4, 128].
    # SparseCore: u32[pages, page_size/4, 128].
    cache_kv_rope: jax.Array,
    topk_indices: jax.Array,  # i32[max_num_tokens, csa_topk]
    page_indices: jax.Array | None,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    sm_scale: float = 1.0,
    k_scale: float = 1.0,
    # Kernel optimization params.
    gather_and_attention_chunk_size: int = (DEFAULT_GATHER_AND_ATTENTION_CHUNK_SIZE),
    attention_kernel_batch_size: int = DEFAULT_ATTENTION_KERNEL_BATCH_SIZE,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    return_lse: bool = False,
    cache_layout: str = TENSORCORE_CACHE_LAYOUT,
    sparsecore_atoms_per_batch: int = 16,
    sparsecore_gather_wait_mode: str = "overlap_rope",
    sparsecore_gather_pipelined: bool = True,
    small_chunk_tokens: int = 16,
    small_chunk_batch_size: int | None = 1,
    small_chunk_live_gather: bool = True,
    small_chunk_gather_atoms: int = 1,
    live_gather_all_chunks: bool = True,
    topk_cap: int | None = None,
    q_pe: jax.Array | None = None,
    lse_lane_dense: bool = False,
    lse_mask_empty: bool = False,
    fused_window_prep: bool = False,
    mixed_live_gather: bool = False,
    mixed_live_widths: tuple[int, ...] = (),
    mixed_body_levers: tuple[str, ...] = (),
    q_rope_table: jax.Array | None = None,
    lse_in_kernel: bool = False,
    prologue_fault: str | None = None,
    kv_write: jax.Array | tuple[jax.Array, jax.Array, jax.Array] | None = None,
    kv_write_fault: str | None = None,
    mixed_fused_prep: bool = False,
    decode_row_widths: tuple[int, ...] = (),
) -> jax.Array | tuple[jax.Array, ...]:
    """MLA Ragged paged attention that supports mixed prefill and decode.

    `decode_row_widths` (TPU_MLA_DECODE_ROW_WIDTHS; capped path, chunks that
    do not take the live-group prep): ascending multiples of 128 below
    `topk_cap`; each grid step fetches and attends only the first w rows of
    its tokens' window-0 tiles, w the smallest width covering the batch's
    largest kv_len (all rows when none does). Same results bit for bit.

    Args:
      q: concatenated all sequences' queries.
      cache_kv_nope: the current kv cache for nope.
      cache_kv_rope: the current kv cache for rope.
      topk_indices: for each query token, the indices of the top k key tokens to
        attend to.
      page_indices: flattened page indices look-up table by (seq_id, page_id).
      cu_q_lens: the cumulative sum of the effective query lengths. Similar to
        kv_lens, only the first num_seqs+1 values are valid.
      distribution: (i, j, k) represents that sequences[0:i] are decode-only,
        sequences[i:j] are chunked-prefill-only, and sequences[j:k] are mixed. The
        k is also the total number of sequences.
      sm_scale: the softmax scale which will be applied to the Q@K^T.
      k_scale: per-tensor dequantization scale for the fp8 nope cache.
      vmem_limit_bytes: the vmem limit for the pallas kernel.
      return_lse: also return the per-(token, head) log-sum-exp of the attention
        scores. Needed only when `topk_indices` holds a slice of a token's
        global top-k and the partial outputs of several ranks must be merged.
      cache_layout: Physical cache layout: ``"tensorcore"`` for tiled uint8
        caches or ``"sparsecore"`` for native uint32 caches. Both modes perform
        the same sparse attention computation.
      sparsecore_atoms_per_batch: Native SparseCore gather pipeline atoms per
        owner batch.
      sparsecore_gather_wait_mode: Whether native SparseCore NOPE output DMA
        overlaps ROPE work.
      sparsecore_gather_pipelined: Whether to pipeline native SparseCore gather
        serving.
      small_chunk_tokens: Chunks of at most this many tokens (the decode
        bucket) take the small-chunk path below; 0 disables it. Full
        `gather_and_attention_chunk_size` prefill chunks are unaffected.
      small_chunk_batch_size: Attention grid step for small chunks, so the
        traced grid `cdiv(real tokens, batch)` walks real tokens only. None
        keeps `attention_kernel_batch_size`.
      small_chunk_live_gather: For small chunks on the SparseCore layout, bound
        the native SparseCore gather to the rows the attention grid reads,
        `[start_pos, batch_end) x topk`; the remaining rows stay unwritten.
      small_chunk_gather_atoms: `atoms_per_batch` for that bounded gather.
      live_gather_all_chunks: Also bound the native SparseCore gather of
        larger chunks to `[start_pos, batch_end) x topk`, keeping
        `sparsecore_atoms_per_batch` (live granularity: one owner round).
        Fully dead chunks issue no gather; an all-live chunk traces the same
        grid length as the unbounded gather.
      topk_cap: SparseCore layout only (ignored otherwise; None disables).
        Gather and attend `topk_cap` top-k slots per token instead of `topk`,
        exactly: rows are prefix-packed, so a token with at most `topk_cap`
        live entries holds them all in its first window, and a token with more
        is recomputed in-kernel over all its `topk_cap`-slot windows (online
        softmax). The native gather is bounded to the chunk's live windows, so
        a chunk with no such token gathers `topk_cap` slots per row only. Must
        be a multiple of 64 dividing `topk`. Pays off when rows hold far fewer
        than `topk` live entries (DCP).
      q_pe: Split-q mode. `q` is then q_nope only, `[max_num_tokens,
        num_q_heads, nope]` with `nope` a multiple of 128 and `num_q_heads` a
        multiple of the dtype packing, and `q_pe` `[max_num_tokens,
        num_q_heads, rope]` its rope part. The kernel joins them in VMEM
        instead of the caller materialising a padded concat; the output
        aliases q_nope and is `nope` wide (the value part only). Same scores
        and output values.
      lse_lane_dense: With `return_lse`, carry the log-sum-exp as f32[tokens,
        heads] through the kernel instead of f32[tokens, heads, 1] (lane-padded
        tiles). Same values.
      lse_mask_empty: With `return_lse`, a token with no live top-k entry
        (all -1, kv_len 0) gets lse = -inf and output 0 from the kernel itself,
        and rows the grid never reaches keep -inf, so a DCP caller needs no
        dummy key or owns mask (`topk_indices` may hold all-empty rows).
      fused_window_prep: With `topk_cap` and a single chunk of at most
        `FUSED_PREP_MAX_TOKENS` tokens (decode buckets), build the window
        indices and the gather's live bound in one Pallas call
        (`prepare_capped_window_fused`). Same gathered rows. Requires
        `lse_mask_empty` (kv_len 0 rows are handed to the kernel as-is).
      mixed_live_gather: With `topk_cap`, buckets of more than
        `FUSED_PREP_MAX_TOKENS` tokens (mixed / prefill), where
        `live_group_supported` holds: lay window 0 out by 64-row atom block
        per attention grid step (`prepare_capped_window_live`), gather only
        each grid step's live blocks (cdiv(max kv_len, 64) of them) and fetch
        only those into the kernel. Same outputs; decode buckets unaffected.
      mixed_live_widths: With `mixed_live_gather` taken: ascending row widths
        (multiples of 128 below `topk_cap`); a grid step whose live atoms fit
        a width runs its attention body on that many rows of each token's
        tile instead of `topk_cap`. Same outputs.
      mixed_body_levers: With `mixed_live_gather` taken: attention-body
        rewrites of the live grid steps (`MIXED_BODY_LEVERS`, documented at
        `_attention_kernel`). `MIXED_BODY_DEFAULT` is bitwise-identical.
      q_rope_table: In-kernel q rope (TPU_MLA_DECODE_PROLOGUE; split q only).
        `q_pe` is then the raw, un-roped q head view `[tokens, heads, 256]`
        (nope 192 | rope 64, NeoX half-split rope order) and this is f32
        `[tokens, 2, 128]`: C = [cos, cos, cos, cos], S = [-sin, sin, -sin,
        sin] of each token's position. The kernel ropes q_pe in VMEM; same
        scores and outputs as the caller roping q_pe in bf16.
      lse_in_kernel: With `return_lse`, `lse_lane_dense` and a single chunk:
        the kernel owns the lse output and lays the sentinel down itself (no
        XLA-initialised aliased input). Same values on every row.
      prologue_fault: test-only fault injection (`PROLOGUE_FAULTS`).
      kv_write: TPU_MLA_KV_WRITE_IN_GATHER, fused-prep decode buckets only:
        `(dst_rows i32[T], nope_rows u32[T, 128], rope_rows u32[T, 32])`, this
        step's tokens packed for the SparseCore caches, or the same in
        `native_sc_gather.pack_write_rows`' one-operand u32[3 * Tp, 128] form.
        The SparseCore gather
        writes them into both caches in place before it gathers (see
        `native_sc_gather.dsa_gather_native_sc`), and the updated caches are
        returned after the other outputs.
      kv_write_fault: test-only (`native_sc_gather.WRITE_FAULTS`).
      mixed_fused_prep: With `mixed_live_gather` taken and
        `mixed_fused_prep_supported`: build kv_lens, the group table and the
        live-group window rows in one call that reads the top-k as 128-lane
        rows and writes one window array per chunk
        (`prepare_capped_window_live_fused`). Same rows, kv_lens and table.

    Returns:
      The output of attention, or `(output, lse)` when `return_lse` is set.
      `lse` is f32[max_num_tokens, actual_num_q_heads]. With `kv_write`, the
      updated `(cache_kv_nope, cache_kv_rope)` follow.
    """
    if cache_layout not in (TENSORCORE_CACHE_LAYOUT, SPARSECORE_CACHE_LAYOUT):
        raise ValueError(
            f"cache_layout must be {TENSORCORE_CACHE_LAYOUT!r} or "
            f"{SPARSECORE_CACHE_LAYOUT!r}, got {cache_layout!r}"
        )
    sparsecore_layout = cache_layout == SPARSECORE_CACHE_LAYOUT
    if sparsecore_layout:
        if (
            cache_kv_nope.dtype != jnp.uint32
            or cache_kv_nope.ndim != 3
            or cache_kv_nope.shape[-1] != 128
        ):
            raise ValueError(
                "sparsecore NOPE cache must be uint32[pages,page_size,128]"
            )
        if (
            cache_kv_rope.dtype != jnp.uint32
            or cache_kv_rope.ndim != 3
            or cache_kv_rope.shape[-1] != 128
        ):
            raise ValueError(
                "sparsecore ROPE cache must be uint32[pages,page_size/4,128]"
            )
    else:
        if (
            cache_kv_nope.dtype != jnp.uint8
            or cache_kv_nope.ndim != 4
            or cache_kv_nope.shape[-2:] != (4, 128)
        ):
            raise ValueError(
                "tensorcore NOPE cache must be uint8[pages,page_size,4,128]"
            )
        if (
            cache_kv_rope.dtype != jnp.uint8
            or cache_kv_rope.ndim != 4
            or cache_kv_rope.shape[-2:] != (4, 128)
        ):
            raise ValueError(
                "tensorcore ROPE cache must be uint8[pages,page_size/4,4,128]"
            )
    if cache_kv_nope.size != 4 * cache_kv_rope.size:
        raise ValueError(
            f"NOPE cache size ({cache_kv_nope.size}) must be exactly "
            f"4 * ROPE cache size ({cache_kv_rope.size})"
        )
    if (
        cache_kv_nope.shape[0] != cache_kv_rope.shape[0]
        or cache_kv_nope.shape[1] != 4 * cache_kv_rope.shape[1]
    ):
        raise ValueError(
            "NOPE and ROPE caches must have matching page counts and page geometry"
        )

    _, actual_num_q_heads, actual_head_dim = q.shape

    if prologue_fault not in PROLOGUE_FAULTS:
        raise ValueError(f"unknown prologue_fault {prologue_fault}")
    if q_rope_table is not None:
        if q_pe is None or q_pe.shape[-1] != Q_ROPE_HEAD_DIM:
            raise ValueError(
                "q_rope_table needs the raw q head view as q_pe, "
                f"[tokens, heads, {Q_ROPE_HEAD_DIM}]"
            )
        if q_rope_table.shape != (q.shape[0], 2, 128) or (
            q_rope_table.dtype != jnp.float32
        ):
            raise ValueError(
                f"q_rope_table {q_rope_table.shape}/{q_rope_table.dtype} must be "
                f"f32[{q.shape[0]}, 2, 128]"
            )
    if q_pe is not None:
        if q_pe.shape[:2] != q.shape[:2] or q_pe.dtype != q.dtype:
            raise ValueError(
                f"q_pe {q_pe.shape}/{q_pe.dtype} must match q_nope "
                f"{q.shape}/{q.dtype} in tokens, heads and dtype"
            )
        if actual_head_dim % 128 or actual_num_q_heads % get_dtype_packing(q.dtype):
            raise ValueError(
                f"split q needs a 128-multiple nope dim and packed heads, got {q.shape}"
            )
    else:
        q = prepare_q_inputs(q)  # [max_num_tokens, num_q_heads, head_dim]

    page_size = cache_kv_nope.shape[1]

    _, num_q_heads, _ = q.shape

    if topk_indices.dtype != jnp.int32 or topk_indices.ndim != 2:
        raise ValueError("topk_indices must be int32[num_tokens, topk]")
    if q.shape[0] != topk_indices.shape[0]:
        raise ValueError(
            f"q tokens ({q.shape[0]}) must match topk_indices tokens "
            f"({topk_indices.shape[0]})"
        )
    topk = topk_indices.shape[-1]
    if sparsecore_layout and topk % native_sc_gather.ROPE_PACKING != 0:
        raise ValueError(
            f"topk ({topk}) must be a multiple of "
            f"{native_sc_gather.ROPE_PACKING} for sparsecore ROPE packing"
        )
    if not sparsecore_layout or (topk_cap is not None and topk_cap >= topk):
        topk_cap = None
    if topk_cap is not None and (topk_cap % GATHER_ATOM_ROWS or topk % topk_cap):
        raise ValueError(
            f"topk_cap ({topk_cap}) must be a multiple of {GATHER_ATOM_ROWS} "
            f"dividing topk ({topk})"
        )

    if mixed_live_widths and (
        topk_cap is None
        or list(mixed_live_widths) != sorted(set(mixed_live_widths))
        or any(w % 128 or not 0 < w < topk_cap for w in mixed_live_widths)
    ):
        raise ValueError(
            f"mixed_live_widths {mixed_live_widths} must be ascending multiples of "
            f"128 below topk_cap ({topk_cap})"
        )
    if decode_row_widths and (
        topk_cap is None
        or list(decode_row_widths) != sorted(set(decode_row_widths))
        or any(w % 128 or not 0 < w < topk_cap for w in decode_row_widths)
    ):
        raise ValueError(
            f"decode_row_widths {decode_row_widths} must be ascending multiples of "
            f"128 below topk_cap ({topk_cap})"
        )
    if not set(mixed_body_levers) <= set(MIXED_BODY_LEVERS):
        raise ValueError(
            f"mixed_body_levers {mixed_body_levers} not in {MIXED_BODY_LEVERS}"
        )
    num_chunks = cdiv(q.shape[0], gather_and_attention_chunk_size)
    # With the cap, translate the top-k once per call straight into the capped
    # window-major layout, live windows only (`prepare_capped_window_indices`),
    # whenever every chunk has one static size; otherwise the full-width
    # `prepare_gather_indices` plus a per-chunk transpose. Same gathered rows.
    uniform_chunk = min(gather_and_attention_chunk_size, q.shape[0])
    window_prep = (
        topk_cap is not None
        and page_indices is not None
        and q.shape[0] % uniform_chunk == 0
    )
    window_indices_all = None
    chunk_window_rows = 0  # rows of `window_indices_all` per chunk
    group_tab = None
    group_num_live = None
    fused_prep = (
        fused_window_prep
        and window_prep
        and num_chunks == 1
        and q.shape[0] <= FUSED_PREP_MAX_TOKENS
    )
    if fused_prep and not lse_mask_empty:
        raise ValueError("fused_window_prep requires lse_mask_empty")
    if kv_write is not None and not fused_prep:
        raise ValueError("kv_write needs the fused-prep decode path (one gather)")
    if fused_prep:
        assert topk_cap is not None
        # Only the attention kernel reads kv_lens here: the fused prep counts
        # live entries itself, so XLA may run this reduce in the gather's shadow.
        kv_lens = jnp.sum(topk_indices != -1, axis=-1)
        physical_indices = None
        prep_small, prep_batch = _chunk_kernel_batch_size(
            uniform_chunk,
            attention_kernel_batch_size,
            small_chunk_tokens,
            small_chunk_batch_size,
        )
        window_indices_all = prepare_capped_window_fused(
            topk_indices,
            page_indices,
            cu_q_lens,
            cu_q_lens[distribution[2]],
            topk_cap=topk_cap,
            page_size=page_size,
            total_num_pages=cache_kv_nope.shape[0],
            kernel_batch_size=prep_batch,
            gather_atoms=(
                small_chunk_gather_atoms
                if prep_small and small_chunk_live_gather
                else sparsecore_atoms_per_batch
            ),
        )
    elif (
        window_prep
        and mixed_live_gather
        and q.shape[0] > FUSED_PREP_MAX_TOKENS
        and live_group_supported(
            uniform_chunk,
            _chunk_kernel_batch_size(
                uniform_chunk,
                attention_kernel_batch_size,
                small_chunk_tokens,
                small_chunk_batch_size,
            )[1],
            cast(int, topk_cap),
        )
    ):
        # Mixed / prefill buckets: window 0 in live-group order, so each
        # chunk gathers and each grid step fetches only its live atom blocks.
        assert topk_cap is not None
        chunk_window_rows = topk * uniform_chunk // 128
        kv_lens = jnp.sum(topk_indices != -1, axis=-1)
        physical_indices = None
        _, prep_batch = _chunk_kernel_batch_size(
            uniform_chunk,
            attention_kernel_batch_size,
            small_chunk_tokens,
            small_chunk_batch_size,
        )
        token_end = cu_q_lens[distribution[2]]
        if mixed_fused_prep and mixed_fused_prep_supported(
            q.shape[0], topk, uniform_chunk, prep_batch, topk_cap
        ):
            window_indices_all, kv_lens, group_tab, group_num_live = (
                prepare_capped_window_live_fused(
                    topk_indices,
                    page_indices,
                    cu_q_lens,
                    token_end,
                    chunk_size=uniform_chunk,
                    kernel_batch_size=prep_batch,
                    topk_cap=topk_cap,
                    page_size=page_size,
                    total_num_pages=cache_kv_nope.shape[0],
                )
            )
        else:
            live_windows = capped_live_windows(
                kv_lens,
                token_end,
                chunk_size=uniform_chunk,
                kernel_batch_size=prep_batch,
                topk_cap=topk_cap,
                num_windows=topk // topk_cap,
            )
            group_tab, group_num_live = live_group_table(
                kv_lens,
                token_end,
                chunk_size=uniform_chunk,
                kernel_batch_size=prep_batch,
                topk_cap=topk_cap,
                num_windows=topk // topk_cap,
            )
            window_indices_all = prepare_capped_window_live(
                topk_indices,
                page_indices,
                cu_q_lens,
                live_windows,
                group_tab,
                chunk_size=uniform_chunk,
                kernel_batch_size=prep_batch,
                topk_cap=topk_cap,
                page_size=page_size,
                total_num_pages=cache_kv_nope.shape[0],
            )
    elif window_prep:
        assert topk_cap is not None
        chunk_window_rows = topk // topk_cap * uniform_chunk
        kv_lens = jnp.sum(topk_indices != -1, axis=-1)
        physical_indices = None
        _, prep_batch = _chunk_kernel_batch_size(
            uniform_chunk,
            attention_kernel_batch_size,
            small_chunk_tokens,
            small_chunk_batch_size,
        )
        live_windows = capped_live_windows(
            kv_lens,
            cu_q_lens[distribution[2]],
            chunk_size=uniform_chunk,
            kernel_batch_size=prep_batch,
            topk_cap=topk_cap,
            num_windows=topk // topk_cap,
        )
        window_indices_all = prepare_capped_window_indices(
            topk_indices,
            page_indices,
            cu_q_lens,
            live_windows,
            chunk_size=uniform_chunk,
            topk_cap=topk_cap,
            page_size=page_size,
            total_num_pages=cache_kv_nope.shape[0],
        )
    else:
        kv_lens, physical_indices = prepare_gather_indices(
            topk_indices,
            page_indices,
            cu_q_lens,
            page_size=page_size,
            total_num_pages=cache_kv_nope.shape[0],
        )

    # Rows the grid never reaches (padding past `batch_end`) keep this
    # sentinel, so a downstream merge weights them at exp(min - max) == 0
    # instead of tripping on uninitialised memory.
    lse_fill = bool(lse_in_kernel and return_lse and lse_lane_dense and num_chunks == 1)
    lse = (
        jnp.full(
            (q.shape[0], num_q_heads)
            if lse_lane_dense
            else (q.shape[0], num_q_heads, 1),
            -jnp.inf if lse_mask_empty else jnp.finfo(jnp.float32).min,
            dtype=jnp.float32,
        )
        if return_lse and not lse_fill
        else None
    )

    for i in range(num_chunks):
        start_pos = i * gather_and_attention_chunk_size
        end_pos = min(start_pos + gather_and_attention_chunk_size, q.shape[0])
        result = sparse_attention_chunk(
            q,
            cache_kv_nope,
            cache_kv_rope,
            kv_lens,
            physical_indices,
            lse,
            start_pos=start_pos,
            end_pos=end_pos,
            token_end=cu_q_lens[distribution[2]],
            sm_scale=sm_scale,
            k_scale=k_scale,
            attention_kernel_batch_size=attention_kernel_batch_size,
            vmem_limit_bytes=vmem_limit_bytes,
            cache_layout=cache_layout,
            sparsecore_atoms_per_batch=sparsecore_atoms_per_batch,
            sparsecore_gather_wait_mode=sparsecore_gather_wait_mode,
            sparsecore_gather_pipelined=sparsecore_gather_pipelined,
            small_chunk_tokens=small_chunk_tokens,
            small_chunk_batch_size=small_chunk_batch_size,
            small_chunk_live_gather=small_chunk_live_gather,
            small_chunk_gather_atoms=small_chunk_gather_atoms,
            live_gather_all_chunks=live_gather_all_chunks,
            topk_cap=topk_cap,
            topk_override=topk,
            window_indices=(
                window_indices_all
                if fused_prep or window_indices_all is None
                else window_indices_all[i]
                if isinstance(window_indices_all, tuple)
                else window_indices_all[
                    i * chunk_window_rows : (i + 1) * chunk_window_rows
                ]
            ),
            gather_live_in_indices=fused_prep,
            q_pe=q_pe,
            lse_lane_dense=lse_lane_dense,
            lse_mask_empty=lse_mask_empty,
            group_tab=group_tab,
            group_num_live=None if group_num_live is None else group_num_live[i],
            group_widths=mixed_live_widths if group_tab is not None else (),
            row_widths=decode_row_widths if group_tab is None else (),
            group_levers=(
                mixed_body_levers
                if group_tab is not None and q.shape[0] > MIXED_BODY_MAX_SKIP_TOKENS
                else ()
            ),
            q_rope_table=q_rope_table,
            lse_fill=lse_fill,
            prologue_fault=prologue_fault,
            kv_write=kv_write,
            kv_write_fault=kv_write_fault,
        )
        if kv_write is not None:
            result, (cache_kv_nope, cache_kv_rope) = result
        if return_lse:
            q, lse = result
        else:
            q = result

    # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    out = prepare_outputs(q, actual_num_q_heads, actual_head_dim)
    caches = () if kv_write is None else (cache_kv_nope, cache_kv_rope)
    if return_lse:
        # `lse` is the kernel's lse array here; the cast only stops the checker
        # reading it as the (caches) tuple of the kv_write return shape.
        lse = cast(jax.Array, lse)
        if lse_lane_dense:
            return out, lse[:, :actual_num_q_heads], *caches
        return out, lse[:, :actual_num_q_heads, 0], *caches
    return (out, *caches) if caches else out


def prepare_gather_indices(
    topk_indices: jax.Array,  # i32[max_num_tokens, csa_topk]
    page_indices: jax.Array | None,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    *,
    page_size: int,
    total_num_pages: int,
) -> tuple[jax.Array, jax.Array]:
    """Per-token valid top-k counts and flat cache row indices to gather.

    Depends only on the indexer output and the page table, not on the query.
    """
    max_num_seqs = cu_q_lens.shape[0] - 1
    num_tokens, topk = topk_indices.shape
    # TODO: skip gather for padding tokens in topk_indices.
    kv_lens = jnp.sum(topk_indices != -1, axis=-1)
    is_padding = topk_indices == -1
    flat_element_index = jnp.arange(num_tokens * topk, dtype=jnp.int32).reshape(
        num_tokens, topk
    )
    if page_indices is not None:
        num_page_indices = page_indices.shape[0]
        if num_page_indices % max_num_seqs != 0:
            raise ValueError(
                f"page_indices size ({num_page_indices}) must be divisible by "
                f"the sequence count ({max_num_seqs})"
            )
        tokens_per_seq = cu_q_lens[1:] - cu_q_lens[:-1]
        seq_ids_segment = jnp.repeat(
            jnp.arange(max_num_seqs), tokens_per_seq, total_repeat_length=num_tokens
        )
        seq_page_ids = topk_indices // page_size
        token_offset = topk_indices % page_size
        page_ids = gather_page_ids(
            page_indices, seq_page_ids, seq_ids_segment, max_num_seqs
        )

        # Scatter "-1" padding across the cache to avoid read hot spots.
        scattered_page_ids = (flat_element_index * 104729) % total_num_pages
        scattered_token_offset = (flat_element_index * 15485863) % page_size
        page_ids = jnp.where(is_padding, scattered_page_ids, page_ids)
        token_offset = jnp.where(is_padding, scattered_token_offset, token_offset)
        physical_indices = page_ids * page_size + token_offset
    else:
        total_tokens = total_num_pages * page_size
        scattered_physical = (flat_element_index * 15485863) % total_tokens
        physical_indices = jnp.where(is_padding, scattered_physical, topk_indices)

    physical_indices = physical_indices.astype(jnp.int32)
    return kv_lens, physical_indices


def sparse_attention_chunk(
    q: jax.Array,  # [max_num_tokens, num_q_heads, head_dim], padded
    # TensorCore: u8[pages, page_size, 4, 128].
    # SparseCore: u32[pages, page_size, 128].
    cache_kv_nope: jax.Array,
    # TensorCore: u8[pages, page_size/4, 4, 128].
    # SparseCore: u32[pages, page_size/4, 128].
    cache_kv_rope: jax.Array,
    kv_lens: jax.Array,  # i32[max_num_tokens]
    physical_indices: jax.Array | None,  # i32[max_num_tokens, csa_topk]
    lse: jax.Array | None,  # f32[max_num_tokens, num_q_heads, 1]
    *,
    start_pos: int,
    end_pos: int,
    token_end: jax.Array,  # i32
    sm_scale: float,
    k_scale: float,
    preserve_suffix: bool = False,
    attention_kernel_batch_size: int = DEFAULT_ATTENTION_KERNEL_BATCH_SIZE,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    cache_layout: str = TENSORCORE_CACHE_LAYOUT,
    sparsecore_atoms_per_batch: int = 16,
    sparsecore_gather_wait_mode: str = "overlap_rope",
    sparsecore_gather_pipelined: bool = True,
    small_chunk_tokens: int = 0,
    small_chunk_batch_size: int | None = None,
    small_chunk_live_gather: bool = False,
    small_chunk_gather_atoms: int = 1,
    live_gather_all_chunks: bool = False,
    topk_cap: int | None = None,
    topk_override: int | None = None,
    window_indices: jax.Array | None = None,
    gather_live_in_indices: bool = False,
    q_pe: jax.Array | None = None,
    lse_lane_dense: bool = False,
    lse_mask_empty: bool = False,
    group_tab: jax.Array | None = None,
    group_num_live: jax.Array | None = None,
    group_widths: tuple[int, ...] = (),
    group_levers: tuple[str, ...] = (),
    q_rope_table: jax.Array | None = None,
    lse_fill: bool = False,
    prologue_fault: str | None = None,
    kv_write: jax.Array | tuple[jax.Array, jax.Array, jax.Array] | None = None,
    kv_write_fault: str | None = None,
    row_widths: tuple[int, ...] = (),
):
    """Gathers and attends query rows `[start_pos, min(end_pos, token_end))`.

    The output aliases `q`. Returns the updated `q`, or `(q, lse)` when `lse`
    is given. The attention grid is rounded up to its kernel batch size, so
    up to `attention_kernel_batch_size - 1` rows at or after `token_end` may
    also be overwritten; rows further on keep their input values. Private
    split routing sets `preserve_suffix` to restrict stores to owned rows.

    `topk_cap` (SparseCore layout only; a multiple of 64 dividing topk) gathers
    the chunk's top-k window-major in `topk_cap`-slot windows, bounded to the
    windows its live rows reach, and attends window 0 per token with an
    in-kernel fix-up over all windows for tokens holding more live entries.
    `window_indices` (with `topk_cap`; from `prepare_capped_window_indices`)
    is that window-major index array prebuilt for this chunk, in which case
    `physical_indices` may be None and `topk_override` must be given.
    `q_pe` (split-q): `q` is q_nope `[max_num_tokens, num_q_heads, nope]`
    (unpadded, nope a multiple of 128) and `q_pe` its rope part; the kernel
    joins them in VMEM and the output (aliasing `q`) is `nope` wide.
    `lse_lane_dense`: `lse` is f32[max_num_tokens, num_q_heads].
    `gather_live_in_indices` (with `window_indices` from
    `prepare_capped_window_fused`): `window_indices` is its 128-lane form and
    carries the gather's live bound, already computed, in the tile after the
    rows. `lse_mask_empty`: kv_len 0 tokens get lse -inf in-kernel.
    `group_tab` / `group_num_live` (with `topk_cap`; from `live_group_table`):
    `window_indices` is this chunk's rows of `prepare_capped_window_live`
    (128-lane linear form, window 0 in live-group order); the gather is
    bounded to `group_num_live` indices and each grid step fetches only its
    live atom blocks. `group_widths` (with `group_tab`): static row widths the
    kernel specializes its body to per grid step (see `_attention_kernel`).
    `kv_write` (with `gather_live_in_indices`): the gather first writes this
    step's packed tokens into both caches; returns `(result, (nope, rope))`.
    """
    if kv_write is not None and not gather_live_in_indices:
        raise ValueError("kv_write rides the fused-prep capped gather only")
    return_lse = lse is not None or lse_fill
    assert not (lse_fill and lse is not None), "lse_fill owns the lse buffer"
    q_rope = q_rope_table is not None
    assert not q_rope or q_pe is not None, "q_rope_table needs split q"
    if topk_override is not None:
        topk = topk_override
    else:
        assert physical_indices is not None
        topk = physical_indices.shape[-1]
    if window_indices is not None and topk_cap is None:
        raise ValueError("window_indices requires topk_cap")
    sparsecore_layout = cache_layout == SPARSECORE_CACHE_LAYOUT
    if topk_cap is not None and not sparsecore_layout:
        raise ValueError("topk_cap requires the sparsecore cache layout")
    if topk_cap is not None and (topk_cap % GATHER_ATOM_ROWS or topk % topk_cap):
        raise ValueError(
            f"topk_cap ({topk_cap}) must be a multiple of {GATHER_ATOM_ROWS} "
            f"dividing topk ({topk})"
        )
    if preserve_suffix and (return_lse or sparsecore_layout):
        raise ValueError("Prefix stores require TensorCore caches without LSE")
    _, num_q_heads, head_dim = q.shape
    split_q = q_pe is not None
    if split_q:
        head_dim = align_to(head_dim + (Q_ROPE_DIM if q_rope else q_pe.shape[-1]), 128)
    chunk_size = end_pos - start_pos
    capped_windows = None if topk_cap is None else topk // topk_cap

    def run_mla_kernel(
        q: jax.Array,  # [max_num_tokens, num_q_heads, head_dim]
        # [num_tokens, topk * WORD_BYTES, TILE_LANE_BYTES] (gathered raw tiles)
        cache_kv_nope: jax.Array,
        cache_kv_rope: jax.Array,  # [num_tokens, topk, rope_dim] (gathered)
        kv_lens: jax.Array,  # i32[max_num_tokens]
        start_seq_idx: jax.Array,  # i32
        end_seq_idx: jax.Array,  # i32
        kernel_batch_size: int,
        lse: jax.Array | None = None,  # f32[max_num_tokens, num_q_heads, 1]
        live_views: tuple[jax.Array, jax.Array] | None = None,
    ):
        batch_size = kernel_batch_size
        lse_shape = (
            (q.shape[0], num_q_heads)
            if lse_fill
            else (None if lse is None else lse.shape)
        )
        end_seq_idx = jnp.maximum(start_seq_idx, end_seq_idx)
        grid = (cdiv(end_seq_idx - start_seq_idx, batch_size),)
        if lse_fill:
            # Never empty: step 0 lays down the lse sentinel.
            grid = (jnp.maximum(grid[0], 1),)
        in_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),  # q
            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv_nope
            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv_rope
        ]
        if split_q:
            in_specs.append(pl.BlockSpec(memory_space=pltpu.HBM))  # q_pe
        if q_rope:
            in_specs.append(pl.BlockSpec(memory_space=pltpu.HBM))  # rope table
        if live_views is not None:
            in_specs += [pl.BlockSpec(memory_space=pltpu.HBM)] * 2  # block views

        out_specs = pl.BlockSpec(memory_space=pltpu.HBM)  # o
        # Each chunk writes only its own token rows, so -- exactly like `q` --
        # the lse buffer is threaded through as an aliased in/out to carry
        # earlier chunks' rows forward.
        if return_lse:
            if not lse_fill:
                in_specs.append(pl.BlockSpec(memory_space=pltpu.HBM))  # lse (in)
            out_specs = [out_specs, pl.BlockSpec(memory_space=pltpu.HBM)]

        # One batch entry's worth of gathered top-k rows, per cache.
        nope_tile = cache_kv_nope.shape[1:]
        rope_tile = cache_kv_rope.shape[1:]
        bkv_nope_double_buf = pltpu.VMEM(
            (2, batch_size, *nope_tile),
            cache_kv_nope.dtype,
        )
        bkv_rope_double_buf = pltpu.VMEM(
            (2, batch_size, *rope_tile),
            cache_kv_rope.dtype,
        )

        bq_double_buf = pltpu.VMEM(
            (2, batch_size, num_q_heads, q.shape[-1]),
            q.dtype,
        )

        bo_double_buf = bq_double_buf

        scratch_shapes = [
            bkv_nope_double_buf,
            bkv_rope_double_buf,
            bq_double_buf,
            bo_double_buf,  # Double buffering for output block.
        ]
        if return_lse:
            scratch_shapes.append(
                pltpu.VMEM(
                    (2, batch_size, num_q_heads)
                    if lse_lane_dense
                    else (2, batch_size, num_q_heads, 1),
                    jnp.float32,
                )
            )
        # Semaphores for double buffering of bkv_nope, bq, bo, bkv_rope
        # (+ blse when the log-sum-exp is exported).
        scratch_shapes.append(pltpu.SemaphoreType.DMA((5 if return_lse else 4, 2)))
        if capped_windows is not None:
            # Overflow fix-up: two window slots per cache, running max / sum /
            # accumulator, and a semaphore per (cache, slot).
            scratch_shapes += [
                pltpu.VMEM((2, *nope_tile), cache_kv_nope.dtype),
                pltpu.VMEM((2, *rope_tile), cache_kv_rope.dtype),
                pltpu.VMEM((num_q_heads, 1), jnp.float32),
                pltpu.VMEM((num_q_heads, 1), jnp.float32),
                pltpu.VMEM((num_q_heads, head_dim), jnp.float32),
                pltpu.SemaphoreType.DMA((2, 2)),
            ]
        if lse_fill:
            # The sentinel tile for the whole lse output, and its DMA semaphore.
            scratch_shapes += [
                pltpu.VMEM(lse_shape, jnp.float32),
                pltpu.SemaphoreType.DMA(()),
            ]
        if q_rope:
            assert q_rope_table is not None
            # Rope table double buffer; just before q_pe's.
            scratch_shapes.append(
                pltpu.VMEM((2, batch_size, *q_rope_table.shape[1:]), jnp.float32)
            )
        if split_q:
            # q_pe double buffer; always the last scratch ref.
            scratch_shapes.append(
                pltpu.VMEM((2, batch_size, num_q_heads, q_pe.shape[-1]), q_pe.dtype)
            )

        bounds = [start_seq_idx, end_seq_idx]
        if preserve_suffix:
            bounds.append(token_end)
        scalar_prefetches = (
            kv_lens,
            jnp.array(bounds, jnp.int32),
            # (bi_sem_idx, bo_sem_idx)
            jnp.zeros((2,), jnp.int32),
        )
        if live_views is not None:
            scalar_prefetches += (group_tab,)
        kernel_body = functools.partial(
            _attention_kernel,
            sm_scale=sm_scale,
            k_scale=k_scale,
            batch_size=batch_size,
            preserve_suffix=preserve_suffix,
            return_lse=return_lse,
            sparsecore_layout=sparsecore_layout,
            capped_windows=capped_windows,
            split_q=split_q,
            lse_lane_dense=lse_lane_dense,
            lse_mask_empty=lse_mask_empty,
            q_rope=q_rope,
            lse_fill=lse_fill,
            fault=prologue_fault,
            row_widths=tuple(row_widths) if capped_windows is not None else (),
        )
        if live_views is not None:
            group_body = functools.partial(
                kernel_body,
                live_widths=tuple(group_widths),
                body_levers=tuple(group_levers),
            )

            def kernel_body(kv_lens_ref, bounds_ref, sem_ids_ref, tab_ref, *rest):
                return group_body(
                    kv_lens_ref, bounds_ref, sem_ids_ref, *rest, group_tab_ref=tab_ref
                )

        scope_name = (
            f"MLA-{cache_layout}-p_{cache_kv_rope.shape[1]}-"
            f"bz_{batch_size}-gcz_{cache_kv_nope.shape[0]}"
        )
        if capped_windows is not None:
            scope_name += f"-capw_{capped_windows}"
        num_prefetch = len(scalar_prefetches)
        kernel = jax.named_scope(scope_name)(
            pl.pallas_call(
                kernel_body,
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=len(scalar_prefetches),
                    in_specs=in_specs,
                    out_specs=out_specs,
                    grid=grid,
                    scratch_shapes=scratch_shapes,
                ),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("arbitrary",),
                    vmem_limit_bytes=vmem_limit_bytes,
                    disable_bounds_checks=True,
                ),
                out_shape=(
                    [
                        jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
                        jax.ShapeDtypeStruct(shape=lse_shape, dtype=jnp.float32),
                    ]
                    if return_lse
                    else jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
                ),
                input_output_aliases=(
                    {num_prefetch: 0}
                    if lse_fill
                    else {
                        num_prefetch: 0,
                        num_prefetch
                        + 3
                        + int(split_q)
                        + int(q_rope)
                        + 2 * int(live_views is not None): 1,
                    }
                    if return_lse
                    else {
                        # 3 = q; operand indices count the scalar prefetches
                        # (DSV4's "4" also counted attention_sinks).
                        num_prefetch: 0,  # Alias output activation with q
                    }
                ),
                name=scope_name,
            )
        )
        operands = [q, cache_kv_nope, cache_kv_rope]
        if split_q:
            operands.append(q_pe)
        if q_rope:
            operands.append(q_rope_table)
        if live_views is not None:
            operands += list(live_views)
        if return_lse and not lse_fill:
            operands.append(lse)
        return kernel(*scalar_prefetches, *operands)

    indices = None if physical_indices is None else physical_indices[start_pos:end_pos]

    # Batching
    small_chunk, kernel_batch_size = _chunk_kernel_batch_size(
        chunk_size,
        attention_kernel_batch_size,
        small_chunk_tokens,
        small_chunk_batch_size,
    )
    assert chunk_size % kernel_batch_size == 0
    # The kernel grid walks [start_pos, batch_end) in `kernel_batch_size`
    # steps, so `batch_end - start_pos` MUST be a multiple of
    # `kernel_batch_size`.
    valid_rows = jnp.maximum(0, jnp.minimum(token_end, end_pos) - start_pos)
    batch_end = start_pos + cdiv(valid_rows, kernel_batch_size) * kernel_batch_size
    num_valid_indices = jnp.maximum(0, batch_end - start_pos) * topk

    # For prefilling of short sequences (or early in the sequence), there are
    # very few number of KVs in the sequence, so different qs' selected topk
    # would have large overlap. This causes gather read hotspotting. We've seen
    # 30%+ performance degradation compared to the no-duplicate-indices case.
    #
    # TODO: we could consider let the caller (tpu-runner) to sort the sequences
    # based on their lengths. For the sequences-segment below certain length,
    # we use a different kernel (dense attention and mask), for the rest of
    # sequences, we use this gather-and-attention kernel.
    live_views = None
    if group_tab is not None and topk_cap is None:
        raise ValueError("group_tab requires topk_cap")
    if topk_cap is not None:
        # Window-major: window w of the chunk's top-k is rows [w*C, (w+1)*C)
        # of a [W*C, topk_cap] index array; window 0 is the plain `topk_cap`
        # slice. The live bound reaches the last window any gridded row needs
        # (always window 0's live rows, so a chunk without overflow gathers
        # exactly `(batch_end - start) * topk_cap` rows).
        assert capped_windows is not None
        gather_atoms = sparsecore_atoms_per_batch
        if small_chunk and small_chunk_live_gather:
            gather_atoms = small_chunk_gather_atoms
        if gather_live_in_indices:
            # `window_indices` is then the fused prep's 128-lane form: hand it
            # to the gather flat (a bitcast, no relayout); its live vector
            # sits right after the chunk's rows.
            live_kwargs = dict(live_vec_offset=chunk_size * topk, linear_indices=True)
        elif group_tab is not None:
            # Live-group prep: 128-lane linear rows, bound from the table.
            assert window_indices is not None and group_num_live is not None
            live_kwargs = dict(num_live_indices=group_num_live, linear_indices=True)
        else:
            live_tokens = batch_end - start_pos
            token_windows = jnp.where(
                jnp.arange(chunk_size) < live_tokens,
                (jnp.maximum(kv_lens[start_pos:end_pos], 1) + topk_cap - 1) // topk_cap,
                1,
            )
            live_windows = jnp.clip(jnp.max(token_windows), 1, capped_windows)
            num_live_capped = (live_windows - 1) * chunk_size * topk_cap + jnp.maximum(
                0, live_tokens
            ) * topk_cap
            live_kwargs = dict(num_live_indices=num_live_capped)
        if window_indices is None:
            assert indices is not None
            window_indices = (
                indices.reshape(chunk_size, capped_windows, topk_cap)
                .transpose(1, 0, 2)
                .reshape(capped_windows * chunk_size, topk_cap)
            )
        gathered = native_sc_gather.dsa_gather_native_sc(
            cache_kv_nope,
            cache_kv_rope,
            window_indices,
            out_size=chunk_size * topk,
            atoms_per_batch=gather_atoms,
            wait_mode=sparsecore_gather_wait_mode,
            serve_pipelined=sparsecore_gather_pipelined,
            live_multiple=kernel_batch_size
            * (GATHER_ATOM_ROWS if group_tab is not None else topk_cap),
            kv_write=kv_write,
            write_fault=kv_write_fault,
            **live_kwargs,
        )
        gathered_nope_buffer, gathered_rope_buffer = gathered[:2]
        if kv_write is not None:
            cache_kv_nope, cache_kv_rope = gathered[2:]
        gathered_nope_buffer = gathered_nope_buffer.reshape(
            capped_windows * chunk_size, topk_cap, 128
        )
        gathered_rope_buffer = gathered_rope_buffer.reshape(
            capped_windows * chunk_size, topk_cap // native_sc_gather.ROPE_PACKING, 128
        )
        if group_tab is not None:
            block = kernel_batch_size * GATHER_ATOM_ROWS
            live_views = (
                gathered_nope_buffer.reshape(
                    chunk_size * topk // block, kernel_batch_size, GATHER_ATOM_ROWS, 128
                ),
                gathered_rope_buffer.reshape(
                    chunk_size * topk // block,
                    kernel_batch_size,
                    GATHER_ATOM_ROWS // native_sc_gather.ROPE_PACKING,
                    128,
                ),
            )
    elif sparsecore_layout:
        live_kwargs = {}
        gather_atoms = sparsecore_atoms_per_batch
        if small_chunk and small_chunk_live_gather:
            gather_atoms = small_chunk_gather_atoms
            live_kwargs = dict(
                num_live_indices=num_valid_indices,
                live_multiple=kernel_batch_size * topk,
            )
        elif live_gather_all_chunks:
            live_kwargs = dict(
                num_live_indices=num_valid_indices,
                live_multiple=kernel_batch_size * topk,
            )
        gathered_nope_buffer, gathered_rope_buffer = (
            native_sc_gather.dsa_gather_native_sc(
                cache_kv_nope,
                cache_kv_rope,
                indices,
                out_size=chunk_size * topk,
                atoms_per_batch=gather_atoms,
                wait_mode=sparsecore_gather_wait_mode,
                serve_pipelined=sparsecore_gather_pipelined,
                **live_kwargs,
            )
        )
        gathered_nope_buffer = gathered_nope_buffer.reshape(chunk_size, topk, 128)
        gathered_rope_buffer = gathered_rope_buffer.reshape(
            chunk_size, topk // native_sc_gather.ROPE_PACKING, 128
        )
    else:
        assert indices is not None
        gathered_nope_buffer, gathered_rope_buffer = dsa_gather.dsa_gather(
            cache_kv_nope,
            cache_kv_rope,
            indices.reshape(-1),
            num_valid_indices=num_valid_indices,
        )
        gathered_nope_buffer = gathered_nope_buffer.reshape(
            chunk_size,
            topk * kv_cache_utils.WORD_BYTES,
            kv_cache_utils.TILE_LANE_BYTES,
        )
        gathered_rope_buffer = gathered_rope_buffer.reshape(chunk_size, topk, -1)
    # We treat each query token as a one independent sequence, attend to their
    # respective gathered kv tokens in the `gathered_kv_buffer`.
    # -1 in topk_indices is padded elements at the end of each row.
    result = run_mla_kernel(
        q,
        gathered_nope_buffer,
        gathered_rope_buffer,
        kv_lens,
        start_seq_idx=start_pos,
        end_seq_idx=batch_end,
        kernel_batch_size=kernel_batch_size,
        lse=lse,
        live_views=live_views,
    )
    if kv_write is not None:
        return result, (cache_kv_nope, cache_kv_rope)
    return result
