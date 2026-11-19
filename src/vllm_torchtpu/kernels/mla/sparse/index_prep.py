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
"""Index preparation kernels and helpers for SparseMLA."""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.mla.sparse import native_sc_gather
from vllm_torchtpu.kernels.mla.sparse.config import (
    FUSED_PREP_MAX_TOKENS,
    FUSED_PREP_META_ROWS,
    GATHER_ATOM_ROWS,
    align_to,
    cdiv,
    live_group_supported,
    mixed_fused_prep_supported,
)
from vllm_torchtpu.kernels.mla.sparse.ref_registry import (
    LiveFusedPrepRefs,
    RefLayout,
    build_live_fused_prep_ref_registry,
)
from vllm_torchtpu.kernels.mla.sparse.testing_faults import (
    NO_FAULT_HOOKS,
    FaultHooks,
)

_GATHER_PAGE_CHUNK = 128
# murmur3 fmix32 multipliers as wrapping int32.
_FMIX32_C1 = np.uint32(0x85EBCA6B).view(np.int32).item()
_FMIX32_C2 = np.uint32(0xC2B2AE35).view(np.int32).item()
_LIVE_TOKEN_MAJOR_SEM = 2


def _token_seq_ids(cu_q_lens: jax.Array, num_tokens: int) -> jax.Array:
    """Maps each token position in ``[0, num_tokens)`` to its sequence id."""
    max_num_seqs = cu_q_lens.shape[0] - 1
    tokens_per_seq = cu_q_lens[1:] - cu_q_lens[:-1]
    return jnp.repeat(
        jnp.arange(max_num_seqs), tokens_per_seq, total_repeat_length=num_tokens
    )


def _page_table_windows(
    page_indices: jax.Array,
    seq_ids_segment: jax.Array,
    max_num_seqs: int,
) -> tuple[jax.Array, int]:
    """Pads the page table to ``_GATHER_PAGE_CHUNK`` and gathers per-token rows."""
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    padded_pps = align_to(pages_per_seq, _GATHER_PAGE_CHUNK)
    page_table = page_indices.reshape(max_num_seqs, pages_per_seq)
    if padded_pps != pages_per_seq:
        page_table = jnp.pad(
            page_table, ((0, 0), (0, padded_pps - pages_per_seq))
        )
    # Per-token page-table window (whole-row gather).
    return page_table[seq_ids_segment], pages_per_seq


def _build_token_page_windows(
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    num_tokens: int,
) -> tuple[jax.Array, int]:
    """Builds padded per-token page-table windows and ``pages_per_seq``.

    Returns ``(windows, pages_per_seq)`` where ``windows`` has shape
    ``i32[num_tokens, align_to(pages_per_seq, _GATHER_PAGE_CHUNK)]``.
    """
    max_num_seqs = cu_q_lens.shape[0] - 1
    seq_ids_segment = _token_seq_ids(cu_q_lens, num_tokens)
    return _page_table_windows(page_indices, seq_ids_segment, max_num_seqs)


def _lookup_physical_page(logical: jax.Array, windows_ref: Any) -> jax.Array:
    """Looks up physical pages from chunked per-token page-table rows.

    Processed in chunks of ``_GATHER_PAGE_CHUNK`` to stay within TPU vector
    register limits.
    """
    page = jnp.zeros_like(logical)
    for pc in range(windows_ref.shape[1] // _GATHER_PAGE_CHUNK):
        window_chunk = windows_ref[
            :, pc * _GATHER_PAGE_CHUNK : (pc + 1) * _GATHER_PAGE_CHUNK
        ]
        local = logical - pc * _GATHER_PAGE_CHUNK
        gathered = jnp.take_along_axis(
            window_chunk, jnp.clip(local, 0, _GATHER_PAGE_CHUNK - 1), axis=1
        )
        page = jnp.where(
            (local >= 0) & (local < _GATHER_PAGE_CHUNK), gathered, page
        )
    return page


def _floor_divmod_const(x: jax.Array, d: int) -> tuple[jax.Array, jax.Array]:
    """Computes exact integer ``(x // d, x % d)`` without hardware integer division.

    TPU vector units lack a native vector ``idiv`` instruction. Powers of two
    use bitwise shift and mask; arbitrary divisors use a float32 reciprocal
    with remainder refinement and +/-1 correction for bit-exact floor semantics.

    Args:
        x: Input int32 tensor of arbitrary shape.
        d: Static positive integer divisor (must be > 0).

    Returns:
        ``(quotient, remainder)`` int32 arrays with exact floor semantics.
    """
    if d & (d - 1) == 0:
        shift = d.bit_length() - 1
        return lax.shift_right_arithmetic(x, jnp.int32(shift)), x & jnp.int32(d - 1)

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


def _cdiv_const(x: jax.Array, d: int) -> jax.Array:
    """Computes ``ceil(x / d)`` for a non-negative int32 array and static ``d``."""
    if d & (d - 1) == 0:
        return lax.shift_right_logical(x + (d - 1), jnp.int32(d.bit_length() - 1))
    return _floor_divmod_const(x + (d - 1), d)[0]


def scatter_padding_slots(
    flat_element_index: jax.Array, total_num_pages: int, page_size: int
) -> tuple[jax.Array, jax.Array]:
    """Hashes flat slot indices to spread-out ``(page, offset)`` for ``-1`` padding.

    Uses murmur3 fmix32 on int32; shared by the Pallas kernels and XLA so both
    produce identical scattered addresses.
    """
    h = flat_element_index.astype(jnp.int32)
    h = h ^ lax.shift_right_logical(h, jnp.int32(16))
    h = h * jnp.int32(_FMIX32_C1)
    h = h ^ lax.shift_right_logical(h, jnp.int32(13))
    h = h * jnp.int32(_FMIX32_C2)
    h = h ^ lax.shift_right_logical(h, jnp.int32(16))
    # Top 23 bits feed the page: non-negative and exact for the f32 divmod.
    _, page = _floor_divmod_const(
        lax.shift_right_logical(h, jnp.int32(9)), total_num_pages
    )
    if page_size & (page_size - 1) == 0:
        offset = h & jnp.int32(page_size - 1)
    else:
        _, offset = _floor_divmod_const(h & jnp.int32((1 << 23) - 1), page_size)
    return page, offset


def _gather_page_ids_kernel(
    windows_ref: Any,
    logical_ref: Any,
    out_ref: Any,
    *,
    num_chunks: int | None = None,
) -> None:
    del num_chunks
    out_ref[...] = _lookup_physical_page(logical_ref[...], windows_ref)


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
    windows, pages_per_seq = _page_table_windows(
        page_indices, seq_ids_segment, max_num_seqs
    )
    padded_pps = windows.shape[1]
    num_chunks = padded_pps // _GATHER_PAGE_CHUNK
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


def _window_physical_rows(
    idx: jax.Array,  # i32[tokens, slots] logical top-k positions, -1 = empty
    windows_ref: Any,  # i32[tokens, padded_pps] page-table row per token
    flat_element_index: jax.Array,  # i32[tokens, slots] row * topk + col
    *,
    page_size: int,
    pages_per_seq: int,
    total_num_pages: int,
) -> jax.Array:
    """Translates a tile of logical slots to physical cache rows.

    Valid slots look up their physical page in the token's page-table row;
    empty slots (``-1``) are hashed by their flat ``(row, col)`` coordinate to
    pseudo-random rows so SparseCore gathers do not hot-spot one memory bank.
    """
    is_padding = idx == -1
    seq_page, token_offset = _floor_divmod_const(idx, page_size)
    logical = jnp.clip(seq_page, 0, pages_per_seq - 1)
    page = _lookup_physical_page(logical, windows_ref)
    scattered_page, scattered_offset = scatter_padding_slots(
        flat_element_index, total_num_pages, page_size
    )
    page = jnp.where(is_padding, scattered_page, page)
    token_offset = jnp.where(is_padding, scattered_offset, token_offset)
    return page * page_size + token_offset


def _pack_live_atom_blocks(
    phys_ref: Any,
    stage_ref: Any,
    slot: jax.Array,
    *,
    live: jax.Array,
    off: jax.Array,
    dead_before: jax.Array,
    bz: int,
    atoms_w: int,
    tile_rows: int,
    region_tiles: int,
    phys_row_stride: int,
    group_row_offset: int | jax.Array = 0,
) -> None:
    """Packs one group's window-0 64-slot atom blocks into live prefix / dead suffix."""
    atom = GATHER_ATOM_ROWS
    lane = lax.broadcasted_iota(jnp.int32, (bz // 2, 128), 1)
    for a in range(atoms_w):
        j, half = divmod(a * atom, 128)
        base = j * phys_row_stride + group_row_offset
        even = phys_ref[pl.ds(base, bz // 2, stride=2), :]
        odd = phys_ref[pl.ds(base + 1, bz // 2, stride=2), :]
        if half == 0:
            tile = jnp.where(lane < atom, even, pltpu.roll(odd, atom, 1))
        else:
            tile = jnp.where(lane < atom, pltpu.roll(even, atom, 1), odd)
        pos = jnp.where(
            a < live, off + a, region_tiles - 1 - (dead_before + a - live)
        )
        start = pl.multiple_of(pos * tile_rows, tile_rows)
        stage_ref[slot, pl.ds(start, tile_rows), :] = tile


def _compute_chunk_window_bounds(
    kv_lens: jax.Array,  # i32[num_tokens]
    token_end: jax.Array,  # i32[]
    *,
    chunk_size: int,
    kernel_batch_size: int,
    topk_cap: int,
    num_windows: int,
) -> tuple[jax.Array, jax.Array]:
    """Computes per-chunk ``(grid_rows, chunk_windows)`` over gridded tokens."""
    num_tokens = kv_lens.shape[0]
    num_chunks = num_tokens // chunk_size
    bz = kernel_batch_size
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
    return grid_rows, chunk_windows


def _capped_window_indices_kernel(
    live_windows_ref: Any,  # i32[1] (scalar prefetch; sizes the grid)
    windows_ref: Any,  # i32[block_tokens, padded_pps] page-table row per token
    topk_ref: Any,  # i32[block_tokens, topk_cap] one window of logical top-k
    out_ref: Any,  # i32[block_tokens, topk_cap] physical rows
    *,
    block_tokens: int,
    chunk_size: int,
    topk: int,
    topk_cap: int,
    page_size: int,
    pages_per_seq: int,
    total_num_pages: int,
) -> None:
    """Translates logical top-k slots to physical rows for one ``(w, c, tb)`` tile."""
    del live_windows_ref
    w = pl.program_id(0)
    c = pl.program_id(1)
    tb = pl.program_id(2)
    idx = topk_ref[...]
    row = (
        c * chunk_size
        + tb * block_tokens
        + lax.broadcasted_iota(jnp.int32, idx.shape, 0)
    )
    col = w * topk_cap + lax.broadcasted_iota(jnp.int32, idx.shape, 1)
    out_ref[...] = _window_physical_rows(
        idx,
        windows_ref,
        row * topk + col,
        page_size=page_size,
        pages_per_seq=pages_per_seq,
        total_num_pages=total_num_pages,
    )


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
    """Translates live capped windows from logical slots to physical cache rows.

    Sizes the Pallas grid dimension to ``live_windows`` so only windows read by
    the downstream gather are translated; higher windows stay unwritten.

    Args:
        topk_indices: Shape ``[num_tokens, topk]`` logical token positions.
        page_indices: Flattened page table mapping logical to physical pages.
        cu_q_lens: Cumulative query token counts per sequence.
        live_windows: Scalar in ``[1, topk // topk_cap]`` from
            ``capped_live_windows``.
        chunk_size: Attention chunk size (e.g. 256).
        topk_cap: Window capacity (e.g. 512).
        page_size: Tokens per physical page (e.g. 256).
        total_num_pages: Total physical cache pages allocated.
        block_tokens: Token sub-tile size inside each chunk (e.g. 16 or 32).

    Returns:
        Array of shape ``[num_chunks * num_windows * chunk_size, topk_cap]``
        in the window-major layout expected by the SparseCore gather.
    """
    num_tokens, topk = topk_indices.shape
    if num_tokens % chunk_size or topk % topk_cap:
        raise ValueError(
            f"num_tokens ({num_tokens}) and topk ({topk}) must be multiples of "
            f"chunk_size ({chunk_size}) and topk_cap ({topk_cap})"
        )
    if block_tokens is None:
        block_tokens = next(
            (b for b in (32, 16, 8) if chunk_size % b == 0), chunk_size
        )
    if chunk_size % block_tokens:
        raise ValueError(
            f"block_tokens ({block_tokens}) must divide chunk_size ({chunk_size})"
        )
    num_windows = topk // topk_cap
    num_chunks = num_tokens // chunk_size
    blocks_per_chunk = chunk_size // block_tokens
    windows, pages_per_seq = _build_token_page_windows(
        page_indices, cu_q_lens, num_tokens
    )
    padded_pps = windows.shape[1]

    def token_block(w: Any, c: Any, tb: Any, lw: Any) -> Any:
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
                pl.BlockSpec(
                    (block_tokens, padded_pps),
                    lambda w, c, tb, lw: (token_block(w, c, tb, lw), 0),
                ),
                pl.BlockSpec(
                    (block_tokens, topk_cap),
                    lambda w, c, tb, lw: (token_block(w, c, tb, lw), w),
                ),
            ],
            out_specs=pl.BlockSpec(
                (block_tokens, topk_cap),
                lambda w, c, tb, lw: (
                    (c * num_windows + w) * blocks_per_chunk + tb,
                    0,
                ),
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
    live_windows_ref: Any,  # i32[1] (scalar prefetch; sizes the grid)
    tab_ref: Any,  # i32[2 * G] (scalar prefetch) `live_group_table`
    windows_ref: Any,  # i32[bz, padded_pps] page-table row per token
    topk_ref: Any,  # i32[bz, topk_cap] one window of logical top-k
    out_hbm_ref: Any,  # i32[num_chunks * W * chunk * topk_cap // 128, 128] (HBM)
    stage_ref: Any,  # VMEM i32[2, chunk * topk_cap // 128, 128] window-0 regions
    # VMEM i32[topk_cap // 128 * bz, 128]: 128-slot tile j at rows [j*bz, (j+1)*bz).
    phys_ref: Any,
    tok_ref: Any,  # VMEM i32[bz * topk_cap // 128, 128] token-major rows (w >= 1)
    sems: Any,  # DMA[3]: window-0 region slots 0 / 1, token-major rows
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
) -> None:
    """Translates one ``(w, c, tb)`` tile into 128-lane linear live-group layout.

    Window 0 packs each group's live 64-slot atom blocks to the front
    (``off(g) + a``) and dead blocks from the back so the region remains a
    permutation of valid physical rows. Groups with all atoms live stay
    token-major at ``off(g)``. Windows ``w >= 1`` keep token-major order.
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

    def region_copy(chunk: Any, slot: Any) -> Any:
        return pltpu.make_async_copy(
            stage_ref.at[slot],
            out_hbm_ref.at[pl.ds(chunk * num_windows * region_rows, region_rows)],
            sems.at[slot],
        )

    @pl.when(w == 0)
    def _window0() -> None:
        slot = lax.rem(c, 2)

        @pl.when((tb == 0) & (c >= 2))
        def _reuse() -> None:
            region_copy(c - 2, slot).wait()

        # Strided loads need a 128-lane memref: stage column tiles row-stacked.
        for jj in range(lane_rows):
            phys_ref[pl.ds(jj * bz, bz), :] = phys[:, jj * 128 : (jj + 1) * 128]
        g = c * gpc + tb
        live = tab_ref[g]
        off = tab_ref[num_groups + g]
        dead_before = tb * atoms_w - off

        @pl.when(live == atoms_w)
        def _whole_group() -> None:
            # Every atom live: keep rows token-major for single-DMA fetches.
            for jj in range(lane_rows):
                tok_ref[pl.ds(jj, bz, stride=lane_rows), :] = phys[
                    :, jj * 128 : (jj + 1) * 128
                ]
            start = pl.multiple_of(off * tile_rows, tile_rows)
            stage_ref[slot, pl.ds(start, bz * lane_rows), :] = tok_ref[...]

        @pl.when(live < atoms_w)
        def _blocks() -> None:
            _pack_live_atom_blocks(
                phys_ref,
                stage_ref,
                slot,
                live=live,
                off=off,
                dead_before=dead_before,
                bz=bz,
                atoms_w=atoms_w,
                tile_rows=tile_rows,
                region_tiles=region_tiles,
                phys_row_stride=bz,
                group_row_offset=0,
            )

        @pl.when(tb == gpc - 1)
        def _flush() -> None:
            region_copy(c, slot).start()

            @pl.when(c == num_chunks - 1)
            def _drain() -> None:
                region_copy(c, slot).wait()
                if num_chunks >= 2:
                    region_copy(c - 1, 1 - slot).wait()

    @pl.when(w > 0)
    def _token_major() -> None:
        for jj in range(lane_rows):
            tok_ref[pl.ds(jj, bz, stride=lane_rows), :] = phys[
                :, jj * 128 : (jj + 1) * 128
            ]
        base = ((c * num_windows + w) * chunk_size + tb * bz) * lane_rows
        cp = pltpu.make_async_copy(
            tok_ref,
            out_hbm_ref.at[pl.ds(base, bz * lane_rows)],
            sems.at[_LIVE_TOKEN_MAJOR_SEM],
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
    """Translates capped windows with window 0 laid out in live-group order.

    Returns one linear ``i32[num_chunks * W * chunk * topk_cap // 128, 128]``
    array for the native gather's ``linear_indices`` operand. Windows past
    ``live_windows`` stay unwritten.
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
    windows, pages_per_seq = _build_token_page_windows(
        page_indices, cu_q_lens, num_tokens
    )
    padded_pps = windows.shape[1]
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
                    (bz, padded_pps),
                    lambda w, c, tb, lw, tab: (c * gpc + tb, 0),
                ),
                pl.BlockSpec(
                    (bz, topk_cap),
                    lambda w, c, tb, lw, tab: (c * gpc + tb, w),
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


def _capped_window_live_fused_kernel(
    *raw_refs: Any,
    ref_layout: RefLayout,
    bz: int,
    chunk_size: int,
    topk: int,
    topk_cap: int,
    page_size: int,
    pages_per_seq: int,
    total_num_pages: int,
    num_chunks: int,
    hooks: FaultHooks = NO_FAULT_HOOKS,
) -> None:
    """Computes ``kv_lens``, ``live_group_table``, and live-group window rows.

    Reads ``topk`` in its 128-lane row form and writes one output array per
    chunk. Windows at or past each chunk's live window bound stay unwritten.
    """
    refs = LiveFusedPrepRefs.from_bound(ref_layout.bind(*raw_refs))
    outs = refs.chunk_outs
    kv_ref, tab_ref, live_ref = refs.kv, refs.tab, refs.live
    stage_ref, phys_ref = refs.stage, refs.phys
    cw_ref, ab_ref, sems = refs.cw, refs.ab, refs.sems
    c = pl.program_id(0)
    gpc = chunk_size // bz
    atom = GATHER_ATOM_ROWS
    atoms_w = topk_cap // atom
    num_windows = topk // topk_cap
    lane_rows = topk_cap // 128
    row_stride = topk // 128
    tile_rows = bz * atom // 128
    region_rows = chunk_size * topk_cap // 128
    region_tiles = region_rows // tile_rows
    num_groups = num_chunks * gpc
    slot = lax.rem(c, 2)
    token_end = refs.token_end[0]

    def region_copy(chunk_out: Any, w: int, slot_: Any, sem: Any) -> Any:
        return pltpu.make_async_copy(
            stage_ref.at[slot_, pl.ds(w * region_rows, region_rows)],
            chunk_out.at[pl.ds(w * region_rows, region_rows)],
            sems.at[sem],
        )

    def wait_chunk(cc: Any, slot_: Any) -> None:
        # Waits only need the semaphore and byte count: any chunk output works.
        region_copy(outs[0], 0, slot_, slot_ * num_windows).wait()
        for w in range(1, num_windows):

            @pl.when(w < cw_ref[cc])
            def _(*, w: int = w) -> None:
                region_copy(outs[0], w, slot_, slot_ * num_windows + w).wait()

    @pl.when(c >= 2)
    def _reuse() -> None:
        wait_chunk(c - 2, slot)

    # 1. kv_lens: tokens on sublanes, one strided load per 128-slot column.
    acc = jnp.zeros((chunk_size, 128), jnp.float32)
    for j in range(row_stride):
        acc = acc + (
            refs.topk[pl.ds(j, chunk_size, stride=row_stride), :] != -1
        ).astype(jnp.float32)
    kv_col = jnp.sum(acc, axis=1, keepdims=True)  # f32[chunk, 1], exact
    for h in range(chunk_size // 128):
        blk = jnp.broadcast_to(kv_col[h * 128 : (h + 1) * 128, :], (128, 128))
        kv_row = jnp.transpose(blk)[0:1, :].astype(jnp.int32)
        kv_row = hooks.corrupt_live_prep_kv_row(kv_row)
        kv_ref[pl.ds(c * (chunk_size // 128) + h, 1), :] = kv_row
    kv_i = kv_col.astype(jnp.int32)

    # 2. Group table and chunk gather bound (matches `live_group_table`).
    chunk_start = c * chunk_size
    valid_rows = jnp.clip(token_end - chunk_start, 0, chunk_size)
    grid_rows = (valid_rows + bz - 1) // bz * bz
    local = lax.broadcasted_iota(jnp.int32, (chunk_size, 1), 0)
    token_windows = jnp.where(
        local < grid_rows,
        (jnp.maximum(kv_i, 1) + topk_cap - 1) // topk_cap,
        1,
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
        off = hooks.corrupt_live_prep_offset(off, tb)
    live_ref[c] = jnp.where(
        chunk_windows == 1,
        off * (bz * atom),
        (chunk_windows - 1) * chunk_size * topk_cap + grid_rows * topk_cap,
    )

    def translate(w: int, jj: int) -> jax.Array:
        # Column tile jj of window w for every token in the chunk.
        idx = refs.topk[pl.ds(w * lane_rows + jj, chunk_size, stride=row_stride), :]
        row = chunk_start + lax.broadcasted_iota(jnp.int32, idx.shape, 0)
        col = w * topk_cap + jj * 128 + lax.broadcasted_iota(jnp.int32, idx.shape, 1)
        return _window_physical_rows(
            idx,
            refs.windows,
            row * topk + col,
            page_size=page_size,
            pages_per_seq=pages_per_seq,
            total_num_pages=total_num_pages,
        )

    # 3a. Window 0 in live-group order; column tile jj of token t at jj * chunk + t.
    for jj in range(lane_rows):
        phys_ref[pl.ds(jj * chunk_size, chunk_size), :] = translate(0, jj)

    def group_body(tb: Any, carry: Any) -> Any:
        live = ab_ref[tb]
        goff = ab_ref[gpc + tb]
        dead_before = tb * atoms_w - goff
        g0 = pl.multiple_of(tb * bz, bz)

        @pl.when(live == atoms_w)
        def _whole_group() -> None:
            start = pl.multiple_of(goff * tile_rows, tile_rows)
            for jj in range(lane_rows):
                stage_ref[slot, pl.ds(start + jj, bz, stride=lane_rows), :] = (
                    phys_ref[pl.ds(jj * chunk_size + g0, bz), :]
                )

        @pl.when(live < atoms_w)
        def _blocks() -> None:
            _pack_live_atom_blocks(
                phys_ref,
                stage_ref,
                slot,
                live=live,
                off=goff,
                dead_before=dead_before,
                bz=bz,
                atoms_w=atoms_w,
                tile_rows=tile_rows,
                region_tiles=region_tiles,
                phys_row_stride=chunk_size,
                group_row_offset=g0,
            )

        return carry

    lax.fori_loop(0, gpc, group_body, 0)

    # 3b. Windows 1 .. W(c) - 1, token-major (overflowing chunks only).
    for w in range(1, hooks.live_prep_overflow_windows_end(num_windows)):

        @pl.when(w < chunk_windows)
        def _window(*, w: int = w) -> None:
            for jj in range(lane_rows):
                stage_ref[
                    slot,
                    pl.ds(w * region_rows + jj, chunk_size, stride=lane_rows),
                    :,
                ] = translate(w, jj)

    # 4. Flush this chunk's live windows to its dedicated output buffer.
    for cc in range(num_chunks):

        @pl.when(c == cc)
        def _flush(*, cc: int = cc) -> None:
            region_copy(outs[cc], 0, slot, slot * num_windows).start()
            for w in range(1, num_windows):

                @pl.when(w < chunk_windows)
                def _(*, w: int = w) -> None:
                    region_copy(outs[cc], w, slot, slot * num_windows + w).start()

    @pl.when(c == num_chunks - 1)
    def _drain() -> None:
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
    hooks: FaultHooks = NO_FAULT_HOOKS,
) -> tuple[tuple[jax.Array, ...], jax.Array, jax.Array, jax.Array]:
    """Fuses ``kv_lens``, ``live_group_table``, and live-group window prep.

    Reads ``topk_indices`` as 128-lane rows and writes one window array per
    chunk so neither the ``[T, topk]`` relayout nor per-chunk slice copies are
    needed.

    Returns:
        ``(rows, kv_lens, tab, num_live)`` where ``rows`` is a per-chunk tuple
        of ``i32[W * chunk * topk_cap // 128, 128]`` arrays, ``kv_lens`` is
        ``i32[num_tokens]``, and ``(tab, num_live)`` match ``live_group_table``.
    """
    if _fault is not None:
        if hooks != NO_FAULT_HOOKS:
            raise ValueError("cannot pass both _fault and hooks")
        hooks = FaultHooks.from_names(live_prep_fault=_fault)
    num_tokens, topk = topk_indices.shape
    bz = kernel_batch_size
    if not mixed_fused_prep_supported(num_tokens, topk, chunk_size, bz, topk_cap):
        raise ValueError(
            f"mixed fused prep unsupported: tokens {num_tokens}, topk {topk}, "
            f"chunk {chunk_size}, batch {bz}, cap {topk_cap}"
        )
    num_chunks = num_tokens // chunk_size
    windows, pages_per_seq = _build_token_page_windows(
        page_indices, cu_q_lens, num_tokens
    )
    registry = build_live_fused_prep_ref_registry(
        token_end=token_end,
        windows=windows,
        topk_indices=topk_indices,
        chunk_size=chunk_size,
        batch_size=bz,
        topk=topk,
        topk_cap=topk_cap,
    )
    kernel = functools.partial(
        _capped_window_live_fused_kernel,
        ref_layout=registry.layout,
        bz=bz,
        chunk_size=chunk_size,
        topk=topk,
        topk_cap=topk_cap,
        page_size=page_size,
        pages_per_seq=pages_per_seq,
        total_num_pages=total_num_pages,
        num_chunks=num_chunks,
        hooks=hooks,
    )
    outs = pl.pallas_call(
        kernel,
        grid_spec=registry.build_grid_spec((num_chunks,)),
        out_shape=registry.out_shape,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=True,
            vmem_limit_bytes=32 * 1024 * 1024,
        ),
        name="capped_window_indices_live_fused",
    )(*registry.call_args())
    rows = tuple(outs[:num_chunks])
    kv_lens, tab, num_live = outs[num_chunks:]
    return rows, kv_lens.reshape(num_tokens), tab, num_live


def _capped_window_fused_kernel(
    token_end_ref: Any,  # i32[1] (scalar prefetch) end of real tokens
    windows_ref: Any,  # i32[T, padded_pps] page-table row per token
    topk_ref: Any,  # i32[T, topk] logical top-k, -1 = empty slot
    rows_hbm_ref: Any,  # i32[W * T * topk_cap // 128 + META, 128] out
    rows_vmem_ref: Any,  # i32[W * T * topk_cap // 128 + META, 128] scratch
    sems: Any,  # DMA[W + 1]
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
) -> None:
    """One-call capped window prep for a single decode chunk.

    Counts each token's live top-k entries on the vector unit, derives the
    live-window bound and native SparseCore gather live vector, translates live
    windows to 128-lane linear physical rows, and appends the live vector in a
    trailing metadata tile so every output DMA runs concurrently.
    """
    num_tokens, topk = topk_ref.shape
    num_windows = topk // topk_cap
    lane_rows = topk_cap // 128
    window_rows = num_tokens * lane_rows
    meta_row = num_windows * window_rows

    def copy(start: int, size: int, sem: int) -> Any:
        return pltpu.make_async_copy(
            rows_vmem_ref.at[pl.ds(start, size)],
            rows_hbm_ref.at[pl.ds(start, size)],
            sems.at[sem],
        )

    def translate(w: int) -> None:
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
        for j in range(lane_rows):
            rows_vmem_ref[
                pl.ds(w * window_rows + j, num_tokens, stride=lane_rows), :
            ] = phys[:, j * 128 : (j + 1) * 128]
        copy(w * window_rows, window_rows, w).start()

    # Window 0 is always live: translate it before the bound is known.
    translate(0)

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
        def _wait(*, w: int = w) -> None:
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
    """Computes window indices and the gather live vector for one decode chunk.

    Replaces ``capped_live_windows``, ``prepare_capped_window_indices``, and
    live-count arithmetic for a single-chunk decode bucket.

    Returns:
        ``i32[W * num_tokens * topk_cap // 128 + FUSED_PREP_META_ROWS, 128]``
        holding window-major physical rows followed by the gather live vector
        at flat offset ``num_tokens * topk``.
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
    windows, pages_per_seq = _build_token_page_windows(
        page_indices, cu_q_lens, num_tokens
    )
    padded_pps = windows.shape[1]

    out_size = num_tokens * topk
    num_cores, num_subcores, _, occurrences_per_batch, num_batches = (
        native_sc_gather.live_vec_geometry(out_size, gather_atoms)
    )
    num_lanes = native_sc_gather.NUM_LANES
    if 2 * num_lanes > 128:
        raise ValueError(
            f"live vector ({2 * num_lanes}) must fit one 128-lane row"
        )
    total_rows = (
        num_windows * num_tokens * (topk_cap // 128) + FUSED_PREP_META_ROWS
    )
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


def live_group_table(
    kv_lens: jax.Array,  # i32[num_tokens]
    token_end: jax.Array,  # i32[]
    *,
    chunk_size: int,
    kernel_batch_size: int,
    topk_cap: int,
    num_windows: int,
) -> tuple[jax.Array, jax.Array]:
    """Computes per-group live atom counts, offsets, and per-chunk gather bounds.

    Group ``g`` has ``A(g) = cdiv(max kv_len in g, 64)`` clipped to the window's
    atoms (0 for groups the attention grid never reaches), and exclusive prefix
    offset ``off(g)`` within its chunk.

    Returns:
        ``(tab, num_live)`` where ``tab`` is ``i32[2 * G]`` (``A`` then ``off``)
        and ``num_live`` is ``i32[num_chunks]``.
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
    grid_rows, chunk_windows = _compute_chunk_window_bounds(
        kv_lens,
        token_end,
        chunk_size=chunk_size,
        kernel_batch_size=bz,
        topk_cap=topk_cap,
        num_windows=num_windows,
    )
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
    """Computes the maximum number of ``topk_cap`` windows needed across tokens.

    Args:
        kv_lens: Shape ``[num_tokens]`` valid KV counts per query token.
        token_end: Scalar marking the end of real tokens (padding starts here).
        chunk_size: Token chunk size processed per attention loop step.
        kernel_batch_size: Attention grid tile size in tokens.
        topk_cap: Capacity of one window in slots.
        num_windows: Maximum number of windows (``topk // topk_cap``).

    Returns:
        Scalar int32 in ``[1, num_windows]``.
    """
    num_tokens = kv_lens.shape[0]
    if num_tokens % chunk_size == 0:
        _, chunk_windows = _compute_chunk_window_bounds(
            kv_lens,
            token_end,
            chunk_size=chunk_size,
            kernel_batch_size=kernel_batch_size,
            topk_cap=topk_cap,
            num_windows=num_windows,
        )
        return jnp.max(chunk_windows).astype(jnp.int32)

    local = np.arange(num_tokens, dtype=np.int32) % chunk_size
    start = np.arange(num_tokens, dtype=np.int32) - local
    valid_rows = jnp.clip(token_end - start, 0, chunk_size)
    grid_rows = (valid_rows + kernel_batch_size - 1) // kernel_batch_size
    grid_rows = grid_rows * kernel_batch_size
    token_windows = jnp.where(
        local < grid_rows,
        (jnp.maximum(kv_lens, 1) + topk_cap - 1) // topk_cap,
        1,
    )
    return jnp.clip(jnp.max(token_windows), 1, num_windows).astype(jnp.int32)


def prepare_gather_indices(
    topk_indices: jax.Array,  # i32[max_num_tokens, csa_topk]
    page_indices: jax.Array | None,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    *,
    page_size: int,
    total_num_pages: int,
    is_dcp_mode: bool = False,
    num_valid_tokens: jax.Array | None = None,  # i32
) -> tuple[jax.Array, jax.Array]:
    """Per-token valid top-k counts and flat physical cache row indices to gather.

    Depends only on the indexer output and the page table, not on the query.
    """
    max_num_seqs = cu_q_lens.shape[0] - 1
    num_tokens, topk = topk_indices.shape
    kv_lens = jnp.sum(topk_indices != -1, axis=-1)
    if is_dcp_mode and num_valid_tokens is not None:
        # Padding tokens own no rows, so the kernel can skip them outright.
        kv_lens = jnp.where(
            jnp.arange(num_tokens, dtype=jnp.int32) < num_valid_tokens, kv_lens, 0
        )
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
        seq_ids_segment = _token_seq_ids(cu_q_lens, num_tokens)
        seq_page_ids = topk_indices // page_size
        token_offset = topk_indices % page_size
        page_ids = gather_page_ids(
            page_indices, seq_page_ids, seq_ids_segment, max_num_seqs
        )

        # Scatter "-1" padding across the cache to avoid read hot spots.
        scattered_page_ids, scattered_token_offset = scatter_padding_slots(
            flat_element_index, total_num_pages, page_size
        )
        page_ids = jnp.where(is_padding, scattered_page_ids, page_ids)
        token_offset = jnp.where(is_padding, scattered_token_offset, token_offset)
        physical_indices = page_ids * page_size + token_offset
    else:
        scattered_page_ids, scattered_token_offset = scatter_padding_slots(
            flat_element_index, total_num_pages, page_size
        )
        scattered_physical = scattered_page_ids * page_size + scattered_token_offset
        physical_indices = jnp.where(is_padding, scattered_physical, topk_indices)

    physical_indices = physical_indices.astype(jnp.int32)
    return kv_lens, physical_indices
