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
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.mla import kv_cache_utils
from vllm_torchtpu.kernels.mla.sparse import dsa_gather, native_sc_gather
from vllm_torchtpu.utils import largest_divisor

DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024


def _default_vmem_limit_bytes() -> int:
    if jax.default_backend() != "tpu":
        return 64 * 1024 * 1024
    cap = pltpu.get_tpu_info().vmem_capacity_bytes
    # Allow Mosaic >64 MiB compilation headroom before spilling live tiles to HBM.
    return DEFAULT_VMEM_LIMIT_BYTES if cap >= 64 * 1024 * 1024 else cap


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
      a remainder refinement and a +/-1 clamp to guarantee bit-exact results.

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


def _token_seq_ids(cu_q_lens: jax.Array, num_tokens: int) -> jax.Array:
    max_num_seqs = cu_q_lens.shape[0] - 1
    tokens_per_seq = cu_q_lens[1:] - cu_q_lens[:-1]
    seq_iota = jnp.arange(max_num_seqs, dtype=jnp.int32)
    seg = jnp.repeat(seq_iota, tokens_per_seq, total_repeat_length=num_tokens)
    last_active = jnp.max(jnp.where(tokens_per_seq > 0, seq_iota, 0))
    return jnp.where(
        jnp.arange(num_tokens, dtype=jnp.int32) < cu_q_lens[-1], seg, last_active
    )


_FMIX32_C1 = np.uint32(0x85EBCA6B).view(np.int32).item()
_FMIX32_C2 = np.uint32(0xC2B2AE35).view(np.int32).item()


def scatter_padding_slots(
    flat_element_index: jax.Array, total_num_pages: int, page_size: int
) -> tuple[jax.Array, jax.Array]:
    """Hash flat slot indices to spread-out `(page, offset)` for `-1` padding."""
    h = flat_element_index.astype(jnp.int32)
    h ^= lax.shift_right_logical(h, jnp.int32(16))
    h *= jnp.int32(_FMIX32_C1)
    h ^= lax.shift_right_logical(h, jnp.int32(13))
    h *= jnp.int32(_FMIX32_C2)
    h ^= lax.shift_right_logical(h, jnp.int32(16))
    _, page = _floor_divmod_const(
        lax.shift_right_logical(h, jnp.int32(9)), total_num_pages
    )
    offset = (
        h & jnp.int32(page_size - 1)
        if page_size & (page_size - 1) == 0
        else _floor_divmod_const(h & jnp.int32((1 << 23) - 1), page_size)[1]
    )
    return page, offset


def _padded_token_page_table(
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    num_tokens: int,
) -> tuple[jax.Array, int, int]:
    max_num_seqs = cu_q_lens.shape[0] - 1
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    padded_pps = align_to(pages_per_seq, _GATHER_PAGE_CHUNK)
    page_table = page_indices.reshape(max_num_seqs, pages_per_seq)
    if padded_pps != pages_per_seq:
        page_table = jnp.pad(page_table, ((0, 0), (0, padded_pps - pages_per_seq)))
    return page_table[_token_seq_ids(cu_q_lens, num_tokens)], pages_per_seq, padded_pps


def _window_physical_rows(
    idx: jax.Array,
    windows_ref,
    flat_element_index: jax.Array,
    *,
    page_size: int,
    pages_per_seq: int,
    total_num_pages: int,
) -> jax.Array:
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
    scattered_page, scattered_offset = scatter_padding_slots(
        flat_element_index, total_num_pages, page_size
    )
    page = jnp.where(is_padding, scattered_page, page)
    token_offset = jnp.where(is_padding, scattered_offset, token_offset)
    return page * page_size + token_offset


def _scatter_live_group_atoms(
    phys_ref,
    stage_ref,
    slot,
    *,
    g0,
    phys_stride: int,
    live,
    goff,
    tb,
    atoms_w: int,
    bz: int,
    lane_rows: int,
    tile_rows: int,
    region_tiles: int,
) -> None:
    atom = GATHER_ATOM_ROWS
    dead_before = tb * atoms_w - goff

    @pl.when(live == atoms_w)
    def _whole_group():
        start = pl.multiple_of(goff * tile_rows, tile_rows)
        for jj in range(lane_rows):
            stage_ref[slot, pl.ds(start + jj, bz, stride=lane_rows), :] = phys_ref[
                pl.ds(jj * phys_stride + g0, bz), :
            ]

    @pl.when(live < atoms_w)
    def _blocks():
        lane = lax.broadcasted_iota(jnp.int32, (bz // 2, 128), 1)
        for a in range(atoms_w):
            j, hh = divmod(a * atom, 128)
            even = phys_ref[pl.ds(j * phys_stride + g0, bz // 2, stride=2), :]
            odd = phys_ref[pl.ds(j * phys_stride + g0 + 1, bz // 2, stride=2), :]
            tile = (
                jnp.where(lane < atom, even, pltpu.roll(odd, atom, 1))
                if hh == 0
                else jnp.where(lane < atom, pltpu.roll(even, atom, 1), odd)
            )
            pos = jnp.where(
                a < live, goff + a, region_tiles - 1 - (dead_before + a - live)
            )
            start = pl.multiple_of(pos * tile_rows, tile_rows)
            stage_ref[slot, pl.ds(start, tile_rows), :] = tile


def _translate_window_tile(
    idx: jax.Array,
    windows_ref,
    row0,
    col0,
    *,
    topk: int,
    page_size: int,
    pages_per_seq: int,
    total_num_pages: int,
) -> jax.Array:
    row = row0 + lax.broadcasted_iota(jnp.int32, idx.shape, 0)
    col = col0 + lax.broadcasted_iota(jnp.int32, idx.shape, 1)
    return _window_physical_rows(
        idx,
        windows_ref,
        row * topk + col,
        page_size=page_size,
        pages_per_seq=pages_per_seq,
        total_num_pages=total_num_pages,
    )


def _capped_window_indices_kernel(
    live_windows_ref,
    windows_ref,
    topk_ref,
    out_ref,
    *,
    block_tokens: int,
    chunk_size: int,
    topk: int,
    topk_cap: int,
    page_size: int,
    pages_per_seq: int,
    total_num_pages: int,
):
    del live_windows_ref
    w, c, tb = pl.program_id(0), pl.program_id(1), pl.program_id(2)
    out_ref[...] = _translate_window_tile(
        topk_ref[...],
        windows_ref,
        c * chunk_size + tb * block_tokens,
        w * topk_cap,
        topk=topk,
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
    windows, pages_per_seq, padded_pps = _padded_token_page_table(
        page_indices, cu_q_lens, num_tokens
    )
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
                    lambda w, c, tb, lw: (c * blocks_per_chunk + tb, 0),
                ),
                # Spec 1: Top-k indices for this token block
                pl.BlockSpec(
                    (block_tokens, topk_cap),
                    lambda w, c, tb, lw: (c * blocks_per_chunk + tb, w),
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


def mixed_fused_prep_supported(
    num_tokens: int, topk: int, chunk_size: int, kernel_batch_size: int, topk_cap: int
) -> bool:
    """Whether `prepare_capped_window_live_fused` serves a mixed bucket."""
    return (
        live_group_supported(chunk_size, kernel_batch_size, topk_cap)
        and num_tokens % chunk_size == 0
        and chunk_size % 128 == 0
        and topk % topk_cap == 0
        and topk % 128 == 0
    )


def _capped_window_live_fused_kernel(
    token_end_ref,
    windows_ref,
    topk_ref,
    *refs,
    bz: int,
    chunk_size: int,
    topk: int,
    topk_cap: int,
    page_size: int,
    pages_per_seq: int,
    total_num_pages: int,
    num_chunks: int,
):
    outs = refs[:num_chunks]
    kv_ref, tab_ref, live_ref = refs[num_chunks : num_chunks + 3]
    stage_ref, phys_ref, cw_ref, ab_ref, sems = refs[num_chunks + 3 :]
    c = pl.program_id(0)
    gpc, atom = chunk_size // bz, GATHER_ATOM_ROWS
    atoms_w, num_windows = topk_cap // atom, topk // topk_cap
    lane_rows, row_stride = topk_cap // 128, topk // 128
    tile_rows = bz * atom // 128
    region_rows = chunk_size * lane_rows
    region_tiles, num_groups = region_rows // tile_rows, num_chunks * gpc
    slot, token_end, chunk_start = lax.rem(c, 2), token_end_ref[0], c * chunk_size

    def region_copy(chunk_out, w, slot_, sem):
        return pltpu.make_async_copy(
            stage_ref.at[slot_, pl.ds(w * region_rows, region_rows)],
            chunk_out.at[pl.ds(w * region_rows, region_rows)],
            sems.at[sem],
        )

    def wait_chunk(cc, slot_):
        region_copy(outs[0], 0, slot_, slot_ * num_windows).wait()
        for w in range(1, num_windows):
            pl.when(w < cw_ref[cc])(
                lambda w=w: region_copy(
                    outs[0], w, slot_, slot_ * num_windows + w
                ).wait()
            )

    pl.when(c >= 2)(lambda: wait_chunk(c - 2, slot))

    acc = jnp.zeros((chunk_size, 128), jnp.float32)
    for j in range(row_stride):
        acc += (topk_ref[pl.ds(j, chunk_size, stride=row_stride), :] != -1).astype(
            jnp.float32
        )
    kv_col = jnp.sum(acc, axis=1, keepdims=True)
    for h in range(chunk_size // 128):
        blk = jnp.broadcast_to(kv_col[h * 128 : (h + 1) * 128, :], (128, 128))
        kv_ref[pl.ds(c * (chunk_size // 128) + h, 1), :] = jnp.transpose(blk)[
            0:1, :
        ].astype(jnp.int32)
    kv_i = kv_col.astype(jnp.int32)

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
        a_live = jnp.where(
            chunk_start + tb * bz < token_end,
            jnp.minimum((kv_max + atom - 1) // atom, atoms_w),
            0,
        )
        g = c * gpc + tb
        tab_ref[g], tab_ref[num_groups + g] = a_live, off
        ab_ref[tb], ab_ref[gpc + tb] = a_live, off
        off = off + a_live
    live_ref[c] = jnp.where(
        chunk_windows == 1,
        off * (bz * atom),
        (chunk_windows - 1) * chunk_size * topk_cap + grid_rows * topk_cap,
    )

    def translate(w, jj):
        return _translate_window_tile(
            topk_ref[pl.ds(w * lane_rows + jj, chunk_size, stride=row_stride), :],
            windows_ref,
            chunk_start,
            w * topk_cap + jj * 128,
            topk=topk,
            page_size=page_size,
            pages_per_seq=pages_per_seq,
            total_num_pages=total_num_pages,
        )

    for jj in range(lane_rows):
        phys_ref[pl.ds(jj * chunk_size, chunk_size), :] = translate(0, jj)

    def group_body(tb, carry):
        _scatter_live_group_atoms(
            phys_ref,
            stage_ref,
            slot,
            g0=pl.multiple_of(tb * bz, bz),
            phys_stride=chunk_size,
            live=ab_ref[tb],
            goff=ab_ref[gpc + tb],
            tb=tb,
            atoms_w=atoms_w,
            bz=bz,
            lane_rows=lane_rows,
            tile_rows=tile_rows,
            region_tiles=region_tiles,
        )
        return carry

    lax.fori_loop(0, gpc, group_body, 0)

    for w in range(1, num_windows):

        @pl.when(w < chunk_windows)
        def _window(*, w=w):
            for jj in range(lane_rows):
                stage_ref[
                    slot, pl.ds(w * region_rows + jj, chunk_size, stride=lane_rows), :
                ] = translate(w, jj)

    for cc in range(num_chunks):

        @pl.when(c == cc)
        def _flush(*, cc=cc):
            region_copy(outs[cc], 0, slot, slot * num_windows).start()
            for w in range(1, num_windows):
                pl.when(w < chunk_windows)(
                    lambda cc=cc, w=w: region_copy(
                        outs[cc], w, slot, slot * num_windows + w
                    ).start()
                )

    @pl.when(c == num_chunks - 1)
    def _drain():
        wait_chunk(c, slot)
        if num_chunks >= 2:
            wait_chunk(c - 1, 1 - slot)


def prepare_capped_window_live_fused(
    topk_indices: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    token_end: jax.Array,
    *,
    chunk_size: int,
    kernel_batch_size: int,
    topk_cap: int,
    page_size: int,
    total_num_pages: int,
) -> tuple[tuple[jax.Array, ...], jax.Array, jax.Array, jax.Array]:
    """Fused `kv_lens` + live-group table + window-0 live-atom / window-major rows."""
    num_tokens, topk = topk_indices.shape
    bz = kernel_batch_size
    if not mixed_fused_prep_supported(num_tokens, topk, chunk_size, bz, topk_cap):
        cfg = (num_tokens, topk, chunk_size, bz, topk_cap)
        raise ValueError(f"mixed fused prep unsupported: {cfg}")
    num_windows, num_chunks, gpc = (
        topk // topk_cap,
        num_tokens // chunk_size,
        chunk_size // bz,
    )
    windows, pages_per_seq, padded_pps = _padded_token_page_table(
        page_indices, cu_q_lens, num_tokens
    )
    region_rows, row_stride = chunk_size * topk_cap // 128, topk // 128
    hbm, smem = (
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(memory_space=pltpu.SMEM),
    )
    chunk_sd = jax.ShapeDtypeStruct((num_windows * region_rows, 128), jnp.int32)
    outs = pl.pallas_call(
        functools.partial(
            _capped_window_live_fused_kernel,
            bz=bz,
            chunk_size=chunk_size,
            topk=topk,
            topk_cap=topk_cap,
            page_size=page_size,
            pages_per_seq=pages_per_seq,
            total_num_pages=total_num_pages,
            num_chunks=num_chunks,
        ),
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
            *([chunk_sd] * num_chunks),
            jax.ShapeDtypeStruct((num_tokens // 128, 128), jnp.int32),
            jax.ShapeDtypeStruct((2 * num_tokens // bz,), jnp.int32),
            jax.ShapeDtypeStruct((num_chunks,), jnp.int32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=True,
            vmem_limit_bytes=min(32 * 1024 * 1024, _default_vmem_limit_bytes()),
        ),
        name="capped_window_indices_live_fused",
    )(
        jnp.reshape(token_end, (1,)).astype(jnp.int32),
        windows,
        topk_indices.reshape(num_tokens * row_stride, 128),
    )
    kv_lens, tab, num_live = outs[num_chunks:]
    return tuple(outs[:num_chunks]), kv_lens.reshape(num_tokens), tab, num_live


FUSED_PREP_MAX_TOKENS = 32
FUSED_PREP_META_ROWS = 8


def _cdiv_const(x: jax.Array, d: int) -> jax.Array:
    if d & (d - 1) == 0:
        return lax.shift_right_logical(x + (d - 1), jnp.int32(d.bit_length() - 1))
    return _floor_divmod_const(x + (d - 1), d)[0]


def _capped_window_fused_kernel(
    token_end_ref,
    windows_ref,
    topk_ref,
    rows_hbm_ref,
    rows_vmem_ref,
    sems,
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
    num_tokens, topk = topk_ref.shape
    num_windows, lane_rows = topk // topk_cap, topk_cap // 128
    window_rows = num_tokens * lane_rows
    meta_row = num_windows * window_rows

    def copy(start, size, sem):
        return pltpu.make_async_copy(
            rows_vmem_ref.at[pl.ds(start, size)],
            rows_hbm_ref.at[pl.ds(start, size)],
            sems.at[sem],
        )

    def translate(w):
        phys = _translate_window_tile(
            topk_ref[:, w * topk_cap : (w + 1) * topk_cap],
            windows_ref,
            0,
            w * topk_cap,
            topk=topk,
            page_size=page_size,
            pages_per_seq=pages_per_seq,
            total_num_pages=total_num_pages,
        )
        for j in range(lane_rows):
            rows_vmem_ref[
                pl.ds(w * window_rows + j, num_tokens, stride=lane_rows), :
            ] = phys[:, j * 128 : (j + 1) * 128]
        copy(w * window_rows, window_rows, w).start()

    translate(0)

    grid_rows = (
        cdiv(jnp.clip(token_end_ref[0], 0, num_tokens), kernel_batch_size)
        * kernel_batch_size
    )
    kv = jnp.sum(
        (topk_ref[...] != -1).astype(jnp.float32), axis=1, keepdims=True
    ).astype(jnp.int32)
    token_windows = jnp.where(
        lax.broadcasted_iota(jnp.int32, kv.shape, 0) < grid_rows,
        _cdiv_const(jnp.maximum(kv, 1), topk_cap),
        1,
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
        pl.when(w < live_windows)(
            lambda w=w: copy(w * window_rows, window_rows, w).wait()
        )
    meta_copy.wait()


def prepare_capped_window_fused(
    topk_indices: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    token_end: jax.Array,
    *,
    topk_cap: int,
    page_size: int,
    total_num_pages: int,
    kernel_batch_size: int,
    gather_atoms: int,
) -> jax.Array:
    """Build window-major 128-lane physical rows + trailing gather live vector."""
    num_tokens, topk = topk_indices.shape
    if num_tokens > FUSED_PREP_MAX_TOKENS:
        raise ValueError(f"fused prep max {FUSED_PREP_MAX_TOKENS}, got {num_tokens}")
    if topk % topk_cap or topk_cap % 128:
        raise ValueError(
            f"topk_cap ({topk_cap}) must be a multiple of 128 dividing topk ({topk})"
        )
    num_windows = topk // topk_cap
    windows, pages_per_seq, padded_pps = _padded_token_page_table(
        page_indices, cu_q_lens, num_tokens
    )
    num_cores, num_subcores, _, occurrences_per_batch, num_batches = (
        native_sc_gather.live_vec_geometry(num_tokens * topk, gather_atoms)
    )
    num_lanes = native_sc_gather.NUM_LANES
    if 2 * num_lanes > 128:
        raise ValueError(f"live vector ({2 * num_lanes}) must fit one 128-lane row")
    total_rows = num_windows * num_tokens * (topk_cap // 128) + FUSED_PREP_META_ROWS
    return pl.pallas_call(
        functools.partial(
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
        ),
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
            dimension_semantics=("arbitrary",), disable_bounds_checks=True
        ),
        name="capped_window_indices_fused",
    )(jnp.reshape(token_end, (1,)).astype(jnp.int32), windows, topk_indices)


def live_group_supported(
    chunk_size: int, kernel_batch_size: int, topk_cap: int
) -> bool:
    """Whether live-group window-0 layout supports this bucket geometry."""
    atom = GATHER_ATOM_ROWS
    tile = kernel_batch_size * atom
    return (
        chunk_size % kernel_batch_size == 0
        and tile % (8 * 128) == 0
        and topk_cap % 128 == 0
        and topk_cap % atom == 0
        and tile % topk_cap == 0
    )


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


Q_ROPE_DIM = 64
Q_ROPE_HEAD_DIM = 256


def rope_q_block(x: jax.Array, tab: jax.Array) -> jax.Array:
    """NeoX RoPE of the `q_pe` lanes `[64:128]` of one raw head block `[heads, 128]`."""
    xf = x.astype(jnp.float32)
    lane = lax.broadcasted_iota(jnp.int32, xf.shape, 1)
    rot = jnp.where(lane < 96, pltpu.roll(xf, 96, 1), pltpu.roll(xf, 32, 1))
    c = tab[0:1, :]
    s = tab[1:2, :]
    a = (xf * c).astype(x.dtype).astype(jnp.float32)
    b = (rot * s).astype(x.dtype).astype(jnp.float32)
    r = pltpu.roll(a + b, 64, 1)
    return jnp.where(lane < 64, r, 0.0).astype(x.dtype)


MIXED_BODY_LEVERS = ("tr", "zu", "pv512", "noinf", "fxskip", "pair")
MIXED_BODY_DEFAULT = ("tr", "zu", "pv512", "noinf")
DECODE_BODY_LEVERS = (
    "tr",
    "zu",
    "pv512",
    "noinf",
    "pair",
    "fxskip",
    "quad",
    "lrows",
    "step8",
)
DECODE_STEP8_TOKENS = 32
DECODE_BODY_DEFAULT = ("tr", "zu", "pv512", "noinf", "quad", "fxskip", "lrows", "step8")
MIXED_KERNEL_V2_LEVERS = ("fxskip", "pair")
MIXED_BODY_MAX_SKIP_TOKENS = 64

SEND_PAD_ROWS = 16
SEND_SLAB_ROW0 = SEND_PAD_ROWS // 2
SEND_EXCHANGE_COLLECTIVE_ID = 13
SEND_EXCHANGE_VMEM_LIMIT_BYTES = 16 * 1024 * 1024


def send_layout_lanes(num_tokens: int, num_heads: int, world: int) -> int:
    """Lane width of the in-kernel LSE word build."""
    rows = num_tokens * (num_heads // world)
    return align_to(max(2 * rows, rows + num_heads), 128)


def send_layout_supported(
    num_tokens: int, num_heads: int, world: int, vdim: int, batch_size: int
) -> bool:
    """Whether the kernel can write the send buffer for this geometry."""
    if world < 2 or num_heads % world or vdim % 128:
        return False
    per = num_heads // world
    rows = num_tokens * per
    lanes = send_layout_lanes(num_tokens, num_heads, world)
    return (
        per & (per - 1) == 0
        and rows & (rows - 1) == 0
        and lanes & (lanes - 1) == 0
        and 2 * rows <= vdim
        and lanes <= vdim
        and (world - 1) * per < 128
        and num_tokens % batch_size == 0
        and (batch_size * per) % SEND_PAD_ROWS == 0
    )


def send_layout_shape(
    num_tokens: int, num_heads: int, world: int, vdim: int
) -> tuple[int, int, int]:
    """Shape of one rank's int16 send buffer."""
    return (world, SEND_PAD_ROWS + num_tokens * (num_heads // world), vdim)


def pack_send_lse_words(
    x_lse: jax.Array,
    *,
    num_tokens: int,
    num_q_heads: int,
    send_world: int,
    send_lanes: int,
    fault: str | None = None,
) -> jax.Array:
    """Pack block-reversed f32 `[T, lanes]` LSE rows into i32 `[W, lanes]` words.

    Output lanes `[0, send_rows)` carry signed `hi16` and `[send_rows, 2*send_rows)`
    carry signed `lo16`, where `send_rows = num_tokens * (num_q_heads // send_world)`.
    Trailing lanes `[2*send_rows, send_lanes)` are zero.
    """
    send_per = num_q_heads // send_world
    send_rows = num_tokens * send_per
    bits = lax.bitcast_convert_type(x_lse, jnp.int32)
    hi = lax.shift_right_arithmetic(bits, jnp.int32(16))
    # Signed 16-bit representation of the low halfword for int16 packing.
    lo = ((bits & 0xFFFF) ^ 0x8000) - 0x8000
    # Sublane d is rolled right by d * send_per at the end (one strided roll);
    # lane j of sublane d therefore maps to destination lane
    # big_j = (j + d * send_per) mod send_lanes.
    shape = (send_world, send_lanes)
    big_j = (
        lax.broadcasted_iota(jnp.int32, shape, 1)
        + lax.broadcasted_iota(jnp.int32, shape, 0) * send_per
    ) & (send_lanes - 1)
    tok = lax.shift_right_logical(
        big_j & (send_rows - 1), jnp.int32(send_per.bit_length() - 1)
    )
    tok = jnp.where(big_j < 2 * send_rows, tok, -1)
    acc = jnp.zeros(shape, jnp.int32)
    if send_rows >= num_q_heads:
        lane = lax.broadcasted_iota(jnp.int32, x_lse.shape, 1)
        lo_src = pltpu.roll(lo, send_rows, 1)
        if fault == "send_lse_lo_skip":
            lo_src = jnp.zeros_like(lo_src)
        src = jnp.where(
            lane < num_q_heads,
            hi,
            jnp.where(
                (lane >= send_rows) & (lane < send_rows + num_q_heads), lo_src, 0
            ),
        )
        for t in range(num_tokens):
            shift = (t * send_per - (send_world - 1) * send_per) % send_lanes
            row = src[t : t + 1, :]
            if shift:
                row = pltpu.roll(row, shift, 1)
            acc = jnp.where(tok == t, jnp.broadcast_to(row, shape), acc)
    else:
        # Avoid packing hi and lo into one row when num_tokens < send_world:
        # [send_rows, send_rows + num_q_heads) overlaps [0, num_q_heads), so
        # hi would overwrite lo halfwords for ranks with overlapping lanes.
        if fault == "send_lse_lo_skip":
            lo = jnp.zeros_like(lo)
        is_lo = big_j >= send_rows
        for t in range(num_tokens):
            hi_shift = (t * send_per - (send_world - 1) * send_per) % send_lanes
            lo_shift = (
                send_rows + t * send_per - (send_world - 1) * send_per
            ) % send_lanes
            hi_row = pltpu.roll(hi[t : t + 1, :], hi_shift, 1)
            lo_row = lo[t : t + 1, :]
            if lo_shift:
                lo_row = pltpu.roll(lo_row, lo_shift, 1)
            row = jnp.where(
                is_lo,
                jnp.broadcast_to(lo_row, shape),
                jnp.broadcast_to(hi_row, shape),
            )
            acc = jnp.where(tok == t, row, acc)
    return pltpu.roll(acc, 0, 1, stride=send_per, stride_axis=0)


def xchg_live_row_tiers(
    slab_row0: int, send_per: int, num_tokens: int, xchg_rows: int
) -> tuple[int, ...]:
    if num_tokens <= 4:
        token_breaks = (1, 2, 3)
    elif num_tokens <= 8:
        token_breaks = (2, 4, 6)
    else:
        token_breaks = (
            max(1, num_tokens // 8),
            max(1, num_tokens // 4),
            max(1, num_tokens // 2),
            max(1, (3 * num_tokens) // 4),
        )
    tiers = sorted(
        {
            min(xchg_rows, align_to(slab_row0 + tok * send_per, 8))
            for tok in token_breaks
        }
        | {xchg_rows}
    )
    return tuple(tiers)


def xchg_peer_order(me: Any, world: int) -> list[Any]:
    # Avoid lockstep j=0..world-1 receiver incast; on world==16 launch the
    # opposite 8-rank half first so inter-host transfers overlap local sends.
    if world == 16:
        low = me & 7
        remote_hi = (me ^ 8) & 8
        local_hi = me & 8
        return [(((low + s) & 7) | remote_hi) for s in range(8)] + [
            (((low + s) & 7) | local_hi) for s in range(1, 8)
        ]
    if isinstance(me, int):
        return [(me + s) % world for s in range(1, world)]
    return [lax.rem(me + jnp.int32(s), jnp.int32(world)) for s in range(1, world)]


def merge_received_slabs(
    recv_v_ref: Any,
    *,
    send_world: int,
    send_rows: int,
    slab_row0: int,
    o_dim: int,
    live_rows: jax.Array,
) -> jax.Array:
    """Unpack and LSE-merge received slabs into bf16 ``[send_rows, o_dim]``."""
    n, tl = send_world, send_rows
    assert tl <= 128 and 2 * tl <= o_dim, (tl, o_dim)
    hdr = SEND_PAD_ROWS
    r128 = lax.broadcasted_iota(jnp.int32, (tl, 128), 0)
    c128 = lax.broadcasted_iota(jnp.int32, (tl, 128), 1)
    l_cols = []
    for w in range(n):
        # int16 words to int32 lanes (sign-extended): row slab_row0 - 1
        # of the slab is the lse word row.
        head = recv_v_ref[w, pl.ds(0, hdr), :].astype(jnp.int32)
        words = head[slab_row0 - 1 : slab_row0]  # [1, o_dim]
        lo_src = pltpu.roll(words, o_dim - tl, 1)  # lane r <- lane tl + r
        bits = lax.shift_left(words, jnp.int32(16)) | (lo_src & 0xFFFF)
        l_row = lax.bitcast_convert_type(bits, jnp.float32)[:, :128]
        l_cols.append(
            jnp.max(
                jnp.where(r128 == c128, jnp.broadcast_to(l_row, (tl, 128)), -jnp.inf),
                axis=1,
                keepdims=True,
            )
        )
    finite = [jnp.isfinite(c) for c in l_cols]
    shift = jnp.where(finite[0], l_cols[0], -jnp.inf)
    for w in range(1, n):
        shift = jnp.maximum(shift, jnp.where(finite[w], l_cols[w], -jnp.inf))
    shift = jnp.where(jnp.isfinite(shift), shift, 0.0)
    weights = [jnp.where(finite[w], jnp.exp(l_cols[w] - shift), 0.0) for w in range(n)]
    denom = functools.reduce(jnp.add, weights)
    denom = jnp.where(denom > 0, denom, 1.0)
    rows_i = lax.broadcasted_iota(jnp.int32, (tl, o_dim), 0)
    acc_hi = jnp.zeros((tl, o_dim), jnp.float32)
    acc_lo = jnp.zeros((tl, o_dim), jnp.float32)
    for w in range(n):
        wt = weights[w] / denom
        w_hi = wt.astype(jnp.bfloat16).astype(jnp.float32)
        xi = recv_v_ref[w].astype(jnp.int32)
        part = lax.bitcast_convert_type(
            lax.shift_left(xi[slab_row0 : slab_row0 + tl], jnp.int32(16)),
            jnp.float32,
        )
        part = jnp.where((rows_i < live_rows) & ~jnp.isnan(part), part, 0.0)
        acc_hi += part * w_hi
        acc_lo += part * (wt - w_hi)
    return (acc_hi + acc_lo).astype(jnp.bfloat16)


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
    is_dcp_mode: bool = False,
    split_q: bool = False,
    lse_lane_dense: bool = False,
    lse_mask_empty: bool = False,
    group_tab_ref=None,
    live_widths: tuple[int, ...] = (),
    body_levers: tuple[str, ...] = (),
    q_rope: bool = False,
    lse_fill: bool = False,
    row_widths: tuple[int, ...] = (),
    send_world: int = 0,
    out_row_offset: int = 0,
    out_in: bool = False,
    send_exchange: tuple | None = None,
    recv_in: bool = False,
    head_major_out: bool = False,
):
    bo_hm_x2_ref = None
    if head_major_out:
        assert (
            not return_lse
            and send_world == 0
            and not preserve_suffix
            and capped_windows is None
        )
        bo_hm_x2_ref = refs[-1]  # [2, num_q_heads, bsz, head_dim]
        refs = refs[:-1]
    xchg_merge = bool(
        send_exchange is not None and len(send_exchange) > 4 and send_exchange[4]
    )
    recv_v_ref: Any = None
    if xchg_merge:
        recv_v_ref, refs = refs[-1], refs[:-1]
    kv_stage_ref: Any = None
    if "pair" in body_levers or "quad" in body_levers:
        kv_stage_ref, refs = refs[-1], refs[:-1]
    q_pe_hbm_ref = bq_pe_x2_ref = tab_hbm_ref = bq_tab_x2_ref = cast(Any, None)
    if split_q:
        q_pe_hbm_ref, bq_pe_x2_ref, refs = refs[0], refs[-1], refs[1:-1]
        if q_rope:
            tab_hbm_ref, bq_tab_x2_ref, refs = refs[0], refs[-1], refs[1:-1]
    assert not q_rope or split_q
    send, xchg = send_world > 0, send_exchange is not None
    assert (not xchg or send) and (not recv_in or xchg)
    fill_ref = fill_sem = send_wbuf = xchg_sems = cast(Any, None)
    if lse_fill:
        assert return_lse and lse_lane_dense
        if send and xchg:
            fill_ref, xchg_sems, fill_sem, refs = (
                refs[-3],
                refs[-2],
                refs[-1],
                refs[:-3],
            )
        elif send:
            fill_ref, send_wbuf, fill_sem, refs = (
                refs[-3],
                refs[-2],
                refs[-1],
                refs[:-3],
            )
        else:
            fill_ref, fill_sem, refs = refs[-2], refs[-1], refs[:-2]
    assert not send or (lse_fill and split_q and not preserve_suffix)
    live_nope_hbm_ref = live_rope_hbm_ref = None
    if group_tab_ref is not None:
        assert capped_windows is not None
        live_nope_hbm_ref, live_rope_hbm_ref, refs = refs[0], refs[1], refs[2:]
    assert (not live_widths or group_tab_ref is not None) and (
        not row_widths or group_tab_ref is None
    )
    assert (
        not body_levers
        or group_tab_ref is not None
        or row_widths
        or (sparsecore_layout and capped_windows is None)
    )
    assert set(body_levers) <= set(MIXED_BODY_LEVERS) | {"quad", "lrows", "step8"}
    assert not ("quad" in body_levers and "pair" in body_levers)
    fix_refs = None
    if capped_windows is not None:
        assert sparsecore_layout
        fix_refs = refs[-6:]
        refs = refs[:-6]
    if lse_fill and not recv_in:
        refs = (cast(Any, None), *refs)
    if out_in:
        assert return_lse and not lse_fill
        refs = (refs[0], *refs[2:])
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
    if head_major_out:
        assert (
            o_hbm_ref.shape[0] == q_hbm_ref.shape[1]
            and o_hbm_ref.shape[2] == q_hbm_ref.shape[2]
        ), (o_hbm_ref.shape, q_hbm_ref.shape)
    else:
        assert q_hbm_ref.shape[1:] == o_hbm_ref.shape[1:]

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
    if send:
        assert send_layout_supported(
            num_tokens, num_q_heads, send_world, o_dim, batch_size
        ), (num_tokens, num_q_heads, send_world, o_dim, batch_size)
        send_per = num_q_heads // send_world
        send_rows = num_tokens * send_per
        send_lanes = send_layout_lanes(num_tokens, num_q_heads, send_world)
    # First partial row of a destination's slab in the bo tile.
    # Avoid sending 7 empty header rows in private-VMEM xchg_merge (m=4):
    # row 3 holds LSE words and rows 4..7 hold token 0, fitting B=1 in 8 rows.
    compact_merge_slab = (
        xchg_merge
        and send_per == 4  # pyrefly: ignore[unbound-name]
        and SEND_SLAB_ROW0 >= 8
    )
    slab_row0 = 4 if compact_merge_slab else (SEND_SLAB_ROW0 if xchg else 0)
    if xchg:
        assert not preserve_suffix and batch_size == num_tokens, (
            "the fused exchange runs on a one-step grid"
        )
        assert send, "the fused exchange needs the send layout"
        assert bo_x2_ref is not None and lse_hbm_ref is not None
        assert send_exchange is not None
        xchg_axis, xchg_lane_axis = send_exchange[:2]
        xchg_rows = SEND_SLAB_ROW0 + send_rows  # pyrefly: ignore[unbound-name]
        xchg_block = SEND_PAD_ROWS + send_rows
        xchg_has_valid = start_end_seq_idx_ref.shape[0] >= 5
        xchg_tiers = (
            xchg_live_row_tiers(slab_row0, send_per, num_tokens, xchg_rows)
            if xchg_has_valid
            else (xchg_rows,)
        )
        assert bo_x2_ref.shape[2] == xchg_rows, bo_x2_ref.shape
        if xchg_merge:
            assert recv_v_ref is not None
            assert recv_v_ref.shape == (send_world, xchg_rows, o_dim), recv_v_ref.shape
            assert lse_hbm_ref.shape == (send_rows, o_dim), lse_hbm_ref.shape
            assert lse_hbm_ref.dtype == jnp.int16 and send_rows % 16 == 0
        else:
            assert lse_hbm_ref.shape == (send_world * xchg_block, o_dim), (
                lse_hbm_ref.shape
            )

    start_seq_idx = start_end_seq_idx_ref[0]
    end_seq_idx = start_end_seq_idx_ref[1]
    if preserve_suffix:
        prefix_end = start_end_seq_idx_ref[2]

    batch_start_seq_idx = start_seq_idx + pl.program_id(0) * batch_size
    batch_end_seq_idx = batch_start_seq_idx + batch_size - 1

    def get_batch_max_kv_len(seq_idx_start):
        max_kv = jnp.int32(0)
        for b in range(batch_size):
            max_kv = jnp.maximum(max_kv, kv_lens_ref[seq_idx_start + b])
        return max_kv

    kv_rungs = [bkv_sz]
    for div in (8, 4, 2):
        if bkv_sz % div == 0 and bkv_sz // div >= 64 and (bkv_sz // div) % 64 == 0:
            kv_rungs = [bkv_sz // div * k for k in (1, 2, 4, 8) if k <= div]
            break

    def attention_step1_qk_softmax(
        q,  # [num_q_heads, head_dim]
        kv,  # [bkv_sz, head_dim] <- Correspond to data from bkv_*_x2_ref
        kv_len,  # i32[] number of valid (non "-1") top-k rows in kv
    ):
        assert len(q.shape) == 2
        assert len(kv.shape) == 2
        assert q.shape[0] % num_q_heads == 0
        assert q.shape[1] == head_dim
        assert kv.shape[1] == head_dim

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

    def _fetch_live_rows(seq_idx_start, bkv_sem_idx, rows, *, wait=False):
        sem_nope = sems.at[0, bkv_sem_idx]
        sem_rope = sems.at[3, bkv_sem_idx]

        bkv_nope_vmem_ref = bkv_nope_x2_ref.at[bkv_sem_idx, :]
        bkv_rope_vmem_ref = bkv_rope_x2_ref.at[bkv_sem_idx, :]
        page_idx_start = seq_idx_start - start_seq_idx
        rope_div = bkv_sz // cache_kv_rope_hbm_ref.shape[1]
        for b in range(batch_size):
            kv_b = kv_lens_ref[seq_idx_start + b]
            n_b = jnp.clip((kv_b + 31) // 32 * 32, 32, rows)
            n_b = pl.multiple_of(n_b, 32)
            r_b = pl.multiple_of(n_b // rope_div, 8)
            nope_dst = bkv_nope_vmem_ref.at[pl.ds(b, 1), pl.ds(0, n_b)]
            rope_dst = bkv_rope_vmem_ref.at[pl.ds(b, 1), pl.ds(0, r_b)]
            if wait:
                _async_copy(nope_dst, nope_dst, sem_nope, True)
                _async_copy(rope_dst, rope_dst, sem_rope, True)
            else:
                _async_copy(
                    cache_kv_nope_hbm_ref.at[
                        pl.ds(page_idx_start + b, 1), pl.ds(0, n_b)
                    ],
                    nope_dst,
                    sem_nope,
                    False,
                )
                _async_copy(
                    cache_kv_rope_hbm_ref.at[
                        pl.ds(page_idx_start + b, 1), pl.ds(0, r_b)
                    ],
                    rope_dst,
                    sem_rope,
                    False,
                )

    if is_dcp_mode:

        def _fetch_bkv_slice(seq_idx_start, bkv_sem_idx, slice_sz, *, wait=False):
            sem_nope = sems.at[0, bkv_sem_idx]
            sem_rope = sems.at[3, bkv_sem_idx]
            nope_rows = (
                slice_sz if sparsecore_layout else slice_sz * kv_cache_utils.WORD_BYTES
            )
            rope_packing = (
                bkv_sz // cache_kv_rope_hbm_ref.shape[1] if sparsecore_layout else 1
            )
            rope_rows = slice_sz // rope_packing if sparsecore_layout else slice_sz
            bkv_nope_vmem_ref = bkv_nope_x2_ref.at[
                bkv_sem_idx, :, pl.ds(0, nope_rows), :
            ]
            bkv_rope_vmem_ref = bkv_rope_x2_ref.at[
                bkv_sem_idx, :, pl.ds(0, rope_rows), :
            ]
            page_idx_start = seq_idx_start - start_seq_idx
            if not wait:
                _async_copy(
                    cache_kv_nope_hbm_ref.at[
                        pl.ds(page_idx_start, batch_size), pl.ds(0, nope_rows), :
                    ],
                    bkv_nope_vmem_ref,
                    sem_nope,
                    wait,
                )
                _async_copy(
                    cache_kv_rope_hbm_ref.at[
                        pl.ds(page_idx_start, batch_size), pl.ds(0, rope_rows), :
                    ],
                    bkv_rope_vmem_ref,
                    sem_rope,
                    wait,
                )
            else:
                _async_copy(
                    src=bkv_nope_vmem_ref,
                    dst=bkv_nope_vmem_ref,
                    sem=sem_nope,
                    wait=True,
                )
                _async_copy(
                    src=bkv_rope_vmem_ref,
                    dst=bkv_rope_vmem_ref,
                    sem=sem_rope,
                    wait=True,
                )

        def _fetch_bkv_batch(seq_idx_start, bkv_sem_idx, *, wait=False, rows=None):
            if group_tab_ref is not None:
                _fetch_live_blocks(
                    seq_idx_start,
                    bkv_nope_x2_ref.at[bkv_sem_idx, :],
                    bkv_rope_x2_ref.at[bkv_sem_idx, :],
                    sems.at[0, bkv_sem_idx],
                    sems.at[3, bkv_sem_idx],
                    wait,
                )
                return
            if (
                row_widths
                and sparsecore_layout
                and rows is not None
                and "lrows" in body_levers
            ):
                _fetch_live_rows(seq_idx_start, bkv_sem_idx, rows, wait=wait)
                return
            if row_widths:
                _fetch_bkv_slice(
                    seq_idx_start,
                    bkv_sem_idx,
                    bkv_sz if rows is None else rows,
                    wait=wait,
                )
                return
            batch_max_kv = get_batch_max_kv_len(seq_idx_start)
            lo = 0
            for i, rung in enumerate(kv_rungs):
                hi_ok = True if i == len(kv_rungs) - 1 else batch_max_kv <= rung
                pl.when((batch_max_kv > lo) & hi_ok)(
                    functools.partial(
                        _fetch_bkv_slice, seq_idx_start, bkv_sem_idx, rung, wait=wait
                    )
                )
                lo = rung
    else:

        def _fetch_bkv_batch(seq_idx_start, bkv_sem_idx, *, wait=False, rows=None):
            sem_nope = sems.at[0, bkv_sem_idx]
            sem_rope = sems.at[3, bkv_sem_idx]

            bkv_nope_vmem_ref = bkv_nope_x2_ref.at[bkv_sem_idx, :]
            bkv_rope_vmem_ref = bkv_rope_x2_ref.at[bkv_sem_idx, :]

            page_idx_start = seq_idx_start - start_seq_idx
            if rows is not None and "lrows" in body_levers:
                _fetch_live_rows(seq_idx_start, bkv_sem_idx, rows, wait=wait)
                return
            if rows is not None:
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
            def _copy_block(*, a=a):
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
        if xchg:
            return  # The slabs leave by the exchange's remote pushes.
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

        if head_major_out:
            assert bo_hm_x2_ref is not None
            hm_ref = bo_hm_x2_ref.at[bo_sem_idx]
            if not wait:
                # Avoid per-token strided writes into [H, bsz, L]: transpose once.
                bo_hm_x2_ref[bo_sem_idx] = jnp.swapaxes(bo_x2_ref[0], 0, 1)
                hm_row = pl.multiple_of(seq_idx_start - out_row_offset, batch_size)
                dst = o_hbm_ref.at[:, pl.ds(hm_row, batch_size), :]
                _async_copy(hm_ref, dst, sem, wait)
            else:
                _async_copy(src=hm_ref, dst=hm_ref, sem=sem, wait=True)
            return

        if not wait:
            if send:
                row = pl.multiple_of(
                    SEND_PAD_ROWS + seq_idx_start * send_per, SEND_PAD_ROWS
                )
                assert lse_hbm_ref is not None
                dst = lse_hbm_ref.at[:, pl.ds(row, batch_size * send_per), :]
            else:
                dst = o_hbm_ref.at[pl.ds(seq_idx_start - out_row_offset, batch_size)]
            _async_copy(vmem_ref, dst, sem, wait)
        else:
            _async_copy(src=vmem_ref, dst=vmem_ref, sem=sem, wait=True)

    def _send_blse_batch(seq_idx_start, bo_sem_idx, *, wait=False):
        sem = sems.at[4, bo_sem_idx]
        vmem_ref = blse_x2_ref.at[bo_sem_idx, :]

        if send:
            if not wait:
                assert fill_ref is not None
                fill_ref[lax.div(seq_idx_start, batch_size)] = vmem_ref[...]
            return
        if not wait:
            _async_copy(
                vmem_ref,
                lse_hbm_ref.at[pl.ds(seq_idx_start - out_row_offset, batch_size)],
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
        lower = None
        for upper, width in zip((*row_widths, None), widths):
            if lower is None:
                assert upper is not None  # row_widths is non-empty here
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
            raw = bq_pe_x2_ref.at[bq_sem_idx, batch_idx][:, 128:Q_ROPE_HEAD_DIM]
            tab = bq_tab_x2_ref.at[bq_sem_idx, batch_idx][...]
            q = jnp.concatenate([q, rope_q_block(raw, tab)], -1)
        elif split_q:
            # The caller's concat + zero pad, rebuilt in VMEM.
            q = jnp.concatenate([q, bq_pe_x2_ref.at[bq_sem_idx, batch_idx][...]], -1)
            if q.shape[-1] < head_dim:
                q = jnp.pad(q, ((0, 0), (0, head_dim - q.shape[-1])))
        return q

    def store_out(bo_sem_idx, batch_idx, out):
        """Stores one token's [num_q_heads, >= o_dim] output into the bo tile."""
        if head_major_out:
            bo_x2_ref.at[0, batch_idx][...] = out[:, :o_dim]
            return
        if not send:
            bo_x2_ref.at[bo_sem_idx, batch_idx][...] = out[:, :o_dim]
            return
        words = pltpu.bitcast(out[:, :o_dim], jnp.int16)
        if not isinstance(batch_idx, int):
            for b in range(batch_size):
                pl.when(batch_idx == b)(
                    functools.partial(store_out, bo_sem_idx, b, out)
                )
            return
        rows = pl.ds(slab_row0 + batch_idx * send_per, send_per)
        assert bo_x2_ref is not None
        for w in range(send_world):
            bo_x2_ref[bo_sem_idx, w, rows, :] = words[w * send_per : (w + 1) * send_per]

    def build_send_lse_words():
        x = jnp.concatenate(
            [fill_ref[i] for i in range(num_tokens // batch_size)], axis=0
        )  # f32 [tokens, lanes], block-reversed heads
        acc = pack_send_lse_words(
            x,
            num_tokens=num_tokens,
            num_q_heads=num_q_heads,
            send_world=send_world,
            send_lanes=send_lanes,
        )
        if send_lanes < o_dim:
            acc = jnp.concatenate(
                [acc, jnp.zeros((send_world, o_dim - send_lanes), jnp.int32)], axis=1
            )
        # int16 row 2r+1 is the high half of int32 row r: the word row goes
        # to int16 row SEND_PAD_ROWS - 1, every other head row is zero.
        tile_shape = (send_world, SEND_PAD_ROWS // 2, o_dim)
        r = lax.broadcasted_iota(jnp.int32, tile_shape, 1)
        word = jnp.broadcast_to(
            lax.shift_left(acc, jnp.int32(16))[:, None, :], tile_shape
        )
        if xchg:
            assert bo_x2_ref is not None
            old = pltpu.bitcast(bo_x2_ref[0, :, pl.ds(0, SEND_PAD_ROWS), :], jnp.int32)
            tile = jnp.where(
                r == slab_row0 // 2 - 1,
                word,
                jnp.where(r < slab_row0 // 2, 0, old),
            )
            bo_x2_ref[0, :, pl.ds(0, SEND_PAD_ROWS), :] = pltpu.bitcast(tile, jnp.int16)
            return
        assert send_wbuf is not None
        send_wbuf[...] = pltpu.bitcast(
            jnp.where(r == SEND_PAD_ROWS // 2 - 1, word, 0), jnp.int16
        )

    def send_words_copy():
        assert lse_hbm_ref is not None
        return pltpu.make_async_copy(
            send_wbuf, lse_hbm_ref.at[:, pl.ds(0, SEND_PAD_ROWS), :], fill_sem
        )

    def xchg_peer(j):
        ids = {xchg_axis: jnp.int32(j)}
        if xchg_lane_axis is not None:
            ids[xchg_lane_axis] = start_end_seq_idx_ref[3]
        return ids

    def xchg_signal():
        bsem = pltpu.get_barrier_semaphore()
        for j in range(send_world):
            pl.semaphore_signal(
                bsem, 1, device_id=xchg_peer(j), device_id_type=pl.DeviceIdType.MESH
            )

    def xchg_round():
        """Run slab exchange with live-row trimming and rotated Z-first peer order."""
        n = send_world
        me = start_end_seq_idx_ref[2]
        send_sem, recv_sem, local_sem = (xchg_sems.at[i] for i in range(3))
        pl.semaphore_wait(pltpu.get_barrier_semaphore(), n)
        expect = n - 1

        def rows_of(k, nrows):
            if xchg_merge:
                # Merge mode: slot k of the receive scratch (VMEM).
                return (
                    recv_v_ref.at[k]
                    if nrows == xchg_rows
                    else recv_v_ref.at[k, pl.ds(0, nrows)]
                )
            start = k * xchg_block + SEND_SLAB_ROW0
            if not isinstance(k, int):
                start = pl.multiple_of(start, 8)
            assert lse_hbm_ref is not None
            return lse_hbm_ref.at[pl.ds(start, nrows)]

        def src_of(j, nrows):
            return (
                bo_x2_ref.at[0, j]
                if nrows == xchg_rows
                else bo_x2_ref.at[0, j, pl.ds(0, nrows)]
            )

        def others_ref(nrows):
            if xchg_merge:
                return (
                    recv_v_ref.at[pl.ds(0, expect)]
                    if nrows == xchg_rows
                    else recv_v_ref.at[pl.ds(0, expect), pl.ds(0, nrows)]
                )
            assert lse_hbm_ref is not None
            return lse_hbm_ref.at[pl.ds(0, expect * nrows)]

        def _issue_and_wait_recv(nrows):
            mine = rows_of(me, nrows)
            for peer_j in xchg_peer_order(me, n):
                pltpu.make_async_remote_copy(
                    src_ref=src_of(peer_j, nrows),
                    dst_ref=mine,
                    send_sem=send_sem,
                    recv_sem=recv_sem,
                    device_id=xchg_peer(peer_j),
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()

            local = pltpu.make_async_copy(
                src_of(me, nrows), rows_of(me, nrows), local_sem
            )
            local.start()
            others = others_ref(nrows)
            pltpu.make_async_copy(others, others, recv_sem).wait()
            local.wait()

        def _wait_sends_for(nrows):
            others = others_ref(nrows)
            pltpu.make_async_copy(others, others, send_sem).wait()

        if len(xchg_tiers) == 1:
            _issue_and_wait_recv(xchg_tiers[0])
            return lambda: _wait_sends_for(xchg_tiers[0])

        need_rows = jnp.int32(slab_row0) + jnp.maximum(
            start_end_seq_idx_ref[4], jnp.int32(0)
        ) * jnp.int32(send_per)

        def _dispatch_tiers(fn):
            for idx, tier_rows in enumerate(xchg_tiers):
                if idx == 0:
                    cond = need_rows <= jnp.int32(tier_rows)
                elif idx == len(xchg_tiers) - 1:
                    cond = need_rows > jnp.int32(xchg_tiers[idx - 1])
                else:
                    cond = (need_rows > jnp.int32(xchg_tiers[idx - 1])) & (
                        need_rows <= jnp.int32(tier_rows)
                    )

                @pl.when(cond)
                def _(tr=tier_rows):
                    fn(tr)

        _dispatch_tiers(_issue_and_wait_recv)
        return lambda: _dispatch_tiers(_wait_sends_for)

    def xchg_merge_store():
        tl = send_rows
        assert (
            bo_x2_ref is not None and recv_v_ref is not None and lse_hbm_ref is not None
        )
        out = merge_received_slabs(
            recv_v_ref,
            send_world=send_world,
            send_rows=tl,
            slab_row0=slab_row0,
            o_dim=o_dim,
            live_rows=start_end_seq_idx_ref[4] * send_per,
        )
        bo_x2_ref[1, 0, pl.ds(0, tl), :] = pltpu.bitcast(out, jnp.int16)
        copy = pltpu.make_async_copy(
            bo_x2_ref.at[1, 0, pl.ds(0, tl)], lse_hbm_ref, fill_sem
        )
        copy.start()
        copy.wait()

    def store_blse(bo_sem_idx, batch_idx, lse_col):
        """Stores one token's f32[num_q_heads, 1] log-sum-exp column."""
        if not lse_lane_dense:
            blse_x2_ref.at[bo_sem_idx, batch_idx][...] = lse_col
            return
        # Heads from sublanes to lanes, exactly: each column of the masked
        # [heads, heads] tile holds one live entry, the rest -inf.
        if send:
            eye_shape = (num_q_heads, send_lanes)
            h = lax.broadcasted_iota(jnp.int32, eye_shape, 0)
            target = (send_world - 1) * send_per - h + 2 * (h & (send_per - 1))
            eye = lax.broadcasted_iota(jnp.int32, eye_shape, 1) == target
        else:
            eye_shape = (num_q_heads, num_q_heads)
            eye = lax.broadcasted_iota(jnp.int32, eye_shape, 0) == (
                lax.broadcasted_iota(jnp.int32, eye_shape, 1)
            )
        row = jnp.max(
            jnp.where(eye, jnp.broadcast_to(lse_col, eye_shape), -jnp.inf),
            axis=0,
            keepdims=True,
        )
        assert blse_x2_ref is not None
        tile_ref = blse_x2_ref.at[bo_sem_idx]  # [bsz, num_q_heads]
        if isinstance(batch_idx, int):
            tile_ref[pl.ds(batch_idx, 1), :] = row
        else:
            pick = lax.broadcasted_iota(jnp.int32, tile_ref.shape, 0) == batch_idx
            tile_ref[...] = jnp.where(pick, row, tile_ref[...])

    rope_packing = bkv_sz // cache_kv_rope_hbm_ref.shape[1] if sparsecore_layout else 1

    def _unpack_sc_rope(raw_rope, slice_sz):
        if rope_packing == native_sc_gather.COMPACT_ROPE_PACKING:
            # Each atom of 64 tokens sits in one uint32[8, 128] tile: lanes 0..63
            # hold tokens 0..31 and lanes 64..127 hold tokens 32..63.
            n_atoms = slice_sz // native_sc_gather.OCCURRENCES_PER_ATOM
            lane_id = lax.broadcasted_iota(jnp.int32, raw_rope.shape, 1)
            left = jnp.where(lane_id < 64, raw_rope, jnp.uint32(0))
            right = jnp.where(lane_id < 64, pltpu.roll(raw_rope, 64, 1), jnp.uint32(0))
            tiles = []
            for a in range(n_atoms):
                r0 = a * native_sc_gather.COMPACT_ROWS_PER_ATOM
                r1 = r0 + native_sc_gather.COMPACT_ROWS_PER_ATOM
                tiles.append(left[r0:r1, :])
                tiles.append(right[r0:r1, :])
            expanded = jnp.concatenate(tiles, axis=0)
            return pltpu.bitcast(expanded, jnp.uint8).reshape(slice_sz, 128)
        return pltpu.bitcast(raw_rope, jnp.uint8).reshape(slice_sz, 128)

    def load_bkv_levers(bkv_sem_idx, batch_idx, rows, levers):
        nope_ref = bkv_nope_x2_ref.at[bkv_sem_idx, batch_idx]
        rope_ref = bkv_rope_x2_ref.at[bkv_sem_idx, batch_idx]
        rope_ref = rope_ref.at[pl.ds(0, rows * rope_ref.shape[0] // bkv_sz)]
        kv_len = kv_lens_ref[batch_start_seq_idx + batch_idx]
        zu = "zu" in levers
        if "tr" in levers:
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
        bkv_rope = _unpack_sc_rope(rope_ref[...], rows)
        if zu:
            rope_key = lax.broadcasted_iota(jnp.int32, bkv_rope.shape, 0)
            bkv_rope = jnp.where(rope_key < kv_len, bkv_rope, jnp.uint8(0))
        bkv_rope = pltpu.bitcast(bkv_rope, jnp.float8_e4m3fn)
        bkv = jnp.concatenate([bkv_nope, bkv_rope], axis=-1)
        if bkv.shape[-1] < head_dim:
            bkv = jnp.pad(bkv, ((0, 0), (0, head_dim - bkv.shape[-1])))
        if not zu:
            k_span = lax.broadcasted_iota(jnp.int32, bkv.shape, 0)
            bkv = jnp.where(k_span < kv_len, bkv, 0)
        return bkv

    def load_bkv(bkv_sem_idx, batch_idx, rows=None, levers=()):
        slice_sz = bkv_sz if rows is None else rows
        if levers:
            return load_bkv_levers(bkv_sem_idx, batch_idx, slice_sz, levers)
        nope_ref = bkv_nope_x2_ref.at[bkv_sem_idx, batch_idx]
        rope_ref = bkv_rope_x2_ref.at[bkv_sem_idx, batch_idx]
        return unpack_bkv(
            nope_ref,
            rope_ref,
            lambda: kv_lens_ref[batch_start_seq_idx + batch_idx],
            slice_sz=slice_sz,
        )

    def unpack_bkv(bkv_nope_ref, bkv_rope_ref, get_kv_len, slice_sz=bkv_sz):
        if slice_sz == bkv_sz:
            bkv_nope = bkv_nope_ref[...]
        elif sparsecore_layout:
            bkv_nope = bkv_nope_ref.at[pl.ds(0, slice_sz), :][...]
        else:
            bkv_nope = bkv_nope_ref.at[
                pl.ds(0, slice_sz * kv_cache_utils.WORD_BYTES), :
            ][...]
        if sparsecore_layout:
            # Native SC returns one uint32[128] row per token. Bitcasting
            # expands the packed byte lanes before folding them into the
            # logical 512-byte NOPE vector.
            bkv_nope = pltpu.bitcast(bkv_nope, jnp.uint8)
        bkv_nope = bkv_nope.reshape(slice_sz, nope_dim)
        bkv_nope = pltpu.bitcast(bkv_nope, jnp.float8_e4m3fn)

        if slice_sz == bkv_sz:
            bkv_rope = bkv_rope_ref[...]
        elif sparsecore_layout:
            bkv_rope = bkv_rope_ref.at[pl.ds(0, slice_sz // rope_packing), :][...]
        else:
            bkv_rope = bkv_rope_ref.at[pl.ds(0, slice_sz), :][...]
        if sparsecore_layout:
            bkv_rope = _unpack_sc_rope(bkv_rope, slice_sz)
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
                store_out(bo_sem_idx, batch_idx, out)
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
            if is_dcp_mode or lse_mask_empty:
                kv_len = kv_lens_ref[batch_start_seq_idx + batch_idx]
                lse_col = jnp.where(kv_len > 0, lse_col, -jnp.inf)
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

        def _softmax_stacked(s, lim):
            s *= sm_scale
            if k_scale != 1.0:
                s *= k_scale
            span = lax.broadcasted_iota(jnp.int32, s.shape, 1)
            s = jnp.where(span < lim, s, jnp.finfo(jnp.float32).min)
            m = jnp.max(s, axis=1, keepdims=True)
            p = jnp.exp(s - m)
            return p, jnp.sum(p, axis=1, keepdims=True), m

        def _finish_staged_token(slot, sub, tok_idx, w, p, l_sum, m, idx, levers):
            h = num_q_heads
            kv = kv_stage_ref.at[slot, sub, pl.ds(0, w)][...]
            sl = slice(idx * h, (idx + 1) * h)
            store_out(
                bo_sem_idx,
                tok_idx,
                attention_step2_pv(p[sl], kv, l_sum[sl], levers),
            )
            write_blse(tok_idx, m[sl], l_sum[sl])

        def pair_finish(k, par, p, l_sum, m, levers):
            for j in range(2):
                _finish_staged_token(
                    par, j, 2 * k + j, p.shape[1], p, l_sum, m, j, levers
                )

        def run_pairs(rows, levers):
            assert batch_size % 2 == 0
            h, prev = num_q_heads, None
            for k in range(batch_size // 2):
                par, t0, t1 = k % 2, 2 * k, 2 * k + 1
                kv_len0 = kv_lens_ref[batch_start_seq_idx + t0]
                kv_len1 = kv_lens_ref[batch_start_seq_idx + t1]
                kv0 = load_bkv(bi_sem_idx, t0, rows, levers)
                kv1 = load_bkv(bi_sem_idx, t1, rows, levers)
                w = kv0.shape[0]
                kv_stage_ref.at[par, 0, pl.ds(0, w)][...] = kv0
                kv_stage_ref.at[par, 1, pl.ds(0, w)][...] = kv1
                s = jnp.concatenate(
                    [
                        jnp.einsum(
                            "nd,md->nm",
                            load_bq(bi_sem_idx, t0),
                            kv0,
                            preferred_element_type=jnp.float32,
                        ),
                        jnp.einsum(
                            "nd,md->nm",
                            load_bq(bi_sem_idx, t1),
                            kv1,
                            preferred_element_type=jnp.float32,
                        ),
                    ],
                    axis=0,
                )
                lim = jnp.where(
                    lax.broadcasted_iota(jnp.int32, s.shape, 0) < h, kv_len0, kv_len1
                )
                p, l_sum, m = _softmax_stacked(s, lim)
                if prev is not None:
                    pair_finish(k - 1, *prev, levers)
                prev = (par, p, l_sum, m)
            assert prev is not None
            pair_finish(batch_size // 2 - 1, *prev, levers)

        def run_quad(rows, levers):
            h, scores, kv_lens, w = num_q_heads, [], [], None
            for t in range(batch_size):
                kv_lens.append(kv_lens_ref[batch_start_seq_idx + t])
                kv = load_bkv(bi_sem_idx, t, rows, levers)
                w = kv.shape[0]
                kv_stage_ref.at[0, t, pl.ds(0, w)][...] = kv
                scores.append(
                    jnp.einsum(
                        "nd,md->nm",
                        load_bq(bi_sem_idx, t),
                        kv,
                        preferred_element_type=jnp.float32,
                    )
                )
            s = jnp.concatenate(scores, axis=0)
            row = lax.broadcasted_iota(jnp.int32, s.shape, 0)
            lim = jnp.full(s.shape, kv_lens[batch_size - 1], jnp.int32)
            for t in range(batch_size - 2, -1, -1):
                lim = jnp.where(row < (t + 1) * h, kv_lens[t], lim)
            p, l_sum, m = _softmax_stacked(s, lim)
            for t in range(batch_size):
                _finish_staged_token(0, t, t, w, p, l_sum, m, t, levers)

        def run_steps(rows=None, levers=()):
            if "pair" in levers:
                return run_pairs(rows, levers)
            if "quad" in levers:
                return run_quad(rows, levers)
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
                    dep = prev_out
                    if dep.shape != bq.shape:  # "pv512": a NOPE-wide output
                        dep = dep[:, :1]
                    bq = jnp.where(dep == jnp.inf, dep, bq)

                p, l_sum, m = attention_step1_qk_softmax(bq, bkv, kv_len)

                if prev_p is not None:
                    assert prev_bkv is not None
                    assert prev_l_sum is not None
                    out = attention_step2_pv(prev_p, prev_bkv, prev_l_sum, levers)

                    store_out(bo_sem_idx, batch_idx - 1, out)
                    write_blse(batch_idx - 1, prev_m, prev_l_sum)
                    prev_out = out

                prev_p = p
                prev_bkv = bkv
                prev_l_sum = l_sum
                prev_m = m

            out = attention_step2_pv(prev_p, prev_bkv, prev_l_sum, levers)
            store_out(bo_sem_idx, batch_size - 1, out)
            write_blse(batch_size - 1, prev_m, prev_l_sum)

        if group_tab_ref is not None:
            if live_widths:
                assert live_nope_hbm_ref is not None
                atom_rows = live_nope_hbm_ref.shape[2]
                need = group_tab_ref[batch_start_seq_idx // batch_size] * atom_rows
                lower = None
                for width in tuple(live_widths) + (bkv_sz,):
                    fits = (
                        need <= width
                        if lower is None
                        else (need > lower) & (need <= width)
                    )
                    pl.when(fits)(
                        functools.partial(run_steps, width, tuple(body_levers))
                    )
                    lower = width
            else:
                run_steps(bkv_sz, tuple(body_levers))
        elif is_dcp_mode:
            batch_max_kv_len = get_batch_max_kv_len(batch_start_seq_idx)

            @pl.when(batch_max_kv_len == 0)
            def _zero_batch():
                zero_out = jnp.zeros((num_q_heads, o_dim), dtype=q_dtype)
                neg_inf_lse = jnp.full((num_q_heads, 1), -jnp.inf, dtype=jnp.float32)
                for batch_idx in range(batch_size):
                    store_out(bo_sem_idx, batch_idx, zero_out)
                    if return_lse:
                        store_blse(bo_sem_idx, batch_idx, neg_inf_lse)

            if row_widths:

                def wait_and_run(rows):
                    wait_fetch_bkv_batch(batch_start_seq_idx, bi_sem_idx, rows)
                    pl.when(batch_max_kv_len > 0)(
                        functools.partial(
                            run_steps,
                            bkv_sz if rows is None else rows,
                            tuple(body_levers),
                        )
                    )

                with_row_width(batch_start_seq_idx, wait_and_run)
            else:
                lo = 0
                for i, rung in enumerate(kv_rungs):
                    hi_ok = True if i == len(kv_rungs) - 1 else batch_max_kv_len <= rung
                    pl.when((batch_max_kv_len > lo) & hi_ok)(
                        functools.partial(run_steps, rung)
                    )
                    lo = rung
        elif row_widths:

            def wait_and_run(rows):
                wait_fetch_bkv_batch(batch_start_seq_idx, bi_sem_idx, rows)
                run_steps(bkv_sz if rows is None else rows, tuple(body_levers))

            with_row_width(batch_start_seq_idx, wait_and_run)
        else:
            run_steps(None, tuple(body_levers))
        if capped_windows is not None and "fxskip" in body_levers:
            fix_base = batch_start_seq_idx
            fix_need = kv_lens_ref[fix_base]
            for b in range(1, batch_size):
                fix_need = jnp.maximum(fix_need, kv_lens_ref[fix_base + b])
            need_fix = fix_need > bkv_sz
            pl.when(need_fix)(
                functools.partial(fixup_capped_overflow, bi_sem_idx, bo_sem_idx)
            )
        elif capped_windows is not None:
            fixup_capped_overflow(bi_sem_idx, bo_sem_idx)
        start_send_bo_batch(batch_start_seq_idx, bo_sem_idx)
        if lse_fill and not send:

            @pl.when(pl.program_id(0) == 0)
            def _():
                fill_copy().wait()

        if return_lse:
            _send_blse_batch(batch_start_seq_idx, bo_sem_idx)

    ### ------- Kernel start ------- ###

    has_work = end_seq_idx > start_seq_idx if lse_fill else True

    if lse_fill:

        @pl.when(pl.program_id(0) == 0)
        def _fill_lse():
            sentinel = -jnp.inf if lse_mask_empty else jnp.finfo(jnp.float32).min
            assert fill_ref is not None
            fill_ref[...] = jnp.full(fill_ref.shape, sentinel, jnp.float32)
            if send:
                return  # VMEM only: the word build reads it.
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

    if xchg:
        xchg_signal()

    if lse_fill:
        pl.when(has_work)(process)
    else:
        process()

    is_last = pl.program_id(0) == pl.num_programs(0) - 1
    if xchg:
        build_send_lse_words()
        wait_xchg_sends = xchg_round()
        if xchg_merge:
            # Avoid blocking xchg_merge_store (reads recv_v_ref, writes bo slot 1)
            # on trailing send_sem completions guarding bo slot 0.
            xchg_merge_store()
        wait_xchg_sends()
    elif send:

        @pl.when(is_last)
        def _send_words():
            build_send_lse_words()
            send_words_copy().start()

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

    if send and not xchg:
        pl.when(is_last)(lambda: send_words_copy().wait())

    ### ------- Kernel end ------- ###


def merge_pieces_supported(num_tokens: int, chunk_size: int, pieces: int) -> bool:
    """Whether `out_pieces=pieces` applies to a `num_tokens` bucket attended in
    `chunk_size` chunks: several whole chunks, and pieces that are whole chunks
    or split one evenly (a piece is then that chunk's output)."""
    if pieces <= 1 or num_tokens <= chunk_size or num_tokens % chunk_size:
        return False
    if num_tokens % pieces:
        return False
    rows = num_tokens // pieces
    return rows % chunk_size == 0 or chunk_size % rows == 0


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
    return small_chunk, largest_divisor(chunk_size, batch_cap)


def supports_split_q(q_nope: jax.Array, q_pe: jax.Array) -> bool:
    """Whether `(q_nope, q_pe)` satisfies the `q_pe=` split-Q kernel envelope."""
    return (
        q_nope.ndim == 3
        and q_pe.ndim == 3
        and q_pe.shape[:2] == q_nope.shape[:2]
        and q_pe.dtype == q_nope.dtype
        and q_nope.shape[-1] % 128 == 0
        and q_nope.shape[1] % get_dtype_packing(q_nope.dtype) == 0
    )


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
        "is_dcp_mode",
        "capped_live_rows",
        "lse_lane_dense",
        "lse_mask_empty",
        "fused_window_prep",
        "mixed_live_gather",
        "mixed_live_widths",
        "mixed_body_levers",
        "lse_in_kernel",
        "decode_row_widths",
        "decode_body_levers",
        "send_world",
        "send_exchange",
        "out_pieces",
        "packed_rope",
        "head_major_out_chunks",
    ),
)
def sparse_ragged_paged_attention(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, head_dim]
    # TensorCore: u8[pages, page_size, 4, 128].
    # SparseCore: u32[pages, page_size, 128].
    cache_kv_nope: jax.Array,
    # TensorCore: u8[pages, page_size/4, 4, 128].
    # SparseCore: u32[pages, page_size/tokens_per_row, 128].
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
    vmem_limit_bytes: int | None = None,
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
    is_dcp_mode: bool = False,
    capped_live_rows: bool = False,
    q_pe: jax.Array | None = None,
    lse_lane_dense: bool = False,
    lse_mask_empty: bool = False,
    fused_window_prep: bool = False,
    mixed_live_gather: bool = False,
    mixed_live_widths: tuple[int, ...] = (),
    mixed_body_levers: tuple[str, ...] = (),
    q_rope_table: jax.Array | None = None,
    lse_in_kernel: bool = False,
    kv_write: jax.Array | tuple[jax.Array, jax.Array, jax.Array] | None = None,
    decode_row_widths: tuple[int, ...] = (),
    send_world: int = 0,
    out_pieces: int = 0,
    decode_body_levers: tuple[str, ...] = (),
    send_exchange: tuple | None = None,
    send_peer: jax.Array | None = None,
    send_recv_in: jax.Array | None = None,
    packed_rope: bool | None = None,
    head_major_out_chunks: bool = False,
) -> jax.Array | tuple[jax.Array, ...]:
    """Sparse MLA ragged paged attention supporting TensorCore and SparseCore caches."""
    if vmem_limit_bytes is None:
        vmem_limit_bytes = _default_vmem_limit_bytes()
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
                "sparsecore ROPE cache must be uint32"
                "[pages,page_size/tokens_per_row,128]"
            )
        rope_rows = cache_kv_rope.shape[1]
        if rope_rows <= 0 or cache_kv_nope.shape[1] % rope_rows != 0:
            raise ValueError(
                "sparsecore NOPE page_size must be a positive multiple of ROPE rows"
            )
        rope_tokens_per_row = cache_kv_nope.shape[1] // rope_rows
        rope_words = (
            128 // rope_tokens_per_row if 128 % rope_tokens_per_row == 0 else -1
        )
        if (
            rope_words <= 0
            or rope_words % native_sc_gather.NUM_LANES != 0
            or native_sc_gather.ROPE_PACKING * rope_words > 128
        ):
            raise ValueError(
                f"sparsecore ROPE cache packs {rope_tokens_per_row} tokens per "
                "128-word row, which is incompatible with SparseCore gather"
            )
    else:
        rope_tokens_per_row = 4
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
    if cache_kv_nope.size != rope_tokens_per_row * cache_kv_rope.size:
        raise ValueError(
            f"NOPE cache size ({cache_kv_nope.size}) must be exactly "
            f"{rope_tokens_per_row} * ROPE cache size ({cache_kv_rope.size})"
        )
    if (
        cache_kv_nope.shape[0] != cache_kv_rope.shape[0]
        or cache_kv_nope.shape[1] != rope_tokens_per_row * cache_kv_rope.shape[1]
    ):
        raise ValueError(
            "NOPE and ROPE caches must have matching page counts and page geometry"
        )

    _, actual_num_q_heads, actual_head_dim = q.shape

    if send_exchange is not None and (
        not send_world or send_peer is None or len(send_exchange) not in (2, 3, 4, 5)
    ):
        raise ValueError(
            "send_exchange = (axis, lane_axis) needs send_world and send_peer "
            "(int32 [2]: rank, lane)"
        )
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
        if q_pe.ndim != 3 or q_pe.shape[:2] != q.shape[:2] or q_pe.dtype != q.dtype:
            raise ValueError(
                f"q_pe {q_pe.shape}/{q_pe.dtype} must be 3D and match q_nope "
                f"{q.shape}/{q.dtype} in tokens, heads and dtype"
            )
        if not supports_split_q(q, q_pe):
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

    for w_name, widths in (
        ("mixed_live_widths", mixed_live_widths),
        ("decode_row_widths", decode_row_widths),
    ):
        if widths and (
            topk_cap is None
            or list(widths) != sorted(set(widths))
            or any(w % 128 or not 0 < w < topk_cap for w in widths)
        ):
            raise ValueError(
                f"{w_name} {widths} must be ascending multiples of 128 < {topk_cap}"
            )
    if not set(mixed_body_levers) <= set(MIXED_BODY_LEVERS):
        raise ValueError(f"invalid mixed_body_levers: {mixed_body_levers}")
    if not set(decode_body_levers) <= set(DECODE_BODY_LEVERS):
        raise ValueError(f"invalid decode_body_levers: {decode_body_levers}")
    if "quad" in decode_body_levers and "pair" in decode_body_levers:
        raise ValueError("decode_body_levers: 'pair' and 'quad' are alternatives")
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
    window_indices_all: Any = None
    chunk_window_rows = 0  # rows of `window_indices_all` per chunk
    group_tab = group_num_live = None
    prep_small, prep_batch = _chunk_kernel_batch_size(
        uniform_chunk,
        attention_kernel_batch_size,
        small_chunk_tokens,
        small_chunk_batch_size,
    )
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
    token_end = cu_q_lens[distribution[2]]
    if fused_prep:
        assert topk_cap is not None and page_indices is not None
        kv_lens = jnp.where(
            jnp.arange(topk_indices.shape[0], dtype=jnp.int32) < token_end,
            jnp.sum(topk_indices != -1, axis=-1),
            0,
        )
        physical_indices = None
        window_indices_all = prepare_capped_window_fused(
            topk_indices,
            page_indices,
            cu_q_lens,
            token_end,
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
        and topk_cap is not None
        and mixed_fused_prep_supported(
            q.shape[0], topk, uniform_chunk, prep_batch, topk_cap
        )
    ):
        assert page_indices is not None
        chunk_window_rows = topk * uniform_chunk // 128
        physical_indices = None
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
    elif window_prep:
        assert topk_cap is not None and page_indices is not None
        chunk_window_rows = topk // topk_cap * uniform_chunk
        kv_lens = jnp.where(
            jnp.arange(topk_indices.shape[0], dtype=jnp.int32) < token_end,
            jnp.sum(topk_indices != -1, axis=-1),
            0,
        )
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
            is_dcp_mode=is_dcp_mode,
            num_valid_tokens=cu_q_lens[distribution[2]],
        )

    # Rows the grid never reaches (padding past `batch_end`) keep this
    # sentinel, so a downstream merge weights them at exp(min - max) == 0
    # instead of tripping on uninitialised memory.
    lse_fill = bool(lse_in_kernel and return_lse and lse_lane_dense and num_chunks == 1)
    if send_world:
        if not (lse_fill and q_pe is not None and q.dtype == jnp.bfloat16):
            raise ValueError(
                "send_world needs lse_in_kernel on one chunk, split q and bf16 q"
            )
        _, send_batch = _chunk_kernel_batch_size(
            uniform_chunk,
            attention_kernel_batch_size,
            small_chunk_tokens,
            small_chunk_batch_size,
        )
        if q.shape[1] != actual_num_q_heads or not send_layout_supported(
            q.shape[0], q.shape[1], send_world, q.shape[2], send_batch
        ):
            raise ValueError(
                f"send layout unsupported: q {q.shape}, world {send_world}, "
                f"batch {send_batch}"
            )
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

    piece_rows = 0
    if out_pieces > 1 and merge_pieces_supported(
        q.shape[0], gather_and_attention_chunk_size, out_pieces
    ):
        if not (
            return_lse
            and lse_lane_dense
            and lse_mask_empty
            and not lse_fill
            and kv_write is None
        ):
            raise ValueError(
                "out_pieces needs return_lse, lse_lane_dense and lse_mask_empty"
            )
        piece_rows = max(q.shape[0] // out_pieces, gather_and_attention_chunk_size)
    piece_outs: list[jax.Array] = []
    piece_lses: list[jax.Array] = []

    def run_chunk(i, chunk_lse, q_in=None, **extra) -> Any:
        start_pos = i * gather_and_attention_chunk_size
        end_pos = min(start_pos + gather_and_attention_chunk_size, q.shape[0])
        return sparse_attention_chunk(
            q if q_in is None else q_in,
            cache_kv_nope,
            cache_kv_rope,
            kv_lens,
            physical_indices,
            chunk_lse,
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
            is_dcp_mode=is_dcp_mode,
            small_chunk_tokens=small_chunk_tokens,
            small_chunk_batch_size=small_chunk_batch_size,
            small_chunk_live_gather=small_chunk_live_gather,
            small_chunk_gather_atoms=small_chunk_gather_atoms,
            live_gather_all_chunks=live_gather_all_chunks,
            topk_cap=topk_cap,
            capped_live_rows=capped_live_rows,
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
            row_levers=(
                decode_body_levers
                if group_tab is None and fused_prep and decode_row_widths
                else (
                    mixed_body_levers
                    if group_tab is None
                    and not decode_row_widths
                    and sparsecore_layout
                    and (q_pe is not None or "zu" not in mixed_body_levers)
                    and q.shape[0] > MIXED_BODY_MAX_SKIP_TOKENS
                    else ()
                )
            ),
            q_rope_table=q_rope_table,
            lse_fill=lse_fill,
            kv_write=kv_write,
            send_world=send_world,
            send_exchange=send_exchange,
            send_peer=send_peer,
            send_recv_in=send_recv_in,
            packed_rope=packed_rope,
            head_major_out=head_major_out_chunks,
            **extra,
        )

    prev_out = None
    hm_chunks: list[jax.Array] = []
    if head_major_out_chunks and (
        return_lse or kv_write is not None or piece_rows or send_world
    ):
        raise ValueError(
            "head_major_out_chunks requires no "
            "return_lse/kv_write/out_pieces/send_world"
        )
    for i in range(num_chunks):
        start_pos = i * gather_and_attention_chunk_size
        out_piece = None
        piece_lse = lse
        if piece_rows:
            offset = start_pos // piece_rows * piece_rows
            carried = None
            if start_pos == offset:
                piece_lse = jnp.full((piece_rows, num_q_heads), -jnp.inf, jnp.float32)
            else:
                piece_lse = piece_lses.pop()
                carried = piece_outs.pop()
            out_piece = (piece_rows, offset, carried)
        extra: dict[str, Any] = {"out_piece": out_piece}
        if piece_rows and i:
            # Serialize chunk launches so live piece outputs do not overlap in HBM/VMEM.
            extra["q_in"], _ = lax.optimization_barrier((q, prev_out))
        elif head_major_out_chunks and i:
            extra["q_in"], hm_chunks[-1] = lax.optimization_barrier((q, hm_chunks[-1]))
        result = run_chunk(i, piece_lse, **extra)
        prev_out = result if head_major_out_chunks else result[0]
        if kv_write is not None:
            result, (cache_kv_nope, cache_kv_rope) = result
        if piece_rows:
            piece_outs.append(result[0])
            piece_lses.append(result[1])
        elif head_major_out_chunks:
            hm_chunks.append(result)
        elif return_lse:
            q, lse = result
        else:
            q = result

    if head_major_out_chunks:
        return tuple(
            c
            if (c.shape[0] == actual_num_q_heads and c.shape[2] == actual_head_dim)
            else c[:actual_num_q_heads, :, :actual_head_dim]
            for c in hm_chunks
        )

    if piece_rows:
        q = jnp.concatenate(piece_outs, axis=0)
        lse = jnp.concatenate(piece_lses, axis=0)

    # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    out = prepare_outputs(q, actual_num_q_heads, actual_head_dim)
    caches = () if kv_write is None else (cache_kv_nope, cache_kv_rope)
    if return_lse:
        lse = cast(jax.Array, lse)
        if send_world:
            return out, lse, *caches
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
    is_dcp_mode: bool = False,
    num_valid_tokens: jax.Array | None = None,  # i32
) -> tuple[jax.Array, jax.Array]:
    """Per-token valid top-k counts and flat cache row indices to gather.

    Depends only on the indexer output and the page table, not on the query.
    """
    max_num_seqs = cu_q_lens.shape[0] - 1
    num_tokens, topk = topk_indices.shape
    # TODO: skip gather for padding tokens in topk_indices.
    kv_lens = jnp.sum(topk_indices != -1, axis=-1)
    if is_dcp_mode and num_valid_tokens is not None:
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
        seq_page_ids = jnp.where(is_padding, 0, topk_indices // page_size)
        token_offset = jnp.where(is_padding, 0, topk_indices % page_size)
        page_ids = gather_page_ids(
            page_indices, seq_page_ids, seq_ids_segment, max_num_seqs
        )
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


def sparse_attention_chunk(
    q: jax.Array,  # [max_num_tokens, num_q_heads, head_dim], padded
    # TensorCore: u8[pages, page_size, 4, 128].
    # SparseCore: u32[pages, page_size, 128].
    cache_kv_nope: jax.Array,
    # TensorCore: u8[pages, page_size/4, 4, 128].
    # SparseCore: u32[pages, page_size / tokens_per_row, 128].
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
    vmem_limit_bytes: int | None = None,
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
    capped_live_rows: bool = False,
    topk_override: int | None = None,
    is_dcp_mode: bool = False,
    window_indices: jax.Array | None = None,
    gather_live_in_indices: bool = False,
    q_pe: jax.Array | None = None,
    lse_lane_dense: bool = False,
    lse_mask_empty: bool = False,
    group_tab: jax.Array | None = None,
    group_num_live: jax.Array | None = None,
    group_widths: tuple[int, ...] = (),
    group_levers: tuple[str, ...] = (),
    row_levers: tuple[str, ...] = (),
    q_rope_table: jax.Array | None = None,
    lse_fill: bool = False,
    kv_write: jax.Array | tuple[jax.Array, jax.Array, jax.Array] | None = None,
    row_widths: tuple[int, ...] = (),
    send_world: int = 0,
    out_piece: tuple[int, int, jax.Array | None] | None = None,
    send_exchange: tuple | None = None,
    send_peer: jax.Array | None = None,
    send_recv_in: jax.Array | None = None,
    packed_rope: bool | None = None,
    head_major_out: bool = False,
):
    """Gather KV slots and attend query rows `[start_pos, min(end_pos, token_end))`."""
    resolved_vmem_limit = (
        _default_vmem_limit_bytes() if vmem_limit_bytes is None else vmem_limit_bytes
    )
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
    if send_world and not (lse_fill and q_pe is not None):
        raise ValueError("send_world needs lse_fill and split q")
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
        start_seq_idx: jax.Array | int,  # i32
        end_seq_idx: jax.Array,  # i32
        kernel_batch_size: int,
        lse: jax.Array | None = None,  # f32[max_num_tokens, num_q_heads, 1]
        is_dcp_mode: bool = False,
        live_views: tuple[jax.Array, jax.Array] | None = None,
    ):
        batch_size = kernel_batch_size
        out_rows, out_offset, out_carried = (
            (q.shape[0], 0, None) if out_piece is None else out_piece
        )
        if out_piece is not None:
            assert return_lse and not lse_fill and not preserve_suffix
            assert lse is not None and lse.shape[0] == out_rows
        lse_shape = (
            (q.shape[0], num_q_heads)
            if lse_fill
            else (None if lse is None else lse.shape)
        )
        end_seq_idx = jnp.maximum(start_seq_idx, end_seq_idx)
        grid = (cdiv(end_seq_idx - start_seq_idx, batch_size),)
        if lse_fill:
            grid = (jnp.maximum(grid[0], 1),)
        if send_exchange is not None:
            assert send_world and lse_fill and batch_size == q.shape[0], (
                send_world,
                lse_fill,
                batch_size,
                q.shape,
            )
            grid = (1,)
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
            elif send_recv_in is not None:
                in_specs.append(pl.BlockSpec(memory_space=pltpu.HBM))  # recv (in)
            out_specs = [out_specs, pl.BlockSpec(memory_space=pltpu.HBM)]
        if out_carried is not None:
            in_specs.append(pl.BlockSpec(memory_space=pltpu.HBM))  # piece out (in)

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

        if head_major_out:
            bo_double_buf = pltpu.VMEM(
                (1, batch_size, num_q_heads, q.shape[-1]),
                q.dtype,
            )
        else:
            bo_double_buf = bq_double_buf
        if send_world:
            bo_double_buf = pltpu.VMEM(
                (
                    2,
                    send_world,
                    (SEND_SLAB_ROW0 if send_exchange is not None else 0)
                    + batch_size * num_q_heads // send_world,
                    q.shape[-1],
                ),
                jnp.int16,
            )
        send_lanes = (
            send_layout_lanes(q.shape[0], num_q_heads, send_world) if send_world else 0
        )

        scratch_shapes = [
            bkv_nope_double_buf,
            bkv_rope_double_buf,
            bq_double_buf,
            bo_double_buf,  # Double buffering for output block.
        ]
        if return_lse:
            scratch_shapes.append(
                pltpu.VMEM(
                    (2, batch_size, send_lanes or num_q_heads)
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
        if lse_fill and send_world:
            scratch_shapes += [
                pltpu.VMEM(
                    (q.shape[0] // batch_size, batch_size, send_lanes), jnp.float32
                ),
                pltpu.SemaphoreType.DMA((3,))
                if send_exchange is not None
                else pltpu.VMEM((send_world, SEND_PAD_ROWS, q.shape[-1]), jnp.int16),
                pltpu.SemaphoreType.DMA(()),
            ]
        elif lse_fill:
            assert lse_shape is not None
            scratch_shapes += [
                pltpu.VMEM(lse_shape, jnp.float32),
                pltpu.SemaphoreType.DMA(()),
            ]
        if q_rope:
            assert q_rope_table is not None
            scratch_shapes.append(
                pltpu.VMEM((2, batch_size, *q_rope_table.shape[1:]), jnp.float32)
            )
        stage_levers = (
            tuple(group_levers) if live_views is not None else tuple(row_levers)
        )
        if split_q:
            pe_lanes = q_pe.shape[-1]
            scratch_shapes.append(
                pltpu.VMEM((2, batch_size, num_q_heads, pe_lanes), q_pe.dtype)
            )
        if "pair" in stage_levers or "quad" in stage_levers:
            pair_head_dim = (
                align_to(q.shape[-1] + (Q_ROPE_DIM if q_rope else q_pe.shape[-1]), 128)
                if split_q
                else q.shape[-1]
            )
            assert sparsecore_layout
            scratch_shapes.append(
                pltpu.VMEM(
                    (1, batch_size, cache_kv_nope.shape[1], pair_head_dim)
                    if "quad" in stage_levers
                    else (2, 2, cache_kv_nope.shape[1], pair_head_dim),
                    jnp.float8_e4m3fn,
                )
            )

        xchg_merge = bool(
            send_exchange is not None and len(send_exchange) > 4 and send_exchange[4]
        )
        if xchg_merge:
            assert send_recv_in is None, "merge mode has no receive buffer"
            scratch_shapes.append(
                pltpu.VMEM(
                    (
                        send_world,
                        SEND_SLAB_ROW0 + batch_size * num_q_heads // send_world,
                        q.shape[-1],
                    ),
                    jnp.int16,
                )
            )
        if head_major_out:
            if (
                return_lse
                or send_world
                or preserve_suffix
                or capped_windows is not None
                or out_piece is not None
            ):
                raise ValueError(
                    "head_major_out requires no return_lse/send_world/"
                    "preserve_suffix/capped_windows/out_piece"
                )
            scratch_shapes.append(
                pltpu.VMEM(
                    (2, num_q_heads, batch_size, q.shape[-1]),
                    q.dtype,
                )
            )
        bounds = [start_seq_idx, end_seq_idx]
        if preserve_suffix:
            bounds.append(token_end)
        if send_exchange is not None:
            # (rank, lane[, real_tokens]) ride the bounds prefetch: no extra operand.
            assert not preserve_suffix and send_peer is not None
            bounds += [send_peer[0], send_peer[1]]
            if xchg_merge or send_peer.shape[0] >= 3:
                bounds.append(send_peer[2])  # real tokens
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
            is_dcp_mode=is_dcp_mode,
            split_q=split_q,
            lse_lane_dense=lse_lane_dense,
            lse_mask_empty=lse_mask_empty,
            q_rope=q_rope,
            lse_fill=lse_fill,
            row_widths=tuple(row_widths) if capped_windows is not None else (),
            send_world=send_world,
            out_row_offset=start_pos if head_major_out else out_offset,
            out_in=out_carried is not None,
            send_exchange=send_exchange,
            recv_in=send_recv_in is not None,
            head_major_out=head_major_out,
        )
        if (
            live_views is None
            and row_levers
            and (
                (row_widths and capped_windows)
                or (not row_widths and capped_windows is None)
            )
        ):
            kernel_body = functools.partial(kernel_body, body_levers=tuple(row_levers))
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
        if send_exchange is not None:
            scope_name += "-xa2a"
        num_prefetch = len(scalar_prefetches)
        lse_operand = (
            num_prefetch
            + 3
            + int(split_q)
            + int(q_rope)
            + 2 * int(live_views is not None)
        )
        piece_aliases = {lse_operand: 1}
        if out_carried is not None:
            piece_aliases[lse_operand + 1] = 0
        if xchg_merge:
            lse_out_sd = jax.ShapeDtypeStruct(
                (q.shape[0] * num_q_heads // send_world, q.shape[-1]), jnp.int16
            )
        elif send_exchange is not None:
            s_shape = send_layout_shape(
                q.shape[0], num_q_heads, send_world, q.shape[-1]
            )
            lse_out_sd = jax.ShapeDtypeStruct(
                (int(np.prod(s_shape[:2])), q.shape[-1]), jnp.int16
            )
        elif send_world:
            lse_out_sd = jax.ShapeDtypeStruct(
                send_layout_shape(q.shape[0], num_q_heads, send_world, q.shape[-1]),
                jnp.int16,
            )
        else:
            lse_out_sd = (
                jax.ShapeDtypeStruct(lse_shape, jnp.float32)
                if lse_shape is not None
                else None
            )
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
                    vmem_limit_bytes=(
                        min(
                            resolved_vmem_limit,
                            send_exchange[2]
                            if len(send_exchange) > 2
                            else SEND_EXCHANGE_VMEM_LIMIT_BYTES,
                        )
                        if send_exchange is not None
                        else resolved_vmem_limit
                    ),
                    disable_bounds_checks=True,
                    # pyrefly: ignore[bad-argument-type]
                    **(
                        {"collective_id": SEND_EXCHANGE_COLLECTIVE_ID}
                        if send_exchange is not None
                        else {}
                    ),
                ),
                out_shape=(
                    [
                        jax.ShapeDtypeStruct(
                            shape=(out_rows, *q.shape[1:]), dtype=q.dtype
                        ),
                        lse_out_sd,
                    ]
                    if return_lse
                    else jax.ShapeDtypeStruct(
                        shape=(num_q_heads, chunk_size, q.shape[-1]),
                        dtype=q.dtype,
                    )
                    if head_major_out
                    else jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
                ),
                input_output_aliases=(
                    {}
                    if head_major_out
                    else piece_aliases
                    if out_piece is not None
                    else {num_prefetch: 0, lse_operand: 1}
                    if lse_fill and send_recv_in is not None
                    else {num_prefetch: 0}
                    if lse_fill
                    else {num_prefetch: 0, lse_operand: 1}
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
        roles = ["q", "kv", "kv"]
        if split_q:
            operands.append(q_pe)
            roles.append("q_pe")
        if q_rope and q_rope_table is not None:
            operands.append(q_rope_table)
            roles.append("tab")
        if live_views is not None:
            operands += list(live_views)
            roles += ["kv"] * len(live_views)
        if return_lse and not lse_fill and lse is not None:
            operands.append(lse)
            roles.append("lse")
        elif return_lse and send_recv_in is not None:
            operands.append(send_recv_in)
            roles.append("recv")
        if out_carried is not None:
            operands.append(out_carried)
            roles.append("out")
        if send_exchange is not None:
            pin = send_exchange[3] if len(send_exchange) > 3 else True
            operands = [
                pltpu.with_memory_space_constraint(x, pltpu.HBM)
                if pin is True or (pin and r in pin)
                else x
                for x, r in zip(operands, roles)
            ]
        return kernel(*scalar_prefetches, *operands)

    indices = None if physical_indices is None else physical_indices[start_pos:end_pos]

    tc_num_row_subchunks = 32
    dcp_gather_atoms = sparsecore_atoms_per_batch
    gather_step = native_sc_gather.OCCURRENCES_PER_ATOM
    if (
        is_dcp_mode
        and capped_windows is None
        and topk >= gather_step
        and topk % gather_step == 0
    ):
        tc_steps_per_window = largest_divisor(topk // gather_step, 4)
        tc_num_row_subchunks = tc_steps_per_window * 4
        dcp_gather_atoms = largest_divisor(
            topk // gather_step, sparsecore_atoms_per_batch
        )
        gather_window_size = (
            dcp_gather_atoms * gather_step
            if sparsecore_layout
            else tc_steps_per_window * gather_step
        )
        window_start = (
            jnp.arange(topk, dtype=jnp.int32) // gather_window_size
        ) * gather_window_size
        if indices is not None:
            indices = jnp.where(
                window_start[None, :] >= kv_lens[start_pos:end_pos, None],
                dsa_gather.SKIP_INDEX,
                indices,
            )

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
    attn_batch_size, attn_batch_end = kernel_batch_size, batch_end
    if "step8" in row_levers and small_chunk:
        attn_batch_size = largest_divisor(chunk_size, DECODE_STEP8_TOKENS)
        attn_batch_end = start_pos + cdiv(valid_rows, attn_batch_size) * attn_batch_size
    num_valid_indices = jnp.maximum(0, batch_end - start_pos) * topk

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
            live_kwargs: dict[str, Any] = dict(
                live_vec_offset=chunk_size * topk, linear_indices=True
            )
        elif group_tab is not None:
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
            if capped_live_rows:
                window_start = topk_cap * jnp.arange(capped_windows, dtype=jnp.int32)
                live_slots = jnp.clip(
                    kv_lens[start_pos:end_pos][None, :] - window_start[:, None],
                    0,
                    topk_cap,
                )
                live_slots = jnp.where(
                    jnp.arange(chunk_size) < live_tokens, live_slots, 0
                )
                live_kwargs["row_live_atoms"] = (
                    (live_slots + GATHER_ATOM_ROWS - 1) // GATHER_ATOM_ROWS
                ).reshape(-1)
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
                    chunk_size * topk // block,
                    kernel_batch_size,
                    GATHER_ATOM_ROWS,
                    128,
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
        eff_packed_rope = (
            (cache_kv_nope.shape[1] == 8 * cache_kv_rope.shape[1])
            if packed_rope is None
            else bool(packed_rope)
        )
        compact_rope = (
            eff_packed_rope and not is_dcp_mode and topk % GATHER_ATOM_ROWS == 0
        )
        rope_out_packing = (
            native_sc_gather.COMPACT_ROPE_PACKING
            if compact_rope
            else native_sc_gather.ROPE_PACKING
        )
        assert indices is not None
        gathered_nope_buffer, gathered_rope_buffer = (
            native_sc_gather.dsa_gather_native_sc(
                cache_kv_nope,
                cache_kv_rope,
                indices,
                out_size=chunk_size * topk,
                atoms_per_batch=(dcp_gather_atoms if is_dcp_mode else gather_atoms),
                wait_mode=sparsecore_gather_wait_mode,
                serve_pipelined=sparsecore_gather_pipelined,
                linear_indices=not is_dcp_mode,
                **({"is_dcp_mode": True} if is_dcp_mode else {}),
                packed_rope=packed_rope,
                compact_rope=compact_rope,
                **live_kwargs,
            )
        )
        gathered_nope_buffer = gathered_nope_buffer.reshape(chunk_size, topk, 128)
        gathered_rope_buffer = gathered_rope_buffer.reshape(
            chunk_size, topk // rope_out_packing, 128
        )
    else:
        assert indices is not None
        gathered_nope_buffer, gathered_rope_buffer = dsa_gather.dsa_gather(
            cache_kv_nope,
            cache_kv_rope,
            indices.reshape(-1),
            num_valid_indices=num_valid_indices,
            **(
                {"num_row_subchunks": tc_num_row_subchunks, "is_dcp_mode": True}
                if is_dcp_mode
                else {}
            ),
        )
        gathered_nope_buffer = gathered_nope_buffer.reshape(
            chunk_size,
            topk * kv_cache_utils.WORD_BYTES,
            kv_cache_utils.TILE_LANE_BYTES,
        )
        gathered_rope_buffer = gathered_rope_buffer.reshape(chunk_size, topk, -1)
    result = run_mla_kernel(
        q,
        gathered_nope_buffer,
        gathered_rope_buffer,
        kv_lens,
        start_seq_idx=start_pos,
        end_seq_idx=attn_batch_end,
        kernel_batch_size=attn_batch_size,
        lse=lse,
        is_dcp_mode=is_dcp_mode,
        live_views=live_views,
    )
    if kv_write is not None:
        return result, (cache_kv_nope, cache_kv_rope)
    return result
