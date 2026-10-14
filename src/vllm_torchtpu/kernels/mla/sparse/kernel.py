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
):
    # Split-q mode (`split_q`): `q_hbm_ref` holds q_nope only and the rope
    # part arrives as a separate input `q_pe_hbm_ref` (first of `refs`), with
    # its own VMEM double buffer (last scratch ref). The two are joined and
    # zero-padded in VMEM, so the scores see exactly the operand the caller's
    # concat used to build. The output is q_nope's width and aliases it.
    #
    # Lane-dense lse (`lse_lane_dense`): `lse_hbm_ref` is f32[tokens, heads]
    # and the blse buffer f32[2, bsz, heads]; each token's [heads, 1] column
    # is moved to a row exactly (a masked max) before it is stored.

    # Paths with split_q and lase_lance_dense should be the default but kept
    # as False to maintain compatibility with existing code paths. Some clean up
    # work can be done later.
    #
    # Capped mode (`capped_windows` = W set, SparseCore layout only): the
    # gathered buffers hold W windows of `bkv_sz` rows per token, window-major
    # (`[W * chunk, bkv_sz, ...]`), and the pipelined body attends window 0
    # exactly like the uncapped path. Tokens with more than `bkv_sz` live rows
    # are recomputed over all their windows by an online-softmax fix-up with
    # double-buffered window DMAs (two-slot scratch trails the usual scratch).
    q_pe_hbm_ref = None
    bq_pe_x2_ref = None
    if split_q:
        q_pe_hbm_ref = refs[0]  # [max_num_tokens, num_q_heads, rope_dim]
        bq_pe_x2_ref = refs[-1]  # [2, bsz, num_q_heads, rope_dim]
        refs = refs[1:-1]
    fix_refs = None
    if capped_windows is not None:
        assert sparsecore_layout
        fix_refs = refs[-6:]
        refs = refs[:-6]
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

    assert q_hbm_ref.shape == o_hbm_ref.shape

    num_tokens, num_q_heads, head_dim = q_hbm_ref.shape
    # Width of the stored output (q_nope's in split-q mode, else head_dim).
    o_dim = head_dim
    if split_q:
        assert q_pe_hbm_ref is not None
        head_dim = align_to(head_dim + q_pe_hbm_ref.shape[-1], 128)
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
        assert kv.shape == (bkv_sz, head_dim)

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
    ):
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

    def _fetch_bkv_batch(seq_idx_start, bkv_sem_idx, *, wait=False):
        sem_nope = sems.at[0, bkv_sem_idx]
        sem_rope = sems.at[3, bkv_sem_idx]

        bkv_nope_vmem_ref = bkv_nope_x2_ref.at[bkv_sem_idx, :]
        bkv_rope_vmem_ref = bkv_rope_x2_ref.at[bkv_sem_idx, :]

        # The index into cache_kv_hbm_ref should be relative to the current
        # chunk.
        page_idx_start = seq_idx_start - start_seq_idx
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
        else:
            _async_copy(src=bq_vmem_ref, dst=bq_vmem_ref, sem=sem, wait=True)
            if split_q:
                bq_pe_vmem_ref = bq_pe_x2_ref.at[bq_sem_idx, :]
                _async_copy(src=bq_pe_vmem_ref, dst=bq_pe_vmem_ref, sem=sem, wait=True)

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
        assert blse_x2_ref is not None and lse_hbm_ref is not None
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

    def start_fetch_bkv_batch(seq_idx_start, bkv_sem_idx):
        return _fetch_bkv_batch(seq_idx_start, bkv_sem_idx)

    def wait_fetch_bkv_batch(seq_idx_start, bkv_sem_idx):
        return _fetch_bkv_batch(seq_idx_start, bkv_sem_idx, wait=True)

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
        if split_q:
            # The caller's concat + zero pad, rebuilt in VMEM.
            q = jnp.concatenate([q, bq_pe_x2_ref.at[bq_sem_idx, batch_idx][...]], -1)
            if q.shape[-1] < head_dim:
                q = jnp.pad(q, ((0, 0), (0, head_dim - q.shape[-1])))
        return q

    def store_blse(bo_sem_idx, batch_idx, lse_col):
        """Stores one token's f32[num_q_heads, 1] log-sum-exp column."""
        assert blse_x2_ref is not None
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

    def load_bkv(bkv_sem_idx, batch_idx):
        return unpack_bkv(
            bkv_nope_x2_ref.at[bkv_sem_idx, batch_idx],
            bkv_rope_x2_ref.at[bkv_sem_idx, batch_idx],
            lambda: kv_lens_ref[batch_start_seq_idx + batch_idx],
        )

    def unpack_bkv(bkv_nope_ref, bkv_rope_ref, get_kv_len):
        bkv_nope = bkv_nope_ref[...]
        if sparsecore_layout:
            # Native SC returns one uint32[128] row per token. Bitcasting
            # expands the packed byte lanes before folding them into the
            # logical 512-byte NOPE vector.
            bkv_nope = pltpu.bitcast(bkv_nope, jnp.uint8)
        bkv_nope = bkv_nope.reshape(bkv_sz, nope_dim)
        bkv_nope = pltpu.bitcast(bkv_nope, jnp.float8_e4m3fn)

        bkv_rope = bkv_rope_ref[...]
        if sparsecore_layout:
            # The native gather's delta-swap produces four-token grouped
            # uint32[128] rows. Expanding their byte lanes recovers one
            # contiguous 128-byte row per selected token.
            bkv_rope = pltpu.bitcast(bkv_rope, jnp.uint8)
            bkv_rope = bkv_rope.reshape(bkv_sz, 128)
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
            start_fetch_bkv_batch(next_seq_idx, next_bi_sem_idx)

        bo_sem_idx = sem_ids_ref[1]
        sem_ids_ref[1] = lax.select(bo_sem_idx == 0, 1, 0)

        prev_p = None
        prev_bkv = None
        prev_l_sum = None
        prev_m = None
        prev_out = None

        def write_blse(batch_idx, m, l_sum):
            # log-sum-exp in the *unshifted* frame, so a peer holding a
            # disjoint slice of the same token's top-k can merge with
            # o = sum_r softmax(lse_r) * o_r without knowing our max.
            if not return_lse:
                return
            store_blse(bo_sem_idx, batch_idx, m + jnp.log(l_sum))

        # Wait for cur blocks if not ready yet
        wait_fetch_bq_batch(batch_start_seq_idx, bi_sem_idx)
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

        for batch_idx in range(batch_size):
            kv_len = kv_lens_ref[batch_start_seq_idx + batch_idx]
            bkv = load_bkv(bi_sem_idx, batch_idx)
            bq = load_bq(bi_sem_idx, batch_idx)

            if prev_out is not None:
                # Artificial dependency to force MXU/VPU interleaving by limiting
                # LLO's QK runahead
                # We use jnp.where to prevent XLA from optimizing the dependency away.
                # prev_out won't be inf in practice.
                bq = jnp.where(prev_out == jnp.inf, prev_out, bq)

            p, l_sum, m = attention_step1_qk_softmax(bq, bkv, kv_len)

            if prev_p is not None:
                assert prev_bkv is not None
                assert prev_l_sum is not None
                out = attention_step2_pv(prev_p, prev_bkv, prev_l_sum)

                # Store output from acc to bo.
                bo_x2_ref.at[bo_sem_idx, batch_idx - 1][...] = out[:, :o_dim]
                write_blse(batch_idx - 1, prev_m, prev_l_sum)
                prev_out = out

            prev_p = p
            prev_bkv = bkv
            prev_l_sum = l_sum
            prev_m = m

        # end of pipelining loop
        out = attention_step2_pv(prev_p, prev_bkv, prev_l_sum)
        bo_x2_ref.at[bo_sem_idx, batch_size - 1][...] = out[:, :o_dim]
        write_blse(batch_size - 1, prev_m, prev_l_sum)
        if capped_windows is not None:
            fixup_capped_overflow(bi_sem_idx, bo_sem_idx)
        start_send_bo_batch(batch_start_seq_idx, bo_sem_idx)
        if return_lse:
            _send_blse_batch(batch_start_seq_idx, bo_sem_idx)

    ### ------- Kernel start ------- ###

    @pl.when(batch_start_seq_idx == start_seq_idx)
    def prologue():
        start_fetch_bq_batch(batch_start_seq_idx, 0)
        start_fetch_bkv_batch(batch_start_seq_idx, 0)

    process()

    @pl.when(batch_end_seq_idx == end_seq_idx - 1)
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
    prepared_kv_lens: jax.Array | None = None,
    prepared_token_end: jax.Array | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """MLA Ragged paged attention that supports mixed prefill and decode.

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
      prepared_kv_lens: if set, `topk_indices` already holds flat physical
        cache rows (`prepare_gather_indices` output) and this is the matching
        per-token valid count; `page_indices`, `cu_q_lens` and `distribution`
        are then ignored. Lets a caller attend an arbitrary slice of a batch's
        tokens (token-sharded prefill).
      prepared_token_end: with `prepared_kv_lens`, the number of leading rows
        to attend; rows past it are padding and are skipped.

    Returns:
      The output of attention, or `(output, lse)` when `return_lse` is set.
      `lse` is f32[max_num_tokens, actual_num_q_heads].
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

    num_chunks = cdiv(q.shape[0], gather_and_attention_chunk_size)
    token_end = cu_q_lens[distribution[2]]
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
    if prepared_kv_lens is not None:
        assert prepared_token_end is not None
        kv_lens, physical_indices = prepared_kv_lens, topk_indices
        token_end = prepared_token_end
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
    lse = (
        jnp.full(
            (q.shape[0], num_q_heads)
            if lse_lane_dense
            else (q.shape[0], num_q_heads, 1),
            jnp.finfo(jnp.float32).min,
            dtype=jnp.float32,
        )
        if return_lse
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
            token_end=token_end,
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
                None
                if window_indices_all is None
                else window_indices_all[
                    i * chunk_window_rows : (i + 1) * chunk_window_rows
                ]
            ),
            q_pe=q_pe,
            lse_lane_dense=lse_lane_dense,
        )
        if return_lse:
            q, lse = result
        else:
            q = result

    # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    out = prepare_outputs(q, actual_num_q_heads, actual_head_dim)
    if return_lse:
        if lse_lane_dense:
            return out, cast(jax.Array, lse)[:, :actual_num_q_heads]
        return out, cast(jax.Array, lse)[:, :actual_num_q_heads, 0]
    return out


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
    q_pe: jax.Array | None = None,
    lse_lane_dense: bool = False,
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
    """
    return_lse = lse is not None
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
        head_dim = align_to(head_dim + q_pe.shape[-1], 128)
    chunk_size = end_pos - start_pos
    capped_windows = None if topk_cap is None else topk // topk_cap

    def run_mla_kernel(
        q: jax.Array,  # [max_num_tokens, num_q_heads, head_dim]
        # [num_tokens, topk * WORD_BYTES, TILE_LANE_BYTES] (gathered raw tiles)
        cache_kv_nope: jax.Array,
        cache_kv_rope: jax.Array,  # [num_tokens, topk, rope_dim] (gathered)
        kv_lens: jax.Array,  # i32[max_num_tokens]
        start_seq_idx: int,
        end_seq_idx: jax.Array,  # i32
        kernel_batch_size: int,
        lse: jax.Array | None = None,  # f32[max_num_tokens, num_q_heads, 1]
    ):
        batch_size = kernel_batch_size
        end_seq_idx = jnp.maximum(start_seq_idx, end_seq_idx)
        grid = (cdiv(end_seq_idx - start_seq_idx, batch_size),)
        in_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),  # q
            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv_nope
            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv_rope
        ]
        if split_q:
            in_specs.append(pl.BlockSpec(memory_space=pltpu.HBM))  # q_pe

        out_specs = pl.BlockSpec(memory_space=pltpu.HBM)  # o
        # Each chunk writes only its own token rows, so -- exactly like `q` --
        # the lse buffer is threaded through as an aliased in/out to carry
        # earlier chunks' rows forward.
        if return_lse:
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
                        jax.ShapeDtypeStruct(
                            shape=cast(jax.Array, lse).shape,
                            dtype=cast(jax.Array, lse).dtype,
                        ),
                    ]
                    if return_lse
                    else jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
                ),
                input_output_aliases=(
                    {
                        num_prefetch: 0,
                        num_prefetch + 3 + int(split_q): 1,
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
        if return_lse:
            operands.append(cast(jax.Array, lse))
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
        if window_indices is None:
            assert indices is not None
            window_indices = (
                indices.reshape(chunk_size, capped_windows, topk_cap)
                .transpose(1, 0, 2)
                .reshape(capped_windows * chunk_size, topk_cap)
            )
        gathered_nope_buffer, gathered_rope_buffer = (
            native_sc_gather.dsa_gather_native_sc(
                cache_kv_nope,
                cache_kv_rope,
                window_indices,
                out_size=chunk_size * topk,
                atoms_per_batch=gather_atoms,
                wait_mode=sparsecore_gather_wait_mode,
                serve_pipelined=sparsecore_gather_pipelined,
                num_live_indices=num_live_capped,
                live_multiple=kernel_batch_size * topk_cap,
            )
        )
        gathered_nope_buffer = gathered_nope_buffer.reshape(
            capped_windows * chunk_size, topk_cap, 128
        )
        gathered_rope_buffer = gathered_rope_buffer.reshape(
            capped_windows * chunk_size, topk_cap // native_sc_gather.ROPE_PACKING, 128
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
    return run_mla_kernel(
        q,
        gathered_nope_buffer,
        gathered_rope_buffer,
        kv_lens,
        start_seq_idx=start_pos,
        end_seq_idx=batch_end,
        kernel_batch_size=kernel_batch_size,
        lse=lse,
    )
