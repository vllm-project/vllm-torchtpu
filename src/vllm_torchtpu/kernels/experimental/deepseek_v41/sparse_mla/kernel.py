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
"""TPU7x Pallas kernels for DeepSeek-V4.1 Sparse MLA + SWA and 2-bank KV gather.

Architecture & Kernel Design:
1. Sparse FP8 KV & BF16 RoPE Gather (``gather_sparse_fp8_kv``):
   - VMEM Page-ID Resolution (``_gather_page_ids`` / ``dsv41_gather_page_ids``):
     Maps ragged token indices to sequence IDs via ``_token_seq_ids``, pads
     ``topk_indices`` to a multiple of ``SC_GATHER_TOPK = 256``, and resolves
     ``block_tables[seq_ids, logical_page]`` inside VMEM over ``ROW_WORDS = 128``
     page chunks via ``jax.lax.fori_loop`` + ``jnp.take_along_axis`` (``0``
     register spills, ``0`` SMEM DMAs).
   - 1-Bank / 2-Bank SparseCore Gather (``_gather_bank``):
     Dispatches ``csa_gather`` on ``kv_pool_lo`` / ``rope_pool_lo`` (and
     ``kv_pool_hi`` / ``rope_pool_hi`` when 2-bank pools are supplied), routing
     invalid indices to deterministic prime-strided fallback slots
     (``SC_FALLBACK_STRIDE_PRIME = 104_729``) to prevent SparseCore bank conflicts.
   - RoPE Quarter Unpacking (``_unpack_csa_rope_to_quarters``):
     Unpacks SparseCore ``csa_gather`` interleaved RoPE pairs into ``32``-word
     token quarters packed 4 tokens per ``128``-word row.
2. Pipelined Sparse MLA + SWA Attention (``sparse_mla_swa_attention``):
   - Compact 2D SWA Window Extraction (``_extract_swa_windows_2d``):
     When ``swa_kv`` is passed in compact 2D right-aligned layout
     ``bf16[num_seqs * swa_width + num_tokens, 512]``, assembles per-token causal
     windows inside each ``jax.lax.scan`` chunk via async VMEM DMAs without
     materializing ``bf16[num_tokens, swa_width, 512]`` in HBM.
   - Tile Dequantization & Online Softmax (``_sparse_mla_swa_tile``):
     Streams ``COMBINED_ROWS = 336`` rows per tile through ``pltpu.emit_pipeline``,
     unpacking ``SWA_SUB_K = 128`` BF16 SWA keys or dequantizing ``SUB_K = 256``
     sparse keys (``NOPE_DIM = 448`` FP8 E4M3FN with ``NOPE_SCALE_PERIOD = 64``
     lane-periodic UE8M0 scales + ``ROPE_DIM = 64`` BF16 RoPE) and updating
     online softmax state initialized from ``attention_sinks``.
   - HBM Transient Bounding & VMEM Capacity Cost Model (``_max_vmem_token_heads``):
     Each token-head requires ``2,048 B`` BF16 query/output + ``3,072 B`` FP32
     online-softmax state (``STATE_COLS = 768``) = ``5,120 B``. Using ``5/8`` of
     ``64 MiB`` TPU7x VMEM (``40 MiB``) yields ``MAX_VMEM_TOKEN_HEADS = 8,192``
     token-heads per ``jax.lax.scan`` chunk, keeping VMEM under ``55%`` and
     bounding packed chunk HBM transients to ``<= 168 MiB`` at ``K = 2,048``.
"""

from __future__ import annotations

import contextlib
import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.core_attention.csa_gather import csa_gather

# TPU7x vector register / DMA tile geometry (8 sublanes x 128 lanes) and CSA layout.
TPU7X_GENERATION = 7
SUBLANES = 8
ROW_WORDS = 128
WORD_BYTES = 4
BF16_BYTES = 2
FP32_BYTES = 4
BF16_PER_WORD = WORD_BYTES // BF16_BYTES
BYTE_BITS = 8
HALF_WORD_BITS = 16
TOP_BYTE_SHIFT_BITS = 24
BYTE_MASK = 0xFF
LOW_HALF_WORD_MASK = 0xFFFF
HIGH_HALF_WORD_MASK = 0xFFFF0000

NOPE_DIM = 448
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM
WORDS_PER_HEAD_DIM = HEAD_DIM // BF16_PER_WORD
HEAD_SLABS_PER_ROW = HEAD_DIM // ROW_WORDS
NOPE_SCALE_PERIOD = 64
NOPE_FULL_WORD_SLABS = NOPE_DIM // ROW_WORDS
NOPE_FULL_SLAB_CHANNELS = NOPE_FULL_WORD_SLABS * ROW_WORDS
NOPE_TAIL_LANES = NOPE_DIM - NOPE_FULL_SLAB_CHANNELS
ROPE_WORDS = ROPE_DIM // BF16_PER_WORD
ROPE_TOKENS_PER_ROW = ROW_WORDS // ROPE_WORDS
ROPE_LOW_SHIFT_BASE = 2 * ROPE_WORDS
ROPE_HIGH_LANE_START = ROW_WORDS - ROPE_WORDS

SC_GATHER_TOPK = 256
SC_FALLBACK_STRIDE_PRIME = 104_729
INVALID_INDEX = -1

SUB_K = 256
SWA_SUB_K = 128
SUB_ROPE = SUB_K // ROPE_TOKENS_PER_ROW
VALID_SUB_TILES = SUB_K // ROW_WORDS
VALID_ROWS = VALID_SUB_TILES * SUBLANES
# VMEM tile capacity: COMBINED_ROWS = 256 + 64 + 16 = 336 rows x 512 B = 168 KiB/tile
# STATE_COLS = 512 + 128 + 128 = 768 fp32 cols (3072 B/token-head) + 2048 B q/out =
# 5120 B/token-head -> 40 MiB (5/8 of 64 MiB VMEM) / 5120 B = 8192 token-heads.
COMBINED_ROWS = SUB_K + SUB_ROPE + VALID_ROWS
STATE_MAX_COL_START = HEAD_DIM
STATE_SUM_COL_START = HEAD_DIM + ROW_WORDS
STATE_COLS = HEAD_DIM + 2 * ROW_WORDS
PER_TOKEN_HEAD_VMEM_BYTES = 2 * HEAD_DIM * BF16_BYTES + STATE_COLS * FP32_BYTES
VMEM_TOKEN_HEAD_NUMERATOR = 5
VMEM_TOKEN_HEAD_DENOMINATOR = 8
MAX_VMEM_TOKEN_HEADS = 8192
# NOTE(contract): SCORE_MIN = -1e30 assumes >= 1 valid key per query row (guaranteed in
# production by SWA self-key + CED replay_start filtering) or finite attn_sink > -1e30.
SCORE_MIN = -1e30

_HBM_SPEC = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
_VMEM_SPEC = pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM)
_bitcast = jax.lax.bitcast_convert_type


class _AttentionTileConfig(NamedTuple):
    """Static compile-time parameters for ``_sparse_mla_swa_tile``."""

    num_swa_blocks: int
    num_blocks: int
    sm_scale: float
    enable_named_scope: bool


def _expect_arr(
    array: jax.Array,
    name: str,
    spec: Any,
    dtype: Any = jnp.int32,
    is_valid: bool = True,
) -> None:
    """Validate ``array`` against shape/ndim ``spec``, ``dtype``, and ``is_valid``."""
    shape_matches = (
        array.shape == spec if isinstance(spec, tuple) else array.ndim == spec
    )
    if not (shape_matches and array.dtype == dtype and is_valid):
        dtype_name = jnp.dtype(dtype).name
        raise ValueError(
            f"Expected {name} {dtype_name}({spec}), got {array.dtype}{array.shape}."
        )


def _check_pool_pair(
    kv_pool: jax.Array, rope_pool: jax.Array, page_size: int, bank_tag: str
) -> int:
    """Validate a ``(kv_pool, rope_pool)`` bank pair and return ``num_pages``."""
    kv_valid = (
        kv_pool.shape[0] >= 1 and kv_pool.shape[1:] == (page_size, ROW_WORDS)
        if kv_pool.ndim == 3
        else False
    )
    _expect_arr(kv_pool, f"kv_pool_{bank_tag}", 3, is_valid=kv_valid)
    _expect_arr(
        rope_pool,
        f"rope_pool_{bank_tag}",
        (kv_pool.shape[0], page_size // ROPE_TOKENS_PER_ROW, ROW_WORDS),
    )
    return kv_pool.shape[0]


def _named_scope(enable: bool, name: str) -> contextlib.AbstractContextManager[Any]:
    """Return ``jax.named_scope(name)`` when ``enable`` is True, else nullcontext."""
    return jax.named_scope(name) if enable else contextlib.nullcontext()


def _check_tpu7x() -> None:
    """Fail fast unless executing on TPU7x (Ironwood, generation 7, 128 lanes).

    Raises:
        RuntimeError: If the active TPU device or abstract mesh is not generation 7
            with 128 lanes and 8 sublanes.
    """
    info = pltpu.get_tpu_info()
    if (
        info.generation != TPU7X_GENERATION
        or info.num_lanes != ROW_WORDS
        or info.num_sublanes != SUBLANES
    ):
        raise RuntimeError(f"DeepSeek-V4.1 sparse MLA kernels require TPU7x: {info}.")


def _max_vmem_token_heads() -> int:
    """Derive per-chunk token-head cap from TPU7x VMEM capacity.

    Uses ``5/8`` of ``vmem_capacity_bytes`` (``40 MiB`` on ``64 MiB`` TPU7x VMEM)
    divided by ``PER_TOKEN_HEAD_VMEM_BYTES = 5,120 B`` per token-head (``2,048 B``
    BF16 query/output + ``3,072 B`` FP32 ``state_scr_ref`` across ``STATE_COLS = 768``),
    capped at ``MAX_VMEM_TOKEN_HEADS = 8,192`` token-heads.

    Returns:
        Maximum ``(tokens * padded_heads)`` product per ``pl.pallas_call`` chunk.
    """
    vmem_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes)
    budget = (
        (vmem_bytes * VMEM_TOKEN_HEAD_NUMERATOR) // VMEM_TOKEN_HEAD_DENOMINATOR
    ) // PER_TOKEN_HEAD_VMEM_BYTES
    return max(SUBLANES, min(MAX_VMEM_TOKEN_HEADS, budget))


def _pallas(
    kernel_fn: Any,
    out_shape: Any,
    grid_spec: Any,
    interpret: bool,
    name: str,
    *operands: Any,
) -> Any:
    """Invoke ``pl.pallas_call`` with TPU compiler parameters from ``grid_spec``."""
    has_grid = bool(grid_spec.grid)
    dim_semantics = (pltpu.ARBITRARY,) if has_grid else ()
    compiler_params = pltpu.CompilerParams(
        dimension_semantics=dim_semantics, disable_bounds_checks=has_grid
    )
    return pl.pallas_call(
        kernel_fn,
        out_shape=out_shape,
        compiler_params=compiler_params,
        grid_spec=grid_spec,
        interpret=interpret,
        name=name,
    )(*operands)


def _token_seq_ids(
    num_tokens: int, cu_q_lens: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Map token indices ``0..num_tokens-1`` to ``(valid_token_mask, seq_ids)``."""
    token_iota = jnp.arange(num_tokens, dtype=jnp.int32)[:, None]
    in_seq = (token_iota >= cu_q_lens[:-1][None, :]) & (
        token_iota < cu_q_lens[1:][None, :]
    )
    return jnp.any(in_seq, axis=1), jnp.argmax(in_seq, axis=1).astype(jnp.int32)


def _gather_page_ids(
    block_tables: jax.Array,
    clipped_logical_pages: jax.Array,
    seq_ids: jax.Array,
    interpret: bool,
) -> jax.Array:
    """Translate logical page IDs via per-token VMEM ``fori_loop`` lookup."""
    num_tokens, padded_topk = clipped_logical_pages.shape
    num_page_chunks = (block_tables.shape[1] + ROW_WORDS - 1) // ROW_WORDS
    padded_tables = jnp.pad(
        block_tables,
        ((0, 0), (0, num_page_chunks * ROW_WORDS - block_tables.shape[1])),
    )
    padded_tokens = ((num_tokens + SUBLANES - 1) // SUBLANES) * SUBLANES
    token_tables_padded = jnp.pad(
        padded_tables[seq_ids], ((0, padded_tokens - num_tokens), (0, 0))
    )
    tables_4d = token_tables_padded.reshape(
        -1, SUBLANES, num_page_chunks, ROW_WORDS
    ).transpose(0, 2, 1, 3)

    def _gather_page_ids_kernel(
        table_ref: Any, logical_page_ref: Any, out_ref: Any
    ) -> None:
        target_pages = logical_page_ref[...]

        def _accumulate_chunk(chunk_idx: jax.Array, acc_pages: jax.Array) -> jax.Array:
            chunk_offsets = target_pages - chunk_idx * ROW_WORDS
            chunk_hits = jnp.take_along_axis(
                table_ref[0, chunk_idx, :, :],
                jnp.clip(chunk_offsets, 0, ROW_WORDS - 1),
                axis=1,
            )
            in_chunk = (chunk_offsets >= 0) & (chunk_offsets < ROW_WORDS)
            return jnp.where(in_chunk, chunk_hits, acc_pages)

        out_ref[...] = jax.lax.fori_loop(
            0,
            num_page_chunks,
            _accumulate_chunk,
            jnp.zeros_like(target_pages),
        )

    table_spec = pl.BlockSpec(
        (1, num_page_chunks, SUBLANES, ROW_WORDS),
        lambda token_tile_idx: (token_tile_idx, 0, 0, 0),
    )
    page_spec = pl.BlockSpec(
        (SUBLANES, padded_topk), lambda token_tile_idx: (token_tile_idx, 0)
    )
    grid_spec = pltpu.PrefetchScalarGridSpec(
        0,
        (padded_tokens // SUBLANES,),
        [table_spec, page_spec],
        page_spec,
    )
    out_type = jax.ShapeDtypeStruct((padded_tokens, padded_topk), jnp.int32)
    padded_logical_pages = jnp.pad(
        clipped_logical_pages, ((0, padded_tokens - num_tokens), (0, 0))
    )
    gathered_pages = _pallas(
        _gather_page_ids_kernel,
        out_type,
        grid_spec,
        interpret,
        "dsv41_gather_page_ids",
        tables_4d,
        padded_logical_pages,
    )
    return gathered_pages[:num_tokens]


def _unpack_csa_rope_to_quarters(
    rope_sc: jax.Array,
    valid_mask: jax.Array,
    num_tokens: int,
    padded_topk: int,
) -> jax.Array:
    """Unpack SparseCore ``csa_gather`` interleaved RoPE pairs to 32-word quarters."""
    rope_u32 = _bitcast(rope_sc, jnp.uint32).reshape(
        -1,
        SC_GATHER_TOPK // ROPE_TOKENS_PER_ROW,
        BF16_PER_WORD,
        BF16_PER_WORD,
        ROPE_WORDS,
    )
    low_words = rope_u32[:, :, :, 0, :]
    high_words = rope_u32[:, :, :, 1, :]
    even_words = (low_words & jnp.uint32(LOW_HALF_WORD_MASK)) | (
        high_words << jnp.uint32(HALF_WORD_BITS)
    )
    odd_words = (low_words >> jnp.uint32(HALF_WORD_BITS)) | (
        high_words & jnp.uint32(HIGH_HALF_WORD_MASK)
    )
    quarters = jnp.stack([even_words, odd_words], axis=3).transpose(0, 2, 1, 3, 4)
    quarters_i32 = _bitcast(quarters, jnp.int32).reshape(
        num_tokens,
        padded_topk // ROPE_TOKENS_PER_ROW,
        ROPE_TOKENS_PER_ROW,
        ROPE_WORDS,
    )
    masked_quarters = jnp.where(
        valid_mask.reshape(
            num_tokens,
            padded_topk // ROPE_TOKENS_PER_ROW,
            ROPE_TOKENS_PER_ROW,
            1,
        ),
        quarters_i32,
        0,
    )
    return masked_quarters.reshape(
        num_tokens, padded_topk // ROPE_TOKENS_PER_ROW, ROW_WORDS
    )


def _gather_bank(
    kv_pool: jax.Array,
    rope_pool: jax.Array,
    flat_indices: jax.Array,
    interpret: bool,
) -> tuple[jax.Array, jax.Array]:
    """Gather one HBM bank via SparseCore ``csa_gather`` (or CPU interpret fallback)."""
    num_tokens, padded_topk = flat_indices.shape
    indices_1d = flat_indices.reshape(-1)
    num_rows = kv_pool.shape[0] * kv_pool.shape[1]
    valid_mask = (indices_1d >= 0) & (indices_1d < num_rows)
    fallback_indices = (
        jnp.arange(indices_1d.size, dtype=jnp.int32) * SC_FALLBACK_STRIDE_PRIME
    ) % num_rows
    safe_indices = jnp.where(valid_mask, indices_1d, fallback_indices)
    if not interpret:
        nope_sc, rope_sc = csa_gather(
            kv_pool, rope_pool, safe_indices, top_k=SC_GATHER_TOPK
        )
        raw_nope = nope_sc.reshape(num_tokens, padded_topk, ROW_WORDS)
        out_rope = _unpack_csa_rope_to_quarters(
            rope_sc, valid_mask, num_tokens, padded_topk
        )
    else:
        raw_nope = kv_pool.reshape(num_rows, ROW_WORDS)[safe_indices].reshape(
            num_tokens, padded_topk, ROW_WORDS
        )
        rope_quarters = rope_pool.reshape(num_rows, ROPE_WORDS)[safe_indices].reshape(
            num_tokens, -1, ROPE_TOKENS_PER_ROW, ROPE_WORDS
        )
        rope_valid = valid_mask.reshape(num_tokens, -1, ROPE_TOKENS_PER_ROW, 1)
        out_rope = jnp.where(rope_valid, rope_quarters, 0).reshape(
            num_tokens, padded_topk // ROPE_TOKENS_PER_ROW, ROW_WORDS
        )
    masked_nope = jnp.where(valid_mask.reshape(num_tokens, padded_topk, 1), raw_nope, 0)
    return masked_nope, out_rope


def gather_sparse_fp8_kv(
    kv_pool_lo: jax.Array,
    rope_pool_lo: jax.Array,
    topk_indices: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    *,
    page_size: int,
    kv_pool_hi: jax.Array | None = None,
    rope_pool_hi: jax.Array | None = None,
    interpret: bool = False,
    enable_named_scope: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Gather 1/2-bank paged FP8 NoPE rows and packed BF16 RoPE quarters.

    Args:
        kv_pool_lo: Low-bank FP8+UE8M0 NoPE pool
            ``int32[num_pages_lo, page_size, 128]``.
        rope_pool_lo: Low-bank BF16 RoPE pool
            ``int32[num_pages_lo, page_size // 4, 128]``.
        topk_indices: Selected logical KV slot indices ``int32[num_tokens, topk]``
            with ``topk % 4 == 0``.
        block_tables: Per-sequence page tables ``int32[num_seqs, max_pages]``.
        cu_q_lens: Cumulative query token offsets ``int32[num_seqs + 1]``.
        page_size: Tokens per physical KV page (positive multiple of 4).
        kv_pool_hi: Optional high-bank FP8+UE8M0 NoPE pool
            ``int32[num_pages_hi, page_size, 128]``.
        rope_pool_hi: Optional high-bank BF16 RoPE pool
            ``int32[num_pages_hi, page_size // 4, 128]``.
        interpret: Run Pallas calls in CPU interpret mode when ``True``.
        enable_named_scope: Emit XProf ``jax.named_scope("gather_sparse_fp8_kv")``
            when ``True``.

    Returns:
        Tuple ``(gathered_nope, gathered_rope)`` with shapes
        ``int32[num_tokens, topk, 128]`` and ``int32[num_tokens, topk // 4, 128]``.

    Raises:
        RuntimeError: If the active TPU device or abstract mesh is not TPU7x.
        ValueError: If any input tensor shape, dtype, or ``page_size`` invariant
            is violated.

    Data Lifecycle:
        - Created: ``(gathered_nope, gathered_rope)`` (unmapped/OOB pages zero-filled).
        - Used: ``kv_pool_lo``, ``rope_pool_lo``, ``topk_indices``, ``block_tables``,
          ``cu_q_lens``, and optional ``kv_pool_hi`` / ``rope_pool_hi`` (read-only).
        - Changed: None.
    """
    _check_tpu7x()
    if page_size <= 0 or page_size % ROPE_TOKENS_PER_ROW != 0:
        raise ValueError(f"Expected page_size > 0, got {page_size}.")
    if (kv_pool_hi is None) != (rope_pool_hi is None):
        raise ValueError("kv_pool_hi and rope_pool_hi must both be set or None.")
    num_pages_lo = _check_pool_pair(kv_pool_lo, rope_pool_lo, page_size, "lo")
    num_pages_hi = 0
    if kv_pool_hi is not None and rope_pool_hi is not None:
        num_pages_hi = _check_pool_pair(kv_pool_hi, rope_pool_hi, page_size, "hi")
    topk_valid = (
        topk_indices.shape[1] % ROPE_TOKENS_PER_ROW == 0
        if topk_indices.ndim == 2
        else False
    )
    _expect_arr(topk_indices, "topk_indices", 2, is_valid=topk_valid)
    tables_valid = min(block_tables.shape) >= 1 if block_tables.ndim == 2 else False
    _expect_arr(block_tables, "block_tables", 2, is_valid=tables_valid)
    _expect_arr(cu_q_lens, "cu_q_lens", (block_tables.shape[0] + 1,))
    num_tokens, topk = topk_indices.shape
    if num_tokens == 0 or topk == 0:
        zero_nope = jnp.zeros((num_tokens, topk, ROW_WORDS), jnp.int32)
        return zero_nope, zero_nope[:, : topk // ROPE_TOKENS_PER_ROW, :]
    padded_topk = ((topk + SC_GATHER_TOPK - 1) // SC_GATHER_TOPK) * SC_GATHER_TOPK
    padded_indices = jnp.pad(
        topk_indices,
        ((0, 0), (0, padded_topk - topk)),
        constant_values=INVALID_INDEX,
    )
    valid_token, seq_ids = _token_seq_ids(num_tokens, cu_q_lens)
    max_pages = block_tables.shape[1]
    logical_pages = padded_indices // page_size
    page_offsets = padded_indices % page_size
    valid_logical = (
        valid_token[:, None] & (padded_indices >= 0) & (logical_pages < max_pages)
    )
    clipped_logical_pages = jnp.clip(logical_pages, 0, max_pages - 1)
    physical_pages = _gather_page_ids(
        block_tables, clipped_logical_pages, seq_ids, interpret
    )
    valid_physical = (
        valid_logical
        & (physical_pages >= 0)
        & (physical_pages < num_pages_lo + num_pages_hi)
    )
    flat_row_indices = physical_pages * page_size + page_offsets
    lo_indices = jnp.where(
        valid_physical & (physical_pages < num_pages_lo),
        flat_row_indices,
        INVALID_INDEX,
    )
    with _named_scope(enable_named_scope, "gather_sparse_fp8_kv"):
        out_nope, out_rope = _gather_bank(
            kv_pool_lo, rope_pool_lo, lo_indices, interpret
        )
        if kv_pool_hi is not None and rope_pool_hi is not None:
            in_hi_bank = valid_physical & (physical_pages >= num_pages_lo)
            hi_indices = jnp.where(
                in_hi_bank,
                flat_row_indices - num_pages_lo * page_size,
                INVALID_INDEX,
            )
            hi_nope, hi_rope = _gather_bank(
                kv_pool_hi, rope_pool_hi, hi_indices, interpret
            )
            out_nope = out_nope | hi_nope
            out_rope = out_rope | hi_rope
    return (
        out_nope[:, :topk, :],
        out_rope[:, : topk // ROPE_TOKENS_PER_ROW, :],
    )


def _pack_bf16_pair_to_i32(lo_bf16: jax.Array, hi_bf16: jax.Array) -> jax.Array:
    """Pack two ``bfloat16`` arrays into low and high 16-bit halves of ``int32``."""
    lo_u32 = _bitcast(lo_bf16.astype(jnp.float32), jnp.uint32) >> jnp.uint32(
        HALF_WORD_BITS
    )
    hi_u32 = _bitcast(hi_bf16.astype(jnp.float32), jnp.uint32) & jnp.uint32(
        HIGH_HALF_WORD_MASK
    )
    return (lo_u32 | hi_u32).astype(jnp.int32)


def _unpack_bf16_pair_from_u32(
    packed_u32: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Unpack low and high ``bfloat16`` arrays from packed ``uint32`` words."""
    lo_f32 = _bitcast(
        (packed_u32 << jnp.uint32(HALF_WORD_BITS)).astype(jnp.int32),
        jnp.float32,
    )
    hi_f32 = _bitcast(
        (packed_u32 & jnp.uint32(HIGH_HALF_WORD_MASK)).astype(jnp.int32),
        jnp.float32,
    )
    return lo_f32.astype(jnp.bfloat16), hi_f32.astype(jnp.bfloat16)


def _unpack_rope_quarter_i32(rope_words_u32: jax.Array, quarter_idx: int) -> jax.Array:
    """Unpack 32-word RoPE quarter ``quarter_idx`` into lanes ``64..127``."""
    lo_raw = (rope_words_u32 << jnp.uint32(HALF_WORD_BITS)).astype(jnp.int32)
    hi_raw = (rope_words_u32 & jnp.uint32(HIGH_HALF_WORD_MASK)).astype(jnp.int32)
    shift_lo = (ROPE_LOW_SHIFT_BASE - ROPE_WORDS * quarter_idx) % ROW_WORDS
    shift_hi = (ROPE_HIGH_LANE_START - ROPE_WORDS * quarter_idx) % ROW_WORDS
    lo_part = pltpu.roll(lo_raw, shift_lo, 1) if shift_lo else lo_raw
    hi_part = pltpu.roll(hi_raw, shift_hi, 1) if shift_hi else hi_raw
    is_hi_lane = (
        jax.lax.broadcasted_iota(jnp.int32, rope_words_u32.shape, 1)
        >= ROPE_HIGH_LANE_START
    )
    return jnp.where(is_hi_lane, hi_part, lo_part)


def _extract_swa_windows_2d(
    packed_swa_words: jax.Array,
    metadata: jax.Array,
    half_stride: int,
    swa_width: int,
    padded_swa_width: int,
    interpret: bool,
) -> jax.Array:
    """Assemble per-token SWA BF16 word windows in VMEM from compact 2D ``swa_kv``."""
    chunk_tokens = metadata.shape[1]
    num_swa_blocks = padded_swa_width // SWA_SUB_K

    def _extract_windows_kernel(
        meta_ref: Any,
        packed_swa_ref: Any,
        out_ref: Any,
        prefix_scr_ref: Any,
        current_scr_ref: Any,
        dma_sem: Any,
    ) -> None:
        token_idx = pl.program_id(0)
        prefix_row = meta_ref[0, token_idx]
        current_row = meta_ref[1, token_idx]
        delta_token = meta_ref[2, token_idx]
        window_iota = jax.lax.broadcasted_iota(
            jnp.int32, (padded_swa_width, ROW_WORDS), 0
        )
        use_prefix = window_iota < (swa_width - 1 - delta_token)
        valid_window = window_iota < swa_width
        for half_idx in range(BF16_PER_WORD):
            for row_offset, scratch_ref in (
                (prefix_row + half_idx * half_stride, prefix_scr_ref),
                (current_row + half_idx * half_stride, current_scr_ref),
            ):
                copy_desc = pltpu.make_async_copy(
                    packed_swa_ref.at[pl.ds(row_offset, padded_swa_width), :],
                    scratch_ref,
                    dma_sem,
                )
                copy_desc.start()
                copy_desc.wait()
            merged_words = jnp.where(
                valid_window,
                jnp.where(use_prefix, prefix_scr_ref[...], current_scr_ref[...]),
                0,
            )
            out_ref[
                0,
                half_idx * padded_swa_width : (half_idx + 1) * padded_swa_width,
                :,
            ] = merged_words

    out_type = jax.ShapeDtypeStruct(
        (chunk_tokens, BF16_PER_WORD * padded_swa_width, ROW_WORDS), jnp.int32
    )
    out_spec = pl.BlockSpec(
        (1, BF16_PER_WORD * padded_swa_width, ROW_WORDS),
        lambda token_idx, _meta_ref: (token_idx, 0, 0),
    )
    vmem_scratch = pltpu.VMEM((padded_swa_width, ROW_WORDS), jnp.int32)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        1,
        (chunk_tokens,),
        [_HBM_SPEC],
        out_spec,
        [vmem_scratch, vmem_scratch, pltpu.SemaphoreType.DMA],
    )
    extracted = _pallas(
        _extract_windows_kernel,
        out_type,
        grid_spec,
        interpret,
        "dsv41_extract_swa_windows",
        metadata,
        packed_swa_words,
    )
    windows_5d = extracted.reshape(
        chunk_tokens, BF16_PER_WORD, num_swa_blocks, SWA_SUB_K, ROW_WORDS
    ).transpose(0, 2, 1, 3, 4)
    return windows_5d.reshape(chunk_tokens, num_swa_blocks, SUB_K, ROW_WORDS)


def _sparse_mla_swa_tile(
    query_ref: Any,
    sinks_ref: Any,
    out_ref: Any,
    state_scr_ref: Any,
    kv_scr_ref: Any,
    rope_scr_ref: Any,
    tile_ref: Any,
    *,
    config: _AttentionTileConfig,
) -> None:
    """Process one fused SWA (128-key) or Sparse MLA (256-key) block in VMEM."""
    token_idx = pl.program_id(0)
    block_idx = pl.program_id(1)
    padded_heads = query_ref.shape[1]

    @pl.when(block_idx == 0)
    def _init_softmax_state() -> None:
        state_scr_ref[:, :STATE_MAX_COL_START] = jnp.zeros(
            (padded_heads, HEAD_DIM), dtype=jnp.float32
        )
        state_scr_ref[:, STATE_MAX_COL_START:STATE_SUM_COL_START] = sinks_ref[0, :, :]
        state_scr_ref[:, STATE_SUM_COL_START:STATE_COLS] = jnp.ones(
            (padded_heads, ROW_WORDS), dtype=jnp.float32
        )

    valid_row_start = SUB_K + SUB_ROPE
    valid_even_f32 = _bitcast(
        tile_ref[0, valid_row_start : valid_row_start + SUBLANES, :],
        jnp.float32,
    )
    valid_odd_f32 = _bitcast(
        tile_ref[0, valid_row_start + SUBLANES : COMBINED_ROWS, :],
        jnp.float32,
    )
    valid_f32 = jnp.concatenate([valid_even_f32, valid_odd_f32], axis=1)
    with _named_scope(config.enable_named_scope, "dequant_fp8_kv_and_rope"):

        @pl.when(block_idx < config.num_swa_blocks)
        def _unpack_swa_tile() -> None:
            for half_idx, row_start in enumerate((0, SWA_SUB_K)):
                swa_words_u32 = tile_ref[
                    0, row_start : row_start + SWA_SUB_K, :
                ].astype(jnp.uint32)
                lo_bf16, hi_bf16 = _unpack_bf16_pair_from_u32(swa_words_u32)
                col_base = WORDS_PER_HEAD_DIM * half_idx
                kv_scr_ref[:SWA_SUB_K, col_base : col_base + ROW_WORDS] = lo_bf16
                kv_scr_ref[
                    :SWA_SUB_K, col_base + ROW_WORDS : col_base + WORDS_PER_HEAD_DIM
                ] = hi_bf16
            kv_scr_ref[SWA_SUB_K:SUB_K, :] = jnp.zeros(
                (SWA_SUB_K, HEAD_DIM), jnp.bfloat16
            )

        @pl.when(block_idx >= config.num_swa_blocks)
        def _dequant_sparse_tile() -> None:
            nope_words_u32 = tile_ref[0, :SUB_K, :].astype(jnp.uint32)
            rope_words_u32 = tile_ref[0, SUB_K : SUB_K + SUB_ROPE, :].astype(jnp.uint32)
            for quarter_idx in range(ROPE_TOKENS_PER_ROW):
                rope_scr_ref[quarter_idx::ROPE_TOKENS_PER_ROW, :] = (
                    _unpack_rope_quarter_i32(rope_words_u32, quarter_idx)
                )
            rope_bf16 = _bitcast(rope_scr_ref[...], jnp.float32).astype(jnp.bfloat16)
            lane_ids = jax.lax.broadcasted_iota(jnp.int32, nope_words_u32.shape, 1)
            top_byte_u32 = (
                nope_words_u32 >> jnp.uint32(TOP_BYTE_SHIFT_BITS)
            ) & jnp.uint32(BYTE_MASK)
            top_byte_rolled = pltpu.roll(top_byte_u32, NOPE_SCALE_PERIOD, 1)
            scale_u8 = jnp.where(
                lane_ids < NOPE_TAIL_LANES, top_byte_rolled, top_byte_u32
            ).astype(jnp.uint8)
            scale_bf16 = _bitcast(scale_u8, jnp.float8_e8m0fnu).astype(jnp.bfloat16)
            for slab_idx in range(NOPE_FULL_WORD_SLABS):
                byte_u8 = (nope_words_u32 >> jnp.uint32(BYTE_BITS * slab_idx)).astype(
                    jnp.uint8
                )
                fp8_bf16 = _bitcast(byte_u8, jnp.float8_e4m3fn).astype(jnp.bfloat16)
                kv_scr_ref[:, slab_idx * ROW_WORDS : (slab_idx + 1) * ROW_WORDS] = (
                    fp8_bf16 * scale_bf16
                )
            tail_fp8_bf16 = _bitcast(
                top_byte_u32.astype(jnp.uint8), jnp.float8_e4m3fn
            ).astype(jnp.bfloat16)
            kv_scr_ref[:, NOPE_FULL_SLAB_CHANNELS:] = jnp.where(
                lane_ids < NOPE_TAIL_LANES,
                tail_fp8_bf16 * scale_bf16,
                rope_bf16,
            )

    with _named_scope(config.enable_named_scope, "mla_online_softmax_update"):
        kv_masked = jnp.where(
            valid_f32[:1, :].T > 0.0, kv_scr_ref[...], jnp.bfloat16(0)
        )
        q_dot = jnp.dot(
            query_ref[token_idx, :, :],
            kv_masked.T,
            preferred_element_type=jnp.float32,
        )
        valid_mask = jnp.tile(valid_f32, (padded_heads // SUBLANES, 1)) > 0.0
        logits = jnp.where(valid_mask, q_dot * config.sm_scale, jnp.float32(SCORE_MIN))
        max_prev = state_scr_ref[:, STATE_MAX_COL_START:STATE_SUM_COL_START]
        sum_prev = state_scr_ref[:, STATE_SUM_COL_START:STATE_COLS]
        row_max = jnp.max(logits, 1, keepdims=True)
        max_new = jnp.maximum(max_prev, jnp.broadcast_to(row_max, max_prev.shape))
        rescale_factor = jnp.exp(max_prev - max_new)
        probs = jnp.where(
            valid_mask,
            jnp.exp(logits - jnp.tile(max_new, (1, VALID_SUB_TILES))),
            0.0,
        )
        probs_sum = jnp.broadcast_to(
            jnp.sum(probs, axis=1, keepdims=True), sum_prev.shape
        )
        state_scr_ref[:, STATE_MAX_COL_START:STATE_SUM_COL_START] = max_new
        state_scr_ref[:, STATE_SUM_COL_START:STATE_COLS] = (
            sum_prev * rescale_factor + probs_sum
        )
        weighted_values = jnp.dot(
            probs.astype(jnp.bfloat16),
            kv_masked,
            preferred_element_type=jnp.float32,
        )
        acc_prev = state_scr_ref[:, :STATE_MAX_COL_START]
        state_scr_ref[:, :STATE_MAX_COL_START] = (
            acc_prev * jnp.tile(rescale_factor, (1, HEAD_SLABS_PER_ROW))
            + weighted_values
        )

    @pl.when(block_idx == config.num_blocks - 1)
    def _write_normalized_output() -> None:
        denom = jnp.tile(
            state_scr_ref[:, STATE_SUM_COL_START:STATE_COLS],
            (1, HEAD_SLABS_PER_ROW),
        )
        out_ref[token_idx, :, :] = (
            state_scr_ref[:, :STATE_MAX_COL_START] / denom
        ).astype(jnp.bfloat16)


def sparse_mla_swa_attention(
    q: jax.Array,
    gathered_nope: jax.Array,
    gathered_rope: jax.Array,
    topk_indices: jax.Array,
    attention_sinks: jax.Array,
    seq_lens: jax.Array,
    cu_q_lens: jax.Array,
    *,
    sm_scale: float,
    swa_window: int,
    swa_kv: jax.Array | None = None,
    kv_compress_ratio: int = 1,
    swa_min_pos: jax.Array | None = None,
    interpret: bool = False,
    enable_named_scope: bool = False,
) -> jax.Array:
    """Compute fused Sparse MLA + Sliding Window Attention + sink online softmax.

    Note:
        Compact 2D ``swa_kv`` windows are right-aligned
        (``[pos - swa_width + 1 .. pos]``), whereas 3D ``swa_kv`` windows are
        left-aligned (starting at ``max(0, pos - swa_width + 1)``).

    Args:
        q: Query tensor ``bf16[num_tokens, num_heads, 512]``
            (``448`` NoPE + ``64`` RoPE).
        gathered_nope: Gathered FP8+UE8M0 NoPE rows ``int32[num_tokens, topk, 128]``.
        gathered_rope: Gathered BF16 RoPE quarter rows
            ``int32[num_tokens, topk // 4, 128]``.
        topk_indices: Selected logical KV indices ``int32[num_tokens, topk]``
            (``< 0`` masked).
        attention_sinks: Per-head sink logits ``f32[num_heads]``.
        seq_lens: Total sequence lengths including current step ``int32[num_seqs]``.
        cu_q_lens: Cumulative query token offsets ``int32[num_seqs + 1]``.
        sm_scale: Softmax logit scale factor (typically ``1 / sqrt(512)``).
        swa_window: Sliding window causal span ``>= 0`` (``0`` disables SWA).
        swa_kv: Optional BF16 SWA KV cache, either 3D left-aligned
            ``bf16[num_tokens, swa_width, 512]`` or compact 2D right-aligned
            ``bf16[num_seqs * swa_width + num_tokens, 512]``.
        kv_compress_ratio: Tokens per compressed sparse KV slot (``1`` or ``2``).
        swa_min_pos: Optional inclusive per-sequence lower bound ``int32[num_seqs]``
            on valid SWA KV positions under bounded context-eviction replay.
        interpret: Run Pallas calls in CPU interpret mode when ``True``.
        enable_named_scope: Emit XProf ``jax.named_scope`` annotations when ``True``.

    Returns:
        Attention output ``bf16[num_tokens, num_heads, 512]`` (zero for padded
        trailing tokens outside ``cu_q_lens``).

    Raises:
        RuntimeError: If the active TPU device or abstract mesh is not TPU7x.
        ValueError: If any input tensor shape, dtype, ``swa_window``, or
            ``kv_compress_ratio`` invariant is violated.

    Data Lifecycle:
        - Created: Output tensor ``bf16[num_tokens, num_heads, 512]``.
        - Used: ``q``, ``gathered_nope``, ``gathered_rope``, ``topk_indices``,
          ``attention_sinks``, ``seq_lens``, ``cu_q_lens``, ``swa_kv``, and
          ``swa_min_pos`` (read-only).
        - Changed: None.
    """
    _check_tpu7x()
    _expect_arr(q, "q", 3, jnp.bfloat16, q.ndim == 3 and q.shape[2] == HEAD_DIM)
    num_tokens, num_heads, _ = q.shape
    topk_valid = (
        topk_indices.shape[:1] == (num_tokens,)
        and topk_indices.shape[-1] % ROPE_TOKENS_PER_ROW == 0
    )
    _expect_arr(topk_indices, "topk_indices", 2, is_valid=topk_valid)
    topk = topk_indices.shape[1]
    _expect_arr(gathered_nope, "gathered_nope", (num_tokens, topk, ROW_WORDS))
    _expect_arr(
        gathered_rope,
        "gathered_rope",
        (num_tokens, topk // ROPE_TOKENS_PER_ROW, ROW_WORDS),
    )
    _expect_arr(attention_sinks, "attention_sinks", (num_heads,), jnp.float32)
    _expect_arr(seq_lens, "seq_lens", 1, is_valid=seq_lens.shape[0] >= 1)
    num_seqs = seq_lens.shape[0]
    _expect_arr(cu_q_lens, "cu_q_lens", (num_seqs + 1,))
    if swa_min_pos is not None:
        _expect_arr(swa_min_pos, "swa_min_pos", (num_seqs,))
    if swa_window < 0 or kv_compress_ratio not in (1, 2):
        raise ValueError(f"Expected swa_window >= 0 and ratio in (1, 2): {swa_window}")
    raw_swa_width = 0
    if swa_kv is not None:
        if swa_kv.ndim == 2:
            prefix_rows = swa_kv.shape[0] - num_tokens
            valid_2d = (
                swa_kv.shape[1] == HEAD_DIM
                and prefix_rows >= num_seqs
                and prefix_rows % num_seqs == 0
            )
            _expect_arr(swa_kv, "swa_kv", 2, jnp.bfloat16, valid_2d)
            raw_swa_width = prefix_rows // num_seqs
        else:
            valid_3d = swa_kv.shape[0] == num_tokens and swa_kv.shape[-1] == HEAD_DIM
            _expect_arr(swa_kv, "swa_kv", 3, jnp.bfloat16, valid_3d)
            raw_swa_width = swa_kv.shape[1]
    if num_tokens == 0 or num_heads == 0:
        return jnp.zeros((num_tokens, num_heads, HEAD_DIM), dtype=jnp.bfloat16)

    valid_token, seq_ids = _token_seq_ids(num_tokens, cu_q_lens)
    token_indices = jnp.arange(num_tokens, dtype=jnp.int32)
    token_positions = seq_lens[seq_ids] - (cu_q_lens[seq_ids + 1] - token_indices)
    padded_topk = max(SUB_K, ((topk + SUB_K - 1) // SUB_K) * SUB_K)
    num_sparse_blocks = padded_topk // SUB_K
    pad_topk_cols = padded_topk - topk
    swa_width = raw_swa_width if swa_window > 0 else 0
    padded_swa_width = ((swa_width + SWA_SUB_K - 1) // SWA_SUB_K) * SWA_SUB_K
    num_swa_blocks = padded_swa_width // SWA_SUB_K
    pad_swa_cols = padded_swa_width - swa_width
    total_blocks = num_swa_blocks + num_sparse_blocks
    _slice0 = jax.lax.dynamic_slice_in_dim

    packed_swa_buf: tuple[jax.Array, int] | None = None
    if swa_kv is not None and num_swa_blocks > 0 and swa_kv.ndim == 2:
        num_prefix_rows = num_seqs * swa_width
        tail_pad = padded_swa_width + ((-(num_prefix_rows + num_tokens)) % SUBLANES)
        swa_quads = swa_kv.reshape(
            num_prefix_rows + num_tokens,
            BF16_PER_WORD,
            BF16_PER_WORD,
            ROW_WORDS,
        )
        packed_halves = _pack_bf16_pair_to_i32(
            swa_quads[:, :, 0, :], swa_quads[:, :, 1, :]
        ).transpose(1, 0, 2)
        padded_prefix = jnp.pad(
            packed_halves[:, :num_prefix_rows],
            ((0, 0), (0, padded_swa_width), (0, 0)),
        )
        padded_current = jnp.pad(
            packed_halves[:, num_prefix_rows:],
            ((0, 0), (0, tail_pad), (0, 0)),
        )
        packed_swa_3d = jnp.concatenate([padded_prefix, padded_current], axis=1)
        packed_swa_buf = (
            packed_swa_3d.reshape(-1, ROW_WORDS),
            packed_swa_3d.shape[1],
        )

    def _pack_valid_rows(
        valid_float: jax.Array, chunk_tokens: int, num_blocks: int
    ) -> jax.Array:
        target_shape = (
            chunk_tokens,
            num_blocks,
            VALID_SUB_TILES,
            SUBLANES,
            ROW_WORDS,
        )
        broadcast_valid = jnp.broadcast_to(
            valid_float.reshape(
                chunk_tokens, num_blocks, VALID_SUB_TILES, 1, ROW_WORDS
            ),
            target_shape,
        )
        return _bitcast(
            broadcast_valid.reshape(chunk_tokens, num_blocks, VALID_ROWS, ROW_WORDS),
            jnp.int32,
        )

    def _pack_chunk_tiles(start_token: jax.Array | int, chunk_tokens: int) -> jax.Array:
        valid_tok_chunk = _slice0(valid_token, start_token, chunk_tokens)
        pos_chunk = _slice0(token_positions, start_token, chunk_tokens)
        topk_chunk = _slice0(topk_indices, start_token, chunk_tokens)
        in_causal_bounds = (topk_chunk >= 0) & (
            topk_chunk < ((pos_chunk + 1) // kv_compress_ratio)[:, None]
        )
        valid_sparse_f32 = jnp.pad(
            (valid_tok_chunk[:, None] & in_causal_bounds).astype(jnp.float32),
            ((0, 0), (0, pad_topk_cols)),
        )
        nope_chunk = jnp.pad(
            _slice0(gathered_nope, start_token, chunk_tokens),
            ((0, 0), (0, pad_topk_cols), (0, 0)),
        )
        rope_chunk = jnp.pad(
            _slice0(gathered_rope, start_token, chunk_tokens),
            ((0, 0), (0, pad_topk_cols // ROPE_TOKENS_PER_ROW), (0, 0)),
        )
        nope_blocks = nope_chunk.reshape(
            chunk_tokens, num_sparse_blocks, SUB_K, ROW_WORDS
        )
        rope_blocks = rope_chunk.reshape(
            chunk_tokens, num_sparse_blocks, SUB_ROPE, ROW_WORDS
        )
        combined_blocks = jnp.concatenate(
            [
                nope_blocks,
                rope_blocks,
                _pack_valid_rows(valid_sparse_f32, chunk_tokens, num_sparse_blocks),
            ],
            axis=2,
        )
        if swa_kv is not None and num_swa_blocks > 0:
            seq_chunk = _slice0(seq_ids, start_token, chunk_tokens)
            if packed_swa_buf is not None:
                base_pos = (pos_chunk - swa_width + 1)[:, None]
                token_chunk = start_token + jnp.arange(chunk_tokens, dtype=jnp.int32)
                delta_chunk = jnp.maximum(0, token_chunk - cu_q_lens[seq_chunk])
                prefix_starts = seq_chunk * swa_width + jnp.minimum(
                    delta_chunk + 1, swa_width
                )
                current_starts = (
                    (num_seqs * swa_width + padded_swa_width)
                    + token_chunk
                    - (swa_width - 1)
                )
                metadata = jnp.stack(
                    [prefix_starts, current_starts, delta_chunk], axis=0
                )
                swa_words_all = _extract_swa_windows_2d(
                    packed_swa_buf[0],
                    metadata,
                    packed_swa_buf[1],
                    swa_width,
                    padded_swa_width,
                    interpret,
                )
            else:
                base_pos = jnp.maximum(0, pos_chunk - swa_width + 1)[:, None]
                swa_padded = jnp.pad(
                    _slice0(swa_kv, start_token, chunk_tokens),
                    ((0, 0), (0, pad_swa_cols), (0, 0)),
                )
                swa_quads_3d = swa_padded.reshape(
                    chunk_tokens,
                    num_swa_blocks,
                    SWA_SUB_K,
                    BF16_PER_WORD,
                    BF16_PER_WORD,
                    ROW_WORDS,
                )
                swa_packed_pairs = _pack_bf16_pair_to_i32(
                    swa_quads_3d[..., 0, :], swa_quads_3d[..., 1, :]
                )
                swa_words_all = swa_packed_pairs.transpose(0, 1, 3, 2, 4).reshape(
                    chunk_tokens, num_swa_blocks, SUB_K, ROW_WORDS
                )
            min_pos = (
                0 if swa_min_pos is None else jnp.maximum(0, swa_min_pos[seq_chunk])
            )
            kv_positions = base_pos + jnp.arange(swa_width, dtype=jnp.int32)[None, :]
            window_start = jnp.maximum(min_pos, pos_chunk - swa_window + 1)[:, None]
            swa_valid = (
                valid_tok_chunk[:, None]
                & (kv_positions >= window_start)
                & (kv_positions <= pos_chunk[:, None])
            )
            swa_valid_f32 = jnp.pad(
                swa_valid.astype(jnp.float32), ((0, 0), (0, pad_swa_cols))
            )
            swa_valid_padded = jnp.pad(
                swa_valid_f32.reshape(chunk_tokens, num_swa_blocks, -1),
                ((0, 0), (0, 0), (0, ROW_WORDS)),
            )
            zero_rope_rows = jnp.zeros(
                (chunk_tokens, num_swa_blocks, SUB_ROPE, ROW_WORDS),
                dtype=jnp.int32,
            )
            swa_valid_rows = _pack_valid_rows(
                swa_valid_padded, chunk_tokens, num_swa_blocks
            )
            swa_combined = jnp.concatenate(
                [swa_words_all, zero_rope_rows, swa_valid_rows], axis=2
            )
            combined_blocks = jnp.concatenate([swa_combined, combined_blocks], axis=1)
        return combined_blocks.reshape(
            chunk_tokens, total_blocks * COMBINED_ROWS, ROW_WORDS
        )

    padded_heads = ((num_heads + SUBLANES - 1) // SUBLANES) * SUBLANES
    sinks_padded = jnp.pad(attention_sinks, (0, padded_heads - num_heads))[
        None, :, None
    ]
    sinks_vmem = jnp.broadcast_to(sinks_padded, (1, padded_heads, ROW_WORDS))
    query_padded = jnp.pad(q, ((0, 0), (0, padded_heads - num_heads), (0, 0)))
    vmem_state = pltpu.VMEM((padded_heads, STATE_COLS), jnp.float32)
    vmem_kv = pltpu.VMEM((SUB_K, HEAD_DIM), jnp.bfloat16)
    scratch_shapes = [
        vmem_state,
        vmem_kv,
        pltpu.VMEM((SUB_K, ROW_WORDS), jnp.int32),
    ]
    grid_spec = pltpu.PrefetchScalarGridSpec(
        0,
        (),
        [_HBM_SPEC, _VMEM_SPEC, _VMEM_SPEC],
        _VMEM_SPEC,
        scratch_shapes,
    )
    tile_config = _AttentionTileConfig(
        num_swa_blocks, total_blocks, sm_scale, enable_named_scope
    )

    def _run_chunk(start_token: jax.Array | int, chunk_tokens: int) -> jax.Array:
        def _attention_chunk_entry(
            tiles_ref: Any,
            query_ref: Any,
            sinks_ref: Any,
            out_ref: Any,
            state_scr_ref: Any,
            kv_scr_ref: Any,
            rope_scr_ref: Any,
        ) -> None:
            tile_fn = functools.partial(
                _sparse_mla_swa_tile,
                query_ref,
                sinks_ref,
                out_ref,
                state_scr_ref,
                kv_scr_ref,
                rope_scr_ref,
                config=tile_config,
            )
            tile_spec = pl.BlockSpec(
                (1, COMBINED_ROWS, ROW_WORDS),
                lambda token_idx, block_idx: (token_idx, block_idx, 0),
            )
            pipeline_semantics: Any = ("parallel", "arbitrary")
            pltpu.emit_pipeline(
                tile_fn,
                grid=(chunk_tokens, total_blocks),
                in_specs=[tile_spec],
                out_specs=[],
                dimension_semantics=pipeline_semantics,
                trace_scopes=tile_config.enable_named_scope,
            )(tiles_ref)

        out_type = jax.ShapeDtypeStruct(
            (chunk_tokens, padded_heads, HEAD_DIM), jnp.bfloat16
        )
        chunk_tiles = _pack_chunk_tiles(start_token, chunk_tokens)
        query_chunk = _slice0(query_padded, start_token, chunk_tokens)
        return _pallas(
            _attention_chunk_entry,
            out_type,
            grid_spec,
            interpret,
            "dsv41_sparse_mla_swa",
            chunk_tiles,
            query_chunk,
            sinks_vmem,
        )

    chunk_step = max(1, _max_vmem_token_heads() // padded_heads)
    if num_tokens <= chunk_step:
        out_padded = _run_chunk(0, num_tokens)
    else:
        num_full_chunks, rem_tokens = divmod(num_tokens, chunk_step)
        chunk_indices = jnp.arange(num_full_chunks, dtype=jnp.int32)
        _, full_chunks = jax.lax.scan(
            lambda _carry, chunk_idx: (
                0,
                _run_chunk(chunk_idx * chunk_step, chunk_step),
            ),
            0,
            chunk_indices,
        )
        out_padded = full_chunks.reshape(
            num_full_chunks * chunk_step, padded_heads, HEAD_DIM
        )
        if rem_tokens:
            out_padded = jnp.concatenate(
                [out_padded, _run_chunk(num_full_chunks * chunk_step, rem_tokens)],
                axis=0,
            )
    return jnp.where(
        valid_token[:, None, None],
        out_padded[:, :num_heads, :],
        jnp.bfloat16(0),
    )
