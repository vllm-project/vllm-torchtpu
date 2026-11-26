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
"""Utilities for paged KV cache indexing and byte packing.

NOTE: Currently designed for 8-bit / FP8 KV caches (1 byte per element). We use
32-bit physical storage for SparseCore layout but the logical cache is still fp8
"""

import dataclasses
import enum
import functools
import typing

import jax
import jax.numpy as jnp
import torch

_tpu_pallas: typing.Any
try:
    from torch_tpu._internal.pallas import pallas as _tpu_pallas
except ImportError:
    _tpu_pallas = None

# Both layouts store a token as 32-bit words: the native SparseCore cache holds
# them as uint32, the TensorCore-tiled cache as WORD_BYTES uint8 rows that
# `ref.bitcast(jnp.int32)` folds into one word row inside the kernel.
WORD_BYTES = 4
TILE_LANE_BYTES = 128  # lane width of a TensorCore tile, in bytes
# Minimum SparseCore indirect-gather width: 16 vector lanes of uint32 words.
SC_ROPE_ALIGN_BYTES = 16 * WORD_BYTES

# Sentinels for tokens that must not be written. JAX drops out-of-bounds
# scatters; the Pallas writers predicate on the sign instead.
OOB_PAGE = 2**30
SKIP_ROW = -1


class KVCacheLayout(enum.Enum):
    """Physical HBM layout of a paged KV cache.

    TENSORCORE: uint8, 2D-tiled. 4 sublanes fold into 32-bit words via hardware
    TC tiling.
    SPARSECORE: uint32, 1D-flat. The writer explicitly bit-packs `WORD_BYTES`
    byte bands into 32-bit words (pack_tokens) for native SC DMA.
    """

    SPARSECORE = "sparsecore"
    TENSORCORE = "tensorcore"


class KVCacheType(enum.Enum):
    """The type of a paged KV cache."""

    NOPE = "nope"
    ROPE = "rope"


_LAYOUT_DTYPES = {
    KVCacheLayout.TENSORCORE: jnp.uint8,
    KVCacheLayout.SPARSECORE: jnp.uint32,
}


def align_to(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def _xla_unpadded_word_rows(rows: int) -> bool:
    """Whether `rows` 128-word rows per page avoid XLA `(8, 128)` tile padding."""
    return rows > 0 and (rows % 8 == 0 or (rows < 8 and (rows & (rows - 1)) == 0))


def get_dtype_bitwidth(dtype: jnp.dtype) -> int:
    return jax.dtypes.itemsize_bits(dtype)


def get_dtype_packing(dtype: jnp.dtype) -> int:
    bits = get_dtype_bitwidth(dtype)
    return 32 // bits


def get_kv_cache_shape(
    total_num_pages: int,
    page_size: int,
    kv_dim: int,
    kv_dtype: jnp.dtype | None = None,
    kv_packing: int | None = None,
) -> tuple[int, int, int, int]:
    """Computes padded 4D shape
    `(total_num_pages, page_size // kv_packing, kv_packing, kv_dim)`."""
    if kv_packing is None:
        assert kv_dtype is not None, "must pass kv_dtype if kv_packing is None"
        kv_packing = get_dtype_packing(kv_dtype)
    return (
        total_num_pages,
        align_to(page_size, kv_packing) // kv_packing,
        kv_packing,
        align_to(kv_dim, TILE_LANE_BYTES),
    )


@dataclasses.dataclass(frozen=True)
class SparseMLAKVCacheSpec:
    cache_type: KVCacheType
    layout: KVCacheLayout
    num_pages: int
    page_size: int
    head_dim: int
    kv_packing: int

    @classmethod
    def create(
        cls,
        cache_type: KVCacheType,
        layout: KVCacheLayout,
        num_pages: int,
        page_size: int,
        head_dim: int,
        kv_packing: int = 4,
        packed_rope: bool | None = None,
    ) -> "SparseMLAKVCacheSpec":
        """Builds a spec with the kernel's page/head-dim padding applied."""
        if packed_rope is True and (
            cache_type is not KVCacheType.ROPE or layout is not KVCacheLayout.SPARSECORE
        ):
            raise ValueError("packed_rope requires ROPE cache on SPARSECORE layout")
        unpadded_head_dim = head_dim
        num_pages, packed_page_size, kv_packing, head_dim = get_kv_cache_shape(
            num_pages, page_size, head_dim, None, kv_packing
        )
        page_size = packed_page_size * kv_packing

        if (
            packed_rope is not False
            and cache_type is KVCacheType.ROPE
            and layout is KVCacheLayout.SPARSECORE
            and kv_packing == 4
        ):
            # Native SparseCore ROPE packs whole tokens into 128-word (512B)
            # rows at the 16-lane vector granularity (64B) rather than padding
            # each token to a 128B TensorCore lane row, whenever the resulting
            # per-page row count stays unpadded under XLA's (8, 128) HBM tile.
            row_bytes = WORD_BYTES * TILE_LANE_BYTES
            sc_head_dim = align_to(unpadded_head_dim, SC_ROPE_ALIGN_BYTES)
            if (
                0 < sc_head_dim <= TILE_LANE_BYTES
                and row_bytes % sc_head_dim == 0
                and page_size % (row_bytes // sc_head_dim) == 0
                and _xla_unpadded_word_rows(page_size // (row_bytes // sc_head_dim))
            ):
                head_dim = sc_head_dim

        if layout is KVCacheLayout.SPARSECORE:
            assert head_dim % WORD_BYTES == 0, (
                f"head_dim {head_dim} must be a multiple of {WORD_BYTES}"
            )
            if cache_type is KVCacheType.ROPE:
                row_bytes = WORD_BYTES * TILE_LANE_BYTES
                if (
                    head_dim > TILE_LANE_BYTES
                    or row_bytes % head_dim
                    or page_size % (row_bytes // head_dim)
                ):
                    raise ValueError(
                        f"SparseCore ROPE head_dim {head_dim} must not exceed "
                        f"{TILE_LANE_BYTES} or fail to divide {row_bytes}, and "
                        f"page_size {page_size} must hold whole {row_bytes}-byte rows"
                    )
        elif layout is KVCacheLayout.TENSORCORE:
            if cache_type is KVCacheType.NOPE:
                assert head_dim == WORD_BYTES * TILE_LANE_BYTES, (
                    f"NOPE TC head_dim {head_dim} must equal"
                    f" {WORD_BYTES * TILE_LANE_BYTES}"
                    f" ({WORD_BYTES * TILE_LANE_BYTES}B per token)"
                )

        return cls(
            cache_type=cache_type,
            layout=layout,
            num_pages=num_pages,
            page_size=page_size,
            head_dim=head_dim,
            kv_packing=kv_packing,
        )

    @property
    def jax_dtype(self) -> jnp.dtype:
        return _LAYOUT_DTYPES[self.layout]

    @property
    def torch_dtype(self) -> torch.dtype:
        dt = jnp.dtype(self.jax_dtype)
        if _tpu_pallas is not None:
            return _tpu_pallas.JAX_TO_TORCH_DTYPE_MAP[dt]
        return getattr(torch, dt.name)

    @property
    def token_bytes(self) -> int:
        """Bytes reserved per token slot (assumes 1 byte/element for FP8)."""
        return self.head_dim

    @property
    def packed_rope(self) -> bool:
        return (
            self.cache_type is KVCacheType.ROPE
            and self.layout is KVCacheLayout.SPARSECORE
            and self.head_dim < TILE_LANE_BYTES
        )

    @property
    def tokens_per_row(self) -> int:
        """Tokens sharing one `TILE_LANE_BYTES`-word row of a ROPE/SC cache."""
        return WORD_BYTES * TILE_LANE_BYTES // self.token_bytes

    @property
    def shape(self) -> tuple[int, ...]:
        if self.cache_type is KVCacheType.NOPE:
            if self.layout is KVCacheLayout.TENSORCORE:
                return (
                    self.num_pages,
                    self.page_size,
                    self.head_dim // TILE_LANE_BYTES,
                    TILE_LANE_BYTES,
                )
            return (self.num_pages, self.page_size, self.head_dim // WORD_BYTES)
        if self.cache_type is KVCacheType.ROPE:
            if self.layout is KVCacheLayout.TENSORCORE:
                return (
                    self.num_pages,
                    self.page_size // self.kv_packing,
                    self.kv_packing,
                    self.head_dim,
                )
            # `tokens_per_row` tokens of `head_dim // WORD_BYTES` words each
            # fill one 128-lane row, so the tiled HBM allocation carries no
            # per-token padding.
            return (
                self.num_pages,
                self.page_size // self.tokens_per_row,
                TILE_LANE_BYTES,
            )
        raise ValueError(f"unsupported cache type {self.cache_type}")


def as_token_bytes(values: jax.Array, token_bytes: int) -> jax.Array:
    """`[num_tokens, n]` -> `[num_tokens, token_bytes]` uint8, zero-padded."""
    u8 = jax.lax.bitcast_convert_type(values, jnp.uint8).reshape(values.shape[0], -1)
    assert u8.shape[-1] <= token_bytes, (
        f"token is {u8.shape[-1]}B, more than the {token_bytes}B reserved for it"
    )
    pad = token_bytes - u8.shape[-1]
    return jnp.pad(u8, ((0, 0), (0, pad))) if pad else u8


def pack_tokens(values: jax.Array, token_bytes: int) -> jax.Array:
    """`[num_tokens, n]` -> `[num_tokens, token_bytes // WORD_BYTES]` uint32.

    Each word takes one byte from each of WORD_BYTES equal bands of the token,
    the same interleave `ref.bitcast(jnp.int32)` produces from uint8 rows.
    """
    assert token_bytes % WORD_BYTES == 0, (
        f"token_bytes ({token_bytes}) must be a multiple of {WORD_BYTES}"
    )
    u8 = as_token_bytes(values, token_bytes)
    band = token_bytes // WORD_BYTES
    words = u8[:, :band].astype(jnp.uint32)
    for b in range(1, WORD_BYTES):
        words |= u8[:, b * band : (b + 1) * band].astype(jnp.uint32) << (8 * b)
    return words


def _local_row_positions(
    tok: jax.Array,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-row `(sequence, global KV position, valid)` for arbitrary token IDs."""
    num_seqs = seq_lens.shape[0]
    seq_id = jnp.searchsorted(
        query_start_loc[1:], tok, side="right", method="compare_all"
    ).astype(jnp.int32)
    valid = tok < query_start_loc[-1]

    seq_id = jnp.minimum(seq_id, num_seqs - 1)
    pos = tok + (seq_lens - query_start_loc[1:])[seq_id]
    return seq_id, pos, valid


def _row_positions(
    num_tokens: int,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Per-row `(token, sequence, global KV position, valid)`.

    `pos` counts back from `seq_lens`, which already includes this step's
    tokens, so a chunked-prefill continuation resumes at the right offset
    rather than at zero.

    Padded tokens index past the last sequence; `seq_id` is clamped so the
    gathers stay in bounds and `valid` is returned alongside so every caller
    masks them out the same way. Shared by the sharded and unsharded writers so
    they cannot drift into two different notions of where a row belongs.
    """
    tok = jnp.arange(num_tokens, dtype=jnp.int32)
    seq_id, pos, valid = _local_row_positions(tok, seq_lens, query_start_loc)
    return tok, seq_id, pos, valid


def get_page_and_slot(
    num_tokens: int,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    page_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Computes target (page, slot) in the paged KV cache for each token."""
    _, seq_id, pos, valid = _row_positions(num_tokens, seq_lens, query_start_loc)

    block_tables_2d = block_tables.reshape(seq_lens.shape[0], -1)
    max_pages_per_seq = block_tables_2d.shape[1]
    safe_page_idx = jnp.clip(pos // page_size, 0, max_pages_per_seq - 1)
    page = jnp.where(valid, block_tables_2d[seq_id, safe_page_idx], jnp.int32(OOB_PAGE))
    slot = jnp.where(valid, pos % page_size, 0)
    return page, slot


def get_dst_rows(
    num_tokens: int,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    page_size: int,
) -> jax.Array:
    """Computes flattened row indices for SparseCore scatter kernels."""
    page, slot = get_page_and_slot(
        num_tokens, seq_lens, block_tables, query_start_loc, page_size
    )
    return _dst_rows(page, slot, page_size)


def _dst_rows(page: jax.Array, slot: jax.Array, page_size: int) -> jax.Array:
    """(page, slot) -> flattened row, `SKIP_ROW` wherever `page` is `OOB_PAGE`."""
    return jnp.where(page < OOB_PAGE, page * page_size + slot, SKIP_ROW)


# ---------------------------------------------------------------------------
# Cache writers. The SparseCore Pallas kernel in `scatter.py` writes every
# layout it can express; the pure-jnp XLA scatter below covers the rest, and is
# the golden the kernel is checked against (no Pallas, so it runs on CPU).
# ---------------------------------------------------------------------------


def _scatter_rows(
    cache: jax.Array,
    spec: SparseMLAKVCacheSpec,
    values: jax.Array,
    page: jax.Array,
    slot: jax.Array,
) -> jax.Array:
    """Write each token's `values` into the (`page`, `slot`) it lands in."""
    if spec.layout is KVCacheLayout.TENSORCORE:
        src = as_token_bytes(values, spec.token_bytes)
        if spec.cache_type is KVCacheType.NOPE:
            return cache.at[page, slot].set(src.reshape(src.shape[0], *cache.shape[2:]))
        return cache.at[page, slot // spec.kv_packing, slot % spec.kv_packing].set(src)
    if spec.layout is KVCacheLayout.SPARSECORE:
        words = pack_tokens(values, spec.token_bytes)
        rows = cache.reshape(spec.num_pages, spec.page_size, -1)
        return rows.at[page, slot].set(words).reshape(cache.shape)

    raise ValueError(f"unsupported layout {spec.layout}")


def _check_inputs(
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    kv_c_normed: jax.Array,
    nope_spec: SparseMLAKVCacheSpec,
    rope_spec: SparseMLAKVCacheSpec,
) -> None:
    """Assert the caches were allocated from the specs they are written with."""
    assert kv_c_normed.dtype == jnp.float8_e4m3fn, (
        f"sparse MLA kernel requires --kv-cache-dtype fp8 (got {kv_c_normed.dtype})"
    )
    assert kv_cache_nope.dtype == nope_spec.jax_dtype, (
        f"nope cache {kv_cache_nope.dtype} does not match its spec "
        f"{nope_spec.jax_dtype}"
    )
    assert kv_cache_rope.dtype == rope_spec.jax_dtype, (
        f"rope cache {kv_cache_rope.dtype} does not match its spec "
        f"{rope_spec.jax_dtype}"
    )
    assert kv_cache_nope.shape == nope_spec.shape, (
        f"nope cache {kv_cache_nope.shape} does not match its spec {nope_spec.shape}"
    )
    assert kv_cache_rope.shape == rope_spec.shape, (
        f"rope cache {kv_cache_rope.shape} does not match its spec {rope_spec.shape}"
    )
    assert nope_spec.page_size == rope_spec.page_size, (
        f"nope page size {nope_spec.page_size} != rope page size {rope_spec.page_size}"
    )


def update_sparse_mla_kv_cache(
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    kv_c_normed: jax.Array,
    k_pe: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    *,
    nope_spec: SparseMLAKVCacheSpec,
    rope_spec: SparseMLAKVCacheSpec,
) -> tuple[jax.Array, jax.Array]:
    """Scatter this step's new MLA latents into the split sparse-MLA cache.

    Both nope layouts and rope/SPARSECORE go through the SparseCore Pallas
    writer. Rope/TENSORCORE falls back to the XLA scatter: its token is a single
    lane row, not the `WORD_BYTES` rows the TC sublane fold packs into a word.

    Args:
      kv_cache_nope: Nope cache, shaped and typed by `nope_spec`.
      kv_cache_rope: Rope cache, shaped and typed by `rope_spec`.
      kv_c_normed: This step's compressed KV latents (`[num_tokens, lkv_dim]`).
      k_pe: Decoupled RoPE keys (`[num_tokens, rope_dim]`).
      seq_lens: Per-sequence total KV length including the tokens being inserted
        in this step (`[num_seqs]`).
      block_tables: Flattened per-sequence-padded page table
        (`[num_seqs * pages_per_seq]`).
      query_start_loc: Cumulative new-token counts (`[num_seqs + 1]`).
      nope_spec: Layout descriptor the nope cache was allocated from.
      rope_spec: Layout descriptor the rope cache was allocated from.

    Returns:
      The updated (nope, rope) kv caches, same shapes/dtypes as the inputs.
    """
    _check_inputs(kv_cache_nope, kv_cache_rope, kv_c_normed, nope_spec, rope_spec)
    page, slot = get_page_and_slot(
        kv_c_normed.shape[0],
        seq_lens,
        block_tables,
        query_start_loc,
        nope_spec.page_size,
    )
    return _scatter_caches(
        kv_cache_nope,
        kv_cache_rope,
        kv_c_normed,
        k_pe,
        page,
        slot,
        nope_spec,
        rope_spec,
    )


def _scatter_caches(
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    kv_c_normed: jax.Array,
    k_pe: jax.Array,
    page: jax.Array,
    slot: jax.Array,
    nope_spec: SparseMLAKVCacheSpec,
    rope_spec: SparseMLAKVCacheSpec,
) -> tuple[jax.Array, jax.Array]:
    """Write both caches at (`page`, `slot`), with Pallas wherever it can.

    Both nope layouts and rope/SPARSECORE go through the SparseCore Pallas
    writer; rope/TENSORCORE falls back to the XLA scatter. Shared by the
    unsharded and DCP writers, which differ only in how they derive
    (`page`, `slot`).
    """
    # Imported here, not at module scope: `scatter` imports this module back.
    from vllm_torchtpu.kernels.mla import scatter

    dst_rows = _dst_rows(page, slot, nope_spec.page_size)
    kv_cache_nope = scatter.scatter(
        kv_cache_nope, kv_c_normed, nope_spec, dst_rows=dst_rows
    )
    if rope_spec.layout is KVCacheLayout.SPARSECORE:
        kv_cache_rope = scatter.scatter(
            kv_cache_rope, k_pe, rope_spec, dst_rows=dst_rows
        )
    else:
        kv_cache_rope = _scatter_rows(kv_cache_rope, rope_spec, k_pe, page, slot)
    return kv_cache_nope, kv_cache_rope


def update_sparse_mla_kv_cache_jax(
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    kv_c_normed: jax.Array,
    k_pe: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    *,
    nope_spec: SparseMLAKVCacheSpec,
    rope_spec: SparseMLAKVCacheSpec,
) -> tuple[jax.Array, jax.Array]:
    """Pure-XLA `update_sparse_mla_kv_cache`, for every layout.

    Needs no TPU, so it runs on CPU: this is the golden the SparseCore writer is
    checked against, and what `update_sparse_mla_kv_cache` falls back to for the
    layouts that writer cannot express. Same arguments and returns as it.
    """
    _check_inputs(kv_cache_nope, kv_cache_rope, kv_c_normed, nope_spec, rope_spec)
    page, slot = get_page_and_slot(
        kv_c_normed.shape[0],
        seq_lens,
        block_tables,
        query_start_loc,
        nope_spec.page_size,
    )
    return (
        _scatter_rows(kv_cache_nope, nope_spec, kv_c_normed, page, slot),
        _scatter_rows(kv_cache_rope, rope_spec, k_pe, page, slot),
    )


_COMPACT_BLOCK = 256


def _owned_rows_bound(
    num_tokens: int, *, num_seqs: int, dcp_size: int, interleave_size: int
) -> int:
    cap = -(-num_tokens // dcp_size) + 2 * interleave_size * num_seqs
    return (cap + 7) // 8 * 8


def _compact_owned_indices(
    valid: jax.Array,
    *,
    num_seqs: int,
    dcp_size: int,
    interleave_size: int,
) -> tuple[jax.Array, jax.Array] | None:
    num_tokens = valid.shape[0]
    cap = _owned_rows_bound(
        num_tokens,
        num_seqs=num_seqs,
        dcp_size=dcp_size,
        interleave_size=interleave_size,
    )
    if cap >= num_tokens:
        return None
    csum = jnp.cumsum(valid.astype(jnp.int32))
    j = jnp.arange(cap, dtype=jnp.int32)
    if num_tokens >= 2 * _COMPACT_BLOCK and num_tokens % _COMPACT_BLOCK == 0:
        num_blocks = num_tokens // _COMPACT_BLOCK
        csum_2d = csum.reshape(num_blocks, _COMPACT_BLOCK)
        blk = jnp.sum(
            (csum_2d[:, _COMPACT_BLOCK - 1][None, :] <= j[:, None]).astype(jnp.int32),
            axis=1,
        )
        blk_clamped = jnp.minimum(blk, num_blocks - 1)
        in_blk = jnp.sum((csum_2d[blk_clamped] <= j[:, None]).astype(jnp.int32), axis=1)
        idx = jnp.minimum(blk * _COMPACT_BLOCK + in_blk, num_tokens - 1)
    else:
        idx = jnp.sum((csum[None, :] <= j[:, None]).astype(jnp.int32), axis=1)
        idx = jnp.minimum(idx, num_tokens - 1)
    return idx, j < csum[-1]


def _compact_dcp_mla_rows(
    kv_c_normed: jax.Array,
    k_pe: jax.Array,
    page: jax.Array,
    slot: jax.Array,
    *,
    num_seqs: int,
    dcp_size: int,
    interleave_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    compacted = _compact_owned_indices(
        page < OOB_PAGE,
        num_seqs=num_seqs,
        dcp_size=dcp_size,
        interleave_size=interleave_size,
    )
    if compacted is None:
        return kv_c_normed, k_pe, page, slot
    idx, keep = compacted
    page_c = jnp.where(keep, jnp.take(page, idx), jnp.int32(OOB_PAGE))
    slot_c = jnp.where(keep, jnp.take(slot, idx), 0)
    return (
        jnp.take(kv_c_normed, idx, axis=0),
        jnp.take(k_pe, idx, axis=0),
        page_c,
        slot_c,
    )


def update_sparse_mla_kv_cache_dcp(
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    kv_c_normed: jax.Array,
    k_pe: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    dcp_rank: jax.Array,
    *,
    nope_spec: SparseMLAKVCacheSpec,
    rope_spec: SparseMLAKVCacheSpec,
    dcp_size: int,
    interleave_size: int,
    compact_rows: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Scatter this step's MLA latents into a DCP position-sharded cache.

    The DCP twin of `update_sparse_mla_kv_cache`, and the reason DCP is cheap:
    under PCP the cache is replicated, so every rank must be handed *every*
    rank's new rows and the caller pays an all-gather to do it. Under DCP the
    queries are replicated, so `kv_c_normed`/`k_pe` are computed from a
    residual stream every rank already holds -- the rows are local to begin
    with and each rank simply masks its write down to the positions it owns.
    The all-gather disappears entirely rather than moving.

    Ownership is chunk-interleaved: a *virtual* page spans
    `page_size * dcp_size` positions, and rank `r` holds the
    `interleave_size`-sized chunk at offset `r * interleave_size` within each
    `dcp_size * interleave_size` cycle. `block_tables` therefore holds virtual
    page ordinals, resolved against this rank's own shard by
    `virtual % num_pages` -- the same resolution the readers do, which is what
    keeps the indexer cache, the latent cache and the top-k coordinates in one
    coordinate space and lets a global top-k index be handed to a rank as a
    plain local cache index.

    Rows owned by another rank are dropped by aiming them at an out-of-bounds
    page -- the same trick the unsharded scatter uses for padding, and it is
    why this needs no collective and no branch. The writes then go through the
    same SparseCore Pallas writer as the unsharded path, so the caches are
    updated in place in the layout the readers use.

    Must be called from inside a `shard_map`; `dcp_rank` is this shard's index
    along the DCP axis as traced data, so one compiled program serves every
    rank.

    Args:
      kv_cache_nope, kv_cache_rope: this rank's shards, shaped by the specs.
      kv_c_normed, k_pe: this step's rows, in request-major order. Identical on
        every rank; the mask, not the data, is what differs.
      seq_lens, block_tables, query_start_loc: global sequence coordinates,
        replicated. `block_tables` holds *virtual* page ordinals.
      dcp_rank: i32 scalar, this shard's DCP index.
      nope_spec, rope_spec: layout descriptors the shards were allocated from.
      dcp_size, interleave_size: the shard layout.

    Returns:
      The updated (nope, rope) shards, same shapes/dtypes as the inputs.
    """
    _check_inputs(kv_cache_nope, kv_cache_rope, kv_c_normed, nope_spec, rope_spec)
    page, slot = _get_dcp_page_and_slot(
        kv_c_normed.shape[0],
        seq_lens,
        block_tables,
        query_start_loc,
        dcp_rank,
        nope_spec=nope_spec,
        dcp_size=dcp_size,
        interleave_size=interleave_size,
    )
    if compact_rows:
        kv_c_normed, k_pe, page, slot = _compact_dcp_mla_rows(
            kv_c_normed,
            k_pe,
            page,
            slot,
            num_seqs=seq_lens.shape[0],
            dcp_size=dcp_size,
            interleave_size=interleave_size,
        )
    return _scatter_caches(
        kv_cache_nope,
        kv_cache_rope,
        kv_c_normed,
        k_pe,
        page,
        slot,
        nope_spec,
        rope_spec,
    )


def update_sparse_mla_kv_cache_dcp_jax(
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    kv_c_normed: jax.Array,
    k_pe: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    dcp_rank: jax.Array,
    *,
    nope_spec: SparseMLAKVCacheSpec,
    rope_spec: SparseMLAKVCacheSpec,
    dcp_size: int,
    interleave_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Pure-XLA `update_sparse_mla_kv_cache_dcp`, for every layout.

    The DCP twin of `update_sparse_mla_kv_cache_jax`: runs on CPU, and is the
    golden the SparseCore DCP writer is checked against. Same arguments and
    returns as it.
    """
    _check_inputs(kv_cache_nope, kv_cache_rope, kv_c_normed, nope_spec, rope_spec)
    page, slot = _get_dcp_page_and_slot(
        kv_c_normed.shape[0],
        seq_lens,
        block_tables,
        query_start_loc,
        dcp_rank,
        nope_spec=nope_spec,
        dcp_size=dcp_size,
        interleave_size=interleave_size,
    )
    return (
        _scatter_rows(kv_cache_nope, nope_spec, kv_c_normed, page, slot),
        _scatter_rows(kv_cache_rope, rope_spec, k_pe, page, slot),
    )


def _get_dcp_page_and_slot(
    num_tokens: int,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    dcp_rank: jax.Array,
    *,
    nope_spec: SparseMLAKVCacheSpec,
    dcp_size: int,
    interleave_size: int,
) -> tuple[jax.Array, jax.Array]:
    """This rank's (page, slot) per token; `OOB_PAGE` for rows it does not own."""
    # `_check_inputs` asserts it equal to the rope page size, so one
    # (page, slot) addresses both caches.
    page_size = nope_spec.page_size
    if interleave_size % WORD_BYTES:
        raise ValueError(
            f"interleave_size={interleave_size} must be a multiple of "
            f"WORD_BYTES={WORD_BYTES}, or a rope tile -- which packs that many "
            "consecutive tokens into one word -- would straddle two ranks."
        )
    if page_size % interleave_size:
        raise ValueError(
            f"interleave_size={interleave_size} must divide "
            f"page_size={page_size}. Otherwise an interleave chunk straddles "
            "a page boundary and the local index a reader derives from the "
            "global position stops agreeing with the (page, slot) written "
            "here."
        )

    _, seq_id, pos, in_batch = _row_positions(num_tokens, seq_lens, query_start_loc)

    virtual_page_size = jnp.int32(page_size * dcp_size)
    cycle = jnp.int32(dcp_size * interleave_size)
    interleave = jnp.int32(interleave_size)
    offset = pos % virtual_page_size
    owner = (offset % cycle) // interleave
    slot = (offset // cycle) * interleave + (offset % interleave)

    # Virtual ordinal -> physical block in this rank's own shard, exactly as
    # the DCP readers resolve it.
    local_block_tables = jnp.mod(block_tables, jnp.int32(nope_spec.num_pages)).reshape(
        seq_lens.shape[0], -1
    )
    max_pages_per_seq = local_block_tables.shape[1]
    virtual_page = jnp.clip(pos // virtual_page_size, 0, max_pages_per_seq - 1)

    valid = jnp.logical_and(in_batch, owner == dcp_rank)
    page = jnp.where(
        valid, local_block_tables[seq_id, virtual_page], jnp.int32(OOB_PAGE)
    )
    return page, slot


_SHARDED_A2A_FAST_CAP = 128
_SHARDED_COMPACT_BLOCK = 64


def _sharded_a2a_cap(
    rows_local: int,
    *,
    num_seqs: int,
    dcp_size: int,
    interleave_size: int,
    zigzag_bands: int = 1,
) -> int:
    # Exact supremum over `rows_local` rows split across `zigzag_bands`
    # contiguous bands with `<= num_seqs` sequences: at most `zigzag_bands - 1`
    # extra sequence cuts across band gaps, yielding `<= num_seqs + zigzag_bands - 1`
    # spans, each contributing `<= L_s / dcp + il * (dcp - 1) / dcp` to any owner.
    eff_spans = num_seqs + max(zigzag_bands - 1, 0)
    raw = (rows_local + eff_spans * (dcp_size - 1) * interleave_size) // dcp_size
    return min(rows_local, ((raw + 15) // 16) * 16)


def _sharded_a2a_max_send_bound(
    query_start_loc: jax.Array,
    *,
    rows_local: int,
    dcp_size: int,
    interleave_size: int,
    zigzag_bands: int = 1,
) -> jax.Array:
    """Return replicated sound int32 upper bound on `max_{rank, owner} send_count`.

    Every contiguous span of length ``L`` inside a block hits any single DCP
    owner at most ``(L // cycle) * interleave_size + min(L % cycle, interleave_size)``
    times regardless of starting KV position. Summing across active sequences
    and each rank's zigzag bands avoids cross-rank collectives while strictly
    bounding the true worst-case per-owner slot count.
    """
    nb = zigzag_bands if zigzag_bands > 1 else 1
    band = rows_local // nb
    num_blocks = dcp_size * nb
    total_tokens = dcp_size * rows_local
    cycle = jnp.int32(dcp_size * interleave_size)
    il = jnp.int32(interleave_size)
    blk_lo = jnp.arange(num_blocks, dtype=jnp.int32)[:, None] * jnp.int32(band)
    blk_hi = blk_lo + jnp.int32(band)
    q_s = jnp.clip(query_start_loc[:-1], 0, total_tokens)[None, :]
    q_s = q_s.at[:, 0].set(jnp.int32(0))
    q_e = jnp.clip(query_start_loc[1:], 0, total_tokens)[None, :]
    span = jnp.maximum(
        jnp.int32(0), jnp.minimum(q_e, blk_hi) - jnp.maximum(q_s, blk_lo)
    )
    blk_ub = jnp.sum((span // cycle) * il + jnp.minimum(span % cycle, il), axis=1)
    if nb == 1:
        return jnp.max(blk_ub).astype(jnp.int32)
    r_idx = jnp.arange(dcp_size, dtype=jnp.int32)[:, None]
    k_idx = jnp.arange(nb, dtype=jnp.int32)[None, :]
    blk_id = jnp.where(
        (k_idx & 1) == 0,
        k_idx * dcp_size + r_idx,
        (k_idx + 1) * dcp_size - 1 - r_idx,
    )
    return jnp.max(jnp.sum(blk_ub[blk_id], axis=1)).astype(jnp.int32)


def _sharded_a2a_cap_tiers(
    rows_local: int,
    *,
    cap_full: int,
    dcp_size: int,
) -> tuple[int, ...]:
    """Return sorted distinct all_to_all cap tiers ending at `cap_full`."""
    if dcp_size <= 1 or cap_full <= 16:
        return (cap_full,)
    # Avoid non-power-of-2 intermediate tiers (e.g. 72/80/96): irregular row
    # counts split vector tiling in all_to_all/gather/scatter and compile extra
    # conditional branches slower than the 128-row tier.
    base = max(16, ((rows_local + dcp_size - 1) // dcp_size + 15) // 16 * 16)
    candidates = (base, _SHARDED_A2A_FAST_CAP)
    tiers = sorted({c for c in candidates if c < cap_full})
    tiers.append(cap_full)
    return tuple(tiers)


def _build_sharded_dcp_a2a_plan(
    rows_local: int,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    dcp_rank: jax.Array,
    *,
    nope_spec: SparseMLAKVCacheSpec,
    dcp_size: int,
    interleave_size: int,
    cap: int,
    axis_name: str,
    pack_dst_rows: bool,
    zigzag_bands: int = 1,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    page_size = nope_spec.page_size
    if interleave_size % WORD_BYTES:
        raise ValueError(
            f"interleave_size={interleave_size} must be a multiple of "
            f"WORD_BYTES={WORD_BYTES}."
        )
    if page_size % interleave_size:
        raise ValueError(
            f"interleave_size={interleave_size} must divide page_size={page_size}."
        )

    if zigzag_bands > 1:
        if zigzag_bands % 2 != 0 or rows_local % zigzag_bands:
            raise ValueError(
                f"zigzag_bands={zigzag_bands} must be even and divide "
                f"rows_local={rows_local}."
            )
        band = rows_local // zigzag_bands
        ks = jnp.arange(zigzag_bands, dtype=jnp.int32)
        r = jnp.asarray(dcp_rank, dtype=jnp.int32).reshape(())
        blk = jnp.where((ks & 1) == 0, ks * dcp_size + r, (ks + 1) * dcp_size - 1 - r)
        tok = (
            blk[:, None] * jnp.int32(band) + jnp.arange(band, dtype=jnp.int32)[None, :]
        ).reshape(rows_local)
    else:
        tok = dcp_rank * jnp.int32(rows_local) + jnp.arange(rows_local, dtype=jnp.int32)
    seq_id, pos, in_batch = _local_row_positions(tok, seq_lens, query_start_loc)

    virtual_page_size = jnp.int32(page_size * dcp_size)
    cycle = jnp.int32(dcp_size * interleave_size)
    interleave = jnp.int32(interleave_size)
    offset = pos % virtual_page_size
    owner = jnp.where(in_batch, (offset % cycle) // interleave, jnp.int32(-1))
    slot = (offset // cycle) * interleave + (offset % interleave)

    local_bt = jnp.mod(block_tables, jnp.int32(nope_spec.num_pages)).reshape(
        seq_lens.shape[0], -1
    )
    virtual_page = jnp.clip(pos // virtual_page_size, 0, local_bt.shape[1] - 1)
    dst_page = jnp.where(in_batch, local_bt[seq_id, virtual_page], jnp.int32(OOB_PAGE))

    owners = jnp.arange(dcp_size, dtype=jnp.int32)[:, None]
    csum = jnp.cumsum((owner[None, :] == owners).astype(jnp.int32), axis=1)
    counts = csum[:, -1]
    j = jnp.arange(cap, dtype=jnp.int32)[None, :, None]
    if (
        rows_local >= 2 * _SHARDED_COMPACT_BLOCK
        and rows_local % _SHARDED_COMPACT_BLOCK == 0
    ):
        num_blocks = rows_local // _SHARDED_COMPACT_BLOCK
        csum_b = csum.reshape(dcp_size, num_blocks, _SHARDED_COMPACT_BLOCK)
        blk = jnp.sum(
            (csum_b[:, :, _SHARDED_COMPACT_BLOCK - 1][:, None, :] <= j).astype(
                jnp.int32
            ),
            axis=2,
        )
        blk_c = jnp.minimum(blk, num_blocks - 1)
        d_idx = jnp.arange(dcp_size, dtype=jnp.int32)[:, None]
        in_blk = jnp.sum((csum_b[d_idx, blk_c] <= j).astype(jnp.int32), axis=2)
        send_idx = jnp.minimum(blk * _SHARDED_COMPACT_BLOCK + in_blk, rows_local - 1)
    else:
        send_idx = jnp.sum((csum[:, None, :] <= j).astype(jnp.int32), axis=2)
        send_idx = jnp.minimum(send_idx, rows_local - 1)

    keep = jnp.arange(cap, dtype=jnp.int32)[None, :] < counts[:, None]
    send_page = jnp.where(keep, dst_page[send_idx], jnp.int32(OOB_PAGE))
    send_slot = jnp.where(keep, slot[send_idx], jnp.int32(0))
    if pack_dst_rows:
        send_dst = _dst_rows(send_page, send_slot, page_size)
        recv_dst = jax.lax.all_to_all(
            send_dst, axis_name, split_axis=0, concat_axis=0, tiled=False
        )
        return send_idx, recv_dst, None
    recv_page = jax.lax.all_to_all(
        send_page, axis_name, split_axis=0, concat_axis=0, tiled=False
    )
    recv_slot = jax.lax.all_to_all(
        send_slot, axis_name, split_axis=0, concat_axis=0, tiled=False
    )
    return send_idx, recv_page, recv_slot


def _update_sparse_mla_kv_cache_dcp_sharded_impl(
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    kv_c_normed: jax.Array,
    k_pe: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    dcp_rank: jax.Array,
    *,
    axis_name: str,
    nope_spec: SparseMLAKVCacheSpec,
    rope_spec: SparseMLAKVCacheSpec,
    dcp_size: int,
    interleave_size: int,
    use_pallas_scatter: bool,
    zigzag_bands: int = 1,
) -> tuple[jax.Array, jax.Array]:
    _check_inputs(kv_cache_nope, kv_cache_rope, kv_c_normed, nope_spec, rope_spec)
    if dcp_size == 1:
        writer = (
            update_sparse_mla_kv_cache_dcp
            if use_pallas_scatter
            else update_sparse_mla_kv_cache_dcp_jax
        )
        return writer(
            kv_cache_nope,
            kv_cache_rope,
            kv_c_normed,
            k_pe,
            seq_lens,
            block_tables,
            query_start_loc,
            dcp_rank,
            nope_spec=nope_spec,
            rope_spec=rope_spec,
            dcp_size=dcp_size,
            interleave_size=interleave_size,
        )

    rows_local = kv_c_normed.shape[0]
    cap_full = _sharded_a2a_cap(
        rows_local,
        num_seqs=seq_lens.shape[0],
        dcp_size=dcp_size,
        interleave_size=interleave_size,
        zigzag_bands=zigzag_bands,
    )
    use_packed_sc = (
        use_pallas_scatter
        and nope_spec.layout is KVCacheLayout.SPARSECORE
        and rope_spec.layout is KVCacheLayout.SPARSECORE
    )
    send_idx, recv_primary, recv_slot = _build_sharded_dcp_a2a_plan(
        rows_local,
        seq_lens,
        block_tables,
        query_start_loc,
        dcp_rank,
        nope_spec=nope_spec,
        dcp_size=dcp_size,
        interleave_size=interleave_size,
        cap=cap_full,
        axis_name=axis_name,
        pack_dst_rows=use_packed_sc,
        zigzag_bands=zigzag_bands,
    )

    if use_packed_sc:
        from vllm_torchtpu.kernels.mla import (  # noqa: PLC0415 -- circular import with scatter_utils
            scatter_utils,
        )

        packed_nope = pack_tokens(kv_c_normed, nope_spec.token_bytes)
        packed_rope = pack_tokens(k_pe, rope_spec.token_bytes)
        nope_words = packed_nope.shape[-1]
        packed_both = jnp.concatenate([packed_nope, packed_rope], axis=-1)
        total_words = packed_both.shape[-1]
        recv_dst_all = recv_primary

        def _scatter_tier(cap: int) -> tuple[jax.Array, jax.Array]:
            sent = packed_both[send_idx[:, :cap]]
            recv = jax.lax.all_to_all(
                sent, axis_name, split_axis=0, concat_axis=0, tiled=False
            ).reshape(-1, total_words)
            recv_dst = recv_dst_all[:, :cap].reshape(-1)
            nope_out = scatter_utils.scatter_rows(
                kv_cache_nope,
                recv[:, :nope_words],
                recv_dst,
                name=f"scatter_{nope_spec.cache_type.value}_{nope_spec.layout.value}_cache",
            )
            rope_out = scatter_utils.scatter_rows(
                kv_cache_rope,
                recv[:, nope_words:],
                recv_dst,
                name=f"scatter_{rope_spec.cache_type.value}_{rope_spec.layout.value}_cache",
            )
            return nope_out, rope_out

    else:
        assert recv_slot is not None
        recv_page = recv_primary
        kv_nope_dim = kv_c_normed.shape[-1]
        kv_u8 = jnp.concatenate(
            [
                jax.lax.bitcast_convert_type(kv_c_normed, jnp.uint8),
                jax.lax.bitcast_convert_type(k_pe, jnp.uint8),
            ],
            axis=-1,
        )
        total_cols = kv_u8.shape[-1]

        def _scatter_tier(cap: int) -> tuple[jax.Array, jax.Array]:
            sent = kv_u8[send_idx[:, :cap]]
            recv = jax.lax.all_to_all(
                sent, axis_name, split_axis=0, concat_axis=0, tiled=False
            ).reshape(-1, total_cols)
            recv_nope = jax.lax.bitcast_convert_type(
                recv[:, :kv_nope_dim], kv_c_normed.dtype
            )
            recv_rope = jax.lax.bitcast_convert_type(recv[:, kv_nope_dim:], k_pe.dtype)
            page_c = recv_page[:, :cap].reshape(-1)
            slot_c = recv_slot[:, :cap].reshape(-1)
            if use_pallas_scatter:
                return _scatter_caches(
                    kv_cache_nope,
                    kv_cache_rope,
                    recv_nope,
                    recv_rope,
                    page_c,
                    slot_c,
                    nope_spec,
                    rope_spec,
                )
            return (
                _scatter_rows(kv_cache_nope, nope_spec, recv_nope, page_c, slot_c),
                _scatter_rows(kv_cache_rope, rope_spec, recv_rope, page_c, slot_c),
            )

    tiers = _sharded_a2a_cap_tiers(rows_local, cap_full=cap_full, dcp_size=dcp_size)
    if len(tiers) > 1 and rows_local % max(zigzag_bands, 1) == 0:
        # Avoid lax.pmax across ranks and conservative (dcp-1)*il per-seq
        # padding: query_start_loc is replicated so all ranks pick the exact
        # same smallest tier cap >= true worst-case per-owner slot count.
        send_ub = _sharded_a2a_max_send_bound(
            query_start_loc,
            rows_local=rows_local,
            dcp_size=dcp_size,
            interleave_size=interleave_size,
            zigzag_bands=zigzag_bands,
        )
        tier_arr = jnp.asarray(tiers, dtype=jnp.int32)
        branch_idx = jnp.minimum(
            jnp.searchsorted(tier_arr, send_ub, side="left", method="compare_all"),
            jnp.int32(len(tiers) - 1),
        )
        return jax.lax.switch(
            branch_idx,
            tuple(functools.partial(_scatter_tier, c) for c in tiers),
        )
    return _scatter_tier(cap_full)


def update_sparse_mla_kv_cache_dcp_sharded(
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    kv_c_normed: jax.Array,
    k_pe: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    dcp_rank: jax.Array,
    *,
    axis_name: str,
    nope_spec: SparseMLAKVCacheSpec,
    rope_spec: SparseMLAKVCacheSpec,
    dcp_size: int,
    interleave_size: int,
    zigzag_bands: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Route sequence-sharded FP8 latents via owner all_to_all into DCP caches.

    Shapes: `kv_c_normed` `[T/dcp, L]` and `k_pe` `[T/dcp, R]` -> updated DCP shards.
    """
    return _update_sparse_mla_kv_cache_dcp_sharded_impl(
        kv_cache_nope,
        kv_cache_rope,
        kv_c_normed,
        k_pe,
        seq_lens,
        block_tables,
        query_start_loc,
        dcp_rank,
        axis_name=axis_name,
        nope_spec=nope_spec,
        rope_spec=rope_spec,
        dcp_size=dcp_size,
        interleave_size=interleave_size,
        use_pallas_scatter=True,
        zigzag_bands=zigzag_bands,
    )


def update_sparse_mla_kv_cache_dcp_sharded_jax(
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    kv_c_normed: jax.Array,
    k_pe: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    dcp_rank: jax.Array,
    *,
    axis_name: str,
    nope_spec: SparseMLAKVCacheSpec,
    rope_spec: SparseMLAKVCacheSpec,
    dcp_size: int,
    interleave_size: int,
    zigzag_bands: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Pure-XLA twin of `update_sparse_mla_kv_cache_dcp_sharded` for CPU tests."""
    return _update_sparse_mla_kv_cache_dcp_sharded_impl(
        kv_cache_nope,
        kv_cache_rope,
        kv_c_normed,
        k_pe,
        seq_lens,
        block_tables,
        query_start_loc,
        dcp_rank,
        axis_name=axis_name,
        nope_spec=nope_spec,
        rope_spec=rope_spec,
        dcp_size=dcp_size,
        interleave_size=interleave_size,
        use_pallas_scatter=False,
        zigzag_bands=zigzag_bands,
    )
