"""Paged attention, exact index selection and index-cache publication."""

import contextlib
import dataclasses
import functools
import math
from collections.abc import Iterator
from typing import Any, cast

import jax
import jax.numpy as jnp
import torch

from vllm_torchtpu.kernels.deepseek_v41 import (
    index_word_scores,
    indexer,
    tuned_block_sizes,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection import (
    decode as decode_index_selection,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection import (
    scores as prefill_index_scores,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection import (
    streaming_topk,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection import (
    tuned_block_sizes as prefill_tuning,
)

from .cache import (
    ROPE_WORDS,
    ROW_WORDS,
    _cache_num_rows,
    gather_cache_page_values,
    gather_cache_page_words,
    gather_cache_pieces,
    gather_cache_values,
    merge_pieces,
    scatter_cache_values,
    scatter_index_keys,
    split_halves,
    split_pieces,
)

_attention_op = None
_attention_2bank_op = None
_fp8_attention_op = None
_fp8_attention_2bank_op = None
_fp8_attention_qo_op = None
_fp8_attention_qo_2bank_op = None
_gather_bf16_op = None
_gather_fp8_op = None
_attention_tp_size = 1


@dataclasses.dataclass
class _GatheredKVCache:
    """One live gather group; records are immutable within a forward stage."""

    source: Any = None
    records: torch.Tensor | None = None
    layout: tuple = ()
    values: tuple[torch.Tensor, torch.Tensor | None] | None = None
    valid_counts: torch.Tensor | None = None

    def get(
        self, source: Any, records: torch.Tensor, layout: tuple
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        if self.source is source and self.records is records and self.layout == layout:
            return self.values
        return None

    def put(
        self,
        source: Any,
        records: torch.Tensor,
        layout: tuple,
        values: tuple[torch.Tensor, torch.Tensor | None],
        *,
        valid_counts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self.source, self.records, self.layout, self.values = (
            source,
            records,
            layout,
            values,
        )
        self.valid_counts = valid_counts
        return values


_active_gathered_kv: _GatheredKVCache | None = None


@contextlib.contextmanager
def gathered_kv_cache_scope() -> Iterator[None]:
    """Reuse KV payloads across layers, never across scheduler steps/stages."""
    global _active_gathered_kv
    previous = _active_gathered_kv
    _active_gathered_kv = _GatheredKVCache()
    try:
        yield
    finally:
        _active_gathered_kv = previous


def _attention_state(q):
    tokens, heads, _ = q.shape
    return (
        jnp.full((tokens, heads), -1e30, jnp.float32),
        jnp.zeros((tokens, heads), jnp.float32),
        jnp.zeros(q.shape, jnp.float32),
    )


def _attend(q, pool, records, state, scale, tile=64, gathered=None):
    """Online-softmax accumulate one record set into (max, denominator, numerator)."""
    tokens, _, dim = q.shape
    width = records.shape[1]
    if not width:
        return state
    records = jnp.pad(records, ((0, 0), (0, -width % tile)), constant_values=-1)

    # Gather BF16 rows once outside; keep FP32 scratch tiled. Rows and q use
    # the split element order (no per-element unpack gather), as stacked
    # pieces contracted in one dot; attention is invariant to a shared
    # permutation of the head dimension. Keep `pieces` on axis 0 (`pthd`) so
    # `(heads, 128)` stays on the minor tiled axes without sublane padding.
    if gathered is None:
        gathered = gather_cache_pieces(pool, records.reshape(-1), dim)
    count = gathered.shape[0]
    gathered = gathered.reshape(count, tokens, records.shape[1], -1)

    q = split_pieces(q, count).astype(jnp.float32)
    maximum, denominator, numerator = state
    state = maximum, denominator, split_pieces(numerator, count)

    def step(index, carry):
        maximum, denominator, numerator = carry
        addresses = jax.lax.dynamic_slice_in_dim(records, index * tile, tile, axis=1)
        valid = addresses >= 0
        kv = jnp.where(
            valid[None, ..., None],
            jax.lax.dynamic_slice_in_dim(gathered, index * tile, tile, axis=2),
            0,
        )
        scores = (
            jnp.einsum(
                "pthd,ptkd->thk",
                q,
                kv.astype(jnp.float32),
                preferred_element_type=jnp.float32,
            )
            * scale
        )
        scores = jnp.where(valid[:, None], scores, -jnp.inf)
        new_max = jnp.maximum(maximum, scores.max(-1))
        correction = jnp.exp(maximum - new_max)
        exponentials = jnp.exp(scores - new_max[..., None])
        denominator = denominator * correction + exponentials.sum(-1)
        numerator = numerator * correction[None, ..., None] + jnp.einsum(
            "thk,ptkd->pthd",
            exponentials.astype(jnp.bfloat16),
            kv,
            preferred_element_type=jnp.float32,
        )
        return new_max, denominator, numerator

    maximum, denominator, numerator = jax.lax.fori_loop(
        0, records.shape[1] // tile, step, state
    )
    return maximum, denominator, merge_pieces(numerator)


def _window_union(records, tile):
    """Per token tile, one record list covering its tokens' sliding windows.

    Window column j of token t is position ``p_t - W + 1 + j``, so a request's
    consecutive tokens ``a..b`` need ``records[a, :b - a]`` followed by
    ``records[b]``. ``b`` is the tile's last valid token and ``a`` the first
    token whose own row (column W - 1) lies in b's window, i.e. the same
    request. Returns ``(union [n, tile + W], mask [n, tile, tile + W],
    uncovered [n, tile])``: ``mask`` marks the union columns in each token's
    own records, ``uncovered`` tokens have a valid record outside the union.
    """
    tokens, width = records.shape
    count = tokens // tile
    rows = records.reshape(count, tile, width)
    own = rows[..., -1]
    live = own >= 0
    last = tile - 1 - jnp.argmax(live[:, ::-1], axis=1)
    last_rows = jnp.take_along_axis(rows, last[:, None, None], axis=1)[:, 0]
    same = live & (own[..., None] == last_rows[:, None, :]).any(-1)
    first = jnp.argmax(same, axis=1)
    first_rows = jnp.take_along_axis(rows, first[:, None, None], axis=1)[:, 0]
    head = jnp.where(
        jnp.arange(tile)[None] < (last - first)[:, None], first_rows[:, :tile], -1
    )
    union = jnp.concatenate([head, last_rows], axis=1)
    # Token u of the block maps window column j to union column
    # u - a + j (head) while that is below b - a, else tile + j - (b - u).
    u = jnp.arange(tile)[None, :, None]
    j = jnp.arange(width)[None, None, :]
    a = first[:, None, None]
    b = last[:, None, None]
    columns = jnp.where(j < b - u, u - a + j, tile + j - (b - u))
    in_block = (u[..., 0] >= first[:, None]) & (u[..., 0] <= last[:, None])
    columns = jnp.where(in_block[..., None], columns, 0)
    mapped = jnp.take_along_axis(
        jnp.broadcast_to(union[:, None], (count, tile, union.shape[1])),
        jnp.clip(columns, 0, union.shape[1] - 1),
        axis=2,
    )
    # Verify instead of trusting contiguity: every own row must land on
    # the same physical row.
    covered = in_block & (mapped == rows).all(-1)
    c = jnp.arange(union.shape[1])[None, None, :]
    band = ((c >= u - a) & (c < b - a)) | (
        (c >= tile) & (c <= tile + width - 1 - (b - u))
    )
    mask = covered[..., None] & band & (union[:, None, :] >= 0)
    uncovered = (rows >= 0).any(-1) & ~covered
    return union, mask, uncovered


def _attend_window(
    q, pool, records, state, scale, tile=64, fallback=64, max_requests=None
):
    """`_attend` for sliding-window records, gathering per token tile.

    Prefill tokens of one request share most of their window, so the per
    token ``[tokens, W]`` gather reads each row up to W times. Here each
    64-token tile gathers one ``tile + W`` union and attends with a
    membership mask. Up to ``fallback`` uncovered tokens (other requests in
    the tile: decode tokens, a previous chunk's tail) are recomputed with
    `_attend`; if more are uncovered, the whole batch uses `_attend`.

    ``max_requests`` (static bound on requests per batch) removes both
    conditionals: only tokens of a request other than their tile's last one
    are uncovered, at most ``tile - 1`` per request boundary, so a patch of
    that static capacity always runs. XLA completes in-flight SparseCore
    gathers before a conditional holding SparseCore ops; without one, the
    CSA gathers can overlap SWA. Tokens beyond the bound (impossible for
    contiguous request tokens) come out NaN rather than silently wrong.
    """
    tokens, heads, dim = q.shape
    width = records.shape[1]
    if (
        tokens % tile
        or tokens < 2 * tile
        or (max_requests is None and tokens % 6 == 0)
        or width % 64
        or width > 512
    ):
        return _attend(q, pool, records, state, scale)
    union, mask, uncovered = _window_union(records, tile)
    count = union.shape[0]
    if max_requests is not None:
        fallback = min(tokens, -(-(tile - 1) * (max_requests - 1) // 8) * 8)

    def fast(_):
        gathered = gather_cache_pieces(pool, union.reshape(-1), dim)
        pieces = gathered.shape[0]
        kv = gathered.reshape(pieces, count, union.shape[1], -1)
        kv = jnp.where((union >= 0)[None, ..., None], kv, 0)

        def split(x):
            return split_pieces(x.reshape(count, tile, heads, dim), pieces)

        maximum, denominator, numerator = state
        shape = (count, tile, heads)
        maximum = maximum.reshape(shape)
        denominator = denominator.reshape(shape)
        numerator = split(numerator)
        scores = (
            jnp.einsum(
                "pnthd,pnkd->nthk",
                split(q).astype(jnp.float32),
                kv.astype(jnp.float32),
                preferred_element_type=jnp.float32,
            )
            * scale
        )
        scores = jnp.where(mask[:, :, None], scores, -jnp.inf)
        new_max = jnp.maximum(maximum, scores.max(-1))
        correction = jnp.exp(maximum - new_max)
        exponentials = jnp.exp(scores - new_max[..., None])
        denominator = denominator * correction + exponentials.sum(-1)
        numerator = numerator * correction[None, ..., None] + jnp.einsum(
            "nthk,pnkd->pnthd",
            exponentials.astype(jnp.bfloat16),
            kv,
            preferred_element_type=jnp.float32,
        )
        result = (
            new_max.reshape(tokens, heads),
            denominator.reshape(tokens, heads),
            merge_pieces(numerator).reshape(tokens, heads, dim),
        )

        # Recompute uncovered tokens from their own records.
        def patch(result):
            index = jnp.nonzero(
                uncovered.reshape(-1), size=fallback, fill_value=tokens
            )[0]
            source = jnp.minimum(index, tokens - 1)
            sub_records = jnp.where((index < tokens)[:, None], records[source], -1)
            sub = _attend(
                q[source], pool, sub_records, tuple(s[source] for s in state), scale
            )
            return tuple(
                full.at[index].set(part, mode="drop") for full, part in zip(result, sub)
            )

        if max_requests is None:
            return jax.lax.cond(uncovered.any(), patch, lambda result: result, result)
        if fallback:
            result = patch(result)
        flat = uncovered.reshape(-1)
        overflow = flat & (jnp.cumsum(flat) > fallback)
        return tuple(
            jnp.where(overflow.reshape(-1, *([1] * (r.ndim - 1))), jnp.nan, r)
            for r in result
        )

    if max_requests is not None:
        return fast(None)
    return jax.lax.cond(
        uncovered.sum() > fallback,
        lambda _: _attend(q, pool, records, state, scale),
        fast,
        None,
    )


def _swa_attention(q, pool, records, sink, scale):
    """Fuse pure SWA without materializing split Q or merged O layouts."""
    from vllm_torchtpu.kernels.mla.swa_attention import attend_swa

    def direct(q, records):
        kv = gather_cache_pieces(pool, records.reshape(-1), q.shape[-1])
        kv = kv.reshape(4, *records.shape, 128)
        return attend_swa(q, kv, (records >= 0)[:, None, :], sink, scale)

    tokens = q.shape[0]
    if tokens < 128 or tokens % 64 or tokens % 6 == 0 or records.shape[1] != 128:
        return direct(q, records)
    union, mask, uncovered = _window_union(records, 64)

    def shared(_):
        # A 192-row union is padded to one 256-row MXU RHS tile.
        padded = jnp.pad(union, ((0, 0), (0, 256 - union.shape[1])), constant_values=-1)
        kv = gather_cache_pieces(pool, padded.reshape(-1), q.shape[-1])
        kv = kv.reshape(4, *padded.shape, 128)
        padded_mask = jnp.pad(mask, ((0, 0), (0, 0), (0, 256 - union.shape[1])))
        result = attend_swa(q, kv, padded_mask, sink, scale)

        def patch(result):
            index = jnp.nonzero(uncovered.reshape(-1), size=64, fill_value=tokens)[0]
            source = jnp.minimum(index, tokens - 1)
            sub_records = jnp.where((index < tokens)[:, None], records[source], -1)
            return result.at[index].set(direct(q[source], sub_records), mode="drop")

        return jax.lax.cond(uncovered.any(), patch, lambda x: x, result)

    return jax.lax.cond(
        uncovered.sum() > 64, lambda _: direct(q, records), shared, None
    )


def paged_attention_jax(
    q: jax.Array,
    swa_pool: jax.Array,
    main_pool: jax.Array,
    swa_records: jax.Array,
    main_records: jax.Array,
    sink: jax.Array,
    scale: float,
    gathered: jax.Array | None = None,
) -> jax.Array:
    from vllm_torchtpu.kernels.mla.swa_attention import supports_swa

    if main_records.shape[1] == 0 and supports_swa(q, swa_records):
        return _swa_attention(q, swa_pool, swa_records, sink, scale)
    # SWA widths in the released model are multiples of 64, preserving the
    # reference's accumulation boundaries when compressed rows follow.
    maximum, denominator, numerator = _attend(
        q,
        main_pool,
        main_records,
        _attend_window(q, swa_pool, swa_records, _attention_state(q), scale),
        scale,
        gathered=gathered,
    )
    denominator += jnp.exp(sink.astype(jnp.float32)[None] - maximum)
    return (numerator / denominator[..., None]).astype(q.dtype)


def _compact_selected(records: jax.Array) -> jax.Array:
    """Sort valid (>= 0) indices ascending and move -1 to the tail."""
    limit = jnp.iinfo(jnp.int32).max
    records = jnp.sort(jnp.where(records >= 0, records, limit), axis=-1)
    return jnp.where(records == limit, -1, records)


def compact_records(records: jax.Array) -> jax.Array:
    """Pad to 128 columns and move -1 to the tail: sparse_mla's valid prefix."""
    width = records.shape[1]
    padded = max(128, -(-width // 128) * 128)
    records = jnp.pad(records, ((0, 0), (0, padded - width)), constant_values=-1)
    return _compact_selected(records)


def _pad_compacted_records(records: jax.Array) -> jax.Array:
    """Pad already-compacted records to a 128-column multiple with -1."""
    width = records.shape[1]
    padded = max(128, -(-width // 128) * 128)
    if padded == width:
        return records
    return jnp.pad(records, ((0, 0), (0, padded - width)), constant_values=-1)


def compute_chunk_union_gather_indices(
    compacted_records: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    *,
    page_size: int,
    total_num_pages: int,
    chunk_size: int = 128,
    union_cap: int = 4096,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute per-chunk sorted unique physical KV row union and inverse map.

    Deduplicates logical ``(seq_id, record)`` entries across each ``chunk_size``
    query chunk via a 128-lane-aligned associative ``int32`` ``.add(1)``
    presence table (offloaded to SparseCore with zero ``jnp.sort``), translates
    only the ``union_cap`` unique logical entries through ``page_indices``
    (eliminating ``gather_page_ids`` on the full ``[num_tokens, topk]`` tensor),
    and builds the exact inverse map ``inv_idx`` via ``jnp.cumsum``.

    Returns:
      union_phys: ``int32[num_chunks, union_cap]`` physical KV rows per chunk.
      inv_idx: ``int32[num_chunks, chunk_size, topk]`` index into ``union_phys``.
      num_unique: ``int32[num_chunks]`` count of unique valid KV rows per chunk.
    """
    num_tokens, topk = compacted_records.shape
    num_chunks = (num_tokens + chunk_size - 1) // chunk_size
    padded_tokens = num_chunks * chunk_size
    max_num_seqs = cu_q_lens.shape[0] - 1
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    logical_rows_per_seq = pages_per_seq * page_size
    total_logical_rows = max_num_seqs * logical_rows_per_seq
    table_size = ((total_logical_rows + 128) // 128) * 128

    tokens_per_seq = cu_q_lens[1:] - cu_q_lens[:-1]
    seq_ids_segment = jnp.repeat(
        jnp.arange(max_num_seqs, dtype=jnp.int32),
        tokens_per_seq,
        total_repeat_length=num_tokens,
    )
    if padded_tokens > num_tokens:
        pad_t = padded_tokens - num_tokens
        compacted_records = jnp.pad(
            compacted_records, ((0, pad_t), (0, 0)), constant_values=-1
        )
        seq_ids_segment = jnp.pad(seq_ids_segment, (0, pad_t), constant_values=0)

    is_pad = compacted_records < 0
    combined_logical = seq_ids_segment[:, None] * logical_rows_per_seq + jnp.clip(
        compacted_records, 0, max(0, logical_rows_per_seq - 1)
    )
    chunk_log = combined_logical.reshape(num_chunks, chunk_size * topk)
    chunk_pad = is_pad.reshape(num_chunks, chunk_size * topk)

    safe_log = jnp.where(chunk_pad, total_logical_rows, chunk_log)
    chunk_rows = jnp.arange(num_chunks, dtype=jnp.int32)[:, None]
    counts = (
        jnp.zeros((num_chunks, table_size), dtype=jnp.int32)
        .at[chunk_rows, safe_log]
        .add(1)
    )
    present = counts[:, :total_logical_rows] > 0
    num_unique = jnp.sum(present, axis=1)

    union_log = jax.vmap(
        lambda p: jnp.nonzero(p, size=union_cap, fill_value=0)[0].astype(jnp.int32)
    )(present)
    if jax.default_backend() == "tpu":
        from vllm_torchtpu.kernels.deepseek_v4.core_attention import sparse_mla

        union_page_ids = sparse_mla.gather_page_ids(
            page_indices,
            union_log // page_size,
            jnp.zeros((num_chunks,), dtype=jnp.int32),
            1,
            block_tokens=sparse_mla._largest_divisor(num_chunks, 8),
        )
    else:
        union_page_ids = page_indices[union_log // page_size]
    union_phys_valid = union_page_ids * page_size + (union_log % page_size)

    u_arange = jnp.arange(union_cap, dtype=jnp.int32)[None, :]
    flat_u = jnp.arange(num_chunks * union_cap, dtype=jnp.int32).reshape(
        num_chunks, union_cap
    )
    scattered_phys = ((flat_u * 104729) % total_num_pages) * page_size + (
        (flat_u * 15485863) % page_size
    )
    union_phys = jnp.where(
        u_arange < num_unique[:, None], union_phys_valid, scattered_phys
    )

    row_to_slot = jnp.cumsum(present.astype(jnp.int32), axis=1) - 1
    clamped_log = jnp.where(chunk_pad, 0, chunk_log)
    inv_idx = jnp.clip(
        jnp.take_along_axis(row_to_slot, clamped_log, axis=1),
        0,
        union_cap - 1,
    ).reshape(num_chunks, chunk_size, topk)
    return union_phys, inv_idx, num_unique


def unpack_union_rope_words(u_rope: jax.Array, topk: int) -> jax.Array:
    """Unpack ``csa_gather``'s ``(union_cap // 4, 128)`` to ``(union_cap, 32)``."""
    union_cap = u_rope.shape[0] * 4
    num_periods = union_cap // topk
    half_rows = topk // 4
    periods = u_rope.reshape(num_periods, half_rows, 128)

    def _unpack_half(h: jax.Array) -> jax.Array:
        lo, hi = h[:, :, :32], h[:, :, 32:64]
        even = jnp.bitwise_or(jnp.bitwise_and(lo, 0xFFFF), jnp.left_shift(hi, 16))
        odd = jnp.bitwise_or(
            jax.lax.shift_right_logical(lo, 16),
            jnp.bitwise_and(hi, jnp.int32(-65536)),
        )
        return jnp.stack([even, odd], axis=2).reshape(num_periods, half_rows * 2, 32)

    return jnp.concatenate(
        [_unpack_half(periods[:, :, :64]), _unpack_half(periods[:, :, 64:])],
        axis=1,
    ).reshape(union_cap, 32)


def repack_chunk_rope_words(r32: jax.Array) -> jax.Array:
    """Pack ``(chunk_size, topk, 32)`` words into ``(chunk_size, topk // 4, 128)``."""
    chunk_size, topk, _ = r32.shape
    half_k = topk // 2
    half_rows = topk // 4

    def _pack_half(h: jax.Array) -> jax.Array:
        pairs = h.reshape(chunk_size, half_rows, 2, 32)
        even, odd = pairs[:, :, 0, :], pairs[:, :, 1, :]
        c0 = jnp.bitwise_or(jnp.bitwise_and(even, 0xFFFF), jnp.left_shift(odd, 16))
        c1 = jnp.bitwise_or(
            jax.lax.shift_right_logical(even, 16),
            jnp.bitwise_and(odd, jnp.int32(-65536)),
        )
        return jnp.concatenate([c0, c1], axis=2)

    return jnp.concatenate(
        [_pack_half(r32[:, :half_k, :]), _pack_half(r32[:, half_k:, :])],
        axis=2,
    )


def remap_union_kv_to_chunk(
    u_nope: jax.Array,
    u_rope: jax.Array,
    inv_idx: jax.Array,
    topk: int,
) -> tuple[jax.Array, jax.Array]:
    """Remap gathered union ``(u_nope, u_rope)`` to ``(chunk_size, topk, ...)``."""
    g_nope = u_nope[inv_idx]
    u_rope_words = unpack_union_rope_words(u_rope, topk)
    g_rope = repack_chunk_rope_words(u_rope_words[inv_idx])
    return g_nope, g_rope


def _paged_pool_shape(
    cache: jax.Array | tuple[jax.Array, jax.Array],
) -> tuple[int, int, int]:
    if isinstance(cache, tuple):
        cache_lo, cache_hi = cache
        return (
            cache_lo.shape[0] + cache_hi.shape[0],
            cache_lo.shape[1],
            cache_lo.shape[2],
        )
    return cache.shape


def _reshape_paged_pool(
    pool: jax.Array | tuple[jax.Array, jax.Array], page_size: int
) -> jax.Array | tuple[jax.Array, jax.Array]:
    if isinstance(pool, tuple):
        pool_lo, pool_hi = pool
        return (
            pool_lo.reshape(pool_lo.shape[0] // page_size, page_size, pool_lo.shape[1]),
            pool_hi.reshape(pool_hi.shape[0] // page_size, page_size, pool_hi.shape[1]),
        )
    return pool.reshape(pool.shape[0] // page_size, page_size, pool.shape[1])


def _csa_gather_pool(
    nope_cache: jax.Array | tuple[jax.Array, jax.Array],
    rope_cache: jax.Array | tuple[jax.Array, jax.Array],
    indices: jax.Array,
    num_valid_indices: jax.Array | int | None = None,
    *,
    top_k: int,
) -> tuple[jax.Array, jax.Array]:
    """Gather NoPE and RoPE rows from a 1-bank or 2-bank paged pool."""
    if isinstance(nope_cache, tuple) and jax.default_backend() == "tpu":
        from vllm_torchtpu.kernels.deepseek_v4.core_attention import csa_gather

        return csa_gather.csa_gather(
            nope_cache,
            rope_cache,
            indices,
            num_valid_indices=num_valid_indices,
            top_k=top_k,
        )
    if isinstance(nope_cache, tuple):
        assert isinstance(rope_cache, tuple)
        nope_lo, nope_hi = nope_cache
        rope_lo, rope_hi = rope_cache
        lo_tokens = nope_lo.shape[0] * nope_lo.shape[1]
        hi_tokens = nope_hi.shape[0] * nope_hi.shape[1]
        out_size = indices.size
        num_periods = out_size // top_k
        in_hi = indices >= lo_tokens
        spread = jnp.arange(out_size, dtype=indices.dtype) * 104729
        idx_lo = jnp.where(in_hi | (indices < 0), spread % lo_tokens, indices)
        idx_hi = jnp.where(
            in_hi,
            jnp.minimum(indices - lo_tokens, hi_tokens - 1),
            spread % hi_tokens,
        )
        if num_valid_indices is None:
            valid_mask = jnp.ones((out_size,), dtype=jnp.bool_)
        else:
            valid_mask = jnp.arange(out_size, dtype=jnp.int32) < num_valid_indices
        period_ids = jnp.arange(num_periods, dtype=jnp.int32)
        period_lo = jnp.any((valid_mask & ~in_hi).reshape(num_periods, top_k), axis=1)
        period_hi = jnp.any((valid_mask & in_hi).reshape(num_periods, top_k), axis=1)
        valid_lo = jnp.where(
            jnp.any(period_lo),
            (jnp.max(jnp.where(period_lo, period_ids, 0)) + 1) * top_k,
            jnp.int32(0),
        )
        valid_hi = jnp.where(
            jnp.any(period_hi),
            (jnp.max(jnp.where(period_hi, period_ids, 0)) + 1) * top_k,
            jnp.int32(0),
        )
        nope_0, rope_0 = _csa_gather_pool(
            nope_lo, rope_lo, idx_lo, num_valid_indices=valid_lo, top_k=top_k
        )
        nope_1, rope_1 = _csa_gather_pool(
            nope_hi, rope_hi, idx_hi, num_valid_indices=valid_hi, top_k=top_k
        )
        nope_out = jnp.where(in_hi[:, None], nope_1, nope_0)
        # Select in the packed layout. Unpacking both payloads and repacking
        # them expands the hot gather path into many tiled layout conversions.
        # Each selected token owns both BF16 halves of its original words;
        # packing an all-bits mask gives the exact bit ownership in the output.
        word_mask = jnp.broadcast_to(
            jnp.where(in_hi[:, None], jnp.int32(-1), jnp.int32(0)),
            (out_size, ROPE_WORDS),
        )
        packed_mask = repack_chunk_rope_words(
            word_mask.reshape(num_periods, top_k, -1)
        ).reshape(rope_0.shape)
        rope_out = rope_0 ^ ((rope_0 ^ rope_1) & packed_mask)
        return nope_out, rope_out

    assert not isinstance(rope_cache, tuple)
    if jax.default_backend() == "tpu":
        from vllm_torchtpu.kernels.deepseek_v4.core_attention import csa_gather

        return csa_gather.csa_gather(
            nope_cache,
            rope_cache,
            indices,
            num_valid_indices=num_valid_indices,
            top_k=top_k,
        )
    total_num_pages, page_size, nope_lanes = nope_cache.shape
    _, _, rope_lanes = rope_cache.shape
    flat_nope = nope_cache.reshape(total_num_pages * page_size, nope_lanes)
    flat_rope_words = rope_cache.reshape(total_num_pages * page_size, rope_lanes // 4)
    u_nope = flat_nope[indices]
    u_rope = repack_chunk_rope_words(
        flat_rope_words[indices].reshape(indices.size // top_k, top_k, rope_lanes // 4)
    ).reshape(indices.size // 4, rope_lanes)
    return u_nope, u_rope


def gather_sparse_kv_chunk(
    cache_kv_nope: jax.Array | tuple[jax.Array, jax.Array],
    cache_kv_rope: jax.Array | tuple[jax.Array, jax.Array],
    chunk_records: jax.Array,
    chunk_seq_ids: jax.Array,
    page_indices: jax.Array,
    *,
    max_num_seqs: int,
    start_pos: int,
    batch_end: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Gather packed selected KV, preserving SparseMLA's per-token RoPE layout."""
    total_num_pages, page_size, _ = _paged_pool_shape(cache_kv_nope)
    chunk_size, topk = chunk_records.shape
    seq_page_ids = chunk_records // page_size
    token_offset = chunk_records % page_size
    if jax.default_backend() == "tpu":
        from vllm_torchtpu.kernels.deepseek_v4.core_attention import sparse_mla

        page_ids = sparse_mla.gather_page_ids(
            page_indices,
            seq_page_ids,
            chunk_seq_ids,
            max_num_seqs,
            # The page gather pads tail rows internally. Keep TPU blocks aligned
            # even when the whole stage has, for example, twelve token rows.
            block_tokens=8,
        )
    else:
        page_table = page_indices.reshape(max_num_seqs, -1)
        page_ids = page_table[chunk_seq_ids[:, None], jnp.clip(seq_page_ids, 0)]

    is_padding = chunk_records == -1
    flat_idx = start_pos * topk + jnp.arange(
        chunk_size * topk, dtype=jnp.int32
    ).reshape(chunk_size, topk)
    scattered_page_ids = (flat_idx * 104729) % total_num_pages
    scattered_token_offset = (flat_idx * 15485863) % page_size
    page_ids = jnp.where(is_padding, scattered_page_ids, page_ids)
    token_offset = jnp.where(is_padding, scattered_token_offset, token_offset)
    direct_indices = (page_ids * page_size + token_offset).reshape(-1)
    g_nope, g_rope = _csa_gather_pool(
        cache_kv_nope,
        cache_kv_rope,
        direct_indices,
        num_valid_indices=jnp.maximum(0, batch_end - start_pos) * topk,
        top_k=topk,
    )
    return (
        g_nope.reshape(chunk_size, topk, -1),
        g_rope.reshape(chunk_size, topk // 4, -1),
    )


def gather_sparse_kv_chunk_with_union_fallback(
    cache_kv_nope: jax.Array | tuple[jax.Array, jax.Array],
    cache_kv_rope: jax.Array | tuple[jax.Array, jax.Array],
    union_phys_chunk: jax.Array,
    inv_idx_chunk: jax.Array,
    num_unique_chunk: jax.Array,
    chunk_records: jax.Array,
    chunk_seq_ids: jax.Array,
    page_indices: jax.Array,
    *,
    max_num_seqs: int,
    start_pos: int,
    batch_end: jax.Array,
    topk: int,
    use_union: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Gather chunk KV via union table, falling back to direct gather on overflow.

    When ``num_unique_chunk <= union_cap``, gathers only the ``union_cap`` unique
    physical rows and remaps via ``inv_idx_chunk``. If ``num_unique_chunk >
    union_cap``, ``jax.lax.cond`` falls back to direct ``[chunk_size, topk]``
    gather so that ``num_unique > union_cap`` never truncates KV rows.
    """
    union_cap = union_phys_chunk.shape[0]
    has_valid = batch_end > start_pos

    def _union_branch(_: None) -> tuple[jax.Array, jax.Array]:
        from vllm_torchtpu.kernels.deepseek_v4.core_attention import sparse_mla

        num_valid_union = jnp.where(
            has_valid,
            jnp.minimum(union_cap, sparse_mla.align_to(num_unique_chunk, topk)),
            0,
        )
        u_nope, u_rope = _csa_gather_pool(
            cache_kv_nope,
            cache_kv_rope,
            union_phys_chunk,
            num_valid_indices=num_valid_union,
            top_k=topk,
        )
        return remap_union_kv_to_chunk(u_nope, u_rope, inv_idx_chunk, topk)

    def _direct_branch(_: None) -> tuple[jax.Array, jax.Array]:
        return gather_sparse_kv_chunk(
            cache_kv_nope,
            cache_kv_rope,
            chunk_records,
            chunk_seq_ids,
            page_indices,
            max_num_seqs=max_num_seqs,
            start_pos=start_pos,
            batch_end=batch_end,
        )

    if not use_union:
        return _direct_branch(None)
    return jax.lax.cond(
        num_unique_chunk <= union_cap,
        _union_branch,
        _direct_branch,
        operand=None,
    )


def sparse_ragged_paged_attention_union_gather(
    q: jax.Array,
    cache_kv_nope: jax.Array | tuple[jax.Array, jax.Array],
    cache_kv_rope: jax.Array | tuple[jax.Array, jax.Array],
    topk_indices: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    attention_sinks: jax.Array,
    swa_accumution: jax.Array,
    swa_l: jax.Array,
    swa_m: jax.Array,
    *,
    sm_scale: float = 1.0,
    gather_and_attention_chunk_size: int = 128,
    attention_kernel_batch_size: int | None = None,
    union_cap: int = 4096,
    union_gather: bool = True,
    gathered_nope: jax.Array | None = None,
    gathered_rope: jax.Array | None = None,
) -> jax.Array:
    """``sparse_ragged_paged_attention`` with per-chunk SparseCore Union Gather."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.deepseek_v4.core_attention import sparse_mla

    _, actual_num_q_heads, actual_head_dim = q.shape
    q = sparse_mla.prepare_q_inputs(q)
    head_dim = q.shape[-1]
    attention_sinks = jnp.pad(
        attention_sinks,
        (0, q.shape[1] - actual_num_q_heads),
        constant_values=jnp.finfo(attention_sinks.dtype).min,
    )
    swa_accumution, swa_l, swa_m = sparse_mla.prepare_swa_inputs(
        swa_accumution, swa_l, swa_m
    )

    total_num_pages, page_size, _ = _paged_pool_shape(cache_kv_nope)
    _, num_q_heads, _ = q.shape
    topk = topk_indices.shape[-1]
    if gathered_nope is not None:
        assert gathered_rope is not None
        assert gathered_nope.shape == (q.shape[0], topk, ROW_WORDS)
        assert gathered_rope.shape == (q.shape[0], topk // 4, ROW_WORDS)
        # These buffers already cover the stage. Keep them resident in one
        # attention pipeline instead of materializing per-chunk HBM slices.
        gather_and_attention_chunk_size = q.shape[0]
    if attention_kernel_batch_size is None:
        attention_kernel_batch_size = 16
        if (
            gathered_nope is not None
            and q.shape[0] >= 128
            and jax.default_backend() == "tpu"
        ):
            from vllm_torchtpu.kernels.deepseek_v4.core_attention import (
                tuned_block_sizes as sparse_tuning,
            )

            attention_kernel_batch_size = sparse_tuning.prefill_batch_size(
                str(pltpu.get_tpu_info().chip_version), q.shape[1], head_dim, topk
            )
    kv_lens = jnp.sum(topk_indices != -1, axis=-1)
    union_cap_aligned = max(topk, sparse_mla.align_to(union_cap, topk))
    max_num_seqs = cu_q_lens.shape[0] - 1
    tokens_per_seq = cu_q_lens[1:] - cu_q_lens[:-1]
    seq_ids_segment = jnp.repeat(
        jnp.arange(max_num_seqs, dtype=jnp.int32),
        tokens_per_seq,
        total_repeat_length=q.shape[0],
    )

    num_chunks = sparse_mla.cdiv(q.shape[0], gather_and_attention_chunk_size)
    if union_gather and gathered_nope is None:
        union_phys, inv_idx_all, num_unique_all = compute_chunk_union_gather_indices(
            topk_indices,
            page_indices,
            cu_q_lens,
            page_size=page_size,
            total_num_pages=total_num_pages,
            chunk_size=gather_and_attention_chunk_size,
            union_cap=union_cap_aligned,
        )
    else:
        union_phys = jnp.zeros((num_chunks, topk), dtype=jnp.int32)
        inv_idx_all = jnp.zeros(
            (num_chunks, gather_and_attention_chunk_size, topk), dtype=jnp.int32
        )
        num_unique_all = jnp.zeros((num_chunks,), dtype=jnp.int32)

    valid_total = cu_q_lens[distribution[2]]

    def gather_chunk(i, cache_kv_nope, cache_kv_rope):
        start_pos = i * gather_and_attention_chunk_size
        end_pos = min(start_pos + gather_and_attention_chunk_size, q.shape[0])
        chunk_size = end_pos - start_pos
        kbs = sparse_mla._largest_divisor(chunk_size, attention_kernel_batch_size)
        batch_end = start_pos + (
            sparse_mla.cdiv(
                jnp.maximum(0, jnp.minimum(valid_total, end_pos) - start_pos),
                kbs,
            )
            * kbs
        )
        if gathered_nope is not None:
            assert gathered_rope is not None
            return (
                gathered_nope[start_pos:end_pos],
                gathered_rope[start_pos:end_pos],
                start_pos,
                batch_end,
                kbs,
                chunk_size,
            )
        gathered_nope_buffer, gathered_rope_buffer = (
            gather_sparse_kv_chunk_with_union_fallback(
                cache_kv_nope,
                cache_kv_rope,
                union_phys[i],
                inv_idx_all[i, :chunk_size],
                num_unique_all[i],
                topk_indices[start_pos:end_pos],
                seq_ids_segment[start_pos:end_pos],
                page_indices,
                max_num_seqs=max_num_seqs,
                start_pos=start_pos,
                batch_end=batch_end,
                topk=topk,
                use_union=union_gather,
            )
        )

        return (
            gathered_nope_buffer,
            gathered_rope_buffer,
            start_pos,
            batch_end,
            kbs,
            chunk_size,
        )

    if num_chunks == 0:
        return sparse_mla.prepare_outputs(q, actual_num_q_heads, actual_head_dim)
    current = gather_chunk(0, cache_kv_nope, cache_kv_rope)
    for i in range(num_chunks):
        pending = current
        if i + 1 < num_chunks:
            # Keep the pending chunk independent of the current attention.
            # The previous attention's q gates prefetch to one chunk ahead,
            # bounding gathered HBM buffers without a large while-loop carry.
            q, cache_kv_nope, cache_kv_rope = jax.lax.optimization_barrier(
                (q, cache_kv_nope, cache_kv_rope)
            )
            pending = gather_chunk(i + 1, cache_kv_nope, cache_kv_rope)
        (
            gathered_nope_buffer,
            gathered_rope_buffer,
            start_pos,
            batch_end,
            kbs,
            chunk_size,
        ) = current
        end_seq_idx = jnp.maximum(start_pos, batch_end)
        grid = (sparse_mla.cdiv(end_seq_idx - start_pos, kbs),)
        in_specs = [
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ]
        out_specs = pl.BlockSpec(memory_space=pltpu.HBM)
        _, topk_rows, lanes = gathered_nope_buffer.shape
        _, rope_rows, _ = gathered_rope_buffer.shape
        num_l_heads = sparse_mla.align_to(num_q_heads, 128)
        scratch_shapes = [
            pltpu.VMEM((2, kbs, topk_rows * 4, lanes), jnp.uint8),
            pltpu.VMEM((2, kbs, rope_rows * 2, lanes), jnp.bfloat16),
            pltpu.VMEM((2, kbs, num_q_heads, head_dim), q.dtype),
            pltpu.VMEM((2, kbs, num_q_heads, head_dim), q.dtype),
            pltpu.VMEM((2, kbs, num_l_heads), jnp.float32),
            pltpu.VMEM((2, kbs, num_l_heads), jnp.float32),
            pltpu.VMEM((2, kbs, num_q_heads, head_dim), q.dtype),
            pltpu.SemaphoreType.DMA((7, 2)),
        ]
        scalar_prefetches = (
            kv_lens,
            jnp.array([start_pos, end_seq_idx, valid_total], jnp.int32),
            jnp.zeros((2,), jnp.int32),
            jnp.full((2,), -1, jnp.int32),
        )
        scope_name = f"SparseMLA-p_{topk_rows}-bz_{kbs}-gcz_{chunk_size}"
        kernel = jax.named_scope(scope_name)(
            pl.pallas_call(
                functools.partial(
                    sparse_mla._attention_kernel,
                    sm_scale=sm_scale,
                    batch_size=kbs,
                ),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=len(scalar_prefetches),
                    in_specs=in_specs,
                    out_specs=out_specs,
                    grid=grid,
                    scratch_shapes=scratch_shapes,
                ),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("arbitrary",),
                    vmem_limit_bytes=sparse_mla.DEFAULT_VMEM_LIMIT_BYTES,
                    disable_bounds_checks=True,
                ),
                out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
                input_output_aliases={5: 0},
                name=scope_name,
            )
        )
        q = kernel(
            *scalar_prefetches,
            attention_sinks,
            q,
            gathered_nope_buffer,
            gathered_rope_buffer,
            swa_accumution,
            swa_l,
            swa_m,
        )
        current = pending
    return sparse_mla.prepare_outputs(q, actual_num_q_heads, actual_head_dim)


def sparse_fp8_attention_jax(
    q: jax.Array,
    swa_pool: jax.Array | tuple[jax.Array, jax.Array],
    main_pool: jax.Array | tuple[jax.Array, jax.Array],
    rope_pool: jax.Array | tuple[jax.Array, jax.Array],
    swa_records: jax.Array,
    main_records: jax.Array,
    pages: jax.Array,
    query_start_loc: jax.Array,
    sink: jax.Array,
    scale: float,
    page_size: int,
    max_requests: int | None = None,
    precompacted: bool = False,
    union_gather: bool = False,
    union_cap: int = 4096,
    gathered_nope: jax.Array | None = None,
    gathered_rope: jax.Array | None = None,
) -> jax.Array:
    """BF16 SWA partial merged into V4's sparse MLA over FP8 NoPE + RoPE rows.

    ``main_records`` index rows through ``pages`` ``[requests, pages_per_seq]``:
    row ``r`` of request ``s`` is raw row ``pages[s, r // page_size] *
    page_size + r % page_size``; the RoPE companion holds raw row i's 32 words
    at i. With one request, one page of every row and ``query_start_loc``
    ``[0, tokens]``, records are raw rows.

    ``max_requests`` bounds requests per batch (see `_attend_window`).
    ``precompacted`` skips ``compact_records`` when ``main_records`` already has
    valid indices sorted ascending at the prefix and ``-1`` at the tail.
    """
    from vllm_torchtpu.kernels.deepseek_v4.core_attention.sparse_mla import (
        sparse_ragged_paged_attention,
    )
    from vllm_torchtpu.kernels.mla import tuned_window_attention, window_attention

    if (
        max_requests == 1
        and jax.default_backend() == "tpu"
        and window_attention.supports_window_attention(q, swa_records)
    ):
        union, mask, uncovered = _window_union(swa_records, 64)
        kv = gather_cache_pieces(swa_pool, union.reshape(-1), q.shape[-1])
        kv = kv.reshape(kv.shape[0], *union.shape, kv.shape[-1])
        key = (q.shape[1], q.shape[2], swa_records.shape[1])
        state = window_attention.attend_windows(
            q,
            kv,
            mask,
            scale,
            query_block=tuned_window_attention.QUERY_BLOCKS[key],
        )
        # Match the branch-free window path's failure for noncontiguous records.
        state = tuple(
            jnp.where(uncovered.reshape(-1, *([1] * (value.ndim - 1))), jnp.nan, value)
            for value in state
        )
    else:
        state = _attend_window(
            q,
            swa_pool,
            swa_records,
            _attention_state(q),
            scale,
            max_requests=max_requests,
        )
    # The SWA conditional holds SparseCore gathers, and XLA completes every
    # in-flight `sc_csa_gather` before such a conditional: the TensorCore then
    # idles through the whole gather chain (sc_gather_diag.md). Launch the CSA
    # gathers after SWA instead so they pipeline with the SparseMLA chunks.
    # Without the conditional (``max_requests``) this ordering still measured
    # faster at 100k (16.0K vs 15.7K tok/s).
    (maximum, denominator, numerator), main_pool, rope_pool = (
        jax.lax.optimization_barrier((state, main_pool, rope_pool))
    )
    _cache_num_rows(main_pool)
    compacted = (
        _pad_compacted_records(main_records)
        if precompacted
        else compact_records(main_records)
    )
    nope_3d = _reshape_paged_pool(main_pool, page_size)
    rope_3d = _reshape_paged_pool(rope_pool, page_size // 4)
    if gathered_nope is not None or union_gather or isinstance(main_pool, tuple):
        return sparse_ragged_paged_attention_union_gather(
            q,
            nope_3d,
            rope_3d,
            compacted,
            pages.reshape(-1),
            query_start_loc,
            jnp.array([0, 0, pages.shape[0]], jnp.int32),
            sink.astype(jnp.float32),
            numerator.astype(q.dtype),
            denominator,
            maximum,
            sm_scale=scale,
            union_cap=union_cap,
            union_gather=union_gather,
            gathered_nope=gathered_nope,
            gathered_rope=gathered_rope,
        )
    assert not isinstance(nope_3d, tuple) and not isinstance(rope_3d, tuple)
    return sparse_ragged_paged_attention(
        q,
        nope_3d,
        rope_3d,
        compacted,
        pages.reshape(-1),
        query_start_loc,
        jnp.array([0, 0, pages.shape[0]], jnp.int32),
        sink.astype(jnp.float32),
        numerator.astype(q.dtype),
        denominator,
        maximum,
        sm_scale=scale,
    )


def slice_token_parallel_attention_inputs(
    swa_records: jax.Array,
    main_records: jax.Array,
    query_start_loc: jax.Array,
    tp_rank: jax.Array,
    tp_size: int,
    tokens: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Slice ``[tokens, ...]`` records and ``query_start_loc`` to this rank's shard."""
    tokens_local = tokens // tp_size
    rank_idx = tp_rank.reshape(())
    start = rank_idx * tokens_local
    swa_local = jax.lax.dynamic_slice_in_dim(swa_records, start, tokens_local, axis=0)
    main_local = jax.lax.dynamic_slice_in_dim(main_records, start, tokens_local, axis=0)
    q_loc_local = jnp.clip(query_start_loc - start, 0, tokens_local)
    return swa_local, main_local, q_loc_local


def gather_bf16_kv_jax(
    pool: jax.Array,
    pool_hi: jax.Array | None,
    records: jax.Array,
    dim: int,
) -> jax.Array:
    """Materialize the split BF16 pieces consumed by `_attend`."""
    records = jnp.pad(
        records, ((0, 0), (0, -records.shape[1] % 64)), constant_values=-1
    )
    return gather_cache_pieces(
        pool if pool_hi is None else (pool, pool_hi), records.reshape(-1), dim
    )


def gather_fp8_kv_jax(
    pool: jax.Array,
    pool_hi: jax.Array | None,
    rope: jax.Array,
    rope_hi: jax.Array | None,
    records: jax.Array,
    pages: jax.Array,
    starts: jax.Array,
    rank: jax.Array,
    page_size: int,
    shards: int,
    precompacted: bool,
) -> tuple[jax.Array, jax.Array]:
    """Gather once per source/index group, retaining packed FP8 and BF16 RoPE."""
    if shards > 1:
        tokens = records.shape[0] // shards
        start = rank.reshape(()) * tokens
        records = jax.lax.dynamic_slice_in_dim(records, start, tokens, axis=0)
        starts = jnp.clip(starts - start, 0, tokens)
        records = jnp.pad(records, ((0, -tokens % 8), (0, 0)), constant_values=-1)
    compacted = (
        _pad_compacted_records(records) if precompacted else compact_records(records)
    )
    nope_pool = _reshape_paged_pool(
        pool if pool_hi is None else (pool, pool_hi), page_size
    )
    rope_pool = _reshape_paged_pool(
        rope if rope_hi is None else (rope, rope_hi), page_size // 4
    )
    seq_ids = jnp.repeat(
        jnp.arange(pages.shape[0], dtype=jnp.int32),
        starts[1:] - starts[:-1],
        total_repeat_length=compacted.shape[0],
    )
    return gather_sparse_kv_chunk(
        nope_pool,
        rope_pool,
        compacted,
        seq_ids,
        pages.reshape(-1),
        max_num_seqs=pages.shape[0],
        start_pos=0,
        batch_end=starts[-1],
    )


def _fp8_query_shards(
    tokens: int, tp_size: int, token_parallel: bool, local_tokens: int | None = None
) -> int:
    """Use the same query partition for the gather and attention bridges."""
    if local_tokens is not None:
        use_shards = (
            tp_size > 1
            and tokens >= 128 * tp_size
            and tokens % (128 * tp_size) == 0
            and local_tokens * tp_size == tokens
        )
    else:
        use_shards = (
            token_parallel
            and tp_size > 1
            and tokens >= 6 * tp_size
            and tokens % tp_size == 0
        )
    return tp_size if use_shards else 1


def initialize_paged_attention():
    global _attention_op, _attention_2bank_op
    global _fp8_attention_op, _fp8_attention_2bank_op
    global _fp8_attention_qo_op, _fp8_attention_qo_2bank_op
    global _gather_bf16_op, _gather_fp8_op, _attention_tp_size
    if _attention_op is not None:
        return
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas
    from vllm.distributed import get_tp_group

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_pallas_op_mesh,
        get_vllm_model_wrapper_context,
    )

    # jax_op's type stub omits PartitionSpec and None for static arguments.
    _attention_op = pallas.jax_op(
        "pallas::dsv41v2_paged_bf16_attention",
        paged_attention_jax,
        mesh=get_pallas_op_mesh(),
        input_partition_specs=cast(
            tuple[tuple[str, ...], ...], cast(Any, (P(),) * 6 + (None, P()))
        ),
    )
    _attention_op.register_fake(lambda q, *args: q.new_empty(q.shape))

    def paged_attention_2bank(
        q: jax.Array,
        swa_lo: jax.Array,
        swa_hi: jax.Array,
        main_lo: jax.Array,
        main_hi: jax.Array,
        swa_records: jax.Array,
        main_records: jax.Array,
        sink: jax.Array,
        scale: float,
        gathered: jax.Array | None = None,
    ) -> jax.Array:
        return paged_attention_jax(
            q,
            cast(Any, (swa_lo, swa_hi)),
            cast(Any, (main_lo, main_hi)),
            swa_records,
            main_records,
            sink,
            scale,
            gathered,
        )

    _attention_2bank_op = pallas.jax_op(
        "pallas::dsv41v2_paged_bf16_attention_2bank",
        paged_attention_2bank,
        mesh=get_pallas_op_mesh(),
        input_partition_specs=cast(
            tuple[tuple[str, ...], ...], cast(Any, (P(),) * 8 + (None, P()))
        ),
    )
    _attention_2bank_op.register_fake(lambda q, *args: q.new_empty(q.shape))
    mesh = get_vllm_model_wrapper_context().mesh
    try:
        tp_group = get_tp_group().device_group
        tp_size = tp_group.size() if tp_group is not None else 1
    except Exception:
        tp_group = None
        tp_size = 1
    if tp_size > 1 and tp_group is not None:
        tp_mesh = pallas.get_pallas_mesh(axis_names=("tp",), process_group=tp_group)
        tp_op_mesh = tp_mesh
    else:
        tp_mesh = mesh
        tp_op_mesh = get_pallas_op_mesh()
    _attention_tp_size = tp_size
    _gather_bf16_op = pallas.jax_op(
        "pallas::dsv41v2_gather_bf16_kv",
        gather_bf16_kv_jax,
        mesh=get_pallas_op_mesh(),
        input_partition_specs=cast(
            tuple[tuple[str, ...], ...], cast(Any, (P(),) * 3 + (None,))
        ),
    )
    _gather_bf16_op.register_fake(
        lambda pool, pool_hi, records, dim: pool.new_empty(
            (
                max(2, dim // 128),
                records.shape[0] * (-(-records.shape[1] // 64) * 64),
                min(128, dim // 2),
            ),
            dtype=torch.bfloat16,
        )
    )

    def gather_fp8_bridge(
        pool: jax.Array,
        pool_hi: jax.Array | None,
        rope: jax.Array,
        rope_hi: jax.Array | None,
        records: jax.Array,
        pages: jax.Array,
        starts: jax.Array,
        rank: jax.Array,
        page_size: int,
        shards: int,
        precompacted: bool,
    ) -> tuple[jax.Array, jax.Array]:
        return jax.shard_map(
            functools.partial(
                gather_fp8_kv_jax,
                page_size=page_size,
                shards=shards,
                precompacted=precompacted,
            ),
            mesh=tp_mesh,
            in_specs=(P(),) * 8,
            out_specs=(P(), P()),
            check_vma=False,
        )(pool, pool_hi, rope, rope_hi, records, pages, starts, rank)

    _gather_fp8_op = pallas.jax_op(
        "pallas::dsv41v2_gather_fp8_kv",
        gather_fp8_bridge,
        mesh=tp_op_mesh,
        input_partition_specs=cast(
            tuple[tuple[str, ...], ...], cast(Any, (P(),) * 8 + (None,) * 3)
        ),
    )

    def fake_gather_fp8(
        pool,
        pool_hi,
        rope,
        rope_hi,
        records,
        pages,
        starts,
        rank,
        page_size,
        shards,
        precompacted,
    ):
        tokens = (
            records.shape[0]
            if shards == 1
            else -(-(records.shape[0] // shards) // 8) * 8
        )
        topk = max(128, -(-records.shape[1] // 128) * 128)
        return pool.new_empty((tokens, topk, ROW_WORDS)), rope.new_empty(
            (tokens, topk // 4, ROW_WORDS)
        )

    _gather_fp8_op.register_fake(fake_gather_fp8)
    in_specs = (P(),) * 13
    out_spec = P()

    def fp8_attention(
        q: jax.Array,
        swa_pool: jax.Array,
        main_pool: jax.Array,
        rope_pool: jax.Array,
        swa_records: jax.Array,
        main_records: jax.Array,
        pages: jax.Array,
        query_start_loc: jax.Array,
        sink: jax.Array,
        full_sink: jax.Array,
        tp_rank: jax.Array,
        scale: float,
        page_size: int,
        max_requests: int | None,
        precompacted: bool,
        token_parallel: bool,
        union_gather: bool,
        union_cap: int,
        gathered_nope: jax.Array | None = None,
        gathered_rope: jax.Array | None = None,
    ) -> jax.Array:
        tokens = q.shape[0]
        use_token_parallel = _fp8_query_shards(tokens, tp_size, token_parallel) > 1
        if use_token_parallel:
            tokens_local = tokens // tp_size
            padded_local = -(-tokens_local // 8) * 8
            pad_rows = padded_local - tokens_local

            def _token_parallel_body(
                q_l, s_p, m_p, r_p, s_r, m_r, pgs, q_loc, _snk, f_snk, r_id, g_n, g_r
            ):
                q_tok = jax.lax.all_to_all(
                    q_l, axis_name="tp", split_axis=0, concat_axis=1, tiled=True
                )
                s_r_tok, m_r_tok, q_loc_tok = slice_token_parallel_attention_inputs(
                    s_r, m_r, q_loc, r_id, tp_size, tokens
                )
                if pad_rows:
                    q_tok = jnp.pad(q_tok, ((0, pad_rows), (0, 0), (0, 0)))
                    s_r_tok = jnp.pad(
                        s_r_tok, ((0, pad_rows), (0, 0)), constant_values=-1
                    )
                    m_r_tok = jnp.pad(
                        m_r_tok, ((0, pad_rows), (0, 0)), constant_values=-1
                    )
                out_tok = sparse_fp8_attention_jax(
                    q_tok,
                    s_p,
                    m_p,
                    r_p,
                    s_r_tok,
                    m_r_tok,
                    pgs,
                    q_loc_tok,
                    f_snk,
                    scale=scale,
                    page_size=page_size,
                    max_requests=max_requests,
                    precompacted=precompacted,
                    union_gather=union_gather,
                    union_cap=union_cap,
                    gathered_nope=g_n,
                    gathered_rope=g_r,
                )
                if pad_rows:
                    out_tok = out_tok[:-pad_rows]
                return jax.lax.all_to_all(
                    out_tok, axis_name="tp", split_axis=1, concat_axis=0, tiled=True
                )

            return jax.shard_map(
                _token_parallel_body,
                mesh=tp_mesh,
                in_specs=in_specs,
                out_specs=out_spec,
                check_vma=False,
            )(
                q,
                swa_pool,
                main_pool,
                rope_pool,
                swa_records,
                main_records,
                pages,
                query_start_loc,
                sink,
                full_sink,
                tp_rank,
                gathered_nope,
                gathered_rope,
            )

        # Mosaic calls require explicit manual partitioning on a multi-device
        # mesh; each device attends over its own replicated cache.
        return jax.shard_map(
            lambda q_l,
            s_p,
            m_p,
            r_p,
            s_r,
            m_r,
            pgs,
            q_loc,
            snk,
            _f_snk,
            _r_id,
            g_n,
            g_r: (
                sparse_fp8_attention_jax(
                    q_l,
                    s_p,
                    m_p,
                    r_p,
                    s_r,
                    m_r,
                    pgs,
                    q_loc,
                    snk,
                    scale=scale,
                    page_size=page_size,
                    max_requests=max_requests,
                    precompacted=precompacted,
                    union_gather=union_gather,
                    union_cap=union_cap,
                    gathered_nope=g_n,
                    gathered_rope=g_r,
                )
            ),
            mesh=tp_mesh,
            in_specs=in_specs,
            out_specs=out_spec,
            check_vma=False,
        )(
            q,
            swa_pool,
            main_pool,
            rope_pool,
            swa_records,
            main_records,
            pages,
            query_start_loc,
            sink,
            full_sink,
            tp_rank,
            gathered_nope,
            gathered_rope,
        )

    _fp8_attention_op = pallas.jax_op(
        "pallas::dsv41v2_sparse_fp8_attention",
        fp8_attention,
        mesh=tp_op_mesh,
        input_partition_specs=cast(
            tuple[tuple[str, ...], ...],
            cast(Any, (P(),) * 11 + (None,) * 7 + (P(),) * 2),
        ),
    )
    _fp8_attention_op.register_fake(lambda q, *args: q.new_empty(q.shape))

    def fp8_attention_2bank(
        q: jax.Array,
        swa_lo: jax.Array,
        swa_hi: jax.Array,
        main_lo: jax.Array,
        main_hi: jax.Array,
        rope_lo: jax.Array,
        rope_hi: jax.Array,
        swa_records: jax.Array,
        main_records: jax.Array,
        pages: jax.Array,
        query_start_loc: jax.Array,
        sink: jax.Array,
        full_sink: jax.Array,
        tp_rank: jax.Array,
        scale: float,
        page_size: int,
        max_requests: int | None,
        precompacted: bool,
        token_parallel: bool,
        union_gather: bool,
        union_cap: int,
        gathered_nope: jax.Array | None = None,
        gathered_rope: jax.Array | None = None,
    ) -> jax.Array:
        return fp8_attention(
            q,
            cast(Any, (swa_lo, swa_hi)),
            cast(Any, (main_lo, main_hi)),
            cast(Any, (rope_lo, rope_hi)),
            swa_records,
            main_records,
            pages,
            query_start_loc,
            sink,
            full_sink,
            tp_rank,
            scale,
            page_size,
            max_requests,
            precompacted,
            token_parallel,
            union_gather,
            union_cap,
            gathered_nope,
            gathered_rope,
        )

    _fp8_attention_2bank_op = pallas.jax_op(
        "pallas::dsv41v2_sparse_fp8_attention_2bank",
        fp8_attention_2bank,
        mesh=tp_op_mesh,
        input_partition_specs=cast(
            tuple[tuple[str, ...], ...],
            cast(Any, (P(),) * 14 + (None,) * 7 + (P(),) * 2),
        ),
    )
    _fp8_attention_2bank_op.register_fake(lambda q, *args: q.new_empty(q.shape))

    def fp8_attention_qo(
        qr: jax.Array,
        qr_local: jax.Array,
        wq_b: jax.Array,
        wq_b_scale: jax.Array,
        wq_b_full: jax.Array,
        wq_b_full_scale: jax.Array,
        positions: jax.Array,
        cos_sin_cache: jax.Array,
        swa_pool: jax.Array,
        main_pool: jax.Array,
        rope_pool: jax.Array,
        swa_records: jax.Array,
        main_records: jax.Array,
        pages: jax.Array,
        query_start_loc: jax.Array,
        sink: jax.Array,
        full_sink: jax.Array,
        tp_rank: jax.Array,
        wo_a: jax.Array,
        wo_a_scale: jax.Array,
        wo_a_full: jax.Array,
        wo_a_full_scale: jax.Array,
        valid: jax.Array,
        scale: float,
        page_size: int,
        max_requests: int | None,
        precompacted: bool,
        head_dim: int,
        gathered_nope: jax.Array | None = None,
        gathered_rope: jax.Array | None = None,
    ) -> jax.Array:
        """q projection + RoPE, sparse FP8 attention and inverse-RoPE ``wo_a``.

        Returns this rank's ``wo_b`` input ``[T, o_lora_rank]``. With the
        token-parallel shape condition of `fp8_attention`, each rank projects q
        for its own SP token shard ``qr_local`` with the all-gathered
        ``wq_b_full`` (no q ``all_to_all``) and applies ``wo_a`` for every
        ``o_group`` on the shard, so the return ``all_to_all`` carries
        ``[T/tp, groups, o_lora_rank]`` instead of ``[T/tp, heads, head_dim]``.
        Otherwise it runs the head-sharded path with the
        local ``wq_b``/``wo_a``. Per token, head and group both paths are the
        same quantized matmul, RoPE, attention and ``wo_a`` math as the unfused
        `_prepare_tpu` + ``_o_proj`` sequence.
        """
        from vllm_torchtpu.layers.adapter.linear_common import _quantized_matmul_jax
        from vllm_torchtpu.models.vllm.deepseek_v4.attention import (
            _o_proj_jax,
            _rope_jax,
        )

        tokens = positions.shape[0]
        tokens_local = qr_local.shape[0]
        attn_kwargs = dict(
            scale=scale,
            page_size=page_size,
            max_requests=max_requests,
            precompacted=precompacted,
        )
        if _fp8_query_shards(tokens, tp_size, True, tokens_local) > 1:

            def _local_body(
                q_in,
                wq,
                wqs,
                pos,
                cs,
                s_p,
                m_p,
                r_p,
                s_r,
                m_r,
                pgs,
                q_loc,
                f_snk,
                r_id,
                w,
                ws,
                ok,
                g_n,
                g_r,
            ):
                s_r_tok, m_r_tok, q_loc_tok = slice_token_parallel_attention_inputs(
                    s_r, m_r, q_loc, r_id, tp_size, tokens
                )
                start = r_id.reshape(()) * tokens_local
                pos_tok = jax.lax.dynamic_slice_in_dim(pos, start, tokens_local, axis=0)
                ok_tok = jax.lax.dynamic_slice_in_dim(ok, start, tokens_local, axis=0)
                q_tok = _quantized_matmul_jax(q_in, wq, wqs)
                q_tok = _rope_jax(
                    q_tok.reshape(tokens_local, -1, head_dim), pos_tok, cs
                )
                out_tok = sparse_fp8_attention_jax(
                    q_tok,
                    s_p,
                    m_p,
                    r_p,
                    s_r_tok,
                    m_r_tok,
                    pgs,
                    q_loc_tok,
                    f_snk,
                    **attn_kwargs,
                    gathered_nope=g_n,
                    gathered_rope=g_r,
                )
                out_tok = jnp.where(
                    ok_tok[:, None, None] != 0, out_tok, jnp.zeros((), out_tok.dtype)
                )
                projected = _o_proj_jax(out_tok, pos_tok, cs, w, ws, head_dim=head_dim)
                projected = projected.reshape(tokens_local, tp_size, -1)
                gathered = jax.lax.all_to_all(
                    projected, axis_name="tp", split_axis=1, concat_axis=0, tiled=True
                )
                return gathered.reshape(tokens, -1)

            return jax.shard_map(
                _local_body,
                mesh=tp_mesh,
                in_specs=(P(),) * 19,
                out_specs=out_spec,
                check_vma=False,
            )(
                qr_local,
                wq_b_full,
                wq_b_full_scale,
                positions,
                cos_sin_cache,
                swa_pool,
                main_pool,
                rope_pool,
                swa_records,
                main_records,
                pages,
                query_start_loc,
                full_sink,
                tp_rank,
                wo_a_full,
                wo_a_full_scale,
                valid,
                gathered_nope,
                gathered_rope,
            )

        def _sharded_body(
            q_in,
            wq,
            wqs,
            pos,
            cs,
            s_p,
            m_p,
            r_p,
            s_r,
            m_r,
            pgs,
            q_loc,
            snk,
            w,
            ws,
            ok,
            g_n,
            g_r,
        ):
            if tp_size > 1 and q_in.shape[0] < tokens:
                q_in = jax.lax.all_gather(q_in, axis_name="tp", axis=0, tiled=True)[
                    :tokens
                ]
            q = _quantized_matmul_jax(q_in, wq, wqs)
            q = _rope_jax(q.reshape(tokens, -1, head_dim), pos, cs)
            out = sparse_fp8_attention_jax(
                q,
                s_p,
                m_p,
                r_p,
                s_r,
                m_r,
                pgs,
                q_loc,
                snk,
                **attn_kwargs,
                gathered_nope=g_n,
                gathered_rope=g_r,
            )
            out = jnp.where(ok[:, None, None] != 0, out, jnp.zeros((), out.dtype))
            return _o_proj_jax(out, pos, cs, w, ws, head_dim=head_dim)

        return jax.shard_map(
            _sharded_body,
            mesh=tp_mesh,
            in_specs=(P(),) * 18,
            out_specs=out_spec,
            check_vma=False,
        )(
            qr,
            wq_b,
            wq_b_scale,
            positions,
            cos_sin_cache,
            swa_pool,
            main_pool,
            rope_pool,
            swa_records,
            main_records,
            pages,
            query_start_loc,
            sink,
            wo_a,
            wo_a_scale,
            valid,
            gathered_nope,
            gathered_rope,
        )

    _fp8_attention_qo_op = pallas.jax_op(
        "pallas::dsv41v2_sparse_fp8_attention_qo",
        fp8_attention_qo,
        mesh=tp_op_mesh,
        input_partition_specs=cast(
            tuple[tuple[str, ...], ...],
            cast(Any, (P(),) * 23 + (None,) * 5 + (P(),) * 2),
        ),
    )
    # [T, o_lora_rank]: positions is args[5], wo_a is args[17].
    _fp8_attention_qo_op.register_fake(
        lambda qr, *args: qr.new_empty((args[5].shape[0], args[17].shape[-1]))
    )

    def fp8_attention_qo_2bank(
        qr: jax.Array,
        qr_local: jax.Array,
        wq_b: jax.Array,
        wq_b_scale: jax.Array,
        wq_b_full: jax.Array,
        wq_b_full_scale: jax.Array,
        positions: jax.Array,
        cos_sin_cache: jax.Array,
        swa_lo: jax.Array,
        swa_hi: jax.Array,
        main_lo: jax.Array,
        main_hi: jax.Array,
        rope_lo: jax.Array,
        rope_hi: jax.Array,
        swa_records: jax.Array,
        main_records: jax.Array,
        pages: jax.Array,
        query_start_loc: jax.Array,
        sink: jax.Array,
        full_sink: jax.Array,
        tp_rank: jax.Array,
        wo_a: jax.Array,
        wo_a_scale: jax.Array,
        wo_a_full: jax.Array,
        wo_a_full_scale: jax.Array,
        valid: jax.Array,
        scale: float,
        page_size: int,
        max_requests: int | None,
        precompacted: bool,
        head_dim: int,
        gathered_nope: jax.Array | None = None,
        gathered_rope: jax.Array | None = None,
    ) -> jax.Array:
        return fp8_attention_qo(
            qr,
            qr_local,
            wq_b,
            wq_b_scale,
            wq_b_full,
            wq_b_full_scale,
            positions,
            cos_sin_cache,
            cast(Any, (swa_lo, swa_hi)),
            cast(Any, (main_lo, main_hi)),
            cast(Any, (rope_lo, rope_hi)),
            swa_records,
            main_records,
            pages,
            query_start_loc,
            sink,
            full_sink,
            tp_rank,
            wo_a,
            wo_a_scale,
            wo_a_full,
            wo_a_full_scale,
            valid,
            scale,
            page_size,
            max_requests,
            precompacted,
            head_dim,
            gathered_nope,
            gathered_rope,
        )

    _fp8_attention_qo_2bank_op = pallas.jax_op(
        "pallas::dsv41v2_sparse_fp8_attention_qo_2bank",
        fp8_attention_qo_2bank,
        mesh=tp_op_mesh,
        input_partition_specs=cast(
            tuple[tuple[str, ...], ...],
            cast(Any, (P(),) * 26 + (None,) * 5 + (P(),) * 2),
        ),
    )
    # [T, o_lora_rank]: positions is args[5], wo_a is args[20].
    _fp8_attention_qo_2bank_op.register_fake(
        lambda qr, *args: qr.new_empty((args[5].shape[0], args[20].shape[-1]))
    )


def _has_hi_bank(cache) -> bool:
    return "raw_hi" in getattr(cache, "__dict__", {}) and cache.raw_hi is not None


def _resolve_fp8_pages(main, main_records, tokens):
    page_size = main.shape[1]
    if not _has_hi_bank(main):
        page_size = main.raw.shape[0]
        pages = main_records.new_zeros((1, 1))
    else:
        page_size = math.gcd(main.raw.shape[0], main.raw_hi.shape[0])
        total_pages = main.total_rows // page_size
        pages = torch.arange(
            total_pages, device=main_records.device, dtype=torch.int32
        ).unsqueeze(0)
    starts = main_records.new_tensor([0, tokens])
    return page_size, pages, starts


def _gather_fp8_for_stage(
    main, records, pages, starts, rank, page_size, shards, precompacted
):
    cache = _active_gathered_kv
    if cache is None:
        return None, None
    layout = ("fp8", page_size, shards, precompacted)
    values = cache.get(main, records, layout)
    if values is not None:
        return values
    assert _gather_fp8_op is not None
    values = _gather_fp8_op(
        main.raw,
        main.raw_hi,
        main.rope,
        main.rope_hi,
        records.int(),
        pages.int(),
        starts.int(),
        rank.reshape(1).int(),
        page_size,
        shards,
        precompacted,
    )
    return cache.put(main, records, layout, values)


def sparse_fp8_attention_qo(
    qr,
    qr_local,
    wq_b,
    wq_b_scale,
    wq_b_full,
    wq_b_full_scale,
    positions,
    cos_sin_cache,
    swa,
    main,
    swa_records,
    main_records,
    sink,
    full_sink,
    tp_rank,
    wo_a,
    wo_a_scale,
    wo_a_full,
    wo_a_full_scale,
    valid,
    scale,
    pages,
    starts,
    max_requests,
    precompacted,
    head_dim,
):
    """`fp8_attention_qo` entry: ``qr`` -> this rank's ``wo_b`` input ``[T, R]``."""
    assert _fp8_attention_qo_op is not None and main.rope is not None
    page_size = main.shape[1]
    if pages is None:
        page_size, pages, starts = _resolve_fp8_pages(
            main, main_records, positions.shape[0]
        )
    tokens = positions.shape[0]
    shards = _fp8_query_shards(tokens, _attention_tp_size, True, qr_local.shape[0])
    gathered = _gather_fp8_for_stage(
        main, main_records, pages, starts, tp_rank, page_size, shards, precompacted
    )
    tail_args = (
        swa_records.int(),
        main_records.int(),
        pages.int(),
        starts.int(),
        sink,
        full_sink,
        tp_rank.reshape(1).int(),
        wo_a,
        wo_a_scale.reshape(-1),
        wo_a_full,
        wo_a_full_scale.reshape(-1),
        valid.int(),
        scale,
        page_size,
        max_requests,
        precompacted,
        head_dim,
        *gathered,
    )
    if _has_hi_bank(main):
        assert (
            _fp8_attention_qo_2bank_op is not None
            and swa.raw_hi is not None
            and main.rope_hi is not None
        )
        return _fp8_attention_qo_2bank_op(
            qr,
            qr_local,
            wq_b,
            wq_b_scale,
            wq_b_full,
            wq_b_full_scale,
            positions,
            cos_sin_cache,
            swa.raw,
            swa.raw_hi,
            main.raw,
            main.raw_hi,
            main.rope,
            main.rope_hi,
            *tail_args,
        )
    return _fp8_attention_qo_op(
        qr,
        qr_local,
        wq_b,
        wq_b_scale,
        wq_b_full,
        wq_b_full_scale,
        positions,
        cos_sin_cache,
        swa.raw,
        main.raw,
        main.rope,
        *tail_args,
    )


def paged_attention(q, swa, main, swa_records, main_records, sink, scale):
    gathered = None
    cache = _active_gathered_kv
    if cache is not None and main_records.shape[1]:
        layout = ("bf16", q.shape[-1])
        values = cache.get(main, main_records, layout)
        if values is None:
            assert _gather_bf16_op is not None
            values = cache.put(
                main,
                main_records,
                layout,
                (
                    _gather_bf16_op(
                        main.raw, main.raw_hi, main_records.int(), q.shape[-1]
                    ),
                    None,
                ),
            )
        gathered = values[0]
    if _has_hi_bank(main):
        assert _attention_2bank_op is not None and swa.raw_hi is not None
        return _attention_2bank_op(
            q,
            swa.raw,
            swa.raw_hi,
            main.raw,
            main.raw_hi,
            swa_records.int(),
            main_records.int(),
            sink,
            scale,
            gathered,
        )
    assert _attention_op is not None
    return _attention_op(
        q,
        swa.raw,
        main.raw,
        swa_records.int(),
        main_records.int(),
        sink,
        scale,
        gathered,
    )


def sparse_fp8_attention(
    q,
    swa,
    main,
    swa_records,
    main_records,
    sink,
    scale,
    pages=None,
    starts=None,
    max_requests=None,
    precompacted=False,
    token_parallel=False,
    full_sink=None,
    tp_rank=None,
    union_gather=False,
    union_cap=4096,
):
    """``main_records`` are raw rows, or rows paged by ``pages`` (main.pages)."""
    assert _fp8_attention_op is not None and main.rope is not None
    page_size = main.shape[1]
    if pages is None or starts is None:
        page_size, pages, starts = _resolve_fp8_pages(main, main_records, q.shape[0])
    if full_sink is None:
        full_sink = sink
    if tp_rank is None:
        tp_rank = main_records.new_zeros((1,), dtype=torch.int32)
    else:
        tp_rank = tp_rank.reshape(1).int()
    tokens = q.shape[0]
    shards = _fp8_query_shards(tokens, _attention_tp_size, token_parallel)
    gathered = _gather_fp8_for_stage(
        main, main_records, pages, starts, tp_rank, page_size, shards, precompacted
    )
    tail_args = (
        swa_records.int(),
        main_records.int(),
        pages.int(),
        starts.int(),
        sink,
        full_sink,
        tp_rank,
        scale,
        page_size,
        max_requests,
        precompacted,
        token_parallel,
        union_gather,
        union_cap,
        *gathered,
    )
    if _has_hi_bank(main):
        assert (
            _fp8_attention_2bank_op is not None
            and swa.raw_hi is not None
            and main.rope_hi is not None
        )
        return _fp8_attention_2bank_op(
            q,
            swa.raw,
            swa.raw_hi,
            main.raw,
            main.raw_hi,
            main.rope,
            main.rope_hi,
            *tail_args,
        )
    return _fp8_attention_op(
        q,
        swa.raw,
        main.raw,
        main.rope,
        *tail_args,
    )


# Index-score tile: about `_INDEX_KEYS_PER_ITER` keys (all query rows) per
# loop iteration, at most `_INDEX_MAX_TILE` columns (see paged_index_scores_jax).
# 64K/8K vs 16K/2K: L24 candidate_filter 0.307 -> 0.254 ms at 3 rows, 100K
# context; parts of >= 128 rows (prefill) keep the 512 floor.
# 16K vs 8K max tile at 2..8 rows (decode B <= 64 at TP=8, 128K / 16K):
# 106K keys 0.268 -> 0.227 ms, 16K keys 0.062 -> 0.052 ms at 2 rows, and
# 128K keeps tile=16K at part=6,8 (B=48,64: candidate_filter -13.0% / -10.2%);
# 32K+ is slower.
_INDEX_KEYS_PER_ITER = 262144
_INDEX_MAX_TILE = 16384
# The packed scorer unrolls its query loop. Keep shared decode groups small;
# large prefill groups use the existing shared matmul.
_INDEX_MAX_SHARED_QUERIES = 8
# Packed word-row scoring (`index_word_scores`) streams pool rows straight into
# the kernel, so a loop iteration costs one paged gather + one kernel launch +
# one score write regardless of size. At TP1/DP8 C384 decode (288 rows, ~96K
# context) the `_INDEX_KEYS_PER_ITER` budget gives 512-key tiles: ~200 tiny
# iterations per layer, dominated by per-iteration overhead. Word-row paths
# use a 4.5x larger key budget; grouped rows size the tile by request rows.
_INDEX_WORD_KEYS_PER_ITER = 4608 * 256  # 4096-key tiles at 288 rows; prefill keeps 512
# Decode-shaped batches (few queries per request row) skip query tiling in
# `paged_indexer_jax` while their full [queries, context] FP32 score buffer
# stays within this many elements (512 MiB), keeping grouped scoring intact.
_INDEX_UNTILED_SCORE_ELEMENTS = 1 << 27


def paged_index_scores_jax(
    q: jax.Array,
    pool: jax.Array,
    records: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    requests: jax.Array | None = None,
    page_records: int | None = None,
    halfplane: bool = False,
) -> jax.Array:
    """Index scores ``[tokens, width]``.

    ``records`` is ``[tokens, width]``, or ``[requests, width]`` when
    ``requests`` maps each token to its row: every token of a request then
    shares one paged gather of the key tile instead of reading it again.
    ``page_records``: records per cache page when logical rows ``k * page ..``
    of every records row are one page's contiguous, pool-word-aligned records
    (`_request_row_slots` layout); score tiles, shared or per-token, then
    gather whole pages (see `gather_cache_page_values`).
    """
    tokens, dim = q.shape[0], q.shape[-1]
    width = records.shape[1]
    # Keep about `_INDEX_KEYS_PER_ITER` keys per iteration: small (decode)
    # batches otherwise run many latency-bound tiny gathers. Prefill batches use
    # 512: with 128 the loop is iteration-latency bound (2.3x slower at T=512
    # over 16.9K rows).
    eff_tokens = (
        tokens // 6
        if (
            tokens <= records.shape[0]
            and tokens <= 64
            and tokens >= 6
            and tokens % 6 == 0
        )
        else tokens
    )
    word_rows = (
        page_records is not None
        and dim == index_word_scores.DIM
        and q.dtype == jnp.bfloat16
    )
    keys_per_iter = _INDEX_WORD_KEYS_PER_ITER if word_rows else _INDEX_KEYS_PER_ITER
    budget = min(
        max(512, min(_INDEX_MAX_TILE, keys_per_iter // max(eff_tokens, 1))),
        -(-width // 128) * 128,
    )
    tile = 1 << (budget.bit_length() - 1)
    if (
        word_rows
        and requests is not None
        and records.shape[0] > 1
        and records.shape[0] < tokens <= _INDEX_MAX_SHARED_QUERIES * records.shape[0]
    ):
        # Grouped decode scoring reads `requests * tile` keys per iteration,
        # not `tokens * tile`: size the tile by request rows.
        grouped = min(
            _INDEX_MAX_TILE,
            _INDEX_KEYS_PER_ITER // records.shape[0],
            -(-width // 128) * 128,
        )
        if grouped > tile:
            tile = 1 << (grouped.bit_length() - 1)
    padded_width = width + (-width % tile)
    scores = jnp.full((tokens, padded_width), -jnp.inf, jnp.float32)
    q = split_halves(q)  # keys are read in the split element order
    weights = weights.astype(jnp.float32)[:, :, None]

    def head_sum(logits):
        # Upstream quantized MQA logits and the weighted head reduction are
        # FP32. BF16 cache storage is not a request to round the dot product.
        return (jnp.maximum(logits, 0) * weights).sum(1)

    pages = (
        page_records is not None
        and tile % page_records == 0
        and (page_records * dim // 2) % ROW_WORDS == 0
    )

    if requests is not None and tokens > records.shape[0]:
        # One request row at a time, each bounded by its own longest token:
        # all tokens score the row's shared key tile, and only that row's
        # tokens keep the result. Rows without tokens (padding) run no tiles,
        # so work scales with active requests, not the padded row count.
        records = jnp.pad(records, ((0, 0), (0, -width % tile)), constant_values=-1)
        row_lengths = (
            jnp.zeros((records.shape[0],), jnp.int32)
            .at[requests]
            .max(jnp.maximum(lengths, 0))
        )

        def row_step(row, result):
            mine = (requests == row)[:, None]

            def step(index, result):
                addresses = jax.lax.dynamic_slice(
                    records, (row, index * tile), (1, tile)
                ).reshape(-1)
                if pages:
                    keys = gather_cache_page_values(
                        pool,
                        addresses[::page_records],
                        page_records,
                        dim,
                        halfplane=halfplane,
                    )
                else:
                    keys = gather_cache_values(
                        pool, addresses, dim, split=True, halfplane=halfplane
                    )
                values = head_sum(
                    jnp.einsum(
                        "thd,cd->thc",
                        q,
                        keys.reshape(tile, dim),
                        preferred_element_type=jnp.float32,
                    )
                )
                current = jax.lax.dynamic_slice_in_dim(
                    result, index * tile, tile, axis=1
                )
                return jax.lax.dynamic_update_slice_in_dim(
                    result, jnp.where(mine, values, current), index * tile, axis=1
                )

            trips = jnp.minimum(
                (row_lengths[row] + tile - 1) // tile, padded_width // tile
            )
            return jax.lax.fori_loop(0, trips, step, result)

        def shared_rows(_):
            return jax.lax.fori_loop(0, records.shape[0], row_step, scores)[:, :width]

        if (
            pages
            and dim == index_word_scores.DIM
            and q.dtype == jnp.bfloat16
            and records.shape[0] > 1
            and tokens <= _INDEX_MAX_SHARED_QUERIES * records.shape[0]
        ):
            # Balanced batches can score only each request's own queries. The
            # old row loop contracts every query against every request, then
            # discards the cross-request scores. Ignore zero-length padding
            # when checking capacity; uneven batches retain the general path.
            assert page_records is not None  # implied by `pages`
            groups = records.shape[0]
            group_tokens = -(-tokens // groups)
            belongs = (jnp.arange(groups)[:, None] == requests) & (lengths > 0)
            fits = jnp.all(belongs.sum(axis=1) <= group_tokens)

            def grouped_rows(_):
                token_ids = jax.vmap(
                    lambda mask: jnp.nonzero(
                        mask, size=group_tokens, fill_value=tokens
                    )[0]
                )(belongs).reshape(-1)
                grouped_q = jnp.pad(q, ((0, 1), (0, 0), (0, 0)))[token_ids]
                grouped_weights = jnp.pad(weights, ((0, 1), (0, 0), (0, 0)))[token_ids]

                def grouped_step(index, result):
                    addresses = jax.lax.dynamic_slice_in_dim(
                        records, index * tile, tile, axis=1
                    )
                    words = gather_cache_page_words(
                        pool, addresses[:, ::page_records], page_records, dim
                    )
                    values = _score_word_rows(
                        grouped_q,
                        grouped_weights,
                        words.reshape(
                            groups, tile // index_word_scores.SLOTS, ROW_WORDS
                        ),
                        halfplane=halfplane,
                    )
                    # Sentinel token IDs are dropped, including duplicate
                    # padding slots. Every active token is written once.
                    ordered = jnp.full((tokens, tile), -jnp.inf, jnp.float32)
                    ordered = ordered.at[token_ids].set(values, mode="drop")
                    return jax.lax.dynamic_update_slice_in_dim(
                        result, ordered, index * tile, axis=1
                    )

                trips = jnp.minimum(
                    (row_lengths.max() + tile - 1) // tile, padded_width // tile
                )
                return jax.lax.fori_loop(0, trips, grouped_step, scores)[:, :width]

            return jax.lax.cond(fits, grouped_rows, shared_rows, None)
        return shared_rows(None)

    trips = jnp.minimum(
        (jnp.maximum(lengths.max(), 0) + tile - 1) // tile, padded_width // tile
    )
    if pages and dim == index_word_scores.DIM and q.dtype == jnp.bfloat16:
        # Score the gathered word rows in place: splitting them into
        # BF16 keys and relaying out [rows, 2, 64] -> [keys, 128] cost three
        # HBM passes per tile (8.7 of 17.5 us per 8K-key tile at 2 rows).
        # The kernel emits record order: a packed [tokens, slots, rows]
        # result cost 68 us per call to interleave in XLA.
        assert page_records is not None  # implied by `pages`
        page = page_records
        rows = tile // index_word_scores.SLOTS

        def _run_word_loop(rec):
            rec = jnp.pad(rec, ((0, 0), (0, -width % tile)), constant_values=-1)
            n_rows = rec.shape[0]

            def word_step(index, result):
                addresses = jax.lax.dynamic_slice_in_dim(
                    rec, index * tile, tile, axis=1
                )
                words = gather_cache_page_words(pool, addresses[:, ::page], page, dim)
                values = _score_word_rows(
                    q,
                    weights,
                    words.reshape(n_rows, rows, ROW_WORDS),
                    halfplane=halfplane,
                )
                return jax.lax.dynamic_update_slice_in_dim(
                    result, values, index * tile, axis=1
                )

            return jax.lax.fori_loop(0, trips, word_step, scores)[:, :width]

        if requests is not None and tokens >= 6 and tokens % 6 == 0:
            group_tokens = 6
            groups = tokens // group_tokens
            can_group = jnp.all(
                requests.reshape(groups, group_tokens) == requests[::group_tokens, None]
            )
            return jax.lax.cond(
                can_group,
                lambda _: _run_word_loop(
                    jnp.take(records, requests[::group_tokens], axis=0)
                ),
                lambda _: _run_word_loop(jnp.take(records, requests, axis=0)),
                None,
            )
        if requests is not None:
            records = jnp.take(records, requests, axis=0)
        return _run_word_loop(records)

    if requests is not None:
        records = jnp.take(records, requests, axis=0)
    records = jnp.pad(records, ((0, 0), (0, -width % tile)), constant_values=-1)

    def step(index, result):
        # Read only the tile consumed by this iteration. Gathering before
        # the loop materializes [tokens, max_context / ratio, dim] even when
        # lengths restrict execution to a single tile (or no tiles).
        addresses = jax.lax.dynamic_slice_in_dim(records, index * tile, tile, axis=1)
        if pages:
            assert page_records is not None
            # Tiles start on page boundaries: one gather per page.
            keys = gather_cache_page_values(
                pool,
                addresses[:, ::page_records],
                page_records,
                dim,
                halfplane=halfplane,
            )
        else:
            keys = gather_cache_values(
                pool, addresses.reshape(-1), dim, split=True, halfplane=halfplane
            )
        keys = keys.reshape(records.shape[0], tile, dim)
        logits = jnp.einsum("thd,tcd->thc", q, keys, preferred_element_type=jnp.float32)
        return jax.lax.dynamic_update_slice_in_dim(
            result, head_sum(logits), index * tile, axis=1
        )

    return jax.lax.fori_loop(0, trips, step, scores)[:, :width]


_indexer_op = None
_indexer_2bank_op = None


def _decode_select(
    q,
    weights,
    pool,
    pages,
    lengths,
    *,
    page_records,
    candidate_count=0,
    candidate_block_size=1,
    halfplane: bool = False,
):
    """Run the complete TC/SC decode pipeline inside the manual wrapper mesh."""
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_vllm_model_wrapper_context,
    )

    select = functools.partial(
        decode_index_selection.decode_indexer,
        page_records=page_records,
        key_tile=4096,
        candidate_count=candidate_count,
        candidate_block_size=candidate_block_size,
        halfplane=halfplane,
    )
    try:
        mesh = get_vllm_model_wrapper_context().mesh
    except AssertionError:
        return select(q, weights, pool, pages, lengths)
    return jax.shard_map(
        select, mesh=mesh, in_specs=(P(),) * 5, out_specs=P(), check_vma=False
    )(q, weights, pool, pages, lengths)


def _prefill_score(
    q,
    weights,
    pool,
    records,
    limit,
    *,
    page_records,
    halfplane: bool = False,
    scheduling_group_id: int | None = None,
):
    """Resident-query scorer, manually replicated across the wrapper mesh."""
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_vllm_model_wrapper_context,
    )

    def score(q, weights, pool, records, limit):
        return prefill_index_scores.prefill_index_scores(
            q,
            weights,
            pool,
            records[:, ::page_records],
            limit,
            page_records=page_records,
            query_tile=prefill_tuning.QUERY_TILE,
            key_tile=prefill_tuning.KEY_TILE,
            records=records,
            halfplane=halfplane,
            scheduling_group_id=scheduling_group_id,
        )

    try:
        mesh = get_vllm_model_wrapper_context().mesh
    except AssertionError:
        return score(q, weights, pool, records, limit)
    return jax.shard_map(
        score, mesh=mesh, in_specs=(P(),) * 5, out_specs=P(), check_vma=False
    )(q, weights, pool, records, limit)


def _prefill_top_k(scores, lengths, *, scheduling_group_id: int | None = None):
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_vllm_model_wrapper_context,
    )

    def select(s, n):
        ids = streaming_topk.streaming_topk(
            s, valid_lengths=n, k=512, scheduling_group_id=scheduling_group_id
        )
        values = jnp.take_along_axis(s, jnp.maximum(ids, 0), axis=1)
        ids = jnp.where((ids >= 0) & (values > -jnp.inf), ids, -1)
        return _compact_selected(ids)

    try:
        mesh = get_vllm_model_wrapper_context().mesh
    except AssertionError:
        return select(scores, lengths)
    return jax.shard_map(
        select, mesh=mesh, in_specs=(P(), P()), out_specs=P(), check_vma=False
    )(scores, lengths)


def _score_word_rows(
    q: jax.Array, weights: jax.Array, words: jax.Array, halfplane: bool = False
) -> jax.Array:
    """`index_word_scores.score_word_rows` on every device (replicated)."""
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_vllm_model_wrapper_context,
    )

    score = functools.partial(
        index_word_scores.score_word_rows,
        interpret=jax.default_backend() != "tpu",
        halfplane=halfplane,
    )
    try:
        mesh = get_vllm_model_wrapper_context().mesh
    except Exception:  # Standalone (tests): one device, no mesh.
        return score(q, weights, words)
    # Mosaic calls need manual partitioning; every device scores locally.
    return jax.shard_map(
        score, mesh=mesh, in_specs=(P(), P(), P()), out_specs=P(), check_vma=False
    )(q, weights, words)


def _prefix_count(mask: jax.Array, block: int = 128) -> jax.Array:
    """Inclusive ``cumsum(mask, -1)`` as exact blockwise 0/1 matmuls.

    A full-width cumsum is ~0.5 ms at ``[512, 16896]`` on TPU; the MXU is not.
    """
    rows, width = mask.shape
    blocks = -(-width // block)
    x = jnp.pad(mask, ((0, 0), (0, blocks * block - width))).astype(jnp.bfloat16)
    x = x.reshape(rows, blocks, block)
    upper = jnp.triu(jnp.ones((block, block), jnp.bfloat16))
    inner = jnp.einsum("nbi,ij->nbj", x, upper, preferred_element_type=jnp.float32)
    totals = inner[..., -1].astype(jnp.bfloat16)  # at most `block`: exact
    strict = jnp.triu(jnp.ones((blocks, blocks), jnp.bfloat16), 1)
    before = jnp.einsum("nb,bc->nc", totals, strict, preferred_element_type=jnp.float32)
    return (inner + before[..., None]).reshape(rows, -1)[:, :width].astype(jnp.int32)


def _threshold_top_k(scores: jax.Array, k: int) -> jax.Array:
    """Ascending indices of ``jax.lax.top_k(scores, k)``, no sort.

    Finds the k-th largest value by bisecting an order-preserving integer key,
    breaks ties at it by lowest index (as top_k does), then compacts.
    """
    rows = scores.shape[0]
    # No -0.0 -> +0.0 folding: top_k ranks +0.0 above -0.0 (TPU and CPU), as
    # does the order-preserving key below.
    bits = jax.lax.bitcast_convert_type(scores, jnp.int32)
    key = jnp.where(bits < 0, bits ^ jnp.int32(0x7FFFFFFF), bits) ^ jnp.int32(-(2**31))
    key = jax.lax.bitcast_convert_type(key, jnp.uint32)

    def bisect(bit, threshold):
        candidate = threshold | (jnp.uint32(1) << (31 - bit).astype(jnp.uint32))
        enough = (key >= candidate[:, None]).sum(-1) >= k
        return jnp.where(enough, candidate, threshold)

    threshold = jax.lax.fori_loop(0, 32, bisect, jnp.zeros((rows,), jnp.uint32))
    above = key > threshold[:, None]
    tied = key == threshold[:, None]
    remaining = k - above.sum(-1)
    chosen = above | (tied & (_prefix_count(tied) <= remaining[:, None]))
    ranks = _prefix_count(chosen)
    targets = jnp.broadcast_to(jnp.arange(1, k + 1, dtype=jnp.int32), (rows, k))
    return jax.vmap(functools.partial(jnp.searchsorted, side="left"))(
        ranks, targets
    ).astype(jnp.int32)


def _sparsecore_top_k(scores: jax.Array, k: int, lengths: jax.Array) -> jax.Array:
    """V4's SparseCore top-k (`kernels/deepseek_v4/sparsecore_topk.py`).

    Same set as ``jax.lax.top_k`` over the finite scores (ties at the k-th
    value by lowest index), unordered, ``-1`` suffix-padded; ``-inf`` and
    columns ``>= lengths`` are never selected.
    """
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import sparsecore_topk
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_vllm_model_wrapper_context,
    )

    select = functools.partial(sparsecore_topk, k=k)
    try:
        mesh = get_vllm_model_wrapper_context().mesh
    except Exception:  # Standalone (tests): one device, no mesh.
        return select(scores, row_lengths=lengths)
    # Mosaic calls need manual partitioning; every device selects locally.
    return jax.shard_map(
        lambda s, n: select(s, row_lengths=n),
        mesh=mesh,
        in_specs=(P(), P()),
        out_specs=P(),
        check_vma=False,
    )(scores, lengths)


def _visible_top_k(
    scores: jax.Array,
    k: int,
    bound: jax.Array,
    lengths: jax.Array | None = None,
    unordered: bool = False,
) -> jax.Array:
    """Ascending ``jax.lax.top_k(scores, k)`` indices when columns ``>= bound``
    are ``-inf``.

    Selects only within a static power-of-two bucket of columns covering
    ``bound`` (the full width otherwise). Exact: buckets hold at least ``k``
    columns, so ``-inf`` ties still resolve to the lowest indices. Large
    selections use `_threshold_top_k`; small ones (decode) sort.

    ``unordered`` callers (which re-sort and drop ``-1``) accept the finite
    top-k set in any order, ``-1`` padded instead of ``-inf`` columns; large
    selections then run on SparseCore over the full width, bounded by
    ``lengths`` instead of buckets (no SparseCore op inside `lax.switch`).
    Decode query parts (fewer than 16 rows) also run on SparseCore, unpadded:
    XLA ``top_k`` over 100K columns was 0.53 ms slower at L20 for 3 rows, and
    padding to a 16-row tile ran 2 subcore waves instead of 1.
    """
    width = scores.shape[1]
    if (
        unordered
        and lengths is not None
        and width >= 2048
        and width % 16 == 0
        and k % 16 == 0
    ):
        return _sparsecore_top_k(scores, k, jnp.minimum(lengths, width))

    def select(columns, scores):
        scores = scores[:, :columns]
        if scores.shape[0] >= 16 and columns >= 2048:
            return _threshold_top_k(scores, k)
        return jnp.sort(jax.lax.top_k(scores, k)[1], axis=-1)

    buckets = []
    bucket = max(k, 1024)
    while bucket < width:
        buckets.append(bucket)
        bucket *= 2
    buckets.append(width)
    if len(buckets) == 1:
        return select(width, scores)
    branch = jnp.searchsorted(jnp.asarray(buckets[:-1]), bound, side="left")
    return jax.lax.switch(
        branch, [functools.partial(select, b) for b in buckets], scores
    )


def select_indices_jax(
    scores: jax.Array,
    lengths: jax.Array,
    candidates: jax.Array,
    topk: int,
    block_size: int,
    candidate_write: bool,
    candidate_filter: bool,
    unordered: bool = False,
    batch_max: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Source publishes candidates without filtering its own selection.

    ``unordered``: see `_visible_top_k` (the selection may come back in any
    order with ``-1`` anywhere; FP8 attention re-sorts via `compact_records`).
    ``batch_max``: the whole batch's longest length when ``scores`` holds only
    part of the batch's queries (defaults to ``lengths.max()``).
    """
    count, width = scores.shape
    if topk < 1 or width < 1 or block_size < 1:
        raise ValueError("Positive top-k, score width and block size required")
    if candidate_write and candidate_filter:
        raise ValueError("Candidate sources must not filter their own scores")
    if candidates.ndim != 2 or candidates.shape[0] != count:
        raise ValueError("Candidate rows must match query rows")
    lengths = jnp.clip(lengths.reshape(count), 0, width)
    batch_max = (
        lengths.max() if batch_max is None else jnp.clip(batch_max.max(), 0, width)
    )
    logical = jnp.arange(width)[None, :]
    visible = logical < lengths[:, None]
    scores = jnp.where(visible, scores, -jnp.inf)
    published = candidates
    # When the candidate buffer covers every block, publication/filtering are
    # identities. Keep the caller's buffer untouched in this static case.
    if (candidate_write or candidate_filter) and candidates.shape[
        1
    ] * block_size < width:
        blocks = (width + block_size - 1) // block_size
        if candidate_write:
            block_scores = (
                jnp.pad(
                    scores,
                    ((0, 0), (0, blocks * block_size - width)),
                    constant_values=-jnp.inf,
                )
                .reshape(count, blocks, block_size)
                .max(-1)
            )
            last = (lengths - 1) // block_size
            block_scores = jnp.where(
                jnp.arange(blocks)[None] == last[:, None], jnp.inf, block_scores
            )
            cand_k = candidates.shape[1]
            if unordered and blocks >= 2048 and blocks % 16 == 0 and cand_k % 16 == 0:
                block_lengths = (lengths + block_size - 1) // block_size
                published = _sparsecore_top_k(
                    block_scores, cand_k, jnp.minimum(block_lengths, blocks)
                ).astype(jnp.int32)
            else:
                values, ids = jax.lax.top_k(block_scores, cand_k)
                published = jnp.where(values > -jnp.inf, ids, -1).astype(jnp.int32)
        else:
            # One extra column absorbs invalid candidate IDs; no 3-D membership
            # tensor and no dependence on the order of candidate IDs.
            padded_blocks = ((blocks + 128) // 128) * 128
            valid = (candidates >= 0) & (candidates < blocks)
            ids = jnp.where(valid, candidates, blocks)
            membership = (
                jnp.zeros((count, padded_blocks), jnp.int32)
                .at[jnp.arange(count)[:, None], ids]
                .add(1)
            )
            allowed = jnp.broadcast_to(
                membership[:, :blocks, None] > 0, (count, blocks, block_size)
            ).reshape(count, blocks * block_size)[:, :width]
            # Match the upstream batch-wide short-context bypass.
            scores = jnp.where(allowed | (batch_max <= topk), scores, -jnp.inf)
    # Ascending: preserves V4.1's logical, hence attention accumulation, order.
    k = min(topk, width)
    indices = _visible_top_k(scores, k, lengths.max(), lengths, unordered)
    if unordered and width >= 2048 and width % 16 == 0 and k % 16 == 0:
        return _compact_selected(indices.astype(jnp.int32)), published
    values = jnp.take_along_axis(scores, jnp.maximum(indices, 0), axis=1)
    valid = (indices >= 0) & (values > -jnp.inf)
    selected = jnp.where(valid, indices, -1).astype(jnp.int32)
    if unordered:
        selected = _compact_selected(selected)
    return selected, published


def paged_indexer_jax(
    q: jax.Array,
    pool: jax.Array,
    records: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    candidates: jax.Array,
    requests: jax.Array,
    topk: int,
    block_size: int,
    candidate_write: bool,
    candidate_filter: bool,
    unordered: bool = False,
    batch_max: jax.Array | None = None,
    page_records: int | None = None,
    prefill_mode: str = "streaming",
    halfplane: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """``records`` is ``[requests, width]``; ``requests`` maps tokens to rows.

    ``batch_max``: see `select_indices_jax`. ``page_records``: see
    `paged_index_scores_jax`; candidate scoring gathers runs of
    ``gcd(block_size, page_records)`` records (a block lies in one page).
    """

    def score_context(records, query_tile, mode="baseline"):
        # Finish selection before reusing a query tile's score buffer. Key
        # tiling alone still materializes [all queries, max context]: at
        # 2048 x 1M a single FP32 score matrix is 8 GiB, before top-k copies.
        # Reuse the native indexer's query tile and score/selection pipeline.
        rows = q.shape[0]
        longest = lengths.max() if batch_max is None else batch_max
        group_tokens = -(-rows // records.shape[0])
        use_prefill = mode != "baseline"

        def run(grouped):
            tile = min(rows, query_tile)
            inputs = (q, weights, lengths, candidates, requests)
            if grouped:
                # Keep a request's queries together and slice its records too:
                # flat query tiles otherwise gather the same keys per query.
                group_rows = min(records.shape[0], tile // group_tokens)
                tile = group_rows * group_tokens
                token_ids = jax.vmap(
                    lambda mask: jnp.nonzero(mask, size=group_tokens, fill_value=rows)[
                        0
                    ]
                )(belongs).reshape(-1)
                token_ids = jnp.pad(
                    token_ids, (0, -token_ids.size % tile), constant_values=rows
                )
                inputs = jax.tree.map(
                    lambda x: jnp.pad(x, ((0, 1),) + ((0, 0),) * (x.ndim - 1))[
                        token_ids
                    ],
                    inputs,
                )
                local_requests = jnp.repeat(jnp.arange(group_rows), group_tokens)
                record_tiles = jnp.pad(
                    records,
                    ((0, -records.shape[0] % group_rows), (0, 0)),
                    constant_values=-1,
                )
                count = token_ids.size
            else:
                count = rows + (-rows % tile)
                inputs = jax.tree.map(
                    lambda x: jnp.pad(
                        x, ((0, count - rows),) + ((0, 0),) * (x.ndim - 1)
                    ),
                    inputs,
                )

            def prepare(i):
                values = jax.tree.map(
                    lambda x: jax.lax.dynamic_slice_in_dim(x, i * tile, tile), inputs
                )
                if grouped:
                    row_records = jax.lax.dynamic_slice_in_dim(
                        record_tiles, i * group_rows, group_rows
                    )
                    return (*values[:-1], local_requests, row_records)
                return (*values, records)

            def score(args, scheduling_group_id: int | None = None):
                queries, row_weights, row_lengths, _, row_requests, row_records = args
                if use_prefill:
                    # The outer route proves every live query shares a request.
                    # Entries are first-record addresses, not page numbers.
                    request = requests[jnp.argmax(lengths)]
                    shared_records = jax.lax.dynamic_slice_in_dim(
                        row_records, request, 1
                    )
                    return _prefill_score(
                        split_halves(queries),
                        row_weights.astype(jnp.float32)[:, :, None],
                        pool,
                        shared_records,
                        jnp.max(row_lengths, keepdims=True),
                        page_records=page_records,
                        halfplane=halfplane,
                        scheduling_group_id=scheduling_group_id,
                    )
                else:
                    scores = paged_index_scores_jax(
                        queries,
                        pool,
                        row_records,
                        row_weights,
                        row_lengths,
                        row_requests,
                        page_records,
                        halfplane=halfplane,
                    )
                return jnp.where(
                    jnp.take(row_records >= 0, row_requests, axis=0), scores, -jnp.inf
                )

            def select(i, scores, scheduling_group_id: int | None = None):
                _, _, row_lengths, row_candidates, _, _ = prepare(i)
                if use_prefill:
                    selected = _prefill_top_k(
                        scores, row_lengths, scheduling_group_id=scheduling_group_id
                    )
                    return selected, row_candidates
                return select_indices_jax(
                    scores,
                    row_lengths,
                    row_candidates,
                    topk,
                    block_size,
                    candidate_write,
                    candidate_filter,
                    unordered,
                    longest,
                )

            if not grouped and rows <= tile:
                return select(jnp.int32(0), score(prepare(jnp.int32(0))))

            def write(i, selected, output):
                if grouped:
                    ids = jax.lax.dynamic_slice_in_dim(token_ids, i * tile, tile)
                    return jax.tree.map(
                        lambda dst, src: dst.at[ids].set(src, mode="drop"),
                        output,
                        selected,
                    )
                return jax.tree.map(
                    lambda dst, src: jax.lax.dynamic_update_slice_in_dim(
                        dst, src, i * tile, axis=0
                    ),
                    output,
                    selected,
                )

            if grouped:
                # Zero-length queries were omitted from the request groups.
                published = (
                    jnp.full_like(candidates, -1)
                    if candidate_write
                    and candidates.shape[1] * block_size < records.shape[1]
                    else candidates
                )
            else:
                published = jnp.zeros((count, candidates.shape[1]), candidates.dtype)
            output = (
                jnp.full(
                    (rows if grouped else count, min(topk, records.shape[1])),
                    -1,
                    jnp.int32,
                ),
                published,
            )
            pipeline = (
                indexer.pipeline_score_topk_bounded
                if use_prefill
                else indexer.pipeline_score_topk
            )
            selected, published = pipeline(
                count // tile, prepare, score, select, write, output
            )
            return selected[:rows], published[:rows]

        if (
            not use_prefill
            and page_records is not None
            and q.dtype == jnp.bfloat16
            and q.shape[-1] == index_word_scores.DIM
            and records.shape[0] > 1
            and 1 < group_tokens <= min(query_tile, _INDEX_MAX_SHARED_QUERIES)
        ):
            belongs = (jnp.arange(records.shape[0])[:, None] == requests) & (
                lengths > 0
            )
            fits = jnp.all(belongs.sum(axis=1) <= group_tokens)
            return jax.lax.cond(fits, lambda _: run(True), lambda _: run(False), None)
        return run(False)

    def prefill_context():
        widths = prefill_index_scores.context_buckets(records.shape[1])

        def run(columns):
            bounded_records = records[:, :columns]
            bounded_records = jnp.pad(
                bounded_records,
                ((0, 0), (0, columns - bounded_records.shape[1])),
                constant_values=-1,
            )
            return score_context(
                bounded_records, prefill_tuning.QUERY_TILE, prefill_mode
            )

        def all_indices(_):
            # No ranking is necessary when every visible key fits in TopK.
            ids = jnp.broadcast_to(
                jnp.arange(min(topk, records.shape[1]), dtype=jnp.int32),
                (q.shape[0], min(topk, records.shape[1])),
            )
            valid = (ids < lengths[:, None]) & (
                jnp.take(records[:, : ids.shape[1]], requests, axis=0) >= 0
            )
            return _compact_selected(jnp.where(valid, ids, -1)), candidates

        def ranked(_):
            branch = jnp.searchsorted(
                jnp.asarray(widths[:-1]), lengths.max(), side="left"
            )
            return jax.lax.switch(branch, [functools.partial(run, b) for b in widths])

        return jax.lax.cond(lengths.max() <= topk, all_indices, ranked, None)

    def decode_context():
        count = records.shape[0]
        queries = q.shape[0] // count
        # Pad the compact page table, not the full per-token record matrix.
        width = max(4096, 1 << (records.shape[1] - 1).bit_length())
        pages = jnp.pad(
            records[:, ::page_records],
            ((0, 0), (0, (width - records.shape[1]) // page_records)),
            constant_values=-1,
        )
        publish = (
            candidate_write and candidates.shape[1] * block_size < records.shape[1]
        )
        result = _decode_select(
            split_halves(q).reshape(count, queries, *q.shape[1:]),
            weights.astype(jnp.float32).reshape(count, queries, q.shape[1], 1),
            pool,
            pages,
            lengths.reshape(count, queries),
            page_records=page_records,
            candidate_count=candidates.shape[1] if publish else 0,
            candidate_block_size=block_size,
            halfplane=halfplane,
        )
        return result if publish else (result, candidates)

    def full_context(_):
        if (
            jax.default_backend() == "tpu"
            and unordered
            and not candidate_filter
            and (
                not candidate_write
                or (
                    block_size > 0
                    and 4096 % block_size == 0
                    and candidates.shape[1] > 0
                    and candidates.shape[1] % 16 == 0
                )
            )
            and topk == 512
            and decode_index_selection.supports_decode(q, pool, records, page_records)
        ):
            expected = jnp.repeat(
                jnp.arange(records.shape[0], dtype=jnp.int32),
                q.shape[0] // records.shape[0],
            )
            # Draft/mixed batches can share the padded shape but have different
            # request boundaries. Never reinterpret those rows as uniform groups.
            regular = jnp.all((requests == expected) | (lengths == 0))
            return jax.lax.cond(regular, decode_context, existing_context)
        return existing_context()

    def existing_context():
        rows = q.shape[0]
        if (
            records.shape[0] > 1
            and records.shape[0] < rows <= _INDEX_MAX_SHARED_QUERIES * records.shape[0]
            and rows * records.shape[1] <= _INDEX_UNTILED_SCORE_ELEMENTS
        ):
            # Decode-shaped batch (e.g. DSpark verify: 48 requests x 6 rows).
            # Query tiles of 32/128 rows hold fewer rows than request rows, so
            # `paged_index_scores_jax` could not share each request's key tile
            # across its queries and would gather every row's context again.
            # The untiled score buffer is small here; keep one query tile.
            return score_context(records, rows)
        columns, query_tile = tuned_block_sizes.PAGED_INDEXER_PREFILL_BUCKET
        if (
            not candidate_write
            and not candidate_filter
            and topk <= columns
            and q.shape[0] >= query_tile
            and records.shape[1] > columns
        ):
            # Select a smaller working shape without limiting request length.
            # Publication/filtering retain their full-width candidate contract.
            def bounded(_):
                return score_context(records[:, :columns], query_tile)

            return jax.lax.cond(
                lengths.max() <= columns,
                bounded,
                lambda _: score_context(records, indexer.QUERY_TILE),
                None,
            )
        return score_context(records, indexer.QUERY_TILE)

    if (
        prefill_mode != "baseline"
        and jax.default_backend() == "tpu"
        and unordered
        and not candidate_write
        and not candidate_filter
        and topk == 512
        and records.shape[1] >= topk
        and prefill_index_scores.supports_chunked_prefill(q, pool, page_records)
    ):
        live = lengths > 0
        request = requests[jnp.argmax(lengths)]
        # Infer eligibility from live rows, not a prefill/decode phase flag.
        # Short contexts need no ranking; padding does not select a request.
        prefill = (live.sum() >= prefill_tuning.QUERY_TILE) | (lengths.max() <= topk)
        prefill &= jnp.all((requests == request) | ~live)
        return jax.lax.cond(prefill, prefill_context, lambda: full_context(None))

    width = records.shape[1]
    if not candidate_filter or candidates.shape[1] * block_size >= width:
        return full_context(None)

    # Bind shared-query prefill to one scoring pipeline when its static key
    # volume is no larger than per-token candidate scoring. Both paths apply
    # the same candidate mask and batch-wide short-context bypass. Keeping
    # this choice static avoids compiling both pipelines into every layer.
    if (
        q.shape[0] > records.shape[0]
        and records.shape[0] * width <= q.shape[0] * candidates.shape[1] * block_size
    ):
        return full_context(None)

    def candidate_context(_):
        # Candidates differ per token, so this path reads per-token records.
        # Sort before expanding so equal-score ties retain the full-context
        # top_k's logical ordering. Duplicate and invalid blocks cannot select
        # the same key twice or read outside this request's causal history.
        blocks = (width + block_size - 1) // block_size
        ids = jnp.sort(
            jnp.where((candidates >= 0) & (candidates < blocks), candidates, blocks),
            axis=-1,
        )
        unique = jnp.concatenate(
            (jnp.ones_like(ids[:, :1], dtype=jnp.bool_), ids[:, 1:] != ids[:, :-1]),
            axis=-1,
        )
        logical = (ids[..., None] * block_size + jnp.arange(block_size)).reshape(
            q.shape[0], -1
        )
        valid = logical < jnp.minimum(lengths[:, None], width)
        valid &= jnp.repeat(unique & (ids < blocks), block_size, axis=-1)
        # Gather only the candidate columns: per-token copies of whole
        # records rows ([tokens, width]) cost more than the candidates.
        compact = records[requests[:, None], jnp.minimum(logical, width - 1)]
        compact = jnp.where(valid, compact, -1)
        # This bounds both key reads and QK work by the candidate pool size;
        # zero-length padding rows do not trigger key reads.
        compact_lengths = jnp.where(lengths > 0, compact.shape[1], 0)
        # Candidate blocks are aligned runs of a page's contiguous records.
        segment = math.gcd(block_size, page_records) if page_records else None
        scores = paged_index_scores_jax(
            q,
            pool,
            compact,
            weights,
            compact_lengths,
            page_records=segment,
            halfplane=halfplane,
        )
        scores = jnp.where(compact >= 0, scores, -jnp.inf)
        output_width = min(topk, width)
        if (
            unordered
            and scores.shape[1] >= 2048
            and scores.shape[1] % 16 == 0
            and output_width % 16 == 0
        ):
            # Same finite top-k set as `jax.lax.top_k` below (SparseCore never
            # selects -inf); `_compact_selected` sorts, so order is irrelevant.
            picked = _sparsecore_top_k(scores, output_width, compact_lengths)
            selected = jnp.take_along_axis(logical, jnp.maximum(picked, 0), axis=1)
            return (
                _compact_selected(
                    jnp.where(picked >= 0, selected, -1).astype(jnp.int32)
                ),
                candidates,
            )
        padding = max(0, output_width - scores.shape[1])
        scores = jnp.pad(scores, ((0, 0), (0, padding)), constant_values=-jnp.inf)
        logical = jnp.pad(logical, ((0, 0), (0, padding)), constant_values=width)
        values, indices = jax.lax.top_k(scores, output_width)
        selected = jnp.take_along_axis(logical, indices, axis=1)
        selected_valid = values > -jnp.inf
        if unordered:
            return (
                _compact_selected(
                    jnp.where(selected_valid, selected, -1).astype(jnp.int32)
                ),
                candidates,
            )
        # The dense reference fills an underfull top-k with the lowest logical
        # -inf positions, then sorts all positions before masking them. Preserve
        # those padding locations too: moving padding across a 64-key attention
        # tile changes BF16 accumulation boundaries.
        occupied = jnp.zeros((q.shape[0], output_width + 1), jnp.bool_)
        occupied_ids = jnp.where(
            selected_valid & (selected < output_width), selected, output_width
        )
        occupied = occupied.at[jnp.arange(q.shape[0])[:, None], occupied_ids].set(True)
        fillers = jnp.sort(
            jnp.where(~occupied[:, :output_width], jnp.arange(output_width), width),
            axis=-1,
        )
        filler_rank = jnp.maximum(
            jnp.arange(output_width) - selected_valid.sum(-1)[:, None], 0
        )
        selected = jnp.where(
            selected_valid, selected, jnp.take_along_axis(fillers, filler_rank, axis=1)
        )
        order = jnp.argsort(selected, axis=-1)
        selected = jnp.take_along_axis(selected, order, axis=1)
        selected_valid = jnp.take_along_axis(selected_valid, order, axis=1)
        return jnp.where(selected_valid, selected, -1).astype(jnp.int32), candidates

    # Preserve the reference's batch-wide short-context bypass, and use the
    # shared-key-tile full_context path whenever the active requests' total
    # key reads are at most the per-token candidate key reads in prefill
    # (q.shape[0] > records.shape[0], where all tokens of a request share one
    # key tile in full_context).
    longest = lengths.max() if batch_max is None else batch_max.max()
    if q.shape[0] > records.shape[0]:
        row_lengths = (
            jnp.zeros((records.shape[0],), jnp.int32)
            .at[requests]
            .max(jnp.maximum(lengths, 0))
        )
        cand_keys = jnp.sum(lengths > 0) * (candidates.shape[1] * block_size)
        use_full_context = (longest <= topk) | (row_lengths.sum() <= cand_keys)
    else:
        use_full_context = longest <= topk
    return jax.lax.cond(
        use_full_context,
        full_context,
        candidate_context,
        None,
    )


def write_index_keys(
    pool: jax.Array | tuple[jax.Array, jax.Array],
    write_records: jax.Array,
    keys: jax.Array,
    halfplane: bool = False,
):
    """Publish index records through the same cache writer as KV and state."""
    if keys.dtype != jnp.bfloat16:
        raise ValueError("Index keys must be BF16")
    if keys.ndim != 2 or write_records.shape != (keys.shape[0],):
        raise ValueError("One physical record index per key is required")
    if halfplane:
        return scatter_index_keys(pool, write_records, keys)
    return scatter_cache_values(cast(Any, pool), write_records, keys)


def update_and_select_jax(
    q: jax.Array,
    pool: jax.Array,
    records: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    candidates: jax.Array,
    write_records: jax.Array,
    keys: jax.Array,
    requests: jax.Array,
    batch_max: jax.Array,
    topk: int,
    block_size: int,
    candidate_write: bool,
    candidate_filter: bool,
    unordered: bool,
    page_records: int = 0,
    prefill_mode: str = "streaming",
    halfplane: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``page_records`` 0 disables page gathers (see `paged_indexer_jax`)."""
    updated = write_index_keys(pool, write_records, keys, halfplane=halfplane)
    selected, published = paged_indexer_jax(
        q,
        updated,
        records,
        weights,
        lengths,
        candidates,
        requests,
        topk,
        block_size,
        candidate_write,
        candidate_filter,
        unordered,
        batch_max,
        page_records or None,
        prefill_mode,
        halfplane=halfplane,
    )
    return updated, selected, published


def update_and_select_2bank_jax(
    q: jax.Array,
    pool_lo: jax.Array,
    pool_hi: jax.Array,
    records: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    candidates: jax.Array,
    write_records: jax.Array,
    keys: jax.Array,
    requests: jax.Array,
    batch_max: jax.Array,
    topk: int,
    block_size: int,
    candidate_write: bool,
    candidate_filter: bool,
    unordered: bool,
    page_records: int = 0,
    prefill_mode: str = "streaming",
    halfplane: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    (updated_lo, updated_hi), selected, published = cast(
        Any,
        update_and_select_jax(
            q,
            cast(Any, (pool_lo, pool_hi)),
            records,
            weights,
            lengths,
            candidates,
            write_records,
            keys,
            requests,
            batch_max,
            topk,
            block_size,
            candidate_write,
            candidate_filter,
            unordered,
            page_records,
            prefill_mode,
            halfplane=halfplane,
        ),
    )
    return updated_lo, updated_hi, selected, published


def initialize_indexer():
    from .candidate_cache import initialize_candidate_cache

    initialize_candidate_cache()
    global _indexer_op, _indexer_2bank_op
    if _indexer_op is not None:
        return
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_pallas_op_mesh,
    )

    _indexer_op = pallas.jax_op(
        "pallas::dsv41v2_exact_indexer",
        update_and_select_jax,
        donate_argnums=(1,),
        mesh=get_pallas_op_mesh(),
        input_partition_specs=cast(
            tuple[tuple[str, ...], ...], (P(),) * 10 + (None,) * 8
        ),
    )

    def fake(
        q,
        pool,
        records,
        weights,
        lengths,
        candidates,
        write_records,
        keys,
        requests,
        batch_max,
        topk,
        block_size,
        candidate_write,
        candidate_filter,
        unordered,
        page_records=0,
        prefill_mode="streaming",
        halfplane=False,
    ):
        return (
            torch.empty_like(pool),
            records.new_empty(
                (q.shape[0], min(topk, records.shape[1])), dtype=torch.int32
            ),
            torch.empty_like(candidates),
        )

    _indexer_op.register_fake(fake)
    _indexer_2bank_op = pallas.jax_op(
        "pallas::dsv41v2_exact_indexer_2bank",
        update_and_select_2bank_jax,
        donate_argnums=(1, 2),
        mesh=get_pallas_op_mesh(),
        input_partition_specs=cast(
            tuple[tuple[str, ...], ...], (P(),) * 11 + (None,) * 8
        ),
    )

    def fake_2bank(
        q,
        pool_lo,
        pool_hi,
        records,
        weights,
        lengths,
        candidates,
        write_records,
        keys,
        requests,
        batch_max,
        topk,
        block_size,
        candidate_write,
        candidate_filter,
        unordered,
        page_records=0,
        prefill_mode="streaming",
        halfplane=False,
    ):
        return (
            torch.empty_like(pool_lo),
            torch.empty_like(pool_hi),
            records.new_empty(
                (q.shape[0], min(topk, records.shape[1])), dtype=torch.int32
            ),
            torch.empty_like(candidates),
        )

    _indexer_2bank_op.register_fake(fake_2bank)


def paged_indexer(
    q,
    cache,
    records,
    weights,
    lengths,
    candidates,
    write_records,
    keys,
    requests,
    *,
    topk,
    block_size,
    candidate_write,
    candidate_filter,
    unordered=False,
    batch_max=None,
    page_records=None,
    prefill_mode=None,
):
    """``records`` is ``[requests, width]``; ``requests`` maps tokens to rows.

    ``batch_max`` (default ``lengths.max()``) is the longest length of the
    whole batch when the queries are only part of it. ``page_records``: see
    `paged_index_scores_jax`.
    """
    # Resolve the comparison selector in the torch-to-JAX wrapper and pass it
    # explicitly: compiled JAX functions must not read environment state.
    if prefill_mode is None:
        prefill_mode = "streaming"
    assert _indexer_op is not None
    if batch_max is None:
        batch_max = lengths.max()
    tail_args = (
        records.int(),
        weights,
        lengths.flatten().int(),
        candidates.int(),
        write_records.int(),
        keys,
        requests.int(),
        batch_max.reshape(1).int(),
        topk,
        block_size,
        candidate_write,
        candidate_filter,
        unordered,
        page_records or 0,
        prefill_mode,
        getattr(cache, "index_key_layout", "ordinary") == "halfplane",
    )
    if _has_hi_bank(cache):
        assert _indexer_2bank_op is not None
        updated_lo, updated_hi, selected, published = _indexer_2bank_op(
            q,
            cache.raw,
            cache.raw_hi,
            *tail_args,
        )
        return (updated_lo, updated_hi), selected, published
    return _indexer_op(
        q,
        cache.raw,
        *tail_args,
    )
