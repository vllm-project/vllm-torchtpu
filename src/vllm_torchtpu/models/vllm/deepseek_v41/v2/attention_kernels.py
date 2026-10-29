"""Paged attention and stage-local KV gathering."""

import contextlib
import dataclasses
import functools
import math
from collections.abc import Iterator
from typing import Any, cast

import jax
import jax.numpy as jnp
import torch

from .cache import (
    ROPE_WORDS,
    ROW_WORDS,
    _cache_num_rows,
    _has_hi_bank,
    gather_cache_pieces,
    merge_pieces,
    split_pieces,
)
from .indexer_impl import _compact_selected

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


def _general_paged_attention_jax(
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
    if (
        q.shape == (240, 64, 512)
        and q.dtype == jnp.bfloat16
        and swa_records.shape == (240, 192)
        and main_records.shape[1] == 0
        and isinstance(swa_pool, tuple)
        and scale == 512**-0.5
    ):
        shared = jnp.all(swa_records.reshape(48, 5, 192) == swa_records[::5, None, :])
        return jax.lax.cond(
            shared,
            lambda _: _shared_draft_attention(q, swa_pool, swa_records, sink),
            lambda _: _general_paged_attention_jax(
                q, swa_pool, main_pool, swa_records, main_records, sink, scale, gathered
            ),
            None,
        )
    return _general_paged_attention_jax(
        q, swa_pool, main_pool, swa_records, main_records, sink, scale, gathered
    )


def _shared_draft_attention(q, pool, records, sink):
    from vllm_torchtpu.kernels.shared_record_attention import shared_record_attention

    requests = q.shape[0] // 5
    pieces = split_pieces(q.reshape(requests, 320, 512), 4)
    shared = records[::5]
    kv = gather_cache_pieces(pool, shared.reshape(-1), 512).reshape(
        4, requests, 192, 128
    )
    kv = jnp.where((shared >= 0)[None, :, :, None], kv, 0)
    mask = jnp.pad((shared >= 0).astype(jnp.int32), ((0, 0), (0, 64)))[:, None, :]
    return shared_record_attention(
        mask, pieces, kv, jnp.tile(sink, 5)[:, None]
    ).reshape(q.shape)


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
    valid indices at the prefix and ``-1`` at the tail; their order is arbitrary.
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
