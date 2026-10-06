"""Paged attention, exact index selection and index-cache publication."""

import functools

import jax
import jax.numpy as jnp
import torch

from .cache import (
    gather_cache_pieces,
    gather_cache_values,
    merge_halves,
    scatter_cache_values,
    split_halves,
)

_attention_op = None
_fp8_attention_op = None


def _attention_state(q):
    tokens, heads, _ = q.shape
    return (
        jnp.full((tokens, heads), -1e30, jnp.float32),
        jnp.zeros((tokens, heads), jnp.float32),
        jnp.zeros(q.shape, jnp.float32),
    )


def _attend(q, pool, records, state, scale, tile=64):
    """Online-softmax accumulate one record set into (max, denominator, numerator)."""
    tokens, _, dim = q.shape
    width = records.shape[1]
    if not width:
        return state
    records = jnp.pad(records, ((0, 0), (0, -width % tile)), constant_values=-1)

    # Gather BF16 rows once outside; keep FP32 scratch tiled. Rows and q use
    # the split element order (no per-element unpack gather), as stacked
    # pieces contracted in one dot; attention is invariant to a shared
    # permutation of the head dimension.
    gathered = gather_cache_pieces(pool, records.reshape(-1), dim)
    count = gathered.shape[0]
    gathered = gathered.reshape(count, tokens, records.shape[1], -1)

    def pieces(x):
        return split_halves(x).reshape(*x.shape[:-1], count, -1)

    q = pieces(q).astype(jnp.float32)
    maximum, denominator, numerator = state
    state = maximum, denominator, pieces(numerator)

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
                "thpd,ptkd->thk",
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
        numerator = numerator * correction[..., None, None] + jnp.einsum(
            "thk,ptkd->thpd",
            exponentials.astype(jnp.bfloat16),
            kv,
            preferred_element_type=jnp.float32,
        )
        return new_max, denominator, numerator

    maximum, denominator, numerator = jax.lax.fori_loop(
        0, records.shape[1] // tile, step, state
    )
    return maximum, denominator, merge_halves(numerator.reshape(*q.shape[:2], dim))


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
    # Verify instead of trusting contiguity: every own record must land on
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
    if tokens % tile or tokens < 2 * tile or width % 64:
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
            return split_halves(x).reshape(count, tile, heads, pieces, -1)

        maximum, denominator, numerator = state
        shape = (count, tile, heads)
        maximum = maximum.reshape(shape)
        denominator = denominator.reshape(shape)
        numerator = split(numerator)
        scores = (
            jnp.einsum(
                "nthpd,pnkd->nthk",
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
        numerator = numerator * correction[..., None, None] + jnp.einsum(
            "nthk,pnkd->nthpd",
            exponentials.astype(jnp.bfloat16),
            kv,
            preferred_element_type=jnp.float32,
        )
        result = (
            new_max.reshape(tokens, heads),
            denominator.reshape(tokens, heads),
            merge_halves(numerator.reshape(tokens, heads, dim)),
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


def paged_attention_jax(
    q: jax.Array,
    swa_pool: jax.Array,
    main_pool: jax.Array,
    swa_records: jax.Array,
    main_records: jax.Array,
    sink: jax.Array,
    scale: float,
) -> jax.Array:
    # SWA widths in the released model are multiples of 64, preserving the
    # reference's accumulation boundaries when compressed rows follow.
    maximum, denominator, numerator = _attend(
        q,
        main_pool,
        main_records,
        _attend_window(q, swa_pool, swa_records, _attention_state(q), scale),
        scale,
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


def gather_sparse_kv_chunk_with_union_fallback(
    cache_kv_nope: jax.Array,
    cache_kv_rope: jax.Array,
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
) -> tuple[jax.Array, jax.Array]:
    """Gather chunk KV via union table, falling back to direct gather on overflow.

    When ``num_unique_chunk <= union_cap``, gathers only the ``union_cap`` unique
    physical rows and remaps via ``inv_idx_chunk``. If ``num_unique_chunk >
    union_cap``, ``jax.lax.cond`` falls back to direct ``[chunk_size, topk]``
    gather so that ``num_unique > union_cap`` never truncates KV rows.
    """
    total_num_pages, page_size, nope_lanes = cache_kv_nope.shape
    _, _, rope_lanes = cache_kv_rope.shape
    chunk_size = chunk_records.shape[0]
    union_cap = union_phys_chunk.shape[0]
    has_valid = batch_end > start_pos

    def _union_branch(_: None) -> tuple[jax.Array, jax.Array]:
        if jax.default_backend() == "tpu":
            from vllm_torchtpu.kernels.deepseek_v4.core_attention import (
                csa_gather,
                sparse_mla,
            )

            num_valid_union = jnp.where(
                has_valid,
                jnp.minimum(union_cap, sparse_mla.align_to(num_unique_chunk, topk)),
                0,
            )
            u_nope, u_rope = csa_gather.csa_gather(
                cache_kv_nope,
                cache_kv_rope,
                union_phys_chunk,
                num_valid_indices=num_valid_union,
                top_k=topk,
            )
        else:
            flat_nope = cache_kv_nope.reshape(total_num_pages * page_size, nope_lanes)
            flat_rope_words = cache_kv_rope.reshape(
                total_num_pages * page_size, rope_lanes // 4
            )
            u_nope = flat_nope[union_phys_chunk]
            u_rope = repack_chunk_rope_words(
                flat_rope_words[union_phys_chunk].reshape(
                    union_cap // topk, topk, rope_lanes // 4
                )
            ).reshape(union_cap // 4, rope_lanes)
        return remap_union_kv_to_chunk(u_nope, u_rope, inv_idx_chunk, topk)

    def _direct_branch(_: None) -> tuple[jax.Array, jax.Array]:
        seq_page_ids = chunk_records // page_size
        token_offset = chunk_records % page_size
        if jax.default_backend() == "tpu":
            from vllm_torchtpu.kernels.deepseek_v4.core_attention import (
                csa_gather,
                sparse_mla,
            )

            page_ids = sparse_mla.gather_page_ids(
                page_indices,
                seq_page_ids,
                chunk_seq_ids,
                max_num_seqs,
                block_tokens=sparse_mla._largest_divisor(chunk_size, 8),
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

        if jax.default_backend() == "tpu":
            from vllm_torchtpu.kernels.deepseek_v4.core_attention import csa_gather

            g_nope, g_rope = csa_gather.csa_gather(
                cache_kv_nope,
                cache_kv_rope,
                direct_indices,
                num_valid_indices=(batch_end - start_pos) * topk,
                top_k=topk,
            )
            return (
                g_nope.reshape(chunk_size, topk, -1),
                g_rope.reshape(chunk_size, topk // 4, -1),
            )
        flat_nope = cache_kv_nope.reshape(total_num_pages * page_size, nope_lanes)
        flat_rope_words = cache_kv_rope.reshape(
            total_num_pages * page_size, rope_lanes // 4
        )
        g_nope = flat_nope[direct_indices].reshape(chunk_size, topk, nope_lanes)
        g_rope = repack_chunk_rope_words(
            flat_rope_words[direct_indices].reshape(chunk_size, topk, rope_lanes // 4)
        )
        return g_nope, g_rope

    return jax.lax.cond(
        num_unique_chunk <= union_cap,
        _union_branch,
        _direct_branch,
        operand=None,
    )


def sparse_ragged_paged_attention_union_gather(
    q: jax.Array,
    cache_kv_nope: jax.Array,
    cache_kv_rope: jax.Array,
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
    attention_kernel_batch_size: int = 16,
    union_cap: int = 4096,
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

    total_num_pages, page_size, _ = cache_kv_nope.shape
    _, num_q_heads, _ = q.shape
    topk = topk_indices.shape[-1]
    kv_lens = jnp.sum(topk_indices != -1, axis=-1)
    union_cap_aligned = max(topk, sparse_mla.align_to(union_cap, topk))
    max_num_seqs = cu_q_lens.shape[0] - 1
    tokens_per_seq = cu_q_lens[1:] - cu_q_lens[:-1]
    seq_ids_segment = jnp.repeat(
        jnp.arange(max_num_seqs, dtype=jnp.int32),
        tokens_per_seq,
        total_repeat_length=q.shape[0],
    )

    union_phys, inv_idx_all, num_unique_all = compute_chunk_union_gather_indices(
        topk_indices,
        page_indices,
        cu_q_lens,
        page_size=page_size,
        total_num_pages=total_num_pages,
        chunk_size=gather_and_attention_chunk_size,
        union_cap=union_cap_aligned,
    )

    num_chunks = sparse_mla.cdiv(q.shape[0], gather_and_attention_chunk_size)
    valid_total = cu_q_lens[distribution[2]]

    for i in range(num_chunks):
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
            )
        )

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
    return sparse_mla.prepare_outputs(q, actual_num_q_heads, actual_head_dim)


def sparse_fp8_attention_jax(
    q: jax.Array,
    swa_pool: jax.Array,
    main_pool: jax.Array,
    rope_pool: jax.Array,
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
    count = main_pool.shape[0] // page_size
    compacted = (
        _pad_compacted_records(main_records)
        if precompacted
        else compact_records(main_records)
    )
    if union_gather:
        return sparse_ragged_paged_attention_union_gather(
            q,
            main_pool.reshape(count, page_size, main_pool.shape[1]),
            rope_pool.reshape(count, page_size // 4, rope_pool.shape[1]),
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
        )
    return sparse_ragged_paged_attention(
        q,
        main_pool.reshape(count, page_size, main_pool.shape[1]),
        rope_pool.reshape(count, page_size // 4, rope_pool.shape[1]),
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


def initialize_paged_attention():
    global _attention_op, _fp8_attention_op
    if _attention_op is not None:
        return
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas
    from vllm.distributed import get_tp_group

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_vllm_model_wrapper_context,
    )

    _attention_op = pallas.jax_op(
        "pallas::dsv41v2_paged_bf16_attention",
        paged_attention_jax,
        mesh=get_vllm_model_wrapper_context().mesh,
        input_partition_specs=(P(), P(), P(), P(), P(), P(), None),
    )
    _attention_op.register_fake(lambda q, *args: q.new_empty(q.shape))
    mesh = get_vllm_model_wrapper_context().mesh
    try:
        tp_group = get_tp_group().device_group
        tp_size = tp_group.size() if tp_group is not None else 1
    except Exception:
        tp_group = None
        tp_size = 1
    if tp_size > 1 and tp_group is not None:
        tp_mesh = pallas.get_pallas_mesh(axis_names=("tp",), process_group=tp_group)
    else:
        tp_mesh = mesh
    in_specs = (P(),) * 11
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
    ) -> jax.Array:
        tokens = q.shape[0]
        use_token_parallel = (
            token_parallel
            and tp_size > 1
            and tokens >= 128 * tp_size
            and tokens % (128 * tp_size) == 0
        )
        if use_token_parallel:

            def _token_parallel_body(
                q_l, s_p, m_p, r_p, s_r, m_r, pgs, q_loc, _snk, f_snk, r_id
            ):
                q_tok = jax.lax.all_to_all(
                    q_l, axis_name="tp", split_axis=0, concat_axis=1, tiled=True
                )
                s_r_tok, m_r_tok, q_loc_tok = slice_token_parallel_attention_inputs(
                    s_r, m_r, q_loc, r_id, tp_size, tokens
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
                )
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
            )

        # Mosaic calls require explicit manual partitioning on a multi-device
        # mesh; each device attends over its own replicated cache.
        return jax.shard_map(
            lambda q_l, s_p, m_p, r_p, s_r, m_r, pgs, q_loc, snk, _f_snk, _r_id: (
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
        )

    _fp8_attention_op = pallas.jax_op(
        "pallas::dsv41v2_sparse_fp8_attention",
        fp8_attention,
        mesh=tp_mesh,
        input_partition_specs=in_specs + (None,) * 7,
    )
    _fp8_attention_op.register_fake(lambda q, *args: q.new_empty(q.shape))


def paged_attention(q, swa, main, swa_records, main_records, sink, scale):
    assert _attention_op is not None
    return _attention_op(
        q, swa.raw, main.raw, swa_records.int(), main_records.int(), sink, scale
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
    if pages is None:
        # Identity table: one request whose single page is the whole pool.
        page_size = main.raw.shape[0]
        pages = main_records.new_zeros((1, 1))
        starts = main_records.new_tensor([0, q.shape[0]])
    if full_sink is None:
        full_sink = sink
    if tp_rank is None:
        tp_rank = main_records.new_zeros((1,), dtype=torch.int32)
    else:
        tp_rank = tp_rank.reshape(1).int()
    return _fp8_attention_op(
        q,
        swa.raw,
        main.raw,
        main.rope,
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
    )


def paged_index_scores_jax(
    q: jax.Array,
    pool: jax.Array,
    records: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    requests: jax.Array | None = None,
) -> jax.Array:
    """Index scores ``[tokens, width]``.

    ``records`` is ``[tokens, width]``, or ``[requests, width]`` when
    ``requests`` maps each token to its row: every token of a request then
    shares one paged gather of the key tile instead of reading it again.
    """
    tokens, dim = q.shape[0], q.shape[-1]
    if requests is not None and tokens <= records.shape[0]:
        # Decode: request rows are padded, so sharing gathers only adds reads.
        records, requests = jnp.take(records, requests, axis=0), None
    width = records.shape[1]
    # Keep about 16K keys per iteration: small (decode) batches otherwise run
    # many latency-bound tiny gathers. Prefill batches use 512: with 128 the
    # loop is iteration-latency bound (2.3x slower at T=512 over 16.9K rows).
    budget = min(max(512, min(2048, 16384 // max(tokens, 1))), -(-width // 128) * 128)
    tile = 1 << (budget.bit_length() - 1)
    records = jnp.pad(records, ((0, 0), (0, -width % tile)), constant_values=-1)
    scores = jnp.full((tokens, records.shape[1]), -jnp.inf, jnp.float32)
    q = split_halves(q)  # keys are read in the split element order
    weights = weights.astype(jnp.float32)[:, :, None]

    def head_sum(logits):
        # Upstream quantized MQA logits and the weighted head reduction are
        # FP32. BF16 cache storage is not a request to round the dot product.
        return (jnp.maximum(logits, 0) * weights).sum(1)

    if requests is not None:
        # One request row at a time, each bounded by its own longest token:
        # all tokens score the row's shared key tile, and only that row's
        # tokens keep the result. Rows without tokens (padding) run no tiles,
        # so work scales with active requests, not the padded row count.
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
                keys = gather_cache_values(pool, addresses, dim, split=True)
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
                (row_lengths[row] + tile - 1) // tile, records.shape[1] // tile
            )
            return jax.lax.fori_loop(0, trips, step, result)

        return jax.lax.fori_loop(0, records.shape[0], row_step, scores)[:, :width]

    def step(index, result):
        # Read only the tile consumed by this iteration. Gathering before
        # the loop materializes [tokens, max_context / ratio, dim] even when
        # lengths restrict execution to a single tile (or no tiles).
        addresses = jax.lax.dynamic_slice_in_dim(records, index * tile, tile, axis=1)
        keys = gather_cache_values(
            pool, addresses.reshape(-1), dim, split=True
        ).reshape(records.shape[0], tile, dim)
        logits = jnp.einsum("thd,tcd->thc", q, keys, preferred_element_type=jnp.float32)
        return jax.lax.dynamic_update_slice_in_dim(
            result, head_sum(logits), index * tile, axis=1
        )

    return jax.lax.fori_loop(
        0,
        jnp.minimum(
            (jnp.maximum(lengths.max(), 0) + tile - 1) // tile, records.shape[1] // tile
        ),
        step,
        scores,
    )[:, :width]


_indexer_op = None


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
    """
    rows, width = scores.shape
    if (
        unordered
        and lengths is not None
        and width >= 2048
        and width % 16 == 0
        and k % 16 == 0
    ):
        # SparseCore top-k selects 16-row tiles; padding rows select nothing.
        padding = -rows % 16
        scores = jnp.pad(scores, ((0, padding), (0, 0)), constant_values=-jnp.inf)
        lengths = jnp.pad(jnp.minimum(lengths, width), (0, padding))
        return _sparsecore_top_k(scores, k, lengths)[:rows]

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
            values, ids = jax.lax.top_k(block_scores, candidates.shape[1])
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
    indices = _visible_top_k(
        scores, min(topk, width), lengths.max(), lengths, unordered
    )
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
) -> tuple[jax.Array, jax.Array]:
    """``records`` is ``[requests, width]``; ``requests`` maps tokens to rows.

    ``batch_max``: see `select_indices_jax`.
    """

    def full_context(_):
        scores = paged_index_scores_jax(q, pool, records, weights, lengths, requests)
        scores = jnp.where(jnp.take(records >= 0, requests, axis=0), scores, -jnp.inf)
        return select_indices_jax(
            scores,
            lengths,
            candidates,
            topk,
            block_size,
            candidate_write,
            candidate_filter,
            unordered,
            batch_max,
        )

    width = records.shape[1]
    if not candidate_filter or candidates.shape[1] * block_size >= width:
        return full_context(None)

    def candidate_context(_):
        # Candidates differ per token, so this path reads per-token records.
        token_records = jnp.take(records, requests, axis=0)
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
        compact = jnp.take_along_axis(
            token_records, jnp.minimum(logical, width - 1), axis=1
        )
        compact = jnp.where(valid, compact, -1)
        # This bounds both key reads and QK work by the candidate pool size;
        # zero-length padding rows do not trigger key reads.
        compact_lengths = jnp.where(lengths > 0, compact.shape[1], 0)
        scores = paged_index_scores_jax(q, pool, compact, weights, compact_lengths)
        scores = jnp.where(compact >= 0, scores, -jnp.inf)
        output_width = min(topk, width)
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
    # key reads are at most the per-token candidate key reads (e.g. prefill,
    # where all tokens of a request share one key tile in full_context).
    longest = lengths.max() if batch_max is None else batch_max.max()
    row_lengths = (
        jnp.zeros((records.shape[0],), jnp.int32)
        .at[requests]
        .max(jnp.maximum(lengths, 0))
    )
    cand_keys = jnp.sum(lengths > 0) * (candidates.shape[1] * block_size)
    return jax.lax.cond(
        (longest <= topk) | (row_lengths.sum() <= cand_keys),
        full_context,
        candidate_context,
        None,
    )


def write_index_keys(
    pool: jax.Array, write_records: jax.Array, keys: jax.Array
) -> jax.Array:
    """Publish index records through the same cache writer as KV and state."""
    if keys.dtype != jnp.bfloat16:
        raise ValueError("Index keys must be BF16")
    if keys.ndim != 2 or write_records.shape != (keys.shape[0],):
        raise ValueError("One physical record index per key is required")
    return scatter_cache_values(pool, write_records, keys)


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
) -> tuple[jax.Array, jax.Array, jax.Array]:
    updated = write_index_keys(pool, write_records, keys)
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
    )
    return updated, selected, published


def initialize_indexer():
    global _indexer_op
    if _indexer_op is not None:
        return
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_vllm_model_wrapper_context,
    )

    _indexer_op = pallas.jax_op(
        "pallas::dsv41v2_exact_indexer",
        update_and_select_jax,
        donate_argnums=(1,),
        mesh=get_vllm_model_wrapper_context().mesh,
        input_partition_specs=(P(),) * 10 + (None,) * 5,
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
    ):
        return (
            torch.empty_like(pool),
            records.new_empty(
                (q.shape[0], min(topk, records.shape[1])), dtype=torch.int32
            ),
            torch.empty_like(candidates),
        )

    _indexer_op.register_fake(fake)


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
):
    """``records`` is ``[requests, width]``; ``requests`` maps tokens to rows.

    ``batch_max`` (default ``lengths.max()``) is the longest length of the
    whole batch when the queries are only part of it.
    """
    assert _indexer_op is not None
    if batch_max is None:
        batch_max = lengths.max()
    return _indexer_op(
        q,
        cache.raw,
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
    )
