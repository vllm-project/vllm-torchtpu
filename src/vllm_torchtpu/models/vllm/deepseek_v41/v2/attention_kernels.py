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


def compact_records(records: jax.Array) -> jax.Array:
    """Pad to 128 columns and move -1 to the tail: sparse_mla's valid prefix."""
    width = records.shape[1]
    padded = max(128, -(-width // 128) * 128)
    records = jnp.pad(records, ((0, 0), (0, padded - width)), constant_values=-1)
    limit = jnp.iinfo(jnp.int32).max
    records = jnp.sort(jnp.where(records >= 0, records, limit), axis=-1)
    return jnp.where(records == limit, -1, records)


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
) -> jax.Array:
    """BF16 SWA partial merged into V4's sparse MLA over FP8 NoPE + RoPE rows.

    ``main_records`` index rows through ``pages`` ``[requests, pages_per_seq]``:
    row ``r`` of request ``s`` is raw row ``pages[s, r // page_size] *
    page_size + r % page_size``; the RoPE companion holds raw row i's 32 words
    at i. With one request, one page of every row and ``query_start_loc``
    ``[0, tokens]``, records are raw rows.

    ``max_requests`` bounds requests per batch (see `_attend_window`).
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
    return sparse_ragged_paged_attention(
        q,
        main_pool.reshape(count, page_size, main_pool.shape[1]),
        rope_pool.reshape(count, page_size // 4, rope_pool.shape[1]),
        compact_records(main_records),
        pages.reshape(-1),
        query_start_loc,
        jnp.array([0, 0, pages.shape[0]], jnp.int32),
        sink.astype(jnp.float32),
        numerator.astype(q.dtype),
        denominator,
        maximum,
        sm_scale=scale,
    )


def initialize_paged_attention():
    global _attention_op, _fp8_attention_op
    if _attention_op is not None:
        return
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

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
        scale: float,
        page_size: int,
        max_requests: int | None,
    ) -> jax.Array:
        # Mosaic calls require explicit manual partitioning on a multi-device
        # mesh; each device attends over its own replicated cache.
        return jax.shard_map(
            functools.partial(
                sparse_fp8_attention_jax,
                scale=scale,
                page_size=page_size,
                max_requests=max_requests,
            ),
            mesh=mesh,
            in_specs=(P(),) * 9,
            out_specs=P(),
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
        )

    _fp8_attention_op = pallas.jax_op(
        "pallas::dsv41v2_sparse_fp8_attention",
        fp8_attention,
        mesh=mesh,
        input_partition_specs=(P(),) * 9 + (None, None, None),
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
):
    """``main_records`` are raw rows, or rows paged by ``pages`` (main.pages)."""
    assert _fp8_attention_op is not None and main.rope is not None
    page_size = main.shape[1]
    if pages is None:
        # Identity table: one request whose single page is the whole pool.
        page_size = main.raw.shape[0]
        pages = main_records.new_zeros((1, 1))
        starts = main_records.new_tensor([0, q.shape[0]])
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
        scale,
        page_size,
        max_requests,
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
    width = scores.shape[1]
    if (
        unordered
        and lengths is not None
        and scores.shape[0] >= 16
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
            values, ids = jax.lax.top_k(block_scores, candidates.shape[1])
            published = jnp.where(values > -jnp.inf, ids, -1).astype(jnp.int32)
        else:
            # One extra column absorbs invalid candidate IDs; no 3-D membership
            # tensor and no dependence on the order of candidate IDs.
            valid = (candidates >= 0) & (candidates < blocks)
            ids = jnp.where(valid, candidates, blocks)
            membership = jnp.zeros((count, blocks + 1), jnp.bool_)
            membership = membership.at[jnp.arange(count)[:, None], ids].set(True)
            allowed = jnp.take_along_axis(membership, logical // block_size, axis=1)
            # Match the upstream batch-wide short-context bypass.
            scores = jnp.where(allowed | (batch_max <= topk), scores, -jnp.inf)
    # Ascending: preserves V4.1's logical, hence attention accumulation, order.
    indices = _visible_top_k(
        scores, min(topk, width), lengths.max(), lengths, unordered
    )
    values = jnp.take_along_axis(scores, jnp.maximum(indices, 0), axis=1)
    valid = (indices >= 0) & (values > -jnp.inf)
    return jnp.where(valid, indices, -1).astype(jnp.int32), published


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
        # This bounds both key reads and QK work by the candidate pool size.
        compact_lengths = jnp.full_like(lengths, compact.shape[1])
        scores = paged_index_scores_jax(q, pool, compact, weights, compact_lengths)
        scores = jnp.where(compact >= 0, scores, -jnp.inf)
        output_width = min(topk, width)
        padding = max(0, output_width - scores.shape[1])
        scores = jnp.pad(scores, ((0, 0), (0, padding)), constant_values=-jnp.inf)
        logical = jnp.pad(logical, ((0, 0), (0, padding)), constant_values=width)
        values, indices = jax.lax.top_k(scores, output_width)
        selected = jnp.take_along_axis(logical, indices, axis=1)
        selected_valid = values > -jnp.inf
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

    # Preserve the reference's batch-wide short-context bypass. That branch
    # scans at most topk keys; long contexts never enter the full-history path.
    longest = lengths.max() if batch_max is None else batch_max.max()
    return jax.lax.cond(longest <= topk, full_context, candidate_context, None)


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
