"""Paged attention and stage-local KV gathering."""

import contextlib
import dataclasses
from collections.abc import Iterator
from typing import Any, cast

import jax
import jax.numpy as jnp
import torch

from .cache import (
    _has_hi_bank,
    gather_cache_pieces,
    merge_pieces,
    split_pieces,
)
from .indexer_select import _compact_selected

_attention_op = None
_attention_2bank_op = None
_gather_bf16_op = None


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


def _swa_attention(q, pool, records, sink, scale, query_group_size=1):
    """Fuse pure SWA without materializing split Q or merged O layouts."""
    from vllm_torchtpu.kernels.mla.swa_attention import attend_swa

    def direct(q, records):
        kv = gather_cache_pieces(pool, records.reshape(-1), q.shape[-1])
        kv = kv.reshape(4, *records.shape, 128)
        return attend_swa(q, kv, (records >= 0)[:, None, :], sink, scale)

    tokens, width = records.shape

    def fallback(_):
        if tokens < 128 or tokens % 64 or tokens % 6 == 0 or records.shape[1] != 128:
            return direct(q, records)
        union, mask, uncovered = _window_union(records, 64)

        def shared(_):
            # A 192-row union is padded to one 256-row MXU RHS tile.
            padded = jnp.pad(
                union, ((0, 0), (0, 256 - union.shape[1])), constant_values=-1
            )
            kv = gather_cache_pieces(pool, padded.reshape(-1), q.shape[-1])
            kv = kv.reshape(4, *padded.shape, 128)
            padded_mask = jnp.pad(mask, ((0, 0), (0, 0), (0, 256 - union.shape[1])))
            result = attend_swa(q, kv, padded_mask, sink, scale)

            def patch(result):
                index = jnp.nonzero(uncovered.reshape(-1), size=64, fill_value=tokens)[
                    0
                ]
                source = jnp.minimum(index, tokens - 1)
                sub_records = jnp.where((index < tokens)[:, None], records[source], -1)
                return result.at[index].set(direct(q[source], sub_records), mode="drop")

            return jax.lax.cond(uncovered.any(), patch, lambda x: x, result)

        return jax.lax.cond(
            uncovered.sum() > 64, lambda _: direct(q, records), shared, None
        )

    queries = query_group_size
    # The existing kernel holds at most 256 keys. Share only when padding
    # still reduces gathering, and prove overlap instead of trusting the hint.
    if (
        queries > 1
        and tokens > 0
        and tokens % queries == 0
        and width + queries - 1 <= 256 < queries * width
    ):
        rows = records.reshape(-1, queries, width)
        union = jnp.concatenate((rows[:, 0], rows[:, -1, -(queries - 1) :]), axis=1)
        offsets = jnp.arange(queries)[:, None] + jnp.arange(width)[None, :]
        valid = jnp.all(rows == union[:, offsets])

        def shared_queries(_):
            padded = jnp.pad(
                union, ((0, 0), (0, 256 - union.shape[1])), constant_values=-1
            )
            kv = gather_cache_pieces(pool, padded.reshape(-1), q.shape[-1])
            kv = kv.reshape(4, *padded.shape, 128)
            column = jnp.arange(padded.shape[1])[None, None, :]
            slot = jnp.arange(queries)[None, :, None]
            mask = (
                (column >= slot) & (column < width + slot) & (padded[:, None, :] >= 0)
            )
            return attend_swa(q, kv, mask, sink, scale)

        return jax.lax.cond(valid, shared_queries, fallback, None)

    return fallback(None)


def _general_paged_attention_jax(
    q: jax.Array,
    swa_pool: jax.Array,
    main_pool: jax.Array,
    swa_records: jax.Array,
    main_records: jax.Array,
    sink: jax.Array,
    scale: float,
    gathered: jax.Array | None = None,
    query_group_size: int = 1,
) -> jax.Array:
    from vllm_torchtpu.kernels.mla.swa_attention import supports_swa

    if main_records.shape[1] == 0 and supports_swa(q, swa_records):
        return _swa_attention(q, swa_pool, swa_records, sink, scale, query_group_size)
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
    query_group_size: int = 1,
    row_valid: jax.Array | None = None,
) -> jax.Array:
    """Attend physical cache records; query_group_size is a checked sharing hint.

    ``row_valid`` (``[tokens]`` int32, optional) zeroes the output rows it marks
    0, the caller's mask of invalid SWA rows. The shared DSpark draft path
    applies it in the kernel's write-back instead of a separate pass over the
    ``[240, 64, 512]`` output (~13 us per draft layer on one v7x chip).
    """

    def general(_):
        out = _general_paged_attention_jax(
            q,
            swa_pool,
            main_pool,
            swa_records,
            main_records,
            sink,
            scale,
            gathered,
            query_group_size,
        )
        if row_valid is None:
            return out
        return jnp.where((row_valid != 0)[:, None, None], out, jnp.zeros((), out.dtype))

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
            lambda _: _shared_draft_attention(
                q, swa_pool, swa_records, sink, row_valid
            ),
            general,
            None,
        )
    return general(None)


def _shared_draft_attention(q, pool, records, sink, row_valid=None):
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
        mask, pieces, kv, jnp.tile(sink, 5)[:, None], row_valid
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


def initialize_paged_attention():
    global _attention_op, _attention_2bank_op, _gather_bf16_op
    if _attention_op is not None:
        return
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_pallas_op_mesh,
    )

    # jax_op's type stub omits PartitionSpec and None for static arguments.
    _attention_op = pallas.jax_op(
        "pallas::dsv41v2_paged_bf16_attention",
        paged_attention_jax,
        mesh=get_pallas_op_mesh(),
        input_partition_specs=cast(
            tuple[tuple[str, ...], ...], cast(Any, (P(),) * 6 + (None, P(), None, P()))
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
        query_group_size: int = 1,
        row_valid: jax.Array | None = None,
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
            query_group_size,
            row_valid,
        )

    _attention_2bank_op = pallas.jax_op(
        "pallas::dsv41v2_paged_bf16_attention_2bank",
        paged_attention_2bank,
        mesh=get_pallas_op_mesh(),
        input_partition_specs=cast(
            tuple[tuple[str, ...], ...], cast(Any, (P(),) * 8 + (None, P(), None, P()))
        ),
    )
    _attention_2bank_op.register_fake(lambda q, *args: q.new_empty(q.shape))
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


def paged_attention(
    q,
    swa,
    main,
    swa_records,
    main_records,
    sink,
    scale,
    query_group_size=1,
    row_valid=None,
):
    """``row_valid`` (``[tokens]`` bool, optional): rows marked False come back
    zeroed, so the caller needs no separate mask of the output."""
    if row_valid is not None:
        row_valid = row_valid.int()
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
            query_group_size,
            row_valid,
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
        query_group_size,
        row_valid,
    )
