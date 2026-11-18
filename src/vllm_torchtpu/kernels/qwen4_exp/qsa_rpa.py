# SPDX-License-Identifier: Apache-2.0
"""QSA with native HEAD_ALONG_SUBLANE storage and existing batched RPA decode.

Scheduler pages remain intact. Selected compression groups become virtual RPA
pages by a view; the returned cache is threaded through donation, never dropped.
Prefill reuses KV tiles through block-sparse attention; decode uses batched RPA,
or the fused sparse GQA decode kernel under VLLM_TPU_QSA_FUSED_SPARSE_DECODE.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.kernels.qwen4_exp.decode_indexer import select_decode_blocks
from vllm_torchtpu.kernels.qwen4_exp.qsa import (
    _token_metadata,
    _update_indexer,
    bf16,
    produce,
    quantize_kv,
)
from vllm_torchtpu.kernels.qwen4_exp.request_types import (
    short_request_prefix_tokens,
)


def pack_rows(key, value, *, fp8, k_scale=1.0, v_scale=1.0):
    pad = ((0, 0), (0, 0), (0, (-key.shape[-1]) % 128))
    key, value = jnp.pad(key, pad), jnp.pad(value, pad)
    if fp8:
        key, value = quantize_kv(key, k_scale), quantize_kv(value, v_scale)
    packing = 4 if fp8 else 2
    joined = jnp.stack((key, value), axis=-2).reshape(len(key), -1, key.shape[-1])
    joined = jnp.pad(joined, ((0, 0), (0, (-joined.shape[1]) % packing), (0, 0)))
    joined = joined.reshape(len(key), -1, packing, key.shape[-1])
    return joined


def dcp_interleave(dcp, page_size: int) -> int:
    """Tokens per ownership unit of a main-KV DCP ``dcp = (rank, D[, I])``:
    ``I`` when given and nonzero, else one page (the decode's layout)."""
    return dcp[2] if len(dcp) > 2 and dcp[2] else page_size


def dcp_owner_row(t, page_size: int, world: int, interleave: int):
    """Owner rank and in-page row of absolute position ``t`` under main-KV DCP.

    A block-table entry covers ``world * page_size`` tokens; units of
    ``interleave`` tokens alternate over the ranks, and each rank packs its
    units of the entry into the entry's page in order. ``interleave ==
    page_size`` gives ``((t // page) % world, t % page)``.
    """
    offset = t % (world * page_size)
    owner = (offset // interleave) % world
    row = offset // (interleave * world) * interleave + offset % interleave
    return owner, row


def virtual_page_metadata(
    blocks, lengths, table, *, ratio=4, page_size=128, dcp=None
):
    """Map selected logical groups to physical micro-pages; no KV gathering.

    ``dcp = (rank, D)`` (main-KV decode context parallel, ``rank`` an int32[1]
    operand): a table entry covers ``D * page_size`` tokens and this rank holds
    rows ``t % page_size`` of the tokens with ``(t // page_size) % D == rank``.
    Only those groups are kept, ascending at the front of each row, then the
    partial tail group when this rank owns it; ``kv_lens`` counts owned tokens.
    """
    if dcp is not None:
        return _dcp_page_metadata(
            blocks, lengths, table, ratio=ratio, page_size=page_size, dcp=dcp
        )
    batch, budget = blocks.shape
    complete = lengths // ratio
    count = jnp.minimum(complete, budget)
    columns = jnp.arange(budget + 1)[None, :]
    groups = jnp.concatenate((blocks, jnp.zeros((batch, 1), jnp.int32)), axis=1)
    groups = jnp.where(columns == count[:, None], complete[:, None], groups)
    valid = (columns < count[:, None]) | (
        (columns == count[:, None]) & ((lengths % ratio)[:, None] > 0)
    )
    safe = jnp.where(valid, groups, 0)
    # Reuse sparse MLA's vector lookup for small page tables; the generic
    # gather lowers to a slower indirect lookup on this path.
    from vllm_torchtpu.kernels.mla.sparse.kernel import gather_page_ids

    physical = gather_page_ids(
        table.reshape(-1),
        safe // (page_size // ratio),
        jnp.arange(batch, dtype=jnp.int32),
        batch,
    )
    pages = jnp.where(
        valid, physical * (page_size // ratio) + safe % (page_size // ratio), 0
    )
    kv_lens = count * ratio + lengths % ratio
    cu_q_lens = jnp.concatenate(
        (jnp.zeros((1,), jnp.int32), jnp.cumsum((lengths > 0).astype(jnp.int32)))
    )
    return pages, kv_lens, cu_q_lens, jnp.full((3,), batch, jnp.int32)


def _dcp_page_metadata(blocks, lengths, table, *, ratio, page_size, dcp):
    """:func:`virtual_page_metadata` for this DCP rank's owned groups; rows
    with no owned token get ``kv_lens`` 0 and no query (see ``cu_q_lens``).

    A true ``dcp[3]`` (``pooled_qsa.compressed_dcp_own_blocks``): ``blocks``
    hold only selected groups of this rank, -1 elsewhere, at any column, and
    a row's selection is its entries below its completed groups (rows the RPA
    batch pads carry zeros and length 0); otherwise it is its first
    ``min(complete, budget)`` columns."""
    rank, world = dcp[0], dcp[1]
    own_list = len(dcp) > 3 and dcp[3]
    unit_groups = dcp_interleave(dcp, page_size) // ratio
    rank = jnp.asarray(rank, jnp.int32).reshape(-1)[0]
    batch, budget = blocks.shape
    page_groups = page_size // ratio
    big = jnp.iinfo(jnp.int32).max
    complete = lengths // ratio
    count = jnp.minimum(complete, budget)
    columns = jnp.arange(budget)[None, :]
    if own_list:
        selected = (blocks >= 0) & (blocks < complete[:, None])
    else:
        selected = columns < count[:, None]
    owned = selected & ((blocks // unit_groups) % world == rank)
    # Owned groups ascending at the front, the rest after them.
    groups = jax.lax.sort(jnp.where(owned, blocks, big), dimension=1, is_stable=False)
    owned_count = jnp.sum(owned, axis=1, dtype=jnp.int32)
    tail = ((lengths % ratio) > 0) & ((complete // unit_groups) % world == rank)
    columns = jnp.arange(budget + 1)[None, :]
    groups = jnp.concatenate((groups, jnp.zeros((batch, 1), jnp.int32)), axis=1)
    groups = jnp.where(columns == owned_count[:, None], complete[:, None], groups)
    valid = (columns < owned_count[:, None]) | (
        (columns == owned_count[:, None]) & tail[:, None]
    )
    safe = jnp.where(valid, groups, 0)
    from vllm_torchtpu.kernels.mla.sparse.kernel import gather_page_ids

    # One table entry covers `world` pages' worth of tokens; this rank's half
    # sits in that entry's physical page at row t % page_size.
    physical = gather_page_ids(
        table.reshape(-1),
        safe // (world * page_groups),
        jnp.arange(batch, dtype=jnp.int32),
        batch,
    )
    # The group's row group in the entry's page (dcp_owner_row in group units).
    _, row_group = dcp_owner_row(safe, page_groups, world, unit_groups)
    pages = jnp.where(valid, physical * page_groups + row_group, 0)
    kv_lens = owned_count * ratio + jnp.where(tail, lengths % ratio, 0)
    cu_q_lens = jnp.concatenate(
        (jnp.zeros((1,), jnp.int32), jnp.cumsum((kv_lens > 0).astype(jnp.int32)))
    )
    return pages, kv_lens, cu_q_lens, jnp.full((3,), batch, jnp.int32)


def _rpa_attention_batch(
    q,
    gate,
    blocks,
    lengths,
    table,
    cache,
    *,
    kv_heads,
    ratio,
    bkv_size=256,
    batch_size=8,
    k_scale=1.0,
    v_scale=1.0,
    kv_head_offset=0,
    decode_only=False,
    lean_fetch=False,
    slot_read=False,
    compact=False,
    fast_fetch=False,
    skip_empty=0,
    dcp=None,
    return_lse=False,
    read_only=False,
):
    """With ``return_lse`` returns ``(out, lse, cache)``: ``out`` f32 normalized
    over the attended keys and not gated, ``lse`` [rows, heads] (natural log);
    rows with no key get 0 and -inf. ``dcp``: see ``virtual_page_metadata``.
    ``read_only``: RPA neither writes nor returns the cache (the returned
    cache is the argument), for callers inside a conditional."""
    from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

    kv_slot = None
    if slot_read and slot_readable(cache, kv_heads, kv_head_offset):
        # Read this layer's pair of the packed row directly: one KV head, no
        # zero queries for the other layer's head, one pair unpacked.
        kv_slot, kv_head_offset = kv_head_offset, 0
    pages, kv_lens, cu_q_lens, distribution = virtual_page_metadata(
        blocks,
        lengths,
        table,
        ratio=ratio,
        page_size=cache.shape[1],
        **({"dcp": dcp} if dcp is not None else {}),
    )
    # A row has a query when it attends any key: every live row, or under DCP
    # every row this rank owns a key of.
    live = kv_lens > 0 if dcp is not None else lengths > 0
    # Queries are packed in request order; zero-length requests have no query.
    order = jnp.argsort(~live if dcp is not None else lengths == 0, stable=True)
    query = jnp.pad(
        q[order].astype(jnp.float32),
        ((0, 0), (0, 0), (0, cache.shape[-1] - q.shape[-1])),
    )
    virtual_cache = cache.reshape(-1, ratio, *cache.shape[2:])
    # Paired pages store a second layer's KV head after the first. RPA reads
    # a row's leading heads; the cache then presents kv_head_offset + kv_heads
    # heads and zero queries attend the heads ahead of this layer's.
    skipped_heads = kv_head_offset * (q.shape[1] // kv_heads)
    if kv_head_offset:
        if (kv_head_offset + kv_heads) * 2 > cache.shape[2] * cache.shape[3]:
            raise ValueError("KV head offset exceeds the cache row's head slots")
        query = jnp.pad(query, ((0, 0), (skipped_heads, 0), (0, 0)))
    dummy_k = jnp.zeros(
        (q.shape[0], kv_heads + kv_head_offset, query.shape[-1]), cache.dtype
    )
    decode = configs.BlockSizes(
        bq_sz=1, bq_c_sz=1, bkv_sz=bkv_size, batch_size=batch_size
    )
    prefill = configs.BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=bkv_size, batch_size=1)
    result = wrapper.ragged_paged_attention(
        query,
        dummy_k,
        dummy_k,
        virtual_cache,
        kv_lens,
        pages.reshape(-1),
        cu_q_lens,
        distribution,
        sm_scale=q.shape[-1] ** -0.5,
        q_scale=None,
        k_scale=k_scale,
        v_scale=v_scale,
        out_dtype=jnp.dtype(jnp.float32),
        skip_kv_update=True,
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
        decode_block_sizes=decode,
        prefill_block_sizes=prefill,
        # virtual_page_metadata makes every row a decode row.
        decode_only=decode_only,
        lean_kv_fetch=lean_fetch or fast_fetch,
        fast_kv_fetch=fast_fetch,
        kv_slot=kv_slot,
        compact_kv_schedule=compact,
        **({"kv_skip_group": skip_empty} if compact and skip_empty else {}),
        **({"return_lse": True} if return_lse else {}),
        **({"read_only_cache": True} if read_only else {}),
        schedule_in_hbm=envs.VLLM_TPU_RPA_SCHEDULE_HBM,
    )
    out, next_cache = result[0], result[1]
    rows = jnp.minimum(cu_q_lens[:-1], len(q) - 1)
    out = out[rows, :, : q.shape[-1]]
    if kv_head_offset:
        out = out[:, skipped_heads:]
    if return_lse:
        lse = result[2][rows]
        if kv_head_offset:
            lse = lse[:, skipped_heads:]
        out = jnp.where(live[:, None, None], out, 0)
        lse = jnp.where(live[:, None], lse.astype(jnp.float32), -jnp.inf)
        return out, lse, next_cache.reshape(cache.shape)
    out = bf16(
        bf16(out).astype(jnp.float32)
        * bf16(jax.nn.sigmoid(gate.astype(jnp.float32))).astype(jnp.float32)
    )
    out = jnp.where((lengths > 0)[:, None, None], out, 0)
    return out, next_cache.reshape(cache.shape)


def slot_readable(cache, kv_heads, kv_head_offset):
    """RPA can attend head slot ``kv_head_offset`` of a packed row directly:
    one KV head and rows of one packed sublane (FP8 paired pages)."""
    return (
        kv_heads == 1
        and cache.ndim == 5
        and cache.shape[2] == 1
        and kv_head_offset < cache.shape[3] // 2
    )


def rpa_query_batch_size(page_columns):
    # RPA's minimum 128-step, batch8 schedule occupies about 815 KiB
    # of v7's 1 MiB SMEM. A 64-query, 513-column page table fits alongside
    # that schedule; keep larger tables within a 160 KiB scalar-data budget.
    max_queries = min(64, (160 * 1024) // (4 * page_columns))
    return 1 << (max_queries.bit_length() - 1)


def rpa_attention(
    q,
    gate,
    blocks,
    lengths,
    table,
    cache,
    *,
    kv_heads,
    ratio,
    k_scale=1.0,
    v_scale=1.0,
    kv_head_offset=0,
    decode_only=False,
    lean_fetch=False,
    slot_read=False,
    compact=False,
    fast_fetch=False,
    dcp=None,
    return_lse=False,
    read_only=False,
):
    """With ``return_lse`` returns ``(out, lse, cache)`` (see
    ``_rpa_attention_batch``); ``dcp`` attends this DCP rank's owned keys.
    ``read_only``: see ``_rpa_attention_batch``; the chunk loop then takes the
    cache as a loop constant instead of carrying it."""
    batch_size = rpa_query_batch_size(blocks.shape[1] + 1)
    if compact and envs.VLLM_TPU_QSA_DECODE_RPA_MAX_QUERIES:
        # The compact schedule holds one entry per lane-tile instead of one
        # per 4-token page, leaving SMEM for a larger scalar-prefetched page
        # table: a full MTP verify step (24 requests x 4 rows) runs as one
        # RPA call per layer instead of two 64-row chunks.
        batch_size = envs.VLLM_TPU_QSA_DECODE_RPA_MAX_QUERIES
    attend = functools.partial(
        _rpa_attention_batch,
        kv_heads=kv_heads,
        ratio=ratio,
        k_scale=k_scale,
        v_scale=v_scale,
        **({"decode_only": True} if decode_only else {}),
        **({"lean_fetch": True} if lean_fetch else {}),
        **({"slot_read": True} if slot_read else {}),
        **({"fast_fetch": True} if fast_fetch else {}),
        **(
            dict(
                compact=True,
                bkv_size=envs.VLLM_TPU_QSA_DECODE_RPA_BKV,
                batch_size=envs.VLLM_TPU_QSA_DECODE_RPA_LANES,
                **(
                    {"skip_empty": envs.VLLM_TPU_QSA_DECODE_RPA_SKIP_EMPTY}
                    if envs.VLLM_TPU_QSA_DECODE_RPA_SKIP_EMPTY
                    else {}
                ),
            )
            if compact
            else {}
        ),
        **({"kv_head_offset": kv_head_offset} if kv_head_offset else {}),
        **({"dcp": dcp} if dcp is not None else {}),
        **({"return_lse": True} if return_lse else {}),
        **({"read_only": True} if read_only else {}),
    )
    if len(q) <= batch_size:
        return attend(q, gate, blocks, lengths, table, cache)
    padding = (-len(q)) % batch_size

    def split(x):
        x = jnp.pad(x, ((0, padding),) + ((0, 0),) * (x.ndim - 1))
        return x.reshape(-1, batch_size, *x.shape[1:])

    def step(cache, inputs):
        query, gate, blocks, lengths, table = inputs
        *output, cache = attend(query, gate, blocks, lengths, table, cache)
        return cache, tuple(output)

    if read_only:
        base = cache

        def step(carry, inputs):  # noqa: F811
            query, gate, blocks, lengths, table = inputs
            *output, _ = attend(query, gate, blocks, lengths, table, base)
            return carry, tuple(output)

        cache = jnp.zeros((), jnp.int32)

    # Keep the small-step path fully unrolled, but bound the unroll for large
    # prefill steps. Fully expanding a 4K step repeats the RPA body 128 times
    # in each QSA layer and makes whole-model compilation impractical.
    num_chunks = (len(q) + batch_size - 1) // batch_size
    cache, output = jax.lax.scan(
        step,
        cache,
        tuple(map(split, (q, gate, blocks, lengths, table))),
        unroll=min(num_chunks, 4),
    )
    if read_only:
        cache = base
    out = output[0].reshape(-1, *q.shape[1:])[: len(q)]
    if return_lse:
        return out, output[1].reshape(-1, q.shape[1])[: len(q)], cache
    return out, cache


logger = init_logger(__name__)

UNION_REQUESTS = 32  # requests per union RPA call; its [32, P] page table at
# P = 2080 is 266 KB of scalar memory, as the 128-row per-row call's is.


def union_page_slots(budget, ratio, tile, rows=4):
    """Union page slots a request: ``rows`` rows' ``budget`` groups and their
    tail groups, in whole ``tile``-token KV tiles."""
    per_tile = tile // ratio
    return -(-(rows * (budget + 1)) // per_tile) * per_tile


# Diagnostics for the bench (scripts/bench_qsa_rpa_decode.py f15c): build the
# union metadata before the route instead of inside its union branch, or skip
# the route (wrong for requests of more than max_query_len rows).
_UNION_META_FIRST = False
_UNION_ROUTE = True


def union_request_slots(rows, requests, max_query_len):
    """Request slots the union path covers: whole calls of ``UNION_REQUESTS``,
    as many as a step of ``rows`` rows holding ``max_query_len`` rows a request
    fills (a verify step's request slots past them are empty), at most the
    program's ``requests``."""
    calls = -(-rows // (UNION_REQUESTS * max_query_len))
    return min(calls * UNION_REQUESTS, -(-requests // UNION_REQUESTS) * UNION_REQUESTS)


def union_decode_applies(cache, *, kv_heads, ratio, kv_head_offset, max_query_len, dcp):
    """VLLM_TPU_QSA_DECODE_RPA_UNION covers a verify step's decode branch:
    main-KV DCP in the decode layout, the compact schedule and slot reads,
    1 < max_query_len <= 4, 512-token pages of 4-token groups."""
    tile = envs.VLLM_TPU_QSA_DECODE_RPA_UNION
    if not tile:
        return False
    if tile % 128:
        raise ValueError(f"VLLM_TPU_QSA_DECODE_RPA_UNION={tile} must be a multiple of 128")
    return bool(
        1 < max_query_len <= 4
        and dcp is not None
        and dcp_interleave(dcp, cache.shape[1]) == cache.shape[1]
        and ratio == 4
        and cache.shape[1] == 512
        and envs.VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE
        and envs.VLLM_TPU_QSA_DECODE_SLOT_READ
        and slot_readable(cache, kv_heads, kv_head_offset)
    )


def union_metadata(blocks, lengths, table, starts, *, rows, ratio, max_query_len, dcp):
    """``(starts, pages, sel, meta, live)`` for ``union_decode_attention``: the
    covered requests' starts, and their union pages, per-token row words and
    meta from the SparseCore kernel (qsa_union_meta), with each row's
    liveness."""
    from vllm_torchtpu.kernels.qwen4_exp.qsa_union_meta import (
        union_row_live,
        union_sc_metadata,
    )

    slots = union_page_slots(blocks.shape[1], ratio, envs.VLLM_TPU_QSA_DECODE_RPA_UNION)
    starts = starts.astype(jnp.int32)
    covered = union_request_slots(rows, len(starts) - 1, max_query_len)
    if covered < len(starts) - 1:
        starts = starts[: covered + 1]
    # write="tile": page and word slots past the tile holding a request's last
    # page are left unwritten; RPA reads no tile past kv_lens.
    pages, sel, meta = union_sc_metadata(
        blocks,
        lengths,
        starts,
        table,
        dcp[0],
        world=dcp[1],
        pages_per_seq=slots,
        # A true dcp[3]: rank-only lists, as _dcp_page_metadata reads them.
        own_list=len(dcp) > 3 and bool(dcp[3]),
        emit_sel=True,
        write="tile",
    )
    return starts, pages, sel, meta, union_row_live(meta, starts, rows)


def union_decode_attention(
    q,
    blocks,
    lengths,
    table,
    cache,
    starts,
    *,
    ratio,
    k_scale,
    v_scale,
    kv_head_offset,
    max_query_len,
    dcp,
    read_only=False,
    metadata=None,
):
    """Verify-step attention on this DCP rank's owned keys, one RPA sequence
    per request over the union of its rows' owned pages.

    Same contract as ``decode_attention(..., dcp=dcp, return_lse=True)``:
    ``(out, lse, cache)``, ``out`` f32 normalized over the row's owned keys
    and not gated, ``lse`` natural log; a row with no owned key gives 0 and
    -inf. Each row attends exactly the keys it attends there; the sums run in
    another order. Requests go to RPA in calls of ``UNION_REQUESTS``, each
    over its contiguous rows. A verify step holds at most ``max_query_len``
    rows a request; rows past a request's fourth get 0 and -inf, and a call
    attends no row past its ``UNION_REQUESTS * max_query_len``-row slice
    (``qsa_attention`` routes steps with longer requests to the per-row
    path). Only the first ``union_request_slots`` requests are attended; the
    caller routes a step with live requests past them to the per-row path.
    ``read_only``: see ``_rpa_attention_batch``. ``metadata``:
    ``union_metadata``'s result, when the caller built it.
    """
    from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

    tile = envs.VLLM_TPU_QSA_DECODE_RPA_UNION
    if metadata is None:
        metadata = union_metadata(
            blocks, lengths, table, starts, rows=len(q), ratio=ratio,
            max_query_len=max_query_len, dcp=dcp,
        )
    starts, pages, sel, meta, live = metadata
    requests = len(starts) - 1
    chunks = -(-requests // UNION_REQUESTS)
    pad = chunks * UNION_REQUESTS - requests
    if pad > 0:
        starts = jnp.concatenate([starts, jnp.broadcast_to(starts[-1:], (pad,))])
        pages, sel, meta = (jnp.pad(x, ((0, pad), (0, 0))) for x in (pages, sel, meta))
    call_rows = UNION_REQUESTS * max_query_len
    # VLLM_TPU_QSA_UNION_LEAN_GLUE: when one call covers the step, it runs over
    # the step's own rows, so its query, output and LSE buffers have the step's
    # size and call row r is step row r. A row before starts[0] belongs to no
    # request; union_row_live masks whatever the kernel leaves in it.
    # VLLM_TPU_QSA_UNION_CHAINED_CALLS: with several calls, each runs over the
    # whole step too and takes the previous call's output and LSE buffers, in
    # the kernel's layout, as its own (wrapper carry). A call reads the queries
    # of its own sequences, rows no earlier call wrote, and writes only them.
    lean = bool(envs.VLLM_TPU_QSA_UNION_LEAN_GLUE)
    chained = bool(envs.VLLM_TPU_QSA_UNION_CHAINED_CALLS) and chunks > 1
    whole = (lean and chunks == 1 and len(q) <= call_rows) or chained
    span = call_rows
    if whole:
        call_rows = len(q)
    # Rows past the last call's slice are padding, so a slice never clamps.
    query = jnp.pad(
        q.astype(jnp.float32),
        ((0, 0 if whole else call_rows), (0, 0), (0, cache.shape[-1] - q.shape[-1])),
    )
    dummy_k = jnp.zeros((call_rows, 1, query.shape[-1]), cache.dtype)
    virtual_cache = cache.reshape(-1, ratio, *cache.shape[2:])
    blocks_cfg = configs.BlockSizes(
        bq_sz=max_query_len,
        bq_c_sz=max_query_len,
        bkv_sz=tile,
        batch_size=envs.VLLM_TPU_QSA_DECODE_RPA_LANES,
    )
    skip = envs.VLLM_TPU_QSA_DECODE_RPA_SKIP_EMPTY
    out = None if whole else jnp.zeros(query.shape, jnp.float32)
    lse = None if whole else jnp.full(query.shape[:2], -jnp.inf, jnp.float32)
    carry = None
    for c in range(chunks):
        first = c * UNION_REQUESTS
        part = slice(first, first + UNION_REQUESTS)
        row0 = 0 if whole else starts[first]
        edges = starts[first : first + UNION_REQUESTS + 1]
        if chained:
            # The rows a sliced call would attend, [starts[first], + span),
            # in step coordinates.
            cu_q_lens = jnp.clip(edges, edges[0], jnp.minimum(edges[0] + span, len(q)))
        else:
            cu_q_lens = jnp.clip(edges - row0, 0, call_rows)
        out_c, virtual_cache, lse_c = wrapper.ragged_paged_attention(
            query if whole else jax.lax.dynamic_slice_in_dim(query, row0, call_rows),
            dummy_k,
            dummy_k,
            virtual_cache,
            meta[part, 1],
            pages[part].reshape(-1),
            # A request of more than max_query_len rows (the precompile's
            # dummy steps) could carry a call past its query slice; rows
            # past the slice are not attended.
            cu_q_lens,
            jnp.full((3,), UNION_REQUESTS, jnp.int32),
            sm_scale=q.shape[-1] ** -0.5,
            q_scale=None,
            k_scale=k_scale,
            v_scale=v_scale,
            out_dtype=jnp.dtype(jnp.float32),
            skip_kv_update=True,
            kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
            decode_block_sizes=blocks_cfg,
            prefill_block_sizes=dataclasses.replace(blocks_cfg, batch_size=1),
            decode_only=True,
            lean_kv_fetch=bool(
                envs.VLLM_TPU_QSA_DECODE_LEAN_FETCH or envs.VLLM_TPU_QSA_DECODE_FAST_FETCH
            ),
            fast_kv_fetch=bool(envs.VLLM_TPU_QSA_DECODE_FAST_FETCH),
            kv_slot=kv_head_offset,
            compact_kv_schedule=True,
            **({"kv_skip_group": skip} if skip else {}),
            return_lse=True,
            decode_query_size=max_query_len,
            selection_bits=sel[part],
            **({"read_only_cache": True} if read_only else {}),
            schedule_in_hbm=envs.VLLM_TPU_RPA_SCHEDULE_HBM,
            **({"selection_word_rows": True} if lean else {}),
            **({"carry": carry, "return_carry": True} if chained else {}),
        )
        if chained:
            carry = (out_c, lse_c)
            continue
        if whole:
            out, lse = out_c, lse_c.astype(jnp.float32)
            continue
        # Later calls start at their own first row, so each row keeps the
        # value its own call wrote.
        out = jax.lax.dynamic_update_slice_in_dim(out, out_c, row0, 0)
        lse = jax.lax.dynamic_update_slice_in_dim(lse, lse_c.astype(jnp.float32), row0, 0)
    if chained:
        out, lse = wrapper.finish_carry(carry, query.shape, 1, query.dtype)
    out = jnp.where(live[:, None, None], out[: len(q), :, : q.shape[-1]], 0)
    lse = jnp.where(live[:, None], lse[: len(q)], -jnp.inf)
    return out, lse, virtual_cache.reshape(cache.shape)


def fused_sparse_decode(
    q,
    gate,
    blocks,
    lengths,
    table,
    cache,
    *,
    kv_heads,
    ratio,
    k_scale=1.0,
    v_scale=1.0,
    kv_head_offset=0,
):
    """``rpa_attention`` on the fused sparse GQA decode kernel, or None.

    Same contract as ``rpa_attention``: one query row per token, gated output
    (the kernel applies the sigmoid gate before its output DMA, so callers
    must not gate again), zero rows for zero lengths, and the cache returned
    unchanged through donation. Returns None when the inputs fall outside the
    kernel's static envelope (non-TPU7x, unsupported shapes or dtypes) so the
    caller can fall back to RPA.

    ``kv_head_offset`` follows ``rpa_attention``: the kernel reads a row's
    leading heads, so the cache presents ``kv_head_offset + kv_heads`` heads
    and zero queries/gates attend the heads ahead of this layer's.
    """
    from vllm_torchtpu.kernels import sparse_gqa_decode
    from vllm_torchtpu.kernels.sparse_gqa_decode import decode

    if kv_heads <= 0 or kv_head_offset < 0 or q.shape[1] % kv_heads:
        return None
    skipped_heads = kv_head_offset * (q.shape[1] // kv_heads)
    if skipped_heads:
        pad = ((0, 0), (skipped_heads, 0), (0, 0))
        q, gate = jnp.pad(q, pad), jnp.pad(gate, pad)
    options = dict(
        kv_heads=kv_heads + kv_head_offset,
        ratio=ratio,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    launch = sparse_gqa_decode.plan_decode(
        q, gate, blocks, lengths, table, cache, **options
    )
    if launch is None:
        return None
    out, cache = decode.execute_decode(
        q,
        gate,
        blocks,
        lengths,
        table,
        cache,
        launch=launch,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    return (out[:, skipped_heads:] if skipped_heads else out), cache


def decode_attention(
    q,
    gate,
    blocks,
    lengths,
    table,
    cache,
    *,
    kv_heads,
    ratio,
    k_scale=1.0,
    v_scale=1.0,
    kv_head_offset=0,
    fused=None,
    shared_schedule=None,
    dcp=None,
    return_lse=False,
    read_only=False,
):
    """Per-row decode attention: fused sparse decode when enabled, else RPA.

    ``dcp`` / ``return_lse``: see ``rpa_attention`` (RPA only: the fused
    sparse decode kernel has no LSE output). With ``return_lse`` returns
    ``(out, lse, cache)``.

    ``fused`` defaults to ``VLLM_TPU_QSA_FUSED_SPARSE_DECODE`` (read at trace
    time). Ineligible inputs fall back to RPA.

    ``shared_schedule`` (default ``VLLM_TPU_QSA_DECODE_SHARED_SCHEDULE``)
    presents every head slot of the cache row, with zero queries for the heads
    that are not this layer's, so both layers of a paired main KV page (and
    unpaired layers, whose FP8 rows hold the same slots) launch one kernel
    configuration. The decode schedule depends only on the step's lengths
    and that configuration, so XLA computes it once per step instead of once
    per slot; RPA also skips its empty mixed-sequence pass. With
    ``VLLM_TPU_QSA_DECODE_SLOT_READ`` RPA reads each layer's own head slot
    instead, so every layer already launches one KV head and nothing is
    padded.
    """
    shared = (
        envs.VLLM_TPU_QSA_DECODE_SHARED_SCHEDULE
        if shared_schedule is None
        else shared_schedule
    )
    trailing = 0
    if shared and cache.ndim == 5:
        trailing = cache.shape[2] * cache.shape[3] // 2 - kv_head_offset - kv_heads
    use_fused = envs.VLLM_TPU_QSA_FUSED_SPARSE_DECODE if fused is None else fused
    if return_lse or dcp is not None:
        use_fused = fused = False
    if envs.VLLM_TPU_QSA_DECODE_SLOT_READ and not use_fused and slot_readable(
        cache, kv_heads, kv_head_offset
    ):
        # Every layer already launches one KV head (slot reads); no padding.
        trailing = 0
    if not shared or trailing <= 0 or q.shape[1] % kv_heads:
        trailing = 0
    pad = ((0, 0), (0, trailing * (q.shape[1] // kv_heads)), (0, 0))
    result = _decode_attention(
        jnp.pad(q, pad) if trailing else q,
        jnp.pad(gate, pad) if trailing else gate,
        blocks,
        lengths,
        table,
        cache,
        kv_heads=kv_heads + trailing,
        ratio=ratio,
        k_scale=k_scale,
        v_scale=v_scale,
        kv_head_offset=kv_head_offset,
        fused=fused,
        decode_only=shared,
        **({"dcp": dcp} if dcp is not None else {}),
        **({"return_lse": True} if return_lse else {}),
        **({"read_only": True} if read_only else {}),
    )
    *outs, cache = result
    outs = [x[:, : q.shape[1]] if trailing else x for x in outs]
    return (*outs, cache)


def _decode_attention(
    q,
    gate,
    blocks,
    lengths,
    table,
    cache,
    *,
    kv_heads,
    ratio,
    k_scale,
    v_scale,
    kv_head_offset,
    fused,
    decode_only=False,
    dcp=None,
    return_lse=False,
    read_only=False,
):
    options = dict(kv_heads=kv_heads, ratio=ratio, k_scale=k_scale, v_scale=v_scale)
    if kv_head_offset:
        options["kv_head_offset"] = kv_head_offset
    if not return_lse and dcp is None and (
        envs.VLLM_TPU_QSA_FUSED_SPARSE_DECODE if fused is None else fused
    ):
        result = fused_sparse_decode(q, gate, blocks, lengths, table, cache, **options)
        if result is not None:
            return result
    if decode_only:
        options["decode_only"] = True
    if envs.VLLM_TPU_QSA_DECODE_LEAN_FETCH:
        options["lean_fetch"] = True
    if envs.VLLM_TPU_QSA_DECODE_SLOT_READ:
        options["slot_read"] = True
    if envs.VLLM_TPU_QSA_DECODE_FAST_FETCH:
        options["fast_fetch"] = True
    if envs.VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE:
        options["compact"] = True
    if dcp is not None:
        options["dcp"] = dcp
    if return_lse:
        options["return_lse"] = True
    if read_only:
        options["read_only"] = True

    return rpa_attention(q, gate, blocks, lengths, table, cache, **options)


def _prefill_attention(
    query,
    gate,
    blocks,
    lengths,
    table,
    cache,
    starts,
    *,
    kv_heads,
    ratio,
    k_scale=1.0,
    v_scale=1.0,
    kv_head_offset=0,
    dcp=None,
    return_lse=False,
    compare=True,
):
    """With ``return_lse`` returns ``(out, lse, cache)`` (see
    ``block_sparse_gqa_attention``). ``compare=False`` skips the compare core
    (its own fallback: the flag-off call)."""
    if (
        compare
        and envs.VLLM_TPU_QSA_PREFILL_COMPARE_CORE
        and dcp is None
        and not return_lse
        and kv_head_offset == 0
        and ratio == 4
        and cache.dtype == jnp.float8_e4m3fn
        and cache.shape[1] % 128 == 0
        and tuple(cache.shape[2:]) == (1, 4, 256)
        and query.shape[1:] == (24, 256)
        and kv_heads == 2
        and query.shape[0] % 16 == 0
        and query.shape[0] >= 1024
    ):
        from vllm_torchtpu.kernels.qwen4_exp.qsa_compare_core import compare_core_prefill

        def fallback():
            output, _ = _prefill_attention(
                query, gate, blocks, lengths, table, cache, starts, kv_heads=kv_heads, ratio=ratio,
                k_scale=k_scale, v_scale=v_scale, compare=False,
            )
            return output

        return compare_core_prefill(
            query, gate, blocks, lengths, table, cache, starts, kv_heads=kv_heads, ratio=ratio,
            k_scale=k_scale, v_scale=v_scale, union_cap=envs.VLLM_TPU_QSA_PREFILL_COMPARE_CORE_CAP,
            min_context=envs.VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT, fallback=fallback,
            depth=envs.VLLM_TPU_QSA_PREFILL_COMPARE_CORE_DEPTH,
            bitset="sparsecore" if envs.VLLM_TPU_QSA_PREFILL_UNION_SC_BITSET else "tensorcore",
            lister=envs.VLLM_TPU_QSA_PREFILL_COMPARE_CORE_TABLES,
            stats=envs.VLLM_TPU_QSA_PREFILL_COMPARE_CORE_STATS,
            held_tokens=envs.VLLM_TPU_QSA_PREFILL_COMPARE_CORE_HELD,
        )
    union_group = envs.VLLM_TPU_QSA_PREFILL_UNION_WALK
    if union_group and (128 % union_group or union_group < 8):
        raise ValueError("VLLM_TPU_QSA_PREFILL_UNION_WALK must be a divisor of 128, at least 8")
    if (
        union_group
        and dcp is None
        and not return_lse
        and kv_head_offset == 0
        and ratio == 4
        and cache.dtype == jnp.float8_e4m3fn
        and cache.shape[1] % 128 == 0
        and tuple(cache.shape[2:]) == (1, 4, 256)
        and query.shape[1:] == (24, 256)
        and kv_heads == 2
        and query.shape[0] % union_group == 0
        and query.shape[0] >= 1024
    ):
        from vllm_torchtpu.kernels.qwen4_exp.qsa_union_dense import union_walk_prefill

        return union_walk_prefill(
            query,
            gate,
            blocks,
            lengths,
            table,
            cache,
            starts,
            kv_heads=kv_heads,
            ratio=ratio,
            k_scale=k_scale,
            v_scale=v_scale,
            group=union_group,
            kv_tile=envs.VLLM_TPU_QSA_PREFILL_UNION_KV_TILE,
            union_cap=envs.VLLM_TPU_QSA_PREFILL_UNION_CAP,
            min_context=envs.VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT,
            lean=envs.VLLM_TPU_QSA_PREFILL_LEAN,
            listed=envs.VLLM_TPU_QSA_PREFILL_UNION_LISTED or None,
            share_bits=envs.VLLM_TPU_QSA_PREFILL_UNION_SHARE_BITS,
            union_alpha=envs.VLLM_TPU_QSA_PREFILL_UNION_ALPHA,
            qk16=envs.VLLM_TPU_QSA_PREFILL_UNION_QK16,
            run_tables=envs.VLLM_TPU_QSA_PREFILL_UNION_RUN_TABLES,
            lazy_share=envs.VLLM_TPU_QSA_PREFILL_UNION_LAZY_SHARE,
            gathered=envs.VLLM_TPU_QSA_PREFILL_UNION_GATHERED or None,
            gather_chunks=envs.VLLM_TPU_QSA_PREFILL_UNION_GATHER_CHUNKS,
            gather_growth=envs.VLLM_TPU_QSA_PREFILL_UNION_GATHER_GROWTH,
            compact_reqs=envs.VLLM_TPU_QSA_PREFILL_UNION_COMPACT_REQS,
            compact_copy_method=envs.VLLM_TPU_QSA_PREFILL_UNION_COMPACT_COPY,
            sc_bitset=envs.VLLM_TPU_QSA_PREFILL_UNION_SC_BITSET,
            **({"in_place": True} if envs.VLLM_TPU_QSA_PREFILL_UNION_IN_PLACE else {}),
            **({"sc_bits_plan": True} if envs.VLLM_TPU_QSA_PREFILL_UNION_SC_BITS_PLAN else {}),
            **({"gathered_lean": True} if envs.VLLM_TPU_QSA_PREFILL_UNION_GATHERED_LEAN else {}),
            **({"mask_view": True} if envs.VLLM_TPU_QSA_PREFILL_UNION_MASK_VIEW else {}),
            **({"vguard_lean": True} if envs.VLLM_TPU_QSA_PREFILL_UNION_VGUARD_LEAN else {}),
            **({"align_gather": True} if envs.VLLM_TPU_QSA_PREFILL_UNION_ALIGN_GATHER else {}),
        )
    from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (
        block_sparse_gqa_attention,
    )
    from vllm_torchtpu.kernels.qwen4_exp.prefill_tiles import (
        get_sparse_prefill_tiles,
    )

    query_block, kv_tile, query_tile = get_sparse_prefill_tiles(
        query, cache, kv_heads=kv_heads, ratio=ratio, with_query_tile=True
    )
    if dcp is not None:
        # One KV tile per ownership unit of every rank (a table entry, dcp
        # pages' worth of tokens, by default); the paired-page kernel fetches
        # this rank's rows of each tile.
        kv_tile = dcp[1] * dcp_interleave(dcp, cache.shape[1])
        if query_block % 128 or query_block * query.shape[1] > 256 * 24:
            query_block, query_tile = 256, None
        if query_tile and query_block % query_tile:
            query_tile = None
    return block_sparse_gqa_attention(
        query,
        gate,
        blocks,
        lengths,
        table,
        cache,
        starts,
        kv_heads=kv_heads,
        ratio=ratio,
        query_block=query_block,
        kv_tile=kv_tile,
        k_scale=k_scale,
        v_scale=v_scale,
        **({"kv_head_offset": kv_head_offset} if kv_head_offset else {}),
        **({"query_tile": query_tile} if query_tile else {}),
        **({"dcp": dcp} if dcp is not None else {}),
        **({"return_lse": True} if return_lse else {}),
    )


def _with_lse_channel(attend):
    """``attend(..., return_lse=True) -> (out, lse, cache)`` as ``(out_ext, cache)``
    with ``lse`` appended to ``out``'s lane axis, so the decode/prefill routing
    moves both with the same row selects."""

    def run(*args, **kwargs):
        out, lse, cache = attend(*args, **kwargs)
        return jnp.concatenate([out.astype(jnp.float32), lse[..., None]], axis=-1), cache

    return run


def _split_lse_channel(output, lengths, cache):
    live = (lengths > 0)[:, None]
    out = jnp.where(live[..., None], output[..., :-1], 0)
    lse = jnp.where(live, output[..., -1], -jnp.inf)
    return out, lse, cache


def dcp_combine(out_pair, lse_pair, gate, rank, *, out_dtype=jnp.bfloat16):
    """Merge a DCP group's partial attention for this rank's heads, then gate.

    ``out_pair`` [size, T, size * H, D] f32 and ``lse_pair`` [size, T, size * H]
    are every group rank's ``qsa_attention(dcp=...)`` results over the group's
    concatenated heads (rank 0's first); ``gate`` [T, H, D] is this rank's
    gate; ``rank`` int32 ``[1]``. Rows no rank owns a key of stay 0. The gate
    rounding matches the replicated kernels'.
    """
    size, tokens, heads_all, dim = out_pair.shape
    heads = heads_all // size
    start = rank.reshape(()).astype(jnp.int32) * heads
    out_pair = jax.lax.dynamic_slice_in_dim(out_pair, start, heads, axis=2)
    lse_pair = jax.lax.dynamic_slice_in_dim(lse_pair, start, heads, axis=2)
    top = jnp.max(lse_pair, axis=0)
    finite = jnp.isfinite(top)
    weight = jnp.where(finite[None], jnp.exp(lse_pair - jnp.where(finite, top, 0)[None]), 0)
    total = jnp.sum(weight, axis=0)
    out = jnp.sum(weight[..., None] * out_pair, axis=0) / jnp.where(total > 0, total, 1)[..., None]
    gated = bf16(
        bf16(out).astype(jnp.float32)
        * bf16(jax.nn.sigmoid(gate.astype(jnp.float32))).astype(jnp.float32)
    )
    return jnp.where(finite[..., None], gated, 0).astype(out_dtype)


def dcp_combine_step(
    out_pair: jax.Array, lse_pair: jax.Array, gate: jax.Array, rank: jax.Array
) -> jax.Array:
    """``dcp_combine`` with a fully annotated signature for ``jax_op``."""
    return dcp_combine(out_pair, lse_pair, gate, rank)


def _rank_weight(out, lse, lse_all):
    """This rank's partial scaled by ``exp(lse - max_r lse_r)`` over every head,
    the max, and whether any rank owns a key of each (token, head)."""
    top = jnp.max(lse_all, axis=0)
    finite = jnp.isfinite(top)
    weight = jnp.where(finite, jnp.exp(lse - jnp.where(finite, top, 0)), 0)
    # where, not *: a rank that owns no key of a row may hold garbage there.
    return jnp.where(weight[..., None] > 0, weight[..., None] * out, 0), top, finite


def dcp_merge_scale(out, lse, lse_all, *, out_dtype=jnp.float32):
    """This rank's addend of a reduce-scatter DCP merge (VLLM_TPU_QSA_DCP_MERGE=rs).

    ``out`` [T, S * H, D] f32 and ``lse`` [T, S * H] are this rank's
    ``qsa_attention(dcp=...)`` results over the group's concatenated heads;
    ``lse_all`` [S, T, S * H] is every group rank's ``lse``. Returns
    [S, T, H, D]: block ``d`` holds group rank ``d``'s heads scaled by
    ``exp(lse - max_r lse_r)``, the product ``dcp_combine`` sums, so a
    reduce-scatter of the blocks over the group gives each rank the numerator of
    its heads' merge (``dcp_merge_finish`` divides and gates it).
    """
    size = lse_all.shape[0]
    tokens, heads_all, dim = out.shape
    scaled = _rank_weight(out, lse, lse_all)[0]
    scaled = scaled.reshape(tokens, size, heads_all // size, dim).transpose(1, 0, 2, 3)
    return scaled.astype(out_dtype)


def dcp_merge_scale_step(out: jax.Array, lse: jax.Array, lse_all: jax.Array) -> jax.Array:
    """``dcp_merge_scale`` (f32 addends) with a fully annotated signature for ``jax_op``."""
    return dcp_merge_scale(out, lse, lse_all)


def dcp_merge_scale_bf16_step(
    out: jax.Array, lse: jax.Array, lse_all: jax.Array
) -> jax.Array:
    """``dcp_merge_scale`` with bf16 addends (VLLM_TPU_QSA_DCP_MERGE=rs_bf16)."""
    return dcp_merge_scale(out, lse, lse_all, out_dtype=jnp.bfloat16)


def dcp_merge_finish(summed, lse_all, gate, rank, *, out_dtype=jnp.bfloat16):
    """This rank's heads of the merge from the reduce-scattered numerator.

    ``summed`` [T, H, D] is the group's sum of ``dcp_merge_scale`` block
    ``rank``; ``lse_all`` [S, T, S * H]; ``gate`` [T, H, D] and ``rank`` int32
    ``[1]`` as in ``dcp_combine``, whose weights, normalization and gate
    rounding this repeats (only the order of the sum over ranks differs).
    """
    size, tokens, heads_all = lse_all.shape
    heads = heads_all // size
    start = rank.reshape(()).astype(jnp.int32) * heads
    lse_own = jax.lax.dynamic_slice_in_dim(lse_all, start, heads, axis=2)
    top = jnp.max(lse_own, axis=0)
    finite = jnp.isfinite(top)
    weight = jnp.where(finite[None], jnp.exp(lse_own - jnp.where(finite, top, 0)[None]), 0)
    total = jnp.sum(weight, axis=0)
    out = summed.astype(jnp.float32) / jnp.where(total > 0, total, 1)[..., None]
    gated = bf16(
        bf16(out).astype(jnp.float32)
        * bf16(jax.nn.sigmoid(gate.astype(jnp.float32))).astype(jnp.float32)
    )
    return jnp.where(finite[..., None], gated, 0).astype(out_dtype)


def dcp_merge_finish_step(
    summed: jax.Array, lse_all: jax.Array, gate: jax.Array, rank: jax.Array
) -> jax.Array:
    """``dcp_merge_finish`` with a fully annotated signature for ``jax_op``."""
    return dcp_merge_finish(summed, lse_all, gate, rank)


def dcp_merge_proj_share(out, lse, lse_all, gate_all, *, out_dtype=jnp.bfloat16):
    """This rank's share of the gated merge of every head of its DCP group
    (VLLM_TPU_QSA_DCP_MERGE=proj).

    Normalized weight times partial times the rounded gate, [T, S * H, D]. The
    merge and the gate are linear in the partials, so the group's shares summed
    give the gated merge; projecting each share by the output projection's
    columns of all S * H heads lets the projection's TP all-reduce do that sum.
    ``gate_all`` [T, S * H, D] is the group's gate.
    """
    scaled, top, finite = _rank_weight(out, lse, lse_all)
    weight = jnp.where(finite[None], jnp.exp(lse_all - jnp.where(finite, top, 0)[None]), 0)
    total = jnp.sum(weight, axis=0)
    share = scaled / jnp.where(total > 0, total, 1)[..., None]
    gated = share * bf16(jax.nn.sigmoid(gate_all.astype(jnp.float32))).astype(jnp.float32)
    return jnp.where(finite[..., None], gated, 0).astype(out_dtype)


def dcp_merge_proj_share_step(
    out: jax.Array, lse: jax.Array, lse_all: jax.Array, gate_all: jax.Array
) -> jax.Array:
    """``dcp_merge_proj_share`` with a fully annotated signature for ``jax_op``."""
    return dcp_merge_proj_share(out, lse, lse_all, gate_all)


def dcp_merge_check_stats(new, old, lse_all):
    """Agreement of a merge's [T, H, D] result with ``dcp_combine``'s ``old``
    (VLLM_TPU_QSA_DCP_MERGE_CHECK), f32 [8]: max abs error, max |old|, max
    error relative to its head vector's largest |old|, fraction of elements
    that differ, NaN count of ``new``, max abs difference between the merged
    LSE ``top + log(total)`` and ``logsumexp``, 1 (a call count), 0."""
    new32, old32 = new.astype(jnp.float32), old.astype(jnp.float32)
    nan = jnp.isnan(new32)
    err = jnp.where(nan, 0, jnp.abs(new32 - old32))
    head = jnp.max(jnp.abs(old32), axis=-1)
    rel = jnp.where(head > 0, jnp.max(err, axis=-1) / jnp.where(head > 0, head, 1), 0)
    top = jnp.max(lse_all, axis=0)
    finite = jnp.isfinite(top)
    shift = jnp.where(finite, top, 0)
    merged = shift + jnp.log(jnp.sum(jnp.exp(lse_all - shift[None]), axis=0))
    lse_err = jnp.where(finite, jnp.abs(merged - jax.nn.logsumexp(lse_all, axis=0)), 0)
    return jnp.stack(
        [
            jnp.max(err),
            jnp.max(jnp.abs(old32)),
            jnp.max(rel),
            jnp.mean((new32 != old32).astype(jnp.float32)),
            jnp.sum(nan.astype(jnp.float32)),
            jnp.max(lse_err),
            jnp.float32(1),
            jnp.float32(0),
        ]
    ).astype(jnp.float32)


def dcp_merge_check_stats_step(
    new: jax.Array, old: jax.Array, lse_all: jax.Array
) -> jax.Array:
    """``dcp_merge_check_stats`` with a fully annotated signature for ``jax_op``."""
    return dcp_merge_check_stats(new, old, lse_all)


def dcp_qgate_check_stats(new, old, head_dim):
    """Agreement of the locally projected group query/gate ``new`` with the
    all-gathered ``old`` (VLLM_TPU_QSA_DCP_LOCAL_QGATE under
    VLLM_TPU_QSA_DCP_MERGE_CHECK), both [T, heads * 2 * head_dim]: the f32 [8]
    of ``dcp_merge_check_stats`` over [T, heads, 2 * head_dim] vectors, LSE
    entry 0."""
    tokens = new.shape[0]
    new32 = new.astype(jnp.float32).reshape(tokens, -1, 2 * head_dim)
    old32 = old.astype(jnp.float32).reshape(tokens, -1, 2 * head_dim)
    nan = jnp.isnan(new32)
    err = jnp.where(nan, 0, jnp.abs(new32 - old32))
    head = jnp.max(jnp.abs(old32), axis=-1)
    rel = jnp.where(head > 0, jnp.max(err, axis=-1) / jnp.where(head > 0, head, 1), 0)
    return jnp.stack(
        [
            jnp.max(err),
            jnp.max(jnp.abs(old32)),
            jnp.max(rel),
            jnp.mean((new32 != old32).astype(jnp.float32)),
            jnp.sum(nan.astype(jnp.float32)),
            jnp.float32(0),
            jnp.float32(1),
            jnp.float32(0),
        ]
    ).astype(jnp.float32)


def dcp_qgate_check_stats_step(new: jax.Array, old: jax.Array, *, head_dim: int) -> jax.Array:
    """``dcp_qgate_check_stats`` with a fully annotated signature for ``jax_op``."""
    return dcp_qgate_check_stats(new, old, head_dim)


def pcp_combine(out_pair, lse_pair, gate, *, out_dtype=jnp.bfloat16):
    """Merge a PCP pair's partial attention over every head, then gate.

    Under PCP each rank attends all heads of all the step's tokens over the
    main-KV rows it owns, so ``out_pair`` [size, T, H, D] and ``lse_pair``
    [size, T, H] hold every rank's partial result for the same heads (no
    head slice as in :func:`dcp_combine`); ``gate`` [T, H, D]."""
    size = out_pair.shape[0]
    rank = jnp.zeros((1,), jnp.int32)
    tile = lambda x: jnp.concatenate([x] * size, axis=2)  # noqa: E731
    # dcp_combine slices heads [rank * H, (rank + 1) * H) of size * H heads.
    return dcp_combine(tile(out_pair), tile(lse_pair), gate, rank, out_dtype=out_dtype)


def pcp_combine_step(out_pair: jax.Array, lse_pair: jax.Array, gate: jax.Array) -> jax.Array:
    """``pcp_combine`` with a fully annotated signature for ``jax_op``."""
    return pcp_combine(out_pair, lse_pair, gate)


def qsa_attention(
    query,
    gate,
    blocks,
    lengths,
    table,
    cache,
    starts,
    *,
    kv_heads,
    ratio,
    k_scale=1.0,
    v_scale=1.0,
    kv_head_offset=0,
    max_query_len=1,
    dcp=None,
    union=False,
):
    """Route a decode prefix to per-row decode and the remaining rows to prefill.

    Decode rows run on batched RPA, or on the fused sparse GQA decode kernel
    when ``VLLM_TPU_QSA_FUSED_SPARSE_DECODE`` is set and the inputs qualify.

    ``max_query_len`` is the most tokens a decoding request carries in one step
    (``num_speculative_tokens + 1`` under MTP). A step with at most that many
    tokens per request row is a decode/verify step and runs every token on RPA
    as its own causal row; comparing tokens with request rows alone sent every
    MTP verify step with more than ``rows / (K + 1)`` requests to prefill.

    ``kv_head_offset`` reads this layer's KV heads after that many other heads
    of each cache row (the second layer of a paired main KV page).

    ``dcp=(rank, size)`` (VLLM_TPU_QSA_MAIN_KV_DCP): attend only this rank's
    main-KV rows and return ``(out, lse, cache)``: ``out`` [T, heads, D] f32
    normalized over the owned keys and NOT gated, ``lse`` [T, heads] f32
    (natural log, ``-inf`` when the row owns no key). ``dcp_combine`` merges
    the pair.

    ``union`` (target verify steps): when VLLM_TPU_QSA_DECODE_RPA_UNION
    covers the decode branch (``union_decode_applies``), a step whose requests
    all hold at most ``max_query_len`` rows takes ``union_decode_attention``
    and any other step the per-row path, chosen on device. Both read the cache
    only, so the conditional returns out and LSE and not the cache (XLA would
    copy the pool for a returned cache).
    """
    options = dict(kv_heads=kv_heads, ratio=ratio, k_scale=k_scale, v_scale=v_scale)
    if kv_head_offset:
        options["kv_head_offset"] = kv_head_offset
    if dcp is not None:
        options.update(dcp=dcp, return_lse=True)
        decode_fn = _with_lse_channel(decode_attention)
        prefill_fn = _with_lse_channel(_prefill_attention)
    else:
        decode_fn, prefill_fn = decode_attention, _prefill_attention
    if dcp is not None and dcp_interleave(dcp, cache.shape[1]) != cache.shape[1]:
        # Interleaved ownership is the PCP prefill's layout: a prefill engine,
        # so every row (short ones too) runs on the paired prefill kernel
        # rather than the decode path's micro-page RPA.
        output, cache = prefill_fn(query, gate, blocks, lengths, table, cache, starts, **options)
        return _split_lse_channel(output, lengths, cache)
    if len(query) <= (len(starts) - 1) * max_query_len:
        if union and union_decode_applies(
            cache,
            kv_heads=kv_heads,
            ratio=ratio,
            kv_head_offset=kv_head_offset,
            max_query_len=max_query_len,
            dcp=dcp,
        ):
            first = None
            if _UNION_META_FIRST:
                first = union_metadata(
                    blocks, lengths, table, starts, rows=len(query), ratio=ratio,
                    max_query_len=max_query_len, dcp=dcp,
                )

            def by_union(operands):
                query, gate, blocks, lengths, table, starts, cache, meta = operands
                out, lse, _ = union_decode_attention(
                    query,
                    blocks,
                    lengths,
                    table,
                    cache,
                    starts,
                    ratio=ratio,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    kv_head_offset=kv_head_offset,
                    max_query_len=max_query_len,
                    dcp=dcp,
                    read_only=True,
                    metadata=meta,
                )
                return out, lse

            def by_rows(operands):
                query, gate, blocks, lengths, table, starts, cache, _ = operands
                out, lse, _ = decode_attention(
                    query, gate, blocks, lengths, table, cache, read_only=True, **options
                )
                live = (lengths > 0)[:, None]
                return (
                    jnp.where(live[..., None], out.astype(jnp.float32), 0),
                    jnp.where(live, lse.astype(jnp.float32), -jnp.inf),
                )

            # Every request at most max_query_len rows, and none past the
            # request slots the union path covers.
            covered = union_request_slots(len(query), len(starts) - 1, max_query_len)
            # Traced once per program: shows which decode programs hold the route.
            logger.info_once(
                "QSA decode RPA union: %d-token tile, %d rows, %d request slots, "
                "%d covered, own lists %s",
                envs.VLLM_TPU_QSA_DECODE_RPA_UNION, len(query), len(starts) - 1,
                covered, len(dcp) > 3 and bool(dcp[3]),
            )
            fits = jnp.all(starts[1:] - starts[:-1] <= max_query_len)
            if covered < len(starts) - 1:
                fits = fits & (starts[covered] == starts[-1])
            operands = (query, gate, blocks, lengths, table, starts, cache, first)
            if not _UNION_ROUTE:
                out, lse = by_union(operands)
                return out, lse, cache
            out, lse = jax.lax.cond(fits, by_union, by_rows, operands)
            return out, lse, cache
        output, cache = decode_fn(query, gate, blocks, lengths, table, cache, **options)
        return _split_lse_channel(output, lengths, cache) if dcp is not None else (output, cache)

    # RPA rows: one per request row, times the tokens a verify request holds.
    capacity = min(len(query), (len(starts) - 1) * max_query_len)
    # Steps with a short-request prefix run its tokens on RPA and the rest on
    # prefill; other steps run every row on prefill. Pick the route by masking
    # the inputs, not with lax.cond: a conditional that returns the cache
    # makes XLA copy it, which for the unified pool is a second full pool in
    # HBM. RPA then runs on every step, on zero-length rows when unused, and
    # the cache threads through both kernels exactly once.
    prefix = short_request_prefix_tokens(starts, capacity)
    decode_lengths = jnp.where(jnp.arange(capacity) < prefix, lengths[:capacity], 0)
    decoded, cache = decode_fn(
        query[:capacity],
        gate[:capacity],
        blocks[:capacity],
        decode_lengths,
        table[:capacity],
        cache,
        **options,
    )
    if envs.VLLM_TPU_QSA_ZERO_PREFIX_FAST and dcp is None:
        # Steps without a decode prefix hand the prefill kernel the rows in
        # place; the rest route as below. Same values.
        return _zero_prefix_fast_route(
            query,
            gate,
            blocks,
            lengths,
            table,
            cache,
            starts,
            decoded,
            prefix,
            capacity=capacity,
            prefill_fn=prefill_fn,
            options=options,
        )
    # Keep bucket shapes fixed while removing the decode prefix on device.
    rows = jnp.arange(len(query))
    if envs.VLLM_TPU_QSA_PREFIX_SLICE:
        # Contiguous dynamic slices instead of row gathers: the same rows (clamped at the
        # last one), read at HBM rate rather than per-row index rate.
        def shift(x):
            return _shift_rows_up(x, prefix, capacity)

        prefill_lengths = jnp.where(rows < starts[-1] - prefix, shift(lengths), 0)
        prefilled, cache = prefill_fn(
            shift(query),
            shift(gate),
            shift(blocks),
            prefill_lengths,
            shift(table),
            cache,
            jnp.maximum(starts - prefix, 0),
            **options,
        )
        output = jnp.where(
            (rows < prefix)[:, None, None],
            _pad_rows(decoded, len(query)),
            _shift_rows_down(prefilled, prefix, capacity),
        )
        if dcp is not None:
            return _split_lse_channel(output, lengths, cache)
        return jnp.where((lengths > 0)[:, None, None], output, 0), cache
    ids = jnp.minimum(rows + prefix, len(query) - 1)
    prefill_lengths = jnp.where(rows < starts[-1] - prefix, lengths[ids], 0)
    prefilled, cache = prefill_fn(
        query[ids],
        gate[ids],
        blocks[ids],
        prefill_lengths,
        table[ids],
        cache,
        jnp.maximum(starts - prefix, 0),
        **options,
    )
    output = jnp.where(
        (rows < prefix)[:, None, None],
        decoded[jnp.minimum(rows, capacity - 1)],
        prefilled[jnp.maximum(rows - prefix, 0)],
    )
    if dcp is not None:
        return _split_lse_channel(output, lengths, cache)
    return jnp.where((lengths > 0)[:, None, None], output, 0), cache


def _zero_prefix_fast_route(
    query,
    gate,
    blocks,
    lengths,
    table,
    cache,
    starts,
    decoded,
    prefix,
    *,
    capacity,
    prefill_fn,
    options,
):
    """``qsa_attention``'s prefill rows under VLLM_TPU_QSA_ZERO_PREFIX_FAST.

    The same values as the routed path. A step without a decode prefix
    (``prefix == 0``: every step of a disaggregated prefill engine but a tiny
    leading request) hands the prefill kernel its rows in place: no row
    gathers in, no gather and ``where`` merge out, only the final
    ``lengths > 0`` mask. Steps with a prefix move the rows as before (row
    gathers, or VLLM_TPU_QSA_PREFIX_SLICE's slices) and merge the decoded rows
    back. ``lax.cond`` selects only the row movement and the output merge; the
    prefill kernel and the cache stay outside it, so the walk is traced once
    and the cache is never a conditional's result (that would copy the pool).

    With ``prefix == 0`` the routed path's ``ids`` are ``rows``, its prefill
    lengths ``where(rows < starts[-1], lengths, 0)``, its starts ``starts``
    and its output ``prefilled``.
    """
    n = len(query)
    rows = jnp.arange(n)
    move_in, move_out, decoded_rows = _prefix_moves(decoded, prefix, n, capacity)
    in_place = prefix == 0
    moved = jax.lax.cond(
        in_place,
        lambda xs: xs,
        lambda xs: tuple(move_in(x) for x in xs),
        (query, gate, blocks, lengths, table),
    )
    prefill_lengths = jnp.where(rows < starts[-1] - prefix, moved[3], 0)
    prefilled, cache = prefill_fn(
        moved[0],
        moved[1],
        moved[2],
        prefill_lengths,
        moved[4],
        cache,
        jnp.maximum(starts - prefix, 0),
        **options,
    )
    live = (lengths > 0)[:, None, None]

    def direct(out):
        return jnp.where(live, out, 0)

    def merged(out):
        head = (rows < prefix)[:, None, None]
        return jnp.where(live, jnp.where(head, decoded_rows(), move_out(out)), 0)

    return jax.lax.cond(in_place, direct, merged, prefilled), cache


def _prefix_moves(decoded, prefix, n, capacity):
    """The routed path's row movement: ``(move_in, move_out, decoded_rows)``.

    ``move_in(x)[r] = x[min(r + prefix, n - 1)]``, ``move_out(x)[r] =
    x[max(r - prefix, 0)]`` and ``decoded_rows()[r] = decoded[min(r,
    capacity - 1)]``: row gathers, or contiguous dynamic slices under
    VLLM_TPU_QSA_PREFIX_SLICE.
    """
    if envs.VLLM_TPU_QSA_PREFIX_SLICE:

        def move_in(x):
            return _shift_rows_up(x, prefix, capacity)

        def move_out(x):
            return _shift_rows_down(x, prefix, capacity)

        def decoded_rows():
            return _pad_rows(decoded, n)

        return move_in, move_out, decoded_rows
    rows = jnp.arange(n)

    def gather_in(x):
        return x[jnp.minimum(rows + prefix, n - 1)]

    def gather_out(x):
        return x[jnp.maximum(rows - prefix, 0)]

    def gather_decoded():
        return decoded[jnp.minimum(rows, capacity - 1)]

    return gather_in, gather_out, gather_decoded


def _pad_rows(x: jax.Array, n: int) -> jax.Array:
    """``x[min(r, len(x) - 1)]`` for ``r < n``: the last row repeated, a static concat."""
    if len(x) >= n:
        return x[:n]
    return jnp.concatenate([x, jnp.broadcast_to(x[-1:], (n - len(x),) + x.shape[1:])])


def _shift_rows_up(x: jax.Array, shift: jax.Array, max_shift: int) -> jax.Array:
    """``x[min(r + shift, len(x) - 1)]`` for ``0 <= shift <= max_shift``, one dynamic slice."""
    return jax.lax.dynamic_slice_in_dim(_pad_rows(x, len(x) + max_shift), shift, len(x), 0)


def _shift_rows_down(x: jax.Array, shift: jax.Array, max_shift: int) -> jax.Array:
    """``x[max(r - shift, 0)]`` for ``0 <= shift <= max_shift``, one dynamic slice."""
    head = jnp.broadcast_to(x[:1], (max_shift,) + x.shape[1:])
    return jax.lax.dynamic_slice_in_dim(
        jnp.concatenate([head, x]), max_shift - shift, len(x), 0
    )


def qsa_step(
    qgate: jax.Array,
    key: jax.Array,
    value: jax.Array,
    index_qk: jax.Array,
    nq: jax.Array,
    nk: jax.Array,
    niq: jax.Array,
    nik: jax.Array,
    positions: jax.Array,
    trig: jax.Array,
    cache: jax.Array,
    compressed: jax.Array,
    ring: jax.Array,
    tags: jax.Array,
    starts: jax.Array,
    context: jax.Array,
    slots: jax.Array,
    main_table: jax.Array,
    compressed_table: jax.Array,
    compress_ratio: int,
    token_topk: int,
    norm_eps: float,
    rotary_dim: int,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    if len(qgate) == 0:
        return (
            cache,
            compressed,
            ring,
            tags,
            jnp.empty((0, qgate.shape[1] // 2), jnp.bfloat16),
            jnp.empty((0, token_topk + compress_ratio - 1), jnp.int32),
        )
    q, k, v, gate, iq, raw = produce(
        qgate, key, value, index_qk, nq, nk, niq, positions, trig, norm_eps, rotary_dim
    )
    fp8 = cache.dtype == jnp.float8_e4m3fn
    req, logical, valid = _token_metadata(len(q), starts, context, slots)
    page = jnp.where(valid, main_table[req, logical // cache.shape[1]], cache.shape[0])
    cache = cache.at[page, logical % cache.shape[1]].set(
        pack_rows(k, v, fp8=fp8, k_scale=k_scale, v_scale=v_scale), mode="drop"
    )
    compressed, ring, tags = _update_indexer(
        raw,
        positions,
        compressed,
        ring,
        tags,
        nik,
        trig,
        starts,
        context,
        slots,
        compressed_table,
        compress_ratio,
        norm_eps,
        rotary_dim,
    )

    # Query-specific sparse pages: prefill queries of the same request can
    # select different groups and have different causal lengths.
    lengths = jnp.where(valid, logical + 1, 0).astype(jnp.int32)
    blocks = select_decode_blocks(
        iq,
        compressed,
        lengths,
        compressed_table,
        req,
        compress_ratio=compress_ratio,
        block_topk=token_topk // compress_ratio,
    )
    # Keep the persistent owner in RPA's native dtype. Full-cache uint8/FP8
    # bitcast-converts lower to HBM reads/writes even when layouts match.
    output, cache = qsa_attention(
        q,
        gate,
        blocks,
        lengths,
        main_table[req],
        cache,
        starts,
        kv_heads=k.shape[1],
        ratio=compress_ratio,
        k_scale=k_scale if fp8 else 1.0,
        v_scale=v_scale if fp8 else 1.0,
    )
    selected = selected_tokens(blocks, lengths, token_topk, compress_ratio)
    return cache, compressed, ring, tags, output.reshape(len(q), -1), selected


def selected_tokens(blocks, lengths, token_topk, compress_ratio):
    selected = jnp.where(
        blocks[..., None] >= 0,
        blocks[..., None] * compress_ratio + jnp.arange(compress_ratio),
        -1,
    ).reshape(len(blocks), token_topk)
    selected = jnp.pad(selected, ((0, 0), (0, compress_ratio - 1)), constant_values=-1)
    count = jnp.minimum(lengths // compress_ratio, blocks.shape[1])
    for tail in range(compress_ratio - 1):
        token = lengths // compress_ratio * compress_ratio + tail
        selected = selected.at[
            jnp.arange(len(blocks)), count * compress_ratio + tail
        ].set(jnp.where(token < lengths, token, -1))
    return selected
