# SPDX-License-Identifier: Apache-2.0
"""QSA with native HEAD_ALONG_SUBLANE storage and existing batched RPA decode.

Scheduler pages remain intact. Selected compression groups become virtual RPA
pages by a view; the returned cache is threaded through donation, never dropped.
Prefill reuses KV tiles through block-sparse attention; decode uses batched RPA,
or the fused sparse GQA decode kernel under VLLM_TPU_QSA_FUSED_SPARSE_DECODE.
"""

import functools

import jax
import jax.numpy as jnp

from vllm_torchtpu import envs
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


def virtual_page_metadata(blocks, lengths, table, *, ratio=4, page_size=128):
    """Map selected logical groups to physical micro-pages; no KV gathering."""
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
):
    from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

    kv_slot = None
    if slot_read and slot_readable(cache, kv_heads, kv_head_offset):
        # Read this layer's pair of the packed row directly: one KV head, no
        # zero queries for the other layer's head, one pair unpacked.
        kv_slot, kv_head_offset = kv_head_offset, 0
    pages, kv_lens, cu_q_lens, distribution = virtual_page_metadata(
        blocks, lengths, table, ratio=ratio, page_size=cache.shape[1]
    )
    # Queries are packed in request order; zero-length requests have no query.
    order = jnp.argsort(lengths == 0, stable=True)
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
    out, next_cache = wrapper.ragged_paged_attention(
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
    )
    out = out[jnp.minimum(cu_q_lens[:-1], len(q) - 1), :, : q.shape[-1]]
    if kv_head_offset:
        out = out[:, skipped_heads:]
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
):
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
            )
            if compact
            else {}
        ),
        **({"kv_head_offset": kv_head_offset} if kv_head_offset else {}),
    )
    if len(q) <= batch_size:
        return attend(q, gate, blocks, lengths, table, cache)
    padding = (-len(q)) % batch_size

    def split(x):
        x = jnp.pad(x, ((0, padding),) + ((0, 0),) * (x.ndim - 1))
        return x.reshape(-1, batch_size, *x.shape[1:])

    def step(cache, inputs):
        query, gate, blocks, lengths, table = inputs
        output, cache = attend(query, gate, blocks, lengths, table, cache)
        return cache, output

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
    return output.reshape(-1, *q.shape[1:])[: len(q)], cache


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
):
    """Per-row decode attention: fused sparse decode when enabled, else RPA.

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
    if envs.VLLM_TPU_QSA_DECODE_SLOT_READ and not use_fused and slot_readable(
        cache, kv_heads, kv_head_offset
    ):
        # Every layer already launches one KV head (slot reads); no padding.
        trailing = 0
    if not shared or trailing <= 0 or q.shape[1] % kv_heads:
        trailing = 0
    pad = ((0, 0), (0, trailing * (q.shape[1] // kv_heads)), (0, 0))
    out, cache = _decode_attention(
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
    )
    return (out[:, : q.shape[1]] if trailing else out), cache


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
):
    options = dict(kv_heads=kv_heads, ratio=ratio, k_scale=k_scale, v_scale=v_scale)
    if kv_head_offset:
        options["kv_head_offset"] = kv_head_offset
    if envs.VLLM_TPU_QSA_FUSED_SPARSE_DECODE if fused is None else fused:
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
):
    from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (
        block_sparse_gqa_attention,
    )
    from vllm_torchtpu.kernels.qwen4_exp.prefill_tiles import (
        get_sparse_prefill_tiles,
    )

    query_block, kv_tile, query_tile = get_sparse_prefill_tiles(
        query, cache, kv_heads=kv_heads, ratio=ratio, with_query_tile=True
    )
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
    )


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
    """
    options = dict(kv_heads=kv_heads, ratio=ratio, k_scale=k_scale, v_scale=v_scale)
    if kv_head_offset:
        options["kv_head_offset"] = kv_head_offset
    if len(query) <= (len(starts) - 1) * max_query_len:
        return decode_attention(query, gate, blocks, lengths, table, cache, **options)

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
    decoded, cache = decode_attention(
        query[:capacity],
        gate[:capacity],
        blocks[:capacity],
        decode_lengths,
        table[:capacity],
        cache,
        **options,
    )
    # Keep bucket shapes fixed while removing the decode prefix on device.
    rows = jnp.arange(len(query))
    ids = jnp.minimum(rows + prefix, len(query) - 1)
    prefill_lengths = jnp.where(rows < starts[-1] - prefix, lengths[ids], 0)
    prefilled, cache = _prefill_attention(
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
    return jnp.where((lengths > 0)[:, None, None], output, 0), cache


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
