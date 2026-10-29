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
"""QSA with native HEAD_ALONG_SUBLANE storage and existing batched RPA decode.

Scheduler pages remain intact. Selected compression groups become virtual RPA
pages by a view; the returned cache is threaded through donation, never dropped.
Prefill reuses KV tiles through block-sparse attention; decode uses batched RPA.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.topk_backend import (
    TopKBackend,
)
from vllm_torchtpu.kernels.qwen4_exp.decode_indexer import (
    _request_bounds,
    select_decode_blocks,
)
from vllm_torchtpu.kernels.qwen4_exp.qsa import (
    _token_metadata,
    _update_indexer,
    bf16,
    produce,
    quantize_kv,
)

# Performance scheduling cap; SMEM capacity is checked separately.
RPA_MAX_QUERY_BATCH = 64


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
    k_scale=1.0,
    v_scale=1.0,
):
    from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

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
    dummy_k = jnp.zeros((q.shape[0], kv_heads, query.shape[-1]), cache.dtype)
    decode = configs.BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=bkv_size, batch_size=8)
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
    )
    out = out[jnp.minimum(cu_q_lens[:-1], len(q) - 1), :, : q.shape[-1]]
    out = bf16(
        bf16(out).astype(jnp.float32)
        * bf16(jax.nn.sigmoid(gate.astype(jnp.float32))).astype(jnp.float32)
    )
    out = jnp.where((lengths > 0)[:, None, None], out, 0)
    return out, next_cache.reshape(cache.shape)


def rpa_query_batch_size(
    page_columns: int,
    *,
    query: jax.Array | jax.ShapeDtypeStruct,
    cache: jax.Array | jax.ShapeDtypeStruct,
    kv_heads: int,
    ratio: int,
) -> int:
    from vllm_torchtpu.kernels.experimental.batched_rpa import configs, schedule

    if page_columns < 1 or ratio < 1:
        raise ValueError("QSA page width and compression ratio must be positive")
    info = pltpu.get_tpu_info()
    batch = RPA_MAX_QUERY_BATCH
    while batch:
        cfgs = configs.RpaConfigs(
            block=configs.BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=256, batch_size=8),
            model=configs.ModelConfigs(
                num_q_heads=query.shape[1],
                num_kv_heads=kv_heads,
                head_dim=cache.shape[-1],
                mask_value=float(jnp.finfo(jnp.float32).min),
            ),
            serve=configs.ServingConfigs(
                num_seqs=batch,
                page_size=ratio,
                total_q_tokens=batch,
                num_page_indices=batch * page_columns,
                dtype_q=jnp.dtype(jnp.float32),
                dtype_kv=cache.dtype,
                dtype_out=jnp.dtype(jnp.float32),
                skip_kv_update=True,
            ),
            mode=configs.RpaCase.DECODE,
            vmem_limit_bytes=info.vmem_capacity_bytes,
        )
        scratch = schedule.BaseMetadataComputer.get_rpa_schedule(cfgs).scratch_shapes()
        scratch_bytes = sum(
            math.prod(ref.shape) * ref.dtype.itemsize
            for ref in jax.tree.leaves(scratch)
        )
        scalar_shapes = (
            (batch + 1,),
            (batch,),
            (batch,),
            (batch,),
            (batch * page_columns,),
        )
        scalar_bytes = sum(
            math.prod(shape) * jnp.dtype(jnp.int32).itemsize for shape in scalar_shapes
        )
        # Schedule generation has distribution[3] in place of the page table.
        scalar_bytes = max(
            scalar_bytes, (4 * batch + 1 + 3) * jnp.dtype(jnp.int32).itemsize
        )
        if scratch_bytes + scalar_bytes <= info.smem_capacity_bytes:
            return batch
        batch //= 2
    raise ValueError(
        "One QSA page-table row and the RPA schedule exceed SMEM capacity; "
        "reduce the sparse page budget."
    )


def rpa_attention(
    q, gate, blocks, lengths, table, cache, *, kv_heads, ratio, k_scale=1.0, v_scale=1.0
):
    batch_size = rpa_query_batch_size(
        blocks.shape[1] + 1, query=q, cache=cache, kv_heads=kv_heads, ratio=ratio
    )
    attend = functools.partial(
        _rpa_attention_batch,
        kv_heads=kv_heads,
        ratio=ratio,
        k_scale=k_scale,
        v_scale=v_scale,
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
):
    from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (
        block_sparse_gqa_attention,
    )
    from vllm_torchtpu.kernels.qwen4_exp.prefill_tiles import (
        get_sparse_prefill_tiles,
    )
    from vllm_torchtpu.kernels.sparse_gqa_prefill.adaptive import (
        adaptive_prefill,
        is_supported_by_adaptive_prefill,
    )

    if is_supported_by_adaptive_prefill(
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
    ):
        return adaptive_prefill(query, gate, blocks, lengths, table, cache, starts)

    query_block, kv_tile = get_sparse_prefill_tiles(
        query, cache, kv_heads=kv_heads, ratio=ratio
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
    request_distribution: jax.Array,
    kv_heads,
    ratio,
    k_scale=1.0,
    v_scale=1.0,
):
    """Use Runner order: decode/verify prefix, then prefill in original rows."""
    tokens = len(query)
    if tokens == 0:
        return jnp.zeros_like(query), cache
    prefix_tokens, active_tokens = _request_bounds(starts, request_distribution, tokens)
    options = dict(kv_heads=kv_heads, ratio=ratio, k_scale=k_scale, v_scale=v_scale)

    def decode_batch(_):
        return rpa_attention(query, gate, blocks, lengths, table, cache, **options)

    def prefill_batch(_):
        positions = jnp.arange(tokens, dtype=jnp.int32)
        suffix = (positions >= prefix_tokens) & (positions < active_tokens)
        output, owner = _prefill_attention(
            query,
            gate,
            blocks,
            jnp.where(suffix, lengths, 0),
            table,
            cache,
            jnp.clip(starts, prefix_tokens, active_tokens),
            **options,
        )
        batch = min(
            tokens,
            rpa_query_batch_size(
                blocks.shape[1] + 1,
                query=query,
                cache=cache,
                kv_heads=kv_heads,
                ratio=ratio,
            ),
        )

        def prefix(index, state):
            output, owner = state
            positions = index * batch + jnp.arange(batch, dtype=jnp.int32)
            valid = positions < prefix_tokens
            rows = jnp.minimum(positions, tokens - 1)
            result, owner = _rpa_attention_batch(
                jnp.where(valid[:, None, None], query[rows], 0),
                gate[rows],
                jnp.where(valid[:, None], blocks[rows], -1),
                jnp.where(valid, lengths[rows], 0),
                table[rows],
                owner,
                **options,
            )
            destination = jnp.where(valid, positions, tokens)
            return output.at[destination].set(result, mode="drop"), owner

        return jax.lax.fori_loop(
            0, (prefix_tokens + batch - 1) // batch, prefix, (output, owner)
        )

    return jax.lax.cond(
        active_tokens == 0,
        lambda _: (jnp.zeros_like(query), cache),
        lambda _: jax.lax.cond(
            prefix_tokens == active_tokens, decode_batch, prefill_batch, None
        ),
        None,
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
    request_distribution: jax.Array,
    compress_ratio: int,
    token_topk: int,
    norm_eps: float,
    rotary_dim: int,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    *,
    topk_backend: TopKBackend | None = None,
    scheduling_group_namespace: int | None = None,
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
        starts=starts,
        context=context,
        request_distribution=request_distribution,
        topk_backend=topk_backend,
        scheduling_group_namespace=scheduling_group_namespace,
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
        request_distribution=request_distribution,
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


# Bind the protocol object outside the Torch schema: jax_op accepts only POD
# static arguments. Factories may bind a different immutable backend with partial.
qsa_step = functools.partial(qsa_step, topk_backend=None)
