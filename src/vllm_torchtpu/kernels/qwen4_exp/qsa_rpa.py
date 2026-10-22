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
"""QSA with native HEAD_ALONG_SUBLANE storage and fused sparse decode.

Scheduler pages remain intact. Selected compression groups become virtual RPA
pages by a view; the returned cache is threaded through donation, never dropped.
Prefill reuses KV tiles through block-sparse attention; decode reuses the
batched RPA compute body.
"""

import functools

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.topk_backend import (
    TopKBackend,
)
from vllm_torchtpu.kernels.qwen4_exp.decode_indexer import select_decode_blocks
from vllm_torchtpu.kernels.qwen4_exp.qsa import (
    _token_metadata,
    _update_indexer,
    bf16,
    produce,
    quantize_kv,
)
from vllm_torchtpu.kernels.sparse_gqa_decode.batching import (
    attend_in_chunks,
)
from vllm_torchtpu.kernels.sparse_gqa_decode.batching import (
    rpa_query_batch_size as rpa_query_batch_size,
)
from vllm_torchtpu.kernels.sparse_gqa_decode.metadata import (
    virtual_page_metadata as sparse_page_metadata,
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
    """Map sparse pages and pack the legacy RPA request metadata."""
    pages, kv_lens = sparse_page_metadata(
        blocks, lengths, table, ratio=ratio, page_size=page_size
    )
    cu_q_lens = jnp.concatenate(
        (jnp.zeros((1,), jnp.int32), jnp.cumsum((lengths > 0).astype(jnp.int32)))
    )
    return pages, kv_lens, cu_q_lens, jnp.full((3,), len(blocks), jnp.int32)


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


def rpa_attention(
    q, gate, blocks, lengths, table, cache, *, kv_heads, ratio, k_scale=1.0, v_scale=1.0
):
    attend = functools.partial(
        _rpa_attention_batch,
        kv_heads=kv_heads,
        ratio=ratio,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    return attend_in_chunks(attend, q, gate, blocks, lengths, table, cache)


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
):
    """Use query-block reuse for large buckets, per-query attention otherwise.

    T<=request capacity can include multi-token prefill. Each row still uses
    its own request table, selected groups and causal length on that route.
    T>padded request capacity is a static scheduling heuristic, not a semantic
    mode classifier, safety requirement or proven profitability boundary.
    Per-request starts/context and per-query causal lengths are authoritative.
    """
    if len(query) > len(starts) - 1:
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
    from vllm_torchtpu.kernels.sparse_gqa_decode import plan_decode
    from vllm_torchtpu.kernels.sparse_gqa_decode.decode import execute_decode

    launch = plan_decode(
        query,
        gate,
        blocks,
        lengths,
        table,
        cache,
        kv_heads=kv_heads,
        ratio=ratio,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    if launch is not None:
        return execute_decode(
            query,
            gate,
            blocks,
            lengths,
            table,
            cache,
            launch=launch,
            k_scale=k_scale,
            v_scale=v_scale,
        )
    return rpa_attention(
        query,
        gate,
        blocks,
        lengths,
        table,
        cache,
        kv_heads=kv_heads,
        ratio=ratio,
        k_scale=k_scale,
        v_scale=v_scale,
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
    *,
    topk_backend: TopKBackend | None = None,
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
        topk_backend=topk_backend,
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


# Bind the protocol object outside the Torch schema: jax_op accepts only POD
# static arguments. Factories may bind a different immutable backend with partial.
qsa_step = functools.partial(qsa_step, topk_backend=None)
