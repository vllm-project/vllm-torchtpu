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
import math

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp.decode_indexer import _request_bounds
from vllm_torchtpu.kernels.qwen4_exp.qsa import (
    bf16,
    quantize_kv,
)
from vllm_torchtpu.kernels.sparse_gqa_decode import batching
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


def virtual_page_metadata(blocks, lengths, table, *, ratio: int, page_size: int):
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


def _rpa_attention_in_chunks(
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
    active_tokens: jax.Array | None = None,
    initial_output: jax.Array | None = None,
):
    attend = functools.partial(
        _rpa_attention_batch,
        kv_heads=kv_heads,
        ratio=ratio,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    return batching.attend_in_chunks(
        attend,
        q,
        gate,
        blocks,
        lengths,
        table,
        cache,
        active_tokens=active_tokens,
        initial_output=initial_output,
        batch_size=batching.rpa_query_batch_size(
            blocks.shape[1] + 1,
            query=q,
            cache=cache,
            kv_heads=kv_heads,
            ratio=ratio,
        ),
    )


def rpa_attention(
    query,
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
    active_tokens: jax.Array | None = None,
    initial_output: jax.Array | None = None,
):
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
            active_tokens=active_tokens,
            initial_output=initial_output,
            k_scale=k_scale,
            v_scale=v_scale,
        )
    return _rpa_attention_in_chunks(
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
        active_tokens=active_tokens,
        initial_output=initial_output,
    )


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
        return adaptive_prefill(
            query,
            gate,
            blocks,
            lengths,
            table,
            cache,
            starts,
            kv_heads=kv_heads,
            ratio=ratio,
        )

    mask_tile = ratio * jnp.iinfo(jnp.int32).bits
    if math.gcd(cache.shape[1], mask_tile) % pltpu.get_tpu_info().num_lanes:
        return _rpa_attention_in_chunks(
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
            active_tokens=jnp.max(
                jnp.where(lengths > 0, jnp.arange(len(query)) + 1, 0)
            ),
        )

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

    positions = jnp.arange(tokens, dtype=jnp.int32)
    suffix = (positions >= prefix_tokens) & (positions < active_tokens)
    # Schedule both request segments. The leaves bound attention work using
    # their live metadata; neither segment owns or updates the written cache.
    prefill, _ = _prefill_attention(
        query,
        gate,
        blocks,
        jnp.where(suffix, lengths, 0),
        table,
        cache,
        jnp.clip(starts, prefix_tokens, active_tokens),
        **options,
    )
    output, _ = rpa_attention(
        query,
        gate,
        blocks,
        jnp.where(positions < prefix_tokens, lengths, 0),
        table,
        cache,
        active_tokens=prefix_tokens,
        initial_output=prefill,
        **options,
    )
    return output, cache


def selected_tokens(blocks, lengths, token_topk, compress_ratio):
    selected = jnp.where(
        blocks[..., None] >= 0,
        blocks[..., None] * compress_ratio + jnp.arange(compress_ratio),
        -1,
    ).reshape(len(blocks), token_topk)
    selected = jnp.pad(selected, ((0, 0), (0, compress_ratio - 1)), constant_values=-1)
    count = jnp.minimum(lengths // compress_ratio, blocks.shape[1])
    start = count * compress_ratio
    # Preserve scatter's negative-index normalization as well as idle rows.
    start = jnp.where(start < 0, start + selected.shape[1], start)
    tail = jnp.arange(selected.shape[1])[None, :] - start[:, None]
    token = (lengths // compress_ratio * compress_ratio)[:, None] + tail
    return jnp.where(
        (tail >= 0) & (tail < compress_ratio - 1),
        jnp.where(token < lengths[:, None], token, -1),
        selected,
    )
