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

"""Sparse decode adapter; shape/dtype eligibility is enforced by the public API."""

import dataclasses
import functools

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.qwen4_exp.rounding import bf16
from vllm_torchtpu.kernels.sparse_gqa_decode import schedule
from vllm_torchtpu.kernels.sparse_gqa_decode.batching import attend_in_chunks
from vllm_torchtpu.kernels.sparse_gqa_decode.fused_kernel import decode_kernel
from vllm_torchtpu.kernels.sparse_gqa_decode.metadata import virtual_page_metadata
from vllm_torchtpu.kernels.sparse_gqa_decode.planning import DecodeLaunch


def _attend(q, gate, blocks, lengths, table, cache, *, launch, k_scale, v_scale):
    batch, heads, dim = q.shape
    cfgs = launch.config_for_batch(batch)
    kv_heads, ratio = cfgs.model.num_kv_heads, cfgs.serve.page_size
    pages, kv_lens = virtual_page_metadata(
        blocks, lengths, table, ratio=ratio, page_size=cache.shape[1]
    )
    cfgs = dataclasses.replace(
        cfgs, serve=dataclasses.replace(cfgs.serve, scale_k=k_scale, scale_v=v_scale)
    )
    # Keep request order; the compact schedule omits inactive requests.
    query = jnp.where((lengths > 0)[:, None, None], q, 0).astype(jnp.float32)
    query = query.reshape(batch, kv_heads, heads // kv_heads, 1, dim).swapaxes(0, 1)
    virtual_cache = cache.reshape(-1, ratio, *cache.shape[2:])
    # RPA's buffer binding accepts a new-KV operand even when cache updates and
    # new-KV DMAs are disabled. No dummy K/V preparation or concatenation needed.
    unused_new_kv = jnp.zeros((batch, *cache.shape[2:]), cache.dtype)
    cu_q_lens = jnp.arange(batch + 1, dtype=jnp.int32)
    q_lens = (kv_lens > 0).astype(jnp.int32)
    metadata = schedule.generate(cu_q_lens, kv_lens, cfgs)
    out, next_cache = decode_kernel(
        cu_q_lens=cu_q_lens,
        compressed_query_offsets=kv_lens - q_lens,
        q_lens=q_lens,
        pages=pages.reshape(-1),
        metadata=metadata,
        query=query,
        unused_new_kv=unused_new_kv,
        cache=virtual_cache,
        gate=gate,
        cfgs=cfgs,
        policy=launch.policy,
    )
    out = bf16(out.swapaxes(0, 1).reshape(q.shape))
    out = jnp.where((lengths > 0)[:, None, None], out, 0)
    return out, next_cache.reshape(cache.shape)


def execute_decode(
    q: jax.Array,
    gate: jax.Array,
    blocks: jax.Array,
    lengths: jax.Array,
    table: jax.Array,
    cache: jax.Array,
    *,
    launch: DecodeLaunch,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    attend = functools.partial(_attend, launch=launch, k_scale=k_scale, v_scale=v_scale)
    return attend_in_chunks(
        attend,
        q,
        gate,
        blocks,
        lengths,
        table,
        cache,
        batch_size=launch.chunk_queries,
    )


def qsa_decode_attention(
    q: jax.Array,
    gate: jax.Array,
    blocks: jax.Array,
    lengths: jax.Array,
    table: jax.Array,
    cache: jax.Array,
    *,
    kv_heads: int,
    ratio: int,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Retain the existing module entry; internal dispatch uses execute_decode."""
    from . import qsa_decode_attention as checked_decode

    return checked_decode(
        q,
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
