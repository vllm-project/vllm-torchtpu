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
import functools
import math
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax.experimental import shard_map
from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel as splash,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask as mask_lib,
)
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

import vllm_torchtpu.kernels.experimental.batched_rpa.wrapper as rpa_batched
import vllm_torchtpu.kernels.mla.sparse.kernel as sparse_mla_kernel
import vllm_torchtpu.kernels.mla.v2.kernel as mla_v2_kernel
import vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel as rpa_default
import vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel_hd64 as rpa_hd64
from vllm_torchtpu import envs
from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk
from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs as batched_rpa_configs,
)
from vllm_torchtpu.kernels.experimental.batched_rpa import (
    prologue as rpa_prologue,
)
from vllm_torchtpu.kernels.flash_attention.kernel import flash_attention
from vllm_torchtpu.kernels.mla import dispatch as mla_dispatch
from vllm_torchtpu.kernels.mla import kv_cache_utils
from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    SparseMLAKVCacheSpec as SparseMLAKVCacheSpec,
)
from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    update_sparse_mla_kv_cache as update_sparse_mla_kv_cache,
)
from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    update_sparse_mla_kv_cache_dcp as update_sparse_mla_kv_cache_dcp,
)
from vllm_torchtpu.kernels.mla.sparse import native_sc_gather
from vllm_torchtpu.kernels.mla.v2.tuned_params import TuningKey, get_tuned_params
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.glm_dsa_dispatch import (
    DCP_MERGE_CHUNKED_DECODE_SPLITS,
    DCP_MERGE_CHUNKED_LSE_AG_MAX_TOKENS,
    DCP_MERGE_CHUNKED_MIN_TOKENS,
    DCP_MERGE_CHUNKED_PREFILL_SPLITS,
    DCP_MLA_MERGE_PIECES,
    DCP_MLA_MERGE_PIECES_MIN_TOKENS,
    DCP_MLA_SMALL_CHUNK_BATCH,
    DCP_MLA_SMALL_CHUNK_TOKENS,
    MLA_DECODE_ROW_WIDTHS,
    MLA_FUSED_SMALL_A2A_MERGE_MAX_TOKENS,
    MLA_MIXED_LIVE_WIDTHS,
    bucket_gate_edge,
    fused_small_a2a_bucket_config,
    glm_dsa_v7x_active,
    resolve_dcp_mla_topk_cap,
    resolve_dcp_mla_topk_cap_chunk,
)
from vllm_torchtpu.utils import get_megacore

logger = init_logger(__name__)

# Based on experiments on v5e, 256x1024 results in smem oom but 128x1024 not.
# TODO: Adjust this based on TPU version.
MAX_ALLOWED_PAGE_INDICES_N = 128 * 1024

# Default and experimental batched RPA kernels are loaded unconditionally.
# Selection happens per attention layer via the `use_batched_rpa` flag plumbed
# from `PallasAttentionBackendImpl` / `PallasBatchedRPAAttentionBackendImpl`.

ragged_paged_attention = rpa_default.ragged_paged_attention
ragged_paged_attention_batched = rpa_batched.ragged_paged_attention
get_kv_cache_shape = rpa_default.get_kv_cache_shape

ragged_paged_attention_hd64 = rpa_hd64.ragged_paged_attention_hd64
get_kv_cache_shape_hd64 = rpa_hd64.get_kv_cache_shape

mla_ragged_paged_attention = mla_v2_kernel.mla_ragged_paged_attention
sparse_mla_ragged_paged_attention = sparse_mla_kernel.sparse_ragged_paged_attention


def sparse_mla_chunk_kwargs() -> dict[str, int]:
    """Kernel kwargs for the sparse MLA gather + attention chunk size.

    Empty unless `TPU_SPARSE_MLA_CHUNK_TOKENS` is set, so the kernel keeps its
    default chunk. The chunk must be a multiple of the attention kernel's batch
    size: the attention grid walks a chunk in whole batches, and the SparseCore
    gather splits a chunk's `chunk * topk` rows into whole owner batches.
    """
    chunk = envs.TPU_SPARSE_MLA_CHUNK_TOKENS
    if chunk is None:
        return {}
    batch = sparse_mla_kernel.DEFAULT_ATTENTION_KERNEL_BATCH_SIZE
    if chunk % batch:
        raise ValueError(
            f"TPU_SPARSE_MLA_CHUNK_TOKENS={chunk} must be a multiple of the "
            f"sparse MLA attention kernel batch size ({batch})."
        )
    return {"gather_and_attention_chunk_size": chunk}


def _resolve_sparse_mla_split_q(ql_nope: jax.Array, q_pe: jax.Array) -> bool:
    want_split = bool(envs.TPU_MLA_SPARSE_SPLIT_Q or glm_dsa_v7x_active())
    if not want_split:
        return False
    if sparse_mla_kernel.supports_split_q(ql_nope, q_pe):
        return True
    if envs.TPU_MLA_SPARSE_SPLIT_Q:
        logger.warning_once(
            "TPU_MLA_SPARSE_SPLIT_Q=1 requires 128-aligned q_nope dim and "
            "packed per-device query heads, got ql_nope (heads, dim)=%s/%s "
            "and q_pe (heads, dim)=%s/%s; falling back to [ql_nope, q_pe] concat.",
            tuple(ql_nope.shape[1:]),
            ql_nope.dtype,
            tuple(q_pe.shape[1:]),
            q_pe.dtype,
        )
    return False


def dcp_mla_topk_cap(dcp_size: int) -> int:
    """Effective per-rank DCP top-k gather cap (0 means uncapped)."""
    return resolve_dcp_mla_topk_cap(envs.TPU_DCP_MLA_TOPK_CAP, dcp_size)


def sharded_flash_attention(
    mesh: Mesh,
    causal: bool = True,
    sm_scale: float | None = None,
    vmem_limit_bytes: int | None = None,
) -> Callable[..., Any]:
    in_specs = (
        P(None, "model", None, None),  # q
        P(None, "model", None, None),  # k
        P(None, "model", None, None),  # v
        P(),  # segment_ids
    )
    out_specs = P(None, "model", None, None)

    def _flash_attention(q, k, v, segment_ids):
        return flash_attention(
            q,
            k,
            v,
            segment_ids=segment_ids,
            sm_scale=sm_scale,
            causal=causal,
            vmem_limit_bytes=vmem_limit_bytes,
        )

    return jax.jit(
        shard_map.shard_map(
            _flash_attention,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_rep=False,
        )
    )


def sharded_paged_attention(
    mesh: Mesh,
    attn_logits_soft_cap: float | None = None,
) -> Callable[..., Any]:
    """Shards GQA PagedAttention along KV heads."""
    in_specs = (
        P(None, "model", None),  # q
        P("model", None, None, None),  # k
        P("model", None, None, None),  # v
        P(),  # lengths
        P(),  # page_indices
    )
    out_specs = P(None, "model", None)

    def _paged_attention_fn(q, k, v, lengths, page_indices):
        if page_indices.size > MAX_ALLOWED_PAGE_INDICES_N:
            raise ValueError(
                "This will result in smem OOM. Use `paged_attention_with_guarded_smem` "
                "to run with minibatches."
            )
        return paged_attention(
            q,
            k,
            v,
            lengths,
            page_indices,
            attn_logits_soft_cap=attn_logits_soft_cap,
            pages_per_compute_block=min(
                16, page_indices.shape[1]
            ),  # 512 / page_size:32,
            megacore_mode="kv_head" if get_megacore() else None,
        )

    return jax.jit(
        shard_map.shard_map(
            _paged_attention_fn,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_rep=False,
        )
    )


# TODO(xiangxu): merge this with sharded_paged_attention
@functools.partial(jax.jit, static_argnums=[0])
def paged_attention_with_guarded_smem(
    paged_attention_kernel: Callable,
    q: jax.Array,
    k_pages: jax.Array,
    v_pages: jax.Array,
    lengths: jax.Array,
    page_indices: jax.Array,
):
    # Paged attention kernel stores `lengths` (batch_size * 4 bytes) and
    # `page_indices` (batch_size * num_blocks_per_seq * 4 bytes) in SMEM.
    # Capacity of SMEM is quite limited which is also TPU version dependent.
    # Models with higher context length or higher batch size, can cause OOM in SMEM.
    # There are two solutions:
    # 1. Reduce blocks per seq by increasing page size.
    # 2. Splitting the batch into several minibatches
    #    (Higher perf based on my benchmark).

    batch_size, blocks_per_seq = page_indices.shape

    if page_indices.size <= MAX_ALLOWED_PAGE_INDICES_N:
        return paged_attention_kernel(q, k_pages, v_pages, lengths, page_indices)

    mini_batch_size = MAX_ALLOWED_PAGE_INDICES_N // blocks_per_seq

    # If batch_size is not disible by mini_batch_size,
    # we set mini_batch_size to a smaller value, i.e GCD,
    # which will trigger more kernel launches but it's fine.
    # TODO: Fix --decode_seqs_padding with this limitation.
    mini_batch_size = math.gcd(batch_size, mini_batch_size)

    num_kernel_launches = batch_size // mini_batch_size

    outputs = jnp.zeros_like(q).reshape(
        (num_kernel_launches, mini_batch_size, *q.shape[1:])
    )
    q = q.reshape((num_kernel_launches, mini_batch_size, *q.shape[1:]))
    seq_lens = lengths.reshape((num_kernel_launches, mini_batch_size))
    block_indices = page_indices.reshape(
        (num_kernel_launches, mini_batch_size, page_indices.shape[1])
    )

    for i in range(num_kernel_launches):
        outputs = outputs.at[i].set(
            paged_attention_kernel(
                q[i], k_pages, v_pages, seq_lens[i], block_indices[i]
            )
        )

    outputs = outputs.reshape((batch_size, *outputs.shape[2:]))

    return outputs


# ruff: noqa: E741
def update_cache(
    is_prefill,
    cache,
    indices,
    operand,
    prefill_seq_len=None,
    sliding_window=None,
) -> jax.Array:
    # (8, 55640, 32, 128) (1, 8, 256, 128) -> K (8, 8, 32, 128)
    # I = B * T // S
    # k cache, operand

    B, K, T, H = operand.shape
    K_c, L, S, H = cache.shape
    assert K_c == K
    # NOTE: The cache updating is pretty tricky:
    # 1. The random access updating cache is not as performant as the slice updating.
    #    If the random access is necessary, make sure the indexing count is
    #    as small as possible.
    # 2. The random access updating may trigger extra tranpose (memory copy) of cache,
    #    which is a disaster because the cache is huge. This is a data formatting op
    #    inserted by the XLA compiler and not well documented.
    # To mitigate the issues above:
    # For prefill:
    # We reshape the operand so that we can update the cache in block wise, which only
    # requires the block indices.
    # For decode:
    # We reshape the cache so that we can update the cache in token wise, which only
    # requires the token indices (block_id + offset).
    if is_prefill:
        # In the case of sliding window, we should select sliding_window tokens from
        # actual prompt, not from the padded tokens.
        if sliding_window and sliding_window < T:
            assert B == 1
            start_index = jax.lax.max(0, prefill_seq_len - sliding_window)
            operand = jax.lax.dynamic_slice_in_dim(
                operand, start_index, sliding_window, axis=2
            )  # TODO: @pooyam Perf check this.
            T = sliding_window

        I = B * T // S
        # cache: (K, L, S, H)
        # operand: (B, K, T, H) -> (K, I, S, H)
        # indices: (B, T // S) -> (I,)
        operand = jnp.swapaxes(operand, 0, 1).reshape(K, I, S, H)
        indices = indices.reshape(I)
        cache = cache.at[:, indices, :, :].set(operand)
    else:
        # cache: (K, L, S, H) -> (K, L * S, H)
        # operand: (B, K, 1, H) -> (K, B, H)
        # indices: (B,)
        cache = cache.reshape(K, L * S, H)
        operand = jnp.swapaxes(operand, 0, 1).reshape(K, B, H)
        # NOTE: `cache.[:, indices, :].set()` will trigger the extra tranpose of the
        # cache. The `jnp.arange(K)[..., None]` trick is to avoid it. WTF?
        cache = cache.at[jnp.arange(K)[..., None], indices, :].set(operand)
        cache = cache.reshape(K, L, S, H)
    return cache


@functools.partial(
    jax.jit, static_argnames=["window_size", "attn_logits_soft_cap", "is_mqa"]
)
def apply_splash(q, k, v, window_size, attn_logits_soft_cap, is_mqa) -> jax.Array:
    # q: (batch_size, num_heads, seq_len, head_dim)
    num_heads = q.shape[1]
    q_seq_len = q.shape[2]
    kv_seq_len = k.shape[2]
    assert kv_seq_len >= q_seq_len

    masks = [
        mask_lib.LocalMask(
            (q_seq_len, kv_seq_len), (window_size, 0), kv_seq_len - q_seq_len
        )
        for _ in range(num_heads)
    ]
    mask = mask_lib.MultiHeadMask(tuple(m for m in masks))
    block_sizes = splash.BlockSizes.get_default()

    if is_mqa:
        attn = splash.make_splash_mqa_single_device(
            mask, block_sizes=block_sizes, attn_logits_soft_cap=attn_logits_soft_cap
        )
    else:
        attn = splash.make_splash_mha_single_device(
            mask, block_sizes=block_sizes, attn_logits_soft_cap=attn_logits_soft_cap
        )
    attn = jax.vmap(attn)
    outputs = attn(q, k, v, None)

    return cast(jax.Array, outputs)


def sharded_splash_attention(
    mesh: Mesh,
    window_size: int | None = None,
    attn_logits_soft_cap: float | None = None,
    is_mqa: bool = False,
) -> Callable[..., Any]:
    in_specs = (
        P(None, "model", None, None),  # q
        P(None, "model", None, None),  # k
        P(None, "model", None, None),  # vx
    )
    out_specs = P(None, "model", None, None)
    return jax.jit(
        shard_map.shard_map(
            functools.partial(
                apply_splash,
                window_size=window_size,
                attn_logits_soft_cap=attn_logits_soft_cap,
                is_mqa=is_mqa,
            ),
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_rep=False,
        )
    )


def sharded_ragged_paged_attention(
    mesh: Mesh,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    attention_sink: jax.Array | None,
    sm_scale: float,
    attention_chunk_size: int | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    skip_kv_update: bool = False,
    rpa_func: Callable = ragged_paged_attention,
    soft_cap: float | None = None,
    shard: bool = True,
    kv_block_cap: int | None = None,
    use_causal_mask: bool = True,
    kv_layout: batched_rpa_configs.KVLayout | None = None,
):
    """Shards along KV heads."""

    qkv_spec = P(None, "model", None)
    use_hd64 = q.shape[-1] == 64
    layout_kwargs: dict[str, Any] = {}
    kv_cache_spec = P(None, None, "model", None, None)
    page_size_axis = 1
    if kv_layout is not None and not use_hd64:
        layout_kwargs["kv_layout"] = rpa_batched.configs.KVLayout(kv_layout)
        if kv_layout == batched_rpa_configs.KVLayout.SEQ_ALONG_LANE:
            kv_cache_spec = P(None, "model", None, None, None)
            page_size_axis = 4
    in_specs = (
        qkv_spec,  # q
        qkv_spec,  # k
        qkv_spec,  # v
        kv_cache_spec,  # kv cache
        P(None),  # kv_lens
        P(None),  # page_indices
        P(None),  # cu_q_lens
        P(None),  # distribution
    )
    out_specs = (qkv_spec, kv_cache_spec)

    args = (q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution)

    if use_hd64:
        # Batched RPA has no hd64 variant; head_dim==64 always uses default.
        func = functools.partial(
            ragged_paged_attention_hd64, strict_sliding_window=True
        )
    else:
        func = rpa_func

    if attention_sink is not None:
        if not use_hd64:
            raise NotImplementedError(
                "Attention sink support is only available when head_dim==64"
            )

        in_specs += (P("model"),)
        args += (attention_sink,)

    # Speculative decoding draft-only VMEM relief: cap the KV-fetch block on the
    # local path.
    block_kwargs: dict[str, Any] = {}
    if not shard and kv_block_cap is not None and not use_hd64:
        page_size = kv_cache.shape[page_size_axis]
        max_num_seqs = kv_lens.shape[0]
        pages_per_seq = page_indices.shape[0] // max_num_seqs
        cap_tokens = max(page_size, (kv_block_cap // page_size) * page_size)

        def _capped(case):
            bs = rpa_default.get_default_block_sizes(
                q.dtype,
                kv_cache.dtype,
                q.shape[1],  # actual_num_q_heads
                k.shape[1],  # actual_num_kv_heads
                q.shape[2],  # head_dim
                page_size,
                q.shape[0],  # max_num_tokens
                max_num_seqs,
                pages_per_seq,
                case=case,
            )
            bkv = min(bs["bkv_sz"], cap_tokens)
            bkv_csz = min(bs["bkv_csz"], bkv)
            return (bs["bq_sz"], bkv, bs["bq_csz"], bkv_csz)

        block_kwargs["d_block_sizes"] = _capped(rpa_default.RpaCase.DECODE)
        block_kwargs["m_block_sizes"] = _capped(rpa_default.RpaCase.MIXED)
        if attention_chunk_size is not None:
            block_kwargs["p_block_sizes"] = _capped(rpa_default.RpaCase.PREFILL)

    def _ragged_paged_attention(*args):
        return func(
            *args,
            sm_scale=sm_scale,
            sliding_window=attention_chunk_size,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            soft_cap=soft_cap,
            skip_kv_update=skip_kv_update,
            use_causal_mask=use_causal_mask,
            **layout_kwargs,
            **block_kwargs,
        )

    if not shard:
        # Local per-worker path: run the kernel directly on the worker's device.
        return _ragged_paged_attention(*args)

    return shard_map.shard_map(
        _ragged_paged_attention,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_rep=False,
    )(*args)


def sharded_ragged_paged_attention_from_qkv(
    mesh: Mesh,
    qkv: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    q_norm_weight: jax.Array,
    k_norm_weight: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
    sm_scale: float,
    k_scale: float,
    v_scale: float,
    attention_chunk_size: int | None = None,
    soft_cap: float | None = None,
    **rpa_kwargs: Any,
) -> tuple[jax.Array, jax.Array]:
    """Run fused QKV prologue + batched RPA over a single-device model axis."""
    shards = mesh.shape["model"]
    if shards != 1:
        raise NotImplementedError(
            f"fused qkv rows need a one-device model axis, got {shards} shards"
        )
    q_heads, kv_heads = num_q_heads, num_kv_heads
    kv_layout = rpa_kwargs.get(
        "kv_layout", batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE
    )
    seq_along_lane = kv_layout == batched_rpa_configs.KVLayout.SEQ_ALONG_LANE
    kv_cache_spec = (
        P(None, "model", None, None, None)
        if seq_along_lane
        else P(None, None, "model", None, None)
    )
    page_size = kv_cache.shape[4] if seq_along_lane else kv_cache.shape[1]
    prologue_dma_stores = rpa_kwargs.pop("prologue_dma_stores", False)
    prologue_fp8_queries = rpa_kwargs.pop("prologue_fp8_queries", False)

    def _attend(qkv, cos, sin, wq, wk, kv_cache, kv_lens, pages, cu_q_lens, dist):
        prepared = rpa_prologue.qkv_to_rpa_inputs(
            qkv,
            cos,
            sin,
            wq,
            wk,
            num_q_heads=q_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            eps=eps,
            k_scale=k_scale,
            v_scale=v_scale,
            kv_dtype=kv_cache.dtype,
            dma_stores=prologue_dma_stores,
            fp8_queries=prologue_fp8_queries,
            kv_layout=kv_layout,
            page_size=page_size,
            shared_kv_rotary_dims=rpa_kwargs.get("shared_kv_rotary_dims", 0),
        )
        tokens = qkv.shape[0]
        q = jnp.zeros((tokens, q_heads, head_dim), qkv.dtype)
        kv = jnp.zeros((tokens, kv_heads, head_dim), kv_cache.dtype)
        return rpa_batched.ragged_paged_attention(
            q,
            kv,
            kv,
            kv_cache,
            kv_lens,
            pages,
            cu_q_lens,
            dist,
            sm_scale=sm_scale,
            sliding_window=attention_chunk_size,
            k_scale=k_scale,
            v_scale=v_scale,
            soft_cap=soft_cap,
            prepared_inputs=prepared,
            **rpa_kwargs,
        )

    replicated = P(None, None)
    out_spec = (
        P(None, "model", None)
        if rpa_kwargs.get("output_quant_dtype") is None
        else (P(None, "model"), replicated)
    )
    return shard_map.shard_map(
        _attend,
        mesh=mesh,
        in_specs=(
            P(None, "model"),  # qkv: the worker's q | k | v heads
            replicated,  # cos
            replicated,  # sin
            P(None),  # q norm weight
            P(None),  # k norm weight
            kv_cache_spec,
            P(None),  # kv_lens
            P(None),  # page_indices
            P(None),  # cu_q_lens
            P(None),  # distribution
        ),
        out_specs=(out_spec, kv_cache_spec),
        check_rep=False,
    )(
        qkv,
        cos,
        sin,
        q_norm_weight,
        k_norm_weight,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
    )


def attention_bundled(
    kv_cache_bundle: jax.Array,
    layer_idx: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    attention_metadata: AttentionMetadata,
    mesh: Mesh,
    head_dim_original: int | None = None,
    attention_chunk_size: int | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    sm_scale: float | None = None,
    soft_cap: float | None = None,
    use_causal_mask: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Dispatches ragged paged attention over the bundled block-major KV cache
    across the TPU mesh.

    Args:
        kv_cache_bundle: Full-model KV cache bundle of shape
            `[num_pages, num_layers, ...]`.
        layer_idx: Dynamic scalar integer tensor specifying the target layer's
            bundle index.
        q: Query tensor of shape `[num_tokens, num_heads, head_dim]`.
        k: Key tensor of shape `[num_tokens, num_kv_heads, head_dim]`.
        v: Value tensor of shape `[num_tokens, num_kv_heads, head_dim]`.
        attention_metadata: Metadata containing sequence lengths, block tables, and
            batch bounds.
        mesh: TPU device mesh for Tensor Parallelism (TP).
        head_dim_original: Unpadded head dimension, if different from q.shape[-1].
        attention_chunk_size: Optional sliding window / chunk size.
        q_scale: Optional FP8 query scale.
        k_scale: Optional FP8 key scale.
        v_scale: Optional FP8 value scale.
        sm_scale: Softmax temperature scale.
        soft_cap: Optional attention logits soft-capping threshold.
        use_causal_mask: Whether to apply lower-triangular causal masking.

    Returns:
        A tuple of (new_bundle, output), where new_bundle aliases the input bundle
        in-place and output is the computed attention result tensor.
    """
    from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import (
        ragged_paged_attention_bundled,
    )

    if head_dim_original is None:
        head_dim_original = cast(int, q.shape[-1])
    md = attention_metadata
    if sm_scale is None:
        sm_scale = head_dim_original**-0.5

    def _run(
        q,
        k,
        v,
        kv_cache_bundle,
        layer_idx,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
    ):
        output, new_bundle = ragged_paged_attention_bundled(
            q,
            k,
            v,
            kv_cache_bundle,
            layer_idx,
            seq_lens,
            block_tables,
            query_start_loc,
            request_distribution,
            sm_scale=sm_scale,
            sliding_window=attention_chunk_size,
            soft_cap=soft_cap,
            use_causal_mask=use_causal_mask,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        return new_bundle, output

    # Shard across KV heads along the TPU "model" axis (Tensor Parallelism):
    # - Dense bundle (6D):
    #   [num_pages, num_layers, page_size, num_kv_heads, p, head_dim] -> shard dim 3.
    # The layer dimension (dim 1) remains unsharded.
    # Query, Key, and Value shard on the head dimension (dim 1); metadata is replicated;
    # layer_idx is a scalar.
    qkv_spec = P(None, "model", None)
    bundle_spec = P(None, None, None, "model", None, None)
    data_spec = P(None)
    in_specs = (
        qkv_spec,  # q
        qkv_spec,  # k
        qkv_spec,  # v
        bundle_spec,  # kv_cache_bundle (num_layers dim unsharded)
        P(),  # layer_idx (replicated scalar)
        data_spec,  # seq_lens
        data_spec,  # block_tables
        data_spec,  # query_start_loc
        data_spec,  # request_distribution
    )
    args = (
        q,
        k,
        v,
        kv_cache_bundle,
        layer_idx,
        md.seq_lens,
        md.block_tables,
        md.query_start_loc,
        md.request_distribution,
    )

    # Output partition specs mirror (new_bundle, output).
    out_specs = (bundle_spec, qkv_spec)
    return shard_map.shard_map(
        _run,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_rep=False,
    )(*args)


def attention(
    kv_cache: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    attention_metadata: AttentionMetadata,
    mesh: Mesh,
    head_dim_original: int | None = None,  # before padding,
    attention_chunk_size: int | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    sinks: jax.Array | None = None,
    skip_kv_update: bool = False,
    rpa_func: Callable = ragged_paged_attention,
    sm_scale: float | None = None,
    soft_cap: float | None = None,
    shard: bool = True,
    kv_block_cap: int | None = None,
    use_causal_mask: bool = True,
    kv_layout: batched_rpa_configs.KVLayout | None = None,
) -> tuple[jax.Array, jax.Array]:
    # T: seq_len
    # N: num_heads
    # K: num_kv_heads
    # D: hidden_size
    # H: head_dim
    # L: num_blocks
    # S: block_size

    # TODO(jevinjiang, cuiq): transpose q weight offline.
    # q: (T, N, H)
    # k,v: (T, K, H)

    if head_dim_original is None:
        head_dim_original = cast(int, q.shape[-1])

    if sm_scale is None:
        sm_scale = cast(float, head_dim_original**-0.5)

    md = attention_metadata

    # (T, N, H)
    output, kv_cache = sharded_ragged_paged_attention(
        mesh,
        q,
        k,
        v,
        kv_cache,
        cast(jax.Array, md.seq_lens),
        cast(jax.Array, md.block_tables),
        cast(jax.Array, md.query_start_loc),
        cast(jax.Array, md.request_distribution),
        sinks,
        sm_scale=sm_scale,
        attention_chunk_size=attention_chunk_size,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        skip_kv_update=skip_kv_update,
        rpa_func=rpa_func,
        kv_layout=kv_layout,
        soft_cap=soft_cap,
        shard=shard,
        kv_block_cap=kv_block_cap,
        use_causal_mask=use_causal_mask,
    )

    return kv_cache, output


def mla_attention(
    q_TNA: jax.Array,
    q_rope_TNH: jax.Array,
    k_SA: jax.Array,
    k_rope_SH: jax.Array,
    kv_cache: jax.Array,
    md: AttentionMetadata,
    mesh: Mesh,
    num_attention_heads: int,
    qk_nope_head_dim: int,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    sm_scale: float | None = None,
    use_causal_mask: bool = True,
    return_lse: bool = False,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
    """Main shared interface for Multi-Head Latent Attention (MLA).

    Computes sharded MLA paged attention and applies in-place KV cache updates across
    the device mesh using custom Pallas kernels.

    Args:
        q_TNA: NOPE query activations in token-major format `(T, N, A)`
            (`[num_tokens, num_heads, lkv_dim]`).
        q_rope_TNH: RoPE query activations in token-major format `(T, N, H)`
            (`[num_tokens, num_heads, rope_dim]`).
        k_SA: New compressed latent keys to insert into the KV cache `(S, A)`
            (`[num_tokens, lkv_dim]`).
        k_rope_SH: New RoPE keys to insert into the KV cache `(S, H)`
            (`[num_tokens, rope_dim]`).
        kv_cache: Persistent paged latent KV cache tensor residing across devices.
        md: Attention metadata containing block tables, sequence lengths, and
            layout indices.
        mesh: JAX execution device mesh dictating parallel shard routing.
        num_attention_heads: Total number of attention heads across the layer.
        qk_nope_head_dim: Inner compressed projection dimension (`lkv_dim`).
        q_scale: Optional scalar activation quantization scale for queries.
        k_scale: Optional scalar parameter quantization scale for latent keys.
        v_scale: Optional scalar parameter quantization scale for latent values.
        sm_scale: Softmax temperature scale factor.
        return_lse: Also return each (token, head)'s log-sum-exp of the scaled
            scores, f32 `(T, N)`, with which this output can be merged with the
            output of the same queries over a disjoint set of keys (context
            parallelism). The output and the cache are unchanged by the flag.

    Returns:
        Tuple of `(updated_kv_cache, output_TNA)` in token-major sequence
        layout `(T, N, D)`, plus `lse_TN` when `return_lse`.
    """
    in_specs = (
        P(None, "model", None),  # q_TNA
        P(None, "model", None),  # q_rope_TNH
        P(None, None),  # k_SA
        P(None, None),  # k_rope_SH
        P(None),  # kv_cache
        P(None),  # md.seq_lens
        P(None),  # md.block_tables
        P(None),  # md.query_start_loc
        P(None),  # md.distribution
    )
    out_specs = (
        P(None, "model", None),  # attn output in token-major sequence format (T, N, D)
        P(None),  # kv cache
    )
    if return_lse:
        out_specs = (*out_specs, P(None, "model"))  # lse (T, N)

    def _mla_ragged_paged_attention(
        q,
        q_rope,
        k,
        k_rope,
        cache,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
    ):
        max_num_tokens = q.shape[0]  # q is in token-major sequence format (T, N, A)
        actual_r_dim = q_rope.shape[2]
        kv_dtype_str = (
            "float8_e4m3fn"
            if any(x in str(cache.dtype).lower() for x in ("fp8", "e4m3"))
            else "bfloat16"
        )

        decode_key = TuningKey(
            case="batched_decode",
            max_num_tokens=max_num_tokens,
            actual_num_q_heads=num_attention_heads,
            actual_lkv_dim=qk_nope_head_dim,
            actual_r_dim=actual_r_dim,
            kv_dtype=kv_dtype_str,
        )
        decode_tuned = get_tuned_params(decode_key)

        _, page_size_per_kv_packing, kv_packing, _ = cache.shape
        max_num_seqs = seq_lens.shape[0]
        mixed_key = TuningKey(
            case="mixed",
            max_num_tokens=max_num_tokens,
            actual_num_q_heads=q.shape[1],
            actual_lkv_dim=q.shape[-1],
            actual_r_dim=actual_r_dim,
            q_dtype=jnp.dtype(q.dtype).name,
            kv_dtype=kv_dtype_str,
            page_size_per_kv_packing=page_size_per_kv_packing,
            kv_packing=kv_packing,
            max_num_seqs=max_num_seqs,
            pages_per_seq=block_tables.shape[0] // max_num_seqs,
        )
        mixed_tuned = get_tuned_params(mixed_key)

        # Pass explicit block ratio tuples for (decode, prefill, mixed) stages
        num_kv_pages_per_blocks = (
            decode_tuned.num_kv_pages_per_block,
            1,
            envs.MIXED_NUM_KV_PAGES_PER_BLOCK or mixed_tuned.num_kv_pages_per_block,
        )
        num_queries_per_blocks = (
            decode_tuned.num_queries_per_block,
            16,
            envs.MIXED_NUM_QUERIES_PER_BLOCK or mixed_tuned.num_queries_per_block,
        )

        # tpu-inference MLA kernel expects ql_nope directly in head-major
        # (N, T, L) layout: [num_heads, num_tokens, lkv_dim]
        q = q.transpose((1, 0, 2))

        results = mla_ragged_paged_attention(
            q,
            q_rope,
            k,
            k_rope,
            cache,
            seq_lens,
            block_tables,
            query_start_loc,
            request_distribution,
            sm_scale=sm_scale or 1.0,
            num_kv_pages_per_block=num_kv_pages_per_blocks,
            num_queries_per_block=num_queries_per_blocks,
            vmem_limit_bytes=min(
                decode_tuned.vmem_limit_bytes, mixed_tuned.vmem_limit_bytes
            ),
            decode_batch_size=decode_tuned.decode_batch_size,
            mixed_q_split=envs.MIXED_Q_SPLIT or mixed_tuned.q_split,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            use_causal_mask=use_causal_mask,
            return_lse=return_lse,
        )
        out, new_cache = results[0], results[1]

        # tpu-inference kernel returns out in head-major (N, T, D) layout:
        # [num_heads, num_tokens, head_dim]. Transpose back to (T, N, D).
        out = out.transpose((1, 0, 2))
        if return_lse:
            return out, new_cache, results[2]

        return out, new_cache

    results = jax.jit(
        shard_map.shard_map(
            _mla_ragged_paged_attention,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_rep=False,
        )
    )(
        q_TNA,
        q_rope_TNH,
        k_SA,
        k_rope_SH,
        kv_cache,
        md.seq_lens,
        md.block_tables,
        md.query_start_loc,
        md.request_distribution,
    )
    output_TNA, kv_cache = results[0], results[1]
    if return_lse:
        return kv_cache, output_TNA, results[2]
    return kv_cache, output_TNA


def resolve_decode_topk_cap(
    topk_cap: int, dcp_size: int, topk: int, *, decode_bucket: bool
) -> int:
    if decode_bucket and dcp_size >= 16 and topk_cap > 256 and topk > 256:
        return 256
    return topk_cap


def filter_mixed_live_widths(
    override: tuple[int, ...] | None,
    default_widths: str | tuple[int, ...],
    topk_cap: int | None,
) -> tuple[int, ...]:
    if override is not None:
        raw = override
    elif isinstance(default_widths, str):
        raw = tuple(int(w) for w in default_widths.split(",") if w)
    else:
        raw = default_widths
    return tuple(w for w in raw if not topk_cap or w < topk_cap)


def build_fused_exchange_peer(
    dcp_rank: jax.Array, dcp_lane: jax.Array, query_start_loc: jax.Array
) -> jax.Array:
    return jnp.stack(
        [
            dcp_rank.astype(jnp.int32),
            dcp_lane.astype(jnp.int32),
            query_start_loc[-1].astype(jnp.int32),
        ]
    )


def _mixed_body_levers(*, enabled: bool) -> tuple[str, ...]:
    if not enabled:
        return ()
    levers = sparse_mla_kernel.MIXED_BODY_DEFAULT
    return levers + tuple(
        v for v in sparse_mla_kernel.MIXED_KERNEL_V2_LEVERS if v not in levers
    )


def _decode_body_levers(*, enabled: bool) -> tuple[str, ...]:
    return sparse_mla_kernel.DECODE_BODY_DEFAULT if enabled else ()


def _lse_gathered_once(num_tokens: int) -> bool:
    cap = DCP_MERGE_CHUNKED_LSE_AG_MAX_TOKENS
    return cap <= 0 or num_tokens <= cap


def _merge_chunks(num_tokens: int, *, glm_active: bool | None = None) -> int:
    if not (glm_dsa_v7x_active() if glm_active is None else glm_active):
        return 1
    if num_tokens <= bucket_gate_edge(DCP_MERGE_CHUNKED_MIN_TOKENS):
        return 1
    chunks = (
        DCP_MERGE_CHUNKED_DECODE_SPLITS
        if _lse_gathered_once(num_tokens)
        else DCP_MERGE_CHUNKED_PREFILL_SPLITS
    )
    return 1 if chunks <= 1 or num_tokens % (8 * chunks) else chunks


def _validate_sparse_mla_specs(
    kv_c_normed: jax.Array,
    k_pe: jax.Array,
    nope_spec: SparseMLAKVCacheSpec,
    rope_spec: SparseMLAKVCacheSpec,
    k_scale: float | None,
) -> tuple[float, str]:
    lkv_dim, rope_dim = kv_c_normed.shape[-1], k_pe.shape[-1]
    word_bytes, lane_bytes = kv_cache_utils.WORD_BYTES, kv_cache_utils.TILE_LANE_BYTES
    assert lkv_dim == word_bytes * lane_bytes and rope_dim * 2 == lane_bytes, (
        f"dsa_gather needs fp8 nope dim {word_bytes * lane_bytes} and rope dim "
        f"{lane_bytes // 2}, got {lkv_dim}+{rope_dim}"
    )
    if nope_spec.layout is not rope_spec.layout:
        raise ValueError(
            "sparse MLA attention requires matching NoPE and RoPE layouts; "
            f"got {nope_spec.layout.value} and {rope_spec.layout.value}"
        )
    return (float(k_scale) if k_scale is not None else 1.0), nope_spec.layout.value


def _mask_unpack(
    recv: jax.Array, rows: jax.Array, tl: int
) -> tuple[jax.Array, jax.Array]:
    world, _, vdim = recv.shape
    pad_rows = sparse_mla_kernel.SEND_PAD_ROWS
    live = jnp.clip(rows[0] - 1, 0, tl).astype(jnp.int32)
    row = jax.lax.broadcasted_iota(jnp.int32, (world, tl, vdim), 1)
    part = recv[:, pad_rows:]
    part = jnp.where(row < live, part, jnp.zeros_like(part))
    o = jax.lax.bitcast_convert_type(part, jnp.bfloat16)
    words = recv[:, pad_rows - 1]
    hi = words[:, :tl].astype(jnp.int32)
    lo = words[:, tl : 2 * tl].astype(jnp.int32)
    lse = jax.lax.bitcast_convert_type((hi << 16) | (lo & 0xFFFF), jnp.float32)
    return o, lse


def merge_received_jax(o: jax.Array, lse: jax.Array) -> jax.Array:
    """LSE-merge ``[R, n, V]`` bf16 partials and ``[R, n]`` f32 LSE in JAX."""
    finite = jnp.isfinite(lse)
    shift = jnp.where(finite, lse, -jnp.inf).max(axis=0)
    shift = jnp.where(jnp.isfinite(shift), shift, 0.0)
    weights = jnp.where(finite, jnp.exp(lse - shift[None]), 0.0)
    denom = weights.sum(axis=0)
    weights = weights / jnp.where(denom > 0, denom, 1.0)
    of = o.astype(jnp.float32)
    of = jnp.where(jnp.isnan(of), 0.0, of)
    w_hi = weights.astype(jnp.bfloat16).astype(jnp.float32)
    w_lo = weights - w_hi
    return (of * w_hi[..., None]).sum(axis=0) + (of * w_lo[..., None]).sum(axis=0)


def kv_write_in_gather_supported(
    nope_spec: SparseMLAKVCacheSpec,
    rope_spec: SparseMLAKVCacheSpec,
    num_tokens: int,
    topk: int,
    dcp_size: int = 16,
) -> bool:
    """Whether SparseCore decode gather can fuse this step's KV cache write."""
    topk_cap = dcp_mla_topk_cap(dcp_size)
    rope_words = rope_spec.token_bytes // kv_cache_utils.WORD_BYTES
    return (
        0 < num_tokens <= sparse_mla_kernel.FUSED_PREP_MAX_TOKENS
        and bool(topk_cap)
        and topk > topk_cap
        and nope_spec.layout is kv_cache_utils.KVCacheLayout.SPARSECORE
        and rope_spec.layout is kv_cache_utils.KVCacheLayout.SPARSECORE
        and nope_spec.token_bytes // kv_cache_utils.WORD_BYTES == 128
        and nope_spec.shape[-1] == 128
        and rope_words
        in (native_sc_gather.PACKED_ROPE_WORDS, native_sc_gather.ROPE_WORDS)
        and rope_spec.shape[-1] % rope_words == 0
    )


def sparse_mla_attention(
    ql_nope: jax.Array,
    q_pe: jax.Array,
    kv_c_normed: jax.Array,
    k_pe: jax.Array,
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    topk_indices: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    mesh: Mesh,
    nope_spec: SparseMLAKVCacheSpec,
    rope_spec: SparseMLAKVCacheSpec,
    sm_scale: float | None = None,
    k_scale: float | None = None,
    *,
    allow_masked_dense: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sharded wrapper for unsharded-cache Sparse (DSA) MLA attention."""
    dequant_scale, cache_layout = _validate_sparse_mla_specs(
        kv_c_normed, k_pe, nope_spec, rope_spec, k_scale
    )

    in_specs = (
        P(None, None, None),
        P(None, None, None),
        P(None, None),
        P(None, None),
        *(P(None),) * 7,
    )
    out_specs = (P(None, None, None), P(None), P(None))
    chunk_kwargs = sparse_mla_chunk_kwargs()
    split_q = _resolve_sparse_mla_split_q(ql_nope, q_pe)

    def _sparse_mla_ragged_paged_attention(
        ql_nope_,
        q_pe_,
        kv_c_normed_,
        k_pe_,
        kv_cache_nope_,
        kv_cache_rope_,
        topk_idx,
        seq_lens_,
        block_tables_,
        query_start_loc_,
        request_distribution_,
    ):
        kv_cache_nope_, kv_cache_rope_ = update_sparse_mla_kv_cache(
            kv_cache_nope_,
            kv_cache_rope_,
            kv_c_normed_,
            k_pe_,
            seq_lens_,
            block_tables_,
            query_start_loc_,
            nope_spec=nope_spec,
            rope_spec=rope_spec,
        )

        # Opt-in masked-dense routing builds Q inside each device branch.
        if allow_masked_dense and mla_dispatch.applies(
            ql_nope_, q_pe_, kv_cache_nope_, kv_cache_rope_, topk_idx, cache_layout
        ):
            output_ = mla_dispatch.attention_from_parts(
                ql_nope_,
                q_pe_,
                kv_cache_nope_,
                kv_cache_rope_,
                topk_idx,
                seq_lens_,
                block_tables_,
                query_start_loc_,
                request_distribution_,
                sm_scale=sm_scale or 1.0,
                k_scale=dequant_scale,
            )
            return output_, kv_cache_nope_, kv_cache_rope_

        if split_q:
            q, split_kwargs = ql_nope_, {"q_pe": q_pe_}
        else:
            q, split_kwargs = jnp.concatenate([ql_nope_, q_pe_], axis=-1), {}
        output_ = sparse_mla_ragged_paged_attention(
            q,
            kv_cache_nope_,
            kv_cache_rope_,
            topk_idx,
            block_tables_,
            query_start_loc_,
            request_distribution_,
            sm_scale=sm_scale or 1.0,
            k_scale=dequant_scale,
            cache_layout=cache_layout,
            **chunk_kwargs,
            **split_kwargs,
        )
        return output_[..., : ql_nope_.shape[-1]], kv_cache_nope_, kv_cache_rope_

    output, new_kv_cache_nope, new_kv_cache_rope = jax.jit(
        shard_map.shard_map(
            _sparse_mla_ragged_paged_attention,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_rep=False,
        )
    )(
        ql_nope,
        q_pe,
        kv_c_normed,
        k_pe,
        kv_cache_nope,
        kv_cache_rope,
        topk_indices,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
    )
    return new_kv_cache_nope, new_kv_cache_rope, output


def sparse_mla_attention_dcp(
    ql_nope: jax.Array,
    q_pe: jax.Array,
    kv_c_normed: jax.Array,
    k_pe: jax.Array,
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    local_topk_indices: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    mesh: Mesh,
    nope_spec: SparseMLAKVCacheSpec,
    rope_spec: SparseMLAKVCacheSpec,
    sm_scale: float | None = None,
    k_scale: float | None = None,
    *,
    dcp_size: int,
    interleave_size: int,
    dcp_axis_name: str = streamindex_topk.DCP_AXIS_NAME,
    q_rope_table: jax.Array | None = None,
    dcp_rank_ids: jax.Array | None = None,
    send_layout: bool = False,
    fused_exchange: bool = False,
    dcp_lane_ids: jax.Array | None = None,
    fused_merge: bool = False,
    merge_pieces: bool = True,
    skip_kv_write: bool = False,
    mixed_live_widths_override: tuple[int, ...] | None = None,
    attention_kernel_batch_size: int | None = None,
) -> tuple[jax.Array, ...]:
    """Sparse MLA attention over a DCP position-sharded latent cache."""
    if dcp_size <= 1:
        raise ValueError(
            f"sparse_mla_attention_dcp requires dcp_size > 1, got {dcp_size}; "
            "use sparse_mla_attention for the unsharded case."
        )
    if dcp_axis_name not in mesh.axis_names:
        raise ValueError(
            f"dcp_size={dcp_size} requires a mesh with a "
            f"{dcp_axis_name!r} axis, got {mesh.axis_names}."
        )
    if mesh.shape[dcp_axis_name] != dcp_size:
        raise ValueError(
            f"dcp_size={dcp_size} does not match mesh axis "
            f"{dcp_axis_name!r} of size {mesh.shape[dcp_axis_name]}."
        )
    num_tokens = ql_nope.shape[0]
    if local_topk_indices.shape[0] != dcp_size * num_tokens:
        raise ValueError(
            "local_topk_indices must carry one row per (rank, token): "
            f"expected {dcp_size * num_tokens} rows for {num_tokens} tokens "
            f"at dcp_size={dcp_size}, got {local_topk_indices.shape[0]}."
        )

    dequant_scale, cache_layout = _validate_sparse_mla_specs(
        kv_c_normed, k_pe, nope_spec, rope_spec, k_scale
    )
    glm_v7x = glm_dsa_v7x_active()
    split_q = _resolve_sparse_mla_split_q(ql_nope, q_pe)
    in_specs: tuple[P, ...] = (
        P(None, None, None),
        P(None, None, None),
        P(None, None),
        P(None, None),
        P(dcp_axis_name),
        P(dcp_axis_name),
        P(dcp_axis_name, None),
        *(P(None),) * 4,
    )
    q_rope = q_rope_table is not None
    if q_rope:
        if not split_q:
            raise ValueError("q_rope_table requires split-Q Sparse MLA")
        in_specs += (P(None, None, None),)

    out_specs: tuple[P, ...] = (
        P(dcp_axis_name),
        P(dcp_axis_name),
        P(dcp_axis_name, None, None),
        P(dcp_axis_name, None),
    )
    if send_layout:
        if not q_rope:
            raise ValueError("send_layout requires q_rope_table decode prologue")
        if num_tokens > sparse_mla_kernel.FUSED_PREP_MAX_TOKENS:
            raise ValueError(f"send_layout is decode-only, got {num_tokens} tokens")
        out_specs = out_specs[:2] + (P(dcp_axis_name, None, None),)
    lane_axis: str | None = None
    if fused_exchange:
        if not (send_layout and dcp_rank_ids is not None and dcp_lane_ids is not None):
            raise ValueError(
                "fused_exchange needs send_layout, dcp_rank_ids and dcp_lane_ids"
            )
        lane_axes = tuple(a for a in mesh.axis_names if a != dcp_axis_name)
        if len(lane_axes) > 1:
            raise ValueError(
                f"fused_exchange: at most one lane axis, got {mesh.axis_names}"
            )
        lane_axis = lane_axes[0] if lane_axes else None
        out_specs = out_specs[:2] + (
            (P(dcp_axis_name, None, None),)
            if fused_merge
            else (P(dcp_axis_name, None, None), P(dcp_axis_name, None))
        )

    def _local(
        ql_nope_,
        q_pe_,
        kv_c_normed_,
        k_pe_,
        kv_cache_nope_,
        kv_cache_rope_,
        topk_idx,
        seq_lens_,
        block_tables_,
        query_start_loc_,
        request_distribution_,
        *extra_,
    ):
        rope_table_ = extra_[0] if q_rope else None
        dcp_lane = extra_[-1][0] if fused_exchange else None
        fuse_here = bool(fused_exchange and topk_idx.shape[0] >= 1)
        merge_here = bool(
            fused_merge and topk_idx.shape[0] <= MLA_FUSED_SMALL_A2A_MERGE_MAX_TOKENS
        )
        if dcp_rank_ids is not None:
            dcp_rank = extra_[int(q_rope)][0]
        else:
            dcp_rank = streamindex_topk.cp_rank_as_data(dcp_axis_name, dcp_size)

        decode_bucket = topk_idx.shape[0] <= sparse_mla_kernel.FUSED_PREP_MAX_TOKENS
        tuned_dcp = bool(glm_v7x or send_layout or q_rope)
        mask_in_kernel = tuned_dcp
        fused_prep = bool(tuned_dcp and decode_bucket)
        kv_in_gather = (
            not skip_kv_write
            and fused_prep
            and kv_write_in_gather_supported(
                nope_spec,
                rope_spec,
                kv_c_normed_.shape[0],
                topk_idx.shape[-1],
                dcp_size,
            )
        )
        kv_write = None
        if kv_in_gather:
            page, slot = kv_cache_utils._get_dcp_page_and_slot(
                kv_c_normed_.shape[0],
                seq_lens_,
                block_tables_,
                query_start_loc_,
                dcp_rank,
                nope_spec=nope_spec,
                dcp_size=dcp_size,
                interleave_size=interleave_size,
            )
            kv_write = native_sc_gather.pack_write_rows(
                kv_cache_utils._dst_rows(page, slot, nope_spec.page_size),
                kv_cache_utils.pack_tokens(kv_c_normed_, nope_spec.token_bytes),
                kv_cache_utils.pack_tokens(k_pe_, rope_spec.token_bytes),
            )
        elif not skip_kv_write:
            kv_cache_nope_, kv_cache_rope_ = update_sparse_mla_kv_cache_dcp(
                kv_cache_nope_,
                kv_cache_rope_,
                kv_c_normed_,
                k_pe_,
                seq_lens_,
                block_tables_,
                query_start_loc_,
                dcp_rank,
                nope_spec=nope_spec,
                rope_spec=rope_spec,
                dcp_size=dcp_size,
                interleave_size=interleave_size,
                compact_rows=bool(glm_v7x),
            )

        owns = None
        if not mask_in_kernel:
            owns = jnp.any(topk_idx >= 0, axis=-1)
            topk_idx = topk_idx.at[:, 0].set(
                jnp.where(owns, topk_idx[:, 0], jnp.int32(0))
            )

        if split_q:
            split_kwargs: dict[str, object] = {"q_pe": q_pe_, "lse_lane_dense": True}
            q = ql_nope_
            if q_rope:
                split_kwargs.update(
                    q_rope_table=rope_table_,
                    lse_in_kernel=mask_in_kernel and decode_bucket,
                )
                if send_layout:
                    split_kwargs["send_world"] = dcp_size
                if fused_exchange and fuse_here:
                    pin, vmem_mib = fused_small_a2a_bucket_config(num_tokens)
                    split_kwargs["send_exchange"] = (
                        dcp_axis_name,
                        lane_axis,
                        int(vmem_mib) * 1024 * 1024,
                        pin,
                        bool(merge_here),
                    )
                    assert dcp_lane is not None
                    split_kwargs["send_peer"] = build_fused_exchange_peer(
                        dcp_rank, dcp_lane, query_start_loc_
                    )
        else:
            q = jnp.concatenate([ql_nope_, q_pe_], axis=-1)
            split_kwargs = {}

        topk_cap: int | None = resolve_decode_topk_cap(
            dcp_mla_topk_cap(dcp_size),
            dcp_size,
            topk_idx.shape[-1],
            decode_bucket=decode_bucket,
        )
        if not (
            topk_cap
            and cache_layout == sparse_mla_kernel.SPARSECORE_CACHE_LAYOUT
            and topk_idx.shape[-1] > topk_cap
        ):
            topk_cap = None
        chunk_kwargs: dict[str, int | bool] = (
            {
                "gather_and_attention_chunk_size": resolve_dcp_mla_topk_cap_chunk(
                    envs.TPU_DCP_MLA_TOPK_CAP_CHUNK
                ),
                "capped_live_rows": True,
            }
            if topk_cap
            else dict(sparse_mla_chunk_kwargs())
        )
        if attention_kernel_batch_size is not None:
            chunk_kwargs = {
                **chunk_kwargs,
                "attention_kernel_batch_size": attention_kernel_batch_size,
            }
        live_widths = (
            filter_mixed_live_widths(
                mixed_live_widths_override, MLA_MIXED_LIVE_WIDTHS, topk_cap
            )
            if (glm_v7x or mixed_live_widths_override is not None)
            else ()
        )
        # Virtual page ordinal -> physical block in this rank's own shard, the
        # same resolution the scatter above and `streamindex_topk_dcp` use.
        local_block_tables = jnp.mod(block_tables_, jnp.int32(nope_spec.num_pages))
        small_chunk_batch = DCP_MLA_SMALL_CHUNK_BATCH if glm_v7x else 1
        small_chunk_tokens = (
            DCP_MLA_SMALL_CHUNK_TOKENS
            if glm_v7x
            else sparse_mla_kernel.FUSED_PREP_MAX_TOKENS // 2
        )
        out_pieces = 0
        if (
            merge_pieces
            and glm_v7x
            and DCP_MLA_MERGE_PIECES > 0
            and not decode_bucket
            and mask_in_kernel
            and split_q
            and topk_cap
            and topk_idx.shape[0] > bucket_gate_edge(DCP_MLA_MERGE_PIECES_MIN_TOKENS)
        ):
            out_pieces = _merge_chunks(topk_idx.shape[0])

        attn = sparse_mla_kernel.sparse_ragged_paged_attention(
            q,
            kv_cache_nope_,
            kv_cache_rope_,
            topk_idx,
            local_block_tables,
            query_start_loc_,
            request_distribution_,
            sm_scale=sm_scale or 1.0,
            k_scale=dequant_scale,
            return_lse=True,
            is_dcp_mode=True,
            cache_layout=cache_layout,
            # Ranks own ~topk/dcp of each row: a static top-k width with an
            # exact in-kernel overflow pass for the rows that need more.
            topk_cap=topk_cap,
            small_chunk_batch_size=small_chunk_batch,
            small_chunk_tokens=small_chunk_tokens,
            small_chunk_gather_atoms=4 if q.shape[0] >= 16 else 1,
            lse_mask_empty=mask_in_kernel,
            fused_window_prep=fused_prep,
            mixed_live_gather=bool(glm_v7x),
            mixed_live_widths=live_widths,
            mixed_body_levers=_mixed_body_levers(enabled=bool(glm_v7x)),
            kv_write=kv_write,
            decode_row_widths=(
                tuple(w for w in MLA_DECODE_ROW_WIDTHS if w < topk_cap)
                if (glm_v7x and topk_cap)
                else ()
            ),
            decode_body_levers=_decode_body_levers(enabled=bool(glm_v7x)),
            out_pieces=out_pieces,
            **chunk_kwargs,  # type: ignore[arg-type]
            **split_kwargs,  # type: ignore[arg-type]
        )
        output, lse = attn[:2]
        if kv_write is not None:
            kv_cache_nope_, kv_cache_rope_ = attn[2:]
        if fused_exchange:
            per = ql_nope_.shape[1] // dcp_size
            block = sparse_mla_kernel.SEND_PAD_ROWS + num_tokens * per
            rows = 1 + query_start_loc_[-1:].astype(jnp.int32) * per
            if merge_here:
                merged = jax.lax.bitcast_convert_type(lse, jnp.bfloat16)
                return (
                    kv_cache_nope_,
                    kv_cache_rope_,
                    merged.reshape(num_tokens, per, -1),
                )
            o_recv, l_recv = _mask_unpack(
                lse.reshape(dcp_size, block, lse.shape[-1]),
                rows,
                block - sparse_mla_kernel.SEND_PAD_ROWS,
            )
            if fused_merge:
                merged = merge_received_jax(o_recv, l_recv).astype(jnp.bfloat16)
                return (
                    kv_cache_nope_,
                    kv_cache_rope_,
                    merged.reshape(num_tokens, per, -1),
                )
            return kv_cache_nope_, kv_cache_rope_, o_recv, l_recv
        if send_layout:
            return (kv_cache_nope_, kv_cache_rope_, lse)
        if not mask_in_kernel:
            assert owns is not None
            lse = jnp.where(owns[:, None], lse, -jnp.inf)
        return (kv_cache_nope_, kv_cache_rope_, output[..., : ql_nope_.shape[-1]], lse)

    rank_args: tuple[jax.Array, ...] = () if dcp_rank_ids is None else (dcp_rank_ids,)
    if rank_args:
        in_specs = in_specs + (P(dcp_axis_name),)
    if fused_exchange:
        assert dcp_lane_ids is not None
        rank_args = rank_args + (dcp_lane_ids,)
        in_specs = in_specs + (P(dcp_axis_name),)
    return jax.jit(
        shard_map.shard_map(
            _local, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False
        )
    )(
        ql_nope,
        q_pe,
        kv_c_normed,
        k_pe,
        kv_cache_nope,
        kv_cache_rope,
        local_topk_indices,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
        *((q_rope_table,) if q_rope else ()),
        *rank_args,
    )
