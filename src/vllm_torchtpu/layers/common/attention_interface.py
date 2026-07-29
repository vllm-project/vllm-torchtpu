import functools
import math
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import shard_map
from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention
from jax.experimental.pallas.ops.tpu.splash_attention import \
    splash_attention_kernel as splash
from jax.experimental.pallas.ops.tpu.splash_attention import \
    splash_attention_mask as mask_lib
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

import vllm_torchtpu.kernels.experimental.batched_rpa.wrapper as rpa_batched
import vllm_torchtpu.kernels.mla.v2.kernel as mla_v2_kernel
import vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel as rpa_default
import vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel_hd64 as rpa_hd64
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.wrapper import \
    PCP_AXIS_NAME
from vllm_torchtpu.kernels.experimental.rpa_v3_cp.kernel import \
    ragged_paged_attention as pcp_cp_kernel
from vllm_torchtpu.kernels.flash_attention.kernel import flash_attention
from vllm_torchtpu.kernels.mla.v2.tuned_params import (TuningKey,
                                                       get_tuned_params)
from vllm_torchtpu.layers.common.attention_metadata import AttentionMetadata
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import get_megacore

logger = init_logger(__name__)

MAX_ALLOWED_PAGE_INDICES_N = (
    128 * 1024
)  # Based on experiments on v5e, 256x1024 results in smem oom but 128x1024 not. TODO: Adjust this based on TPU version.

# Default and experimental batched RPA kernels are loaded unconditionally.
# Selection happens per attention layer via the `use_batched_rpa` flag plumbed
# from `PallasAttentionBackendImpl` / `PallasBatchedRPAAttentionBackendImpl`.
ragged_paged_attention = rpa_default.ragged_paged_attention
ragged_paged_attention_batched = rpa_batched.ragged_paged_attention
get_kv_cache_shape = rpa_default.get_kv_cache_shape

ragged_paged_attention_hd64 = rpa_hd64.ragged_paged_attention_hd64
get_kv_cache_shape_hd64 = rpa_hd64.get_kv_cache_shape

mla_ragged_paged_attention = mla_v2_kernel.mla_ragged_paged_attention


def sharded_flash_attention(
    mesh: Mesh,
    causal: bool = True,
    sm_scale: Optional[float] = None,
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
        return flash_attention(q,
                               k,
                               v,
                               segment_ids=segment_ids,
                               sm_scale=sm_scale,
                               causal=causal,
                               vmem_limit_bytes=vmem_limit_bytes)

    return jax.jit(
        shard_map.shard_map(_flash_attention,
                            mesh=mesh,
                            in_specs=in_specs,
                            out_specs=out_specs,
                            check_rep=False))


def sharded_paged_attention(
    mesh: Mesh,
    attn_logits_soft_cap: Optional[float] = None,
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
                "This will result in smem OOM. Use `paged_attention_with_guarded_smem` to run with minibatches."
            )
        return paged_attention(
            q,
            k,
            v,
            lengths,
            page_indices,
            attn_logits_soft_cap=attn_logits_soft_cap,
            pages_per_compute_block=min(
                16, page_indices.shape[1]),  # 512 / page_size:32,
            megacore_mode="kv_head" if get_megacore() else None,
        )

    return jax.jit(
        shard_map.shard_map(
            _paged_attention_fn,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_rep=False,
        ))


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
    # Addresses b/336316706. Summary:
    # Paged attention kernel stores `lengths` (batch_size * 4 bytes) and `page_indices` (batch_size * num_blocks_per_seq * 4 bytes) in SMEM.
    # Capacity of SMEM is quite limited which is also TPU version dependent. Models with higher context length or higher batch size, can cause OOM in SMEM.
    # There are two solutions:
    # 1. Reduce blocks per seq by increasing page size.
    # 2. Splitting the batch into several minibatches (Higher perf based on my benchmark).

    batch_size, blocks_per_seq = page_indices.shape

    if page_indices.size <= MAX_ALLOWED_PAGE_INDICES_N:
        return paged_attention_kernel(q, k_pages, v_pages, lengths,
                                      page_indices)

    mini_batch_size = MAX_ALLOWED_PAGE_INDICES_N // blocks_per_seq

    # If batch_size is not disible by mini_batch_size,
    # we set mini_batch_size to a smaller value, i.e GCD,
    # which will trigger more kernel launches but it's fine.
    # TODO: Fix --decode_seqs_padding with this limitation.
    mini_batch_size = math.gcd(batch_size, mini_batch_size)

    num_kernel_launches = batch_size // mini_batch_size

    outputs = jnp.zeros_like(q).reshape(
        (num_kernel_launches, mini_batch_size, *q.shape[1:]))
    q = q.reshape((num_kernel_launches, mini_batch_size, *q.shape[1:]))
    seq_lens = lengths.reshape((num_kernel_launches, mini_batch_size))
    block_indices = page_indices.reshape(
        (num_kernel_launches, mini_batch_size, page_indices.shape[1]))

    for i in range(num_kernel_launches):
        outputs = outputs.at[i].set(
            paged_attention_kernel(q[i], k_pages, v_pages, seq_lens[i],
                                   block_indices[i]))

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
    assert K == K_c
    # NOTE: The cache updating is pretty tricky:
    # 1. The random access updating cache is not as performant as the slice updating.
    #    If the random access is necessary, make sure the indexing count is as small as possible.
    # 2. The random access updating may trigger extra tranpose (memory copy) of cache,
    #    which is a disaster because the cache is huge. This is a data formatting op inserted by
    #    the XLA compiler and not well documented.
    # To mitigate the issues above:
    # For prefill:
    # We reshape the operand so that we can update the cache in block wise, which only requires the block indices.
    # For decode:
    # We reshape the cache so that we can update the cache in token wise, which only requires the token indices (block_id + offset).
    if is_prefill:
        # In the case of sliding window, we should select sliding_window tokens from actual prompt, not from the padded tokens.
        if sliding_window and T > sliding_window:
            assert B == 1
            start_index = jax.lax.max(0, prefill_seq_len - sliding_window)
            operand = jax.lax.dynamic_slice_in_dim(
                operand, start_index, sliding_window,
                axis=2)  # TODO: @pooyam Perf check this.
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
        # NOTE: `cache.[:, indices, :].set()` will trigger the extra tranpose of the cache.
        # The `jnp.arange(K)[..., None]` trick is to avoid it. WTF?
        cache = cache.at[jnp.arange(K)[..., None], indices, :].set(operand)
        cache = cache.reshape(K, L, S, H)
    return cache


@functools.partial(
    jax.jit, static_argnames=["window_size", "attn_logits_soft_cap", "is_mqa"])
def apply_splash(q, k, v, window_size, attn_logits_soft_cap,
                 is_mqa) -> jax.Array:
    # q: (batch_size, num_heads, seq_len, head_dim)
    num_heads = q.shape[1]
    q_seq_len = q.shape[2]
    kv_seq_len = k.shape[2]
    assert kv_seq_len >= q_seq_len

    masks = [
        mask_lib.LocalMask((q_seq_len, kv_seq_len), (window_size, 0),
                           kv_seq_len - q_seq_len) for _ in range(num_heads)
    ]
    mask = mask_lib.MultiHeadMask(tuple((m for m in masks)))
    block_sizes = splash.BlockSizes.get_default()

    if is_mqa:
        attn = splash.make_splash_mqa_single_device(
            mask,
            block_sizes=block_sizes,
            attn_logits_soft_cap=attn_logits_soft_cap)
    else:
        attn = splash.make_splash_mha_single_device(
            mask,
            block_sizes=block_sizes,
            attn_logits_soft_cap=attn_logits_soft_cap)
    attn = jax.vmap(attn)
    outputs = attn(q, k, v, None)

    return outputs


def sharded_splash_attention(
    mesh: Mesh,
    window_size: Optional[int] = None,
    attn_logits_soft_cap: Optional[float] = None,
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
        ))


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
):
    """Shards along KV heads."""

    qkv_spec = P(None, "model", None)
    kv_cache_spec = P(None, None, "model", None, None)
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

    use_hd64 = q.shape[-1] == 64

    if use_hd64:
        # Batched RPA has no hd64 variant; head_dim==64 always uses default.
        func = functools.partial(ragged_paged_attention_hd64,
                                 strict_sliding_window=True)
    else:
        func = rpa_func

    if attention_sink is not None:
        if not use_hd64:
            raise NotImplementedError(
                "Attention sink support is only available when head_dim==64")

        in_specs += (P("model"), )
        args += (attention_sink, )

    # Speculative decoding draft-only VMEM relief: cap the KV-fetch block on the local path.
    block_kwargs: dict[str, Any] = {}
    if not shard and kv_block_cap is not None and not use_hd64:
        page_size = kv_cache.shape[1]
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
            block_kwargs["p_block_sizes"] = _capped(
                rpa_default.RpaCase.PREFILL)

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
) -> Tuple[jax.Array, jax.Array]:
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
        head_dim_original = q.shape[-1]

    if sm_scale is None:
        sm_scale = head_dim_original**-0.5

    md = attention_metadata

    # (T, N, H)
    output, kv_cache = sharded_ragged_paged_attention(
        mesh,
        q,
        k,
        v,
        kv_cache,
        md.seq_lens,
        md.block_tables,
        md.query_start_loc,
        md.request_distribution,
        sinks,
        sm_scale=sm_scale,
        attention_chunk_size=attention_chunk_size,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        skip_kv_update=skip_kv_update,
        rpa_func=rpa_func,
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
        sm_scale: float | None = None) -> Tuple[jax.Array, jax.Array]:
    """Main shared interface for Multi-Head Latent Attention (MLA).

    Computes sharded MLA paged attention and applies in-place KV cache updates across
    the device mesh using custom Pallas kernels.

    Args:
        q_TNA: NOPE query activations in token-major format `(T, N, A)` (`[num_tokens, num_heads, lkv_dim]`).
        q_rope_TNH: RoPE query activations in token-major format `(T, N, H)` (`[num_tokens, num_heads, rope_dim]`).
        k_SA: New compressed latent keys to insert into the KV cache `(S, A)` (`[num_tokens, lkv_dim]`).
        k_rope_SH: New RoPE keys to insert into the KV cache `(S, H)` (`[num_tokens, rope_dim]`).
        kv_cache: Persistent paged latent KV cache tensor residing across devices.
        md: Attention metadata containing block tables, sequence lengths, and layout indices.
        mesh: JAX execution device mesh dictating parallel shard routing.
        num_attention_heads: Total number of attention heads across the layer.
        qk_nope_head_dim: Inner compressed projection dimension (`lkv_dim`).
        q_scale: Optional scalar activation quantization scale for queries.
        k_scale: Optional scalar parameter quantization scale for latent keys.
        v_scale: Optional scalar parameter quantization scale for latent values.
        sm_scale: Softmax temperature scale factor.

    Returns:
        Tuple of `(updated_kv_cache, output_TNA)` in token-major sequence layout `(T, N, D)`.
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
        P(None, "model",
          None),  # attn output in token-major sequence format (T, N, D)
        P(None),  # kv cache
    )

    def _mla_ragged_paged_attention(q, q_rope, k, k_rope, cache, seq_lens,
                                    block_tables, query_start_loc,
                                    request_distribution):
        max_num_tokens = q.shape[
            0]  # q is in token-major sequence format (T, N, A)
        actual_r_dim = q_rope.shape[2]
        kv_dtype_str = "float8_e4m3fn" if any(
            x in str(cache.dtype).lower()
            for x in ("fp8", "e4m3")) else "bfloat16"

        tuning_key = TuningKey(
            case="batched_decode",
            max_num_tokens=max_num_tokens,
            actual_num_q_heads=num_attention_heads,
            actual_lkv_dim=qk_nope_head_dim,
            actual_r_dim=actual_r_dim,
            kv_dtype=kv_dtype_str,
        )
        tuned = get_tuned_params(tuning_key)
        decode_batch_size = tuned.decode_batch_size

        # Pass explicit block ratio tuples for (decode, prefill, mixed) stages
        num_kv_pages_per_blocks = (tuned.num_kv_pages_per_block, 1, 1)
        num_queries_per_blocks = (tuned.num_queries_per_block, 16, 16)

        # tpu-inference MLA kernel expects ql_nope directly in head-major (N, T, L) layout: [num_heads, num_tokens, lkv_dim]
        q = q.transpose((1, 0, 2))

        out, new_cache = mla_ragged_paged_attention(
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
            decode_batch_size=decode_batch_size,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale)

        # tpu-inference kernel returns out in head-major (N, T, D) layout: [num_heads, num_tokens, head_dim]. Transpose back to (T, N, D).
        out = out.transpose((1, 0, 2))

        return out, new_cache

    output_TNA, kv_cache = jax.jit(
        shard_map.shard_map(_mla_ragged_paged_attention,
                            mesh=mesh,
                            in_specs=in_specs,
                            out_specs=out_specs,
                            check_rep=False))(q_TNA, q_rope_TNH, k_SA,
                                              k_rope_SH, kv_cache, md.seq_lens,
                                              md.block_tables,
                                              md.query_start_loc,
                                              md.request_distribution)
    return kv_cache, output_TNA
# ---------------------------------------------------------------------------
# All-gather prefill context parallel (PCP) attention.
#
# """All-gather prefill context parallel (PCP) attention.
#
# An alternative to the streaming-ring PCP implementation in
# `vllm_torchtpu.kernels.experimental.pcp_streaming_rpa`. Ported from
# tpu-inference's `attention_interface.pcp_ragged_paged_attention`, with the
# shard_map specs adapted to this repo's pcp-only mesh (heads are already
# per-device under vLLM TP, so only the token axis is sharded here).
#
# KV-cache layout (INCOMPATIBLE with streaming -- see below): the cache is
# round-robin striped with interleave 1, i.e. global token `g` lives on rank
# `g % pcp` at local slot `g // pcp` (local page `g // (pcp * page_size)` via the
# block table, offset `(g // pcp) % page_size`).
#
# The streaming path instead stripes with interleave `page_size`: token `g` lives
# on rank `(g // page_size) % pcp`. The two layouts therefore do NOT interoperate
# -- a cache written by one path is garbage to the other. Concretely this path
# requires `--cp-kv-cache-interleave-size 1` while streaming requires it to equal
# `page_size`, so `PCP_IMPL` is fixed for the lifetime of a server and cannot be
# switched against a warm cache.
#
# tpu-inference expresses the same striping by sharding the `page_size` dim of a
# `page_size*pcp` global page; here the cache is sharded on the block dim with
# `page_size`-sized pages. The per-rank tensor the kernel sees is identical.
#
# Token layout (differs from streaming!): the request's `S = 2*pcp*C` current
# tokens are split head-tail into `2*pcp` chunks of size `C`; rank `r` holds
# chunk `r` (head) and chunk `2*pcp-1-r` (tail), so its local buffer is
# `[head_chunk | tail_chunk]`. This balances the causal triangle across ranks.
# ---------------------------------------------------------------------------


def merge_attn_states(context_out: jax.Array, context_lse: jax.Array,
                      query_out: jax.Array,
                      query_lse: jax.Array) -> tuple[jax.Array, jax.Array]:
    """LSE-combine the cache term and the current term (disjoint KV)."""
    max_lse = jnp.maximum(context_lse, query_lse)
    max_lse_safe = jnp.where(jnp.isinf(max_lse), 0.0, max_lse)
    exp_context = jnp.exp(context_lse - max_lse_safe)
    exp_query = jnp.exp(query_lse - max_lse_safe)

    sum_exp = exp_context + exp_query
    denom = jnp.where(sum_exp == 0.0, 1.0, sum_exp)
    merged_out = (context_out * exp_context[..., None] +
                  query_out * exp_query[..., None]) / denom[..., None]
    merged_lse = jnp.where(sum_exp == 0.0, -jnp.inf,
                           max_lse_safe + jnp.log(denom))
    return merged_out, merged_lse


def _lse_all_to_all(o: jax.Array, lse: jax.Array, axis: str, axis_size: int):
    """LSE-merge partial attention across `axis`, keeping only THIS rank's block.

    `o`/`lse` hold the partial softmax for ALL `axis_size` query blocks computed
    against this rank's KV shard. An all-to-all gives each rank every rank's
    partial for its OWN block (moves 1x instead of axis_size x the data), then
    the merge is local.

    o:   [axis_size*blk, nq, hd] -> [blk, nq, hd]
    lse: [axis_size*blk, nq]     -> [blk, nq]
    """
    n = o.shape[0] // axis_size
    nq = o.shape[1]
    o = o.reshape(axis_size, n, *o.shape[1:])
    lse = lse.reshape(axis_size, n, nq)
    # send block j to rank j; axis 0 becomes the SOURCE rank
    o = lax.all_to_all(o, axis, 0, 0, tiled=True)
    lse = lax.all_to_all(lse, axis, 0, 0, tiled=True)
    m = jnp.max(lse, axis=0, keepdims=True)
    m_safe = jnp.where(jnp.isinf(m), 0.0, m)
    w = jnp.exp(lse - m_safe)
    denom = jnp.sum(w, axis=0)
    w_o = w[..., None].astype(o.dtype)
    o_sum = jnp.sum(o * w_o, axis=0)
    denom_safe = jnp.where(denom == 0.0, 1.0, denom)[..., None]
    o_merged = o_sum.astype(denom.dtype) / denom_safe
    lse_merged = jnp.where(denom == 0.0, -jnp.inf, m_safe[0] + jnp.log(denom))
    return o_merged, lse_merged


def sharded_pcp_allgather_ragged_paged_attention(
    mesh: Mesh,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    kv_cache_lens: jax.Array,
    pcp_q_pos_offsets: jax.Array,
    sm_scale: float,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    update_kv_cache: bool = True,
    use_causal_mask: bool = True,
    cp_rank: int | None = None,
):
    """Single-request prefill context-parallel attention (all-gather variant).

    Two kernel launches per rank, LSE-combined:
      * cache phase:   all-gather Q across pcp, attend the pcp-strided cache
                       (non-causal, no write), then all-to-all LSE merge.
      * current phase: local Q attends the current KV (causal, per-chunk
                       `q_pos_offset`); the tail seq writes the strided cache.
    """
    if PCP_AXIS_NAME not in mesh.axis_names:
        raise ValueError(
            f"all-gather PCP requires a mesh with a '{PCP_AXIS_NAME}' axis; "
            f"got {mesh.axis_names}.")
    pcp_size = mesh.shape[PCP_AXIS_NAME]
    if cp_rank is not None and not 0 <= cp_rank < pcp_size:
        raise ValueError(
            f"all-gather PCP cp_rank must lie in [0, {pcp_size}); got "
            f"{cp_rank!r}.")
    two_p = 2 * pcp_size
    padded_q_len = q.shape[0]          # current tokens across all pcp ranks
    if padded_q_len % two_p:
        raise ValueError(
            f"PCP head-tail split needs q tokens divisible by 2*pcp: "
            f"{padded_q_len=} {two_p=}.")
    C = padded_q_len // two_p          # head-tail chunk size

    _row = [c for r in range(pcp_size) for c in (r, two_p - 1 - r)]
    _inv = [0] * two_p
    for _i, _c in enumerate(_row):
        _inv[_c] = _i
    inv_row = jnp.array(_inv, jnp.int32)

    # pcp-only mesh: shard the token axis; heads are already per-device (TP).
    tok_spec = P(PCP_AXIS_NAME)
    pcp_spec = P(PCP_AXIS_NAME, None)
    repl = P()

    def _fn(q_l, k_l, v_l, kvc, kvl, kvcl, pi, cu_ql, dist2, pcp_qpos):
        # Under MPMD (the vLLM server) pass an explicit cp_rank: lax.axis_index
        # lowers to a PartitionId instruction, which torchtpu's MPMD compiler
        # rejects ("not supported for SPMD partitioning ... the meaning is
        # ambiguous"), and every rank is its own process that knows its index at
        # trace time. The streaming wrapper avoids axis_index for the same
        # reason -- see _select_replicated_shard_for_pcp_rank there.
        #
        # cp_rank=None keeps the SPMD path tpu-inference uses, where one process
        # drives every rank and a baked-in constant would be the same value on
        # all of them. That is what the multi-device tests exercise.
        if cp_rank is None:
            r = jnp.reshape(lax.axis_index(PCP_AXIS_NAME), (1, )).astype(
                jnp.int32)
        else:
            r = jnp.full((1, ), cp_rank, jnp.int32)

        def ag(x):
            return lax.all_gather(x, PCP_AXIS_NAME, axis=0, tiled=True)

        def to_token_order(x):  # rank-order chunks -> global token order
            g = ag(x).reshape(two_p, C, *x.shape[1:])
            return g[inv_row].reshape(padded_q_len, *x.shape[1:])

        common = dict(cp_rank=r,
                      cp_group_size=pcp_size,
                      return_lse=True,
                      sm_scale=sm_scale,
                      q_scale=q_scale,
                      k_scale=k_scale,
                      v_scale=v_scale)

        # ---- cache phase: all-gather Q, attend this rank's strided cache ----
        ag_q = ag(q_l)
        cu_cache = jnp.zeros_like(cu_ql[0]).at[1:].set(padded_q_len)
        dist_cache = jnp.array([0, 0, 1], jnp.int32)
        o1, kvc1, l1 = pcp_cp_kernel(ag_q,
                                     k_l,
                                     v_l,
                                     kvc,
                                     kvl,
                                     pi,
                                     cu_cache,
                                     dist_cache,
                                     kv_cache_lens=kvcl,
                                     skip_current_attn=True,
                                     use_causal_mask=False,
                                     update_kv_cache=False,
                                     **common)
        o1, l1 = _lse_all_to_all(o1, l1, PCP_AXIS_NAME, pcp_size)

        # ---- current phase: local Q vs current KV (causal), tail writes ----
        # Rank-order KV remap avoids the head-tail permute copies, but is only
        # valid when a prefetch block fits one chunk (kernel caps bkv_sz <= C).
        page_size_local = kvc.shape[1]
        remap_kv = (C >= page_size_local) and (C % page_size_local == 0)
        if remap_kv:
            k_cur, v_cur = ag(k_l), ag(v_l)
        else:
            k_cur, v_cur = to_token_order(k_l), to_token_order(v_l)
        o2, kvc2, l2 = pcp_cp_kernel(q_l,
                                     k_cur,
                                     v_cur,
                                     kvc1,
                                     kvl,
                                     pi,
                                     cu_ql[0],
                                     dist2,
                                     kv_cache_lens=kvcl,
                                     q_pos_offsets=pcp_qpos[0],
                                     pcp_chunk_size=(C if remap_kv else None),
                                     skip_cache_attn=True,
                                     use_causal_mask=use_causal_mask,
                                     update_kv_cache=update_kv_cache,
                                     write_last_seq_only=True,
                                     **common)

        out, _ = merge_attn_states(o1, l1, o2, l2)  # [2C] = [head | tail]
        return out.astype(q.dtype), kvc2

    return jax.shard_map(
        _fn,
        mesh=mesh,
        in_specs=(tok_spec, tok_spec, tok_spec, tok_spec, repl, repl, repl,
                  pcp_spec, repl, pcp_spec),
        out_specs=(tok_spec, tok_spec),
        check_vma=False,
    )(q, k, v, kv_cache, kv_lens, kv_cache_lens, page_indices, cu_q_lens,
      distribution, pcp_q_pos_offsets)
