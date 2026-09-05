import functools
import math
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.experimental import shard_map
from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention
from jax.experimental.pallas.ops.tpu.splash_attention import \
    splash_attention_kernel as splash
from jax.experimental.pallas.ops.tpu.splash_attention import \
    splash_attention_mask as mask_lib
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

import vllm_torchtpu.kernels.mla.sparse.kernel as sparse_mla_kernel
import vllm_torchtpu.kernels.mla.v2.kernel as mla_v2_kernel
import vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel as rpa_default
import vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel_hd64 as rpa_hd64
from vllm_torchtpu import envs
from vllm_torchtpu.kernels.experimental.batched_rpa import \
    configs as batched_rpa_configs
from vllm_torchtpu.kernels.flash_attention.kernel import flash_attention
from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    SparseMLAKVCacheSpec, update_sparse_mla_kv_cache)
from vllm_torchtpu.kernels.mla.sparse import dsa_gather
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

# Temporary: selects the batched_rpa_longctx fork over mainline batched_rpa
if envs.USE_BATCHED_RPA_LONGCTX:
    import vllm_torchtpu.kernels.experimental.batched_rpa_longctx.wrapper as rpa_batched
else:
    import vllm_torchtpu.kernels.experimental.batched_rpa.wrapper as rpa_batched

ragged_paged_attention = rpa_default.ragged_paged_attention
ragged_paged_attention_batched = rpa_batched.ragged_paged_attention
get_kv_cache_shape = rpa_default.get_kv_cache_shape

ragged_paged_attention_hd64 = rpa_hd64.ragged_paged_attention_hd64
get_kv_cache_shape_hd64 = rpa_hd64.get_kv_cache_shape

mla_ragged_paged_attention = mla_v2_kernel.mla_ragged_paged_attention
sparse_mla_ragged_paged_attention = sparse_mla_kernel.sparse_ragged_paged_attention


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
) -> Tuple[jax.Array, jax.Array]:
    """Dispatches ragged paged attention over the bundled block-major KV cache across the TPU mesh.

    Args:
        kv_cache_bundle: Full-model KV cache bundle of shape `[num_pages, num_layers, ...]`.
        layer_idx: Dynamic scalar integer tensor specifying the target layer's bundle index.
        q: Query tensor of shape `[num_tokens, num_heads, head_dim]`.
        k: Key tensor of shape `[num_tokens, num_kv_heads, head_dim]`.
        v: Value tensor of shape `[num_tokens, num_kv_heads, head_dim]`.
        attention_metadata: Metadata containing sequence lengths, block tables, and batch bounds.
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
        A tuple of (new_bundle, output), where new_bundle aliases the input bundle in-place
        and output is the computed attention result tensor.
    """
    from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import \
        ragged_paged_attention_bundled

    if head_dim_original is None:
        head_dim_original = q.shape[-1]
    md = attention_metadata
    if sm_scale is None:
        sm_scale = head_dim_original**-0.5

    def _run(q, k, v, kv_cache_bundle, layer_idx, seq_lens, block_tables,
             query_start_loc, request_distribution):
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
            v_scale=v_scale)
        return new_bundle, output

    # Shard across KV heads along the TPU "model" axis (Tensor Parallelism):
    # - Dense bundle (6D): [num_pages, num_layers, page_size, num_kv_heads, p, head_dim] -> shard dim 3.
    # The layer dimension (dim 1) remains unsharded.
    # Query, Key, and Value shard on the head dimension (dim 1); metadata is replicated; layer_idx is a scalar.
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
    args = (q, k, v, kv_cache_bundle, layer_idx, md.seq_lens, md.block_tables,
            md.query_start_loc, md.request_distribution)

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

        num_kv_pages_per_blocks = (
            decode_tuned.num_kv_pages_per_block,
            1,
            mixed_tuned.num_kv_pages_per_block,
        )
        num_queries_per_blocks = (
            decode_tuned.num_queries_per_block,
            16,
            mixed_tuned.num_queries_per_block,
        )

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
            vmem_limit_bytes=min(decode_tuned.vmem_limit_bytes,
                                 mixed_tuned.vmem_limit_bytes),
            decode_batch_size=decode_tuned.decode_batch_size,
            mixed_q_split=mixed_tuned.q_split,
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
        k_scale: float | None = None
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sharded wrapper for the GLM-5.2 sparse (DSA) MLA Pallas kernel.

    Args:
      ql_nope: Query nope latents (`[num_tokens, num_heads, lkv_dim]`).
      q_pe: Decoupled RoPE query components (`[num_tokens, num_heads, rope_dim]`).
      kv_c_normed: This step's compressed KV latents (`[num_tokens, lkv_dim]`).
      k_pe: Decoupled RoPE keys (`[num_tokens, rope_dim]`).
      kv_cache_nope: Paged nope cache, shaped and typed by `nope_spec`.
      kv_cache_rope: Paged rope cache, shaped and typed by `rope_spec`.
      topk_indices: Indexer output specifying tokens to gather (`[num_tokens, topk]`).
      seq_lens: Per-sequence total KV length including the tokens being inserted in this step (`[num_seqs]`).
      block_tables: Flattened per-sequence-padded page table (`[num_seqs * pages_per_seq]`).
      query_start_loc: Cumulative new-token counts (`[num_seqs + 1]`).
      request_distribution: Tensor of (decode_end, prefill_end, num_seqs) indexing bounds.
      mesh: Target sharding mesh.
      nope_spec: Layout descriptor the nope cache was allocated from.
      rope_spec: Layout descriptor the rope cache was allocated from.
      sm_scale: Softmax temperature scale.
      k_scale: Dequantization scale for fp8 keys.

    Returns:
      A tuple containing:
        - The updated `kv_cache_nope`.
        - The updated `kv_cache_rope`.
        - The sparse attention output (`[num_tokens, num_heads, lkv_dim]`).

    Contract notes:
      * The kernel takes a SINGLE `q` of shape (T, N, nope+rope) in
        token-major order -- no separate rope argument, and no head-major
        transpose (the v2 kernel wants (N, T, L); this one does not).
      * `q` must be bf16 (pre-quantization): the kernel dequantizes the
        gathered KV to bf16 and has no q-scale plumbing.
      * Fused KV-cache insert: writes current step's `kv_c_normed` and `k_pe`
        into `kv_cache` before SparseCore gather execution.
      * Per-token kv_lens derive from the "-1" tail padding in
        `topk_indices`; every token must have at least one valid entry.
    """
    # dsa_gather moves one (`TILE_SUBROWS`, `TILE_LANE_BYTES`) uint8 tile of
    # nope and one `TILE_LANE_BYTES`-byte lane row of rope per token; other
    # head dims don't fit its address arithmetic.
    lkv_dim, rope_dim = kv_c_normed.shape[-1], k_pe.shape[-1]
    tile_subrows = dsa_gather.TILE_SUBROWS
    lane_bytes = dsa_gather.TILE_LANE_BYTES
    assert (
        lkv_dim == tile_subrows * lane_bytes and rope_dim * 2 == lane_bytes), (
            "dsa_gather used in the sparse MLA kernel needs the fp8 nope "
            f"head dimension to be {tile_subrows * lane_bytes} and the fp8 "
            f"rope head dimension to be {lane_bytes // 2}, got {lkv_dim}+"
            f"{rope_dim}")

    in_specs = (
        P(None, None, None),  # ql_nope
        P(None, None, None),  # q_pe
        P(None, None),  # kv_c_normed
        P(None, None),  # k_pe
        P(None),  # kv_cache_nope
        P(None),  # kv_cache_rope
        P(None),  # topk_indices
        P(None),  # seq_lens
        P(None),  # block_tables
        P(None),  # query_start_loc
        P(None),  # request_distribution
    )
    out_specs = (
        P(None, None, None),  # attn output (T, N, D)
        P(None),  # updated nope cache
        P(None),  # updated rope cache
    )

    dequant_scale = float(k_scale) if k_scale is not None else 1.0

    def _sparse_mla_ragged_paged_attention(ql_nope, q_pe, kv_c_normed, k_pe,
                                           kv_cache_nope, kv_cache_rope,
                                           topk_idx, seq_lens_, block_tables_,
                                           query_start_loc_,
                                           request_distribution_):
        kv_cache_nope, kv_cache_rope = update_sparse_mla_kv_cache(
            kv_cache_nope,
            kv_cache_rope,
            kv_c_normed,
            k_pe,
            seq_lens_,
            block_tables_,
            query_start_loc_,
            nope_spec=nope_spec,
            rope_spec=rope_spec)

        q = jnp.concatenate([ql_nope, q_pe], axis=-1)
        output = sparse_mla_ragged_paged_attention(
            q,
            kv_cache_nope,
            kv_cache_rope,
            topk_idx,
            block_tables_,
            query_start_loc_,
            request_distribution_,
            sm_scale=sm_scale or 1.0,
            k_scale=dequant_scale,
        )

        lkv_dim = ql_nope.shape[-1]
        # The kernel's output is lkv_dim + rope_dim wide because it attends
        # over concatenated (nope + rope) keys. Only the nope part is the MLA
        # value (P @ latent); drop the rope tail before the W_UV projection.
        return output[..., :lkv_dim], kv_cache_nope, kv_cache_rope

    output, new_kv_cache_nope, new_kv_cache_rope = jax.jit(
        shard_map.shard_map(_sparse_mla_ragged_paged_attention,
                            mesh=mesh,
                            in_specs=in_specs,
                            out_specs=out_specs,
                            check_rep=False))(ql_nope, q_pe, kv_c_normed, k_pe,
                                              kv_cache_nope, kv_cache_rope,
                                              topk_indices, seq_lens,
                                              block_tables, query_start_loc,
                                              request_distribution)
    return new_kv_cache_nope, new_kv_cache_rope, output
