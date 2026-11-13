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
"""DeepSeek-V4.1 SWA-only and CSA attention ops, kept apart from the V4 ops."""

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.core_attention.mla_swa import (
    mla_sliding_window_ragged_paged_attention,
)
from vllm_torchtpu.kernels.deepseek_v41.sparse_mla import (
    sparse_ragged_paged_attention,
)
from vllm_torchtpu.utils import align_to


def prepare_swa_page_indices_jax(
    swa_page_indices: jax.Array,
    *,
    sw_cache_num_pages: int,
) -> jax.Array:
    """Maps raw SWA page table indices onto the physical ring-buffer once per stage."""
    if sw_cache_num_pages > 1:
        return jnp.where(
            swa_page_indices > 0,
            ((swa_page_indices - 1) % (sw_cache_num_pages - 1)) + 1,
            0,
        )
    return swa_page_indices


def prepare_swa_metadata_jax(
    swa_kv_lens: jax.Array,
    swa_page_indices: jax.Array,
    swa_cu_q_lens: jax.Array,
    *,
    num_tokens: int,
    sw_cache_num_pages: int,
    sliding_window: int,
    logical_page_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Slices SWA page table to reachable window pages and ring-maps once per stage."""
    return _prepare_swa_metadata(
        swa_kv_lens,
        swa_page_indices,
        swa_cu_q_lens,
        num_tokens=num_tokens,
        sw_cache_num_pages=sw_cache_num_pages,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
        page_indices_ring_mapped=False,
    )


def _prepare_swa_metadata(
    swa_kv_lens: jax.Array,
    swa_page_indices: jax.Array,
    swa_cu_q_lens: jax.Array,
    *,
    num_tokens: int,
    sw_cache_num_pages: int,
    sliding_window: int,
    logical_page_size: int,
    page_indices_ring_mapped: bool = False,
) -> tuple[jax.Array, jax.Array]:
    max_num_seqs = swa_kv_lens.shape[0]
    pages_per_seq = swa_page_indices.size // max_num_seqs
    window_pages = min(
        pages_per_seq,
        align_to(
            (num_tokens + sliding_window) // logical_page_size + 2,
            16,
        ),
    )
    # Trim oversized page tables BEFORE ring-mapping so scalar prefetch into
    # SMEM and modulo mapping operate on ~16 pages/seq instead of 2048.
    if pages_per_seq > window_pages * 2:
        eff_span_packed = swa_kv_lens >> 20
        raw_kv_lens = swa_kv_lens & 0xFFFFF
        q_lens = swa_cu_q_lens[1:] - swa_cu_q_lens[:-1]
        eff_span = jnp.maximum(q_lens, eff_span_packed)
        base_pages = (
            jnp.maximum(0, raw_kv_lens - eff_span - sliding_window) // logical_page_size
        )
        raw_kv_lens = jnp.maximum(0, raw_kv_lens - base_pages * logical_page_size)
        swa_kv_lens = raw_kv_lens | (eff_span_packed << 20)
        page_table_2d = swa_page_indices[
            : max_num_seqs * pages_per_seq
        ].reshape(max_num_seqs, pages_per_seq)
        col_idx = jnp.minimum(
            base_pages[:, None] + jnp.arange(window_pages, dtype=jnp.int32)[None, :],
            pages_per_seq - 1,
        )
        pad_seqs = align_to(max_num_seqs, 8) - max_num_seqs
        if pad_seqs > 0:
            page_table_2d = jnp.pad(page_table_2d, ((0, pad_seqs), (0, 0)))
            col_idx = jnp.pad(col_idx, ((0, pad_seqs), (0, 0)))
        gathered = jnp.take_along_axis(page_table_2d, col_idx, axis=1)
        if pad_seqs > 0:
            gathered = gathered[:max_num_seqs]
        swa_page_indices = gathered.reshape(-1)
    if sw_cache_num_pages > 1 and not page_indices_ring_mapped:
        swa_page_indices = jnp.where(
            swa_page_indices > 0,
            ((swa_page_indices - 1) % (sw_cache_num_pages - 1)) + 1,
            0,
        )
    return swa_kv_lens, swa_page_indices


def _run_swa(
    q: jax.Array,
    new_kv: jax.Array,
    sw_cache: jax.Array,
    swa_kv_lens: jax.Array,
    swa_page_indices: jax.Array,
    swa_cu_q_lens: jax.Array,
    swa_distribution: jax.Array,
    attention_sinks: jax.Array,
    *,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    swa_only: bool,
    ced_decoder: bool = False,
    token_parallel: bool = False,
    page_indices_ring_mapped: bool = False,
    non_causal_block: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """SWA pass; output stays unnormalized when CSA follows (folds swa_l/swa_m)."""
    if sw_cache.shape[0] == 0:
        # Profiling-shape trace: no cache to attend over.
        return (
            jnp.zeros_like(q),
            sw_cache,
            jnp.zeros((q.shape[0], q.shape[1]), dtype=jnp.float32),
            jnp.full((q.shape[0], q.shape[1]), -1e9, dtype=jnp.float32),
        )
    max_num_seqs = swa_kv_lens.shape[0]
    eff_max_seqs = min(max_num_seqs, max(8, align_to(q.shape[0], 8)))
    if eff_max_seqs < max_num_seqs:
        pages_per_seq = swa_page_indices.size // max_num_seqs
        swa_kv_lens = swa_kv_lens[:eff_max_seqs]
        swa_page_indices = swa_page_indices[: eff_max_seqs * pages_per_seq]
        swa_cu_q_lens = swa_cu_q_lens[: eff_max_seqs + 1]
        swa_distribution = jnp.minimum(swa_distribution, eff_max_seqs)
    swa_kv_lens, swa_page_indices = _prepare_swa_metadata(
        swa_kv_lens,
        swa_page_indices,
        swa_cu_q_lens,
        num_tokens=q.shape[0],
        sw_cache_num_pages=sw_cache.shape[0],
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
        page_indices_ring_mapped=page_indices_ring_mapped,
    )
    bq_prefill = 64 if q.shape[1] > 64 else 128
    bkv_p_decode = max(2, (sliding_window + logical_page_size - 1) // logical_page_size)
    bkv_p_prefill = max(
        2, (sliding_window + bq_prefill + logical_page_size - 1) // logical_page_size
    )
    return mla_sliding_window_ragged_paged_attention(
        q=q,
        new_kv=new_kv,
        cache_kv=sw_cache,
        kv_lens=swa_kv_lens,
        page_indices=swa_page_indices,
        cu_q_lens=swa_cu_q_lens,
        distribution=swa_distribution,
        attention_sinks=attention_sinks,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
        # Tuned on TPU v7x:
        # - Decode (idx 0): bkv_p = cdiv(sliding_window, logical_page_size) so
        #   bkv_sz >= sliding_window activates run_mla_batched_decode_kernel.
        # - Prefill/Mixed (idx 1, 2): bkv_p = cdiv(sliding_window + bq_prefill,
        #   logical_page_size) covers the entire (bq_sz + sliding_window - 1) span
        #   in a single bkv block across H in {16, 64, 128} so acc_scratch is (1, D)
        #   and never spills or OOMs VMEM.
        num_kv_pages_per_block=(
            (1, 1, 1)
            if sliding_window == 1
            else (
                bkv_p_decode,
                bkv_p_prefill,
                bkv_p_prefill,
            )
        ),
        num_queries_per_block=(
            1,
            bq_prefill,
            8 if ced_decoder else bq_prefill,
        ),
        q_compute_block_size=(
            8
            if (q.shape[1] > 64 and q.shape[0] > 384)
            else (16 if q.shape[1] > 32 else (32 if q.shape[1] > 8 else 64))
        ),
        decode_seq_batch_size=(
            8
            if q.shape[0] <= 16
            else (
                16
                if q.shape[0] <= 128
                else (8 if q.shape[1] > 32 else 32)
            )
        ),
        unnormalized_output=not swa_only or sliding_window == 1,
        ced_decoder=ced_decoder,
        keep_lm_aligned=not swa_only,
        skip_kv_update=token_parallel,
        verify_q_len=_resolve_verify_q_len(q.shape[0], non_causal_block),
        non_causal_block=non_causal_block,
    )


def _resolve_verify_q_len(q_tokens: int, non_causal_block: bool) -> int:
    if q_tokens > 1536:
        return 0
    try:
        from vllm_torchtpu.layers.adapter.utils import (
            get_vllm_model_wrapper_context,
        )

        vllm_cfg = getattr(get_vllm_model_wrapper_context(), "vllm_config", None)
        spec_cfg = getattr(vllm_cfg, "speculative_config", None)
        num_spec = getattr(spec_cfg, "num_speculative_tokens", None)
        if isinstance(num_spec, int) and num_spec >= 1:
            expected_q = num_spec if non_causal_block else (num_spec + 1)
            if 2 <= expected_q <= 8 and q_tokens % expected_q == 0:
                return expected_q
        if vllm_cfg is not None:
            return 0
    except Exception:
        pass
    candidates = (5, 3, 4) if non_causal_block else (6, 5, 4)
    for cand in candidates:
        if q_tokens % cand == 0:
            return cand
    return 0


# Module-level so pallas.jax_op can trace and register it as a torch op.
def _attention_swa(
    q: jax.Array,
    new_kv: jax.Array,
    sw_cache: jax.Array,
    swa_kv_lens: jax.Array,
    swa_page_indices: jax.Array,
    swa_cu_q_lens: jax.Array,
    swa_distribution: jax.Array,
    main_cache_kv: jax.Array,
    main_kv_lens: jax.Array,
    extra: jax.Array,
    main_page_indices: jax.Array,
    main_cu_q_lens: jax.Array,
    main_distribution: jax.Array,
    attention_sinks: jax.Array,
    *,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    ced_decoder: bool = False,
    token_parallel: bool = False,
    page_indices_ring_mapped: bool = False,
    non_causal_block: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """SWA-only layers; the main-cache inputs keep the CSA op's signature."""
    output, updated_sw_cache, _, _ = _run_swa(
        q=q,
        new_kv=new_kv,
        sw_cache=sw_cache,
        swa_kv_lens=swa_kv_lens,
        swa_page_indices=swa_page_indices.flatten(),
        swa_cu_q_lens=swa_cu_q_lens,
        swa_distribution=swa_distribution,
        attention_sinks=attention_sinks,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
        swa_only=True,
        ced_decoder=ced_decoder,
        token_parallel=token_parallel,
        page_indices_ring_mapped=page_indices_ring_mapped,
        non_causal_block=non_causal_block,
    )
    return output, updated_sw_cache


def _attention_csa(
    q: jax.Array,
    new_kv: jax.Array,
    sw_cache: jax.Array,
    swa_kv_lens: jax.Array,
    swa_page_indices: jax.Array,
    swa_cu_q_lens: jax.Array,
    swa_distribution: jax.Array,
    main_cache_kv: jax.Array,
    main_kv_lens: jax.Array,
    extra: jax.Array,  # encode_csa_gather_indices output from the indexer
    main_page_indices: jax.Array,
    main_cu_q_lens: jax.Array,
    main_distribution: jax.Array,
    attention_sinks: jax.Array,
    main_cache_rope: jax.Array,
    union_gather_indices: jax.Array | None = None,
    cand_bits_tiled: jax.Array | None = None,
    n_unique: jax.Array | None = None,
    pre_gathered_nope: jax.Array | None = None,
    pre_gathered_rope: jax.Array | None = None,
    *,
    precomputed_union: tuple[jax.Array, ...]
    | None = None,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    two_caches_same_buffer: bool,
    ced_decoder: bool = False,
    token_parallel: bool = False,
    page_indices_ring_mapped: bool = False,
    non_causal_block: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """CSA: SWA pass, then sparse attention over the indexer's top-k rows."""
    if (
        sw_cache.shape[0] if two_caches_same_buffer else main_cache_kv.shape[0]
    ) == 0 or main_cache_rope.shape[0] == 0:
        return jnp.zeros(q.shape, dtype=q.dtype), sw_cache

    swa_output, updated_sw_cache, swa_l, swa_m = _run_swa(
        q=q,
        new_kv=new_kv,
        sw_cache=sw_cache,
        swa_kv_lens=swa_kv_lens,
        swa_page_indices=swa_page_indices.flatten(),
        swa_cu_q_lens=swa_cu_q_lens,
        swa_distribution=swa_distribution,
        attention_sinks=attention_sinks,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
        swa_only=False,
        ced_decoder=ced_decoder,
        token_parallel=token_parallel,
        page_indices_ring_mapped=page_indices_ring_mapped,
        non_causal_block=non_causal_block,
    )

    # When both caches are one buffer, the CSA read must see the SWA kernel's
    # write from this pass, not the pre-write snapshot.
    if two_caches_same_buffer:
        main_cache_kv = updated_sw_cache

    eff_union = (
        (
            union_gather_indices,
            cand_bits_tiled,
            n_unique,
            pre_gathered_nope,
            pre_gathered_rope,
        )
        if not two_caches_same_buffer
        and union_gather_indices is not None
        and cand_bits_tiled is not None
        and n_unique is not None
        and pre_gathered_nope is not None
        and pre_gathered_rope is not None
        else (
            (union_gather_indices, cand_bits_tiled, n_unique)
            if union_gather_indices is not None
            and cand_bits_tiled is not None
            and n_unique is not None
            else precomputed_union
        )
    )

    output = sparse_ragged_paged_attention(
        q=q,
        cache_kv_nope=main_cache_kv,
        cache_kv_rope=main_cache_rope,
        topk_indices=extra,
        cu_q_lens=main_cu_q_lens,
        distribution=main_distribution,
        attention_sinks=attention_sinks,
        swa_accumution=swa_output,
        swa_l=swa_l,
        swa_m=swa_m,
        precomputed_union=eff_union,
        sm_scale=sm_scale,
        gather_and_attention_chunk_size=min(
            q.shape[0], 256 if q.shape[1] == 64 else 512
        ),
        ced_decoder=ced_decoder,
    )
    return output, updated_sw_cache
