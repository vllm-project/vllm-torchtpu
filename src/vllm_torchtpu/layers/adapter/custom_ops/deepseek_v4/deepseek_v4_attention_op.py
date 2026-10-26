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

import jax
import jax.numpy as jnp
import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention_layer_base import (
    AttentionBackend,  # isort: skip
)
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWABackend,
    DeepseekV4SWACache,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowMLASpec

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.deepseek_v4.core_attention import (
    csa_union_dense,
    csa_window_union,
)
from vllm_torchtpu.kernels.deepseek_v4.core_attention import fused_hca_decode
from vllm_torchtpu.kernels.deepseek_v4.core_attention import fused_hca_mixed
from vllm_torchtpu.kernels.deepseek_v4.core_attention.mla import (
    mla_ragged_paged_attention,
)
from vllm_torchtpu.kernels.deepseek_v4.core_attention.mla_swa import (
    mla_sliding_window_ragged_paged_attention,
)
from vllm_torchtpu.kernels.deepseek_v4.core_attention.sparse_mla import (
    sparse_ragged_paged_attention,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import align_to

logger = init_logger(__name__)

BATCH_AXIS = None

# Tokens per SWA KV block, tuned on microbenchmarks as two 128-token pages. The
# page count follows the SWA page size (64 tokens at the default 512-token KV
# block): a fixed count doubles the block at 128-token pages, which overflows
# VMEM.
_SWA_KV_BLOCK_TOKENS = 256


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
    non_causal_block: bool = False,
    verify_q_len: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Sliding-window pass: updates the SWA cache and returns its partials.

    When a compressed-KV pass follows, the output is left unnormalized so
    that pass can fold `swa_l` / `swa_m` into its own softmax accumulation.
    """
    if sw_cache.shape[0] == 0:
        # Profiling-shape trace: no cache to attend over.
        return (
            jnp.zeros_like(q),
            sw_cache,
            jnp.zeros((q.shape[0], q.shape[1]), dtype=jnp.float32),
            jnp.full((q.shape[0], q.shape[1]), -1e9, dtype=jnp.float32),
        )
    kv_pages_per_block = max(1, envs.TPU_DSV4_SWA_KV_BLOCK_TOKENS // logical_page_size)
    single_pass = (
        envs.TPU_DSV4_SWA_SINGLE_PASS
        and q.shape[0] <= envs.TPU_DSV4_SWA_SINGLE_PASS_MAX_TOKENS
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
        # The following parameters are tuned based on microbenchmark results.
        num_kv_pages_per_block=(kv_pages_per_block,) * 3,
        num_queries_per_block=envs.TPU_DSV4_SWA_NUM_QUERIES_PER_BLOCK,
        q_compute_block_size=envs.TPU_DSV4_SWA_Q_COMPUTE_BLOCK_SIZE,
        unnormalized_output=not swa_only,
        non_causal_block=non_causal_block,
        chunk_prefill_size=verify_q_len,
        single_pass=single_pass,
        single_pass_bq=envs.TPU_DSV4_SWA_SINGLE_PASS_BQ if single_pass else None,
    )


# Module-level so `pallas.jax_op` can trace and register it as a torch op.
def _attention_hca(
    q: jax.Array,
    new_kv: jax.Array,
    sw_cache: jax.Array,
    swa_kv_lens: jax.Array,
    swa_page_indices: jax.Array,
    swa_cu_q_lens: jax.Array,
    swa_distribution: jax.Array,
    main_cache_kv: jax.Array,
    main_kv_lens: jax.Array,
    extra: jax.Array,  # pass kv_lens_to_attend for HCA
    main_page_indices: jax.Array,
    main_cu_q_lens: jax.Array,
    main_distribution: jax.Array,
    attention_sinks: jax.Array,
    *,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    swa_only: bool,
    two_caches_same_buffer: bool,
    hca_num_kv_pages_per_block: tuple[int, int, int],
    hca_num_queries_per_block: tuple[int, int, int],
    non_causal_block: bool = False,
    verify_q_len: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """SWA-only layers, and HCA layers over a single raw-bf16 latent array.

    `verify_q_len`: requests [distribution[0], distribution[1]) are speculative
    verify windows of exactly this many tokens; they run the kernels' static
    query-length pass instead of the mixed one.
    """
    if main_cache_kv.shape[0] == 0 or (swa_only and sw_cache.shape[0] == 0):
        # Profiling-shape trace: the kernel is skipped. Unused inputs stay in
        # the compiled signature because torch_tpu jits with keep_unused=True.
        return jnp.zeros(q.shape, dtype=q.dtype), sw_cache

    chunk = verify_q_len or 8
    if envs.TPU_DSV4_FUSED_HCA_DECODE and fused_hca_decode.supported(
        tokens=q.shape[0],
        heads=q.shape[1],
        chunk=chunk,
        two_caches_same_buffer=two_caches_same_buffer,
        swa_only=swa_only,
        non_causal_block=non_causal_block,
    ):
        # Static gate (the step's bucket): the kernel serves every request
        # shape of the step, so no runtime branch (STRAT-DSV4-01).
        return fused_hca_decode.fused_hca_attention(
            q,
            new_kv,
            sw_cache,
            swa_kv_lens,
            swa_page_indices,
            swa_cu_q_lens,
            swa_distribution,
            main_cache_kv,
            extra,
            main_page_indices,
            attention_sinks,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            logical_page_size=logical_page_size,
            chunk=chunk,
            blk_pages=envs.TPU_DSV4_FUSED_HCA_BLK_PAGES,
        )

    mixed_bq = max(8, envs.TPU_DSV4_FUSED_HCA_MIXED_Q_ROWS // max(q.shape[1], 1))
    if envs.TPU_DSV4_FUSED_HCA_MIXED and fused_hca_mixed.supported(
        tokens=q.shape[0],
        heads=q.shape[1],
        chunk=chunk,
        bq=mixed_bq,
        two_caches_same_buffer=two_caches_same_buffer,
        swa_only=swa_only,
        non_causal_block=non_causal_block,
    ):
        # Static gate (T > 32 bucket; T <= 32 stays on fused_hca_decode above).
        return fused_hca_mixed.fused_hca_mixed_attention(
            q,
            new_kv,
            sw_cache,
            swa_kv_lens,
            swa_page_indices,
            swa_cu_q_lens,
            swa_distribution,
            main_cache_kv,
            extra,
            main_page_indices,
            attention_sinks,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            logical_page_size=logical_page_size,
            chunk=chunk,
            bq=mixed_bq,
            kt=envs.TPU_DSV4_FUSED_HCA_MIXED_KT,
            kt_tail=envs.TPU_DSV4_FUSED_HCA_MIXED_KT_TAIL,
            stats128=True,
            swa_fc=True,
            blk_pages=envs.TPU_DSV4_FUSED_HCA_BLK_PAGES,
        )

    swa_page_indices = swa_page_indices.flatten()
    main_page_indices = main_page_indices.flatten()

    swa_output, updated_sw_cache, swa_l, swa_m = _run_swa(
        q=q,
        new_kv=new_kv,
        sw_cache=sw_cache,
        swa_kv_lens=swa_kv_lens,
        swa_page_indices=swa_page_indices,
        swa_cu_q_lens=swa_cu_q_lens,
        swa_distribution=swa_distribution,
        attention_sinks=attention_sinks,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
        swa_only=swa_only,
        non_causal_block=non_causal_block,
        verify_q_len=verify_q_len,
    )

    if swa_only:
        return swa_output, updated_sw_cache

    # When both caches are one buffer, the compressed-KV read must see the
    # SWA kernel's write from this pass, not the pre-write snapshot.
    if two_caches_same_buffer:
        main_cache_kv = updated_sw_cache

    output = mla_ragged_paged_attention(
        q=q,
        cache_kv=main_cache_kv,
        kv_lens=main_kv_lens,
        kv_lens_to_attend=extra,
        page_indices=main_page_indices,
        cu_q_lens=main_cu_q_lens,
        distribution=main_distribution,
        attention_sinks=attention_sinks,
        swa_accumution=swa_output,
        swa_l=swa_l,
        swa_m=swa_m,
        sm_scale=sm_scale,
        num_kv_pages_per_block=hca_num_kv_pages_per_block,
        num_queries_per_block=hca_num_queries_per_block,
        chunk_prefill_size=verify_q_len,
        prefetch_early=envs.TPU_DSV4_HCA_PREFETCH_EARLY,
    )
    return output, updated_sw_cache


def _attention_hca_rope(
    q_flat: jax.Array,  # [T, H * head_dim]: the raw q projection (before RMSNorm + RoPE)
    kv_raw: jax.Array,  # [T, head_dim]: kv after kv_norm, before RoPE
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
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    *,
    head_dim: int,
    q_eps: float,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    swa_only: bool,
    two_caches_same_buffer: bool,
    hca_num_kv_pages_per_block: tuple[int, int, int],
    hca_num_queries_per_block: tuple[int, int, int],
    non_causal_block: bool = False,
    verify_q_len: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """`_attention_hca` with the q per-head RMSNorm + RoPE and the kv RoPE inside
    (TPU_DSV4_FUSED_HCA_ROPE): in the fused kernel's prologue when it serves the
    step, else the stock rope kernels then `_attention_hca`."""
    from vllm_torchtpu.kernels.deepseek_v4 import rope as rope_kernel

    kwargs = dict(
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
        swa_only=swa_only,
        two_caches_same_buffer=two_caches_same_buffer,
        hca_num_kv_pages_per_block=hca_num_kv_pages_per_block,
        hca_num_queries_per_block=hca_num_queries_per_block,
        non_causal_block=non_causal_block,
        verify_q_len=verify_q_len,
    )
    T = q_flat.shape[0]
    H = q_flat.shape[1] // head_dim
    chunk = verify_q_len or 8
    if (
        main_cache_kv.shape[0] != 0
        and envs.TPU_DSV4_FUSED_HCA_DECODE
        and q_flat.dtype == jnp.bfloat16
        and rope_kernel.can_untile_heads(T, H, head_dim)
        and fused_hca_decode.supported(
            tokens=T,
            heads=H,
            chunk=chunk,
            two_caches_same_buffer=kwargs["two_caches_same_buffer"],
            swa_only=kwargs["swa_only"],
            non_causal_block=kwargs.get("non_causal_block", False),
        )
    ):
        return fused_hca_decode.fused_hca_attention_rope(
            q_flat,
            kv_raw,
            positions,
            cos_sin_cache,
            sw_cache,
            swa_kv_lens,
            swa_page_indices,
            swa_cu_q_lens,
            swa_distribution,
            main_cache_kv,
            extra,
            main_page_indices,
            attention_sinks,
            eps=q_eps,
            heads=H,
            sm_scale=kwargs["sm_scale"],
            sliding_window=kwargs["sliding_window"],
            logical_page_size=kwargs["logical_page_size"],
            chunk=chunk,
            blk_pages=envs.TPU_DSV4_FUSED_HCA_BLK_PAGES,
        )
    q = rope_kernel.qnorm_rope_flat(
        q_flat, positions, cos_sin_cache, num_heads=H, eps=q_eps
    )
    kv = rope_kernel.rope(kv_raw, positions, cos_sin_cache)
    return _attention_hca(
        q,
        kv,
        sw_cache,
        swa_kv_lens,
        swa_page_indices,
        swa_cu_q_lens,
        swa_distribution,
        main_cache_kv,
        main_kv_lens,
        extra,
        main_page_indices,
        main_cu_q_lens,
        main_distribution,
        attention_sinks,
        **kwargs,
    )


def _window_mixed_prepare(topk, kv_lens, page_indices, cu_q_lens, distribution,
                          cache_kv, cache_rope, groups, wq=8):
    """L1 of the mixed-bucket union-of-window route: one SparseCore call, or one
    per group (`groups` > 1) so group g + 1's runs under group g's attention.
    `wq` != 8 (16-token windows) always takes the grouped layout."""
    if groups > 1 or wq != 8:
        return csa_window_union.prepare_grouped(
            topk, kv_lens, page_indices, cu_q_lens, distribution, cache_kv,
            cache_rope, groups=groups, wq=wq,
        )  # fmt: skip
    return csa_window_union.prepare(
        topk, kv_lens, page_indices, cu_q_lens, distribution, cache_kv, cache_rope,
    )  # fmt: skip


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
    extra: jax.Array,  # pass topk_indices for CSA
    main_page_indices: jax.Array,
    main_cu_q_lens: jax.Array,
    main_distribution: jax.Array,
    attention_sinks: jax.Array,
    main_cache_rope: jax.Array,
    *,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    swa_only: bool,
    two_caches_same_buffer: bool,
    csa_gather_and_attention_chunk_size: int,
    non_causal_block: bool = False,
    verify_q_len: int | None = None,
    csa_gather_no_pad: bool = False,
    union_dense: bool = False,
    union_dense_min_tokens: int = 512,
    union_dense_rows_per_token: int = 160,
    union_dense_max_heads: int = 32,
    union_dense_max_kv_tp4: int = 0,
    union_dense_plan_unconditional: bool = False,
    topk_after_q_above: int = 0,
    topk_after_kv: bool = False,
    union_verify: bool = False,
    union_verify_max_tokens: int = 128,
    union_verify_above: int = 0,
    union_verify_early: bool = True,
    union_verify_fence: int = 0,
    window_mixed: bool = False,
    window_mixed_above: int = 256,
    window_mixed_min_heads: int = 32,
    window_mixed_groups: int = 1,
    window_mixed_max_tokens: int = 2048,
    window_mixed_wq: int = 8,
) -> tuple[jax.Array, jax.Array]:
    """CSA: sparse gather over top-k rows, NoPE and RoPE in two arrays.

    `verify_q_len` as in `_attention_hca`; the sparse gather is per token and
    reads only distribution[2].
    """
    if main_cache_kv.shape[0] == 0 or main_cache_rope.shape[0] == 0:
        return jnp.zeros(q.shape, dtype=q.dtype), sw_cache

    swa_page_indices = swa_page_indices.flatten()
    main_page_indices = main_page_indices.flatten()

    if topk_after_q_above and q.shape[0] > topk_after_q_above:
        # TPU_DSV4_CSA_TOPK_AFTER_Q_ABOVE: the top-k (under the indexer token
        # split, an all-gather of every TP rank's rows) is first needed by the
        # union plan issued below, which XLA starts as early as it can -- so
        # the gather's done sat right behind its start, ~21 us exposed per
        # CSA layer at c16 T512, while the q projection and its norm/rope,
        # which do not need it, waited. Fencing the top-k with q moves the
        # plan's start behind them and lets the gather complete under them;
        # the plan still has the SWA pass and the compressor to run behind.
        # Scheduling only: same values.
        if topk_after_kv:
            # TPU_DSV4_CSA_TOPK_AFTER_KV: also behind the roped kv, so the
            # top-k all-gather (issued once the q projection is fenced with
            # the top-k, TPU_DSV4_CSA_TOPK_FENCE_Q) completes under kv_rope.
            extra, q, new_kv = jax.lax.optimization_barrier((extra, q, new_kv))
        else:
            extra, q = jax.lax.optimization_barrier((extra, q))

    swa_output, updated_sw_cache, swa_l, swa_m = _run_swa(
        q=q,
        new_kv=new_kv,
        sw_cache=sw_cache,
        swa_kv_lens=swa_kv_lens,
        swa_page_indices=swa_page_indices,
        swa_cu_q_lens=swa_cu_q_lens,
        swa_distribution=swa_distribution,
        attention_sinks=attention_sinks,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
        swa_only=False,
        non_causal_block=non_causal_block,
        verify_q_len=verify_q_len,
    )

    if two_caches_same_buffer:
        main_cache_kv = updated_sw_cache

    gather_and_attention_chunk_size = csa_gather_and_attention_chunk_size
    attention_kernel_batch_size = 16
    # small chunks override:
    if q.shape[0] <= 128:
        gather_and_attention_chunk_size = 32
        attention_kernel_batch_size = 8

    # First use of the top-k after the SWA pass: the indexer's SparseCore
    # top-k then completes behind the q projection and SWA, which do not need
    # it, instead of stalling the TensorCore before them.
    # TPU_DSV4_CSA_WINDOW_MIXED: buckets of T > ..._ABOVE tokens (mixed /
    # prefill) take the union-of-window route of the decode buckets
    # (`csa_window_union`, 8-token windows, assignment waves over the
    # SparseCore workers) instead of union-dense: an 8-token window's union is
    # ~half a 32-token block's (real top-k), the union tile's cost is per query
    # row (8 x 32 heads = 256 rows per tile at TP4), and there is no context
    # cap, no count pass and no sparse fallback.
    use_window_mixed = (
        window_mixed
        and q.shape[0] > window_mixed_above
        and q.shape[0] <= window_mixed_max_tokens
        and q.shape[0] > union_verify_max_tokens
        and q.shape[1] >= window_mixed_min_heads
        and csa_window_union.supported_mixed(q.shape[0], main_cu_q_lens.shape[0] - 1)
    )
    use_union_dense = (
        union_dense
        and not use_window_mixed
        and q.shape[0] >= union_dense_min_tokens
        and q.shape[1] <= union_dense_max_heads
    )
    # ~1024 MXU rows per pushed KV tile: 64 tokens x 16 heads (TP8), 32 x 32.
    union_bq = 64 if q.shape[1] <= 16 else 32
    union_plan = None
    if use_union_dense:
        # The union plan needs only the top-k: issued here, its SparseCore
        # kernels run behind the SWA pass on the TensorCore.
        union_plan = csa_union_dense.prepare(
            extra,
            main_kv_lens,
            main_page_indices,
            main_cu_q_lens,
            main_distribution,
            page_size=main_cache_kv.shape[1],
            bq=union_bq,
            rows_per_token=union_dense_rows_per_token,
            # TP4 (32 heads, T512 buckets): one pass, two subcores per block.
            count_first=q.shape[1] <= 16,
            split=1 if q.shape[1] <= 16 else 2,
            **(
                {"max_kv": union_dense_max_kv_tp4}
                if q.shape[1] > 16 and union_dense_max_kv_tp4 > 0
                else {}
            ),
            unconditional=union_dense_plan_unconditional and q.shape[1] > 16,
        )
    # Decode buckets: union of each <= 8-token window (`csa_window_union`).
    # Its SparseCore call needs only the top-k and the compressed cache, so it
    # is issued here and runs behind the SWA pass -- unless the compressed
    # cache is the SWA kernel's output buffer.
    use_union_verify = (
        union_verify
        and not use_union_dense
        and q.shape[0] <= union_verify_max_tokens
        and q.shape[0] > union_verify_above
        and csa_window_union.supported(q.shape[0], main_cu_q_lens.shape[0] - 1)
    )
    union_rows = None
    if use_window_mixed and not two_caches_same_buffer:
        union_rows = _window_mixed_prepare(
            extra, main_kv_lens, main_page_indices, main_cu_q_lens,
            main_distribution, main_cache_kv, main_cache_rope, window_mixed_groups,
            window_mixed_wq,
        )  # fmt: skip
    if use_union_verify and union_verify_early and not two_caches_same_buffer:
        topk_for_union = extra
        if union_verify_fence == 1:
            topk_for_union, _ = jax.lax.optimization_barrier((extra, q))
        elif union_verify_fence >= 2:
            topk_for_union, _, _ = jax.lax.optimization_barrier((extra, q, new_kv))
        union_rows = csa_window_union.prepare(
            topk_for_union, main_kv_lens, main_page_indices, main_cu_q_lens,
            main_distribution, main_cache_kv, main_cache_rope,
        )  # fmt: skip
    extra, swa_output, union_plan, union_rows = jax.lax.optimization_barrier(
        (extra, swa_output, union_plan, union_rows)
    )

    if use_window_mixed:
        if union_rows is None:
            union_rows = _window_mixed_prepare(
                extra, main_kv_lens, main_page_indices, main_cu_q_lens,
                main_distribution, main_cache_kv, main_cache_rope, window_mixed_groups,
                window_mixed_wq,
            )  # fmt: skip
        if window_mixed_groups > 1 or window_mixed_wq != 8:
            output = csa_window_union.attend_grouped(
                q, union_rows, attention_sinks, swa_output, swa_l, swa_m,
                sm_scale=sm_scale, groups=window_mixed_groups, wq=window_mixed_wq,
            )  # fmt: skip
        else:
            output = csa_window_union.attend(
                q, union_rows, attention_sinks, swa_output, swa_l, swa_m,
                sm_scale=sm_scale,
            )  # fmt: skip
        return output, updated_sw_cache

    if use_union_verify:
        if union_rows is None:
            union_rows = csa_window_union.prepare(
                extra, main_kv_lens, main_page_indices, main_cu_q_lens,
                main_distribution, main_cache_kv, main_cache_rope,
            )  # fmt: skip
        output = csa_window_union.attend(
            q, union_rows, attention_sinks, swa_output, swa_l, swa_m,
            sm_scale=sm_scale,
        )  # fmt: skip
        return output, updated_sw_cache

    if use_union_dense:
        output = csa_union_dense.attention(
            q,
            main_cache_kv,
            main_cache_rope,
            extra,
            main_kv_lens,
            main_page_indices,
            main_cu_q_lens,
            main_distribution,
            attention_sinks,
            swa_output,
            swa_l,
            swa_m,
            sm_scale=sm_scale,
            bq=union_bq,
            rows_per_token=union_dense_rows_per_token,
            sparse_chunk=gather_and_attention_chunk_size,
            sparse_batch=attention_kernel_batch_size,
            gather_no_pad=csa_gather_no_pad,
            plan=union_plan,
        )
        return output, updated_sw_cache

    output = sparse_ragged_paged_attention(
        q=q,
        cache_kv_nope=main_cache_kv,
        cache_kv_rope=main_cache_rope,
        topk_indices=extra,
        page_indices=main_page_indices,
        cu_q_lens=main_cu_q_lens,
        distribution=main_distribution,
        attention_sinks=attention_sinks,
        swa_accumution=swa_output,
        swa_l=swa_l,
        swa_m=swa_m,
        sm_scale=sm_scale,
        gather_and_attention_chunk_size=gather_and_attention_chunk_size,
        attention_kernel_batch_size=attention_kernel_batch_size,
        gather_no_pad=csa_gather_no_pad,
    )
    return output, updated_sw_cache


def get_packed_mla_head_size(hf_config) -> int:
    """Calculate TPU 128-byte aligned packed width for DSv4 FP8 cache entries."""
    kv_bytes = getattr(hf_config, "kv_lora_rank", 448)
    rope_bytes = getattr(hf_config, "qk_rope_head_dim", 64) * 2
    scale_bytes = kv_bytes // 64
    return align_to(kv_bytes + rope_bytes + scale_bytes, 128)


class VllmDeepseekSparseSWABackend(DeepseekSparseSWABackend):
    """TPU attention backend for DeepSeek-V4 SWA cache."""

    @classmethod
    def get_min_page_size(cls, vllm_config: VllmConfig) -> int:
        return 1


class VllmDeepseekV4SWACache(DeepseekV4SWACache):
    """Sliding-window KV cache, paged to fit inside a CSA main page."""

    def __init__(
        self,
        head_dim: int,
        window_size: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config,
        backend_cls: type[AttentionBackend] | None = None,
    ) -> None:
        super().__init__(
            head_dim,
            window_size,
            dtype,
            prefix,
            cache_config,
            backend_cls=backend_cls or VllmDeepseekSparseSWABackend,
        )
        # Initialize with default block size at construction time;
        # get_kv_cache_spec recomputes self.block_size from finalized runtime config.
        self.block_size = self._swa_block_size(cache_config.block_size, window_size)

    @staticmethod
    def _swa_block_size(compressed_kv_cache_bz: int, window_size: int | None) -> int:
        # We would like to overlay the SWA cache with CSA's main NOPE cache
        # on the same KV-Tensor, whose shape is [num_pages, page_size, 4, 128]
        # u8.
        # Thus set swa cache's block size accordingly.
        # In SWA, we store kv cache in bf16 to avoid expensive quantization
        # and dequantization. The extra storage overhead is small since kvs
        # outside the sliding window can be reclaimed as needed.
        # 2 because bf16 occupies 2 bytes per element.
        csa_compression_ratio = 4
        block_size = compressed_kv_cache_bz // csa_compression_ratio // 2
        if window_size is None:
            return block_size
        return min(block_size, window_size)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # Update self.block_size so the returned spec and the Pallas kernel's
        # logical_page_size in attention.py agree on the final runtime block size.
        self.block_size = self._swa_block_size(
            vllm_config.cache_config.block_size, self.window_size
        )
        # `mla_swa` keeps the SWA entries as raw bf16, packed as uint8.
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=1024,
            dtype=torch.uint8,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            alignment=None,
        )
