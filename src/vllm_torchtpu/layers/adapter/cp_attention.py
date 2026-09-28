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
"""DCP (Decode Context Parallelism) attention: kernel building and forward
orchestration, on top of batched_rpa's schedule_cp.CPMetadataComputer.
Process-group access lives in distributed/dcp.py.

Each DCP rank stores a shard of the KV cache (interleaved by page_size).
DCP ranks own *different* Q heads but share one GQA-replicated KV head.

Forward splits into two kernel passes, merged with LSE-weighted combination:
  1. CACHE_ONLY — query is AllGathered by heads so every rank's own heads
     can attend against every partner's KV shard, then partial
     outputs/LSEs are AllGathered and folded, then sliced back down to
     this rank's own heads.
  2. NEW_TOKENS_ONLY — new tokens aren't DCP-sharded (every rank has the
     full local K/V for this step), so this rank's own heads attend
     directly; no gather/combine needed.
"""

from __future__ import annotations

import functools

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.distributed.dcp import get_dcp_group
from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs as _batched_rpa_configs,
)
from vllm_torchtpu.kernels.experimental.batched_rpa import (
    wrapper as _batched_rpa_wrapper,
)
from vllm_torchtpu.utils import synchronize_tensors

# ---------------------------------------------------------------------------
# LSE-weighted combination
# ---------------------------------------------------------------------------


def lse_weighted_combine(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Numerically stable LSE-weighted combination of two partial attentions.

    Args:
        out_a / out_b: [T, H, D]
        lse_a / lse_b: [T, H]  (log-sum-exp in natural-log base)
    Returns:
        (combined_out [T, H, D], combined_lse [T, H])
    """
    m = torch.maximum(lse_a, lse_b)
    m_finite = torch.isfinite(m)
    m_safe = torch.where(m_finite, m, torch.zeros_like(m))
    ea = torch.exp(lse_a - m_safe)
    eb = torch.exp(lse_b - m_safe)
    norm = ea + eb
    out_a = torch.nan_to_num(out_a, nan=0.0, posinf=0.0, neginf=0.0)
    out_b = torch.nan_to_num(out_b, nan=0.0, posinf=0.0, neginf=0.0)
    norm_safe = torch.where(norm > 0, norm, torch.ones_like(norm))
    out = (ea.unsqueeze(-1) * out_a + eb.unsqueeze(-1) * out_b) / norm_safe.unsqueeze(
        -1
    )
    lse_out = torch.where(m_finite, m + torch.log(norm_safe), m)
    return out, lse_out


def dcp_allgather_lse_combine(
    output: torch.Tensor,
    lse: torch.Tensor,
    dcp_group,
    dcp_world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """AllGather partial outputs and LSEs across DCP ranks, then fold them.

    Args:
        output: [T, H, D] — this rank's partial attention output.
        lse:    [T, H]    — corresponding log-sum-exp values.
        dcp_group: vLLM GroupCoordinator for the DCP process group.
        dcp_world_size: number of DCP ranks.
    Returns:
        (combined_out [T, H, D], combined_lse [T, H]) after folding all ranks.
    """
    # [T, H, D] -> [dcp_world_size * T, H, D]
    all_outputs = dcp_group.all_gather(output, dim=0)
    # [T, H]   -> [dcp_world_size * T, H]
    all_lses = dcp_group.all_gather(lse, dim=0)
    out_chunks = all_outputs.chunk(dcp_world_size, dim=0)
    lse_chunks = all_lses.chunk(dcp_world_size, dim=0)
    out_combined, lse_combined = out_chunks[0], lse_chunks[0]
    for i in range(1, dcp_world_size):
        out_combined, lse_combined = lse_weighted_combine(
            out_combined, lse_combined, out_chunks[i], lse_chunks[i]
        )
    return out_combined, lse_combined


# ---------------------------------------------------------------------------
# DCP kernel entry functions (for pallas.jax_op)
# ---------------------------------------------------------------------------


def _pallas_rpa_kernel_dcp(
    kv_cache: jax.Array,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    *,
    sliding_window: int | None,
    sm_scale: float | None = None,
    soft_cap: float | None = None,
    cp_group_size: int,
    cp_rank_val: int,
    attention_scope: _batched_rpa_configs.AttentionScope,
    kv_layout: _batched_rpa_configs.KVLayout = (
        _batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE
    ),
    decode_query_size: int = 1,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One DCP pass (CACHE_ONLY or NEW_TOKENS_ONLY). Returns (new_kv_cache,
    output, lse). decode_query_size widens the DECODE bucket so speculative
    verify windows (K + 1 tokens) run there instead of in MIXED.
    cp_group_size/cp_rank select schedule_cp.CPMetadataComputer
    inside ragged_paged_attention, needed by both scopes: CACHE_ONLY to
    compute this rank's local cache length, NEW_TOKENS_ONLY to land the
    write on the right physical page of this rank's DCP-interleaved
    kv_cache (its own write offset is otherwise global, not rank-local).
    """
    import jax.numpy as jnp

    cp_rank_arr = jnp.array([cp_rank_val], dtype=jnp.int32)
    output, new_kv_cache, lse = _batched_rpa_wrapper.ragged_paged_attention(
        queries=query,
        keys=key,
        values=value,
        kv_cache=kv_cache,
        kv_lens=seq_lens,
        page_indices=block_tables,
        cu_q_lens=query_start_loc,
        distribution=request_distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        cp_group_size=cp_group_size,
        cp_rank=cp_rank_arr,
        attention_scope=attention_scope,
        decode_query_size=decode_query_size,
        return_lse=True,
        kv_layout=kv_layout,
    )
    return new_kv_cache, output, lse


# ---------------------------------------------------------------------------
# Kernel op builder
# ---------------------------------------------------------------------------

# Class-level registry shared across all PallasAttentionBackendImpl instances.
# Keyed by config tuple; values are the (cache_kernel_impl, new_kernel_impl)
# pair so layers with identical configs share one set of compiled ops.
_DCP_KERNEL_REGISTRY: dict = {}
_DCP_KERNEL_INSTANCE_COUNTER = 0


def _alloc_instance_id() -> int:
    global _DCP_KERNEL_INSTANCE_COUNTER
    _id = _DCP_KERNEL_INSTANCE_COUNTER
    _DCP_KERNEL_INSTANCE_COUNTER += 1
    return _id


def _fake_dcp(kv_cache: torch.Tensor, query: torch.Tensor, *args, **kwargs):
    lse = torch.empty(
        query.shape[0], query.shape[1], dtype=query.dtype, device=query.device
    )
    return torch.empty_like(kv_cache), torch.empty_like(query), lse


def _build_dcp_kernel_op(op_name: str, fn):
    """Wrap fn (a _pallas_rpa_kernel_dcp partial) as a callable kernel.

    kv_cache (arg 0) is donated: CACHE_ONLY never mutates it, so its
    returned new_kv_cache is identical to the input; NEW_TOKENS_ONLY writes
    the new tokens into it. Either way donation lets XLA alias the output
    onto the input pool instead of materializing a second, full-sized copy.
    The kv_cache.copy_ below keeps the caller's tensor validly bound to
    that (possibly aliased) result and compiles to a no-op for CACHE_ONLY.
    """
    op = pallas.jax_op(
        op_name, fn, donate_argnums=(0,), mesh=None, input_partition_specs=None
    )
    op.register_fake(_fake_dcp)

    def kernel_impl(kv_cache, *args, **kwargs):
        new_kv_cache, output, lse = op(kv_cache, *args, **kwargs)
        kv_cache.copy_(new_kv_cache)
        return output, lse

    return kernel_impl


def build_dcp_kernels(
    sliding_window: int | None,
    sm_scale: float,
    logits_soft_cap: float | None,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    cp_group_size: int,
    cp_rank: int,
    kv_layout: _batched_rpa_configs.KVLayout = (
        _batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE
    ),
    decode_query_size: int = 1,
) -> tuple:
    """Build and cache the CACHE_ONLY and NEW_TOKENS_ONLY jax_ops.

    Returns:
        (cache_kernel, new_kernel) where each is callable as
        ``kernel(kv_cache, query, key, value, seq_lens, block_tables,
                 query_start_loc, request_distribution, q_scale, k_scale,
                 v_scale) -> (output [T,H,D], lse [T,H])``
    """
    # The adapter uses batched_rpa's equivalent enum. Normalize here so both
    # entry points bind LONGCTX's enum and share ops for the same layout.
    kv_layout = _batched_rpa_configs.KVLayout(kv_layout)
    base_key = (
        sliding_window,
        sm_scale,
        logits_soft_cap,
        q_scale,
        k_scale,
        v_scale,
        cp_group_size,
        cp_rank,
        kv_layout,
        decode_query_size,
    )
    cache_key = ("dcp_cache", *base_key)
    new_key = ("dcp_new", *base_key)

    cached_cache = _DCP_KERNEL_REGISTRY.get(cache_key)
    cached_new = _DCP_KERNEL_REGISTRY.get(new_key)
    if cached_cache is not None and cached_new is not None:
        return cached_cache, cached_new

    if cached_cache is None:
        fn = functools.partial(
            _pallas_rpa_kernel_dcp,
            sliding_window=sliding_window,
            sm_scale=sm_scale,
            soft_cap=logits_soft_cap,
            cp_group_size=cp_group_size,
            cp_rank_val=cp_rank,
            attention_scope=_batched_rpa_configs.AttentionScope.CACHE_ONLY,
            kv_layout=kv_layout,
            decode_query_size=decode_query_size,
        )
        cached_cache = _DCP_KERNEL_REGISTRY[cache_key] = _build_dcp_kernel_op(
            f"pallas::rpa_dcp_cache_{_alloc_instance_id()}", fn
        )

    if cached_new is None:
        fn = functools.partial(
            _pallas_rpa_kernel_dcp,
            sliding_window=sliding_window,
            sm_scale=sm_scale,
            soft_cap=logits_soft_cap,
            cp_group_size=cp_group_size,
            cp_rank_val=cp_rank,
            attention_scope=_batched_rpa_configs.AttentionScope.NEW_TOKENS_ONLY,
            kv_layout=kv_layout,
            decode_query_size=decode_query_size,
        )
        cached_new = _DCP_KERNEL_REGISTRY[new_key] = _build_dcp_kernel_op(
            f"pallas::rpa_dcp_new_{_alloc_instance_id()}", fn
        )

    return cached_cache, cached_new


# ---------------------------------------------------------------------------
# Top-level DCP forward
# ---------------------------------------------------------------------------


def forward_with_dcp(
    *,
    layer,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
    sliding_window: int | None,
    sm_scale: float,
    logits_soft_cap: float | None,
    kv_cache_quantized_dtype,
    dcp_world_size: int,
    dcp_rank: int,
    kv_layout: _batched_rpa_configs.KVLayout = (
        _batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE
    ),
    decode_query_size: int = 1,
) -> torch.Tensor:
    """Orchestrate the two-pass DCP attention forward.

    CACHE_ONLY pass: AllGather Q (by heads) -> attend against this rank's KV
        shard -> AllGather + LSE-combine partial outputs -> select this
        rank's own head slice back out.
    NEW_TOKENS_ONLY pass: this rank's own heads only (not DCP-sharded).
    Final: LSE-combine the context result with the new-token result.
    """
    dcp_group = get_dcp_group()
    if dcp_group is None:
        raise RuntimeError("DCP group not initialized; cannot run DCP attention.")

    q_scale = None
    k_scale = layer._k_scale_float if kv_cache_quantized_dtype else None
    v_scale = layer._v_scale_float if kv_cache_quantized_dtype else None

    cache_kernel, new_kernel = build_dcp_kernels(
        sliding_window=sliding_window,
        sm_scale=sm_scale,
        logits_soft_cap=logits_soft_cap,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        cp_group_size=dcp_world_size,
        cp_rank=dcp_rank,
        kv_layout=kv_layout,
        decode_query_size=decode_query_size,
    )

    own_num_heads = query.shape[1]

    # This rank's own Q heads must be attended against every DCP partner's
    # KV shard, not just this rank's own -- so gather every partner's Q
    # heads onto every rank (cheap: one token) instead of moving the KV
    # cache around (which DCP shards specifically to avoid).
    query_across_dcp = dcp_group.all_gather(query.contiguous(), dim=1)

    shared_args = (
        key,
        value,
        attn_metadata.seq_lens,
        attn_metadata.block_tables,
        attn_metadata.query_start_loc,
        attn_metadata.request_distribution,
        q_scale,
        k_scale,
        v_scale,
    )

    ctx_output, ctx_lse = cache_kernel(kv_cache, query_across_dcp, *shared_args)

    combined_output, combined_lse = dcp_allgather_lse_combine(
        ctx_output, ctx_lse, dcp_group, dcp_world_size
    )

    # all_gather above concatenates in rank order, so this rank's own
    # heads sit at [rank * own_num_heads : (rank + 1) * own_num_heads).
    head_start = dcp_rank * own_num_heads
    head_end = head_start + own_num_heads
    combined_output = combined_output[:, head_start:head_end, :]
    combined_lse = combined_lse[:, head_start:head_end]

    new_output, new_lse = new_kernel(kv_cache, query, *shared_args)

    output, _ = lse_weighted_combine(combined_output, combined_lse, new_output, new_lse)

    if not torch.compiler.is_compiling():
        synchronize_tensors(kv_cache, wait=False)
    return output
