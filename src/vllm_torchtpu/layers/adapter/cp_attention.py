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
from vllm_torchtpu.distributed.pcp import (
    all_gather_equal_tokens,
    get_or_create_pcp_mesh,
    get_pcp_group,
    get_pcp_rank,
    get_pcp_world_size,
)
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
    kv_layout: _batched_rpa_configs.KVLayout = _batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One DCP pass (CACHE_ONLY or NEW_TOKENS_ONLY). Returns (new_kv_cache,
    output, lse). cp_group_size/cp_rank select schedule_cp.CPMetadataComputer
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
    kv_layout: _batched_rpa_configs.KVLayout = _batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE,
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
    kv_layout: _batched_rpa_configs.KVLayout = _batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE,
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


# ---------------------------------------------------------------------------
# PCP (Prefill Context Parallelism) kernel entry functions & forward
# ---------------------------------------------------------------------------


def _pallas_rpa_kernel_pcp_cache(
    kv_cache: jax.Array,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    kv_new_lens: jax.Array,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    *,
    sliding_window: int | None,
    sm_scale: float | None = None,
    soft_cap: float | None = None,
    cp_group_size: int,
    cp_rank_val: int,
    pcp_ring_axis_name: str | None = "pcp",
    pcp_ring_mesh_axis_names: tuple[str, ...] | None = None,
    kv_layout: _batched_rpa_configs.KVLayout = _batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    import jax.numpy as jnp

    cp_rank_arr = jnp.array([cp_rank_val], dtype=jnp.int32)
    output, new_kv_cache, lse = _batched_rpa_wrapper.ragged_paged_attention(
        queries=query,
        keys=key,
        values=value,
        kv_cache=kv_cache,
        kv_lens=seq_lens,
        page_indices=block_tables,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        cp_group_size=cp_group_size,
        cp_rank=cp_rank_arr,
        kv_new_lens=kv_new_lens,
        attention_scope=_batched_rpa_configs.AttentionScope.CACHE_ONLY,
        pcp_ring_axis_name=pcp_ring_axis_name,
        pcp_ring_mesh_axis_names=pcp_ring_mesh_axis_names,
        update_kv_cache=False,
        return_lse=True,
        kv_layout=kv_layout,
    )
    return new_kv_cache, output, lse


def _pallas_rpa_kernel_pcp_new(
    kv_cache: jax.Array,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    kv_new_lens: jax.Array,
    q_positions: jax.Array,
    update_kv_cache: jax.Array,
    new_kv_page_indices: jax.Array,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    *,
    sliding_window: int | None,
    sm_scale: float | None = None,
    soft_cap: float | None = None,
    cp_group_size: int,
    cp_rank_val: int,
    use_causal_mask: bool = True,
    kv_layout: _batched_rpa_configs.KVLayout = _batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    import jax.numpy as jnp

    cp_rank_arr = jnp.array([cp_rank_val], dtype=jnp.int32)
    new_kv_pages_arg = None if new_kv_page_indices.size == 0 else new_kv_page_indices
    output, new_kv_cache, lse = _batched_rpa_wrapper.ragged_paged_attention(
        queries=query,
        keys=key,
        values=value,
        kv_cache=kv_cache,
        kv_lens=seq_lens,
        page_indices=block_tables,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        cp_group_size=cp_group_size,
        cp_rank=cp_rank_arr,
        kv_new_lens=kv_new_lens,
        q_positions=q_positions,
        update_kv_cache=update_kv_cache,
        new_kv_page_indices=new_kv_pages_arg,
        attention_scope=_batched_rpa_configs.AttentionScope.NEW_TOKENS_ONLY,
        use_causal_mask=use_causal_mask,
        return_lse=True,
        kv_layout=kv_layout,
    )
    return new_kv_cache, output, lse


_PCP_KERNEL_REGISTRY: dict = {}
_PCP_KERNEL_INSTANCE_COUNTER = 0


def _alloc_pcp_instance_id() -> int:
    global _PCP_KERNEL_INSTANCE_COUNTER
    _id = _PCP_KERNEL_INSTANCE_COUNTER
    _PCP_KERNEL_INSTANCE_COUNTER += 1
    return _id


def _build_pcp_kernel_op(op_name: str, fn, mesh: Any = None):
    op = pallas.jax_op(
        op_name, fn, donate_argnums=(0,), mesh=mesh, input_partition_specs=None
    )
    op.register_fake(_fake_dcp)

    def kernel_impl(kv_cache, *args, **kwargs):
        new_kv_cache, output, lse = op(kv_cache, *args, **kwargs)
        kv_cache.copy_(new_kv_cache)
        return output, lse

    return kernel_impl


def build_pcp_kernels(
    sliding_window: int | None,
    sm_scale: float,
    logits_soft_cap: float | None,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    cp_group_size: int,
    cp_rank: int,
    use_causal_mask: bool = True,
    kv_layout: _batched_rpa_configs.KVLayout = _batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE,
) -> tuple:
    """Build and cache the CACHE_ONLY and NEW_TOKENS_ONLY jax_ops for PCP."""
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
        use_causal_mask,
        kv_layout,
    )
    cache_key = ("pcp_cache", *base_key)
    new_key = ("pcp_new", *base_key)

    cached_cache = _PCP_KERNEL_REGISTRY.get(cache_key)
    cached_new = _PCP_KERNEL_REGISTRY.get(new_key)
    if cached_cache is not None and cached_new is not None:
        return cached_cache, cached_new

    try:
        mesh = get_or_create_pcp_mesh(axis_name="pcp")
    except Exception:
        mesh = None

    if cached_cache is None:
        fn = functools.partial(
            _pallas_rpa_kernel_pcp_cache,
            sliding_window=sliding_window,
            sm_scale=sm_scale,
            soft_cap=logits_soft_cap,
            cp_group_size=cp_group_size,
            cp_rank_val=cp_rank,
            pcp_ring_axis_name="pcp",
            pcp_ring_mesh_axis_names=None,
            kv_layout=kv_layout,
        )
        cached_cache = _PCP_KERNEL_REGISTRY[cache_key] = _build_pcp_kernel_op(
            f"pallas::rpa_pcp_cache_{_alloc_pcp_instance_id()}", fn, mesh=mesh
        )

    if cached_new is None:
        fn = functools.partial(
            _pallas_rpa_kernel_pcp_new,
            sliding_window=sliding_window,
            sm_scale=sm_scale,
            soft_cap=logits_soft_cap,
            cp_group_size=cp_group_size,
            cp_rank_val=cp_rank,
            use_causal_mask=use_causal_mask,
            kv_layout=kv_layout,
        )
        cached_new = _PCP_KERNEL_REGISTRY[new_key] = _build_pcp_kernel_op(
            f"pallas::rpa_pcp_new_{_alloc_pcp_instance_id()}", fn, mesh=mesh
        )

    return cached_cache, cached_new


def _make_cu_cache(cu_q_lens: torch.Tensor, num_seqs: int) -> torch.Tensor:
    cu = cu_q_lens[::2]
    if cu.shape[0] < num_seqs + 1:
        last_val = cu[-1]
        pad_len = num_seqs + 1 - cu.shape[0]
        padding = torch.full((pad_len,), last_val.item(), dtype=cu.dtype, device=cu.device)
        cu = torch.cat([cu, padding], dim=0)
    return cu


def _per_request_1d(x: torch.Tensor, num_seqs: int) -> torch.Tensor:
    heads = x[::2]
    if heads.shape[0] < num_seqs:
        pad_len = num_seqs - heads.shape[0]
        zeros = torch.zeros((pad_len,), dtype=heads.dtype, device=heads.device)
        heads = torch.cat([heads, zeros], dim=0)
    return heads


def _per_request_pages(block_tables: torch.Tensor, num_seqs: int) -> torch.Tensor:
    orig_shape = block_tables.shape
    bt_2d = block_tables.reshape(num_seqs, -1)
    heads = bt_2d[::2]
    if heads.shape[0] < num_seqs:
        pad_rows = num_seqs - heads.shape[0]
        zeros = torch.zeros((pad_rows, heads.shape[1]), dtype=heads.dtype, device=heads.device)
        heads = torch.cat([heads, zeros], dim=0)
    return heads.reshape(orig_shape)


def forward_with_pcp(
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
    pcp_world_size: int,
    pcp_rank: int,
    kv_layout: _batched_rpa_configs.KVLayout = _batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE,
    use_causal_mask: bool = True,
) -> torch.Tensor:
    """Orchestrate the two-pass PCP attention forward.

    CACHE_ONLY pass: in-kernel ring attention over the page-striped KV cache
        rotating around the PCP ring axis.
    NEW_TOKENS_ONLY pass: all-gathered current KV (via all_gather across PCP)
        attended with local Q, writing back to striped KV cache.
    Final: LSE-combine the context result with the new-token result.
    """
    pcp_meta = attn_metadata.pcp
    if pcp_meta is None:
        raise ValueError("forward_with_pcp requires attn_metadata.pcp to be set.")

    q_scale = None
    k_scale = layer._k_scale_float if kv_cache_quantized_dtype else None
    v_scale = layer._v_scale_float if kv_cache_quantized_dtype else None

    cache_kernel, new_kernel = build_pcp_kernels(
        sliding_window=sliding_window,
        sm_scale=sm_scale,
        logits_soft_cap=logits_soft_cap,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        cp_group_size=pcp_world_size,
        cp_rank=pcp_rank,
        use_causal_mask=use_causal_mask,
        kv_layout=kv_layout,
    )

    num_seqs = attn_metadata.seq_lens.shape[0]

    def _to_torch(x, dtype=torch.int32):
        if x is None:
            return torch.empty((0,), dtype=dtype, device=query.device)
        if isinstance(x, torch.Tensor):
            return x.to(device=query.device, dtype=dtype)
        import numpy as np

        return torch.from_numpy(np.asarray(x)).to(device=query.device, dtype=dtype)

    pcp_cu = _to_torch(pcp_meta.query_start_loc)
    pcp_cache_lens = _to_torch(pcp_meta.kv_cache_lens)
    pcp_qpos = _to_torch(pcp_meta.q_pos_offsets)
    pcp_update = _to_torch(pcp_meta.update_kv_cache, dtype=torch.int32)
    pcp_new_pages = _to_torch(pcp_meta.new_kv_page_indices)

    if pcp_cu.ndim == 2:
        cu_q_lens_local = pcp_cu[pcp_rank]
    else:
        cu_q_lens_local = pcp_cu

    if pcp_qpos.ndim == 2:
        q_pos_offsets_local = pcp_qpos[pcp_rank]
    else:
        q_pos_offsets_local = pcp_qpos

    if pcp_update.ndim == 2:
        update_kv_cache_local = pcp_update[pcp_rank]
    else:
        update_kv_cache_local = pcp_update

    kv_new_lens = attn_metadata.seq_lens - pcp_cache_lens

    # ---- 1. CACHE_ONLY pass ----
    cache_pages = pcp_meta.cache_pages
    if cache_pages == 0:
        ctx_output = ctx_lse = None
    else:
        cu_cache = _make_cu_cache(cu_q_lens_local, num_seqs)
        dist_cache = attn_metadata.request_distribution.clone()
        dist_cache[0] = 0
        dist_cache[1] = 0
        dist_cache[2] = attn_metadata.request_distribution[2] // 2

        seq_lens_req = _per_request_1d(attn_metadata.seq_lens, num_seqs)
        block_tables_req = _per_request_pages(attn_metadata.block_tables, num_seqs)
        kv_new_lens_req = _per_request_1d(kv_new_lens, num_seqs)

        ctx_output, ctx_lse = cache_kernel(
            kv_cache,
            query,
            key,
            value,
            seq_lens_req,
            block_tables_req,
            cu_cache,
            dist_cache,
            kv_new_lens_req,
            q_scale,
            k_scale,
            v_scale,
        )

    # ---- 2. NEW_TOKENS_ONLY pass ----
    k_curr = all_gather_equal_tokens(key, dim=0)
    v_curr = all_gather_equal_tokens(value, dim=0)

    q_positions = pcp_cache_lens + q_pos_offsets_local

    curr_output, curr_lse = new_kernel(
        kv_cache,
        query,
        k_curr,
        v_curr,
        attn_metadata.seq_lens,
        attn_metadata.block_tables,
        cu_q_lens_local,
        attn_metadata.request_distribution,
        kv_new_lens,
        q_positions,
        update_kv_cache_local,
        pcp_new_pages,
        q_scale,
        k_scale,
        v_scale,
    )

    # ---- 3. Combine ----
    if ctx_output is None:
        output = curr_output
    else:
        output, _ = lse_weighted_combine(ctx_output, ctx_lse, curr_output, curr_lse)

    if not torch.compiler.is_compiling():
        synchronize_tensors(kv_cache, wait=False)
    return output


def merge_attn_states(
    cache_out: jax.Array,
    cache_lse: jax.Array,
    query_out: jax.Array,
    query_lse: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """LSE-weighted merge of two disjoint attention spans in JAX."""
    import jax.numpy as jnp

    max_lse = jnp.maximum(cache_lse, query_lse)
    max_lse_safe = jnp.where(jnp.isinf(max_lse), 0.0, max_lse)
    exp_cache = jnp.exp(cache_lse - max_lse_safe)
    exp_query = jnp.exp(query_lse - max_lse_safe)
    sum_exp = exp_cache + exp_query
    denom = jnp.where(sum_exp == 0.0, 1.0, sum_exp)
    merged_out = (
        cache_out * exp_cache[..., None] + query_out * exp_query[..., None]
    ) / denom[..., None]
    merged_lse = jnp.where(
        sum_exp == 0.0, -jnp.inf, max_lse_safe + jnp.log(denom)
    )
    return merged_out, merged_lse


def pcp_forward_batched(
    mesh: Any,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    md: Any,
    sm_scale: float,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    use_causal_mask: bool = True,
    kv_layout: _batched_rpa_configs.KVLayout = (
        _batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE
    ),
) -> tuple[jax.Array, jax.Array]:
    """PCP attention forward on the batched RPA kernel using JAX shard_map."""
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import PartitionSpec as P

    pcp_axis = "pcp"
    axis_sizes = dict(zip(mesh.axis_names, mesh.devices.shape))
    pcp_size = axis_sizes.get(pcp_axis, 1)
    num_seqs = md.seq_lens.shape[0]
    if md.pcp.new_kv_page_indices is None or md.pcp.update_kv_cache is None:
        raise ValueError(
            "pcp_forward_batched needs pcp.new_kv_page_indices and "
            "pcp.update_kv_cache."
        )

    q_spec = P(None, None, None)
    kv_spec = P(None, None, None)
    if kv_layout == _batched_rpa_configs.KVLayout.SEQ_ALONG_LANE:
        kv_cache_spec = P(None, None, None, None, pcp_axis)
    else:
        kv_cache_spec = P(None, pcp_axis, None, None, None)

    common = dict(
        sm_scale=sm_scale,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        cp_group_size=pcp_size,
        kv_layout=kv_layout,
        return_lse=True,
    )

    cache_pages = md.pcp.cache_pages

    def _shard_fn(
        q_local,
        k_local,
        v_local,
        kv_cache_local,
        kv_lens_local,
        kv_cache_lens_local,
        page_indices_local,
        distribution_local,
        new_kv_page_indices,
        pcp_cu_q_lens_local,
        pcp_q_pos_offsets_local,
        pcp_update_kv_cache_local,
    ):
        cp_rank = jnp.reshape(lax.axis_index(pcp_axis), (1,)).astype(jnp.int32)
        kv_new_lens = kv_lens_local - kv_cache_lens_local

        def all_gather_tokens(x):
            return lax.all_gather(x, pcp_axis, axis=0, tiled=True)

        def _per_request(x):
            heads = x[::2]
            pad = [(0, num_seqs - heads.shape[0])] + [(0, 0)] * (x.ndim - 1)
            return jnp.pad(heads, pad)

        cu_cache = pcp_cu_q_lens_local[0][::2]
        cu_cache = jnp.pad(
            cu_cache, (0, num_seqs + 1 - cu_cache.shape[0]), mode="edge"
        )
        dist_cache = jnp.array([0, 0, 0], jnp.int32).at[2].set(
            distribution_local[2] // 2
        )

        if cache_pages == 0:
            context_out = context_lse = None
        else:
            context_out, _, context_lse = _batched_rpa_wrapper.ragged_paged_attention(
                q_local,
                k_local,
                v_local,
                kv_cache_local,
                _per_request(kv_lens_local),
                _per_request(page_indices_local.reshape(num_seqs, -1)).reshape(
                    -1
                ),
                cu_cache,
                dist_cache,
                cp_rank=cp_rank,
                kv_new_lens=_per_request(kv_new_lens),
                attention_scope=_batched_rpa_configs.AttentionScope.CACHE_ONLY,
                pcp_ring_axis_name=pcp_axis,
                pcp_ring_mesh_axis_names=tuple(mesh.axis_names),
                update_kv_cache=False,
                **common,
            )
            context_out = context_out.astype(jnp.float32)
            context_lse = context_lse.astype(jnp.float32)

        q_positions = kv_cache_lens_local + pcp_q_pos_offsets_local[0]
        k_curr = all_gather_tokens(k_local)
        v_curr = all_gather_tokens(v_local)
        curr_out, kv_cache_updated, curr_lse = (
            _batched_rpa_wrapper.ragged_paged_attention(
                q_local,
                k_curr,
                v_curr,
                kv_cache_local,
                kv_lens_local,
                page_indices_local,
                pcp_cu_q_lens_local[0],
                distribution_local,
                cp_rank=cp_rank,
                kv_new_lens=kv_new_lens,
                q_positions=q_positions,
                update_kv_cache=pcp_update_kv_cache_local[0],
                new_kv_page_indices=new_kv_page_indices,
                attention_scope=_batched_rpa_configs.AttentionScope.NEW_TOKENS_ONLY,
                use_causal_mask=use_causal_mask,
                **common,
            )
        )

        if context_out is None:
            out = curr_out
        else:
            out, _ = merge_attn_states(
                context_out,
                context_lse,
                curr_out.astype(jnp.float32),
                curr_lse.astype(jnp.float32),
            )
        return kv_cache_updated, out.astype(q.dtype)

    return jax.shard_map(
        _shard_fn,
        mesh=mesh,
        in_specs=(
            q_spec,
            kv_spec,
            kv_spec,
            kv_cache_spec,
            P(),  # kv_lens: replicated
            P(),  # pcp.kv_cache_lens: replicated
            P(),  # page_indices: replicated
            P(),  # distribution: replicated
            P(),  # pcp.new_kv_page_indices: replicated
            P(pcp_axis, None),  # pcp.query_start_loc: per-rank cu_q_lens
            P(pcp_axis, None),  # pcp.q_pos_offsets: per-rank position offsets
            P(pcp_axis, None),  # pcp.update_kv_cache: per-rank writer flags
        ),
        out_specs=(kv_cache_spec, q_spec),
        check_vma=False,
    )(
        q,
        k,
        v,
        kv_cache,
        md.seq_lens,
        md.pcp.kv_cache_lens,
        md.block_tables,
        md.request_distribution,
        md.pcp.new_kv_page_indices,
        md.pcp.query_start_loc,
        md.pcp.q_pos_offsets,
        md.pcp.update_kv_cache,
    )

