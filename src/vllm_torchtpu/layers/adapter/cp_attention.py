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
orchestration, on top of batched_rpa's scheduler_cp.StripedFullScheduler.
Process-group access lives in distributed/dcp.py.

Each DCP rank stores a shard of the KV cache (interleaved by page_size).
DCP ranks own *different* Q heads but share one GQA-replicated KV head.
"""

from __future__ import annotations

import functools

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.distributed.dcp import get_dcp_group
from vllm_torchtpu.distributed.dcp_comm import cp_lse_ag_out_rs
from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs as _batched_rpa_configs,
)
from vllm_torchtpu.kernels.experimental.batched_rpa import (
    scheduler_cp as _batched_rpa_scheduler_cp,
)
from vllm_torchtpu.kernels.experimental.batched_rpa import (
    wrapper as _batched_rpa_wrapper,
)
from vllm_torchtpu.utils import synchronize_tensors

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
    kv_layout: _batched_rpa_configs.KVLayout = (
        _batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE
    ),
    decode_query_size: int = 1,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The single DCP pass. Returns (new_kv_cache, output, lse).
    decode_query_size widens the DECODE bucket so speculative verify windows
    (K + 1 tokens) run there instead of in MIXED.
    StripedFullScheduler attends this rank's page-striped share of both the
    cached and the new tokens, and writes each new token only on the rank
    that owns its page of the DCP-interleaved kv_cache.
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
        scheduler=_batched_rpa_scheduler_cp.StripedFullScheduler(
            cp_group_size=cp_group_size
        ),
        extra_scalars=(cp_rank_arr,),
        decode_query_size=decode_query_size,
        return_lse=True,
        kv_layout=kv_layout,
    )
    return new_kv_cache, output, lse


# ---------------------------------------------------------------------------
# Kernel op builder
# ---------------------------------------------------------------------------

# Class-level registry shared across all PallasAttentionBackendImpl instances.
# Keyed by config tuple so layers with identical configs share one compiled op.
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

    kv_cache (arg 0) is donated: the kernel writes the new tokens into it,
    and donation lets XLA alias the output onto the input pool instead of
    materializing a second, full-sized copy. The kv_cache.copy_ below keeps
    the caller's tensor validly bound to that (possibly aliased) result.
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


def build_dcp_kernel(
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
):
    """Build and cache the DCP jax_op.

    Returns:
        kernel callable as
        ``kernel(kv_cache, query, key, value, seq_lens, block_tables,
                 query_start_loc, request_distribution, q_scale, k_scale,
                 v_scale) -> (output [T,H,D], lse [T,H])``
    """
    # The adapter uses batched_rpa's equivalent enum. Normalize here so both
    # entry points bind LONGCTX's enum and share ops for the same layout.
    kv_layout = _batched_rpa_configs.KVLayout(kv_layout)
    key = (
        "dcp",
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

    cached = _DCP_KERNEL_REGISTRY.get(key)
    if cached is not None:
        return cached

    fn = functools.partial(
        _pallas_rpa_kernel_dcp,
        sliding_window=sliding_window,
        sm_scale=sm_scale,
        soft_cap=logits_soft_cap,
        cp_group_size=cp_group_size,
        cp_rank_val=cp_rank,
        kv_layout=kv_layout,
        decode_query_size=decode_query_size,
    )
    cached = _DCP_KERNEL_REGISTRY[key] = _build_dcp_kernel_op(
        f"pallas::rpa_dcp_{_alloc_instance_id()}", fn
    )
    return cached


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
    """Single-pass DCP attention forward.

    The query is all-gathered by heads, and each rank attends every head
    against its page-striped share of the cached and new KV in one kernel.
    `cp_lse_ag_out_rs` then merges the partials back to this rank's heads. A
    rank that holds none of a sequence's tokens returns lse = -inf for it,
    which the merge weighs to zero.
    """
    dcp_group = get_dcp_group()
    if dcp_group is None:
        raise RuntimeError("DCP group not initialized; cannot run DCP attention.")

    q_scale = None
    k_scale = layer._k_scale_float if kv_cache_quantized_dtype else None
    v_scale = layer._v_scale_float if kv_cache_quantized_dtype else None

    kernel = build_dcp_kernel(
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

    partial_output, partial_lse = kernel(kv_cache, query_across_dcp, *shared_args)

    output = cp_lse_ag_out_rs(partial_output, partial_lse, dcp_group)

    if not torch.compiler.is_compiling():
        synchronize_tensors(kv_cache, wait=False)
    return output
