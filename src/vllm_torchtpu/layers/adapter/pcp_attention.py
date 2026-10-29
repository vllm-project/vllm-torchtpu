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
"""PCP attention under the PCP manager (`USE_TPU_PCP_MANAGER=1`): kernel
building and forward orchestration on top of `batched_rpa_new`, the way
`cp_attention.py` does DCP.

Q/K/V are this rank's tokens in the sharding plan's layout; the attention
metadata is global request-major. One logical KV block is one page on each
PCP rank (`owner = page % pcp`). Two passes, merged by LSE:

1. Cache: this rank's queries over the cached KV, rotated over the PCP ranks
   (`RingCacheScheduler`). It only reads the KV cache.
2. New tokens: this rank's queries over the step's all-gathered new KV
   (`StripedNewTokensScheduler`), which also writes the new pages this rank
   owns. Only this pass donates and updates the KV cache.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.pcp import get_or_create_pcp_mesh
from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs as _batched_rpa_configs,
)
from vllm_torchtpu.kernels.experimental.batched_rpa_new import schedule_cp
from vllm_torchtpu.kernels.experimental.batched_rpa_new.wrapper import (
    ragged_paged_attention,
)
from vllm_torchtpu.layers.adapter.cp_attention import merge_attn_results

PCP_AXIS_NAME = "pcp"
TP_AXIS_NAME = "tp"

_KV_CACHE_SPEC = P(PCP_AXIS_NAME, None, TP_AXIS_NAME)
_TOKEN_SPEC = P(PCP_AXIS_NAME, TP_AXIS_NAME)
_INPUT_PARTITION_SPECS = (
    _KV_CACHE_SPEC,  # kv_cache
    _TOKEN_SPEC,  # query
    _TOKEN_SPEC,  # key
    _TOKEN_SPEC,  # value
    P(),  # seq_lens
    P(),  # block_tables
    P(),  # query_start_loc
    P(),  # request_distribution
)
# Output and LSE share the token and head axes.
_CACHE_OUTPUT_PARTITION_SPECS = (_TOKEN_SPEC, _TOKEN_SPEC)
_NEW_OUTPUT_PARTITION_SPECS = (_KV_CACHE_SPEC, _TOKEN_SPEC, _TOKEN_SPEC)

# `TPU_PCP_SHARDING_PLAN` -> the kernel's `sharding_mode`; the PCP manager
# picks its host plan from the same flag.
_KERNEL_SHARDING_MODES = {"headtail": "head_tail", "interleaved": "interleaved"}


def pcp_kernel_sharding_mode(sharding_plan: str) -> str:
    return _KERNEL_SHARDING_MODES[sharding_plan.lower()]


def _shard_map_specs(mesh: jax.sharding.Mesh) -> tuple[tuple, P, P]:
    """`(in_specs, token_spec, kv_cache_spec)` on `mesh`, with or without TP."""
    if TP_AXIS_NAME in mesh.axis_names:
        return _INPUT_PARTITION_SPECS, _TOKEN_SPEC, _KV_CACHE_SPEC
    token, kv_cache = P(PCP_AXIS_NAME), P(PCP_AXIS_NAME)
    return (kv_cache, token, token, token, P(), P(), P(), P()), token, kv_cache


def _cp_rank(pcp_size: int) -> jax.Array:
    """This rank's index on the PCP axis, as `[1]`.

    Not `lax.axis_index`: its partitionId has no meaning under the SPMD
    partitioning that runs the shard_map. Every rank sends `r` to rank `r`.
    """
    return jax.lax.all_to_all(
        jnp.arange(pcp_size, dtype=jnp.int32),
        axis_name=PCP_AXIS_NAME,
        split_axis=0,
        concat_axis=0,
        tiled=True,
    )[:1]


def pcp_cache_attention(
    mesh: jax.sharding.Mesh,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    sharding_mode: str,
    cp_kv_cache_interleave_size: int,
    kv_layout: _batched_rpa_configs.KVLayout,
) -> tuple[jax.Array, jax.Array]:
    """Pass 1 over global Q/K/V and KV cache: returns `(output, lse)`."""
    pcp_size = int(mesh.shape[PCP_AXIS_NAME])

    def per_rank(kv_cache, q, k, v, kv_lens, page_indices, cu_q_lens, distribution):
        scheduler, local_cu_q_lens, local_distribution = (
            schedule_cp.RingCacheScheduler.from_global_metadata(
                cu_q_lens,
                kv_lens,
                distribution,
                cp_rank=_cp_rank(pcp_size),
                cp_group_size=pcp_size,
                sharding_mode=sharding_mode,
                cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
                ring_axis_name=PCP_AXIS_NAME,
                ring_mesh_axis_names=tuple(mesh.axis_names),
            )
        )
        output, _, lse = ragged_paged_attention(
            q,
            k,
            v,
            kv_cache,
            page_indices=page_indices.reshape(-1),
            cu_q_lens=local_cu_q_lens,
            distribution=local_distribution,
            scheduler=scheduler,
            sm_scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            kv_layout=kv_layout,
            return_lse=True,
            skip_kv_update=True,
        )
        return output, lse

    in_specs, token_spec, _ = _shard_map_specs(mesh)
    return jax.shard_map(
        per_rank,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=(token_spec, token_spec),
        check_vma=False,
    )(kv_cache, q, k, v, kv_lens, page_indices, cu_q_lens, distribution)


def pcp_new_tokens_attention(
    mesh: jax.sharding.Mesh,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    sharding_mode: str,
    cp_kv_cache_interleave_size: int,
    kv_layout: _batched_rpa_configs.KVLayout,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Pass 2 over global Q/K/V and KV cache: returns `(output, new_kv_cache,
    lse)`."""
    pcp_size = int(mesh.shape[PCP_AXIS_NAME])

    def per_rank(kv_cache, q, k, v, kv_lens, page_indices, cu_q_lens, distribution):
        global_k, global_v = (
            schedule_cp.unpermute_pcp_kv(
                jax.lax.all_gather(x, axis_name=PCP_AXIS_NAME, axis=0, tiled=False),
                cu_q_lens,
                pcp_size,
                sharding_mode=sharding_mode,
                cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            )
            for x in (k, v)
        )
        scheduler, local_cu_q_lens, local_distribution = (
            schedule_cp.StripedNewTokensScheduler.from_global_metadata(
                cu_q_lens,
                kv_lens,
                distribution,
                cp_rank=_cp_rank(pcp_size),
                cp_group_size=pcp_size,
                sharding_mode=sharding_mode,
                cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
                update_kv_cache=True,
            )
        )
        return ragged_paged_attention(
            q,
            global_k,
            global_v,
            kv_cache,
            page_indices=page_indices.reshape(-1),
            cu_q_lens=local_cu_q_lens,
            distribution=local_distribution,
            scheduler=scheduler,
            sm_scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            kv_layout=kv_layout,
            return_lse=True,
        )

    in_specs, token_spec, kv_cache_spec = _shard_map_specs(mesh)
    return jax.shard_map(
        per_rank,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=(token_spec, kv_cache_spec, token_spec),
        check_vma=False,
    )(kv_cache, q, k, v, kv_lens, page_indices, cu_q_lens, distribution)


def _kernel_view(kv_cache, block_tables, query, key, kv_layout):
    # `attention.py` imports this module.
    from vllm_torchtpu.layers.adapter.attention import _to_kernel_kv_cache

    return _to_kernel_kv_cache(
        kv_cache, block_tables, query, key, kv_layout=kv_layout, block_major=False
    )


def _pallas_rpa_kernel_pcp_cache(
    kv_cache: jax.Array,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    sm_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    sharding_mode: str,
    cp_kv_cache_interleave_size: int,
    kv_layout: _batched_rpa_configs.KVLayout,
) -> tuple[jax.Array, jax.Array]:
    """Returns (output, lse)."""
    kv_cache, block_tables, _ = _kernel_view(
        kv_cache, block_tables, query, key, kv_layout
    )
    return pcp_cache_attention(
        mesh,
        query,
        key,
        value,
        kv_cache,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
        sm_scale=sm_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        sharding_mode=sharding_mode,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        kv_layout=kv_layout,
    )


def _pallas_rpa_kernel_pcp_new(
    kv_cache: jax.Array,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    sm_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    sharding_mode: str,
    cp_kv_cache_interleave_size: int,
    kv_layout: _batched_rpa_configs.KVLayout,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Returns (new_kv_cache, output, lse)."""
    kv_cache, block_tables, to_pool = _kernel_view(
        kv_cache, block_tables, query, key, kv_layout
    )
    output, new_kv_cache, lse = pcp_new_tokens_attention(
        mesh,
        query,
        key,
        value,
        kv_cache,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
        sm_scale=sm_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        sharding_mode=sharding_mode,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        kv_layout=kv_layout,
    )
    return to_pool(new_kv_cache), output, lse


# Keyed by config so layers with identical configs share one set of ops.
_PCP_KERNEL_REGISTRY: dict = {}


def _fake_lse(query: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        query.shape[0], query.shape[1], dtype=query.dtype, device=query.device
    )


def _fake_pcp_cache(kv_cache: torch.Tensor, query: torch.Tensor, *args, **kwargs):
    return torch.empty_like(query), _fake_lse(query)


def _fake_pcp_new(kv_cache: torch.Tensor, query: torch.Tensor, *args, **kwargs):
    return torch.empty_like(kv_cache), torch.empty_like(query), _fake_lse(query)


def build_pcp_kernels(
    sm_scale: float,
    k_scale: float | None,
    v_scale: float | None,
    sharding_mode: str,
    cp_kv_cache_interleave_size: int,
    kv_layout: _batched_rpa_configs.KVLayout = (
        _batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE
    ),
) -> tuple:
    """Build and cache the cache-pass and new-tokens-pass jax_ops.

    Returns:
        (cache_kernel, new_kernel), each callable as ``kernel(kv_cache, query,
        key, value, seq_lens, block_tables, query_start_loc,
        request_distribution) -> (output [T,H,D], lse [T,H])``; `new_kernel`
        also writes the new KV into `kv_cache`.
    """
    if sharding_mode not in _KERNEL_SHARDING_MODES.values():
        raise ValueError(f"Unknown PCP sharding_mode {sharding_mode!r}.")
    if sharding_mode == "interleaved" and cp_kv_cache_interleave_size <= 0:
        raise ValueError("Interleaved PCP needs cp_kv_cache_interleave_size > 0.")
    kv_layout = _batched_rpa_configs.KVLayout(kv_layout)
    config = (
        sm_scale,
        k_scale,
        v_scale,
        sharding_mode,
        cp_kv_cache_interleave_size,
        kv_layout,
    )
    cached = _PCP_KERNEL_REGISTRY.get(config)
    if cached is not None:
        return cached

    mesh = get_or_create_pcp_mesh(axis_name=PCP_AXIS_NAME, tp_axis_name=TP_AXIS_NAME)
    kwargs = dict(
        mesh=mesh,
        sm_scale=sm_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        sharding_mode=sharding_mode,
        # HeadTail positions do not depend on an interleave.
        cp_kv_cache_interleave_size=(
            cp_kv_cache_interleave_size if sharding_mode == "interleaved" else 1
        ),
        kv_layout=kv_layout,
    )
    instance = len(_PCP_KERNEL_REGISTRY)
    cache_op = pallas.jax_op(
        f"pallas::rpa_pcp_cache_{instance}",
        functools.partial(_pallas_rpa_kernel_pcp_cache, **kwargs),
        mesh=mesh,
        input_partition_specs=_INPUT_PARTITION_SPECS,
        output_partition_specs=_CACHE_OUTPUT_PARTITION_SPECS,
    )
    cache_op.register_fake(_fake_pcp_cache)
    # kv_cache (arg 0) is donated and written back with `copy_`, except in
    # kernel-iteration mode, as `attention.py`'s RPA ops do.
    new_op = pallas.jax_op(
        f"pallas::rpa_pcp_new_{instance}",
        functools.partial(_pallas_rpa_kernel_pcp_new, **kwargs),
        donate_argnums=None if envs.TPU_KERNEL_ITER_MODE else (0,),
        mesh=mesh,
        input_partition_specs=_INPUT_PARTITION_SPECS,
        output_partition_specs=_NEW_OUTPUT_PARTITION_SPECS,
    )
    new_op.register_fake(_fake_pcp_new)

    def new_kernel(kv_cache, *args):
        new_kv_cache, output, lse = new_op(kv_cache, *args)
        kv_cache.copy_(new_kv_cache)
        return output, lse

    kernels = (cache_op, new_kernel)
    _PCP_KERNEL_REGISTRY[config] = kernels
    return kernels


def forward_with_pcp(
    *,
    cache_kernel,
    new_kernel,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
) -> torch.Tensor:
    """Two-pass PCP attention forward: the cache pass, then the new-tokens
    pass (which writes the new KV), merged by LSE."""
    args = (
        query,
        key,
        value,
        attn_metadata.seq_lens,
        attn_metadata.block_tables,
        attn_metadata.query_start_loc,
        attn_metadata.request_distribution,
    )
    cache_output, cache_lse = cache_kernel(kv_cache, *args)
    new_output, new_lse = new_kernel(kv_cache, *args)
    output, _ = merge_attn_results(cache_output, cache_lse, new_output, new_lse)
    return output.to(query.dtype)
