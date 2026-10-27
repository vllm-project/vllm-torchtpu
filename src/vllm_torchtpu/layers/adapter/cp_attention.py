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
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import torch
from torch.nn import Parameter
from torch_tpu._internal import pallas
from vllm.config import get_current_vllm_config_or_none
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.linear import (
    WEIGHT_LOADER_V2_SUPPORTED,
    LinearBase,
    set_weight_attrs,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.dcp import get_dcp_group
from vllm_torchtpu.distributed.dcp_comm import (
    cp_lse_ag_out_rs,
    cp_lse_ag_out_scale,
)
from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs as _batched_rpa_configs,
)
from vllm_torchtpu.kernels.experimental.batched_rpa import (
    wrapper as _batched_rpa_wrapper,
)
from vllm_torchtpu.layers.adapter.linear import TpuRowParallelLinear
from vllm_torchtpu.utils import synchronize_tensors


def is_dcp_duplicate_o_proj_enabled(vllm_config: Any | None = None) -> bool:
    """Return True when DCP duplicate o_proj is enabled."""
    vllm_config = vllm_config or get_current_vllm_config_or_none()
    parallel_config = (
        None if vllm_config is None else getattr(vllm_config, "parallel_config", None)
    )
    if (
        parallel_config is not None
        and getattr(parallel_config, "prefill_context_parallel_size", 1) > 1
    ):
        return False
    dcp_size = (
        1
        if parallel_config is None
        else getattr(parallel_config, "decode_context_parallel_size", 1)
    )
    if dcp_size <= 1:
        dcp_group = get_dcp_group()
        if dcp_group is not None:
            dcp_size = int(dcp_group.world_size)
    flag = bool(envs.TPU_DCP_DUPLICATE_O_PROJ) or (
        parallel_config is not None
        and bool(getattr(parallel_config, "dcp_duplicate_o_proj", False))
    )
    return dcp_size > 1 and flag


class DCPGroupRowParallelLinear(TpuRowParallelLinear):
    """RowParallelLinear sharded across DCP groups (tp_size // dcp_size)."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        reduce_results: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        vllm_config = get_current_vllm_config_or_none()
        parallel_config = (
            None
            if vllm_config is None
            else getattr(vllm_config, "parallel_config", None)
        )
        dcp_world_size = (
            None
            if parallel_config is None
            else getattr(parallel_config, "decode_context_parallel_size", None)
        )
        if dcp_world_size is None:
            dcp_group = get_dcp_group()
            dcp_world_size = int(dcp_group.world_size) if dcp_group is not None else 1

        rank = get_tensor_model_parallel_rank() if not disable_tp else 0
        world_size = get_tensor_model_parallel_world_size() if not disable_tp else 1

        self.group_size = max(int(dcp_world_size), 1) if not disable_tp else 1
        self.orep_active = self.group_size > 1
        self.tp_world_size = world_size

        self.tp_rank = rank // self.group_size
        self.tp_size = divide(world_size, self.group_size)

        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]

        LinearBase.__init__(
            self,
            input_size,
            output_size,
            bias,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )

        if not input_is_parallel:
            raise ValueError(
                "DCPGroupRowParallelLinear requires input_is_parallel=True."
            )
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        loader = (
            self.weight_loader_v2
            if any(
                cls.__name__ in WEIGHT_LOADER_V2_SUPPORTED
                for cls in self.quant_method.__class__.__mro__
            )
            else self.weight_loader
        )
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=loader,
        )
        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError(
                "When not reduce the results, adding bias to the "
                "results can lead to incorrect results"
            )

        if bias:
            self.bias = Parameter(torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)
        self.update_param_tp_status()

    def forward(
        self,
        input_: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        output_parallel = self.quant_method.apply(self, input_, None)

        if self.reduce_results and self.tp_world_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        if not self.skip_bias_add and self.bias is not None:
            output = output + self.bias

        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


# ---------------------------------------------------------------------------
# LSE-weighted combination
# ---------------------------------------------------------------------------


def merge_attn_results(
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


def merge_dcp_new_into_ctx(
    ctx_output: torch.Tensor,
    ctx_lse: torch.Tensor,
    new_output: torch.Tensor,
    new_lse: torch.Tensor,
    *,
    dcp_rank: int,
    dcp_world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge new-token attention into this DCP rank's slice of context attention."""
    if ctx_output.shape[1] != new_output.shape[1] * dcp_world_size:
        raise ValueError(
            f"ctx_output heads {ctx_output.shape[1]} must equal new_output heads "
            f"{new_output.shape[1]} * dcp_world_size {dcp_world_size}"
        )

    heads_local = new_output.shape[1]
    start_head = dcp_rank * heads_local
    end_head = (dcp_rank + 1) * heads_local

    own_ctx_out = ctx_output[:, start_head:end_head, :]
    own_ctx_lse = ctx_lse[:, start_head:end_head].to(torch.float32)

    merged_own_out, merged_own_lse = merge_attn_results(
        own_ctx_out,
        own_ctx_lse,
        new_output,
        new_lse.to(torch.float32),
    )

    merged_ctx_out = ctx_output.clone()
    merged_ctx_out[:, start_head:end_head, :] = merged_own_out.to(ctx_output.dtype)
    merged_ctx_lse = ctx_lse.to(torch.float32).clone()
    merged_ctx_lse[:, start_head:end_head] = merged_own_lse
    return merged_ctx_out, merged_ctx_lse


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
    q_dcp_replicated: bool = False,
    dcp_duplicate_o_proj: bool = False,
) -> torch.Tensor:
    """Two-pass DCP attention forward.
    1. CACHE_ONLY: query is all-gathered by heads to attend every rank's KV
        shard, then partials are merged by `cp_lse_ag_out_rs` back to own heads.
    2. NEW_TOKENS_ONLY: new tokens aren't DCP-sharded (every rank has the
        full local K/V for this step), so this rank's own heads attend
            directly; no gather/combine needed.

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

    if q_dcp_replicated:
        if query.shape[1] % dcp_world_size != 0:
            raise ValueError(
                f"Query heads {query.shape[1]} must be divisible by "
                f"dcp_world_size {dcp_world_size}"
            )
        own_q_heads = query.shape[1] // dcp_world_size
        query_across_dcp = query.contiguous()
        query_local = query.narrow(1, dcp_rank * own_q_heads, own_q_heads).contiguous()
    else:
        query_across_dcp = dcp_group.all_gather(query.contiguous(), dim=1)
        query_local = query

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
    new_output, new_lse = new_kernel(kv_cache, query_local, *shared_args)

    if dcp_duplicate_o_proj:
        merged_ctx_out, merged_ctx_lse = merge_dcp_new_into_ctx(
            ctx_output,
            ctx_lse,
            new_output,
            new_lse,
            dcp_rank=dcp_rank,
            dcp_world_size=dcp_world_size,
        )
        scaled_out = cp_lse_ag_out_scale(
            merged_ctx_out, merged_ctx_lse, dcp_group, return_lse=False
        )
        output = scaled_out.to(new_output.dtype)
    else:
        ctx_output, ctx_lse = cp_lse_ag_out_rs(
            ctx_output, ctx_lse, dcp_group, return_lse=True
        )
        output, _ = merge_attn_results(ctx_output, ctx_lse, new_output, new_lse)
        output = output.to(new_output.dtype)  # f32 ctx_lse promotes the merge

    if not torch.compiler.is_compiling():
        synchronize_tensors((kv_cache, output), wait=False)
    return output
