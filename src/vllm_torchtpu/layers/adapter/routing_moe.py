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
"""Torch/TPU bridge for Tokamax single-shard / EP-shard `routing_moe`."""

from __future__ import annotations

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import torch
from jax.experimental import pallas as pl
from torch_tpu._internal import pallas

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.experimental.routing_moe.common import arg
from vllm_torchtpu.kernels.experimental.routing_moe.gmm import kernel as gmm
from vllm_torchtpu.kernels.experimental.routing_moe.gmm import plan as gmm_plan_lib
from vllm_torchtpu.kernels.experimental.routing_moe.io_relayout import (
    input_pack,
    output_reduce,
)

_SUBLANES = 8
_PACKED_SORT_SHIFT = 20
_MAX_PACKED_GROUPS = 1 << (31 - _PACKED_SORT_SHIFT)
_MAX_PACKED_ROWS = 1 << _PACKED_SORT_SHIFT

_kernel_counter = 0
_routing_moe_cache: dict[tuple[int, int, str, bool, str, int | None], Callable] = {}


def _select_gmm_plan(
    num_tokens: int,
    *,
    acc_dtype_name: str,
    x_quant_block_size: int | None,
) -> gmm_plan_lib.RoutingGmmPlan:
    acc_dtype = jnp.float32 if acc_dtype_name == "float32" else jnp.bfloat16
    aligned = max(_SUBLANES, _SUBLANES * pl.cdiv(num_tokens, _SUBLANES))
    if num_tokens <= 64:
        return gmm_plan_lib.RoutingGmmPlan(
            tile_m=min(16, aligned),
            bucket_base=_SUBLANES,
            num_w_buffers=3,
            acc_dtype=acc_dtype,
            x_quant_block_size=x_quant_block_size,
        )
    if num_tokens <= 256:
        return gmm_plan_lib.RoutingGmmPlan(
            tile_m=min(32, aligned),
            bucket_base=min(16, aligned),
            num_w_buffers=3,
            acc_dtype=acc_dtype,
            x_quant_block_size=x_quant_block_size,
        )
    return gmm_plan_lib.RoutingGmmPlan(
        acc_dtype=acc_dtype,
        x_quant_block_size=x_quant_block_size,
    )


def _sort_routed_indices(
    flat_indices: jax.Array, num_groups: int
) -> tuple[jax.Array, jax.Array]:
    num_rows = flat_indices.size
    if num_groups < _MAX_PACKED_GROUPS and num_rows <= _MAX_PACKED_ROWS:
        iota = jnp.arange(num_rows, dtype=jnp.uint32)
        packed = (flat_indices.astype(jnp.uint32) << _PACKED_SORT_SHIFT) | iota
        sorted_packed = jax.lax.sort(packed)
        row_group = (sorted_packed >> _PACKED_SORT_SHIFT).astype(jnp.int32)
        mask = jnp.uint32(_MAX_PACKED_ROWS - 1)
        argsort_indices = (sorted_packed & mask).astype(jnp.int32)
        return row_group, argsort_indices
    return jax.lax.sort_key_val(
        flat_indices,
        jnp.arange(num_rows, dtype=jnp.int32),
        is_stable=True,
    )


def routing_moe_jax_fn(
    hidden_states: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2_bias: jax.Array | None,
    topk_weights: jax.Array,
    topk_ids: jax.Array,
    experts_start: jax.Array,
    *,
    topk: int,
    global_num_experts: int,
    activation: str,
    use_ep: bool,
    acc_dtype_name: str,
    x_quant_block_size: int | None,
) -> jax.Array:
    if w1_bias is not None or w2_bias is not None:
        raise NotImplementedError("routing_moe does not support expert bias.")

    local_num_experts = w1.shape[0]
    num_experts = max(int(global_num_experts), local_num_experts)
    group_offset = (
        experts_start.reshape(1).astype(jnp.int32)
        if use_ep
        else jnp.zeros((1,), dtype=jnp.int32)
    )

    # Biasless gated MLPs map an all-zero input row to an all-zero output row,
    # so zeroed padding rows skip expert routing without host-side masks.
    nonzero_token = jnp.any(hidden_states != 0, axis=-1, keepdims=True)
    effective_weights = jnp.where(nonzero_token, topk_weights, 0)
    ids_i32 = topk_ids.astype(jnp.int32)
    valid = (ids_i32 >= 0) & (ids_i32 < num_experts) & (effective_weights != 0)
    effective_ids = jnp.where(valid, ids_i32, jnp.int32(num_experts))
    flat_ids = effective_ids.reshape(-1)

    row_group, argsort_indices = _sort_routed_indices(flat_ids, num_experts + 1)
    group_sizes = jax.nn.one_hot(flat_ids, num_experts, dtype=jnp.int32).sum(axis=0)

    x_compact = input_pack.input_pack(hidden_states)
    gmm_plan = _select_gmm_plan(
        hidden_states.shape[0],
        acc_dtype_name=acc_dtype_name,
        x_quant_block_size=x_quant_block_size,
    )
    gmm_res = gmm.routing_gmm(
        x=x_compact,
        w=arg.SingleWeightWrapper.create(w1, w1_scale, None, fused_gate_up=True),
        w2=arg.SingleWeightWrapper.create(w2, w2_scale, None),
        group_sizes=group_sizes,
        gather=arg.RoutingWrapper.create(argsort_indices // topk),
        scatter=arg.RoutingWrapper.create(argsort_indices),
        fuse_act=activation,
        group_offset=group_offset,
        row_group=row_group,
        custom_plan=gmm_plan,
    )
    if use_ep or num_experts > local_num_experts:
        return output_reduce.output_reduce_unpack(
            gmm_res,
            effective_weights,
            effective_ids,
            group_offset,
            local_num_experts,
            out_dtype=hidden_states.dtype,
        )
    return output_reduce.output_reduce_unpack(
        gmm_res,
        effective_weights,
        out_dtype=hidden_states.dtype,
    )


def _get_routing_moe_op(
    *,
    topk: int,
    global_num_experts: int,
    activation: str,
    use_ep: bool,
    acc_dtype_name: str | None = None,
    x_quant_block_size: int | None = None,
) -> Callable:
    resolved_acc = (acc_dtype_name or envs.VLLM_TPU_ROUTING_MOE_ACC_DTYPE).lower()
    if resolved_acc not in ("float32", "bfloat16"):
        raise ValueError(
            "VLLM_TPU_ROUTING_MOE_ACC_DTYPE must be 'float32' or 'bfloat16', "
            f"got {resolved_acc!r}."
        )
    resolved_x_block = (
        envs.VLLM_TPU_ROUTING_MOE_X_QUANT_BLOCK
        if x_quant_block_size is None
        else x_quant_block_size
    )
    key = (
        int(topk),
        int(global_num_experts),
        str(activation),
        bool(use_ep),
        resolved_acc,
        resolved_x_block,
    )
    cached = _routing_moe_cache.get(key)
    if cached is not None:
        return cached

    global _kernel_counter
    op_name = f"pallas::tokamax_routing_moe_{_kernel_counter}"
    _kernel_counter += 1

    wrapped = functools.partial(
        routing_moe_jax_fn,
        topk=int(topk),
        global_num_experts=int(global_num_experts),
        activation=str(activation),
        use_ep=bool(use_ep),
        acc_dtype_name=resolved_acc,
        x_quant_block_size=resolved_x_block,
    )
    custom_op = pallas.jax_op(op_name, wrapped)

    def _fake_routing_moe(
        hidden_states: torch.Tensor, *_args, **_kwargs
    ) -> torch.Tensor:
        return torch.empty_like(hidden_states)

    custom_op.register_fake(_fake_routing_moe)
    _routing_moe_cache[key] = custom_op
    return custom_op


def prebuild_routing_moe_kernel(
    *,
    topk: int,
    global_num_experts: int,
    activation: str,
    use_ep: bool,
) -> None:
    """Prebuild and cache `routing_moe` custom op outside `torch.compile`."""
    _get_routing_moe_op(
        topk=topk,
        global_num_experts=global_num_experts,
        activation=activation,
        use_ep=use_ep,
    )


def routing_moe_forward(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    w1_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    experts_start: torch.Tensor | None,
    topk: int,
    global_num_experts: int,
    activation: str,
) -> torch.Tensor:
    """Run Tokamax `routing_moe` on one shard or EP shard."""
    use_ep = experts_start is not None
    op = _get_routing_moe_op(
        topk=topk,
        global_num_experts=global_num_experts,
        activation=activation,
        use_ep=use_ep,
    )
    if experts_start is None:
        experts_start = torch.zeros((), dtype=torch.int32, device=hidden_states.device)
    return op(
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        w1_bias,
        w2_bias,
        topk_weights,
        topk_ids,
        experts_start,
    )
