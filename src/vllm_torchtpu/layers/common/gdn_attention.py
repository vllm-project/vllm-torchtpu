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
"""
Bridge the torch gdn_attention_core op for gated deltanet attention TPU impl

"""
import dataclasses
import functools
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

import vllm_torchtpu.layers.common.ragged_gated_delta_rule_wrapper as ragged_gated_delta_rule_wrapper
from vllm_torchtpu.kernels.causal_conv1d import causal_conv1d
from vllm_torchtpu.kernels.gdn.v3 import wrapper as gdn_v3_wrapper
from vllm_torchtpu.layers.common.ragged_gated_delta_rule_ref import \
    ragged_gated_delta_rule as ragged_gated_delta_rule_ref
from vllm_torchtpu.layers.common.sharding import ShardingAxisName
from vllm_torchtpu.layers.common.utils import (
    inverse_reorder_for_sharding, reorder_concatenated_tensor_for_sharding)
from vllm_torchtpu.utils import get_mesh_shape_product

RaggedGatedDeltaRuleImpl = ragged_gated_delta_rule_wrapper.RaggedGatedDeltaRuleImpl


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GdnAttentionConfig:
    ragged_gated_delta_rule_impl: RaggedGatedDeltaRuleImpl = (
        RaggedGatedDeltaRuleImpl.REF)


def run_jax_gdn_attention_local(
    mixed_qkv: jnp.ndarray,
    b: jnp.ndarray,
    a: jnp.ndarray,
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    conv_weight: jnp.ndarray,
    conv_bias: Optional[jnp.ndarray],
    A_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    distribution: jnp.ndarray,
    seq_lens: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    dp_enabled: bool,
    config: GdnAttentionConfig = GdnAttentionConfig(),
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Runs the local JAX GDN attention mechanism with combined QKV tensors.

    Args:
        mixed_qkv: Combined QKV tensor of shape `(num_tokens, dim)`.
        b: B tensor of shape `(num_tokens, n_v)`.
        a: A tensor of shape `(num_tokens, n_v)`.
        conv_state: Combined convolutional state of shape `(num_blocks,
          kernel_size - 1, dim)`. `num_blocks` is always equal or larger than
          `max_seqs + 1`. The first block is a null_block and only used for
          padded / invalid tokens.
        recurrent_state: Recurrent state of shape `(num_blocks, n_v, d_k, d_v)`.
        conv_weight: Combined convolutional weight of shape `(dim, 1,
          kernel_size)`.
        conv_bias: Optional combined convolutional bias of shape `(dim,)`.
        A_log: Log of A parameter of shape `(n_v,)`.
        dt_bias: Delta T bias of shape `(n_v,)`.
        query_start_loc: Tensor of shape `(num_seqs + 1,)` with start locations of
          each sequence.
        state_indices: Tensor of shape `(max_reqs,)` mapping request index to
          state index.
        distribution: Tensor of shape `(3,)` int32 — `(decode_end, prefill_end,
          mixed_end)`.
        seq_lens: Tensor of shape `(max_reqs,)` with the total sequence length
          per request (computed + scheduled). Used to derive
          ``has_initial_state`` so brand-new prefills don't read stale state
          from a reused mamba slot, mirroring GPU's
          ``initial_state[~has_initial_state, ...] = 0`` in
          ``gdn_linear_attn._forward_core``.
        n_kq: Number of key/query heads.
        n_v: Number of value heads.
        d_k: Dimension of key.
        d_v: Dimension of value.
        kernel_size: Convolution kernel size.
        config: Configuration for implementation selection.

    Returns:
        A tuple containing:
        - A tuple of (new_conv_state, new_recurrent_state).
        - The output tensor of shape `(num_tokens, n_v * d_v)`.
    """
    # has_initial_state[i] = True iff request i already has computed
    # tokens in its mamba slot (chunked-prefill continuation, prefix-cache
    # hit, or running decode). False for brand-new prefills, in which
    # case the conv1d, the chunked / ref delta-rule impls, and the fused
    # Pallas recurrent kernel all zero the slot's prior state before
    # the update so a freshly-allocated mamba slot can't leak its
    # previous tenant's state. context_len = seq_len - query_len.
    max_reqs = seq_lens.shape[0]
    query_lens = query_start_loc[1:max_reqs + 1] - query_start_loc[:max_reqs]
    has_initial_state = (seq_lens - query_lens) > 0

    if config.ragged_gated_delta_rule_impl == RaggedGatedDeltaRuleImpl.CHUNKED_KERNEL_V3_PD:
        return gdn_v3_wrapper.fused_conv1d_gdn(
            mixed_qkv,
            b,
            a,
            conv_state,
            recurrent_state,
            conv_weight,
            conv_bias,
            A_log,
            dt_bias,
            query_start_loc,
            state_indices,
            distribution,
            seq_lens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
        )

    out_mixed_qkv, new_conv_state = causal_conv1d.ragged_causal_conv1d(
        mixed_qkv,
        conv_state,
        conv_weight,
        conv_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        kernel_size=kernel_size,
    )

    if config.ragged_gated_delta_rule_impl == RaggedGatedDeltaRuleImpl.REF:
        ragged_gdn_impl = functools.partial(
            ragged_gated_delta_rule_ref,
            has_initial_state=has_initial_state,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
        )
        new_recurrent_state, output = ragged_gdn_impl(
            out_mixed_qkv,
            b,
            a,
            recurrent_state,
            A_log,
            dt_bias,
            query_start_loc,
            state_indices,
            distribution,
        )
    else:
        wrapper_config = config.ragged_gated_delta_rule_impl.to_config()
        # DP replicas hold the full-width GDN heads, so the wide (64) scan chunk
        # overflows VMEM (CompileTimeScopedVmemOom) -- use 32 there.
        chunk_size = 32 if dp_enabled else 64
        new_recurrent_state, output = ragged_gated_delta_rule_wrapper.ragged_gated_delta_rule_wrapper(
            mixed_qkv=out_mixed_qkv,
            b=b,
            a=a,
            recurrent_state=recurrent_state,
            A_log=A_log,
            dt_bias=dt_bias,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            distribution=distribution,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            config=wrapper_config,
            chunk_size=chunk_size,
            has_initial_state=has_initial_state,
        )

    return (new_conv_state, new_recurrent_state), output


def run_jax_gdn_attention(
    j_mixed_qkv: jnp.ndarray,
    j_b: jnp.ndarray,
    j_a: jnp.ndarray,
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    j_conv_weight: jnp.ndarray,
    j_conv_bias: Optional[jnp.ndarray],
    j_A_log: jnp.ndarray,
    j_dt_bias: jnp.ndarray,
    state_indices: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    distribution: jnp.ndarray,
    seq_lens: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    mesh: jax.sharding.Mesh,
    dp_enabled: bool,
    config: GdnAttentionConfig = GdnAttentionConfig(),
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Runs the Jax GDN attention mechanism.

    Args:
        j_mixed_qkv: Input tensor of shape `(num_tokens, dim)`.
        j_b: Input tensor of shape `(num_tokens, n_v)`.
        j_a: Input tensor of shape `(num_tokens, n_v)`.
        conv_state: Convolutional state tensor of shape `(num_blocks, kernel_size
          - 1, dim)`. `num_blocks` is always equal or larger than `max_seqs +
          1`. The first block is a null_block and only used for padded / invalid
          tokens.
        recurrent_state: Recurrent state tensor of shape `(num_blocks, n_v, d_k,
          d_v)`.
        j_conv_weight: Convolutional weight tensor of shape `(dim, 1,
          kernel_size)`.
        j_conv_bias: Optional convolutional bias tensor of shape `(dim,)`.
        j_A_log: Log of A parameter tensor of shape `(n_v,)`.
        j_dt_bias: Delta T bias tensor of shape `(n_v,)`.
        state_indices: Tensor of shape `(max_reqs,)` mapping request index to
          state index.
        query_start_loc: Tensor of shape `(num_seqs + 1,)` with start locations of
          each sequence.
        distribution: Tensor of shape `(3,)` int32 — `(decode_end, prefill_end,
          mixed_end)`.
        seq_lens: Tensor of shape `(max_reqs,)` with the total sequence length
          per request (computed + scheduled). Used inside the local function
          to derive ``has_initial_state``.
        n_kq: Number of key/query heads.
        n_v: Number of value heads.
        d_k: Dimension of key.
        d_v: Dimension of value.
        kernel_size: Convolution kernel size.
        mesh: The device mesh for distributed computation.
        config: Configuration for implementation selection.

    Returns:
        A tuple containing:
        - A tuple of (new_conv_state, new_recurrent_state).
          - new_conv_state: `(num_blocks, kernel_size - 1, dim)`
          - new_recurrent_state: `(num_blocks, n_v, d_k, d_v)`
        - The output tensor of shape `(num_tokens, n_v * d_v)`.
    """
    in_specs = (
        P(ShardingAxisName.ATTN_DATA,
          ShardingAxisName.ATTN_HEAD),  # j_mixed_qkv
        P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD),  # j_b
        P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD),  # j_a
        P(ShardingAxisName.ATTN_DATA, None,
          ShardingAxisName.ATTN_HEAD),  # conv_state
        P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None,
          None),  # recurrent_state
        P(ShardingAxisName.ATTN_HEAD, None, None),  # j_conv_weight
        P(ShardingAxisName.ATTN_HEAD)
        if j_conv_bias is not None else None,  # j_conv_bias
        P(ShardingAxisName.ATTN_HEAD),  # j_A_log
        P(ShardingAxisName.ATTN_HEAD),  # j_dt_bias
        P(ShardingAxisName.ATTN_DATA),  # query_start_loc
        P(ShardingAxisName.ATTN_DATA),  # state_indices
        P(ShardingAxisName.ATTN_DATA),  # distribution
        P(ShardingAxisName.ATTN_DATA),  # seq_lens
    )

    out_specs = (
        (
            P(ShardingAxisName.ATTN_DATA, None,
              ShardingAxisName.ATTN_HEAD),  # new_conv_state
            P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None,
              None),  # new_recurrent_state
        ),
        P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD),  # output
    )

    tp_size = get_mesh_shape_product(mesh, ShardingAxisName.ATTN_HEAD)

    p_run_jax_gdn_attention_local = functools.partial(
        run_jax_gdn_attention_local,
        n_kq=n_kq // tp_size,
        n_v=n_v // tp_size,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        dp_enabled=dp_enabled,
        config=config,
    )

    mapped_fn = jax.shard_map(
        p_run_jax_gdn_attention_local,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )

    (new_conv_state, new_recurrent_state), output = mapped_fn(
        j_mixed_qkv,
        j_b,
        j_a,
        conv_state,
        recurrent_state,
        j_conv_weight,
        j_conv_bias,
        j_A_log,
        j_dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        seq_lens,
    )

    return (new_conv_state, new_recurrent_state), output


def _exchange_pcp_token_shards_for_head_shards(
    tensor: jnp.ndarray,
    pcp_axis: str,
    pcp_size: int,
    *,
    head_axis: int = -1,
) -> jnp.ndarray:
    """Exchange PCP token shards for PCP head shards.

    Native torchtpu-vLLM PCP feeds rank-local token shards into this op. The
    op-local PCP mesh then treats PCP as extra head parallelism for the GDN
    recurrent scan: each PCP rank receives all tokens for only its local head
    slice.
    """
    if head_axis < 0:
        head_axis += tensor.ndim
    assert tensor.shape[head_axis] % pcp_size == 0
    return jax.lax.all_to_all(tensor,
                              axis_name=pcp_axis,
                              split_axis=head_axis,
                              concat_axis=0,
                              tiled=True)


def _select_replicated_shard_for_pcp_rank(
    tensor: jnp.ndarray,
    pcp_axis: str,
    pcp_size: int,
    *,
    axis: int,
) -> jnp.ndarray:
    """Select this PCP rank shard without emitting axis_index/partitionId."""
    if axis < 0:
        axis += tensor.ndim
    assert tensor.shape[axis] % pcp_size == 0
    shard_size = tensor.shape[axis] // pcp_size
    # Every PCP rank starts with the same replicated full tensor. all_to_all
    # routes split ``rank`` to PCP rank ``rank``; because all inputs are equal,
    # each rank receives pcp_size copies of its own shard. Taking the first
    # shard with a constant slice avoids emitting a partitionId/axis_index op.
    exchanged = jax.lax.all_to_all(tensor,
                                   axis_name=pcp_axis,
                                   split_axis=axis,
                                   concat_axis=axis,
                                   tiled=True)
    return jax.lax.dynamic_slice_in_dim(exchanged, 0, shard_size, axis=axis)


def _pcp_rank_token_count_before(x: jnp.ndarray, ranks: jnp.ndarray, *,
                                 pcp_size: int,
                                 interleave_size: int) -> jnp.ndarray:
    cycle = pcp_size * interleave_size
    full_cycles = x // cycle
    cycle_offsets = x - full_cycles * cycle
    rank_starts = ranks * interleave_size
    in_cycle = jnp.clip(cycle_offsets - rank_starts,
                        min=0,
                        max=interleave_size)
    return full_cycles * interleave_size + in_cycle


def _derive_pcp_rank_major_reorder_indices(
    query_start_loc: jnp.ndarray,
    *,
    pcp_size: int,
    interleave_size: int,
    local_padded_num_tokens: int,
    seq_lens: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Rebuild packed-rank-major -> request-major token indices."""
    q_lens = query_start_loc[1:] - query_start_loc[:-1]
    req_count = q_lens.shape[0]
    if seq_lens is None:
        q_global_starts = jnp.zeros_like(q_lens)
    else:
        q_global_starts = seq_lens[:req_count].astype(q_lens.dtype) - q_lens
    q_global_ends = q_global_starts + q_lens

    rank_ids = jnp.arange(pcp_size, dtype=jnp.int32).reshape(pcp_size, 1)
    rank_counts = (
        _pcp_rank_token_count_before(q_global_ends.reshape(1, req_count),
                                     rank_ids,
                                     pcp_size=pcp_size,
                                     interleave_size=interleave_size) -
        _pcp_rank_token_count_before(q_global_starts.reshape(1, req_count),
                                     rank_ids,
                                     pcp_size=pcp_size,
                                     interleave_size=interleave_size))
    rank_req_ends = jnp.cumsum(rank_counts, axis=1)
    rank_req_starts = rank_req_ends - rank_counts

    full_padded_num_tokens = local_padded_num_tokens * pcp_size
    original_indices = jnp.arange(full_padded_num_tokens, dtype=jnp.int32)
    total_valid_tokens = query_start_loc[-1].astype(jnp.int32)
    original_valid = original_indices < total_valid_tokens
    request_ids = jnp.sum(
        original_indices[:, None]
        >= query_start_loc[1:].astype(jnp.int32)[None, :],
        axis=1,
    ).astype(jnp.int32)
    max_request_id = req_count - 1
    safe_request_ids = jnp.minimum(request_ids, max_request_id)
    original_offsets = (original_indices -
                        query_start_loc[safe_request_ids].astype(jnp.int32))
    original_global_positions = (
        q_global_starts[safe_request_ids].astype(jnp.int32) + original_offsets)

    cycle = pcp_size * interleave_size
    cycle_offsets = (original_global_positions -
                     (original_global_positions // cycle) * cycle)
    original_ranks = (cycle_offsets // interleave_size).astype(jnp.int32)
    in_req_rank_offsets = (
        _pcp_rank_token_count_before(original_global_positions,
                                     original_ranks,
                                     pcp_size=pcp_size,
                                     interleave_size=interleave_size) -
        _pcp_rank_token_count_before(q_global_starts[safe_request_ids].astype(
            jnp.int32),
                                     original_ranks,
                                     pcp_size=pcp_size,
                                     interleave_size=interleave_size))
    rank_offsets = rank_req_starts[original_ranks, safe_request_ids]
    packed_destinations = (original_ranks * local_padded_num_tokens +
                           rank_offsets + in_req_rank_offsets)

    sentinel = jnp.full_like(packed_destinations, full_padded_num_tokens)
    packed_destinations = jnp.where(original_valid, packed_destinations,
                                    sentinel)
    reorder_with_sentinel = jnp.zeros((full_padded_num_tokens + 1, ),
                                      dtype=jnp.int32)
    reorder_values = jnp.where(original_valid, original_indices,
                               jnp.zeros_like(original_indices))
    reorder_with_sentinel = reorder_with_sentinel.at[packed_destinations].set(
        reorder_values)
    valid_with_sentinel = jnp.zeros((full_padded_num_tokens + 1, ),
                                    dtype=jnp.bool_)
    valid_with_sentinel = valid_with_sentinel.at[packed_destinations].set(
        original_valid)
    reorder_indices = reorder_with_sentinel[:-1]
    valid_mask = valid_with_sentinel[:-1]
    return jnp.where(valid_mask, reorder_indices, -1).astype(jnp.int32)


def run_jax_gdn_attention_pcp_tp_prefill(
    j_mixed_qkv: jnp.ndarray,
    j_b: jnp.ndarray,
    j_a: jnp.ndarray,
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    j_conv_weight: jnp.ndarray,
    j_conv_bias: Optional[jnp.ndarray],
    j_A_log: jnp.ndarray,
    j_dt_bias: jnp.ndarray,
    state_indices: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    distribution: jnp.ndarray,
    seq_lens: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    pcp_size: int,
    interleave_size: int,
    mesh: jax.sharding.Mesh,
    config: GdnAttentionConfig = GdnAttentionConfig(),
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """GDN PCP prefill using an op-local PCP mesh.

    The enclosing vLLM worker is already TP-local. This function only adds PCP
    as extra head parallelism inside the custom op, without reintroducing a
    global JAX ``pcp`` axis in sharding.py.

    The packed-rank-major token order is reconstructed from replicated metadata
    inside this op so callers do not need to pass a GDN-specific reorder tensor.
    """
    pcp_axis = "pcp"
    if pcp_axis not in mesh.axis_names:
        raise NotImplementedError("GDN PCP prefill requires a pcp mesh axis.")
    if mesh.shape[pcp_axis] != pcp_size:
        raise ValueError(f"pcp_size={pcp_size} does not match mesh axis size "
                         f"{mesh.shape[pcp_axis]}.")
    if n_kq % pcp_size != 0:
        raise ValueError(
            f"n_kq={n_kq} must be divisible by pcp_size={pcp_size}.")
    if n_v % pcp_size != 0:
        raise ValueError(
            f"n_v={n_v} must be divisible by pcp_size={pcp_size}.")

    local_n_kq = n_kq // pcp_size
    local_n_v = n_v // pcp_size
    local_key_dim = n_kq * d_k
    local_value_dim = n_v * d_v

    token_spec = P(pcp_axis, None)
    replicated_spec = P()
    in_specs = (
        token_spec,  # j_mixed_qkv
        token_spec,  # j_b
        token_spec,  # j_a
        replicated_spec,  # conv_state
        replicated_spec,  # recurrent_state
        replicated_spec,  # j_conv_weight
        replicated_spec if j_conv_bias is not None else None,  # j_conv_bias
        replicated_spec,  # j_A_log
        replicated_spec,  # j_dt_bias
        replicated_spec,  # query_start_loc
        replicated_spec,  # state_indices
        replicated_spec,  # distribution
        replicated_spec,  # seq_lens
    )
    out_specs = ((replicated_spec, replicated_spec), replicated_spec)

    def _pcp_prefill_fn(
        local_qkv,
        local_b,
        local_a,
        conv_state_,
        recurrent_state_,
        conv_weight_,
        conv_bias_,
        A_log_,
        dt_bias_,
        query_start_loc_,
        state_indices_,
        distribution_,
        seq_lens_,
    ):
        interleaved_qkv = reorder_concatenated_tensor_for_sharding(
            local_qkv,
            [local_key_dim, local_key_dim, local_value_dim],
            pcp_size,
            -1,
        )
        packed_qkv_shard = _exchange_pcp_token_shards_for_head_shards(
            interleaved_qkv, pcp_axis, pcp_size)
        packed_b_shard = _exchange_pcp_token_shards_for_head_shards(
            local_b, pcp_axis, pcp_size)
        packed_a_shard = _exchange_pcp_token_shards_for_head_shards(
            local_a, pcp_axis, pcp_size)

        full_reorder = _derive_pcp_rank_major_reorder_indices(
            query_start_loc_,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            local_padded_num_tokens=local_qkv.shape[0],
            seq_lens=seq_lens_,
        )
        valid_mask = full_reorder >= 0
        scatter_indices = jnp.where(valid_mask, full_reorder,
                                    full_reorder.size)
        gather_indices = jnp.where(valid_mask, full_reorder, 0)

        qkv_shard = jnp.zeros_like(packed_qkv_shard).at[scatter_indices].set(
            packed_qkv_shard, mode="drop")
        b_shard = jnp.zeros_like(packed_b_shard).at[scatter_indices].set(
            packed_b_shard, mode="drop")
        a_shard = jnp.zeros_like(packed_a_shard).at[scatter_indices].set(
            packed_a_shard, mode="drop")

        conv_weight_interleaved = reorder_concatenated_tensor_for_sharding(
            conv_weight_,
            [local_key_dim, local_key_dim, local_value_dim],
            pcp_size,
            0,
        )
        weight_shard = _select_replicated_shard_for_pcp_rank(
            conv_weight_interleaved, pcp_axis, pcp_size, axis=0)
        if conv_bias_ is None:
            bias_shard = None
        else:
            conv_bias_interleaved = reorder_concatenated_tensor_for_sharding(
                conv_bias_,
                [local_key_dim, local_key_dim, local_value_dim],
                pcp_size,
                0,
            )
            bias_shard = _select_replicated_shard_for_pcp_rank(
                conv_bias_interleaved, pcp_axis, pcp_size, axis=0)
        A_shard = _select_replicated_shard_for_pcp_rank(A_log_,
                                                        pcp_axis,
                                                        pcp_size,
                                                        axis=0)
        dt_shard = _select_replicated_shard_for_pcp_rank(dt_bias_,
                                                         pcp_axis,
                                                         pcp_size,
                                                         axis=0)

        conv_state_interleaved = reorder_concatenated_tensor_for_sharding(
            conv_state_,
            [local_key_dim, local_key_dim, local_value_dim],
            pcp_size,
            -1,
        )
        conv_state_shard = _select_replicated_shard_for_pcp_rank(
            conv_state_interleaved, pcp_axis, pcp_size, axis=-1)
        recurrent_state_shard = _select_replicated_shard_for_pcp_rank(
            recurrent_state_, pcp_axis, pcp_size, axis=1)

        (new_conv_shard,
         new_rec_shard), seq_output_shard = (run_jax_gdn_attention_local(
             qkv_shard,
             b_shard,
             a_shard,
             conv_state_shard,
             recurrent_state_shard,
             weight_shard,
             bias_shard,
             A_shard,
             dt_shard,
             query_start_loc_,
             state_indices_,
             distribution_,
             seq_lens_,
             n_kq=local_n_kq,
             n_v=local_n_v,
             d_k=d_k,
             d_v=d_v,
             kernel_size=kernel_size,
             dp_enabled=False,
             config=config,
         ))

        seq_output = jax.lax.all_gather(seq_output_shard,
                                        axis_name=pcp_axis,
                                        axis=-1,
                                        tiled=True)
        new_conv_gathered = jax.lax.all_gather(new_conv_shard,
                                               axis_name=pcp_axis,
                                               axis=-1,
                                               tiled=True)
        new_rec = jax.lax.all_gather(new_rec_shard,
                                     axis_name=pcp_axis,
                                     axis=1,
                                     tiled=True)
        new_conv = inverse_reorder_for_sharding(
            new_conv_gathered,
            [local_key_dim, local_key_dim, local_value_dim],
            pcp_size,
            -1,
        )

        packed_output = seq_output[gather_indices]
        packed_output = jnp.where(valid_mask[:, None], packed_output, 0.0)
        return (new_conv, new_rec), packed_output

    mapped_fn = jax.shard_map(
        _pcp_prefill_fn,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )

    (new_conv_state, new_recurrent_state), output = mapped_fn(
        j_mixed_qkv,
        j_b,
        j_a,
        conv_state,
        recurrent_state,
        j_conv_weight,
        j_conv_bias,
        j_A_log,
        j_dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        seq_lens,
    )
    return (new_conv_state, new_recurrent_state), output
