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
import functools
import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.experimental.layout import Layout, with_layout_constraint
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.gdn_pool_layout import derive_pooled_gdn_state_layout
from vllm_torchtpu.kernels import pool_adapters
from vllm_torchtpu.kernels.gdn.v3 import wrapper as gdn_v3_wrapper
from vllm_torchtpu.layers.common.utils import \
    reorder_concatenated_tensor_for_sharding
from vllm_torchtpu.utils import get_mesh_shape_product


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
    slot_read_offsets: Optional[jnp.ndarray] = None,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    mesh: jax.sharding.Mesh,
    num_spec_tokens: int = 0,
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

    Returns:
        A tuple containing:
        - A tuple of (new_conv_state, new_recurrent_state).
          - new_conv_state: `(num_blocks, kernel_size - 1, dim)`
          - new_recurrent_state: `(num_blocks, n_v, d_k, d_v)`
        - The output tensor of shape `(num_tokens, n_v * d_v)`.
    """
    in_specs = (
        P(None, "model"),  # j_mixed_qkv
        P(None, "model"),  # j_b
        P(None, "model"),  # j_a
        P(None, None, "model"),  # conv_state
        P(None, "model", None, None),  # recurrent_state
        P("model", None, None),  # j_conv_weight
        P("model") if j_conv_bias is not None else None,  # j_conv_bias
        P("model"),  # j_A_log
        P("model"),  # j_dt_bias
        P(None),  # query_start_loc
        P(None),  # state_indices
        P(None),  # distribution
        P(None),  # seq_lens
    )
    if slot_read_offsets is not None:
        # Per-slot buffer, replicated like the other per-sequence inputs
        # (query_start_loc / state_indices / seq_lens).
        in_specs = in_specs + (P(None), )

    out_specs = (
        (
            P(None, None, "model"),  # new_conv_state
            P(None, "model", None, None),  # new_recurrent_state
        ),
        P(None, "model"),  # output
    )

    tp_size = get_mesh_shape_product(mesh, "model")

    p_run_jax_gdn_attention_local = functools.partial(
        gdn_v3_wrapper.fused_conv1d_gdn,
        n_kq=n_kq // tp_size,
        n_v=n_v // tp_size,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        num_spec_tokens=num_spec_tokens,
    )

    mapped_fn = jax.shard_map(
        p_run_jax_gdn_attention_local,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )

    extra_args = (() if slot_read_offsets is None else (slot_read_offsets, ))
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
        *extra_args,
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


def _derive_pcp_ragged_exchange_descriptors(
    reorder_indices: jnp.ndarray,
    *,
    pcp_size: int,
    interleave_size: int,
    local_padded_num_tokens: int,
    max_num_requests: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compress rank-major token mappings into contiguous exchange runs."""
    rank_major = reorder_indices.reshape(pcp_size, local_padded_num_tokens)
    valid = rank_major >= 0
    previous = jnp.concatenate(
        (jnp.zeros((pcp_size, 1), dtype=jnp.int32), rank_major[:, :-1]),
        axis=1,
    )
    previous_valid = jnp.concatenate(
        (jnp.zeros((pcp_size, 1), dtype=jnp.bool_), valid[:, :-1]),
        axis=1,
    )
    run_starts = jnp.logical_and(
        valid,
        jnp.logical_or(jnp.logical_not(previous_valid), rank_major
                       != previous + 1),
    )
    run_ids = (jnp.cumsum(run_starts.astype(jnp.int32), axis=1) - 1)

    # For one request and rank, every run is an interleave chunk. All chunks
    # except the first and last have I tokens, so their count is at most
    # ceil(local_tokens / I) + 1. Summing that per-request bound and using
    # sum(ceil(n_r / I)) <= ceil(sum(n_r) / I) + R - 1 gives the looser static
    # bound ceil(local_padded_tokens / I) + 2R used below.
    max_runs = (
        (local_padded_num_tokens + interleave_size - 1) // interleave_size +
        2 * max_num_requests)
    local_positions = jnp.arange(local_padded_num_tokens, dtype=jnp.int32)
    drop_index = jnp.asarray(max_runs, dtype=jnp.int32)

    def _summarize_rank(reorder_row, valid_row, run_start_row, run_id_row):
        start_slots = jnp.where(run_start_row, run_id_row, drop_index)
        token_slots = jnp.where(valid_row, run_id_row, drop_index)
        input_starts = jnp.zeros(
            (max_runs, ), dtype=jnp.int32).at[start_slots].set(local_positions,
                                                               mode="drop")
        output_starts = jnp.zeros(
            (max_runs, ), dtype=jnp.int32).at[start_slots].set(reorder_row,
                                                               mode="drop")
        sizes = jnp.zeros(
            (max_runs, ),
            dtype=jnp.int32).at[token_slots].add(valid_row.astype(jnp.int32),
                                                 mode="drop")
        return input_starts, sizes, output_starts

    return jax.vmap(_summarize_rank)(rank_major, valid, run_starts, run_ids)


def _validate_pcp_ragged_exchange_layout_support() -> None:
    generation = pltpu.get_tpu_info().generation
    # TODO: Validate this explicit tiled layout on TPU v4/v5e/v6 and relax
    # this guard once compile and numerical coverage exists for those targets.
    if generation != 7:
        raise NotImplementedError(
            "GDN PCP ragged exchange layout is only validated on TPU "
            f"generation 7, got generation {generation}.")


def _ragged_exchange_pcp_token_shards_for_head_shards(
    tensor: jnp.ndarray,
    pcp_axis: str,
    pcp_size: int,
    local_descriptors: jnp.ndarray,
    send_sizes_by_rank: jnp.ndarray,
) -> jnp.ndarray:
    """Exchange contiguous token runs directly into request-major order."""
    _validate_pcp_ragged_exchange_layout_support()
    if tensor.ndim != 2:
        raise ValueError("PCP ragged exchange expects a rank-2 tensor.")
    if tensor.shape[1] % pcp_size != 0:
        raise ValueError("PCP ragged exchange requires a sharded last dim.")

    local_padded_num_tokens = tensor.shape[0]
    shard_width = tensor.shape[1] // pcp_size
    if shard_width % 2 != 0:
        raise ValueError("PCP ragged exchange requires an even shard width.")
    max_runs = local_descriptors.shape[0]

    # Match the TPU ragged A2A payload tile before transposing destinations;
    # otherwise XLA materializes a costly [P, L, W] -> [P * L, 2, W / 2] repack.
    tiled_shard_shape = (2, shard_width // 2)
    tensor_by_destination = tensor.reshape(local_padded_num_tokens, pcp_size,
                                           *tiled_shard_shape)
    operand = jnp.transpose(tensor_by_destination, (1, 0, 2, 3)).reshape(
        pcp_size * local_padded_num_tokens, *tiled_shard_shape)

    local_input_starts = local_descriptors[:, 0]
    local_send_sizes = local_descriptors[:, 1]
    local_output_starts = local_descriptors[:, 2]
    destination_bases = (jnp.arange(pcp_size, dtype=jnp.int32) *
                         local_padded_num_tokens)
    input_offsets = (destination_bases[:, None] +
                     local_input_starts[None, :]).reshape(pcp_size * max_runs)
    send_sizes = jnp.broadcast_to(local_send_sizes[None, :],
                                  (pcp_size, max_runs)).reshape(-1)
    output_offsets = jnp.broadcast_to(local_output_starts[None, :],
                                      (pcp_size, max_runs)).reshape(-1)
    recv_sizes = send_sizes_by_rank.reshape(-1)
    output = jnp.zeros(
        (pcp_size * local_padded_num_tokens, *tiled_shard_shape),
        dtype=tensor.dtype)
    exchanged = jax.lax.ragged_all_to_all(
        operand,
        output,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        axis_name=pcp_axis,
    )
    exchanged = exchanged.reshape(pcp_size * local_padded_num_tokens,
                                  shard_width)
    # Keep the BF16 result in the layout expected by the GDN FP32 conversion.
    return with_layout_constraint(
        exchanged,
        Layout((0, 1), tiling=((8, 128), (2, 1))),
    )


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
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """GDN PCP prefill using an op-local PCP mesh.

    The enclosing vLLM worker is already TP-local. This function only adds PCP
    as extra head parallelism inside the custom op, without reintroducing a
    global JAX ``pcp`` axis.

    The packed-rank-major token order is reconstructed from replicated metadata
    inside this op so callers do not need to pass a GDN-specific reorder tensor.
    The cache state is PCP-local at the torch boundary. The PCP custom-op
    adapter represents that as a logical full-width JAX array sharded over this
    op-local mesh, so the shard_map body receives only the rank-local state.
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
    shard_key_dim = local_n_kq * d_k
    shard_value_dim = local_n_v * d_v
    local_conv_state_dim = 2 * shard_key_dim + shard_value_dim

    token_spec = P(pcp_axis, None)
    replicated_spec = P()
    conv_state_spec = P(None, None, pcp_axis)
    recurrent_state_spec = P(None, pcp_axis, None, None)
    in_specs = (
        token_spec,  # j_mixed_qkv
        token_spec,  # j_b
        token_spec,  # j_a
        conv_state_spec,  # conv_state
        recurrent_state_spec,  # recurrent_state
        replicated_spec,  # j_conv_weight
        replicated_spec if j_conv_bias is not None else None,  # j_conv_bias
        replicated_spec,  # j_A_log
        replicated_spec,  # j_dt_bias
        replicated_spec,  # query_start_loc
        replicated_spec,  # state_indices
        replicated_spec,  # distribution
        replicated_spec,  # seq_lens
    )
    output_spec = P(pcp_axis, None, None)
    out_specs = ((conv_state_spec, recurrent_state_spec), output_spec)

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
        local_ba = jnp.stack((local_b, local_a),
                             axis=-1).reshape(local_b.shape[0], -1)
        packed_ba_shard = _exchange_pcp_token_shards_for_head_shards(
            local_ba, pcp_axis, pcp_size)

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
        gather_indices = jnp.where(valid_mask, full_reorder, full_reorder.size)

        (input_starts_by_rank, send_sizes_by_rank,
         output_starts_by_rank) = _derive_pcp_ragged_exchange_descriptors(
             full_reorder,
             pcp_size=pcp_size,
             interleave_size=interleave_size,
             local_padded_num_tokens=local_qkv.shape[0],
             max_num_requests=query_start_loc_.shape[0] - 1,
         )
        descriptors_by_rank = jnp.stack(
            (input_starts_by_rank, send_sizes_by_rank, output_starts_by_rank),
            axis=-1,
        )
        local_descriptors = _select_replicated_shard_for_pcp_rank(
            descriptors_by_rank, pcp_axis, pcp_size, axis=0)[0]

        qkv_shard = _ragged_exchange_pcp_token_shards_for_head_shards(
            interleaved_qkv,
            pcp_axis,
            pcp_size,
            local_descriptors,
            send_sizes_by_rank,
        )
        ba_shard = jnp.zeros_like(packed_ba_shard).at[scatter_indices].set(
            packed_ba_shard, mode="drop")
        ba_shard = ba_shard.reshape(ba_shard.shape[0], local_n_v, 2)
        b_shard = ba_shard[:, :, 0]
        a_shard = ba_shard[:, :, 1]

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

        if conv_state_.shape[-1] != local_conv_state_dim:
            raise ValueError("GDN PCP conv_state must be PCP-local: "
                             f"expected last dim {local_conv_state_dim}, "
                             f"got {conv_state_.shape[-1]}.")
        if recurrent_state_.shape[1] != local_n_v:
            raise ValueError("GDN PCP recurrent_state must be PCP-local: "
                             f"expected head dim {local_n_v}, "
                             f"got {recurrent_state_.shape[1]}.")

        (new_conv_shard,
         new_rec_shard), seq_output_shard = gdn_v3_wrapper.fused_conv1d_gdn(
             qkv_shard,
             b_shard,
             a_shard,
             conv_state_,
             recurrent_state_,
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
         )

        seq_output_shard = seq_output_shard.reshape(seq_output_shard.shape[0],
                                                    local_n_v, d_v)
        seq_output_shard = jnp.concatenate(
            (seq_output_shard,
             jnp.zeros((1, local_n_v, d_v), dtype=seq_output_shard.dtype)),
            axis=0,
        )
        packed_output_shard = seq_output_shard[gather_indices]
        packed_output = jax.lax.all_to_all(packed_output_shard,
                                           axis_name=pcp_axis,
                                           split_axis=0,
                                           concat_axis=1,
                                           tiled=True)
        return (new_conv_shard, new_rec_shard), packed_output

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


# ---------------------------------------------------------------
# Unified block pool variants: the mamba state crosses the op boundary
# once as the attention-shaped pool (ssm | conv slot | pad token-ranges
# inside each block); the state is gathered/scattered through the pool
# adapters around stock kernels. The split-state functions above serve
# the non-pooled (disagg / kv-transfer) layout.
# ---------------------------------------------------------------


def run_jax_gdn_attention_pooled_local(
    mixed_qkv: jnp.ndarray,
    b: jnp.ndarray,
    a: jnp.ndarray,
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
    pool_block_tokens: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """GDN attention over the unified block pool.

    ``recurrent_state`` is the attention-shaped pool holding the f32 ssm
    region at token offset 0 and the conv region in the tokens right after
    it. All pool knowledge lives here: for the chunked impls, conv state
    is gathered/scattered through `pool_adapters` around the stock conv
    kernel and the ssm region is threaded into the stock delta-rule via
    pluggable `StateOps`; the fused V3 kernel instead streams both regions
    in place through a `v3_state_source` copy-plan — the model kernels'
    math is unmodified either way.
    """

    # Backends with a fixed kernel block (batched RPA) get the pool born
    # at kernel granularity; a manager block is `split` consecutive kernel
    # blocks (upstream map_to_kernel_blocks). State regions address
    # manager-block token ranges; the adapters route them (split kwarg) —
    # the pool itself is never reshaped (an XLA reshape of the pool
    # materializes as a pool-sized relayout copy per step).
    assert pool_block_tokens % recurrent_state.shape[1] == 0, (
        pool_block_tokens, recurrent_state.shape)
    split = pool_block_tokens // recurrent_state.shape[1]

    conv_dim = conv_weight.shape[0]
    per_tok_elems = math.prod(recurrent_state.shape[2:])
    tok_bytes = per_tok_elems * jnp.dtype(recurrent_state.dtype).itemsize
    state_layout = derive_pooled_gdn_state_layout(
        ssm_bytes=n_v * d_k * d_v * 4,
        conv_bytes=(kernel_size - 1) * conv_dim * 2,
        token_bytes=tok_bytes,
    )
    ssm_ntok = state_layout.ssm_tokens
    assert ssm_ntok <= pool_block_tokens, (
        "ssm state does not fit the attention page", ssm_ntok,
        pool_block_tokens)

    # The conv slot occupies whole tokens right after the ssm region,
    # padded up so the slot's token range satisfies the tok0 % ntok == 0
    # layout rule; the pad tokens are dead bytes inside the slot.
    conv_ntok = state_layout.conv_tokens
    assert state_layout.required_tokens <= pool_block_tokens, (
        "mamba slot exceeds the attention page", ssm_ntok, conv_ntok,
        pool_block_tokens)

    # Fused conv+GDN kernel: both state regions stream directly
    # between the pool and the kernel's double-buffered pipeline (one
    # contiguous DMA per slot per region) — no external gather/scatter
    # round trip. The kernel masks fresh slots via has_initial_state,
    # so a newly-allocated block's bytes are never read, and padded
    # slots move no bytes in either direction.
    plan = pool_adapters.v3_state_source(
        recurrent_state,
        split=split,
        ssm_ntok=ssm_ntok,
        conv_tok0=ssm_ntok,
        conv_ntok=conv_ntok,
        conv_dim=conv_dim,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        qk_pair_layout=tpu_envs.TPU_GDN_CONV_QK_PAIR_LAYOUT,
    )
    recurrent_state, output = gdn_v3_wrapper.fused_conv1d_gdn(
        mixed_qkv,
        b,
        a,
        None,
        None,
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
        state_source=recurrent_state,
        state_plan=plan,
    )
    return recurrent_state, output


def run_jax_gdn_attention_pooled(
    j_mixed_qkv: jnp.ndarray,
    j_b: jnp.ndarray,
    j_a: jnp.ndarray,
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
    pool_block_tokens: int,
    mesh: jax.sharding.Mesh,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Runs GDN attention over the unified block pool, sharded on the mesh.

    Args:
        j_mixed_qkv: Input tensor of shape `(num_tokens, dim)`.
        j_b: Input tensor of shape `(num_tokens, n_v)`.
        j_a: Input tensor of shape `(num_tokens, n_v)`.
        recurrent_state: The attention-shaped pool
          `(num_blocks, block_size, num_kv_heads * 2, head_size)` in the KV
          dtype, carrying the f32 ssm region and the bf16 conv slot as
          token-ranges of each state block. Block 0 is the null block, only
          used for padded / invalid tokens.
        j_conv_weight: Convolutional weight tensor of shape `(dim, 1,
          kernel_size)`.
        j_conv_bias: Optional convolutional bias tensor of shape `(dim,)`.
        j_A_log: Log of A parameter tensor of shape `(n_v,)`.
        j_dt_bias: Delta T bias tensor of shape `(n_v,)`.
        state_indices: Tensor of shape `(max_reqs,)` mapping request index to
          state block index.
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

    Returns:
        A tuple containing:
        - The updated pool (the in-place written state regions).
        - The output tensor of shape `(num_tokens, n_v * d_v)`.
    """
    pool_spec = P(None, None, None, None)  # attention-shaped pool
    in_specs = (
        P(None, "model"),  # j_mixed_qkv
        P(None, "model"),  # j_b
        P(None, "model"),  # j_a
        pool_spec,  # recurrent_state (attention-shaped pool)
        P("model", None, None),  # j_conv_weight
        P("model") if j_conv_bias is not None else None,  # j_conv_bias
        P("model"),  # j_A_log
        P("model"),  # j_dt_bias
        P(None),  # query_start_loc
        P(None),  # state_indices
        P(None),  # distribution
        P(None),  # seq_lens
    )

    out_specs = (
        pool_spec,  # new_recurrent_state (attention-shaped pool)
        P(None, "model"),  # output
    )

    tp_size = get_mesh_shape_product(mesh, "model")

    p_run_jax_gdn_attention_pooled_local = functools.partial(
        run_jax_gdn_attention_pooled_local,
        n_kq=n_kq // tp_size,
        n_v=n_v // tp_size,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        pool_block_tokens=pool_block_tokens,
    )
    mapped_fn = jax.shard_map(
        p_run_jax_gdn_attention_pooled_local,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )

    outputs = mapped_fn(
        j_mixed_qkv,
        j_b,
        j_a,
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

    return outputs


def run_jax_gdn_attention_pooled_pcp_prefill(
    j_mixed_qkv: jnp.ndarray,
    j_b: jnp.ndarray,
    j_a: jnp.ndarray,
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
    pool_block_tokens: int,
    pcp_size: int,
    interleave_size: int,
    mesh: jax.sharding.Mesh,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """GDN PCP prefill over the unified block pool.

    Pooled counterpart of ``run_jax_gdn_attention_pcp_tp_prefill``: the same
    token-shard -> head-shard exchange, but the state lives in the
    rank-local attention-shaped pool instead of dense state tensors.
    """
    pcp_axis = "pcp"
    if pcp_axis not in mesh.axis_names:
        raise NotImplementedError(
            "GDN pooled PCP prefill requires a pcp mesh axis.")
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
    # Rank-local pool: the op adapter presents the per-rank pools as one
    # logical array stacked on axis 0, so the shard_map body receives
    # exactly this rank's pool.
    pool_spec = P(pcp_axis)
    in_specs = (
        token_spec,  # j_mixed_qkv
        token_spec,  # j_b
        token_spec,  # j_a
        pool_spec,  # recurrent_state (attention-shaped pool)
        replicated_spec,  # j_conv_weight
        replicated_spec if j_conv_bias is not None else None,  # j_conv_bias
        replicated_spec,  # j_A_log
        replicated_spec,  # j_dt_bias
        replicated_spec,  # query_start_loc
        replicated_spec,  # state_indices
        replicated_spec,  # distribution
        replicated_spec,  # seq_lens
    )
    output_spec = P(pcp_axis, None, None)
    out_specs = (pool_spec, output_spec)

    def _pooled_pcp_prefill_fn(
        local_qkv,
        local_b,
        local_a,
        pool_,
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
        local_ba = jnp.stack((local_b, local_a),
                             axis=-1).reshape(local_b.shape[0], -1)
        packed_ba_shard = _exchange_pcp_token_shards_for_head_shards(
            local_ba, pcp_axis, pcp_size)

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
        gather_indices = jnp.where(valid_mask, full_reorder, full_reorder.size)

        (input_starts_by_rank, send_sizes_by_rank,
         output_starts_by_rank) = _derive_pcp_ragged_exchange_descriptors(
             full_reorder,
             pcp_size=pcp_size,
             interleave_size=interleave_size,
             local_padded_num_tokens=local_qkv.shape[0],
             max_num_requests=query_start_loc_.shape[0] - 1,
         )
        descriptors_by_rank = jnp.stack(
            (input_starts_by_rank, send_sizes_by_rank, output_starts_by_rank),
            axis=-1,
        )
        local_descriptors = _select_replicated_shard_for_pcp_rank(
            descriptors_by_rank, pcp_axis, pcp_size, axis=0)[0]

        qkv_shard = _ragged_exchange_pcp_token_shards_for_head_shards(
            interleaved_qkv,
            pcp_axis,
            pcp_size,
            local_descriptors,
            send_sizes_by_rank,
        )
        ba_shard = jnp.zeros_like(packed_ba_shard).at[scatter_indices].set(
            packed_ba_shard, mode="drop")
        ba_shard = ba_shard.reshape(ba_shard.shape[0], local_n_v, 2)
        b_shard = ba_shard[:, :, 0]
        a_shard = ba_shard[:, :, 1]

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

        new_pool, seq_output_shard = run_jax_gdn_attention_pooled_local(
            qkv_shard,
            b_shard,
            a_shard,
            pool_,
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
            pool_block_tokens=pool_block_tokens,
        )

        seq_output_shard = seq_output_shard.reshape(seq_output_shard.shape[0],
                                                    local_n_v, d_v)
        seq_output_shard = jnp.concatenate(
            (seq_output_shard,
             jnp.zeros((1, local_n_v, d_v), dtype=seq_output_shard.dtype)),
            axis=0,
        )
        packed_output_shard = seq_output_shard[gather_indices]
        packed_output = jax.lax.all_to_all(packed_output_shard,
                                           axis_name=pcp_axis,
                                           split_axis=0,
                                           concat_axis=1,
                                           tiled=True)
        return new_pool, packed_output

    mapped_fn = jax.shard_map(
        _pooled_pcp_prefill_fn,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )

    return mapped_fn(
        j_mixed_qkv,
        j_b,
        j_a,
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
