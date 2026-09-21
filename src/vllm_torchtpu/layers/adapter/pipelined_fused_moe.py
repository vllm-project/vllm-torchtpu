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
"""Pipelined Fused MoE wrapper with collective communication chunking.

This module provides `pipelined_fused_moe_gmm`, which acts as a wrapper around
`fused_moe_gmm` (or a custom MoE kernel) to manage and overlap DP or PCP
collective communications (AllGather and ReduceScatter) with MoE computation.
"""

import torch
from vllm.distributed.parallel_state import get_dp_group

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.pcp import get_pcp_group
from vllm_torchtpu.layers.adapter.fused_moe import fused_moe_gmm
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def enable_pipelined_collective_and_compute() -> bool:
    """Check whether collective communication and compute pipelining is enabled.

    Returns:
        bool: True if `envs.TPU_MOE_COLLECTION_CHUNK_SIZE > 0`, False otherwise.

    """
    return envs.TPU_MOE_COLLECTION_CHUNK_SIZE > 0


# Alias for alternative naming convention
enable_pipeline_collective_and_compute = enable_pipelined_collective_and_compute


def calculate_moe_chunks(seq_len: int, parallel_size: int,
                         chunk_size: int) -> tuple[int, int]:
    """Calculate the number of chunks and local chunk size for MoE chunk pipelining.

    Uses ceiling division so `chunk_size` acts as a maximum global chunk threshold:
        num_chunks = ceil((seq_len * parallel_size) / chunk_size)
        chunk_size_local = seq_len // num_chunks

    Args:
        seq_len: Number of local tokens on this rank (S).
        parallel_size: World size of the active DP or PCP collective group.
        chunk_size: Maximum post-gather chunk size threshold in tokens (C).

    Returns:
        tuple[int, int]: (num_chunks N_chunk, chunk_size_local S_chunk).

    Raises:
        ValueError: If local sequence length is not divisible by chunk count.
    """
    if chunk_size <= 0 or parallel_size <= 1 or seq_len <= 0:
        return 1, seq_len

    total_tokens = seq_len * parallel_size
    num_chunks = (total_tokens + chunk_size - 1) // chunk_size
    if seq_len % num_chunks != 0:
        raise ValueError(
            f"Local sequence length ({seq_len}) is not divisible by chunk count ({num_chunks})."
        )

    chunk_size_local = seq_len // num_chunks
    return num_chunks, chunk_size_local


def _get_moe_collective_group():
    """Select the collective group whose expert shards the kernel combines."""
    pcp_group = get_pcp_group()
    dp_group = get_dp_group()
    pcp_size = int(pcp_group.world_size) if pcp_group is not None else 1
    dp_size = int(dp_group.world_size) if dp_group is not None else 1

    if pcp_size > 1 and dp_size > 1:
        raise NotImplementedError(
            "MoE collective chunking does not yet support simultaneous "
            f"PCP ({pcp_size}) and DP ({dp_size}).")
    if pcp_size > 1:
        return pcp_group, "PCP"
    return dp_group, "DP"


@torch.compiler.assume_constant_result
def _log_active_chunk_pipeline() -> None:
    """Emit trace-time evidence without passing symbolic values to logging."""
    collective_group, collective_kind = _get_moe_collective_group()
    parallel_size = (int(collective_group.world_size)
                     if collective_group is not None else 1)
    logger.info_once(
        "MoE multi-chunk pipeline branch active: group=%s, world_size=%d, "
        "threshold=%d", collective_kind, parallel_size,
        envs.TPU_MOE_COLLECTION_CHUNK_SIZE)


def pipelined_fused_moe_gmm(
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
    activation: str,
    rhs_quant_dtype=None,
    skip_padded_tokens: bool | None = None,
) -> torch.Tensor:
    """Wrapper around fused_moe_gmm to pipeline DP/PCP collectives and compute.

    This function wraps `fused_moe_gmm` by taking control of the active DP or
    PCP collective communications (AllGather of activations/routing tensors
    and ReduceScatter of outputs) and staging them in chunks to overlap network
    communication with TPU matrix multiplication compute.

    When `enable_pipelined_collective_and_compute()` is True and the collective
    world size is greater than one, the sequence is partitioned into balanced
    chunks and executed via a 3-stage pipelined schedule:
      ag0_start -> ag1_start || moe0(ag0_done) -> rs0_start || moe1(ag1_done) -> rs1_start || rs0_done -> rs1_done -> concat

    Args:
        hidden_states: Local input activations [S, H].
        w1: First expert projection weights [E, 2*I, H] or [E, H, 2*I]
            (concatenated gate W1 and up W3 projections with intermediate dimension
            2*I for gated activations like SwiGLU when activation="silu").
        w2: Second expert projection weights [E, H, I] or [E, I, H]
            (down projection W2 from intermediate dimension I back to hidden dimension H).
        w1_scale: Optional first projection scale tensor.
        w2_scale: Optional second projection scale tensor.
        w1_bias: Optional first projection bias tensor.
        w2_bias: Optional second projection bias tensor.
        topk_weights: Routing top-k weights [S, topk].
        topk_ids: Routing top-k expert IDs [S, topk].
        experts_start: Optional 0-d tensor of starting expert ID for this shard.
        topk: Number of experts selected per token.
        activation: Gated activation function string (e.g., "silu" for SwiGLU).
        rhs_quant_dtype: Optional RHS quantization data type.
        skip_padded_tokens: Optional flag to skip padded tokens.

    Returns:
        torch.Tensor: Reduced local output activations [S, H].
    """
    chunk_size = envs.TPU_MOE_COLLECTION_CHUNK_SIZE
    collective_group, _ = _get_moe_collective_group()

    seq_len = hidden_states.shape[0]
    parallel_size = (int(collective_group.world_size)
                     if collective_group is not None else 1)
    num_chunks, chunk_size_local = calculate_moe_chunks(
        seq_len, parallel_size, chunk_size)

    if num_chunks == 1 or parallel_size == 1:
        if parallel_size > 1:
            ag_hidden_states = collective_group.all_gather(hidden_states,
                                                           dim=0)
            ag_topk_weights = (collective_group.all_gather(topk_weights, dim=0)
                               if topk_weights is not None else None)
            ag_topk_ids = (collective_group.all_gather(topk_ids, dim=0)
                           if topk_ids is not None else None)
        else:
            ag_hidden_states = hidden_states
            ag_topk_weights = topk_weights
            ag_topk_ids = topk_ids

        out = fused_moe_gmm(
            hidden_states=ag_hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            topk_weights=ag_topk_weights,
            topk_ids=ag_topk_ids,
            experts_start=experts_start,
            topk=topk,
            activation=activation,
            rhs_quant_dtype=rhs_quant_dtype,
            skip_padded_tokens=skip_padded_tokens,
        )

        if parallel_size > 1:
            return collective_group.reduce_scatter(out, dim=0)
        return out

    _log_active_chunk_pipeline()

    # Slice local inputs into num_chunks chunks along dim 0
    hs_slices = [
        hidden_states[i * chunk_size_local:(i + 1) * chunk_size_local]
        for i in range(num_chunks)
    ]
    weights_slices = [
        topk_weights[i * chunk_size_local:(i + 1) * chunk_size_local]
        for i in range(num_chunks)
    ] if topk_weights is not None else [None] * num_chunks
    ids_slices = [
        topk_ids[i * chunk_size_local:(i + 1) * chunk_size_local]
        for i in range(num_chunks)
    ] if topk_ids is not None else [None] * num_chunks

    # Prime pipeline with initial all-gather for chunk 0
    ag_hs_curr = collective_group.all_gather(hs_slices[0], dim=0)
    ag_tw_curr = (collective_group.all_gather(weights_slices[0], dim=0)
                  if weights_slices[0] is not None else None)
    ag_ti_curr = (collective_group.all_gather(ids_slices[0], dim=0)
                  if ids_slices[0] is not None else None)

    rs_outputs = []

    for i in range(num_chunks):
        # Issue asynchronous all-gather for next chunk if available
        if i + 1 < num_chunks:
            ag_hs_next = collective_group.all_gather(hs_slices[i + 1], dim=0)
            ag_tw_next = (collective_group.all_gather(weights_slices[i + 1],
                                                      dim=0)
                          if weights_slices[i + 1] is not None else None)
            ag_ti_next = (collective_group.all_gather(ids_slices[i + 1], dim=0)
                          if ids_slices[i + 1] is not None else None)

        # Compute MoE kernel for current gathered chunk
        out_i = fused_moe_gmm(
            hidden_states=ag_hs_curr,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            topk_weights=ag_tw_curr,
            topk_ids=ag_ti_curr,
            experts_start=experts_start,
            topk=topk,
            activation=activation,
            rhs_quant_dtype=rhs_quant_dtype,
            skip_padded_tokens=skip_padded_tokens,
        )

        # Reduce-scatter partial output for current chunk
        rs_i = collective_group.reduce_scatter(out_i, dim=0)
        rs_outputs.append(rs_i)

        if i + 1 < num_chunks:
            ag_hs_curr, ag_tw_curr, ag_ti_curr = ag_hs_next, ag_tw_next, ag_ti_next

    return torch.cat(rs_outputs, dim=0)
