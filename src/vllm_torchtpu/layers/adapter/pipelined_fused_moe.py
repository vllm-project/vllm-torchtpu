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
from vllm_torchtpu.layers.core.quantization import (FP8_ACT_BLOCK_SIZE,
                                                    dequantize_block_fp8,
                                                    dynamic_block_quantize_fp8)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Activation dtypes worth quantizing before the gather. fp8 halves what goes
# on the wire against either of these; anything already narrower has nothing
# to gain and would only be rounded twice.
_FP8_ALL_GATHER_ACT_DTYPES = (torch.bfloat16, torch.float16)


def enable_pipelined_collective_and_compute() -> bool:
    """Check whether collective communication and compute pipelining is enabled.

    Returns:
        bool: True if `envs.TPU_MOE_COLLECTION_CHUNK_SIZE > 0`, False otherwise.

    """
    return envs.TPU_MOE_COLLECTION_CHUNK_SIZE > 0


# Alias for alternative naming convention
enable_pipeline_collective_and_compute = enable_pipelined_collective_and_compute


def enable_fp8_all_gather() -> bool:
    """Whether MoE activations cross the DP/PCP gather quantized to fp8.

    This is a second, independent reason for a quantization method to claim
    the dispatch/combine collectives from vLLM: the gather is only ours to
    quantize if we are the ones issuing it. A method that turns this on must
    therefore report `supports_internal_mk` and route through
    `pipelined_fused_moe_gmm`, which degrades to a single chunk when chunk
    pipelining is off.
    """
    return envs.TPU_MOE_FP8_ALL_GATHER


def _fp8_all_gather_applies(act_dtype: torch.dtype,
                            enabled: bool | None = None) -> bool:
    """Whether this activation dtype is worth quantizing for the gather.

    `enabled` lets a caller hand over the decision it has already made; left
    out, the env is asked.
    """
    if enabled is None:
        enabled = enable_fp8_all_gather()
    return enabled and act_dtype in _FP8_ALL_GATHER_ACT_DTYPES


def _quantized_all_gather(collective_group,
                          hidden_states: torch.Tensor) -> torch.Tensor:
    """All-gather activations over the wire as fp8, restored on arrival.

    Only the collective sees the narrow form. The rows are rescaled back to
    their own dtype the moment they land, so everything downstream -- the
    permute, both grouped matmuls, the kernel's own activation quantization --
    runs exactly as it does without the flag. What the flag buys is one byte
    per element on the wire instead of two; what it costs is one rounding of
    the activations, plus a second collective for the scales.

    That rounding is why the scales are per `FP8_ACT_BLOCK_SIZE`-wide block of
    the hidden axis rather than per whole row. The gmm re-quantizes its LHS in
    blocks of the same width, and a block's amax survives its own quantization
    -- the element that set it maps to e4m3's maximum exactly -- so rounding on
    the same grid here leaves the kernel's own pass nothing left to do. What it
    feeds the MXU comes out bit-identical to what it would have computed from
    activations that never crossed the wire quantized: at a hidden size of 6144
    the flag costs no accuracy at all. Per-row scales move the values off that
    grid, and the kernel then rounds a second time for real.

    The grids line up only when the kernel's `tile_k` is a whole number of
    blocks. The kernel quantizes over `range(0, tile_k, 512)` within a tile
    (`gmm_v2._call_gmm_kernel`), not over the global k axis, so a `tile_k` of,
    say, 768 would split into 512 and 256 and the boundaries would drift. For
    the shapes this path runs on -- k=6144, n=2048 with `fuse_act` -- the tile
    search fits `tile_n` before it ever reduces `tile_k`, leaving `tile_k` at
    6144 for every vmem budget, so the grids do line up. Nothing enforces that,
    and a shape where they do not still computes the right answer; it just pays
    the second rounding, the way per-row scales already do.

    The scales stay small enough for this to be free on the wire -- twelve f32
    per row at 6144, against the 6144 bytes the row itself costs.

    A hidden size that is not a whole number of blocks is zero-padded, and that
    pad does cross the wire. 6144 is twelve whole blocks, so nothing is padded
    there; a size below one block is the case to watch, since it pays a full
    block either way.

    The rows travel bitcast to int8. They are the same eight bits either way,
    and int8 is a dtype every collective backend carries, while fp8 is not one
    they can be assumed to.
    """
    quantized, scale = dynamic_block_quantize_fp8(
        hidden_states, block_size=FP8_ACT_BLOCK_SIZE)
    ag_rows = collective_group.all_gather(quantized.view(torch.int8), dim=0)
    ag_scale = collective_group.all_gather(scale, dim=0)
    # Rescale in f32, not in the row's dtype: `scale` was computed there, and
    # a bf16 product would throw away most of what the fp8 rows still hold.
    return dequantize_block_fp8(ag_rows.view(torch.float8_e4m3fn), ag_scale,
                                hidden_states.dtype, hidden_states.shape[-1])


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
    fp8_all_gather: bool = False,
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
        fp8_all_gather: Row-quantize activations to fp8 for the gather and
            restore them on arrival, so only the wire is narrowed.

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

    # There is no gather to make cheaper on a single rank, so quantizing there
    # would only round the activations for nothing.
    quantize_ag = (parallel_size > 1 and _fp8_all_gather_applies(
        hidden_states.dtype, fp8_all_gather))

    if num_chunks == 1 or parallel_size == 1:
        if parallel_size > 1:
            if quantize_ag:
                ag_hidden_states = _quantized_all_gather(
                    collective_group, hidden_states)
            else:
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

    def _all_gather_hs(hs_slice):
        if quantize_ag:
            return _quantized_all_gather(collective_group, hs_slice)
        return collective_group.all_gather(hs_slice, dim=0)

    # Prime pipeline with initial all-gather for chunk 0
    ag_hs_curr = _all_gather_hs(hs_slices[0])
    ag_tw_curr = (collective_group.all_gather(weights_slices[0], dim=0)
                  if weights_slices[0] is not None else None)
    ag_ti_curr = (collective_group.all_gather(ids_slices[0], dim=0)
                  if ids_slices[0] is not None else None)

    rs_outputs = []

    for i in range(num_chunks):
        # Issue asynchronous all-gather for next chunk if available
        if i + 1 < num_chunks:
            ag_hs_next = _all_gather_hs(hs_slices[i + 1])
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
