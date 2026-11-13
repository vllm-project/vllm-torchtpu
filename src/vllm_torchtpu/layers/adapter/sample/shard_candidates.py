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
"""Exact non-greedy spec verify from each rank's LM-head vocab shard
(VLLM_TPU_VERIFY_SHARD_CANDIDATES).

Bit-for-bit the result of the fused verify (`tpu_runner._verify_fused_impl`)
with top-k candidates (TPU_SAMPLER_TOPK_CANDIDATES) and candidates-only target
rows (VLLM_TPU_SAMPLER_CANDIDATES_ONLY), on the same uniforms, without
all-gathering the [rows, vocab] logits. Every value that reaches the output
is a function of:

- the 128-wide block maxima: built from per-shard sub-block maxima (a shard
  holds whole sub-blocks of gcd(shard width, 128) logits; at vocab 248,320
  over 8 ranks the shard width 31,040 splits every other block), all-gathered
  and max-reduced (max is exact). The same `select_candidate_blocks` call on
  the same [rows, blocks] values picks the same blocks in the same order.
- the logits of the picked blocks (plus each target row's first-argmax
  block): every rank reads the columns it owns, the ranks all-gather those
  reads and each column takes its owner's copy (a copy, no arithmetic).
- the uniforms at those columns, and for the bonus rows, at the rank's own
  vocab slice: the bonus sample masks, adds noise to and argmaxes the FULL
  row (`apply_top_k_top_p_candidates` keeps any logit at the cutoff, and
  padded rows sample unmasked), so each rank runs those elementwise ops on
  its slice, and the ranks exchange (max, first id) pairs; the first rank
  holding the row max has the first full-row argmax.

The target rows' candidate arithmetic (cutoff, softmax, accept test, recovery)
is `_candidate_target_dist`'s, through the same helpers.
"""

import math
from collections.abc import Callable
from typing import NamedTuple

import torch

from vllm_torchtpu.layers.adapter.sample.rejection_sampler import (
    candidate_dist_from_candidates,
    candidate_target_scale,
    rejection_from_target_dist,
)
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import (
    CANDIDATE_BLOCK,
    candidate_cutoff,
    mask_below_cutoff,
    masked_candidate_logits,
    select_candidate_blocks,
)

# tpu_runner's greedy threshold for the bonus sample (`sample_from_logits`).
SAMPLING_EPS = 1e-5

AllGather = Callable[[torch.Tensor, int], torch.Tensor]


class ShardCandidateLogits(NamedTuple):
    """This rank's [rows, vocab shard] logits, standing in for the full
    logits of a verify chunk on VLLM_TPU_VERIFY_SHARD_CANDIDATES."""

    local: torch.Tensor


def tp_all_gather(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Concatenate every TP rank's `x` along `dim`, in rank order."""
    from vllm.distributed import tensor_model_parallel_all_gather

    return tensor_model_parallel_all_gather(x, dim=dim)


def sub_block_width(shard_width: int) -> int:
    """Largest sub-block of a 128-block that no shard boundary splits."""
    return math.gcd(shard_width, CANDIDATE_BLOCK)


def _bonus_scale(x: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    """tpu_runner `_apply_temperature_impl`."""
    safe_temperatures = torch.where(temperatures == 0.0, 1.0, temperatures)
    return x / safe_temperatures


def _gather_blocks(
    rows: torch.Tensor,
    block_ids: torch.Tensor,
    start: int,
    tp_size: int,
    all_gather: AllGather,
) -> torch.Tensor:
    """The logits [R, n * 128] of blocks `block_ids` [R, n] of each full row,
    from this rank's columns [start, start + width) of the rows [R, width]."""
    num_rows, width = rows.shape
    lane = torch.arange(CANDIDATE_BLOCK, device=rows.device, dtype=block_ids.dtype)
    cols = (block_ids.unsqueeze(-1) * CANDIDATE_BLOCK + lane).reshape(num_rows, -1)
    local = rows.gather(1, (cols - start).clamp(0, width - 1))
    every = all_gather(local.unsqueeze(0), 0)  # [tp, R, n * 128]
    owner = torch.div(cols, width, rounding_mode="floor").clamp(0, tp_size - 1)
    return every.gather(0, owner.unsqueeze(0)).squeeze(0)


def _first_argmax_over_ranks(
    values: torch.Tensor, start: int, tp_size: int, all_gather: AllGather
) -> torch.Tensor:
    """`torch.argmax(full_row, dim=-1)` from this rank's slice [B, width] of
    each row: (max, first id) pairs exchanged, first rank with the max wins.
    Ids travel as float32 (exact below 2**24)."""
    idx = torch.argmax(values, dim=-1, keepdim=True)
    best = values.gather(1, idx).to(torch.float32)
    pair = torch.cat([best, (idx + start).to(torch.float32)], dim=-1)
    every = all_gather(pair, -1).view(values.shape[0], tp_size, 2)
    rank = torch.argmax(every[:, :, 0], dim=-1, keepdim=True)
    return every[:, :, 1].gather(1, rank).view(-1).to(torch.int64)


def verify_shard_candidates(
    local_logits: torch.Tensor,
    bonus_logits_indices: torch.Tensor,
    target_logits_indices: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    bonus_u: torch.Tensor,
    accept_u: torch.Tensor,
    recover_u: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_lengths: torch.Tensor,
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    synthetic_conditional_rates: torch.Tensor | None,
    max_draft_tokens: int,
    synthetic_mode: bool,
    topk_candidates: int,
    tp_rank: int,
    tp_size: int,
    all_gather: AllGather | None = None,
) -> torch.Tensor:
    """Per-rank body: `_verify_fused_impl(full_logits, ..., topk_candidates,
    candidates_only=True)`'s [num_reqs, 1 + K] tokens, from this rank's
    contiguous vocab slice `local_logits` [N, V / tp_size] (rank r owns
    columns [r * width, (r + 1) * width)). bonus_u [B, V] and recover_u
    [T, V] are the full-vocab uniforms. Three small TP all-gathers."""
    gather = tp_all_gather if all_gather is None else all_gather
    num_blocks = topk_candidates
    bonus_local = torch.index_select(local_logits, 0, bonus_logits_indices)
    target_local = torch.index_select(local_logits, 0, target_logits_indices)
    num_bonus = bonus_local.shape[0]
    rows = torch.cat([bonus_local, target_local], dim=0)
    num_rows, width = rows.shape
    vocab = width * tp_size
    start = tp_rank * width

    # Block maxima of the full rows, raw logits [R, V / 128].
    sub = sub_block_width(width)
    sub_max = rows.reshape(num_rows, width // sub, sub).amax(dim=-1)
    block_max = (
        gather(sub_max, -1)
        .reshape(num_rows, vocab // CANDIDATE_BLOCK, CANDIDATE_BLOCK // sub)
        .amax(dim=-1)
    )

    # Bonus rows: `_sample_from_logits_impl` -> `apply_top_k_top_p_candidates`
    # (scale is monotone, so scale(block max) is the scaled rows' block max).
    bonus_ids = select_candidate_blocks(
        _bonus_scale(block_max[:num_bonus], temperatures), num_blocks
    )
    # Target rows: `_candidate_target_dist`.
    target_temperatures = temperatures[segment_ids]
    target_top_k = top_k[segment_ids]
    target_top_p = top_p[segment_ids]
    target_scale = candidate_target_scale(target_temperatures)
    target_ids = select_candidate_blocks(
        target_scale(block_max[num_bonus:]), num_blocks
    )
    # `blocked_argmax`'s block: the first block holding the row max.
    target_argmax_block = torch.argmax(block_max[num_bonus:], dim=-1, keepdim=True)
    picked = torch.cat(
        [
            torch.cat([bonus_ids, bonus_ids[:, :1]], dim=1),
            torch.cat([target_ids, target_argmax_block], dim=1),
        ],
        dim=0,
    )
    picked_logits = _gather_blocks(rows, picked, start, tp_size, gather)
    span = num_blocks * CANDIDATE_BLOCK

    # Bonus token: cutoff from the candidates, then the full row's mask,
    # noise and argmax on this rank's slice.
    scaled_local = _bonus_scale(bonus_local, temperatures)
    cutoff = candidate_cutoff(
        _bonus_scale(picked_logits[:num_bonus, :span], temperatures), top_k, top_p
    )
    masked_local = mask_below_cutoff(scaled_local, cutoff, top_k)
    u_local = bonus_u[:, start : start + width]
    u_clamped = torch.clamp(
        u_local,
        min=torch.finfo(u_local.dtype).tiny,
        max=1.0 - torch.finfo(u_local.dtype).eps,
    )
    gumbel_noise = -torch.log(-torch.log(u_clamped))
    noisy_local = masked_local + gumbel_noise
    is_greedy = temperatures <= SAMPLING_EPS
    final_local = torch.where(is_greedy, bonus_local, noisy_local)
    bonus_token_ids = _first_argmax_over_ranks(final_local, start, tp_size, gather)

    # Target rows on the candidates.
    target_picked = picked_logits[num_bonus:]
    argmax_block = target_picked[:, span:]
    target_logits_argmax = (
        target_argmax_block.view(-1) * CANDIDATE_BLOCK
        + torch.argmax(argmax_block, dim=-1)
    ).to(torch.int32)
    ids, masked_logits = masked_candidate_logits(
        target_ids,
        target_scale(target_picked[:, :span]),
        target_top_k,
        target_top_p,
    )
    draft_token_ids = draft_token_ids.to(torch.int64)
    dist = candidate_dist_from_candidates(
        draft_token_ids, target_logits_argmax, ids, masked_logits, recover_u
    )
    return rejection_from_target_dist(
        draft_token_ids,
        draft_lengths,
        bonus_token_ids,
        segment_ids,
        group_indices,
        max_draft_tokens,
        target_temperatures,
        accept_u,
        synthetic_conditional_rates,
        synthetic_mode,
        dist,
    )
