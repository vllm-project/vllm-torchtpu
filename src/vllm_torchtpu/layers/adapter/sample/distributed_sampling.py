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
"""Device-side non-greedy sampling from tensor-parallel LM head shards.

For ``TP > 1``, each rank projects onto its own ``[B, V_shard]`` slice via
``LogitsProcessor(..., skip_gather=True)`` and only small per-row tensors cross
TP:

- ``TOPK_TOPP`` (``1 <= top_k <= MAX_DISTRIBUTED_TOPK``, any ``top_p``): each
  rank extracts the exact top ``CANDIDATES_PER_SHARD`` (= 128) ``(logit, id)``
  pairs via a two-stage block-max filter + ``torch.topk``, all-gathers them
  across TP, and runs ``apply_top_k_top_p`` + Gumbel-max on the merged set.
  Matches full-vocab ``apply_top_k_top_p`` unless more than
  ``CANDIDATES_PER_SHARD - top_k + 1`` tokens tie at the ``top_k``-th logit on
  a single shard.
- ``PURE_TEMP`` (no ``top_k``, ``top_p = 1.0``): each rank samples one token
  from its shard via two-level Gumbel-max and computes its shard ``logsumexp``;
  the winning shard is drawn with ``argmax(shard_lse + Gumbel)``. All ranks
  share the in-shard uniform columns because only the winning shard's draw is
  observed and the cross-shard draw is independent, keeping the marginal exact.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.utils.math_utils import cdiv

from vllm_torchtpu.layers.adapter.sample.top_k_top_p import (
    CANDIDATE_BLOCK,
    MASKED_LOGIT_VALUE,
    SAMPLING_EPS,
    apply_top_k_top_p,
    gumbel_noise_from_uniform,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

MAX_DISTRIBUTED_TOPK = 100
# 128 matches the native TPU vector lane width and gives >= 29 tie slots at
# MAX_DISTRIBUTED_TOPK=100 (65 at top_k=64), matching apply_top_k_top_p_candidates.
CANDIDATES_PER_SHARD = CANDIDATE_BLOCK
# Minimizes total two-stage top-k width (V_shard / b + M * b): optimum is
# sqrt(V_shard / M) in [16, 32] for V_shard in [32k, 131k] and M = 128.
_STAGE1_BLOCK = 32


class DistributedSamplingMode(StrEnum):
    TOPK_TOPP = "topk_topp"
    PURE_TEMP = "pure_temp"
    MIXED = "mixed"


def rng_width(mode: DistributedSamplingMode, tp_size: int, shard_vocab: int) -> int:
    """Per-row width of the ``float32`` uniform draw consumed by ``mode``."""
    if mode is DistributedSamplingMode.TOPK_TOPP:
        return tp_size * min(shard_vocab, CANDIDATES_PER_SHARD)
    if mode is DistributedSamplingMode.PURE_TEMP:
        return tp_size + cdiv(shard_vocab, CANDIDATE_BLOCK) + CANDIDATE_BLOCK
    raise ValueError(f"rng_width expects TOPK_TOPP or PURE_TEMP, got {mode!r}")


def _mask_shard_padding(
    logits: torch.Tensor, num_padding: torch.Tensor
) -> torch.Tensor:
    """Sets trailing ``num_padding`` padding columns of ``logits`` to ``-inf``."""
    shard_vocab = logits.shape[-1]
    cols = torch.arange(shard_vocab, dtype=torch.int32, device=logits.device)
    return torch.where(
        cols.unsqueeze(0) < shard_vocab - num_padding,
        logits,
        torch.full_like(logits, float("-inf")),
    )


def _prepare_shard_logits(
    owner: torch.nn.Module, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Projects shard logits, masks vocab padding, and packs the greedy pair."""
    processor = cast("LogitsProcessor", owner.logits_processor)
    head = cast("VocabParallelEmbedding", owner.lm_head)
    bias = getattr(head, "bias", None)
    logits = processor(head, hidden_states, bias, skip_gather=True)
    assert logits is not None
    vocab_start = cast(torch.Tensor, head.org_vocab_start_index).to(torch.int32)
    num_padding = cast(torch.Tensor, head.num_org_vocab_padding).to(torch.int32)
    logits = _mask_shard_padding(logits, num_padding)
    shard_max, shard_argmax = torch.max(logits, dim=-1, keepdim=True)
    greedy_pair = torch.cat(
        [
            shard_max.to(torch.float32).view(torch.int32),
            shard_argmax.to(torch.int32) + vocab_start,
        ],
        dim=-1,
    )
    return logits, vocab_start, greedy_pair, int(head.tp_size)


def _pad_to_blocks(logits: torch.Tensor, block_size: int) -> torch.Tensor:
    """Pads ``[B, V]`` with ``-inf`` to ``block_size`` as ``[B, N, block_size]``."""
    rows, width = logits.shape
    tail = width % block_size
    if tail:
        logits = F.pad(logits, (0, block_size - tail), value=float("-inf"))
    return logits.reshape(rows, -1, block_size)


def _extract_candidates(
    logits: torch.Tensor, vocab_start: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact top-``CANDIDATES_PER_SHARD`` ``(logits, global_token_ids)`` per shard.

    Direct ``torch.topk`` over a full shard (e.g. 131k columns) takes 7.7 ms on
    TPU v7x. Filtering first by block maxima of width ``_STAGE1_BLOCK`` shrinks
    both top-k inputs to ~4k elements (~0.34 ms on v7x) while preserving every
    top-k token (exact unless > ``CANDIDATES_PER_SHARD - k + 1`` tokens tie at
    the k-th logit).
    """
    rows, shard_vocab = logits.shape
    if shard_vocab <= CANDIDATES_PER_SHARD:
        ids = torch.arange(shard_vocab, dtype=torch.int32, device=logits.device)
        return logits, ids.unsqueeze(0).expand(rows, shard_vocab) + vocab_start
    if shard_vocab <= CANDIDATES_PER_SHARD * _STAGE1_BLOCK:
        vals, idx = torch.topk(logits, CANDIDATES_PER_SHARD, dim=-1, sorted=False)
        return vals, idx.to(torch.int32) + vocab_start

    blocks = _pad_to_blocks(logits, _STAGE1_BLOCK)
    block_max = blocks.amax(dim=-1)
    top_blocks = torch.topk(
        block_max, CANDIDATES_PER_SHARD, dim=-1, sorted=False
    ).indices
    gathered = torch.gather(
        blocks, 1, top_blocks.unsqueeze(-1).expand(-1, -1, _STAGE1_BLOCK)
    ).reshape(rows, CANDIDATES_PER_SHARD * _STAGE1_BLOCK)
    vals, local_idx = torch.topk(gathered, CANDIDATES_PER_SHARD, dim=-1, sorted=False)
    orig_block = torch.gather(top_blocks, 1, local_idx // _STAGE1_BLOCK)
    global_ids = (orig_block * _STAGE1_BLOCK + (local_idx % _STAGE1_BLOCK)).to(
        torch.int32
    ) + vocab_start
    return vals, global_ids


def _select_column(table: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
    """``table[row, col[row]]`` as ``[B, 1]`` int64, without an HLO gather."""
    cols = torch.arange(table.shape[1], dtype=col.dtype, device=table.device)
    picked = torch.where(cols.unsqueeze(0) == col, table, torch.zeros_like(table))
    return picked.sum(dim=-1, keepdim=True, dtype=torch.int32).to(torch.int64)


def _gather_with_greedy(
    payload: torch.Tensor, greedy_pair: torch.Tensor, tp_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-gathers ``[payload, greedy_pair]`` across TP and decodes greedy tokens."""
    packed = torch.cat([payload, greedy_pair], dim=-1)
    rows, width = packed.shape
    gathered = tensor_model_parallel_all_gather(packed, dim=-1).reshape(
        rows, tp_size, width
    )
    shard_max = gathered[:, :, -2].view(torch.float32)
    shard_argmax = gathered[:, :, -1]
    greedy = _select_column(shard_argmax, torch.argmax(shard_max, dim=-1, keepdim=True))
    return gathered[:, :, :-2], greedy


def _safe_temperatures(temperatures: torch.Tensor) -> torch.Tensor:
    temps = temperatures.to(torch.float32)
    return torch.where(temps <= SAMPLING_EPS, 1.0, temps)


def sample_topk_topp(
    owner: torch.nn.Module,
    hidden_states: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """``TOPK_TOPP`` mode: sample from merged exact per-shard top candidates."""
    rows = hidden_states.shape[0]
    logits, vocab_start, greedy_pair, tp_size = _prepare_shard_logits(
        owner, hidden_states
    )
    cand_logits, cand_ids = _extract_candidates(logits, vocab_start)
    num_cands = cand_logits.shape[-1]

    payload = torch.cat(
        [cand_logits.to(torch.float32).view(torch.int32), cand_ids], dim=-1
    )
    gathered, greedy = _gather_with_greedy(payload, greedy_pair, tp_size)
    all_logits = gathered[:, :, :num_cands].reshape(rows, -1).view(torch.float32)
    all_ids = gathered[:, :, num_cands : 2 * num_cands].reshape(rows, -1)

    scaled = all_logits / _safe_temperatures(temperatures)
    masked = apply_top_k_top_p(scaled, top_k, top_p, topk_candidates=0)
    masked = torch.where(
        masked <= MASKED_LOGIT_VALUE, torch.full_like(masked, float("-inf")), masked
    )
    noisy = masked + gumbel_noise_from_uniform(u[:, : all_logits.shape[-1]])
    sampled = _select_column(all_ids, torch.argmax(noisy, dim=-1, keepdim=True))
    return torch.where(temperatures <= SAMPLING_EPS, greedy, sampled)


def _sample_within_shard(
    blocks: torch.Tensor,
    vocab_start: torch.Tensor,
    tp_size: int,
    u: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-level Gumbel-max over ``[B, num_blocks, CANDIDATE_BLOCK]`` logits."""
    num_blocks = blocks.shape[1]
    block_lse = torch.logsumexp(blocks, dim=-1)
    shard_lse = torch.logsumexp(block_lse, dim=-1, keepdim=True)

    u_cross = u[:, :tp_size]
    u_local = u[:, tp_size : tp_size + num_blocks + CANDIDATE_BLOCK]
    block = torch.argmax(
        block_lse + gumbel_noise_from_uniform(u_local[:, :num_blocks]),
        dim=-1,
        keepdim=True,
    )
    block_ids = torch.arange(num_blocks, dtype=block.dtype, device=blocks.device)
    block_logits = torch.where(
        block_ids.view(1, num_blocks, 1) == block.unsqueeze(-1),
        blocks,
        torch.full_like(blocks, float("-inf")),
    ).amax(dim=1)
    offset = torch.argmax(
        block_logits + gumbel_noise_from_uniform(u_local[:, num_blocks:]),
        dim=-1,
        keepdim=True,
    )
    token_id = (block * CANDIDATE_BLOCK + offset).to(torch.int32) + vocab_start
    return shard_lse, token_id, u_cross


def sample_pure_temp(
    owner: torch.nn.Module,
    hidden_states: torch.Tensor,
    temperatures: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """``PURE_TEMP`` mode: exact sample from ``softmax(logits / T)``."""
    logits, vocab_start, greedy_pair, tp_size = _prepare_shard_logits(
        owner, hidden_states
    )
    scaled = logits.to(torch.float32) / _safe_temperatures(temperatures)
    blocks = _pad_to_blocks(scaled, CANDIDATE_BLOCK)
    shard_lse, token_id, u_cross = _sample_within_shard(blocks, vocab_start, tp_size, u)

    payload = torch.cat([shard_lse.view(torch.int32), token_id], dim=-1)
    gathered, greedy = _gather_with_greedy(payload, greedy_pair, tp_size)
    winner = torch.argmax(
        gathered[:, :, 0].view(torch.float32) + gumbel_noise_from_uniform(u_cross),
        dim=-1,
        keepdim=True,
    )
    sampled = _select_column(gathered[:, :, 1], winner)
    return torch.where(temperatures <= SAMPLING_EPS, greedy, sampled)
