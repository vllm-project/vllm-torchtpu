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
"""Non-greedy sampling from tensor-parallel LM head shards.

For ``TP > 1``, the default sampler all-gathers the full ``[B, V]`` logits and
draws a ``[B, V]`` uniform tensor. For the two modes below each TP rank works
on its own ``[B, V / tp]`` shard and only small per-row tensors cross TP:

- ``TOPK_TOPP`` (every non-greedy row has ``1 <= top_k <= MAX_DISTRIBUTED_TOPK``,
  any ``top_p``): each rank applies softcap/scale, masks vocab padding, extracts
  candidate ``(logit, token_id)`` pairs from its shard, all-gathers the
  candidates, and runs top-k/top-p plus Gumbel-max sampling on the merged set.
- ``PURE_TEMP`` (no non-greedy row sets ``top_k`` or ``top_p < 1``): each rank
  samples one token from its shard via two-level Gumbel-max and computes the
  shard logsumexp; the winning shard is drawn with ``argmax(shard_lse + Gumbel)``.
  This is an exact sample from ``softmax(logits / T)``.

Any other slice (``top_p``-only, ``top_k > MAX_DISTRIBUTED_TOPK``, or a mix of
both modes) gets ``None`` from ``select_chunk_mode`` and uses the full-logits
sampler.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.lora.layers import LogitsProcessorWithLoRA
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.utils.math_utils import cdiv

from vllm_torchtpu.layers.adapter import dp_vocab_parallel
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import (
    CANDIDATE_BLOCK,
    MASKED_LOGIT_VALUE,
    SAMPLING_EPS,
    apply_top_k_top_p,
    gumbel_noise_from_uniform,
    safe_logsumexp,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner import greedy_top_tokens

if TYPE_CHECKING:
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

logger = init_logger(__name__)

MAX_DISTRIBUTED_TOPK = 100
# Candidate extraction reshapes each shard into [num_blocks, CANDIDATE_LANES]
# (2 * CANDIDATE_BLOCK = two 128-wide TPU vector lanes) and keeps the top
# `rounds` entries per lane (6 for TP <= 2, 4 for TP >= 3). A true top-k token
# on a shard is missed only if > `rounds` higher-ranked tokens on that shard
# fall into the same lane mod 256 (including when high-probability tokens
# cluster on a single shard's 256 lanes).
CANDIDATE_LANES = 2 * CANDIDATE_BLOCK
_ROUNDS_SMALL_TP = 6
_ROUNDS_LARGE_TP = 4
# Stored as a tensor buffer on lm_head so the traced graph stays identical
# across TP ranks.
_TP_RANK_BUFFER = "_tpu_tp_rank"


class DistributedSamplingMode(StrEnum):
    TOPK_TOPP = "topk_topp"
    PURE_TEMP = "pure_temp"


def resolve_head_owner(
    model: torch.nn.Module,
    dp_lockstep: bool,
    *,
    speculative: bool = False,
) -> torch.nn.Module:
    """Returns the module holding ``lm_head`` and ``logits_processor``.

    Raises ``NotImplementedError`` when the configuration cannot sample from
    TP-sharded LM head weights.
    """
    if speculative:
        raise NotImplementedError(
            "TPU_DISTRIBUTED_SAMPLING: speculative decoding uses the "
            "full-logits rejection sampler; unset TPU_DISTRIBUTED_SAMPLING."
        )
    owner = greedy_top_tokens.find_vocab_parallel_head_owner(
        model, flag_name="TPU_DISTRIBUTED_SAMPLING"
    )
    processor = owner.logits_processor
    head = cast("VocabParallelEmbedding", owner.lm_head)
    if (
        getattr(head, dp_vocab_parallel.DP_VOCAB_ATTR, False)
        or getattr(head, "parallel_group", None) is not None
    ):
        raise NotImplementedError(
            "TPU_DISTRIBUTED_SAMPLING: lm_head is sharded over a custom or EP "
            "group (TPU_DP_VOCAB_PARALLEL) rather than TP; unset "
            "TPU_DISTRIBUTED_SAMPLING."
        )
    if dp_lockstep:
        raise NotImplementedError(
            "TPU_DISTRIBUTED_SAMPLING with expert-parallel data parallelism "
            "is not supported; unset TPU_DISTRIBUTED_SAMPLING or use tensor "
            "parallelism without EP lockstep."
        )
    if head.tp_size <= 1:
        raise NotImplementedError(
            "TPU_DISTRIBUTED_SAMPLING requires a tensor-parallel lm_head "
            "(tp_size > 1); unset TPU_DISTRIBUTED_SAMPLING."
        )
    if head.shard_indices.num_added_elements > 0:
        raise NotImplementedError(
            "TPU_DISTRIBUTED_SAMPLING: added vocabulary elements on lm_head "
            "require full-logits mapping; unset TPU_DISTRIBUTED_SAMPLING."
        )
    if isinstance(processor, LogitsProcessorWithLoRA):
        raise NotImplementedError(
            "TPU_DISTRIBUTED_SAMPLING: LoRA-wrapped LogitsProcessor requires "
            "the full-logits LoRA delta path; unset TPU_DISTRIBUTED_SAMPLING."
        )
    if type(processor) not in (LogitsProcessor, dp_vocab_parallel.TpuLogitsProcessor):
        raise NotImplementedError(
            f"TPU_DISTRIBUTED_SAMPLING: unsupported logits processor "
            f"{type(processor).__name__}; unset TPU_DISTRIBUTED_SAMPLING."
        )
    processor = cast(LogitsProcessor, processor)
    if processor.logits_as_input:
        raise NotImplementedError(
            "TPU_DISTRIBUTED_SAMPLING: logits_as_input=True requires full "
            "logits; unset TPU_DISTRIBUTED_SAMPLING."
        )
    if processor.soft_cap is not None and processor.soft_cap <= 0.0:
        raise NotImplementedError(
            f"TPU_DISTRIBUTED_SAMPLING: non-positive soft_cap="
            f"{processor.soft_cap} is not monotonically increasing; unset "
            "TPU_DISTRIBUTED_SAMPLING."
        )
    vocab_start_buf = getattr(head, "org_vocab_start_index", None)
    if vocab_start_buf is None or not isinstance(vocab_start_buf, torch.Tensor):
        raise NotImplementedError(
            "TPU_DISTRIBUTED_SAMPLING: lm_head was not built through the TPU "
            "VocabParallelEmbedding patch that registers its shard bounds as "
            "buffers; unset TPU_DISTRIBUTED_SAMPLING."
        )
    head.register_buffer(
        _TP_RANK_BUFFER,
        torch.tensor(
            int(head.tp_rank), dtype=torch.int32, device=vocab_start_buf.device
        ),
        persistent=False,
    )
    return owner


def head_tp_and_shard_vocab(owner: torch.nn.Module) -> tuple[int, int]:
    """Returns ``(tp_size, shard_vocab)`` from ``owner.lm_head``."""
    head = cast("VocabParallelEmbedding", owner.lm_head)
    return int(head.tp_size), int(head.num_embeddings_per_partition)


def step_eligible(
    input_batch: InputBatch, requests: Mapping[str, CachedRequestState]
) -> bool:
    """Whether a step may defer the LM head projection to sampling.

    Requires at least one non-greedy request, no logprobs, and no grammar.
    """
    if input_batch.all_greedy or input_batch.max_num_logprobs is not None:
        return False
    if greedy_top_tokens.has_structured_output(input_batch, requests):
        logger.info_once(
            "TPU_DISTRIBUTED_SAMPLING: steps with a structured-output request "
            "gather the full logits"
        )
        return False
    return True


def _classify_non_greedy_rows(
    input_batch: InputBatch, start: int, end: int, vocab_size: int
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Per non-greedy row in ``[start, end)``: ``(is_topk_topp, is_pure_temp)``.

    Returns ``None`` when the slice has no non-greedy row.
    """
    non_greedy = input_batch.temperature_cpu_tensor[start:end] > SAMPLING_EPS
    if not bool(non_greedy.any()):
        return None
    top_k = input_batch.top_k_cpu_tensor[start:end][non_greedy]
    top_p = input_batch.top_p_cpu_tensor[start:end][non_greedy]
    has_top_k = (top_k >= 1) & (top_k < vocab_size)
    is_topk_topp = has_top_k & (top_k <= MAX_DISTRIBUTED_TOPK)
    is_pure_temp = ~has_top_k & (top_p >= 1.0)
    return is_topk_topp, is_pure_temp


def select_chunk_mode(
    input_batch: InputBatch, start: int, end: int, vocab_size: int
) -> DistributedSamplingMode | None:
    """Picks the sampling mode for requests ``[start, end)`` from CPU metadata.

    An all-greedy slice inside a mixed multi-chunk step uses ``TOPK_TOPP``,
    which carries the exact cross-shard argmax for greedy rows. ``None`` means
    full-logits sampling.
    """
    if end <= start:
        return None
    classified = _classify_non_greedy_rows(input_batch, start, end, vocab_size)
    if classified is None:
        return DistributedSamplingMode.TOPK_TOPP
    is_topk_topp, is_pure_temp = classified
    if bool(is_topk_topp.all()):
        return DistributedSamplingMode.TOPK_TOPP
    if bool(is_pure_temp.all()):
        return DistributedSamplingMode.PURE_TEMP
    logger.info_once(
        "TPU_DISTRIBUTED_SAMPLING: slices outside 1 <= top_k <= %d or pure "
        "temperature gather the full logits",
        MAX_DISTRIBUTED_TOPK,
    )
    return None


def step_has_eligible_request(input_batch: InputBatch, vocab_size: int) -> bool:
    """Whether any non-greedy request in the batch can use a distributed mode."""
    classified = _classify_non_greedy_rows(
        input_batch, 0, input_batch.num_reqs, vocab_size
    )
    if classified is None:
        return False
    is_topk_topp, is_pure_temp = classified
    return bool((is_topk_topp | is_pure_temp).any())


def _candidate_rounds(tp_size: int) -> int:
    return _ROUNDS_SMALL_TP if tp_size <= 2 else _ROUNDS_LARGE_TP


def rng_width(mode: DistributedSamplingMode, tp_size: int, shard_vocab: int) -> int:
    """Per-row width of the ``float32`` uniform draw consumed by ``mode``."""
    if mode is DistributedSamplingMode.TOPK_TOPP:
        rounds = _candidate_rounds(tp_size)
        if shard_vocab <= rounds * CANDIDATE_LANES:
            return tp_size * shard_vocab
        return tp_size * rounds * CANDIDATE_LANES
    num_blocks = cdiv(shard_vocab, CANDIDATE_BLOCK)
    return tp_size + tp_size * (num_blocks + CANDIDATE_BLOCK)


def log_step_path(mode: DistributedSamplingMode, req_ids: Sequence[str | None]) -> None:
    """Logs once per mode when served traffic (not warmup) takes this path."""
    if greedy_top_tokens.is_synthetic_warmup(req_ids):
        return
    logger.info_once("TPU_DISTRIBUTED_SAMPLING: step took mode=%s", mode.value)


def _project_shard_logits(
    processor: LogitsProcessor,
    head: VocabParallelEmbedding,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Projects ``hidden_states`` onto this rank's ``[B, V_local]`` shard."""
    bias = getattr(head, "bias", None)
    logits = processor._apply_head(head, hidden_states, bias)
    vocab_start = cast(torch.Tensor, head.org_vocab_start_index).to(torch.int32)
    num_padding = cast(torch.Tensor, head.num_org_vocab_padding).to(torch.int32)
    tp_rank = cast(torch.Tensor, getattr(head, _TP_RANK_BUFFER))
    return logits, vocab_start, num_padding, tp_rank


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


def _apply_softcap_and_scale(
    logits: torch.Tensor, processor: LogitsProcessor
) -> torch.Tensor:
    """Applies ``soft_cap`` then ``scale`` in ``LogitsProcessor.forward`` order."""
    out = logits
    if processor.soft_cap is not None:
        out = torch.tanh(out / processor.soft_cap) * processor.soft_cap
    if processor.scale != 1.0:
        out = out * processor.scale
    return out


def _pad_to_blocks(logits: torch.Tensor, block_size: int) -> torch.Tensor:
    """Pads ``[B, V]`` with ``-inf`` to ``block_size`` as ``[B, N, block_size]``."""
    rows, width = logits.shape
    tail = width % block_size
    if tail:
        logits = F.pad(logits, (0, block_size - tail), value=float("-inf"))
    return logits.reshape(rows, -1, block_size)


def _extract_candidates(
    logits: torch.Tensor, vocab_start: torch.Tensor, tp_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Candidate ``(logits, global_token_ids)`` per row of a shard.

    Uses repeated ``max`` over the block axis instead of ``torch.topk``, which
    lowers to a sort that XLA offloads to SparseCore. Returns the whole shard
    when it is no wider than the candidate set.
    """
    rows, shard_vocab = logits.shape
    rounds = _candidate_rounds(tp_size)
    if shard_vocab <= rounds * CANDIDATE_LANES:
        ids = torch.arange(shard_vocab, dtype=torch.int32, device=logits.device)
        return logits, ids.unsqueeze(0).expand(rows, shard_vocab) + vocab_start

    blocks = _pad_to_blocks(logits, CANDIDATE_LANES)
    num_blocks = blocks.shape[1]
    neg_inf = torch.full_like(blocks, float("-inf"))
    block_ids = torch.arange(num_blocks, dtype=torch.int32, device=logits.device)
    lane_ids = torch.arange(CANDIDATE_LANES, dtype=torch.int32, device=logits.device)

    values: list[torch.Tensor] = []
    token_ids: list[torch.Tensor] = []
    for round_idx in range(rounds):
        best, best_block = torch.max(blocks, dim=1)
        best_block = best_block.to(torch.int32)
        values.append(best)
        token_ids.append(best_block * CANDIDATE_LANES + lane_ids + vocab_start)
        if round_idx + 1 < rounds:
            taken = block_ids.view(1, num_blocks, 1) == best_block.unsqueeze(1)
            blocks = torch.where(taken, neg_inf, blocks)
    return torch.cat(values, dim=-1), torch.cat(token_ids, dim=-1)


def _gather_across_tp(packed: torch.Tensor, tp_size: int) -> torch.Tensor:
    """All-gathers ``[B, W]`` int32 across TP into ``[B, tp_size, W]``."""
    rows, width = packed.shape
    gathered = tensor_model_parallel_all_gather(packed, dim=-1)
    return gathered.reshape(rows, tp_size, width)


def _select_column(table: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
    """``table[row, col[row]]`` as ``[B, 1]`` int64, without an HLO gather."""
    cols = torch.arange(table.shape[1], dtype=col.dtype, device=table.device)
    picked = torch.where(cols.unsqueeze(0) == col, table, torch.zeros_like(table))
    return picked.sum(dim=-1, keepdim=True, dtype=torch.int32).to(torch.int64)


def _greedy_tokens(shard_max: torch.Tensor, shard_argmax: torch.Tensor) -> torch.Tensor:
    """Cross-shard argmax ``[B, 1]``; ties go to the lowest rank, then id."""
    return _select_column(shard_argmax, torch.argmax(shard_max, dim=-1, keepdim=True))


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
    """``TOPK_TOPP`` mode: sample from the merged per-shard candidates."""
    processor = cast("LogitsProcessor", owner.logits_processor)
    head = cast("VocabParallelEmbedding", owner.lm_head)
    tp_size = head.tp_size
    rows = hidden_states.shape[0]

    raw_logits, vocab_start, num_padding, _ = _project_shard_logits(
        processor, head, hidden_states
    )
    # Apply softcap/scale in the head output dtype before masking padding so
    # greedy ties on capped values resolve to the lowest token ID (matching
    # full-logits argmax) and -inf padding stays -inf.
    logits = _mask_shard_padding(
        _apply_softcap_and_scale(raw_logits, processor), num_padding
    )
    shard_max, shard_argmax = torch.max(logits, dim=-1, keepdim=True)
    cand_logits, cand_ids = _extract_candidates(logits, vocab_start, tp_size)
    num_cands = cand_logits.shape[-1]

    cand_logits_f32 = cand_logits.to(torch.float32)
    shard_max_f32 = shard_max.to(torch.float32)
    packed = torch.cat(
        [
            cand_logits_f32.view(torch.int32),
            cand_ids,
            shard_max_f32.view(torch.int32),
            shard_argmax.to(torch.int32) + vocab_start,
        ],
        dim=-1,
    )
    gathered = _gather_across_tp(packed, tp_size)
    all_logits = gathered[:, :, :num_cands].reshape(rows, -1).view(torch.float32)
    all_ids = gathered[:, :, num_cands : 2 * num_cands].reshape(rows, -1)
    greedy = _greedy_tokens(
        gathered[:, :, 2 * num_cands].view(torch.float32),
        gathered[:, :, 2 * num_cands + 1],
    )

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
    tp_rank: torch.Tensor,
    u: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-level Gumbel-max over ``[B, num_blocks, CANDIDATE_BLOCK]`` logits.

    Picks a block by its logsumexp, then a token inside it, which is an exact
    sample from the shard softmax. Returns ``(shard_lse, token_id, u_cross)``.
    """
    rows, num_blocks, _ = blocks.shape
    block_lse = safe_logsumexp(blocks, dim=-1)
    shard_lse = safe_logsumexp(block_lse, dim=-1, keepdim=True)

    # Every rank draws the same u; each uses its own slice for the in-shard
    # draw and the shared first tp_size columns for the cross-shard draw.
    per_rank = num_blocks + CANDIDATE_BLOCK
    u_cross = u[:, :tp_size]
    u_ranks = u[:, tp_size : tp_size + tp_size * per_rank].reshape(
        rows, tp_size, per_rank
    )
    rank_ids = torch.arange(tp_size, dtype=torch.int32, device=u.device)
    u_local = torch.where(
        rank_ids.view(1, tp_size, 1) == tp_rank, u_ranks, torch.zeros_like(u_ranks)
    ).sum(dim=1)

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
    processor = cast("LogitsProcessor", owner.logits_processor)
    head = cast("VocabParallelEmbedding", owner.lm_head)
    tp_size = head.tp_size

    raw_logits, vocab_start, num_padding, tp_rank = _project_shard_logits(
        processor, head, hidden_states
    )
    logits = _mask_shard_padding(
        _apply_softcap_and_scale(raw_logits, processor).to(torch.float32),
        num_padding,
    )
    shard_max, shard_argmax = torch.max(logits, dim=-1, keepdim=True)

    scaled = logits / _safe_temperatures(temperatures)
    blocks = _pad_to_blocks(scaled, CANDIDATE_BLOCK)
    shard_lse, token_id, u_cross = _sample_within_shard(
        blocks, vocab_start, tp_size, tp_rank, u
    )

    packed = torch.cat(
        [
            shard_lse.view(torch.int32),
            token_id,
            shard_max.view(torch.int32),
            shard_argmax.to(torch.int32) + vocab_start,
        ],
        dim=-1,
    )
    gathered = _gather_across_tp(packed, tp_size)
    winner = torch.argmax(
        gathered[:, :, 0].view(torch.float32) + gumbel_noise_from_uniform(u_cross),
        dim=-1,
        keepdim=True,
    )
    sampled = _select_column(gathered[:, :, 1], winner)
    greedy = _greedy_tokens(gathered[:, :, 2].view(torch.float32), gathered[:, :, 3])
    return torch.where(temperatures <= SAMPLING_EPS, greedy, sampled)
