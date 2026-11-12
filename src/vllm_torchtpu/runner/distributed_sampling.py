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
"""Vocabulary-parallel non-greedy sampling across tensor-parallel LM head shards.

Eliminates full-vocabulary ``[B, V]`` logit ``all-gather`` collectives, full-vocab
logit softcapping (when ``1 <= top_k <= MAX_DISTRIBUTED_TOPK``), and full-vocab
``[B, V]`` uniform/Gumbel RNG materialization on non-speculative steps that
require neither logprobs, grammar bitmasks, nor ``min_tokens`` stop-token masks:

1. ``topk_topp`` (``1 <= top_k <= MAX_DISTRIBUTED_TOPK``, any ``top_p <= 1.0``):
   Each TP shard projects its local LM head slice ``[B, V_local]``, extracts
   candidate ``(value, token_id)`` pairs via pure-TensorCore ``256``-lane
   strided VPU ``max`` passes (avoiding ``HloOpcode::kTopK``, ``kSort``, and
   ``kGather`` SparseCore offloads inside collective HLO modules), defers
   monotonic ``soft_cap`` (``c * tanh(x / c)``) and ``scale`` to only those
   extracted candidates, exchanges packed ``(value, token_id, greedy_pair)``
   entries across TP, applies ``apply_top_k_top_p`` on the compact candidate
   vector, and samples via Gumbel-Max.

2. ``pure_temp`` (``no_top_k`` and ``no_top_p``, ``temp > 0``):
   Each TP shard computes its local softcapped/scaled logits in ``128``-token
   blocks, evaluates exact ``block_lse = logsumexp(block)`` and ``shard_lse``,
   samples one local candidate token via 2-level Gumbel-Max, exchanges only
   ``[B, 4]`` int32 words (``(shard_lse, local_token, greedy_pair)``) across TP,
   and samples the winning shard via ``argmax(shard_lse + Gumbel)``. By Gumbel
   max-stability, this samples from the exact categorical distribution
   ``softmax(logits / T)``.

Requests setting ``top_p < 1.0`` without ``top_k`` (or ``top_k > 100``, or mixed
``top_k`` and ``no_top_k`` batches) return ``None`` from ``select_chunk_mode``
on CPU metadata and fall back cleanly to full-logits sampling without issuing
multi-round cross-shard binary-search collectives.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    tensor_model_parallel_all_gather,
)
from vllm.lora.layers import LogitsProcessorWithLoRA
from vllm.model_executor.layers.logits_processor import LogitsProcessor

from vllm_torchtpu.layers.adapter import dp_vocab_parallel
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import (
    CANDIDATE_BLOCK,
    MASKED_LOGIT_VALUE,
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

SAMPLING_EPS = 1e-5
LOCAL_TOPK_CANDIDATES = 128
STRIDE_LANES = 256
# Covers standard top_k settings (e.g. k=1..100, including Gemma 4 k=64) with
# generous multi-round strided candidate slack across TP shards.
MAX_DISTRIBUTED_TOPK = 100


class DistributedSamplingMode(Enum):
    TOPK_TOPP = "topk_topp"
    PURE_TEMP = "pure_temp"


def _register_shard_buffers(head: VocabParallelEmbedding) -> None:
    """Registers shard index and TP-rank buffers on ``head``'s device.

    Reading ``nn.Module`` tensor buffers rather than Python ``int`` constants
    inside ``@torch.compile`` keeps the traced MLIR/HLO identical across all TP
    ranks so SPMD compilation cache hits cleanly across ranks ``0..tp_size - 1``.
    """
    weight = cast(torch.Tensor, head.weight)
    dev = weight.device
    tp_rank = 0
    if int(head.tp_size) > 1:
        try:
            tp_rank = get_tensor_model_parallel_rank()
        except AssertionError:
            tp_rank = 0

    for attr, val in (
        ("org_vocab_start_index", int(head.shard_indices.org_vocab_start_index)),
        ("num_org_vocab_padding", int(head.shard_indices.num_org_vocab_padding)),
        ("_tpu_tp_rank", tp_rank),
    ):
        existing = getattr(head, attr, None)
        if existing is not None and isinstance(existing, torch.Tensor):
            if existing.device != dev:
                setattr(head, attr, existing.to(device=dev))
        else:
            head.register_buffer(
                attr,
                torch.tensor(val, dtype=torch.int64, device=dev),
                persistent=False,
            )


def _get_shard_buffers(
    head: VocabParallelEmbedding, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(start_idx, num_pad, tp_rank)`` as scalar ``int32`` tensors."""
    start_buf = getattr(head, "org_vocab_start_index", None)
    pad_buf = getattr(head, "num_org_vocab_padding", None)
    rank_buf = getattr(head, "_tpu_tp_rank", None)
    if (
        start_buf is not None
        and pad_buf is not None
        and rank_buf is not None
        and isinstance(start_buf, torch.Tensor)
        and isinstance(pad_buf, torch.Tensor)
        and isinstance(rank_buf, torch.Tensor)
    ):
        return (
            start_buf.to(device=device, dtype=torch.int32),
            pad_buf.to(device=device, dtype=torch.int32),
            rank_buf.to(device=device, dtype=torch.int32),
        )
    tp_rank = get_tensor_model_parallel_rank() if int(head.tp_size) > 1 else 0
    return (
        torch.tensor(
            int(head.shard_indices.org_vocab_start_index),
            dtype=torch.int32,
            device=device,
        ),
        torch.tensor(
            int(head.shard_indices.num_org_vocab_padding),
            dtype=torch.int32,
            device=device,
        ),
        torch.tensor(tp_rank, dtype=torch.int32, device=device),
    )


def resolve_head_owner(model: torch.nn.Module, dp_lockstep: bool) -> torch.nn.Module:
    """Returns the module holding ``lm_head`` and ``logits_processor`` for
    vocabulary-parallel non-greedy sampling, or raises ``NotImplementedError``."""
    try:
        owner = greedy_top_tokens.resolve_head_owner(model, dp_lockstep)
    except NotImplementedError as exc:
        msg = str(exc).replace("TPU_GREEDY_TOP_TOKENS", "TPU_DISTRIBUTED_SAMPLING")
        raise NotImplementedError(msg) from exc

    processor = owner.logits_processor
    head = cast("VocabParallelEmbedding", owner.lm_head)
    if getattr(head, dp_vocab_parallel.DP_VOCAB_ATTR, False):
        raise NotImplementedError(
            "TPU_DISTRIBUTED_SAMPLING: TPU_DP_VOCAB_PARALLEL shards lm_head "
            "over EP rather than TP; unset TPU_DISTRIBUTED_SAMPLING."
        )
    if int(head.shard_indices.num_added_elements) > 0:
        raise NotImplementedError(
            "TPU_DISTRIBUTED_SAMPLING: added vocabulary elements on lm_head "
            "require full-logits mapping; unset TPU_DISTRIBUTED_SAMPLING."
        )
    if isinstance(processor, LogitsProcessorWithLoRA):
        raise NotImplementedError(
            "TPU_DISTRIBUTED_SAMPLING: LoRA-wrapped LogitsProcessor requires "
            "the full-logits LoRA delta path; unset TPU_DISTRIBUTED_SAMPLING."
        )
    if not isinstance(processor, LogitsProcessor):
        raise NotImplementedError(
            f"TPU_DISTRIBUTED_SAMPLING: unsupported logits processor "
            f"{type(processor).__name__}; unset TPU_DISTRIBUTED_SAMPLING."
        )
    if processor.logits_as_input:
        raise NotImplementedError(
            "TPU_DISTRIBUTED_SAMPLING: logits_as_input=True requires full logits; "
            "unset TPU_DISTRIBUTED_SAMPLING."
        )
    if processor.soft_cap is not None and processor.soft_cap <= 0.0:
        raise NotImplementedError(
            f"TPU_DISTRIBUTED_SAMPLING: non-positive soft_cap={processor.soft_cap} "
            "is not monotonically increasing; unset TPU_DISTRIBUTED_SAMPLING."
        )
    _register_shard_buffers(head)
    return owner


def step_eligible(
    input_batch: InputBatch, requests: Mapping[str, CachedRequestState]
) -> bool:
    """Whether a non-greedy step can defer LM head projection to ``sample_tokens``.

    Requires at least one non-greedy request, no logprobs, and no structured
    decoding grammar bitmask. (Purely greedy batches stay on their greedy path.)
    """
    if input_batch.all_greedy or input_batch.max_num_logprobs is not None:
        return False
    for req_id in input_batch.req_ids:
        if req_id is None:
            continue
        params = requests[req_id].sampling_params
        if params is not None and params.structured_outputs is not None:
            logger.info_once(
                "TPU_DISTRIBUTED_SAMPLING: steps with a structured-output request "
                "gather the full logits"
            )
            return False
    return True


def select_chunk_mode(
    input_batch: InputBatch,
    cur_start_idx: int,
    cur_end_idx: int,
    vocab_size: int,
) -> DistributedSamplingMode | None:
    """Selects the compiled vocabulary-parallel sampling mode on CPU metadata.

    Returns ``DistributedSamplingMode.TOPK_TOPP`` when all non-greedy requests in
    the slice satisfy ``1 <= top_k <= MAX_DISTRIBUTED_TOPK`` (with any
    ``top_p <= 1.0``), ``DistributedSamplingMode.PURE_TEMP`` when no non-greedy
    request sets ``top_k`` or ``top_p < 1.0``, and ``None`` otherwise (including
    ``top_p``-only requests, ``top_k > MAX_DISTRIBUTED_TOPK``, and mixed
    ``top_k`` / non-``top_k`` slices) so those slices fall back cleanly to
    full-logits sampling.
    """
    if cur_end_idx <= cur_start_idx:
        return None
    temps = input_batch.temperature_cpu_tensor[cur_start_idx:cur_end_idx]
    non_greedy = temps > SAMPLING_EPS
    if not bool(non_greedy.any()):
        return DistributedSamplingMode.TOPK_TOPP

    top_k = input_batch.top_k_cpu_tensor[cur_start_idx:cur_end_idx]
    top_p = input_batch.top_p_cpu_tensor[cur_start_idx:cur_end_idx]
    active_k = top_k[non_greedy]
    active_p = top_p[non_greedy]

    has_valid_topk = (active_k >= 1) & (active_k < vocab_size)
    if bool((has_valid_topk & (active_k <= MAX_DISTRIBUTED_TOPK)).all()):
        return DistributedSamplingMode.TOPK_TOPP

    no_topk_in_chunk = not bool(has_valid_topk.any())
    if no_topk_in_chunk and not bool((active_p < 1.0).any()):
        return DistributedSamplingMode.PURE_TEMP

    logger.info_once(
        "TPU_DISTRIBUTED_SAMPLING: slices outside 1 <= top_k <= %d or pure "
        "temperature gather the full logits",
        MAX_DISTRIBUTED_TOPK,
    )
    return None


def step_has_eligible_request(input_batch: InputBatch, vocab_size: int) -> bool:
    """Returns whether any non-greedy request in ``input_batch`` can use a
    distributed sampling mode when ``execute_model`` runs multiple chunks."""
    num_reqs = int(input_batch.num_reqs)
    if num_reqs <= 0:
        return False
    temps = input_batch.temperature_cpu_tensor[:num_reqs]
    non_greedy = temps > SAMPLING_EPS
    if not bool(non_greedy.any()):
        return False
    active_k = input_batch.top_k_cpu_tensor[:num_reqs][non_greedy]
    active_p = input_batch.top_p_cpu_tensor[:num_reqs][non_greedy]
    topk_ok = (active_k >= 1) & (active_k <= min(MAX_DISTRIBUTED_TOPK, vocab_size - 1))
    pure_temp_ok = ((active_k < 1) | (active_k >= vocab_size)) & (active_p >= 1.0)
    return bool((topk_ok | pure_temp_ok).any())


def _local_topk_rounds(tp_size: int) -> int:
    return 6 if tp_size <= 2 else 4


def _local_topk_cand_width(tp_size: int, v_local: int) -> int:
    rounds = _local_topk_rounds(tp_size)
    max_cands = rounds * STRIDE_LANES
    if v_local <= max_cands:
        return v_local
    num_blocks = (v_local + STRIDE_LANES - 1) // STRIDE_LANES
    return min(rounds, num_blocks) * STRIDE_LANES


def topk_topp_rng_width(tp_size: int, v_local: int = LOCAL_TOPK_CANDIDATES) -> int:
    """Width of the per-row ``float32`` RNG draw for ``TOPK_TOPP``."""
    return tp_size * _local_topk_cand_width(tp_size, v_local)


def pure_temp_rng_width(v_local: int, tp_size: int) -> int:
    """Width of the per-row ``float32`` RNG draw for ``PURE_TEMP``."""
    num_blocks = (v_local + CANDIDATE_BLOCK - 1) // CANDIDATE_BLOCK
    return tp_size + tp_size * (num_blocks + CANDIDATE_BLOCK)


def log_step_path(mode: DistributedSamplingMode, req_ids: Sequence[str | None]) -> None:
    """Logs once per mode when served traffic takes distributed sampling."""
    if all(req_id is None or req_id.startswith("__") for req_id in req_ids):
        return
    logger.info_once(
        "TPU_DISTRIBUTED_SAMPLING: non-speculative step took mode=%s", mode.value
    )


def _compute_masked_local_logits(
    processor: LogitsProcessor,
    head: VocabParallelEmbedding,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Projects ``hidden_states`` onto this rank's LM head shard and masks padding."""
    local_logits = processor._apply_head(head, hidden_states, None)
    start_idx, num_pad, tp_rank = _get_shard_buffers(head, local_logits.device)
    v_local = local_logits.shape[-1]
    valid_cols = v_local - num_pad
    col_iota = torch.arange(
        v_local, dtype=torch.int32, device=local_logits.device
    ).unsqueeze(0)
    local_logits = torch.where(
        col_iota < valid_cols,
        local_logits,
        torch.full_like(local_logits, float("-inf")),
    )
    return local_logits, start_idx, tp_rank


def _apply_softcap_and_scale(
    logits: torch.Tensor, processor: LogitsProcessor
) -> torch.Tensor:
    """Applies ``soft_cap`` then ``scale`` matching ``LogitsProcessor.forward``."""
    if processor.soft_cap is not None:
        logits = torch.tanh(logits / processor.soft_cap) * processor.soft_cap
    if processor.scale != 1.0:
        logits = logits * processor.scale
    return logits


def _extract_local_topk(
    local_logits: torch.Tensor, start_idx: torch.Tensor, tp_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-TensorCore extraction of candidate ``(values, global_token_ids)`` per row.

    Reshapes ``local_logits`` into ``[rows, num_blocks, STRIDE_LANES]`` and runs
    ``rounds`` VPU ``torch.max(dim=1)`` passes rather than ``torch.topk`` or
    ``torch.gather``, avoiding XLA SparseCore offload inside collective HLO
    modules across all ``TP`` sizes and batch buckets. When ``v_local`` already
    fits within ``rounds * STRIDE_LANES``, returns the full local shard directly.
    """
    rows, v_local = local_logits.shape
    rounds = _local_topk_rounds(tp_size)
    if v_local <= rounds * STRIDE_LANES:
        ids = (
            torch.arange(v_local, dtype=torch.int32, device=local_logits.device)
            .unsqueeze(0)
            .expand(rows, v_local)
            + start_idx
        )
        return local_logits, ids

    rem = v_local % STRIDE_LANES
    if rem != 0:
        local_logits = F.pad(local_logits, (0, STRIDE_LANES - rem), value=float("-inf"))
    num_blocks = local_logits.shape[-1] // STRIDE_LANES
    work = local_logits.reshape(rows, num_blocks, STRIDE_LANES)
    neg_inf = torch.full_like(work, float("-inf"))
    block_iota = torch.arange(num_blocks, dtype=torch.int32, device=work.device).view(
        1, num_blocks, 1
    )
    lane_iota = torch.arange(STRIDE_LANES, dtype=torch.int32, device=work.device).view(
        1, STRIDE_LANES
    )

    vals_list: list[torch.Tensor] = []
    ids_list: list[torch.Tensor] = []
    actual_rounds = min(rounds, num_blocks)
    for r in range(actual_rounds):
        v_r, idx_r = torch.max(work, dim=1)
        idx_i32 = idx_r.to(torch.int32)
        gid_r = idx_i32 * STRIDE_LANES + lane_iota + start_idx
        vals_list.append(v_r)
        ids_list.append(gid_r)
        if r + 1 < actual_rounds:
            work = torch.where(block_iota == idx_i32.unsqueeze(1), neg_inf, work)

    return torch.cat(vals_list, dim=-1), torch.cat(ids_list, dim=-1)


def _gather_across_tp(packed_i32: torch.Tensor, tp_size: int) -> torch.Tensor:
    """All-gathers ``[B, W]`` int32 across TP into ``[B, tp_size, W]``."""
    rows, width = packed_i32.shape
    if tp_size == 1:
        return packed_i32.unsqueeze(1)
    gathered = tensor_model_parallel_all_gather(packed_i32, dim=-1)
    return gathered.reshape(rows, tp_size, width)


def _select_one_col_i32(table_i32: torch.Tensor, col_idx: torch.Tensor) -> torch.Tensor:
    """Selects 1 column per row from ``[B, W]`` int32 without HLO gather."""
    cols = torch.arange(
        table_i32.shape[1], dtype=col_idx.dtype, device=table_i32.device
    ).unsqueeze(0)
    return torch.sum(
        torch.where(cols == col_idx, table_i32, torch.zeros_like(table_i32)),
        dim=-1,
        keepdim=True,
        dtype=torch.int32,
    )


def _greedy_from_pairs(
    max_vals_f32: torch.Tensor, max_ids_i32: torch.Tensor
) -> torch.Tensor:
    """Selects greedy token ``[B, 1]`` int64; ties resolve to lowest rank/id."""
    best_rank = torch.argmax(max_vals_f32, dim=-1, keepdim=True)
    return _select_one_col_i32(max_ids_i32, best_rank).to(torch.int64)


def distributed_topk_topp_sample(
    owner: torch.nn.Module,
    hidden_states: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Vocabulary-parallel ``top_k`` + ``top_p`` sampling with deferred softcapping."""
    processor = cast("LogitsProcessor", owner.logits_processor)
    head = cast("VocabParallelEmbedding", owner.lm_head)
    tp_size = int(head.tp_size)
    rows = hidden_states.shape[0]

    local_logits, start_idx, _ = _compute_masked_local_logits(
        processor, head, hidden_states
    )
    max_vals, max_local_ids = torch.max(local_logits, dim=-1, keepdim=True)
    max_global_ids = max_local_ids.to(torch.int32) + start_idx

    local_vals, local_ids = _extract_local_topk(local_logits, start_idx, tp_size)
    cand_k = local_vals.shape[-1]

    # Defer monotonic tanh softcapping and logit scaling until after candidate
    # extraction instead of evaluating tanh over the full [B, V_local] shard.
    vals_f32 = _apply_softcap_and_scale(local_vals, processor).to(torch.float32)
    max_vals_f32 = _apply_softcap_and_scale(max_vals, processor).to(torch.float32)

    packed = torch.cat(
        [
            vals_f32.view(torch.int32),
            local_ids,
            max_vals_f32.view(torch.int32),
            max_global_ids,
        ],
        dim=-1,
    )
    gathered = _gather_across_tp(packed, tp_size)

    all_vals = (
        gathered[:, :, :cand_k].reshape(rows, tp_size * cand_k).view(torch.float32)
    )
    all_ids = gathered[:, :, cand_k : 2 * cand_k].reshape(rows, tp_size * cand_k)
    all_max_vals = gathered[:, :, 2 * cand_k].view(torch.float32)
    all_max_ids = gathered[:, :, 2 * cand_k + 1]

    greedy_tokens = _greedy_from_pairs(all_max_vals, all_max_ids)

    temps_f32 = temperatures.to(torch.float32)
    safe_temps = torch.where(temps_f32 <= SAMPLING_EPS, 1.0, temps_f32)
    scaled_vals = all_vals / safe_temps

    # Pure-TensorCore binary search over the compact candidate vector avoids
    # HloOpcode::kSort and HloOpcode::kGather SparseCore offloads inside
    # collective HLO modules.
    total_cands = scaled_vals.shape[-1]
    masked_vals = apply_top_k_top_p(scaled_vals, top_k, top_p, topk_candidates=0)
    masked_vals = torch.where(
        masked_vals <= MASKED_LOGIT_VALUE,
        torch.full_like(masked_vals, float("-inf")),
        masked_vals,
    )

    noisy = masked_vals + gumbel_noise_from_uniform(u[:, :total_cands])
    pick_idx = torch.argmax(noisy, dim=-1, keepdim=True)
    rand_tokens = _select_one_col_i32(all_ids, pick_idx).to(torch.int64)

    return torch.where(temperatures <= SAMPLING_EPS, greedy_tokens, rand_tokens)


def _sample_from_sharded_blocks(
    z_blocks: torch.Tensor,
    start_idx: torch.Tensor,
    tp_size: int,
    tp_rank: torch.Tensor,
    u: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """2-level intra-shard Gumbel-Max returning ``(shard_lse, local_tok, u_cross)``."""
    rows, num_blocks, _ = z_blocks.shape
    block_lse = safe_logsumexp(z_blocks, dim=-1)
    shard_lse = safe_logsumexp(block_lse, dim=-1, keepdim=True)

    per_rank_width = num_blocks + CANDIDATE_BLOCK
    u_cross = u[:, :tp_size]
    u_ranks = u[:, tp_size : tp_size + tp_size * per_rank_width].reshape(
        rows, tp_size, per_rank_width
    )
    rank_iota = torch.arange(tp_size, dtype=torch.int32, device=u.device).view(
        1, tp_size, 1
    )
    u_local = torch.where(
        rank_iota == tp_rank.view(1, 1, 1),
        u_ranks,
        torch.zeros_like(u_ranks),
    ).sum(dim=1)
    u_block = u_local[:, :num_blocks]
    u_intra = u_local[:, num_blocks:]

    win_block = torch.argmax(
        block_lse + gumbel_noise_from_uniform(u_block), dim=-1, keepdim=True
    )
    # Use masked TensorCore reduction instead of [rows, 1, CANDIDATE_BLOCK]
    # torch.gather, which triggers SparseCore gather-offload on small B.
    block_idx = torch.arange(
        num_blocks, dtype=win_block.dtype, device=z_blocks.device
    ).view(1, num_blocks, 1)
    win_logits = torch.where(
        block_idx == win_block.unsqueeze(-1),
        z_blocks,
        torch.full_like(z_blocks, float("-inf")),
    ).amax(dim=1)
    win_offset = torch.argmax(
        win_logits + gumbel_noise_from_uniform(u_intra), dim=-1, keepdim=True
    )
    local_tok_id = (win_block * CANDIDATE_BLOCK + win_offset).to(
        torch.int32
    ) + start_idx
    return shard_lse, local_tok_id, u_cross


def distributed_pure_temp_sample(
    owner: torch.nn.Module,
    hidden_states: torch.Tensor,
    temperatures: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Exact vocabulary-parallel sampling for ``no_top_k`` and ``no_top_p``."""
    processor = cast("LogitsProcessor", owner.logits_processor)
    head = cast("VocabParallelEmbedding", owner.lm_head)
    tp_size = int(head.tp_size)
    rows = hidden_states.shape[0]

    local_logits, start_idx, tp_rank = _compute_masked_local_logits(
        processor, head, hidden_states
    )
    capped = _apply_softcap_and_scale(local_logits, processor).to(torch.float32)
    max_vals_f32, max_local_ids = torch.max(capped, dim=-1, keepdim=True)
    max_global_ids = max_local_ids.to(torch.int32) + start_idx

    temps_f32 = temperatures.to(torch.float32)
    safe_temps = torch.where(temps_f32 <= SAMPLING_EPS, 1.0, temps_f32)
    z_local = capped / safe_temps

    v_local = z_local.shape[-1]
    rem = v_local % CANDIDATE_BLOCK
    if rem != 0:
        z_local = F.pad(z_local, (0, CANDIDATE_BLOCK - rem), value=float("-inf"))
    num_blocks = z_local.shape[-1] // CANDIDATE_BLOCK
    z_blocks = z_local.reshape(rows, num_blocks, CANDIDATE_BLOCK)

    shard_lse, local_tok_id, u_cross = _sample_from_sharded_blocks(
        z_blocks, start_idx, tp_size, tp_rank, u
    )

    packed = torch.cat(
        [
            shard_lse.view(torch.int32),
            local_tok_id,
            max_vals_f32.view(torch.int32),
            max_global_ids,
        ],
        dim=-1,
    )
    gathered = _gather_across_tp(packed, tp_size)
    all_lse = gathered[:, :, 0].view(torch.float32)
    all_toks = gathered[:, :, 1]
    all_max_vals = gathered[:, :, 2].view(torch.float32)
    all_max_ids = gathered[:, :, 3]

    win_shard = torch.argmax(
        all_lse + gumbel_noise_from_uniform(u_cross), dim=-1, keepdim=True
    )
    rand_tokens = _select_one_col_i32(all_toks, win_shard).to(torch.int64)
    greedy_tokens = _greedy_from_pairs(all_max_vals, all_max_ids)
    return torch.where(temperatures <= SAMPLING_EPS, greedy_tokens, rand_tokens)
