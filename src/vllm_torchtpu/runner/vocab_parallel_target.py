# SPDX-License-Identifier: Apache-2.0
"""Speculative verify steps sampled on each rank's vocabulary shard.

With ``TPU_SAMPLER_VOCAB_PARALLEL_TARGET``, a speculative verify step whose
requests are not all greedy and set nothing beyond temperature and top_p (no
logprobs, grammar, top_k, min_p, penalties or logit bias) keeps the target
LM head's logits sharded: the head returns this rank's [rows, V/tp] columns
(``LogitsProcessor(..., skip_gather=True)``) instead of all-gathering them,
and the verify runs on the shard
(``layers/adapter/sample/vocab_parallel_verify.py``). Every other step, and
every chunk of the step without drafts, takes the full logits.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

logger = init_logger(__name__)


@dataclass
class VocabShardHead:
    """The target LM head and this rank's slice of its vocabulary."""

    owner: torch.nn.Module  # holds lm_head and logits_processor
    # Global id of local column 0, as a device int32 scalar: an input of the
    # compiled programs, so every rank runs the same program.
    vocab_start: torch.Tensor
    num_org_local: int  # local columns that are real vocabulary
    width: int  # local columns, padding included
    tp_size: int
    vocab_size: int  # the full logits' columns (the original vocabulary)


def resolve_head(model: torch.nn.Module) -> VocabShardHead:
    """The vocabulary shard of the target's LM head.

    Raises when the model's logits are not ``logits_processor(lm_head, x)``
    on a vocab-parallel head without added vocabulary.
    """
    compute_logits = getattr(type(model), "compute_logits", None)
    if getattr(compute_logits, "__module__", "").startswith("vllm_torchtpu."):
        raise NotImplementedError(
            f"TPU_SAMPLER_VOCAB_PARALLEL_TARGET: {type(model).__name__} computes "
            "its logits in a TPU-specific compute_logits that the shard head "
            "does not reproduce; unset TPU_SAMPLER_VOCAB_PARALLEL_TARGET."
        )
    for module in model.modules():
        processor = getattr(module, "logits_processor", None)
        head = getattr(module, "lm_head", None)
        if hasattr(processor, "_get_logits") and hasattr(head, "shard_indices"):
            break
    else:
        raise NotImplementedError(
            f"TPU_SAMPLER_VOCAB_PARALLEL_TARGET: {type(model).__name__} has no "
            "module with a vocab-parallel lm_head and a logits_processor; unset "
            "TPU_SAMPLER_VOCAB_PARALLEL_TARGET."
        )
    assert processor is not None and head is not None
    shard = head.shard_indices
    if processor.logits_as_input or shard.num_added_elements_padded:
        raise NotImplementedError(
            "TPU_SAMPLER_VOCAB_PARALLEL_TARGET needs the logits computed from "
            "hidden states on a head without added vocabulary; unset "
            "TPU_SAMPLER_VOCAB_PARALLEL_TARGET."
        )
    start = int(shard.padded_org_vocab_start_index)
    return VocabShardHead(
        owner=module,
        vocab_start=torch.tensor(start, dtype=torch.int32).to(head.weight.device),
        num_org_local=int(shard.num_org_elements),
        width=int(shard.num_org_elements_padded),
        tp_size=int(head.tp_size),
        vocab_size=int(processor.org_vocab_size),
    )


def shard_logits(head: VocabShardHead, hidden_states: torch.Tensor) -> torch.Tensor:
    """This rank's [rows, width] columns of the logits of ``hidden_states``:
    the full path's logits before the all-gather, soft cap and scale
    included."""
    processor = cast("LogitsProcessor", head.owner.logits_processor)
    return processor(head.owner.lm_head, hidden_states, skip_gather=True)


def step_eligible(
    input_batch: InputBatch, requests: Mapping[str, CachedRequestState]
) -> bool:
    """Whether a verify step's sampling needs nothing but temperature and
    top_p of each request: not every request greedy (those take the
    all-greedy path), no logprobs, and no grammar, top_k, min_p, penalty or
    logit bias, each of which reads or masks the full logits."""
    if input_batch.all_greedy or input_batch.max_num_logprobs is not None:
        return False
    for req_id in input_batch.req_ids:
        if req_id is None:
            continue
        params = requests[req_id].sampling_params
        if params is None:
            continue
        if (
            params.structured_outputs is not None
            or params.top_k > 0
            or params.min_p > 0.0
            or params.presence_penalty != 0.0
            or params.frequency_penalty != 0.0
            or params.repetition_penalty != 1.0
            or params.logit_bias
        ):
            return False
    return True


def log_step_path(req_ids: Sequence[str | None]) -> None:
    """Logs once that a served verify step sampled on the vocabulary shard.

    Steps holding only the runner's synthetic warmup requests (ids
    ``__..__``) are skipped, so the line marks served traffic.
    """
    if all(req_id is None or req_id.startswith("__") for req_id in req_ids):
        return
    logger.info_once(
        "TPU_SAMPLER_VOCAB_PARALLEL_TARGET: a speculative verify step sampled "
        "on the vocabulary shard"
    )
