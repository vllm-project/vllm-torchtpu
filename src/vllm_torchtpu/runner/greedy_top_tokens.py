# SPDX-License-Identifier: Apache-2.0
"""Greedy steps sampled from the vocabulary-parallel argmax.

With ``TPU_GREEDY_TOP_TOKENS``, a step whose requests are all greedy, ask for
no logprobs and carry no grammar takes vLLM's
``LogitsProcessor.get_top_tokens``: each rank argmaxes its slice of the LM
head and only the (value, index) pairs are exchanged, so the full logits are
never gathered. Ties resolve to the lowest token id, the same as
``torch.argmax`` over the full logits: the local max keeps the first index of
its slice and the cross-rank argmax keeps the first rank, which owns the
lower ids.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

import torch

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

logger = init_logger(__name__)


def resolve_head_owner(model: torch.nn.Module, dp_lockstep: bool) -> torch.nn.Module:
    """The module holding ``lm_head`` and its ``logits_processor``.

    Qwen3.5's wrapper keeps both on ``language_model``. Raises when the model
    cannot be served by the vocabulary-parallel argmax.
    """
    compute_logits = getattr(type(model), "compute_logits", None)
    if getattr(compute_logits, "__module__", "").startswith("vllm_torchtpu."):
        raise NotImplementedError(
            f"TPU_GREEDY_TOP_TOKENS: {type(model).__name__} computes its logits "
            "in a TPU-specific compute_logits that get_top_tokens does not "
            "reproduce; unset TPU_GREEDY_TOP_TOKENS for this model."
        )
    for module in model.modules():
        processor = getattr(module, "logits_processor", None)
        head = getattr(module, "lm_head", None)
        if hasattr(processor, "get_top_tokens") and hasattr(head, "shard_indices"):
            break
    else:
        raise NotImplementedError(
            f"TPU_GREEDY_TOP_TOKENS: {type(model).__name__} has no module with "
            "an lm_head and a logits_processor.get_top_tokens; unset "
            "TPU_GREEDY_TOP_TOKENS for this model."
        )
    assert processor is not None and head is not None
    if processor.scale <= 0.0:
        raise NotImplementedError(
            "TPU_GREEDY_TOP_TOKENS: the argmax reduction needs a positive "
            f"logit scale, got {processor.scale}; unset TPU_GREEDY_TOP_TOKENS."
        )
    if dp_lockstep and head.tp_size > 1:
        # An idle engine issues the full-logits collective in lockstep with
        # its busy peers, which would be waiting in the pair exchange.
        raise NotImplementedError(
            "TPU_GREEDY_TOP_TOKENS with expert-parallel data parallelism and a "
            "tensor-parallel LM head is not supported; unset "
            "TPU_GREEDY_TOP_TOKENS or use tensor parallelism alone."
        )
    return module


def top_tokens(owner: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """The greedy token of each row, ``[rows, 1]`` int64 like
    ``torch.argmax(logits, dim=-1, keepdim=True)``."""
    processor = cast("LogitsProcessor", owner.logits_processor)
    head = cast("VocabParallelEmbedding", owner.lm_head)
    tokens = processor.get_top_tokens(head, hidden_states)
    return tokens.to(torch.int64).view(-1, 1)


def log_step_path(is_spec_step: bool, req_ids: Sequence[str | None]) -> None:
    """Logs, once per path, that a step's tokens came from the top tokens.

    Steps holding only the runner's synthetic warmup requests (ids
    ``__..__``) are skipped, so the line marks served traffic.
    """
    if all(req_id is None or req_id.startswith("__") for req_id in req_ids):
        return
    if is_spec_step:
        logger.info_once(
            "TPU_GREEDY_TOP_TOKENS: a speculative verify step took the top tokens"
        )
    else:
        logger.info_once(
            "TPU_GREEDY_TOP_TOKENS: a non-speculative step took the top tokens"
        )


def step_eligible(
    input_batch: InputBatch, requests: Mapping[str, CachedRequestState]
) -> bool:
    """Whether the step's tokens are fully determined by the top tokens:
    every request greedy, no logprobs and no grammar, which needs the full
    logits to mask."""
    if not input_batch.all_greedy or input_batch.max_num_logprobs is not None:
        return False
    for req_id in input_batch.req_ids:
        if req_id is None:
            continue
        params = requests[req_id].sampling_params
        if params is not None and params.structured_outputs is not None:
            logger.warning_once(
                "TPU_GREEDY_TOP_TOKENS: steps with a structured-output request "
                "gather the full logits"
            )
            return False
    return True
