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
from vllm.lora.layers import LogitsProcessorWithLoRA

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    from vllm_torchtpu.runner.vllm_v1.worker.gpu_input_batch import (
        CachedRequestState,
        InputBatch,
    )

logger = init_logger(__name__)


class _LogitsProbe(torch.nn.Module):
    """Sentinel logits_processor used to verify passthrough ``compute_logits``."""

    def __init__(self, expected_head: torch.nn.Module) -> None:
        super().__init__()
        self.expected_head = expected_head
        self.sentinel_in = object()
        self.sentinel_out = object()
        self.calls = 0
        self.matched = False

    def forward(
        self,
        lm_head: torch.nn.Module,
        hidden_states: object,
        embedding_bias: torch.Tensor | None = None,
    ) -> object:
        expected_bias = getattr(self.expected_head, "bias", None)
        self.calls += 1
        self.matched = (
            lm_head is self.expected_head
            and hidden_states is self.sentinel_in
            and embedding_bias is expected_bias
        )
        return self.sentinel_out


def _verify_passthrough_compute_logits(
    model: torch.nn.Module, owner: torch.nn.Module
) -> bool:
    """True when ``model.compute_logits`` is absent or passes straight through
    ``owner.logits_processor(owner.lm_head, hidden_states, lm_head.bias)``."""
    compute_logits = getattr(model, "compute_logits", None)
    if not callable(compute_logits):
        return True
    probe = _LogitsProbe(cast(torch.nn.Module, owner.lm_head))
    orig_processor = owner.logits_processor
    owner.logits_processor = probe
    try:
        out = compute_logits(probe.sentinel_in)
    except Exception:
        # Non-passthrough compute_logits inspect tensor attrs/ops on the
        # object() sentinels and raise; fail closed to NotImplementedError.
        out = None
    finally:
        owner.logits_processor = orig_processor
    return probe.calls == 1 and probe.matched and out is probe.sentinel_out


def resolve_head_owner(model: torch.nn.Module, dp_lockstep: bool) -> torch.nn.Module:
    """The module holding ``lm_head`` and its ``logits_processor``.

    Qwen3.5's wrapper keeps both on ``language_model``. Raises when the model
    cannot be served by the vocabulary-parallel argmax, including when
    ``compute_logits`` performs pre- or post-processing beyond
    ``logits_processor(lm_head, hidden_states, lm_head.bias)``.
    """
    for module in model.modules():
        processor = getattr(module, "logits_processor", None)
        head = getattr(module, "lm_head", None)
        # vLLM's LoRA wrappers forward attribute lookups to the layer they
        # wrap, so checking the instance would find the base layer's
        # get_top_tokens and drop the wrapper's change to the logits. Checking
        # the class makes such a wrapper fail here; top_tokens handles the
        # LoRA one.
        has_top_tokens = isinstance(processor, LogitsProcessorWithLoRA) or hasattr(
            type(processor), "get_top_tokens"
        )
        if has_top_tokens and hasattr(head, "shard_indices"):
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
    if not _verify_passthrough_compute_logits(model, module):
        raise NotImplementedError(
            f"TPU_GREEDY_TOP_TOKENS: {type(model).__name__}.compute_logits "
            "performs custom pre- or post-processing beyond "
            "logits_processor(lm_head, hidden_states, lm_head.bias); "
            "unset TPU_GREEDY_TOP_TOKENS for this model."
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
    processor = cast(
        "LogitsProcessor | LogitsProcessorWithLoRA", owner.logits_processor
    )
    head = cast("VocabParallelEmbedding", owner.lm_head)
    bias = getattr(head, "bias", None)
    if isinstance(processor, LogitsProcessorWithLoRA):
        # The LoRA wrapper has no get_top_tokens, so the call would reach the
        # base layer and drop the lm_head LoRA delta. Take the argmax of the
        # wrapper's logits, which include the delta.
        return torch.argmax(processor(head, hidden_states, bias), dim=-1, keepdim=True)
    tokens = processor.get_top_tokens(head, hidden_states, bias)
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
