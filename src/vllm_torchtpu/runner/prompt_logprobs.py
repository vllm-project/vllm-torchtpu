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
"""Prompt logprobs (``SamplingParams.prompt_logprobs``) for the TPU runner.

vLLM's completions endpoint sets them for ``echo=True`` with ``logprobs``,
which is how loglikelihood evaluations score a continuation. Row ``i`` of a
prompt of ``P`` tokens scores prompt token ``i + 1`` from the hidden state of
token ``i``, so a prompt has ``P - 1`` rows. Under chunked prefill they are
filled chunk by chunk into a per-request CPU buffer, which is returned once
the prompt is computed: the semantics of vLLM's
``GPUModelRunner._get_prompt_logprobs_dict``.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu_input_batch import CachedRequestState


@dataclass(frozen=True)
class PromptLogprobsSpan:
    """One request's prompt-logprob rows within one forward chunk."""

    req_id: str
    # The span's first row in the chunk's hidden states.
    hidden_start: int
    # The span's first row in the request's prompt-logprobs buffer: the
    # number of tokens computed before this chunk.
    prompt_row_start: int
    num_rows: int
    # This chunk completes the request's prompt rows.
    completes_prompt: bool


@dataclass
class PromptLogprobsChunk:
    """A chunk's spans and the device logprobs of their rows, in span order.

    The rows go through the device in slices; each slice's tensors are padded
    to a precompiled request bucket and carry their real row count.
    """

    spans: list[PromptLogprobsSpan]
    slices: list[tuple[LogprobsTensors, int]] = field(default_factory=list)


def plan_chunk(
    req_ids: Sequence[str],
    query_start_loc: np.ndarray,
    num_computed_tokens: np.ndarray,
    num_scheduled_tokens: Mapping[str, int],
    requests: Mapping[str, CachedRequestState],
    num_prompt_logprobs: Mapping[str, int],
    max_logprobs: int,
) -> list[PromptLogprobsSpan]:
    """The prompt-logprob spans of one forward chunk.

    `req_ids`, `query_start_loc` (chunk-local, `len(req_ids) + 1` entries) and
    `num_computed_tokens` describe the chunk's requests in layout order.
    """
    spans = []
    for i, req_id in enumerate(req_ids):
        num_logprobs = num_prompt_logprobs.get(req_id)
        if num_logprobs is None:
            continue
        if num_logprobs > max(max_logprobs, 0):
            raise NotImplementedError(
                f"prompt_logprobs={num_logprobs} exceeds the TPU runner's fixed "
                f"top-k of max_logprobs={max_logprobs}; full-vocabulary prompt "
                "logprobs (-1) are not supported. Request at most max_logprobs, "
                "or raise --max-logprobs."
            )
        prompt_token_ids = requests[req_id].prompt_token_ids
        if prompt_token_ids is None:
            # As vLLM: no prompt logprobs for prompt embeddings.
            continue
        start = int(num_computed_tokens[i])
        num_remaining = len(prompt_token_ids) - (start + 1)
        num_scheduled = num_scheduled_tokens[req_id]
        # A chunk that ends one token short of the prompt has scored every row,
        # but returns them with the next chunk, as vLLM does.
        completes_prompt = num_scheduled > num_remaining
        spans.append(
            PromptLogprobsSpan(
                req_id=req_id,
                hidden_start=int(query_start_loc[i]),
                prompt_row_start=start,
                num_rows=max(num_remaining, 0) if completes_prompt else num_scheduled,
                completes_prompt=completes_prompt,
            )
        )
    return spans


def rows_and_targets(
    spans: Sequence[PromptLogprobsSpan],
    requests: Mapping[str, CachedRequestState],
) -> tuple[np.ndarray, np.ndarray]:
    """The hidden-state rows to score and the prompt token each row predicts."""
    rows, targets = [], []
    for span in spans:
        prompt_token_ids = requests[span.req_id].prompt_token_ids
        assert prompt_token_ids is not None
        first = span.prompt_row_start + 1
        rows.append(np.arange(span.hidden_start, span.hidden_start + span.num_rows))
        targets.append(np.asarray(prompt_token_ids[first : first + span.num_rows]))
    return (
        np.concatenate(rows).astype(np.int32),
        np.concatenate(targets).astype(np.int64),
    )


def emit(
    chunks: Sequence[PromptLogprobsChunk],
    requests: Mapping[str, CachedRequestState],
    num_prompt_logprobs: MutableMapping[str, int],
) -> dict[str, LogprobsTensors | None]:
    """Copy a step's prompt logprobs to the requests' CPU buffers.

    Returns the buffers of the requests whose prompt this step completed, and
    removes those requests from `num_prompt_logprobs`.
    """
    completed: dict[str, LogprobsTensors | None] = {}
    for chunk in chunks:
        host = _host_rows(chunk.slices) if chunk.slices else None
        offset = 0
        for span in chunk.spans:
            request = requests[span.req_id]
            width = num_prompt_logprobs[span.req_id] + 1
            buffer = request.in_progress_prompt_logprobs_cpu
            if buffer is None:
                assert request.prompt_token_ids is not None
                buffer = LogprobsTensors.empty_cpu(
                    len(request.prompt_token_ids) - 1, width
                )
                request.in_progress_prompt_logprobs_cpu = buffer
            if span.num_rows:
                assert host is not None
                rows = slice(offset, offset + span.num_rows)
                dst = slice(
                    span.prompt_row_start, span.prompt_row_start + span.num_rows
                )
                buffer.logprob_token_ids[dst] = host.logprob_token_ids[rows, :width]
                buffer.logprobs[dst] = host.logprobs[rows, :width]
                buffer.selected_token_ranks[dst] = host.selected_token_ranks[rows]
                offset += span.num_rows
            if span.completes_prompt:
                completed[span.req_id] = buffer
                request.in_progress_prompt_logprobs_cpu = None
                del num_prompt_logprobs[span.req_id]
    return completed


def _host_rows(slices: Sequence[tuple[LogprobsTensors, int]]) -> LogprobsTensors:
    """The real rows of a chunk's padded device slices, on the host, in order."""
    return LogprobsTensors(
        torch.cat([t.logprob_token_ids.cpu()[:n] for t, n in slices]),
        torch.cat([t.logprobs.cpu()[:n] for t, n in slices]),
        torch.cat([t.selected_token_ranks.cpu()[:n] for t, n in slices]),
    )
