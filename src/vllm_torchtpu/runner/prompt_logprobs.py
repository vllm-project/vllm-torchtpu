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
``GPUModelRunner._get_prompt_logprobs_dict``. The step does not wait for
their copy to the host: it lands with the step's sampled tokens
(``AsyncTPUCopyState``), and the buffers are filled when the step's output is
read.

The rows are scored with the runner's fixed top-k (``max_logprobs``), so a
request for more prompt logprobs than that gets ``max_logprobs`` of them.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from vllm.v1.outputs import LogprobsTensors

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.vllm_v1.worker.gpu_input_batch import CachedRequestState

logger = init_logger(__name__)


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


def unsupported_reason(
    *, selected_logits_hidden_states: bool, lora: bool
) -> str | None:
    """Why the runner cannot compute prompt logprobs, with the fix; None if it
    can."""
    if selected_logits_hidden_states:
        # Each rank selects its own logits rows; the prompt rows are not local.
        return (
            "prefill context parallelism (serve with --prefill-context-parallel-size 1)"
        )
    if lora:
        # The LoRA logits mapping has one adapter slot per request, not per row.
        return "LoRA (serve without --enable-lora)"
    return None


def max_logits_calls(num_rows: int, rows_per_call: int, num_chunks: int) -> int:
    """Upper bound on the logits calls needed to score `num_rows` prompt rows.

    A call scores up to `rows_per_call` rows, all from the same forward chunk,
    so each chunk's rows are scored separately and each chunk can leave one
    call partly filled. However the rows split across the `num_chunks`
    chunks, the count is at most ceil(num_rows / rows_per_call) +
    num_chunks - 1. For example, 16 rows at 8 per call take 2 calls in one
    chunk, but 3 if split 9 + 7 across two chunks.
    """
    if num_rows == 0:
        return 0
    return -(-num_rows // rows_per_call) + num_chunks - 1


def plan_chunk(
    req_ids: Sequence[str],
    query_start_loc: np.ndarray,
    num_computed_tokens: np.ndarray,
    num_scheduled_tokens: Mapping[str, int],
    requests: Mapping[str, CachedRequestState],
    num_prompt_logprobs: Mapping[str, int],
) -> list[PromptLogprobsSpan]:
    """The prompt-logprob spans of one forward chunk.

    `req_ids`, `query_start_loc` (chunk-local, `len(req_ids) + 1` entries) and
    `num_computed_tokens` describe the chunk's requests in layout order.
    """
    spans = []
    for i, req_id in enumerate(req_ids):
        if req_id not in num_prompt_logprobs:
            continue
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


@dataclass
class _Write:
    """Rows `src` of a chunk's host rows go to rows `dst` of a request's
    buffer, `width` columns wide."""

    buffer: LogprobsTensors
    src: slice
    dst: slice
    width: int


@dataclass
class PendingPromptLogprobs:
    """A step's prompt logprobs, from the end of the step until their copy to
    the host lands.

    `prepare` builds it on the runner's thread and does all the bookkeeping
    there. `fill` reads the values once the copy has landed. Under async
    scheduling it runs on the worker's output thread, concurrently with the
    next step, so it writes only to the buffers listed here.
    """

    # Per chunk: its slices (device tensors until `copy_to_host`, then their
    # host copies) and the buffer writes they feed.
    chunks: list[tuple[list[tuple[LogprobsTensors, int]], list[_Write]]]
    # The buffers of the requests whose prompt the step completes.
    completed: dict[str, LogprobsTensors]

    def copy_to_host(self) -> list[LogprobsTensors]:
        """Start copying the slices to the host without waiting. Returns their
        device tensors, which must stay alive until the copy lands."""
        sources = [t for slices, _ in self.chunks for t, _ in slices]
        self.chunks = [
            ([(t.to_cpu_nonblocking(), n) for t, n in slices], writes)
            for slices, writes in self.chunks
        ]
        return sources

    def fill(self) -> dict[str, LogprobsTensors | None]:
        """Write the copied rows into the requests' buffers, once the copy has
        landed, and return the completed requests' buffers."""
        for slices, writes in self.chunks:
            if not writes:
                continue
            host = LogprobsTensors.cat(
                [
                    LogprobsTensors(
                        t.logprob_token_ids[:n],
                        t.logprobs[:n],
                        t.selected_token_ranks[:n],
                    )
                    for t, n in slices
                ]
            )
            for w in writes:
                w.buffer.logprob_token_ids[w.dst] = host.logprob_token_ids[
                    w.src, : w.width
                ]
                w.buffer.logprobs[w.dst] = host.logprobs[w.src, : w.width]
                w.buffer.selected_token_ranks[w.dst] = host.selected_token_ranks[w.src]
        return dict(self.completed)


def prepare(
    chunks: Sequence[PromptLogprobsChunk],
    requests: Mapping[str, CachedRequestState],
    num_prompt_logprobs: MutableMapping[str, int],
) -> PendingPromptLogprobs | None:
    """Plan where a step's prompt logprobs go, without reading them.

    Creates a request's CPU buffer at its first chunk. A request whose prompt
    this step completes leaves `num_prompt_logprobs` and its request state,
    and its buffer moves to the result. None for a step without prompt
    logprobs.
    """
    if not chunks:
        return None
    pending = PendingPromptLogprobs(chunks=[], completed={})
    for chunk in chunks:
        # The rows' top-k width, known from their shape before their values.
        num_columns = chunk.slices[0][0].logprobs.shape[1] if chunk.slices else None
        writes = []
        offset = 0
        for span in chunk.spans:
            request = requests[span.req_id]
            # The prompt token's logprob, then the top-k.
            width = num_prompt_logprobs[span.req_id] + 1
            if num_columns is not None and num_columns < width:
                logger.warning_once(
                    "prompt_logprobs=%d exceeds the TPU runner's top-k of %d "
                    "(--max-logprobs); returning that many. Set a finite "
                    "--max-logprobs of at least the requested number.",
                    width - 1,
                    num_columns - 1,
                )
                width = num_columns
            buffer = request.in_progress_prompt_logprobs_cpu
            if buffer is None:
                assert request.prompt_token_ids is not None
                buffer = LogprobsTensors.empty_cpu(
                    len(request.prompt_token_ids) - 1, width
                )
                request.in_progress_prompt_logprobs_cpu = buffer
            if span.num_rows:
                writes.append(
                    _Write(
                        buffer=buffer,
                        src=slice(offset, offset + span.num_rows),
                        dst=slice(
                            span.prompt_row_start,
                            span.prompt_row_start + span.num_rows,
                        ),
                        width=width,
                    )
                )
                offset += span.num_rows
            if span.completes_prompt:
                pending.completed[span.req_id] = buffer
                request.in_progress_prompt_logprobs_cpu = None
                del num_prompt_logprobs[span.req_id]
        pending.chunks.append((chunk.slices, writes))
    return pending
