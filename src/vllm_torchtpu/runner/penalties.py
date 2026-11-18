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
"""Penalty inputs for the TPU sampler, read from vLLM's ``InputBatch``.

vLLM's GPU sampler applies repetition, frequency and presence penalties after
the logits processors and before sampling, for greedy and random requests
alike. The TPU runner samples on its own, so it applies them itself
(``layers/adapter/sample/penalties.py``) to each one-row-per-request chunk,
through a ``chunk_processors.PlainChunkProcessor`` (``PenaltiesChunk``).

Per-row penalties come from ``InputBatch.*_penalties_cpu``, filled from
``SamplingParams``, so ``generation_config`` defaults apply. Each penalized
row's history is ``token_ids_cpu[i, :num_tokens_no_spec[i]]`` with its output
starting at ``num_prompt_tokens[i]``; async placeholders must be settled
before it is read. The history goes to the device as encoded
``[rows, PENALTY_HISTORY_WIDTH]`` int32 tables, one per pass; rows without
penalties stay all padding and neutral.

The penalized logits are float32, as in vLLM's sampler (a bfloat16 result
would round small penalties away). That is the one exception to the
processor contract's same-dtype rule: processors after it and the sampler
also have float32 graphs warmed (``_sampler_logits_dtypes`` in the runner).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from vllm_torchtpu.layers.adapter.sample.penalties import (
    PAD_TOKEN,
    PENALTY_HISTORY_WIDTH,
)

if TYPE_CHECKING:
    import torch
    from vllm.v1.worker.gpu_input_batch import InputBatch

# Columns of the per-row parameter table.
NUM_PENALTY_PARAMS = 3  # repetition, presence, frequency


def penalized_rows(input_batch: InputBatch, start: int, end: int) -> np.ndarray:
    """Indices (relative to ``start``) of the rows ``start .. end - 1`` that
    use a non-default penalty."""
    rep = input_batch.repetition_penalties_cpu[start:end]
    pres = input_batch.presence_penalties_cpu[start:end]
    freq = input_batch.frequency_penalties_cpu[start:end]
    return np.flatnonzero((rep != 1.0) | (pres != 0.0) | (freq != 0.0))


def num_passes(input_batch: InputBatch, start: int, rows: np.ndarray) -> int:
    """Encoded tables the longest history among ``rows`` needs (at least 1)."""
    longest = int(input_batch.num_tokens_no_spec[start + rows].max())
    return max(1, -(-longest // PENALTY_HISTORY_WIDTH))


def fill_params(
    out: np.ndarray, input_batch: InputBatch, start: int, rows: np.ndarray
) -> None:
    """Writes ``(repetition, presence, frequency)`` of ``rows`` into the
    ``[num_rows, NUM_PENALTY_PARAMS]`` float32 ``out``; other rows neutral."""
    out[:, 0] = 1.0
    out[:, 1:] = 0.0
    index = start + rows
    out[rows, 0] = input_batch.repetition_penalties_cpu[index]
    out[rows, 1] = input_batch.presence_penalties_cpu[index]
    out[rows, 2] = input_batch.frequency_penalties_cpu[index]


def fill_history_tables(
    out: list[np.ndarray], input_batch: InputBatch, start: int, rows: np.ndarray
) -> None:
    """Writes the encoded histories of ``rows`` into ``out``, one ``[num_rows,
    PENALTY_HISTORY_WIDTH]`` int32 table per pass (``num_passes`` of them):
    pass p holds tokens ``p * width .. (p + 1) * width - 1`` of each row."""
    width = PENALTY_HISTORY_WIDTH
    for table in out:
        table.fill(PAD_TOKEN)
    for row in rows.tolist():
        index = start + row
        length = int(input_batch.num_tokens_no_spec[index])
        prompt = int(input_batch.num_prompt_tokens[index])
        tokens = input_batch.token_ids_cpu[index, :length].astype(np.int32)
        tokens[prompt:] = -tokens[prompt:] - 2  # output tokens, see PAD_TOKEN
        for p, begin in enumerate(range(0, length, width)):
            segment = tokens[begin : begin + width]
            out[p][row, : segment.size] = segment


class PenaltiesChunk:
    """``chunk_processors.PlainChunkProcessor`` for one chunk with a penalized
    row. ``penalize`` maps the chunk's ``[padded_num_reqs, vocab]`` logits to
    float32 penalized logits (``TPUModelRunner._apply_penalties``). Penalties
    keep no state on the device, and ``greedy_top_tokens.step_eligible`` keeps
    a step with a penalized request on the full logits."""

    def __init__(self, penalize: Callable[[torch.Tensor], torch.Tensor]):
        self._penalize = penalize

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        return self._penalize(logits)

    def override_top_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "a step with a penalized request takes the full logits "
            "(greedy_top_tokens.step_eligible)"
        )

    def on_sampled(self, token_ids: torch.Tensor) -> None:
        return None
