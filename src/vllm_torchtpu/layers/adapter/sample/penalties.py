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
"""Repetition, frequency and presence penalties on ``[rows, vocab]`` logits.

The semantics are those of ``vllm.model_executor.layers.utils.apply_penalties``:

* repetition: a token seen in the prompt or the output so far has its logit
  divided by the penalty if positive, multiplied by it otherwise;
* frequency: the logit drops by ``penalty * count`` of the token in the output;
* presence: the logit drops by ``penalty`` if the token is in the output.

The result is float32, as in vLLM's sampler, so small penalties are not
rounded away in bfloat16.

Two input layouts are supported:

* the serving runner's ``rows * PENALTY_HISTORY_WIDTH`` encoded token
  tables, counted in as many passes as a row's history
  needs, so only one table width is ever compiled;
* the export sampler's ``history`` / ``output_start`` pair
  (``penalize_logits``).
"""

from __future__ import annotations

import torch

# Tokens per encoded table. Longer histories take several tables (passes).
PENALTY_HISTORY_WIDTH = 2048

# Encoded tables hold a prompt token as its id, an output token as
# ``-(id + 2)`` and padding as -1, so one int32 table carries both counts.
PAD_TOKEN = -1


def penalize(
    logits: torch.Tensor,
    seen: torch.Tensor,
    counts: torch.Tensor,
    repetition: torch.Tensor,
    presence: torch.Tensor,
    frequency: torch.Tensor,
) -> torch.Tensor:
    """Penalized float32 logits ``[R, V]``. ``seen`` is > 0 for tokens in the
    prompt or output, ``counts`` holds output-token counts; ``repetition`` /
    ``presence`` / ``frequency`` are per-row (1 / 0 / 0 leave a row as is)."""
    logits = logits.to(torch.float32)
    rep = repetition.to(torch.float32).unsqueeze(-1)
    logits = torch.where(
        seen > 0, torch.where(logits > 0, logits / rep, logits * rep), logits
    )
    return (
        logits
        - presence.to(torch.float32).unsqueeze(-1) * (counts > 0).to(torch.float32)
        - frequency.to(torch.float32).unsqueeze(-1) * counts
    )


def _scatter_counts(
    rows: int, vocab: int, ids: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    zeros = torch.zeros((rows, vocab), dtype=torch.float32, device=ids.device)
    return zeros.scatter_add(1, ids, weights.to(torch.float32))


def penalize_logits(
    logits: torch.Tensor,
    history: torch.Tensor,
    output_start: torch.Tensor,
    repetition: torch.Tensor,
    presence: torch.Tensor,
    frequency: torch.Tensor,
) -> torch.Tensor:
    """``penalize`` with row r's prompt then output token ids in
    ``history[r]`` (padded with -1), its output starting at column
    ``output_start[r]``."""
    rows, vocab = int(logits.shape[0]), int(logits.shape[1])
    valid = history >= 0
    ids = torch.where(valid, history, torch.zeros_like(history)).to(torch.int64)
    column = torch.arange(
        int(history.shape[1]), device=logits.device, dtype=torch.int32
    ).unsqueeze(0)
    generated = valid & (column >= output_start.to(torch.int32).unsqueeze(-1))
    seen = _scatter_counts(rows, vocab, ids, valid)
    counts = _scatter_counts(rows, vocab, ids, generated)
    return penalize(logits, seen, counts, repetition, presence, frequency)


def _encoded_counts(
    rows: int, vocab: int, table: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    is_prompt = table >= 0
    is_output = table <= -2
    ids = torch.where(
        is_prompt,
        table,
        torch.where(is_output, -table - 2, torch.zeros_like(table)),
    ).to(torch.int64)
    seen = _scatter_counts(rows, vocab, ids, is_prompt | is_output)
    counts = _scatter_counts(rows, vocab, ids, is_output)
    return seen, counts


# The compiled functions take the tables and parameters flat, as the runner
# stages them (one 1-D H2D buffer each), and shape them inside the graph: a
# device-side ``.view`` of a staged buffer would reach the graph as a different
# input and compile a second program per bucket at serving time.


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def penalty_counts(
    logits: torch.Tensor, table: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """(seen, counts) of one encoded table of ``R * PENALTY_HISTORY_WIDTH``
    tokens (flat, row-major); ``logits`` only gives the shape."""
    rows, vocab = int(logits.shape[0]), int(logits.shape[1])
    return _encoded_counts(rows, vocab, table.reshape(rows, -1))


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def add_penalty_counts(
    seen: torch.Tensor, counts: torch.Tensor, table: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``penalty_counts`` of the next table, added to the running counts."""
    rows, vocab = int(seen.shape[0]), int(seen.shape[1])
    more_seen, more_counts = _encoded_counts(rows, vocab, table.reshape(rows, -1))
    return seen + more_seen, counts + more_counts


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def apply_penalty_counts(
    logits: torch.Tensor, seen: torch.Tensor, counts: torch.Tensor, params: torch.Tensor
) -> torch.Tensor:
    """``penalize`` with the flat ``R * 3`` ``params``, row ``r`` holding
    (repetition, presence, frequency)."""
    params = params.reshape(int(logits.shape[0]), -1)
    return penalize(logits, seen, counts, params[:, 0], params[:, 1], params[:, 2])
