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
"""``presence_penalty`` and ``frequency_penalty`` on device.

Both read a per-request count of each output token. The counts live on the
device in an ``int32 [num_slots + 1, vocab]`` table
(``vllm_torchtpu.runner.output_penalties`` owns it and maps requests to
slots). Row 0, the null slot, is never written and stays all zero, so a
padded row or a row without these penalties reads it and is left unchanged.

As in vLLM's ``apply_penalties``, for each row::

    logits -= frequency_penalty * count + presence_penalty * (count > 0)

The math is in ``float32``. By default the result goes back to
``logits.dtype``, as ``repetition_penalty`` does
(``layers/adapter/sample/repetition_penalty.py``), so the sampler's
precompiled graphs take it; ``FLOAT32_LOGITS`` keeps ``float32`` instead,
which vLLM samples from.
"""

import torch

# Return float32 logits instead of casting back to logits.dtype. Needs the
# sampler's float32 graphs warmed as well, so it is off.
FLOAT32_LOGITS = False


def subtract_output_penalties(
    logits: torch.Tensor,
    counts: torch.Tensor,
    presence: torch.Tensor,
    frequency: torch.Tensor,
) -> torch.Tensor:
    """vLLM's presence and frequency penalties on ``float32`` ``logits
    [rows, vocab]`` with output ``counts [rows, vocab]`` and per-row
    ``presence`` / ``frequency`` ``[rows, 1]``. The exported sampler
    (``experimental/export/sampling.py``) calls this function too."""
    counts = counts.to(torch.float32)
    return logits - frequency * counts - presence * (counts > 0).to(torch.float32)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def apply_output_penalties(
    logits: torch.Tensor,
    counts: torch.Tensor,
    slots: torch.Tensor,
    params: torch.Tensor,
) -> torch.Tensor:
    """Penalize ``logits [rows, vocab]``. Row ``r`` reads ``counts[slots[r]]``;
    ``params[r]`` is ``(presence, frequency)``. Columns past the count table's
    width (vocabulary padding) are left unchanged."""
    rows = torch.index_select(counts, 0, slots)
    width = rows.shape[-1]
    head = subtract_output_penalties(
        logits[:, :width].to(torch.float32), rows, params[:, 0:1], params[:, 1:2]
    )
    out_dtype = torch.float32 if FLOAT32_LOGITS else logits.dtype
    if width == logits.shape[-1]:
        return head.to(out_dtype)
    return torch.cat([head.to(out_dtype), logits[:, width:].to(out_dtype)], dim=-1)


# The graphs below update the count table in place, so it stays one buffer.


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def add_sampled_tokens(
    counts: torch.Tensor,
    slots: torch.Tensor,
    token_ids: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    """``counts[slots[r], token_ids[r]] += weights[r]`` for each row, in place,
    with ``token_ids [rows, 1]`` as the sampler returns them. Rows with weight
    0 (padding, discarded partial-prefill tokens, unpenalized requests) and
    out-of-range ids change nothing."""
    tokens = token_ids.reshape(-1).to(torch.int64)
    valid = (tokens >= 0) & (tokens < counts.shape[-1])
    tokens = torch.where(valid, tokens, torch.zeros_like(tokens))
    weights = torch.where(valid, weights, torch.zeros_like(weights)).to(counts.dtype)
    counts.index_put_((slots.to(torch.int64), tokens), weights, accumulate=True)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def seed_slot(
    counts: torch.Tensor, slot: torch.Tensor, token_ids: torch.Tensor
) -> None:
    """Reset row ``slot`` (``[1]``) and count ``token_ids`` (``[width]``,
    ``-1`` for padding) into it, in place."""
    counts.index_fill_(0, slot.to(torch.int64), 0)
    add_to_slot(counts, slot, token_ids)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def add_to_slot_compiled(
    counts: torch.Tensor, slot: torch.Tensor, token_ids: torch.Tensor
) -> None:
    """``add_to_slot`` as its own graph, for outputs longer than one
    ``token_ids`` width."""
    add_to_slot(counts, slot, token_ids)


def add_to_slot(
    counts: torch.Tensor, slot: torch.Tensor, token_ids: torch.Tensor
) -> None:
    """Count ``token_ids`` (``-1`` and out-of-range ids skipped) into row
    ``slot`` of ``counts``, in place."""
    valid = (token_ids >= 0) & (token_ids < counts.shape[-1])
    tokens = torch.where(valid, token_ids, torch.zeros_like(token_ids)).to(torch.int64)
    rows = slot.to(torch.int64).expand(tokens.shape[0])
    counts.index_put_((rows, tokens), valid.to(counts.dtype), accumulate=True)
