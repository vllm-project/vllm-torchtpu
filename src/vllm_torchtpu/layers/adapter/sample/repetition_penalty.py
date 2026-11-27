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
"""``repetition_penalty`` scaling of the logits, on device.

The seen tokens of each row come as a packed bitmask (``token_bitmask``).
``vllm_torchtpu.runner.repetition_penalty`` builds it on the host.
"""

import torch

from vllm_torchtpu.compilation.fast_dispatch import tpu_compile
from vllm_torchtpu.layers.adapter.sample.token_bitmask import unpack_token_bitmask


@tpu_compile
def apply_repetition_penalty(
    logits: torch.Tensor,
    seen_bitmask: torch.Tensor,
    penalties: torch.Tensor,
) -> torch.Tensor:
    return _apply_repetition_penalty(logits, seen_bitmask, penalties)


def _apply_repetition_penalty(
    logits: torch.Tensor,
    seen_bitmask: torch.Tensor,
    penalties: torch.Tensor,
) -> torch.Tensor:
    """Unpack ``seen_bitmask`` (``[rows, cdiv(vocab_size, 32)]`` ``int32``) and
    scale ``logits`` in ``float32`` by ``1 / penalties`` where ``logits > 0`` and
    ``penalties`` otherwise on seen positions, preserving ``logits.dtype``.

    ``logits`` can be wider than the bitmask (columns after ``vocab_size``,
    padded for hardware alignment, as ``apply_grammar_bitmask`` accepts). No
    token id maps to those columns, so they are not seen and stay unchanged.
    """
    shifts = torch.arange(32, dtype=torch.int32, device=seen_bitmask.device)
    seen_mask = unpack_token_bitmask(seen_bitmask, shifts, logits.shape[-1])
    num_unmapped = logits.shape[-1] - seen_mask.shape[-1]
    if num_unmapped > 0:
        seen_mask = torch.nn.functional.pad(seen_mask, (0, num_unmapped), value=False)
    scaled = scale_seen_logits(logits.to(torch.float32), seen_mask, penalties)
    return scaled.to(logits.dtype)


def scale_seen_logits(
    logits: torch.Tensor, seen: torch.Tensor, penalties: torch.Tensor
) -> torch.Tensor:
    """vLLM's ``repetition_penalty`` on ``logits [rows, vocab]``.

    Where ``seen`` is set, positive logits are multiplied by ``1 / penalties``
    and the others by ``penalties`` (``[rows, 1]``), as vLLM's torch path
    (``apply_repetition_penalties_torch``) does; vLLM's CUDA kernel divides
    instead. The reciprocal is taken per row, before the broadcast to
    ``vocab``. The exported sampler (``experimental/export/sampling.py``)
    calls this function too.
    """
    scale = torch.where(logits > 0, 1.0 / penalties, penalties)
    return torch.where(seen, logits * scale, logits)
