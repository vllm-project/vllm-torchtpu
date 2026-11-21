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
"""Argmax over a vocabulary-parallel score matrix without gathering it.

Each TP rank holds the scores of its contiguous vocab shard (rank r owns the
ids from ``vocab_start(r)`` on, increasing with r). A rank reduces its shard to
one ``(max, global id)`` pair per row; the ranks all-gather the pairs
(``[rows, 2]`` each) and every rank picks the same winner. The result equals
``torch.argmax`` over the gathered scores, ties included: the local max keeps
the first index of its shard, and the cross-rank argmax keeps the first rank,
which owns the lower ids. A NaN wins like it does in ``torch.argmax``: the
rank's max is the NaN at its first NaN index, and the first such rank wins.

Same reduction as vLLM's ``LogitsProcessor.get_top_tokens``, split in two so a
caller can add a per-row bias to the shard first (the DSpark Markov head) and
so the two halves can be tested without a process group.
"""

from __future__ import annotations

import torch

# float32 holds every integer id below 2**24 exactly.
MAX_VOCAB_FOR_FLOAT_IDS = 1 << 24


def shard_argmax_pair(
    local_scores: torch.Tensor, vocab_start: torch.Tensor | int
) -> torch.Tensor:
    """``[rows, 2]`` float32 ``(max score, global id)`` of this rank's shard.

    ``vocab_start`` is the shard's first global id; pass it as a tensor so the
    traced program does not bake a rank-specific constant.
    """
    local_max, local_idx = local_scores.max(dim=-1)
    global_idx = local_idx + vocab_start
    return torch.stack([local_max.float(), global_idx.float()], dim=-1)


def reduce_shard_argmax_pairs(gathered: torch.Tensor, tp_size: int) -> torch.Tensor:
    """Global argmax ids (int64 ``[rows]``) from the rank-major all-gather of
    every rank's ``shard_argmax_pair`` (``[rows, 2 * tp_size]``)."""
    pairs = gathered.view(gathered.shape[0], tp_size, 2)
    best_rank = pairs[..., 0].argmax(dim=-1, keepdim=True)
    return pairs[..., 1].gather(-1, best_rank).squeeze(-1).to(torch.int64)
