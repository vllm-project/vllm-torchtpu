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
"""Per-chunk sampling processor contracts for ``TPUModelRunner``.

``TPUModelRunner._sample_plain_chunk`` runs ``PlainChunkProcessor`` hooks on
plain decode chunks and on draft-less chunks of speculative steps;
``TPUModelRunner._sample_spec_verify_chunk`` runs ``SpecChunkProcessor`` hooks
on verify chunks. Stateful processors, such as penalties, ``bad_words`` or a
thinking-token budget, implement them; the stateless ``min_tokens`` mask is
wrapped in ``StatelessPlainChunkProcessor``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

type LogitsProcessorFn = Callable[[torch.Tensor], torch.Tensor]


class PlainChunkProcessor(Protocol):
    """Per-chunk pre-sample mask and post-sample state update hook.

    On the full-logits path (``top_tokens=False``), ``mask_logits`` runs before
    sampling and ``on_sampled`` runs after sampling. On the ``top_tokens=True``
    fast path (``TPU_GREEDY_TOP_TOKENS``), ``mask_logits`` and ``on_sampled``
    are not called; ``override_top_tokens`` must both override token IDs and
    advance any per-step processor state.
    """

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor: ...

    def on_sampled(self, selected_token_ids: torch.Tensor) -> None: ...

    def override_top_tokens(self, selected_token_ids: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class StatelessPlainChunkProcessor:
    """Adapter wrapping a stateless ``logits -> logits`` mask function."""

    mask_fn: LogitsProcessorFn

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return self.mask_fn(logits)

    def on_sampled(self, selected_token_ids: torch.Tensor) -> None:
        return None

    def override_top_tokens(self, selected_token_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "StatelessPlainChunkProcessor cannot mask pre-selected top_tokens "
            "(TPU_GREEDY_TOP_TOKENS). Keep such steps on full logits, as "
            "TPUModelRunner._greedy_top_tokens_this_step does for min_tokens, "
            "or implement a PlainChunkProcessor whose override_top_tokens "
            "rewrites the tokens."
        )


class SpecChunkProcessor(Protocol):
    """Per-chunk hooks around one speculative verify chunk's rejection sampling.

    On the full-logits path, ``prepare_logits`` masks the gathered target and
    bonus logits before the bonus token is sampled; on the
    ``TPU_GREEDY_TOP_TOKENS`` path, ``prepare_tokens`` rewrites the target
    argmax and bonus token IDs instead. Both also return the draft token IDs
    to verify, with ``-1`` for drafts the processor rejects. ``finish`` runs
    once on the rejection sampler's ``[num_reqs, max_draft_tokens + 1]``
    output to advance any per-step processor state.
    """

    def prepare_logits(
        self,
        target_logits: torch.Tensor,
        bonus_logits: torch.Tensor,
        draft_token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        group_indices: torch.Tensor,
        max_draft_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def prepare_tokens(
        self,
        target_argmax: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        draft_token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        group_indices: torch.Tensor,
        max_draft_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def finish(self, next_tokens: torch.Tensor, max_draft_tokens: int) -> None: ...
