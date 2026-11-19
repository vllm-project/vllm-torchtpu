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

``TPUModelRunner._sample_plain_chunk`` runs ``PlainChunkProcessor`` hooks on a
chunk with one logits row per request (plain decode, and the draft-less chunks
of speculative steps); ``TPUModelRunner._sample_spec_verify_chunk`` runs
``SpecChunkProcessor`` hooks on a verify chunk. The thinking-token budget
(``runner/thinking_budget_tracker.py``) implements both.

Why not upstream ``LogitsProcessor`` (``vllm.v1.sample.logits_processor``):
it is applied once per step to the full batch's logits and keeps its state on
the host, updated through ``update_state``. The TPU runner samples each chunk
separately, on bucket-padded device tensors inside compiled graphs, so a
stateful processor has to advance its state on device right after sampling,
without a host sync. The runner also has a speculative verify path (target and
bonus rows plus the drafts to verify) and a ``TPU_GREEDY_TOP_TOKENS`` path with
no full logits, which ``LogitsProcessor`` does not cover.

Construction: a processor's owner builds one handle per chunk and binds the
chunk's context into it. It maps the chunk's rows to its own per-request state
and maps padded rows and partial-prefill rows (whose sampled token is
discarded) to a no-op entry, so the hooks need only the bucketed tensors, as
``ThinkingBudgetStepState.plain_chunk`` / ``spec_chunk`` do.

Device rules for every hook:

- Inputs are device tensors with bucket padding. They include padded rows and
  the rows of partial-prefill requests.
- Do not read them on the host (``.item()``, ``.tolist()``, ``bool(...)``):
  that stops the step for a device-to-host copy.
- Return tensors with the input shapes and dtypes. Another shape or dtype
  compiles a new ``dynamic=False`` graph while serving.
- Warm up compiled functions from the runner's precompile through the owner,
  as ``TPUModelRunner._precompile_thinking_budget`` does.

Top tokens: on the ``TPU_GREEDY_TOP_TOKENS`` path the runner calls
``override_top_tokens`` / ``prepare_tokens`` instead of ``apply`` /
``prepare_logits``. A processor that cannot rewrite tokens must keep its steps
on full logits before the forward pass, in
``TPUModelRunner._greedy_top_tokens_this_step``, as ``min_tokens`` does with
``MinTokensTracker.any_in``.
"""

from __future__ import annotations

from typing import Protocol

import torch


class PlainChunkProcessor(Protocol):
    """Hooks for a chunk with one row per request.

    Full logits: ``apply`` runs after the grammar and ``min_tokens`` masks
    (``logprob_logits`` is taken before both), the runner samples, then
    ``on_sampled`` runs. Top tokens: ``override_top_tokens`` rewrites the
    greedy tokens, then ``on_sampled`` runs. Processors run in order, each on
    the previous one's output, and ``on_sampled`` gets the final tokens.
    """

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Rewrites ``[padded_num_reqs, vocab]`` logits."""
        ...

    def override_top_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Rewrites ``[padded_num_reqs, 1]`` greedy token IDs. Leaves the
        processor's state to ``on_sampled``."""
        ...

    def on_sampled(self, token_ids: torch.Tensor) -> None:
        """Advances the processor's state with the chunk's
        ``[padded_num_reqs, 1]`` tokens."""
        ...


class SpecChunkProcessor(Protocol):
    """Hooks around one verify chunk's rejection sampling.

    Target rows are ``[padded_logits_length, ...]`` and bonus rows
    ``[padded_num_reqs, ...]``. ``draft_token_ids``, ``segment_ids`` and
    ``group_indices`` are ``[padded_logits_length]``; padded rows have
    ``segment_ids == num_reqs`` and ``group_indices == 0``.
    ``max_draft_tokens`` is the speculative config's
    ``num_speculative_tokens``.

    Full logits: ``prepare_logits`` rewrites the target and bonus logits, after
    the grammar and ``min_tokens`` masks and before the bonus token is sampled.
    Top tokens: ``prepare_tokens`` rewrites the target argmax and bonus token
    IDs. Both return the drafts to verify, with ``-1`` for drafts the
    processor rejects. ``finish`` then runs once on the rejection sampler's
    output.
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

    def finish(self, next_tokens: torch.Tensor) -> None:
        """Advances the processor's state with the rejection sampler's
        ``[padded_num_reqs, max_draft_tokens + 1]`` output: each row's accepted
        tokens, then ``-1``. Padded rows hold their bonus token in column 0."""
        ...
