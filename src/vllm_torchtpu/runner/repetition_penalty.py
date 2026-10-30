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
"""``repetition_penalty`` for the TPU sampler.

The TPU runner samples on device without running vLLM's GPU ``Sampler``
penalties pass. Requests with ``repetition_penalty != 1.0`` are tracked by
request id (``RepetitionPenaltyTracker``) so batch condensing and RPA request
reordering never misalign a request's token history with its logits row.

For each tracked request, ``seen_mask`` marks every in-range vocabulary id
``[0, vocab_size)`` present in ``prompt_token_ids`` or generated
``output_token_ids`` (out-of-range ids such as multimodal placeholders or text
tokens wider than a codec head are ignored). Positive logits on seen tokens are
divided by ``penalty`` and non-positive logits are multiplied by ``penalty`` in
``float32`` before casting back to ``logits.dtype``.

Under plain ``async_scheduling`` (no speculative drafts), ``output_token_ids``
ends with one optimistic ``0`` placeholder while the previous step's sampled
token is still on device in ``next_tokens_tpu``. Rather than stalling the host
on a D2H copy or falsely penalizing token ``0``, the host reads
``output_token_ids[:-1]`` and ``mark_pending_async_tokens`` scatters the single
on-device pending token into ``seen_mask`` before scaling logits.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm import SamplingParams


@dataclass
class RepetitionPenaltyRequest:
    """Per-request repetition penalty config and incremental token mask."""

    penalty: float
    prompt_ids: np.ndarray
    _vocab_size: int = field(default=0, init=False, repr=False)
    _prompt_mask: np.ndarray | None = field(default=None, init=False, repr=False)
    _seen_mask: np.ndarray | None = field(default=None, init=False, repr=False)
    _committed_len: int = field(default=0, init=False, repr=False)

    def seen_mask(self, committed_output: Sequence[int], vocab_size: int) -> np.ndarray:
        """Return the boolean ``[vocab_size]`` presence mask over prompt and
        committed output tokens, updating incrementally as outputs grow."""
        if self._prompt_mask is None or self._vocab_size != vocab_size:
            prompt_mask = np.zeros((vocab_size,), dtype=np.bool_)
            if self.prompt_ids.size > 0:
                valid = self.prompt_ids[
                    (self.prompt_ids >= 0) & (self.prompt_ids < vocab_size)
                ]
                prompt_mask[valid] = True
            self._vocab_size = vocab_size
            self._prompt_mask = prompt_mask
            self._seen_mask = prompt_mask.copy()
            self._committed_len = 0

        cur_len = len(committed_output)
        if cur_len < self._committed_len:
            assert self._prompt_mask is not None
            self._seen_mask = self._prompt_mask.copy()
            self._committed_len = 0

        assert self._seen_mask is not None
        if cur_len > self._committed_len:
            delta = committed_output[self._committed_len : cur_len]
            if len(delta) <= 8:
                for tok in delta:
                    if 0 <= tok < vocab_size:
                        self._seen_mask[tok] = True
            else:
                arr = np.asarray(delta, dtype=np.int64)
                valid = arr[(arr >= 0) & (arr < vocab_size)]
                self._seen_mask[valid] = True
            self._committed_len = cur_len

        return self._seen_mask


@dataclass(frozen=True)
class RepetitionPenaltyRowState:
    """Resolved batch-row state for one step."""

    penalty: float
    entry: RepetitionPenaltyRequest
    committed_output: Sequence[int]
    pending_slot: int = -1


# persistent-batch index -> resolved repetition penalty state for this step.
RepetitionPenaltyState = dict[int, RepetitionPenaltyRowState]


class RepetitionPenaltyTracker:
    """Tracks active requests whose ``repetition_penalty != 1.0`` by id."""

    def __init__(self) -> None:
        self.requests: dict[str, RepetitionPenaltyRequest] = {}

    def __bool__(self) -> bool:
        return bool(self.requests)

    def add(
        self,
        req_id: str,
        params: SamplingParams | None,
        prompt_token_ids: Sequence[int] | None = None,
    ) -> None:
        self.requests.pop(req_id, None)
        if params is None or params.repetition_penalty == 1.0:
            return
        prompt_arr = (
            np.asarray(prompt_token_ids, dtype=np.int64)
            if prompt_token_ids
            else np.empty((0,), dtype=np.int64)
        )
        self.requests[req_id] = RepetitionPenaltyRequest(
            penalty=float(params.repetition_penalty),
            prompt_ids=prompt_arr,
        )

    def discard(self, req_ids: Iterable[str]) -> None:
        for req_id in req_ids:
            self.requests.pop(req_id, None)

    def any_in(self, req_id_to_index: Mapping[str, int]) -> bool:
        return any(req_id in req_id_to_index for req_id in self.requests)

    def batch_state(
        self,
        req_id_to_index: Mapping[str, int],
        lookup_output: Callable[[str], tuple[Sequence[int], int] | None],
    ) -> RepetitionPenaltyState:
        """Build per-batch-index state for tracked requests in the current step.

        ``lookup_output(req_id)`` returns ``(committed_output, pending_slot)``
        for active requests, or ``None`` if the request is no longer in
        ``runner.requests`` (in which case it is evicted). Requests temporarily
        absent from ``req_id_to_index`` (e.g. preempted) remain tracked.
        """
        state: RepetitionPenaltyState = {}
        for req_id, req in list(self.requests.items()):
            looked_up = lookup_output(req_id)
            if looked_up is None:
                del self.requests[req_id]
                continue
            index = req_id_to_index.get(req_id)
            if index is not None:
                committed_output, pending_slot = looked_up
                state[index] = RepetitionPenaltyRowState(
                    penalty=req.penalty,
                    entry=req,
                    committed_output=committed_output,
                    pending_slot=pending_slot,
                )
        return state


class _EmptyRepetitionPenaltyTracker(RepetitionPenaltyTracker):
    """Immutable empty sentinel used as a class-level default on runners."""

    def add(
        self,
        req_id: str,
        params: SamplingParams | None,
        prompt_token_ids: Sequence[int] | None = None,
    ) -> None:
        raise RuntimeError(
            "Cannot mutate EMPTY_TRACKER; instantiate RepetitionPenaltyTracker"
        )


EMPTY_TRACKER: RepetitionPenaltyTracker = _EmptyRepetitionPenaltyTracker()


def _in_chunk(state: RepetitionPenaltyState, start: int, end: int) -> list[int]:
    return [i for i in state if start <= i < end]


def build_plain_inputs(
    state: RepetitionPenaltyState,
    start: int,
    end: int,
    num_rows: int,
    vocab_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
    """Host-side ``(seen_mask, penalties, pre_indices)`` for a plain 1-row/req
    logits chunk, or ``None`` when no request in ``[start, end)`` is tracked."""
    indices = _in_chunk(state, start, end)
    if not indices or vocab_size <= 0:
        return None
    seen_mask = np.zeros((num_rows, vocab_size), dtype=np.bool_)
    penalties = np.ones((num_rows, 1), dtype=np.float32)
    pre_indices: np.ndarray | None = None
    for batch_idx in indices:
        row = batch_idx - start
        row_state = state[batch_idx]
        seen_mask[row] = row_state.entry.seen_mask(
            row_state.committed_output, vocab_size
        )
        penalties[row, 0] = row_state.penalty
        if row_state.pending_slot >= 0:
            if pre_indices is None:
                pre_indices = np.full((num_rows,), -1, dtype=np.int32)
            pre_indices[row] = row_state.pending_slot
    return seen_mask, penalties, pre_indices


def _fill_prefix_rows(
    seen_mask: np.ndarray,
    penalties: np.ndarray,
    base_mask: np.ndarray,
    penalty: float,
    draft_tokens: Sequence[int],
    row_indices: Sequence[int],
    draft_prefixes: Sequence[int],
    vocab_size: int,
) -> None:
    """Populate rows whose seen set is ``base_mask`` plus ``draft_tokens[:p]``."""
    for row, prefix_len in zip(row_indices, draft_prefixes):
        seen_mask[row] = base_mask
        for tok in draft_tokens[:prefix_len]:
            if 0 <= tok < vocab_size:
                seen_mask[row, tok] = True
        penalties[row, 0] = penalty


def build_verify_logits_inputs(
    state: RepetitionPenaltyState,
    start: int,
    draft_lengths: Sequence[int],
    draft_tokens_by_req: Sequence[Sequence[int]],
    num_rows: int,
    vocab_size: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Host-side ``(seen_mask, penalties)`` for an unsplit spec-verify logits
    chunk (draft positions ``0..n-1`` followed by the bonus position ``n``)."""
    indices = _in_chunk(state, start, start + len(draft_lengths))
    if not indices or vocab_size <= 0:
        return None
    offsets = np.cumsum([0, *(int(n) + 1 for n in draft_lengths)])
    seen_mask = np.zeros((num_rows, vocab_size), dtype=np.bool_)
    penalties = np.ones((num_rows, 1), dtype=np.float32)
    for batch_idx in indices:
        k = batch_idx - start
        n = int(draft_lengths[k])
        first = int(offsets[k])
        row_state = state[batch_idx]
        assert row_state.pending_slot < 0, "spec drafts must be settled first"
        base_mask = row_state.entry.seen_mask(row_state.committed_output, vocab_size)
        drafts = draft_tokens_by_req[k] if k < len(draft_tokens_by_req) else ()
        _fill_prefix_rows(
            seen_mask,
            penalties,
            base_mask,
            row_state.penalty,
            drafts,
            range(first, first + n + 1),
            range(n + 1),
            vocab_size,
        )
    return seen_mask, penalties


def build_verify_split_inputs(
    state: RepetitionPenaltyState,
    start: int,
    draft_lengths: Sequence[int],
    draft_tokens_by_req: Sequence[Sequence[int]],
    num_target_rows: int,
    num_bonus_rows: int,
    vocab_size: int,
) -> tuple[
    tuple[np.ndarray, np.ndarray] | None,
    tuple[np.ndarray, np.ndarray] | None,
]:
    """Host-side ``(target_inputs, bonus_inputs)`` after target/bonus gathers."""
    indices = _in_chunk(state, start, start + len(draft_lengths))
    if not indices or vocab_size <= 0:
        return None, None
    offsets = np.cumsum([0, *draft_lengths])
    target_mask: np.ndarray | None = None
    target_penalties: np.ndarray | None = None
    bonus_mask = np.zeros((num_bonus_rows, vocab_size), dtype=np.bool_)
    bonus_penalties = np.ones((num_bonus_rows, 1), dtype=np.float32)
    for batch_idx in indices:
        k = batch_idx - start
        n = int(draft_lengths[k])
        row_state = state[batch_idx]
        assert row_state.pending_slot < 0, "spec drafts must be settled first"
        base_mask = row_state.entry.seen_mask(row_state.committed_output, vocab_size)
        drafts = draft_tokens_by_req[k] if k < len(draft_tokens_by_req) else ()
        if n > 0:
            if target_mask is None:
                target_mask = np.zeros((num_target_rows, vocab_size), dtype=np.bool_)
                target_penalties = np.ones((num_target_rows, 1), dtype=np.float32)
            assert target_penalties is not None
            first = int(offsets[k])
            _fill_prefix_rows(
                target_mask,
                target_penalties,
                base_mask,
                row_state.penalty,
                drafts,
                range(first, first + n),
                range(n),
                vocab_size,
            )
        _fill_prefix_rows(
            bonus_mask,
            bonus_penalties,
            base_mask,
            row_state.penalty,
            drafts,
            (k,),
            (n,),
            vocab_size,
        )
    target_inputs = (
        None
        if target_mask is None or target_penalties is None
        else (target_mask, target_penalties)
    )
    return target_inputs, (bonus_mask, bonus_penalties)


def apply_penalties_table(
    logits: torch.Tensor,
    seen_mask_np: np.ndarray,
    penalties_np: np.ndarray,
    pre_indices_np: np.ndarray | None = None,
    next_tokens_tpu: torch.Tensor | None = None,
) -> torch.Tensor:
    """Stage host penalty tables to ``logits.device`` and apply scaling."""
    device = logits.device
    seen_mask = torch.from_numpy(seen_mask_np).to(device)
    if pre_indices_np is not None and next_tokens_tpu is not None:
        pre_indices = torch.from_numpy(pre_indices_np).to(device)
        seen_mask = mark_pending_async_tokens(seen_mask, next_tokens_tpu, pre_indices)
    penalties = torch.from_numpy(penalties_np).to(device)
    return apply_repetition_penalty(logits, seen_mask, penalties)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def mark_pending_async_tokens(
    seen_mask: torch.Tensor,
    next_tokens_tpu: torch.Tensor,
    pre_indices: torch.Tensor,
) -> torch.Tensor:
    return _mark_pending_async_tokens(seen_mask, next_tokens_tpu, pre_indices)


def _mark_pending_async_tokens(
    seen_mask: torch.Tensor,
    next_tokens_tpu: torch.Tensor,
    pre_indices: torch.Tensor,
) -> torch.Tensor:
    """Scatter each row's unsettled async token from ``next_tokens_tpu`` into
    ``seen_mask``. Rows with ``pre_indices == -1`` or out-of-range tokens write
    their existing column value back as a no-op."""
    vocab_size = seen_mask.shape[1]
    safe_slots = pre_indices.clamp(min=0).long()
    tok = next_tokens_tpu.index_select(0, safe_slots).long()
    valid = (pre_indices >= 0) & (tok >= 0) & (tok < vocab_size)
    col = tok.clamp(min=0, max=vocab_size - 1).unsqueeze(1)
    prev = seen_mask.gather(1, col)
    return seen_mask.scatter(1, col, prev | valid.unsqueeze(1))


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def apply_repetition_penalty(
    logits: torch.Tensor,
    seen_mask: torch.Tensor,
    penalties: torch.Tensor,
) -> torch.Tensor:
    return _apply_repetition_penalty(logits, seen_mask, penalties)


def _apply_repetition_penalty(
    logits: torch.Tensor,
    seen_mask: torch.Tensor,
    penalties: torch.Tensor,
) -> torch.Tensor:
    """Scale ``logits`` in ``float32`` by ``1 / penalties`` where ``logits > 0``
    and ``penalties`` otherwise on ``seen_mask`` positions, preserving dtype."""
    logits_f32 = logits.to(torch.float32)
    row_penalties = torch.where(seen_mask, penalties, 1.0)
    scaling = torch.where(logits_f32 > 0, 1.0 / row_penalties, row_penalties)
    return (logits_f32 * scaling).to(logits.dtype)
