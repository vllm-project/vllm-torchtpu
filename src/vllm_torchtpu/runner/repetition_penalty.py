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
        lookup_output: Callable[[str], Sequence[int] | None],
    ) -> RepetitionPenaltyState:
        """Build per-batch-index state for tracked requests in the current step.

        ``lookup_output(req_id)`` returns the committed output token ids for
        active requests, or ``None`` if the request is no longer in
        ``runner.requests`` (in which case it is evicted). Requests temporarily
        absent from ``req_id_to_index`` (e.g. preempted) remain tracked.
        """
        state: RepetitionPenaltyState = {}
        for req_id, req in list(self.requests.items()):
            committed_output = lookup_output(req_id)
            if committed_output is None:
                del self.requests[req_id]
                continue
            index = req_id_to_index.get(req_id)
            if index is not None:
                state[index] = RepetitionPenaltyRowState(
                    penalty=req.penalty,
                    entry=req,
                    committed_output=committed_output,
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
) -> tuple[np.ndarray, np.ndarray] | None:
    """Host-side ``(seen_mask, penalties)`` for a plain 1-row/req logits chunk,
    or ``None`` when no request in ``[start, end)`` is tracked."""
    indices = _in_chunk(state, start, end)
    if not indices or vocab_size <= 0:
        return None
    seen_mask = np.zeros((num_rows, vocab_size), dtype=np.bool_)
    penalties = np.ones((num_rows, 1), dtype=np.float32)
    for batch_idx in indices:
        row = batch_idx - start
        row_state = state[batch_idx]
        seen_mask[row] = row_state.entry.seen_mask(
            row_state.committed_output, vocab_size
        )
        penalties[row, 0] = row_state.penalty
    return seen_mask, penalties


def apply_penalties_table(
    logits: torch.Tensor,
    seen_mask_np: np.ndarray,
    penalties_np: np.ndarray,
) -> torch.Tensor:
    """Stage host penalty tables to ``logits.device`` and apply scaling."""
    device = logits.device
    seen_mask = torch.from_numpy(seen_mask_np).to(device)
    penalties = torch.from_numpy(penalties_np).to(device)
    return apply_repetition_penalty(logits, seen_mask, penalties)


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
