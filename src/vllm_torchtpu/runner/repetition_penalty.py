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

For each tracked request, ``seen_bitmask`` packs presence bits over vocabulary
ids ``[0, vocab_size)`` from ``prompt_token_ids`` and generated
``output_token_ids`` into ``cdiv(vocab_size, 32)`` ``int32`` words (out-of-range
ids such as multimodal placeholders or text tokens wider than a codec head are
ignored). On device, the bitmask is unpacked and seen positions are scaled in
``float32`` by ``1 / penalty`` where ``logits > 0`` and ``penalty`` otherwise
before casting back to ``logits.dtype``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm import SamplingParams

_BIT_MASKS: np.ndarray = (np.uint32(1) << np.arange(32, dtype=np.uint32)).view(
    np.int32
)


def num_bitmask_words(vocab_size: int) -> int:
    """Number of 32-bit words needed to pack ``vocab_size`` token presence bits."""
    return (vocab_size + 31) >> 5


@dataclass
class RepetitionPenaltyRequest:
    """Per-request repetition penalty config and incremental packed bitmask."""

    penalty: float
    prompt_ids: np.ndarray
    _vocab_size: int = field(default=0, init=False, repr=False)
    _bitmask: np.ndarray | None = field(default=None, init=False, repr=False)
    _committed_len: int = field(default=0, init=False, repr=False)

    def seen_bitmask(
        self, committed_output: Sequence[int], vocab_size: int
    ) -> np.ndarray:
        """Return the ``int32`` ``[cdiv(vocab_size, 32)]`` presence bitmask over
        prompt and committed output tokens, updating incrementally as outputs grow."""
        num_words = num_bitmask_words(vocab_size)
        cur_len = len(committed_output)
        if (
            self._bitmask is None
            or self._vocab_size != vocab_size
            or cur_len < self._committed_len
        ):
            if self._bitmask is None or self._bitmask.shape[0] != num_words:
                self._bitmask = np.zeros((num_words,), dtype=np.int32)
            else:
                self._bitmask.fill(0)
            if self.prompt_ids.size > 0:
                valid = self.prompt_ids[
                    (self.prompt_ids >= 0) & (self.prompt_ids < vocab_size)
                ]
                np.bitwise_or.at(
                    self._bitmask, valid >> 5, _BIT_MASKS[valid & 31]
                )
            self._vocab_size = vocab_size
            self._committed_len = 0

        if cur_len > self._committed_len:
            bitmask = self._bitmask
            for tok in committed_output[self._committed_len : cur_len]:
                if 0 <= tok < vocab_size:
                    bitmask[tok >> 5] |= _BIT_MASKS[tok & 31]
            self._committed_len = cur_len

        return self._bitmask


# persistent-batch index -> (request entry, committed output token ids).
RepetitionPenaltyState = dict[int, tuple[RepetitionPenaltyRequest, Sequence[int]]]


class RepetitionPenaltyTracker:
    """Tracks active requests whose ``repetition_penalty != 1.0`` by id."""

    def __init__(self) -> None:
        self.requests: dict[str, RepetitionPenaltyRequest] = {}
        self._bitmask_buf: np.ndarray | None = None
        self._penalties_buf: np.ndarray | None = None
        self._dirty_rows: list[int] = []

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
            if prompt_token_ids is not None and len(prompt_token_ids) > 0
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
        stale: list[str] | None = None
        for req_id, req in self.requests.items():
            committed_output = lookup_output(req_id)
            if committed_output is None:
                if stale is None:
                    stale = []
                stale.append(req_id)
                continue
            index = req_id_to_index.get(req_id)
            if index is not None:
                state[index] = (req, committed_output)
        if stale:
            for req_id in stale:
                del self.requests[req_id]
        return state

    def build_plain_inputs(
        self,
        state: RepetitionPenaltyState,
        start: int,
        end: int,
        num_rows: int,
        vocab_size: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Stage ``(seen_bitmask, penalties)`` for a plain 1-row/req chunk into
        reusable host buffers, or return ``None`` when no row in ``[start, end)``
        is tracked."""
        indices = _in_chunk(state, start, end)
        if not indices or vocab_size <= 0:
            return None
        num_words = num_bitmask_words(vocab_size)
        if (
            self._bitmask_buf is None
            or self._penalties_buf is None
            or self._bitmask_buf.shape[0] < num_rows
            or self._bitmask_buf.shape[1] != num_words
        ):
            self._bitmask_buf = np.zeros((num_rows, num_words), dtype=np.int32)
            self._penalties_buf = np.ones((num_rows, 1), dtype=np.float32)
            self._dirty_rows.clear()
        else:
            for prev_row in self._dirty_rows:
                self._bitmask_buf[prev_row].fill(0)
                self._penalties_buf[prev_row, 0] = 1.0
            self._dirty_rows.clear()

        for batch_idx in indices:
            row = batch_idx - start
            entry, committed_output = state[batch_idx]
            self._bitmask_buf[row] = entry.seen_bitmask(
                committed_output, vocab_size
            )
            self._penalties_buf[row, 0] = entry.penalty
            self._dirty_rows.append(row)
        if self._bitmask_buf.shape[0] == num_rows:
            return self._bitmask_buf, self._penalties_buf
        return self._bitmask_buf[:num_rows], self._penalties_buf[:num_rows]


def _in_chunk(state: RepetitionPenaltyState, start: int, end: int) -> list[int]:
    return [i for i in state if start <= i < end]


def build_plain_inputs(
    state: RepetitionPenaltyState,
    start: int,
    end: int,
    num_rows: int,
    vocab_size: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Host-side ``(seen_bitmask, penalties)`` for a plain 1-row/req logits chunk,
    or ``None`` when no request in ``[start, end)`` is tracked."""
    indices = _in_chunk(state, start, end)
    if not indices or vocab_size <= 0:
        return None
    num_words = num_bitmask_words(vocab_size)
    seen_bitmask = np.zeros((num_rows, num_words), dtype=np.int32)
    penalties = np.ones((num_rows, 1), dtype=np.float32)
    for batch_idx in indices:
        row = batch_idx - start
        entry, committed_output = state[batch_idx]
        seen_bitmask[row] = entry.seen_bitmask(committed_output, vocab_size)
        penalties[row, 0] = entry.penalty
    return seen_bitmask, penalties


def apply_penalties_table(
    logits: torch.Tensor,
    seen_bitmask_np: np.ndarray,
    penalties_np: np.ndarray,
) -> torch.Tensor:
    """Stage host packed bitmask and penalty tables to ``logits.device`` and apply
    scaling."""
    device = logits.device
    seen_bitmask = torch.from_numpy(seen_bitmask_np).to(device)
    penalties = torch.from_numpy(penalties_np).to(device)
    return apply_repetition_penalty(logits, seen_bitmask, penalties)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
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
    ``penalties`` otherwise on seen positions, preserving ``logits.dtype``."""
    target_dim = logits.shape[-1]
    shifts = torch.arange(32, dtype=torch.int32, device=seen_bitmask.device)
    seen_mask = (
        torch.bitwise_right_shift(seen_bitmask[:, :, None], shifts[None, None, :]) & 1
    ) != 0
    seen_mask = seen_mask.reshape(logits.shape[0], -1)[:, :target_dim]
    if seen_mask.shape[1] < target_dim:
        seen_mask = torch.nn.functional.pad(
            seen_mask, (0, target_dim - seen_mask.shape[1]), value=False
        )
    logits_f32 = logits.to(torch.float32)
    inv_penalties = 1.0 / penalties
    pos_scale = torch.where(logits_f32 > 0, inv_penalties, penalties)
    scaling = torch.where(seen_mask, pos_scale, 1.0)
    return (logits_f32 * scaling).to(logits.dtype)
