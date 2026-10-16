# SPDX-License-Identifier: Apache-2.0
"""``min_tokens`` for the TPU sampler.

Since vLLM 0.29 (#54218) the scheduler no longer holds back a stop below
``min_tokens``; ``MinTokensLogitsProcessor`` masks the request's stop tokens
in the sampler instead. The TPU runner samples on its own and never runs
logits processors, so it sets the stop-token logits to -inf itself, on device,
before a token is picked.

The runner keeps its own record by request id (``MinTokensTracker``) rather
than reading that processor's state, which is keyed by batch index and goes
stale within a step: the TPU runner reorders the batch after the state is
refreshed, and under async scheduling with drafts the output lists still hold
the previous step's optimistic placeholders when the state is refreshed. Batch
indices are looked up when the mask is built, after the reorder, and output
counts are taken once those placeholders are settled.

A row is masked while ``num_output_tokens + ahead < min_tokens``, where
``ahead`` is the row's position past the request's current output: 0 for a
plain step, ``j`` for draft position ``j`` of a verify chunk, and the draft
length for its bonus row. As in vLLM, a structured-output row whose logits are
all -inf after masking gets its stop tokens back.

Stop-token ids go to the device as ``[rows, STOP_IDS_WIDTH]`` tables, the only
width compiled: a request with more stop ids is masked in several passes of
that width. A masked row repeats its first id to fill the table (so every slot
it writes is a stop), an unmasked row is all -1.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm import SamplingParams

STOP_IDS_WIDTH = 8


@dataclass(frozen=True)
class MinTokensRequest:
    min_tokens: int
    stop_ids: tuple[int, ...]  # sorted
    structured: bool


# persistent-batch index -> (min_tokens, number of output tokens, stop token
# ids, uses structured output), for the requests still below min_tokens.
MinTokensState = dict[int, tuple[int, int, tuple[int, ...], bool]]


class MinTokensTracker:
    """The requests that asked for min_tokens and may still be below it, by
    request id."""

    def __init__(self) -> None:
        self.requests: dict[str, MinTokensRequest] = {}

    def __bool__(self) -> bool:
        return bool(self.requests)

    def add(self, req_id: str, params: SamplingParams | None) -> None:
        # A streaming-input continuation reuses its request id with new params.
        self.requests.pop(req_id, None)
        if params is None or not params.min_tokens or not params.all_stop_token_ids:
            return
        self.requests[req_id] = MinTokensRequest(
            params.min_tokens,
            tuple(sorted(params.all_stop_token_ids)),
            params.structured_outputs is not None,
        )

    def discard(self, req_ids: Iterable[str]) -> None:
        for req_id in req_ids:
            self.requests.pop(req_id, None)

    def any_in(self, req_id_to_index: Mapping[str, int]) -> bool:
        """Whether a tracked request is in the batch. Requests that reached
        min_tokens since the last ``batch_state`` still count."""
        return any(req_id in req_id_to_index for req_id in self.requests)

    def batch_state(
        self,
        req_id_to_index: Mapping[str, int],
        num_output_tokens: Callable[[str], int | None],
    ) -> MinTokensState:
        """The batch rows still below min_tokens. ``num_output_tokens`` must
        count settled tokens for the requests in the batch (None for a request
        that is gone); in-batch requests at or past min_tokens are dropped.
        Requests outside the batch are kept as they are: a preempted request
        can still hold unsettled placeholders, and resuming restores its
        output."""
        state: MinTokensState = {}
        for req_id, req in list(self.requests.items()):
            num_output = num_output_tokens(req_id)
            index = req_id_to_index.get(req_id)
            if num_output is None or (
                index is not None and num_output >= req.min_tokens
            ):
                del self.requests[req_id]
            elif index is not None:
                state[index] = (
                    req.min_tokens,
                    num_output,
                    req.stop_ids,
                    req.structured,
                )
        return state


# batch index -> (first logits row, ahead of that row, number of rows), for the
# batch indices in a MinTokensState only: a request's rows in one logits tensor
# are consecutive and one position apart.
StateRows = dict[int, tuple[int, int, int]]


def build_stop_table(
    state: MinTokensState,
    rows: StateRows,
    num_rows: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Host-side mask inputs for one logits tensor of ``num_rows`` rows, from
    the rows of the tracked requests (``plain_rows`` / ``verify_rows`` /
    ``verify_logits_rows``). Returns ``(stop_ids [num_rows, k *
    STOP_IDS_WIDTH] int32, restore [num_rows] bool)``, or None when no row
    needs masking.
    """
    # (stop ids, structured) -> ([first rows], [row counts]); requests usually
    # share their stop ids, so each group is written with one indexed store.
    groups: dict[tuple[tuple[int, ...], bool], tuple[list[int], list[int]]] = {}
    for batch_index, (first, ahead, count) in rows.items():
        min_toks, num_output, stop_ids, structured = state[batch_index]
        # Rows first .. first + n - 1 are still below min_tokens.
        n = min(count, min_toks - num_output - ahead)
        if n > 0:
            starts, lengths = groups.setdefault((stop_ids, structured), ([], []))
            starts.append(first)
            lengths.append(n)
    if not groups:
        return None
    longest = max(len(ids) for ids, _ in groups)
    width = -(-longest // STOP_IDS_WIDTH) * STOP_IDS_WIDTH
    stop_ids = np.full((num_rows, width), -1, dtype=np.int32)
    restore = np.zeros((num_rows,), dtype=np.bool_)
    for (ids, structured), (starts, lengths) in groups.items():
        lens = np.asarray(lengths)
        # The rows of every run: arange over the total, shifted per run.
        shift = np.repeat(np.asarray(starts) - (np.cumsum(lens) - lens), lens)
        index = np.arange(int(lens.sum())) + shift
        padded = np.full((width,), ids[0], dtype=np.int32)
        padded[: len(ids)] = ids
        stop_ids[index] = padded
        restore[index] = structured
    return stop_ids, restore


def _in_chunk(state: MinTokensState, start: int, end: int) -> list[int]:
    return [i for i in state if start <= i < end]


def plain_rows(state: MinTokensState, start: int, end: int) -> StateRows:
    """Rows of a chunk with one logits row per request."""
    return {i: (i - start, 0, 1) for i in _in_chunk(state, start, end)}


def verify_rows(
    state: MinTokensState, start: int, draft_lengths: Sequence[int]
) -> tuple[StateRows, StateRows]:
    """(target rows, bonus rows) of a verify chunk: request ``start + k`` owns
    ``draft_lengths[k]`` consecutive target rows and bonus row ``k``."""
    offsets = np.cumsum([0, *draft_lengths])
    target: StateRows = {}
    bonus: StateRows = {}
    for i in _in_chunk(state, start, start + len(draft_lengths)):
        k = i - start
        n = int(draft_lengths[k])
        target[i] = (int(offsets[k]), 0, n)
        bonus[i] = (k, n, 1)
    return target, bonus


def verify_logits_rows(
    state: MinTokensState, start: int, draft_lengths: Sequence[int]
) -> StateRows:
    """Rows of a verify chunk's logits, before the target / bonus gathers:
    request ``start + k`` owns ``draft_lengths[k] + 1`` consecutive rows, its
    draft positions then its bonus."""
    offsets = np.cumsum([0, *(int(n) + 1 for n in draft_lengths)])
    return {
        i: (int(offsets[i - start]), 0, int(draft_lengths[i - start]) + 1)
        for i in _in_chunk(state, start, start + len(draft_lengths))
    }


def apply_stop_table(
    logits: torch.Tensor, stop_ids: np.ndarray, restore: np.ndarray
) -> torch.Tensor:
    """Masks ``logits`` with a ``build_stop_table`` result."""
    masked = logits
    for start in range(0, stop_ids.shape[1], STOP_IDS_WIDTH):
        chunk = np.ascontiguousarray(stop_ids[:, start : start + STOP_IDS_WIDTH])
        masked = mask_stop_tokens(masked, torch.from_numpy(chunk).to(logits.device))
    if restore.any():
        masked = restore_all_masked_rows(
            logits, masked, torch.from_numpy(restore).to(logits.device)
        )
    return masked


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def mask_stop_tokens(logits: torch.Tensor, stop_ids: torch.Tensor) -> torch.Tensor:
    return _mask_stop_tokens(logits, stop_ids)


def _mask_stop_tokens(logits: torch.Tensor, stop_ids: torch.Tensor) -> torch.Tensor:
    """Sets ``logits[r, stop_ids[r]]`` to -inf. A row of -1 writes its own
    column-0 logit back to column 0; a masked row has no -1 slot, so the
    scatter never writes two different values to one place."""
    index = stop_ids.clamp(min=0).long()
    values = torch.where(
        stop_ids >= 0,
        torch.full(
            index.shape, float("-inf"), dtype=logits.dtype, device=logits.device
        ),
        logits.gather(1, index),
    )
    return logits.scatter(1, index, values)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def restore_all_masked_rows(
    logits: torch.Tensor, masked: torch.Tensor, restore: torch.Tensor
) -> torch.Tensor:
    return _restore_all_masked_rows(logits, masked, restore)


def _restore_all_masked_rows(
    logits: torch.Tensor, masked: torch.Tensor, restore: torch.Tensor
) -> torch.Tensor:
    """Rows with ``restore`` that are all -inf after masking go back to
    ``logits``; masking changed only their stop-token logits."""
    restore_row = restore & torch.isneginf(masked).all(dim=-1)
    return torch.where(restore_row[:, None], logits, masked)
