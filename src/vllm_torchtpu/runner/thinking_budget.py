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
"""Per-request ``SamplingParams.thinking_token_budget`` tracking and forcing.

``TPUModelRunner`` runs its own compiled sampling and speculative rejection
pipeline, reorders batch rows mid-step in ``_reorder_batch_for_rpa``, and under
``--async-scheduling`` leaves the previous step's sampled tokens unsettled on
device when ``sample_tokens()`` begins.

``ThinkingBudgetTracker`` keys budgets by ``req_id`` and maintains a
fixed-shape ``[capacity, 7]`` ``int32`` Knuth-Morris-Pratt (KMP) state table on
device so plain decode, ``TPU_GREEDY_TOP_TOKENS``, async scheduling, and
speculative verification advance without host stalls or XLA recompilations.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import torch

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm import SamplingParams
    from vllm.config.reasoning import ReasoningConfig
    from vllm.v1.worker.gpu_input_batch import CachedRequestState

    from vllm_torchtpu.runner.speculative_decoding_manager import SpecDecodeMetadata

logger = init_logger(__name__)

INT32_MAX: Final[int] = 2_147_483_647
NO_FORCED_TOKEN: Final[int] = -1

# Per-slot int32 columns in ``state_table`` ([capacity, NUM_STATE_COLS]).
COL_BUDGET: Final[int] = 0
COL_IN_THINK: Final[int] = 1
COL_THINK_COUNT: Final[int] = 2
COL_START_MATCH: Final[int] = 3
COL_NAT_END_MATCH: Final[int] = 4
COL_END_MATCH: Final[int] = 5
COL_FORCING_END: Final[int] = 6
NUM_STATE_COLS: Final[int] = 7

StateTuple = tuple[int, int, int, int, int, int, int]
UNTRACKED_STATE: Final[StateTuple] = (-1, 0, 0, 0, 0, 0, 0)


def compute_kmp_failure(pattern: Sequence[int]) -> tuple[int, ...]:
    """Return KMP fallback table ``fail[0 .. len(pattern)]`` for ``pattern``."""
    m = len(pattern)
    if m == 0:
        return ()
    lps = [0] * m
    length = 0
    idx = 1
    while idx < m:
        if pattern[idx] == pattern[length]:
            length += 1
            lps[idx] = length
            idx += 1
        elif length > 0:
            length = lps[length - 1]
        else:
            lps[idx] = 0
            idx += 1
    return (0, *lps)


@dataclass(frozen=True)
class ThinkingBudgetMarkers:
    """Tokenized start/end markers and KMP fallback tables for reasoning."""

    start_ids: tuple[int, ...]
    start_fail: tuple[int, ...]
    end_ids: tuple[int, ...]
    end_fail: tuple[int, ...]
    natural_end_ids: tuple[int, ...]
    natural_end_fail: tuple[int, ...]

    @classmethod
    def from_reasoning_config(
        cls, reasoning_config: "ReasoningConfig | None"
    ) -> "ThinkingBudgetMarkers | None":
        if reasoning_config is None or not reasoning_config.enabled:
            return None
        start_raw = reasoning_config.reasoning_start_token_ids
        end_raw = reasoning_config.reasoning_end_token_ids
        if not start_raw or not end_raw:
            return None
        nat_raw = reasoning_config.natural_reasoning_end_token_ids or end_raw
        start_ids = tuple(int(t) for t in start_raw)
        end_ids = tuple(int(t) for t in end_raw)
        natural_end_ids = tuple(int(t) for t in nat_raw)
        return cls(
            start_ids=start_ids,
            start_fail=compute_kmp_failure(start_ids),
            end_ids=end_ids,
            end_fail=compute_kmp_failure(end_ids),
            natural_end_ids=natural_end_ids,
            natural_end_fail=compute_kmp_failure(natural_end_ids),
        )

    def kernel_args(self) -> tuple[tuple[int, ...], ...]:
        return (
            self.start_ids,
            self.start_fail,
            self.natural_end_ids,
            self.natural_end_fail,
            self.end_ids,
            self.end_fail,
        )


def _kmp_step_py(
    pattern: tuple[int, ...], fail: tuple[int, ...], match_len: int, tok: int
) -> int:
    q = match_len
    while q > 0 and tok != pattern[q]:
        q = fail[q]
    return q + 1 if tok == pattern[q] else 0


def advance_state_py(
    state: StateTuple, tok: int, markers: ThinkingBudgetMarkers
) -> StateTuple:
    """Advance scalar 7-tuple KMP thinking-budget state on one token ``tok``."""
    (
        budget,
        in_think,
        think_count,
        start_match,
        nat_end_match,
        end_match,
        forcing_end,
    ) = state
    if budget < 0:
        return UNTRACKED_STATE
    if in_think == 0:
        next_start = _kmp_step_py(
            markers.start_ids, markers.start_fail, start_match, tok
        )
        if next_start == len(markers.start_ids):
            return (budget, 1, 0, 0, 0, 0, 0)
        return (budget, 0, 0, next_start, 0, 0, 0)

    forcing_now = forcing_end == 1 or think_count >= budget
    actively_forcing_seq = forcing_now and tok == markers.end_ids[end_match]
    next_nat = _kmp_step_py(
        markers.natural_end_ids, markers.natural_end_fail, nat_end_match, tok
    )
    next_end = _kmp_step_py(markers.end_ids, markers.end_fail, end_match, tok)
    closed = (
        next_end == len(markers.end_ids)
        if actively_forcing_seq
        else (
            next_end == len(markers.end_ids) or next_nat == len(markers.natural_end_ids)
        )
    )
    if closed:
        return (budget, 0, 0, 0, 0, 0, 0)
    if next_nat == len(markers.natural_end_ids):
        next_nat = markers.natural_end_fail[len(markers.natural_end_ids)]
    if next_end == len(markers.end_ids):
        next_end = markers.end_fail[len(markers.end_ids)]
    next_count = think_count + 1 if think_count < INT32_MAX else INT32_MAX
    return (budget, 1, next_count, 0, next_nat, next_end, int(actively_forcing_seq))


def query_forced_token_py(state: StateTuple, markers: ThinkingBudgetMarkers) -> int:
    """Return the forced token ID for ``state``, or ``NO_FORCED_TOKEN`` (-1)."""
    budget, in_think, think_count, _, _, end_match, forcing_end = state
    if budget < 0 or in_think == 0:
        return NO_FORCED_TOKEN
    if forcing_end == 1 or think_count >= budget:
        return markers.end_ids[end_match]
    return NO_FORCED_TOKEN


def advance_sequence_state(
    state: StateTuple,
    tokens: Sequence[int] | np.ndarray,
    markers: ThinkingBudgetMarkers,
) -> StateTuple:
    """Advance ``state`` across ``tokens``, jumping between marker candidates.

    Whenever partial multi-token marker state is clear
    (``start_match == nat_end_match == end_match == forcing_end == 0``), tokens
    outside ``{start_ids[0], natural_end_ids[0], end_ids[0]}`` cannot alter
    ``in_think == 0`` state and only advance ``think_count`` when
    ``in_think == 1``, so ``np.flatnonzero`` skips non-marker spans directly.
    """
    if state[COL_BUDGET] < 0:
        return UNTRACKED_STATE
    arr = np.asarray(tokens, dtype=np.int64)
    n = int(arr.shape[0])
    if n == 0:
        return state
    if not (markers.start_ids and markers.natural_end_ids and markers.end_ids):
        for tok in arr.tolist():
            state = advance_state_py(state, int(tok), markers)
        return state

    s0 = markers.start_ids[0]
    n0 = markers.natural_end_ids[0]
    e0 = markers.end_ids[0]
    budget = state[COL_BUDGET]
    hit_mask = (arr == s0) | (arr == e0)
    if n0 != e0 and n0 != s0:
        hit_mask |= arr == n0
    hits = np.flatnonzero(hit_mask)
    num_hits = int(hits.shape[0])
    pos = 0
    hit_ptr = 0

    while pos < n:
        if (
            state[COL_START_MATCH]
            | state[COL_NAT_END_MATCH]
            | state[COL_END_MATCH]
            | state[COL_FORCING_END]
        ) != 0:
            state = advance_state_py(state, int(arr[pos]), markers)
            pos += 1
            continue
        while hit_ptr < num_hits and int(hits[hit_ptr]) < pos:
            hit_ptr += 1
        next_hit = int(hits[hit_ptr]) if hit_ptr < num_hits else n
        if next_hit > pos:
            if state[COL_IN_THINK] != 0:
                new_cnt = min(state[COL_THINK_COUNT] + (next_hit - pos), INT32_MAX)
                state = (budget, 1, new_cnt, 0, 0, 0, 0)
            pos = next_hit
        if pos < n:
            state = advance_state_py(state, int(arr[pos]), markers)
            pos += 1
            hit_ptr += 1
    return state


def compute_sequence_state(
    tokens: Sequence[int] | np.ndarray, budget: int, markers: ThinkingBudgetMarkers
) -> StateTuple:
    """Compute the 7-tuple KMP state after consuming ``tokens``."""
    init: StateTuple = (min(int(budget), INT32_MAX), 0, 0, 0, 0, 0, 0)
    return advance_sequence_state(init, tokens, markers)


def _compute_request_state(
    req_state: "CachedRequestState | None",
    budget: int,
    markers: ThinkingBudgetMarkers,
) -> StateTuple:
    """Reconstruct KMP state from ``req_state``'s prompt and committed outputs.

    Placeholder invariants:
    * ``on_resumed`` runs in ``TPUModelRunner._update_states`` after
      ``super()._update_states`` aligns ``req_state.output_token_ids`` to the
      scheduler's committed ``num_output_tokens`` (truncating trailing ``[0]``
      async placeholders when ``num_output_tokens == 0``, or replacing
      ``req_state.output_token_ids`` from ``all_token_ids`` under async
      scheduling when ``num_output_tokens > 0``); any negative scheduler
      placeholders are skipped by ``out_arr >= 0``.
    * ``prepare_step`` only calls ``_compute_request_state`` on a cold reseed
      when a budgeted request has neither an allocated slot nor a seeded
      ``_dirty_states`` entry; ``TPUModelRunner.sample_tokens`` settles any
      in-flight ``_pre_async_results`` (``[0] * n_new`` placeholders) via
      ``_modify_prev_results`` whenever ``needs_output_reseed`` is true.
    """
    if req_state is None:
        return compute_sequence_state((), budget, markers)
    state = compute_sequence_state(req_state.prompt_token_ids or (), budget, markers)
    out_ids = req_state.output_token_ids
    if out_ids:
        out_arr = np.asarray(out_ids, dtype=np.int64)
        state = advance_sequence_state(state, out_arr[out_arr >= 0], markers)
    return state


def _kmp_step_tensor(
    pattern: tuple[int, ...],
    fail: tuple[int, ...],
    match_len: torch.Tensor,
    tok: torch.Tensor,
) -> torch.Tensor:
    if len(pattern) == 1:
        return (tok == pattern[0]).to(match_len.dtype)
    zero = torch.zeros_like(match_len)
    out_for_q: list[torch.Tensor] = []
    for q, pat_tok in enumerate(pattern):
        fallback = zero if q == 0 else out_for_q[fail[q]]
        out_for_q.append(torch.where(tok == pat_tok, q + 1, fallback))
    res = out_for_q[0]
    for q in range(1, len(pattern)):
        res = torch.where(match_len == q, out_for_q[q], res)
    return res


def _gather_end_token_tensor(
    end_ids: tuple[int, ...], end_match: torch.Tensor
) -> torch.Tensor:
    res = torch.full_like(end_match, end_ids[0])
    for q in range(1, len(end_ids)):
        res = torch.where(end_match == q, end_ids[q], res)
    return res


def _query_forced_token_tensor(
    st: torch.Tensor, end_ids: tuple[int, ...]
) -> torch.Tensor:
    budget = st[:, COL_BUDGET]
    should_force = (
        (budget >= 0)
        & (st[:, COL_IN_THINK] != 0)
        & ((st[:, COL_FORCING_END] != 0) | (st[:, COL_THINK_COUNT] >= budget))
    )
    return torch.where(
        should_force,
        _gather_end_token_tensor(end_ids, st[:, COL_END_MATCH]),
        torch.full_like(budget, NO_FORCED_TOKEN),
    )


def _advance_state_tensor(
    st: torch.Tensor,
    tok: torch.Tensor,
    start_ids: tuple[int, ...],
    start_fail: tuple[int, ...],
    nat_ids: tuple[int, ...],
    nat_fail: tuple[int, ...],
    end_ids: tuple[int, ...],
    end_fail: tuple[int, ...],
) -> torch.Tensor:
    budget = st[:, COL_BUDGET]
    in_think = st[:, COL_IN_THINK]
    think_count = st[:, COL_THINK_COUNT]
    zero = torch.zeros_like(budget)
    one = torch.ones_like(budget)

    next_start = _kmp_step_tensor(start_ids, start_fail, st[:, COL_START_MATCH], tok)
    entered_think = next_start == len(start_ids)
    out_not_in_think = torch.stack(
        [
            budget,
            torch.where(entered_think, one, zero),
            zero,
            torch.where(entered_think, zero, next_start),
            zero,
            zero,
            zero,
        ],
        dim=-1,
    )

    forcing_now = (st[:, COL_FORCING_END] != 0) | (think_count >= budget)
    expected_end_tok = _gather_end_token_tensor(end_ids, st[:, COL_END_MATCH])
    actively_forcing_seq = forcing_now & (tok == expected_end_tok)

    next_end = _kmp_step_tensor(end_ids, end_fail, st[:, COL_END_MATCH], tok)
    end_completed = next_end == len(end_ids)
    wrapped_end = torch.where(
        end_completed, torch.full_like(next_end, end_fail[len(end_ids)]), next_end
    )
    if (nat_ids, nat_fail) == (end_ids, end_fail):
        nat_completed = end_completed
        wrapped_nat = wrapped_end
    else:
        next_nat = _kmp_step_tensor(nat_ids, nat_fail, st[:, COL_NAT_END_MATCH], tok)
        nat_completed = next_nat == len(nat_ids)
        wrapped_nat = torch.where(
            nat_completed,
            torch.full_like(next_nat, nat_fail[len(nat_ids)]),
            next_nat,
        )
    closed = torch.where(
        actively_forcing_seq, end_completed, end_completed | nat_completed
    )
    inc_count = torch.where(
        think_count < INT32_MAX,
        think_count + 1,
        torch.full_like(think_count, INT32_MAX),
    )
    out_in_think = torch.stack(
        [
            budget,
            torch.where(closed, zero, one),
            torch.where(closed, zero, inc_count),
            zero,
            torch.where(closed, zero, wrapped_nat),
            torch.where(closed, zero, wrapped_end),
            torch.where(closed, zero, torch.where(actively_forcing_seq, one, zero)),
        ],
        dim=-1,
    )

    untracked = torch.stack(
        [torch.full_like(budget, -1), zero, zero, zero, zero, zero, zero], dim=-1
    )
    active_out = torch.where(
        (in_think != 0).unsqueeze(-1), out_in_think, out_not_in_think
    )
    return torch.where((budget >= 0).unsqueeze(-1), active_out, untracked)


_AMAX_SENTINEL: Final[int] = -2_147_483_647


def _scatter_state_rows(
    state_table: torch.Tensor, slot_ids: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    """Return ``state_table`` with ``out[slot_ids[i]] = rows[i]`` for active slots.

    Uses a masked broadcast + ``amax`` reduction (``<= 7 MiB`` at
    ``max_num_reqs=512``) rather than ``Tensor.scatter(0, ...)``, whose XLA
    lowering for dynamic colliding indices adds ~7 ms/step on TPU v7x. Each
    positive slot ``slot_ids[i] > 0`` must appear at most once per call.
    """
    if state_table.device.type == "cpu" and not torch.compiler.is_compiling():
        pos_slots = slot_ids[slot_ids > 0]
        assert pos_slots.numel() == torch.unique(pos_slots).numel(), (
            "Duplicate positive slot IDs passed to _scatter_state_rows"
        )
    table_idx = torch.arange(
        state_table.shape[0], dtype=slot_ids.dtype, device=state_table.device
    ).unsqueeze(1)
    match = (slot_ids > 0).unsqueeze(0) & (table_idx == slot_ids.unsqueeze(0))
    cand = torch.where(
        match.unsqueeze(-1), rows.to(state_table.dtype).unsqueeze(0), _AMAX_SENTINEL
    ).amax(dim=1)
    return torch.where(match.any(dim=1, keepdim=True), cand, state_table)


def _seed_states_impl(
    state_table: torch.Tensor, seed_slots: torch.Tensor, seed_values: torch.Tensor
) -> torch.Tensor:
    return _scatter_state_rows(state_table, seed_slots, seed_values)


def _override_and_advance_plain_impl(
    state_table: torch.Tensor,
    slot_ids: torch.Tensor,
    selected_token_ids: torch.Tensor,
    start_ids: tuple[int, ...],
    start_fail: tuple[int, ...],
    nat_ids: tuple[int, ...],
    nat_fail: tuple[int, ...],
    end_ids: tuple[int, ...],
    end_fail: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    active_slot = slot_ids > 0
    safe_slots = torch.where(active_slot, slot_ids, torch.zeros_like(slot_ids)).to(
        torch.int64
    )
    st = torch.index_select(state_table, 0, safe_slots)
    forced_id = _query_forced_token_tensor(st, end_ids)
    out_tokens = torch.where(
        (active_slot & (forced_id >= 0)).unsqueeze(-1),
        forced_id.to(selected_token_ids.dtype).unsqueeze(-1),
        selected_token_ids,
    )
    next_st = _advance_state_tensor(
        st,
        out_tokens.squeeze(-1).to(torch.int32),
        start_ids,
        start_fail,
        nat_ids,
        nat_fail,
        end_ids,
        end_fail,
    )
    return out_tokens, _scatter_state_rows(state_table, slot_ids, next_st)


def _prepare_spec_verify_drafts(
    state_table: torch.Tensor,
    slot_ids: torch.Tensor,
    draft_token_ids: torch.Tensor,
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    max_draft_tokens: int,
    start_ids: tuple[int, ...],
    start_fail: tuple[int, ...],
    nat_ids: tuple[int, ...],
    nat_fail: tuple[int, ...],
    end_ids: tuple[int, ...],
    end_fail: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-draft-row forced token IDs and invalidated draft IDs.

    ``slot_ids`` is sized to ``max_num_reqs`` during serving so the compiled
    spec-verify kernels depend only on ``draft_token_ids.shape[0]``
    (``verify_logits_paddings``) rather than the ``(num_tokens, num_reqs)``
    Cartesian product. ``SpecDecodeMetadata`` marks padded draft positions with
    ``segment_ids == real_num_reqs``, where ``slot_ids[real_num_reqs] == 0``
    (or ``real_num_reqs == max_num_reqs``, out of ``[0, num_reqs)``).
    """
    num_reqs = slot_ids.shape[0]
    active_slot = slot_ids > 0
    safe_slots = torch.where(active_slot, slot_ids, torch.zeros_like(slot_ids)).to(
        torch.int64
    )
    st = torch.index_select(state_table, 0, safe_slots)
    draft_i32 = draft_token_ids.to(torch.int32)
    no_tok = torch.full_like(draft_i32, NO_FORCED_TOKEN)

    if max_draft_tokens > 0:
        num_cells = num_reqs * max_draft_tokens
        seg_in_range = (segment_ids >= 0) & (segment_ids < num_reqs)
        safe_seg = torch.where(seg_in_range, segment_ids, 0).to(torch.int64)
        seg_active = seg_in_range & (torch.index_select(slot_ids, 0, safe_seg) > 0)
        in_bounds = (
            seg_active & (group_indices >= 0) & (group_indices < max_draft_tokens)
        )
        safe_grp = torch.where(in_bounds, group_indices, 0).to(torch.int64)
        flat_idx = safe_seg * max_draft_tokens + safe_grp
        if state_table.device.type == "cpu" and not torch.compiler.is_compiling():
            valid_cells = flat_idx[in_bounds]
            assert valid_cells.numel() == torch.unique(valid_cells).numel(), (
                "Duplicate active (segment_id, group_index) in "
                "_prepare_spec_verify_drafts"
            )
        cells = torch.arange(
            num_cells, dtype=torch.int64, device=state_table.device
        ).unsqueeze(1)
        cell_match = in_bounds.unsqueeze(0) & (cells == flat_idx.unsqueeze(0))
        any_cell = cell_match.any(dim=1)
        draft_grid = torch.where(
            any_cell,
            torch.where(cell_match, draft_i32.unsqueeze(0), _AMAX_SENTINEL).amax(dim=1),
            NO_FORCED_TOKEN,
        ).reshape(num_reqs, max_draft_tokens)

        no_req_tok = torch.full(
            (num_reqs,), NO_FORCED_TOKEN, dtype=torch.int32, device=state_table.device
        )
        forced_cols: list[torch.Tensor] = []
        for step_idx in range(max_draft_tokens):
            step_forced_req = torch.where(
                active_slot, _query_forced_token_tensor(st, end_ids), no_req_tok
            )
            forced_cols.append(step_forced_req)
            d_tok = draft_grid[:, step_idx]
            step_active = active_slot & (d_tok >= 0)
            next_st = _advance_state_tensor(
                st,
                d_tok,
                start_ids,
                start_fail,
                nat_ids,
                nat_fail,
                end_ids,
                end_fail,
            )
            st = torch.where(step_active.unsqueeze(-1), next_st, st)
        forced_flat = torch.stack(forced_cols, dim=1).reshape(-1)
        forced_target_ids = torch.where(
            in_bounds, torch.index_select(forced_flat, 0, flat_idx), no_tok
        )
    else:
        forced_target_ids = no_tok

    effective_draft_ids = torch.where(
        (forced_target_ids >= 0) & (draft_i32 != forced_target_ids),
        no_tok,
        draft_i32,
    )
    return forced_target_ids, effective_draft_ids


def _force_logits_rows_impl(
    logits: torch.Tensor, forced_ids: torch.Tensor
) -> torch.Tensor:
    is_forced = (forced_ids >= 0).unsqueeze(-1)
    masked = torch.where(is_forced, float("-inf"), logits)
    safe_ids = forced_ids.clamp(min=0, max=logits.shape[-1] - 1).to(torch.int64)
    zero_src = torch.zeros(
        (logits.shape[0], 1), dtype=logits.dtype, device=logits.device
    )
    forced_val = torch.where(
        is_forced, zero_src, logits.gather(-1, safe_ids.unsqueeze(-1))
    )
    return masked.scatter(-1, safe_ids.unsqueeze(-1), forced_val)


def _force_token_ids_1d_impl(
    token_ids: torch.Tensor, forced_ids: torch.Tensor
) -> torch.Tensor:
    return torch.where(forced_ids >= 0, forced_ids.to(token_ids.dtype), token_ids)


def _prepare_spec_verify_logits_impl(
    state_table: torch.Tensor,
    slot_ids: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_logits: torch.Tensor,
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    max_draft_tokens: int,
    start_ids: tuple[int, ...],
    start_fail: tuple[int, ...],
    nat_ids: tuple[int, ...],
    nat_fail: tuple[int, ...],
    end_ids: tuple[int, ...],
    end_fail: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    forced_target_ids, effective_draft_ids = _prepare_spec_verify_drafts(
        state_table,
        slot_ids,
        draft_token_ids,
        segment_ids,
        group_indices,
        max_draft_tokens,
        start_ids,
        start_fail,
        nat_ids,
        nat_fail,
        end_ids,
        end_fail,
    )
    return (
        _force_logits_rows_impl(target_logits, forced_target_ids),
        effective_draft_ids,
    )


def _prepare_spec_verify_tokens_impl(
    state_table: torch.Tensor,
    slot_ids: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_argmax: torch.Tensor,
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    max_draft_tokens: int,
    start_ids: tuple[int, ...],
    start_fail: tuple[int, ...],
    nat_ids: tuple[int, ...],
    nat_fail: tuple[int, ...],
    end_ids: tuple[int, ...],
    end_fail: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    forced_target_ids, effective_draft_ids = _prepare_spec_verify_drafts(
        state_table,
        slot_ids,
        draft_token_ids,
        segment_ids,
        group_indices,
        max_draft_tokens,
        start_ids,
        start_fail,
        nat_ids,
        nat_fail,
        end_ids,
        end_fail,
    )
    return (
        _force_token_ids_1d_impl(target_argmax, forced_target_ids),
        effective_draft_ids,
    )


def _advance_spec_states_impl(
    state_table: torch.Tensor,
    slot_ids: torch.Tensor,
    next_tokens: torch.Tensor,
    max_draft_tokens: int,
    start_ids: tuple[int, ...],
    start_fail: tuple[int, ...],
    nat_ids: tuple[int, ...],
    nat_fail: tuple[int, ...],
    end_ids: tuple[int, ...],
    end_fail: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Override forced bonus/accepted tokens in ``next_tokens`` and commit KMP state.

    ``RejectionSampler`` only writes ``bonus_token_ids`` into ``next_tokens``
    at ``accepted_count`` when all drafts pass; overriding ``tok >= 0`` here
    with ``_query_forced_token_tensor(st, end_ids)`` handles bonus forcing in
    the same pass that advances ``state_table`` without a separate 1D kernel.
    """
    chunk_slots = slot_ids[: next_tokens.shape[0]]
    active_slot = chunk_slots > 0
    safe_slots = torch.where(
        active_slot, chunk_slots, torch.zeros_like(chunk_slots)
    ).to(torch.int64)
    st = torch.index_select(state_table, 0, safe_slots)
    out_cols: list[torch.Tensor] = []
    for col_idx in range(max_draft_tokens + 1):
        tok = next_tokens[:, col_idx].to(torch.int32)
        valid_col = active_slot & (tok >= 0)
        forced_id = _query_forced_token_tensor(st, end_ids)
        eff_tok = torch.where(valid_col & (forced_id >= 0), forced_id, tok)
        out_cols.append(eff_tok.to(next_tokens.dtype))
        next_st = _advance_state_tensor(
            st, eff_tok, start_ids, start_fail, nat_ids, nat_fail, end_ids, end_fail
        )
        st = torch.where(valid_col.unsqueeze(-1), next_st, st)
    out_tokens = torch.stack(out_cols, dim=1)
    return out_tokens, _scatter_state_rows(state_table, chunk_slots, st)


_tpu_compile = torch.compile(backend="tpu", fullgraph=True, dynamic=False)
_seed_states_tpu = _tpu_compile(_seed_states_impl)
_override_and_advance_plain_tpu = _tpu_compile(_override_and_advance_plain_impl)
_prepare_spec_verify_logits_tpu = _tpu_compile(_prepare_spec_verify_logits_impl)
_prepare_spec_verify_tokens_tpu = _tpu_compile(_prepare_spec_verify_tokens_impl)
_advance_spec_states_tpu = _tpu_compile(_advance_spec_states_impl)


def seed_states(
    state_table: torch.Tensor, seed_slots: torch.Tensor, seed_values: torch.Tensor
) -> torch.Tensor:
    fn = _seed_states_tpu if state_table.device.type == "tpu" else _seed_states_impl
    return fn(state_table, seed_slots, seed_values)


def override_and_advance_plain(
    state_table: torch.Tensor,
    slot_ids: torch.Tensor,
    selected_token_ids: torch.Tensor,
    markers: ThinkingBudgetMarkers,
) -> tuple[torch.Tensor, torch.Tensor]:
    fn = (
        _override_and_advance_plain_tpu
        if state_table.device.type == "tpu"
        else _override_and_advance_plain_impl
    )
    return fn(state_table, slot_ids, selected_token_ids, *markers.kernel_args())


def prepare_spec_verify_logits(
    state_table: torch.Tensor,
    slot_ids: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_logits: torch.Tensor,
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    max_draft_tokens: int,
    markers: ThinkingBudgetMarkers,
) -> tuple[torch.Tensor, torch.Tensor]:
    fn = (
        _prepare_spec_verify_logits_tpu
        if state_table.device.type == "tpu"
        else _prepare_spec_verify_logits_impl
    )
    return fn(
        state_table,
        slot_ids,
        draft_token_ids,
        target_logits,
        segment_ids,
        group_indices,
        max_draft_tokens,
        *markers.kernel_args(),
    )


def prepare_spec_verify_tokens(
    state_table: torch.Tensor,
    slot_ids: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_argmax: torch.Tensor,
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    max_draft_tokens: int,
    markers: ThinkingBudgetMarkers,
) -> tuple[torch.Tensor, torch.Tensor]:
    fn = (
        _prepare_spec_verify_tokens_tpu
        if state_table.device.type == "tpu"
        else _prepare_spec_verify_tokens_impl
    )
    return fn(
        state_table,
        slot_ids,
        draft_token_ids,
        target_argmax,
        segment_ids,
        group_indices,
        max_draft_tokens,
        *markers.kernel_args(),
    )


def advance_spec_states(
    state_table: torch.Tensor,
    slot_ids: torch.Tensor,
    next_tokens: torch.Tensor,
    max_draft_tokens: int,
    markers: ThinkingBudgetMarkers,
) -> tuple[torch.Tensor, torch.Tensor]:
    fn = (
        _advance_spec_states_tpu
        if state_table.device.type == "tpu"
        else _advance_spec_states_impl
    )
    return fn(
        state_table, slot_ids, next_tokens, max_draft_tokens, *markers.kernel_args()
    )


def _same_target_device(actual: torch.device, requested: torch.device) -> bool:
    """Match e.g. ``device('tpu', 0)`` against unindexed ``device('tpu')``."""
    if actual.type != requested.type:
        return False
    if requested.index is not None:
        return actual.index == requested.index
    if actual.index is None:
        return True
    default_idx = (
        torch.accelerator.current_device_index()
        if torch.accelerator.is_available()
        and (acc := torch.accelerator.current_accelerator()) is not None
        and acc.type == requested.type
        else 0
    )
    return actual.index == default_idx


class ThinkingBudgetTracker:
    """Request-ID-keyed ``thinking_token_budget`` lifecycle and device state."""

    def __init__(
        self,
        reasoning_config: "ReasoningConfig | None",
        max_num_reqs: int,
    ) -> None:
        self._markers = ThinkingBudgetMarkers.from_reasoning_config(reasoning_config)
        self._max_num_reqs = max(int(max_num_reqs), 1)
        # The scheduler bounds running requests by max_num_reqs, and finished /
        # preempted owners release their slots in _update_states (matching
        # TPUModelRunner._free_mamba_slots). Slot 0 is the permanent untracked
        # sentinel; slots 1..max_num_reqs serve active budgeted requests.
        self._capacity = self._max_num_reqs + 1
        self._budgets: dict[str, int] = {}
        self._slot_by_req_id: dict[str, int] = {}
        self._free_slots: list[int] = list(range(self._capacity - 1, 0, -1))
        self._dirty_states: dict[str, StateTuple] = {}
        self._state_table: torch.Tensor | None = None

    @property
    def enabled(self) -> bool:
        return self._markers is not None

    @property
    def max_num_reqs(self) -> int:
        return self._max_num_reqs

    @property
    def markers(self) -> ThinkingBudgetMarkers | None:
        return self._markers

    @property
    def requests(self) -> Mapping[str, int]:
        return self._budgets

    def __bool__(self) -> bool:
        return self._markers is not None and bool(self._budgets)

    def commit_state_table(self, state_table: torch.Tensor) -> None:
        self._state_table = state_table

    def ensure_state_table(self, device: torch.device) -> torch.Tensor:
        if self._state_table is None or not _same_target_device(
            self._state_table.device, device
        ):
            if self._state_table is not None:
                self._slot_by_req_id.clear()
                self._free_slots = list(range(self._capacity - 1, 0, -1))
            table = torch.zeros(
                (self._capacity, NUM_STATE_COLS), dtype=torch.int32, device="cpu"
            )
            table[:, COL_BUDGET] = -1
            self._state_table = table.to(device)
        return self._state_table

    def _release_slot(self, req_id: str) -> None:
        slot = self._slot_by_req_id.pop(req_id, None)
        if slot is not None:
            self._free_slots.append(slot)

    def discard(self, req_ids: Iterable[str]) -> None:
        if not self._budgets:
            return
        for req_id in req_ids:
            self._budgets.pop(req_id, None)
            self._dirty_states.pop(req_id, None)
            self._release_slot(req_id)

    def on_preempted(self, req_ids: Iterable[str] | None) -> None:
        if not req_ids or not self._budgets:
            return
        for req_id in req_ids:
            if req_id in self._budgets:
                self._dirty_states.pop(req_id, None)
                self._release_slot(req_id)

    def add(
        self,
        req_id: str,
        sampling_params: "SamplingParams | None",
        prompt_token_ids: Sequence[int] | None,
    ) -> None:
        raw_budget = (
            None if sampling_params is None else sampling_params.thinking_token_budget
        )
        if raw_budget is None or raw_budget < 0:
            if self._budgets:
                self.discard((req_id,))
            return
        if self._markers is None:
            logger.warning_once(
                "Ignoring thinking_token_budget because reasoning start/end "
                "token IDs are not configured on the model runner."
            )
            return
        clamped_budget = min(int(raw_budget), INT32_MAX)
        self._budgets[req_id] = clamped_budget
        self._dirty_states[req_id] = compute_sequence_state(
            prompt_token_ids or (), clamped_budget, self._markers
        )

    def needs_output_reseed(
        self,
        batch_req_ids: Iterable[str | None],
        device: torch.device | None = None,
    ) -> bool:
        """Return True if ``prepare_step`` will read ``output_token_ids``."""
        if self._markers is None or not self._budgets:
            return False
        slots_valid = self._state_table is None or (
            device is None or _same_target_device(self._state_table.device, device)
        )
        return any(
            req_id is not None
            and req_id in self._budgets
            and (not slots_valid or req_id not in self._slot_by_req_id)
            and req_id not in self._dirty_states
            for req_id in batch_req_ids
        )

    def step_state(
        self,
        device: torch.device,
        batch_req_ids: Sequence[str | None],
        requests: Mapping[str, "CachedRequestState"],
        discard_req_ids: AbstractSet[str],
    ) -> "ThinkingBudgetStepState | None":
        """Prepare this step's device KMP state for ``sample_tokens``."""
        if not self:
            return None
        markers = self._markers
        assert markers is not None
        state_table = self.prepare_step(device, batch_req_ids, requests)
        if state_table is None:
            return None
        return ThinkingBudgetStepState(self, state_table, markers, discard_req_ids)

    def on_resumed(
        self,
        resumed_req_ids: Iterable[str],
        requests: Mapping[str, "CachedRequestState"],
    ) -> None:
        if self._markers is None or not self._budgets:
            return
        for req_id in resumed_req_ids:
            budget = self._budgets.get(req_id)
            req_state = None if budget is None else requests.get(req_id)
            if budget is None or req_state is None:
                continue
            self._dirty_states[req_id] = _compute_request_state(
                req_state, budget, self._markers
            )

    def _allocate_slot(self, req_id: str) -> int:
        existing = self._slot_by_req_id.get(req_id)
        if existing is not None:
            return existing
        assert self._free_slots, (
            f"ThinkingBudgetTracker slot pool exhausted "
            f"(max_num_reqs={self._max_num_reqs}, capacity={self._capacity})"
        )
        slot = self._free_slots.pop()
        self._slot_by_req_id[req_id] = slot
        return slot

    def prepare_step(
        self,
        device: torch.device,
        batch_req_ids: Sequence[str | None],
        requests: Mapping[str, "CachedRequestState"],
    ) -> torch.Tensor | None:
        """Ensure all scheduled budgeted requests own seeded slots on ``device``."""
        if self._markers is None or not self._budgets:
            return None
        batch_tracked = list(
            dict.fromkeys(
                req_id
                for req_id in batch_req_ids
                if req_id is not None and req_id in self._budgets
            )
        )
        if not batch_tracked:
            return None

        if requests:
            batch_set = set(batch_tracked)
            for owner_id in list(self._slot_by_req_id):
                if owner_id not in requests and owner_id not in batch_set:
                    self._release_slot(owner_id)

        state_table = self.ensure_state_table(device)
        to_seed: list[tuple[int, StateTuple]] = []
        for req_id in batch_tracked:
            had_slot = req_id in self._slot_by_req_id
            slot = self._allocate_slot(req_id)
            dirty = self._dirty_states.pop(req_id, None)
            if dirty is None and not had_slot:
                dirty = _compute_request_state(
                    requests.get(req_id), self._budgets[req_id], self._markers
                )
            if dirty is not None:
                to_seed.append((slot, dirty))

        if to_seed:
            seed_width = self._max_num_reqs
            for start in range(0, len(to_seed), seed_width):
                chunk = to_seed[start : start + seed_width]
                slots_np = np.zeros(seed_width, dtype=np.int32)
                vals_np = np.zeros((seed_width, NUM_STATE_COLS), dtype=np.int32)
                vals_np[:, COL_BUDGET] = -1
                for idx, (slot, st_tuple) in enumerate(chunk):
                    slots_np[idx] = slot
                    vals_np[idx] = st_tuple
                # Synchronous .to(device): sample_tokens() runs after the runner's
                # input-staging fence, so async H2D from temporary CPU buffers
                # would risk reading freed host memory in TorchTPU.
                state_table = seed_states(
                    state_table,
                    torch.from_numpy(slots_np).to(device),
                    torch.from_numpy(vals_np).to(device),
                )
            self._state_table = state_table
        return state_table

    def build_chunk_slot_ids(
        self,
        device: torch.device,
        batch_req_ids: Sequence[str | None],
        start_idx: int,
        end_idx: int,
        padded_num_reqs: int,
        discard_req_ids: AbstractSet[str] | None = None,
    ) -> torch.Tensor | None:
        """Build ``[padded_num_reqs]`` int32 slot IDs for ``[start_idx, end_idx)``."""
        upper = min(end_idx, len(batch_req_ids), start_idx + padded_num_reqs)
        slots_np = np.zeros(padded_num_reqs, dtype=np.int32)
        seen_slots: set[int] = set()
        any_active = False
        for batch_i in range(start_idx, upper):
            req_id = batch_req_ids[batch_i]
            if req_id is None:
                continue
            slot = self._slot_by_req_id.get(req_id)
            if slot is None or (
                discard_req_ids is not None and req_id in discard_req_ids
            ):
                continue
            assert slot not in seen_slots, (
                f"Duplicate slot {slot} for req_id={req_id!r} in chunk"
            )
            seen_slots.add(slot)
            slots_np[batch_i - start_idx] = slot
            any_active = True
        if not any_active:
            return None
        return torch.from_numpy(slots_np).to(device)

    def precompile(
        self,
        device: torch.device,
        num_reqs_paddings: Sequence[int],
        verify_logits_paddings: Sequence[int],
        max_draft_tokens: int,
        dummy_logits_fn: Callable[[int], torch.Tensor],
        sync_fn: Callable[..., None],
        precompile_top_tokens: bool = False,
    ) -> None:
        """Pre-warm all thinking-budget subgraphs across compiled bucket shapes."""
        if self._markers is None:
            return
        markers = self._markers
        state_table = self.ensure_state_table(device)

        # Allocate warmup inputs on CPU and upload via .to(device) (mirroring
        # TPUModelRunner._precompile_rejection_sampler) so compiled kernels are
        # traced on genuine H2D PJRT buffers rather than lazy device splats.
        seed_slots = torch.zeros(self._max_num_reqs, dtype=torch.int32).to(device)
        seed_vals = torch.zeros((self._max_num_reqs, NUM_STATE_COLS), dtype=torch.int32)
        seed_vals[:, COL_BUDGET] = -1
        state_table = seed_states(state_table, seed_slots, seed_vals.to(device))
        sync_fn(state_table)

        for num_reqs in num_reqs_paddings:
            slot_ids = torch.zeros(num_reqs, dtype=torch.int32).to(device)
            selected = torch.zeros((num_reqs, 1), dtype=torch.int64).to(device)
            out_plain, state_table = override_and_advance_plain(
                state_table, slot_ids, selected, markers
            )
            sync_fn([out_plain, state_table])
            logger.info("  -- plain num_seqs: %d", num_reqs)

        self._state_table = state_table
        if max_draft_tokens <= 0 or not verify_logits_paddings:
            return

        spec_slots = torch.zeros(self._max_num_reqs, dtype=torch.int32).to(device)
        for num_tokens in verify_logits_paddings:
            draft_ids = torch.zeros(num_tokens, dtype=torch.int32).to(device)
            seg_ids = torch.full(
                (num_tokens,), self._max_num_reqs, dtype=torch.int64
            ).to(device)
            grp_ids = torch.zeros(num_tokens, dtype=torch.int32).to(device)
            logits = dummy_logits_fn(num_tokens)
            out_logits, eff_drafts = prepare_spec_verify_logits(
                state_table,
                spec_slots,
                draft_ids,
                logits,
                seg_ids,
                grp_ids,
                max_draft_tokens,
                markers,
            )
            sync_fn([out_logits, eff_drafts])
            if precompile_top_tokens:
                out_toks, eff_top_drafts = prepare_spec_verify_tokens(
                    state_table,
                    spec_slots,
                    draft_ids,
                    torch.zeros(num_tokens, dtype=torch.int64).to(device),
                    seg_ids,
                    grp_ids,
                    max_draft_tokens,
                    markers,
                )
                sync_fn([out_toks, eff_top_drafts])

        for num_reqs in num_reqs_paddings:
            out_spec, state_table = advance_spec_states(
                state_table,
                spec_slots,
                torch.full(
                    (num_reqs, max_draft_tokens + 1),
                    NO_FORCED_TOKEN,
                    dtype=torch.int32,
                ).to(device),
                max_draft_tokens,
                markers,
            )
            sync_fn([out_spec, state_table])
        self._state_table = state_table


@dataclass
class ThinkingBudgetStepState:
    """Per-step thinking-budget state threaded through ``sample_tokens``."""

    tracker: ThinkingBudgetTracker
    state_table: torch.Tensor
    markers: ThinkingBudgetMarkers
    discard_req_ids: AbstractSet[str]

    def commit(self, state_table: torch.Tensor) -> None:
        self.state_table = state_table
        self.tracker.commit_state_table(state_table)

    def apply_plain(
        self,
        device: torch.device,
        req_ids: Sequence[str | None],
        start_idx: int,
        end_idx: int,
        selected: torch.Tensor,
    ) -> torch.Tensor:
        slot_ids = self.tracker.build_chunk_slot_ids(
            device,
            req_ids,
            start_idx,
            end_idx,
            selected.shape[0],
            self.discard_req_ids,
        )
        if slot_ids is None:
            return selected
        out_tokens, new_table = override_and_advance_plain(
            self.state_table, slot_ids, selected, self.markers
        )
        self.commit(new_table)
        return out_tokens

    def spec_chunk(
        self,
        device: torch.device,
        req_ids: Sequence[str | None],
        start_idx: int,
        end_idx: int,
        min_slot_len: int,
    ) -> "ThinkingBudgetSpecChunk | None":
        slot_ids = self.tracker.build_chunk_slot_ids(
            device,
            req_ids,
            start_idx,
            end_idx,
            max(self.tracker.max_num_reqs, min_slot_len),
            self.discard_req_ids,
        )
        if slot_ids is None:
            return None
        return ThinkingBudgetSpecChunk(step=self, slot_ids=slot_ids)


@dataclass(frozen=True)
class ThinkingBudgetSpecChunk:
    """Per-chunk speculative verification state for thinking budgets."""

    step: ThinkingBudgetStepState
    slot_ids: torch.Tensor

    def prepare_logits(
        self,
        draft_token_ids: torch.Tensor,
        target_logits: torch.Tensor,
        md: "SpecDecodeMetadata",
        max_draft_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return prepare_spec_verify_logits(
            self.step.state_table,
            self.slot_ids,
            draft_token_ids,
            target_logits,
            md.segment_ids,
            md.group_indices,
            max_draft_tokens,
            self.step.markers,
        )

    def prepare_tokens(
        self,
        draft_token_ids: torch.Tensor,
        target_argmax: torch.Tensor,
        md: "SpecDecodeMetadata",
        max_draft_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return prepare_spec_verify_tokens(
            self.step.state_table,
            self.slot_ids,
            draft_token_ids,
            target_argmax,
            md.segment_ids,
            md.group_indices,
            max_draft_tokens,
            self.step.markers,
        )

    def finish(self, out: torch.Tensor, max_draft_tokens: int) -> torch.Tensor:
        out, new_table = advance_spec_states(
            self.step.state_table,
            self.slot_ids,
            out,
            max_draft_tokens,
            self.step.markers,
        )
        self.step.commit(new_table)
        return out
