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
"""Compiled KMP state transitions and logits/token forcing for thinking budgets.

Maintains a fixed-shape ``[capacity, 7]`` ``int32`` Knuth-Morris-Pratt (KMP)
state table on device and shares one transition definition across host sequence
replay and ``@torch.compile(backend="tpu", fullgraph=True, dynamic=False)``
plain-decode and speculative-verification kernels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm.config.reasoning import ReasoningConfig

INT32_MAX: Final[int] = 2_147_483_647
NO_FORCED_TOKEN: Final[int] = -1
_AMAX_SENTINEL: Final[int] = torch.iinfo(torch.int32).min

# Per-slot int32 columns in ``state_table`` ([capacity, NUM_STATE_COLS]).
# ``COL_FORCING_END`` is 0 when inactive, 1 while matching an active ``end_ids``
# prefix without a completed ``natural_end_ids``, and 2 once ``natural_end_ids``
# completed earlier in that ``end_ids`` prefix (so an aborted prefix closes
# ``in_think``).
COL_BUDGET: Final[int] = 0
COL_IN_THINK: Final[int] = 1
COL_THINK_COUNT: Final[int] = 2
COL_START_MATCH: Final[int] = 3
COL_NAT_END_MATCH: Final[int] = 4
COL_END_MATCH: Final[int] = 5
COL_FORCING_END: Final[int] = 6
NUM_STATE_COLS: Final[int] = 7

type StateTuple = tuple[int, int, int, int, int, int, int]
UNTRACKED_STATE: Final[StateTuple] = (-1, 0, 0, 0, 0, 0, 0)

type StateCols[TVal: (int, torch.Tensor)] = tuple[
    TVal, TVal, TVal, TVal, TVal, TVal, TVal
]


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
        cls, reasoning_config: ReasoningConfig | None
    ) -> ThinkingBudgetMarkers | None:
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


# ---------------------------------------------------------------------------
# Unified KMP state machine (shared by Python replay and compiled TPU kernels)
# ---------------------------------------------------------------------------


def _const[TVal: (int, torch.Tensor)](like: TVal, val: int) -> TVal:
    if isinstance(like, int):
        return cast(TVal, val)
    if val == 0:
        return cast(TVal, torch.zeros_like(like))
    if val == 1:
        return cast(TVal, torch.ones_like(like))
    return cast(TVal, torch.full_like(like, val))


def _not(cond: bool | torch.Tensor) -> bool | torch.Tensor:
    return ~cond if isinstance(cond, torch.Tensor) else not cond


def _and(a: bool | torch.Tensor, b: bool | torch.Tensor) -> bool | torch.Tensor:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a & b
    assert isinstance(a, bool) and isinstance(b, bool)
    return a and b


def _or(a: bool | torch.Tensor, b: bool | torch.Tensor) -> bool | torch.Tensor:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a | b
    assert isinstance(a, bool) and isinstance(b, bool)
    return a or b


def _where[TVal: (int, torch.Tensor)](
    cond: bool | torch.Tensor, a: TVal, b: TVal
) -> TVal:
    if isinstance(cond, torch.Tensor):
        assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
        return cast(TVal, torch.where(cond, a, b))
    assert isinstance(a, int) and isinstance(b, int)
    return a if cond else b


def _where_cols[TVal: (int, torch.Tensor)](
    cond: bool | torch.Tensor, a: StateCols[TVal], b: StateCols[TVal]
) -> StateCols[TVal]:
    if isinstance(cond, torch.Tensor):
        return (
            _where(cond, a[0], b[0]),
            _where(cond, a[1], b[1]),
            _where(cond, a[2], b[2]),
            _where(cond, a[3], b[3]),
            _where(cond, a[4], b[4]),
            _where(cond, a[5], b[5]),
            _where(cond, a[6], b[6]),
        )
    return a if cond else b


def _kmp_step[TVal: (int, torch.Tensor)](
    pattern: tuple[int, ...],
    fail: tuple[int, ...],
    match_len: TVal,
    tok: TVal,
) -> TVal:
    zero = _const(match_len, 0)
    if len(pattern) == 1:
        return _where(tok == pattern[0], _const(match_len, 1), zero)
    out_for_q: list[TVal] = []
    for q, pat_tok in enumerate(pattern):
        fallback = zero if q == 0 else out_for_q[fail[q]]
        out_for_q.append(_where(tok == pat_tok, _const(match_len, q + 1), fallback))
    res = out_for_q[0]
    for q in range(1, len(pattern)):
        res = _where(match_len == q, out_for_q[q], res)
    return res


def _gather_end_token[TVal: (int, torch.Tensor)](
    end_ids: tuple[int, ...], end_match: TVal
) -> TVal:
    res = _const(end_match, end_ids[0])
    for q in range(1, len(end_ids)):
        res = _where(end_match == q, _const(end_match, end_ids[q]), res)
    return res


def _query_forced_token_cols[TVal: (int, torch.Tensor)](
    budget: TVal,
    in_think: TVal,
    think_count: TVal,
    end_match: TVal,
    forcing_end: TVal,
    end_ids: tuple[int, ...],
) -> TVal:
    should_force = _and(
        _and(budget >= 0, in_think != 0),
        _or(forcing_end != 0, think_count >= budget),
    )
    return _where(
        should_force,
        _gather_end_token(end_ids, end_match),
        _const(budget, NO_FORCED_TOKEN),
    )


def _advance_state_cols[TVal: (int, torch.Tensor)](
    st: StateCols[TVal],
    tok: TVal,
    start_ids: tuple[int, ...],
    start_fail: tuple[int, ...],
    nat_ids: tuple[int, ...],
    nat_fail: tuple[int, ...],
    end_ids: tuple[int, ...],
    end_fail: tuple[int, ...],
) -> StateCols[TVal]:
    (
        budget,
        in_think,
        think_count,
        start_match,
        nat_end_match,
        end_match,
        forcing_end,
    ) = st
    zero = _const(budget, 0)
    one = _const(budget, 1)
    two = _const(budget, 2)

    next_start = _kmp_step(start_ids, start_fail, start_match, tok)
    entered_think = next_start == len(start_ids)
    out_not_in_think: StateCols[TVal] = (
        budget,
        _where(entered_think, one, zero),
        zero,
        _where(entered_think, zero, next_start),
        zero,
        zero,
        zero,
    )

    forcing_now = _or(forcing_end != 0, think_count >= budget)
    expected_end_tok = _gather_end_token(end_ids, end_match)
    actively_forcing_seq = _and(forcing_now, tok == expected_end_tok)
    prev_had_nat_close = forcing_end == 2

    next_end = _kmp_step(end_ids, end_fail, end_match, tok)
    end_completed = next_end == len(end_ids)
    wrapped_end = _where(
        end_completed, _const(next_end, end_fail[len(end_ids)]), next_end
    )
    if (nat_ids, nat_fail) == (end_ids, end_fail):
        nat_completed = end_completed
        wrapped_nat = wrapped_end
    else:
        next_nat = _kmp_step(nat_ids, nat_fail, nat_end_match, tok)
        nat_completed = next_nat == len(nat_ids)
        wrapped_nat = _where(
            nat_completed, _const(next_nat, nat_fail[len(nat_ids)]), next_nat
        )

    closed = _or(end_completed, _and(_not(actively_forcing_seq), nat_completed))
    inc_count = _where(
        think_count < INT32_MAX,
        think_count + 1,
        _const(think_count, INT32_MAX),
    )
    shadow_start = _where(
        _and(actively_forcing_seq, _and(prev_had_nat_close, _not(nat_completed))),
        _where(entered_think, zero, next_start),
        zero,
    )
    next_forcing = _where(
        actively_forcing_seq,
        _where(_or(prev_had_nat_close, nat_completed), two, one),
        zero,
    )
    out_in_think: StateCols[TVal] = (
        budget,
        _where(closed, zero, one),
        _where(closed, zero, inc_count),
        _where(closed, zero, shadow_start),
        _where(closed, zero, wrapped_nat),
        _where(closed, zero, wrapped_end),
        _where(closed, zero, next_forcing),
    )

    aborted_after_nat_close = _and(prev_had_nat_close, _not(actively_forcing_seq))
    use_in_think = _and(in_think != 0, _not(aborted_after_nat_close))
    active_out = _where_cols(use_in_think, out_in_think, out_not_in_think)
    untracked: StateCols[TVal] = (
        _const(budget, -1),
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
    )
    return _where_cols(budget >= 0, active_out, untracked)


def advance_state(
    state: StateTuple, tok: int, markers: ThinkingBudgetMarkers
) -> StateTuple:
    """Advance scalar 7-tuple KMP thinking-budget state on one token ``tok``."""
    return _advance_state_cols(
        state,
        tok,
        markers.start_ids,
        markers.start_fail,
        markers.natural_end_ids,
        markers.natural_end_fail,
        markers.end_ids,
        markers.end_fail,
    )


def query_forced_token(state: StateTuple, markers: ThinkingBudgetMarkers) -> int:
    """Return the forced token ID for ``state``, or ``NO_FORCED_TOKEN`` (-1)."""
    return _query_forced_token_cols(
        state[COL_BUDGET],
        state[COL_IN_THINK],
        state[COL_THINK_COUNT],
        state[COL_END_MATCH],
        state[COL_FORCING_END],
        markers.end_ids,
    )


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
            state = advance_state(state, int(tok), markers)
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
            state = advance_state(state, int(arr[pos]), markers)
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
            state = advance_state(state, int(arr[pos]), markers)
            pos += 1
            hit_ptr += 1
    return state


def compute_sequence_state(
    tokens: Sequence[int] | np.ndarray,
    budget: int,
    markers: ThinkingBudgetMarkers,
) -> StateTuple:
    """Compute the 7-tuple KMP state after consuming ``tokens``."""
    init: StateTuple = (min(int(budget), INT32_MAX), 0, 0, 0, 0, 0, 0)
    return advance_sequence_state(init, tokens, markers)


def _query_forced_token_tensor(
    st: torch.Tensor, end_ids: tuple[int, ...]
) -> torch.Tensor:
    return _query_forced_token_cols(
        st[:, COL_BUDGET],
        st[:, COL_IN_THINK],
        st[:, COL_THINK_COUNT],
        st[:, COL_END_MATCH],
        st[:, COL_FORCING_END],
        end_ids,
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
    cols = (
        st[:, COL_BUDGET],
        st[:, COL_IN_THINK],
        st[:, COL_THINK_COUNT],
        st[:, COL_START_MATCH],
        st[:, COL_NAT_END_MATCH],
        st[:, COL_END_MATCH],
        st[:, COL_FORCING_END],
    )
    out_cols = _advance_state_cols(
        cols,
        tok,
        start_ids,
        start_fail,
        nat_ids,
        nat_fail,
        end_ids,
        end_fail,
    )
    return torch.stack(out_cols, dim=-1)


# ---------------------------------------------------------------------------
# Shared scatter and logits-forcing primitives
# ---------------------------------------------------------------------------


def _broadcast_amax_scatter(
    base: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Return ``base`` with ``out[indices[i]] = values[i]`` where ``valid[i]``.

    Uses a masked broadcast + ``amax`` reduction (``<= 7 MiB`` at
    ``max_num_reqs=512``) rather than ``Tensor.scatter(0, ...)``, whose XLA
    lowering for dynamic colliding indices adds ~7 ms/step on TPU v7x. Active
    ``indices[valid]`` must be unique within ``[0, base.shape[0])``.
    """
    if indices.shape[0] == 0:
        return base
    if base.device.type == "cpu" and not torch.compiler.is_compiling():
        active_idx = indices[valid]
        assert active_idx.numel() == torch.unique(active_idx).numel(), (
            "Duplicate active indices passed to _broadcast_amax_scatter"
        )
    dest_idx = torch.arange(
        base.shape[0], dtype=indices.dtype, device=base.device
    ).unsqueeze(1)
    match = valid.unsqueeze(0) & (dest_idx == indices.unsqueeze(0))
    trailing_ones = (1,) * (base.ndim - 1)
    match_view = match.reshape(*match.shape, *trailing_ones)
    cand = torch.where(
        match_view,
        values.to(base.dtype).unsqueeze(0),
        _AMAX_SENTINEL,
    ).amax(dim=1)
    any_match = match.any(dim=1).reshape(base.shape[0], *trailing_ones)
    return torch.where(any_match, cand, base)


def _scatter_state_rows(
    state_table: torch.Tensor, slot_ids: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    return _broadcast_amax_scatter(state_table, slot_ids, rows, slot_ids > 0)


def _force_logits_rows_with_mask(
    logits: torch.Tensor, forced_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask ``logits`` to one-hot ``0.0`` at ``forced_ids`` when permitted.

    If ``logits[row, forced_id]`` is already ``-inf`` (for example, masked out
    by structured-output grammar or ``min_tokens``), the row is left unchanged
    so sampling never emits a disallowed token or produces all-``-inf`` NaNs.
    """
    safe_ids = forced_ids.clamp(min=0, max=logits.shape[-1] - 1).to(torch.int64)
    orig_forced = logits.gather(-1, safe_ids.unsqueeze(-1))
    can_force = (forced_ids >= 0).unsqueeze(-1) & ~torch.isneginf(orig_forced)
    masked = torch.where(can_force, float("-inf"), logits)
    forced_val = torch.where(can_force, torch.zeros_like(orig_forced), orig_forced)
    return masked.scatter(-1, safe_ids.unsqueeze(-1), forced_val), can_force.squeeze(-1)


# ---------------------------------------------------------------------------
# Compiled TPU kernels: seeding and plain decode
# ---------------------------------------------------------------------------


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def seed_states(
    state_table: torch.Tensor,
    seed_slots: torch.Tensor,
    seed_values: torch.Tensor,
) -> torch.Tensor:
    """Scatter ``seed_values`` into ``state_table`` at active ``seed_slots``."""
    return _scatter_state_rows(state_table, seed_slots, seed_values)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def mask_plain_logits(
    state_table: torch.Tensor,
    slot_ids: torch.Tensor,
    logits: torch.Tensor,
    end_ids: tuple[int, ...],
) -> torch.Tensor:
    """Mask ``logits`` rows whose slot requires a forced ``end_ids`` token."""
    active_slot = slot_ids > 0
    safe_slots = torch.where(active_slot, slot_ids, torch.zeros_like(slot_ids)).to(
        torch.int64
    )
    st = torch.index_select(state_table, 0, safe_slots)
    no_tok = torch.full_like(slot_ids, NO_FORCED_TOKEN)
    forced_ids = torch.where(
        active_slot, _query_forced_token_tensor(st, end_ids), no_tok
    )
    forced_logits, _ = _force_logits_rows_with_mask(logits, forced_ids)
    return forced_logits


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def advance_plain_states(
    state_table: torch.Tensor,
    slot_ids: torch.Tensor,
    selected_token_ids: torch.Tensor,
    start_ids: tuple[int, ...],
    start_fail: tuple[int, ...],
    nat_ids: tuple[int, ...],
    nat_fail: tuple[int, ...],
    end_ids: tuple[int, ...],
    end_fail: tuple[int, ...],
) -> torch.Tensor:
    """Advance ``state_table`` for active ``slot_ids`` on ``selected_token_ids``."""
    active_slot = slot_ids > 0
    safe_slots = torch.where(active_slot, slot_ids, torch.zeros_like(slot_ids)).to(
        torch.int64
    )
    st = torch.index_select(state_table, 0, safe_slots)
    next_st = _advance_state_tensor(
        st,
        selected_token_ids.reshape(-1).to(torch.int32),
        start_ids,
        start_fail,
        nat_ids,
        nat_fail,
        end_ids,
        end_fail,
    )
    return _scatter_state_rows(state_table, slot_ids, next_st)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def override_and_advance_plain(
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
    """Override ``TPU_GREEDY_TOP_TOKENS`` IDs and advance ``state_table``."""
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
        out_tokens.reshape(-1).to(torch.int32),
        start_ids,
        start_fail,
        nat_ids,
        nat_fail,
        end_ids,
        end_fail,
    )
    return out_tokens, _scatter_state_rows(state_table, slot_ids, next_st)


# ---------------------------------------------------------------------------
# Compiled TPU kernels: speculative verification
# ---------------------------------------------------------------------------


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def prepare_spec_verify_drafts(
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
    """Compute per-draft-row and post-draft bonus forced token IDs.

    ``slot_ids`` is sized to ``max_num_reqs`` during serving so this compiled
    graph depends only on ``draft_token_ids.shape[0]``
    (``verify_logits_paddings``) rather than the ``(num_tokens, num_reqs)``
    Cartesian product. ``SpecDecodeMetadata`` marks padded draft positions with
    ``segment_ids == real_num_reqs``, where ``slot_ids[real_num_reqs] == 0``
    (or ``real_num_reqs == max_num_reqs``, outside ``[0, num_reqs)``).
    """
    num_reqs = slot_ids.shape[0]
    active_slot = slot_ids > 0
    safe_slots = torch.where(active_slot, slot_ids, torch.zeros_like(slot_ids)).to(
        torch.int64
    )
    st = torch.index_select(state_table, 0, safe_slots)
    draft_i32 = draft_token_ids.to(torch.int32)
    no_tok = torch.full_like(draft_i32, NO_FORCED_TOKEN)
    no_req_tok = torch.full(
        (num_reqs,), NO_FORCED_TOKEN, dtype=torch.int32, device=state_table.device
    )

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
        empty_cells = torch.full(
            (num_cells,),
            NO_FORCED_TOKEN,
            dtype=torch.int32,
            device=state_table.device,
        )
        draft_grid = _broadcast_amax_scatter(
            empty_cells, flat_idx, draft_i32, in_bounds
        ).reshape(num_reqs, max_draft_tokens)

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

    forced_bonus_full = torch.where(
        active_slot, _query_forced_token_tensor(st, end_ids), no_req_tok
    )
    return forced_target_ids, forced_bonus_full


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def force_spec_target_logits(
    target_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    forced_target_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask ``target_logits`` and invalidate mismatched ``draft_token_ids``."""
    forced_logits, did_force = _force_logits_rows_with_mask(
        target_logits, forced_target_ids
    )
    draft_i32 = draft_token_ids.to(torch.int32)
    effective_draft_ids = torch.where(
        did_force & (draft_i32 != forced_target_ids),
        torch.full_like(draft_i32, NO_FORCED_TOKEN),
        draft_i32,
    )
    return forced_logits, effective_draft_ids


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def force_bonus_logits(
    bonus_logits: torch.Tensor,
    forced_bonus_full: torch.Tensor,
) -> torch.Tensor:
    """Mask ``bonus_logits`` rows from ``forced_bonus_full`` before sampling."""
    forced_bonus = forced_bonus_full[: bonus_logits.shape[0]]
    forced_logits, _ = _force_logits_rows_with_mask(bonus_logits, forced_bonus)
    return forced_logits


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def force_spec_target_tokens(
    target_argmax: torch.Tensor,
    draft_token_ids: torch.Tensor,
    forced_target_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Override greedy ``target_argmax`` and invalidate mismatched drafts."""
    did_force = forced_target_ids >= 0
    forced_argmax = torch.where(
        did_force, forced_target_ids.to(target_argmax.dtype), target_argmax
    )
    draft_i32 = draft_token_ids.to(torch.int32)
    effective_draft_ids = torch.where(
        did_force & (draft_i32 != forced_target_ids),
        torch.full_like(draft_i32, NO_FORCED_TOKEN),
        draft_i32,
    )
    return forced_argmax, effective_draft_ids


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def force_bonus_tokens(
    bonus_token_ids: torch.Tensor,
    forced_bonus_full: torch.Tensor,
) -> torch.Tensor:
    """Override greedy ``bonus_token_ids`` from ``forced_bonus_full``."""
    forced_bonus = forced_bonus_full[: bonus_token_ids.shape[0]]
    return torch.where(
        forced_bonus >= 0,
        forced_bonus.to(bonus_token_ids.dtype),
        bonus_token_ids,
    )


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def advance_spec_states(
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
) -> torch.Tensor:
    """Advance ``state_table`` across accepted/recovered/bonus ``next_tokens``."""
    chunk_slots = slot_ids[: next_tokens.shape[0]]
    active_slot = chunk_slots > 0
    safe_slots = torch.where(
        active_slot, chunk_slots, torch.zeros_like(chunk_slots)
    ).to(torch.int64)
    st = torch.index_select(state_table, 0, safe_slots)
    for col_idx in range(max_draft_tokens + 1):
        tok = next_tokens[:, col_idx].to(torch.int32)
        valid_col = active_slot & (tok >= 0)
        next_st = _advance_state_tensor(
            st,
            tok,
            start_ids,
            start_fail,
            nat_ids,
            nat_fail,
            end_ids,
            end_fail,
        )
        st = torch.where(valid_col.unsqueeze(-1), next_st, st)
    return _scatter_state_rows(state_table, chunk_slots, st)
