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
"""Request-ID-keyed ``SamplingParams.thinking_token_budget`` runner state.

``TPUModelRunner`` runs its own compiled sampling and speculative rejection
pipeline, reorders batch rows mid-step in ``_reorder_batch_for_rpa``, and under
``--async-scheduling`` leaves the previous step's sampled tokens unsettled on
device when ``sample_tokens()`` begins.

``ThinkingBudgetTracker`` manages request-lifecycle slot ownership and delegates
compiled KMP state transitions and logits/token forcing to
``vllm_torchtpu.layers.adapter.sample.thinking_budget``.
"""

from __future__ import annotations

from collections.abc import Callable, Container, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm_torchtpu.layers.adapter.sample import thinking_budget as tb_kernels
from vllm_torchtpu.layers.adapter.sample.thinking_budget import (
    COL_BUDGET,
    INT32_MAX,
    NO_FORCED_TOKEN,
    NUM_STATE_COLS,
    StateTuple,
    ThinkingBudgetMarkers,
    advance_sequence_state,
    compute_sequence_state,
)
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm import SamplingParams
    from vllm.config.reasoning import ReasoningConfig
    from vllm.v1.worker.gpu_input_batch import CachedRequestState

    from vllm_torchtpu.runner.speculative_decoding_manager import SpecDecodeMetadata

logger = init_logger(__name__)


def _compute_request_state(
    req_state: CachedRequestState | None,
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
        reasoning_config: ReasoningConfig | None,
        max_num_reqs: int,
    ) -> None:
        self._markers = ThinkingBudgetMarkers.from_reasoning_config(reasoning_config)
        self._max_num_reqs = max(int(max_num_reqs), 1)
        # Slot 0 is the permanent untracked sentinel; slots 1..max_num_reqs
        # serve budgeted requests currently in input_batch. Unscheduled /
        # preempted requests release their slots and reseed on re-entry.
        self._capacity = self._max_num_reqs + 1
        self._budgets: dict[str, int] = {}
        self._slot_by_req_id: dict[str, int] = {}
        self._free_slots: list[int] = list(range(self._capacity - 1, 0, -1))
        self._dirty_states: dict[str, StateTuple] = {}
        self._state_table: torch.Tensor | None = None
        self._last_slot_ids: tuple[tuple[int, bytes], torch.Tensor] | None = None

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
                self._last_slot_ids = None
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

    def release_unscheduled(self, active_req_ids: Container[str]) -> None:
        """Free slots for budgeted requests no longer in ``input_batch``."""
        if not self._slot_by_req_id and not self._dirty_states:
            return
        for req_id in list(self._slot_by_req_id):
            if req_id not in active_req_ids:
                self._dirty_states.pop(req_id, None)
                self._release_slot(req_id)
        for req_id in list(self._dirty_states):
            if req_id not in active_req_ids:
                self._dirty_states.pop(req_id, None)

    def add(
        self,
        req_id: str,
        sampling_params: SamplingParams | None,
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
        requests: Mapping[str, CachedRequestState],
        discard_req_ids: AbstractSet[str],
    ) -> ThinkingBudgetStepState | None:
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
        requests: Mapping[str, CachedRequestState],
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
        requests: Mapping[str, CachedRequestState],
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

        batch_set = set(batch_tracked)
        for owner_id in list(self._slot_by_req_id):
            if owner_id not in batch_set:
                self._dirty_states.pop(owner_id, None)
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
                state_table = tb_kernels.seed_states(
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
        key = (padded_num_reqs, slots_np.tobytes())
        prev = self._last_slot_ids
        if (
            prev is not None
            and prev[0] == key
            and _same_target_device(prev[1].device, device)
        ):
            return prev[1]
        dev = torch.from_numpy(slots_np).to(device)
        self._last_slot_ids = (key, dev)
        return dev

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
        kernel_args = markers.kernel_args()
        state_table = self.ensure_state_table(device)

        seed_slots = torch.zeros(self._max_num_reqs, dtype=torch.int32).to(device)
        seed_vals = torch.zeros((self._max_num_reqs, NUM_STATE_COLS), dtype=torch.int32)
        seed_vals[:, COL_BUDGET] = -1
        state_table = tb_kernels.seed_states(
            state_table, seed_slots, seed_vals.to(device)
        )
        sync_fn(state_table)

        for num_reqs in num_reqs_paddings:
            slot_ids = torch.zeros(num_reqs, dtype=torch.int32).to(device)
            selected = torch.zeros((num_reqs, 1), dtype=torch.int64).to(device)
            masked_plain = tb_kernels.mask_plain_logits(
                state_table, slot_ids, dummy_logits_fn(num_reqs), markers.end_ids
            )
            state_table = tb_kernels.advance_plain_states(
                state_table, slot_ids, selected, *kernel_args
            )
            sync_fn([masked_plain, state_table])
            if precompile_top_tokens:
                out_plain, state_table = tb_kernels.override_and_advance_plain(
                    state_table, slot_ids, selected, *kernel_args
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
            forced_tgt, forced_bonus = tb_kernels.prepare_spec_verify_drafts(
                state_table,
                spec_slots,
                draft_ids,
                seg_ids,
                grp_ids,
                max_draft_tokens,
                *kernel_args,
            )
            out_logits, eff_drafts = tb_kernels.force_spec_target_logits(
                dummy_logits_fn(num_tokens), draft_ids, forced_tgt
            )
            sync_fn([forced_bonus, out_logits, eff_drafts])
            if precompile_top_tokens:
                out_toks, eff_top_drafts = tb_kernels.force_spec_target_tokens(
                    torch.zeros(num_tokens, dtype=torch.int64).to(device),
                    draft_ids,
                    forced_tgt,
                )
                sync_fn([out_toks, eff_top_drafts])

        forced_bonus_full = torch.full(
            (self._max_num_reqs,), NO_FORCED_TOKEN, dtype=torch.int32
        ).to(device)
        for num_reqs in num_reqs_paddings:
            out_bonus_logits = tb_kernels.force_bonus_logits(
                dummy_logits_fn(num_reqs), forced_bonus_full
            )
            if precompile_top_tokens:
                out_bonus_toks = tb_kernels.force_bonus_tokens(
                    torch.zeros(num_reqs, dtype=torch.int64).to(device),
                    forced_bonus_full,
                )
                sync_fn(out_bonus_toks)
            state_table = tb_kernels.advance_spec_states(
                state_table,
                spec_slots,
                torch.full(
                    (num_reqs, max_draft_tokens + 1),
                    NO_FORCED_TOKEN,
                    dtype=torch.int32,
                ).to(device),
                max_draft_tokens,
                *kernel_args,
            )
            sync_fn([out_bonus_logits, state_table])
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

    def chunk_slot_ids(
        self,
        device: torch.device,
        req_ids: Sequence[str | None],
        start_idx: int,
        end_idx: int,
        padded_num_reqs: int,
    ) -> torch.Tensor | None:
        return self.tracker.build_chunk_slot_ids(
            device,
            req_ids,
            start_idx,
            end_idx,
            padded_num_reqs,
            self.discard_req_ids,
        )

    def mask_plain_logits(
        self, logits: torch.Tensor, slot_ids: torch.Tensor
    ) -> torch.Tensor:
        return tb_kernels.mask_plain_logits(
            self.state_table, slot_ids, logits, self.markers.end_ids
        )

    def advance_plain(self, slot_ids: torch.Tensor, selected: torch.Tensor) -> None:
        new_table = tb_kernels.advance_plain_states(
            self.state_table, slot_ids, selected, *self.markers.kernel_args()
        )
        self.commit(new_table)

    def override_and_advance_plain(
        self,
        device: torch.device,
        req_ids: Sequence[str | None],
        start_idx: int,
        end_idx: int,
        selected: torch.Tensor,
    ) -> torch.Tensor:
        slot_ids = self.chunk_slot_ids(
            device, req_ids, start_idx, end_idx, selected.shape[0]
        )
        if slot_ids is None:
            return selected
        out_tokens, new_table = tb_kernels.override_and_advance_plain(
            self.state_table, slot_ids, selected, *self.markers.kernel_args()
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
    ) -> ThinkingBudgetSpecChunk | None:
        slot_ids = self.chunk_slot_ids(
            device,
            req_ids,
            start_idx,
            end_idx,
            max(self.tracker.max_num_reqs, min_slot_len),
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
        bonus_logits: torch.Tensor,
        md: SpecDecodeMetadata,
        max_draft_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        forced_target_ids, forced_bonus_full = tb_kernels.prepare_spec_verify_drafts(
            self.step.state_table,
            self.slot_ids,
            draft_token_ids,
            md.segment_ids,
            md.group_indices,
            max_draft_tokens,
            *self.step.markers.kernel_args(),
        )
        out_target, eff_drafts = tb_kernels.force_spec_target_logits(
            target_logits, draft_token_ids, forced_target_ids
        )
        out_bonus = tb_kernels.force_bonus_logits(bonus_logits, forced_bonus_full)
        return out_target, out_bonus, eff_drafts

    def prepare_tokens(
        self,
        draft_token_ids: torch.Tensor,
        target_argmax: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        md: SpecDecodeMetadata,
        max_draft_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        forced_target_ids, forced_bonus_full = tb_kernels.prepare_spec_verify_drafts(
            self.step.state_table,
            self.slot_ids,
            draft_token_ids,
            md.segment_ids,
            md.group_indices,
            max_draft_tokens,
            *self.step.markers.kernel_args(),
        )
        out_target, eff_drafts = tb_kernels.force_spec_target_tokens(
            target_argmax, draft_token_ids, forced_target_ids
        )
        out_bonus = tb_kernels.force_bonus_tokens(bonus_token_ids, forced_bonus_full)
        return out_target, out_bonus, eff_drafts

    def finish(self, out: torch.Tensor, max_draft_tokens: int) -> None:
        new_table = tb_kernels.advance_spec_states(
            self.step.state_table,
            self.slot_ids,
            out,
            max_draft_tokens,
            *self.step.markers.kernel_args(),
        )
        self.step.commit(new_table)
