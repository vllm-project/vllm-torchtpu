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
"""Reusable request-ID-keyed device-slot table for stateful sampling adapters.

``TPUModelRunner`` reorders batch rows mid-step in ``_reorder_batch_for_rpa``
and under ``--async-scheduling`` leaves unsettled output placeholders in flight
when ``sample_tokens()`` begins. Stateful sampling adapters (such as
``thinking_token_budget``, token penalties, or ``bad_words``) can own a
``RequestSlotTable`` to maintain fixed-shape ``[max_num_reqs + 1, num_cols]``
per-request state on device (with slot ``0`` reserved as an untracked sentinel),
release slots across unscheduled waves, and reuse cached per-chunk
``slot_ids`` H2D uploads across steady-state decode steps.
"""

from collections.abc import Callable, Container, Iterable, Sequence
from collections.abc import Set as AbstractSet

import numpy as np
import torch

from vllm_torchtpu.layers.adapter.sample.scatter_ops import seed_slot_rows


class RequestSlotTable[RowT: Sequence[int]]:
    """Request-ID-keyed ``[max_num_reqs + 1, num_cols]`` int32 device-slot pool.

    Slot ``0`` is a permanent untracked sentinel row; slots ``1..max_num_reqs``
    are leased to active requests in ``input_batch`` and released when requests
    finish, are preempted, or rotate out in unscheduled waves. The table lives
    on ``device`` and is allocated on first use.
    """

    def __init__(
        self,
        max_num_reqs: int,
        num_cols: int,
        device: torch.device,
        sentinel_row: Sequence[int] | None = None,
    ) -> None:
        self._max_num_reqs = int(max_num_reqs)
        if self._max_num_reqs <= 0:
            raise ValueError(f"max_num_reqs must be positive, got {max_num_reqs}")
        self._num_cols = int(num_cols)
        if self._num_cols <= 0:
            raise ValueError(f"num_cols must be positive, got {num_cols}")
        self._device = device
        default_row = (
            tuple(int(v) for v in sentinel_row)
            if sentinel_row is not None
            else (0,) * self._num_cols
        )
        if len(default_row) != self._num_cols:
            raise ValueError(
                f"sentinel_row length {len(default_row)} != num_cols {self._num_cols}"
            )
        self._sentinel_row: tuple[int, ...] = default_row
        self._capacity = self._max_num_reqs + 1
        self._slot_by_req_id: dict[str, int] = {}
        self._free_slots: list[int] = list(range(self._capacity - 1, 0, -1))
        self._dirty_rows: dict[str, RowT] = {}
        self._table: torch.Tensor | None = None
        # Last slot-ID upload per chunk (start_idx, width) with the host bytes
        # it came from, so every chunk of a multi-chunk step reuses its own.
        self._slot_ids_cache: dict[tuple[int, int], tuple[bytes, torch.Tensor]] = {}

    @property
    def max_num_reqs(self) -> int:
        return self._max_num_reqs

    @property
    def device(self) -> torch.device:
        return self._device

    # Read-only slot/table introspection for unit tests and stateful adapters.
    @property
    def table(self) -> torch.Tensor | None:
        return self._table

    def slot_for(self, req_id: str) -> int | None:
        return self._slot_by_req_id.get(req_id)

    def has_dirty(self, req_id: str) -> bool:
        return req_id in self._dirty_rows

    def allocated_req_ids(self) -> AbstractSet[str]:
        return self._slot_by_req_id.keys()

    def commit_table(self, table: torch.Tensor) -> None:
        self._table = table

    def ensure_table(self) -> torch.Tensor:
        if self._table is None:
            sentinel_np = np.tile(
                np.asarray(self._sentinel_row, dtype=np.int32), (self._capacity, 1)
            )
            self._table = torch.from_numpy(sentinel_np).to(self._device)
            self._slot_ids_cache.clear()
        return self._table

    def mark_dirty(self, req_id: str, row: RowT) -> None:
        self._dirty_rows[req_id] = row

    def _release_slot(self, req_id: str) -> None:
        slot = self._slot_by_req_id.pop(req_id, None)
        if slot is not None:
            self._free_slots.append(slot)

    def release(self, req_ids: Iterable[str] | None) -> None:
        """Free slots and pending dirty rows for ``req_ids``."""
        if not req_ids:
            return
        for req_id in req_ids:
            self._dirty_rows.pop(req_id, None)
            self._release_slot(req_id)

    def release_unscheduled(self, active_req_ids: Container[str]) -> None:
        """Free slots and dirty rows for requests absent from ``active_req_ids``."""
        if not self._slot_by_req_id and not self._dirty_rows:
            return
        for req_id in list(self._slot_by_req_id):
            if req_id not in active_req_ids:
                self._dirty_rows.pop(req_id, None)
                self._release_slot(req_id)
        for req_id in list(self._dirty_rows):
            if req_id not in active_req_ids:
                self._dirty_rows.pop(req_id, None)

    def needs_cold_reseed(
        self,
        batch_req_ids: Iterable[str | None],
        tracked_req_ids: Container[str],
    ) -> bool:
        """Return True if ``sync_step`` will invoke ``compute_cold_row``."""
        return any(
            req_id is not None
            and req_id in tracked_req_ids
            and req_id not in self._slot_by_req_id
            and req_id not in self._dirty_rows
            for req_id in batch_req_ids
        )

    def _allocate_slot(self, req_id: str) -> int:
        existing = self._slot_by_req_id.get(req_id)
        if existing is not None:
            return existing
        assert self._free_slots, (
            f"RequestSlotTable pool exhausted "
            f"(max_num_reqs={self._max_num_reqs}, capacity={self._capacity})"
        )
        slot = self._free_slots.pop()
        self._slot_by_req_id[req_id] = slot
        return slot

    def sync_step(
        self,
        batch_req_ids: Sequence[str | None],
        tracked_req_ids: Container[str],
        compute_cold_row: Callable[[str], RowT],
    ) -> torch.Tensor | None:
        """Lease slots and seed dirty/cold rows for the tracked requests in
        ``batch_req_ids``, after releasing those no longer in it.

        Returns the device table, or None when no tracked request is in the
        batch.
        """
        batch_tracked = list(
            dict.fromkeys(
                req_id
                for req_id in batch_req_ids
                if req_id is not None and req_id in tracked_req_ids
            )
        )
        # Release first, also when no tracked request is left in the batch,
        # so a request that returns later is cold-seeded.
        self.release_unscheduled(set(batch_tracked))
        if not batch_tracked:
            return None

        table = self.ensure_table()
        to_seed: list[tuple[int, RowT]] = []
        for req_id in batch_tracked:
            had_slot = req_id in self._slot_by_req_id
            slot = self._allocate_slot(req_id)
            dirty = self._dirty_rows.pop(req_id, None)
            if dirty is None and not had_slot:
                dirty = compute_cold_row(req_id)
            if dirty is not None:
                to_seed.append((slot, dirty))

        if to_seed:
            # Each row is for a distinct leased slot, so one [max_num_reqs]
            # scatter seeds them all.
            assert len(to_seed) <= self._max_num_reqs
            slots_np = np.zeros(self._max_num_reqs, dtype=np.int32)
            vals_np = np.tile(
                np.asarray(self._sentinel_row, dtype=np.int32),
                (self._max_num_reqs, 1),
            )
            for idx, (slot, row_vals) in enumerate(to_seed):
                slots_np[idx] = slot
                vals_np[idx] = row_vals
            # Synchronous .to(device): sample_tokens() runs after the runner's
            # input-staging fence, so async H2D from temporary CPU buffers
            # would risk reading freed host memory in TorchTPU.
            table = seed_slot_rows(
                table,
                torch.from_numpy(slots_np).to(self._device),
                torch.from_numpy(vals_np).to(self._device),
            )
            self._table = table
        return table

    def build_chunk_slot_ids(
        self,
        batch_req_ids: Sequence[str | None],
        start_idx: int,
        end_idx: int,
        padded_num_reqs: int | None = None,
        discard_req_ids: AbstractSet[str] | None = None,
    ) -> torch.Tensor | None:
        """Build ``[padded_num_reqs]`` int32 slot IDs for ``[start_idx, end_idx)``."""
        width = self._max_num_reqs if padded_num_reqs is None else int(padded_num_reqs)
        upper = min(end_idx, len(batch_req_ids))
        assert upper - start_idx <= width, (
            f"Chunk [{start_idx}, {upper}) has {upper - start_idx} rows, more "
            f"than its {width} slot IDs"
        )
        slots_np = np.zeros(width, dtype=np.int32)
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
        key = (start_idx, width)
        slots_bytes = slots_np.tobytes()
        cached = self._slot_ids_cache.get(key)
        if cached is not None and cached[0] == slots_bytes:
            return cached[1]
        dev = torch.from_numpy(slots_np).to(self._device)
        self._slot_ids_cache[key] = (slots_bytes, dev)
        return dev

    def precompile(self, sync_fn: Callable[..., None]) -> torch.Tensor:
        """Warm ``seed_slot_rows`` at ``[max_num_reqs]`` width."""
        table = self.ensure_table()
        seed_slots = torch.zeros(self._max_num_reqs, dtype=torch.int32).to(self._device)
        seed_vals = np.tile(
            np.asarray(self._sentinel_row, dtype=np.int32), (self._max_num_reqs, 1)
        )
        table = seed_slot_rows(
            table, seed_slots, torch.from_numpy(seed_vals).to(self._device)
        )
        sync_fn(table)
        self._table = table
        return table
