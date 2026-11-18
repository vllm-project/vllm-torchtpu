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
"""Request-ID-keyed slots and a device table of small int32 rows.

``RequestSlotAllocator`` maps request IDs to slots ``1..max_num_reqs`` on the
host. Slot ``0`` is a null slot: chunk slot-ID arrays point padding, untracked
and discarded rows at it.

``RequestSlotTable`` keeps one fixed-width int32 row per leased slot in a
``[max_num_reqs + 1, num_cols]`` tensor on the device. It is meant for a few
int32 values of per-request sampling state that must stay on the device across
steps while requests change batch rows: ``TPUModelRunner`` reorders the batch
in ``_reorder_batch_for_rpa``, drops a request from the batch in a step that
does not schedule it, and under ``--async-scheduling`` still has output
placeholders in flight when ``sample_tokens()`` begins. Keying slots on request
IDs keeps each row with its request through all three.
"""

import operator
from collections.abc import Callable, Container, Iterable, Sequence
from collections.abc import Set as AbstractSet

import numpy as np
import torch

from vllm_torchtpu.layers.adapter.sample.scatter_ops import seed_slot_rows
from vllm_torchtpu.utils import synchronize_tensors

_INT32 = np.iinfo(np.int32)


class RequestSlotAllocator:
    """Host-side map from request IDs to slots ``1..max_num_reqs``.

    Slot ``0`` is never leased. The owner decides when a request leases and
    releases its slot; the allocator keeps no device state.
    """

    def __init__(self, max_num_reqs: int) -> None:
        if max_num_reqs <= 0:
            raise ValueError(f"max_num_reqs must be positive, got {max_num_reqs}")
        self._max_num_reqs = max_num_reqs
        self._slot_by_req_id: dict[str, int] = {}
        # Popped from the end, so the first lease gets slot 1.
        self._free_slots = list(range(max_num_reqs, 0, -1))

    @property
    def max_num_reqs(self) -> int:
        return self._max_num_reqs

    def slot_for(self, req_id: str) -> int | None:
        """Return the slot leased to ``req_id``, or None."""
        return self._slot_by_req_id.get(req_id)

    def allocated_req_ids(self) -> AbstractSet[str]:
        """Return a live view of the request IDs that hold a slot."""
        return self._slot_by_req_id.keys()

    def lease(self, req_id: str) -> int:
        """Return the slot of ``req_id``, leasing a free one if it holds none.

        Raises:
            RuntimeError: all ``max_num_reqs`` slots are leased.
        """
        slot = self._slot_by_req_id.get(req_id)
        if slot is None:
            if not self._free_slots:
                raise RuntimeError(
                    f"RequestSlotAllocator is full: all {self._max_num_reqs} "
                    "slots are leased"
                )
            slot = self._free_slots.pop()
            self._slot_by_req_id[req_id] = slot
        return slot

    def release(self, req_ids: Iterable[str]) -> None:
        """Free the slots of the requests in ``req_ids`` that hold one."""
        for req_id in req_ids:
            slot = self._slot_by_req_id.pop(req_id, None)
            if slot is not None:
                self._free_slots.append(slot)

    def build_chunk_slot_ids(
        self,
        batch_req_ids: Sequence[str | None],
        start_idx: int,
        end_idx: int,
        discard_req_ids: AbstractSet[str] = frozenset(),
    ) -> np.ndarray | None:
        """Return the slot IDs of batch rows ``[start_idx, end_idx)``.

        The result is a ``[max_num_reqs]`` int32 array: entry ``i`` is the
        slot of ``batch_req_ids[start_idx + i]``, and 0 for rows past the end
        of the batch, empty rows, requests without a slot, requests in
        ``discard_req_ids`` and the padding. Returns None when no row has a
        slot. The caller uploads the array.

        Raises:
            ValueError: the range is invalid, the chunk has more than
                ``max_num_reqs`` rows, or two rows hold the same slot.
        """
        if start_idx < 0 or end_idx < start_idx:
            raise ValueError(f"invalid chunk range [{start_idx}, {end_idx})")
        upper = min(end_idx, len(batch_req_ids))
        if upper - start_idx > self._max_num_reqs:
            raise ValueError(
                f"chunk [{start_idx}, {upper}) has {upper - start_idx} rows, "
                f"more than max_num_reqs={self._max_num_reqs}"
            )
        slots = np.zeros(self._max_num_reqs, dtype=np.int32)
        for row, req_id in enumerate(batch_req_ids[start_idx:upper]):
            if req_id is None or req_id in discard_req_ids:
                continue
            slot = self._slot_by_req_id.get(req_id)
            if slot is not None:
                slots[row] = slot
        active = slots[slots > 0]
        if active.size == 0:
            return None
        if np.unique(active).size != active.size:
            raise ValueError(
                f"chunk [{start_idx}, {upper}) lists a request twice: "
                f"slots {slots[: upper - start_idx].tolist()}"
            )
        return slots


class RequestSlotTable[RowT: Sequence[int]]:
    """``[max_num_reqs + 1, num_cols]`` int32 device table with one row per
    slot of its ``allocator``.

    Row 0 holds ``sentinel_row``, and the slot scatters in ``scatter_ops``
    leave it unchanged, so kernels can read it for padding and untracked rows.
    One step goes in this order:

    1. The owner calls ``release`` for requests that finished or were
       preempted, and ``mark_dirty`` for rows it has rebuilt (for example a
       resumed request's state).
    2. ``sync_step`` releases the slots of tracked requests that are not in
       the batch, leases a slot for each tracked request that is, and writes
       their new rows to the table in one ``seed_slot_rows`` call.
    3. Kernels read and update the rows through slot IDs from
       ``allocator.build_chunk_slot_ids``, and the owner stores their output
       with ``commit_table``.

    In ``sync_step``, a request's row comes from ``mark_dirty`` when one is
    staged. Otherwise, a request without a slot (new, or back after leaving
    the batch) gets ``compute_cold_row(req_id)``, and a request that keeps its
    slot keeps its row.

    If ``sync_step`` raises, the leases and staged rows no longer match the
    table. The runner stops on such errors, so the table is not reused.
    """

    def __init__(
        self,
        max_num_reqs: int,
        num_cols: int,
        device: torch.device,
        sentinel_row: Sequence[int] | None = None,
    ) -> None:
        self._allocator = RequestSlotAllocator(max_num_reqs)
        if num_cols <= 0:
            raise ValueError(f"num_cols must be positive, got {num_cols}")
        self._num_cols = num_cols
        self._device = device
        self._sentinel_row: tuple[int, ...] = self._check_row(
            (0,) * num_cols if sentinel_row is None else sentinel_row
        )
        self._dirty_rows: dict[str, RowT] = {}
        self._table: torch.Tensor | None = None

    @property
    def allocator(self) -> RequestSlotAllocator:
        return self._allocator

    @property
    def max_num_reqs(self) -> int:
        return self._allocator.max_num_reqs

    @property
    def device(self) -> torch.device:
        return self._device

    def _check_row(self, row: Sequence[int]) -> tuple[int, ...]:
        if len(row) != self._num_cols:
            raise ValueError(f"row {row!r} has {len(row)} values, not {self._num_cols}")
        values = tuple(operator.index(v) for v in row)  # TypeError if not integers
        if not all(_INT32.min <= v <= _INT32.max for v in values):
            raise ValueError(f"row {values} does not fit in int32")
        return values

    def _ensure_table(self) -> torch.Tensor:
        if self._table is None:
            rows = np.tile(
                np.asarray(self._sentinel_row, dtype=np.int32),
                (self.max_num_reqs + 1, 1),
            )
            self._table = torch.from_numpy(rows).to(self._device)
        return self._table

    def commit_table(self, table: torch.Tensor) -> None:
        """Store ``table``, for example a kernel's output, as the device state.

        The leases do not change.

        Raises:
            ValueError: ``table`` does not have this table's shape, int32 dtype
                or device type.
        """
        shape = (self.max_num_reqs + 1, self._num_cols)
        if (
            tuple(table.shape) != shape
            or table.dtype != torch.int32
            or table.device.type != self._device.type
        ):
            raise ValueError(
                f"expected a {shape} int32 table on {self._device.type}, got "
                f"{tuple(table.shape)} {table.dtype} on {table.device.type}"
            )
        self._table = table

    def mark_dirty(self, req_id: str, row: RowT) -> None:
        """Stage ``row`` as the state of ``req_id``.

        The next ``sync_step`` that finds ``req_id`` in the batch writes it,
        whether or not the request holds a slot. ``release`` and
        ``release_unscheduled`` drop it.

        Raises:
            ValueError: ``row`` does not have ``num_cols`` int32 values.
            TypeError: a value is not an integer.
        """
        self._check_row(row)
        self._dirty_rows[req_id] = row

    def release(self, req_ids: Iterable[str]) -> None:
        """Free the slots and staged rows of ``req_ids``.

        Call it before ``sync_step`` for requests that finished or were
        preempted in this step. ``sync_step`` only releases requests that are
        not in the batch, so a request preempted and resumed in the same step
        would otherwise keep its old row.
        """
        released = list(req_ids)
        for req_id in released:
            self._dirty_rows.pop(req_id, None)
        self._allocator.release(released)

    def release_unscheduled(self, active_req_ids: Container[str]) -> None:
        """Free the slots and staged rows of requests not in ``active_req_ids``."""
        self.release(
            [
                req_id
                for req_id in (*self._allocator.allocated_req_ids(), *self._dirty_rows)
                if req_id not in active_req_ids
            ]
        )

    def needs_cold_reseed(
        self,
        batch_req_ids: Iterable[str | None],
        tracked_req_ids: Container[str],
    ) -> bool:
        """Return True if ``sync_step`` will call ``compute_cold_row``: a tracked
        request in the batch holds no slot and has no staged row."""
        return any(
            req_id is not None
            and req_id in tracked_req_ids
            and self._allocator.slot_for(req_id) is None
            and req_id not in self._dirty_rows
            for req_id in batch_req_ids
        )

    def sync_step(
        self,
        batch_req_ids: Sequence[str | None],
        tracked_req_ids: Container[str],
        compute_cold_row: Callable[[str], RowT],
    ) -> torch.Tensor | None:
        """Lease slots and write rows for the tracked requests in
        ``batch_req_ids``, after releasing the tracked requests not in it.

        A request listed twice in ``batch_req_ids`` is handled once. Returns
        the device table, or None when no tracked request is in the batch.

        Raises:
            RuntimeError: more tracked requests than ``max_num_reqs``.
            ValueError, TypeError: ``compute_cold_row`` returned an invalid row.
        """
        batch_tracked = [
            req_id
            for req_id in batch_req_ids
            if req_id is not None and req_id in tracked_req_ids
        ]
        # Release first, also when no tracked request is left in the batch,
        # so a request that returns later gets a cold row.
        self.release_unscheduled(set(batch_tracked))
        if not batch_tracked:
            return None

        table = self._ensure_table()
        to_seed: list[tuple[int, Sequence[int]]] = []
        for req_id in batch_tracked:
            had_slot = self._allocator.slot_for(req_id) is not None
            slot = self._allocator.lease(req_id)
            row = self._dirty_rows.pop(req_id, None)
            if row is None and not had_slot:
                row = self._check_row(compute_cold_row(req_id))
            if row is not None:
                to_seed.append((slot, row))

        if to_seed:
            # Each row has its own slot, so one [max_num_reqs] scatter writes
            # all of them; unused entries point at slot 0, which it leaves
            # unchanged.
            slots_np = np.zeros(self.max_num_reqs, dtype=np.int32)
            rows_np = np.zeros((self.max_num_reqs, self._num_cols), dtype=np.int32)
            for idx, (slot, row) in enumerate(to_seed):
                slots_np[idx] = slot
                rows_np[idx] = row
            # Synchronous .to(device): sample_tokens() runs after the runner's
            # input-staging fence, so an async H2D copy from these temporary
            # arrays could read freed host memory in TorchTPU.
            table = seed_slot_rows(
                table,
                torch.from_numpy(slots_np).to(self._device),
                torch.from_numpy(rows_np).to(self._device),
            )
            self._table = table
        return table

    def precompile(self) -> torch.Tensor:
        """Compile ``seed_slot_rows`` at the shapes ``sync_step`` uses, and
        return the table."""
        table = self._ensure_table()
        slots = torch.zeros(self.max_num_reqs, dtype=torch.int32).to(self._device)
        rows = torch.zeros((self.max_num_reqs, self._num_cols), dtype=torch.int32)
        table = seed_slot_rows(table, slots, rows.to(self._device))
        synchronize_tensors(table)
        self._table = table
        return table
