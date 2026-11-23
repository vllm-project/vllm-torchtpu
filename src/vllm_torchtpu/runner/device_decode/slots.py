# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Per-request device decode state ("slots"), in the style of vLLM's Model
Runner V2, MRV2 (`RequestState` and `BlockTables` in `vllm/v1/worker/gpu/`).

Each live request owns a slot: one row of device tensors holding what a pure
decode step reads to build its inputs and to sample. The host keeps exact
mirrors and uploads a row only when it changes. Row `max_num_reqs` is the null
slot that padded batch rows map to; it holds the neutral values the stock path
pads with and keeps them.
"""

from __future__ import annotations

import bisect
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import torch

from vllm_torchtpu.layers.core.attention_metadata import (
    stage_h2d_upload,
    zeros_h2d_source,
)

NO_SLOT: Final[int] = -1
# Why a batch cannot run on the slots (`DecodeSlots.mismatch_reason`).
SLOT_NOT_READY: Final[str] = "slot_not_ready"
MIRROR_MISMATCH: Final[str] = "mirror_mismatch"

# Columns of a staged scalar row; the float fields travel as their bits.
(
    _COL_SLOT,
    _COL_NUM_COMPUTED,
    _COL_TOP_K,
    _COL_SEED_LO,
    _COL_SEED_HI,
    _COL_MAMBA_SLOT,
    _COL_TEMPERATURE,
    _COL_TOP_P,
) = range(8)
_NUM_SCALAR_COLS: Final[int] = 8


def derive_request_seed(
    model_seed: int, alloc_index: int, req_id: str
) -> tuple[int, int]:
    """The (seed_lo, seed_hi) int32 halves of a request's 64-bit RNG seed.

    MRV2 gives an unseeded request a random seed. This one hashes the model
    seed, the allocation count and the request id instead, so every TP rank
    derives the same seed and a reused request id still gets a fresh one.
    """
    digest = hashlib.blake2b(
        f"{model_seed}:{alloc_index}:{req_id}".encode(), digest_size=8
    ).digest()
    seed = int.from_bytes(digest, "little", signed=True)
    return _as_int32(seed & 0xFFFFFFFF), _as_int32((seed >> 32) & 0xFFFFFFFF)


def _as_int32(value: int) -> int:
    return value - (1 << 32) if value >= 1 << 31 else value


def _float32_bits(value: float) -> int:
    return int(np.float32(value).view(np.int32))


def num_reqs_padding(runner: Any, num_reqs: int) -> int:
    """The stock request bucket of `num_reqs` rows: the first
    `runner.num_reqs_paddings` entry that holds them."""
    paddings = runner.num_reqs_paddings
    return paddings[bisect.bisect_left(paddings, num_reqs)]


def scalar_write_key(num_rows: int) -> tuple:
    return ("slot_scalar_write", num_rows)


def block_write_key(kv_cache_group_id: int, num_rows: int) -> tuple:
    return ("slot_block_write", kv_cache_group_id, num_rows)


def write_back_key(num_rows_padded: int) -> tuple:
    return ("slot_write_back", num_rows_padded)


@dataclass(frozen=True)
class SlotConfig:
    max_num_reqs: int
    # Block-table width of each KV cache group, in the ids its rows hold.
    group_widths: tuple[int, ...]
    vocab_size: int
    model_seed: int

    @property
    def num_slots(self) -> int:
        return self.max_num_reqs + 1

    @property
    def null_slot(self) -> int:
        return self.max_num_reqs

    @staticmethod
    def from_runner(runner: Any) -> SlotConfig:
        num_groups = len(runner.kv_cache_config.kv_cache_groups)
        return SlotConfig(
            max_num_reqs=runner.max_num_reqs,
            group_widths=tuple(
                runner.input_batch.block_table[gid].max_num_blocks_per_req
                for gid in range(num_groups)
            ),
            vocab_size=runner.vocab_size,
            model_seed=int(runner.model_config.seed or 0),
        )


@dataclass(frozen=True)
class StockChunkPlan:
    """Per padded row of a stock-sampled chunk: the slot, the RNG position
    (the last scheduled token), whether the step produces a token, and the
    computed-token count after the step. Padded rows hold the null slot."""

    num_rows_padded: int
    slots: np.ndarray
    positions: np.ndarray
    produced: np.ndarray
    new_num_computed: np.ndarray

    def to_device(self, runner: Any) -> torch.Tensor:
        """int32 [P, 4]: (slot, position, produced, new_num_computed)."""
        host = zeros_h2d_source(runner, self.num_rows_padded * 4)
        rows = host.numpy().reshape(self.num_rows_padded, 4)
        rows[:, 0] = self.slots
        rows[:, 1] = self.positions
        rows[:, 2] = self.produced
        rows[:, 3] = self.new_num_computed
        return stage_h2d_upload(runner, host).view(self.num_rows_padded, 4)


def _write_scalar_rows_impl(
    staged: torch.Tensor,
    num_computed: torch.Tensor,
    top_k: torch.Tensor,
    seed_lo: torch.Tensor,
    seed_hi: torch.Tensor,
    mamba_slot: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
) -> None:
    rows = staged.view(-1, _NUM_SCALAR_COLS)
    float_rows = rows.view(torch.float32)
    index = (rows[:, _COL_SLOT].long(),)
    num_computed.index_put_(index, rows[:, _COL_NUM_COMPUTED])
    top_k.index_put_(index, rows[:, _COL_TOP_K])
    seed_lo.index_put_(index, rows[:, _COL_SEED_LO])
    seed_hi.index_put_(index, rows[:, _COL_SEED_HI])
    mamba_slot.index_put_(index, rows[:, _COL_MAMBA_SLOT])
    temperature.index_put_(index, float_rows[:, _COL_TEMPERATURE])
    top_p.index_put_(index, float_rows[:, _COL_TOP_P])


def _write_block_rows_impl(staged: torch.Tensor, block_table: torch.Tensor) -> None:
    rows = staged.view(-1, 1 + block_table.shape[1])
    block_table.index_put_((rows[:, 0].long(),), rows[:, 1:])


def _write_back_impl(
    plan: torch.Tensor,
    sampled: torch.Tensor,
    num_computed: torch.Tensor,
    last_token: torch.Tensor,
    null_slot: int,
) -> None:
    index = (plan[:, 0].long(),)
    live = plan[:, 0] != null_slot
    num_computed.index_put_(index, torch.where(live, plan[:, 3], num_computed[index]))
    tokens = sampled.reshape(plan.shape[0]).to(torch.int32)
    produced = live & (plan[:, 2] != 0)
    last_token.index_put_(index, torch.where(produced, tokens, last_token[index]))


# Padded rows of every write target the null slot with its own values, so the
# duplicate indices of a scatter all write the same value.
_tpu_compile = torch.compile(backend="tpu", fullgraph=True, dynamic=False)
_write_scalar_rows_compiled = _tpu_compile(_write_scalar_rows_impl)
_write_block_rows_compiled = _tpu_compile(_write_block_rows_impl)
_write_back_compiled = _tpu_compile(_write_back_impl)


class DecodeSlots:
    """Device slot tensors and the host mirrors that decide every write.

    Device state changes only through the programs here and the decode head:
    staged scalar and block-row writes (`apply_staged_writes`), the stock
    write-back (`write_back_stock_chunk`) and the head's post-update of
    `num_computed` / `last_token`. `last_token` is never written from the host.
    """

    def __init__(self, config: SlotConfig, device: torch.device) -> None:
        self.config = config
        num_slots = config.num_slots

        def column(dtype: torch.dtype, fill: float = 0) -> torch.Tensor:
            return torch.full((num_slots,), fill, dtype=dtype).to(device)

        self.num_computed = column(torch.int32)
        self.last_token = column(torch.int32)
        self.temperature = column(torch.float32, 1.0)
        self.top_k = column(torch.int32)
        self.top_p = column(torch.float32, 1.0)
        self.seed_lo = column(torch.int32)
        self.seed_hi = column(torch.int32)
        self.mamba_slot = column(torch.int32)
        self.block_tables = tuple(
            torch.zeros((num_slots, width), dtype=torch.int32).to(device)
            for width in config.group_widths
        )

        # Host mirrors, by slot. A slot's scalar row is kept as last written.
        self._scalars = np.zeros((num_slots, _NUM_SCALAR_COLS), dtype=np.int32)
        self._scalars[:, _COL_SLOT] = config.null_slot
        self._scalars[:, _COL_TEMPERATURE] = _float32_bits(1.0)
        self._scalars[:, _COL_TOP_P] = _float32_bits(1.0)
        self._ready = np.zeros(num_slots, dtype=bool)
        self._num_blocks = np.zeros(
            (len(config.group_widths), num_slots), dtype=np.int32
        )
        self._slot_by_req: dict[str, int] = {}
        # One entry per live slot: the null slot never holds a request.
        self._req_by_slot: list[str | None] = [None] * config.max_num_reqs
        # LIFO free list, as MRV2's RequestState; the lowest slot goes first.
        self._free = list(range(config.max_num_reqs - 1, -1, -1))
        self._num_allocations = 0
        self._dirty_scalars: set[int] = set()
        self._staged_blocks: list[dict[int, np.ndarray]] = [
            {} for _ in config.group_widths
        ]
        # Slots whose compact-mamba slot the stock prefill has yet to assign.
        self._mamba_pending: set[int] = set()
        # Bumped on every allocation and free; keys the row -> slot cache.
        self._epoch = 0
        self._rows_key: tuple[int, tuple[str, ...]] | None = None
        self._rows_slots = np.zeros(0, dtype=np.int32)
        # The request `mismatch_reason` last refused the batch for.
        self.last_mismatch_req_id: str | None = None

    def slot_of(self, req_id: str) -> int:
        return self._slot_by_req.get(req_id, NO_SLOT)

    def decode_ready(self, slot: int) -> bool:
        """Whether the last step of the slot's request produced a token, so
        that `last_token` holds the token it feeds next."""
        return bool(self._ready[slot])

    def sync_from_runner(self, runner: Any, scheduler_output: Any) -> None:
        """Follows the runner's request set after its `_update_states`.

        Frees the slots of finished, preempted and resumed requests (and of
        any request the runner no longer holds), allocates slots for new and
        resumed requests in sorted order, and stages the rows that changed:
        new slots' scalars and block rows, block rows whose block count grew,
        and compact-mamba slots the stock prefill assigned. Host only.
        """
        resumed = scheduler_output.scheduled_cached_reqs.resumed_req_ids
        invalidated = (
            scheduler_output.finished_req_ids
            | (scheduler_output.preempted_req_ids or set())
            | resumed
        )
        stale = self._slot_by_req.keys() - runner.requests.keys()
        for req_id in sorted(invalidated | stale):
            self._free_slot(req_id)
        new_req_ids = {req.req_id for req in scheduler_output.scheduled_new_reqs}
        for req_id in sorted(new_req_ids | resumed):
            # A streaming input update re-sends an existing request.
            self._free_slot(req_id)
            self._allocate(runner, req_id)
        self._stage_grown_block_rows(runner.input_batch)
        if self._mamba_pending:
            self._stage_assigned_mamba_slots(runner._mamba_slot_by_req_id)

    def apply_staged_writes(self, runner: Any) -> None:
        """Writes the staged rows: per kind at most one H2D transfer and one
        program, padded to the request bucket ladder; a no-op when clean."""
        null_slot = self.config.null_slot
        if self._dirty_scalars:
            slots = sorted(self._dirty_scalars)
            self._dirty_scalars.clear()
            num_rows = num_reqs_padding(runner, len(slots))
            host = zeros_h2d_source(runner, num_rows * _NUM_SCALAR_COLS)
            rows = host.numpy().reshape(num_rows, _NUM_SCALAR_COLS)
            rows[:] = self._scalars[null_slot]
            rows[: len(slots)] = self._scalars[slots]
            self._write_scalar_rows(stage_h2d_upload(runner, host))
        for gid, staged in enumerate(self._staged_blocks):
            if not staged:
                continue
            items = sorted(staged.items())
            staged.clear()
            num_rows = num_reqs_padding(runner, len(items))
            row_width = 1 + self.config.group_widths[gid]
            host = zeros_h2d_source(runner, num_rows * row_width)
            rows = host.numpy().reshape(num_rows, row_width)
            rows[:, 0] = null_slot
            for i, (slot, block_ids) in enumerate(items):
                rows[i, 0] = slot
                rows[i, 1 : 1 + block_ids.shape[0]] = block_ids
            self._write_block_rows(stage_h2d_upload(runner, host), gid)

    def idx_mapping_host(self, req_ids: Sequence[str]) -> np.ndarray:
        """int32 [max_num_reqs]: each batch row's slot, the null slot after
        the last row. Raises KeyError for a request without a slot."""
        slots = self._slots_of(req_ids)
        missing = np.flatnonzero(slots == NO_SLOT)
        if missing.size:
            raise KeyError(
                f"Request {req_ids[int(missing[0])]!r} has no device decode slot."
            )
        idx_mapping = np.full(
            self.config.max_num_reqs, self.config.null_slot, dtype=np.int32
        )
        idx_mapping[: slots.shape[0]] = slots
        return idx_mapping

    def mismatch_reason(self, runner: Any, num_reqs: int) -> str | None:
        """Why batch rows [0, num_reqs) cannot run on the slots, or None.

        SLOT_NOT_READY: a row has no slot, or no token was produced for it
        yet (prefill chunks, resumed requests). MIRROR_MISMATCH: a mirror
        disagrees with the runner (computed tokens, block counts, an
        unassigned compact-mamba slot). Sets `last_mismatch_req_id`.
        """
        input_batch = runner.input_batch
        req_ids = input_batch.req_ids[:num_reqs]
        slots = self._slots_of(req_ids)
        self.last_mismatch_req_id = None
        missing = slots == NO_SLOT
        if missing.any():
            return self._refuse(req_ids, missing, SLOT_NOT_READY)
        not_ready = ~self._ready[slots]
        if not_ready.any():
            return self._refuse(req_ids, not_ready, SLOT_NOT_READY)
        drift = (
            self._scalars[slots, _COL_NUM_COMPUTED]
            != input_batch.num_computed_tokens_cpu[:num_reqs]
        )
        if drift.any():
            return self._refuse(req_ids, drift, MIRROR_MISMATCH)
        for gid in range(len(self.config.group_widths)):
            stale = (
                self._num_blocks[gid, slots]
                != input_batch.block_table[gid].num_blocks_per_row[:num_reqs]
            )
            if stale.any():
                return self._refuse(req_ids, stale, MIRROR_MISMATCH)
        if self._mamba_pending:
            pending = np.isin(slots, np.fromiter(self._mamba_pending, dtype=np.int32))
            if pending.any():
                return self._refuse(req_ids, pending, MIRROR_MISMATCH)
        return None

    def note_device_step(self, slots: np.ndarray) -> None:
        """Mirrors the head's post-update: each slot computed one token."""
        live = slots[slots != self.config.null_slot]
        self._scalars[live, _COL_NUM_COMPUTED] += 1

    def plan_stock_chunk(
        self,
        runner: Any,
        scheduler_output: Any,
        start: int,
        end: int,
        num_rows_padded: int,
    ) -> StockChunkPlan:
        """The write-back plan of the stock chunk of batch rows [start, end),
        with the stock rule for a produced token (`sample_tokens` discards
        the token of a request whose sequence is still short of its tokens).
        """
        slots = np.full(num_rows_padded, self.config.null_slot, dtype=np.int32)
        positions = np.zeros(num_rows_padded, dtype=np.int32)
        produced = np.zeros(num_rows_padded, dtype=bool)
        new_num_computed = np.zeros(num_rows_padded, dtype=np.int32)
        req_ids = runner.input_batch.req_ids
        for i, row in enumerate(range(start, end)):
            req_id = req_ids[row]
            slot = self._slot_by_req.get(req_id)
            if slot is None:
                continue
            req_state = runner.requests[req_id]
            seq_len = (
                req_state.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[req_id]
            )
            slots[i] = slot
            positions[i] = seq_len - 1
            produced[i] = seq_len >= req_state.num_tokens
            new_num_computed[i] = seq_len
        return StockChunkPlan(
            num_rows_padded, slots, positions, produced, new_num_computed
        )

    def write_back_stock_chunk(
        self, runner: Any, plan: StockChunkPlan, sampled: torch.Tensor
    ) -> None:
        """Writes a stock chunk's outcome into its slots: `last_token` where
        a token was produced, `num_computed` for every live row. `sampled`
        holds the chunk's [P, 1] (or [P]) tokens. One H2D, one program.

        A live row that produced no token (a prefill chunk, also one that
        recomputes a decode-ready request's tokens after a failed KV load)
        leaves its slot not decode-ready: its `last_token` is stale."""
        live = plan.slots != self.config.null_slot
        self._scalars[plan.slots[live], _COL_NUM_COMPUTED] = plan.new_num_computed[live]
        self._ready[plan.slots[live]] = plan.produced[live]
        plan_dev = plan.to_device(runner)
        fn = _write_back_compiled if plan_dev.device.type == "tpu" else _write_back_impl
        with torch.no_grad():
            fn(
                plan_dev,
                sampled,
                self.num_computed,
                self.last_token,
                self.config.null_slot,
            )

    def precompile(self, runner: Any) -> list[tuple]:
        """Runs every write program once per request bucket on all-null rows,
        which rewrite the null slot's own values. Returns the program keys."""
        device = runner.device
        null_slot = self.config.null_slot
        keys: list[tuple] = []
        for num_rows in runner.num_reqs_paddings:
            scalar_rows = np.tile(self._scalars[null_slot], num_rows)
            self._write_scalar_rows(torch.from_numpy(scalar_rows).to(device))
            keys.append(scalar_write_key(num_rows))
            for gid, width in enumerate(self.config.group_widths):
                block_rows = np.zeros((num_rows, 1 + width), dtype=np.int32)
                block_rows[:, 0] = null_slot
                self._write_block_rows(
                    torch.from_numpy(block_rows.reshape(-1)).to(device), gid
                )
                keys.append(block_write_key(gid, num_rows))
            plan = np.zeros((num_rows, 4), dtype=np.int32)
            plan[:, 0] = null_slot
            fn = _write_back_compiled if device.type == "tpu" else _write_back_impl
            with torch.no_grad():
                fn(
                    # As `StockChunkPlan.to_device`: a view of a flat upload.
                    torch.from_numpy(plan.reshape(-1)).to(device).view(num_rows, 4),
                    torch.zeros((num_rows, 1), dtype=torch.int64).to(device),
                    self.num_computed,
                    self.last_token,
                    null_slot,
                )
            keys.append(write_back_key(num_rows))
        return keys

    def _write_scalar_rows(self, staged: torch.Tensor) -> None:
        fn = (
            _write_scalar_rows_compiled
            if staged.device.type == "tpu"
            else _write_scalar_rows_impl
        )
        with torch.no_grad():
            fn(
                staged,
                self.num_computed,
                self.top_k,
                self.seed_lo,
                self.seed_hi,
                self.mamba_slot,
                self.temperature,
                self.top_p,
            )

    def _write_block_rows(self, staged: torch.Tensor, gid: int) -> None:
        fn = (
            _write_block_rows_compiled
            if staged.device.type == "tpu"
            else _write_block_rows_impl
        )
        with torch.no_grad():
            fn(staged, self.block_tables[gid])

    def _allocate(self, runner: Any, req_id: str) -> None:
        if not self._free:
            raise RuntimeError(
                f"Device decode has no free slot for request {req_id!r}: all "
                f"{self.config.max_num_reqs} slots are held by live requests."
            )
        input_batch = runner.input_batch
        row = input_batch.req_id_to_index[req_id]
        slot = self._free.pop()
        self._slot_by_req[req_id] = slot
        self._req_by_slot[slot] = req_id
        self._epoch += 1
        seed_lo, seed_hi = derive_request_seed(
            self.config.model_seed, self._num_allocations, req_id
        )
        self._num_allocations += 1
        top_k = int(input_batch.top_k_cpu[row])
        scalars = self._scalars[slot]
        scalars[_COL_SLOT] = slot
        scalars[_COL_NUM_COMPUTED] = input_batch.num_computed_tokens_cpu[row]
        # The stock sampler's mapping: top_k >= vocab_size means no top-k.
        scalars[_COL_TOP_K] = 0 if top_k >= self.config.vocab_size else top_k
        scalars[_COL_SEED_LO] = seed_lo
        scalars[_COL_SEED_HI] = seed_hi
        scalars[_COL_MAMBA_SLOT] = 0
        scalars[_COL_TEMPERATURE] = _float32_bits(input_batch.temperature_cpu[row])
        scalars[_COL_TOP_P] = _float32_bits(input_batch.top_p_cpu[row])
        self._dirty_scalars.add(slot)
        self._ready[slot] = False
        if runner._has_mamba_state:
            self._mamba_pending.add(slot)
        for gid in range(len(self.config.group_widths)):
            self._stage_block_row(input_batch.block_table[gid], gid, slot, row)

    def _free_slot(self, req_id: str) -> None:
        slot = self._slot_by_req.pop(req_id, None)
        if slot is None:
            return
        self._req_by_slot[slot] = None
        self._ready[slot] = False
        self._dirty_scalars.discard(slot)
        for staged in self._staged_blocks:
            staged.pop(slot, None)
        self._mamba_pending.discard(slot)
        self._free.append(slot)
        self._epoch += 1

    def _stage_block_row(self, table: Any, gid: int, slot: int, row: int) -> None:
        num_blocks = int(table.num_blocks_per_row[row])
        self._staged_blocks[gid][slot] = (
            table.get_cpu_tensor()[row, :num_blocks].numpy().copy()
        )
        self._num_blocks[gid, slot] = num_blocks

    def _stage_grown_block_rows(self, input_batch: Any) -> None:
        """Stages every batch row whose block count left its slot's mirror.

        The count tracks the contents because, with vLLM 0.33, a request's
        blocks only grow while it keeps its slot. `_update_states` appends a
        running request's new blocks (`append_row`), and condensing or
        reordering the batch moves whole rows (`move_row`, `swap_row`). A
        request that rejoins the batch gets its old blocks back plus any new
        ones (`add_row`). Only a resumed request's blocks are replaced, and
        `sync_from_runner` frees and allocates its slot again, which stages
        the whole row. Kernel block tables only append, and each KV cache
        group has its own table and mirror count. A change that rewrote a
        row in place without changing its count would get past this check.
        """
        num_reqs = input_batch.num_reqs
        if not num_reqs:
            return
        slots = self._slots_of(input_batch.req_ids[:num_reqs])
        live = slots != NO_SLOT
        mirror_slots = np.where(live, slots, self.config.null_slot)
        for gid in range(len(self.config.group_widths)):
            table = input_batch.block_table[gid]
            changed = live & (
                table.num_blocks_per_row[:num_reqs]
                != self._num_blocks[gid, mirror_slots]
            )
            for row in np.flatnonzero(changed).tolist():
                self._stage_block_row(table, gid, int(slots[row]), row)

    def _stage_assigned_mamba_slots(self, mamba_slot_by_req_id: dict[str, int]) -> None:
        assigned = []
        for slot in sorted(self._mamba_pending):
            req_id = self._req_by_slot[slot]
            assert req_id is not None
            mamba_slot = mamba_slot_by_req_id.get(req_id)
            if mamba_slot is not None:
                self._scalars[slot, _COL_MAMBA_SLOT] = mamba_slot
                self._dirty_scalars.add(slot)
                assigned.append(slot)
        self._mamba_pending.difference_update(assigned)

    def _slots_of(self, req_ids: Sequence[str]) -> np.ndarray:
        """int32 slot per request (NO_SLOT if none), cached per batch order."""
        key = (self._epoch, tuple(req_ids))
        if key != self._rows_key:
            self._rows_slots = np.fromiter(
                (self._slot_by_req.get(req_id, NO_SLOT) for req_id in key[1]),
                dtype=np.int32,
                count=len(key[1]),
            )
            self._rows_key = key
        return self._rows_slots

    def _refuse(self, req_ids: Sequence[str], rows: np.ndarray, reason: str) -> str:
        self.last_mismatch_req_id = req_ids[int(np.argmax(rows))]
        return reason
