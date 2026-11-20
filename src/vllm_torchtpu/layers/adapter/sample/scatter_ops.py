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
"""Row writes into slot tables for TPU sampling kernels.

Row 0 of a slot table is a null row. ``RequestSlotTable`` writes new rows with
``seed_slot_rows``, and sampling kernels that keep per-request rows in such a
table write them back with ``scatter_slot_rows``.
"""

import torch

from vllm_torchtpu.compilation.fast_dispatch import tpu_compile


def scatter_slot_rows(
    table: torch.Tensor, slot_ids: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    """Return ``table`` with ``out[slot_ids[i]] = rows[i]`` where ``slot_ids[i] > 0``.

    The other entries write row 0 back to itself, the null-row pattern of
    ``TPUModelRunner._update_mamba_slot_read_offsets``, so one ``index_put``
    covers the whole fixed-size ``slot_ids``. ``slot_ids`` is 1-D and ``rows``
    is ``[len(slot_ids), *table.shape[1:]]`` with ``table``'s dtype. Active
    slots must be distinct and below ``table.shape[0]``.
    """
    torch._assert(slot_ids.dim() == 1, "slot_ids must be 1-D")
    torch._assert(
        rows.shape == (slot_ids.shape[0], *table.shape[1:]),
        "rows must have shape [len(slot_ids), *table.shape[1:]]",
    )
    torch._assert(rows.dtype == table.dtype, "rows must have the table's dtype")
    active = slot_ids > 0
    if table.device.type == "cpu" and not torch.compiler.is_compiling():
        active_slots = slot_ids[active]
        if active_slots.numel() and int(active_slots.max()) >= table.shape[0]:
            raise ValueError(
                f"slot IDs {slot_ids.tolist()} reach past the {table.shape[0]} rows"
            )
        if active_slots.numel() != torch.unique(active_slots).numel():
            raise ValueError(f"slot IDs {slot_ids.tolist()} repeat an active slot")
    targets = torch.where(active, slot_ids, 0).long()
    keep = active.reshape(-1, *(1,) * (rows.dim() - 1))
    return table.index_put((targets,), torch.where(keep, rows, table[0]))


@tpu_compile
def seed_slot_rows(
    table: torch.Tensor,
    seed_slots: torch.Tensor,
    seed_values: torch.Tensor,
) -> torch.Tensor:
    """Write ``seed_values`` into ``table`` at the active ``seed_slots > 0``."""
    return scatter_slot_rows(table, seed_slots, seed_values)
