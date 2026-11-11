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
"""XLA-friendly unique-index scatter helpers for TPU sampling adapters."""

from typing import Final

import torch

AMAX_SENTINEL: Final[int] = torch.iinfo(torch.int32).min


def broadcast_amax_scatter(
    base: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Scatter ``values`` into ``base`` at unique valid ``indices`` via ``amax``.

    Uses a masked broadcast + ``amax`` reduction rather than
    ``Tensor.scatter(0, ...)``, whose XLA lowering for dynamic colliding
    indices adds ~7 ms/step on TPU v7x. Active ``indices[valid_mask]`` must be
    pairwise distinct and within ``[0, base.shape[0])``.
    """
    if indices.shape[0] == 0:
        return base
    if base.device.type == "cpu" and not torch.compiler.is_compiling():
        active_idx = indices[valid_mask]
        assert active_idx.numel() == torch.unique(active_idx).numel(), (
            "Duplicate active indices passed to broadcast_amax_scatter"
        )
    target_idx = torch.arange(
        base.shape[0], dtype=indices.dtype, device=base.device
    ).unsqueeze(1)
    match = valid_mask.unsqueeze(0) & (target_idx == indices.unsqueeze(0))
    trailing_ones = (1,) * (base.ndim - 1)
    match_view = match.reshape(*match.shape, *trailing_ones)
    cand = torch.where(
        match_view,
        values.to(base.dtype).unsqueeze(0),
        AMAX_SENTINEL,
    ).amax(dim=1)
    any_match = match.any(dim=1).reshape(base.shape[0], *trailing_ones)
    return torch.where(any_match, cand, base)


def scatter_slot_rows(
    table: torch.Tensor, slot_ids: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    """Return ``table`` with ``out[slot_ids[i]] = rows[i]`` for ``slot_ids > 0``."""
    return broadcast_amax_scatter(table, slot_ids, rows, slot_ids > 0)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def seed_slot_rows(
    table: torch.Tensor,
    seed_slots: torch.Tensor,
    seed_values: torch.Tensor,
) -> torch.Tensor:
    """Scatter ``seed_values`` into ``table`` at active ``seed_slots > 0``."""
    return scatter_slot_rows(table, seed_slots, seed_values)
