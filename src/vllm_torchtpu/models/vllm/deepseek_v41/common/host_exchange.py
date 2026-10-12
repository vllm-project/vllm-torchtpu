# SPDX-License-Identifier: Apache-2.0
"""Redistribute CPU Engram lookup bytes before uploading local token rows."""

import torch
import torch.distributed as dist


def exchange_rows(
    raw: torch.Tensor,
    scales: torch.Tensor,
    group: dist.ProcessGroup,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exchange destination-major tokens for source-major hash heads.

    Each rank owns a disjoint head shard and looks up all replicas' tokens.
    Only this replica's tokens, with heads in shard order, leave the host.
    Exchange FP8/E8M0 bytes together rather than dequantized BF16 rows.
    """
    slot, remainder = divmod(raw.shape[0], world_size)
    if remainder or raw.shape[:2] != scales.shape[:2]:
        raise ValueError("Engram rows must contain equal DP token slots")
    heads, width = raw.shape[1:]
    packed = torch.cat((raw, scales), dim=-1).contiguous()
    received = torch.empty_like(packed)
    dist.all_to_all_single(received, packed, group=group)
    rows = (
        received.view(world_size, slot, heads, -1)
        .transpose(0, 1)
        .reshape(slot, world_size * heads, -1)
    )
    return rows[..., :width], rows[..., width:]
