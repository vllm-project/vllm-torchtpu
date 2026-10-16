# SPDX-License-Identifier: Apache-2.0
"""Redistribute CPU Engram lookup bytes before uploading local token rows."""

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class PendingRowExchange:
    # Keep the send buffer alive until Gloo has finished reading it.
    packed: torch.Tensor
    received: torch.Tensor
    work: dist.Work
    world_size: int
    width: int

    def wait(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.work.wait()
        tokens, heads, packed_width = self.received.shape
        slot = tokens // self.world_size
        rows = (
            self.received.view(self.world_size, slot, heads, packed_width)
            .transpose(0, 1)
            .reshape(slot, self.world_size * heads, packed_width)
        )
        return rows[..., : self.width], rows[..., self.width :]


def start_exchange_rows(
    raw: torch.Tensor,
    scales: torch.Tensor,
    group: dist.ProcessGroup,
    world_size: int,
) -> PendingRowExchange:
    """Exchange destination-major tokens for source-major hash heads.

    Each rank owns a disjoint head shard and looks up all replicas' tokens.
    Only this replica's tokens, with heads in shard order, leave the host.
    Exchange FP8/E8M0 bytes together rather than dequantized BF16 rows.
    """
    remainder = raw.shape[0] % world_size
    if remainder or raw.shape[:2] != scales.shape[:2]:
        raise ValueError("Engram rows must contain equal DP token slots")
    packed = torch.cat((raw, scales), dim=-1).contiguous()
    received = torch.empty_like(packed)
    work = dist.all_to_all_single(received, packed, group=group, async_op=True)
    assert work is not None
    return PendingRowExchange(packed, received, work, world_size, raw.shape[-1])


def exchange_rows(
    raw: torch.Tensor,
    scales: torch.Tensor,
    group: dist.ProcessGroup,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return start_exchange_rows(raw, scales, group, world_size).wait()
