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
"""Sequence parallelism over the TP group with the TP-loaded weights.

Each TP rank holds 1/tp of the rows and runs a projection with every rank's
weight shard: its own, plus a replica of the others' gathered once at load
(`gather_other_shards`). Column-parallel projections then give every head
group of the rank's rows; one all-to-all (`RowExchange.rows_to_heads`) turns
that into the rank's own head group of all rows, which is exactly a TP
attention input, and a second one (`heads_to_rows`) returns attention outputs
to their rows' owners for the row-parallel projection.
"""

from typing import cast

import torch
from torch import nn

from vllm_torchtpu.distributed.collectives import all_to_all
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR


class RowExchange(nn.Module):
    """This rank's row block, and the all-to-alls between rows and head groups."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        from vllm.distributed import get_tp_group

        self.group = get_tp_group()
        self.world = self.group.world_size
        rank = self.group.rank_in_group
        self.others = [r for r in range(self.world) if r != rank]

        def index(values: list[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int32, device=device)

        # Device tensors, not Python ints: every rank must compile the same
        # program, or torch_tpu pairs their collectives wrongly.
        self.register_buffer("rank", index([rank]), persistent=False)
        self.register_buffer("others_index", index(self.others), persistent=False)
        # Where destination rank d's block sits in [own, *others].
        send_order = [
            0 if d == rank else 1 + self.others.index(d) for d in range(self.world)
        ]
        self.register_buffer("send_order", index(send_order), persistent=False)

    def own_rows(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's block of the rows."""
        return x.unflatten(0, (self.world, -1)).index_select(0, self.rank)[0]

    def rows_to_heads(self, own: torch.Tensor, rest: torch.Tensor) -> torch.Tensor:
        """Head groups of this rank's rows -> this rank's head group of all rows."""
        blocks = torch.cat([own[None], rest]).index_select(0, self.send_order)
        return all_to_all(blocks, self.group).flatten(0, 1)

    def heads_to_rows(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """This rank's head group of all rows -> every head group of its rows,
        as (own group, the others')."""
        blocks = all_to_all(x.unflatten(0, (self.world, -1)), self.group)
        return blocks.index_select(0, self.rank)[0], blocks.index_select(
            0, self.others_index
        )


def in_out(linear: nn.Module) -> torch.Tensor:
    """A linear's own weight shard as [in, out]; the TPU linear method keeps
    most weights flipped that way."""
    weight = cast(torch.Tensor, linear.weight)
    return weight if getattr(linear, WEIGHT_FLIPPED_ATTR, False) else weight.t()


def column(
    linear: nn.Module, rest: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A column-parallel projection of [rows, in]: [rows, out] own,
    [tp - 1, rows, out] rest."""
    return torch.matmul(x, in_out(linear)), torch.matmul(x, rest)


def row(
    linear: nn.Module, rest: torch.Tensor, x_own: torch.Tensor, x_rest: torch.Tensor
) -> torch.Tensor:
    """A row-parallel projection of every rank's input block, summed."""
    rest_out = torch.matmul(x_rest.transpose(0, 1).flatten(1), rest.flatten(0, 1))
    return torch.matmul(x_own, in_out(linear)) + rest_out


def gather_other_shards(shard: torch.Tensor, others: list[int]) -> torch.Tensor:
    """The other TP ranks' copies of this rank's `shard`, stacked in rank order."""
    from vllm.distributed import tensor_model_parallel_all_gather

    every = tensor_model_parallel_all_gather(shard.contiguous()[None], dim=0)
    return every[others].contiguous()
