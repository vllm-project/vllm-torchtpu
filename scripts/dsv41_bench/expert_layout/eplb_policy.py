# SPDX-License-Identifier: Apache-2.0
# Vendored from DeepInfer (github.com/deepseek-ai/DeepInfer, Apache-2.0),
# deepinfer/model_executor/eplb/policy.py @ 9bd9cf2, with the comments
# translated and one constraint added: a rank never holds two copies of one
# logical expert (`unique_per_rank`), which the fused EP kernel's slot table
# and the vLLM expert map both require.
"""Expert load-balancing policy: given per-step, per-layer, per-logical-expert
loads, choose which hot experts to replicate into the spare physical slots and
pack all physical experts onto ranks so that the per-step makespan (the
slowest rank, which under DP lockstep is the layer's cost) summed over the
window is small.

Terms: a *logical* expert is one of the model's; a *physical* expert is a
slot on a rank holding a copy of a logical expert; num_physical = num_logical
+ num_redundant.
"""

import torch


def _batch_greedy_indices(
    tokens_per_expert: torch.Tensor,
    physical_to_logical_map: torch.Tensor,
    logical_count: torch.Tensor,
    capacity: int,
    unique_per_rank: bool = True,
) -> torch.Tensor:
    """Pack physical experts onto ranks, greedily, by historical per-step load.

    tokens_per_expert: [num_steps, num_layers, num_logical]
    physical_to_logical_map: [num_layers, num_physical]
    logical_count: [num_layers, num_logical] copies per logical expert
    capacity: physical experts per rank (num_local_physical_experts)

    Returns indices [num_layers, num_physical]: per layer a permutation of the
    physical ids, consecutive `capacity` entries forming one rank. Physical
    experts are considered in descending order of their per-copy window load
    (ties keep the physical order); each goes to the rank that minimizes
    sum over steps of max over ranks of the rank's per-step load, ties to
    the lowest rank id. With `unique_per_rank`, a rank already holding a copy
    of the same logical expert is excluded.
    """
    weights = tokens_per_expert.sum(dim=0, dtype=torch.float32) / logical_count
    physical_experts = physical_to_logical_map.to(torch.int64)
    physical_order = weights.gather(-1, physical_experts).argsort(
        dim=-1, descending=True, stable=True
    )
    expert_order = physical_experts.gather(-1, physical_order)
    vectors = tokens_per_expert.permute(1, 2, 0).to(
        dtype=torch.float32, memory_format=torch.contiguous_format, copy=True
    )
    vectors.div_(logical_count.unsqueeze(-1))

    num_layers, num_logical, num_steps = vectors.shape
    num_physical = physical_order.size(1)
    assert num_physical % capacity == 0, f"{num_physical=} {capacity=}"
    num_ranks = num_physical // capacity
    loads = vectors.new_zeros((num_layers, num_ranks, num_steps))
    peaks = vectors.new_zeros((num_layers, num_steps))
    occupancy = torch.zeros(
        (num_layers, num_ranks), dtype=torch.int64, device=vectors.device
    )
    held = torch.zeros(
        (num_layers, num_ranks, num_logical), dtype=torch.bool, device=vectors.device
    )
    candidates = torch.empty_like(loads)
    scores = vectors.new_empty((num_layers, num_ranks))
    indices = torch.empty_like(physical_order)
    layers = torch.arange(num_layers, device=vectors.device)
    for job_index in range(num_physical):
        experts = expert_order[:, job_index]
        vector = vectors[layers, experts]
        # candidates[layer][j][step]: rank j's per-step load after adding this
        # expert; against the running peaks that gives the per-step makespan.
        torch.add(loads, vector.unsqueeze(1), out=candidates)
        torch.maximum(candidates, peaks.unsqueeze(1), out=candidates)
        torch.sum(candidates, dim=-1, out=scores)
        scores.masked_fill_(occupancy == capacity, float("inf"))
        if unique_per_rank:
            scores.masked_fill_(held[layers, :, experts], float("inf"))
        rank = scores.argmin(dim=-1)
        if torch.isinf(scores[layers, rank]).any():
            raise ValueError(
                "no rank can take a copy without holding the expert twice; "
                "fewer replicas than ranks are needed"
            )
        occupied = occupancy[layers, rank]
        indices[layers, rank * capacity + occupied] = physical_order[:, job_index]
        occupancy[layers, rank] = occupied + 1
        held[layers, rank, experts] = True
        updated = loads[layers, rank] + vector
        loads[layers, rank] = updated
        torch.maximum(peaks, updated, out=peaks)
    return indices


def rebalance_experts_global(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
    unique_per_rank: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replicate hot experts into the spare slots, then pack every copy.

    tokens_per_expert: [num_steps, num_layers, num_logical]

    Returns
      physical_to_logical_map [num_layers, num_physical]
      logical_to_physical_map [num_layers, num_logical, num_redundant + 1],
                              -1 past the last copy
      logical_count           [num_layers, num_logical] copies per logical expert
    """
    _, num_moe_layers, num_logical_experts = tokens_per_expert.shape
    num_redundant_experts = num_physical_experts - num_logical_experts
    device = tokens_per_expert.device

    physical_to_logical_map = torch.empty(
        num_moe_layers, num_physical_experts, dtype=torch.int, device=device
    )
    logical_to_physical_map = torch.full(
        (num_moe_layers, num_logical_experts, num_redundant_experts + 1),
        -1,
        dtype=torch.int,
        device=device,
    )
    logical_count = torch.ones(
        num_moe_layers, num_logical_experts, dtype=torch.int, device=device
    )

    # Initial layout: the first num_logical slots hold the logical experts,
    # the rest the redundant copies.
    arange_layers = torch.arange(num_moe_layers, dtype=torch.int, device=device)
    arange_logical = torch.arange(num_logical_experts, dtype=torch.int, device=device)
    physical_to_logical_map[:, :num_logical_experts] = arange_logical
    logical_to_physical_map[:, :, 0] = arange_logical

    # Greedy replication, one copy per round: `score[..., j]` is expert j's
    # current per-copy load and `score_added` its per-copy load with one more
    # copy; replacing the argmax by its score_added makes `values[..., j]`
    # the window's maximum per-copy load if the copy went to j, and the argmin
    # of its sum over steps is the expert to replicate.
    tokens_per_expert_all_diff = tokens_per_expert + arange_logical * 1e-4
    # With one copy per rank at most, an expert can have as many copies as
    # there are ranks; the replication loop stops offering it beyond that.
    num_ranks = max(num_physical_experts // max(num_local_physical_experts, 1), 1)
    for i in range(num_redundant_experts):
        score = tokens_per_expert_all_diff / logical_count
        score_added = tokens_per_expert / (logical_count + 1)
        values, indices = score.max(-1, keepdim=True)
        values = values.expand_as(score).contiguous()
        score.scatter_(-1, indices, score_added.gather(-1, indices))
        values.scatter_(-1, indices, score.max(-1, keepdim=True).values)
        summed = values.sum(0)
        if unique_per_rank:
            summed = summed.masked_fill(logical_count >= num_ranks, float("inf"))
        redundancy_indices = summed.argmin(-1)  # [num_moe_layers]

        physical_index = num_logical_experts + i
        physical_to_logical_map[:, physical_index] = redundancy_indices
        redundancy_count = logical_count.gather(
            -1, redundancy_indices.unsqueeze(-1)
        ).squeeze(-1)
        logical_to_physical_map[arange_layers, redundancy_indices, redundancy_count] = (
            physical_index
        )
        logical_count[arange_layers, redundancy_indices] += 1

    if num_local_physical_experts > 1:
        assert num_physical_experts % num_local_physical_experts == 0
        indices = _batch_greedy_indices(
            tokens_per_expert,
            physical_to_logical_map,
            logical_count,
            num_local_physical_experts,
            unique_per_rank=unique_per_rank,
        )
        physical_to_logical_map = physical_to_logical_map.gather(-1, indices)
        # indices.argsort maps old slot -> new slot; remap the slot ids in
        # logical_to_physical_map (-1 entries parked at 0 and restored).
        mask = logical_to_physical_map == -1
        logical_to_physical_map[mask] = 0
        logical_to_physical_map = (
            indices.argsort(-1)
            .gather(
                -1, logical_to_physical_map.view(num_moe_layers, -1).to(torch.int64)
            )
            .view_as(logical_to_physical_map)
            .to(torch.int)
        )
        logical_to_physical_map[mask] = -1

    return physical_to_logical_map, logical_to_physical_map, logical_count


def rank_lists(
    physical_to_logical_map: torch.Tensor, capacity: int
) -> list[list[list[int]]]:
    """[layer][rank] -> the logical experts of that rank's slots in order."""
    out = []
    for row in physical_to_logical_map.tolist():
        out.append(
            [
                row[r * capacity : (r + 1) * capacity]
                for r in range(len(row) // capacity)
            ]
        )
    return out
