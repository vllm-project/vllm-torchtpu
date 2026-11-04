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
# Ported to NumPy from DeepInfer (github.com/deepseek-ai/DeepInfer, Apache-2.0),
# deepinfer/model_executor/eplb/policy.py @ 9bd9cf2, with one constraint
# added: a rank never holds two copies of one logical expert
# (`unique_per_rank`), which the fused EP kernel's slot table requires.
"""Expert load-balancing policy: given per-step, per-layer, per-logical-expert
loads, choose which hot experts to replicate into the spare physical slots and
pack all physical experts onto ranks so that the per-step makespan (the
slowest rank, which under DP lockstep is the layer's cost) summed over the
window is small.

Terms: a *logical* expert is one of the model's; a *physical* expert is a
slot on a rank holding a copy of a logical expert; num_physical = num_logical
+ num_redundant. NumPy only: the solve runs in a subprocess that imports
neither torch nor the accelerator stack.
"""

from __future__ import annotations

import numpy as np


def rebalance_experts(
    tokens_per_expert: np.ndarray,
    num_physical_experts: int,
    num_local_physical_experts: int,
    unique_per_rank: bool = True,
    max_copies: int | None = None,
) -> list[list[list[int]]]:
    """Replicate hot experts into the spare slots, then pack every copy.

    tokens_per_expert: [num_steps, num_layers, num_logical]. max_copies caps
    the copies of one expert (the slot table's fixed width); None leaves the
    one-copy-per-rank bound alone. Returns [layer][rank] -> the logical
    experts of that rank's slots in order.

    Replication: one copy per round to the expert whose extra copy minimizes
    the window's summed per-step maximum per-copy load. Packing: physical
    experts in descending per-copy window load, each to the rank minimizing
    the sum over steps of the max rank load, ties to the lowest rank.
    """
    tpe = np.asarray(tokens_per_expert, dtype=np.float32)
    num_steps, num_layers, num_logical = tpe.shape
    num_redundant = num_physical_experts - num_logical
    capacity = int(num_local_physical_experts)
    num_ranks = max(num_physical_experts // max(capacity, 1), 1)
    copy_cap = num_ranks if max_copies is None else min(num_ranks, int(max_copies))
    if copy_cap < 1:
        raise ValueError(f"max_copies must be >= 1; got {max_copies}")

    phy2log = np.empty((num_layers, num_physical_experts), dtype=np.int64)
    logical_count = np.ones((num_layers, num_logical), dtype=np.int64)
    arange_layers = np.arange(num_layers)
    arange_logical = np.arange(num_logical, dtype=np.int64)
    phy2log[:, :num_logical] = arange_logical

    tpe_diff = tpe + arange_logical.astype(np.float32) * np.float32(1e-4)
    for i in range(num_redundant):
        score = tpe_diff / logical_count[None, :, :]
        score_added = tpe / (logical_count[None, :, :] + 1)
        idx = score.argmax(axis=-1, keepdims=True)
        vals = np.broadcast_to(score.max(axis=-1, keepdims=True), score.shape).copy()
        np.put_along_axis(
            score, idx, np.take_along_axis(score_added, idx, axis=-1), axis=-1
        )
        np.put_along_axis(vals, idx, score.max(axis=-1, keepdims=True), axis=-1)
        summed = vals.sum(axis=0)
        if unique_per_rank or max_copies is not None:
            bound = copy_cap if max_copies is None else int(max_copies)
            summed = np.where(logical_count >= bound, np.float32(np.inf), summed)
        red_idx = summed.argmin(axis=-1)
        phy2log[:, num_logical + i] = red_idx
        logical_count[arange_layers, red_idx] += 1

    if capacity > 1:
        assert num_physical_experts % capacity == 0
        weights = tpe.sum(axis=0, dtype=np.float32) / logical_count
        w_phy = np.take_along_axis(weights, phy2log, axis=-1)
        physical_order = np.argsort(-w_phy, axis=-1, kind="stable")
        expert_order = np.take_along_axis(phy2log, physical_order, axis=-1)
        vectors = np.transpose(tpe, (1, 2, 0)).astype(np.float32, copy=True)
        vectors /= logical_count[:, :, None]

        loads = np.zeros((num_layers, num_ranks, num_steps), dtype=np.float32)
        peaks = np.zeros((num_layers, num_steps), dtype=np.float32)
        occupancy = np.zeros((num_layers, num_ranks), dtype=np.int64)
        held = np.zeros((num_layers, num_ranks, num_logical), dtype=bool)
        indices = np.empty_like(physical_order)

        for job_idx in range(num_physical_experts):
            experts = expert_order[:, job_idx]
            vec = vectors[arange_layers, experts]
            cand = np.maximum(loads + vec[:, None, :], peaks[:, None, :])
            scores = cand.sum(axis=-1)
            scores = np.where(occupancy == capacity, np.float32(np.inf), scores)
            if unique_per_rank:
                scores = np.where(
                    held[arange_layers, :, experts], np.float32(np.inf), scores
                )
            rank = scores.argmin(axis=-1)
            if np.isinf(scores[arange_layers, rank]).any():
                raise ValueError(
                    "no rank can take a copy without holding the expert twice; "
                    "fewer replicas than ranks are needed"
                )
            occ = occupancy[arange_layers, rank]
            indices[arange_layers, rank * capacity + occ] = physical_order[:, job_idx]
            occupancy[arange_layers, rank] = occ + 1
            held[arange_layers, rank, experts] = True
            updated = loads[arange_layers, rank] + vec
            loads[arange_layers, rank] = updated
            np.maximum(peaks, updated, out=peaks)

        phy2log = np.take_along_axis(phy2log, indices, axis=-1)

    out = []
    for row in phy2log.tolist():
        out.append([row[r * capacity : (r + 1) * capacity] for r in range(num_ranks)])
    return out
