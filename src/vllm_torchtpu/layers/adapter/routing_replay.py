# SPDX-License-Identifier: Apache-2.0
"""Replay of captured expert ids, for profiling (``TPU_MOE_ROUTING_REPLAY=<npz>``).

Dummy weights route almost uniformly, so a dummy-weight profile never shows the
member that real routing overloads in an expert-parallel exchange. This
strategy makes the router emit expert ids captured from a real-weight run: the
file's ``ids`` [layers, windows, positions, topk] hold, per layer, the routing
of ``positions`` consecutive prompt positions for each of ``windows`` prompt
windows. MoE layer ``i`` replays capture layer ``i % layers``. Each group of
``positions`` consecutive token rows (one request's verify tokens in a decode
step with ``positions - 1`` drafts) takes one window, row ``j`` of the group its
position ``j``; the window comes from the group's router logits and the
engine's DP rank, so it is deterministic for the same inputs, changes from step
to step, and differs between engines. The weights are the router's softmax at
the replayed ids, renormalized. Profiling only: outputs are meaningless.
"""

from __future__ import annotations

import re

import numpy as np
import torch

from vllm_torchtpu.layers.adapter import expert_placement

_LAYER = re.compile(r"layers\.(\d+)\.")


class RoutingReplay:
    def __init__(self, path: str):
        ids = np.load(path)["ids"]
        if ids.ndim != 4:
            raise ValueError(f"{path}: ids must be [layers, windows, positions, topk]")
        self.num_layers, self.num_windows, self.positions, self.topk = ids.shape
        if ids.min() < 0 or ids.max() >= 1 << 16:
            raise ValueError(f"{path}: ids must lie in [0, 2**16)")
        self._ids = ids.astype(np.int32)

    def table(self, layer, device: torch.device) -> torch.Tensor:
        """[windows * positions, topk] ids that ``layer`` replays, on ``device``.

        Built eagerly at weight load and kept as a layer buffer: a table made
        inside the compiled forward would enter it as a host tensor.
        """
        match = _LAYER.search(str(getattr(layer, "layer_name", "")))
        index = (int(match.group(1)) if match else 0) % self.num_layers
        rows = self._ids[index].reshape(-1, self.topk)
        return torch.from_numpy(rows.copy()).to(device)

    def route(
        self,
        layer,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        topk: int,
        tp_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if topk != self.topk:
            raise ValueError(
                f"routing replay holds top-{self.topk}, layer asks top-{topk}"
            )
        table = getattr(layer, "_routing_replay_table", None)
        if table is None:
            table = self.table(layer, router_logits.device)
        num_tokens, num_experts = router_logits.shape
        windows, positions = self.num_windows, self.positions
        device = router_logits.device
        # Dense ops only (comparisons and matmuls, no gathers), in small
        # integers: each indexing op added seconds of compile per MoE layer.
        logits = router_logits[:, :8].float()
        token_hash = torch.remainder(
            (logits * 65536.0).round().to(torch.int32).sum(dim=-1), windows
        )
        row = torch.arange(num_tokens, device=device, dtype=torch.int32)
        group = torch.div(row, positions, rounding_mode="floor")
        # Each row takes the hash of its group's first row.
        leader = (group[:, None] == group[None, :]) & (row[None, :] % positions == 0)
        group_hash = (leader.to(torch.int32) * token_hash[None, :]).sum(dim=-1)
        # The DP rank as tensor data (a per-rank constant would make the ranks
        # compile different programs around the EP collectives).
        experts_start = getattr(layer, "_experts_start", None)
        salt = 0
        if experts_start is not None:
            # Expert slots per rank, redundant ones included.
            local = layer.global_num_experts // layer.moe_config.ep_size
            local += expert_placement.redundant_slots_per_rank(layer)
            salt = torch.div(experts_start, local * tp_size, rounding_mode="floor")
        window = torch.remainder(group_hash + group * 37 + salt * 101, windows)
        pick = window * positions + row % positions
        slots = torch.arange(table.shape[0], device=device, dtype=torch.int32)
        onehot = (pick[:, None] == slots[None, :]).float()
        # A TPU matmul may round float32 operands to bfloat16, which holds
        # integers exactly only up to 256: look the ids up as two base-256
        # digits (ids < 2**16), each found exactly.
        digits = torch.cat(
            [table % 256, torch.div(table, 256, rounding_mode="floor")], 1
        )
        found = onehot @ digits.float()
        low, high = found[:, : self.topk], found[:, self.topk :]
        topk_ids = (low + 256.0 * high).round().to(torch.int32)
        topk_ids = torch.remainder(topk_ids, num_experts)
        probs = router_logits.float().softmax(dim=-1)
        experts = torch.arange(num_experts, device=device, dtype=torch.int32)
        hit = topk_ids[:, :, None] == experts[None, None, :]
        topk_weights = (hit.float() * probs[:, None, :]).sum(dim=-1)
        topk_weights = topk_weights / torch.clamp(
            topk_weights.sum(dim=-1, keepdim=True), min=1e-20
        )
        return topk_weights.to(hidden_states.dtype), topk_ids.to(torch.int32)
