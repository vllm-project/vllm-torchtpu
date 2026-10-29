# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Static per-layer expert placement for expert-parallel MoE (VLLM_TPU_MOE_EXPERT_PLACEMENT).

A placement table gives, for each MoE layer, the logical expert held at every
physical slot: ``placement[L][p] = e``, physical slot ``p = rank * G + s`` with
``G`` local experts per EP rank. It is computed offline from router statistics
of a fixed calibration set (so the heavily routed experts of a layer spread
over the ranks) and is read once at load; nothing about it depends on the
traffic being served.

Each rank loads the logical experts of its physical slots (the layer's expert
map, and the per-layer checkpoint fetch filter). The router is untouched and
selects logical experts exactly as without placement (same top-k, same
tie-break); the selected ids are then mapped to physical slots by the layer's
table ``_expert_phys`` [experts] int32 -- in ``moe_routing.route`` for the
paths that route in torch (fused MoE decode, GMM) and inside the adaptive fused
EP op right after its top-k. Downstream every kernel reads physical ids with
the usual linear meaning (rank = id // G). Routing is unchanged; the output
differs only by reassociation (experts sum into different per-rank partials
before the all-reduce). The megakernels route in-kernel and do not take the
table yet, so placement refuses to start with VLLM_TPU_MEGAKERNEL set. Layers
absent from the table (the MTP drafter's) keep linear placement.
"""

from __future__ import annotations

import functools
import json
import re

import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@functools.cache
def placement_table() -> dict[int, tuple[int, ...]] | None:
    """{layer index: logical expert per physical slot}, or None when off."""
    path = envs.VLLM_TPU_MOE_EXPERT_PLACEMENT
    if not path:
        return None
    with open(path) as f:
        raw = json.load(f)
    table = {}
    for key, perm in raw["placement"].items():
        m = re.match(r"(\d+)", key)
        if m is None:
            raise ValueError(f"placement key {key!r} has no layer index")
        perm = tuple(int(e) for e in perm)
        if sorted(perm) != list(range(len(perm))):
            raise ValueError(f"placement for layer {key} is not a permutation")
        table[int(m.group(1))] = perm
    if envs.VLLM_TPU_MEGAKERNEL not in ("", "0", "off"):
        raise ValueError(
            "VLLM_TPU_MOE_EXPERT_PLACEMENT does not support the megakernels yet "
            f"(VLLM_TPU_MEGAKERNEL={envs.VLLM_TPU_MEGAKERNEL!r}); unset one of them"
        )
    logger.info("MoE expert placement: %d layers from %s", len(table), path)
    return table


def layer_placement(layer_index: int) -> tuple[int, ...] | None:
    table = placement_table()
    return None if table is None else table.get(layer_index)


def local_experts(perm: tuple[int, ...], ep_rank: int, ep_size: int) -> list[int]:
    """Logical experts of ``ep_rank``'s physical slots, in slot order."""
    g = len(perm) // ep_size
    return list(perm[ep_rank * g : (ep_rank + 1) * g])


def apply_expert_map(routed_experts, layer_index: int) -> None:
    """Point the layer's expert map at its placement: logical expert
    ``perm[rank * G + s]`` -> local slot ``s``, every other expert -1."""
    perm = layer_placement(layer_index)
    if perm is None or not routed_experts.use_ep:
        return
    manager = routed_experts.expert_map_manager
    current = manager.expert_map
    if current is None or current.numel() != len(perm):
        raise ValueError(
            f"layer {layer_index}: placement for {len(perm)} experts, expert map "
            f"{None if current is None else current.numel()}"
        )
    ep = routed_experts.moe_config.moe_parallel_config
    host = [-1] * len(perm)
    for slot, expert in enumerate(local_experts(perm, ep.ep_rank, ep.ep_size)):
        host[expert] = slot
    manager._expert_map = torch.tensor(host, dtype=current.dtype, device=current.device)
    routed_experts.update_expert_map_info()
    phys = [0] * len(perm)
    for slot, expert in enumerate(perm):
        phys[expert] = slot
    routed_experts.register_buffer(
        "_expert_phys",
        torch.tensor(phys, dtype=torch.int32, device=current.device),
        persistent=False,
    )


def map_to_physical(layer, topk_ids: torch.Tensor) -> torch.Tensor:
    """Selected logical expert ids -> physical slots (identity without a table)."""
    phys = getattr(layer, "_expert_phys", None)
    if phys is None:
        return topk_ids
    return phys[topk_ids.long()].to(topk_ids.dtype)


class PerLayerExpertIds(set):
    """The checkpoint fetch filter's local expert set, per layer: the set itself
    is the linear placement (layers without a table entry), ``by_layer`` the
    placed layers' sets."""

    def __init__(self, linear, by_layer):
        super().__init__(linear)
        self.by_layer = by_layer

    def keeps(self, name: str, expert_id: int) -> bool:
        m = _LAYER_RE.search(name)
        if m is not None and int(m.group(1)) in self.by_layer:
            return expert_id in self.by_layer[int(m.group(1))]
        return expert_id in self


def fetch_filter(linear_ids: set[int], ep_rank: int, ep_size: int) -> set[int]:
    """The fetch filter for this rank under the placement table (``linear_ids``
    when placement is off)."""
    table = placement_table()
    if not table:
        return linear_ids
    return PerLayerExpertIds(
        linear_ids,
        {layer: set(local_experts(perm, ep_rank, ep_size)) for layer, perm in table.items()},
    )
