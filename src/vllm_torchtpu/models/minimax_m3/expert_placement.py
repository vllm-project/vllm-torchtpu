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
"""Static per-layer placement of MiniMax-M3's routed experts on EP ranks.

Expert parallelism gives rank r the experts of slots [r * E / ep, (r+1) * E / ep).
With TPU_MOE_EXPERT_PLACEMENT naming a placement file, slot s of MoE layer l
holds checkpoint expert ``placement[l][s]`` instead of expert s: the loader
puts each expert's weights in its slot and permutes the router's gate rows and
correction bias the same way, so routing picks slots directly. A placement
that spreads experts often routed together over different ranks evens out the
per-rank expert load of a decode step, which the post-MoE all-reduce waits for.

A file may also give each of ``groups`` expert blocks ``replicas_per_group``
extra slots holding copies of hot experts of other blocks. On a decode step
the dense fp4 kernel sends every replicated expert, for all its tokens, to the
copy whose block has fewer touched experts, so a hot expert no longer pins one
block's load; other steps use the home copies. Block q then owns physical
experts [q (E/groups + r), (q + 1) (E/groups + r)): its E/groups slots, then
its r replicas.

The file is JSON, ``{"num_experts": E, "layers": {"<layer id>": [E slots]},
"groups": G, "replicas_per_group": r, "replicas": {"<layer id>": [[slot,
block], ...]}}`` (replicas optional, listed in the order the kernel decides
them); a relative path is read from this package's ``placements`` directory.
Layers it does not list keep the identity placement and no replicas.
"""

import functools
import json
import os
import re
from typing import NamedTuple

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Checkpoint names carry a prefix; the model's load_weights sees them without.
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


class _Spec(NamedTuple):
    slots: dict[int, tuple[int, ...]]
    groups: int
    replicas_per_group: int
    # Per layer, in decision order: (primary slot, block holding the copy).
    replicas: dict[int, tuple[tuple[int, int], ...]]


@functools.cache
def _load(path: str, num_experts: int) -> _Spec:
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(__file__), "placements", path)
    with open(path) as f:
        spec = json.load(f)
    if spec["num_experts"] != num_experts:
        raise ValueError(
            f"{path} places {spec['num_experts']} experts; the model has {num_experts}."
        )
    layers = {}
    for layer, slots in spec["layers"].items():
        if sorted(slots) != list(range(num_experts)):
            raise ValueError(f"{path}: layer {layer} is not a permutation.")
        layers[int(layer)] = tuple(slots)
    groups = spec.get("groups", 1)
    per_group = spec.get("replicas_per_group", 0)
    if groups < 1 or num_experts % groups:
        raise ValueError(
            f"{path}: {groups} blocks do not divide {num_experts} experts."
        )
    replicas = {}
    for layer, pairs in spec.get("replicas", {}).items():
        size = num_experts // groups
        counts = [0] * groups
        if len({s for s, _ in pairs}) != len(pairs):
            raise ValueError(f"{path}: layer {layer} copies a slot twice.")
        for slot, block in pairs:
            if not (0 <= slot < num_experts and 0 <= block < groups):
                raise ValueError(f"{path}: layer {layer} replica {slot, block}.")
            if block == slot // size:
                raise ValueError(f"{path}: layer {layer} copies slot {slot} home.")
            counts[block] += 1
        if any(c > per_group for c in counts):
            raise ValueError(f"{path}: layer {layer} overfills a block's replicas.")
        replicas[int(layer)] = tuple((int(s), int(b)) for s, b in pairs)
    logger.info(
        "Routed-expert placement %s: %d MoE layers placed, %d replica slots per "
        "block of %d blocks.",
        path,
        len(layers),
        per_group,
        groups,
    )
    return _Spec(layers, groups, per_group, replicas)


class ExpertPlacement:
    """The placement of TPU_MOE_EXPERT_PLACEMENT, or the identity without it."""

    def __init__(self, num_experts: int) -> None:
        path = envs.TPU_MOE_EXPERT_PLACEMENT
        spec = _load(path, num_experts) if path else _Spec({}, 1, 0, {})
        self._num_experts = num_experts
        self._layers = spec.slots
        self._slot_of = {
            layer: {expert: slot for slot, expert in enumerate(slots)}
            for layer, slots in self._layers.items()
        }
        self.groups = spec.groups
        self.replicas_per_group = spec.replicas_per_group
        self._replicas = spec.replicas

    def __bool__(self) -> bool:
        return bool(self._layers or self._replicas)

    @property
    def num_physical_experts(self) -> int:
        return self._num_experts + self.groups * self.replicas_per_group

    @staticmethod
    def layer_of(name: str) -> int | None:
        m = _LAYER.search(name)
        return int(m.group(1)) if m else None

    def slot(self, name: str, expert: int) -> int:
        """The slot of checkpoint expert ``expert`` in the layer ``name`` is in."""
        layer = self.layer_of(name)
        slots = None if layer is None else self._slot_of.get(layer)
        return expert if slots is None else slots[expert]

    def _physical(self, slot: int) -> int:
        size = self._num_experts // self.groups
        return (slot // size) * (size + self.replicas_per_group) + slot % size

    def _copies(self, layer: int | None, slot: int) -> list[tuple[int, int]]:
        """(block, physical expert) of each replica of ``slot``."""
        size = self._num_experts // self.groups
        pairs = () if layer is None else self._replicas.get(layer, ())
        copies, used = [], [0] * self.groups
        for s, block in pairs:
            if s == slot:
                copies.append(
                    (
                        block,
                        block * (size + self.replicas_per_group) + size + used[block],
                    )
                )
            used[block] += 1
        return copies

    def physical_experts(self, name: str, expert: int) -> list[int]:
        """The physical experts checkpoint expert ``expert`` is loaded into."""
        slot = self.slot(name, expert)
        layer = self.layer_of(name)
        return [self._physical(slot)] + [p for _, p in self._copies(layer, slot)]

    def owner_slots(self, name: str, expert: int) -> list[int]:
        """Slots of the blocks holding checkpoint expert ``expert``, in the
        E-slot numbering the EP weight filter's ownership map uses."""
        slot = self.slot(name, expert)
        size = self._num_experts // self.groups
        layer = self.layer_of(name)
        return [slot] + [block * size for block, _ in self._copies(layer, slot)]

    def to_slots(self, name: str, rows: torch.Tensor) -> torch.Tensor:
        """Per-expert rows (the gate's, the correction bias) in slot order."""
        layer = self.layer_of(name)
        slots = None if layer is None else self._layers.get(layer)
        if slots is None:
            return rows
        return rows[torch.tensor(slots, device=rows.device)]

    def replica_table(self, prefix: str) -> list[int] | None:
        """The copy choices of the layer ``prefix`` names, for the dense fp4
        decode kernel (kernels/experimental/moe_dense_expert_fp4.py), or None
        without replicas: per replica, in decision order, (slot, its home
        block, the block holding the copy, the copy's index among that block's
        experts), flattened; a layer without replicas has an empty table."""
        if not self.replicas_per_group:
            return None
        layer = self.layer_of(prefix + ".")
        reps = [] if layer is None else self._replicas.get(layer, ())
        size = self._num_experts // self.groups
        local = size + self.replicas_per_group
        table = []
        for slot, block in reps:
            (phys,) = [p for b, p in self._copies(layer, slot) if b == block]
            table += [slot, slot // size, block, phys - block * local]
        return table
