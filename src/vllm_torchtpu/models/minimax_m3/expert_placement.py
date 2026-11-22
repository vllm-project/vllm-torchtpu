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
correction bias the same way, so routing picks slots directly and nothing
changes at run time. A placement that spreads experts often routed together
over different ranks evens out the per-rank expert load of a decode step,
which the post-MoE all-reduce waits for.

The file is JSON, ``{"num_experts": E, "layers": {"<layer id>": [E slots]}}``;
a relative path is read from this package's ``placements`` directory. Layers
it does not list keep the identity placement.
"""

import functools
import json
import os
import re

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Checkpoint names carry a prefix; the model's load_weights sees them without.
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@functools.cache
def _load(path: str, num_experts: int) -> dict[int, tuple[int, ...]]:
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
    logger.info("Routed-expert placement %s: %d MoE layers placed.", path, len(layers))
    return layers


class ExpertPlacement:
    """The placement of TPU_MOE_EXPERT_PLACEMENT, or the identity without it."""

    def __init__(self, num_experts: int) -> None:
        path = envs.TPU_MOE_EXPERT_PLACEMENT
        self._layers = _load(path, num_experts) if path else {}
        self._slot_of = {
            layer: {expert: slot for slot, expert in enumerate(slots)}
            for layer, slots in self._layers.items()
        }

    def __bool__(self) -> bool:
        return bool(self._layers)

    @staticmethod
    def layer_of(name: str) -> int | None:
        m = _LAYER.search(name)
        return int(m.group(1)) if m else None

    def slot(self, name: str, expert: int) -> int:
        """The slot of checkpoint expert ``expert`` in the layer ``name`` is in."""
        layer = self.layer_of(name)
        slots = None if layer is None else self._slot_of.get(layer)
        return expert if slots is None else slots[expert]

    def to_slots(self, name: str, rows: torch.Tensor) -> torch.Tensor:
        """Per-expert rows (the gate's, the correction bias) in slot order."""
        layer = self.layer_of(name)
        slots = None if layer is None else self._layers.get(layer)
        if slots is None:
            return rows
        return rows[torch.tensor(slots, device=rows.device)]
