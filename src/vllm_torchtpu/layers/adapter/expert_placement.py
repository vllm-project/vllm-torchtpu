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
"""Static expert placement from a file (``TPU_MOE_EXPERT_PLACEMENT=<npz>``).

For every MoE layer the file names (see ``layers.core.expert_placement``),
logical expert ``e`` lives in physical slot ``slots[e]``:

- at weight load, the layer's expert map sends each checkpoint expert to its
  slot, so every per-expert tensor (weights and scales) lands there;
- after routing, ``to_slots`` replaces each routed expert id by its slot.

Everything after routing (exchange, capacity, expert kernels) runs in slot
space with linear placement, so it is unchanged. Each route meets the same
expert weights as before; only the grouping of a token's routes into
per-rank partial sums changes, and with it the order of the sum.
"""

import numpy as np
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.core import expert_placement as core
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_HOST_SLOTS = "_tpu_expert_slots_host"

# Resolved once, at import, outside the compiled forward.
_PLACEMENT: dict[str, np.ndarray] | None = None
_EP_SIZE = 0
_placed_layers = 0
if envs.TPU_MOE_EXPERT_PLACEMENT:
    _PLACEMENT, _EP_SIZE = core.load(envs.TPU_MOE_EXPERT_PLACEMENT)
    logger.info(
        "TPU_MOE_EXPERT_PLACEMENT=%s: static expert placement for %d MoE "
        "layers, balanced for EP %d",
        envs.TPU_MOE_EXPERT_PLACEMENT,
        len(_PLACEMENT),
        _EP_SIZE,
    )


def _file_slots(layer) -> np.ndarray | None:
    if _PLACEMENT is None:
        return None
    return _PLACEMENT.get(str(getattr(layer, "layer_name", "")))


def host_slots(layer) -> np.ndarray | None:
    """This layer's slots [E] when its experts were placed by the file."""
    return getattr(layer, _HOST_SLOTS, None)


def place_experts(layer) -> None:
    """Point the layer's expert map at the file's slots, before weight load.

    Call from the quant method's ``create_weights``; a no-op for layers the
    file does not name.
    """
    slots = _file_slots(layer)
    if slots is None:
        if _PLACEMENT is not None:
            logger.warning(
                "%s keeps linear expert placement: not in %s",
                layer.layer_name,
                envs.TPU_MOE_EXPERT_PLACEMENT,
            )
        return
    name = layer.layer_name
    parallel = layer.moe_config.moe_parallel_config
    manager = layer.expert_map_manager
    if not parallel.use_ep or manager.expert_map is None:
        raise NotImplementedError(f"{name}: expert placement needs EP")
    if parallel.enable_eplb or manager.placement_strategy != "linear":
        raise NotImplementedError(
            f"{name}: expert placement replaces linear placement; got "
            f"strategy {manager.placement_strategy!r}, EPLB {parallel.enable_eplb}"
        )
    if getattr(manager, "num_fused_shared_experts", 0):
        raise NotImplementedError(f"{name}: fused shared experts are not placed")
    ep_size, num_experts = parallel.ep_size, layer.global_num_experts
    if ep_size != _EP_SIZE or slots.shape[0] != num_experts:
        raise ValueError(
            f"{name}: placement file is for {slots.shape[0]} experts over EP "
            f"{_EP_SIZE}; the layer has {num_experts} over EP {ep_size}"
        )
    local = num_experts // ep_size
    local_index = slots.astype(np.int64) - parallel.ep_rank * local
    expert_map = np.where((local_index >= 0) & (local_index < local), local_index, -1)
    old = manager.expert_map
    manager._expert_map = torch.tensor(expert_map, dtype=old.dtype, device=old.device)
    layer.update_expert_map_info()
    setattr(layer, _HOST_SLOTS, slots)


def register_slots_buffer(layer, *, device: torch.device) -> None:
    """Register ``layer._expert_slots`` (None for layers not placed)."""
    global _placed_layers
    slots = host_slots(layer)
    if slots is None and _file_slots(layer) is not None:
        raise NotImplementedError(
            f"{layer.layer_name}: TPU_MOE_EXPERT_PLACEMENT is implemented for "
            "the NVFP4 grouped-matmul MoE path only"
        )
    buffer = None if slots is None else torch.from_numpy(slots.copy()).to(device)
    layer.register_buffer("_expert_slots", buffer, persistent=False)
    if slots is not None and _PLACEMENT is not None:
        _placed_layers += 1
        if _placed_layers == len(_PLACEMENT):
            logger.info(
                "Expert placement applied to all %d layers of %s",
                _placed_layers,
                envs.TPU_MOE_EXPERT_PLACEMENT,
            )


def to_slots(layer, topk_ids: torch.Tensor) -> torch.Tensor:
    """Routed expert ids -> physical slots (negative ids stay as they are)."""
    slots = getattr(layer, "_expert_slots", None)
    if slots is None:
        return topk_ids
    # Compare, select and max: exact for any slot, and no indexing op
    # (indexing ops compile slowly in the decode graph).
    experts = torch.arange(slots.shape[0], device=topk_ids.device, dtype=torch.int32)
    hit = topk_ids[..., None] == experts
    placed = torch.where(hit, slots, -1).amax(dim=-1)
    return torch.where(topk_ids >= 0, placed, topk_ids).to(topk_ids.dtype)
