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

For every MoE layer the file names (see ``layers.core.expert_permutation``),
logical expert ``e`` lives in physical slot ``slots[e]`` (or, in a file with
copies, in each slot of ``slots[e]``):

- the layer allocates the file's slots per rank (more than ``E / ep_size``
  when the file holds copies);
- at weight load, the layer's expert map sends each checkpoint expert to its
  slot on each rank that holds it, so every per-expert tensor (weights and
  scales) lands there;
- after routing, ``to_slots`` replaces each routed expert id by its slot; a
  route to an expert with ``c`` copies takes copy ``h % c``, ``h`` the top 16
  bits of its routing weight, so a token takes the same copies in any batch.

Everything after routing (exchange, capacity, expert kernels) runs in slot
space with linear placement, so it is unchanged. Each route meets the same
expert weights as before; only the grouping of a token's routes into
per-rank partial sums changes, and with it the order of the sum.
"""

import numpy as np
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.core import expert_permutation as core
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_HOST_SLOTS = "_tpu_expert_slots_host"

# Resolved once, at import, outside the compiled forward.
_PLACEMENT: dict[str, np.ndarray] | None = None
_EP_SIZE = 0
_SLOTS_PER_RANK = 0
_placed_layers = 0
if envs.TPU_MOE_EXPERT_PLACEMENT:
    _PLACEMENT, _EP_SIZE, _SLOTS_PER_RANK = core.load(envs.TPU_MOE_EXPERT_PLACEMENT)
    logger.info(
        "TPU_MOE_EXPERT_PLACEMENT=%s: static expert placement for %d MoE "
        "layers, balanced for EP %d, %d slots per rank, up to %d copies per "
        "expert",
        envs.TPU_MOE_EXPERT_PLACEMENT,
        len(_PLACEMENT),
        _EP_SIZE,
        _SLOTS_PER_RANK,
        max(int((s >= 0).sum(axis=1).max()) for s in _PLACEMENT.values()),
    )


def _file_slots(layer) -> np.ndarray | None:
    """The file's slots [E, C] for this layer, or None."""
    if _PLACEMENT is None:
        return None
    return _PLACEMENT.get(str(getattr(layer, "layer_name", "")))


def host_slots(layer) -> np.ndarray | None:
    """This layer's slots [E, C] when its experts were placed by the file."""
    return getattr(layer, _HOST_SLOTS, None)


def slots_per_rank(layer, default: int) -> int:
    """Expert slots each rank allocates for this layer: the file's for a
    layer it names, else ``default``."""
    return _SLOTS_PER_RANK if _file_slots(layer) is not None else default


def slot_experts(layer) -> np.ndarray | None:
    """The logical expert in each physical slot [ep_size * slots per rank]
    (-1 for an empty slot) when the file placed this layer's experts."""
    slots = host_slots(layer)
    if slots is None:
        return None
    experts = np.full(_EP_SIZE * _SLOTS_PER_RANK, -1, dtype=np.int64)
    e, _ = np.nonzero(slots >= 0)
    experts[slots[slots >= 0]] = e
    return experts


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
    local = _SLOTS_PER_RANK
    allocated = {
        p.shape[0]
        for n, p in layer.named_parameters(recurse=False)
        if n.startswith(("w13", "w2"))
    }
    if allocated != {local}:
        raise ValueError(
            f"{name}: expert weights hold {sorted(allocated)} slots, the "
            f"placement file {local} per rank"
        )
    # Each rank holds at most one copy of an expert: its local index, or -1.
    local_index = slots.astype(np.int64) - parallel.ep_rank * local
    here = (slots >= 0) & (local_index >= 0) & (local_index < local)
    expert_map = np.where(here, local_index, local).min(axis=1)
    expert_map = np.where(expert_map < local, expert_map, -1)
    old = manager.expert_map
    manager._expert_map = torch.tensor(expert_map, dtype=old.dtype, device=old.device)
    layer.update_expert_map_info()
    setattr(layer, _HOST_SLOTS, slots)
    # Slots without an expert are never routed to; zero them so the
    # checkpoint load leaves no uninitialized memory in the weights.
    empty = np.setdiff1d(np.arange(local), expert_map[expert_map >= 0])
    if empty.size:
        index = torch.from_numpy(empty)
        for n, p in layer.named_parameters(recurse=False):
            if n.startswith(("w13", "w2")):
                p.data[index.to(p.device)] = 0


def register_slots_buffer(layer, *, device: torch.device) -> None:
    """Register ``layer._expert_slots`` (one copy per expert) or
    ``layer._expert_copy_digits`` (copies); None for layers not placed."""
    global _placed_layers
    slots = host_slots(layer)
    if slots is None and _file_slots(layer) is not None:
        raise NotImplementedError(
            f"{layer.layer_name}: TPU_MOE_EXPERT_PLACEMENT is implemented for "
            "the NVFP4 grouped-matmul MoE path only"
        )
    buffer = digits = None
    if slots is not None and slots.shape[1] == 1:
        buffer = torch.from_numpy(slots[:, 0].copy()).to(device)
    elif slots is not None:
        # Copies: per expert, its copy count and each copy's slot as two
        # base-256 digits, so that a one-hot matmul finds every entry exactly
        # even with bfloat16 operands.
        copies = (slots >= 0).sum(axis=1, keepdims=True)
        placed = np.maximum(slots, 0)
        table = np.concatenate([copies, placed % 256, placed // 256], axis=1)
        digits = torch.from_numpy(table.astype(np.float32)).to(device)
    layer.register_buffer("_expert_slots", buffer, persistent=False)
    layer.register_buffer("_expert_copy_digits", digits, persistent=False)
    if slots is not None and _PLACEMENT is not None:
        _placed_layers += 1
        if _placed_layers == len(_PLACEMENT):
            logger.info(
                "Expert placement applied to all %d layers of %s",
                _placed_layers,
                envs.TPU_MOE_EXPERT_PLACEMENT,
            )


def to_slots(
    layer, topk_ids: torch.Tensor, topk_weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Routed expert ids -> physical slots (negative ids stay as they are).

    A layer with copies needs ``topk_weights``: they pick each route's copy.
    """
    digits = getattr(layer, "_expert_copy_digits", None)
    if digits is not None:
        if topk_weights is None:
            raise ValueError(f"{layer.layer_name}: copies need the routing weights")
        return _to_copy_slots(digits, topk_ids, topk_weights)
    slots = getattr(layer, "_expert_slots", None)
    if slots is None:
        return topk_ids
    # Compare, select and max: exact for any slot, and no indexing op
    # (indexing ops compile slowly in the decode graph).
    experts = torch.arange(slots.shape[0], device=topk_ids.device, dtype=torch.int32)
    hit = topk_ids[..., None] == experts
    placed = torch.where(hit, slots, -1).amax(dim=-1)
    return torch.where(topk_ids >= 0, placed, topk_ids).to(topk_ids.dtype)


def copy_hash(topk_weights: torch.Tensor) -> torch.Tensor:
    """The top 16 bits of each routing weight as float32 (its bfloat16 bits),
    int32: a function of the token alone, never of its batch position."""
    bits = topk_weights.float().contiguous().view(torch.int32)
    return torch.bitwise_and(torch.bitwise_right_shift(bits, 16), 0xFFFF)


def _to_copy_slots(
    digits: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
) -> torch.Tensor:
    """Routes [T, K] -> the slot of copy ``copy_hash(weight) % c`` of each
    routed expert (``c`` = its copies). ``digits`` [E, 1 + 2C]: copies, then
    the slots' low and high base-256 digits."""
    num_experts = digits.shape[0]
    width = (digits.shape[1] - 1) // 2
    num_tokens, topk = topk_ids.shape
    experts = torch.arange(num_experts, device=topk_ids.device, dtype=torch.int32)
    onehot = (topk_ids.reshape(-1, 1) == experts).float()
    found = (onehot @ digits).round().to(torch.int32).reshape(num_tokens, topk, -1)
    copies = found[..., 0]
    slot = found[..., 1 : 1 + width] + 256 * found[..., 1 + width :]
    pick = torch.remainder(copy_hash(topk_weights), torch.clamp(copies, min=1))
    columns = torch.arange(width, device=topk_ids.device, dtype=torch.int32)
    placed = torch.where(pick[..., None] == columns, slot, -1).amax(dim=-1)
    return torch.where(topk_ids >= 0, placed, topk_ids).to(topk_ids.dtype)
