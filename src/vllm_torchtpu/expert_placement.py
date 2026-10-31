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
"""Initial expert slot layout and runtime slot-table state for Dynamic EPLB.

When `TPU_MOE_EPLB_DYNAMIC=1` is enabled, each target RoutedExperts layer is
initialized with a blind contiguous-plus-one layout (zero prior profiling):
  * EP rank `r` holds its contiguous block `[r * (E // EP) .. (r + 1) * (E // EP) - 1]`
    plus the first expert of the next rank `((r + 1) % EP) * (E // EP)` in one
    redundant active slot (`DEFAULT_EXTRA_SLOTS = 1`);
  * `DEFAULT_STAGING = 2` spare physical slots sit behind the active slots on
    every rank for background weight staging (`expert_relayout.py`);
  * `expert_slot_table` lists, per logical expert, the physical slot ids
    `r * slots + j` holding it, padded with -1 to `replica_cap = EP`.

Only the fused EP kernel reads slot ids, so a placed layer must be armed for
it; `check_layer_served` refuses anything else.
"""

import re
import threading
from dataclasses import dataclass

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# `model.layers.7.ffn.experts` (a module prefix) and `layers.7.ffn.experts.3.w1`
# (a checkpoint tensor) both name layer 7.
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
# A DSpark drafter's MoE layer i is built from the checkpoint's `mtp.i.*`
# tensors; the drafter module itself is named `model.layers.i`.
_DRAFTER_TENSOR_RE = re.compile(r"^mtp\.(\d+)\.")

TARGET = "target"

DEFAULT_EXTRA_SLOTS = 1
DEFAULT_STAGING = 2

# The layer attribute the slot table is registered under.
EXPERT_SLOT_ATTR = "_tpu_expert_slot"
# The layer attribute naming its (group, layer index) under the layout.
LAYOUT_KEY_ATTR = "_tpu_layout_key"

# (group, layer index) -> the placed layer module, for the dynamic re-layout
# (its tables to install, its weights to move). Modules live as long as the
# model; this process holds one model.
_LAYER_MODULES: dict[tuple[str, int], object] = {}
# Active target group definition populated when target MoE layers are built.
_GROUP: "Group | None" = None
_LOCK = threading.Lock()


@dataclass(frozen=True)
class Group:
    """Initial layout configuration for the target model's MoE layers.

    `slots` is the physical slots per rank: the `active` ones `initial_ranks`
    fills, plus `staging` spare slots a dynamic re-layout stages the next
    generation's experts into (unreferenced by the table, never routed to,
    skipped by the kernel's visit list). `replica_cap` fixes the slot table's
    width so a re-layout can give an expert more copies without changing a
    shape; None takes the most copies any expert has.
    """

    experts: int
    shards: int
    slots: int
    initial_ranks: tuple[tuple[int, ...], ...]
    staging: int = DEFAULT_STAGING
    replica_cap: int | None = None

    def group(self, name: str) -> "Group | None":
        return self if name == TARGET else None

    @property
    def active(self) -> int:
        """Slots per rank the initial layout fills."""
        return self.slots - self.staging

    @property
    def physical(self) -> int:
        """Physical slots over the mesh, the kernel's `e_total`."""
        return self.shards * self.slots

    def ranks(self, _layer: int = 0) -> tuple[tuple[int, ...], ...]:
        """Per rank, the global experts it holds initially in slot order."""
        return self.initial_ranks

    def slot_table(self, _layer: int = 0) -> list[list[int]]:
        """Row e: the slots r * slots + j holding expert e, padded with -1 to
        `replica_cap` (or the most replicas any expert has)."""
        held_at: list[list[int]] = [[] for _ in range(self.experts)]
        for r, held in enumerate(self.initial_ranks):
            for j, e in enumerate(held):
                held_at[e].append(r * self.slots + j)
        width = max(len(h) for h in held_at)
        if self.replica_cap is not None:
            width = max(width, self.replica_cap)
        return [h + [-1] * (width - len(h)) for h in held_at]

    def expert_map(self, rank: int) -> list[int]:
        """Global expert -> this rank's local slot, or -1."""
        table = [-1] * self.experts
        for j, e in enumerate(self.initial_ranks[rank]):
            table[e] = j
        return table


def default_ranks(
    experts: int, shards: int, extra_slots: int = DEFAULT_EXTRA_SLOTS
) -> tuple[tuple[int, ...], ...]:
    """Contiguous block per rank plus the first `extra_slots` expert(s) of the
    following rank(s), wrapping modulo `experts`."""
    if shards <= 0 or experts <= 0 or experts % shards != 0:
        raise ValueError(
            f"expert placement: {experts} experts do not divide over {shards} ranks"
        )
    per_shard = experts // shards
    if extra_slots < 0 or per_shard + extra_slots > experts:
        raise ValueError(
            f"expert placement: invalid extra_slots={extra_slots} for "
            f"{experts} experts over {shards} ranks"
        )
    width = per_shard + extra_slots
    return tuple(
        tuple((r * per_shard + j) % experts for j in range(width))
        for r in range(shards)
    )


def current(vllm_config=None) -> Group | None:
    """The process's active dynamic placement group, or None when disabled."""
    del vllm_config
    with _LOCK:
        return _GROUP


def layer_index(name: str) -> int | None:
    """The decoder layer a module prefix or checkpoint tensor name belongs to."""
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _is_draft_runner() -> bool:
    """Whether the model currently being constructed is the speculative drafter."""
    try:
        from vllm.config import get_current_vllm_config

        return get_current_vllm_config().model_config.runner_type == "draft"
    except AssertionError:
        return False


def has_applied_layers() -> bool:
    return _GROUP is not None


def fetch_filter_ids(name: str, ep_rank: int) -> frozenset[int] | None:
    """The experts the fetch-time filter keeps for a target MoE tensor.

    None defers to the default contiguous filter (either Dynamic EPLB is not
    active, or the tensor belongs to a draft `mtp.*` or non-layer module).
    """
    group = _GROUP
    if group is None or _DRAFTER_TENSOR_RE.match(name) is not None:
        return None
    if layer_index(name) is None:
        return None
    return frozenset(group.initial_ranks[ep_rank])


def apply_to_layer(layer) -> bool:
    """Initialize a target RoutedExperts layer with the default Dynamic EPLB
    layout (contiguous block + next rank's first expert + staging slots).

    Must run before the weights load (from the MoE method's create_weights).
    Registers the slot table the fused EP op takes. Returns whether the layer
    is placed.
    """
    import torch

    global _GROUP
    if not envs.TPU_MOE_EPLB_DYNAMIC or _is_draft_runner():
        return False
    layer_idx = layer_index(layer.layer_name)
    if layer_idx is None:
        return False
    manager = layer.expert_map_manager
    experts = int(manager.global_num_experts)
    shards = int(manager.ep_size)
    if shards <= 1 or experts <= 0 or experts % shards != 0:
        return False
    if manager.num_fused_shared_experts or manager.expert_mask is not None:
        return False

    with _LOCK:
        group = _GROUP
        if group is None:
            ranks = default_ranks(experts, shards, extra_slots=DEFAULT_EXTRA_SLOTS)
            group = Group(
                experts=experts,
                shards=shards,
                slots=len(ranks[0]) + DEFAULT_STAGING,
                initial_ranks=ranks,
                staging=DEFAULT_STAGING,
                replica_cap=shards,
            )
            _GROUP = group
        elif group.experts != experts or group.shards != shards:
            raise ValueError(
                f"expert placement is for {group.shards} ranks x {group.experts} "
                f"experts; {layer.layer_name} has {shards} x {experts}"
            )
        _LAYER_MODULES[(TARGET, layer_idx)] = layer

    rank = manager.ep_rank
    expert_map = torch.tensor(group.expert_map(rank), dtype=torch.int32)
    device = manager.expert_map.device
    manager._expert_map = expert_map.to(device)
    manager._local_num_experts = group.slots
    layer.update_expert_map_info()
    layer.register_buffer(
        EXPERT_SLOT_ATTR,
        torch.tensor(group.slot_table(), dtype=torch.int32, device=device),
        persistent=False,
    )
    setattr(layer, LAYOUT_KEY_ATTR, (TARGET, layer_idx))
    logger.info_once(
        "Dynamic EPLB initial layout applied: rank %d of %d, "
        "%d slots per rank (%d active + %d staging); first placed layer holds %s",
        rank,
        group.shards,
        group.slots,
        group.active,
        group.staging,
        group.initial_ranks[rank][:6],
    )
    return True


def expert_slot_table(layer):
    """The [experts, replicas] i32 slot table of a placed layer, or None."""
    return getattr(layer, EXPERT_SLOT_ATTR, None)


def applied_layers() -> dict[tuple[str, int], object]:
    """(group, layer index) -> placed layer module, for every applied layer."""
    with _LOCK:
        return dict(_LAYER_MODULES)


def layout_key(layer) -> tuple[str, int] | None:
    """The (group, layer index) a placed layer follows, or None."""
    return getattr(layer, LAYOUT_KEY_ATTR, None)


def install_table(layer, table) -> None:
    """Write new slot-table VALUES into a placed layer's table buffer.

    The buffer's shape (experts x replica cap) is a compiled-program input and
    does not change; the values do, between two steps, once every rank has the
    experts the new table names in the slots it names (expert_relayout.py).
    """
    import torch

    buf = expert_slot_table(layer)
    if buf is None:
        raise RuntimeError(f"{getattr(layer, 'layer_name', layer)} has no slot table")
    new = torch.as_tensor(table, dtype=torch.int32)
    if tuple(new.shape) != tuple(buf.shape):
        raise ValueError(
            f"slot table {tuple(new.shape)} does not match the layer's "
            f"{tuple(buf.shape)}; the replica cap is fixed at load"
        )
    buf.copy_(new.to(buf.device))


def physical_experts(layer) -> int | None:
    """The placed layer's physical slot count over the mesh, or None."""
    if expert_slot_table(layer) is None:
        return None
    group = current()
    return group.physical if group is not None else None


def check_layer_served(layer, fused_armed: bool) -> None:
    """A placed layer's weights sit in slot order only the fused op undoes."""
    if expert_slot_table(layer) is not None and not fused_armed:
        raise RuntimeError(
            f"expert layout covers {layer.layer_name}, but the fused EP "
            "MoE kernel is not armed for it; the grouped-matmul path reads "
            "contiguous expert blocks and would run the wrong experts"
        )
