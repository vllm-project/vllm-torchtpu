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
"""A static expert layout for expert parallelism: which experts each EP rank
holds, in which slot, with spare slots holding replicas of hot experts.

Selected by `additional_config.deepseek_v41.expert_layout` (a shipped layout's
name, a path or a gs:// URI) or, for harnesses, by `TPU_EXPERT_PLACEMENT`.
The file names, per MoE layer, which global (logical) experts each EP rank
holds and in what slot order:

    {"experts": 384, "shards": 32, "slots": 13,
     "layers": {"3": [[17, 250, ...13 ids...], ...32 ranks...], ...},
     "draft": {"experts": 128, "slots": 5,
               "layers": {"0": [[...5 ids...], ...32 ranks...], ...}}}

Every rank of a listed layer lists `slots` ids, at least experts / shards;
every expert of the layer is held at least once and at most once per rank.
Slots beyond experts / shards hold REPLICAS of hot experts: each copy computes
the share of that expert's tokens the kernel routes to it, at the cost of one
more expert's weights in HBM per rank. Layers that are not listed keep the
checkpoint's contiguous blocks. The optional `draft` group describes the
DSpark drafter's MoE layers, which are a separate model whose `model.layers.i`
is built from the checkpoint's `mtp.i.*` tensors; it is keyed by the model's
`runner_type`, never by layer index.

What follows from a listed layer, all decided at load:
  * the layer's expert map puts expert `ranks[r][j]` in slot j of rank r, so
    vLLM's weight loader and the fetch-time expert filter load exactly those
    (and the layer holds `slots` experts, not experts / shards);
  * `expert_slot_table` lists, per global expert, the SLOT ids holding it,
    `r * slots + j`, the contiguous-block numbering the fused EP kernel
    addresses, padded with -1 to the most replicas any expert has. The kernel
    maps the selected ids after top-k, so selection, tie-breaking, the router
    weights and the order of the k combine terms are those of the unplaced
    model: every copy holds the same weights, so per token the output is the
    same bits; only where the matmuls run moves.

Only the fused EP kernel reads slot ids, so a placed layer must be armed for
it; `check_layer_served` refuses anything else.

This is the design of PR #1408's first revision (`2c55c0109`), with the
`slots` field and the drafter group added.
"""

import functools
import hashlib
import json
import os
import re
import subprocess
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
DRAFT = "draft"

# The layer attribute the slot table is registered under.
EXPERT_SLOT_ATTR = "_tpu_expert_slot"
# The layer attribute recording which group (target or draft) placed it.
_GROUP_ATTR = "_tpu_expert_group"

# (group, layer index) pairs whose expert map follows the layout in this
# process. The fetch-time filter consults the layout only for these, so a model
# whose MoE method never applied it keeps the default filter.
_APPLIED_LAYERS: set[tuple[str, int]] = set()
_LOCK = threading.Lock()


@dataclass(frozen=True)
class Group:
    """One layout group: a model's MoE layers over `shards` ranks."""

    experts: int
    shards: int
    slots: int
    layers: dict[int, tuple[tuple[int, ...], ...]]

    @property
    def physical(self) -> int:
        """Physical slots over the mesh, the kernel's `e_total`."""
        return self.shards * self.slots

    def ranks(self, layer: int) -> tuple[tuple[int, ...], ...] | None:
        """Per rank, the global experts it holds in slot order; None if unlisted."""
        return self.layers.get(layer)

    def slot_table(self, layer: int) -> list[list[int]]:
        """Row e: the slots r * slots + j holding expert e, padded with -1 to
        the most replicas any expert of the layer has."""
        held_at: list[list[int]] = [[] for _ in range(self.experts)]
        for r, held in enumerate(self.layers[layer]):
            for j, e in enumerate(held):
                held_at[e].append(r * self.slots + j)
        width = max(len(h) for h in held_at)
        return [h + [-1] * (width - len(h)) for h in held_at]

    def expert_map(self, layer: int, rank: int) -> list[int]:
        """Global expert -> this rank's local slot, or -1."""
        table = [-1] * self.experts
        for j, e in enumerate(self.layers[layer][rank]):
            table[e] = j
        return table


@dataclass(frozen=True)
class Placement:
    target: Group
    draft: Group | None = None

    def group(self, name: str) -> Group | None:
        return self.target if name == TARGET else self.draft

    # Compatibility with the single-group reading of the table.
    @property
    def experts(self) -> int:
        return self.target.experts

    @property
    def shards(self) -> int:
        return self.target.shards

    def slots(self, layer: int) -> int:
        return self.target.slots

    def ranks(self, layer: int):
        return self.target.ranks(layer)

    def slot_table(self, layer: int):
        return self.target.slot_table(layer)


def _parse_group(table: dict, source: str, what: str, shards: int | None) -> Group:
    experts = int(table["experts"])
    if shards is None:
        shards = int(table["shards"])
    if experts % shards:
        raise ValueError(
            f"expert layout {source}: {what} {experts} experts do not divide "
            f"over {shards} ranks"
        )
    layers = {}
    slots = int(table["slots"]) if "slots" in table else None
    for key, ranks in table["layers"].items():
        width = len(ranks[0]) if ranks else 0
        if slots is None:
            slots = width
        if (
            len(ranks) != shards
            or width != slots
            or slots < experts // shards
            or any(len(held) != slots for held in ranks)
        ):
            widths = sorted({len(held) for held in ranks})
            raise ValueError(
                f"expert layout {source}: {what} layer {key} must list {shards} "
                f"ranks of {slots if slots is not None else 'one'} slots each, "
                f"at least {experts // shards}; found {len(ranks)} ranks of "
                f"widths {widths}"
            )
        ranks = tuple(tuple(int(e) for e in held) for held in ranks)
        if any(len(set(held)) != len(held) for held in ranks):
            raise ValueError(
                f"expert layout {source}: {what} layer {key} holds an expert "
                "twice on one rank"
            )
        if sorted({e for held in ranks for e in held}) != list(range(experts)):
            raise ValueError(
                f"expert layout {source}: {what} layer {key} must hold every "
                f"one of the {experts} experts"
            )
        layers[int(key)] = ranks
    if slots is None:
        slots = experts // shards
    return Group(experts=experts, shards=shards, slots=slots, layers=layers)


def parse(table: dict, source: str = "<dict>") -> Placement:
    """Validate a layout table: every expert held, no rank holds one twice,
    equal slots on every rank of a layer."""
    target = _parse_group(table, source, TARGET, None)
    draft = None
    if table.get("draft") is not None:
        draft = _parse_group(table["draft"], source, DRAFT, target.shards)
    return Placement(target=target, draft=draft)


def _read(uri: str) -> str:
    if uri.startswith("gs://"):
        return subprocess.run(
            ["gcloud", "storage", "cat", uri],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    with open(uri) as f:
        return f.read()


@functools.cache
def _read_once(path: str) -> tuple[str, Placement]:
    """One read per file for both the digest and the table."""
    text = _read(path)
    return hashlib.sha256(text.encode()).hexdigest(), parse(json.loads(text), path)


def _digest(path: str) -> str:
    return _read_once(path)[0]


def _load(path: str) -> Placement:
    return _read_once(path)[1]


# Layouts shipped with the package, by name (a file stem in this directory).
PLACEMENT_DIR = os.path.join(os.path.dirname(__file__), "expert_placements")


def resolve(value: str) -> str:
    """A path or gs:// URI as given, or a shipped layout's name -> its file."""
    if value.startswith("gs://") or os.sep in value or value.endswith(".json"):
        return value
    path = os.path.join(PLACEMENT_DIR, f"{value}.json")
    if not os.path.exists(path):
        shipped = sorted(
            f[:-5] for f in os.listdir(PLACEMENT_DIR) if f.endswith(".json")
        )
        raise FileNotFoundError(
            f"expert_layout={value!r} is neither a path nor a shipped layout {shipped}"
        )
    return path


def configured_value(vllm_config=None) -> str | None:
    """The layout the serving policy or the environment selects, or None.

    The typed policy field wins; `TPU_EXPERT_PLACEMENT` is the harness alias.
    `vllm_config` may be passed by a caller that has one (the compile hash);
    otherwise the current config is consulted when there is one.
    """
    if vllm_config is None:
        try:
            from vllm.config import get_current_vllm_config

            vllm_config = get_current_vllm_config()
        except Exception:  # noqa: BLE001 - outside any served model
            vllm_config = None
    additional = getattr(vllm_config, "additional_config", None)
    if isinstance(additional, dict):
        policy = additional.get("deepseek_v41")
        if isinstance(policy, dict) and policy.get("expert_layout"):
            return str(policy["expert_layout"])
    return envs.TPU_EXPERT_PLACEMENT or None


def content_hash(vllm_config=None) -> str | None:
    """The layout file's content digest for the compile cache key."""
    value = configured_value(vllm_config)
    return _digest(resolve(value)) if value else None


def current(vllm_config=None) -> Placement | None:
    """The process's layout, or None when none is selected."""
    value = configured_value(vllm_config)
    return _load(resolve(value)) if value else None


def layer_index(name: str) -> int | None:
    """The decoder layer a module prefix or checkpoint tensor name belongs to."""
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _runner_group() -> tuple[str, int]:
    """Which group the model being built belongs to, and the module index
    its first MoE layer carries.

    The current config's `runner_type` says "draft" for the DSpark drafter.
    The drafter's decoder layers are modules `layers.{n + i}` of an n-layer
    target (models/vllm/deepseek_v41/v2/dspark.py), built from the
    checkpoint's `mtp.{i}.*` tensors, so draft layer i is module n + i; the
    layout's draft group is keyed by i.
    """
    try:
        from vllm.config import get_current_vllm_config

        model_config = get_current_vllm_config().model_config
        runner_type = model_config.runner_type
    except Exception:  # noqa: BLE001 - no config: the target
        return TARGET, 0
    if runner_type != "draft":
        return TARGET, 0
    base = int(getattr(model_config.hf_config, "num_hidden_layers", 0) or 0)
    return DRAFT, base


def has_applied_layers() -> bool:
    return bool(_APPLIED_LAYERS)


def fetch_filter_ids(name: str, ep_rank: int) -> frozenset[int] | None:
    """The experts the fetch-time filter keeps for this tensor's layer.

    None defers to the default contiguous filter: either there is no
    layout for the layer, or no MoE method applied one to it. A `mtp.i.*`
    tensor belongs to draft layer i; everything else to the target.
    """
    drafter = _DRAFTER_TENSOR_RE.match(name)
    if drafter is not None:
        group_name, layer = DRAFT, int(drafter.group(1))
    else:
        group_name, layer = TARGET, layer_index(name)
    if layer is None or (group_name, layer) not in _APPLIED_LAYERS:
        return None
    placement = current()
    group = placement.group(group_name) if placement is not None else None
    if group is None:
        return None
    ranks = group.ranks(layer)
    if ranks is None:
        return None
    return frozenset(ranks[ep_rank])


def apply_to_layer(layer) -> bool:
    """Point a RoutedExperts layer's expert map at its layout.

    Must run before the weights load (from the MoE method's create_weights).
    Registers the slot table the fused EP op takes. Returns whether the layer
    is placed.
    """
    import torch

    placement = current()
    if placement is None:
        return False
    group_name, base = _runner_group()
    group = placement.group(group_name)
    layer_idx = layer_index(layer.layer_name)
    if group is None or layer_idx is None:
        return False
    if group_name == DRAFT and layer_idx >= base:
        # Module `layers.{n + i}` of the drafter is its draft layer i.
        layer_idx -= base
    ranks = group.ranks(layer_idx)
    if ranks is None:
        return False
    manager = layer.expert_map_manager
    if manager.global_num_experts != group.experts or manager.ep_size != group.shards:
        raise ValueError(
            f"expert layout {configured_value()} ({group_name}) is for "
            f"{group.shards} ranks x {group.experts} experts; "
            f"{layer.layer_name} has {manager.ep_size} x "
            f"{manager.global_num_experts}"
        )
    if manager.num_fused_shared_experts or manager.expert_mask is not None:
        raise ValueError(
            f"expert layout: {layer.layer_name} fuses shared experts into "
            "its expert map, which a layout does not describe"
        )
    rank = manager.ep_rank
    expert_map = torch.tensor(group.expert_map(layer_idx, rank), dtype=torch.int32)
    device = manager.expert_map.device
    manager._expert_map = expert_map.to(device)
    manager._local_num_experts = group.slots
    layer.update_expert_map_info()
    layer.register_buffer(
        EXPERT_SLOT_ATTR,
        torch.tensor(group.slot_table(layer_idx), dtype=torch.int32, device=device),
        persistent=False,
    )
    setattr(layer, _GROUP_ATTR, group_name)
    with _LOCK:
        _APPLIED_LAYERS.add((group_name, layer_idx))
    logger.info_once(
        "Expert layout %s applied (%s group): rank %d of %d, %d placed layers, "
        "%d slots per rank; first placed layer holds %s",
        configured_value(),
        group_name,
        rank,
        group.shards,
        len(group.layers),
        group.slots,
        ranks[rank][:6],
    )
    return True


def expert_slot_table(layer):
    """The [experts, replicas] i32 slot table of a placed layer, or None."""
    return getattr(layer, EXPERT_SLOT_ATTR, None)


def physical_experts(layer) -> int | None:
    """The placed layer's physical slot count over the mesh, or None."""
    table = expert_slot_table(layer)
    if table is None:
        return None
    placement = current()
    group = placement.group(getattr(layer, _GROUP_ATTR, TARGET))
    return group.physical


def check_layer_served(layer, fused_armed: bool) -> None:
    """A placed layer's weights sit in slot order only the fused op undoes."""
    if expert_slot_table(layer) is not None and not fused_armed:
        raise RuntimeError(
            f"expert layout covers {layer.layer_name}, but the fused EP "
            "MoE kernel is not armed for it; the grouped-matmul path reads "
            "contiguous expert blocks and would run the wrong experts"
        )
