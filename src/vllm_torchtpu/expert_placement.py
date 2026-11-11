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
"""Expert slot layouts for expert parallelism: a static layout from a file,
and the runtime slot-table state of Dynamic EPLB.

Two sources decide which experts each EP rank holds, in which slot, with
spare slots holding replicas of hot experts:

  * A STATIC layout, selected by `additional_config.deepseek_v41.expert_layout`
    (a shipped layout's name, a path or a gs:// URI) or, for harnesses, by
    `TPU_EXPERT_PLACEMENT`. The file names, per MoE layer, which global
    (logical) experts each EP rank holds and in what slot order:

        {"experts": 384, "shards": 32, "slots": 13,
         "layers": {"3": [[17, 250, ...13 ids...], ...32 ranks...], ...},
         "draft": {"experts": 128, "slots": 5,
                   "layers": {"0": [[...5 ids...], ...32 ranks...], ...}}}

    Every rank of a listed layer lists `slots` ids, at least experts / shards;
    every expert of the layer is held at least once and at most once per rank.
    Slots beyond experts / shards hold REPLICAS of hot experts: each copy
    computes the share of that expert's tokens the kernel routes to it, at the
    cost of one more expert's weights in HBM per rank. Layers that are not
    listed keep the checkpoint's contiguous blocks. The optional `draft` group
    describes the DSpark drafter's MoE layers, a separate model whose
    `model.layers.i` is built from the checkpoint's `mtp.i.*` tensors; it is
    keyed by the model's `runner_type`, never by layer index.

  * DYNAMIC EPLB (`TPU_MOE_EPLB_DYNAMIC=1`, expert_relayout.py): the target
    layers and the drafter's, each model its own group, start from an initial
    layout and are re-laid at runtime from the kernel's load counts. Without a
    static layout (or its group) the initial layout is blind
    (zero prior profiling): EP rank `r` holds its contiguous block plus the
    first expert of the next rank in one redundant active slot
    (`DEFAULT_EXTRA_SLOTS`). With a static layout, that layout is the initial
    one, layer by layer. Either way `DEFAULT_STAGING` spare physical slots sit
    behind the active slots on every rank for background weight staging, and
    the slot table is `replica_cap = EP` wide so a re-layout can give an
    expert more copies without changing a shape.

What follows from a placed layer, all decided at load:
  * the layer's expert map puts expert `ranks[r][j]` in slot j of rank r, so
    vLLM's weight loader and the fetch-time expert filter load exactly those
    (and the layer holds `slots` experts, not experts / shards);
  * `expert_slot_table` lists, per global expert, the SLOT ids holding it,
    `r * slots + j`, the contiguous-block numbering the fused EP kernel
    addresses, padded with -1. The fused EP kernel (`fused_moe_ep.py`) and the
    shared PyTorch routing entry point (`moe_routing.route` + `fused_moe_gmm`)
    map the selected ids after top-k, so selection, tie-breaking, the router
    weights and the order of the k combine terms are those of the unplaced
    model: every copy holds the same weights, so per token the output is the
    same bits; only where the matmuls run moves. Both accumulate per-slot
    routed row counts into `expert_load` when Dynamic EPLB is on.

The static layout is the design of PR #1408's first revision (`2c55c0109`)
with the `slots` field and the drafter group added; the dynamic state is
#2523's.
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

# Dynamic EPLB's blind initial layout: one redundant active slot per rank, and
# the spare physical slots a re-layout stages the next generation into.
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
# (group, layer index) pairs whose expert map follows a layout in this
# process. The fetch-time filter consults the layout only for these, so a
# model whose MoE method never applied it keeps the default filter.
_APPLIED_LAYERS: set[tuple[str, int]] = set()
# The active Dynamic EPLB groups by name: the target model's MoE layers and
# the drafter's, each built when its first layer is placed with
# `TPU_MOE_EPLB_DYNAMIC` on.
_GROUPS: dict[str, "Group"] = {}
_LOCK = threading.Lock()


@dataclass(frozen=True)
class Group:
    """One layout group: a model's MoE layers over `shards` ranks.

    `slots` is the physical slots per rank: the `active` ones a layer's ranks
    fill, plus `staging` spare slots a dynamic re-layout stages the next
    generation's experts into (unreferenced by the table, never routed to,
    skipped by the kernel's visit list). `layers` holds the per-layer rank
    lists of a static layout; `initial_ranks` is the layout of a layer it
    does not list (Dynamic EPLB's blind default), used only when
    `blind_fallback` is set. `replica_cap` fixes the slot table's width so a
    re-layout can give an expert more copies without changing a shape; None
    takes the most copies any expert has.
    """

    experts: int
    shards: int
    slots: int
    initial_ranks: tuple[tuple[int, ...], ...]
    staging: int = 0
    replica_cap: int | None = None
    layers: dict[int, tuple[tuple[int, ...], ...]] | None = None
    blind_fallback: bool = True

    def group(self, name: str) -> "Group | None":
        return self if name == TARGET else None

    @property
    def active(self) -> int:
        """Slots per rank the layout fills."""
        return self.slots - self.staging

    @property
    def physical(self) -> int:
        """Physical slots over the mesh, the kernel's `e_total`."""
        return self.shards * self.slots

    def ranks(self, layer: int = 0) -> tuple[tuple[int, ...], ...] | None:
        """Per rank, the global experts it holds in slot order; None when the
        layer is unlisted and the group has no blind default."""
        if self.layers is not None and layer in self.layers:
            return self.layers[layer]
        return self.initial_ranks if self.blind_fallback else None

    def slot_table(self, layer: int = 0) -> list[list[int]]:
        """Row e: the slots r * slots + j holding expert e, padded with -1 to
        `replica_cap` (or the most replicas any expert of the layer has)."""
        ranks = self.ranks(layer)
        if ranks is None:
            raise KeyError(f"layer {layer} is not listed in the layout")
        held_at: list[list[int]] = [[] for _ in range(self.experts)]
        for r, held in enumerate(ranks):
            for j, e in enumerate(held):
                held_at[e].append(r * self.slots + j)
        width = max(len(h) for h in held_at)
        if self.replica_cap is not None:
            width = max(width, self.replica_cap)
        return [h + [-1] * (width - len(h)) for h in held_at]

    def expert_map(self, rank: int, layer: int = 0) -> list[int]:
        """Global expert -> this rank's local slot, or -1."""
        ranks = self.ranks(layer)
        if ranks is None:
            raise KeyError(f"layer {layer} is not listed in the layout")
        table = [-1] * self.experts
        for j, e in enumerate(ranks[rank]):
            table[e] = j
        return table


@dataclass(frozen=True)
class Placement:
    """A static layout file: the target group and, optionally, the drafter's."""

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
    return Group(
        experts=experts,
        shards=shards,
        slots=slots,
        initial_ranks=default_ranks(experts, shards, slots - experts // shards),
        staging=0,
        replica_cap=None,
        layers=layers,
        blind_fallback=False,
    )


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
    """The static layout the serving policy or the environment selects, or None.

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
    """The static layout file's content digest for the compile cache key."""
    value = configured_value(vllm_config)
    return _digest(resolve(value)) if value else None


def static_placement(vllm_config=None) -> Placement | None:
    """The process's static layout, or None when none is selected."""
    value = configured_value(vllm_config)
    return _load(resolve(value)) if value else None


def current(vllm_config=None) -> Placement | None:
    """The process's active Dynamic EPLB groups, or None when it is off: the
    target group and, once the drafter placed a layer, the drafter's."""
    del vllm_config
    with _LOCK:
        target = _GROUPS.get(TARGET)
        if target is None:
            return None
        return Placement(target=target, draft=_GROUPS.get(DRAFT))


def dynamic_group(name: str) -> Group | None:
    """The active Dynamic EPLB group of `name` ("target" or "draft"), or None."""
    with _LOCK:
        return _GROUPS.get(name)


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


def _is_draft_runner() -> bool:
    """Whether the model currently being constructed is the speculative drafter."""
    return _runner_group()[0] == DRAFT


def has_applied_layers() -> bool:
    """Whether any layout gates the fetch-time filter: a dynamic group covers
    every layer of its model once it exists; a static layout covers the layers
    an MoE method applied it to."""
    return bool(_GROUPS) or bool(_APPLIED_LAYERS)


def _group_for(key: tuple[str, int]) -> Group | None:
    """The group a placed (group, layer) follows: its dynamic group while
    Dynamic EPLB is on, else the static layout's group."""
    group_name = key[0]
    dynamic = dynamic_group(group_name)
    if dynamic is not None:
        return dynamic
    placement = static_placement()
    return placement.group(group_name) if placement is not None else None


def _tensor_key(name: str) -> tuple[str, int] | None:
    """The (group, layer) a checkpoint tensor belongs to: a `mtp.i.*` tensor
    to draft layer i, a `layers.j.*` tensor to target layer j."""
    drafter = _DRAFTER_TENSOR_RE.match(name)
    if drafter is not None:
        return DRAFT, int(drafter.group(1))
    layer = layer_index(name)
    return None if layer is None else (TARGET, layer)


def fetch_filter_ids(name: str, ep_rank: int) -> frozenset[int] | None:
    """The experts the fetch-time filter keeps for this tensor's layer.

    None defers to the default contiguous filter: either there is no layout
    for the layer, or no MoE method applied a static one to it. A `mtp.i.*`
    tensor belongs to draft layer i; everything else to the target.
    """
    key = _tensor_key(name)
    if key is None:
        return None
    group_name, layer = key
    dynamic = dynamic_group(group_name)
    if dynamic is not None:
        # Dynamic EPLB lays out every layer of its group, blind default included.
        ranks = dynamic.ranks(layer)
        return None if ranks is None else frozenset(ranks[ep_rank])
    if key not in _APPLIED_LAYERS:
        return None
    placement = static_placement()
    group = placement.group(group_name) if placement is not None else None
    ranks = group.ranks(layer) if group is not None else None
    return None if ranks is None else frozenset(ranks[ep_rank])


def apply_to_layer(layer) -> bool:
    """Point a RoutedExperts layer's expert map at its layout.

    Must run before the weights load (from `VllmQuantConfig.get_moe_config` or
    the MoE method's `create_weights`). A layer the static layout lists takes
    that layout; with `TPU_MOE_EPLB_DYNAMIC` a target or drafter layer also
    joins its model's dynamic group (the static ranks as its initial layout,
    or the blind default) and gains the staging slots. Registers the slot
    table the fused EP op and `moe_routing.route` take. Idempotent if called
    more than once on the same layer. Returns whether the layer is placed.
    """
    import torch

    if expert_slot_table(layer) is not None:
        return True
    group_name, base = _runner_group()
    layer_idx = layer_index(layer.layer_name)
    if layer_idx is None:
        return False
    if group_name == DRAFT and layer_idx >= base:
        # Module `layers.{n + i}` of the drafter is its draft layer i.
        layer_idx -= base
    placement = static_placement()
    static_group = placement.group(group_name) if placement is not None else None
    static_ranks = static_group.ranks(layer_idx) if static_group is not None else None
    dynamic = bool(envs.TPU_MOE_EPLB_DYNAMIC)
    if not dynamic and static_ranks is None:
        return False

    manager = layer.expert_map_manager
    experts = int(manager.global_num_experts)
    shards = int(manager.ep_size)
    if (
        static_group is not None
        and static_ranks is not None
        and (experts != static_group.experts or shards != static_group.shards)
    ):
        raise ValueError(
            f"expert layout {configured_value()} ({group_name}) is for "
            f"{static_group.shards} ranks x {static_group.experts} experts; "
            f"{layer.layer_name} has {shards} x {experts}"
        )
    if manager.num_fused_shared_experts or manager.expert_mask is not None:
        if static_ranks is not None:
            raise ValueError(
                f"expert layout: {layer.layer_name} fuses shared experts into "
                "its expert map, which a layout does not describe"
            )
        return False
    if shards <= 1 or experts <= 0 or experts % shards != 0:
        return False

    key = (group_name, layer_idx)
    if dynamic:
        per_shard = experts // shards
        with _LOCK:
            group = _GROUPS.get(group_name)
            if group is None:
                if static_group is not None:
                    active = static_group.slots
                    layers = static_group.layers
                else:
                    active = per_shard + DEFAULT_EXTRA_SLOTS
                    layers = None
                group = Group(
                    experts=experts,
                    shards=shards,
                    slots=active + DEFAULT_STAGING,
                    initial_ranks=default_ranks(experts, shards, active - per_shard),
                    staging=DEFAULT_STAGING,
                    replica_cap=shards,
                    layers=layers,
                    blind_fallback=True,
                )
                _GROUPS[group_name] = group
            elif group.experts != experts or group.shards != shards:
                raise ValueError(
                    f"expert placement is for {group.shards} ranks x "
                    f"{group.experts} experts; {layer.layer_name} has "
                    f"{shards} x {experts}"
                )
    else:
        assert static_group is not None
        group = static_group

    ranks = group.ranks(layer_idx)
    assert ranks is not None
    rank = manager.ep_rank
    expert_map = torch.tensor(group.expert_map(rank, layer_idx), dtype=torch.int32)
    device = manager.expert_map.device
    manager._expert_map = expert_map.to(device)
    manager._local_num_experts = group.slots
    moe_config = getattr(layer, "moe_config", None)
    if moe_config is not None and hasattr(moe_config, "num_local_experts"):
        moe_config.num_local_experts = group.slots
    layer.update_expert_map_info()
    layer.register_buffer(
        EXPERT_SLOT_ATTR,
        torch.tensor(group.slot_table(layer_idx), dtype=torch.int32, device=device),
        persistent=False,
    )
    setattr(layer, LAYOUT_KEY_ATTR, key)
    with _LOCK:
        _LAYER_MODULES[key] = layer
        _APPLIED_LAYERS.add(key)
    logger.info_once(
        "Expert layout applied (%s group, %s%s): rank %d of %d, %d slots per "
        "rank (%d active + %d staging); first placed layer holds %s",
        group_name,
        "dynamic EPLB" if dynamic else "static",
        f", initial layout {configured_value()}" if static_ranks is not None else "",
        rank,
        group.shards,
        group.slots,
        group.active,
        group.staging,
        ranks[rank][:6],
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
    key = layout_key(layer)
    group = _group_for(key) if key is not None else dynamic_group(TARGET)
    return group.physical if group is not None else None


def check_layer_served(layer, fused_armed: bool = True) -> None:
    """Log when a placed layer is served by the GMM routing path rather than
    the fused EP kernel; `moe_routing.route` maps its ids through the slot
    table as the kernel does."""
    if expert_slot_table(layer) is not None and not fused_armed:
        logger.info_once(
            "Placed layer %s is served via moe_routing.route + fused_moe_gmm",
            getattr(layer, "layer_name", "<unnamed>"),
        )
