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
"""TPU_EXPERT_PLACEMENT=<json>: a static expert placement for expert parallelism.

The file names, per MoE layer, which global experts each EP rank holds and in
what slot order:

    {"experts": 384, "shards": 16,
     "layers": {"3": [[17, 250, ...24 ids...], ...16 ranks...],
                "7": [[94, 13, ...24 ids...], ...16 ranks...], ...}}

A listed layer is a PERMUTATION of its experts: one list per EP rank, each
holding experts / shards of them (24 above), and every expert appears exactly
once. Placement only moves
experts between ranks; it never replicates one, so every rank keeps the
checkpoint's expert count and memory. Layers that are not listed keep the
checkpoint's contiguous blocks.

A placed layer is renumbered at load: expert `ranks[r][j]` becomes expert
`r * (experts / shards) + j`, its SLOT.
  * `apply` points the layer's expert map at the placement before the weights
    load, so vLLM's weight loader and the fetch-time expert filter put expert
    `ranks[r][j]` in slot j of rank r;
  * `permute_routers` renumbers the router the same way once they have
    loaded: the gate's rows (and bias), `e_score_correction_bias`, and the
    expert ids in a hash-routed layer's table.
Both run for every model, drafters included, with no code in the model: the
TPU MoE runner calls `apply` for each layer it builds, and the model-loader
patch calls `permute_routers` between a loader writing the checkpoint and
post-processing it. A layer qualifies when its MoE runner computes the router
logits from a gate it holds and selects among single experts rather than
groups of them (DeepSeek-V4, GLM-5 and Qwen3-MoE, for example).

Copy slots (optional). A table may also give every rank `copies_per_rank`
more slots after its primaries, each holding a redundant copy of another
rank's expert:

    {"experts": 512, "shards": 8, "copies_per_rank": 4,
     "layers": {"0": [[...64 primaries..., ...4 copies...], ...8 ranks...]},
     "copy_order": {"0": [463, 367, ...the layer's copied experts...]}}

The first experts / shards ids of every rank are the permutation above,
checked exactly as without copies; the router, the slot numbering and every
kernel but the few-token MoE kernel see only those primaries. A copied expert
has one copy, on a rank that does not hold it as a primary. `copy_order`
lists each layer's copied experts hottest first (the few-token kernel's
preference); without it the copy slots are taken in rank order. Copy slots
change how many experts a rank holds, which vLLM fixes when it allocates the
layer's weights, before `apply` runs, so a table with copies also installs
`install_copy_slot_hook`. A table without the two keys is the permutation
table above, parsed and applied exactly as before.

Every MoE kernel then sees an ordinary layer whose experts sit in contiguous
blocks, so no kernel and no compiled program changes. The router computes the
same scores in slot order, so tokens select the same experts with the same
weights. Only an exact tie between two experts' selection scores can resolve
differently, since top-k breaks ties by index. With bf16 router logits such
ties are common enough that greedy outputs drift from the unplaced model's
within a few dozen tokens; a tie carries no router preference, so quality does
not move.
"""

import functools
import json
import re
import subprocess
from dataclasses import dataclass, field

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# `model.layers.7.ffn.experts` (a module prefix) and `layers.7.ffn.experts.3.w1`
# (a checkpoint tensor) both name layer 7.
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
# A drafter's MoE layer i is layer n + i of an n-layer target. A DSpark
# drafter builds it as module `layers.{n + i}` from the checkpoint's
# `mtp.{i}.*` tensors; Qwen3.5 and Qwen3-Next MTP name both the module and the
# tensors `mtp.layers.{i}.*`.
_DRAFTER_RE = re.compile(r"(?:^|\.)mtp\.(?:layers\.)?(\d+)\.")
# A stalled GCS or metadata endpoint must not hang model init on every rank.
_GCS_READ_TIMEOUT_S = 120

# The attribute a placed RoutedExperts layer holds its slot order under:
# entry s is the checkpoint's id of the expert in slot s.
EXPERT_ORDER_ATTR = "_tpu_expert_order"
# Set on a placed RoutedExperts layer once its router is in slot order.
ROUTER_RENUMBERED_ATTR = "_tpu_router_renumbered"
# Set on a RoutedExperts layer whose weights carry redundant expert copies after
# the rank's primary experts: the number of primary slots. Unset, every slot is
# a primary (main's permutation tables never set it).
PRIMARY_EXPERTS_ATTR = "_tpu_primary_experts"

# Layer index -> the EP rank whose slots this process's expert map holds, for
# every layer whose expert map follows the placement. The fetch-time filter
# consults the placement only for these, and at exactly that rank, so it keeps
# what the expert map loads by construction (under hierarchical EP the rank is
# the chip's, not the one the dp/pcp/tp formula gives), and a layer the
# placement was never applied to keeps the default filter.
_APPLIED_LAYERS: dict[int, int] = {}
# The placed layers whose router `permute_routers` has renumbered.
_PERMUTED_LAYERS: set[int] = set()


@dataclass(frozen=True)
class Placement:
    experts: int
    shards: int
    layers: dict[int, tuple[tuple[int, ...], ...]]
    # Copy slots after every rank's primaries, and per layer the copied
    # experts hottest first (absent: the copy slots in rank order).
    copies_per_rank: int = 0
    copy_order: dict[int, tuple[int, ...]] = field(default_factory=dict)

    @property
    def per_rank(self) -> int:
        """Primary slots per rank: experts / shards."""
        return self.experts // self.shards

    @property
    def slots_per_rank(self) -> int:
        """Every slot a rank holds: its primaries, then its copies."""
        return self.per_rank + self.copies_per_rank

    def ranks(self, layer: int) -> tuple[tuple[int, ...], ...] | None:
        """Per rank, the global experts it holds in slot order (primaries, then
        copies); None if unlisted."""
        return self.layers.get(layer)

    def slot_order(self, layer: int) -> tuple[int, ...]:
        """Entry s: the primary expert in slot s, r * (experts / shards) + j."""
        n = self.per_rank
        return tuple(e for held in self.layers[layer] for e in held[:n])

    def virtual_ids(self, layer: int) -> dict[int, int]:
        """Global expert -> its primary slot r * (experts / shards) + j, the id
        the renumbered router gives it."""
        return {e: s for s, e in enumerate(self.slot_order(layer))}

    def holder_tables(
        self, layer: int, rank: int
    ) -> tuple[list[int], list[int], list[int]]:
        """For the few-token kernel, in primary (router) ids: rank `rank`'s slot
        -> id table (primaries, then its copies), and per id the rank of its
        primary and of its copy (-1 without one)."""
        vid = self.virtual_ids(layer)
        n = self.per_rank
        prim = [v // n for v in range(self.experts)]
        copy = [-1] * self.experts
        for r, held in enumerate(self.layers[layer]):
            for e in held[n:]:
                copy[vid[e]] = r
        gid = [vid[e] for e in self.layers[layer][rank]]
        return gid, prim, copy

    def hot_order(self, layer: int) -> list[int]:
        """The layer's copied experts in primary ids, hottest first: the
        table's copy_order, else the copy slots in rank order."""
        vid = self.virtual_ids(layer)
        if layer in self.copy_order:
            return [vid[e] for e in self.copy_order[layer]]
        n = self.per_rank
        return [vid[e] for held in self.layers[layer] for e in held[n:]]

    def slot_table(self, layer: int) -> list[int]:
        """Entry e: the slot holding expert e."""
        slot = [0] * self.experts
        for s, e in enumerate(self.slot_order(layer)):
            slot[e] = s
        return slot


def parse(table: dict, source: str = "<dict>") -> Placement:
    """Validate a placement table: each listed layer permutes its experts over
    the ranks, experts / shards to a rank."""
    experts = int(table["experts"])
    shards = int(table["shards"])
    if experts <= 0 or shards <= 0:
        raise ValueError(
            f"expert placement {source}: experts and shards must be positive, "
            f"not {experts} and {shards}"
        )
    if experts % shards:
        raise ValueError(
            f"expert placement {source}: {experts} experts do not divide over "
            f"{shards} ranks"
        )
    per_rank = experts // shards
    copies = table.get("copies_per_rank", 0)
    if type(copies) is not int or copies < 0:
        raise ValueError(
            f"expert placement {source}: copies_per_rank must be a non-negative "
            f"integer, not {copies!r}"
        )
    layers = {}
    for key, ranks in table["layers"].items():
        try:
            layer = int(key)
        except ValueError:
            raise ValueError(
                f"expert placement {source}: layer key {key!r} is not a layer index"
            ) from None
        if layer < 0:
            raise ValueError(f"expert placement {source}: layer {key} is negative")
        if layer in layers:
            raise ValueError(
                f"expert placement {source}: layer {layer} is listed twice"
            )
        slots = per_rank + copies
        if len(ranks) != shards or any(len(held) != slots for held in ranks):
            raise ValueError(
                f"expert placement {source}: layer {key} must list {shards} "
                f"ranks of {slots} experts each"
            )
        ranks = tuple(tuple(int(e) for e in held) for held in ranks)
        if sorted(e for held in ranks for e in held[:per_rank]) != list(range(experts)):
            raise ValueError(
                f"expert placement {source}: layer {key} must hold each of the "
                f"{experts} experts exactly once"
            )
        if copies:
            _check_copies(source, key, ranks, experts, per_rank)
        layers[layer] = ranks
    copy_order = _parse_copy_order(source, table.get("copy_order"), layers, per_rank)
    return Placement(
        experts=experts,
        shards=shards,
        layers=layers,
        copies_per_rank=copies,
        copy_order=copy_order,
    )


def _check_copies(source, key, ranks, experts, per_rank) -> None:
    """Every copy slot holds an in-range expert that its rank does not hold as
    a primary, and each copied expert has exactly one copy."""
    copied = []
    for r, held in enumerate(ranks):
        own = set(held[:per_rank])
        for e in held[per_rank:]:
            if not 0 <= e < experts:
                raise ValueError(
                    f"expert placement {source}: layer {key} rank {r} copies "
                    f"expert {e}, outside 0..{experts - 1}"
                )
            if e in own:
                raise ValueError(
                    f"expert placement {source}: layer {key} rank {r} copies "
                    f"its own primary expert {e}"
                )
            copied.append(e)
    if len(copied) != len(set(copied)):
        raise ValueError(
            f"expert placement {source}: layer {key} copies an expert twice; "
            "each copied expert has one copy"
        )


def _parse_copy_order(source, raw, layers, per_rank) -> dict[int, tuple[int, ...]]:
    """`copy_order`: per listed layer, exactly its copied experts."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"expert placement {source}: copy_order must be an object")
    order = {}
    for key, experts in raw.items():
        try:
            layer = int(key)
        except ValueError:
            raise ValueError(
                f"expert placement {source}: copy_order key {key!r} is not a "
                "layer index"
            ) from None
        if layer not in layers:
            raise ValueError(
                f"expert placement {source}: copy_order lists layer {key}, "
                "which the table does not place"
            )
        experts = tuple(int(e) for e in experts)
        copied = sorted(e for held in layers[layer] for e in held[per_rank:])
        if sorted(experts) != copied:
            raise ValueError(
                f"expert placement {source}: copy_order for layer {key} must "
                "list exactly the layer's copied experts"
            )
        order[layer] = experts
    return order


def _read(uri: str) -> str:
    if uri.startswith("gs://"):
        try:
            return subprocess.run(
                ["gcloud", "storage", "cat", uri],
                check=True,
                capture_output=True,
                text=True,
                timeout=_GCS_READ_TIMEOUT_S,
            ).stdout
        except FileNotFoundError as e:
            raise RuntimeError(
                f"TPU_EXPERT_PLACEMENT={uri} is read with the gcloud CLI, which "
                "this image does not have; copy the table to a local path"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"TPU_EXPERT_PLACEMENT={uri}: gcloud storage cat timed out after "
                f"{_GCS_READ_TIMEOUT_S} s"
            ) from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"TPU_EXPERT_PLACEMENT={uri}: gcloud storage cat failed: "
                f"{e.stderr.strip()}"
            ) from e
    with open(uri) as f:
        return f.read()


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """A JSON object as a dict, refusing a key it repeats (json.loads would
    silently keep the last)."""
    seen = set()
    for key, _ in pairs:
        if key in seen:
            raise ValueError(f"key {key!r} appears twice")
        seen.add(key)
    return dict(pairs)


@functools.cache
def _load(path: str) -> Placement:
    """The table, read once per process: the expert maps, the fetch filter and
    the routers all follow the same read, even if the file is rewritten."""
    try:
        table = json.loads(_read(path), object_pairs_hook=_unique_keys)
    except ValueError as e:  # also json.JSONDecodeError
        raise ValueError(f"expert placement {path}: {e}") from e
    return parse(table, path)


def reset_model_state() -> None:
    """Forget the layers placed so far and the target's depth, so a second
    model load in the process starts from nothing; the table read is kept."""
    _target_layers.cache_clear()
    _APPLIED_LAYERS.clear()
    _PERMUTED_LAYERS.clear()


def clear_cache() -> None:
    """Forget every table read so far, for a caller that rewrites one, and
    every model's placement state."""
    _load.cache_clear()
    reset_model_state()


def current() -> Placement | None:
    """The process's placement, or None when TPU_EXPERT_PLACEMENT is unset."""
    value = envs.TPU_EXPERT_PLACEMENT
    return _load(value) if value else None


def layer_index(name: str) -> int | None:
    """The table layer a module prefix or checkpoint tensor name belongs to:
    its decoder layer, or n + i for a drafter's MoE layer i (an n-layer
    target)."""
    drafter = _DRAFTER_RE.search(name)
    if drafter is not None:
        return _target_layers() + int(drafter.group(1))
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def placed_ranks(name: str) -> tuple[tuple[int, ...], ...] | None:
    """The placement of the layer `name` belongs to, or None if it has none."""
    placement = current()
    layer = layer_index(name)
    if placement is None or layer is None:
        return None
    return placement.ranks(layer)


def has_applied_layers() -> bool:
    return bool(_APPLIED_LAYERS)


def _require() -> Placement:
    placement = current()
    if placement is None:
        raise RuntimeError("no expert placement: TPU_EXPERT_PLACEMENT is unset")
    return placement


# The table's facts by the names the few-token MoE kernel and the MoE methods
# ask for them; all but has_copies and placed need a table.
def has_copies() -> bool:
    """Whether the table gives the ranks copy slots after their primaries."""
    placement = current()
    return placement is not None and placement.copies_per_rank > 0


def placed(name: str) -> bool:
    """Whether the table places the layer a prefix or tensor name belongs to."""
    return placed_ranks(name) is not None


def shards() -> int:
    return _require().shards


def global_count() -> int:
    return _require().experts


def owned() -> int:
    """Primary slots per rank, the slots the router and the other kernels see."""
    return _require().per_rank


def slots() -> int:
    """Every slot a rank holds, primaries then copies."""
    return _require().slots_per_rank


def holder_tables(layer: int, rank: int) -> tuple[list[int], list[int], list[int]]:
    return _require().holder_tables(layer, rank)


def hot_order(layer: int) -> list[int]:
    return _require().hot_order(layer)


def fetch_filter_ids(name: str) -> frozenset[int] | None:
    """The experts the fetch-time filter keeps for this tensor's layer.

    None defers to the default contiguous filter: either there is no
    placement for the layer, or it was not applied to it. Otherwise the
    experts of the EP rank `apply` built the layer's expert map for, so the
    fetch and the expert map cannot pick different ranks.
    """
    layer = layer_index(name)
    placement = current()
    if placement is None or layer is None or layer not in _APPLIED_LAYERS:
        return None
    ranks = placement.ranks(layer)
    if ranks is None:
        return None
    return frozenset(ranks[_APPLIED_LAYERS[layer]])


def _unsupported_routing(runner) -> str | None:
    """Why renumbering this layer's experts would change its routing, if it
    would: the router must take its logits from a gate the runner holds and
    choose among single experts, not contiguous groups of them."""
    layer = runner.routed_experts
    if runner.gate is None:
        return "computes its router logits outside the MoE layer"
    if layer.use_grouped_topk and (layer.num_expert_group or 1) > 1:
        return "selects its experts in groups of contiguous expert ids"
    if layer.custom_routing_function is not None:
        return "routes with a custom routing function"
    return None


def apply(runner) -> bool:
    """Point a MoE layer's expert map at its placement; True if it is placed.

    Called for every MoE layer by the TPU MoE runner's constructor: before the
    weights load, whatever the layer's quant method and kernel. The router is
    renumbered to match after the load, by `permute_routers`.
    """
    placement = current()
    if placement is None:
        return False
    layer = runner.routed_experts
    name = layer.layer_name
    layer_idx = layer_index(name)
    ranks = placed_ranks(name)
    if layer_idx is None or ranks is None:
        return False
    manager = layer.expert_map_manager
    if manager.global_num_experts != placement.experts or manager.ep_size != (
        placement.shards
    ):
        raise ValueError(
            f"expert placement {envs.TPU_EXPERT_PLACEMENT} is for "
            f"{placement.shards} ranks x {placement.experts} experts; "
            f"{name} has {manager.ep_size} x {manager.global_num_experts}"
        )
    if manager.num_fused_shared_experts or manager.expert_mask is not None:
        raise ValueError(
            f"expert placement: {name} fuses shared experts into its expert "
            "map, which a placement does not describe"
        )
    if (
        layer.moe_config.moe_parallel_config.enable_eplb
        or manager.placement_strategy != "linear"
    ):
        raise ValueError(
            f"expert placement: vLLM already places {name}'s experts "
            f"(EPLB or {manager.placement_strategy!r} placement)"
        )
    unsupported = _unsupported_routing(runner)
    if unsupported is not None:
        raise ValueError(
            f"expert placement: {name} {unsupported}, so renumbering its "
            "experts would change which ones its tokens select"
        )
    _target_layers()  # cached under the model config, for the loader's threads
    rank = manager.ep_rank
    held = ranks[rank]
    if len(held) != manager.local_num_experts:
        # A permutation keeps every rank's expert count, so only the map moves;
        # copy slots add to it, which install_copy_slot_hook gave the layer
        # before its weights were allocated.
        raise ValueError(
            f"expert placement: {name} holds {manager.local_num_experts} "
            f"experts on rank {rank}, the placement {len(held)}"
        )
    # Built on the host: the runner is built under the target device, where
    # the scatter would run on the chip and cost a host sync per layer.
    expert_map = torch.full((placement.experts,), -1, dtype=torch.int32, device="cpu")
    expert_map[list(held)] = torch.arange(len(held), dtype=torch.int32, device="cpu")
    # A single shard leaves no expert map to take the device from.
    device = (
        manager.expert_map.device
        if manager.expert_map is not None
        else torch.get_default_device()
    )
    manager._expert_map = expert_map.to(device)
    layer.update_expert_map_info()
    setattr(layer, EXPERT_ORDER_ATTR, placement.slot_order(layer_idx))
    if placement.copies_per_rank:
        # The copies follow the primaries; every kernel but the few-token one
        # serves the primaries alone.
        setattr(layer, PRIMARY_EXPERTS_ATTR, placement.per_rank)
    _APPLIED_LAYERS[layer_idx] = rank
    logger.info_once(
        "Expert placement %s applied: rank %d of %d, %d placed layers",
        envs.TPU_EXPERT_PLACEMENT,
        rank,
        placement.shards,
        len(placement.layers),
    )
    return True


# The layer prefix FusedMoEFactory is building, for the map manager it creates.
_FACTORY_PREFIX = ""


def install_copy_slot_hook() -> bool:
    """Size a copies table's MoE layers before vLLM allocates their weights.

    A table with copy slots gives every rank `slots_per_rank` slots, its
    primaries and then its copies, which is more than the checkpoint's
    experts / shards. vLLM allocates a MoE layer's expert weights inside
    `FusedMoEFactory`, sized by `ExpertMapManager.local_num_experts`, before
    the TPU MoE runner (where `apply` runs) exists, and vLLM 0.33 has no
    extension point that sees the layer's name before that allocation. So,
    for a table with copies only, this wraps the factory (by every name it is
    already bound to) to remember the layer's prefix, and
    `ExpertMapManager._calculate_expert_maps` to give a placed layer the
    table's slot count and its expert map of primaries and copies. A
    permutation-only table, or no table, installs nothing: `apply` serves it
    exactly as before. Returns whether the hook is installed.
    """
    placement = current()
    if placement is None or not placement.copies_per_rank:
        return False
    import inspect
    import sys

    from vllm.model_executor.layers import fused_moe as fused_pkg
    from vllm.model_executor.layers.fused_moe import expert_map_manager as emm
    from vllm.model_executor.layers.fused_moe import layer as fused_layer

    if getattr(fused_layer, "_tpu_copy_slot_hook", False):
        return True
    original_factory = fused_layer.FusedMoEFactory
    original_calc = emm.ExpertMapManager._calculate_expert_maps
    signature = inspect.signature(original_factory)

    def FusedMoEFactory(*args, **kwargs):
        global _FACTORY_PREFIX
        bound = signature.bind_partial(*args, **kwargs)
        _FACTORY_PREFIX = str(bound.arguments.get("prefix", "") or "")
        try:
            return original_factory(*args, **kwargs)
        finally:
            _FACTORY_PREFIX = ""

    def _calculate_expert_maps(self) -> None:
        original_calc(self)
        _size_copy_slots(self, _FACTORY_PREFIX)

    fused_layer.FusedMoEFactory = FusedMoEFactory
    fused_pkg.FusedMoEFactory = FusedMoEFactory
    name = "FusedMoEFactory"  # rebound in every model module that imported it
    for module in list(sys.modules.values()):
        if getattr(module, name, None) is original_factory:
            setattr(module, name, FusedMoEFactory)
    emm.ExpertMapManager._calculate_expert_maps = _calculate_expert_maps
    fused_layer._tpu_copy_slot_hook = True
    logger.info(
        "Expert placement %s has %d copy slots per rank: MoE layers it places "
        "are sized to %d slots before their weights are allocated.",
        envs.TPU_EXPERT_PLACEMENT,
        placement.copies_per_rank,
        placement.slots_per_rank,
    )
    return True


def patch_copy_slot_hook() -> None:
    """The patch registry's entry: install the hook when the table has copies
    (a no-op otherwise, at every stage)."""
    install_copy_slot_hook()


def _size_copy_slots(manager, prefix: str) -> None:
    """A placed layer's slot count and expert map (primaries, then copies),
    set where vLLM computes them; other layers keep vLLM's."""
    placement = current()
    if placement is None or not prefix or not manager.use_ep:
        return
    ranks = placed_ranks(prefix)
    if ranks is None:
        return
    if manager.ep_size != placement.shards or (
        manager.global_num_experts != placement.experts
    ):
        raise ValueError(
            f"expert placement {envs.TPU_EXPERT_PLACEMENT} is for "
            f"{placement.shards} ranks x {placement.experts} experts; {prefix} "
            f"has {manager.ep_size} x {manager.global_num_experts}"
        )
    held = ranks[manager.ep_rank]
    device = (
        manager._expert_map.device
        if manager._expert_map is not None
        else torch.device("cpu")
    )
    expert_map = torch.full((placement.experts,), -1, dtype=torch.int32)
    expert_map[list(held)] = torch.arange(len(held), dtype=torch.int32)
    manager._expert_map = expert_map.to(device)
    manager._local_num_experts = len(held) + manager.num_fused_shared_experts


@functools.cache
def _target_layers() -> int:
    from vllm.config import get_current_vllm_config

    # hf_text_config: a multimodal config (Qwen3.5-MoE, for example) keeps
    # num_hidden_layers on its text config.
    model_config = get_current_vllm_config().model_config
    return model_config.hf_text_config.num_hidden_layers


def expert_order(layer) -> tuple[int, ...] | None:
    """A placed layer's slot order (entry s: the expert in slot s), or None."""
    return getattr(layer, EXPERT_ORDER_ATTR, None)


def _router_tensors(runner) -> list[tuple[str, torch.Tensor]]:
    """The tensors that number a layer's experts, as (kind, tensor): "rows"
    for one indexed by expert along dim 0, "ids" for one holding expert ids."""
    layer = runner.routed_experts
    gate = runner.gate
    tensors = [("rows", gate.weight)]
    if getattr(gate, "bias", None) is not None:
        tensors.append(("rows", gate.bias))
    if layer.e_score_correction_bias is not None:
        tensors.append(("rows", layer.e_score_correction_bias))
    # The table the TPU MoE path routes hash layers with (moe_routing).
    if getattr(layer, "hash_indices_table", None) is not None:
        tensors.append(("ids", layer.hash_indices_table))
    # One entry per tensor, so none is permuted twice: a model may pass its
    # gate's own bias as the score bias (vLLM's K2 Horizon does).
    tensors = list({id(t): (kind, t) for kind, t in tensors}.values())
    known = {id(t) for _, t in tensors}
    other = [n for n, p in gate.named_parameters() if id(p) not in known]
    if other:
        raise ValueError(
            f"expert placement: {layer.layer_name}'s gate also holds {other}, "
            "which renumbering its experts does not cover"
        )
    experts = layer.global_num_experts
    shapes = [tuple(t.shape) for kind, t in tensors if kind == "rows"]
    if any(shape[0] != experts for shape in shapes):
        raise ValueError(
            f"expert placement: {layer.layer_name}'s router tensors {shapes} "
            f"are not all indexed by its {experts} experts"
        )
    return tensors


def permute_routers(model: torch.nn.Module) -> None:
    """Renumber the routers of `model`'s placed MoE layers into slot order.

    The model-loader patch calls this for every model a loader builds, the
    drafter included, once the checkpoint is written and before
    process_weights_after_loading stages anything from the router. Each
    layer's router is renumbered once: a layer already in slot order is left
    alone, so a second call cannot permute it twice.
    """
    if not _APPLIED_LAYERS:
        return
    renumbered = 0
    for module in model.modules():
        layer = getattr(module, "routed_experts", None)
        order = None if layer is None else expert_order(layer)
        if layer is None or order is None:
            continue
        if getattr(layer, ROUTER_RENUMBERED_ATTR, False):
            continue
        tensors = _router_tensors(module)
        rows = torch.tensor(order, dtype=torch.long)
        slots = torch.empty_like(rows)
        slots[rows] = torch.arange(len(order))
        for kind, tensor in tensors:
            host = tensor.data.to("cpu")
            permuted = host[rows] if kind == "rows" else slots[host.long()]
            # One whole-tensor copy: slice writes into a device tensor leak on
            # this backend (see _HostStagedParams in the DeepSeek-V4 model).
            tensor.data.copy_(permuted.to(host.dtype))
        setattr(layer, ROUTER_RENUMBERED_ATTR, True)
        index = layer_index(layer.layer_name)
        assert index is not None  # `apply` placed the layer by its index
        _PERMUTED_LAYERS.add(index)
        renumbered += 1
    if renumbered:
        logger.info(
            "Expert placement: renumbered the routers of %d placed MoE layers.",
            renumbered,
        )


def check_routers_permuted() -> None:
    """Refuse to serve a placed layer whose router was never renumbered: its
    experts sit in slot order, so its router would pick the wrong ones. Call
    once every model of the process has loaded."""
    missing = sorted(set(_APPLIED_LAYERS) - _PERMUTED_LAYERS)
    if missing:
        raise RuntimeError(
            f"TPU_EXPERT_PLACEMENT={envs.TPU_EXPERT_PLACEMENT} placed the "
            f"experts of MoE layers {missing}, but their routers were never "
            "renumbered: the model was loaded without passing through "
            "vLLM's process_weights_after_loading, where the model-loader "
            "patch calls expert_placement.permute_routers"
        )
    placement = current()
    unused = sorted(set(placement.layers) - set(_APPLIED_LAYERS)) if placement else []
    if unused:
        logger.warning(
            "TPU_EXPERT_PLACEMENT=%s lists layers %s, which have no MoE layer "
            "in this process: another pipeline stage's, or a table built for "
            "another model",
            envs.TPU_EXPERT_PLACEMENT,
            unused,
        )
