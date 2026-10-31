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

A listed layer is a PERMUTATION of its experts: every rank lists experts /
shards of them, and every expert appears exactly once. Placement only moves
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
groups of them, as in DeepSeek-V4, GLM-5 and Qwen3-MoE.

Every MoE kernel then sees an ordinary layer whose experts sit in contiguous
blocks, so no kernel and no compiled program changes. The router computes the
same scores in slot order, so tokens select the same experts with the same
weights. Only an exact tie between two experts' selection scores can resolve
differently, since top-k breaks ties by index. Such ties are common enough on
DeepSeek-V4, whose router logits are bf16 here, that greedy outputs drift from
the unplaced model's within a few dozen tokens; a tie carries no router
preference, so quality does not move.
"""

import functools
import json
import os
import re
import subprocess
from dataclasses import dataclass

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# `model.layers.7.ffn.experts` (a module prefix) and `layers.7.ffn.experts.3.w1`
# (a checkpoint tensor) both name layer 7.
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
# A DSpark drafter's MoE layers are modules `layers.{n + i}` of an n-layer
# target, built from the checkpoint's `mtp.{i}.*` tensors.
_DRAFTER_TENSOR_RE = re.compile(r"^mtp\.(\d+)\.")

# The attribute a placed RoutedExperts layer holds its slot order under:
# entry s is the checkpoint's id of the expert in slot s.
EXPERT_ORDER_ATTR = "_tpu_expert_order"
# Set on a placed RoutedExperts layer once its router is in slot order.
ROUTER_RENUMBERED_ATTR = "_tpu_router_renumbered"

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

    def ranks(self, layer: int) -> tuple[tuple[int, ...], ...] | None:
        """Per rank, the global experts it holds in slot order; None if unlisted."""
        return self.layers.get(layer)

    def slot_order(self, layer: int) -> tuple[int, ...]:
        """Entry s: the expert in slot s, r * (experts / shards) + j."""
        return tuple(e for held in self.layers[layer] for e in held)

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
    if experts % shards:
        raise ValueError(
            f"expert placement {source}: {experts} experts do not divide over "
            f"{shards} ranks"
        )
    per_rank = experts // shards
    layers = {}
    for key, ranks in table["layers"].items():
        if len(ranks) != shards or any(len(held) != per_rank for held in ranks):
            raise ValueError(
                f"expert placement {source}: layer {key} must list {shards} "
                f"ranks of {per_rank} experts each"
            )
        ranks = tuple(tuple(int(e) for e in held) for held in ranks)
        if sorted(e for held in ranks for e in held) != list(range(experts)):
            raise ValueError(
                f"expert placement {source}: layer {key} must hold each of the "
                f"{experts} experts exactly once"
            )
        layers[int(key)] = ranks
    return Placement(experts=experts, shards=shards, layers=layers)


def _read(uri: str) -> str:
    if uri.startswith("gs://"):
        try:
            return subprocess.run(
                ["gcloud", "storage", "cat", uri],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except FileNotFoundError as e:
            raise RuntimeError(
                f"TPU_EXPERT_PLACEMENT={uri} is read with the gcloud CLI, which "
                "this image does not have; copy the table to a local path or "
                "name a shipped placement"
            ) from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"TPU_EXPERT_PLACEMENT={uri}: gcloud storage cat failed: "
                f"{e.stderr.strip()}"
            ) from e
    with open(uri) as f:
        return f.read()


@functools.cache
def _load(path: str) -> Placement:
    """The table, read once per process: the expert maps, the fetch filter and
    the routers all follow the same read, even if the file is rewritten."""
    return parse(json.loads(_read(path)), path)


def clear_cache() -> None:
    """Forget every table read so far, for a caller that rewrites one."""
    _load.cache_clear()


# Placements shipped with the package, by name (a file stem in this directory).
PLACEMENT_DIR = os.path.join(os.path.dirname(__file__), "expert_placements")


def resolve(value: str) -> str:
    """A path or gs:// URI as given, or a shipped placement's name -> its file."""
    if value.startswith("gs://") or os.sep in value or value.endswith(".json"):
        return value
    path = os.path.join(PLACEMENT_DIR, f"{value}.json")
    if not os.path.exists(path):
        shipped = sorted(f[:-5] for f in os.listdir(PLACEMENT_DIR))
        raise FileNotFoundError(
            f"TPU_EXPERT_PLACEMENT={value!r} is neither a path nor a shipped "
            f"placement {shipped}"
        )
    return path


def current() -> Placement | None:
    """The process's placement, or None when TPU_EXPERT_PLACEMENT is unset."""
    value = envs.TPU_EXPERT_PLACEMENT
    return _load(resolve(value)) if value else None


def layer_index(name: str) -> int | None:
    """The decoder layer a module prefix or checkpoint tensor name belongs to."""
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


def fetch_filter_ids(name: str) -> frozenset[int] | None:
    """The experts the fetch-time filter keeps for this tensor's layer.

    None defers to the default contiguous filter: either there is no
    placement for the layer, or it was not applied to it. Otherwise the
    experts of the EP rank `apply` built the layer's expert map for, so the
    fetch and the expert map cannot pick different ranks.
    """
    layer = layer_index(name)
    drafter = _DRAFTER_TENSOR_RE.match(name)
    if drafter is not None:
        layer = _target_layers() + int(drafter.group(1))
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
        # A permutation keeps every rank's expert count, so only the map moves.
        raise ValueError(
            f"expert placement: {name} holds {manager.local_num_experts} "
            f"experts on rank {rank}, the placement {len(held)}"
        )
    expert_map = torch.full((placement.experts,), -1, dtype=torch.int32)
    expert_map[list(held)] = torch.arange(len(held), dtype=torch.int32)
    manager._expert_map = expert_map.to(manager.expert_map.device)
    layer.update_expert_map_info()
    setattr(layer, EXPERT_ORDER_ATTR, placement.slot_order(layer_idx))
    _APPLIED_LAYERS[layer_idx] = rank
    logger.info_once(
        "Expert placement %s applied: rank %d of %d, %d placed layers",
        envs.TPU_EXPERT_PLACEMENT,
        rank,
        placement.shards,
        len(placement.layers),
    )
    return True


@functools.cache
def _target_layers() -> int:
    from vllm.config import get_current_vllm_config

    return get_current_vllm_config().model_config.hf_config.num_hidden_layers


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
