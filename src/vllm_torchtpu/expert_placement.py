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
"""Static per-layer expert placement for expert parallelism.

``TPU_EXPERT_PLACEMENT`` names a JSON table (a path or a ``gs://`` URI) that
lists, per MoE layer, the global experts each EP rank holds in slot order. The
first ``owned_per_core`` slots of the ranks hold every expert once (the
primaries); with ``"copies": true`` the slots after them hold one extra copy of
some experts, served by the small-batch MoE kernel.

A placed layer's expert map, the EP weight filter and the router's rows follow
the table. The router rows are reordered so that rank r's primaries become
expert ids ``r * owned .. (r + 1) * owned - 1``, so every kernel keeps its
contiguous-block ownership and nothing runs per token. Layers the table does
not list, and every layer under an ``mtp`` module (a draft model, whose layer
numbers restart at 0), keep vLLM's linear placement.
"""

import dataclasses
import functools
import hashlib
import json
import re
import subprocess
from typing import NamedTuple

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_ROUTER_RE = re.compile(r"\.mlp\.gate\.(weight|bias)$|e_score_correction_bias$")
# Load formats served by the two loaders whose weight streams are reordered.
_STREAMED_LOAD_FORMATS = (
    "auto",
    "hf",
    "safetensors",
    "fastsafetensors",
    "instanttensor",
    "mistral",
    "npcache",
    "pt",
    "runai_streamer",
)
# Per placed layer, the experts this rank holds; recorded with its expert map.
_LOCAL_IDS: dict[int, frozenset[int]] = {}


@dataclasses.dataclass(frozen=True)
class Placement:
    experts: int
    ranks: int
    owned: int
    slots: int
    # Per layer and rank, the global expert in each slot.
    cores: dict[int, tuple[tuple[int, ...], ...]]
    # Per layer, the copied experts, hottest first.
    hot: dict[int, tuple[int, ...]]


class CopyLayout(NamedTuple):
    """A layer's copies on one rank, in router-row (virtual) expert ids."""

    gid: list[int]  # the expert in each local slot: primaries, then copies
    copy: list[int]  # per expert, the rank holding its copy, -1 without one
    hot: list[int]  # the copied experts, hottest first


def parse(table: dict, source: str) -> Placement:
    """Validate a placement table: primaries a permutation, copies unique."""
    experts = int(table["experts"])
    ranks = int(table["shards"])
    slots = int(table["slots_per_core"])
    copies = bool(table.get("copies", False))
    owned = int(table["owned_per_core"]) if copies else slots
    if ranks * owned != experts or owned > slots:
        raise ValueError(
            f"expert placement {source}: {ranks} ranks x {owned} primaries must "
            f"equal {experts} experts, with at most {slots} slots per rank"
        )
    cores, hot = {}, {}
    for key, entry in table["layers"].items():
        layer = int(key)
        rows = tuple(tuple(int(e) for e in c) for c in entry["cores"])
        if len(rows) != ranks or any(len(c) != slots for c in rows):
            raise ValueError(
                f"expert placement {source}: layer {layer} must list {ranks} "
                f"ranks x {slots} slots"
            )
        if sorted(e for c in rows for e in c[:owned]) != list(range(experts)):
            raise ValueError(
                f"expert placement {source}: layer {layer} must hold every "
                "expert exactly once among the primaries"
            )
        copied = {e: r for r, c in enumerate(rows) for e in c[owned:]}
        if any(not 0 <= e < experts for e in copied):
            raise ValueError(
                f"expert placement {source}: layer {layer} has an invalid copy id"
            )
        if len(copied) != ranks * (slots - owned):
            raise ValueError(
                f"expert placement {source}: layer {layer} holds an expert in "
                "more than one copy slot"
            )
        if any(e in rows[r][:owned] for e, r in copied.items()):
            raise ValueError(
                f"expert placement {source}: layer {layer} holds a copy on the "
                "rank that owns the expert"
            )
        declared = entry.get("copies")
        if declared is not None and {int(e): r for e, r in declared.items()} != copied:
            raise ValueError(
                f"expert placement {source}: layer {layer}'s copies must match "
                "its copy slots"
            )
        cores[layer] = rows
        hot[layer] = tuple(int(e) for e in declared) if declared else tuple(copied)
    return Placement(experts, ranks, owned, slots, cores, hot)


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
def _digest(uri: str) -> str:
    return hashlib.sha256(_read(uri).encode()).hexdigest()


def content_hash() -> str | None:
    """The placement file's content digest for the compile cache key."""
    uri = envs.TPU_EXPERT_PLACEMENT
    return _digest(uri) if uri else None


@functools.cache
def _load(uri: str) -> Placement:
    table = parse(json.loads(_read(uri)), uri)
    logger.info(
        "Expert placement %s: %d layers, %d ranks x %d slots (%d primaries)",
        uri,
        len(table.cores),
        table.ranks,
        table.slots,
        table.owned,
    )
    return table


def placement() -> Placement | None:
    uri = envs.TPU_EXPERT_PLACEMENT
    return _load(uri) if uri else None


def layer_index(name: str) -> int | None:
    """The placed layer a module or parameter name belongs to, or None."""
    table = placement()
    match = _LAYER_RE.search(name)
    if table is None or match is None or "mtp" in name.split("."):
        return None
    layer = int(match.group(1))
    return layer if layer in table.cores else None


def expert_map(table: Placement, layer: int, rank: int) -> list[int]:
    """vLLM's expert map: each global expert's local slot, -1 if absent."""
    local = [-1] * table.experts
    for slot, expert in enumerate(table.cores[layer][rank]):
        local[expert] = slot
    return local


def virtual_order(table: Placement, layer: int) -> list[int]:
    """The global expert behind each router row after the reorder."""
    return [e for c in table.cores[layer] for e in c[: table.owned]]


def local_expert_ids(name: str) -> frozenset[int] | None:
    """The experts this rank loads for a placed layer's weight, else None."""
    layer = layer_index(name) if _LOCAL_IDS else None
    return None if layer is None else _LOCAL_IDS.get(layer)


def num_primary_experts(layer_name: str) -> int | None:
    """Primaries per rank of a layer with copy slots, else None (all slots)."""
    table = placement()
    if table is None or table.slots == table.owned or layer_index(layer_name) is None:
        return None
    return table.owned


def copy_layout(layer_name: str, rank: int) -> CopyLayout | None:
    """The copies of a placed layer on `rank`, or None without copies."""
    if num_primary_experts(layer_name) is None:
        return None
    table = placement()
    layer = layer_index(layer_name)
    virtual = {e: v for v, e in enumerate(virtual_order(table, layer))}
    copy = [-1] * table.experts
    for r, c in enumerate(table.cores[layer]):
        for e in c[table.owned :]:
            copy[virtual[e]] = r
    return CopyLayout(
        gid=[virtual[e] for e in table.cores[layer][rank]],
        copy=copy,
        hot=[virtual[e] for e in table.hot[layer]],
    )


def permute_router_rows(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """A placed layer's router weight or bias in the reordered row order."""
    layer = layer_index(name) if _ROUTER_RE.search(name) else None
    if layer is None:
        return tensor
    table = placement()
    if tensor.shape[0] != table.experts:
        raise ValueError(
            f"expert placement: {name} has {tensor.shape[0]} rows, the table "
            f"{table.experts} experts"
        )
    order = torch.tensor(virtual_order(table, layer), device=tensor.device)
    return tensor.index_select(0, order)


def _refuse_unsupported(layer) -> None:
    from vllm.config import get_current_vllm_config

    load_format = get_current_vllm_config().load_config.load_format
    if load_format not in _STREAMED_LOAD_FORMATS:
        raise NotImplementedError(
            "TPU_EXPERT_PLACEMENT reorders the router rows in the default and "
            f"Run:AI loaders' weight streams, which --load-format {load_format} "
            "does not use. Load with --load-format auto or runai_streamer, or "
            "unset TPU_EXPERT_PLACEMENT."
        )
    if num_primary_experts(layer.layer_name) is None:
        return
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8Config

    if envs.USE_MOE_FUSED_EP_KERNEL:
        raise NotImplementedError(
            "an expert placement with copies is served by the grouped-matmul "
            "and small-batch MoE paths, not the fused EP kernel. Unset "
            "MOE_FUSED_EP_KERNEL_IMPL / USE_MOE_FUSED_EP_KERNEL, or use a "
            "placement without copies."
        )
    if not isinstance(layer.quant_config, VllmFp8Config):
        raise NotImplementedError(
            f"{layer.layer_name}: an expert placement with copies needs the FP8 "
            "MoE method, which bounds the grouped path to the primaries. Use a "
            "placement without copies for this checkpoint."
        )


def install_expert_placement() -> None:
    """Apply TPU_EXPERT_PLACEMENT to MoE layers as vLLM builds them.

    Each placed ``RoutedExperts`` rebuilds its expert map through
    ``ExpertMapManager.update`` (main's host-built map), with
    ``determine_expert_map`` answering from the table, before its weights are
    created. Both loaders' weight streams reorder the router rows.
    """
    if not envs.TPU_EXPERT_PLACEMENT:
        return
    from vllm.model_executor.layers.fused_moe import expert_map_manager
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.model_loader import default_loader, runai_streamer_loader

    if getattr(RoutedExperts, "_tpu_expert_placement_patch", False):
        return
    determine = expert_map_manager.determine_expert_map
    update_info = RoutedExperts.update_expert_map_info
    get_all_weights = default_loader.DefaultModelLoader.get_all_weights
    streamed = runai_streamer_loader.RunaiModelStreamerLoader._get_weights_iterator
    building: list[int] = []

    def determine_expert_map(
        ep_size,
        ep_rank,
        global_num_experts,
        expert_placement_strategy="linear",
        num_fused_shared_experts=0,
        return_expert_mask=False,
    ):
        if not building:
            return determine(
                ep_size,
                ep_rank,
                global_num_experts,
                expert_placement_strategy,
                num_fused_shared_experts,
                return_expert_mask,
            )
        table, layer = placement(), building[-1]
        if (ep_size, global_num_experts) != (table.ranks, table.experts):
            raise ValueError(
                f"expert placement: the table is for {table.ranks} ranks x "
                f"{table.experts} experts; layer {layer} has {ep_size} ranks x "
                f"{global_num_experts} experts"
            )
        if num_fused_shared_experts or return_expert_mask:
            raise NotImplementedError(
                "expert placement does not support fused shared experts"
            )
        local = expert_map(table, layer, ep_rank)
        return table.slots, torch.tensor(local, dtype=torch.int32, device="cpu"), None

    def update_expert_map_info(self) -> None:
        use_ep = self.moe_config.moe_parallel_config.use_ep
        layer = layer_index(self.layer_name) if use_ep else None
        if layer is not None:
            _refuse_unsupported(self)
            manager = self.expert_map_manager
            building.append(layer)
            try:
                manager.update(manager.moe_parallel_config, manager.global_num_experts)
            finally:
                building.pop()
            self.moe_config.num_local_experts = manager.local_num_experts
            _LOCAL_IDS[layer] = frozenset(placement().cores[layer][manager.ep_rank])
        elif use_ep:
            logger.info(
                "Expert placement: %s keeps vLLM's linear placement", self.layer_name
            )
        update_info(self)

    def placed_all_weights(self, model_config, model):
        for name, tensor in get_all_weights(self, model_config, model):
            yield name, permute_router_rows(name, tensor)

    def placed_weights_iterator(self, model_or_path, revision):
        for name, tensor in streamed(self, model_or_path, revision):
            yield name, permute_router_rows(name, tensor)

    expert_map_manager.determine_expert_map = determine_expert_map
    RoutedExperts.update_expert_map_info = update_expert_map_info
    default_loader.DefaultModelLoader.get_all_weights = placed_all_weights
    runai_streamer_loader.RunaiModelStreamerLoader._get_weights_iterator = (
        placed_weights_iterator
    )
    RoutedExperts._tpu_expert_placement_patch = True
    logger.info(
        "Applied TPU patch: static expert placement from %s", envs.TPU_EXPERT_PLACEMENT
    )
