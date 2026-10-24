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

What follows from it, all decided at load:
  * the layer's expert map puts expert `ranks[r][j]` in slot j of rank r, so
    vLLM's weight loader and the fetch-time expert filter load exactly those;
  * `expert_slot_table` gives, per global expert, the SLOT holding it,
    `r * (experts / shards) + j`, the contiguous-block numbering the fused EP
    kernel addresses. The kernel relabels the selected ids after top-k, so
    selection, tie-breaking, the router weights and the order of the k
    combine terms are those of the unplaced model: per token the output is
    the same bits; only where the matmuls run moves.

Only the fused EP kernel reads slot ids, so a placed layer must be armed for
it; `check_layer_served` refuses anything else.
"""

import functools
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# `model.layers.7.ffn.experts` (a module prefix) and `layers.7.ffn.experts.3.w1`
# (a checkpoint tensor) both name layer 7.
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
# A DSpark drafter's MoE layers are modules `layers.{n + i}` of an n-layer
# target, built from the checkpoint's `mtp.{i}.*` tensors.
_DRAFTER_TENSOR_RE = re.compile(r"^mtp\.(\d+)\.")

# The layer attribute the slot table is registered under.
EXPERT_SLOT_ATTR = "_tpu_expert_slot"

# Layer index -> the EP rank whose slots this process's expert map holds, for
# every layer whose expert map follows the placement. The fetch-time filter
# consults the placement only for these, and at exactly that rank, so it keeps
# what the expert map loads by construction (under hierarchical EP the rank is
# the chip's, not the one the dp/pcp/tp formula gives), and a model whose MoE
# method never applied the placement keeps the default filter.
_APPLIED_LAYERS: dict[int, int] = {}


@dataclass(frozen=True)
class Placement:
    experts: int
    shards: int
    layers: dict[int, tuple[tuple[int, ...], ...]]

    def ranks(self, layer: int) -> tuple[tuple[int, ...], ...] | None:
        """Per rank, the global experts it holds in slot order; None if unlisted."""
        return self.layers.get(layer)

    def slot_table(self, layer: int) -> list[int]:
        """Entry e: the slot r * (experts / shards) + j holding expert e."""
        per_rank = self.experts // self.shards
        slot = [0] * self.experts
        for r, held in enumerate(self.layers[layer]):
            for j, e in enumerate(held):
                slot[e] = r * per_rank + j
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
def _text(uri: str) -> str:
    """The table's text, read once per process: the compile cache key and the
    table the layers load come from the same read, so a file rewritten between
    the two cannot key one placement and serve another."""
    return _read(uri)


@functools.cache
def _digest(uri: str) -> str:
    return hashlib.sha256(_text(uri).encode()).hexdigest()


@functools.cache
def _load(path: str) -> Placement:
    return parse(json.loads(_text(path)), path)


def clear_cache() -> None:
    """Forget every table read so far, for a caller that rewrites one."""
    _text.cache_clear()
    _digest.cache_clear()
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


def content_hash() -> str | None:
    """The placement file's content digest for the compile cache key."""
    value = envs.TPU_EXPERT_PLACEMENT
    return _digest(resolve(value)) if value else None


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
    placement for the layer, or no MoE method applied one to it. Otherwise
    the experts of the EP rank `apply_to_layer` built the layer's expert map
    for, so the fetch and the expert map cannot pick different ranks.
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


def apply_to_layer(layer) -> bool:
    """Point a RoutedExperts layer's expert map at its placement.

    Must run before the weights load (from the MoE method's create_weights).
    Registers the slot table the fused EP op takes. Returns whether the layer
    is placed.
    """
    import torch

    placement = current()
    layer_idx = layer_index(layer.layer_name)
    ranks = placed_ranks(layer.layer_name)
    if placement is None or layer_idx is None or ranks is None:
        return False
    if not envs.USE_MOE_FUSED_EP_KERNEL:
        # check_layer_served would refuse this layer anyway, but only after
        # the whole checkpoint has loaded; the flag is known now.
        raise RuntimeError(
            f"TPU_EXPERT_PLACEMENT={envs.TPU_EXPERT_PLACEMENT} places "
            f"{layer.layer_name}'s experts for the fused EP MoE kernel, which "
            "USE_MOE_FUSED_EP_KERNEL does not enable"
        )
    manager = layer.expert_map_manager
    if manager.global_num_experts != placement.experts or manager.ep_size != (
        placement.shards
    ):
        raise ValueError(
            f"expert placement {envs.TPU_EXPERT_PLACEMENT} is for "
            f"{placement.shards} ranks x {placement.experts} experts; "
            f"{layer.layer_name} has {manager.ep_size} x "
            f"{manager.global_num_experts}"
        )
    if manager.num_fused_shared_experts or manager.expert_mask is not None:
        raise ValueError(
            f"expert placement: {layer.layer_name} fuses shared experts into "
            "its expert map, which a placement does not describe"
        )
    _target_layers()  # cached under the model config, for the loader's threads
    rank = manager.ep_rank
    held = ranks[rank]
    if len(held) != manager.local_num_experts:
        # A permutation keeps every rank's expert count, so only the map moves.
        raise ValueError(
            f"expert placement: {layer.layer_name} holds "
            f"{manager.local_num_experts} experts on rank {rank}, the placement "
            f"{len(held)}"
        )
    expert_map = torch.full((placement.experts,), -1, dtype=torch.int32)
    expert_map[list(held)] = torch.arange(len(held), dtype=torch.int32)
    manager._expert_map = expert_map.to(manager.expert_map.device)
    layer.update_expert_map_info()
    layer.register_buffer(
        EXPERT_SLOT_ATTR,
        torch.tensor(
            placement.slot_table(layer_idx),
            dtype=torch.int32,
            device=manager.expert_map.device,
        ),
        persistent=False,
    )
    _APPLIED_LAYERS[layer_idx] = rank
    logger.info_once(
        "Expert placement %s applied: rank %d of %d, %d placed layers; "
        "first placed layer holds %s",
        envs.TPU_EXPERT_PLACEMENT,
        rank,
        placement.shards,
        len(placement.layers),
        ranks[rank][:6],
    )
    return True


@functools.cache
def _target_layers() -> int:
    from vllm.config import get_current_vllm_config

    return get_current_vllm_config().model_config.hf_config.num_hidden_layers


def expert_slot_table(layer):
    """The [experts] i32 slot table of a placed layer, or None."""
    return getattr(layer, EXPERT_SLOT_ATTR, None)


def check_layer_served(layer, fused_armed: bool) -> None:
    """A placed layer's weights sit in slot order only the fused op undoes."""
    if expert_slot_table(layer) is not None and not fused_armed:
        raise RuntimeError(
            f"expert placement covers {layer.layer_name}, but the fused EP "
            "MoE kernel is not armed for it; the grouped-matmul path reads "
            "contiguous expert blocks and would run the wrong experts"
        )
