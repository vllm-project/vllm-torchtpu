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
"""TPU_EXPERT_PLACEMENT=<json>: a static per-layer expert placement for
expert parallelism, applied as a permutation of the checkpoint at load time.

The file says, per MoE layer, which global experts each expert-parallel rank
holds and in what slot order: ``layers[<n>].cores[r]`` is a list of
``slots_per_core`` global expert ids, and together the ranks hold every
expert exactly once. vLLM's EPLB policy
(``DefaultEplbPolicy.rebalance_experts``) run on a routing recording
produces such a file. Off by default (no file): the checkpoint's own
contiguous placement, nothing touched.

What it changes, and where (all at model construction or weight load;
nothing runs per token):

* the layer's ``expert_map`` (vLLM's global-to-local slot map, used by the
  MoE weight loader) is this rank's slot index of each global expert, so
  expert ``cores[r][j]`` lands in local slot ``j``;
* the loader's EP weight filter keeps the experts the placement gives this
  rank (``model_loader_patches._should_skip``);
* the router's rows (one per expert) are reordered at load into the virtual
  order ``v = r * slots + j``, so the router's top-k already names
  (rank, slot) in the contiguous-block layout every MoE path on this
  package assumes (``moe_routing.get_experts_start``, the routed and dense
  GMM paths, the fused and adaptive EP kernels).

Numerics: each token selects the same experts with the same weights and
runs the same expert weights; only the order in which the k expert
contributions are summed can differ (rounding level).

Tables with copy slots (an expert held by two ranks) need a kernel that
picks one holder per expert; none on this package does, so they are
refused.
"""

import functools
import json
import re

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_DATA: dict | None = None
_DATA_PATH = ""
_LOGGED: set[str] = set()
_LAYER_RE = re.compile(r"layers\.(\d+)\.")
# The MTP draft's MoE layer (``mtp.layers.<i>``) keeps the default placement:
# its checkpoint names and module prefixes both carry ``mtp.``.
_MTP_RE = re.compile(r"(^|\.)mtp\.")
_ROUTER_RE = re.compile(r"\.mlp\.gate\.(weight|bias)$|e_score_correction_bias$")
_QUANTIZED_ROUTER_RE = re.compile(r"\.mlp\.gate\.(weight_scale\w*|input_scale)$")
# The EP rank the expert maps were built for; the load-time expert filter
# reuses it so the two agree by construction.
_EP_RANK: int | None = None
_CURRENT_PREFIX = ""


def _path() -> str:
    return envs.TPU_EXPERT_PLACEMENT


def enabled() -> bool:
    """Whether a placement table is set (TPU_EXPERT_PLACEMENT)."""
    return bool(_path())


def _log_once(key: str, message: str) -> None:
    """One line per process per event kind, so a boot log shows the
    placement was applied."""
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    logger.info(message)


def data() -> dict:
    """The placement table, loaded once per path and checked to give every
    expert exactly one slot per layer."""
    global _DATA, _DATA_PATH
    path = _path()
    if _DATA is None or path != _DATA_PATH:
        with open(path) as f:
            table = json.load(f)
        experts = int(table["experts"])
        ranks = int(table["shards"])
        per_rank = int(table["slots_per_core"])
        if table.get("copies") or int(table.get("owned_per_core", per_rank)) != (
            per_rank
        ):
            raise ValueError(
                f"expert placement {path}: copy slots are not supported; "
                "use a table where each rank holds slots_per_core primaries"
            )
        if ranks * per_rank != experts:
            raise ValueError(
                f"expert placement {path}: {ranks} ranks x {per_rank} slots "
                f"must equal {experts} experts"
            )
        for n, layer in table["layers"].items():
            cores = layer["cores"]
            if len(cores) != ranks or any(len(c) != per_rank for c in cores):
                raise ValueError(
                    f"expert placement {path}: layer {n} must list {ranks} "
                    f"ranks x {per_rank} slots"
                )
            if sorted(e for c in cores for e in c) != list(range(experts)):
                raise ValueError(
                    f"expert placement {path}: layer {n} must hold every "
                    "expert exactly once"
                )
        _DATA, _DATA_PATH = table, path
    return _DATA


def layer_index(name: str) -> int | None:
    """The placed MoE layer index in a parameter name or module prefix, or
    None (no layer index, or an MTP draft layer)."""
    if _MTP_RE.search(name):
        return None
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def shards() -> int:
    """The number of expert-parallel ranks the table was built for."""
    return int(data()["shards"])


def slots() -> int:
    """The number of expert slots each rank holds."""
    return int(data()["slots_per_core"])


def global_count() -> int:
    """The number of global experts per layer."""
    return int(data()["experts"])


def cores(layer: int) -> list[list[int]]:
    """Per rank, the global experts it holds for the layer, in slot order."""
    return data()["layers"][str(layer)]["cores"]


def has_layer(layer: int) -> bool:
    return str(layer) in data()["layers"]


@functools.cache
def _local_ids(path: str, layer: int, rank: int) -> frozenset[int]:
    del path  # part of the key: a new table gives new sets
    return frozenset(cores(layer)[rank])


def local_ids(layer: int, rank: int) -> frozenset[int]:
    """The global experts rank ``rank`` holds for this layer."""
    return _local_ids(_path(), layer, rank)


def expert_map(layer: int, rank: int) -> list[int]:
    """vLLM's expert_map for this layer on this rank: the local slot of each
    global expert, -1 when absent."""
    m = [-1] * global_count()
    for j, e in enumerate(cores(layer)[rank]):
        m[e] = j
    return m


def virtual_order(layer: int) -> list[int]:
    """The global expert of each virtual id ``v = r * slots + j`` (the
    router's row order after the permutation)."""
    return [e for c in cores(layer) for e in c]


def permute_router_rows(name: str, tensor):
    """The router's weight (rows = experts) or per-expert bias reordered into
    the virtual order at load. Returns the tensor unchanged when the name is
    not a router parameter of a placed layer."""
    if not enabled():
        return tensor
    if _QUANTIZED_ROUTER_RE.search(name) and layer_index(name) is not None:
        raise NotImplementedError(
            f"expert placement: quantized router {name} is not supported"
        )
    if not _ROUTER_RE.search(name):
        return tensor
    layer = layer_index(name)
    if layer is None or not has_layer(layer):
        return tensor
    if tensor.shape[0] != global_count():
        raise ValueError(
            f"expert placement: {name} has {tensor.shape[0]} rows, the table "
            f"{global_count()} experts"
        )
    import torch

    _log_once(
        "router",
        "[expert placement] router rows reordered into the virtual order at "
        f"load: {name} {tuple(tensor.shape)}",
    )
    order = torch.tensor(virtual_order(layer), dtype=torch.long, device=tensor.device)
    return tensor[order].contiguous()


def skip_expert_weight(name: str, expert_id: int) -> bool | None:
    """For the EP weight filter: whether this rank skips a per-expert tensor
    under the placement, or None when the placement does not decide (off,
    no placed layer in the name)."""
    if not enabled():
        return None
    layer = layer_index(name)
    if layer is None or not has_layer(layer):
        return None
    if _EP_RANK is None:
        raise RuntimeError(
            "expert placement: weights are filtered before any MoE layer "
            "built its expert map, so this rank's EP rank is unknown"
        )
    return expert_id not in local_ids(layer, _EP_RANK)


def _apply_to_expert_map_manager(manager) -> None:
    """Replace a layer's linear expert map with the placement's, after the
    original ``_calculate_expert_maps`` ran."""
    global _EP_RANK
    import torch

    if not manager.moe_parallel_config.use_ep:
        raise ValueError(
            "expert placement: TPU_EXPERT_PLACEMENT needs --enable-expert-parallel"
        )
    if not _CURRENT_PREFIX:
        raise RuntimeError(
            "expert placement: an expert map is computed outside the layer "
            "factory (no layer prefix), so the placement cannot be applied"
        )
    n = layer_index(_CURRENT_PREFIX)
    if n is None:
        return  # an MTP draft layer: default placement
    if not has_layer(n):
        raise RuntimeError(
            f"expert placement: the table has no layer {n} (prefix {_CURRENT_PREFIX})"
        )
    if shards() != int(manager.ep_size) or global_count() != int(
        manager.global_num_experts
    ):
        raise ValueError(
            f"expert placement: the table is for {shards()} ranks x "
            f"{global_count()} experts; this layer has {manager.ep_size} "
            f"ranks x {manager.global_num_experts}"
        )
    if int(manager.num_fused_shared_experts):
        raise NotImplementedError(
            "expert placement: fused shared experts are not supported"
        )
    rank = int(manager.ep_rank)
    if _EP_RANK is not None and rank != _EP_RANK:
        raise RuntimeError(
            f"expert placement: EP rank changed from {_EP_RANK} to {rank}"
        )
    _EP_RANK = rank
    device = (
        manager._expert_map.device
        if manager._expert_map is not None
        else torch.device("cpu")
    )
    manager._local_num_experts = slots()
    manager._expert_map = torch.tensor(expert_map(n, rank), dtype=torch.int32).to(
        device
    )
    if manager._expert_mask is not None:
        manager._expert_mask = (manager._expert_map >= 0).to(manager._expert_mask.dtype)
    _log_once(
        "map",
        f"[expert placement] {_path()}: layer {n} rank {rank} holds {slots()} "
        f"slots; first slots {cores(n)[rank][:4]}",
    )


def install_model_patches() -> None:
    """Give every MoE layer the placement's expert map for its layer, by
    wrapping vLLM's layer factory (which knows the layer's prefix) and the
    map computation it does inside, and reorder the router rows in both
    loaders' weight streams. A no-op without TPU_EXPERT_PLACEMENT.

    Must run after the host expert-map patch and the loader patches, so it
    wraps their versions.
    """
    if not enabled():
        return
    if envs.TPU_MOE_HIERARCHICAL_EP:
        raise ValueError(
            "expert placement: TPU_EXPERT_PLACEMENT and TPU_MOE_HIERARCHICAL_EP "
            "are mutually exclusive"
        )
    data()  # fail at startup on a bad table
    import inspect
    import sys

    from vllm.model_executor.layers import fused_moe as fused_pkg
    from vllm.model_executor.layers.fused_moe import expert_map_manager as emm
    from vllm.model_executor.layers.fused_moe import layer as fused_layer

    if getattr(fused_layer, "_tpu_placement_patch", False):
        return
    original_factory = getattr(fused_layer, "FusedMoEFactory", None)
    if original_factory is None or "prefix" not in (
        inspect.signature(original_factory).parameters
    ):
        raise RuntimeError(
            "expert placement: this vLLM has no FusedMoEFactory(prefix=...)"
        )
    original_calc = emm.ExpertMapManager._calculate_expert_maps
    signature = inspect.signature(original_factory)

    @functools.wraps(original_factory)
    def FusedMoEFactory(*args, **kwargs):  # noqa: N802 - vLLM's name
        global _CURRENT_PREFIX
        bound = signature.bind_partial(*args, **kwargs)
        _CURRENT_PREFIX = str(bound.arguments.get("prefix", "") or "")
        try:
            return original_factory(*args, **kwargs)
        finally:
            _CURRENT_PREFIX = ""

    def _calculate_expert_maps(self) -> None:
        original_calc(self)
        _apply_to_expert_map_manager(self)

    # The factory by every name it is already bound to (the package export
    # and any model module that imported it).
    for mod in list(sys.modules.values()):
        if getattr(mod, "FusedMoEFactory", None) is original_factory:
            mod.FusedMoEFactory = FusedMoEFactory
    fused_layer.FusedMoEFactory = FusedMoEFactory
    fused_pkg.FusedMoEFactory = FusedMoEFactory
    emm.ExpertMapManager._calculate_expert_maps = _calculate_expert_maps
    fused_layer._tpu_placement_patch = True

    # The router rows reordered in both loaders' weight streams: vLLM's
    # default safetensors loader and the Run:ai streamer (the gs:// path,
    # whose per-rank iterator model_loader_patches installs first).
    from vllm.model_executor.model_loader import default_loader as dl
    from vllm.model_executor.model_loader import runai_streamer_loader as rsl

    original_get_all = dl.DefaultModelLoader.get_all_weights

    def get_all_weights(self, *args, **kwargs):
        for name, tensor in original_get_all(self, *args, **kwargs):
            yield name, permute_router_rows(name, tensor)

    dl.DefaultModelLoader.get_all_weights = get_all_weights
    original_streamed = rsl.RunaiModelStreamerLoader._get_weights_iterator

    def _get_weights_iterator(self, *args, **kwargs):
        for name, tensor in original_streamed(self, *args, **kwargs):
            yield name, permute_router_rows(name, tensor)

    rsl.RunaiModelStreamerLoader._get_weights_iterator = _get_weights_iterator
    logger.info("Applied TPU patch: static expert placement from %s", _path())
