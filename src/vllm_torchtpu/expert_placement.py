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

``TPU_EXPERT_PLACEMENT`` names a JSON table (a path, a ``gs://`` URI or the name
of a table shipped in ``expert_placements/``) that lists, per MoE layer, the
global experts each EP rank holds in slot order:
``{"experts": E, "shards": S, "layers": {"<n>": [[ids...], ...S ranks...]}}``.
The first ``E / S`` ids of every rank hold every expert exactly once across the
ranks: the primaries. Longer rank lists hold, after them, one extra copy of some
experts, each on a rank that does not own it (expert copies, with the top-level
``copy_share`` and ``copy_min_tokens``). Other keys (such as ``meta``) are
ignored. Without copies a table is a permutation of vLLM's linear placement.

A placed layer's expert map, the EP weight filter (copies included) and the router's
rows follow the table. The router rows are reordered so that rank r's primaries become
expert ids ``r * owned .. (r + 1) * owned - 1``, so every kernel keeps its
contiguous-block ownership and nothing runs per token. Each token still picks
the same experts with the same weights (an exact tie of biased scores may
break the other way); only the order in which its top-k contributions are
summed can change. Layers the table does not list, and every layer under an
``mtp`` module (a draft model, whose layer numbers restart at 0), keep vLLM's
linear placement. Only the router rows of layers whose expert map this process
placed are reordered, and a weight stream that ends without the router weight
(``.mlp.gate.weight``) of every placed layer is refused.

Copy slots make a rank hold more weight slots than ``experts / ranks``. Only
the FP8 grouped-matmul path serves them, and only for GLM-5.2: every layer with
copy slots must list ``copy_share``, and the
GLM-5.2 patch (``models/vllm/glm_moe_dsa_patch.py``) routes a share of their
traffic to them (a layer's ``copy_share`` and the table's ``copy_min_tokens``,
both validated here). Paths that own experts by id at ``experts / ranks`` per rank
(moe_tp_decode, fused_moe_decode, fused EP, adaptive fused MoE) decline or refuse
a layer with copy slots through their equal-block checks, and other MoE
quantization methods are refused.

Tables and deployments that cannot be placed safely are refused at load (list:
the "Static Expert Placement" section of ``docs/user_guide/parallelism.md``).
"""

import dataclasses
import functools
import hashlib
import json
import os
import re
import subprocess

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_ROUTER_RE = re.compile(r"\.mlp\.gate\.(weight|bias)$|e_score_correction_bias$")
_ROUTER_WEIGHT_RE = re.compile(r"\.mlp\.gate\.weight$")
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
# Placed layers whose router weight a weight stream has reordered.
_ROUTED: set[int] = set()
# This process's EP rank, the table checked against the model, and the
# once-per-process marker lines.
_STATE: dict[str, object] = {}


@dataclasses.dataclass(frozen=True)
class Placement:
    experts: int
    ranks: int
    # Primary experts per rank: across the ranks, every expert once.
    owned: int
    # Weight slots per rank: the primaries, then the copy slots.
    slots: int
    # Per layer and rank, the global expert in each slot.
    cores: dict[int, tuple[tuple[int, ...], ...]]
    # Per layer that lists copy_share: per copied expert, the share of its
    # (token, k) pairs its copy serves (GLM-5.2's copy routing).
    shares: dict[int, dict[int, float]] = dataclasses.field(default_factory=dict)
    # Tokens per rank of the MoE all-gather below which no pair is split.
    copy_min_tokens: int = 0
    # sha256 of the table file's bytes (empty for a table parsed from a dict).
    sha256: str = ""


def _parse_shares(
    raw: dict, copied: dict[int, int], layer: int, source: str
) -> dict[int, float]:
    """A layer's copy_share: exactly its copied experts, each in [0, 1]."""
    if not copied:
        raise ValueError(
            f"expert placement {source}: layer {layer} lists copy_share but the "
            "table has no copy slots"
        )
    shares = {int(e): float(s) for e, s in raw.items()}
    if set(shares) != set(copied):
        raise ValueError(
            f"expert placement {source}: layer {layer}'s copy_share must list "
            "exactly its copied experts"
        )
    for expert, value in shares.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"expert placement {source}: layer {layer}'s copy_share for "
                f"expert {expert} is {value}, outside [0, 1]"
            )
    return shares


def parse(table: dict, source: str) -> Placement:
    """Validate a placement table: each layer's primaries (the first experts /
    shards ids of every rank) a permutation of the experts, then at most one
    copy of an expert, on a rank that does not own it, and the copy routing
    fields (copy_share, copy_min_tokens)."""
    experts = int(table["experts"])
    ranks = int(table["shards"])
    if experts % ranks:
        raise ValueError(
            f"expert placement {source}: {experts} experts do not divide over "
            f"{ranks} ranks"
        )
    owned = experts // ranks
    # Every rank list holds as many slots as the first one: the primaries,
    # then (if longer) the copy slots.
    first = next(iter(table["layers"].values()), None)
    slots = len(first[0]) if first else owned
    copies = slots > owned
    min_tokens = table.get("copy_min_tokens", 0)
    if (
        isinstance(min_tokens, bool)
        or not isinstance(min_tokens, int)
        or min_tokens < 0
    ):
        raise ValueError(
            f"expert placement {source}: copy_min_tokens must be a non-negative "
            f"integer, not {min_tokens!r}"
        )
    share_by_layer = {int(k): v for k, v in table.get("copy_share", {}).items()}
    cores, shares = {}, {}
    for key, held in table["layers"].items():
        layer = int(key)
        if layer in cores:
            raise ValueError(
                f"expert placement {source}: layer {layer} is listed twice"
            )
        rows = tuple(tuple(int(e) for e in c) for c in held)
        if len(rows) != ranks or any(len(c) != slots for c in rows):
            raise ValueError(
                f"expert placement {source}: layer {layer} must list {ranks} "
                f"ranks of {slots} experts each"
            )
        if sorted(e for c in rows for e in c[:owned]) != list(range(experts)):
            raise ValueError(
                f"expert placement {source}: layer {layer} must hold each of the "
                f"{experts} experts exactly once"
                + (" among the primaries" if copies else "")
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
        if copied and layer not in share_by_layer:
            raise ValueError(
                f"expert placement {source}: layer {layer} has copy slots but no "
                "copy_share (the share of each copied expert's pairs its copy "
                "serves); copies that serve no traffic are not supported"
            )
        if layer in share_by_layer:
            shares[layer] = _parse_shares(share_by_layer[layer], copied, layer, source)
        cores[layer] = rows
    unplaced = sorted(set(share_by_layer) - set(cores))
    if unplaced:
        raise ValueError(
            f"expert placement {source}: copy_share lists layers {unplaced}, "
            "which the table does not place"
        )
    return Placement(experts, ranks, owned, slots, cores, shares, min_tokens)


def check_layers(
    table: Placement, first_moe_layer: int, num_layers: int, source: str
) -> None:
    """Refuse a table that lists a layer outside [first_moe_layer, num_layers).

    Dense layers have no experts to place, and an MTP layer stored after the
    decoder (layer ``num_hidden_layers``) would be placed by its stored name.
    """
    bad = sorted(n for n in table.cores if not first_moe_layer <= n < num_layers)
    if bad:
        raise ValueError(
            f"expert placement {source}: lists layers {bad}; only the routed "
            f"decoder layers {first_moe_layer}..{num_layers - 1} may be placed "
            "(dense layers and MTP layers keep vLLM's linear placement)"
        )


def check_routing(table: Placement, hf: object, source: str) -> None:
    """Refuse routing that depends on expert ids beyond the router's scores.

    Grouped top-k picks experts from groups of contiguous ids, and hash routing
    picks them from a token-id table: a reordered router would change both.
    """
    groups = getattr(hf, "n_group", None)
    topk_groups = getattr(hf, "topk_group", None)
    if (
        isinstance(groups, int)
        and isinstance(topk_groups, int)
        and groups > 1
        and topk_groups < groups
    ):
        raise ValueError(
            f"expert placement {source}: grouped top-k ({topk_groups} of {groups} "
            "expert groups) selects experts by contiguous id; unset "
            "TPU_EXPERT_PLACEMENT for this model"
        )
    hashed = getattr(hf, "num_hash_layers", None)
    if isinstance(hashed, int):
        bad = sorted(n for n in table.cores if n < hashed)
        if bad:
            raise ValueError(
                f"expert placement {source}: lists hash-routed layers {bad} "
                f"(layers below {hashed} pick experts by token id)"
            )


def _read(uri: str) -> bytes:
    if uri.startswith("gs://"):
        return subprocess.run(
            ["gcloud", "storage", "cat", uri],
            check=True,
            capture_output=True,
        ).stdout
    with open(uri, "rb") as f:
        return f.read()


@functools.cache
def _load(uri: str) -> Placement:
    """The table, read once per process: the placement, its copy routing
    fields and the sha256 all come from these bytes."""
    raw = _read(uri)
    table = dataclasses.replace(
        parse(json.loads(raw), uri), sha256=hashlib.sha256(raw).hexdigest()
    )
    logger.info(
        "Expert placement %s: %d layers, %d ranks x %d slots (%d primaries)",
        uri,
        len(table.cores),
        table.ranks,
        table.slots,
        table.owned,
    )
    fused_ep = (
        envs.USE_MOE_FUSED_EP_KERNEL and envs.MOE_FUSED_EP_KERNEL_IMPL != "adaptive"
    )
    if table.slots > table.owned and (envs.USE_MOE_TP_DECODE_KERNEL or fused_ep):
        # Their equal-block checks decline every layer with copy slots (adaptive
        # fused MoE raises instead).
        logger.warning(
            "Expert placement %s: layers with copies run on the grouped-matmul "
            "path, not on USE_MOE_TP_DECODE_KERNEL or the fused EP kernel",
            uri,
        )
    return table


# Tables shipped with the package, by name (a file stem in this directory).
PLACEMENT_DIR = os.path.join(os.path.dirname(__file__), "expert_placements")


def resolve(value: str) -> str:
    """A path or gs:// URI as given, or a shipped table's name -> its file."""
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


def placement() -> Placement | None:
    value = envs.TPU_EXPERT_PLACEMENT
    return _load(resolve(value)) if value else None


def _placed_layer(table: Placement | None, name: str) -> int | None:
    match = _LAYER_RE.search(name)
    if table is None or match is None or "mtp" in name.split("."):
        return None
    layer = int(match.group(1))
    return layer if layer in table.cores else None


def layer_index(name: str) -> int | None:
    """The placed layer a module or parameter name belongs to, or None."""
    return _placed_layer(placement(), name)


def expert_map(table: Placement, layer: int, rank: int) -> list[int]:
    """vLLM's expert map: each global expert's local slot, -1 if absent."""
    local = [-1] * table.experts
    for slot, expert in enumerate(table.cores[layer][rank]):
        local[expert] = slot
    return local


def virtual_order(table: Placement, layer: int) -> list[int]:
    """The global expert behind each router row after the reorder."""
    # Copy slots have no router row: the router keeps one row per expert.
    return [e for c in table.cores[layer] for e in c[: table.owned]]


def local_expert_ids(name: str) -> frozenset[int] | None:
    """The experts this rank loads for a placed layer's weight, else None."""
    layer = layer_index(name) if _LOCAL_IDS else None
    return None if layer is None else _LOCAL_IDS.get(layer)


def num_primary_experts(layer_name: str) -> int | None:
    """Primaries per rank of a layer this worker placed with copy slots, else
    None (every weight slot holds a primary)."""
    table = placement()
    layer = _placed_layer(table, layer_name)
    if table is None or layer not in _LOCAL_IDS or table.slots == table.owned:
        return None
    return table.owned


def permute_router_rows(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """A placed layer's router weight or bias in the reordered row order.

    Only layers whose expert map this process placed are reordered, so the
    router and the experts always follow the same order.
    """
    table = placement() if _ROUTER_RE.search(name) else None
    layer = _placed_layer(table, name)
    if table is None or layer is None or layer not in _LOCAL_IDS:
        return tensor
    if tensor.shape[0] != table.experts:
        raise ValueError(
            f"expert placement: {name} has {tensor.shape[0]} rows, the table "
            f"{table.experts} experts"
        )
    if _ROUTER_WEIGHT_RE.search(name):
        _ROUTED.add(layer)
    order = torch.tensor(virtual_order(table, layer), device=tensor.device)
    return tensor.index_select(0, order)


def _placed_stream(weights):
    """A weight stream with the placed layers' router rows reordered; refused
    if it ends without the router weight of every placed layer."""
    _announce(placement())
    for name, tensor in weights:
        yield name, permute_router_rows(name, tensor)
    missing = sorted(set(_LOCAL_IDS) - _ROUTED)
    if missing:
        raise ValueError(
            f"expert placement: placed layers {missing} got no router weight "
            "(.mlp.gate.weight) from the weight stream, so their experts would "
            "not match their router; this model's router is not supported by "
            "TPU_EXPERT_PLACEMENT"
        )


def _refuse_unsupported_load_format() -> None:
    from vllm.config import get_current_vllm_config

    load_format = get_current_vllm_config().load_config.load_format
    if load_format not in _STREAMED_LOAD_FORMATS:
        raise NotImplementedError(
            "TPU_EXPERT_PLACEMENT reorders the router rows in the default and "
            f"Run:AI loaders' weight streams, which --load-format {load_format} "
            "does not use. Load with --load-format auto or runai_streamer, or "
            "unset TPU_EXPERT_PLACEMENT."
        )


def _refuse_copies_without_fp8(layer_name: str, method: object) -> None:
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU

    if not isinstance(method, VllmFp8MoEMethodTPU):
        raise NotImplementedError(
            f"{layer_name}: TPU_EXPERT_PLACEMENT gives this layer expert "
            "copies, which only the FP8 MoE method serves (through GLM-5.2's "
            "copy routing on the grouped-matmul path); this layer uses "
            f"{type(method).__name__}. Use a table without copies for this "
            "checkpoint."
        )


def _check_deployment(table: Placement) -> None:
    """Once per process, at the first MoE layer: refuse EP x TP expert
    splitting and a table that lists a dense or MTP layer of this model."""
    if _STATE.get("checked") is table:
        return
    from vllm_torchtpu.layers.adapter import moe_hierarchical

    ep_tp = moe_hierarchical.ep_tp_size()
    if ep_tp > 1 or envs.TPU_MOE_HIERARCHICAL_EP:
        raise ValueError(
            "TPU_EXPERT_PLACEMENT needs flat expert parallelism: set "
            f"TPU_MOE_EP_TP_SIZE=1 (it is {ep_tp} here) "
            "and leave TPU_MOE_HIERARCHICAL_EP off, or unset TPU_EXPERT_PLACEMENT."
        )
    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    strategy = getattr(
        getattr(vllm_config, "parallel_config", None),
        "expert_placement_strategy",
        "linear",
    )
    if strategy != "linear":
        raise ValueError(
            "TPU_EXPERT_PLACEMENT replaces vLLM's linear expert placement: "
            f"--expert-placement-strategy is {strategy!r} here; use 'linear' or "
            "unset TPU_EXPERT_PLACEMENT."
        )
    model_config = getattr(vllm_config, "model_config", None)
    hf = getattr(model_config, "hf_text_config", None) or getattr(
        model_config, "hf_config", None
    )
    num_layers = getattr(hf, "num_hidden_layers", None)
    if isinstance(num_layers, int):
        first = getattr(hf, "first_k_dense_replace", None)
        first = first if isinstance(first, int) else 0
        check_layers(table, first, num_layers, envs.TPU_EXPERT_PLACEMENT)
    check_routing(table, hf, envs.TPU_EXPERT_PLACEMENT)
    _STATE["checked"] = table


def _announce(table: Placement | None) -> None:
    """The once-per-worker delivery marker, logged when this worker starts
    streaming its weights (every placed layer is built by then)."""
    if table is None or _STATE.get("announced"):
        return
    _STATE["announced"] = True
    rank = _STATE.get("ep_rank", -1)
    if _LOCAL_IDS:
        copies = table.slots - table.owned
        logger.info(
            "[eplb] placement installed ep_rank=%d layers=%d table_sha256=%s%s",
            rank,
            len(_LOCAL_IDS),
            table.sha256[:16],
            f" copies={copies}" if copies else "",
        )
    else:
        raise ValueError(
            f"[eplb] placement EMPTY ep_rank={rank}: TPU_EXPERT_PLACEMENT="
            f"{envs.TPU_EXPERT_PLACEMENT} placed no layer of this model; unset it"
        )


def install_expert_placement() -> None:
    """Apply TPU_EXPERT_PLACEMENT to MoE layers as vLLM builds them.

    Each placed ``RoutedExperts`` rebuilds its expert map through
    ``ExpertMapManager.update`` (main's host-built map), with
    ``determine_expert_map`` answering from the table, before its weights are
    created (with copy slots, ``moe_config.num_local_experts`` follows, so the
    weights get every slot). A placed layer with copy slots must get the FP8
    MoE method. Both loaders' weight streams reorder the router rows.
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
    get_quant_method = RoutedExperts._get_quant_method
    get_all_weights = default_loader.DefaultModelLoader.get_all_weights
    streamed = runai_streamer_loader.RunaiModelStreamerLoader._get_weights_iterator
    building: list[tuple[Placement, int]] = []

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
        table, layer = building[-1]
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
        if not use_ep and _placed_layer(placement(), self.layer_name) is not None:
            raise ValueError(
                f"TPU_EXPERT_PLACEMENT lists {self.layer_name}, which runs without "
                "expert parallelism: pass --enable-expert-parallel or unset "
                "TPU_EXPERT_PLACEMENT"
            )
        table = placement() if use_ep else None
        if table is not None:
            _check_deployment(table)
        layer = _placed_layer(table, self.layer_name)
        if table is not None and layer is not None:
            _refuse_unsupported_load_format()
            manager = self.expert_map_manager
            building.append((table, layer))
            try:
                manager.update(manager.moe_parallel_config, manager.global_num_experts)
            finally:
                building.pop()
            # The weights are created with this count: the slots, copies included.
            self.moe_config.num_local_experts = manager.local_num_experts
            _LOCAL_IDS[layer] = frozenset(table.cores[layer][manager.ep_rank])
            _STATE["ep_rank"] = manager.ep_rank
        elif use_ep:
            logger.info(
                "Expert placement: %s keeps vLLM's linear placement", self.layer_name
            )
        update_info(self)

    def placed_quant_method(self, *args, **kwargs):
        # RoutedExperts.__init__ picks the method right after the expert map.
        method = get_quant_method(self, *args, **kwargs)
        if num_primary_experts(self.layer_name) is not None:
            _refuse_copies_without_fp8(self.layer_name, method)
        return method

    def placed_all_weights(self, model_config, model):
        yield from _placed_stream(get_all_weights(self, model_config, model))

    def placed_weights_iterator(self, model_or_path, revision):
        yield from _placed_stream(streamed(self, model_or_path, revision))

    expert_map_manager.determine_expert_map = determine_expert_map
    RoutedExperts.update_expert_map_info = update_expert_map_info
    RoutedExperts._get_quant_method = placed_quant_method
    default_loader.DefaultModelLoader.get_all_weights = placed_all_weights
    runai_streamer_loader.RunaiModelStreamerLoader._get_weights_iterator = (
        placed_weights_iterator
    )
    RoutedExperts._tpu_expert_placement_patch = True
    logger.info(
        "Applied TPU patch: static expert placement from %s", envs.TPU_EXPERT_PLACEMENT
    )
