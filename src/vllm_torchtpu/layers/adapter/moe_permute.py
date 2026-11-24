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
"""Static expert permutation for the TPU grouped-matmul EP MoE path.

``VLLM_TPU_MOE_EXPERT_PERMUTATION=<spec>`` (default empty = off) changes which
EP rank holds which expert, per MoE layer, without adding expert slots (no
extra HBM). ``spec`` is ``builtin:<name>`` (``moe_placements/<name>.json``), a
JSON file path, or inline JSON: ``{"moe<i>": [slot of logical expert e, ...]}``
where ``moe<i>`` is the i-th MoE layer of the backbone (route-trace numbering)
and slot ``p`` lives on rank ``p // L`` at local index ``p % L``
(``L = E // ep``). Layers without a table keep the linear placement.

Why: a decode-sized grouped matmul is weight-read bound, so a rank's time
follows the number of distinct experts it computes, and the all-reduce after
the layer waits for the busiest rank. Served routing has co-hit expert groups
(tokens of one request share experts), so the linear placement often puts
several co-hit experts on one rank. The tables are fit to served routing so
co-hit experts land on different ranks.

Mechanics: the layer's global->local expert map is replaced by the permuted
one before the weights load, so the checkpoint loader writes logical expert
``e`` into local slot ``perm[e] % L`` of rank ``perm[e] // L``; every step the
routed logical ids are mapped through ``perm`` (one gather). ``experts_start``
stays ``rank * L``, so ``fused_moe_func`` needs no change. The same experts with
the same weights are evaluated for every token; only the rank computing each
(token, expert) pair changes, so the result differs from linear placement only
by the summation grouping across ranks before the EP all-reduce.

Loading: the EP weight filters (``--enable-ep-weight-filter``) skip experts that
are not local under the linear placement, so they are made permutation-aware
rather than turned off: the expert-id filter is widened to the union of this
rank's linear and permuted-local experts, and while the backbone streams, a
per-layer check (``skip_expert_tensor``) keeps exactly the permuted-local
experts of each permuted layer. ``install`` raises if any permuted-local slot
was not written, so a filter that still drops experts fails the boot instead of
serving uninitialized expert weights.
"""

from __future__ import annotations

import builtins
import contextlib
import json
import os
import re
from collections.abc import Iterator
from typing import Any

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Fixed at import, like the other MoE selectors: it decides the traced graph.
SPEC: str = str(envs.VLLM_TPU_MOE_EXPERT_PERMUTATION or "").strip()
ENABLED: bool = bool(SPEC)

_BUILTIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moe_placements")

_source_cache: dict[str, Any] | None = None
_ORDINAL = [0]
_STATS = {"layers": 0, "moved": 0, "prepared": 0, "checked_experts": 0}

# Layer index in module and checkpoint names (``model.layers.<i>.``).
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
# Decoder layer index -> logical experts this rank holds, for every backbone
# MoE layer prepared while the permutation is on (linear for layers without a
# table). Drives the per-layer load filter.
_LOCAL_BY_LAYER: dict[int, frozenset[int]] = {}
# Union over layers of this rank's local experts (linear and permuted).
_LOCAL_UNION: set[int] = set()
# True while a model with permuted layers loads its weights.
_LOADING = [False]


def load_source(spec: str) -> dict[str, Any]:
    spec = spec.strip()
    if not spec:
        return {}
    if spec.startswith("builtin:"):
        name = spec.split(":", 1)[1]
        path = os.path.join(_BUILTIN_DIR, f"{name}.json")
        if not os.path.isfile(path):
            have = sorted(f[:-5] for f in os.listdir(_BUILTIN_DIR) if f.endswith(".json"))
            raise ValueError(
                f"VLLM_TPU_MOE_EXPERT_PERMUTATION={spec!r}: unknown builtin (have {have})"
            )
        with open(path) as f:
            data = json.load(f)
    elif spec.startswith("{"):
        data = json.loads(spec)
    else:
        with open(os.path.expanduser(spec)) as f:
            data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("expert permutation must be a JSON object")
    return {k: v for k, v in data.items() if not k.startswith("_")}


def source() -> dict[str, Any]:
    global _source_cache
    if _source_cache is None:
        _source_cache = load_source(SPEC)
    return _source_cache


def validate(perm: Any, num_logical: int) -> list[int]:
    perm = [int(p) for p in perm]
    if sorted(perm) != list(range(num_logical)):
        raise ValueError(
            f"expert permutation must be a permutation of 0..{num_logical - 1}"
        )
    return perm


def _layer_ep(layer: Any) -> tuple[bool, int, int]:
    cfg = layer.moe_config
    return bool(cfg.moe_parallel_config.use_ep), int(cfg.ep_size), int(cfg.ep_rank)


def _model_tag() -> str:
    try:
        from vllm.compilation import backends
    except Exception:  # pragma: no cover - host-only tests without vLLM
        return "backbone"
    return str(getattr(backends, "model_tag", "backbone"))


def _layer_index(name: str) -> int | None:
    m = _LAYER_RE.search(name or "")
    return int(m.group(1)) if m else None


def _register_local(layer: Any, local_ids: set[int]) -> None:
    idx = _layer_index(str(getattr(layer, "layer_name", "") or ""))
    if idx is not None:
        _LOCAL_BY_LAYER[idx] = frozenset(local_ids)
    _LOCAL_UNION.update(local_ids)


_filter_patched = [False]


def _patch_expert_id_filter() -> None:
    """Widen vLLM's expert-id load filter to every expert this rank holds.

    The loaders resolve the filter through
    ``default_loader.compute_local_expert_ids`` when loading starts (after the
    model is built, so every layer is prepared by then). The union keeps the
    filter on: a rank still skips experts it holds in no layer.
    """
    if _filter_patched[0]:
        return
    try:
        from vllm.model_executor.model_loader import default_loader
    except Exception:  # pragma: no cover - host-only tests without vLLM
        return
    original = default_loader.compute_local_expert_ids

    def compute_local_expert_ids(*args: Any, **kwargs: Any):
        ids = original(*args, **kwargs)
        return widen_local_expert_ids(ids)

    default_loader.compute_local_expert_ids = compute_local_expert_ids
    _filter_patched[0] = True
    logger.info(
        "[moe-permute] EP weight filter is permutation-aware: each rank "
        "reads its permuted-local experts."
    )


def widen_local_expert_ids(ids: Any) -> Any:
    """``ids`` plus every expert this rank holds in a permuted layer."""
    if ids is None or not _LOCAL_UNION:
        return ids
    return set(ids) | _LOCAL_UNION


def widen_external_filter() -> None:
    """Widen a launcher-installed expert filter the same way.

    Some launchers (the exos runners) patch the Run:ai loader to skip
    non-local experts by a linear split they compute themselves, published
    as ``builtins._EXOS_MODEL_LOAD_FILTER = (is_mtp, pp_missing, mtp_prefix,
    local_expert_ids)`` for the duration of ``load_weights``. Call while that
    load runs, before the checkpoint files are selected.
    """
    if not _LOADING[0]:
        return
    flt = getattr(builtins, "_EXOS_MODEL_LOAD_FILTER", None)
    if not isinstance(flt, tuple) or len(flt) != 4 or flt[3] is None or flt[0]:
        return
    widened = set(flt[3]) | _LOCAL_UNION
    if widened != set(flt[3]):
        builtins._EXOS_MODEL_LOAD_FILTER = (flt[0], flt[1], flt[2], widened)
        logger.info_once(
            "[moe-permute] launcher expert filter widened %d -> %d experts.",
            len(flt[3]),
            len(widened),
        )


def has_permuted_layers(model: Any) -> bool:
    return any(
        getattr(m, "_perm_logical_to_physical", None) is not None
        for m in model.modules()
    )


@contextlib.contextmanager
def loading(model: Any) -> Iterator[None]:
    """Turn the per-layer load filter on while ``model`` loads, if permuted."""
    active = ENABLED and model is not None and has_permuted_layers(model)
    prev = _LOADING[0]
    _LOADING[0] = active or prev
    try:
        yield
    finally:
        _LOADING[0] = prev


def skip_expert_tensor(name: str, expert_id: int) -> bool | None:
    """Per-layer load filter for one expert tensor of the loading backbone.

    True/False for an expert of a MoE layer this rank prepared (skip unless
    the expert is local to this rank in that layer); None when it does not
    decide (not loading a permuted model, or not a known MoE layer), so the
    caller's expert-id filter applies.
    """
    if not _LOADING[0]:
        return None
    idx = _layer_index(name)
    local = _LOCAL_BY_LAYER.get(idx) if idx is not None else None
    if local is None:
        return None
    return expert_id not in local


def write_tracking() -> bool:
    """Whether the expert write hook (model_loader_patches) records writes;
    without it ``install`` cannot see them and skips the check."""
    from vllm_torchtpu import model_loader_patches

    return bool(model_loader_patches.EXPERT_WRITE_HOOK)


def note_write(layer: Any, expert_id: int, weight_name: str, shard_id: Any) -> None:
    """Record one successful expert write into a permuted layer."""
    if getattr(layer, "_perm_logical_to_physical", None) is None:
        return
    writes = layer.__dict__.setdefault("_perm_writes", {})
    param = str(weight_name).rsplit(".", 1)[-1]
    writes.setdefault(int(expert_id), set()).add((param, str(shard_id)))


def check_complete(layer: Any) -> int:
    """Raise unless every permuted-local expert of ``layer`` was written.

    Every local expert must have received the same (parameter, shard) writes
    as the most-written one. Returns the number of experts checked.
    """
    local_ids = sorted(getattr(layer, "_perm_local_ids", ()))
    writes: dict[int, set] = layer.__dict__.get("_perm_writes", {})
    full: set = set().union(*writes.values()) if writes else set()
    missing = [e for e in local_ids if e not in writes]
    partial = [e for e in local_ids if e in writes and writes[e] != full]
    if missing or partial or not full:
        _, ep, rank = _layer_ep(layer)
        raise RuntimeError(
            f"[moe-permute] {getattr(layer, 'layer_name', '?')} rank {rank}/{ep}: "
            f"permuted-local experts not loaded: missing={missing} "
            f"partial={partial} (expected writes per expert {sorted(full)}). "
            "An EP weight filter still skips experts outside the linear "
            "placement; refusing to serve uninitialized expert weights."
        )
    return len(local_ids)


def prepare(layer: Any) -> bool:
    """Install this layer's permuted global->local map before weights load.

    Call from ``create_weights``. Returns True when the layer is permuted.
    """
    if not ENABLED:
        return False
    from vllm_torchtpu.layers.adapter import moe_redundant

    use_ep, ep, rank = _layer_ep(layer)
    num_logical = int(layer.global_num_experts)
    reason = None
    if moe_redundant.ENABLED:
        reason = "VLLM_TPU_MOE_REDUNDANT_EXPERTS is on"
    elif not use_ep or ep < 2:
        reason = "expert parallelism is off"
    elif num_logical % ep:
        reason = f"{num_logical} experts do not split evenly over ep={ep}"
    elif getattr(layer, "expert_placement_strategy", "linear") != "linear":
        reason = "placement strategy is not linear"
    elif envs.MOE_FUSED_EP_KERNEL_IMPL == "adaptive":
        reason = "the adaptive fused EP kernel does not take a permuted expert map"
    if reason is None and _model_tag() != "backbone":
        logger.info_once(
            "[moe-permute] %r graph keeps the linear placement.", _model_tag()
        )
        return False
    if reason is not None:
        logger.warning_once(
            "[moe-permute] VLLM_TPU_MOE_EXPERT_PERMUTATION ignored: %s.", reason
        )
        return False
    ordinal = _ORDINAL[0]
    _ORDINAL[0] += 1
    local = num_logical // ep
    lo = rank * local
    table = source().get(f"moe{ordinal}")
    if table is None:
        _register_local(layer, set(range(lo, lo + local)))
        return False
    perm = validate(table, num_logical)
    to_local = [p - lo if lo <= p < lo + local else -1 for p in perm]
    local_ids = {e for e, p in enumerate(perm) if lo <= p < lo + local}
    _register_local(layer, local_ids)
    _patch_expert_id_filter()
    layer._perm_logical_to_physical = perm
    layer._perm_local_ids = frozenset(local_ids)
    _STATS["prepared"] += 1
    # Instance attribute shadows the class method: vLLM's expert weight
    # loader resolves the local slot of every checkpoint expert through it.
    layer.__dict__["_map_global_expert_id_to_local_expert_id"] = (
        lambda gid, _m=to_local: _m[gid] if 0 <= gid < len(_m) else -1
    )
    _STATS["moved"] += sum(1 for e, p in enumerate(perm) if p // local != e // local)
    return True


def install(layer: Any) -> None:
    """Register the logical->physical routing table (after weights load)."""
    perm = getattr(layer, "_perm_logical_to_physical", None)
    if perm is None:
        return
    tracked = write_tracking()
    if tracked:
        _STATS["checked_experts"] += check_complete(layer)
    else:
        logger.warning_once(
            "[moe-permute] expert write hook not installed: load "
            "completeness is not checked."
        )
    device = layer.w13_weight.device
    layer.register_buffer(
        "_perm_table", torch.tensor(perm, dtype=torch.int32, device=device),
        persistent=False,
    )
    _STATS["layers"] += 1
    if _STATS["layers"] == 1:
        _, ep, rank = _layer_ep(layer)
        logger.info(
            "[moe-permute] VLLM_TPU_MOE_EXPERT_PERMUTATION=%s on: E=%d ep=%d rank=%d "
            "first_layer=%s (no extra expert slots)",
            SPEC, len(perm), ep, rank, getattr(layer, "layer_name", "?"),
        )
    if _STATS["layers"] == _STATS["prepared"]:
        _, ep, rank = _layer_ep(layer)
        logger.info(
            "[moe-permute] load check rank %d/%d: %d/%d permuted layers, "
            "%d permuted-local experts written, 0 incomplete (checked=%s)",
            rank, ep, _STATS["layers"], _STATS["prepared"],
            _STATS["checked_experts"], tracked,
        )


def summary() -> dict[str, int]:
    return dict(_STATS)


def to_physical(layer: Any, topk_ids: torch.Tensor) -> torch.Tensor:
    """Map logical routed ids ``[T, K]`` to physical ids (traced code)."""
    table = getattr(layer, "_perm_table", None)
    if table is None:
        return topk_ids
    ids = topk_ids.to(torch.int32)
    phys = table[ids.clamp(min=0).to(torch.int64)]
    return torch.where(ids >= 0, phys, ids).to(topk_ids.dtype)
