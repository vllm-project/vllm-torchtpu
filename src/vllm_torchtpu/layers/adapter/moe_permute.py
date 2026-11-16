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
"""

from __future__ import annotations

import json
import os
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
_STATS = {"layers": 0, "moved": 0}


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


def _widen_ep_weight_filter() -> None:
    # Permuted ranks hold experts outside the linear block, so the EP weight
    # filter (local ids by linear placement) must not skip them.
    from vllm_torchtpu.layers.adapter import moe_redundant

    moe_redundant._widen_ep_weight_filter()


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
    table = source().get(f"moe{ordinal}")
    if table is None:
        return False
    perm = validate(table, num_logical)
    local = num_logical // ep
    lo = rank * local
    to_local = [p - lo if lo <= p < lo + local else -1 for p in perm]
    _widen_ep_weight_filter()
    layer._perm_logical_to_physical = perm
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
