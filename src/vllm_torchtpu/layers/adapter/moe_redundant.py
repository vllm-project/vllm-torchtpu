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
"""Static redundant experts for the TPU grouped-matmul EP MoE path.

``VLLM_TPU_MOE_REDUNDANT_EXPERTS=<n>`` (default 0 = off) gives every EP rank
``n`` extra expert slots per MoE layer. Each slot holds a full copy of one
other rank's expert, loaded from the checkpoint like the native experts.
Every step the routed logical expert ids are remapped to physical slots, so a
replicated expert's work can run on either copy. The same experts with the
same weights are evaluated for every token; only the rank that computes a
(token, expert) pair changes. The per-token sum over experts is therefore the
same math, grouped differently across ranks before the EP all-reduce.

Why it is not vLLM's EPLB: ``--enable-eplb`` is rejected on TPU by
``ParallelConfig`` (CUDA/ROCm only), the TPU runner never builds an
``EplbState``, and the monolithic TPU router never maps logical ids to
physical ones. EPLB's rebalancer also balances the *window-averaged* load,
which is already even here; the imbalance is per step.

Physical layout (linear EP, ``E`` logical experts, ``ep`` ranks,
``L = E // ep`` native experts per rank, ``S = L + n`` slots per rank):

- rank ``r`` owns physical ids ``[r * S, (r + 1) * S)``;
- logical ``l`` natively lives at ``(l // L) * S + l % L``;
- redundant slot ``j`` of rank ``r`` is physical ``r * S + L + j`` and holds
  logical ``rep_logical[r * n + j]``.

``fused_moe_func`` needs no change: ``experts_start = rank * S`` and the
local weight tensors have ``S`` experts, so its ``id - experts_start`` remap
and range mask select exactly this rank's slots.

Replica choice per step: decode-sized grouped matmuls are weight-read bound,
so a rank's time follows the number of distinct experts it computes, and the
all-reduce after the layer waits for the busiest rank. For batches of at most
``MAX_ROUTES`` routes, ``kernels/moe_replica_assign.py`` sends every touched
expert *whole* to one copy (single-copy experts first, then each replicated
one to its least-loaded copy), so no expert's weights are read twice. Larger
batches (prefill) use the native copies and compute exactly as without
replicas.

The placement (which experts are replicated, which rank holds each copy) is
identical on all ranks and reaches the compiled graph only as buffers, so a
placement change never needs a recompile and never desyncs ranks.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import jax
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Fixed at import, like the other MoE selectors: they decide the traced graph.
SLOTS_PER_RANK: int = int(envs.VLLM_TPU_MOE_REDUNDANT_EXPERTS)
ENABLED: bool = SLOTS_PER_RANK > 0
# Batches above this many routes (T * top_k) use the native copies.
MAX_ROUTES: int = 2048
# Expert ids handed to the loader for replica writes. Distinct from every
# real id so the expert-write staging tracker counts each slot once.
_REPLICA_ID_BASE = 1 << 20


@dataclass(frozen=True)
class LayerPlacement:
    """Global (rank-independent) redundant-expert placement of one layer."""

    num_logical: int
    ep: int
    slots: int
    # Logical expert held by redundant slot j of rank r, at index r*slots+j.
    rep_logical: tuple[int, ...]

    @property
    def local_native(self) -> int:
        return self.num_logical // self.ep

    @property
    def slots_total(self) -> int:
        return self.local_native + self.slots

    @property
    def num_physical(self) -> int:
        return self.ep * self.slots_total

    def home_rank(self, logical: int) -> int:
        return logical // self.local_native

    def native_physical(self, logical: int) -> int:
        local = self.local_native
        return (logical // local) * self.slots_total + logical % local

    def alt_rank(self, index: int) -> int:
        return index // self.slots

    def alt_physical(self, index: int) -> int:
        rank, j = divmod(index, self.slots)
        return rank * self.slots_total + self.local_native + j

    def physical_to_logical(self) -> list[int]:
        out = [-1] * self.num_physical
        for logical in range(self.num_logical):
            out[self.native_physical(logical)] = logical
        for i, logical in enumerate(self.rep_logical):
            out[self.alt_physical(i)] = logical
        return out

    def local_replicas(self, rank: int) -> dict[int, list[int]]:
        """Logical expert -> local slot indices of ``rank``'s replica slots."""
        out: dict[int, list[int]] = {}
        for j in range(self.slots):
            logical = self.rep_logical[rank * self.slots + j]
            out.setdefault(logical, []).append(self.local_native + j)
        return out


def compute_placement(
    num_logical: int,
    ep: int,
    slots: int,
    hot: Iterable[int] | dict[int, float] | None = None,
) -> LayerPlacement:
    """Pick ``ep * slots`` experts to replicate and a rank for each copy.

    ``hot`` lists experts hottest first (or maps expert -> load share). The
    listed experts are taken in order, then the rest of the slots are filled
    round-robin over ranks with each rank's lowest unlisted local experts.
    Each copy goes to a rank other than the expert's home rank, preferring
    ranks whose native experts carry the least known load, so the levers
    point from heavy ranks to light ones. Deterministic: every rank computes
    the same placement.
    """
    if ep < 2 or slots < 1 or num_logical % ep:
        raise ValueError(
            f"redundant experts need ep >= 2, slots >= 1 and E % ep == 0; got "
            f"E={num_logical} ep={ep} slots={slots}"
        )
    local = num_logical // ep
    total = ep * slots
    if total > num_logical:
        raise ValueError(f"{total} redundant slots exceed {num_logical} experts")
    weights: dict[int, float] = {}
    order: list[int] = []
    if isinstance(hot, dict):
        items = sorted(
            ((int(k), float(v)) for k, v in hot.items()), key=lambda kv: -kv[1]
        )
        for k, v in items:
            if 0 <= k < num_logical and k not in weights:
                weights[k] = v
                order.append(k)
    elif hot is not None:
        listed = [int(x) for x in hot]
        for rank_pos, k in enumerate(listed):
            if 0 <= k < num_logical and k not in weights:
                weights[k] = float(len(listed) - rank_pos)
                order.append(k)
    # At most (ep - 1) * slots copies of one rank's experts fit on the other
    # ranks; with that cap a valid assignment always exists (Hall), and the
    # greedy below finds one with at most one swap per expert.
    cap = (ep - 1) * slots
    per_home = [0] * ep
    chosen: list[int] = []
    taken: set[int] = set()

    def _take(logical: int) -> None:
        home = logical // local
        if logical in taken or per_home[home] >= cap or len(chosen) >= total:
            return
        chosen.append(logical)
        taken.add(logical)
        per_home[home] += 1

    for logical in order:
        _take(logical)
    k = 0
    while len(chosen) < total and k < local:
        for r in range(ep):
            _take(r * local + k)
        k += 1
    # Known native load per rank (all listed experts, not only the chosen).
    rank_load = [0.0] * ep
    for logical, w in weights.items():
        rank_load[logical // local] += w
    free = [slots] * ep
    assign: list[int] = [-1] * total  # alt rank of chosen[i]
    added = [0.0] * ep
    for i, logical in enumerate(chosen):
        home = logical // local
        cands = [r for r in range(ep) if r != home and free[r] > 0]
        if not cands:
            # Only the home rank has room: swap with an earlier copy whose
            # rank can take this expert and whose expert can go home's slot.
            for j in range(i):
                a = assign[j]
                if a != home and chosen[j] // local != home:
                    assign[i], assign[j] = a, home
                    free[home] -= 1
                    break
            else:  # pragma: no cover - excluded by the per-home cap
                raise RuntimeError("redundant expert placement failed")
            continue
        r = min(cands, key=lambda c: (rank_load[c] + added[c], slots - free[c], c))
        assign[i] = r
        free[r] -= 1
        # The two copies then share the expert's load about evenly.
        share = weights.get(logical, 0.0) / 2
        added[r] += share
        added[home] -= share
    rep: list[int] = [-1] * total
    next_slot = [0] * ep
    for i, logical in enumerate(chosen):
        r = assign[i]
        rep[r * slots + next_slot[r]] = logical
        next_slot[r] += 1
    assert -1 not in rep
    return LayerPlacement(num_logical, ep, slots, tuple(rep))


def layer_key(name: str) -> str:
    """``model.layers.17.mlp.experts`` -> ``L17``; MTP/draft -> ``mtp<N>``.

    Same naming as the ``[diag:model:moe]`` lines, so their hot tables can be
    pasted into a placement file.
    """
    parts = str(name).split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            prefix = "mtp" if "mtp" in name or "draft" in name else "L"
            return f"{prefix}{parts[i + 1]}"
    return parts[-1] if parts else str(name)


# ---------------------------------------------------------------------------
# Placement source
# ---------------------------------------------------------------------------

# Placement tables measured on served routing (route-trace lines), one JSON per
# workload: {"moe<i>": {"<expert>": <% of decode steps touching it>, ...}}.
_BUILTIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moe_placements")

_placement_cache: dict[str, Any] | None = None


def _load_source(spec: str) -> dict[str, Any]:
    spec = spec.strip()
    if not spec:
        return {}
    if spec.startswith("builtin:"):
        name = spec.split(":", 1)[1]
        path = os.path.join(_BUILTIN_DIR, f"{name}.json")
        if not os.path.isfile(path):
            have = sorted(f[:-5] for f in os.listdir(_BUILTIN_DIR) if f.endswith(".json"))
            raise ValueError(
                f"VLLM_TPU_MOE_REDUNDANT_PLACEMENT={spec!r}: unknown builtin "
                f"(have {have})"
            )
        with open(path) as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if not k.startswith("_")}
    if spec.startswith("{"):
        data = json.loads(spec)
    else:
        with open(os.path.expanduser(spec)) as f:
            data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("redundant-expert placement must be a JSON object")
    return data


def placement_source() -> dict[str, Any]:
    global _placement_cache
    if _placement_cache is None:
        _placement_cache = _load_source(envs.VLLM_TPU_MOE_REDUNDANT_PLACEMENT)
    return _placement_cache


def hot_for_layer(source: dict[str, Any], name: str, ordinal: int = -1) -> Any:
    """Hot experts of a layer: by ``L<i>`` key, full name, or ``moe<ordinal>``
    (the layer's position among the MoE layers, as the route trace numbers
    them)."""
    key = layer_key(name)
    if key in source:
        return source[key]
    if str(name) in source:
        return source[str(name)]
    return source.get(f"moe{ordinal}")


# ---------------------------------------------------------------------------
# Layer hooks
# ---------------------------------------------------------------------------


def _layer_ep(layer: Any) -> tuple[bool, int, int]:
    cfg = layer.moe_config
    use_ep = bool(cfg.moe_parallel_config.use_ep)
    return use_ep, int(cfg.ep_size), int(cfg.ep_rank)


def _model_tag() -> str:
    try:
        # Read at call time: `set_model_tag` rebinds the module global.
        from vllm.compilation import backends
    except Exception:  # pragma: no cover - host-only tests without vLLM
        return "backbone"
    return str(getattr(backends, "model_tag", "backbone"))


def _env_flag(name: str) -> bool:
    # Knobs that only exist on some branches of vllm-torchtpu.
    return name in envs.environment_variables and bool(getattr(envs, name))


def _pipelined() -> bool:
    try:
        from vllm_torchtpu.layers.adapter.pipelined_fused_moe import (
            enable_pipelined_collective_and_compute,
        )
    except Exception:  # pragma: no cover - import guard for host-only tests
        return False
    return bool(enable_pipelined_collective_and_compute())


_ORDINAL = [0]


def prepare(layer: Any) -> int:
    """Decide this layer's placement; return the extra local expert slots.

    Call from ``create_weights`` before the expert parameters are allocated.
    Returns 0 (and leaves the layer untouched) when off or not applicable.
    """
    if not ENABLED:
        return 0
    use_ep, ep, _ = _layer_ep(layer)
    num_logical = int(layer.global_num_experts)
    reason = None
    if not use_ep or ep < 2:
        reason = "expert parallelism is off"
    elif num_logical % ep:
        reason = f"{num_logical} experts do not split evenly over ep={ep}"
    elif getattr(layer, "expert_placement_strategy", "linear") != "linear":
        reason = "placement strategy is not linear"
    elif envs.MOE_FUSED_EP_KERNEL_IMPL == "adaptive":
        reason = "the adaptive fused EP kernel does not take a physical expert map"
    elif _pipelined():
        reason = "pipelined collective+compute MoE is on"
    elif _env_flag("TPU_MOE_SEQUENCE_PARALLEL"):
        reason = "TPU_MOE_SEQUENCE_PARALLEL routes rank-local token slices"
    if reason is None and _model_tag() != "backbone":
        # Only the backbone is re-traced per token-bucket group
        # (shape_variants). Other graphs (the MTP/EAGLE draft, tag
        # "eagle_head") are traced once for every bucket, so the remap's
        # token-count branch would freeze there and fail the other buckets
        # ("Token bucket 512 cannot be served ... assumes 10*s18 <= 4096").
        logger.info_once(
            "[moe-redundant] %r graph keeps plain experts: it is traced once "
            "for every bucket.",
            _model_tag(),
        )
        return 0
    if reason is not None:
        logger.warning_once(
            "[moe-redundant] VLLM_TPU_MOE_REDUNDANT_EXPERTS=%d ignored: %s.",
            SLOTS_PER_RANK,
            reason,
        )
        return 0
    _widen_ep_weight_filter()
    name = str(getattr(layer, "layer_name", "") or "")
    ordinal = _ORDINAL[0]
    _ORDINAL[0] += 1
    hot = hot_for_layer(placement_source(), name, ordinal)
    plan = compute_placement(num_logical, ep, SLOTS_PER_RANK, hot)
    layer._rx_plan = plan
    layer._rx_from_source = hot is not None
    return SLOTS_PER_RANK


_filter_widened = False


def _widen_ep_weight_filter() -> None:
    """Let every rank read every expert's checkpoint tensors.

    ``--enable-ep-weight-filter`` skips non-local experts before reading them,
    but replica slots hold other ranks' experts. vLLM's loader resolves the
    filter through ``default_loader.compute_local_expert_ids`` when it starts
    loading (after the model is built), so returning ``None`` there turns the
    filter off, as vLLM itself does under ``--enable-eplb``.
    """
    global _filter_widened
    if _filter_widened:
        return
    try:
        from vllm.model_executor.model_loader import default_loader
    except Exception:  # pragma: no cover - host-only tests without vLLM
        return

    def no_filter(*args: Any, **kwargs: Any) -> None:
        return None

    default_loader.compute_local_expert_ids = no_filter
    _filter_widened = True
    logger.info(
        "[moe-redundant] EP weight filter off: replica slots read other "
        "ranks' experts."
    )


@contextlib.contextmanager
def _local_slot_override(layer: Any, slot: int):
    # Instance attribute shadows the class method for the duration of one
    # loader call; both the TPU staging wrapper and vLLM's loader resolve
    # the local expert through it.
    layer.__dict__["_map_global_expert_id_to_local_expert_id"] = lambda _gid: slot
    try:
        yield
    finally:
        layer.__dict__.pop("_map_global_expert_id_to_local_expert_id", None)


def wrap_weight_loaders(layer: Any) -> None:
    """Make every expert parameter's loader also fill this rank's replicas.

    The checkpoint loader offers every logical expert to every rank. A rank
    keeps its native experts as before and, for the experts it holds a copy
    of, writes the same checkpoint tensor once more into the replica slot.
    """
    plan: LayerPlacement | None = getattr(layer, "_rx_plan", None)
    if plan is None:
        return
    _, _, rank = _layer_ep(layer)
    replicas = plan.local_replicas(rank)
    loaded: dict[str, set[int]] = {}
    layer._rx_loaded = loaded
    for pname, param in list(layer.named_parameters(recurse=False)):
        if not pname.startswith(("w13_", "w2_")):
            continue
        orig = getattr(param, "weight_loader", None)
        if orig is None or getattr(orig, "_rx_wrapped", False):
            continue

        def loader(
            param,
            loaded_weight,
            weight_name,
            shard_id,
            expert_id,
            return_success=False,
            _orig=orig,
            _pname=pname,
        ):
            ok = _orig(
                param,
                loaded_weight,
                weight_name,
                shard_id,
                expert_id,
                return_success=True,
            )
            for slot in replicas.get(int(expert_id), ()):
                with _local_slot_override(layer, slot):
                    done = _orig(
                        param,
                        loaded_weight,
                        weight_name,
                        shard_id,
                        _REPLICA_ID_BASE + slot,
                        return_success=True,
                    )
                if done:
                    loaded.setdefault(_pname, set()).add(slot)
                ok = bool(ok) or bool(done)
            return ok if return_success else None

        loader._rx_wrapped = True
        param.weight_loader = loader


def _dummy_weights() -> bool:
    """``--load-format dummy`` never calls the weight loaders."""
    try:
        from vllm.config import get_current_vllm_config

        return get_current_vllm_config().load_config.load_format == "dummy"
    except Exception:
        return False


def install(layer: Any) -> None:
    """Register the routing buffers and the physical ``experts_start``.

    Call from ``process_weights_after_loading`` after
    ``register_experts_start_buffer`` (this overrides it).
    """
    plan: LayerPlacement | None = getattr(layer, "_rx_plan", None)
    if plan is None:
        return
    _, ep, rank = _layer_ep(layer)
    w13 = layer.w13_weight
    if int(w13.shape[0]) != plan.slots_total:
        raise RuntimeError(
            f"[moe-redundant] {layer.layer_name}: w13 has {w13.shape[0]} experts, "
            f"placement expects {plan.slots_total}"
        )
    loaded = getattr(layer, "_rx_loaded", {})
    want = set(plan.local_replicas(rank).keys())
    want_slots = {s for slots in plan.local_replicas(rank).values() for s in slots}
    for pname in ("w13_weight", "w2_weight"):
        missing = want_slots - loaded.get(pname, set())
        if want and missing and not _dummy_weights():
            raise RuntimeError(
                f"[moe-redundant] {layer.layer_name}: replica slots {sorted(missing)} "
                f"of {pname} were never loaded (is --enable-ep-weight-filter on?)"
            )
    device = w13.device
    layer.register_buffer(
        "_experts_start",
        torch.tensor(rank * plan.slots_total, dtype=torch.int32, device=device),
        persistent=False,
    )
    layer.register_buffer(
        "_rx_copies", copies_table(plan).to(device), persistent=False
    )
    if device.type == "tpu":
        # Register outside tracing: Dynamo cannot trace the registration.
        _get_assign_op()
    layer._rx_num_physical = plan.num_physical
    per_slot = sum(
        int(p[0].numel()) * p.element_size()
        for n, p in layer.named_parameters(recurse=False)
        if n.startswith(("w13_", "w2_"))
    )
    _STATS["layers"] += 1
    _STATS["from_source"] += int(bool(getattr(layer, "_rx_from_source", False)))
    _STATS["bytes"] += per_slot * plan.slots
    if _STATS["layers"] == 1:
        logger.info(
            "[moe-redundant] VLLM_TPU_MOE_REDUNDANT_EXPERTS=%d on: E=%d ep=%d "
            "S=%d physical_E=%d max_routes=%d placement=%r first_layer=%s rank=%d "
            "replicas=%s slot_bytes=%d",
            plan.slots,
            plan.num_logical,
            ep,
            plan.slots_total,
            plan.num_physical,
            MAX_ROUTES,
            envs.VLLM_TPU_MOE_REDUNDANT_PLACEMENT,
            layer_key(layer.layer_name),
            rank,
            ",".join(
                f"{log}->r{plan.alt_rank(i)}" for i, log in enumerate(plan.rep_logical)
            ),
            per_slot,
        )


_STATS = {"layers": 0, "from_source": 0, "bytes": 0}


def summary() -> dict[str, int]:
    """Installed layers, layers placed from the source, extra bytes (this rank)."""
    return dict(_STATS)


def num_physical(layer: Any) -> int | None:
    return getattr(layer, "_rx_num_physical", None)


def copies_table(plan: LayerPlacement) -> torch.Tensor:
    """``[E, 1 + max copies]`` physical slot ids per logical expert.

    Column 0 is the native slot, further columns the replica slots, ``-1``
    where an expert has fewer copies.
    """
    extra: dict[int, list[int]] = {}
    for i, logical in enumerate(plan.rep_logical):
        extra.setdefault(logical, []).append(plan.alt_physical(i))
    width = 1 + max((len(v) for v in extra.values()), default=1)
    table = torch.full((plan.num_logical, width), -1, dtype=torch.int32)
    for logical in range(plan.num_logical):
        table[logical, 0] = plan.native_physical(logical)
        for k, p in enumerate(extra.get(logical, ())):
            table[logical, 1 + k] = p
    return table


def to_physical(
    layer: Any, topk_ids: torch.Tensor, topk_weights: torch.Tensor
) -> torch.Tensor:
    """Remap logical routed ids ``[T, K]`` to physical slot ids (traced code).

    A no-op for layers without a placement. Every expert a step touches runs
    on one of its copies, picked by ``kernels/moe_replica_assign.py``; batches
    above ``MAX_ROUTES`` routes use the native copies.
    """
    plan: LayerPlacement | None = getattr(layer, "_rx_plan", None)
    if plan is None:
        return topk_ids
    ids = topk_ids
    if envs.TPU_MOE_SKIP_PADDED_TOKENS:
        # Zero-weight rows are skipped by the grouped matmul; they must not
        # count as load (and stay skipped: -1 is outside every rank's range).
        ids = torch.where(topk_weights != 0, topk_ids, torch.full_like(topk_ids, -1))
    return remap_ids(ids, layer._rx_copies, ep=plan.ep, slots=plan.slots_total)


_assign_op = None


def _assign_jax(ids: jax.Array, copies: jax.Array, ep: int, slots: int) -> jax.Array:
    """jax_op body (arrays, then statics; annotations resolved by jax_op)."""
    from vllm_torchtpu.kernels.moe_replica_assign import replica_assign

    return replica_assign(ids, copies, ep=ep, slots=slots)


def _get_assign_op():
    global _assign_op
    if _assign_op is None:
        from torch_tpu._internal import pallas

        op = pallas.jax_op("pallas::moe_replica_assign", _assign_jax)
        op.register_fake(
            lambda ids, copies, ep, slots: torch.empty_like(ids, dtype=torch.int32)
        )
        _assign_op = op
    return _assign_op


def remap_ids(
    topk_ids: torch.Tensor, copies: torch.Tensor, *, ep: int, slots: int
) -> torch.Tensor:
    """Physical slot ids for logical ``topk_ids`` (see ``copies_table``)."""
    ids = topk_ids.to(torch.int32)
    if ids.device.type == "tpu":
        return _get_assign_op()(ids, copies, ep, slots)
    return remap_ids_reference(ids, copies, ep=ep, slots=slots)


def remap_ids_reference(
    topk_ids: torch.Tensor, copies: torch.Tensor, *, ep: int, slots: int
) -> torch.Tensor:
    """Host reference of the kernel's assignment (tests, non-TPU devices)."""
    ids = topk_ids.to(torch.int64)
    tab = copies.to(torch.int64).cpu()
    native = tab[:, 0].to(ids.device)[ids.clamp(min=0)]
    if ids.numel() > MAX_ROUTES or tab.shape[1] < 2:
        return torch.where(ids >= 0, native, -1).to(torch.int32)
    hit = list(dict.fromkeys(e for e in ids.flatten().tolist() if e >= 0))
    load = [0] * ep
    choice: dict[int, int] = {}
    flexible = []
    for e in hit:
        if int(tab[e, 1]) < 0:
            p = int(tab[e, 0])
            load[p // slots] += 1
            choice[e] = p
        else:
            flexible.append(e)
    for e in flexible:
        best = int(tab[e, 0])
        for k in range(1, tab.shape[1]):
            p = int(tab[e, k])
            if p >= 0 and load[p // slots] < load[best // slots]:
                best = p
        load[best // slots] += 1
        choice[e] = best
    out = torch.tensor(
        [choice[int(e)] if e >= 0 else -1 for e in ids.flatten().tolist()],
        dtype=torch.int32,
    )
    return out.view(ids.shape).to(topk_ids.device)


def placement_from_trace_lines(lines: Iterable[str], top: int = 64) -> dict[str, Any]:
    """Placement JSON from ``[moe-route-trace]`` lines: per layer, the
    ``top`` experts by the share of decode steps that touch them."""
    from vllm_torchtpu.layers.adapter.moe_route_trace import parse_line

    hits: list[dict[int, int]] = []
    steps = 0
    for line in lines:
        parsed = parse_line(line)
        if parsed is None or parsed[1] * 8 > MAX_ROUTES:
            continue
        _, _, layers = parsed
        steps += 1
        while len(hits) < len(layers):
            hits.append({})
        for j, d in enumerate(layers):
            for e in d:
                hits[j][e] = hits[j].get(e, 0) + 1
    out: dict[str, Any] = {}
    for j, h in enumerate(hits):
        items = sorted(h.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
        out[f"moe{j}"] = {str(e): round(100.0 * c / max(steps, 1), 2) for e, c in items}
    return out


def main(argv: list[str] | None = None) -> int:
    """``python -m vllm_torchtpu.layers.adapter.moe_redundant LOG [TOP]``.

    Prints a placement JSON built from the log's ``[moe-route-trace]`` lines.
    """
    import sys

    args = sys.argv[1:] if argv is None else argv
    if not args:
        print(main.__doc__)
        return 2
    top = int(args[1]) if len(args) > 1 else 16
    with open(args[0], errors="replace") as f:
        data = placement_from_trace_lines(f, top=top)
    print(json.dumps(data, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
