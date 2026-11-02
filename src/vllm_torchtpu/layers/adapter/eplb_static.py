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
"""Static EPLB for the fused EP MoE kernel: redundant copies of hot experts.

The fused kernel splits the experts into one equal contiguous block per EP
rank, and every rank waits at the in-kernel barrier for the busiest one. With
concentrated routing that rank sets the layer time. This module gives every
rank ``R`` extra *physical* expert slots, filled at weight-load time with
copies of hot logical experts, and remaps the router's logical top-k ids to
physical ids. Remapping is locality-first: a row routed from EP rank ``s`` uses
the copy of its expert on rank ``s`` when there is one (no ICI traffic for that
pair), otherwise one of the expert's copies chosen by ``s`` (spreads remote
traffic over the replicas). Replicating the hottest experts on every rank
therefore both balances the destination ranks and keeps their rows local.

The kernel body is untouched: it sees ``E_phys = E + ep * R`` experts and an
equal split of ``G + R`` per rank (``G = E / ep``), contiguous.

Physical layout, per EP rank ``r`` (vLLM's linear placement for the home
experts):

    phys r * (G + R) + j        home expert  r * G + j          (j < G)
    phys r * (G + R) + G + s    replica slot s of rank r        (s < R)

Enabled only when both env vars are set (default OFF):

    DSV41_EPLB_SLOTS_PER_RANK=<R>
    DSV41_EPLB_PLAN=<plan.json>

Plan JSON (``scripts/dsv41_bench/static_eplb_plan.py`` builds one from routing
dumps)::

    {"num_experts": 384, "ep": 16, "slots_per_rank": 1,
     "layers": {"<layer>": [[logical_expert, target_rank], ...], ...}}

``<layer>`` is the decoder layer index parsed from the MoE layer's name
(``...layers.<N>....``). A rank may receive at most ``R`` replicas per layer;
unused slots get a copy of one of the rank's own home experts and are never
routed to. Layers missing from the plan get only such idle slots.

Correctness does not depend on the plan: a replica slot holds bit-identical
weights (same checkpoint bytes, same per-layer requantization) as its logical
expert, so the per-token result is the same up to accumulation order.
"""

from __future__ import annotations

import functools
import json
import os
import re
from dataclasses import dataclass

import numpy as np

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

SLOTS_ENV = "DSV41_EPLB_SLOTS_PER_RANK"
PLAN_ENV = "DSV41_EPLB_PLAN"


def slots_per_rank() -> int:
    try:
        return max(int(os.environ.get(SLOTS_ENV, "0") or 0), 0)
    except ValueError:
        return 0


def enabled() -> bool:
    return slots_per_rank() > 0 and bool(os.environ.get(PLAN_ENV, ""))


@functools.cache
def load_plan(path: str | None = None) -> dict:
    path = path or os.environ[PLAN_ENV]
    with open(path) as fh:
        plan = json.load(fh)
    layers = {
        int(k): [(int(e), int(r)) for e, r in v] for k, v in plan["layers"].items()
    }
    return {
        "num_experts": int(plan["num_experts"]),
        "ep": int(plan["ep"]),
        "slots_per_rank": int(plan.get("slots_per_rank", slots_per_rank())),
        "layers": layers,
    }


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def layer_index(layer_name: str) -> int | None:
    m = _LAYER_RE.search(layer_name or "")
    return int(m.group(1)) if m else None


@dataclass(frozen=True)
class EplbSpec:
    """One layer's replica placement. ``slots[r][s]`` is the logical expert
    held by replica slot ``s`` of EP rank ``r``, or -1 for an idle slot."""

    num_experts: int
    ep: int
    r: int
    slots: tuple[tuple[int, ...], ...]
    layer: int | None = None

    @property
    def g(self) -> int:
        return self.num_experts // self.ep

    @property
    def num_physical(self) -> int:
        return self.num_experts + self.ep * self.r

    def home_phys(self, e: int) -> int:
        return (e // self.g) * (self.g + self.r) + e % self.g

    def slot_phys(self, rank: int, s: int) -> int:
        return rank * (self.g + self.r) + self.g + s

    def local_sources(self, rank: int) -> list[int]:
        """Logical expert to load into each of ``rank``'s replica slots.
        Idle slots copy one of the rank's own home experts (always local)."""
        return [
            e if e >= 0 else rank * self.g + (s % self.g)
            for s, e in enumerate(self.slots[rank])
        ]

    def replicas(self) -> dict[int, list[int]]:
        """logical expert -> physical ids, home first."""
        out = {e: [self.home_phys(e)] for e in range(self.num_experts)}
        for rank in range(self.ep):
            for s, e in enumerate(self.slots[rank]):
                if e >= 0:
                    out[e].append(self.slot_phys(rank, s))
        return out

    def table(self) -> np.ndarray:
        """int32 ``[E, ep]``: entry ``[e, s]`` is the physical id that rows
        routed from EP rank ``s`` use for logical expert ``e`` -- the copy on
        rank ``s`` (home or replica) when there is one, else copy ``s % n``.
        The shape depends only on (E, ep), so all layers share one op."""
        t = np.zeros((self.num_experts, self.ep), np.int32)
        for e, phys in self.replicas().items():
            for src in range(self.ep):
                local = [p for p in phys if p // (self.g + self.r) == src]
                t[e, src] = local[0] if local else phys[src % len(phys)]
        return t


def build_spec(
    num_experts: int, ep: int, r: int, replicas: list[tuple[int, int]], layer=None
) -> EplbSpec:
    if num_experts % ep:
        raise ValueError(f"EPLB: {num_experts} experts do not split over ep={ep}")
    slots = [[] for _ in range(ep)]
    for e, rank in replicas:
        if not (0 <= e < num_experts and 0 <= rank < ep):
            raise ValueError(f"EPLB layer {layer}: bad replica ({e}, {rank})")
        if len(slots[rank]) >= r:
            raise ValueError(
                f"EPLB layer {layer}: rank {rank} gets more than {r} replicas"
            )
        slots[rank].append(e)
    for s in slots:
        s.extend([-1] * (r - len(s)))
    return EplbSpec(num_experts, ep, r, tuple(tuple(s) for s in slots), layer)


def spec_for(layer_name: str, num_experts: int, ep: int) -> EplbSpec | None:
    """The spec for a MoE layer, or None when EPLB does not apply to it."""
    if not enabled():
        return None
    plan = load_plan()
    r = slots_per_rank()
    if plan["num_experts"] != num_experts or plan["ep"] != ep:
        logger.warning_once(
            "Static EPLB not applied to %s: plan is for E=%d ep=%d, layer has "
            "E=%d ep=%d.",
            layer_name,
            plan["num_experts"],
            plan["ep"],
            num_experts,
            ep,
        )
        return None
    if plan["slots_per_rank"] != r:
        raise ValueError(
            f"{SLOTS_ENV}={r} but the EPLB plan was made for "
            f"slots_per_rank={plan['slots_per_rank']}"
        )
    idx = layer_index(layer_name)
    return build_spec(num_experts, ep, r, plan["layers"].get(idx, []), idx)


def wrap_weight_loader(layer, weight_loader, spec: EplbSpec, ep_rank: int):
    """Also load each checkpoint expert into this rank's replica slots.

    vLLM's RoutedExperts.weight_loader maps a global expert id to a local slot
    through ``layer._map_global_expert_id_to_local_expert_id``; for a replica
    the same loader runs again with that lookup pinned to the replica slot, so
    every per-shard rule (w1/w3 halves, TP narrowing, scales) is reused.
    """
    g = spec.g
    by_expert: dict[int, list[int]] = {}
    for s, e in enumerate(spec.local_sources(ep_rank)):
        by_expert.setdefault(e, []).append(g + s)

    def loader(
        param,
        loaded_weight,
        weight_name=None,
        shard_id=None,
        expert_id=None,
        return_success=False,
        **kw,
    ):
        if expert_id is None:
            return weight_loader(
                param,
                loaded_weight,
                weight_name,
                shard_id,
                return_success=return_success,
                **kw,
            )
        ok = weight_loader(
            param,
            loaded_weight,
            weight_name,
            shard_id=shard_id,
            expert_id=expert_id,
            return_success=True,
            **kw,
        )
        for local in by_expert.get(int(expert_id), ()):
            layer._map_global_expert_id_to_local_expert_id = lambda _e, _local=local: (
                _local
            )
            try:
                ok_r = weight_loader(
                    param,
                    loaded_weight,
                    weight_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                    **kw,
                )
            finally:
                del layer._map_global_expert_id_to_local_expert_id
            ok = bool(ok) or bool(ok_r)
        return ok if return_success else None

    return loader


_FILTER_PATCHED = False


def extend_ep_weight_filter() -> None:
    """Make ``--enable-ep-weight-filter`` (and the Run:AI sharded fetch) also
    keep the logical experts this rank replicates, in any planned layer."""
    global _FILTER_PATCHED
    if _FILTER_PATCHED or not enabled():
        return
    from vllm.model_executor.model_loader import default_loader

    original = default_loader.compute_local_expert_ids

    def with_replicas(num_experts, ep_size, ep_rank, placement="linear"):
        ids = original(num_experts, ep_size, ep_rank, placement=placement)
        if ids is None:
            return ids
        plan = load_plan()
        if plan["num_experts"] != num_experts or plan["ep"] != ep_size:
            return ids
        extra = set()
        for idx, reps in plan["layers"].items():
            spec = build_spec(num_experts, ep_size, slots_per_rank(), reps, idx)
            extra.update(spec.local_sources(ep_rank))
        return set(ids) | extra

    default_loader.compute_local_expert_ids = with_replicas
    _FILTER_PATCHED = True


def remap_topk_ids(topk_idx, table, src_rank):
    """JAX: logical top-k ids ``[rows, k]`` of rows routed from EP rank
    ``src_rank`` -> physical ids via ``table`` (see ``EplbSpec.table``). One
    ``[rows, k]`` gather. Ids outside ``[0, E)`` (the kernel's empty-row
    sentinel) pass through; the router weight stays with its top-k slot."""
    import jax.numpy as jnp

    e = table.shape[0]
    valid = (topk_idx >= 0) & (topk_idx < e)
    safe = jnp.where(valid, topk_idx, 0)
    return jnp.where(valid, table[safe, src_rank], topk_idx).astype(topk_idx.dtype)
