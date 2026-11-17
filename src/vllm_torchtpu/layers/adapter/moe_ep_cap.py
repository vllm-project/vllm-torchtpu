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
"""Decode-step EP load cap for the grouped-matmul MoE path (``TPU_MOE_EP_CAP``).

With linear EP placement, rank ``r`` owns experts ``[r*L, (r+1)*L)``. In a
decode step every rank streams the weights of each local expert it is routed
to, and the MoE all-reduce waits for the rank with the most distinct experts.
Served routing is uneven per step (the tokens of one request share experts),
so the busiest rank holds several more experts than the mean.

On, in steps of at most ``max_tokens`` tokens (an in-graph mask), every rank keeps at most
``ceil(D / ep) + slack`` of its hit experts (``D`` = experts hit by the step),
ranked by routed weight; an expert that is some token's top-1 is always kept.
Routes to an expert that is not kept are

* dropped (``drop``, default): the token's remaining weights are scaled back to
  its original weight sum, or
* substituted (``sub<N>``): moved to the token's best already-kept expert that
  is in its top-``N`` by selection score (sigmoid score + correction bias) and
  not already one of its experts; its weight is the token's renormalized sigmoid
  score. Routes without such a candidate are dropped as above.

Routing is computed redundantly on every TP rank from the same hidden states,
so all ranks take the same decision. Tokens whose routes are all kept keep
their weights bitwise. The cap changes numerics (an approximation of top-k
routing) and is off by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_PROTECT = 1.0e6  # key offset of an expert that is some token's top-1


@dataclass(frozen=True)
class CapSpec:
    slack: int
    mode: str  # "drop" | "sub"
    sub_topn: int
    max_tokens: int
    in_graph: bool = False  # True: mask large steps in-graph (computes everywhere)


def parse_spec(spec: str) -> CapSpec | None:
    """``"<slack>[:drop|:sub<N>][@<max_tokens>]"`` -> CapSpec; "" -> None."""
    spec = spec.strip()
    if not spec:
        return None
    max_tokens, in_graph = 128, False
    if "@" in spec:
        spec, _, mt = spec.partition("@")
        if mt.startswith("g"):
            in_graph, mt = True, mt[1:]
        max_tokens = int(mt)
    mode, sub_topn = "drop", 0
    if ":" in spec:
        spec, _, m = spec.partition(":")
        if m == "drop":
            pass
        elif m.startswith("sub"):
            mode, sub_topn = "sub", int(m[3:] or 8)
        else:
            raise ValueError(f"TPU_MOE_EP_CAP: unknown mode {m!r}")
    slack = int(spec)
    if slack < 0 or max_tokens <= 0 or (mode == "sub" and sub_topn < 1):
        raise ValueError(f"TPU_MOE_EP_CAP: bad spec {spec!r}")
    return CapSpec(
        slack=slack, mode=mode, sub_topn=sub_topn, max_tokens=max_tokens,
        in_graph=in_graph,
    )


SPEC: CapSpec | None = parse_spec(envs.TPU_MOE_EP_CAP)
ENABLED: bool = SPEC is not None
STATS_EVERY: int = int(envs.TPU_MOE_EP_CAP_STATS) if ENABLED else 0

# [layer-steps, routes, routes dropped, routes substituted, weight, weight
#  dropped, sum busiest-rank experts before, after]
_NSTAT = 8
_stats: torch.Tensor | None = None
_logged = False
_hooked = False
_calls = 0


def _is_target_layer(name: str) -> bool:
    return ".layers." in name and "draft" not in name and "dspark" not in name


def ep_size_of(layer: Any) -> int:
    moe_config = getattr(layer, "moe_config", None)
    if moe_config is None or not moe_config.moe_parallel_config.use_ep:
        return 1
    return int(moe_config.ep_size)


def applies(layer: Any) -> bool:
    """Static (trace-time) gate: on, EP, linear placement, target model.

    Deliberately independent of the token count: a python branch on it would
    split the traced backbone per bucket (compilation/shape_variants.py). The
    ``max_tokens`` limit is applied in-graph instead (``cap_routes``)."""
    if SPEC is None:
        return False
    if getattr(layer, "_rx_plan", None) is not None:  # redundant experts own it
        return False
    ep = ep_size_of(layer)
    if ep <= 1 or int(layer.global_num_experts) % ep:
        return False
    return _is_target_layer(str(getattr(layer, "layer_name", "") or ""))


def cap_routes(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    ep: int,
    slack: int,
    mode: str = "drop",
    sub_topn: int = 0,
    max_tokens: int | None = None,
    router_logits: torch.Tensor | None = None,
    score_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cap per-rank distinct experts. Returns ``(ids, weights, stats[8])``.

    ``topk_ids`` [T, K] global expert ids (linear placement, ``num_experts / ep``
    per rank); ``topk_weights`` [T, K]. Rows of weight 0 (padding) are dropped
    (their outputs are discarded anyway). Dropped routes get id -1, weight 0.
    """
    num_tokens, k = topk_ids.shape
    per_rank = num_experts // ep
    device = topk_ids.device
    w = topk_weights.float()
    ids = topk_ids.long()
    valid = (ids >= 0) & (w != 0)
    safe_ids = torch.where(valid, ids, torch.zeros_like(ids))
    # One-hot compares, not index_add: a TPU scatter costs ~10-17 us per call at
    # decode sizes, the fused compare-and-reduce ~2-3 us for the whole cap.
    experts = torch.arange(num_experts, device=device)
    hit = (safe_ids.unsqueeze(-1) == experts) & valid.unsqueeze(-1)  # [T, K, E]
    onehot = hit.float()
    count = onehot.sum(dim=(0, 1))
    mass = (onehot * w.unsqueeze(-1)).sum(dim=(0, 1))
    wmax = torch.where(valid, w, torch.full_like(w, -1.0)).amax(dim=-1, keepdim=True)
    top1 = valid & (w == wmax)
    protected = (onehot * top1.unsqueeze(-1).float()).amax(dim=(0, 1)) > 0
    active = count > 0
    key = torch.where(active, mass + _PROTECT * protected.float(), -1.0)
    kr = key.view(ep, per_rank)
    lane = torch.arange(per_rank, device=device)
    above = (kr.unsqueeze(1) > kr.unsqueeze(2)) | (
        (kr.unsqueeze(1) == kr.unsqueeze(2)) & (lane.view(1, 1, -1) < lane.view(1, -1, 1))
    )
    pos = above.sum(dim=-1).view(num_experts)  # experts of the rank ranked above e
    distinct = active.sum()
    cap = torch.div(distinct + (ep - 1), ep, rounding_mode="floor") + slack
    keep = active & ((pos < cap) | protected)
    if max_tokens is not None:
        # In-graph (no python branch on the token count): larger steps keep all.
        tokens = torch.ones_like(ids[:, 0]).sum()
        keep = keep | (active & (tokens > max_tokens))
    route_keep = valid & keep[safe_ids]
    dropped = valid & ~route_keep

    w_sum = torch.where(valid, w, 0.0).sum(dim=-1, keepdim=True)
    new_ids = torch.where(route_keep, ids, torch.full_like(ids, -1))
    substituted = torch.zeros_like(dropped)
    if mode == "sub":
        assert router_logits is not None
        scores = router_logits.float().sigmoid()  # [T, E]
        select = scores if score_bias is None else scores + score_bias.float()
        thresh = torch.topk(select, sub_topn, dim=-1).values[:, -1:]
        mine = hit.any(dim=1)  # [T, E] the token's own (valid) experts
        cand_ok = keep.unsqueeze(0) & ~mine & (select >= thresh)
        cand_sel = torch.where(cand_ok, select, torch.full_like(select, -1.0e30))
        cand_val, cand_idx = torch.topk(cand_sel, k, dim=-1)  # [T, K]
        order = torch.cumsum(dropped.long(), dim=-1) - 1  # j-th dropped route
        order = order.clamp(min=0)
        pick = torch.gather(cand_idx, 1, order)
        pick_ok = torch.gather(cand_val, 1, order) > -1.0e29
        substituted = dropped & pick_ok
        new_ids = torch.where(substituted, pick, new_ids)
        live = new_ids >= 0
        s_new = torch.gather(scores, 1, torch.where(live, new_ids, 0))
        s_new = torch.where(live, s_new, 0.0)
        w_new = s_new * (w_sum / s_new.sum(dim=-1, keepdim=True).clamp(min=1e-20))
    else:
        w_kept = torch.where(route_keep, w, 0.0)
        w_new = w_kept * (w_sum / w_kept.sum(dim=-1, keepdim=True).clamp(min=1e-20))
    changed = dropped.any(dim=-1, keepdim=True)
    w_out = torch.where(changed, w_new, w).to(topk_weights.dtype)
    w_out = torch.where(new_ids >= 0, w_out, torch.zeros_like(w_out))

    before = active.view(ep, per_rank).sum(dim=-1).amax()
    # Substitutes land on kept experts, so the kept set is the hit set after.
    after = keep.view(ep, per_rank).sum(dim=-1).amax()
    stats = torch.stack(
        [
            torch.ones((), device=device),
            valid.float().sum(),
            dropped.float().sum(),
            substituted.float().sum(),
            w_sum.sum(),
            torch.where(dropped, w, 0.0).sum(),
            before.float(),
            after.float(),
        ]
    )
    return new_ids.to(topk_ids.dtype), w_out, stats


def apply(
    layer: Any,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    router_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Traced-code entry from a quant method's ``apply_monolithic``."""
    assert SPEC is not None
    if not SPEC.in_graph and topk_ids.shape[0] > SPEC.max_tokens:
        # Trace-time branch on the token count (compilation/shape_variants.py
        # splits the buckets between traces). Both sides touch the stats buffer
        # so every trace has the same inputs.
        if _stats is not None:
            _stats.add_(torch.zeros_like(_stats))
        return topk_ids, topk_weights
    ids, weights, stats = cap_routes(
        topk_ids,
        topk_weights,
        num_experts=int(layer.global_num_experts),
        ep=ep_size_of(layer),
        slack=SPEC.slack,
        mode=SPEC.mode,
        sub_topn=SPEC.sub_topn,
        max_tokens=SPEC.max_tokens if SPEC.in_graph else None,
        router_logits=router_logits,
        score_bias=getattr(layer, "e_score_correction_bias", None),
    )
    if _stats is not None:
        _stats.add_(stats)
    return ids, weights


def register(layer: Any) -> None:
    """Proof line and stats buffer. Call from ``process_weights_after_loading``
    (eager code: the traced ``apply`` must not log)."""
    global _stats
    _log_once(layer)
    if STATS_EVERY <= 0 or _stats is not None:
        return
    _stats = torch.zeros((_NSTAT,), dtype=torch.float32, device=layer.w13_weight.device)
    _install_runner_hook()


def _log_once(layer: Any) -> None:
    global _logged
    if _logged:
        return
    _logged = True
    logger.info(
        "[moe-ep-cap] TPU_MOE_EP_CAP=%s on: slack=%d mode=%s sub_topn=%d "
        "max_tokens=%d E=%d ep=%d stats_every=%d",
        envs.TPU_MOE_EP_CAP,
        SPEC.slack,
        SPEC.mode,
        SPEC.sub_topn,
        SPEC.max_tokens,
        int(layer.global_num_experts),
        ep_size_of(layer),
        STATS_EVERY,
    )


def _tp_rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:
        return 0


def dump() -> None:
    if _stats is None or _tp_rank() != 0:
        return
    n, routes, drop, sub, wt, wdrop, before, after = _stats.cpu().tolist()
    if n <= 0:
        return
    logger.info(
        "[moe-ep-cap-stats] layer_steps=%d routes=%d dropped=%.4f%% substituted=%.4f%% "
        "weight_dropped=%.4f%% busiest_rank_experts before=%.3f after=%.3f",
        int(n),
        int(routes),
        100.0 * drop / max(routes, 1.0),
        100.0 * sub / max(routes, 1.0),
        100.0 * wdrop / max(wt, 1e-20),
        before / n,
        after / n,
    )


def _install_runner_hook() -> None:
    global _hooked
    if _hooked:
        return
    from vllm_torchtpu.runner import tpu_runner

    cls = tpu_runner.TPUModelRunner
    orig = cls.execute_model

    def execute_model(self, *args, **kwargs):
        global _calls
        out = orig(self, *args, **kwargs)
        _calls += 1
        if _calls % STATS_EVERY == 0:
            try:
                dump()
            except Exception as exc:  # diagnostics must never kill serving
                logger.warning_once("[moe-ep-cap] stats dump failed: %s", exc)
        return out

    cls.execute_model = execute_model
    _hooked = True
