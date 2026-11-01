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
"""MoE / library-geometry weight helpers: the per-layer expert-copy table
(`extract_ecopy_layer`), the EP -> TP expert reshard and the TP MoE flags.

The generic weight source (`TorchArrayOps`, `VLLMWeightSource`,
`load_vllm_weights`, `find_backbone`) is `weight_source`, re-exported here.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

import torch

from vllm_torchtpu.megakernel import planner
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel_runtime.weight_source import (
    WEIGHT_FLIPPED_ATTR,
    TorchArrayOps,
    VLLMWeightSource,
    _need,
    _own,
    find_backbone,
    load_vllm_weights,
)
from vllm_torchtpu.megakernel_runtime.weight_source import (
    _orient_2d_kn as _orient_2d_kn,
)
from vllm_torchtpu.megakernel_runtime.weight_source import _stack as _stack


def _try_set(obj: Any, name: str, value: Any) -> None:
    """Set an existing attribute on `obj`, skipping read-only properties."""
    if hasattr(obj, name):
        with suppress(AttributeError):
            setattr(obj, name, value)


def get_routed_experts(mlp: Any) -> Any:
    """Return the underlying `FusedMoE` / `RoutedExperts` module on `mlp`."""
    experts = mlp.experts
    return getattr(experts, "routed_experts", experts)


def extract_ecopy_layer(
    layer_idx: int | None,
    layer: Any,
    g: Any,
    ex: ExecSpec,
    dev: torch.device,
    *,
    expert_placement: Any = None,
    experts_path: str = "mlp.experts",
) -> torch.Tensor | None:
    """Build the `[E]` `float32` `ecopy` descriptor vector for `smalls['ecopy']`.

    Returns `None` when no explicit `_ecopy`, `_ecopy_cores`, or matching
    `expert_placement` manifest is present on the module, so `smalls['ecopy']`
    remains all zeros (primary-only routing).
    """
    er = g.ER(ex.R)
    ew = g.EW(ex.R)
    if g.moe_tp or g.e_slots <= er:
        return None
    cur: Any = layer
    for attr in experts_path.split("."):
        cur = getattr(cur, attr, None)
        if cur is None:
            break
    routed = getattr(cur, "routed_experts", cur) if cur is not None else None
    if routed is not None:
        ec_buf = getattr(routed, "_ecopy", None)
        if ec_buf is not None:
            return ec_buf.to(device=dev, dtype=torch.float32).reshape(g.E)
        ec_cores = getattr(routed, "_ecopy_cores", None)
        if ec_cores is not None:
            return torch.tensor(
                planner.encode_ecopy(g.E, er, ew, g.moe_tp, ec_cores),
                dtype=torch.float32,
                device=dev,
            )
        if expert_placement is None:
            expert_placement = getattr(routed, "_expert_placement", None)
    if expert_placement is not None:
        if layer_idx is None and routed is not None:
            prefix = getattr(routed, "layer_name", "") or ""
            layer_idx = expert_placement.layer_index(prefix)
        if (
            layer_idx is not None
            and expert_placement.has_copies()
            and int(expert_placement.slots()) == int(ew)
            and int(expert_placement.global_count()) == int(g.E)
            and int(expert_placement.shards()) == int(ex.R)
            and str(layer_idx) in expert_placement.data().get("layers", {})
        ):
            cores = expert_placement.cores(layer_idx)
            return torch.tensor(
                planner.encode_ecopy(g.E, er, ew, g.moe_tp, cores),
                dtype=torch.float32,
                device=dev,
            )
    return None


def reshard_experts_ep_to_tp_local(
    w13_all: torch.Tensor,
    w2_all: torch.Tensor,
    s13_all: torch.Tensor | None,
    s2_all: torch.Tensor | None,
    tp_rank: int,
    g: Any,
    ex: ExecSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Slice global `E`-expert weights/scales into `tp_rank`'s TP-sharded tensors
    (scales `None` for bf16 experts)."""
    ir = g.I // ex.R
    g0 = tp_rank * ir
    u0 = g.I + tp_rank * ir

    w13_tp = _own(
        torch.cat([w13_all[:, :, g0 : g0 + ir], w13_all[:, :, u0 : u0 + ir]], dim=-1)
    )
    w2_tp = _own(w2_all[:, g0 : g0 + ir, :])
    if s13_all is None or s2_all is None:
        return w13_tp, w2_tp, None, None

    s13_flat = s13_all.float().reshape(g.E, 2 * g.I)
    s2_flat = s2_all.float().reshape(g.E, g.H)
    s13_tp = _own(
        torch.cat(
            [s13_flat[:, g0 : g0 + ir], s13_flat[:, u0 : u0 + ir]], dim=-1
        ).reshape(g.E, 1, 1, 2 * ir)
    )
    s2_tp = _own(s2_flat.reshape(g.E, 1, 1, g.H))
    return w13_tp, w2_tp, s13_tp, s2_tp


def apply_tp_moe_flags(experts: Any, *, prebuild_moe_fn: Any = None) -> None:
    """Update a `FusedMoE` module's flags after in-place EP->TP resharding."""
    targets = [experts]
    routed = getattr(experts, "routed_experts", None)
    if routed is not None and routed is not experts:
        targets.append(routed)
    for tgt in targets:
        moe_cfg = getattr(tgt, "moe_config", None)
        if moe_cfg is not None:
            p_cfg = getattr(moe_cfg, "moe_parallel_config", None)
            if p_cfg is not None:
                _try_set(p_cfg, "use_ep", False)
                if (
                    getattr(p_cfg, "ep_size", 1) > 1
                    and getattr(p_cfg, "tp_size", 1) == 1
                ):
                    _try_set(p_cfg, "tp_size", p_cfg.ep_size)
                    _try_set(p_cfg, "tp_rank", getattr(p_cfg, "ep_rank", 0))
                    _try_set(p_cfg, "ep_size", 1)
                    _try_set(p_cfg, "ep_rank", 0)
                _try_set(p_cfg, "sp_size", 1)
                _try_set(p_cfg, "is_sequence_parallel", False)
            _try_set(moe_cfg, "use_ep", False)
            _try_set(moe_cfg, "is_sequence_parallel", False)
        _try_set(tgt, "_tpu_fused_ep_eligible", False)
        for buf in ("_experts_start", "_expert_map", "expert_mask"):
            if buf in getattr(tgt, "_buffers", {}):
                tgt.register_buffer(buf, None, persistent=False)
            else:
                _try_set(tgt, buf, None)
        _try_set(tgt, "_gmm_primary_experts", None)
        _try_set(tgt, "_lowtok_has_copies", False)
        if hasattr(tgt, "global_num_experts"):
            _try_set(tgt, "local_num_experts", tgt.global_num_experts)
        for qm_attr in ("quant_method", "_quant_method"):
            qm = getattr(tgt, qm_attr, None)
            if qm is not None:
                _try_set(qm, "_tpu_fused_ep_eligible", False)
                _try_set(qm, "_moe_lowtok_scores_op", None)
                _try_set(qm, "_fused_moe_ep_op", None)
                _try_set(qm, "_use_adaptive_fused_moe", False)
        if (
            prebuild_moe_fn is not None
            and moe_cfg is not None
            and hasattr(moe_cfg, "experts_per_token")
            and hasattr(tgt, "activation")
        ):
            act_val = getattr(tgt.activation, "value", tgt.activation)
            prebuild_moe_fn(
                topk=moe_cfg.experts_per_token,
                activation=act_val,
                use_ep=False,
            )


def _expert_scale_attr(experts: Any, weight: str) -> str:
    for attr in (f"{weight}_scale_inv", f"{weight}_scale"):
        if hasattr(experts, attr):
            return attr
    raise ValueError(f"FP8 experts are missing {weight}_scale (or _scale_inv)")


def reshard_layer_experts_ep_to_tp(
    mlp: Any,
    tp_rank: int,
    g: Any,
    ex: ExecSpec,
    *,
    all_gather_fn: Any = None,
    prebuild_moe_fn: Any = None,
) -> None:
    """Reshard `mlp.experts` from EP (`[ER, H, 2*I]`) to TP (`[E, H, 2*IR]`)."""
    experts = get_routed_experts(mlp)
    er = g.ER(ex.R)
    w13_ep = experts.w13_weight.data[:er].contiguous()
    w2_ep = experts.w2_weight.data[:er].contiguous()

    # bf16 (unquantized) experts carry no scales; fp8 experts must.
    fp8 = w13_ep.dtype == torch.float8_e4m3fn
    _need(
        fp8 == (w2_ep.dtype == torch.float8_e4m3fn),
        f"w13/w2 expert dtypes differ ({w13_ep.dtype} vs {w2_ep.dtype})",
    )
    s13_attr = s2_attr = ""
    s13_ep = s2_ep = None
    if fp8:
        s13_attr = _expert_scale_attr(experts, "w13_weight")
        s2_attr = _expert_scale_attr(experts, "w2_weight")
        s13_ep = getattr(experts, s13_attr).data[:er].reshape(er, 2 * g.I).contiguous()
        s2_ep = getattr(experts, s2_attr).data[:er].reshape(er, g.H).contiguous()

    if all_gather_fn is None:
        _need(
            ex.R == 1,
            "all_gather_fn must be provided to reshard_layer_experts_ep_to_tp "
            "when R > 1",
        )
        w13_all, w2_all, s13_all, s2_all = w13_ep, w2_ep, s13_ep, s2_ep
    else:
        w13_all = all_gather_fn(w13_ep)
        w2_all = all_gather_fn(w2_ep)
        s13_all = all_gather_fn(s13_ep) if fp8 else None
        s2_all = all_gather_fn(s2_ep) if fp8 else None

    w13_tp, w2_tp, s13_tp, s2_tp = reshard_experts_ep_to_tp_local(
        w13_all, w2_all, s13_all, s2_all, tp_rank=tp_rank, g=g, ex=ex
    )
    del w13_all, w2_all, s13_all, s2_all
    del w13_ep, w2_ep, s13_ep, s2_ep

    experts.w13_weight = torch.nn.Parameter(w13_tp, requires_grad=False)
    experts.w2_weight = torch.nn.Parameter(w2_tp, requires_grad=False)
    synced = [experts.w13_weight, experts.w2_weight]
    if s13_tp is not None and s2_tp is not None:
        s13_new_param = torch.nn.Parameter(s13_tp, requires_grad=False)
        s2_new_param = torch.nn.Parameter(s2_tp, requires_grad=False)
        setattr(experts, s13_attr, s13_new_param)
        setattr(experts, s2_attr, s2_new_param)
        synced += [s13_new_param, s2_new_param]

    if w13_tp.device.type == "tpu":
        from torch_tpu._internal import sync

        sync.synchronize(tensors=synced, wait=True)

    apply_tp_moe_flags(mlp.experts, prebuild_moe_fn=prebuild_moe_fn)


__all__ = [
    "TorchArrayOps",
    "VLLMWeightSource",
    "WEIGHT_FLIPPED_ATTR",
    "apply_tp_moe_flags",
    "extract_ecopy_layer",
    "find_backbone",
    "get_routed_experts",
    "load_vllm_weights",
    "reshard_experts_ep_to_tp_local",
    "reshard_layer_experts_ep_to_tp",
]
