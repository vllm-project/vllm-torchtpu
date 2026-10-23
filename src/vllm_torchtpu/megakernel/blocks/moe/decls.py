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
"""`MoEBlock` geometry and parameter declarations (`moe_hf_params`)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...decls import (
    REQUIRED,
    BlockBinding,
    HFParamDecl,
    ParamTransform,
    ResolvedBlockGeometry,
    VLLMAlias,
    apply_decl_overrides,
)
from ...errors import MegakernelConfigError
from ...ops import BF16, F32, FP8, rup

MOE_PARAM_NAMES: frozenset[str] = frozenset(
    {
        "router",
        "sh_gate",
        "sh_gate_proj",
        "sh_gate_proj_s",
        "sh_up_proj",
        "sh_up_proj_s",
        "sh2",
        "sh2_s",
        "e_gate_proj",
        "e_gate_proj_s",
        "e_up_proj",
        "e_up_proj_s",
        "e2",
        "e2_s",
    }
)

MOE_GEOMETRY_FIELDS: dict[str, tuple[str, Any]] = {
    "hidden": ("hidden_size", REQUIRED),
    "experts": ("num_experts", REQUIRED),
    "top_k": ("num_experts_per_tok", REQUIRED),
    "inter": ("moe_intermediate_size", REQUIRED),
    "shared_inter": ("shared_expert_intermediate_size", REQUIRED),
    "eps": ("rms_norm_eps", 1e-6),
}

MOE_FIELD_ALIASES: dict[str, str] = {
    "H": "hidden",
    "hidden_size": "hidden",
    "E": "experts",
    "num_experts": "experts",
    "TOPK": "top_k",
    "num_experts_per_tok": "top_k",
    "I": "inter",
    "moe_intermediate_size": "inter",
    "IS": "shared_inter",
    "shared_expert_intermediate_size": "shared_inter",
    "rms_norm_eps": "eps",
}

MOE_OPTIONS: dict[str, Any] = {
    "shared_gate": True,
    "moe_tp": None,
    "e_slots": None,
    "expert_quant": None,
}


def nvfp4_raw_n(n_logical: int) -> int:
    """Physical uint8 columns for `n_logical` NVFP4 channels: `n + rup(n // 8, 128)`."""
    return int(n_logical) + rup(int(n_logical) // 8, 128)


def _moe_ew(g: Any, R: int) -> int:
    e_slots = int(g.options.get("e_slots", g.values.get("e_slots", 0)))
    if e_slots > 0:
        return e_slots
    moe_tp = bool(g.options.get("moe_tp", g.values.get("moe_tp", True)))
    return int(g.E) if moe_tp else int(g.E) // R


def _moe_sh_rep(g: Any, R: int) -> int:
    return max(1, R // max(1, int(g.IS) // 128))


MOE_GEOMETRY_METHODS: dict[str, Any] = {
    "ER": lambda g, R: int(g.E) // R,
    "sh_rep": _moe_sh_rep,
    "ISR": lambda g, R: int(g.IS) // max(1, R // _moe_sh_rep(g, R)),
    "IR": lambda g, R: (
        int(g.I) // R
        if bool(g.options.get("moe_tp", g.values.get("moe_tp", True)))
        else int(g.I)
    ),
    "EW": _moe_ew,
}


def validate_moe_options(options: Mapping[str, Any]) -> None:
    if not bool(options.get("shared_gate", True)):
        raise MegakernelConfigError(
            "MoEBlock requires shared_gate=True (the kernel always gates the "
            "shared expert with sh_gate)"
        )
    eq = options.get("expert_quant")
    if eq is not None and str(eq) not in ("fp8", "nvfp4"):
        raise MegakernelConfigError(
            f"MoEBlock requires expert_quant in ('fp8', 'nvfp4'), got {eq!r}"
        )


def resolve_moe_binding(
    cls: Any, binding: BlockBinding, resolved: ResolvedBlockGeometry
) -> ResolvedBlockGeometry:
    vals = dict(resolved.values)
    opts = {k: v for k, v in binding.options.items() if v is not None}
    vals.update(
        {
            "H": int(vals["hidden"]),
            "E": int(vals["experts"]),
            "TOPK": int(vals["top_k"]),
            "I": int(vals["inter"]),
            "IS": int(vals["shared_inter"]),
            "eps": float(vals["eps"]),
            "moe_tp": bool(opts.get("moe_tp", True)),
            "e_slots": int(opts.get("e_slots", 0)),
            "expert_quant": str(opts.get("expert_quant", "fp8")),
        }
    )
    return ResolvedBlockGeometry(
        block_cls=cls,
        prefix=binding.prefix,
        values=vals,
        options=opts,
        rename=binding.rename,
        transforms=binding.transforms,
    )


def moe_hf_params(
    prefix: str,
    geom: Any,
    *,
    shared_gate: bool = True,
    moe_tp: bool | None = None,
) -> list[HFParamDecl]:
    """Return canonical (unfused) HF parameter declarations for `MoEBlock`."""
    p = f"{prefix}." if prefix else ""
    H, E, IS = int(geom.H), int(geom.E), int(geom.IS)
    I = int(geom.I)  # noqa: E741
    moe_tp = bool(getattr(geom, "moe_tp", True)) if moe_tp is None else bool(moe_tp)
    shard_e13 = -1 if moe_tp else 0
    shard_e2 = 1 if moe_tp else 0
    shard_e2_s = None if moe_tp else 0
    rename = geom.rename if isinstance(geom, ResolvedBlockGeometry) else {}

    def _n(canon: str) -> str:
        return rename.get(canon, canon)

    def _sh_gu_pair(
        canon: str, hf_stem: str, order: int, aliases: tuple[VLLMAlias, ...] = ()
    ) -> list[HFParamDecl]:
        w_n, s_n = _n(canon), _n(f"{canon}_s")
        return [
            HFParamDecl(
                name=w_n,
                hf_name=f"{p}shared_expert.{hf_stem}.weight",
                shape=(H, IS),
                hf_shape=(IS, H),
                dtype=FP8,
                transform=ParamTransform(
                    fuse_into="sh13",
                    fuse_order=order,
                    fuse_axis=-1,
                    transpose=(1, 0),
                    quantize="fp8",
                    shard_axis=-1,
                ),
                vllm_aliases=aliases,
            ),
            HFParamDecl(
                name=s_n,
                hf_name=f"{p}shared_expert.{hf_stem}.weight_scale_inv",
                shape=(IS,),
                hf_shape=(IS,),
                dtype=F32,
                transform=ParamTransform(
                    fuse_into="sh13_s",
                    fuse_order=order,
                    fuse_axis=-1,
                    scale_for=w_n,
                    shard_axis=-1,
                ),
            ),
        ]

    def _exp_gu_pair(
        canon: str, hf_stem: str, order: int, aliases: tuple[VLLMAlias, ...] = ()
    ) -> list[HFParamDecl]:
        w_n, s_n = _n(canon), _n(f"{canon}_s")
        return [
            HFParamDecl(
                name=w_n,
                hf_name=f"{p}experts.{hf_stem}.weight",
                shape=(E, H, I),
                hf_shape=(E, I, H),
                dtype=FP8,
                transform=ParamTransform(
                    fuse_into="e13",
                    fuse_order=order,
                    fuse_axis=-1,
                    transpose=(0, 2, 1),
                    quantize="fp8",
                    shard_axis=shard_e13,
                    expert_axis=0,
                ),
                vllm_aliases=aliases,
            ),
            HFParamDecl(
                name=s_n,
                hf_name=f"{p}experts.{hf_stem}.weight_scale_inv",
                shape=(E, I),
                hf_shape=(E, I),
                dtype=F32,
                transform=ParamTransform(
                    fuse_into="e13_s",
                    fuse_order=order,
                    fuse_axis=-1,
                    scale_for=w_n,
                    shard_axis=shard_e13,
                    expert_axis=0,
                ),
            ),
        ]

    decls: list[HFParamDecl] = [
        HFParamDecl(
            name=_n("router"),
            hf_name=f"{p}gate.weight",
            shape=(H, E),
            hf_shape=(E, H),
            dtype=BF16,
            transform=ParamTransform(transpose=(1, 0), shard_axis=None),
            vllm_aliases=(
                VLLMAlias(
                    source=f"{p}gate.weight",
                    targets=(_n("router"),),
                    already_sharded=False,
                    already_fp8=False,
                    flipped=False,
                ),
                VLLMAlias(
                    source=f"{p}_fused_gate_gu_fp8_weight",
                    targets=(_n("router"), _n("sh_gate_proj"), _n("sh_up_proj")),
                    scale_targets=(_n("sh_gate_proj_s"), _n("sh_up_proj_s")),
                    scale_source=f"{p}_fused_gate_gu_fp8_scale",
                    router_scale_source=f"{p}_fused_gate_gu_fp8_scale",
                    shard_axis=-1,
                    already_sharded=False,
                    already_fp8=True,
                    flipped=True,
                ),
            ),
        ),
    ]
    if shared_gate:
        decls.append(
            HFParamDecl(
                name=_n("sh_gate"),
                hf_name=f"{p}shared_expert_gate.weight",
                shape=(H,),
                hf_shape=(1, H),
                dtype=F32,
                transform=ParamTransform(shard_axis=None, replicate_ranks=True),
                vllm_aliases=(
                    VLLMAlias(
                        source=f"{p}shared_expert_gate.weight",
                        targets=(_n("sh_gate"),),
                        already_sharded=False,
                        already_fp8=False,
                        flipped=False,
                    ),
                    VLLMAlias(
                        source=f"{p}_fused_se_gate_weight",
                        targets=(_n("sh_gate"),),
                        already_sharded=False,
                        already_fp8=False,
                        flipped=False,
                    ),
                ),
            )
        )
    sh_gu_alias = (
        VLLMAlias(
            source=f"{p}shared_expert.gate_up_proj",
            targets=(_n("sh_gate_proj"), _n("sh_up_proj")),
            scale_targets=(_n("sh_gate_proj_s"), _n("sh_up_proj_s")),
            scale_source=f"{p}shared_expert.gate_up_proj.weight_scale",
            shard_axis=-1,
            already_sharded=True,
            already_fp8=True,
            flipped=True,
        ),
    )
    e13_alias = (
        VLLMAlias(
            source=f"{p}experts.w13_weight",
            targets=(_n("e_gate_proj"), _n("e_up_proj")),
            scale_targets=(_n("e_gate_proj_s"), _n("e_up_proj_s")),
            scale_source=f"{p}experts.w13_weight_scale",
            shard_axis=shard_e13,
            already_sharded=True,
            already_fp8=True,
            flipped=True,
        ),
    )
    decls.extend(
        [
            *_sh_gu_pair("sh_gate_proj", "gate_proj", 0, sh_gu_alias),
            *_sh_gu_pair("sh_up_proj", "up_proj", 1),
            HFParamDecl(
                name=_n("sh2"),
                hf_name=f"{p}shared_expert.down_proj.weight",
                shape=(IS, H),
                hf_shape=(H, IS),
                dtype=FP8,
                transform=ParamTransform(
                    transpose=(1, 0), quantize="fp8", shard_axis=0
                ),
                vllm_aliases=(
                    VLLMAlias(
                        source=f"{p}shared_expert.down_proj",
                        targets=(_n("sh2"),),
                        scale_targets=(_n("sh2_s"),),
                        scale_source=f"{p}shared_expert.down_proj.weight_scale",
                        shard_axis=0,
                        already_sharded=True,
                        already_fp8=True,
                        flipped=True,
                    ),
                    VLLMAlias(
                        source=f"{p}_fused_se_down_fp8_weight",
                        targets=(_n("sh2"),),
                        scale_targets=(_n("sh2_s"),),
                        scale_source=f"{p}_fused_se_down_fp8_scale",
                        shard_axis=0,
                        already_sharded=False,
                        already_fp8=True,
                        flipped=True,
                    ),
                ),
            ),
            HFParamDecl(
                name=_n("sh2_s"),
                hf_name=f"{p}shared_expert.down_proj.weight_scale_inv",
                shape=(H,),
                hf_shape=(H,),
                dtype=F32,
                transform=ParamTransform(
                    scale_for=_n("sh2"), shard_axis=None, replicate_ranks=True
                ),
            ),
            *_exp_gu_pair("e_gate_proj", "gate_proj", 0, e13_alias),
            *_exp_gu_pair("e_up_proj", "up_proj", 1),
            HFParamDecl(
                name=_n("e2"),
                hf_name=f"{p}experts.down_proj.weight",
                shape=(E, I, H),
                hf_shape=(E, H, I),
                dtype=FP8,
                transform=ParamTransform(
                    transpose=(0, 2, 1),
                    quantize="fp8",
                    shard_axis=shard_e2,
                    expert_axis=0,
                ),
                vllm_aliases=(
                    VLLMAlias(
                        source=f"{p}experts.w2_weight",
                        targets=(_n("e2"),),
                        scale_targets=(_n("e2_s"),),
                        scale_source=f"{p}experts.w2_weight_scale",
                        shard_axis=shard_e2,
                        already_sharded=True,
                        already_fp8=True,
                        flipped=True,
                    ),
                ),
            ),
            HFParamDecl(
                name=_n("e2_s"),
                hf_name=f"{p}experts.down_proj.weight_scale_inv",
                shape=(E, H),
                hf_shape=(E, H),
                dtype=F32,
                transform=ParamTransform(
                    scale_for=_n("e2"),
                    shard_axis=shard_e2_s,
                    replicate_ranks=moe_tp,
                    expert_axis=0,
                ),
            ),
        ]
    )
    return apply_decl_overrides(decls, geom)
