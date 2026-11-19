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
"""`GQABlock` geometry, parameter, and KV-cache declarations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...decls import (
    REQUIRED,
    BlockBinding,
    HFParamDecl,
    KVCacheDecl,
    ParamTransform,
    ResolvedBlockGeometry,
    VLLMAlias,
    apply_decl_overrides,
)
from ...errors import MegakernelConfigError
from ...ops import F32, FP8

GQA_PARAM_NAMES: frozenset[str] = frozenset(
    {
        "q_proj",
        "q_proj_s",
        "k_proj",
        "k_proj_s",
        "v_proj",
        "v_proj_s",
        "att_out",
        "att_out_s",
        "qn",
        "kn",
    }
)

GQA_GEOMETRY_FIELDS: dict[str, tuple[str, Any]] = {
    "hidden": ("hidden_size", REQUIRED),
    "heads": ("num_attention_heads", REQUIRED),
    "kv_heads": ("num_key_value_heads", REQUIRED),
    "head_dim": ("head_dim", None),
    "rot": ("partial_rotary_factor", 1.0),
    "eps": ("rms_norm_eps", 1e-6),
    "theta": ("rope_theta", REQUIRED),
}

GQA_FIELD_ALIASES: dict[str, str] = {
    "H": "hidden",
    "hidden_size": "hidden",
    "NH": "heads",
    "num_attention_heads": "heads",
    "NKV": "kv_heads",
    "num_key_value_heads": "kv_heads",
    "HD": "head_dim",
    "ROT": "rot",
    "rotary_dim": "rot",
    "partial_rotary_factor": "rot",
    "rms_norm_eps": "eps",
    "rope_theta": "theta",
}

GQA_OPTIONS: dict[str, Any] = {"qk_norm": True, "output_gate": True}

GQA_GEOMETRY_METHODS: dict[str, Any] = {
    "nh": lambda g, R: int(g.NH) // R,
    "G": lambda g, R: R // int(g.NKV),
    "NAI": lambda g, R: (2 * (int(g.NH) // R) + 2) * int(g.HD),
}


def validate_gqa_options(options: Mapping[str, Any]) -> None:
    if not bool(options.get("qk_norm", True)):
        raise MegakernelConfigError(
            "GQABlock requires qk_norm=True (the kernel always applies q/k RMSNorm)"
        )
    if not bool(options.get("output_gate", True)):
        raise MegakernelConfigError(
            "GQABlock requires output_gate=True (the kernel always applies the "
            "sigmoid output gate)"
        )


def resolve_gqa_binding(
    cls: Any, binding: BlockBinding, resolved: ResolvedBlockGeometry
) -> ResolvedBlockGeometry:
    vals = dict(resolved.values)
    H, NH = int(vals["hidden"]), int(vals["heads"])
    hd_raw = vals.get("head_dim")
    hd = H // NH if hd_raw is None else int(hd_raw)
    rot_raw = vals.get("rot", 1.0)
    rot_val = (
        int(round(rot_raw * hd))
        if isinstance(rot_raw, float) and 0.0 < rot_raw <= 1.0
        else int(rot_raw)
    )
    vals.update(
        {
            "head_dim": hd,
            "H": H,
            "NH": NH,
            "NKV": int(vals["kv_heads"]),
            "HD": hd,
            "rot": rot_val,
            "ROT": rot_val,
            "eps": float(vals["eps"]),
            "theta": float(vals["theta"]),
        }
    )
    return ResolvedBlockGeometry(
        block_cls=cls,
        prefix=binding.prefix,
        values=vals,
        options=binding.options,
        rename=binding.rename,
        transforms=binding.transforms,
    )


def gqa_hf_params(
    prefix: str,
    geom: Any,
    *,
    qk_norm: bool = True,
    output_gate: bool = True,
) -> list[HFParamDecl]:
    """Return canonical (unfused) HF parameter declarations for `GQABlock`."""
    p = f"{prefix}." if prefix else ""
    H, NH, NKV, HD = int(geom.H), int(geom.NH), int(geom.NKV), int(geom.HD)
    q_dim = NH * (2 if output_gate else 1) * HD
    rename = geom.rename if isinstance(geom, ResolvedBlockGeometry) else {}

    def _n(canon: str) -> str:
        return rename.get(canon, canon)

    def _kv_decls(proj: str, order: int) -> list[HFParamDecl]:
        w_name, s_name = _n(f"{proj}_proj"), _n(f"{proj}_proj_s")
        return [
            HFParamDecl(
                name=w_name,
                hf_name=f"{p}{proj}_proj.weight",
                shape=(NKV, H, HD),
                hf_shape=(NKV * HD, H),
                dtype=FP8,
                transform=ParamTransform(
                    fuse_into="att_in",
                    fuse_order=order,
                    fuse_axis=-1,
                    transpose=(0, 2, 1),
                    quantize="fp8",
                    shard_axis=0,
                    kv_repeat=True,
                ),
                vllm_aliases=(
                    VLLMAlias(
                        source=f"{p}{proj}_proj",
                        targets=(w_name,),
                        scale_targets=(s_name,),
                        scale_source=f"{p}{proj}_proj.weight_scale",
                        shard_axis=0,
                        already_sharded=True,
                        already_fp8=True,
                        flipped=True,
                    ),
                ),
            ),
            HFParamDecl(
                name=s_name,
                hf_name=f"{p}{proj}_proj.weight_scale_inv",
                shape=(NKV, HD),
                hf_shape=(NKV * HD,),
                dtype=F32,
                transform=ParamTransform(
                    fuse_into="att_in_s",
                    fuse_order=order,
                    fuse_axis=-1,
                    scale_for=w_name,
                    shard_axis=0,
                    kv_repeat=True,
                ),
            ),
        ]

    decls: list[HFParamDecl] = [
        HFParamDecl(
            name=_n("q_proj"),
            hf_name=f"{p}q_proj.weight",
            shape=(H, q_dim),
            hf_shape=(q_dim, H),
            dtype=FP8,
            transform=ParamTransform(
                fuse_into="att_in",
                fuse_order=0,
                fuse_axis=-1,
                transpose=(1, 0),
                quantize="fp8",
                shard_axis=-1,
            ),
            vllm_aliases=(
                VLLMAlias(
                    source=f"{p}qkv_proj",
                    targets=(_n("q_proj"), _n("k_proj"), _n("v_proj")),
                    scale_targets=(_n("q_proj_s"), _n("k_proj_s"), _n("v_proj_s")),
                    scale_source=f"{p}qkv_proj.weight_scale",
                    shard_axis=-1,
                    already_sharded=True,
                    already_fp8=True,
                    flipped=True,
                ),
                VLLMAlias(
                    source=f"{p}q_proj",
                    targets=(_n("q_proj"),),
                    scale_targets=(_n("q_proj_s"),),
                    scale_source=f"{p}q_proj.weight_scale",
                    shard_axis=-1,
                    already_sharded=True,
                    already_fp8=True,
                    flipped=True,
                ),
            ),
        ),
        HFParamDecl(
            name=_n("q_proj_s"),
            hf_name=f"{p}q_proj.weight_scale_inv",
            shape=(q_dim,),
            hf_shape=(q_dim,),
            dtype=F32,
            transform=ParamTransform(
                fuse_into="att_in_s",
                fuse_order=0,
                fuse_axis=-1,
                scale_for=_n("q_proj"),
                shard_axis=-1,
            ),
        ),
        *_kv_decls("k", 1),
        *_kv_decls("v", 2),
        HFParamDecl(
            name=_n("att_out"),
            hf_name=f"{p}o_proj.weight",
            shape=(NH * HD, H),
            hf_shape=(H, NH * HD),
            dtype=FP8,
            transform=ParamTransform(transpose=(1, 0), quantize="fp8", shard_axis=0),
            vllm_aliases=(
                VLLMAlias(
                    source=f"{p}o_proj",
                    targets=(_n("att_out"),),
                    scale_targets=(_n("att_out_s"),),
                    scale_source=f"{p}o_proj.weight_scale",
                    shard_axis=0,
                    already_sharded=True,
                    already_fp8=True,
                    flipped=True,
                ),
            ),
        ),
        HFParamDecl(
            name=_n("att_out_s"),
            hf_name=f"{p}o_proj.weight_scale_inv",
            shape=(H,),
            hf_shape=(H,),
            dtype=F32,
            transform=ParamTransform(
                scale_for=_n("att_out"), shard_axis=None, replicate_ranks=True
            ),
        ),
    ]
    if qk_norm:
        decls.extend(
            [
                HFParamDecl(
                    name=_n("qn"),
                    hf_name=f"{p}q_norm.weight",
                    shape=(NH, HD),
                    hf_shape=(HD,),
                    dtype=F32,
                    transform=ParamTransform(shard_axis=0, head_reduce="mean"),
                    vllm_aliases=(
                        VLLMAlias(
                            source=f"{p}q_norm.weight",
                            targets=(_n("qn"),),
                            already_sharded=False,
                            already_fp8=False,
                            flipped=False,
                        ),
                    ),
                ),
                HFParamDecl(
                    name=_n("kn"),
                    hf_name=f"{p}k_norm.weight",
                    shape=(NKV, HD),
                    hf_shape=(HD,),
                    dtype=F32,
                    transform=ParamTransform(shard_axis=0, kv_repeat=True),
                    vllm_aliases=(
                        VLLMAlias(
                            source=f"{p}k_norm.weight",
                            targets=(_n("kn"),),
                            already_sharded=False,
                            already_fp8=False,
                            flipped=False,
                        ),
                    ),
                ),
            ]
        )
    return apply_decl_overrides(decls, geom)


def gqa_cache_decl(geom: Any | None, *, pool_layout: Any = None) -> KVCacheDecl:
    """Return the `KVCacheDecl` describing one attention layer's KV page geometry."""
    if pool_layout is not None:
        return KVCacheDecl(
            hd=int(pool_layout.hd),
            kvh=int(pool_layout.kvh),
            page_size=int(pool_layout.page_size),
            page_dtype=pool_layout.page_dtype,
        )
    assert geom is not None
    return KVCacheDecl(
        hd=int(geom.HD),
        kvh=int(getattr(geom, "kvh", 1)),
        page_size=int(getattr(geom, "page_size", 128)),
        page_dtype=FP8,
    )
