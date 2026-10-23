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
"""`GDNBlock` geometry, parameter, and recurrent-cache declarations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...decls import (
    REQUIRED,
    BlockBinding,
    HFParamDecl,
    ParamTransform,
    RecurrentCacheDecl,
    ResolvedBlockGeometry,
    VLLMAlias,
    apply_decl_overrides,
)
from ...errors import MegakernelConfigError
from ...ops import BF16, F32, FP8
from .params import gdn_lin_in_layout

GDN_PARAM_NAMES: frozenset[str] = frozenset(
    {
        "lin_q",
        "lin_q_s",
        "lin_k",
        "lin_k_s",
        "lin_v",
        "lin_v_s",
        "lin_z",
        "lin_z_s",
        "lin_b",
        "lin_b_s",
        "lin_a",
        "lin_a_s",
        "lin_out",
        "lin_out_s",
        "conv",
        "alog",
        "dtb",
        "gnorm",
    }
)

GDN_GEOMETRY_FIELDS: dict[str, tuple[str, Any]] = {
    "hidden": ("hidden_size", REQUIRED),
    "k_heads": ("linear_num_key_heads", REQUIRED),
    "v_heads": ("linear_num_value_heads", REQUIRED),
    "k_dim": ("linear_key_head_dim", 128),
    "v_dim": ("linear_value_head_dim", 128),
    "conv_kernel": ("linear_conv_kernel_dim", 4),
    "eps": ("rms_norm_eps", 1e-6),
    "gdn_l2_eps": ("gdn_l2_eps", 1e-6),
}

GDN_FIELD_ALIASES: dict[str, str] = {
    "H": "hidden",
    "hidden_size": "hidden",
    "KH": "k_heads",
    "linear_num_key_heads": "k_heads",
    "VH": "v_heads",
    "linear_num_value_heads": "v_heads",
    "KD": "k_dim",
    "linear_key_head_dim": "k_dim",
    "VD": "v_dim",
    "linear_value_head_dim": "v_dim",
    "linear_conv_kernel_dim": "conv_kernel",
    "rms_norm_eps": "eps",
}

GDN_OPTIONS: dict[str, Any] = {
    "decay_type": "head",
    "decay_law": "softplus",
    "gate_type": "silu",
    "decay_bound": 5.0,
}


def validate_gdn_options(options: Mapping[str, Any]) -> None:
    dt = str(options.get("decay_type", "head"))
    if dt not in ("head", "key"):
        raise MegakernelConfigError(
            f"GDNBlock requires decay_type in ('head', 'key'), got {dt!r}"
        )
    dl = str(options.get("decay_law", "softplus"))
    if dl not in ("softplus", "bounded_sigmoid"):
        raise MegakernelConfigError(
            "GDNBlock requires decay_law in ('softplus', 'bounded_sigmoid'), "
            f"got {dl!r}"
        )
    gt = str(options.get("gate_type", "silu"))
    if gt not in ("silu", "sigmoid"):
        raise MegakernelConfigError(
            f"GDNBlock requires gate_type in ('silu', 'sigmoid'), got {gt!r}"
        )


GDN_GEOMETRY_METHODS: dict[str, Any] = {
    "kh_rep": lambda g, R: max(1, R // int(g.KH)) if int(g.KH) > 0 else 1,
    "kh": lambda g, R: gdn_lin_in_layout(g, R).kh,
    "vh": lambda g, R: gdn_lin_in_layout(g, R).vh,
    "QKV": lambda g, R: gdn_lin_in_layout(g, R).QKV,
    "Z": lambda g, R: gdn_lin_in_layout(g, R).Z,
    "BA": lambda g, R: gdn_lin_in_layout(g, R).BA,
    "NLI": lambda g, R: gdn_lin_in_layout(g, R).NLI,
}


def _check_conv_kernel(conv_kernel: int) -> int:
    if conv_kernel != 4:
        raise MegakernelConfigError(
            "GDNBlock requires conv_kernel == 4 (3-tap causal conv window), "
            f"got conv_kernel={conv_kernel}"
        )
    return conv_kernel


def resolve_gdn_binding(
    cls: Any, binding: BlockBinding, resolved: ResolvedBlockGeometry
) -> ResolvedBlockGeometry:
    vals = dict(resolved.values)
    opts = dict(GDN_OPTIONS)
    for k, v in binding.options.items():
        if v is not None:
            opts[k] = v
    validate_gdn_options(opts)
    conv_k = _check_conv_kernel(int(vals.get("conv_kernel", 4)))
    vals.update(
        {
            "H": int(vals["hidden"]),
            "KH": int(vals["k_heads"]),
            "VH": int(vals["v_heads"]),
            "KD": int(vals["k_dim"]),
            "VD": int(vals["v_dim"]),
            "conv_kernel": conv_k,
            "conv_taps": conv_k - 1,
            "eps": float(vals["eps"]),
            "gdn_l2_eps": float(vals["gdn_l2_eps"]),
            "decay_type": str(opts["decay_type"]),
            "decay_law": str(opts["decay_law"]),
            "gate_type": str(opts["gate_type"]),
            "decay_bound": float(opts["decay_bound"]),
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


def gdn_hf_params(prefix: str, geom: Any) -> list[HFParamDecl]:
    """Return canonical (unfused) HF parameter declarations for `GDNBlock`."""
    conv_kernel = _check_conv_kernel(int(getattr(geom, "conv_kernel", 4)))
    p = f"{prefix}." if prefix else ""
    H, KH, VH = int(geom.H), int(geom.KH), int(geom.VH)
    KD, VD = int(geom.KD), int(geom.VD)
    kq_dim, v_dim = KH * KD, VH * VD
    qkv_full = 2 * kq_dim + v_dim
    lay = gdn_lin_in_layout(geom, 1)
    b_slot_width = lay.A_OFF - lay.BA
    a_slot_width = lay.NLI - lay.A_OFF
    rename = geom.rename if isinstance(geom, ResolvedBlockGeometry) else {}

    def _n(canon: str) -> str:
        return rename.get(canon, canon)

    qkv_hf = f"{p}in_proj_qkv.weight"
    qkv_s_hf = f"{p}in_proj_qkv.weight_scale_inv"

    def _qkv_slice(
        canon: str,
        order: int,
        out_dim: int,
        sl: tuple[int, int],
        aliases: tuple[VLLMAlias, ...] = (),
    ) -> list[HFParamDecl]:
        w_n, s_n = _n(canon), _n(f"{canon}_s")
        return [
            HFParamDecl(
                name=w_n,
                hf_name=qkv_hf,
                shape=(H, out_dim),
                hf_shape=(qkv_full, H),
                dtype=FP8,
                transform=ParamTransform(
                    fuse_into="lin_in",
                    fuse_order=order,
                    fuse_axis=-1,
                    split_from=qkv_hf,
                    split_slice=sl,
                    transpose=(1, 0),
                    quantize="fp8",
                    shard_axis=-1,
                ),
                vllm_aliases=aliases,
            ),
            HFParamDecl(
                name=s_n,
                hf_name=qkv_s_hf,
                shape=(out_dim,),
                hf_shape=(qkv_full,),
                dtype=F32,
                transform=ParamTransform(
                    fuse_into="lin_in_s",
                    fuse_order=order,
                    fuse_axis=-1,
                    split_from=qkv_s_hf,
                    split_slice=sl,
                    scale_for=w_n,
                    shard_axis=-1,
                ),
            ),
        ]

    def _ba_pair(
        canon: str,
        hf_stem: str,
        order: int,
        slot_w: int,
        aliases: tuple[VLLMAlias, ...] = (),
        out_dim: int = VH,
    ) -> list[HFParamDecl]:
        w_n, s_n = _n(canon), _n(f"{canon}_s")
        return [
            HFParamDecl(
                name=w_n,
                hf_name=f"{p}{hf_stem}.weight",
                shape=(H, out_dim),
                hf_shape=(out_dim, H),
                dtype=FP8,
                transform=ParamTransform(
                    fuse_into="lin_in",
                    fuse_order=order,
                    fuse_axis=-1,
                    transpose=(1, 0),
                    pad=((0, 0), (0, slot_w)),
                    quantize="per_channel_fp8",
                    shard_axis=-1,
                ),
                vllm_aliases=aliases,
            ),
            HFParamDecl(
                name=s_n,
                hf_name=f"{p}{hf_stem}.weight_scale_inv",
                shape=(out_dim,),
                hf_shape=(out_dim,),
                dtype=F32,
                transform=ParamTransform(
                    fuse_into="lin_in_s",
                    fuse_order=order,
                    fuse_axis=-1,
                    pad=((0, slot_w),),
                    scale_for=w_n,
                    shard_axis=-1,
                ),
            ),
        ]

    qkvz_alias = (
        VLLMAlias(
            source=f"{p}in_proj_qkvz",
            targets=(_n("lin_q"), _n("lin_k"), _n("lin_v"), _n("lin_z")),
            scale_targets=(
                _n("lin_q_s"),
                _n("lin_k_s"),
                _n("lin_v_s"),
                _n("lin_z_s"),
            ),
            scale_source=f"{p}in_proj_qkvz.weight_scale",
            shard_axis=-1,
            already_sharded=True,
            already_fp8=True,
            flipped=True,
        ),
    )
    ba_alias = (
        VLLMAlias(
            source=f"{p}in_proj_ba",
            targets=(_n("lin_b"), _n("lin_a")),
            scale_targets=(_n("lin_b_s"), _n("lin_a_s")),
            shard_axis=-1,
            already_sharded=True,
            already_fp8=False,
            flipped=True,
        ),
    )

    def _vec1d(canon: str, hf_stem: str, dim: int, shard: int | None) -> HFParamDecl:
        n = _n(canon)
        return HFParamDecl(
            name=n,
            hf_name=f"{p}{hf_stem}",
            shape=(dim,),
            hf_shape=(dim,),
            dtype=F32,
            transform=ParamTransform(shard_axis=shard, replicate_ranks=(shard is None)),
            vllm_aliases=(
                VLLMAlias(
                    source=f"{p}{hf_stem}",
                    targets=(n,),
                    shard_axis=shard,
                    already_sharded=(shard is not None),
                    already_fp8=False,
                    flipped=False,
                ),
            ),
        )

    decls = [
        *_qkv_slice("lin_q", 0, kq_dim, (0, kq_dim), qkvz_alias),
        *_qkv_slice("lin_k", 1, kq_dim, (kq_dim, 2 * kq_dim)),
        *_qkv_slice("lin_v", 2, v_dim, (2 * kq_dim, qkv_full)),
        HFParamDecl(
            name=_n("lin_z"),
            hf_name=f"{p}in_proj_z.weight",
            shape=(H, v_dim),
            hf_shape=(v_dim, H),
            dtype=FP8,
            transform=ParamTransform(
                fuse_into="lin_in",
                fuse_order=3,
                fuse_axis=-1,
                transpose=(1, 0),
                quantize="fp8",
                shard_axis=-1,
            ),
        ),
        HFParamDecl(
            name=_n("lin_z_s"),
            hf_name=f"{p}in_proj_z.weight_scale_inv",
            shape=(v_dim,),
            hf_shape=(v_dim,),
            dtype=F32,
            transform=ParamTransform(
                fuse_into="lin_in_s",
                fuse_order=3,
                fuse_axis=-1,
                scale_for=_n("lin_z"),
                shard_axis=-1,
            ),
        ),
        *_ba_pair("lin_b", "in_proj_b", 4, b_slot_width, ba_alias),
        *_ba_pair("lin_a", "in_proj_a", 5, a_slot_width),
        HFParamDecl(
            name=_n("lin_out"),
            hf_name=f"{p}out_proj.weight",
            shape=(v_dim, H),
            hf_shape=(H, v_dim),
            dtype=FP8,
            transform=ParamTransform(transpose=(1, 0), quantize="fp8", shard_axis=0),
            vllm_aliases=(
                VLLMAlias(
                    source=f"{p}out_proj",
                    targets=(_n("lin_out"),),
                    scale_targets=(_n("lin_out_s"),),
                    scale_source=f"{p}out_proj.weight_scale",
                    shard_axis=0,
                    already_sharded=True,
                    already_fp8=True,
                    flipped=True,
                ),
            ),
        ),
        HFParamDecl(
            name=_n("lin_out_s"),
            hf_name=f"{p}out_proj.weight_scale_inv",
            shape=(H,),
            hf_shape=(H,),
            dtype=F32,
            transform=ParamTransform(
                scale_for=_n("lin_out"), shard_axis=None, replicate_ranks=True
            ),
        ),
        HFParamDecl(
            name=_n("conv"),
            hf_name=f"{p}conv1d.weight",
            shape=(conv_kernel, qkv_full),
            hf_shape=(qkv_full, 1, conv_kernel),
            dtype=F32,
            transform=ParamTransform(
                shard_axis=-1,
                channel_splits=(kq_dim, kq_dim, v_dim),
                flatten_tail=True,
            ),
            vllm_aliases=(
                VLLMAlias(
                    source=f"{p}conv1d.weight",
                    targets=(_n("conv"),),
                    shard_axis=-1,
                    already_sharded=True,
                    already_fp8=False,
                    flipped=False,
                ),
            ),
        ),
        _vec1d("alog", "A_log", VH, 0),
        _vec1d("dtb", "dt_bias", VH, 0),
        _vec1d("gnorm", "norm.weight", VD, None),
    ]
    return apply_decl_overrides(decls, geom)


def gdn_cache_decl(geom: Any) -> RecurrentCacheDecl:
    """Return the `RecurrentCacheDecl` describing one GDN layer's slot geometry."""
    conv_kernel = _check_conv_kernel(int(getattr(geom, "conv_kernel", 4)))
    conv_taps = int(getattr(geom, "conv_taps", conv_kernel - 1))
    return RecurrentCacheDecl(
        kh=int(geom.KH),
        vh=int(geom.VH),
        KD=int(geom.KD),
        VD=int(geom.VD),
        rec_dtype=BF16,
        rec_pages_per_slot=None,
        conv_taps=conv_taps,
        conv_dtype=BF16,
        conv_pages_per_slot=None,
    )
