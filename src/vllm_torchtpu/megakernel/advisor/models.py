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
"""Model profile registry, HuggingFace config.json validation, and preset resolution."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import gcd
from pathlib import Path

from ..core import errors
from .constants import (
    BITS_FP4,
    BITS_FP8,
    KDA_MODEL_TYPES,
    MIN_HEAD_DIM,
    MIN_INTERVAL,
    MULTIMODAL_KEYS,
    PRESETS_DIR,
    VALID_LAYER_TOKENS,
)


class UnsupportedArchitectureError(errors.MegakernelConfigError, ValueError):
    """Raised when a model preset or HuggingFace config is unsupported or malformed."""


RECOGNIZED_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "deepseek_v2",
        "deepseek_v3",
        "deepseek_v4",
        "glm4_moe",
        "glm5",
        "glm_moe",
        "gpt_oss",
        "kimi_k2",
        "kimi_k3",
        "kimi_linear",
        "llama",
        "llama4",
        "llama4_text",
        "mistral",
        "mixtral",
        "qwen2",
        "qwen2_moe",
        "qwen3",
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
        "qwen3_5_text",
        "qwen3_8",
        "qwen3_8_moe",
        "qwen3_moe",
        "qwen3_next",
        "qwen4_exp",
    }
)

SUPPORTED_MODEL_TYPES: frozenset[str] = RECOGNIZED_MODEL_TYPES


@dataclass(frozen=True)
class ModelProfile:
    """Architecture geometry and dtype profile for megakernel scope evaluation."""

    name: str
    num_layers: int
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    num_linear_layers: int = 0
    linear_key_heads: int = 0
    linear_value_heads: int = 0
    num_experts: int = 0
    top_k: int = 0
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    num_moe_layers: int = -1
    num_attention_layers: int = 0
    kv_lora_rank: int = 0
    q_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    linear_key_dim: int = 128
    linear_value_dim: int = 128
    linear_conv_kernel: int = 4
    routed_expert_hidden_size: int = 0
    is_periodic: bool = True
    ar_slots_per_layer: int = 2
    weight_dtype: str = "fp8"
    expert_dtype: str = ""
    act_dtype: str = ""
    weight_block_k: int = 0
    kv_dtype: str = "fp8"
    state_dtype: str = "bf16"
    model_type: str = ""
    architecture_unverified: bool = False
    layer_families: tuple[str, ...] = ()
    has_attn_output_gate: bool = False
    non_homogeneous_reasons: tuple[str, ...] = ()
    kv_compress_factor: float = 1.0
    residual_streams: int = 1
    dtype_source: str = ""


def _non_homogeneous_reasons(num_layers: int, moe_layers: int) -> tuple[str, ...]:
    """Return non-homogeneity tags for mixed dense-prefix and MoE stacks."""
    if 0 < moe_layers < num_layers:
        return ("dense_prefix_moe",)
    return ()


def _derive_layer_families(
    profile: ModelProfile, *, attn_layers: int, moe_layers: int
) -> tuple[str, ...]:
    """Derive canonical layer families from resolved model layer counts."""
    families: list[str] = []
    if attn_layers > 0:
        families.append("mla" if profile.kv_lora_rank > 0 else "gqa")
    if profile.num_linear_layers > 0:
        mtype = profile.model_type.strip().lower()
        families.append("kda" if mtype in KDA_MODEL_TYPES else "gdn")
    if moe_layers > 0:
        families.append("moe")
    if moe_layers < profile.num_layers and profile.intermediate_size > 0:
        families.append("dense_mlp")
    if 0 < moe_layers < profile.num_layers:
        families.append("dense_prefix_moe")
    if profile.kv_compress_factor < 1.0 and "sparse_csa" not in families:
        families.append("sparse_csa")
    if profile.residual_streams > 1 and "hyper_connection" not in families:
        families.append("hyper_connection")
    return tuple(families)


def finalize_model_profile(profile: ModelProfile) -> ModelProfile:
    """Return an immutable ModelProfile with derived layer metadata populated."""
    num_layers = profile.num_layers
    has_experts = profile.num_experts > 0
    attn_layers = profile.num_attention_layers or (
        num_layers - profile.num_linear_layers
    )
    resolved_moe_layers = (
        num_layers * has_experts
        if profile.num_moe_layers < 0
        else profile.num_moe_layers
    )
    periodic = profile.is_periodic and not (0 < resolved_moe_layers < num_layers)
    moe_inter = profile.moe_intermediate_size or (
        profile.intermediate_size * has_experts
    )
    exp_dt = profile.expert_dtype or profile.weight_dtype
    w_dt_low = profile.weight_dtype.lower()
    default_act_dt = (
        "fp8" if w_dt_low in ("fp8", "nvfp4", "mxfp4", "fp4", "int8") else w_dt_low
    )
    act_dt = (profile.act_dtype or default_act_dt).lower()
    mtype = profile.model_type.strip().lower()
    unverified = profile.architecture_unverified or bool(
        mtype and mtype not in RECOGNIZED_MODEL_TYPES
    )
    families = profile.layer_families or _derive_layer_families(
        profile, attn_layers=attn_layers, moe_layers=resolved_moe_layers
    )
    reasons = profile.non_homogeneous_reasons or _non_homogeneous_reasons(
        num_layers, resolved_moe_layers
    )
    return replace(
        profile,
        num_attention_layers=attn_layers,
        num_moe_layers=resolved_moe_layers,
        is_periodic=periodic,
        moe_intermediate_size=moe_inter,
        expert_dtype=exp_dt,
        act_dtype=act_dt,
        architecture_unverified=unverified,
        layer_families=families,
        non_homogeneous_reasons=reasons,
    )


def _as_mapping(val: object) -> Mapping[str, object] | None:
    """Return val when it is a Mapping, else None."""
    match val:
        case Mapping() as mapping:
            return mapping
        case _:
            return None


def _as_sequence(val: object) -> tuple[object, ...] | None:
    """Return val as a tuple when it is a list or tuple, else None."""
    match val:
        case list() | tuple():
            return tuple(val)
        case _:
            return None


def _as_scalar_number(val: object) -> float | None:
    """Return finite float for numeric scalars while rejecting bools and strings."""
    match val:
        case bool():
            return None
        case int() | float():
            num = float(val)
            return num if math.isfinite(num) else None
        case _:
            return None


def _int_field(
    mapping: Mapping[str, object], keys: tuple[str, ...], *, default: int = 0
) -> int:
    """Return the first non-zero integer value matching keys in mapping."""
    for key in keys:
        val = mapping.get(key)
        match val:
            case bool():
                continue
            case int() | float():
                if int(val) != 0:
                    return int(val)
            case str():
                stripped = val.strip()
                if stripped:
                    parsed = int(stripped)
                    if parsed != 0:
                        return parsed
            case _:
                continue
    return default


@dataclass(frozen=True, slots=True)
class _NormalizedHfConfig:
    """Normalized HuggingFace configuration parsed at the JSON entry boundary."""

    model_type: str
    is_multimodal: bool
    has_gdn_kda_heads: bool
    has_ssm: bool
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    num_linear_layers: int
    is_periodic: bool
    lin_k_heads: int
    lin_v_heads: int
    num_experts: int
    dense_prefix: int
    ffn_inter: int
    moe_inter: int
    num_shared: int
    shared_inter: int
    top_k: int
    kv_lora_rank: int
    q_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    linear_key_dim: int
    linear_value_dim: int
    linear_conv_kernel: int
    routed_expert_hidden_size: int
    has_attn_output_gate: bool
    torch_dtype: str
    quant_tag: str
    quant_bits: int
    quant_w_type: str
    raw_weight_dtype: str
    raw_expert_dtype: str
    raw_layer_types: tuple[object, ...] | None
    raw_act_dtype: str = ""
    weight_block_k: int = 0
    kv_compress_factor: float = 1.0
    residual_streams: int = 1
    has_quant_config: bool = False
    has_explicit_torch_dtype: bool = False

    def validate_supported(self, *, require_model_type: bool = False) -> None:
        """Validate that the configuration describes a supported causal LM."""
        _ = require_model_type
        if not self.model_type or self.model_type not in SUPPORTED_MODEL_TYPES:
            msg = f"Unsupported architecture model_type {self.model_type!r}"
            raise UnsupportedArchitectureError(msg)
        if self.has_ssm and not (self.num_heads or self.has_gdn_kda_heads):
            msg = "Unmodeled state-space config missing attention and GDN/KDA heads"
            raise UnsupportedArchitectureError(msg)

        for key_name, val in (
            ("num_hidden_layers", self.num_layers),
            ("hidden_size", self.hidden_size),
            ("num_attention_heads", self.num_heads),
        ):
            if val < 1:
                msg = (
                    "HuggingFace config missing required causal-LM geometry key "
                    f"{key_name!r} >= 1"
                )
                raise UnsupportedArchitectureError(msg)
        if self.num_kv_heads < 0 or self.num_kv_heads > self.num_heads:
            msg = (
                f"Invalid num_kv_heads ({self.num_kv_heads}) for "
                f"num_attention_heads ({self.num_heads})"
            )
            raise UnsupportedArchitectureError(msg)
        if self.head_dim < MIN_HEAD_DIM or self.head_dim % MIN_HEAD_DIM != 0:
            msg = (
                f"Invalid head_dim ({self.head_dim}): must be >= {MIN_HEAD_DIM} "
                f"and divisible by {MIN_HEAD_DIM}"
            )
            raise UnsupportedArchitectureError(msg)
        self._validate_layer_tokens()

    def _validate_layer_tokens(self) -> None:
        """Validate explicit layer_types tokens when present."""
        if self.raw_layer_types is None:
            return
        for token in self.raw_layer_types:
            tok_s = str(token)
            if tok_s not in VALID_LAYER_TOKENS:
                msg = (
                    f"Unknown layer_type {tok_s!r}; supported tokens: "
                    f"{sorted(VALID_LAYER_TOKENS)}"
                )
                raise UnsupportedArchitectureError(msg)
            if tok_s == "mamba" and not self.has_gdn_kda_heads:
                msg = "Layer token 'mamba' requires GDN/KDA head dimensions"
                raise UnsupportedArchitectureError(msg)

    def infer_quant_dtypes(
        self,
        weight_dtype: str | None,
        expert_dtype: str | None,
    ) -> tuple[str, str]:
        """Infer weight and expert dtypes from normalized quant fields."""
        has_fp8_tag = any(k in self.quant_tag for k in ("fp8", "float8", "e4m3"))
        is_int_quant = self.quant_w_type in ("int", "int8") or any(
            k in self.quant_tag
            for k in ("int8", "w8a8", "smoothquant", "gptq", "awq", "int-quantized")
        )
        if has_fp8_tag:
            fallback_dt = "fp8"
        elif self.quant_bits == BITS_FP8:
            fallback_dt = "int8" if is_int_quant else "fp8"
        elif is_int_quant and "int8" in self.quant_tag:
            fallback_dt = "int8"
        else:
            fallback_dt = "fp32" if "32" in self.quant_tag else "bf16"
        inferred_dt = (
            "nvfp4"
            if "nvfp4" in self.quant_tag
            else (
                "mxfp4"
                if "mxfp4" in self.quant_tag or self.quant_bits == BITS_FP4
                else fallback_dt
            )
        )
        resolved_wd = str(weight_dtype or self.raw_weight_dtype or inferred_dt).lower()
        resolved_ed = str(expert_dtype or self.raw_expert_dtype or resolved_wd).lower()
        return resolved_wd, resolved_ed

    def infer_quant_dtypes_with_source(
        self,
        weight_dtype: str | None,
        expert_dtype: str | None,
    ) -> tuple[str, str, str]:
        """Infer weight and expert dtypes and resolve provenance source."""
        resolved_wd, resolved_ed = self.infer_quant_dtypes(weight_dtype, expert_dtype)
        if weight_dtype is not None or expert_dtype is not None:
            dtype_source = "flag"
        elif self.raw_weight_dtype or self.raw_expert_dtype or self.has_quant_config:
            dtype_source = "quantization_config"
        elif self.has_explicit_torch_dtype:
            dtype_source = "default (torch_dtype; no quantization_config)"
        else:
            dtype_source = "default (bf16 fallback; no quantization_config)"
        return resolved_wd, resolved_ed, dtype_source

    def to_model_profile(  # noqa: PLR0913
        self,
        name: str = "model",
        weight_dtype: str | None = None,
        expert_dtype: str | None = None,
        kv_dtype: str = "fp8",
        state_dtype: str = "bf16",
        *,
        act_dtype: str | None = None,
    ) -> ModelProfile:
        """Construct authoritative ModelProfile from normalized configuration."""
        if self.num_linear_layers > 0 and (
            self.lin_k_heads < 1 or self.lin_v_heads < 1
        ):
            msg = (
                "Model with linear-attention layers missing GDN/KDA key/value "
                "head counts"
            )
            raise UnsupportedArchitectureError(msg)

        resolved_wd, resolved_ed, dtype_source = self.infer_quant_dtypes_with_source(
            weight_dtype, expert_dtype
        )
        resolved_act_dt = str(act_dtype or self.raw_act_dtype or "").lower()

        return finalize_model_profile(
            ModelProfile(
                name=name,
                num_layers=self.num_layers,
                hidden_size=self.hidden_size,
                intermediate_size=self.ffn_inter,
                num_attention_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                num_linear_layers=self.num_linear_layers,
                linear_key_heads=self.lin_k_heads,
                linear_value_heads=self.lin_v_heads,
                num_experts=self.num_experts,
                top_k=self.top_k,
                moe_intermediate_size=self.moe_inter,
                shared_expert_intermediate_size=self.shared_inter,
                num_moe_layers=max(0, self.num_layers - self.dense_prefix)
                * (self.num_experts > 0),
                num_attention_layers=0,
                kv_lora_rank=self.kv_lora_rank,
                q_lora_rank=self.q_lora_rank,
                qk_nope_head_dim=self.qk_nope_head_dim,
                qk_rope_head_dim=self.qk_rope_head_dim,
                linear_key_dim=self.linear_key_dim,
                linear_value_dim=self.linear_value_dim,
                linear_conv_kernel=self.linear_conv_kernel,
                routed_expert_hidden_size=self.routed_expert_hidden_size,
                is_periodic=self.is_periodic,
                ar_slots_per_layer=2,
                weight_dtype=resolved_wd,
                expert_dtype=resolved_ed,
                act_dtype=resolved_act_dt,
                weight_block_k=self.weight_block_k,
                kv_dtype=kv_dtype.lower(),
                state_dtype=state_dtype.lower(),
                model_type=self.model_type,
                has_attn_output_gate=self.has_attn_output_gate,
                kv_compress_factor=self.kv_compress_factor,
                residual_streams=self.residual_streams,
                dtype_source=dtype_source,
            )
        )


def _extract_text_subconfig(
    raw_cfg: Mapping[str, object],
) -> tuple[Mapping[str, object], bool]:
    """Extract text sub-configuration and multimodal flag from raw HF config."""
    raw_map = _as_mapping(raw_cfg)
    if raw_map is None or len(raw_map) == 0:
        msg = "Empty HuggingFace config mapping"
        raise UnsupportedArchitectureError(msg)
    for sub_key in ("text_config", "language_config"):
        if sub_key in raw_map and _as_mapping(raw_map.get(sub_key)) is None:
            msg = f"Multimodal config has non-mapping {sub_key!r}"
            raise UnsupportedArchitectureError(msg)
    sub_cfg = _as_mapping(raw_map.get("text_config")) or _as_mapping(
        raw_map.get("language_config")
    )
    tc: Mapping[str, object] = sub_cfg if sub_cfg is not None else raw_map
    is_multimodal = (
        "text_config" in raw_map
        or "language_config" in raw_map
        or any(k in raw_map for k in MULTIMODAL_KEYS)
    )
    if is_multimodal and not tc.get("hidden_size"):
        msg = "Multimodal config missing extractable text_config.hidden_size"
        raise UnsupportedArchitectureError(msg)
    return tc, is_multimodal


def _resolve_linear_schedule(
    tc: Mapping[str, object],
    lin_map: Mapping[str, object],
    num_layers: int,
    raw_layer_types: tuple[object, ...] | None,
) -> tuple[int, bool]:
    """Resolve (num_linear_layers, is_periodic) from HF layer schedule fields."""
    interval = _int_field(tc, ("full_attention_interval",))
    if raw_layer_types:
        if len(raw_layer_types) != num_layers:
            msg = (
                f"layer_types len {len(raw_layer_types)} != num_hidden_layers "
                f"{num_layers}"
            )
            raise UnsupportedArchitectureError(msg)
        num_lin = sum(
            any(k in str(t).lower() for k in ("lin", "gdn", "mamba", "kda"))
            for t in raw_layer_types
        )
        if 0 < num_lin < num_layers:
            period_gcd = gcd(num_layers - num_lin, num_layers)
            stride = num_layers // period_gcd
            is_per = list(raw_layer_types[:stride]) * period_gcd == list(
                raw_layer_types
            )
        else:
            is_per = True
        return num_lin, is_per
    kda_seq = _as_sequence(lin_map.get("kda_layers"))
    kda_layers = (
        tuple(v for v in kda_seq if type(v) is int) if kda_seq is not None else ()
    )
    if kda_layers:
        return len(kda_layers), True
    fa_seq = _as_sequence(lin_map.get("full_attn_layers"))
    full_attn_layers = (
        tuple(v for v in fa_seq if type(v) is int) if fa_seq is not None else ()
    )
    if full_attn_layers:
        return max(0, num_layers - len(full_attn_layers)), True
    num_lin = num_layers - num_layers // interval if interval > 1 else 0
    is_per = interval < MIN_INTERVAL or num_layers % interval == 0
    return num_lin, is_per


def _extract_quant_fields(
    tc: Mapping[str, object],
    raw_cfg: Mapping[str, object],
) -> tuple[str, str, int, str, str, str, str, int, bool, bool]:
    """Extract normalized quantization fields from config mappings."""
    quant_raw = tc.get("quantization_config") or raw_cfg.get("quantization_config")
    quant_map = _as_mapping(quant_raw) or {}
    has_quant_config = bool(quant_map)
    has_explicit_torch_dtype = any(
        tc.get(k) or raw_cfg.get(k) for k in ("dtype", "torch_dtype")
    )
    torch_dt = str(
        tc.get("dtype")
        or raw_cfg.get("dtype")
        or tc.get("torch_dtype")
        or raw_cfg.get("torch_dtype")
        or "bf16"
    )
    bits = _int_field(quant_map, ("bits", "w_bit"))
    quant_w_type = str(quant_map.get("type") or "").strip().lower()
    cfg_groups = _as_mapping(quant_map.get("config_groups"))
    if cfg_groups is not None:
        for grp in cfg_groups.values():
            grp_map = _as_mapping(grp)
            w_grp = _as_mapping(grp_map.get("weights")) if grp_map is not None else None
            if w_grp is not None:
                bits = bits or _int_field(w_grp, ("num_bits",))
                quant_w_type = (
                    quant_w_type or str(w_grp.get("type") or "").strip().lower()
                )
                if bits > 0 and quant_w_type:
                    break
    quant_tag = (
        f"{quant_map.get('quant_method', '')} "
        f"{quant_map.get('format', '')} "
        f"{quant_map.get('quant_algo', '')} {torch_dt}"
    ).lower()
    raw_wd = str(tc.get("weight_dtype") or raw_cfg.get("weight_dtype") or "").lower()
    raw_ed = str(
        quant_map.get("expert_dtype")
        or tc.get("expert_dtype")
        or raw_cfg.get("expert_dtype")
        or ""
    ).lower()
    raw_act_dt = str(
        quant_map.get("act_dtype")
        or tc.get("act_dtype")
        or raw_cfg.get("act_dtype")
        or ""
    ).lower()
    if not raw_act_dt:
        act_scheme = str(quant_map.get("activation_scheme") or "").strip().lower()
        if act_scheme == "none" or any(
            k in quant_tag for k in ("w8a16", "w4a16", "weight_only", "weight-only")
        ):
            raw_act_dt = "bf16"
    block_seq = _as_sequence(
        quant_map.get("weight_block_size") or quant_map.get("block_size")
    )
    weight_block_k = _int_field(quant_map, ("weight_block_k", "block_k")) or _int_field(
        tc, ("weight_block_k",)
    )
    if not weight_block_k and block_seq:
        last_dim = _as_scalar_number(block_seq[-1])
        if last_dim is not None and int(last_dim) > 0:
            weight_block_k = int(last_dim)
    return (
        torch_dt,
        quant_tag,
        bits,
        quant_w_type,
        raw_wd,
        raw_ed,
        raw_act_dt,
        weight_block_k,
        has_quant_config,
        has_explicit_torch_dtype,
    )


def _normalize_hf_config(raw_cfg: Mapping[str, object]) -> _NormalizedHfConfig:
    """Normalize a raw HuggingFace configuration mapping into a typed configuration."""
    tc, is_multimodal = _extract_text_subconfig(raw_cfg)
    mtype = str(tc.get("model_type") or raw_cfg.get("model_type") or "").strip().lower()

    lin_map = _as_mapping(tc.get("linear_attn_config")) or {}
    has_gdn_kda_heads = bool(
        tc.get("linear_num_key_heads")
        or tc.get("linear_num_value_heads")
        or lin_map.get("num_key_heads")
        or lin_map.get("num_value_heads")
        or lin_map.get("num_heads")
    )
    has_ssm = "ssm_cfg" in tc or "state_size" in tc

    raw_layer_types = _as_sequence(tc.get("layer_types"))

    num_layers = _int_field(tc, ("num_hidden_layers",))
    hidden_size = _int_field(tc, ("hidden_size",))
    num_heads = _int_field(tc, ("num_attention_heads",))
    num_linear_layers, is_periodic = _resolve_linear_schedule(
        tc, lin_map, num_layers, raw_layer_types
    )

    merged: dict[str, object] = {**lin_map, **tc}
    lin_k_heads = _int_field(
        merged,
        ("linear_num_key_heads", "num_key_heads", "num_k_heads", "num_heads"),
    )
    lin_v_heads = _int_field(
        merged,
        ("linear_num_value_heads", "num_value_heads", "num_v_heads", "num_heads"),
    )

    num_experts = _int_field(
        tc, ("num_experts", "n_routed_experts", "num_local_experts")
    )
    mlp_only = _as_sequence(tc.get("mlp_only_layers"))
    dense_prefix = _int_field(tc, ("first_k_dense_replace",)) or (
        len(mlp_only) if mlp_only is not None else 0
    )

    ffn_inter = _int_field(
        tc, ("intermediate_size", "ffn_dim"), default=4 * hidden_size
    )
    moe_inter = _int_field(
        tc,
        ("moe_intermediate_size", "latent_moe_dim"),
        default=ffn_inter * (num_experts > 0),
    )
    num_shared = _int_field(tc, ("n_shared_experts", "num_shared_experts"))
    shared_inter = _int_field(
        tc, ("shared_expert_intermediate_size",), default=num_shared * moe_inter
    )
    top_k = _int_field(
        tc, ("num_experts_per_tok", "num_experts_per_token", "num_selected_experts")
    ) * (num_experts > 0)
    (
        torch_dt,
        quant_tag,
        bits,
        quant_w_type,
        raw_wd,
        raw_ed,
        raw_act_dt,
        weight_block_k,
        has_quant_config,
        has_explicit_torch_dtype,
    ) = _extract_quant_fields(tc, raw_cfg)
    kv_compress = _extract_kv_compress_factor(tc)

    return _NormalizedHfConfig(
        model_type=mtype,
        is_multimodal=is_multimodal,
        has_gdn_kda_heads=has_gdn_kda_heads,
        has_ssm=has_ssm,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=_int_field(tc, ("num_key_value_heads",), default=num_heads),
        head_dim=_int_field(
            tc,
            ("v_head_dim", "head_dim"),
            default=hidden_size // num_heads if num_heads else 0,
        ),
        num_linear_layers=num_linear_layers,
        is_periodic=is_periodic,
        lin_k_heads=lin_k_heads,
        lin_v_heads=lin_v_heads,
        num_experts=num_experts,
        dense_prefix=dense_prefix,
        ffn_inter=ffn_inter,
        moe_inter=moe_inter,
        num_shared=num_shared,
        shared_inter=shared_inter,
        top_k=top_k,
        kv_lora_rank=_int_field(tc, ("kv_lora_rank",)),
        q_lora_rank=_int_field(tc, ("q_lora_rank",)),
        qk_nope_head_dim=_int_field(tc, ("qk_nope_head_dim",)),
        qk_rope_head_dim=_int_field(tc, ("qk_rope_head_dim",)),
        linear_key_dim=_int_field(
            merged, ("linear_key_head_dim", "key_head_dim", "head_k_dim"), default=128
        ),
        linear_value_dim=_int_field(
            merged,
            ("linear_value_head_dim", "value_head_dim", "head_v_dim"),
            default=128,
        ),
        linear_conv_kernel=_int_field(
            merged, ("linear_conv_kernel_dim", "short_conv_kernel_size"), default=4
        ),
        routed_expert_hidden_size=_int_field(tc, ("routed_expert_hidden_size",)),
        has_attn_output_gate=bool(tc.get("attn_output_gate", False)),
        torch_dtype=torch_dt,
        quant_tag=quant_tag,
        quant_bits=bits,
        quant_w_type=quant_w_type,
        raw_weight_dtype=raw_wd,
        raw_expert_dtype=raw_ed,
        raw_layer_types=raw_layer_types,
        raw_act_dtype=raw_act_dt,
        weight_block_k=weight_block_k,
        kv_compress_factor=kv_compress,
        residual_streams=max(
            1,
            _int_field(
                tc,
                ("hc_mult", "hc_count", "mhc_num_streams", "residual_streams"),
                default=1,
            ),
        ),
        has_quant_config=has_quant_config,
        has_explicit_torch_dtype=has_explicit_torch_dtype,
    )


def _extract_kv_compress_factor(tc: Mapping[str, object]) -> float:
    """Extract average KV cache compression ratio in (0, 1] from HF config."""
    if "compress_ratios" in tc and tc["compress_ratios"] is not None:
        raw_ratios = _as_sequence(tc["compress_ratios"])
        if raw_ratios is None or not raw_ratios:
            msg = "compress_ratios must be a non-empty sequence of positive numbers"
            raise UnsupportedArchitectureError(msg)
        vals: list[float] = []
        for idx, entry in enumerate(raw_ratios):
            num = _as_scalar_number(entry)
            if num is None or num < 0.0:
                msg = f"Invalid compress_ratios[{idx}]={entry!r}: expected >= 0"
                raise UnsupportedArchitectureError(msg)
            vals.append(1.0 if num == 0.0 else (1.0 / num if num > 1.0 else num))
        return min(1.0, sum(vals) / len(vals))
    if "kv_compress_factor" in tc and tc["kv_compress_factor"] is not None:
        raw_kcf = tc["kv_compress_factor"]
        num = _as_scalar_number(raw_kcf)
        if num is None or num <= 0.0 or num > 1.0:
            msg = f"Invalid kv_compress_factor={raw_kcf!r}: expected value in (0, 1]"
            raise UnsupportedArchitectureError(msg)
        return num
    return 1.0


def supports_hf_config(
    raw_cfg: Mapping[str, object], *, require_model_type: bool = False
) -> None:
    """Validate that a HuggingFace config mapping describes a supported causal LM."""
    norm = _normalize_hf_config(raw_cfg)
    norm.validate_supported(require_model_type=require_model_type)


def _load_model_presets() -> tuple[
    Mapping[str, ModelProfile], Mapping[str, str], Mapping[str, str]
]:
    """Load builtin model presets, aliases, and source provenance from JSON."""
    payload = json.loads((PRESETS_DIR / "models.json").read_text("utf-8"))
    raw_presets = payload["presets"]
    presets: dict[str, ModelProfile] = {}
    sources: dict[str, str] = {}
    for slug, entry in raw_presets.items():
        sources[slug] = str(entry.get("source", f"models.json:{slug}"))
        presets[slug] = finalize_model_profile(
            ModelProfile(
                name=slug,
                num_layers=int(entry["num_layers"]),
                hidden_size=int(entry.get("hidden_size", 4096)),
                intermediate_size=int(entry.get("intermediate_size", 11008)),
                num_attention_heads=int(entry.get("num_attention_heads", 32)),
                num_kv_heads=int(entry.get("num_kv_heads", 8)),
                head_dim=int(entry.get("head_dim", 128)),
                num_linear_layers=int(entry.get("num_linear_layers", 0)),
                linear_key_heads=int(entry.get("linear_key_heads", 0)),
                linear_value_heads=int(entry.get("linear_value_heads", 0)),
                num_experts=int(entry.get("num_experts", 0)),
                top_k=int(entry.get("top_k", 0)),
                moe_intermediate_size=int(entry.get("moe_intermediate_size", 0)),
                shared_expert_intermediate_size=int(
                    entry.get("shared_expert_intermediate_size", 0)
                ),
                num_moe_layers=int(entry.get("num_moe_layers", -1)),
                kv_lora_rank=int(entry.get("kv_lora_rank", 0)),
                q_lora_rank=int(entry.get("q_lora_rank", 0)),
                qk_nope_head_dim=int(entry.get("qk_nope_head_dim", 0)),
                qk_rope_head_dim=int(entry.get("qk_rope_head_dim", 0)),
                routed_expert_hidden_size=int(
                    entry.get("routed_expert_hidden_size", 0)
                ),
                weight_dtype=str(entry.get("weight_dtype", "fp8")),
                expert_dtype=str(entry.get("expert_dtype", "")),
                act_dtype=str(entry.get("act_dtype", "")),
                weight_block_k=int(entry.get("weight_block_k", 0)),
                model_type=str(entry.get("model_type", "")),
                has_attn_output_gate=bool(entry.get("has_attn_output_gate", False)),
                kv_compress_factor=float(entry.get("kv_compress_factor", 1.0)),
                residual_streams=int(entry.get("residual_streams", 1)),
                layer_families=tuple(str(x) for x in entry.get("layer_families", ())),
                dtype_source=f"preset ({slug})",
            )
        )
    aliases: dict[str, str] = {str(k): str(v) for k, v in payload["aliases"].items()}
    return presets, aliases, sources


MODEL_PRESETS, REGISTERED_MODEL_ALIASES, MODEL_PRESET_SOURCES = _load_model_presets()


@dataclass(frozen=True)
class ParallelLayout:
    """One serving role's mesh: parallel degrees, sequence sharding, layer groups."""

    tp: int
    ep: int
    dp: int = 1
    pp: int = 1
    dcp: int = 1
    pcp: int = 1
    spec_tokens: int = 0
    layer_groups: tuple[int, ...] = ()
    megakernel: Mapping[str, object] | None = None

    def label(self) -> str:
        """Return the layout as `TP8 x EP8 x DCP4`, omitting degree-1 axes."""
        axes = (
            ("TP", self.tp),
            ("DP", self.dp),
            ("EP", self.ep),
            ("PP", self.pp),
            ("PCP", self.pcp),
            ("DCP", self.dcp),
        )
        return " x ".join(f"{name}{deg}" for name, deg in axes if deg > 1) or "TP1"


@dataclass(frozen=True)
class DeploymentPreset:
    """Shipped prefill and decode layouts for one model, used before auto-pick."""

    decode: ParallelLayout
    prefill: ParallelLayout
    saturation: ParallelLayout | None = None
    prefill_windows: tuple[int, ...] = ()


def _int_tuple(raw: object, field_name: str) -> tuple[int, ...]:
    """Parse an integer tuple from JSON sequence, rejecting invalid scalars."""
    if raw is None:
        return ()
    seq = _as_sequence(raw)
    if seq is None:
        msg = f"{field_name} must be a sequence of integers"
        raise errors.MegakernelConfigError(msg)
    out: list[int] = []
    for item in seq:
        match item:
            case bool():
                msg = f"{field_name} entries must be integers, got bool"
                raise errors.MegakernelConfigError(msg)
            case int():
                out.append(item)
            case _:
                msg = (
                    f"{field_name} entries must be integers, got {type(item).__name__}"
                )
                raise errors.MegakernelConfigError(msg)
    return tuple(out)


def _preset_int(raw: Mapping[str, object], key: str, default: int) -> int:
    """Parse an integer preset field while rejecting bools and floats."""
    val = raw.get(key, default)
    match val:
        case bool():
            msg = f"Deployment preset {key!r} must be an integer, got bool"
            raise errors.MegakernelConfigError(msg)
        case int():
            return val
        case _:
            msg = (
                f"Deployment preset {key!r} must be an integer, "
                f"got {type(val).__name__}"
            )
            raise errors.MegakernelConfigError(msg)


def _parse_parallel_layout(raw: Mapping[str, object]) -> ParallelLayout:
    """Parse a ParallelLayout mapping from deployments.json."""
    groups = _int_tuple(raw.get("layer_groups"), "layer_groups")
    mk_map = _as_mapping(raw.get("megakernel"))
    mk_decl: Mapping[str, object] | None = None
    if mk_map is not None:
        if not str(mk_map.get("source", "")).strip():
            msg = (
                "megakernel declaration in deployments.json requires non-empty 'source'"
            )
            raise errors.MegakernelConfigError(msg)
        mk_decl = dict(mk_map)
    return ParallelLayout(
        tp=_preset_int(raw, "tp", 1),
        ep=_preset_int(raw, "ep", 1),
        dp=_preset_int(raw, "dp", 1),
        pp=_preset_int(raw, "pp", 1),
        dcp=_preset_int(raw, "dcp", 1),
        pcp=_preset_int(raw, "pcp", 1),
        spec_tokens=_preset_int(raw, "spec_tokens", 0),
        layer_groups=groups,
        megakernel=mk_decl,
    )


def _load_deployment_presets() -> tuple[
    Mapping[str, DeploymentPreset], Mapping[str, str]
]:
    """Load deployment layout presets and source strings from JSON."""
    payload = json.loads((PRESETS_DIR / "deployments.json").read_text("utf-8"))
    raw_deps = payload["deployments"]
    deps: dict[str, DeploymentPreset] = {}
    sources: dict[str, str] = {}
    for slug, entry in raw_deps.items():
        sources[slug] = str(entry.get("source", f"deployments.json:{slug}"))
        sat_map = _as_mapping(entry.get("saturation"))
        sat = _parse_parallel_layout(sat_map) if sat_map is not None else None
        windows = _int_tuple(entry.get("prefill_windows"), "prefill_windows")
        deps[slug] = DeploymentPreset(
            decode=_parse_parallel_layout(entry["decode"]),
            prefill=_parse_parallel_layout(entry["prefill"]),
            saturation=sat,
            prefill_windows=windows,
        )
    return deps, sources


DEPLOYMENT_PRESETS, DEPLOYMENT_PRESET_SOURCES = _load_deployment_presets()


def _resolve_hf_profile(
    raw_cfg: Mapping[str, object],
    name: str,
    dtypes: tuple[str | None, str | None, str, str],
    *,
    act_dtype: str | None = None,
    require_model_type: bool = False,
) -> ModelProfile:
    """Resolve normalized HuggingFace config with explicit dtype/state overrides."""
    weight_dtype, expert_dtype, kv_dtype, state_dtype = dtypes
    norm = _normalize_hf_config(raw_cfg)
    norm.validate_supported(require_model_type=require_model_type)
    return norm.to_model_profile(
        name=name,
        weight_dtype=weight_dtype,
        expert_dtype=expert_dtype,
        kv_dtype=kv_dtype,
        state_dtype=state_dtype,
        act_dtype=act_dtype,
    )


def model_profile_from_hf_config(  # noqa: PLR0913
    raw_cfg: Mapping[str, object],
    name: str = "model",
    weight_dtype: str | None = None,
    expert_dtype: str | None = None,
    *,
    act_dtype: str | None = None,
    state_dtype: str = "bf16",
) -> ModelProfile:
    """Parse and validate a HuggingFace config mapping into a ModelProfile."""
    return _resolve_hf_profile(
        raw_cfg,
        name,
        (weight_dtype, expert_dtype, "fp8", state_dtype),
        act_dtype=act_dtype,
    )


def resolve_hf_snapshot_config(
    repo_id: str, hf_home: Path | None = None
) -> Path | None:
    """Resolve newest snapshot config.json for a HuggingFace repository."""
    raw_str = repo_id.strip()
    if "/" not in raw_str or raw_str.endswith(".json"):
        return None
    org_repo = raw_str.strip("/").replace("/", "--")
    home = hf_home or (Path.home() / ".cache" / "huggingface")
    for root in (home / "hub", home):
        repo_dir = root / f"models--{org_repo}"
        snap_dir = repo_dir / "snapshots"
        if not snap_dir.is_dir():
            continue
        ref_main = repo_dir / "refs" / "main"
        if ref_main.is_file():
            rev = ref_main.read_text("utf-8").strip()
            cand_cfg = snap_dir / rev / "config.json"
            if cand_cfg.is_file():
                return cand_cfg
        snaps = sorted(
            snap_dir.glob("*/config.json"),
            key=lambda p: (p.stat().st_mtime_ns, p.parent.name),
        )
        if snaps:
            return snaps[-1]
    return None


def resolve_model_profile(  # noqa: PLR0913
    target: object,
    weight_dtype: str | None = None,
    expert_dtype: str | None = None,
    kv_dtype: str | None = None,
    state_dtype: str | None = None,
    *,
    act_dtype: str | None = None,
) -> ModelProfile:
    """Resolve a ModelProfile from a preset, repo alias, dict, or config path."""
    kv_val = (kv_dtype or "fp8").lower()
    st_val = (state_dtype or "bf16").lower()
    dtypes = (weight_dtype, expert_dtype, kv_val, st_val)
    match target:
        case ModelProfile():
            return target
        case Mapping() as mapping:
            return _resolve_hf_profile(mapping, "model", dtypes, act_dtype=act_dtype)
        case _:
            pass

    raw_str = str(target).strip()
    path = Path(raw_str)
    if path.is_file():
        return _resolve_hf_profile(
            json.loads(path.read_text("utf-8")),
            path.parent.name or path.stem,
            dtypes,
            act_dtype=act_dtype,
        )
    if not raw_str.endswith(".json") and (path / "config.json").is_file():
        cfg_file = path / "config.json"
        return _resolve_hf_profile(
            json.loads(cfg_file.read_text("utf-8")),
            path.name or cfg_file.stem,
            dtypes,
            act_dtype=act_dtype,
        )

    tail = raw_str.lower().strip("/").split("/")[-1]
    preset_key = REGISTERED_MODEL_ALIASES.get(tail)
    if preset_key is not None and not raw_str.endswith(".json"):
        model_profile = MODEL_PRESETS[preset_key]
        resolved_w = (weight_dtype or model_profile.weight_dtype).lower()
        resolved_kv = (kv_dtype or model_profile.kv_dtype).lower()
        resolved_ed = (
            expert_dtype or weight_dtype or model_profile.expert_dtype
        ).lower()
        resolved_st = (state_dtype or model_profile.state_dtype).lower()
        if act_dtype is not None:
            resolved_act = act_dtype.lower()
        elif weight_dtype is not None:
            resolved_act = (
                "fp8"
                if resolved_w in ("fp8", "nvfp4", "mxfp4", "fp4", "int8")
                else resolved_w
            )
        else:
            resolved_act = model_profile.act_dtype
        resolved_src = (
            "flag"
            if (
                weight_dtype is not None
                or expert_dtype is not None
                or act_dtype is not None
            )
            else model_profile.dtype_source
        )
        return replace(
            model_profile,
            weight_dtype=resolved_w,
            expert_dtype=resolved_ed,
            act_dtype=resolved_act,
            kv_dtype=resolved_kv,
            state_dtype=resolved_st,
            dtype_source=resolved_src,
        )

    snap_cfg = resolve_hf_snapshot_config(raw_str)
    if snap_cfg is not None:
        return _resolve_hf_profile(
            json.loads(snap_cfg.read_text("utf-8")),
            tail,
            dtypes,
            act_dtype=act_dtype,
            require_model_type=True,
        )

    msg = (
        f"Unregistered model or missing config path: {target!r}. "
        f"Registered presets: {sorted(MODEL_PRESETS)}; "
        f"verified HF model_types: {sorted(RECOGNIZED_MODEL_TYPES)}"
    )
    raise UnsupportedArchitectureError(msg)
