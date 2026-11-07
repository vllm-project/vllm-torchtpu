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
from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import gcd
from pathlib import Path

from .. import errors

_MIN_INTERVAL = 2
_BITS_FP8 = 8
_BITS_FP4 = 4
_PRESETS_DIR = Path(__file__).resolve().parent / "data" / "presets"


class UnsupportedArchitectureError(errors.MegakernelConfigError, ValueError):
    """Raised when a model preset or HuggingFace config is unsupported or malformed."""


VERIFIED_MEGAKERNEL_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "deepseek_v2",
        "deepseek_v3",
        "glm4_moe",
        "glm5",
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
    }
)

SUPPORTED_MODEL_TYPES: frozenset[str] = VERIFIED_MEGAKERNEL_MODEL_TYPES

_UNSUPPORTED_NON_CAUSAL_TYPES: frozenset[str] = frozenset(
    {"bert", "roberta", "t5", "whisper", "vit", "clip", "dit", "flux", "sdxl"}
)

_UNSUPPORTED_SSM_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "mamba",
        "mamba2",
        "rwkv",
        "rwkv6",
        "falcon_mamba",
        "xlstm",
        "retnet",
        "hyena",
        "stripedhyena",
    }
)

_MULTIMODAL_KEYS: tuple[str, ...] = (
    "vision_config",
    "audio_config",
    "video_config",
    "mm_projector_type",
    "image_token_index",
)

_VALID_LAYER_TOKENS: frozenset[str] = frozenset(
    {
        "attention",
        "dense_mlp",
        "dsa",
        "dsa_attention",
        "full_attention",
        "gated_deltanet",
        "gdn",
        "gqa",
        "kda",
        "kda_attention",
        "kimi_linear",
        "linear_attention",
        "mamba",
        "mla",
        "mla_attention",
        "mlp",
        "moe",
        "mtp",
        "ple",
        "sliding_attention",
        "sliding_window_attention",
        "sparse_mla",
        "swa",
    }
)

_KDA_MODEL_TYPES: frozenset[str] = frozenset({"kimi_k2", "kimi_k3", "kimi_linear"})


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
    kv_dtype: str = "fp8"
    state_dtype: str = "bf16"
    model_type: str = ""
    architecture_unverified: bool = False
    layer_families: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Derive attention/MoE layer counts, periodicity, and layer families."""
        num_layers = self.num_layers
        has_experts = self.num_experts > 0
        moe_layers = self.num_moe_layers
        attn_layers = self.num_attention_layers or (num_layers - self.num_linear_layers)
        moe_layers = num_layers * has_experts if moe_layers < 0 else moe_layers
        periodic = self.is_periodic and not (0 < moe_layers < num_layers - 1)
        moe_inter = self.moe_intermediate_size or (self.intermediate_size * has_experts)
        exp_dt = self.expert_dtype or self.weight_dtype
        mtype = self.model_type.strip().lower()
        unverified = self.architecture_unverified or bool(
            mtype and mtype not in VERIFIED_MEGAKERNEL_MODEL_TYPES
        )
        families = self.layer_families or _derive_layer_families(
            self, attn_layers=attn_layers, moe_layers=moe_layers
        )
        updates = (
            ("num_attention_layers", attn_layers),
            ("num_moe_layers", moe_layers),
            ("is_periodic", periodic),
            ("moe_intermediate_size", moe_inter),
            ("expert_dtype", exp_dt),
            ("architecture_unverified", unverified),
            ("layer_families", families),
        )
        for key, val in updates:
            object.__setattr__(self, key, val)


def _derive_layer_families(
    profile: ModelProfile, *, attn_layers: int, moe_layers: int
) -> tuple[str, ...]:
    """Derive canonical layer families from a ModelProfile's layer counts."""
    families: list[str] = []
    if attn_layers > 0:
        families.append("mla" if profile.kv_lora_rank > 0 else "gqa")
    if profile.num_linear_layers > 0:
        mtype = profile.model_type.strip().lower()
        families.append("kda" if mtype in _KDA_MODEL_TYPES else "gdn")
    if moe_layers > 0:
        families.append("moe")
    if moe_layers < profile.num_layers and profile.intermediate_size > 0:
        families.append("dense_mlp")
    if 0 < moe_layers < profile.num_layers - 1:
        families.append("dense_prefix_moe")
    return tuple(families)


def _int_field(mapping: Mapping[str, object], *keys: str, default: int = 0) -> int:
    """Return the first non-zero integer value matching keys in mapping."""
    for key in keys:
        val = mapping.get(key)
        if isinstance(val, (int, float, str)) and int(val) != 0:
            return int(val)
    return default


def _extract_text_config(raw_cfg: Mapping[str, object]) -> Mapping[str, object]:
    """Extract and validate the language-model sub-config from raw_cfg."""
    if not raw_cfg:
        msg = "Empty HuggingFace config mapping"
        raise UnsupportedArchitectureError(msg)
    for sub_key in ("text_config", "language_config"):
        if sub_key in raw_cfg:
            sub = raw_cfg.get(sub_key)
            if sub is None or not isinstance(sub, Mapping):
                msg = f"Multimodal config has non-mapping {sub_key!r}"
                raise UnsupportedArchitectureError(msg)
    sub_cfg = raw_cfg.get("text_config") or raw_cfg.get("language_config")
    tc: Mapping[str, object] = sub_cfg if isinstance(sub_cfg, Mapping) else raw_cfg
    is_multimodal = (
        "text_config" in raw_cfg
        or "language_config" in raw_cfg
        or any(k in raw_cfg for k in _MULTIMODAL_KEYS)
    )
    if is_multimodal and not tc.get("hidden_size"):
        msg = "Multimodal config missing extractable text_config.hidden_size"
        raise UnsupportedArchitectureError(msg)
    return tc


def _has_linear_heads(tc: Mapping[str, object]) -> bool:
    """Check whether tc or its linear_attn_config specifies GDN/KDA heads."""
    lin_raw = tc.get("linear_attn_config")
    lin_map: Mapping[str, object] = lin_raw if isinstance(lin_raw, Mapping) else {}
    return bool(
        tc.get("linear_num_key_heads")
        or tc.get("linear_num_value_heads")
        or lin_map.get("num_key_heads")
        or lin_map.get("num_value_heads")
        or lin_map.get("num_heads")
    )


def _validate_layer_tokens(
    tc: Mapping[str, object], *, has_gdn_kda_heads: bool
) -> None:
    """Validate every token in layer_types against _VALID_LAYER_TOKENS."""
    raw_types = tc.get("layer_types")
    if not isinstance(raw_types, (list, tuple)):
        return
    for token in raw_types:
        tok_s = str(token)
        if tok_s not in _VALID_LAYER_TOKENS:
            msg = (
                f"Unknown layer_type {tok_s!r}; supported tokens: "
                f"{sorted(_VALID_LAYER_TOKENS)}"
            )
            raise UnsupportedArchitectureError(msg)
        if tok_s == "mamba" and not has_gdn_kda_heads:
            msg = "Layer token 'mamba' requires GDN/KDA head dimensions"
            raise UnsupportedArchitectureError(msg)


def _validate_required_geometry(tc: Mapping[str, object]) -> None:
    """Validate required causal-LM geometry keys on a HuggingFace config."""
    for req_key in ("num_hidden_layers", "hidden_size", "num_attention_heads"):
        if _int_field(tc, req_key) < 1:
            msg = (
                f"HuggingFace config missing required causal-LM geometry key "
                f"{req_key!r} >= 1"
            )
            raise UnsupportedArchitectureError(msg)


def supports_hf_config(
    raw_cfg: Mapping[str, object], *, require_model_type: bool = False
) -> None:
    """Validate that a HuggingFace config mapping describes a supported causal LM."""
    tc = _extract_text_config(raw_cfg)
    mtype = str(tc.get("model_type") or raw_cfg.get("model_type") or "").strip().lower()
    if require_model_type and not mtype:
        msg = "HuggingFace config missing required 'model_type' field"
        raise UnsupportedArchitectureError(msg)
    if mtype in _UNSUPPORTED_NON_CAUSAL_TYPES:
        msg = f"Unsupported non-causal-LM model_type {mtype!r}"
        raise UnsupportedArchitectureError(msg)

    has_gdn_kda_heads = _has_linear_heads(tc)
    if mtype in _UNSUPPORTED_SSM_MODEL_TYPES and not has_gdn_kda_heads:
        msg = f"Unsupported state-space model_type {mtype!r} without GDN/KDA heads"
        raise UnsupportedArchitectureError(msg)
    if ("ssm_cfg" in tc or "state_size" in tc) and not (
        tc.get("num_attention_heads") or has_gdn_kda_heads
    ):
        msg = "Unmodeled state-space config missing attention and GDN/KDA heads"
        raise UnsupportedArchitectureError(msg)
    _validate_required_geometry(tc)
    _validate_layer_tokens(tc, has_gdn_kda_heads=has_gdn_kda_heads)


def _load_model_presets() -> tuple[
    Mapping[str, ModelProfile], Mapping[str, str], Mapping[str, str]
]:
    """Load builtin model presets, aliases, and source provenance from JSON."""
    payload = json.loads((_PRESETS_DIR / "models.json").read_text("utf-8"))
    raw_presets = payload["presets"]
    presets: dict[str, ModelProfile] = {}
    sources: dict[str, str] = {}
    for slug, entry in raw_presets.items():
        sources[slug] = str(entry.get("source", f"models.json:{slug}"))
        presets[slug] = ModelProfile(
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
            routed_expert_hidden_size=int(entry.get("routed_expert_hidden_size", 0)),
            weight_dtype=str(entry.get("weight_dtype", "fp8")),
            expert_dtype=str(entry.get("expert_dtype", "")),
            model_type=str(entry.get("model_type", "")),
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

    def label(self) -> str:
        """Return the layout as `TP8 x DCP4 x EP8`, omitting degree-1 axes."""
        axes = (
            ("DP", self.dp),
            ("PP", self.pp),
            ("TP", self.tp),
            ("DCP", self.dcp),
            ("PCP", self.pcp),
            ("EP", self.ep),
        )
        return " x ".join(f"{n}{v}" for n, v in axes if v > 1 or n == "TP")


@dataclass(frozen=True)
class DeploymentPreset:
    """Shipped prefill and decode layouts for one model, used before auto-pick."""

    decode: ParallelLayout
    prefill: ParallelLayout
    saturation: ParallelLayout | None = None
    # Prefill token windows (or chunks) the shipped prefill engine compiles.
    prefill_windows: tuple[int, ...] = ()


def _parse_parallel_layout(raw: Mapping[str, object]) -> ParallelLayout:
    """Parse a ParallelLayout mapping from deployments.json."""
    raw_groups = raw.get("layer_groups")
    groups: tuple[int, ...] = (
        tuple(int(str(x)) for x in raw_groups) if type(raw_groups) is list else ()
    )
    return ParallelLayout(
        tp=int(str(raw.get("tp", 1))),
        ep=int(str(raw.get("ep", 1))),
        dp=int(str(raw.get("dp", 1))),
        pp=int(str(raw.get("pp", 1))),
        dcp=int(str(raw.get("dcp", 1))),
        pcp=int(str(raw.get("pcp", 1))),
        spec_tokens=int(str(raw.get("spec_tokens", 0))),
        layer_groups=groups,
    )


def _load_deployment_presets() -> tuple[
    Mapping[str, DeploymentPreset], Mapping[str, str]
]:
    """Load deployment layout presets and source strings from JSON."""
    payload = json.loads((_PRESETS_DIR / "deployments.json").read_text("utf-8"))
    raw_deps = payload["deployments"]
    deps: dict[str, DeploymentPreset] = {}
    sources: dict[str, str] = {}
    for slug, entry in raw_deps.items():
        sources[slug] = str(entry.get("source", f"deployments.json:{slug}"))
        sat_raw = entry.get("saturation")
        sat = _parse_parallel_layout(sat_raw) if type(sat_raw) is dict else None
        win_raw = entry.get("prefill_windows")
        windows: tuple[int, ...] = (
            tuple(int(x) for x in win_raw) if type(win_raw) is list else ()
        )
        deps[slug] = DeploymentPreset(
            decode=_parse_parallel_layout(entry["decode"]),
            prefill=_parse_parallel_layout(entry["prefill"]),
            saturation=sat,
            prefill_windows=windows,
        )
    return deps, sources


DEPLOYMENT_PRESETS, DEPLOYMENT_PRESET_SOURCES = _load_deployment_presets()


def _resolve_linear_topology(
    cfg: Mapping[str, object],
    lin_cfg: Mapping[str, object],
    num_layers: int,
) -> tuple[int, bool]:
    """Derive (num_linear_layers, is_periodic) from layer_types or interval fields."""
    interval = _int_field(cfg, "full_attention_interval")
    layer_types = cfg.get("layer_types") or ()
    kda_layers = lin_cfg.get("kda_layers") or ()
    full_attn_layers = lin_cfg.get("full_attn_layers") or ()
    if isinstance(layer_types, (list, tuple)) and len(layer_types) > 0:
        if len(layer_types) != num_layers:
            msg = (
                f"layer_types len {len(layer_types)} != num_hidden_layers {num_layers}"
            )
            raise UnsupportedArchitectureError(msg)
        num_linear_layers = sum(
            any(k in str(t).lower() for k in ("lin", "gdn", "mamba", "kda"))
            for t in layer_types
        )
        if 0 < num_linear_layers < num_layers:
            period_gcd = gcd(num_layers - num_linear_layers, num_layers)
            stride = num_layers // period_gcd
            is_periodic = list(layer_types[:stride]) * period_gcd == list(layer_types)
        else:
            is_periodic = True
        return num_linear_layers, is_periodic
    if isinstance(kda_layers, (list, tuple)) and kda_layers:
        return len(kda_layers), True
    if isinstance(full_attn_layers, (list, tuple)) and full_attn_layers:
        return max(0, num_layers - len(full_attn_layers)), True
    num_linear_layers = num_layers - num_layers // interval if interval > 1 else 0
    is_periodic = interval < _MIN_INTERVAL or num_layers % interval == 0
    return num_linear_layers, is_periodic


def _infer_quant_dtypes(
    cfg: Mapping[str, object],
    raw_cfg: Mapping[str, object],
    quant_cfg: Mapping[str, object],
    weight_dtype: str | None,
    expert_dtype: str | None,
) -> tuple[str, str]:
    """Resolve (weight_dtype, expert_dtype) from quantization_config and overrides."""
    torch_dt = (
        cfg.get("dtype")
        or raw_cfg.get("dtype")
        or cfg.get("torch_dtype")
        or raw_cfg.get("torch_dtype")
        or "bf16"
    )
    bits = _int_field(quant_cfg, "bits", "w_bit")
    quant_w_type = str(quant_cfg.get("type") or "").strip().lower()
    cfg_groups = quant_cfg.get("config_groups")
    if isinstance(cfg_groups, Mapping):
        for grp in cfg_groups.values():
            if isinstance(grp, Mapping) and isinstance(grp.get("weights"), Mapping):
                w_grp = grp["weights"]
                nb = _int_field(w_grp, "num_bits")
                if bits == 0 and nb > 0:
                    bits = nb
                if not quant_w_type and w_grp.get("type"):
                    quant_w_type = str(w_grp.get("type")).strip().lower()
                if bits > 0 and quant_w_type:
                    break
    quant_tag = (
        f"{quant_cfg.get('quant_method', '')} "
        f"{quant_cfg.get('format', '')} "
        f"{quant_cfg.get('quant_algo', '')} {torch_dt}"
    ).lower()
    has_fp8_tag = any(k in quant_tag for k in ("fp8", "float8", "e4m3"))
    is_int_quant = quant_w_type in ("int", "int8") or any(
        k in quant_tag
        for k in ("int8", "w8a8", "smoothquant", "gptq", "awq", "int-quantized")
    )
    if has_fp8_tag:
        fallback_dt = "fp8"
    elif bits == _BITS_FP8:
        fallback_dt = "int8" if is_int_quant else "fp8"
    elif is_int_quant and "int8" in quant_tag:
        fallback_dt = "int8"
    else:
        fallback_dt = "fp32" if "32" in quant_tag else "bf16"
    inferred_dt = (
        "nvfp4"
        if "nvfp4" in quant_tag
        else ("mxfp4" if "mxfp4" in quant_tag or bits == _BITS_FP4 else fallback_dt)
    )
    resolved_wd = str(
        weight_dtype
        or cfg.get("weight_dtype")
        or raw_cfg.get("weight_dtype")
        or inferred_dt
    ).lower()
    resolved_ed = str(
        expert_dtype
        or quant_cfg.get("expert_dtype")
        or cfg.get("expert_dtype")
        or raw_cfg.get("expert_dtype")
        or resolved_wd
    ).lower()
    return resolved_wd, resolved_ed


def model_profile_from_hf_config(
    raw_cfg: Mapping[str, object],
    name: str = "model",
    weight_dtype: str | None = None,
    expert_dtype: str | None = None,
    kv_dtype: str = "fp8",
    **extra_opts: object,
) -> ModelProfile:
    """Parse and validate a HuggingFace config mapping into a ModelProfile."""
    state_dtype = str(extra_opts.get("state_dtype") or "bf16")
    require_model_type = bool(extra_opts.get("require_model_type", False))
    supports_hf_config(raw_cfg, require_model_type=require_model_type)
    sub_cfg = raw_cfg.get("text_config") or raw_cfg.get("language_config")
    cfg: Mapping[str, object] = sub_cfg if isinstance(sub_cfg, Mapping) else raw_cfg
    lin_raw = cfg.get("linear_attn_config")
    lin_cfg: Mapping[str, object] = lin_raw if isinstance(lin_raw, Mapping) else {}
    quant_raw = cfg.get("quantization_config") or raw_cfg.get("quantization_config")
    quant_cfg: Mapping[str, object] = (
        quant_raw if isinstance(quant_raw, Mapping) else {}
    )
    merged: dict[str, object] = {**lin_cfg, **cfg}

    num_layers = _int_field(cfg, "num_hidden_layers")
    hidden_size = _int_field(cfg, "hidden_size")
    num_heads = _int_field(cfg, "num_attention_heads")

    num_linear_layers, is_periodic = _resolve_linear_topology(cfg, lin_cfg, num_layers)
    lin_k_heads = _int_field(
        merged, "linear_num_key_heads", "num_key_heads", "num_k_heads", "num_heads"
    )
    lin_v_heads = _int_field(
        merged,
        "linear_num_value_heads",
        "num_value_heads",
        "num_v_heads",
        "num_heads",
    )
    if num_linear_layers > 0 and (lin_k_heads < 1 or lin_v_heads < 1):
        msg = "Model with linear-attention layers missing GDN/KDA key/value head counts"
        raise UnsupportedArchitectureError(msg)

    num_experts = _int_field(
        cfg, "num_experts", "n_routed_experts", "num_local_experts"
    )
    mlp_only = cfg.get("mlp_only_layers")
    dense_prefix = _int_field(cfg, "first_k_dense_replace") or (
        len(mlp_only) if isinstance(mlp_only, (list, tuple)) else 0
    )
    resolved_wd, resolved_ed = _infer_quant_dtypes(
        cfg, raw_cfg, quant_cfg, weight_dtype, expert_dtype
    )

    ffn_inter = _int_field(cfg, "intermediate_size", "ffn_dim", default=4 * hidden_size)
    moe_inter = _int_field(
        cfg,
        "moe_intermediate_size",
        "latent_moe_dim",
        default=ffn_inter * (num_experts > 0),
    )
    num_shared = _int_field(cfg, "n_shared_experts", "num_shared_experts")
    shared_inter = _int_field(
        cfg, "shared_expert_intermediate_size", default=num_shared * moe_inter
    )
    top_k = _int_field(
        cfg,
        "num_experts_per_tok",
        "num_experts_per_token",
        "num_selected_experts",
    ) * (num_experts > 0)
    mtype = (
        str(cfg.get("model_type") or raw_cfg.get("model_type") or "").strip().lower()
    )

    return ModelProfile(
        name=name,
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=ffn_inter,
        num_attention_heads=num_heads,
        num_kv_heads=_int_field(cfg, "num_key_value_heads", default=num_heads),
        head_dim=_int_field(
            cfg, "v_head_dim", "head_dim", default=hidden_size // num_heads
        ),
        num_linear_layers=num_linear_layers,
        linear_key_heads=lin_k_heads,
        linear_value_heads=lin_v_heads,
        num_experts=num_experts,
        top_k=top_k,
        moe_intermediate_size=moe_inter,
        shared_expert_intermediate_size=shared_inter,
        num_moe_layers=max(0, num_layers - dense_prefix) * (num_experts > 0),
        num_attention_layers=0,
        kv_lora_rank=_int_field(cfg, "kv_lora_rank"),
        q_lora_rank=_int_field(cfg, "q_lora_rank"),
        qk_nope_head_dim=_int_field(cfg, "qk_nope_head_dim"),
        qk_rope_head_dim=_int_field(cfg, "qk_rope_head_dim"),
        linear_key_dim=_int_field(
            merged, "linear_key_head_dim", "key_head_dim", "head_k_dim", default=128
        ),
        linear_value_dim=_int_field(
            merged,
            "linear_value_head_dim",
            "value_head_dim",
            "head_v_dim",
            default=128,
        ),
        linear_conv_kernel=_int_field(
            merged,
            "linear_conv_kernel_dim",
            "short_conv_kernel_size",
            default=4,
        ),
        routed_expert_hidden_size=_int_field(cfg, "routed_expert_hidden_size"),
        is_periodic=is_periodic,
        ar_slots_per_layer=2,
        weight_dtype=resolved_wd,
        expert_dtype=resolved_ed,
        kv_dtype=kv_dtype.lower(),
        state_dtype=state_dtype.lower(),
        model_type=mtype,
    )


def resolve_model_profile(
    target: object,
    weight_dtype: str | None = None,
    expert_dtype: str | None = None,
    kv_dtype: str | None = None,
    state_dtype: str | None = None,
) -> ModelProfile:
    """Resolve a ModelProfile from a preset, repo alias, dict, or config path."""
    if isinstance(target, ModelProfile):
        return target
    kv_val = kv_dtype or "fp8"
    st_val = state_dtype or "bf16"
    if isinstance(target, Mapping):
        return model_profile_from_hf_config(
            target,
            "model",
            weight_dtype,
            expert_dtype,
            kv_val,
            state_dtype=st_val,
        )

    raw_str = str(target).strip()
    path = Path(raw_str)
    if path.is_file():
        return model_profile_from_hf_config(
            json.loads(path.read_text("utf-8")),
            path.parent.name or path.stem,
            weight_dtype,
            expert_dtype,
            kv_val,
            state_dtype=st_val,
        )
    if not raw_str.endswith(".json") and (path / "config.json").is_file():
        cfg_file = path / "config.json"
        return model_profile_from_hf_config(
            json.loads(cfg_file.read_text("utf-8")),
            path.name or cfg_file.stem,
            weight_dtype,
            expert_dtype,
            kv_val,
            state_dtype=st_val,
        )

    tail = raw_str.lower().strip("/").split("/")[-1]
    preset_key = REGISTERED_MODEL_ALIASES.get(tail)
    if preset_key is not None and not raw_str.endswith(".json"):
        prof = MODEL_PRESETS[preset_key]
        resolved_w = (weight_dtype or prof.weight_dtype).lower()
        resolved_kv = (kv_dtype or prof.kv_dtype).lower()
        resolved_ed = (expert_dtype or weight_dtype or prof.expert_dtype).lower()
        resolved_st = (state_dtype or prof.state_dtype).lower()
        return replace(
            prof,
            weight_dtype=resolved_w,
            expert_dtype=resolved_ed,
            kv_dtype=resolved_kv,
            state_dtype=resolved_st,
        )

    if "/" in raw_str and not raw_str.endswith(".json"):
        org_repo = raw_str.strip("/").replace("/", "--")
        hf_home = Path.home() / ".cache" / "huggingface"
        for root in (hf_home / "hub", hf_home):
            repo_dir = root / f"models--{org_repo}"
            snap_dir = repo_dir / "snapshots"
            if snap_dir.is_dir():
                cfg_file_path: Path | None = None
                ref_main = repo_dir / "refs" / "main"
                if ref_main.is_file():
                    rev = ref_main.read_text("utf-8").strip()
                    cand_cfg = snap_dir / rev / "config.json"
                    if cand_cfg.is_file():
                        cfg_file_path = cand_cfg
                if cfg_file_path is None:
                    snaps = sorted(
                        snap_dir.glob("*/config.json"),
                        key=lambda p: (p.stat().st_mtime_ns, p.parent.name),
                    )
                    if snaps:
                        cfg_file_path = snaps[-1]
                if cfg_file_path is not None:
                    return model_profile_from_hf_config(
                        json.loads(cfg_file_path.read_text("utf-8")),
                        tail,
                        weight_dtype,
                        expert_dtype,
                        kv_val,
                        state_dtype=st_val,
                        require_model_type=True,
                    )

    msg = (
        f"Unregistered model or missing config path: {target!r}. "
        f"Registered presets: {sorted(MODEL_PRESETS)}; "
        f"verified HF model_types: {sorted(VERIFIED_MEGAKERNEL_MODEL_TYPES)}"
    )
    raise UnsupportedArchitectureError(msg)
