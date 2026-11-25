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
"""Model profile registry, HuggingFace config validation, and TPU cache policy.

HuggingFace `config.json` files own model geometry and weight quantization
(`num_layers`, `hidden_size`, `weight_dtype`, `expert_dtype`, `weight_block_k`).
TPU serving cache dtypes (`kv_dtype="fp8"`, GDN recurrent `state_dtype="bf16"`,
KDA `state_dtype="fp32"`) belong to the serving cache policy matching
`megakernel/core` unless overridden explicitly.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from math import ceil, gcd
from pathlib import Path
from typing import NoReturn, TypedDict, Unpack, cast

from vllm_torchtpu.megakernel.core import errors

from .constants import (
    BITS_FP8,
    DTYPE_BYTES,
    FP8_DTYPES,
    INDEXER_SCALE_BLOCK_SIZE,
    KDA_MODEL_TYPES,
    MIN_HEAD_DIM,
    MIN_INTERVAL,
    MULTIMODAL_KEYS,
    PRESETS_DIR,
    STRUCTURAL_LAYER_TAGS,
    SUPPORTED_ONCHIP_FAMILIES,
    VALID_LAYER_TOKENS,
    WORD_ALIGN_BYTES,
)
from .specs import (
    _checked_int,
    as_mapping,
    as_scalar_number,
    as_sequence,
    int_tuple,
    scalar_float,
    scalar_int,
)


class UnsupportedArchitectureError(errors.MegakernelConfigError, ValueError):
    """Raised when a model preset or HuggingFace config is unsupported or malformed."""


ONBOARDING_DOC_REF = "src/vllm_torchtpu/megakernel/advisor/README.md (Adding a Model)"
ATTENTION_MODULE_PREFIXES: tuple[str, ...] = (
    "model.layers",
    "model.language_model.layers",
    "language_model.model.layers",
)
ATTENTION_PROJECTION_NAMES: tuple[str, ...] = ("q_proj", "q_a_proj")
VALID_INDEXER_SCHEDULE_TOKENS: frozenset[str] = frozenset({"full", "shared", "none"})
SUB_BYTE_QUANT_GROUP_METHOD = "compressed-tensors"
SUPPORTED_QUANT_SCHEMES: frozenset[tuple[str, str | None]] = frozenset(
    {
        ("fp8", None),
        (SUB_BYTE_QUANT_GROUP_METHOD, "mxfp4-pack-quantized"),
    }
)
QUANT_SUB_8BIT_WEIGHT_DTYPES: frozenset[str] = FP8_DTYPES | frozenset(
    {"nvfp4", "mxfp4", "fp4"}
)
DTYPE_BASIS_PHRASES: Mapping[str, str] = {"config": "dtypes from the cited config"}
PRESET_DTYPE_BASES: frozenset[str] = frozenset(DTYPE_BASIS_PHRASES)
PRESET_REQUIRED_INT_KEYS: tuple[str, ...] = (
    "num_layers",
    "hidden_size",
    "num_attention_heads",
    "num_kv_heads",
    "head_dim",
)
PRESET_OPTIONAL_INT_DEFAULTS: Mapping[str, int] = {
    "intermediate_size": 0,
    "num_linear_layers": 0,
    "linear_key_heads": 0,
    "linear_value_heads": 0,
    "linear_key_dim": 128,
    "linear_value_dim": 128,
    "linear_conv_kernel": 4,
    "num_experts": 0,
    "top_k": 0,
    "moe_intermediate_size": 0,
    "shared_expert_intermediate_size": 0,
    "num_moe_layers": -1,
    "kv_lora_rank": 0,
    "q_lora_rank": 0,
    "qk_nope_head_dim": 0,
    "qk_rope_head_dim": 0,
    "index_topk": 0,
    "index_head_dim": 0,
    "index_n_heads": 0,
    "num_indexer_layers": 0,
    "routed_expert_hidden_size": 0,
    "weight_block_k": 0,
    "residual_streams": 1,
}


class _ProfileIntFields(TypedDict):
    """Typed integer geometry and schedule fields unpacked into `ModelProfile`."""

    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    num_linear_layers: int
    linear_key_heads: int
    linear_value_heads: int
    linear_key_dim: int
    linear_value_dim: int
    linear_conv_kernel: int
    num_experts: int
    top_k: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    num_moe_layers: int
    kv_lora_rank: int
    q_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    index_topk: int
    index_head_dim: int
    index_n_heads: int
    num_indexer_layers: int
    routed_expert_hidden_size: int
    weight_block_k: int
    residual_streams: int


class DtypeOverrides(TypedDict, total=False):
    """Optional keyword dtype overrides shared across model-resolution entrypoints."""

    weight_dtype: str | None
    expert_dtype: str | None
    kv_dtype: str | None
    state_dtype: str | None
    act_dtype: str | None


@dataclass(frozen=True, slots=True)
class _QuantFields:
    """Normalized quantization metadata parsed from an HF config mapping."""

    torch_dtype: str
    quant_method: str
    raw_weight_dtype: str
    raw_expert_dtype: str
    raw_activation_dtype: str
    weight_block_k: int
    has_quant_config: bool
    has_explicit_torch_dtype: bool
    dense_unquantized: bool


@dataclass(frozen=True)
class ModelProfile:
    """Architecture geometry and dtype profile for megakernel scope evaluation."""

    name: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int = 0
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
    index_topk: int = 0
    index_head_dim: int = 0
    index_n_heads: int = 0
    num_indexer_layers: int = 0
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
    layer_families: tuple[str, ...] = ()
    has_attn_output_gate: bool = False
    kv_compress_factor: float = 1.0
    residual_streams: int = 1
    dtype_source: str = ""


def _arch_err(msg: str) -> NoReturn:
    """Raise `UnsupportedArchitectureError` with message `msg`."""
    raise UnsupportedArchitectureError(msg)


def _cfg_err(msg: str) -> NoReturn:
    """Raise `MegakernelConfigError` with message `msg`."""
    raise errors.MegakernelConfigError(msg)


def _first_str(
    mappings: tuple[Mapping[str, object], ...],
    keys: tuple[str, ...],
    default: str = "",
) -> str:
    """Return the first truthy string value matching `keys` across `mappings`."""
    for key in keys:
        for mapping in mappings:
            if val := mapping.get(key):
                return str(val)
    return default


def has_dsa_profile(profile: ModelProfile) -> bool:
    """Return whether `profile` carries sparse attention and indexer geometry."""
    return (
        profile.index_topk > 0
        and profile.index_n_heads > 0
        and profile.index_head_dim > 0
    )


def indexer_record_bytes(index_head_dim: int) -> int:
    """Return packed 4-byte-aligned FP8 key plus block-scale indexer record bytes."""
    if index_head_dim <= 0:
        return 0
    raw_bytes = index_head_dim + ceil(index_head_dim / INDEXER_SCALE_BLOCK_SIZE)
    return ((raw_bytes + WORD_ALIGN_BYTES - 1) // WORD_ALIGN_BYTES) * WORD_ALIGN_BYTES


def _derive_layer_families(
    profile: ModelProfile, *, attn_layers: int, moe_layers: int
) -> tuple[str, ...]:
    """Derive canonical layer families from resolved model layer counts."""
    families: list[str] = []
    if attn_layers > 0:
        families.append(
            "dsa"
            if has_dsa_profile(profile)
            else ("mla" if profile.kv_lora_rank > 0 else "gqa")
        )
    if profile.num_linear_layers > 0:
        normalized_model_type = profile.model_type.strip().lower()
        families.append("kda" if normalized_model_type in KDA_MODEL_TYPES else "gdn")
    if moe_layers > 0:
        families.append("moe")
    if moe_layers < profile.num_layers and profile.intermediate_size > 0:
        families.append("dense_mlp")
    if 0 < moe_layers < profile.num_layers:
        families.append("dense_prefix_moe")
    return tuple(families)


def _require_known_dtype(raw_val: str, field_name: str) -> str:
    """Normalize `raw_val` and validate it against `DTYPE_BYTES`."""
    normalized = raw_val.strip().lower()
    if normalized not in DTYPE_BYTES:
        _cfg_err(
            f"Unsupported {field_name}={raw_val!r}; expected {sorted(DTYPE_BYTES)}"
        )
    return normalized


def _validate_dtype_fields(profile: ModelProfile) -> None:
    """Validate all non-empty dtype fields on `profile` against `DTYPE_BYTES`."""
    for field_name in (
        "weight_dtype",
        "expert_dtype",
        "act_dtype",
        "kv_dtype",
        "state_dtype",
    ):
        if dtype_val := getattr(profile, field_name):
            _require_known_dtype(dtype_val, field_name)


def finalize_model_profile(profile: ModelProfile) -> ModelProfile:
    """Return an immutable ModelProfile with derived layer metadata populated."""
    num_layers = profile.num_layers
    if not (0 <= profile.num_linear_layers <= num_layers):
        _cfg_err(
            f"Model {profile.name!r} num_linear_layers ({profile.num_linear_layers}) "
            f"must lie in [0, num_layers={num_layers}]"
        )
    has_exp = profile.num_experts > 0
    attn_layers = profile.num_attention_layers or (
        num_layers - profile.num_linear_layers
    )
    moe = num_layers * has_exp if profile.num_moe_layers < 0 else profile.num_moe_layers
    idx_layers = (
        0
        if not has_dsa_profile(profile)
        else (
            attn_layers
            if profile.num_indexer_layers <= 0
            else min(attn_layers, profile.num_indexer_layers)
        )
    )
    periodic = (
        profile.is_periodic
        and not (0 < moe < num_layers)
        and not (0 < idx_layers < attn_layers)
    )
    w_dtype = profile.weight_dtype.strip().lower()
    default_act_dtype = "fp8" if w_dtype in QUANT_SUB_8BIT_WEIGHT_DTYPES else w_dtype
    finalized = replace(
        profile,
        num_attention_layers=attn_layers,
        num_moe_layers=moe,
        num_indexer_layers=idx_layers,
        is_periodic=periodic,
        moe_intermediate_size=profile.moe_intermediate_size
        or (profile.intermediate_size * has_exp),
        weight_dtype=w_dtype,
        expert_dtype=(profile.expert_dtype or w_dtype).strip().lower(),
        act_dtype=(profile.act_dtype or default_act_dtype).strip().lower(),
        kv_dtype=profile.kv_dtype.strip().lower(),
        state_dtype=profile.state_dtype.strip().lower(),
        layer_families=profile.layer_families
        or _derive_layer_families(profile, attn_layers=attn_layers, moe_layers=moe),
    )
    _validate_dtype_fields(finalized)
    return finalized


def _int_field(
    mapping: Mapping[str, object], keys: tuple[str, ...], *, default: int = 0
) -> int:
    """Return the first non-zero integer value matching keys in mapping."""
    for key in keys:
        val = mapping.get(key)
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)):
            if not math.isfinite(float(val)):
                _arch_err(f"Invalid non-finite number for {key!r}: {val!r}")
            if int(val) != 0:
                return int(val)
        elif isinstance(val, str) and val.strip():
            try:
                parsed = int(val.strip())
            except ValueError as exc:
                msg = f"Invalid integer value for {key!r}: {val!r}"
                raise UnsupportedArchitectureError(msg) from exc
            if parsed != 0:
                return parsed
    return default


def _default_serving_state_dtype(model_type: str) -> str:
    """Return the default TPU serving recurrent cache dtype for `model_type`.

    HuggingFace checkpoints record training/reference SSM precision (such as
    `mamba_ssm_dtype: "float32"` on Qwen3.5), not the serving cache layout.
    `megakernel/core` stores GDN recurrent state in `bf16`, whereas KDA
    (`KDA_MODEL_TYPES`) uses `fp32` recurrent state.
    """
    normalized = model_type.strip().lower().removesuffix("_text")
    return "fp32" if normalized in KDA_MODEL_TYPES else "bf16"


def _extract_text_subconfig(raw_config: Mapping[str, object]) -> Mapping[str, object]:
    """Extract and validate text sub-configuration from raw HF config."""
    raw_map = as_mapping(raw_config)
    if not raw_map:
        _arch_err("Empty HuggingFace config mapping")
    sub_config = as_mapping(raw_map.get("text_config"))
    if "text_config" in raw_map and sub_config is None:
        _arch_err("Multimodal config has non-mapping 'text_config'")
    text_config = sub_config if sub_config is not None else raw_map
    is_multimodal = "text_config" in raw_map or any(
        k in raw_map for k in MULTIMODAL_KEYS
    )
    if is_multimodal and not text_config.get("hidden_size"):
        _arch_err("Multimodal config missing extractable text_config.hidden_size")
    return text_config


def _is_periodic_schedule(flags: Sequence[bool]) -> bool:
    """Return whether `flags` matches a core `SwitchSlot` repeating cycle.

    Per `SwitchSlot` in `megakernel/core/spec.py`, branches run in consecutive
    layers per cycle, so a valid period `p` dividing `len(flags)` repeats
    verbatim with at most two contiguous runs inside one cycle.
    """
    total = len(flags)
    if total == 0:
        return True
    flags_list = list(flags)
    max_runs_per_cycle = 2
    for period in range(1, total + 1):
        if total % period != 0:
            continue
        cycle = flags_list[:period]
        if cycle * (total // period) != flags_list:
            continue
        runs = 1 + sum(cycle[idx] != cycle[idx - 1] for idx in range(1, period))
        if runs <= max_runs_per_cycle:
            return True
    return False


def _validated_layer_indices(
    raw_indices: object, key: str, num_layers: int
) -> list[int] | None:
    """Validate non-empty 0-based layer index sequences, or return None."""
    seq = as_sequence(raw_indices)
    if not seq:
        return None
    validated: list[int] = []
    seen: set[int] = set()
    for raw_idx in seq:
        if type(raw_idx) is not int or not (0 <= raw_idx < num_layers):
            _arch_err(f"{key} index {raw_idx!r} out of range [0, {num_layers})")
        if raw_idx in seen:
            _arch_err(f"{key} contains duplicate layer index {raw_idx}")
        seen.add(raw_idx)
        validated.append(raw_idx)
    return validated


def _resolve_linear_schedule(
    text_config: Mapping[str, object],
    linear_config: Mapping[str, object],
    num_layers: int,
    raw_layer_types: tuple[object, ...] | None,
) -> tuple[int, bool]:
    """Resolve `(num_linear_layers, is_periodic)` from HF layer schedule fields."""
    if raw_layer_types:
        if len(raw_layer_types) != num_layers:
            _arch_err(
                f"layer_types len {len(raw_layer_types)} != "
                f"num_hidden_layers {num_layers}"
            )
        flags = [
            any(kind in str(layer_token).lower() for kind in ("lin", "gdn", "kda"))
            for layer_token in raw_layer_types
        ]
        return sum(flags), _is_periodic_schedule(flags)
    for key, is_linear in (("kda_layers", True), ("full_attn_layers", False)):
        indices = _validated_layer_indices(linear_config.get(key), key, num_layers)
        if indices is not None:
            index_set = set(indices)
            flags = [
                (layer_idx in index_set) if is_linear else (layer_idx not in index_set)
                for layer_idx in range(num_layers)
            ]
            return sum(flags), _is_periodic_schedule(flags)
    interval = _int_field(text_config, ("full_attention_interval",))
    if interval >= MIN_INTERVAL and num_layers > 0:
        flags = [(layer_idx + 1) % interval != 0 for layer_idx in range(num_layers)]
        return sum(flags), _is_periodic_schedule(flags)
    return 0, True


def _compile_skip_pattern(pattern: str) -> Callable[[str], re.Match[str] | None]:
    """Compile one quantization skip pattern into a bound matcher."""
    if pattern.startswith("re:"):
        try:
            return re.compile(pattern.removeprefix("re:")).match
        except re.error as exc:
            msg = f"Invalid quantization skip pattern {pattern!r}: {exc}"
            raise UnsupportedArchitectureError(msg) from exc
    glob_body = ".*".join(re.escape(part) for part in pattern.split("*"))
    return re.compile(rf"(?:^|\.){glob_body}(?:\.|$)").search


def _quant_skips_attention(
    quant_map: Mapping[str, object], num_layers: int, full_attn_layers: int
) -> bool:
    """Return whether quantization skips query projections on full-attention layers."""
    patterns = [
        str(e)
        for k in ("ignore", "modules_to_not_convert")
        for e in (as_sequence(quant_map.get(k)) or ())
    ]
    if not patterns or full_attn_layers <= 0:
        return False
    matchers = tuple(_compile_skip_pattern(p) for p in patterns)
    covered = 0
    for idx in range(num_layers):
        probes = (
            f"{module_prefix}.{idx}.self_attn.{projection_name}"
            for module_prefix in ATTENTION_MODULE_PREFIXES
            for projection_name in ATTENTION_PROJECTION_NAMES
        )
        if any(m(probe) is not None for probe in probes for m in matchers):
            covered += 1
            if covered >= full_attn_layers:
                return True
    return False


def _extract_compressed_tensors_weights(
    quant_map: Mapping[str, object],
) -> tuple[int, int]:
    """Extract `(bits, group_size)` from `config_groups.*.weights`."""
    bits = _int_field(quant_map, ("bits", "w_bit"))
    group_size = 0
    for group in (as_mapping(quant_map.get("config_groups")) or {}).values():
        group_map = as_mapping(group)
        weights_map = (
            as_mapping(group_map.get("weights")) if group_map is not None else None
        )
        if weights_map is None:
            continue
        bits = bits or _int_field(weights_map, ("num_bits",))
        group_size = group_size or _int_field(weights_map, ("group_size",))
        if bits > 0 and group_size > 0:
            break
    return bits, group_size


def _extract_quant_fields(
    text_config: Mapping[str, object],
    raw_config: Mapping[str, object],
    num_layers: int,
    full_attn_layers: int,
) -> _QuantFields:
    """Extract normalized quantization fields from HF config mappings."""
    raw_qcfg = text_config.get("quantization_config")
    if raw_qcfg is None:
        raw_qcfg = raw_config.get("quantization_config")
    if raw_qcfg is not None and as_mapping(raw_qcfg) is None:
        _arch_err(
            f"quantization_config must be a mapping, got {type(raw_qcfg).__name__}"
        )
    quant_map = as_mapping(raw_qcfg) or {}
    q_method = str(quant_map.get("quant_method") or "").strip().lower()
    raw_format = quant_map.get("format")
    q_format = (
        str(raw_format).strip().lower()
        if raw_format is not None and str(raw_format).strip()
        else None
    )
    if quant_map and (q_method, q_format) not in SUPPORTED_QUANT_SCHEMES:
        supported = sorted(
            (method, fmt or "<none>") for method, fmt in SUPPORTED_QUANT_SCHEMES
        )
        _arch_err(
            f"Unsupported quantization_config scheme {(q_method, q_format)!r}; "
            f"supported: {supported}"
        )
    cfg_pair = (text_config, raw_config)
    has_torch_dtype = bool(_first_str(cfg_pair, ("dtype", "torch_dtype")))
    torch_dtype = _first_str(cfg_pair, ("dtype", "torch_dtype"), "bf16")
    bits, group_size = _extract_compressed_tensors_weights(quant_map)
    raw_act = _first_str((quant_map, text_config, raw_config), ("act_dtype",)).lower()
    if (
        not raw_act
        and str(quant_map.get("activation_scheme") or "").strip().lower() == "none"
    ):
        raw_act = "bf16"
    block_size_seq = as_sequence(
        quant_map.get("weight_block_size") or quant_map.get("block_size")
    )
    block_k = _int_field(quant_map, ("weight_block_k", "block_k")) or _int_field(
        text_config, ("weight_block_k",)
    )
    if (
        not block_k
        and block_size_seq
        and (last := as_scalar_number(block_size_seq[-1]))
        and int(last) > 0
    ):
        block_k = int(last)
    if (
        not block_k
        and q_method == SUB_BYTE_QUANT_GROUP_METHOD
        and 0 < bits < BITS_FP8
        and group_size > 0
    ):
        block_k = group_size
    return _QuantFields(
        torch_dtype=torch_dtype,
        quant_method=q_method,
        raw_weight_dtype=_first_str(cfg_pair, ("weight_dtype",)).lower(),
        raw_expert_dtype=_first_str(
            (quant_map, text_config, raw_config), ("expert_dtype",)
        ).lower(),
        raw_activation_dtype=raw_act,
        weight_block_k=block_k,
        has_quant_config=bool(quant_map),
        has_explicit_torch_dtype=has_torch_dtype,
        dense_unquantized=_quant_skips_attention(
            quant_map, num_layers, full_attn_layers
        ),
    )


def _infer_quant_dtypes_with_source(quant: _QuantFields) -> tuple[str, str, str]:
    """Infer `(weight_dtype, expert_dtype, dtype_source)` from quant metadata."""
    normalized_torch = quant.torch_dtype.lower()
    unquantized = (
        "fp32"
        if any(token in normalized_torch for token in ("fp32", "float32"))
        else "bf16"
    )
    if quant.quant_method == "fp8":
        inferred = "fp8"
    elif quant.quant_method == SUB_BYTE_QUANT_GROUP_METHOD:
        inferred = "mxfp4"
    else:
        inferred = unquantized
    dense_default = unquantized if quant.dense_unquantized else inferred
    w_dtype = str(quant.raw_weight_dtype or dense_default).lower()
    exp_dtype = str(
        quant.raw_expert_dtype or (inferred if quant.dense_unquantized else w_dtype)
    ).lower()
    if quant.dense_unquantized:
        src = "quantization_config (experts); default (unquantized dense)"
    elif quant.raw_weight_dtype or quant.raw_expert_dtype or quant.has_quant_config:
        src = "quantization_config"
    elif quant.has_explicit_torch_dtype:
        src = "default (torch_dtype; no quantization_config)"
    else:
        src = "default (bf16 fallback; no quantization_config)"
    return w_dtype, exp_dtype, src


def _classify_indexer_schedule_entry(entry: object, layer_idx: int) -> bool:
    """Return whether one schedule entry marks a full indexer layer, or raise."""
    if isinstance(entry, str):
        norm = entry.strip().lower()
        if norm not in VALID_INDEXER_SCHEDULE_TOKENS:
            valid = sorted(VALID_INDEXER_SCHEDULE_TOKENS)
            _arch_err(
                f"Invalid indexer entry at layer {layer_idx}: {entry!r}; "
                f"expected {valid}"
            )
        return norm == "full"
    num = as_scalar_number(entry)
    if num is None or num < 0.0:
        _arch_err(
            f"Invalid numeric indexer entry at layer {layer_idx}: {entry!r}; "
            "expected >= 0"
        )
    return num > 0.0


def _resolve_indexer_layers(
    text_config: Mapping[str, object],
    num_layers: int,
    attn_layers: int,
    index_dims: tuple[int, int, int],
) -> int:
    """Resolve the count of full sparse-indexer layers from HF config keys."""
    if min(index_dims) <= 0:
        return 0
    if (explicit := _int_field(text_config, ("num_indexer_layers",))) > 0:
        return min(attn_layers, explicit)
    raw_sched = text_config.get("indexer_types")
    if raw_sched is None:
        raw_sched = text_config.get("index_topk_pattern")
    if raw_sched is not None:
        seq = as_sequence(raw_sched)
        if seq is None or len(seq) != num_layers:
            _arch_err(
                f"Indexer schedule length must equal num_hidden_layers ({num_layers})"
            )
        full = sum(
            1
            for i, entry in enumerate(seq)
            if _classify_indexer_schedule_entry(entry, i)
        )
        if full <= 0:
            _arch_err("Indexer schedule must include at least one full indexer layer")
        return min(attn_layers, full)
    freq = _int_field(text_config, ("index_topk_freq",), default=1)
    offset = _int_field(text_config, ("index_skip_topk_offset",), default=0)
    if freq > 1:
        sched = sum(
            1 for i in range(num_layers) if i < offset or (i - offset + 1) % freq == 0
        )
        if sched <= 0:
            _arch_err(f"index_topk_freq={freq} selects zero indexer layers")
        return min(attn_layers, sched)
    return attn_layers


def _resolve_dsa_fields(
    text_config: Mapping[str, object], num_layers: int, attn_layers: int
) -> tuple[int, int, int, int]:
    """Return `(index_topk, index_head_dim, index_n_heads, num_indexer_layers)`."""
    dsa_map = as_mapping(text_config.get("dsa_config")) or {}
    cfg: dict[str, object] = {**dsa_map, **text_config}
    if (dsa_layers := as_sequence(dsa_map.get("dsa_layers"))) and not cfg.get(
        "num_indexer_layers"
    ):
        cfg["num_indexer_layers"] = len(dsa_layers)
    topk = _int_field(cfg, ("index_topk",))
    hdim = _int_field(cfg, ("index_head_dim",))
    nheads = _int_field(cfg, ("index_n_heads", "index_num_heads"))
    return (
        topk,
        hdim,
        nheads,
        _resolve_indexer_layers(cfg, num_layers, attn_layers, (topk, hdim, nheads)),
    )


def _extract_moe_dims(
    text_config: Mapping[str, object], num_layers: int, hidden_size: int
) -> tuple[int, int, int, int, int, int]:
    """Return MoE layer and hidden-dimension tuple from `text_config`."""
    num_experts = _int_field(text_config, ("num_experts", "n_routed_experts"))
    mlp_only = as_sequence(text_config.get("mlp_only_layers"))
    dense_prefix_layers = _int_field(text_config, ("first_k_dense_replace",)) or (
        len(mlp_only) if mlp_only is not None else 0
    )
    moe_layers = max(0, num_layers - dense_prefix_layers) * (num_experts > 0)
    default_ffn_size = (
        0 if (num_layers > 0 and moe_layers == num_layers) else 4 * hidden_size
    )
    intermediate_size = _int_field(
        text_config, ("intermediate_size",), default=default_ffn_size
    )
    moe_inter = _int_field(
        text_config,
        ("moe_intermediate_size",),
        default=intermediate_size * (num_experts > 0),
    )
    n_shared = _int_field(text_config, ("n_shared_experts", "num_shared_experts"))
    shared_inter = _int_field(
        text_config,
        ("shared_expert_intermediate_size",),
        default=n_shared * moe_inter,
    )
    top_k = _int_field(
        text_config, ("num_experts_per_tok", "num_experts_per_token")
    ) * (num_experts > 0)
    return num_experts, top_k, intermediate_size, moe_inter, shared_inter, moe_layers


def _validate_hf_profile(
    profile: ModelProfile, raw_layer_types: tuple[object, ...] | None
) -> None:
    """Validate normalized HF ModelProfile architecture, geometry, and tokens."""
    base_mt = profile.model_type.strip().lower().removesuffix("_text")
    if not base_mt or base_mt not in SUPPORTED_MODEL_TYPES:
        repos = ", ".join(SUPPORTED_REPOSITORIES)
        _arch_err(
            f"Unsupported architecture model_type {profile.model_type!r}. "
            f"Supported repositories: {repos}. See {ONBOARDING_DOC_REF}."
        )
    for k, v in (
        ("num_hidden_layers", profile.num_layers),
        ("hidden_size", profile.hidden_size),
        ("num_attention_heads", profile.num_attention_heads),
    ):
        if v < 1:
            _arch_err(f"HuggingFace config missing causal-LM geometry key {k!r} >= 1")
    checks = (
        (
            profile.num_linear_layers > profile.num_layers,
            f"num_linear_layers ({profile.num_linear_layers}) > "
            f"num_hidden_layers ({profile.num_layers})",
        ),
        (
            profile.num_kv_heads < 0
            or profile.num_kv_heads > profile.num_attention_heads,
            f"Invalid num_kv_heads ({profile.num_kv_heads}) for "
            f"num_attention_heads ({profile.num_attention_heads})",
        ),
        (
            profile.head_dim < MIN_HEAD_DIM or profile.head_dim % MIN_HEAD_DIM != 0,
            f"Invalid head_dim ({profile.head_dim}): must be >= {MIN_HEAD_DIM} "
            f"and divisible by {MIN_HEAD_DIM}",
        ),
        (
            profile.num_linear_layers > 0
            and (profile.linear_key_heads < 1 or profile.linear_value_heads < 1),
            "Model with linear-attention layers missing GDN/KDA key/value head counts",
        ),
    )
    for bad, msg in checks:
        if bad:
            _arch_err(msg)
    if raw_layer_types is not None:
        bad_tokens = [
            str(e) for e in raw_layer_types if str(e) not in VALID_LAYER_TOKENS
        ]
        if bad_tokens:
            _arch_err(
                f"Unknown layer_type {bad_tokens[0]!r}; "
                f"supported: {sorted(VALID_LAYER_TOKENS)}"
            )


def parse_hf_config_profile(
    raw_config: Mapping[str, object], *, name: str = "model"
) -> ModelProfile:
    """Parse a HuggingFace config mapping without enforcing preset identity."""
    text_config = _extract_text_subconfig(raw_config)
    model_type = (
        str(text_config.get("model_type") or raw_config.get("model_type") or "")
        .strip()
        .lower()
    )
    linear_config = as_mapping(text_config.get("linear_attn_config")) or {}
    raw_layer_types = as_sequence(text_config.get("layer_types"))
    num_layers = _int_field(text_config, ("num_hidden_layers",))
    hidden_size = _int_field(text_config, ("hidden_size",))
    num_heads = _int_field(text_config, ("num_attention_heads",))
    num_lin, is_periodic = _resolve_linear_schedule(
        text_config, linear_config, num_layers, raw_layer_types
    )
    merged_linear_config: dict[str, object] = {**linear_config, **text_config}
    (
        num_experts,
        top_k,
        intermediate_size,
        moe_inter,
        shared_intermediate_size,
        moe_layers,
    ) = _extract_moe_dims(text_config, num_layers, hidden_size)
    attn_layers = max(0, num_layers - num_lin)
    quant = _extract_quant_fields(text_config, raw_config, num_layers, attn_layers)
    idx_topk, index_head_dim, idx_nheads, idx_layers = _resolve_dsa_fields(
        text_config, num_layers, attn_layers
    )
    w_dtype, exp_dtype, dtype_src = _infer_quant_dtypes_with_source(quant)
    profile = ModelProfile(
        name=name,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_kv_heads=_int_field(
            text_config, ("num_key_value_heads",), default=num_heads
        ),
        head_dim=_int_field(
            text_config,
            ("v_head_dim", "head_dim"),
            default=hidden_size // num_heads if num_heads else 0,
        ),
        intermediate_size=intermediate_size,
        num_linear_layers=num_lin,
        linear_key_heads=_int_field(
            merged_linear_config,
            ("linear_num_key_heads", "num_key_heads", "num_k_heads", "num_heads"),
        ),
        linear_value_heads=_int_field(
            merged_linear_config,
            ("linear_num_value_heads", "num_value_heads", "num_v_heads", "num_heads"),
        ),
        linear_key_dim=_int_field(
            merged_linear_config,
            ("linear_key_head_dim", "key_head_dim", "head_k_dim"),
            default=128,
        ),
        linear_value_dim=_int_field(
            merged_linear_config,
            ("linear_value_head_dim", "value_head_dim", "head_v_dim"),
            default=128,
        ),
        linear_conv_kernel=_int_field(
            merged_linear_config,
            ("linear_conv_kernel_dim", "short_conv_kernel_size"),
            default=4,
        ),
        num_experts=num_experts,
        top_k=top_k,
        moe_intermediate_size=moe_inter,
        shared_expert_intermediate_size=shared_intermediate_size,
        num_moe_layers=moe_layers,
        kv_lora_rank=_int_field(text_config, ("kv_lora_rank",)),
        q_lora_rank=_int_field(text_config, ("q_lora_rank",)),
        qk_nope_head_dim=_int_field(text_config, ("qk_nope_head_dim",)),
        qk_rope_head_dim=_int_field(text_config, ("qk_rope_head_dim",)),
        index_topk=idx_topk,
        index_head_dim=index_head_dim,
        index_n_heads=idx_nheads,
        num_indexer_layers=idx_layers,
        routed_expert_hidden_size=_int_field(
            text_config, ("routed_expert_hidden_size",)
        ),
        is_periodic=is_periodic,
        weight_dtype=w_dtype,
        expert_dtype=exp_dtype,
        act_dtype=str(quant.raw_activation_dtype or "").lower(),
        weight_block_k=quant.weight_block_k,
        # Apply TPU serving cache policy rather than HF mamba_ssm_dtype.
        state_dtype=_default_serving_state_dtype(model_type),
        model_type=model_type,
        has_attn_output_gate=bool(
            text_config.get("attn_output_gate", False)
            or text_config.get("mla_use_output_gate", False)
        ),
        dtype_source=dtype_src,
    )
    _validate_hf_profile(profile, raw_layer_types)
    return finalize_model_profile(profile)


def supports_hf_config(raw_config: Mapping[str, object]) -> None:
    """Validate that a HuggingFace config mapping matches a shipped preset."""
    _adopt_matching_preset_name(parse_hf_config_profile(raw_config, name="model"))


def _preset_to_model_profile(slug: str, entry: Mapping[str, object]) -> ModelProfile:
    """Construct and validate a ModelProfile from one `models.json` preset entry."""
    ints: dict[str, int] = {}
    for key in PRESET_REQUIRED_INT_KEYS:
        if key not in entry:
            _cfg_err(f"Preset {slug!r} missing required geometry key {key!r}")
        ints[key] = scalar_int(entry[key], f"{slug}.{key}")
    for key, default in PRESET_OPTIONAL_INT_DEFAULTS.items():
        ints[key] = scalar_int(entry.get(key, default), f"{slug}.{key}")
    model_type = str(entry.get("model_type", ""))
    default_state_dtype = _default_serving_state_dtype(model_type)
    profile = finalize_model_profile(
        ModelProfile(
            name=slug,
            **cast("_ProfileIntFields", ints),
            weight_dtype=str(entry.get("weight_dtype", "fp8")).lower(),
            expert_dtype=str(entry.get("expert_dtype", "")).lower(),
            act_dtype=str(entry.get("act_dtype", "")).lower(),
            kv_dtype=str(entry.get("kv_dtype", "fp8")).lower(),
            state_dtype=str(entry.get("state_dtype", default_state_dtype)).lower(),
            model_type=model_type,
            has_attn_output_gate=bool(entry.get("has_attn_output_gate", False)),
            kv_compress_factor=scalar_float(
                entry.get("kv_compress_factor", 1.0), f"{slug}.kv_compress_factor"
            ),
            layer_families=tuple(
                str(item) for item in (as_sequence(entry.get("layer_families")) or ())
            ),
            dtype_source=f"preset ({slug})",
        )
    )
    allowed = SUPPORTED_ONCHIP_FAMILIES | STRUCTURAL_LAYER_TAGS
    unmodeled = [fam for fam in profile.layer_families if fam not in allowed]
    if unmodeled:
        _arch_err(f"Preset {slug!r} contains unpriced layer families: {unmodeled}")
    return profile


def load_model_presets() -> tuple[
    Mapping[str, ModelProfile], Mapping[str, str], Mapping[str, str], Mapping[str, str]
]:
    """Load and validate shipped presets, aliases, sources, and dtype bases."""
    payload = json.loads((PRESETS_DIR / "models.json").read_text("utf-8"))
    presets: dict[str, ModelProfile] = {}
    sources: dict[str, str] = {}
    dtype_basis: dict[str, str] = {}
    for slug, entry in payload["presets"].items():
        basis_val = str(entry.get("dtype_basis", ""))
        if basis_val not in PRESET_DTYPE_BASES:
            valid = sorted(PRESET_DTYPE_BASES)
            _cfg_err(
                f"Preset {slug!r} dtype_basis must be one of {valid}, got {basis_val!r}"
            )
        dtype_basis[slug] = basis_val
        sources[slug] = str(entry.get("source", f"models.json:{slug}"))
        presets[slug] = _preset_to_model_profile(slug, entry)
    aliases = {str(k): str(v) for k, v in payload["aliases"].items()}
    return presets, aliases, sources, dtype_basis


def _repo_id_from_source(source: str) -> str:
    """Extract `org/repo` from a preset source string."""
    return next((item for item in source.split() if "/" in item), source)


(
    MODEL_PRESETS,
    REGISTERED_MODEL_ALIASES,
    MODEL_PRESET_SOURCES,
    MODEL_PRESET_DTYPE_BASIS,
) = load_model_presets()
SUPPORTED_MODEL_TYPES: frozenset[str] = frozenset(
    p.model_type.strip().lower().removesuffix("_text")
    for p in MODEL_PRESETS.values()
    if p.model_type
)
SUPPORTED_REPOSITORIES: tuple[str, ...] = tuple(
    _repo_id_from_source(MODEL_PRESET_SOURCES[slug]) for slug in MODEL_PRESETS
)
SUPPORTED_REPOSITORY_IDS: frozenset[str] = frozenset(
    repo.strip("/").lower() for repo in SUPPORTED_REPOSITORIES
)
SUPPORTED_OVERRIDE_DTYPES: frozenset[str] = frozenset(
    {"bf16"}
    | {preset.weight_dtype for preset in MODEL_PRESETS.values()}
    | {preset.expert_dtype for preset in MODEL_PRESETS.values() if preset.expert_dtype}
)


@dataclass(frozen=True)
class ParallelLayout:
    """One serving role's 6-axis mesh: parallel degrees, sequence sharding, groups."""

    tp: int
    ep: int
    dp: int = 1
    pp: int = 1
    dcp: int = 1
    pcp: int = 1
    spec_tokens: int = 0
    layer_groups: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Validate positive mesh degrees, non-negative spec tokens, and groups."""
        for name in ("tp", "ep", "dp", "pp", "dcp", "pcp"):
            _checked_int(getattr(self, name), f"ParallelLayout.{name}", lo=1)
        _checked_int(self.spec_tokens, "ParallelLayout.spec_tokens", lo=0)
        for idx, count in enumerate(self.layer_groups):
            _checked_int(count, f"ParallelLayout.layer_groups[{idx}]", lo=1)

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

    def submodule_label(self) -> str:
        """Return the attention mesh and the routed-expert mesh as one label."""
        attention = replace(self, ep=1).label()
        if self.ep <= 1:
            return f"Attn: {attention} \u00b7 MoE: TP{self.tp}"
        inner_tp = max(1, self.tp // gcd(self.ep, self.tp))
        moe = f"EP{self.ep}" if inner_tp == 1 else f"TP{inner_tp} x EP{self.ep}"
        return f"Attn: {attention} \u00b7 MoE: {moe}"


@dataclass(frozen=True)
class DeploymentPreset:
    """Shipped 6-axis prefill and decode reference layouts for one model."""

    decode: ParallelLayout
    prefill: ParallelLayout
    saturation: ParallelLayout | None = None
    prefill_windows: tuple[int, ...] = ()
    source: str = ""

    def named_layouts(self) -> tuple[tuple[str, ParallelLayout], ...]:
        """Return shipped role labels paired with their ParallelLayout."""
        layouts = [("decode", self.decode), ("prefill", self.prefill)]
        if self.saturation is not None:
            layouts.append(("saturation", self.saturation))
        return tuple(layouts)


def parse_parallel_layout(raw: Mapping[str, object]) -> ParallelLayout:
    """Parse a 6-axis `ParallelLayout` mapping from `deployments.json`."""
    axes = ("tp", "ep", "dp", "pp", "dcp", "pcp")
    return ParallelLayout(
        spec_tokens=scalar_int(raw.get("spec_tokens", 0), "spec_tokens"),
        layer_groups=int_tuple(raw.get("layer_groups"), "layer_groups"),
        **{k: scalar_int(raw.get(k, 1), k) for k in axes},
    )


def load_deployment_presets() -> Mapping[str, DeploymentPreset]:
    """Load deployment layout presets from JSON."""
    raw_deps = json.loads((PRESETS_DIR / "deployments.json").read_text("utf-8"))[
        "deployments"
    ]
    deployments: dict[str, DeploymentPreset] = {}
    for slug, entry in raw_deps.items():
        src = str(entry.get("source", f"deployments.json:{slug}"))
        sat_map = as_mapping(entry.get("saturation"))
        deployments[slug] = DeploymentPreset(
            decode=parse_parallel_layout(entry["decode"]),
            prefill=parse_parallel_layout(entry["prefill"]),
            saturation=parse_parallel_layout(sat_map) if sat_map is not None else None,
            prefill_windows=int_tuple(entry.get("prefill_windows"), "prefill_windows"),
            source=src,
        )
    return deployments


DEPLOYMENT_PRESETS = load_deployment_presets()


def _identity_view(profile: ModelProfile) -> ModelProfile:
    """Return `profile` with non-geometric fields blanked for preset matching."""
    return replace(
        finalize_model_profile(profile),
        name="",
        weight_dtype="",
        expert_dtype="",
        act_dtype="",
        kv_dtype="",
        state_dtype="",
        weight_block_k=0,
        dtype_source="",
        layer_families=(),
        model_type=profile.model_type.strip().lower().removesuffix("_text"),
    )


def match_preset_by_geometry(profile: ModelProfile) -> str | None:
    """Return the shipped preset matching `profile` geometry and config dtypes."""
    finalized = finalize_model_profile(profile)
    wanted = _identity_view(finalized)
    for preset_name, preset in MODEL_PRESETS.items():
        if _identity_view(preset) != wanted:
            continue
        if MODEL_PRESET_DTYPE_BASIS[preset_name] == "config" and (
            preset.weight_dtype.lower() != finalized.weight_dtype.lower()
            or preset.expert_dtype.lower() != finalized.expert_dtype.lower()
        ):
            continue
        return preset_name
    return None


def _adopt_matching_preset_name(profile: ModelProfile) -> ModelProfile:
    """Rename a config-derived profile to its matching shipped preset or fail closed."""
    if (preset_name := match_preset_by_geometry(profile)) is not None:
        return replace(profile, name=preset_name)
    repos = ", ".join(SUPPORTED_REPOSITORIES)
    _arch_err(
        f"Config {profile.name!r} (model_type={profile.model_type!r}) matches no "
        f"validated preset by geometry and config dtypes. Supported repositories: "
        f"{repos}. See {ONBOARDING_DOC_REF}."
    )


def _apply_preset_dtype_overrides(
    model_profile: ModelProfile, overrides: DtypeOverrides
) -> ModelProfile:
    """Return `model_profile` updated with any explicit dtype override flags."""
    pairs: tuple[tuple[str, str | None], ...] = (
        ("weight_dtype", overrides.get("weight_dtype")),
        ("expert_dtype", overrides.get("expert_dtype")),
        ("kv_dtype", overrides.get("kv_dtype")),
        ("state_dtype", overrides.get("state_dtype")),
        ("act_dtype", overrides.get("act_dtype")),
    )
    normalized_overrides = {k: v.strip().lower() for k, v in pairs if v is not None}
    for field_name in ("weight_dtype", "expert_dtype"):
        if (
            dtype_val := normalized_overrides.get(field_name)
        ) is not None and dtype_val not in SUPPORTED_OVERRIDE_DTYPES:
            _arch_err(
                f"Unsupported {field_name}={dtype_val!r}; "
                f"supported: {sorted(SUPPORTED_OVERRIDE_DTYPES)}"
            )
    normalized_overrides = {
        k: _require_known_dtype(v, k) for k, v in normalized_overrides.items()
    }
    weight_override = normalized_overrides.get("weight_dtype")
    resolved_weight = (weight_override or model_profile.weight_dtype).lower()
    distinct_exp = bool(
        model_profile.expert_dtype
        and model_profile.expert_dtype.lower() != model_profile.weight_dtype.lower()
    )
    resolved_expert = normalized_overrides.get(
        "expert_dtype",
        model_profile.expert_dtype.lower()
        if distinct_exp
        else (weight_override or model_profile.expert_dtype.lower()),
    )
    if "act_dtype" in normalized_overrides:
        resolved_act = normalized_overrides["act_dtype"]
    elif weight_override is not None:
        resolved_act = (
            "fp8"
            if resolved_weight in QUANT_SUB_8BIT_WEIGHT_DTYPES
            else resolved_weight
        )
    else:
        resolved_act = model_profile.act_dtype
    updated = replace(
        model_profile,
        weight_dtype=resolved_weight,
        expert_dtype=resolved_expert,
        act_dtype=resolved_act,
        kv_dtype=normalized_overrides.get("kv_dtype", model_profile.kv_dtype).lower(),
        state_dtype=normalized_overrides.get(
            "state_dtype", model_profile.state_dtype
        ).lower(),
        dtype_source=(
            "flag"
            if any(
                k in normalized_overrides
                for k in ("weight_dtype", "expert_dtype", "act_dtype")
            )
            else model_profile.dtype_source
        ),
    )
    _validate_dtype_fields(updated)
    return updated


def model_profile_from_hf_config(
    raw_config: Mapping[str, object],
    name: str = "model",
    **overrides: Unpack[DtypeOverrides],
) -> ModelProfile:
    """Parse and validate a HuggingFace config mapping into a shipped preset profile."""
    matched = _adopt_matching_preset_name(
        parse_hf_config_profile(raw_config, name=name)
    )
    if any(val is not None for val in overrides.values()):
        return _apply_preset_dtype_overrides(matched, overrides)
    return matched


def resolve_hf_snapshot_config(
    repo_id: str, hf_home: Path | None = None
) -> Path | None:
    """Resolve newest snapshot config.json for a HuggingFace repository."""
    raw_target = repo_id.strip()
    if "/" not in raw_target or raw_target.endswith(".json"):
        return None
    org_repo = raw_target.strip("/").replace("/", "--")
    home = hf_home or (Path.home() / ".cache" / "huggingface")
    for root in (home / "hub", home):
        repo_dir = root / f"models--{org_repo}"
        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.is_dir():
            continue
        ref_main = repo_dir / "refs" / "main"
        if ref_main.is_file():
            cand = snapshots_dir / ref_main.read_text("utf-8").strip() / "config.json"
            if cand.is_file():
                return cand
        snaps = sorted(
            snapshots_dir.glob("*/config.json"),
            key=lambda p: (p.stat().st_mtime_ns, p.parent.name),
        )
        if snaps:
            return snaps[-1]
    return None


def _load_json_mapping(config_path: Path) -> Mapping[str, object]:
    """Read and decode a JSON object file, raising `MegakernelConfigError` on error."""
    try:
        parsed: object = json.loads(config_path.read_text("utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON in config file {config_path}: {exc}"
        raise errors.MegakernelConfigError(msg) from exc
    mapping = as_mapping(parsed)
    if mapping is None:
        msg = f"Config file {config_path} must contain a JSON object"
        raise errors.MegakernelConfigError(msg)
    return mapping


def resolve_model_profile(
    target: object, **overrides: Unpack[DtypeOverrides]
) -> ModelProfile:
    """Resolve a ModelProfile from a preset, repo alias, dict, or config path."""
    has_overrides = any(val is not None for val in overrides.values())
    if type(target) is ModelProfile:
        return (
            _apply_preset_dtype_overrides(target, overrides)
            if has_overrides
            else target
        )
    if isinstance(target, Mapping):
        return model_profile_from_hf_config(target, name="model", **overrides)
    raw = str(target).strip()
    path = Path(raw)
    if path.is_file():
        return model_profile_from_hf_config(
            _load_json_mapping(path),
            name=path.parent.name or path.stem,
            **overrides,
        )
    cfg_file = path / "config.json"
    if not raw.endswith(".json") and cfg_file.is_file():
        return model_profile_from_hf_config(
            _load_json_mapping(cfg_file),
            name=path.name or cfg_file.stem,
            **overrides,
        )
    tail = raw.lower().strip("/").split("/")[-1]
    alias_allowed = "/" not in raw or raw.strip("/").lower() in SUPPORTED_REPOSITORY_IDS
    if (
        alias_allowed
        and (preset_key := REGISTERED_MODEL_ALIASES.get(tail)) is not None
        and not raw.endswith(".json")
    ):
        return _apply_preset_dtype_overrides(MODEL_PRESETS[preset_key], overrides)
    if (snap := resolve_hf_snapshot_config(raw)) is not None:
        return model_profile_from_hf_config(
            _load_json_mapping(snap), name=tail, **overrides
        )
    repos = ", ".join(SUPPORTED_REPOSITORIES)
    _arch_err(
        f"Unregistered model or missing config path: {target!r}. Registered presets: "
        f"{sorted(MODEL_PRESETS)}; supported repositories: {repos}. See "
        f"{ONBOARDING_DOC_REF}."
    )
