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
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from math import ceil, gcd
from pathlib import Path
from typing import Literal, cast

from ..core import errors
from .constants import (
    BF16_BYTES,
    BITS_FP4,
    BITS_FP8,
    DECLARED_COLLECTIVE_KINDS,
    DECLARED_EXPERT_MODES,
    DEFAULT_DECLARED_CALLS_PER_LAYER,
    DEFAULT_DECLARED_PHASES,
    ENGRAM_SCALE_BLOCK_SIZE,
    FP32_BYTES,
    HC_SUBLAYERS_PER_LAYER,
    INDEXER_SCALE_BLOCK_SIZE,
    KDA_MODEL_TYPES,
    KV_DTYPE_ALIASES,
    MHC_DYNAMIC_PROJ_TERMS,
    MIN_COLLECTIVE_GROUP_SIZE,
    MIN_HEAD_DIM,
    MIN_INTERVAL,
    MULTIMODAL_KEYS,
    MXFP4_BLOCK_K,
    PLE_PADDED_ROWS_PER_SHARD,
    PLE_SHARD_DIVISOR,
    PRESETS_DIR,
    STRUCTURAL_LAYER_TAGS,
    SUPPORTED_ONCHIP_FAMILIES,
    VALID_LAYER_TOKENS,
    WORD_ALIGN_BYTES,
    CollectiveStepKind,
    DcpSpan,
    DeclaredExpertMode,
    HostTableShardAxis,
    HyperConnectionKind,
    SparseAttnKind,
    SparseCacheReplication,
)


class UnsupportedArchitectureError(errors.MegakernelConfigError, ValueError):
    """Raised when a model preset or HuggingFace config is unsupported or malformed."""


ONBOARDING_DOC_REF = "src/vllm_torchtpu/megakernel/advisor/README.md (Adding a Model)"


@dataclass(frozen=True)
class SparseCacheOwner:
    """Shared or per-layer sparse KV or indexer cache owner geometry."""

    layer: int
    record_tokens: float = 1.0
    record_elems: int = 0
    elem_bytes: float | None = None
    record_overhead_bytes: float = 0.0
    replication: SparseCacheReplication = "per_rank"
    users: tuple[int, ...] = ()
    max_scan_records: int | None = None


@dataclass(frozen=True)
class SparseAttnSpec:
    """Unified sparse-attention specification for QSA, CSA2+SWA, and MSA."""

    kind: SparseAttnKind
    window_tokens: int = 0
    window_bytes_per_token: float = 0.0
    window_layers: tuple[int, ...] = ()
    select_tokens: int = 0
    index_heads: int = 0
    index_kv_heads: int = 1
    index_dim: int = 0
    index_compress_ratio: int = 1
    kv_owners: tuple[SparseCacheOwner, ...] = ()
    index_owners: tuple[SparseCacheOwner, ...] = ()
    shared_kv_mqa: bool = False
    kv_owner_ring_bytes_per_req: int = 0


@dataclass(frozen=True)
class HyperConnectionSpec:
    """Multi-stream residual hyper-connection (`lowrank` or Sinkhorn `mhc`)."""

    kind: HyperConnectionKind
    num_streams: int = 4
    lowrank_dim: int = 0
    sinkhorn_iters: int = 0
    has_final_mixer: bool = False


@dataclass(frozen=True)
class HostTableBreakSpec:
    """Large host-offloaded or HBM-resident embedding lookup table specification."""

    name: str
    layers: tuple[int, ...]
    table_bytes_total: int
    h2d_bytes_per_token: float
    hbm_resident: bool = False
    shard_axis: HostTableShardAxis = "tp"
    state_bytes_per_request: int = 0
    dense_proj_params: int = 0


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
    non_homogeneous_reasons: tuple[str, ...] = ()
    kv_compress_factor: float = 1.0
    residual_streams: int = 1
    dtype_source: str = ""
    o_lora_rank: int = 0
    o_groups: int = 1
    ced_boundary: int = 0
    dcp_span: DcpSpan = "kv_heads"
    expert_act_dtype: str = ""
    sparse_attn: SparseAttnSpec | None = None
    hyper_connection: HyperConnectionSpec | None = None
    host_tables: tuple[HostTableBreakSpec, ...] = ()


def hyper_connection_total_params(profile: ModelProfile) -> int:
    """Return unsharded total hyper-connection parameter count across all layers."""
    hc = profile.hyper_connection
    if hc is None or hc.num_streams <= 1:
        return 0
    streams = hc.num_streams
    hidden = profile.hidden_size
    if hc.kind == "lowrank":
        rank = hc.lowrank_dim
        per_sublayer = 2 * streams * hidden * rank + streams * streams * hidden
        total = profile.num_layers * HC_SUBLAYERS_PER_LAYER * per_sublayer
        if hc.has_final_mixer:
            total += 2 * streams * hidden * rank
        return total
    per_sublayer = (streams * streams + MHC_DYNAMIC_PROJ_TERMS * streams) * (
        streams * hidden
    )
    return profile.num_layers * HC_SUBLAYERS_PER_LAYER * per_sublayer


def _non_homogeneous_reasons(num_layers: int, moe_layers: int) -> tuple[str, ...]:
    """Return non-homogeneity tags for mixed dense-prefix and MoE stacks."""
    if 0 < moe_layers < num_layers:
        return ("dense_prefix_moe",)
    return ()


def has_dsa_profile(profile: ModelProfile) -> bool:
    """Return whether `profile` carries legacy DSA sparse attention geometry."""
    if profile.sparse_attn is not None:
        return False
    return (
        profile.index_topk > 0
        and profile.index_n_heads > 0
        and profile.index_head_dim > 0
    )


def dsa_query_gather_dim(profile: ModelProfile) -> int:
    """Return per-head query width gathered across DCP ranks for DSA attention."""
    if profile.kv_lora_rank > 0:
        return profile.kv_lora_rank + profile.qk_rope_head_dim
    return profile.head_dim


def indexer_record_bytes(index_head_dim: int) -> int:
    """Return packed 4-byte-aligned FP8 key plus block-scale indexer record bytes."""
    if index_head_dim <= 0:
        return 0
    scale_bytes = ceil(index_head_dim / INDEXER_SCALE_BLOCK_SIZE)
    raw_bytes = index_head_dim + scale_bytes
    return ((raw_bytes + WORD_ALIGN_BYTES - 1) // WORD_ALIGN_BYTES) * WORD_ALIGN_BYTES


def _derive_layer_families(
    profile: ModelProfile, *, attn_layers: int, moe_layers: int
) -> tuple[str, ...]:
    """Derive canonical layer families from resolved model layer counts."""
    families: list[str] = []
    if attn_layers > 0:
        if profile.sparse_attn is not None:
            families.append(profile.sparse_attn.kind)
        elif has_dsa_profile(profile):
            families.append("dsa")
        else:
            families.append("mla" if profile.kv_lora_rank > 0 else "gqa")
    if profile.num_linear_layers > 0:
        mtype = profile.model_type.strip().lower()
        families.append("kda" if mtype in KDA_MODEL_TYPES else "gdn")
    if moe_layers > 0:
        families.append("moe")
    if moe_layers < profile.num_layers and profile.intermediate_size > 0:
        families.append("dense_mlp")
    if profile.hyper_connection is not None:
        families.append(
            "mhc" if profile.hyper_connection.kind == "mhc" else "hyper_connection"
        )
    elif profile.residual_streams > 1:
        families.append("hyper_connection")
    if 0 < moe_layers < profile.num_layers:
        families.append("dense_prefix_moe")
    if any(not tbl.hbm_resident for tbl in profile.host_tables) or (
        profile.ced_boundary > 0
    ):
        families.append("fusion_break")
    return tuple(families)


def finalize_model_profile(profile: ModelProfile) -> ModelProfile:
    """Return an immutable ModelProfile with derived layer metadata populated."""
    if profile.sparse_attn is not None:
        profile = replace(
            profile,
            index_topk=0,
            index_head_dim=0,
            index_n_heads=0,
            num_indexer_layers=0,
            kv_compress_factor=1.0,
        )
    kv_low = profile.kv_dtype.strip().lower()
    profile = replace(
        profile,
        residual_streams=(
            profile.hyper_connection.num_streams
            if profile.hyper_connection is not None
            else profile.residual_streams
        ),
        kv_dtype=KV_DTYPE_ALIASES.get(kv_low, kv_low),
    )
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
    if not has_dsa_profile(profile):
        resolved_indexer_layers = 0
    elif profile.num_indexer_layers <= 0:
        resolved_indexer_layers = attn_layers
    else:
        resolved_indexer_layers = min(attn_layers, profile.num_indexer_layers)
    periodic = (
        profile.is_periodic
        and not (0 < resolved_moe_layers < num_layers)
        and not (0 < resolved_indexer_layers < attn_layers)
    )
    moe_inter = profile.moe_intermediate_size or (
        profile.intermediate_size * has_experts
    )
    exp_dt = profile.expert_dtype or profile.weight_dtype
    w_dt_low = profile.weight_dtype.lower()
    default_act_dt = (
        "fp8" if w_dt_low in ("fp8", "nvfp4", "mxfp4", "fp4", "int8") else w_dt_low
    )
    act_dt = (profile.act_dtype or default_act_dt).lower()
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
        num_indexer_layers=resolved_indexer_layers,
        is_periodic=periodic,
        moe_intermediate_size=moe_inter,
        expert_dtype=exp_dt,
        act_dtype=act_dt,
        expert_act_dtype=profile.expert_act_dtype.strip().lower(),
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
    index_topk: int = 0
    index_head_dim: int = 0
    index_n_heads: int = 0
    num_indexer_layers: int = 0
    dense_unquantized: bool = False
    raw_state_dtype: str = "bf16"
    act_scheme: str = ""
    o_lora_rank: int = 0
    o_groups: int = 1
    ced_boundary: int = 0
    dcp_span: DcpSpan = "kv_heads"
    sparse_attn: SparseAttnSpec | None = None
    hyper_connection: HyperConnectionSpec | None = None
    host_tables: tuple[HostTableBreakSpec, ...] = ()

    def validate_supported(self, *, require_model_type: bool = False) -> None:
        """Validate that the configuration describes a supported causal LM."""
        _ = require_model_type
        base_mtype = self.model_type.strip().lower().removesuffix("_text")
        if not base_mtype or base_mtype not in SUPPORTED_MODEL_TYPES:
            msg = (
                f"Unsupported architecture model_type {self.model_type!r}. "
                f"Supported repositories: {', '.join(SUPPORTED_REPOSITORIES)}. "
                f"See {ONBOARDING_DOC_REF}."
            )
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
        # A checkpoint that skips attention keeps dense weights unquantized.
        unquantized_dt = "fp32" if "32" in self.torch_dtype.lower() else "bf16"
        dense_default = unquantized_dt if self.dense_unquantized else inferred_dt
        resolved_wd = str(
            weight_dtype or self.raw_weight_dtype or dense_default
        ).lower()
        expert_default = inferred_dt if self.dense_unquantized else resolved_wd
        resolved_ed = str(
            expert_dtype or self.raw_expert_dtype or expert_default
        ).lower()
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
        elif self.dense_unquantized:
            dtype_source = "quantization_config (experts); default (unquantized dense)"
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
        state_dtype: str | None = None,
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
        resolved_state_dt = str(state_dtype or self.raw_state_dtype or "bf16").lower()
        resolved_exp_act_dt = (
            "fp8"
            if (
                act_dtype is None
                and self.dense_unquantized
                and resolved_wd in ("bf16", "fp16", "fp32")
                and resolved_ed == "fp8"
                and self.act_scheme == "dynamic"
            )
            else ""
        )

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
                index_topk=self.index_topk,
                index_head_dim=self.index_head_dim,
                index_n_heads=self.index_n_heads,
                num_indexer_layers=self.num_indexer_layers,
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
                state_dtype=resolved_state_dt,
                model_type=self.model_type,
                has_attn_output_gate=self.has_attn_output_gate,
                kv_compress_factor=self.kv_compress_factor,
                residual_streams=self.residual_streams,
                dtype_source=dtype_source,
                o_lora_rank=self.o_lora_rank,
                o_groups=self.o_groups,
                ced_boundary=self.ced_boundary,
                dcp_span=self.dcp_span,
                expert_act_dtype=resolved_exp_act_dt,
                sparse_attn=self.sparse_attn,
                hyper_connection=self.hyper_connection,
                host_tables=self.host_tables,
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
    if "text_config" in raw_map and _as_mapping(raw_map.get("text_config")) is None:
        msg = "Multimodal config has non-mapping 'text_config'"
        raise UnsupportedArchitectureError(msg)
    sub_cfg = _as_mapping(raw_map.get("text_config"))
    tc: Mapping[str, object] = sub_cfg if sub_cfg is not None else raw_map
    is_multimodal = "text_config" in raw_map or any(
        k in raw_map for k in MULTIMODAL_KEYS
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
            any(k in str(t).lower() for k in ("lin", "gdn", "kda"))
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


ATTENTION_MODULE_PREFIXES: tuple[str, ...] = (
    "model.layers",
    "model.language_model.layers",
    "language_model.model.layers",
)
ATTENTION_PROJECTION_NAMES: tuple[str, ...] = (
    "q_proj",
    "q_a_proj",
)


def _compile_skip_pattern(pattern: str) -> Callable[[str], re.Match[str] | None]:
    """Compile one quantization skip pattern into a bound matcher.

    A `re:` prefix marks a regular expression matched from the start of the
    name. Any other pattern is a dotted module path, matched on whole path
    components, where `*` spans any run of characters.
    """
    if pattern.startswith("re:"):
        try:
            return re.compile(pattern.removeprefix("re:")).match
        except re.error as exc:
            msg = f"Invalid quantization skip pattern {pattern!r}: {exc}"
            raise UnsupportedArchitectureError(msg) from exc
    glob_body = ".*".join(re.escape(part) for part in pattern.split("*"))
    return re.compile(rf"(?:^|\.){glob_body}(?:\.|$)").search


def _quant_skips_attention(
    quant_map: Mapping[str, object],
    num_layers: int,
    full_attn_layers: int,
) -> bool:
    """Return whether quantization skips query projections on full-attention layers.

    Reads the compressed-tensors `ignore` list and the `modules_to_not_convert`
    list across layer indices `0 .. num_layers - 1`, path prefixes, and query
    projection names (`q_proj`, `q_a_proj`). Dense weights keep the checkpoint's
    unquantized dtype only when covered layers reach a positive
    `full_attn_layers` count.
    """
    patterns: list[str] = []
    for key in ("ignore", "modules_to_not_convert"):
        entries = _as_sequence(quant_map.get(key))
        if entries is not None:
            patterns.extend(str(entry) for entry in entries)
    if not patterns:
        return False
    matchers = tuple(_compile_skip_pattern(pattern) for pattern in patterns)
    if full_attn_layers <= 0:
        return False
    covered_layers = 0
    for layer_index in range(num_layers):
        layer_probes = (
            f"{prefix}.{layer_index}.self_attn.{proj}"
            for prefix in ATTENTION_MODULE_PREFIXES
            for proj in ATTENTION_PROJECTION_NAMES
        )
        if any(
            matcher(probe) is not None for probe in layer_probes for matcher in matchers
        ):
            covered_layers += 1
            if covered_layers >= full_attn_layers:
                return True
    return False


def _extract_quant_fields(
    tc: Mapping[str, object],
    raw_cfg: Mapping[str, object],
    num_layers: int,
    full_attn_layers: int,
) -> tuple[str, str, int, str, str, str, str, int, bool, bool, bool, str]:
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
    q_method = str(quant_map.get("quant_method") or "").strip().lower()
    quant_tag = (
        f"{q_method} "
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
    moe_q_algo = str(quant_map.get("moe_quant_algo") or "").strip().lower()
    if not raw_ed and moe_q_algo == "nvfp4":
        raw_ed = "nvfp4"
    act_scheme = str(quant_map.get("activation_scheme") or "").strip().lower()
    raw_act_dt = str(
        quant_map.get("act_dtype")
        or tc.get("act_dtype")
        or raw_cfg.get("act_dtype")
        or ""
    ).lower()
    if not raw_act_dt and (
        act_scheme == "none"
        or any(k in quant_tag for k in ("w8a16", "w4a16", "weight_only", "weight-only"))
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
    scale_fmt = str(quant_map.get("scale_fmt") or "").strip().lower()
    if raw_ed == "fp4" and scale_fmt == "ue8m0" and weight_block_k == MXFP4_BLOCK_K:
        raw_ed = "mxfp4"
    if q_method == "mxfp8":
        if not raw_wd:
            raw_wd = "fp8"
        weight_block_k = 0
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
        _quant_skips_attention(quant_map, num_layers, full_attn_layers),
        act_scheme,
    )


VALID_INDEXER_SCHEDULE_TOKENS: frozenset[str] = frozenset({"full", "shared", "none"})


def _classify_indexer_schedule_entry(entry: object, layer_index: int) -> bool:
    """Return whether one schedule token marks a full indexer layer, or raise."""
    match entry:
        case str() as raw_token:
            normalized = raw_token.strip().lower()
            if normalized not in VALID_INDEXER_SCHEDULE_TOKENS:
                valid_tokens = sorted(VALID_INDEXER_SCHEDULE_TOKENS)
                msg = (
                    f"Invalid indexer schedule entry at layer {layer_index}: "
                    f"{entry!r}; expected one of {valid_tokens}"
                )
                raise UnsupportedArchitectureError(msg)
            return normalized == "full"
        case _:
            num = _as_scalar_number(entry)
            if num is None or num < 0.0:
                msg = (
                    f"Invalid numeric indexer schedule entry at layer {layer_index}: "
                    f"{entry!r}; expected >= 0"
                )
                raise UnsupportedArchitectureError(msg)
            return num > 0.0


def _resolve_indexer_layers(
    text_cfg: Mapping[str, object],
    num_layers: int,
    attn_layers: int,
    index_dims: tuple[int, int, int],
) -> int:
    """Resolve the count of full sparse-indexer layers from HF config keys."""
    index_topk, index_head_dim, index_n_heads = index_dims
    if index_topk <= 0 or index_head_dim <= 0 or index_n_heads <= 0:
        return 0
    explicit_count = _int_field(text_cfg, ("num_indexer_layers",))
    if explicit_count > 0:
        return min(attn_layers, explicit_count)
    raw_schedule_obj = (
        text_cfg.get("indexer_types")
        if text_cfg.get("indexer_types") is not None
        else text_cfg.get("index_topk_pattern")
    )
    if raw_schedule_obj is not None:
        raw_schedule = _as_sequence(raw_schedule_obj)
        if raw_schedule is None or len(raw_schedule) != num_layers:
            msg = (
                "Indexer schedule length must equal num_hidden_layers "
                f"({num_layers}); got {raw_schedule_obj!r}"
            )
            raise UnsupportedArchitectureError(msg)
        full_count = sum(
            1
            for layer_index, entry in enumerate(raw_schedule)
            if _classify_indexer_schedule_entry(entry, layer_index)
        )
        if full_count <= 0:
            msg = "Indexer schedule must include at least one full indexer layer"
            raise UnsupportedArchitectureError(msg)
        return min(attn_layers, full_count)
    topk_freq = _int_field(text_cfg, ("index_topk_freq",), default=1)
    skip_offset = _int_field(text_cfg, ("index_skip_topk_offset",), default=0)
    if topk_freq > 1:
        scheduled = sum(
            1
            for layer_index in range(num_layers)
            if layer_index >= skip_offset
            and (layer_index - skip_offset + 1) % topk_freq == 0
        )
        if scheduled <= 0:
            msg = (
                f"index_topk_freq={topk_freq} with index_skip_topk_offset="
                f"{skip_offset} selects zero indexer layers across {num_layers} layers"
            )
            raise UnsupportedArchitectureError(msg)
        return min(attn_layers, scheduled)
    return attn_layers


def _resolve_dsa_fields(
    text_cfg: Mapping[str, object], num_layers: int, attn_layers: int
) -> tuple[int, int, int, int]:
    """Resolve `(index_topk, index_head_dim, index_n_heads, num_indexer_layers)`.

    Keys are read from the text config and from its nested `dsa_config`. A
    `dsa_layers` list there gives the count of full indexer layers.
    """
    dsa_map = _as_mapping(text_cfg.get("dsa_config")) or {}
    index_cfg: dict[str, object] = {**dsa_map, **text_cfg}
    dsa_layers = _as_sequence(dsa_map.get("dsa_layers"))
    if dsa_layers and not index_cfg.get("num_indexer_layers"):
        index_cfg["num_indexer_layers"] = len(dsa_layers)
    index_topk = _int_field(index_cfg, ("index_topk",))
    index_head_dim = _int_field(index_cfg, ("index_head_dim",))
    index_n_heads = _int_field(index_cfg, ("index_n_heads", "index_num_heads"))
    num_indexer_layers = _resolve_indexer_layers(
        index_cfg,
        num_layers,
        attn_layers,
        (index_topk, index_head_dim, index_n_heads),
    )
    return index_topk, index_head_dim, index_n_heads, num_indexer_layers


def _int_seq(raw: object) -> list[int]:
    """Extract a list of ints from an optional JSON/HF sequence."""
    return [int(_as_scalar_number(x) or 0) for x in (_as_sequence(raw) or ())]


def _build_csa_owners(
    sources: Sequence[int],
    ratios: Sequence[int],
    num_layers: int,
    elems_and_cand: tuple[int, int, int],
) -> tuple[SparseCacheOwner, ...]:
    """Construct CSA2 KV or index owner tuples across consumer layer ranges."""
    record_elems, cand_src, cand_cap = elems_and_cand
    sorted_srcs = sorted(s for s in sources if 0 <= s < num_layers)
    owners: list[SparseCacheOwner] = []
    for pos, src in enumerate(sorted_srcs):
        stop = sorted_srcs[pos + 1] if pos + 1 < len(sorted_srcs) else num_layers
        users = tuple(
            lyr
            for lyr in range(src, stop)
            if not ratios or (lyr < len(ratios) and ratios[lyr] > 0)
        )
        ratio = float(ratios[src]) if src < len(ratios) and ratios[src] > 0 else 1.0
        scan_cap = cand_cap if (cand_cap > 0 and src > cand_src) else None
        owners.append(
            SparseCacheOwner(
                src,
                ratio,
                record_elems,
                1.0,
                float(WORD_ALIGN_BYTES),
                "per_rank",
                users,
                scan_cap,
            )
        )
    return tuple(owners)


def _parse_hf_sparse_attn(
    tc: Mapping[str, object],
    attn_geom: tuple[int, int, int, int],
) -> tuple[SparseAttnSpec | None, DcpSpan, int]:
    """Parse QSA, CSA2+SWA, or MSA sparse attention geometry from HF config."""
    num_layers, num_kv_heads, head_dim, o_lora_rank = attn_geom
    idx_budget = _int_field(tc, ("indexer_budget",))
    if idx_budget > 0:
        ratio = _int_field(tc, ("indexer_compress_ratio",), default=4)
        return (
            SparseAttnSpec(
                kind="qsa",
                window_tokens=0,
                select_tokens=idx_budget + ratio - 1,
                index_heads=_int_field(tc, ("indexer_n_heads",), default=4),
                index_kv_heads=_int_field(tc, ("indexer_kv_heads",), default=1),
                index_dim=_int_field(tc, ("indexer_head_dim",), default=128),
                index_compress_ratio=ratio,
            ),
            "kv_heads",
            0,
        )
    kv_srcs = _int_seq(tc.get("kv_source_layer_ids"))
    if kv_srcs:
        idx_srcs = _int_seq(tc.get("index_source_layer_ids"))
        ratios = _int_seq(tc.get("compress_ratios"))[:num_layers]
        shared_mqa = num_kv_heads == 1 and o_lora_rank > 0
        kv_elems = (1 if shared_mqa else 2) * max(1, num_kv_heads) * head_dim
        idx_dim = _int_field(tc, ("index_head_dim",), default=128)
        cand_src = _int_field(tc, ("candidate_source_layer_id",))
        cand_cap = _int_field(tc, ("candidate_topk_blocks",)) * _int_field(
            tc, ("candidate_block_size",)
        )
        kv_owners = _build_csa_owners(kv_srcs, ratios, num_layers, (kv_elems, 0, 0))
        idx_owners = _build_csa_owners(
            idx_srcs, ratios, num_layers, (idx_dim, cand_src, cand_cap)
        )
        swa_win = _int_field(tc, ("sliding_window",))
        win_b = float(kv_elems * BF16_BYTES) if swa_win > 0 else 0.0
        ring_b = (
            sum(1 for o in kv_owners if o.record_tokens > 1.0)
            * 2
            * head_dim
            * FP32_BYTES
        )
        ced_lyr = cand_src if (num_layers > 0 and cand_src == num_layers // 2) else 0
        return (
            SparseAttnSpec(
                kind="csa",
                window_tokens=swa_win,
                window_bytes_per_token=win_b,
                window_layers=tuple(range(num_layers)) if swa_win > 0 else (),
                select_tokens=_int_field(tc, ("index_topk",)),
                index_heads=_int_field(tc, ("index_n_heads",)),
                index_dim=idx_dim,
                kv_owners=kv_owners,
                index_owners=idx_owners,
                shared_kv_mqa=shared_mqa,
                kv_owner_ring_bytes_per_req=ring_b,
            ),
            "kv_heads",
            ced_lyr,
        )
    sp_map = _as_mapping(tc.get("sparse_attention_config"))
    if sp_map is not None and bool(sp_map.get("use_sparse_attention", False)):
        blk_sz = _int_field(sp_map, ("sparse_block_size",), default=128)
        return (
            SparseAttnSpec(
                kind="msa",
                window_tokens=_int_field(sp_map, ("sparse_local_block",), default=1)
                * blk_sz,
                select_tokens=_int_field(sp_map, ("sparse_topk_blocks",), default=16)
                * blk_sz,
                index_heads=_int_field(sp_map, ("sparse_num_index_heads",), default=4),
                index_dim=_int_field(sp_map, ("sparse_index_dim",), default=128),
            ),
            "tp_group",
            0,
        )
    return None, "kv_heads", 0


def _parse_hf_hyper_connection(tc: Mapping[str, object]) -> HyperConnectionSpec | None:
    """Parse low-rank HyperConnection or Sinkhorn mHC config from HF mapping."""
    hc_count = _int_field(tc, ("hc_count",))
    hc_lowrank = _int_field(tc, ("hc_lowrank",))
    if hc_count > 1 and hc_lowrank > 0:
        return HyperConnectionSpec(
            kind="lowrank",
            num_streams=hc_count,
            lowrank_dim=hc_lowrank,
            has_final_mixer=True,
        )
    hc_mult = _int_field(tc, ("hc_mult",))
    sinkhorn_iters = _int_field(tc, ("hc_sinkhorn_iters",))
    if hc_mult > 1 and sinkhorn_iters > 0:
        return HyperConnectionSpec(
            kind="mhc",
            num_streams=hc_mult,
            sinkhorn_iters=sinkhorn_iters,
        )
    return None


def _parse_hf_host_tables(
    tc: Mapping[str, object], num_layers: int, hidden_size: int
) -> tuple[HostTableBreakSpec, ...]:
    """Parse PLE and Engram embedding lookup tables from HF text_config."""
    tables: list[HostTableBreakSpec] = []
    ple_ids = _int_seq(tc.get("ple_layer_ids"))
    if ple_ids:
        ple_layers = tuple(sorted({x - 1 for x in ple_ids if 1 <= x <= num_layers}))
        ple_dim = _int_field(tc, ("ple_embed_dim",), default=hidden_size)
        shards = _int_field(tc, ("split_ngram_parts",), default=128)
        shard_dim = max(1, ple_dim // PLE_SHARD_DIVISOR)
        ple_conv = _int_field(tc, ("ple_conv_kernel_size",), default=4)
        streams = max(1, _int_field(tc, ("hc_count",), default=1))
        state_b = max(0, ple_conv - 1) * (streams + 2) * ple_dim * FP32_BYTES
        proj_p = hidden_size * (streams * hidden_size) + hidden_size * ple_dim
        tables.append(
            HostTableBreakSpec(
                name="ple",
                layers=ple_layers,
                table_bytes_total=shards * PLE_PADDED_ROWS_PER_SHARD * shard_dim,
                h2d_bytes_per_token=float(ple_dim),
                hbm_resident=True,
                shard_axis="tp",
                state_bytes_per_request=state_b,
                dense_proj_params=proj_p,
            )
        )
    eng_ids = _int_seq(tc.get("engram_layer_ids"))
    if eng_ids:
        rows_seq = _int_seq(tc.get("engram_num_embeddings"))
        orders = max(1, _int_field(tc, ("engram_max_ngram_size",), default=4) - 1)
        eng_heads = _int_field(tc, ("engram_n_heads",), default=8)
        eng_dim = _int_field(tc, ("engram_head_dim",), default=256)
        row_b = (eng_dim * (ENGRAM_SCALE_BLOCK_SIZE + 1)) // ENGRAM_SCALE_BLOCK_SIZE
        h2d_b = float(
            orders * eng_heads * eng_dim * (1.0 + 1.0 / ENGRAM_SCALE_BLOCK_SIZE)
        )
        for idx, lyr in enumerate(eng_ids):
            rows = rows_seq[idx] if idx < len(rows_seq) else 0
            tables.append(
                HostTableBreakSpec(
                    name=f"engram_l{lyr}",
                    layers=(lyr,),
                    table_bytes_total=rows * row_b,
                    h2d_bytes_per_token=h2d_b,
                    hbm_resident=False,
                    shard_axis="dp_local",
                )
            )
    return tuple(tables)


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
    moe_freq_seq = _as_sequence(tc.get("moe_layer_freq"))
    if moe_freq_seq is not None and len(moe_freq_seq) == num_layers:
        moe_layer_count = sum(
            1 for entry in moe_freq_seq if bool(_as_scalar_number(entry))
        )
        dense_prefix = max(0, num_layers - moe_layer_count)
    else:
        dense_prefix = _int_field(tc, ("first_k_dense_replace",)) or (
            len(mlp_only) if mlp_only is not None else 0
        )

    raw_inter = _int_field(tc, ("intermediate_size",), default=4 * hidden_size)
    moe_inter = _int_field(
        tc,
        ("moe_intermediate_size", "latent_moe_dim"),
        default=raw_inter * (num_experts > 0),
    )
    ffn_inter = _int_field(tc, ("dense_intermediate_size",), default=raw_inter)
    num_shared = _int_field(tc, ("n_shared_experts", "num_shared_experts"))
    shared_inter = _int_field(
        tc,
        ("shared_expert_intermediate_size", "shared_intermediate_size"),
        default=num_shared * moe_inter,
    )
    top_k = _int_field(tc, ("num_experts_per_tok", "num_experts_per_token")) * (
        num_experts > 0
    )
    attn_layers = max(0, num_layers - num_linear_layers)
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
        dense_unquantized,
        act_scheme,
    ) = _extract_quant_fields(tc, raw_cfg, num_layers, attn_layers)
    num_kv_heads = _int_field(tc, ("num_key_value_heads",), default=num_heads)
    head_dim = _int_field(
        tc,
        ("v_head_dim", "head_dim"),
        default=hidden_size // num_heads if num_heads else 0,
    )
    o_lora_rank = _int_field(tc, ("o_lora_rank",))
    o_groups = _int_field(tc, ("o_groups",), default=1)
    sparse_attn, dcp_span, ced_boundary = _parse_hf_sparse_attn(
        tc, (num_layers, num_kv_heads, head_dim, o_lora_rank)
    )
    hyper_conn = _parse_hf_hyper_connection(tc)
    host_tables = _parse_hf_host_tables(tc, num_layers, hidden_size)
    index_topk, index_head_dim, index_n_heads, num_indexer_layers = (
        (0, 0, 0, 0)
        if sparse_attn is not None
        else _resolve_dsa_fields(tc, num_layers, attn_layers)
    )
    ssm_dt = str(merged.get("mamba_ssm_dtype") or "").strip().lower()
    if ssm_dt in ("float32", "fp32", "torch.float32"):
        raw_state_dt = "fp32"
    elif ssm_dt in ("float16", "fp16", "torch.float16"):
        raw_state_dt = "fp16"
    else:
        raw_state_dt = "bf16"
    out_gate_str = str(tc.get("output_gate_type") or "").strip().lower()
    has_attn_gate = bool(tc.get("attn_output_gate", False)) or (
        out_gate_str not in ("", "none", "null", "false")
    )

    return _NormalizedHfConfig(
        model_type=mtype,
        is_multimodal=is_multimodal,
        has_gdn_kda_heads=has_gdn_kda_heads,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
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
        index_topk=index_topk,
        index_head_dim=index_head_dim,
        index_n_heads=index_n_heads,
        num_indexer_layers=num_indexer_layers,
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
        has_attn_output_gate=has_attn_gate,
        torch_dtype=torch_dt,
        quant_tag=quant_tag,
        quant_bits=bits,
        quant_w_type=quant_w_type,
        raw_weight_dtype=raw_wd,
        raw_expert_dtype=raw_ed,
        raw_layer_types=raw_layer_types,
        raw_act_dtype=raw_act_dt,
        weight_block_k=weight_block_k,
        has_quant_config=has_quant_config,
        has_explicit_torch_dtype=has_explicit_torch_dtype,
        dense_unquantized=dense_unquantized,
        raw_state_dtype=raw_state_dt,
        act_scheme=act_scheme,
        o_lora_rank=o_lora_rank,
        o_groups=o_groups,
        ced_boundary=ced_boundary,
        dcp_span=dcp_span,
        sparse_attn=sparse_attn,
        hyper_connection=hyper_conn,
        host_tables=host_tables,
    )


def supports_hf_config(
    raw_cfg: Mapping[str, object], *, require_model_type: bool = False
) -> None:
    """Validate that a HuggingFace config mapping matches a shipped preset."""
    norm = _normalize_hf_config(raw_cfg)
    norm.validate_supported(require_model_type=require_model_type)
    _adopt_matching_preset_name(norm.to_model_profile(name="model"))


def _parse_preset_owner(raw: Mapping[str, object]) -> SparseCacheOwner:
    """Parse one SparseCacheOwner entry from models.json."""
    elem_raw = raw.get("elem_bytes")
    max_scan_raw = raw.get("max_scan_records")
    return SparseCacheOwner(
        layer=int(str(raw["layer"])),
        record_tokens=float(str(raw.get("record_tokens", 1.0))),
        record_elems=int(str(raw.get("record_elems", 0))),
        elem_bytes=float(str(elem_raw)) if elem_raw is not None else None,
        record_overhead_bytes=float(str(raw.get("record_overhead_bytes", 0.0))),
        replication=cast(
            "SparseCacheReplication", str(raw.get("replication", "per_rank"))
        ),
        users=tuple(int(str(u)) for u in (_as_sequence(raw.get("users")) or ())),
        max_scan_records=int(str(max_scan_raw)) if max_scan_raw is not None else None,
    )


def _parse_preset_sparse_attn(raw: object) -> SparseAttnSpec | None:
    """Parse optional SparseAttnSpec dictionary from models.json."""
    mapping = _as_mapping(raw)
    if mapping is None:
        return None
    kv_seq = _as_sequence(mapping.get("kv_owners")) or ()
    idx_seq = _as_sequence(mapping.get("index_owners")) or ()
    win_seq = _as_sequence(mapping.get("window_layers")) or ()
    return SparseAttnSpec(
        kind=cast("SparseAttnKind", str(mapping["kind"])),
        window_tokens=int(str(mapping.get("window_tokens", 0))),
        window_bytes_per_token=float(str(mapping.get("window_bytes_per_token", 0.0))),
        window_layers=tuple(int(str(x)) for x in win_seq),
        select_tokens=int(str(mapping.get("select_tokens", 0))),
        index_heads=int(str(mapping.get("index_heads", 0))),
        index_kv_heads=int(str(mapping.get("index_kv_heads", 1))),
        index_dim=int(str(mapping.get("index_dim", 0))),
        index_compress_ratio=int(str(mapping.get("index_compress_ratio", 1))),
        kv_owners=tuple(
            _parse_preset_owner(m) for x in kv_seq if (m := _as_mapping(x)) is not None
        ),
        index_owners=tuple(
            _parse_preset_owner(m) for x in idx_seq if (m := _as_mapping(x)) is not None
        ),
        shared_kv_mqa=bool(mapping.get("shared_kv_mqa", False)),
        kv_owner_ring_bytes_per_req=int(
            str(mapping.get("kv_owner_ring_bytes_per_req", 0))
        ),
    )


def _parse_preset_hyper_connection(raw: object) -> HyperConnectionSpec | None:
    """Parse optional HyperConnectionSpec dictionary from models.json."""
    mapping = _as_mapping(raw)
    if mapping is None:
        return None
    return HyperConnectionSpec(
        kind=cast("HyperConnectionKind", str(mapping["kind"])),
        num_streams=int(str(mapping.get("num_streams", 4))),
        lowrank_dim=int(str(mapping.get("lowrank_dim", 0))),
        sinkhorn_iters=int(str(mapping.get("sinkhorn_iters", 0))),
        has_final_mixer=bool(mapping.get("has_final_mixer", False)),
    )


def _parse_preset_host_tables(raw: object) -> tuple[HostTableBreakSpec, ...]:
    """Parse optional HostTableBreakSpec list from models.json."""
    return tuple(
        HostTableBreakSpec(
            name=str(m["name"]),
            layers=tuple(int(str(x)) for x in (_as_sequence(m.get("layers")) or ())),
            table_bytes_total=int(str(m["table_bytes_total"])),
            h2d_bytes_per_token=float(str(m.get("h2d_bytes_per_token", 0.0))),
            hbm_resident=bool(m.get("hbm_resident", False)),
            shard_axis=cast("HostTableShardAxis", str(m.get("shard_axis", "tp"))),
            state_bytes_per_request=int(str(m.get("state_bytes_per_request", 0))),
            dense_proj_params=int(str(m.get("dense_proj_params", 0))),
        )
        for item in (_as_sequence(raw) or ())
        if (m := _as_mapping(item)) is not None
    )


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
                index_topk=int(entry.get("index_topk", 0)),
                index_head_dim=int(entry.get("index_head_dim", 0)),
                index_n_heads=int(entry.get("index_n_heads", 0)),
                num_indexer_layers=int(entry.get("num_indexer_layers", 0)),
                routed_expert_hidden_size=int(
                    entry.get("routed_expert_hidden_size", 0)
                ),
                weight_dtype=str(entry.get("weight_dtype", "fp8")),
                expert_dtype=str(entry.get("expert_dtype", "")),
                act_dtype=str(entry.get("act_dtype", "")),
                weight_block_k=int(entry.get("weight_block_k", 0)),
                kv_dtype=str(entry.get("kv_dtype", "fp8")),
                state_dtype=str(entry.get("state_dtype", "bf16")),
                model_type=str(entry.get("model_type", "")),
                has_attn_output_gate=bool(entry.get("has_attn_output_gate", False)),
                kv_compress_factor=float(entry.get("kv_compress_factor", 1.0)),
                residual_streams=int(entry.get("residual_streams", 1)),
                layer_families=tuple(str(x) for x in entry.get("layer_families", ())),
                dtype_source=f"preset ({slug})",
                o_lora_rank=int(entry.get("o_lora_rank", 0)),
                o_groups=int(entry.get("o_groups", 1)),
                ced_boundary=int(entry.get("ced_boundary", 0)),
                dcp_span=cast("DcpSpan", str(entry.get("dcp_span", "kv_heads"))),
                expert_act_dtype=str(entry.get("expert_act_dtype", "")),
                sparse_attn=_parse_preset_sparse_attn(entry.get("sparse_attn")),
                hyper_connection=_parse_preset_hyper_connection(
                    entry.get("hyper_connection")
                ),
                host_tables=_parse_preset_host_tables(entry.get("host_tables")),
            )
        )
        unmodeled = [
            fam
            for fam in presets[slug].layer_families
            if fam not in SUPPORTED_ONCHIP_FAMILIES and fam not in STRUCTURAL_LAYER_TAGS
        ]
        if unmodeled:
            msg = f"Preset {slug!r} contains unpriced layer families: {unmodeled}"
            raise UnsupportedArchitectureError(msg)
    aliases: dict[str, str] = {str(k): str(v) for k, v in payload["aliases"].items()}
    return presets, aliases, sources


def _repo_id_from_source(source: str) -> str:
    """Extract `org/repo` from a preset source string."""
    for token in source.split():
        if "/" in token:
            return token
    return source


def load_model_presets() -> tuple[
    Mapping[str, ModelProfile], Mapping[str, str], Mapping[str, str]
]:
    """Load and validate the shipped model presets, aliases, and sources."""
    return _load_model_presets()


def parse_hf_config_profile(
    raw_cfg: Mapping[str, object], *, name: str = "model"
) -> ModelProfile:
    """Parse a HuggingFace config mapping without enforcing preset identity."""
    return _normalize_hf_config(raw_cfg).to_model_profile(name=name)


MODEL_PRESETS, REGISTERED_MODEL_ALIASES, MODEL_PRESET_SOURCES = load_model_presets()
SUPPORTED_MODEL_TYPES: frozenset[str] = frozenset(
    profile.model_type.strip().lower().removesuffix("_text")
    for profile in MODEL_PRESETS.values()
    if profile.model_type
)
SUPPORTED_REPOSITORIES: tuple[str, ...] = tuple(
    _repo_id_from_source(MODEL_PRESET_SOURCES[slug]) for slug in MODEL_PRESETS
)


DTYPE_BASIS_PHRASES: Mapping[str, str] = {
    "config": "dtypes from the cited config",
}
PRESET_DTYPE_BASES: frozenset[str] = frozenset(DTYPE_BASIS_PHRASES)


def load_preset_dtype_basis() -> Mapping[str, str]:
    """Load each preset's dtype basis from models.json (`config` only)."""
    payload = json.loads((PRESETS_DIR / "models.json").read_text("utf-8"))
    basis: dict[str, str] = {}
    for slug, entry in payload["presets"].items():
        value = str(entry.get("dtype_basis", ""))
        if value not in PRESET_DTYPE_BASES:
            msg = (
                f"Preset {slug!r} dtype_basis must be one of "
                f"{sorted(PRESET_DTYPE_BASES)}, got {value!r}"
            )
            raise errors.MegakernelConfigError(msg)
        basis[slug] = value
    return basis


MODEL_PRESET_DTYPE_BASIS = load_preset_dtype_basis()


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

    def submodule_label(self) -> str:
        """Return the attention mesh and the routed-expert mesh as one label.

        Attention uses every axis except EP. Routed experts split the ranks
        into expert owner groups (a declared `expert_owner_groups`, else
        `ep`) and tensor-shard inside each group, as in `MoE: TP4 x EP8`.
        """
        attention = replace(self, ep=1).label()
        if self.ep <= 1:
            return f"Attn: {attention} \u00b7 MoE: TP{self.tp}"
        declared = self.megakernel or {}
        groups = int(str(declared.get("expert_owner_groups", self.ep)))
        ranks = max(self.ep, self.tp)
        inner_tp = max(1, ranks // max(1, groups))
        moe = f"EP{groups}" if inner_tp == 1 else f"TP{inner_tp} x EP{groups}"
        return f"Attn: {attention} \u00b7 MoE: {moe}"


@dataclass(frozen=True)
class DeploymentPreset:
    """Shipped prefill and decode layouts for one model, used before auto-pick."""

    decode: ParallelLayout
    prefill: ParallelLayout
    saturation: ParallelLayout | None = None
    prefill_windows: tuple[int, ...] = ()
    megakernel_layouts: tuple[ParallelLayout, ...] = ()
    unmodeled_notes: tuple[str, ...] = ()

    def named_layouts(self) -> tuple[tuple[str, ParallelLayout], ...]:
        """Return shipped role labels paired with their ParallelLayout."""
        layouts: list[tuple[str, ParallelLayout]] = [
            ("decode", self.decode),
            ("prefill", self.prefill),
        ]
        if self.saturation is not None:
            layouts.append(("saturation", self.saturation))
        layouts.extend(
            ("megakernel layout", extra_layout)
            for extra_layout in self.megakernel_layouts
        )
        return tuple(layouts)


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


InCallExpertMode = Literal["tp", "ep", "none"]


def _decl_int(val: object, field_name: str) -> int:
    """Parse an integer declaration field while rejecting bools and non-ints."""
    match val:
        case bool():
            msg = f"{field_name} must be an integer, got bool"
            raise errors.MegakernelConfigError(msg)
        case int():
            return val
        case _:
            msg = f"{field_name} must be an integer, got {type(val).__name__}"
            raise errors.MegakernelConfigError(msg)


def _decl_str(val: object, field_name: str) -> str:
    """Parse a string declaration field while rejecting non-strings."""
    if not isinstance(val, str):
        msg = f"{field_name} must be a string, got {type(val).__name__}"
        raise errors.MegakernelConfigError(msg)
    return val


@dataclass(frozen=True)
class DeclaredCollectiveStep:
    """One per-sublayer collective step declared inside a fused megakernel."""

    kind: CollectiveStepKind
    group_size: int
    calls_per_layer: int = DEFAULT_DECLARED_CALLS_PER_LAYER

    def __post_init__(self) -> None:
        """Validate collective kind, power-of-two group size, and call count."""
        if self.kind not in DECLARED_COLLECTIVE_KINDS:
            msg = (
                f"DeclaredCollectiveStep.kind must be one of "
                f"{sorted(DECLARED_COLLECTIVE_KINDS)}, got {self.kind!r}"
            )
            raise errors.MegakernelConfigError(msg)
        if (
            isinstance(self.group_size, bool)
            or not isinstance(self.group_size, int)
            or self.group_size < MIN_COLLECTIVE_GROUP_SIZE
            or (self.group_size & (self.group_size - 1)) != 0
        ):
            msg = (
                "DeclaredCollectiveStep.group_size must be a power-of-two "
                f"integer >= {MIN_COLLECTIVE_GROUP_SIZE}, got {self.group_size!r}"
            )
            raise errors.MegakernelConfigError(msg)
        if (
            isinstance(self.calls_per_layer, bool)
            or not isinstance(self.calls_per_layer, int)
            or self.calls_per_layer < 1
        ):
            msg = (
                "DeclaredCollectiveStep.calls_per_layer must be an integer >= 1, "
                f"got {self.calls_per_layer!r}"
            )
            raise errors.MegakernelConfigError(msg)


@dataclass(frozen=True)
class DeclaredCollectives:
    """Per-step all-reduce count and collective steps declared for a megakernel."""

    all_reduces: int | None = None
    steps: tuple[DeclaredCollectiveStep, ...] = ()

    def __post_init__(self) -> None:
        """Validate non-negative integer all_reduces override when set."""
        if self.all_reduces is not None and (
            isinstance(self.all_reduces, bool)
            or not isinstance(self.all_reduces, int)
            or self.all_reduces < 0
        ):
            msg = (
                "DeclaredCollectives.all_reduces must be an integer >= 0, "
                f"got {self.all_reduces!r}"
            )
            raise errors.MegakernelConfigError(msg)


@dataclass(frozen=True)
class ExpertModeRule:
    """Row-count threshold rule selecting in-call TP or EP expert execution."""

    max_rows: int
    mode: DeclaredExpertMode

    def __post_init__(self) -> None:
        """Validate positive integer max_rows threshold and expert mode."""
        if (
            isinstance(self.max_rows, bool)
            or not isinstance(self.max_rows, int)
            or self.max_rows < 1
        ):
            msg = (
                "ExpertModeRule.max_rows must be an integer >= 1, "
                f"got {self.max_rows!r}"
            )
            raise errors.MegakernelConfigError(msg)
        if self.mode not in DECLARED_EXPERT_MODES:
            msg = (
                f"ExpertModeRule.mode must be one of "
                f"{sorted(DECLARED_EXPERT_MODES)}, got {self.mode!r}"
            )
            raise errors.MegakernelConfigError(msg)


def _parse_declared_step_kind(raw_kind: object, field_name: str) -> CollectiveStepKind:
    """Validate and narrow a collective step kind string."""
    kind_str = _decl_str(raw_kind, field_name)
    if kind_str in DECLARED_COLLECTIVE_KINDS:
        return cast("CollectiveStepKind", kind_str)
    msg = (
        f"{field_name} must be one of {sorted(DECLARED_COLLECTIVE_KINDS)}, "
        f"got {kind_str!r}"
    )
    raise errors.MegakernelConfigError(msg)


def _parse_declared_expert_mode(
    raw_mode: object, field_name: str
) -> DeclaredExpertMode:
    """Validate and narrow a declared expert execution mode string."""
    mode_str = _decl_str(raw_mode, field_name)
    if mode_str in DECLARED_EXPERT_MODES:
        return cast("DeclaredExpertMode", mode_str)
    msg = (
        f"{field_name} must be one of {sorted(DECLARED_EXPERT_MODES)}, got {mode_str!r}"
    )
    raise errors.MegakernelConfigError(msg)


def parse_declared_collectives(raw: object) -> DeclaredCollectives:
    """Parse and validate a `megakernel.collectives` mapping."""
    if isinstance(raw, DeclaredCollectives):
        return raw
    if not isinstance(raw, Mapping):
        msg = f"megakernel.collectives must be a mapping, got {type(raw).__name__}"
        raise errors.MegakernelConfigError(msg)
    allowed_keys = {"all_reduces", "steps"}
    unknown_keys = sorted(set(raw.keys()) - allowed_keys)
    if unknown_keys:
        msg = f"Unsupported keys in megakernel.collectives: {unknown_keys}"
        raise errors.MegakernelConfigError(msg)
    ar_raw = raw.get("all_reduces")
    all_reduces = (
        _decl_int(ar_raw, "megakernel.collectives.all_reduces")
        if ar_raw is not None
        else None
    )
    steps_raw = raw.get("steps")
    steps: list[DeclaredCollectiveStep] = []
    if steps_raw is not None:
        if not isinstance(steps_raw, (list, tuple)):
            msg = (
                "megakernel.collectives.steps must be a list, "
                f"got {type(steps_raw).__name__}"
            )
            raise errors.MegakernelConfigError(msg)
        for idx, item in enumerate(steps_raw):
            if not isinstance(item, Mapping):
                msg = f"megakernel.collectives.steps[{idx}] must be a mapping"
                raise errors.MegakernelConfigError(msg)
            step_allowed = {"kind", "group_size", "calls_per_layer"}
            step_unknown = sorted(set(item.keys()) - step_allowed)
            if step_unknown:
                msg = (
                    f"Unsupported keys in megakernel.collectives.steps[{idx}]: "
                    f"{step_unknown}"
                )
                raise errors.MegakernelConfigError(msg)
            kind = _parse_declared_step_kind(
                item.get("kind"), f"megakernel.collectives.steps[{idx}].kind"
            )
            group_size = _decl_int(
                item.get("group_size"),
                f"megakernel.collectives.steps[{idx}].group_size",
            )
            calls_per_layer = _decl_int(
                item.get("calls_per_layer", DEFAULT_DECLARED_CALLS_PER_LAYER),
                f"megakernel.collectives.steps[{idx}].calls_per_layer",
            )
            steps.append(
                DeclaredCollectiveStep(
                    kind=kind,
                    group_size=group_size,
                    calls_per_layer=calls_per_layer,
                )
            )
    return DeclaredCollectives(all_reduces=all_reduces, steps=tuple(steps))


def parse_expert_mode_rules(raw: object) -> tuple[ExpertModeRule, ...]:
    """Parse strictly ascending row-threshold rules from `expert_mode_by_rows`."""
    if not isinstance(raw, (list, tuple)) or not raw:
        msg = "megakernel.expert_mode_by_rows must be a non-empty list of rules"
        raise errors.MegakernelConfigError(msg)
    rules: list[ExpertModeRule] = []
    prev_max_rows = 0
    for idx, item in enumerate(raw):
        if isinstance(item, ExpertModeRule):
            rule = item
        elif isinstance(item, Mapping):
            rule_allowed = {"max_rows", "mode"}
            rule_unknown = sorted(set(item.keys()) - rule_allowed)
            if rule_unknown:
                msg = f"Unsupported keys in expert_mode_by_rows[{idx}]: {rule_unknown}"
                raise errors.MegakernelConfigError(msg)
            max_rows = _decl_int(
                item.get("max_rows"), f"expert_mode_by_rows[{idx}].max_rows"
            )
            mode = _parse_declared_expert_mode(
                item.get("mode"), f"expert_mode_by_rows[{idx}].mode"
            )
            rule = ExpertModeRule(max_rows=max_rows, mode=mode)
        else:
            msg = f"expert_mode_by_rows[{idx}] must be a mapping"
            raise errors.MegakernelConfigError(msg)
        if rule.max_rows <= prev_max_rows:
            msg = (
                "expert_mode_by_rows max_rows thresholds must be strictly "
                f"ascending, got {rule.max_rows} after {prev_max_rows}"
            )
            raise errors.MegakernelConfigError(msg)
        prev_max_rows = rule.max_rows
        rules.append(rule)
    return tuple(rules)


def _validate_megakernel_declaration(
    mk_map: Mapping[str, object],
    *,
    ep_size: int,
) -> dict[str, object]:
    """Validate and normalize a megakernel declaration mapping."""
    if not str(mk_map.get("source", "")).strip():
        msg = "megakernel declaration in deployments.json requires non-empty 'source'"
        raise errors.MegakernelConfigError(msg)
    normalized = dict(mk_map)
    if "expert_owner_groups" in mk_map:
        raw_groups = mk_map["expert_owner_groups"]
        if (
            isinstance(raw_groups, bool)
            or not isinstance(raw_groups, int)
            or not (1 <= raw_groups <= ep_size)
            or ep_size % raw_groups != 0
        ):
            msg = (
                "megakernel 'expert_owner_groups' must be an int in "
                f"[1, ep={ep_size}] that divides ep={ep_size}, got {raw_groups!r}"
            )
            raise errors.MegakernelConfigError(msg)
    if "collectives" in mk_map:
        parsed_coll = parse_declared_collectives(mk_map["collectives"])
        coll_dict: dict[str, object] = {
            "steps": [
                {
                    "kind": step.kind,
                    "group_size": step.group_size,
                    "calls_per_layer": step.calls_per_layer,
                }
                for step in parsed_coll.steps
            ]
        }
        if parsed_coll.all_reduces is not None:
            coll_dict["all_reduces"] = parsed_coll.all_reduces
        normalized["collectives"] = coll_dict
    if "expert_mode_by_rows" in mk_map:
        parsed_rules = parse_expert_mode_rules(mk_map["expert_mode_by_rows"])
        normalized["expert_mode_by_rows"] = [
            {"max_rows": rule.max_rows, "mode": rule.mode} for rule in parsed_rules
        ]
    return normalized


def parse_parallel_layout(raw: Mapping[str, object]) -> ParallelLayout:
    """Parse a ParallelLayout mapping from deployments.json."""
    groups = _int_tuple(raw.get("layer_groups"), "layer_groups")
    mk_map = _as_mapping(raw.get("megakernel"))
    mk_decl: Mapping[str, object] | None = None
    ep_size = _preset_int(raw, "ep", 1)
    if mk_map is not None:
        mk_decl = _validate_megakernel_declaration(mk_map, ep_size=ep_size)
    return ParallelLayout(
        tp=_preset_int(raw, "tp", 1),
        ep=ep_size,
        dp=_preset_int(raw, "dp", 1),
        pp=_preset_int(raw, "pp", 1),
        dcp=_preset_int(raw, "dcp", 1),
        pcp=_preset_int(raw, "pcp", 1),
        spec_tokens=_preset_int(raw, "spec_tokens", 0),
        layer_groups=groups,
        megakernel=mk_decl,
    )


def _declared_strings(raw: object) -> tuple[str, ...]:
    """Return string entries from a declaration sequence field."""
    seq = _as_sequence(raw)
    if seq is None:
        return ()
    return tuple(item for item in seq if type(item) is str and item)


def validate_declared_layouts(slug: str, preset: DeploymentPreset) -> None:
    """Validate non-duplicate 6-axis declarations and expert-group divisibility."""
    seen_keys: set[tuple[tuple[int, int, int, int, int, int], str, str]] = set()
    for _role, layout in preset.named_layouts():
        if layout.megakernel is None:
            continue
        declaration = layout.megakernel
        if not str(declaration.get("source", "")).strip():
            continue
        owner_groups = declaration.get("expert_owner_groups")
        if isinstance(owner_groups, int) and not isinstance(owner_groups, bool):
            model_profile = MODEL_PRESETS[slug]
            if (
                model_profile.num_experts > 0
                and model_profile.num_experts % owner_groups != 0
            ):
                msg = (
                    f"megakernel 'expert_owner_groups' ({owner_groups}) in {slug!r} "
                    f"must divide num_experts ({model_profile.num_experts})"
                )
                raise errors.MegakernelConfigError(msg)
        axis_key = (
            layout.tp,
            layout.ep,
            layout.dp,
            layout.pp,
            layout.dcp,
            layout.pcp,
        )
        scopes = _declared_strings(declaration.get("scopes"))
        phases = _declared_strings(declaration.get("phases")) or DEFAULT_DECLARED_PHASES
        block_keys = {(axis_key, scope, phase) for scope in scopes for phase in phases}
        overlapping_keys = seen_keys & block_keys
        if overlapping_keys:
            _, dup_scope, dup_phase = min(overlapping_keys)
            msg = (
                f"Duplicate megakernel declaration in {slug!r} on {layout.label()} "
                f"for scope={dup_scope!r}, phase={dup_phase!r}"
            )
            raise errors.MegakernelConfigError(msg)
        seen_keys.update(block_keys)


def _parse_megakernel_layouts(slug: str, raw: object) -> tuple[ParallelLayout, ...]:
    """Parse optional megakernel_layouts entries for a deployment preset."""
    if raw is None:
        return ()
    match raw:
        case str() | bytes() | bytearray():
            seq = None
        case Sequence():
            seq = tuple(raw)
        case _:
            seq = None
    if seq is None:
        msg = f"deployments.json {slug!r} megakernel_layouts must be a sequence"
        raise errors.MegakernelConfigError(msg)
    layouts: list[ParallelLayout] = []
    for idx, item in enumerate(seq):
        item_map = _as_mapping(item)
        if item_map is None:
            msg = (
                f"deployments.json {slug!r} megakernel_layouts[{idx}] must be a mapping"
            )
            raise errors.MegakernelConfigError(msg)
        layout = parse_parallel_layout(item_map)
        if layout.megakernel is None:
            msg = (
                f"deployments.json {slug!r} megakernel_layouts[{idx}] "
                "requires a 'megakernel' block"
            )
            raise errors.MegakernelConfigError(msg)
        layouts.append(layout)
    return tuple(layouts)


def _note_tuple(slug: str, raw: object) -> tuple[str, ...]:
    """Parse the optional `unmodeled_notes` list of one deployment entry."""
    if raw is None:
        return ()
    entries = _as_sequence(raw)
    if entries is None:
        msg = f"Deployment preset {slug!r} unmodeled_notes must be a list of strings"
        raise errors.MegakernelConfigError(msg)
    return tuple(_decl_str(entry, f"{slug}.unmodeled_notes") for entry in entries)


def load_deployment_presets() -> tuple[
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
        sat = parse_parallel_layout(sat_map) if sat_map is not None else None
        windows = _int_tuple(entry.get("prefill_windows"), "prefill_windows")
        mk_layouts = _parse_megakernel_layouts(slug, entry.get("megakernel_layouts"))
        preset = DeploymentPreset(
            decode=parse_parallel_layout(entry["decode"]),
            prefill=parse_parallel_layout(entry["prefill"]),
            saturation=sat,
            prefill_windows=windows,
            megakernel_layouts=mk_layouts,
            unmodeled_notes=_note_tuple(slug, entry.get("unmodeled_notes")),
        )
        validate_declared_layouts(slug, preset)
        deps[slug] = preset
    return deps, sources


DEPLOYMENT_PRESETS, DEPLOYMENT_PRESET_SOURCES = load_deployment_presets()


def _identity_view(profile: ModelProfile) -> ModelProfile:
    """Return `profile` with non-geometric fields blanked for preset matching.

    Quantization block size, KV/activation/state dtypes, dense FFN width, and a
    multimodal `_text` suffix are normalized away; weight and expert dtypes are
    checked separately when a preset fixes its dtypes from config.
    """
    return replace(
        finalize_model_profile(profile),
        name="",
        intermediate_size=0,
        weight_dtype="",
        expert_dtype="",
        act_dtype="",
        expert_act_dtype="",
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


def normalize_preset_name(name: str) -> str:
    """Normalize a model name the way declaration lookup matches preset slugs."""
    return name.lower().replace(".", "_").replace("-", "_")


def _adopt_matching_preset_name(profile: ModelProfile) -> ModelProfile:
    """Rename a config-derived profile to its matching shipped preset or fail closed.

    A config whose geometry and config dtypes match no shipped preset raises
    `UnsupportedArchitectureError` naming the supported repositories and
    onboarding documentation.
    """
    preset_name = match_preset_by_geometry(profile)
    if preset_name is not None:
        return replace(profile, name=preset_name)
    msg = (
        f"Config {profile.name!r} (model_type={profile.model_type!r}) matches no "
        "validated preset by geometry and config dtypes. "
        f"Supported repositories: {', '.join(SUPPORTED_REPOSITORIES)}. "
        f"See {ONBOARDING_DOC_REF}."
    )
    raise UnsupportedArchitectureError(msg)


def _resolve_hf_profile(
    raw_cfg: Mapping[str, object],
    name: str,
    dtypes: tuple[str | None, str | None, str, str | None],
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
    state_dtype: str | None = None,
) -> ModelProfile:
    """Parse and validate a HuggingFace config mapping into a shipped preset profile."""
    return _adopt_matching_preset_name(
        _resolve_hf_profile(
            raw_cfg,
            name,
            (weight_dtype, expert_dtype, "fp8", state_dtype),
            act_dtype=act_dtype,
        )
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
    kv_raw = (kv_dtype or "fp8").lower()
    kv_val = KV_DTYPE_ALIASES.get(kv_raw, kv_raw)
    st_val = state_dtype.lower() if state_dtype is not None else None
    dtypes = (weight_dtype, expert_dtype, kv_val, st_val)
    match target:
        case ModelProfile():
            return finalize_model_profile(target)
        case Mapping() as mapping:
            return _adopt_matching_preset_name(
                _resolve_hf_profile(mapping, "model", dtypes, act_dtype=act_dtype)
            )
        case _:
            pass

    raw_str = str(target).strip()
    path = Path(raw_str)
    if path.is_file():
        return _adopt_matching_preset_name(
            _resolve_hf_profile(
                json.loads(path.read_text("utf-8")),
                path.parent.name or path.stem,
                dtypes,
                act_dtype=act_dtype,
            )
        )
    if not raw_str.endswith(".json") and (path / "config.json").is_file():
        cfg_file = path / "config.json"
        return _adopt_matching_preset_name(
            _resolve_hf_profile(
                json.loads(cfg_file.read_text("utf-8")),
                path.name or cfg_file.stem,
                dtypes,
                act_dtype=act_dtype,
            )
        )

    tail = raw_str.lower().strip("/").split("/")[-1]
    preset_key = REGISTERED_MODEL_ALIASES.get(tail)
    if preset_key is not None and not raw_str.endswith(".json"):
        model_profile = MODEL_PRESETS[preset_key]
        resolved_w = (weight_dtype or model_profile.weight_dtype).lower()
        resolved_kv_raw = (kv_dtype or model_profile.kv_dtype).lower()
        resolved_kv = KV_DTYPE_ALIASES.get(resolved_kv_raw, resolved_kv_raw)
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
        return _adopt_matching_preset_name(
            _resolve_hf_profile(
                json.loads(snap_cfg.read_text("utf-8")),
                tail,
                dtypes,
                act_dtype=act_dtype,
                require_model_type=True,
            )
        )

    msg = (
        f"Unregistered model or missing config path: {target!r}. "
        f"Registered presets: {sorted(MODEL_PRESETS)}; "
        f"supported repositories: {', '.join(SUPPORTED_REPOSITORIES)}. "
        f"See {ONBOARDING_DOC_REF}."
    )
    raise UnsupportedArchitectureError(msg)
