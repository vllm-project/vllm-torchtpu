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
"""Model geometry, canonical model identity registry, and support guards."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Literal

from .constants import (
    BF16_BYTES,
    BLOCK_PARALLEL_ENABLED,
    CSA2_INDEX_RECORD_BYTES,
    FP8_BYTES,
    FP8_UE8M0_32X32_BYTES,
    FP8_UE8M0_BYTES,
    FP32_BYTES,
    GLU_MATMUL_COUNT,
    HF_REPO_TO_CANONICAL_ID,
    HYBRID_PATTERN_MAP,
    LAYER_TYPE_MAP,
    LINEAR_ATTENTION_LAYER_TOKENS,
    MULTIMODAL_KEYS,
    NON_GATED_MATMUL_COUNT,
    SUPPORTED_MODELS,
    VALID_LAYER_TOKENS,
    WEIGHT_BLOCK_DIM_LEN,
    ModelSpec,
)

__all__ = [
    "DraftSpec",
    "ExecutionMode",
    "HostTableResidency",
    "HostTableSpec",
    "KeyTracker",
    "LayerCounts",
    "ModelGeometry",
    "ModelIdentity",
    "SparseAttnSpec",
    "UnsupportedArchitectureError",
    "count_layer_types",
    "draft_layer_geometry",
    "from_hf_config",
    "host_table_hbm_fraction",
    "resolve_model_identity",
    "supports",
]

type HostTableResidency = Literal["auto", "hbm", "host", "first"]
type ExecutionMode = Literal["full", "ced"]


class KeyTracker(Mapping[str, object]):
    """Read-only mapping wrapper that records dotted paths on value reads."""

    def __init__(
        self,
        data: Mapping[str, object],
        *,
        prefix: str = "",
        consumed: set[str] | None = None,
    ) -> None:
        """Initialize tracker over ``data`` sharing a consumed-path set."""
        self._data = data
        self._prefix = prefix
        self._consumed: set[str] = consumed if consumed is not None else set()
        self._children: dict[str, KeyTracker] = {}

    @property
    def raw_data(self) -> Mapping[str, object]:
        """Return the underlying mapping without recording key reads."""
        return self._data

    @property
    def consumed_keys(self) -> frozenset[str]:
        """Return dotted key paths whose values were accessed."""
        return frozenset(self._consumed)

    def _path(self, key: str) -> str:
        """Return dotted key path relative to this mapping prefix."""
        return f"{self._prefix}.{key}" if self._prefix else key

    def __contains__(self, key: object) -> bool:
        """Check key presence without marking it as consumed."""
        return key in self._data

    def __getitem__(self, key: str) -> object:
        """Return value for ``key`` and record its dotted path as consumed."""
        val = self._data[key]
        path = self._path(key)
        self._consumed.add(path)
        if isinstance(val, Mapping) and not isinstance(val, KeyTracker):
            child = self._children.get(key)
            if child is None:
                child = KeyTracker(val, prefix=path, consumed=self._consumed)
                self._children[key] = child
            return child
        return val

    def __iter__(self) -> Iterator[str]:
        """Iterate keys without marking them as consumed."""
        return iter(self._data)

    def __len__(self) -> int:
        """Return key count of the underlying mapping."""
        return len(self._data)

    def __str__(self) -> str:
        """Format underlying mapping string without marking keys consumed."""
        return repr(self._data)

    def __repr__(self) -> str:
        """Format underlying mapping representation without marking keys consumed."""
        return repr(self._data)


def _first_present(mapping: Mapping[str, object], *keys: str) -> object:
    """Return first truthy value across ``keys``, failing on conflicting aliases."""
    chosen: object = None
    chosen_key: str | None = None
    for key in keys:
        if key not in mapping:
            continue
        val = mapping[key]
        if val is None or val == 0:
            continue
        if chosen is None:
            chosen = val
            chosen_key = key
        elif val != chosen:
            msg = (
                f"Conflicting HF config alias keys: {chosen_key}={chosen!r} "
                f"vs {key}={val!r}"
            )
            raise ValueError(msg)
    return chosen


def _first_non_none(
    mappings: Sequence[Mapping[str, object]],
    *keys: str,
) -> object:
    """Return first non-None value across scopes and keys without short-circuiting."""
    chosen: object = None
    for mapping in mappings:
        for key in keys:
            if key in mapping:
                val = mapping[key]
                if chosen is None and val is not None:
                    chosen = val
    return chosen


def _as_mapping(value: object) -> Mapping[str, object]:
    """Return value as a mapping when compatible with HF config JSON, else empty."""
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: object) -> Sequence[object] | None:
    """Return value as a non-string sequence when valid, else None."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return None


def _as_int(value: object, default: int = 0) -> int:
    """Convert JSON scalar number or numeric string to int cleanly."""
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float, str)):
        return int(value)
    return default


def _as_float(value: object, default: float = 0.0) -> float:
    """Convert JSON scalar number or numeric string to float cleanly."""
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float, str)):
        return float(value)
    return default


class UnsupportedArchitectureError(ValueError):
    """Raised when a HuggingFace config describes an unmodeled architecture."""


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Canonical resolved model identity from HF config and optional repo ID."""

    canonical_id: str
    model_type: str
    architectures: tuple[str, ...]
    hf_repo: str | None = None


_MODEL_TYPE_TO_SPEC: dict[str, ModelSpec] = {
    mt: spec for spec in SUPPORTED_MODELS.values() for mt in spec.model_types
}
_VALID_INDEXER_TYPES: frozenset[str] = frozenset({"full", "shared"})
_MLP_LAYER_TYPE_MAP: dict[str, str] = {"dense": "dense", "sparse": "moe", "moe": "moe"}


@dataclass(frozen=True, slots=True)
class LayerCounts:
    """Transformer layer counts grouped by attention and recurrence family.

    `mla` and `eff_mla` include DSA layers (`mla >= dsa`), so `attn_total`
    (`gqa + swa + mla`) spans all KV-bearing attention layers without double
    counting.
    """

    gqa: int = 0
    swa: int = 0
    mla: int = 0
    dsa: int = 0
    gdn: int = 0
    kda: int = 0
    eff_gqa: float = 0.0
    eff_swa: float = 0.0
    eff_mla: float = 0.0
    eff_dsa: float = 0.0

    @property
    def attn_total(self) -> int:
        """Total number of KV-bearing attention layers."""
        return self.gqa + self.swa + self.mla


def _require_non_negative(label: str, *values: float) -> None:
    """Validate that every physical dimension or byte width is non-negative."""
    if min(values) < 0.0:
        msg = f"{label} numeric fields must be >= 0"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class SparseAttnSpec:
    """Pure-physics specification for sparse attention (CSA2, MSA, or QSA)."""

    kind: Literal["csa2", "msa", "qsa"]
    kv_source_layer_ids: tuple[int, ...] = ()
    index_source_layer_ids: tuple[int, ...] = ()
    candidate_scan_cap: int = 0
    index_record_bytes: float = 0.0
    compressed_key_bytes_per_tok: float = 0.0
    index_cache_sharded: bool = False
    sparse_block_size: int = 0
    sparse_topk_blocks: int = 0
    sparse_local_block: int = 0
    dcp_span: str = "kv_head_group"
    ced_boundary: int = 0
    swa_bytes_per_tok: float = 0.0
    o_lora_rank: int = 0
    o_groups: int = 1
    fixed_state_page_bytes: float = 0.0
    ced_post_only: bool = False

    def __post_init__(self) -> None:
        """Validate non-negative sparse attention dimensions and group count."""
        if self.o_groups < 1:
            msg = f"SparseAttnSpec.o_groups must be >= 1, got {self.o_groups}"
            raise ValueError(msg)
        _require_non_negative(
            "SparseAttnSpec",
            self.candidate_scan_cap,
            self.index_record_bytes,
            self.compressed_key_bytes_per_tok,
            self.sparse_block_size,
            self.sparse_topk_blocks,
            self.sparse_local_block,
            self.ced_boundary,
            self.swa_bytes_per_tok,
            self.o_lora_rank,
            self.fixed_state_page_bytes,
        )


@dataclass(frozen=True, slots=True)
class HostTableSpec:
    """Pure-physics specification for a large host- or HBM-resident lookup table."""

    name: str
    layer_ids: tuple[int, ...]
    elem_count: int
    scale_ratio: float = 0.0
    table_dtype_role: Literal["w_expert", "w_dense", "w_aux"] = "w_expert"
    dense_proj_dtype_role: Literal["w_dense", "w_aux"] = "w_dense"
    default_hbm_resident: bool = False
    shard_axis: Literal["tp", "dp_local"] = "tp"
    h2d_bytes_per_tok: float = 0.0
    state_bytes_per_req: float = 0.0
    dense_proj_params: int = 0
    resident_on_first: bool | None = None

    def __post_init__(self) -> None:
        """Validate non-negative sizes and 0-indexed layer IDs."""
        if any(layer_id < 0 for layer_id in self.layer_ids):
            msg = f"HostTableSpec.layer_ids must be 0-indexed, got {self.layer_ids}"
            raise ValueError(msg)
        _require_non_negative(
            "HostTableSpec",
            self.elem_count,
            self.scale_ratio,
            self.h2d_bytes_per_tok,
            self.state_bytes_per_req,
            self.dense_proj_params,
        )


@dataclass(frozen=True, slots=True)
class DraftSpec:
    """Pure-physics specification for an MTP or parallel block drafter."""

    kind: Literal["mtp", "dspark"]
    num_layers: int
    layer_kind: str
    num_experts: int = 0
    top_k: int = 0
    moe_inter: int = 0
    shared_inter: int = 0
    dense_inter: int = 0
    sliding_window: int = 0
    block_size: int = 1
    fc_in_mult: int = 1
    fc_replicated: bool = False
    markov_rank: int = 0
    draft_kv_dtype: str = ""
    skip_draft_kv_on_offload: bool = False
    single_row_lm_head_verify: bool = False

    def __post_init__(self) -> None:
        """Validate non-negative draft layer counts and block size."""
        if self.block_size < 1:
            msg = f"DraftSpec.block_size must be >= 1, got {self.block_size}"
            raise ValueError(msg)
        _require_non_negative(
            "DraftSpec",
            self.num_layers,
            self.num_experts,
            self.top_k,
            self.moe_inter,
            self.shared_inter,
            self.dense_inter,
            self.sliding_window,
            self.fc_in_mult,
            self.markov_rank,
        )


def host_table_hbm_fraction(
    spec: HostTableSpec, spec_index: int, residency: HostTableResidency
) -> float:
    """Return fraction of ``spec`` resident in HBM under ``residency``."""
    if residency in ("hbm", "host"):
        return 1.0 if residency == "hbm" else 0.0
    if residency == "auto":
        return 1.0 if spec.default_hbm_resident else 0.0
    if spec.resident_on_first is not None:
        return 1.0 if spec.resident_on_first else 0.0
    return 0.0 if spec_index > 0 else 1.0 / float(max(1, len(spec.layer_ids)))


@dataclass(frozen=True, slots=True)
class ModelGeometry:
    """Architecture dimensions shared across attention, recurrence, and FFN families."""

    hidden: int
    num_layers: int
    layer_types: tuple[str, ...]
    num_heads: int
    num_kv_heads: int
    head_dim: int
    attn_output_gate: bool
    linear_k_heads: int
    linear_v_heads: int
    linear_k_dim: int
    linear_v_dim: int
    conv_kernel: int
    num_experts: int
    top_k: int
    moe_inter: int
    shared_inter: int
    vocab: int
    mtp_layers: int
    dense_inter: int = 0
    hc_mult: int = 1
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    index_n_heads: int = 0
    index_head_dim: int = 0
    index_topk: int = 0
    sliding_window: int = 0
    first_k_dense_replace: int = 0
    ple_dim: int = 0
    compress_ratios: tuple[float, ...] = ()
    block_parallel: int = 1
    mtp_inter: int = 0
    n_moe_layers: int | None = None
    n_dense_layers: int | None = None
    weight_block_size: tuple[int, int] = (1, 1)
    global_head_dim: int = 0
    ffn_kinds: tuple[str, ...] = ()
    routed_expert_hidden: int = 0
    ffn_num_matrices: int = GLU_MATMUL_COUNT
    tie_word_embeddings: bool = False
    model_id: str = ""
    indexer_types: tuple[str, ...] = ()
    index_share_for_mtp_iteration: bool = False
    qsa_budget: int = 0
    hc_lowrank: int = 0
    ple_layer_ids: tuple[int, ...] = ()
    max_position_embeddings: int = 0
    sparse_attn: SparseAttnSpec | None = None
    host_tables: tuple[HostTableSpec, ...] = ()
    draft: DraftSpec | None = None

    def ffn_layer_counts(self) -> tuple[int, int]:
        """Return (n_moe_ffn, n_dense_ffn) counted from layer_ffn_kinds."""
        kinds = layer_ffn_kinds(self)
        return (kinds.count("moe"), kinds.count("dense"))


def draft_layer_geometry(
    geometry: ModelGeometry, *, single_layer: bool = False
) -> ModelGeometry:
    """Build standalone ModelGeometry describing ``geometry.draft`` layers."""
    draft = geometry.draft
    if draft is None or draft.num_layers <= 0:
        msg = "draft_layer_geometry requires geometry.draft with num_layers > 0"
        raise ValueError(msg)
    n_l, use_moe, sparse = (
        1 if single_layer else draft.num_layers,
        draft.num_experts > 0,
        geometry.sparse_attn,
    )
    draft_sparse = None
    if sparse is not None and (draft.layer_kind, sparse.kind) in (
        ("csa2", "csa2"),
        ("indexed_attention", "qsa"),
    ):
        ids = tuple(range(n_l)) if sparse.kind == "qsa" else ()
        draft_sparse = replace(
            sparse,
            kv_source_layer_ids=ids,
            index_source_layer_ids=ids,
            ced_boundary=0,
            ced_post_only=False,
        )
    idx_t = (
        ("shared",) * n_l
        if (draft.layer_kind == "dsa" and geometry.index_share_for_mtp_iteration)
        else ()
    )
    return replace(
        geometry,
        num_layers=n_l,
        layer_types=(draft.layer_kind,) * n_l,
        linear_k_heads=0,
        linear_v_heads=0,
        linear_k_dim=0,
        linear_v_dim=0,
        conv_kernel=0,
        num_experts=draft.num_experts if use_moe else 0,
        top_k=(draft.top_k or geometry.top_k) if use_moe else 0,
        moe_inter=(draft.moe_inter or geometry.moe_inter) if use_moe else 0,
        shared_inter=draft.shared_inter if use_moe else 0,
        dense_inter=0 if use_moe else (draft.dense_inter or geometry.dense_inter),
        sliding_window=draft.sliding_window or geometry.sliding_window,
        first_k_dense_replace=0,
        ffn_kinds=("moe" if use_moe else "dense",) * n_l,
        n_moe_layers=n_l if use_moe else 0,
        n_dense_layers=0 if use_moe else n_l,
        compress_ratios=(),
        indexer_types=idx_t,
        mtp_layers=0,
        ple_dim=0,
        ple_layer_ids=(),
        sparse_attn=draft_sparse,
        host_tables=(),
        draft=None,
    )


def _slice_sparse_spec(
    sparse: SparseAttnSpec | None, start: int, end: int
) -> SparseAttnSpec | None:
    """Rebase SparseAttnSpec source layer IDs and CED boundary into ``[start, end)``."""
    if sparse is None:
        return None
    post_only = sparse.ced_post_only or (0 < sparse.ced_boundary <= start)
    ced_b = (
        sparse.ced_boundary - start
        if (not post_only and start < sparse.ced_boundary < end)
        else 0
    )
    return replace(
        sparse,
        kv_source_layer_ids=tuple(
            i - start for i in sparse.kv_source_layer_ids if start <= i < end
        ),
        index_source_layer_ids=tuple(
            i - start for i in sparse.index_source_layer_ids if start <= i < end
        ),
        ced_boundary=ced_b,
        ced_post_only=post_only,
    )


def _slice_host_tables(
    tables: tuple[HostTableSpec, ...],
    start: int,
    end: int,
    num_layers: int,
) -> tuple[HostTableSpec, ...]:
    """Rebase HostTableSpec entries into ``[start, end)`` preserving residency."""
    out: list[HostTableSpec] = []
    for idx, spec in enumerate(tables):
        rebased = tuple(i - start for i in spec.layer_ids if start <= i < end)
        if spec.layer_ids and not rebased:
            continue
        frac = (
            float(len(rebased)) / float(len(spec.layer_ids))
            if spec.layer_ids
            else float(max(0, end - start)) / float(max(1, num_layers))
        )
        res_first = spec.resident_on_first
        if res_first is None:
            if idx > 0 or (spec.layer_ids and not (start <= spec.layer_ids[0] < end)):
                res_first = False
            elif spec.layer_ids and len(rebased) == 1:
                res_first = True
        out.append(
            replace(
                spec,
                layer_ids=rebased,
                elem_count=round(spec.elem_count * frac),
                h2d_bytes_per_tok=spec.h2d_bytes_per_tok * frac,
                state_bytes_per_req=spec.state_bytes_per_req * frac,
                dense_proj_params=round(spec.dense_proj_params * frac),
                resident_on_first=res_first,
            )
        )
    return tuple(out)


def slice_geometry_layers(
    geometry: ModelGeometry, start: int, end: int
) -> ModelGeometry:
    """Build a contiguous sub-geometry covering layers ``[start, end)``."""
    cnt = max(0, end - start)
    lt = geometry.layer_types[start:end] if geometry.layer_types else ()
    cr = (
        geometry.compress_ratios[start:end]
        if len(geometry.compress_ratios) >= end
        else ()
    )
    ffn_all = layer_ffn_kinds(geometry)
    ffn = ffn_all[start:end] if len(ffn_all) >= end else ("dense",) * cnt
    n_moe, n_dense = ffn.count("moe"), ffn.count("dense")
    idx = (
        geometry.indexer_types[start:end] if len(geometry.indexer_types) >= end else ()
    )
    ple = tuple(i - start for i in geometry.ple_layer_ids if start <= i < end)
    ple_d = (
        geometry.ple_dim if (not geometry.ple_layer_ids or ple or "ple" in lt) else 0
    )
    return replace(
        geometry,
        num_layers=cnt,
        layer_types=lt,
        compress_ratios=cr,
        indexer_types=idx,
        ffn_kinds=ffn,
        n_moe_layers=n_moe,
        n_dense_layers=n_dense,
        first_k_dense_replace=0 if n_moe > 0 else n_dense,
        mtp_layers=0,
        ple_layer_ids=ple,
        ple_dim=ple_d,
        sparse_attn=_slice_sparse_spec(geometry.sparse_attn, start, end),
        host_tables=_slice_host_tables(
            geometry.host_tables, start, end, geometry.num_layers
        ),
        draft=geometry.draft if end == geometry.num_layers else None,
    )


def _resolve_layer_family(
    layer_tok: str,
    kv_lora_rank: int,
    index_topk: int,
) -> tuple[str, ...]:
    """Map a single layer token to its attention/recurrent model families."""
    if layer_tok == "gqa":
        return ("gqa",)
    mapped = LAYER_TYPE_MAP.get(layer_tok)
    if mapped == "dsa":
        return ("mla", "dsa")
    if mapped is not None:
        return (mapped,)
    if kv_lora_rank > 0 and index_topk > 0:
        return ("mla", "dsa")
    if kv_lora_rank > 0:
        return ("mla",)
    return ("gqa",)


@lru_cache(maxsize=256)
def count_layer_types(geometry: ModelGeometry) -> LayerCounts:
    """Count occurrences of attention/recurrent layers and KV retention weights."""
    counts = {"gqa": 0, "swa": 0, "mla": 0, "dsa": 0, "gdn": 0, "kda": 0}
    kv_sums = {"gqa": 0.0, "swa": 0.0, "mla": 0.0, "dsa": 0.0}
    compress_ratios = geometry.compress_ratios or ()
    num_compress_ratios = len(compress_ratios)
    ignored = {"ffn_only", "mlp", "dense_mlp", "moe", "mtp", "ple"}
    for layer_idx, layer_tok in enumerate(geometry.layer_types):
        if layer_tok in ignored:
            continue
        ratio_i = compress_ratios[layer_idx] if layer_idx < num_compress_ratios else 1.0
        kv_ret = (1.0 / ratio_i) if ratio_i > 1.0 else ratio_i
        for family in _resolve_layer_family(
            layer_tok, geometry.kv_lora_rank, geometry.index_topk
        ):
            counts[family] += 1
            if family in kv_sums:
                kv_sums[family] += kv_ret
    has_compress = bool(compress_ratios)
    return LayerCounts(
        gqa=counts["gqa"],
        swa=counts["swa"],
        mla=counts["mla"],
        dsa=counts["dsa"],
        gdn=counts["gdn"],
        kda=counts["kda"],
        eff_gqa=kv_sums["gqa"] if has_compress else float(counts["gqa"]),
        eff_swa=kv_sums["swa"] if has_compress else float(counts["swa"]),
        eff_mla=kv_sums["mla"] if has_compress else float(counts["mla"]),
        eff_dsa=kv_sums["dsa"] if has_compress else float(counts["dsa"]),
    )


@lru_cache(maxsize=256)
def layer_ffn_kinds(geometry: ModelGeometry) -> tuple[str, ...]:
    """Return per-layer FFN kinds ('dense', 'moe', or 'none') for model geometry."""
    if geometry.ffn_kinds and len(geometry.ffn_kinds) == geometry.num_layers:
        return geometry.ffn_kinds
    if geometry.num_experts > 0:
        first_k = min(geometry.num_layers, geometry.first_k_dense_replace)
        return ("dense",) * first_k + ("moe",) * (geometry.num_layers - first_k)
    return ("dense",) * geometry.num_layers


def _validate_text_config(config: Mapping[str, object]) -> Mapping[str, object]:
    """Validate and extract text_config mapping from top-level config."""
    if not config:
        msg = "Empty HuggingFace config mapping"
        raise UnsupportedArchitectureError(msg)
    has_text_key = "text_config" in config
    subconfig = _as_mapping(config.get("text_config"))
    if has_text_key and not subconfig:
        msg = "Multimodal config has non-mapping text_config"
        raise UnsupportedArchitectureError(msg)
    text_cfg: Mapping[str, object] = subconfig if subconfig else config
    is_multimodal = has_text_key or any(
        multimodal_key in config for multimodal_key in MULTIMODAL_KEYS
    )
    if is_multimodal and not text_cfg.get("hidden_size"):
        msg = "Multimodal config missing extractable text_config.hidden_size"
        raise UnsupportedArchitectureError(msg)
    return text_cfg


def _has_gdn_kda_heads(text_cfg: Mapping[str, object]) -> bool:
    """Return whether text config declares linear or recurrent attention heads."""
    lin_map = _as_mapping(text_cfg.get("linear_attn_config"))
    return bool(
        text_cfg.get("linear_num_key_heads")
        or text_cfg.get("linear_num_value_heads")
        or lin_map.get("num_key_heads")
        or lin_map.get("num_value_heads")
        or lin_map.get("num_heads")
        or text_cfg.get("mamba_num_heads")
        or text_cfg.get("attn_type_list")
        or text_cfg.get("hybrid_override_pattern")
    )


def _is_csa2_config(text_config: Mapping[str, object]) -> bool:
    """Return True when config declares CSA2 sparse KV or index source layers."""
    return bool(
        _as_sequence(text_config.get("kv_source_layer_ids"))
        or _as_sequence(text_config.get("index_source_layer_ids"))
    )


def _parse_layer_id_tuple(
    raw: object,
    key_name: str,
    num_layers: int,
) -> tuple[int, ...]:
    """Parse and range-check a sequence of 0-based layer indices."""
    seq = _as_sequence(raw)
    if seq is None:
        return ()
    ids: list[int] = []
    for item in seq:
        idx = _as_int(item, -1)
        if idx < 0 or idx >= num_layers:
            msg = f"{key_name} entry {item!r} out of range [0, {num_layers})"
            raise ValueError(msg)
        ids.append(idx)
    return tuple(ids)


def _resolve_csa2_sparse_spec(
    text_config: Mapping[str, object],
    num_layers: int,
    head_dim: int,
    sliding_window: int,
    *,
    allow_csa2: bool = False,
) -> SparseAttnSpec | None:
    """Build CSA2 SparseAttnSpec when sparse KV/index source layers are declared."""
    if not _is_csa2_config(text_config):
        return None
    if not allow_csa2:
        msg = (
            "CSA2 sparse attention keys (kv_source_layer_ids / "
            "index_source_layer_ids) are only supported for CSA2 models"
        )
        raise UnsupportedArchitectureError(msg)
    kv_sources = _parse_layer_id_tuple(
        text_config.get("kv_source_layer_ids"),
        "kv_source_layer_ids",
        num_layers,
    )
    index_sources = _parse_layer_id_tuple(
        text_config.get("index_source_layer_ids"),
        "index_source_layer_ids",
        num_layers,
    )
    cand_blocks = _as_int(text_config.get("candidate_topk_blocks"))
    cand_block_size = _as_int(text_config.get("candidate_block_size"))
    candidate_scan_cap = (
        cand_blocks * cand_block_size
        if (cand_blocks > 0 and cand_block_size > 0)
        else 0
    )
    raw_ced = text_config.get("candidate_source_layer_id")
    ced_boundary = _as_int(raw_ced) if raw_ced is not None else 0
    if ced_boundary < 0 or ced_boundary >= num_layers:
        msg = f"candidate_source_layer_id {raw_ced!r} out of range [0, {num_layers})"
        raise ValueError(msg)
    o_lora_rank = _as_int(text_config.get("o_lora_rank"))
    o_groups = max(1, _as_int(text_config.get("o_groups"), 1))
    index_head_dim = _as_int(text_config.get("index_head_dim"))
    index_record_bytes = (
        float(index_head_dim) * FP8_BYTES + FP32_BYTES
        if index_head_dim > 0
        else CSA2_INDEX_RECORD_BYTES
    )
    swa_bytes = float(head_dim) * BF16_BYTES if sliding_window > 0 else 0.0
    return SparseAttnSpec(
        kind="csa2",
        kv_source_layer_ids=kv_sources,
        index_source_layer_ids=index_sources,
        candidate_scan_cap=candidate_scan_cap,
        index_record_bytes=index_record_bytes,
        ced_boundary=ced_boundary,
        swa_bytes_per_tok=swa_bytes,
        o_lora_rank=o_lora_rank,
        o_groups=o_groups,
    )


def _resolve_engram_host_tables(
    text_config: Mapping[str, object],
    num_layers: int,
    hidden: int,
    hc_mult: int,
) -> tuple[HostTableSpec, ...]:
    """Parse Engram host-offloadable embedding tables and dense projections."""
    layer_ids = _parse_layer_id_tuple(
        text_config.get("engram_layer_ids"), "engram_layer_ids", num_layers
    )
    if not layer_ids:
        return ()
    raw_counts = _as_sequence(text_config.get("engram_num_embeddings"))
    if raw_counts is None or len(raw_counts) != len(layer_ids):
        msg = (
            "engram_num_embeddings length must match engram_layer_ids "
            f"({len(layer_ids)})"
        )
        raise ValueError(msg)
    engram_head_dim = _as_int(text_config.get("engram_head_dim"))
    engram_n_heads = _as_int(text_config.get("engram_n_heads"))
    max_ngram_size = _as_int(text_config.get("engram_max_ngram_size"))
    if engram_head_dim <= 0 or engram_n_heads <= 0:
        msg = (
            "Engram config requires positive engram_head_dim and engram_n_heads, "
            f"got ({engram_head_dim}, {engram_n_heads})"
        )
        raise ValueError(msg)
    ngram_orders = max(1, max_ngram_size - 1)
    proj_width = ngram_orders * engram_n_heads * engram_head_dim
    h2d_per_layer = float(proj_width) * FP8_UE8M0_BYTES
    dense_proj_per_layer = (hc_mult + 1) * hidden * proj_width
    tables: list[HostTableSpec] = []
    for layer_id, raw_rows in zip(layer_ids, raw_counts, strict=True):
        num_rows = _as_int(raw_rows)
        if num_rows <= 0:
            msg = f"engram_num_embeddings entry must be positive, got {raw_rows!r}"
            raise ValueError(msg)
        tables.append(
            HostTableSpec(
                name=f"engram_l{layer_id}",
                layer_ids=(layer_id,),
                elem_count=num_rows * engram_head_dim,
                scale_ratio=FP8_UE8M0_BYTES - FP8_UE8M0_32X32_BYTES,
                table_dtype_role="w_dense",
                dense_proj_dtype_role="w_dense",
                default_hbm_resident=False,
                shard_axis="dp_local",
                h2d_bytes_per_tok=h2d_per_layer,
                dense_proj_params=dense_proj_per_layer,
            )
        )
    return tuple(tables)


def _resolve_inline_dspark_spec(
    text_config: Mapping[str, object],
    layer_types: tuple[str, ...],
    dims: tuple[int, int, int, int],
) -> DraftSpec | None:
    """Parse inline DSpark draft metadata declared inside main text_config."""
    mtp_layers, moe_inter, shared_inter, sliding_window = dims
    if mtp_layers <= 0:
        return None
    raw_experts = text_config.get("dspark_n_routed_experts")
    raw_block = text_config.get("dspark_block_size")
    raw_targets = _as_sequence(text_config.get("dspark_target_layer_ids"))
    raw_markov = text_config.get("dspark_markov_rank")
    if all(v is None for v in (raw_experts, raw_block, raw_targets, raw_markov)):
        return None
    draft_experts = _as_int(raw_experts)
    draft_top_k = _as_int(text_config.get("dspark_num_experts_per_tok"))
    layer_kind = (
        "csa2"
        if _is_csa2_config(text_config)
        else (layer_types[0] if layer_types else "full_attention")
    )
    return DraftSpec(
        kind="dspark",
        num_layers=mtp_layers,
        layer_kind=layer_kind,
        num_experts=draft_experts,
        top_k=draft_top_k,
        moe_inter=moe_inter if draft_experts > 0 else 0,
        shared_inter=shared_inter if draft_experts > 0 else 0,
        sliding_window=sliding_window,
        block_size=max(1, _as_int(raw_block, 1)),
        fc_in_mult=len(raw_targets) if raw_targets else 1,
        markov_rank=_as_int(raw_markov),
    )


def _validate_modeled_attention_variants(
    text_config: Mapping[str, object],
    *,
    allow_csa2: bool = False,
) -> None:
    """Fail closed on unmodeled LoRA output projections or SWA-only DSA configs."""
    if _is_csa2_config(text_config):
        if not allow_csa2:
            msg = (
                "CSA2 sparse attention keys (kv_source_layer_ids / "
                "index_source_layer_ids) are only supported for CSA2 models"
            )
            raise UnsupportedArchitectureError(msg)
        return
    if _as_int(text_config.get("o_lora_rank")) > 0:
        msg = "Non-zero o_lora_rank attention output projection is not modeled"
        raise UnsupportedArchitectureError(msg)
    sliding_window = _as_int(text_config.get("sliding_window"))
    index_topk = _as_int(text_config.get("index_topk"))
    kv_lora_rank = _as_int(text_config.get("kv_lora_rank"))
    if sliding_window > 0 and index_topk > 0 and kv_lora_rank <= 0:
        msg = (
            "Hybrid sliding_window + index_topk without kv_lora_rank > 0 is not modeled"
        )
        raise UnsupportedArchitectureError(msg)


def _resolve_attn_output_gate(
    text_config: Mapping[str, object],
    config: Mapping[str, object],
) -> bool:
    """Resolve whether attention output gating is enabled in text or root config."""
    scopes = (text_config, config) if text_config is not config else (text_config,)
    raw_flag = _first_non_none(scopes, "attn_output_gate", "attention_output_gate")
    raw_gate_type = _first_non_none(scopes, "output_gate_type")
    if raw_flag is not None and bool(raw_flag):
        return True
    if isinstance(raw_gate_type, str):
        return raw_gate_type.strip().lower() not in ("", "none", "false")
    return False


def _validate_layer_types(text_cfg: Mapping[str, object], *, has_gdn: bool) -> None:
    """Validate optional hybrid_override_pattern and layer_types declarations."""
    hybrid_pattern = text_cfg.get("hybrid_override_pattern")
    if isinstance(hybrid_pattern, str):
        pattern_text = hybrid_pattern.strip()
        if any(ch not in HYBRID_PATTERN_MAP for ch in pattern_text):
            msg = f"Unsupported hybrid_override_pattern tokens in {pattern_text!r}"
            raise UnsupportedArchitectureError(msg)

    raw_types = _as_sequence(text_cfg.get("layer_types"))
    if raw_types is None:
        return
    for layer_tok in raw_types:
        layer_kind = str(layer_tok)
        if layer_kind not in VALID_LAYER_TOKENS:
            msg = (
                f"Unknown layer_type {layer_kind!r}; supported tokens: "
                f"{sorted(VALID_LAYER_TOKENS)}"
            )
            raise UnsupportedArchitectureError(msg)
        if layer_kind == "mamba" and not has_gdn:
            msg = "Layer token 'mamba' requires GDN/KDA head dimensions"
            raise UnsupportedArchitectureError(msg)


def _supported_repos_summary() -> str:
    """Format canonical supported HF repos for actionable fail-closed errors."""
    repos = [repo for spec in SUPPORTED_MODELS.values() for repo in spec.hf_repos]
    return ", ".join(repos)


def _normalize_hf_repo_ref(model_ref: str | None) -> str | None:
    """Extract normalized ``org/repo`` when ``model_ref`` looks like an HF repo."""
    if not model_ref:
        return None
    cleaned = model_ref.strip().rstrip("/")
    for prefix in ("https://huggingface.co/", "http://huggingface.co/"):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    parts = [part for part in cleaned.split("/") if part]
    min_repo_parts = 2
    if len(parts) == min_repo_parts and not cleaned.startswith((".", "/", "~")):
        return f"{parts[0]}/{parts[1]}"
    return None


def _resolve_architectures(
    config: Mapping[str, object],
    text_cfg: Mapping[str, object],
) -> tuple[str, ...]:
    """Read and normalize HuggingFace ``architectures`` across root and text_config."""
    scopes = (config, text_cfg) if text_cfg is not config else (config,)
    found: list[str] = []
    for scope in scopes:
        if "architectures" in scope:
            raw = _as_sequence(scope["architectures"])
            if raw is not None:
                found.extend(str(item) for item in raw if str(item))
    return tuple(dict.fromkeys(found))


def _resolve_ple_layer_ids(
    text_config: Mapping[str, object],
    config: Mapping[str, object],
    num_layers: int,
) -> tuple[int, ...]:
    """Read 1-based HF ``ple_layer_ids`` and convert to 0-based layer indices."""
    scopes = (text_config, config) if text_config is not config else (text_config,)
    raw = _first_non_none(scopes, "ple_layer_ids")
    seq = _as_sequence(raw)
    if seq is None:
        return ()
    zero_based: list[int] = []
    for item in seq:
        one_based = _as_int(item, -1)
        if one_based < 1 or one_based > num_layers:
            msg = f"ple_layer_ids entry {item!r} out of 1-based range [1, {num_layers}]"
            raise ValueError(msg)
        zero_based.append(one_based - 1)
    return tuple(zero_based)


def _resolve_draft_kv_flags(
    all_scopes: tuple[Mapping[str, object], ...],
    draft_archs: tuple[str, ...],
) -> tuple[str, bool, bool]:
    """Resolve (draft_kv_dtype, skip_draft_kv_on_offload, single_row_lm_head_verify)."""
    is_minimax_draft = "DSparkMiniMaxDraftModel" in draft_archs
    raw_draft_kv = _first_non_none(all_scopes, "draft_kv_dtype")
    draft_kv_dtype = (
        str(raw_draft_kv)
        if raw_draft_kv is not None
        else ("fp8" if is_minimax_draft else "")
    )
    raw_skip_offload = _first_non_none(all_scopes, "skip_draft_kv_on_offload")
    skip_offload = (
        bool(raw_skip_offload) if raw_skip_offload is not None else is_minimax_draft
    )
    raw_single_row = _first_non_none(all_scopes, "single_row_lm_head_verify")
    single_row = (
        bool(raw_single_row) if raw_single_row is not None else is_minimax_draft
    )
    return draft_kv_dtype, skip_offload, single_row


def _resolve_draft_spec(draft_config: Mapping[str, object] | None) -> DraftSpec | None:
    """Parse an optional companion draft model config JSON into a DraftSpec."""
    if draft_config is None:
        return None
    if not draft_config:
        msg = "Draft config requires positive 'num_hidden_layers'"
        raise ValueError(msg)
    draft_text = _as_mapping(draft_config.get("text_config")) or draft_config
    dflash_cfg = _as_mapping(
        _first_non_none((draft_text, draft_config), "dflash_config")
    )
    dspark_cfg = _as_mapping(
        _first_non_none((draft_text, draft_config), "dspark_config")
    )
    all_scopes = (draft_text, draft_config, dspark_cfg)
    top_scopes = (draft_text, dflash_cfg, draft_config)

    def _get_int(scopes: tuple[Mapping[str, object], ...], *keys: str) -> int:
        """Read first integer value across config scopes."""
        return _as_int(_first_non_none(scopes, *keys) or 0)

    num_layers = _get_int((dflash_cfg, draft_text, draft_config), "num_hidden_layers")
    if num_layers <= 0:
        msg = "Draft config requires positive 'num_hidden_layers'"
        raise ValueError(msg)
    targets = _as_sequence(
        _first_non_none(top_scopes, "dspark_target_layer_ids", "target_layer_ids")
    )
    use_dense = bool(
        _first_non_none(all_scopes, "dspark_use_dense_ffn", "use_dense_ffn")
    )
    raw_layer_types = _as_sequence(
        _first_non_none((draft_text, dflash_cfg, draft_config), "layer_types")
    )
    if raw_layer_types:
        layer_toks = [str(item) for item in raw_layer_types]
        for layer_tok in layer_toks:
            if layer_tok not in VALID_LAYER_TOKENS:
                msg = (
                    f"Unknown draft layer_type {layer_tok!r}; "
                    f"supported tokens: {sorted(VALID_LAYER_TOKENS)}"
                )
                raise UnsupportedArchitectureError(msg)
        if len(set(layer_toks)) > 1:
            msg = "Mixed layer_types in draft config are not modeled"
            raise UnsupportedArchitectureError(msg)
        layer_kind = layer_toks[0]
    else:
        layer_kind = "full_attention"
    ffn_keys = ("dspark_ffn_hidden_size", "ffn_hidden_size", "dense_intermediate_size")
    draft_archs = _resolve_architectures(draft_config, draft_text)
    draft_kv_dtype, skip_offload, single_row = _resolve_draft_kv_flags(
        all_scopes, draft_archs
    )
    return DraftSpec(
        kind="dspark",
        num_layers=num_layers,
        layer_kind=layer_kind,
        num_experts=0 if use_dense else _get_int(all_scopes, "dspark_n_routed_experts"),
        top_k=0 if use_dense else _get_int(all_scopes, "dspark_num_experts_per_tok"),
        dense_inter=_get_int(all_scopes, *ffn_keys) if use_dense else 0,
        sliding_window=_get_int(top_scopes, "sliding_window"),
        block_size=max(1, _get_int(top_scopes, "dspark_block_size", "block_size")),
        fc_in_mult=len(targets) if targets else 1,
        markov_rank=_get_int(all_scopes, "dspark_markov_rank", "markov_rank"),
        draft_kv_dtype=draft_kv_dtype,
        skip_draft_kv_on_offload=skip_offload,
        single_row_lm_head_verify=single_row,
    )


def _resolve_msa_sparse_attn(
    text_config: Mapping[str, object],
    num_layers: int,
) -> tuple[SparseAttnSpec | None, int, int]:
    """Parse sparse_attention_config into (SparseAttnSpec, index_heads, index_dim)."""
    sparse_cfg = _as_mapping(text_config.get("sparse_attention_config"))
    if not sparse_cfg or not bool(sparse_cfg.get("use_sparse_attention", False)):
        return None, 0, 0
    raw_freq = _as_sequence(sparse_cfg.get("sparse_attention_freq"))
    if raw_freq is not None:
        if len(raw_freq) != num_layers:
            msg = (
                f"sparse_attention_freq length {len(raw_freq)} "
                f"!= num_hidden_layers {num_layers}"
            )
            raise ValueError(msg)
        sparse_ids = tuple(
            idx for idx, flag in enumerate(raw_freq) if _as_int(flag) != 0
        )
    else:
        sparse_ids = tuple(range(num_layers))
    index_heads = _as_int(sparse_cfg.get("sparse_num_index_heads"))
    index_dim = _as_int(sparse_cfg.get("sparse_index_dim"))
    block_size = _as_int(sparse_cfg.get("sparse_block_size"))
    topk_blocks = _as_int(sparse_cfg.get("sparse_topk_blocks"))
    for key, val in (
        ("sparse_block_size", block_size),
        ("sparse_topk_blocks", topk_blocks),
        ("sparse_num_index_heads", index_heads),
        ("sparse_index_dim", index_dim),
    ):
        if val <= 0:
            msg = f"MSA sparse_attention_config requires positive '{key}'"
            raise UnsupportedArchitectureError(msg)
    local_block = _as_int(sparse_cfg.get("sparse_local_block")) + _as_int(
        sparse_cfg.get("sparse_init_block")
    )
    spec = SparseAttnSpec(
        kind="msa",
        kv_source_layer_ids=sparse_ids,
        index_source_layer_ids=sparse_ids,
        sparse_block_size=block_size,
        sparse_topk_blocks=topk_blocks,
        sparse_local_block=local_block,
        dcp_span="tp",
        index_cache_sharded=False,
    )
    return spec, index_heads, index_dim


def _validate_listed_model_spec(
    spec: ModelSpec,
    model_type: str,
    architectures: tuple[str, ...],
    hf_repo: str | None,
    repo_canonical: str | None,
) -> None:
    """Validate 1:1 repo mapping and causal target architectures for a spec."""
    if hf_repo is not None and repo_canonical is None:
        msg = (
            f"Unlisted HuggingFace repo {hf_repo!r}; fails closed. "
            f"Supported models: {_supported_repos_summary()}"
        )
        raise UnsupportedArchitectureError(msg)
    if repo_canonical is not None and repo_canonical != spec.canonical_id:
        msg = (
            f"HF repo {hf_repo!r} maps to canonical model {repo_canonical!r}, "
            f"which does not match config model_type {model_type!r} "
            f"({spec.canonical_id!r})"
        )
        raise UnsupportedArchitectureError(msg)

    if architectures and not any(arch in spec.architectures for arch in architectures):
        msg = (
            f"Unsupported architectures {list(architectures)!r} for canonical model "
            f"{spec.canonical_id!r}; expected one of {list(spec.architectures)!r}"
        )
        raise UnsupportedArchitectureError(msg)


def resolve_model_identity(
    config: Mapping[str, object],
    *,
    model_ref: str | None = None,
) -> ModelIdentity:
    """Resolve and validate canonical model identity from HF config and repo ref."""
    text_cfg = _validate_text_config(config)
    scopes = (text_cfg, config) if text_cfg is not config else (text_cfg,)
    is_causal = _first_non_none(scopes, "is_causal")
    model_type = str(_first_non_none(scopes, "model_type") or "").strip().lower()
    architectures = _resolve_architectures(config, text_cfg)
    hf_repo = _normalize_hf_repo_ref(model_ref)
    repo_canonical = (
        HF_REPO_TO_CANONICAL_ID.get(hf_repo.lower()) if hf_repo is not None else None
    )

    if is_causal is False:
        msg = "Non-causal architecture (is_causal=False) is not supported"
        raise UnsupportedArchitectureError(msg)
    if not model_type:
        msg = (
            "HuggingFace config missing required 'model_type'; fails closed. "
            f"Supported models: {_supported_repos_summary()}"
        )
        raise UnsupportedArchitectureError(msg)

    spec = _MODEL_TYPE_TO_SPEC.get(model_type)
    if spec is None:
        msg = (
            f"Unrecognized or unsupported model_type {model_type!r}; fails closed. "
            f"Supported models: {_supported_repos_summary()}"
        )
        raise UnsupportedArchitectureError(msg)

    _validate_listed_model_spec(
        spec, model_type, architectures, hf_repo, repo_canonical
    )
    return ModelIdentity(
        canonical_id=spec.canonical_id,
        model_type=model_type,
        architectures=architectures,
        hf_repo=hf_repo,
    )


def _validate_structural_support(
    config: Mapping[str, object],
    *,
    allow_csa2: bool | None = None,
) -> None:
    """Validate attention, recurrence, and layer-type structural support."""
    text_cfg = _validate_text_config(config)
    has_gdn = _has_gdn_kda_heads(text_cfg)
    if ("ssm_cfg" in text_cfg or "state_size" in text_cfg) and not (
        text_cfg.get("num_attention_heads") or has_gdn
    ):
        msg = "Unmodeled state-space config missing attention and GDN/KDA head keys"
        raise UnsupportedArchitectureError(msg)
    if allow_csa2 is None:
        scopes = (text_cfg, config) if text_cfg is not config else (text_cfg,)
        model_type = str(_first_non_none(scopes, "model_type") or "").strip().lower()
        spec = _MODEL_TYPE_TO_SPEC.get(model_type)
        eff_allow_csa2 = spec is not None and "csa2" in spec.allowed_layer_kinds
    else:
        eff_allow_csa2 = allow_csa2
    _validate_modeled_attention_variants(text_cfg, allow_csa2=eff_allow_csa2)
    _validate_layer_types(text_cfg, has_gdn=has_gdn)


def _validate_family_layer_kinds(
    identity: ModelIdentity,
    layer_types: tuple[str, ...],
    kv_lora_rank: int,
    index_topk: int,
) -> None:
    """Reject layer families outside the canonical family structural allowlist."""
    spec = SUPPORTED_MODELS[identity.canonical_id]
    ignored = {"ffn_only", "mlp", "dense_mlp", "moe", "mtp", "ple"}
    canonical_kinds = {
        fam
        for tok in layer_types
        if tok not in ignored
        for fam in _resolve_layer_family(tok, kv_lora_rank, index_topk)
    }
    disallowed = sorted(canonical_kinds - set(spec.allowed_layer_kinds))
    if disallowed:
        msg = (
            f"Canonical model {spec.canonical_id!r} does not allow layer kinds "
            f"{disallowed!r}; expected subset of "
            f"{sorted(spec.allowed_layer_kinds)!r}"
        )
        raise UnsupportedArchitectureError(msg)


def supports(
    config: Mapping[str, object],
    *,
    model_ref: str | None = None,
) -> None:
    """Validate canonical identity and modeled attention/recurrence/FFN families."""
    identity = resolve_model_identity(config, model_ref=model_ref)
    allow_csa2 = "csa2" in SUPPORTED_MODELS[identity.canonical_id].allowed_layer_kinds
    _validate_structural_support(config, allow_csa2=allow_csa2)
    text_cfg = _validate_text_config(config)
    raw_layers = text_cfg.get("num_hidden_layers")
    if raw_layers is not None:
        num_layers = _as_int(raw_layers)
        if num_layers > 0:
            kv_lora_rank = _as_int(text_cfg.get("kv_lora_rank"))
            index_topk = _as_int(text_cfg.get("index_topk"))
            sliding_window = _as_int(text_cfg.get("sliding_window"))
            specs = (kv_lora_rank, index_topk, sliding_window)
            layer_types = _resolve_layer_types(
                text_cfg, config, num_layers, specs, allow_csa2=allow_csa2
            )
            _validate_family_layer_kinds(
                identity, layer_types, kv_lora_rank, index_topk
            )


def _is_kda_config(text_config: Mapping[str, object]) -> bool:
    """Return True when config declares KDA-style linear attention keys."""
    lin_cfg = _as_mapping(text_config.get("linear_attn_config"))
    if (
        lin_cfg.get("kda_layers") is not None
        or lin_cfg.get("short_conv_kernel_size")
        or lin_cfg.get("kda")
    ):
        return True
    raw_types = _as_sequence(text_config.get("layer_types")) or ()
    return any(str(tok) in ("kda", "kda_attention") for tok in raw_types)


def _parse_dsa_layer_set(
    text_config: Mapping[str, object],
    config: Mapping[str, object],
    num_layers: int,
) -> set[int] | None:
    """Return 0-based DSA layer indices when dsa_layers/indexer_layers is set."""
    scopes = (text_config, config) if text_config is not config else (text_config,)
    raw_dsa = _first_non_none(scopes, "dsa_layers", "indexer_layers")
    if raw_dsa is None:
        return None
    seq = _as_sequence(raw_dsa)
    if seq is None:
        msg = f"Expected sequence for dsa_layers/indexer_layers, got {raw_dsa!r}"
        raise ValueError(msg)
    if seq and all(isinstance(item, bool) for item in seq):
        if len(seq) != num_layers:
            msg = (
                f"Boolean dsa_layers length {len(seq)} "
                f"!= num_hidden_layers {num_layers}"
            )
            raise ValueError(msg)
        return {idx for idx, flag in enumerate(seq) if flag}
    indices: set[int] = set()
    for item in seq:
        idx = item if isinstance(item, int) and not isinstance(item, bool) else None
        if idx is None:
            msg = (
                "dsa_layers/indexer_layers must contain integer indices or a "
                f"boolean mask of length {num_layers}, got {item!r}"
            )
            raise ValueError(msg)
        if idx < 0 or idx >= num_layers:
            msg = f"dsa_layers index {idx} out of range [0, {num_layers})"
            raise ValueError(msg)
        indices.add(idx)
    return indices


def _infer_pattern_layer_types(
    text_config: Mapping[str, object],
    num_layers: int,
    specs: tuple[int, int, int],
    *,
    allow_csa2: bool = False,
) -> tuple[str, ...]:
    """Infer layer types from interval, attention list, or override pattern."""
    kv_lora_rank, index_topk, sliding_window = specs
    lin_cfg = _as_mapping(text_config.get("linear_attn_config"))
    kda_layers = _as_sequence(lin_cfg.get("kda_layers"))
    if kda_layers is not None:
        kda_set = {_as_int(x) for x in kda_layers}
        att_tok = (
            "dsa"
            if (kv_lora_rank > 0 and index_topk > 0)
            else ("mla" if kv_lora_rank > 0 else "full_attention")
        )
        return tuple(
            "kda" if (i + 1) in kda_set else att_tok for i in range(num_layers)
        )
    raw_interval = text_config.get("full_attention_interval")
    interval = _as_int(raw_interval, 0)
    if raw_interval is not None and interval <= 0:
        msg = f"full_attention_interval must be positive, got {raw_interval!r}"
        raise ValueError(msg)
    if interval > 0:
        att_tok = (
            "dsa"
            if (kv_lora_rank > 0 and index_topk > 0)
            else ("mla" if kv_lora_rank > 0 else "full_attention")
        )
        lin_tok = "kda" if _is_kda_config(text_config) else "linear_attention"
        return tuple(
            att_tok if (i + 1) % interval == 0 else lin_tok for i in range(num_layers)
        )
    attn_list = _as_sequence(text_config.get("attn_type_list"))
    if attn_list is not None and len(attn_list) == num_layers:
        return tuple(
            "full_attention" if _as_int(v, 0) == 1 else "linear_attention"
            for v in attn_list
        )
    hybrid_pattern = text_config.get("hybrid_override_pattern")
    if isinstance(hybrid_pattern, str) and len(hybrid_pattern.strip()) == num_layers:
        pattern_text = hybrid_pattern.strip()
        if any(ch not in HYBRID_PATTERN_MAP for ch in pattern_text):
            msg = f"Unsupported hybrid_override_pattern tokens in {pattern_text!r}"
            raise UnsupportedArchitectureError(msg)
        return tuple(HYBRID_PATTERN_MAP[ch] for ch in pattern_text)
    sw_pat = _as_int(
        text_config.get("sliding_window_pattern")
        or text_config.get("_sliding_window_pattern"),
        0,
    )
    use_sw = text_config.get("use_sliding_window")
    eff_sliding_window = 0 if use_sw is False else sliding_window
    if allow_csa2 and _is_csa2_config(text_config):
        default_layers = ("csa2",) * num_layers
    elif eff_sliding_window > 0:
        if sw_pat < 1:
            msg = (
                "Config specifies sliding_window without explicit layer_types "
                "or sliding_window_pattern"
            )
            raise ValueError(msg)
        default_layers = tuple(
            "full_attention" if (i + 1) % sw_pat == 0 else "sliding_attention"
            for i in range(num_layers)
        )
    elif kv_lora_rank > 0:
        default_mla = "dsa" if index_topk > 0 else "mla"
        default_layers = (default_mla,) * num_layers
    else:
        sparse_cfg = _as_mapping(text_config.get("sparse_attention_config"))
        if bool(sparse_cfg.get("use_sparse_attention", False)):
            raw_freq = _as_sequence(sparse_cfg.get("sparse_attention_freq"))
            if raw_freq is not None:
                if len(raw_freq) != num_layers:
                    msg = (
                        f"sparse_attention_freq length {len(raw_freq)} "
                        f"!= num_hidden_layers {num_layers}"
                    )
                    raise ValueError(msg)
                default_layers = tuple(
                    "sparse_gqa" if _as_int(flag) != 0 else "full_attention"
                    for flag in raw_freq
                )
            else:
                default_layers = ("sparse_gqa",) * num_layers
        else:
            default_layers = ("full_attention",) * num_layers
    return default_layers


_MLA_OR_FULL_ATTN_TOKENS: frozenset[str] = frozenset(
    {tok for tok, fam in LAYER_TYPE_MAP.items() if fam in ("mla", "dsa")}
    | {"full_attention", "attention"}
)


def _validate_full_attention_interval(
    raw_interval: object,
    layer_types: tuple[str, ...],
) -> None:
    """Cross-check full_attention_interval against explicit layer_types."""
    if raw_interval is None:
        return
    interval = _as_int(raw_interval, 0)
    if interval <= 0:
        msg = f"full_attention_interval must be positive, got {raw_interval!r}"
        raise ValueError(msg)
    full_tokens = _MLA_OR_FULL_ATTN_TOKENS | {"indexed_attention"}
    mismatches = [
        idx
        for idx, tok in enumerate(layer_types)
        if ((idx + 1) % interval == 0) != (tok in full_tokens)
    ]
    if mismatches:
        msg = (
            f"full_attention_interval={interval} conflicts with explicit "
            f"layer_types at indices {mismatches[:5]}"
        )
        raise ValueError(msg)


def _resolve_layer_types(
    text_config: Mapping[str, object],
    config: Mapping[str, object],
    num_layers: int,
    specs: tuple[int, int, int],
    *,
    allow_csa2: bool = False,
) -> tuple[str, ...]:
    """Resolve sequence of layer type strings for each transformer layer."""
    kv_lora_rank, _, _ = specs
    dsa_set = _parse_dsa_layer_set(text_config, config, num_layers)
    raw_interval = text_config.get("full_attention_interval")
    raw_types = _as_sequence(text_config.get("layer_types"))
    if raw_types is not None:
        if len(raw_types) != num_layers:
            msg = (
                f"layer_types length {len(raw_types)} != num_hidden_layers {num_layers}"
            )
            raise ValueError(msg)
        layer_types = tuple(str(x) for x in raw_types)
        for token in layer_types:
            if token not in VALID_LAYER_TOKENS:
                supported = sorted(VALID_LAYER_TOKENS)
                msg = f"Unknown layer_type {token!r}; supported tokens: {supported}"
                raise ValueError(msg)
        _validate_full_attention_interval(raw_interval, layer_types)
    else:
        layer_types = _infer_pattern_layer_types(
            text_config, num_layers, specs, allow_csa2=allow_csa2
        )
    if dsa_set is not None and kv_lora_rank > 0:
        return tuple(
            ("dsa" if idx in dsa_set else "mla")
            if layer_tok in _MLA_OR_FULL_ATTN_TOKENS
            else layer_tok
            for idx, layer_tok in enumerate(layer_types)
        )
    return layer_types


# Config keys that name the routed experts per token, in lookup order. A sampling
# `top_k` in the same config is not one of them.
_ROUTED_EXPERTS_PER_TOKEN_KEYS: tuple[str, ...] = (
    "num_experts_per_tok",
    "num_experts_per_token",
    "experts_per_token",
    "num_selected_experts",
)


def _resolve_dense_and_moe_inter(
    text_config: Mapping[str, object],
    num_experts: int,
) -> tuple[int, int]:
    """Resolve (dense_inter, moe_inter), supporting separate dense and routed keys."""
    raw_dense = _as_int(text_config.get("dense_intermediate_size"))
    raw_inter = _as_int(text_config.get("intermediate_size"))
    dense_inter = raw_dense if raw_dense > 0 else raw_inter
    moe_inter = _as_int(
        _first_present(text_config, "moe_intermediate_size", "latent_moe_dim")
    )
    if num_experts > 0 and moe_inter == 0:
        moe_inter = raw_inter if raw_inter > 0 else raw_dense
    return dense_inter, moe_inter


def _resolve_moe_dimensions(
    text_config: Mapping[str, object],
) -> tuple[int, int, int, int, int]:
    """Resolve (num_experts, top_k, moe_inter, num_shared_experts, shared_inter)."""
    num_experts = _as_int(
        _first_present(
            text_config, "num_experts", "n_routed_experts", "num_local_experts"
        )
    )
    _, moe_inter = _resolve_dense_and_moe_inter(text_config, num_experts)

    num_shared = _as_int(
        _first_present(text_config, "n_shared_experts", "num_shared_experts")
    )
    shared_inter = _as_int(
        _first_present(
            text_config,
            "shared_expert_intermediate_size",
            "shared_intermediate_size",
        )
    )
    if shared_inter == 0 and num_shared > 0 and moe_inter > 0:
        shared_inter = num_shared * moe_inter

    top_k = _as_int(_first_present(text_config, *_ROUTED_EXPERTS_PER_TOKEN_KEYS))
    if num_experts > 0 and top_k <= 0:
        accepted_keys = ", ".join(_ROUTED_EXPERTS_PER_TOKEN_KEYS)
        msg = f"MoE config names no routed experts per token ({accepted_keys})"
        raise ValueError(msg)
    return num_experts, top_k, moe_inter, num_shared, shared_inter


def _parse_moe_layer_freq_mask(
    raw_freq: Sequence[object],
    num_layers: int,
) -> tuple[str, ...]:
    """Validate and convert a per-layer 0/1 moe_layer_freq list to FFN kinds."""
    if len(raw_freq) != num_layers:
        msg = f"moe_layer_freq length {len(raw_freq)} != num_hidden_layers {num_layers}"
        raise ValueError(msg)
    resolved: list[str] = []
    for item in raw_freq:
        val = 1 if item is True else (0 if item is False else _as_int(item, -1))
        if val not in (0, 1):
            msg = f"moe_layer_freq entries must be 0 or 1, got {item!r}"
            raise ValueError(msg)
        resolved.append("moe" if val == 1 else "dense")
    return tuple(resolved)


def _parse_mlp_layer_types(
    raw_mlp_layer_types: Sequence[object],
    num_layers: int,
) -> tuple[str, ...]:
    """Validate and normalize ``mlp_layer_types`` tokens to 'dense' or 'moe'."""
    if len(raw_mlp_layer_types) != num_layers:
        msg = (
            f"mlp_layer_types length {len(raw_mlp_layer_types)} "
            f"!= num_hidden_layers {num_layers}"
        )
        raise ValueError(msg)
    resolved: list[str] = []
    for item in raw_mlp_layer_types:
        token = str(item).strip().lower()
        mapped = _MLP_LAYER_TYPE_MAP.get(token)
        if mapped is None:
            msg = f"Unknown mlp_layer_types token {token!r}"
            raise ValueError(msg)
        resolved.append(mapped)
    return tuple(resolved)


def _resolve_ffn_kinds(
    text_config: Mapping[str, object],
    config: Mapping[str, object],
    num_layers: int,
    num_experts: int,
) -> tuple[str, ...]:
    """Resolve (moe, dense, none) kind for each layer's feed-forward network."""
    scopes = (text_config, config) if text_config is not config else (text_config,)
    # Read structural FFN metadata upfront so KeyTracker records explicit handling.
    raw_mlp_layer_types = _as_sequence(_first_non_none(scopes, "mlp_layer_types"))
    raw_mlp_only_layers = _as_sequence(_first_non_none(scopes, "mlp_only_layers"))
    raw_moe_layer_freq = _first_non_none(scopes, "moe_layer_freq")
    moe_freq_seq = _as_sequence(raw_moe_layer_freq)
    moe_layer_freq = 0 if moe_freq_seq is not None else _as_int(raw_moe_layer_freq or 0)
    first_k_dense = _as_int(_first_non_none(scopes, "first_k_dense_replace") or 0)
    hybrid_override = str(text_config.get("hybrid_override_pattern") or "").strip()
    raw_moe_layers = _as_sequence(text_config.get("moe_layers"))
    interleave_step = _as_int(text_config.get("interleave_moe_layer_step"), 0)

    if raw_mlp_layer_types is not None:
        return _parse_mlp_layer_types(raw_mlp_layer_types, num_layers)
    if moe_freq_seq is not None:
        return _parse_moe_layer_freq_mask(moe_freq_seq, num_layers)
    if hybrid_override and len(hybrid_override) == num_layers:
        return tuple(
            "moe" if ch == "E" else ("dense" if ch == "-" else "none")
            for ch in hybrid_override
        )
    if raw_moe_layers is not None:
        moe_set = {_as_int(x) for x in raw_moe_layers}
        return tuple("moe" if i in moe_set else "dense" for i in range(num_layers))
    if num_experts <= 0:
        return ("dense",) * num_layers
    if raw_mlp_only_layers:
        dense_set = {_as_int(x) for x in raw_mlp_only_layers}
        return tuple(
            "dense" if (i in dense_set or i < first_k_dense) else "moe"
            for i in range(num_layers)
        )
    if interleave_step > 1:
        return tuple(
            "moe" if (i + 1) % interleave_step == 0 else "dense"
            for i in range(num_layers)
        )
    if moe_layer_freq > 1:
        return tuple(
            "moe" if (i >= first_k_dense and i % moe_layer_freq == 0) else "dense"
            for i in range(num_layers)
        )
    return tuple("dense" if i < first_k_dense else "moe" for i in range(num_layers))


def _resolve_linear_attention_dimensions(
    text_config: Mapping[str, object],
    layer_types: tuple[str, ...],
    attn_dims: tuple[int, int, int],
) -> tuple[int, int, int, int, int]:
    """Resolve (lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, conv_kernel)."""
    num_heads, num_kv_heads, head_dim = attn_dims
    lin_cfg = _as_mapping(text_config.get("linear_attn_config"))
    has_attn_list = text_config.get("attn_type_list") is not None
    lin_k_heads = _as_int(
        text_config.get("linear_num_key_heads")
        or lin_cfg.get("num_key_heads")
        or lin_cfg.get("num_heads")
        or text_config.get("mamba_num_heads")
        or (num_kv_heads if has_attn_list else 0)
    )
    lin_v_heads = _as_int(
        text_config.get("linear_num_value_heads")
        or lin_cfg.get("num_value_heads")
        or lin_cfg.get("num_heads")
        or text_config.get("mamba_num_heads")
        or (num_heads if has_attn_list else 0)
    )
    lin_k_dim = _as_int(
        text_config.get("linear_key_head_dim")
        or lin_cfg.get("key_head_dim")
        or lin_cfg.get("head_dim")
        or text_config.get("mamba_head_dim")
        or (head_dim if has_attn_list else 0)
    )
    lin_v_dim = _as_int(
        text_config.get("linear_value_head_dim")
        or lin_cfg.get("value_head_dim")
        or lin_cfg.get("head_dim")
        or text_config.get("mamba_head_dim")
        or (head_dim if has_attn_list else 0)
    )

    has_linear = any(t in LINEAR_ATTENTION_LAYER_TOKENS for t in layer_types)
    raw_conv = (
        text_config.get("linear_conv_kernel_dim")
        or lin_cfg.get("short_conv_kernel_size")
        or lin_cfg.get("conv_kernel_size")
        or lin_cfg.get("linear_conv_kernel_dim")
        or text_config.get("conv_kernel")
        or text_config.get("d_conv")
    )
    if has_linear:
        if raw_conv is None:
            if has_attn_list:
                return lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, 0
            msg = (
                "Model with linear-attention layers missing required field: "
                "'linear_conv_kernel_dim'"
            )
            raise ValueError(msg)
        conv_kernel = _as_int(raw_conv)
    else:
        conv_kernel = 0
    return lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, conv_kernel


def _resolve_mtp_layers(
    text_config: Mapping[str, object], config: Mapping[str, object]
) -> int:
    """Resolve number of Multi-Token Prediction (MTP) layers."""
    mtp_raw = text_config.get("mtp_num_hidden_layers")
    if mtp_raw is None:
        mtp_raw = config.get("mtp_num_hidden_layers")
    if mtp_raw is None:
        mtp_raw = text_config.get(
            "num_nextn_predict_layers", config.get("num_nextn_predict_layers", 0)
        )
    return _as_int(mtp_raw)


def _resolve_compress_ratios(
    text_config: Mapping[str, object],
    config: Mapping[str, object],
    num_layers: int,
    mtp_layers: int = 0,
    *,
    allow_zero: bool = False,
) -> tuple[float, ...]:
    """Resolve per-layer KV compression or retention ratios."""
    raw_compress_ratios = _as_sequence(
        text_config.get("compress_ratios", config.get("compress_ratios"))
    )
    if raw_compress_ratios is None or not raw_compress_ratios:
        return ()
    if mtp_layers > 0 and len(raw_compress_ratios) == num_layers + mtp_layers:
        raw_compress_ratios = raw_compress_ratios[:num_layers]
    if len(raw_compress_ratios) != num_layers:
        msg = (
            f"compress_ratios length {len(raw_compress_ratios)} != "
            f"num_hidden_layers {num_layers}"
        )
        raise ValueError(msg)
    compress_ratios = tuple(_as_float(item) for item in raw_compress_ratios)
    if allow_zero:
        if any(ratio < 0.0 for ratio in compress_ratios):
            msg = f"compress_ratios entries must be >= 0, got {compress_ratios}"
            raise ValueError(msg)
    elif any(ratio <= 0.0 for ratio in compress_ratios):
        msg = f"compress_ratios entries must be > 0, got {compress_ratios}"
        raise ValueError(msg)
    return compress_ratios


def _resolve_block_parallel(
    text_config: Mapping[str, object], config: Mapping[str, object]
) -> int:
    """Resolve block parallelism factor (1 or 2)."""
    raw_block_parallel = text_config.get("block_parallel", config.get("block_parallel"))
    if raw_block_parallel is None:
        raw_block_parallel = text_config.get(
            "parallel_attn_mlp", config.get("parallel_attn_mlp", 1)
        )
    if isinstance(raw_block_parallel, bool):
        return BLOCK_PARALLEL_ENABLED if raw_block_parallel else 1
    return max(1, _as_int(raw_block_parallel, 1))


def _resolve_ffn_num_matrices(
    text_config: Mapping[str, object],
    config: Mapping[str, object],
) -> int:
    """Resolve per-layer FFN matrix multiplier (2 non-gated, 3 gated)."""
    scopes = (text_config, config) if text_config is not config else (text_config,)
    raw_num_matrices = _first_non_none(scopes, "ffn_num_matrices")
    if raw_num_matrices is not None:
        num_matrices = _as_int(raw_num_matrices)
        if num_matrices not in (NON_GATED_MATMUL_COUNT, GLU_MATMUL_COUNT):
            msg = f"Unsupported ffn_num_matrices={num_matrices!r}; expected 2 or 3"
            raise ValueError(msg)
        return num_matrices
    raw_gated = _first_non_none(scopes, "is_gated_mlp")
    if raw_gated is not None:
        return GLU_MATMUL_COUNT if bool(raw_gated) else NON_GATED_MATMUL_COUNT
    return GLU_MATMUL_COUNT


def _resolve_indexer_types(
    text_config: Mapping[str, object],
    config: Mapping[str, object],
    num_layers: int,
) -> tuple[str, ...]:
    """Resolve optional per-layer IndexShare indexer_types ('full' or 'shared')."""
    scopes = (text_config, config) if text_config is not config else (text_config,)
    raw = _as_sequence(_first_non_none(scopes, "indexer_types"))
    if raw is None:
        return ()
    if len(raw) != num_layers:
        msg = f"indexer_types length {len(raw)} != num_hidden_layers {num_layers}"
        raise ValueError(msg)
    tokens = tuple(str(item).strip().lower() for item in raw)
    invalid_tokens = [token for token in tokens if token not in _VALID_INDEXER_TYPES]
    if invalid_tokens:
        msg = (
            f"Unknown indexer_types tokens {invalid_tokens!r}; "
            "expected 'full' or 'shared'"
        )
        raise ValueError(msg)
    return tokens


@dataclass(frozen=True, slots=True)
class _ExtendedGeometryFields:
    """Auxiliary architecture dimensions parsed alongside core attention/FFN sizes."""

    mtp_layers: int
    ple_layer_ids: tuple[int, ...]
    ple_dim: int
    indexer_types: tuple[str, ...]
    index_share_for_mtp_iteration: bool
    qsa_budget: int
    hc_lowrank: int
    first_k: int
    compress_ratios: tuple[float, ...]
    block_parallel: int
    mtp_inter: int
    hc_mult: int
    tie_word_embeddings: bool
    vocab: int
    weight_block_size: tuple[int, int]
    max_position_embeddings: int


def _extract_extended_fields(
    text_config: Mapping[str, object],
    config: Mapping[str, object],
    num_layers: int,
    *,
    allow_csa2: bool = False,
) -> _ExtendedGeometryFields:
    """Parse MTP, PLE, IndexShare, HC, vocab, and block-quant dimensions."""
    scopes = (text_config, config) if text_config is not config else (text_config,)
    mtp_layers = _resolve_mtp_layers(text_config, config)
    ple_layer_ids = _resolve_ple_layer_ids(text_config, config, num_layers)
    dedicated_mtp_emb = _first_non_none(scopes, "mtp_use_dedicated_embeddings")
    if dedicated_mtp_emb is True and mtp_layers > 0:
        msg = (
            "Dedicated MTP embeddings (mtp_use_dedicated_embeddings=True) "
            "are not modeled yet"
        )
        raise UnsupportedArchitectureError(msg)
    raw_hc = (
        _first_non_none(
            scopes,
            "mhc_num_streams",
            "hc_mult",
            "hc_count",
            "hyper_connection_mult",
        )
        or 1
    )
    qcfg = _as_mapping(_first_non_none(scopes, "quantization_config"))
    raw_wbs = _as_sequence(qcfg.get("weight_block_size"))
    weight_block_size = (
        (
            max(1, _as_int(raw_wbs[0], 1)),
            max(1, _as_int(raw_wbs[1], 1)),
        )
        if raw_wbs is not None and len(raw_wbs) == WEIGHT_BLOCK_DIM_LEN
        else (1, 1)
    )
    return _ExtendedGeometryFields(
        mtp_layers=mtp_layers,
        ple_layer_ids=ple_layer_ids,
        ple_dim=_as_int(_first_present(text_config, "per_layer_input_dim", "ple_dim")),
        indexer_types=_resolve_indexer_types(text_config, config, num_layers),
        index_share_for_mtp_iteration=bool(
            _first_non_none(scopes, "index_share_for_mtp_iteration") or False
        ),
        qsa_budget=_as_int(_first_non_none(scopes, "indexer_budget") or 0),
        hc_lowrank=_as_int(_first_non_none(scopes, "hc_lowrank") or 0),
        first_k=_as_int(_first_non_none(scopes, "first_k_dense_replace") or 0),
        compress_ratios=_resolve_compress_ratios(
            text_config,
            config,
            num_layers,
            mtp_layers,
            allow_zero=allow_csa2 and _is_csa2_config(text_config),
        ),
        block_parallel=_resolve_block_parallel(text_config, config),
        mtp_inter=_as_int(_first_non_none(scopes, "mtp_intermediate_size") or 0),
        hc_mult=max(1, _as_int(raw_hc, 1)),
        tie_word_embeddings=bool(
            _first_non_none(scopes, "tie_word_embeddings") or False
        ),
        vocab=_as_int(_first_non_none(scopes, "vocab_size") or 0),
        weight_block_size=weight_block_size,
        max_position_embeddings=_as_int(
            _first_non_none(scopes, "max_position_embeddings") or 0
        ),
    )


def _extract_geometry_for_identity(
    config: Mapping[str, object],
    identity: ModelIdentity,
    *,
    draft_config: Mapping[str, object] | None = None,
) -> ModelGeometry:
    """Extract ModelGeometry from cfg after identity and structural checks."""
    subconfig = _as_mapping(config.get("text_config"))
    text_config: Mapping[str, object] = subconfig if subconfig else config
    spec = SUPPORTED_MODELS.get(identity.canonical_id)
    allow_csa2 = spec is not None and "csa2" in spec.allowed_layer_kinds
    _validate_modeled_attention_variants(text_config, allow_csa2=allow_csa2)
    hidden = _as_int(text_config["hidden_size"])
    num_layers = _as_int(text_config["num_hidden_layers"])
    num_heads = _as_int(text_config["num_attention_heads"])
    raw_kv_heads = text_config.get("num_key_value_heads")
    num_kv_heads = num_heads if raw_kv_heads is None else _as_int(raw_kv_heads)
    raw_head_dim = _as_int(text_config.get("head_dim"))
    if raw_head_dim <= 0:
        if num_heads <= 0:
            msg = (
                "num_attention_heads must be positive when head_dim is absent, "
                f"got {num_heads}"
            )
            raise ValueError(msg)
        head_dim = hidden // num_heads
    else:
        head_dim = raw_head_dim

    kv_lora_rank = _as_int(text_config.get("kv_lora_rank"))
    q_lora_rank = _as_int(text_config.get("q_lora_rank"))
    qk_nope_head_dim = _as_int(text_config.get("qk_nope_head_dim"))
    qk_rope_head_dim = _as_int(text_config.get("qk_rope_head_dim"))
    v_head_dim = _as_int(text_config.get("v_head_dim"))
    index_n_heads = _as_int(text_config.get("index_n_heads"))
    index_head_dim = _as_int(text_config.get("index_head_dim"))
    index_topk = _as_int(text_config.get("index_topk"))
    sliding_window = _as_int(text_config.get("sliding_window"))
    global_head_dim = _as_int(text_config.get("global_head_dim"))

    layer_types = _resolve_layer_types(
        text_config,
        config,
        num_layers,
        (kv_lora_rank, index_topk, sliding_window),
        allow_csa2=allow_csa2,
    )
    msa_sparse, sparse_idx_heads, sparse_idx_dim = _resolve_msa_sparse_attn(
        text_config, num_layers
    )
    if index_n_heads <= 0:
        index_n_heads = sparse_idx_heads
    if index_head_dim <= 0:
        index_head_dim = sparse_idx_dim
    num_experts, top_k, moe_inter, _num_shared, shared_inter = _resolve_moe_dimensions(
        text_config
    )
    dense_inter, _ = _resolve_dense_and_moe_inter(text_config, num_experts)
    ffn_kinds = _resolve_ffn_kinds(text_config, config, num_layers, num_experts)
    ffn_num_matrices = _resolve_ffn_num_matrices(text_config, config)
    lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, conv_kernel = (
        _resolve_linear_attention_dimensions(
            text_config, layer_types, (num_heads, num_kv_heads, head_dim)
        )
    )
    extended = _extract_extended_fields(
        text_config, config, num_layers, allow_csa2=allow_csa2
    )
    sparse_attn = (
        _resolve_csa2_sparse_spec(
            text_config, num_layers, head_dim, sliding_window, allow_csa2=allow_csa2
        )
        or msa_sparse
    )
    host_tables = _resolve_engram_host_tables(
        text_config, num_layers, hidden, extended.hc_mult
    )
    draft = _resolve_draft_spec(draft_config)
    if draft is None:
        draft = _resolve_inline_dspark_spec(
            text_config,
            layer_types,
            (extended.mtp_layers, moe_inter, shared_inter, sliding_window),
        )

    return ModelGeometry(
        hidden=hidden,
        num_layers=num_layers,
        layer_types=layer_types,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        attn_output_gate=_resolve_attn_output_gate(text_config, config),
        linear_k_heads=lin_k_heads,
        linear_v_heads=lin_v_heads,
        linear_k_dim=lin_k_dim,
        linear_v_dim=lin_v_dim,
        conv_kernel=conv_kernel,
        num_experts=num_experts,
        top_k=top_k,
        moe_inter=moe_inter,
        shared_inter=shared_inter,
        vocab=extended.vocab,
        mtp_layers=extended.mtp_layers,
        dense_inter=dense_inter,
        hc_mult=extended.hc_mult,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        sliding_window=sliding_window,
        first_k_dense_replace=extended.first_k,
        ple_dim=extended.ple_dim,
        compress_ratios=extended.compress_ratios,
        block_parallel=extended.block_parallel,
        mtp_inter=extended.mtp_inter,
        n_moe_layers=ffn_kinds.count("moe"),
        n_dense_layers=ffn_kinds.count("dense"),
        global_head_dim=global_head_dim,
        ffn_kinds=ffn_kinds,
        weight_block_size=extended.weight_block_size,
        routed_expert_hidden=_as_int(text_config.get("routed_expert_hidden_size", 0)),
        ffn_num_matrices=ffn_num_matrices,
        tie_word_embeddings=extended.tie_word_embeddings,
        model_id=identity.canonical_id,
        indexer_types=extended.indexer_types,
        index_share_for_mtp_iteration=extended.index_share_for_mtp_iteration,
        qsa_budget=extended.qsa_budget,
        hc_lowrank=extended.hc_lowrank,
        ple_layer_ids=extended.ple_layer_ids,
        max_position_embeddings=extended.max_position_embeddings,
        sparse_attn=sparse_attn,
        host_tables=host_tables,
        draft=draft,
    )


def _from_hf_config_dry_run(
    config: Mapping[str, object],
    *,
    draft_config: Mapping[str, object] | None = None,
) -> ModelGeometry:
    """Private dry-run extractor for audit_hf_config on blocked unlisted configs."""
    text_cfg = _validate_text_config(config)
    scopes = (text_cfg, config) if text_cfg is not config else (text_cfg,)
    model_type = str(_first_non_none(scopes, "model_type") or "").strip().lower()
    spec = _MODEL_TYPE_TO_SPEC.get(model_type)
    allow_csa2 = spec is not None and "csa2" in spec.allowed_layer_kinds
    _validate_structural_support(config, allow_csa2=allow_csa2)
    identity = ModelIdentity(
        canonical_id=spec.canonical_id
        if spec is not None
        else (model_type or "unlisted"),
        model_type=model_type,
        architectures=_resolve_architectures(config, text_cfg),
        hf_repo=None,
    )
    return _extract_geometry_for_identity(config, identity, draft_config=draft_config)


def from_hf_config(
    config: Mapping[str, object],
    *,
    model_ref: str | None = None,
    draft_config: Mapping[str, object] | None = None,
) -> ModelGeometry:
    """Build ModelGeometry from a supported HuggingFace config mapping."""
    identity = resolve_model_identity(config, model_ref=model_ref)
    allow_csa2 = "csa2" in SUPPORTED_MODELS[identity.canonical_id].allowed_layer_kinds
    _validate_structural_support(config, allow_csa2=allow_csa2)
    geometry = _extract_geometry_for_identity(
        config, identity, draft_config=draft_config
    )
    _validate_family_layer_kinds(
        identity, geometry.layer_types, geometry.kv_lora_rank, geometry.index_topk
    )
    return geometry
