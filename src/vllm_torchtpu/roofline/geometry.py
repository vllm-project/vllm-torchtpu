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
"""Model geometry, support guards, and 10-family layer registry."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from .constants import (
    BLOCK_PARALLEL_ENABLED,
    GLU_MATMUL_COUNT,
    HYBRID_PATTERN_MAP,
    HYBRID_SSM_MODEL_TYPES,
    LAYER_TYPE_MAP,
    LINEAR_ATTENTION_LAYER_TOKENS,
    MULTIMODAL_KEYS,
    NON_GATED_MATMUL_COUNT,
    NON_GATED_MLP_MODEL_TYPES,
    SUPPORTED_MODEL_TYPES,
    VALID_LAYER_TOKENS,
    WEIGHT_BLOCK_DIM_LEN,
)

__all__ = [
    "KeyTracker",
    "LayerCounts",
    "ModelGeometry",
    "UnsupportedArchitectureError",
    "count_layer_types",
    "from_hf_config",
    "supports",
]


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
    """Return first truthy value across ``keys``, reading all present aliases."""
    chosen: object = None
    for key in keys:
        if key in mapping:
            val = mapping[key]
            if not chosen and val:
                chosen = val
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
    num_shared_experts: int = 0
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

    def ffn_layer_counts(self) -> tuple[int, int]:
        """Return (n_moe_ffn, n_dense_ffn) counted from layer_ffn_kinds."""
        kinds = layer_ffn_kinds(self)
        return (kinds.count("moe"), kinds.count("dense"))


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


def count_layer_types(geom: ModelGeometry) -> LayerCounts:
    """Count occurrences of attention/recurrent layers and KV retention weights."""
    counts = {"gqa": 0, "swa": 0, "mla": 0, "dsa": 0, "gdn": 0, "kda": 0}
    kv_sums = {"gqa": 0.0, "swa": 0.0, "mla": 0.0, "dsa": 0.0}
    compress_ratios = geom.compress_ratios or ()
    num_compress_ratios = len(compress_ratios)
    ignored = {"ffn_only", "mlp", "dense_mlp", "moe", "mtp", "ple"}
    for layer_idx, layer_tok in enumerate(geom.layer_types):
        if layer_tok in ignored:
            continue
        ratio_i = compress_ratios[layer_idx] if layer_idx < num_compress_ratios else 1.0
        kv_ret = (1.0 / ratio_i) if ratio_i > 1.0 else ratio_i
        for family in _resolve_layer_family(
            layer_tok, geom.kv_lora_rank, geom.index_topk
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


def layer_ffn_kinds(geom: ModelGeometry) -> tuple[str, ...]:
    """Return per-layer FFN kinds ('dense', 'moe', or 'none') for model geometry."""
    if geom.ffn_kinds and len(geom.ffn_kinds) == geom.num_layers:
        return geom.ffn_kinds
    if geom.num_experts > 0:
        first_k = min(geom.num_layers, geom.first_k_dense_replace)
        return ("dense",) * first_k + ("moe",) * (geom.num_layers - first_k)
    return ("dense",) * geom.num_layers


def _validate_text_config(cfg: Mapping[str, object]) -> Mapping[str, object]:
    """Validate and extract text_config mapping from top-level config."""
    if not cfg:
        msg = "Empty HuggingFace config mapping"
        raise UnsupportedArchitectureError(msg)
    has_text_key = "text_config" in cfg
    subconfig = _as_mapping(cfg.get("text_config"))
    if has_text_key and not subconfig:
        msg = "Multimodal config has non-mapping text_config"
        raise UnsupportedArchitectureError(msg)
    text_cfg: Mapping[str, object] = subconfig if subconfig else cfg
    is_multimodal = has_text_key or any(
        multimodal_key in cfg for multimodal_key in MULTIMODAL_KEYS
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


def _validate_modeled_attention_variants(tc: Mapping[str, object]) -> None:
    """Fail closed on unmodeled LoRA output projections or SWA-only DSA configs."""
    if _as_int(tc.get("o_lora_rank")) > 0:
        msg = "Non-zero o_lora_rank attention output projection is not modeled"
        raise UnsupportedArchitectureError(msg)
    sliding_window = _as_int(tc.get("sliding_window"))
    index_topk = _as_int(tc.get("index_topk"))
    kv_lora_rank = _as_int(tc.get("kv_lora_rank"))
    if sliding_window > 0 and index_topk > 0 and kv_lora_rank <= 0:
        msg = (
            "Hybrid sliding_window + index_topk without kv_lora_rank > 0 is not modeled"
        )
        raise UnsupportedArchitectureError(msg)


def _resolve_attn_output_gate(
    tc: Mapping[str, object],
    cfg: Mapping[str, object],
) -> bool:
    """Resolve whether attention output gating is enabled in text or root config."""
    scopes = (tc, cfg) if tc is not cfg else (tc,)
    raw_flag = _first_non_none(scopes, "attn_output_gate")
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


def supports(
    cfg: Mapping[str, object],
    *,
    allow_unlisted_model_type: bool = False,
) -> None:
    """Validate that cfg maps onto modeled attention, recurrence, and FFN families."""
    text_cfg = _validate_text_config(cfg)
    scopes = (text_cfg, cfg) if text_cfg is not cfg else (text_cfg,)
    model_type = str(_first_non_none(scopes, "model_type") or "").strip().lower()

    has_gdn = _has_gdn_kda_heads(text_cfg)

    # Fail closed on unlisted model types or unsupported SSMs
    if model_type:
        if model_type not in SUPPORTED_MODEL_TYPES and not allow_unlisted_model_type:
            msg = f"Unrecognized or unsupported model_type {model_type!r}; fails closed"
            raise UnsupportedArchitectureError(msg)
        if model_type in HYBRID_SSM_MODEL_TYPES and not has_gdn:
            msg = (
                f"Hybrid state-space model_type {model_type!r} "
                "missing required GDN/KDA head keys"
            )
            raise UnsupportedArchitectureError(msg)

    if ("ssm_cfg" in text_cfg or "state_size" in text_cfg) and not (
        text_cfg.get("num_attention_heads") or has_gdn
    ):
        msg = "Unmodeled state-space config missing attention and GDN/KDA head keys"
        raise UnsupportedArchitectureError(msg)

    _validate_modeled_attention_variants(text_cfg)
    _validate_layer_types(text_cfg, has_gdn=has_gdn)


def _is_kda_config(tc: Mapping[str, object]) -> bool:
    """Return True when config declares KDA-style linear attention keys."""
    lin_cfg = _as_mapping(tc.get("linear_attn_config"))
    if (
        lin_cfg.get("kda_layers") is not None
        or lin_cfg.get("short_conv_kernel_size")
        or lin_cfg.get("kda")
    ):
        return True
    raw_types = _as_sequence(tc.get("layer_types")) or ()
    return any(str(tok) in ("kda", "kda_attention") for tok in raw_types)


def _parse_dsa_layer_set(
    tc: Mapping[str, object],
    cfg: Mapping[str, object],
    num_layers: int,
) -> set[int] | None:
    """Return 0-based DSA layer indices when dsa_layers/indexer_layers is set."""
    scopes = (tc, cfg) if tc is not cfg else (tc,)
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
    tc: Mapping[str, object],
    num_layers: int,
    specs: tuple[int, int, int],
) -> tuple[str, ...]:
    """Infer layer types from interval, attention list, or override pattern."""
    kv_lora_rank, index_topk, sliding_window = specs
    lin_cfg = _as_mapping(tc.get("linear_attn_config"))
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
    interval = _as_int(tc.get("full_attention_interval"), 0)
    if interval > 0:
        att_tok = (
            "dsa"
            if (kv_lora_rank > 0 and index_topk > 0)
            else ("mla" if kv_lora_rank > 0 else "full_attention")
        )
        lin_tok = "kda" if _is_kda_config(tc) else "linear_attention"
        return tuple(
            att_tok if (i + 1) % interval == 0 else lin_tok for i in range(num_layers)
        )
    attn_list = _as_sequence(tc.get("attn_type_list"))
    if attn_list is not None and len(attn_list) == num_layers:
        return tuple(
            "full_attention" if _as_int(v, 0) == 1 else "linear_attention"
            for v in attn_list
        )
    hybrid_pattern = tc.get("hybrid_override_pattern")
    if isinstance(hybrid_pattern, str) and len(hybrid_pattern.strip()) == num_layers:
        pattern_text = hybrid_pattern.strip()
        if any(ch not in HYBRID_PATTERN_MAP for ch in pattern_text):
            msg = f"Unsupported hybrid_override_pattern tokens in {pattern_text!r}"
            raise UnsupportedArchitectureError(msg)
        return tuple(HYBRID_PATTERN_MAP[ch] for ch in pattern_text)
    sw_pat = _as_int(
        tc.get("sliding_window_pattern") or tc.get("_sliding_window_pattern"), 0
    )
    if sliding_window > 0:
        default_layers = (
            tuple(
                "full_attention" if (i + 1) % sw_pat == 0 else "sliding_attention"
                for i in range(num_layers)
            )
            if sw_pat >= 1
            else ("sliding_attention",) * num_layers
        )
    elif kv_lora_rank > 0:
        default_mla = "dsa" if index_topk > 0 else "mla"
        default_layers = (default_mla,) * num_layers
    else:
        default_layers = ("full_attention",) * num_layers
    return default_layers


_MLA_OR_FULL_ATTN_TOKENS: frozenset[str] = frozenset(
    {tok for tok, fam in LAYER_TYPE_MAP.items() if fam in ("mla", "dsa")}
    | {"full_attention", "attention"}
)


def _resolve_layer_types(
    tc: Mapping[str, object],
    cfg: Mapping[str, object],
    num_layers: int,
    specs: tuple[int, int, int],
) -> tuple[str, ...]:
    """Resolve sequence of layer type strings for each transformer layer."""
    kv_lora_rank, _, _ = specs
    dsa_set = _parse_dsa_layer_set(tc, cfg, num_layers)
    raw_types = _as_sequence(tc.get("layer_types"))
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
    else:
        layer_types = _infer_pattern_layer_types(tc, num_layers, specs)
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


def _resolve_moe_dimensions(
    tc: Mapping[str, object],
) -> tuple[int, int, int, int, int]:
    """Resolve (num_experts, top_k, moe_inter, num_shared_experts, shared_inter)."""
    num_experts = _as_int(
        _first_present(tc, "num_experts", "n_routed_experts", "num_local_experts")
    )
    dense_inter = _as_int(
        _first_present(tc, "intermediate_size", "dense_intermediate_size")
    )
    moe_inter = _as_int(_first_present(tc, "moe_intermediate_size", "latent_moe_dim"))
    if num_experts > 0 and moe_inter == 0 and dense_inter > 0:
        moe_inter = dense_inter

    num_shared = _as_int(_first_present(tc, "n_shared_experts", "num_shared_experts"))
    shared_inter = _as_int(tc.get("shared_expert_intermediate_size"))
    if shared_inter == 0 and num_shared > 0 and moe_inter > 0:
        shared_inter = num_shared * moe_inter

    top_k = _as_int(_first_present(tc, *_ROUTED_EXPERTS_PER_TOKEN_KEYS))
    if num_experts > 0 and top_k <= 0:
        accepted_keys = ", ".join(_ROUTED_EXPERTS_PER_TOKEN_KEYS)
        msg = f"MoE config names no routed experts per token ({accepted_keys})"
        raise ValueError(msg)
    return num_experts, top_k, moe_inter, num_shared, shared_inter


def _resolve_ffn_kinds(
    tc: Mapping[str, object],
    cfg: Mapping[str, object],
    num_layers: int,
    num_experts: int,
) -> tuple[str, ...]:
    """Resolve (moe, dense, none) kind for each layer's feed-forward network."""
    hyb_s = str(tc.get("hybrid_override_pattern") or "").strip()
    if hyb_s and len(hyb_s) == num_layers:
        return tuple(
            "moe" if ch == "E" else ("dense" if ch == "-" else "none") for ch in hyb_s
        )
    raw_moe_layers = _as_sequence(tc.get("moe_layers"))
    if raw_moe_layers is not None:
        moe_set = {_as_int(x) for x in raw_moe_layers}
        return tuple("moe" if i in moe_set else "dense" for i in range(num_layers))
    interleave_step = _as_int(tc.get("interleave_moe_layer_step"), 0)
    if num_experts > 0 and interleave_step > 1:
        return tuple(
            "moe" if (i + 1) % interleave_step == 0 else "dense"
            for i in range(num_layers)
        )
    scopes = (tc, cfg) if tc is not cfg else (tc,)
    first_k_dense = _as_int(_first_non_none(scopes, "first_k_dense_replace") or 0)
    if num_experts > 0:
        return tuple("dense" if i < first_k_dense else "moe" for i in range(num_layers))
    return ("dense",) * num_layers


def _resolve_linear_attention_dimensions(
    tc: Mapping[str, object],
    layer_types: tuple[str, ...],
    attn_dims: tuple[int, int, int],
) -> tuple[int, int, int, int, int]:
    """Resolve (lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, conv_kernel)."""
    num_heads, num_kv_heads, head_dim = attn_dims
    lin_cfg = _as_mapping(tc.get("linear_attn_config"))
    has_attn_list = tc.get("attn_type_list") is not None
    lin_k_heads = _as_int(
        tc.get("linear_num_key_heads")
        or lin_cfg.get("num_key_heads")
        or lin_cfg.get("num_heads")
        or tc.get("mamba_num_heads")
        or (num_kv_heads if has_attn_list else 0)
    )
    lin_v_heads = _as_int(
        tc.get("linear_num_value_heads")
        or lin_cfg.get("num_value_heads")
        or lin_cfg.get("num_heads")
        or tc.get("mamba_num_heads")
        or (num_heads if has_attn_list else 0)
    )
    lin_k_dim = _as_int(
        tc.get("linear_key_head_dim")
        or lin_cfg.get("key_head_dim")
        or lin_cfg.get("head_dim")
        or tc.get("mamba_head_dim")
        or (head_dim if has_attn_list else 0)
    )
    lin_v_dim = _as_int(
        tc.get("linear_value_head_dim")
        or lin_cfg.get("value_head_dim")
        or lin_cfg.get("head_dim")
        or tc.get("mamba_head_dim")
        or (head_dim if has_attn_list else 0)
    )

    has_linear = any(t in LINEAR_ATTENTION_LAYER_TOKENS for t in layer_types)
    raw_conv = (
        tc.get("linear_conv_kernel_dim")
        or lin_cfg.get("short_conv_kernel_size")
        or lin_cfg.get("conv_kernel_size")
        or lin_cfg.get("linear_conv_kernel_dim")
        or tc.get("conv_kernel")
        or tc.get("d_conv")
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


def _resolve_mtp_layers(tc: Mapping[str, object], cfg: Mapping[str, object]) -> int:
    """Resolve number of Multi-Token Prediction (MTP) layers."""
    mtp_raw = tc.get("mtp_num_hidden_layers")
    if mtp_raw is None:
        mtp_raw = cfg.get("mtp_num_hidden_layers")
    if mtp_raw is None:
        mtp_raw = tc.get(
            "num_nextn_predict_layers", cfg.get("num_nextn_predict_layers", 0)
        )
    return _as_int(mtp_raw)


def _resolve_compress_ratios(
    tc: Mapping[str, object],
    cfg: Mapping[str, object],
    num_layers: int,
) -> tuple[float, ...]:
    """Resolve per-layer KV compression or retention ratios."""
    raw_cr = _as_sequence(tc.get("compress_ratios", cfg.get("compress_ratios")))
    if raw_cr is None or not raw_cr:
        return ()
    if len(raw_cr) > num_layers:
        raw_cr = raw_cr[:num_layers]
    elif len(raw_cr) != num_layers:
        msg = f"compress_ratios length {len(raw_cr)} != num_hidden_layers {num_layers}"
        raise ValueError(msg)
    compress_ratios = tuple(_as_float(x) for x in raw_cr)
    if any(r <= 0.0 for r in compress_ratios):
        msg = f"compress_ratios entries must be > 0, got {compress_ratios}"
        raise ValueError(msg)
    return compress_ratios


def _resolve_block_parallel(tc: Mapping[str, object], cfg: Mapping[str, object]) -> int:
    """Resolve block parallelism factor (1 or 2)."""
    raw_bp = tc.get("block_parallel", cfg.get("block_parallel"))
    if raw_bp is None:
        raw_bp = tc.get("parallel_attn_mlp", cfg.get("parallel_attn_mlp", 1))
    if isinstance(raw_bp, bool):
        return BLOCK_PARALLEL_ENABLED if raw_bp else 1
    return max(1, _as_int(raw_bp, 1))


def _resolve_ffn_num_matrices(
    tc: Mapping[str, object],
    cfg: Mapping[str, object],
    model_type: str,
) -> int:
    """Resolve per-layer FFN matrix multiplier (2 non-gated, 3 gated)."""
    scopes = (tc, cfg) if tc is not cfg else (tc,)
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
    if model_type in NON_GATED_MLP_MODEL_TYPES:
        return NON_GATED_MATMUL_COUNT
    return GLU_MATMUL_COUNT


def from_hf_config(
    cfg: Mapping[str, object],
    *,
    allow_unlisted_model_type: bool = False,
) -> ModelGeometry:
    """Build ModelGeometry from a raw or nested HF config mapping."""
    supports(cfg, allow_unlisted_model_type=allow_unlisted_model_type)
    sub = _as_mapping(cfg.get("text_config"))
    tc: Mapping[str, object] = sub if sub else cfg
    _validate_modeled_attention_variants(tc)
    scopes = (tc, cfg) if tc is not cfg else (tc,)
    model_type = str(_first_non_none(scopes, "model_type") or "").strip().lower()
    hidden = _as_int(tc["hidden_size"])
    num_layers = _as_int(tc["num_hidden_layers"])
    num_heads = _as_int(tc["num_attention_heads"])
    raw_kv_heads = tc.get("num_key_value_heads")
    num_kv_heads = num_heads if raw_kv_heads is None else _as_int(raw_kv_heads)
    raw_head_dim = _as_int(tc.get("head_dim"))
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

    kv_lora_rank = _as_int(tc.get("kv_lora_rank"))
    q_lora_rank = _as_int(tc.get("q_lora_rank"))
    qk_nope_head_dim = _as_int(tc.get("qk_nope_head_dim"))
    qk_rope_head_dim = _as_int(tc.get("qk_rope_head_dim"))
    v_head_dim = _as_int(tc.get("v_head_dim"))
    index_n_heads = _as_int(tc.get("index_n_heads"))
    index_head_dim = _as_int(tc.get("index_head_dim"))
    index_topk = _as_int(tc.get("index_topk"))
    sliding_window = _as_int(tc.get("sliding_window"))
    global_head_dim = _as_int(tc.get("global_head_dim"))

    layer_types = _resolve_layer_types(
        tc, cfg, num_layers, (kv_lora_rank, index_topk, sliding_window)
    )
    dense_inter = _as_int(
        _first_present(tc, "intermediate_size", "dense_intermediate_size")
    )
    num_experts, top_k, moe_inter, num_shared, shared_inter = _resolve_moe_dimensions(
        tc
    )
    ffn_kinds = _resolve_ffn_kinds(tc, cfg, num_layers, num_experts)
    ffn_num_matrices = _resolve_ffn_num_matrices(tc, cfg, model_type)
    lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, conv_kernel = (
        _resolve_linear_attention_dimensions(
            tc, layer_types, (num_heads, num_kv_heads, head_dim)
        )
    )
    mtp_layers = _resolve_mtp_layers(tc, cfg)
    ple_dim = _as_int(_first_present(tc, "per_layer_input_dim", "ple_dim"))
    first_k = _as_int(_first_non_none(scopes, "first_k_dense_replace") or 0)
    compress_ratios = _resolve_compress_ratios(tc, cfg, num_layers)
    block_parallel = _resolve_block_parallel(tc, cfg)
    mtp_inter = _as_int(_first_non_none(scopes, "mtp_intermediate_size") or 0)
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
    hc_mult = max(1, _as_int(raw_hc, 1))
    tie_word_embeddings = bool(_first_non_none(scopes, "tie_word_embeddings") or False)
    vocab = _as_int(_first_non_none(scopes, "vocab_size") or 0)

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

    return ModelGeometry(
        hidden=hidden,
        num_layers=num_layers,
        layer_types=layer_types,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        attn_output_gate=_resolve_attn_output_gate(tc, cfg),
        linear_k_heads=lin_k_heads,
        linear_v_heads=lin_v_heads,
        linear_k_dim=lin_k_dim,
        linear_v_dim=lin_v_dim,
        conv_kernel=conv_kernel,
        num_experts=num_experts,
        top_k=top_k,
        moe_inter=moe_inter,
        shared_inter=shared_inter,
        vocab=vocab,
        mtp_layers=mtp_layers,
        dense_inter=dense_inter,
        hc_mult=hc_mult,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        sliding_window=sliding_window,
        num_shared_experts=num_shared,
        first_k_dense_replace=first_k,
        ple_dim=ple_dim,
        compress_ratios=compress_ratios,
        block_parallel=block_parallel,
        mtp_inter=mtp_inter,
        n_moe_layers=ffn_kinds.count("moe"),
        n_dense_layers=ffn_kinds.count("dense"),
        global_head_dim=global_head_dim,
        ffn_kinds=ffn_kinds,
        weight_block_size=weight_block_size,
        routed_expert_hidden=_as_int(tc.get("routed_expert_hidden_size", 0)),
        ffn_num_matrices=ffn_num_matrices,
        tie_word_embeddings=tie_word_embeddings,
    )
