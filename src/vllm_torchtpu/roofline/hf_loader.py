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
"""HuggingFace config.json resolution and quantization dtype extraction."""

from __future__ import annotations

import contextlib
import http.client
import json
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .. import envs
from .constants import (
    BASE_DTYPE_BYTES,
    BENIGN_HF_METADATA_KEYS,
    DEFAULT_HTTP_PORT,
    DEFAULT_HTTPS_PORT,
    DEFAULT_SSM_STATE_BYTES,
    EIGHT_BITS,
    FOUR_BITS,
    FP8_BLOCK_SCALE_BYTES,
    FP8_BYTES,
    HTTP_OK,
    HTTP_TIMEOUT_S,
    MULTIMODAL_KEYS,
    MXFP4_BYTES,
    NVFP4_BYTES,
    SIXTEEN_BITS,
    SUPPORTED_MODEL_TYPES,
    WEIGHT_BLOCK_DIM_LEN,
    WEIGHT_ONLY_QUANT_METHODS,
)
from .geometry import (
    KeyTracker,
    LayerCounts,
    ModelGeometry,
    UnsupportedArchitectureError,
    _first_non_none,
    count_layer_types,
    from_hf_config,
    supports,
)
from .memory import ParamBreakdown, param_counts
from .modules import Dtypes

_MIN_PRINTABLE_ASCII = 32
_DEL_ASCII = 127
_HUB_REPO_SEGMENTS = 2
_EXPERT_TARGET_TOKENS = ("moe", "expert")
_DENSE_TARGET_TOKENS = ("linear", "attn", "q_proj", "k_proj", "v_proj", "o_proj")
_STRUCTURAL_QUANT_KEYS = (
    "quant_method",
    "format",
    "quant_algo",
    "algo",
    "scheme",
    "activation_scheme",
    "dense_dtype",
    "expert_dtype",
)


def _as_mapping(value: object) -> Mapping[str, object]:
    """Return value as a mapping when compatible with HF config JSON, else empty."""
    return value if isinstance(value, Mapping) else {}


def _read_cached_hub_repo(repo_dir: Path) -> dict[str, object] | None:
    """Read cached config.json from one HuggingFace Hub repo cache directory."""
    snap_dir = repo_dir / "snapshots"
    ref_main = repo_dir / "refs" / "main"
    if ref_main.is_file():
        rev = ref_main.read_text().strip()
        ref_cfg = snap_dir / rev / "config.json"
        if ref_cfg.is_file():
            return dict(json.loads(ref_cfg.read_text()))
    if snap_dir.is_dir():
        snaps = list(snap_dir.glob("*/config.json"))
        if snaps:
            latest = max(snaps, key=lambda item: item.stat().st_mtime)
            return dict(json.loads(latest.read_text()))
    return None


def _resolve_local_or_hub_url(raw_str: str) -> tuple[dict[str, object] | None, str]:
    """Resolve local JSON or cached Hub snapshot, else return remote config.json URL."""
    p = Path(raw_str)
    if p.is_file():
        return dict(json.loads(p.read_text())), ""
    if p.is_dir() and (p / "config.json").is_file():
        return dict(json.loads((p / "config.json").read_text())), ""

    if raw_str.startswith(("http://", "https://")):
        return None, raw_str

    if (
        p.is_absolute()
        or raw_str.endswith(".json")
        or raw_str.startswith((".", "/", "~"))
        or "://" in raw_str
        or any(
            c.isspace() or ord(c) < _MIN_PRINTABLE_ASCII or ord(c) == _DEL_ASCII
            for c in raw_str
        )
    ):
        msg = f"Could not resolve HF config path: {raw_str}"
        raise FileNotFoundError(msg)

    segments = raw_str.strip("/").split("/")
    if len(segments) == _HUB_REPO_SEGMENTS and segments[0] and segments[1]:
        org_repo = f"{segments[0]}--{segments[1]}"
        roots: list[Path] = []
        if envs.HF_HUB_CACHE:
            roots.append(Path(envs.HF_HUB_CACHE))
        hf_home = Path(envs.HF_HOME or (Path.home() / ".cache" / "huggingface"))
        roots.extend((hf_home / "hub", hf_home))
        for root in roots:
            cached = _read_cached_hub_repo(root / f"models--{org_repo}")
            if cached is not None:
                return cached, ""
        return (
            None,
            f"https://huggingface.co/{segments[0]}/{segments[1]}/resolve/main/config.json",
        )

    msg = f"Could not resolve HF config path: {raw_str}"
    raise FileNotFoundError(msg)


def _fetch_remote_hf_config(url: str, token: str) -> dict[str, object]:
    """Fetch remote HF config via http.client to avoid urllib scheme auditing flags."""
    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.scheme not in ("https", "http") or not parsed_url.hostname:
        msg = f"Unsupported URL scheme for HF config: {url!r}"
        raise ValueError(msg)

    headers = {"Accept": "application/json"}
    if (
        token
        and parsed_url.scheme == "https"
        and parsed_url.hostname == "huggingface.co"
    ):
        headers["Authorization"] = f"Bearer {token}"

    is_https = parsed_url.scheme == "https"
    port = parsed_url.port or (DEFAULT_HTTPS_PORT if is_https else DEFAULT_HTTP_PORT)
    conn_cls = http.client.HTTPSConnection if is_https else http.client.HTTPConnection
    path = parsed_url.path or "/"
    if parsed_url.query:
        path = f"{path}?{parsed_url.query}"
    try:
        with contextlib.closing(
            conn_cls(parsed_url.hostname, port=port, timeout=HTTP_TIMEOUT_S)
        ) as conn:
            conn.request("GET", path, headers=headers)
            resp = conn.getresponse()
            if resp.status != HTTP_OK:
                msg = f"HTTP {resp.status} fetching HF config from {url}"
                raise FileNotFoundError(msg)
            data = resp.read()
    except (TimeoutError, OSError) as exc:
        msg = f"Network timeout or error fetching HF config from {url}: {exc}"
        raise RuntimeError(msg) from exc
    return dict(json.loads(data.decode("utf-8")))


def load_hf_config(
    model_or_path: str | Path | Mapping[str, object],
    *,
    offline: bool = False,
) -> dict[str, object]:
    """Load HF config.json from mapping, local path, HF cache, or Hub URL."""
    if isinstance(model_or_path, Mapping):
        return dict(model_or_path)
    raw_str = str(model_or_path)
    cached_cfg, remote_url = _resolve_local_or_hub_url(raw_str)
    if cached_cfg is not None:
        return cached_cfg

    if offline or envs.HF_HUB_OFFLINE or envs.TRANSFORMERS_OFFLINE:
        msg = f"Offline mode is enabled and {raw_str!r} was not found in local cache"
        raise FileNotFoundError(msg)

    token = envs.HF_TOKEN or envs.HUGGING_FACE_HUB_TOKEN
    return _fetch_remote_hf_config(remote_url, token)


def _extract_base_and_ssm_dtypes(
    tc: Mapping[str, object],
    cfg: Mapping[str, object],
    *,
    has_qcfg: bool = False,
) -> tuple[float, float, str]:
    """Extract standard base model and SSM state byte sizes from config."""
    scopes = (tc, cfg) if tc is not cfg else (tc,)
    raw_dtype = _first_non_none(scopes, "dtype", "torch_dtype") or (
        "bfloat16" if has_qcfg else None
    )
    if not raw_dtype:
        msg = "HF config missing required field: 'dtype' or 'torch_dtype'"
        raise ValueError(msg)
    base_name = str(raw_dtype).lower().removeprefix("torch.")
    if base_name not in BASE_DTYPE_BYTES:
        supp = sorted(BASE_DTYPE_BYTES)
        msg = f"Unknown base dtype: {base_name!r}; supported: {supp}"
        raise ValueError(msg)
    base_b = BASE_DTYPE_BYTES[base_name]

    ssm_dtype_raw = _first_non_none(scopes, "mamba_ssm_dtype", "ssm_dtype")
    if ssm_dtype_raw:
        ssm_name = str(ssm_dtype_raw).lower().removeprefix("torch.")
        if ssm_name not in BASE_DTYPE_BYTES:
            msg = f"Unknown mamba_ssm_dtype: {ssm_dtype_raw!r}"
            raise ValueError(msg)
        state_b = BASE_DTYPE_BYTES[ssm_name]
    else:
        state_b = DEFAULT_SSM_STATE_BYTES
    return base_b, state_b, base_name


def _to_int(value: object, default: int = 0) -> int:
    """Convert JSON scalar number or numeric string to int cleanly."""
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float, str)):
        return int(value)
    return default


def _to_float(value: object, default: float = 0.0) -> float:
    """Convert JSON scalar number or numeric string to float cleanly."""
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float, str)):
        return float(value)
    return default


def _to_seq(value: object) -> Sequence[object] | None:
    """Return value as a non-string sequence when valid, else None."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return None


@dataclass(frozen=True, slots=True)
class _QuantParams:
    """Extracted quantization parameters for one module role."""

    q_method: str
    q_algo: str
    q_fmt: str
    bits: int
    group_size: int | None
    scale_bytes: float | None
    act_bits: int | None
    weight_type: str = ""


def _classify_group_role(
    targets: tuple[str, ...],
) -> Literal["dense", "expert", "both"]:
    """Classify a config_groups entry into dense, expert, or shared scope."""
    if not targets:
        return "both"
    has_expert = any(tok in t for t in targets for tok in _EXPERT_TARGET_TOKENS)
    has_dense = any(tok in t for t in targets for tok in _DENSE_TARGET_TOKENS)
    if has_expert and not has_dense:
        return "expert"
    if has_dense and not has_expert:
        return "dense"
    return "both"


def _extract_single_group_params(
    grp: Mapping[str, object],
    qcfg: Mapping[str, object],
) -> _QuantParams:
    """Extract quantization parameters from one config_groups entry."""
    w = _as_mapping(grp.get("weights"))
    bits = _to_int(w.get("num_bits", 0))
    gs = w.get("group_size")
    group_size = _to_int(gs) if gs is not None else None
    sb = w.get("scale_bytes")
    scale_bytes = _to_float(sb) if sb is not None else None
    sd = w.get("scale_dtype")
    if sd is not None and scale_bytes is None:
        sd_name = str(sd).lower().removeprefix("torch.")
        if sd_name in BASE_DTYPE_BYTES:
            scale_bytes = BASE_DTYPE_BYTES[sd_name]
    ia = _as_mapping(grp.get("input_activations"))
    ia_bits = _to_int(ia.get("num_bits", 0)) if ia else 0
    act_bits = ia_bits if ia_bits > 0 else None
    grp_fmt = str(
        grp.get("format") or w.get("format") or qcfg.get("format") or ""
    ).lower()
    grp_algo = str(
        grp.get("quant_algo")
        or w.get("quant_algo")
        or qcfg.get("quant_algo")
        or qcfg.get("algo")
        or ""
    ).lower()
    grp_method = str(qcfg.get("quant_method") or grp_fmt or "").lower()
    weight_type = str(w.get("type") or "").lower()
    return _QuantParams(
        q_method=grp_method,
        q_algo=grp_algo,
        q_fmt=grp_fmt,
        bits=bits,
        group_size=group_size,
        scale_bytes=scale_bytes,
        act_bits=act_bits,
        weight_type=weight_type,
    )


def _extract_scale_bytes(
    qcfg: Mapping[str, object],
    fallback_sb: float | None,
) -> float | None:
    """Extract scale_bytes from scale_bytes or scale_dtype."""
    if qcfg.get("scale_bytes") is not None:
        return _to_float(qcfg["scale_bytes"])
    if qcfg.get("scale_dtype") is not None:
        sd = str(qcfg["scale_dtype"]).lower().removeprefix("torch.")
        if sd not in BASE_DTYPE_BYTES:
            msg = f"Unknown 4-bit scale_dtype: {sd!r}"
            raise ValueError(msg)
        return BASE_DTYPE_BYTES[sd]
    return fallback_sb


def _merge_top_quant_params(
    qcfg: Mapping[str, object],
    grp_qp: _QuantParams | None,
) -> _QuantParams:
    """Merge top-level quantization_config fields with an optional group entry."""
    q_fmt = str(qcfg.get("format") or "").lower()
    q_method = str(qcfg.get("quant_method") or q_fmt or "").lower()
    q_algo = str(qcfg.get("quant_algo") or qcfg.get("algo") or "").lower()
    top_bits = _to_int(qcfg.get("bits", 0))
    raw_gs = qcfg.get("group_size")
    top_gs = _to_int(raw_gs) if raw_gs is not None else None
    if grp_qp is None:
        return _QuantParams(
            q_method=q_method,
            q_algo=q_algo,
            q_fmt=q_fmt,
            bits=top_bits,
            group_size=top_gs,
            scale_bytes=_extract_scale_bytes(qcfg, None),
            act_bits=None,
        )
    bits = top_bits if top_bits > 0 else grp_qp.bits
    group_size = top_gs if top_gs is not None else grp_qp.group_size
    scale_bytes = _extract_scale_bytes(qcfg, grp_qp.scale_bytes)
    return _QuantParams(
        q_method=grp_qp.q_method or q_method,
        q_algo=grp_qp.q_algo or q_algo,
        q_fmt=grp_qp.q_fmt or q_fmt,
        bits=bits,
        group_size=group_size,
        scale_bytes=scale_bytes,
        act_bits=grp_qp.act_bits,
        weight_type=grp_qp.weight_type,
    )


def _extract_role_quant_params(
    qcfg: Mapping[str, object],
) -> tuple[_QuantParams, _QuantParams]:
    """Extract (dense_qp, expert_qp), separating heterogeneous Linear vs MoE groups."""
    dense_groups: list[_QuantParams] = []
    expert_groups: list[_QuantParams] = []
    both_groups: list[_QuantParams] = []
    cfg_groups = _as_mapping(qcfg.get("config_groups"))
    for grp_raw in cfg_groups.values():
        grp = _as_mapping(grp_raw)
        if not grp:
            continue
        parsed = _extract_single_group_params(grp, qcfg)
        raw_targets = _to_seq(grp.get("targets"))
        targets = tuple(str(item).lower() for item in (raw_targets or ()))
        role = _classify_group_role(targets)
        if role == "dense":
            dense_groups.append(parsed)
        elif role == "expert":
            expert_groups.append(parsed)
        else:
            both_groups.append(parsed)

    dense_grp: _QuantParams | None = None
    expert_grp: _QuantParams | None = None
    if dense_groups and expert_groups:
        dense_grp = dense_groups[0]
        expert_grp = expert_groups[0]
    elif expert_groups:
        expert_grp = expert_groups[0]
        dense_grp = both_groups[0] if both_groups else None
    elif dense_groups:
        dense_grp = dense_groups[0]
        expert_grp = both_groups[0] if both_groups else dense_groups[0]
    elif both_groups:
        dense_grp = both_groups[0]
        expert_grp = both_groups[0]
    return (
        _merge_top_quant_params(qcfg, dense_grp),
        _merge_top_quant_params(qcfg, expert_grp),
    )


def _extract_ignored_modules(
    qcfg: Mapping[str, object],
) -> tuple[bool, bool, bool]:
    """Detect whether attention, lm_head, and aux layers skip quantization."""
    ign_raw = (
        qcfg.get("modules_to_not_convert")
        or qcfg.get("ignore")
        or qcfg.get("ignored_layers")
    )
    ign_seq = _to_seq(ign_raw)
    has_ign_list = ign_seq is not None and len(ign_seq) > 0
    raw_ignored = tuple(str(x).lower() for x in (ign_seq or ()))
    ignored = tuple(
        s for s in raw_ignored if not any(v in s for v in ("visual", "vision", "audio"))
    )
    attn_ignored = any(
        "linear_attn" not in x
        and "indexer" not in x
        and "layernorm" not in x
        and "norm" not in x
        and any(
            tok in x
            for tok in ("self_attn", ".attn.", "q_proj", "k_proj", "v_proj", "o_proj")
        )
        for x in ignored
    )
    lm_ignored = any("lm_head" in x for x in ignored) or not bool(
        qcfg.get("quantize_lm_head", False)
    )
    aux_ignored = (
        any(k in x for x in ignored for k in ("mtp", "gate", "ba", "router"))
        if has_ign_list
        else not bool(qcfg.get("quantize_aux", False))
    )
    return attn_ignored, lm_ignored, aux_ignored


def _derive_nvfp4_dense(
    qcfg: Mapping[str, object],
    fp8_b: float,
    base_b: float,
    *,
    attn_ignored: bool,
) -> float:
    """Derive dense weight bytes for NVFP4."""
    dense_override = str(qcfg.get("dense_dtype", "")).lower()
    if dense_override == "fp8":
        return fp8_b
    if attn_ignored:
        return base_b
    return NVFP4_BYTES


def _derive_four_bit_bytes(
    group_size: int,
    scale_bytes: float | None,
) -> float:
    """Derive weight bytes for 4-bit grouped quantization."""
    if scale_bytes is None or scale_bytes <= 0.0:
        msg = (
            "4-bit quantization config missing required field: "
            "'scale_bytes' or 'scale_dtype'"
        )
        raise ValueError(msg)
    return 0.5 + scale_bytes / float(group_size)


def _is_mxfp4(
    qp: _QuantParams,
    role_dtype: str,
    qcfg: Mapping[str, object],
) -> bool:
    """Return True when role quantization config specifies MXFP4."""
    q_fmt = str(qcfg.get("format") or "").lower()
    return (
        role_dtype in ("fp4", "mxfp4")
        or "mxfp4" in qp.q_method
        or "mxfp4" in qp.q_algo
        or "mxfp4" in qp.q_fmt
        or "mxfp4" in q_fmt
    )


def _is_nvfp4(
    qp: _QuantParams,
    role_dtype: str,
    qcfg: Mapping[str, object],
) -> bool:
    """Return True when role quantization config specifies NVFP4."""
    q_fmt = str(qcfg.get("format") or "").lower()
    q_algo = str(qcfg.get("quant_algo") or qcfg.get("algo") or "").lower()
    return (
        role_dtype == "nvfp4"
        or "nvfp4" in qp.q_method
        or "nvfp4" in qp.q_algo
        or "nvfp4" in qp.q_fmt
        or "nvfp4" in q_fmt
        or "nvfp4" in q_algo
    )


def _derive_quantized_expert_bytes(
    qp: _QuantParams,
    fp8_b: float,
    base_b: float,
    qcfg: Mapping[str, object],
    expert_dtype: str,
) -> float:
    """Derive routed MoE expert weight byte footprint."""
    if _is_mxfp4(qp, expert_dtype, qcfg):
        return MXFP4_BYTES
    if _is_nvfp4(qp, expert_dtype, qcfg):
        return NVFP4_BYTES
    if qp.bits == FOUR_BITS and qp.group_size is not None:
        return _derive_four_bit_bytes(qp.group_size, qp.scale_bytes)
    if "fp8" in qp.q_method or qp.bits == EIGHT_BITS:
        return fp8_b
    return base_b


def _derive_quantized_dense_bytes(
    qp: _QuantParams,
    qcfg: Mapping[str, object],
    weights: tuple[float, float, float],
    *,
    attn_ignored: bool,
) -> float:
    """Derive dense linear projection weight byte footprint."""
    base_b, fp8_b, w_expert = weights
    dense_dtype = str(qcfg.get("dense_dtype") or "").lower()
    if _is_mxfp4(qp, dense_dtype, qcfg):
        if attn_ignored or not qcfg.get("quantize_attn", False):
            return base_b
        return MXFP4_BYTES
    if _is_nvfp4(qp, dense_dtype, qcfg):
        return _derive_nvfp4_dense(qcfg, fp8_b, base_b, attn_ignored=attn_ignored)
    if qp.bits == FOUR_BITS and qp.group_size is not None:
        if attn_ignored:
            return base_b
        if w_expert <= MXFP4_BYTES:
            return _derive_four_bit_bytes(qp.group_size, qp.scale_bytes)
        return w_expert
    if "fp8" in qp.q_method or qp.bits == EIGHT_BITS:
        return base_b if attn_ignored else fp8_b
    return base_b


def _validate_grouped_four_bit(
    qp: _QuantParams,
    *,
    is_fp4: bool,
) -> None:
    """Validate required group_size for non-FP4 4-bit weight quantization."""
    if qp.bits != FOUR_BITS or is_fp4:
        return
    if qp.group_size is None:
        msg = "4-bit quantization config missing required field: 'group_size'"
        raise ValueError(msg)
    if qp.group_size <= 0:
        msg = f"Invalid 4-bit group_size: {qp.group_size}"
        raise ValueError(msg)


def _derive_quantized_weight_bytes(
    role_qps: tuple[_QuantParams, _QuantParams],
    fp8_bytes_pair: tuple[float, float],
    base_b: float,
    qcfg: Mapping[str, object],
    *,
    attn_ignored: bool,
) -> tuple[float, float]:
    """Derive (w_dense, w_expert) byte footprints across quantization schemes."""
    dense_qp, expert_qp = role_qps
    fp8_b_dense, fp8_b_expert = fp8_bytes_pair
    dense_dtype = str(qcfg.get("dense_dtype") or "").lower()
    expert_dtype = str(qcfg.get("expert_dtype") or "").lower()
    is_fp4_expert = _is_mxfp4(expert_qp, expert_dtype, qcfg) or _is_nvfp4(
        expert_qp, expert_dtype, qcfg
    )
    is_fp4_dense = _is_mxfp4(dense_qp, dense_dtype, qcfg) or _is_nvfp4(
        dense_qp, dense_dtype, qcfg
    )
    _validate_grouped_four_bit(expert_qp, is_fp4=is_fp4_expert)
    if not attn_ignored:
        _validate_grouped_four_bit(dense_qp, is_fp4=is_fp4_dense)

    w_expert = _derive_quantized_expert_bytes(
        expert_qp, fp8_b_expert, base_b, qcfg, expert_dtype
    )
    w_dense = _derive_quantized_dense_bytes(
        dense_qp, qcfg, (base_b, fp8_b_dense, w_expert), attn_ignored=attn_ignored
    )
    return w_dense, w_expert


def _collect_structured_quant_tokens(
    dense_qp: _QuantParams,
    expert_qp: _QuantParams,
    qcfg: Mapping[str, object],
) -> tuple[str, ...]:
    """Collect normalized quantization tokens from explicit structured fields only."""
    raw_values: list[object] = [qcfg.get(k) for k in _STRUCTURAL_QUANT_KEYS]
    raw_values.extend(
        [
            dense_qp.q_method,
            dense_qp.q_algo,
            dense_qp.q_fmt,
            expert_qp.q_method,
            expert_qp.q_algo,
            expert_qp.q_fmt,
        ]
    )
    return tuple(str(v).lower() for v in raw_values if isinstance(v, str) and v)


def _derive_activation_mode(
    dense_qp: _QuantParams,
    expert_qp: _QuantParams,
    qcfg: Mapping[str, object],
) -> Literal["a8", "a16"]:
    """Derive compute activation peak ('a8' vs 'a16') from quantization_config."""
    if EIGHT_BITS in (expert_qp.act_bits, dense_qp.act_bits):
        return "a8"
    if (
        (expert_qp.act_bits == SIXTEEN_BITS and dense_qp.act_bits != EIGHT_BITS)
        or dense_qp.q_method in WEIGHT_ONLY_QUANT_METHODS
        or expert_qp.q_method in WEIGHT_ONLY_QUANT_METHODS
    ):
        return "a16"
    if qcfg.get("weight_only") or (
        "config_groups" in qcfg
        and dense_qp.act_bits is None
        and expert_qp.act_bits is None
    ):
        return "a16"
    tokens = _collect_structured_quant_tokens(dense_qp, expert_qp, qcfg)
    if any(tok in t for t in tokens for tok in ("w4a16", "w8a16")):
        return "a16"
    if any(
        tok in t for t in tokens for tok in ("fp8", "nvfp4", "mxfp4", "w4a8", "w8a8")
    ) or any(t == "a8" or t.endswith("_a8") for t in tokens):
        return "a8"
    return "a16"


def _derive_dense_dtype_name(
    qp: _QuantParams,
    qcfg: Mapping[str, object],
    w_dense: float,
    fp8_b: float,
    base_name: str,
    *,
    attn_ignored: bool,
) -> str:
    """Derive descriptive name for dense weight dtype."""
    if attn_ignored:
        return base_name
    dense_dt_str = str(qcfg.get("dense_dtype") or "").lower()
    if _is_mxfp4(qp, dense_dt_str, qcfg):
        return "mxfp4"
    if _is_nvfp4(qp, dense_dt_str, qcfg):
        return "nvfp4"
    if w_dense == fp8_b and fp8_b > FP8_BYTES:
        return "fp8_block"
    if "compressed-tensors" in qp.q_method:
        if qp.bits == FOUR_BITS:
            return "compressed-tensors_int4_packed"
        if qp.bits == EIGHT_BITS or "fp8" in qp.q_fmt:
            return "fp8"
    if "fp8" in qp.q_method or qp.bits == EIGHT_BITS:
        return "fp8"
    if qp.q_method:
        return qp.q_method
    return base_name


def _derive_expert_dtype_name(
    qp: _QuantParams,
    qcfg: Mapping[str, object],
    w_expert: float,
    fp8_b: float,
    base_name: str,
) -> str:
    """Derive descriptive name for expert weight dtype."""
    expert_dt_str = str(qcfg.get("expert_dtype") or "").lower()
    if _is_mxfp4(qp, expert_dt_str, qcfg):
        return "mxfp4"
    if _is_nvfp4(qp, expert_dt_str, qcfg):
        return "nvfp4"
    if w_expert == fp8_b and fp8_b > FP8_BYTES:
        return "fp8_block"
    if "compressed-tensors" in qp.q_method:
        if qp.bits == FOUR_BITS:
            return "compressed-tensors_int4_packed"
        if qp.bits == EIGHT_BITS or "fp8" in qp.q_fmt:
            return "fp8"
    if "fp8" in qp.q_method or qp.bits == EIGHT_BITS:
        return "fp8"
    if qp.q_method:
        return qp.q_method
    return base_name


def _fp8_block_bytes(qcfg: Mapping[str, object], qp: _QuantParams) -> float:
    """Derive FP8 weight bytes per element including block-scale overhead."""
    block_sz = qcfg.get("weight_block_size")
    if (
        isinstance(block_sz, Sequence)
        and not isinstance(block_sz, (str, bytes))
        and len(block_sz) == WEIGHT_BLOCK_DIM_LEN
    ):
        b0, b1 = _to_int(block_sz[0]), _to_int(block_sz[1])
        if b0 > 0 and b1 > 0:
            scale_b = (
                qp.scale_bytes if qp.scale_bytes is not None else FP8_BLOCK_SCALE_BYTES
            )
            return FP8_BYTES + scale_b / float(b0 * b1)
    return FP8_BYTES


def _parse_mapping_kv_scheme(
    raw_kv: Mapping[str, object],
    base_b: float,
    base_name: str,
) -> tuple[float, str, bool]:
    """Parse a Mapping kv_cache_scheme entry and fail closed on 4-bit/FP4 KV."""
    kv_type = str(
        raw_kv.get("type") or raw_kv.get("dtype") or raw_kv.get("quant_algo") or ""
    ).lower()
    kv_bits = _to_int(raw_kv.get("num_bits", 0))
    if (
        any(tok in kv_type for tok in ("fp4", "nvfp4", "mxfp4", "int4"))
        or kv_bits == FOUR_BITS
    ):
        msg = f"Unsupported 4-bit/FP4 KV cache scheme: {dict(raw_kv)!r}"
        raise ValueError(msg)
    if "int8" in kv_type:
        return (FP8_BYTES, "int8", True)
    if "fp8" in kv_type or kv_bits == EIGHT_BITS:
        return (FP8_BYTES, "fp8", True)
    if kv_bits == SIXTEEN_BITS or kv_type in (
        "bfloat16",
        "float16",
        "fp16",
        "bf16",
    ):
        dt_name = kv_type if kv_type in BASE_DTYPE_BYTES else base_name
        return (BASE_DTYPE_BYTES.get(dt_name, base_b), dt_name, True)
    msg = f"Unsupported Mapping kv_cache_scheme: {dict(raw_kv)!r}"
    raise ValueError(msg)


def _parse_kv_cache_scheme(
    qcfg: Mapping[str, object],
    base_b: float,
    base_name: str,
) -> tuple[float, str, bool]:
    """Parse kv_cache_scheme / kv_cache_dtype, failing closed on 4-bit/FP4 KV."""
    raw_kv = (
        qcfg.get("kv_cache_scheme")
        or qcfg.get("kv_cache_dtype")
        or qcfg.get("kv_cache_quant_algo")
    )
    if raw_kv is None or raw_kv == "":
        return (base_b, base_name, False)
    if isinstance(raw_kv, Mapping):
        return _parse_mapping_kv_scheme(raw_kv, base_b, base_name)

    kv_scheme = str(raw_kv).lower()
    if any(tok in kv_scheme for tok in ("fp4", "nvfp4", "mxfp4", "int4")):
        msg = f"Unsupported 4-bit/FP4 KV cache scheme: {raw_kv!r}"
        raise ValueError(msg)
    if "fp8" in kv_scheme or "8" in kv_scheme:
        return (FP8_BYTES, kv_scheme, True)
    if kv_scheme in BASE_DTYPE_BYTES:
        return (BASE_DTYPE_BYTES[kv_scheme], kv_scheme, True)
    return (base_b, kv_scheme, True)


def dtypes_from_hf_config(cfg: Mapping[str, object]) -> Dtypes:
    """Derive Dtypes from torch_dtype, quantization_config, and ignored module lists."""
    tc = _as_mapping(cfg.get("text_config")) or cfg
    scopes = (tc, cfg) if tc is not cfg else (tc,)
    qcfg = _as_mapping(_first_non_none(scopes, "quantization_config"))
    raw_expert_dt = _first_non_none(scopes, "expert_dtype")
    if raw_expert_dt and len(qcfg) > 0 and "expert_dtype" not in qcfg:
        qcfg = {**qcfg, "expert_dtype": raw_expert_dt}
    has_qcfg = len(qcfg) > 0

    base_b, state_b, base_name = _extract_base_and_ssm_dtypes(
        tc, cfg, has_qcfg=has_qcfg
    )
    if not has_qcfg:
        act_mode = "a8" if base_b <= FP8_BYTES else "a16"
        return Dtypes(
            w_dense=base_b,
            w_expert=base_b,
            w_aux=base_b,
            kv=base_b,
            state=state_b,
            conv=base_b,
            router=base_b,
            lm_head=base_b,
            act=act_mode,
            w_dense_source="default",
            w_expert_source="default",
            kv_source="default",
            w_dense_dtype=base_name,
            w_expert_dtype=base_name,
            kv_dtype=base_name,
        )

    dense_qp, expert_qp = _extract_role_quant_params(qcfg)
    attn_ignored, lm_ignored, aux_ignored = _extract_ignored_modules(qcfg)
    fp8_b_dense = _fp8_block_bytes(qcfg, dense_qp)
    fp8_b_expert = _fp8_block_bytes(qcfg, expert_qp)

    w_dense, w_expert = _derive_quantized_weight_bytes(
        (dense_qp, expert_qp),
        (fp8_b_dense, fp8_b_expert),
        base_b,
        qcfg,
        attn_ignored=attn_ignored,
    )

    kv_b, kv_dt, kv_explicit = _parse_kv_cache_scheme(qcfg, base_b, base_name)

    quant_conv = bool(qcfg.get("quantize_conv", False))
    conv_b = w_dense if quant_conv else base_b

    quant_router = bool(qcfg.get("quantize_router", False)) or (
        bool(qcfg.get("quantize_aux", False)) and not aux_ignored
    )
    router_b = w_dense if quant_router else base_b

    act_mode = _derive_activation_mode(dense_qp, expert_qp, qcfg)

    w_dense_src: Literal["quantization_config", "flag", "default"] = (
        "quantization_config"
    )
    w_dense_dt = _derive_dense_dtype_name(
        dense_qp, qcfg, w_dense, fp8_b_dense, base_name, attn_ignored=attn_ignored
    )

    w_expert_src: Literal["quantization_config", "flag", "default"] = (
        "quantization_config"
    )
    w_expert_dt = _derive_expert_dtype_name(
        expert_qp, qcfg, w_expert, fp8_b_expert, base_name
    )

    kv_src: Literal["quantization_config", "flag", "default"] = (
        "quantization_config" if kv_explicit else "default"
    )

    return Dtypes(
        w_dense=w_dense,
        w_expert=w_expert,
        w_aux=base_b if aux_ignored else w_dense,
        kv=kv_b,
        state=state_b,
        conv=conv_b,
        router=router_b,
        lm_head=base_b if lm_ignored else w_dense,
        act=act_mode,
        w_dense_source=w_dense_src,
        w_expert_source=w_expert_src,
        kv_source=kv_src,
        w_dense_dtype=w_dense_dt,
        w_expert_dtype=w_expert_dt,
        kv_dtype=kv_dt,
    )


@dataclass(frozen=True, slots=True)
class ConfigAudit:
    """Onboarding readiness and zero-drift key audit for an HF config mapping."""

    onboarding_tier: Literal[
        "tier0_ready", "tier1_config_needed", "tier2_primitive_needed"
    ]
    model_type: str
    model_type_status: Literal[
        "listed", "unlisted_structural_ok", "unlisted_blocked", "omitted"
    ]
    unmapped_structural_keys: tuple[str, ...]
    ignored_non_text_keys: tuple[str, ...]
    blocking_reasons: tuple[str, ...]
    geometry: ModelGeometry | None
    dtypes: Dtypes | None
    layer_counts: LayerCounts | None
    param_counts: ParamBreakdown | None


def _collect_unmapped_structural_keys(
    raw_cfg: Mapping[str, object],
    consumed: frozenset[str],
) -> tuple[str, ...]:
    """Return sorted text-config keys omitted from both value reads and benign sets."""
    raw_sub = _as_mapping(raw_cfg.get("text_config"))
    unmapped: list[str] = []
    if raw_sub:
        unmapped.extend(
            key
            for key in raw_sub
            if f"text_config.{key}" not in consumed
            and key not in BENIGN_HF_METADATA_KEYS
        )
        unmapped.extend(
            key
            for key in raw_cfg
            if key != "text_config"
            and key not in MULTIMODAL_KEYS
            and key not in raw_sub
            and key not in consumed
            and key not in BENIGN_HF_METADATA_KEYS
        )
    else:
        unmapped.extend(
            key
            for key in raw_cfg
            if key not in consumed
            and key not in BENIGN_HF_METADATA_KEYS
            and key not in MULTIMODAL_KEYS
        )
    return tuple(sorted(unmapped))


def _resolve_model_type_status(
    model_type: str,
    *,
    allow_unlisted_model_type: bool,
    structural_ok: bool,
) -> Literal["listed", "unlisted_structural_ok", "unlisted_blocked", "omitted"]:
    """Classify config model_type registry and allowlist status."""
    if not model_type:
        return "omitted"
    if model_type in SUPPORTED_MODEL_TYPES:
        return "listed"
    if allow_unlisted_model_type and structural_ok:
        return "unlisted_structural_ok"
    return "unlisted_blocked"


def _run_tracked_extraction(
    raw_cfg: Mapping[str, object],
    *,
    allow_unlisted_model_type: bool,
) -> tuple[
    ModelGeometry,
    Dtypes,
    LayerCounts,
    ParamBreakdown,
    tuple[str, ...],
]:
    """Extract geometry, dtypes, counts, params, and unmapped keys under KeyTracker."""
    tracker = KeyTracker(raw_cfg)
    geom = from_hf_config(tracker, allow_unlisted_model_type=allow_unlisted_model_type)
    dtypes = dtypes_from_hf_config(tracker)
    counts = count_layer_types(geom)
    params = param_counts(geom)
    unmapped = _collect_unmapped_structural_keys(raw_cfg, tracker.consumed_keys)
    return geom, dtypes, counts, params, unmapped


def audit_hf_config(
    cfg: Mapping[str, object],
    *,
    allow_unlisted_model_type: bool = False,
) -> ConfigAudit:
    """Audit HF config onboarding readiness, key coverage, and parameter counts."""
    raw_cfg: Mapping[str, object] = cfg.raw_data if isinstance(cfg, KeyTracker) else cfg
    raw_sub = _as_mapping(raw_cfg.get("text_config"))
    raw_text_cfg = raw_sub if raw_sub else raw_cfg
    scopes = (raw_text_cfg, raw_cfg) if raw_text_cfg is not raw_cfg else (raw_text_cfg,)
    model_type = str(_first_non_none(scopes, "model_type") or "").strip().lower()
    ignored_non_text = tuple(sorted(k for k in raw_cfg if k in MULTIMODAL_KEYS))

    try:
        supports(raw_cfg, allow_unlisted_model_type=True)
    except UnsupportedArchitectureError as exc:
        reasons = [str(exc)]
        if (
            model_type
            and model_type not in SUPPORTED_MODEL_TYPES
            and not allow_unlisted_model_type
        ):
            reasons.insert(
                0,
                f"Unrecognized or unsupported model_type {model_type!r}; fails closed",
            )
        return ConfigAudit(
            onboarding_tier="tier2_primitive_needed",
            model_type=model_type,
            model_type_status=_resolve_model_type_status(
                model_type, allow_unlisted_model_type=False, structural_ok=False
            ),
            unmapped_structural_keys=(),
            ignored_non_text_keys=ignored_non_text,
            blocking_reasons=tuple(reasons),
            geometry=None,
            dtypes=None,
            layer_counts=None,
            param_counts=None,
        )

    try:
        supports(raw_cfg, allow_unlisted_model_type=allow_unlisted_model_type)
    except UnsupportedArchitectureError as gate_exc:
        reasons = [str(gate_exc)]
        unmapped: tuple[str, ...] = ()
        tier: Literal[
            "tier0_ready", "tier1_config_needed", "tier2_primitive_needed"
        ] = "tier1_config_needed"
        try:
            _, _, _, _, unmapped = _run_tracked_extraction(
                raw_cfg, allow_unlisted_model_type=True
            )
        except UnsupportedArchitectureError as dry_exc:
            reasons.append(str(dry_exc))
            tier = "tier2_primitive_needed"
        except (ValueError, KeyError) as dry_exc:
            reasons.append(str(dry_exc))
        return ConfigAudit(
            onboarding_tier=tier,
            model_type=model_type,
            model_type_status="unlisted_blocked",
            unmapped_structural_keys=unmapped,
            ignored_non_text_keys=ignored_non_text,
            blocking_reasons=tuple(reasons),
            geometry=None,
            dtypes=None,
            layer_counts=None,
            param_counts=None,
        )

    model_type_status = _resolve_model_type_status(
        model_type,
        allow_unlisted_model_type=allow_unlisted_model_type,
        structural_ok=True,
    )
    try:
        geom, dtypes, counts, params, unmapped = _run_tracked_extraction(
            raw_cfg, allow_unlisted_model_type=allow_unlisted_model_type
        )
    except UnsupportedArchitectureError as exc:
        return ConfigAudit(
            onboarding_tier="tier2_primitive_needed",
            model_type=model_type,
            model_type_status=model_type_status,
            unmapped_structural_keys=(),
            ignored_non_text_keys=ignored_non_text,
            blocking_reasons=(str(exc),),
            geometry=None,
            dtypes=None,
            layer_counts=None,
            param_counts=None,
        )
    except (ValueError, KeyError) as exc:
        reason = (
            f"Missing required config field: {exc}"
            if isinstance(exc, KeyError)
            else str(exc)
        )
        return ConfigAudit(
            onboarding_tier="tier1_config_needed",
            model_type=model_type,
            model_type_status=model_type_status,
            unmapped_structural_keys=(),
            ignored_non_text_keys=ignored_non_text,
            blocking_reasons=(reason,),
            geometry=None,
            dtypes=None,
            layer_counts=None,
            param_counts=None,
        )

    return ConfigAudit(
        onboarding_tier="tier0_ready",
        model_type=model_type,
        model_type_status=model_type_status,
        unmapped_structural_keys=unmapped,
        ignored_non_text_keys=ignored_non_text,
        blocking_reasons=(),
        geometry=geom,
        dtypes=dtypes,
        layer_counts=counts,
        param_counts=params,
    )
