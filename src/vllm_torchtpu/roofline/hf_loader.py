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

import json
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .. import envs
from .modules import Dtypes

_BASE_DTYPE_BYTES: dict[str, float] = {
    "float32": 4.0,
    "fp32": 4.0,
    "bfloat16": 2.0,
    "bf16": 2.0,
    "float16": 2.0,
    "fp16": 2.0,
    "float8_e4m3fn": 1.0,
    "fp8": 1.0,
}


def load_hf_config(model_or_path: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load HF config.json from an in-memory dict, local path, HF cache, or Hub URL."""
    if type(model_or_path) is dict:
        return dict(model_or_path)
    raw_str = str(model_or_path)
    p = Path(raw_str)
    if p.is_file():
        return dict(json.loads(p.read_text()))
    if p.is_dir() and (p / "config.json").is_file():
        return dict(json.loads((p / "config.json").read_text()))

    if "/" in raw_str and not raw_str.startswith(("http://", "https://")):
        org_repo = raw_str.strip("/").replace("/", "--")
        hf_home = Path(envs.HF_HOME or (Path.home() / ".cache" / "huggingface"))
        for root in (hf_home / "hub", hf_home):
            snap_dir = root / f"models--{org_repo}" / "snapshots"
            if snap_dir.is_dir():
                snaps = sorted(snap_dir.glob("*/config.json"))
                if snaps:
                    return dict(json.loads(snaps[-1].read_text()))
        url = f"https://huggingface.co/{raw_str.strip('/')}/resolve/main/config.json"
    elif raw_str.startswith(("http://", "https://")):
        url = raw_str
    else:
        raise FileNotFoundError(f"Could not resolve HF config path: {raw_str}")

    headers = {"Accept": "application/json"}
    token = envs.HF_TOKEN or envs.HUGGING_FACE_HUB_TOKEN
    parsed_url = urllib.parse.urlparse(url)
    if (
        token
        and parsed_url.scheme == "https"
        and parsed_url.hostname == "huggingface.co"
    ):
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return dict(json.loads(resp.read().decode("utf-8")))


def dtypes_from_hf_config(cfg: Mapping[str, Any]) -> Dtypes:
    """Derive Dtypes from torch_dtype, quantization_config, and ignored module lists."""
    sub = cfg.get("text_config")
    tc: Mapping[str, Any] = sub if sub is not None and type(sub) is dict else cfg
    raw_dtype = (
        tc.get("dtype")
        or cfg.get("dtype")
        or tc.get("torch_dtype")
        or cfg.get("torch_dtype")
    )
    if not raw_dtype:
        raise ValueError("HF config missing required field: 'dtype' or 'torch_dtype'")
    base_name = str(raw_dtype).lower().removeprefix("torch.")
    if base_name not in _BASE_DTYPE_BYTES:
        raise ValueError(
            f"Unknown base dtype: {base_name!r}; supported: {sorted(_BASE_DTYPE_BYTES)}"
        )
    base_b = _BASE_DTYPE_BYTES[base_name]

    ssm_dtype_raw = (
        tc.get("mamba_ssm_dtype")
        or cfg.get("mamba_ssm_dtype")
        or tc.get("ssm_dtype")
        or cfg.get("ssm_dtype")
    )
    if ssm_dtype_raw:
        ssm_name = str(ssm_dtype_raw).lower().removeprefix("torch.")
        if ssm_name not in _BASE_DTYPE_BYTES:
            raise ValueError(f"Unknown mamba_ssm_dtype: {ssm_dtype_raw!r}")
        state_b = _BASE_DTYPE_BYTES[ssm_name]
    else:
        state_b = base_b

    qraw = tc.get("quantization_config") or cfg.get("quantization_config")
    qcfg: Mapping[str, Any] = qraw if qraw is not None and type(qraw) is dict else {}
    if not qcfg:
        act_mode = "a8" if base_b <= 1.0 else "a16"
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
        )

    q_method = str(qcfg.get("quant_method") or qcfg.get("format") or "").lower()
    q_algo = str(qcfg.get("quant_algo") or "").lower()
    bits = int(qcfg.get("bits", 0) or 0)
    group_size: int | None = None
    if "group_size" in qcfg and qcfg.get("group_size") is not None:
        group_size = int(qcfg["group_size"])
    if type(qcfg.get("config_groups")) is dict:
        for grp in qcfg["config_groups"].values():
            if type(grp) is dict and type(grp.get("weights")) is dict:
                nb = int(grp["weights"].get("num_bits", 0) or 0)
                if nb > 0 and bits == 0:
                    bits = nb
                gs = grp["weights"].get("group_size")
                if gs is not None and group_size is None:
                    group_size = int(gs)

    if bits == 4:
        if group_size is None:
            raise ValueError(
                "4-bit quantization config missing required field: 'group_size'"
            )
        if group_size <= 0:
            raise ValueError(f"Invalid 4-bit group_size: {group_size}")

    ign_raw = (
        qcfg.get("modules_to_not_convert")
        or qcfg.get("ignore")
        or qcfg.get("ignored_layers")
    )
    has_ign_list = ign_raw is not None and len(tuple(ign_raw)) > 0
    ignored = tuple(str(x).lower() for x in (ign_raw or ()))
    attn_ignored = any(
        "linear_attn" not in x
        and any(
            seg in ("self_attn", "attn", "q_proj", "k_proj", "v_proj", "o_proj")
            for seg in x.replace("/", ".").split(".")
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

    block_sz = qcfg.get("weight_block_size")
    fp8_b = 1.0
    if block_sz is not None and type(block_sz) in (list, tuple) and len(block_sz) == 2:
        b0, b1 = int(block_sz[0]), int(block_sz[1])
        if b0 > 0 and b1 > 0:
            fp8_b = 1.0 + 4.0 / float(b0 * b1)

    scale_bytes: float | None = None
    if qcfg.get("scale_bytes") is not None:
        scale_bytes = float(qcfg["scale_bytes"])
    elif qcfg.get("scale_dtype") is not None:
        sd = str(qcfg["scale_dtype"]).lower().removeprefix("torch.")
        if sd not in _BASE_DTYPE_BYTES:
            raise ValueError(f"Unknown 4-bit scale_dtype: {sd!r}")
        scale_bytes = _BASE_DTYPE_BYTES[sd]
    elif type(qcfg.get("config_groups")) is dict:
        for grp in qcfg["config_groups"].values():
            if type(grp) is dict and type(grp.get("weights")) is dict:
                sb = grp["weights"].get("scale_bytes")
                if sb is not None and scale_bytes is None:
                    scale_bytes = float(sb)

    if "mxfp4" in q_method or "mxfp4" in q_algo:
        mxfp4_b = 0.5 + 1.0 / 32.0
        w_expert = mxfp4_b
        w_dense = (
            base_b
            if (attn_ignored or not qcfg.get("quantize_attn", False))
            else mxfp4_b
        )
    elif (
        "nvfp4" in q_method
        or "nvfp4" in q_algo
        or ("compressed" in q_method and "nvfp4" in str(qcfg).lower())
        or ("modelopt" in q_method and "nvfp4" in str(qcfg).lower())
    ):
        nvfp4_b = 0.5 + 1.0 / 16.0
        w_expert = nvfp4_b
        dense_override = str(qcfg.get("dense_dtype", "")).lower()
        if dense_override == "fp8":
            w_dense = fp8_b
        elif attn_ignored:
            w_dense = base_b
        else:
            w_dense = nvfp4_b
    elif bits == 4 and group_size is not None:
        if scale_bytes is None or scale_bytes <= 0.0:
            raise ValueError(
                "4-bit quantization config missing required field: "
                "'scale_bytes' or 'scale_dtype'"
            )
        w_expert = 0.5 + scale_bytes / float(group_size)
        w_dense = base_b if attn_ignored else w_expert
    elif "fp8" in q_method or bits == 8:
        w_expert = fp8_b
        w_dense = base_b if attn_ignored else fp8_b
    else:
        w_expert = base_b
        w_dense = base_b

    kv_scheme = str(
        qcfg.get("kv_cache_scheme") or qcfg.get("kv_cache_dtype") or ""
    ).lower()
    kv_b = 1.0 if "fp8" in kv_scheme or "8" in kv_scheme else base_b

    quant_conv = bool(qcfg.get("quantize_conv", False))
    conv_b = w_dense if quant_conv else base_b

    quant_router = bool(qcfg.get("quantize_router", False)) or (
        bool(qcfg.get("quantize_aux", False)) and not aux_ignored
    )
    router_b = w_dense if quant_router else base_b

    return Dtypes(
        w_dense=w_dense,
        w_expert=w_expert,
        w_aux=base_b if aux_ignored else w_dense,
        kv=kv_b,
        state=state_b,
        conv=conv_b,
        router=router_b,
        lm_head=base_b if lm_ignored else w_dense,
        act="a8",
    )
