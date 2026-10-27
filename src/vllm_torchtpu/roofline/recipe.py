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
"""Parse vLLM and SGLang server CLI arguments and environment into a roofline plan."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Literal

from .modules import Dtypes, Parallel

_KV_CACHE_DTYPE_BYTES: dict[str, float] = {
    "fp8": 1.0,
    "fp8_e4m3": 1.0,
    "fp8_e5m2": 1.0,
    "bfloat16": 2.0,
    "bf16": 2.0,
    "float16": 2.0,
    "fp16": 2.0,
}


def plan_from_serve_args(
    argv: Sequence[str],
    env: Mapping[str, str],
    *,
    model_dtype_bytes: float = 2.0,
) -> tuple[Parallel, Dtypes]:
    """Derive Parallel and Dtypes from vLLM or SGLang CLI flags and container env."""
    tp = 1
    dp = 1
    ep = 1
    pp = 1
    pcp = 1
    moe: Literal["tp_slice", "ep"] = "tp_slice"
    dp_attn = False
    kv_split = 1
    kv_bytes = model_dtype_bytes  # auto / omitted preserves model torch_dtype

    tokens = list(argv)
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if "=" in tok and tok.startswith("-"):
            flag, val = tok.split("=", 1)
            step = 1
        else:
            flag = tok
            val = tokens[i + 1] if i + 1 < len(tokens) else ""
            step = 2

        if flag in ("--tensor-parallel-size", "-tp", "--tp-size"):
            tp = int(val)
            i += step
        elif flag in ("--data-parallel-size", "-dp", "--dp-size"):
            dp = int(val)
            i += step
        elif flag == "--ep-size":
            ep = int(val)
            if ep > 1:
                moe = "ep"
            i += step
        elif flag in ("--pipeline-parallel-size", "--pp-size"):
            pp = int(val)
            i += step
        elif flag == "--prefill-context-parallel-size":
            pcp = int(val)
            i += step
        elif flag in ("--enable-expert-parallel", "--enable-ep-moe"):
            moe = "ep"
            i += 1
        elif flag == "--enable-dp-attention":
            dp_attn = True
            i += 1
        elif flag == "--kv-cache-dtype":
            low = val.lower()
            kv_bytes = (
                model_dtype_bytes
                if low == "auto"
                else _KV_CACHE_DTYPE_BYTES.get(low, model_dtype_bytes)
            )
            i += step
        elif flag == "--decode-context-parallel-size":
            kv_split = int(val)
            i += step
        else:
            i += 1

    residual: Literal["replicated", "sp"] = (
        "sp" if env.get("TPU_MOE_SEQUENCE_PARALLEL") == "1" else "replicated"
    )
    moe_dispatch: Literal["allreduce", "a2a_push", "ag_ep", "fused_rs"]
    if dp_attn and dp > 1:
        total_ranks = tp
        tp = max(1, total_ranks // dp)
        if moe == "ep" and ep <= 1:
            ep = total_ranks
    elif moe == "ep" and ep <= 1 and dp > 1:
        ep = tp * dp
    if env.get("SGLANG_FUSED_RS") == "1":
        moe_dispatch = "fused_rs"
    elif env.get("USE_MOE_FUSED_EP_KERNEL") == "1":
        moe_dispatch = "a2a_push"
    elif moe == "ep" and (ep > tp or ep >= 16):
        moe_dispatch = "ag_ep"
    else:
        moe_dispatch = "allreduce"

    par = Parallel(
        tp=tp,
        kv_split=kv_split,
        kv_merge="ag_rs",
        moe=moe,
        moe_dispatch=moe_dispatch,
        residual=residual,
        dp=dp,
        ep=ep,
        pp=pp,
        pcp=pcp,
    )
    dt = Dtypes(kv=kv_bytes)
    return par, dt


def memory_utilization_from_serve_args(
    argv: Sequence[str], default: float = 0.90
) -> float:
    """Extract engine memory utilization fraction from vLLM or SGLang serve flags."""
    for i, arg in enumerate(argv):
        if arg.startswith("--gpu-memory-utilization="):
            val = float(arg.split("=", 1)[1])
            if 0.0 < val <= 1.0:
                return val
            raise ValueError(f"--gpu-memory-utilization must be in (0, 1], got {val}")
        if arg == "--gpu-memory-utilization" and i + 1 < len(argv):
            val = float(argv[i + 1])
            if 0.0 < val <= 1.0:
                return val
            raise ValueError(f"--gpu-memory-utilization must be in (0, 1], got {val}")
        if arg.startswith("--mem-fraction-static="):
            val = float(arg.split("=", 1)[1])
            if 0.0 < val <= 1.0:
                return val
            raise ValueError(f"--mem-fraction-static must be in (0, 1], got {val}")
        if arg == "--mem-fraction-static" and i + 1 < len(argv):
            val = float(argv[i + 1])
            if 0.0 < val <= 1.0:
                return val
            raise ValueError(f"--mem-fraction-static must be in (0, 1], got {val}")
    return default


def spec_window_from_serve_args(argv: Sequence[str]) -> int:
    """Draft tokens per step from ``--speculative-config`` (0 when absent)."""
    tokens = list(argv)
    for i, tok in enumerate(tokens):
        flag, _, val = tok.partition("=")
        if flag == "--speculative-config":
            raw = val if val else tokens[i + 1]
            return int(json.loads(raw)["num_speculative_tokens"])
    return 0
