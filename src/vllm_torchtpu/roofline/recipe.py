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
"""Parse vllm serve arguments and environment variables into a roofline plan."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from .modules import Dtypes, Parallel

_KV_CACHE_DTYPE_BYTES: dict[str, float] = {
    "fp8": 1.0,
    "fp8_e4m3": 1.0,
    "fp8_e5m2": 1.0,
    "auto": 2.0,
    "bfloat16": 2.0,
    "bf16": 2.0,
    "float16": 2.0,
    "fp16": 2.0,
}


def plan_from_serve_args(
    argv: Sequence[str],
    env: Mapping[str, str],
) -> tuple[Parallel, Dtypes]:
    """Derive Parallel and Dtypes from vllm serve CLI flags and container env."""
    tp = 1
    moe: Literal["tp_slice", "ep"] = "tp_slice"
    kv_split = 1
    kv_bytes = 1.0

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

        if flag in ("--tensor-parallel-size", "-tp"):
            tp = int(val)
            i += step
        elif flag == "--enable-expert-parallel":
            moe = "ep"
            i += 1
        elif flag == "--kv-cache-dtype":
            kv_bytes = _KV_CACHE_DTYPE_BYTES.get(val.lower(), 2.0)
            i += step
        elif flag == "--decode-context-parallel-size":
            kv_split = int(val)
            i += step
        else:
            i += 1

    residual: Literal["replicated", "sp"] = (
        "sp" if env.get("TPU_MOE_SEQUENCE_PARALLEL") == "1" else "replicated"
    )
    moe_dispatch: Literal["allreduce", "a2a_push"] = (
        "a2a_push" if env.get("USE_MOE_FUSED_EP_KERNEL") == "1" else "allreduce"
    )

    par = Parallel(
        tp=tp,
        kv_split=kv_split,
        kv_merge="ag_rs",
        moe=moe,
        moe_dispatch=moe_dispatch,
        residual=residual,
    )
    dt = Dtypes(kv=kv_bytes)
    return par, dt
