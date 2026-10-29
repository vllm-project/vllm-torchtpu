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
"""Backwards-compatible `megakernel.ops` package re-exporting Layer 1 VREG ops."""

from . import attention, base, gdn, quant
from .attention import (
    combine_bf16_lse,
    online_softmax_lse_merge,
    online_softmax_stats_merge,
    split_bf16_lse,
)
from .base import (
    BF16,
    F32,
    FP8,
    I32,
    U8,
    U32,
    bswitch,
    rms_zc,
    rmsnorm,
    rnd,
    rope_partial,
    rup,
    scope_fn,
    silu_mul,
)
from .gdn import gdn_recurrence_step, kda_step
from .quant import (
    FP8_E4M3_MAX,
    nvfp4_dot_subtiled,
    unpack_e8m0_scale,
    unpack_mxfp4_u32_bf16,
    unpack_mxfp4_u32_fp8,
    unpack_nvfp4_tile,
    unpack_nvfp4_tile_fp8,
)

__all__ = [
    "BF16",
    "F32",
    "FP8",
    "FP8_E4M3_MAX",
    "I32",
    "U8",
    "U32",
    "attention",
    "base",
    "bswitch",
    "combine_bf16_lse",
    "gdn",
    "gdn_recurrence_step",
    "kda_step",
    "nvfp4_dot_subtiled",
    "online_softmax_lse_merge",
    "online_softmax_stats_merge",
    "quant",
    "rms_zc",
    "rmsnorm",
    "rnd",
    "rope_partial",
    "rup",
    "scope_fn",
    "silu_mul",
    "split_bf16_lse",
    "unpack_e8m0_scale",
    "unpack_mxfp4_u32_bf16",
    "unpack_mxfp4_u32_fp8",
    "unpack_nvfp4_tile",
    "unpack_nvfp4_tile_fp8",
]
