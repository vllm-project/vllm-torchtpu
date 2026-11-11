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
"""Muse Spark Open 816B A42B Single-Dispatch Pallas Decode Megakernel package."""

from __future__ import annotations

from .config import (
    F32,
    HIGHEST,
    KV_GROUP,
    NVFP4_LUT,
    PAGE_TOKENS,
    SUBPAGE_TOKENS,
    SW_RING,
    TOPK_OUT,
    TP_AXIS_NAME,
    Config,
    MuseMegakernelConfig,
    dequantize_nvfp4,
    residual_coefficients,
)
from .kernel import (
    _make_decode_block,
    empty_hybrid_kv_cache,
    logical_mesh,
    make_decode,
    make_decode_batch,
    make_hybrid_decode_step,
    make_prefill,
    make_verify,
    make_verify_block,
    prepare_decode_weights,
    prepare_verify_cache,
    prepare_verify_weights,
)
from .load import (
    KERNEL_WEIGHT_NAMES,
    Checkpoint,
    canonical_test_weights,
    lm_pad,
    lm_width,
    load_sharded_weights,
    prepare_tp8_weights,
    router_order_indices,
)

__all__ = [
    "Checkpoint",
    "Config",
    "F32",
    "HIGHEST",
    "KERNEL_WEIGHT_NAMES",
    "KV_GROUP",
    "MuseMegakernelConfig",
    "NVFP4_LUT",
    "PAGE_TOKENS",
    "SUBPAGE_TOKENS",
    "SW_RING",
    "TOPK_OUT",
    "TP_AXIS_NAME",
    "_make_decode_block",
    "canonical_test_weights",
    "dequantize_nvfp4",
    "empty_hybrid_kv_cache",
    "lm_pad",
    "lm_width",
    "load_sharded_weights",
    "logical_mesh",
    "make_decode",
    "make_decode_batch",
    "make_hybrid_decode_step",
    "make_prefill",
    "make_verify",
    "make_verify_block",
    "prepare_decode_weights",
    "prepare_tp8_weights",
    "prepare_verify_cache",
    "prepare_verify_weights",
    "residual_coefficients",
    "router_order_indices",
]
