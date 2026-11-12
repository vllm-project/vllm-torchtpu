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
"""Canonical constants for Qwen3.8-2.4T decode megakernel plugin."""

from __future__ import annotations

# Production and mini model geometry
PROD_NUM_LAYERS: int = 92
PROD_HIDDEN_SIZE: int = 8192
MINI_HIDDEN_SIZE: int = 128
EXPECTED_TP_SIZE: int = 32
EXPECTED_DCP_SIZE: int = 8
NVFP4_GROUP_SIZE: int = 16

MINI_NUM_LAYERS: int = 4
MINI_KV_HEADS: int = 4
MINI_HEAD_DIM: int = 128
MINI_ROPE_DIM: int = 32
MINI_GDN_K_HEADS: int = 32
MINI_EXPERTS: int = 16
MINI_TOP_K: int = 2
MINI_MAX_PAGES_PER_SEQ: int = 4
MINI_NUM_PAGES: int = 96
DEFAULT_ATT_EVERY: int = 4
DEFAULT_Q_HEADS: int = 128
DEFAULT_GDN_V_HEADS: int = 64
DEFAULT_CONV_KERNEL: int = 4
DEFAULT_EXPERT_INTER: int = 2048
DEFAULT_HIER_GROUP: int = 4

# Plan and layout constants
PAD2_SIZE: int = 2
MIN_MATRIX_NDIM: int = 2
MOE_WEIGHT_NDIM: int = 6
BF16_BYTES: int = 2
F32_BYTES: int = 4
I32_BYTES: int = 4
VMEM_ALIGN_BYTES: int = 128
STAGE_ROW_TILE: int = 32
MIB_BYTES: int = 1024 * 1024
TP_AXIS_NAME: str = "tp"
TPU_V7X_DEVICE_KIND: str = "TPU7x"

# H-tiled weight keys for 4D and 5D DMA
H_TILED_4D_KEYS: frozenset[str] = frozenset({"wqkv", "w_inproj", "w_se13"})
H_TILED_4D_SCALE_KEYS: frozenset[str] = frozenset({"wqkv_s", "w_inproj_s", "w_se13_s"})
H_TILED_5D_KEYS: frozenset[str] = frozenset({"w_e13"})
H_TILED_5D_SCALE_KEYS: frozenset[str] = frozenset({"w_e13_s"})
H_TILED_ALL_KEYS: frozenset[str] = (
    H_TILED_4D_KEYS | H_TILED_4D_SCALE_KEYS | H_TILED_5D_KEYS | H_TILED_5D_SCALE_KEYS
)

RESIDENT_WEIGHT_KEYS: frozenset[str] = frozenset(
    {
        "in_norm",
        "post_norm",
        "final_norm",
        "q_norm",
        "k_norm",
        "a_log",
        "dt_bias",
        "gdn_norm",
    }
)

# Kernel and buffer dimensions
PAGES_PER_GDN_SLOT: int = 5
GDN_REC_PAGES: int = 4
RANK_POOL_NDIM: int = 6
UNRANKED_POOL_NDIM: int = 5
GLOBAL_POOL_NDIM: int = 7
NORM_WQKV_NDIM: int = 5

IN_IDX_X: int = 0
IN_IDX_POOL: int = 3
IN_IDX_CONV: int = 4
IN_IDX_REC: int = 5
OUT_IDX_POOL: int = 1
OUT_IDX_CONV: int = 2
OUT_IDX_REC: int = 3
INPUT_OUTPUT_ALIASES: dict[int, int] = {
    IN_IDX_POOL: OUT_IDX_POOL,
    IN_IDX_CONV: OUT_IDX_CONV,
    IN_IDX_REC: OUT_IDX_REC,
}

# Weights extraction keys and aliases
WEIGHT_KEYS: tuple[str, ...] = (
    "in_norm",
    "post_norm",
    "final_norm",
    "wqkv",
    "wqkv_s",
    "wo",
    "wo_s",
    "q_norm",
    "k_norm",
    "w_inproj",
    "w_inproj_s",
    "w_ba",
    "w_conv",
    "a_log",
    "dt_bias",
    "gdn_norm",
    "w_out",
    "w_out_s",
    "w_gate",
    "w_se13",
    "w_se13_s",
    "w_se2",
    "w_se2_s",
    "w_se_gate",
    "w_e13",
    "w_e13_s",
    "w_e2",
    "w_e2_s",
)

U8_WEIGHT_KEYS: frozenset[str] = frozenset(
    {"wqkv", "wo", "w_inproj", "w_out", "w_se13", "w_se2", "w_e13", "w_e2"}
)
F32_WEIGHT_KEYS: frozenset[str] = frozenset({"a_log", "dt_bias"})
EXPERT_MARKER: str = "mlp.experts."
MIN_LAYER_PARTS: int = 2

EXPERT_ATTR_ALIASES: dict[str, tuple[str, ...]] = {
    "w13_weight": (
        "w13_weight",
        "gate_up_proj_weight",
        "gate_up_proj.weight",
        "gate_up_proj",
    ),
    "w13_weight_scale": (
        "w13_weight_scale",
        "gate_up_proj_scale",
        "gate_up_proj.weight_scale",
        "gate_up_proj.scale",
        "gate_up_proj_weight_scale",
    ),
    "w13_weight_scale_2": (
        "w13_weight_scale_2",
        "gate_up_proj_scale_2",
        "gate_up_proj.weight_scale_2",
        "gate_up_proj_weight_scale_2",
    ),
    "w2_weight": ("w2_weight", "down_proj_weight", "down_proj.weight", "down_proj"),
    "w2_weight_scale": (
        "w2_weight_scale",
        "down_proj_scale",
        "down_proj.weight_scale",
        "down_proj.scale",
        "down_proj_weight_scale",
    ),
    "w2_weight_scale_2": (
        "w2_weight_scale_2",
        "down_proj_scale_2",
        "down_proj_weight_scale_2",
    ),
}
EXPERT_ATTR_NAMES: frozenset[str] = frozenset(
    {
        "w13_weight",
        "w13_weight_scale",
        "w13_weight_scale_2",
        "w2_weight",
        "w2_weight_scale",
        "w2_weight_scale_2",
        "gate_up_proj.weight",
        "gate_up_proj.weight_scale",
        "gate_up_proj.weight_scale_2",
        "gate_up_proj",
        "gate_up_proj_scale",
        "gate_up_proj_scale_2",
        "down_proj.weight",
        "down_proj.weight_scale",
        "down_proj.weight_scale_2",
        "down_proj",
        "down_proj_scale",
        "down_proj_scale_2",
    }
)

# Adapter error codes and options
ERR_POOL_UNKNOWN: str = "pool_layout_unknown"
ERR_POOL_MISMATCH: str = "pool_layout_mismatch"
ERR_UNSUPPORTED_ARCH: str = "unsupported_arch"
ERR_UNSUPPORTED_GEOM: str = "unsupported_geometry"
ERR_UNSUPPORTED_QUANT: str = "unsupported_quant"
ERR_UNSUPPORTED_TP: str = "unsupported_tp_size"
ERR_UNSUPPORTED_DCP: str = "unsupported_dcp_size"
ERR_UNALIGNED_CACHE: str = "unaligned_cache_layout"
ERR_BACKBONE_NOT_FOUND: str = "backbone_not_found"
ERR_EMPTY_BUCKETS: str = "empty_buckets"
NUM_META_ARGS: int = 4
MINI_ALLOWED_LAYERS: frozenset[int] = frozenset({4, 8})
DEFAULT_SINGLE_SHAPE_DIM: int = 1

OPTION_KEYS: frozenset[str] = frozenset(
    {"buckets", "compile_hlo_check", "allow_mini_geometry", "interpret"}
)
DEFAULT_BUCKETS: tuple[int, ...] = (1, 2, 4, 8, 16)
SUPPORTED_ARCHS: frozenset[str] = frozenset(
    {"Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM"}
)
SUPPORTED_MODEL_TYPES: frozenset[str] = frozenset({"qwen3_5_moe", "qwen3_5_moe_text"})
PAGE_TOKENS: int = 128
MAMBA_SLOT_TOKENS: int = PAGES_PER_GDN_SLOT * PAGE_TOKENS
KV_CACHE_NDIM: int = 5
