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
"""Test, mock, and oracle constants for Qwen3.8-2.4T megakernel tests."""

from __future__ import annotations

# Mock pool and slot dimensions
PAGES_PER_MAMBA_SLOT: int = 5
REC_PAGES_PER_MAMBA_SLOT: int = 4
GLOBAL_POOL_NDIM: int = 7
RANK_POOL_NDIM: int = 6
CONV_BF16_BYTES: int = 2

# Production reference dimensions for mock model
PROD_Q_HEADS: int = 128
PROD_KV_HEADS: int = 4
PROD_HEAD_DIM: int = 256
PROD_NUM_LAYERS: int = 92

# Oracle numerical parameters
SOFTPLUS_CUTOFF: float = 30.0
ROUTER_WEIGHT_SUM_TOL: float = 1e-5

# Test expected sizes and counts
EXPECTED_BLOCK_SIZE: int = 128
EXPECTED_PAGE_SIZE: int = 128
EXPECTED_DCP_INTERLEAVE_SIZE: int = 128
EXPECTED_MAMBA_SLOT_TOKENS: int = 640
EXPECTED_SALT_LEN: int = 16
EXPECTED_STREAM_ITEMS: int = 5
EXPECTED_TWO_RANKS: int = 2
EXPECTED_BATCH_2: int = 2
EXPECTED_L8_REGIONS: int = 2
EXPECTED_L8_OUT_ROWS: int = 2
EXPECTED_PREFILL_TOKENS: int = 2
EXPECTED_BUCKET_4: int = 4
EXPECTED_RANKS: int = 32
EXPECTED_DCP: int = 8
EXPECTED_LAYERS: int = 92
EXPECTED_KV_HEADS: int = 4
EXPECTED_VH: int = 64
EXPECTED_WEIGHT_KEYS: int = 28

# Custom byte markers and tolerances
CUSTOM_W13_BYTE: int = 0x33
CUSTOM_L1_W2_BYTE: int = 0x55
MIN_CPU_DEVICES: int = 32
MIN_TILED_DIM: int = 2
MAX_KV_ULP_TOLERANCE: int = 2

# Package and weight sets
PKG: str = "vllm_torchtpu.models.qwen38_2_4t.megakernel"
U8_WEIGHT_KEYS: frozenset[str] = frozenset(
    {"wqkv", "wo", "w_inproj", "w_out", "w_se13", "w_se2", "w_e13", "w_e2"}
)
