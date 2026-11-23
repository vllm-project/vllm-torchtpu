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
"""Shared byte-unit, topology, and layout constants for Kimi-K3 megakernels."""

from __future__ import annotations

# Memory and bit-width
MIB_BYTES: int = 1 << 20
GIB_BYTES: int = 1 << 30
BITS_PER_WORD: int = 32

# Tiling and lane geometry
SUBLANE_ROWS: int = 8
VREG_LANE_COUNT: int = 128
STORE_TILE: int = 128

# Mesh and parallelism
AXIS_TP: str = "tp"
AXIS_SP: str = "k3_sp"
TP_SIZE: int = 32
TP_PEERS: int = TP_SIZE - 1
DP2_GROUPS: int = 2
DP4_GROUPS: int = DP2_GROUPS * 2

# Model architecture
NUM_LAYERS: int = 93
HIDDEN_SIZE: int = 7168
# Kimi-K3 released checkpoint vocab_size.
DEFAULT_VOCAB_SIZE: int = 163840
HEAD_DIM: int = 128
# 96 KDA heads partitioned across 32 tensor-parallel ranks.
KDA_HEADS: int = 3
# KDA Q, K, V convolution projection parts.
KDA_QKV_PARTS: int = 3
# Causal short-convolution tap count per KDA layer.
CONV_TAPS: int = 3
# KDA head axis padded from 3 to 4 rows in pooled state blocks.
CONV_HEAD_ROWS: int = 4
# Tokens stored per paged KV cache block (two 128-token read tiles).
PAGE_TOKENS: int = 256
# 512 KV-LoRA and 64 RoPE dims padded to 5 lane blocks of 128.
MLA_LATENT_ROW_LANES: int = 5 * VREG_LANE_COUNT
# Layer index stride between MLA layers in the 4-layer repeating block.
MLA_LAYER_STRIDE: int = 4
# Remainder selecting the MLA layer within each 4-layer block.
MLA_LAYER_MODULO: int = 3
# Routed MoE expert count per layer in Kimi-K3.
ROUTED_EXPERTS: int = 896
PREFILL_WINDOWS: tuple[int, ...] = (512, 1024, 2048, 4096, 8192)
OVERFLOW_WINDOW_TOKENS: int = PREFILL_WINDOWS[-1]

# Decode batch schedules
# Power-of-two batch buckets shared by decode and program configuration.
POWER_OF_TWO_BATCHES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
# Valid batch buckets under two-group attention data parallelism.
DP_VALID_BATCHES: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 96, 128)
# Batch buckets eligible for token-scattered decode execution.
TOKEN_SCATTER_BATCHES: tuple[int, ...] = (8, 12, 16, 20, 24, 28, 32, 64, 96, 128)

# Draft and shard layout
# Padded intermediate width of the DSpark draft MLP projections.
MLP_PADDED: int = 512
DSPARK_HIDDEN_CHUNK: int = HIDDEN_SIZE // 8
# Concatenated width of 5 auxiliary hidden states split across 4 TP8 hosts.
FC_IN: int = 5 * HIDDEN_SIZE // 4
# Packed 256-column row blocks across HIDDEN_SIZE.
ROW_PACKED: int = HIDDEN_SIZE // 256
KERNEL_SHARD_FORMAT: int = 1
EXPERT_FAMILIES: tuple[str, ...] = (
    "expert_gu",
    "expert_gus",
    "expert_down",
    "expert_ds",
)
# Leading characters of the resolved checkpoint directory name in cache keys.
REV_PREFIX_LEN: int = 8
# Extra RoPE table positions allocated beyond max_position for speculation.
ROPE_POS_SLACK: int = 96
# Maximum total token rows processed per DSpark MLA draft call.
DSPARK_MLA_MAX_ROWS: int = 64

__all__ = [
    "AXIS_SP",
    "AXIS_TP",
    "BITS_PER_WORD",
    "CONV_HEAD_ROWS",
    "CONV_TAPS",
    "DEFAULT_VOCAB_SIZE",
    "DP2_GROUPS",
    "DP4_GROUPS",
    "DP_VALID_BATCHES",
    "DSPARK_HIDDEN_CHUNK",
    "DSPARK_MLA_MAX_ROWS",
    "EXPERT_FAMILIES",
    "FC_IN",
    "GIB_BYTES",
    "HEAD_DIM",
    "HIDDEN_SIZE",
    "KDA_HEADS",
    "KDA_QKV_PARTS",
    "KERNEL_SHARD_FORMAT",
    "MIB_BYTES",
    "MLA_LATENT_ROW_LANES",
    "MLA_LAYER_MODULO",
    "MLA_LAYER_STRIDE",
    "MLP_PADDED",
    "NUM_LAYERS",
    "OVERFLOW_WINDOW_TOKENS",
    "PAGE_TOKENS",
    "POWER_OF_TWO_BATCHES",
    "PREFILL_WINDOWS",
    "REV_PREFIX_LEN",
    "ROPE_POS_SLACK",
    "ROUTED_EXPERTS",
    "ROW_PACKED",
    "STORE_TILE",
    "SUBLANE_ROWS",
    "TOKEN_SCATTER_BATCHES",
    "TP_PEERS",
    "TP_SIZE",
    "VREG_LANE_COUNT",
]
