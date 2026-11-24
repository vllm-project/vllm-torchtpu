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
"""Measured tiles for fused EP MoE routing."""

from jax.experimental.pallas import tpu as pltpu

# (chip version, local rows, experts, topk). Two-table sqrtsoftplus, FP32 scoring;
# JAX 0.11.2 / libtpu 0.0.50; calibrated with isolated router measurements.
NORMALIZED_ROUTER_BLOCK_ROWS: dict[tuple[pltpu.ChipVersion, int, int, int], int] = {
    (pltpu.ChipVersion.TPU_7X, 288, 384, 6): 288,
}

# Independent input mechanisms, keyed by
# (chip, tile rows, hidden, padded intermediate, scale group).
# Input3 needs one index window per tile; layout caching needs every emitted
# height rung divisible by its row chunk. Neither mechanism requires the other.
INPUT_SLOTS: dict[tuple[pltpu.ChipVersion, int, int, int, int], int] = {
    (pltpu.ChipVersion.TPU_7X, 128, 5120, 2560, 512): 3,
}
# The smallest per-shard slab (host.ragged_stride_bound of the token bucket)
# at which INPUT_SLOTS applies. Three interleaved input slots pay at large
# buckets and cost at small ones. Measured at the DeepSeek-V4.1 shape on v7x,
# EP8, 3 slots vs 2 per MoE call, outputs bit-identical: 32..512 tokens per
# shard +73%..+6% (even routing), 1024 -2.5%, 2048 -5.3%. 49,664 rows is the
# 1024-tokens-per-shard bucket at EP8, top-6, 384 experts.
INPUT_SLOTS_MIN_ROWS: dict[tuple[pltpu.ChipVersion, int, int, int, int], int] = {
    (pltpu.ChipVersion.TPU_7X, 128, 5120, 2560, 512): 49664,
}
INPUT_LAYOUT_ROWS: dict[tuple[pltpu.ChipVersion, int, int, int, int], int] = {
    (pltpu.ChipVersion.TPU_7X, 128, 5120, 2560, 512): 32,
}
