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

# (chip version, local rows, experts, topk). Two-table sqrtsoftplus, BF16 logits;
# JAX 0.11.2 / libtpu 0.0.50. Packed normalization: 1.440576 us selector,
# 7.978391 us isolated router (synthetic input; no model-level latency claim).
NORMALIZED_ROUTER_BLOCK_ROWS: dict[tuple[pltpu.ChipVersion, int, int, int], int] = {
    (pltpu.ChipVersion.TPU_7X, 288, 384, 6): 288,
}
