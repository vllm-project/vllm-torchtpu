# Copyright 2025 Google LLC
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
"""Binary search over float32 bits for top-k and top-p masking.

Re-exports core primitives from `llm_tpu_core.runner.top_k_top_p`.
"""

from vllm_torchtpu.llm_tpu_core.runner.top_k_top_p import (
    MASKED_LOGIT_VALUE,
    _float32_bsearch,
    _int32_bsearch,
    _monotonic_int32_to_float32,
    _monotonic_int32_to_float32_bit_pattern,
    apply_top_k_top_p,
    topk_mask,
    topp_mask,
)

__all__ = [
    "MASKED_LOGIT_VALUE",
    "_float32_bsearch",
    "_int32_bsearch",
    "_monotonic_int32_to_float32",
    "_monotonic_int32_to_float32_bit_pattern",
    "apply_top_k_top_p",
    "topk_mask",
    "topp_mask",
]
