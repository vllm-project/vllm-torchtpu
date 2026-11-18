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
"""Fused Pallas mHC kernels for GLM 5.3 Flash."""

from vllm_torchtpu.kernels.glm53_mhc.kernel import (
    DEFAULT_VMEM_LIMIT_BYTES,
    SUBLANE,
    mhc_fused_post_pre_norm,
    mhc_pre_norm_fused,
    pad_to,
    round_up,
    select_fused_token_block,
    select_token_block,
    split_fn3,
    trim_to,
)

__all__ = [
    "DEFAULT_VMEM_LIMIT_BYTES",
    "SUBLANE",
    "mhc_fused_post_pre_norm",
    "mhc_pre_norm_fused",
    "pad_to",
    "round_up",
    "select_fused_token_block",
    "select_token_block",
    "split_fn3",
    "trim_to",
]
