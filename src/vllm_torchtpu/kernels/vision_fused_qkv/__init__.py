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

"""Vision Encoder Fused QKV Projection and 2D mRoPE Kernel Package."""

from vllm_torchtpu.kernels.vision_fused_qkv.configs import (
    Norm,
    VisionAttentionConfig,
    VisionMropeSpec,
)
from vllm_torchtpu.kernels.vision_fused_qkv.kernel import vision_fused_qkv_project
from vllm_torchtpu.kernels.vision_fused_qkv.mrope import (
    LaneRope2D,
    compute_2d_mrope_angles,
    compute_inv_freq,
)
from vllm_torchtpu.kernels.vision_fused_qkv.reference import reference_neox_2d_mrope

__all__ = [
    "Norm",
    "VisionAttentionConfig",
    "VisionMropeSpec",
    "vision_fused_qkv_project",
    "LaneRope2D",
    "compute_2d_mrope_angles",
    "compute_inv_freq",
    "reference_neox_2d_mrope",
]
