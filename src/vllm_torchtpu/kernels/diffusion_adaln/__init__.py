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
"""Fused Pallas AdaLN modulation and gated residual kernels for diffusion DiTs."""

from vllm_torchtpu.kernels.diffusion_adaln.kernel import (
    indexed_gate_pallas,
    indexed_gate_rms_norm_scale_shift_pallas,
    indexed_scale_shift_pallas,
    rms_norm_indexed_scale_shift_pallas,
)

__all__ = [
    "indexed_gate_pallas",
    "indexed_gate_rms_norm_scale_shift_pallas",
    "indexed_scale_shift_pallas",
    "rms_norm_indexed_scale_shift_pallas",
]
