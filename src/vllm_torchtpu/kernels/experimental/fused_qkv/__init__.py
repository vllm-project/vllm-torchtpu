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
"""Fused attention and GDN prologue Pallas kernels."""

from __future__ import annotations

from vllm_torchtpu.kernels.experimental.fused_qkv.kernel.configs import (
    AttentionConfig,
    GdnConfig,
    Norm,
    QkvWeights,
    Rope,
)
from vllm_torchtpu.kernels.experimental.fused_qkv.kernel.gdn import (
    GdnWeights,
    fused_gdn,
)
from vllm_torchtpu.kernels.experimental.fused_qkv.kernel.rpa import (
    fused_qkv,
)

__all__ = [
    "AttentionConfig",
    "GdnConfig",
    "GdnWeights",
    "Norm",
    "QkvWeights",
    "Rope",
    "fused_gdn",
    "fused_qkv",
]
