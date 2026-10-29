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
"""Model specifications and reference oracles built on the TPU megakernel library."""

from . import (
    qwen35,
    qwen35_candidate30,
    qwen35_pool,
    qwen35_profile,
    qwen38_moe,
    qwen38_moe_profile,
)
from .qwen35 import MKConfig, Qwen35, Qwen35Geometry, qwen35_spec
from .qwen38_moe import Qwen38Moe, Qwen38MoeGeometry, qwen38_moe_spec

__all__ = [
    "MKConfig",
    "Qwen35",
    "Qwen35Geometry",
    "Qwen38Moe",
    "Qwen38MoeGeometry",
    "qwen35",
    "qwen35_candidate30",
    "qwen35_pool",
    "qwen35_profile",
    "qwen35_spec",
    "qwen38_moe",
    "qwen38_moe_profile",
    "qwen38_moe_spec",
]
