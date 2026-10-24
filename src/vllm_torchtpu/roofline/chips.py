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
"""Per-TensorCore hardware specifications mirroring JAX tpu_info."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ChipSpec:
    """Hardware ceiling parameters for one TensorCore."""

    name: str
    hbm_bytes: int
    hbm_bw: float
    bf16_flops: float
    fp8_flops: float
    vmem_bytes: int
    ici_bw: float = math.inf


CHIPS: dict[str, ChipSpec] = {
    "tpu7x": ChipSpec(
        name="tpu7x",
        hbm_bytes=103_000_000_000,
        hbm_bw=3.7e12,
        bf16_flops=1.155e15,
        fp8_flops=2.3e15,
        vmem_bytes=67_108_864,
        # per-chip ICI egress of 560.04 GB/s split over 2 TensorCores
        ici_bw=280.02e9,
    ),
}
