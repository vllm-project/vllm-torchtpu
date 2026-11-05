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
"""Row-block sizing helpers for row-wise Pallas kernels."""

import math

from jax.experimental.pallas import tpu as pltpu

_CPU_FALLBACK_VMEM_BYTES = 32 * 1024 * 1024
_SUBLANE_ALIGNMENT = 8
_BUFFER_FRACTION = 0.8


def get_vmem_capacity_bytes() -> int:
    """Return active TPU core VMEM capacity in bytes, or a 32 MiB CPU fallback."""
    try:
        return int(pltpu.get_tpu_info().vmem_capacity_bytes)
    except ValueError:
        return _CPU_FALLBACK_VMEM_BYTES


def fit_block_rows(
    num_rows: int,
    buffer_bytes_per_row: int,
    max_rows: int,
    vmem_limit_bytes: int | None = None,
) -> int:
    """Select the largest power-of-two row block dividing ``num_rows`` that fits VMEM.

    Leaves ``1 - _BUFFER_FRACTION`` of ``vmem_limit_bytes`` for compiler
    temporaries.
    """
    if max_rows < _SUBLANE_ALIGNMENT or max_rows & (max_rows - 1):
        raise ValueError(
            f"max_rows must be a power of two >= {_SUBLANE_ALIGNMENT}, got {max_rows}"
        )
    if num_rows <= 0 or num_rows % _SUBLANE_ALIGNMENT != 0:
        raise ValueError(
            f"num_rows must be a positive multiple of {_SUBLANE_ALIGNMENT}, "
            f"got {num_rows}"
        )
    if vmem_limit_bytes is None:
        vmem_limit_bytes = get_vmem_capacity_bytes()
    rows = math.gcd(num_rows, max_rows)
    while (
        rows > _SUBLANE_ALIGNMENT
        and rows * buffer_bytes_per_row > _BUFFER_FRACTION * vmem_limit_bytes
    ):
        rows //= 2
    return rows
