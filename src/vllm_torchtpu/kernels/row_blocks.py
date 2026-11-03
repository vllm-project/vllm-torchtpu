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
"""Row blocks of the row-wise fused kernels (norm_quant, gelu_quant)."""

import math

# Share of a kernel's VMEM limit its double-buffered row blocks may take; the
# rest is left to the compiler's temporaries.
_BUFFER_FRACTION = 0.8


def fit_block_rows(
    num_rows: int, buffer_bytes_per_row: int, max_rows: int, vmem_limit_bytes: int
) -> int:
    """The largest power of two that divides `num_rows` (buckets are multiples
    of 16), is at most `max_rows`, and whose double-buffered row blocks
    (`buffer_bytes_per_row` per row, both slots counted) fit in
    `_BUFFER_FRACTION` of `vmem_limit_bytes`."""
    if max_rows < 8 or max_rows & (max_rows - 1):
        raise ValueError(f"block rows must be a power of two >= 8, got {max_rows}")
    rows = math.gcd(num_rows, max_rows)
    while (
        rows > 8 and rows * buffer_bytes_per_row > _BUFFER_FRACTION * vmem_limit_bytes
    ):
        rows //= 2
    return rows
