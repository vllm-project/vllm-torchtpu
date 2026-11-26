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
"""Token unpacking for the MSA caches' packed pages.

The caches use the packed MLA layout ``[pages, S / p, p, W]``: `p` tokens
share each 32-bit word of a row (2 for bf16, 4 for fp8). Unpacking yields
`p` arrays of ``[S / p, W]``; array `j` holds tokens ``j, p + j, 2p + j, ...``.
"""

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

# Cache dtypes the MSA kernels read, by tokens per 32-bit word.
SUPPORTED_DTYPES = {2: jnp.bfloat16, 4: jnp.float8_e4m3fn}
_UINT_BY_BITS = {16: jnp.uint16, 8: jnp.uint8}


def check_page_dtype(packing: int, dtype) -> None:
    if SUPPORTED_DTYPES.get(packing) != dtype:
        raise NotImplementedError(
            f"MSA caches support {list(SUPPORTED_DTYPES.values())}, got {dtype} "
            f"with {packing} tokens per word."
        )


def unpack_page(page_ref) -> list[jax.Array]:
    """``[1, S / p, p, W]`` page -> `p` bf16 arrays ``[S / p, W]``."""
    packing = page_ref.shape[2]
    bits = 32 // packing
    # Same rank in and out: Mosaic bitcasts the packed axis p -> 1.
    packed = page_ref.bitcast(jnp.uint32)[0, :, 0, :]
    mask = (1 << bits) - 1
    return [
        pltpu.bitcast(
            ((packed >> (bits * j)) & mask).astype(_UINT_BY_BITS[bits]),
            page_ref.dtype,
        ).astype(jnp.bfloat16)
        for j in range(packing)
    ]


def slot_ids(rows: int, page_size: int, packing: int) -> jax.Array:
    """Slot of each column when a page's unpacked arrays are concatenated."""
    lane = jax.lax.broadcasted_iota(jnp.int32, (rows, page_size), 1)
    per_array = page_size // packing
    return (lane % per_array) * packing + lane // per_array
