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

"""Experimental TPU7x sparse decode.

JAX-facing API; page lookup indirectly imports the existing MLA/Torch helper.
Callers guarantee valid group/page IDs and lengths, and thread returned cache
ownership through outer jit donation. Shape eligibility does not inspect IDs.
"""

import jax

from .decode import execute_decode as _decode
from .eligibility import is_supported_by_fused_decode as is_supported_by_fused_decode
from .eligibility import plan_decode as plan_decode


def qsa_decode_attention(
    q: jax.Array,
    gate: jax.Array,
    blocks: jax.Array,
    lengths: jax.Array,
    table: jax.Array,
    cache: jax.Array,
    *,
    kv_heads: int,
    ratio: int = 4,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    launch = plan_decode(
        q,
        gate,
        blocks,
        lengths,
        table,
        cache,
        kv_heads=kv_heads,
        ratio=ratio,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    if launch is None:
        raise NotImplementedError(
            "Fused decode requires TPU7x, divisible Q/KV heads, "
            "lane-aligned dimensions, "
            "a ratio dividing the KV tile, a launch fitting SMEM/VMEM, "
            "BF16 query/gate, native BF16/FP8 cache, positive ratio-aligned "
            "pages and valid nonempty int32 metadata; use RPA otherwise."
        )
    return _decode(
        q,
        gate,
        blocks,
        lengths,
        table,
        cache,
        launch=launch,
        k_scale=k_scale,
        v_scale=v_scale,
    )
