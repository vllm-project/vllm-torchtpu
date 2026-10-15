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
"""MLA query absorption `q_nope @ W_UK`, read and written token-major (Pallas).

The MLA attention kernels take q token-major, `[T, H, kv_lora_rank]`, but a
batched matmul over heads is head-major: XLA transposes q_nope to
`[H, T, nope]`, multiplies, and transposes the result back. This kernel reads
q_nope and writes the absorbed q token-major.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Tokens per grid step, measured fastest at 64 heads x 512; 256 does not fit
# VMEM beside W_UK.
MAX_BLOCK_TOKENS = 128
# Below this, XLA's matmul is as fast: it prefetches W_UK into VMEM under the
# previous op, while the kernel has to read it itself (measured at decode).
MIN_TOKENS = 256


def _kernel(q_ref, w_ref, o_ref):
    q = pltpu.einshape("thd->htd", q_ref[...])
    out = [
        jnp.dot(q[h], w_ref[h], preferred_element_type=jnp.float32)
        for h in range(q.shape[0])
    ]
    o_ref[...] = pltpu.einshape("htd->thd", jnp.stack(out).astype(o_ref.dtype))


def absorb_q(q_nope: jax.Array, w_uk_t: jax.Array) -> jax.Array:
    """`out[t, h] = q_nope[t, h] @ w_uk_t[h]`.

    Args:
      q_nope: `[T, H, nope]`.
      w_uk_t: `[H, nope, R]`.

    Returns:
      `[T, H, R]` in `q_nope.dtype`.
    """
    tokens, heads, nope = q_nope.shape
    rank = w_uk_t.shape[2]
    if tokens < MIN_TOKENS:
        out = jnp.einsum(
            "thd,hdr->thr", q_nope, w_uk_t, preferred_element_type=jnp.float32
        )
        return out.astype(q_nope.dtype)
    block = min(tokens, MAX_BLOCK_TOKENS)
    return pl.pallas_call(
        _kernel,
        grid=(pl.cdiv(tokens, block),),
        in_specs=[
            pl.BlockSpec((block, heads, nope), lambda i: (i, 0, 0)),
            # Read once: the index never changes.
            pl.BlockSpec(
                w_uk_t.shape, lambda i: (0, 0, 0), pipeline_mode=pl.Buffered(1)
            ),
        ],
        out_specs=pl.BlockSpec((block, heads, rank), lambda i: (i, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((tokens, heads, rank), q_nope.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes * 7 // 8,
        ),
    )(q_nope, w_uk_t)
