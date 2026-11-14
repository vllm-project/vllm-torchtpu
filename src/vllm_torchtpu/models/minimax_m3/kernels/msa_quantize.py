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
"""Rotate and quantize a step's K/V heads into 4-bit cache rows in one Pallas
pass: `msa_fp4.quantize(msa_fp4.rotate_keys(x))` of every head of ``k`` and
``v``, written as the cache rows ``[K_0 | V_0 | ... | K_G-1 | V_G-1]``.

As XLA fusions (8K tokens, 4 heads, c64 twin j-9a9fccc2) the rotation wrote
f32 keys and values (16 MB each) for the quantize to read back, and the
interleave of the nibbles into rows was another copy: ~50 us per layer.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4

BLOCK_TOKENS = 512
# Steps of at least this many tokens use the kernel (`msa.write_fp4`); decode
# steps keep the XLA fusions (as `msa_dcp_heads.HEAD_PARALLEL_MIN_TOKENS`).
MIN_TOKENS = 2048


def _kernel(h_ref, k_ref, v_ref, rows_ref, scales_ref, *, heads, head_dim):
    for h in range(heads):
        for i, x_ref in enumerate((k_ref, v_ref)):
            x = x_ref[:, h * head_dim : (h + 1) * head_dim]
            y = jnp.dot(x, h_ref[...], preferred_element_type=jnp.float32)
            # msa_fp4.quantize, op for op.
            amax = jnp.max(jnp.abs(y), axis=-1, keepdims=True)
            inv = jnp.where(
                amax > 0, msa_fp4.INT4_MAX / jnp.where(amax > 0, amax, 1.0), 0.0
            )
            q = jnp.clip(jnp.round(y * inv), -msa_fp4.INT4_MAX, msa_fp4.INT4_MAX)
            col = (2 * h + i) * head_dim
            rows_ref[:, col : col + head_dim] = (q.astype(jnp.int32) & 0xF).astype(
                rows_ref.dtype
            )
            scales_ref[2 * h + i : 2 * h + i + 1, :] = (amax / msa_fp4.INT4_MAX).T


def quantize_kv_rows(
    k: jax.Array, v: jax.Array, *, interpret: bool = False
) -> tuple[jax.Array, jax.Array]:
    """bf16 ``k``/``v [T, G, D]`` -> (uint8 nibble rows ``[T, 2 * G * D]``,
    f32 scales ``[T, 2 * G]`` ordered ``[k_0, v_0, ..., k_G-1, v_G-1]``)."""
    num_tokens, heads, head_dim = k.shape
    block_t = min(num_tokens, BLOCK_TOKENS)
    rows, scales = pl.pallas_call(
        lambda *refs: _kernel(*refs, heads=heads, head_dim=head_dim),
        grid=(pl.cdiv(num_tokens, block_t),),
        in_specs=[
            pl.BlockSpec((head_dim, head_dim), lambda i: (0, 0)),
            pl.BlockSpec((block_t, heads * head_dim), lambda i: (i, 0)),
            pl.BlockSpec((block_t, heads * head_dim), lambda i: (i, 0)),
        ],
        out_specs=[
            pl.BlockSpec((block_t, 2 * heads * head_dim), lambda i: (i, 0)),
            pl.BlockSpec((2 * heads, block_t), lambda i: (0, i)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((num_tokens, 2 * heads * head_dim), jnp.uint8),
            jax.ShapeDtypeStruct((2 * heads, num_tokens), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="minimax_m3_msa_quantize_kv_rows",
    )(
        jnp.asarray(msa_fp4.hadamard(head_dim), jnp.bfloat16),
        k.reshape(num_tokens, -1),
        v.reshape(num_tokens, -1),
    )
    return rows, scales.T
