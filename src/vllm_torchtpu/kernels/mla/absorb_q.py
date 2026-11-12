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

import functools

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


def _kernel_flat(q_ref, w_ref, o_ref, *, head_stride: int):
    heads, nope, _ = w_ref.shape
    nope_aligned = pl.cdiv(nope, 128) * 128
    out = [
        jnp.dot(
            q_ref[:, h * head_stride : h * head_stride + nope_aligned][:, :nope],
            w_ref[h],
            preferred_element_type=jnp.float32,
        ).astype(o_ref.dtype)
        for h in range(heads)
    ]
    o_ref[...] = pltpu.einshape("htd->thd", jnp.stack(out))


def _absorb_q_impl(
    q_nope: jax.Array, w_uk_t: jax.Array, *, interpret: bool = False
) -> jax.Array:
    heads, nope, rank = w_uk_t.shape
    if q_nope.ndim == 2:
        tokens, total_width = q_nope.shape
        if total_width % heads != 0:
            raise ValueError(
                f"2D q_nope width ({total_width}) must be divisible by heads ({heads})"
            )
        head_stride = total_width // heads
        if head_stride < nope:
            raise ValueError(
                f"2D q_nope head_stride ({head_stride}) must be >= nope ({nope})"
            )
        nope_aligned = pl.cdiv(nope, 128) * 128
        sublane_tile = 8 * max(1, 4 // jnp.dtype(q_nope.dtype).itemsize)
        if (
            tokens >= MIN_TOKENS
            and tokens % sublane_tile == 0
            and head_stride % 128 == 0
            and head_stride >= nope_aligned
            and heads % 8 == 0
            and rank % 128 == 0
        ):
            block = min(tokens, MAX_BLOCK_TOKENS)
            if block % sublane_tile == 0 or block == tokens:
                vmem_limit = (
                    None
                    if interpret
                    else pltpu.get_tpu_info().vmem_capacity_bytes * 7 // 8
                )
                return pl.pallas_call(
                    functools.partial(_kernel_flat, head_stride=head_stride),
                    grid=(pl.cdiv(tokens, block),),
                    in_specs=[
                        pl.BlockSpec((block, total_width), lambda i: (i, 0)),
                        # Read once: the index never changes.
                        pl.BlockSpec(
                            w_uk_t.shape,
                            lambda i: (0, 0, 0),
                            pipeline_mode=pl.Buffered(1),
                        ),
                    ],
                    out_specs=pl.BlockSpec((block, heads, rank), lambda i: (i, 0, 0)),
                    out_shape=jax.ShapeDtypeStruct((tokens, heads, rank), q_nope.dtype),
                    compiler_params=pltpu.CompilerParams(
                        dimension_semantics=("parallel",),
                        vmem_limit_bytes=vmem_limit,
                    ),
                    interpret=interpret,
                )(q_nope, w_uk_t)
        q_nope = q_nope.reshape(tokens, heads, head_stride)[:, :, :nope]
    elif q_nope.ndim == 3:
        tokens, q_heads, q_nope_dim = q_nope.shape
        if (q_heads, q_nope_dim) != (heads, nope):
            raise ValueError(
                f"q_nope shape {q_nope.shape} incompatible with w_uk_t {w_uk_t.shape}"
            )
    else:
        raise ValueError(f"q_nope must be 2D or 3D, got {q_nope.shape}")

    if tokens < MIN_TOKENS:
        out = jnp.einsum(
            "thd,hdr->thr", q_nope, w_uk_t, preferred_element_type=jnp.float32
        )
        return out.astype(q_nope.dtype)
    block = min(tokens, MAX_BLOCK_TOKENS)
    vmem_limit = (
        None if interpret else pltpu.get_tpu_info().vmem_capacity_bytes * 7 // 8
    )
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
            vmem_limit_bytes=vmem_limit,
        ),
        interpret=interpret,
    )(q_nope, w_uk_t)


def absorb_q(q_nope: jax.Array, w_uk_t: jax.Array) -> jax.Array:
    """`out[t, h] = q_nope[t, h] @ w_uk_t[h]`.

    Args:
      q_nope: `[T, H, nope]` or flat `[T, H * head_stride]` (`head_stride >= nope`).
      w_uk_t: `[H, nope, R]`.

    Returns:
      `[T, H, R]` in `q_nope.dtype`.
    """
    return _absorb_q_impl(q_nope, w_uk_t)
