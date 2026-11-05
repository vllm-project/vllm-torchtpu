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

`absorb_q_head` takes the q projection's output `[T, H * (nope + rope)]` as
the projection writes it and returns the absorbed q and, in the attention
kernel's head-major tiles, the last `HEAD_TAIL_LANES` lanes of each head (all the
attention kernel reads of the raw q head), so XLA never copies the whole q
into the attention kernel's layout.
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
# The 2-D input leaves VMEM for 256 tokens per grid step beside W_UK.
MAX_HEAD_BLOCK_TOKENS = 256
# The raw q head lanes the sparse MLA attention kernel reads with its
# in-kernel rope: q_nope[128:192] | q_pe.
HEAD_TAIL_LANES = 128


def _kernel(q_ref, w_ref, o_ref):
    q = pltpu.einshape("thd->htd", q_ref[...])
    out = [
        jnp.dot(q[h], w_ref[h], preferred_element_type=jnp.float32)
        for h in range(q.shape[0])
    ]
    o_ref[...] = pltpu.einshape("htd->thd", jnp.stack(out).astype(o_ref.dtype))


def _head_kernel(q_ref, w_ref, o_ref, pe_ref):
    heads, nope, _ = w_ref.shape
    width = q_ref.shape[1] // heads
    lanes = pe_ref.shape[2]
    out = [
        jnp.dot(
            q_ref[:, h * width : h * width + nope],
            w_ref[h],
            preferred_element_type=jnp.float32,
        ).astype(o_ref.dtype)
        for h in range(heads)
    ]
    o_ref[...] = pltpu.einshape("htd->thd", jnp.stack(out))
    pe = [q_ref[:, (h + 1) * width - lanes : (h + 1) * width] for h in range(heads)]
    pe_ref[...] = pltpu.einshape("htd->thd", jnp.stack(pe))


def absorb_q_head(q2d: jax.Array, w_uk_t: jax.Array) -> tuple[jax.Array, jax.Array]:
    """`out[t, h] = q[t, h, :nope] @ w_uk_t[h]` (the same bits as `absorb_q`
    of that slice) and `pe[t, h] = q[t, h, -HEAD_TAIL_LANES:]`, with `q` the
    `[T, H, nope + rope]` view of `q2d`.

    Args:
      q2d: `[T, H * (nope + rope)]`, q_nope in the first `nope` lanes of each
        head; `T >= MIN_TOKENS`.
      w_uk_t: `[H, nope, R]`.

    Returns:
      `[T, H, R]` and `[T, H, HEAD_TAIL_LANES]` in `q2d.dtype`.
    """
    tokens = q2d.shape[0]
    heads, _, rank = w_uk_t.shape
    pe_lanes = HEAD_TAIL_LANES
    assert tokens >= MIN_TOKENS, tokens
    block = min(tokens, MAX_HEAD_BLOCK_TOKENS)
    return pl.pallas_call(
        _head_kernel,
        grid=(pl.cdiv(tokens, block),),
        in_specs=[
            pl.BlockSpec((block, q2d.shape[1]), lambda i: (i, 0)),
            pl.BlockSpec(
                w_uk_t.shape, lambda i: (0, 0, 0), pipeline_mode=pl.Buffered(1)
            ),
        ],
        out_specs=[
            pl.BlockSpec((block, heads, rank), lambda i: (i, 0, 0)),
            pl.BlockSpec((block, heads, pe_lanes), lambda i: (i, 0, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((tokens, heads, rank), q2d.dtype),
            jax.ShapeDtypeStruct((tokens, heads, pe_lanes), q2d.dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes * 7 // 8,
        ),
    )(q2d, w_uk_t)


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
