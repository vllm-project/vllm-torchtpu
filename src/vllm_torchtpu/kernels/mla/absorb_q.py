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

`absorb_q_full` (TPU_MLA_ABSORB_FULL_Q) reads the q_b projection output itself,
`[T, H * head_dim]`, in the layout q_b wrote it (tokens on sublanes). Viewing
that as `[T, H, head_dim]` for `absorb_q` makes XLA relayout the whole of q
(heads on sublanes) ahead of the kernel; here each head's nope lanes are a
128-aligned lane slice of the block, so no relayout runs anywhere. It also
writes each head's last 128 lanes (raw q_pe and the nope lanes beside it) for
the TPU_MLA_DECODE_PROLOGUE attention kernel, the other reader of q.
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
# `absorb_q_full`'s second output: each head's last 128 lanes.
TAIL_LANES = 128


def _kernel(q_ref, w_ref, o_ref):
    q = pltpu.einshape("thd->htd", q_ref[...])
    out = [
        jnp.dot(q[h], w_ref[h], preferred_element_type=jnp.float32)
        for h in range(q.shape[0])
    ]
    o_ref[...] = pltpu.einshape("htd->thd", jnp.stack(out).astype(o_ref.dtype))


def _full_q_kernel(q_ref, w_ref, o_ref, tail_ref, *, head_dim):
    heads, nope = w_ref.shape[0], w_ref.shape[1]
    # Head h's nope lanes, [h * head_dim, h * head_dim + nope): a static slice
    # starting on a 128-lane boundary. The same (block, nope) @ (nope, R)
    # products as `_kernel`, so the same bits.
    out = [
        jnp.dot(
            q_ref[:, h * head_dim : h * head_dim + nope],
            w_ref[h],
            preferred_element_type=jnp.float32,
        )
        for h in range(heads)
    ]
    o_ref[...] = pltpu.einshape("htd->thd", jnp.stack(out).astype(o_ref.dtype))
    # Each head's last 128 lanes, as they are (one aligned vreg column): the
    # raw q_pe and the nope lanes beside it, what the prologue kernel ropes.
    tail = [
        q_ref[:, (h + 1) * head_dim - TAIL_LANES : (h + 1) * head_dim]
        for h in range(heads)
    ]
    tail_ref[...] = pltpu.einshape("htd->thd", jnp.stack(tail))


def absorb_q_full(
    q: jax.Array, w_uk_t: jax.Array, *, interpret: bool = False
) -> tuple[jax.Array, jax.Array]:
    """`absorb_q(q.reshape(T, H, head_dim)[..., :nope], w_uk_t)` and the raw
    rope lanes, both read in one pass over q as the q_b projection wrote it.

    Args:
      q: `[T, H * head_dim]`, each head's nope channels first.
      w_uk_t: `[H, nope, R]`, `nope <= head_dim`.
      interpret: run the Pallas kernel in interpret mode (CPU tests).

    Returns:
      `[T, H, R]` in `q.dtype`, bit-identical to `absorb_q` on the slice, and
      `q.reshape(T, H, head_dim)[..., -128:]` (`[T, H, 128]`, un-roped: q_pe
      in its last `head_dim - nope` lanes) -- the lanes of the raw q head the
      TPU_MLA_DECODE_PROLOGUE kernel reads, without the `[T, H, head_dim]`
      relayout of q.
    """
    tokens, width = q.shape
    heads, nope, rank = w_uk_t.shape
    head_dim = width // heads
    if head_dim * heads != width or nope > head_dim or head_dim % 128:
        raise ValueError(
            f"q {q.shape} is not [T, {heads} * head_dim] with a 128-aligned "
            f"head_dim >= nope ({nope})"
        )
    if tokens < MIN_TOKENS:
        q3 = q.reshape(tokens, heads, head_dim)
        return absorb_q(q3[..., :nope], w_uk_t), q3[..., -TAIL_LANES:]
    block = min(tokens, MAX_BLOCK_TOKENS)
    if interpret:
        vmem_limit = None
    else:
        vmem_limit = pltpu.get_tpu_info().vmem_capacity_bytes * 7 // 8
    return pl.pallas_call(
        functools.partial(_full_q_kernel, head_dim=head_dim),
        grid=(pl.cdiv(tokens, block),),
        in_specs=[
            # All H * head_dim lanes: the rope lanes ride along (the block is
            # the same VMEM as absorb_q's, whose 192 lanes pad to 256).
            pl.BlockSpec((block, width), lambda i: (i, 0)),
            pl.BlockSpec(
                w_uk_t.shape, lambda i: (0, 0, 0), pipeline_mode=pl.Buffered(1)
            ),
        ],
        out_specs=[
            pl.BlockSpec((block, heads, rank), lambda i: (i, 0, 0)),
            pl.BlockSpec((block, heads, TAIL_LANES), lambda i: (i, 0, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((tokens, heads, rank), q.dtype),
            jax.ShapeDtypeStruct((tokens, heads, TAIL_LANES), q.dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=vmem_limit,
        ),
        interpret=interpret,
    )(q, w_uk_t)


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
