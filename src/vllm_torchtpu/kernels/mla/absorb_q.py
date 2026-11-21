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
from jax import lax
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


def absorb_q_split(
    q_nope: jax.Array, w_uk_t: jax.Array, split_rows: int
) -> tuple[jax.Array, jax.Array]:
    """Absorb `q_nope` `[T, H, nope]` @ `w_uk_t` `[H, nope, R]` into head/tail outputs.

    Reads `w_uk_t` once into VMEM and writes the leading `split_rows` and tail
    blocks to separate HBM outputs, avoiding unaligned post-kernel slice copies.
    """
    tokens, heads, nope = q_nope.shape
    rank = w_uk_t.shape[2]
    tail_tokens = tokens - split_rows
    block = MAX_BLOCK_TOKENS
    # Avoid launching the double-buffered HBM DMA kernel on small/unaligned shapes:
    # XLA einsum prefetches w_uk_t faster below MIN_TOKENS.
    if (
        tokens < MIN_TOKENS
        or split_rows <= 0
        or split_rows > block
        or split_rows % 8 != 0
        or tail_tokens < block
        or tail_tokens % block != 0
    ):
        return absorb_q(q_nope[:split_rows], w_uk_t), absorb_q(
            q_nope[split_rows:], w_uk_t
        )

    num_tail_blocks = tail_tokens // block

    def _split_kernel(
        q_hbm,
        w_ref,
        out_head_hbm,
        out_tail_hbm,
        q_vmem,
        o_vmem,
        sem_in,
        sem_out,
    ):
        def start_q(buf, start_row, nrows):
            pltpu.make_async_copy(
                q_hbm.at[pl.ds(start_row, nrows)],
                q_vmem.at[buf, pl.ds(0, nrows)],
                sem_in.at[buf],
            ).start()

        def wait_q(buf, start_row, nrows):
            pltpu.make_async_copy(
                q_hbm.at[pl.ds(start_row, nrows)],
                q_vmem.at[buf, pl.ds(0, nrows)],
                sem_in.at[buf],
            ).wait()

        def compute(buf, nrows):
            q_sl = pltpu.einshape("thd->htd", q_vmem[buf, :nrows])
            out = [
                jnp.dot(q_sl[h], w_ref[h], preferred_element_type=jnp.float32)
                for h in range(heads)
            ]
            o_vmem[buf, :nrows] = pltpu.einshape(
                "htd->thd", jnp.stack(out).astype(o_vmem.dtype)
            )

        start_q(0, 0, split_rows)
        start_q(1, split_rows, block)

        wait_q(0, 0, split_rows)
        compute(0, split_rows)
        head_out_cp = pltpu.make_async_copy(
            o_vmem.at[0, pl.ds(0, split_rows)],
            out_head_hbm.at[pl.ds(0, split_rows)],
            sem_out.at[0],
        )
        head_out_cp.start()

        if num_tail_blocks > 1:
            start_q(0, split_rows + block, block)
        wait_q(1, split_rows, block)
        compute(1, block)
        pltpu.make_async_copy(
            o_vmem.at[1],
            out_tail_hbm.at[pl.ds(0, block)],
            sem_out.at[1],
        ).start()
        if num_tail_blocks > 2:
            start_q(1, split_rows + 2 * block, block)

        head_out_cp.wait()

        # Avoid Python-unrolling all tail blocks: fori_loop keeps compile time
        # constant across buckets while matching double-buffered throughput.
        def body(b, _):
            buf = (b + 1) & 1
            wait_q(buf, split_rows + b * block, block)

            @pl.when(b >= 2)
            def _wait_prev_store():
                pltpu.make_async_copy(
                    o_vmem.at[buf],
                    out_tail_hbm.at[pl.ds((b - 2) * block, block)],
                    sem_out.at[buf],
                ).wait()

            compute(buf, block)
            pltpu.make_async_copy(
                o_vmem.at[buf],
                out_tail_hbm.at[pl.ds(b * block, block)],
                sem_out.at[buf],
            ).start()

            @pl.when(b + 2 < num_tail_blocks)
            def _prefetch_next():
                start_q(buf, split_rows + (b + 2) * block, block)

        if num_tail_blocks > 1:
            lax.fori_loop(1, num_tail_blocks, body, None)

        for rem in range(max(0, num_tail_blocks - 2), num_tail_blocks):
            buf = (rem + 1) & 1
            pltpu.make_async_copy(
                o_vmem.at[buf],
                out_tail_hbm.at[pl.ds(rem * block, block)],
                sem_out.at[buf],
            ).wait()

    return pl.pallas_call(
        _split_kernel,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(w_uk_t.shape, lambda: (0, 0, 0), pipeline_mode=pl.Buffered(1)),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        scratch_shapes=[
            pltpu.VMEM((2, block, heads, nope), q_nope.dtype),
            pltpu.VMEM((2, block, heads, rank), q_nope.dtype),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
        ],
        out_shape=(
            jax.ShapeDtypeStruct((split_rows, heads, rank), q_nope.dtype),
            jax.ShapeDtypeStruct((tail_tokens, heads, rank), q_nope.dtype),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes * 7 // 8,
        ),
    )(q_nope, w_uk_t)


def absorb_q_halves(
    q_nope: jax.Array, w_uk_t_lo: jax.Array, w_uk_t_hi: jax.Array
) -> jax.Array:
    """`out[t, h] = q_nope[t, h] @ concat(w_uk_t_lo, w_uk_t_hi)[h]`."""
    tokens = q_nope.shape[0]
    half = w_uk_t_lo.shape[0]
    if tokens < MIN_TOKENS:
        # Avoid concatenating w_uk_t halves before einsum: keeping them as two
        # independent operands lets XLA prefetch each half into VMEM separately.
        q_lo = jnp.einsum(
            "thd,hdr->thr",
            q_nope[:, :half],
            w_uk_t_lo,
            preferred_element_type=jnp.float32,
        ).astype(q_nope.dtype)
        q_hi = jnp.einsum(
            "thd,hdr->thr",
            q_nope[:, half:],
            w_uk_t_hi,
            preferred_element_type=jnp.float32,
        ).astype(q_nope.dtype)
        return jnp.concatenate([q_lo, q_hi], axis=1)
    return absorb_q(q_nope, jnp.concatenate([w_uk_t_lo, w_uk_t_hi], axis=0))


def absorb_q_split_halves(
    q_nope: jax.Array,
    w_uk_t_lo: jax.Array,
    w_uk_t_hi: jax.Array,
    split_rows: int,
) -> tuple[jax.Array, jax.Array]:
    """Split-head variant of `absorb_q_split` taking disjoint `[H/2, d, R]` halves."""
    if q_nope.shape[0] < MIN_TOKENS:
        return (
            absorb_q_halves(q_nope[:split_rows], w_uk_t_lo, w_uk_t_hi),
            absorb_q_halves(q_nope[split_rows:], w_uk_t_lo, w_uk_t_hi),
        )
    return absorb_q_split(
        q_nope, jnp.concatenate([w_uk_t_lo, w_uk_t_hi], axis=0), split_rows
    )
