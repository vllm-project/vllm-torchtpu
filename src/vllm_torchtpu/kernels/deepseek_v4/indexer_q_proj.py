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
"""The indexer's fp8 query projection with its RoPE and fp8 quantize fused in.

Computes what ``util.xla_prequantized_matmul`` followed by ``rope.rope_quant``
compute, but the bf16 projection never leaves VMEM. Each grid step projects a
tile of tokens through the resident weight, one column chunk at a time, into
one of two VMEM buffers, issued a chunk ahead. Each chunk's bf16 projection
is stored token-major, each token's heads on consecutive rows as in the
``[T, H, LANE]`` query: two heads' dot rows are packed into the bf16 pairs of
one u32 vreg and stored through a stride of one token. The rotation and the
scaling then run on rows that are (token, head) pairs, every token's cos and
sin broadcast from its row, and each fp8 tile of the query is cast and
stored whole once its heads are scaled. Each row's scale goes to lane t of
an ``[H, TILE_T]`` scratch, transposed into the ``[T, H]`` scales at the end
of the step.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.rope import (
    LANE,
    _cos_sin_lanes,
    _gptj_partners,
    _gptj_sign,
    _rope_rows_signed,
    _scale_rows,
)

# Fewest tokens the kernel takes; smaller (decode) buckets keep the XLA dot
# and the separate rope_quant.
MIN_ROWS = 512
# Tokens per grid step: every step pushes the whole weight through the MXU.
TILE_T = 512
# Projection columns per dot, and per weight DMA on the first step.
N_CHUNK = 2048
# Tokens rotated and scaled per block of rows.
ROPE_TOKENS = 64
VMEM_LIMIT_BYTES = 64 * 1024 * 1024
SUBLANE = 8
# Heads per fp8 (32, LANE) tile.
FP8_SUBLANES = 32


def _vmem_bytes(k: int, n: int) -> int:
    """VMEM the kernel's buffers take."""
    weight = k * n + n * 4  # fp8 weight and f32 scales, resident
    x = 2 * TILE_T * (k + LANE * 4)  # fp8 rows and their (lane-padded) scales
    proj = 2 * TILE_T * N_CHUNK * 2  # bf16
    scales = TILE_T * n // LANE * 4  # f32, transposed
    held = TILE_T * FP8_SUBLANES * LANE * 4  # f32 heads waiting for their tile
    cos_sin = 2 * TILE_T * LANE * 4
    lanes = 3 * TILE_T * LANE * 4
    out = 2 * TILE_T * (n + LANE * 4)  # fp8 query and its (lane-padded) scales
    dot = 2 * TILE_T * N_CHUNK * 4  # one chunk's f32 result, and its epilogue
    return weight + x + proj + scales + held + cos_sin + lanes + out + dot


def applies(
    x_q: jax.Array,
    w_q: jax.Array,
    cos_sin_cache: jax.Array,
    out_dtype: jnp.dtype,
    head_dim: int,
) -> bool:
    """Whether ``q_proj_rope_quant`` takes these operands, the projection's
    output being heads of ``head_dim`` side by side."""
    m, k = x_q.shape
    n = w_q.shape[1]
    return (
        x_q.dtype == jnp.float8_e4m3fn
        and w_q.dtype == jnp.float8_e4m3fn
        and out_dtype == jnp.bfloat16
        and head_dim == LANE
        and m >= MIN_ROWS
        and m % TILE_T == 0
        and n % N_CHUNK == 0
        and (
            N_CHUNK // LANE % FP8_SUBLANES == 0 or FP8_SUBLANES % (N_CHUNK // LANE) == 0
        )
        and TILE_T % ROPE_TOKENS == 0
        and n // LANE <= LANE
        and n // LANE % FP8_SUBLANES == 0
        and cos_sin_cache.shape[1] <= LANE
        and _vmem_bytes(k, n) <= 0.9 * VMEM_LIMIT_BYTES
        # The kernel's fp8 x fp8 dot needs an MXU that multiplies fp8.
        and pltpu.get_tpu_info().fp8_ops_per_second > 0
    )


def _kernel(
    # scalar prefetch
    positions_ref: jax.Ref,  # (M,) SMEM
    # inputs
    x_q_ref: jax.Ref,  # (TILE_T, K) fp8
    x_scale_ref: jax.Ref,  # (TILE_T, 1) f32
    w_hbm_ref: jax.Ref,  # (K, N) fp8, HBM
    w_scale_ref: jax.Ref,  # (N,) f32
    cos_sin_hbm_ref: jax.Ref,  # (max_position, rotary_dim) f32, HBM
    # outputs
    q_ref: jax.Ref,  # (TILE_T, H, LANE)
    scale_ref: jax.Ref,  # (TILE_T, H) f32
    # scratch
    w_ref: jax.Ref,  # (K, N) fp8
    # (2, TILE_T * N_CHUNK // LANE, LANE) bf16: two chunks' projections,
    # token-major: head h of the chunk's token t in row t * N_CHUNK // LANE + h.
    proj_ref: jax.Ref,
    cos_sin_ref: jax.Ref,  # (2, TILE_T, rotary_dim) f32
    # (3, TILE_T, LANE) f32: the tile's cos, signed sin and x scales, lane-wide
    lanes_ref: jax.Ref,
    scale_t_ref: jax.Ref,  # (H, TILE_T) f32: the tile's row scales, transposed
    held_ref: jax.Ref,  # (TILE_T, FP8_SUBLANES, LANE) f32: scaled heads of a tile
    w_sems: jax.Ref,  # (N // N_CHUNK,) DMA
    cos_sin_sems: jax.Ref,  # (2,) DMA
):
    step = pl.program_id(0)
    num_steps = pl.num_programs(0)
    num_chunks = w_ref.shape[1] // N_CHUNK
    heads_per_chunk = N_CHUNK // LANE
    quant_dtype = q_ref.dtype

    def weight_copy(c):
        cols = pl.ds(c * N_CHUNK, N_CHUNK)
        return pltpu.make_async_copy(
            w_hbm_ref.at[:, cols], w_ref.at[:, cols], w_sems.at[c]
        )

    def gather_cos_sin(tile, slot):
        """Starts one row DMA per token of ``tile``; all signal ``slot``."""

        @pl.loop(0, TILE_T, unroll=8)
        def _(t):
            position = positions_ref[tile * TILE_T + t]
            pltpu.make_async_copy(
                cos_sin_hbm_ref.at[position],
                cos_sin_ref.at[slot, t],
                cos_sin_sems.at[slot],
            ).start()

    @pl.when(step == 0)
    def _():
        gather_cos_sin(0, 0)
        for c in range(num_chunks):
            weight_copy(c).start()

    @pl.when(step + 1 < num_steps)
    def _():
        gather_cos_sin(step + 1, (step + 1) % 2)

    slot = step % 2
    # One wait for the slot's TILE_T row DMAs: same byte count.
    pltpu.make_async_copy(
        cos_sin_ref.at[slot], cos_sin_ref.at[slot], cos_sin_sems.at[slot]
    ).wait()
    cos, sin = _cos_sin_lanes(
        cos_sin_ref[slot], rotary_dim=cos_sin_ref.shape[-1], inverse=False
    )
    lanes_ref[0] = cos
    lanes_ref[1] = sin * _gptj_sign()
    lanes_ref[2] = jnp.broadcast_to(x_scale_ref[...], (TILE_T, LANE))

    def project(c):
        """Chunk ``c`` of the projection, rounded to bf16, into its buffer."""

        @pl.when(step == 0)
        def _():
            weight_copy(c).wait()

        # `xla_prequantized_matmul`: f32 dot, then `* x_scale`, `* w_scale`
        # and the cast to bf16.
        cols = pl.ds(c * N_CHUNK, N_CHUNK)
        x = jax.lax.dot_general(
            x_q_ref[...],
            w_ref[:, cols],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        x = x * jnp.concatenate([lanes_ref[2]] * heads_per_chunk, axis=1)
        x = x * w_scale_ref[cols][None, :]
        # The u32 view of the buffer holds rows 2r and 2r + 1 in the low and
        # high half of word row r.
        words = proj_ref.at[c % 2].bitcast(jnp.uint32)
        for p in range(heads_per_chunk // 2):
            pair = [x[:, (2 * p + i) * LANE : (2 * p + i + 1) * LANE] for i in range(2)]
            words[pl.ds(p, TILE_T, stride=heads_per_chunk // 2), :] = (
                pltpu.pack_elementwise(pair, packed_dtype=jnp.bfloat16)
            )

    def rope_quant(c):
        """Rotates and scales chunk ``c``'s heads, ROPE_TOKENS tokens at a time."""
        h0 = c * heads_per_chunk
        tile_h0 = h0 // FP8_SUBLANES * FP8_SUBLANES
        completes_tile = (h0 + heads_per_chunk) % FP8_SUBLANES == 0
        for t0 in range(0, TILE_T, ROPE_TOKENS):
            rows = pl.ds(t0 * heads_per_chunk, ROPE_TOKENS * heads_per_chunk)
            x_bf16 = proj_ref[c % 2, rows, :]
            x = x_bf16.astype(jnp.float32)
            partners = _gptj_partners(x_bf16).astype(jnp.float32)

            def per_token(k):
                """``lanes_ref[k]``'s row of each token, on each of its heads."""
                return jnp.concatenate(
                    [
                        jnp.broadcast_to(
                            lanes_ref[k, pl.ds(t0 + i, 1), :], (heads_per_chunk, LANE)
                        )
                        for i in range(ROPE_TOKENS)
                    ],
                    axis=0,
                )

            y = _rope_rows_signed(x, partners, per_token(0), per_token(1))
            scaled, scale = _scale_rows(y, quant_dtype)
            scaled = scaled.reshape(ROPE_TOKENS, heads_per_chunk, LANE)
            tokens = pl.ds(t0, ROPE_TOKENS)
            if not completes_tile:
                held_ref[tokens, pl.ds(h0 - tile_h0, heads_per_chunk), :] = scaled
            else:
                if h0 > tile_h0:
                    held = held_ref[tokens, pl.ds(0, h0 - tile_h0), :]
                    scaled = jnp.concatenate([held, scaled], axis=1)
                q_ref[tokens, pl.ds(tile_h0, h0 + heads_per_chunk - tile_h0), :] = (
                    scaled.astype(quant_dtype)
                )
            for r in range(0, ROPE_TOKENS * heads_per_chunk, SUBLANE):
                scale_t_ref[
                    pl.ds(h0 + r % heads_per_chunk, SUBLANE),
                    pl.ds(t0 + r // heads_per_chunk, 1),
                ] = scale[r : r + SUBLANE]

    # Chunk c + 1's dot is issued ahead of chunk c's rope and scaling, which
    # read the other buffer, so the MXU works under them.
    project(0)
    for c in range(num_chunks):
        if c + 1 < num_chunks:
            project(c + 1)
        rope_quant(c)
    scale_ref[...] = scale_t_ref[...].T


@functools.partial(jax.jit, static_argnames=("quant_dtype",))
def q_proj_rope_quant(
    x_q: jax.Array,  # [M, K] fp8
    x_scale: jax.Array,  # [M, 1] f32
    w_q: jax.Array,  # [K, N] fp8
    w_scale: jax.Array,  # [N]
    positions: jax.Array,  # [M] int
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] f32
    *,
    quant_dtype: jnp.dtype = jnp.float8_e4m3fn,
) -> tuple[jax.Array, jax.Array]:
    """``rope_quant(xla_prequantized_matmul(x_q, x_scale, w_q, w_scale,
    jnp.bfloat16).reshape(M, N // LANE, LANE), positions, cos_sin_cache,
    quant_dtype=quant_dtype)`` where ``applies`` holds.

    Returns the ``[M, N // LANE, LANE]`` quantized query and its
    ``[M, N // LANE]`` f32 row scales.
    """
    assert applies(x_q, w_q, cos_sin_cache, jnp.bfloat16, LANE), (x_q.shape, w_q.shape)
    m, k = x_q.shape
    n = w_q.shape[1]
    num_heads = n // LANE
    rotary_dim = cos_sin_cache.shape[1]
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(m // TILE_T,),
        in_specs=[
            pl.BlockSpec((TILE_T, k), lambda i, _: (i, 0)),
            pl.BlockSpec((TILE_T, 1), lambda i, _: (i, 0)),
            pl.BlockSpec(memory_space=pltpu.HBM),
            # Whole: a rank-1 block must be a power of two or 1024-aligned.
            pl.BlockSpec((n,), lambda i, _: (0,)),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=[
            pl.BlockSpec((TILE_T, num_heads, LANE), lambda i, _: (i, 0, 0)),
            pl.BlockSpec((TILE_T, num_heads), lambda i, _: (i, 0)),
        ],
        scratch_shapes=[
            pltpu.VMEM((k, n), w_q.dtype),
            pltpu.VMEM((2, TILE_T * N_CHUNK // LANE, LANE), jnp.bfloat16),
            pltpu.VMEM((2, TILE_T, rotary_dim), cos_sin_cache.dtype),
            pltpu.VMEM((3, TILE_T, LANE), jnp.float32),
            pltpu.VMEM((num_heads, TILE_T), jnp.float32),
            pltpu.VMEM((TILE_T, FP8_SUBLANES, LANE), jnp.float32),
            pltpu.SemaphoreType.DMA((n // N_CHUNK,)),
            pltpu.SemaphoreType.DMA((2,)),
        ],
    )
    return pl.pallas_call(
        _kernel,
        out_shape=[
            jax.ShapeDtypeStruct((m, num_heads, LANE), quant_dtype),
            jax.ShapeDtypeStruct((m, num_heads), jnp.float32),
        ],
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(
            # Later steps read the weight the first step copied in.
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
        name="indexer_q_proj_rope_quant",
    )(
        positions,
        x_q,
        x_scale.astype(jnp.float32),
        w_q,
        w_scale.astype(jnp.float32),
        cos_sin_cache,
    )
