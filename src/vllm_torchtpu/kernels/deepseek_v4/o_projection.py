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

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.rope import (
    LANE,
    CosSinRef,
    _cos_sin_lanes,
    _gptj_sign,
    _rotate_gptj,
    _swap_pairs,
)
from vllm_torchtpu.layers.core.quantization import FP8_E4M3_MAX

DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024

SUBLANE = 8


def _largest_divisor(x: int, cap: int) -> int:
    """Largest divisor of ``x`` that is <= ``cap``."""
    for candidate in range(min(x, cap), 0, -1):
        if x % candidate == 0:
            return candidate
    return 1


def _cos_sin_body(
    cos_sin_ref: jax.Ref,
    out_ref: jax.Ref,
    *,
    rotary_dim: int,
    inverse: bool,
    out_dtype: jnp.dtype,
) -> None:
    """Expand a ``(tile_n, rotary_dim)`` block to ``(tile_n, 2 * LANE)``."""
    cos, sin = _cos_sin_lanes(cos_sin_ref[...], rotary_dim=rotary_dim, inverse=inverse)
    out_ref[...] = jnp.concatenate([cos, sin], axis=-1).astype(out_dtype)


def _cos_sin_kernel(
    # scalar prefetch
    positions_ref: jax.Ref,
    # HBM input
    cos_sin_cache_hbm_ref: jax.Ref,
    # HBM output
    out_hbm_ref: jax.Ref,
    *,
    num_tiles: int,
    tile_n: int,
    rotary_dim: int,
    inverse: bool,
    cos_sin_dtype: jnp.dtype,
    out_dtype: jnp.dtype,
) -> None:
    cos_sin_spec = pl.BlockSpec(
        block_shape=(tile_n, rotary_dim),
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i, 0),
    )
    out_spec = pl.BlockSpec(
        block_shape=(tile_n, 2 * LANE),
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i, 0),
    )
    # The per-token row gather; one buffer of lookahead hides its DMAs.
    cos_sin_alloc = CosSinRef.create(
        spec=cos_sin_spec, dtype=cos_sin_dtype, buffer_count=2, tile_n=tile_n
    )

    pipeline = pltpu.emit_pipeline(
        functools.partial(
            _cos_sin_body,
            rotary_dim=rotary_dim,
            inverse=inverse,
            out_dtype=out_dtype,
        ),
        grid=(num_tiles,),
        in_specs=[cos_sin_spec],
        out_specs=[out_spec],
    )
    allocations = (
        cos_sin_alloc,
        pltpu.BufferedRef.output(out_spec, out_dtype, buffer_count=2),
    )

    @pl.with_scoped(allocations=allocations)
    def _run(allocations):
        pipeline(
            (cos_sin_cache_hbm_ref, positions_ref),
            out_hbm_ref,
            allocations=allocations,
        )

    _run()


@functools.partial(
    jax.jit,
    static_argnames=("inverse", "out_dtype", "name"),
)
def gather_cos_sin(
    positions: jax.Array,  # [T] int
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    *,
    inverse: bool = False,
    out_dtype: jnp.dtype = jnp.float32,
    name: str = "gather_cos_sin",
) -> jax.Array:
    """Builds ``wo_a_projection``'s ``[T, 2 * LANE]`` cos/sin table.

    Gathers ``cos_sin_cache[positions]`` based on the token positions, expands
    each row to ``[2 * LANE]``
    """
    assert positions.ndim == 1
    assert cos_sin_cache.ndim == 2

    num_tokens = positions.shape[0]
    rotary_dim = cos_sin_cache.shape[1]
    assert rotary_dim % 2 == 0 and rotary_dim <= LANE

    tile_n = _largest_divisor(num_tokens, cap=128)

    return pl.pallas_call(
        functools.partial(
            _cos_sin_kernel,
            num_tiles=num_tokens // tile_n,
            tile_n=tile_n,
            rotary_dim=rotary_dim,
            inverse=inverse,
            cos_sin_dtype=cos_sin_cache.dtype,
            out_dtype=out_dtype,
        ),
        out_shape=jax.ShapeDtypeStruct((num_tokens, 2 * LANE), out_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),),
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
        name=name,
    )(positions, cos_sin_cache)


def _rope_heads(x: jax.Array, cos_sin: jax.Array, *, head_dim: int) -> jax.Array:
    """apply RoPE to the last rope dim of head dim of ``x``."""
    cos = cos_sin[:, :LANE]
    sin = cos_sin[:, LANE:]
    lo = head_dim - LANE
    tail = x[:, :, lo:].astype(jnp.float32)
    roped = tail * cos[:, None, :] + _rotate_gptj(tail) * sin[:, None, :]
    return jnp.concatenate([x[:, :, :lo], roped.astype(x.dtype)], axis=-1)


def _kernel(
    # inputs
    x_ref: jax.Ref,  # VMEM (tile_t, H, head_dim) bf16
    w_ref: jax.Ref,  # VMEM (D, tile_r) fp8
    scale_ref: jax.Ref,  # VMEM (1, tile_r) f32
    cos_sin_ref: jax.Ref,  # VMEM (tile_t, 2 * LANE) f32
    # output
    out_ref: jax.Ref,  # VMEM (tile_t, tile_r) out_dtype
    *,
    tile_t: int,
    num_sub_t: int,
    quantize_activations: bool,
    head_dim: int,
) -> None:
    rhs = w_ref[...]
    scale = scale_ref[...]
    cos_sin = cos_sin_ref[...]
    sub_t = tile_t // num_sub_t
    out_dtype = x_ref.dtype
    for s in range(num_sub_t):
        # Subchunk the tile to allow more Instruction level parallelism
        # opportunities to be exploited by the LLO compiler.
        rows = slice(s * sub_t, (s + 1) * sub_t)
        x = _rope_heads(x_ref[rows], cos_sin[rows], head_dim=head_dim)
        x = x.reshape(sub_t, -1)
        inv = None
        if quantize_activations:
            amax = jnp.max(jnp.abs(x), axis=1, keepdims=True)
            inv = (FP8_E4M3_MAX / jnp.maximum(amax, jnp.bfloat16(1e-30))).astype(
                jnp.bfloat16
            )
            lhs = (x * inv).astype(jnp.float8_e4m3fn)
        else:
            lhs = x

        partial = jax.lax.dot_general(
            lhs,
            rhs,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        if quantize_activations:
            assert inv is not None
            partial = partial * (1.0 / inv.astype(jnp.float32))
        out_ref[rows] = (partial * scale).astype(out_dtype)


# The pipelined prefill kernel (_pipelined_kernel). Tokens per x row block, i.e.
# per wo_a dot: a group's weights are pushed into the MXU once per this many
# tokens instead of once per 128.
WO_A_ROWS = 512
# Tokens per RoPE + quantize sub-chunk of a row block.
WO_A_PREP_ROWS = 64
# Calls with fewer tokens (the decode buckets) keep _kernel.
WO_A_MIN_TOKENS = 512
# VMEM Mosaic needs beyond the scratch buffers (the f32 dot result, spills).
WO_A_VMEM_HEADROOM_BYTES = 8388608  # 8 MiB
# Tokens per cos/sin cache gather check: one DMA when their positions are
# consecutive, one per token otherwise.
WO_A_GATHER_RUN = 32
# Independent min chains in the scalar consecutiveness check of a gather run.
WO_A_GATHER_CHAINS = 4
# DMAs (K row slices) per w group, started over the previous group's row
# blocks so its transfer is spread rather than bunched at a group's start.
WO_A_W_PARTS = 2


def _pipelined_rows(num_tokens: int) -> int | None:
    """_pipelined_kernel's row block at ``num_tokens``, None if it does not apply."""
    rows = min(WO_A_ROWS, num_tokens)
    if num_tokens < WO_A_MIN_TOKENS or num_tokens % rows or rows % WO_A_PREP_ROWS:
        return None
    assert (
        WO_A_GATHER_RUN > WO_A_GATHER_CHAINS and WO_A_PREP_ROWS % WO_A_GATHER_RUN == 0
    )
    return rows


def _pipelined_kernel(
    # scalar prefetch
    positions_ref: jax.Ref,  # SMEM (T,) int32
    # inputs
    x_hbm_ref: jax.Ref,  # HBM (T, G * H, head_dim) bf16
    w_hbm_ref: jax.Ref,  # HBM (D, G * R) fp8
    scale_hbm_ref: jax.Ref,  # HBM (1, G * R) f32
    cos_sin_cache_hbm_ref: jax.Ref,  # HBM (max_position, rotary_dim) f32
    # output
    out_hbm_ref: jax.Ref,  # HBM (T, G * R) bf16
    # scratch
    scale_ref: jax.Ref,  # (1, G * R) f32
    cos_sin_rows_ref: jax.Ref,  # (T, rotary_dim) f32, the gathered cache rows
    cos_sin_ref: jax.Ref,  # (T, 2 * LANE) f32
    x_buf: jax.Ref,  # (2, rows, H, head_dim) bf16
    x_q_ref: jax.Ref,  # (2, rows, D) fp8
    x_rinv_ref: jax.Ref,  # (2, rows, 1) f32
    w_buf: jax.Ref,  # (2, D, R) fp8
    out_buf: jax.Ref,  # (2, rows, R) bf16
    x_sems: jax.Ref,  # (2,) DMA
    w_sems: jax.Ref,  # (2,) DMA
    out_sems: jax.Ref,  # (2,) DMA
    x0_sems: jax.Ref,  # (rows // WO_A_PREP_ROWS,) DMA, block 0's sub-chunks
    const_sems: jax.Ref,  # (1,) DMA, scale
    cos_sin_sems: jax.Ref,  # (T // WO_A_PREP_ROWS,) DMA, cache rows per sub-chunk
    *,
    num_groups: int,
    groups_per_step: int,
    head_dim: int,
    inverse: bool,
) -> None:
    """``_kernel`` with quantize_activations, by row blocks of ``rows`` tokens.

    Row block n (group g, rows r) is RoPE'd and quantized into x_q's slot n % 2,
    then projected by one [rows, D] x [D, R] dot. Segment n of the schedule runs
    the dot of block n (MXU) and the prep of block n + 1 (VPU), also across
    group and grid-step edges, so the VPU work hides under the MXU work. A grid
    step holds an even number of blocks, so every slot index is static and the
    scheduler can tell the two x_q slots apart. DMA waits and branches sit only
    at segment tops; each segment's output DMA starts at its end, since a DMA
    start orders the memory ops around it. Grid step 0 also does
    gather_cos_sin's work: it gathers the cache rows of ``positions`` and
    widens them into cos_sin_ref, a sub-chunk ahead of each block-0 prep.
    """
    num_tokens = cos_sin_ref.shape[0]
    _, rows, heads, _ = x_buf.shape
    lora_rank = w_buf.shape[2]
    blocks = num_tokens // rows  # row blocks per group
    per_step = groups_per_step * blocks  # even
    num_steps = num_groups // groups_per_step
    total = num_groups * blocks
    prep_rows = WO_A_PREP_ROWS
    run = WO_A_GATHER_RUN
    w_part_rows = w_buf.shape[1] // WO_A_W_PARTS
    step = pl.program_id(0)

    def locate(l):
        """(group, row block, slot) of step-local block l, which may lie in a
        neighbouring step; l is static."""
        q, l2 = divmod(l, per_step)
        j, r = divmod(l2, blocks)
        return (step + q) * groups_per_step + j, r, l % 2

    def w_slot(g):
        if groups_per_step == 1:
            return g % 2
        return (g - step * groups_per_step) % 2  # static

    def x_copy(g, r, slot):
        return pltpu.make_async_copy(
            x_hbm_ref.at[pl.ds(r * rows, rows), pl.ds(g * heads, heads)],
            x_buf.at[slot],
            x_sems.at[slot],
        )

    def w_copy(g, part):
        col0 = pl.multiple_of(g * lora_rank, lora_rank)
        k = pl.ds(part * w_part_rows, w_part_rows)
        return pltpu.make_async_copy(
            w_hbm_ref.at[k, pl.ds(col0, lora_rank)],
            w_buf.at[w_slot(g), k],
            w_sems.at[w_slot(g)],
        )

    def out_copy(g, r, slot):
        col0 = pl.multiple_of(g * lora_rank, lora_rank)
        return pltpu.make_async_copy(
            out_buf.at[slot],
            out_hbm_ref.at[pl.ds(r * rows, rows), pl.ds(col0, lora_rank)],
            out_sems.at[slot],
        )

    def gather(chunk):
        """Start the cache row copies of tokens [chunk * run, + run): one copy
        when their positions are consecutive, one per token otherwise."""
        row0 = chunk * run
        # Strictly increasing and spanning run - 1 means consecutive. The min
        # step is taken in a few independent chains: few live registers.
        pos = [positions_ref[row0 + i] for i in range(run)]
        p0 = pos[0]
        chains = [pos[i + 1] - pos[i] for i in range(WO_A_GATHER_CHAINS)]
        for i in range(WO_A_GATHER_CHAINS, run - 1):
            k = i % WO_A_GATHER_CHAINS
            chains[k] = jnp.minimum(chains[k], pos[i + 1] - pos[i])
        consecutive = jnp.logical_and(
            functools.reduce(jnp.minimum, chains) >= 1, pos[-1] - p0 == run - 1
        )
        sem = cos_sin_sems.at[row0 // prep_rows]

        @pl.when(consecutive)
        def _():
            pltpu.make_async_copy(
                cos_sin_cache_hbm_ref.at[pl.ds(p0, run)],
                cos_sin_rows_ref.at[pl.ds(row0, run)],
                sem,
            ).start()

        @pl.when(jnp.logical_not(consecutive))
        def _():
            @pl.loop(0, run)
            def _(i):
                pltpu.make_async_copy(
                    cos_sin_cache_hbm_ref.at[positions_ref[row0 + i]],
                    cos_sin_rows_ref.at[row0 + i],
                    sem,
                ).start()

    def expand(s):
        """gather_cos_sin's lane expansion of sub-chunk s's rows, once landed."""
        sub = pl.ds(s * prep_rows, prep_rows)
        rows_ref = cos_sin_rows_ref.at[sub]
        pltpu.make_async_copy(rows_ref, rows_ref, cos_sin_sems.at[s]).wait()
        cos, sin = _cos_sin_lanes(
            rows_ref[...], rotary_dim=cos_sin_rows_ref.shape[1], inverse=inverse
        )
        cos_sin_ref[sub] = jnp.concatenate([cos, sin], axis=-1).astype(
            cos_sin_ref.dtype
        )

    def prep(r, slot, sub_ready=None):
        """RoPE + per-row quantize of the x block in ``slot`` (row block r),
        each sub-chunk after ``sub_ready(s)``, if given."""
        lo = head_dim - LANE
        for s in range(rows // prep_rows):
            sub = pl.ds(s * prep_rows, prep_rows)
            if sub_ready is not None:
                sub_ready(s)
            cos_sin = cos_sin_ref[pl.ds(r * rows + s * prep_rows, prep_rows)]
            # _rotate_gptj's sign is +-1, so folding it into sin is exact.
            cos, sin = cos_sin[:, :LANE], cos_sin[:, LANE:] * _gptj_sign()
            # Heads major first: the [rows, H * head_dim] relayout the dot needs
            # is then one transpose, and cos/sin apply per row unbroadcast.
            x_heads = jnp.swapaxes(x_buf[slot, sub], 0, 1)
            parts = []
            for xh in x_heads:
                tail = xh[:, lo:].astype(jnp.float32)
                roped = tail * cos + _swap_pairs(tail) * sin
                parts += [xh[:, :lo], roped.astype(xh.dtype)]
            x = jnp.concatenate(parts, axis=-1)
            amax = jnp.max(jnp.abs(x), axis=1, keepdims=True)
            inv = (FP8_E4M3_MAX / jnp.maximum(amax, jnp.bfloat16(1e-30))).astype(
                jnp.bfloat16
            )
            x_q_ref[slot, sub] = (x * inv).astype(jnp.float8_e4m3fn)
            x_rinv_ref[slot, sub] = 1.0 / inv.astype(jnp.float32)

    def dot(g, slot):
        """wo_a dot + dequantize of the block in x_q's ``slot`` into out_buf."""
        partial = jax.lax.dot_general(
            x_q_ref[slot],
            w_buf[w_slot(g)],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        partial = partial * x_rinv_ref[slot]
        col0 = pl.multiple_of(g * lora_rank, lora_rank)
        out_buf[slot] = (partial * scale_ref[:, pl.ds(col0, lora_rank)]).astype(
            out_buf.dtype
        )

    @pl.when(step == 0)
    def _():
        # scale is fetched here, beside block 0, rather than staged in VMEM by
        # an XLA copy ahead of the call.
        scale_copy = pltpu.make_async_copy(scale_hbm_ref, scale_ref, const_sems.at[0])
        scale_copy.start()
        # Block 0 by sub-chunks, so its prep starts after the first one lands,
        # each one's cache rows requested just ahead of it.
        sub_copies = [
            pltpu.make_async_copy(
                x_hbm_ref.at[pl.ds(s * prep_rows, prep_rows), pl.ds(0, heads)],
                x_buf.at[0, pl.ds(s * prep_rows, prep_rows)],
                x0_sems.at[s],
            )
            for s in range(rows // prep_rows)
        ]
        for s, copy in enumerate(sub_copies):
            for chunk in range(s * prep_rows // run, (s + 1) * prep_rows // run):
                gather(chunk)
            copy.start()
        for part in range(WO_A_W_PARTS):
            w_copy(0, part).start()
        if num_tokens > rows:
            pl.loop(rows // run, num_tokens // run)(gather)
        x_copy(*locate(1)).start()

        def sub_ready(s):
            sub_copies[s].wait()
            expand(s)

        prep(0, 0, sub_ready)
        for s in range(rows // prep_rows, num_tokens // prep_rows):
            expand(s)
        scale_copy.wait()

    for l in range(per_step):
        g, r, slot = locate(l)
        _, r1, slot1 = locate(l + 1)
        last = l + 1 == per_step
        if r == 0:
            for part in range(WO_A_W_PARTS):
                w_copy(g, part).wait()

        def fetch_next():
            x_copy(*locate(l + 1)).wait()

            @pl.when(step * per_step + l + 2 < total)
            def _():
                x_copy(*locate(l + 2)).start()

        if last:
            pl.when(step + 1 < num_steps)(fetch_next)
        else:
            fetch_next()

        # After x: the next segment's top waits on that x block, not on w.
        parts = [p for p in range(WO_A_W_PARTS) if p * blocks // WO_A_W_PARTS == r]
        if parts:

            @pl.when(g + 1 < num_groups)
            def _():
                for part in parts:
                    w_copy(g + 1, part).start()

        if l >= 2:
            out_copy(*locate(l - 2)).wait()
        else:
            pl.when(step > 0)(lambda: out_copy(*locate(l - 2)).wait())

        if not last:
            dot(g, slot)
            prep(r1, slot1)
            out_copy(g, r, slot).start()
            continue

        # Peeled, so the dot and the next step's first prep share a block.
        @pl.when(step + 1 < num_steps)
        def _():
            dot(g, slot)
            prep(r1, slot1)
            out_copy(g, r, slot).start()

        @pl.when(step + 1 == num_steps)
        def _():
            dot(g, slot)
            out_copy(g, r, slot).start()
            out_copy(*locate(l - 1)).wait()
            out_copy(g, r, slot).wait()


def _wo_a_pipelined(
    x: jax.Array,  # [T, G * H, head_dim] bf16
    positions: jax.Array,  # [T] int
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    wo_a: jax.Array,  # [D, G * R] fp8
    wo_a_scale: jax.Array,  # [G * R] float32
    *,
    inverse: bool,
    rows: int,
    name: str,
) -> jax.Array:
    num_tokens, num_heads, head_dim = x.shape
    reduction, out_features = wo_a.shape
    assert reduction == SUBLANE * head_dim and num_heads % SUBLANE == 0
    assert reduction % WO_A_W_PARTS == 0
    num_groups = num_heads // SUBLANE
    lora_rank = out_features // num_groups
    rotary_dim = cos_sin_cache.shape[1]
    assert positions.shape == (num_tokens,)
    assert rotary_dim % 2 == 0 and rotary_dim <= LANE
    groups_per_step = 1 if (num_tokens // rows) % 2 == 0 else 2
    assert num_groups % groups_per_step == 0
    scratch = [
        pltpu.VMEM((1, out_features), jnp.float32),
        pltpu.VMEM((num_tokens, rotary_dim), cos_sin_cache.dtype),
        pltpu.VMEM((num_tokens, 2 * LANE), jnp.float32),
        pltpu.VMEM((2, rows, SUBLANE, head_dim), x.dtype),
        pltpu.VMEM((2, rows, reduction), wo_a.dtype),
        pltpu.VMEM((2, rows, 1), jnp.float32),
        pltpu.VMEM((2, reduction, lora_rank), wo_a.dtype),
        pltpu.VMEM((2, rows, lora_rank), x.dtype),
    ]
    # x_rinv's trailing 1 is padded to a full lane tile.
    scratch_bytes = sum(
        math.prod(b.shape[:-1]) * max(b.shape[-1], LANE) * jnp.dtype(b.dtype).itemsize
        for b in scratch
    )
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        functools.partial(
            _pipelined_kernel,
            num_groups=num_groups,
            groups_per_step=groups_per_step,
            head_dim=head_dim,
            inverse=inverse,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(num_groups // groups_per_step,),
            in_specs=[hbm] * 4,
            out_specs=hbm,
            scratch_shapes=[
                *scratch,
                *[pltpu.SemaphoreType.DMA((2,))] * 3,
                pltpu.SemaphoreType.DMA((rows // WO_A_PREP_ROWS,)),
                pltpu.SemaphoreType.DMA((1,)),
                pltpu.SemaphoreType.DMA((num_tokens // WO_A_PREP_ROWS,)),
            ],
        ),
        # Pinned: with the small VMEM limit XLA would otherwise place the 16 MB
        # output in VMEM and pay a copy back to HBM for the consumer.
        out_shape=pltpu.HBM((num_tokens, out_features), x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=scratch_bytes + WO_A_VMEM_HEADROOM_BYTES,
            disable_bounds_checks=True,
        ),
        name=name,
    )(
        # Where their producers wrote them; otherwise XLA may stage them in
        # VMEM with copies ahead of the call.
        pltpu.with_memory_space_constraint(positions, pltpu.HBM),
        pltpu.with_memory_space_constraint(x, pltpu.HBM),
        wo_a,
        pltpu.with_memory_space_constraint(
            wo_a_scale.reshape(1, out_features), pltpu.HBM
        ),
        pltpu.with_memory_space_constraint(cos_sin_cache, pltpu.HBM),
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "tile_t",
        "tile_r",
        "sub_t",
        "quantize_activations",
        "name",
    ),
)
def wo_a_projection(
    x: jax.Array,  # [T, G * H, head_dim] bf16
    wo_a: jax.Array,  # [D, G * R] fp8
    wo_a_scale: jax.Array,  # [G * R] float32
    cos_sin: jax.Array,  # [T, 2 * LANE] float32
    *,
    tile_t: int | None = None,
    tile_r: int | None = None,
    sub_t: int | None = None,
    quantize_activations: bool = True,
    name: str = "wo_a_projection",
) -> jax.Array:
    """Do DSv4 wo_a projection.

    T: num_tokens
    G: num_groups
    H: num_heads_per_group
    D: H * head_dim
    R: o_lora_dim

    Equivalent to
    x = rope(x)
    return ``einsum("tgd,dgr->tgr", x.view(t, g, d), wo_a.view(d, g, r),
    preferred_element_type=f32) * wo_a_scale.view(g, r).to(bf16)``

    ``quantize_activations`` quantizes each activation block's rows to fp8 in
    the kernel so both MXU operands are fp8.
    """
    assert x.ndim == 3
    assert x.dtype == jnp.bfloat16
    assert wo_a.ndim == 2 and wo_a_scale.ndim == 1

    num_tokens, num_heads, head_dim = x.shape
    reduction, out_features = wo_a.shape
    assert out_features == wo_a_scale.shape[0]

    assert reduction % head_dim == 0
    heads_per_group = reduction // head_dim

    # DSv4's heads_per_group is 8, exactly equal to SUBLANE
    # The kernel is based on this assumption, which makes it
    # simpler.
    assert heads_per_group == SUBLANE
    assert num_heads % heads_per_group == 0
    num_groups = num_heads // heads_per_group
    assert out_features % num_groups == 0
    lora_rank = out_features // num_groups

    assert cos_sin.ndim == 2 and cos_sin.shape == (num_tokens, 2 * LANE)
    assert head_dim % LANE == 0

    if tile_r is None:
        tile_r = lora_rank
    if tile_t is None:
        tile_t = _largest_divisor(num_tokens, cap=1024)
    if sub_t is None:
        # 128 is chosen based on large batch size (t) microbenchmarks
        sub_t = _largest_divisor(tile_t, cap=128)
    assert lora_rank % tile_r == 0
    assert num_tokens % tile_t == 0
    assert tile_t % sub_t == 0
    num_t_tiles = num_tokens // tile_t
    num_r_tiles = lora_rank // tile_r

    return pl.pallas_call(
        functools.partial(
            _kernel,
            tile_t=tile_t,
            num_sub_t=tile_t // sub_t,
            quantize_activations=quantize_activations,
            head_dim=head_dim,
        ),
        out_shape=jax.ShapeDtypeStruct((num_tokens, out_features), x.dtype),
        grid=(num_groups, num_t_tiles, num_r_tiles),
        in_specs=[
            # x: one group's heads, [t, g, 0] over [T, G * H, head_dim].
            pl.BlockSpec(
                (tile_t, heads_per_group, head_dim), lambda g, t, r: (t, g, 0)
            ),
            # wo_a: the whole reduction, column block g * R + r * tile_r.
            pl.BlockSpec((reduction, tile_r), lambda g, t, r: (0, g * num_r_tiles + r)),
            # wo_a_scale
            pl.BlockSpec((1, tile_r), lambda g, t, r: (0, g * num_r_tiles + r)),
            # cos_sin: one token block
            pl.BlockSpec((tile_t, 2 * LANE), lambda g, t, r: (t, 0)),
        ],
        out_specs=pl.BlockSpec(
            (tile_t, tile_r), lambda g, t, r: (t, g * num_r_tiles + r)
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
        name=name,
    )(x, wo_a, wo_a_scale.reshape(1, out_features), cos_sin)


@functools.partial(
    jax.jit,
    static_argnames=(
        "head_dim",
        "inverse",
        "tile_t",
        "tile_r",
        "sub_t",
        "quantize_activations",
        "name",
    ),
)
def fused_reverse_rope_wo_a_projection(
    x: jax.Array,  # [T, G * H, head_dim] bf16
    positions: jax.Array,  # [T] int
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    wo_a: jax.Array,  # [D, G * R] fp8
    wo_a_scale: jax.Array,  # [G * R] float32
    *,
    head_dim: int,
    inverse: bool = True,
    tile_t: int | None = None,
    tile_r: int | None = None,
    sub_t: int | None = None,
    quantize_activations: bool = True,
    name: str = "wo_a_projection",
) -> jax.Array:
    """``The RoPE-fused ``wo_a_projection``."""
    assert x.ndim == 3
    assert x.shape[2] == head_dim

    rows = _pipelined_rows(x.shape[0])
    if quantize_activations and tile_t is tile_r is sub_t is None and rows:
        return _wo_a_pipelined(
            x,
            positions,
            cos_sin_cache,
            wo_a,
            wo_a_scale,
            inverse=inverse,
            rows=rows,
            name=name,
        )

    cos_sin = gather_cos_sin(
        positions,
        cos_sin_cache,
        inverse=inverse,
        name=f"{name}_gather_cos_sin",
    )
    return wo_a_projection(
        x,
        wo_a,
        wo_a_scale,
        cos_sin,
        tile_t=tile_t,
        tile_r=tile_r,
        sub_t=sub_t,
        quantize_activations=quantize_activations,
        name=name,
    )
