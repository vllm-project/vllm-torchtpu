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

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.rope import (
    LANE,
    CosSinRef,
    _cos_sin_lanes,
    _rotate_gptj,
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


def _rope_heads(
    x: jax.Array,
    cos_sin: jax.Array,
    *,
    head_dim: int,
    heads_per_group: int,
) -> jax.Array:
    """Applies RoPE to the trailing ``LANE`` lanes of each head of a flat ``x``.

    ``x`` is the flat ``(rows, heads_per_group * head_dim)`` view rather than
    the 3D one. Because ``head_dim`` and ``lo`` are both multiples of ``LANE``,
    every slice boundary below falls on a 128-lane register tile, so the
    concatenate is just a lane-tile relabel. Reshaping after the RoPE instead
    forces the compiler to rebuild a 3D value and relayout it.

    Args:
        x: Flat ``(rows, heads_per_group * head_dim)`` activations.
        cos_sin: ``(rows, 2 * LANE)`` cosine half followed by sine half.
        head_dim: Per-head width; must be a multiple of ``LANE``.
        heads_per_group: Number of heads packed along the last axis of ``x``.

    Returns:
        The roped flat ``(rows, heads_per_group * head_dim)`` value.
    """
    cos = cos_sin[:, :LANE]
    sin = cos_sin[:, LANE:]
    lo = head_dim - LANE
    pieces = []
    for h in range(heads_per_group):
        base = h * head_dim
        if lo:
            pieces.append(x[:, base : base + lo])
        tail = x[:, base + lo : base + head_dim].astype(jnp.float32)
        roped = tail * cos + _rotate_gptj(tail) * sin
        pieces.append(roped.astype(x.dtype))
    return jnp.concatenate(pieces, axis=-1)


def _wo_a_sub_tile(
    x: jax.Array,  # (rows, H, head_dim) bf16
    cos_sin: jax.Array,  # (rows, 2 * LANE) f32
    rhs: jax.Array,  # (D, cols) fp8
    scale: jax.Array,  # (1, cols) f32
    *,
    head_dim: int,
    quantize_activations: bool,
    out_dtype: jnp.dtype,
) -> jax.Array:
    """One sub-tile of wo_a: rope, optional per-row fp8 quant, dot, rescale."""
    rows = x.shape[0]
    x = _rope_heads(
        x.reshape(rows, -1), cos_sin, head_dim=head_dim, heads_per_group=x.shape[1]
    )
    inv = None
    if quantize_activations:
        amax = jnp.max(jnp.abs(x), axis=1, keepdims=True)
        inv = (FP8_E4M3_MAX / jnp.maximum(amax, jnp.bfloat16(1e-30))).astype(
            jnp.bfloat16
        )
        lhs = (x * inv).astype(jnp.float8_e4m3fn)
    else:
        lhs = x

    acc = jax.lax.dot_general(
        lhs,
        rhs,
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    if quantize_activations:
        assert inv is not None
        acc = acc * (1.0 / inv.astype(jnp.float32))
    return (acc * scale).astype(out_dtype)


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
        out_ref[rows] = _wo_a_sub_tile(
            x_ref[rows],
            cos_sin[rows],
            rhs,
            scale,
            head_dim=head_dim,
            quantize_activations=quantize_activations,
            out_dtype=out_dtype,
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


# The fused o-fetch kernel covers one token tile of the fused op, up to the
# largest tested size: 2T <= 512.
O_FETCH_MAX_TOKENS = 512
# Barrier group of the chip-pair kernels; each call signals and waits once, so
# calls in program order keep the count balanced. Distinct from the fused MoE
# kernel's (0).
O_FETCH_COLLECTIVE_ID = 3


def _o_fetch_kernel(
    neighbor_mesh_index_ref,  # SMEM int32 [1]: neighboring chiplet's mesh index
    chiplet_index_ref,  # SMEM int32 [1]: the current chiplet's index (0 or 1)
    o_ref,  # VMEM (T, heads_per_step, head_dim): current tokens, this step's heads
    o_hbm,  # HBM (T, 2 * local_chiplet_heads, head_dim): current tokens, all heads
    w_ref,  # VMEM (reduction, cols_per_step) fp8
    scale_ref,  # VMEM (1, cols_per_step) f32
    cos_sin_ref,  # VMEM (2T, 2 * LANE) f32
    out_ref,  # VMEM (2T, cols_per_step)
    recv,  # VMEM (steps, T, heads_per_step, head_dim): neighboring chiplet's tokens
    send_sem,
    recv_sem,  # DMA (steps,)
    *,
    head_dim: int,
    groups_per_step: int,
    num_steps: int,
):
    """wo_a over [chiplet 0 tokens; chiplet 1 tokens] with the current chiplet's
    heads, the neighboring chiplet's rows arriving by remote DMA during the kernel.

    Step g needs the neighboring chiplet's tokens for the current chiplet's heads
    of step g, which the neighboring chiplet sends from its own `o`; the current
    chiplet sends it the matching slice of its own. Two chunks are in flight at
    a time. For T > 64 each step runs the current chiplet's matmuls before
    waiting for the neighboring chiplet's chunk; for smaller T it waits, then
    runs one matmul over both chiplets' rows. Each matmul is `_wo_a_sub_tile`,
    as in `_kernel`.
    """
    g = pl.program_id(0)
    neighbor = (neighbor_mesh_index_ref[0],)
    current_chiplet = chiplet_index_ref[0]
    neighbor_chiplet = 1 - current_chiplet
    num_tokens, heads_per_step = o_ref.shape[0], o_ref.shape[1]
    local_chiplet_heads = o_hbm.shape[1] // 2

    def send(c):
        # Send out what the neighboring chiplet needs.
        start = neighbor_chiplet * local_chiplet_heads + c * heads_per_step
        src = o_hbm.at[:, pl.ds(start, heads_per_step), :]
        return pltpu.make_async_remote_copy(
            src_ref=src,
            dst_ref=recv.at[c],
            send_sem=send_sem,
            recv_sem=recv_sem.at[c],
            device_id=neighbor,
            device_id_type=pl.DeviceIdType.MESH,
        )

    @pl.when(g == 0)
    def _():
        bsem = pltpu.get_barrier_semaphore()
        pl.semaphore_signal(
            bsem, 1, device_id=neighbor, device_id_type=pl.DeviceIdType.MESH
        )
        pl.semaphore_wait(bsem, 1)
        for c in range(min(2, num_steps)):
            send(c).start()

    heads = heads_per_step // groups_per_step
    cols = w_ref.shape[1] // groups_per_step
    neighbor_tokens = recv.at[g]

    # Output rows [0, T) are chiplet 0's tokens and [T, 2T) chiplet 1's.
    current_row0 = current_chiplet * num_tokens
    neighbor_row0 = neighbor_chiplet * num_tokens

    def wait_neighbor():
        pltpu.make_async_copy(neighbor_tokens, neighbor_tokens, recv_sem.at[g]).wait()

    def sub_tile(x, cos_sin, j):
        return _wo_a_sub_tile(
            x,
            cos_sin,
            w_ref[:, pl.ds(j * cols, cols)],
            scale_ref[:, pl.ds(j * cols, cols)],
            head_dim=head_dim,
            quantize_activations=True,
            out_dtype=o_ref.dtype,
        )

    if 2 * num_tokens > 128:
        # Larger T: one matmul per group over each chiplet's T rows; the current
        # chiplet's run while the neighboring chiplet's arrive.
        def project(src_ref, row0):
            rows = pl.ds(pl.multiple_of(row0, SUBLANE), num_tokens)
            for j in range(groups_per_step):
                x = src_ref[:, pl.ds(j * heads, heads)]
                out_ref[rows, pl.ds(j * cols, cols)] = sub_tile(x, cos_sin_ref[rows], j)

        project(o_ref, current_row0)
        wait_neighbor()
        project(neighbor_tokens, neighbor_row0)
    else:
        # Small T (2T <= 128): one 2T-row matmul of [current; neighboring] tokens
        # per group, so each group's weights stream through the MXU once; rows are
        # independent, so each half is stored at its own offset.
        current_rows = pl.ds(pl.multiple_of(current_row0, SUBLANE), num_tokens)
        neighbor_rows = pl.ds(pl.multiple_of(neighbor_row0, SUBLANE), num_tokens)
        cos_sin = jnp.concatenate(
            [cos_sin_ref[current_rows], cos_sin_ref[neighbor_rows]]
        )
        wait_neighbor()
        for j in range(groups_per_step):
            heads_j = pl.ds(j * heads, heads)
            x = jnp.concatenate([o_ref[:, heads_j], neighbor_tokens[:, heads_j]])
            y = sub_tile(x, cos_sin, j)
            out_ref[current_rows, pl.ds(j * cols, cols)] = y[:num_tokens]
            out_ref[neighbor_rows, pl.ds(j * cols, cols)] = y[num_tokens:]

    # Keep two chunks in flight: at the end of step g, send chunk g + 2.
    @pl.when(g + 2 < num_steps)
    def _():
        send(g + 2).start()

    @pl.when(g == num_steps - 1)
    def _():
        # Byte-counted: every chunk sent from the current chiplet has left.
        pltpu.make_async_copy(recv, recv, send_sem).wait()


@functools.partial(jax.jit, static_argnames=("head_dim", "inverse", "name"))
def fused_reverse_rope_wo_a_projection_o_fetch(
    o: jax.Array,  # [T, H, head_dim] bf16: current chiplet's tokens, all heads
    positions: jax.Array,  # [2T] int: chiplet 0's tokens, then chiplet 1's
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    wo_a: jax.Array,  # [reduction, local_num_groups * lora_rank] fp8: local groups
    wo_a_scale: jax.Array,  # [local_num_groups * lora_rank] float32
    neighbor_mesh_index: jax.Array,  # int32 [1]: neighboring chiplet's mesh index
    chiplet_index: jax.Array,  # int32 [1]: the current chiplet's index (0 or 1)
    *,
    head_dim: int,
    inverse: bool = True,
    name: str = "wo_a_projection_o_fetch",
) -> jax.Array:
    """Inside a shard_map over a mesh holding both chiplets of a chip:
    ``fused_reverse_rope_wo_a_projection`` (activation-quantized) of
    [chiplet 0 tokens; chiplet 1 tokens] with the current chiplet's half of the
    heads, where the neighboring chiplet's rows are fetched from its ``o`` by
    remote DMA inside the kernel, overlapped with the work on the current
    chiplet's tokens. Each output row is the fused op's on the same row, for
    2T <= O_FETCH_MAX_TOKENS.
    """
    num_tokens, num_heads, _ = o.shape
    assert o.dtype == jnp.bfloat16, o.dtype
    assert head_dim % LANE == 0, head_dim
    # Row offsets (chiplet * T, and the chunk starts) carry 8-row alignment
    # hints, so T must be a multiple of the sublane tile.
    assert num_tokens % SUBLANE == 0, f"T = {num_tokens} is not a multiple of 8"
    local_chiplet_heads = num_heads // 2
    total = 2 * num_tokens
    reduction, out_features = wo_a.shape
    assert reduction % head_dim == 0
    heads_per_group = reduction // head_dim
    # The only tested layout (DSv4: 8 heads per group).
    assert heads_per_group == SUBLANE
    assert total <= O_FETCH_MAX_TOKENS, "the pair kernel assumes a single token tile"
    assert local_chiplet_heads % heads_per_group == 0
    local_num_groups = local_chiplet_heads // heads_per_group
    assert out_features % local_num_groups == 0
    lora_rank = out_features // local_num_groups
    groups_per_step = 2 if local_num_groups % 2 == 0 else 1
    heads_per_step = groups_per_step * heads_per_group
    num_steps = local_num_groups // groups_per_step
    cols_per_step = groups_per_step * lora_rank
    cos_sin = gather_cos_sin(
        positions, cos_sin_cache, inverse=inverse, name=f"{name}_gather_cos_sin"
    )
    return pl.pallas_call(
        functools.partial(
            _o_fetch_kernel,
            head_dim=head_dim,
            groups_per_step=groups_per_step,
            num_steps=num_steps,
        ),
        out_shape=jax.ShapeDtypeStruct((total, out_features), o.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(num_steps,),
            in_specs=[
                pl.BlockSpec(
                    (num_tokens, heads_per_step, head_dim),
                    lambda g, p, c: (0, c[0] * num_steps + g, 0),
                ),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((reduction, cols_per_step), lambda g, p, c: (0, g)),
                pl.BlockSpec((1, cols_per_step), lambda g, p, c: (0, g)),
                pl.BlockSpec((total, 2 * LANE), lambda g, p, c: (0, 0)),
            ],
            out_specs=pl.BlockSpec((total, cols_per_step), lambda g, p, c: (0, g)),
            scratch_shapes=[
                pltpu.VMEM((num_steps, num_tokens, heads_per_step, head_dim), o.dtype),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((num_steps,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            collective_id=O_FETCH_COLLECTIVE_ID,
            vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
        name=name,
    )(
        neighbor_mesh_index.astype(jnp.int32),
        chiplet_index.astype(jnp.int32),
        o,
        o,
        wo_a,
        wo_a_scale.reshape(1, out_features),
        cos_sin,
    )
