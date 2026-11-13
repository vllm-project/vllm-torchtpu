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
    _lane_gather,
    _rotate_gptj,
)
from vllm_torchtpu import envs
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
    each row to ``[2 * LANE]`` in XLA so the gather CSEs across layers.
    """
    del name
    assert positions.ndim == 1
    assert cos_sin_cache.ndim == 2

    num_tokens = positions.shape[0]
    rotary_dim = cos_sin_cache.shape[1]
    assert rotary_dim % 2 == 0 and rotary_dim <= LANE

    half = rotary_dim // 2
    cs = cos_sin_cache[positions]
    pad_w = LANE - rotary_dim
    if pad_w == 0 and not inverse:
        return jnp.repeat(cs, 2, axis=-1).astype(out_dtype)
    c = jnp.repeat(cs[:, :half], 2, axis=-1)
    s = jnp.repeat(cs[:, half:], 2, axis=-1)
    if inverse:
        s = -s
    if pad_w > 0:
        ones = jnp.ones((num_tokens, pad_w), dtype=cs.dtype)
        zeros = jnp.zeros((num_tokens, pad_w), dtype=cs.dtype)
        return jnp.concatenate([ones, c, zeros, s], axis=-1).astype(out_dtype)
    return jnp.concatenate([c, s], axis=-1).astype(out_dtype)


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
        head_dim: Per-head width; a multiple of ``LANE`` and larger than it.
        heads_per_group: Number of heads packed along the last axis of ``x``.

    Returns:
        The roped flat ``(rows, heads_per_group * head_dim)`` value.
    """
    # Every head has channels before its roped tail, so no slice is empty.
    assert head_dim > LANE, f"head_dim {head_dim} must be larger than {LANE}"
    cos = cos_sin[:, :LANE]
    sin = cos_sin[:, LANE:]
    lo = head_dim - LANE
    pieces = []
    for h in range(heads_per_group):
        base = h * head_dim
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


def _rope_heads_2d(
    x_2d: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    *,
    heads_per_group: int,
    head_dim: int,
    lane_perm: jax.Array,
    sign: jax.Array,
    out_dtype: jnp.dtype,
) -> jax.Array:
    """Apply RoPE directly on 2D ``(sub_t, heads_per_group * head_dim)``."""
    lo = head_dim - LANE
    sin_signed = sin * sign
    pieces = []
    for h in range(heads_per_group):
        base_col = h * head_dim
        pieces.append(x_2d[:, base_col : base_col + lo])
        tail = x_2d[:, base_col + lo : base_col + head_dim].astype(jnp.float32)
        rot = _lane_gather(tail, lane_perm)
        roped = (tail * cos + rot * sin_signed).astype(out_dtype)
        pieces.append(roped)
    return jnp.concatenate(pieces, axis=1)


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
    inverse: bool = False,
) -> None:
    rhs = w_ref[...]
    scale = scale_ref[...]
    cos_sin = cos_sin_ref[...]
    sub_t = tile_t // num_sub_t
    out_dtype = x_ref.dtype
    lane = jnp.arange(LANE)
    lane_perm = lane ^ 1
    sign = jnp.where(lane % 2 == 0, -1.0, 1.0).astype(jnp.float32)

    def _prep_lhs(s: int):
        rows = slice(s * sub_t, (s + 1) * sub_t)
        x_2d = x_ref[rows].reshape(sub_t, SUBLANE * head_dim)
        cs = cos_sin[rows]
        sin_val = -cs[:, LANE:] if inverse else cs[:, LANE:]
        x = _rope_heads_2d(
            x_2d,
            cs[:, :LANE],
            sin_val,
            heads_per_group=SUBLANE,
            head_dim=head_dim,
            lane_perm=lane_perm,
            sign=sign,
            out_dtype=out_dtype,
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
        return lhs, inv

    lhs_curr, inv_curr = _prep_lhs(0)
    for s in range(num_sub_t):
        rows = slice(s * sub_t, (s + 1) * sub_t)
        partial = jax.lax.dot_general(
            lhs_curr,
            rhs,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        if s + 1 < num_sub_t:
            lhs_next, inv_next = _prep_lhs(s + 1)
        else:
            lhs_next, inv_next = None, None
        if quantize_activations:
            assert inv_curr is not None
            partial = partial * (1.0 / inv_curr.astype(jnp.float32))
        out_ref[rows] = (partial * scale).astype(out_dtype)
        lhs_curr, inv_curr = lhs_next, inv_next


@functools.partial(
    jax.jit,
    static_argnames=(
        "tile_t",
        "tile_r",
        "sub_t",
        "quantize_activations",
        "inverse",
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
    inverse: bool = False,
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
            inverse=inverse,
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
        inverse=False,
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
        inverse=inverse,
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


def _wq_b_q_rope_kernel_inquant(
    qr_ref: jax.Ref,  # VMEM (tile_t, K) bf16
    w_ref: jax.Ref,  # VMEM (K, heads_per_group * head_dim) fp8
    w_s_ref: jax.Ref,  # VMEM (1, heads_per_group * head_dim) f32
    cos_sin_ref: jax.Ref,  # VMEM (tile_t, 2 * LANE) f32 [cos, sin]
    out_ref: jax.Ref,  # VMEM (tile_t, heads_per_group, head_dim) bf16
    *,
    tile_t: int,
    num_sub_t: int,
    heads_per_group: int,
    head_dim: int,
) -> None:
    rhs = w_ref[...]
    rhs_s = w_s_ref[...]
    cos_sin = cos_sin_ref[...]
    sub_t = tile_t // num_sub_t
    lane = jnp.arange(LANE)
    lane_perm = lane ^ 1
    sign = jnp.where(lane % 2 == 0, -1.0, 1.0).astype(jnp.float32)

    def _prep_and_dot(s: int):
        rows = slice(s * sub_t, (s + 1) * sub_t)
        qr_f32 = qr_ref[rows].astype(jnp.float32)
        abs_max = jnp.max(jnp.abs(qr_f32), axis=1, keepdims=True)
        lhs_s = abs_max / FP8_E4M3_MAX
        lhs_s_safe = jnp.where(lhs_s == 0, 1.0, lhs_s)
        qr_fp8 = jnp.clip(qr_f32 / lhs_s_safe, -FP8_E4M3_MAX, FP8_E4M3_MAX).astype(
            jnp.float8_e4m3fn
        )
        partial = jax.lax.dot_general(
            qr_fp8,
            rhs,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        return partial, lhs_s_safe

    partial_curr, lhs_s_curr = _prep_and_dot(0)
    for s in range(num_sub_t):
        rows = slice(s * sub_t, (s + 1) * sub_t)
        if s + 1 < num_sub_t:
            partial_next, lhs_s_next = _prep_and_dot(s + 1)
        else:
            partial_next, lhs_s_next = None, None
        cs = cos_sin[rows]
        cos = cs[:, :LANE]
        sin = cs[:, LANE : 2 * LANE]
        q_bf16_2d = ((partial_curr * lhs_s_curr) * rhs_s).astype(jnp.bfloat16)
        roped_2d = _rope_heads_2d(
            q_bf16_2d,
            cos,
            sin,
            heads_per_group=heads_per_group,
            head_dim=head_dim,
            lane_perm=lane_perm,
            sign=sign,
            out_dtype=jnp.bfloat16,
        )
        out_ref[rows] = roped_2d.reshape(sub_t, heads_per_group, head_dim)
        partial_curr, lhs_s_curr = partial_next, lhs_s_next


def _wq_b_q_rope_kernel(
    qr_fp8_ref: jax.Ref,  # VMEM (tile_t, K) fp8
    w_ref: jax.Ref,  # VMEM (K, heads_per_group * head_dim) fp8
    w_s_ref: jax.Ref,  # VMEM (1, heads_per_group * head_dim) f32
    cos_sin_ref: jax.Ref,  # VMEM (tile_t, 2 * LANE) f32 [cos, sin]
    lhs_s_ref: jax.Ref,  # VMEM (tile_t, LANE) f32 [lhs_s]
    out_ref: jax.Ref,  # VMEM (tile_t, heads_per_group, head_dim) bf16
    *,
    tile_t: int,
    num_sub_t: int,
    heads_per_group: int,
    head_dim: int,
) -> None:
    rhs = w_ref[...]
    rhs_s = w_s_ref[...]
    cos_sin = cos_sin_ref[...]
    lhs_s_all = lhs_s_ref[...]
    sub_t = tile_t // num_sub_t
    lane = jnp.arange(LANE)
    lane_perm = lane ^ 1
    sign = jnp.where(lane % 2 == 0, -1.0, 1.0).astype(jnp.float32)

    def _dot(s: int):
        rows = slice(s * sub_t, (s + 1) * sub_t)
        return jax.lax.dot_general(
            qr_fp8_ref[rows],
            rhs,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

    partial_curr = _dot(0)
    for s in range(num_sub_t):
        rows = slice(s * sub_t, (s + 1) * sub_t)
        partial_next = _dot(s + 1) if s + 1 < num_sub_t else None
        cs = cos_sin[rows]
        cos = cs[:, :LANE]
        sin = cs[:, LANE : 2 * LANE]
        lhs_s = lhs_s_all[rows, :1]
        q_bf16_2d = ((partial_curr * lhs_s) * rhs_s).astype(jnp.bfloat16)
        roped_2d = _rope_heads_2d(
            q_bf16_2d,
            cos,
            sin,
            heads_per_group=heads_per_group,
            head_dim=head_dim,
            lane_perm=lane_perm,
            sign=sign,
            out_dtype=jnp.bfloat16,
        )
        out_ref[rows] = roped_2d.reshape(sub_t, heads_per_group, head_dim)
        partial_curr = partial_next


@functools.partial(
    jax.jit,
    static_argnames=("head_dim", "cap_t", "name"),
)
def fused_wq_b_q_rope_projection(
    qr: jax.Array,  # [T, K] bf16
    positions: jax.Array,  # [T] int32
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    wq_b: jax.Array,  # [K, H_q * head_dim] fp8
    wq_b_scale: jax.Array,  # [H_q * head_dim] float32
    *,
    head_dim: int,
    cap_t: int = 512,
    name: str = "wq_b_q_rope_projection",
) -> jax.Array:
    """Fused FP8 activation quant + wq_b projection + forward RoPE."""
    assert qr.ndim == 2 and qr.dtype == jnp.bfloat16
    assert wq_b.ndim == 2
    num_tokens, k_dim = qr.shape
    wq_k, out_features = wq_b.shape
    assert k_dim == wq_k
    assert out_features % head_dim == 0
    num_heads = out_features // head_dim
    if num_heads == 128 and num_tokens == 48:
        heads_per_group = 32
    elif (num_heads == 128 and num_tokens == 96) or (num_heads == 16 and num_tokens in (192, 384, 2048)):
        heads_per_group = 16
    else:
        heads_per_group = SUBLANE
    assert num_heads % heads_per_group == 0
    num_groups = num_heads // heads_per_group
    cols_per_g = heads_per_group * head_dim

    pad_t = (SUBLANE - (num_tokens % SUBLANE)) % SUBLANE
    if pad_t > 0:
        qr = jnp.pad(qr, ((0, pad_t), (0, 0)))
        positions = jnp.pad(positions, (0, pad_t))
    t_pad = qr.shape[0]
    eff_cap_t = 1024 if (num_heads >= 64 and t_pad >= 1024 and cap_t == 512) else cap_t
    tile_t = SUBLANE * _largest_divisor(t_pad // SUBLANE, cap=eff_cap_t // SUBLANE)
    sub_t = SUBLANE * _largest_divisor(tile_t // SUBLANE, cap=128 // SUBLANE)
    num_t_tiles = t_pad // tile_t

    cos_sin = gather_cos_sin(
        positions,
        cos_sin_cache,
        inverse=False,
        name=f"{name}_gather_cos_sin",
    )
    w_s_2d = wq_b_scale.reshape(1, out_features).astype(jnp.float32)

    if num_groups <= 2:
        out = pl.pallas_call(
            functools.partial(
                _wq_b_q_rope_kernel_inquant,
                tile_t=tile_t,
                num_sub_t=tile_t // sub_t,
                heads_per_group=heads_per_group,
                head_dim=head_dim,
            ),
            out_shape=jax.ShapeDtypeStruct((t_pad, num_heads, head_dim), jnp.bfloat16),
            grid=(num_groups, num_t_tiles),
            in_specs=[
                pl.BlockSpec((tile_t, k_dim), lambda g, t: (t, 0)),
                pl.BlockSpec((k_dim, cols_per_g), lambda g, t: (0, g)),
                pl.BlockSpec((1, cols_per_g), lambda g, t: (0, g)),
                pl.BlockSpec((tile_t, 2 * LANE), lambda g, t: (t, 0)),
            ],
            out_specs=pl.BlockSpec(
                (tile_t, heads_per_group, head_dim), lambda g, t: (t, g, 0)
            ),
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
                disable_bounds_checks=True,
            ),
            name=name,
        )(qr, wq_b, w_s_2d, cos_sin)
    else:
        qr_f32 = qr.astype(jnp.float32)
        abs_max = jnp.max(jnp.abs(qr_f32), axis=-1, keepdims=True)
        lhs_s = abs_max / FP8_E4M3_MAX
        lhs_s_safe = jnp.where(lhs_s == 0, 1.0, lhs_s)
        qr_fp8 = jnp.clip(qr_f32 / lhs_s_safe, -FP8_E4M3_MAX, FP8_E4M3_MAX).astype(
            jnp.float8_e4m3fn
        )
        lhs_s_128 = jnp.broadcast_to(lhs_s_safe, (t_pad, LANE))
        out = pl.pallas_call(
            functools.partial(
                _wq_b_q_rope_wide3d_kernel if heads_per_group > SUBLANE else _wq_b_q_rope_kernel,
                tile_t=tile_t,
                num_sub_t=tile_t // sub_t,
                heads_per_group=heads_per_group,
                head_dim=head_dim,
            ),
            out_shape=jax.ShapeDtypeStruct((t_pad, num_heads, head_dim), jnp.bfloat16),
            grid=(num_groups, num_t_tiles),
            in_specs=[
                pl.BlockSpec((tile_t, k_dim), lambda g, t: (t, 0)),
                pl.BlockSpec((k_dim, cols_per_g), lambda g, t: (0, g)),
                pl.BlockSpec((1, cols_per_g), lambda g, t: (0, g)),
                pl.BlockSpec((tile_t, 2 * LANE), lambda g, t: (t, 0)),
                pl.BlockSpec((tile_t, LANE), lambda g, t: (t, 0)),
            ],
            out_specs=pl.BlockSpec(
                (tile_t, heads_per_group, head_dim), lambda g, t: (t, g, 0)
            ),
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
                disable_bounds_checks=True,
            ),
            name=name,
        )(qr_fp8, wq_b, w_s_2d, cos_sin, lhs_s_128)
    return out[:num_tokens] if pad_t > 0 else out


@functools.partial(
    jax.jit,
    static_argnames=("head_dim", "cap_t", "name"),
)
def fused_wq_b_q_rope_prequant_projection(
    qr_fp8: jax.Array,  # [T, K] fp8_e4m3fn
    qr_s_128: jax.Array,  # [T, LANE] float32
    positions: jax.Array,  # [T] int32
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    wq_b: jax.Array,  # [K, H_q * head_dim] fp8
    wq_b_scale: jax.Array,  # [H_q * head_dim] float32
    *,
    head_dim: int,
    cap_t: int = 512,
    name: str = "wq_b_q_rope_prequant_projection",
) -> jax.Array:
    """Fused wq_b projection + forward RoPE consuming pre-quantized (qr_fp8, qr_s_128)."""
    assert qr_fp8.ndim == 2 and qr_fp8.dtype == jnp.float8_e4m3fn
    assert wq_b.ndim == 2
    num_tokens, k_dim = qr_fp8.shape
    wq_k, out_features = wq_b.shape
    assert k_dim == wq_k
    assert out_features % head_dim == 0
    num_heads = out_features // head_dim
    heads_per_group = SUBLANE
    assert num_heads % heads_per_group == 0
    num_groups = num_heads // heads_per_group
    cols_per_g = heads_per_group * head_dim

    pad_t = (SUBLANE - (num_tokens % SUBLANE)) % SUBLANE
    if pad_t > 0:
        qr_fp8 = jnp.pad(qr_fp8, ((0, pad_t), (0, 0)))
        qr_s_128 = jnp.pad(qr_s_128, ((0, pad_t), (0, 0)), constant_values=1.0)
        positions = jnp.pad(positions, (0, pad_t))
    t_pad = qr_fp8.shape[0]
    eff_cap_t = 1024 if (num_heads >= 64 and t_pad >= 1024 and cap_t == 512) else cap_t
    tile_t = SUBLANE * _largest_divisor(t_pad // SUBLANE, cap=eff_cap_t // SUBLANE)
    sub_t = SUBLANE * _largest_divisor(tile_t // SUBLANE, cap=128 // SUBLANE)
    num_t_tiles = t_pad // tile_t

    cos_sin = gather_cos_sin(
        positions,
        cos_sin_cache,
        inverse=False,
        name=f"{name}_gather_cos_sin",
    )
    w_s_2d = wq_b_scale.reshape(1, out_features).astype(jnp.float32)

    out = pl.pallas_call(
        functools.partial(
            _wq_b_q_rope_kernel,
            tile_t=tile_t,
            num_sub_t=tile_t // sub_t,
            heads_per_group=heads_per_group,
            head_dim=head_dim,
        ),
        out_shape=jax.ShapeDtypeStruct((t_pad, num_heads, head_dim), jnp.bfloat16),
        grid=(num_groups, num_t_tiles),
        in_specs=[
            pl.BlockSpec((tile_t, k_dim), lambda g, t: (t, 0)),
            pl.BlockSpec((k_dim, cols_per_g), lambda g, t: (0, g)),
            pl.BlockSpec((1, cols_per_g), lambda g, t: (0, g)),
            pl.BlockSpec((tile_t, 2 * LANE), lambda g, t: (t, 0)),
            pl.BlockSpec((tile_t, LANE), lambda g, t: (t, 0)),
        ],
        out_specs=pl.BlockSpec(
            (tile_t, heads_per_group, head_dim), lambda g, t: (t, g, 0)
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
        name=name,
    )(qr_fp8, wq_b, w_s_2d, cos_sin, qr_s_128)
    return out[:num_tokens] if pad_t > 0 else out

def _wo_a_wo_b_fused_g2_kernel(
    x_ref: jax.Ref,      # VMEM (tile_t, 16, head_dim) bf16
    wa_ref: jax.Ref,     # VMEM (D, 2 * R) fp8
    sa_ref: jax.Ref,     # VMEM (1, 2 * R) f32
    wb_ref: jax.Ref,     # VMEM (2 * R, H_out) fp8
    sb_ref: jax.Ref,     # VMEM (K_blocks, H_out) f32
    cs_ref: jax.Ref,     # VMEM (tile_t, 2 * LANE) f32
    out_ref: jax.Ref,    # VMEM (tile_t, H_out) bf16
    *,
    tile_t: int,
    num_sub_t: int,
    lora_rank: int,
    head_dim: int,
    inverse: bool = True,
) -> None:
    wb = wb_ref[...]
    sb = sb_ref[...]
    cos_sin = cs_ref[...]
    sub_t = tile_t // num_sub_t
    lane = jnp.arange(LANE)
    lane_perm = lane ^ 1
    sign = jnp.where(lane % 2 == 0, -1.0, 1.0).astype(jnp.float32)
    wa0 = wa_ref[:, :lora_rank]
    wa1 = wa_ref[:, lora_rank:]
    sa0 = sa_ref[:, :lora_rank]
    sa1 = sa_ref[:, lora_rank:]
    k_blocks = sb.shape[0]
    block_k = wb.shape[0] // k_blocks

    for s in range(num_sub_t):
        rows = slice(s * sub_t, (s + 1) * sub_t)
        cs = cos_sin[rows]
        sin_val = -cs[:, LANE:] if inverse else cs[:, LANE:]
        x0 = _rope_heads_2d(
            x_ref[rows, :SUBLANE, :].reshape(sub_t, SUBLANE * head_dim),
            cs[:, :LANE],
            sin_val,
            heads_per_group=SUBLANE,
            head_dim=head_dim,
            lane_perm=lane_perm,
            sign=sign,
            out_dtype=jnp.bfloat16,
        )
        inv0 = (
            FP8_E4M3_MAX / jnp.maximum(jnp.max(jnp.abs(x0), axis=1, keepdims=True), jnp.bfloat16(1e-30))
        ).astype(jnp.bfloat16)
        lhs0 = (x0 * inv0).astype(jnp.float8_e4m3fn)
        p0 = jax.lax.dot_general(
            lhs0, wa0, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )

        x1 = _rope_heads_2d(
            x_ref[rows, SUBLANE:, :].reshape(sub_t, SUBLANE * head_dim),
            cs[:, :LANE],
            sin_val,
            heads_per_group=SUBLANE,
            head_dim=head_dim,
            lane_perm=lane_perm,
            sign=sign,
            out_dtype=jnp.bfloat16,
        )
        inv1 = (
            FP8_E4M3_MAX / jnp.maximum(jnp.max(jnp.abs(x1), axis=1, keepdims=True), jnp.bfloat16(1e-30))
        ).astype(jnp.bfloat16)
        lhs1 = (x1 * inv1).astype(jnp.float8_e4m3fn)
        z0 = ((p0 * (1.0 / inv0.astype(jnp.float32))) * sa0).astype(jnp.bfloat16)
        p1 = jax.lax.dot_general(
            lhs1, wa1, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )
        z1 = ((p1 * (1.0 / inv1.astype(jnp.float32))) * sa1).astype(jnp.bfloat16)
        z_row = jnp.concatenate([z0, z1], axis=1)

        if k_blocks == 1:
            z_amax = jnp.max(jnp.abs(z_row), axis=1, keepdims=True)
            z_sc = jnp.where(
                z_amax == 0,
                jnp.bfloat16(1.0),
                z_amax / jnp.bfloat16(FP8_E4M3_MAX),
            )
            z_q = (z_row / z_sc).clip(-FP8_E4M3_MAX, FP8_E4M3_MAX).astype(jnp.float8_e4m3fn)
            out_f32 = jax.lax.dot_general(
                z_q, wb, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
            )
            out_ref[rows] = (out_f32 * z_sc.astype(jnp.float32) * sb).astype(jnp.bfloat16)
        else:
            acc = jnp.zeros((sub_t, wb.shape[1]), dtype=jnp.float32)
            for kb in range(k_blocks):
                zb = z_row[:, kb * block_k : (kb + 1) * block_k]
                zb_amax = jnp.max(jnp.abs(zb), axis=1, keepdims=True)
                zb_sc = jnp.where(
                    zb_amax == 0,
                    jnp.bfloat16(1.0),
                    zb_amax / jnp.bfloat16(FP8_E4M3_MAX),
                )
                zb_q = (zb / zb_sc).clip(-FP8_E4M3_MAX, FP8_E4M3_MAX).astype(jnp.float8_e4m3fn)
                dot_b = jax.lax.dot_general(
                    zb_q,
                    wb[kb * block_k : (kb + 1) * block_k, :],
                    (((1,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
                acc = acc + dot_b * zb_sc.astype(jnp.float32) * sb[kb : kb + 1, :]
            out_ref[rows] = acc.astype(jnp.bfloat16)


@functools.partial(
    jax.jit,
    static_argnames=("head_dim", "inverse", "name"),
)
def fused_reverse_rope_wo_a_wo_b_projection(
    x: jax.Array,           # [T, 16, head_dim] bf16
    positions: jax.Array,   # [T] int32
    cos_sin_cache: jax.Array,
    wo_a: jax.Array,        # [D, 2 * R] fp8
    wo_a_scale: jax.Array,  # [2 * R] f32
    wo_b: jax.Array,        # [2 * R, H_out] fp8
    wo_b_scale: jax.Array,  # [H_out] or [1, K_blocks, 1, H_out] f32
    *,
    head_dim: int,
    inverse: bool = True,
    name: str = "fused_wo_a_wo_b",
) -> jax.Array:
    """Single-kernel fused reverse-RoPE + wo_a + VMEM quantize + wo_b for H=16 (G=2)."""
    num_tokens, num_heads, _ = x.shape
    pad_t = ((num_tokens + 7) & ~7) - num_tokens
    if pad_t > 0:
        x = jnp.pad(x, ((0, pad_t), (0, 0), (0, 0)))
        positions = jnp.pad(positions, (0, pad_t))
    t_pad = x.shape[0]
    reduction, out_features = wo_a.shape
    _, hidden_dim = wo_b.shape
    lora_rank = out_features // 2
    cap = 256 if t_pad == 768 else 512
    tile_t = SUBLANE * _largest_divisor(t_pad // SUBLANE, cap=cap // SUBLANE)
    sub_t = SUBLANE * _largest_divisor(tile_t // SUBLANE, cap=128 // SUBLANE)
    cos_sin = gather_cos_sin(
        positions, cos_sin_cache, inverse=False, name=f"{name}_gather_cos_sin"
    )
    sb_2d = (
        wo_b_scale.reshape(1, hidden_dim)
        if wo_b_scale.ndim == 1
        else wo_b_scale.reshape(-1, hidden_dim)
    )
    k_blocks = sb_2d.shape[0]
    out = pl.pallas_call(
        functools.partial(
            _wo_a_wo_b_fused_g2_kernel,
            tile_t=tile_t,
            num_sub_t=tile_t // sub_t,
            lora_rank=lora_rank,
            head_dim=head_dim,
            inverse=True,
        ),
        out_shape=jax.ShapeDtypeStruct((t_pad, hidden_dim), jnp.bfloat16),
        grid=(t_pad // tile_t,),
        in_specs=[
            pl.BlockSpec((tile_t, num_heads, head_dim), lambda t: (t, 0, 0)),
            pl.BlockSpec((reduction, out_features), lambda t: (0, 0)),
            pl.BlockSpec((1, out_features), lambda t: (0, 0)),
            pl.BlockSpec((out_features, hidden_dim), lambda t: (0, 0)),
            pl.BlockSpec((k_blocks, hidden_dim), lambda t: (0, 0)),
            pl.BlockSpec((tile_t, 2 * LANE), lambda t: (t, 0)),
        ],
        out_specs=pl.BlockSpec((tile_t, hidden_dim), lambda t: (t, 0)),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
        name=name,
    )(
        x,
        wo_a,
        wo_a_scale.reshape(1, out_features),
        wo_b,
        sb_2d,
        cos_sin,
    )
    return out[:num_tokens] if pad_t > 0 else out


def _wo_a_wo_b_h128_vmem_acc_kernel(
    x_ref: jax.Ref,      # VMEM (sub_t, 16, head_dim) bf16
    wa_ref: jax.Ref,     # VMEM (D, 2048) fp8
    sa_ref: jax.Ref,     # VMEM (1, 2048) f32
    wb_ref: jax.Ref,     # VMEM (2048, 5120) fp8
    sb_ref: jax.Ref,     # VMEM (1, k_sub, 1, 5120) f32
    cs_ref: jax.Ref,     # VMEM (sub_t, 2 * LANE) f32
    out_ref: jax.Ref,    # VMEM (sub_t, 5120) bf16
    acc_vmem: jax.Ref,   # VMEM scratch (tile_t, 5120) f32
    *,
    num_blocks: int,
    sub_t: int,
    num_sub_t: int,
    lora_rank: int,
    head_dim: int,
    inverse: bool = True,
) -> None:
    b = pl.program_id(1)
    s = pl.program_id(2)
    cs = cs_ref[...]
    sin_val = -cs[:, LANE:] if inverse else cs[:, LANE:]
    lane = jnp.arange(LANE)
    lane_perm = lane ^ 1
    sign = jnp.where(lane % 2 == 0, -1.0, 1.0).astype(jnp.float32)

    x_2d_0 = x_ref[:, :SUBLANE, :].reshape(sub_t, SUBLANE * head_dim)
    x_r0 = _rope_heads_2d(
        x_2d_0,
        cs[:, :LANE],
        sin_val,
        heads_per_group=SUBLANE,
        head_dim=head_dim,
        lane_perm=lane_perm,
        sign=sign,
        out_dtype=jnp.bfloat16,
    )
    inv0 = (
        FP8_E4M3_MAX / jnp.maximum(jnp.max(jnp.abs(x_r0), axis=1, keepdims=True), jnp.bfloat16(1e-30))
    ).astype(jnp.bfloat16)
    lhs0 = (x_r0 * inv0).astype(jnp.float8_e4m3fn)
    p0 = jax.lax.dot_general(
        lhs0, wa_ref[:, :lora_rank], (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
    )

    x_2d_1 = x_ref[:, SUBLANE:, :].reshape(sub_t, SUBLANE * head_dim)
    x_r1 = _rope_heads_2d(
        x_2d_1,
        cs[:, :LANE],
        sin_val,
        heads_per_group=SUBLANE,
        head_dim=head_dim,
        lane_perm=lane_perm,
        sign=sign,
        out_dtype=jnp.bfloat16,
    )
    inv1 = (
        FP8_E4M3_MAX / jnp.maximum(jnp.max(jnp.abs(x_r1), axis=1, keepdims=True), jnp.bfloat16(1e-30))
    ).astype(jnp.bfloat16)
    lhs1 = (x_r1 * inv1).astype(jnp.float8_e4m3fn)
    z0 = ((p0 * (1.0 / inv0.astype(jnp.float32))) * sa_ref[:, :lora_rank]).astype(jnp.bfloat16)
    p1 = jax.lax.dot_general(
        lhs1, wa_ref[:, lora_rank:], (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
    )
    z1 = ((p1 * (1.0 / inv1.astype(jnp.float32))) * sa_ref[:, lora_rank:]).astype(jnp.bfloat16)

    z_b = jnp.concatenate([z0, z1], axis=1)
    wb = wb_ref[...]
    k_sub = sb_ref.shape[1]
    if k_sub == 1:
        zb_inv = (
            FP8_E4M3_MAX / jnp.maximum(jnp.max(jnp.abs(z_b), axis=1, keepdims=True), jnp.bfloat16(1e-30))
        ).astype(jnp.bfloat16)
        zb_q = (z_b * zb_inv).astype(jnp.float8_e4m3fn)
        dot_b = jax.lax.dot_general(
            zb_q, wb, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )
        step_out = dot_b * (1.0 / zb_inv.astype(jnp.float32)) * sb_ref[0, 0, :, :]
    else:
        sub_k = wb.shape[0] // k_sub
        step_out = jnp.zeros((sub_t, wb.shape[1]), dtype=jnp.float32)
        for kb in range(k_sub):
            z_chunk = z_b[:, kb * sub_k : (kb + 1) * sub_k]
            zb_inv = (
                FP8_E4M3_MAX / jnp.maximum(jnp.max(jnp.abs(z_chunk), axis=1, keepdims=True), jnp.bfloat16(1e-30))
            ).astype(jnp.bfloat16)
            zb_q = (z_chunk * zb_inv).astype(jnp.float8_e4m3fn)
            dot_b = jax.lax.dot_general(
                zb_q,
                wb[kb * sub_k : (kb + 1) * sub_k, :],
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            step_out = step_out + dot_b * (1.0 / zb_inv.astype(jnp.float32)) * sb_ref[0, kb, :, :]

    for s_const in range(num_sub_t):
        @pl.when(s == s_const)
        def _upd():
            rows = slice(s_const * sub_t, (s_const + 1) * sub_t)
            prev = jnp.where(
                b == 0,
                jnp.zeros((sub_t, wb.shape[1]), dtype=jnp.float32),
                acc_vmem[rows],
            )
            nxt = prev + step_out
            acc_vmem[rows] = nxt
            @pl.when(b == num_blocks - 1)
            def _emit():
                out_ref[...] = nxt.astype(jnp.bfloat16)


def _wo_a_wo_b_h128_weight_outer_kernel(
    x_ref: jax.Ref,        # VMEM (tile_t, 16, head_dim) bf16
    wa_ref: jax.Ref,       # VMEM (D, 2048) fp8
    sa_ref: jax.Ref,       # VMEM (1, 2048) f32
    wb_ref: jax.Ref,       # VMEM (2048, 5120) fp8
    sb_ref: jax.Ref,       # VMEM (1, k_sub, 1, 5120) f32
    cs_ref: jax.Ref,       # VMEM (tile_t, 2 * LANE) f32
    acc_in_ref: jax.Ref,   # VMEM (tile_t, 5120) bf16 (aliased)
    acc_out_ref: jax.Ref,  # VMEM (tile_t, 5120) bf16 (aliased)
    *,
    tile_t: int,
    num_sub_t: int,
    lora_rank: int,
    head_dim: int,
    inverse: bool = True,
) -> None:
    b = pl.program_id(0)
    wa = wa_ref[...]
    wb = wb_ref[...]
    cos_sin = cs_ref[...]
    sub_t = tile_t // num_sub_t
    lane = jnp.arange(LANE)
    lane_perm = lane ^ 1
    sign = jnp.where(lane % 2 == 0, -1.0, 1.0).astype(jnp.float32)
    k_sub = sb_ref.shape[1]

    for s in range(num_sub_t):
        rows = slice(s * sub_t, (s + 1) * sub_t)
        cs = cos_sin[rows]
        sin_val = -cs[:, LANE:] if inverse else cs[:, LANE:]
        x0 = _rope_heads_2d(
            x_ref[rows, :SUBLANE, :].reshape(sub_t, SUBLANE * head_dim),
            cs[:, :LANE],
            sin_val,
            heads_per_group=SUBLANE,
            head_dim=head_dim,
            lane_perm=lane_perm,
            sign=sign,
            out_dtype=jnp.bfloat16,
        )
        inv0 = (
            FP8_E4M3_MAX / jnp.maximum(jnp.max(jnp.abs(x0), axis=1, keepdims=True), jnp.bfloat16(1e-30))
        ).astype(jnp.bfloat16)
        lhs0 = (x0 * inv0).astype(jnp.float8_e4m3fn)
        p0 = jax.lax.dot_general(
            lhs0, wa[:, :lora_rank], (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )

        x1 = _rope_heads_2d(
            x_ref[rows, SUBLANE:, :].reshape(sub_t, SUBLANE * head_dim),
            cs[:, :LANE],
            sin_val,
            heads_per_group=SUBLANE,
            head_dim=head_dim,
            lane_perm=lane_perm,
            sign=sign,
            out_dtype=jnp.bfloat16,
        )
        inv1 = (
            FP8_E4M3_MAX / jnp.maximum(jnp.max(jnp.abs(x1), axis=1, keepdims=True), jnp.bfloat16(1e-30))
        ).astype(jnp.bfloat16)
        lhs1 = (x1 * inv1).astype(jnp.float8_e4m3fn)
        z0 = ((p0 * (1.0 / inv0.astype(jnp.float32))) * sa_ref[:, :lora_rank]).astype(jnp.bfloat16)
        p1 = jax.lax.dot_general(
            lhs1, wa[:, lora_rank:], (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )
        z1 = ((p1 * (1.0 / inv1.astype(jnp.float32))) * sa_ref[:, lora_rank:]).astype(jnp.bfloat16)

        z_b = jnp.concatenate([z0, z1], axis=1)
        if k_sub == 1:
            zb_inv = (
                FP8_E4M3_MAX / jnp.maximum(jnp.max(jnp.abs(z_b), axis=1, keepdims=True), jnp.bfloat16(1e-30))
            ).astype(jnp.bfloat16)
            zb_q = (z_b * zb_inv).astype(jnp.float8_e4m3fn)
            dot_b = jax.lax.dot_general(
                zb_q, wb, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
            )
            step_out = dot_b * (1.0 / zb_inv.astype(jnp.float32)) * sb_ref[0, 0, :, :]
        else:
            sub_k = wb.shape[0] // k_sub
            step_out = jnp.zeros((sub_t, wb.shape[1]), dtype=jnp.float32)
            for kb in range(k_sub):
                z_chunk = z_b[:, kb * sub_k : (kb + 1) * sub_k]
                zb_inv = (
                    FP8_E4M3_MAX / jnp.maximum(jnp.max(jnp.abs(z_chunk), axis=1, keepdims=True), jnp.bfloat16(1e-30))
                ).astype(jnp.bfloat16)
                zb_q = (z_chunk * zb_inv).astype(jnp.float8_e4m3fn)
                dot_b = jax.lax.dot_general(
                    zb_q,
                    wb[kb * sub_k : (kb + 1) * sub_k, :],
                    (((1,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
                step_out = step_out + dot_b * (1.0 / zb_inv.astype(jnp.float32)) * sb_ref[0, kb, :, :]
        prev = jnp.where(
            b == 0,
            jnp.zeros((sub_t, wb.shape[1]), dtype=jnp.float32),
            acc_in_ref[rows].astype(jnp.float32),
        )
        acc_out_ref[rows] = (prev + step_out).astype(jnp.bfloat16)


@functools.partial(
    jax.jit,
    static_argnames=("head_dim", "inverse", "name"),
)
def fused_reverse_rope_wo_a_wo_b_h128_projection(
    x: jax.Array,           # [T, H, head_dim] bf16 where H in {64, 128}
    positions: jax.Array,   # [T] int32
    cos_sin_cache: jax.Array,
    wo_a: jax.Array,        # [D, 128 * H] fp8
    wo_a_scale: jax.Array,  # [128 * H] f32
    wo_b: jax.Array,        # [128 * H, 5120] fp8
    wo_b_scale: jax.Array,  # [5120] or [1, K_blocks, 1, 5120] f32
    *,
    head_dim: int,
    inverse: bool = True,
    name: str = "fused_wo_a_wo_b_h128",
) -> jax.Array:
    """Streaming VMEM / Weight-Outer single-kernel reverse-RoPE + wo_a + quant + wo_b
    for multi-head MLA configurations (H=64 CED / TP=2 and H=128 TP_ATTN=1 / DP=8).
    """
    num_tokens, num_heads, _ = x.shape
    num_blocks = num_heads // 16
    pad_t = ((num_tokens + 7) & ~7) - num_tokens
    if pad_t > 0:
        x = jnp.pad(x, ((0, pad_t), (0, 0), (0, 0)))
        positions = jnp.pad(positions, (0, pad_t))
    t_pad = x.shape[0]
    reduction, out_features = wo_a.shape
    _, hidden_dim = wo_b.shape
    lora_rank = out_features // (2 * num_blocks)

    cos_sin = gather_cos_sin(
        positions, cos_sin_cache, inverse=False, name=f"{name}_gather_cos_sin"
    )
    if wo_b_scale.ndim == 1:
        sb_4d = jnp.broadcast_to(
            wo_b_scale.reshape(1, 1, 1, hidden_dim), (1, num_blocks, 1, hidden_dim)
        )
    elif wo_b_scale.shape[1] < num_blocks:
        sb_4d = jnp.repeat(
            wo_b_scale, num_blocks // wo_b_scale.shape[1], axis=1
        )
    else:
        sb_4d = wo_b_scale
    k_sub = sb_4d.shape[1] // num_blocks

    # [merge] Decode (D) keeps the f32 VMEM accumulator up to t_pad <= 384
    # (compile sizes 48/192/384), sub-tiling rows above 256; the prefill
    # node (VLLM_DSV41_PURE_20L_PREFILL=1) keeps P's t_pad <= 256 cut-over
    # to the Weight-Outer kernel.
    vmem_max_t = 256 if envs.VLLM_DSV41_PURE_20L_PREFILL else 384
    if t_pad <= vmem_max_t:
        sub_t = (
            t_pad
            if t_pad <= 256
            else SUBLANE * _largest_divisor(t_pad // SUBLANE, cap=128 // SUBLANE)
        )
        num_sub_t = t_pad // sub_t
        out = pl.pallas_call(
            functools.partial(
                _wo_a_wo_b_h128_vmem_acc_kernel,
                num_blocks=num_blocks,
                sub_t=sub_t,
                num_sub_t=num_sub_t,
                lora_rank=lora_rank,
                head_dim=head_dim,
                inverse=inverse,
            ),
            out_shape=jax.ShapeDtypeStruct((t_pad, hidden_dim), jnp.bfloat16),
            scratch_shapes=[pltpu.VMEM((t_pad, hidden_dim), jnp.float32)],
            grid=(1, num_blocks, num_sub_t),
            in_specs=[
                pl.BlockSpec((sub_t, 16, head_dim), lambda t, b, s: (s, b, 0)),
                pl.BlockSpec((reduction, 2048), lambda t, b, s: (0, b)),
                pl.BlockSpec((1, 2048), lambda t, b, s: (0, b)),
                pl.BlockSpec((2048, hidden_dim), lambda t, b, s: (b, 0)),
                pl.BlockSpec((1, k_sub, 1, hidden_dim), lambda t, b, s: (0, b, 0, 0)),
                pl.BlockSpec((sub_t, 2 * LANE), lambda t, b, s: (s, 0)),
            ],
            out_specs=pl.BlockSpec((sub_t, hidden_dim), lambda t, b, s: (s, 0)),
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
                disable_bounds_checks=True,
                dimension_semantics=("parallel", "arbitrary", "arbitrary"),
            ),
            name=f"{name}_h{num_heads}_vmem",
        )(x, wo_a, wo_a_scale.reshape(1, out_features), wo_b, sb_4d, cos_sin)
        return out[:num_tokens] if pad_t > 0 else out

    tile_t = SUBLANE * _largest_divisor(t_pad // SUBLANE, cap=256 // SUBLANE)
    out_init = jax.lax.empty((t_pad, hidden_dim), dtype=jnp.bfloat16)
    out = pl.pallas_call(
        functools.partial(
            _wo_a_wo_b_h128_weight_outer_kernel,
            tile_t=tile_t,
            num_sub_t=1,
            lora_rank=lora_rank,
            head_dim=head_dim,
            inverse=inverse,
        ),
        out_shape=jax.ShapeDtypeStruct((t_pad, hidden_dim), jnp.bfloat16),
        grid=(num_blocks, t_pad // tile_t),
        in_specs=[
            pl.BlockSpec((tile_t, 16, head_dim), lambda b, t: (t, b, 0)),
            pl.BlockSpec((reduction, 2048), lambda b, t: (0, b)),
            pl.BlockSpec((1, 2048), lambda b, t: (0, b)),
            pl.BlockSpec((2048, hidden_dim), lambda b, t: (b, 0)),
            pl.BlockSpec((1, k_sub, 1, hidden_dim), lambda b, t: (0, b, 0, 0)),
            pl.BlockSpec((tile_t, 2 * LANE), lambda b, t: (t, 0)),
            pl.BlockSpec((tile_t, hidden_dim), lambda b, t: (t, 0)),
        ],
        out_specs=pl.BlockSpec((tile_t, hidden_dim), lambda b, t: (t, 0)),
        input_output_aliases={6: 0},
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
            dimension_semantics=("arbitrary", "parallel"),
        ),
        name=f"{name}_wouter",
    )(x, wo_a, wo_a_scale.reshape(1, out_features), wo_b, sb_4d, cos_sin, out_init)
    return out[:num_tokens] if pad_t > 0 else out


def _wo_a_with_amax_kernel(
    x_ref: jax.Ref,       # VMEM (tile_t, 8, head_dim) bf16
    w_ref: jax.Ref,       # VMEM (D, 1024) fp8
    scale_ref: jax.Ref,   # VMEM (1, 1024) f32
    cs_ref: jax.Ref,      # VMEM (tile_t, 2 * LANE) f32
    out_ref: jax.Ref,     # VMEM (tile_t, 1024) bf16
    amax_ref: jax.Ref,    # VMEM (1, tile_t, 128) bf16
    *,
    tile_t: int,
    num_sub_t: int,
    head_dim: int,
    inverse: bool = True,
) -> None:
    rhs = w_ref[...]
    scale = scale_ref[...]
    cos_sin = cs_ref[...]
    sub_t = tile_t // num_sub_t
    lane = jnp.arange(LANE)
    lane_perm = lane ^ 1
    sign = jnp.where(lane % 2 == 0, -1.0, 1.0).astype(jnp.float32)

    def _prep_lhs(s: int):
        rows = slice(s * sub_t, (s + 1) * sub_t)
        x_2d = x_ref[rows].reshape(sub_t, SUBLANE * head_dim)
        cs = cos_sin[rows]
        sin_val = -cs[:, LANE:] if inverse else cs[:, LANE:]
        x_r = _rope_heads_2d(
            x_2d,
            cs[:, :LANE],
            sin_val,
            heads_per_group=SUBLANE,
            head_dim=head_dim,
            lane_perm=lane_perm,
            sign=sign,
            out_dtype=jnp.bfloat16,
        )
        amax = jnp.max(jnp.abs(x_r), axis=1, keepdims=True)
        inv = (FP8_E4M3_MAX / jnp.maximum(amax, jnp.bfloat16(1e-30))).astype(jnp.bfloat16)
        return (x_r * inv).astype(jnp.float8_e4m3fn), inv

    lhs_curr, inv_curr = _prep_lhs(0)
    for s in range(num_sub_t):
        rows = slice(s * sub_t, (s + 1) * sub_t)
        partial = jax.lax.dot_general(
            lhs_curr, rhs, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )
        if s + 1 < num_sub_t:
            lhs_next, inv_next = _prep_lhs(s + 1)
        else:
            lhs_next, inv_next = None, None
        partial = partial * (1.0 / inv_curr.astype(jnp.float32))
        z_g = (partial * scale).astype(jnp.bfloat16)
        out_ref[rows] = z_g
        amax_g = jnp.max(jnp.abs(z_g), axis=1, keepdims=True)
        amax_ref[:, rows, :] = jnp.broadcast_to(amax_g[None], (1, sub_t, 128))
        lhs_curr, inv_curr = lhs_next, inv_next


@functools.partial(
    jax.jit,
    static_argnames=("head_dim", "inverse", "name"),
)
def fused_reverse_rope_wo_a_wo_b_h128_1d_exact_projection(
    x: jax.Array,           # [T, 128, head_dim] bf16
    positions: jax.Array,   # [T] int32
    cos_sin_cache: jax.Array,
    wo_a: jax.Array,        # [D, 16384] fp8
    wo_a_scale: jax.Array,  # [16384] f32
    wo_b: jax.Array,        # [16384, 5120] fp8
    wo_b_scale: jax.Array,  # [5120] f32
    *,
    head_dim: int,
    inverse: bool = True,
    name: str = "fused_wo_a_wo_b_h128_1d_exact",
) -> jax.Array:
    """Bit-exact (0.00e+00) VMEM group-amax pre-reduction + fused scale projection for DP=8 H=128.

    Emits per-group row max |z_g| directly from VMEM registers alongside z_g,
    eliminating XLA's separate 268.44 MB HBM amax scan over [T, 16384] while
    matching xla_quantized_matmul bit-for-bit (0.00e+00 max diff).
    """
    orig_m, num_heads, _ = x.shape
    pad_t = ((orig_m + 7) & ~7) - orig_m
    x_p = jnp.pad(x, ((0, pad_t), (0, 0), (0, 0))) if pad_t > 0 else x
    p_p = jnp.pad(positions, (0, pad_t)) if pad_t > 0 else positions
    t_pad = x_p.shape[0]
    reduction, out_features = wo_a.shape
    num_groups = num_heads // SUBLANE
    lora_rank = out_features // num_groups
    tile_t = SUBLANE * _largest_divisor(t_pad // SUBLANE, cap=1024 // SUBLANE)
    sub_t = SUBLANE * _largest_divisor(tile_t // SUBLANE, cap=128 // SUBLANE)
    cos_sin = gather_cos_sin(
        p_p, cos_sin_cache, inverse=False, name=f"{name}_gather_cos_sin"
    )
    z_p, amax_16 = pl.pallas_call(
        functools.partial(
            _wo_a_with_amax_kernel,
            tile_t=tile_t,
            num_sub_t=tile_t // sub_t,
            head_dim=head_dim,
            inverse=inverse,
        ),
        out_shape=(
            jax.ShapeDtypeStruct((t_pad, out_features), jnp.bfloat16),
            jax.ShapeDtypeStruct((num_groups, t_pad, 128), jnp.bfloat16),
        ),
        grid=(num_groups, t_pad // tile_t),
        in_specs=[
            pl.BlockSpec((tile_t, SUBLANE, head_dim), lambda g, t: (t, g, 0)),
            pl.BlockSpec((reduction, lora_rank), lambda g, t: (0, g)),
            pl.BlockSpec((1, lora_rank), lambda g, t: (0, g)),
            pl.BlockSpec((tile_t, 2 * LANE), lambda g, t: (t, 0)),
        ],
        out_specs=(
            pl.BlockSpec((tile_t, lora_rank), lambda g, t: (t, g)),
            pl.BlockSpec((1, tile_t, 128), lambda g, t: (g, t, 0)),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
        name=name,
    )(x_p, wo_a, wo_a_scale.reshape(1, out_features), cos_sin)

    z = z_p[:orig_m] if pad_t > 0 else z_p
    amax_row = jnp.max(
        amax_16[:, :orig_m, :1] if pad_t > 0 else amax_16[:, :, :1], axis=0
    )
    f8_max = jnp.finfo(jnp.float8_e4m3fn).max.astype(jnp.float32)
    f8_min = jnp.finfo(jnp.float8_e4m3fn).min.astype(jnp.float32)
    scale = amax_row / f8_max
    scale = jnp.where(scale == 0, 1.0, scale)
    z_q = (z / scale).clip(f8_min, f8_max).astype(jnp.float8_e4m3fn)
    out = jax.lax.dot_general(
        z_q, wo_b, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
    ).astype(jnp.float32)
    out = out * scale * wo_b_scale[None, :]
    return out.astype(jnp.bfloat16)

def _wqa_wkv_norm_rope_quant_kernel(
    h_ref: jax.Ref,         # VMEM (tile_n, 5120) bf16
    w_wqa_ref: jax.Ref,     # VMEM (5120, 2048) fp8
    ws_wqa_ref: jax.Ref,    # VMEM (1, 2048) f32
    q_w_ref: jax.Ref,       # VMEM (1, 1536) bf16
    kv_w_ref: jax.Ref,      # VMEM (1, 512) bf16
    cs_ref: jax.Ref,        # VMEM (tile_n, 256) f32
    qr_fp8_ref: jax.Ref,    # VMEM (tile_n, 1536) fp8
    qr_s_ref: jax.Ref,      # VMEM (tile_n, 128) f32
    kv_out_ref: jax.Ref,    # VMEM (tile_n, 512) bf16
    *,
    q_dim: int,
    q_eps: float,
    kv_eps: float,
) -> None:
    from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor

    h_q, h_sc = quantize_tensor(h_ref[...], jnp.float8_e4m3fn)
    qr_kv = (
        jax.lax.dot_general(
            h_q,
            w_wqa_ref[...],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        * h_sc
        * ws_wqa_ref[...]
    ).astype(jnp.bfloat16)

    q_x = qr_kv[:, :q_dim].astype(jnp.float32)
    kv_x = qr_kv[:, q_dim:].astype(jnp.float32)

    q_rms = jax.lax.rsqrt(jnp.mean(q_x * q_x, axis=-1, keepdims=True) + q_eps)
    qr_normed = (q_x * q_rms).astype(jnp.bfloat16) * q_w_ref[...]
    qr_f32 = qr_normed.astype(jnp.float32)
    abs_max = jnp.max(jnp.abs(qr_f32), axis=-1, keepdims=True)
    lhs_s = abs_max / FP8_E4M3_MAX
    lhs_s_safe = jnp.where(lhs_s == 0.0, 1.0, lhs_s)
    qr_fp8_ref[...] = jnp.clip(
        qr_f32 / lhs_s_safe, -FP8_E4M3_MAX, FP8_E4M3_MAX
    ).astype(jnp.float8_e4m3fn)
    qr_s_ref[...] = jnp.broadcast_to(lhs_s_safe, (qr_kv.shape[0], LANE))

    kv_rms = jax.lax.rsqrt(jnp.mean(kv_x * kv_x, axis=-1, keepdims=True) + kv_eps)
    kv_normed = (kv_x * kv_rms).astype(jnp.bfloat16) * kv_w_ref[...]
    tail = kv_normed[:, -LANE:].astype(jnp.float32)
    cs = cs_ref[...]
    roped = (tail * cs[:, :LANE] + _rotate_gptj(tail) * cs[:, LANE:]).astype(
        jnp.bfloat16
    )
    kv_out_ref[...] = jnp.concatenate([kv_normed[:, :-LANE], roped], axis=-1)


def _wqa_wkv_prequant_norm_rope_quant_kernel(
    h_fp8_ref: jax.Ref,     # VMEM (tile_n, 5120) fp8
    h_sc_ref: jax.Ref,      # VMEM (tile_n, 128) f32
    w_wqa_ref: jax.Ref,     # VMEM (5120, 2048) fp8
    ws_wqa_ref: jax.Ref,    # VMEM (1, 2048) f32
    q_w_ref: jax.Ref,       # VMEM (1, 1536) bf16
    kv_w_ref: jax.Ref,      # VMEM (1, 512) bf16
    cs_ref: jax.Ref,        # VMEM (tile_n, 256) f32
    qr_fp8_ref: jax.Ref,    # VMEM (tile_n, 1536) fp8
    qr_s_ref: jax.Ref,      # VMEM (tile_n, 128) f32
    kv_out_ref: jax.Ref,    # VMEM (tile_n, 512) bf16
    *,
    q_dim: int,
    q_eps: float,
    kv_eps: float,
) -> None:
    qr_kv = (
        jax.lax.dot_general(
            h_fp8_ref[...],
            w_wqa_ref[...],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        * h_sc_ref[:, :1]
        * ws_wqa_ref[...]
    ).astype(jnp.bfloat16)

    q_x = qr_kv[:, :q_dim].astype(jnp.float32)
    kv_x = qr_kv[:, q_dim:].astype(jnp.float32)

    q_rms = jax.lax.rsqrt(jnp.mean(q_x * q_x, axis=-1, keepdims=True) + q_eps)
    qr_normed = (q_x * q_rms).astype(jnp.bfloat16) * q_w_ref[...]
    qr_f32 = qr_normed.astype(jnp.float32)
    abs_max = jnp.max(jnp.abs(qr_f32), axis=-1, keepdims=True)
    lhs_s = abs_max / FP8_E4M3_MAX
    lhs_s_safe = jnp.where(lhs_s == 0.0, 1.0, lhs_s)
    qr_fp8_ref[...] = jnp.clip(
        qr_f32 / lhs_s_safe, -FP8_E4M3_MAX, FP8_E4M3_MAX
    ).astype(jnp.float8_e4m3fn)
    qr_s_ref[...] = jnp.broadcast_to(lhs_s_safe, (qr_kv.shape[0], LANE))

    kv_rms = jax.lax.rsqrt(jnp.mean(kv_x * kv_x, axis=-1, keepdims=True) + kv_eps)
    kv_normed = (kv_x * kv_rms).astype(jnp.bfloat16) * kv_w_ref[...]
    tail = kv_normed[:, -LANE:].astype(jnp.float32)
    cs = cs_ref[...]
    roped = (tail * cs[:, :LANE] + _rotate_gptj(tail) * cs[:, LANE:]).astype(
        jnp.bfloat16
    )
    kv_out_ref[...] = jnp.concatenate([kv_normed[:, :-LANE], roped], axis=-1)


@functools.partial(
    jax.jit,
    static_argnames=("q_eps", "kv_eps", "name"),
)
def fused_wqa_wkv_norm_rope_quant(
    h: jax.Array,
    w_wqa: jax.Array,
    ws_wqa: jax.Array,
    q_w: jax.Array,
    kv_w: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    *,
    q_eps: float = 1e-6,
    kv_eps: float = 1e-6,
    name: str = "fused_wqa_wkv_norm_rope_quant",
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Single-kernel fused w_qkv_a + q_norm + kv_norm + RoPE + qr_fp8 quant (0.00e+00 diff)."""
    from vllm_torchtpu.kernels.deepseek_v4 import rope as rope_mod
    from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul

    num_tokens, in_dim = h.shape
    if num_tokens not in (8,) and num_tokens < 384:
        qr_kv = xla_quantized_matmul(h, w_wqa, ws_wqa)
        return rope_mod.qkv_norm_rope_quant(
            qr_kv, q_w, kv_w, positions, cos_sin_cache, q_eps=q_eps, kv_eps=kv_eps
        )
    _, total_dim = w_wqa.shape
    q_dim = q_w.shape[0]
    kv_dim = kv_w.shape[0]
    pad_t = ((num_tokens + 7) & ~7) - num_tokens
    if pad_t > 0:
        h = jnp.pad(h, ((0, pad_t), (0, 0)))
        positions = jnp.pad(positions, (0, pad_t))
    t_pad = h.shape[0]
    tile_n = SUBLANE * _largest_divisor(t_pad // SUBLANE, cap=256 // SUBLANE)
    cos_sin = gather_cos_sin(
        positions, cos_sin_cache, inverse=False, name=f"{name}_gather_cos_sin"
    )
    qr_fp8, qr_s, kv_out = pl.pallas_call(
        functools.partial(
            _wqa_wkv_norm_rope_quant_kernel,
            q_dim=q_dim,
            q_eps=q_eps,
            kv_eps=kv_eps,
        ),
        out_shape=(
            jax.ShapeDtypeStruct((t_pad, q_dim), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((t_pad, LANE), jnp.float32),
            jax.ShapeDtypeStruct((t_pad, kv_dim), jnp.bfloat16),
        ),
        grid=(t_pad // tile_n,),
        in_specs=[
            pl.BlockSpec((tile_n, in_dim), lambda i: (i, 0)),
            pl.BlockSpec((in_dim, total_dim), lambda i: (0, 0)),
            pl.BlockSpec((1, total_dim), lambda i: (0, 0)),
            pl.BlockSpec((1, q_dim), lambda i: (0, 0)),
            pl.BlockSpec((1, kv_dim), lambda i: (0, 0)),
            pl.BlockSpec((tile_n, 2 * LANE), lambda i: (i, 0)),
        ],
        out_specs=(
            pl.BlockSpec((tile_n, q_dim), lambda i: (i, 0)),
            pl.BlockSpec((tile_n, LANE), lambda i: (i, 0)),
            pl.BlockSpec((tile_n, kv_dim), lambda i: (i, 0)),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
        name=name,
    )(
        h,
        w_wqa,
        ws_wqa.reshape(1, total_dim),
        q_w.astype(jnp.bfloat16).reshape(1, q_dim),
        kv_w.astype(jnp.bfloat16).reshape(1, kv_dim),
        cos_sin,
    )
    if pad_t > 0:
        return qr_fp8[:num_tokens], qr_s[:num_tokens], kv_out[:num_tokens]
    return qr_fp8, qr_s, kv_out


@functools.partial(
    jax.jit,
    static_argnames=("q_eps", "kv_eps", "name"),
)
def fused_wqa_wkv_prequant_norm_rope_quant(
    h_fp8: jax.Array,
    h_sc: jax.Array,
    w_wqa: jax.Array,
    ws_wqa: jax.Array,
    q_w: jax.Array,
    kv_w: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    *,
    q_eps: float = 1e-6,
    kv_eps: float = 1e-6,
    name: str = "fused_wqa_wkv_prequant_norm_rope_quant",
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Single-kernel fused prequant w_qkv_a + q_norm + kv_norm + RoPE + qr_fp8 quant."""
    from vllm_torchtpu.kernels.deepseek_v4 import rope as rope_mod

    num_tokens, in_dim = h_fp8.shape
    if num_tokens not in (48,) and num_tokens < 384:
        qr_kv = (
            jax.lax.dot_general(
                h_fp8, w_wqa, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
            )
            * h_sc[:, :1]
            * ws_wqa[None, :]
        ).astype(jnp.bfloat16)
        return rope_mod.qkv_norm_rope_quant(
            qr_kv, q_w, kv_w, positions, cos_sin_cache, q_eps=q_eps, kv_eps=kv_eps
        )
    _, total_dim = w_wqa.shape
    q_dim = q_w.shape[0]
    kv_dim = kv_w.shape[0]
    tile_n = SUBLANE * _largest_divisor(num_tokens // SUBLANE, cap=256 // SUBLANE)
    cos_sin = gather_cos_sin(
        positions, cos_sin_cache, inverse=False, name=f"{name}_gather_cos_sin"
    )
    return pl.pallas_call(
        functools.partial(
            _wqa_wkv_prequant_norm_rope_quant_kernel,
            q_dim=q_dim,
            q_eps=q_eps,
            kv_eps=kv_eps,
        ),
        out_shape=(
            jax.ShapeDtypeStruct((num_tokens, q_dim), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((num_tokens, LANE), jnp.float32),
            jax.ShapeDtypeStruct((num_tokens, kv_dim), jnp.bfloat16),
        ),
        grid=(num_tokens // tile_n,),
        in_specs=[
            pl.BlockSpec((tile_n, in_dim), lambda i: (i, 0)),
            pl.BlockSpec((tile_n, LANE), lambda i: (i, 0)),
            pl.BlockSpec((in_dim, total_dim), lambda i: (0, 0)),
            pl.BlockSpec((1, total_dim), lambda i: (0, 0)),
            pl.BlockSpec((1, q_dim), lambda i: (0, 0)),
            pl.BlockSpec((1, kv_dim), lambda i: (0, 0)),
            pl.BlockSpec((tile_n, 2 * LANE), lambda i: (i, 0)),
        ],
        out_specs=(
            pl.BlockSpec((tile_n, q_dim), lambda i: (i, 0)),
            pl.BlockSpec((tile_n, LANE), lambda i: (i, 0)),
            pl.BlockSpec((tile_n, kv_dim), lambda i: (i, 0)),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
        name=name,
    )(
        h_fp8,
        h_sc,
        w_wqa,
        ws_wqa.reshape(1, total_dim),
        q_w.astype(jnp.bfloat16).reshape(1, q_dim),
        kv_w.astype(jnp.bfloat16).reshape(1, kv_dim),
        cos_sin,
    )

def _wqa_wkv_norm_rope_split_kernel(
    h_ref: jax.Ref,         # VMEM (tile_n, 5120) bf16
    w_wqa_ref: jax.Ref,     # VMEM (5120, 2048) fp8
    ws_wqa_ref: jax.Ref,    # VMEM (1, 2048) f32
    q_w_ref: jax.Ref,       # VMEM (1, 1536) bf16
    kv_w_ref: jax.Ref,      # VMEM (1, 512) bf16
    cs_ref: jax.Ref,        # VMEM (tile_n, 256) f32
    qr_out_ref: jax.Ref,    # VMEM (tile_n, 1536) bf16
    kv_out_ref: jax.Ref,    # VMEM (tile_n, 512) bf16
    *,
    q_dim: int,
    q_eps: float,
    kv_eps: float,
) -> None:
    from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor

    h_q, h_sc = quantize_tensor(h_ref[...], jnp.float8_e4m3fn)
    qr_kv = (
        jax.lax.dot_general(
            h_q,
            w_wqa_ref[...],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        * h_sc
        * ws_wqa_ref[...]
    ).astype(jnp.bfloat16)

    q_x = qr_kv[:, :q_dim].astype(jnp.float32)
    kv_x = qr_kv[:, q_dim:].astype(jnp.float32)

    q_rms = jax.lax.rsqrt(jnp.mean(q_x * q_x, axis=-1, keepdims=True) + q_eps)
    qr_out_ref[...] = (q_x * q_rms).astype(jnp.bfloat16) * q_w_ref[...]

    kv_rms = jax.lax.rsqrt(jnp.mean(kv_x * kv_x, axis=-1, keepdims=True) + kv_eps)
    kv_normed = (kv_x * kv_rms).astype(jnp.bfloat16) * kv_w_ref[...]
    tail = kv_normed[:, -LANE:].astype(jnp.float32)
    cs = cs_ref[...]
    roped = (tail * cs[:, :LANE] + _rotate_gptj(tail) * cs[:, LANE:]).astype(
        jnp.bfloat16
    )
    kv_out_ref[...] = jnp.concatenate([kv_normed[:, :-LANE], roped], axis=-1)


def _wqa_wkv_prequant_norm_rope_split_kernel(
    h_fp8_ref: jax.Ref,     # VMEM (tile_n, 5120) fp8
    h_sc_ref: jax.Ref,      # VMEM (tile_n, 128) f32
    w_wqa_ref: jax.Ref,     # VMEM (5120, 2048) fp8
    ws_wqa_ref: jax.Ref,    # VMEM (1, 2048) f32
    q_w_ref: jax.Ref,       # VMEM (1, 1536) bf16
    kv_w_ref: jax.Ref,      # VMEM (1, 512) bf16
    cs_ref: jax.Ref,        # VMEM (tile_n, 256) f32
    qr_out_ref: jax.Ref,    # VMEM (tile_n, 1536) bf16
    kv_out_ref: jax.Ref,    # VMEM (tile_n, 512) bf16
    *,
    q_dim: int,
    q_eps: float,
    kv_eps: float,
) -> None:
    qr_kv = (
        jax.lax.dot_general(
            h_fp8_ref[...],
            w_wqa_ref[...],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        * h_sc_ref[:, :1]
        * ws_wqa_ref[...]
    ).astype(jnp.bfloat16)

    q_x = qr_kv[:, :q_dim].astype(jnp.float32)
    kv_x = qr_kv[:, q_dim:].astype(jnp.float32)

    q_rms = jax.lax.rsqrt(jnp.mean(q_x * q_x, axis=-1, keepdims=True) + q_eps)
    qr_out_ref[...] = (q_x * q_rms).astype(jnp.bfloat16) * q_w_ref[...]

    kv_rms = jax.lax.rsqrt(jnp.mean(kv_x * kv_x, axis=-1, keepdims=True) + kv_eps)
    kv_normed = (kv_x * kv_rms).astype(jnp.bfloat16) * kv_w_ref[...]
    tail = kv_normed[:, -LANE:].astype(jnp.float32)
    cs = cs_ref[...]
    roped = (tail * cs[:, :LANE] + _rotate_gptj(tail) * cs[:, LANE:]).astype(
        jnp.bfloat16
    )
    kv_out_ref[...] = jnp.concatenate([kv_normed[:, :-LANE], roped], axis=-1)


@functools.partial(
    jax.jit,
    static_argnames=("q_eps", "kv_eps", "name"),
)
def fused_wqa_wkv_norm_rope_split(
    h: jax.Array,
    w_wqa: jax.Array,
    ws_wqa: jax.Array,
    q_w: jax.Array,
    kv_w: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    *,
    q_eps: float = 1e-6,
    kv_eps: float = 1e-6,
    name: str = "fused_wqa_wkv_norm_rope_split",
) -> tuple[jax.Array, jax.Array]:
    num_tokens, in_dim = h.shape
    _, total_dim = w_wqa.shape
    q_dim = q_w.shape[0]
    kv_dim = kv_w.shape[0]
    pad_t = ((num_tokens + 7) & ~7) - num_tokens
    if pad_t > 0:
        h = jnp.pad(h, ((0, pad_t), (0, 0)))
        positions = jnp.pad(positions, (0, pad_t))
    t_pad = h.shape[0]
    tile_n = SUBLANE * _largest_divisor(t_pad // SUBLANE, cap=256 // SUBLANE)
    cos_sin = gather_cos_sin(
        positions, cos_sin_cache, inverse=False, name=f"{name}_gather_cos_sin"
    )
    qr_out, kv_out = pl.pallas_call(
        functools.partial(
            _wqa_wkv_norm_rope_split_kernel,
            q_dim=q_dim,
            q_eps=q_eps,
            kv_eps=kv_eps,
        ),
        out_shape=(
            jax.ShapeDtypeStruct((t_pad, q_dim), jnp.bfloat16),
            jax.ShapeDtypeStruct((t_pad, kv_dim), jnp.bfloat16),
        ),
        grid=(t_pad // tile_n,),
        in_specs=[
            pl.BlockSpec((tile_n, in_dim), lambda i: (i, 0)),
            pl.BlockSpec((in_dim, total_dim), lambda i: (0, 0)),
            pl.BlockSpec((1, total_dim), lambda i: (0, 0)),
            pl.BlockSpec((1, q_dim), lambda i: (0, 0)),
            pl.BlockSpec((1, kv_dim), lambda i: (0, 0)),
            pl.BlockSpec((tile_n, 2 * LANE), lambda i: (i, 0)),
        ],
        out_specs=(
            pl.BlockSpec((tile_n, q_dim), lambda i: (i, 0)),
            pl.BlockSpec((tile_n, kv_dim), lambda i: (i, 0)),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
        name=name,
    )(
        h,
        w_wqa,
        ws_wqa.reshape(1, total_dim),
        q_w.astype(jnp.bfloat16).reshape(1, q_dim),
        kv_w.astype(jnp.bfloat16).reshape(1, kv_dim),
        cos_sin,
    )
    if pad_t > 0:
        return qr_out[:num_tokens], kv_out[:num_tokens]
    return qr_out, kv_out


@functools.partial(
    jax.jit,
    static_argnames=("q_eps", "kv_eps", "name"),
)
def fused_wqa_wkv_prequant_norm_rope_split(
    h_fp8: jax.Array,
    h_sc: jax.Array,
    w_wqa: jax.Array,
    ws_wqa: jax.Array,
    q_w: jax.Array,
    kv_w: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    *,
    q_eps: float = 1e-6,
    kv_eps: float = 1e-6,
    name: str = "fused_wqa_wkv_prequant_norm_rope_split",
) -> tuple[jax.Array, jax.Array]:
    num_tokens, in_dim = h_fp8.shape
    _, total_dim = w_wqa.shape
    q_dim = q_w.shape[0]
    kv_dim = kv_w.shape[0]
    tile_n = SUBLANE * _largest_divisor(num_tokens // SUBLANE, cap=256 // SUBLANE)
    cos_sin = gather_cos_sin(
        positions, cos_sin_cache, inverse=False, name=f"{name}_gather_cos_sin"
    )
    return pl.pallas_call(
        functools.partial(
            _wqa_wkv_prequant_norm_rope_split_kernel,
            q_dim=q_dim,
            q_eps=q_eps,
            kv_eps=kv_eps,
        ),
        out_shape=(
            jax.ShapeDtypeStruct((num_tokens, q_dim), jnp.bfloat16),
            jax.ShapeDtypeStruct((num_tokens, kv_dim), jnp.bfloat16),
        ),
        grid=(num_tokens // tile_n,),
        in_specs=[
            pl.BlockSpec((tile_n, in_dim), lambda i: (i, 0)),
            pl.BlockSpec((tile_n, LANE), lambda i: (i, 0)),
            pl.BlockSpec((in_dim, total_dim), lambda i: (0, 0)),
            pl.BlockSpec((1, total_dim), lambda i: (0, 0)),
            pl.BlockSpec((1, q_dim), lambda i: (0, 0)),
            pl.BlockSpec((1, kv_dim), lambda i: (0, 0)),
            pl.BlockSpec((tile_n, 2 * LANE), lambda i: (i, 0)),
        ],
        out_specs=(
            pl.BlockSpec((tile_n, q_dim), lambda i: (i, 0)),
            pl.BlockSpec((tile_n, kv_dim), lambda i: (i, 0)),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
        name=name,
    )(
        h_fp8,
        h_sc,
        w_wqa,
        ws_wqa.reshape(1, total_dim),
        q_w.astype(jnp.bfloat16).reshape(1, q_dim),
        kv_w.astype(jnp.bfloat16).reshape(1, kv_dim),
        cos_sin,
    )


def _wq_b_q_rope_wide3d_kernel(
    qr_fp8_ref: jax.Ref,
    w_ref: jax.Ref,
    w_s_ref: jax.Ref,
    cos_sin_ref: jax.Ref,
    lhs_s_ref: jax.Ref,
    out_ref: jax.Ref,
    *,
    tile_t: int,
    num_sub_t: int,
    heads_per_group: int,
    head_dim: int,
) -> None:
    rhs = w_ref[...]
    rhs_s = w_s_ref[...]
    cos_sin = cos_sin_ref[...]
    lhs_s_all = lhs_s_ref[...]
    sub_t = tile_t // num_sub_t
    lo = head_dim - LANE
    lane = jnp.arange(LANE)
    lane_perm = lane ^ 1
    sign = jnp.where(lane % 2 == 0, -1.0, 1.0).astype(jnp.float32)

    def _dot(s: int):
        rows = slice(s * sub_t, (s + 1) * sub_t)
        return jax.lax.dot_general(
            qr_fp8_ref[rows],
            rhs,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

    partial_curr = _dot(0)
    for s in range(num_sub_t):
        rows = slice(s * sub_t, (s + 1) * sub_t)
        partial_next = _dot(s + 1) if s + 1 < num_sub_t else None
        cs = cos_sin[rows]
        cos = cs[:, None, :LANE]
        sin_signed = (cs[:, None, LANE : 2 * LANE]) * sign[None, None, :]
        lhs_s = lhs_s_all[rows, :1]
        q_3d = ((partial_curr * lhs_s) * rhs_s).astype(jnp.bfloat16).reshape(
            sub_t, heads_per_group, head_dim
        )
        tail = q_3d[:, :, lo:].astype(jnp.float32)
        rot = _lane_gather(tail, lane_perm)
        roped = (tail * cos + rot * sin_signed).astype(jnp.bfloat16)
        out_ref[rows] = jnp.concatenate([q_3d[:, :, :lo], roped], axis=-1)
        partial_curr = partial_next
