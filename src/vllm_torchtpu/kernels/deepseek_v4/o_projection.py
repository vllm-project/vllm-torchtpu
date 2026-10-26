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
    groups_per_step: int = 1,
) -> None:
    cos_sin = cos_sin_ref[...]
    sub_t = tile_t // num_sub_t
    out_dtype = x_ref.dtype
    heads = x_ref.shape[1] // groups_per_step
    cols = w_ref.shape[1] // groups_per_step
    for j in range(groups_per_step):
        _group(
            x_ref.at[:, pl.ds(j * heads, heads)],
            w_ref.at[:, pl.ds(j * cols, cols)],
            scale_ref.at[:, pl.ds(j * cols, cols)],
            cos_sin,
            out_ref.at[:, pl.ds(j * cols, cols)],
            sub_t=sub_t,
            num_sub_t=num_sub_t,
            quantize_activations=quantize_activations,
            head_dim=head_dim,
            out_dtype=out_dtype,
        )


def _group(
    x_ref,
    w_ref,
    scale_ref,
    cos_sin,
    out_ref,
    *,
    sub_t,
    num_sub_t,
    quantize_activations,
    head_dim,
    out_dtype,
):
    """One group's heads against its wo_a column block."""
    rhs = w_ref[...]
    scale = scale_ref[...]
    for s in range(num_sub_t):
        # Subchunk the tile to allow more Instruction level parallelism
        # opportunities to be exploited by the LLO compiler.
        rows = slice(s * sub_t, (s + 1) * sub_t)
        x = _rope_heads(
            x_ref[rows].reshape(sub_t, -1),
            cos_sin[rows],
            head_dim=head_dim,
            heads_per_group=x_ref.shape[1],
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


@functools.partial(
    jax.jit,
    static_argnames=(
        "tile_t",
        "tile_r",
        "sub_t",
        "quantize_activations",
        "name",
        "groups_per_step",
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
    groups_per_step: int | None = None,
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
    if groups_per_step is None:
        # Two groups per grid step halve the steps (and their fixed cost) at
        # decode-sized token tiles; a larger tile's doubled blocks overrun VMEM.
        # rc7 tpu7x one core, quantized activations (j-1e69970b), per call:
        # 4 groups (TP4) T=16/128/256: 14.7/19.5/25.6 -> 13.4/18.0/24.9 us;
        # 2 groups (TP8) within 0.2 us; T=512 is slower (24.8 -> 26.0 us at
        # TP8, 38.7 -> 39.9 at TP4), so the cap is 256.
        groups_per_step = 2 if tile_t <= 256 and num_r_tiles == 1 else 1
    gps = groups_per_step if num_groups % groups_per_step == 0 else 1
    assert gps == 1 or num_r_tiles == 1

    return pl.pallas_call(
        functools.partial(
            _kernel,
            tile_t=tile_t,
            num_sub_t=tile_t // sub_t,
            quantize_activations=quantize_activations,
            head_dim=head_dim,
            groups_per_step=gps,
        ),
        out_shape=jax.ShapeDtypeStruct((num_tokens, out_features), x.dtype),
        grid=(num_groups // gps, num_t_tiles, num_r_tiles),
        in_specs=[
            # x: `gps` groups' heads, [t, g, 0] over [T, G * H, head_dim].
            pl.BlockSpec(
                (tile_t, gps * heads_per_group, head_dim), lambda g, t, r: (t, g, 0)
            ),
            # wo_a: the whole reduction, column block g * R + r * tile_r.
            pl.BlockSpec(
                (reduction, gps * tile_r), lambda g, t, r: (0, g * num_r_tiles + r)
            ),
            # wo_a_scale
            pl.BlockSpec((1, gps * tile_r), lambda g, t, r: (0, g * num_r_tiles + r)),
            # cos_sin: one token block
            pl.BlockSpec((tile_t, 2 * LANE), lambda g, t, r: (t, 0)),
        ],
        out_specs=pl.BlockSpec(
            (tile_t, gps * tile_r), lambda g, t, r: (t, g * num_r_tiles + r)
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


# The chip-pair swap kernel covers one token tile of the fused op: 2T <= 1024.
PAIR_MAX_TOKENS = 1024
# Barrier group of the chip-pair kernels; each call signals and waits once, so
# calls in program order keep the count balanced. Distinct from the fused MoE
# kernel's (0).
PAIR_COLLECTIVE_ID = 3


def _pair_sub_tile(x, cos_sin, rhs, scale, *, head_dim, heads, out_dtype):
    """`_kernel`'s per-sub-tile body: rope, per-row fp8 quant, dot, rescale."""
    x = _rope_heads(
        x.reshape(x.shape[0], -1), cos_sin, head_dim=head_dim, heads_per_group=heads
    )
    amax = jnp.max(jnp.abs(x), axis=1, keepdims=True)
    inv = (FP8_E4M3_MAX / jnp.maximum(amax, jnp.bfloat16(1e-30))).astype(jnp.bfloat16)
    lhs = (x * inv).astype(jnp.float8_e4m3fn)
    partial = jax.lax.dot_general(
        lhs,
        rhs,
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    partial = partial * (1.0 / inv.astype(jnp.float32))
    return (partial * scale).astype(out_dtype)


def _pair_rows(first_ref, second_ref, r0, r1, heads):
    """Rows [r0, r1) of the 2T-token tile whose rows [0, T) are `first_ref`'s
    and [T, 2T) `second_ref`'s, for the head slice `heads`."""
    num_tokens = first_ref.shape[0]
    pieces = []
    if r0 < num_tokens:
        pieces.append(first_ref[r0 : min(r1, num_tokens), heads])
    if r1 > num_tokens:
        pieces.append(
            second_ref[max(r0, num_tokens) - num_tokens : r1 - num_tokens, heads]
        )
    return pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces, axis=0)


def _pair_swap_kernel(
    peer_ref,  # SMEM int32 [mesh_ndim]: the peer core's mesh coordinates
    rank_ref,  # SMEM int32 [1]: my index in the pair
    o_ref,  # VMEM (T, gps * H, head_dim): my tokens, my heads of this step
    o_hbm,  # HBM (T, 2 * H_half, head_dim): my tokens, all heads (DMA source)
    w_ref,  # VMEM (D, gps * R) fp8
    scale_ref,  # VMEM (1, gps * R) f32
    cos_sin_ref,  # VMEM (2T, 2 * LANE) f32
    out_ref,  # VMEM (2T, gps * R)
    recv,  # VMEM (steps, T, gps * H, head_dim): the peer's tokens, my heads
    send_sem,
    recv_sem,  # DMA (steps,)
    *,
    sub_t: int,
    head_dim: int,
    groups_per_step: int,
    num_steps: int,
    mesh_ndim: int,
):
    """wo_a over [core 0 tokens; core 1 tokens] with my heads, the peer's rows
    arriving by remote DMA during the kernel.

    Step g needs the peer's tokens for my heads of step g, which the peer sends
    from its own `o`; I send it the matching slice of mine. Two chunks are in
    flight at a time, and each step computes the sub-tiles of my own tokens
    before waiting for the peer's chunk. Every sub-tile is `_kernel`'s on the
    same rows.
    """
    g = pl.program_id(0)
    peer = tuple(peer_ref[i] for i in range(mesh_ndim))
    me = rank_ref[0]
    num_tokens, width = o_ref.shape[0], o_ref.shape[1]
    half_heads = o_hbm.shape[1] // 2

    def send(c):
        src = o_hbm.at[:, pl.ds((1 - me) * half_heads + c * width, width), :]
        return pltpu.make_async_remote_copy(
            src_ref=src,
            dst_ref=recv.at[c],
            send_sem=send_sem,
            recv_sem=recv_sem.at[c],
            device_id=peer,
            device_id_type=pl.DeviceIdType.MESH,
        )

    @pl.when(g == 0)
    def _():
        bsem = pltpu.get_barrier_semaphore()
        pl.semaphore_signal(
            bsem, 1, device_id=peer, device_id_type=pl.DeviceIdType.MESH
        )
        pl.semaphore_wait(bsem, 1)
        for c in range(min(2, num_steps)):
            send(c).start()

    cos_sin = cos_sin_ref[...]
    heads = width // groups_per_step
    cols = w_ref.shape[1] // groups_per_step
    peer_rows = recv.at[g]

    def compute(first_ref, second_ref, mine0):
        mine = range(mine0, mine0 + num_tokens)
        subs = [(s * sub_t, (s + 1) * sub_t) for s in range(2 * num_tokens // sub_t)]
        only_mine = [r for r in subs if r[0] in mine and r[1] - 1 in mine]
        order = only_mine + [r for r in subs if r not in only_mine]
        for i, (r0, r1) in enumerate(order):
            if i == len(only_mine):
                pltpu.make_async_copy(peer_rows, peer_rows, recv_sem.at[g]).wait()
            for j in range(groups_per_step):
                x = _pair_rows(first_ref, second_ref, r0, r1, pl.ds(j * heads, heads))
                out_ref[r0:r1, pl.ds(j * cols, cols)] = _pair_sub_tile(
                    x,
                    cos_sin[r0:r1],
                    w_ref[:, pl.ds(j * cols, cols)],
                    scale_ref[:, pl.ds(j * cols, cols)],
                    head_dim=head_dim,
                    heads=heads,
                    out_dtype=o_ref.dtype,
                )

    @pl.when(me == 0)
    def _():
        compute(o_ref, peer_rows, 0)

    @pl.when(me == 1)
    def _():
        compute(peer_rows, o_ref, num_tokens)

    for c in range(2, num_steps):

        @pl.when(g == c - 2)
        def _(c=c):
            send(c).start()

    @pl.when(g == num_steps - 1)
    def _():
        # Byte-counted: every chunk I sent has left.
        pltpu.make_async_copy(recv, recv, send_sem).wait()


@functools.partial(jax.jit, static_argnames=("head_dim", "inverse", "name"))
def fused_reverse_rope_wo_a_projection_pair_swap(
    o: jax.Array,  # [T, H, head_dim] bf16: my tokens, all heads
    positions: jax.Array,  # [2T] int: core 0's tokens, then core 1's
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    wo_a: jax.Array,  # [D, G * R] fp8: my groups
    wo_a_scale: jax.Array,  # [G * R] float32
    peer: jax.Array,  # int32 [mesh_ndim]: the peer core's mesh coordinates
    rank: jax.Array,  # int32 [1]: my index in the pair
    *,
    head_dim: int,
    inverse: bool = True,
    name: str = "wo_a_projection_swap",
) -> jax.Array:
    """Inside a shard_map over a mesh holding both cores of a chip pair:
    ``fused_reverse_rope_wo_a_projection`` (activation-quantized) of
    [core 0 tokens; core 1 tokens] with my half of the heads, where the peer's
    tokens' rows are fetched from the peer's ``o`` by remote DMA inside the
    kernel, overlapped with the work on my own tokens. Each output row is the
    fused op's on the same row, for 2T <= PAIR_MAX_TOKENS.
    """
    num_tokens, num_heads, _ = o.shape
    half_heads = num_heads // 2
    total = 2 * num_tokens
    reduction, out_features = wo_a.shape
    assert reduction % head_dim == 0 and reduction // head_dim == SUBLANE
    assert total <= PAIR_MAX_TOKENS, "the pair kernel assumes a single token tile"
    num_groups = half_heads // SUBLANE
    lora_rank = out_features // num_groups
    sub_t = _largest_divisor(total, cap=128)
    gps = 2 if total <= 512 and num_groups % 2 == 0 else 1
    width = gps * SUBLANE
    num_steps = num_groups // gps
    blocks_per_half = half_heads // width
    cos_sin = gather_cos_sin(
        positions, cos_sin_cache, inverse=inverse, name=f"{name}_gather_cos_sin"
    )
    mesh_ndim = peer.shape[0]
    return pl.pallas_call(
        functools.partial(
            _pair_swap_kernel,
            sub_t=sub_t,
            head_dim=head_dim,
            groups_per_step=gps,
            num_steps=num_steps,
            mesh_ndim=mesh_ndim,
        ),
        out_shape=jax.ShapeDtypeStruct((total, out_features), o.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(num_steps,),
            in_specs=[
                pl.BlockSpec(
                    (num_tokens, width, head_dim),
                    lambda g, p, r: (0, r[0] * blocks_per_half + g, 0),
                ),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((reduction, gps * lora_rank), lambda g, p, r: (0, g)),
                pl.BlockSpec((1, gps * lora_rank), lambda g, p, r: (0, g)),
                pl.BlockSpec((total, 2 * LANE), lambda g, p, r: (0, 0)),
            ],
            out_specs=pl.BlockSpec((total, gps * lora_rank), lambda g, p, r: (0, g)),
            scratch_shapes=[
                pltpu.VMEM((num_steps, num_tokens, width, head_dim), o.dtype),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((num_steps,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            collective_id=PAIR_COLLECTIVE_ID,
            vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
        name=name,
    )(
        peer.astype(jnp.int32),
        rank.astype(jnp.int32),
        o,
        o,
        wo_a,
        wo_a_scale.reshape(1, out_features),
        cos_sin,
    )
