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
"""DeepSeek-V4.1 Pallas fused reverse-RoPE and wo_a projection kernels."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.deepseek_v41.rope import (
    LANE,
    CosSinRef,
    cos_sin_lanes,
    largest_divisor,
    rotate_gptj,
)

FP8_E4M3_MAX = float(jnp.finfo(jnp.float8_e4m3fn).max)
HEADS_PER_GROUP = 8


def _default_vmem_limit_bytes() -> int:
    return int(pltpu.get_tpu_info().vmem_capacity_bytes) * 3 // 4


def _cos_sin_body(
    cos_sin_ref: jax.Ref,
    out_ref: jax.Ref,
    *,
    rotary_dim: int,
    inverse: bool,
    out_dtype: jnp.dtype,
) -> None:
    cos, sin = cos_sin_lanes(cos_sin_ref[...], rotary_dim=rotary_dim, inverse=inverse)
    out_ref[...] = jnp.concatenate([cos, sin], axis=-1).astype(out_dtype)


def _cos_sin_kernel(
    positions_ref: jax.Ref,
    cos_sin_cache_hbm_ref: jax.Ref,
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
    def _run(allocations: tuple[CosSinRef, pltpu.BufferedRef]) -> None:
        pipeline(
            (cos_sin_cache_hbm_ref, positions_ref),
            out_hbm_ref,
            allocations=allocations,
        )

    _run()


@functools.partial(
    jax.jit,
    static_argnames=("inverse", "out_dtype", "name", "interpret"),
)
def gather_cos_sin(
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    *,
    inverse: bool = False,
    out_dtype: jnp.dtype = jnp.float32,
    name: str = "gather_cos_sin",
    interpret: bool = False,
) -> jax.Array:
    """Gather ``cos_sin_cache[positions]`` and expand to ``[T, 2 * LANE]``."""
    assert positions.ndim == 1
    assert cos_sin_cache.ndim == 2

    num_tokens = positions.shape[0]
    rotary_dim = cos_sin_cache.shape[1]
    assert rotary_dim % 2 == 0 and rotary_dim <= LANE

    tile_n = largest_divisor(num_tokens, cap=128)

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
            vmem_limit_bytes=_default_vmem_limit_bytes(),
            disable_bounds_checks=True,
        ),
        name=name,
        interpret=interpret,
    )(positions, cos_sin_cache)


def _rope_heads(
    x: jax.Array,
    cos_sin: jax.Array,
    *,
    head_dim: int,
    heads_per_group: int,
) -> jax.Array:
    # Avoid 3D reshape after per-head RoPE: slicing 128-lane-aligned 2D blocks
    # makes jnp.concatenate a zero-cost lane-tile relabel instead of a relayout.
    assert head_dim > LANE, f"head_dim {head_dim} must be larger than {LANE}"
    cos = cos_sin[:, :LANE]
    sin = cos_sin[:, LANE:]
    lo = head_dim - LANE
    pieces = []
    for h in range(heads_per_group):
        base = h * head_dim
        pieces.append(x[:, base : base + lo])
        tail = x[:, base + lo : base + head_dim].astype(jnp.float32)
        roped = tail * cos + rotate_gptj(tail) * sin
        pieces.append(roped.astype(x.dtype))
    return jnp.concatenate(pieces, axis=-1)


def _wo_a_sub_tile(
    x: jax.Array,
    cos_sin: jax.Array,
    rhs: jax.Array,
    scale: jax.Array,
    *,
    head_dim: int,
    quantize_activations: bool,
    out_dtype: jnp.dtype,
) -> jax.Array:
    rows = x.shape[0]
    x = _rope_heads(
        x.reshape(rows, -1), cos_sin, head_dim=head_dim, heads_per_group=x.shape[1]
    )
    if quantize_activations:
        amax = jnp.max(jnp.abs(x), axis=1, keepdims=True)
        inv: jax.Array | None = (
            FP8_E4M3_MAX / jnp.maximum(amax, jnp.bfloat16(1e-30))
        ).astype(jnp.bfloat16)
        lhs = (x * inv).astype(jnp.float8_e4m3fn)
    else:
        inv = None
        lhs = x

    acc = jax.lax.dot_general(
        lhs,
        rhs,
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    if inv is not None:
        acc = acc * (1.0 / inv.astype(jnp.float32))
    return (acc * scale).astype(out_dtype)


def _wo_a_kernel(
    x_ref: jax.Ref,
    w_ref: jax.Ref,
    scale_ref: jax.Ref,
    cos_sin_ref: jax.Ref,
    out_ref: jax.Ref,
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
        "interpret",
    ),
)
def wo_a_projection(
    x: jax.Array,
    wo_a: jax.Array,
    wo_a_scale: jax.Array,
    cos_sin: jax.Array,
    *,
    tile_t: int | None = None,
    tile_r: int | None = None,
    sub_t: int | None = None,
    quantize_activations: bool = True,
    name: str = "wo_a_projection",
    interpret: bool = False,
) -> jax.Array:
    """Apply per-group RoPE and FP8 ``wo_a`` matmul, returning ``bf16[T, G * R]``.

    Expects ``x: bf16[T, G * 8, head_dim]``, ``wo_a: fp8[8 * head_dim, G * R]``,
    ``wo_a_scale: f32[G * R]``, and ``cos_sin: f32[T, 2 * LANE]``.
    """
    assert x.ndim == 3
    assert x.dtype == jnp.bfloat16
    assert wo_a.ndim == 2 and wo_a_scale.ndim == 1

    num_tokens, num_heads, head_dim = x.shape
    reduction, out_features = wo_a.shape
    assert out_features == wo_a_scale.shape[0]

    assert reduction % head_dim == 0
    heads_per_group = reduction // head_dim
    assert heads_per_group == HEADS_PER_GROUP
    assert num_heads % heads_per_group == 0
    num_groups = num_heads // heads_per_group
    assert out_features % num_groups == 0
    lora_rank = out_features // num_groups

    assert cos_sin.ndim == 2 and cos_sin.shape == (num_tokens, 2 * LANE)
    assert head_dim % LANE == 0

    if tile_r is None:
        tile_r = lora_rank
    if tile_t is None:
        tile_t = largest_divisor(num_tokens, cap=1024)
    if sub_t is None:
        sub_t = largest_divisor(tile_t, cap=128)
    assert lora_rank % tile_r == 0
    assert num_tokens % tile_t == 0
    assert tile_t % sub_t == 0
    num_t_tiles = num_tokens // tile_t
    num_r_tiles = lora_rank // tile_r

    return pl.pallas_call(
        functools.partial(
            _wo_a_kernel,
            tile_t=tile_t,
            num_sub_t=tile_t // sub_t,
            quantize_activations=quantize_activations,
            head_dim=head_dim,
        ),
        out_shape=jax.ShapeDtypeStruct((num_tokens, out_features), x.dtype),
        grid=(num_groups, num_t_tiles, num_r_tiles),
        in_specs=[
            pl.BlockSpec(
                (tile_t, heads_per_group, head_dim), lambda g, t, r: (t, g, 0)
            ),
            pl.BlockSpec((reduction, tile_r), lambda g, t, r: (0, g * num_r_tiles + r)),
            pl.BlockSpec((1, tile_r), lambda g, t, r: (0, g * num_r_tiles + r)),
            pl.BlockSpec((tile_t, 2 * LANE), lambda g, t, r: (t, 0)),
        ],
        out_specs=pl.BlockSpec(
            (tile_t, tile_r), lambda g, t, r: (t, g * num_r_tiles + r)
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=_default_vmem_limit_bytes(),
            disable_bounds_checks=True,
        ),
        name=name,
        interpret=interpret,
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
        "interpret",
    ),
)
def fused_reverse_rope_wo_a_projection(
    x: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    wo_a: jax.Array,
    wo_a_scale: jax.Array,
    *,
    head_dim: int,
    inverse: bool = True,
    tile_t: int | None = None,
    tile_r: int | None = None,
    sub_t: int | None = None,
    quantize_activations: bool = True,
    name: str = "wo_a_projection",
    interpret: bool = False,
) -> jax.Array:
    """Gather ``cos_sin`` by ``positions`` and run fused RoPE + ``wo_a`` matmul."""
    assert x.ndim == 3
    assert x.shape[2] == head_dim

    cos_sin = gather_cos_sin(
        positions,
        cos_sin_cache,
        inverse=inverse,
        name=f"{name}_gather_cos_sin",
        interpret=interpret,
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
        interpret=interpret,
    )


__all__ = [
    "FP8_E4M3_MAX",
    "HEADS_PER_GROUP",
    "LANE",
    "fused_reverse_rope_wo_a_projection",
    "gather_cos_sin",
    "wo_a_projection",
]
