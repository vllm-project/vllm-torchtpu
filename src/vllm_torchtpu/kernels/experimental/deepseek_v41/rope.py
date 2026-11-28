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
"""DeepSeek-V4.1 Pallas in-place trailing-channel GPT-J RoPE kernel."""

import dataclasses
import functools
from typing import cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

LANE = 128


def _default_vmem_limit_bytes() -> int:
    return int(pltpu.get_tpu_info().vmem_capacity_bytes) * 3 // 4


def largest_divisor(value: int, cap: int) -> int:
    """Return the largest divisor of ``value`` that does not exceed ``cap``."""
    candidate = min(value, cap)
    while value % candidate != 0:
        candidate -= 1
    return candidate


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class CosSinRef(pltpu.BufferedRef):
    """Per-token async HBM-to-VMEM gather of ``cos_sin_cache`` rows."""

    tile_n: int = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create(
        cls,
        *,
        spec: pl.BlockSpec,
        dtype: jnp.dtype,
        buffer_count: int,
        tile_n: int,
    ) -> "CosSinRef":
        standard_ref = pltpu.BufferedRef.create(
            spec=spec,
            dtype_or_type=dtype,
            buffer_type=pltpu.BufferType.INPUT,
            buffer_count=buffer_count,
            grid_rank=1,
            use_lookahead=False,
        )
        return cls(
            tile_n=tile_n,
            **{
                f.name: getattr(standard_ref, f.name)
                for f in dataclasses.fields(pltpu.BufferedRef)
            },
        )

    def copy_in(
        self,
        src_ref: tuple[jax.Ref, jax.Ref],
        grid_indices: tuple[int | jax.Array, ...],
    ) -> None:
        assert self.sem_recvs is not None
        assert self.window_ref is not None
        cos_sin_cache_ref, positions_ref = src_ref
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        pid = grid_indices[0]

        dest_ref = cast(jax.Ref, self.window_ref.at[slot])
        width = self.window_ref.shape[-1]

        @pl.loop(0, self.tile_n, unroll=True)
        def _(i: int) -> None:
            position = positions_ref[pid * self.tile_n + i]
            src = cos_sin_cache_ref.at[position, pl.ds(0, width)]
            pltpu.make_async_copy(src, dest_ref.at[i, :], sem).start()

    def wait_in(
        self,
        src_ref: tuple[jax.Ref, jax.Ref],
        grid_indices: tuple[int | jax.Array, ...],
    ) -> None:
        del src_ref, grid_indices
        assert self.sem_recvs is not None
        assert self.window_ref is not None
        slot = self.current_wait_in_slot
        vmem_ref = self.window_ref.at[slot]
        pltpu.make_async_copy(vmem_ref, vmem_ref, self.sem_recvs.at[slot]).wait()


def _lane_gather(operand: jax.Array, indices: jax.Array) -> jax.Array:
    rank = operand.ndim
    lanes = indices.shape[0]
    coords = jnp.broadcast_to(indices, (*operand.shape[:-1], lanes))[..., None]
    batching = tuple(range(rank - 1))
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(),
        collapsed_slice_dims=(rank - 1,),
        start_index_map=(rank - 1,),
        operand_batching_dims=batching,
        start_indices_batching_dims=batching,
    )
    return jax.lax.gather(
        operand,
        coords,
        dimension_numbers=dimension_numbers,
        slice_sizes=(1,) * rank,
        unique_indices=False,
        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )


def cos_sin_lanes(
    cos_sin: jax.Array, *, rotary_dim: int, inverse: bool
) -> tuple[jax.Array, jax.Array]:
    """Expand packed ``[cos | sin]`` rows to ``LANE`` with identity NoPE lanes."""
    half = rotary_dim // 2
    channel = jnp.arange(LANE) - (LANE - rotary_dim)
    is_rot = channel >= 0
    # Avoid OOB lane gather on NoPE lanes before masking with jnp.where.
    freq = jnp.maximum(channel, 0) // 2

    cos = _lane_gather(cos_sin, freq)
    sin = _lane_gather(cos_sin, half + freq)
    if inverse:
        sin = -sin
    return jnp.where(is_rot, cos, 1.0), jnp.where(is_rot, sin, 0.0)


def rotate_gptj(x: jax.Array) -> jax.Array:
    """Compute interleaved GPT-J partner rotation ``[-x1, x0, ...]`` over ``LANE``."""
    lane = jnp.arange(LANE)
    sign = jnp.where(lane % 2 == 0, -1.0, 1.0).astype(jnp.float32)
    return _lane_gather(x, lane ^ 1) * sign


def _rope_body(
    x_ref: jax.Ref,
    cos_sin_ref: jax.Ref,
    *,
    rotary_dim: int,
    inverse: bool,
    out_dtype: jnp.dtype,
) -> None:
    x = x_ref[...].astype(jnp.float32)
    cos, sin = cos_sin_lanes(cos_sin_ref[...], rotary_dim=rotary_dim, inverse=inverse)
    if x.ndim == 3:
        cos = cos[:, None, :]
        sin = sin[:, None, :]
    x_ref[...] = (x * cos + rotate_gptj(x) * sin).astype(out_dtype)


def _rope_kernel(
    positions_ref: jax.Ref,
    x_hbm_ref: jax.Ref,
    cos_sin_cache_hbm_ref: jax.Ref,
    out_hbm_ref: jax.Ref,
    *,
    num_tiles: int,
    tile_n: int,
    x_block_shape: tuple[int, ...],
    lane_block: int,
    rotary_dim: int,
    inverse: bool,
    out_dtype: jnp.dtype,
    cos_sin_dtype: jnp.dtype,
) -> None:
    # Avoid reading x_hbm_ref separately: it is aliased in place to out_hbm_ref.
    del x_hbm_ref

    def x_index_map(i: int) -> tuple[int, ...]:
        if len(x_block_shape) == 3:
            return (i, 0, lane_block)
        return (i, lane_block)

    x_spec = pl.BlockSpec(
        block_shape=x_block_shape, memory_space=pltpu.VMEM, index_map=x_index_map
    )
    cos_sin_spec = pl.BlockSpec(
        block_shape=(tile_n, rotary_dim),
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i, 0),
    )
    cos_sin_alloc = CosSinRef.create(
        spec=cos_sin_spec, dtype=cos_sin_dtype, buffer_count=2, tile_n=tile_n
    )
    x_alloc = pltpu.BufferedRef.input_output(x_spec, out_dtype, buffer_count=2)

    pipeline = pltpu.emit_pipeline(
        functools.partial(
            _rope_body,
            rotary_dim=rotary_dim,
            inverse=inverse,
            out_dtype=out_dtype,
        ),
        grid=(num_tiles,),
        in_specs=[x_spec, cos_sin_spec],
        out_specs=[],
    )

    @pl.with_scoped(allocations=(x_alloc, cos_sin_alloc))
    def _run(allocations: tuple[pltpu.BufferedRef, CosSinRef]) -> None:
        pipeline(
            out_hbm_ref,
            (cos_sin_cache_hbm_ref, positions_ref),
            allocations=allocations,
        )

    _run()


@functools.partial(
    jax.jit,
    static_argnames=("inverse", "name", "interpret"),
    donate_argnames=("x",),
)
def rope(
    x: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    *,
    inverse: bool = False,
    name: str = "rope",
    interpret: bool = False,
) -> jax.Array:
    """Apply GPT-J RoPE in place to the trailing ``rotary_dim`` channels of ``x``.

    Expects ``x: (T, D)`` or ``(T, H, D)`` with ``D % 128 == 0``,
    ``positions: int[T]``, and ``cos_sin_cache: f32[max_pos, rotary_dim]``.
    """
    if x.ndim not in (2, 3):
        raise ValueError(f"x must be rank 2 or 3, got {x.shape}")
    assert positions.ndim == 1 and positions.shape[0] == x.shape[0]
    assert cos_sin_cache.ndim == 2

    num_tokens = x.shape[0]
    head_dim = x.shape[-1]
    rows_per_token = x.shape[1] if x.ndim == 3 else 1
    rotary_dim = cos_sin_cache.shape[1]

    assert head_dim % LANE == 0
    assert rotary_dim % 2 == 0 and rotary_dim <= LANE

    tile_n = largest_divisor(num_tokens, cap=128)
    assert num_tokens % tile_n == 0

    x_block_shape = (tile_n, rows_per_token, LANE) if x.ndim == 3 else (tile_n, LANE)
    kernel = functools.partial(
        _rope_kernel,
        num_tiles=num_tokens // tile_n,
        tile_n=tile_n,
        x_block_shape=x_block_shape,
        lane_block=head_dim // LANE - 1,
        rotary_dim=rotary_dim,
        inverse=inverse,
        out_dtype=x.dtype,
        cos_sin_dtype=cos_sin_cache.dtype,
    )

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        in_specs=(
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ),
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
    )

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid_spec=grid_spec,
        input_output_aliases={1: 0},
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=_default_vmem_limit_bytes(),
            disable_bounds_checks=True,
        ),
        name=name,
        interpret=interpret,
    )(positions, x, cos_sin_cache)


__all__ = [
    "CosSinRef",
    "LANE",
    "cos_sin_lanes",
    "largest_divisor",
    "rope",
    "rotate_gptj",
]
