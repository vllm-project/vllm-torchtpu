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

import dataclasses
import functools
from typing import cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.quantized_matmul.util import (
    DENSE_PROJ_MIN_TOKENS,
    DENSE_PROJ_SUB_M,
    DENSE_PROJ_SUB_N,
    DenseProjEpilogue,
    dense_proj_matmul,
    dense_proj_matmul_applies,
    quantize_tensor,
    xla_quantized_matmul,
)
from vllm_torchtpu.layers.core.quantization import dtype_max
from vllm_torchtpu.utils import largest_divisor

# Native lane count; the roped channels are the tail of the last lane block.
LANE = 128

DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024

# Tokens per RMSNorm + RoPE step in project_qnorm_rope's matmul epilogue, and
# per cos/sin lane expansion on its first grid step.
QNORM_EPILOGUE_ROWS = 32

# project_rope_quant's kernel: tokens per dot, and token pairs per matmul that
# swaps the RoPE lane pairs.
ROPE_QUANT_PROJ_ROWS = 256
ROPE_QUANT_PROJ_PAIRS = 8
ROPE_QUANT_PROJ_VMEM_LIMIT_BYTES = 20 * 1024 * 1024
# Most tokens per call: the activation and the cos/sin rows stay whole in VMEM.
ROPE_QUANT_PROJ_MAX_TOKENS = 1024


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class CosSinRef(pltpu.BufferedRef):
    """Per-token gather of ``cos_sin_cache`` rows into a ``(tile_n, rotary_dim)``"""

    tile_n: int = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create(
        cls, *, spec: pl.BlockSpec, dtype: jnp.dtype, buffer_count: int, tile_n: int
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

        dest_ref = cast(jax.Ref, self.window_ref.at[slot])  # (tile_n, rotary_dim)
        width = self.window_ref.shape[-1]

        @pl.loop(0, self.tile_n, unroll=True)
        def _(i):
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
    """``out[b..., i] = operand[b..., indices[i]]`` over the last axis.

    The leading axes come along as batch dimensions. ``indices`` is a constant
    1-D array.
    """
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


def _cos_sin_lanes(
    cos_sin: jax.Array, *, rotary_dim: int, inverse: bool
) -> tuple[jax.Array, jax.Array]:
    """The reference's ``np.split`` + ``np.repeat(..., 2)``, over a whole block.

    ``cos_sin`` is one packed ``[cos | sin]`` row per token. The reference splits
    it in half, repeats each half so both channels of a pair share a frequency,
    rotates ``x[..., -rotary_dim:]`` and concatenates the NoPE part back on.

    A kernel block is ``LANE`` lanes wide and the roped channels are its tail, so
    rather than slicing them out we widen cos/sin to the full block and hand the
    NoPE lanes the identity rotation -- multiplying by ``cos = 1``, ``sin = 0``
    is exactly the reference's pass-through concatenation::

        lane      0 ... 63 | 64  65  66  67 ...   (LANE 128, rotary_dim 64)
        channel   -- NoPE -|  0   1   2   3
        cos       1 ...  1 | c0  c0  c1  c1       <- np.repeat(cos, 2)
        sin       0 ...  0 | s0  s0  s1  s1
    """
    half = rotary_dim // 2
    channel = jnp.arange(LANE) - (LANE - rotary_dim)  # < 0 on the NoPE lanes
    is_rot = channel >= 0
    # ``np.repeat(..., 2)``: channels 2k and 2k+1 both read frequency k. Clamped
    # to keep the gather in bounds on the NoPE lanes, replaced just below.
    freq = jnp.maximum(channel, 0) // 2

    cos = _lane_gather(cos_sin, freq)  # np.split(...)[0], repeated
    sin = _lane_gather(cos_sin, half + freq)  # np.split(...)[1], repeated
    if inverse:
        sin = -sin
    return jnp.where(is_rot, cos, 1.0), jnp.where(is_rot, sin, 0.0)


def _rotate_gptj(x: jax.Array) -> jax.Array:
    """The reference's ``rotated``: ``[-x1, x0, -x3, x2, ...]`` along the lanes.

    ``np.stack([-x[..., 1::2], x[..., 0::2]], -1).reshape(...)`` says, lane by
    lane, "take the partner lane ``i ^ 1``, negated on the even ones" -- a
    constant-index lane shuffle and a sign, no reshape. Lane parity may be read
    for channel parity because ``LANE - rotary_dim`` is even.
    """
    return _swap_pairs(x) * _gptj_sign()


def _swap_pairs(x: jax.Array) -> jax.Array:
    """``_rotate_gptj`` without its sign: lane ``i`` reads lane ``i ^ 1``."""
    return _lane_gather(x, jnp.arange(LANE) ^ 1)


def _gptj_sign() -> jax.Array:
    """``_rotate_gptj``'s sign per lane: -1 on even lanes, 1 on odd ones."""
    return jnp.where(jnp.arange(LANE) % 2 == 0, -1.0, 1.0).astype(jnp.float32)


def _rope_body(
    x_ref: jax.Ref,
    cos_sin_ref: jax.Ref,
    *out_refs: jax.Ref,
    rotary_dim: int,
    inverse: bool,
    out_dtype: jnp.dtype,
    quant_dtype: jnp.dtype | None,
) -> None:
    """Process one ``(tile_n, [num_heads,] LANE)`` chunk.

    Without ``quant_dtype`` the rotation is written back into ``x_ref`` in place.
    With it, the rotated chunk is quantized along the lanes and written to the
    ``(q_ref, scale_ref)`` outputs instead.
    """
    x = x_ref[...].astype(jnp.float32)
    cos, sin = _cos_sin_lanes(cos_sin_ref[...], rotary_dim=rotary_dim, inverse=inverse)
    if x.ndim == 3:  # [tokens, heads, lanes]: cos/sin are per token
        cos = cos[:, None, :]
        sin = sin[:, None, :]

    rotated = _rotate_gptj(x)
    y = x * cos + rotated * sin

    if quant_dtype is None:
        x_ref[...] = y.astype(out_dtype)
        return

    q_ref, scale_ref = out_refs
    qmax = dtype_max(quant_dtype)
    # One scale per row, i.e. over the whole ``LANE``-wide channel axis.
    abs_max = jnp.max(jnp.abs(y), axis=-1, keepdims=True)
    scale = abs_max / qmax
    scale_inv = jnp.where(abs_max == 0.0, 0.0, 1.0 / scale)
    q_ref[...] = (y * scale_inv).astype(quant_dtype)
    scale_ref[...] = scale.reshape(scale_ref.shape)


def _kernel(
    # scalar prefetch
    positions_ref: jax.Ref,
    # HBM inputs
    x_hbm_ref: jax.Ref,
    cos_sin_cache_hbm_ref: jax.Ref,
    # HBM outputs: (out,) aliased onto x_hbm_ref, or (q, scales) when quantizing
    *out_hbm_refs: jax.Ref,
    num_tiles: int,
    tile_n: int,
    x_block_shape: tuple[int, ...],
    scale_block_shape: tuple[int, ...],
    lane_block: int,
    rotary_dim: int,
    inverse: bool,
    in_dtype: jnp.dtype,
    out_dtype: jnp.dtype,
    cos_sin_dtype: jnp.dtype,
    quant_dtype: jnp.dtype | None,
) -> None:
    def x_index_map(i):
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

    if quant_dtype is None:
        del x_hbm_ref  # aliased to the output; read/written through the latter.
        (out_hbm_ref,) = out_hbm_refs
        x_ref_args = (out_hbm_ref,)
        out_ref_args = ()
        out_specs = []
        # input_output since we only update the last rope_dim channels in place.
        x_alloc = pltpu.BufferedRef.input_output(x_spec, out_dtype, buffer_count=2)
        out_allocs = ()
    else:
        q_hbm_ref, scale_hbm_ref = out_hbm_refs
        x_ref_args = (x_hbm_ref,)
        out_ref_args = (q_hbm_ref, scale_hbm_ref)
        q_spec = pl.BlockSpec(
            block_shape=x_block_shape,
            memory_space=pltpu.VMEM,
            index_map=x_index_map,
        )
        scale_spec = pl.BlockSpec(
            block_shape=scale_block_shape,
            memory_space=pltpu.VMEM,
            # Same block as ``q``, minus the channel axis it reduces away.
            index_map=lambda i: x_index_map(i)[:-1],
        )
        out_specs = [q_spec, scale_spec]
        x_alloc = pltpu.BufferedRef.input(x_spec, in_dtype, buffer_count=2)
        out_allocs = (
            pltpu.BufferedRef.output(q_spec, quant_dtype, buffer_count=2),
            pltpu.BufferedRef.output(scale_spec, jnp.float32, buffer_count=2),
        )

    pipeline = pltpu.emit_pipeline(
        functools.partial(
            _rope_body,
            rotary_dim=rotary_dim,
            inverse=inverse,
            out_dtype=out_dtype,
            quant_dtype=quant_dtype,
        ),
        grid=(num_tiles,),
        in_specs=[x_spec, cos_sin_spec],
        out_specs=out_specs,
    )

    @pl.with_scoped(allocations=(x_alloc, cos_sin_alloc, *out_allocs))
    def _run(allocations):
        pipeline(
            *x_ref_args,
            (cos_sin_cache_hbm_ref, positions_ref),
            *out_ref_args,
            allocations=allocations,
        )

    _run()


def _qnorm_rope_body(
    x_ref: jax.Ref,
    cos_sin_ref: jax.Ref,
    out_ref: jax.Ref,
    *,
    rotary_dim: int,
    eps: float,
    inverse: bool,
    out_dtype: jnp.dtype,
) -> None:
    """Process one ``(tile_n, num_heads, head_dim)`` chunk.

    Per-head RMSNorm (no weight) over the whole ``head_dim``, then the same
    rotation ``_rope_body`` applies.
    """
    x = x_ref[...].astype(jnp.float32)
    cos, sin = _cos_sin_lanes(cos_sin_ref[...], rotary_dim=rotary_dim, inverse=inverse)
    assert x.ndim == 3
    y = _qnorm_rope_rows(x, cos[:, None, :], sin[:, None, :], eps=eps)
    out_ref[...] = y.astype(out_dtype)


def _qnorm_rope_rows(
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    *,
    eps: float,
    signed_sin: bool = False,
) -> jax.Array:
    """RMSNorm (no weight) of each ``head_dim`` row of f32 ``x``, then RoPE on
    its last lane block with ``cos``/``sin`` from ``_cos_sin_lanes``.

    With ``signed_sin``, ``sin`` already carries ``_gptj_sign``. Multiplying by
    +-1 is exact, so the result is the same bits for one multiply less.
    """
    rms = jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    y = x * rms

    # Only the last lane block holds roped channels; the rest of the row passes
    # through with nothing but the norm applied.
    tail = y[..., -LANE:]
    rotated = _swap_pairs(tail) if signed_sin else _rotate_gptj(tail)
    roped = tail * cos + rotated * sin
    return jnp.concatenate([y[..., :-LANE], roped], axis=-1)


def _qnorm_rope_kernel(
    # scalar prefetch
    positions_ref: jax.Ref,
    # HBM inputs
    x_hbm_ref: jax.Ref,
    cos_sin_cache_hbm_ref: jax.Ref,
    # HBM output, aliased onto x_hbm_ref
    out_hbm_ref: jax.Ref,
    *,
    num_tiles: int,
    tile_n: int,
    x_block_shape: tuple[int, ...],
    rotary_dim: int,
    eps: float,
    inverse: bool,
    in_dtype: jnp.dtype,
    cos_sin_dtype: jnp.dtype,
) -> None:
    x_spec = pl.BlockSpec(
        block_shape=x_block_shape,
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i, 0, 0),
    )
    cos_sin_spec = pl.BlockSpec(
        block_shape=(tile_n, rotary_dim),
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i, 0),
    )
    cos_sin_alloc = CosSinRef.create(
        spec=cos_sin_spec, dtype=cos_sin_dtype, buffer_count=2, tile_n=tile_n
    )

    pipeline = pltpu.emit_pipeline(
        functools.partial(
            _qnorm_rope_body,
            rotary_dim=rotary_dim,
            eps=eps,
            inverse=inverse,
            out_dtype=in_dtype,
        ),
        grid=(num_tiles,),
        in_specs=[x_spec, cos_sin_spec],
        out_specs=[x_spec],
    )
    allocations = (
        pltpu.BufferedRef.input(x_spec, in_dtype, buffer_count=2),
        cos_sin_alloc,
        pltpu.BufferedRef.output(x_spec, in_dtype, buffer_count=2),
    )

    @pl.with_scoped(allocations=allocations)
    def _run(allocations):
        pipeline(
            x_hbm_ref,
            (cos_sin_cache_hbm_ref, positions_ref),
            out_hbm_ref,
            allocations=allocations,
        )

    _run()


def _rope_call(
    x: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    *,
    inverse: bool,
    quant_dtype: jnp.dtype | None,
    name: str,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Builds the ``pallas_call``; see the public wrappers for the contract."""
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
    if quant_dtype is not None and head_dim != LANE:
        # Only the last lane block is pipelined in, but the scale is a reduction
        # over the whole row, so the two have to coincide (the indexer's head_dim
        # is exactly LANE).
        raise ValueError(f"quantization requires head_dim == {LANE}, got {head_dim}")

    tile_n = largest_divisor(num_tokens, cap=128)
    assert num_tokens % tile_n == 0

    x_block_shape = (tile_n, rows_per_token, LANE) if x.ndim == 3 else (tile_n, LANE)
    # One scale per row: x.shape minus the channel axis.
    scale_block_shape = x_block_shape[:-1]
    kernel = functools.partial(
        _kernel,
        num_tiles=num_tokens // tile_n,
        tile_n=tile_n,
        x_block_shape=x_block_shape,
        scale_block_shape=scale_block_shape,
        lane_block=head_dim // LANE - 1,  # last lane block has the roped channels
        rotary_dim=rotary_dim,
        inverse=inverse,
        in_dtype=x.dtype,
        out_dtype=x.dtype,
        cos_sin_dtype=cos_sin_cache.dtype,
        quant_dtype=quant_dtype,
    )

    if quant_dtype is None:
        out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
        out_specs = pl.BlockSpec(memory_space=pltpu.HBM)
        input_output_aliases = {1: 0}  # x (after the scalar prefetch) -> out
    else:
        out_shape = (
            jax.ShapeDtypeStruct(x.shape, quant_dtype),
            jax.ShapeDtypeStruct(x.shape[:-1], jnp.float32),
        )
        out_specs = (
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        )
        input_output_aliases = {}

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        in_specs=(
            pl.BlockSpec(memory_space=pltpu.HBM),  # x
            pl.BlockSpec(memory_space=pltpu.HBM),  # cos_sin_cache
        ),
        out_specs=out_specs,
    )

    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid_spec=grid_spec,
        input_output_aliases=input_output_aliases,
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
        name=name,
    )(positions, x, cos_sin_cache)


@functools.partial(
    jax.jit,
    static_argnames=("inverse", "name"),
    donate_argnames=("x",),
)
def rope(
    x: jax.Array,  # [num_tokens, head_dim] | [num_tokens, num_heads, head_dim]
    positions: jax.Array,  # [num_tokens] int
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    *,
    inverse: bool = False,
    name: str = "rope",
) -> jax.Array:
    """Applies DeepSeek-V4 RoPE to the trailing ``rotary_dim`` channels of ``x``.

    Args:
      x: ``[num_tokens, head_dim]`` or ``[num_tokens, num_heads, head_dim]``.
      positions: ``[num_tokens]``, the RoPE position of each token.
      cos_sin_cache: ``[max_position, rotary_dim]``, ``[cos | sin]`` packed side
        by side (``rotary_dim // 2`` columns each), as built by
        ``DeepseekV4ScalingRotaryEmbedding``.
      inverse: negate ``sin``, i.e. apply the transposed rotation.
      name: kernel name.

    Returns:
      ``x`` with the trailing ``rotary_dim`` channels rotated, same shape and
      dtype. ``x`` is donated -- the kernel updates the buffer in place.
    """
    return cast(
        jax.Array,
        _rope_call(
            x,
            positions,
            cos_sin_cache,
            inverse=inverse,
            quant_dtype=None,
            name=name,
        ),
    )


@functools.partial(
    jax.jit,
    static_argnames=("eps", "inverse", "name"),
    donate_argnames=("x",),
)
def qnorm_rope(
    x: jax.Array,  # [num_tokens, num_heads, head_dim] | [num_tokens, head_dim]
    positions: jax.Array,  # [num_tokens] int
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    *,
    eps: float = 1e-6,
    inverse: bool = False,
    name: str = "qnorm_rope",
) -> jax.Array:
    """Per-head RMSNorm (no weight) fused with DeepSeek-V4 RoPE."""
    assert x.ndim == 3
    assert positions.ndim == 1 and positions.shape[0] == x.shape[0]
    assert cos_sin_cache.ndim == 2

    num_tokens = x.shape[0]
    head_dim = x.shape[-1]
    rows_per_token = x.shape[1]
    rotary_dim = cos_sin_cache.shape[1]
    assert head_dim % LANE == 0
    assert rotary_dim % 2 == 0 and rotary_dim <= LANE
    tile_n = largest_divisor(num_tokens, cap=64)

    x_block_shape = (tile_n, rows_per_token, head_dim)
    kernel = functools.partial(
        _qnorm_rope_kernel,
        num_tiles=num_tokens // tile_n,
        tile_n=tile_n,
        x_block_shape=x_block_shape,
        rotary_dim=rotary_dim,
        eps=eps,
        inverse=inverse,
        in_dtype=x.dtype,
        cos_sin_dtype=cos_sin_cache.dtype,
    )

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        in_specs=(
            pl.BlockSpec(memory_space=pltpu.HBM),  # x
            pl.BlockSpec(memory_space=pltpu.HBM),  # cos_sin_cache
        ),
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
    )

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid_spec=grid_spec,
        input_output_aliases={1: 0},  # x (after the scalar prefetch) -> out
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
        name=name,
    )(positions, x, cos_sin_cache)


def _qnorm_rope_epilogue(
    positions: jax.Array, cos_sin_cache: jax.Array, *, eps: float
) -> DenseProjEpilogue:
    """qnorm_rope as dense_proj_matmul's epilogue, one head per output chunk."""
    num_tokens = positions.shape[0]
    rotary_dim = cos_sin_cache.shape[1]
    rows = largest_divisor(num_tokens, cap=QNORM_EPILOGUE_ROWS)

    def init(operand_refs, scratch_refs):
        (cos_sin_ref,) = operand_refs
        cos_ref, sin_ref = scratch_refs
        for r in range(0, num_tokens, rows):
            cos, sin = _cos_sin_lanes(
                cos_sin_ref[pl.ds(r, rows), :], rotary_dim=rotary_dim, inverse=False
            )
            cos_ref[pl.ds(r, rows), :] = cos
            sin_ref[pl.ds(r, rows), :] = sin * _gptj_sign()

    def block(y, row0, operand_refs, scratch_refs):
        del operand_refs
        cos_ref, sin_ref = scratch_refs
        cos = cos_ref[pl.ds(row0, rows), :]
        sin = sin_ref[pl.ds(row0, rows), :]
        return _qnorm_rope_rows(
            y.astype(jnp.float32), cos, sin, eps=eps, signed_sin=True
        ).astype(y.dtype)

    return DenseProjEpilogue(
        operands=(cos_sin_cache[positions],),
        scratch_shapes=(
            pltpu.VMEM((num_tokens, LANE), jnp.float32),
            pltpu.VMEM((num_tokens, LANE), jnp.float32),
        ),
        rows=rows,
        init=init,
        block=block,
    )


def project_qnorm_rope(
    x: jax.Array,  # [num_tokens, q_lora_rank] bf16
    w_q: jax.Array,  # [q_lora_rank, num_heads * head_dim] fp8
    w_scale: jax.Array,  # [num_heads * head_dim]
    positions: jax.Array,  # [num_tokens] int
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    *,
    head_dim: int,
    eps: float,
) -> jax.Array:
    """The per-channel fp8 wq_b projection followed by ``qnorm_rope``.

    Where dense_proj_matmul runs the projection, qnorm_rope's arithmetic runs in
    its epilogue on each head while the head is still in VMEM, so q is written
    to HBM once, already normed and roped. Otherwise this is the two ops back
    to back. Returns ``[num_tokens, num_heads, head_dim]``.
    """
    num_tokens = x.shape[0]
    q_shape = (num_tokens, w_q.shape[1] // head_dim, head_dim)
    if not dense_proj_matmul_applies(x, w_q, w_scale):
        q = xla_quantized_matmul(x, w_q, w_scale)
        return qnorm_rope(q.reshape(q_shape), positions, cos_sin_cache, eps=eps)
    x_q, x_scale = quantize_tensor(x, w_q.dtype)
    # One head per epilogue chunk, and all tokens in one row chunk: the cos/sin
    # scratch for more tokens does not fit next to the matmul's tiles.
    if head_dim != DENSE_PROJ_SUB_N or num_tokens > DENSE_PROJ_SUB_M:
        q = dense_proj_matmul(x_q, x_scale, w_q, w_scale, x.dtype)
        return qnorm_rope(q.reshape(q_shape), positions, cos_sin_cache, eps=eps)
    epilogue = _qnorm_rope_epilogue(positions, cos_sin_cache, eps=eps)
    q = dense_proj_matmul(x_q, x_scale, w_q, w_scale, x.dtype, epilogue=epilogue)
    return q.reshape(q_shape)


@functools.partial(jax.jit, static_argnames=("inverse", "quant_dtype", "name"))
def rope_quant(
    x: jax.Array,  # [num_tokens, head_dim] | [num_tokens, num_heads, head_dim]
    positions: jax.Array,  # [num_tokens] int
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    *,
    inverse: bool = False,
    quant_dtype: jnp.dtype = jnp.float8_e4m3fn,
    name: str = "rope_quant",
) -> tuple[jax.Array, jax.Array]:
    """RoPE fused with per-row dynamic quantization of the rotated values.

    Same rotation as ``rope``, but the result is quantized to ``quant_dtype``
    inside the kernel.
    """
    return cast(
        tuple[jax.Array, jax.Array],
        _rope_call(
            x,
            positions,
            cos_sin_cache,
            inverse=inverse,
            quant_dtype=quant_dtype,
            name=name,
        ),
    )


def _project_rope_quant_kernel(
    x_scale_ref,  # (1, m) f32, lane-dense
    w_scale_ref,  # (1, n) f32
    x_ref,  # (m, k) fp8, whole in VMEM
    cos_sin_ref,  # (m, rotary_dim) f32, this call's cos_sin_cache rows
    w_hbm_ref,  # (k, n) fp8 in HBM
    q_hbm_ref,  # (m, n // LANE, LANE) quant_dtype in HBM
    scale_ref,  # (heads, m) f32, streamed
    x_scale_lanes_ref,  # (m, LANE) f32 scratch
    cos_ref,  # (m, LANE) f32 scratch
    sin_ref,  # (m, LANE) f32 scratch, with the GPT-J sign folded in
    stage_ref,  # (rows // 2 * heads, LANE) int32 scratch
    w_buf,  # (2, k, tile_n) fp8 scratch
    q_buf,  # (2, LANE, heads, LANE) quant_dtype scratch
    sems,  # (2, 2) DMA semaphores: weight tiles, q blocks
    *,
    rotary_dim: int,
    quant_dtype: jnp.dtype,
):
    """One weight tile of q heads: the wq_b product, rounded to bf16 as
    xla_quantized_matmul rounds it, then rope_quant on every token.

    The MXU leaves tokens in rows, while rope_quant's per-row scale and the q
    layout want each token's heads in rows. So each block of ``rows`` tokens
    is staged as bf16 token pairs, turned over with in-register sublane
    transposes, and read back one token pair at a time.
    """
    m, tile_n = x_ref.shape[0], w_buf.shape[2]
    heads = tile_n // LANE
    rows = ROPE_QUANT_PROJ_ROWS
    pairs = ROPE_QUANT_PROJ_PAIRS
    step = pl.program_id(0)
    num_steps = pl.num_programs(0)
    blocks_per_step = m // LANE

    # Weight tiles are fetched by hand one step ahead, so XLA leaves the
    # weight in HBM rather than staging it through VMEM.
    def w_copy(tile):
        return pltpu.make_async_copy(
            w_hbm_ref.at[:, pl.ds(pl.multiple_of(tile * tile_n, tile_n), tile_n)],
            w_buf.at[tile % 2],
            sems.at[0, tile % 2],
        )

    # q leaves a block of LANE tokens at a time, so only the last block's copy
    # is exposed at the end.
    def q_copy(block):
        t0 = pl.multiple_of(block % blocks_per_step * LANE, LANE)
        head0 = pl.multiple_of(block // blocks_per_step * heads, heads)
        return pltpu.make_async_copy(
            q_buf.at[block % 2],
            q_hbm_ref.at[pl.ds(t0, LANE), pl.ds(head0, heads)],
            sems.at[1, block % 2],
        )

    @pl.when(step == 0)
    def _():
        w_copy(0).start()

    @pl.when(step + 1 < num_steps)
    def _():
        w_copy(step + 1).start()

    w_copy(step).wait()
    w_ref = w_buf.at[step % 2]
    qmax = dtype_max(quant_dtype)
    # Heads per sublane transpose: an int32 vreg holds 8 token pairs of a head.
    group = 8
    high_mask = jnp.int32(-(1 << 16))
    # _swap_pairs as a matmul: x @ swap reads lane i ^ 1, exact for bf16 x.
    swap = (
        jax.lax.broadcasted_iota(jnp.int32, (LANE, LANE), 0)
        == jax.lax.broadcasted_iota(jnp.int32, (LANE, LANE), 1) ^ 1
    ).astype(jnp.bfloat16)

    @pl.when(step == 0)
    def _():
        x_scale_lanes_ref[...] = jnp.broadcast_to(
            jnp.transpose(x_scale_ref[...]), x_scale_lanes_ref.shape
        )
        r = QNORM_EPILOGUE_ROWS
        for r0 in range(0, m, r):
            cos, sin = _cos_sin_lanes(
                cos_sin_ref[pl.ds(r0, r), :], rotary_dim=rotary_dim, inverse=False
            )
            cos_ref[pl.ds(r0, r), :] = cos
            sin_ref[pl.ds(r0, r), :] = sin * _gptj_sign()

    def project(r0):
        """Stage word row ``p * heads + h``: head h of tokens r0 + 2p (low
        half) and r0 + 2p + 1 (high half)."""
        x_scale = x_scale_lanes_ref[pl.ds(r0, rows), :]
        for c in range(heads // group):
            # K stays whole in one dot, as in XLA's fusion, so the f32 sums match.
            acc = jax.lax.dot_general(
                x_ref[pl.ds(r0, rows), :],
                w_ref[:, pl.ds(c * group * LANE, group * LANE)],
                dimension_numbers=(((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            words = []
            for j in range(group):
                n0 = pl.multiple_of(step * tile_n + (c * group + j) * LANE, LANE)
                # xla_quantized_matmul's epilogue order: x scale, then w scale.
                y = acc[:, j * LANE : (j + 1) * LANE] * x_scale
                y = y * w_scale_ref[:, pl.ds(n0, LANE)]
                words.append(pltpu.bitcast(y.astype(jnp.bfloat16), jnp.int32))
            for p0 in range(0, rows // 2, group):
                # [head, pair] -> [pair, head]
                block = jnp.stack([w[p0 : p0 + group] for w in words])
                block = jnp.swapaxes(block, 0, 1)
                for i in range(group):
                    row = (p0 + i) * heads + c * group
                    stage_ref[pl.ds(row, group), :] = block[i]

    def rope_quantize(t0):
        """rope_quant on tokens t0 ... t0 + LANE, staged from row ``s0``."""
        s0 = t0 % rows
        scale_block_ref = scale_ref.at[:, pl.ds(pl.multiple_of(t0, LANE), LANE)]
        block = step * blocks_per_step + t0 // LANE
        q_block_ref = q_buf.at[block % 2]

        @pl.when(block >= 2)
        def _():
            q_copy(block - 2).wait()

        for g in range(LANE // 2 // pairs):
            row = pl.multiple_of((s0 // 2 + g * pairs) * heads, heads)
            words = stage_ref[pl.ds(row, pairs * heads), :]
            # The low halves' tokens, then the high halves', each heads x LANE.
            x = jnp.concatenate(
                [
                    pltpu.bitcast(jax.lax.shift_left(words, 16), jnp.float32),
                    pltpu.bitcast(words & high_mask, jnp.float32),
                ]
            )
            x_swap = jax.lax.dot_general(
                x.astype(jnp.bfloat16),
                swap,
                dimension_numbers=(((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            for half in range(2):
                for i in range(pairs):
                    lane = 2 * (g * pairs + i) + half
                    t = t0 + lane
                    r = (half * pairs + i) * heads
                    # rope_quant's arithmetic, with the GPT-J sign folded into sin.
                    y = x[r : r + heads] * cos_ref[pl.ds(t, 1), :]
                    y = y + x_swap[r : r + heads] * sin_ref[pl.ds(t, 1), :]
                    abs_max = jnp.max(jnp.abs(y), axis=-1, keepdims=True)
                    scale = abs_max / qmax
                    scale_inv = jnp.where(abs_max == 0.0, 0.0, 1.0 / scale)
                    q_block_ref[lane] = (y * scale_inv).astype(quant_dtype)
                    scale_block_ref[:, pl.ds(lane, 1)] = scale
        q_copy(block).start()

    @pl.loop(0, m, step=rows)
    def _(r0):
        project(r0)
        pl.loop(r0, r0 + rows, step=LANE)(rope_quantize)

    @pl.when(step == num_steps - 1)
    def _():
        last = num_steps * blocks_per_step - 1
        q_copy(last - 1).wait()
        q_copy(last).wait()


def project_rope_quant(
    x: jax.Array,  # [num_tokens, q_lora_rank] bf16
    w_q: jax.Array,  # [q_lora_rank, num_heads * head_dim] fp8
    w_scale: jax.Array,  # [num_heads * head_dim]
    positions: jax.Array,  # [num_tokens] int
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    *,
    head_dim: int,
    quant_dtype: jnp.dtype = jnp.float8_e4m3fn,
) -> tuple[jax.Array, jax.Array]:
    """The per-channel fp8 wq_b projection followed by ``rope_quant``.

    For prefill-sized batches one kernel does both, so the bf16 q never
    reaches HBM. Otherwise this is the two ops back to back. Returns q
    ``[num_tokens, num_heads, head_dim]`` in ``quant_dtype`` and its per-row
    scales ``[num_tokens, num_heads]``, bit for bit as the two ops compute them.
    """
    num_tokens, k = x.shape
    num_heads = w_q.shape[1] // head_dim
    # A grid step's weight tile holds one sublane tile of q heads.
    heads_per_tile = 8 * (4 // jnp.dtype(quant_dtype).itemsize)
    if not (
        x.dtype == jnp.bfloat16
        and w_q.dtype == quant_dtype
        and w_scale.ndim == 1
        and head_dim == LANE
        and cos_sin_cache.shape[1] <= LANE
        and num_heads % heads_per_tile == 0
        and DENSE_PROJ_MIN_TOKENS <= num_tokens <= ROPE_QUANT_PROJ_MAX_TOKENS
        and num_tokens % ROPE_QUANT_PROJ_ROWS == 0
    ):
        q = xla_quantized_matmul(x, w_q, w_scale).reshape(num_tokens, num_heads, -1)
        return cast(
            tuple[jax.Array, jax.Array],
            rope_quant(q, positions, cos_sin_cache, quant_dtype=quant_dtype),
        )
    x_q, x_scale = quantize_tensor(x, w_q.dtype)
    m, n = num_tokens, w_q.shape[1]
    tile_n = heads_per_tile * LANE
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    q, scales = pl.pallas_call(
        functools.partial(
            _project_rope_quant_kernel,
            rotary_dim=cos_sin_cache.shape[1],
            quant_dtype=quant_dtype,
        ),
        grid=(n // tile_n,),
        in_specs=[vmem, vmem, vmem, vmem, hbm],
        out_specs=[hbm, pl.BlockSpec((heads_per_tile, m), lambda j: (j, 0))],
        out_shape=[
            jax.ShapeDtypeStruct((m, num_heads, LANE), quant_dtype),
            jax.ShapeDtypeStruct((num_heads, m), jnp.float32),
        ],
        scratch_shapes=[
            pltpu.VMEM((m, LANE), jnp.float32),
            pltpu.VMEM((m, LANE), jnp.float32),
            pltpu.VMEM((m, LANE), jnp.float32),
            pltpu.VMEM((ROPE_QUANT_PROJ_ROWS // 2 * heads_per_tile, LANE), jnp.int32),
            pltpu.VMEM((2, k, tile_n), w_q.dtype),
            pltpu.VMEM((2, LANE, heads_per_tile, LANE), quant_dtype),
            pltpu.SemaphoreType.DMA((2, 2)),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=ROPE_QUANT_PROJ_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
        name="project_rope_quant",
    )(
        x_scale.reshape(1, m),
        w_scale.astype(jnp.float32).reshape(1, n),
        x_q,
        cos_sin_cache[positions],
        w_q,
    )
    return q, scales.T
