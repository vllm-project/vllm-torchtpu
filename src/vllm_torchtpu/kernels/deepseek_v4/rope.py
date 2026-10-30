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
import math
from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.quantized_matmul.util import (
    _SCALE_LANES,
    QUANTIZE_GROUP_ROWS,
    _quantize_row_group,
)
from vllm_torchtpu.layers.core.quantization import dtype_max
from vllm_torchtpu.utils import largest_divisor

# Native lane count; the roped channels are the tail of the last lane block.
LANE = 128

DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024

# Rows (tokens) per tile of a bf16 projection output read in its native tiling.
TOKEN_TILE = 8
# Heads per transposed store: a strided load covers 8, two fill a bf16 tile.
UNTILE_HEADS = 16
# Spare (TOKEN_TILE, LANE) tiles after each head's tiles in qnorm_rope's input
# buffer: with D // LANE == 4 the strided head loads otherwise land on one VMEM
# bank and split into four loads per vreg (deviceless tpu7x LLO).
QNORM_HEAD_PAD_TILES = 1


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
    return _lane_gather(x, _partner_lanes()) * _rotation_sign()


def _partner_lanes() -> jax.Array:
    return jnp.arange(LANE) ^ 1


def _rotation_sign() -> jax.Array:
    """`_rotate_gptj`'s sign: negate the partner on even lanes."""
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
    q, scale = _quantize_lanes(y, quant_dtype)
    q_ref[...] = q
    scale_ref[...] = scale.reshape(scale_ref.shape)


def _quantize_lanes(
    y: jax.Array, quant_dtype: jnp.dtype
) -> tuple[jax.Array, jax.Array]:
    """``y`` quantized with one scale per row of its last axis, and the scales."""
    qmax = dtype_max(quant_dtype)
    # One scale per row, i.e. over the whole ``LANE``-wide channel axis.
    abs_max = jnp.max(jnp.abs(y), axis=-1, keepdims=True)
    scale = abs_max / qmax
    scale_inv = jnp.where(abs_max == 0.0, 0.0, 1.0 / scale)
    return (y * scale_inv).astype(quant_dtype), scale


def _low_bf16(words: jax.Array) -> jax.Array:
    """The low bf16 halves of uint32 ``words``, exactly, as f32."""
    return pltpu.bitcast(words << 16, jnp.float32)


def _high_bf16(words: jax.Array) -> jax.Array:
    """The high bf16 halves of uint32 ``words``, exactly, as f32."""
    return pltpu.bitcast(words & jnp.uint32(0xFFFF0000), jnp.float32)


def _transpose_bytes(words: list[jax.Array]) -> list[jax.Array]:
    """Byte ``j`` of ``out[k]`` is byte ``k`` of ``words[j]``, for four uint32
    arrays: 2x2 transposes of their 16-bit halves, then of the bytes in those."""
    w0, w1, w2, w3 = words
    lo, hi = jnp.uint32(0x0000FFFF), jnp.uint32(0xFFFF0000)
    even, odd = jnp.uint32(0x00FF00FF), jnp.uint32(0xFF00FF00)
    halves = [
        ((w0 & lo) | (w2 << 16), (w1 & lo) | (w3 << 16)),
        ((w0 >> 16) | (w2 & hi), (w1 >> 16) | (w3 & hi)),
    ]
    out = []
    for a, b in halves:
        out.append((a & even) | ((b & even) << 8))
        out.append(((a >> 8) & even) | (b & odd))
    return out


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


def token_tiles(x: jax.Array, num_heads: int) -> jax.Array:
    """``[tokens, num_heads * head_dim]`` as its own ``(TOKEN_TILE, LANE)`` tiles.

    Returns ``(tokens // TOKEN_TILE, num_heads, head_dim // LANE, TOKEN_TILE,
    LANE)`` holding the bytes of ``x`` in its native tiling, so XLA passes a
    projection's output through as a bitcast instead of relaying it out to
    ``[tokens, num_heads, head_dim]``.
    """
    num_tokens, width = x.shape
    head_dim = width // num_heads
    return x.reshape(
        num_tokens // TOKEN_TILE, TOKEN_TILE, num_heads, head_dim // LANE, LANE
    ).transpose(0, 2, 3, 1, 4)


def can_untile_heads(num_tokens: int, num_heads: int, head_dim: int) -> bool:
    return (
        num_tokens % TOKEN_TILE == 0
        and num_heads % UNTILE_HEADS == 0
        and head_dim % LANE == 0
    )


def _untile_heads(words_ref: jax.Ref, out_ref: jax.Ref) -> None:
    """``out[t, h, :]`` = token ``t``, head ``h`` of a `token_tiles` block.

    ``words_ref`` is that bf16 block bitcast to uint32: ``(tile_n //
    TOKEN_TILE, heads, >= head_dim // LANE, TOKEN_TILE // 2, LANE)``, a word
    holding tokens ``2s`` (low half) and ``2s + 1``. One sublane-strided load
    gathers a token pair for 8 heads and its halves are the two tokens' rows.
    Pure data movement: bf16 -> f32 -> bf16 is exact.
    """
    num_groups, num_heads, lane_blocks, pairs, lane = words_ref.shape
    head_dim = out_ref.shape[-1]
    head_stride = lane_blocks * pairs

    def group(g, carry):
        rows = words_ref.at[g].reshape(num_heads * head_stride, lane)
        for s in range(pairs):
            t = g * 2 * pairs + 2 * s
            for h0 in range(0, num_heads, UNTILE_HEADS):
                for c in range(head_dim // lane):
                    first = (h0 * lane_blocks + c) * pairs + s
                    w = rows[pl.ds(first, UNTILE_HEADS, stride=head_stride), :]
                    lo = pltpu.bitcast(w << 16, jnp.float32)
                    hi = pltpu.bitcast(w & jnp.uint32(0xFFFF0000), jnp.float32)
                    heads = pl.ds(h0, UNTILE_HEADS)
                    cols = pl.ds(c * lane, lane)
                    out_ref[t, heads, cols] = lo.astype(out_ref.dtype)
                    out_ref[t + 1, heads, cols] = hi.astype(out_ref.dtype)
        return carry

    jax.lax.fori_loop(0, num_groups, group, 0, unroll=True)


def _rope_quant_tiles_kernel(
    # scalar prefetch
    positions_ref: jax.Ref,
    # HBM inputs
    x_hbm_ref: jax.Ref,  # `token_tiles` of [tokens, heads * LANE]
    cos_sin_cache_hbm_ref: jax.Ref,
    # HBM outputs
    q_hbm_ref: jax.Ref,
    scale_hbm_ref: jax.Ref,
    *,
    num_tiles: int,
    tile_n: int,
    num_heads: int,
    rotary_dim: int,
    inverse: bool,
    in_dtype: jnp.dtype,
    cos_sin_dtype: jnp.dtype,
    quant_dtype: jnp.dtype,
) -> None:
    x_spec = pl.BlockSpec(
        block_shape=(tile_n // TOKEN_TILE, num_heads, 1, TOKEN_TILE, LANE),
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i, 0, 0, 0, 0),
    )
    q_spec = pl.BlockSpec(
        block_shape=(tile_n, num_heads, LANE),
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i, 0, 0),
    )
    scale_spec = pl.BlockSpec(
        block_shape=(tile_n, num_heads),
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i, 0),
    )
    cos_sin_spec = pl.BlockSpec(
        block_shape=(tile_n, rotary_dim),
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i, 0),
    )
    cos_sin_alloc = CosSinRef.create(
        spec=cos_sin_spec, dtype=cos_sin_dtype, buffer_count=2, tile_n=tile_n
    )
    allocations = (
        pltpu.BufferedRef.input(x_spec, in_dtype, buffer_count=2),
        cos_sin_alloc,
        pltpu.BufferedRef.output(q_spec, quant_dtype, buffer_count=2),
        pltpu.BufferedRef.output(scale_spec, jnp.float32, buffer_count=2),
    )

    @pl.with_scoped(
        allocations=(allocations, pltpu.VMEM((tile_n, num_heads, LANE), in_dtype))
    )
    def _run(allocations):
        allocations, rows_ref = allocations

        def body(x_ref, cos_sin_ref, q_ref, scale_ref):
            _untile_heads(x_ref.bitcast(jnp.uint32), rows_ref)
            _rope_body(
                rows_ref,
                cos_sin_ref,
                q_ref,
                scale_ref,
                rotary_dim=rotary_dim,
                inverse=inverse,
                out_dtype=in_dtype,
                quant_dtype=quant_dtype,
            )

        pipeline = pltpu.emit_pipeline(
            body,
            grid=(num_tiles,),
            in_specs=[x_spec, cos_sin_spec],
            out_specs=[q_spec, scale_spec],
        )
        pipeline(
            x_hbm_ref,
            (cos_sin_cache_hbm_ref, positions_ref),
            q_hbm_ref,
            scale_hbm_ref,
            allocations=allocations,
        )

    _run()


def _qnorm_rope_body(
    x_ref: jax.Ref,
    cos_sin_ref: jax.Ref,
    out_ref: jax.Ref,
    kv_ref: jax.Ref | None = None,
    *,
    rotary_dim: int,
    eps: float,
    inverse: bool,
    out_dtype: jnp.dtype,
) -> None:
    """Process one ``(tile_n, num_heads, head_dim)`` chunk.

    Per-head RMSNorm (no weight) over the whole ``head_dim``, then the same
    rotation ``_rope_body`` applies. ``kv_ref``, the same tokens' roped lane
    block of the decoupled key, is rotated in place with ``_rope_body``'s ops.
    """
    x = x_ref[...].astype(jnp.float32)
    rms = jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    y = x * rms

    # Only the last lane block holds roped channels; the rest of the row passes
    # through with nothing but the norm applied.
    tail = y[..., -LANE:]
    cos, sin = _cos_sin_lanes(cos_sin_ref[...], rotary_dim=rotary_dim, inverse=inverse)
    if kv_ref is not None:
        kv = kv_ref[...].astype(jnp.float32)
        kv_ref[...] = (kv * cos + _rotate_gptj(kv) * sin).astype(kv_ref.dtype)
    assert tail.ndim == 3
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    roped = tail * cos + _rotate_gptj(tail) * sin

    out_ref[...] = jnp.concatenate([y[..., :-LANE], roped], axis=-1).astype(out_dtype)


def cos_sin_lanes(
    cos_sin: jax.Array, *, inverse: bool = False
) -> tuple[jax.Array, jax.Array]:
    """`_cos_sin_lanes` of ``[cos | sin]`` rows as wide as the rotary dim."""
    return _cos_sin_lanes(cos_sin, rotary_dim=cos_sin.shape[-1], inverse=inverse)


def rope_lanes(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """`_rope_body`'s f32 rotation of ``(tokens, LANE)`` rows."""
    return x * cos + _rotate_gptj(x) * sin


def cos_signed_sin_lanes(cos_sin: jax.Array) -> tuple[jax.Array, jax.Array]:
    """`cos_sin_lanes` with `_rotate_gptj`'s sign folded into ``sin``."""
    cos, sin = cos_sin_lanes(cos_sin)
    return cos, sin * _rotation_sign()


def rope_lanes_signed_partner(
    x: jax.Array, partner: jax.Array, cos: jax.Array, signed_sin: jax.Array
) -> jax.Array:
    """`rope_lanes_signed` given ``partner``, ``x``'s lanes ``i ^ 1``."""
    return x * cos + partner * signed_sin


def rope_lanes_signed(x: jax.Array, cos: jax.Array, signed_sin: jax.Array) -> jax.Array:
    """`rope_lanes` given `cos_signed_sin_lanes`, one multiply fewer per value.

    Bit-identical: ``(g * s) * sin`` and ``g * (s * sin)`` round the same
    product when ``s`` is +-1.
    """
    return x * cos + _lane_gather(x, _partner_lanes()) * signed_sin


def qnorm_rope_head(
    x: jax.Array, cos: jax.Array, sin: jax.Array, *, eps: float
) -> jax.Array:
    """`_qnorm_rope_body`'s f32 math on ``(tokens, head_dim)`` rows of one head."""
    rms = jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    y = x * rms
    tail = y[..., -LANE:]
    roped = tail * cos + _rotate_gptj(tail) * sin
    return jnp.concatenate([y[..., :-LANE], roped], axis=-1)


def qnorm_rope_head_signed(
    x: jax.Array, cos: jax.Array, signed_sin: jax.Array, *, eps: float
) -> jax.Array:
    """`qnorm_rope_head` given `cos_signed_sin_lanes`, bit-identically
    (see `rope_lanes_signed`)."""
    rms = jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    y = x * rms
    roped = rope_lanes_signed(y[..., -LANE:], cos, signed_sin)
    return jnp.concatenate([y[..., :-LANE], roped], axis=-1)


def _qnorm_rope_kernel(
    # scalar prefetch
    positions_ref: jax.Ref,
    # HBM inputs
    x_hbm_ref: jax.Ref,
    cos_sin_cache_hbm_ref: jax.Ref,
    # [kv input,] q output (aliased onto x_hbm_ref unless the input is tiled),
    # [kv output, aliased onto the kv input]
    *refs: jax.Ref,
    num_tiles: int,
    tile_n: int,
    x_block_shape: tuple[int, ...],
    out_block_shape: tuple[int, ...],
    kv_lane_block: int | None,
    rotary_dim: int,
    eps: float,
    inverse: bool,
    in_dtype: jnp.dtype,
    cos_sin_dtype: jnp.dtype,
) -> None:
    tiled = len(x_block_shape) != len(out_block_shape)
    x_spec = pl.BlockSpec(
        block_shape=x_block_shape,
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i,) + (0,) * (len(x_block_shape) - 1),
    )
    out_spec = pl.BlockSpec(
        block_shape=out_block_shape,
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
    in_specs = [x_spec, cos_sin_spec]
    in_allocs = [
        pltpu.BufferedRef.input(x_spec, in_dtype, buffer_count=2),
        cos_sin_alloc,
    ]
    in_refs = [x_hbm_ref, (cos_sin_cache_hbm_ref, positions_ref)]
    if kv_lane_block is None:
        (out_hbm_ref,) = refs
    else:
        # The kv input is aliased to its output; read/written through the latter.
        _, out_hbm_ref, kv_out_hbm_ref = refs
        kv_spec = pl.BlockSpec(
            block_shape=(tile_n, LANE),
            memory_space=pltpu.VMEM,
            index_map=lambda i: (i, kv_lane_block),
        )
        in_specs.append(kv_spec)
        in_allocs.append(
            pltpu.BufferedRef.input_output(
                kv_spec, kv_out_hbm_ref.dtype, buffer_count=2
            )
        )
        in_refs.append(kv_out_hbm_ref)

    def body(x_ref, cos_sin_ref, *refs):
        *kv_refs, out_ref = refs
        if tiled:  # transpose into the output block, then work there in place
            _untile_heads(x_ref.bitcast(jnp.uint32), out_ref)
            x_ref = out_ref
        _qnorm_rope_body(
            x_ref,
            cos_sin_ref,
            out_ref,
            *kv_refs,
            rotary_dim=rotary_dim,
            eps=eps,
            inverse=inverse,
            out_dtype=in_dtype,
        )

    pipeline = pltpu.emit_pipeline(
        body,
        grid=(num_tiles,),
        in_specs=in_specs,
        out_specs=[out_spec],
    )
    allocations = (
        *in_allocs,
        pltpu.BufferedRef.output(out_spec, in_dtype, buffer_count=2),
    )

    @pl.with_scoped(allocations=allocations)
    def _run(allocations):
        pipeline(*in_refs, out_hbm_ref, allocations=allocations)

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
        out_block_shape=x_block_shape,
        kv_lane_block=None,
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


@functools.partial(jax.jit, static_argnames=("num_heads", "eps", "inverse", "name"))
def qnorm_rope_flat(
    x: jax.Array,  # [num_tokens, num_heads * head_dim]
    positions: jax.Array,  # [num_tokens] int
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    *,
    num_heads: int,
    eps: float = 1e-6,
    inverse: bool = False,
    name: str = "qnorm_rope",
) -> jax.Array:
    """`qnorm_rope` of a flat q projection; returns ``[tokens, heads, head_dim]``.

    The projection writes each token's heads side by side along the lanes, and
    ``[tokens, heads, head_dim]`` puts the heads on sublanes, so XLA used to
    relayout the whole q (on the SparseCore at prefill, a TC copy at decode)
    before the kernel. Here the kernel reads the projection's tiles and does
    that transpose in VMEM, under its own HBM traffic.
    """
    num_tokens, width = x.shape
    head_dim = width // num_heads
    if x.dtype != jnp.bfloat16 or not can_untile_heads(num_tokens, num_heads, head_dim):
        q = x.reshape(num_tokens, num_heads, head_dim)
        return qnorm_rope(
            q, positions, cos_sin_cache, eps=eps, inverse=inverse, name=name
        )
    return cast(
        jax.Array,
        _qnorm_rope_flat_call(
            x,
            None,
            positions,
            cos_sin_cache,
            num_heads=num_heads,
            eps=eps,
            inverse=inverse,
            name=name,
        ),
    )


@functools.partial(jax.jit, static_argnames=("num_heads", "eps", "inverse", "name"))
def qnorm_kv_rope_flat(
    x: jax.Array,  # [num_tokens, num_heads * head_dim]
    kv: jax.Array,  # [num_tokens, kv_dim]
    positions: jax.Array,  # [num_tokens] int
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    *,
    num_heads: int,
    eps: float = 1e-6,
    inverse: bool = False,
    name: str = "qnorm_rope",
) -> tuple[jax.Array, jax.Array]:
    """`qnorm_rope_flat` of ``x`` and `rope` of ``kv`` in one kernel.

    Both rotate the same tokens by the same ``cos_sin_cache`` rows, so the
    key's roped lane block rides along in qnorm_rope's pipeline and reuses its
    cos/sin gather instead of launching a second kernel that gathers them
    again. Returns ``(q, kv)``, each bit-identical to the separate kernels.
    """
    num_tokens, width = x.shape
    head_dim = width // num_heads
    if x.dtype != jnp.bfloat16 or not can_untile_heads(num_tokens, num_heads, head_dim):
        q = x.reshape(num_tokens, num_heads, head_dim)
        return (
            qnorm_rope(
                q, positions, cos_sin_cache, eps=eps, inverse=inverse, name=name
            ),
            rope(kv, positions, cos_sin_cache, inverse=inverse),
        )
    return cast(
        tuple[jax.Array, jax.Array],
        _qnorm_rope_flat_call(
            x,
            kv,
            positions,
            cos_sin_cache,
            num_heads=num_heads,
            eps=eps,
            inverse=inverse,
            name=name,
        ),
    )


def _qnorm_rope_flat_call(
    x: jax.Array,
    kv: jax.Array | None,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    *,
    num_heads: int,
    eps: float,
    inverse: bool,
    name: str,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Builds the tiled-input ``pallas_call``; see the public wrappers."""
    num_tokens, width = x.shape
    head_dim = width // num_heads
    assert positions.ndim == 1 and positions.shape[0] == num_tokens
    rotary_dim = cos_sin_cache.shape[1]
    assert rotary_dim % 2 == 0 and rotary_dim <= LANE

    tile_n = TOKEN_TILE * largest_divisor(
        num_tokens // TOKEN_TILE, cap=64 // TOKEN_TILE
    )
    lane_blocks = head_dim // LANE
    x_block_shape = (
        tile_n // TOKEN_TILE,
        num_heads,
        lane_blocks + QNORM_HEAD_PAD_TILES,  # past the array: left unfilled
        TOKEN_TILE,
        LANE,
    )
    kv_lane_block = None
    if kv is not None:
        assert kv.ndim == 2 and kv.shape[0] == num_tokens, (kv.shape, x.shape)
        assert kv.shape[1] % LANE == 0
        kv_lane_block = kv.shape[1] // LANE - 1  # its roped channels
    kernel = functools.partial(
        _qnorm_rope_kernel,
        num_tiles=num_tokens // tile_n,
        tile_n=tile_n,
        x_block_shape=x_block_shape,
        out_block_shape=(tile_n, num_heads, head_dim),
        kv_lane_block=kv_lane_block,
        rotary_dim=rotary_dim,
        eps=eps,
        inverse=inverse,
        in_dtype=x.dtype,
        cos_sin_dtype=cos_sin_cache.dtype,
    )

    q_shape = jax.ShapeDtypeStruct((num_tokens, num_heads, head_dim), x.dtype)
    in_specs = [
        pl.BlockSpec(memory_space=pltpu.HBM),  # x tiles
        pl.BlockSpec(memory_space=pltpu.HBM),  # cos_sin_cache
    ]
    operands = [positions, token_tiles(x, num_heads), cos_sin_cache]
    if kv is None:
        out_shape = q_shape
        out_specs = pl.BlockSpec(memory_space=pltpu.HBM)
        input_output_aliases = {}
    else:
        in_specs.append(pl.BlockSpec(memory_space=pltpu.HBM))  # kv
        operands.append(kv)
        out_shape = (q_shape, jax.ShapeDtypeStruct(kv.shape, kv.dtype))
        out_specs = (
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        )
        input_output_aliases = {3: 1}  # kv (after the scalar prefetch) -> kv out

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        in_specs=in_specs,
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
    )(*operands)


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


@functools.partial(
    jax.jit, static_argnames=("num_heads", "inverse", "quant_dtype", "name")
)
def rope_quant_flat(
    x: jax.Array,  # [num_tokens, num_heads * LANE]
    positions: jax.Array,  # [num_tokens] int
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    *,
    num_heads: int,
    inverse: bool = False,
    quant_dtype: jnp.dtype = jnp.float8_e4m3fn,
    name: str = "rope_quant",
) -> tuple[jax.Array, jax.Array]:
    """`rope_quant` of a flat ``[tokens, heads * LANE]`` projection output.

    Returns ``q`` as ``[tokens, heads, LANE]`` and the scales as ``[tokens,
    heads]``, like `rope_quant` on the 3-D view, but reads the projection's
    own tiles (see `qnorm_rope_flat`), so XLA does not relayout ``x`` first.
    """
    num_tokens, width = x.shape
    if x.dtype != jnp.bfloat16 or not can_untile_heads(num_tokens, num_heads, LANE):
        q = x.reshape(num_tokens, num_heads, width // num_heads)
        return rope_quant(
            q,
            positions,
            cos_sin_cache,
            inverse=inverse,
            quant_dtype=quant_dtype,
            name=name,
        )
    assert width == num_heads * LANE, (x.shape, num_heads)
    assert positions.ndim == 1 and positions.shape[0] == num_tokens
    rotary_dim = cos_sin_cache.shape[1]
    assert rotary_dim % 2 == 0 and rotary_dim <= LANE

    tile_n = TOKEN_TILE * largest_divisor(
        num_tokens // TOKEN_TILE, cap=128 // TOKEN_TILE
    )
    kernel = functools.partial(
        _rope_quant_tiles_kernel,
        num_tiles=num_tokens // tile_n,
        tile_n=tile_n,
        num_heads=num_heads,
        rotary_dim=rotary_dim,
        inverse=inverse,
        in_dtype=x.dtype,
        cos_sin_dtype=cos_sin_cache.dtype,
        quant_dtype=quant_dtype,
    )
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        in_specs=(
            pl.BlockSpec(memory_space=pltpu.HBM),  # x tiles
            pl.BlockSpec(memory_space=pltpu.HBM),  # cos_sin_cache
        ),
        out_specs=(
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ),
    )
    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((num_tokens, num_heads, LANE), quant_dtype),
            jax.ShapeDtypeStruct((num_tokens, num_heads), jnp.float32),
        ),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
        name=name,
    )(positions, token_tiles(x, num_heads), cos_sin_cache)


# Heads per output tile of `quantized_matmul_heads`, an even multiple of the
# 8-row tiling of its bf16 [tokens, heads, head_dim] output.
HEADS_PER_TILE = 16
# Rows of x per grid step of `quantized_matmul_heads`.
HEADS_BLOCK_ROWS = 512
# Fewest rows routed to `quantized_matmul_heads`.
HEADS_MIN_ROWS = 128
# Rows of a head pair's product per pass of the norm/RoPE epilogue.
HEADS_EPILOGUE_ROWS = 128
# Table-row copies issued per iteration of the gather loop.
HEADS_GATHER_UNROLL = 8
# Heads per output tile of a quantizing `quantized_matmul_heads`: one 32-row
# tile of its 8-bit [tokens, heads, head_dim] output per token.
HEADS_QUANT_PER_TILE = 32
# Rows per pass of a quantizing `quantized_matmul_heads`' per-head epilogue, a
# multiple of the 32-row tile of its 8-bit words.
HEADS_QUANT_ROWS = 32


def _quantized_matmul_heads_kernel(
    indices_ref,
    x_ref,
    table_hbm,
    w_hbm,
    ws_hbm,
    *refs,
    num_row_blocks: int,
    num_col_blocks: int,
    head_fn: Callable[..., jax.Array],
    row_fn: Callable[[jax.Array], tuple[jax.Array, ...]],
    kv_fn: Callable[..., jax.Array] | None,
    quant_dtype: jnp.dtype | None,
):
    """One (weight tile j, row block i) step of `quantized_matmul_heads`.

    Steps run i fastest, so a weight tile is fetched once and reused for every
    row block while the next tile streams in. On its first visit (j == 0) row
    block i is quantized into the resident x_q, and its table rows, gathered
    during the step before, go through ``row_fn`` into resident row values.
    With ``kv_fn``, the same visit also rewrites row block i of kv's last lane
    block in place with ``kv_fn`` of those row values.

    With ``quant_dtype``, the dots first leave each head pair's product as
    packed bf16 words, and the same words with their lanes ``i ^ 1`` gathered,
    in VMEM. ``head_fn`` and the per-row quantize then run on each head's
    columns, and the heads' 8-bit words are byte-transposed into o_buf, which
    holds the block's rows ``t, t + pack, ...`` as its slab ``t``.
    """
    if kv_fn is not None:  # the kv input is aliased to kv_hbm, its output
        _, o_hbm, kv_hbm, *refs = refs
        *refs, kv_buf, kv_sems = refs
    elif quant_dtype is not None:
        o_hbm, s_hbm, *refs = refs
        *refs, s_buf, pair_buf = refs
    else:
        o_hbm, *refs = refs
    xq_ref, xs_ref, w_buf, ws_buf, o_buf, table_buf, *refs = refs
    *row_refs, w_sems, o_sems, table_sems = refs
    s = pl.program_id(0)
    j = s // num_row_blocks
    i = s % num_row_blocks
    w_slot = j % 2
    o_slot = s % 2
    block_rows = x_ref.shape[0]
    heads = o_hbm.shape[1] // num_col_blocks
    head_dim = o_hbm.shape[2]
    lane_blocks = head_dim // LANE
    block_cols = heads * head_dim
    rows = pl.ds(pl.multiple_of(i * block_rows, block_rows), block_rows)
    # Bytes of a quantized head row per uint32 word.
    pack = None if quant_dtype is None else 32 // dtypes.itemsize_bits(quant_dtype)

    def weight_copies(t, sl):
        cols = pl.ds(pl.multiple_of(t * block_cols, block_cols), block_cols)
        return (
            pltpu.make_async_copy(w_hbm.at[:, cols], w_buf.at[sl], w_sems.at[0, sl]),
            pltpu.make_async_copy(ws_hbm.at[:, cols], ws_buf.at[sl], w_sems.at[1, sl]),
        )

    def out_copies(sl):
        dst = o_hbm.at[rows, pl.ds(pl.multiple_of(j * heads, heads), heads)]
        if quant_dtype is not None:
            slab_rows = block_rows // pack
            slabs = o_hbm.reshape(o_hbm.shape[0] // pack, pack, *o_hbm.shape[1:])
            slabs = slabs.at[pl.ds(pl.multiple_of(i * slab_rows, slab_rows), slab_rows)]
            return [
                *(
                    pltpu.make_async_copy(
                        o_buf.at[sl, t],
                        slabs.at[:, t, pl.ds(pl.multiple_of(j * heads, heads), heads)],
                        o_sems.at[sl],
                    )
                    for t in range(pack)
                ),
                pltpu.make_async_copy(s_buf.at[sl], s_hbm.at[j, rows], o_sems.at[sl]),
            ]
        # o_buf holds each 128-lane block of the tile's heads on its own.
        return [
            pltpu.make_async_copy(
                o_buf.at[sl, c],
                dst.at[:, :, pl.ds(c * LANE, LANE)],
                o_sems.at[sl],
            )
            for c in range(lane_blocks)
        ]

    def gather_table_rows(block, sl):
        @pl.loop(0, block_rows, unroll=HEADS_GATHER_UNROLL)
        def _(r):
            row = indices_ref[block * block_rows + r]
            pltpu.make_async_copy(
                table_hbm.at[pl.ds(row, 1)],
                table_buf.at[sl, pl.ds(r, 1)],
                table_sems.at[sl],
            ).start()

    def kv_copy(block, sl, *, write):
        span = pl.ds(pl.multiple_of(block * block_rows, block_rows), block_rows)
        hbm = kv_hbm.at[span, pl.ds(kv_hbm.shape[1] - LANE, LANE)]
        if write:
            return pltpu.make_async_copy(kv_buf.at[sl], hbm, kv_sems.at[1, sl])
        return pltpu.make_async_copy(hbm, kv_buf.at[sl], kv_sems.at[0, sl])

    def wait_table_rows(sl):
        # One wait for the bytes of every row copy into the slot.
        whole = table_buf.at[sl]
        pltpu.make_async_copy(whole, whole, table_sems.at[sl]).wait()

    @pl.when(s == 0)
    def _():
        for copy in weight_copies(0, 0):
            copy.start()
        gather_table_rows(0, 0)
        if kv_fn is not None:
            kv_copy(0, 0, write=False).start()

    @pl.when(i == 0)
    def _():
        for copy in weight_copies(j, w_slot):
            copy.wait()

        @pl.when(j + 1 < num_col_blocks)
        def _():
            for copy in weight_copies(j + 1, 1 - w_slot):
                copy.start()

    @pl.when(j == 0)
    def _():
        row0 = pl.multiple_of(i * block_rows, block_rows)
        group_rows = min(QUANTIZE_GROUP_ROWS, block_rows)
        for start in range(0, block_rows, group_rows):
            _quantize_row_group(
                x_ref,
                pl.ds(start, group_rows),
                xq_ref,
                pl.ds(row0 + start, group_rows),
                xs_ref,
            )
        wait_table_rows(i % 2)

        @pl.when(i + 1 < num_row_blocks)
        def _():
            gather_table_rows(i + 1, (i + 1) % 2)
            if kv_fn is not None:
                # The slot's last user, block i - 1, has to be back in HBM.
                @pl.when(i >= 1)
                def _():
                    kv_copy(i - 1, (i + 1) % 2, write=True).wait()

                kv_copy(i + 1, (i + 1) % 2, write=False).start()

        values = row_fn(table_buf[i % 2])
        for ref, value in zip(row_refs, values):
            ref[rows, :] = value
        if kv_fn is not None:
            kv_slot = i % 2
            kv_copy(i, kv_slot, write=False).wait()
            kv = kv_buf[kv_slot].astype(jnp.float32)
            kv_buf[kv_slot] = kv_fn(kv, *values).astype(kv_buf.dtype)
            kv_copy(i, kv_slot, write=True).start()

    # The output buffer this step fills left for HBM two steps ago.
    @pl.when(s >= 2)
    def _():
        for copy in out_copies(o_slot):
            copy.wait()

    # A bf16 tile packs heads 2p (low half) and 2p + 1 of a token in word p;
    # strided stores need a 128-lane buffer, hence o_buf's lane blocks.
    words_per_row = heads // 2
    words_ref = o_buf.bitcast(jnp.uint32).at[o_slot]
    if quant_dtype is None:
        words_ref = words_ref.reshape(lane_blocks, block_rows * words_per_row, LANE)
    else:
        words_ref = words_ref.reshape(block_rows * heads // pack, head_dim)
    chunk_rows = math.gcd(HEADS_EPILOGUE_ROWS, block_rows)

    def pair_products(r_lo, r_hi):
        """Every head pair's product for rows [r_lo, r_hi) of the block, put
        through ``head_fn`` and stored as packed words."""
        span = pl.ds(pl.multiple_of(i * block_rows + r_lo, r_hi - r_lo), r_hi - r_lo)
        x_scale = xs_ref[span, 0:1]
        for p in range(words_per_row):
            cols = slice(2 * p * head_dim, (2 * p + 2) * head_dim)
            # xla_quantized_matmul's dot and epilogue, rounded to its output.
            acc = jnp.dot(
                xq_ref[span, :],
                w_buf[w_slot, :, cols],
                preferred_element_type=jnp.float32,
            )
            acc = acc * x_scale
            acc = acc * ws_buf[w_slot, :, cols]
            acc = acc.astype(jnp.bfloat16).astype(jnp.float32)
            # The epilogue goes over row chunks, so each chunk's norm, RoPE and
            # stores need only its own rows of the product.
            for c0 in range(0, r_hi - r_lo, chunk_rows):
                r0 = r_lo + c0
                heads_of_pair = [
                    acc[c0 : c0 + chunk_rows, h * head_dim : (h + 1) * head_dim]
                    for h in range(2)
                ]
                row0 = pl.multiple_of(i * block_rows + r0, chunk_rows)
                chunk = pl.ds(row0, chunk_rows)
                row_values = [ref[chunk, :] for ref in row_refs]
                pair = [head_fn(head, *row_values) for head in heads_of_pair]
                words = pltpu.pack_elementwise(pair, packed_dtype=jnp.bfloat16)
                dst = pl.ds(r0 * words_per_row + p, chunk_rows, stride=words_per_row)
                for c in range(lane_blocks):
                    words_ref.at[c][dst, :] = words[:, c * LANE : (c + 1) * LANE]

    def pair_words():
        """Every head pair's product, rounded to bf16 and packed as words
        (head 2p low), into pair_buf[0, p], and its lanes ``i ^ 1`` into
        pair_buf[1, p]: the gather moves both heads' partners at once."""
        x_scale = xs_ref[rows, 0:1]
        for p in range(heads // 2):
            cols = slice(2 * p * head_dim, (2 * p + 2) * head_dim)
            acc = jnp.dot(
                xq_ref[rows, :],
                w_buf[w_slot, :, cols],
                preferred_element_type=jnp.float32,
            )
            acc = acc * x_scale
            acc = acc * ws_buf[w_slot, :, cols]
            pair = [acc[:, h * head_dim : (h + 1) * head_dim] for h in range(2)]
            words = pltpu.pack_elementwise(pair, packed_dtype=jnp.bfloat16)
            pair_buf[0, p] = words
            pair_buf[1, p] = _lane_gather(words, _partner_lanes())

    def quantized_heads():
        """``head_fn`` and the per-row quantize on each head's columns; each
        group of ``pack`` heads' 8-bit words is transposed so that one word
        holds the group's bytes of one row, as o_buf's tiles hold them."""
        quant_rows = math.gcd(HEADS_QUANT_ROWS, block_rows)
        slab_rows = block_rows // pack
        # Group by group, so each group's quantize needs only its own dots.
        for g in range(heads // pack):
            for r0 in range(0, block_rows, quant_rows):
                chunk = pl.ds(r0, quant_rows)
                row = pl.ds(pl.multiple_of(i * block_rows + r0, quant_rows), quant_rows)
                row_values = [ref[row, :] for ref in row_refs]
                words = []
                for p in range(g * pack // 2, (g + 1) * pack // 2):
                    pair, partner = pair_buf[0, p, chunk], pair_buf[1, p, chunk]
                    for half in (_low_bf16, _high_bf16):
                        y = head_fn(half(pair), half(partner), *row_values)
                        q, scale = _quantize_lanes(y, quant_dtype)
                        head = pl.ds(g * pack + len(words), 1)
                        s_buf[o_slot, chunk, head] = scale
                        words.append(pltpu.bitcast(q, jnp.uint32))
                # Word s of words[k] holds rows pack * s + t, so out[t] holds
                # rows t, t + pack, ...: rows of o_buf's slab t.
                for t, w in enumerate(_transpose_bytes(words)):
                    first = (t * slab_rows + r0 // pack) * (heads // pack) + g
                    dst = pl.ds(first, quant_rows // pack, stride=heads // pack)
                    words_ref[dst, :] = w

    if quant_dtype is None:
        pair_products(0, block_rows)
    else:
        pair_words()
        quantized_heads()

    for copy in out_copies(o_slot):
        copy.start()

    @pl.when(s == num_row_blocks * num_col_blocks - 1)
    def _():
        steps = num_row_blocks * num_col_blocks
        for sl in (1 - o_slot, o_slot) if steps > 1 else (o_slot,):
            for copy in out_copies(sl):
                copy.wait()
        if kv_fn is not None:
            for block in range(max(0, num_row_blocks - 2), num_row_blocks):
                kv_copy(block, block % 2, write=True).wait()


def quantized_matmul_heads_supported(
    m: int, k: int, n: int, head_dim: int, quantize: bool = False
) -> bool:
    """Whether `quantized_matmul_heads` takes an (m, k) @ (k, n) problem."""
    block_rows = min(HEADS_BLOCK_ROWS, m)
    heads_per_tile = HEADS_QUANT_PER_TILE if quantize else HEADS_PER_TILE
    return (
        m >= HEADS_MIN_ROWS
        and m % block_rows == 0
        # The 8-bit x_q tile is 32 rows tall.
        and block_rows % 32 == 0
        and block_rows % min(QUANTIZE_GROUP_ROWS, block_rows) == 0
        and k % LANE == 0
        and head_dim % LANE == 0
        and n % (heads_per_tile * head_dim) == 0
        and (not quantize or head_dim == LANE)
    )


def quantized_matmul_heads(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    head_dim: int,
    head_fn: Callable[..., jax.Array],
    row_fn: Callable[[jax.Array], tuple[jax.Array, ...]],
    indices: jax.Array,
    table: jax.Array,
    kv: jax.Array | None = None,
    kv_fn: Callable[..., jax.Array] | None = None,
    quant_dtype: jnp.dtype | None = None,
    interpret: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """``head_fn`` on each head of ``xla_quantized_matmul(x, w_q, w_scale)``.

    The bf16 product of an 8-bit float ``w_q`` is split into heads of
    ``head_dim`` columns; each head, upcast to f32, goes through
    ``head_fn(head, *row_fn(table[indices]))`` for its rows and is rounded
    back to bf16. Returns ``[m, n // head_dim, head_dim]`` bf16, written
    straight from VMEM, so the product never round-trips through HBM before
    ``head_fn`` reads it.

    With ``kv`` (``[m, c * LANE]``), also returns ``kv`` with its last lane
    block replaced by ``kv_fn(block, *row_fn(table[indices]))`` in f32,
    rounded back; ``kv`` is donated.

    With ``quant_dtype`` (and ``head_dim == LANE``, no ``kv``), ``head_fn``
    is ``head_fn(head, partner, *row_fn(table[indices]))``, ``partner`` the
    head's lanes ``i ^ 1``, and its result is quantized like `rope_quant`'s.
    Returns ``(q, scales)``: ``q`` ``[m, n // head_dim, head_dim]`` in
    ``quant_dtype`` and ``scales`` ``[m, n // head_dim]`` f32.
    """
    m, k = x.shape
    n = w_q.shape[1]
    quantize = quant_dtype is not None
    if not quantized_matmul_heads_supported(m, k, n, head_dim, quantize):
        raise ValueError(f"unsupported shape {(m, k, n, head_dim)}")
    assert not (quantize and kv is not None)
    heads_per_tile = HEADS_QUANT_PER_TILE if quantize else HEADS_PER_TILE
    block_rows = min(HEADS_BLOCK_ROWS, m)
    block_cols = heads_per_tile * head_dim
    num_row_blocks = m // block_rows
    num_col_blocks = n // block_cols
    table_rows = jax.ShapeDtypeStruct((block_rows, table.shape[1]), table.dtype)
    row_shapes = jax.eval_shape(row_fn, table_rows)

    def x_block(s, indices_ref):
        return (jnp.minimum(s, num_row_blocks - 1), 0)

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    out_dtype = jnp.bfloat16
    # Pinned to HBM, so XLA neither prefetches them nor evicts the output.
    out_shape = pltpu.HBM((m, n // head_dim, head_dim), out_dtype)
    operands = [
        indices.astype(jnp.int32),
        x,
        pltpu.with_memory_space_constraint(table, pltpu.HBM),
        pltpu.with_memory_space_constraint(w_q, pltpu.HBM),
        pltpu.with_memory_space_constraint(w_scale.reshape(1, n), pltpu.HBM),
    ]
    in_specs = [pl.BlockSpec((block_rows, k), x_block), hbm, hbm, hbm]
    out_specs = hbm
    kv_scratch = []
    input_output_aliases = {}
    o_buf = pltpu.VMEM(
        (2, head_dim // LANE, block_rows, heads_per_tile, LANE), out_dtype
    )
    quant_scratch = []
    if quantize:
        num_heads = n // head_dim
        # Scales per weight tile, as [tile, m, heads of the tile].
        out_shape = (
            pltpu.HBM((m, num_heads, head_dim), quant_dtype),
            pltpu.HBM((num_col_blocks, m, heads_per_tile), jnp.float32),
        )
        out_specs = (hbm, hbm)
        # Slab t holds rows t, t + pack, ... of the block (see the kernel).
        pack = 32 // dtypes.itemsize_bits(quant_dtype)
        o_buf = pltpu.VMEM(
            (2, pack, block_rows // pack, heads_per_tile, head_dim), quant_dtype
        )
        quant_scratch = [
            pltpu.VMEM((2, block_rows, heads_per_tile), jnp.float32),
            pltpu.VMEM((2, heads_per_tile // 2, block_rows, LANE), jnp.uint32),
        ]
    if kv is not None:
        assert kv_fn is not None
        assert kv.shape[0] == m and kv.ndim == 2 and kv.shape[1] % LANE == 0
        input_output_aliases = {len(operands): 1}
        operands.append(pltpu.with_memory_space_constraint(kv, pltpu.HBM))
        in_specs.append(hbm)
        out_shape = (out_shape, pltpu.HBM(kv.shape, kv.dtype))
        out_specs = (hbm, hbm)
        kv_scratch = [
            pltpu.VMEM((2, block_rows, LANE), kv.dtype),
            pltpu.SemaphoreType.DMA((2, 2)),
        ]
    out = pl.pallas_call(
        functools.partial(
            _quantized_matmul_heads_kernel,
            num_row_blocks=num_row_blocks,
            num_col_blocks=num_col_blocks,
            head_fn=head_fn,
            row_fn=row_fn,
            kv_fn=None if kv is None else kv_fn,
            quant_dtype=quant_dtype,
        ),
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(num_row_blocks * num_col_blocks,),
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=[
                pltpu.VMEM((m, k), w_q.dtype),
                pltpu.VMEM((m, _SCALE_LANES), jnp.float32),
                pltpu.VMEM((2, k, block_cols), w_q.dtype),
                pltpu.VMEM((2, 1, block_cols), jnp.float32),
                o_buf,
                pltpu.VMEM((2, *table_rows.shape), table.dtype),
                *(pltpu.VMEM((m, r.shape[1]), r.dtype) for r in row_shapes),
                pltpu.SemaphoreType.DMA((2, 2)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                *kv_scratch,
                *quant_scratch,
            ],
        ),
        input_output_aliases=input_output_aliases,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=True,
            vmem_limit_bytes=_heads_vmem_bytes(
                m, k, block_rows, block_cols, x.dtype, w_q.dtype, row_shapes, quantize
            ),
        ),
        name="quantized_matmul_heads",
        interpret=interpret,
    )(*operands)
    if not quantize:
        return out
    q, scales = out
    return q, scales.transpose(1, 0, 2).reshape(m, n // head_dim)


def _heads_vmem_bytes(
    m, k, rows, cols, x_dtype, w_dtype, row_shapes, quantize=False
) -> int:
    """Double-buffered blocks, the resident x_q, scales and row values, the
    quantizing epilogue's packed products, their partner lanes and scale
    blocks, and headroom."""

    def nbytes(elems, dtype):
        return elems * dtypes.itemsize_bits(dtype) // 8

    blocks = (
        nbytes(rows * k, x_dtype)
        + nbytes(k * cols, w_dtype)
        + nbytes(rows * cols, jnp.bfloat16)
        + 8 * cols * 4
        + rows * LANE * 4
    )
    scratch = (
        nbytes(m * k, w_dtype)
        + m * _SCALE_LANES * 4
        + sum(m * max(r.shape[1], LANE) * 4 for r in row_shapes)
    )
    if quantize:
        scratch += 2 * nbytes(rows * cols, jnp.bfloat16) + 2 * rows * LANE * 4
    return 2 * blocks + scratch + (4 << 20)
