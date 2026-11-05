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
    _cos_sin_lanes,
    _rotate_gptj,
)
from vllm_torchtpu.layers.core.quantization import FP8_E4M3_MAX

DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024

SUBLANE = 8

# Token rows per grid step, and per sub-tile matmul within it: each weight
# tile pushed to the MXU is reused across a sub-tile's rows.
TILE_T_CAP = 1024
SUB_T_CAP = 256

# Tokens per gather_cos_sin group: a group whose positions are consecutive is
# fetched with one DMA instead of one per token.
GATHER_GROUP = 8
# Tokens per gather_cos_sin chunk: a chunk is laid out and written back while
# the row DMAs of later chunks land.
GATHER_CHUNK = 128

# The o-projection kernels' default name. wo_a_cos_sin's gather is shared only
# while its name matches fused_reverse_rope_wo_a_projection's.
WO_A_NAME = "wo_a_projection"


def _largest_divisor(x: int, cap: int) -> int:
    """Largest divisor of ``x`` that is <= ``cap``."""
    for candidate in range(min(x, cap), 0, -1):
        if x % candidate == 0:
            return candidate
    return 1


@functools.partial(jax.jit, static_argnames=("inverse", "num_positions"))
def cos_sin_table(
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    *,
    inverse: bool,
    num_positions: int,
) -> jax.Array:
    """``gather_cos_sin``'s ``[2 * num_positions, LANE]`` table, built at load.

    Position ``p``'s cache row widened to the full lane block
    (``_cos_sin_lanes``) is row ``2p`` (cos) and row ``2p + 1`` (sin). A
    ``LANE``-wide ``(8, 128)``-tiled array is row-major, so consecutive
    positions are one contiguous run that one DMA copies from any offset.
    It costs 1 KiB per position.
    """
    assert cos_sin_cache.ndim == 2
    assert num_positions <= cos_sin_cache.shape[0]
    rotary_dim = cos_sin_cache.shape[1]
    assert rotary_dim % 2 == 0 and rotary_dim <= LANE
    cos, sin = _cos_sin_lanes(
        cos_sin_cache[:num_positions], rotary_dim=rotary_dim, inverse=inverse
    )
    table = jnp.stack([cos, sin], axis=1).reshape(2 * num_positions, LANE)
    return table.astype(jnp.float32)


def _cos_sin_kernel(
    # scalar prefetch
    positions_ref: jax.Ref,
    # HBM input
    table_hbm_ref: jax.Ref,
    # HBM output
    out_hbm_ref: jax.Ref,
    # scratch
    rows_ref: jax.Ref,
    out_ref: jax.Ref,
    gather_sems: jax.Ref,
    writeback_sems: jax.Ref,
    *,
    group: int,
    chunk: int,
) -> None:
    """Copies each token's two table rows, then lays them out side by side.

    A group of ``group`` tokens at consecutive positions (a prefill request's)
    is one DMA; any other group is one DMA per token. All row DMAs are issued
    up front; each ``chunk`` of tokens is laid out and written back once its
    own DMAs land.
    """
    num_chunks = out_ref.shape[0] // chunk
    last_position = table_hbm_ref.shape[0] // 2 - 1

    def fetch(token, position, rows: int, sem) -> None:
        pltpu.make_async_copy(
            table_hbm_ref.at[pl.ds(2 * position, 2 * rows)],
            rows_ref.at[pl.ds(2 * token, 2 * rows)],
            sem,
        ).start()

    def fetch_chunk(c: int) -> None:
        sem = gather_sems.at[c]

        @pl.loop(c * chunk // group, (c + 1) * chunk // group)
        def _(g):
            token = g * group
            first = positions_ref[token]
            # Nonzero iff some position in the group is not ``first + j``.
            terms = [(positions_ref[token + j] - first) ^ j for j in range(1, group)]
            while len(terms) > 1:
                pairs = [a | b for a, b in zip(terms[::2], terms[1::2])]
                terms = pairs + terms[len(pairs) * 2 :]
            mismatch = terms[0] if terms else jnp.int32(0)
            # The group's rows are one run of the table.
            run = (mismatch == 0) & (first <= last_position + 1 - group)

            @pl.when(run)
            def _():
                fetch(token, first, group, sem)

            @pl.when(jnp.logical_not(run))
            def _():
                # A rolled loop keeps this rare path a branch: if-converted,
                # its per-token copies would cost issue slots in every group.
                @pl.loop(token, token + group)
                def _(t):
                    fetch(t, jnp.minimum(positions_ref[t], last_position), 1, sem)

    for c in range(num_chunks):
        fetch_chunk(c)

    def writeback(c: int):
        tokens = pl.ds(c * chunk, chunk)
        return pltpu.make_async_copy(
            out_ref.at[tokens], out_hbm_ref.at[tokens], writeback_sems.at[c]
        )

    for c in range(num_chunks):
        rows = rows_ref.at[pl.ds(2 * c * chunk, 2 * chunk)]
        # One wait covers the chunk's row DMAs: they share its semaphore.
        pltpu.make_async_copy(rows, rows, gather_sems.at[c]).wait()
        tokens = pl.ds(c * chunk, chunk)
        out_ref[tokens, :LANE] = rows[pl.ds(0, chunk, stride=2), :]
        out_ref[tokens, LANE:] = rows[pl.ds(1, chunk, stride=2), :]
        writeback(c).start()
    for c in range(num_chunks):
        writeback(c).wait()


@functools.partial(jax.jit, static_argnames=("name",))
def gather_cos_sin(
    positions: jax.Array,  # [T] int
    table: jax.Array,  # [2 * num_positions, LANE] float32
    *,
    name: str = "gather_cos_sin",
) -> jax.Array:
    """Gathers ``wo_a_projection``'s ``[T, 2 * LANE]`` cos/sin rows.

    ``table`` is ``cos_sin_table(...)``'s. A position at or past its
    ``num_positions`` (a draft token past max_model_len, whose output is
    discarded) reads the last position's rows rather than past the table.
    """
    assert positions.ndim == 1
    assert table.ndim == 2 and table.shape[1] == LANE

    num_tokens = positions.shape[0]
    chunk = _largest_divisor(num_tokens, cap=GATHER_CHUNK)
    num_chunks = num_tokens // chunk
    group = _largest_divisor(chunk, cap=GATHER_GROUP)
    dtype = table.dtype
    return pl.pallas_call(
        functools.partial(_cos_sin_kernel, group=group, chunk=chunk),
        out_shape=jax.ShapeDtypeStruct((num_tokens, 2 * LANE), dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),),
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=(
                pltpu.VMEM((2 * num_tokens, LANE), dtype),
                pltpu.VMEM((num_tokens, 2 * LANE), dtype),
                pltpu.SemaphoreType.DMA((num_chunks,)),
                pltpu.SemaphoreType.DMA((num_chunks,)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
        name=name,
    )(positions, pltpu.with_memory_space_constraint(table, pltpu.HBM))


def wo_a_cos_sin(
    positions: jax.Array,  # [T] int
    table: jax.Array,  # [2 * num_positions, LANE] float32
    *,
    name: str = WO_A_NAME,
) -> jax.Array:
    """The ``[T, 2 * LANE]`` cos/sin rows ``fused_reverse_rope_wo_a_projection``
    gathers. XLA computes identical calls once, so another kernel taking them
    from here shares the o-projection's gather."""
    return gather_cos_sin(positions, table, name=f"{name}_gather_cos_sin")


def _load_head_pair(
    x_ref: jax.Ref, *, t0: int, sub_t: int, pair: int
) -> tuple[jax.Array, jax.Array]:
    """Heads ``2 * pair`` and ``2 * pair + 1`` of one lane chunk, token-major f32.

    ``x_ref`` is a ``(tile_t, H, LANE)`` bf16 lane chunk. Its u32 view holds a
    token's heads ``(2p, 2p + 1)`` in one word row, so one sublane-strided load
    puts tokens on sublanes and the heads never pass through a sublane-to-lane
    relayout. The even head is the word's low half; each half widens to f32
    exactly by a shift or a mask.
    """
    pairs = x_ref.shape[1] // 2
    words = x_ref.bitcast(jnp.uint32).reshape(-1, LANE)
    w = words[pl.ds(t0 * pairs + pair, sub_t, stride=pairs), :]
    first = pltpu.bitcast(w << 16, jnp.float32)
    second = pltpu.bitcast(w & jnp.uint32(0xFFFF0000), jnp.float32)
    return first, second


def _rope_tail(tail: jax.Array, cos_sin: jax.Array) -> jax.Array:
    """Apply RoPE to one head's last ``LANE`` lanes, ``tail`` f32 ``(t, LANE)``."""
    cos = cos_sin[:, :LANE]
    sin = cos_sin[:, LANE:]
    return tail * cos + _rotate_gptj(tail) * sin


def _kernel(
    *refs: jax.Ref,
    tile_t: int,
    num_sub_t: int,
    quantize_activations: bool,
) -> None:
    # Inputs, all VMEM: one (tile_t, H, LANE) bf16 x block per LANE chunk of
    # head_dim, w (D, tile_r) fp8, scale (1, tile_r) f32 and cos_sin
    # (tile_t, 2 * LANE) f32. Output: out (tile_t, tile_r) in x's dtype.
    *x_refs, w_ref, scale_ref, cos_sin_ref, out_ref = refs
    rhs = w_ref[...]
    scale = scale_ref[...]
    sub_t = tile_t // num_sub_t
    num_heads = x_refs[0].shape[1]
    out_dtype = x_refs[0].dtype
    for s in range(num_sub_t):
        # Subchunk the tile to allow more Instruction level parallelism
        # opportunities to be exploited by the LLO compiler.
        t0 = s * sub_t
        rows = pl.ds(t0, sub_t)
        cos_sin = cos_sin_ref[rows, :]
        heads = [[] for _ in range(num_heads)]
        for c, x_ref in enumerate(x_refs):
            for pair in range(num_heads // 2):
                for h, chunk in zip(
                    (2 * pair, 2 * pair + 1),
                    _load_head_pair(x_ref, t0=t0, sub_t=sub_t, pair=pair),
                ):
                    if c == len(x_refs) - 1:
                        chunk = _rope_tail(chunk, cos_sin)
                    heads[h].append(chunk.astype(out_dtype))
        # Lane-aligned: the heads' chunks land in lanes without data movement.
        x = jnp.concatenate([chunk for chunks in heads for chunk in chunks], axis=1)
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
    name: str = WO_A_NAME,
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
        tile_t = _largest_divisor(num_tokens, cap=TILE_T_CAP)
    if sub_t is None:
        sub_t = _largest_divisor(tile_t, cap=SUB_T_CAP)
    assert lora_rank % tile_r == 0
    assert num_tokens % tile_t == 0
    assert tile_t % sub_t == 0
    num_t_tiles = num_tokens // tile_t
    num_r_tiles = lora_rank // tile_r
    num_lane_chunks = head_dim // LANE

    return pl.pallas_call(
        functools.partial(
            _kernel,
            tile_t=tile_t,
            num_sub_t=tile_t // sub_t,
            quantize_activations=quantize_activations,
        ),
        out_shape=jax.ShapeDtypeStruct((num_tokens, out_features), x.dtype),
        grid=(num_groups, num_t_tiles, num_r_tiles),
        in_specs=[
            # x: one group's heads, [t, g, c] over [T, G * H, head_dim], one
            # LANE-wide chunk c per operand: strided loads need a LANE-wide ref.
            *(
                pl.BlockSpec(
                    (tile_t, heads_per_group, LANE),
                    functools.partial(lambda g, t, r, c: (t, g, c), c=c),
                )
                for c in range(num_lane_chunks)
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
    )(*[x] * num_lane_chunks, wo_a, wo_a_scale.reshape(1, out_features), cos_sin)


@functools.partial(
    jax.jit,
    static_argnames=(
        "head_dim",
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
    table: jax.Array,  # [2 * num_positions, LANE] float32
    wo_a: jax.Array,  # [D, G * R] fp8
    wo_a_scale: jax.Array,  # [G * R] float32
    *,
    head_dim: int,
    tile_t: int | None = None,
    tile_r: int | None = None,
    sub_t: int | None = None,
    quantize_activations: bool = True,
    name: str = WO_A_NAME,
) -> jax.Array:
    """``The RoPE-fused ``wo_a_projection``.

    ``table`` is ``cos_sin_table(cos_sin_cache, ...)``, whose ``inverse``
    picks the rotation's direction.
    """
    assert x.ndim == 3
    assert x.shape[2] == head_dim

    cos_sin = wo_a_cos_sin(positions, table, name=name)
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
