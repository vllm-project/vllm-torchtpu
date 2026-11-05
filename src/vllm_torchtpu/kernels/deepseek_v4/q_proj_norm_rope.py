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
"""The query's fp8 projection with its per-head RMSNorm and RoPE fused in.

Computes what ``fused_quantize_matmul`` followed by ``rope.qnorm_rope``
compute, but the un-normalized bf16 query never goes to HBM and back. The
matmul is ``fused_quantize_matmul``'s: the first grid step quantizes the
resident activation and every step multiplies it by one output tile of the
weight, here a whole number of heads. Each head's columns are rounded to bf16
as the plain matmul stores them, then normed and rotated by ``qnorm_rope``'s
own ops. Each token's cos/sin lanes come in already gathered: the o-projection's
inverse-rotation rows, which the first step turns into the forward rotation
once for all steps.

On a layer with an indexer, the kernel also computes the indexer's query
from the same resident quantized activation, with ``indexer_q_proj``'s ops.
Its work comes in units of one fp8 tile of heads (FP8_SUBLANES) for one tile
of tokens, one unit per grid step: each step rotates and quantizes the unit
the step before projected, under its first query heads' dots, and projects
the next unit under its last heads' epilogues.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import indexer_q_proj
from vllm_torchtpu.kernels.deepseek_v4.rope import (
    LANE,
    _gptj_sign,
    _qnorm_rope_rows,
)
from vllm_torchtpu.kernels.quantized_matmul import fused_quantize_matmul
from vllm_torchtpu.kernels.quantized_matmul.tuned_block_sizes import (
    get_device_vmem_limit,
)

# Indexer query tokens rotated and scaled per block of rows.
INDEXER_ROPE_TOKENS = 32
# Indexer query heads each grid step projects: one fp8 (32, LANE) tile.
FP8_SUBLANES = indexer_q_proj.FP8_SUBLANES


def _tile_n(m: int, k: int, n: int, itemsize: int, head_dim: int) -> int | None:
    """``fused_quantize_matmul``'s output tile width, in whole heads."""
    # Each token's cos and sin lanes, f32.
    cos_sin_bytes = m * 2 * LANE * 4
    return fused_quantize_matmul.output_tile_n(
        m,
        k,
        n,
        itemsize,
        get_device_vmem_limit(),
        multiple_of=head_dim,
        extra_vmem_bytes=cos_sin_bytes,
    )


def applies(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    head_dim: int,
    rotary_dim: int,
) -> bool:
    """Whether ``q_proj_norm_rope`` takes these operands, the projection's
    output being heads of ``head_dim`` side by side with their last
    ``rotary_dim`` channels roped."""
    m, k = x.shape
    n = w_q.shape[1]
    return (
        fused_quantize_matmul.applies(x, w_q, w_scale)
        and head_dim % LANE == 0
        and n % head_dim == 0
        and rotary_dim % 2 == 0
        and rotary_dim <= LANE
        and _tile_n(m, k, n, x.dtype.itemsize, head_dim) is not None
    )


def _indexer_tile_t(m: int, n: int, tile_n: int, indexer_n: int) -> int | None:
    """Tokens of each grid step's indexer tile: the steps walk the indexer's
    FP8_SUBLANES-head groups, each over all token tiles."""
    steps = n // tile_n
    group_n = FP8_SUBLANES * LANE
    if indexer_n % group_n or steps % (indexer_n // group_n):
        return None
    tiles = steps // (indexer_n // group_n)
    # Step 0 projects units 0 and 1 from one group's weights.
    if tiles < 2 or m % tiles:
        return None
    tile_t = m // tiles
    # Each token's scale goes to a lane of its tile's scales.
    if tile_t % INDEXER_ROPE_TOKENS or tile_t > LANE:
        return None
    return tile_t


def _fused_vmem_bytes(
    m: int, k: int, n: int, tile_n: int, head_dim: int, indexer_n: int, tile_t: int
) -> int:
    """VMEM the kernel takes with the indexer's projection fused in. Unlike
    ``fused_quantize_matmul``'s budget, which ``_tile_n`` keeps, the query's
    f32 dots are a head wide."""
    group_n = FP8_SUBLANES * LANE
    query = m * k * 3 + m * LANE * 4 + 2 * n * 4  # activation, fp8 copy, scales
    query += m * 2 * LANE * 4  # cos and sin
    query += 2 * k * tile_n + 2 * m * tile_n * 2  # double-buffered tiles
    query += 2 * m * head_dim * 4  # two heads' f32 dots in flight
    indexer = 2 * k * group_n + 2 * indexer_n * 4  # double-buffered, and scales
    indexer += 2 * tile_t * group_n + 2 * m * LANE * 4  # fp8 tile, scales
    indexer += 3 * m * LANE * 4  # cos, signed sin and x scales, lane-wide
    indexer += 2 * tile_t * group_n * 2  # two units' bf16 projections
    indexer += (m // tile_t) * indexer_n // LANE * LANE * 4  # scales, transposed
    indexer += tile_t * group_n * 4  # a unit's f32 dot
    return query + indexer


def fuses_indexer(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    head_dim: int,
    rotary_dim: int,
    indexer_w_q: jax.Array,
    indexer_head_dim: int,
) -> bool:
    """Whether ``q_proj_norm_rope`` takes these operands with the indexer's
    projection ``indexer_w_q`` to heads of ``indexer_head_dim``, roped on the
    same ``rotary_dim`` channels."""
    if not applies(x, w_q, w_scale, head_dim, rotary_dim):
        return False
    m, k = x.shape
    n = w_q.shape[1]
    tile_n = _tile_n(m, k, n, x.dtype.itemsize, head_dim)
    indexer_n = indexer_w_q.shape[1]
    tile_t = _indexer_tile_t(m, n, tile_n, indexer_n)
    return (
        indexer_w_q.dtype == w_q.dtype
        and indexer_w_q.shape[0] == k
        and indexer_head_dim == LANE
        and tile_t is not None
        and _fused_vmem_bytes(m, k, n, tile_n, head_dim, indexer_n, tile_t)
        <= fused_quantize_matmul.VMEM_BUDGET_FRACTION * get_device_vmem_limit()
    )


def _kernel(
    # inputs
    x_hbm_ref: jax.Ref,  # (M, K) bf16, HBM
    w_ref: jax.Ref,  # (K, tile_n) fp8
    w_scale_ref: jax.Ref,  # (N,) f32, resident
    cos_sin_hbm_ref: jax.Ref,  # (M, 2 * LANE) f32, HBM
    *refs: jax.Ref,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    return_quantized: bool,
    indexer: bool,
):
    if indexer:
        # iw_ref: (K, FP8_SUBLANES * LANE) fp8, the step's indexer heads.
        # iw_scale_ref: (NI,) f32, resident.
        iw_ref, iw_scale_ref, *refs = refs
    # out_ref: (M, tile_n)
    out_ref, *refs = refs
    if indexer:
        # iq_ref: (tile_t, FP8_SUBLANES, LANE) fp8, the step's indexer query.
        # iscale_ref: (M, HI) f32, resident.
        iq_ref, iscale_ref, *refs = refs
    if return_quantized:
        # The (M, K) fp8 and (M, 1) f32 outputs, and their (2,) DMA semaphores.
        x_q_out_ref, x_scale_out_ref, *refs, out_sems = refs
    if indexer:
        # ilanes_ref: (tiles, 3, tile_t, LANE) f32, each token tile's cos,
        # signed sin and x scales, lane-wide.
        # iproj_ref: (2, tile_t * FP8_SUBLANES, LANE) bf16, two units'
        # projections, token-major.
        # iscale_t_ref: (tiles, HI, LANE) f32, each tile's scales transposed.
        *refs, ilanes_ref, iproj_ref, iscale_t_ref = refs
    # cos_sin_ref: (M, 2 * LANE) f32, each token's forward cos and sin lanes.
    x_ref, x_q_ref, x_scale_ref, sems, cos_sin_ref, cos_sin_sem = refs
    step = pl.program_id(0)

    def quantized_copies():
        return [
            pltpu.make_async_copy(x_q_ref, x_q_out_ref, out_sems.at[0]),
            pltpu.make_async_copy(x_scale_ref, x_scale_out_ref, out_sems.at[1]),
        ]

    if indexer:
        indexer_tiles = iscale_t_ref.shape[0]
        indexer_groups = iscale_t_ref.shape[1] // iq_ref.shape[1]

    def indexer_project(unit):
        """Unit ``unit``'s indexer heads for its token tile, token-major into
        its buffer of iproj_ref."""
        tile_t = iq_ref.shape[0]
        tile = unit % indexer_tiles
        tokens = pl.ds(pl.multiple_of(tile * tile_t, tile_t), tile_t)
        group_n = iw_ref.shape[1]
        # Past the last unit (the last step) this projects a unit nobody reads.
        group = jnp.minimum(unit // indexer_tiles, indexer_groups - 1)
        cols = pl.ds(pl.multiple_of(group * group_n, group_n), group_n)
        indexer_q_proj.project_token_major(
            x_q_ref.at[tokens],
            iw_ref,
            ilanes_ref.at[tile, 2],
            iw_scale_ref.at[cols],
            iproj_ref.at[unit % 2],
        )

    def indexer_rope_quant(unit, t0):
        """Rotates, scales and casts unit ``unit``'s indexer heads of tokens
        t0 .. t0 + INDEXER_ROPE_TOKENS of its tile."""
        heads = iq_ref.shape[1]
        tile = unit % indexer_tiles
        tokens = pl.ds(t0, INDEXER_ROPE_TOKENS)
        scaled, scale = indexer_q_proj.rope_scale_token_major(
            iproj_ref[unit % 2, pl.ds(t0 * heads, INDEXER_ROPE_TOKENS * heads), :],
            ilanes_ref.at[tile, 0, tokens],
            ilanes_ref.at[tile, 1, tokens],
            iq_ref.dtype,
        )
        iq_ref[tokens] = scaled.astype(iq_ref.dtype)
        indexer_q_proj.store_scales_transposed(
            scale,
            iscale_t_ref.at[tile],
            pl.multiple_of(unit // indexer_tiles * heads, heads),
            t0,
            heads,
        )

    @pl.when(step == 0)
    def _():
        copies = fused_quantize_matmul.start_activation_copies(x_hbm_ref, x_ref, sems)
        # Lands while the activation is quantized.
        cos_sin_copy = pltpu.make_async_copy(cos_sin_hbm_ref, cos_sin_ref, cos_sin_sem)
        cos_sin_copy.start()
        fused_quantize_matmul.quantize_activation(x_ref, x_q_ref, x_scale_ref, copies)
        if return_quantized:
            # Written back under the remaining steps' matmuls.
            for copy in quantized_copies():
                copy.start()
        cos_sin_copy.wait()
        # The inverse rotation's sin is the forward one negated on the roped
        # lanes; the NoPE lanes keep the forward rotation's +0. This is exactly
        # ``_cos_sin_lanes(..., inverse=False)``'s sin, so ``qnorm_rope``'s.
        inverse_sin = cos_sin_ref[:, LANE:]
        lane = jax.lax.broadcasted_iota(jnp.int32, inverse_sin.shape, 1)
        cos_sin_ref[:, LANE:] = jnp.where(lane >= LANE - rotary_dim, -inverse_sin, 0.0)
        if indexer:
            tiles, _, tile_t, _ = ilanes_ref.shape
            # `indexer_q_proj`'s lanes: the forward rotation is its rotation.
            lanes = [
                cos_sin_ref[:, :LANE],
                cos_sin_ref[:, LANE:] * _gptj_sign(),
                jnp.broadcast_to(x_scale_ref[...], (tiles * tile_t, LANE)),
            ]
            for i, value in enumerate(lanes):
                ilanes_ref[:, i] = value.reshape(tiles, tile_t, LANE)
            # Step 0's own unit; every step projects the next one's.
            indexer_project(0)

    tile_n = out_ref.shape[1]

    def project(h):
        """fused_quantize_matmul's dot, scaling and bf16 store of head h."""
        cols = pl.ds(h * head_dim, head_dim)
        acc = jax.lax.dot_general(
            x_q_ref[...],
            w_ref[:, cols],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        acc = acc * x_scale_ref[...]
        first = pl.multiple_of(step * tile_n + h * head_dim, LANE)
        acc = acc * w_scale_ref[pl.ds(first, head_dim)][None, :]
        return acc.astype(out_ref.dtype)

    num_heads = tile_n // head_dim
    # The step's unit was projected by the step before, so its rotation
    # needs nothing of this step's and fills the start of the step, where the
    # query heads wait on their first dots. Half of it goes after head 0's
    # epilogue, which schedules tighter than all of it first.
    indexer_blocks = []
    if indexer:
        num_blocks = iq_ref.shape[0] // INDEXER_ROPE_TOKENS
        for b in range(num_blocks // 2):
            indexer_rope_quant(step, b * INDEXER_ROPE_TOKENS)
        indexer_blocks = range(num_blocks // 2, num_blocks)
    # Head h + 1's dot is issued ahead of head h's norm and rotation, so the
    # MXU has work while the VPU runs the epilogue.
    q_next = project(0)
    for h in range(num_heads):
        q = q_next
        if h + 1 < num_heads:
            q_next = project(h + 1)
        # The bf16 query read back as qnorm_rope does.
        y = _qnorm_rope_rows(
            q.astype(jnp.float32),
            cos_sin_ref[:, :LANE],
            cos_sin_ref[:, LANE:],
            eps=eps,
        )
        out_ref[:, pl.ds(h * head_dim, head_dim)] = y.astype(out_ref.dtype)
        if h == 0:
            for b in indexer_blocks:
                indexer_rope_quant(step, b * INDEXER_ROPE_TOKENS)

    if indexer:
        # The next step's unit, its dot under the last heads' epilogues. Not
        # skipped on the last step: a branch would keep it from overlapping.
        indexer_project(step + 1)

        @pl.when(step == pl.num_programs(0) - 1)
        def _():
            tile_t = iq_ref.shape[0]
            for t in range(iscale_t_ref.shape[0]):
                iscale_ref[pl.ds(t * tile_t, tile_t), :] = iscale_t_ref[t, :, :tile_t].T

    if return_quantized:

        @pl.when(step == pl.num_programs(0) - 1)
        def _():
            for copy in quantized_copies():
                copy.wait()


@functools.partial(
    jax.jit, static_argnames=("head_dim", "rotary_dim", "eps", "return_quantized")
)
def q_proj_norm_rope(
    x: jax.Array,  # [M, K] bf16
    w_q: jax.Array,  # [K, N] fp8
    w_scale: jax.Array,  # [N]
    cos_sin: jax.Array,  # [M, 2 * LANE] float32
    indexer_w_q: jax.Array | None = None,  # [K, NI] fp8
    indexer_w_scale: jax.Array | None = None,  # [NI]
    *,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    return_quantized: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array, jax.Array]:
    """``qnorm_rope(fused_quantize_matmul(x, w_q, w_scale).reshape(M, -1,
    head_dim), positions, cos_sin_cache, eps=eps).reshape(M, N)`` where
    ``applies`` holds.

    ``cos_sin`` is ``o_projection.gather_cos_sin(positions, table)`` for
    ``table = o_projection.cos_sin_table(cos_sin_cache, inverse=True, ...)``:
    the rows the o-projection reads, so one gather serves both.

    With ``return_quantized``, also returns the fp8 activation and its (M, 1)
    row scales, as ``fused_quantize_matmul`` does.

    With the indexer's ``indexer_w_q`` and ``indexer_w_scale``, where
    ``fuses_indexer`` holds, also returns the indexer's query
    ``indexer_q_proj.q_proj_rope_quant(*util.quantize_tensor(x, w_q.dtype),
    indexer_w_q, indexer_w_scale, positions, cos_sin_cache)``: the
    ``[M, NI // LANE, LANE]`` fp8 heads and their ``[M, NI // LANE]`` scales.
    """
    assert applies(x, w_q, w_scale, head_dim, rotary_dim), (
        x.shape,
        w_q.shape,
        head_dim,
        rotary_dim,
    )
    m, k = x.shape
    n = w_q.shape[1]
    assert cos_sin.shape == (m, 2 * LANE) and cos_sin.dtype == jnp.float32
    tile_n = _tile_n(m, k, n, x.dtype.itemsize, head_dim)
    assert tile_n is not None
    indexer = indexer_w_q is not None
    w_scale = w_scale.astype(jnp.float32)
    operands = [x, w_q, w_scale, cos_sin]
    in_specs = [
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec((k, tile_n), lambda j: (0, j)),
        # Whole: a rank-1 block must be a power of two or 1024-aligned.
        pl.BlockSpec((n,), lambda j: (0,)),
        pl.BlockSpec(memory_space=pltpu.HBM),
    ]
    out_shape = [jax.ShapeDtypeStruct((m, n), x.dtype)]
    out_specs = [pl.BlockSpec((m, tile_n), lambda j: (0, j))]
    if indexer:
        assert not return_quantized
        assert fuses_indexer(
            x, w_q, w_scale, head_dim, rotary_dim, indexer_w_q, LANE
        ), (
            x.shape,
            indexer_w_q.shape,
        )
        indexer_n = indexer_w_q.shape[1]
        indexer_heads = indexer_n // LANE
        tile_t = _indexer_tile_t(m, n, tile_n, indexer_n)
        tiles = m // tile_t
        groups = indexer_heads // FP8_SUBLANES
        operands += [indexer_w_q, indexer_w_scale.astype(jnp.float32)]
        in_specs += [
            # Step j projects unit j + 1; step 0 also unit 0, of the same group.
            pl.BlockSpec(
                (k, FP8_SUBLANES * LANE),
                lambda j: (0, jnp.minimum((j + 1) // tiles, groups - 1)),
            ),
            pl.BlockSpec((indexer_n,), lambda j: (0,)),
        ]
        out_shape += [
            jax.ShapeDtypeStruct((m, indexer_heads, LANE), indexer_w_q.dtype),
            jax.ShapeDtypeStruct((m, indexer_heads), jnp.float32),
        ]
        out_specs += [
            pl.BlockSpec(
                (tile_t, FP8_SUBLANES, LANE), lambda j: (j % tiles, j // tiles, 0)
            ),
            pl.BlockSpec((m, indexer_heads), lambda j: (0, 0)),
        ]
    scratch_shapes = [
        pltpu.VMEM((m, k), x.dtype),
        pltpu.VMEM((m, k), w_q.dtype),
        pltpu.VMEM((m, 1), jnp.float32),
        pltpu.SemaphoreType.DMA((m // fused_quantize_matmul.SUB_M,)),
        pltpu.VMEM((m, 2 * LANE), jnp.float32),
        pltpu.SemaphoreType.DMA(()),
    ]
    if return_quantized:
        out_shape += [
            jax.ShapeDtypeStruct((m, k), w_q.dtype),
            jax.ShapeDtypeStruct((m, 1), jnp.float32),
        ]
        out_specs += [pl.BlockSpec(memory_space=pltpu.HBM)] * 2
        scratch_shapes.append(pltpu.SemaphoreType.DMA((2,)))
    if indexer:
        scratch_shapes += [
            pltpu.VMEM((tiles, 3, tile_t, LANE), jnp.float32),
            pltpu.VMEM((2, tile_t * FP8_SUBLANES, LANE), jnp.bfloat16),
            pltpu.VMEM((tiles, indexer_heads, LANE), jnp.float32),
        ]
    outs = pl.pallas_call(
        functools.partial(
            _kernel,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            eps=eps,
            return_quantized=return_quantized,
            indexer=indexer,
        ),
        out_shape=out_shape,
        grid=(n // tile_n,),
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch_shapes,
        compiler_params=pltpu.CompilerParams(
            # Steps after the first read the activation and cos/sin the first
            # one wrote.
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=get_device_vmem_limit(),
        ),
        name="q_proj_norm_rope",
    )(*operands)
    return tuple(outs) if return_quantized or indexer else outs[0]
