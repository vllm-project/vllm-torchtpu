# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""A window of a core's routed-expert GEMMs as one Pallas call.

Each expert's routes in the window (sorted by expert, ``moe.core_routes``)
are taken in tiles of up to ``ROWS`` routes. A tile's rows come quantized per
row to FP8 (gathered in route order, as ``[slab, H / slab]`` slabs so that any
row is whole VMEM tiles). The expert's weights are E2M1 with a scale per
output channel and a ratio per 64-row block (``requantization_scales``); each
weight tile is requantized to FP8 in VMEM once and multiplied in one dot over
the whole contraction. ``silu(gate) * up`` is quantized per row again for the
down projection, and the tile's BF16 outputs go to its routes' places in the
window, as slabs.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.qwen38.reference import BF16, F32

F8 = jnp.float8_e4m3fn
F8_MAX = float(jnp.finfo(F8).max)
E2M1_MAX = float(jnp.finfo(jnp.float4_e2m1fn).max)
BLOCK = 64  # the weights' ratio block along the contraction
ROWS = 1024  # routes per tile: an expert's routes at a 32K step mostly fit one
CHUNK = 64  # rows per staged load and store
SCALE_ALIGN = 128  # a tile's row scales are copied from this alignment
BUCKETS = (256, 512, 768, ROWS)  # a tile's rows in the dots: the smallest holding them
GATE_TILE = 256  # gate columns per gate/up tile (with the matching up columns)
OUT_SLAB = 16  # the outputs' slab rows: [16, H / 16] BF16 rows are whole tiles
DOWN_TILE = 1024  # output columns per down tile
SUB_TILE = 256  # columns per dot: the MXU's width
WEIGHT_SLOTS = 3  # weight tiles in VMEM: two on their way while one is used


def requantization_scales(block_scale: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Per-channel scales ``[..., 1, N]`` (FP32) and per-block ratios ``[...,
    K / BLOCK, N]`` (BF16) from E2M1 block scales ``[..., K / BLOCK, N]``: an
    E2M1 value times its ratio fits E4M3, and times the channel scale is the
    weight."""
    channel = jnp.max(block_scale, axis=-2, keepdims=True) * (E2M1_MAX / F8_MAX)
    ratio = block_scale / jnp.where(channel == 0, 1, channel)
    return channel.astype(F32), ratio.astype(BF16)


def nvfp4_expert_weights(
    packed: jax.Array, scale: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``expert_products``' weights from the NVFP4 checkpoint's: packed E2M1
    ``[..., N, K / 2]`` and fused block-16 scales ``[..., N, K / 16]`` ->
    E2M1 ``[..., K, N]``, per-channel scales and per-block ratios."""
    from vllm_torchtpu.layers.core.fused_moe_gmm import requant_unpack_kmajor

    weight, block_scale = requant_unpack_kmajor(packed, scale, BLOCK)
    return weight, *requantization_scales(jnp.squeeze(block_scale, -2))


def per_row_fp8(x):
    """``x`` ``[m, k]`` F32 as E4M3 with one scale per row ``[m, 1]``."""
    scale = jnp.max(jnp.abs(x), axis=1, keepdims=True) / F8_MAX
    return (x / jnp.where(scale == 0, 1, scale)).astype(F8), scale


def _requantized(w, ratio):
    """An E2M1 tile ``[k, n]`` times its BF16 ratios ``[k / BLOCK, n]``, as
    E4M3 (the products are exact in BF16 up to the ratio's rounding)."""
    k, n = w.shape
    scaled = w.astype(BF16).reshape(k // BLOCK, BLOCK, n) * ratio[:, None, :]
    return scaled.reshape(k, n).astype(F8)


def reference_products(rows, w13, c13, r13, w2, c2, r2):
    """The kernel's math for one expert's rows ``[m, H]`` (BF16), unfused."""
    inter = w2.shape[0]
    xq, xs = per_row_fp8(rows.astype(F32))
    acc = jnp.dot(xq, _requantized(w13, r13), preferred_element_type=F32) * xs * c13
    hidden = (jax.nn.silu(acc[:, :inter]) * acc[:, inter:]).astype(BF16)
    hq, hs = per_row_fp8(hidden.astype(F32))
    out = jnp.dot(hq, _requantized(w2, r2), preferred_element_type=F32) * hs * c2
    return out.astype(BF16)


def _tiles(group_start, group_size, span: int):
    """The window's tiles (group, first row, rows), live ones first, and how
    many are live: at most one partial tile per group."""
    groups = group_start.shape[0]
    chunks = -(-span // ROWS)
    limit = groups + chunks
    chunk = jnp.arange(chunks, dtype=jnp.int32)
    live = chunk[None, :] * ROWS < group_size[:, None]
    key = jnp.where(
        live,
        jnp.arange(groups, dtype=jnp.int32)[:, None] * chunks + chunk[None, :],
        groups * chunks,
    ).reshape(-1)
    key = lax.sort(key)[:limit]
    group = jnp.minimum(key // chunks, groups - 1)
    offset = key % chunks * ROWS
    first = group_start[group] + offset
    rows = jnp.clip(group_size[group] - offset, 0, ROWS)
    return group, first, rows, jnp.sum(live, dtype=jnp.int32)


def expert_products(
    rows, row_scale, group_start, group_size, weights, *, span, interpret=False
):
    """The window's expert outputs ``[span, 16, H / 16]`` BF16 from its routes'
    rows ``rows`` ``[span + ROWS, slab, H / slab]`` FP8 (sorted, the tail
    padding) and their scales ``row_scale`` ``[span + ROWS + SCALE_ALIGN]``
    (``per_row_fp8``), ``group_start`` / ``group_size`` ``[G]`` each expert's rows
    in the window and ``weights`` = (``w13`` ``[G, H, 2 I]`` E2M1 with ``c13``
    ``[G, 1, 2 I]`` and ``r13`` ``[G, H / BLOCK, 2 I]``, ``w2`` ``[G, I, H]``
    with ``c2``, ``r2``). Rows of no tile are left unwritten. ``interpret``
    runs on the host.

    A tile's rows arrive in chunks during the previous tile's down steps; its
    outputs become slabs and leave in chunks during the next tile's gate/up
    steps: that vector work shares its step with an MXU dot. Each weight tile
    streams once per tile of its expert."""
    w13, c13, r13, w2, c2, r2 = weights
    _, slab, columns = rows.shape
    hidden = slab * columns
    out_slab = OUT_SLAB
    out_columns = hidden // out_slab
    inter = w2.shape[1]
    gate_steps, down_steps = inter // GATE_TILE, hidden // DOWN_TILE
    steps = gate_steps + down_steps
    chunks = ROWS // CHUNK
    assert chunks % gate_steps == 0 and chunks % down_steps == 0
    per_gate, per_down = chunks // gate_steps, chunks // down_steps
    t_group, t_first, t_rows, live = _tiles(group_start, group_size, span)

    def kernel(
        live_ref,
        group_ref,
        first_ref,
        rows_ref,
        x_hbm,
        scale_hbm,
        w13_hbm,
        c13_hbm,
        r13_hbm,
        w2_hbm,
        c2_hbm,
        r2_hbm,
        out_hbm,
        x_stage,
        scale_smem,
        xq,
        xs,
        w_gu,
        r_gu,
        c_gu,
        w_dn,
        r_dn,
        c_dn,
        hidden_vmem,
        hq,
        hs,
        out_rows,
        out_stage,
        x_sems,
        scale_sems,
        gu_sems,
        dn_sems,
        out_sems,
    ):
        n_tiles = live_ref[0]

        def x_chunk(t, c):
            return pltpu.make_async_copy(
                x_hbm.at[pl.ds(first_ref[t] + c * CHUNK, CHUNK)],
                x_stage.at[c % 2],
                x_sems.at[c % 2],
            )

        def used(t):
            """Tile ``t``'s chunks holding rows."""
            return pl.cdiv(rows_ref[t], CHUNK)

        def scale_base(t):
            return pl.multiple_of(
                first_ref[t] // SCALE_ALIGN * SCALE_ALIGN, SCALE_ALIGN
            )

        def scale_copy(t):
            return pltpu.make_async_copy(
                scale_hbm.at[pl.ds(scale_base(t), ROWS + SCALE_ALIGN)],
                scale_smem.at[t % 2],
                scale_sems.at[t % 2],
            )

        def load_rows(t):
            """Tile ``t``'s row scales and first two chunks start loading."""
            scale_copy(t).start()
            for c in range(2):
                pl.when(c < used(t))(x_chunk(t, c).start)

        def take_rows(t, c):
            """Chunk ``c`` of tile ``t``'s rows into ``xq`` / ``xs``, and the
            chunk after the next on its way into the freed staging slot."""
            pl.when(c == 0)(scale_copy(t).wait)
            x_chunk(t, c).wait()
            part = pl.ds(pl.multiple_of(c * CHUNK, CHUNK), CHUNK)
            xq[part] = x_stage[c % 2].reshape(CHUNK, hidden)
            skip = first_ref[t] - scale_base(t) + c * CHUNK
            for r in range(CHUNK):
                xs[pl.ds(c * CHUNK + r, 1)] = jnp.full(
                    (1, xs.shape[1]), scale_smem[t % 2, skip + r], F32
                )

            @pl.when(c + 2 < used(t))
            def _():
                x_chunk(t, c + 2).start()

        def out_chunk(t, c):
            valid = jnp.minimum(rows_ref[t] - c * CHUNK, CHUNK)
            return pltpu.make_async_copy(
                out_stage.at[c % 2, pl.ds(0, valid)],
                out_hbm.at[pl.ds(first_ref[t] + c * CHUNK, valid)],
                out_sems.at[c % 2],
            )

        def send_rows(t, c):
            """Chunk ``c`` of tile ``t``'s outputs as slabs, on its way out
            (once the chunk two before has left the staging slot)."""

            @pl.when(c * CHUNK < rows_ref[t])
            def _():
                @pl.when(c >= 2)
                def _():
                    out_chunk(t, c - 2).wait()

                part = pl.ds(pl.multiple_of(c * CHUNK, CHUNK), CHUNK)
                out_stage[c % 2] = out_rows[part].reshape(CHUNK, out_slab, out_columns)
                out_chunk(t, c).start()

        def sent(t):
            """Tile ``t``'s last two chunks have left (the others were waited
            for by the chunks two after them)."""
            for back in (2, 1):
                pl.when(used(t) - back >= 0)(out_chunk(t, used(t) - back).wait)

        def weight_copies(u):
            """Step ``u``'s weight, ratio and channel-scale tiles: gate/up
            (the gate's columns and the matching up columns), then down."""
            t, s, slot = u // steps, u % steps, u % WEIGHT_SLOTS
            g = group_ref[t]
            gate = pl.ds(pl.multiple_of(s * GATE_TILE, 128), GATE_TILE)
            up = pl.ds(pl.multiple_of(inter + s * GATE_TILE, 128), GATE_TILE)
            halves = (pl.ds(0, GATE_TILE), pl.ds(GATE_TILE, GATE_TILE))
            gate_up = [
                pltpu.make_async_copy(
                    src.at[g, :, cols], dst.at[slot, :, half], gu_sems.at[k, slot]
                )
                for k, (src, dst) in enumerate(
                    ((w13_hbm, w_gu), (r13_hbm, r_gu), (c13_hbm, c_gu))
                )
                for cols, half in zip((gate, up), halves, strict=True)
            ]
            out_cols = pl.ds(
                pl.multiple_of((s - gate_steps) * DOWN_TILE, 128), DOWN_TILE
            )
            down = [
                pltpu.make_async_copy(
                    src.at[g, :, out_cols], dst.at[slot], dn_sems.at[k, slot]
                )
                for k, (src, dst) in enumerate(
                    ((w2_hbm, w_dn), (r2_hbm, r_dn), (c2_hbm, c_dn))
                )
            ]
            return s < gate_steps, gate_up, down

        def start_weights(u):
            is_gate, gate_up, down = weight_copies(u)

            @pl.when(is_gate)
            def _():
                for copy in gate_up:
                    copy.start()

            @pl.when(jnp.logical_not(is_gate))
            def _():
                for copy in down:
                    copy.start()

        def wait_weights(u):
            is_gate, gate_up, down = weight_copies(u)

            @pl.when(is_gate)
            def _():
                for copy in gate_up:
                    copy.wait()

            @pl.when(jnp.logical_not(is_gate))
            def _():
                for copy in down:
                    copy.wait()

        def sub_dot(lhs, w_ref, r_ref, c_ref, slot, cols):
            """``lhs`` times the requantized columns ``cols`` of a weight tile
            and their channel scales. A tile is split into sub-tiles of
            ``SUB_TILE`` columns, so one's requantization (vector unit) can
            overlap another's dot (MXU)."""
            w = _requantized(w_ref[slot, :, cols], r_ref[slot, :, cols])
            return jnp.dot(lhs, w, preferred_element_type=F32) * c_ref[slot, :, cols]

        def gate_up_step(m, s, slot):
            """``m`` rows (a static bucket) through gate/up tile ``s``."""
            for k in range(GATE_TILE // SUB_TILE):
                gate, up = (
                    sub_dot(
                        xq[:m],
                        w_gu,
                        r_gu,
                        c_gu,
                        slot,
                        pl.ds(h + k * SUB_TILE, SUB_TILE),
                    )
                    * xs[:m, :1]
                    for h in (0, GATE_TILE)
                )
                cols = pl.ds(
                    pl.multiple_of(s * GATE_TILE + k * SUB_TILE, 128), SUB_TILE
                )
                hidden_vmem[:m, cols] = (jax.nn.silu(gate) * up).astype(BF16)

        def down_step(m, j, slot):
            for k in range(DOWN_TILE // SUB_TILE):
                part = pl.ds(k * SUB_TILE, SUB_TILE)
                acc = sub_dot(hq[:m], w_dn, r_dn, c_dn, slot, part) * hs[:m, :1]
                cols = pl.ds(
                    pl.multiple_of(j * DOWN_TILE + k * SUB_TILE, 128), SUB_TILE
                )
                out_rows[:m, cols] = acc.astype(BF16)

        def by_bucket(t, step):
            """``step(m)`` for tile ``t``'s bucket ``m``."""
            low = 0
            for m in BUCKETS:
                pl.when((rows_ref[t] > low) & (rows_ref[t] <= m))(
                    functools.partial(step, m)
                )
                low = m

        @pl.when(n_tiles > 0)
        def _():
            load_rows(0)

            @pl.loop(0, used(0))
            def _(c):
                take_rows(0, c)

            start_weights(0)
            pl.when(n_tiles * steps > 1)(functools.partial(start_weights, 1))

        @pl.loop(0, n_tiles * steps)
        def _(u):
            t, s, slot = u // steps, u % steps, u % WEIGHT_SLOTS
            wait_weights(u)

            @pl.when(u + 2 < n_tiles * steps)
            def _():
                start_weights(u + 2)

            @pl.when(s < gate_steps)
            def _():
                by_bucket(t, lambda m: gate_up_step(m, s, slot))

                # The next tile's first rows start loading.
                @pl.when((s == 0) & (t + 1 < n_tiles))
                def _():
                    load_rows(t + 1)

                # The previous tile's outputs leave, a few chunks a step.
                @pl.when(t > 0)
                def _():
                    for k in range(per_gate):
                        send_rows(t - 1, s * per_gate + k)

                    pl.when(s == gate_steps - 1)(functools.partial(sent, t - 1))

            @pl.when(s == gate_steps - 1)
            def _():
                @pl.loop(0, used(t))
                def _(c):
                    part = pl.ds(pl.multiple_of(c * CHUNK, CHUNK), CHUNK)
                    q, scale = per_row_fp8(hidden_vmem[part].astype(F32))
                    hq[part] = q
                    hs[part] = jnp.broadcast_to(scale, (CHUNK, hs.shape[1]))

            @pl.when(s >= gate_steps)
            def _():
                j = s - gate_steps
                by_bucket(t, lambda m: down_step(m, j, slot))

                # The next tile's rows are quantized, a few chunks a step.
                @pl.when(t + 1 < n_tiles)
                def _():
                    for k in range(per_down):
                        c = j * per_down + k
                        pl.when(c < used(t + 1))(functools.partial(take_rows, t + 1, c))

        # The last tile's outputs.
        @pl.when(n_tiles > 0)
        def _():
            @pl.loop(0, chunks)
            def _(c):
                send_rows(n_tiles - 1, c)

            sent(n_tiles - 1)

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    f4 = w13.dtype
    vmem_bytes = (
        2 * CHUNK * hidden * (1 + 2)  # both row and both output staging slots
        + ROWS * hidden  # FP8 rows
        + WEIGHT_SLOTS * hidden * 2 * GATE_TILE  # gate/up weight tiles, E4M3 at most
        + hidden * 2 * GATE_TILE * 4  # one's F32 requantization
        + ROWS * inter * (2 + 1)
        + CHUNK * inter * 4  # hidden and its FP8 copy
        + ROWS * hidden * 2  # output rows
        + ROWS * DOWN_TILE * 4  # F32 products
    )
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((span, out_slab, out_columns), BF16),
        in_specs=[smem, smem, smem, smem] + [hbm] * 8,
        out_specs=hbm,
        scratch_shapes=[
            pltpu.VMEM((2, CHUNK, slab, columns), rows.dtype),
            pltpu.SMEM((2, ROWS + SCALE_ALIGN), F32),
            pltpu.VMEM((ROWS, hidden), F8),
            pltpu.VMEM((ROWS, 128), F32),
            pltpu.VMEM((WEIGHT_SLOTS, hidden, 2 * GATE_TILE), f4),
            pltpu.VMEM((WEIGHT_SLOTS, hidden // BLOCK, 2 * GATE_TILE), BF16),
            pltpu.VMEM((WEIGHT_SLOTS, 1, 2 * GATE_TILE), F32),
            pltpu.VMEM((WEIGHT_SLOTS, inter, DOWN_TILE), f4),
            pltpu.VMEM((WEIGHT_SLOTS, inter // BLOCK, DOWN_TILE), BF16),
            pltpu.VMEM((WEIGHT_SLOTS, 1, DOWN_TILE), F32),
            pltpu.VMEM((ROWS, inter), BF16),
            pltpu.VMEM((ROWS, inter), F8),
            pltpu.VMEM((ROWS, 128), F32),
            pltpu.VMEM((ROWS, hidden), BF16),
            pltpu.VMEM((2, CHUNK, out_slab, out_columns), BF16),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((3, WEIGHT_SLOTS)),
            pltpu.SemaphoreType.DMA((3, WEIGHT_SLOTS)),
            pltpu.SemaphoreType.DMA((2,)),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=min(vmem_bytes + 2**23, 60 * 2**20)
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
        name="qwen38_expert_products",
    )(live[None], t_group, t_first, t_rows, rows, row_scale, w13, c13, r13, w2, c2, r2)
