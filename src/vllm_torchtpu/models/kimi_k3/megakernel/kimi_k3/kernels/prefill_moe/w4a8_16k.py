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

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .. import prefill_megakernel as k

LATENT_E4M3_DTYPE = jnp.float8_e4m3fn
LATENT_E4M3_TAIL_SHAPE = (4, 896)
LATENT_SCALES_SHAPE = (128, 128)
E4M3_MAX = 448.0
FP8_SCALE_EPS = 1e-12
GU_E8M0_SHIFT = 10
GU_MAX_SAFE_E8M0 = 123
GU_POST_SCALE = float(2.0 ** (-GU_E8M0_SHIFT))
SITU_UP_COEFF = 0.04
SITU_SCALE_UP = 25.0
LATENT_14P_PANEL_COLS = 256
LATENT_14P_NUM_PANELS = 14
LATENT_TOTAL_COLS = LATENT_14P_NUM_PANELS * LATENT_14P_PANEL_COLS
GU_M_TILE_ROWS = 128


def quantize_latent_rows_e4m3_ref(
    latent_f32: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    quad_f32 = latent_f32.astype(jnp.float32).reshape(-1, *LATENT_E4M3_TAIL_SHAPE)
    row_amax = jnp.max(jnp.abs(quad_f32), axis=(1, 2))
    scales_f32 = jnp.maximum(row_amax / E4M3_MAX, FP8_SCALE_EPS)
    quad_bf16 = quad_f32.astype(jnp.bfloat16).astype(jnp.float32)
    quantized = jnp.clip(
        quad_bf16 * (1.0 / scales_f32)[:, None, None], -E4M3_MAX, E4M3_MAX
    ).astype(LATENT_E4M3_DTYPE)
    return quantized, scales_f32


def dequant_mxfp4_shifted_e4m3_ref(
    fp4_vals_bf16: jax.Array,
    e8m0_scales: jax.Array,
    *,
    exp_shift: int = GU_E8M0_SHIFT,
) -> jax.Array:
    shifted = e8m0_scales.astype(jnp.uint32) + jnp.uint32(exp_shift)
    powers_bf16 = lax.bitcast_convert_type(
        shifted << jnp.uint32(23), jnp.float32
    ).astype(jnp.bfloat16)
    return (fp4_vals_bf16.astype(jnp.bfloat16) * powers_bf16).astype(LATENT_E4M3_DTYPE)


def compute_dynamic_latent_e4m3(
    own_ref,
    latent_weight,
    buffers,
    semaphores,
    latent_payload_ref,
    lat_scale_payload_ref,
    *,
    prestarted: bool = True,
    on_panels_done: Callable[[], None] | None = None,
) -> None:
    rows = own_ref.shape[0]
    assert rows % 128 == 0
    row_tile = min(rows, 256)
    panel_cols = LATENT_14P_PANEL_COLS
    n_panels = LATENT_14P_NUM_PANELS
    strip_cols = LATENT_E4M3_TAIL_SHAPE[1]

    def copy(panel: int):
        return pltpu.make_async_copy(
            latent_weight.at[:, pl.ds(panel * panel_cols, panel_cols)],
            buffers[panel % 4],
            semaphores[panel % 4],
        )

    @functools.partial(
        pl.run_scoped,
        lat_bf16=pltpu.VMEM((rows, LATENT_TOTAL_COLS), jnp.bfloat16),
    )
    def run(lat_bf16):
        if not prestarted:
            for panel in range(2):
                copy(panel).start()
        row_amax = jnp.zeros((rows,), jnp.float32)
        for panel in range(0, n_panels, 2):
            for j in range(2):
                copy(panel + j).wait()
            for j in range(2):
                if panel + 2 + j < n_panels:
                    copy(panel + 2 + j).start()
            for j in range(2):
                p_idx = panel + j
                w_buf = buffers[p_idx % 4][...]
                tiles = [
                    jnp.matmul(
                        own_ref[pl.ds(r0, row_tile), :],
                        w_buf,
                        preferred_element_type=jnp.float32,
                    )
                    for r0 in range(0, rows, row_tile)
                ]
                prod = jnp.concatenate(tiles, axis=0)
                lat_bf16[:, pl.ds(p_idx * panel_cols, panel_cols)] = prod.astype(
                    jnp.bfloat16
                )
                row_amax = jnp.maximum(row_amax, jnp.max(jnp.abs(prod), axis=1))
        if on_panels_done is not None:
            on_panels_done()
        lat_scale = jnp.maximum(row_amax / E4M3_MAX, FP8_SCALE_EPS)
        inv_scale = 1.0 / lat_scale
        lat_scale_payload_ref[...] = lat_scale.reshape(rows // 128, 128)
        for strip in range(LATENT_E4M3_TAIL_SHAPE[0]):
            col_start = strip * strip_cols
            for r0 in range(0, rows, 32):
                blk = lat_bf16[pl.ds(r0, 32), pl.ds(col_start, strip_cols)].astype(
                    jnp.float32
                )
                quantized = jnp.clip(
                    blk * inv_scale[r0 : r0 + 32, None], -E4M3_MAX, E4M3_MAX
                ).astype(LATENT_E4M3_DTYPE)
                latent_payload_ref[pl.ds(r0, 32), strip, :] = quantized


def _shifted_e8m0_powers_bf16(ref, *, exp_shift_u32: jax.Array) -> jax.Array:
    exponents = ref.scale[...]
    shifted = exponents.astype(jnp.uint32) + exp_shift_u32
    return lax.bitcast_convert_type(shifted << jnp.uint32(23), jnp.float32).astype(
        jnp.bfloat16
    )


def _dequant_mxfp4_panel_e4m3(
    panel_rhs,
    *,
    cfgs,
    panel_width: int,
    exp_shift_u32: jax.Array,
) -> jax.Array:
    block = cfgs.rhs_cfgs.quant_block_size
    w_bf16 = (
        panel_rhs.get_weight()
        .astype(jnp.bfloat16)
        .reshape(cfgs.num_quant_blocks_per_tile_k, block, panel_width)
    )
    p_gate = _shifted_e8m0_powers_bf16(panel_rhs.gate, exp_shift_u32=exp_shift_u32)
    p_up = _shifted_e8m0_powers_bf16(panel_rhs.up, exp_shift_u32=exp_shift_u32)
    powers = k.interleave_lane(p_gate, p_up)
    s_bf16 = jnp.broadcast_to(
        powers[:, None, :],
        (cfgs.num_quant_blocks_per_tile_k, block, panel_width),
    )
    return (
        (w_bf16 * s_bf16)
        .reshape(cfgs.tiles.tile_k, panel_width)
        .astype(LATENT_E4M3_DTYPE)
    )


def _situ_with_row_scales(
    acc_raw: jax.Array,
    *,
    s_gate: jax.Array,
    s_up: jax.Array,
) -> jax.Array:
    acc_gate_raw, acc_up_raw = k.deinterleave_lane(acc_raw)
    acc_gate = acc_gate_raw * s_gate
    gate = 4.0 * jnp.tanh(acc_gate * 0.25) * jax.nn.sigmoid(acc_gate)
    up = SITU_SCALE_UP * jnp.tanh(acc_up_raw * s_up)
    return gate * up


def _fill_w4a8_gu_weight_cache(
    rhs,
    w_cache,
    *,
    cfgs,
    panel_width: int,
    n_panels: int,
    exp_shift_u32: jax.Array,
) -> None:
    half_pw = panel_width // 2

    @pl.loop(0, n_panels)
    def dequant_panel(panel_idx):
        col0_half = pl.multiple_of(panel_idx * half_pw, half_pw)
        p_rhs = k.FusedWeightsRef(
            dataclasses.replace(
                rhs.gate,
                weight=rhs.gate.weight.at[:, pl.ds(col0_half, half_pw)],
                scale=rhs.gate.scale.at[..., pl.ds(col0_half, half_pw)],
            ),
            dataclasses.replace(
                rhs.up,
                weight=rhs.up.weight.at[:, pl.ds(col0_half, half_pw)],
                scale=rhs.up.scale.at[..., pl.ds(col0_half, half_pw)],
            ),
        )
        w_cache[panel_idx] = _dequant_mxfp4_panel_e4m3(
            p_rhs,
            cfgs=cfgs,
            panel_width=panel_width,
            exp_shift_u32=exp_shift_u32,
        )


def _gu_inner_16k_w4a8(
    lhs,
    rhs,
    output,
    partial,
    metadata,
    tile_lat_scales_vmem,
    w_cache,
    *,
    cfgs,
    panel_width: int,
) -> None:
    group = pl.program_id(1)
    start = metadata.gm_id_to_m_offset[group]
    end = metadata.gm_id_to_m_offset[group + 1]
    sub = cfgs.dims.size_lhs_sublane
    aligned = start - start % sub
    first, last = start - aligned, end - aligned
    width = 2 * rhs.gate.weight.shape[-1]
    assert width % panel_width == 0
    n_panels = width // panel_width
    out_w = panel_width // 2
    m_tile = GU_M_TILE_ROWS
    assert cfgs.tiles.tile_m % m_tile == 0 and m_tile % sub == 0
    m_tile_sub = m_tile // sub
    exp_shift_u32 = jnp.uint32(GU_E8M0_SHIFT)
    post_scale = jnp.float32(GU_POST_SCALE)
    post_scale_up = jnp.float32(GU_POST_SCALE * SITU_UP_COEFF)

    cur_gid = metadata.gm_id_to_group_id[group]
    prev_gid = metadata.gm_id_to_group_id[jnp.maximum(group - 1, 0)]
    same_prev = (group > 0) & (cur_gid == prev_gid)

    @pl.when(~same_prev)
    def refresh_w_cache():
        _fill_w4a8_gu_weight_cache(
            rhs,
            w_cache,
            cfgs=cfgs,
            panel_width=panel_width,
            n_panels=n_panels,
            exp_shift_u32=exp_shift_u32,
        )

    n_m_tiles = jnp.maximum((last + (m_tile - 1)) // m_tile, jnp.int32(1))

    @pl.loop(0, n_m_tiles)
    def run_m_tile(m_blk):
        m0 = pl.multiple_of(m_blk * m_tile, m_tile)
        m0_sub = pl.multiple_of(m_blk * m_tile_sub, m_tile_sub)
        act = lhs[pl.ds(m0, m_tile)].reshape(m_tile, cfgs.tiles.tile_k)
        raw_scales = tile_lat_scales_vmem[pl.ds(m0, m_tile)]
        s_gate = (raw_scales * post_scale).reshape(m_tile, 1)
        s_up = (raw_scales * post_scale_up).reshape(m_tile, 1)
        row = m0 + lax.broadcasted_iota(jnp.int32, (m_tile, out_w), 0)
        mask = (first <= row) & (row < last)
        for p_idx in range(n_panels):
            prod_raw = jnp.matmul(
                act, w_cache[p_idx], preferred_element_type=jnp.float32
            )
            val = _situ_with_row_scales(prod_raw, s_gate=s_gate, s_up=s_up)
            val = jnp.where(mask, val, 0.0)
            output[pl.ds(m0_sub, m_tile_sub), :, pl.ds(p_idx * out_w, out_w)] = (
                val.astype(output.dtype).reshape(m_tile_sub, sub, out_w)
            )

    zeros = jnp.zeros_like(partial)
    output[0] += jnp.where(group == 0, zeros, partial[...])
    last_row = jnp.minimum(last // sub, (cfgs.tiles.tile_m // sub) - 1)
    partial[...] = jnp.where(last % sub == 0, zeros, output[last_row])


def run_gu_w4a8_16k(
    gu,
    gu_ws,
    gu_ys,
    metadata,
    latent_ref,
    latent_scales_hbm,
    destinations_ref,
    middle_ref,
    *,
    gu_cfg,
    active_ng,
    ng,
    sub: int,
    tm: int,
    mid_width: int,
    tokens: int,
    n_act_banks: int,
    panel_width: int,
    buf_n: Callable[[object, int], object],
) -> None:
    assert (2 * mid_width) % panel_width == 0
    n_panels = (2 * mid_width) // panel_width

    @functools.partial(
        pl.run_scoped,
        gu_partial=pltpu.VMEM((sub, mid_width), gu_cfg.out_dtype),
        acts=pltpu.VMEM((n_act_banks, tm, *LATENT_E4M3_TAIL_SHAPE), LATENT_E4M3_DTYPE),
        act_sems=pltpu.SemaphoreType.DMA((n_act_banks,)),
        lat_scales_smem=pltpu.SMEM(LATENT_SCALES_SHAPE, jnp.float32),
        lat_scales_load_sem=pltpu.SemaphoreType.DMA,
        tile_scales_smem=pltpu.SMEM((n_act_banks * tm,), jnp.float32),
        tile_scales_vmem=pltpu.VMEM((n_act_banks * tm,), jnp.float32),
        tile_scale_sems=pltpu.SemaphoreType.DMA((n_act_banks,)),
        w_cache=pltpu.VMEM(
            (n_panels, gu_cfg.tiles.tile_k, panel_width), LATENT_E4M3_DTYPE
        ),
    )
    def run_gu(
        gu_partial,
        acts,
        act_sems,
        lat_scales_smem,
        lat_scales_load_sem,
        tile_scales_smem,
        tile_scales_vmem,
        tile_scale_sems,
        w_cache,
    ):
        lat_scales_flat = lat_scales_smem.reshape(1, tokens)
        gu_ws3 = jax.tree.map(
            lambda s: buf_n(s, 3),
            gu_ws,
            is_leaf=lambda v: isinstance(v, pl.BlockSpec),
        )

        def active_bounds(g):
            raw_start = metadata.gm_id_to_m_offset[g]
            raw_end = metadata.gm_id_to_m_offset[g + 1]
            begin = raw_start // sub * sub
            return (raw_start, raw_start - begin, raw_end - raw_start)

        def start_gather(g, bank):
            raw_start, first, n_active = active_bounds(g)
            bank_off = bank * tm
            unroll_g = 16
            full = (n_active // unroll_g) * unroll_g

            @pl.loop(0, n_active // unroll_g)
            def row_vec(blk):
                base = blk * unroll_g
                sources = [
                    destinations_ref[raw_start + base + u] for u in range(unroll_g)
                ]
                for u in range(unroll_g):
                    pltpu.make_async_copy(
                        latent_ref.at[pl.ds(sources[u], 1)],
                        acts.at[bank, pl.ds(first + base + u, 1)],
                        act_sems.at[bank],
                    ).start()
                for u in range(unroll_g):
                    tile_scales_smem[bank_off + first + base + u] = lat_scales_flat[
                        0, sources[u]
                    ]

            @pl.loop(full, n_active)
            def row_rem(i):
                source = destinations_ref[raw_start + i]
                pltpu.make_async_copy(
                    latent_ref.at[pl.ds(source, 1)],
                    acts.at[bank, pl.ds(first + i, 1)],
                    act_sems.at[bank],
                ).start()
                tile_scales_smem[bank_off + first + i] = lat_scales_flat[0, source]

            pltpu.make_async_copy(
                tile_scales_smem.at[pl.ds(bank_off, tm)],
                tile_scales_vmem.at[pl.ds(bank_off, tm)],
                tile_scale_sems.at[bank],
            ).start()

        @pl.when(active_ng > 0)
        def init_before_pipeline():
            ls_copy = pltpu.make_async_copy(
                latent_scales_hbm, lat_scales_smem, lat_scales_load_sem
            )
            ls_copy.start()
            acts[...] = jnp.zeros(acts.shape, acts.dtype)
            tile_scales_vmem[...] = jnp.zeros(tile_scales_vmem.shape, jnp.float32)
            zero_scales = pltpu.make_async_copy(
                tile_scales_vmem, tile_scales_smem, tile_scale_sems.at[0]
            )
            zero_scales.start()
            ls_copy.wait()
            zero_scales.wait()

            @pl.loop(0, jnp.minimum(ng, jnp.int32(n_act_banks)))
            def seed_bank(p):
                start_gather(p, p)

        def gu_step(
            gw,
            gy,
            metadata,
            gu_partial,
            acts,
            act_sems,
            tile_scales_smem,
            tile_scales_vmem,
            tile_scale_sems,
            w_cache,
        ):
            g = pl.program_id(1)
            bank = g % n_act_banks

            @pl.when(active_ng > 0)
            def gate_up():
                bank_off = bank * tm
                _, first, n_active = active_bounds(g)

                @pl.when(n_active > 0)
                def wait_active():
                    active = acts.at[bank, pl.ds(first, n_active)]
                    pltpu.make_async_copy(active, active, act_sems.at[bank]).wait()

                pltpu.make_async_copy(
                    tile_scales_smem.at[pl.ds(bank_off, tm)],
                    tile_scales_vmem.at[pl.ds(bank_off, tm)],
                    tile_scale_sems.at[bank],
                ).wait()

                _gu_inner_16k_w4a8(
                    acts.at[bank],
                    gw,
                    gy,
                    gu_partial,
                    metadata,
                    tile_scales_vmem.at[pl.ds(bank_off, tm)],
                    w_cache,
                    cfgs=gu_cfg,
                    panel_width=panel_width,
                )

                @pl.when(g + n_act_banks < ng)
                def prefetch_ahead():
                    start_gather(g + n_act_banks, bank)

        pltpu.emit_pipeline(
            gu_step,
            grid=(1, ng, 1),
            in_specs=(gu_ws3,),
            out_specs=gu_ys,
        )(
            gu,
            middle_ref.reshape(-1, sub, mid_width),
            scratches=(
                metadata,
                gu_partial,
                acts,
                act_sems,
                tile_scales_smem,
                tile_scales_vmem,
                tile_scale_sems,
                w_cache,
            ),
        )
