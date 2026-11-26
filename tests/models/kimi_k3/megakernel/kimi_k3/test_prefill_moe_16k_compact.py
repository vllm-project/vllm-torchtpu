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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels.prefill_moe import (
    routed_16k,
    shared_down_16k,
    w4a8_16k,
)


def test_should_select_smallest_covering_bucket_across_boundary_pairs():
    pass0_buckets = routed_16k.DOWN_PASS0_BUCKETS_16K
    all_buckets = routed_16k.DOWN_ALL_BUCKETS_16K

    pass0_cases = (
        (0, 0),
        (1, 0),
        (64, 0),
        (65, 0),
        (128, 0),
        (129, 1),
        (192, 1),
        (193, 2),
        (256, 2),
        (257, 3),
        (320, 3),
        (321, 4),
        (384, 4),
    )
    for last_rows, expected_idx in pass0_cases:
        got = int(routed_16k.select_bucket_index_for_rows(last_rows, pass0_buckets))
        assert got == expected_idx, f"last={last_rows}: got {got}, want {expected_idx}"
        assert pass0_buckets[got] >= last_rows
        if got > 0:
            assert pass0_buckets[got - 1] < last_rows

    all_cases = (
        (0, 0),
        (1, 0),
        (64, 0),
        (65, 1),
        (128, 1),
        (129, 2),
        (192, 2),
        (193, 3),
        (256, 3),
        (257, 4),
        (320, 4),
        (321, 5),
        (384, 5),
    )
    for last_rows, expected_idx in all_cases:
        got = int(routed_16k.select_bucket_index_for_rows(last_rows, all_buckets))
        assert got == expected_idx

    assert int(routed_16k.select_bucket_index_for_rows(129, pass0_buckets)) != int(
        routed_16k.select_bucket_index_for_rows(128, pass0_buckets)
    )

    assert int(routed_16k.select_bucket_index_for_rows(1, (384,))) == 0
    assert int(routed_16k.select_bucket_index_for_rows(384, (384,))) == 0

    for bad_spec in ((), (0, 128), (-64, 128), (128, 128), (256, 128)):
        with pytest.raises(ValueError, match="strictly increasing"):
            routed_16k.select_bucket_index_for_rows(64, bad_spec)


def test_should_preserve_compact_16k_down_pass_and_w4a8_gu_contracts():
    p0, p1, p2 = routed_16k.DOWN_PASS_SPECS_16K

    assert routed_16k.DOWN_OVERLAP_UNROLL_16K == 16
    assert p0.overlap_unroll == 16
    assert p0.skip_last_panel_overlap is True
    assert (
        p0.bucket_rows
        == routed_16k.DOWN_PASS0_BUCKETS_16K
        == (
            128,
            192,
            256,
            320,
            384,
        )
    )

    for pass_spec in (p1, p2):
        assert pass_spec.overlap_unroll == 0
        assert pass_spec.skip_last_panel_overlap is False
        assert (
            pass_spec.bucket_rows
            == routed_16k.DOWN_ALL_BUCKETS_16K
            == (
                64,
                128,
                192,
                256,
                320,
                384,
            )
        )

    for pass_spec in routed_16k.DOWN_PASS_SPECS_16K:
        assert pass_spec.bucket_rows[-1] == routed_16k.TILE_M_16K
        assert all(
            rows % routed_16k.BUCKET_BASE_16K == 0 for rows in pass_spec.bucket_rows
        )
        assert list(pass_spec.bucket_rows) == sorted(set(pass_spec.bucket_rows))

    uniform_cw = routed_16k.HALF_RING_1536_COL_SPEC[0][1]
    assert uniform_cw == 512
    assert all(
        col_start == i * uniform_cw and col_width == uniform_cw
        for i, (col_start, col_width) in enumerate(routed_16k.HALF_RING_1536_COL_SPEC)
    )
    assert len({cw for _, cw in routed_16k.HALF_RING_1280_COL_SPEC}) > 1

    assert w4a8_16k.GU_M_TILE_ROWS == 128
    assert w4a8_16k.GU_M_TILE_ROWS % 128 == 0
    assert routed_16k.TILE_M_16K % w4a8_16k.GU_M_TILE_ROWS == 0

    tm = routed_16k.TILE_M_16K
    tile_k = routed_16k.OUTPUT_WIDTH_16K
    n_panels = (2 * 1536) // routed_16k.GU_PANEL_WIDTH_16K
    acts_e4m3_bytes = routed_16k.GU_ACT_BANKS_16K * tm * tile_k
    w_cache_e4m3_bytes = n_panels * tile_k * routed_16k.GU_PANEL_WIDTH_16K
    gu_weights_buf3_bytes = 3 * (tile_k * 3072 // 2)
    gu_out_buf2_bytes = 2 * (tm * 1536 * 2)
    tile_scales_bytes = routed_16k.GU_ACT_BANKS_16K * tm * 4
    assert (
        acts_e4m3_bytes
        + w_cache_e4m3_bytes
        + gu_weights_buf3_bytes
        + gu_out_buf2_bytes
        + tile_scales_bytes
        < 36 * 1024 * 1024
    )


@pytest.mark.parametrize("parity", (0, 1))
def test_should_match_fp32_reference_in_compact_shared_down_512_poll2_for_both_parities(
    parity: int,
):
    rng = np.random.default_rng(7900 + parity)
    rows = 128
    inter = shared_down_16k.SHARED_INTER_16K
    hidden = shared_down_16k.SHARED_HIDDEN_16K
    n_panels = shared_down_16k.SHARED_DOWN_NUM_PANELS_16K
    panel_cols = shared_down_16k.SHARED_DOWN_PANEL_COLS_16K

    middle_np = np.concatenate(
        [
            rng.normal(0.20, 0.35, size=(rows, inter)).astype(np.float32),
            rng.normal(-0.30, 0.45, size=(rows, inter)).astype(np.float32),
        ],
        axis=0,
    )
    panel_ramps = np.repeat(
        np.linspace(0.5, 1.8, n_panels, dtype=np.float32), panel_cols
    )
    w_down_np = (
        rng.normal(0.0, 0.04, size=(inter, hidden)).astype(np.float32) * panel_ramps
    )
    middle_bf16 = jnp.asarray(middle_np, dtype=jnp.bfloat16)
    w_down_bf16 = jnp.asarray(w_down_np, dtype=jnp.bfloat16)

    expected_bf16 = (
        jnp.matmul(middle_bf16[:rows], w_down_bf16, preferred_element_type=jnp.float32)
        + jnp.matmul(
            middle_bf16[rows:], w_down_bf16, preferred_element_type=jnp.float32
        )
    ).astype(jnp.bfloat16)

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    def kernel(mid_hbm, w_down_hbm, shared_hbm, poll_cnt_ref):
        poll_cnt_ref[0] = 0

        def on_poll() -> None:
            poll_cnt_ref[0] = poll_cnt_ref[0] + 1

        def body(sc) -> None:
            pltpu.sync_copy(mid_hbm, sc["middle_v"])
            shared_down_16k.run_shared_down_512_poll2(
                sc["middle_v"],
                w_down_hbm,
                shared_hbm,
                sc["down_ready"],
                rows=rows,
                parity=parity,
                peer=0,
                on_poll=on_poll,
            )

        pl.run_scoped(
            body,
            dict(
                middle_v=pltpu.VMEM((2 * rows, inter), jnp.bfloat16),
                down_ready=pltpu.SemaphoreType.REGULAR,
            ),
        )

    call = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((rows, hidden), jnp.bfloat16),
            jax.ShapeDtypeStruct((1,), jnp.int32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=(hbm, hbm),
            out_specs=(hbm, pl.BlockSpec(memory_space=pltpu.SMEM)),
        ),
        interpret=pltpu.InterpretParams(detect_races=True),
    )
    got_shared, got_polls = call(middle_bf16, w_down_bf16)

    np.testing.assert_array_equal(
        np.asarray(got_shared.astype(jnp.float32)),
        np.asarray(expected_bf16.astype(jnp.float32)),
    )
    assert int(got_polls[0]) == 2 * n_panels

    w_perturbed = w_down_bf16.at[:, -panel_cols:].set(-w_down_bf16[:, -panel_cols:])
    perturbed_ref = (
        jnp.matmul(middle_bf16[:rows], w_perturbed, preferred_element_type=jnp.float32)
        + jnp.matmul(
            middle_bf16[rows:], w_perturbed, preferred_element_type=jnp.float32
        )
    ).astype(jnp.bfloat16)
    assert not np.array_equal(
        np.asarray(got_shared.astype(jnp.float32)),
        np.asarray(perturbed_ref.astype(jnp.float32)),
    )


def test_should_match_w4a8_gu_m_tile_128_and_w_cache_reuse_across_tile_boundaries():
    rng = np.random.default_rng(79128)
    sub = 8
    m_tile = w4a8_16k.GU_M_TILE_ROWS
    tm = routed_16k.TILE_M_16K
    tile_k = 3584
    mid_width = 32
    tile_gids = (0, 0, 1, 1)
    offsets = (0, 131, 391, 448, 448)
    total_rows = offsets[-1]

    latent_f32 = rng.normal(0.0, 0.3, size=(total_rows, tile_k)).astype(np.float32)
    latent_f32[0] *= 10.0
    latent_f32[128] *= 4.0
    q_lat, lat_scales = w4a8_16k.quantize_latent_rows_e4m3_ref(jnp.asarray(latent_f32))
    q_lat_2d = q_lat.reshape(total_rows, tile_k)

    fp4_lut = np.array(
        [
            -6.0,
            -4.0,
            -3.0,
            -2.0,
            -1.5,
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
        ],
        dtype=np.float32,
    )
    gate_fp4 = jnp.asarray(
        rng.choice(fp4_lut, size=(2, tile_k, mid_width)), dtype=jnp.bfloat16
    )
    up_fp4 = jnp.asarray(
        rng.choice(fp4_lut, size=(2, tile_k, mid_width)), dtype=jnp.bfloat16
    )
    gate_scales = jnp.asarray(
        rng.integers(112, 123, size=(2, tile_k, mid_width)), dtype=jnp.uint8
    )
    up_scales = jnp.asarray(
        rng.integers(112, 123, size=(2, tile_k, mid_width)), dtype=jnp.uint8
    )

    def _situ_w4a8(act_e4m3, scales_row, w_g_e4m3, w_u_e4m3):
        s_g = (scales_row * jnp.float32(w4a8_16k.GU_POST_SCALE))[:, None]
        s_u = (
            scales_row * jnp.float32(w4a8_16k.GU_POST_SCALE * w4a8_16k.SITU_UP_COEFF)
        )[:, None]
        acc_g = jnp.matmul(act_e4m3, w_g_e4m3, preferred_element_type=jnp.float32) * s_g
        acc_u = jnp.matmul(act_e4m3, w_u_e4m3, preferred_element_type=jnp.float32) * s_u
        gate = 4.0 * jnp.tanh(acc_g * 0.25) * jax.nn.sigmoid(acc_g)
        up = w4a8_16k.SITU_SCALE_UP * jnp.tanh(acc_u)
        return (gate * up).astype(jnp.bfloat16)

    expected_rows = []
    for g, gid in enumerate(tile_gids):
        s0, s1 = offsets[g], offsets[g + 1]
        if s1 == s0:
            continue
        wg = w4a8_16k.dequant_mxfp4_shifted_e4m3_ref(gate_fp4[gid], gate_scales[gid])
        wu = w4a8_16k.dequant_mxfp4_shifted_e4m3_ref(up_fp4[gid], up_scales[gid])
        expected_rows.append(_situ_w4a8(q_lat_2d[s0:s1], lat_scales[s0:s1], wg, wu))
    expected = jnp.concatenate(expected_rows, axis=0)

    def _run_tiled_model(
        *,
        apply_partial: bool = True,
        refresh_on_expert_change: bool = True,
        force_single_m_tile: bool = False,
    ) -> tuple[jax.Array, int]:
        out_hbm = jnp.zeros((total_rows // sub, sub, mid_width), dtype=jnp.bfloat16)
        partial = jnp.zeros((sub, mid_width), dtype=jnp.bfloat16)
        w_cache_g = None
        w_cache_u = None
        dequant_count = 0

        for g, gid in enumerate(tile_gids):
            start, end = offsets[g], offsets[g + 1]
            aligned = start - start % sub
            first, last = start - aligned, end - aligned
            same_prev = g > 0 and gid == tile_gids[g - 1]
            if w_cache_g is None or (refresh_on_expert_change and not same_prev):
                w_cache_g = w4a8_16k.dequant_mxfp4_shifted_e4m3_ref(
                    gate_fp4[gid], gate_scales[gid]
                )
                w_cache_u = w4a8_16k.dequant_mxfp4_shifted_e4m3_ref(
                    up_fp4[gid], up_scales[gid]
                )
                dequant_count += 1

            acts = jnp.full((tm, tile_k), 99.0, dtype=w4a8_16k.LATENT_E4M3_DTYPE)
            t_scales = jnp.full((tm,), 99.0, dtype=jnp.float32)
            if end > start:
                acts = acts.at[first:last].set(q_lat_2d[start:end])
                t_scales = t_scales.at[first:last].set(lat_scales[start:end])

            n_m_tiles = (
                1 if force_single_m_tile else max((last + m_tile - 1) // m_tile, 1)
            )
            tile_out = jnp.zeros((tm // sub, sub, mid_width), dtype=jnp.bfloat16)
            for m_blk in range(n_m_tiles):
                m0 = m_blk * m_tile
                row_idx = m0 + jnp.arange(m_tile)[:, None]
                mask = (first <= row_idx) & (row_idx < last)
                val = _situ_w4a8(
                    acts[m0 : m0 + m_tile],
                    t_scales[m0 : m0 + m_tile],
                    w_cache_g,
                    w_cache_u,
                )
                val = jnp.where(mask, val, jnp.bfloat16(0.0))
                tile_out = tile_out.at[m0 // sub : (m0 + m_tile) // sub].set(
                    val.reshape(m_tile // sub, sub, mid_width)
                )

            if apply_partial and g > 0:
                tile_out = tile_out.at[0].set(tile_out[0] + partial)
            last_row = min(last // sub, (tm // sub) - 1)
            partial = jnp.zeros_like(partial) if last % sub == 0 else tile_out[last_row]
            write_blocks = min(
                (n_m_tiles * m_tile) // sub, (total_rows - aligned) // sub
            )
            if write_blocks > 0:
                out_hbm = out_hbm.at[
                    aligned // sub : aligned // sub + write_blocks
                ].set(tile_out[:write_blocks])

        return out_hbm.reshape(total_rows, mid_width), dequant_count

    got, dequants = _run_tiled_model()
    assert dequants == 2
    np.testing.assert_array_equal(
        np.asarray(got.astype(jnp.float32)),
        np.asarray(expected.astype(jnp.float32)),
    )

    bad_partial, _ = _run_tiled_model(apply_partial=False)
    assert not np.array_equal(
        np.asarray(bad_partial.astype(jnp.float32)),
        np.asarray(expected.astype(jnp.float32)),
    )
    bad_cache, _ = _run_tiled_model(refresh_on_expert_change=False)
    assert not np.array_equal(
        np.asarray(bad_cache.astype(jnp.float32)),
        np.asarray(expected.astype(jnp.float32)),
    )
    bad_mtile, _ = _run_tiled_model(force_single_m_tile=True)
    assert not np.array_equal(
        np.asarray(bad_mtile.astype(jnp.float32)),
        np.asarray(expected.astype(jnp.float32)),
    )
