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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh

from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import config as k3_config
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.config import KernelConfig
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels.prefill_moe import (
    fused,
    routed_16k,
    w4a8_16k,
)


def _situ_f32(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    sig = 1.0 / (1.0 + np.exp(-np.clip(gate, -40.0, 40.0)))
    return (4.0 * np.tanh(gate * 0.25) * sig) * (25.0 * np.tanh(up * 0.04))


def _mxfp8_block32_quant_dequant(x_f32: np.ndarray) -> np.ndarray:
    orig_shape = x_f32.shape
    blocks = x_f32.reshape(-1, 32)
    amax = np.max(np.abs(blocks), axis=1, keepdims=True)
    exp = np.where(
        amax > 0.0, np.ceil(np.log2(np.maximum(amax, 1e-38) / 448.0)), -127.0
    )
    exp = np.clip(exp, -126.0, 127.0)
    scale = np.where(amax > 0.0, np.exp2(exp), 1.0).astype(np.float32)
    q = jnp.clip(jnp.asarray(blocks / scale), -448.0, 448.0).astype(jnp.float8_e4m3fn)
    return (np.asarray(q.astype(jnp.float32)) * scale).reshape(orig_shape)


def test_should_keep_prefill_moe_w4a8_off_when_disabled(monkeypatch):
    monkeypatch.setattr(k3_config, "DEFAULT_PREFILL_MOE_W4A8", False)
    monkeypatch.setattr(k3_config, "DEFAULT_PREFILL_MOE_16K_MONOLITHIC", False)
    cfg = KernelConfig()

    assert cfg.prefill_moe_w4a8 is False
    mono = cfg.replace(prefill_moe_16k_monolithic=True)
    assert mono.key() != mono.replace(prefill_moe_w4a8=True).key()


def test_should_enable_prefill_moe_w4a8_when_env_set(monkeypatch):
    monkeypatch.setattr(k3_config, "DEFAULT_PREFILL_MOE_16K_MONOLITHIC", True)
    monkeypatch.setattr(k3_config, "DEFAULT_PREFILL_MOE_W4A8", True)

    cfg = KernelConfig()
    assert cfg.prefill_moe_16k_monolithic is True
    assert cfg.prefill_moe_w4a8 is True


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"prefill_fused_moe": False}, "prefill_fused_moe"),
        ({"prefill_moe_16k_monolithic": False}, "prefill_moe_16k_monolithic"),
        ({"prefill_expert_parallel": 8}, "prefill_expert_parallel"),
        ({"prefill_expert_parallel": 32}, "prefill_expert_parallel"),
        ({"prefill_moe_dense_fp8": True}, "prefill_moe_dense_fp8"),
    ],
)
def test_should_reject_incompatible_config_when_moe_w4a8_enabled(overrides, match):
    base = {"prefill_moe_16k_monolithic": True, "prefill_moe_w4a8": True}
    base.update(overrides)
    with pytest.raises(ValueError, match=match):
        KernelConfig(**base)


def test_should_quantize_latent_rows_e4m3_across_bos_tiny_rms_zero_and_roundup():
    rng = np.random.default_rng(20261114)
    rows = np.zeros((5, 3584), dtype=np.float32)
    rows[0] = rng.normal(0.03, 0.14, size=(3584,)).astype(np.float32)
    rows[1] = rng.normal(0.0, 0.08, size=(3584,)).astype(np.float32)
    rows[1, 17] = 61.5
    rows[1, 1024] = -48.25
    rows[2] = rng.normal(0.0, 1e-4, size=(3584,)).astype(np.float32)
    rows[3] = 0.0
    rows[4] = np.float32(447.0)
    rows[4, 0] = np.float32(448.0 * (1.0 - 1e-4))

    q_e4m3, scales_f32 = w4a8_16k.quantize_latent_rows_e4m3_ref(jnp.asarray(rows))

    assert q_e4m3.dtype == w4a8_16k.LATENT_E4M3_DTYPE
    assert q_e4m3.shape == (5, *w4a8_16k.LATENT_E4M3_TAIL_SHAPE)
    assert scales_f32.dtype == jnp.float32
    assert scales_f32.shape == (5,)
    assert bool(jnp.all(jnp.isfinite(scales_f32)))
    assert bool(jnp.all(scales_f32 > 0.0))
    assert bool(jnp.all(jnp.isfinite(q_e4m3.astype(jnp.float32))))

    recon = (
        np.asarray(q_e4m3.astype(jnp.float32)).reshape(5, 3584)
        * np.asarray(scales_f32)[:, None]
    )
    for row_idx in (0, 1, 2, 4):
        rel_l2 = float(
            np.linalg.norm(recon[row_idx] - rows[row_idx])
            / np.linalg.norm(rows[row_idx])
        )
        assert rel_l2 < 0.035, f"row {row_idx} rel_l2={rel_l2}"
    assert np.all(recon[3] == 0.0)
    assert float(scales_f32[3]) == pytest.approx(w4a8_16k.FP8_SCALE_EPS)

    unscaled = np.asarray(q_e4m3.astype(jnp.float32)).reshape(5, 3584)
    pos_ctrl_tiny = float(
        np.linalg.norm(unscaled[2] - rows[2]) / np.linalg.norm(rows[2])
    )
    assert pos_ctrl_tiny > 1e4


def test_should_dequantize_mxfp4_shifted_e8m0_losslessly_across_k3_scale_range():
    fp4_mag = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    fp4_all = np.concatenate((-fp4_mag, fp4_mag))
    k3_exps = np.array([0, *range(109, w4a8_16k.GU_MAX_SAFE_E8M0 + 1)], dtype=np.uint8)
    w_grid, e_grid = np.meshgrid(fp4_all, k3_exps, indexing="ij")

    q_e4m3 = w4a8_16k.dequant_mxfp4_shifted_e4m3_ref(
        jnp.asarray(w_grid, dtype=jnp.bfloat16), jnp.asarray(e_grid, dtype=jnp.uint8)
    )
    assert q_e4m3.dtype == w4a8_16k.LATENT_E4M3_DTYPE
    q_f32 = np.asarray(q_e4m3.astype(jnp.float32))
    assert np.all(np.isfinite(q_f32))

    reconstructed = q_f32 * w4a8_16k.GU_POST_SCALE
    expected = np.where(
        e_grid == 0,
        0.0,
        w_grid * np.exp2(e_grid.astype(np.float32) - 127.0),
    ).astype(np.float32)
    np.testing.assert_array_equal(reconstructed, expected)

    unshifted = np.asarray(
        w4a8_16k.dequant_mxfp4_shifted_e4m3_ref(
            jnp.asarray(w_grid, dtype=jnp.bfloat16),
            jnp.asarray(e_grid, dtype=jnp.uint8),
            exp_shift=0,
        ).astype(jnp.float32)
    )
    low_exp_mask = (e_grid >= 109) & (e_grid <= 114) & (w_grid != 0.0)
    assert np.all(unshifted[low_exp_mask] == 0.0)

    overflow_e4m3 = w4a8_16k.dequant_mxfp4_shifted_e4m3_ref(
        jnp.asarray([6.0], dtype=jnp.bfloat16),
        jnp.asarray([w4a8_16k.GU_MAX_SAFE_E8M0 + 1], dtype=jnp.uint8),
    )
    assert bool(jnp.isnan(overflow_e4m3.astype(jnp.float32))[0])


def test_should_match_fp32_and_mxfp8_ref_on_in_situ_bf16_middle_contract():
    rng = np.random.default_rng(314159)
    n_rows, k_dim, mid_dim = 4, 3584, 128
    lat = np.zeros((n_rows, k_dim), dtype=np.float32)
    lat[0] = rng.normal(0.0, 0.14, size=(k_dim,)).astype(np.float32)
    lat[1] = rng.normal(0.0, 0.08, size=(k_dim,)).astype(np.float32)
    lat[1, 31] = 61.5
    lat[2] = rng.normal(0.0, 1e-4, size=(k_dim,)).astype(np.float32)
    lat[3] = 0.0

    fp4_lut = np.array(
        [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32
    )
    w_gate_fp4 = rng.choice(fp4_lut, size=(k_dim, mid_dim)).astype(np.float32)
    w_up_fp4 = rng.choice(fp4_lut, size=(k_dim, mid_dim)).astype(np.float32)
    w_down_fp4 = rng.choice(fp4_lut, size=(mid_dim, 256)).astype(np.float32)
    e_gate = rng.integers(109, 124, size=(k_dim // 32, mid_dim), dtype=np.uint8)
    e_up = rng.integers(109, 124, size=(k_dim // 32, mid_dim), dtype=np.uint8)
    e_down = rng.integers(109, 124, size=(mid_dim // 32, 256), dtype=np.uint8)
    e_gate[0, :4] = 0

    s_gate_true = np.repeat(
        np.where(e_gate == 0, 0.0, np.exp2(e_gate.astype(np.float32) - 127.0)),
        32,
        axis=0,
    )
    s_up_true = np.repeat(
        np.where(e_up == 0, 0.0, np.exp2(e_up.astype(np.float32) - 127.0)),
        32,
        axis=0,
    )
    s_down_true = np.repeat(
        np.where(e_down == 0, 0.0, np.exp2(e_down.astype(np.float32) - 127.0)),
        32,
        axis=0,
    )
    w_gate_f32 = w_gate_fp4 * s_gate_true
    w_up_f32 = w_up_fp4 * s_up_true
    w_down_f32 = w_down_fp4 * s_down_true

    ref_out = _situ_f32(lat @ w_gate_f32, lat @ w_up_f32) @ w_down_f32
    lat_mxfp8_ref = _mxfp8_block32_quant_dequant(lat)
    mxfp8_ref_mid = _mxfp8_block32_quant_dequant(
        _situ_f32(lat_mxfp8_ref @ w_gate_f32, lat_mxfp8_ref @ w_up_f32)
    )
    mxfp8_ref_out = mxfp8_ref_mid @ w_down_f32

    q_lat, s_lat = w4a8_16k.quantize_latent_rows_e4m3_ref(jnp.asarray(lat))
    q_lat_2d = np.asarray(q_lat.astype(jnp.float32)).reshape(n_rows, k_dim)
    w_gate_e4m3 = np.asarray(
        w4a8_16k.dequant_mxfp4_shifted_e4m3_ref(
            jnp.asarray(w_gate_fp4, dtype=jnp.bfloat16),
            jnp.asarray(np.repeat(e_gate, 32, axis=0), dtype=jnp.uint8),
        ).astype(jnp.float32)
    )
    w_up_e4m3 = np.asarray(
        w4a8_16k.dequant_mxfp4_shifted_e4m3_ref(
            jnp.asarray(w_up_fp4, dtype=jnp.bfloat16),
            jnp.asarray(np.repeat(e_up, 32, axis=0), dtype=jnp.uint8),
        ).astype(jnp.float32)
    )
    s_gate_row = (np.asarray(s_lat) * w4a8_16k.GU_POST_SCALE)[:, None]
    s_up_row = (np.asarray(s_lat) * (w4a8_16k.GU_POST_SCALE * w4a8_16k.SITU_UP_COEFF))[
        :, None
    ]
    acc_gate = (q_lat_2d @ w_gate_e4m3) * s_gate_row
    gate_act = (
        4.0
        * np.tanh(acc_gate * 0.25)
        * (1.0 / (1.0 + np.exp(-np.clip(acc_gate, -40, 40))))
    )
    up_act = w4a8_16k.SITU_SCALE_UP * np.tanh((q_lat_2d @ w_up_e4m3) * s_up_row)
    w4a8_mid_bf16 = np.asarray(
        jnp.asarray(gate_act * up_act, dtype=jnp.bfloat16), dtype=np.float32
    )
    w4a8_out = w4a8_mid_bf16 @ np.asarray(
        jnp.asarray(w_down_f32, dtype=jnp.bfloat16), dtype=np.float32
    )

    assert np.all(np.isfinite(w4a8_out))
    assert np.all(w4a8_out[3] == 0.0)
    for row_idx in (0, 1, 2):
        rel_w4a8 = float(
            np.linalg.norm(w4a8_out[row_idx] - ref_out[row_idx])
            / np.linalg.norm(ref_out[row_idx])
        )
        rel_mxfp8_ref = float(
            np.linalg.norm(mxfp8_ref_out[row_idx] - ref_out[row_idx])
            / np.linalg.norm(ref_out[row_idx])
        )
        assert rel_w4a8 < 0.06, (
            f"row {row_idx} rel_w4a8={rel_w4a8} vs rel_mxfp8_ref={rel_mxfp8_ref}"
        )
        assert rel_w4a8 <= rel_mxfp8_ref * 1.10, (
            f"row {row_idx} rel_w4a8={rel_w4a8} exceeded rel_mxfp8_ref={rel_mxfp8_ref}"
        )

    bad_gate = (q_lat_2d @ w_gate_e4m3) * np.asarray(s_lat)[:, None]
    bad_out = (
        _situ_f32(bad_gate, (q_lat_2d @ w_up_e4m3) * np.asarray(s_lat)[:, None])
        @ w_down_f32
    )
    assert (
        float(np.linalg.norm(bad_out[0] - ref_out[0]) / np.linalg.norm(ref_out[0]))
        > 0.5
    )


def test_should_match_reference_in_pallas_compute_dynamic_latent_e4m3_interpret():
    rng = np.random.default_rng(707)
    rows, hidden_size = 128, 256
    own_np = rng.normal(0.0, 0.15, size=(rows, hidden_size)).astype(np.float32)
    own_np[0] *= 30.0
    own_np[1] *= 1e-3
    own_np[2] = 0.0
    w_np = rng.normal(0.0, 0.08, size=(hidden_size, 3584)).astype(np.float32)
    own_bf16 = jnp.asarray(own_np, dtype=jnp.bfloat16)
    w_bf16 = jnp.asarray(w_np, dtype=jnp.bfloat16)

    expected_f32 = jnp.matmul(own_bf16, w_bf16, preferred_element_type=jnp.float32)
    expected_q, expected_s = w4a8_16k.quantize_latent_rows_e4m3_ref(expected_f32)

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    def body(own_hbm, w_hbm, q_out, s_out):
        def inner(s):
            c = pltpu.make_async_copy(own_hbm, s["own"], s["in_sem"])
            c.start()
            pltpu.make_async_copy(
                w_hbm.at[:, pl.ds(0, 256)], s["bufs"][0], s["sems"][0]
            ).start()
            pltpu.make_async_copy(
                w_hbm.at[:, pl.ds(256, 256)], s["bufs"][1], s["sems"][1]
            ).start()
            c.wait()
            w4a8_16k.compute_dynamic_latent_e4m3(
                s["own"],
                w_hbm,
                s["bufs"],
                s["sems"],
                s["lat_q"],
                s["lat_s"],
                prestarted=True,
            )
            cq = pltpu.make_async_copy(s["lat_q"], q_out, s["out_sems"][0])
            cs = pltpu.make_async_copy(s["lat_s"], s_out, s["out_sems"][1])
            cq.start()
            cs.start()
            cq.wait()
            cs.wait()

        pl.run_scoped(
            inner,
            dict(
                own=pltpu.VMEM((rows, hidden_size), jnp.bfloat16),
                bufs=tuple(
                    pltpu.VMEM((hidden_size, 256), jnp.bfloat16) for _ in range(4)
                ),
                sems=tuple(pltpu.SemaphoreType.DMA for _ in range(4)),
                in_sem=pltpu.SemaphoreType.DMA,
                out_sems=(pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA),
                lat_q=pltpu.VMEM(
                    (rows, *w4a8_16k.LATENT_E4M3_TAIL_SHAPE), w4a8_16k.LATENT_E4M3_DTYPE
                ),
                lat_s=pltpu.VMEM((rows // 128, 128), jnp.float32),
            ),
        )

    call = pl.pallas_call(
        body,
        out_shape=(
            jax.ShapeDtypeStruct(
                (rows, *w4a8_16k.LATENT_E4M3_TAIL_SHAPE), w4a8_16k.LATENT_E4M3_DTYPE
            ),
            jax.ShapeDtypeStruct((rows // 128, 128), jnp.float32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0, in_specs=(hbm, hbm), out_specs=(hbm, hbm)
        ),
        interpret=pltpu.InterpretParams(detect_races=True),
    )
    got_q, got_s = call(own_bf16, w_bf16)

    np.testing.assert_allclose(
        np.asarray(got_s).reshape(-1), np.asarray(expected_s), rtol=1e-6, atol=0.0
    )
    np.testing.assert_array_equal(
        np.asarray(got_q.astype(jnp.float32)),
        np.asarray(expected_q.astype(jnp.float32)),
    )


def test_should_enforce_w4a8_layout_alignment_and_hbm_vmem_savings():
    tokens = routed_16k.TOKENS_16K
    tm = routed_16k.TILE_M_16K
    n_banks = routed_16k.GU_ACT_BANKS_16K

    assert (
        w4a8_16k.LATENT_E4M3_TAIL_SHAPE[0] * w4a8_16k.LATENT_E4M3_TAIL_SHAPE[1]
        == routed_16k.OUTPUT_WIDTH_16K
    )
    assert tm % 128 == 0
    assert (n_banks * tm) % 128 == 0
    assert w4a8_16k.LATENT_SCALES_SHAPE[0] * w4a8_16k.LATENT_SCALES_SHAPE[1] == tokens

    bf16_latent_bytes = tokens * 2 * 1792 * 2
    w4a8_latent_bytes = (
        tokens
        * w4a8_16k.LATENT_E4M3_TAIL_SHAPE[0]
        * w4a8_16k.LATENT_E4M3_TAIL_SHAPE[1]
        * 1
    )
    w4a8_scales_bytes = (
        w4a8_16k.LATENT_SCALES_SHAPE[0] * w4a8_16k.LATENT_SCALES_SHAPE[1] * 4
    )
    assert bf16_latent_bytes - (w4a8_latent_bytes + w4a8_scales_bytes) > 58_000_000

    rows = tokens // 32
    prologue_peak_bytes = (
        (2 * rows * 7168 * 2)
        + (rows * 8 * 128 * 4)
        + (rows * 4 * 128 * 4)
        + (rows * 4 * 896 * 1)
        + (2 * rows * 3072 * 2)
        + (7168 * 1024 * 2)
        + (rows * 4 * 896 * 2)
        + w4a8_scales_bytes
    )
    assert prologue_peak_bytes < 50 * 1024 * 1024


@pytest.mark.parametrize(
    ("latent_shape", "latent_dtype", "scales_shape", "scales_dtype"),
    [
        ((16384, 4, 896), jnp.bfloat16, (128, 128), jnp.float32),
        ((16384, 2, 1792), jnp.float8_e4m3fn, (128, 128), jnp.float32),
        ((16384, 4, 896), jnp.float8_e4m3fn, (16384,), jnp.float32),
        ((16384, 4, 896), jnp.float8_e4m3fn, (128, 128), jnp.bfloat16),
    ],
)
def test_should_reject_malformed_w4a8_latent_or_scale_specs_in_routed_16k(
    latent_shape, latent_dtype, scales_shape, scales_dtype
):
    latent = jax.ShapeDtypeStruct(latent_shape, latent_dtype)
    gu_weights = jax.ShapeDtypeStruct((1, 56, 448, 3072), jnp.uint32)
    gu_scales = jax.ShapeDtypeStruct((1, 56, 112, 3072), jnp.uint8)
    rhs = jax.ShapeDtypeStruct((1, 56, 192, 3584), jnp.uint32)
    scales = jax.ShapeDtypeStruct((1, 56, 48, 3584), jnp.uint8)
    counts = jax.ShapeDtypeStruct((56,), jnp.int32)
    middle = jax.ShapeDtypeStruct((98304, 1536), jnp.bfloat16)
    output = jax.ShapeDtypeStruct((512, 3584), jnp.bfloat16)
    subtotal = jax.ShapeDtypeStruct(
        (routed_16k.TOKENS_16K, routed_16k.SUBTOTAL_16K_COLS), jnp.bfloat16
    )
    bad_scales = jax.ShapeDtypeStruct(scales_shape, scales_dtype)

    with pytest.raises(AssertionError):
        routed_16k.routed_experts_16k(
            latent,
            gu_weights,
            gu_scales,
            rhs,
            scales,
            counts,
            None,
            None,
            None,
            None,
            None,
            middle,
            output,
            probabilities_ready=lambda: None,
            subtotal_hbm=subtotal,
            compact_capacity=98304,
            rmw_batch=128,
            rmw_dummy_slots=16,
            latent_scales_hbm=bad_scales,
        )


@pytest.mark.parametrize("with_mixture", [False, True])
def test_should_route_w4a8_16k_into_monolithic_fused_builder_requiring_32_ranks(
    monkeypatch, with_mixture
):
    cfg = KernelConfig(
        prefill_expert_parallel=16,
        prefill_moe_16k_monolithic=True,
        prefill_moe_w4a8=True,
    )
    monkeypatch.setattr(k3_config, "active", lambda: cfg)
    one_dev_mesh = Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))

    builder = (
        fused.make_indexed_mixture_moe_sp_ring
        if with_mixture
        else fused.make_indexed_moe_sp_ring
    )
    with pytest.raises(ValueError, match="flat 32-rank k3_sp mesh"):
        builder(one_dev_mesh, seq_len=16384)
