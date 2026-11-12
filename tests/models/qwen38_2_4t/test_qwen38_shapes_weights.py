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
"""Tests for Qwen3.8 shapes, pure-JAX compute, weights, and FP64 oracle."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from tests.models.qwen38_2_4t import mock_model
from tests.models.qwen38_2_4t import oracle_fp64 as oracle
from tests.models.qwen38_2_4t.constants import (
    CUSTOM_W13_BYTE,
    EXPECTED_DCP,
    EXPECTED_KV_HEADS,
    EXPECTED_LAYERS,
    EXPECTED_RANKS,
    EXPECTED_VH,
    EXPECTED_WEIGHT_KEYS,
    ROUTER_WEIGHT_SUM_TOL,
    U8_WEIGHT_KEYS,
)
from vllm_torchtpu.models.qwen38_2_4t.megakernel.compute import (
    dcp_merge_lse,
    dcp_online_softmax_update,
    dequant_nvfp4,
    fp4_matmul,
    gdn_recurrent_step,
    moe_router_topk,
    partial_rope,
    rmsnorm,
    swiglu_split,
)
from vllm_torchtpu.models.qwen38_2_4t.megakernel.shapes import (
    Qwen38Config,
    _validate_config,
    mini_parity_config,
)
from vllm_torchtpu.models.qwen38_2_4t.megakernel.weights import (
    WEIGHT_KEYS,
    _extract_linear_fp4,
    _extract_mlp_sublayer,
    _orient_2d,
    _resolve_expert_tensor,
    _transpose_fp4_scale,
    _transpose_packed_fp4,
    extract_backbone_weights_torch,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _make_tiny_cfg() -> Qwen38Config:
    """Return a compact 2-rank TP2xDCP2 configuration for fast CPU parity checks."""
    return Qwen38Config(
        R=2,
        DCP=2,
        B=1,
        H=32,
        L=4,
        att_every=4,
        Q_H=4,
        KV_H=1,
        D=32,
        rope_dim=16,
        KH=2,
        VH=4,
        KD=32,
        VD=32,
        conv_kernel=4,
        E=4,
        K=2,
        I_E=64,
        SE_I=64,
        fp4_group=16,
        page_size=32,
        max_pages_per_seq=2,
        num_pages=64,
        hier_group=2,
        h_tile=32,
    )


def test_qwen38_config_and_validation() -> None:
    """Verify Qwen38Config geometry properties, mini_parity_config, and validation."""
    cfg = Qwen38Config()
    _validate_config(cfg)
    assert cfg.R == EXPECTED_RANKS
    assert cfg.DCP == EXPECTED_DCP
    assert cfg.L == EXPECTED_LAYERS
    assert cfg.KV_H == EXPECTED_KV_HEADS
    assert cfg.VH == EXPECTED_VH

    mini_cfg = mini_parity_config()
    _validate_config(mini_cfg)
    assert mini_cfg.NATT + mini_cfg.NGDN == mini_cfg.L


@pytest.mark.parametrize(
    ("overrides", "err_match"),
    [
        ({"R": 0}, "Invalid R="),
        ({"B": 0}, "Invalid B="),
        ({"K": 16, "E": 8}, "Invalid MoE routing"),
        ({"num_pages": 2, "max_pages_per_seq": 8}, "Invalid pages"),
        ({"R": 8, "DCP": 4, "Q_H": 16, "KV_H": 8}, r"Q_H=.*must be divisible"),
        ({"rope_dim": 63}, r"rope_dim=.*must be positive, even, and <= D"),
        ({"rope_dim": 512}, r"rope_dim=.*must be positive, even, and <= D"),
        ({"KH": 48, "R": 32}, r"KH=.*must divide or be divisible by R"),
        ({"R": 8, "DCP": 8, "KH": 16, "VH": 24}, "Incompatible GDN head counts"),
    ],
)
def test_qwen38_config_validation_errors(
    overrides: dict[str, int],
    err_match: str,
) -> None:
    """Reject invalid sharding, head counts, and page bounds cleanly."""
    with pytest.raises(ValueError, match=err_match):
        Qwen38Config(**overrides)


def test_qwen38_weights_keys() -> None:
    """Verify WEIGHT_KEYS registry and fail-closed validation."""
    assert len(WEIGHT_KEYS) == EXPECTED_WEIGHT_KEYS
    assert "wqkv" in WEIGHT_KEYS
    assert "in_norm" in WEIGHT_KEYS

    with pytest.raises(ValueError, match="Cannot locate"):
        extract_backbone_weights_torch(object(), mini_parity_config())

    tiny_cfg = _make_tiny_cfg()
    bad_mlp = mock_model.MockBadMlp(
        gate=mock_model.MockWeightHolder(weight=torch.ones((tiny_cfg.E, tiny_cfg.H))),
        shared_expert_gate=None,
    )
    untyped_extract_mlp = cast("Callable[..., object]", _extract_mlp_sublayer)
    with pytest.raises(
        ValueError, match=r"Layer 0 MLP is missing required shared_expert_gate\.weight"
    ):
        untyped_extract_mlp(bad_mlp, 0, tiny_cfg)

    with pytest.raises(
        ValueError, match="Layer 0 experts is missing required attribute 'w13_weight'"
    ):
        _resolve_expert_tensor(
            mock_model.MockEmptyExperts(),
            0,
            "w13_weight",
            required=True,
        )

    with pytest.raises(ValueError, match="missing required attribute 'weight'"):
        _extract_linear_fp4(
            mock_model.MockLinearProj(
                weight=cast("torch.Tensor", None),
                weight_scale=cast("torch.Tensor", None),
            ),
            32,
            32,
        )


def test_qwen38_extract_rank_backbones_resolves_dotted_attrs() -> None:
    """Resolve dotted expert submodules alongside standard expert attributes."""
    tiny_cfg = _make_tiny_cfg()
    prob = oracle.make_synthetic_problem(tiny_cfg, seed=5, pos=(0,))
    mock_models = mock_model.build_mock_vllm_qwen38_rank_models(prob, tiny_cfg)
    b0, b1 = mock_models.rank_backbones[0], mock_models.rank_backbones[1]
    ex0 = cast("mock_model.MockExperts", b0.layers[0].mlp.experts)
    ex1 = cast("mock_model.MockExperts", b1.layers[0].mlp.experts)

    expected_rank0_byte = CUSTOM_W13_BYTE
    expected_rank1_byte = 0x11
    sentinel_w2_byte = 0x22
    ex0.w13_weight = torch.full_like(ex0.w13_weight, expected_rank0_byte)
    sentinel_w13 = torch.full_like(ex1.w13_weight, expected_rank1_byte)
    sentinel_w2 = torch.full_like(ex1.w2_weight, sentinel_w2_byte)
    b1.layers[0].mlp.experts = mock_model.MockDottedExperts(
        gate_up_proj=mock_model.MockLinearProj(
            weight=sentinel_w13, weight_scale=ex1.w13_weight_scale
        ),
        down_proj=mock_model.MockLinearProj(
            weight=sentinel_w2, weight_scale=ex1.w2_weight_scale
        ),
    )

    extracted = extract_backbone_weights_torch(
        object(), tiny_cfg, rank_backbones=[b0, b1]
    )
    assert torch.all(extracted["w_e13"][0, 0] == expected_rank0_byte)
    assert torch.all(extracted["w_e13"][1, 0] == expected_rank1_byte)
    assert torch.all(extracted["w_e2"][1, 0] == sentinel_w2_byte)


def test_qwen38_extract_global_unsharded_experts_by_tp_rank() -> None:
    """Slice global unsharded expert weights on the module per `tp_rank`."""
    tiny_cfg = _make_tiny_cfg()
    prob = oracle.make_synthetic_problem(tiny_cfg, seed=19, pos=(0,))
    mock_models = mock_model.build_mock_vllm_qwen38_rank_models(prob, tiny_cfg)
    b0, b1 = mock_models.rank_backbones[0], mock_models.rank_backbones[1]

    for l_idx in range(tiny_cfg.L):
        ex0 = cast("mock_model.MockExperts", b0.layers[l_idx].mlp.experts)
        ex1 = cast("mock_model.MockExperts", b1.layers[l_idx].mlp.experts)
        half_r = tiny_cfg.I_R
        global_w13 = torch.cat(
            [
                ex0.w13_weight[:, :half_r, :],
                ex1.w13_weight[:, :half_r, :],
                ex0.w13_weight[:, half_r:, :],
                ex1.w13_weight[:, half_r:, :],
            ],
            dim=1,
        )
        global_s13 = torch.cat(
            [
                ex0.w13_weight_scale[:, :half_r, :],
                ex1.w13_weight_scale[:, :half_r, :],
                ex0.w13_weight_scale[:, half_r:, :],
                ex1.w13_weight_scale[:, half_r:, :],
            ],
            dim=1,
        )
        global_w2 = torch.cat([ex0.w2_weight, ex1.w2_weight], dim=2)
        global_s2 = torch.cat([ex0.w2_weight_scale, ex1.w2_weight_scale], dim=2)

        global_experts = mock_model.MockExperts(
            w13_weight=global_w13,
            w13_weight_scale=global_s13,
            w2_weight=global_w2,
            w2_weight_scale=global_s2,
        )
        b0.layers[l_idx].mlp.experts = global_experts
        b1.layers[l_idx].mlp.experts = global_experts

    got_r0 = extract_backbone_weights_torch(b0, tiny_cfg, tp_rank=0)
    got_r1 = extract_backbone_weights_torch(b1, tiny_cfg, tp_rank=1)
    ref_w = {k: np.asarray(jax.device_get(v)) for k, v in prob["weights"].items()}
    for key in ("w_e13", "w_e2"):
        np.testing.assert_array_equal(got_r0[key][0].numpy(), ref_w[key][0])
        np.testing.assert_array_equal(got_r1[key][0].numpy(), ref_w[key][1])
    for key in ("w_e13_s", "w_e2_s"):
        np.testing.assert_allclose(
            got_r0[key][0].float().numpy(),
            ref_w[key][0].astype(np.float32),
            atol=1e-3,
            rtol=1e-3,
        )
        np.testing.assert_allclose(
            got_r1[key][0].float().numpy(),
            ref_w[key][1].astype(np.float32),
            atol=1e-3,
            rtol=1e-3,
        )


def test_compute_rmsnorm_and_rope_parity() -> None:
    """Verify RMSNorm and partial RoPE compute helpers against the FP64 oracle."""
    cfg = _make_tiny_cfg()
    prob = oracle.make_synthetic_problem(cfg, seed=13, pos=(7,))
    x_bf16 = prob["x"][0]
    w_in = prob["weights"]["in_norm"][0, 0][None, :]
    rms_got = np.asarray(rmsnorm(x_bf16, w_in).astype(jnp.float32), dtype=np.float64)
    rms_ref = oracle.rmsnorm_f64(
        np.asarray(x_bf16.astype(jnp.float32), dtype=np.float64),
        np.asarray(w_in.astype(jnp.float32), dtype=np.float64),
    )
    bound_rms = oracle.layer_bf16_error_bound(float(np.sqrt(np.mean(rms_ref**2))), 1, 1)
    assert float(np.sqrt(np.mean((rms_got - rms_ref) ** 2))) <= bound_rms

    rng = np.random.default_rng(13)
    q_np = oracle.bf16_f64(
        rng.normal(0.0, 0.4, size=(cfg.B, cfg.Q_R, cfg.D)).astype(np.float32)
    )
    pos_jnp = prob["pos"][0]
    rope_got = np.asarray(
        partial_rope(
            jnp.asarray(q_np, dtype=jnp.bfloat16), pos_jnp, cfg.rope_dim
        ).astype(jnp.float32),
        dtype=np.float64,
    )
    rope_ref = oracle.rope_f64(q_np, np.asarray(pos_jnp, dtype=np.int32), cfg.rope_dim)
    bound_rope = oracle.layer_bf16_error_bound(
        float(np.sqrt(np.mean(rope_ref**2))), 1, 1
    )
    assert float(np.sqrt(np.mean((rope_got - rope_ref) ** 2))) <= bound_rope


def test_compute_nvfp4_dequant_and_matmul_parity() -> None:
    """Verify NVFP4 dequantization, matmul parity, and nibble-flip sensitivity."""
    cfg = _make_tiny_cfg()
    prob = oracle.make_synthetic_problem(cfg, seed=13, pos=(7,))
    x_bf16 = prob["x"][0]
    w_in = prob["weights"]["in_norm"][0, 0][None, :]
    rms_ref = oracle.rmsnorm_f64(
        np.asarray(x_bf16.astype(jnp.float32), dtype=np.float64),
        np.asarray(w_in.astype(jnp.float32), dtype=np.float64),
    )
    wqkv_u8 = prob["weights"]["wqkv"][0, 0]
    wqkv_s = prob["weights"]["wqkv_s"][0, 0]
    w_deq = np.asarray(
        dequant_nvfp4(wqkv_u8, wqkv_s).astype(jnp.float32), dtype=np.float64
    )
    w_ref = oracle.decode_fp4_f64(
        np.asarray(wqkv_u8, dtype=np.uint8),
        np.asarray(wqkv_s.astype(jnp.float32), dtype=np.float32),
    )
    np.testing.assert_allclose(w_deq, w_ref, atol=1e-4, rtol=1e-4)

    mm_got = np.asarray(
        fp4_matmul(jnp.asarray(rms_ref, dtype=jnp.bfloat16), wqkv_u8, wqkv_s),
        dtype=np.float64,
    )
    mm_ref = rms_ref @ w_ref
    bound_mm = oracle.layer_bf16_error_bound(float(np.sqrt(np.mean(mm_ref**2))), 1, 1)
    assert float(np.sqrt(np.mean((mm_got - mm_ref) ** 2))) <= bound_mm

    w_u8_np = np.asarray(wqkv_u8, dtype=np.uint8)
    flipped = jnp.asarray(
        ((w_u8_np & 0x0F) << 4) | ((w_u8_np >> 4) & 0x0F), dtype=jnp.uint8
    )
    mm_flip = np.asarray(
        fp4_matmul(jnp.asarray(rms_ref, dtype=jnp.bfloat16), flipped, wqkv_s),
        dtype=np.float64,
    )
    rmse_flip = float(np.sqrt(np.mean((mm_flip - mm_ref) ** 2)))
    assert rmse_flip > float(np.sqrt(np.mean((mm_got - mm_ref) ** 2))) * 10.0


def test_compute_dcp_online_softmax_and_lse_merge() -> None:
    """Verify DCP online softmax + LSE merge parity, including FP8 NaN masking."""
    cfg = _make_tiny_cfg()
    prob = oracle.make_synthetic_problem(cfg, seed=13, pos=(3,))
    pos_jnp = prob["pos"][0]
    pool_f64 = np.repeat(
        np.asarray(prob["pool"].astype(jnp.float32), dtype=np.float64)[:, None, ...],
        cfg.NATT,
        axis=1,
    )
    attn_state = oracle.AttnRefState(
        pos=np.asarray(pos_jnp, dtype=np.int32),
        pt=np.asarray(prob["page_table"], dtype=np.int32),
        pool=pool_f64,
    )
    rng = np.random.default_rng(13)
    q_grp_np = oracle.bf16_f64(
        rng.normal(0.0, 0.3, size=(cfg.B, cfg.DCP * cfg.Q_R, cfg.D)).astype(np.float32)
    )
    attn_ref = oracle.compute_dcp_group_attention(
        q_grp_np, r_base=0, state=attn_state, att_idx=0, cfg=cfg
    )

    dcp_q = cfg.DCP * cfg.Q_R
    lse_ranks = []
    for d_idx in range(cfg.DCP):
        m_st = jnp.full((dcp_q,), -1e30, dtype=jnp.float32)
        l_st = jnp.zeros((dcp_q,), dtype=jnp.float32)
        acc_st = jnp.zeros((dcp_q, cfg.D), dtype=jnp.float32)
        pb = jnp.asarray(int(attn_state.pos[0]), dtype=jnp.int32)
        for lp in range(cfg.max_pages_per_seq):
            p_idx = int(attn_state.pt[d_idx, 0, 0, lp])
            k_pg = prob["pool"][d_idx, p_idx, 0]
            v_pg = prob["pool"][d_idx, p_idx, 1]
            if lp == 0:
                v_u8 = v_pg.view(jnp.uint8).at[:, :, -1].set(jnp.uint8(0x7F))
                v_pg = v_u8.view(jnp.float8_e4m3fn)
            tok_pos = jnp.asarray(
                (lp * cfg.DCP + d_idx) * cfg.page_size
                + np.arange(cfg.page_size, dtype=np.int32),
                dtype=jnp.int32,
            )
            m_st, l_st, acc_st = dcp_online_softmax_update(
                jnp.asarray(q_grp_np[0], dtype=jnp.bfloat16),
                (k_pg, v_pg),
                (pb, tok_pos),
                (m_st, l_st, acc_st),
                cfg,
            )
        lse_ranks.append(
            jnp.concatenate([acc_st, m_st[:, None], l_st[:, None]], axis=-1)[None, ...]
        )

    q_gate_np = oracle.bf16_f64(
        rng.normal(0.0, 0.2, size=(cfg.B, cfg.Q_R, cfg.D)).astype(np.float32)
    )
    merged_got = np.asarray(
        dcp_merge_lse(
            jnp.stack(lse_ranks, axis=0),
            jnp.asarray(q_gate_np, dtype=jnp.bfloat16),
            dcp_slot=0,
            cfg=cfg,
        ).astype(jnp.float32),
        dtype=np.float64,
    )
    assert not np.isnan(merged_got).any()
    o_slice_ref = oracle.bf16_f64(attn_ref[:, : cfg.Q_R, :])
    merged_ref = oracle.bf16_f64(o_slice_ref * oracle.sig_f64(q_gate_np)).reshape(
        cfg.B, cfg.Q_R * cfg.D
    )
    np.testing.assert_allclose(merged_got, merged_ref, atol=2e-2, rtol=2e-2)


def test_compute_gdn_and_moe_parity() -> None:
    """Verify GDN recurrent step, MoE router, SwiGLU split, and ULP metrics."""
    cfg = _make_tiny_cfg()
    prob = oracle.make_synthetic_problem(cfg, seed=13, pos=(7,))
    w = {k: np.asarray(jax.device_get(v)) for k, v in prob["weights"].items()}
    rms_ref = oracle.rmsnorm_f64(
        np.asarray(prob["x"][0].astype(jnp.float32), dtype=np.float64),
        np.asarray(w["in_norm"][0, 0][None, :].astype(np.float32), dtype=np.float64),
    )

    w_in = oracle.decode_fp4_f64(w["w_inproj"][0, 0], w["w_inproj_s"][0, 0])
    qz_np = rms_ref @ w_in
    y_gdn, conv_next, rec_next = gdn_recurrent_step(
        (
            jnp.asarray(qz_np[:, : cfg.QKV], dtype=jnp.bfloat16),
            jnp.asarray(qz_np[:, cfg.QKV :], dtype=jnp.bfloat16),
            jnp.asarray(rms_ref @ oracle.bf16_f64(w["w_ba"][0, 0]), dtype=jnp.float32),
        ),
        (prob["conv"][0, 0], prob["rec"][0, 0]),
        (
            prob["weights"]["w_conv"][0, 0],
            prob["weights"]["a_log"][0, 0],
            prob["weights"]["dt_bias"][0, 0],
            prob["weights"]["gdn_norm"][0, 0][None, :],
        ),
        cfg,
    )
    gdn_state = oracle.GdnRefState(
        conv=np.asarray(prob["conv"].astype(jnp.float32), dtype=np.float64),
        rec=np.asarray(prob["rec"], dtype=np.float64),
    )
    _ = oracle.ref_gdn_step(rms_ref, state=gdn_state, w=w, g_idx=0, cfg=cfg)
    assert y_gdn.shape == (cfg.B, cfg.Z_R)
    np.testing.assert_allclose(
        np.asarray(conv_next.astype(jnp.float32), dtype=np.float64),
        gdn_state.conv[0, 0],
        atol=1e-3,
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(rec_next, dtype=np.float64),
        gdn_state.rec[0, 0],
        atol=1e-3,
        rtol=1e-3,
    )

    x_norm_jnp = jnp.asarray(rms_ref, dtype=jnp.bfloat16)
    forced_jnp = prob["forced_top_ids"][0, 0]
    ids_f, w_f = moe_router_topk(
        x_norm_jnp, prob["weights"]["w_gate"][0, 0], cfg, forced_ids=forced_jnp
    )
    np.testing.assert_array_equal(np.asarray(ids_f), np.asarray(forced_jnp))
    assert float(jnp.max(jnp.abs(jnp.sum(w_f, axis=-1) - 1.0))) <= ROUTER_WEIGHT_SUM_TOL

    ids_u, w_u = moe_router_topk(
        x_norm_jnp, prob["weights"]["w_gate"][0, 0], cfg, forced_ids=None
    )
    logits_ref = rms_ref @ oracle.bf16_f64(w["w_gate"][0, 0])
    top_idx_expected = np.argsort(-logits_ref, axis=-1, kind="stable")[:, : cfg.K]
    np.testing.assert_array_equal(np.asarray(ids_u), top_idx_expected)
    assert float(jnp.max(jnp.abs(jnp.sum(w_u, axis=-1) - 1.0))) <= ROUTER_WEIGHT_SUM_TOL

    rng = np.random.default_rng(13)
    gu_np = rng.normal(0.0, 0.5, size=(cfg.B, 2 * cfg.SE_R)).astype(np.float32)
    sw_got = np.asarray(
        swiglu_split(jnp.asarray(gu_np), cfg.SE_R).astype(jnp.float32), dtype=np.float64
    )
    sw_ref = oracle.bf16_f64(
        oracle.silu_f64(gu_np[:, : cfg.SE_R]) * gu_np[:, cfg.SE_R :]
    )
    np.testing.assert_allclose(sw_got, sw_ref, atol=1e-3, rtol=1e-3)

    fp8_sample = jnp.asarray(
        np.array([1.0, -2.5], dtype=np.float32), dtype=jnp.float8_e4m3fn
    )
    assert oracle.max_fp8_ulp_diff(fp8_sample, fp8_sample) == 0


def test_qwen38_oracle_bf16_cast_and_weight_extraction() -> None:
    """Verify non-exact BF16 RNE rounding and full 28-key weight round-trip."""
    val = np.array(
        [
            1.0 + 2.0**-8,
            1.0 + 3.0 * 2.0**-8,
            1.0 + 3.0 * 2.0**-9,
            -1.0 - 3.0 * 2.0**-8,
            1.0 / 3.0,
        ],
        dtype=np.float64,
    )
    out = oracle.bf16_f64(val)
    assert not np.array_equal(out, val)
    expected = (
        torch.tensor(val, dtype=torch.float32)
        .to(torch.bfloat16)
        .to(torch.float64)
        .numpy()
    )
    np.testing.assert_array_equal(out, expected)

    tiny_cfg = _make_tiny_cfg()
    prob = oracle.make_synthetic_problem(tiny_cfg, seed=0, pos=(0,))
    ref = oracle.run_fp64_reference(prob, tiny_cfg)
    assert ref["x"].shape == (tiny_cfg.B, tiny_cfg.H)
    mock_models = mock_model.build_mock_vllm_qwen38_rank_models(prob, tiny_cfg)

    for extracted, ref_slice in (
        (
            extract_backbone_weights_torch(mock_models.model, tiny_cfg),
            lambda k: np.asarray(jax.device_get(prob["weights"][k]))[:1],
        ),
        (
            extract_backbone_weights_torch(
                mock_models, tiny_cfg, rank_backbones=mock_models.rank_backbones
            ),
            lambda k: np.asarray(jax.device_get(prob["weights"][k])),
        ),
    ):
        for k in WEIGHT_KEYS:
            got_arr = extracted[k].float().numpy()
            ref_arr = ref_slice(k)
            assert got_arr.shape == ref_arr.shape, f"Shape mismatch on {k}"
            if k in U8_WEIGHT_KEYS:
                np.testing.assert_array_equal(
                    extracted[k].numpy(), ref_arr.astype(np.uint8)
                )
            else:
                np.testing.assert_allclose(
                    got_arr, ref_arr.astype(np.float32), atol=1e-3, rtol=1e-3
                )

    orig_wqkv = np.asarray(
        jax.device_get(prob["weights"]["wqkv"][0, 0]), dtype=np.uint8
    )
    pk_nk, _ = mock_model.pack_rows_fp4(
        orig_wqkv,
        np.asarray(jax.device_get(prob["weights"]["wqkv_s"][0, 0]), dtype=np.float32),
    )
    swapped_nk = ((pk_nk.numpy() & 0x0F) << 4) | ((pk_nk.numpy() >> 4) & 0x0F)
    unpacked_swapped = _transpose_packed_fp4(swapped_nk, tiny_cfg.H, tiny_cfg.ATT_QKV_R)
    assert not np.array_equal(unpacked_swapped, orig_wqkv)


@pytest.mark.parametrize("prefix", [(), (2,)])
def test_qwen38_square_transpose_prefers_raw_layout(prefix: tuple[int, ...]) -> None:
    """Square FP4 scale and 2D matrices transpose from raw `(N, G)` / `(cols, rows)`."""
    n_dim = 4
    k_dim = n_dim * 16
    scale_raw = np.arange(int(np.prod((*prefix, n_dim, n_dim))), dtype=np.float32)
    scale_raw = scale_raw.reshape(*prefix, n_dim, n_dim)
    scale_got = _transpose_fp4_scale(scale_raw, k_dim=k_dim, n_dim=n_dim, group_size=16)
    np.testing.assert_array_equal(scale_got, scale_raw.swapaxes(-2, -1))
    if not prefix:
        mat_raw = np.arange(n_dim * n_dim, dtype=np.float32).reshape(n_dim, n_dim)
        mat_got = _orient_2d(mat_raw, n_dim, n_dim, name="square")
        np.testing.assert_array_equal(mat_got, mat_raw.T)


def test_qwen38_extract_post_load_4d_scales_and_kmajor_fp4() -> None:
    """Extract post-load 4D fused scales and K-major FP4 weights."""
    tiny_cfg = _make_tiny_cfg()
    prob = oracle.make_synthetic_problem(tiny_cfg, seed=23, pos=(0,))
    mock_models = mock_model.build_mock_vllm_qwen38_rank_models(prob, tiny_cfg)
    backbone = mock_models.rank_backbones[0]
    ref_w = {k: np.asarray(jax.device_get(v)) for k, v in prob["weights"].items()}

    # Simulate VllmNvFp4LinearMethod and VllmNvFp4MoEMethod post-load tensors:
    # K-major [..., K, N // 2] uint8 weights and 4D [1|E, G, 1, N] float32 scales
    # with weight_scale_2 already fused and deleted.
    attn_idx = 0
    gdn_idx = 0
    for l_idx, layer in enumerate(backbone.layers):
        if (l_idx + 1) % tiny_cfg.att_every == 0:
            sa = layer.self_attn
            sa.qkv_proj = mock_model.MockLinearProj(
                weight=torch.from_numpy(ref_w["wqkv"][0, attn_idx].copy()),
                weight_scale=torch.from_numpy(
                    ref_w["wqkv_s"][0, attn_idx][None, :, None, :].astype(np.float32)
                ),
            )
            sa.o_proj = mock_model.MockLinearProj(
                weight=torch.from_numpy(ref_w["wo"][0, attn_idx].copy()),
                weight_scale=torch.from_numpy(
                    ref_w["wo_s"][0, attn_idx][None, :, None, :].astype(np.float32)
                ),
            )
            attn_idx += 1
        else:
            la = layer.linear_attn
            la.in_proj_qkvz = mock_model.MockLinearProj(
                weight=torch.from_numpy(ref_w["w_inproj"][0, gdn_idx].copy()),
                weight_scale=torch.from_numpy(
                    ref_w["w_inproj_s"][0, gdn_idx][None, :, None, :].astype(np.float32)
                ),
            )
            la.out_proj = mock_model.MockLinearProj(
                weight=torch.from_numpy(ref_w["w_out"][0, gdn_idx].copy()),
                weight_scale=torch.from_numpy(
                    ref_w["w_out_s"][0, gdn_idx][None, :, None, :].astype(np.float32)
                ),
            )
            gdn_idx += 1

        layer.mlp.experts = mock_model.MockExperts(
            w13_weight=torch.from_numpy(ref_w["w_e13"][0, l_idx].copy()),
            w13_weight_scale=torch.from_numpy(
                ref_w["w_e13_s"][0, l_idx][:, :, None, :].astype(np.float32)
            ),
            w2_weight=torch.from_numpy(ref_w["w_e2"][0, l_idx].copy()),
            w2_weight_scale=torch.from_numpy(
                ref_w["w_e2_s"][0, l_idx][:, :, None, :].astype(np.float32)
            ),
        )

    got = extract_backbone_weights_torch(backbone, tiny_cfg, tp_rank=0)
    for key in ("wqkv", "wo", "w_inproj", "w_out", "w_e13", "w_e2"):
        np.testing.assert_array_equal(got[key][0].numpy(), ref_w[key][0])
    for key in ("wqkv_s", "wo_s", "w_inproj_s", "w_out_s", "w_e13_s", "w_e2_s"):
        np.testing.assert_allclose(
            got[key][0].float().numpy(),
            ref_w[key][0].astype(np.float32),
            atol=1e-3,
            rtol=1e-3,
        )
