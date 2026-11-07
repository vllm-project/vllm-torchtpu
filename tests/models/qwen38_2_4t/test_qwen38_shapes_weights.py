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

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

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
    _observed_layer_expert_attr,
    _resolve_expert_tensor,
    _transpose_packed_fp4_2d,
    clear_observed_weights,
    extract_backbone_weights_torch,
    load_tensor,
)

_TEST_DIR = str(Path(__file__).resolve().parent)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)
oracle = importlib.import_module("oracle_fp64")

EXPECTED_RANKS = 32
EXPECTED_DCP = 8
EXPECTED_LAYERS = 92
EXPECTED_KV_HEADS = 4
EXPECTED_VH = 64
EXPECTED_WEIGHT_KEYS = 28
U8_WEIGHT_KEYS = frozenset(
    {
        "wqkv",
        "wo",
        "w_inproj",
        "w_out",
        "w_se13",
        "w_se2",
        "w_e13",
        "w_e2",
    }
)


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

    with pytest.raises(ValueError, match="Invalid R="):
        Qwen38Config(R=0)
    with pytest.raises(ValueError, match=r"Q_H=.*must be divisible by KV_H\*DCP"):
        Qwen38Config(R=8, DCP=4, Q_H=16, KV_H=8)
    with pytest.raises(
        ValueError, match=r"rope_dim=.*must be positive, even, and <= D"
    ):
        Qwen38Config(rope_dim=63)
    with pytest.raises(
        ValueError, match=r"rope_dim=.*must be positive, even, and <= D"
    ):
        Qwen38Config(rope_dim=512)
    with pytest.raises(ValueError, match=r"KH=.*must divide or be divisible by R"):
        Qwen38Config(KH=48, R=32)
    with pytest.raises(ValueError, match="Incompatible GDN head counts"):
        Qwen38Config(R=8, DCP=8, KH=16, VH=24)


def test_qwen38_weights_keys() -> None:
    """Verify WEIGHT_KEYS registry, load_tensor hook, and fail-closed validation."""
    assert len(WEIGHT_KEYS) == EXPECTED_WEIGHT_KEYS
    assert "wqkv" in WEIGHT_KEYS
    assert "in_norm" in WEIGHT_KEYS

    clear_observed_weights()
    t = torch.arange(16, dtype=torch.uint8).reshape(4, 4)
    load_tensor("model.layers.0.mlp.experts.w13_weight", t)
    captured = _observed_layer_expert_attr(0, "w13_weight")
    assert captured is not None
    assert torch.equal(captured, t)
    assert captured.data_ptr() != t.data_ptr()

    # Non-expert checkpoint tensors must be ignored.
    load_tensor("model.layers.0.self_attn.qkv_proj.weight", t)
    assert _observed_layer_expert_attr(0, "qkv_proj") is None

    clear_observed_weights()
    assert _observed_layer_expert_attr(0, "w13_weight") is None

    untyped_load: Any = load_tensor
    with pytest.raises(TypeError, match="Expected str tensor name"):
        untyped_load(123, t)
    with pytest.raises(TypeError, match=r"Expected torch\.Tensor"):
        untyped_load("model.layers.0.mlp.experts.w13_weight", None)

    with pytest.raises(ValueError, match="Cannot locate"):
        extract_backbone_weights_torch(object(), mini_parity_config())

    # Missing shared_expert_gate, expert attr, or linear weight raises ValueError.
    tiny_cfg = _make_tiny_cfg()
    bad_mlp = SimpleNamespace(
        gate=SimpleNamespace(weight=torch.ones((tiny_cfg.E, tiny_cfg.H))),
        shared_expert_gate=None,
    )
    untyped_extract_mlp: Any = _extract_mlp_sublayer
    with pytest.raises(
        ValueError, match=r"Layer 0 MLP is missing required shared_expert_gate\.weight"
    ):
        untyped_extract_mlp(bad_mlp, 0, tiny_cfg, use_observed=False)

    with pytest.raises(
        ValueError, match="Layer 0 experts is missing required attribute 'w13_weight'"
    ):
        _resolve_expert_tensor(
            SimpleNamespace(), 0, "w13_weight", use_observed=False, required=True
        )

    with pytest.raises(ValueError, match="missing required attribute 'weight'"):
        _extract_linear_fp4(SimpleNamespace(weight=None), 32, 32)


def test_qwen38_compute_parity_with_fp64_oracle() -> None:
    """Verify all pure-JAX compute functions match the FP64 oracle."""
    cfg = _make_tiny_cfg()
    prob = oracle.make_synthetic_problem(cfg, seed=7, pos=(3,))
    w = {k: np.asarray(jax.device_get(v)) for k, v in prob["weights"].items()}
    x_np = oracle.bf16_f64(np.asarray(jax.device_get(prob["x"]))[0])

    # 1. RMSNorm parity against oracle.rmsnorm_f64 (rounded to bf16).
    rms_got = np.asarray(
        rmsnorm(prob["x"][0], prob["weights"]["in_norm"][0, 0]).astype(jnp.float32),
        dtype=np.float64,
    )
    rms_ref = oracle.rmsnorm_f64(x_np, w["in_norm"][0, 0])
    np.testing.assert_allclose(rms_got, oracle.bf16_f64(rms_ref), atol=1e-6, rtol=0.0)

    # 2. Partial RoPE parity against oracle.rope_f64 + bf16 rounding.
    rng = np.random.default_rng(11)
    q_in = jnp.asarray(
        rng.normal(0.0, 0.5, size=(cfg.B, cfg.Q_R, cfg.D)).astype(np.float32),
        dtype=jnp.bfloat16,
    )
    pos_jnp = prob["pos"][0]
    rope_got = np.asarray(
        partial_rope(q_in, pos_jnp, cfg.rope_dim).astype(jnp.float32),
        dtype=np.float64,
    )
    rope_ref = oracle.bf16_f64(
        oracle.rope_f64(
            np.asarray(q_in.astype(jnp.float32), dtype=np.float64),
            np.asarray(pos_jnp),
            cfg.rope_dim,
        )
    )
    np.testing.assert_allclose(rope_got, rope_ref, atol=1e-2, rtol=1e-2)

    # 3. dequant_nvfp4 + fp4_matmul parity against oracle.decode_fp4_f64 + control.
    pk_jnp = prob["weights"]["wqkv"][0, 0]
    sc_jnp = prob["weights"]["wqkv_s"][0, 0]
    deq_got = np.asarray(
        dequant_nvfp4(pk_jnp, sc_jnp, group_size=cfg.fp4_group).astype(jnp.float32),
        dtype=np.float64,
    )
    deq_ref = oracle.decode_fp4_f64(w["wqkv"][0, 0], w["wqkv_s"][0, 0])
    np.testing.assert_allclose(deq_got, deq_ref, atol=1e-6, rtol=0.0)

    # Must-fail control: flipping a nibble code in packed_u8 changes dequant output.
    pk_perturbed = pk_jnp.at[0, 0].set(jnp.uint8(int(pk_jnp[0, 0]) ^ 0x07))
    sc_nonzero = sc_jnp.at[0, 0].set(jnp.bfloat16(1.0))
    deq_clean = np.asarray(
        dequant_nvfp4(pk_jnp, sc_nonzero, group_size=cfg.fp4_group).astype(jnp.float32)
    )
    deq_bad = np.asarray(
        dequant_nvfp4(pk_perturbed, sc_nonzero, group_size=cfg.fp4_group).astype(
            jnp.float32
        )
    )
    min_perturbed_diff = 0.5
    assert float(np.max(np.abs(deq_clean - deq_bad))) >= min_perturbed_diff

    mm_got = np.asarray(
        fp4_matmul(
            jnp.asarray(rms_ref, dtype=jnp.bfloat16),
            pk_jnp,
            sc_jnp,
            group_size=cfg.fp4_group,
        ),
        dtype=np.float64,
    )
    mm_ref = oracle.bf16_f64(rms_ref) @ deq_ref
    bound_mm = oracle.layer_bf16_error_bound(
        float(np.sqrt(np.mean(mm_ref**2))), n_subblocks=1, n_layers=1
    )
    assert float(np.sqrt(np.mean((mm_got - mm_ref) ** 2))) <= bound_mm

    _verify_dcp_online_softmax_and_lse_parity(cfg, prob, pos_jnp, rng)
    _verify_gdn_and_moe_parity(cfg, prob, w, rms_ref, rng)


def _verify_dcp_online_softmax_and_lse_parity(
    cfg: Qwen38Config,
    prob: dict[str, Any],
    pos_jnp: jax.Array,
    rng: np.random.Generator,
) -> None:
    """Verify dcp_online_softmax_update + dcp_merge_lse against the FP64 oracle."""
    # 4. dcp_online_softmax_update + dcp_merge_lse parity against the FP64 oracle,
    #    including an FP8 NaN value in an unwritten tail KV slot past pos_b.
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
            # Inject an FP8 NaN (0x7F) at the last page slot (past pos_b=3) on
            # page 0 to verify unwritten V slots are masked before contraction.
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
    lse_recv = jnp.stack(lse_ranks, axis=0)

    q_gate_np = oracle.bf16_f64(
        rng.normal(0.0, 0.2, size=(cfg.B, cfg.Q_R, cfg.D)).astype(np.float32)
    )
    merged_got = np.asarray(
        dcp_merge_lse(
            lse_recv,
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


def _verify_gdn_and_moe_parity(
    cfg: Qwen38Config,
    prob: dict[str, Any],
    w: dict[str, np.ndarray],
    rms_ref: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Verify GDN recurrent step, MoE router, SwiGLU split, and error bound helpers."""
    # 5. gdn_recurrent_step parity against oracle.ref_gdn_step intermediate equations.
    w_in = oracle.decode_fp4_f64(w["w_inproj"][0, 0], w["w_inproj_s"][0, 0])
    qz_np = rms_ref @ w_in
    qkv_in = jnp.asarray(qz_np[:, : cfg.QKV], dtype=jnp.bfloat16)
    z_in = jnp.asarray(qz_np[:, cfg.QKV :], dtype=jnp.bfloat16)
    ba_in = jnp.asarray(rms_ref @ oracle.bf16_f64(w["w_ba"][0, 0]), dtype=jnp.float32)
    y_gdn, conv_next, rec_next = gdn_recurrent_step(
        (qkv_in, z_in, ba_in),
        (prob["conv"][0, 0], prob["rec"][0, 0]),
        (
            prob["weights"]["w_conv"][0, 0],
            prob["weights"]["a_log"][0, 0],
            prob["weights"]["dt_bias"][0, 0],
            prob["weights"]["gdn_norm"][0, 0],
        ),
        cfg,
    )
    gdn_state = oracle.GdnRefState(
        conv=oracle.bf16_f64(np.asarray(jax.device_get(prob["conv"]))).copy(),
        rec=np.asarray(jax.device_get(prob["rec"]), dtype=np.float64).copy(),
    )
    _ = oracle.ref_gdn_step(rms_ref, gdn_state, w, g_idx=0, cfg=cfg)
    np.testing.assert_allclose(
        np.asarray(conv_next.astype(jnp.float32), dtype=np.float64),
        gdn_state.conv[0, 0],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(rec_next, dtype=np.float64),
        gdn_state.rec[0, 0],
        atol=1e-4,
        rtol=1e-4,
    )
    assert y_gdn.shape == (cfg.B, cfg.Z_R)

    # 6. moe_router_topk and swiglu_split parity against FP64 oracle equations.
    top_ids, top_w = moe_router_topk(
        jnp.asarray(rms_ref, dtype=jnp.bfloat16),
        prob["weights"]["w_gate"][0, 0],
        cfg,
    )
    logits_ref = oracle.bf16_f64(rms_ref) @ oracle.bf16_f64(w["w_gate"][0, 0])
    probs_ref = np.exp(logits_ref - np.max(logits_ref, axis=-1, keepdims=True))
    probs_ref = probs_ref / np.sum(probs_ref, axis=-1, keepdims=True)
    ids_ref = np.argsort(-probs_ref, axis=-1)[:, : cfg.K]
    p_ref = np.take_along_axis(probs_ref, ids_ref, axis=-1)
    w_ref = p_ref / np.sum(p_ref, axis=-1, keepdims=True)
    np.testing.assert_array_equal(np.asarray(top_ids), ids_ref)
    np.testing.assert_allclose(np.asarray(top_w), w_ref, atol=1e-4, rtol=1e-4)

    gu_np = rng.normal(0.0, 0.5, size=(cfg.B, 2 * cfg.SE_R)).astype(np.float32)
    act_got = np.asarray(
        swiglu_split(jnp.asarray(gu_np), inter_dim=cfg.SE_R).astype(jnp.float32),
        dtype=np.float64,
    )
    act_ref = oracle.bf16_f64(
        oracle.silu_f64(gu_np[:, : cfg.SE_R].astype(np.float64))
        * gu_np[:, cfg.SE_R :].astype(np.float64)
    )
    np.testing.assert_allclose(act_got, act_ref, atol=1e-3, rtol=1e-2)

    # 7. Verify max_fp8_ulp_diff returns 0 on identical tensors and >0 on perturbed FP8.
    fp8_sample = prob["pool"][0, 0, 0, :, :, 0]
    assert oracle.max_fp8_ulp_diff(fp8_sample, fp8_sample) == 0
    assert oracle.max_fp8_ulp_diff(fp8_sample, jnp.zeros_like(fp8_sample)) > 0


def test_qwen38_oracle_bf16_cast_and_weight_extraction() -> None:
    """Verify non-exact BF16 RNE rounding and full 28-key weight round-trip."""
    # Test values that are not exact in bfloat16, including round-to-nearest-even ties:
    # 1 + 2^-8 (halfway between 1.0 and 1.0 + 2^-7; even LSB -> rounds down to 1.0),
    # 1 + 3 * 2^-8 (halfway between 1 + 2^-7 and 1 + 2^-6; odd LSB rounds up),
    # plus non-tie values and negative tie values.
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

    mock_models = oracle.build_mock_vllm_qwen38_rank_models(prob, tiny_cfg)

    # 1. Test the production single-backbone extraction path (`(1, ...)`).
    extracted_single = extract_backbone_weights_torch(mock_models.model, tiny_cfg)
    for k in WEIGHT_KEYS:
        got_arr = extracted_single[k].float().numpy()
        ref_arr = np.asarray(jax.device_get(prob["weights"][k]))[:1]
        assert got_arr.shape == ref_arr.shape, f"Shape mismatch on {k}"
        if k in U8_WEIGHT_KEYS:
            np.testing.assert_array_equal(
                extracted_single[k].numpy(), ref_arr.astype(np.uint8)
            )
        else:
            np.testing.assert_allclose(
                got_arr, ref_arr.astype(np.float32), atol=1e-3, rtol=1e-3
            )

    # 2. Test the multi-rank extraction path across all R backbones (`(R, ...)`).
    extracted_all = extract_backbone_weights_torch(
        mock_models, tiny_cfg, rank_backbones=mock_models.rank_backbones
    )
    for k in WEIGHT_KEYS:
        got_arr = extracted_all[k].float().numpy()
        ref_arr = np.asarray(jax.device_get(prob["weights"][k]))
        assert got_arr.shape == ref_arr.shape, f"Multi-rank shape mismatch on {k}"
        if k in U8_WEIGHT_KEYS:
            np.testing.assert_array_equal(
                extracted_all[k].numpy(), ref_arr.astype(np.uint8)
            )
        else:
            np.testing.assert_allclose(
                got_arr, ref_arr.astype(np.float32), atol=1e-3, rtol=1e-3
            )

    # 3. Negative control: swapping lo/hi nibbles breaks round-trip packed equality.
    orig_wqkv = np.asarray(
        jax.device_get(prob["weights"]["wqkv"][0, 0]), dtype=np.uint8
    )
    pk_nk, _ = oracle.pack_rows_fp4(
        orig_wqkv,
        np.asarray(jax.device_get(prob["weights"]["wqkv_s"][0, 0]), dtype=np.float32),
    )
    swapped_nk = ((pk_nk.numpy() & 0x0F) << 4) | ((pk_nk.numpy() >> 4) & 0x0F)
    unpacked_swapped = _transpose_packed_fp4_2d(
        swapped_nk, tiny_cfg.H, tiny_cfg.ATT_QKV_R
    )
    assert not np.array_equal(unpacked_swapped, orig_wqkv)
