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
"""Tests for Hadamard-rotated FP4 (e2m1) EP combine collective."""

from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.fused_moe.v2 import host, kernel, layer
from vllm_torchtpu.layers.adapter import fused_moe_ep as bridge
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform


class _FakeChip:
    num_lanes = 128
    generation = host.MIN_GENERATION

    def __init__(self, vmem_capacity_bytes: int = 128 * 2**20):
        self.vmem_capacity_bytes = vmem_capacity_bytes

    def get_sublane_tiling(self, dtype):
        return 32 // jnp.dtype(dtype).itemsize


def test_hadamard_128_dyadic_identity_is_exact():
    """Dyadic split (1/8 forward, 1/16 inverse) gives H_fwd @ H_inv = I exactly."""
    h_fwd = host.hadamard_128_bf16(inverse=False)
    h_inv = host.hadamard_128_bf16(inverse=True)
    assert h_fwd.shape == (128, 128)
    assert h_inv.shape == (128, 128)
    assert h_fwd.dtype == jnp.bfloat16
    assert h_inv.dtype == jnp.bfloat16

    prod = jnp.matmul(h_fwd, h_inv, preferred_element_type=jnp.float32)
    eye = jnp.eye(128, dtype=jnp.float32)
    assert float(jnp.max(jnp.abs(prod - eye))) == 0.0


@pytest.mark.parametrize(
    "hidden,weight_format,expected",
    [
        (8192, host.WeightFormat.FP4, True),
        (16384, "fp4", True),
        (7168, host.WeightFormat.FP4, False),
        (4096, host.WeightFormat.FP4, False),
        (0, host.WeightFormat.FP4, False),
        (8192, host.WeightFormat.FP8, False),
        (8192, host.WeightFormat.BF16, False),
    ],
)
def test_can_use_fp4_combine_alignment_and_format(hidden, weight_format, expected):
    assert host.can_use_fp4_combine(hidden, weight_format) is expected


def test_fp4_combine_vmem_scratch_and_tile_body_accounting():
    """combine_fp4 shrinks persistent out_vm by 1.0 MiB and models tile temporaries."""
    chip = _FakeChip()
    tile_m, hidden, inter = 128, 8192, 2048
    scratch_kw = dict(
        g_local=16,
        capacity=tile_m,
        hidden=hidden,
        inter=inter,
        nbuf=2,
        weight_format=host.WeightFormat.FP4,
        rhs_qb=512,
    )
    s_fp8 = sum(
        host.array_vmem_bytes(sh, dt, chip)
        for _, sh, dt in host.vmem_scratch_arrays(**scratch_kw, combine_fp4=False)
    )
    s_fp4 = sum(
        host.array_vmem_bytes(sh, dt, chip)
        for _, sh, dt in host.vmem_scratch_arrays(**scratch_kw, combine_fp4=True)
    )
    # 2 out_vm parities shrink from 1 byte/elem (FP8) to 0.5 byte/elem (packed FP4).
    assert s_fp8 - s_fp4 == 2 * tile_m * (hidden // 2)

    fp8_total = host.vmem_estimate_bytes(**scratch_kw, info=chip, combine_fp4=False)
    fp4_total = host.vmem_estimate_bytes(**scratch_kw, info=chip, combine_fp4=True)
    # Tile body swaps 1.0 MiB FP8 wire_rows for 32 KiB H128 + 2.0 MiB down_rot_bf16
    # + 0.5 MiB packed wire_rows (+1.5 MiB + 32 KiB), net +0.5 MiB + 32 KiB.
    expected_delta = (tile_m * (hidden // 2)) + (128 * 128 * 2)
    assert fp4_total - fp8_total == expected_delta


def test_rowquant_fp4_hadamard_subnormal_and_zero_blocks_have_no_nan():
    """Subnormal rows whose 1/scale overflows bf16 must zero cleanly without NaN."""
    row = np.zeros((4, 256), dtype=np.float32)
    # Row 0: subnormal in block 0, exact zero in block 1 -> 0 * (+inf) trap test.
    row[0, 0] = 1.2e-38
    # Row 1: all zeros.
    # Row 2: normal values.
    row[2, :] = np.linspace(-2.0, 2.0, 256, dtype=np.float32)
    # Row 3: non-finite input should also zero cleanly.
    row[3, 0] = np.inf
    x = jnp.asarray(row, dtype=jnp.bfloat16)
    q, scale = kernel.rowquant_fp4_hadamard(x)
    assert not bool(jnp.any(jnp.isnan(q.astype(jnp.float32))))
    assert not bool(jnp.any(jnp.isnan(scale)))
    assert float(scale[0, 0]) == 0.0
    assert float(scale[1, 0]) == 0.0
    assert float(scale[3, 0]) == 0.0
    assert float(scale[2, 0]) > 0.0


def test_hadamard_fp4_combine_roundtrip_suppresses_channel_outliers():
    """Block-128 Hadamard rotation preserves topk=10 combine cosine > 0.99."""
    rng = np.random.default_rng(42)
    t_local, topk, hidden = 8, 10, 8192
    lane_blocks = hidden // host.HIDDEN_LANE_BLOCK
    total_routes = t_local * topk

    # Simulate expert down_proj outputs with 12x fixed channel outliers.
    raw = rng.standard_normal((total_routes, hidden)).astype(np.float32) * 0.25
    outlier_cols = np.arange(0, hidden, 256)
    raw[:, outlier_cols] *= 12.0
    acc2 = jnp.asarray(raw, dtype=jnp.float32)

    weights_np = rng.dirichlet(np.ones(topk), size=t_local).astype(np.float32)
    topk_weights = jnp.asarray(weights_np, dtype=jnp.float32)

    # Exact reference weighted sum [t_local, hidden].
    ref = jnp.sum(
        acc2.reshape(t_local, topk, hidden) * topk_weights[:, :, None], axis=1
    )

    q_rows, row_scales = kernel.wire_row_and_scale(
        acc2, None, wire_dtype=jnp.dtype(host.FP4), tile_m=total_routes
    )
    assert q_rows.dtype == host.FP4
    assert q_rows.shape == (total_routes, hidden)

    arrivals = q_rows.reshape(total_routes, lane_blocks, host.HIDDEN_LANE_BLOCK)
    tile_m = 128
    arrival_scales = (
        jnp.zeros((1, tile_m), dtype=jnp.float32)
        .at[0, :total_routes]
        .set(row_scales.reshape(-1))
    )
    pos = jnp.arange(total_routes, dtype=jnp.int32).reshape(t_local, topk)

    combined = layer._combine_arrivals(
        arrivals, arrival_scales, pos, pos, topk_weights, jnp.bfloat16
    ).astype(jnp.float32)

    cos = float(
        jnp.sum(ref * combined) / (jnp.linalg.norm(ref) * jnp.linalg.norm(combined))
    )
    rel_err = float(jnp.linalg.norm(ref - combined) / jnp.linalg.norm(ref))
    assert cos > 0.99, f"expected cosine > 0.99, got {cos:.6f}"
    assert rel_err < 0.14, f"expected rel_err < 0.14, got {rel_err:.6f}"


def test_direct_unaligned_combine_fp4_raises_value_error():
    """Direct kernel and combine builders reject unaligned hidden=7168."""
    with pytest.raises(ValueError, match="divisible by 8192"):
        kernel._build_fused_ep_moe_kernel(
            g_local=2,
            capacity=128,
            hidden=7168,
            inter=512,
            ep=2,
            ragged_rows_alloc=256,
            weight_format=host.WeightFormat.FP4,
            rhs_qb=512,
            combine_fp4=True,
        )
    with pytest.raises(ValueError, match="divisible by 8192"):
        kernel.build_combine_kernel(
            t_local=256,
            topk=2,
            hidden=7168,
            lane_blocks=7168 // host.HIDDEN_LANE_BLOCK,
            wire_dtype=host.FP4,
            out_dtype=jnp.bfloat16,
        )


def test_moe_fused_ep_combine_dtype_env_validation(monkeypatch):
    assert "MOE_FUSED_EP_COMBINE_DTYPE" in TpuPlatform.additional_env_vars
    monkeypatch.delenv("MOE_FUSED_EP_COMBINE_DTYPE", raising=False)
    assert envs.MOE_FUSED_EP_COMBINE_DTYPE is None
    monkeypatch.setenv("MOE_FUSED_EP_COMBINE_DTYPE", "fp4")
    assert envs.MOE_FUSED_EP_COMBINE_DTYPE == "fp4"
    monkeypatch.setenv("MOE_FUSED_EP_COMBINE_DTYPE", "invalid")
    with pytest.raises(ValueError, match="Invalid value"):
        _ = envs.MOE_FUSED_EP_COMBINE_DTYPE


def test_prebuild_fused_moe_ep_arms_fp4_combine_and_warns_on_fallback():
    """Arms combine_fp4 on aligned FP4 shapes and warns on fallback."""
    captured: dict[str, object] = {}

    def fake_build_op(*_args, **kwargs):
        captured["combine_fp4"] = kwargs.get("combine_fp4")
        return "op"

    def make_layer(hidden: int):
        experts, inter = 4, 512
        return SimpleNamespace(
            w13_weight=torch.zeros(experts, hidden // 2, 2 * inter, dtype=torch.uint8),
            w2_weight=torch.zeros(experts, inter // 2, hidden, dtype=torch.uint8),
            w13_weight_scale=torch.ones(experts, hidden // 512, 1, 2 * inter),
            w2_weight_scale=torch.ones(experts, inter // 512, 1, hidden),
            global_num_experts=experts * 2,
            custom_routing_function=None,
            scoring_func="softmax",
            e_score_correction_bias=None,
            use_grouped_topk=False,
            num_expert_group=None,
            topk_group=None,
            routed_scaling_factor=1.0,
            moe_config=SimpleNamespace(
                experts_per_token=2,
                moe_parallel_config=SimpleNamespace(
                    use_ep=True, pcp_size=1, is_sequence_parallel=False
                ),
            ),
        )

    mesh = SimpleNamespace(shape={bridge.EP_AXIS_NAME: 2})
    with (
        patch.object(envs, "USE_MOE_FUSED_EP_KERNEL", True),
        patch.object(envs, "MOE_FUSED_EP_ENABLE_W4A8", True),
        patch.object(envs, "MOE_FUSED_EP_KERNEL_MIN_TOKENS", 64),
        patch.object(envs, "MOE_FUSED_EP_COMBINE_DTYPE", "fp4"),
        patch.object(bridge, "build_ep_mesh", return_value=mesh),
        patch.object(bridge, "ep_mesh_index", return_value=0),
        patch.object(bridge, "ep_rank_order", return_value=None),
        patch.object(bridge, "ep_token_replica_groups", return_value=((0,), (1,))),
        patch.object(bridge, "_max_node_tokens", return_value=4096),
        patch.object(bridge, "_smem_capacity_bytes", return_value=1024 * 1024),
        patch.object(bridge, "_scales_expert_second_minor", return_value=False),
        patch.object(bridge, "_build_op", side_effect=fake_build_op),
        patch.object(bridge.logger, "warning_once") as mock_warn,
    ):
        lyr_8192 = make_layer(8192)
        weights_8192 = (
            torch.empty(4, 8192, 512, dtype=torch.float4_e2m1fn_x2, device="meta"),
            torch.empty(4, 512, 8192 // 2, dtype=torch.float4_e2m1fn_x2, device="meta"),
            torch.empty(4, 8192 // 512, 1, 1024, dtype=torch.float32, device="meta"),
            torch.empty(4, 512 // 512, 1, 8192, dtype=torch.float32, device="meta"),
        )
        assert (
            bridge.prebuild_fused_moe_ep(
                lyr_8192,
                topk=2,
                renormalize=True,
                activation="silu",
                weight_format="fp4",
                rhs_qb=512,
                weights=weights_8192,
            )
            == "op"
        )
        assert captured["combine_fp4"] is True
        mock_warn.assert_not_called()

        # Unaligned hidden=7168 logs warning_once and falls back to FP8 combine.
        lyr_7168 = make_layer(7168)
        weights_7168 = (
            torch.empty(4, 7168, 512, dtype=torch.float4_e2m1fn_x2, device="meta"),
            torch.empty(4, 512, 7168 // 2, dtype=torch.float4_e2m1fn_x2, device="meta"),
            torch.empty(4, 7168 // 512, 1, 1024, dtype=torch.float32, device="meta"),
            torch.empty(4, 512 // 512, 1, 7168, dtype=torch.float32, device="meta"),
        )
        assert (
            bridge.prebuild_fused_moe_ep(
                lyr_7168,
                topk=2,
                renormalize=True,
                activation="silu",
                weight_format="fp4",
                rhs_qb=512,
                weights=weights_7168,
            )
            == "op"
        )
        assert captured["combine_fp4"] is False
        mock_warn.assert_called_once()


@pytest.mark.tpu_test
def test_pallas_combine_kernel_fp4_matches_xla_reference():
    """Packed 4-bit arrivals in _build_combine_kernel match XLA gather+sum."""
    if jax.default_backend() != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {jax.default_backend()!r}")

    t_local, topk, hidden = 256, 4, 8192
    assert kernel.combine_step_tokens(t_local) > 0
    lane_blocks = hidden // host.HIDDEN_LANE_BLOCK
    recv_rows = 512

    key = jax.random.PRNGKey(7)
    k1, k2, k3 = jax.random.split(key, 3)
    arrivals_bf16 = jax.random.normal(
        k1, (recv_rows, lane_blocks, host.HIDDEN_LANE_BLOCK), dtype=jnp.bfloat16
    )
    arrivals_fp4 = jnp.clip(arrivals_bf16, -host.FP4_MAX, host.FP4_MAX).astype(host.FP4)
    pos = jax.random.randint(k2, (t_local, topk), 0, recv_rows, dtype=jnp.int32)
    coef = jax.random.uniform(k3, (t_local, topk), dtype=jnp.float32) * 0.25

    combine_fn = kernel.build_combine_kernel(
        t_local=t_local,
        topk=topk,
        hidden=hidden,
        lane_blocks=lane_blocks,
        wire_dtype=arrivals_fp4.dtype,
        out_dtype=jnp.bfloat16,
    )
    got = combine_fn(arrivals_fp4, coef, pos).astype(jnp.float32)

    ref = (
        jnp.sum(arrivals_fp4[pos].astype(jnp.float32) * coef[:, :, None, None], axis=1)
        .astype(jnp.bfloat16)
        .reshape(t_local, hidden)
        .astype(jnp.float32)
    )

    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-2, atol=2e-2)


@pytest.mark.tpu_test
def test_pallas_rowquant_fp4_hadamard_captures_no_constants():
    """rowquant_fp4_hadamard lowers inside pallas_call and matches XLA values."""
    if jax.default_backend() != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {jax.default_backend()!r}")

    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    x = jax.random.normal(jax.random.PRNGKey(3), (64, 256), dtype=jnp.bfloat16)

    def _body(x_ref, q_f32_ref, s_ref):
        q, s = kernel.rowquant_fp4_hadamard(x_ref[...])
        packed_u32 = pltpu.bitcast(q, jnp.uint32)
        q_f32_ref[...] = pltpu.bitcast(packed_u32, host.FP4).astype(jnp.float32)
        s_ref[...] = jnp.broadcast_to(s, (64, 128))

    q_unpacked, scales = pl.pallas_call(
        _body,
        out_shape=(
            jax.ShapeDtypeStruct((64, 256), jnp.float32),
            jax.ShapeDtypeStruct((64, 128), jnp.float32),
        ),
    )(x)
    q_ref_xla, s_ref_xla = kernel.rowquant_fp4_hadamard(x)
    np.testing.assert_allclose(
        np.asarray(scales[:, :1]), np.asarray(s_ref_xla), rtol=1e-3, atol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(q_unpacked),
        np.asarray(q_ref_xla.astype(jnp.float32)),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.multichip
@pytest.mark.parametrize("t_local", [64, 256])
def test_fused_ep_moe_v2_end_to_end_fp4_combine_multichip(t_local: int):
    """End-to-end fused_ep_moe_v2 with combine_fp4=True across VMEM & HBM combine."""
    if jax.default_backend() != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {jax.default_backend()!r}")
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("multichip test requires >= 2 TPU devices")

    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    ep = min(len(devices), 8)
    mesh = Mesh(np.asarray(devices[:ep]), ("ep",))
    hidden, inter, g_local, topk, rhs_qb = 8192, 512, 2, 4, 512
    e_total = ep * g_local
    total_tokens = ep * t_local

    key = jax.random.PRNGKey(19 + t_local)
    kx, kw1, kw2, kg = jax.random.split(key, 4)
    x = jax.device_put(
        jax.random.normal(kx, (total_tokens, hidden), dtype=jnp.bfloat16) * 0.25,
        NamedSharding(mesh, P("ep", None)),
    )
    w1_fp4 = jax.device_put(
        jnp.clip(
            jax.random.normal(kw1, (e_total, hidden, 2 * inter), dtype=jnp.bfloat16),
            -host.FP4_MAX,
            host.FP4_MAX,
        ).astype(host.FP4),
        NamedSharding(mesh, P("ep", None, None)),
    )
    w2_fp4 = jax.device_put(
        jnp.clip(
            jax.random.normal(kw2, (e_total, inter, hidden), dtype=jnp.bfloat16),
            -host.FP4_MAX,
            host.FP4_MAX,
        ).astype(host.FP4),
        NamedSharding(mesh, P("ep", None, None)),
    )
    w1_scale = jax.device_put(
        jnp.full((e_total, hidden // rhs_qb, 2 * inter), 0.05, dtype=jnp.float32),
        NamedSharding(mesh, P("ep", None, None)),
    )
    w2_scale = jax.device_put(
        jnp.full((e_total, inter // rhs_qb, hidden), 0.05, dtype=jnp.float32),
        NamedSharding(mesh, P("ep", None, None)),
    )
    gating = jax.device_put(
        jax.random.normal(kg, (total_tokens, e_total), dtype=jnp.float32),
        NamedSharding(mesh, P("ep", None)),
    )

    common_kw = dict(
        topk=topk,
        renormalize=True,
        mesh=mesh,
        capacity=128,
        weight_format=host.WeightFormat.FP4,
        rhs_qb=rhs_qb,
    )
    out_fp8 = layer.fused_ep_moe_v2(
        x, w1_fp4, w2_fp4, w1_scale, w2_scale, gating, combine_fp4=False, **common_kw
    ).astype(jnp.float32)
    out_fp4 = layer.fused_ep_moe_v2(
        x, w1_fp4, w2_fp4, w1_scale, w2_scale, gating, combine_fp4=True, **common_kw
    ).astype(jnp.float32)

    a = np.asarray(out_fp8)
    b = np.asarray(out_fp4)
    assert not np.any(np.isnan(b))
    cos = float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    assert cos > 0.985, f"t_local={t_local} end-to-end combine_fp4 cosine={cos:.6f}"
