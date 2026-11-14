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
"""Tests for Hadamard-rotated FP4 (e2m1) EP combine wire transport."""

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


class _FakeChip:
    num_lanes = 128
    generation = host.MIN_GENERATION

    def __init__(self, vmem_capacity_bytes: int = 128 * 2**20):
        self.vmem_capacity_bytes = vmem_capacity_bytes

    def get_sublane_tiling(self, dtype):
        return 32 // jnp.dtype(dtype).itemsize


def test_hadamard_128_dyadic_identity_is_exact():
    """Dyadic split (1/8 forward, 1/16 inverse) gives H_fwd @ H_inv = I with 0 error."""
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
        (16384, host.WeightFormat.FP4, True),
        (7168, host.WeightFormat.FP4, False),
        (4096, host.WeightFormat.FP4, False),
        (0, host.WeightFormat.FP4, False),
        (8192, host.WeightFormat.FP8, False),
        (8192, host.WeightFormat.BF16, False),
    ],
)
def test_can_use_fp4_combine_wire_alignment_and_format(hidden, weight_format, expected):
    assert host.can_use_fp4_combine_wire(hidden, weight_format) is expected


def test_fp4_combine_wire_reduces_vmem_estimate():
    """wire_fp4 halves out_vm (2 parities) and tile wire_rows (1 copy) in VMEM."""
    chip = _FakeChip()
    shape = dict(
        g_local=16,
        capacity=128,
        hidden=8192,
        inter=2048,
        nbuf=2,
        weight_format=host.WeightFormat.FP4,
        rhs_qb=512,
        info=chip,
    )
    fp8_wire_bytes = host.vmem_estimate_bytes(**shape, wire_fp4=False)
    fp4_wire_bytes = host.vmem_estimate_bytes(**shape, wire_fp4=True)
    # 2 out_vm parities + 1 wire_rows buffer = 3 * (128 * 8192 * 0.5) = 1.5 MiB saved.
    expected_saving = 3 * 128 * (8192 // 2)
    assert fp8_wire_bytes - fp4_wire_bytes == expected_saving


def test_hadamard_fp4_wire_roundtrip_suppresses_channel_outliers():
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
        acc2, None, wire_dtype=host.FP4, tile_m=total_routes
    )
    assert q_rows.dtype == host.FP4
    assert q_rows.shape == (total_routes, hidden)

    arrivals = q_rows.reshape(total_routes, lane_blocks, host.HIDDEN_LANE_BLOCK)
    # Place scales in a single sublane of width 128 (total_routes=80 <= 128).
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


def test_prebuild_fused_moe_ep_wires_fp4_flag(monkeypatch):
    """prebuild_fused_moe_ep enables wire_fp4 only on aligned FP4 shapes."""
    captured = {}

    def fake_build_op(*_args, **kwargs):
        captured["wire_fp4"] = kwargs.get("wire_fp4")
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
        patch.object(envs, "MOE_FUSED_EP_COMBINE_WIRE_FP4", True),
        patch.object(bridge, "build_ep_mesh", return_value=mesh),
        patch.object(bridge, "ep_mesh_index", return_value=0),
        patch.object(bridge, "ep_rank_order", return_value=None),
        patch.object(bridge, "ep_token_replica_groups", return_value=((0,), (1,))),
        patch.object(bridge, "_max_node_tokens", return_value=4096),
        patch.object(bridge, "_smem_capacity_bytes", return_value=1024 * 1024),
        patch.object(bridge, "_build_op", side_effect=fake_build_op),
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
        assert captured["wire_fp4"] is True

        # Unaligned hidden=7168 falls back cleanly to FP8 combine wire (wire_fp4=False).
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
        assert captured["wire_fp4"] is False


@pytest.mark.tpu_test
def test_pallas_combine_kernel_fp4_wire_matches_xla_reference():
    """On TPU, _build_combine_kernel with packed 4-bit wire matches XLA gather+sum."""
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU backend for Mosaic Pallas combine kernel")

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
        wire_dtype=host.FP4,
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
