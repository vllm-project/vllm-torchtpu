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
"""Host checks for choosing asymmetric weight buffers within VMEM."""

import dataclasses
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import host, layer
from vllm_torchtpu.kernels.fused_moe.v2 import kernel as v2_kernel


@pytest.mark.parametrize(
    "hidden,inter,weight_format,rhs_qb,expected,expected_mib",
    [
        (4096, 1024, host.WeightFormat.FP8, 512, (3, 2), 40.390625),
        (8192, 2048, host.WeightFormat.FP4, 64, (2, 1), 60.765625),
    ],
)
def test_model_geometry_selects_buffers_that_fit(
    hidden, inter, weight_format, rhs_qb, expected, expected_mib
):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    shape = {
        "g_local": 64,
        "capacity": 128,
        "hidden": hidden,
        "inter": inter,
        "weight_format": weight_format,
        "rhs_qb": rhs_qb,
    }
    plan = host.select_weight_buffers(**shape, info=info)
    assert (plan.w1_nbuf, plan.w2_nbuf) == expected
    assert plan.vmem_bytes == int(expected_mib * 2**20)
    arrays = {
        name: dims
        for name, dims, _ in host.vmem_scratch_arrays(
            **shape, nbuf=plan.w1_nbuf, w2_nbuf=plan.w2_nbuf
        )
    }
    assert arrays["w1_vm"][0] == expected[0]
    assert arrays["w2_vm"][0] == expected[1]
    scale_slots = expected[0] if weight_format == host.WeightFormat.FP4 else 64
    assert arrays["w1s_vm"][0] == arrays["w2s_vm"][0] == scale_slots


def test_budget_boundary_falls_back_and_rejects_when_minimum_does_not_fit(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    shape = {"g_local": 64, "capacity": 128, "hidden": 4096, "inter": 1024}
    monkeypatch.setattr(host, "VMEM_FRACTION", 1.0)
    full = host.vmem_estimate_bytes(**shape, nbuf=3, w2_nbuf=2, info=info)
    minimum = host.vmem_estimate_bytes(**shape, nbuf=2, w2_nbuf=1, info=info)
    exact = dataclasses.replace(info, vmem_capacity_bytes=full)
    tight = dataclasses.replace(info, vmem_capacity_bytes=full - 1)
    exhausted = dataclasses.replace(info, vmem_capacity_bytes=minimum - 1)
    assert host.select_weight_buffers(**shape, info=exact).w1_nbuf == 3
    assert host.select_weight_buffers(**shape, info=tight).w1_nbuf == 2
    with pytest.raises(ValueError, match="minimum W1/W2=2/1"):
        host.select_weight_buffers(**shape, info=exhausted)


def test_adaptive_combine_arrivals_dispatches_on_smem_budget(monkeypatch):
    """adaptive_fused_moe._combine_arrivals uses the shared SMEM budget gate."""
    t_local = v2_kernel.COMBINE_MIN_TOKENS
    topk, lane_blocks, lanes, tile_m = 10, 2, 128, 128
    hidden = lane_blocks * lanes

    rng = np.random.default_rng(7)
    recv_rows = 64
    arrivals_np = rng.uniform(-1.0, 1.0, (recv_rows, lane_blocks, lanes)).astype(
        np.float32
    )
    scales_np = rng.uniform(0.5, 1.5, (2, tile_m)).astype(np.float32)
    pos_np = rng.integers(0, recv_rows, (t_local, topk), dtype=np.int32)
    mirror_pos_np = rng.integers(0, 2 * tile_m, (t_local, topk), dtype=np.int32)
    weights_np = rng.uniform(0.05, 0.25, (t_local, topk)).astype(np.float32)

    row_scales = scales_np[mirror_pos_np // tile_m, mirror_pos_np % tile_m]
    coef_np = row_scales * weights_np
    expected = (
        (arrivals_np[pos_np] * coef_np[:, :, None, None])
        .sum(axis=1)
        .reshape(t_local, hidden)
    )

    with jax.default_device(jax.devices("cpu")[0]):
        args = (
            jnp.asarray(arrivals_np),
            jnp.asarray(scales_np),
            jnp.asarray(pos_np),
            jnp.asarray(mirror_pos_np),
            jnp.asarray(weights_np),
            jnp.bfloat16,
        )

        # Over budget: must skip Pallas combine kernel and match XLA fallback.
        monkeypatch.setattr(
            pltpu,
            "get_tpu_info",
            lambda: types.SimpleNamespace(smem_capacity_bytes=48 * 1024),
        )

        def _fail_if_built(**kwargs):
            raise AssertionError(
                f"build_combine_kernel should not be called when over SMEM: {kwargs}"
            )

        monkeypatch.setattr(layer, "build_combine_kernel", _fail_if_built)
        got = layer._combine_arrivals(*args)
        assert got.shape == (t_local, hidden)
        assert got.dtype == jnp.bfloat16
        np.testing.assert_allclose(
            np.asarray(got, dtype=np.float32), expected, rtol=1e-2, atol=1e-2
        )

        # Within TPU v7x SMEM budget: must select build_combine_kernel.
        monkeypatch.setattr(
            pltpu,
            "get_tpu_info",
            lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1),
        )
        recorded = {}
        sentinel = jnp.full((t_local, hidden), 3.0, dtype=jnp.bfloat16)

        def _fake_builder(**kwargs):
            recorded["kwargs"] = kwargs
            return lambda arrivals, coef, rows: sentinel

        monkeypatch.setattr(layer, "build_combine_kernel", _fake_builder)
        assert layer._combine_arrivals(*args) is sentinel
        assert recorded["kwargs"] == {
            "t_local": t_local,
            "topk": topk,
            "hidden": hidden,
            "lane_blocks": lane_blocks,
            "wire_dtype": jnp.float32,
            "out_dtype": jnp.bfloat16,
        }
