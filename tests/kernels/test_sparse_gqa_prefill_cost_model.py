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
"""Default work accounting derives tensor geometry and hardware rates."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.sparse_gqa_prefill.cost_model import CostModel, build


@pytest.fixture(autouse=True)
def target(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)


def shapes(
    q=8192, heads=24, kv=2, dim=256, dtype=jnp.float8_e4m3fn, pages=15561, budget=512
):
    S = jax.ShapeDtypeStruct
    packing = 4 // jnp.dtype(dtype).itemsize
    return (
        S((q, heads, dim), jnp.bfloat16),
        S((q, budget), jnp.int32),
        S((q, 158), jnp.int32),
        S((pages, 1664, (2 * kv + packing - 1) // packing, packing, dim), dtype),
        S((9,), jnp.int32),
    )


def test_six_work_costs_remain_monotone():
    model = CostModel(1.0, 0.01, 0.1, 2.0, 0.2, 0.3)
    assert model.tc_ms(128, 4) == 1 + 0.01 * 128 + 0.1 * 4
    assert model.sc_ms(8, 3) == 2 + 0.2 * 8 + 0.3 * 3
    assert model.tc_ms(128, 0) <= model.tc_ms(128, 1)


@pytest.mark.parametrize(
    "q,heads,kv,dim,dtype",
    [
        (256, 12, 1, 128, jnp.float8_e4m3fn),
        (8192, 24, 2, 256, jnp.float8_e4m3fn),
        (512, 8, 2, 128, jnp.bfloat16),
    ],
)
def test_logical_flops_and_allocation_match_independent_geometry(
    q, heads, kv, dim, dtype
):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    xs = shapes(q, heads, kv, dim, dtype)
    m = build(*xs, kv_heads=kv, ratio=4, info=info)
    head_group = ((heads // kv + 7) // 8) * 8
    assert (
        m.work["SC_logical_flops_per_group_tile"] == 4 * 8 * kv * head_group * 256 * dim
    )
    assert m.work["TC_logical_flops_per_visit"] == 4 * 256 * heads * 256 * dim
    assert (
        m.work["peak_bandwidth_bytes_per_second"]
        == info.mem_bw_bytes_per_second * info.num_cores
    )
    assert (
        m.work["peak_bf16_ops_per_second"] == info.bf16_ops_per_second * info.num_cores
    )
    assert m.work["zeroed_slots"] == 2 and m.work["microchunks"] == (q + 63) // 64
    assert (
        m.work["page_bytes"]
        == 1664 * xs[3].shape[2] * xs[3].shape[3] * dim * jnp.dtype(dtype).itemsize
    )
    assert (
        m.work["producer_read_write_bytes"]
        == 2 * m.work["microchunks"] * m.work["slot_bytes"]
    )
    assert not m.work["overlap_assumed"]
    if q == 8192:
        assert m.work["slot_bytes"] == 144 * 1024**2


def test_pool_capacity_does_not_change_cost_or_admit_a_workload():
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    a = build(*shapes(pages=79), kv_heads=2, ratio=4, info=info)
    b = build(*shapes(pages=15561), kv_heads=2, ratio=4, info=info)
    assert a == b and a.work == b.work
    slow = build(
        *shapes(),
        kv_heads=2,
        ratio=4,
        info=replace(info, mem_bw_bytes_per_second=info.mem_bw_bytes_per_second // 2),
    )
    assert slow.sc_fixed_ms > b.sc_fixed_ms and slow.tc_visit_ms > b.tc_visit_ms
