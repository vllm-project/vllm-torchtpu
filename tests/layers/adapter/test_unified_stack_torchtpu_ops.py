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
"""Verify torch_tpu custom-op registration and Dynamo export for the decode
optimization stack:

1. ``cp_mla_attention._ensure_online_lse_combine_op`` (all 4 ``sharded_jax_op``
   signature branches + dynamic-batch ``torch._dynamo.export``).
2. ``VllmTPUSparseAttnIndexer._build_streamidx_dcp_qb_absorb_op`` across all
   ``qb_mode`` and lane modes (``pcp_streaming_jax_op``).
"""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
import torch
from jax.sharding import Mesh

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed import dcp
from vllm_torchtpu.layers.adapter import cp_mla_attention
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op

pytestmark = [pytest.mark.tpu_test, pytest.mark.multichip]


class _FakeGroup:
    def __init__(self, ranks, rank_in_group=0):
        self.ranks = list(ranks)
        self.world_size = len(ranks)
        self.rank_in_group = rank_in_group
        self.cpu_group = None


@pytest.mark.parametrize("multi_lane", [False, True])
@pytest.mark.parametrize("use_live", [False, True])
def test_online_lse_combine_sharded_jax_op_registration_and_dynamo(
    monkeypatch, multi_lane, use_live
):
    devs = jax.devices()
    if len(devs) < 8:
        pytest.skip(f"needs >= 8 JAX devices, got {len(devs)}")

    cp_mla_attention._ONLINE_COMBINE_OPS.clear()
    cp_mla_attention._DCP_CHIP_PAIRS_CACHE.clear()
    monkeypatch.setattr(
        cp_mla_attention, "_DCP_RANK", torch.zeros(1, dtype=torch.int32)
    )
    monkeypatch.setattr(
        cp_mla_attention, "_DCP_LANE", torch.zeros(1, dtype=torch.int32)
    )
    monkeypatch.setenv("TPU_DCP_ONLINE_LSE_COMBINE", "flat")

    if multi_lane:
        world = 4
        mesh = Mesh(
            np.array(devs[:8]).reshape(2, 4),
            (dcp.DCP_LANE_AXIS_NAME, dcp.DCP_AXIS_NAME),
        )
    else:
        world = 8
        mesh = Mesh(np.array(devs[:8]), (dcp.DCP_AXIS_NAME,))

    fake_group = _FakeGroup(range(world), rank_in_group=0)
    monkeypatch.setattr(dcp, "get_or_create_dcp_mesh", lambda *a, **k: mesh)
    monkeypatch.setattr(cp_mla_attention, "get_dcp_group", lambda: fake_group)

    op, got_multi_lane = cp_mla_attention._ensure_online_lse_combine_op(
        world, use_live=use_live, group=fake_group
    )
    assert got_multi_lane is multi_lane

    t, h, d = 16, 64, 512
    o = torch.zeros(t, h, d, dtype=torch.bfloat16)
    lse = torch.zeros(t, h, dtype=torch.float32)
    rank = torch.zeros(1, dtype=torch.int32)
    lane = torch.zeros(1, dtype=torch.int32)
    live = torch.tensor([t], dtype=torch.int32)
    torch._dynamo.mark_dynamic(o, 0)
    torch._dynamo.mark_dynamic(lse, 0)

    if multi_lane and use_live:
        exported = torch._dynamo.export(lambda a, b: op(a, b, live, rank, lane))(o, lse)
    elif multi_lane:
        exported = torch._dynamo.export(lambda a, b: op(a, b, rank, lane))(o, lse)
    elif use_live:
        exported = torch._dynamo.export(lambda a, b: op(a, b, live, rank))(o, lse)
    else:
        exported = torch._dynamo.export(lambda a, b: op(a, b, rank))(o, lse)
    assert exported is not None


@pytest.mark.parametrize("qb_mode", ["pallas", "pallas_hbm", "prequant", "xla"])
@pytest.mark.parametrize("use_lane", [False, True])
def test_indexer_qb_absorb_pcp_streaming_jax_op_registration(
    monkeypatch, qb_mode, use_lane
):
    devs = jax.devices()
    if len(devs) < 8:
        pytest.skip(f"needs >= 8 JAX devices, got {len(devs)}")

    monkeypatch.setenv("TPU_DCP_INDEXER_A2A_MERGE", "1")
    monkeypatch.setenv("TPU_DCP_INDEXER_A2A_SMALL_COLL", "1" if use_lane else "0")
    monkeypatch.setenv("TPU_INDEXER_OVERLAP_QB_ABSORB", "1")
    monkeypatch.setenv("TPU_DCP_RANK_OPERAND", "1")
    monkeypatch.setenv("TPU_DCP_INDEXER_SMALL_AG", "1" if use_lane else "0")

    if use_lane:
        dcp_size = 4
        mesh = Mesh(
            np.array(devs[:8]).reshape(2, 4),
            (dcp.DCP_LANE_AXIS_NAME, dcp.DCP_AXIS_NAME),
        )
    else:
        dcp_size = 8
        mesh = Mesh(np.array(devs[:8]), (dcp.DCP_AXIS_NAME,))

    monkeypatch.setattr(dcp, "get_or_create_dcp_mesh", lambda *a, **k: mesh)
    monkeypatch.setattr(mla_op, "get_or_create_dcp_mesh", lambda *a, **k: mesh)
    monkeypatch.setattr(
        mla_op, "dcp_rank_operand", lambda *a, **k: torch.zeros(1, dtype=torch.int32)
    )
    monkeypatch.setattr(
        mla_op, "dcp_lane_operand", lambda *a, **k: torch.zeros(1, dtype=torch.int32)
    )
    assert envs.TPU_INDEXER_OVERLAP_QB_ABSORB

    fake_self = SimpleNamespace(
        topk_tokens=2048,
        _shared_meta=False,
        k_cache=SimpleNamespace(prefix=f"model.layers.0.idx_{qb_mode}_{int(use_lane)}"),
    )
    op = mla_op.VllmTPUSparseAttnIndexer._build_streamidx_dcp_qb_absorb_op(
        fake_self, dcp_size=dcp_size, interleave_size=128, qb_mode=qb_mode
    )
    assert callable(op)
