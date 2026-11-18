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
"""Verify torch_tpu custom-op registration for the overlapped DCP indexer
``q_b + W_UK`` absorb path across all ``q_b`` quantization modes and lane
meshes."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
import torch
from jax.sharding import Mesh

from vllm_torchtpu import envs
from vllm_torchtpu.distributed import dcp
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op

pytestmark = [pytest.mark.tpu_test, pytest.mark.multichip]


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
