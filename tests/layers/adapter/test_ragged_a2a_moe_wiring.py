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
"""Op wiring of the ragged all-to-all EP exchange."""

from types import SimpleNamespace

import pytest
import vllm.config

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter import ragged_a2a_moe as ragged_a2a
from vllm_torchtpu.layers.core import fused_moe_gmm


def _fake_vllm_config(max_num_seqs, compile_sizes, dp, draft_tokens=0):
    spec = None
    if draft_tokens:
        spec = SimpleNamespace(num_speculative_tokens=draft_tokens)
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        speculative_config=spec,
        compilation_config=SimpleNamespace(compile_sizes=compile_sizes),
        parallel_config=SimpleNamespace(data_parallel_size=dp),
    )


@pytest.mark.parametrize(
    "max_num_seqs,compile_sizes,draft_tokens,expected",
    [
        # 48 requests per engine, 2 engines: two 48-token regions x top-10.
        (48, [48, 1024], 0, 960),
        # 12 requests x (1 + 3 draft tokens) per engine.
        (12, [48, 1024], 3, 960),
        # Below the SparseCore block the auto threshold stands.
        (12, [12, 1024], 0, 511),
    ],
)
def test_ragged_a2a_op_threshold_covers_the_ep_decode_step(
    monkeypatch, max_num_seqs, compile_sizes, draft_tokens, expected
):
    monkeypatch.setattr(envs, "ONEHOT_MOE_PERMUTE_THRESHOLD", None, raising=False)
    monkeypatch.setattr(envs, "TPU_MOE_A2A_CAPACITY", 2.0, raising=False)
    monkeypatch.setattr(envs, "TPU_MOE_A2A_ROUTE_CAPACITY", 3.0, raising=False)
    sc_info = SimpleNamespace(num_lanes=16, num_cores=2, num_subcores=16)
    monkeypatch.setattr(
        fused_moe_gmm.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(sparse_core=sc_info),
    )
    cfg = _fake_vllm_config(max_num_seqs, compile_sizes, 2, draft_tokens)
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: cfg)
    monkeypatch.setattr(
        ragged_a2a, "get_dp_tp_mesh", lambda: SimpleNamespace(shape={"dp": 2, "tp": 4})
    )
    monkeypatch.setattr(ragged_a2a, "_OPS", {})
    seen = {}

    def _ragged_a2a_moe(*args, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(ragged_a2a, "ragged_a2a_moe", _ragged_a2a_moe)

    class _Op:
        def register_fake(self, fn):
            pass

    def _jax_op(name, fn, **kwargs):
        fn(*([None] * 8))
        return _Op()

    monkeypatch.setattr(ragged_a2a.pallas, "jax_op", _jax_op)
    ragged_a2a._get_op(
        topk=10, activation="silu", rhs_quant_dtype=None, skip_padded_tokens=False
    )
    assert seen["onehot_moe_permute_threshold"] == expected
    assert seen["capacity_factor"] == 2.0
    assert seen["route_factor"] == 3.0


@pytest.mark.parametrize(
    "placement,rows,routes,expected",
    [
        ("", None, None, (2.5, 3.0)),
        ("/tmp/placement.npz", None, None, (2.0, 2.0)),
        # Explicit factors win either way.
        ("/tmp/placement.npz", "2.5", "3", (2.5, 3.0)),
        ("", "1.5", "0", (1.5, 0.0)),
    ],
)
def test_capacity_defaults_follow_the_expert_placement(
    monkeypatch, placement, rows, routes, expected
):
    monkeypatch.setenv("TPU_MOE_EXPERT_PLACEMENT", placement)
    for name, value in (
        ("TPU_MOE_A2A_CAPACITY", rows),
        ("TPU_MOE_A2A_ROUTE_CAPACITY", routes),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    # Read the knobs themselves: a monkeypatched attribute restored by an
    # earlier test would shadow the module's lazy lookup.
    got = tuple(
        envs.environment_variables[name]()
        for name in ("TPU_MOE_A2A_CAPACITY", "TPU_MOE_A2A_ROUTE_CAPACITY")
    )
    assert got == expected
