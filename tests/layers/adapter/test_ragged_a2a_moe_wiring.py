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
from unittest import mock

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
    "placement,dp_size,placed,rows,routes,expected",
    [
        # Up to 8 members: x2.5 / x3, or x2 / x2 for every layer of a placement.
        ("", 8, False, None, None, (2.5, 3.0)),
        ("/tmp/placement.npz", 8, True, None, None, (2.0, 2.0)),
        ("/tmp/placement.npz", 8, False, None, None, (2.0, 2.0)),
        # Larger groups: layers the file places, and layers in linear placement
        # with or without a placement file.
        ("/tmp/placement.npz", 16, True, None, None, (4.0, 4.0)),
        ("/tmp/placement.npz", 16, False, None, None, (4.0, 5.0)),
        ("", 16, False, None, None, (4.0, 5.0)),
        ("/tmp/placement.npz", 32, True, None, None, (7.0, 7.0)),
        ("/tmp/placement.npz", 32, False, None, None, (7.0, 8.5)),
        ("", 32, False, None, None, (7.0, 8.5)),
        # Explicit factors win, each on its own.
        ("/tmp/placement.npz", 16, True, 2.5, 3.0, (2.5, 3.0)),
        ("/tmp/placement.npz", 32, False, 3.5, None, (3.5, 8.5)),
        ("", 8, False, 1.5, 0.0, (1.5, 0.0)),
    ],
)
def test_capacity_defaults_follow_the_group_size_and_placement(
    placement, dp_size, placed, rows, routes, expected
):
    # Patched with create=True and removed afterwards, so later tests read the
    # environment again (see envs.__getattr__).
    with mock.patch.multiple(
        envs,
        create=True,
        TPU_MOE_EXPERT_PLACEMENT=placement,
        TPU_MOE_A2A_CAPACITY=rows,
        TPU_MOE_A2A_ROUTE_CAPACITY=routes,
    ):
        assert ragged_a2a.capacity_factors(dp_size, placed=placed) == expected


def test_placed_and_linear_layers_get_their_own_ops_and_factors(monkeypatch):
    """With a placement at 16 members (DP16 x TP4, 28 requests x 4 tokens per
    engine), a placed layer and a layer left in linear placement (the bf16
    draft layer) get separate ops, each built and logged with its factors."""
    sc_info = SimpleNamespace(num_lanes=16, num_cores=2, num_subcores=16)
    monkeypatch.setattr(
        fused_moe_gmm.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(sparse_core=sc_info),
    )
    cfg = _fake_vllm_config(28, [112, 1024], 16, draft_tokens=3)
    cfg.model_config = SimpleNamespace(hf_text_config=SimpleNamespace(num_experts=512))
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: cfg)
    monkeypatch.setattr(
        ragged_a2a,
        "get_dp_tp_mesh",
        lambda: SimpleNamespace(shape={"dp": 16, "tp": 4}),
    )
    monkeypatch.setattr(ragged_a2a, "_OPS", {})
    factors, lines = [], []

    def _ragged_a2a_moe(*args, **kwargs):
        factors.append((kwargs["capacity_factor"], kwargs["route_factor"]))

    monkeypatch.setattr(ragged_a2a, "ragged_a2a_moe", _ragged_a2a_moe)
    monkeypatch.setattr(
        ragged_a2a,
        "logger",
        SimpleNamespace(
            info=lambda msg, *args: lines.append(msg % args),
            info_once=lambda *args: None,
        ),
    )

    class _Op:
        def register_fake(self, fn):
            pass

    def _jax_op(name, fn, **kwargs):
        fn(*range(len(kwargs["input_partition_specs"])))
        return _Op()

    monkeypatch.setattr(ragged_a2a.pallas, "jax_op", _jax_op)
    with mock.patch.multiple(
        envs,
        create=True,
        TPU_MOE_EXPERT_PLACEMENT="/tmp/placement.npz",
        TPU_MOE_A2A_CAPACITY=None,
        TPU_MOE_A2A_ROUTE_CAPACITY=None,
        USE_MOE_SPARSE_CORE=True,
        ONEHOT_MOE_PERMUTE_THRESHOLD=None,
    ):
        ragged_a2a.prebuild_ragged_a2a_moe(topk=10, activation="silu", placed=True)
        ragged_a2a.prebuild_ragged_a2a_moe(topk=10, activation="silu", scaled=False)
    assert len(ragged_a2a._OPS) == 2
    assert factors == [(4.0, 4.0), (4.0, 5.0)]
    assert lines == [
        "Ragged A2A MoE capacity x4 (routes x4) | decode step: 1056 of 1792 rows, "
        "1152 of 17920 routes per rank",
        "Ragged A2A MoE capacity x4 (routes x5) | decode step: 1056 of 1792 rows, "
        "1408 of 17920 routes per rank",
    ]


def test_op_never_skips_routes_by_weight(monkeypatch):
    """Padded tokens reach the op unrouted (TPU_MOE_SKIP_PADDED_TOKENS drops
    them at top-k), so the prebuilt op and the one each call uses are the op
    without zero-weight skipping, with or without the switch."""
    import torch

    monkeypatch.setenv("TPU_MOE_SKIP_PADDED_TOKENS", "1")
    built = []

    def _get_op(**kwargs):
        built.append(kwargs["skip_padded_tokens"])
        return lambda *operands: operands[0]

    monkeypatch.setattr(ragged_a2a, "_get_op", _get_op)
    ragged_a2a.prebuild_ragged_a2a_moe(topk=2, activation="silu")
    x = torch.zeros(4, 8)
    ragged_a2a.ragged_a2a_fused_moe_gmm(
        x,
        torch.zeros(2, 8, 8),
        torch.zeros(2, 4, 8),
        torch.zeros(2, 1, 8),
        torch.zeros(2, 1, 8),
        None,
        None,
        torch.zeros(4, 2),
        torch.full((4, 2), -1, dtype=torch.int32),
        torch.zeros((), dtype=torch.int32),
        topk=2,
        activation="silu",
    )
    assert built == [False, False]


@pytest.mark.cpu_test
@pytest.mark.parametrize("routing_dma", [False, True])
def test_routing_dma_knob_reaches_the_exchange(monkeypatch, routing_dma):
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: None)
    monkeypatch.setattr(
        ragged_a2a, "get_dp_tp_mesh", lambda: SimpleNamespace(shape={"dp": 16, "tp": 4})
    )
    monkeypatch.setattr(ragged_a2a, "_OPS", {})
    monkeypatch.setattr(ragged_a2a, "_routed_rows_per_decode_step", lambda topk: 0)
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
    monkeypatch.setenv("TPU_MOE_RAGGED_A2A", "1")
    if not routing_dma:
        monkeypatch.setenv("TPU_DECODE_FAST_PATHS_OFF", "a2a_routing_dma")
    ragged_a2a._get_op(
        topk=10, activation="silu", rhs_quant_dtype=None, skip_padded_tokens=False
    )
    assert seen["routing_dma"] is routing_dma
