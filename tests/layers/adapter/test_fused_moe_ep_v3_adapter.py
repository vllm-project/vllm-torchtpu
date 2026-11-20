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
"""Host-side rules of the fused EP v3 bridge (layers/adapter/fused_moe_ep_v3.py):
the per-layout step gate, the meeting point with the few-token kernel, the
copy-slot slice of a placed layer's weights, and the routing it admits."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.distributed import parallel_state

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter import expert_placement
from vllm_torchtpu.layers.adapter import fused_moe_ep_v3 as v3

pytestmark = pytest.mark.cpu_test

EP = 2


@pytest.fixture
def groups(monkeypatch):
    def set_layout(dp, tp):
        monkeypatch.setattr(
            parallel_state, "model_parallel_is_initialized", lambda: True
        )
        monkeypatch.setattr(
            parallel_state, "get_dp_group", lambda: SimpleNamespace(world_size=dp)
        )
        monkeypatch.setattr(
            parallel_state, "get_tp_group", lambda: SimpleNamespace(world_size=tp)
        )

    return set_layout


def test_step_gate_defaults_before_the_groups_exist(monkeypatch):
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: False)
    assert v3.step_min_tokens() == v3.DEFAULT_STEP_MIN_TOKENS == 64


@pytest.mark.parametrize(
    "layout,gate", [((4, 2), 32), ((1, 8), 33), ((2, 4), 64), ((1, 1), 64)]
)
def test_step_gate_follows_the_layout(groups, layout, gate):
    groups(*layout)
    assert v3.step_min_tokens() == gate
    explicit = SimpleNamespace(
        data_parallel_size=layout[0], tensor_parallel_size=layout[1]
    )
    assert v3.step_min_tokens(explicit) == gate


def test_no_few_token_kernel_means_no_meeting_point(groups):
    groups(1, 1)
    v3.check_kernel_threshold(ep=8, tp=1)
    v3.check_kernel_threshold(ep=8, tp=1, lowtok_cap=0)


@pytest.mark.parametrize("cap,meets", [(8 * 63, True), (8 * 63 - 1, False)])
def test_few_token_cap_must_reach_the_fused_gate_across_the_node(groups, cap, meets):
    """At tp 1 an engine's rows below the gate (64) are gathered from all 8
    engines, so the few-token kernel must cover 8 * 63 rows."""
    groups(1, 1)
    if meets:
        v3.check_kernel_threshold(ep=8, tp=1, lowtok_cap=cap)
    else:
        with pytest.raises(ValueError, match="leave steps between the two kernels"):
            v3.check_kernel_threshold(ep=8, tp=1, lowtok_cap=cap)


@pytest.mark.parametrize("threshold", [0, 256])
def test_tensor_parallel_gate_refuses_a_first_sliced_bucket_below_it(
    groups, monkeypatch, threshold
):
    """TP8 (gate 33) at a sequence-parallel threshold of 0 (the default: every
    sequence-parallel step is sliced, so the first sliced bucket is one row)
    or 256 (32 rows per rank): the first sliced bucket is below the gate, so
    a few-token cap is refused."""
    from vllm_torchtpu.layers.adapter import sequence_parallel

    monkeypatch.setattr(sequence_parallel, "DEFAULT_MIN_ROWS", threshold)
    groups(1, 8)
    with pytest.raises(ValueError, match="leave steps between the two kernels"):
        v3.check_kernel_threshold(ep=8, tp=8, lowtok_cap=448)


def test_tensor_parallel_gate_is_checked_at_the_first_sliced_bucket(
    groups, monkeypatch
):
    """TP8 at a 512-row sequence-parallel threshold (the value the TP8 decode
    legs ran; the threshold defaults to 0 until a model follows each step's
    form): 64 rows per rank at the first sliced bucket, which meets the TP8
    gate of 33."""
    from vllm_torchtpu.layers.adapter import sequence_parallel

    monkeypatch.setattr(sequence_parallel, "DEFAULT_MIN_ROWS", 512)
    groups(1, 8)
    v3.check_kernel_threshold(ep=8, tp=8, lowtok_cap=448)


def _layer(slots, primaries=None, hidden=128, inter=256):
    layer = SimpleNamespace(
        w13_weight=torch.zeros(slots, hidden, 2 * inter, dtype=torch.float8_e4m3fn),
        w2_weight=torch.zeros(slots, inter, hidden, dtype=torch.float8_e4m3fn),
        w13_weight_scale_inv=torch.ones(slots, 1, 1, 2 * inter),
        w2_weight_scale_inv=torch.ones(slots, 1, 1, hidden),
        global_num_experts=4 * EP,
        custom_routing_function=None,
        scoring_func="softmax",
        e_score_correction_bias=None,
        use_grouped_topk=False,
        num_expert_group=None,
        topk_group=None,
        routed_scaling_factor=1.0,
        moe_config=SimpleNamespace(
            experts_per_token=2,
            is_sequence_parallel=False,
            moe_parallel_config=SimpleNamespace(
                use_ep=True, pcp_size=1, is_sequence_parallel=False
            ),
        ),
    )
    if primaries is not None:
        setattr(layer, expert_placement.PRIMARY_EXPERTS_ATTR, primaries)
    return layer


@pytest.fixture
def armed(monkeypatch):
    """Everything prebuild needs besides the layer, with _build_op recording
    the expert count it is built for."""
    built = {}
    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", True)
    monkeypatch.setattr(envs, "MOE_FUSED_EP_KERNEL_MIN_TOKENS", 1024)
    monkeypatch.setattr(v3, "_OP", None)
    monkeypatch.setattr(v3, "_OP_CONTRACT", None)
    monkeypatch.setattr(v3, "_RANK_BUFFER", None)
    monkeypatch.setattr(v3, "_EXPERT_ORDER", None)
    # prebuild records both process-wide.
    monkeypatch.setattr(v3, "_TP_SIZE", None)
    monkeypatch.setattr(v3, "_EP_SIZE", None)
    monkeypatch.setattr(
        v3, "build_ep_mesh", lambda: SimpleNamespace(shape={v3.EP_AXIS_NAME: EP})
    )
    monkeypatch.setattr(v3, "_max_node_tokens", lambda ep: 4096)
    monkeypatch.setattr(v3, "_expert_block_permutation", lambda ep, dev: None)

    def build_op(*args, local_experts, **kwargs):
        built["local_experts"] = local_experts
        v3._OP = "v3_op"

    monkeypatch.setattr(v3, "_build_op", build_op)
    return built


def _prebuild(layer, renormalize=True):
    with patch.object(v3, "ep_mesh_index", return_value=0):
        v3.prebuild_fused_moe_ep(
            layer, topk=2, renormalize=renormalize, activation="silu"
        )
    return getattr(layer, "_tpu_fused_ep_eligible", False)


def test_the_bridge_asks_the_kernel_for_its_lowering_by_argument(monkeypatch):
    """The bridge's op calls the kernel without its inner jit, without side
    effects and with the rank tensor as the shard index, as arguments of
    that call; the kernel package has no process-wide flags left that a
    bridge could set for every other caller."""
    from vllm_torchtpu.kernels.fused_moe.v3 import layer, layout

    calls = []
    monkeypatch.setattr(v3, "fused_ep_moe", lambda *a, **kw: calls.append(kw))
    run_moe = v3.run_moe_v3(
        SimpleNamespace(axis_names=(v3.MESH_AXIS,), shape={v3.MESH_AXIS: EP}),
        2,
        True,
        "silu",
        hidden=128,
        inter=256,
        experts_per_shard=4,
    )
    rank = object()
    run_moe(*[None] * 6, rank, None)
    (kwargs,) = calls
    assert kwargs["side_effects"] is False
    assert kwargs["wrap_in_jit"] is False
    assert kwargs["shard_index"] is rank
    for module, flag in (
        (layer, "WRAP_IN_JIT"),
        (layout, "SIDE_EFFECTS"),
        (layout, "SHARD_INDEX_OPERAND"),
    ):
        assert not hasattr(module, flag), flag


def test_a_placed_layer_streams_its_primaries_only(armed):
    """4 primaries and 2 copy slots per rank over EP 2: the kernel is built for
    the 4 primaries, which is what makes 8 experts split over 2 shards."""
    assert _prebuild(_layer(slots=6, primaries=4))
    assert armed["local_experts"] == 4


def test_without_the_attribute_every_slot_is_an_expert(armed):
    """The control: the same weights with no primary count are 6 experts per
    shard, which do not match the router's 8 experts over 2 shards."""
    assert not _prebuild(_layer(slots=6))
    assert "local_experts" not in armed


def test_an_unplaced_layer_is_unchanged(armed):
    assert _prebuild(_layer(slots=4))
    assert armed["local_experts"] == 4


def test_a_refused_contract_keeps_the_rank_and_the_expert_order(armed, monkeypatch):
    """The second layer's top_k differs from the operation's, so it is refused
    before it can replace the rank buffer or the expert order the first layer
    set."""
    order, other = torch.tensor([1, 0]), torch.tensor([0, 1])
    monkeypatch.setattr(v3, "_expert_block_permutation", lambda ep, dev: order)
    first = _layer(slots=4)
    with patch.object(v3, "ep_mesh_index", return_value=1):
        v3.prebuild_fused_moe_ep(first, topk=2, renormalize=True, activation="silu")
    assert first._tpu_fused_ep_eligible
    rank = v3._RANK_BUFFER
    assert rank.item() == 1 and v3._EXPERT_ORDER is order

    monkeypatch.setattr(v3, "_expert_block_permutation", lambda ep, dev: other)
    second = _layer(slots=4)
    with patch.object(v3, "ep_mesh_index", return_value=0):
        v3.prebuild_fused_moe_ep(second, topk=4, renormalize=True, activation="silu")
    assert not second._tpu_fused_ep_eligible
    assert v3._RANK_BUFFER is rank and rank.item() == 1
    assert v3._EXPERT_ORDER is order


def test_an_ep_mesh_index_outside_the_mesh_is_refused(armed):
    layer = _layer(slots=4)
    with patch.object(v3, "ep_mesh_index", return_value=EP):
        v3.prebuild_fused_moe_ep(layer, topk=2, renormalize=True, activation="silu")
    assert not layer._tpu_fused_ep_eligible
    assert v3._OP is None and v3._RANK_BUFFER is None


@pytest.mark.parametrize(("ep", "tp"), [(2, 2), (8, 2), (8, 8)])
def test_the_batch_floor_counts_one_scheduler_batch_per_dp_rank(
    armed, monkeypatch, ep, tp
):
    """Under TP with sequence parallelism the TP ranks of a DP rank split one
    scheduler batch, so the batch floor sees ep // tp batches, not ep: 4 at
    DP4 x TP2 (EP 8), 1 at TP8 (EP 8)."""
    from vllm_torchtpu.layers.adapter import sequence_parallel

    batches = []
    monkeypatch.setattr(v3, "_max_node_tokens", lambda n: batches.append(n) or 4096)
    monkeypatch.setattr(v3, "_tensor_parallel_size", lambda: tp)
    monkeypatch.setattr(sequence_parallel, "enabled", lambda: True)
    monkeypatch.setattr(
        v3, "build_ep_mesh", lambda: SimpleNamespace(shape={v3.EP_AXIS_NAME: ep})
    )
    layer = _layer(slots=4)
    layer.global_num_experts = 4 * ep
    layer.moe_config.is_sequence_parallel = True
    assert _prebuild(layer)
    assert batches == [ep // tp]


@pytest.mark.parametrize(
    "routing",
    [
        dict(scoring_func="sigmoid"),
        dict(scoring_func="sqrtsoftplus"),
        dict(e_score_correction_bias=torch.zeros(4 * EP)),
        dict(routed_scaling_factor=2.5),
    ],
    ids=["sigmoid", "sqrtsoftplus", "correction_bias", "routed_scaling"],
)
def test_routing_the_v3_router_cannot_compute_is_refused(armed, routing):
    """The v2 bridge's admission takes these (the v2 kernel has operands for
    them); the v3 router is a plain softmax with no bias or scale."""
    layer = _layer(slots=4)
    for name, value in routing.items():
        setattr(layer, name, value)
    assert not _prebuild(layer)
    assert "local_experts" not in armed


def test_simulated_routing_is_refused(armed, monkeypatch):
    """uniform_random simulation is served by the v2 kernel through routing
    operands; the v3 kernel routes by itself, so it is refused."""
    from vllm_torchtpu.layers.adapter import moe_routing

    monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", object())
    monkeypatch.setattr(moe_routing, "_strategy_name", "uniform_random")
    assert not _prebuild(_layer(slots=4))


def test_raw_router_weights_are_refused(armed):
    """The kernel refuses renormalize=False at trace time (no TPU case covers
    the raw weights), so the bridge declines such a layer up front, as it
    declines the other routing it cannot serve, and the layer keeps the
    grouped-matmul path."""
    assert not _prebuild(_layer(slots=4), renormalize=False)
    assert "local_experts" not in armed


def test_the_softmax_router_is_admitted(armed):
    """The control: softmax scoring, no bias, scale 1.0 and no simulation,
    the Qwen3.5 router."""
    assert _prebuild(_layer(slots=4))
