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
the per-layout step gate, the meeting point with the few-token kernel, and the
copy-slot slice of a placed layer's weights."""

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


def _prebuild(layer):
    with patch.object(v3, "ep_mesh_index", return_value=0):
        v3.prebuild_fused_moe_ep(layer, topk=2, renormalize=True, activation="silu")
    return getattr(layer, "_tpu_fused_ep_eligible", False)


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
