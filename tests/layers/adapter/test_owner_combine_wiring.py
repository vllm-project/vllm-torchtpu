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
"""One owner-combine decision per layer, shared by the MoE runner (which
skips the TP all-reduce of a layer whose op already summed over TP) and the
ragged all-to-all op."""

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

from vllm_torchtpu import decode_switches
from vllm_torchtpu.layers.adapter import ragged_a2a_moe as ragged_a2a
from vllm_torchtpu.layers.adapter.custom_ops.moe_runner import VllmMoERunner

pytestmark = pytest.mark.cpu_test


def _runner(monkeypatch, **config):
    """A runner built through its own ``__init__`` over a stand-in base."""
    fields = dict(
        is_sequence_parallel=False, pcp_size=1, skip_final_all_reduce=False
    ) | config.pop("moe_config", {})
    scaling = config.pop("scaling", 1.0)
    transform = config.pop("transform", None)

    def base_init(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self.moe_config = SimpleNamespace(**fields)
        self.router = object()
        self.routed_output_transform = transform
        self.routed_scaling_factor = scaling
        self.routed_experts = SimpleNamespace(
            _ensure_moe_quant_config_init=lambda: None,
            quant_method=SimpleNamespace(is_monolithic=True),
        )

    monkeypatch.setattr(MoERunner, "__init__", base_init)
    return VllmMoERunner()


@pytest.fixture
def ragged_owner_line(monkeypatch):
    monkeypatch.setenv("TPU_MOE_RAGGED_A2A", "1")


@pytest.mark.parametrize(
    "kwargs,armed",
    [
        ({}, True),
        ({"scaling": 2.0}, False),
        ({"transform": torch.nn.Identity()}, False),
        ({"moe_config": {"pcp_size": 2}}, False),
        ({"moe_config": {"is_sequence_parallel": True}}, False),
        ({"moe_config": {"skip_final_all_reduce": True}}, False),
    ],
)
def test_runner_and_op_agree_for_every_layer(
    monkeypatch, ragged_owner_line, kwargs, armed
):
    runner = _runner(monkeypatch, **kwargs)
    layer = runner.routed_experts
    # Before weight load nothing is armed.
    assert runner._owner_combine_armed() is False
    # The quant method decides once, at weight load.
    assert decode_switches.arm_owner_combine(layer) is armed
    assert runner._owner_combine_armed() is armed
    if armed:
        assert runner._fused_output_is_reduced is True
    built = []
    monkeypatch.setattr(ragged_a2a, "_get_op", lambda **kw: built.append(kw))
    ragged_a2a.prebuild_ragged_a2a_moe(
        topk=2,
        activation="silu",
        owner_combine=getattr(layer, decode_switches.OWNER_COMBINE_ATTR),
    )
    # The op the layer calls is built with the runner's answer, and with the
    # shared-expert variant only where the runner passes the shared output.
    assert [(kw["owner_combine"], kw["shared"]) for kw in built] == (
        [(True, False), (True, True)] if armed else [(False, False)]
    )


def test_a_layer_without_the_exchange_is_never_armed(monkeypatch):
    monkeypatch.delenv("TPU_MOE_RAGGED_A2A", raising=False)
    runner = _runner(monkeypatch)
    assert decode_switches.arm_owner_combine(runner.routed_experts) is False
    assert runner._owner_combine_armed() is False


def test_a_layer_is_not_armed_with_the_owner_combine_path_off(
    monkeypatch, ragged_owner_line
):
    monkeypatch.setenv("TPU_DECODE_FAST_PATHS_OFF", "a2a_owner_combine")
    runner = _runner(monkeypatch)
    assert decode_switches.arm_owner_combine(runner.routed_experts) is False
    assert runner._owner_combine_armed() is False


def test_the_op_refuses_a_shared_output_the_layer_does_not_sum(monkeypatch):
    with pytest.raises(ValueError, match="shared_output needs the owner combine"):
        ragged_a2a.ragged_a2a_fused_moe_gmm(
            torch.zeros(4, 8),
            torch.zeros(2, 8, 8),
            torch.zeros(2, 4, 8),
            torch.zeros(2, 1, 8),
            torch.zeros(2, 1, 8),
            None,
            None,
            torch.zeros(4, 2),
            torch.zeros(4, 2, dtype=torch.int32),
            torch.zeros((), dtype=torch.int32),
            topk=2,
            activation="silu",
            shared_output=torch.zeros(4, 8),
            owner_combine=False,
        )
