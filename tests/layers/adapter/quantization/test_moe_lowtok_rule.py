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
"""When the few-token MoE kernel serves a model, and up to how many rows.

It is a rule, not an option: the expert-parallel MoE layers of the checkpoint
and device it was measured on (Qwen3.5-397B-A17B on TPU7x), with a row cap per
measured expert-parallel size (EP8). Every other model, other Qwen3.5
checkpoints and other expert-parallel sizes included, keeps the default MoE
path, which is what main does without this kernel.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_torchtpu.layers.adapter.quantization import moe_lowtok_scores as scores
from vllm_torchtpu.layers.core import fused_moe_gmm
from vllm_torchtpu.models.qwen3_5 import hooks

pytestmark = pytest.mark.cpu_test

QWEN3_5 = "Qwen3_5MoeForConditionalGeneration"
# The checkpoint the kernel was measured on (Qwen3.5-397B-A17B's text config)
# and main's CI checkpoint (Qwen3.5-35B-A3B's).
QWEN3_5_397B = dict(hooks._CALIBRATED_SHAPE)
QWEN3_5_35B = {"num_experts": 256, "hidden_size": 2048, "num_hidden_layers": 40}


@pytest.fixture(autouse=True)
def _tpu7x(monkeypatch):
    monkeypatch.setattr(hooks, "_chip_family", lambda: "7x")


def _config(arch=QWEN3_5, ep=True, shape=None):
    text = SimpleNamespace(**(QWEN3_5_397B if shape is None else shape))
    return SimpleNamespace(
        model_config=SimpleNamespace(architectures=[arch], hf_text_config=text),
        parallel_config=SimpleNamespace(enable_expert_parallel=ep),
    )


@pytest.mark.parametrize(
    "config,expected",
    [
        (_config(), True),
        (_config(ep=False), False),
        (_config(shape=QWEN3_5_35B), False),
        (_config(arch="DeepseekV3ForCausalLM"), False),
        (SimpleNamespace(model_config=None, parallel_config=None), False),
    ],
    ids=[
        "qwen3_5_397b_ep",
        "qwen3_5_397b_no_ep",
        "qwen3_5_35b_ep",
        "other_model",
        "no_model",
    ],
)
def test_the_kernel_serves_the_measured_checkpoint_with_expert_parallelism_only(
    config, expected
):
    assert scores.enabled(config) is expected


@pytest.mark.parametrize("chip", ["v6e", "v5p", None])
def test_the_kernel_serves_the_measured_checkpoint_on_tpu7x_only(monkeypatch, chip):
    monkeypatch.setattr(hooks, "_chip_family", lambda: chip)
    assert scores.enabled(_config()) is False


@pytest.mark.parametrize("ep,cap", [(8, 448), (4, 0), (16, 0), (2, 0), (1, 0)])
def test_row_cap_only_at_the_measured_expert_parallel_size(ep, cap):
    assert scores.max_tokens(ep) == cap


def test_routed_ids_path_is_off_until_a_model_arms_it():
    """Main's grouped path is unchanged for a model the rule does not cover."""
    assert fused_moe_gmm._MOE_LOWTOK_MAX_TOKENS == 0


def _layer(ep_size=8):
    return SimpleNamespace(
        layer_name="model.layers.0.mlp.experts",
        moe_config=SimpleNamespace(
            ep_size=ep_size, moe_parallel_config=SimpleNamespace(use_ep=False)
        ),
    )


@pytest.fixture
def armed_state(monkeypatch):
    setter = Mock()
    monkeypatch.setattr(scores, "_ARMED_MAX_TOKENS", 0)
    monkeypatch.setattr(scores, "_LAYERS", [])
    monkeypatch.setattr(scores, "set_lowtok_max_tokens", setter)
    return setter


@pytest.mark.parametrize(
    "config,ep_size,cap",
    [
        (_config(), 8, 448),
        (_config(), 4, 0),
        (_config(shape=QWEN3_5_35B), 4, 0),
        (_config(shape=QWEN3_5_35B), 8, 0),
        (_config(arch="DeepseekV3ForCausalLM"), 8, 0),
    ],
    ids=["397b_ep8", "397b_ep4", "35b_ep4", "35b_ep8", "other_model_ep8"],
)
def test_prebuild_arms_the_cap_by_the_rule(
    monkeypatch, armed_state, config, ep_size, cap
):
    import vllm.config

    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: config)
    # This layer is not expert-parallel, so no op is built either way; what
    # the rule decides is whether the kernel is armed at all.
    assert scores.prebuild(SimpleNamespace(), _layer(ep_size), "silu") is None
    assert scores.armed_max_tokens() == cap
    if cap:
        armed_state.assert_called_once_with(cap)
    else:
        armed_state.assert_not_called()
        assert scores._LAYERS == []
