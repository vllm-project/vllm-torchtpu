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
path, which is what main does without this kernel. Which layers the rule
arms at load is tested in tests/models/qwen3_5/test_qwen3_5_measured_levers.py.
"""

from types import SimpleNamespace

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
