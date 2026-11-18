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
"""The Qwen3.5 rule for the quantized-matmul quantize arithmetic: the
reciprocal scale for the checkpoint and device it was measured on
(Qwen3.5-397B-A17B on TPU7x), main's arithmetic for every other model,
other Qwen3.5 checkpoints and devices included."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.kernels.quantized_matmul import util
from vllm_torchtpu.models.qwen3_5 import hooks

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _mains_default(monkeypatch):
    monkeypatch.setattr(util, "RECIPROCAL_SCALE", False)


def _model(arch, shape=None):
    # By default the checkpoint shape the rule was measured on
    # (Qwen3.5-397B-A17B).
    text = SimpleNamespace(**(hooks._CALIBRATED_SHAPE if shape is None else shape))
    return SimpleNamespace(architectures=[arch], hf_text_config=text)


@pytest.fixture(autouse=True)
def _tpu7x(monkeypatch):
    monkeypatch.setattr(hooks, "_chip_family", lambda: "7x")


def test_qwen3_5_quantizes_with_the_reciprocal_scale():
    assert (
        hooks.apply_kernel_numerics(_model("Qwen3_5MoeForConditionalGeneration"))
        is None
    )
    assert util.RECIPROCAL_SCALE is True


@pytest.mark.parametrize("arch", ["DeepseekV3ForCausalLM", "Qwen3MoeForCausalLM"])
def test_other_models_keep_mains_arithmetic(arch):
    assert hooks.apply_kernel_numerics(_model(arch)) is None
    assert util.RECIPROCAL_SCALE is False


def test_main_ci_qwen3_5_checkpoint_keeps_mains_arithmetic():
    """Qwen3.5-35B-A3B, main's CI model: a Qwen3.5 architecture with another
    checkpoint shape."""
    qwen3_5_35b = {"num_experts": 256, "hidden_size": 2048, "num_hidden_layers": 40}
    model = _model("Qwen3_5MoeForConditionalGeneration", shape=qwen3_5_35b)
    assert hooks.apply_kernel_numerics(model) is None
    assert util.RECIPROCAL_SCALE is False


@pytest.mark.parametrize("chip", ["v6e", None])
def test_qwen3_5_397b_on_another_device_keeps_mains_arithmetic(monkeypatch, chip):
    monkeypatch.setattr(hooks, "_chip_family", lambda: chip)
    assert (
        hooks.apply_kernel_numerics(_model("Qwen3_5MoeForConditionalGeneration"))
        is None
    )
    assert util.RECIPROCAL_SCALE is False


def test_a_config_without_architectures_keeps_mains_arithmetic():
    stand_in = SimpleNamespace(hf_config=SimpleNamespace(model_type="llama"))
    assert hooks.apply_kernel_numerics(stand_in) is None
    assert util.RECIPROCAL_SCALE is False


def test_waits_for_a_model_config():
    assert hooks.apply_kernel_numerics(None) is False
    assert util.RECIPROCAL_SCALE is False
