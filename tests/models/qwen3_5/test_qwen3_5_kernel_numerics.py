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
"""The Qwen3.5 rule for kernel numerics and code paths: the quantize
helpers' reciprocal scale, the grouped matmul's hand-streamed whole-expert
slabs and its tuned FP8 MoE tiling and rhs-scale hoist for Qwen3.5, main's
arithmetic and code for every other model."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu import patch_registry
from vllm_torchtpu.kernels.megablox import gmm_v2
from vllm_torchtpu.kernels.quantized_matmul import util
from vllm_torchtpu.models.qwen3_5 import hooks

pytestmark = pytest.mark.cpu_test

QWEN35 = "Qwen3_5MoeForConditionalGeneration"
TARGET = "vllm_torchtpu.models.qwen3_5.hooks:apply_kernel_numerics"


@pytest.fixture(autouse=True)
def _mains_default(monkeypatch):
    monkeypatch.setattr(util, "RECIPROCAL_SCALE", False)
    monkeypatch.setattr(gmm_v2, "RHS_STREAM", False)
    monkeypatch.setattr(gmm_v2, "TUNED_FP8_MOE", False)


def _model(arch):
    # The checkpoint shape the rule was measured on (Qwen3.5-397B-A17B).
    text = SimpleNamespace(**hooks._CALIBRATED_SHAPE)
    return SimpleNamespace(architectures=[arch], hf_text_config=text)


@pytest.fixture(autouse=True)
def _tpu7x(monkeypatch):
    monkeypatch.setattr(hooks, "_chip_family", lambda: "7x")


def test_qwen3_5_takes_the_reciprocal_scale_and_streamed_slabs():
    assert hooks.apply_kernel_numerics(_model(QWEN35)) is None
    assert util.RECIPROCAL_SCALE is True
    assert gmm_v2.RHS_STREAM is True
    assert gmm_v2.TUNED_FP8_MOE is True


@pytest.mark.parametrize("arch", ["DeepseekV3ForCausalLM", "Qwen3MoeForCausalLM"])
def test_other_models_keep_mains_arithmetic_and_code(arch):
    """False leaves the patch pending: the registry marks a patch done by its
    target, not by model."""
    assert hooks.apply_kernel_numerics(_model(arch)) is False
    assert util.RECIPROCAL_SCALE is False
    assert gmm_v2.RHS_STREAM is False
    assert gmm_v2.TUNED_FP8_MOE is False


def test_qwen3_5_on_another_device_keeps_mains_arithmetic_and_code(monkeypatch):
    monkeypatch.setattr(hooks, "_chip_family", lambda: "v6e")
    assert hooks.apply_kernel_numerics(_model(QWEN35)) is False
    assert util.RECIPROCAL_SCALE is False
    assert gmm_v2.RHS_STREAM is False
    assert gmm_v2.TUNED_FP8_MOE is False


def test_a_config_without_architectures_keeps_mains_arithmetic_and_code():
    """A config that names no architectures is not a Qwen3.5 config."""
    stand_in = SimpleNamespace(
        architectures=None, hf_config=SimpleNamespace(model_type="llama")
    )
    assert hooks.apply_kernel_numerics(stand_in) is False
    assert util.RECIPROCAL_SCALE is False
    assert gmm_v2.RHS_STREAM is False
    assert gmm_v2.TUNED_FP8_MOE is False


def test_waits_for_a_model_config():
    assert hooks.apply_kernel_numerics(None) is False
    assert util.RECIPROCAL_SCALE is False
    assert gmm_v2.RHS_STREAM is False
    assert gmm_v2.TUNED_FP8_MOE is False


def test_a_later_qwen3_5_load_in_the_process_still_takes_them(monkeypatch):
    """Through the patch registry: another model's load at platform activation
    leaves the patch pending, so the Qwen3.5-397B-A17B load that follows in
    the same process turns the levers on."""
    (patch,) = [p for p in patch_registry.PATCHES if p.target == TARGET]
    monkeypatch.setattr(patch_registry, "PATCHES", (patch,))
    monkeypatch.setattr(patch_registry, "_applied", set())
    patch_registry.apply(
        "platform_activation", model_config=_model("DeepseekV3ForCausalLM")
    )
    assert TARGET not in patch_registry._applied
    assert util.RECIPROCAL_SCALE is False
    patch_registry.apply("model_load", model_config=_model(QWEN35))
    assert TARGET in patch_registry._applied
    assert util.RECIPROCAL_SCALE is True
    assert gmm_v2.RHS_STREAM is True
    assert gmm_v2.TUNED_FP8_MOE is True
