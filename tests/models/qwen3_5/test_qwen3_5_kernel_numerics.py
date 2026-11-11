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
helpers' reciprocal scale and the grouped matmul's hand-streamed whole-expert
slabs for Qwen3.5, main's arithmetic and code for every other model."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.kernels.megablox import gmm_v2
from vllm_torchtpu.kernels.quantized_matmul import util
from vllm_torchtpu.models.qwen3_5 import hooks

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _mains_default(monkeypatch):
    monkeypatch.setattr(util, "RECIPROCAL_SCALE", False)
    monkeypatch.setattr(gmm_v2, "RHS_STREAM", False)


def _model(arch):
    return SimpleNamespace(architectures=[arch])


def test_qwen3_5_takes_the_reciprocal_scale_and_streamed_slabs():
    assert (
        hooks.apply_kernel_numerics(_model("Qwen3_5MoeForConditionalGeneration"))
        is None
    )
    assert util.RECIPROCAL_SCALE is True
    assert gmm_v2.RHS_STREAM is True


@pytest.mark.parametrize("arch", ["DeepseekV3ForCausalLM", "Qwen3MoeForCausalLM"])
def test_other_models_keep_mains_arithmetic_and_code(arch):
    assert hooks.apply_kernel_numerics(_model(arch)) is None
    assert util.RECIPROCAL_SCALE is False
    assert gmm_v2.RHS_STREAM is False


def test_a_config_without_architectures_keeps_mains_arithmetic_and_code():
    """A config that does not name its architectures (a stand-in such as the
    patch registry's tests pass) is not a Qwen3.5 config."""
    stand_in = SimpleNamespace(hf_config=SimpleNamespace(model_type="llama"))
    assert hooks.apply_kernel_numerics(stand_in) is None
    assert util.RECIPROCAL_SCALE is False
    assert gmm_v2.RHS_STREAM is False


def test_waits_for_a_model_config():
    assert hooks.apply_kernel_numerics(None) is False
    assert util.RECIPROCAL_SCALE is False
    assert gmm_v2.RHS_STREAM is False
