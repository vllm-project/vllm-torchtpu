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
"""The float32 RMSNorm weight copy made at load gives bit-identical results."""

import contextlib

import pytest
import torch
from vllm import ir
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm_torchtpu.layers.adapter import layernorm as tpu_layernorm
from vllm_torchtpu.layers.adapter.ir_ops import TPU_IR_PROVIDER

HIDDEN = 256
EPS = 1e-5


@contextlib.contextmanager
def _priority(providers: list[str]):
    with (
        ir.ops.rms_norm.set_priority(providers),
        ir.ops.fused_add_rms_norm.set_priority(providers),
    ):
        yield


@pytest.fixture(autouse=True)
def _vllm_config():
    # CustomOp.__init__ reads the compilation config to pick its forward.
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        yield


def _norm(hidden: int = HIDDEN, seed: int = 0, **kwargs) -> RMSNorm:
    norm = RMSNorm(hidden, eps=EPS, dtype=torch.bfloat16, **kwargs)
    g = torch.Generator().manual_seed(seed)
    if norm.has_weight:
        # Weights away from 1.0, so that a wrong weight dtype shows in the output.
        norm.weight.data.copy_(1.0 + 0.3 * torch.randn(hidden, generator=g))
    return norm


def _inputs(hidden: int = HIDDEN) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    x = (3.0 * torch.randn(16, hidden, generator=g)).to(torch.bfloat16)
    residual = torch.randn(16, hidden, generator=g).to(torch.bfloat16)
    return x, residual


def _assert_bitwise_equal(a, b):
    if isinstance(a, tuple):
        assert isinstance(b, tuple) and len(a) == len(b)
        for x, y in zip(a, b):
            _assert_bitwise_equal(x, y)
        return
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))


def test_rmsnorm_is_replaced_by_tpu_subclass():
    norm = RMSNorm(HIDDEN, eps=EPS)
    assert type(norm) is tpu_layernorm.TpuRMSNorm
    assert isinstance(norm, RMSNorm)


@pytest.mark.parametrize("var_hidden_size", [None, 128])
@pytest.mark.parametrize("with_residual", [False, True])
def test_armed_forward_is_bitwise_identical(with_residual, var_hidden_size):
    x, residual = _inputs()
    reference = _norm(var_hidden_size=var_hidden_size)
    armed = _norm(var_hidden_size=var_hidden_size)
    with _priority([TPU_IR_PROVIDER, "native"]):
        assert tpu_layernorm.arm_fp32_weights(armed) == 1
        assert armed.weight_fp32.dtype == torch.float32
        assert torch.equal(armed.weight_fp32, armed.weight.data.float())
        args = (x.clone(), residual.clone()) if with_residual else (x.clone(),)
        expected = reference.forward_native(*args)
        args = (x.clone(), residual.clone()) if with_residual else (x.clone(),)
        actual = armed.forward_native(*args)
    _assert_bitwise_equal(actual, expected)


def test_armed_forward_passes_the_float32_buffer(monkeypatch):
    """Guards that the armed path is taken: the IR op receives the buffer."""
    seen: list[torch.dtype] = []
    impl = ir.ops.rms_norm.impls[TPU_IR_PROVIDER]
    original = impl.impl_fn

    def spy(x, weight, epsilon, variance_size=None):
        seen.append(weight.dtype)
        return original(x, weight, epsilon, variance_size)

    monkeypatch.setattr(impl, "impl_fn", spy)
    norm = _norm()
    x, _ = _inputs()
    with _priority([TPU_IR_PROVIDER, "native"]):
        norm.forward_native(x)
        tpu_layernorm.arm_fp32_weights(norm)
        norm.forward_native(x)
    assert seen == [torch.bfloat16, torch.float32]


def test_not_armed_when_native_provider_is_first():
    """``native`` rounds to ``weight.dtype`` first; a float32 weight would
    change its result, so the copy must not be made."""
    norm = _norm()
    with _priority(["native"]):
        assert tpu_layernorm.arm_fp32_weights(norm) == 0
    assert getattr(norm, "weight_fp32", None) is None


def test_arm_skips_ineligible_and_shared_modules():
    shared = _norm(seed=2)
    model_a = torch.nn.ModuleList([_norm(seed=3), shared])
    model_b = torch.nn.ModuleList([shared, _norm(seed=4)])
    weightless = _norm(has_weight=False)
    fp32 = RMSNorm(HIDDEN, eps=EPS, dtype=torch.float32)
    with _priority([TPU_IR_PROVIDER, "native"]):
        assert (
            tpu_layernorm.arm_fp32_weights(model_a, None, model_b, weightless, fp32)
            == 3
        )
        # A second call finds every module armed already.
        assert tpu_layernorm.arm_fp32_weights(model_a, model_b) == 0
    assert getattr(weightless, "weight_fp32", None) is None
    assert getattr(fp32, "weight_fp32", None) is None


def test_fp32_copy_is_not_in_the_state_dict():
    norm = _norm()
    with _priority([TPU_IR_PROVIDER, "native"]):
        tpu_layernorm.arm_fp32_weights(norm)
    assert set(norm.state_dict()) == {"weight"}
