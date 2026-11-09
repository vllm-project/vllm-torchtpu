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
"""TPU_NORM_WEIGHT_HOIST: GemmaRMSNorm gives the same outputs with and
without the precomputed float32 ``1 + weight`` buffer."""

import pytest
import torch
from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm

from vllm_torchtpu.layers.adapter import norm_weight_hoist
from vllm_torchtpu.layers.adapter.norm_weight_hoist import WEIGHT_PLUS_ONE_BUFFER

HIDDEN_SIZE = 64


@pytest.fixture(autouse=True)
def restore_gemma_rms_norm_class():
    """install() patches GemmaRMSNorm for the whole process; undo it."""
    original_forward_native = GemmaRMSNorm.forward_native
    yield
    GemmaRMSNorm.forward_native = original_forward_native


def _build_norms(custom_ops: str, num_norms: int = 2) -> torch.nn.Sequential:
    torch.manual_seed(0)
    config = VllmConfig(compilation_config=CompilationConfig(custom_ops=[custom_ops]))
    # CustomOp picks its forward from the config at construction.
    with set_current_vllm_config(config):
        norms = [GemmaRMSNorm(HIDDEN_SIZE, eps=1e-6) for _ in range(num_norms)]
    for norm in norms:
        # The model loads the weight in the bf16 params dtype.
        norm.weight.data = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16) * 0.1
    return torch.nn.Sequential(*norms)


def _run_norms(model: torch.nn.Sequential, x: torch.Tensor, residual: torch.Tensor):
    with set_current_vllm_config(VllmConfig()):
        return [(norm(x.clone()), norm(x.clone(), residual.clone())) for norm in model]


@pytest.mark.parametrize("custom_ops", ["all", "none"])
def test_outputs_match_with_and_without_hoist(custom_ops):
    model = _build_norms(custom_ops)
    x = torch.randn(8, HIDDEN_SIZE, dtype=torch.bfloat16)
    residual = torch.randn(8, HIDDEN_SIZE, dtype=torch.bfloat16)
    expected = _run_norms(model, x, residual)

    assert norm_weight_hoist.install(model) == 2

    actual = _run_norms(model, x, residual)
    for (plain, with_residual), (expected_plain, expected_with_residual) in zip(
        actual, expected
    ):
        torch.testing.assert_close(plain, expected_plain, rtol=0, atol=0)
        for out, expected_out in zip(with_residual, expected_with_residual):
            torch.testing.assert_close(out, expected_out, rtol=0, atol=0)


@pytest.mark.parametrize("custom_ops", ["all", "none"])
def test_forward_reads_the_buffer(custom_ops):
    model = _build_norms(custom_ops, num_norms=1)
    norm_weight_hoist.install(model)
    norm = model[0]
    buffer = getattr(norm, WEIGHT_PLUS_ONE_BUFFER)
    assert buffer.dtype == torch.float32
    torch.testing.assert_close(buffer, norm.weight.float() + 1.0, rtol=0, atol=0)
    assert WEIGHT_PLUS_ONE_BUFFER not in norm.state_dict()

    x = torch.randn(4, HIDDEN_SIZE, dtype=torch.bfloat16)
    buffer.zero_()
    with set_current_vllm_config(VllmConfig()):
        out = norm(x)
    # A zero weight zeroes the output only if the forward reads the buffer.
    assert torch.count_nonzero(out) == 0


def test_install_is_idempotent_and_refreshes_buffer():
    model = _build_norms("all")
    assert norm_weight_hoist.install(model) == 2
    with torch.no_grad():
        model[0].weight.add_(1.0)
    assert norm_weight_hoist.install(model) == 2
    assert [name for name, _ in model[0].named_buffers()] == [WEIGHT_PLUS_ONE_BUFFER]
    torch.testing.assert_close(
        getattr(model[0], WEIGHT_PLUS_ONE_BUFFER),
        model[0].weight.float() + 1.0,
        rtol=0,
        atol=0,
    )


def test_norms_without_buffer_keep_original_numerics():
    hoisted = _build_norms("all", num_norms=1)
    norm_weight_hoist.install(hoisted)
    # A norm built later (for example a draft model's) has no buffer.
    later = _build_norms("all", num_norms=1)
    x = torch.randn(4, HIDDEN_SIZE, dtype=torch.bfloat16)
    assert not hasattr(later[0], WEIGHT_PLUS_ONE_BUFFER)
    with set_current_vllm_config(VllmConfig()):
        torch.testing.assert_close(later[0](x), hoisted[0](x), rtol=0, atol=0)


def test_off_by_default_and_gated_by_env(monkeypatch):
    model = _build_norms("all")
    monkeypatch.delenv("TPU_NORM_WEIGHT_HOIST", raising=False)
    assert norm_weight_hoist.install_if_enabled(model) == 0
    assert not hasattr(model[0], WEIGHT_PLUS_ONE_BUFFER)
    assert GemmaRMSNorm.forward_native is not (
        norm_weight_hoist._forward_native_with_precomputed_weight
    )

    monkeypatch.setenv("TPU_NORM_WEIGHT_HOIST", "1")
    assert norm_weight_hoist.install_if_enabled(model) == 2
    assert hasattr(model[0], WEIGHT_PLUS_ONE_BUFFER)


def test_model_without_gemma_norms_is_untouched():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    assert norm_weight_hoist.install(model) == 0
    assert GemmaRMSNorm.forward_native is not (
        norm_weight_hoist._forward_native_with_precomputed_weight
    )
