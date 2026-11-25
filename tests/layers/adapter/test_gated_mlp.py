# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""The shared gated MLP and its SITU activation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

from vllm_torchtpu.layers.adapter import gated_mlp
from vllm_torchtpu.layers.adapter.gated_mlp import GatedMLP, SituAndMul

pytestmark = pytest.mark.usefixtures("cpu_model_tests")


@pytest.mark.parametrize("linear_beta", [None, 25.0])
def test_situ_and_mul(linear_beta: float | None) -> None:
    activation = SituAndMul(beta=4.0, linear_beta=linear_beta)
    inputs = torch.tensor([[-3.0, 0.5, 7.0, -30.0, 2.0, 40.0]])
    gate, up = inputs.chunk(2, dim=-1)
    expected_gate = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
    expected_up = (
        up if linear_beta is None else linear_beta * torch.tanh(up / linear_beta)
    )

    torch.testing.assert_close(activation(inputs), expected_gate * expected_up)


class _Linear(nn.Module):
    """A dense stand-in for vLLM's TP linear layers (TP size 1)."""

    def __init__(self, input_size, output_size, *args, **kwargs) -> None:
        super().__init__()
        del args, kwargs
        output_size = sum(output_size) if isinstance(output_size, list) else output_size
        self.weight = nn.Parameter(torch.randn(output_size, input_size))
        self.quant_method = SimpleNamespace(
            apply=lambda layer, x, bias: x @ layer.weight.T
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return x @ self.weight.T, None


@pytest.fixture
def dense_linears(monkeypatch: pytest.MonkeyPatch):
    """Dense linears, and the current vLLM config the activation op reads."""
    monkeypatch.setattr(gated_mlp, "MergedColumnParallelLinear", _Linear)
    monkeypatch.setattr(gated_mlp, "RowParallelLinear", _Linear)
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        yield


@pytest.mark.usefixtures("dense_linears")
def test_sequence_parallel_call_uses_the_given_token_collectives() -> None:
    """The caller's collectives gather the token shards before the MLP and
    reduce-scatter its output; a plain call does neither."""
    torch.manual_seed(0)
    calls = []

    class Collectives:
        def all_gather(self, x, dim):
            calls.append(("gather", dim))
            return torch.cat((x, x), dim=dim)

        def reduce_scatter(self, x, dim):
            calls.append(("scatter", dim))
            return x.chunk(2, dim=dim)[0] * 2

    mlp = GatedMLP(4, 6, "silu", sp_group=Collectives())
    x = torch.randn(3, 4)
    gate, up = (x @ mlp.gate_up_proj.weight.T).chunk(2, dim=-1)
    reference = (torch.nn.functional.silu(gate) * up) @ mlp.down_proj.weight.T

    torch.testing.assert_close(mlp(x), reference)
    assert calls == []
    torch.testing.assert_close(mlp(x, sequence_parallel=True), 2 * reference)
    assert calls == [("gather", 0), ("scatter", 0)]


@pytest.mark.usefixtures("dense_linears")
def test_unsupported_activation_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported gated MLP activation 'gelu'"):
        GatedMLP(4, 6, "gelu")
