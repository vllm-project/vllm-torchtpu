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
"""TPU tests for vllm_omni.env_override."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from torch._dynamo.exc import FailOnRecompileLimitHit
from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.env_override import configure_torch_dynamo
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

_COMPILE_PARITY_MAX_REL_L2 = 1e-4
_COMPILE_PARITY_MAX_ABS = 1e-4


class _TinyBlock(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.proj_in = nn.Linear(dim, dim * 2, bias=False)
        self.proj_out = nn.Linear(dim * 2, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_out(torch.relu(self.proj_in(x)))


class _TinyRepeatedModel(nn.Module):
    _repeated_blocks = ["_TinyBlock"]

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.block = _TinyBlock(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "OmniTpuPlatform.supports_torch_inductor() returns False; "
        "diffusion torch.compile is disabled on TPU."
    ),
)
def test_torch_dynamo_recompile_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Torch Dynamo Recompile Limit.

    - Set VLLM_OMNI_TORCH_DYNAMO_RECOMPILE_LIMIT=2, run configure_torch_dynamo(),
      and exercise the diffusion compile gate plus regionally_compile on TPU.
    - Setup check: configure_torch_dynamo() updates
      torch._dynamo.config.recompile_limit to 2 and rejects non-positive values.
    - Outcome check: current_omni_platform.supports_torch_inductor() is True,
      two shapes match CPU fp32 via assert_parity, and a third shape raises
      FailOnRecompileLimitHit under error_on_recompile=True.
    """
    monkeypatch.setattr(
        torch._dynamo.config,
        "recompile_limit",
        torch._dynamo.config.recompile_limit,
    )
    monkeypatch.setattr(
        torch._dynamo.config,
        "error_on_recompile",
        torch._dynamo.config.error_on_recompile,
    )

    monkeypatch.setenv("VLLM_OMNI_TORCH_DYNAMO_RECOMPILE_LIMIT", "0")
    with pytest.raises(ValueError, match="must be positive"):
        configure_torch_dynamo()

    monkeypatch.setenv("VLLM_OMNI_TORCH_DYNAMO_RECOMPILE_LIMIT", "2")
    configure_torch_dynamo()
    assert torch._dynamo.config.recompile_limit == 2, (
        "Expected torch._dynamo.config.recompile_limit == 2, "
        f"got {torch._dynamo.config.recompile_limit}"
    )

    assert current_omni_platform.supports_torch_inductor() is True, (
        "Expected current_omni_platform.supports_torch_inductor() to be True on TPU"
    )

    torch.manual_seed(0)
    cpu_model = _TinyRepeatedModel(dim=16).eval()
    tpu_model = copy.deepcopy(cpu_model).to(torch.device("tpu"))
    regionally_compile(tpu_model, backend="tpu")

    for seq_len in (8, 12):
        x_cpu = torch.randn(1, seq_len, 16, dtype=torch.float32)
        with torch.no_grad():
            ref_out = cpu_model(x_cpu)
            tpu_out = tpu_model(x_cpu.to(torch.device("tpu")))
        assert_parity(
            tpu_out,
            ref_out,
            name=f"regionally_compile_seq_{seq_len}",
            max_rel_l2=_COMPILE_PARITY_MAX_REL_L2,
            max_abs=_COMPILE_PARITY_MAX_ABS,
        )

    torch._dynamo.config.error_on_recompile = True
    x_overflow = torch.randn(1, 16, 16, dtype=torch.float32, device=torch.device("tpu"))
    with pytest.raises(FailOnRecompileLimitHit), torch.no_grad():
        tpu_model(x_overflow)
