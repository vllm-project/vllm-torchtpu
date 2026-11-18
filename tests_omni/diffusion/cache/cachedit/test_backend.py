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
"""TPU component tests for CacheDiTBackend."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from cache_dit import ForwardPattern
from vllm_omni.diffusion.cache.cachedit.backend import CacheDiTBackend, cache_summary
from vllm_omni.diffusion.cache.cachedit.config import CacheDiTAdapterConfig
from vllm_omni.diffusion.data import DiffusionCacheConfig

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

CACHED_DIT_MAX_REL_L2 = 3.0e-2
NUM_STEPS = 4


class _TinyDiTBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.calls = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return hidden_states + 0.15 * torch.tanh(self.proj(hidden_states))


class _TinyFourBlockDiT(nn.Module):
    _cache_dit_adapter_config = CacheDiTAdapterConfig(
        block_forward_patterns={"blocks": ForwardPattern.Pattern_3},
        check_forward_pattern=False,
    )

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_TinyDiTBlock(dim) for _ in range(4)])
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        h = hidden_states
        for block in self.blocks:
            h = block(h)
        return self.proj_out(h)


def _run_steps(
    model: _TinyFourBlockDiT,
    step_inputs: list[torch.Tensor],
    device: str,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    outputs: list[torch.Tensor] = []
    for x in step_inputs:
        outputs.append(model(x.to(device=device, dtype=dtype)))
    return outputs


@pytest.mark.parametrize(
    "cache_kwargs",
    [
        pytest.param(
            {
                "Fn_compute_blocks": 1,
                "Bn_compute_blocks": 1,
                "max_warmup_steps": 1,
                "residual_diff_threshold": 10.0,
            },
            id="dbcache",
        ),
        pytest.param(
            {
                "Fn_compute_blocks": 1,
                "Bn_compute_blocks": 1,
                "max_warmup_steps": 1,
                "residual_diff_threshold": 10.0,
                "enable_taylorseer": True,
                "taylorseer_order": 1,
            },
            id="taylorseer",
        ),
    ],
)
def test_cache_dit(cache_kwargs: dict[str, object]) -> None:
    """Cache-DiT

    - TPU path: Enable CacheDiTBackend + refresh() + cache_summary() on a tiny
      4-block DiT on "tpu" (bfloat16) across 4 steps, parametrized over DBCache
      and TaylorSeer with residual_diff_threshold=10.0 to pin post-warmup
      caching.
    - Setup check: Assert backend.is_enabled(), len(pipeline._cache_dit_targets)
      == 1, and middle blocks execute fewer times than total steps.
    - Outcome check: Assert TPU cached output differs from uncached TPU output
      and matches CPU float32 CacheDiTBackend reference within bf16 relative-L2
      tolerance.
    """
    torch.manual_seed(0)
    base_model = _TinyFourBlockDiT()
    step_inputs = [torch.randn(1, 8, 16) for _ in range(NUM_STEPS)]

    tpu_model = copy.deepcopy(base_model).to(device="tpu", dtype=torch.bfloat16)
    uncached_tpu_model = copy.deepcopy(base_model).to(
        device="tpu", dtype=torch.bfloat16
    )
    cpu_model = copy.deepcopy(base_model).to(device="cpu", dtype=torch.float32)

    cache_cfg = DiffusionCacheConfig(**cache_kwargs)
    tpu_backend = CacheDiTBackend(cache_cfg)
    tpu_pipe = SimpleNamespace(transformer=tpu_model)
    tpu_backend.enable(tpu_pipe)
    tpu_backend.refresh(tpu_pipe, num_inference_steps=NUM_STEPS, verbose=False)

    assert tpu_backend.is_enabled(), "CacheDiTBackend should report enabled"
    assert len(tpu_pipe._cache_dit_targets) == 1, (
        f"Expected 1 Cache-DiT target, got {len(tpu_pipe._cache_dit_targets)}"
    )

    cpu_backend = CacheDiTBackend(cache_cfg)
    cpu_pipe = SimpleNamespace(transformer=cpu_model)
    cpu_backend.enable(cpu_pipe)
    cpu_backend.refresh(cpu_pipe, num_inference_steps=NUM_STEPS, verbose=False)

    with torch.no_grad():
        tpu_outs = _run_steps(tpu_model, step_inputs, "tpu", torch.bfloat16)
        uncached_outs = _run_steps(
            uncached_tpu_model, step_inputs, "tpu", torch.bfloat16
        )
        cpu_outs = _run_steps(cpu_model, step_inputs, "cpu", torch.float32)
        cache_summary(tpu_pipe, details=True)

    mid_calls = tpu_model.blocks[1].calls
    assert mid_calls < NUM_STEPS, (
        f"Expected middle block calls < {NUM_STEPS} with Cache-DiT, got {mid_calls}"
    )
    assert not torch.allclose(tpu_outs[-1].cpu(), uncached_outs[-1].cpu(), atol=1e-4), (
        "Cache-DiT TPU output should differ from uncached compute"
    )
    assert_parity(
        tpu_outs[-1].cpu(),
        cpu_outs[-1],
        name="Cache-DiT TPU bf16 vs CPU fp32",
        max_rel_l2=CACHED_DIT_MAX_REL_L2,
    )
