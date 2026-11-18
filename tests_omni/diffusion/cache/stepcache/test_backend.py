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
"""TPU component tests for StepCacheBackend."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from vllm_omni.diffusion.cache.stepcache.backend import (
    StepCacheBackend,
    get_stepcache_state,
)
from vllm_omni.diffusion.data import DiffusionCacheConfig
from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [pytest.mark.omni, pytest.mark.timeout(PR_TIMEOUT_S)]

STEPCACHE_MAX_REL_L2 = 3.0e-2
NUM_STEPS = 5


class _TinyVelocityDiT(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.calls = 0

    def forward(self, latents: torch.Tensor, timestep: float) -> torch.Tensor:
        self.calls += 1
        return 0.5 * latents + 0.1 * torch.tanh(self.proj(latents)) + 0.01 * timestep


def _run_euler_loop(
    dit: _TinyVelocityDiT,
    init_latents: torch.Tensor,
    state: object | None,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int]:
    latents = init_latents.to(device=device, dtype=dtype)
    prev_predictions: list[tuple[torch.Tensor, ...]] = []
    dt = 1.0 / NUM_STEPS
    max_countdown = 0
    for step in range(NUM_STEPS):
        should_run = True if state is None else state.should_run_step(prev_predictions)
        if state is not None:
            max_countdown = max(max_countdown, state.skip_countdown)
        if should_run:
            vel = dit(latents, 1.0 - step * dt)
            prev_predictions.append((vel,))
            if state is not None:
                state.trim_history(prev_predictions)
        else:
            vel = prev_predictions[-1][0]
        latents = latents - dt * vel
    return latents, max_countdown


def test_stepcache() -> None:
    """StepCache

    - TPU path: Enable StepCacheBackend (StepCacheState via get_stepcache_state)
      on DreamZeroPipeline.__new__(DreamZeroPipeline) and run a 5-step Euler
      denoise loop with a tiny DiT on "tpu" (bfloat16) using pinned
      velocity_sim_thresholds=[0.5] and velocity_skip_countdowns=[2].
    - Setup check: Assert backend.is_enabled(), get_stepcache_state(pipeline) is
      active, TPU velocity cosine similarity triggers skip_countdown > 0, and
      DiT forward runs fewer than 5 times.
    - Outcome check: Assert TPU StepCache denoised output differs from uncached
      5-step TPU output and matches CPU float32 StepCacheBackend denoise
      reference within bf16 relative-L2 tolerance.
    """
    torch.manual_seed(0)
    base_dit = _TinyVelocityDiT()
    init_latents = torch.randn(1, 8, 16)

    tpu_dit = copy.deepcopy(base_dit).to(device="tpu", dtype=torch.bfloat16)
    uncached_tpu_dit = copy.deepcopy(base_dit).to(device="tpu", dtype=torch.bfloat16)
    cpu_dit = copy.deepcopy(base_dit).to(device="cpu", dtype=torch.float32)

    cache_cfg = DiffusionCacheConfig(
        step_cache_dit_enabled=True,
        velocity_sim_thresholds=[0.5],
        velocity_skip_countdowns=[2],
    )
    tpu_pipe = object.__new__(DreamZeroPipeline)
    tpu_backend = StepCacheBackend(cache_cfg)
    tpu_backend.enable(tpu_pipe)
    tpu_backend.refresh(tpu_pipe, num_inference_steps=NUM_STEPS, verbose=False)
    tpu_state = get_stepcache_state(tpu_pipe)

    assert tpu_backend.is_enabled(), "StepCacheBackend should report enabled"
    assert tpu_state is not None, (
        "Expected StepCacheState on DreamZeroPipeline after enable/refresh, got None"
    )
    assert tpu_state.config.enabled, (
        f"Expected tpu_state.config.enabled=True, got {tpu_state.config.enabled!r}"
    )

    cpu_pipe = object.__new__(DreamZeroPipeline)
    cpu_backend = StepCacheBackend(cache_cfg)
    cpu_backend.enable(cpu_pipe)
    cpu_backend.refresh(cpu_pipe, num_inference_steps=NUM_STEPS, verbose=False)
    cpu_state = get_stepcache_state(cpu_pipe)

    with torch.no_grad():
        tpu_out, max_countdown = _run_euler_loop(
            tpu_dit, init_latents, tpu_state, "tpu", torch.bfloat16
        )
        uncached_out, _ = _run_euler_loop(
            uncached_tpu_dit, init_latents, None, "tpu", torch.bfloat16
        )
        cpu_out, _ = _run_euler_loop(
            cpu_dit, init_latents, cpu_state, "cpu", torch.float32
        )

    assert max_countdown > 0, (
        f"Expected skip_countdown > 0 on TPU velocity match, got {max_countdown}"
    )
    assert tpu_dit.calls < NUM_STEPS, (
        f"Expected DiT calls < {NUM_STEPS} with StepCache, got {tpu_dit.calls}"
    )
    assert not torch.allclose(tpu_out.cpu(), uncached_out.cpu(), atol=1e-4), (
        "StepCache TPU output should differ from uncached 5-step compute"
    )
    assert_parity(
        tpu_out.cpu(),
        cpu_out,
        name="StepCache TPU bf16 vs CPU fp32",
        max_rel_l2=STEPCACHE_MAX_REL_L2,
    )
