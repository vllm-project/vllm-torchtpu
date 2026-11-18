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
"""TPU component tests for MagCacheBackend."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from vllm_omni.diffusion.cache.magcache.backend import MagCacheBackend
from vllm_omni.diffusion.data import DiffusionCacheConfig

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [pytest.mark.omni, pytest.mark.timeout(PR_TIMEOUT_S)]

MAGCACHE_MAX_REL_L2 = 3.0e-2
NUM_STEPS = 4


class _TinyFluxMagBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.img_proj = nn.Linear(dim, dim)
        self.txt_proj = nn.Linear(dim, dim)
        self.calls = 0

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls += 1
        h = hidden_states + 0.15 * torch.tanh(self.img_proj(hidden_states))
        e = encoder_hidden_states + 0.1 * torch.tanh(
            self.txt_proj(encoder_hidden_states)
        )
        return e, h


class FluxTransformer2DModel(nn.Module):
    """Minimal Flux-named multi-block transformer for MagCacheBackend."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [_TinyFluxMagBlock(dim) for _ in range(3)]
        )
        self.proj_out = nn.Linear(dim, dim)

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        h = hidden_states
        e = encoder_hidden_states
        for block in self.transformer_blocks:
            e, h = block(h, e)
        return self.proj_out(h)


def _run_magcache_steps(
    model: FluxTransformer2DModel,
    step_inputs: list[tuple[torch.Tensor, torch.Tensor]],
    head_hook: object,
    device: str,
    dtype: torch.dtype,
) -> tuple[list[torch.Tensor], str, int]:
    outputs: list[torch.Tensor] = []
    residual_device = ""
    num_norm_ratios = 0
    for step_idx, (h_in, e_in) in enumerate(step_inputs):
        out = model(
            h_in.to(device=device, dtype=dtype),
            e_in.to(device=device, dtype=dtype),
        )
        outputs.append(out)
        if step_idx == NUM_STEPS - 2:
            state = head_hook.state_manager.get_state()
            residual_device = (
                state.previous_residual.device.type
                if state.previous_residual is not None
                else ""
            )
            num_norm_ratios = len(state.norm_ratios)
    return outputs, residual_device, num_norm_ratios


@pytest.mark.parametrize(
    "mode_id, cache_kwargs, expected_mid_calls, expected_norm_ratios",
    [
        pytest.param(
            "skip",
            {
                "num_inference_steps": NUM_STEPS,
                "mag_threshold": 0.5,
                "mag_max_skip_steps": 3,
                "mag_retention_ratio": 0.25,
                "mag_calibrate": False,
            },
            1,
            0,
            id="skip",
        ),
        pytest.param(
            "calibrate",
            {
                "num_inference_steps": NUM_STEPS,
                "mag_calibrate": True,
            },
            NUM_STEPS,
            NUM_STEPS - 1,
            id="calibrate",
        ),
    ],
)
def test_magcache(
    mode_id: str,
    cache_kwargs: dict[str, object],
    expected_mid_calls: int,
    expected_norm_ratios: int,
) -> None:
    """MagCache

    - TPU path: Enable MagCacheBackend (FluxMagCacheStrategy +
      MagCacheHeadHook/MagCacheBlockHook) on a tiny FluxTransformer2DModel-named
      multi-block DiT on "tpu" (bfloat16) across 4 steps, parametrized over
      pinned skip mode and on-the-fly calibration.
    - Setup check: Assert backend.is_enabled(), MagCache config is attached,
      state.previous_residual resides on "tpu", and the middle-block call count
      and len(state.norm_ratios) match the mode (skip mode skips 3 of 4 steps
      and records no ratios; calibrate mode runs every step and records 3).
    - Outcome check: Assert TPU multi-step outputs match CPU float32
      MagCacheBackend reference within bf16 relative-L2 tolerance.
    """
    torch.manual_seed(0)
    base_model = FluxTransformer2DModel()
    step_inputs = [
        (torch.randn(1, 6, 16), torch.randn(1, 4, 16)) for _ in range(NUM_STEPS)
    ]

    tpu_model = copy.deepcopy(base_model).to(device="tpu", dtype=torch.bfloat16)
    cpu_model = copy.deepcopy(base_model).to(device="cpu", dtype=torch.float32)

    cache_cfg = DiffusionCacheConfig(**cache_kwargs)
    tpu_backend = MagCacheBackend(cache_cfg)
    tpu_pipe = SimpleNamespace(transformer=tpu_model)
    tpu_backend.enable(tpu_pipe)
    tpu_backend.refresh(tpu_pipe, num_inference_steps=NUM_STEPS)

    assert tpu_backend.is_enabled(), "MagCacheBackend should report enabled"
    assert tpu_backend._magcache_config is not None, (
        "MagCacheBackend should attach _magcache_config"
    )
    tpu_head_hook = tpu_model.transformer_blocks[0]._hook_registry.get_hook(
        "mag_cache_leader_block_hook"
    )
    assert tpu_head_hook is not None, "MagCacheHeadHook should be registered"

    cpu_backend = MagCacheBackend(cache_cfg)
    cpu_pipe = SimpleNamespace(transformer=cpu_model)
    cpu_backend.enable(cpu_pipe)
    cpu_backend.refresh(cpu_pipe, num_inference_steps=NUM_STEPS)
    cpu_head_hook = cpu_model.transformer_blocks[0]._hook_registry.get_hook(
        "mag_cache_leader_block_hook"
    )

    with torch.no_grad():
        tpu_outs, residual_device, num_norm_ratios = _run_magcache_steps(
            tpu_model, step_inputs, tpu_head_hook, "tpu", torch.bfloat16
        )
        cpu_outs, _, _ = _run_magcache_steps(
            cpu_model, step_inputs, cpu_head_hook, "cpu", torch.float32
        )

    assert residual_device == "tpu", (
        f"Expected MagCache previous_residual on tpu, got {residual_device!r}"
    )
    mid_calls = tpu_model.transformer_blocks[1].calls
    assert mid_calls == expected_mid_calls, (
        f"{mode_id}: middle block called {mid_calls} times over {NUM_STEPS} "
        f"steps, expected {expected_mid_calls}"
    )
    assert num_norm_ratios == expected_norm_ratios, (
        f"{mode_id}: {num_norm_ratios} norm_ratios before the final reset, "
        f"expected {expected_norm_ratios}"
    )

    assert_parity(
        tpu_outs[-1].cpu(),
        cpu_outs[-1],
        name=f"MagCache {mode_id} TPU bf16 vs CPU fp32",
        max_rel_l2=MAGCACHE_MAX_REL_L2,
    )
