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
"""TPU component tests for TeaCacheBackend."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
from vllm_omni.diffusion.cache.teacache.backend import TeaCacheBackend
from vllm_omni.diffusion.data import DiffusionCacheConfig

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

TEACACHE_MAX_REL_L2 = 3.0e-2


class _TinyNorm1(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, hidden_states: torch.Tensor, emb: torch.Tensor
    ) -> tuple[torch.Tensor]:
        return (self.proj(hidden_states) + emb.unsqueeze(1),)


class _TinyFluxBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm1 = _TinyNorm1(dim)
        self.img_proj = nn.Linear(dim, dim)
        self.txt_proj = nn.Linear(dim, dim)
        self.calls = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del image_rotary_emb, joint_attention_kwargs
        self.calls += 1
        h = hidden_states + 0.2 * torch.tanh(self.img_proj(hidden_states))
        h = h + 0.05 * temb.unsqueeze(1)
        e = encoder_hidden_states + 0.1 * torch.tanh(
            self.txt_proj(encoder_hidden_states)
        )
        return e, h


class _TinyTimeTextEmbed(nn.Module):
    def __init__(self, pooled_dim: int, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(pooled_dim, dim)

    def forward(
        self, timestep: torch.Tensor, pooled_projections: torch.Tensor
    ) -> torch.Tensor:
        return self.proj(pooled_projections) + timestep.unsqueeze(-1) * 1e-3


class _TinyPosEmbed(nn.Module):
    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ones = torch.ones((ids.shape[0], 4), device=ids.device, dtype=torch.float32)
        return ones, ones


class _TinyNormOut(nn.Module):
    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        return hidden_states + 0.01 * temb.unsqueeze(1)


class FluxTransformer2DModel(nn.Module):
    """Minimal Flux-structured transformer matching TeaCache's Flux extractor."""

    def __init__(self, in_dim: int = 8, txt_dim: int = 8, dim: int = 16) -> None:
        super().__init__()
        self.do_true_cfg = False
        self.x_embedder = nn.Linear(in_dim, dim)
        self.context_embedder = nn.Linear(txt_dim, dim)
        self.time_text_embed = _TinyTimeTextEmbed(dim, dim)
        self.pos_embed = _TinyPosEmbed()
        self.transformer_blocks = nn.ModuleList([_TinyFluxBlock(dim) for _ in range(2)])
        self.single_transformer_blocks = nn.ModuleList([])
        self.norm_out = _TinyNormOut()
        self.proj_out = nn.Linear(dim, in_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        return_dict: bool = False,
    ) -> tuple[torch.Tensor]:
        del return_dict
        h = self.x_embedder(hidden_states)
        t = timestep.to(device=h.device, dtype=h.dtype) * 1000
        temb = self.time_text_embed(t, pooled_projections)
        e = self.context_embedder(encoder_hidden_states)
        ids = torch.cat((txt_ids, img_ids), dim=0)
        rot = self.pos_embed(ids)
        for block in self.transformer_blocks:
            e, h = block(
                hidden_states=h,
                encoder_hidden_states=e,
                temb=temb,
                image_rotary_emb=rot,
            )
        return (self.proj_out(self.norm_out(h, temb)),)


def _run_three_steps(
    transformer: FluxTransformer2DModel,
    backend: TeaCacheBackend | None,
    pipeline: SimpleNamespace | None,
    inputs: list[tuple[torch.Tensor, ...]],
    device: str,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    outputs: list[torch.Tensor] = []
    for step_idx, step_inputs in enumerate(inputs):
        if step_idx == 2 and backend is not None and pipeline is not None:
            backend.refresh(pipeline, num_inference_steps=3, verbose=False)
        dev_inputs = tuple(t.to(device=device, dtype=dtype) for t in step_inputs)
        out = transformer(*dev_inputs, return_dict=False)[0]
        outputs.append(out)
    return outputs


def test_teacache() -> None:
    """TeaCache

    - TPU path: Enable TeaCacheBackend (TeaCacheHook + FluxTransformer2DModel
      extractor) on a tiny multi-block Flux-structured DiT on "tpu" (bfloat16)
      across 3 timesteps with pinned rel_l1_thresh so step 0 computes, step 1
      reuses cached residual, and backend.refresh() resets state.
    - Setup check: Assert backend.is_enabled(), "teacache" hook is registered,
      state.previous_residual resides on "tpu", and transformer blocks run only
      2 times across 3 steps.
    - Outcome check: Assert step-1 cached TPU output differs from uncached TPU
      compute and matches CPU float32 TeaCacheBackend reference within bf16
      relative-L2 tolerance.
    """
    torch.manual_seed(0)
    base_model = FluxTransformer2DModel()
    base_hidden = torch.randn(1, 6, 8)
    base_encoder = torch.randn(1, 4, 8)
    base_pooled = torch.randn(1, 16)
    step_inputs = [
        (
            base_hidden + idx * 0.005,
            base_encoder + idx * 0.005,
            base_pooled + idx * 0.005,
            torch.tensor([1.0 - idx * 0.01]),
            torch.zeros(6, 3),
            torch.zeros(4, 3),
        )
        for idx in range(3)
    ]

    tpu_model = copy.deepcopy(base_model).to(device="tpu", dtype=torch.bfloat16)
    uncached_tpu_model = copy.deepcopy(base_model).to(
        device="tpu", dtype=torch.bfloat16
    )
    cpu_model = copy.deepcopy(base_model).to(device="cpu", dtype=torch.float32)

    cache_cfg = DiffusionCacheConfig(rel_l1_thresh=10.0)
    tpu_backend = TeaCacheBackend(cache_cfg)
    tpu_pipe = SimpleNamespace(transformer=tpu_model)
    tpu_backend.enable(tpu_pipe)
    tpu_backend.refresh(tpu_pipe, num_inference_steps=3, verbose=False)

    assert tpu_backend.is_enabled(), "TeaCacheBackend should report enabled"
    hook = tpu_model._hook_registry.get_hook("teacache")
    assert hook is not None, "TeaCacheHook should be registered on transformer"

    cpu_backend = TeaCacheBackend(cache_cfg)
    cpu_pipe = SimpleNamespace(transformer=cpu_model)
    cpu_backend.enable(cpu_pipe)
    cpu_backend.refresh(cpu_pipe, num_inference_steps=3, verbose=False)

    with torch.no_grad():
        tpu_outs = _run_three_steps(
            tpu_model, tpu_backend, tpu_pipe, step_inputs, "tpu", torch.bfloat16
        )
        uncached_outs = _run_three_steps(
            uncached_tpu_model, None, None, step_inputs, "tpu", torch.bfloat16
        )
        cpu_outs = _run_three_steps(
            cpu_model, cpu_backend, cpu_pipe, step_inputs, "cpu", torch.float32
        )

    hook.state_manager.set_context("teacache_positive")
    state = hook.state_manager.get_state()
    assert state.previous_residual is not None, "TeaCache did not cache residual"
    assert state.previous_residual.device.type == "tpu", (
        f"Expected cached residual on tpu, got {state.previous_residual.device.type}"
    )
    block_calls = tpu_model.transformer_blocks[0].calls
    assert block_calls == 2, (
        f"Expected 2 block executions across 3 steps (1 cached), got {block_calls}"
    )
    assert not torch.allclose(tpu_outs[1].cpu(), uncached_outs[1].cpu(), atol=1e-4), (
        "Step-1 TeaCache TPU output should differ from uncached compute"
    )
    assert_parity(
        tpu_outs[1].cpu(),
        cpu_outs[1],
        name="TeaCache step-1 TPU bf16 vs CPU fp32",
        max_rel_l2=TEACACHE_MAX_REL_L2,
    )
