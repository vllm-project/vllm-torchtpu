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
"""TPU component tests for SeaCacheBackend and apply_sea_filter."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
from vllm_omni.diffusion.cache.seacache.backend import SeaCacheBackend
from vllm_omni.diffusion.cache.seacache.sea_filter import apply_sea_filter
from vllm_omni.diffusion.data import DiffusionCacheConfig

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [pytest.mark.omni, pytest.mark.timeout(PR_TIMEOUT_S)]

SEA_FILTER_MAX_REL_L2 = 2.0e-2
SEACACHE_MAX_REL_L2 = 3.0e-2
SIGMAS = [0.9, 0.88, 0.86, 0.84, 0.82]


class Cosmos3VFMTransformer(nn.Module):
    """Minimal Cosmos3-named transformer for SeaCacheRootHook."""

    def __init__(self, channels: int = 4, hidden_dim: int = 8) -> None:
        super().__init__()
        self.in_proj = nn.Linear(channels, hidden_dim)
        self.block_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, channels)
        self.stack_calls = 0

    def _gen_preprocess(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        video_shape: tuple[int, int, int],
        **kwargs: Any,
    ) -> SimpleNamespace:
        del text_ids, text_mask, video_shape, kwargs
        b, c, t, h, w = hidden_states.shape
        flat = hidden_states.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)
        hidden_gen = self.in_proj(flat)
        time_embed = timestep.to(device=hidden_gen.device, dtype=hidden_gen.dtype)
        return SimpleNamespace(
            hidden_gen=hidden_gen,
            time_embed=time_embed,
            shape=(b, c, t, h, w),
        )

    def _shard_gen_prep(
        self, prep: SimpleNamespace, defer_gather: bool = True
    ) -> SimpleNamespace:
        del defer_gather
        return prep

    def _run_gen_stack(self, prep: SimpleNamespace) -> torch.Tensor:
        self.stack_calls += 1
        return prep.hidden_gen + 0.15 * torch.tanh(self.block_proj(prep.hidden_gen))

    def _gen_postprocess(
        self, hidden: torch.Tensor, prep: SimpleNamespace
    ) -> torch.Tensor:
        b, c, t, h, w = prep.shape
        out = self.out_proj(hidden)
        return out.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        video_shape: tuple[int, int, int],
        **kwargs: Any,
    ) -> torch.Tensor:
        prep = self._gen_preprocess(
            hidden_states, timestep, text_ids, text_mask, video_shape, **kwargs
        )
        prep = self._shard_gen_prep(prep, defer_gather=True)
        return self._gen_postprocess(self._run_gen_stack(prep), prep)


class Cosmos3OmniDiffusersPipeline:
    """Minimal Cosmos3 pipeline holder for SeaCacheBackend."""

    def __init__(self, transformer: Cosmos3VFMTransformer) -> None:
        self.transformer = transformer
        self.current_step_index = 0
        self.current_sigma = SIGMAS[0]
        self.num_timesteps = len(SIGMAS)
        self._cache_context_factory: Any = None


def _run_seacache_steps(
    pipeline: Cosmos3OmniDiffusersPipeline,
    latents_list: list[torch.Tensor],
    device: str,
    dtype: torch.dtype,
    *,
    use_cache_ctx: bool,
) -> list[torch.Tensor]:
    outputs: list[torch.Tensor] = []
    text_ids = torch.zeros(1, 2, dtype=torch.long, device=device)
    text_mask = torch.ones(1, 2, dtype=torch.bool, device=device)
    for step_idx, (sigma, latent) in enumerate(zip(SIGMAS, latents_list, strict=True)):
        pipeline.current_step_index = step_idx
        pipeline.current_sigma = sigma
        ctx = (
            pipeline._cache_context_factory("sea_cache")
            if use_cache_ctx
            else nullcontext()
        )
        with ctx:
            out = pipeline.transformer(
                latent.to(device=device, dtype=dtype),
                torch.tensor([sigma], device=device, dtype=dtype),
                text_ids,
                text_mask,
                (2, 4, 4),
            )
        outputs.append(out)
    return outputs


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "apply_sea_filter (torch.fft.fftn/ifftn) on TPU has rel_l2 0.1129 > 0.02 "
        "vs CPU fp32 (CPU bf16 is 2.39e-3)"
    ),
)
def test_seacache() -> None:
    """SeaCache

    - TPU path: Run apply_sea_filter on a 5D "tpu" tensor and run
      SeaCacheBackend (SeaCacheRootHook + extrapolate_residual) on a
      Cosmos3OmniDiffusersPipeline-named holder with a tiny DiT on "tpu"
      (bfloat16) across 5 decreasing sigma steps with sea_threshold=10.0.
    - Setup check: Assert backend.is_enabled(), hook.skip_count > 0, and DiT
      forward count < 5.
    - Outcome check: Assert TPU apply_sea_filter output and 5-step
      SeaCacheBackend output differ from uncached compute and match CPU float32
      reference within relative-L2 tolerance.
    """
    torch.manual_seed(0)
    filter_in = torch.randn(2, 4, 4, 2, 2)
    tpu_filtered = apply_sea_filter(
        filter_in.to(device="tpu", dtype=torch.bfloat16), sigma=0.5, power_exp=3.0
    )
    cpu_filtered = apply_sea_filter(filter_in.float(), sigma=0.5, power_exp=3.0)
    assert tpu_filtered.device.type == "tpu", (
        f"Expected apply_sea_filter output on tpu, got {tpu_filtered.device.type}"
    )
    assert_parity(
        tpu_filtered.cpu(),
        cpu_filtered,
        name="apply_sea_filter TPU bf16 vs CPU fp32",
        max_rel_l2=SEA_FILTER_MAX_REL_L2,
    )

    base_model = Cosmos3VFMTransformer()
    step_latents = [torch.randn(1, 4, 2, 4, 4) for _ in SIGMAS]

    tpu_pipe = Cosmos3OmniDiffusersPipeline(
        copy.deepcopy(base_model).to(device="tpu", dtype=torch.bfloat16)
    )
    uncached_pipe = Cosmos3OmniDiffusersPipeline(
        copy.deepcopy(base_model).to(device="tpu", dtype=torch.bfloat16)
    )
    cpu_pipe = Cosmos3OmniDiffusersPipeline(
        copy.deepcopy(base_model).to(device="cpu", dtype=torch.float32)
    )

    cache_cfg = DiffusionCacheConfig(sea_threshold=10.0, sea_residual_order=1)
    tpu_backend = SeaCacheBackend(cache_cfg)
    tpu_backend.enable(tpu_pipe)
    tpu_backend.refresh(tpu_pipe, num_inference_steps=len(SIGMAS), verbose=False)

    assert tpu_backend.is_enabled(), "SeaCacheBackend should report enabled"
    hook = tpu_pipe.transformer._hook_registry.get_hook("sea_cache")
    assert hook is not None, "SeaCacheRootHook should be registered on transformer"

    cpu_backend = SeaCacheBackend(cache_cfg)
    cpu_backend.enable(cpu_pipe)
    cpu_backend.refresh(cpu_pipe, num_inference_steps=len(SIGMAS), verbose=False)

    with torch.no_grad():
        tpu_outs = _run_seacache_steps(
            tpu_pipe, step_latents, "tpu", torch.bfloat16, use_cache_ctx=True
        )
        uncached_outs = _run_seacache_steps(
            uncached_pipe, step_latents, "tpu", torch.bfloat16, use_cache_ctx=False
        )
        cpu_outs = _run_seacache_steps(
            cpu_pipe, step_latents, "cpu", torch.float32, use_cache_ctx=True
        )

    assert hook.skip_count > 0, (
        f"Expected SeaCache skip_count > 0 on TPU, got {hook.skip_count}"
    )
    assert tpu_pipe.transformer.stack_calls < len(SIGMAS), (
        f"Expected DiT stack_calls < {len(SIGMAS)}, got "
        f"{tpu_pipe.transformer.stack_calls}"
    )
    assert not torch.allclose(tpu_outs[-2].cpu(), uncached_outs[-2].cpu(), atol=1e-4), (
        "Cached step-3 SeaCache TPU output should differ from uncached compute"
    )
    assert_parity(
        tpu_outs[-1].cpu(),
        cpu_outs[-1],
        name="SeaCache 5-step TPU bf16 vs CPU fp32",
        max_rel_l2=SEACACHE_MAX_REL_L2,
    )
