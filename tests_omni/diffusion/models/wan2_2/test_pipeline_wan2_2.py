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
"""TPU tests for Wan2.2 dual-expert boundary ratio routing and flow shift."""

from __future__ import annotations

from collections.abc import Generator

import pytest
import torch
from tqdm import tqdm
from vllm.config import VllmConfig
from vllm.utils.network_utils import get_open_port
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    Wan22Pipeline,
    build_wan_scheduler,
)
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
    ColumnParallelGELU,
    WanTransformer3DModel,
)

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

WAN22_DIFFUSE_MAX_REL_L2 = 2e-2


@pytest.fixture(scope="module")
def omni_ctx() -> Generator[None, None, None]:
    """Single-rank diffusion parallel groups plus a default forward context."""
    owns_world = not torch.distributed.is_initialized()
    owns_model_parallel = False
    if owns_world:
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="gloo",
        )
    try:
        owns_model_parallel = not model_parallel_is_initialized()
        if owns_model_parallel:
            initialize_model_parallel(tensor_parallel_size=1, pipeline_parallel_size=1)
        with set_forward_context(
            vllm_config=VllmConfig(), omni_diffusion_config=OmniDiffusionConfig()
        ):
            yield
    finally:
        if owns_model_parallel:
            destroy_model_parallel()
        if owns_world:
            destroy_distributed_environment()


def _build_tiny_wan22_expert(seed: int) -> WanTransformer3DModel:
    torch.manual_seed(seed)
    model = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=128,
        in_channels=16,
        out_channels=16,
        text_dim=256,
        freq_dim=64,
        ffn_dim=512,
        num_layers=2,
    ).eval()
    # vLLM parallel linear layers allocate their weights with torch.empty.
    for param in model.parameters():
        torch.nn.init.normal_(param, std=0.02)
    return model


def _init_lightweight_pipeline(
    high_expert: WanTransformer3DModel,
    low_expert: WanTransformer3DModel,
    flow_shift: float,
) -> Wan22Pipeline:
    pipe = object.__new__(Wan22Pipeline)
    torch.nn.Module.__init__(pipe)
    pipe.transformer = high_expert
    pipe.transformer_2 = low_expert
    pipe.scheduler = build_wan_scheduler("unipc", flow_shift=flow_shift)
    pipe.scheduler.set_timesteps(4, shift=flow_shift)
    pipe.expand_timesteps = False
    pipe.is_dmd = False
    pipe._current_timestep = None
    pipe.progress_bar = lambda total=None: tqdm(total=total, disable=True)
    pipe.record_denoise_step = lambda step_idx, t: None
    return pipe


def test_video_boundary_ratio_and_flow_shift(
    omni_ctx: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Video Boundary Ratio & Flow Shift

    - Runs 4-step ``Wan22Pipeline.diffuse`` on TPU with two distinct tiny
      ``WanTransformer3DModel`` experts, ``boundary_ratio=0.875``, and
      ``flow_shift=5.0``.
    - Setup check: ``flow_shift=5.0`` shifts scheduler timesteps relative to
      ``flow_shift=1.0``, and spies confirm ``t >= boundary_timestep`` routes to
      ``transformer`` while ``t < boundary_timestep`` routes to ``transformer_2``.
    - Outcome check: TPU denoised latents match a CPU fp32 reference running the
      same experts and schedule via relative L2.
    """
    unshifted_sched = build_wan_scheduler("unipc", flow_shift=1.0)
    unshifted_sched.set_timesteps(4, shift=1.0)
    shifted_sched = build_wan_scheduler("unipc", flow_shift=5.0)
    shifted_sched.set_timesteps(4, shift=5.0)
    assert not torch.allclose(
        shifted_sched.timesteps.float(), unshifted_sched.timesteps.float()
    ), "flow_shift=5.0 did not shift scheduler timesteps relative to flow_shift=1.0"

    cpu_high = _build_tiny_wan22_expert(seed=1)
    cpu_low = _build_tiny_wan22_expert(seed=2)
    tpu_high = _build_tiny_wan22_expert(seed=1).to("tpu")
    tpu_low = _build_tiny_wan22_expert(seed=2).to("tpu")

    cpu_pipe = _init_lightweight_pipeline(cpu_high, cpu_low, flow_shift=5.0)
    tpu_pipe = _init_lightweight_pipeline(tpu_high, tpu_low, flow_shift=5.0)

    # Wan2.2 T2V's ratio; it splits the shifted timesteps (999, 937, 833, 624)
    # two per expert.
    boundary_ratio = 0.875
    boundary_timestep = boundary_ratio * tpu_pipe.scheduler.config.num_train_timesteps
    routed_high: list[float] = []
    routed_low: list[float] = []

    orig_high_forward = tpu_high.forward
    orig_low_forward = tpu_low.forward

    def _spy_high(*args: object, **kwargs: object) -> object:
        routed_high.append(float(kwargs["timestep"][0].item()))
        return orig_high_forward(*args, **kwargs)

    def _spy_low(*args: object, **kwargs: object) -> object:
        routed_low.append(float(kwargs["timestep"][0].item()))
        return orig_low_forward(*args, **kwargs)

    monkeypatch.setattr(tpu_high, "forward", _spy_high)
    monkeypatch.setattr(tpu_low, "forward", _spy_low)

    torch.manual_seed(0)
    latents_cpu = torch.randn((1, 16, 2, 8, 8), dtype=torch.float32)
    prompt_embeds_cpu = torch.randn((1, 4, 256), dtype=torch.float32)

    with torch.no_grad():
        # The TPU fusion fence patched onto ColumnParallelGELU only takes TPU
        # tensors; the CPU reference runs the upstream forward.
        with monkeypatch.context() as cpu_patch:
            cpu_patch.setattr(
                ColumnParallelGELU, "forward", ColumnParallelGELU._tpu_orig_forward
            )
            cpu_out = cpu_pipe.diffuse(
                latents=latents_cpu.clone(),
                timesteps=cpu_pipe.scheduler.timesteps,
                prompt_embeds=prompt_embeds_cpu,
                negative_prompt_embeds=None,
                guidance_low=1.0,
                guidance_high=1.0,
                boundary_timestep=boundary_timestep,
                dtype=torch.float32,
                attention_kwargs={},
            )
        tpu_out = tpu_pipe.diffuse(
            latents=latents_cpu.to("tpu"),
            timesteps=tpu_pipe.scheduler.timesteps.to("tpu"),
            prompt_embeds=prompt_embeds_cpu.to("tpu"),
            negative_prompt_embeds=None,
            guidance_low=1.0,
            guidance_high=1.0,
            boundary_timestep=boundary_timestep,
            dtype=torch.float32,
            attention_kwargs={},
        )

    assert routed_high, "High-noise expert was never routed during Wan2.2 diffuse()"
    assert all(t >= boundary_timestep for t in routed_high), (
        f"High-noise expert routed timesteps < boundary {boundary_timestep}: "
        f"{routed_high}"
    )
    assert routed_low, "Low-noise expert was never routed during Wan2.2 diffuse()"
    assert all(t < boundary_timestep for t in routed_low), (
        f"Low-noise expert routed timesteps >= boundary {boundary_timestep}: "
        f"{routed_low}"
    )

    assert_parity(
        tpu_out,
        cpu_out,
        name="wan2_2 boundary_ratio=0.5 flow_shift=5.0",
        max_rel_l2=WAN22_DIFFUSE_MAX_REL_L2,
    )
