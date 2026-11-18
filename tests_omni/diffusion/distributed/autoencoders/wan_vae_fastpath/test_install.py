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
"""TPU tests for Wan VAE encoder/decoder fast-path installation and parity."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from diffusers.models.autoencoders import AutoencoderKLWan
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    DistributedAutoencoderKLWan,
)
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import (
    is_encoder_installed,
    is_installed,
)
from vllm_omni.diffusion.registry import _apply_wan_vae_fastpath_if_enabled

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

FASTPATH_MAX_REL_L2 = 2e-2


def _tiny_wan_vae_kwargs() -> dict[str, object]:
    return {
        "base_dim": 16,
        "z_dim": 4,
        "dim_mult": [1, 2],
        "num_res_blocks": 1,
        "temperal_downsample": [False],
        "latents_mean": [0.0] * 4,
        "latents_std": [1.0] * 4,
        "is_residual": True,
        "patch_size": 2,
        "scale_factor_temporal": 1,
        "scale_factor_spatial": 2,
    }


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "Wan VAE fast path skips TPU because _apply_wan_vae_fastpath_if_enabled "
        "returns early when not current_omni_platform.is_cuda()"
    ),
)
@pytest.mark.parametrize("level", ["lossless", "channels_last"])
def test_wan_vae_fast_paths(level: str) -> None:
    """Wan VAE Fast Paths (Decoder & Encoder)

    - Builds a tiny residual ``DistributedAutoencoderKLWan`` on TPU and invokes
      ``_apply_wan_vae_fastpath_if_enabled`` for ``lossless`` and ``channels_last``.
    - Setup check: ``is_installed(vae)`` and ``is_encoder_installed(vae)`` are True.
    - Outcome check: ``vae.encode`` and ``vae.decode`` match a CPU fp32
      ``AutoencoderKLWan`` reference via relative L2.
    """
    kwargs = _tiny_wan_vae_kwargs()
    torch.manual_seed(0)
    cpu_vae = AutoencoderKLWan(**kwargs).eval()
    torch.manual_seed(0)
    tpu_vae = DistributedAutoencoderKLWan(**kwargs).eval().to("tpu")

    od_config = OmniDiffusionConfig(
        vae_fast_path=level,
        vae_encode_fast_path=level,
    )
    _apply_wan_vae_fastpath_if_enabled(SimpleNamespace(vae=tpu_vae), od_config)

    assert is_installed(tpu_vae), (
        f"Wan VAE decoder fast path ({level}) was not installed on TPU"
    )
    assert is_encoder_installed(tpu_vae), (
        f"Wan VAE encoder fast path ({level}) was not installed on TPU"
    )

    video_cpu = torch.randn((1, 3, 2, 16, 16), dtype=torch.float32)
    with torch.no_grad():
        cpu_latents = cpu_vae.encode(video_cpu).latent_dist.mode()
        tpu_latents = tpu_vae.encode(video_cpu.to("tpu")).latent_dist.mode()
        cpu_decoded = cpu_vae.decode(cpu_latents).sample
        tpu_decoded = tpu_vae.decode(tpu_latents).sample

    assert_parity(
        tpu_latents,
        cpu_latents,
        name=f"wan_vae_fastpath encode ({level})",
        max_rel_l2=FASTPATH_MAX_REL_L2,
    )
    assert_parity(
        tpu_decoded,
        cpu_decoded,
        name=f"wan_vae_fastpath decode ({level})",
        max_rel_l2=FASTPATH_MAX_REL_L2,
    )
