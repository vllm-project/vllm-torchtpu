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
"""TPU tests for Wan VAE slicing and tiling memory optimizations."""

from __future__ import annotations

import pytest
import torch
from diffusers.models.autoencoders import AutoencoderKLWan
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    DistributedAutoencoderKLWan,
)

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

VAE_ENCODE_MAX_REL_L2 = 2e-2
VAE_DECODE_MAX_REL_L2 = 2e-2


def _tiny_wan_vae_kwargs() -> dict[str, object]:
    return {
        "base_dim": 16,
        "z_dim": 4,
        "dim_mult": [1, 2],
        "num_res_blocks": 1,
        "temperal_downsample": [False],
        "latents_mean": [0.0] * 4,
        "latents_std": [1.0] * 4,
        "scale_factor_temporal": 1,
        "scale_factor_spatial": 2,
    }


def _configure_vae_for_test(
    vae: AutoencoderKLWan,
    *,
    use_slicing: bool,
    use_tiling: bool,
) -> None:
    vae.use_slicing = use_slicing
    vae.use_tiling = use_tiling
    vae.tile_sample_min_height = 16
    vae.tile_sample_min_width = 16
    vae.tile_sample_stride_height = 12
    vae.tile_sample_stride_width = 12


@pytest.mark.parametrize(
    ("vae_use_slicing", "vae_use_tiling"),
    [
        pytest.param(True, False, id="slicing"),
        pytest.param(
            False,
            True,
            id="tiling",
            marks=pytest.mark.xfail(
                strict=True,
                raises=AttributeError,
                reason=(
                    "DistributedAutoencoderKLWan.is_distributed_enabled() "
                    "requires init_distributed() / distributed_executor"
                ),
            ),
        ),
        pytest.param(
            True,
            True,
            id="slicing_and_tiling",
            marks=pytest.mark.xfail(
                strict=True,
                raises=AttributeError,
                reason=(
                    "DistributedAutoencoderKLWan.is_distributed_enabled() "
                    "requires init_distributed() / distributed_executor"
                ),
            ),
        ),
    ],
)
def test_vae_slicing_and_tiling(
    monkeypatch: pytest.MonkeyPatch,
    vae_use_slicing: bool,
    vae_use_tiling: bool,
) -> None:
    """VAE Slicing and Tiling

    - Runs ``DistributedAutoencoderKLWan`` encode and decode on TPU with
      batch=2, 2-frame, 24x24 tensors and 16x16 spatial tiles.
    - Setup check: ``use_slicing`` and ``use_tiling`` match config, and spy
      counts confirm per-item slicing (``_decode``) and spatial tiling
      (``tiled_encode`` / ``tiled_decode``) execute.
    - Outcome check: TPU latents and decoded videos match a CPU fp32
      ``AutoencoderKLWan`` reference with identical slicing/tiling settings.
    """
    kwargs = _tiny_wan_vae_kwargs()
    torch.manual_seed(0)
    cpu_vae = AutoencoderKLWan(**kwargs).eval()
    torch.manual_seed(0)
    tpu_vae = DistributedAutoencoderKLWan(**kwargs).eval().to("tpu")

    _configure_vae_for_test(
        cpu_vae, use_slicing=vae_use_slicing, use_tiling=vae_use_tiling
    )
    _configure_vae_for_test(
        tpu_vae, use_slicing=vae_use_slicing, use_tiling=vae_use_tiling
    )

    assert tpu_vae.use_slicing is vae_use_slicing, (
        f"Expected use_slicing={vae_use_slicing}, got {tpu_vae.use_slicing}"
    )
    assert tpu_vae.use_tiling is vae_use_tiling, (
        f"Expected use_tiling={vae_use_tiling}, got {tpu_vae.use_tiling}"
    )

    counts = {"decode": 0, "tiled_encode": 0, "tiled_decode": 0}
    orig_decode = tpu_vae._decode
    orig_tiled_encode = tpu_vae.tiled_encode
    orig_tiled_decode = tpu_vae.tiled_decode

    def _spy_decode(*args: object, **kwargs_inner: object) -> object:
        counts["decode"] += 1
        return orig_decode(*args, **kwargs_inner)

    def _spy_tiled_encode(*args: object, **kwargs_inner: object) -> object:
        counts["tiled_encode"] += 1
        return orig_tiled_encode(*args, **kwargs_inner)

    def _spy_tiled_decode(*args: object, **kwargs_inner: object) -> object:
        counts["tiled_decode"] += 1
        return orig_tiled_decode(*args, **kwargs_inner)

    monkeypatch.setattr(tpu_vae, "_decode", _spy_decode)
    monkeypatch.setattr(tpu_vae, "tiled_encode", _spy_tiled_encode)
    monkeypatch.setattr(tpu_vae, "tiled_decode", _spy_tiled_decode)

    video_cpu = torch.randn((2, 3, 2, 24, 24), dtype=torch.float32)
    with torch.no_grad():
        cpu_latents = cpu_vae.encode(video_cpu).latent_dist.mode()
        tpu_latents = tpu_vae.encode(video_cpu.to("tpu")).latent_dist.mode()
        cpu_decoded = cpu_vae.decode(cpu_latents).sample
        tpu_decoded = tpu_vae.decode(tpu_latents).sample

    expected_decode_calls = 2 if vae_use_slicing else 1
    assert counts["decode"] == expected_decode_calls, (
        f"Expected _decode called {expected_decode_calls} times, got {counts['decode']}"
    )
    # The tiled paths run exactly when tiling is on.
    assert (counts["tiled_encode"] > 0) is vae_use_tiling, (
        f"tiling={vae_use_tiling}: tiled_encode called {counts['tiled_encode']} times"
    )
    assert (counts["tiled_decode"] > 0) is vae_use_tiling, (
        f"tiling={vae_use_tiling}: tiled_decode called {counts['tiled_decode']} times"
    )

    assert_parity(
        tpu_latents,
        cpu_latents,
        name=f"wan vae encode slicing={vae_use_slicing} tiling={vae_use_tiling}",
        max_rel_l2=VAE_ENCODE_MAX_REL_L2,
    )
    assert_parity(
        tpu_decoded,
        cpu_decoded,
        name=f"wan vae decode slicing={vae_use_slicing} tiling={vae_use_tiling}",
        max_rel_l2=VAE_DECODE_MAX_REL_L2,
    )
