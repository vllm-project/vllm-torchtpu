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
"""TPU tests for MiniMax-H3 LiteVAE pruned decoder checkpoint loading and decode."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [pytest.mark.omni, pytest.mark.timeout(PR_TIMEOUT_S)]

LITEVAE_MAX_REL_L2 = 2e-2


@pytest.mark.xfail(
    strict=True,
    raises=ModuleNotFoundError,
    reason=(
        "vllm_omni.diffusion.models.minimax_h3.lightvae is not in vllm-omni "
        "0.32.0rc1 (added in vllm-omni PR #9245)"
    ),
)
def test_minimax_h3_litevae_decoder(tmp_path: Path) -> None:
    """MiniMax-H3 LiteVAE Decoder

    - Imports ``minimax_h3.lightvae.{checkpoint,native}`` and installs a synthetic
      2-layer pruned LiteVAE checkpoint (``kept_layers=[1, 3]``) into a 4-layer
      native H3 VAE before moving to TPU.
    - Setup check: ``len(vae.decoder.transformer_blocks) == 2``.
    - Outcome check: decodes tiny latent tensors on TPU and checks relative-L2
      parity against a CPU fp32 LiteVAE decoder reference.
    """
    from vllm_omni.diffusion.models.minimax_h3.lightvae.checkpoint import (
        read_h3_lite_decoder_checkpoint,
        resolve_h3_lightvae_decoder_path,
    )
    from vllm_omni.diffusion.models.minimax_h3.lightvae.native import (
        install_h3_lite_decoder,
    )
    from vllm_omni.diffusion.models.minimax_h3.vae import AutoencoderKLH3

    latent_channels = 4
    heads = 2
    head_dim = 24
    native_config = {
        "latent_channels": latent_channels,
        "latents_mean": [0.0] * latent_channels,
        "latents_std": [1.0] * latent_channels,
        "clip_length": 17,
        "token_drop": 3,
        "decoder_num_layers": 4,
        "decoder_num_attention_heads": heads,
        "decoder_attention_head_dim": head_dim,
        "decoder_norm_eps": 1e-6,
        "decoder_rope_theta": 10000.0,
        "decoder_rope_dim_ratio": 1.0,
    }
    torch.manual_seed(0)
    cpu_vae = AutoencoderKLH3(**native_config)
    tpu_vae = AutoencoderKLH3(**native_config)

    kept_layers = [1, 3]
    lite_state: dict[str, torch.Tensor] = {}
    for key, tensor in cpu_vae.post_quant_conv.state_dict().items():
        lite_state[f"post_quant_conv.{key}"] = tensor.clone()
    for key, tensor in cpu_vae.decoder.state_dict().items():
        if key.startswith("transformer_blocks."):
            parts = key.split(".")
            layer_idx = int(parts[1])
            if layer_idx not in kept_layers:
                continue
            parts[1] = str(kept_layers.index(layer_idx))
            key = ".".join(parts)
        lite_state[f"decoder.{key}"] = tensor.clone()

    ckpt_path = tmp_path / "lite_decoder.safetensors"
    architecture = {**native_config, "family": "h3", "kept_layers": kept_layers}
    save_file(
        lite_state,
        str(ckpt_path),
        metadata={"architecture": json.dumps(architecture)},
    )

    resolved = resolve_h3_lightvae_decoder_path(
        {"lightvae_decoder_path": str(ckpt_path)}
    )
    assert resolved == str(ckpt_path), f"Unexpected resolved path {resolved}"
    checkpoint = read_h3_lite_decoder_checkpoint(resolved)

    install_h3_lite_decoder(cpu_vae, checkpoint, native_config=native_config)
    install_h3_lite_decoder(tpu_vae, checkpoint, native_config=native_config)
    num_blocks = len(tpu_vae.decoder.transformer_blocks)
    assert num_blocks == 2, f"Expected 2 pruned decoder blocks, got {num_blocks}"

    tpu_vae = tpu_vae.to("tpu")
    latents_cpu = torch.randn((1, latent_channels, 1, 4, 4), dtype=torch.float32)
    with torch.no_grad():
        cpu_out = cpu_vae.decode(latents_cpu, return_dict=False)[0]
        tpu_out = tpu_vae.decode(latents_cpu.to("tpu"), return_dict=False)[0]

    assert_parity(
        tpu_out,
        cpu_out,
        name="minimax_h3 litevae decode",
        max_rel_l2=LITEVAE_MAX_REL_L2,
    )
