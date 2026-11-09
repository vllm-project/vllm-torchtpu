# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Muse Spark Open waveform audio tower (offline, non streaming).

Port of ``MuseSparkAudioModel`` from transformers main: a causal depthwise
convolution pyramid that maps raw mono 24 kHz audio to one embedding per
``prod(patch_sizes)`` samples, already at the text hidden size. The streaming
padding cache is not needed for whole clip encoding: each causal conv simply
left pads with zeros, which is what HF does when no cache is passed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .configuration import MuseSparkAudioConfig


class MuseSparkAudioRMSNorm(nn.Module):
    """HF ``MuseSparkRMSNorm`` with a plain scale (no ``1 +`` offset)."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        out = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        return (out * self.weight.float()).type_as(x)


class MuseSparkAudioCausalConv1d(nn.Conv1d):
    @property
    def left_pad(self) -> int:
        return (self.kernel_size[0] - 1) * self.dilation[0] + 1 - self.stride[0]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(input, (self.left_pad, 0)))


class MuseSparkAudioPatchEmbedding(nn.Module):
    def __init__(self, config: MuseSparkAudioConfig, layer_idx: int) -> None:
        super().__init__()
        in_channels = 1 if layer_idx == 0 else config.hidden_sizes[layer_idx - 1]
        out_channels = config.hidden_sizes[layer_idx]
        self.patch_size = config.patch_sizes[layer_idx]
        self.proj = nn.Linear(in_channels * self.patch_size, out_channels, bias=True)
        self.norm = MuseSparkAudioRMSNorm(out_channels, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        if seq_len % self.patch_size:
            raise ValueError(
                f"audio length {seq_len} is not divisible by patch size "
                f"{self.patch_size}"
            )
        x = x.reshape(batch, seq_len // self.patch_size, self.patch_size * channels)
        return self.norm(self.proj(x))


class MuseSparkAudioMLP(nn.Module):
    def __init__(self, config: MuseSparkAudioConfig, layer_idx: int) -> None:
        super().__init__()
        hidden = config.hidden_sizes[layer_idx]
        inner = int(hidden * config.mlp_ratio)
        self.fc1 = nn.Linear(hidden, inner, bias=True)
        self.fc2 = nn.Linear(inner, hidden, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class MuseSparkAudioConvBlock(nn.Module):
    def __init__(self, config: MuseSparkAudioConfig, layer_idx: int) -> None:
        super().__init__()
        hidden = config.hidden_sizes[layer_idx]
        self.conv = MuseSparkAudioCausalConv1d(
            hidden,
            hidden,
            kernel_size=config.conv_kernel_size,
            groups=hidden,
            bias=True,
        )
        self.norm = MuseSparkAudioRMSNorm(hidden, eps=config.rms_norm_eps)
        self.mlp = MuseSparkAudioMLP(config, layer_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.mlp(self.norm(h)) + x


class MuseSparkAudioLayer(nn.Module):
    def __init__(self, config: MuseSparkAudioConfig, layer_idx: int) -> None:
        super().__init__()
        self.patch_embedder = MuseSparkAudioPatchEmbedding(config, layer_idx)
        self.conv_blocks = nn.ModuleList(
            [
                MuseSparkAudioConvBlock(config, layer_idx)
                for _ in range(config.depths[layer_idx])
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedder(x)
        for block in self.conv_blocks:
            x = block(x)
        return x


class MuseSparkAudioTower(nn.Module):
    """Raw waveform [B, S] (S a multiple of samples_per_token) to [B, T, out]."""

    def __init__(self, config: MuseSparkAudioConfig) -> None:
        super().__init__()
        if config.hidden_act != "gelu":
            raise ValueError(f"unsupported audio hidden_act {config.hidden_act}")
        self.config = config
        self.samples_per_token = config.samples_per_token
        self.layers = nn.ModuleList(
            [MuseSparkAudioLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = MuseSparkAudioRMSNorm(
            config.hidden_sizes[-1], eps=config.rms_norm_eps
        )
        self.proj = nn.Linear(
            config.hidden_sizes[-1], config.out_hidden_size, bias=True
        )
        self.post_proj_norm = MuseSparkAudioRMSNorm(
            config.out_hidden_size, eps=config.rms_norm_eps
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.proj.weight.dtype

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        if input_values.shape[-1] % self.samples_per_token:
            raise ValueError(
                f"input_values length {input_values.shape[-1]} is not a multiple "
                f"of {self.samples_per_token}"
            )
        x = input_values.to(device=self.proj.weight.device, dtype=self.dtype)
        x = x.unsqueeze(-1)
        for layer in self.layers:
            x = layer(x)
        return self.post_proj_norm(self.proj(self.norm(x)))

    def encode_clips(self, clips: list[torch.Tensor]) -> list[torch.Tensor]:
        """Encode unpadded 1D clips; returns [ceil(len / spt), out] per clip.

        Each clip is right padded with zeros to a token boundary and run on its
        own. The tower is causal, so this equals the HF batched forward
        followed by the ``input_values_mask`` token selection.
        """
        spt = self.samples_per_token
        outputs = []
        for clip in clips:
            clip = clip.reshape(-1)
            num_tokens = -(-clip.numel() // spt)
            padded = F.pad(clip, (0, num_tokens * spt - clip.numel()))
            outputs.append(self.forward(padded.unsqueeze(0))[0])
        return outputs
