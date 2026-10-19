# SPDX-License-Identifier: Apache-2.0
"""Synthetic, randomly initialised audio VAE for MiniMax-H3 serving tests.

Written from scratch for the vllm-omni TPU bring-up. It implements the remote
surface vllm-omni's ``MiniMaxH3AudioVAE`` calls (``preprocess``, ``encoder``,
``pre_block``, ``mean_proj``, ``remote.decode``) and the source-loader hooks
its encoder-only path needs (``_load_yaml`` and a class named ``DacAudioVAE``
-- that name is the interface vllm-omni looks up, nothing more). Geometry:
mono waveforms at 32 kHz (stereo is carried as batch 2) and a 40 Hz latent
(hop 800 samples) with 32 channels, as pinned by vllm-omni. Weights are random.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

CONFIG_NAME = "config.json"
ENCODER_BASE_CHANNELS = 8
ENCODER_MAX_CHANNELS = 64
DECODER_MIN_CHANNELS = 8
DECODER_TYPE = "synthetic_conv_transpose"


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_yaml(path: str | Path) -> dict[str, Any]:
    """The generator writes this file as JSON, which is a subset of YAML 1.2."""
    return _read_json(path)


class SyntheticAudioEncoder(nn.Module):
    def __init__(self, rates: list[int]) -> None:
        super().__init__()
        channels = ENCODER_BASE_CHANNELS
        self.stem = nn.Conv1d(1, channels, kernel_size=7, padding=3)
        self.downs = nn.ModuleList()
        for rate in rates:
            next_channels = min(2 * channels, ENCODER_MAX_CHANNELS)
            self.downs.append(
                nn.Conv1d(
                    channels, next_channels, kernel_size=int(rate), stride=int(rate)
                )
            )
            channels = next_channels
        self.out = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.out_channels = channels

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(audio)
        for down in self.downs:
            hidden = down(F.silu(hidden))
        return self.out(F.silu(hidden))


class SyntheticAudioPreBlock(nn.Module):
    """Residual MLP over (B, T, C) latent frames."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.fc2(F.silu(self.fc1(self.norm(hidden))))


class SyntheticAudioDecoder(nn.Module):
    def __init__(self, rates: list[int], dim: int) -> None:
        super().__init__()
        channels = int(dim)
        self.ups = nn.ModuleList()
        for rate in rates:
            next_channels = max(channels // 2, DECODER_MIN_CHANNELS)
            self.ups.append(
                nn.ConvTranspose1d(
                    channels, next_channels, kernel_size=int(rate), stride=int(rate)
                )
            )
            channels = next_channels
        self.out = nn.Conv1d(channels, 1, kernel_size=7, padding=3)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for up in self.ups:
            hidden = up(F.silu(hidden))
        return torch.tanh(self.out(F.silu(hidden)))


class DacAudioVAE(nn.Module):
    """``remote.model`` for vllm-omni's ``MiniMaxH3AudioVAE`` (synthetic, not DAC)."""

    def __init__(
        self,
        *,
        encoder_rates: list[int],
        decoder_rates: list[int],
        attn_proj: bool,
        decoder_type: str,
        decoder_dim: int,
        vae_latent_channels: int,
        sample_rate: int,
    ) -> None:
        super().__init__()
        if decoder_type != DECODER_TYPE:
            raise ValueError(
                f"unsupported decoder_type {decoder_type!r}; expected {DECODER_TYPE!r}"
            )
        hop = math.prod(int(rate) for rate in encoder_rates)
        if math.prod(int(rate) for rate in decoder_rates) != hop:
            raise ValueError("encoder and decoder rates must share one hop length")
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop)
        self.latent_dim = int(vae_latent_channels)
        self.attn_proj = bool(attn_proj)
        self.encoder = SyntheticAudioEncoder([int(rate) for rate in encoder_rates])
        width = self.encoder.out_channels
        if self.attn_proj:
            self.pre_block = SyntheticAudioPreBlock(width)
        self.mean_proj = nn.Conv1d(width, self.latent_dim, kernel_size=1)
        self.logs_proj = nn.Conv1d(width, self.latent_dim, kernel_size=1)
        self.dec_in_proj = nn.Conv1d(self.latent_dim, int(decoder_dim), kernel_size=1)
        self.decoder = SyntheticAudioDecoder(
            [int(rate) for rate in decoder_rates], int(decoder_dim)
        )

    def preprocess(
        self, audio_data: torch.Tensor, sample_rate: int | None
    ) -> torch.Tensor:
        if sample_rate is not None and int(sample_rate) != self.sample_rate:
            raise ValueError(f"expected {self.sample_rate} Hz audio, got {sample_rate}")
        if audio_data.ndim != 3 or int(audio_data.shape[1]) != 1:
            raise ValueError(
                f"audio must be (B, 1, samples), got {tuple(audio_data.shape)}"
            )
        pad = (-int(audio_data.shape[-1])) % self.hop_length
        return F.pad(audio_data, (0, pad)) if pad else audio_data

    def encode(self, audio_data: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(self.preprocess(audio_data, self.sample_rate))
        if self.attn_proj:
            latent = self.pre_block(latent.transpose(1, 2)).transpose(1, 2)
        return self.mean_proj(latent)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.dec_in_proj(latent))


class SyntheticH3AudioVAE(nn.Module):
    """``auto_map.AutoModel`` entry point: ``from_pretrained`` -> model/decode."""

    def __init__(self, model: DacAudioVAE) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Path, **kwargs: Any
    ) -> SyntheticH3AudioVAE:
        del kwargs
        root = Path(pretrained_model_name_or_path)
        top = _read_json(root / CONFIG_NAME)
        audio_config = _load_yaml(root / str(top["source_config_path"]))
        metadata = _read_json(root / str(top["source_metadata_path"]))["metadata"][
            "kwargs"
        ]
        model = DacAudioVAE(
            encoder_rates=metadata["encoder_rates"],
            decoder_rates=metadata["decoder_rates"],
            attn_proj=metadata["attn_proj"],
            decoder_type=metadata["decoder_type"],
            decoder_dim=audio_config["model_config"]["decoder_dim"],
            vae_latent_channels=audio_config["model_config"]["vae_latent_channels"],
            sample_rate=metadata["sample_rate"],
        )
        model.load_state_dict(
            load_file(str(root / str(top["source_safetensors_path"])), device="cpu"),
            strict=True,
        )
        model.eval()
        return cls(model)

    def encode(self, audio_data: torch.Tensor) -> torch.Tensor:
        return self.model.encode(audio_data)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.model.decode(latent)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decode(latent)


__all__ = ["DacAudioVAE", "SyntheticH3AudioVAE"]
