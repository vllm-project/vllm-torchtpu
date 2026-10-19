# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replicated waveform encoder producing one token per 1,920 audio samples."""

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.models.utils import maybe_prefix


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Reshape [B, T, D] into [B, T // patch_size, patch_size * D]."""
    b, t, d = x.shape
    assert t % patch_size == 0, f"t={t} is not divisible by patch_size={patch_size}"
    return x.reshape(b, t // patch_size, patch_size * d)


class Patchifier(nn.Module):
    """Patchify -> Linear -> RMSNorm."""

    def __init__(
        self,
        patch_size: int,
        d_input: int,
        d_output: int,
        prefix: str = "",
        disable_tp: bool = True,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = ColumnParallelLinear(
            input_size=d_input * patch_size,
            output_size=d_output,
            bias=True,
            gather_output=True,
            prefix=maybe_prefix(prefix, "proj"),
            disable_tp=disable_tp,
        )
        self.norm = RMSNorm(d_output, eps=norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = patchify(x, self.patch_size)
        x, _ = self.proj(x)
        return self.norm(x)


class DepthwiseCausalConv(nn.Module):
    """Depthwise causal 1-D convolution with optional streaming cache."""

    def __init__(self, d_model: int, kernel_size: int) -> None:
        super().__init__()
        self._kernel_size = kernel_size
        conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            bias=True,
            groups=d_model,
        )
        self.weight = conv.weight
        self.bias = conv.bias

    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply causal convolution to [B, T, D], returning output and cache."""
        x = x.permute(0, 2, 1)  # [B, D, T]

        if cache is None:
            x = F.pad(x, (self._kernel_size - 1, 0))
            new_cache = None
        else:
            x = torch.cat((cache, x), dim=-1)
            new_cache = x[:, :, -self._kernel_size + 1 :]

        x = F.conv1d(x, self.weight, self.bias, groups=self.weight.shape[0])

        # RMSNorm requires contiguous output after the permutation.
        x = x.permute(0, 2, 1).contiguous()  # [B, T, D]
        return x, new_cache


class _AudioMLP(nn.Module):
    fc1: ColumnParallelLinear
    fc2: RowParallelLinear


class Block(nn.Module):
    """DepthwiseCausalConv -> RMSNorm -> FFN (GELU) with a residual."""

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        kernel_size: int,
        prefix: str = "",
        disable_tp: bool = True,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self._d_model = d_model
        self._kernel_size = kernel_size
        self.conv = DepthwiseCausalConv(d_model, kernel_size)
        self.norm = RMSNorm(d_model, eps=norm_eps)
        self.mlp = _AudioMLP()
        self.mlp.fc1 = ColumnParallelLinear(
            input_size=d_model,
            output_size=d_hidden,
            bias=True,
            prefix=maybe_prefix(prefix, "mlp.fc1"),
            disable_tp=disable_tp,
        )
        self.mlp.fc2 = RowParallelLinear(
            input_size=d_hidden,
            output_size=d_model,
            bias=True,
            prefix=maybe_prefix(prefix, "mlp.fc2"),
            disable_tp=disable_tp,
        )
        self.activation = nn.GELU()

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def kernel_size(self) -> int:
        return self._kernel_size

    def forward(
        self,
        inputs: torch.Tensor,
        cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = inputs
        x, cache = self.conv(x, cache)
        x = self.norm(x)
        x, _ = self.mlp.fc1(x)
        x = self.activation(x)
        x, _ = self.mlp.fc2(x)
        return x + inputs, cache


class _AudioStage(nn.Module):
    patch_embedder: Patchifier
    conv_blocks: nn.ModuleList


class AutoEncoder(nn.Module):
    """Waveform autoencoder with replicated weights."""

    def __init__(
        self,
        latent_channels: int,
        patch_size: int = 1920,
        kernel_size: int = 7,
        data_channels: int = 1,
        patchify_sizes: tuple[int, ...] = (16, 6, 5, 4),
        d_models: tuple[int, ...] = (256, 512, 1024, 2048),
        n_layers: tuple[int, ...] = (2, 2, 2, 2),
        prefix: str = "",
        disable_tp: bool = True,
        norm_eps: float = 1e-6,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.data_channels = data_channels

        d_model_list = [data_channels] + list(d_models)
        self.layers = nn.ModuleList(
            [
                self._make_stage(
                    d_input=d_model_list[stage_id],
                    d_model=d_model_list[stage_id + 1],
                    patch_size=patchify_sizes[stage_id],
                    n_layers_count=n_layers[stage_id],
                    s_kernel=kernel_size,
                    mlp_ratio=mlp_ratio,
                    norm_eps=norm_eps,
                    prefix=maybe_prefix(prefix, f"layers.{stage_id}"),
                    disable_tp=disable_tp,
                )
                for stage_id in range(len(n_layers))
            ]
        )

        self.norm = RMSNorm(d_model_list[-1], eps=norm_eps)
        self.proj = ColumnParallelLinear(
            input_size=d_model_list[-1],
            output_size=latent_channels,
            bias=True,
            gather_output=True,
            prefix=maybe_prefix(prefix, "proj"),
            disable_tp=disable_tp,
        )
        self.post_proj_norm = RMSNorm(latent_channels, eps=norm_eps)

    @staticmethod
    def _make_stage(
        d_input: int,
        d_model: int,
        patch_size: int,
        n_layers_count: int,
        s_kernel: int,
        prefix: str = "",
        disable_tp: bool = True,
        norm_eps: float = 1e-6,
        mlp_ratio: float = 4.0,
    ) -> _AudioStage:
        stage = _AudioStage()
        stage.patch_embedder = Patchifier(
            patch_size,
            d_input,
            d_model,
            norm_eps=norm_eps,
            prefix=maybe_prefix(prefix, "patch_embedder"),
            disable_tp=disable_tp,
        )
        stage.conv_blocks = nn.ModuleList(
            [
                Block(
                    d_model,
                    int(d_model * mlp_ratio),
                    norm_eps=norm_eps,
                    kernel_size=s_kernel,
                    prefix=maybe_prefix(prefix, f"conv_blocks.{i}"),
                    disable_tp=disable_tp,
                )
                for i in range(n_layers_count)
            ]
        )
        return stage

    def _encode_prefill(self, x: torch.Tensor) -> torch.Tensor:
        """Encode waveforms without streaming context."""
        for stage in self.layers:
            stage = cast(_AudioStage, stage)
            x = stage.patch_embedder(x)
            for block in stage.conv_blocks:
                x, _ = block(x, None)
        x = self.norm(x)
        x, _ = self.proj(x)
        x = self.post_proj_norm(x)
        return x

    def forward(self, wavs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Encode a list of waveforms into latent token sequences."""
        if not wavs:
            return []

        def _pad(wav: torch.Tensor) -> torch.Tensor:
            wav = wav.reshape(-1, self.data_channels)
            pad_size = (self.patch_size - wav.shape[0]) % self.patch_size
            return F.pad(wav, (0, 0, 0, pad_size))

        embeds: list[torch.Tensor] = []
        for wav in wavs:
            wav_batch = _pad(wav).unsqueeze(0)  # [1, T, C]
            embed = self._encode_prefill(wav_batch)
            embeds.append(embed.squeeze(0))  # [num_tokens, latent_channels]
        return embeds
