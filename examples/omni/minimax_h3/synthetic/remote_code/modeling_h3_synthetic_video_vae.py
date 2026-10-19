# SPDX-License-Identifier: Apache-2.0
"""Synthetic, randomly initialised video VAE for MiniMax-H3 serving tests.

Written from scratch for the vllm-omni TPU bring-up. It implements only the
remote-code surface that vllm-omni's MiniMax-H3 adapter calls
(``vllm_omni/diffusion/models/minimax_h3/vae.py``, ``temporal_chunks.py``,
``chunked_decode.py``) with the latent geometry those files and the H3
pipeline pin:

* 24 latent channels and 16x spatial compression (pipeline ``height // 16``);
* causal 17-frame clips -> 5 latent tokens (token 0 = frame 0 alone, token k =
  frames 4k-3..4k), with the last 3 tokens of every video dropped, so
  ``17n + 5`` frames <-> ``5n + 2`` latents (``time_request.py``,
  ``encoder_processing._temporal_group_max_pool``);
* a single image -> 1 latent frame.

Both directions are token-local in time (stride == kernel), so the chunked
temporal decode and the whole-video decode produce identical frames. The
weights are random and carry no information.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from .parallel import ensure_parallel_state, get_parallel_state

CONFIG_NAME = "config.json"
ARCH_KEY = "synthetic_architecture"


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class _Normalize:
    """(mean, std) holder with torchvision ``Normalize`` semantics: (x - mean) / std."""

    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        self.mean = [float(value) for value in mean]
        self.std = [float(value) for value in std]


class SyntheticVideoProcessor:
    """Host-side geometry helpers read by vllm-omni's video VAE adapter."""

    def __init__(
        self,
        *,
        spatial_ratio: int,
        temporal_ratio: int,
        clip_frames: int,
        tokens_per_clip: int,
        encode_token_drop: int,
    ) -> None:
        self.spatial_ratio = int(spatial_ratio)
        self.temporal_ratio = int(temporal_ratio)
        self.clip_frames = int(clip_frames)
        self.tokens_per_clip = int(tokens_per_clip)
        self.encode_token_drop = int(encode_token_drop)
        # Pixels in [0, 1] -> model space [-1, 1], and its inverse.
        self.transform = _Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.transform_rev = _Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))
        self.use_3d_conv = True
        self.frame_overlap = 0
        self.isolated_last_frame = False

    def _align_to_total_patch_size(self, height: int, width: int) -> tuple[int, int]:
        ratio = self.spatial_ratio
        height, width = int(height), int(width)
        if height < ratio or width < ratio:
            raise ValueError(
                f"frames must be at least {ratio}x{ratio}, got {width}x{height}"
            )
        return height // ratio * ratio, width // ratio * ratio

    def get_latent_length(self, num_frames: int) -> int:
        frames = int(num_frames)
        if frames <= 0:
            raise ValueError(f"num_frames must be positive, got {frames}")
        if frames == 1:
            return 1
        clips = -(-frames // self.clip_frames)
        return clips * self.tokens_per_clip - self.encode_token_drop

    def get_frame_count(self, latent_t: int) -> int:
        full, rest = divmod(int(latent_t), self.tokens_per_clip)
        return full * self.clip_frames + (
            0 if rest == 0 else 1 + self.temporal_ratio * (rest - 1)
        )

    def revert_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(2)
        mean = torch.as_tensor(
            self.transform_rev.mean, dtype=tensor.dtype, device=tensor.device
        ).view(1, 3, 1, 1, 1)
        std = torch.as_tensor(
            self.transform_rev.std, dtype=tensor.dtype, device=tensor.device
        ).view(1, 3, 1, 1, 1)
        return ((tensor - mean) / std).clamp(0.0, 1.0)


class _ResidualStack(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )
        self.conv2 = nn.Conv3d(
            channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.conv2(F.silu(self.conv1(F.silu(hidden))))


class SyntheticVideoEncoder(nn.Module):
    def __init__(
        self, hidden_channels: int, spatial_ratio: int, temporal_ratio: int
    ) -> None:
        super().__init__()
        kernel = (temporal_ratio, spatial_ratio, spatial_ratio)
        self.patch_in = nn.Conv3d(3, hidden_channels, kernel_size=kernel, stride=kernel)
        self.mid = _ResidualStack(hidden_channels)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.mid(self.patch_in(frames))


class SyntheticVideoDecoder(nn.Module):
    def __init__(
        self, hidden_channels: int, spatial_ratio: int, temporal_ratio: int
    ) -> None:
        super().__init__()
        kernel = (temporal_ratio, spatial_ratio, spatial_ratio)
        self.mid = _ResidualStack(hidden_channels)
        self.patch_out = nn.ConvTranspose3d(
            hidden_channels, 3, kernel_size=kernel, stride=kernel
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.patch_out(F.silu(self.mid(hidden))))


class SyntheticH3VideoAutoencoder(nn.Module):
    """The ``remote.model`` object vllm-omni's ``MiniMaxH3VideoVAE`` drives."""

    # vllm-omni ``_load_video_vae_encoder`` keyword -> constructor argument.
    _LOADER_KWARGS = {
        "clip_length": "clip_frames",
        "token_drop": "encode_token_drop",
        "encoder_tiling": "encoder_tiling",
        "decoder_tiling": "decoder_tiling",
        "parallel_tiling": "parallel_tiling",
        "tile_size": "tile_size",
        "tile_overlap_min": "tile_overlap_min",
        "encoder_parallel": "encoder_parallel",
        "decoder_parallel": "decoder_parallel",
        "chunk_dim": "chunk_dim",
    }
    _CHECKED_KWARGS = ("clip_frames", "encode_token_drop")

    def __init__(
        self,
        *,
        latent_channels: int,
        hidden_channels: int,
        spatial_ratio: int,
        temporal_ratio: int,
        clip_frames: int,
        tokens_per_clip: int,
        encode_token_drop: int,
        parallel_tiling: int | bool = 0,
        encoder_tiling: int | bool = 0,
        decoder_tiling: int | bool = 0,
        tile_size: int = 0,
        tile_overlap_min: int = 0,
        encoder_parallel: int | bool = 0,
        decoder_parallel: int | bool = 0,
        chunk_dim: int = 2,
    ) -> None:
        super().__init__()
        if int(clip_frames) != 1 + int(temporal_ratio) * (int(tokens_per_clip) - 1):
            raise ValueError(
                "clip_frames must equal 1 + temporal_ratio * (tokens_per_clip - 1)"
            )
        if not 0 <= int(encode_token_drop) < int(tokens_per_clip):
            raise ValueError("encode_token_drop must be in [0, tokens_per_clip)")
        self.latent_channels = int(latent_channels)
        self.encoder = SyntheticVideoEncoder(
            int(hidden_channels), int(spatial_ratio), int(temporal_ratio)
        )
        self.quant_conv = nn.Conv3d(
            int(hidden_channels), 2 * self.latent_channels, kernel_size=1
        )
        self.post_quant_conv = nn.Conv3d(
            self.latent_channels, int(hidden_channels), kernel_size=1
        )
        self.decoder = SyntheticVideoDecoder(
            int(hidden_channels), int(spatial_ratio), int(temporal_ratio)
        )
        self.processor = SyntheticVideoProcessor(
            spatial_ratio=int(spatial_ratio),
            temporal_ratio=int(temporal_ratio),
            clip_frames=int(clip_frames),
            tokens_per_clip=int(tokens_per_clip),
            encode_token_drop=int(encode_token_drop),
        )
        self.vae_ratio = int(spatial_ratio)
        self.vae_ratio_t = int(temporal_ratio)
        self.temporal_clip_frames = int(clip_frames)
        self.encode_token_drop = int(encode_token_drop)
        # Temporal decode contract read by vllm-omni
        # temporal_chunks.decode_temporal_chunks.
        # Decoding pads to whole clips itself, so no boundary tokens are isolated.
        self.use_3d_conv = True
        self.token_drop = 0
        self.tokens_chunk_size = int(tokens_per_clip)
        self.token_overlap = 0
        self.frame_overlap = 0
        self.frame_pre_padding = 0
        self.isolated_first_frame = False
        self.isolated_last_frame = False
        self.isolated_key_frame = False
        self.parallel_tiling = bool(parallel_tiling)
        self.encoder_tiling = bool(encoder_tiling)
        self.decoder_tiling = bool(decoder_tiling)
        self.tile_size = int(tile_size)
        self.tile_overlap_min = int(tile_overlap_min)
        self.encoder_parallel = bool(encoder_parallel)
        self.decoder_parallel = bool(decoder_parallel)
        self.chunk_dim = int(chunk_dim)

    # -- construction ------------------------------------------------------
    @classmethod
    def load_config(cls, path: str | Path) -> dict[str, Any]:
        return dict(_read_json(Path(path) / CONFIG_NAME)[ARCH_KEY])

    @classmethod
    def from_config(
        cls, config: dict[str, Any], return_unused_kwargs: bool = False, **kwargs: Any
    ):
        args = dict(config)
        unused: dict[str, Any] = {}
        for key, value in kwargs.items():
            target = cls._LOADER_KWARGS.get(key)
            if target is None:
                unused[key] = value
                continue
            if (
                target in cls._CHECKED_KWARGS
                and target in args
                and int(args[target]) != int(value)
            ):
                raise ValueError(
                    f"loader {key}={value} disagrees with architecture "
                    f"{target}={args[target]}"
                )
            args[target] = value
        accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
        for key in list(args):
            if key not in accepted:
                unused[key] = args.pop(key)
        model = cls(**args)
        return (model, unused) if return_unused_kwargs else model

    # -- helpers -----------------------------------------------------------
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _to_model_space(self, pixels01: torch.Tensor) -> torch.Tensor:
        transform = self.processor.transform
        mean = torch.as_tensor(
            transform.mean, dtype=pixels01.dtype, device=pixels01.device
        ).view(3, 1, 1, 1)
        std = torch.as_tensor(
            transform.std, dtype=pixels01.dtype, device=pixels01.device
        ).view(3, 1, 1, 1)
        return (pixels01 - mean) / std

    def _video_to_model_space(self, video: Any) -> torch.Tensor:
        """uint8 (T, H, W, 3) or normalized float (3, T, H, W) -> (1, 3, T, H', W')."""
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(np.ascontiguousarray(video))
        if not torch.is_tensor(video):
            raise TypeError(f"unsupported video input {type(video)!r}")
        device = self._device()
        if video.dtype == torch.uint8:
            if video.ndim != 4 or int(video.shape[-1]) != 3:
                raise ValueError(
                    f"uint8 video must be (T, H, W, 3), got {tuple(video.shape)}"
                )
            pixels = video.to(device).permute(3, 0, 1, 2).to(torch.float32).div_(255.0)
            video = self._to_model_space(pixels)
        else:
            if video.ndim != 4 or int(video.shape[0]) != 3:
                raise ValueError(
                    f"float video must be normalized (3, T, H, W), "
                    f"got {tuple(video.shape)}"
                )
            video = video.to(device=device, dtype=torch.float32)
        height, width = self.processor._align_to_total_patch_size(
            int(video.shape[-2]), int(video.shape[-1])
        )
        return video[None, :, :, :height, :width]

    def _image_to_model_space(self, image: Any) -> torch.Tensor:
        if hasattr(image, "convert"):
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)[None]
            return self._video_to_model_space(array)
        if isinstance(image, np.ndarray) and image.ndim == 3:
            return self._video_to_model_space(image[None])
        if torch.is_tensor(image) and image.ndim == 3:
            return self._video_to_model_space(
                image.unsqueeze(1) if image.shape[0] == 3 else image[None]
            )
        return self._video_to_model_space(image)

    def _encode_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        moments = self.quant_conv(self.encoder(frames))
        return moments[:, : self.latent_channels]

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Model-space (B, 3, T, H, W) -> posterior-mean latent (B, C, T', H/r, W/r)."""
        num_frames = int(frames.shape[2])
        ratio_t = self.vae_ratio_t
        if num_frames == 1:
            return self._encode_tokens(frames[:, :, [0] * ratio_t])
        clip = self.temporal_clip_frames
        latents = []
        for start in range(0, num_frames, clip):
            last = num_frames - 1
            index = [min(start, last)] * ratio_t + [
                min(start + offset, last) for offset in range(1, clip)
            ]
            latents.append(self._encode_tokens(frames[:, :, index]))
        latent = torch.cat(latents, dim=2)
        if self.encode_token_drop:
            latent = latent[:, :, : int(latent.shape[2]) - self.encode_token_drop]
        return latent

    @staticmethod
    def _finish(latent: torch.Tensor, use_fp16_latent: bool) -> torch.Tensor:
        return latent.to(torch.float16) if use_fp16_latent else latent

    # -- encode API used by vllm-omni ---------------------------------------
    @torch.no_grad()
    def encode_images(
        self, images: Any, *, use_fp16_latent: bool = False
    ) -> list[torch.Tensor]:
        items = images if isinstance(images, (list, tuple)) else [images]
        return [
            self._finish(
                self.encode_frames(self._image_to_model_space(item))[0], use_fp16_latent
            )
            for item in items
        ]

    @torch.no_grad()
    def encode_videos(
        self, videos: Any, *, use_fp16_latent: bool = False
    ) -> list[torch.Tensor]:
        items = videos if isinstance(videos, (list, tuple)) else [videos]
        return [
            self._finish(
                self.encode_frames(self._video_to_model_space(item))[0], use_fp16_latent
            )
            for item in items
        ]

    # -- decode API used by vllm-omni ---------------------------------------
    def _decode_clip(self, latent: torch.Tensor) -> torch.Tensor:
        frames = self.decoder(self.post_quant_conv(latent))
        # Token 0 of a clip decodes to ratio_t frames; keep only the
        # last (causal key frame).
        return frames[:, :, self.vae_ratio_t - 1 :]

    def _adaptive_decode(self, latent: torch.Tensor) -> torch.Tensor:
        step = self.tokens_chunk_size
        parts = [
            self._decode_clip(latent[:, :, start : start + step])
            for start in range(0, int(latent.shape[2]), step)
        ]
        return torch.cat(parts, dim=2)

    @torch.no_grad()
    def decode_base(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 5:
            raise ValueError(
                f"latent must be (B, C, T, H, W), got {tuple(latent.shape)}"
            )
        return self._adaptive_decode(latent)

    def _decode_temporal_output_frame_plan(
        self,
        latent: torch.Tensor,
        z_head: torch.Tensor | None,
        z_tail: torch.Tensor | None,
        num_chunks: int,
        pad_tokens: int,
    ) -> tuple[int, int, int]:
        if z_head is not None or z_tail is not None:
            raise ValueError("the synthetic H3 VAE has no isolated boundary latents")
        total_frames = int(num_chunks) * self.temporal_clip_frames
        pad_frames = int(pad_tokens) * self.vae_ratio_t
        return total_frames, pad_frames, total_frames - pad_frames

    @staticmethod
    def blend(
        previous: torch.Tensor, current: torch.Tensor, overlap: int, dim: int = -3
    ) -> torch.Tensor:
        overlap = int(overlap)
        if overlap <= 0:
            return current
        weight_shape = [1] * current.ndim
        weight_shape[dim] = overlap
        weight = torch.linspace(
            0.0, 1.0, overlap, device=current.device, dtype=current.dtype
        ).view(weight_shape)
        head = previous.narrow(dim, previous.shape[dim] - overlap, overlap) * (
            1 - weight
        )
        head = head + current.narrow(dim, 0, overlap) * weight
        return torch.cat(
            (head, current.narrow(dim, overlap, current.shape[dim] - overlap)), dim=dim
        )

    def split_tiles(
        self, size: int, is_decode: bool = False
    ) -> tuple[list[tuple[int, int]], list[int], int]:
        """Single untiled span: the stride-equals-kernel codec needs no
        spatial tiles.
        """
        del is_decode
        return [(0, int(size))], [0], int(size)

    def _all_gather_tiled_results(
        self, tasks: list[torch.Tensor], num_tiles: int
    ) -> list[torch.Tensor]:
        state = get_parallel_state()
        if int(state.get("sp_size", 1)) != 1:
            raise NotImplementedError(
                "the synthetic H3 VAE never splits work across tile ranks"
            )
        if len(tasks) != int(num_tiles):
            raise ValueError(f"expected {num_tiles} tiles, got {len(tasks)}")
        return list(tasks)


class SyntheticH3VideoVAE(nn.Module):
    """``auto_map.AutoModel`` entry point: ``from_pretrained`` -> model."""

    def __init__(self, model: SyntheticH3VideoAutoencoder) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Path, **kwargs: Any
    ) -> SyntheticH3VideoVAE:
        del kwargs
        root = Path(pretrained_model_name_or_path)
        top = _read_json(root / CONFIG_NAME)
        model, _unused = SyntheticH3VideoAutoencoder.from_config(
            SyntheticH3VideoAutoencoder.load_config(root),
            return_unused_kwargs=True,
            clip_length=int(top["vae_clip_length"]),
            token_drop=int(top["vae_token_drop"]),
            encoder_tiling=int(top["vae_encoder_tiling"]),
            decoder_tiling=int(top["vae_decoder_tiling"]),
            parallel_tiling=int(top["vae_parallel_tiling"]),
            tile_size=int(top["vae_tile_size"]),
            tile_overlap_min=int(top["vae_tile_overlap_min"]),
            encoder_parallel=int(top["vae_encoder_parallel"]),
            decoder_parallel=int(top["vae_decoder_parallel"]),
            chunk_dim=int(top["vae_chunk_dim"]),
        )
        weights = root / str(top["source_path"]) / str(top["source_safetensors_path"])
        model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
        if model.parallel_tiling:
            ensure_parallel_state()
        model.eval()
        return cls(model)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("call encode_images / encode_videos / decode_base on .model")


# Read by vllm-omni ``_load_video_vae_encoder`` for encoder-only loading.
_SOURCE_CLASSES = {"SyntheticH3VideoAutoencoder": SyntheticH3VideoAutoencoder}


def _ensure_vae_parallel_state() -> dict[str, Any]:
    return ensure_parallel_state()


__all__ = ["SyntheticH3VideoAutoencoder", "SyntheticH3VideoVAE"]
