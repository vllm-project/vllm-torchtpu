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
"""Muse Spark Open vision tower, adapter and projection.

Port of ``MuseSparkVisionModel``, ``MuseSparkVisionAdapter`` and the vision
part of ``MuseSparkModel.get_image_features`` from transformers main. The
geometry helpers below reproduce ``transformers.vision_utils`` (Apache 2.0)
with ``spatial_merge_size=1`` as the HF encoder calls them, including the
reference's swapped (w, h) read of the position embedding table.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.models.vision import (
    get_fp8_padded_hidden_size,
    is_vit_use_data_parallel,
)

from .configuration import MuseSparkConfig, MuseSparkVisionConfig


def weightless_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """HF ``MuseSparkRMSNorm(with_scale=False)``: fp32 compute, cast back."""
    xf = x.float()
    out = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    return out.type_as(x)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@dataclass
class VisionGeometry:
    """Per batch index tensors derived from ``grid_thw`` (host computed)."""

    num_patches: int
    # [N] gather order into window layout; argsort restores raster order.
    window_index: torch.Tensor
    window_seqlens: list[int]
    full_seqlens: list[int]
    # [N, 2] rope positions (w + 1, h + 1) in window order.
    rope_positions: torch.Tensor
    # [N, 4] position table taps and bilinear weights in raster order.
    interp_indices: torch.Tensor
    interp_weights: torch.Tensor
    # [N] gather that groups each merge x merge block contiguously.
    shuffle_index: torch.Tensor


def _axis_taps(
    index: np.ndarray, size: np.ndarray, side: int
) -> tuple[np.ndarray, np.ndarray]:
    # Bilinear, align_corners=False, zero padding (F.grid_sample semantics).
    src = (index.astype(np.float32) + 0.5) * side / size.astype(np.float32) - 0.5
    floor = np.floor(src)
    offsets = np.arange(2)
    raw = floor.astype(np.int64)[:, None] + offsets
    taps = np.clip(raw, 0, side - 1)
    dist = np.abs(src[:, None] - floor[:, None] - offsets)
    weights = np.clip(1.0 - dist, 0.0, None)
    weights = weights * ((raw >= 0) & (raw <= side - 1))
    return taps, weights.astype(np.float32)


def compute_vision_geometry(
    grid_thw: Sequence[Sequence[int]],
    *,
    pos_emb_side: int,
    window_patches: int,
    merge_size: int,
) -> VisionGeometry:
    window_index: list[np.ndarray] = []
    window_seqlens: list[int] = []
    full_seqlens: list[int] = []
    positions: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    row_sizes: list[np.ndarray] = []
    col_sizes: list[np.ndarray] = []
    shuffle: list[np.ndarray] = []
    offset = 0
    for t, h, w in grid_thw:
        t, h, w = int(t), int(h), int(w)
        if h % merge_size or w % merge_size:
            raise ValueError(f"grid ({h}, {w}) is not divisible by {merge_size}")
        hw = h * w
        # Window attention: per frame, ws x ws patch windows in raster order.
        ws = window_patches
        idx = np.arange(t * hw).reshape(t, h, w)
        pad_h, pad_w = ws - h % ws, ws - w % ws
        nwh, nww = (h + pad_h) // ws, (w + pad_w) // ws
        padded = np.pad(idx, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=-1)
        padded = padded.reshape(t, nwh, ws, nww, ws).transpose(0, 1, 3, 2, 4)
        padded = padded.reshape(t, nwh * nww, ws * ws)
        lens = (padded != -1).sum(-1).reshape(-1)
        flat = padded.reshape(-1)
        window_index.append(flat[flat != -1] + offset)
        window_seqlens.extend(int(n) for n in lens if n > 0)
        # Full attention runs per frame (HF merge_temporal=False).
        full_seqlens.extend([hw] * t)
        # Rope positions are raster (h, w), flipped to (w, h), offset by 1.
        hpos, wpos = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        pos = np.stack([wpos.reshape(-1), hpos.reshape(-1)], axis=-1) + 1
        positions.append(np.tile(pos, (t, 1)))
        # Position table read with (h, w) swapped, as the reference does.
        within = np.tile(np.arange(hw), t)
        rows.append(within // h)
        cols.append(within % h)
        row_sizes.append(np.full(t * hw, w))
        col_sizes.append(np.full(t * hw, h))
        perm = np.arange(hw).reshape(
            h // merge_size, merge_size, w // merge_size, merge_size
        )
        perm = perm.transpose(0, 2, 1, 3).reshape(-1)
        perm = (perm[None, :] + np.arange(t)[:, None] * hw).reshape(-1)
        shuffle.append(perm + offset)
        offset += t * hw

    win = np.concatenate(window_index)
    h_taps, h_w = _axis_taps(
        np.concatenate(rows), np.concatenate(row_sizes), pos_emb_side
    )
    w_taps, w_w = _axis_taps(
        np.concatenate(cols), np.concatenate(col_sizes), pos_emb_side
    )
    interp_idx = (h_taps[:, :, None] * pos_emb_side + w_taps[:, None, :]).reshape(-1, 4)
    interp_w = (h_w[:, :, None] * w_w[:, None, :]).reshape(-1, 4)
    pos = np.concatenate(positions)[win]
    return VisionGeometry(
        num_patches=offset,
        window_index=torch.from_numpy(win),
        window_seqlens=window_seqlens,
        full_seqlens=full_seqlens,
        rope_positions=torch.from_numpy(pos),
        interp_indices=torch.from_numpy(interp_idx),
        interp_weights=torch.from_numpy(interp_w),
        shuffle_index=torch.from_numpy(np.concatenate(shuffle)),
    )


class MuseSparkVisionAttention(nn.Module):
    def __init__(self, config: MuseSparkVisionConfig, prefix: str = "") -> None:
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        if hidden_size % num_heads:
            raise ValueError("vision hidden_size must be divisible by num heads")
        self.head_dim = hidden_size // num_heads
        self.use_data_parallel = is_vit_use_data_parallel(num_heads)
        self.tp_size = (
            1 if self.use_data_parallel else get_tensor_model_parallel_world_size()
        )
        self.num_heads = divide(num_heads, self.tp_size)
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=True,
            prefix=f"{prefix}.qkv_proj",
            disable_tp=self.use_data_parallel,
        )
        self.proj = RowParallelLinear(
            input_size=hidden_size,
            output_size=hidden_size,
            bias=True,
            prefix=f"{prefix}.proj",
            disable_tp=self.use_data_parallel,
        )
        self.attn = MMEncoderAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.head_dim**-0.5,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.view(seq_len, 3, self.num_heads, self.head_dim).unbind(1)
        # HF apply_rotary_pos_emb_vision: fp32 rotation over the full head dim.
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        qf, kf = q.float(), k.float()
        q = (qf * cos + _rotate_half(qf) * sin).to(v.dtype)
        k = (kf * cos + _rotate_half(kf) * sin).to(v.dtype)
        out = self.attn(
            query=q.unsqueeze(0),
            key=k.unsqueeze(0),
            value=v.unsqueeze(0),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        out, _ = self.proj(out.reshape(seq_len, -1))
        return out


class MuseSparkVisionMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        use_data_parallel: bool,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=True,
            prefix=f"{prefix}.fc1",
            disable_tp=use_data_parallel,
        )
        self.fc2 = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=True,
            prefix=f"{prefix}.fc2",
            disable_tp=use_data_parallel,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x, _ = self.fc2(F.gelu(x))
        return x


class MuseSparkVisionEncoderLayer(nn.Module):
    def __init__(self, config: MuseSparkVisionConfig, prefix: str = "") -> None:
        super().__init__()
        if config.hidden_act != "gelu":
            raise ValueError(f"unsupported vision hidden_act {config.hidden_act}")
        # HF hardcodes eps 1e-5 for the per layer norms.
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.attn = MuseSparkVisionAttention(config, prefix=f"{prefix}.attn")
        self.mlp = MuseSparkVisionMLP(
            config.hidden_size,
            config.intermediate_size,
            self.attn.use_data_parallel,
            prefix=f"{prefix}.mlp",
        )

    def forward(self, hidden_states: torch.Tensor, cos, sin, *attn_meta):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cos, sin, *attn_meta
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class MuseSparkVisionPatchEmbedder(nn.Module):
    def __init__(self, config: MuseSparkVisionConfig) -> None:
        super().__init__()
        if config.pos_emb_height != config.pos_emb_width:
            raise ValueError("only square position embedding tables are supported")
        self.patch_embedding = nn.Linear(
            config.patch_temporal * 3 * config.patch_size**2,
            config.hidden_size,
            bias=False,
        )
        self.position_embedding_table = nn.Embedding(
            config.pos_emb_height * config.pos_emb_width, config.hidden_size
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        interp_indices: torch.Tensor,
        interp_weights: torch.Tensor,
    ) -> torch.Tensor:
        weight = self.patch_embedding.weight
        embeddings = self.patch_embedding(pixel_values.to(weight.dtype))
        taps = self.position_embedding_table(interp_indices)
        pos = (taps * interp_weights[:, :, None]).sum(1)
        return embeddings + pos.to(embeddings.dtype)


class MuseSparkVisionTower(nn.Module):
    """ViT over packed patches; returns pixel shuffled features [M, 4 * D]."""

    def __init__(self, config: MuseSparkVisionConfig, prefix: str = "") -> None:
        super().__init__()
        rope = config.rope_parameters or {}
        if rope.get("rope_type", "axial") != "axial":
            raise ValueError(f"vision rope_type must be axial, got {rope}")
        self.config = config
        self.hidden_size = config.hidden_size
        self.merge_size = config.merge_size
        self.layer_types = list(config.layer_types)
        self.head_dim = config.hidden_size // config.num_attention_heads
        if self.head_dim % 4:
            raise ValueError("vision head_dim must be divisible by 4 for axial rope")
        spatial_dim = self.head_dim // 2
        theta = float(rope.get("rope_theta", 10000.0))
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.patch_embedder = MuseSparkVisionPatchEmbedder(config)
        self.ln_pre = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList(
            [
                MuseSparkVisionEncoderLayer(config, prefix=f"{prefix}.layers.{i}")
                for i in range(config.num_hidden_layers)
            ]
        )
        self.ln_post = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        first_layer = cast(MuseSparkVisionEncoderLayer, self.layers[0])
        first_attn = first_layer.attn
        self.use_data_parallel = first_attn.use_data_parallel
        self.tp_size = first_attn.tp_size
        self.attn_backend = first_attn.attn.attn_backend
        self.fp8_padded_hidden_size = get_fp8_padded_hidden_size(
            config.num_attention_heads, self.head_dim
        )

    @property
    def device(self) -> torch.device:
        return self.patch_embedder.patch_embedding.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embedder.patch_embedding.weight.dtype

    def geometry(self, grid_thw: Sequence[Sequence[int]]) -> VisionGeometry:
        return compute_vision_geometry(
            grid_thw,
            pos_emb_side=self.config.pos_emb_height,
            window_patches=self.config.pos_emb_height,
            merge_size=self.merge_size,
        )

    def _attention_metadata(self, seq_lens: Sequence[int], device: torch.device):
        cu = np.concatenate(
            [np.zeros(1, dtype=np.int32), np.cumsum(seq_lens, dtype=np.int32)]
        )
        sequence_lengths = MMEncoderAttention.maybe_compute_seq_lens(
            self.attn_backend, cu, device
        )
        max_seqlen = torch.tensor(
            MMEncoderAttention.compute_max_seqlen(self.attn_backend, cu),
            dtype=torch.int32,
        )
        cu_seqlens = MMEncoderAttention.maybe_recompute_cu_seqlens(
            self.attn_backend,
            cu,
            self.hidden_size,
            self.tp_size,
            device,
            fp8_padded_hidden_size=self.fp8_padded_hidden_size,
        )
        return cu_seqlens, max_seqlen, sequence_lengths

    def _rope(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # positions [N, 2] = (w, h); HF lays the angles out as [w, h, w, h].
        freqs = positions[..., None].float() * self.inv_freq.float()
        freqs = torch.cat([freqs[:, 0], freqs[:, 1]], dim=-1)
        freqs = torch.cat([freqs, freqs], dim=-1)
        # HF rounds cos/sin to the activation dtype before the fp32 rotation.
        return (
            freqs.cos().to(self.dtype).float(),
            freqs.sin().to(self.dtype).float(),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: Sequence[Sequence[int]] | torch.Tensor,
    ) -> torch.Tensor:
        """pixel_values [N, patch_temporal * 3 * patch**2]; grid_thw [B, 3]."""
        if isinstance(grid_thw, torch.Tensor):
            grid_thw = grid_thw.tolist()
        device = self.device
        geo = self.geometry(grid_thw)
        if pixel_values.shape[0] != geo.num_patches:
            raise ValueError(
                f"pixel_values has {pixel_values.shape[0]} patches, grid_thw "
                f"implies {geo.num_patches}"
            )
        window_index = geo.window_index.to(device)
        hidden = self.patch_embedder(
            pixel_values.to(device),
            geo.interp_indices.to(device),
            geo.interp_weights.to(device),
        )
        hidden = self.ln_pre(hidden)[window_index]
        cos, sin = self._rope(geo.rope_positions.to(device))
        full_meta = self._attention_metadata(geo.full_seqlens, device)
        window_meta = self._attention_metadata(geo.window_seqlens, device)
        for layer_type, layer in zip(self.layer_types, self.layers):
            meta = full_meta if layer_type == "full_attention" else window_meta
            hidden = layer(hidden, cos, sin, *meta)
        hidden = hidden[torch.argsort(window_index)]
        hidden = self.ln_post(hidden)
        factor = self.merge_size
        hidden = hidden[geo.shuffle_index.to(device)]
        return (
            hidden.view(-1, factor * factor, self.hidden_size)
            .permute(0, 2, 1)
            .reshape(-1, self.hidden_size * factor * factor)
        )


class MuseSparkVisionAdapter(nn.Module):
    """``act(fc2(act(fc1(x))))`` with exact GELU, no biases."""

    def __init__(
        self, config: MuseSparkConfig, use_data_parallel: bool, prefix: str = ""
    ) -> None:
        super().__init__()
        if config.projector_hidden_act != "gelu":
            raise ValueError(
                f"unsupported projector_hidden_act {config.projector_hidden_act}"
            )
        self.fc1 = ColumnParallelLinear(
            config.out_hidden_size,
            config.projector_hidden_size,
            bias=False,
            prefix=f"{prefix}.fc1",
            disable_tp=use_data_parallel,
        )
        self.fc2 = RowParallelLinear(
            config.projector_hidden_size,
            config.projector_hidden_size,
            bias=False,
            prefix=f"{prefix}.fc2",
            disable_tp=use_data_parallel,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x, _ = self.fc2(F.gelu(x))
        return F.gelu(x)


class MuseSparkPerception(nn.Module):
    """Vision tower, adapter, projection and audio tower of ``MuseSparkModel``.

    ``embed_multimodal`` returns one ``[n_tokens, text_hidden]`` tensor per
    item, already normalized exactly as HF scatters them into the text
    embeddings: vision through ``perception_emb_norm`` and audio through the
    token embedding's weightless ``embed_norm`` (both eps ``rms_norm_eps``).
    """

    # Checkpoint prefixes this module owns (after stripping a leading
    # "model.").
    weight_prefixes = (
        "vision_tower.",
        "vision_adapter.",
        "vision_projection.",
        "audio_tower.",
    )

    def __init__(
        self,
        config: MuseSparkConfig,
        prefix: str = "",
        enable_vision: bool = True,
        enable_audio: bool = True,
    ) -> None:
        super().__init__()
        from .audio import MuseSparkAudioTower

        self.config = config
        text_config = config.get_text_config()
        self.text_hidden_size = text_config.hidden_size
        self.norm_eps = text_config.rms_norm_eps
        self.merge_size = config.vision_config.merge_size
        self.vision_tower: MuseSparkVisionTower | None = None
        self.vision_adapter: MuseSparkVisionAdapter | None = None
        self.vision_projection: ColumnParallelLinear | None = None
        self.audio_tower = None
        if enable_vision:
            self.vision_tower = MuseSparkVisionTower(
                config.vision_config, prefix=f"{prefix}.vision_tower"
            )
            dp = self.vision_tower.use_data_parallel
            self.vision_adapter = MuseSparkVisionAdapter(
                config, dp, prefix=f"{prefix}.vision_adapter"
            )
            self.vision_projection = ColumnParallelLinear(
                config.projector_hidden_size,
                self.text_hidden_size,
                bias=False,
                gather_output=True,
                prefix=f"{prefix}.vision_projection",
                disable_tp=dp,
            )
        if enable_audio:
            self.audio_tower = MuseSparkAudioTower(config.audio_config)

    # Vision

    def encode_vision(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor | Sequence[Sequence[int]],
    ) -> list[torch.Tensor]:
        """Packed patches of several images or videos to per item embeddings."""
        if (
            self.vision_tower is None
            or self.vision_adapter is None
            or self.vision_projection is None
        ):
            raise ValueError("this MuseSparkPerception has no vision tower")
        if isinstance(grid_thw, torch.Tensor):
            grid_thw = grid_thw.reshape(-1, 3).tolist()
        features = self.vision_tower(pixel_values, grid_thw)
        features = self.vision_adapter(features)
        features, _ = self.vision_projection(features)
        features = weightless_rms_norm(features, self.norm_eps)
        sizes = [int(t * h * w) // self.merge_size**2 for t, h, w in grid_thw]
        return list(features.split(sizes))

    # Audio

    def encode_audio(self, clips: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if self.audio_tower is None:
            raise ValueError("this MuseSparkPerception has no audio tower")
        return [
            weightless_rms_norm(feat, self.norm_eps)
            for feat in self.audio_tower.encode_clips(list(clips))
        ]

    # vLLM entry point

    def embed_multimodal(self, **kwargs: object) -> list[torch.Tensor]:
        """Per item embeddings, modalities in kwargs order (vLLM convention).

        Consumes the fields produced by ``MuseSparkMultiModalProcessor``:
        ``pixel_values`` + ``image_grid_thw``, ``pixel_values_videos`` +
        ``video_grid_thw``, and ``input_values`` (one 1D clip per item).
        """
        embeddings: list[torch.Tensor] = []
        for key in kwargs:
            if key == "pixel_values":
                embeddings.extend(
                    self.encode_vision(
                        _cat_patches(kwargs["pixel_values"]),
                        _as_grid(kwargs["image_grid_thw"]),
                    )
                )
            elif key == "pixel_values_videos":
                embeddings.extend(
                    self.encode_vision(
                        _cat_patches(kwargs["pixel_values_videos"]),
                        _as_grid(kwargs["video_grid_thw"]),
                    )
                )
            elif key == "input_values":
                values = kwargs["input_values"]
                if isinstance(values, (list, tuple)):
                    clips = list(values)
                elif isinstance(values, torch.Tensor):
                    clips = list(torch.unbind(values, 0))
                else:
                    raise TypeError(f"unsupported input_values type {type(values)}")
                embeddings.extend(self.encode_audio(clips))
        return embeddings

    def load_weights(self, weights) -> set[str]:
        """Load ``[model.]{vision_tower,vision_adapter,vision_projection,
        audio_tower}.*`` tensors; other names are ignored.

        Returns the loaded parameter names relative to this module.
        """
        from vllm.model_executor.model_loader.weight_utils import (
            default_weight_loader,
        )

        stacked = (
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        )
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, tensor in weights:
            if name.startswith("model."):
                name = name[len("model.") :]
            if not name.startswith(self.weight_prefixes):
                continue
            shard_id = None
            if name.startswith("vision_tower."):
                for packed, part, sid in stacked:
                    if f".attn.{part}." in name:
                        name = name.replace(f".attn.{part}.", f".attn.{packed}.")
                        shard_id = sid
                        break
            param = params.get(name)
            if param is None:
                if name.split(".", 1)[0] in (
                    "vision_tower",
                    "vision_adapter",
                    "vision_projection",
                ) and (self.vision_tower is None):
                    continue
                if name.startswith("audio_tower.") and self.audio_tower is None:
                    continue
                raise KeyError(f"unexpected Muse Spark perception weight {name}")
            custom_loader = getattr(param, "weight_loader", None)
            if custom_loader is None:
                default_weight_loader(param, tensor)
            elif shard_id is None:
                custom_loader(param, tensor)
            else:
                custom_loader(param, tensor, shard_id)
            loaded.add(name)
        return loaded


def _cat_patches(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.reshape(-1, value.shape[-1])
    return torch.cat([v.reshape(-1, v.shape[-1]) for v in value])


def _as_grid(value) -> list[list[int]]:
    if isinstance(value, torch.Tensor):
        return value.reshape(-1, 3).tolist()
    return [
        list(map(int, g))
        for v in value
        for g in torch.as_tensor(v).reshape(-1, 3).tolist()
    ]


__all__ = [
    "MuseSparkPerception",
    "MuseSparkVisionAdapter",
    "MuseSparkVisionTower",
    "compute_vision_geometry",
    "weightless_rms_norm",
]
