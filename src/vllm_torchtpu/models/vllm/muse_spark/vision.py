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
"""Muse Spark NaViT vision tower for TPU.

Upstream source: ``NaViT`` in ``vllm/models/muse_spark/common/vision.py``. The
modules (encoder, adapter, patch and position embeddings), the RoPE table and
the geometry helpers are inherited or imported unchanged.

TPU changes:

- The RoPE table covers the processor's maximum image token count, passed in
  by the caller (which also sizes the replay buckets with it), instead of a
  fixed 4096 tokens.
- ``forward`` is split into ``prepare_buffers`` (patch order, positions, RoPE
  and attention boundaries, with the geometry built on the host so nothing
  synchronizes with the device) and ``encode_buffers`` (a tensor-only encoder
  and adapter whose shapes depend only on the patch count). Eager encoding and
  bucketed encoder replay (``vision_replay.py``) share both. One gather undoes
  the window permutation and applies the pixel shuffle.
"""

import itertools
import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.models.muse_spark.common.vision import NaViT as UpstreamNaViT
from vllm.models.muse_spark.common.vision import (
    _get_pixel_shuffle_perm,
    _get_sparse_attention_perm,
    _iterate_visions,
    _precompute_rope_freqs_cis,
)
from vllm.transformers_utils.configs.muse_spark import MuseSparkVisionConfig

# Keys of the tensors ``prepare_buffers`` returns that encoder replay pads.
VISION_BUFFER_KEYS = (
    "patches",
    "positions",
    "frequencies",
    "permutation",
    "output_permutation",
    "cu_seqlens",
    "sparse_cu_seqlens",
)


def max_rope_grid(max_num_image_tokens: int, config: MuseSparkVisionConfig) -> int:
    """Largest patch-grid side an image within the token budget can reach."""
    max_pre_ds_tokens = max_num_image_tokens * config.merge_size**2
    return max(
        int(math.sqrt(max_pre_ds_tokens)) + 1,
        config.pos_emb_height,
        config.pos_emb_width,
    )


class NaViT(UpstreamNaViT):
    """Upstream NaViT with a configured RoPE extent and host-built geometry."""

    def __init__(
        self,
        config: MuseSparkVisionConfig,
        projector_hidden_size: int,
        max_num_image_tokens: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config,
            projector_hidden_size=projector_hidden_size,
            quant_config=quant_config,
            prefix=prefix,
        )
        grid = max_rope_grid(max_num_image_tokens, config)
        if grid != self.max_rope_grid_h:
            self.max_rope_grid_h = self.max_rope_grid_w = grid
            self.register_buffer(
                "precomputed_freqs_cis",
                _precompute_rope_freqs_cis(
                    dim=self.rope_dim,
                    max_height=grid,
                    max_width=grid,
                    dtype=self.rope_dtype,
                    theta=self.rope_theta,
                    device=(
                        torch.accelerator.current_device_index()
                        if torch.accelerator.is_available()
                        else None
                    ),
                ),
                persistent=False,
            )

    def prepare_buffers(
        self, visions: Sequence[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Encoder inputs for [C, H, W] images and [T, C, H, W] clips."""
        weight = self.positional_embedding_vlm
        device, dtype = weight.device, weight.dtype
        host = torch.device("cpu")
        ph, pw, ds = self.patch_height, self.patch_width, self.downsample_factor
        patches, positions, frequencies = [], [], []
        permutations, downsample = [], []
        image_lengths, window_lengths = [], []
        offset = 0
        for item in _iterate_visions(visions, self.patch_temporal):
            if item.ndim == 3:
                item = item[None]
            if item.shape[0] == 1:
                item = item.expand(self.patch_temporal, -1, -1, -1)
            if item.shape[0] != self.patch_temporal:
                raise ValueError(f"Unexpected NaViT clip shape {tuple(item.shape)}")
            _, channels, height, width = item.shape
            if height % (ph * ds) or width % (pw * ds):
                raise ValueError(
                    f"Image size {height}x{width} must align with patch size "
                    f"{ph}x{pw} and downsampling {ds}"
                )
            gh, gw = height // ph, width // pw
            size = gh * gw
            item = item.to(device=device, dtype=dtype)
            patches.append(
                item.reshape(self.patch_temporal, channels, gh, ph, gw, pw)
                .permute(2, 4, 0, 1, 3, 5)
                .reshape(size, self.patch_temporal * channels * ph * pw)
            )
            pos_h = torch.linspace(
                -1 + 1 / gh, 1 - 1 / gh, gh, device=device, dtype=dtype
            )
            pos_w = torch.linspace(
                -1 + 1 / gw, 1 - 1 / gw, gw, device=device, dtype=dtype
            )
            positions.append(
                torch.stack(
                    torch.meshgrid(pos_h, pos_w, indexing="xy"), dim=-1
                ).reshape(-1, 2)
            )
            if gh > self.max_rope_grid_h or gw > self.max_rope_grid_w:
                freq = _precompute_rope_freqs_cis(
                    self.rope_dim,
                    gh,
                    gw,
                    self.rope_dtype,
                    theta=self.rope_theta,
                    device=device,
                )
            else:
                freq = self.precomputed_freqs_cis[:gh, :gw]
            frequencies.append(freq.reshape(size, self.rope_dim // 2, 2))

            if self.sparse_attention_factor > 1:
                perm, lengths, _ = _get_sparse_attention_perm(
                    offset,
                    gh,
                    gw,
                    self.pos_emb_grid_height,
                    self.pos_emb_grid_width,
                    host,
                )
                window_lengths.extend(lengths.tolist())
            else:
                perm = torch.arange(offset, offset + size)
                window_lengths.append(size)
            permutations.append(perm)
            ds_perm, _ = _get_pixel_shuffle_perm(offset, gh, gw, ds, host)
            downsample.append(ds_perm)
            image_lengths.append(size)
            offset += size

        perm = torch.cat(permutations)
        inverse = torch.empty_like(perm)
        inverse[perm] = torch.arange(offset)

        def boundaries(lengths: list[int]) -> torch.Tensor:
            return torch.tensor(
                [0, *itertools.accumulate(lengths)], dtype=torch.int32, device=device
            )

        return {
            "patches": torch.cat(patches),
            "positions": torch.cat(positions),
            "frequencies": torch.cat(frequencies),
            "permutation": perm.to(device),
            # Restore raster order and pixel-shuffle with one gather.
            "output_permutation": inverse[torch.cat(downsample)].to(device),
            "cu_seqlens": boundaries(image_lengths),
            "sparse_cu_seqlens": boundaries(window_lengths),
            "max_seqlen": torch.tensor(max(image_lengths), dtype=torch.int32),
            "sparse_max_seqlen": torch.tensor(max(window_lengths), dtype=torch.int32),
        }

    def encode_buffers(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encoder and adapter on tensors from ``prepare_buffers``.

        ``max_seqlen`` and ``sparse_max_seqlen`` are optional: the TPU flash
        attention masks by ``cu_seqlens`` alone, so replay leaves them out.
        """
        x, _ = self.conv1_linear(values["patches"])
        positional = (
            self.positional_embedding_vlm.reshape(
                self.pos_emb_grid_height, self.pos_emb_grid_width, -1
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        x = (
            x
            + F.grid_sample(
                positional,
                values["positions"][None, None],
                mode="bilinear",
                align_corners=False,
            )[0, :, 0, :].T
        )
        permutation = values["permutation"]
        x = self.encoder(
            self.ln_pre(x)[permutation],
            freq_cis=values["frequencies"][permutation],
            cu_seqlens=values["cu_seqlens"],
            max_seqlen=values.get("max_seqlen"),
            sparse_cu_seqlens=values["sparse_cu_seqlens"],
            sparse_max_seqlen=values.get("sparse_max_seqlen"),
        )
        x = self.ln_post(x[values["output_permutation"]])
        factor = self.downsample_factor**2
        x = x.reshape(x.shape[0] // factor, factor, x.shape[1]).transpose(1, 2)
        return self.adapter(x.reshape(x.shape[0], -1))

    def forward(self, visions: Sequence[torch.Tensor]) -> torch.Tensor:
        """Encode [C, H, W] images / [T, C, H, W] clips into concatenated tokens."""
        if len(visions) == 0:
            weight = self.positional_embedding_vlm
            return torch.empty(
                0, self.adapter_output_dim, device=weight.device, dtype=weight.dtype
            )
        return self.encode_buffers(self.prepare_buffers(visions))
