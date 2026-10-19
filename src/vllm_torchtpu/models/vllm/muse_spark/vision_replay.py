# SPDX-License-Identifier: Apache-2.0
"""Static image geometry and isolated padding for encoder output-token buckets."""

import itertools
from collections.abc import Hashable
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F
from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphCaptureInputs,
    EncoderCudaGraphConfig,
    EncoderCudaGraphReplayBuffers,
    EncoderItemSpec,
)

from .vision import (
    _get_pixel_shuffle_perm,
    _get_sparse_attention_perm,
    _precompute_rope_freqs_cis,
)

if TYPE_CHECKING:
    from .config import MuseSparkTextConfig
    from .layers import MuseRMSNorm
    from .model import _MuseModelComponents


def _pad_boundaries(dst: torch.Tensor, src: torch.Tensor) -> None:
    # Repeat the real end boundary to isolate padding from image/window attention.
    src = src.to(device=dst.device)
    dst.copy_(src[-1].expand_as(dst))
    dst[: src.numel()].copy_(src)


def prepare_image_buffers(vision, images: list[torch.Tensor]):
    """Preserve NaViT.forward's patch order, XY grid and accelerator RoPE."""
    device = vision.patch_embedder.position_embedding_table.weight.device
    dtype = vision.patch_embedder.position_embedding_table.weight.dtype
    patches, positions, frequencies, permutations, downsample = [], [], [], [], []
    image_lengths, window_lengths = [], []
    offset = 0
    for image in images:
        if image.ndim != 3:
            raise ValueError("Muse image replay expects [C, H, W] images")
        image = image.to(device=device, dtype=vision.input_dtype).to(dtype)
        channels, height, width = image.shape
        ph, pw = vision.patch_height, vision.patch_width
        ds = vision.downsample_factor
        if height % (ph * ds) or width % (pw * ds):
            raise ValueError("Image dimensions must align with NaViT downsampling")
        gh, gw = height // ph, width // pw
        size = gh * gw
        image = image[None].expand(vision.patch_temporal, -1, -1, -1)
        patches.append(
            image.reshape(vision.patch_temporal, channels, gh, ph, gw, pw)
            .permute(2, 4, 0, 1, 3, 5)
            .reshape(size, vision.patch_temporal * channels * ph * pw)
        )
        pos_h = torch.linspace(
            -1 + 1 / gh, 1 - 1 / gh, gh, device=device, dtype=vision.input_dtype
        )
        pos_w = torch.linspace(
            -1 + 1 / gw, 1 - 1 / gw, gw, device=device, dtype=vision.input_dtype
        )
        positions.append(
            torch.stack(torch.meshgrid(pos_h, pos_w, indexing="xy"), dim=-1)
            .reshape(-1, 2)
            .to(dtype)
        )
        if gh > vision.max_rope_grid_h or gw > vision.max_rope_grid_w:
            freq = _precompute_rope_freqs_cis(
                vision.rope_dim,
                gh,
                gw,
                vision.rope_dtype,
                theta=vision.rope_theta,
                device=device,
            )
        else:
            freq = vision.precomputed_freqs_cis[:gh, :gw]
        frequencies.append(freq.reshape(size, vision.rope_dim // 2, 2))

        # Build shape-only geometry on CPU to avoid TPU synchronization.
        if vision.sparse_attention_factor > 1:
            perm, lengths, _ = _get_sparse_attention_perm(
                offset,
                gh,
                gw,
                vision.pos_emb_grid_height,
                vision.pos_emb_grid_width,
                torch.device("cpu"),
            )
            window_lengths.extend(lengths.tolist())
        else:
            perm = torch.arange(offset, offset + size)
            window_lengths.append(size)
        permutations.append(perm)
        ds_perm, _ = _get_pixel_shuffle_perm(offset, gh, gw, ds, torch.device("cpu"))
        downsample.append(ds_perm)
        image_lengths.append(size)
        offset += size

    perm = torch.cat(permutations)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(offset)
    # Restore raster order and pixel-shuffle with one gather.
    output_perm = inverse[torch.cat(downsample)]

    def boundaries(lengths):
        return torch.tensor(
            [0, *itertools.accumulate(lengths)], dtype=torch.int32, device=device
        )

    return {
        "patches": torch.cat(patches),
        "positions": torch.cat(positions),
        "frequencies": torch.cat(frequencies),
        "permutation": perm.to(device),
        "output_permutation": output_perm.to(device),
        "cu_seqlens": boundaries(image_lengths),
        "sparse_cu_seqlens": boundaries(window_lengths),
    }


def image_encoder_forward(vision, values: dict[str, torch.Tensor]):
    """Tensor-only NaViT path; every shape is fixed by the selected bucket."""
    x, _ = vision.patch_embedder.patch_embedding(values["patches"])
    positional = (
        vision.patch_embedder.position_embedding_table.weight.reshape(
            vision.pos_emb_grid_height, vision.pos_emb_grid_width, -1
        )
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    x = (
        x
        + F.grid_sample(
            positional,
            values["positions"][None, None],
            mode=vision.grid_sample_mode,
            align_corners=False,
        )[0, :, 0, :].T
    )
    x = vision.ln_pre(x)[values["permutation"]]
    x = vision.layers(
        x,
        freq_cis=values["frequencies"][values["permutation"]],
        cu_seqlens=values["cu_seqlens"],
        sparse_cu_seqlens=values["sparse_cu_seqlens"],
    )
    x = vision.ln_post(x[values["output_permutation"]])
    factor = vision.downsample_factor**2
    x = x.reshape(x.shape[0] // factor, factor, x.shape[1]).transpose(1, 2)
    return x.reshape(x.shape[0], -1)


class MuseVisionReplayMixin(SupportsEncoderCudaGraph):
    """The existing TPU manager owns bucket compilation, packing and slicing."""

    model: "_MuseModelComponents"
    config: "MuseSparkTextConfig"
    _use_vision_dp: bool
    normalize_tok_embeddings: bool
    perception_emb_norm: "MuseRMSNorm"
    _vision_output_dtype: torch.dtype

    if TYPE_CHECKING:

        def _get_patch_count(self, item: torch.Tensor) -> int: ...

        def _process_image_input(
            self, images: torch.Tensor | list[torch.Tensor]
        ) -> list[torch.Tensor]: ...

        def _encode_vision(
            self, vision_items: list[torch.Tensor]
        ) -> list[torch.Tensor]: ...

    def get_encoder_cudagraph_config(self):
        vision = self.model.vision_tower
        if vision.use_cls_token or vision.downsample_strategy != "ps":
            raise ValueError("Muse image replay requires pixel shuffle without CLS")
        if self._use_vision_dp:
            raise ValueError("Muse image replay currently supports tensor parallelism")
        return EncoderCudaGraphConfig(
            modalities=["image"],
            buffer_keys=[
                "patches",
                "positions",
                "frequencies",
                "permutation",
                "output_permutation",
                "cu_seqlens",
                "sparse_cu_seqlens",
            ],
            out_hidden_size=self.config.hidden_size,
            padding_logics={
                "cu_seqlens": _pad_boundaries,
                "sparse_cu_seqlens": _pad_boundaries,
            },
        )

    def get_encoder_cudagraph_budget_range(self, vllm_config):
        model = vllm_config.model_config
        processor = (
            get_hf_file_to_dict(
                "processor_config.json", model.model, revision=model.revision
            )
            or {}
        )
        maximum = processor.get("image_processor", {}).get("max_image_tokens", 4096)
        return min(64, maximum), maximum

    def get_encoder_cudagraph_item_specs(self, mm_kwargs: dict[str, Any]):
        factor = self.model.vision_tower.downsample_factor**2
        return [
            EncoderItemSpec(
                input_size=self._get_patch_count(image) * factor,
                output_tokens=self._get_patch_count(image),
            )
            for image in self._process_image_input(mm_kwargs["images"])
        ]

    def select_encoder_cudagraph_items(self, mm_kwargs, indices):
        images = self._process_image_input(mm_kwargs["images"])
        return {"images": [images[index] for index in indices]}

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget,
        max_batch_size,
        max_frames_per_batch,
        device,
        dtype,
        path="default",
        axis_keys: tuple[Hashable, ...] | None = None,
    ):
        vision = self.model.vision_tower
        dtype = vision.patch_embedder.position_embedding_table.weight.dtype
        ds = vision.downsample_factor
        count = token_budget * ds**2
        # At most gh*gw/(min(window_h,window_w)*ds) + 1 windows per image.
        windows = (
            count // (min(vision.pos_emb_grid_height, vision.pos_emb_grid_width) * ds)
            + max_batch_size
        )
        patch_dim = vision.patch_embedder.patch_embedding.input_size
        values = {
            "patches": torch.zeros(count, patch_dim, device=device, dtype=dtype),
            "positions": torch.zeros(count, 2, device=device, dtype=dtype),
            "frequencies": torch.zeros(
                count, vision.rope_dim // 2, 2, device=device, dtype=vision.rope_dtype
            ),
            "permutation": torch.arange(count, device=device),
            "output_permutation": torch.arange(count, device=device),
            "cu_seqlens": torch.full(
                (max_batch_size + 1,), count, device=device, dtype=torch.int32
            ),
            "sparse_cu_seqlens": torch.full(
                (windows + 1,), count, device=device, dtype=torch.int32
            ),
        }
        values["frequencies"][..., 0] = 1
        values["cu_seqlens"][0] = 0
        values["sparse_cu_seqlens"][0] = 0
        return EncoderCudaGraphCaptureInputs(values=values)

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs,
        max_batch_size,
        max_frames_per_batch,
        path="default",
    ):
        return EncoderCudaGraphReplayBuffers(
            values=cast(
                dict[str, torch.Tensor | None],
                prepare_image_buffers(
                    self.model.vision_tower,
                    self._process_image_input(mm_kwargs["images"]),
                ),
            )
        )

    def encoder_cudagraph_forward(self, inputs, path="default"):
        x = image_encoder_forward(self.model.vision_tower, inputs)
        x = self.model.vision_adapter(x)
        x, _ = self.model.vision_projection(x)
        if self.normalize_tok_embeddings:
            x = self.perception_emb_norm(x)
        return x.to(self._vision_output_dtype)

    def encoder_eager_forward(self, mm_kwargs, path="default"):
        return torch.cat(
            self._encode_vision(self._process_image_input(mm_kwargs["images"]))
        )
