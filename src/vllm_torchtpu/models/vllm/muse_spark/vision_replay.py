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
"""Bucketed encoder replay of the Muse Spark image encoder.

The NaViT tower builds the encoder inputs (``NaViT.prepare_buffers``) and runs
the tensor-only encoder (``NaViT.encode_buffers``); this mixin pads them to a
token bucket and isolates the padding from image and window attention.
"""

from collections.abc import Hashable
from typing import TYPE_CHECKING, Any, cast

import torch
from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphCaptureInputs,
    EncoderCudaGraphConfig,
    EncoderCudaGraphReplayBuffers,
    EncoderItemSpec,
)

from .vision import VISION_BUFFER_KEYS

if TYPE_CHECKING:
    from vllm.transformers_utils.configs.muse_spark import MuseSparkTextConfig

    from .model import _MuseModelComponents


def _pad_boundaries(dst: torch.Tensor, src: torch.Tensor) -> None:
    # Repeat the real end boundary to isolate padding from image/window attention.
    src = src.to(device=dst.device)
    dst.copy_(src[-1].expand_as(dst))
    dst[: src.numel()].copy_(src)


class MuseVisionReplayMixin(SupportsEncoderCudaGraph):
    """Encoder replay hooks for the TPU runner.

    ``MMEncoderManager`` owns bucket compilation, packing and slicing.
    """

    model: "_MuseModelComponents"
    config: "MuseSparkTextConfig"
    _use_vision_dp: bool
    _max_image_tokens: int

    if TYPE_CHECKING:

        def _get_patch_count(self, item: torch.Tensor) -> int: ...

        def _image_items(self, mm_kwargs: dict[str, Any]) -> list[torch.Tensor]: ...

        def _encode_vision(
            self, vision_items: list[torch.Tensor]
        ) -> list[torch.Tensor]: ...

        def _project_vision(self, h_vision: torch.Tensor) -> torch.Tensor: ...

    def get_encoder_cudagraph_config(self):
        if self._use_vision_dp:
            raise ValueError("Muse image replay currently supports tensor parallelism")
        return EncoderCudaGraphConfig(
            modalities=["image"],
            buffer_keys=list(VISION_BUFFER_KEYS),
            out_hidden_size=self.config.hidden_size,
            padding_logics={
                "cu_seqlens": _pad_boundaries,
                "sparse_cu_seqlens": _pad_boundaries,
            },
        )

    def get_encoder_cudagraph_budget_range(self, vllm_config):
        return min(64, self._max_image_tokens), self._max_image_tokens

    def get_encoder_cudagraph_item_specs(self, mm_kwargs: dict[str, Any]):
        factor = self.model.vision_tower.downsample_factor**2
        return [
            EncoderItemSpec(
                input_size=self._get_patch_count(image) * factor,
                output_tokens=self._get_patch_count(image),
            )
            for image in self._image_items(mm_kwargs)
        ]

    def select_encoder_cudagraph_items(self, mm_kwargs, indices):
        images = self._image_items(mm_kwargs)
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
        dtype = vision.positional_embedding_vlm.dtype
        ds = vision.downsample_factor
        count = token_budget * ds**2
        # At most gh*gw/(min(window_h,window_w)*ds) + 1 windows per image.
        windows = (
            count // (min(vision.pos_emb_grid_height, vision.pos_emb_grid_width) * ds)
            + max_batch_size
        )
        patch_dim = vision.conv1_linear.input_size
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
        buffers = self.model.vision_tower.prepare_buffers(self._image_items(mm_kwargs))
        return EncoderCudaGraphReplayBuffers(
            values=cast(
                dict[str, torch.Tensor | None],
                {key: buffers[key] for key in VISION_BUFFER_KEYS},
            )
        )

    def encoder_cudagraph_forward(self, inputs, path="default"):
        return self._project_vision(self.model.vision_tower.encode_buffers(inputs))

    def encoder_eager_forward(self, mm_kwargs, path="default"):
        return torch.cat(self._encode_vision(self._image_items(mm_kwargs)))
