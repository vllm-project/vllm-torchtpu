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
"""Compiled Pixtral vision using standard vLLM weights and TPU flash attention."""

from collections.abc import Hashable, Iterable, Mapping
from typing import cast

import torch
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph
from vllm.model_executor.models.pixtral import (
    PATCH_MERGE,
    PixtralDummyInputsBuilder,
    PixtralForConditionalGeneration,
    PixtralMultiModalProcessor,
    PixtralProcessingInfo,
    TransformerBlock,
)
from vllm.model_executor.models.vision import is_vit_use_data_parallel
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, NestedTensors
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphCaptureInputs,
    EncoderCudaGraphConfig,
    EncoderCudaGraphReplayBuffers,
    EncoderItemSpec,
)


class TPUPixtralMultiModalProcessor(PixtralMultiModalProcessor):
    def _get_mm_fields_config(
        self,
        hf_inputs: Mapping[str, NestedTensors],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # Pack variable image shapes on the host, then transfer fixed-size
        # patch buffers through MMEncoderManager, avoiding per-resolution JITs.
        return {"images": MultiModalFieldConfig.batched("image", keep_on_cpu=True)}


def _rotate_2d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return (
        torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1)
        .flatten(-2)
        .to(x.dtype)
    )


@MULTIMODAL_REGISTRY.register_processor(
    TPUPixtralMultiModalProcessor,
    info=PixtralProcessingInfo,
    dummy_inputs=PixtralDummyInputsBuilder,
)
class MistralLarge4ForConditionalGeneration(
    PixtralForConditionalGeneration, SupportsEncoderCudaGraph
):
    """Native Pixtral wrapper with a static-buffer TPU encoder entry point.

    The inherited constructor and loader retain the BF16 vision tower,
    projector, and checkpoint names. Only vision execution and host packing
    differ; the text child is the registered TPU ML4 model.
    """

    supports_encoder_cudagraph = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        if not vllm_config.compilation_config.cudagraph_mm_encoder:
            raise ValueError(
                "ML4 images require compilation_config.cudagraph_mm_encoder=true."
            )
        if is_vit_use_data_parallel():
            raise NotImplementedError(
                "ML4 vision currently requires mm_encoder_tp_mode=weights."
            )
        vision_config = vllm_config.model_config.hf_config.vision_config
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        if vision_config.num_attention_heads % tp_size:
            raise ValueError("ML4 vision attention heads must be divisible by TP size.")
        if vision_config.mm_projector_id != PATCH_MERGE:
            raise NotImplementedError("ML4 vision requires the patch_merge projector.")
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        return EncoderCudaGraphConfig(
            modalities=["image"],
            buffer_keys=["patches", "cos", "sin", "cu_seqlens", "merge_indices"],
            out_hidden_size=self.config.text_config.hidden_size,
        )

    def get_input_modality(self, mm_kwargs: dict[str, object]) -> str:
        return "image"

    def get_max_frames_per_video(self) -> int:
        return 1

    def get_encoder_cudagraph_budget_range(
        self, vllm_config: VllmConfig
    ) -> tuple[int, int]:
        args = self.vision_args
        maximum = (args.image_size // (args.patch_size * args.spatial_merge_size)) ** 2
        maximum = min(maximum, vllm_config.model_config.max_model_len)
        return min(
            maximum, vllm_config.scheduler_config.max_num_batched_tokens
        ), maximum

    def _images(self, mm_kwargs: dict[str, object]) -> list[torch.Tensor]:
        return list(cast(list[torch.Tensor] | torch.Tensor, mm_kwargs["images"]))

    def get_encoder_cudagraph_item_specs(
        self, mm_kwargs: dict[str, object]
    ) -> list[EncoderItemSpec]:
        patch, merge = self.vision_args.patch_size, self.vision_args.spatial_merge_size
        specs = []
        for image in self._images(mm_kwargs):
            channels, height, width = image.shape
            if (
                channels != self.vision_args.num_channels
                or height % (patch * merge)
                or width % (patch * merge)
            ):
                raise ValueError(
                    "Pixtral images must have complete spatial-merge patch grids."
                )
            patches = (height // patch) * (width // patch)
            specs.append(EncoderItemSpec(patches, patches // merge**2))
        return specs

    def select_encoder_cudagraph_items(
        self, mm_kwargs: dict[str, object], indices: list[int]
    ) -> dict[str, object]:
        images = self._images(mm_kwargs)
        return {"images": [images[i] for i in indices]}

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
        axis_keys: tuple[Hashable, ...] | None = None,
    ) -> EncoderCudaGraphCaptureInputs:
        if path != "default" or axis_keys:
            raise ValueError(
                "ML4 vision has a single encoder path and no capture axes."
            )
        args = self.vision_args
        patches = token_budget * args.spatial_merge_size**2
        half_head = args.hidden_size // args.num_attention_heads // 2
        return EncoderCudaGraphCaptureInputs(
            values={
                "patches": torch.zeros(
                    patches,
                    args.num_channels * args.patch_size**2,
                    device=device,
                    dtype=dtype,
                ),
                "cos": torch.ones(
                    patches, half_head, device=device, dtype=torch.float32
                ),
                "sin": torch.zeros(
                    patches, half_head, device=device, dtype=torch.float32
                ),
                "cu_seqlens": torch.tensor(
                    [0] + [patches] * (max_batch_size + 1),
                    device=device,
                    dtype=torch.int32,
                ),
                "merge_indices": torch.zeros(
                    token_budget,
                    args.spatial_merge_size**2,
                    device=device,
                    dtype=torch.int64,
                ),
            }
        )

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, object],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ) -> EncoderCudaGraphReplayBuffers:
        images = self._images(mm_kwargs)
        specs = self.get_encoder_cudagraph_item_specs(mm_kwargs)
        if not images or len(images) > max_batch_size:
            raise ValueError("Pixtral replay needs 1..max_batch_size images.")
        args = self.vision_args
        patch, merge = args.patch_size, args.spatial_merge_size
        head_dim = args.hidden_size // args.num_attention_heads
        frequencies = 1.0 / (
            args.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        patches, phases, merge_indices = [], [], []
        boundaries = [0]
        for image, spec in zip(images, specs):
            if image.device.type != "cpu":
                raise ValueError(
                    "Pixtral packing expects images kept on CPU by its processor."
                )
            channels, height, width = image.shape
            rows, cols = height // patch, width // patch
            patches.append(
                image.reshape(channels, rows, patch, cols, patch)
                .permute(1, 3, 0, 2, 4)
                .reshape(spec.input_size, -1)
            )
            y, x = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
            phases.append(
                torch.cat(
                    (
                        y.flatten()[:, None] * frequencies[::2],
                        x.flatten()[:, None] * frequencies[1::2],
                    ),
                    dim=-1,
                )
            )
            indices = torch.arange(spec.input_size).reshape(
                rows // merge, merge, cols // merge, merge
            )
            merge_indices.append(
                indices.permute(0, 2, 1, 3).reshape(-1, merge**2) + boundaries[-1]
            )
            boundaries.append(boundaries[-1] + spec.input_size)
        cu = boundaries + [boundaries[-1]] * (max_batch_size + 2 - len(boundaries))
        phase = torch.cat(phases)
        return EncoderCudaGraphReplayBuffers(
            values={
                "patches": torch.cat(patches).to(
                    self.vision_encoder.patch_conv.weight.dtype
                ),
                "cos": phase.cos(),
                "sin": phase.sin(),
                "cu_seqlens": torch.tensor(cu, dtype=torch.int32),
                "merge_indices": torch.cat(merge_indices),
            }
        )

    def encoder_cudagraph_forward(
        self, inputs: dict[str, torch.Tensor], path: str = "default"
    ) -> torch.Tensor:
        weight = self.vision_encoder.patch_conv.weight
        hidden = F.linear(inputs["patches"], weight.flatten(1)).unsqueeze(0)
        hidden = self.vision_encoder.ln_pre(hidden)
        cos, sin = inputs["cos"], inputs["sin"]
        boundaries = inputs["cu_seqlens"]
        # The last segment owns all bucket padding, isolated from every image.
        cu = torch.cat((boundaries[:-1], boundaries.new_full((1,), hidden.shape[1])))
        for block in cast(
            Iterable[TransformerBlock], self.vision_encoder.transformer.layers
        ):
            attention = block.attention
            normalized = block.attention_norm(hidden)
            q, k, v = attention.qkv_proj(normalized)[0].chunk(3, dim=-1)
            shape = (*q.shape[:-1], attention.n_heads, attention.head_dim)
            q, k, v = q.reshape(shape), k.reshape(shape), v.reshape(shape)
            q, k = _rotate_2d(q, cos, sin), _rotate_2d(k, cos, sin)
            out = attention.attn(q, k, v, cu_seqlens=cu)
            hidden = hidden + attention.o_proj(out.reshape(*hidden.shape[:-1], -1))[0]
            hidden = hidden + block.feed_forward(block.ffn_norm(hidden))
        hidden = hidden.squeeze(0)
        if self.pre_mm_projector_norm is not None:
            hidden = self.pre_mm_projector_norm(hidden)
        gathered = hidden[inputs["merge_indices"]].transpose(1, 2).flatten(1)
        assert self.patch_merger is not None
        merged = self.patch_merger.merging_layer(gathered)
        return self.vision_language_adapter(merged)

    def embed_multimodal(self, **kwargs: object):
        if kwargs.get("images") is None:
            return []
        raise RuntimeError("ML4 images must run through the compiled encoder manager.")

    def encoder_eager_forward(
        self, mm_kwargs: dict[str, object], path: str = "default"
    ):
        raise RuntimeError(
            "ML4 vision requires a compiled encoder bucket; eager fallback is disabled."
        )
