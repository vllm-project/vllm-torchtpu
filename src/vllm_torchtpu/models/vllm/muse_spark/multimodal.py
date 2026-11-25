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
"""Muse Spark text, vision and audio model for TPU.

Upstream source: ``vllm/models/muse_spark/common/omni.py``
(``MuseSpark1xOmniMixin``) and ``vllm/models/muse_spark/nvidia/multimodal.py``.
Placeholder strings, input unpacking, patch counts and ``embed_multimodal`` are
inherited from ``MuseSpark1xOmniMixin``; the processor registration, the NaViT
and audio modules and the checkpoint renames are imported unchanged.

TPU changes:

- The towers sit under ``self.model`` (``vision_tower``, ``vision_projection``,
  ``audio_tower``) next to the TPU text stack instead of upstream's
  ``vision_model`` and ``audio_encoder``, so ``__init__``, ``_encode_vision``,
  ``_encode_audio`` and ``get_mm_mapping`` are overridden, and
  ``hf_to_vllm_mapper`` applies ``OMNI_WEIGHTS_MAPPER``'s vision and audio
  renames with those roots. The vision encoder layers and the projection use
  the TPU unquantized config.
- Images also run as bucketed encoder replay (``vision_replay.py``); eager
  encoding and replay share the NaViT buffers (``vision.py``) and
  ``_project_vision``, and both reject image items with several frames.
- The maximum image token count is read once from the processor config and
  sizes both the NaViT RoPE table and the replay buckets.
- Audio runs through ``audio.py``'s ``AutoEncoder``, which pads waveforms to
  static token buckets and batches them; with ``compile_mm_encoder`` it compiles
  with the TPU backend and static shapes, and ``_encode_audio`` encodes all
  waveforms in one call.
- DP-sharded NaViT materializes and synchronizes its output around the TP
  all-gather.
- ``embed_input_ids`` merges with the shared static-shape
  ``merge_multimodal_embeddings`` (as Kimi-K3 does) and keeps the FP32 residual
  dtype.
- DP attention with EP supports image inputs only, with
  ``cudagraph_mm_encoder``.
"""

import itertools
import math
from typing import Any

import torch
from vllm.compilation.backends import set_model_tag
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix
from vllm.model_executor.models.vision import (
    get_load_balance_assignment,
    is_vit_use_data_parallel,
)
from vllm.models.muse_spark.common.omni import (
    OMNI_WEIGHTS_MAPPER,
    MuseSpark1xOmniMixin,
    register_omni_processor,
)
from vllm.transformers_utils.processor import get_processor_config

from vllm_torchtpu.layers.adapter.quantization.unquantized import VllmUnquantizedConfig
from vllm_torchtpu.utils import (
    dp_engines_in_lockstep,
    fresh_buffer,
    synchronize_tensors,
)

from ..multimodal_merge import merge_multimodal_embeddings
from .audio import AutoEncoder
from .layers import MuseRMSNorm
from .model import MuseSparkForCausalLM, _MuseModelComponents
from .vision import NaViT
from .vision_replay import MuseVisionReplayMixin


@register_omni_processor
# Upstream's mixin and SupportsMultiModal annotate get_language_model's return
# type differently.
# pyrefly: ignore[inconsistent-inheritance]
class MuseSparkForConditionalGeneration(
    MuseSpark1xOmniMixin,
    MuseSparkForCausalLM,
    MuseVisionReplayMixin,
    SupportsMultiModal,
):
    """Muse Spark: text stack plus NaViT vision and waveform audio."""

    # Narrows the mixin's nn.Module to this model's component holder.
    # pyrefly: ignore[bad-override-mutable-attribute]
    model: _MuseModelComponents
    # Every tower tensor must load: vLLM checks for never-loaded parameters only
    # in unquantized checkpoints.
    ignored_weight_prefixes = ()
    # Upstream's vision and audio renames, with the towers kept under model.
    hf_to_vllm_mapper = MuseSparkForCausalLM.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_regex={
            pattern: name
            for pattern, name in OMNI_WEIGHTS_MAPPER.orig_to_new_regex.items()
            if "audio" in pattern.pattern or "vision" in pattern.pattern
        },
        orig_to_new_stacked={f".attn.{p}_proj.": (".attn.qkv_proj.", p) for p in "qkv"},
        orig_to_new_prefix={
            "vision_model.": "model.vision_tower.",
            "vision_projection.": "model.vision_projection.",
            "audio_encoder.": "model.audio_tower.",
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # The mixin's __init__ builds upstream's tower layout; build ours instead.
        MuseSparkForCausalLM.__init__(self, vllm_config=vllm_config, prefix=prefix)
        parallel = vllm_config.parallel_config
        if dp_engines_in_lockstep(parallel):
            mm_config = vllm_config.model_config.get_multimodal_config()
            if any(mm_config.get_limit_per_prompt(m) for m in ("video", "audio")):
                raise ValueError(
                    "Muse Spark DP attention with EP supports image inputs only; "
                    'set --limit-mm-per-prompt \'{"video": 0, "audio": 0}\''
                )
            if (
                mm_config.get_limit_per_prompt("image")
                and not vllm_config.compilation_config.cudagraph_mm_encoder
            ):
                raise ValueError(
                    "Muse Spark image inputs with DP attention and EP require "
                    "cudagraph_mm_encoder=True; set image to 0 for text-only serving"
                )

        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        vision_config = hf_config.vision_config
        audio_config = hf_config.audio_config
        adapter_output_dim = hf_config.projector_hidden_size
        llm_hidden_size = hf_config.text_config.hidden_size
        processor = get_processor_config(
            model_config.model, revision=model_config.revision
        )
        try:
            self._max_image_tokens = int(
                processor["image_processor"]["max_image_tokens"]
            )
        except KeyError as error:
            raise ValueError(
                "Muse Spark needs image_processor.max_image_tokens in the "
                f"processor config of {model_config.model}"
            ) from error
        dense = VllmUnquantizedConfig()

        with set_current_vllm_config(vllm_config, prefix=prefix):
            # The mixin's type-checking stub of this context manager has no
            # return type; SupportsMultiModal's implementation runs.
            # pyrefly: ignore[bad-context-manager]
            with self._mark_tower_model(vllm_config, {"image", "video"}):
                with set_model_tag("NaViTEncoder", is_encoder=True):
                    self.model.vision_tower = NaViT(
                        vision_config,
                        projector_hidden_size=adapter_output_dim,
                        max_num_image_tokens=self._max_image_tokens,
                        quant_config=dense,
                        prefix=maybe_prefix(prefix, "model.vision_tower"),
                    )
                self.model.vision_projection = RowParallelLinear(
                    input_size=adapter_output_dim,
                    output_size=llm_hidden_size,
                    bias=False,
                    input_is_parallel=False,
                    quant_config=dense,
                    prefix=maybe_prefix(prefix, "model.vision_projection"),
                )
                self.perception_emb_norm = MuseRMSNorm(
                    llm_hidden_size, eps=1e-5, has_weight=False
                )

            # pyrefly: ignore[bad-context-manager]
            with self._mark_tower_model(vllm_config, "audio"):
                with set_model_tag("AutoEncoder", is_encoder=True):
                    self.model.audio_tower = AutoEncoder(
                        latent_channels=audio_config.out_hidden_size,
                        patch_size=math.prod(audio_config.patch_sizes),
                        kernel_size=audio_config.conv_kernel_size,
                        patchify_sizes=tuple(audio_config.patch_sizes),
                        d_models=tuple(audio_config.hidden_sizes),
                        n_layers=tuple(audio_config.depths),
                        mlp_ratio=audio_config.mlp_ratio,
                        norm_eps=audio_config.rms_norm_eps,
                        prefix=maybe_prefix(prefix, "model.audio_tower"),
                    )
                if vllm_config.compilation_config.compile_mm_encoder:
                    # Waveforms arrive padded to static buckets (audio.py).
                    self.model.audio_tower._encode_prefill = torch.compile(
                        self.model.audio_tower._encode_prefill,
                        backend="tpu",
                        fullgraph=True,
                        dynamic=False,
                    )

        self._patch_height = vision_config.patch_size
        self._patch_width = vision_config.patch_size
        self._patch_temporal = vision_config.patch_temporal
        self._downsample_factor = vision_config.merge_size

        # Encoder DP shards images across TP ranks.
        self._use_vision_dp = is_vit_use_data_parallel(
            vision_config.num_attention_heads
        )

    def _process_image_input(
        self,
        images: torch.Tensor | list[torch.Tensor],
        target_hw: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """One [C, H, W] tensor per image item.

        Eager encoding and encoder replay both take this list, so an image item
        with several frames is rejected on both paths instead of reaching NaViT
        as a video clip.
        """
        items = MuseSpark1xOmniMixin._process_image_input(self, images, target_hw)
        for item in items:
            if item.dim() != 3:
                raise ValueError(f"Expected [C, H, W] image items, got {item.shape}")
        return items

    def _image_items(self, mm_kwargs: dict[str, Any]) -> list[torch.Tensor]:
        return self._process_image_input(
            mm_kwargs["images"], mm_kwargs.get("image_target_hw")
        )

    def _project_vision(self, h_vision: torch.Tensor) -> torch.Tensor:
        """Project NaViT tokens to the text width and apply perception_emb_norm."""
        h_vision, _ = self.model.vision_projection(h_vision)
        return self.perception_emb_norm(h_vision)

    def _encode_vision(self, vision_items: list[torch.Tensor]) -> list[torch.Tensor]:
        """NaViT -> vision_projection -> perception_emb_norm, split per item."""
        if not vision_items:
            return []
        if self._use_vision_dp:
            h_vision = self._run_dp_sharded_navit(vision_items)
        else:
            h_vision = self.model.vision_tower(vision_items)
        h_vision = self._project_vision(h_vision)

        patch_counts = [self._get_patch_count(item) for item in vision_items]
        assert h_vision.shape[0] == sum(patch_counts), (
            f"NaViT output {h_vision.shape[0]} patches != calculated "
            f"{sum(patch_counts)}. Per-item: {patch_counts}, "
            f"shapes: {[x.shape for x in vision_items]}"
        )
        return list(h_vision.split(patch_counts))

    def _encode_audio(
        self,
        input_features: torch.Tensor | list[torch.Tensor],
        sample_counts: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Encode raw waveforms through the AutoEncoder in one bucketed call."""
        audio_params = next(self.model.audio_tower.parameters(), None)
        # Split stacked waveforms back into individual inputs.
        if isinstance(input_features, torch.Tensor):
            if input_features.dim() >= 2:
                input_features = [
                    input_features[i] for i in range(input_features.shape[0])
                ]
            elif sample_counts is not None and sample_counts.numel() > 1:
                counts = sample_counts.tolist()
                if sum(counts) != input_features.shape[0]:
                    raise ValueError(
                        f"Flat audio samples {input_features.shape[0]} do not "
                        f"match audio_sample_counts {counts}"
                    )
                input_features = list(input_features.split(counts))
            else:
                input_features = [input_features]
        if audio_params is not None:
            input_features = [
                wav.to(device=audio_params.device, dtype=audio_params.dtype)
                for wav in input_features
            ]
        # Audio uses tok_embeddings_norm; perception norm applies only to vision.
        norm = self.model.language_model.tok_embeddings_norm
        return [norm(embed) for embed in self.model.audio_tower(input_features)]

    def _run_dp_sharded_navit(
        self,
        vision_items: list[torch.Tensor],
    ) -> torch.Tensor:
        """Shard images across TP ranks, then gather embeddings in input order."""
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        patch_counts = [self._get_patch_count(item) for item in vision_items]

        shuffle_indices, gpu_sample_counts, gpu_loads = get_load_balance_assignment(
            patch_counts, tp_size
        )
        cum_gpu_sample_counts = [0, *itertools.accumulate(gpu_sample_counts)]

        local_image_indices = shuffle_indices[
            cum_gpu_sample_counts[tp_rank] : cum_gpu_sample_counts[tp_rank + 1]
        ]
        local_vision_items = [vision_items[i] for i in local_image_indices]
        local_output = self.model.vision_tower(local_vision_items)

        # Patch counts already include downsampling.
        max_output_per_rank = max(gpu_loads) if max(gpu_loads) > 0 else 0

        current_len = local_output.shape[0]
        if current_len < max_output_per_rank:
            padding = torch.empty(
                max_output_per_rank - current_len,
                local_output.shape[1],
                dtype=local_output.dtype,
                device=local_output.device,
            )
            local_output_padded = torch.cat([local_output, padding], dim=0)
        else:
            local_output_padded = local_output

        materialized = fresh_buffer(local_output_padded)
        synchronize_tensors(materialized)
        gathered = tensor_model_parallel_all_gather(materialized, dim=0)
        synchronize_tensors(gathered)

        original_order = [torch.Tensor()] * len(vision_items)
        for rank in range(tp_size):
            rank_start = rank * max_output_per_rank
            count = gpu_sample_counts[rank]
            if count == 0:
                continue
            rank_image_indices = shuffle_indices[
                cum_gpu_sample_counts[rank] : cum_gpu_sample_counts[rank + 1]
            ]
            embed_offset = rank_start
            for img_idx in rank_image_indices:
                img_patches = patch_counts[img_idx]
                original_order[img_idx] = gathered[
                    embed_offset : embed_offset + img_patches
                ]
                embed_offset += img_patches

        return torch.cat(original_order, dim=0)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Merge modality embeddings after tok_embeddings_norm, as upstream does."""
        h = self.model.language_model.embed_input_ids(input_ids)
        return merge_multimodal_embeddings(h, multimodal_embeddings, is_multimodal)

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="model.language_model",
            connector=["model.vision_projection.", "perception_emb_norm."],
            tower_model=["model.vision_tower.", "model.audio_tower."],
        )
