# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse text generation with NaViT vision and waveform audio inputs."""

import itertools
import math
from collections.abc import Iterable
from typing import cast

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
from vllm.model_executor.models.utils import _flatten_embeddings, maybe_prefix
from vllm.model_executor.models.vision import (
    get_load_balance_assignment,
    is_vit_use_data_parallel,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_torchtpu.layers.adapter.quantization.unquantized import VllmUnquantizedConfig
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_tensors

from .audio import AutoEncoder
from .layers import MuseRMSNorm
from .model import MuseSparkForCausalLM
from .processor import (
    MuseSparkDummyInputsBuilder,
    MuseSparkMultiModalProcessor,
    MuseSparkProcessingInfo,
)
from .vision import (
    NaViT,
    NaViTAdapter,
    get_navit_patch_count,
    get_navit_video_patch_count,
)
from .vision_replay import MuseVisionReplayMixin

logger = init_logger(__name__)


@MULTIMODAL_REGISTRY.register_processor(
    MuseSparkMultiModalProcessor,
    info=MuseSparkProcessingInfo,
    dummy_inputs=MuseSparkDummyInputsBuilder,
)
class MuseSparkForConditionalGeneration(
    MuseSparkForCausalLM, MuseVisionReplayMixin, SupportsMultiModal
):
    """Muse Spark: text stack plus NaViT vision and waveform audio."""

    supports_encoder_tp_data = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|patch|>"
        if modality.startswith("video"):
            return "<|video|>"
        if modality.startswith("audio"):
            return "<|audio|>"
        raise ValueError(f"Unsupported Muse Spark modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        hf_config = vllm_config.model_config.hf_config
        vision_config = hf_config.vision_config
        audio_config = hf_config.audio_config
        self.normalize_tok_embeddings = True
        adapter_output_dim = hf_config.projector_hidden_size
        llm_hidden_size = hf_config.text_config.hidden_size
        self.vllm_config = vllm_config
        self._vision_output_dtype = cast(torch.dtype, vllm_config.model_config.dtype)
        # FP32 vision avoids BF16 drift from bucket padding and graph fusion.
        vision_dtype = (
            torch.float32
            if vllm_config.compilation_config.cudagraph_mm_encoder
            else self._vision_output_dtype
        )
        dense = VllmUnquantizedConfig()

        with set_current_vllm_config(vllm_config, prefix=prefix):
            with (
                self._mark_tower_model(vllm_config, {"image", "video"}),
                set_default_torch_dtype(vision_dtype),
            ):
                with set_model_tag("NaViTEncoder", is_encoder=True):
                    self.model.vision_tower = NaViT(
                        vision_config,
                        quant_config=dense,
                        prefix=maybe_prefix(prefix, "model.vision_tower"),
                    )
                self.model.vision_tower.input_dtype = self._vision_output_dtype

                self.model.vision_adapter = NaViTAdapter(
                    dim=hf_config.out_hidden_size,
                    output_dim=adapter_output_dim,
                    use_data_parallel=is_vit_use_data_parallel(
                        vision_config.num_attention_heads
                    ),
                    quant_config=dense,
                    prefix=maybe_prefix(prefix, "model.vision_adapter"),
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
                    self.model.audio_tower._encode_prefill = torch.compile(
                        self.model.audio_tower._encode_prefill,
                        fullgraph=True,
                        dynamic=True,
                    )

        self._patch_height = vision_config.patch_size
        self._patch_width = vision_config.patch_size
        self._patch_temporal = vision_config.patch_temporal
        self._downsample_factor = vision_config.merge_size

        # Encoder DP shards images across TP ranks.
        self._use_vision_dp = is_vit_use_data_parallel(
            vision_config.num_attention_heads
        )
        self._vision_output_dim = hf_config.out_hidden_size

    def get_language_model(self) -> MuseSparkForCausalLM:
        return self

    def _process_image_input(
        self, images: torch.Tensor | list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Convert image inputs to a list of individual image tensors."""
        if isinstance(images, list):
            result: list[torch.Tensor] = []
            for img in images:
                if img.dim() in (3, 4):  # [C, H, W] or [N, C, H, W]
                    result.append(img)
                else:
                    raise ValueError(f"Unexpected image shape: {img.shape}")
            return result
        if images.dim() == 4:  # [B, C, H, W]
            return [images[i] for i in range(images.shape[0])]
        if images.dim() == 5:  # [B, N, C, H, W]
            return [images[i] for i in range(images.shape[0])]
        raise ValueError(f"Unexpected batched image shape: {images.shape}")

    def _process_video_input(
        self, videos: torch.Tensor | list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Convert video inputs to a list of individual video tensors."""
        if isinstance(videos, list):
            for v in videos:
                assert v.dim() == 4, (
                    f"Expected [T, C, H, W] video tensor, got {v.shape}"
                )
            return videos
        assert videos.dim() == 5, (
            f"Expected [B, T, C, H, W] video tensor, got {videos.shape}"
        )
        return [videos[i] for i in range(videos.shape[0])]

    def _get_patch_count(self, item: torch.Tensor) -> int:
        """Number of output patches for an image or video tensor."""
        if item.ndim == 3:  # [C, H, W]
            return get_navit_patch_count(
                img_h=item.shape[1],
                img_w=item.shape[2],
                patch_h=self._patch_height,
                patch_w=self._patch_width,
                downsample_factor=self._downsample_factor,
            )
        if item.ndim == 4:  # [T, C, H, W] or [N, C, H, W]
            return get_navit_video_patch_count(
                video_t=item.shape[0],
                video_h=item.shape[2],
                video_w=item.shape[3],
                patch_t=self._patch_temporal,
                patch_h=self._patch_height,
                patch_w=self._patch_width,
                downsample_factor=self._downsample_factor,
            )
        raise ValueError(f"Unexpected vision item shape: {item.shape}")

    def _encode_vision(self, vision_items: list[torch.Tensor]) -> list[torch.Tensor]:
        """NaViT -> vision_projection -> optional perception_emb_norm, split
        back per item."""
        if not vision_items:
            return []
        vision_params = next(self.model.vision_tower.parameters(), None)
        if vision_params is not None:
            vision_items = [
                x.to(
                    device=vision_params.device,
                    dtype=vision_params.dtype,
                    non_blocking=True,
                )
                for x in vision_items
            ]

        if self._use_vision_dp:
            h_vision = self._run_dp_sharded_navit(vision_items)
        else:
            h_vision = self.model.vision_tower(vision_items)

        h_vision = self.model.vision_adapter(h_vision)
        h_vision, _ = self.model.vision_projection(h_vision)

        if self.normalize_tok_embeddings:
            h_vision = self.perception_emb_norm(h_vision)
        h_vision = h_vision.to(self._vision_output_dtype)

        patch_counts = [self._get_patch_count(item) for item in vision_items]
        assert h_vision.shape[0] == sum(patch_counts), (
            f"NaViT output {h_vision.shape[0]} patches != calculated "
            f"{sum(patch_counts)}. Per-item: {patch_counts}, "
            f"shapes: {[x.shape for x in vision_items]}"
        )
        return list(h_vision.split(patch_counts))

    def _encode_audio(
        self, input_features: torch.Tensor | list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Encode raw waveforms through the AutoEncoder."""
        audio_params = next(self.model.audio_tower.parameters(), None)
        # Split stacked waveforms back into individual inputs.
        if isinstance(input_features, torch.Tensor):
            if input_features.dim() >= 2:
                input_features = [
                    input_features[i] for i in range(input_features.shape[0])
                ]
            else:
                input_features = [input_features]
        results: list[torch.Tensor] = []
        for wav in input_features:
            if audio_params is not None:
                wav = wav.to(device=audio_params.device, dtype=audio_params.dtype)
            audio_embeds = self.model.audio_tower([wav])
            if isinstance(audio_embeds, list) and len(audio_embeds) > 0:
                audio_embed = audio_embeds[0]
            else:
                audio_embed = audio_embeds
            # Audio uses tok_embeddings_norm; perception norm applies only to vision.
            if self.normalize_tok_embeddings:
                audio_embed = self.model.language_model.tok_embeddings_norm(audio_embed)
            results.append(audio_embed)
        return results

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """Encode images/videos through NaViT and waveforms through the
        AutoEncoder, in the order the modalities appear in the kwargs."""
        embeddings: list[torch.Tensor] = []
        for key, value in kwargs.items():
            if key == "images":
                embeddings.extend(
                    self._encode_vision(
                        self._process_image_input(
                            cast(torch.Tensor | list[torch.Tensor], value)
                        )
                    )
                )
            elif key == "videos":
                embeddings.extend(
                    self._encode_vision(
                        self._process_video_input(
                            cast(torch.Tensor | list[torch.Tensor], value)
                        )
                    )
                )
            elif key == "input_features":
                embeddings.extend(
                    self._encode_audio(cast(torch.Tensor | list[torch.Tensor], value))
                )
        return embeddings

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

        if len(local_vision_items) > 0:
            local_output = self.model.vision_tower(local_vision_items)
        else:
            device = next(self.model.vision_tower.parameters()).device
            dtype = next(self.model.vision_tower.parameters()).dtype
            local_output = torch.empty(
                0, self._vision_output_dim, device=device, dtype=dtype
            )

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

        materialized = torch.empty_like(local_output_padded).copy_(local_output_padded)
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
        """Merge modality embeddings with the reference normalization order."""
        h = self.model.language_model.embed_tokens(input_ids)

        if self.model.language_model.tok_embeddings_norm is not None:
            h = self.model.language_model.tok_embeddings_norm(
                h, out_dtype=self.model.language_model.residual_dtype
            )

        if (
            multimodal_embeddings is not None
            and len(multimodal_embeddings) > 0
            and is_multimodal is not None
        ):
            # Fixed-size scatter avoids data-dependent shapes on TPU.
            flattened = _flatten_embeddings(multimodal_embeddings).to(h.dtype)
            h = h.masked_scatter(is_multimodal[:, None].expand_as(h), flattened)

        return h

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="model.language_model",
            connector=[
                "model.vision_adapter.",
                "model.vision_projection.",
                "perception_emb_norm.",
            ],
            tower_model=["model.vision_tower.", "model.audio_tower."],
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Stream all towers through their native loaders, with complete coverage."""
        prefixes = (
            "model.vision_tower.",
            "model.vision_adapter.",
            "model.vision_projection.",
            "model.audio_tower.",
            "perception_emb_norm.",
        )
        params = {
            name: param
            for name, param in self.named_parameters()
            if name.startswith(prefixes)
        }
        loaded: set[str] = set()
        seen: dict[tuple[str, str | None], str] = {}
        qkv_parts: dict[str, set[str]] = {}

        def text_weights():
            for original, tensor in weights:
                name = original
                if not name.startswith(prefixes):
                    yield name, tensor
                    continue
                shard = None
                for projection, shard_id in (
                    ("q_proj", "q"),
                    ("k_proj", "k"),
                    ("v_proj", "v"),
                ):
                    if f".attn.{projection}." in name:
                        name = name.replace(f".{projection}.", ".qkv_proj.")
                        shard = shard_id
                        break
                if name not in params:
                    raise ValueError(f"Unexpected Muse modality tensor: {original}")
                key = (name, shard)
                if key in seen:
                    raise ValueError(f"Duplicate Muse modality tensor: {original}")
                seen[key] = original
                param = params[name]
                loader = getattr(param, "weight_loader", None)
                if shard is not None:
                    assert loader is not None
                    loader(param, tensor, shard)
                    qkv_parts.setdefault(name, set()).add(shard)
                    if qkv_parts[name] == {"q", "k", "v"}:
                        loaded.add(name)
                elif loader is not None:
                    loader(param, tensor)
                    loaded.add(name)
                else:
                    if param.shape != tensor.shape:
                        raise ValueError(f"Muse modality shape mismatch: {original}")
                    param.data.copy_(tensor)
                    loaded.add(name)

        text_loaded = super().load_weights(text_weights())
        missing = set(params) - loaded
        if missing:
            raise ValueError(f"Incomplete Muse modality checkpoint: {sorted(missing)}")
        logger.info(
            "Loaded %d Muse modality tensors on TP rank %d", len(loaded), self.tp_rank
        )
        return text_loaded | loaded
