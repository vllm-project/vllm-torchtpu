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
"""Qwen3-VL vision interfaces composed with the native TPU Qwen4 decoder."""

from collections.abc import Iterable
from typing import cast

import torch
from torch import nn
from transformers import Qwen4ExpConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from vllm.config import VllmConfig
from vllm.model_executor.layers.fusion.mm_input_norm import build_mm_input_norm
from vllm.model_executor.models.interfaces import HasInnerState
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.model_executor.models.utils import StageMissingLayer, WeightsMapper
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.tokenizers.registry import cached_tokenizer_from_config

from vllm_torchtpu.logger import init_logger

from .model import Qwen4ExpForCausalLM

logger = init_logger(__name__)


class Qwen4ExpProcessingInfo(Qwen3VLProcessingInfo):
    # The processor accepts the same vision config fields across architectures.
    # pyrefly: ignore[bad-override]
    def get_hf_config(self) -> Qwen4ExpConfig:
        return self.ctx.get_hf_config(Qwen4ExpConfig)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen4ExpProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen4ExpForConditionalGeneration(Qwen3_5ForConditionalGeneration, HasInnerState):
    """Keep upstream processor/vision ownership and native decoder prefixes."""

    # Upstream's composition attributes name its concrete Qwen3 owners.
    # pyrefly: ignore[bad-override-mutable-attribute]
    config: Qwen4ExpConfig
    # pyrefly: ignore[bad-override-mutable-attribute]
    language_model: Qwen4ExpForCausalLM
    # pyrefly: ignore[bad-override-mutable-attribute]
    visual: Qwen3_VisionTransformer | StageMissingLayer

    # Composition follows vllm.models.qwen4_exp.nvidia.model; only the decoder
    # and its state/weight ownership use the TPU implementation.
    requires_raw_input_tokens = True
    packed_modules_mapping = (
        Qwen3_5ForConditionalGeneration.packed_modules_mapping
        | Qwen4ExpForCausalLM.packed_modules_mapping
    )
    hf_to_vllm_mapper = Qwen4ExpForCausalLM.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_prefix={"model.visual.": "visual."}
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        nn.Module.__init__(self)
        self.vllm_config = vllm_config
        self.config = cast(Qwen4ExpConfig, vllm_config.model_config.hf_config)
        vision_config = cast(Qwen3VLVisionConfig, self.config.vision_config)
        text_config = cast(Qwen4ExpTextConfig, self.config.text_config)
        self.model_config = vllm_config.model_config
        self.multimodal_config = self.model_config.get_multimodal_config()
        self.language_model_only = self.multimodal_config.language_model_only
        self.use_data_parallel = self.multimodal_config.mm_encoder_tp_mode == "data"
        self._init_video_pruning(self.multimodal_config)
        self._tokenizer = cached_tokenizer_from_config(self.model_config)
        self.use_deepstack = bool(vision_config.deepstack_visual_indexes)
        if self.use_deepstack:
            raise NotImplementedError(
                "TPU Qwen4 decoder has no deepstack injection contract"
            )
        self.deepstack_num_level = 0
        self.visual_dim = vision_config.out_hidden_size
        self.multiscale_dim = 0
        if self.language_model_only:
            self.visual = StageMissingLayer("vision_tower")
        else:
            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.visual = Qwen3_VisionTransformer(
                    vision_config,
                    norm_eps=text_config.rms_norm_eps,
                    quant_config=vllm_config.quant_config,
                    input_norm=build_mm_input_norm(self.model_config),
                    prefix="visual",
                )
        with self._mark_language_model(vllm_config):
            self.language_model = Qwen4ExpForCausalLM(vllm_config=vllm_config)

    def get_language_model(self) -> Qwen4ExpForCausalLM:
        return self.language_model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            kwargs["intermediate_tensors"] = intermediate_tensors
        return self.language_model(input_ids, positions, inputs_embeds, **kwargs)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        visual_loaded: set[str] = set()
        visual_source: set[str] = set()

        def decoder_weights():
            for name, weight in weights:
                if name.startswith("model.visual."):
                    if self.language_model_only:
                        continue
                    if name in visual_source:
                        raise ValueError(f"Duplicate vision checkpoint tensor {name}")
                    visual_source.add(name)
                    targets = self.visual.load_weights(
                        [(name.removeprefix("model.visual."), weight)]
                    )
                    visual_loaded.update("visual." + target for target in targets)
                else:
                    yield name, weight

        loaded = self.language_model.load_weights(decoder_weights())
        expected = {"visual." + name for name, _ in self.visual.named_parameters()}
        missing = expected - visual_loaded
        if missing:
            raise ValueError(
                f"Incomplete vision checkpoint: {len(missing)} missing parameters"
            )
        self.loaded_visual_parameter_names = tuple(sorted(visual_loaded))
        logger.info(
            "Qwen4Exp vision checkpoint: source_tensors=%d loaded_parameters=%d",
            len(visual_source),
            len(visual_loaded),
        )
        return visual_loaded | {"language_model." + target for target in loaded}

    @classmethod
    # Native pooled state includes the TPU convolution layout.
    # pyrefly: ignore[bad-override]
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, ...]:
        return Qwen4ExpForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    # pyrefly: ignore[bad-override]
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, ...], ...]:
        return Qwen4ExpForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_ple_mamba_state_dtype_from_config(cls, cfg):
        return Qwen4ExpForCausalLM.get_ple_mamba_state_dtype_from_config(cfg)

    @classmethod
    def get_ple_mamba_state_shape_from_config(cls, cfg):
        return Qwen4ExpForCausalLM.get_ple_mamba_state_shape_from_config(cfg)

    @classmethod
    def get_mamba_specs_from_config(cls, cfg):
        return Qwen4ExpForCausalLM.get_mamba_specs_from_config(cfg)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return Qwen4ExpForCausalLM.get_mamba_state_copy_func()

    @classmethod
    def get_mamba_state_copy_funcs(cls, mamba_types):
        return Qwen4ExpForCausalLM.get_mamba_state_copy_funcs(mamba_types)
