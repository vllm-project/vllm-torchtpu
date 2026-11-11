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
"""Muse Spark Open multimodal wrapper for vLLM on TorchTPU.

The text backbone, lm_head and logits live in ``MuseSparkForCausalLM``; this
class adds the perception towers (vision, audio) and the embedding merge.
Subclassing keeps ``model.*`` and ``lm_head`` at the top level, so weight
names, EP metadata and spec decode embedding sharing match the text only
architecture exactly.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
)
from vllm.model_executor.models.utils import _flatten_embeddings
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import NestedTensors

import vllm_torchtpu.envs as envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import merge_multimodal_embeddings_static

from .configuration import MuseSparkConfig
from .processing import (
    AUDIO_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    VIDEO_PLACEHOLDER,
    MuseSparkDummyInputsBuilder,
    MuseSparkMultiModalProcessor,
    MuseSparkProcessingInfo,
)
from .text_model import MuseSparkForCausalLM
from .vision import MuseSparkPerception

logger = init_logger(__name__)


def muse_spark_enabled_towers(vllm_config: VllmConfig) -> tuple[bool, bool]:
    """Return (enable_vision, enable_audio) from vLLM multimodal limits."""
    mm_config = vllm_config.model_config.multimodal_config
    if mm_config is None:
        return False, False
    limit = mm_config.get_limit_per_prompt
    return (limit("image") > 0 or limit("video") > 0), limit("audio") > 0


@MULTIMODAL_REGISTRY.register_processor(
    MuseSparkMultiModalProcessor,
    info=MuseSparkProcessingInfo,
    dummy_inputs=MuseSparkDummyInputsBuilder,
)
class MuseSparkForConditionalGeneration(MuseSparkForCausalLM, SupportsMultiModal):
    """Muse Spark Open (text, image, video, audio) for vLLM on TorchTPU."""

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        del i
        if modality == "image":
            return IMAGE_PLACEHOLDER
        if modality == "video":
            return VIDEO_PLACEHOLDER
        if modality == "audio":
            return AUDIO_PLACEHOLDER
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.vllm_config = vllm_config
        enable_vision, enable_audio = muse_spark_enabled_towers(vllm_config)
        self.perception: MuseSparkPerception | None = None
        if enable_vision or enable_audio:
            self.perception = MuseSparkPerception(
                cast(MuseSparkConfig, vllm_config.model_config.hf_config),
                prefix="perception",
                enable_vision=enable_vision,
                enable_audio=enable_audio,
            )
        logger.info(
            "Muse Spark perception: vision=%s audio=%s", enable_vision, enable_audio
        )
        if envs.VLLM_TPU_MUSE_MEGAKERNEL:
            from . import megakernel_fastpath as _mk_fp

            _mk_fp.prepare_model_for_megakernel(self, vllm_config)

    def get_language_model(self) -> Any:
        return self

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        if self.perception is None:
            return []
        return self.perception.embed_multimodal(**kwargs)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids, self.model.embed_input_ids, is_multimodal=is_multimodal
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        if is_multimodal is None:
            raise ValueError("is_multimodal is required to merge multimodal embeddings")
        return merge_multimodal_embeddings_static(
            inputs_embeds, _flatten_embeddings(multimodal_embeddings), is_multimodal
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        perception_loaded: set[str] = set()
        use_mk = bool(envs.VLLM_TPU_MUSE_MEGAKERNEL)
        _mk_fp: Any = None
        if use_mk:
            from . import megakernel_fastpath as _mk_fp

        def _text_weights():
            for name, tensor in weights:
                bare = name.removeprefix("model.")
                if bare.startswith(MuseSparkPerception.weight_prefixes):
                    if self.perception is not None:
                        perception_loaded.update(
                            f"perception.{n}"
                            for n in self.perception.load_weights([(name, tensor)])
                        )
                    continue
                if (
                    use_mk
                    and _mk_fp is not None
                    and _mk_fp.should_skip_expert_weight(name)
                ):
                    continue
                yield name, tensor

        loaded = super().load_weights(_text_weights())
        if use_mk and _mk_fp is not None:
            loaded |= _mk_fp.skipped_expert_param_names(self)
        return loaded | perception_loaded

    def process_weights_after_loading(self) -> None:
        super().process_weights_after_loading()
        if envs.VLLM_TPU_MUSE_MEGAKERNEL:
            from . import megakernel_fastpath as _mk_fp

            _mk_fp.build_for_model(self)


__all__ = ["MuseSparkForConditionalGeneration", "muse_spark_enabled_towers"]
