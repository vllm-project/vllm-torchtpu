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
"""TPU multimodal composition for Meta Muse Spark Open (vision, video, audio).

Composes upstream vLLM v0.33.0 ``MuseSpark1xOmniMixin`` (same pattern as
``vllm/models/muse_spark/amd/multimodal.py``) over
:class:`TPUMuseSparkForCausalLM`, and re-runs TorchTPU's static
``_merge_multimodal_embeddings`` sweep on ``vllm.models.muse_spark.common.omni``
at import time so ``embed_input_ids`` never triggers dynamic boolean-indexing
recompilations on TPU.
"""

from __future__ import annotations

import vllm.models.muse_spark.common.omni as up_omni
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.models.muse_spark.common.omni import (
    OMNI_WEIGHTS_MAPPER,
    MuseSpark1xOmniMixin,
)
from vllm.models.muse_spark.common.processor import (
    MuseSpark1xOmniDummyInputsBuilder,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_torchtpu import _patch_vllm_merge_multimodal_embeddings
from vllm_torchtpu.models.vllm.muse_spark.mm_processing import (
    TPUMuseSparkOmniMultiModalProcessor,
    TPUMuseSparkOmniProcessingInfo,
)
from vllm_torchtpu.models.vllm.muse_spark.model import TPUMuseSparkForCausalLM

# Ensure upstream omni's imported _merge_multimodal_embeddings binding points to
# TorchTPU's sync-free static prefix-sum merge regardless of import order.
_patch_vllm_merge_multimodal_embeddings()
assert hasattr(up_omni, "_merge_multimodal_embeddings")


@MULTIMODAL_REGISTRY.register_processor(
    TPUMuseSparkOmniMultiModalProcessor,
    info=TPUMuseSparkOmniProcessingInfo,
    dummy_inputs=MuseSpark1xOmniDummyInputsBuilder,
)
class TPUMuseSparkForConditionalGeneration(  # pyrefly: ignore[inconsistent-inheritance]
    MuseSpark1xOmniMixin,
    TPUMuseSparkForCausalLM,
    SupportsMultiModal,
):
    """Muse Spark Open multimodal model (NaViT vision/video + waveform audio)."""

    hf_to_vllm_mapper = TPUMuseSparkForCausalLM.hf_to_vllm_mapper | OMNI_WEIGHTS_MAPPER


__all__ = ["TPUMuseSparkForConditionalGeneration"]
