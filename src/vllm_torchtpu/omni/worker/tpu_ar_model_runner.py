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
"""TPU model runner for autoregressive vLLM-Omni stages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from vllm.multimodal.utils import copy_mm_embedding_modality
from vllm.sequence import IntermediateTensors
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
    needs_omni_connector,
)

from vllm_torchtpu.omni.patches.qwen3omni_thinker import (
    _QWEN3_OMNI_ARCH,
    precompile_qwen3_omni_thinker_mm,
)
from vllm_torchtpu.omni.worker.tpu_model_runner import OmniTPUModelRunner

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class TPUARModelRunner(OmniTPUModelRunner, OmniConnectorModelRunnerMixin):
    """TPU runner for autoregressive stages; counterpart of ``GPUARModelRunner``.

    Serves the Thinker stage of a single-stage pipeline. The other pieces
    ``GPUARModelRunner`` adds are checked for at construction and refused with
    ``NotImplementedError`` until they exist on TPU, so an unsupported stage
    config fails at startup with a clear message:

    - inter-stage connectors (``init_omni_connectors`` with the KV transfer
      manager), which every stage of a multi-stage pipeline needs to send or
      receive payloads;
    - the talker path: ``talker_mtp`` multi-token prediction and the
      preprocess/postprocess hooks around each step.

    ``OmniConnectorModelRunnerMixin`` sits on this leaf, as upstream, so the
    connector methods have their home when the multi-stage work lands.
    """

    SUPPORTED_STAGES = frozenset({"thinker"})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._omni_mm_embeds_list: list[torch.Tensor] = []
        if (
            self.model_stage is not None
            and self.model_stage not in self.SUPPORTED_STAGES
        ):
            raise NotImplementedError(
                f"Stage {self.stage_id} ({self.model_stage!r}) is not supported on "
                "TPU: TPUARModelRunner serves the thinker stage only. The talker "
                "needs talker_mtp and the preprocess/postprocess hooks, which are "
                "not implemented on TPU yet."
            )
        if needs_omni_connector(self.model_config):
            raise NotImplementedError(
                f"Stage {self.stage_id} ({self.model_stage!r}) needs inter-stage Omni "
                "connectors, which are not wired on TPU yet. Run the thinker-only "
                "pipeline (qwen3_omni_moe_thinker_only) or another single-stage "
                "deployment."
            )

    def _gather_mm_embeddings(
        self,
        scheduler_output: SchedulerOutput,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        mm_embeds, is_multimodal = super()._gather_mm_embeddings(
            scheduler_output, *args, **kwargs
        )
        self._omni_mm_embeds_list = list(mm_embeds)
        if mm_embeds:
            total = sum(embed.shape[0] for embed in mm_embeds)
            return [torch.arange(total, dtype=torch.int64)], is_multimodal
        return mm_embeds, is_multimodal

    def capture_model(self) -> None:
        super().capture_model()
        if (
            self.enforce_eager
            or not self.supports_mm_inputs
            or not self._pp_is_first
            or _QWEN3_OMNI_ARCH not in self.model_config.architectures
        ):
            return
        with self._precompile_timed("multimodal encoders and merge"):
            precompile_qwen3_omni_thinker_mm(self.model, self.num_tokens_paddings)

    def _get_model_inputs(
        self,
        input_ids: torch.Tensor,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if mm_embed_inputs is not None:
            chunk_embeds, is_multimodal = mm_embed_inputs
            mm_embeds = self._omni_mm_embeds_list
            if chunk_embeds and mm_embeds and chunk_embeds[0].ndim == 1:
                chunk_start = int(chunk_embeds[0][0])
                chunk_end = chunk_start + chunk_embeds[0].shape[0]
                sliced_embeds: list[torch.Tensor] = []
                offset = 0
                for embed in mm_embeds:
                    rows = embed.shape[0]
                    start = max(0, chunk_start - offset)
                    end = min(rows, chunk_end - offset)
                    if start < end:
                        sliced = embed[start:end]
                        copy_mm_embedding_modality(embed, sliced)
                        sliced_embeds.append(sliced)
                    offset += rows
                if chunk_end >= offset:
                    self._omni_mm_embeds_list.clear()
                mm_embed_inputs = (sliced_embeds, is_multimodal)
        return super()._get_model_inputs(input_ids, mm_embed_inputs)

    def forward_model(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> tuple[torch.Tensor | IntermediateTensors, list[torch.Tensor] | None]:
        out, aux = super().forward_model(
            input_ids,
            positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )
        if isinstance(out, OmniOutput) and out.text_hidden_states is not None:
            out = out.text_hidden_states
        return out, aux
