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
from vllm.multimodal.utils import (
    copy_mm_embedding_modality,
    set_mm_embedding_modality,
)
from vllm.sequence import IntermediateTensors
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
    needs_omni_connector,
)

from vllm_torchtpu.omni.worker.tpu_model_runner import OmniTPUModelRunner
from vllm_torchtpu.utils import synchronize_tensors

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
        if self.enforce_eager or not self.supports_mm_inputs or not self._pp_is_first:
            return
        self._precompile_mm_inputs()

    @torch.no_grad()
    def _precompile_mm_inputs(self) -> None:
        if self.mm_budget is None or not self.mm_budget.mm_max_toks_per_item:
            return
        # Sort modalities so every TP worker runs the same encoder (and its TP
        # collectives) in lockstep; MultiModalBudget builds its dict from a set
        # whose iteration order varies across processes with PYTHONHASHSEED.
        modalities = sorted(self.mm_budget.mm_max_toks_per_item)
        if "image" in modalities and "video" in modalities:
            # Image and video share the vision tower; warming "image" compiles
            # the max patch bucket in one pass instead of synthesizing and
            # chunking a full max_model_len dummy video.
            modalities.remove("video")
        with self._precompile_timed("multimodal encoders and merge"):
            sample_embed: torch.Tensor | None = None
            sample_modality: str | None = None
            for modality in modalities:
                batched_dummy_mm_inputs = self._get_mm_dummy_batch(modality, 1)
                dummy_encoder_outputs = self.model.embed_multimodal(
                    **batched_dummy_mm_inputs
                )
                if not dummy_encoder_outputs:
                    continue
                outputs_list = list(dummy_encoder_outputs)
                synchronize_tensors(outputs_list)
                first_embed = outputs_list[0]
                if (
                    sample_embed is None
                    or first_embed.shape[-1] > sample_embed.shape[-1]
                ):
                    sample_embed = first_embed
                    sample_modality = modality
            self.mm_budget.reset_cache()
            if sample_embed is None or sample_modality is None:
                return
            num_tokens = next((n for n in self.num_tokens_paddings if n > 0), 16)
            dummy_mm = set_mm_embedding_modality(
                torch.zeros_like(sample_embed[:1]), sample_modality
            )
            dummy_input_ids = torch.zeros(
                (num_tokens,), dtype=torch.int32, device=self.device
            )
            is_multimodal = torch.zeros((num_tokens,), dtype=torch.bool)
            is_multimodal[0] = True
            _, inputs_embeds = super()._get_model_inputs(
                dummy_input_ids, ([dummy_mm], is_multimodal)
            )
            if inputs_embeds is not None:
                synchronize_tensors(inputs_embeds)
            _, inputs_embeds = super()._get_model_inputs(dummy_input_ids, None)
            if inputs_embeds is not None:
                synchronize_tensors(inputs_embeds)

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
