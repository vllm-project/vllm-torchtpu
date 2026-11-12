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

from typing import Any, Protocol, cast

import torch
from vllm.sequence import IntermediateTensors
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
    needs_omni_connector,
)

from vllm_torchtpu.omni.patches.qwen3omni_thinker import (
    maybe_patch_qwen3_omni_thinker,
)
from vllm_torchtpu.omni.worker.tpu_model_runner import OmniTPUModelRunner

# Model capability flags that ``GPUARModelRunner`` reads (through
# ``_model_omni_flag``) to drive the per-step preprocess / postprocess hooks.
# vLLM-Omni models set them in ``__init__``: the Qwen3-Omni thinker sets both
# to False, the talker sets both to True.
_UNSUPPORTED_HOOK_FLAGS = ("has_preprocess", "has_postprocess")


class _OmniStageModel(Protocol):
    """The part of a vLLM-Omni stage model ``forward_model`` relies on."""

    def make_omni_output(self, model_outputs: Any, **kwargs: Any) -> OmniOutput: ...


class TPUARModelRunner(OmniTPUModelRunner, OmniConnectorModelRunnerMixin):
    """TPU runner for autoregressive stages; counterpart of ``GPUARModelRunner``.

    Serves stages that vLLM's own execute-and-sample loop covers, which is
    what the Qwen3-Omni thinker-only pipeline needs. What ``GPUARModelRunner``
    adds on top of that loop is refused with ``NotImplementedError`` rather
    than approximated, each where it can first be seen:

    - at construction, a stage config that needs inter-stage connectors
      (``needs_omni_connector``): every stage of a multi-stage pipeline sends
      or receives payloads through them, and they are not wired on TPU;
    - at construction, speculative decoding (``speculative_config``): draft-model
      auxiliary hidden states are not verified or wired on ``TPUARModelRunner``;
    - at ``load_model``, a model that declares the per-step
      ``preprocess`` / ``postprocess`` hooks (``has_preprocess`` /
      ``has_postprocess``): the talker path, which runs multi-token
      prediction around every step.

    ``OmniConnectorModelRunnerMixin`` sits on this leaf, as upstream, so the
    connector methods have their home when the multi-stage work lands.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if needs_omni_connector(self.model_config):
            raise NotImplementedError(
                f"Stage {self.stage_id} ({self.model_stage!r}) needs inter-stage Omni "
                "connectors, which are not wired on TPU yet. Run the thinker-only "
                "pipeline (qwen3_omni_moe_thinker_only) or another single-stage "
                "deployment."
            )
        if getattr(self, "speculative_config", None) is not None:
            raise NotImplementedError(
                f"Stage {self.stage_id} ({self.model_stage!r}): speculative decoding "
                "(speculative_config) is not supported by TPUARModelRunner yet."
            )

    def load_model(self) -> None:
        """Load the stage model, then refuse one whose hooks nobody would call."""
        maybe_patch_qwen3_omni_thinker(self.model_config)
        super().load_model()
        model = self.get_model()
        has_unsupported_flags = [
            flag
            for flag in _UNSUPPORTED_HOOK_FLAGS
            if bool(getattr(model, flag, False))
        ]
        if len(has_unsupported_flags) > 0:
            raise NotImplementedError(
                f"Stage {self.stage_id} ({self.model_stage!r}): "
                f"{type(model).__name__} declares {', '.join(has_unsupported_flags)}, "
                "so it relies on the per-step preprocess/postprocess hooks (the "
                "talker's multi-token prediction), which TPUARModelRunner does not "
                "run yet."
            )

    def forward_model(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> tuple[torch.Tensor | IntermediateTensors, None]:
        """Run the stage model and return ``(hidden_states, aux_hidden_states)``.

        vLLM-Omni's runner (``OmniGPUModelRunner._model_forward``) normalises
        whatever a stage model returns through ``model.make_omni_output`` into
        an ``OmniOutput``: the text hidden states to sample from plus the
        multimodal sidecar the next stage consumes. ``TPUModelRunner.forward_model``
        instead unwraps by shape, for vLLM's ``(hidden_states,
        aux_hidden_states)`` draft-model convention, which an ``OmniOutput``
        (a four-field NamedTuple) or a staged thinker's ``(hidden_states,
        captured_layers)`` would match by accident. This override applies the
        Omni contract: pipeline-parallel ``IntermediateTensors`` pass through,
        everything else goes through ``make_omni_output`` and only
        ``text_hidden_states`` continues. The sidecar is dropped: it is the
        payload for a next stage, and ``__init__`` refuses configs that have
        one. Aux hidden states are always ``None``; no Omni stage registers
        draft-model aux layers.
        """
        kwargs: dict[str, Any] = {}
        if intermediate_tensors is not None:
            kwargs["intermediate_tensors"] = intermediate_tensors
        out = self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if isinstance(out, IntermediateTensors):
            return out, None
        # @support_torch_compile may hand a single tensor back as a 1-element
        # list or tuple; the same unwrap as TPUModelRunner.forward_model.
        if isinstance(out, (list, tuple)) and len(out) == 1:
            out = out[0]
        if not isinstance(out, OmniOutput):
            out = cast(_OmniStageModel, self.model).make_omni_output(out)
        return out.text_hidden_states, None
