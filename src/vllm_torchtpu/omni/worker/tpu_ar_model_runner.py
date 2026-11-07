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

from typing import Any

from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
    needs_omni_connector,
)

from vllm_torchtpu.omni.worker.tpu_model_runner import OmniTPUModelRunner


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
