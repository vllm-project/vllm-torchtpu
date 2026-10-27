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
"""Base TPU Model Runner for Omni models."""

from typing import Any

from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
    needs_omni_connector,
)

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


class OmniTPUModelRunner(OmniConnectorModelRunnerMixin, TPUModelRunner):
    """Shared base for every Omni runner on TPU.

    Owns the Omni connector lifecycle (initialisation after the model loads,
    per-request cleanup) so that the AR runner, and the generation runner that
    follows, only add their stage-specific hooks. Mirrors upstream, where the
    mixin sits on both GPUARModelRunner and GPUGenerationModelRunner.
    """

    # The mixin only annotates this attribute; ``init_omni_connectors`` sets it.
    _omni_connector_initialized: bool = False

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        super().load_model(*args, **kwargs)
        # Same gate as upstream GPUARModelRunner: a stage owns connectors only
        # when it consumes a full payload, produces inter-stage data
        # (custom_process_next_stage_input_func) or is given an explicit
        # connector role. A plain stage must not start the connector threads.
        if needs_omni_connector(self.model_config):
            self.init_omni_connectors(self.model_config)

    def _update_states(self, scheduler_output: Any) -> Any:
        if self._omni_connector_initialized:
            for req_id in getattr(scheduler_output, "finished_req_ids", ()):
                self.cleanup_finished_request(req_id)
        return super()._update_states(scheduler_output)
