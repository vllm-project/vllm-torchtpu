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

    Owns the Omni connector lifecycle: the connectors start after the model
    loads and their per-request state is released when a request finishes.
    The AR runner, and the generation runner that follows, add only their
    stage-specific hooks. vLLM-Omni's GPU runners carry the same mixin.
    """

    # The mixin only annotates this attribute; ``init_omni_connectors`` sets it.
    _omni_connector_initialized: bool = False

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        """Load the model, then start the connectors if this stage owns any.

        A stage owns connectors when it consumes a full payload from the
        previous stage, produces inter-stage data
        (``custom_process_next_stage_input_func``), or is given an explicit
        connector role; ``needs_omni_connector`` is the gate vLLM-Omni's GPU
        AR runner uses. A single-stage run such as the Thinker-only pipeline
        must not start the connector threads.
        """
        super().load_model(*args, **kwargs)
        if needs_omni_connector(self.model_config):
            self.init_omni_connectors(self.model_config)

    def _update_states(self, scheduler_output: Any) -> Any:
        """Release connector state for finished requests, then update the batch.

        ``OmniConnectorModelRunnerMixin`` keeps per-request state of its own:
        the external/internal request-id mapping, pending full-payload sends
        and deferred send cleanup. ``TPUModelRunner._update_states`` removes
        finished requests from the runner's request table and persistent
        batch and knows nothing about the mixin, so that state would leak and
        a stale pending full-payload entry could later be flushed against a
        different request. vLLM-Omni's GPU runners release it inside their
        own ``_update_states``; on TPU the base runner is vLLM-TorchTPU's, so
        this override does it before delegating. The mixin is only usable
        once ``init_omni_connectors`` has completed, hence the flag.
        """
        if self._omni_connector_initialized:
            for req_id in scheduler_output.finished_req_ids:
                self.cleanup_finished_request(req_id)
        return super()._update_states(scheduler_output)
