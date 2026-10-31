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
"""TPU Autoregressive (AR) Model Runner for Omni models."""

from typing import Any

from vllm_torchtpu.omni.omni_tpu_model_runner import OmniTPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUModelRunnerOutput


class OmniTPUARModelRunner(OmniTPUModelRunner):
    """TPU model runner for autoregressive stages (thinker, talker) in Omni models.

    The connector lifecycle lives in ``OmniTPUModelRunner``; this class only
    attaches the connector signals to every output the engine consumes.
    """

    def _with_omni_connector_output(self, output: Any) -> Any:
        """Attach connector signals to the ``ModelRunnerOutput`` the engine reads.

        Under async scheduling ``sample_tokens`` returns an
        ``AsyncTPUModelRunnerOutput`` and the engine consumes ``get_output()``,
        so the signals have to live on the wrapped ``ModelRunnerOutput``
        (upstream GPUARModelRunner sets them on the output before wrapping it),
        not on the wrapper, where they would be dropped.
        """
        if output is None or not self._omni_connector_initialized:
            return output
        if isinstance(output, AsyncTPUModelRunnerOutput):
            output._model_runner_output = self.attach_omni_connector_output(
                output._model_runner_output
            )
            return output
        return self.attach_omni_connector_output(output)

    def execute_model(
        self, scheduler_output: Any, intermediate_tensors: Any = None
    ) -> Any:
        output = super().execute_model(scheduler_output, intermediate_tensors)
        # ``None`` is the normal two-phase path: the engine calls sample_tokens()
        # next, which attaches the signals. A zero-token step (and the
        # KV-connector-only variant) returns an output right here and the engine
        # skips sample_tokens(), so attach on this path as well.
        return self._with_omni_connector_output(output)

    def sample_tokens(self, grammar_output: Any = None) -> Any:
        return self._with_omni_connector_output(super().sample_tokens(grammar_output))
