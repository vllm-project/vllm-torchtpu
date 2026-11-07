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
"""Base TPU model runner for vLLM-Omni stages."""

from typing import Any, cast

from vllm_omni.config import OmniModelConfig

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


class OmniTPUModelRunner(TPUModelRunner):
    """Shared base for every Omni runner on TPU; counterpart of ``OmniGPUModelRunner``.

    Records which pipeline stage the runner serves. vLLM-Omni hands each stage
    an ``OmniModelConfig`` carrying ``model_stage`` and ``stage_id``; a plain
    vLLM model config has neither, and the leaf runners treat that like a
    single-stage thinker. Everything stage-specific (connectors, talker
    multi-token prediction, generation outputs) belongs to the leaf classes,
    as upstream keeps it on ``GPUARModelRunner`` and
    ``GPUGenerationModelRunner``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_stage: str | None = None
        self.stage_id: int | None = None
        if isinstance(self.model_config, OmniModelConfig):
            # The cast is for the type checker only: where vllm-omni is not
            # installed, ``OmniModelConfig`` resolves to ``Any`` and the
            # isinstance check above does not narrow ``ModelConfig``.
            omni_config = cast(OmniModelConfig, self.model_config)
            self.model_stage = omni_config.model_stage
            self.stage_id = omni_config.stage_id
