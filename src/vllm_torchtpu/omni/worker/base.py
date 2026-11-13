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
"""Base TPU worker for vLLM-Omni stages."""

from typing import Any

from vllm_torchtpu.worker.tpu_worker import TPUWorker


class OmniTPUWorkerBase(TPUWorker):
    """Base TPU worker for vLLM-Omni stages, counterpart of ``OmniGPUWorkerBase``.

    Extends ``TPUWorker`` with the vLLM-Omni orchestrator ``collective_rpc``
    interface (``handle_sleep_task``, ``handle_wake_task``, and
    ``encoder_loaded``) while inheriting TPU device initialization, model
    loading, XLA compilation, and HBM KV-cache profiling
    (``determine_available_memory``) from ``TPUWorker``.
    """

    def handle_sleep_task(self, task: Any) -> Any:
        raise NotImplementedError("Sleep mode is not supported for Omni stages on TPU.")

    def handle_wake_task(self, task: Any) -> Any:
        raise NotImplementedError(
            "Wake-up from sleep is not supported for Omni stages on TPU."
        )

    def encoder_loaded(self) -> bool:
        raise NotImplementedError(
            "Querying encoder availability (reference-audio voice cloning) is not "
            "supported for Omni stages on TPU."
        )
