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
    """Base worker for every Omni stage on TPU.

    Mirrors vLLM-Omni's ``OmniGPUWorkerBase``, the surface the orchestrator
    drives through ``collective_rpc`` beyond the plain vLLM worker: sleep and
    wake handshakes and the encoder-availability query. None of them is
    implemented on TPU yet; each raises so an orchestrator feature that relies
    on it fails at the call site instead of being ignored.

    Per-stage memory accounting for stages that share one device, which the
    GPU base folds into ``determine_available_memory``, is not implemented
    either: ``TPUWorker.determine_available_memory`` assumes the stage owns its
    chips, which holds for the device pinning the Omni TPU platform applies.
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
