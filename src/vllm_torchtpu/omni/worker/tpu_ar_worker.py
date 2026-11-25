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
"""TPU worker for autoregressive vLLM-Omni stages."""

from vllm_omni.worker.mixins import OmniWorkerMixin

from vllm_torchtpu.omni.worker.base import OmniTPUWorkerBase
from vllm_torchtpu.omni.worker.tpu_ar_model_runner import TPUARModelRunner


class TPUARWorker(OmniWorkerMixin, OmniTPUWorkerBase):
    """TPU worker for autoregressive stages; the counterpart of ``GPUARWorker``.

    ``OmniTpuPlatform.get_omni_ar_worker_cls`` points vLLM-Omni here. The
    worker only swaps in the AR runner; ``TPUARModelRunner`` decides which
    stages it serves.
    """

    model_runner_cls = TPUARModelRunner
