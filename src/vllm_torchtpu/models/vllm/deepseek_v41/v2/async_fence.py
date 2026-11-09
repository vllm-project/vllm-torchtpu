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
"""Async-scheduling fence for V4.1's Engram lookback upload.

The runner (``vllm_torchtpu/runner/tpu_runner.py`` ``forward_model``) uploads
the lookback ids through upstream's ``GPUModelRunner._prepare_lookback_token_ids``
(vLLM v0.31.1rc0 ``vllm/v1/worker/gpu_model_runner.py:1052``), which refills a
reused host buffer and copies it with ``non_blocking=True``. That upload is
issued after the runner's ``_record_input_staging_fence``, so under async
scheduling the next step can refill the host buffer while the previous upload
still reads it. Rather than edit the shared runner, V4.1 wraps that one method
to fence its own upload: wait for the previous upload before refilling, record
an event after copying (before the forward is dispatched, like the runner's
staging fence).
"""

import torch


def install_lookback_fence() -> None:
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    original = TPUModelRunner._prepare_lookback_token_ids
    if getattr(original, "_deepseek_v41_fenced", False):
        return

    def prepare(self, num_reqs: int) -> torch.Tensor:
        fence = getattr(self, "_deepseek_v41_lookback_fence", None)
        if fence is not None:
            fence.synchronize()
        ids = original(self, num_reqs)
        fence = torch.tpu.Event()
        fence.record()
        self._deepseek_v41_lookback_fence = fence
        return ids

    prepare._deepseek_v41_fenced = True  # type: ignore[attr-defined]
    TPUModelRunner._prepare_lookback_token_ids = prepare  # type: ignore[method-assign]
