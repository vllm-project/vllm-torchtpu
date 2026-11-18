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
"""TPU patch for k2-fsa/OmniVoice: CPU staging for safetensors load_file.

Problem: OmniVoiceGenerator.load_weights and the OmniVoice/HiggsAudio decoder
loaders call ``safetensors.torch.load_file(path, device=str(device))``. On TPU
that is ``"tpu:N"``. safetensors resolves the device in its Rust core, which
only knows built-in device types, and raises
``SafetensorError: device tpu is invalid``, so the k2-fsa/OmniVoice diffusion
worker dies at weight load.

Fix: ``load_file`` is wrapped to try the requested device first. On
``SafetensorError`` for a TPU device, the file is read on CPU and each tensor is
moved with torch's ``.to()``, which torch_tpu supports. Errors for any other
device are re-raised unchanged. If the CPU read also fails, the problem is the
file rather than the device, so the original error is re-raised. Keyword
arguments (e.g. ``backend=`` in safetensors >= 0.8) are passed through. Calls
that succeed today return unchanged, and once safetensors supports the device
the fallback never runs. The loaders import ``load_file`` inside the function
at call time, so rebinding the ``safetensors.torch`` attribute covers every
caller.

Effect: k2-fsa/OmniVoice loads its generator and decoder weights on TPU.

Applied from ``apply_omni_model_specific_patches()`` in this package.

TODO: delete this module, its call site and its tests once safetensors accepts
torch_tpu devices in ``load_file`` and the safetensors pin is bumped to that
release, or once vLLM-Omni's OmniVoice/HiggsAudio loaders read the checkpoint on
CPU before moving it to the device. The canary
``test_torch_tpu_safetensors_still_rejects_tpu_device`` in
``tests_omni/patches/test_omnivoice.py`` starts failing when the first happens.
"""

from functools import cache as run_once
from functools import wraps
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@run_once
def patch_omnivoice_cpu_safetensors_load() -> None:
    """Wrap ``safetensors.torch.load_file`` with a CPU fallback for TPU devices."""
    import safetensors.torch as safetensors_torch  # pyrefly: ignore[missing-import]
    from safetensors import SafetensorError  # pyrefly: ignore[missing-import]

    original = safetensors_torch.load_file

    @wraps(original)
    def load_file(
        filename: Any, device: Any = "cpu", **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        try:
            return original(filename, device, **kwargs)
        except SafetensorError as device_error:
            if str(device).split(":")[0] != "tpu":
                raise
            try:
                state = original(filename, "cpu", **kwargs)
            except Exception as cpu_error:
                raise device_error from cpu_error
            return {name: tensor.to(device) for name, tensor in state.items()}

    safetensors_torch.load_file = load_file
    logger.info("Applied TPU patch: safetensors.load_file falls back to CPU staging.")
