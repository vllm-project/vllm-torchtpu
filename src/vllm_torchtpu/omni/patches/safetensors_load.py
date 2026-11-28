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
"""TPU patch: stage ``safetensors.torch.load_file`` through CPU for TPU devices.

Problem: safetensors resolves the ``device`` argument of ``load_file`` in its
Rust core, which only knows built-in device types, so any TPU device (``"tpu"``,
``"tpu:N"``, ``torch.device("tpu", N)``) raises
``SafetensorError: device tpu is invalid``. vLLM-Omni's k2-fsa/OmniVoice
loaders (``OmniVoiceGenerator.load_weights`` and the OmniVoice/HiggsAudio
decoders) call ``load_file(path, device=str(device))``, so the OmniVoice
diffusion worker dies at weight load on TPU.

Fix: ``load_file`` is wrapped. For a TPU device the file is read on CPU and
each tensor is moved with torch's ``.to(device)``, which torch_tpu supports.
Every other device goes straight to the original function. Errors from the CPU
read (missing or damaged file) reach the caller unchanged, and keyword
arguments (e.g. ``backend=`` in safetensors >= 0.8) are passed through.

Scope: the wrapper replaces the ``safetensors.torch.load_file`` attribute, so it
applies to every caller in the process that looks the function up at call
time, not only to OmniVoice. A module that bound the name with
``from safetensors.torch import load_file`` before the patch ran keeps the
original. At the pinned vLLM-Omni (0.32.0rc1) the OmniVoice and HiggsAudio
loaders import ``load_file`` inside the loading function, so they are covered.

Effect: k2-fsa/OmniVoice loads its generator and decoder weights on TPU.

Applied from ``apply_omni_tpu_patches()`` in this package, which runs when
OmniTpuPlatform is constructed, before any model weights are loaded.

TODO: delete this module, its call site and its tests once safetensors accepts
torch_tpu devices in ``load_file``
(https://github.com/safetensors/safetensors/issues/988), or once the vLLM-Omni
loaders read checkpoints on CPU before moving them to the device. The canary
``test_safetensors_still_rejects_tpu_device`` in
``tests_omni/patches/test_safetensors_load.py`` fails when safetensors starts
accepting the device.
"""

from functools import cache as run_once
from functools import wraps
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@run_once
def patch_safetensors_load_file_tpu_device() -> None:
    """Wrap ``safetensors.torch.load_file`` to read on CPU for TPU devices."""
    import safetensors.torch as safetensors_torch  # pyrefly: ignore[missing-import]

    original = safetensors_torch.load_file

    @wraps(original)
    def load_file(
        filename: Any, device: Any = "cpu", **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        if str(device).split(":")[0] != "tpu":
            return original(filename, device, **kwargs)
        state = original(filename, "cpu", **kwargs)
        return {name: tensor.to(device) for name, tensor in state.items()}

    safetensors_torch.load_file = load_file
    logger.info("Applied TPU patch: safetensors.load_file reads on CPU for TPU.")
