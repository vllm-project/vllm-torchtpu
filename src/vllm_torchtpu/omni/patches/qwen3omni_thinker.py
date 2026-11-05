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
"""TPU patches for the Qwen3-Omni Thinker: audio encoder chunk-length math.

``Qwen3OmniMoeAudioEncoder.forward`` (vLLM-Omni's thinker, a subclass of the
vLLM implementation) splits the mel features into windows of
``n_window * 2 = 100`` frames with

    chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

and then calls ``input_features.T.split(chunk_lengths)`` with one entry per
window. On TPU the float division lowers to a multiplication by the
reciprocal of 100, which float32 cannot represent exactly
(0.01000000074505806), so every exact multiple of the window rounds up:
800 * 0.01000000074505806 = 8.000001, ``torch.ceil`` yields 9 windows for an
8 s clip, the window lengths add up to 900 frames instead of 800, and
``split_with_sizes`` raises. On TPU v6e every multiple of 100 frames up to
100000 is affected and no other length is. CPU evaluates the same expression
exactly.

The patch keeps the upstream ``forward`` and hands it ``feature_lens`` on
CPU. ``chunk_num`` and ``chunk_lengths`` are derived from that tensor and
inherit its device, so the whole chunking computation moves to exact
IEEE-754 arithmetic. ``_get_cnn_output_lengths`` is wrapped so the CNN mask
lengths computed from the CPU chunk lengths come back on the encoder's
device, where ``forward`` compares them with an ``arange`` on the feature
device. ``forward`` is the only caller of ``_get_cnn_output_lengths`` in
both vLLM and vLLM-Omni, so nothing else observes the device change.

Installed through ``patch_registry.PATCHES`` at ``platform_activation`` and
``model_load`` for ``Qwen3OmniMoeForConditionalGeneration`` only. The thinker
stage runs in a ``TPUWorker`` (``TPUARWorker``), which drives both stages; the
callback returns ``False`` for every other architecture so the registry
retries it once a model config is known.

TODO: remove once vLLM-Omni computes ``chunk_num`` with integer
ceil-division, ``(feature_lens + window - 1) // window``, in
https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py
or once TPU float division is exact for these lengths. The canary
``test_tpu_float_ceil_still_inexact_for_window_multiples`` in
``tests_omni/test_qwen3_omni_audio_encoder_tpu.py`` fails when the second
condition holds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

_QWEN3_OMNI_ARCH = "Qwen3OmniMoeForConditionalGeneration"

# Idempotence marker on the wrapped ``forward``. The registry runs a callback
# once per process; the marker covers forks and direct calls from tests.
_AUDIO_CHUNK_LENGTHS_PATCHED = "_tpu_audio_chunk_lengths_patched"


def maybe_patch_qwen3_omni_thinker(model_config: ModelConfig | None = None) -> bool:
    """Route the audio encoder's chunk-length math through CPU on TPU.

    Applies to ``Qwen3OmniMoeForConditionalGeneration`` only, matched
    literally in ``ModelConfig.architectures`` (for a vLLM-Omni stage that is
    the pipeline's ``model_arch``, which is this name for every Qwen3-Omni
    stage). Returns ``False`` without importing vLLM-Omni for anything else,
    including the ``None`` that ``platform_activation`` passes before a model
    is configured, so the registry does not record the patch as applied and
    retries at ``model_load``. For a Qwen3-Omni model the import is
    unconditional: vLLM-Omni is a hard requirement of that model and an import
    error must surface.
    """
    if model_config is None or _QWEN3_OMNI_ARCH not in model_config.architectures:
        return False

    # Imported in the callback, not at module top: the registry imports this
    # module in every process, including ones without the omni extra.
    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
        Qwen3OmniMoeAudioEncoder,
    )

    orig_forward = Qwen3OmniMoeAudioEncoder.forward
    if getattr(orig_forward, _AUDIO_CHUNK_LENGTHS_PATCHED, False):
        return True
    orig_get_cnn_output_lengths = Qwen3OmniMoeAudioEncoder._get_cnn_output_lengths

    def _forward_with_cpu_chunk_lengths(
        self: Any,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
        aftercnn_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Upstream ``forward`` with ``feature_lens`` on CPU.

        ``.cpu()`` returns a copy, so the caller's tensor keeps its device.
        """
        return orig_forward(self, input_features, feature_lens.cpu(), aftercnn_lens)

    def _get_cnn_output_lengths_on_device(
        self: Any, input_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Upstream ``_get_cnn_output_lengths`` moved to the encoder's device.

        ``forward`` feeds it the CPU ``chunk_lengths`` and builds the CNN
        padding mask by comparing the result with an ``arange`` that lives
        on the feature device, so the lengths have to come back there.
        """
        return orig_get_cnn_output_lengths(self, input_lengths).to(
            self.conv2d1.weight.device
        )

    setattr(_forward_with_cpu_chunk_lengths, _AUDIO_CHUNK_LENGTHS_PATCHED, True)
    Qwen3OmniMoeAudioEncoder.forward = _forward_with_cpu_chunk_lengths
    Qwen3OmniMoeAudioEncoder._get_cnn_output_lengths = _get_cnn_output_lengths_on_device
    logger.info("Applied TPU patch: Qwen3-Omni audio encoder CPU chunk length math.")
    return True
