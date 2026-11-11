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
"""TPU patch for the Qwen3-Omni Thinker: audio encoder chunk-length math.

vLLM-Omni's ``Qwen3OmniMoeAudioEncoder.forward`` splits the mel features into
windows of ``n_window * 2 = 100`` frames with

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
device. At the pinned vLLM-Omni (0.32.0rc1), ``forward`` is the only caller
of ``_get_cnn_output_lengths`` (processors and token-counting paths call the
module-level ``_get_feat_extract_output_lengths`` helper), so nothing else
observes the device change.

Applied from ``TPUARModelRunner.load_model`` for
``Qwen3OmniMoeForConditionalGeneration`` before the stage model is loaded
(kept outside ``patch_registry.PATCHES`` until Omni patches have a dedicated
registration hook; see #1862).

TODO: delete this module, its call site in ``TPUARModelRunner.load_model`` and
its tests when vLLM-Omni is bumped to 0.33.0rc1 or later (#2910 /
vllm-omni#9010). From that release ``prepare_encoder_metadata`` computes the
window count on the host with integer arithmetic, ``forward`` takes
``aftercnn_lens=None`` and feeds ``_get_cnn_output_lengths`` a NumPy array, so
this patch is both unnecessary and incompatible there. The strict ``xfail``
test ``test_upstream_forward_miscounts_exact_window_multiples`` in
``tests_omni/models/qwen3_omni/test_qwen3_omni_components.py`` reproduces the
failure the patch works around and turns into an XPASS failure once it is gone
upstream or once TPU float division becomes exact for these lengths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

_QWEN3_OMNI_ARCH = "Qwen3OmniMoeForConditionalGeneration"

# Idempotence marker on the wrapped ``forward``.
_AUDIO_CHUNK_LENGTHS_PATCHED = "_tpu_audio_chunk_lengths_patched"


def _wrap_chunk_length_math(encoder_cls: type[Any]) -> bool:
    """Wrap ``forward`` and ``_get_cnn_output_lengths``; ``False`` if already done."""
    orig_forward = encoder_cls.forward
    if getattr(orig_forward, _AUDIO_CHUNK_LENGTHS_PATCHED, False):
        return False
    orig_get_cnn_output_lengths = encoder_cls._get_cnn_output_lengths

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
    encoder_cls.forward = _forward_with_cpu_chunk_lengths
    encoder_cls._get_cnn_output_lengths = _get_cnn_output_lengths_on_device
    return True


def maybe_patch_qwen3_omni_thinker(
    model_config: ModelConfig | None = None,
) -> bool:
    """Route the audio encoder's chunk-length math through CPU on TPU.

    Applies to ``Qwen3OmniMoeForConditionalGeneration`` only, matched
    literally in ``ModelConfig.architectures`` (the pipeline ``model_arch``
    vLLM-Omni gives every Qwen3-Omni stage). Returns ``True`` when the
    vLLM-Omni ``Qwen3OmniMoeAudioEncoder`` class is newly wrapped, and
    ``False`` when the config does not match or the class is already patched.
    """
    if model_config is None or _QWEN3_OMNI_ARCH not in model_config.architectures:
        return False

    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
        Qwen3OmniMoeAudioEncoder,
    )

    patched = _wrap_chunk_length_math(Qwen3OmniMoeAudioEncoder)
    if patched:
        logger.info(
            "Applied TPU patch: Qwen3-Omni audio encoder CPU chunk length math."
        )
    return patched
