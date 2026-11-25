# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Muse Spark waveform AutoEncoder for TPU.

Upstream source: ``AutoEncoder`` in ``vllm/models/muse_spark/common/audio.py``;
the modules and ``_encode_prefill`` are inherited unchanged.

TPU changes:

- ``forward`` right-pads each waveform to a power-of-two token bucket and
  encodes each bucket as one batch whose size is padded to a power of two, so
  ``_encode_prefill`` sees a small set of static shapes instead of one per
  waveform length. Every layer is causal or per token, so right padding leaves
  the kept tokens unchanged.
"""

import torch
import torch.nn.functional as F
from vllm.models.muse_spark.common.audio import AutoEncoder as UpstreamAutoEncoder

_MIN_BUCKET_TOKENS = 16


def _next_power_of_two(n: int) -> int:
    return 1 << max(n - 1, 0).bit_length()


def audio_bucket_tokens(num_tokens: int) -> int:
    """Token length that a waveform of ``num_tokens`` tokens is padded to."""
    return max(_MIN_BUCKET_TOKENS, _next_power_of_two(num_tokens))


class AutoEncoder(UpstreamAutoEncoder):
    """Upstream AutoEncoder with bucketed, batched waveform encoding."""

    def forward(self, wavs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Encode raw waveforms into [num_tokens, latent_channels] embeddings."""
        samples = [wav.reshape(-1, self.data_channels) for wav in wavs]
        num_tokens = [-(-x.shape[0] // self.patch_size) for x in samples]
        buckets: dict[int, list[int]] = {}
        for index, count in enumerate(num_tokens):
            buckets.setdefault(audio_bucket_tokens(count), []).append(index)

        embeds: dict[int, torch.Tensor] = {}
        for bucket, indices in buckets.items():
            length = bucket * self.patch_size
            rows = [
                F.pad(samples[i], (0, 0, 0, length - samples[i].shape[0]))
                for i in indices
            ]
            rows += [torch.zeros_like(rows[0])] * (
                _next_power_of_two(len(rows)) - len(rows)
            )
            out = self._encode_prefill(torch.stack(rows))
            for row, index in enumerate(indices):
                embeds[index] = out[row, : num_tokens[index]]
        return [embeds[index] for index in range(len(wavs))]
