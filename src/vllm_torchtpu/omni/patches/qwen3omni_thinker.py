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
"""TPU patch for the Qwen3-Omni Thinker audio encoder (chunk lengths and bucketing).

vLLM-Omni's ``Qwen3OmniMoeAudioEncoder.forward`` has two TPU obstacles:

1. **Chunk-length float32 division on TPU (vllm-omni 0.32.0rc1)**:
   ``chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()`` lowers
   to multiplication by ``0.01000000074505806`` in float32 on TPU, rounding up
   every exact multiple of ``100`` frames (e.g. ``800`` -> ``9`` windows) and
   crashing ``split_with_sizes``. Moving the length and ``cu_seqlens`` math to
   CPU evaluates the window counts with exact integer/IEEE-754 arithmetic.
2. **Dynamic conv and attention shapes across audio clip lengths**:
   Without padding, every distinct window count ``W`` and token count ``T``
   triggers a new XLA compilation across the 3 Conv2d layers and 32 transformer
   layers. Padding each clip's chunked mel spectrogram along dim 0 to static
   ``(8, 16, 32)``-window buckets (``104``, ``208``, or ``416`` post-CNN tokens,
   padded to ``128``, ``256``, or ``512`` inside
   ``TpuMMEncoderAttention._flash_attention``) and padding ``cu_seqlens`` to a
   fixed multiple of 128 compiles static encoder graphs for ``<= 8s``,
   ``<= 16s``, and ``<= 32s`` audio clips.

In ``vllm-omni >= 0.33.0rc1`` (#2910 / vllm-omni#9010), ``_process_audio_input``
calls ``audio_tower.prepare_encoder_metadata(...)`` and
``audio_tower.forward_with_encoder_metadata(...)`` directly without calling
``audio_tower.forward(...)``. We patch all three entrypoints on
``Qwen3OmniMoeAudioEncoder`` so both ``0.32.0rc1`` and ``0.33.0rc1`` execute the
same static-shape bucketed path.

Applied from ``TPUARModelRunner.load_model`` for
``Qwen3OmniMoeForConditionalGeneration`` before the stage model is loaded
(kept outside ``patch_registry.PATCHES`` until Omni patches have a dedicated
registration hook; see #1862).

TODO: once vLLM-Omni is bumped to 0.33.0rc1 or later (#2910 / vllm-omni#9010),
the strict ``xfail`` test
``test_upstream_forward_miscounts_exact_window_multiples`` in
``tests_omni/models/qwen3_omni/test_qwen3_omni_components.py`` will turn into an
XPASS failure signaling that the 0.32.0rc1 ``torch.ceil`` workaround is no
longer needed; delete this module, its call site in
``TPUARModelRunner.load_model``, and its tests once upstream
``Qwen3OmniMoeAudioEncoder`` also supports static window bucketing natively on
TPU.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)

_QWEN3_OMNI_ARCH = "Qwen3OmniMoeForConditionalGeneration"

# Idempotence marker on the wrapped ``forward``.
_AUDIO_CHUNK_LENGTHS_PATCHED = "_tpu_audio_chunk_lengths_patched"

# Static window-count buckets (in 100-frame / ~1s mel windows, each yielding 13
# post-CNN tokens after three stride-2 convs, padded to 128 / 256 / 512 inside
# TpuMMEncoderAttention._flash_attention) covering <=8s, <=16s, and <=32s audio
# clips. Clips longer than 32 windows round up to a multiple of 8 windows.
_AUDIO_CHUNK_BUCKETS = (8, 16, 32)


def _select_bucket(n: int, buckets: tuple[int, ...], align: int) -> int:
    for b in buckets:
        if n <= b:
            return b
    return ((n + align - 1) // align) * align


def _pad_cu_seqlens(
    cu_cpu: list[int], padded_total: int, device: torch.device
) -> torch.Tensor:
    """Pad ``cu_seqlens`` on CPU to a static multiple of 128 and move to ``device``.

    ``TpuMMEncoderAttention._flash_attention`` compares token index ``i`` against
    segment bounds ``[start, end)``. Appending ``padded_total`` once places the
    padded rows ``[valid_total, padded_total)`` into a dedicated segment so valid
    tokens never attend to zero-padded keys, and filling the remaining slots with
    ``padded_total + 1`` creates empty segments beyond ``padded_total`` that
    match no token index ``0 <= i < padded_total``.
    """
    seq = list(cu_cpu)
    if seq[-1] < padded_total:
        seq.append(padded_total)
    target_len = ((len(seq) + 127) // 128) * 128
    seq.extend([padded_total + 1] * (target_len - len(seq)))
    return torch.tensor(seq, dtype=torch.int32, device=device)


def _compute_cnn_output_lengths(
    encoder: Any, input_lengths: torch.Tensor
) -> torch.Tensor:
    """Evaluate the 3x stride-2 Conv2d output length formula on CPU."""
    out = encoder._get_cnn_output_lengths(input_lengths.cpu())
    return torch.as_tensor(out, dtype=torch.long).cpu()


def _prepare_audio_encoder_metadata(
    self: Any, feature_lens: list[int] | np.ndarray | torch.Tensor
) -> dict[str, Any]:
    """Build CPU-side window and segment metadata for ``Qwen3OmniMoeAudioEncoder``.

    Compatible with ``Qwen3OmniMoeAudioEncoder.prepare_encoder_metadata`` in
    ``vllm-omni >= 0.33.0rc1`` (#2910 / vllm-omni#9010, where
    ``_process_audio_input`` passes ``feature_lens`` and reads
    ``metadata["aftercnn_lens"]``), while also callable from ``forward`` in
    ``0.32.0rc1``.
    """
    if isinstance(feature_lens, torch.Tensor):
        lens_list = [int(x) for x in feature_lens.detach().cpu().tolist()]
    elif isinstance(feature_lens, np.ndarray):
        lens_list = [int(x) for x in feature_lens.tolist()]
    else:
        lens_list = [int(x) for x in feature_lens]

    window_size = self.n_window * 2
    chunk_nums: list[int] = []
    chunk_lengths: list[int] = []

    for clip_len in lens_list:
        n_chunks = max(1, (clip_len + window_size - 1) // window_size)
        chunk_nums.append(n_chunks)
        for j in range(n_chunks):
            if j == n_chunks - 1:
                rem = clip_len % window_size
                c_len = window_size if (rem == 0 and clip_len > 0) else rem
            else:
                c_len = window_size
            chunk_lengths.append(c_len)

    chunk_lens_t = torch.tensor(chunk_lengths, dtype=torch.long)
    aftercnn_chunk_lens = (
        _compute_cnn_output_lengths(self, chunk_lens_t).tolist()
        if chunk_lengths
        else []
    )

    aftercnn_lens: list[int] = []
    cu_seqlens_per_clip: list[list[int]] = []
    ratio = self.n_window_infer // window_size
    idx = 0
    for n_chunks in chunk_nums:
        clip_chunk_aftercnn = [
            int(x) for x in aftercnn_chunk_lens[idx : idx + n_chunks]
        ]
        idx += n_chunks
        aftercnn_lens.append(sum(clip_chunk_aftercnn))
        cu = [0]
        for s in range(0, n_chunks, ratio):
            cu.append(cu[-1] + sum(clip_chunk_aftercnn[s : s + ratio]))
        cu_seqlens_per_clip.append(cu)

    return {
        "feature_lens": lens_list,
        "chunk_nums": chunk_nums,
        "aftercnn_lens": aftercnn_lens,
        "cu_seqlens_per_clip": cu_seqlens_per_clip,
    }


# Adapted from Qwen3OmniMoeAudioEncoder.forward in
# vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py
# (vllm-omni==0.32.0rc1; split into prepare_encoder_metadata and
# forward_with_encoder_metadata in vllm-omni==0.33.0rc1, vllm-omni#9010).
def _padded_forward_with_encoder_metadata(
    self: Any,
    input_features: torch.Tensor,
    encoder_metadata: dict[str, Any],
) -> torch.Tensor:
    """Static-bucketed audio encoder forward consuming ``encoder_metadata``."""
    device = self.conv2d1.weight.device
    dtype = self.conv2d1.weight.dtype
    feats = input_features.to(device=device, dtype=dtype)

    feature_lens: list[int] = encoder_metadata["feature_lens"]
    if not feature_lens:
        return feats.new_empty((0, self.output_dim))

    window_size = self.n_window * 2
    clip_features = feats.split(feature_lens, dim=-1)
    outputs: list[torch.Tensor] = []

    for clip_idx, clip_feat in enumerate(clip_features):
        clip_len = feature_lens[clip_idx]
        cnn_len = encoder_metadata["aftercnn_lens"][clip_idx]
        if clip_len == 0 or cnn_len == 0:
            outputs.append(feats.new_empty((0, self.output_dim)))
            continue

        n_chunks = encoder_metadata["chunk_nums"][clip_idx]
        num_chunks = _select_bucket(n_chunks, _AUDIO_CHUNK_BUCKETS, align=8)
        padded_frames = num_chunks * window_size
        if clip_feat.shape[-1] < padded_frames:
            clip_padded = F.pad(clip_feat, (0, padded_frames - clip_feat.shape[-1]))
        else:
            clip_padded = clip_feat[:, :padded_frames]

        # [mel_bins, num_chunks * window_size] -> [num_chunks, 1, mel_bins, window_size]
        x = (
            clip_padded.T.contiguous()
            .view(num_chunks, window_size, clip_feat.shape[0])
            .transpose(1, 2)
            .unsqueeze(1)
        )
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))

        b, c, f, t = x.size()
        x = self.conv_out(x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
        if isinstance(x, tuple):
            x = x[0]
        pos_emb = self.positional_embedding.positional_embedding[:t, :].to(
            device=device, dtype=dtype
        )
        x = x + pos_emb.unsqueeze(0)

        padded_tokens = b * t
        hidden_states = x.reshape(padded_tokens, -1)

        cu_seqlens = _pad_cu_seqlens(
            encoder_metadata["cu_seqlens_per_clip"][clip_idx],
            padded_tokens,
            device=device,
        )
        for layer in self.layers:
            # Qwen3OmniMoeAudioEncoderLayer.forward requires a 3rd positional
            # argument (max_seqlen), which TpuMMEncoderAttention.forward_oot
            # ignores on TPU (using cu_seqlens only).
            hidden_states = layer(hidden_states, cu_seqlens, None)
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        # Cut the deferred TorchTPU graph before the clip-length slice so the
        # 32-layer transformer body compiles once at the static bucket shape.
        if hidden_states.device.type == "tpu":
            synchronize_tensors(hidden_states, wait=False)
        outputs.append(hidden_states[:cnn_len])

    if len(outputs) == 1:
        return outputs[0]
    return torch.cat(outputs, dim=0)


def _padded_audio_forward(
    self: Any,
    input_features: torch.Tensor,
    feature_lens: torch.Tensor | list[int] | np.ndarray,
    aftercnn_lens: torch.Tensor | list[int] | None = None,
) -> torch.Tensor:
    """Static-bucketed ``Qwen3OmniMoeAudioEncoder.forward`` for 0.32 and 0.33."""
    del aftercnn_lens
    metadata = _prepare_audio_encoder_metadata(self, feature_lens)
    return _padded_forward_with_encoder_metadata(self, input_features, metadata)


setattr(_padded_audio_forward, _AUDIO_CHUNK_LENGTHS_PATCHED, True)


def _patch_audio_encoder(encoder_cls: type[Any]) -> bool:
    """Install static-bucketed methods on ``Qwen3OmniMoeAudioEncoder``."""
    orig_forward = encoder_cls.forward
    if getattr(orig_forward, _AUDIO_CHUNK_LENGTHS_PATCHED, False):
        return False
    encoder_cls._tpu_orig_forward = orig_forward
    encoder_cls.prepare_encoder_metadata = _prepare_audio_encoder_metadata
    encoder_cls.forward_with_encoder_metadata = _padded_forward_with_encoder_metadata
    encoder_cls.forward = _padded_audio_forward
    return True


def maybe_patch_qwen3_omni_thinker(
    architectures: list[str] | None = None,
) -> bool:
    """Apply TPU patches for the Qwen3-Omni Thinker audio encoder.

    Applies to ``Qwen3OmniMoeForConditionalGeneration`` only, matched
    literally in ``architectures`` (the pipeline ``model_arch``
    vLLM-Omni gives every Qwen3-Omni stage). Returns ``True`` when the
    vLLM-Omni ``Qwen3OmniMoeAudioEncoder`` class is newly wrapped, and
    ``False`` when the architecture does not match or the class is already
    patched.
    """
    if not architectures or _QWEN3_OMNI_ARCH not in architectures:
        return False

    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
        Qwen3OmniMoeAudioEncoder,
    )

    patched = _patch_audio_encoder(Qwen3OmniMoeAudioEncoder)
    if patched:
        logger.info(
            "Applied TPU patch: Qwen3-Omni Thinker audio encoder static bucketing."
        )
    return patched
