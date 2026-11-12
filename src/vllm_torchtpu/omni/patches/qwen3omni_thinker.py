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
"""TPU patch for the Qwen3-Omni Thinker: audio encoder chunk-length and bucketing.

In vLLM-Omni 0.32.0rc1, ``Qwen3OmniMoeAudioEncoder.forward`` splits the mel
features into windows of ``n_window * 2 = 100`` frames with

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

In addition, unpadded chunk and token counts trigger an XLA recompilation for
every distinct audio clip length (both in 0.32.0rc1, which boolean-indexes
``padded_feature_after_cnn[padded_mask_after_cnn]``, and in 0.33.0rc1 / #2910,
whose ``prepare_encoder_metadata`` + ``forward_with_encoder_metadata`` path
gathers an unpadded ``valid_index`` before the 32 transformer layers).

This patch replaces ``prepare_encoder_metadata``,
``forward_with_encoder_metadata``, and ``forward`` on
``Qwen3OmniMoeAudioEncoder`` so both the 0.32.0rc1 ``forward(...)`` call site
and the 0.33.0rc1 ``prepare_encoder_metadata`` +
``forward_with_encoder_metadata`` call site in ``_process_audio_input`` move
mel lengths to CPU, compute window counts with exact integer arithmetic, pad
chunk counts to ``_AUDIO_CHUNK_BUCKETS`` and token counts to multiples of 128,
pad ``cu_seqlens`` to a fixed ``(128,)`` shape, and slice back to ``cnn_len``
valid tokens per clip.

Applied from ``TPUARModelRunner.load_model`` for
``Qwen3OmniMoeForConditionalGeneration`` before the stage model is loaded
(kept outside ``patch_registry.PATCHES`` until Omni patches have a dedicated
registration hook; see #1862).

TODO: remove the ``test_upstream_forward_miscounts_exact_window_multiples``
canary in ``tests_omni/models/qwen3_omni/test_qwen3_omni_components.py`` once
vLLM-Omni is bumped to 0.33.0rc1 or later (#2910 / vllm-omni#9010), where
upstream computes window counts on the host with integer arithmetic; keep the
static-shape bucketing in ``_padded_forward_with_encoder_metadata`` until
upstream buckets audio encoder shapes on TPU.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import align_to, synchronize_tensors

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

_QWEN3_OMNI_ARCH = "Qwen3OmniMoeForConditionalGeneration"

# Idempotence marker on the wrapped ``forward``.
_AUDIO_STATIC_BUCKETING_PATCHED = "_tpu_audio_static_bucketing_patched"

# Each 100-frame window (n_window * 2) is 1 s of 16 kHz audio. A single
# 32-window bucket (32 s -> 512 padded tokens) covers
# WhisperFeatureExtractor.chunk_length (30 s) as well as 32 s clips, so one
# dummy audio warmup in TPUARModelRunner._precompile_mm_inputs compiles the
# bucket used by all standard clips. On TPU v6e-8, one 32-window (512-token)
# pass takes ~4.5 ms on device even for a 1 s clip (and N * ~4.5 ms for N
# clips), vs ~15-30 s for an XLA recompile per distinct clip length. Clips
# longer than 32 windows fall back to 8-window alignment with a warning.
_AUDIO_CHUNK_BUCKETS: tuple[int, ...] = (32,)


def _select_bucket(
    size: int,
    buckets: tuple[int, ...],
    align: int,
) -> int:
    for bucket in buckets:
        if size <= bucket:
            return bucket
    aligned = align_to(max(1, size), align)
    logger.warning_once(
        "Qwen3-Omni Thinker encoder input size %d exceeds max bucket %d; "
        "falling back to %d-aligned size %d (may compile a new XLA graph).",
        size,
        buckets[-1],
        align,
        aligned,
    )
    return aligned


def _pad_rows(tensor: torch.Tensor, padded_rows: int) -> torch.Tensor:
    if tensor.shape[0] >= padded_rows:
        return tensor
    return F.pad(tensor, (0, 0, 0, padded_rows - tensor.shape[0]))


def _unwrap_linear(out: torch.Tensor | tuple[torch.Tensor, Any]) -> torch.Tensor:
    """Unwrap ``(tensor, bias)`` tuples from ``ReplicatedLinear`` or ``nn.Linear``."""
    return out[0] if isinstance(out, tuple) else out


def _build_padded_cu_seqlens(
    seqlens: list[int], padded_total: int, device: torch.device
) -> torch.Tensor:
    # TpuMMEncoderAttention builds segment IDs via (positions >= cu_seqlens).sum(0)
    # over positions in [0, padded_total). Padding cu_seqlens with padded_total + 1
    # keeps a fixed (128,) shape while contributing 0 to every position, so trailing
    # padded rows share one isolated segment ID and never attend to real tokens.
    cu_seqlens = torch.tensor([0] + seqlens, dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    pad_len = (-cu_seqlens.numel()) % 128
    return F.pad(cu_seqlens, (0, pad_len), value=padded_total + 1).to(device)


def _padded_prepare_encoder_metadata(
    self: Any, feature_lens: Sequence[int] | torch.Tensor
) -> dict[str, list[int]]:
    """Host-side mel and post-CNN lengths for padded audio encoder forward."""
    from vllm.model_executor.models.qwen3_omni_moe_thinker import (
        _get_feat_extract_output_lengths,
    )

    if isinstance(feature_lens, torch.Tensor):
        feat_lens = [int(n) for n in feature_lens.cpu().tolist()]
    else:
        feat_lens = [int(n) for n in feature_lens]
    cnn_lens = [int(_get_feat_extract_output_lengths(n)) for n in feat_lens]
    return {
        "feature_lens": feat_lens,
        "aftercnn_lens": cnn_lens,
    }


# Adapted from Qwen3OmniMoeAudioEncoder.forward in
# vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py
# (vllm-omni==0.32.0rc1 and 0.33.0rc1). Changes from upstream:
# 1. Uses host integer feature_lens and aftercnn_lens and computes per-clip
#    chunk counts with integer ceil-division instead of float division
#    on device.
# 2. Runs one encoder pass per audio clip, padding chunk counts to
#    _AUDIO_CHUNK_BUCKETS and token counts to multiples of 128.
# 3. Keeps all num_chunks * tokens_per_chunk rows through the conv and
#    transformer layers instead of boolean-indexing / index_select,
#    pads cu_seqlens to (128,) with padded_tokens + 1, passes max_seqlen=None
#    (unused by TpuMMEncoderAttention), and slices hidden_states[:cnn_len].
def _padded_forward_with_encoder_metadata(
    self: Any,
    input_features: torch.Tensor,
    metadata: Mapping[str, Any],
) -> torch.Tensor:
    """Bucketed audio encoder forward consuming host ``feature_lens`` metadata."""
    feat_lens = [int(n) for n in metadata["feature_lens"]]
    cnn_lens = [int(n) for n in metadata["aftercnn_lens"]]
    window_size = self.n_window * 2
    tokens_per_chunk = (((window_size + 1) // 2 + 1) // 2 + 1) // 2
    window_aftercnn = tokens_per_chunk * (self.n_window_infer // window_size)
    device = self.conv2d1.weight.device
    dtype = self.conv2d1.weight.dtype
    feature_list = input_features.to(device=device, dtype=dtype).split(feat_lens, dim=1)
    pos_emb = (
        self.positional_embedding.positional_embedding[:tokens_per_chunk]
        .unsqueeze(0)
        .to(device=device, dtype=dtype)
    )
    outputs = []
    for feature, feat_len, cnn_len in zip(feature_list, feat_lens, cnn_lens):
        num_chunks = _select_bucket(
            align_to(feat_len, window_size) // window_size,
            _AUDIO_CHUNK_BUCKETS,
            align=8,
        )
        padded_tokens = align_to(num_chunks * tokens_per_chunk, 128)
        chunk_seqlens = [window_aftercnn] * (cnn_len // window_aftercnn)
        if cnn_len % window_aftercnn:
            chunk_seqlens.append(cnn_len % window_aftercnn)
        cu_seqlens = _build_padded_cu_seqlens(chunk_seqlens, padded_tokens, device)
        padded_feature = (
            F.pad(feature, (0, num_chunks * window_size - feat_len))
            .T.reshape(num_chunks, window_size, -1)
            .transpose(1, 2)
            .unsqueeze(1)
            .contiguous()
        )
        if device.type == "tpu":
            synchronize_tensors(padded_feature, wait=False)
        conv_chunks = [
            F.gelu(self.conv2d3(F.gelu(self.conv2d2(F.gelu(self.conv2d1(c))))))
            for c in padded_feature.split(self.conv_chunksize, dim=0)
        ]
        conv_out_in = (
            torch.cat(conv_chunks, dim=0)
            .permute(0, 3, 1, 2)
            .contiguous()
            .view(num_chunks, tokens_per_chunk, -1)
        )
        hidden_states = _unwrap_linear(self.conv_out(conv_out_in))
        hidden_states = _pad_rows(
            (hidden_states + pos_emb).reshape(num_chunks * tokens_per_chunk, -1),
            padded_tokens,
        )
        # Qwen3OmniMoeAudioEncoderLayer passes max_seqlen only to
        # Qwen3OmniMoeAudioAttention -> MMEncoderAttention. Because
        # num_heads == num_kv_heads (is_gqa is False), TPU always takes
        # TpuMMEncoderAttention._flash_attention, which ignores max_seqlen and
        # masks segments solely from cu_seqlens.
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, cu_seqlens, max_seqlen=None)
        hidden_states = _unwrap_linear(self.proj1(self.ln_post(hidden_states)))
        hidden_states = _unwrap_linear(self.proj2(self.act(hidden_states)))
        if device.type == "tpu":
            # Cut the encoder XLA graph at the static (padded_tokens, D)
            # shape so the 32-layer encoder compiles once per bucket;
            # the [:cnn_len] slice below is padded back to the static
            # token bucket in _omni_merge before the LM graph cut.
            synchronize_tensors(hidden_states, wait=False)
        outputs.append(hidden_states[:cnn_len])
    return torch.cat(outputs, dim=0)


def _padded_encoder_forward(
    self: Any,
    input_features: torch.Tensor,
    feature_lens: torch.Tensor,
    aftercnn_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bucketed audio encoder forward with ``feature_lens`` moved to CPU."""
    feat_lens = [int(n) for n in feature_lens.cpu().tolist()]
    if aftercnn_lens is not None:
        cnn_lens = [int(n) for n in aftercnn_lens.cpu().tolist()]
    else:
        from vllm.model_executor.models.qwen3_omni_moe_thinker import (
            _get_feat_extract_output_lengths,
        )

        cnn_lens = [int(_get_feat_extract_output_lengths(n)) for n in feat_lens]
    return _padded_forward_with_encoder_metadata(
        self,
        input_features,
        {"feature_lens": feat_lens, "aftercnn_lens": cnn_lens},
    )


setattr(_padded_encoder_forward, _AUDIO_STATIC_BUCKETING_PATCHED, True)


def _patch_audio_encoder(encoder_cls: type[Any]) -> bool:
    """Install static-shape bucketing on ``Qwen3OmniMoeAudioEncoder``."""
    orig_forward = encoder_cls.forward
    if getattr(orig_forward, _AUDIO_STATIC_BUCKETING_PATCHED, False):
        return False
    encoder_cls._tpu_orig_forward = orig_forward
    encoder_cls.prepare_encoder_metadata = _padded_prepare_encoder_metadata
    encoder_cls.forward_with_encoder_metadata = _padded_forward_with_encoder_metadata
    encoder_cls.forward = _padded_encoder_forward
    return True


def maybe_patch_qwen3_omni_thinker(
    model_config: ModelConfig | None = None,
) -> bool:
    """Apply TPU static-shape bucketing to the Qwen3-Omni audio encoder.

    Applies to ``Qwen3OmniMoeForConditionalGeneration`` only, matched
    literally in ``ModelConfig.architectures`` (the pipeline ``model_arch``
    vLLM-Omni gives every Qwen3-Omni stage). Returns ``True`` when the
    vLLM-Omni ``Qwen3OmniMoeAudioEncoder`` class is newly patched, and
    ``False`` when the config does not match or the class is already patched.
    """
    if model_config is None or _QWEN3_OMNI_ARCH not in model_config.architectures:
        return False

    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
        Qwen3OmniMoeAudioEncoder,
    )

    patched = _patch_audio_encoder(Qwen3OmniMoeAudioEncoder)
    if patched:
        logger.info(
            "Applied TPU patch: Qwen3-Omni audio encoder static-shape bucketing."
        )
    return patched
