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
"""TPU patches for the Qwen3-Omni Thinker audio and vision encoders.

1. **Audio encoder (``Qwen3OmniMoeAudioEncoder``)**:
   Moves window-count and ``cu_seqlens`` math to CPU (avoiding TPU float32
   reciprocal-of-100 rounding on exact multiples of 100 frames) and pads each
   clip's chunked mel spectrogram to static ``(8, 16, 32)``-window buckets
   (``104``, ``208``, or ``416`` post-CNN tokens, padded to ``128``, ``256``, or
   ``512`` inside ``TpuMMEncoderAttention._flash_attention``) with ``cu_seqlens``
   padded to a multiple of 128. Patches
   ``prepare_encoder_metadata``, ``forward_with_encoder_metadata``, and
   ``forward`` for compatibility with both ``vllm-omni 0.32.0rc1`` and
   ``0.33.0rc1`` (#2910).
2. **Vision encoder (``Qwen3Omni_VisionTransformer``)**:
   Greedily packs consecutive small image/video items into groups of at most
   ``16,384`` patches, splits single videos exceeding ``16,384`` patches along
   the temporal frame axis via ``embed_multimodal_chunked``, pads each ViT pass
   and its ``cu_seqlens`` to static ``(4096, 8192, 16384)``-patch buckets,
   delegates block and DeepStack merger execution to
   ``forward_with_encoder_metadata``, and cuts the deferred TorchTPU graph
   between chunks via ``synchronize_tensors(..., wait=False)``.

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
``Qwen3OmniMoeAudioEncoder`` and ``Qwen3Omni_VisionTransformer`` also support
static bucketing natively on TPU.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.mm_video_chunking import embed_multimodal_chunked
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)

_QWEN3_OMNI_ARCH = "Qwen3OmniMoeForConditionalGeneration"

# Idempotence markers on wrapped methods.
_AUDIO_CHUNK_LENGTHS_PATCHED = "_tpu_audio_chunk_lengths_patched"
_VIT_STATIC_BUCKETING_PATCHED = "_tpu_vit_static_bucketing_patched"

# Static window-count buckets (in 100-frame / ~1s mel windows, each yielding 13
# post-CNN tokens after three stride-2 convs, padded to 128 / 256 / 512 inside
# TpuMMEncoderAttention._flash_attention) covering <=8s, <=16s, and <=32s audio
# clips. Clips longer than 32 windows round up to a multiple of 8 windows.
_AUDIO_CHUNK_BUCKETS = (8, 16, 32)

# Static patch-count buckets (1024, 2048, and 4096 merged visual tokens at
# spatial_merge_size=2) covering single images and frame-chunked video sub-grids.
# Qwen3OmniMoeThinkerForConditionalGeneration._process_image_input and
# _process_video_input call self.visual(pixel_values, grid_thw) directly inside
# embed_multimodal rather than routing through MMEncoderManager, and
# MMEncoderManager only splits across a token budget without padding patches
# and cu_seqlens to a static XLA shape.
_VISION_PATCH_BUCKETS = (4096, 8192, 16384)


def _select_bucket(n: int, buckets: tuple[int, ...], align: int) -> int:
    for b in buckets:
        if n <= b:
            return b
    return ((n + align - 1) // align) * align


def _pad_rows(x: torch.Tensor, target_rows: int) -> torch.Tensor:
    pad = target_rows - x.shape[0]
    if pad <= 0:
        return x
    return F.pad(x, (0, 0, 0, pad))


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
    ``vllm-omni >= 0.33.0rc1`` (#2910), while also callable from ``forward`` in
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


# Adapted from Qwen3Omni_VisionTransformer.forward in
# vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py
# (vllm-omni==0.32.0rc1 and 0.33.0rc1). In both versions, _process_image_input
# and _process_video_input only invoke self.visual(pixel_values, grid_thw)
# directly and never pass encoder_metadata or call prepare_encoder_metadata /
# forward_with_encoder_metadata on the ViT from outside.
def _padded_vit_forward(
    self: Any,
    x: torch.Tensor,
    grid_thw: torch.Tensor | list[list[int]],
) -> torch.Tensor:
    """Pack small items, chunk large videos, and pad each ViT pass to a bucket."""
    x_dev = x.to(device=self.device, dtype=self.dtype)
    if isinstance(grid_thw, torch.Tensor):
        grid_cpu = grid_thw.to(device="cpu", dtype=torch.int32)
    else:
        grid_cpu = torch.tensor(grid_thw, dtype=torch.int32, device="cpu")

    merge_unit = self.spatial_merge_size**2
    max_bucket = _VISION_PATCH_BUCKETS[-1]

    def _encode_vit_chunk(
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
        **_: Any,
    ) -> list[torch.Tensor]:
        chunk_grid = (
            video_grid_thw.to(device="cpu", dtype=torch.int32)
            if isinstance(video_grid_thw, torch.Tensor)
            else torch.tensor(video_grid_thw, dtype=torch.int32, device="cpu")
        )
        valid_tokens = int(pixel_values_videos.shape[0])
        # Qwen3-Omni's image/video processor max_pixels keeps single images and
        # per-frame patch counts <= 16384 (h * w <= 16384); align=512 provides
        # a bounded fallback if a caller overrides max_pixels.
        target_tokens = _select_bucket(
            valid_tokens, _VISION_PATCH_BUCKETS, align=max(512, merge_unit)
        )
        pad_tokens = target_tokens - valid_tokens
        if pad_tokens > 0:
            pad_frames = pad_tokens // merge_unit
            pad_row = [[pad_frames, self.spatial_merge_size, self.spatial_merge_size]]
            padded_grid = torch.cat(
                [chunk_grid, torch.tensor(pad_row, dtype=torch.int32)],
                dim=0,
            )
            x_padded = _pad_rows(pixel_values_videos, target_tokens)
        else:
            padded_grid = chunk_grid
            x_padded = pixel_values_videos

        valid_frames = int(chunk_grid[:, 0].sum().item())
        metadata = self.prepare_encoder_metadata(padded_grid)
        cu_list = metadata["cu_seqlens"].detach().cpu().tolist()
        metadata["cu_seqlens"] = _pad_cu_seqlens(
            cu_list[: valid_frames + 1], target_tokens, device=x_dev.device
        )
        metadata["max_seqlen"] = None

        merged_all = self.forward_with_encoder_metadata(x_padded, metadata)
        # Cut the deferred TorchTPU graph before the valid-token slice so the
        # 27-layer ViT and DeepStack mergers compile once at the static
        # 16384-patch bucket shape rather than once per valid_tokens length.
        if merged_all.device.type == "tpu":
            synchronize_tensors(merged_all, wait=False)
        return [merged_all[: valid_tokens // merge_unit]]

    # Greedily pack consecutive items into groups of <= max_bucket patches so a
    # batch of many small images runs in ceil(total / max_bucket) ViT passes
    # rather than one 16384-padded pass per image; only single items that exceed
    # max_bucket on their own are split along the frame axis via
    # embed_multimodal_chunked.
    grid_rows = grid_cpu.tolist()
    item_patches = [int(t) * int(h) * int(w) for t, h, w in grid_rows]
    outputs: list[torch.Tensor] = []
    group_start = 0
    group_patches = 0
    px_start = 0

    for idx, n_patches in enumerate(item_patches):
        if n_patches > max_bucket:
            if group_patches > 0:
                outputs.extend(
                    _encode_vit_chunk(
                        pixel_values_videos=x_dev[px_start : px_start + group_patches],
                        video_grid_thw=grid_cpu[group_start:idx],
                    )
                )
                px_start += group_patches
                group_patches = 0
            item_px = x_dev[px_start : px_start + n_patches]
            outputs.extend(
                embed_multimodal_chunked(
                    _encode_vit_chunk,
                    {
                        "pixel_values_videos": item_px,
                        "video_grid_thw": grid_cpu[idx : idx + 1],
                    },
                    max_bucket,
                )
            )
            px_start += n_patches
            group_start = idx + 1
            continue

        if group_patches + n_patches > max_bucket and group_patches > 0:
            outputs.extend(
                _encode_vit_chunk(
                    pixel_values_videos=x_dev[px_start : px_start + group_patches],
                    video_grid_thw=grid_cpu[group_start:idx],
                )
            )
            px_start += group_patches
            group_start = idx
            group_patches = 0

        group_patches += n_patches

    if group_patches > 0:
        outputs.extend(
            _encode_vit_chunk(
                pixel_values_videos=x_dev[px_start : px_start + group_patches],
                video_grid_thw=grid_cpu[group_start : len(item_patches)],
            )
        )

    if len(outputs) == 1:
        return outputs[0]
    return torch.cat(outputs, dim=0)


setattr(_padded_vit_forward, _VIT_STATIC_BUCKETING_PATCHED, True)


def _patch_vision_encoder(vit_cls: type[Any]) -> bool:
    """Install frame chunking and static bucketing on Qwen3Omni_VisionTransformer."""
    orig_forward = vit_cls.forward
    if getattr(orig_forward, _VIT_STATIC_BUCKETING_PATCHED, False):
        return False
    vit_cls._tpu_orig_forward = orig_forward
    vit_cls.forward = _padded_vit_forward
    return True


def maybe_patch_qwen3_omni_thinker(
    architectures: list[str] | None = None,
) -> bool:
    """Apply TPU patches for Qwen3-Omni Thinker audio and vision encoders.

    Applies to ``Qwen3OmniMoeForConditionalGeneration`` only, matched
    literally in ``architectures`` (the pipeline ``model_arch``
    vLLM-Omni gives every Qwen3-Omni stage). Returns ``True`` when the
    vLLM-Omni thinker encoder classes are newly wrapped, and ``False``
    when the architecture does not match or the classes are already patched.
    """
    if not architectures or _QWEN3_OMNI_ARCH not in architectures:
        return False

    from vllm_omni.model_executor.models.qwen3_omni import (
        qwen3_omni_moe_thinker as thinker_mod,
    )

    patched = _patch_audio_encoder(thinker_mod.Qwen3OmniMoeAudioEncoder)
    patched = _patch_vision_encoder(thinker_mod.Qwen3Omni_VisionTransformer) or patched
    if patched:
        logger.info("Applied TPU patch: Qwen3-Omni Thinker audio and vision encoders.")
    return patched
