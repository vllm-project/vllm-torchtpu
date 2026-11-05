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
"""TPU patches for the Qwen3-Omni Thinker encoders and multimodal merge.

1. Audio encoder chunk-length math & static-shape bucketing:
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
   exactly. In addition, unpadded chunk and token counts trigger an XLA
   recompilation for every distinct audio length. The patched ``forward`` moves
   ``feature_lens`` and ``aftercnn_lens`` to CPU, computes window counts with
   exact integer arithmetic, pads chunk counts to ``_AUDIO_CHUNK_BUCKETS`` and
   token counts to multiples of 128, and slices back to ``cnn_len`` valid
   tokens per clip.

2. Vision encoder frame-chunking & static-shape bucketing:
   ``Qwen3Omni_VisionTransformer.forward`` chunks video frames with
   ``embed_multimodal_chunked`` and pads patch counts to
   ``_VISION_PATCH_BUCKETS`` while keeping 3-int ``grid_thw`` metadata on CPU
   and all weights, submodules, and activations on TPU.

3. Multimodal embedding merge:
   Pads flattened multimodal embeddings to ``inputs_embeds.shape[0]`` for XLA
   static shapes while keeping ``multimodal_embeddings`` on TPU and moving only
   1D boolean masks (``is_video``, ``is_audio``, ``is_multimodal``) to CPU
   where upstream GPU already indexes on host.

Installed through ``patch_registry.PATCHES`` at ``platform_activation`` and
``model_load`` for ``Qwen3OmniMoeForConditionalGeneration`` only. The thinker
stage runs in a ``TPUWorker`` (``TPUARWorker``), which drives both stages; the
callback returns ``False`` for every other architecture so the registry
retries it once a model config is known.

TODO: remove the CPU ``feature_lens`` hop and the canary
``test_tpu_float_ceil_still_inexact_for_window_multiples`` in
``tests_omni/test_qwen3_omni_audio_encoder_tpu.py`` once vLLM-Omni computes
``chunk_num`` with integer ceil-division, ``(feature_lens + window - 1) // window``,
in https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py
or once TPU float division is exact for these lengths; the static-shape
bucketing in ``_padded_encoder_forward``, ``_padded_vit_forward``, and
``_omni_merge`` remains needed until upstream buckets multimodal encoder shapes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.mm_video_chunking import embed_multimodal_chunked
from vllm_torchtpu.utils import align_to, synchronize_tensors

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

_QWEN3_OMNI_ARCH = "Qwen3OmniMoeForConditionalGeneration"

# Idempotence markers on wrapped methods/functions. The registry runs a
# callback once per process; the markers cover forks and direct calls from
# tests.
_AUDIO_CHUNK_LENGTHS_PATCHED = "_tpu_audio_chunk_lengths_patched"
_OMNI_STATIC_PADDED_ATTR = "_is_omni_static_padded"

# Each 100-frame window (n_window * 2) is 1 s of 16 kHz audio. 500 windows
# covers clips up to 500 s (~8.3 min, 63 8-window attention segments -> 64
# cu_seqlens bounds); longer clips fall back to 8-window alignment.
_AUDIO_CHUNK_BUCKETS: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 500)
_VISION_PATCH_BUCKETS: tuple[int, ...] = (512, 1024, 2048, 4096, 8192, 16384)


def _select_bucket(
    size: int,
    buckets: tuple[int, ...],
    align: int,
) -> int:
    for bucket in buckets:
        if size <= bucket:
            return bucket
    return align_to(max(1, size), align)


def _pad_rows(tensor: torch.Tensor, padded_rows: int) -> torch.Tensor:
    if tensor.shape[0] >= padded_rows:
        return tensor
    return F.pad(tensor, (0, 0, 0, padded_rows - tensor.shape[0]))


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


def maybe_patch_qwen3_omni_thinker(model_config: ModelConfig | None = None) -> bool:
    """Apply TPU patches for Qwen3-Omni Thinker encoders and multimodal merge.

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
    import vllm.model_executor.models.utils as vllm_utils
    from vllm.multimodal.utils import get_mm_embedding_modalities
    from vllm_omni.model_executor.models.qwen3_omni import (
        qwen3_omni_moe_thinker as thinker_mod,
    )

    from vllm_torchtpu import _patch_vllm_merge_multimodal_embeddings

    audio_cls = thinker_mod.Qwen3OmniMoeAudioEncoder
    orig_forward = audio_cls.forward
    if not getattr(orig_forward, _AUDIO_CHUNK_LENGTHS_PATCHED, False):
        # Adapted from Qwen3OmniMoeAudioEncoder.forward in
        # vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py
        # (vllm-omni==0.32.0rc1). Changes from upstream:
        # 1. Moves feature_lens and aftercnn_lens to CPU and computes per-clip
        #    chunk counts with integer ceil-division instead of float division
        #    on device.
        # 2. Runs one encoder pass per audio clip, padding chunk counts to
        #    _AUDIO_CHUNK_BUCKETS and token counts to multiples of 128.
        # 3. Keeps all num_chunks * tokens_per_chunk rows through the conv and
        #    transformer layers instead of boolean-indexing with
        #    padded_mask_after_cnn, pads cu_seqlens to (128,) with
        #    padded_tokens + 1, passes max_seqlen=None (unused by
        #    TpuMMEncoderAttention), and slices hidden_states[:cnn_len] at the end.
        def _padded_encoder_forward(
            self: Any,
            input_features: torch.Tensor,
            feature_lens: torch.Tensor,
            aftercnn_lens: torch.Tensor,
        ) -> torch.Tensor:
            """Bucketed audio encoder forward with ``feature_lens`` on CPU."""
            feature_lens = feature_lens.cpu()
            aftercnn_lens = aftercnn_lens.cpu()
            window_size = self.n_window * 2
            tokens_per_chunk = (((window_size + 1) // 2 + 1) // 2 + 1) // 2
            window_aftercnn = tokens_per_chunk * (self.n_window_infer // window_size)
            device = self.conv2d1.weight.device
            dtype = self.conv2d1.weight.dtype
            feat_lens = feature_lens.tolist()
            cnn_lens = aftercnn_lens.tolist()
            feature_list = input_features.to(device=device, dtype=dtype).split(
                feat_lens, dim=1
            )
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
                cu_seqlens = _build_padded_cu_seqlens(
                    chunk_seqlens, padded_tokens, device
                )
                padded_feature = (
                    F.pad(feature, (0, num_chunks * window_size - feat_len))
                    .T.reshape(num_chunks, window_size, -1)
                    .transpose(1, 2)
                    .unsqueeze(1)
                    .contiguous()
                )
                if device.type == "tpu":
                    synchronize_tensors(padded_feature)
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
                hidden_states, _ = self.conv_out(conv_out_in)
                hidden_states = _pad_rows(
                    (hidden_states + pos_emb).reshape(
                        num_chunks * tokens_per_chunk, -1
                    ),
                    padded_tokens,
                )
                for encoder_layer in self.layers:
                    hidden_states = encoder_layer(
                        hidden_states, cu_seqlens, max_seqlen=None
                    )
                hidden_states, _ = self.proj1(self.ln_post(hidden_states))
                hidden_states, _ = self.proj2(self.act(hidden_states))
                if device.type == "tpu":
                    synchronize_tensors(hidden_states)
                outputs.append(hidden_states[:cnn_len])
            return torch.cat(outputs, dim=0)

        setattr(_padded_encoder_forward, _AUDIO_CHUNK_LENGTHS_PATCHED, True)
        audio_cls._tpu_orig_forward = orig_forward
        audio_cls.forward = _padded_encoder_forward

    vit_cls = thinker_mod.Qwen3Omni_VisionTransformer
    if not getattr(vit_cls.forward, _OMNI_STATIC_PADDED_ATTR, False):
        # Adapted from Qwen3Omni_VisionTransformer.prepare_encoder_metadata and
        # forward_with_encoder_metadata in
        # vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py
        # (vllm-omni==0.32.0rc1) and _Qwen3Omni_VisionTransformer.forward in
        # vllm/model_executor/models/qwen3_omni_moe_thinker.py (vllm==0.32.0).
        # Changes from upstream:
        # 1. Keeps grid_thw on CPU as int32 and chunks multi-frame videos to at
        #    most _VISION_PATCH_BUCKETS[-1] patches via embed_multimodal_chunked.
        # 2. Pads patches, rotary cos/sin, and interpolated pos_embeds to
        #    _VISION_PATCH_BUCKETS.
        # 3. Pads cu_seqlens to (128,) with padded_patches + 1, passes
        #    max_seqlen=None and sequence_lengths=None (unused by
        #    TpuMMEncoderAttention), and slices merged output to valid_tokens.
        def _padded_vit_forward(
            self: Any, x: torch.Tensor, grid_thw: torch.Tensor | list[list[int]]
        ) -> torch.Tensor:
            device = self.patch_embed.proj.weight.device
            x_dev = x.to(device=device, dtype=self.dtype)
            # Matches GPU path (`grid_thw.cpu().numpy()`): keep 3-int grid
            # metadata on CPU while weights and activations stay on TPU.
            grid_cpu = torch.as_tensor(grid_thw).cpu().to(torch.int32)

            def _encode_vit_chunk(
                pixel_values_videos: torch.Tensor,
                video_grid_thw: torch.Tensor,
            ) -> list[torch.Tensor]:
                chunk_seqlens = torch.repeat_interleave(
                    video_grid_thw[:, 1] * video_grid_thw[:, 2],
                    video_grid_thw[:, 0],
                ).tolist()
                num_patches = sum(chunk_seqlens)
                padded_patches = _select_bucket(
                    num_patches, _VISION_PATCH_BUCKETS, align=512
                )
                rotary_cos, rotary_sin = self.rot_pos_emb(video_grid_thw)
                cos = _pad_rows(rotary_cos, padded_patches).to(device)
                sin = _pad_rows(rotary_sin, padded_patches).to(device)
                padded_pixels = _pad_rows(pixel_values_videos, padded_patches)
                padded_pos = None
                if self.apply_vit_abs_pos_embed:
                    pos_embeds = self.fast_pos_embed_interpolate(
                        video_grid_thw.tolist()
                    )
                    padded_pos = _pad_rows(pos_embeds, padded_patches).to(
                        device=device, dtype=self.dtype
                    )
                if device.type == "tpu":
                    synchronize_tensors(padded_pixels)
                    synchronize_tensors([cos, sin])
                    if padded_pos is not None:
                        synchronize_tensors(padded_pos)
                hidden_states = self.patch_embed(padded_pixels)
                if padded_pos is not None:
                    hidden_states = hidden_states + padded_pos
                hidden_states = hidden_states.unsqueeze(1)
                cu_seqlens = _build_padded_cu_seqlens(
                    chunk_seqlens, padded_patches, device
                )
                deepstack_indexes = self.deepstack_visual_indexes or ()
                deepstack_features = []
                for layer_idx, block in enumerate(self.blocks):
                    hidden_states = block(
                        hidden_states,
                        cu_seqlens=cu_seqlens,
                        rotary_pos_emb_cos=cos,
                        rotary_pos_emb_sin=sin,
                        max_seqlen=None,
                        sequence_lengths=None,
                    )
                    if layer_idx in deepstack_indexes:
                        deepstack_features.append(hidden_states)
                merged = [self.merger(hidden_states)]
                for merger, feat in zip(self.merger_list, deepstack_features):
                    merged.append(merger(feat))
                hidden_states = torch.cat(merged, dim=1)
                if device.type == "tpu":
                    synchronize_tensors(hidden_states)
                valid_tokens = num_patches // self.spatial_merge_unit
                return [hidden_states[:valid_tokens]]

            chunks = embed_multimodal_chunked(
                _encode_vit_chunk,
                {"pixel_values_videos": x_dev, "video_grid_thw": grid_cpu},
                _VISION_PATCH_BUCKETS[-1],
            )
            return torch.cat(list(chunks), dim=0)

        setattr(_padded_vit_forward, _OMNI_STATIC_PADDED_ATTR, True)
        vit_cls.forward = _padded_vit_forward

    # First call installs static_merge_multimodal_embeddings on vllm_utils so
    # base_merge captures the static-indexing implementation; the second call
    # after rebinding vllm_utils._merge_multimodal_embeddings propagates
    # _omni_merge across all already-imported vllm.model_executor.models.* modules.
    _patch_vllm_merge_multimodal_embeddings()
    base_merge = vllm_utils._merge_multimodal_embeddings
    if not getattr(base_merge, _OMNI_STATIC_PADDED_ATTR, False):

        def _omni_merge(
            inputs_embeds: torch.Tensor,
            multimodal_embeddings: Any,
            is_multimodal: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if len(multimodal_embeddings) > 0:
                flat = vllm_utils._flatten_embeddings(multimodal_embeddings)
                if 0 < flat.shape[0] < inputs_embeds.shape[0]:
                    flat = _pad_rows(flat, inputs_embeds.shape[0])
                if flat.device.type == "tpu":
                    flat = flat.contiguous()
                    synchronize_tensors(flat)
                multimodal_embeddings = [flat]
            if is_multimodal is not None:
                is_multimodal = is_multimodal.to(inputs_embeds.device)
                if is_multimodal.device.type == "tpu":
                    synchronize_tensors(is_multimodal)
            if inputs_embeds.device.type == "tpu":
                synchronize_tensors(inputs_embeds)
            merged = base_merge(inputs_embeds, multimodal_embeddings, is_multimodal)
            if merged.device.type == "tpu":
                synchronize_tensors(merged)
            return merged

        setattr(_omni_merge, _OMNI_STATIC_PADDED_ATTR, True)
        vllm_utils._merge_multimodal_embeddings = _omni_merge
        _patch_vllm_merge_multimodal_embeddings()

    active_merge = vllm_utils._merge_multimodal_embeddings
    thinker_mod._merge_multimodal_embeddings = active_merge

    orig_check = thinker_mod.check_interleaved_audio_video
    if not getattr(orig_check, _OMNI_STATIC_PADDED_ATTR, False):

        def _check_interleaved(
            is_video: torch.Tensor,
            is_audio: torch.Tensor,
            num_videos: int,
            num_audios: int,
        ) -> bool:
            if not num_videos or not num_audios:
                return False
            # Matches upstream GPU path (`input_ids.cpu()` before span check).
            return orig_check(is_video.cpu(), is_audio.cpu(), num_videos, num_audios)

        setattr(_check_interleaved, _OMNI_STATIC_PADDED_ATTR, True)
        thinker_mod.check_interleaved_audio_video = _check_interleaved

    if not getattr(
        thinker_mod.merge_interleaved_embeddings, _OMNI_STATIC_PADDED_ATTR, False
    ):

        def _merge_interleaved(
            inputs_embeds: torch.Tensor,
            multimodal_embeddings: Any,
            is_video: torch.Tensor,
            is_audio: torch.Tensor,
            is_multimodal: torch.Tensor,
        ) -> torch.Tensor:
            is_video_cpu = is_video.cpu()
            is_audio_cpu = is_audio.cpu()
            is_image_cpu = is_multimodal.cpu() & ~is_video_cpu & ~is_audio_cpu
            modalities = get_mm_embedding_modalities(multimodal_embeddings)
            for modality_mask, modality_name in (
                (is_video_cpu, "video"),
                (is_audio_cpu, "audio"),
                (is_image_cpu, "image"),
            ):
                modality_embeds = [
                    embed
                    for embed, mod in zip(multimodal_embeddings, modalities)
                    if mod == modality_name
                ]
                num_expected_tokens = int(modality_mask.sum().item())
                num_actual_tokens = sum(embed.shape[0] for embed in modality_embeds)
                if num_actual_tokens != num_expected_tokens:
                    raise ValueError(
                        f"Attempted to assign {num_actual_tokens} {modality_name} "
                        f"tokens to {num_expected_tokens} placeholders"
                    )
                if modality_embeds:
                    inputs_embeds = active_merge(
                        inputs_embeds, modality_embeds, modality_mask
                    )
            return inputs_embeds

        setattr(_merge_interleaved, _OMNI_STATIC_PADDED_ATTR, True)
        thinker_mod.merge_interleaved_embeddings = _merge_interleaved

    thinker_cls = thinker_mod.Qwen3OmniMoeThinkerForConditionalGeneration
    if not getattr(thinker_cls.embed_input_ids, _OMNI_STATIC_PADDED_ATTR, False):
        orig_embed_input_ids = thinker_cls.embed_input_ids

        def _patched_embed_input_ids(
            self: Any,
            input_ids: torch.Tensor,
            multimodal_embeddings: Any = None,
            *,
            is_multimodal: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if multimodal_embeddings:
                multimodal_embeddings = list(multimodal_embeddings)
            # Matches GPUModelRunner, where `is_mm_embed` stays on CPU so
            # `torch.nonzero(is_multimodal)` runs on host while embeddings
            # stay on device.
            if is_multimodal is not None and is_multimodal.device.type != "cpu":
                is_multimodal = is_multimodal.cpu()
            out = orig_embed_input_ids(
                self,
                input_ids,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
            )
            if (
                multimodal_embeddings
                and isinstance(out, torch.Tensor)
                and out.device.type == "tpu"
            ):
                sync_tensors = [out]
                if self.use_deepstack and self.deepstack_input_embeds_num_tokens > 0:
                    sync_tensors.extend(self.deepstack_input_embeds)
                synchronize_tensors(sync_tensors)
            return out

        setattr(_patched_embed_input_ids, _OMNI_STATIC_PADDED_ATTR, True)
        thinker_cls.embed_input_ids = _patched_embed_input_ids

    logger.info("Applied TPU patch: Qwen3-Omni Thinker encoders and multimodal merge.")
    return True


@torch.no_grad()
def precompile_qwen3_omni_thinker_mm(
    model: Any,
    token_buckets: Sequence[int],
) -> None:
    """Precompile Qwen3-Omni Thinker audio, vision, and MM merge buckets."""
    audio_tower = model.audio_tower
    if audio_tower is not None:
        device = audio_tower.conv2d1.weight.device
        dtype = audio_tower.conv2d1.weight.dtype
        window_size = audio_tower.n_window * 2
        tokens_per_chunk = (((window_size + 1) // 2 + 1) // 2 + 1) // 2
        mel_bins = audio_tower.num_mel_bins
        for num_chunks in _AUDIO_CHUNK_BUCKETS:
            feat_len = num_chunks * window_size
            cnn_len = num_chunks * tokens_per_chunk
            dummy_features = torch.zeros(
                (mel_bins, feat_len), dtype=dtype, device=device
            )
            dummy_feat_lens = torch.tensor([feat_len], dtype=torch.int32)
            dummy_cnn_lens = torch.tensor([cnn_len], dtype=torch.int32)
            out = audio_tower(dummy_features, dummy_feat_lens, dummy_cnn_lens)
            if device.type == "tpu":
                synchronize_tensors(out)
            logger.info("  -- audio_chunks: %d (tokens: %d)", num_chunks, cnn_len)

    visual = model.visual
    if visual is not None:
        device = visual.patch_embed.proj.weight.device
        dtype = visual.dtype
        patch_dim = int(visual.patch_embed.proj.weight[0].numel())
        for num_patches in _VISION_PATCH_BUCKETS:
            dummy_x = torch.zeros((num_patches, patch_dim), dtype=dtype, device=device)
            dummy_grid = torch.tensor([[1, 16, num_patches // 16]], dtype=torch.int32)
            out = visual(dummy_x, dummy_grid)
            if device.type == "tpu":
                synchronize_tensors(out)
            logger.info("  -- vision_patches: %d", num_patches)

    param = next(model.parameters(), None)
    device = param.device if param is not None else torch.device("cpu")
    dtype = param.dtype if param is not None else torch.bfloat16
    hidden_size = model.config.text_config.hidden_size
    multiscale_dim = model.multiscale_dim
    for num_tokens in token_buckets:
        if num_tokens <= 0:
            continue
        dummy_ids = torch.zeros((num_tokens,), dtype=torch.int32, device=device)
        dummy_mask = torch.zeros((num_tokens,), dtype=torch.bool)
        dummy_mask[0] = True
        dummy_mm = torch.zeros(
            (1, hidden_size + multiscale_dim), dtype=dtype, device=device
        )
        out = model.embed_input_ids(
            dummy_ids,
            multimodal_embeddings=[dummy_mm],
            is_multimodal=dummy_mask,
        )
        model._clear_deepstack_input_embeds(num_tokens)
        if device.type == "tpu":
            sync_list = [out]
            if model.use_deepstack:
                sync_list.extend(model.deepstack_input_embeds)
            synchronize_tensors(sync_list)
        logger.info("  -- mm_merge_tokens: %d", num_tokens)
