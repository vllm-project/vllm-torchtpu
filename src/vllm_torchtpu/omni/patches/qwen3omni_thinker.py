# Copyright 2026 Google LLC
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
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
"""TPU static-shape bucketing patch for ``Qwen3OmniMoeAudioEncoder``.

Fixes two TPU issues in pinned ``vllm-omni==0.32.0rc1``
(``Qwen3OmniMoeAudioEncoder.forward``):
1. Evaluates 100-frame window counts with integer division on CPU to avoid TPU
   float32 reciprocal rounding on exact multiples of 100 frames
   (see upstream ``vllm-omni#9010``).
2. Splits clips longer than 128 windows along 128-window (``16 * n_window_infer``)
   boundaries, packs consecutive clips into power-of-2 window buckets
   ``(8, 16, 32, 64, 128)``, and compiles one fused TPU graph per bucket via
   ``torch.compile(..., backend="tpu", fullgraph=True, dynamic=False)``.

TODO: retire the 0.32.0rc1 ``xfail`` test in ``test_qwen3_omni_components.py`` once
vLLM-Omni is bumped to >= 0.33.0rc1 (vllm-omni#9010), and remove this module
once upstream supports native static audio encoder bucketing on TPU (tracked
upstream in vllm-project/vllm-omni#7452 and vllm-project/vllm-omni#9566).
"""

from __future__ import annotations

import types
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from vllm.utils.math_utils import cdiv

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)

_QWEN3_OMNI_ARCHS = frozenset(
    {
        "Qwen3OmniMoeForConditionalGeneration",
        "Qwen3OmniMoeThinkerForConditionalGeneration",
    }
)
# Cap at 128 100-frame windows (1,664 = 13 * 128 post-CNN tokens): exact
# multiple of Pallas FlashAttention's 128-token block size at minimum
# per-window latency before quadratic 2D masked-tile overhead grows at 256+.
_AUDIO_CHUNK_BUCKETS: tuple[int, ...] = (8, 16, 32, 64, 128)
# Uniform 128-entry int32 buffer across all buckets: negligible H2D/broadcast
# cost (512 B) and keeps a single grouping budget across every bucket size.
_CU_SEQLENS_BLOCK_SIZE = 128
# Reserve 1 leading ``0`` entry plus up to 1 trailing ``padded_total`` sentinel.
_CU_SEQLENS_RESERVED_ENTRIES = 2


class _AudioSlice(NamedTuple):
    """Frame range and chunk/token counts for one ``<= max_bucket`` audio slice."""

    frame_start: int
    frame_len: int
    num_chunks: int
    valid_tokens: int


def _pad_cu_seqlens(
    cu_cpu: list[int],
    padded_total: int,
    device: torch.device,
) -> torch.Tensor:
    """Pad ``cu_seqlens`` to fixed ``_CU_SEQLENS_BLOCK_SIZE`` with ``padded_total``.

    Padding with ``padded_total`` keeps trailing segments zero-length for
    ``torch.split`` (CPU/native SDPA) while assigning all FlashAttention
    block-padding positions (``t >= padded_total``) to their own isolated
    segment id in ``TpuMMEncoderAttention``.
    """
    pad_len = _CU_SEQLENS_BLOCK_SIZE - len(cu_cpu)
    if pad_len < 0 or (pad_len == 0 and cu_cpu[-1] < padded_total):
        raise ValueError(
            f"cu_seqlens length ({len(cu_cpu)}) exceeds fixed "
            f"_CU_SEQLENS_BLOCK_SIZE ({_CU_SEQLENS_BLOCK_SIZE})."
        )
    return torch.tensor(
        cu_cpu + [padded_total] * pad_len, dtype=torch.int32, device="cpu"
    ).to(device=device)


def _validate_audio_window_config(
    encoder: Any, window_size: int, max_bucket: int
) -> int:
    """Validate ``n_window_infer`` alignment and return chunks per attention window."""
    if encoder.n_window_infer <= 0 or encoder.n_window_infer % window_size != 0:
        raise ValueError(
            f"n_window_infer ({encoder.n_window_infer}) must be a positive multiple "
            f"of 2 * n_window ({window_size})."
        )
    chunks_per_attn_window = encoder.n_window_infer // window_size
    if max_bucket % chunks_per_attn_window != 0:
        raise ValueError(
            f"max_bucket ({max_bucket}) must be divisible by "
            f"n_window_infer // (2 * n_window) ({chunks_per_attn_window}) "
            "so multi-pass splits preserve attention window boundaries."
        )
    max_item_cu_entries = (
        max_bucket // chunks_per_attn_window + _CU_SEQLENS_RESERVED_ENTRIES
    )
    if max_item_cu_entries > _CU_SEQLENS_BLOCK_SIZE:
        raise ValueError(
            f"A full {max_bucket}-window clip needs up to {max_item_cu_entries} "
            f"cu_seqlens entries (n_window_infer // (2 * n_window)="
            f"{chunks_per_attn_window}), exceeding _CU_SEQLENS_BLOCK_SIZE "
            f"({_CU_SEQLENS_BLOCK_SIZE})."
        )
    return chunks_per_attn_window


def _compute_audio_chunk_metadata(
    encoder: Any, feature_lens: torch.Tensor, window_size: int
) -> tuple[list[int], list[int], list[int], int]:
    """Compute host-side window counts and post-CNN token lengths on CPU.

    Avoids ``vllm-omni==0.32.0rc1``'s TPU float32 ``ceil(feature_lens / 100)``
    reciprocal rounding bug (``vllm-omni#9010``) by evaluating exact integer
    division and ``_get_cnn_output_lengths`` on CPU.
    """
    lens_cpu = feature_lens.detach().cpu().to(dtype=torch.long).reshape(-1)
    if (lens_cpu < 0).any().item():
        raise ValueError(f"feature_lens must be non-negative, got {lens_cpu.tolist()}.")

    chunk_nums_t = torch.div(
        lens_cpu + window_size - 1, window_size, rounding_mode="floor"
    )
    rem = torch.remainder(lens_cpu, window_size)
    tail_frames = torch.where(rem == 0, window_size, rem)
    probe = torch.cat(
        [
            torch.tensor([window_size], dtype=torch.long, device="cpu"),
            tail_frames,
        ]
    )
    cnn_probe = encoder._get_cnn_output_lengths(probe)
    tokens_per_chunk = int(cnn_probe[0].item())
    aftercnn_lens_t = torch.where(
        lens_cpu > 0,
        (chunk_nums_t - 1) * tokens_per_chunk + cnn_probe[1:],
        0,
    )
    return (
        lens_cpu.tolist(),
        chunk_nums_t.tolist(),
        aftercnn_lens_t.tolist(),
        tokens_per_chunk,
    )


def _split_audio_clips(
    feature_lens: list[int],
    chunk_nums: list[int],
    aftercnn_lens: list[int],
    window_size: int,
    max_bucket: int,
    tokens_per_chunk: int,
) -> list[_AudioSlice]:
    """Split mel clips into ``<= max_bucket`` frame ranges without copying tensors."""
    max_frames = max_bucket * window_size
    max_tokens = max_bucket * tokens_per_chunk
    slices: list[_AudioSlice] = []
    col_off = 0
    for clip_len, n_chunks, clip_tokens in zip(
        feature_lens, chunk_nums, aftercnn_lens, strict=True
    ):
        if clip_len > 0:
            full_splits = (n_chunks - 1) // max_bucket
            slices.extend(
                _AudioSlice(
                    frame_start=col_off + i * max_frames,
                    frame_len=max_frames,
                    num_chunks=max_bucket,
                    valid_tokens=max_tokens,
                )
                for i in range(full_splits)
            )
            tail_frame_off = full_splits * max_frames
            slices.append(
                _AudioSlice(
                    frame_start=col_off + tail_frame_off,
                    frame_len=clip_len - tail_frame_off,
                    num_chunks=n_chunks - full_splits * max_bucket,
                    valid_tokens=clip_tokens - full_splits * max_tokens,
                )
            )
        col_off += clip_len
    return slices


def _group_audio_slices(
    slices: list[_AudioSlice],
    max_bucket: int,
    window_aftercnn: int,
) -> list[list[_AudioSlice]]:
    """Greedily pack slices into groups fitting ``max_bucket`` and ``cu_seqlens``."""
    groups: list[list[_AudioSlice]] = []
    cur_group: list[_AudioSlice] = []
    cur_chunks = 0
    cur_cu_entries = _CU_SEQLENS_RESERVED_ENTRIES
    for item in slices:
        item_cu_entries = cdiv(item.valid_tokens, window_aftercnn) + 1
        if (
            cur_chunks + item.num_chunks > max_bucket
            or cur_cu_entries + item_cu_entries > _CU_SEQLENS_BLOCK_SIZE
        ) and cur_group:
            groups.append(cur_group)
            cur_group = []
            cur_chunks = 0
            cur_cu_entries = _CU_SEQLENS_RESERVED_ENTRIES
        cur_group.append(item)
        cur_chunks += item.num_chunks
        cur_cu_entries += item_cu_entries
    if cur_group:
        groups.append(cur_group)
    return groups


def _build_bucket_inputs(
    feats_u8_np: np.ndarray,
    dtype: torch.dtype,
    group: list[_AudioSlice],
    window_size: int,
    tokens_per_chunk: int,
    window_aftercnn: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]], int]:
    """Pack one slice group into a single static bucket tensor and ``cu_seqlens``."""
    mel_bins = feats_u8_np.shape[0]
    elem_size = dtype.itemsize
    row_bytes = window_size * elem_size
    group_chunks = sum(item.num_chunks for item in group)
    bucket_chunks = next(b for b in _AUDIO_CHUNK_BUCKETS if group_chunks <= b)
    # Single-threaded NumPy packing avoids PyTorch's multi-thread OpenMP
    # overhead on small host copies.
    x_np = np.zeros((bucket_chunks, 1, mel_bins, row_bytes), dtype=np.uint8)
    cu_list = [0]
    valid_spans: list[tuple[int, int]] = []
    chunk_off = 0
    for idx, item in enumerate(group):
        full_chunks, tail_frames = divmod(item.frame_len, window_size)
        if full_chunks > 0:
            start_b = item.frame_start * elem_size
            end_b = start_b + full_chunks * row_bytes
            x_np[chunk_off : chunk_off + full_chunks, 0] = (
                feats_u8_np[:, start_b:end_b]
                .reshape(mel_bins, full_chunks, row_bytes)
                .transpose(1, 0, 2)
            )
        if tail_frames > 0:
            tail_start_b = (item.frame_start + full_chunks * window_size) * elem_size
            tail_bytes = tail_frames * elem_size
            x_np[chunk_off + full_chunks, 0, :, :tail_bytes] = feats_u8_np[
                :, tail_start_b : tail_start_b + tail_bytes
            ]

        row_start = chunk_off * tokens_per_chunk
        row_valid_end = row_start + item.valid_tokens
        row_chunk_end = (chunk_off + item.num_chunks) * tokens_per_chunk
        if valid_spans and valid_spans[-1][1] == row_start:
            valid_spans[-1] = (valid_spans[-1][0], row_valid_end)
        else:
            valid_spans.append((row_start, row_valid_end))

        cu_list.extend(
            range(row_start + window_aftercnn, row_valid_end, window_aftercnn)
        )
        cu_list.append(row_valid_end)
        # Fence off dead tail tokens of a partial chunk into their own
        # isolated attention segment before the next packed slice starts.
        if idx + 1 < len(group) and row_valid_end < row_chunk_end:
            cu_list.append(row_chunk_end)
        chunk_off += item.num_chunks

    padded_feature = torch.from_numpy(x_np).view(dtype).to(device=device)
    cu_seqlens = _pad_cu_seqlens(
        cu_list, bucket_chunks * tokens_per_chunk, device=device
    )
    return padded_feature, cu_seqlens, valid_spans, bucket_chunks


def _run_static_bucket_pass(
    encoder: Any,
    padded_feature: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Execute one static-shape Conv2d + Transformer + MLP pass.

    Copied from upstream ``vllm-omni@v0.32.0rc1``
    (``vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py``,
    ``Qwen3OmniMoeAudioEncoder.forward``) with TPU static-shape edits noted inline.
    """
    # TPU: skip conv_chunksize split; callers cap batch size <= max_bucket (128).
    padded_embed = F.gelu(encoder.conv2d1(padded_feature))
    padded_embed = F.gelu(encoder.conv2d2(padded_embed))
    padded_embed = F.gelu(encoder.conv2d3(padded_embed))

    b, c, f, t = padded_embed.size()
    padded_embed, _ = encoder.conv_out(
        padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
    )

    positional_embedding = (
        encoder.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
        .unsqueeze(0)
        .to(device=padded_embed.device, dtype=padded_embed.dtype)
    )
    padded_embed = padded_embed + positional_embedding

    # TPU: flatten static [b * t, d] instead of dynamic padded_embed[mask].
    hidden_states = padded_embed.reshape(b * t, -1)

    # TPU: pass max_seqlen=None to avoid host sync inside fullgraph torch.compile.
    for encoder_layer in encoder.layers:
        hidden_states = encoder_layer(
            hidden_states,
            cu_seqlens,
            None,
        )

    hidden_states = encoder.ln_post(hidden_states)
    hidden_states, _ = encoder.proj1(hidden_states)
    hidden_states = encoder.act(hidden_states)
    hidden_states, _ = encoder.proj2(hidden_states)

    return hidden_states


def _get_compiled_bucket_fn(
    encoder: Any, bucket_chunks: int
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Cache one ``torch.compile(backend="tpu")`` callable with a distinct CodeType."""
    cache: dict[int, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] | None = (
        getattr(encoder, "_tpu_compiled_buckets", None)
    )
    if cache is None:
        cache = {}
        encoder._tpu_compiled_buckets = cache
    compiled_fn = cache.get(bucket_chunks)
    if compiled_fn is None:

        def _bucket_forward(x_in: torch.Tensor, cu_in: torch.Tensor) -> torch.Tensor:
            return _run_static_bucket_pass(encoder, x_in, cu_in)

        unique_fn = types.FunctionType(
            _bucket_forward.__code__.replace(
                co_name=f"_audio_bucket_{bucket_chunks}_{id(encoder)}"
            ),
            _bucket_forward.__globals__,
            name=f"_audio_bucket_{bucket_chunks}",
            closure=_bucket_forward.__closure__,
        )
        compiled_fn = torch.compile(
            unique_fn, backend="tpu", fullgraph=True, dynamic=False
        )
        cache[bucket_chunks] = compiled_fn
    return compiled_fn


def _padded_audio_forward(
    self: Any,
    input_features: torch.Tensor,
    feature_lens: torch.Tensor,
    aftercnn_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run ``Qwen3OmniMoeAudioEncoder.forward`` over static power-of-2 buckets."""
    # Recomputed on CPU in _compute_audio_chunk_metadata to avoid a TPU D2H sync.
    del aftercnn_lens
    window_size = self.n_window * 2
    max_bucket = _AUDIO_CHUNK_BUCKETS[-1]
    chunks_per_attn_window = _validate_audio_window_config(
        self, window_size, max_bucket
    )

    feature_lens_list, chunk_nums, aftercnn_lens_list, tokens_per_chunk = (
        _compute_audio_chunk_metadata(self, feature_lens, window_size)
    )
    device = self.conv2d1.weight.device
    dtype = self.conv2d1.weight.dtype
    if sum(aftercnn_lens_list) == 0:
        return torch.empty((0, self.proj2.output_size), device=device, dtype=dtype)

    total_frames = sum(feature_lens_list)
    if input_features.ndim != 2 or input_features.shape[1] < total_frames:
        raise ValueError(
            f"Expected 2D input_features with >= {total_frames} frames, "
            f"got shape {tuple(input_features.shape)}."
        )

    # Upstream stages input_audio_features on TPU; moving to host once here lets
    # us pack static bucket tensors on CPU instead of keying TPU graphs on raw
    # mel frame counts.
    # TODO: keep input_audio_features on host before audio encoding to eliminate
    # the D2H round-trip.
    feats_cpu = input_features.to(device="cpu", dtype=dtype)
    feats_u8_np = feats_cpu.detach().contiguous().view(torch.uint8).numpy()
    window_aftercnn = tokens_per_chunk * chunks_per_attn_window
    slices = _split_audio_clips(
        feature_lens_list,
        chunk_nums,
        aftercnn_lens_list,
        window_size,
        max_bucket,
        tokens_per_chunk,
    )
    groups = _group_audio_slices(slices, max_bucket, window_aftercnn)

    is_tpu = device.type == "tpu"
    use_compiled = is_tpu and not self._tpu_enforce_eager
    outputs: list[torch.Tensor] = []
    for group in groups:
        x, cu_seqlens, valid_spans, bucket_chunks = _build_bucket_inputs(
            feats_u8_np,
            dtype,
            group,
            window_size,
            tokens_per_chunk,
            window_aftercnn,
            device,
        )
        if use_compiled:
            hidden_states = _get_compiled_bucket_fn(self, bucket_chunks)(x, cu_seqlens)
        else:
            hidden_states = _run_static_bucket_pass(self, x, cu_seqlens)
        if is_tpu:
            # Flush each bucket asynchronously so multi-pass uploads overlap
            # with TPU execution in both compiled and eager modes.
            synchronize_tensors(hidden_states, wait=False)

        outputs.extend(hidden_states[start:end] for start, end in valid_spans)

    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)


def _patch_audio_encoder(
    encoder_cls: type[Any], *, enforce_eager: bool = False
) -> bool:
    """Install static-bucketed ``forward`` on ``Qwen3OmniMoeAudioEncoder``."""
    encoder_cls._tpu_enforce_eager = enforce_eager
    if encoder_cls.forward is _padded_audio_forward:
        return False
    encoder_cls.forward = _padded_audio_forward
    return True


def maybe_patch_qwen3_omni_thinker(
    architectures: list[str] | None = None,
    *,
    enforce_eager: bool = False,
) -> bool:
    """Apply TPU patches when a Qwen3-Omni Thinker architecture is loaded."""
    if not architectures or not _QWEN3_OMNI_ARCHS.intersection(architectures):
        return False

    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
        Qwen3OmniMoeAudioEncoder,
    )

    patched = _patch_audio_encoder(
        Qwen3OmniMoeAudioEncoder, enforce_eager=enforce_eager
    )
    if patched:
        logger.info(
            "Applied TPU patch: Qwen3-Omni Thinker audio encoder static bucketing."
        )
    return patched
