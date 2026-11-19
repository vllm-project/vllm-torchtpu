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
1. Evaluates 100-frame window counts with integer division on ``torch.device("cpu")``
   to avoid TPU float32 reciprocal rounding on exact multiples of 100 frames
   (see upstream ``vllm-omni#9010``).
2. Splits clips longer than 128 windows along 128-window (``16 * n_window_infer``)
   boundaries, packs consecutive clips into power-of-2 window buckets
   ``(8, 16, 32, 64, 128)``, and compiles one fused TPU graph per bucket via
   ``torch.compile(..., backend="tpu", fullgraph=True, dynamic=False)``.

TODO: retire the 0.32.0rc1 ``xfail`` test in ``test_qwen3_omni_components.py`` once
vLLM-Omni is bumped to >= 0.33.0rc1 (#2910 / vllm-omni#9010), and remove this
module once upstream supports static audio encoder bucketing natively on TPU.
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

_QWEN3_OMNI_ARCH = "Qwen3OmniMoeForConditionalGeneration"
_CPU_DEVICE = torch.device("cpu")
_AUDIO_CHUNK_LENGTHS_PATCHED = "_tpu_audio_chunk_lengths_patched"
# Cap at 128 100-frame windows (1,664 = 13 * 128 post-CNN tokens): exact
# multiple of Pallas FlashAttention's 128-token block size at minimum
# per-window latency before quadratic 2D masked-tile overhead grows at 256+.
_AUDIO_CHUNK_BUCKETS: tuple[int, ...] = (8, 16, 32, 64, 128)
_CU_SEQLENS_BLOCK_SIZE = 128
# Reserve 1 leading `0` entry plus up to 1 trailing `padded_total` sentinel.
_CU_SEQLENS_RESERVED_ENTRIES = 2


class _MelSubClip(NamedTuple):
    """Mel-window slice ready for static-bucket packing."""

    chunks_u8: np.ndarray
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
        cu_cpu + [padded_total] * pad_len, dtype=torch.int32, device=_CPU_DEVICE
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
    encoder: Any, feature_lens: torch.Tensor
) -> tuple[list[int], list[int], list[int], int]:
    """Compute host-side window counts and post-CNN token lengths on CPU.

    Avoids ``vllm-omni==0.32.0rc1``'s TPU float32 ``ceil(feature_lens / 100)``
    reciprocal rounding bug (``vllm-omni#9010``) by evaluating exact integer
    division and ``_get_cnn_output_lengths`` on ``_CPU_DEVICE``.
    """
    lens_cpu = feature_lens.detach().cpu().to(dtype=torch.long).reshape(-1)
    if lens_cpu.numel() == 0:
        return [], [], [], 0
    if bool((lens_cpu < 0).any().item()):
        raise ValueError(f"feature_lens must be non-negative, got {lens_cpu.tolist()}.")

    window_size = encoder.n_window * 2
    chunk_nums_t = torch.div(
        lens_cpu + window_size - 1, window_size, rounding_mode="floor"
    )
    rem = torch.remainder(lens_cpu, window_size)
    tail_frames = torch.where(rem == 0, window_size, rem)
    probe = torch.cat(
        [
            torch.tensor([window_size], dtype=torch.long, device=_CPU_DEVICE),
            tail_frames,
        ]
    )
    cnn_probe = torch.as_tensor(
        encoder._get_cnn_output_lengths(probe), dtype=torch.long, device=_CPU_DEVICE
    )
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


def _pack_mel_windows_cpu(
    feats_cpu: torch.Tensor,
    feature_lens: list[int],
    chunk_nums: list[int],
    aftercnn_lens: list[int],
    window_size: int,
    max_bucket: int,
    tokens_per_chunk: int,
) -> list[_MelSubClip]:
    """Chunk, zero-pad, and split ``> max_bucket`` mel clips via ``uint8`` views."""
    feats_contig = feats_cpu if feats_cpu.is_contiguous() else feats_cpu.contiguous()
    elem_size = feats_contig.element_size()
    mel_bins = feats_contig.shape[0]
    # View mel frames as raw uint8 bytes because NumPy has no native bfloat16 dtype.
    raw = (
        feats_contig.detach().view(torch.uint8).numpy().reshape(mel_bins, -1, elem_size)
    )

    sub_items: list[_MelSubClip] = []
    col_off = 0
    for clip_idx, clip_len in enumerate(feature_lens):
        clip_raw = raw[:, col_off : col_off + clip_len]
        col_off += clip_len
        if clip_len == 0:
            continue
        n_chunks = chunk_nums[clip_idx]
        padded_frames = n_chunks * window_size
        if clip_len < padded_frames:
            clip_pad = np.zeros((mel_bins, padded_frames, elem_size), dtype=np.uint8)
            clip_pad[:, :clip_len] = clip_raw
        else:
            clip_pad = clip_raw[:, :padded_frames]
        # Keep a strided view; assigning into x_np copies directly in one pass.
        clip_chunks = clip_pad.reshape(
            mel_bins, n_chunks, window_size * elem_size
        ).transpose(1, 0, 2)
        offset = 0
        rem_chunks = n_chunks
        rem_tokens = aftercnn_lens[clip_idx]
        while rem_chunks > max_bucket:
            sub_toks = max_bucket * tokens_per_chunk
            sub_items.append(
                _MelSubClip(
                    chunks_u8=clip_chunks[offset : offset + max_bucket],
                    num_chunks=max_bucket,
                    valid_tokens=sub_toks,
                )
            )
            offset += max_bucket
            rem_chunks -= max_bucket
            rem_tokens -= sub_toks
        sub_items.append(
            _MelSubClip(
                chunks_u8=clip_chunks[offset:],
                num_chunks=rem_chunks,
                valid_tokens=rem_tokens,
            )
        )
    return sub_items


def _run_static_bucket_pass(
    encoder: Any,
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Execute one static-shape Conv2d + Transformer + MLP pass.

    Adapted from ``vllm-omni@v0.32.0rc1``
    (``vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py``,
    ``Qwen3OmniMoeAudioEncoder.forward``):
    - Runs ``conv2d1..3``, ``conv_out``, ``positional_embedding``, ``layers``,
      ``ln_post``, ``proj1``, ``act``, and ``proj2`` in identical order.

    Adapted for TPU:
    - Omits ``conv_chunksize`` batch splitting because callers cap
      ``x.size(0) <= max_bucket = 128``.
    - Always zero-pads short ``< 100``-frame chunks to ``100`` frames (matching
      upstream batched ``pad_sequence``) rather than the batch max.
    - Skips mid-forward boolean masking (``padded_embed[mask]``) and passes
      ``max_seqlen=None``, keeping ``[b * t, d]`` static for full-graph
      ``torch.compile``.
    """
    x = F.gelu(encoder.conv2d1(x))
    x = F.gelu(encoder.conv2d2(x))
    x = F.gelu(encoder.conv2d3(x))

    b, c, f, t = x.size()
    x, _ = encoder.conv_out(x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
    pos_emb = encoder.positional_embedding.positional_embedding[:t, :].to(
        device=x.device, dtype=x.dtype
    )
    hidden_states = (x + pos_emb.unsqueeze(0)).reshape(b * t, -1)
    for layer in encoder.layers:
        hidden_states = layer(hidden_states, cu_seqlens, None)

    hidden_states = encoder.ln_post(hidden_states)
    hidden_states, _ = encoder.proj1(hidden_states)
    hidden_states = encoder.act(hidden_states)
    hidden_states, _ = encoder.proj2(hidden_states)
    return hidden_states


def _get_compiled_bucket_fn(
    encoder: Any, num_chunks: int
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Cache one ``torch.compile(backend="tpu")`` callable with a distinct CodeType."""
    cache: dict[int, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] | None = (
        getattr(encoder, "_tpu_compiled_buckets", None)
    )
    if cache is None:
        cache = {}
        encoder._tpu_compiled_buckets = cache
    compiled_fn = cache.get(num_chunks)
    if compiled_fn is None:

        def _bucket_forward(x_in: torch.Tensor, cu_in: torch.Tensor) -> torch.Tensor:
            return _run_static_bucket_pass(encoder, x_in, cu_in)

        unique_fn = types.FunctionType(
            _bucket_forward.__code__.replace(
                co_name=f"_audio_bucket_{num_chunks}_{id(encoder)}"
            ),
            _bucket_forward.__globals__,
            name=f"_audio_bucket_{num_chunks}",
            closure=_bucket_forward.__closure__,
        )
        compiled_fn = torch.compile(
            unique_fn, backend="tpu", fullgraph=True, dynamic=False
        )
        cache[num_chunks] = compiled_fn
    return compiled_fn


def _padded_audio_forward(
    self: Any,
    input_features: torch.Tensor,
    feature_lens: torch.Tensor,
    aftercnn_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run ``Qwen3OmniMoeAudioEncoder.forward`` over static power-of-2 buckets.

    Same as upstream ``vllm-omni@v0.32.0rc1`` ``forward``:
    - Preserves per-clip ``window_aftercnn`` attention segments and returns
      ``[sum(aftercnn_lens), proj2.output_size]`` on the encoder device.

    Adapted for TPU:
    - Evaluates window counts on CPU, zero-pads mel windows on host, splits
      clips >128 windows on attention-window boundaries, packs items into
      ``_AUDIO_CHUNK_BUCKETS``, isolates padded spans in fixed-128
      ``cu_seqlens``, runs per-bucket ``torch.compile``, and slices
      ``valid_spans`` after ``proj2``.
    """
    del aftercnn_lens
    window_size = self.n_window * 2
    buckets = _AUDIO_CHUNK_BUCKETS
    max_bucket = buckets[-1]
    chunks_per_attn_window = _validate_audio_window_config(
        self, window_size, max_bucket
    )

    feature_lens_list, chunk_nums, aftercnn_lens_list, tokens_per_chunk = (
        _compute_audio_chunk_metadata(self, feature_lens)
    )
    device = self.conv2d1.weight.device
    dtype = self.conv2d1.weight.dtype
    if len(feature_lens_list) == 0 or sum(aftercnn_lens_list) == 0:
        return torch.empty((0, self.proj2.output_size), device=device, dtype=dtype)

    # Upstream stages input_audio_features on TPU; moving to host once here lets
    # us pack static bucket tensors on CPU instead of keying TPU graphs on raw
    # mel frame counts. Keeping input_audio_features on host is a follow-up.
    feats_cpu = input_features.to(device=_CPU_DEVICE, dtype=dtype)
    mel_bins = feats_cpu.shape[0]
    elem_size = feats_cpu.element_size()
    window_aftercnn = tokens_per_chunk * chunks_per_attn_window

    sub_items = _pack_mel_windows_cpu(
        feats_cpu=feats_cpu,
        feature_lens=feature_lens_list,
        chunk_nums=chunk_nums,
        aftercnn_lens=aftercnn_lens_list,
        window_size=window_size,
        max_bucket=max_bucket,
        tokens_per_chunk=tokens_per_chunk,
    )

    groups: list[list[_MelSubClip]] = []
    cur_group: list[_MelSubClip] = []
    cur_chunks = 0
    cur_cu_entries = _CU_SEQLENS_RESERVED_ENTRIES
    for item in sub_items:
        item_cu_entries = cdiv(item.valid_tokens, window_aftercnn) + 1
        if (
            cur_chunks + item.num_chunks > max_bucket
            or cur_cu_entries + item_cu_entries > _CU_SEQLENS_BLOCK_SIZE
        ) and len(cur_group) > 0:
            groups.append(cur_group)
            cur_group = []
            cur_chunks = 0
            cur_cu_entries = _CU_SEQLENS_RESERVED_ENTRIES
        cur_group.append(item)
        cur_chunks += item.num_chunks
        cur_cu_entries += item_cu_entries
    if len(cur_group) > 0:
        groups.append(cur_group)

    is_tpu = device.type == "tpu"
    outputs: list[torch.Tensor] = []
    for group in groups:
        group_chunks = sum(sub.num_chunks for sub in group)
        num_chunks = next(b for b in buckets if group_chunks <= b)
        x_np = np.zeros(
            (num_chunks, 1, mel_bins, window_size * elem_size), dtype=np.uint8
        )
        cu_list = [0]
        valid_spans: list[tuple[int, int]] = []
        local_chunk_off = 0
        for sub_idx, sub in enumerate(group):
            x_np[local_chunk_off : local_chunk_off + sub.num_chunks, 0] = sub.chunks_u8
            row_start = local_chunk_off * tokens_per_chunk
            row_valid_end = row_start + sub.valid_tokens
            row_chunk_end = (local_chunk_off + sub.num_chunks) * tokens_per_chunk
            if len(valid_spans) > 0 and valid_spans[-1][1] == row_start:
                valid_spans[-1] = (valid_spans[-1][0], row_valid_end)
            else:
                valid_spans.append((row_start, row_valid_end))
            pos = row_start + window_aftercnn
            while pos < row_valid_end:
                cu_list.append(pos)
                pos += window_aftercnn
            cu_list.append(row_valid_end)
            if sub_idx + 1 < len(group) and row_valid_end < row_chunk_end:
                cu_list.append(row_chunk_end)
            local_chunk_off += sub.num_chunks

        x = torch.from_numpy(x_np).view(dtype).to(device=device)
        cu_seqlens = _pad_cu_seqlens(
            cu_list, num_chunks * tokens_per_chunk, device=device
        )
        if is_tpu:
            hidden_states = _get_compiled_bucket_fn(self, num_chunks)(x, cu_seqlens)
            # Flush the compiled bucket asynchronously so multi-pass uploads
            # overlap with execution without blocking the host thread.
            synchronize_tensors(hidden_states, wait=False)
        else:
            hidden_states = _run_static_bucket_pass(self, x, cu_seqlens)

        for span_start, span_end in valid_spans:
            outputs.append(hidden_states[span_start:span_end])

    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)


setattr(_padded_audio_forward, _AUDIO_CHUNK_LENGTHS_PATCHED, True)


def _patch_audio_encoder(encoder_cls: type[Any]) -> bool:
    """Install static-bucketed ``forward`` on ``Qwen3OmniMoeAudioEncoder``."""
    orig_forward = encoder_cls.forward
    if getattr(orig_forward, _AUDIO_CHUNK_LENGTHS_PATCHED, False):
        return False
    encoder_cls._tpu_orig_forward = orig_forward
    encoder_cls.forward = _padded_audio_forward
    return True


def maybe_patch_qwen3_omni_thinker(
    architectures: list[str] | None = None,
) -> bool:
    """Apply TPU patches when ``Qwen3OmniMoeForConditionalGeneration`` is loaded."""
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
