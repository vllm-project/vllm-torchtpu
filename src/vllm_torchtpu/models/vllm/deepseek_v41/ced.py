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
"""Host planning for Context-Encoder-Decoder (CED) bounded SWA replay."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

_EMPTY_I32 = torch.empty((0,), dtype=torch.int32)
_EMPTY_I64 = torch.empty((0,), dtype=torch.int64)


@dataclass(frozen=True)
class DecoderReplayPlan:
    indices: tuple[int, ...]
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    request_distribution: torch.Tensor
    gather_indices: torch.Tensor
    scatter_indices: torch.Tensor
    padded_num_tokens: int
    passthrough: bool = False


def plan_decoder_replay(
    metadata: Any,
    window: int,
    num_tokens: int,
    compile_sizes: Sequence[int],
    *,
    dp_ced_decoder_bucket: int | None = None,
) -> DecoderReplayPlan:
    """Plan the decoder-stage replay that keeps only the prompt-tail SWA window.

    Returns passthrough=True when the decoder can run on the full encoder output.
    """
    bounds = getattr(metadata, "query_start_loc_cpu", None)
    lengths = getattr(metadata, "seq_lens_cpu", None)
    prompts = getattr(metadata, "prompt_lens_cpu", None)
    seq_lens_dev = getattr(metadata, "seq_lens", None)
    max_num_seqs = (
        int(seq_lens_dev.shape[0])
        if seq_lens_dev is not None
        else (len(lengths) if lengths is not None else 1)
    )
    force_ced = dp_ced_decoder_bucket is not None and dp_ced_decoder_bucket < 0
    if force_ced:
        dp_ced_decoder_bucket = -dp_ced_decoder_bucket
    if bounds is None or lengths is None or prompts is None:
        if not force_ced and (
            dp_ced_decoder_bucket is None or dp_ced_decoder_bucket == num_tokens
        ):
            return DecoderReplayPlan(
                indices=(),
                query_start_loc=_EMPTY_I32,
                seq_lens=_EMPTY_I32,
                request_distribution=_EMPTY_I32,
                gather_indices=_EMPTY_I64,
                scatter_indices=_EMPTY_I64,
                padded_num_tokens=num_tokens,
                passthrough=True,
            )
        if dp_ced_decoder_bucket <= 0:
            return DecoderReplayPlan(
                indices=(),
                query_start_loc=torch.zeros((max_num_seqs + 1,), dtype=torch.int32),
                seq_lens=torch.zeros((max_num_seqs,), dtype=torch.int32),
                request_distribution=torch.zeros((3,), dtype=torch.int32),
                gather_indices=_EMPTY_I64,
                scatter_indices=torch.full((num_tokens,), -1, dtype=torch.int64),
                padded_num_tokens=0,
                passthrough=False,
            )
        return DecoderReplayPlan(
            indices=(),
            query_start_loc=torch.zeros((max_num_seqs + 1,), dtype=torch.int32),
            seq_lens=torch.zeros((max_num_seqs,), dtype=torch.int32),
            request_distribution=torch.zeros((3,), dtype=torch.int32),
            gather_indices=torch.zeros((dp_ced_decoder_bucket,), dtype=torch.int64),
            scatter_indices=torch.full((num_tokens,), -1, dtype=torch.int64),
            padded_num_tokens=dp_ced_decoder_bucket,
            passthrough=False,
        )

    if window < 1 or len(bounds) != len(lengths) + 1 or len(prompts) != len(lengths):
        raise ValueError("Invalid decoder replay scheduling metadata")
    if bounds[0] != 0 or bounds[-1] > num_tokens:
        raise ValueError("Query boundaries exceed the input token bucket")

    indices: list[int] = []
    output_bounds: list[int] = [0]
    dec_seq_lens_list: list[int] = []
    decode_end = 0
    all_decode_so_far = True
    active_seq_end = 0

    total_sched = bounds[-1]
    num_active = len(lengths)
    while (
        num_active > 0
        and bounds[num_active - 1] == total_sched
        and lengths[num_active - 1] == 0
        and prompts[num_active - 1] == 0
    ):
        num_active -= 1

    for req_idx, (start, end, length, prompt) in enumerate(
        zip(
            bounds[:num_active],
            bounds[1 : num_active + 1],
            lengths[:num_active],
            prompts[:num_active],
        )
    ):
        count = end - start
        if count < 0 or length < count or prompt < 0:
            raise ValueError("Invalid scheduled query or prompt length")
        replay_start = max(0, prompt - window)
        skip = min(count, max(0, replay_start - (length - count)))
        kept_i = count - skip
        indices.extend(range(start + skip, end))
        output_bounds.append(len(indices))
        dec_seq_lens_list.append(length if kept_i > 0 else 0)
        if kept_i > 0:
            active_seq_end = req_idx + 1
            if all_decode_so_far and count == 1 and kept_i == 1 and length > prompt:
                decode_end = req_idx + 1
            else:
                all_decode_so_far = False
        elif count > 0:
            all_decode_so_far = False

    kept = len(indices)
    if dp_ced_decoder_bucket is not None:
        padded = dp_ced_decoder_bucket
    else:
        buckets = sorted({num_tokens, *(int(n) for n in compile_sizes if int(n) > 0)})
        padded = next(n for n in buckets if n >= kept) if kept else 0

    passthrough = bool(
        not force_ced and kept > 0 and kept == bounds[-1] and padded == num_tokens
    )
    if passthrough:
        return DecoderReplayPlan(
            indices=tuple(indices),
            query_start_loc=_EMPTY_I32,
            seq_lens=_EMPTY_I32,
            request_distribution=_EMPTY_I32,
            gather_indices=_EMPTY_I64,
            scatter_indices=_EMPTY_I64,
            padded_num_tokens=padded,
            passthrough=True,
        )

    q_loc_tensor = torch.zeros((max_num_seqs + 1,), dtype=torch.int32)
    n_b = min(len(output_bounds), max_num_seqs + 1)
    q_loc_tensor[:n_b] = torch.tensor(output_bounds[:n_b], dtype=torch.int32)
    if n_b < max_num_seqs + 1:
        q_loc_tensor[n_b:] = kept

    seq_lens_tensor = torch.zeros((max_num_seqs,), dtype=torch.int32)
    n_s = min(len(dec_seq_lens_list), max_num_seqs)
    seq_lens_tensor[:n_s] = torch.tensor(dec_seq_lens_list[:n_s], dtype=torch.int32)

    dist_tensor = torch.tensor(
        [decode_end, decode_end, active_seq_end], dtype=torch.int32
    )

    gather_tensor = torch.zeros((max(padded, 0),), dtype=torch.int64)
    scatter_tensor = torch.full((num_tokens,), -1, dtype=torch.int64)
    if kept > 0 and padded >= kept:
        idx_t = torch.tensor(indices, dtype=torch.int64)
        # Spread padded rows across kept indices instead of repeating idx 0
        gather_tensor[:] = idx_t[torch.arange(padded, dtype=torch.int64) % kept]
        scatter_tensor[idx_t] = torch.arange(kept, dtype=torch.int64)

    return DecoderReplayPlan(
        indices=tuple(indices),
        query_start_loc=q_loc_tensor,
        seq_lens=seq_lens_tensor,
        request_distribution=dist_tensor,
        gather_indices=gather_tensor,
        scatter_indices=scatter_tensor,
        padded_num_tokens=padded,
        passthrough=False,
    )
