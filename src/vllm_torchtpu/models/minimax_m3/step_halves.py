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
"""Per-half attention metadata of a step (StepHalves in attention.py)."""

import dataclasses

import torch


def split_metadata[M](md: M, split: int) -> tuple[M, M, M]:
    """`md` (an AttentionMetadata: seq_lens, query_start_loc and init=False CPU
    copies) for the queries before token `split`, for the rest, and for the
    whole step (`md` itself). A request straddling the split keeps its pages:
    the first half sees its tokens before the split, its length shortened by
    the rest; the second half sees the remaining tokens at the full length. A
    request without tokens in a half has length 0 there; request_distribution
    is passed through, so the MSA kernels must skip requests without queries
    (the DCP one-KV-head path does: msa_dcp_heads masks q_lens > 0; check the
    others before widening StepHalves' guard in model.py).

    The symbolic `split` enters one op only (clamp(max=)): metadata computed
    as `clamp(min=split) - split` reached the attention kernels wrong in the
    served program although its values read back right (r14 arm 3)."""
    seq_lens = md.seq_lens.view(-1)
    query_start_loc = md.query_start_loc.view(-1)
    first_qsl = query_start_loc.clamp(max=split)
    second_qsl = query_start_loc - first_qsl
    first_q = first_qsl[1:] - first_qsl[:-1]
    tail = second_qsl[1:] - second_qsl[:-1]
    first_lens = (first_q > 0) * (seq_lens - tail)
    second_lens = (tail > 0) * seq_lens
    out = []
    for qsl, lens in ((first_qsl, first_lens), (second_qsl, second_lens)):
        part = dataclasses.replace(
            md,
            seq_lens=lens.to(seq_lens.dtype),
            query_start_loc=qsl.to(query_start_loc.dtype),
        )
        part.query_start_loc_cpu = None
        part.seq_lens_cpu = None
        out.append(part)
    return out[0], out[1], md
