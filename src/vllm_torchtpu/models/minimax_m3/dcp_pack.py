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
"""Fewer, packed DCP collectives for the MSA decode step.

A DCP step of the MSA op runs four small collectives per layer: the query-head
all-gather, the candidate all-gather, the LSE all-gather and the output
reduce-scatter. At decode sizes each one costs its fixed latency (~2-3 us on a
DCP 2 pair) and runs on the critical path, so two packed collectives replace
the four:

* `gather_queries_and_candidates`: one all-gather moves this rank's query heads
  and its packed top-k candidates (`msa_dcp.pack_candidates`) together.
* `merge_head_partials_all_to_all`: one all-to-all sends every rank the partial
  outputs and LSEs of the heads it keeps; the rank then merges them itself.

Both pack their operands as int32 bit patterns, so the collectives move bits
and never see a float (a TPU collective may flush f32 subnormals, which the
bf16 pairs and the candidate ids are as f32).

The merge reproduces `ops.msa.merge_head_partials` bit for bit on a DCP 2
pair: the receiving rank computes the same per-source weights from the same
LSEs, rounds each weighted partial as the reduce-scatter's operand is rounded,
and the sum of two terms does not depend on their order.
"""

import torch


def _bits(x: torch.Tensor) -> torch.Tensor:
    """``x`` (bf16 or f32, last dim even for bf16) as int32 words."""
    return x.contiguous().view(torch.int32)


def gather_queries_and_candidates(
    q: torch.Tensor, candidates: torch.Tensor, group
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(q_all [T, dcp * Hq, D], candidates_all [dcp, T, C])`` from one
    all-gather of ``q [T, Hq, D]`` (bf16) and ``candidates [T, C]`` (f32):
    the same values as ``group.all_gather(q, dim=1)`` and
    ``group.all_gather(candidates.unsqueeze(0), dim=0)``."""
    num_tokens, num_heads, head_dim = q.shape
    q_words = num_heads * head_dim // 2
    payload = torch.cat(
        [_bits(q).reshape(num_tokens, q_words), _bits(candidates)], dim=1
    )
    gathered = group.all_gather(payload.unsqueeze(0), dim=0)  # [dcp, T, W]
    dcp = gathered.shape[0]
    q_all = (
        gathered[:, :, :q_words]
        .contiguous()
        .view(q.dtype)
        .view(dcp, num_tokens, num_heads, head_dim)
        .transpose(0, 1)
        .reshape(num_tokens, dcp * num_heads, head_dim)
    )
    cand_all = gathered[:, :, q_words:].contiguous().view(candidates.dtype)
    return q_all, cand_all


def merge_received_partials(
    out_parts: torch.Tensor,
    lse_parts: torch.Tensor,
    out_dtype: torch.dtype,
    round_operands: bool,
    fence=None,
) -> torch.Tensor:
    """``[R, H, T, D]`` partials and ``[R, H, T]`` f32 LSEs of the same heads
    from every rank ``r`` -> the merged ``[H, T, D]`` in `out_dtype`.

    Per source the weight and the weighted partial are computed as
    `ops.msa.merge_head_partials` computes them on that source; with
    `round_operands` (``TPU_DCP_RS_OUT_PRECISION``) each weighted partial is
    rounded to `out_dtype` before the sum, as the reduce-scatter operand is;
    `fence` (an XLA optimization barrier) keeps that rounding.
    """
    total = torch.logsumexp(lse_parts, dim=0)
    weight = torch.exp(lse_parts - total.unsqueeze(0)).unsqueeze(-1)
    # where, not *: a zero-weight partial may be NaN (NaN * 0 = NaN).
    weighted = out_parts.to(torch.float32) * weight
    weighted = torch.where(weight > 0, weighted, torch.zeros_like(weighted))
    if round_operands:
        weighted = weighted.to(out_dtype)
        # The reduce-scatter receives these rounded values; XLA's excess
        # precision would otherwise drop the rounding inside one fusion.
        if fence is not None:
            weighted = fence(weighted)
    return weighted.sum(dim=0).to(out_dtype)


def pack_partials(out: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    """``out [H, T, D]`` (bf16) and ``lse [H, T]`` as int32 rows
    ``[H, T * D / 2 + T]``: head ``h``'s row carries its partial, then its LSEs."""
    num_heads, num_tokens, head_dim = out.shape
    return torch.cat(
        [
            _bits(out).reshape(num_heads, num_tokens * head_dim // 2),
            _bits(lse.to(torch.float32)),
        ],
        dim=1,
    )


def unpack_partials(
    recv: torch.Tensor, dcp: int, num_tokens: int, head_dim: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """`pack_partials` rows of ``dcp`` sources, ``[dcp * h, W]`` -> ``(out
    [dcp, h, T, D], lse [dcp, h, T])``."""
    words = num_tokens * head_dim // 2
    rows = recv.view(dcp, -1, recv.shape[-1])
    local = rows.shape[1]
    out = (
        rows[:, :, :words]
        .contiguous()
        .view(dtype)
        .view(dcp, local, num_tokens, head_dim)
    )
    lse = rows[:, :, words:].contiguous().view(torch.float32)
    return out, lse


def merge_head_partials_all_to_all(
    out: torch.Tensor,
    lse: torch.Tensor,
    group,
    all_to_all,
    round_operands: bool,
    fence=None,
) -> torch.Tensor:
    """`ops.msa.merge_head_partials` before its unrotation and transpose:
    head-major partials ``out [H, T, D]``, ``lse [H, T]`` -> this rank's
    merged heads ``[H / dcp, T, D]`` in ``out.dtype``, via one `all_to_all`
    (block ``d`` of the leading axis goes to rank ``d``)."""
    dcp = int(group.world_size)
    num_heads, num_tokens, head_dim = out.shape
    recv = all_to_all(pack_partials(out, lse), group)
    out_parts, lse_parts = unpack_partials(
        recv, dcp, num_tokens, head_dim, out.dtype
    )
    return merge_received_partials(
        out_parts, lse_parts, out.dtype, round_operands, fence
    )
