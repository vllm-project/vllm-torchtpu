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

Both move int16 bit patterns, so the collectives never see a float (a TPU
collective may flush f32 subnormals, which candidate ids are as f32). The bf16
operand keeps its layout; the small f32 operands ride as extra sublane rows
(`_rows`), since splitting a bf16 tensor into int32 pairs costs a relayout.

The merge reproduces `ops.msa.merge_head_partials` bit for bit on a DCP 2
pair: the receiving rank computes the same per-source weights from the same
LSEs, rounds each weighted partial as the reduce-scatter's operand is rounded,
and the sum of two terms does not depend on their order.
"""

import torch


def _rows(x: torch.Tensor, lanes: int) -> torch.Tensor:
    """``x [N, M]`` (any 4-byte dtype) as int16 rows ``[N, ceil(2M / lanes), lanes]``,
    zero padded: small operands ride as extra sublane rows of a bf16 payload,
    so the payload never changes its lane layout."""
    words = x.contiguous().view(torch.int16)
    n, m = words.shape
    rows = -(-m // lanes)
    # Always pad (possibly by 0): a Python branch on the token count here
    # splits the backbone's bucket groups and forces a re-trace.
    words = torch.nn.functional.pad(words, (0, rows * lanes - m))
    return words.view(n, rows, lanes)


def _unrows(x: torch.Tensor, count: int, dtype: torch.dtype) -> torch.Tensor:
    """Inverse of `_rows` for ``[..., rows, lanes]``: the first ``count``
    values of `dtype` (4-byte)."""
    flat = x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])
    return flat[..., : 2 * count].contiguous().view(dtype)


def gather_queries_and_candidates(
    q: torch.Tensor, candidates: torch.Tensor, group
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(q_all [T, dcp * Hq, D], candidates_all [dcp, T, C])`` from one
    all-gather of ``q [T, Hq, D]`` (bf16) and ``candidates [T, C]`` (f32):
    the same values as ``group.all_gather(q, dim=1)`` and
    ``group.all_gather(candidates.unsqueeze(0), dim=0)``."""
    num_tokens, num_heads, head_dim = q.shape
    cols = candidates.shape[1]
    payload = torch.cat(
        [q.contiguous().view(torch.int16), _rows(candidates, head_dim)], dim=1
    )
    gathered = group.all_gather(payload.unsqueeze(0), dim=0)  # [dcp, T, Hq + r, D]
    dcp = gathered.shape[0]
    q_all = (
        gathered[:, :, :num_heads]
        .transpose(0, 1)
        .reshape(num_tokens, dcp * num_heads, head_dim)
        .view(q.dtype)
    )
    cand_all = _unrows(gathered[:, :, num_heads:], cols, candidates.dtype)
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
    """``out [H, T, D]`` (bf16) and ``lse [H, T]`` as int16 ``[H, T + r, D]``:
    head ``h``'s partial rows, then its f32 LSEs as ``r`` extra rows."""
    head_dim = out.shape[2]
    return torch.cat(
        [out.contiguous().view(torch.int16), _rows(lse.to(torch.float32), head_dim)],
        dim=1,
    )


def unpack_partials(
    recv: torch.Tensor, dcp: int, num_tokens: int, head_dim: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """`pack_partials` payloads of ``dcp`` sources, ``[dcp * h, T + r, D]`` ->
    ``(out [dcp, h, T, D], lse [dcp, h, T])``."""
    rows = recv.view(dcp, -1, recv.shape[-2], head_dim)
    out = rows[:, :, :num_tokens].view(dtype)
    lse = _unrows(rows[:, :, num_tokens:], num_tokens, torch.float32)
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
