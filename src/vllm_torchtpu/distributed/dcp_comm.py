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
"""DCP merge of partial attention results (GQA and MLA)."""

from __future__ import annotations

from typing import Any, Literal, overload

import torch

from vllm_torchtpu import envs


def dcp_lse_ag_and_weight(
    output: torch.Tensor,
    lse: torch.Tensor,
    group: Any,
    rs_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """All-gather LSE across DCP group, compute logsumexp, and weight output."""
    lse_f32 = lse.to(torch.float32)
    lses = group.all_gather(lse_f32.unsqueeze(0).contiguous(), dim=0)
    merged_lse = torch.logsumexp(lses, dim=0)

    # where, not *: a zero-weight partial may be NaN (NaN * 0 = NaN).
    weight = torch.exp(lse_f32 - merged_lse).unsqueeze(-1)
    weighted = output.to(torch.float32) * weight
    weighted = torch.where(weight > 0, weighted, torch.zeros_like(weighted))
    if rs_dtype is not None:
        return weighted.to(rs_dtype)
    if envs.TPU_DCP_RS_BF16 or envs.TPU_DCP_RS_OUT_PRECISION:
        return weighted.to(output.dtype)
    return weighted


def dcp_reduce_scatter(
    weighted: torch.Tensor,
    out_dtype: torch.dtype,
    group: Any,
) -> torch.Tensor:
    """Reduce-scatter weighted partial attention output across DCP group."""
    merged = group.reduce_scatter(weighted.contiguous(), dim=1)
    return merged.to(out_dtype) if merged.dtype != out_dtype else merged


@overload
def cp_lse_ag_out_rs(
    output: torch.Tensor,
    lse: torch.Tensor,
    group: Any | None,
    return_lse: Literal[False] = False,
    rs_dtype: torch.dtype | None = None,
) -> torch.Tensor: ...


@overload
def cp_lse_ag_out_rs(
    output: torch.Tensor,
    lse: torch.Tensor,
    group: Any | None,
    return_lse: Literal[True],
    rs_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]: ...


def cp_lse_ag_out_rs(
    output: torch.Tensor,
    lse: torch.Tensor,
    group: Any | None,
    return_lse: bool = False,
    rs_dtype: torch.dtype | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Mirror vllm.v1.attention.ops.dcp.cp_lse_ag_out_rs.

    Args:
      output: [T, H * group, D]
      lse: [T, H * group]
      return_lse: also return the merged LSE.
      rs_dtype: dtype the weighted terms cross the reduce-scatter in; None
        keeps f32.

    Returns:
      merged_output [T, H, D] in output.dtype, plus merged_lse [T, H] in f32
      if return_lse. Computed in f32.

    Note: we do not implement Helix-style `dcp_a2a_lse_reduce` merge (one
      a2a of output + LSE, then a local combine), as we observe no benefit
      over AG-RS on TPU topologies.
    """
    if group is None or int(group.world_size) == 1:
        return (output, lse) if return_lse else output

    if not return_lse:
        weighted = dcp_lse_ag_and_weight(output, lse, group, rs_dtype=rs_dtype)
        return dcp_reduce_scatter(weighted, output.dtype, group)

    lse_f32 = lse.to(torch.float32)
    lses = group.all_gather(lse_f32.unsqueeze(0).contiguous(), dim=0)
    merged_lse = torch.logsumexp(lses, dim=0)

    # where, not *: a zero-weight partial may be NaN (NaN * 0 = NaN).
    weight = torch.exp(lse_f32 - merged_lse).unsqueeze(-1)
    weighted = output.to(torch.float32) * weight
    weighted = torch.where(weight > 0, weighted, torch.zeros_like(weighted))
    if rs_dtype is not None:
        weighted = weighted.to(rs_dtype)
    elif envs.TPU_DCP_RS_BF16 or envs.TPU_DCP_RS_OUT_PRECISION:
        weighted = weighted.to(output.dtype)

    merged = dcp_reduce_scatter(weighted, output.dtype, group)
    heads = merged.shape[1]
    start = int(group.rank_in_group) * heads
    return merged, merged_lse[:, start : start + heads]
