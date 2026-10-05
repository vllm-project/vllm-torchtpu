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

from typing import Any

import torch


def cp_lse_ag_out_rs(
    output: torch.Tensor,
    lse: torch.Tensor,
    group: Any | None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Mirror vllm.v1.attention.ops.dcp.cp_lse_ag_out_rs.

    Args:
      output: [T, H * group, D]
      lse: [T, H * group]
      return_lse: also return the merged LSE.

    Returns:
      merged_output [T, H, D] in output.dtype, plus merged_lse [T, H] in f32
      if return_lse. Computed in f32.

    Note: we do not implement Helix-style `dcp_a2a_lse_reduce` merge (one
      a2a of output + LSE, then a local combine), as we observe no benefit
      over AG-RS on TPU topologies.
    """
    if group is None or int(group.world_size) == 1:
        return (output, lse) if return_lse else output

    # TODO(weiyulin): Evaluate if we need LSE and reduce to be fp32.
    lse = lse.to(torch.float32)
    lses = group.all_gather(lse.unsqueeze(0).contiguous(), dim=0)
    merged_lse = torch.logsumexp(lses, dim=0)

    # where, not *: a zero-weight partial may be NaN (NaN * 0 = NaN).
    weight = torch.exp(lse - merged_lse).unsqueeze(-1)
    weighted = output.to(torch.float32) * weight
    weighted = torch.where(weight > 0, weighted, torch.zeros_like(weighted))

    merged = group.reduce_scatter(weighted.contiguous(), dim=1).to(output.dtype)
    if not return_lse:
        return merged

    heads = merged.shape[1]
    start = int(group.rank_in_group) * heads
    return merged, merged_lse[:, start : start + heads]
