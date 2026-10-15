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
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.lse_rescale import lse_rescale_head_major


def _fake_lse_rescale_head_major(
    output: torch.Tensor,
    lse: torch.Tensor,
    merged_lse: torch.Tensor,
    out_dtype: str,
) -> torch.Tensor:
    tokens, heads, dim = output.shape
    return output.new_empty((heads, tokens, dim), dtype=getattr(torch, out_dtype))


# Registered at import: Dynamo cannot trace a lock-guarded registration.
_lse_rescale_head_major = pallas.jax_op(
    "pallas::lse_rescale_head_major", lse_rescale_head_major
)
_lse_rescale_head_major.register_fake(_fake_lse_rescale_head_major)


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

    # TODO(weiyulin): Evaluate if we need LSE and reduce to be fp32.
    lse = lse.to(torch.float32)
    lses = group.all_gather(lse.unsqueeze(0).contiguous(), dim=0)
    merged_lse = torch.logsumexp(lses, dim=0)

    if output.device.type == "tpu":
        # Head-major, so the reduce-scatter's leading axis is the heads and
        # it reads the weighted partial without a transposing copy.
        wire_dtype = str(rs_dtype or torch.float32).removeprefix("torch.")
        weighted = _lse_rescale_head_major(output, lse, merged_lse, wire_dtype)
        merged = group.reduce_scatter(weighted, dim=0).transpose(0, 1).contiguous()
    else:
        # where, not *: a zero-weight partial may be NaN (NaN * 0 = NaN).
        weight = torch.exp(lse - merged_lse).unsqueeze(-1)
        weighted = output.to(torch.float32) * weight
        weighted = torch.where(weight > 0, weighted, torch.zeros_like(weighted))
        if rs_dtype is not None:
            weighted = weighted.to(rs_dtype)
        merged = group.reduce_scatter(weighted.contiguous(), dim=1)
    merged = merged.to(output.dtype)
    if not return_lse:
        return merged

    heads = merged.shape[1]
    start = int(group.rank_in_group) * heads
    return merged, merged_lse[:, start : start + heads]
