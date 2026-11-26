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

from typing import Any, Literal, overload

import torch
from torch_tpu._internal import pallas as _ttpu_pallas

from vllm_torchtpu.kernels.lse_rescale import (
    lse_rescale_head_major_bf16,
    lse_rescale_head_major_f32,
)
from vllm_torchtpu.models.vllm.glm_dsa_dispatch import glm_dsa_v7x_active


def _register_lse_rescale(name: str, fn: Any, dtype: torch.dtype) -> Any:
    op: Any = _ttpu_pallas.jax_op(name, fn)
    op.register_fake(
        lambda out, _lse, _m: out.new_empty(
            (out.shape[1], out.shape[0], out.shape[2]), dtype=dtype
        )
    )
    return op


_lse_rescale_head_major_f32 = _register_lse_rescale(
    "pallas::lse_rescale_head_major_f32", lse_rescale_head_major_f32, torch.float32
)
_lse_rescale_head_major_bf16 = _register_lse_rescale(
    "pallas::lse_rescale_head_major_bf16", lse_rescale_head_major_bf16, torch.bfloat16
)


def lse_rescale_head_major_op(dtype: torch.dtype | None) -> Any | None:
    eff = dtype if dtype is not None else torch.float32
    if eff == torch.float32:
        return _lse_rescale_head_major_f32
    if eff == torch.bfloat16:
        return _lse_rescale_head_major_bf16
    return None


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

    rescale_op = lse_rescale_head_major_op(rs_dtype)
    if output.device.type == "tpu" and rescale_op is not None and glm_dsa_v7x_active():
        weighted = rescale_op(output, lse, merged_lse)
        merged = group.reduce_scatter(weighted, dim=0).transpose(0, 1).contiguous()
    else:
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
