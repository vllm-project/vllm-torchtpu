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

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas


def _jax_dcp_rescale_and_transpose(
    output: jax.Array,
    lse: jax.Array,
    merged_lse: jax.Array,
) -> jax.Array:
    """Fused rescale, NaN-safe masking, downcast, and transpose for DCP merge."""
    weight = jnp.exp(lse - merged_lse)[..., None]
    weighted = output.astype(jnp.float32) * weight
    weighted = jnp.where(weight > 0, weighted, 0.0).astype(output.dtype)
    return jnp.swapaxes(weighted, 0, 1)


_dcp_rescale_and_transpose_op = pallas.jax_op(
    "pallas::dcp_rescale_and_transpose",
    _jax_dcp_rescale_and_transpose,
)


def _fake_dcp_rescale_and_transpose(
    output: torch.Tensor,
    lse: torch.Tensor,
    merged_lse: torch.Tensor,
) -> torch.Tensor:
    num_tokens, num_heads, head_dim = output.shape
    return torch.empty(
        (num_heads, num_tokens, head_dim),
        dtype=output.dtype,
        device=output.device,
    )


_dcp_rescale_and_transpose_op.register_fake(_fake_dcp_rescale_and_transpose)


def _jax_dcp_rescale_and_transpose_f32(
    output: jax.Array,
    lse: jax.Array,
    merged_lse: jax.Array,
) -> jax.Array:
    """Fused rescale, NaN-safe masking, and transpose in float32 for DCP merge."""
    weight = jnp.exp(lse - merged_lse)[..., None]
    weighted = output.astype(jnp.float32) * weight
    weighted = jnp.where(weight > 0, weighted, 0.0)
    return jnp.swapaxes(weighted, 0, 1)


_dcp_rescale_and_transpose_f32_op = pallas.jax_op(
    "pallas::dcp_rescale_and_transpose_f32",
    _jax_dcp_rescale_and_transpose_f32,
)


def _fake_dcp_rescale_and_transpose_f32(
    output: torch.Tensor,
    lse: torch.Tensor,
    merged_lse: torch.Tensor,
) -> torch.Tensor:
    num_tokens, num_heads, head_dim = output.shape
    return torch.empty(
        (num_heads, num_tokens, head_dim),
        dtype=torch.float32,
        device=output.device,
    )


_dcp_rescale_and_transpose_f32_op.register_fake(
    _fake_dcp_rescale_and_transpose_f32
)


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
        # Fused JAX XLA op computes weights, NaN-masks, downcasts, and transposes
        # to [H * group, T, D] in a single pass to eliminate intermediate HBM
        # writes and relayout overhead. Reduce-scatter on dim=0 scatters the
        # leading head dimension directly.
        if rs_dtype is not None:
            weighted = _dcp_rescale_and_transpose_op(output, lse, merged_lse)
            if weighted.dtype != rs_dtype:
                weighted = weighted.to(rs_dtype)
        else:
            weighted = _dcp_rescale_and_transpose_f32_op(output, lse, merged_lse)
        merged = (
            group.reduce_scatter(weighted.contiguous(), dim=0)
            .to(output.dtype)
            .transpose(0, 1)
            .contiguous()
        )
    else:
        # CPU / non-TPU fallback
        # where, not *: a zero-weight partial may be NaN (NaN * 0 = NaN).
        weight = torch.exp(lse - merged_lse).unsqueeze(-1)
        weighted = output.to(torch.float32) * weight
        weighted = torch.where(weight > 0, weighted, torch.zeros_like(weighted))

        if rs_dtype is not None:
            weighted = weighted.to(rs_dtype)
        merged = group.reduce_scatter(weighted.contiguous(), dim=1).to(output.dtype)

    if not return_lse:
        return merged

    heads = merged.shape[1]
    start = int(group.rank_in_group) * heads
    return merged, merged_lse[:, start : start + heads]

