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
"""DCP (Decode Context Parallelism) attention for MLA: the head gather and
the log-sum-exp merge that bracket the sharded attention call.

The MLA counterpart of `cp_attention.py`.

Every collective runs on tensors already in torch's hands, outside
the exported op.

Each DCP rank stores a shard of the KV cache, so it can only produce a
*partial* attention result.

  1. `all_gather_heads` / `all_gather_q_nope_pe_heads` widens this rank's
     query head slice to the whole group's heads using head-major `dim=0`
     collectives so `DeviceCommunicatorBase.all_gather` performs zero
     post-collective `movedim(0, 1)` layout copies.
  2. the kernel runs against this rank's shard alone, returning a partial
     output and its log-sum-exp.
  3. `merge_lse_partials_scatter_heads` folds the partials into the exact
     whole-context result in head-major `[H, T, D]` layout and scatters along
     `dim=0`, eliminating `DeviceCommunicatorBase.reduce_scatter`'s pre- and
     post-collective `movedim(0, 1).contiguous()` copies (`OFFLOAD_DATA_FORMATTING`).
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.dcp import get_dcp_group


def all_gather_heads(q: torch.Tensor) -> torch.Tensor:
    """Widen a rank's query head slice to the whole DCP group's heads.

    DCP doesn't add a new parallel dimension—it borrows from TP. It
    reallocates dcp ways from heads to sequence positions, reducing
    head sharding from tp to tp // dcp.

    Runs the underlying collective along `dim=0` on head-major
    `[tp_local_num_heads, num_tokens, dim]` so `DeviceCommunicatorBase.all_gather`
    returns a zero-copy view of `dist.all_gather_into_tensor` without an extra
    `movedim(0, 1).reshape(...)` copy. When `q` comes from `torch.bmm(q_nope_t, w_uk_t)`
    transposed from `[H_local, T, D]`, `q.transpose(0, 1).contiguous()` is also a
    zero-copy no-op.

    Args:
      q: `[num_tokens, tp_local_num_heads, dim]`, this rank's head slice.

    Returns:
      `[num_tokens, tp_local_num_heads * dcp_size, dim]`, identical
      on every rank. Returned unchanged without a DCP group.
    """
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return q
    q_hm = q.transpose(0, 1).contiguous()
    gathered_hm = group.all_gather(q_hm, dim=0)
    return gathered_hm.transpose(0, 1).contiguous()


def all_gather_q_nope_pe_heads(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_lora_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused head-major `dim=0` all-gather of `(ql_nope, q_pe)` across DCP ranks.

    Eliminates the intermediate contiguous `q_dcp` materialization inside
    `DeviceCommunicatorBase.all_gather(..., dim=1)` by:
      1. Concatenating `(ql_nope, q_pe)` directly in head-major
         `[H_local, T, D_nope + D_pe]`.
      2. Gathering along `dim=0` so `DeviceCommunicatorBase.all_gather` returns a
         zero-copy view of the collective output buffer (`[H_dcp, T, D_nope + D_pe]`).
      3. Slicing `ql_nope_dcp` and `q_pe_dcp` and transposing to `[T, H_dcp, D]`
         in a single materialization pass.
    """
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return ql_nope, q_pe
    q_cat_hm = torch.cat([ql_nope.transpose(0, 1), q_pe.transpose(0, 1)], dim=-1)
    q_dcp_hm = group.all_gather(q_cat_hm, dim=0)
    ql_nope_dcp = q_dcp_hm[..., :kv_lora_rank].transpose(0, 1).contiguous()
    q_pe_dcp = q_dcp_hm[..., kv_lora_rank:].transpose(0, 1).contiguous()
    return ql_nope_dcp, q_pe_dcp


def merge_lse_partials_scatter_heads(
    output: torch.Tensor,
    lse: torch.Tensor,
    *,
    reduce_scatter_bf16: bool | None = None,
) -> torch.Tensor:
    """Combine this rank's partial attention output with its peers'.

    With per-rank normalized outputs `o_r` and `lse_r = m_r + log(l_r)`, the
    exact whole-context result is `o = sum_r softmax_r(lse) * o_r`. Correctness
    does not depend on the order the ranks are combined in.

    Two collectives:

      1. all-gather the log-sum-exps. They are `[num_tokens, num_heads]` f32 --
         Leaves every rank holding every `lse_r`.
      2. reduce-scatter the weighted outputs in head-major `[num_heads, num_tokens, D]`
         along `dim=0`, so `DeviceCommunicatorBase.reduce_scatter`'s pre- and
         post-collective `movedim(0, dim).contiguous()` calls are zero-copy no-ops,
         and the returned `.transpose(0, 1)` view unwraps back to contiguous
         `[tp_local_num_heads, num_tokens, kv_lora_rank]` for the downstream
         `torch.bmm(outputs_t, w_uv)` projection.
         Under `reduce_scatter_bf16=True` (b/568555797), the weighted outputs
         are reduced in `output.dtype` (bfloat16) rather than f32, halving the
         reduce-scatter payload for a 2x collective speedup.

    Args:
      output: `[num_tokens, tp_local_num_heads * dcp_size, kv_lora_rank]`,
        this rank's partial.
      lse: `[num_tokens, tp_local_num_heads * dcp_size]`, the matching log-sum-exps.
        Entries are `-inf` for tokens this rank owns none of the top-k for;
        such a rank was still fed a dummy key (the kernel requires
        `kv_len >= 1`) and its output must not be weighted in.
      reduce_scatter_bf16: Whether to reduce-scatter in output.dtype (bf16) rather
        than f32. Defaults to envs.TPU_DCP_REDUCE_SCATTER_BF16 (True).

    Returns:
      `[num_tokens, tp_local_num_heads, kv_lora_rank]` in `output.dtype` --
      this rank's head slice of the whole-context result (backed by contiguous
      head-major `[tp_local_num_heads, num_tokens, kv_lora_rank]` storage).
    """
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return output

    if reduce_scatter_bf16 is None:
        reduce_scatter_bf16 = getattr(envs, "TPU_DCP_REDUCE_SCATTER_BF16", True)

    weighted_hm = _weight_by_lse(
        output, lse, group, reduce_scatter_bf16=reduce_scatter_bf16
    )
    scattered_hm = group.reduce_scatter(weighted_hm, dim=0).to(output.dtype)
    return scattered_hm.transpose(0, 1)


def _weight_by_lse(
    output: torch.Tensor,
    lse: torch.Tensor,
    group: Any,
    *,
    reduce_scatter_bf16: bool = False,
) -> torch.Tensor:
    """This rank's `softmax_r(lse) * o_r` term, materialized once in head-major
    `[num_heads, num_tokens, kv_lora_rank]` for zero-`movedim` `reduce_scatter(dim=0)`.
    Under `reduce_scatter_bf16=True`, returns `output.dtype` directly (b/568555797).
    """
    lse_f32 = (
        lse
        if (lse.dtype == torch.float32 and lse.is_contiguous())
        else lse.to(torch.float32).contiguous()
    )
    # [world_size, num_tokens, num_heads]: small, and it removes the need for
    # any further reduction over the denominator.
    gathered = group.all_gather(lse_f32.unsqueeze(0), dim=0)
    if not isinstance(gathered, torch.Tensor):
        gathered = torch.cat(list(gathered), dim=0)

    finite = torch.isfinite(gathered)
    shift = torch.where(finite, gathered, -float("inf")).amax(dim=0)
    # A token with no contributor on any rank would give inf - inf; pin the
    # shift to 0 there and let the zero denominator carry the result to zero.
    empty = ~torch.isfinite(shift)
    shift = torch.where(empty, 0.0, shift)

    weights = torch.where(finite, torch.exp(gathered - shift.unsqueeze(0)), 0.0)
    denom = weights.sum(dim=0)
    safe_denom = torch.where(denom > 0, denom, 1.0)

    # This rank's row of `gathered` is `lse_f32` itself. Weighting from the
    # local copy instead of indexing by rank keeps the rank out of the traced
    # program, so every DCP rank compiles the same executable.
    mine = (
        torch.where(
            torch.isfinite(lse_f32),
            torch.exp(lse_f32 - shift),
            0.0,
        )
        / safe_denom
    )
    # Transpose bf16 `output` and f32 `mine` as zero-copy views before upcasting
    # so the f32 product is materialized once in head-major `[H, T, D]` layout.
    output_hm = output.transpose(0, 1)
    mine_hm = mine.transpose(0, 1).unsqueeze(-1)
    weighted = output_hm.to(torch.float32) * mine_hm
    if reduce_scatter_bf16:
        return weighted.to(output.dtype).contiguous()
    return weighted.contiguous()


def is_pcp_dcp_mixed_mode(parallel_config: Any) -> bool:
    """Return whether both PCP and DCP are enabled concurrently."""
    pcp_size = getattr(parallel_config, "prefill_context_parallel_size", 1)
    dcp_size = getattr(parallel_config, "decode_context_parallel_size", 1)
    return (
        isinstance(pcp_size, int)
        and pcp_size > 1
        and isinstance(dcp_size, int)
        and dcp_size > 1
    )


def validate_pcp_dcp_mixed_mode(parallel_config: Any) -> dict[str, Any]:
    """Validate and return execution parameters for PCP+DCP mixed mode."""
    from vllm_torchtpu.kernels.mla import kv_cache_utils

    pcp_size = int(getattr(parallel_config, "prefill_context_parallel_size", 1) or 1)
    dcp_size = int(getattr(parallel_config, "decode_context_parallel_size", 1) or 1)
    interleave_size = int(
        getattr(parallel_config, "cp_kv_cache_interleave_size", 1) or 1
    )

    if pcp_size > 1 and dcp_size > 1:
        if interleave_size % kv_cache_utils.WORD_BYTES != 0:
            raise ValueError(
                "PCP+DCP mixed mode requires cp_kv_cache_interleave_size to be a "
                f"multiple of {kv_cache_utils.WORD_BYTES} "
                f"(kv_cache_utils.WORD_BYTES), got {interleave_size}."
            )

    return {
        "is_mixed_mode": pcp_size > 1 and dcp_size > 1,
        "pcp_size": pcp_size,
        "dcp_size": dcp_size,
        "interleave_size": interleave_size,
    }


def get_mixed_mode_mesh_routing(
    parallel_config: Any,
    *,
    is_prefill: bool,
    default_mesh: Any = None,
) -> tuple[str, Any]:
    """Route mixed mode execution to the appropriate communication mesh.

    Prefill queries route to PCP sequence parallel mesh; decode KV cache
    routes to DCP KV cache mesh.
    """
    from vllm_torchtpu.distributed.dcp import get_or_create_dcp_mesh
    from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.vllm_adapter import (
        get_pcp_streaming_mesh,
    )

    if not is_pcp_dcp_mixed_mode(parallel_config):
        return ("default", default_mesh)

    if is_prefill:
        try:
            return ("pcp", get_pcp_streaming_mesh())
        except Exception:
            return ("pcp_fallback", default_mesh)
    else:
        return ("dcp", get_or_create_dcp_mesh())
