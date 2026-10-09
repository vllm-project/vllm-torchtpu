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

  1. `all_gather_heads` widens this rank's query head slice to the whole
     group's heads using head-major `dim=0` collectives so
     `DeviceCommunicatorBase.all_gather` performs zero post-collective
     `movedim(0, 1)` layout copies.
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
from vllm_torchtpu.distributed.dcp_comm import cp_lse_ag_out_rs


def all_gather_heads(q: torch.Tensor) -> torch.Tensor:
    """Widen a rank's query head slice to the whole DCP group's heads.

    DCP doesn't add a new parallel dimension—it borrows from TP. It
    reallocates dcp ways from heads to sequence positions, reducing
    head sharding from tp to tp // dcp.

    When `TPU_DCP_HEAD_MAJOR_COLLECTIVES` is enabled (default), runs the
    underlying collective along `dim=0` on head-major
    `[tp_local_num_heads, num_tokens, dim]` so `DeviceCommunicatorBase.all_gather`
    returns a zero-copy view of `dist.all_gather_into_tensor` without an extra
    `movedim(0, 1).reshape(...)` copy:
      - When `q` is already a `.transpose(0, 1)` view of contiguous head-major
        `[H_local, T, D]` (`ql_nope` from `torch.bmm(q_nope_t, w_uk_t)`),
        `q.transpose(0, 1).contiguous()` is a zero-copy view unwrap (`data_ptr()`
        unchanged), reducing total copies from 2 to 1.
      - When `q` is contiguous token-major (`q_pe`), transposing before/after
        trades a compact local `[H_local, T, D]` pack for eliminating the
        `dcp_size`-wider post-gather `movedim(0, 1).reshape(...)` copy inside
        `DeviceCommunicatorBase.all_gather`.

    Args:
      q: `[num_tokens, tp_local_num_heads, dim]`, this rank's head slice.

    Returns:
      `[num_tokens, tp_local_num_heads * dcp_size, dim]`, identical
      on every rank. Returned unchanged without a DCP group.
    """
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return q
    if not getattr(envs, "TPU_DCP_HEAD_MAJOR_COLLECTIVES", True):
        return group.all_gather(q.contiguous(), dim=1)
    q_hm = q.transpose(0, 1).contiguous()
    gathered_hm = group.all_gather(q_hm, dim=0)
    return gathered_hm.transpose(0, 1).contiguous()


def all_gather_q_nope_pe_heads(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_lora_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused all-gather of `(ql_nope, q_pe)` across DCP ranks in a single call."""
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return ql_nope, q_pe
    q_dcp = group.all_gather(torch.cat([ql_nope, q_pe], dim=-1), dim=1)
    return (
        q_dcp[..., :kv_lora_rank].contiguous(),
        q_dcp[..., kv_lora_rank:].contiguous(),
    )


def merge_lse_partials_scatter_heads(
    output: torch.Tensor, lse: torch.Tensor, is_backbone: bool = True
) -> torch.Tensor:
    """Combine this rank's partial attention output with its peers'.

    With per-rank normalized outputs `o_r` and `lse_r = m_r + log(l_r)`, the
    exact whole-context result is `o = sum_r softmax_r(lse) * o_r`. Correctness
    does not depend on the order the ranks are combined in.

    Two collectives:

      1. all-gather the log-sum-exps. They are `[num_tokens, num_heads]` f32 --
         Leaves every rank holding every `lse_r`.
      2. reduce-scatter the weighted outputs

    With TPU_DCP_LSE_MERGE_A2A set, the backbone's steps of at most
    TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS tokens instead take
    `merge_lse_partials_all_to_all`, which gives the same result.

    Args:
      output: `[num_tokens, tp_local_num_heads * dcp_size, kv_lora_rank]`,
        this rank's partial.
      lse: `[num_tokens, tp_local_num_heads * dcp_size]`, the matching log-sum-exps.
        Entries are `-inf` for tokens this rank owns none of the top-k for;
        such a rank was still fed a dummy key (the kernel requires
        `kv_len >= 1`) and its output must not be weighted in.
      is_backbone: True if the caller is the backbone, the only model that is
        traced separately per token bucket. Only then may the merge compare
        the token count to TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS, since that
        comparison is fixed at trace time. Pass False from the MTP drafter,
        which is traced once for every bucket: it always takes the
        reduce-scatter, exactly as with TPU_DCP_LSE_MERGE_A2A unset.

    Returns:
      `[num_tokens, tp_local_num_heads, kv_lora_rank]` in `output.dtype` --
      this rank's head slice of the whole-context result.
    """
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return output

    # This `if` runs while tracing, so its answer is baked into the graph.
    # Only the backbone is traced once per bucket group, so only it can pick
    # by size: the all-to-all wins at decode sizes, the reduce-scatter above
    # because it sums in the network. The MTP drafter has one graph for every
    # bucket; comparing its token count would pin that graph to the first
    # size traced. `is_backbone` is checked before the token count, so the
    # drafter never compares it.
    if (
        envs.TPU_DCP_LSE_MERGE_A2A
        and is_backbone
        and output.shape[0] <= envs.TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS
    ):
        return merge_lse_partials_all_to_all(output, lse, group)

    if not getattr(envs, "TPU_DCP_HEAD_MAJOR_COLLECTIVES", True):
        return cp_lse_ag_out_rs(output, lse, group)

    # Materialize weighted output directly in head-major `[H, T, D]` f32 and
    # reduce-scatter along `dim=0` so `DeviceCommunicatorBase.reduce_scatter`'s
    # pre- and post-collective `movedim(0, dim).contiguous()` are zero-copy
    # no-ops (`dim == 0`), and downstream `outputs.reshape(...).transpose(0, 1)`
    # before `torch.bmm(outputs_t, w_uv)` unwraps back to contiguous `[H_local, T, D]`.
    weighted_hm = _weight_by_lse(output, lse, group)
    scattered_hm = group.reduce_scatter(weighted_hm, dim=0).to(output.dtype)
    return scattered_hm.transpose(0, 1)


def _weight_by_lse(
    output: torch.Tensor,
    lse: torch.Tensor,
    group: Any,
    head_major: bool | None = None,
) -> torch.Tensor:
    """This rank's `softmax_r(lse) * o_r` term, materialized once in head-major
    `f32[num_heads, num_tokens, kv_lora_rank]` for zero-`movedim`
    `reduce_scatter(dim=0)` when `TPU_DCP_HEAD_MAJOR_COLLECTIVES` is enabled.
    """
    if head_major is None:
        head_major = bool(getattr(envs, "TPU_DCP_HEAD_MAJOR_COLLECTIVES", True))
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
    if not head_major:
        return (output.to(torch.float32) * mine.unsqueeze(-1)).contiguous()
    # Transpose bf16 `output` and f32 `mine` as zero-copy views before upcasting
    # so the f32 product is materialized once in head-major `[H, T, D]` layout.
    output_hm = output.transpose(0, 1)
    mine_hm = mine.transpose(0, 1).unsqueeze(-1)
    return (output_hm.to(torch.float32) * mine_hm).contiguous()


def _all_to_all(x: torch.Tensor, group: Any) -> torch.Tensor:
    """Block `d` of `x`'s leading axis goes to rank `d`; block `r` of the
    result came from rank `r`."""
    from torch.distributed import _functional_collectives as funcol

    out = funcol.all_to_all_single(x.contiguous(), None, None, group.device_group)
    return funcol.wait_tensor(out)


def merge_lse_partials_all_to_all(
    output: torch.Tensor, lse: torch.Tensor, group: Any
) -> torch.Tensor:
    """Same result as the reduce-scatter merge, via all-to-all.

    Each rank sends head block `d` of its partial and lse to rank `d`, then
    merges its own heads locally in f32. Faster for small token counts.
    """
    world = int(group.world_size)
    num_tokens, num_heads, vdim = output.shape
    local = num_heads // world
    # Destination-major: [world, T, local, V] and [world, T, local].
    o_send = output.reshape(num_tokens, world, local, vdim).transpose(0, 1)
    l_send = lse.to(torch.float32).reshape(num_tokens, world, local).transpose(0, 1)
    o_recv = _all_to_all(o_send, group)
    l_recv = _all_to_all(l_send, group)
    return _merge_received(o_recv, l_recv).to(output.dtype)


def _merge_received(o_recv: torch.Tensor, l_recv: torch.Tensor) -> torch.Tensor:
    """`[R, T, h, V]` partials and `[R, T, h]` lse -> the f32 `[T, h, V]` merge."""
    finite = torch.isfinite(l_recv)
    shift = torch.where(finite, l_recv, torch.full_like(l_recv, float("-inf"))).amax(
        dim=0
    )
    shift = torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift))
    weights = torch.where(
        finite, torch.exp(l_recv - shift.unsqueeze(0)), torch.zeros_like(l_recv)
    )
    denom = weights.sum(dim=0)
    weights = weights / torch.where(denom > 0, denom, torch.ones_like(denom))
    # The compiler lowers multiply-then-sum to a dot that rounds f32 operands to
    # bf16. Splitting the weights into a bf16 part and a residual keeps it exact.
    o_f32 = o_recv.to(torch.float32)
    w_hi = weights.to(torch.bfloat16).to(torch.float32)
    w_lo = weights - w_hi
    return (o_f32 * w_hi.unsqueeze(-1)).sum(dim=0) + (o_f32 * w_lo.unsqueeze(-1)).sum(
        dim=0
    )
