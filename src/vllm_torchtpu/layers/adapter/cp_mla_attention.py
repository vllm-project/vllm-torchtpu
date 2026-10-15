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
     group's heads, so every rank's heads are attended against every rank's
     KV shard.
  2. the kernel runs against this rank's shard alone, returning a partial
     output and its log-sum-exp.
  3. `merge_lse_partials_scatter_heads` folds the partials into the exact
     whole-context result and narrows the head axis back to this rank's own.
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

    Args:
      q: `[num_tokens, tp_local_num_heads, dim]`, this rank's head slice.

    Returns:
      `[num_tokens, tp_local_num_heads * dcp_size, dim]`, identical
      on every rank. Returned unchanged without a DCP group.
    """
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return q
    return group.all_gather(q.contiguous(), dim=1)


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


def project_uv_heads(
    partial: torch.Tensor, w_uv: torch.Tensor, w_uv_scale: torch.Tensor | None
) -> torch.Tensor:
    """`[T, N, kv_lora_rank]` partial -> f32 `[T, N, v_head_dim]`, head n by W_UV[n].

    W_UV is linear and the merge weights are per (token, head), so projecting
    each partial before the merge gives the same result as projecting after.
    """
    # f32 operands make bmm return f32 (out_dtype is ignored on TPU). bf16/fp8
    # convert exactly, so XLA still runs one bf16 MXU pass.
    out = torch.bmm(partial.transpose(0, 1).to(torch.float32), w_uv.to(torch.float32))
    if w_uv_scale is not None:
        out = out * w_uv_scale.to(torch.float32)
    return out.transpose(0, 1)


def merge_lse_partials_scatter_heads(
    output: torch.Tensor,
    lse: torch.Tensor,
    is_backbone: bool = True,
    wire_dtype: torch.dtype | None = None,
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
    # A W_UV-projected f32 partial still crosses in the attention dtype.
    narrow = envs.TPU_DCP_RS_OUT_PRECISION and is_backbone
    if (
        envs.TPU_DCP_LSE_MERGE_A2A
        and is_backbone
        and output.shape[0] <= envs.TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS
    ):
        return merge_lse_partials_all_to_all(
            output, lse, group, send_dtype=wire_dtype if narrow else None
        )

    rs_dtype = (wire_dtype or output.dtype) if narrow else None
    return cp_lse_ag_out_rs(output, lse, group, rs_dtype=rs_dtype)


def _all_to_all(x: torch.Tensor, group: Any) -> torch.Tensor:
    """Block `d` of `x`'s leading axis goes to rank `d`; block `r` of the
    result came from rank `r`."""
    from torch.distributed import _functional_collectives as funcol

    out = funcol.all_to_all_single(x.contiguous(), None, None, group.device_group)
    return funcol.wait_tensor(out)


def merge_lse_partials_all_to_all(
    output: torch.Tensor,
    lse: torch.Tensor,
    group: Any,
    send_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Same result as the reduce-scatter merge, via all-to-all.

    Each rank sends head block `d` of its partial and lse to rank `d`, then
    merges its own heads locally in f32. Faster for small token counts.
    `send_dtype` is the dtype the partial crosses in; None keeps its own.
    """
    world = int(group.world_size)
    num_tokens, num_heads, vdim = output.shape
    local = num_heads // world
    # Destination-major: [world, T, local, V] and [world, T, local].
    o_send = output.reshape(num_tokens, world, local, vdim).transpose(0, 1)
    if send_dtype is not None:
        o_send = o_send.to(send_dtype)
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
