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

    rs_dtype = output.dtype if envs.TPU_DCP_RS_OUT_PRECISION and is_backbone else None
    return cp_lse_ag_out_rs(output, lse, group, rs_dtype=rs_dtype)


def _all_to_all(x: torch.Tensor, group: Any) -> torch.Tensor:
    """Block `d` of `x`'s leading axis goes to rank `d`; block `r` of the
    result came from rank `r`."""
    from torch.distributed import _functional_collectives as funcol

    out = funcol.all_to_all_single(x.contiguous(), None, None, group.device_group)
    if isinstance(out, funcol.AsyncCollectiveTensor):
        return funcol.wait_tensor(out)
    return out


def merge_lse_partials_all_to_all(
    output: torch.Tensor, lse: torch.Tensor, group: Any
) -> torch.Tensor:
    """Same result as the reduce-scatter merge, via all-to-all.

    Each rank sends head block `d` of its partial and lse to rank `d`, then
    merges its own heads locally in f32. Faster for small token counts.

    With TPU_DCP_LSE_MERGE_A2A_FUSED_LSE set, the partials and the lse travel
    in one all-to-all (`_all_to_all_with_lse`) instead of two.
    """
    world = int(group.world_size)
    num_tokens, num_heads, vdim = output.shape
    local = num_heads // world
    # Destination-major: [world, T, local, V] and [world, T, local].
    o_send = output.reshape(num_tokens, world, local, vdim).transpose(0, 1)
    l_send = lse.to(torch.float32).reshape(num_tokens, world, local).transpose(0, 1)
    if envs.TPU_DCP_LSE_MERGE_A2A_FUSED_LSE and output.dtype.itemsize in (2, 4):
        o_recv, l_recv = _all_to_all_with_lse(o_send, l_send, group)
    else:
        o_recv = _all_to_all(o_send, group)
        l_recv = _all_to_all(l_send, group)
    return _merge_received(o_recv, l_recv).to(output.dtype)


def _all_to_all_with_lse(
    o_send: torch.Tensor, l_send: torch.Tensor, group: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exchange the partials and their f32 lse in a single all-to-all.

    A collective has a fixed cost that does not depend on its size. At decode
    sizes the lse block is only a few KB, so a separate all-to-all for it
    costs about as much as its fixed cost. This function instead stores the
    f32 lse bits in elements of the partials' dtype and appends them to the
    partials of the same destination and token. After the exchange it splits
    the two parts and rebuilds the f32 lse from the same bits, so the
    received values are bit-identical to those of the two-collective path.

    Payload layout: `[world, T, local * V + L]`. Row `(d, t)` holds the
    `local * V` partial values of token `t` for rank `d`, then an lse block of
    `L` lanes, where `L` is the lse width rounded up to a multiple of 128.
    The layout follows the TPU memory tiling of a 2-byte array, which is 16
    rows by 128 lanes:

    - The minor dimension `local * V + L` is a multiple of 128 when
      `local * V` is, and the partials end on a 128-lane boundary. The
      split after the exchange is then a tile-aligned slice that a consumer
      fusion reads in place.
    - The second-minor dimension is the token count, the same as in the
      separate partials exchange.
    - A flat `[world, T * local * V + 2 * T * local]` row is avoided. XLA
      splits a minor dimension that large (32,896 at the sizes below) into
      `[2, 16448]`. 16,448 is not a multiple of 128, so XLA inserts a
      relayout copy of the whole payload after the exchange.

    For 2-byte partials, each f32 lse is sent as two 16-bit halves: lanes
    `[0, local)` of the block hold the high halves and lanes
    `[local, 2 * local)` the low halves. The halves are computed with
    integer shifts and masks, and each half is reinterpreted as one element
    of the partials' dtype. A reinterpretation between two dtypes of equal
    width keeps the shape. Viewing f32 as pairs of 16-bit values would add a
    minor dimension of size 2, which the TPU tiling pads to 128 lanes and
    relayouts. For 4-byte partials, each lse is reinterpreted as one element.
    The unused lanes of the block are zero.

    Example: world 16, T = 16, 4 local heads, V = 512, bf16 partials. Each
    row holds 4 * 512 = 2,048 partial values, then a block of 128 lanes whose
    first 8 lanes are the high and low halves of the token's 4 lse values.
    The all-to-all moves bf16[16, 16, 2176], 6.25% more bytes than the
    partials alone.

    Args:
      o_send: `[world, T, local, V]` partials in a 2- or 4-byte dtype. Block
        `d` of the leading axis goes to rank `d`.
      l_send: `[world, T, local]` f32 log-sum-exps, blocked like `o_send`.
      group: the DCP group.

    Returns:
      `(o_recv, l_recv)`: `[world, T, local, V]` in `o_send.dtype` and
      `[world, T, local]` f32. Block `r` of each came from rank `r`.
    """
    world, num_tokens, local, vdim = o_send.shape
    dtype = o_send.dtype
    bits = l_send.contiguous().view(torch.int32)
    if dtype.itemsize == 2:
        # Each half is a value in [-32768, 32767], so the int16 conversion
        # is exact and does not depend on how a backend wraps overflow.
        hi = torch.bitwise_right_shift(bits, 16)
        lo = (bits & 0xFFFF) - ((bits & 0x8000) << 1)
        l_vals = torch.cat([hi, lo], dim=-1).to(torch.int16).view(dtype)
    else:
        l_vals = bits.view(dtype)
    num_l = l_vals.shape[-1]
    l_lanes = -(-num_l // 128) * 128
    l_block = torch.nn.functional.pad(l_vals, (0, l_lanes - num_l))

    num_o = local * vdim
    payload = torch.cat(
        [o_send.reshape(world, num_tokens, num_o), l_block], dim=-1
    )
    recv = _all_to_all(payload, group)

    o_recv = recv[..., :num_o].reshape(world, num_tokens, local, vdim)
    if dtype.itemsize == 2:
        hi_r = recv[..., num_o : num_o + local].view(torch.int16).to(torch.int32)
        lo_r = (
            recv[..., num_o + local : num_o + 2 * local]
            .view(torch.int16)
            .to(torch.int32)
            & 0xFFFF
        )
        l_bits = torch.bitwise_left_shift(hi_r, 16) | lo_r
    else:
        l_bits = recv[..., num_o : num_o + local].view(torch.int32)
    return o_recv, l_bits.view(torch.float32)


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
