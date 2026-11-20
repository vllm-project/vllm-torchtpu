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
    if (
        envs.TPU_DCP_LSE_MERGE_A2A_FUSED_LSE
        and output.dtype.itemsize in (2, 4)
        and _lse_lanes(local, output.dtype) <= vdim
    ):
        o_recv, l_recv = _all_to_all_with_lse(o_send, l_send, group)
    else:
        o_recv = _all_to_all(o_send, group)
        l_recv = _all_to_all(l_send, group)
    return _merge_received(o_recv, l_recv).to(output.dtype)


def _lse_lanes(local: int, dtype: torch.dtype) -> int:
    """Lanes of `dtype` that carry `local` f32 lse values bit-exactly: two
    16-bit halves each for a 2-byte dtype, one element each for 4 bytes."""
    return 2 * local if dtype.itemsize == 2 else local


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

    Payload layout: `[world, T, local + 1, V]`. Rows `[0, local)` of the
    second-minor dimension are the partials of token `t` for rank `d`,
    unchanged. Row `local` is one extra `V`-lane row that carries the token's
    lse values; its unused lanes are zero. The payload keeps the partials'
    own layout:

    - The minor dimension stays `V`, and no reshape merges `local` and `V`
      into one dimension. On TPU, a merged minor dimension (`local * V` or
      larger) gave the all-to-all operand a different tiling from the
      partials, and XLA inserted a relayout copy of the whole payload before
      or after the exchange.
    - The partials and the lse row are joined by a concatenation on dim -2,
      and split again by slices on dim -2. Both are tile-aligned in the minor
      dimension, so XLA can build the payload in one fusion and the consumer
      fusion can read the received slices in place.

    For 2-byte partials, each f32 lse is sent as two 16-bit halves: lanes
    `[0, local)` of the lse row hold the high halves and lanes
    `[local, 2 * local)` the low halves. The halves are computed with
    integer shifts and masks, and each half is reinterpreted as one element
    of the partials' dtype. A reinterpretation between two dtypes of equal
    width keeps the shape. Viewing f32 as pairs of 16-bit values would add a
    minor dimension of size 2, which the TPU tiling pads to 128 lanes and
    relayouts. For 4-byte partials, each lse is reinterpreted as one element.
    The caller must ensure that the lse lanes fit in one row, that is
    `_lse_lanes(local, dtype) <= V`.

    Example: world 16, T = 16, 4 local heads, V = 512, bf16 partials. The
    payload is bf16[16, 16, 5, 512]. Row 4 of token `t` holds the high halves
    of the token's 4 lse values in lanes 0-3, the low halves in lanes 4-7,
    and zeros in lanes 8-511. The TPU tiling of a 2-byte array pads the
    second-minor dimension to 16 rows, so the padded payload has the same
    size in memory as the padded partials alone.

    Args:
      o_send: `[world, T, local, V]` partials in a 2- or 4-byte dtype. Block
        `d` of the leading axis goes to rank `d`.
      l_send: `[world, T, local]` f32 log-sum-exps, blocked like `o_send`.
      group: the DCP group.

    Returns:
      `(o_recv, l_recv)`: `[world, T, local, V]` in `o_send.dtype` and
      `[world, T, local]` f32. Block `r` of each came from rank `r`.
    """
    local, vdim = o_send.shape[-2:]
    dtype = o_send.dtype
    num_l = _lse_lanes(local, dtype)
    assert num_l <= vdim, (local, vdim, dtype)
    bits = l_send.contiguous().view(torch.int32)
    if dtype.itemsize == 2:
        # Each half is a value in [-32768, 32767], so the int16 conversion
        # is exact and does not depend on how a backend wraps overflow.
        hi = torch.bitwise_right_shift(bits, 16)
        lo = (bits & 0xFFFF) - ((bits & 0x8000) << 1)
        l_vals = torch.cat([hi, lo], dim=-1).to(torch.int16).view(dtype)
    else:
        l_vals = bits.view(dtype)
    l_row = torch.nn.functional.pad(l_vals, (0, vdim - num_l)).unsqueeze(-2)

    payload = torch.cat([o_send, l_row], dim=-2)
    recv = _all_to_all(payload, group)

    o_recv = recv[..., :local, :]
    l_recv = recv[..., local, :]
    if dtype.itemsize == 2:
        hi_r = l_recv[..., :local].view(torch.int16).to(torch.int32)
        lo_r = l_recv[..., local : 2 * local].view(torch.int16).to(torch.int32) & 0xFFFF
        l_bits = torch.bitwise_left_shift(hi_r, 16) | lo_r
    else:
        l_bits = l_recv[..., :local].view(torch.int32)
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
