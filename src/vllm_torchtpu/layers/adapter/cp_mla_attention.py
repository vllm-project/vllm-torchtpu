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

from vllm_torchtpu.distributed.dcp import get_dcp_group


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


def merge_lse_partials_scatter_heads(output: torch.Tensor,
                                     lse: torch.Tensor) -> torch.Tensor:
    """Combine this rank's partial attention output with its peers'.

    With per-rank normalized outputs `o_r` and `lse_r = m_r + log(l_r)`, the
    exact whole-context result is `o = sum_r softmax_r(lse) * o_r`. Correctness
    does not depend on the order the ranks are combined in.

    Two collectives:

      1. all-gather the log-sum-exps. They are `[num_tokens, num_heads]` f32 --
         Leaves every rank holding every `lse_r`.
      2. reduce-scatter the weighted outputs

    Args:
      output: `[num_tokens, tp_local_num_heads * dcp_size, kv_lora_rank]`, this rank's partial.
      lse: `[num_tokens, tp_local_num_heads * dcp_size]`, the matching log-sum-exps.
        Entries are `-inf` for tokens this rank owns none of the top-k for;
        such a rank was still fed a dummy key (the kernel requires
        `kv_len >= 1`) and its output must not be weighted in.

    Returns:
      `[num_tokens, tp_local_num_heads, kv_lora_rank]` in `output.dtype` -- this rank's head
      slice of the whole-context result.
    """
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return output

    weighted = _weight_by_lse(output, lse, group)
    return group.reduce_scatter(weighted, dim=1).to(output.dtype)


def _weight_by_lse(output: torch.Tensor, lse: torch.Tensor,
                   group: Any) -> torch.Tensor:
    """This rank's `softmax_r(lse) * o_r` term, in f32."""
    lse_f32 = lse.to(torch.float32).contiguous()
    # [world_size, num_tokens, num_heads]: small, and it removes the need for
    # any further reduction over the denominator.
    gathered = group.all_gather(lse_f32.unsqueeze(0), dim=0)
    if not isinstance(gathered, torch.Tensor):
        gathered = torch.cat(list(gathered), dim=0)

    finite = torch.isfinite(gathered)
    shift = torch.where(finite, gathered,
                        torch.full_like(gathered, float("-inf"))).amax(dim=0)
    # A token with no contributor on any rank would give inf - inf; pin the
    # shift to 0 there and let the zero denominator carry the result to zero.
    empty = ~torch.isfinite(shift)
    shift = torch.where(empty, torch.zeros_like(shift), shift)

    weights = torch.where(finite, torch.exp(gathered - shift.unsqueeze(0)),
                          torch.zeros_like(gathered))
    denom = weights.sum(dim=0)
    safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))

    mine = weights[int(group.rank_in_group)] / safe_denom
    return (output.to(torch.float32) * mine.unsqueeze(-1)).contiguous()
