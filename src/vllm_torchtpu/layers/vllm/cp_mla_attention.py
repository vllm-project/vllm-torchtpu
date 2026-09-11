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

The MLA counterpart of `cp_attention.py`, which does the same job for the GQA
batched-RPA path. Process-group access lives in `distributed/dcp.py`; this is
the attention-side orchestration.

Nothing here touches JAX or Pallas: every collective runs on tensors already
in torch's hands, outside the exported op. That is the point of the file. The
merge could in principle live inside the JAX op, but the torch->JAX export
path constrains how collectives inside `pallas.jax_op` are launched across
worker processes, and the merge needs none of that.

Each DCP rank stores a shard of the KV cache, so it can only produce a
*partial* attention result. Bracketing the kernel call:

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

    Half of the DCP contract, and the half that is easy to forget: DCP does not
    add a parallel axis on top of TP, it *re-spends* `dcp_size` of TP's ways on
    positions instead of heads. So the head sharding coarsens from `tp` to
    `tp // dcp`, and a rank that keeps attending with only its own
    `num_heads // tp` heads is computing a different quantity from its peers --
    which the log-sum-exp merge would then combine as if they were the same
    head. The result is not approximate, it is meaningless.

    Upstream does exactly this (`MLAAttention.forward` ->
    `get_dcp_group().all_gather(mqa_q, dim=1)`), and pairs it with the head
    reduce-scatter in `merge_lse_partials_scatter_heads`.

    Cheap in the direction it matters: `[num_tokens, num_heads, dim]` of query,
    against the `[num_tokens, num_heads, v_dim]` of output the scatter takes
    back. Replicated queries are also what lets the sparse-MLA DCP op declare
    `q` replicated and shard nothing but the cache.

    Args:
      q: `[num_tokens, num_heads, dim]`, this rank's head slice.

    Returns:
      `[num_tokens, num_heads * dcp_size, dim]`, identical on every rank, with
      rank `r`'s heads in block `r`. Returned unchanged without a DCP group.
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
    does not depend on the order the ranks are combined in -- the partial state
    `(m, l, l*o)` is a commutative monoid -- which is exactly why DCP needs one
    reduction rather than a ring: rotating queries around 16 ranks would move
    ~3 MB per token, and this moves ~64 KB.

    Two collectives, deliberately lopsided:

      1. all-gather the log-sum-exps. They are `[num_tokens, num_heads]` f32 --
         256 B per token per rank at 64 heads -- so gathering them outright is
         cheap, and it leaves *every* rank holding every `lse_r`.
      2. reduce-scatter the weighted outputs, `[num_tokens, num_heads, v_dim]`,
         the one big tensor.

    The point of (1) is that it makes the softmax denominator a local
    computation: with all the `lse_r` in hand each rank derives the shared
    maximum and the full sum itself, so there is no second reduction over the
    large tensor and no numerically fragile `exp` of an unshifted logit. It is
    also what leaves (2) a plain sum, which is the only reduction a collective
    can do.

    (2) is the counterpart to `all_gather_heads`, and upstream's
    `cp_lse_ag_out_rs`. Every rank attended with the whole group's heads, so
    the merged output has `dcp_size`x the heads `o_proj` (and `W_UV`) can
    consume; the reduce-scatter does the merge's second collective and the
    narrowing in one step, at half the bytes of an all-reduce and landing
    exactly the shape the caller wants. The head blocks line up with
    `all_gather_heads` by construction: it put rank `r`'s heads in block `r`,
    and the scatter hands block `r` back to rank `r`.

    Args:
      output: `[num_tokens, num_heads * dcp_size, v_dim]`, this rank's partial.
      lse: `[num_tokens, num_heads * dcp_size]`, the matching log-sum-exps.
        Entries are `-inf` for tokens this rank owns none of the top-k for;
        such a rank was still fed a dummy key (the kernel requires
        `kv_len >= 1`) and its output must not be weighted in.

    Returns:
      `[num_tokens, num_heads, v_dim]` in `output.dtype` -- this rank's head
      slice of the whole-context result. Tokens no rank contributed to --
      padding rows -- come back zeroed rather than NaN.
    """
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return output

    weighted = _weight_by_lse(output, lse, group)
    return group.reduce_scatter(weighted, dim=1).to(output.dtype)


def _weight_by_lse(output: torch.Tensor, lse: torch.Tensor,
                   group: Any) -> torch.Tensor:
    """This rank's `softmax_r(lse) * o_r` term, in f32. See the caller."""
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
