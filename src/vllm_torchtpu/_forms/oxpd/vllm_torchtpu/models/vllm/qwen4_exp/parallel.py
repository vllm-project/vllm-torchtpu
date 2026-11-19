# SPDX-License-Identifier: Apache-2.0
"""PLE embedding ownership and physical DP slots; groups are borrowed."""

import torch


class PLEParallel:
    """ETP=TP, or ETP=TP*DP using the existing two-dimensional groups.

    No group is created, cached globally or destroyed here. Request history
    and ordinary projections remain local regardless of table ownership.
    """

    def __init__(self, tp_group, dp_group, *, embedding_across_dp: bool):
        self.across_dp = embedding_across_dp and dp_group.world_size > 1
        self.tp_group = tp_group
        self.dp_group = dp_group
        self.group = dp_group if self.across_dp else tp_group
        self.rank = tp_group.rank_in_group
        self.world_size = tp_group.world_size
        if self.across_dp:
            self.rank += dp_group.rank_in_group * tp_group.world_size
            self.world_size *= dp_group.world_size

    def gather_ids(self, ids):
        if self.across_dp:
            return self.dp_group.all_gather(ids, dim=0)
        return ids

    def combine(self, raw_embeddings, local_tokens):
        # Raw lookup emits 16 rows of 160 INT8/INT16 elements per token.
        # Each INT32 word stays within one row, hence has exactly one owner:
        # integer SUM preserves its bits and sends the same number of bytes.
        # The mixed TP/DP path needs word carriers and an independent local
        # slice: compiled narrow-type reductions or an aliased second in-place
        # reduction corrupt small/odd buckets on TPU.
        packed = self.across_dp and self.tp_group.world_size > 1
        carrier = raw_embeddings.dtype
        result = self.group.all_reduce(
            raw_embeddings.view(torch.int32) if packed else raw_embeddings
        )
        if self.across_dp:
            # Counts describe physical buckets, never cumulative live tokens.
            result = result.narrow(
                0, self.dp_group.rank_in_group * local_tokens, local_tokens
            )
            # DP sums cover one TP column of owners. Complete the owner sum
            # across TP after selecting local rows, keeping its payload small.
            if packed:
                result = result.clone()
            # TP1 all_reduce is a no-op.
            result = self.tp_group.all_reduce(result)
        return result.view(carrier) if packed else result
