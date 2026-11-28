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

# Provenance: copied from src/vllm_torchtpu/models/vllm/qwen4_exp/parallel.py at
# 764b70e2a9cd69bcb5cbc9822d0936771148ef44.
#   Owned by models/qwen4_exp from this copy on.
"""PLE embedding ownership; the TP group is borrowed."""


class PLEParallel:
    """The PLE table is split by rows across the TP ranks (ETP=TP).

    No group is created, cached globally or destroyed here. Request history
    and ordinary projections remain local regardless of table ownership.
    """

    def __init__(self, tp_group):
        self.tp_group = tp_group
        self.group = tp_group
        self.rank = tp_group.rank_in_group
        self.world_size = tp_group.world_size

    def combine(self, raw_embeddings):
        # Each raw row has exactly one owning TP rank; the others contribute
        # zeros, so the SUM reassembles the rows. TP1 all_reduce is a no-op.
        return self.group.all_reduce(raw_embeddings)
