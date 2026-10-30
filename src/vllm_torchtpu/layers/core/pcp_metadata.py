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
"""`PcpMetadata`: the PCP manager's per-step attention metadata.

`PcpShardingPlan.build_metadata` (`runner/pcp/sharding_plan.py`) builds it;
attention layers read it from `AttentionMetadata.pcp_metadata`.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import torch


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "split_block_tables",
        "split_seq_lens",
        "split_query_start_loc",
        "hidden_restore_idx",
        "rep_row_idx",
        "rep_block_tables",
        "rep_seq_lens",
        "rep_query_start_loc",
        "rep_request_distribution",
        "ws_req",
        "ws_vpage",
        "ws_page_offset",
        "num_split_reqs",
        "ws_used_pages",
    ],
    meta_fields=[],
)
@dataclass
class PcpMetadata:
    """Attention metadata a `PcpShardingPlan` builds (`build_metadata`).

    Every sharding plan fills only its own fields and leaves the rest None.
    """

    # --- HeadTailPcpShardingPlan, KV-gather (sparse MLA) ---
    # The split prefills, request-major: block tables, seq_lens,
    # query_start_loc, and each split token's PCP-gathered row.
    split_block_tables: jax.Array | torch.Tensor | None = None
    split_seq_lens: jax.Array | torch.Tensor | None = None
    split_query_start_loc: jax.Array | torch.Tensor | None = None
    hidden_restore_idx: jax.Array | torch.Tensor | None = None
    # The replicated decodes as their own request window: local rows, block
    # tables, seq_lens, query_start_loc, request_distribution.
    rep_row_idx: jax.Array | torch.Tensor | None = None
    rep_block_tables: jax.Array | torch.Tensor | None = None
    rep_seq_lens: jax.Array | torch.Tensor | None = None
    rep_query_start_loc: jax.Array | torch.Tensor | None = None
    rep_request_distribution: jax.Array | torch.Tensor | None = None
    # KV workspace: request and virtual page of each workspace page, first
    # workspace page of each local split segment, split request count (1,),
    # pages in use (1,).
    ws_req: jax.Array | torch.Tensor | None = None
    ws_vpage: jax.Array | torch.Tensor | None = None
    ws_page_offset: jax.Array | torch.Tensor | None = None
    num_split_reqs: jax.Array | torch.Tensor | None = None
    ws_used_pages: jax.Array | torch.Tensor | None = None
