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
"""PCP manager sparse MLA attention helpers."""

from __future__ import annotations

from typing import Any

import torch

from vllm_torchtpu import envs


def is_pcp_kv_gather_metadata(attn_metadata: Any) -> bool:
    return getattr(attn_metadata, "pcp_metadata", None) is not None


def pcp_chunk_pages(page_size: int, dcp_size: int) -> int:
    """Workspace pages (DCP virtual pages) per gather chunk."""
    return max(1, envs.TPU_PCP_KV_CHUNK_TOKENS // (int(page_size) * int(dcp_size)))


def take_decode_rows(x: torch.Tensor, md: Any) -> torch.Tensor:
    """Select the replicated decode window rows at `pcp_metadata.rep_row_idx`."""
    return x.index_select(0, md.pcp_metadata.rep_row_idx.to(torch.long))


def merge_decode_rows(
    prefill_rows: torch.Tensor, decode_rows: torch.Tensor, md: Any
) -> torch.Tensor:
    """Splice valid decode rows from `decode_rows` back into `prefill_rows`."""
    pcp = md.pcp_metadata
    num_rows = prefill_rows.shape[0]
    rows = torch.arange(num_rows, dtype=torch.int32, device=prefill_rows.device)
    slot = rows - pcp.rep_row_idx[0]
    is_decode = (slot >= 0) & (slot < pcp.rep_request_distribution[0])
    slot = torch.clamp(slot, 0, decode_rows.shape[0] - 1).to(torch.long)
    picked = decode_rows.index_select(0, slot)
    mask = is_decode.view(-1, *([1] * (prefill_rows.dim() - 1)))
    return torch.where(mask, picked, prefill_rows)
