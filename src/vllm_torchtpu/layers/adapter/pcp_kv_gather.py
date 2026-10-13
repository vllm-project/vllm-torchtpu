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
"""PCP manager sparse MLA attention (after upstream vLLM's
`pcp_dcp_kv_gather`).

The KV cache stays DCP-sharded; queries are split across PCP ranks. Per layer:

  1. write the split prefill rows (PCP-gathered) into the DCP shards;
  2. decodes (replicated) take the DCP path from their own request window
     (`pcp_rep_*`), which writes their rows and attends;
  3. split prefills gather their KV into a workspace over the DCP axis and
     attend it with the unsharded kernel, under a device-side
     `lax.cond(pcp_num_split_reqs > 0)`.

See `kernels/mla/pcp_workspace.py`.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.dcp import get_dcp_group
from vllm_torchtpu.distributed.pcp import get_pcp_group

PCP_KV_GATHER_SEQUENCE_LAYOUT_PROTOCOL = "pcp_kv_gather"


def is_pcp_kv_gather_metadata(attn_metadata: Any) -> bool:
    return (
        getattr(attn_metadata, "sequence_layout_protocol", None)
        == PCP_KV_GATHER_SEQUENCE_LAYOUT_PROTOCOL
    )


def pcp_chunk_pages(page_size: int, dcp_size: int) -> int:
    """Workspace pages (DCP virtual pages) per gather chunk."""
    return max(1, envs.TPU_PCP_KV_CHUNK_TOKENS // (int(page_size) * int(dcp_size)))


_DCP_RANK_ORDER_VERIFIED = False


def verify_dcp_rank_order() -> None:
    """Check DCP rank d is PCP rank d // tp, which `pcp_gather_rows` assumes.

    Runs a CPU collective: call it eagerly, not from a traced forward.
    """
    global _DCP_RANK_ORDER_VERIFIED
    if _DCP_RANK_ORDER_VERIFIED:
        return
    dcp_group = get_dcp_group()
    pcp_group = get_pcp_group()
    if dcp_group is None or pcp_group is None:
        return
    pcp_size = int(pcp_group.world_size)
    dcp_size = int(dcp_group.world_size)
    tp_size = dcp_size // pcp_size
    gathered: list[Any] = [None] * dcp_size
    torch.distributed.all_gather_object(
        gathered, int(pcp_group.rank_in_group), group=dcp_group.cpu_group
    )
    expected = [d // tp_size for d in range(dcp_size)]
    if gathered != expected:
        raise RuntimeError(
            "The PCP manager needs DCP rank d to be PCP rank d // tp_size; got PCP "
            f"ranks {gathered} by DCP rank (expected {expected})."
        )
    _DCP_RANK_ORDER_VERIFIED = True


def take_decode_rows(x: torch.Tensor, md: Any) -> torch.Tensor:
    """The decode window: `[rep_window, ...]` rows at `pcp_rep_row_idx`."""
    return x.index_select(0, md.pcp_rep_row_idx.to(torch.long))


def merge_decode_rows(
    prefill_rows: torch.Tensor, decode_rows: torch.Tensor, md: Any
) -> torch.Tensor:
    """`prefill_rows` with the decode rows `[start, start + n)` taken from
    `decode_rows` (window slot k is row start + k)."""
    num_rows = prefill_rows.shape[0]
    rows = torch.arange(num_rows, dtype=torch.int32, device=prefill_rows.device)
    slot = rows - md.pcp_rep_row_idx[0]
    is_decode = (slot >= 0) & (slot < md.pcp_rep_request_distribution[0])
    slot = torch.clamp(slot, 0, decode_rows.shape[0] - 1).to(torch.long)
    picked = decode_rows.index_select(0, slot)
    mask = is_decode.view(-1, *([1] * (prefill_rows.dim() - 1)))
    return torch.where(mask, picked, prefill_rows)
