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
"""Two-phase local-query prefill context parallelism (PCP) for DeepSeek-V4.1."""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import torch
from torch import Tensor

from vllm_torchtpu.distributed.pcp import all_gather_equal_tokens
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata


def split_size(total: Tensor, pcp_size: int) -> Tensor:
    """Return contiguous rows per PCP rank: ``max(1, ceil(total / pcp_size))``."""
    return torch.clamp((total + (pcp_size - 1)) // pcp_size, min=1)


def request_major_index(
    total: Tensor, local_tokens: int, pcp_size: int
) -> tuple[Tensor, Tensor]:
    """Map each request-major row to its packed all-gather row and valid mask."""
    rows = local_tokens * pcp_size
    interleave = split_size(total, pcp_size)
    req = torch.arange(rows, dtype=torch.long, device=total.device)
    valid = req < total
    safe = torch.where(valid, req, torch.zeros_like(req))
    packed = (safe // interleave) * local_tokens + safe % interleave
    return packed.clamp(0, rows - 1), valid


def gather_request_major(
    x: Tensor, packed_to_req_index: Tensor, req_valid_mask: Tensor
) -> Tensor:
    """All-gather rank-local rows across PCP and restore request-major order."""
    packed = all_gather_equal_tokens(x, dim=0)
    rows = torch.index_select(packed, 0, packed_to_req_index)
    mask = req_valid_mask.view(-1, *([1] * (x.dim() - 1)))
    return torch.where(mask, rows, torch.zeros_like(rows))


def local_slice(total: Tensor, rank: Tensor, pcp_size: int) -> tuple[Tensor, Tensor]:
    """Return ``(local_start, num_local)`` for ``rank``'s request-major slice."""
    interleave = split_size(total, pcp_size)
    start = torch.minimum(rank.to(torch.long) * interleave, total)
    end = torch.minimum(start + interleave, total)
    return start, end - start


def select_local_rows(
    x: Tensor, local_start: Tensor, num_local: Tensor, local_tokens: int
) -> Tensor:
    """Slice ``x[local_start : local_start + num_local]`` padded to ``local_tokens``."""
    rows = torch.arange(local_tokens, dtype=torch.long, device=x.device)
    index = (local_start + rows).clamp(0, max(x.shape[0] - 1, 0))
    out = torch.index_select(x, 0, index)
    mask = (rows < num_local).view(-1, *([1] * (x.dim() - 1)))
    return torch.where(mask, out, torch.zeros_like(out))


@dataclasses.dataclass(frozen=True)
class LocalRequestSlice:
    """Compacted per-rank request metadata for a contiguous PCP query slice."""

    seq_lens: Tensor
    query_start_loc: Tensor
    request_distribution: Tensor
    local_req_sources: Tensor
    local_start: Tensor
    num_local: Tensor
    total_tokens: Tensor


def local_request_metadata(
    seq_lens: Tensor,
    query_start_loc: Tensor,
    request_distribution: Tensor | None,
    rank: Tensor,
    pcp_size: int,
) -> LocalRequestSlice:
    """Narrow global request metadata to ``rank``'s compacted query slice."""
    if request_distribution is None:
        raise ValueError("AttentionMetadata.request_distribution is required for PCP.")
    max_seqs = seq_lens.shape[0]
    qsl = query_start_loc.to(torch.long)
    dist = request_distribution.to(torch.long)
    n_reqs = dist[2].clamp(0, max_seqs)
    total = torch.gather(qsl, 0, n_reqs.view(1)).squeeze(0)
    start, num_local = local_slice(total, rank, pcp_size)
    end = start + num_local

    req = torch.arange(max_seqs, dtype=torch.long, device=seq_lens.device)
    q_start = torch.minimum(torch.maximum(qsl[:-1], start), end)
    q_end = torch.minimum(torch.maximum(qsl[1:], start), end)
    has_local = (req < n_reqs) & (q_end > q_start)
    n_local = has_local.sum()
    first = torch.where(
        n_local > 0,
        torch.argmax(has_local.to(torch.int32)),
        torch.zeros_like(n_local),
    )
    local_req_sources = torch.clamp(first + req, max=max(max_seqs - 1, 0))
    valid = req < n_local

    trimmed = qsl[1:] - q_end
    local_seq_lens = torch.where(
        valid,
        seq_lens.to(torch.long)[local_req_sources] - trimmed[local_req_sources],
        torch.zeros_like(req),
    ).to(seq_lens.dtype)
    local_qsl = torch.cat(
        [
            torch.where(valid, q_start[local_req_sources] - start, num_local),
            num_local.view(1),
        ]
    ).to(query_start_loc.dtype)
    n_dec = torch.clamp(torch.minimum(dist[0], first + n_local) - first, min=0)
    n_dec = torch.where(n_local > 0, n_dec, torch.zeros_like(n_dec))
    local_dist = torch.stack([n_dec, n_dec, n_local]).to(request_distribution.dtype)
    return LocalRequestSlice(
        seq_lens=local_seq_lens,
        query_start_loc=local_qsl,
        request_distribution=local_dist,
        local_req_sources=local_req_sources,
        local_start=start,
        num_local=num_local,
        total_tokens=total,
    )


@dataclasses.dataclass
class PcpStep:
    """Per-step PCP mappings shared across all layers in a compiled stage."""

    pcp_size: int
    packed_to_req_index: Tensor
    req_valid_mask: Tensor
    positions: Tensor
    local_start: Tensor
    num_local: Tensor
    local_req_sources: Tensor
    seq_lens: Tensor
    query_start_loc: Tensor
    request_distribution: Tensor
    local_window_starts: Tensor | None = None
    _local_cache: dict[int, AttentionMetadata] = dataclasses.field(default_factory=dict)

    @classmethod
    def build(
        cls,
        metadata: dict[str, Any],
        positions: Tensor,
        rank: Tensor,
        pcp_size: int,
        window_starts: Tensor | None = None,
    ) -> PcpStep:
        """Construct a :class:`PcpStep` from active stage metadata and positions."""
        first = cast(AttentionMetadata, next(iter(metadata.values())))
        first_seq_lens = cast(Tensor, first.seq_lens)
        first_qsl = cast(Tensor, first.query_start_loc)
        first_dist = cast(Tensor | None, first.request_distribution)
        local_tokens = positions.shape[0]
        sliced = local_request_metadata(
            first_seq_lens,
            first_qsl,
            first_dist,
            rank,
            pcp_size,
        )
        packed_to_req_index, req_valid_mask = request_major_index(
            sliced.total_tokens, local_tokens, pcp_size
        )
        global_positions = gather_request_major(
            positions, packed_to_req_index, req_valid_mask
        )
        local_starts: Tensor | None = None
        if window_starts is not None:
            sources = sliced.local_req_sources
            rows = torch.arange(sources.shape[0], device=sources.device)
            safe_sources = sources.clamp_max(max(window_starts.shape[0] - 1, 0))
            local_starts = torch.where(
                rows < sliced.request_distribution[2],
                window_starts[safe_sources],
                0,
            ).to(window_starts.dtype)
        return cls(
            pcp_size=pcp_size,
            packed_to_req_index=packed_to_req_index,
            req_valid_mask=req_valid_mask,
            positions=global_positions,
            local_start=sliced.local_start,
            num_local=sliced.num_local,
            local_req_sources=sliced.local_req_sources,
            seq_lens=sliced.seq_lens,
            query_start_loc=sliced.query_start_loc,
            request_distribution=sliced.request_distribution,
            local_window_starts=local_starts,
        )

    def gather(self, x: Tensor) -> Tensor:
        """All-gather ``x`` across PCP ranks in global request-major order."""
        return gather_request_major(x, self.packed_to_req_index, self.req_valid_mask)

    def local_rows(self, x: Tensor, local_tokens: int) -> Tensor:
        """Select this rank's rows from request-major ``x`` with zero padding."""
        return select_local_rows(x, self.local_start, self.num_local, local_tokens)

    def local(self, meta: AttentionMetadata) -> AttentionMetadata:
        """Return ``meta`` narrowed to this rank's compacted local requests.

        ``mamba_*`` per-request fields stay global because circular compressor
        state is only read and updated during Phase 1 (global KV publication).
        """
        cache_key = id(meta)
        cached_local = self._local_cache.get(cache_key)
        if cached_local is not None:
            return cached_local
        rows = self.seq_lens.shape[0]
        raw_tables = cast(Tensor, meta.block_tables)
        local_tables = torch.index_select(
            raw_tables.reshape(rows, -1), 0, self.local_req_sources
        ).reshape(raw_tables.shape)
        local_meta = dataclasses.replace(
            meta,
            block_tables=local_tables,
            seq_lens=self.seq_lens,
            query_start_loc=self.query_start_loc,
            request_distribution=self.request_distribution,
        )
        self._local_cache[cache_key] = local_meta
        return local_meta
