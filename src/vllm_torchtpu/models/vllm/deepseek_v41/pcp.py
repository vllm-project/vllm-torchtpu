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

import contextlib
import dataclasses
from collections.abc import Iterator
from typing import Any, cast

import torch
from torch import Tensor

from vllm_torchtpu.distributed.pcp import all_gather_equal_tokens
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata

_active_step: PcpStep | None = None
_global_phase = False


def split_size(total: Tensor, size: int) -> Tensor:
    """Return contiguous rows per PCP rank: ``max(1, ceil(total / size))``."""
    return torch.clamp((total + (size - 1)) // size, min=1)


def _resolve_distribution(
    seq_lens: Tensor,
    query_start_loc: Tensor,
    request_distribution: Tensor | None,
) -> Tensor:
    """Return ``int64[3]`` ``[num_decode, num_decode, num_reqs]`` on device."""
    if request_distribution is not None:
        return request_distribution.to(torch.long)
    qsl = query_start_loc.to(torch.long)
    q_lens = qsl[1:] - qsl[:-1]
    active = (seq_lens > 0) & (q_lens > 0)
    n_reqs = active.sum(dtype=torch.long)
    n_dec = (active & (q_lens == 1)).sum(dtype=torch.long)
    return torch.stack([n_dec, n_dec, n_reqs])


def request_major_index(
    total: Tensor, local_tokens: int, size: int
) -> tuple[Tensor, Tensor]:
    """Map each request-major row to its packed all-gather row and valid mask."""
    rows = local_tokens * size
    interleave = split_size(total, size)
    req = torch.arange(rows, dtype=torch.long, device=total.device)
    valid = req < total
    safe = torch.where(valid, req, torch.zeros_like(req))
    packed = (safe // interleave) * local_tokens + safe % interleave
    return packed.clamp(0, rows - 1), valid


def gather_request_major(x: Tensor, index: Tensor, valid: Tensor) -> Tensor:
    """All-gather rank-local rows across PCP and restore request-major order."""
    packed = all_gather_equal_tokens(x, dim=0)
    rows = torch.index_select(packed, 0, index)
    mask = valid.view(-1, *([1] * (x.dim() - 1)))
    return torch.where(mask, rows, torch.zeros_like(rows))


def local_slice(total: Tensor, rank: Tensor, size: int) -> tuple[Tensor, Tensor]:
    """Return ``(start, num_local)`` for ``rank``'s request-major token slice."""
    interleave = split_size(total, size)
    start = torch.minimum(rank.to(torch.long) * interleave, total)
    end = torch.minimum(start + interleave, total)
    return start, end - start


def select_local_rows(
    x: Tensor, start: Tensor, num_local: Tensor, local_tokens: int
) -> Tensor:
    """Slice ``x[start : start + num_local]`` and zero-pad to ``local_tokens``."""
    rows = torch.arange(local_tokens, dtype=torch.long, device=x.device)
    index = (start + rows).clamp(0, max(x.shape[0] - 1, 0))
    out = torch.index_select(x, 0, index)
    mask = (rows < num_local).view(-1, *([1] * (x.dim() - 1)))
    return torch.where(mask, out, torch.zeros_like(out))


def local_request_metadata(
    seq_lens: Tensor,
    query_start_loc: Tensor,
    request_distribution: Tensor | None,
    rank: Tensor,
    size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Narrow global request metadata to ``rank``'s compacted query slice."""
    max_seqs = seq_lens.shape[0]
    qsl = query_start_loc.to(torch.long)
    dist = _resolve_distribution(seq_lens, query_start_loc, request_distribution)
    dist_dtype = (
        request_distribution.dtype if request_distribution is not None else torch.int32
    )
    n_reqs = dist[2].clamp(0, max_seqs)
    total = torch.gather(qsl, 0, n_reqs.view(1)).squeeze(0)
    start, num_local = local_slice(total, rank, size)
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
    source = torch.clamp(first + req, max=max(max_seqs - 1, 0))
    valid = req < n_local

    trimmed = qsl[1:] - q_end
    local_seq_lens = torch.where(
        valid,
        seq_lens.to(torch.long)[source] - trimmed[source],
        torch.zeros_like(req),
    ).to(seq_lens.dtype)
    local_qsl = torch.cat(
        [
            torch.where(valid, q_start[source] - start, num_local),
            num_local.view(1),
        ]
    ).to(query_start_loc.dtype)
    n_dec = torch.clamp(torch.minimum(dist[0], first + n_local) - first, min=0)
    n_dec = torch.where(n_local > 0, n_dec, torch.zeros_like(n_dec))
    local_dist = torch.stack([n_dec, n_dec, n_local]).to(dist_dtype)
    return local_seq_lens, local_qsl, local_dist, source, start, num_local, total


@dataclasses.dataclass
class PcpStep:
    """Per-step PCP mappings shared across all layers in a compiled stage."""

    size: int
    index: Tensor
    valid: Tensor
    positions: Tensor
    start: Tensor
    num_local: Tensor
    source: Tensor
    seq_lens: Tensor
    query_start_loc: Tensor
    request_distribution: Tensor
    global_window_starts: Tensor | None = None
    local_window_starts: Tensor | None = None
    _local_cache: list[tuple[AttentionMetadata, AttentionMetadata]] = dataclasses.field(
        default_factory=list
    )

    @classmethod
    def build(
        cls,
        metadata: dict[str, Any],
        positions: Tensor,
        rank: Tensor,
        size: int,
        window_starts: Tensor | None = None,
    ) -> PcpStep:
        """Construct a :class:`PcpStep` from active stage metadata and positions."""
        first = cast(AttentionMetadata, next(iter(metadata.values())))
        first_seq_lens = cast(Tensor, first.seq_lens)
        first_qsl = cast(Tensor, first.query_start_loc)
        first_dist = cast(Tensor | None, first.request_distribution)
        local_tokens = positions.shape[0]
        seq_lens, qsl, dist, source, start, num_local, total = local_request_metadata(
            first_seq_lens,
            first_qsl,
            first_dist,
            rank,
            size,
        )
        index, valid = request_major_index(total, local_tokens, size)
        global_positions = gather_request_major(positions, index, valid)
        local_starts: Tensor | None = None
        if window_starts is not None:
            rows = torch.arange(source.shape[0], device=source.device)
            safe_source = source.clamp_max(max(window_starts.shape[0] - 1, 0))
            local_starts = torch.where(
                rows < dist[2], window_starts[safe_source], 0
            ).to(window_starts.dtype)
        return cls(
            size=size,
            index=index,
            valid=valid,
            positions=global_positions,
            start=start,
            num_local=num_local,
            source=source,
            seq_lens=seq_lens,
            query_start_loc=qsl,
            request_distribution=dist,
            global_window_starts=window_starts,
            local_window_starts=local_starts,
        )

    def gather(self, x: Tensor) -> Tensor:
        """All-gather ``x`` across PCP ranks in global request-major order."""
        return gather_request_major(x, self.index, self.valid)

    def local_rows(self, x: Tensor, local_tokens: int) -> Tensor:
        """Select this rank's rows from request-major ``x`` with zero padding."""
        return select_local_rows(x, self.start, self.num_local, local_tokens)

    def local(self, meta: AttentionMetadata) -> AttentionMetadata:
        """Return ``meta`` narrowed to this rank's compacted local requests."""
        for global_meta, cached_local in self._local_cache:
            if global_meta is meta:
                return cached_local
        rows = self.seq_lens.shape[0]
        raw_tables = cast(Tensor, meta.block_tables)
        local_tables = torch.index_select(
            raw_tables.reshape(rows, -1), 0, self.source
        ).reshape(raw_tables.shape)
        local_meta = dataclasses.replace(
            meta,
            block_tables=local_tables,
            seq_lens=self.seq_lens,
            query_start_loc=self.query_start_loc,
            request_distribution=self.request_distribution,
        )
        self._local_cache.append((meta, local_meta))
        return local_meta


def current() -> PcpStep | None:
    """Return the active :class:`PcpStep` for the current forward stage, if any."""
    return _active_step


def in_global_phase() -> bool:
    """Return ``True`` while executing Phase 1 (global KV publication)."""
    return _global_phase


@contextlib.contextmanager
def step_scope(step: PcpStep | None) -> Iterator[None]:
    """Install ``step`` as the active PCP context for one forward stage."""
    global _active_step
    previous = _active_step
    _active_step = step
    try:
        yield
    finally:
        _active_step = previous


@contextlib.contextmanager
def global_phase() -> Iterator[None]:
    """Switch metadata and replay resolution to global request-major Phase 1."""
    global _global_phase
    previous = _global_phase
    _global_phase = True
    try:
        yield
    finally:
        _global_phase = previous
