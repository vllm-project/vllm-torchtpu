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
"""Sizing prefill chunks for the pipeline chunk scheduler.

The stages of a pipeline advance in lockstep (``distributed/pp_wave.py``).
While the pending prefill work cannot fill every stage with a full step, a
step is held to that work spread over the stages (`step_limit`), so a lone
prompt runs as one small step per stage instead of trailing itself through
the pipeline one full step at a time.

Chunks are cut in granules: an eighth of a step, rounded up to whole KV
blocks when a mamba state cache needs block-aligned chunk ends. Buckets at
every granule multiple are compiled so a shortened step pads to its own size.

The batched attention kernel schedules one (query block, KV block) pair per
entry of an SMEM table of fixed capacity, so a step also holds at most that
many pairs: a chunk of ``c`` tokens behind ``p`` tokens of context needs
about ``c/bq * (p + c/2)/bkv`` of them, which is what bounds a chunk deep in
a very long prompt.
"""

import numpy as np


def chunk_granularity(block_size: int, max_tokens: int, block_aligned: bool) -> int:
    """Token multiple that prefill chunks are cut at: an eighth of a step,
    rounded up to whole KV blocks when chunk ends must stay block aligned and
    a step spans at least one block. A block larger than a step leaves the
    scheduler's own split to stop a chunk at the block boundary."""
    granule = max(max_tokens // 8, 1)
    if block_aligned and block_size <= max_tokens:
        granule = -(-granule // block_size) * block_size
    return granule


def chunk_buckets(max_tokens: int, granularity: int) -> list[int]:
    """Token buckets at every granule multiple below ``max_tokens``."""
    return list(range(granularity, max_tokens, granularity))


def chunk_pairs(schedule: tuple[int, int, int] | None, chunk: int, prefix: int) -> int:
    """(query block, KV block) pairs the attention kernel schedules for a
    chunk behind a prefix with ``schedule`` = (capacity, query tile, KV
    tile): every query block attends to the KV blocks up to its own last
    token. 0 without a schedule."""
    if schedule is None or chunk <= 0:
        return 0
    _, bq, bkv = schedule
    total = 0
    for start in range(0, chunk, bq):
        total += -(-(prefix + min(start + bq, chunk)) // bkv)
    return total


def schedule_pairs(q_lens, kv_lens, bq: int, bkv: int) -> int:
    """(query block, KV block) pairs the attention kernel schedules for
    sequences with ``q_lens`` new tokens at the end of ``kv_lens`` tokens of
    context, with query tile ``bq`` and KV tile ``bkv``: every query block
    attends to the KV blocks up to its own last token. The multi-sequence
    form of `chunk_pairs`."""
    q = np.asarray(q_lens, dtype=np.int64)
    kv = np.asarray(kv_lens, dtype=np.int64)
    keep = q > 0
    q, kv = q[keep], kv[keep]
    if q.size == 0:
        return 0
    blocks = -(-q // bq)
    i = np.arange(int(blocks.max()))
    q_end = np.minimum((i[None, :] + 1) * bq, q[:, None])
    k_end = -(-(kv[:, None] - q[:, None] + q_end) // bkv)
    return int(np.where(i[None, :] < blocks[:, None], k_end, 0).sum())


def step_limit(pending: int, pp_size: int, granularity: int, step: int) -> int | None:
    """Token bucket a step is held to while ``pending`` prefill tokens cannot
    fill all ``pp_size`` stages with full steps of ``step`` tokens: the
    pending work spread over the stages, rounded up to a whole
    ``granularity`` and at most a full step. None when no work is pending or
    the spread fills a full step."""
    if pending <= 0:
        return None
    share = -(-pending // pp_size)
    if share >= step:
        return None
    return min(-(-share // granularity) * granularity, step)
