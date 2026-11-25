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
"""Per-tier hit-source attribution helpers for Raiden KV cache offloading.

`compute_load_tokens_by_source` and `split_external_hit` run in `EngineCore`
to split each external prefix-cache hit across `host`, `p2p`, and `disk` for
vLLM's `vllm:prompt_tokens_cached_by_source` counter.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Iterable, Mapping

from vllm.v1.kv_offload.base import OffloadKey, get_offload_group_idx
from vllm.v1.metrics.cache_hit_source import CacheHitSource

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


# Tier of each sub-block of one `OffloadKey`, as `(source, count)` runs of
# consecutive sub-blocks in the same tier; the counts sum to
# `device_block_size_factor`. For example, with 4 sub-blocks per key,
# `((HOST, 3), (DISK, 1))` means sub-blocks 0-2 are in host DRAM and 3 is on
# disk, and a key entirely in host DRAM is `((HOST, 4),)`.
SubBlockSourceCounts = tuple[tuple[CacheHitSource, int], ...]


def split_external_hit(
    tokens_by_source: Mapping[CacheHitSource, int], num_external_tokens: int
) -> dict[CacheHitSource, int]:
    """Complete a per-tier hit split so it sums to `num_external_tokens`.

    Upstream `PrefixCacheStats.record` replaces any split that does not sum to
    the hit count with all-`external_unspecified`. Tokens no tier supplied
    (spans no loaded key covers, such as omitted sliding-window history) are
    reported as `EXTERNAL_UNSPECIFIED` so the known tiers survive. For example,
    `split_external_hit({HOST: 768, DISK: 512}, 1536)` returns
    `{HOST: 768, DISK: 512, EXTERNAL_UNSPECIFIED: 256}`; the incomplete split
    would have been reported as `{EXTERNAL_UNSPECIFIED: 1536}`.
    """
    split = dict(tokens_by_source)
    unattributed = num_external_tokens - sum(split.values())
    if unattributed > 0:
        unspecified = CacheHitSource.EXTERNAL_UNSPECIFIED
        split[unspecified] = split.get(unspecified, 0) + unattributed
    return split


def compute_load_tokens_by_source(
    *,
    start_token: int,
    end_token: int,
    load_key_ranges: Iterable[tuple[int, int, OffloadKey]],
    key_sources: Mapping[OffloadKey, SubBlockSourceCounts],
    tokens_per_chunk_by_group: Mapping[int, int],
    mamba_group_ids: Collection[int] = (),
) -> dict[CacheHitSource, int]:
    """Compute deduplicated external token counts at sub-block granularity.

    Phase 1 converts each key's clipped `[lo, hi)` span into interval events
    from its `SubBlockSourceCounts` in `key_sources`:
    - Uniform keys (`len(source_counts) == 1`) emit a single `[lo, hi)`
      interval in O(1) regardless of `device_block_size_factor`.
    - Mixed Attention keys split `[lo, hi)` across `(source, count)` pairs
      (`count * tokens_per_sub` tokens each), preserving exact `HOST` /
      `P2P` / `DISK` ratios without iterating individual sub-blocks.
    - Recurrent (`MambaSpec`) keys slice a single state checkpoint by bytes
      rather than token positions; all byte slices are needed together to
      restore that checkpoint, so `[lo, hi)` uses `CacheHitSource.outermost`.
    - A mixed key that cannot be split evenly by sub-block also uses
      `CacheHitSource.outermost`, and logs a warning.

    Phase 2 sweeps sorted event boundaries across `[start_token, end_token)`
    to deduplicate overlapping ranges emitted by multi-group / hybrid models
    for the same prompt tokens, attributing shared spans via
    `CacheHitSource.outermost` (the slowest tier required to materialize the
    span). Tokens no key covers are left out; `split_external_hit`
    reports them to upstream as `EXTERNAL_UNSPECIFIED`.

    For example, in a hybrid model whose attention keys for tokens
    `[0, 1536)` are in host DRAM and whose Mamba key for `[768, 1536)` is on
    disk, `[768, 1536)` needs both tiers and counts as the slower one:
    `{HOST: 768, DISK: 768}`.
    """
    if end_token <= start_token:
        return {}

    # Phase 1: build `(token_pos, +1/-1, source)` interval events.
    events: list[tuple[int, int, CacheHitSource]] = []
    for raw_lo, raw_hi, key in load_key_ranges:
        # Clip out tokens already resident in device HBM (`< start_token`).
        lo = max(start_token, raw_lo)
        hi = min(end_token, raw_hi)
        if lo >= hi:
            continue
        source_counts = key_sources[key]
        if len(source_counts) == 1:
            only_source = source_counts[0][0]
            events.extend(((lo, 1, only_source), (hi, -1, only_source)))
            continue

        group_idx = get_offload_group_idx(key)
        is_mamba = group_idx in mamba_group_ids
        tokens_per_chunk = tokens_per_chunk_by_group.get(group_idx, 0)
        num_sub = sum(count for _, count in source_counts)
        if is_mamba or tokens_per_chunk <= 0 or tokens_per_chunk % num_sub != 0:
            if not is_mamba:
                logger.warning_once(
                    "TPURaidenOffloadingConnector: cannot split a mixed-tier "
                    "key of KV cache group %d by sub-block (%d tokens per "
                    "chunk, %d sub-blocks); attributing it to its slowest tier.",
                    group_idx,
                    tokens_per_chunk,
                    num_sub,
                )
            source = CacheHitSource.outermost(src for src, _ in source_counts)
            events.extend(((lo, 1, source), (hi, -1, source)))
            continue

        tokens_per_sub = tokens_per_chunk // num_sub
        chunk_start = (lo // tokens_per_chunk) * tokens_per_chunk
        sub_offset = 0
        for source, count in source_counts:
            span_lo = max(lo, chunk_start + sub_offset * tokens_per_sub)
            sub_offset += count
            span_hi = min(hi, chunk_start + sub_offset * tokens_per_sub)
            if span_lo < span_hi:
                events.extend(((span_lo, 1, source), (span_hi, -1, source)))

    if not events:
        return {}

    # Phase 2: sweep sorted endpoints to deduplicate overlapping KV groups.
    sources: dict[CacheHitSource, int] = {}
    active: Counter[CacheHitSource] = Counter()
    position = start_token
    for boundary, delta, source in sorted(events):
        if boundary > position:
            active_sources = [src for src, cnt in active.items() if cnt > 0]
            if active_sources:
                resolved = CacheHitSource.outermost(active_sources)
                sources[resolved] = sources.get(resolved, 0) + (boundary - position)
        active[source] += delta
        position = boundary
    return sources
