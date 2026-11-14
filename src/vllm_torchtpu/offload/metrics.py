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
"""Observability metrics and collectors for Raiden KV cache offloading.

In vLLM V1, the scheduler (`EngineCore`) and Prometheus `/metrics` endpoint
(`API Server`) run in separate processes, and Raiden executes all HBM <->
Host/Peer/Storage transfers from the scheduler-side `KVCacheStore` controller
(worker-side `get_finished()` is always empty):

- `RaidenOffloadStatsCollector` runs in `EngineCore`, tracking per-sub-block
  tier residency and deduplicated token counts, draining per-step deltas as
  `OffloadingConnectorStats` via `pop_stats()`.
- `RaidenOffloadPromMetrics` runs in the API server process, registering
  Raiden's `OffloadingSpec` metric definitions and applying IPC stats deltas to
  Prometheus counters and histograms.
"""

from __future__ import annotations

import copy
import enum
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    PromMetric,
    PromMetricT,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
    OffloadPromMetrics,
)
from vllm.v1.kv_offload.base import (
    OffloadingCounterMetadata,
    OffloadingMetricMetadata,
    OffloadingSpec,
    OffloadKey,
    get_offload_group_idx,
)
from vllm.v1.metrics.cache_hit_source import CacheHitSource

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


# Load/hit metrics label where data is read from (`source`), aligned with
# upstream `vllm:prompt_tokens_cached_by_source` for PromQL rollups. `backend`
# names what serves the tier: `dram` for host, `rdma` for p2p, and the
# configured storage type for disk, so a new storage backend adds label values
# rather than metrics.
OFFLOAD_LOAD_LABELNAMES: tuple[str, str] = ("source", "backend")
UNKNOWN_BACKEND_LABEL = "unknown"
HOST_OFFLOAD_LABELS: tuple[str, str] = (CacheHitSource.HOST.value, "dram")
P2P_OFFLOAD_LABELS: tuple[str, str] = (CacheHitSource.P2P.value, "rdma")
# Reserved strictly for unrecognized C++ `BlockStatus` values (never used for
# keys missing from `_step_key_sources`).
UNSPECIFIED_OFFLOAD_LABELS: tuple[str, str] = (
    CacheHitSource.EXTERNAL_UNSPECIFIED.value,
    UNKNOWN_BACKEND_LABEL,
)

# Tier of each sub-block of one `OffloadKey`, as `(source, count)` runs of
# consecutive sub-blocks in the same tier; the counts sum to
# `device_block_size_factor`. For example, with 4 sub-blocks per key,
# `((HOST, 3), (DISK, 1))` means sub-blocks 0-2 are in host DRAM and 3 is on
# disk, and a key entirely in host DRAM is `((HOST, 4),)`.
SubBlockSourceCounts = tuple[tuple[CacheHitSource, int], ...]


class RaidenOffloadMetricName(enum.StrEnum):
    """Directional Prometheus metrics emitted by Raiden KV offload."""

    HIT_TOKENS = "vllm:kv_offload_hit_tokens"


def build_raiden_offload_metric_definitions(
    extra_config: Mapping[str, Any],
) -> dict[str, OffloadingMetricMetadata]:
    """Return Prometheus metric definitions for TPURaidenStoreOffloadingSpec."""
    del extra_config
    return {
        RaidenOffloadMetricName.HIT_TOKENS: OffloadingCounterMetadata(
            documentation=(
                "Number of prompt tokens matched and pinned in external KV "
                "offload tiers at scheduler admission, labeled by source and "
                "backend. Unlike vllm:prompt_tokens_cached_by_source, splits "
                "disk hits by storage backend and counts only pinned tokens."
            ),
            labelnames=OFFLOAD_LOAD_LABELNAMES,
        ),
    }


class RaidenOffloadPromMetrics(OffloadPromMetrics):
    """Prometheus metrics binder for TPURaidenOffloadingConnector.

    `TPURaidenOffloadingConnector` instantiates `TPURaidenStoreOffloadingSpec`
    directly (bypassing `OffloadingSpecFactory`), whereas upstream
    `OffloadPromMetrics.__init__` in the API server process resolves
    `spec_cls.build_metric_definitions` via `OffloadingSpecFactory` using
    `kv_connector_extra_config["spec_name"]`. This subclass names `spec_cls` as
    an out-of-tree spec (`spec_name` + `spec_module_path`) on shallow config
    copies, so neither the factory registry nor the caller's `VllmConfig` and
    `KVTransferConfig` are mutated.

    `configured_tier_labels` holds the `(source, backend)` labels of the tiers
    the deployment enables. Each engine's `vllm:kv_offload_hit_tokens` series
    for these tiers are created at 0 on startup. Prometheus counts a series
    from the first value it scrapes, so `rate()` and `increase()` would miss
    the first hit of a series that only appeared with that hit.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
        *,
        spec_cls: type[OffloadingSpec],
        configured_tier_labels: Collection[tuple[str, str]],
    ):
        kv_transfer_config = vllm_config.kv_transfer_config
        assert kv_transfer_config is not None
        scoped_transfer_config = copy.copy(kv_transfer_config)
        scoped_transfer_config.kv_connector_extra_config = {
            **kv_transfer_config.kv_connector_extra_config,
            "spec_name": spec_cls.__name__,
            "spec_module_path": spec_cls.__module__,
        }
        scoped_vllm_config = copy.copy(vllm_config)
        scoped_vllm_config.kv_transfer_config = scoped_transfer_config
        super().__init__(
            scoped_vllm_config,
            metric_types,
            labelnames,
            per_engine_labelvalues,
        )
        zero_hits = OffloadingConnectorStats()
        for labelvalues in configured_tier_labels:
            zero_hits.increase_counter(
                RaidenOffloadMetricName.HIT_TOKENS, 0, labelvalues=labelvalues
            )
        for engine_idx in per_engine_labelvalues:
            self.observe(zero_hits.data, engine_idx)


def status_to_hit_source(status: Any, block_status_cls: Any) -> CacheHitSource:
    """Map a Raiden C++ `BlockStatus` value to upstream `CacheHitSource`."""
    if status in (block_status_cls.HOST, block_status_cls.HOST_AND_HBM):
        return CacheHitSource.HOST
    if status == block_status_cls.REMOTE:
        return CacheHitSource.P2P
    if status == block_status_cls.SHARED_STORAGE:
        return CacheHitSource.DISK
    return CacheHitSource.EXTERNAL_UNSPECIFIED


def external_sources_with_tokens(
    tokens_by_source: Mapping[CacheHitSource, int],
) -> dict[CacheHitSource, int]:
    """Keep the external sources that have at least one token.

    For example, `{DEVICE: 16, HOST: 64, P2P: 0, DISK: 128}` becomes
    `{HOST: 64, DISK: 128}`: `DEVICE` is local HBM, never an offload tier.
    Checks `isinstance(source, CacheHitSource)` explicitly because
    `CacheHitSource` inherits from `str` (`"host" == CacheHitSource.HOST`).
    """
    return {
        source: count
        for source, count in tokens_by_source.items()
        if (
            count > 0
            and isinstance(source, CacheHitSource)
            and source != CacheHitSource.DEVICE
        )
    }


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


class RaidenOffloadStatsCollector:
    """Scheduler-side collector producing `OffloadingConnectorStats` deltas."""

    def __init__(self, *, storage_backend: str | None = None):
        self._storage_backend = storage_backend
        self._stats = OffloadingConnectorStats()
        # Sub-block tiers of each key pinned this step, recorded by
        # `record_prepared_load` and cleared in `on_schedule_end()`.
        self._step_key_sources: dict[OffloadKey, SubBlockSourceCounts] = {}

    def source_labels(self, source: CacheHitSource) -> tuple[str, str]:
        """Map an external `CacheHitSource` to `(source, backend)` labels.

        For example, `HOST` maps to `("host", "dram")`, `P2P` to
        `("p2p", "rdma")`, and `DISK` to `("disk", "posix")` with a POSIX
        storage tier.

        Assumes one storage backend per engine: Raiden's `SHARED_STORAGE`
        status does not say which backend holds a block. If an engine read from
        two storage backends, hits from both would get the `storage_backend`
        label until Raiden reports the backend per block.
        """
        # Reject raw `str` before `==` checks since `CacheHitSource` is a `str`.
        if not isinstance(source, CacheHitSource):
            raise ValueError(f"Unsupported offload source for metrics: {source!r}")
        if source == CacheHitSource.HOST:
            return HOST_OFFLOAD_LABELS
        if source == CacheHitSource.P2P:
            return P2P_OFFLOAD_LABELS
        if source == CacheHitSource.DISK:
            if self._storage_backend is None:
                logger.warning_once(
                    "TPURaidenOffloadingConnector: disk source observed without "
                    "configured storage_backend; using backend=%r.",
                    UNKNOWN_BACKEND_LABEL,
                )
                return (CacheHitSource.DISK.value, UNKNOWN_BACKEND_LABEL)
            return (CacheHitSource.DISK.value, self._storage_backend)
        if source == CacheHitSource.EXTERNAL_UNSPECIFIED:
            return UNSPECIFIED_OFFLOAD_LABELS
        raise ValueError(f"Unsupported offload source for metrics: {source!r}")

    def _resolve_block_source(
        self, status: Any, block_status_cls: Any
    ) -> CacheHitSource:
        source = status_to_hit_source(status, block_status_cls)
        if source == CacheHitSource.EXTERNAL_UNSPECIFIED:
            logger.warning_once(
                "TPURaidenOffloadingConnector: unrecognized Raiden BlockStatus %s; "
                "attributing offload metrics to (%r, %r).",
                repr(status),
                *UNSPECIFIED_OFFLOAD_LABELS,
            )
        return source

    def record_prepared_load(
        self,
        keys: Sequence[OffloadKey],
        device_block_size_factor: int,
        remote_slices: Mapping[bytes, Any],
        sub_hashes: Sequence[bytes],
        block_status_cls: Any,
    ) -> None:
        """Record each key's `SubBlockSourceCounts` from `prepare_load`.

        `remote_slices` contains only the sub-hashes that remain non-local after
        pinning (any sub-hash promoted to local `HOST` DRAM before `prepare_load`
        is absent from `remote_slices`); all other sub-blocks are in host DRAM.
        For example, with `device_block_size_factor=4` and only the last of a
        key's 4 sub-hashes in `remote_slices` (status `SHARED_STORAGE`), the
        key is recorded as `((HOST, 3), (DISK, 1))`.
        """
        if not keys or device_block_size_factor <= 0:
            return
        if not remote_slices:
            # Fast path when every pinned sub-block is resident in local HOST
            # DRAM: share one 1-element tuple across all keys without scanning
            # `sub_hashes`.
            uniform_host: SubBlockSourceCounts = (
                (CacheHitSource.HOST, device_block_size_factor),
            )
            for key in keys:
                self._step_key_sources[key] = uniform_host
            return

        num_hashes = len(sub_hashes)
        resolved: dict[Any, CacheHitSource] = {}
        for idx, key in enumerate(keys):
            start = idx * device_block_size_factor
            end = min(start + device_block_size_factor, num_hashes)
            source_counts: list[tuple[CacheHitSource, int]] = []
            cur_source: CacheHitSource | None = None
            cur_count = 0
            for pos in range(start, end):
                remote_block = remote_slices.get(sub_hashes[pos])
                if remote_block is None:
                    source = CacheHitSource.HOST
                else:
                    source = resolved.get(remote_block.status)
                    if source is None:
                        source = resolved[remote_block.status] = (
                            self._resolve_block_source(
                                remote_block.status, block_status_cls
                            )
                        )
                if source == cur_source:
                    cur_count += 1
                else:
                    if cur_source is not None:
                        source_counts.append((cur_source, cur_count))
                    cur_source = source
                    cur_count = 1
            if cur_source is not None:
                source_counts.append((cur_source, cur_count))
            if source_counts:
                self._step_key_sources[key] = tuple(source_counts)

    def get_load_sub_sources(self, key: OffloadKey) -> SubBlockSourceCounts | None:
        """Return the `SubBlockSourceCounts` recorded for `key`, or `None` on miss."""
        source_counts = self._step_key_sources.get(key)
        if source_counts:
            return source_counts
        logger.warning_once(
            "TPURaidenOffloadingConnector: offload key missed step source cache; "
            "skipping offload metric attribution for unrecorded key."
        )
        return None

    def compute_load_tokens_by_source(
        self,
        *,
        start_token: int,
        end_token: int,
        load_key_ranges: Iterable[tuple[int, int, OffloadKey]],
        tokens_per_chunk_by_group: Mapping[int, int],
        mamba_group_ids: Collection[int] = (),
    ) -> dict[CacheHitSource, int]:
        """Compute deduplicated external token counts at sub-block granularity.

        Phase 1 converts each key's clipped `[lo, hi)` span into interval events
        from its `SubBlockSourceCounts`:
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
        - Keys missing from `_step_key_sources` are skipped rather than
          mislabeled as `EXTERNAL_UNSPECIFIED`.

        Phase 2 sweeps sorted event boundaries across `[start_token, end_token)`
        to deduplicate overlapping ranges emitted by multi-group / hybrid models
        for the same prompt tokens, attributing shared spans via
        `CacheHitSource.outermost` (the slowest tier required to materialize the
        span). Tokens no recorded key covers are left out; `split_external_hit`
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
            source_counts = self.get_load_sub_sources(key)
            if source_counts is None:
                continue
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
        return external_sources_with_tokens(sources)

    def on_schedule_end(self) -> None:
        self._step_key_sources.clear()

    def record_admission_hits(
        self,
        tokens_by_source: Mapping[CacheHitSource, int],
        *,
        preempted: bool = False,
    ) -> None:
        """Increment `vllm:kv_offload_hit_tokens` for non-preempted admissions.

        Matches upstream `Scheduler.schedule()`: re-admissions after preemption
        (`request.num_preemptions > 0`) skip `HIT_TOKENS` so per-request prefix
        hits are not double-counted.
        """
        if preempted:
            return
        for source, count in external_sources_with_tokens(tokens_by_source).items():
            self._stats.increase_counter(
                RaidenOffloadMetricName.HIT_TOKENS,
                count,
                labelvalues=self.source_labels(source),
            )

    def pop_stats(self) -> OffloadingConnectorStats | None:
        """Drain accumulated per-step stats deltas for IPC, or `None` if idle.

        Resets `self._stats` so each delta is shipped to `RaidenOffloadPromMetrics`
        in the API server process at most once.
        """
        if self._stats.is_empty():
            return None
        stats = self._stats
        self._stats = OffloadingConnectorStats()
        return stats
