# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""HBM block eviction age tracking and hardware wastage attribute helpers."""

import time
from collections.abc import Iterable, Mapping
from typing import Any

from vllm_torchtpu.telemetry.token_breakdown import BoundedLRU


class BlockEvictionTracker:
    """Tracks HBM block evictions, replica LRU horizon, and per-session eviction timestamps."""

    def __init__(self, maxsize: int = 131072, session_maxsize: int = 65536) -> None:
        self._evicted_at: BoundedLRU[Any, float] = BoundedLRU(maxsize)
        self._block_owner: BoundedLRU[Any, str] = BoundedLRU(maxsize)
        self._session_evicted_wall_ms: BoundedLRU[str, int] = BoundedLRU(
            session_maxsize
        )
        self._last_evicted_ts: float | None = None

    def register_session_blocks(
        self, session_id: str | None, block_hashes: Iterable[Any]
    ) -> None:
        """Associate `block_hashes` with `session_id` for deduplicated eviction tracking."""
        if not session_id:
            return
        for blk_hash in block_hashes:
            if blk_hash is not None:
                self._block_owner.put(blk_hash, session_id)

    def record_eviction(
        self,
        block_hash: Any,
        now: float | None = None,
        wall_time_ms: int | None = None,
    ) -> None:
        if block_hash is None:
            return
        ts = time.monotonic() if now is None else now
        self._evicted_at.put(block_hash, ts)
        self._last_evicted_ts = ts

        owner = self._block_owner.peek(block_hash)
        if owner is not None:
            evict_wall = (
                int(time.time() * 1000) if wall_time_ms is None else int(wall_time_ms)
            )
            self._session_evicted_wall_ms.put(owner, evict_wall)

    def last_eviction_age_ms(self, now: float | None = None) -> float | None:
        """Return milliseconds since the most recent block eviction on this pool."""
        if self._last_evicted_ts is None:
            return None
        ts_now = time.monotonic() if now is None else now
        return max(0.0, (ts_now - self._last_evicted_ts) * 1000.0)

    def lru_horizon_ms(self, now: float | None = None) -> float | None:
        """Return replica LRU eviction horizon in ms for cross-replica OTel joiner checks."""
        return self.last_eviction_age_ms(now=now)

    def session_evicted_at_ms(self, session_id: str | None) -> int | None:
        """Return epoch-ms timestamp when `session_id` blocks were evicted on this replica."""
        if not session_id:
            return None
        return self._session_evicted_wall_ms.peek(session_id)

    def time_since_eviction_ms(
        self, missed_block_hashes: Iterable[Any], now: float | None = None
    ) -> float | None:
        """Return milliseconds since the most recent eviction among `missed_block_hashes`."""
        ts_now = time.monotonic() if now is None else now
        latest_evict: float | None = None
        for blk_hash in missed_block_hashes:
            evict_ts = self._evicted_at.peek(blk_hash)
            if evict_ts is not None and (
                latest_evict is None or evict_ts > latest_evict
            ):
                latest_evict = evict_ts
        if latest_evict is None:
            return None
        return max(0.0, (ts_now - latest_evict) * 1000.0)

    def clear(self) -> None:
        self._evicted_at.clear()
        self._block_owner.clear()
        self._session_evicted_wall_ms.clear()
        self._last_evicted_ts = None

    def __len__(self) -> int:
        return len(self._evicted_at)


def compute_request_padding_tokens(
    request_scheduled_tokens: int,
    total_scheduled_tokens: int,
    padded_total_scheduled_tokens: int,
) -> int:
    """Attribute batch-level TPU static bucket padding tokens to a single request."""
    if (
        request_scheduled_tokens <= 0
        or total_scheduled_tokens <= 0
        or padded_total_scheduled_tokens <= total_scheduled_tokens
    ):
        return 0
    batch_padding = padded_total_scheduled_tokens - total_scheduled_tokens
    share = (batch_padding * request_scheduled_tokens) / total_scheduled_tokens
    return int(round(share))


def extract_hardware_span_attributes(
    *,
    kv_transfer_params: Mapping[str, Any] | None = None,
    num_external_computed_tokens: int = 0,
    time_since_eviction_ms: float | None = None,
    lru_horizon_ms: float | None = None,
    session_evicted_at_ms: int | None = None,
    tpu_shape_padding_tokens: int | None = None,
) -> dict[str, Any]:
    """Build hardware-level OTel span attributes from EngineCore outputs."""
    attrs: dict[str, Any] = {}
    if time_since_eviction_ms is not None:
        attrs["vllm.kv_cache.time_since_eviction_ms"] = round(
            float(time_since_eviction_ms), 3
        )
    if lru_horizon_ms is not None:
        attrs["vllm.kv_cache.lru_horizon_ms"] = round(float(lru_horizon_ms), 3)
    if session_evicted_at_ms is not None:
        attrs["vllm.kv_cache.session_evicted_at_ms"] = int(session_evicted_at_ms)
    if tpu_shape_padding_tokens is not None and tpu_shape_padding_tokens >= 0:
        attrs["vllm.tpu.shape_padding_tokens"] = int(tpu_shape_padding_tokens)
    if num_external_computed_tokens > 0:
        attrs["vllm.kv_cache.external_computed_tokens"] = int(
            num_external_computed_tokens
        )
    if kv_transfer_params:
        for src_key, attr_key in (
            ("dram_tokens", "vllm.kv_cache.dram_tokens"),
            ("dram_bytes", "vllm.kv_cache.dram_bytes"),
            ("dram_ms", "vllm.kv_cache.dram_ms"),
            ("ssd_tokens", "vllm.kv_cache.ssd_tokens"),
            ("ssd_bytes", "vllm.kv_cache.ssd_bytes"),
            ("ssd_ms", "vllm.kv_cache.ssd_ms"),
            ("remote_tokens", "vllm.kv_cache.remote_tokens"),
            ("kv_transfer_ms", "vllm.kv_cache.transfer_ms"),
            ("tpu_shape_padding_tokens", "vllm.tpu.shape_padding_tokens"),
            ("time_since_eviction_ms", "vllm.kv_cache.time_since_eviction_ms"),
            ("lru_horizon_ms", "vllm.kv_cache.lru_horizon_ms"),
            ("session_evicted_at_ms", "vllm.kv_cache.session_evicted_at_ms"),
        ):
            val = kv_transfer_params.get(src_key)
            if val is not None:
                attrs[attr_key] = val
    return attrs
