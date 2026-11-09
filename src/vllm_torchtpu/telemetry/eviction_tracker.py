# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""HBM block eviction age tracking and hardware wastage attribute helpers."""

import time
from collections.abc import Iterable, Mapping
from typing import Any

from vllm_torchtpu.telemetry.token_breakdown import BoundedLRU


class TpuPaddingCounterTracker:
    """Tracks server-level TPU static bucket padding across prompt and reply rounds."""

    def __init__(self) -> None:
        self.prompt_rounds: int = 0
        self.prompt_real_tokens: int = 0
        self.prompt_filler_tokens: int = 0
        self.reply_rounds: int = 0
        self.reply_real_tokens: int = 0
        self.reply_filler_tokens: int = 0

    def record_round(
        self,
        scheduled_tokens: int,
        padded_tokens: int,
        *,
        is_prompt_round: bool,
    ) -> None:
        """Accumulate real and filler tokens for one TPU execution round."""
        if scheduled_tokens <= 0 or padded_tokens <= 0:
            return
        real = int(scheduled_tokens)
        filler = max(0, int(padded_tokens) - real)
        if is_prompt_round:
            self.prompt_rounds += 1
            self.prompt_real_tokens += real
            self.prompt_filler_tokens += filler
        else:
            self.reply_rounds += 1
            self.reply_real_tokens += real
            self.reply_filler_tokens += filler

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_rounds": self.prompt_rounds,
            "prompt_real_tokens": self.prompt_real_tokens,
            "prompt_filler_tokens": self.prompt_filler_tokens,
            "reply_rounds": self.reply_rounds,
            "reply_real_tokens": self.reply_real_tokens,
            "reply_filler_tokens": self.reply_filler_tokens,
        }

    def update_from_dict(self, data: Mapping[str, Any]) -> None:
        self.prompt_rounds = max(
            self.prompt_rounds, int(data.get("prompt_rounds", 0))
        )
        self.prompt_real_tokens = max(
            self.prompt_real_tokens, int(data.get("prompt_real_tokens", 0))
        )
        self.prompt_filler_tokens = max(
            self.prompt_filler_tokens, int(data.get("prompt_filler_tokens", 0))
        )
        self.reply_rounds = max(
            self.reply_rounds, int(data.get("reply_rounds", 0))
        )
        self.reply_real_tokens = max(
            self.reply_real_tokens, int(data.get("reply_real_tokens", 0))
        )
        self.reply_filler_tokens = max(
            self.reply_filler_tokens, int(data.get("reply_filler_tokens", 0))
        )

    def clear(self) -> None:
        self.prompt_rounds = 0
        self.prompt_real_tokens = 0
        self.prompt_filler_tokens = 0
        self.reply_rounds = 0
        self.reply_real_tokens = 0
        self.reply_filler_tokens = 0


class BlockEvictionTracker:
    """Tracks HBM block evictions, replica LRU horizon, and per-session eviction timestamps."""

    def __init__(self, maxsize: int = 131072, session_maxsize: int = 65536) -> None:
        self._evicted_at: BoundedLRU[Any, float] = BoundedLRU(maxsize)
        self._block_owner: BoundedLRU[Any, str] = BoundedLRU(maxsize)
        self._session_evicted_ts: BoundedLRU[str, float] = BoundedLRU(
            session_maxsize
        )
        self._session_evicted_wall_ms: BoundedLRU[str, int] = BoundedLRU(
            session_maxsize
        )
        self._dirty_sessions: set[str] = set()
        self._last_evicted_ts: float | None = None
        self._last_evicted_wall_ms: int | None = None

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
        *,
        session_id: str | None = None,
    ) -> None:
        if block_hash is None and not session_id:
            return
        ts = time.monotonic() if now is None else now
        evict_wall = (
            int(time.time() * 1000) if wall_time_ms is None else int(wall_time_ms)
        )
        if block_hash is not None:
            self._evicted_at.put(block_hash, ts)
        self._last_evicted_ts = ts
        self._last_evicted_wall_ms = evict_wall

        owner = session_id or (
            self._block_owner.peek(block_hash) if block_hash is not None else None
        )
        if owner is not None:
            self._session_evicted_ts.put(owner, ts)
            self._session_evicted_wall_ms.put(owner, evict_wall)
            self._dirty_sessions.add(owner)

    def last_eviction_age_ms(self, now: float | None = None) -> float | None:
        """Return milliseconds since the most recent block eviction on this pool."""
        if self._last_evicted_ts is None:
            return None
        ts_now = time.monotonic() if now is None else now
        return max(0.0, (ts_now - self._last_evicted_ts) * 1000.0)

    def session_eviction_age_ms(
        self, session_id: str | None, now: float | None = None
    ) -> float | None:
        """Return milliseconds since `session_id` blocks were last evicted."""
        if not session_id:
            return None
        evict_ts = self._session_evicted_ts.peek(session_id)
        if evict_ts is None:
            return None
        ts_now = time.monotonic() if now is None else now
        return max(0.0, (ts_now - evict_ts) * 1000.0)

    def lru_horizon_ms(
        self, now: float | None = None, *, session_id: str | None = None
    ) -> float | None:
        """Return per-session (or fallback pool) LRU eviction horizon in ms."""
        if session_id is not None:
            return self.session_eviction_age_ms(session_id, now=now)
        return self.last_eviction_age_ms(now=now)

    def session_evicted_at_ms(self, session_id: str | None) -> int | None:
        """Return epoch-ms timestamp when `session_id` blocks were evicted on this replica."""
        if not session_id:
            return None
        return self._session_evicted_wall_ms.peek(session_id)

    def time_since_eviction_ms(
        self,
        missed_block_hashes: Iterable[Any],
        now: float | None = None,
        *,
        session_id: str | None = None,
    ) -> float | None:
        """Return milliseconds since the most recent eviction among `missed_block_hashes` or `session_id`."""
        ts_now = time.monotonic() if now is None else now
        latest_evict: float | None = None
        for blk_hash in missed_block_hashes:
            evict_ts = self._evicted_at.peek(blk_hash)
            if evict_ts is not None and (
                latest_evict is None or evict_ts > latest_evict
            ):
                latest_evict = evict_ts
        if latest_evict is None and session_id:
            latest_evict = self._session_evicted_ts.peek(session_id)
        if latest_evict is None:
            return None
        return max(0.0, (ts_now - latest_evict) * 1000.0)

    def pop_dirty_session_state(self) -> dict[str, list[float | int]]:
        """Pop and return only sessions whose eviction timestamp changed since the last flush."""
        if not self._dirty_sessions:
            return {}
        dirty = tuple(self._dirty_sessions)
        self._dirty_sessions.clear()
        out: dict[str, list[float | int]] = {}
        for sid in dirty:
            ts = self._session_evicted_ts.peek(sid)
            wall_ms = self._session_evicted_wall_ms.peek(sid)
            if ts is not None and wall_ms is not None:
                out[sid] = [float(ts), int(wall_ms)]
        return out

    def export_recent_session_state(
        self, limit: int = 1024
    ) -> dict[str, list[float | int]]:
        """Export up to `limit` most recently evicted sessions for sync log compaction."""
        items = list(self._session_evicted_ts._data.items())
        if limit > 0 and len(items) > limit:
            items = items[-limit:]
        out: dict[str, list[float | int]] = {}
        for sid, ts in items:
            wall_ms = self._session_evicted_wall_ms.peek(sid)
            if wall_ms is not None:
                out[sid] = [float(ts), int(wall_ms)]
        return out

    def export_session_state(self) -> dict[str, list[float | int]]:
        """Serialize per-session eviction state for once-per-step IPC sync."""
        out: dict[str, list[float | int]] = {}
        for sid, ts in self._session_evicted_ts._data.items():
            wall_ms = self._session_evicted_wall_ms.peek(sid)
            if wall_ms is not None:
                out[sid] = [float(ts), int(wall_ms)]
        return out

    def import_session_state(self, data: Mapping[str, Any]) -> None:
        """Merge per-session eviction state synced from EngineCore."""
        for sid, pair in data.items():
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            ts = float(pair[0])
            wall_ms = int(pair[1])
            cur_ts = self._session_evicted_ts.peek(sid)
            if cur_ts is None or ts > cur_ts:
                self._session_evicted_ts.put(sid, ts)
                self._session_evicted_wall_ms.put(sid, wall_ms)
            if self._last_evicted_ts is None or ts > self._last_evicted_ts:
                self._last_evicted_ts = ts
                self._last_evicted_wall_ms = wall_ms

    def clear(self) -> None:
        self._evicted_at.clear()
        self._block_owner.clear()
        self._session_evicted_ts.clear()
        self._session_evicted_wall_ms.clear()
        self._dirty_sessions.clear()
        self._last_evicted_ts = None
        self._last_evicted_wall_ms = None

    def __len__(self) -> int:
        return len(self._evicted_at)


def compute_tpu_bucket_padding_tokens(scheduled_tokens: int) -> int:
    """Return static TPU power-of-2 bucket padding tokens (min bucket 16) for `scheduled_tokens`."""
    if scheduled_tokens <= 0:
        return 0
    padded = max(16, 1 << (scheduled_tokens - 1).bit_length())
    return max(0, padded - scheduled_tokens)


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
        attrs.setdefault(
            "vllm.kv_cache.dram_tokens", int(num_external_computed_tokens)
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
