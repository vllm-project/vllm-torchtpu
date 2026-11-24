# SPDX-License-Identifier: Apache-2.0
"""Scan-resistant Segmented LRU (SLRU) controller for stock Raiden host stores.

Raiden's C++ ``KVCacheStore`` maintains a single host-DRAM LRU list and exposes
no ``evict()`` or custom eviction-policy hook on stock wheels. However, it
exposes ``lookup(..., pin_found=True)`` + ``release()``, which re-inserts a
local entry at the MRU head of that LRU without transferring bytes.

``SlruTracker`` layers a two-segment SLRU (``probation`` / ``protected``) over
that single LRU:

* Newly stored keys enter ``probation``.
* Reused keys (landed local load, ``touch()``, or host-resident victim re-hit)
  advance their hit counter and are promoted into ``protected`` once they reach
  ``promote_hits``. When ``protected`` exceeds ``protected_cap``, its least
  recently accessed key is demoted back to ``probation`` (keeping its current
  position in the shadow LRU order so it can age out naturally).
* Before a store job calls ``KVCacheStore.insert()`` when the host tier is near
  capacity, ``rescue_candidates()`` scans the LRU tail of the shadow order and
  returns any ``protected`` keys sitting ahead of the ``probation`` victims that
  would otherwise be evicted. The manager moves only those entries to the MRU
  head via a pin-balanced ``lookup`` + ``release`` (``Δpin == 0``), leaving
  cold one-shot ``probation`` entries at the tail to be evicted instead.
* ``rescue`` deliberately preserves each key's true recency inside ``protected``
  (it is not a cache hit), so genuinely stale entries are still the first ones
  demoted when hotter keys are promoted.

Pure Python stdlib (no ``torch`` / ``vllm`` dependency). All synchronization is
owned by the caller (``RaidenOffloadingManager._eviction_lock``). Never call
into C++ ``KVCacheStore`` while holding the lock.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

EVICT_POLICIES: tuple[str, ...] = ("lru", "slru", "slru_all")


def normalize_evict_policy(value: str | None) -> str:
    """Validate and normalize ``VLLM_TPU_OFFLOAD_TIER_EVICT_POLICY``."""
    cleaned = (value or "").strip().lower()
    if cleaned == "":
        return "lru"
    if cleaned not in EVICT_POLICIES:
        raise ValueError(
            f"VLLM_TPU_OFFLOAD_TIER_EVICT_POLICY={value!r}: expected one of "
            f"{('',) + EVICT_POLICIES}"
        )
    return cleaned


def validate_probation_frac(frac: float) -> float:
    """Validate ``VLLM_TPU_OFFLOAD_TIER_SLRU_PROBATION_FRAC`` in ``(0, 1)``."""
    val = float(frac)
    if not (0.0 < val < 1.0):
        raise ValueError(
            "VLLM_TPU_OFFLOAD_TIER_SLRU_PROBATION_FRAC must be strictly "
            f"between 0 and 1, got {frac!r}"
        )
    return val


@dataclass(frozen=True)
class SlruStats:
    """Point-in-time SLRU counters for diagnostics and telemetry."""

    probation: int
    protected: int
    protected_cap: int
    tracked: int
    promotes: int
    demotes: int
    rescues: int
    pruned: int


class SlruTracker:
    """Two-segment (probation / protected) shadow tracker over a single LRU."""

    def __init__(
        self,
        capacity: int,
        probation_frac: float = 0.25,
        *,
        promote_hits: int = 1,
    ) -> None:
        if capacity < 0:
            raise ValueError(f"SLRU capacity must be >= 0, got {capacity}")
        if promote_hits < 1:
            raise ValueError(f"SLRU promote_hits must be >= 1, got {promote_hits}")
        self.capacity = int(capacity)
        self.probation_frac = validate_probation_frac(probation_frac)
        self.promote_hits = int(promote_hits)
        min_probation = max(1, int(round(self.capacity * self.probation_frac)))
        self.protected_cap = max(0, self.capacity - min_probation)

        # Ordered oldest -> newest matching the physical C++ LRU order.
        self._lru: OrderedDict[Any, None] = OrderedDict()
        # key -> reuse hits accumulated while in probation.
        self._probation: OrderedDict[Any, int] = OrderedDict()
        # Ordered oldest real access -> newest real access (size <= protected_cap).
        self._protected: OrderedDict[Any, None] = OrderedDict()

        self.promotes = 0
        self.demotes = 0
        self.rescues = 0
        self.pruned = 0

    def __len__(self) -> int:
        return len(self._lru)

    def contains(self, key: Any) -> bool:
        return key in self._lru

    def is_protected(self, key: Any) -> bool:
        return key in self._protected

    def is_probation(self, key: Any) -> bool:
        return key in self._probation

    def _trim_overflow(self) -> None:
        """Bound shadow tracking if eviction callbacks are delayed or unavailable."""
        limit = max(self.capacity * 2, 16) if self.capacity > 0 else 0
        while len(self._lru) > limit:
            oldest, _ = self._lru.popitem(last=False)
            self._probation.pop(oldest, None)
            self._protected.pop(oldest, None)
            self.pruned += 1

    def record_insert(self, key: Any) -> None:
        """Record a key admitted to the store (new keys enter probation at MRU)."""
        if self.capacity == 0:
            return
        if key in self._protected:
            self._lru[key] = None
            self._lru.move_to_end(key)
            return
        if key in self._probation:
            self._probation.move_to_end(key)
            self._lru[key] = None
            self._lru.move_to_end(key)
            return
        self._probation[key] = 0
        self._lru[key] = None
        self._trim_overflow()

    def record_hit(self, key: Any) -> bool:
        """Record a reuse hit (local load / touch / resident re-eviction).

        Returns True if the key transitioned from probation (or untracked) into
        the protected segment.
        """
        if self.capacity == 0:
            return False
        if key in self._protected:
            self._protected.move_to_end(key)
            self._lru[key] = None
            self._lru.move_to_end(key)
            return False
        hits = self._probation.get(key, 0) + 1
        if hits >= self.promote_hits and self.protected_cap > 0:
            self._probation.pop(key, None)
            self._protected[key] = None
            self._protected.move_to_end(key)
            self.promotes += 1
            while len(self._protected) > self.protected_cap:
                demoted, _ = self._protected.popitem(last=False)
                # Keep demoted entry in place inside self._lru so its physical
                # position continues drifting toward the eviction tail.
                self._probation[demoted] = 0
                self.demotes += 1
            self._lru[key] = None
            self._lru.move_to_end(key)
            self._trim_overflow()
            return True
        self._probation[key] = hits
        self._probation.move_to_end(key)
        self._lru[key] = None
        self._lru.move_to_end(key)
        self._trim_overflow()
        return False

    def record_lru_refresh(self, key: Any) -> None:
        """Mirror a C++ ``release()`` that moved an entry to MRU without a real hit."""
        if key in self._lru:
            self._lru.move_to_end(key)

    def demote(self, key: Any) -> bool:
        """Demote a protected key back to probation without touching its LRU position."""
        if key not in self._protected:
            return False
        del self._protected[key]
        self._probation[key] = 0
        self.demotes += 1
        return True

    def on_evict(self, key: Any) -> None:
        """Remove a key reported evicted by ``KVCacheStore``."""
        self._protected.pop(key, None)
        self._probation.pop(key, None)
        self._lru.pop(key, None)

    def mark_missing(self, key: Any) -> None:
        """Drop a key observed missing during a rescue probe."""
        if key in self._lru or key in self._protected or key in self._probation:
            self.pruned += 1
        self.on_evict(key)

    def mark_rescued(self, key: Any) -> None:
        """Record that ``key`` was moved to C++ MRU via a pin-balanced touch.

        Does NOT update ``_protected`` recency order so stale protected keys
        remain first in line for demotion when newer keys are promoted.
        """
        if key in self._lru:
            self._lru.move_to_end(key)
        self.rescues += 1

    def rescue_candidates(
        self,
        needed: int,
        *,
        exclude: Iterable[Any] = (),
        max_rescues: int | None = None,
    ) -> list[Any]:
        """Return protected keys sitting at the LRU tail ahead of ``needed``
        evictable probation victims.

        Returns an empty list while the store still has headroom below
        ``capacity`` or when no evictable probation victim exists (so rescuing
        would leave the relative eviction order unchanged).
        """
        if needed <= 0 or self.protected_cap <= 0 or not self._protected:
            return []
        deficit = len(self._lru) + needed - self.capacity
        if deficit <= 0:
            return []
        excluded = exclude if isinstance(exclude, (set, frozenset)) else set(exclude)
        if not any(k not in excluded for k in self._probation):
            return []
        rescue: list[Any] = []
        victims_seen = 0
        for key in self._lru:
            if victims_seen >= deficit:
                break
            if key in excluded:
                continue
            if key in self._protected:
                if max_rescues is not None and len(rescue) >= max_rescues:
                    break
                rescue.append(key)
            else:
                victims_seen += 1
        return rescue

    def stats(self) -> SlruStats:
        return SlruStats(
            probation=len(self._probation),
            protected=len(self._protected),
            protected_cap=self.protected_cap,
            tracked=len(self._lru),
            promotes=self.promotes,
            demotes=self.demotes,
            rescues=self.rescues,
            pruned=self.pruned,
        )
