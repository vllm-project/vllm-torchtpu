# SPDX-License-Identifier: Apache-2.0
"""Supersede-aware recurrent-state checkpoints in the host tier (VLLM_TPU_OFFLOAD_TIER_DEMOTE_SUPERSEDED).

Port of the Qwen3.5 tree's state demotion (code/vllm-torchtpu-q35c4 5a1c826860:
``raiden_connector._init_state_demotion`` / ``_demote_superseded_states`` and ``raiden_store.demote``) to the Q38
victim policy and size classes (design: reports/lane_pstep_tier.md section 7 item 3 "(b)").

A conversation's next turn resumes from the newest recurrent-state checkpoint (GDN A / B / C, PLE) of its prefix chain.
An older boundary of the same chain serves only a fork that resumes there. Under the victim policy
(offload/victim.py) the tier stores a checkpoint when HBM evicts it, and HBM frees the older boundaries of a request
first (``VLLM_TPU_MAMBA_PIN_HIT_CHECKPOINT``), so the state classes fill with superseded checkpoints. With the flag:

- On request finish (``request_finished``, before its blocks are freed) the tracker finds the request's kept boundary:
  the newest boundary of its prompt whose every state group is resident, in HBM (the block pool's cached hashes) or in
  the tier. Each older boundary whose state HBM still holds is recorded as superseded by the kept one (a bounded map,
  ``MAP_CAP`` entries, oldest dropped), and the older boundaries' state keys the tier holds are DEMOTED: they still
  answer lookups, but their class evicts them before any live entry when it needs room
  (``RaidenOffloadingManager.demote``).
- When HBM evicts a state block (the victim store, ``TPURaidenOffloadingScheduler._store_victims``) whose boundary is
  recorded as superseded and the later boundary is resident (HBM, the tier, or stored in the same batch; a skipped
  later boundary is followed to its own successor), the block is not stored.

Counters: ``ck_skip_superseded`` (victim stores skipped), ``ck_demote_superseded`` (tier keys demoted at finish),
``ck_sup_finished`` (finishes looked at). Attention blocks are never touched. The cost, priced by the PSTEP-TIER
replay: a fork that resumes from a mid-chain boundary finds no checkpoint there.

This module imports nothing from vLLM: the connector injects the key maker, the tier probe and the demote call, and
hands over the engine's block pool (captured from the victim tap's first eviction).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from typing import Any

# Most superseded boundaries remembered (one entry a boundary hash; the oldest is dropped beyond it). A boundary is
# recorded only while HBM holds its state, so the live set is bounded by the HBM pool; this cap is the hard bound.
MAP_CAP = 1 << 17
# Most successor hops a victim's check follows (a later boundary itself superseded and skipped in the same batch).
MAX_HOPS = 8
COUNTERS = ("ck_sup_finished", "ck_skip_superseded", "ck_demote_superseded")


class SupersedeTracker:
    """Scheduler-side chain map: older boundary hash -> the newest kept boundary hash of its chain."""

    def __init__(
        self,
        state_groups: Sequence[int],
        hashes_per_chunk: int,
        make_key: Callable[[bytes, int], bytes],
        tier_holds: Callable[[bytes], bool],
        demote: Callable[[list], int],
        cap: int = MAP_CAP,
    ):
        if not state_groups:
            raise ValueError("supersede: no recurrent-state group is offloaded")
        if hashes_per_chunk < 1 or cap < 1:
            raise ValueError(f"supersede: bad hashes_per_chunk {hashes_per_chunk} or cap {cap}")
        self._groups = tuple(sorted(state_groups))
        self._group_set = frozenset(self._groups)
        self._step = int(hashes_per_chunk)
        self._make_key = make_key
        self._tier_holds = tier_holds
        self._demote = demote
        self._cap = int(cap)
        self._later: OrderedDict[bytes, bytes] = OrderedDict()
        # The engine's BlockPool (HBM prefix cache), set by the connector from the victim tap's first eviction.
        self.pool: Any = None
        self.counts = dict.fromkeys(COUNTERS, 0)

    # -- residency ----------------------------------------------------------------------------------------------

    def hbm_has(self, block_hash: bytes, group: int) -> bool:
        pool = self.pool
        if pool is None:
            return False
        return pool.get_cached_block(block_hash, [group]) is not None

    def resident(self, block_hash: bytes, batch: frozenset = frozenset()) -> bool:
        """Every state group's checkpoint at the boundary is in HBM, in the tier, or stored in this victim batch."""
        for group in self._groups:
            key = self._make_key(block_hash, group)
            if key in batch or self.hbm_has(block_hash, group) or self._tier_holds(key):
                continue
            return False
        return True

    def boundary_hashes(self, block_hashes: Sequence[bytes]) -> list[bytes]:
        """The hash of each complete boundary of a request's chain, in order (the key's hash a state group uses)."""
        step = self._step
        return [block_hashes[(chunk + 1) * step - 1] for chunk in range(len(block_hashes) // step)]

    # -- request finish -----------------------------------------------------------------------------------------

    def on_finished(self, block_hashes: Sequence[bytes]) -> tuple[int, int]:
        """Records the older boundaries HBM holds as superseded by the kept (newest resident) one, and demotes the older
        state keys the tier holds. Returns (boundaries recorded, keys demoted)."""
        self.counts["ck_sup_finished"] += 1
        hashes = self.boundary_hashes(block_hashes)
        kept = None
        for chunk in reversed(range(len(hashes))):
            if self.resident(hashes[chunk]):
                kept = chunk
                break
        if not kept:
            return 0, 0
        newest = hashes[kept]
        recorded, stale = 0, []
        for chunk in range(kept):
            block_hash = hashes[chunk]
            if any(self.hbm_has(block_hash, group) for group in self._groups):
                self._later[block_hash] = newest
                self._later.move_to_end(block_hash)
                recorded += 1
            stale.extend(
                key
                for key in (self._make_key(block_hash, group) for group in self._groups)
                if self._tier_holds(key)
            )
        while len(self._later) > self._cap:
            self._later.popitem(last=False)
        demoted = self._demote(stale) if stale else 0
        self.counts["ck_demote_superseded"] += demoted
        return recorded, demoted

    # -- victim store -------------------------------------------------------------------------------------------

    def superseded_by(self, block_hash: bytes, batch: frozenset) -> bytes | None:
        """The resident later boundary of this boundary's chain, or None (store it)."""
        later = self._later.get(block_hash)
        hops = 0
        while later is not None and hops < MAX_HOPS:
            if self.resident(later, batch):
                return later
            later = self._later.get(later)
            hops += 1
        return None

    def filter_victims(self, keep: Iterable[tuple[Any, int]], group_of: Callable[[Any], int]) -> tuple[list, int]:
        """The victims to store: a state block whose later boundary is resident is dropped. Returns (kept, skipped)."""
        keep = list(keep)
        batch = frozenset(bytes(key) for key, _ in keep)
        out, skipped = [], 0
        for key, block_id in keep:
            if group_of(key) in self._group_set and self.superseded_by(bytes(key)[:-4], batch) is not None:
                skipped += 1
                continue
            out.append((key, block_id))
        self.counts["ck_skip_superseded"] += skipped
        return out, skipped

    def map_size(self) -> int:
        return len(self._later)

    def telemetry(self) -> str:
        return f"supersede=map:{len(self._later)},map_cap:{self._cap},pool:{int(self.pool is not None)}"
