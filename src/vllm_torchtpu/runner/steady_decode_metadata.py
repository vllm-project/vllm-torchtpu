# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_STEADY_DECODE_METADATA: incremental attention metadata for steady
decode.

At low concurrency the scheduled request set, its order and every request's
token count (1 + drafts) stay the same for hundreds of consecutive decode
steps; only the requests' computed-token counts advance by their accepted
tokens. `TPUModelRunner._prepare_inputs` still rebuilt every batch-shape
product from scratch each step. With the flag on it keys each single-chunk
pure-decode step on everything those products depend on
(`steady_decode_key`) and, while the key repeats, reuses them:

* host arrays that depend only on the per-request token counts: the general
  path's `req_indices` / per-token `arange`, `query_start_loc` (whose host
  buffer then keeps the previous, identical values) and the per-request
  draft counts;
* device uploads of values that depend only on those counts and on the
  previous step's request positions: `query_start_loc` and the async
  rejected-token subtract indices;
* the staged walk's unified-pool mamba state/checkpoint indices while no
  request can cross a mamba block (`mamba_offsets_stable`): they are a
  function of the block tables (unchanged: the staged upload was reused) and
  of `(seq_lens - 1) // block_size`, and a step that keeps every row's block
  yields the same values.

What changes every step (positions, input ids, seq_lens and their uploads,
the async rejected-token subtract itself, draft token ids) is recomputed as
before, so every tensor is bitwise identical to the full rebuild. Any step
whose key differs (request add/finish/reorder/preempt, a prefill piece, a
different token count or padding bucket, PCP/mrope/LoRA, multi-chunk) is a
miss: it runs the full rebuild and re-seeds the cache.

Everything here is plain numpy / Python so the bookkeeping is unit-testable
without a device (tests/runner/test_steady_decode_metadata.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def steady_decode_key(
    req_ids: tuple,
    num_scheduled: np.ndarray,
    num_draft: np.ndarray | None,
    *,
    start_index: int,
    batch_num_reqs: int,
    use_max_model_len: bool,
    target_num_reqs: int,
    padded_num_reqs: int,
    rpa_num_decode_reqs: int,
    num_windowed_reqs: int,
) -> tuple | None:
    """The cache key of one `_prepare_inputs` chunk, or None when the chunk
    is not a steady-decode candidate at all.

    Candidates are single-chunk (the chunk starts at 0 and covers the whole
    persistent batch) and prefill-free: every request schedules exactly its
    drafts plus one token (a plain decode with no drafts schedules one).
    """
    n = len(req_ids)
    if start_index != 0 or n == 0 or n != batch_num_reqs:
        return None
    if num_draft is None:
        if int(num_scheduled.max()) != 1:
            return None
        draft_bytes = b""
    else:
        if not np.array_equal(num_scheduled, num_draft + 1):
            return None
        draft_bytes = num_draft.tobytes()
    return (
        req_ids,
        num_scheduled.tobytes(),
        draft_bytes,
        use_max_model_len,
        target_num_reqs,
        padded_num_reqs,
        rpa_num_decode_reqs,
        num_windowed_reqs,
    )


def scheduler_output_is_quiet(scheduler_output: Any) -> bool:
    """No request admitted, resumed, preempted or finished by this schedule.

    The batch key alone already changes on add/finish/reorder; this also
    turns away a step that swaps a request out and back in under the same id
    (its block table rows and computed-token count are replaced).
    """
    if getattr(scheduler_output, "scheduled_new_reqs", None):
        return False
    if getattr(scheduler_output, "finished_req_ids", None):
        return False
    if getattr(scheduler_output, "preempted_req_ids", None):
        return False
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    if cached is not None:
        if getattr(cached, "resumed_req_ids", None):
            return False
        resumed = getattr(cached, "resumed_from_preemption", None)
        if resumed is not None and any(resumed):
            return False
    return True


def general_path_indices(
    arange_np: np.ndarray, start_index: int, num_scheduled: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`_prepare_inputs`'s mixed-path `req_indices` and batched `arange`
    (e.g. [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]); identical math."""
    n = len(num_scheduled)
    req_indices = np.repeat(arange_np[start_index : start_index + n], num_scheduled)
    arange = np.concatenate([arange_np[:k] for k in num_scheduled])
    return req_indices, arange


def walk_seq_bounds(
    seq_lens_np: np.ndarray, num_reqs: int, target_num_reqs: int, margin: int
) -> tuple[np.ndarray, np.ndarray, bytes]:
    """Bounds of the device `seq_lens` the staged walk will read.

    The host fills `seq_lens_np` optimistically; the async spec path then
    subtracts the previous step's rejected count (at most its drafts, so
    below `margin`) on device. Real rows thus lie in
    `[seq - margin, seq]`; padded rows are not subtracted, so their exact
    host values are returned as bytes.
    """
    hi = seq_lens_np[:num_reqs].astype(np.int64)
    lo = hi - margin
    tail = seq_lens_np[num_reqs:target_num_reqs].tobytes()
    return lo, hi, tail


def _row_offsets(x: np.ndarray, num_blocks: int, block_size: int) -> np.ndarray:
    # Same as the walk: clamp((seq_lens - 1) // bs, 0, t - 1).
    return np.clip((x - 1) // block_size, 0, num_blocks - 1)


def mamba_offsets_stable(
    prev: tuple[np.ndarray, np.ndarray, bytes],
    cur: tuple[np.ndarray, np.ndarray, bytes],
    mamba_geom: tuple,
) -> bool:
    """True when every row's walk products (`active = seq_lens > 0` and the
    clamped `(seq_lens - 1) // block_size` offset, per (num_blocks,
    block_size) class) are the same for any device `seq_lens` inside `prev`
    and any inside `cur` (see `walk_seq_bounds`).

    The offset is monotone in `seq_lens`, so it is constant on the union
    interval iff it agrees at the union's two ends; `active` is constant iff
    the union stays >= 1 (rows are real, so it cannot be all zero).
    """
    prev_lo, prev_hi, prev_tail = prev
    lo, hi, tail = cur
    if prev_tail != tail or prev_lo.shape != lo.shape:
        return False
    union_lo = np.minimum(prev_lo, lo)
    union_hi = np.maximum(prev_hi, hi)
    if union_lo.size and int(union_lo.min()) < 1:
        return False
    for t, bs in {(t, bs) for _, t, bs in mamba_geom}:
        if not np.array_equal(
            _row_offsets(union_lo, t, bs), _row_offsets(union_hi, t, bs)
        ):
            return False
    return True


@dataclass
class SteadyDecodeEntry:
    """The previous chunk's reusable products, valid while `key` repeats."""

    key: tuple
    # General (multi-token) path only; None for 1-token decode steps.
    req_indices: np.ndarray | None = None
    arange: np.ndarray | None = None
    num_draft: np.ndarray | None = None
    # Device uploads keyed further on the layout (cache_key, padded tokens).
    layout_key: tuple | None = None
    query_start_loc: Any = None
    # Async rejected-token subtract: (positions of these requests in the
    # previous step's results) -> (seq_idx upload, pos_idx upload).
    subtract_key: tuple | None = None
    subtract_idx: tuple | None = None


@dataclass
class SteadyDecodeMetadataCache:
    entry: SteadyDecodeEntry | None = None
    hits: int = 0
    misses: int = 0
    walk_skips: int = 0
    walks: int = 0
    stats: dict = field(default_factory=dict)

    def lookup(self, key: tuple | None) -> SteadyDecodeEntry | None:
        if key is not None and self.entry is not None and self.entry.key == key:
            self.hits += 1
            return self.entry
        self.misses += 1
        self.entry = None
        return None

    def seed(self, key: tuple | None, **parts: Any) -> SteadyDecodeEntry | None:
        self.entry = None if key is None else SteadyDecodeEntry(key=key, **parts)
        return self.entry

    def invalidate(self) -> None:
        self.entry = None


@dataclass
class SteadyWalkProducts:
    """VLLM_TPU_STEADY_DECODE_METADATA: the last target walk's mamba products
    (`_issue_walk_products`), reusable while the staged upload key repeats
    and `mamba_offsets_stable` holds against `bounds`."""

    stage_key: tuple
    geometry: tuple
    bounds: tuple[np.ndarray, np.ndarray, bytes]
    staged_block_tables: dict
    products: tuple
    n_tables: int
