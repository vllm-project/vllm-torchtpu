# SPDX-License-Identifier: Apache-2.0
"""q38 L4 VPSHOST: host-only levers on the VPS verify path of the decode step.

Each lever is an l6_ab class (``core/l6_ab.py`` LEVERS, read once per worker
step, never at trace time). None of them changes a traced program or the
inputs a compiled program receives, so each env var is in the compile-key
ignore list (``compilation/tpu_compiler.py``).

* ``sampfb`` (``VLLM_TPU_L4_SAMP_FB_ORDER``, T-2 stage 0, inline in
  ``TPUModelRunner.execute_model``): on the VPS fallback path the fallback
  decides whether a chunk takes the VPS verify, overwriting the top-k
  candidate test, so that test is skipped.
* ``sampsnap`` (``VLLM_TPU_L4_SAMP_SNAP``, T-2 stage 1, ``SampSnap``): the
  VPS verify's host-built inputs (the fallback row count F, the three padded
  sampling-parameter uploads and the num_active upload) are kept per batch
  snapshot and reused while the batch's requests and their parameters stay
  the same.
* ``statsasync`` (``VLLM_TPU_L4_VPS_STATS_ASYNC``, T-1(i), ``AsyncCounts``):
  the VPS_FALLBACK row counts, read every ``VLLM_TPU_VPS_STATS_EVERY`` chunks,
  are copied to the host without blocking the host on the device; a later
  chunk logs the same line once the copy has landed.
"""

from __future__ import annotations

import collections
import os
from typing import Any

import torch

from vllm_torchtpu.core import l6_ab as _l6_ab
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

SNAP_CHECK_ENV = "VLLM_TPU_L4_SAMP_SNAP_CHECK"
MISS = object()


def _event() -> Any:
    """A device event recorded behind everything enqueued so far (torch_tpu)."""
    ev = torch.tpu.Event()
    ev.record()
    return ev


class AsyncCounts:
    """T-1(i) STATS-ASYNC: small device counter tensors copied to the host
    without blocking the host on the device.

    ``issue(tag, dev)`` starts a non-blocking device-to-host copy of ``dev`` and
    records an event behind it (the sampled-token copy in
    ``tpu_runner_async_output.py`` uses the same pair). ``ready()`` returns, in
    issue order, ``(tag, values)`` for every copy whose event has completed and
    stops at the first one still in flight; ``ready(wait=True)`` waits for all.
    The device tensor stays referenced until its copy is read.
    """

    __slots__ = ("pending",)

    def __init__(self) -> None:
        self.pending: collections.deque = collections.deque()

    def issue(self, tag: Any, dev: torch.Tensor) -> None:
        host = dev.to("cpu", non_blocking=True)
        self.pending.append((tag, dev, host, _event()))

    def ready(self, wait: bool = False) -> list[tuple[Any, list]]:
        out = []
        while self.pending:
            tag, _dev, host, ev = self.pending[0]
            if wait:
                ev.synchronize()
            elif not ev.query():
                break
            self.pending.popleft()
            out.append((tag, host.tolist()))
        return out


def _check_allowed() -> bool:
    # q38-l6-next bounds every CHECK mode by VLLM_TPU_L6_CHECK_STEPS through
    # l6_ab.check_allowed(); trees without it (STACK-D v3) allow every check.
    return getattr(_l6_ab, "check_allowed", lambda: True)()


class _Entry:
    __slots__ = ("rows", "params", "fp")

    def __init__(self) -> None:
        self.rows: dict = {}      # (fb_max, tiegap) -> F (int, or None = today's path)
        self.params: dict = {}    # (padded rows, dtype, device) -> (temps, top_k, top_p)
        self.fp: Any = None       # parameter bytes at build time (check mode only)


def _fingerprint(batch: Any, s: int, e: int) -> tuple:
    return (batch.temperature_cpu_tensor[s:e].numpy().tobytes(),
            batch.top_k_cpu_tensor[s:e].numpy().tobytes(),
            batch.top_p_cpu_tensor[s:e].numpy().tobytes())


class SampSnap:
    """T-2 stage 1 SAMP-SNAP: per-batch snapshot of the VPS verify's host-built inputs.

    Ports and extends IX a46d91d20 (Jordan Dotzel; port by Shrihari Murali), whose
    ``_padded_sampling_params`` cache keyed the padded parameter uploads by their
    values. Here the key is the chunk's request ids, so a hit skips the CPU
    staging and the value key altogether, and the snapshot also holds the chunk's
    fallback row count F (``_vps_fallback_rows``) and the num_active upload.

    Key: ``(tuple(req_ids[s:e]), s, e)``, then ``(padded rows, dtype, device)`` for
    the parameters and ``(fb_max, tiegap)`` for F; num_active by its value.

    Validity: a request's sampling rows are written only by
    ``InputBatch.add_request`` (vLLM v1 ``gpu_input_batch.py``; ``swap_states`` and
    ``condense`` move rows together with their request ids, and the runner never
    writes them). So the snapshot wraps the live batch's ``add_request`` and drops
    every entry on each call, and a new InputBatch object (KV-cache
    re-initialisation) is wrapped and drops them too. With
    ``VLLM_TPU_L4_SAMP_SNAP_CHECK = N`` (> 0) every N-th hit compares the rows'
    current bytes with the bytes the entry was built from (host only, no device
    read; F and parameter lookups both count as hits); a mismatch is logged as an
    ERROR, counted (``sampsnap.stale``) and rebuilt from the batch. The check is
    bounded by ``l6_ab.check_allowed()`` where the tree has it (q38-l6-next).
    """

    def __init__(self, cap: int = 64) -> None:
        self.cap = cap
        raw = os.environ.get(SNAP_CHECK_ENV, "").strip()
        self.check_every = int(raw) if raw else 0
        self._batch: Any = None
        self._d: dict = {}
        self._nact: dict = {}
        self.gen = 0
        self.hits = 0
        self.checks = 0
        self.stale = 0

    # ---- validity ----
    def invalidate(self) -> None:
        self.gen += 1
        self._d.clear()

    def _watch(self, batch: Any) -> None:
        if batch is self._batch:
            return
        orig = batch.add_request
        snap = self

        def snap_add_request(*args, **kwargs):
            snap.invalidate()
            return orig(*args, **kwargs)

        batch.add_request = snap_add_request
        self._batch = batch
        self.invalidate()

    def _entry(self, batch: Any, s: int, e: int) -> _Entry:
        self._watch(batch)
        key = (tuple(batch.req_ids[s:e]), s, e)
        ent = self._d.get(key)
        if ent is not None and self.check_every:
            self.hits += 1
            if self.hits % self.check_every == 0 and _check_allowed():
                ent = self._check(batch, s, e, ent)
        if ent is None:
            if len(self._d) >= self.cap:
                self._d.pop(next(iter(self._d)))
            ent = self._d[key] = _Entry()
            if self.check_every:
                ent.fp = _fingerprint(batch, s, e)
        return ent

    def _check(self, batch: Any, s: int, e: int, ent: _Entry) -> _Entry | None:
        """Check mode: `ent` if the rows still hold the bytes it was built from; else None (every entry dropped)."""
        self.checks += 1
        _l6_ab.note("sampsnap", "check")
        if _fingerprint(batch, s, e) == ent.fp:
            return ent
        self.stale += 1
        _l6_ab.note("sampsnap", "stale")
        logger.error("q38 SAMPSNAP stale snapshot for rows [%d, %d) (check %d, stale %d): rebuilt from the batch",
                     s, e, self.checks, self.stale)
        self.invalidate()
        return None

    # ---- the three host-built inputs ----
    @staticmethod
    def _fbkey(runner: Any) -> tuple[int, int]:
        return (runner._vps_fb_max, runner._vps_tiegap)

    def get_rows(self, runner: Any, s: int, e: int) -> Any:
        """The chunk's fallback row count F from the snapshot, or MISS."""
        return self._entry(runner.input_batch, s, e).rows.get(self._fbkey(runner), MISS)

    def put_rows(self, runner: Any, s: int, e: int, rows: int | None) -> None:
        self._entry(runner.input_batch, s, e).rows[self._fbkey(runner)] = rows

    def params(self, runner: Any, s: int, e: int, padded: int, dtype: torch.dtype,
               device: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The padded (temperature, top_k, top_p) uploads `_padded_sampling_params` returns, from the snapshot."""
        ent = self._entry(runner.input_batch, s, e)
        key = (padded, dtype, device)
        got = ent.params.get(key)
        if got is None:
            got = ent.params[key] = runner._padded_sampling_params(s, e, padded, dtype, device)
            _l6_ab.note("sampsnap", "miss")
        else:
            _l6_ab.note("sampsnap", "hit")
        return got

    def num_active(self, runner: Any, n: int) -> torch.Tensor:
        """The [n] int32 upload `_vps_verify_fb` builds, one per value of n."""
        got = self._nact.get(n)
        if got is None:
            if len(self._nact) >= 4 * self.cap:
                self._nact.clear()
            got = self._nact[n] = runner._upload_unless_unchanged(
                "vps_num_active", torch.tensor([n], dtype=torch.int32))
        return got
