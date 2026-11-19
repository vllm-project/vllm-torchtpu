# SPDX-License-Identifier: Apache-2.0
"""q38 L5 SAB: a step-synchronous A/B of the whole-step megakernel inside one served run.

``VLLM_TPU_SERVED_AB=whole:P[:D]`` (default "" = off; P >= 2 * D, D >= 1, default D = 64) cuts the
decode engine's serving forwards into windows of P driver calls. Window ``k = n // P`` (n counts the
driver's calls after ``runner.warmup_done``) is an ON window when k is even and an OFF window when k
is odd. In an OFF window the first D calls still run whole-step (the confirmation delay below); from
call ``k * P + D`` to the end of the window every whole-step-eligible step runs the stock forward
instead, exactly as ``wholedry=1`` routes it (stock is the precompiled fallback at those buckets; no
new program, no compile, and the flag is outside the compile key).

Lockstep and its fail-safe. A step whose ranks disagree on whole-step vs stock would pair mismatched
collectives, so no rank may flip unless every rank flips at the same step. Each rank counts calls
itself (all TP ranks run the same scheduler outputs, so the counts agree when the system is healthy),
and proves the agreement before every OFF window without any blocking collective:
  * at the roll into window k (call k * P) each rank writes a fingerprint of that step's batch (the
    request ids in batch order and their computed-token counts; identical on every rank at the same
    step) to ``<dir>/w<k>_r<rank>``, ``dir = VLLM_TPU_SERVED_AB_DIR`` or ``/dev/shm/q38_sab_<ppid>``
    (the TP workers of one engine share a host and a parent process);
  * at call ``k * P + D`` of an OFF window each rank reads the entries of window k. It flips to OFF
    only if all ``world`` entries exist and are equal. Ranks are at most a step or two apart (their
    device collectives keep them in step), so with D in the tens every rank reads the same complete
    set and reaches the same decision at the same call.
  * Any missing or unequal entry (a desynchronised rank rolls at a different step, so its batch
    fingerprint differs) disables SAB on that rank for the rest of the run: whole-step stays ON, and
    the rank logs exactly one ERROR. All ranks see the same entries, so all of them disable.
Log lines (every rank): ``q38 SAB rank R roll window K arm on|off calls N fp F`` at each roll,
``q38 SAB rank R window K OFF from calls N`` at each confirmed flip, and the single ERROR
``q38 SAB rank R DISABLED ...``. The driver also notes ``ab_off_b<rows>`` in its step mix for every
step SAB sends to stock, and counts its drafts as stock drafts (per-path acceptance).
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Callable

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

DEFAULT_CONFIRM = 64


def parse(spec: str) -> tuple[int, int] | None:
    """'whole:P[:D]' -> (P, D); '' -> None; anything else raises ValueError."""
    spec = (spec or "").strip()
    if not spec:
        return None
    parts = spec.split(":")
    if parts[0] != "whole" or len(parts) not in (2, 3):
        raise ValueError(f"VLLM_TPU_SERVED_AB={spec!r}: expected 'whole:P[:D]'")
    period = int(parts[1])
    confirm = int(parts[2]) if len(parts) == 3 else DEFAULT_CONFIRM
    if confirm < 1 or period < 2 * confirm:
        raise ValueError(f"VLLM_TPU_SERVED_AB={spec!r}: needs D >= 1 and P >= 2 * D")
    return period, confirm


def batch_fingerprint(runner: Any) -> str:
    """The step's batch, identical on every TP rank at the same step: request ids in batch order and
    their computed-token counts."""
    try:
        ib = runner.input_batch
        ids = list(ib.req_ids)
        n = len(ids)
        toks = [int(x) for x in ib.num_computed_tokens_cpu[:n]]
        raw = repr((ids, toks)).encode()
    except Exception as exc:  # noqa: BLE001 - a missing batch is itself a fingerprint
        raw = repr(("no-batch", type(exc).__name__)).encode()
    return hashlib.sha1(raw).hexdigest()[:16]


class ServedAB:
    def __init__(self, period: int, confirm: int, rank: int, world: int, store: str,
                 fingerprint: Callable[[Any], str] = batch_fingerprint):
        self.period, self.confirm, self.rank, self.world = period, confirm, rank, world
        self.store = store
        self.fingerprint = fingerprint
        self.n = 0  # serving calls seen
        self.off = False  # this step runs stock instead of whole-step
        self.disabled = False
        os.makedirs(store, exist_ok=True)

    def _path(self, k: int, r: int) -> str:
        return os.path.join(self.store, f"w{k}_r{r}")

    def _disable(self, why: str) -> None:
        self.disabled, self.off = True, False
        logger.error("q38 SAB rank %d DISABLED (whole-step stays on for the rest of the run): %s", self.rank, why)

    def tick(self, runner: Any) -> None:
        """Call once per driver call, before the step's whole-step decision."""
        if self.disabled or not getattr(runner, "warmup_done", False):
            self.off = False
            return
        n, P, D = self.n, self.period, self.confirm
        self.n += 1
        k, i = divmod(n, P)
        if i == 0:
            self.off = False
            fp = self.fingerprint(runner)
            try:
                tmp = self._path(k, self.rank) + ".tmp"
                with open(tmp, "w") as f:
                    f.write(fp)
                os.replace(tmp, self._path(k, self.rank))
            except OSError as exc:
                self._disable(f"cannot write {self._path(k, self.rank)}: {exc}")
                return
            logger.info("q38 SAB rank %d roll window %d arm %s calls %d fp %s", self.rank, k,
                        "off" if k % 2 else "on", n, fp)
        elif k % 2 and i == D:
            seen = {}
            for r in range(self.world):
                try:
                    with open(self._path(k, r)) as f:
                        seen[r] = f.read()
                except OSError:
                    seen[r] = None
            if any(v is None for v in seen.values()) or len(set(seen.values())) != 1:
                self._disable(f"window {k}: rank fingerprints {seen} (missing or unequal: ranks out of step)")
                return
            self.off = True
            logger.info("q38 SAB rank %d window %d OFF from calls %d", self.rank, k, n)
        elif i == 0 or (k % 2 == 0):
            self.off = False

    def whole_on(self) -> bool:
        return not self.off


def from_env(spec: str, rank: int, world: int, store: str | None = None) -> ServedAB | None:
    """The driver's constructor: None when unset; None plus one ERROR when the spec is invalid."""
    try:
        pc = parse(spec)
    except ValueError as exc:
        logger.error("q38 SAB off: %s", exc)
        return None
    if pc is None:
        return None
    store = store or os.environ.get("VLLM_TPU_SERVED_AB_DIR") or f"/dev/shm/q38_sab_{os.getppid()}"
    ab = ServedAB(pc[0], pc[1], rank, world, store)
    logger.info("q38 SAB armed on rank %d of %d: period %d calls, confirm %d calls, store %s", rank, world, pc[0], pc[1], store)
    return ab
