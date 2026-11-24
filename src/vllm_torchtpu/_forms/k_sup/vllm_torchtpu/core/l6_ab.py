# SPDX-License-Identifier: Apache-2.0
"""q38 BLSTACK-SEED: a minimal stand-in for the q38 tree's ``core/l6_ab.py`` on Benliang Wang's tree.

Only lever ``seedspec`` (``runner/l6_seedspec.py``, HCDEC-HOSTADM SEED-SPEC) uses it here. Same names as the q38 module
(``LEVERS``, ``configured``, ``on``, ``begin_step``, ``note``) so ``l6_seedspec.py`` reads unchanged, but no in-run A/B:

* the lever's env value is read once (``_resolve``, on first use) and is the lever's state for the process life:
  unset / "0" = off, "1" = on (the plain path);
* "ab:<classes>" (the q38 tree's in-run A/B windows) is REFUSED with a ValueError at the first read, never silently
  treated as on or off: there is no window machinery on this tree;
* ``note`` keeps host-only integer counters (no device work, no compiled program, not part of the compile key) and
  logs a 'q38 BLSTACK seedspec counts' summary every ``_SUMMARY_EVERY`` speculative seeds;
* the first ``on()`` that reads True logs 'q38 BLSTACK seedspec ENGAGED' once per process.
"""

from __future__ import annotations

import os

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

LEVERS: dict[str, str] = {
    # HCDEC-HOSTADM SEED-SPEC (runner/l6_seedspec.py): the GDN seed at a block edge staged without the copy wait, the
    # column resolved at the next step. Read per worker step in _collect_mamba_state_seed_copies.
    "seedspec": "VLLM_TPU_L6_SEED_SPEC",
}

_SUMMARY_EVERY = 1000  # speculative seeds ("<lever>.spec" notes) between count summaries
_state: dict = {"resolved": False, "on": {}, "counts": {}, "engaged": set()}


def _parse(lever: str, raw: str) -> bool:
    v = raw.strip()
    if v in ("", "0"):
        return False
    if v == "1":
        return True
    if v.startswith("ab:"):
        raise ValueError(
            f"{LEVERS[lever]}={raw!r}: in-run A/B windows (ab:) are not supported on this tree "
            "(q38 BLSTACK-SEED l6_ab shim); use 1 (on) or 0 / unset (off)"
        )
    raise ValueError(f"{LEVERS[lever]}={raw!r}: expected 1 (on) or 0 / unset (off)")


def _resolve() -> None:
    """Read every lever's env value (again): the state for the process life."""
    _state["on"] = {lever: _parse(lever, os.environ.get(env, "")) for lever, env in LEVERS.items()}
    _state["counts"] = {}
    _state["engaged"] = set()
    _state["resolved"] = True


def configured(lever: str) -> bool:
    """The lever's env value is set to on (the only configured state here)."""
    if not _state["resolved"]:
        _resolve()
    return bool(_state["on"].get(lever, False))


def on(lever: str) -> bool:
    """The lever's state (= configured: no A/B windows on this tree). Logs the engagement line once."""
    st = configured(lever)
    if st and lever not in _state["engaged"]:
        _state["engaged"].add(lever)
        logger.info("q38 BLSTACK %s ENGAGED (%s=1, plain; ab: refused)", lever, LEVERS[lever])
    return st


def begin_step(role: str = "worker") -> None:
    """No windows to advance; kept for the q38 module's call shape."""
    if not _state["resolved"]:
        _resolve()


def note(lever: str, key: str, n: int = 1) -> None:
    """Host-only counter ``<lever>.<key> += n``; a count summary every _SUMMARY_EVERY speculative seeds."""
    counts = _state["counts"]
    name = f"{lever}.{key}"
    counts[name] = counts.get(name, 0) + int(n)
    if key == "spec" and counts[name] % _SUMMARY_EVERY == 0:
        pre = lever + "."
        logger.info(
            "q38 BLSTACK %s counts %s",
            lever,
            " ".join(f"{k[len(pre):]}={v}" for k, v in sorted(counts.items()) if k.startswith(pre)),
        )


def counts() -> dict[str, int]:
    return dict(_state["counts"])
