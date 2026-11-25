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
"""q38 D1-norm in-process windowed A/B (prereg q38-d1norm rev 2 §4; default off).

``VLLM_TPU_D1N_AB_PERIOD_MS=P`` (0 = off) cuts wall-clock time into windows
``k = time_ns // (P ms)``; class ``k mod 4`` is 0 = L-PREP on, 1 = off_a,
2 = L-GDNCOPY on, 3 = off_b. P = 20000 divides 86,400,000 with a quotient
divisible by 4, so the class is fixed by the time of day on every process.
An armed lever's flag turns it on in its own class only; with P = 0 the lever
is uniformly on. The windows are refused (every lever off, one ERROR) when
D1e's ``VLLM_TPU_HOST_AB_PERIOD_MS`` is also set: rev 2 allows one windowed
A/B per decode.

Each process logs one line per window it saw, when the next window starts:
``q38 D1N window=<k> class=<c> vectorized=<n> fallback=<n> steps=<n> syncs=<n>``
(vectorized: packed L-GDNCOPY crossing steps; fallback: crossing steps in an
on-window that took the per-tensor path; steps: seed-copy collector calls;
syncs: blocking waits for the previous step's sampled tokens at a crossing).
With P = 0 the same line is logged every 20 s with ``class=uniform``.
"""

from __future__ import annotations

import time

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

ARM_CLASS = {"prep": 0, "gdncopy": 2}
_UNIFORM_LOG_MS = 20000

# Read at import; tests set the module globals and call _resolve().
PERIOD_MS = int(envs.VLLM_TPU_D1N_AB_PERIOD_MS)
D1E_PERIOD_MS = int(envs.VLLM_TPU_HOST_AB_PERIOD_MS)
REFUSED = False
_state: dict = {}


def _resolve() -> None:
    global REFUSED
    REFUSED = PERIOD_MS > 0 and D1E_PERIOD_MS > 0
    _state.clear()
    _state.update(win=None, cls=None, counts=dict(vectorized=0, fallback=0, steps=0, syncs=0), warned=False)


_resolve()


def _now_ns() -> int:
    return time.time_ns()


def _window(now_ns: int) -> tuple[int, str]:
    if PERIOD_MS > 0:
        k = now_ns // (PERIOD_MS * 1_000_000)
        return k, str(k % 4)
    return now_ns // (_UNIFORM_LOG_MS * 1_000_000), "uniform"


def _roll() -> str:
    """Advance to the current window, logging the one that ended; return its class."""
    k, cls = _window(_now_ns())
    if _state["win"] is None:
        _state["win"], _state["cls"] = k, cls
    elif k != _state["win"]:
        c = _state["counts"]
        logger.info(
            "q38 D1N window=%d class=%s vectorized=%d fallback=%d steps=%d syncs=%d",
            _state["win"], _state["cls"], c["vectorized"], c["fallback"], c["steps"], c["syncs"],
        )
        _state["win"], _state["cls"] = k, cls
        _state["counts"] = dict(vectorized=0, fallback=0, steps=0, syncs=0)
    return cls


def arm_on(arm: str, flag: bool) -> bool:
    """Whether lever `arm` runs on this step: its flag is set, the A/B is not
    refused, and (when windowed) the current window is the arm's class."""
    if not flag:
        return False
    if REFUSED:
        if not _state["warned"]:
            _state["warned"] = True
            logger.error(
                "q38 D1N: VLLM_TPU_D1N_AB_PERIOD_MS=%d with D1e's VLLM_TPU_HOST_AB_PERIOD_MS=%d; "
                "rev 2 allows one windowed A/B per decode, so every D1-norm lever stays off.",
                PERIOD_MS, D1E_PERIOD_MS)
        return False
    cls = _roll()
    return cls == "uniform" or cls == str(ARM_CLASS[arm])


def note(kind: str, n: int = 1) -> None:
    """Count an event in the current window (no-op unless a D1-norm flag is in use)."""
    if not (envs.VLLM_TPU_GDN_SEED_COPY_PACKED or PERIOD_MS > 0):
        return
    _roll()
    _state["counts"][kind] += n


def counts() -> dict:
    return dict(_state["counts"])
