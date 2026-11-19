# SPDX-License-Identifier: Apache-2.0
"""q38 L6 NODEFER census (host_levers_h2.md section 4.1 row 3; log-only).

Question: is the 60-100 us unexplained inside vfy_rejection a block on deferred
arguments of the VFY call? torch_tpu defers eager ops (kDeferAndFuse defers the
eager rand draws), and a compiled call with a not-yet-materializing argument
first blocks until it has started (Materialize(kBlockUntilStarted)).

With the l6_ab lever ``nodefer`` (VLLM_TPU_L6_NODEFER_CENSUS; 0 / 1 / ab:<c>;
compile-ignored) on, the 1st, (E+1)-th, (2E+1)-th ... VFY call (E =
VLLM_TPU_L6_NODEFER_EVERY, default 64) counts its tensor arguments for which
``torch_tpu._internal.sync.is_materializing(t)`` is False, immediately before
the compiled call, and counts them on the L6AB window line:
``nodefer.calls`` (VFY calls seen), ``nodefer.sampled``, ``nodefer.deferred``
(summed over sampled calls), ``nodefer.a<i>`` (sampled calls in which positional
argument i was deferred), ``nodefer.nosync`` / ``nodefer.err`` when the probe is
unavailable or raises. The census never changes an argument or the call.

The census and the sub-marks ~vr_draw (from the vfy_rand mark to the call:
argument build, the fallback rows' uniform draw, the num_active upload, the
census) and ~vr_call (exactly the compiled call) live in ``VfyCall``, which wraps
the two VPS verify compiled calls at their definitions in runner/tpu_runner.py,
so the runner's method bodies are unchanged. ~vr_post (the fallback stats and
checks after the call, up to the vfy_rejection mark) is the runner's sub-mark
right before that mark. They split vfy_rejection (HOSTSUBS / HOSTSLOW only).
Warm-up calls go through the wrapper too, so ``nodefer.calls`` counts them in
the warm-up windows.
"""

from __future__ import annotations

import importlib
import os

import torch

from vllm_torchtpu.core import l6_ab
from vllm_torchtpu.runner import host_phases

EVERY = 64
_st = {"n": 0}


def _resolve() -> None:
    global EVERY
    raw = os.environ.get("VLLM_TPU_L6_NODEFER_EVERY", "").strip()
    EVERY = int(raw) if raw else 64
    if EVERY < 1:
        raise ValueError("VLLM_TPU_L6_NODEFER_EVERY must be >= 1")
    _st["n"] = 0


_resolve()


def _sync():
    return importlib.import_module("torch_tpu._internal.sync")


def census(call_args: tuple) -> None:
    """Immediately before a VFY compiled call, with its positional arguments."""
    if not l6_ab.on("nodefer"):
        return
    _st["n"] += 1
    l6_ab.note("nodefer", "calls")
    if (_st["n"] - 1) % EVERY:
        return
    try:
        sync = _sync()
    except Exception:  # noqa: BLE001 - no torch_tpu probe: say so, never fail the call
        l6_ab.note("nodefer", "nosync")
        return
    l6_ab.note("nodefer", "sampled")
    k = 0
    for i, t in enumerate(call_args):
        if not isinstance(t, torch.Tensor):
            continue
        try:
            started = sync.is_materializing(t)
        except Exception:  # noqa: BLE001
            l6_ab.note("nodefer", "err")
            continue
        if not started:
            k += 1
            l6_ab.note("nodefer", f"a{i}")
    l6_ab.note("nodefer", "deferred", k)


class VfyCall:
    """A VFY compiled call with the census and ~vr_draw / ~vr_call around exactly
    the call; attribute access passes through to the wrapped callable."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        census(args)
        host_phases.sub("vr_draw")
        out = self.fn(*args, **kwargs)
        host_phases.sub("vr_call")
        return out

    def __getattr__(self, name):
        if name == "fn":
            raise AttributeError(name)
        return getattr(self.fn, name)


def wrap(fn) -> VfyCall:
    return VfyCall(fn)
