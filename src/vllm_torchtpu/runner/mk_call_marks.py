# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_HOST_PHASES: split of the ``x_mk_try_forward`` entry by sub-step.

The Qwen3.8 megakernel driver's ``try_forward`` is timed as one ``x_`` entry.
With HOST_PHASES on, laps inside it (and inside ``DirectExec`` while it runs)
are recorded as further ``x_mkf_*`` entries, which sum to the call:

==================  ======================================================
x_mkf_inspect       entry, ``_inspect_step_detail``
x_mkf_gate          whole-step self-check, bucket and eligibility checks
x_mkf_pre           step-mix notes, check gates, ``_whole_direct()``
x_mkf_ctx_enter     building and entering the model-tag / config contexts
x_mkf_de_check      ``DirectExec.__call__`` checks before the bytecode
x_mkf_gather        Dynamo bytecode up to the backend call (operand gather)
x_mkf_key           interceptor bucket key, record lookup, int / dup checks,
                    picking the runner operands out of the arguments
x_mkf_defscan       (MK_FLUSH only) counting the operands still deferred
x_mkf_flush         (MK_FLUSH only) ``synchronize(None, wait=False)``: the
                    pending fused-eager ops, flushed ahead of the runner
x_mkf_runner        the C++ runner call (PJRT execute; without MK_FLUSH it
                    also flushes whatever eager work its operands still need)
x_mkf_epilogue      the input-mutation ``copy_`` replay
x_mkf_call          rest of the program call: outputs and returns (the whole
                    call when the normal, non-direct path ran)
x_mkf_ctx_exit      leaving the contexts
x_mkf_tail          after the call (stats hooks, tuple), or everything after
                    the last lap on a stock step
==================  ======================================================

``x_mkf_call_total`` (the program call alone) and ``x_mkf_ndef`` (a count,
not microseconds) are not part of that sum.

``VLLM_TPU_HOST_PHASES_MK_FLUSH=1`` adds the flush probe. It flushes all
pending eager ops, not only the operands' producers, so it can change how
later eager ops are cut into programs (and compile those): a diagnostic arm
only. ``VLLM_TPU_HOST_PHASES_MK_STEP=K`` (default 32, 0 = off) logs one
``HOSTMKF`` line per rank on every K-th driver step that ran the whole-step
program, with the entry wall time and that step's laps, for a per-step
cross-rank join. Both are read straight from the environment (not the envs
registry), so they stay out of the compile-cache key.

With HOST_PHASES off ``ON`` is False and nothing here runs: the call sites
test a module constant.
"""

import os
import time

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner import host_phases

logger = init_logger(__name__)

ON = host_phases.enabled()
FLUSH = ON and os.environ.get("VLLM_TPU_HOST_PHASES_MK_FLUSH", "0") not in ("", "0")
STEP_EVERY = int(os.environ.get("VLLM_TPU_HOST_PHASES_MK_STEP", "32") or 0) if ON else 0

# True between start() and finish(): DirectExec laps only inside try_forward.
active = False
# Set by the driver when the whole-step program ran in this call.
whole = False
_t = 0
_t_call = 0
_wall = 0.0
_laps: list[tuple[str, int]] = []


def start() -> None:
    global active, whole, _t, _wall
    active, whole = True, False
    _laps.clear()
    _wall = time.time()
    _t = time.perf_counter_ns()


def lap(name: str) -> None:
    """Charge the host time since the previous lap to ``name``."""
    global _t
    now = time.perf_counter_ns()
    d = now - _t
    _t = now
    host_phases.note(name, d / 1e3)
    _laps.append((name, d))


def call_begin() -> None:
    global _t_call
    _t_call = _t


def call_end() -> None:
    d = _t - _t_call
    host_phases.note("x_mkf_call_total", d / 1e3)


def flush(operands) -> None:
    """Count the operands not yet materializing, then submit all pending eager
    ops without waiting; each timed as its own lap."""
    from torch_tpu._internal.sync import sync

    pending = sum(
        1
        for t in operands
        if getattr(t, "device", None) is not None
        and t.device.type != "cpu"
        and not sync.is_materializing(t)
    )
    lap("x_mkf_defscan")
    host_phases.note("x_mkf_ndef", float(pending))
    sync.synchronize(None, wait=False)
    lap("x_mkf_flush")


def finish(seq: int, rows: int) -> None:
    global active
    lap("x_mkf_tail")
    active = False
    if whole and STEP_EVERY and seq % STEP_EVERY == 0:
        logger.info(
            "HOSTMKF rank=%d seq=%d rows=%d wall=%.6f total_us=%.1f %s",
            host_phases._rank,
            seq,
            rows,
            _wall,
            sum(d for _, d in _laps) / 1e3,
            " ".join(f"{n}={d / 1e3:.1f}" for n, d in _laps),
        )
