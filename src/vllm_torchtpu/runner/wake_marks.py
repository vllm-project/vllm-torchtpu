# SPDX-License-Identifier: Apache-2.0
"""Diagnostics for the "thread wake" hypothesis on the served decode step.

Served, three host calls each pay ~100 us more than in a back-to-back
microbenchmark, all after the PJRT enqueue: the folded step-input program
(``_prep_fold_compiled``), the megakernel whole-step runner and the first H2D
of the step. The hypothesis: a runtime or transfer thread sleeps between calls
and every call pays a wake-up / hand-back cost.

Marks (VLLM_TPU_HOST_PHASES=N, recorded as ``x_`` notes, not summed):

=====================  ===================================================
x_pu_fc_front          ``_prep_fold_compiled`` entry up to the C++ runner:
                       ``_static_shape_args``, Dynamo cache lookup and
                       guards, the AOT prologue
x_pu_fc_flush          (FC_FLUSH only) ``synchronize(None, wait=False)``
                       right before the runner
x_pu_fc_runner         ``TorchTpuCompiledExecutable._fast_runner(args)``:
                       argument conversion, PJRT Execute, output wrap
x_pu_fc_post           runner return up to the fold call's return (AOT and
                       Dynamo return path)
x_pu_fc_norun          the fold call when no executable ran (should not
                       appear)
x_vf_front/runner/     the same split for the verify program
post                   (``_verify_*_compiled``; with VERIFY_RO_INGRAPH the
                       ``_verify_one_draw_ro*`` programs)
x_pu_h2d1_pre          last ``host_phases.sub`` (or the fence) up to the
                       first CPU->TPU ``Tensor.to`` after the input-staging
                       fence: in the prep path, the input_ids slice
x_pu_h2d1_to           that ``Tensor.to`` call alone
=====================  ===================================================

The megakernel whole-step runner (MK_WHOLE_DIRECT_EXEC=1) does not go through
``TorchTpuCompiledExecutable.__call__``; ``x_mkf_runner`` already brackets its
C++ runner call (runner/mk_call_marks.py).

``x_pu_fc_front + runner + post`` is the fold call minus about a microsecond of
wrapper frames (it sits inside ``x_pu_fold_call`` when the prep-split marks are
merged).

Prewarm arm (``VLLM_TPU_DIAG_PREWARM``, a bit mask, works with or without
HOST_PHASES):

- ``1``: right before the fold call, a non-blocking 32-byte H2D of a
  persistent CPU tensor (``x_pw_h2d``). Its device buffer is never read.
- ``2``: then a call of a 1-op program (``int32[8] + 1``) on a persistent
  device tensor (``x_pw_ex_total``, split as ``x_pw_ex_front/runner/post``),
  compiled once after ``capture_model``, called under
  ``set_stance("fail_on_recompile")`` (a recompile disables it, no serve-time
  compile). Adds one tiny program per folded step.
- ``4``: right before the first H2D of the step, the same 32-byte H2D
  (``x_pw_h2d_first``).

Outputs are unchanged: the prewarm buffers feed nothing. With the wake theory,
prewarm ON moves ~100 us from ``x_pu_fc_runner`` (and ``x_pu_h2d1_to``) into
``x_pw_*``; with no wake cost both stay where they were and ``x_pw_*`` is a few
microseconds. Prewarm times sit inside the enclosing prep-split spans
(``x_pu_fold_call``, ``x_pu_h2d_input_ids``); subtract them there.

``VLLM_TPU_HOST_PHASES_FC_FLUSH=1`` adds the flush lap (a diagnostic arm: it
flushes all pending eager ops, which can change how later ones are cut).

All switches are read straight from ``os.environ`` (not the envs registry), so
they stay out of the compile-cache key. With HOST_PHASES off and PREWARM 0,
``install`` returns at once and nothing is patched.
"""

import functools
import os
import sys
import time

import torch

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner import host_phases

logger = init_logger(__name__)

ON = host_phases.enabled()
PREWARM = int(os.environ.get("VLLM_TPU_DIAG_PREWARM", "0") or 0)
FC_FLUSH = ON and os.environ.get("VLLM_TPU_HOST_PHASES_FC_FLUSH", "0") not in ("", "0")

_perf = time.perf_counter_ns

FOLD_SITE = "x_pu_fc"
VERIFY_SITE = "x_vf"
VERIFY_GLOBALS = (
    "_verify_fused_compiled",
    "_verify_fused_one_draw_compiled",
    "_verify_one_draw_ro_compiled",
    "_verify_one_draw_ro_lb_compiled",
)


class _S:
    # Site whose next executable call is split; None outside a site.
    site: str | None = None
    t0 = 0
    t_run = 0
    ran = False
    # First-H2D split: armed after the fence until the first CPU->TPU .to.
    armed = False
    t_fence = 0
    runner = None
    # Prewarm state.
    pw_src: torch.Tensor | None = None
    pw_dev: torch.Tensor | None = None
    pw_keep: object = None
    noop = None
    noop_ok = False


_orig_exec_call = None
_orig_to = torch.Tensor.to


def _note(name: str, ns: int) -> None:
    host_phases.note(name, ns / 1e3)


def _patched_exec_call(self, *args, output_shapes=None):
    site = _S.site
    if site is None:
        return _orig_exec_call(self, *args, output_shapes=output_shapes)
    _S.site = None  # only the site's first executable call is split
    t = _perf()
    _note(site + "_front", t - _S.t0)
    if FC_FLUSH:
        from torch_tpu._internal.sync import sync

        sync.synchronize(None, wait=False)
        t2 = _perf()
        _note(site + "_flush", t2 - t)
        t = t2
    fr = getattr(self, "_fast_runner", None)
    if fr is not None and not output_shapes:
        out = fr(args)
    else:
        out = _orig_exec_call(self, *args, output_shapes=output_shapes)
    t3 = _perf()
    _note(site + "_runner", t3 - t)
    _S.t_run = t3
    _S.ran = True
    return out


def _split_call(prefix: str, fn, *args, **kwargs):
    """Call ``fn`` with its first compiled-executable call split into
    ``<prefix>_front / _runner / _post``."""
    if not ON or _S.site is not None:
        return fn(*args, **kwargs)
    _S.ran = False
    _S.site = prefix
    _S.t0 = _perf()
    try:
        return fn(*args, **kwargs)
    finally:
        t = _perf()
        if _S.ran:
            _note(prefix + "_post", t - _S.t_run)
        else:
            _note(prefix + "_norun", t - _S.t0)
        _S.site = None
        _S.ran = False


def _site_wrapper(fn, prefix: str, prewarm: bool):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        _disarm()
        if prewarm and PREWARM & 3:
            _prewarm_fold()
        return _split_call(prefix, fn, *args, **kwargs)

    wrapper._wake_marks_wrapped = fn
    return wrapper


# ---------------------------------------------------------------- prewarm


def _pw_h2d(device) -> None:
    _S.pw_keep = _orig_to(_S.pw_src, device, non_blocking=True)


def _prewarm_fold() -> None:
    runner = _S.runner
    if runner is None or _S.pw_src is None:
        return
    if PREWARM & 1:
        t = _perf()
        _pw_h2d(runner.device)
        _note("x_pw_h2d", _perf() - t)
    if PREWARM & 2 and _S.noop_ok:
        t = _perf()
        try:
            with torch.no_grad(), torch.compiler.set_stance("fail_on_recompile"):
                _S.pw_keep = _split_call("x_pw_ex", _S.noop, _S.pw_dev)
        except Exception as e:  # a recompile: never compile at serve time
            _S.noop_ok = False
            logger.warning("VLLM_TPU_DIAG_PREWARM: exec prewarm disabled: %s", e)
        _note("x_pw_ex_total", _perf() - t)


def _noop_impl(x: torch.Tensor) -> torch.Tensor:
    return x + 1


def warm(runner) -> None:
    """Compile the exec-prewarm program once (after capture_model)."""
    if not PREWARM & 2 or _S.noop_ok:
        return
    try:
        _S.pw_dev = _orig_to(_S.pw_src, runner.device)
        _S.noop = torch.compile(
            _noop_impl, backend="tpu", fullgraph=True, dynamic=False
        )
        with torch.no_grad():
            for _ in range(2):
                out = _S.noop(_S.pw_dev)
        out.cpu()
        _S.noop_ok = True
        logger.info("VLLM_TPU_DIAG_PREWARM: exec prewarm program compiled")
    except Exception as e:
        logger.warning("VLLM_TPU_DIAG_PREWARM: exec prewarm unavailable: %s", e)


# ------------------------------------------------------- first H2D split


def _is_dev_target(args, kwargs) -> bool:
    d = kwargs.get("device")
    if d is None and args and isinstance(args[0], (torch.device, str)):
        d = args[0]
    if d is None:
        return False
    try:
        return torch.device(d).type != "cpu"
    except (RuntimeError, TypeError):
        return False


def _timed_to(self, *args, **kwargs):
    if (
        not _S.armed
        or type(self) is not torch.Tensor
        or self.device.type != "cpu"
        or not _is_dev_target(args, kwargs)
        or torch.compiler.is_compiling()
    ):
        return _orig_to(self, *args, **kwargs)
    _disarm()
    t = _perf()
    if ON:
        # The prep-split marks' host_phases has _sub_last (sub()); else the fence.
        sub_last = getattr(host_phases, "_sub_last", None) or 0
        start = max(sub_last, host_phases._last or 0, _S.t_fence)
        _note("x_pu_h2d1_pre", t - start)
        t = _perf()
    if PREWARM & 4 and _S.pw_src is not None:
        _pw_h2d(_S.runner.device)
        t2 = _perf()
        _note("x_pw_h2d_first", t2 - t)
        t = t2
    out = _orig_to(self, *args, **kwargs)
    _note("x_pu_h2d1_to", _perf() - t)
    return out


def _disarm() -> None:
    if _S.armed:
        _S.armed = False
        torch.Tensor.to = _orig_to


def _fence_wrapper(fn, runner):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        _disarm()
        out = fn(*args, **kwargs)
        # Serving prep path only (not _dummy_run, not warmup).
        if (
            getattr(runner, "warmup_done", False)
            and sys._getframe(1).f_code.co_name == "_prepare_inputs"
        ):
            _S.t_fence = _perf()
            _S.armed = True
            torch.Tensor.to = _timed_to
        return out

    return wrapper


def _capture_wrapper(fn, runner):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        warm(runner)
        return out

    return wrapper


# ---------------------------------------------------------------- install


def install(runner, module=None) -> None:
    """Patch the marks / prewarm in; a no-op with HOST_PHASES off and PREWARM 0.

    ``module`` holds the compiled globals (default: runner/tpu_runner.py)."""
    global _orig_exec_call
    if not (ON or PREWARM):
        return
    if module is None:
        from vllm_torchtpu.runner import tpu_runner as module
    _S.runner = runner
    _S.pw_src = torch.zeros(8, dtype=torch.int32)
    if ON and _orig_exec_call is None:
        from torch_tpu._internal.compile.torch_tpu_compiled_executable import (
            TorchTpuCompiledExecutable,
        )

        _orig_exec_call = TorchTpuCompiledExecutable.__call__
        TorchTpuCompiledExecutable.__call__ = _patched_exec_call
    fold = getattr(module, "_prep_fold_compiled", None)
    if fold is not None and not hasattr(fold, "_wake_marks_wrapped"):
        module._prep_fold_compiled = _site_wrapper(fold, FOLD_SITE, prewarm=True)
    if ON:
        for name in VERIFY_GLOBALS:
            fn = getattr(module, name, None)
            if fn is not None and not hasattr(fn, "_wake_marks_wrapped"):
                setattr(module, name, _site_wrapper(fn, VERIFY_SITE, prewarm=False))
    if ON or PREWARM & 4:
        runner._wait_input_staging_fence = _fence_wrapper(
            runner._wait_input_staging_fence, runner
        )
    if PREWARM & 2:
        runner.capture_model = _capture_wrapper(runner.capture_model, runner)
    logger.info(
        "WAKE_MARKS: host_phases=%s prewarm=%d fc_flush=%s", ON, PREWARM, FC_FLUSH
    )
