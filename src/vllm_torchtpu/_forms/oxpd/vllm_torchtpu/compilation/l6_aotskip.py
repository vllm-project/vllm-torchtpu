# SPDX-License-Identifier: Apache-2.0
"""q38 L6-H2 CF-2 AOTSKIP-AB: the AOT guard-check skip as a per-window l6_ab lever.

X-H14-AOTSKIP (compilation/aot_guard_skip.py, q38 L4) calls
``AOTCompiledFunction.disable_guard_check()`` once, right after vLLM's fresh AOT
compile, under ``VLLM_TPU_AOT_SKIP_GUARD_CHECK``. That flag is part of the hashed
compile key, which kept it out of STACK-D v4. torch's ``AOTCompiledFunction``
reads ``_guard_check_enabled`` on every call (``__call__``: ``if
self._guard_check_enabled and not self.guard_check(...)``), so the same skip can
be toggled per step with no recompile and no compile-key change:

* ``install()`` (patch registry, only when ``VLLM_TPU_L6_AOT_SKIP`` is set) wraps
  vLLM's ``TorchCompileWithNoGuardsWrapper.aot_compile`` and
  ``decorators._try_load_aot_compiled_fn`` so every artifact they return (the
  whole-step forward, the draft's forward0 and loop, the stock backbone) is kept
  with the guard state it was returned with -- but only where vLLM itself does
  not evaluate guards, the scope of X-H14-AOTSKIP: a fresh artifact whose
  ``compilation_config.dynamic_shapes_config.evaluate_guards`` is False (missing
  counts as True), and a cache-loaded artifact that vLLM left with its guard
  check off (it does that exactly when evaluate_guards is False). An artifact
  whose guards vLLM evaluates is never kept, so the lever never skips a guard
  vLLM asked for (counted as ``aotskip.not_kept``);
* ``on_step()`` (worker execute_model, after ``l6_ab.begin_step``) writes
  ``_guard_check_enabled = original and not l6_ab.on('aotskip')`` into every kept
  artifact when that value changes (window boundaries), so a step's calls all run
  in one state. ``original`` matters: an artifact vLLM loaded from its cache with
  guards already off (``evaluate_guards`` False) is never switched on.

CHECK (``VLLM_TPU_L6_AOT_SKIP_CHECK=N`` > 0): every N-th lever-on step evaluates
the guards as the stock path does (bounded by ``l6_ab.check_allowed()``), so a
guard that would fail raises exactly as it would with the lever off; counted as
``aotskip.check``.

Engagement (L6AB window line): ``aotskip.handles`` (artifacts kept, counted once
each), ``aotskip.not_kept`` (artifacts whose guards vLLM evaluates, left alone),
``aotskip.on_steps`` / ``aotskip.off_steps`` (steps per state), ``aotskip.flips``
(artifact writes).
"""

from __future__ import annotations

import functools
import os

from vllm_torchtpu.core import l6_ab
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

LEVER = "aotskip"
ENV = "VLLM_TPU_L6_AOT_SKIP"
CHECK_ENV = "VLLM_TPU_L6_AOT_SKIP_CHECK"

# (artifact, the _guard_check_enabled value it was returned with)
_handles: list[tuple[object, bool]] = []
_state: dict = {"applied": None, "check_n": 0, "new": 0, "not_kept": 0}


def flag_set() -> bool:
    """Whether the lever's env var asks for anything (the patch is installed only then)."""
    return (os.environ.get(ENV, "") or "").strip().lower() not in ("", "0", "false", "off")


def _check_every() -> int:
    raw = (os.environ.get(CHECK_ENV, "") or "").strip()
    return int(raw) if raw else 0


def register(fn) -> None:
    """Keep one artifact (anything with a boolean ``_guard_check_enabled``) and put it in
    the current state at once, so an artifact compiled mid-run never runs one step stale."""
    if fn is None or not isinstance(getattr(fn, "_guard_check_enabled", None), bool):
        return
    orig = bool(fn._guard_check_enabled)
    _handles.append((fn, orig))
    _state["new"] += 1
    applied = _state["applied"]
    if applied is not None:
        fn._guard_check_enabled = orig and not applied


def _evaluates_guards(owner) -> bool:
    """vLLM's evaluate_guards for a fresh artifact's owner, read as X-H14-AOTSKIP reads it
    (compilation/aot_guard_skip.py); a missing config counts as evaluating."""
    config = getattr(owner, "compilation_config", None)
    return bool(getattr(getattr(config, "dynamic_shapes_config", None), "evaluate_guards", True))


def _not_kept() -> None:
    _state["not_kept"] += 1


def _want() -> bool:
    """The skip state for this step: lever on, and not a CHECK step."""
    if not l6_ab.on(LEVER):
        return False
    n = _check_every()
    if n > 0 and l6_ab.check_allowed():
        _state["check_n"] += 1
        if _state["check_n"] >= n:
            _state["check_n"] = 0
            l6_ab.note(LEVER, "check")
            return False
    return True


def apply(skip: bool) -> int:
    """Write ``orig and not skip`` into every kept artifact; returns the number written."""
    n = 0
    for fn, orig in _handles:
        fn._guard_check_enabled = orig and not skip
        n += 1
    _state["applied"] = skip
    return n


def on_step() -> None:
    """Per worker step, after ``l6_ab.begin_step()``. Writes only when the state changes."""
    if _state["new"]:
        l6_ab.note(LEVER, "handles", _state["new"])
        _state["new"] = 0
    if _state["not_kept"]:
        l6_ab.note(LEVER, "not_kept", _state["not_kept"])
        _state["not_kept"] = 0
    skip = _want()
    l6_ab.note(LEVER, "on_steps" if skip else "off_steps")
    if skip != _state["applied"]:
        l6_ab.note(LEVER, "flips", apply(skip))


def reset() -> None:
    """Tests: forget every kept artifact and the applied state."""
    _handles.clear()
    _state.update(applied=None, check_n=0, new=0, not_kept=0)


def install() -> None:
    """Wrap vLLM's AOT compile and cache-load entry points so their artifacts are kept.
    Idempotent; a no-op unless ``VLLM_TPU_L6_AOT_SKIP`` is set."""
    if not flag_set():
        return
    if (os.environ.get("VLLM_TPU_AOT_SKIP_GUARD_CHECK", "") or "").strip().lower() in ("1", "true", "yes", "on"):
        # X-H14-AOTSKIP already turns the check off at compile; a lever-off window here would turn it back on.
        logger.error("q38 L6 aotskip: not installed next to VLLM_TPU_AOT_SKIP_GUARD_CHECK (one owner per artifact)")
        return
    from vllm.compilation import decorators
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper as W

    if not getattr(W.aot_compile, "_l6_aotskip", False):
        original = W.aot_compile

        @functools.wraps(original)
        def aot_compile(self, *args, **kwargs):
            compiled = original(self, *args, **kwargs)
            if _evaluates_guards(self):
                _not_kept()  # vLLM evaluates this artifact's guards: never skip them
            else:
                register(compiled)
            return compiled

        aot_compile._l6_aotskip = True
        W.aot_compile = aot_compile

    load = getattr(decorators, "_try_load_aot_compiled_fn", None)
    if load is not None and not getattr(load, "_l6_aotskip", False):

        @functools.wraps(load)
        def _try_load_aot_compiled_fn(model, *args, **kwargs):
            loaded = load(model, *args, **kwargs)
            if loaded is not None and getattr(loaded, "_guard_check_enabled", None) is True:
                _not_kept()  # vLLM left the check on (evaluate_guards): never skip it
            else:
                register(loaded)
            return loaded

        _try_load_aot_compiled_fn._l6_aotskip = True
        decorators._try_load_aot_compiled_fn = _try_load_aot_compiled_fn
    logger.info("q38 L6 aotskip: AOT artifacts kept for the per-window guard-check skip (pid %d)", os.getpid())
